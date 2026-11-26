"""Weight-average ("model soup") several fine-tunes of the SAME starting checkpoint into one net.

    python -m nn.soup --runs soup_8x128_s0r1_a,soup_8x128_s0r1_b,soup_8x128_s0r1_c --out soup_8x128_s0r1

Each member's runs/<run>/checkpoints/best.pt is loaded, every floating-point tensor of the
state dict (weights, biases, BatchNorm running mean/var) is averaged, and the result is written
as a normal run: runs/<out>/checkpoints/best.pt, best.npz, model.npz, config.json.

Only meaningful when the members share one initialisation (e.g. all fine-tuned from
score_8x128_s0_r1 with different seeds): nets trained from different seeds have permuted
channels and their average is garbage.  BatchNorm statistics are averaged, not recomputed.
The play settings (play_head, play_mirror, ...) are copied from the first member's best.npz.
"""

import argparse
import json
import os

import numpy as np

from .common import RUNS, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma list of member run names")
    ap.add_argument("--out", required=True, help="name of the new run")
    a = ap.parse_args()

    import torch
    from .export import export_model
    from .model import load_checkpoint, save_checkpoint

    members = [r for r in a.runs.split(",") if r]
    if len(members) < 2:
        raise SystemExit("need at least two runs")
    out_dir = os.path.join(RUNS, a.out)
    if os.path.exists(out_dir):
        raise SystemExit("runs/%s already exists - choose another --out name" % a.out)

    models, states = [], []
    for r in members:
        p = os.path.join(RUNS, r, "checkpoints", "best.pt")
        m, _ = load_checkpoint(p, "cpu")
        models.append(m)
        states.append({k: v.detach().clone() for k, v in m.state_dict().items()})
        log("loaded %s" % p)
    keys = list(states[0])
    for s, r in zip(states[1:], members[1:]):
        if list(s) != keys or any(s[k].shape != states[0][k].shape for k in keys):
            raise SystemExit("%s does not have the same architecture as %s" % (r, members[0]))

    avg = {}
    for k in keys:
        t = states[0][k]
        if torch.is_floating_point(t):
            avg[k] = torch.stack([s[k].double() for s in states]).mean(0).to(t.dtype)
        else:                                   # e.g. BatchNorm num_batches_tracked
            avg[k] = t.clone()
    soup = models[0]
    soup.load_state_dict(avg)
    soup.eval()

    # how far apart were the members? (relative L2 distance of each to the average)
    dist = {}
    for r, s in zip(members, states):
        num = sum(float(((s[k].double() - avg[k].double()) ** 2).sum()) for k in keys if torch.is_floating_point(s[k]))
        den = sum(float((avg[k].double() ** 2).sum()) for k in keys if torch.is_floating_point(s[k]))
        dist[r] = round((num / den) ** 0.5, 5)
    log("relative L2 distance of each member to the soup: %s" % dist)

    meta = {}
    first_npz = os.path.join(RUNS, members[0], "best.npz")
    if os.path.isfile(first_npz):
        with np.load(first_npz, allow_pickle=False) as z:
            if "meta_json" in z.files:
                meta = json.loads(str(z["meta_json"]))
    meta = dict(meta, soup_of=members)

    os.makedirs(os.path.join(out_dir, "checkpoints"))
    save_checkpoint(os.path.join(out_dir, "checkpoints", "best.pt"), soup, {"soup_of": ",".join(members)})
    for name in ("best.npz", "model.npz"):
        export_model(soup, os.path.join(out_dir, name), meta)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"phase": "soup", "args": {"runs": members, "out": a.out}, "model": soup.config,
                   "member_distance_to_soup": dist, "play_meta_from": members[0]}, f, indent=2)
    log("wrote runs/%s (best.npz, model.npz, checkpoints/best.pt)" % a.out)


if __name__ == "__main__":
    main()
