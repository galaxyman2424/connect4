"""Linear probes inside the open-source AlphaZero net (Bruneton et al., 1 block x 256).

    python -m research.interp_bruneton

Same 50,000 positions, splits and concepts as research/interp.py (seed 2026), so the
numbers sit next to our nets.  Their architecture differs: 3 input planes (yellow, red,
side-to-move +-1), a 4x4 stem conv that maps 6x7 to 5x6, one residual block, a policy
head (2 filters) and a tanh value head (1 filter -> 256 hidden -> 1).  Layers probed:
input, stem, block1, pol_conv, val_conv, val_hidden.  Not done here (needs our TorchNet):
channel census, group ablations, patching, opp_reply.  Done: block knock-out.

Writes experiments/results/research/interp_bruneton.json.
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nn.encode import bits_to_grid, encode, legal_mask  # noqa: E402

from .interp import CONCEPTS, METRIC, concept_labels, load_data, log, norm_skill, rnd, splits  # noqa: E402
from .interp_core import chance_score, compute_concepts, train_probes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BDIR = os.path.join(ROOT, "arena", "third_party", "bruneton_az")
OUT = os.path.join(ROOT, "experiments", "results", "research", "interp_bruneton.json")
SKIP = {"opp_reply", "opp_net_move"}


def their_planes(cur, mask, Game):
    me = bits_to_grid(cur)
    th = bits_to_grid(cur ^ mask)
    out = np.zeros((len(cur), 3, 6, 7), np.float32)
    g = Game()
    for i in range(len(cur)):
        first = me[i].sum() == th[i].sum()          # side to move is yellow (first player)
        yel, red = (me[i], th[i]) if first else (th[i], me[i])
        ys = rs = 0
        for r in range(6):
            h = 5 - r
            for c in range(7):
                if yel[r, c]:
                    ys |= 1 << (8 * c + h)
                elif red[r, c]:
                    rs |= 1 << (8 * c + h)
        out[i] = g.state_flattener([ys, rs, 1 if first else -1]).reshape(3, 6, 7)
    return out


def run(net, x, capture, skip_block=False):
    acts = {}
    h = net.group1(x)
    acts["stem"] = h
    if skip_block:
        h = torch.relu(h)
    else:
        h = net.layer1(h)
    acts["block1"] = h
    p = net.relu_pol(net.bnpolicy(net.policy_entrance(h)))
    acts["pol_conv"] = p
    logits = net.fcpol2(p.flatten(1))
    v = net.relu_val(net.bnvalue(net.value_entrance(h)))
    acts["val_conv"] = v
    vh = net.relu_hidden_val(net.hidden_dense_value(v.flatten(1)))
    acts["val_hidden"] = vh
    val = torch.tanh(net.fcval(vh))[:, 0]
    return logits, val, {k: acts[k] for k in capture}


@torch.no_grad()
def batched(net, X, layer=None, skip_block=False, bs=2048):
    L, V, A = [], [], []
    for s in range(0, len(X), bs):
        xb = torch.as_tensor(X[s:s + bs])
        lg, v, acts = run(net, xb, (layer,) if layer and layer != "input" else (), skip_block)
        L.append(lg)
        V.append(v)
        if layer == "input":
            A.append(xb.flatten(1))
        elif layer:
            A.append(acts[layer].flatten(1))
    return torch.cat(L).numpy(), torch.cat(V).numpy(), (torch.cat(A) if A else None)


def main():
    t0 = time.time()
    sys.path.insert(0, BDIR)
    import Game_bitboard
    import ResNet
    nthreads = torch.get_num_threads()
    net = ResNet.resnet18()
    torch.set_num_threads(nthreads)
    net.load_state_dict(torch.load(os.path.join(BDIR, "best_model_resnet.pth"), map_location="cpu", weights_only=True))
    net.eval()
    params = int(sum(p.numel() for p in net.parameters()))
    cur, mask, scores, ply = load_data(50000, 2026)
    tr, va, te = splits(len(cur))
    c, extra = compute_concepts(cur, mask, scores, ply)
    labels = {k: v for k, v in concept_labels(c).items() if k not in SKIP}
    kinds = {cid: (kind, K) for cid, _, _, kind, K in CONCEPTS}
    Xb = their_planes(cur, mask, Game_bitboard.Game)
    log("bruneton: planes built (%.0fs)" % (time.time() - t0))
    legal = legal_mask(mask)
    lg, val, _ = batched(net, Xb)
    lg[~legal] = -np.inf
    labels["net_move"] = lg.argmax(1)
    best = scores.max(1)
    opt = scores == best[:, None]
    outcome = np.where(best > 0, 0, np.where(best == 0, 1, 2))
    ix = np.arange(len(cur))

    def quality(lgx, vx):
        lgx = lgx.copy()
        lgx[~legal] = -np.inf
        mv = lgx.argmax(1)
        pred = np.where(vx > 0.33, 0, np.where(vx < -0.33, 2, 1))
        return {"agree": float(opt[ix, mv][te].mean()), "value_acc_thirds": float((pred == outcome)[te].mean()),
                "value_corr": float(np.corrcoef(vx[te], -np.sign(outcome[te] - 1))[0, 1])}

    intact = quality(lg, val)
    lg2, val2, _ = batched(net, Xb, skip_block=True)
    knock = quality(lg2, val2)
    heads = [{"id": k, "kind": kinds[k][0], "K": kinds[k][1], "y": y} for k, y in labels.items()]
    chance = {k: chance_score(kinds[k][0], y[tr], y[te], kinds[k][1]) for k, y in labels.items()}
    probes = {k: {} for k in labels}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for layer in ("input", "stem", "block1", "pol_conv", "val_conv", "val_hidden"):
        ts = time.time()
        A = batched(net, Xb, layer)[2].to(dev)
        flat, _ = train_probes(A, heads, tr, va, te, rounds=10, seed=1)
        shuf, _ = train_probes(A, heads, tr, va, te, rounds=10, seed=2, shuffle=True)
        for k in labels:
            probes[k][layer] = {"flat": flat.get(k), "shuffled": shuf.get(k),
                                "skill": norm_skill(flat.get(k), chance[k])}
        log("  %s: %d features, %.0fs" % (layer, A.shape[1], time.time() - ts))
        del A
        torch.cuda.empty_cache()
    # same input through OUR encoding for reference: not needed (input probes use their planes)
    R = {"id": "interp_bruneton", "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "params": params,
         "layers": ["input", "stem", "block1", "pol_conv", "val_conv", "val_hidden"],
         "chance": chance, "metric": {cid: METRIC[kind] for cid, _, _, kind, _ in CONCEPTS},
         "probes": probes, "intact": intact, "block1_skipped": knock,
         "notes": ["Positions, splits and concept labels identical to research/interp.py (seed 2026, n=50000).",
                   "value_acc_thirds: tanh value > 1/3 -> win, < -1/3 -> loss, else draw (their head has no draw class).",
                   "Their stem is a 4x4 conv with padding 1, so every layer after the input is 5x6 = 30 cells.",
                   "block1_skipped: residual block replaced by identity (its input is post-ReLU)."],
         "seconds": round(time.time() - t0, 1)}
    json.dump(rnd(R), open(OUT, "w"), indent=1)
    log("wrote %s (%.0fs)" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
