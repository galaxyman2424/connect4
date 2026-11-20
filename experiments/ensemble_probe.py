"""Quick probe: does averaging several trained nets (plus mirror images) raise policy
accuracy on the 3,000-position benchmark?  No training; numpy only.

    python -m experiments.ensemble_probe
    python -m experiments.ensemble_probe --nets rs_A_pons_8x128,rs_A_pons_10x128,rs_A_pons_6x96

Prints optimal / keeps-result accuracy for each net alone (policy, policy+mirror) and for
the ensemble, plus 1-ply lookahead with the ensemble's value head.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nn.bench import load_accuracy_bench  # noqa: E402
from nn.encode import encode, legal_mask, mirror_bits, to_u64  # noqa: E402
from nn.numpy_net import NumpyNet, _softmax  # noqa: E402

H1 = 7


def scores(rows, cols):
    opt = np.mean([c in r["optimal"] for r, c in zip(rows, cols)])
    keep = [(r["scores"][c] > 0) - (r["scores"][c] < 0) == (r["best"] > 0) - (r["best"] < 0)
            for r, c in zip(rows, cols) if r["best"] >= 0]
    return opt, float(np.mean(keep))


def net_probs(net, curs, masks, mirror=True):
    """policy probs (B,7) and wdl (B,3), optionally averaged with the mirror image."""
    def run(c, m):
        lg, v = net.forward(encode(c, m))
        return lg, _softmax(v)
    lg, wdl = run(curs, masks)
    legal = legal_mask(masks)
    p = _softmax(np.where(legal, lg, -np.inf))
    if mirror:
        lg2, wdl2 = run(mirror_bits(curs), mirror_bits(masks))
        p2 = _softmax(np.where(legal, lg2[:, ::-1], -np.inf))
        p, wdl = (p + p2) / 2, (wdl + wdl2) / 2
    return p, wdl


def one_ply(nets, rows, curs, masks):
    """pick the column whose child position is worst for the opponent (ensemble value), ties by policy."""
    B = len(rows)
    val = np.full((B, 7), -9.0)
    for c in range(7):
        idx, ch_cur, ch_mask, win = [], [], [], []
        for i in range(B):
            cur, mask = int(curs[i]), int(masks[i])
            if mask >> (c * H1 + 5) & 1:
                continue
            move = (mask + (1 << (c * H1))) & (((1 << 6) - 1) << (c * H1))
            nm = mask | move
            nc = cur | (move & ~mask)
            # immediate win check on nc
            won = False
            for s in (1, H1, H1 - 1, H1 + 1):
                m = nc & (nc >> s)
                if m & (m >> 2 * s):
                    won = True
                    break
            if won:
                val[i, c] = 2.0
                continue
            idx.append(i)
            ch_cur.append(nc ^ nm)        # opponent to move
            ch_mask.append(nm)
        if not idx:
            continue
        cc, cm = to_u64(ch_cur), to_u64(ch_mask)
        v = np.zeros(len(idx))
        for n in nets:
            _, w = net_probs(n, cc, cm)
            v += w[:, 0] - w[:, 2]
        val[idx, c] = -v / len(nets)
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="rs_A_pons_8x128,rs_A_pons_10x128,rs_A_pons_6x96,sup_6x64,rs_A_pons_6x64,rs_A_pons_6x64_s1")
    a = ap.parse_args()
    rows = load_accuracy_bench()
    curs, masks = to_u64([r["cur"] for r in rows]), to_u64([r["mask"] for r in rows])
    nets, probs, vals = [], [], []
    for name in a.nets.split(","):
        n = NumpyNet(os.path.join(ROOT, "runs", name, "model.npz"))
        p0, _ = net_probs(n, curs, masks, mirror=False)
        p, _ = net_probs(n, curs, masks)
        print("%-22s policy %.4f   +mirror %.4f   keeps-result %.4f" % (name, scores(rows, p0.argmax(1))[0],
                                                                       *scores(rows, p.argmax(1))))
        nets.append(n)
        probs.append(p)
    P = np.mean(probs, axis=0)
    print("%-22s +mirror %.4f   keeps-result %.4f" % ("ENSEMBLE (%d)" % len(nets), *scores(rows, P.argmax(1))))
    V = one_ply(nets, rows, curs, masks)
    combo = V + 0.05 * P                         # value first, policy breaks near-ties
    print("%-22s 1-ply    %.4f   keeps-result %.4f" % ("ENSEMBLE value", *scores(rows, combo.argmax(1))))


if __name__ == "__main__":
    main()
