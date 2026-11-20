"""Benchmark score-head nets (and ensembles) on the 3,000 held-out positions.

    python -m nn.eval_score --runs score_8x128_s0
    python -m nn.eval_score --runs score_8x128_s0,score_8x128_s1,score_8x128_s2 --lookahead
    python -m nn.eval_score --runs score_8x128_s0,rs_A_pons_8x128 --out experiments/results/score/ens.json

For each net (mirror-averaged) and for the ensemble of all of them it prints the share of
positions where the move is optimal / keeps the game result, for every head:
  policy   argmax of the policy head
  score    argmax of the score head (needs a nn/train_score.py net)
  combo w  argmax of score + w * log(policy)
  look     1-ply lookahead with the score head: value of column c = exact score if c wins
           at once, else minus the best predicted score of the opponent in the child position
  look+    average of "score" and "look" (two estimates of the same number)
Older nets without a score head contribute their policy to the ensemble only.
Uses best.pt on the GPU when it exists (fast), else best.npz / model.npz with numpy.
"""

import argparse
import json
import os

import numpy as np

from .common import RUNS, log, pick_device

H1 = 7
COMBO_W = (0.25, 0.5, 1.0, 2.0, 4.0)


class Net:
    def __init__(self, run, device):
        self.run = run
        ck = os.path.join(RUNS, run, "checkpoints", "best.pt")
        self.torch = None
        if device.type == "cuda" and os.path.isfile(ck):
            import torch
            from .model import load_checkpoint
            self.torch = torch
            self.model, _ = load_checkpoint(ck, device)
            self.device = device
            self.has_score = getattr(self.model, "has_score", False)
        else:
            from .numpy_net import NumpyNet
            p = os.path.join(RUNS, run, "best.npz")
            self.np = NumpyNet(p if os.path.isfile(p) else os.path.join(RUNS, run, "model.npz"))
            self.has_score = self.np.has_score

    def heads(self, curs, masks, batch=8192):
        """(log policy (B,7), scores in Pons units (B,7) or None), mirror-averaged."""
        from .encode import encode, legal_mask
        legal = legal_mask(masks)
        lps, scs = [], []
        for i in range(0, len(curs), batch):
            x = encode(curs[i:i + batch], masks[i:i + batch])
            if self.torch is not None:
                t = self.torch.from_numpy(x).to(self.device)
                with self.torch.inference_mode():
                    p1, _, s1 = self.model.forward_all(t)
                    p2, _, s2 = self.model.forward_all(t.flip(3))
                p = ((p1.float() + p2.float().flip(1)) / 2).cpu().numpy()
                s = None if s1 is None else ((s1.float() + s2.float().flip(1)) / 2 * 10.0).cpu().numpy()
            else:
                p1, _, s1 = self.np.forward_all(x)
                p2, _, s2 = self.np.forward_all(x[..., ::-1])
                p = (p1 + p2[:, ::-1]) / 2
                s = None if s1 is None else (s1 + s2[:, ::-1]) / 2
            lps.append(p)
            scs.append(s)
        p = np.concatenate(lps)
        p = np.where(legal, p, -1e9)
        p = p - p.max(1, keepdims=True)
        lp = p - np.log(np.exp(p).sum(1, keepdims=True))
        s = None if scs[0] is None else np.concatenate(scs)
        return lp, s


def _wins(cur):
    for sft in (1, H1, H1 - 1, H1 + 1):
        m = cur & (cur >> sft)
        if m & (m >> (2 * sft)):
            return True
    return False


def children(rows):
    """for every (position, column): child (cur, mask) with the opponent to move, or a win flag."""
    idx, ccur, cmask, win = [], [], [], {}
    for i, r in enumerate(rows):
        cur, mask = int(r["cur"]), int(r["mask"])
        for c in range(7):
            if mask >> (c * H1 + 5) & 1:
                continue
            move = (mask + (1 << (c * H1))) & (((1 << 6) - 1) << (c * H1))
            nm, nc = mask | move, cur | move
            if _wins(nc):
                win[(i, c)] = (43 - bin(mask).count("1")) // 2
                continue
            idx.append((i, c))
            ccur.append(nc ^ nm)
            cmask.append(nm)
    return idx, np.array(ccur, np.uint64), np.array(cmask, np.uint64), win


def score_rows(rows, choice):
    sc = [r["scores"] for r in rows]
    opt = np.mean([c in r["optimal"] for r, c in zip(rows, choice)])
    sg = lambda v: (v > 0) - (v < 0)  # noqa: E731
    keep = [sg(s[c]) == sg(r["best"]) for r, s, c in zip(rows, sc, choice) if r["best"] >= 0]
    return round(float(opt), 4), round(float(np.mean(keep)), 4)


def report(name, rows, lp, s, look=None):
    out = {"policy": score_rows(rows, lp.argmax(1))}
    if s is not None:
        legal = lp > -1e8
        sm = np.where(legal, s, -1e9)
        out["score"] = score_rows(rows, sm.argmax(1))
        for w in COMBO_W:
            out["combo%g" % w] = score_rows(rows, (sm + w * lp).argmax(1))
        if look is not None:
            lk = np.where(legal, look, -1e9)
            out["look"] = score_rows(rows, lk.argmax(1))
            out["look+"] = score_rows(rows, ((sm + lk) / 2).argmax(1))
    best = max(out, key=lambda k: out[k][0])
    log("%-28s " % name + "  ".join("%s %.4f" % (k, v[0]) for k, v in out.items()) +
        "   | best: %s %.4f (keeps result %.4f)" % (best, out[best][0], out[best][1]))
    return {k: {"optimal": v[0], "keeps_nl": v[1]} for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma list of run names")
    ap.add_argument("--lookahead", action="store_true", help="also 1-ply lookahead with the score head")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from .bench import load_accuracy_bench
    rows = load_accuracy_bench()
    curs = np.array([r["cur"] for r in rows], np.uint64)
    masks = np.array([r["mask"] for r in rows], np.uint64)
    device = pick_device(a.device)
    if a.lookahead:
        cidx, ccur, cmask, wins = children(rows)
    res, LPs, Ss, LKs = {}, [], [], []
    for run in a.runs.split(","):
        net = Net(run, device)
        lp, s = net.heads(curs, masks)
        look = None
        if a.lookahead and s is not None:
            _, cs = net.heads(ccur, cmask)
            look = np.full((len(rows), 7), -1e9)
            best_child = np.where(cmask_legal(cmask), cs, -1e9).max(1)
            for (i, c), v in zip(cidx, best_child):
                look[i, c] = -v
            for (i, c), v in wins.items():
                look[i, c] = v
            LKs.append(look)
        res[run] = report(run, rows, lp, s, look)
        LPs.append(lp)
        if s is not None:
            Ss.append(s)
    if len(LPs) > 1:
        lp = np.log(np.mean(np.exp(LPs), 0) + 1e-12)
        lp = np.where(LPs[0] > -1e8, lp, -1e9)
        s = np.mean(Ss, 0) if Ss else None
        look = np.mean(LKs, 0) if LKs else None
        res["ENSEMBLE"] = report("ENSEMBLE (%d nets)" % len(LPs), rows, lp, s, look)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"runs": a.runs.split(","), "lookahead": a.lookahead, "results": res}, f, indent=2)
        log("wrote %s" % a.out)


def cmask_legal(masks):
    from .encode import legal_mask
    return legal_mask(masks)


if __name__ == "__main__":
    main()
