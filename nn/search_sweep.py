"""How much does more search help a net?  MCTS budget x value source, plus how deep the tree looks.

    python -m nn.search_sweep --run score_8x128_s0_r1                    # GPU, ~20-40 min
    python -m nn.search_sweep --run score_8x128_s0_r1 --sims 0,50,200,800,3200 --positions 1200
    python -m nn.search_sweep --run sup_6x64 --values wdl --bench data/accuracy_positions_50k.csv --positions 6000
    python -m nn.search_sweep --model runs/a/best.npz+runs/b/best.npz+runs/c/best.npz --run ens3

For every (value source, sims) on a ply-stratified sample of the benchmark:
  optimal / keeps_nl (same definitions as nn.evaluate, with 95% intervals),
  ms per position, and what the search tree actually looked at:
    pv_len     length of the principal variation (most-visited line) = how far ahead
               the search "sees" along its main line
    max_depth  deepest node in the tree
    root_children_visited  how many of the legal moves got >= 1 visit
  sims 0 = the net's own move (no search).

Value sources (score-head nets only for 'score'):
  wdl    P(win) - P(loss) from the WDL head (what MCTS always used): cannot tell a fast win
         from a slow one, so MCTS happily plays a slower win -> 'optimal' undercounts
  score  value from the score head's best column (nn/numpy_net.score_to_value): ranks
         faster wins higher, so search can also chase the FASTEST win
Writes experiments/results/search_sweep_<run>.json.
"""

import argparse
import json
import os
import time

import numpy as np

from . import bench, mcts
from .common import RESULTS, RUNS, load_evaluator, log


def tree_stats(root):
    """(pv length, max depth, root children visited)."""
    if root.c_max_depth is not None:          # searched by the C tree
        return len(root.c_pv), root.c_max_depth, sum(1 for c in root.legal if root.N[c] > 0)
    best = 0
    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        best = max(best, d)
        if node.expanded:
            for a in node.legal or []:
                ch = node.children[a]
                if ch is not None and node.N[a] > 0:
                    stack.append((ch, d + 1))
    return len(mcts.principal_variation(root, max_len=42)), best, sum(1 for c in root.legal if root.N[c] > 0)


def run_one(ev, rows, sims, batch, c_puct, root_exact_from=0):
    t0 = time.perf_counter()
    solved = {}
    if root_exact_from:
        from .exact_tail import exact_value
        for i, r in enumerate(rows):
            if r["ply"] >= root_exact_from:
                solved[i] = exact_value(r["cur"], r["mask"])[1]
    todo = [i for i in range(len(rows)) if i not in solved]
    all_rows, rows = rows, [rows[i] for i in todo]
    if sims == 0:
        cols = []
        for i in range(0, len(rows), 1024):
            ch = rows[i:i + 1024]
            pri, _, _ = ev.evaluate([r["cur"] for r in ch], [r["mask"] for r in ch])
            for r, p in zip(ch, pri):
                legal = [c for c in range(7) if r["scores"][c] is not None]
                cols.append(max(legal, key=lambda c: p[c]))
        st = None
    else:
        cols, pv, md, rc = [], [], [], []
        for i in range(0, len(rows), batch):
            ch = rows[i:i + batch]
            roots = [mcts.Node(r["cur"], r["mask"]) for r in ch]
            mcts.search(roots, ev, sims, c_puct=c_puct)
            for rt in roots:
                cols.append(mcts.choose_move(rt, 0.0))
                a, b, c = tree_stats(rt)
                pv.append(a); md.append(b); rc.append(c)
        st = {"pv_len_mean": round(float(np.mean(pv)), 2), "pv_len_p90": int(np.percentile(pv, 90)),
              "max_depth_mean": round(float(np.mean(md)), 2), "root_children_visited_mean": round(float(np.mean(rc)), 2)}
    dt = time.perf_counter() - t0
    full = [None] * len(all_rows)
    for i, c in solved.items():
        full[i] = c
    for i, c in zip(todo, cols):
        full[i] = c
    rows, cols = all_rows, full
    s = bench.accuracy_summary(rows, cols)
    out = {"sims": sims, "optimal": s["optimal"], "optimal_ci95": s["optimal_ci95"], "keeps_nl": s["keeps_nl"],
           "keeps_nl_ci95": s["keeps_nl_ci95"], "by_bucket": {k: v["optimal"] for k, v in s["by_bucket"].items()},
           "ms_per_position": round(1000 * dt / len(rows), 2)}
    if st:
        out["tree"] = st
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run name (results file name; model from runs/<run>/ unless --model)")
    ap.add_argument("--model", default="", help="model path(s); default runs/<run>/best.npz")
    ap.add_argument("--sims", default="0,50,200,800,1600")
    ap.add_argument("--values", default="wdl,score", help="value sources to compare: wdl, score")
    ap.add_argument("--positions", type=int, default=1200, help="ply-stratified benchmark sample (0 = all)")
    ap.add_argument("--bench", default="")
    ap.add_argument("--batch", type=int, default=600, help="positions searched in lock-step")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--exact-from", type=int, default=0,
                    help="MCTS leaves at this ply or later get exact values from the C alpha-beta (nn/exact_tail.py); 0 = off")
    ap.add_argument("--root-exact-from", type=int, default=0,
                    help="positions at this ply or later are solved by the C alpha-beta instead of the net (hybrid); 0 = off")
    ap.add_argument("--backend", default="torch", choices=("numpy", "torch"))
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    model = a.model or os.path.join(RUNS, a.run, "best.npz")
    rows = bench.stratified(bench.load_accuracy_bench(a.bench or None), a.positions, seed=5)
    res = {"run": a.run, "model": model, "positions": len(rows), "benchmark": bench.bench_path(a.bench or None),
           "c_puct": a.c_puct, "exact_from": a.exact_from, "root_exact_from": a.root_exact_from, "results": {}}
    os.makedirs(RESULTS, exist_ok=True)
    tag = a.run.replace("/", "_") + ("_exact%d" % a.exact_from if a.exact_from else "") + \
        ("_root%d" % a.root_exact_from if a.root_exact_from else "")
    out_path = os.path.join(RESULTS, "search_sweep_%s.json" % tag)
    for vs in [v for v in a.values.split(",") if v]:
        os.environ["C4_VALUE_SOURCE"] = vs
        ev = load_evaluator(model, a.device, backend=a.backend)
        if a.exact_from:
            from .exact_tail import ExactTailEvaluator
            ev0 = ev
            ev = ExactTailEvaluator(ev0, a.exact_from)
            ev.has_score = all(getattr(e, "has_score", False) for e in getattr(ev0, "evs", [ev0]))
        has_score = all(getattr(e, "has_score", False) for e in getattr(ev, "evs", [ev]))
        if vs == "score" and not has_score:
            log("skipping value source 'score': %s has no score head" % model)
            continue
        res["results"][vs] = []
        for s in [int(x) for x in a.sims.split(",") if x != ""]:
            if vs == "score" and s == 0 and res["results"].get("wdl"):
                continue                        # identical to wdl at 0 sims (no search)
            r = run_one(ev, rows, s, a.batch, a.c_puct, a.root_exact_from)
            res["results"][vs].append(r)
            log("value=%-5s sims=%-5d optimal %.4f (%.4f-%.4f) keeps %.4f  %7.1f ms/pos  %s" % (
                vs, s, r["optimal"], r["optimal_ci95"][0], r["optimal_ci95"][1], r["keeps_nl"],
                r["ms_per_position"], r.get("tree", "")))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
    os.environ.pop("C4_VALUE_SOURCE", None)
    log("wrote %s" % out_path)


if __name__ == "__main__":
    main()
