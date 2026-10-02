"""Cost of strength (statement 4.3 #4) and responsiveness (statement 5).

    python experiments/cost.py                 # ~3.5 min on 2 cores
    python experiments/cost.py --per-bucket 4 --max-depth 8   # quick

Part 1 - nodes and ms per move vs depth 1..10 on a fixed position set:
  ``--per-bucket`` positions from each ply bucket of data/accuracy_positions.csv
  (seeded) plus 6 opening positions (empty board and 5 tournament openings).
  Ablation configs (same search semantics, same chosen move and score):
    full        transposition table + move ordering (center-first, TT move,
                killers, history) + iterative deepening   [the shipped bot]
    no_tt       ordering, no TT (and therefore no iterative deepening)
    no_order    TT + iterative deepening, columns searched left-to-right
    plain       plain alpha-beta: no TT, no ordering
  A (position, config) series stops once one search exceeds --max-ms, so the
  slow configs do not blow up the run time (missing points are reported).
  ``depth_reached`` < depth means the search proved a win/loss early and
  stopped (deeper search cannot change a proven result).

Part 2 - per slider level ms per move (responsiveness), same positions,
  run with ONE process so timings are not disturbed by parallel workers.

Summary rows come in two subsets: "all" positions, and "open" positions
(not proven won/lost by the shipped search at --max-depth, so every search
really runs to full depth; this is the curve to read for exponential growth).

Outputs: results/cost_raw.csv, results/cost_summary.csv,
results/responsiveness.csv
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RESULTS, Board, default_workers, ensure_dirs, load_accuracy_positions,  # noqa: E402
                    log, make_agent, pmap, random_openings, read_csv, write_csv)
from engine.minimax import search  # noqa: E402

CONFIGS = {"full": (True, True), "no_tt": (False, True), "no_order": (True, False),
           "plain": (False, False)}


def positions(per_bucket, seed):
    rows = load_accuracy_positions()
    rng = random.Random(seed)
    out = [("opening", "")] + [("opening", m) for m in random_openings(5, 2026)]
    for bk in sorted({r["bucket"] for r in rows}, key=lambda s: int(s.split("-")[0])):
        cand = [r for r in rows if r["bucket"] == bk]
        out += [(bk, r["moves"]) for r in rng.sample(cand, per_bucket)]
    return out


def _series(task):
    bucket, moves, cfg, max_depth, max_ms = task
    use_tt, ordering = CONFIGS[cfg]
    b = Board.from_moves(moves)
    out = []
    for d in range(1, max_depth + 1):
        r = search(b, d, use_tt=use_tt, ordering=ordering)
        out.append({"bucket": bucket, "moves": moves, "ply": len(moves), "config": cfg,
                    "depth": d, "nodes": r.nodes, "ms": round(r.ms, 3),
                    "depth_reached": r.depth_reached, "col": r.col, "score": r.score})
        if r.ms > max_ms:
            break
    return out


def summarize(raw, cfgs, max_depth, n_pos):
    """Per config x depth, over ALL positions and over the 'open' positions
    (those the shipped search has NOT proven won/lost by max_depth, i.e.
    where the search really runs to full depth - the honest cost curve)."""
    proven = {r["moves"] for r in raw if r["config"] == "full" and r["depth"] == max_depth
              and r["depth_reached"] < max_depth}
    summ = []
    for subset in ("all", "open"):
        n_sub = n_pos if subset == "all" else n_pos - len(proven)
        for c in cfgs:
            for d in range(1, max_depth + 1):
                sub = [r for r in raw if r["config"] == c and r["depth"] == d
                       and (subset == "all" or r["moves"] not in proven)]
                if not sub:
                    continue
                ms = sorted(r["ms"] for r in sub)
                nodes = sorted(r["nodes"] for r in sub)
                summ.append({"subset": subset, "config": c, "depth": d, "n": len(sub),
                             "complete": int(len(sub) == n_sub),
                             "mean_nodes": round(sum(nodes) / len(sub), 1),
                             "median_nodes": nodes[len(nodes) // 2],
                             "mean_ms": round(sum(ms) / len(sub), 3),
                             "median_ms": ms[len(ms) // 2], "max_ms": ms[-1],
                             "proven_early": sum(1 for r in sub if r["depth_reached"] < d),
                             "mean_depth_reached": round(sum(r["depth_reached"] for r in sub) / len(sub), 2)})
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-bucket", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--max-ms", type=float, default=3000.0,
                    help="stop a (position, config) series after a search this slow")
    ap.add_argument("--configs", default="full,no_tt,no_order,plain")
    ap.add_argument("--no-levels", action="store_true")
    ap.add_argument("--summarize-only", action="store_true",
                    help="recompute cost_summary.csv from an existing cost_raw.csv")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args(argv)
    ensure_dirs()
    pos = positions(args.per_bucket, args.seed)
    cfgs = args.configs.split(",")
    if args.summarize_only:
        raw = read_csv(os.path.join(RESULTS, "cost_raw.csv"))
        for r in raw:
            for k in ("depth", "nodes", "depth_reached"):
                r[k] = int(r[k])
            r["ms"] = float(r["ms"])
        write_csv(os.path.join(RESULTS, "cost_summary.csv"),
                  summarize(raw, cfgs, args.max_depth, len(pos)))
        return
    tasks = [(bk, mv, c, args.max_depth, args.max_ms) for c in cfgs for bk, mv in pos]
    tasks.sort(key=lambda t: (t[2] != "plain", t[2] != "no_order", len(t[1])))
    log("%d positions x %d configs, depths 1-%d" % (len(pos), len(cfgs), args.max_depth))
    t0 = time.time()
    res = pmap(_series, tasks, args.workers, label="series")
    raw = [r for rs in res for r in rs]
    write_csv(os.path.join(RESULTS, "cost_raw.csv"), raw)

    # consistency check: every config must choose the same column/score
    ref = {(r["moves"], r["depth"]): (r["col"], r["score"]) for r in raw if r["config"] == "full"}
    mism = [r for r in raw if (r["moves"], r["depth"]) in ref
            and (r["col"], r["score"]) != ref[(r["moves"], r["depth"])]]
    log("ablation consistency: %d / %d searches differ from 'full'" % (len(mism), len(raw)))

    summ = summarize(raw, cfgs, args.max_depth, len(pos))
    write_csv(os.path.join(RESULTS, "cost_summary.csv"), summ)
    log("cost part done in %.0f s" % (time.time() - t0))
    log("config depth  n  mean_nodes  mean_ms  max_ms  proven_early")
    for s in summ:
        if s["subset"] != "all":
            continue
        log("%-9s %3d %3d %11.0f %8.1f %8.0f %4d" % (s["config"], s["depth"], s["n"], s["mean_nodes"],
                                                    s["mean_ms"], s["max_ms"], s["proven_early"]))

    if args.no_levels:
        return
    # ---- responsiveness: single process, every slider level on every position
    log("\nresponsiveness (single process) ...")
    lv = []
    for level in range(1, 11):
        ms = []
        depths = []
        for i, (bk, mv) in enumerate(pos):
            ag = make_agent("L%d" % level, seed=i)
            ag.epsilon = 0.0           # time the search, not the coin flip
            b = Board.from_moves(mv)
            t = time.perf_counter()
            ag.choose(b)
            ms.append((time.perf_counter() - t) * 1000.0)
            depths.append(ag.last_result.depth_reached)
        ms.sort()
        lv.append({"level": level, "n": len(ms), "mean_ms": round(sum(ms) / len(ms), 1),
                   "median_ms": round(ms[len(ms) // 2], 1),
                   "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1), "max_ms": round(ms[-1], 1),
                   "mean_depth_reached": round(sum(depths) / len(depths), 2),
                   "under_2s": int(ms[-1] < 2000)})
        log("  L%-2d mean %7.1f ms  max %7.1f ms  depth %.1f" % (level, lv[-1]["mean_ms"],
                                                                  lv[-1]["max_ms"], lv[-1]["mean_depth_reached"]))
    write_csv(os.path.join(RESULTS, "responsiveness.csv"), lv)


if __name__ == "__main__":
    main()
