"""Accuracy against perfect play (statement 4.3 #2).

    python experiments/accuracy.py                     # data/accuracy_positions.csv, depths 1-10
    python experiments/accuracy.py --source tonycwang  # data/tonycwang_sample.csv (after data/download_data.py)
    python experiments/accuracy.py --source uci        # data/uci/uci_positions.csv (after data/download_data.py)

Metrics per depth (solver sources):
  optimal     chosen column is one of the solver's max-score columns
  keeps       chosen column keeps the game-theoretic outcome (win stays win,
              draw stays draw; in lost positions every move keeps it)
  keeps_nl    'keeps' restricted to positions that are not already lost
              (the informative version: share of non-blunders)
  random baseline = expected value of the same metric for a uniformly random
              legal move (so the numbers can be read against chance).
  Paired exact McNemar tests between consecutive depths (same positions).

UCI source: positions after 8 plies labeled win/loss/draw for the FIRST
player (who is to move at ply 8).  The data has no per-column labels, so the
metric is outcome agreement: does the sign of the bot's root score (from
P1's view) match the label?  Draws are reported separately; a heuristic score
is not a calibrated outcome prediction, so read this as a weak sanity check.

Outputs (experiments/results/): accuracy_<source>_moves.csv (position x
depth), accuracy_<source>_summary.csv (by depth, and by depth x ply bucket),
accuracy_<source>_tests.csv (McNemar d vs d+1).
"""

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, RESULTS, Board, board_from_grid_any, default_workers,  # noqa: E402
                    ensure_dirs, load_accuracy_positions, log, pmap, write_csv)
import stats  # noqa: E402
from engine.minimax import search  # noqa: E402

DOWNLOAD_HINT = "run `python data/download_data.py --%s` first (see data/README.md)"


def parse_depths(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",") if x]


def bucket_of(ply):
    for lo, hi in ((8, 11), (12, 15), (16, 19), (20, 23), (24, 27), (28, 30)):
        if lo <= ply <= hi:
            return "%d-%d" % (lo, hi)
    return "%d-%d" % (ply // 4 * 4, ply // 4 * 4 + 3)


# ------------------------------------------------------------- loaders
def load_solver_source(name):
    if name == "accuracy":
        rows = load_accuracy_positions()
        for i, r in enumerate(rows):
            r["pid"] = i
        return rows
    path = os.path.join(DATA, "tonycwang_sample.csv")
    if not os.path.exists(path):
        sys.exit("%s not found - %s" % (path, DOWNLOAD_HINT % "tonycwang"))
    rows, skipped, flipped = [], 0, 0
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            b, flip = board_from_grid_any(json.loads(r["grid"]))
            if b is None or b.is_terminal():
                skipped += 1
                continue
            flipped += bool(flip)
            scores = json.loads(r["scores"])
            legal = set(b.legal_moves())
            if any((s is None) != (c not in legal) for c, s in enumerate(scores)):
                skipped += 1    # score vector inconsistent with this board
                continue
            try:
                mv = b.to_move_string()
            except ValueError:
                skipped += 1
                continue
            rows.append({"pid": i, "moves": mv, "ply": b.moves_played, "scores": scores,
                         "optimal_cols": json.loads(r["optimal_cols"]), "source": "tonycwang",
                         "bucket": bucket_of(b.moves_played)})
    log("tonycwang: %d usable positions, %d skipped (invalid/terminal/inconsistent), "
        "%d read with row0=bottom" % (len(rows), skipped, flipped))
    if not rows:
        sys.exit("no usable positions in %s" % path)
    return rows


def load_uci(n, seed):
    path = os.path.join(DATA, "uci", "uci_positions.csv")
    if not os.path.exists(path):
        sys.exit("%s not found - %s" % (path, DOWNLOAD_HINT % "uci"))
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append({"pid": i, "grid": json.loads(r["grid"]), "label": r["label"]})
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


# ---------------------------------------------------------- evaluation
def _eval_solver(task):
    pos, depths = task
    b = Board.from_moves(pos["moves"])
    sc = pos["scores"]
    legal = [c for c in range(7) if sc[c] is not None]
    best = max(sc[c] for c in legal)
    opt = set(pos["optimal_cols"])
    sign = lambda v: (v > 0) - (v < 0)  # noqa: E731
    rand_opt = len(opt) / len(legal)
    rand_keep = sum(1 for c in legal if sign(sc[c]) == sign(best)) / len(legal)
    out = []
    for d in depths:
        r = search(b, d)
        out.append({"pid": pos["pid"], "moves": pos["moves"], "ply": pos["ply"],
                    "bucket": pos["bucket"], "outcome": ("win", "draw", "loss")[1 - sign(best)],
                    "depth": d, "col": r.col, "optimal": int(r.col in opt),
                    "keeps": int(sign(sc[r.col]) == sign(best)),
                    "chosen_score": sc[r.col], "best_score": best,
                    "rand_optimal": round(rand_opt, 4), "rand_keeps": round(rand_keep, 4),
                    "eval": r.score, "nodes": r.nodes, "ms": round(r.ms, 3),
                    "depth_reached": r.depth_reached})
    return out


def _eval_uci(task):
    pos, depths = task
    b, _ = board_from_grid_any(pos["grid"])
    out = []
    if b is None or b.is_terminal():
        return out
    for d in depths:
        r = search(b, d)
        p1_score = r.score if b.to_move == 1 else -r.score
        pred = "win" if p1_score > 0 else ("loss" if p1_score < 0 else "draw")
        out.append({"pid": pos["pid"], "label": pos["label"], "depth": d, "col": r.col,
                    "eval_p1": p1_score, "pred": pred, "agree": int(pred == pos["label"]),
                    "proven": int(abs(r.score) >= 999900), "nodes": r.nodes,
                    "ms": round(r.ms, 3)})
    return out


def summarize_solver(rows, depths, tag):
    summ = []
    groups = [("all", lambda r: True)] + \
        [(bk, (lambda bk: lambda r: r["bucket"] == bk)(bk))
         for bk in sorted({r["bucket"] for r in rows}, key=lambda s: int(s.split("-")[0]))]
    for d in depths:
        for gname, gf in groups:
            sub = [r for r in rows if r["depth"] == d and gf(r)]
            if not sub:
                continue
            n = len(sub)
            k = sum(r["optimal"] for r in sub)
            nl = [r for r in sub if r["outcome"] != "loss"]
            knl = sum(r["keeps"] for r in nl)
            ci = stats.wilson(k, n)
            ci2 = stats.wilson(knl, len(nl)) if nl else (float("nan"),) * 2
            summ.append({"depth": d, "bucket": gname, "n": n,
                         "optimal": round(k / n, 4), "opt_lo": round(ci[0], 4), "opt_hi": round(ci[1], 4),
                         "keeps": round(sum(r["keeps"] for r in sub) / n, 4),
                         "n_not_lost": len(nl),
                         "keeps_nl": round(knl / len(nl), 4) if nl else "",
                         "keeps_nl_lo": round(ci2[0], 4) if nl else "",
                         "keeps_nl_hi": round(ci2[1], 4) if nl else "",
                         "rand_optimal": round(sum(r["rand_optimal"] for r in sub) / n, 4),
                         "rand_keeps_nl": round(sum(r["rand_keeps"] for r in nl) / len(nl), 4) if nl else "",
                         "mean_eval": round(sum(max(-1000, min(1000, r["eval"])) for r in sub) / n, 3),
                         "mean_nodes": round(sum(r["nodes"] for r in sub) / n, 1),
                         "mean_ms": round(sum(r["ms"] for r in sub) / n, 3)})
    write_csv(os.path.join(RESULTS, "accuracy_%s_summary.csv" % tag), summ)
    # McNemar tests between consecutive depths
    by = {(r["pid"], r["depth"]): r for r in rows}
    pids = sorted({r["pid"] for r in rows})
    tests = []
    for d1, d2 in zip(depths, depths[1:]):
        for metric in ("optimal", "keeps"):
            common = [p for p in pids if (p, d1) in by and (p, d2) in by]
            b = sum(1 for p in common if by[(p, d1)][metric] and not by[(p, d2)][metric])
            c = sum(1 for p in common if by[(p, d2)][metric] and not by[(p, d1)][metric])
            tests.append({"from": d1, "to": d2, "metric": metric, "n": len(common),
                          "only_shallower": b, "only_deeper": c,
                          "delta": round((c - b) / len(common), 4) if common else "",
                          "mcnemar_p": "%.3g" % stats.mcnemar(b, c)})
    write_csv(os.path.join(RESULTS, "accuracy_%s_tests.csv" % tag), tests)
    log("\ndepth   n   optimal [95% CI]      keeps(not lost)  random-opt  ms/move")
    for s in summ:
        if s["bucket"] == "all":
            log("%5d %5d   %.3f [%.3f,%.3f]   %s           %.3f     %.1f" % (
                s["depth"], s["n"], s["optimal"], s["opt_lo"], s["opt_hi"], s["keeps_nl"],
                s["rand_optimal"], s["mean_ms"]))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["accuracy", "tonycwang", "uci"], default="accuracy")
    ap.add_argument("--depths", default="1-10")
    ap.add_argument("--deep-depths", default="",
                    help="extra depths evaluated only on a stratified subset ('' to skip)")
    ap.add_argument("--deep-sample", type=int, default=600,
                    help="subset size for --deep-depths (stratified by ply bucket)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N positions (0 = all)")
    ap.add_argument("--uci-n", type=int, default=2000, help="UCI positions to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args(argv)
    ensure_dirs()
    depths = parse_depths(args.depths)
    deep = parse_depths(args.deep_depths) if args.deep_depths else []

    if args.source == "uci":
        pos = load_uci(args.uci_n, args.seed)
        res = pmap(_eval_uci, [(p, depths) for p in pos], args.workers, label="uci")
        rows = [r for rs in res for r in rs]
        write_csv(os.path.join(RESULTS, "accuracy_uci_moves.csv"), rows)
        summ = []
        for d in depths:
            sub = [r for r in rows if r["depth"] == d]
            dec = [r for r in sub if r["label"] != "draw"]
            k = sum(r["agree"] for r in dec)
            ci = stats.wilson(k, len(dec))
            summ.append({"depth": d, "n": len(sub), "n_decisive": len(dec),
                         "sign_agree_decisive": round(k / len(dec), 4) if dec else "",
                         "lo": round(ci[0], 4), "hi": round(ci[1], 4),
                         "proven_share": round(sum(r["proven"] for r in sub) / len(sub), 4),
                         "majority_baseline": round(max(
                             sum(1 for r in dec if r["label"] == "win"),
                             sum(1 for r in dec if r["label"] == "loss")) / len(dec), 4) if dec else ""})
        write_csv(os.path.join(RESULTS, "accuracy_uci_summary.csv"), summ)
        for s in summ:
            log("depth %(depth)d: sign agreement on decisive %(sign_agree_decisive)s "
                "[%(lo)s, %(hi)s] (majority baseline %(majority_baseline)s), proven %(proven_share)s" % s)
        return

    pos = load_solver_source(args.source)
    if args.limit:
        pos = pos[:args.limit]
    tag = args.source
    log("%s: %d positions, depths %s" % (tag, len(pos), depths))
    res = pmap(_eval_solver, [(p, depths) for p in pos], args.workers, label="positions")
    rows = [r for rs in res for r in rs]
    if deep:
        rng = random.Random(args.seed)
        buckets = sorted({p["bucket"] for p in pos})
        per = max(1, args.deep_sample // len(buckets))
        sub = []
        for bk in buckets:
            cand = [p for p in pos if p["bucket"] == bk]
            sub += rng.sample(cand, min(per, len(cand)))
        log("deep depths %s on %d positions" % (deep, len(sub)))
        res = pmap(_eval_solver, [(p, deep) for p in sub], args.workers, label="deep")
        deep_rows = [r for rs in res for r in rs]
        sub_ids = {p["pid"] for p in sub}
        for r in rows + deep_rows:
            r["deep_subset"] = int(r["pid"] in sub_ids)
        rows += deep_rows
        # summary on the subset alone (so depths 1-10 are compared on equal footing)
        summarize_solver([r for r in rows if r["deep_subset"]], depths + deep, tag + "_subset")
    write_csv(os.path.join(RESULTS, "accuracy_%s_moves.csv" % tag), rows)
    summarize_solver([r for r in rows if r["depth"] in depths], depths, tag)


if __name__ == "__main__":
    main()
