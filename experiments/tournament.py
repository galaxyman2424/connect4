"""Depth-vs-depth round robin and slider-level matches  (statement 4.3 #1, 5).

    python experiments/tournament.py                       # full run (~19 min on 2 cores)
    python experiments/tournament.py --openings 10 --l10-openings 2   # quick smoke test

Design
* Openings: ``--openings`` (default 50) distinct random openings of 2-4 plies
  (distinct up to left-right mirroring, seed ``--seed``).  The SAME openings
  are used for every pairing (common random numbers).  With <= 4 plies no side
  can have a win or a threat, so no opening needs to be filtered; we do not
  filter by theoretical value either (the solver is too slow at < 12 stones
  without an opening book) - playing each opening twice with colours swapped
  cancels any opening bias within a pairing.
* Part "depth": fixed-depth bots d1..d8, every pair, every opening x 2 colours
  -> 28 pairings x 100 games.
* Part "slider": every level L1..L10 vs RandomAgent and vs the next level.
  Levels 4-9 are exactly d3..d8 (no randomness, no time limit), so L_k vs
  L_k+1 for k=4..8 reuses the depth games.  Games involving level 10
  (1.9 s/move iterative deepening) use only ``--l10-openings`` openings.
* Seeds: every game gets a deterministic seed from (seed, pairing, opening,
  colour) so epsilon/random agents are reproducible.

Outputs (experiments/results/):
  results.csv              one row per game
  tournament_winrate.csv   depth score matrix (row vs column, draw = 0.5)
  elo.json                 Bradley-Terry Elo (d1 = 1000) with 95% cluster-bootstrap CIs
  tournament_adjacent.csv  d vs d+1 W/D/L, score CI, binomial test
  slider_summary.csv       slider matches (W/D/L, score, Wilson CI, binomial test, ms)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RESULTS, default_workers, ensure_dirs, level_equivalent, log,  # noqa: E402
                    play_game, pmap, random_openings, read_csv, spec_sort_key, write_csv,
                    write_json)
import numpy as np  # noqa: E402

import stats  # noqa: E402

# rough relative cost per game, used only to schedule expensive games first
_COST = {"random": 0, "d1": 1, "d2": 1, "d3": 2, "d4": 4, "d5": 10, "d6": 20,
         "d7": 50, "d8": 100, "L1": 1, "L2": 1, "L3": 1, "L10": 3000}


def _cost(spec):
    return _COST.get(level_equivalent(spec), 1)


def _run(task):
    exp, a, b, oi, opening, a_first, seed = task
    row = play_game(level_equivalent(a), level_equivalent(b), opening, a_first,
                    "%s|%s|%s|%d|%d" % (seed, a, b, oi, a_first))
    row["a"], row["b"] = a, b
    row["first"] = a if a_first else b
    row["winner"] = {1.0: a, 0.0: b, 0.5: "draw"}[row["score_a"]]
    row["experiment"] = exp
    row["opening_id"] = oi
    return row


def build_tasks(args, openings):
    depth_specs = ["d%d" % d for d in range(1, args.max_depth + 1)]
    pairs = []  # (experiment, a, b, n_openings)
    if "depth" in args.parts:
        for i, a in enumerate(depth_specs):
            for b in depth_specs[i + 1:]:
                pairs.append(("depth", a, b, len(openings)))
    if "slider" in args.parts:
        for lvl in range(1, 11):
            n = args.l10_openings if lvl == 10 else len(openings)
            pairs.append(("slider", "L%d" % lvl, "random", n))
        for lvl in range(1, 10):
            n = args.l10_openings if lvl + 1 == 10 else len(openings)
            pairs.append(("slider", "L%d" % lvl, "L%d" % (lvl + 1), n))
    tasks, reused = [], []
    for exp, a, b, n in pairs:
        ca, cb = level_equivalent(a), level_equivalent(b)
        if exp == "slider" and ca.startswith("d") and cb.startswith("d") and \
                int(cb[1:]) <= args.max_depth:
            reused.append((a, b, ca, cb))      # identical bots already in "depth"
            continue
        for oi in range(n):
            for a_first in (True, False):
                tasks.append((exp, a, b, oi, openings[oi], a_first, args.seed))
    tasks.sort(key=lambda t: -(_cost(t[1]) + _cost(t[2])))
    return tasks, reused


def analyze(rows, args):
    """Compute and save the summary tables from results.csv rows (next to --out)."""
    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)
    for r in rows:
        r["score_a"] = float(r["score_a"])
        for k in ("a_ms", "b_ms", "a_max_ms", "b_max_ms", "a_nodes", "b_nodes"):
            r[k] = float(r[k])
    depth_rows = [r for r in rows if r["experiment"] == "depth"]
    players = sorted({r["a"] for r in depth_rows} | {r["b"] for r in depth_rows},
                     key=spec_sort_key)
    m, n = stats.winrate_matrix([(r["a"], r["b"], r["score_a"]) for r in depth_rows], players)
    out = []
    for i, p in enumerate(players):
        row = {"player": p}
        for j, q in enumerate(players):
            row[q] = "" if i == j else round(float(m[i, j]), 4)
        out.append(row)
    write_csv(os.path.join(outdir, "tournament_winrate.csv"), out)

    games = [{"a": r["a"], "b": r["b"], "score_a": r["score_a"],
              "cluster": r["opening_id"]} for r in depth_rows]
    elo, boot = stats.elo_with_bootstrap(games, players, "d1", 1000.0,
                                         reps=args.boot, seed=args.seed)
    write_json(os.path.join(outdir, "elo.json"),
               {"anchor": "d1 = 1000", "method": "max-likelihood Bradley-Terry, draws = 0.5; "
                "95%% CI from %d cluster-bootstrap resamples of openings within each pairing"
                % args.boot,
                "ratings": {p: {"elo": round(v[0], 1), "lo": round(v[1], 1), "hi": round(v[2], 1)}
                            for p, v in elo.items()},
                "step_gain": {"%s->%s" % (players[i], players[i + 1]): {
                    "gain": round(float(elo[players[i + 1]][0] - elo[players[i]][0]), 1),
                    "lo": round(float(np.percentile(boot[:, i + 1] - boot[:, i], 2.5)), 1),
                    "hi": round(float(np.percentile(boot[:, i + 1] - boot[:, i], 97.5)), 1)}
                    for i in range(len(players) - 1)}})

    def wdl(sub, p):
        w = sum(1 for r in sub if r["winner"] == p)
        d = sum(1 for r in sub if r["winner"] == "draw")
        return w, d, len(sub) - w - d

    adj = []
    for i in range(len(players) - 1):
        lo_p, hi_p = players[i], players[i + 1]
        sub = [r for r in depth_rows if {r["a"], r["b"]} == {lo_p, hi_p}]
        w, d, l_ = wdl(sub, hi_p)
        sc = (w + 0.5 * d) / len(sub)
        ci = stats.wilson(w + 0.5 * d, len(sub))
        adj.append({"deeper": hi_p, "shallower": lo_p, "games": len(sub), "wins": w,
                    "draws": d, "losses": l_, "score": round(sc, 4),
                    "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                    "binom_p": "%.3g" % stats.binom_test(w, l_)})
    write_csv(os.path.join(outdir, "tournament_adjacent.csv"), adj)

    slider = []
    srows = [r for r in rows if r["experiment"] == "slider"]
    for a, b in sorted({(r["a"], r["b"]) for r in srows},
                       key=lambda t: (t[1] != "random", spec_sort_key(t[0]))):
        sub = [r for r in srows if r["a"] == a and r["b"] == b]
        w, d, l_ = wdl(sub, a)
        sc = (w + 0.5 * d) / len(sub)
        ci = stats.wilson(w + 0.5 * d, len(sub))
        ms_a = [r["a_ms"] for r in sub if r["a_ms"] > 0]
        mx_a = [r["a_max_ms"] for r in sub]
        slider.append({"level": a, "opponent": b, "games": len(sub), "wins": w, "draws": d,
                       "losses": l_, "score": round(sc, 4), "ci_lo": round(ci[0], 4),
                       "ci_hi": round(ci[1], 4), "binom_p": "%.3g" % stats.binom_test(w, l_),
                       "mean_ms": round(sum(ms_a) / len(ms_a), 1) if ms_a else 0.0,
                       "max_ms": round(max(mx_a), 1) if mx_a else 0.0,
                       "mean_plies": round(sum(int(r["plies"]) for r in sub) / len(sub), 1)})
    write_csv(os.path.join(outdir, "slider_summary.csv"), slider)

    log("\nDepth score matrix (row vs column):")
    log("      " + " ".join("%6s" % q for q in players))
    for i, p in enumerate(players):
        log("%5s " % p + " ".join("%6s" % ("-" if i == j else "%.2f" % m[i, j])
                                  for j in range(len(players))))
    log("\nElo (d1 = 1000, 95% CI):")
    for p in players:
        log("  %s  %7.0f  [%5.0f, %5.0f]" % (p, *elo[p]))
    log("\nAdjacent depths:")
    for a in adj:
        log("  %(deeper)s vs %(shallower)s: +%(wins)d =%(draws)d -%(losses)d  score %(score).3f "
            "[%(ci_lo).3f, %(ci_hi).3f]  p=%(binom_p)s" % a)
    if slider:
        log("\nSlider:")
        for s in slider:
            log("  %(level)s vs %(opponent)s: +%(wins)d =%(draws)d -%(losses)d score %(score).3f "
                "mean %(mean_ms).0f ms (max %(max_ms).0f)" % s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--openings", type=int, default=50, help="openings per pairing (x2 games)")
    ap.add_argument("--l10-openings", type=int, default=25,
                    help="openings for games involving slider level 10 (slow: ~2 s/move)")
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--parts", default="depth,slider",
                    help="which parts to (re)play; rows of other parts are kept from --out. "
                         "The slider part reuses the depth games, so run depth first.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples for Elo CIs")
    ap.add_argument("--analyze-only", action="store_true",
                    help="recompute summaries from an existing results.csv")
    ap.add_argument("--out", default=os.path.join(RESULTS, "results.csv"),
                    help="per-game CSV; the summary tables are written to the same folder")
    args = ap.parse_args(argv)
    args.parts = args.parts.split(",")
    ensure_dirs()

    if not args.analyze_only:
        openings = random_openings(args.openings, args.seed)
        tasks, reused = build_tasks(args, openings)
        log("%d games to play on %d workers (%d slider pairings reuse depth games)"
            % (len(tasks), args.workers, len(reused)))
        t0 = time.time()
        rows = pmap(_run, tasks, args.workers, label="games")
        kept = []
        if os.path.exists(args.out):
            kept = [r for r in read_csv(args.out) if r["experiment"] not in args.parts]
            for r in kept:
                r["score_a"] = float(r["score_a"])
                r["opening_id"] = int(r["opening_id"])
        depth_rows = [r for r in rows + kept if r["experiment"] == "depth"]
        for a, b, ca, cb in reused:
            for r in depth_rows:
                if {r["a"], r["b"]} == {ca, cb}:
                    c = dict(r)
                    c["experiment"] = "slider"
                    if r["a"] != ca:   # re-orient so that column a is level a
                        for k in list(c):
                            if k.startswith("a_"):
                                c[k], c["b_" + k[2:]] = r["b_" + k[2:]], r[k]
                        c["score_a"] = 1.0 - r["score_a"]
                    name = {ca: a, cb: b}
                    c["a"], c["b"] = a, b
                    c["first"] = name[r["first"]]
                    c["winner"] = name.get(r["winner"], "draw")
                    rows.append(c)
        rows += kept
        rows.sort(key=lambda r: (r["experiment"], spec_sort_key(r["a"]), spec_sort_key(r["b"]),
                                 r["opening_id"], r["first"] != r["a"]))
        cols = ["experiment", "a", "b", "opening_id", "opening", "first", "winner", "score_a",
                "plies", "moves", "a_moves", "a_random_moves", "a_ms", "a_max_ms", "a_nodes",
                "a_depth", "b_moves", "b_random_moves", "b_ms", "b_max_ms", "b_nodes", "b_depth"]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_csv(args.out, rows, cols)
        log("wrote %s (%d games) in %.0f s" % (args.out, len(rows), time.time() - t0))
    rows = read_csv(args.out)
    analyze(rows, args)


if __name__ == "__main__":
    main()
