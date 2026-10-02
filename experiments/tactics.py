"""Tactics suite (statement 4.3 #3, 5 "Tactical soundness").

    python experiments/tactics.py               # evaluate (generates the suite first if missing)
    python experiments/tactics.py --regenerate  # rebuild tactics_positions.json (needs the solver)

Categories (13 positions each, 52 total), side to move = "we":
  win1   we have an immediate winning move.
         Solved = play a winning column.
  block  we have no immediate win; the opponent has exactly one playable
         winning cell, so every other column loses on the spot.
         Solved = play that column.
  win2   forced win with our 2nd move (3 plies), no win in 1, opponent has no
         playable threat (pure attack).  Solved = play a column that keeps the
         fastest forced win.
  win3   forced win with our 3rd move (5 plies), no faster win, opponent has
         no playable threat.  Solved = play a column that keeps the fastest
         forced win.  (We also report the lenient "still winning" rate.)

Generation: positions from seeded semi-random games (each move: 50% random,
50% depth-2 bot), plies 14-34, deduplicated up to mirroring; candidates found
with the engine, then EVERY label is verified with Pons' solver:
  win1 : max solver score = 21 - ply//2 (win at our next stone)
  win2 : max = 20 - ply//2;   win3 : max = 19 - ply//2
  block: the block column is the unique solver-optimal column
and the stored solution set is exactly the solver's set of max-score columns.

Outputs: experiments/tactics_positions.json, results/tactics_results.csv
(per position x depth), results/tactics_summary.csv (share solved by
category x depth, plus the expected rate of slider levels 1-3 with random
moves).
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (HERE, LEVELS, RESULTS, Board, canonical_key, default_workers,  # noqa: E402
                    ensure_dirs, log, pmap, solver_available, write_csv)
from engine.board import BOARD_MASK, BOTTOM_MASK, COLUMN_MASK, winning_cells_mask  # noqa: E402
from engine.minimax import WIN, search  # noqa: E402

SUITE = os.path.join(HERE, "tactics_positions.json")
CATS = ("win1", "block", "win2", "win3")


def _cols(bits):
    return [c for c in range(7) if bits & COLUMN_MASK[c]]


def classify(b):
    """Engine-based candidate category for board b (or None)."""
    cur = b.current_bits()
    mask = b.mask
    opp = cur ^ mask
    possible = (mask + BOTTOM_MASK) & BOARD_MASK
    mine = winning_cells_mask(cur, mask) & possible
    theirs = winning_cells_mask(opp, mask) & possible
    if mine:
        return "win1", _cols(mine)
    if theirs:
        c = _cols(theirs)
        return ("block", c) if len(c) == 1 else (None, None)
    r3 = search(b, 3)
    if r3.score == WIN - 3:
        return "win2", None
    r5 = search(b, 5)
    if r5.score == WIN - 5:
        return "win3", None
    return None, None


def generate(per_cat, seed, timeout):
    from tools.solver import Solver, SolverError
    from engine.agents import MinimaxAgent
    rng = random.Random(seed)
    bot = MinimaxAgent(2)
    chosen = {c: [] for c in CATS}
    seen = set()
    games = 0
    with Solver(timeout=timeout) as solver:
        while any(len(v) < per_cat for v in chosen.values()) and games < 5000:
            games += 1
            b = Board()
            cands = []
            while not b.is_terminal():
                if 14 <= b.moves_played <= 34:
                    cands.append(b.to_move_string())
                col = rng.choice(b.legal_moves()) if rng.random() < 0.5 else bot.choose(b)
                b.drop(col)
            # at most one position per category per game, for variety
            rng.shuffle(cands)
            used = set()
            for mv in cands:
                pos = Board.from_moves(mv)
                cat, cols = classify(pos)
                if cat is None or cat in used or len(chosen[cat]) >= per_cat:
                    continue
                k = canonical_key(pos)
                if k in seen:
                    continue
                try:
                    sc = solver.solve_all(mv)
                except SolverError:
                    continue
                ply = len(mv)
                best = max(x for x in sc if x is not None)
                opt = [c for c in range(7) if sc[c] == best]
                target = {"win1": 21, "win2": 20, "win3": 19}.get(cat)
                if cat == "block":
                    ok = opt == cols
                    sol = cols
                else:
                    ok = best == target - ply // 2
                    sol = opt
                    if cat == "win1":
                        ok = ok and set(opt) == set(cols)
                if not ok:
                    log("  solver rejected %s candidate %s (scores %s)" % (cat, mv, sc))
                    continue
                seen.add(k)
                used.add(cat)
                chosen[cat].append({
                    "id": "%s_%02d" % (cat, len(chosen[cat]) + 1), "category": cat,
                    "moves": mv, "ply": ply, "to_move": "P1" if ply % 2 == 0 else "P2",
                    "solution_cols": sol, "solver_scores": sc,
                    "winning_cols": [c for c in range(7) if sc[c] is not None and sc[c] > 0]})
            if games % 50 == 0:
                log("  %d games, found %s" % (games, {c: len(v) for c, v in chosen.items()}))
    suite = [p for c in CATS for p in chosen[c]]
    return suite, games


def _evaluate(task):
    p, depth = task
    b = Board.from_moves(p["moves"])
    r = search(b, depth)
    return {"id": p["id"], "category": p["category"], "ply": p["ply"], "depth": depth,
            "chosen": r.col, "solved": int(r.col in p["solution_cols"]),
            "still_winning": int(r.col in p["winning_cols"]) if p["category"] != "block"
            else int(r.col in p["solution_cols"]),
            "nodes": r.nodes, "ms": round(r.ms, 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--per-category", type=int, default=13)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--depths", default="1-8", help="e.g. 1-8 or 1,2,4")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--solver-timeout", type=float, default=30.0)
    args = ap.parse_args(argv)
    ensure_dirs()

    if args.regenerate or not os.path.exists(SUITE):
        if not solver_available():
            sys.exit("tactics_positions.json missing and solver binary not found "
                     "(build tools/pons/c4solver) - cannot generate verified labels.")
        log("generating tactics suite (%d per category) ..." % args.per_category)
        suite, games = generate(args.per_category, args.seed, args.solver_timeout)
        with open(SUITE, "w", encoding="utf-8") as f:
            json.dump({"description": __doc__.split("Outputs:")[0].strip(),
                       "seed": args.seed, "games_sampled": games, "positions": suite}, f, indent=1)
        log("wrote %s (%d positions from %d games)" % (SUITE, len(suite), games))
    with open(SUITE, encoding="utf-8") as f:
        suite = json.load(f)["positions"]

    if "-" in args.depths:
        lo, hi = args.depths.split("-")
        depths = list(range(int(lo), int(hi) + 1))
    else:
        depths = [int(x) for x in args.depths.split(",")]
    rows = pmap(_evaluate, [(p, d) for p in suite for d in depths], args.workers, label="tactics")
    write_csv(os.path.join(RESULTS, "tactics_results.csv"), rows)

    summary = []
    for d in depths:
        for cat in CATS + ("all",):
            sub = [r for r in rows if r["depth"] == d and (cat == "all" or r["category"] == cat)]
            summary.append({"agent": "d%d" % d, "depth": d, "category": cat, "n": len(sub),
                            "solved": sum(r["solved"] for r in sub),
                            "share_solved": round(sum(r["solved"] for r in sub) / len(sub), 4),
                            "share_still_winning": round(sum(r["still_winning"] for r in sub) / len(sub), 4)})
    # slider levels with random moves: expected share = (1-eps)*solved(depth) + eps*|sol|/|legal|
    pos_by_id = {p["id"]: p for p in suite}
    for lvl in (1, 2, 3):
        depth, eps = LEVELS[lvl]
        for cat in CATS + ("all",):
            sub = [r for r in rows if r["depth"] == depth and (cat == "all" or r["category"] == cat)]
            if not sub:
                continue
            exp = 0.0
            for r in sub:
                p = pos_by_id[r["id"]]
                legal = len(Board.from_moves(p["moves"]).legal_moves())
                exp += (1 - eps) * r["solved"] + eps * len(p["solution_cols"]) / legal
            summary.append({"agent": "L%d" % lvl, "depth": depth, "category": cat, "n": len(sub),
                            "solved": round(exp, 2), "share_solved": round(exp / len(sub), 4),
                            "share_still_winning": ""})
    write_csv(os.path.join(RESULTS, "tactics_summary.csv"), summary)
    log("\nShare solved (strict):")
    log("agent " + " ".join("%7s" % c for c in CATS + ("all",)))
    for ag in ["d%d" % d for d in depths] + ["L1", "L2", "L3"]:
        vals = {s["category"]: s["share_solved"] for s in summary if s["agent"] == ag}
        if vals:
            log("%5s " % ag + " ".join("%7.2f" % vals[c] for c in CATS + ("all",)))


if __name__ == "__main__":
    main()
