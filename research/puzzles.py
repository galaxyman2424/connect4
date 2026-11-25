"""Board puzzles for the Research page ("Nets vs Pons" tab): tactical positions sorted by
type (win now, block, forced win in 2 / 3 / 4+, only drawing move, real arena divergences),
with every showcase bot's chosen move next to Pons's per-column scores.

    python -m research.puzzles                       # 10 positions per type, all showcase bots
    python -m research.puzzles --per 16 --seed 1
    python -m research.puzzles --bots pons_perfect,nn_sup_6x64_mcts200 --solver ~/pons/c4solver

Positions come from data/accuracy_positions_60k_raw.csv (already solved: 7 Pons scores per
position) and from the arena divergences in report.json (solved here with tools/pons/c4solver).
Each bot is built exactly as the arena builds it (arena.registry), so its move here is the move
it would play in a tournament game from that position.  Bots that cannot be built on this
machine (e.g. bruneton_az without torch) are listed under "unavailable" and skipped.

Writes experiments/results/research/puzzles.json.  Moves in the file are 0-indexed columns.
"""
import argparse
import csv
import datetime
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)                      # model paths in the registry are relative to the repo

from engine.board import Board, CENTER_ORDER  # noqa: E402

RAW = os.path.join(ROOT, "data", "accuracy_positions_60k_raw.csv")
REPORT = os.path.join(ROOT, "experiments", "results", "research", "report.json")
OUT = os.path.join(ROOT, "experiments", "results", "research", "puzzles.json")

# the bots of the last arena (top8_o100_mcts_fixed); label, family
SHOWCASE = [
    ("pons_perfect", "Pons solver", "perfect play (reference)"),
    ("nn_ens3_score_8x128_mcts200", "3-net score ensemble + MCTS200", "ours: score head, solver labels"),
    ("nn_score_8x128_s0_r1_mcts200", "score 8x128 s0 + MCTS200", "ours: score head, solver labels"),
    ("nn_ens3_score_8x128_policy", "3-net score ensemble (policy only)", "ours: score head, no search"),
    ("nn_sup_6x64_mcts200", "sup 6x64 + MCTS200", "ours: supervised (solver labels)"),
    ("nn_az_6x64_mcts200", "az 6x64 + MCTS200", "ours: AlphaZero self-play"),
    ("nn_ppo_from_sup_policy", "ppo_from_sup (policy only)", "ours: PPO from supervised, no search"),
    ("nn_ppo_6x64_mcts200", "ppo 6x64 + MCTS200", "ours: PPO self-play"),
    ("nn_distill_az_2x32_policy", "distill_az 2x32 (policy only)", "ours: 50k-param distilled net"),
    ("bruneton_az", "Bruneton AlphaZero", "open source AlphaZero"),
]
EXTRA_ALIASES = {
    "nn_ens3_score_8x128_mcts200": "nn:runs/score_8x128_s0_r1/best.npz+runs/score_8x128_s1_r1/best.npz"
                                   "+runs/score_8x128_s2_r1/best.npz:mcts:200",
    "nn_score_8x128_s0_r1_mcts200": "nn:runs/score_8x128_s0_r1/best.npz:mcts:200",
}

CATEGORIES = [
    ("win1", "Win now", "The side to move has a four-in-a-row available this move."),
    ("block", "Block", "No win of its own; the opponent threatens to win next move in exactly one column, "
                       "and blocking it keeps the game (draw or better)."),
    ("win2", "Win in 2", "No immediate tactics; exactly one column wins, and the win comes on the mover's 2nd move."),
    ("win3", "Win in 3", "As above, win on the mover's 3rd move."),
    ("win4", "Win in 4-6", "As above, win on the mover's 4th to 6th move (deeper forced wins)."),
    ("hold", "Only drawing move", "No immediate tactics, no winning move, and exactly one column keeps the draw; "
                                  "every other column loses."),
    ("arena", "Arena divergences", "Real positions from the last arena where Pons and the 3-net ensemble + MCTS, "
                                   "facing the same opponent from the same opening, first chose differently."),
]
_RANK = {c: i for i, c in enumerate(CENTER_ORDER)}


def wins_now(b, c):
    b.drop(c)
    w = b.check_win()
    b.undo()
    return bool(w)


def tactics(b):
    """own_wins: columns that win now; safe: columns after which the opponent cannot win at once;
    threats: columns where the opponent would win if it could play there now."""
    legal = b.legal_moves()
    own = [c for c in legal if wins_now(b, c)]
    safe = []
    for c in legal:
        b.drop(c)
        lost = (not b.check_win()) and any(wins_now(b, d) for d in b.legal_moves())
        b.undo()
        if not lost:
            safe.append(c)
    threats = []
    for c in legal:                        # opponent threat at c: play elsewhere, opponent plays c and wins
        others = [x for x in legal if x != c]
        for x in others:
            b.drop(x)
            hit = (not b.check_win()) and b.can_play(c) and wins_now(b, c)
            b.undo()
            if hit:
                threats.append(c)
                break
    return own, safe, threats


def classify(moves, scores):
    b = Board.from_moves(moves)
    n = b.moves_played
    legal = [c for c in range(7) if scores[c] is not None]
    best = max(scores[c] for c in legal)
    own, safe, threats = tactics(b)
    if own:
        return "win1"
    if len(threats) == 1 and safe == threats and scores[threats[0]] >= 0:
        return "block"
    if threats or len(safe) < 2:
        return None
    if best > 0 and sum(1 for c in legal if scores[c] > 0) == 1:
        k = 22 - n // 2 - best                # own moves until the win (1 = this move); see tools/solver.py
        return {2: "win2", 3: "win3", 4: "win4", 5: "win4", 6: "win4"}.get(k)
    if best == 0 and sum(1 for c in legal if scores[c] >= 0) == 1:
        return "hold"
    return None


def load_raw():
    out = []
    with open(RAW, newline="") as f:
        for r in csv.DictReader(f):
            out.append((r["moves"], json.loads(r["scores"])))
    return out


def pick(cands, k, rng):
    """k positions spread over game phase: sort by ply, cut into k bins, one random pick per bin."""
    cands = sorted(set(cands), key=lambda m: (len(m), m))
    if len(cands) <= k:
        return cands
    out = []
    for i in range(k):
        lo, hi = i * len(cands) // k, (i + 1) * len(cands) // k
        out.append(cands[rng.randrange(lo, hi)])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=10, help="positions per category (default 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bots", default=",".join(b[0] for b in SHOWCASE))
    ap.add_argument("--solver", default=None, help="c4solver binary (default tools/pons/c4solver)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--budget", type=float, default=0,
                    help="stop after this many seconds and save progress; re-run to continue (0 = no limit)")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)

    raw = load_raw()
    score_of = dict(raw)
    by_cat = {c[0]: [] for c in CATEGORIES}
    for mv, sc in raw:
        cat = classify(mv, sc)
        if cat:
            by_cat[cat].append(mv)
    counts = {k: len(v) for k, v in by_cat.items()}

    chosen = []                                    # (cat, moves, extra)
    for cat, _, _ in CATEGORIES:
        if cat == "arena":
            continue
        for mv in pick(by_cat[cat], a.per, rng):
            chosen.append((cat, mv, {}))

    # arena divergences (need the solver)
    from tools.solver import Solver
    kw = {"timeout": 120}
    if a.solver:
        kw["binary"] = os.path.expanduser(a.solver)
    solver = None
    try:
        with open(REPORT, encoding="utf-8") as f:
            ex = json.load(f)["tables"]["examples"]
        solver = Solver(**kw)
        for r in ex:
            mv = r["position"].replace(" ", "")
            score_of[mv] = solver.solve_all(mv)
            chosen.append(("arena", mv, {"opponent": r["opponent"], "state": r["state"],
                                         "arena_pons_move": r["pons_move"] - 1, "arena_net_move": r["net_move"] - 1,
                                         "pons_result": r["pons_score"], "net_result": r["net_score"]}))
        counts["arena"] = len(ex)
    except Exception as e:  # noqa: BLE001
        print("arena divergences skipped: %s" % e, file=sys.stderr)

    # bots
    from arena import registry
    registry.ALIASES.update(EXTRA_ALIASES)
    names = [x for x in a.bots.split(",") if x]
    meta = {b[0]: b for b in SHOWCASE}
    bots, unavailable = [], []
    for nm in names:
        if nm == "pons_perfect":           # same rule as arena.adapters.PonsSolver, from the stored scores
            bots.append((nm, None))
            continue
        try:
            ent = registry.resolve([nm])
            key = list(ent)[0]
            bots.append((nm, registry.build(ent[key])))
            print("built %s" % nm, file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            unavailable.append({"id": nm, "label": meta.get(nm, (nm, nm))[1], "reason": str(e)[:200]})
            print("UNAVAILABLE %s: %s" % (nm, e), file=sys.stderr)

    # resume: picks already computed by an earlier (time-limited) run with the same positions
    part = a.out + ".progress"
    done = {}
    if os.path.exists(part):
        try:
            with open(part, encoding="utf-8") as f:
                prog = json.load(f)
            if prog.get("seed") == a.seed and prog.get("per") == a.per:
                done = prog.get("picks", {})
        except (OSError, ValueError):
            done = {}
    t_start = time.time()
    for cat, mv, extra in chosen:
        if a.budget and time.time() - t_start > a.budget:
            break
        b = Board.from_moves(mv)
        row = done.setdefault(cat + "|" + mv, {})
        for nm, bot in bots:
            if nm in row:
                continue
            if bot is None:
                sc = score_of[mv]
                best = max(x for x in sc if x is not None)
                row[nm] = [min((c for c in range(7) if sc[c] == best), key=lambda c: _RANK[c]), 0.0]
                continue
            t0 = time.perf_counter()
            row[nm] = [int(bot.choose(b.copy())), round(time.perf_counter() - t0, 4)]
        with open(part, "w", encoding="utf-8") as f:
            json.dump({"seed": a.seed, "per": a.per, "picks": done}, f)
        print("[%d/%d] %s %s" % (sum(1 for x in done.values() if len(x) >= len(bots)), len(chosen), cat, mv), file=sys.stderr)
    missing = [k for k in (c + "|" + m for c, m, _ in chosen) if len(done.get(k, {})) < len(bots)]
    if missing:
        print("%d positions still to do: run the same command again to continue" % len(missing))
        return

    positions = []
    t_bot = {nm: 0.0 for nm, _ in bots}
    for i, (cat, mv, extra) in enumerate(chosen):
        sc = score_of[mv]
        legal = [c for c in range(7) if sc[c] is not None]
        best = max(sc[c] for c in legal)
        b = Board.from_moves(mv)
        picks = {}
        for nm, _ in bots:
            col, sec = done[cat + "|" + mv][nm]
            t_bot[nm] += sec
            picks[nm] = col
        own, safe, threats = tactics(b)
        positions.append(dict({"id": "%s%02d" % (cat, sum(1 for p in positions if p["cat"] == cat)),
                               "cat": cat, "moves": mv, "ply": len(mv), "to_move": 1 if len(mv) % 2 == 0 else 2,
                               "scores": sc, "best": best, "optimal": [c for c in legal if sc[c] == best],
                               "keeps": [c for c in legal if (sc[c] > 0) == (best > 0) and (sc[c] >= 0) == (best >= 0)],
                               "threats": threats, "picks": picks}, **extra))
        print("[%d/%d] %s %s" % (i + 1, len(chosen), cat, mv), file=sys.stderr)

    out = {"id": "puzzles", "title": "Board puzzles: every showcase bot vs Pons",
           "generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "source": "data/accuracy_positions_60k_raw.csv + report.json examples", "seed": a.seed,
           "categories": [{"id": c, "label": l, "rule": r, "candidates": counts.get(c, 0)} for c, l, r in CATEGORIES],
           "bots": [{"id": nm, "label": meta.get(nm, (nm, nm, ""))[1], "family": meta.get(nm, (nm, nm, ""))[2],
                     "sec_total": round(t_bot[nm], 2)} for nm, _ in bots],
           "unavailable": unavailable,
           "notes": ["Scores are Pons scores for the side to move after playing that column: >0 win (bigger = faster), "
                     "0 draw, <0 loss. Columns are 0-indexed here and shown 1-7 on the page.",
                     "optimal = columns with the best score; keeps = columns that keep the same result "
                     "(win stays a win, draw stays a draw) even if slower.",
                     "Positions are a ply-stratified random sample (seed %d) of each type, not hand-picked." % a.seed],
           "positions": positions}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    os.replace(tmp, a.out)
    print("wrote %s (%d positions, %d bots, %d unavailable)" % (a.out, len(positions), len(bots), len(unavailable)))
    if solver:
        solver.close()


if __name__ == "__main__":
    main()
