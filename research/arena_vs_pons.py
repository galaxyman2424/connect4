"""Why do the score-head nets + MCTS out-Elo the perfect solver?  (2026-11-24)

    python -m research.arena_vs_pons                         # top8_o100_mcts_fixed
    python -m research.arena_vs_pons --run top15_o100_ens --workers 8

Every position in which a bot moved is solved with Pons' solver (weak mode: win/draw/loss
per column), so each game becomes a value trajectory.  For bot B:

    score - par  =  (gifts: value B gained from opponent mistakes)
                  - (own losses: value B threw away itself)

where par = the result of the opening under perfect play.  Pons never makes a mistake,
so its score is exactly par + gifts.  A net can only outscore Pons against an opponent X
if X makes more mistakes against the net than against Pons, by more than the net's own
mistakes.  The script measures that and asks why:

  * per (bot, opponent): par, gifts, own losses, opponent error rate, how often the
    opponent faced an "only move" (exactly one non-losing column), mean safe moves
  * matched games: Pons vs X and net vs X from the same opening and colours.  All players
    are deterministic, so the two games are identical up to the first ply where Pons and
    the net choose differently.  At that ply the position is the same, so the difference
    in outcome is caused by that choice (and what followed).
  * weighted records: head-to-head vs Pons, Elo-predicted vs actual, Elo refit on the top
    group only, and the score against the common field.

Writes experiments/results/research/arena_vs_pons.json.  Solver cache (JSON) is kept in
~/.cache/c4_weak_solver_cache.json so reruns are instant.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from .arena_analysis import (_load_cache, bradley_terry, canon, load_games, log, lookup,  # noqa: E402
                             players_of, score_of, sign, solve_positions)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARENA = os.path.join(ROOT, "experiments", "results", "arena")
OUT = os.path.join(ROOT, "experiments", "results", "research", "arena_vs_pons.json")
CACHE = os.path.expanduser("~/.cache/c4_weak_solver_cache.json")
PONS = "pons_perfect"
VAL = {1: "won", 0: "drawn", -1: "lost"}


def r4(x):
    return None if x is None else round(float(x), 4)


def game_trace(g, cache):
    """Per ply from len(opening): (mover, best value for mover, value of chosen move,
    number of safe columns, number of legal columns).  None if any position is unsolved."""
    p1, p2 = players_of(g)
    out = []
    for i in range(len(g["opening"]), g["plies"]):
        sc = lookup(cache, g["moves"][:i])
        if sc is None:
            return None
        signs = [sign(v) for v in sc if v is not None]
        best = max(signs)
        col = int(g["moves"][i]) - 1
        out.append((p1 if i % 2 == 0 else p2, best, sign(sc[col]), signs.count(best), len(signs), i))
    return out


def per_game(g, tr, bot):
    """Value accounting for `bot` in game g."""
    v0 = tr[0][1] if tr[0][0] == bot else -tr[0][1]
    gifts = own = 0.0
    opp_turns = opp_err = opp_only = opp_safe = 0
    own_turns = own_err = 0
    state_turns = {1: 0, 0: 0, -1: 0}
    for mover, best, ch, nsafe, nlegal, _ in tr:
        drop = best - ch
        if mover == bot:
            own_turns += 1
            state_turns[best] += 1
            if drop > 0:
                own_err += 1
                own += drop / 2
        else:
            if best >= 0:                       # the opponent still had something to lose
                opp_turns += 1
                opp_safe += nsafe
                opp_only += nsafe == 1
                if drop > 0:
                    opp_err += 1
            if drop > 0:
                gifts += drop / 2
    par = (v0 + 1) / 2
    s = score_of(g, bot)
    assert abs(par + gifts - own - s) < 1e-9, (g, bot, par, gifts, own, s)
    return {"par": par, "gifts": gifts, "own": own, "score": s, "v0": v0,
            "opp_turns": opp_turns, "opp_err": opp_err, "opp_only": opp_only, "opp_safe": opp_safe,
            "own_turns": own_turns, "own_err": own_err, "state_turns": state_turns}


def accumulate(rows):
    n = len(rows)
    t = lambda k: sum(r[k] for r in rows)  # noqa: E731
    ot = max(t("opp_turns"), 1)
    return {"games": n, "score": r4(t("score") / n), "par": r4(t("par") / n),
            "gifts": r4(t("gifts") / n), "own_losses": r4(t("own") / n),
            "above_par": r4((t("score") - t("par")) / n),
            "opp_error_rate": r4(t("opp_err") / ot), "opp_errors": t("opp_err"), "opp_turns": t("opp_turns"),
            "opp_only_move_share": r4(t("opp_only") / ot), "opp_mean_safe_moves": r4(t("opp_safe") / ot),
            "own_error_rate": r4(t("own_err") / max(t("own_turns"), 1)), "own_errors": t("own_err"),
            "start_won": sum(r["v0"] == 1 for r in rows), "start_drawn": sum(r["v0"] == 0 for r in rows),
            "start_lost": sum(r["v0"] == -1 for r in rows)}


def matched(games, traces, strong, other, opponents):
    """Compare strong vs X with other vs X, same opening + colours.  Returns summary dict."""
    idx = {}
    for gi, g in enumerate(games):
        for me in (strong, other):
            if me in (g["a"], g["b"]):
                x = g["b"] if g["a"] == me else g["a"]
                if x in opponents:
                    p1, _ = players_of(g)
                    idx[(me, x, g["opening_id"], p1 == me)] = gi
    by_state = defaultdict(lambda: {"n": 0, "other_better": 0, "same": 0, "other_worse": 0,
                                    "other_move_also_safe": 0, "score_diff_sum": 0.0,
                                    "opp_err_after_strong": 0, "opp_turns_after_strong": 0,
                                    "opp_err_after_other": 0, "opp_turns_after_other": 0,
                                    "only_after_strong": 0, "only_after_other": 0})
    examples = []
    identical = 0
    pairs = 0
    for (me, x, oid, first), gi in idx.items():
        if me != strong or (other, x, oid, first) not in idx:
            continue
        gs, go = games[gi], games[idx[(other, x, oid, first)]]
        ts, to = traces[gi], traces[idx[(other, x, oid, first)]]
        if ts is None or to is None:
            continue
        pairs += 1
        ms, mo = gs["moves"], go["moves"]
        d = next((i for i in range(min(len(ms), len(mo))) if ms[i] != mo[i]), None)
        if d is None:
            identical += 1
            continue
        # who diverged?  the mover at ply d must be the strong side (X is deterministic)
        p1 = players_of(gs)[0]
        mover_is_me = (d % 2 == 0) == (p1 == strong)
        if not mover_is_me:
            by_state["opponent_diverged"]["n"] += 1   # X not deterministic in this position
            continue
        sc = lookup(_CACHE, ms[:d])
        best = max(sign(v) for v in sc if v is not None)
        cs, co = sign(sc[int(ms[d]) - 1]), sign(sc[int(mo[d]) - 1])
        b = by_state[VAL[best]]
        b["n"] += 1
        b["other_move_also_safe"] += co == best
        s_s, s_o = score_of(gs, strong), score_of(go, other)
        b["score_diff_sum"] += s_o - s_s
        b["other_better" if s_o > s_s else "other_worse" if s_o < s_s else "same"] += 1
        for tr, key in ((ts, "strong"), (to, "other")):
            for mover, bst, ch, nsafe, _, i in tr:
                if i > d and mover == x and bst >= 0:
                    b["opp_turns_after_" + key] += 1
                    b["opp_err_after_" + key] += ch < bst
                    b["only_after_" + key] += nsafe == 1
        if s_o != s_s and len(examples) < 40:
            examples.append({"opponent": x, "opening": gs["opening"], "strong_first": first, "diverge_ply": d,
                             "position": ms[:d], "state": VAL[best], "strong_move": int(ms[d]),
                             "other_move": int(mo[d]), "other_move_value": VAL[co], "strong_move_value": VAL[cs],
                             "strong_game": ms, "other_game": mo, "strong_score": s_s, "other_score": s_o})
    out = {}
    for k, b in by_state.items():
        n = max(b["n"], 1)
        out[k] = dict(b)
        if k != "opponent_diverged":
            out[k]["mean_score_diff"] = r4(b["score_diff_sum"] / n)
            out[k]["opp_err_rate_after_strong"] = r4(b["opp_err_after_strong"] / max(b["opp_turns_after_strong"], 1))
            out[k]["opp_err_rate_after_other"] = r4(b["opp_err_after_other"] / max(b["opp_turns_after_other"], 1))
            out[k]["only_move_share_after_strong"] = r4(b["only_after_strong"] / max(b["opp_turns_after_strong"], 1))
            out[k]["only_move_share_after_other"] = r4(b["only_after_other"] / max(b["opp_turns_after_other"], 1))
    return {"pairs": pairs, "identical_games": identical, "by_state_at_divergence": out, "examples": examples}


def elo_fit(games, players, anchor, anchor_value):
    ix = {p: i for i, p in enumerate(players)}
    k = len(players)
    s = np.zeros((k, k))
    n = np.zeros((k, k))
    for g in games:
        if g["a"] in ix and g["b"] in ix:
            i, j = ix[g["a"]], ix[g["b"]]
            s[i, j] += g["score_a"]
            s[j, i] += 1 - g["score_a"]
            n[i, j] += 1
            n[j, i] += 1
    e = bradley_terry(s, n, anchor=ix[anchor], anchor_value=anchor_value)
    return {p: round(float(e[ix[p]]), 1) for p in players}


def h2h(games, x, y):
    w = d = l = 0
    for g in games:
        if {g["a"], g["b"]} == {x, y}:
            s = score_of(g, x)
            w += s == 1
            d += s == 0.5
            l += s == 0
    n = w + d + l
    p = float(sstats.binomtest(w, w + l).pvalue) if w + l else None
    return {"w": w, "d": d, "l": l, "score": r4((w + d / 2) / n) if n else None, "sign_test_p": r4(p)}


_CACHE = {}


def main():
    global _CACHE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="top8_o100_mcts_fixed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    t0 = time.time()
    d = os.path.join(ARENA, a.run)
    games = [g for g in load_games(os.path.join(d, "games.csv")) if not g["forfeit"]]
    summ = json.load(open(os.path.join(d, "summary.json")))
    bots = summ["bots"]
    cache = _load_cache(CACHE)
    pos = {g["moves"][:i] for g in games for i in range(len(g["opening"]), g["plies"])}
    sv = solve_positions(sorted(pos), cache, CACHE, workers=a.workers, timeout=300.0)
    _CACHE = cache
    log("solved: %s" % sv)
    traces = [game_trace(g, cache) for g in games]
    log("traced %d / %d games" % (sum(t is not None for t in traces), len(games)))

    # ---- per bot / per (bot, opponent) accounting
    rows = defaultdict(list)
    for g, tr in zip(games, traces):
        if tr is None:
            continue
        for b in (g["a"], g["b"]):
            x = g["b"] if g["a"] == b else g["a"]
            r = per_game(g, tr, b)
            rows[(b, x)].append(r)
            rows[(b, "*")].append(r)
    per_bot = {b: accumulate(rows[(b, "*")]) for b in bots}
    pair = {b: {x: accumulate(rows[(b, x)]) for x in bots if x != b and rows[(b, x)]} for b in bots}
    strong = [b for b in bots if b != PONS]
    # common field = everyone except Pons and the two score-net MCTS bots
    top = [PONS, "nn_ens3_score_8x128_mcts200", "nn_score_8x128_s0_r1_mcts200"]
    field = [b for b in bots if b not in top]
    common = {b: accumulate([r for x in field if x != b for r in rows[(b, x)]]) for b in bots}

    # ---- matched comparisons: Pons vs each net, against the common field
    match = {o: matched(games, traces, PONS, o, set(field) - {o}) for o in strong}

    # ---- weighted records
    elo = {b: summ["elo"][b]["elo"] for b in bots}
    rec = {}
    for b in strong:
        hh = h2h(games, b, PONS)
        de = elo[b] - elo[PONS]
        rec[b] = {"elo": elo[b], "elo_minus_pons": round(de, 1),
                  "elo_predicted_score_vs_pons": r4(1 / (1 + 10 ** (-de / 400))),
                  "actual_vs_pons": hh, "score_vs_field": common[b]["score"],
                  "pons_score_vs_field": common[PONS]["score"]}
    # Elo using only the top group (who beats whom among the strongest bots)
    top_group = top + ["nn_ens3_score_8x128_policy", "nn_sup_6x64_mcts200", "nn_az_6x64_mcts200"]
    elo_top = elo_fit(games, top_group, PONS, elo[PONS])
    # Elo with the weak half removed one by one isn't needed; also a pons-only scale:
    pons_scale = {}
    for b in strong:
        s = rec[b]["actual_vs_pons"]["score"]
        s = min(max(s, 1e-3), 1 - 1e-3)
        pons_scale[b] = round(elo[PONS] + 400 * math.log10(s / (1 - s)), 1)
    # strength-weighted score: each opponent weighted by its own score vs the field
    strength = {x: per_bot[x]["score"] for x in bots}
    weighted = {}
    for b in bots:
        num = den = 0.0
        for x in bots:
            if x == b or not rows[(b, x)]:
                continue
            w = strength[x]
            num += w * pair[b][x]["score"]
            den += w
        weighted[b] = r4(num / den)

    R = {"run": a.run, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "games": len(games),
         "solver": sv, "positions": len(pos), "bots": bots, "elo": elo,
         "per_bot": per_bot, "pair": pair, "common_field": field, "vs_common_field": common,
         "matched_vs_pons": match, "records": rec, "elo_top_group_only": elo_top,
         "elo_from_h2h_vs_pons_only": pons_scale, "strength_weighted_score": weighted,
         "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    log("wrote %s in %.0fs" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
