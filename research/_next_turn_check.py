"""One-off check: opponent's safe replies on the turn right after Pons / a net diverge."""
import json
import os
from collections import defaultdict

from .arena_analysis import _load_cache, load_games, lookup, players_of, sign
from .arena_vs_pons import ARENA, CACHE, PONS

cache = _load_cache(CACHE)
games = load_games(os.path.join(ARENA, "top8_o100_mcts_fixed", "games.csv"))
A = json.load(open(os.path.join(os.path.dirname(ARENA), "research", "arena_vs_pons.json")))
field = set(A["common_field"])
out = {}
for net in ["nn_ens3_score_8x128_mcts200", "nn_score_8x128_s0_r1_mcts200", "nn_ens3_score_8x128_policy",
            "nn_sup_6x64_mcts200"]:
    idx = {}
    for g in games:
        for me in (PONS, net):
            if me in (g["a"], g["b"]):
                x = g["b"] if g["a"] == me else g["a"]
                if x in field and x != net:
                    idx[(me, x, g["opening_id"], players_of(g)[0] == me)] = g
    acc = defaultdict(lambda: {"n": 0, "safe_pons": 0, "safe_net": 0, "only_pons": 0, "only_net": 0,
                               "err_pons": 0, "err_net": 0})
    for (me, x, oid, f), gp in idx.items():
        if me != PONS or (net, x, oid, f) not in idx:
            continue
        gn = idx[(net, x, oid, f)]
        a, b = gp["moves"], gn["moves"]
        d = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        if d is None:
            continue
        sc = lookup(cache, a[:d])
        st = max(sign(v) for v in sc if v is not None)
        if st == 1:
            continue
        ok = True
        rec = {}
        for key, ms, g in (("pons", a, gp), ("net", b, gn)):
            if len(ms) <= d + 1 or g["plies"] <= d + 1:
                ok = False
                break
            s2 = lookup(cache, ms[:d + 1])
            if s2 is None:
                ok = False
                break
            signs = [sign(v) for v in s2 if v is not None]
            best = max(signs)
            rec[key] = (signs.count(best), sign(s2[int(ms[d + 1]) - 1]) < best)
        if not ok:
            continue
        r = acc["lost" if st == -1 else "drawn"]
        r["n"] += 1
        for key in ("pons", "net"):
            r["safe_" + key] += rec[key][0]
            r["only_" + key] += rec[key][0] == 1
            r["err_" + key] += rec[key][1]
    out[net] = {k: {"n": v["n"], "mean_safe_pons": round(v["safe_pons"] / v["n"], 3),
                    "mean_safe_net": round(v["safe_net"] / v["n"], 3),
                    "only_move_pons": round(v["only_pons"] / v["n"], 4), "only_move_net": round(v["only_net"] / v["n"], 4),
                    "next_move_err_pons": round(v["err_pons"] / v["n"], 4), "next_move_err_net": round(v["err_net"] / v["n"], 4)}
                for k, v in acc.items()}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(ARENA), "research", "arena_next_turn.json"), "w"), indent=1)
