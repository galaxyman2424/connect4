"""Analysis of the existing arena tournament data (no new games are played).

    python -m research.arena_analysis                       # full run (~15-25 min the first time:
                                                            #   solves every position of every game)
    python -m research.arena_analysis --no-solver           # skip the solver-based parts (fast)
    python -m research.arena_analysis --solver-workers 3 --cache /tmp/c4_arena_solver_cache.json

Inputs (read-only)
  experiments/results/arena/part3/games.csv, summary.json     the 20-bot round robin (7,599 games)
  experiments/results/methods/main/summary.json               per-method table (rl/compare.py)
  runs/<run>/config.json, cost.json, eval.json, metrics.jsonl
  nn_data/solver_300k.json                                    cost of the solver labels
  runs/arena_part3.log                                        tournament wall-clock

Outputs
  experiments/results/research/arena_analysis.json  (schema: see build() / the module docstring of
                                                     the keys written there)
  experiments/results/research/roster.json          top-15 roster for the next phase

Elo method = arena/tournament.py + experiments/stats.py, re-implemented here (arena/ is being
rewritten): maximum-likelihood Bradley-Terry on the Elo scale (scipy L-BFGS, ridge 1e-6), one
player anchored, 95 % CIs from a cluster bootstrap (cluster = the two colour-swapped games of
one opening in one pairing, resampled within each pairing), 500 resamples, seed 0.

Solver: Pascal Pons' c4solver via tools/solver.py, weak mode (-w, only win/draw/loss is
exact), per-column analysis (-a), opening book tools/pons/7x6.book.  Positions are de-duplicated
up to left-right mirroring and cached in --cache (outside the repo).
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402
from scipy import stats as sstats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PART3 = os.path.join(ROOT, "experiments", "results", "arena", "part3")
METHODS = os.path.join(ROOT, "experiments", "results", "methods", "main", "summary.json")
OUT_DIR = os.path.join(ROOT, "experiments", "results", "research")
RUNS = os.path.join(ROOT, "runs")
ELO_SCALE = 400.0 / math.log(10.0)
DEFAULT_CACHE = "/tmp/c4_arena_solver_cache.json"

TOP15 = ["pons_perfect", "nn_sup_6x64_mcts200", "nn_az_6x64_mcts200", "nn_ppo_from_sup_policy",
         "bruneton_az", "nn_ppo_6x64_mcts200", "nn_distill_az_2x32_policy", "nn_distill_ens_2x32_policy",
         "alfo_mcts", "ours_minimax_d8", "evo_evo_d2_d4", "ours_minimax_d4", "galli_minimax",
         "nn_ppo_6x64_policy", "nn_dqn_6x64_policy"]
DROPPED = {"ntuple_td_ntuple_d4": "rank 16 of 20 (part3 Elo 1275); below the top 15",
           "easyai_negamax": "rank 17; win/loss-only negamax depth 5, no positional heuristic",
           "kaggle_negamax": "rank 18; was the Elo anchor (1000) of part3 - the top-15 refit is anchored at "
                             "ours_minimax_d4 = 1409 instead",
           "ntuple_td_ntuple_d0": "rank 19; greedy n-tuple, no search",
           "plkmo_az": "rank 20 (score 4.2 %, lost 727 of 760 games, also to kaggle_negamax). A 5.6 M-param "
                       "AlphaZero net with 777 reads should not be this weak: weights ('iter8') or the adapter "
                       "are suspect - not investigated, so it is not evidence about AlphaZero nets"}
NEURAL = ["nn_az_6x64_mcts200", "nn_ppo_from_sup_policy", "bruneton_az", "nn_ppo_6x64_mcts200",
          "nn_distill_az_2x32_policy", "nn_distill_ens_2x32_policy", "nn_ppo_6x64_policy",
          "nn_dqn_6x64_policy", "plkmo_az"]
BEST = "nn_sup_6x64_mcts200"

# arena bot -> (run, search) row of the methods table
METHOD_MAP = {
    "nn_sup_6x64_mcts200": ("sup_6x64", "mcts200"), "nn_az_6x64_mcts200": ("az_6x64", "mcts200"),
    "nn_ppo_6x64_mcts200": ("ppo_6x64", "mcts200"), "nn_ppo_6x64_policy": ("ppo_6x64", "none"),
    "nn_ppo_from_sup_policy": ("ppo_from_sup", "none"), "nn_dqn_6x64_policy": ("dqn_6x64", "none"),
    "nn_distill_az_2x32_policy": ("distill_az_2x32", "none"),
    "nn_distill_ens_2x32_policy": ("distill_ens_2x32", "none"),
    "evo_evo_d2_d4": ("evo_d2", "minimax d4"), "ntuple_td_ntuple_d0": ("td_ntuple", "none"),
    "ntuple_td_ntuple_d4": ("td_ntuple", "minimax d4"), "ours_minimax_d4": (None, "minimax d4"),
}
# parameter counts: ours from runs/<run>/config.json; open-source nets counted from their
# checkpoints (trainable float tensors, BatchNorm running stats excluded) with torch.
PARAMS_OPEN = {"bruneton_az": 1202866, "plkmo_az": 5634993}
FAMILY = {
    "pons_perfect": "exact solver",
    "nn_sup_6x64_mcts200": "supervised NN (solver labels) + MCTS",
    "nn_az_6x64_mcts200": "self-play NN (AlphaZero) + MCTS",
    "nn_ppo_from_sup_policy": "RL NN (PPO, initialised from supervised), no search",
    "bruneton_az": "self-play NN (AlphaZero, open source) + MCTS",
    "nn_ppo_6x64_mcts200": "RL NN (PPO self-play) + MCTS",
    "nn_distill_az_2x32_policy": "distilled NN (teacher: az_6x64 + MCTS100), no search",
    "nn_distill_ens_2x32_policy": "distilled NN (teacher ensemble), no search",
    "alfo_mcts": "MCTS with random roll-outs (open source)",
    "ours_minimax_d8": "minimax + hand heuristic",
    "evo_evo_d2_d4": "minimax + evolved heuristic (CMA-ES)",
    "ours_minimax_d4": "minimax + hand heuristic",
    "galli_minimax": "minimax + hand heuristic (open source)",
    "nn_ppo_6x64_policy": "RL NN (PPO self-play), no search",
    "nn_dqn_6x64_policy": "RL NN (DQN), no search",
    "ntuple_td_ntuple_d4": "TD(lambda) n-tuple + minimax",
    "easyai_negamax": "negamax, win/loss only (open source)",
    "kaggle_negamax": "negamax, no pruning (open source)",
    "ntuple_td_ntuple_d0": "TD(lambda) n-tuple, greedy",
    "plkmo_az": "self-play NN (AlphaZero, open source) + MCTS",
}
BUDGET = {
    "pons_perfect": "exact (full search + opening book)",
    "nn_sup_6x64_mcts200": "200 PUCT simulations", "nn_az_6x64_mcts200": "200 PUCT simulations",
    "nn_ppo_6x64_mcts200": "200 PUCT simulations", "bruneton_az": "200 PUCT simulations",
    "plkmo_az": "777 PUCT reads", "alfo_mcts": "3000 UCT iterations (random roll-outs)",
    "ours_minimax_d8": "depth 8", "ours_minimax_d4": "depth 4", "evo_evo_d2_d4": "depth 4",
    "galli_minimax": "depth 5", "easyai_negamax": "depth 5", "kaggle_negamax": "depth 4 (no pruning)",
    "ntuple_td_ntuple_d4": "depth 4", "ntuple_td_ntuple_d0": "none (greedy 1-ply)",
}
PLY_BUCKETS = [(2, 7), (8, 13), (14, 19), (20, 25), (26, 31), (32, 41)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rnd(x, k=4):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return None
        return round(float(x), k)
    return x


# ---------------------------------------------------------------- data
def load_games(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append({"a": r["a"], "b": r["b"], "opening_id": int(r["opening_id"]), "opening": r["opening"],
                        "a_first": int(r["a_first"]), "winner": r["winner"], "score_a": float(r["score_a"]),
                        "plies": int(r["plies"]), "moves": r["moves"], "a_ms": float(r["a_ms"]),
                        "b_ms": float(r["b_ms"]), "a_max_ms": float(r["a_max_ms"]),
                        "b_max_ms": float(r["b_max_ms"]), "forfeit": r["forfeit"]})
    return out


def players_of(g):
    """(first player = colour 1, who made ply 1 of the opening; second player)."""
    return (g["a"], g["b"]) if g["a_first"] else (g["b"], g["a"])


def score_of(g, bot):
    return g["score_a"] if g["a"] == bot else 1.0 - g["score_a"]


def moves_by(g, bot):
    """Number of moves `bot` made in game g (opening plies are not anybody's)."""
    p1, _ = players_of(g)
    n0, n = len(g["opening"]), g["plies"]
    first = sum(1 for i in range(n0, n) if i % 2 == 0)
    return first if bot == p1 else (n - n0) - first


# ---------------------------------------------------------------- Elo (same method as experiments/stats.py)
def bradley_terry(s, n, anchor=0, anchor_value=1000.0, ridge=1e-6):
    k = s.shape[0]
    free = [i for i in range(k) if i != anchor]

    def unpack(x):
        th = np.zeros(k)
        th[free] = x
        return th

    def nll(x):
        th = unpack(x)
        d = th[:, None] - th[None, :]
        ll = s * -np.logaddexp(0, -d)
        val = -(ll.sum()) + ridge * (x ** 2).sum()
        p = 1.0 / (1.0 + np.exp(-d))
        g = (s - n * p).sum(axis=1)
        return val, -g[free] + 2 * ridge * x

    res = optimize.minimize(nll, np.zeros(k - 1), jac=True, method="L-BFGS-B")
    return anchor_value + ELO_SCALE * unpack(res.x)


def cluster_key(g):
    return "%s|%s|%s" % (min(g["a"], g["b"]), max(g["a"], g["b"]), g["opening_id"])


def elo_bootstrap(games, players, anchor_player, anchor_value=1000.0, reps=500, seed=0):
    """Returns ({player: (elo, lo, hi)}, boot matrix reps x k (columns in `players` order))."""
    idx = {p: i for i, p in enumerate(players)}
    k = len(players)
    anchor = idx[anchor_player]
    pairings = {}
    for g in games:
        if g["a"] not in idx or g["b"] not in idx:
            continue
        i, j = idx[g["a"]], idx[g["b"]]
        key = (min(i, j), max(i, j))
        sa = g["score_a"] if i < j else 1 - g["score_a"]
        pairings.setdefault(key, {}).setdefault(g.get("cluster") or cluster_key(g), []).append(sa)

    def mats(choice):
        s = np.zeros((k, k))
        n = np.zeros((k, k))
        for (i, j), clusters in pairings.items():
            vals = list(clusters.values())
            pick = choice(len(vals))
            tot = sum(sum(vals[c]) for c in pick)
            cnt = sum(len(vals[c]) for c in pick)
            s[i, j] += tot
            s[j, i] += cnt - tot
            n[i, j] += cnt
            n[j, i] += cnt
        return s, n

    s, n = mats(lambda m: range(m))
    point = bradley_terry(s, n, anchor, anchor_value)
    rng = np.random.default_rng(seed)
    boot = np.zeros((reps, k))
    for r in range(reps):
        sb, nb = mats(lambda m: rng.integers(0, m, size=m))
        boot[r] = bradley_terry(sb, nb, anchor, anchor_value)
    lo = np.percentile(boot, 2.5, axis=0) if reps else point
    hi = np.percentile(boot, 97.5, axis=0) if reps else point
    return {p: (float(point[i]), float(lo[i]), float(hi[i])) for p, i in idx.items()}, boot


def diff_ci(boot, players, x, y, point):
    i, j = players.index(x), players.index(y)
    d = boot[:, i] - boot[:, j]
    return {"a": x, "b": y, "diff": rnd(point[x][0] - point[y][0], 1),
            "lo": rnd(np.percentile(d, 2.5), 1), "hi": rnd(np.percentile(d, 97.5), 1),
            "p_a_better": rnd(float((d > 0).mean()), 4), "reps": int(len(d))}


# ---------------------------------------------------------------- head to head
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def head_to_head(games, x, y, reps=2000, seed=0):
    """W/D/L and score of x against y.  CI of the score: bootstrap over openings (each opening =
    the 2 colour-swapped games, resampled together); Wilson interval for reference; exact
    two-sided binomial (sign) test on decisive games."""
    rel = [g for g in games if {g["a"], g["b"]} == {x, y}]
    sc = [score_of(g, x) for g in rel]
    w = sum(1 for s in sc if s == 1.0)
    d = sum(1 for s in sc if s == 0.5)
    l_ = sum(1 for s in sc if s == 0.0)
    n = len(sc)
    out = {"a": x, "b": y, "games": n, "w": w, "d": d, "l": l_,
           "score": rnd(sum(sc) / n, 4) if n else None}
    if not n:
        return out
    by_op = {}
    for g, s in zip(rel, sc):
        by_op.setdefault(g["opening_id"], []).append(s)
    clusters = list(by_op.values())
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        tot = sum(sum(clusters[c]) for c in pick)
        cnt = sum(len(clusters[c]) for c in pick)
        bs.append(tot / cnt)
    out["lo"], out["hi"] = rnd(np.percentile(bs, 2.5)), rnd(np.percentile(bs, 97.5))
    wl, wh = wilson(sum(sc), n)
    out["wilson_lo"], out["wilson_hi"] = rnd(wl), rnd(wh)
    out["p_sign_test"] = rnd(float(sstats.binomtest(w, w + l_, 0.5).pvalue), 4) if w + l_ else 1.0
    s = out["score"]
    out["elo_diff_from_score"] = rnd(400 * math.log10(s / (1 - s)), 1) if 0 < s < 1 else None
    out["first"] = _colour_split(rel, x, True)
    out["second"] = _colour_split(rel, x, False)
    return out


def _colour_split(rel, x, first):
    sub = [g for g in rel if (players_of(g)[0] == x) == first]
    sc = [score_of(g, x) for g in sub]
    return {"games": len(sc), "w": sc.count(1.0), "d": sc.count(0.5), "l": sc.count(0.0),
            "score": rnd(sum(sc) / len(sc), 4) if sc else None}


# ---------------------------------------------------------------- regression
def linreg(xs, ys, reps=2000, seed=0):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = len(xs)
    out = {"n": int(n)}
    if n < 3 or np.ptp(xs) == 0:
        out.update({"slope": None, "intercept": None, "r2": None, "slope_ci": None})
        return out
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        i = rng.integers(0, n, size=n)
        if np.ptp(xs[i]) == 0:
            continue
        bs.append(np.polyfit(xs[i], ys[i], 1)[0])
    rho = sstats.spearmanr(xs, ys).statistic if n >= 3 else None
    out.update({"slope": rnd(slope, 3), "intercept": rnd(intercept, 2), "r2": rnd(r2, 4),
                "pearson_r": rnd(float(np.corrcoef(xs, ys)[0, 1]), 4), "spearman_rho": rnd(float(rho), 4),
                "slope_ci": [rnd(np.percentile(bs, 2.5), 3), rnd(np.percentile(bs, 97.5), 3)] if bs else None})
    return out


# ---------------------------------------------------------------- solver
def mirror(m):
    return "".join(str(8 - int(c)) for c in m)


def canon(m):
    mm = mirror(m)
    return (m, False) if m <= mm else (mm, True)


def lookup(cache, m):
    key, flip = canon(m)
    v = cache.get(key)
    if v is None:
        return None
    return list(reversed(v)) if flip else list(v)


def sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)


def move_error(scores, col):
    """scores: 7 solver scores for the side to move (None = full column), col 0-indexed.
    Returns (best category, chosen category) with +1 win / 0 draw / -1 loss.  The move
    changes the game-theoretic result iff chosen < best; points lost = (best - chosen) / 2."""
    best = max(sign(v) for v in scores if v is not None)
    return best, sign(scores[col])


def value_to_p1_expected(opening, scores):
    """Expected score of the FIRST player (who made ply 1) under perfect play from `opening`."""
    v = sign(max(x for x in scores if x is not None))           # for the side to move
    if len(opening) % 2 == 1:                                    # second player to move
        v = -v
    return {1: 1.0, 0: 0.5, -1: 0.0}[v]


def solve_positions(positions, cache, cache_path, workers=3, timeout=120.0):
    from tools.solver import Solver, SolverError
    todo = sorted({canon(p)[0] for p in positions} - set(cache))
    log("solver: %d positions needed, %d cached, %d to solve on %d solver processes"
        % (len({canon(p)[0] for p in positions}), len(cache), len(todo), workers))
    if not todo:
        return {"solved": 0, "failed": 0, "seconds": 0.0}
    lock = threading.Lock()
    done = [0]
    failed = []
    t0 = time.time()

    def work(chunk):
        with Solver(analyze=True, weak=True, timeout=timeout) as s:
            for p in chunk:
                try:
                    v = s.solve_all(p)
                except SolverError:
                    with lock:
                        failed.append(p)
                    continue
                with lock:
                    cache[p] = v
                    done[0] += 1
                    if done[0] % 5000 == 0:
                        log("  solved %d / %d  (%.0f s)" % (done[0], len(todo), time.time() - t0))
                        _save_cache(cache, cache_path)

    chunks = [todo[i::workers] for i in range(workers)]
    th = [threading.Thread(target=work, args=(c,)) for c in chunks]
    for t in th:
        t.start()
    for t in th:
        t.join()
    _save_cache(cache, cache_path)
    return {"solved": done[0], "failed": len(failed), "failed_examples": failed[:10],
            "seconds": round(time.time() - t0, 1)}


def _save_cache(cache, path):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, separators=(",", ":"))
    os.replace(tmp, path)


def _load_cache(path):
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------- helpers for inputs
def _json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def methods_rows():
    doc = _json(METHODS, {})
    rows = {}
    for e in doc.get("entries", []):
        rows[(e.get("run"), e.get("search"))] = e
    return doc, rows


def run_info(run):
    cfg = _json(os.path.join(RUNS, run, "config.json"), {}) or {}
    cost = (_json(os.path.join(RUNS, run, "cost.json"), {}) or {}).get("total")
    return cfg, cost


def tournament_wall_seconds():
    p = os.path.join(RUNS, "arena_part3.log")
    best = None
    try:
        with open(p, errors="replace") as f:
            for line in f:
                m = re.search(r"(\d+)/(\d+) games\s+(\d+)s elapsed", line)
                if m:
                    best = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except OSError:
        pass
    return best


# ---------------------------------------------------------------- analysis pieces
def per_bot_basic(games, bots):
    out = {}
    for b in bots:
        gs = [g for g in games if b in (g["a"], g["b"])]
        sc = [score_of(g, b) for g in gs]
        ms_tot = 0.0
        nmoves = 0
        avg_list, max_list = [], []
        for g in gs:
            k = moves_by(g, b)
            avg = g["a_ms"] if g["a"] == b else g["b_ms"]
            mx = g["a_max_ms"] if g["a"] == b else g["b_max_ms"]
            ms_tot += avg * k
            nmoves += k
            avg_list.append(avg)
            max_list.append(mx)
        first = [score_of(g, b) for g in gs if players_of(g)[0] == b]
        second = [score_of(g, b) for g in gs if players_of(g)[1] == b]
        out[b] = {"games": len(gs), "score": rnd(sum(sc) / len(sc)), "moves": nmoves,
                  "think_seconds": rnd(ms_tot / 1000.0, 1),
                  "ms_per_move_weighted": rnd(ms_tot / max(nmoves, 1), 2),
                  "ms_per_move_game_avg": rnd(float(np.mean(avg_list)), 2),
                  "max_ms": rnd(max(max_list), 1),
                  "score_first": rnd(sum(first) / len(first)) if first else None,
                  "score_second": rnd(sum(second) / len(second)) if second else None}
    return out


def blunder_stats(games, cache, bots):
    """Per bot: every move it made, judged by the solver (weak: win/draw/loss)."""
    res = {}
    first_err = {}
    for b in bots:
        res[b] = {"moves": 0, "judged": 0, "errors": 0, "points_lost": 0.0,
                  "in_won": 0, "threw_win": 0, "win_to_loss": 0, "in_drawn": 0, "draw_to_loss": 0,
                  "in_lost": 0,
                  "buckets": {"%d-%d" % bk: {"moves": 0, "errors": 0, "in_won": 0, "threw_win": 0}
                              for bk in PLY_BUCKETS}}
        first_err[b] = []
    missing = 0
    for g in games:
        p1, p2 = players_of(g)
        seen_err = {p1: None, p2: None}
        for i in range(len(g["opening"]), g["plies"]):
            mover = p1 if i % 2 == 0 else p2
            if mover not in res:
                continue
            r = res[mover]
            r["moves"] += 1
            sc = lookup(cache, g["moves"][:i])
            if sc is None:
                missing += 1
                continue
            col = int(g["moves"][i]) - 1
            best, ch = move_error(sc, col)
            err = best - ch
            r["judged"] += 1
            bk = next("%d-%d" % b2 for b2 in PLY_BUCKETS if b2[0] <= i <= b2[1])
            bb = r["buckets"][bk]
            bb["moves"] += 1
            if best == 1:
                r["in_won"] += 1
                bb["in_won"] += 1
            elif best == 0:
                r["in_drawn"] += 1
            else:
                r["in_lost"] += 1
            if err > 0:
                r["errors"] += 1
                bb["errors"] += 1
                r["points_lost"] += err / 2.0
                if best == 1:
                    r["threw_win"] += 1
                    bb["threw_win"] += 1
                    if ch == -1:
                        r["win_to_loss"] += 1
                elif best == 0:
                    r["draw_to_loss"] += 1
                if seen_err[mover] is None:
                    seen_err[mover] = i
        for p in (p1, p2):
            if p in first_err:
                first_err[p].append(seen_err[p])
    out = {}
    for b, r in res.items():
        j = max(r["judged"], 1)
        fe = [x for x in first_err[b] if x is not None]
        out[b] = {"moves": r["moves"], "judged": r["judged"], "errors": r["errors"],
                  "error_rate": rnd(r["errors"] / j, 5),
                  "points_lost_per_100_moves": rnd(100 * r["points_lost"] / j, 3),
                  "moves_in_won_positions": r["in_won"], "threw_win": r["threw_win"],
                  "threw_win_rate": rnd(r["threw_win"] / max(r["in_won"], 1), 5),
                  "win_to_loss": r["win_to_loss"],
                  "moves_in_drawn_positions": r["in_drawn"], "draw_to_loss": r["draw_to_loss"],
                  "draw_to_loss_rate": rnd(r["draw_to_loss"] / max(r["in_drawn"], 1), 5),
                  "moves_in_lost_positions": r["in_lost"],
                  "games": len(first_err[b]), "games_with_error": len(fe),
                  "games_with_error_share": rnd(len(fe) / max(len(first_err[b]), 1), 4),
                  "first_error_ply_median": rnd(float(np.median(fe)), 1) if fe else None,
                  "first_error_ply_hist": {"%d-%d" % bk: sum(1 for x in fe if bk[0] <= x <= bk[1])
                                           for bk in PLY_BUCKETS},
                  "by_ply": [{"bucket": k, "moves": v["moves"], "errors": v["errors"],
                              "error_rate": rnd(v["errors"] / v["moves"], 5) if v["moves"] else None,
                              "moves_in_won": v["in_won"], "threw_win": v["threw_win"],
                              "threw_win_rate": rnd(v["threw_win"] / v["in_won"], 5) if v["in_won"] else None}
                             for k, v in r["buckets"].items()]}
    return out, missing


def vs_expectation(games, bots, p1_exp):
    """Score minus game-theoretic expectation, per bot: all games, games vs pons_perfect, by colour."""
    out = {}
    for b in bots:
        rows = []
        for g in games:
            if b not in (g["a"], g["b"]):
                continue
            e1 = p1_exp[g["opening_id"]]
            first = players_of(g)[0] == b
            exp = e1 if first else 1 - e1
            opp = g["b"] if g["a"] == b else g["a"]
            rows.append((score_of(g, b), exp, first, opp, g["opening_id"]))

        def agg(rs):
            if not rs:
                return None
            return {"games": len(rs), "score": rnd(sum(r[0] for r in rs) / len(rs)),
                    "expected": rnd(sum(r[1] for r in rs) / len(rs)),
                    "delta_per_game": rnd(sum(r[0] - r[1] for r in rs) / len(rs)),
                    "points_lost": rnd(sum(max(0.0, r[1] - r[0]) for r in rs), 1),
                    "points_gained": rnd(sum(max(0.0, r[0] - r[1]) for r in rs), 1)}
        vp = [r for r in rows if r[3] == "pons_perfect"]
        out[b] = {"all": agg(rows), "first": agg([r for r in rows if r[2]]),
                  "second": agg([r for r in rows if not r[2]]),
                  "vs_pons": agg(vp), "vs_pons_first": agg([r for r in vp if r[2]]),
                  "vs_pons_second": agg([r for r in vp if not r[2]])}
        if b != "pons_perfect":
            out[b]["vs_pons_by_opening"] = [
                {"opening_id": oid, "games": len(rs), "score": rnd(sum(r[0] for r in rs), 1),
                 "expected": rnd(sum(r[1] for r in rs), 1)}
                for oid in sorted({r[4] for r in vp}) for rs in [[r for r in vp if r[4] == oid]]]
    return out


def game_lengths(games, bots):
    out = {}
    edges = list(range(6, 44, 4))
    for b in bots:
        gs = [g for g in games if b in (g["a"], g["b"])]
        wins = [g["plies"] for g in gs if score_of(g, b) == 1.0]
        losses = [g["plies"] for g in gs if score_of(g, b) == 0.0]
        draws = [g["plies"] for g in gs if score_of(g, b) == 0.5]

        def hist(xs):
            h, _ = np.histogram(xs, bins=edges + [43])
            return [int(v) for v in h]
        out[b] = {"bins": ["%d-%d" % (edges[i], (edges + [43])[i + 1] - 1) for i in range(len(edges))],
                  "wins": hist(wins), "losses": hist(losses), "draws": hist(draws),
                  "median_win_plies": rnd(float(np.median(wins)), 1) if wins else None,
                  "median_loss_plies": rnd(float(np.median(losses)), 1) if losses else None}
    return out


# ---------------------------------------------------------------- main build
def build(args):
    t_start = time.time()
    summ = _json(os.path.join(PART3, "summary.json"))
    games = load_games(os.path.join(PART3, "games.csv"))
    bots = summ["bots"]
    log("%d games, %d bots" % (len(games), len(bots)))
    stored = summ["elo"]
    info = summ.get("bot_info", {})

    # ---- 1. Elo reproduction on the full pool
    full, boot_full = elo_bootstrap(games, bots, summ.get("elo_anchor") or "kaggle_negamax", 1000.0,
                                    reps=args.boot, seed=0)
    dev_pt = max(abs(full[b][0] - stored[b]["elo"]) for b in bots)
    dev_ci = max(max(abs(full[b][1] - stored[b]["lo"]), abs(full[b][2] - stored[b]["hi"])) for b in bots)
    log("Elo reproduction: max |point dev| %.2f, max |CI bound dev| %.2f" % (dev_pt, dev_ci))
    order_full = sorted(bots, key=lambda b: -full[b][0])
    elo_full = [{"name": b, "rank": i + 1, "elo": rnd(full[b][0], 1), "lo": rnd(full[b][1], 1),
                 "hi": rnd(full[b][2], 1), "stored_elo": stored[b]["elo"], "stored_lo": stored[b]["lo"],
                 "stored_hi": stored[b]["hi"], "score": summ["per_bot"][b]["score"],
                 "group": info.get(b, {}).get("group"), "family": FAMILY.get(b)}
                for i, b in enumerate(order_full)]
    # a bigger bootstrap for differences between bots
    _, boot_diff = elo_bootstrap(games, bots, "kaggle_negamax", 1000.0, reps=args.diff_boot, seed=1)

    # ---- top-15 refit
    g15 = [g for g in games if g["a"] in TOP15 and g["b"] in TOP15]
    top, boot_top = elo_bootstrap(g15, TOP15, "ours_minimax_d4", 1409.0, reps=args.boot, seed=0)
    order_top = sorted(TOP15, key=lambda b: -top[b][0])
    rank_full = {b: i + 1 for i, b in enumerate(order_full)}
    elo_top15 = [{"name": b, "rank": i + 1, "rank_full_pool": rank_full[b], "rank_change": rank_full[b] - (i + 1),
                  "elo": rnd(top[b][0], 1), "lo": rnd(top[b][1], 1), "hi": rnd(top[b][2], 1),
                  "elo_full_pool": rnd(full[b][0], 1), "shift_vs_full_pool": rnd(top[b][0] - full[b][0], 1)}
                 for i, b in enumerate(order_top)]
    log("top-15 refit done (%d games)" % len(g15))

    # ---- opening values (exact solver, not weak)
    from tools.solver import Solver
    openings = {}
    for g in games:
        openings[g["opening_id"]] = g["opening"]
    p1_exp = {}
    op_rows = []
    with Solver(analyze=True, weak=False, timeout=120) as s:
        for oid in sorted(openings):
            o = openings[oid]
            sc = s.solve_all(o)
            p1_exp[oid] = value_to_p1_expected(o, sc)
            gs = [g for g in games if g["opening_id"] == oid]
            p1s = [g["score_a"] if g["a_first"] else 1 - g["score_a"] for g in gs]
            op_rows.append({"opening_id": oid, "moves": o, "plies": len(o), "solver_scores_side_to_move": sc,
                            "side_to_move": "first" if len(o) % 2 == 0 else "second",
                            "value_first_player": {1.0: "win", 0.5: "draw", 0.0: "loss"}[p1_exp[oid]],
                            "expected_score_first_player": p1_exp[oid], "games": len(gs),
                            "observed_score_first_player": rnd(sum(p1s) / len(p1s)),
                            "plies_median": rnd(float(np.median([g["plies"] for g in gs])), 1),
                            "share_games_le_9_plies": rnd(sum(1 for g in gs if g["plies"] <= 9) / len(gs)),
                            "first_player_win_share": rnd(sum(1 for x in p1s if x == 1.0) / len(p1s))})
    for r in op_rows:
        for b in [BEST, "nn_az_6x64_mcts200", "bruneton_az", "pons_perfect", "nn_ppo_from_sup_policy"]:
            gs = [g for g in games if g["opening_id"] == r["opening_id"] and b in (g["a"], g["b"])]
            r["score_" + b] = rnd(sum(score_of(g, b) for g in gs) / len(gs)) if gs else None
    log("opening values: %s" % [r["value_first_player"] for r in op_rows])

    # ---- head to head
    h2h_best = [head_to_head(games, BEST, y) for y in NEURAL + ["pons_perfect"]]
    extra_pairs = [("nn_az_6x64_mcts200", "bruneton_az"), ("nn_az_6x64_mcts200", "pons_perfect"),
                   ("bruneton_az", "pons_perfect"), ("nn_ppo_from_sup_policy", "bruneton_az"),
                   ("nn_az_6x64_mcts200", "nn_ppo_from_sup_policy")]
    h2h_extra = [head_to_head(games, x, y) for x, y in extra_pairs]
    k = len(TOP15)
    mat = [[None] * k for _ in range(k)]
    for i, x in enumerate(TOP15):
        for j, y in enumerate(TOP15):
            if i != j:
                gs = [g for g in games if {g["a"], g["b"]} == {x, y}]
                mat[i][j] = rnd(sum(score_of(g, x) for g in gs) / len(gs), 4) if gs else None
    diffs_full = [diff_ci(boot_diff, bots, BEST, y, full) for y in NEURAL + ["pons_perfect"]]
    diffs_full += [diff_ci(boot_diff, bots, x, y, full) for x, y in extra_pairs]
    diffs_top = [diff_ci(boot_top, TOP15, BEST, y, top) for y in NEURAL + ["pons_perfect"] if y in TOP15]
    # sensitivity: drop the near-trivial openings (games that end in <= 9 plies in most pairings)
    triv_ids = {r["opening_id"] for r in op_rows if (r["share_games_le_9_plies"] or 0) > 0.5}
    g_nt = [g for g in games if g["opening_id"] not in triv_ids]
    nt, boot_nt = elo_bootstrap(g_nt, bots, "kaggle_negamax", 1000.0, reps=args.diff_boot, seed=2)
    sens = {"excluded_openings": sorted(triv_ids), "games": len(g_nt),
            "differences": [diff_ci(boot_nt, bots, BEST, y, nt) for y in
                            ["nn_az_6x64_mcts200", "bruneton_az", "nn_ppo_from_sup_policy", "pons_perfect"]],
            "h2h_sup_vs_az": head_to_head(g_nt, BEST, "nn_az_6x64_mcts200"),
            "elo": {b: rnd(nt[b][0], 1) for b in bots}}
    head = {"best_vs": h2h_best, "other_pairs": h2h_extra,
            "top15_score_matrix": {"bots": TOP15, "matrix": mat,
                                   "note": "row bot's mean score against column bot (40 games per cell, "
                                           "39 for bruneton_az vs nn_sup_6x64_mcts200)"}}

    # ---- per-bot basic + vs expectation + lengths
    basic = per_bot_basic(games, bots)
    vexp = vs_expectation(games, bots, p1_exp)
    lengths = game_lengths(games, [BEST, "nn_az_6x64_mcts200", "bruneton_az", "pons_perfect",
                                   "nn_ppo_from_sup_policy", "nn_ppo_6x64_mcts200"])

    # ---- solver blunder analysis
    blunders, solver_meta = None, {"run": False}
    if not args.no_solver:
        positions = set()
        for g in games:
            for i in range(len(g["opening"]), g["plies"]):
                positions.add(g["moves"][:i])
        cache = _load_cache(args.cache)
        sm = solve_positions(positions, cache, args.cache, workers=args.solver_workers)
        blunders, missing = blunder_stats(games, cache, bots)
        solver_meta = {"run": True, "mode": "weak (-w): win/draw/loss exact, distance-to-win not",
                       "analyze": True, "book": os.path.isfile(os.path.join(ROOT, "tools", "pons", "7x6.book")),
                       "positions_total_moves": sum(g["plies"] - len(g["opening"]) for g in games),
                       "unique_positions": len(positions),
                       "unique_up_to_mirror": len({canon(p)[0] for p in positions}),
                       "sampled": False, "unjudged_moves": missing, **sm}
        log("blunder analysis done; unjudged moves: %d" % missing)

    # ---- methods table, runs, regressions
    mdoc, mrows = methods_rows()
    mset = mdoc.get("settings", {})

    def mrow(bot):
        run, srch = METHOD_MAP.get(bot, (None, None))
        if bot == "ours_minimax_d4":
            return next((e for e in mdoc.get("entries", []) if e.get("method") == "hand"), None)
        return mrows.get((run, srch))

    def m(run, srch):
        return mrows.get((run, srch))

    policy_q = []
    for run in ["sup_6x64", "az_6x64", "ppo_6x64", "ppo_from_sup", "dqn_6x64", "distill_az_2x32",
                "distill_ens_2x32", "td_ntuple"]:
        for srch in ("none", "mcts200"):
            e = m(run, srch)
            if e:
                policy_q.append({"run": run, "search": srch, "optimal": e["optimal"], "keeps_nl": e["keeps_nl"],
                                 "ladder_elo": e["elo"], "ms_per_move": e["ms_per_move"], "params": e.get("params")})
    sup_eval = _json(os.path.join(RUNS, "sup_6x64", "eval.json"), {}) or {}
    search_gain = []
    for run in ["sup_6x64", "az_6x64", "ppo_6x64", "ppo_from_sup", "pilot_cpu_sup", "pilot_cpu_az"]:
        a, b = m(run, "none"), m(run, "mcts200")
        if a and b:
            search_gain.append({"run": run, "policy_ladder_elo": a["elo"], "mcts200_ladder_elo": b["elo"],
                                "ladder_gain": rnd(b["elo"] - a["elo"], 1), "policy_optimal": a["optimal"],
                                "mcts200_optimal": b["optimal"], "optimal_gain": rnd(b["optimal"] - a["optimal"], 3)})
    arena_pair = {"run": "ppo_6x64", "policy_arena_elo": stored["nn_ppo_6x64_policy"]["elo"],
                  "mcts200_arena_elo": stored["nn_ppo_6x64_mcts200"]["elo"],
                  "arena_gain": rnd(stored["nn_ppo_6x64_mcts200"]["elo"] - stored["nn_ppo_6x64_policy"]["elo"], 1),
                  "arena_gain_ci": [diff_ci(boot_diff, bots, "nn_ppo_6x64_mcts200", "nn_ppo_6x64_policy", full)[k2]
                                    for k2 in ("lo", "hi")]}
    labels_meta = _json(os.path.join(ROOT, "nn_data", "solver_300k.json"), {}) or {}
    training = []
    base = stored["ours_minimax_d4"]["elo"]
    for bot, run, signal in [(BEST, "sup_6x64", "perfect solver labels (300k positions)"),
                             ("nn_az_6x64_mcts200", "az_6x64", "AlphaZero self-play (100 gens x 1000 games)"),
                             ("nn_ppo_6x64_mcts200", "ppo_6x64", "PPO self-play (300 iters)"),
                             ("nn_ppo_from_sup_policy", "ppo_from_sup", "PPO fine-tune of sup_6x64"),
                             ("nn_dqn_6x64_policy", "dqn_6x64", "DQN self-play"),
                             ("nn_distill_az_2x32_policy", "distill_az_2x32", "distillation of az_6x64+MCTS100"),
                             ("nn_distill_ens_2x32_policy", "distill_ens_2x32", "distillation of an ensemble"),
                             ("evo_evo_d2_d4", "evo_d2", "CMA-ES on game results"),
                             ("ntuple_td_ntuple_d4", "td_ntuple", "TD(lambda) self-play")]:
        cfg, cost = run_info(run)
        e = mrow(bot) or {}
        c = e.get("cost") or cost or {}
        wall = c.get("wall_hours")
        kwh = c.get("energy_kwh")
        incl = (c.get("including_teachers") or {})
        row = {"bot": bot, "run": run, "signal": signal, "arena_elo": stored[bot]["elo"],
               "elo_above_minimax_d4": rnd(stored[bot]["elo"] - base, 1),
               "wall_hours": wall, "gpu_hours": c.get("gpu_hours"), "energy_kwh": kwh,
               "energy_kwh_incl_teachers": incl.get("energy_kwh"), "wall_hours_incl_teachers": incl.get("wall_hours"),
               "cost_source": c.get("source", "cost.json" if cost else None),
               "params": (cfg or {}).get("params")}
        if run == "sup_6x64":
            row["label_generation_hours_wall"] = rnd(labels_meta.get("seconds", 0) / 3600.0, 3)
            row["label_generation_note"] = ("solver labelling of 300k positions: %s s wall on 14 workers, not metered "
                                            "(no cost.json for sup_6x64; energy unknown)" % labels_meta.get("seconds"))
            row["wall_hours_incl_labels"] = rnd((wall or 0) + labels_meta.get("seconds", 0) / 3600.0, 3)
        if run == "ppo_from_sup":
            row["note"] = "starts from sup_6x64 weights: its true cost includes sup_6x64 (+0.757 h, + labels)"
        if wall:
            row["elo_above_d4_per_hour"] = rnd(row["elo_above_minimax_d4"] / wall, 1)
        if kwh:
            row["elo_above_d4_per_kwh"] = rnd(row["elo_above_minimax_d4"] / kwh, 1)
        training.append(row)

    size_rows = []
    for bot in [BEST, "nn_az_6x64_mcts200", "bruneton_az", "plkmo_az", "nn_ppo_6x64_mcts200",
                "nn_distill_az_2x32_policy", "nn_distill_ens_2x32_policy", "nn_ppo_from_sup_policy"]:
        run = METHOD_MAP.get(bot, (None,))[0]
        p = PARAMS_OPEN.get(bot) or (run_info(run)[0].get("params") if run else None)
        size_rows.append({"bot": bot, "params": p, "arena_elo": stored[bot]["elo"],
                          "search": BUDGET.get(bot, "none (1 forward pass)"),
                          "arena_ms_per_move": summ["per_bot"][bot]["avg_ms"]})

    # time confound
    tw = tournament_wall_seconds()
    time_conf = {
        "arena_avg_ms": {b: summ["per_bot"][b]["avg_ms"] for b in
                         [BEST, "nn_az_6x64_mcts200", "nn_ppo_6x64_mcts200", "bruneton_az", "plkmo_az",
                          "nn_ppo_6x64_policy", "nn_ppo_from_sup_policy", "nn_dqn_6x64_policy",
                          "nn_distill_az_2x32_policy", "nn_distill_ens_2x32_policy", "alfo_mcts"]},
        "arena_max_ms": {b: summ["per_bot"][b]["max_ms"] for b in [BEST, "nn_az_6x64_mcts200", "bruneton_az"]},
        "methods_table_ms": {"sup_6x64 policy": (m("sup_6x64", "none") or {}).get("ms_per_move"),
                             "sup_6x64 mcts200": (m("sup_6x64", "mcts200") or {}).get("ms_per_move"),
                             "az_6x64 mcts200": (m("az_6x64", "mcts200") or {}).get("ms_per_move"),
                             "distill_az_2x32 policy": (m("distill_az_2x32", "none") or {}).get("ms_per_move")},
        "measured_now_single_thread_ms": {"sup_6x64 policy": 7.5, "distill_az_2x32 policy": 2.05,
                                          "sup_6x64 mcts200": 950.0,
                                          "note": "measured 2026-11-16 with OPENBLAS_NUM_THREADS=1 on the same "
                                                  "machine while other training jobs ran (load ~25), 200 calls "
                                                  "(policy) / 3 moves (MCTS) on position '4453'"},
        "slowdown_factor_policy_6x64": rnd(summ["per_bot"]["nn_ppo_6x64_policy"]["avg_ms"]
                                           / (m("ppo_6x64", "none") or {"ms_per_move": float("nan")})["ms_per_move"], 0),
        "slowdown_factor_mcts200": rnd(summ["per_bot"][BEST]["avg_ms"]
                                       / (m("sup_6x64", "mcts200") or {"ms_per_move": float("nan")})["ms_per_move"], 0),
        "explanation": (
            "Our nets run in numpy (nn/numpy_net.py: 3x3 convs as tensordot -> OpenBLAS). The arena uses 14 "
            "spawned workers; each imports arena.tournament, which imports numpy at module load, before "
            "_init() sets OMP_NUM_THREADS=1 (setdefault, and OPENBLAS_NUM_THREADS is never set). OpenBLAS "
            "therefore starts its default pool (16 threads) in every worker -> up to 14 x 16 BLAS threads on "
            "16 cores. A 6x64 forward pass (tensordot over 64 channels) is large enough for OpenBLAS to "
            "multithread, so it suffers; the 2x32 distills' matrices are small enough to stay single-threaded "
            "(2.0 ms in the arena = 2 ms measured alone). Torch bots (bruneton_az) call "
            "torch.set_num_threads(1) in _init and are unaffected. This is an inference from the code plus "
            "the timings above; it was not re-run with BLAS pinned to 1 thread inside the arena."),
        "affects_strength": (
            "No, as far as the code shows: our MCTS bots use a fixed 200 simulations and play "
            "deterministically (no time control; no move hit the 300 s timeout; 0 forfeits), so the same "
            "moves would have been played at ~0.3-1 s/move. The confound matters only for any "
            "'equal wall-clock' claim: none can be made from this run."),
        "equal_budget": (
            "Nominally equal search budget between our MCTS nets and bruneton_az: 200 PUCT simulations each, "
            "fresh tree every move, exact terminal values in both. Differences: c_puct 1.5 (ours) vs 1 "
            "(bruneton), centre-first vs random tie-break, and bruneton's net is 2.6x larger (1.20 M vs "
            "0.458 M params), so its per-simulation compute is higher, not lower. plkmo_az used 777 reads and "
            "alfo_mcts 3000 random-roll-out iterations: not equal budgets. Policy-only nets use no search."),
        "tournament_wall_seconds": tw[2] if tw else None,
    }

    # where games are decided
    decided = {"first_player_score_overall": summ.get("first_player_score"),
               "first_second_by_bot": {b: {"first": basic[b]["score_first"], "second": basic[b]["score_second"]}
                                       for b in bots},
               "game_lengths": lengths, "openings": op_rows}

    # regressions
    regs = []

    def pts(bots_, xf):
        out = []
        for b in bots_:
            x = xf(b)
            if x is None:
                continue
            out.append({"name": b, "x": rnd(x, 5), "y": stored[b]["elo"]})
        return out

    with_m = [b for b in bots if mrow(b)]
    pol = [b for b in with_m if (mrow(b) or {}).get("search") == "none"]
    sea = [b for b in with_m if (mrow(b) or {}).get("search") != "none"]

    def add(rid, x, series, p, y="arena Elo (part3)"):
        r = linreg([q["x"] for q in p], [q["y"] for q in p])
        regs.append({"id": rid, "x": x, "y": y, "series": series, "points": p, **r})

    for xname, f in [("optimal (methods table, %d solver positions)" % mset.get("positions", 0),
                      lambda b: mrow(b)["optimal"]),
                     ("keeps_nl (methods table)", lambda b: mrow(b)["keeps_nl"]),
                     ("log10 ms/move (methods table, single process)",
                      lambda b: math.log10(mrow(b)["ms_per_move"])),
                     ("methods-table ladder Elo (vs minimax d1-d8)", lambda b: mrow(b)["elo"])]:
        rid = {"o": "elo_vs_optimal", "k": "elo_vs_keeps_nl", "l": "elo_vs_log_ms",
               "m": "elo_vs_ladder_elo"}[xname[0]]
        for sname, bl in [("all", with_m), ("policy-only", pol), ("with search", sea)]:
            add(rid, xname, sname, pts(bl, f))

    def params_of(b):
        if b in PARAMS_OPEN:
            return PARAMS_OPEN[b]
        run = METHOD_MAP.get(b, (None,))[0]
        if run and b.startswith("nn_"):
            return run_info(run)[0].get("params")
        return None
    nets = [b for b in bots if b.startswith("nn_")]
    add("elo_vs_log_params", "log10 parameters", "our nets", pts(nets, lambda b: math.log10(params_of(b))))
    add("elo_vs_log_params", "log10 parameters", "our nets + open-source AlphaZero",
        pts(nets + ["bruneton_az", "plkmo_az"], lambda b: math.log10(params_of(b))))
    add("elo_vs_log_params", "log10 parameters", "our nets + bruneton_az (plkmo excluded)",
        pts(nets + ["bruneton_az"], lambda b: math.log10(params_of(b))))
    add("elo_vs_log_arena_ms", "log10 arena ms/move (confounded by BLAS oversubscription)", "all 20 bots",
        pts(bots, lambda b: math.log10(max(summ["per_bot"][b]["avg_ms"], 0.01))))

    def cost_of(b, key):
        e = mrow(b)
        c = (e or {}).get("cost") or {}
        v = c.get(key)
        return math.log10(v) if v and v > 0 else None
    trained = [b for b in with_m if METHOD_MAP.get(b, (None,))[0]]
    for key, lab in [("energy_kwh", "log10 training energy (kWh)"), ("wall_hours", "log10 training wall hours")]:
        add("elo_vs_log_" + key, lab, "all", pts(trained, lambda b: cost_of(b, key)))
        add("elo_vs_log_" + key, lab, "policy-only", pts([b for b in trained if b in pol], lambda b: cost_of(b, key)))
        add("elo_vs_log_" + key, lab, "with search", pts([b for b in trained if b in sea], lambda b: cost_of(b, key)))
    pred_table = sorted([{"id": r["id"], "series": r["series"], "n": r["n"], "r2": r["r2"],
                          "spearman_rho": r.get("spearman_rho")}
                         for r in regs if r["series"] in ("all", "our nets", "all 20 bots") and r["r2"] is not None],
                        key=lambda r: -(r["r2"] or 0))

    # size contrast
    def mm(run, s_):
        e = m(run, s_) or {}
        return {"optimal": e.get("optimal"), "keeps_nl": e.get("keeps_nl"), "ladder_elo": e.get("elo"),
                "ms_per_move": e.get("ms_per_move"), "params": e.get("params")}
    size = {
        "rows": [{"bot": b, "arena_elo": stored[b]["elo"], "arena_lo": stored[b]["lo"], "arena_hi": stored[b]["hi"],
                  "arena_ms": summ["per_bot"][b]["avg_ms"], **mm(*METHOD_MAP[b])}
                 for b in ["nn_ppo_6x64_policy", "nn_ppo_from_sup_policy", "nn_dqn_6x64_policy",
                           "nn_distill_az_2x32_policy", "nn_distill_ens_2x32_policy"]],
        "methods_only_rows": {"sup_6x64 policy": mm("sup_6x64", "none"), "az_6x64 policy": mm("az_6x64", "none")},
        "distill_vs_teacher": {
            "student": "nn_distill_az_2x32_policy (2x32, 50,146 params, no search)",
            "teacher": "az_6x64 + MCTS 100 (458,338 params)",
            "student_arena_elo": stored["nn_distill_az_2x32_policy"]["elo"],
            "closest_teacher_in_arena": "nn_az_6x64_mcts200 (MCTS 200, not 100)",
            "teacher_arena_elo": stored["nn_az_6x64_mcts200"]["elo"],
            "diff": diff_ci(boot_diff, bots, "nn_distill_az_2x32_policy", "nn_az_6x64_mcts200", full),
            "h2h": head_to_head(games, "nn_distill_az_2x32_policy", "nn_az_6x64_mcts200"),
            "student_methods": mm("distill_az_2x32", "none"), "teacher_policy_methods": mm("az_6x64", "none"),
            "teacher_mcts200_methods": mm("az_6x64", "mcts200"),
            "missing": ("part3 has neither az_6x64 policy-only nor az_6x64 + MCTS100 (the actual teacher), and no "
                        "6x64 vs 2x32 pair trained the same way. So the student-vs-teacher gap and the pure "
                        "size effect cannot be read from the arena; the methods table (ladder Elo, 100 games "
                        "each) has az policy-only 1547 vs student 1579 and optimal 0.727 vs 0.724.")},
    }

    # health
    pair_counts = {}
    for g in games:
        key = tuple(sorted((g["a"], g["b"])))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    expected = {(o, af) for o in openings for af in (0, 1)}
    missing_games = []
    for key, c in pair_counts.items():
        if c != 40:
            have = set()
            for g in games:
                if tuple(sorted((g["a"], g["b"]))) == key:
                    af = g["a_first"] if g["a"] == key[0] else 1 - g["a_first"]
                    have.add((g["opening_id"], af))
            for (o, af) in sorted(expected - have):
                missing_games.append({"pair": list(key), "opening_id": o, "opening": openings[o],
                                      "first_player": key[0] if af else key[1]})
    tot_think = sum(basic[b]["think_seconds"] for b in bots)
    health = {
        "games": len(games), "expected_games": len(bots) * (len(bots) - 1) // 2 * 40,
        "pairings": len(pair_counts),
        "games_per_pairing": {str(c): sum(1 for v in pair_counts.values() if v == c)
                              for c in sorted(set(pair_counts.values()))},
        "missing": missing_games, "forfeits": sum(1 for g in games if g["forfeit"]),
        "draws": sum(1 for g in games if g["score_a"] == 0.5),
        "wall_seconds": tw[2] if tw else None, "wall_hours": rnd(tw[2] / 3600, 2) if tw else None,
        "wall_note": ("last progress line of runs/arena_part3.log (%d/%d games); the final ~150 games are not "
                      "in the log, so true wall time is slightly longer" % (tw[0], tw[1])) if tw else None,
        "workers": summ.get("meta", {}).get("workers"),
        "think_seconds_total": rnd(tot_think, 0),
        "per_bot": [{"name": b, "avg_ms": summ["per_bot"][b]["avg_ms"], "max_ms": summ["per_bot"][b]["max_ms"],
                     "moves": basic[b]["moves"], "think_seconds": basic[b]["think_seconds"],
                     "think_hours": rnd(basic[b]["think_seconds"] / 3600, 2),
                     "share": rnd(basic[b]["think_seconds"] / tot_think, 4)}
                    for b in sorted(bots, key=lambda b: -basic[b]["think_seconds"])],
        "note": ("think_seconds = sum over games of (per-game avg ms x moves the bot made). Workers ran in "
                 "parallel, so total thinking time exceeds wall time."),
    }

    # ---- best-nn decomposition + answer
    h_az = next(h for h in h2h_best if h["b"] == "nn_az_6x64_mcts200")
    h_br = next(h for h in h2h_best if h["b"] == "bruneton_az")
    h_po = next(h for h in h2h_best if h["b"] == "pons_perfect")
    d_az = next(d for d in diffs_full if d["b"] == "nn_az_6x64_mcts200" and d["a"] == BEST)
    d_br = next(d for d in diffs_full if d["b"] == "bruneton_az" and d["a"] == BEST)
    d_po = next(d for d in diffs_full if d["b"] == "pons_perfect" and d["a"] == BEST)
    d_pfs = next(d for d in diffs_full if d["b"] == "nn_ppo_from_sup_policy" and d["a"] == BEST)
    decomposition = {
        "a_policy_quality": {"rows": policy_q, "sup_eval_by_ply": sup_eval.get("accuracy"),
                             "positions": mset.get("positions"),
                             "note": "methods table used %s of the 3,000 held-out positions (random.Random(5) "
                                     "sample); runs/sup_6x64/eval.json has all 3,000" % mset.get("positions")},
        "b_search_gain": {"methods_table": search_gain, "arena": arena_pair,
                          "note": "only ppo_6x64 appears in part3 both with and without search; sup_6x64 and "
                                  "az_6x64 policy-only were not in the arena"},
        "c_training_signal_and_cost": training,
        "d_net_size": size_rows,
        "e_where_decided": decided,
        "f_blunders": {"meta": solver_meta, "per_bot": blunders},
        "g_time_confound": time_conf,
        "vs_game_theoretic_expectation": vexp,
    }

    def bl(b, key):
        return (blunders or {}).get(b, {}).get(key)
    supported, suggestive = [], []
    supported.append(
        "Among the 19 non-solver bots of part3, %s has the highest point-estimate Elo (%.0f, CI %.0f-%.0f). "
        "Its lead is significant over every neural net except our own az_6x64 (see suggestive). Against the open-source AlphaZero net bruneton_az the gap is %+.0f "
        "Elo (bootstrap CI %+.0f to %+.0f, P>0 = %.3f) and head-to-head %d-%d-%d (score %.3f, CI %.3f-%.3f)."
        % (BEST, full[BEST][0], full[BEST][1], full[BEST][2], d_br["diff"], d_br["lo"], d_br["hi"],
           d_br["p_a_better"], h_br["w"], h_br["d"], h_br["l"], h_br["score"], h_br["lo"], h_br["hi"]))
    supported.append(
        "Better than our PPO/DQN/distilled nets: sup - ppo_from_sup = %+.0f Elo (CI %+.0f to %+.0f)."
        % (d_pfs["diff"], d_pfs["lo"], d_pfs["hi"]))
    supported.append(
        "Its policy head alone is the most accurate move predictor we have: optimal %.3f vs 0.727 (az), "
        "0.786 (ppo_from_sup), 0.838 (dqn) on the same %s solver positions; with MCTS200 0.940 vs 0.834 (az). "
        "(Note dqn: high 'optimal' but low arena Elo - per-position accuracy alone does not rank bots.)"
        % ((m("sup_6x64", "none") or {}).get("optimal", float("nan")), mset.get("positions")))
    sup_vs_az_txt = ("sup vs az (our self-play net): %+.0f Elo, bootstrap CI %+.0f to %+.0f, P(sup>az) = %.3f; "
                     "head-to-head %d-%d-%d, score %.3f (CI %.3f-%.3f), sign test p = %.3f."
                     % (d_az["diff"], d_az["lo"], d_az["hi"], d_az["p_a_better"], h_az["w"], h_az["d"], h_az["l"],
                        h_az["score"], h_az["lo"], h_az["hi"], h_az["p_sign_test"]))
    d_az15 = next(d for d in diffs_top if d["b"] == "nn_az_6x64_mcts200")
    d_aznt = next(d for d in sens["differences"] if d["b"] == "nn_az_6x64_mcts200")
    sup_vs_az_txt += (" Top-15 pool: %+.0f (CI %+.0f to %+.0f). Without the two near-trivial openings: %+.0f "
                      "(CI %+.0f to %+.0f)." % (d_az15["diff"], d_az15["lo"], d_az15["hi"],
                                               d_aznt["diff"], d_aznt["lo"], d_aznt["hi"]))
    if d_az["lo"] > 0 and d_az15["lo"] > 0 and d_aznt["lo"] > 0 and h_az["p_sign_test"] < 0.05:
        supported.append(sup_vs_az_txt)
    else:
        suggestive.append(sup_vs_az_txt + " Not robust: the interval includes 0 in at least one analysis and/or "
                          "the direct match is not significant, so 'better than our AlphaZero net' is NOT "
                          "established; the full-pool gap comes partly from results against weaker bots.")
    vp = {b: (vexp[b]["vs_pons"] or {}).get("points_lost") for b in [BEST, "nn_az_6x64_mcts200", "bruneton_az"]}
    suggestive.append(
        "Elo cannot separate sup from perfect play (sup - pons = %+.0f, CI %+.0f to %+.0f), but sup is not "
        "perfect: against pons (40 games, expected score 0.5 from the opening values) it scored %.3f (%d-%d-%d), "
        "i.e. it gave away %.1f of 20 available points; az_6x64 gave away %.1f and bruneton_az %.1f. The Elo tie "
        "only means the 20-opening pool cannot tell them apart (pons' Elo is capped by its lost openings)."
        % (d_po["diff"], d_po["lo"], d_po["hi"], h_po["score"], h_po["w"], h_po["d"], h_po["l"],
           vp[BEST], vp["nn_az_6x64_mcts200"], vp["bruneton_az"]))
    if blunders:
        suggestive.append(
            "Solver blunder rates (fraction of own moves that change the game-theoretic result): sup %.4f, az "
            "%.4f, bruneton %.4f, ppo_from_sup %.4f. These are over different positions (each bot reaches "
            "its own), so they are descriptive, not a controlled comparison."
            % (bl(BEST, "error_rate"), bl("nn_az_6x64_mcts200", "error_rate"), bl("bruneton_az", "error_rate"),
               bl("nn_ppo_from_sup_policy", "error_rate")))
    suggestive.append(
        "Why (hypothesis): supervised training on perfect solver labels gives a better prior and value than "
        "12.3 h of self-play at this size, and MCTS200 adds most where the prior is good (ladder %+.0f for sup "
        "vs %+.0f for az). Confounds: sup and az differ in data, training length, and the sup net was "
        "selected on bench_optimal (best.npz); one run each, no seeds."
        % (next(r["ladder_gain"] for r in search_gain if r["run"] == "sup_6x64"),
           next(r["ladder_gain"] for r in search_gain if r["run"] == "az_6x64")))
    needed = [
        "Re-run the top neural bots with BLAS pinned to 1 thread (OPENBLAS_NUM_THREADS=1 before numpy import) "
        "and add an equal wall-clock condition (e.g. 1 s/move for every MCTS bot, bruneton_az included), "
        "plus bruneton_az at 800+ sims; this settles whether the ranking is about the net or the budget.",
        "More openings (e.g. 100-200 instead of 20) for the sup vs az vs bruneton vs pons sub-pool: the CI on "
        "sup - az is driven by 20 opening clusters per pairing.",
        "Size/signal ablation: train az and sup at identical size, steps and seeds (>=3 seeds each), and put "
        "sup/az policy-only and az+MCTS100 (the distillation teacher) into the arena.",
    ]
    answer = {"supported": supported, "suggestive": suggestive, "needed_experiments": needed}

    # ---- findings
    findings = []

    def F(id_, title, claim, evidence, caveat):
        findings.append({"id": "arena-" + id_, "title": title, "claim": claim, "evidence": evidence,
                         "caveat": caveat})
    F("elo-repro", "Elo reproduced",
      "Re-implemented Bradley-Terry + cluster bootstrap reproduces the stored part3 Elo of all 20 bots to "
      "within %.2f points (CI bounds within %.1f)." % (dev_pt, dev_ci),
      "elo_full[].elo vs stored_elo", "CI bounds depend on the resampling order; reps = %d, seed 0." % args.boot)
    F("best-nn-vs-open-source", "Our supervised net beats the open-source AlphaZero net",
      "nn_sup_6x64_mcts200 is %+.0f Elo above bruneton_az (CI %+.0f to %+.0f); head-to-head score %.3f."
      % (d_br["diff"], d_br["lo"], d_br["hi"], h_br["score"]),
      "best_nn.pairwise, head_to_head.best_vs",
      "Same nominal budget (200 sims) but different code (c_puct, tie-break); bruneton's 'best' weights as "
      "shipped; one weight file only.")
    F("sup-vs-az", "Supervised vs self-play net",
      sup_vs_az_txt, "best_nn.pairwise (diff with bootstrap), head_to_head.best_vs",
      "One training run each; differ in data, size of training and checkpoint selection.")
    F("sup-vs-pons", "Not distinguishable from the solver by Elo",
      "sup - pons_perfect = %+.0f Elo (CI %+.0f to %+.0f); vs pons directly sup scored %.3f in 40 games."
      % (d_po["diff"], d_po["lo"], d_po["hi"], h_po["score"]),
      "best_nn.pairwise", "Elo of a perfect player is capped by lost openings; see openings table.")
    F("search-gain", "Search helps most for the best prior",
      "MCTS200 adds %s ladder Elo (methods table); in the arena ppo_6x64 gains %+.0f (CI %+.0f to %+.0f)."
      % (", ".join("%+.0f to %s" % (r["ladder_gain"], r["run"]) for r in search_gain),
         arena_pair["arena_gain"], arena_pair["arena_gain_ci"][0], arena_pair["arena_gain_ci"][1]),
      "best_nn.decomposition.b_search_gain",
      "Ladder Elo extrapolates from near-100 %% scores vs minimax d1-d8 (100 games): 2180 for sup+MCTS is "
      "far outside the ladder (max d8 = 1500) and not comparable to arena Elo.")
    F("time-confound", "Time per move is a hardware artefact",
      "Our MCTS200 nets averaged %.0f s/move in part3 vs 0.29 s in the methods table (~%dx); policy-only 6x64 "
      "nets ~0.49 s vs 2.9 ms; bruneton_az 0.78 s."
      % (summ["per_bot"][BEST]["avg_ms"] / 1000, int(time_conf["slowdown_factor_mcts200"] or 0)),
      "best_nn.decomposition.g_time_confound",
      "Explained from code (BLAS thread oversubscription), not re-measured inside the arena; fixed-sim "
      "search means strength is unaffected, but no equal-time claim can be made.")
    if blunders:
        F("blunders", "Who throws away won positions",
          "Share of own moves that change the solver result: sup %.2f %%, az %.2f %%, bruneton %.2f %%, "
          "pons 0 %%. Share of moves in won positions that give the win away: sup %.2f %%, az %.2f %%, "
          "bruneton %.2f %%."
          % (100 * bl(BEST, "error_rate"), 100 * bl("nn_az_6x64_mcts200", "error_rate"),
             100 * bl("bruneton_az", "error_rate"), 100 * bl(BEST, "threw_win_rate"),
             100 * bl("nn_az_6x64_mcts200", "threw_win_rate"), 100 * bl("bruneton_az", "threw_win_rate")),
          "best_nn.decomposition.f_blunders (every move of every game solved, weak mode)",
          "Each bot faces different positions; rates are descriptive.")
    non_elo = [r for r in pred_table if r["id"] != "elo_vs_ladder_elo"]
    best_pred = non_elo[0] if non_elo else None
    lad = next((r for r in pred_table if r["id"] == "elo_vs_ladder_elo"), None)
    if best_pred:
        F("predictor", "Best single predictor of arena Elo",
          "Of the non-Elo measurements, %s predicts arena Elo best (%s series, R^2 = %.2f, n = %d); "
          "'optimal' is weaker (R^2 = %.2f). The other strength measure, methods-table ladder Elo, has "
          "R^2 = %.2f (Spearman %.2f) but compresses at the top (slope %.2f)."
          % (best_pred["id"], best_pred["series"], best_pred["r2"], best_pred["n"],
             next((r["r2"] for r in pred_table if r["id"] == "elo_vs_optimal"), float("nan")),
             lad["r2"] if lad else float("nan"), lad["spearman_rho"] if lad else float("nan"),
             next((r["slope"] for r in regs if r["id"] == "elo_vs_ladder_elo" and r["series"] == "all"), float("nan"))),
          "regressions, predictor_ranking",
          "n = 8-20 points, one run per method; bootstrap slope CIs are wide and R^2 differences between "
          "predictors are not significant.")
    F("health", "Tournament health",
      "%d of %d games played (1 missing: %s), 0 forfeits; our three MCTS200 nets used %.0f %% of all "
      "thinking time." % (len(games), health["expected_games"],
                          "%s vs %s opening %s" % (missing_games[0]["pair"][0], missing_games[0]["pair"][1],
                                                   missing_games[0]["opening_id"]) if missing_games else "none",
                          100 * sum(next(r["share"] for r in health["per_bot"] if r["name"] == b)
                                    for b in [BEST, "nn_az_6x64_mcts200", "nn_ppo_6x64_mcts200"])),
      "health", "Think time is reconstructed from per-game averages.")

    triv = [r for r in op_rows if (r["share_games_le_9_plies"] or 0) > 0.5]
    vals = [r["value_first_player"] for r in op_rows]
    F("openings", "Opening set: values and two near-trivial openings",
      "Solver values of the 20 openings for the first player: %d win, %d draw, %d loss. %d openings (%s) end "
      "in <= 9 plies in %s of games: the side to move has an immediate, unstoppable threat, so ~%d %% of all "
      "games carry almost no information about skill (sup lost %d of its %d losses there)."
      % (vals.count("win"), vals.count("draw"), vals.count("loss"), len(triv),
         ", ".join(r["moves"] for r in triv), "/".join("%.0f %%" % (100 * r["share_games_le_9_plies"]) for r in triv),
         round(100 * sum(r["games"] for r in triv) / len(games)),
         sum(1 for g in games if g["opening_id"] in {r["opening_id"] for r in triv} and BEST in (g["a"], g["b"])
             and score_of(g, BEST) == 0.0),
         summ["per_bot"][BEST]["losses"]),
      "best_nn.decomposition.e_where_decided.openings",
      "Openings are random 2-4 ply sequences (seed 4150) filtered only for mirror duplicates; a new opening set "
      "should exclude positions with a forced win in <= 3 moves.")

    doc = {
        "id": "arena", "title": "Arena analysis: part3 round robin (20 bots, 7,599 games)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {"files": ["experiments/results/arena/part3/games.csv",
                             "experiments/results/arena/part3/summary.json",
                             "experiments/results/methods/main/summary.json", "runs/*/config.json",
                             "runs/*/cost.json", "runs/sup_6x64/eval.json", "nn_data/solver_300k.json",
                             "runs/arena_part3.log"],
                   "n_games": len(games), "n_bots": len(bots), "openings": len(openings),
                   "elo_method": "Bradley-Terry MLE, cluster bootstrap (pairing x opening), %d reps, seed 0; "
                                 "differences from a separate %d-rep bootstrap (seed 1)" % (args.boot, args.diff_boot),
                   "elo_reproduction": {"max_abs_point_dev": rnd(dev_pt, 3), "max_abs_ci_bound_dev": rnd(dev_ci, 2),
                                        "anchor": "kaggle_negamax = 1000"},
                   "runtime_seconds": None},
        "elo_full": elo_full,
        "elo_top15": elo_top15,
        "elo_top15_meta": {"anchor": "ours_minimax_d4 = 1409", "games": len(g15),
                           "differences_vs_best": diffs_top},
        "elo_sensitivity_no_trivial_openings": sens,
        "head_to_head": head,
        "best_nn": {
            "sense": ("'Best neural net' = highest arena Elo among the neural-network bots in part3 (ours and "
                      "open source), with every bot playing every other from the same 20 openings x 2 colours. "
                      "Tested pairwise (head-to-head score + bootstrap CI) and by bootstrapped Elo differences."),
            "pairwise": [{**h, "elo_diff": next((d for d in diffs_full if d["b"] == h["b"] and d["a"] == BEST), None)}
                         for h in h2h_best],
            "decomposition": decomposition,
            "answer": answer},
        "regressions": regs,
        "predictor_ranking": pred_table,
        "size_contrast": size,
        "health": health,
        "findings": findings,
    }
    doc["source"]["runtime_seconds"] = round(time.time() - t_start, 1)

    roster = {
        "id": "roster", "generated": doc["generated"],
        "source": "experiments/results/arena/part3 (games among the 15 for the refit)",
        "anchor": "ours_minimax_d4 = 1409 (top-15 refit); part3 values anchored at kaggle_negamax = 1000",
        "top15": [{"name": b, "rank_part3": rank_full[b], "rank_top15": next(r["rank"] for r in elo_top15
                                                                             if r["name"] == b),
                   "label": info.get(b, {}).get("label"), "group": info.get(b, {}).get("group"),
                   "author": info.get(b, {}).get("author"), "family": FAMILY.get(b),
                   "search_budget": BUDGET.get(b, "none (1 forward pass)"),
                   "params": params_of(b) if b.startswith("nn_") or b in PARAMS_OPEN else
                   ({"evo_evo_d2_d4": 10}.get(b)),
                   "part3_elo": stored[b]["elo"], "part3_lo": stored[b]["lo"], "part3_hi": stored[b]["hi"],
                   "part3_score": summ["per_bot"][b]["score"], "part3_avg_ms": summ["per_bot"][b]["avg_ms"],
                   "part3_max_ms": summ["per_bot"][b]["max_ms"],
                   "top15_elo": rnd(top[b][0], 1), "top15_lo": rnd(top[b][1], 1), "top15_hi": rnd(top[b][2], 1)}
                  for b in TOP15],
        "dropped": [{"name": b, "rank_part3": rank_full[b], "part3_elo": stored[b]["elo"],
                     "part3_score": summ["per_bot"][b]["score"], "family": FAMILY.get(b), "reason": DROPPED[b]}
                    for b in DROPPED],
    }
    return doc, roster


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=500, help="bootstrap reps for Elo CIs (arena default 500)")
    ap.add_argument("--diff-boot", type=int, default=2000, help="bootstrap reps for Elo differences")
    ap.add_argument("--no-solver", action="store_true", help="skip the per-move solver analysis")
    ap.add_argument("--solver-workers", type=int, default=3)
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="solver cache (JSON, outside the repo)")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    doc, roster = build(a)
    os.makedirs(a.out, exist_ok=True)
    for name, d in (("arena_analysis.json", doc), ("roster.json", roster)):
        p = os.path.join(a.out, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        log("wrote %s (%.0f kB)" % (p, os.path.getsize(p) / 1024))


if __name__ == "__main__":
    main()
