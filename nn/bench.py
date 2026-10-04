"""Benchmarks shared by the training loops and nn/evaluate.py.

Every number here is directly comparable to the minimax results already in
experiments/results/ because it uses the same data and the same metric
definitions as experiments/accuracy.py and experiments/tactics.py:

* Accuracy vs perfect play: data/accuracy_positions.csv (3,000 positions,
  plies 8-30, each column scored by Pascal Pons' solver).
    optimal   chosen column is one of the solver's best columns
    keeps_nl  among positions that are not already lost, the chosen column
              keeps the game-theoretic result (win stays win, draw stays
              draw) - i.e. 1 - blunder rate.
  These positions are NEVER used for training (make_dataset.py removes them
  and their mirror images), so they are an honest held-out test set.
* Value accuracy: does argmax of the WDL head equal the solver's outcome?
* Tactics suite: experiments/tactics_positions.json (win in 1, block, win in
  2, win in 3; 13 positions each).
* Matches: games from random 2-4 ply openings, each opening played twice
  with colours swapped (same protocol as experiments/tournament.py).
"""

import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from engine.board import Board

from . import mcts
from .common import DATA, ROOT

BUCKETS = ("8-11", "12-15", "16-19", "20-23", "24-27", "28-30")


def _sign(v):
    return (v > 0) - (v < 0)


# ------------------------------------------------------------ accuracy set
def load_accuracy_bench(path=None, limit=None):
    path = path or os.path.join(DATA, "accuracy_positions.csv")
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            b = Board.from_moves(r["moves"])
            sc = json.loads(r["scores"])
            legal = [c for c in range(7) if sc[c] is not None]
            best = max(sc[c] for c in legal)
            rows.append({"moves": r["moves"], "ply": int(r["ply"]), "bucket": r["bucket"],
                         "cur": b.current_bits(), "mask": b.mask, "scores": sc,
                         "optimal": set(json.loads(r["optimal_cols"])), "best": best,
                         "outcome": 1 - _sign(best)})   # 0 win, 1 draw, 2 loss (side to move)
            if limit and len(rows) >= limit:
                break
    return rows


def accuracy_summary(bench, cols):
    """cols[i] = column chosen in bench[i].  Returns overall + per-bucket metrics."""
    def summ(idx):
        n = len(idx)
        if n == 0:
            return None
        opt = sum(cols[i] in bench[i]["optimal"] for i in idx)
        nl = [i for i in idx if bench[i]["outcome"] != 2]
        keep = sum(_sign(bench[i]["scores"][cols[i]]) == _sign(bench[i]["best"]) for i in nl)
        return {"n": n, "optimal": round(opt / n, 4),
                "keeps_nl": round(keep / len(nl), 4) if nl else None, "n_not_lost": len(nl)}
    out = summ(list(range(len(bench))))
    out["by_bucket"] = {}
    for bk in BUCKETS:
        s = summ([i for i in range(len(bench)) if bench[i]["bucket"] == bk])
        if s:
            out["by_bucket"][bk] = s
    return out


def policy_eval(evaluator, bench, batch=512):
    """Policy head alone (no search) + value head accuracy on the bench."""
    cols, vacc, wdl_all = [], [], []
    for i in range(0, len(bench), batch):
        chunk = bench[i:i + batch]
        pri, val, wdl = evaluator.evaluate([r["cur"] for r in chunk], [r["mask"] for r in chunk])
        for r, p, w in zip(chunk, pri, wdl):
            legal = [c for c in range(7) if r["scores"][c] is not None]
            cols.append(max(legal, key=lambda c: p[c]))
            vacc.append(int(np.argmax(w)) == r["outcome"])
            wdl_all.append(w)
    res = accuracy_summary(bench, cols)
    res["value_acc"] = round(float(np.mean(vacc)), 4)
    res["value_acc_by_bucket"] = {
        bk: round(float(np.mean([vacc[i] for i in range(len(bench)) if bench[i]["bucket"] == bk])), 4)
        for bk in BUCKETS if any(r["bucket"] == bk for r in bench)}
    # confusion matrix truth x predicted (win, draw, loss)
    conf = [[0] * 3 for _ in range(3)]
    for r, w in zip(bench, wdl_all):
        conf[r["outcome"]][int(np.argmax(w))] += 1
    res["value_confusion"] = conf
    return res


def mcts_eval(evaluator, bench, sims, batch=256, c_puct=1.5):
    """MCTS with ``sims`` simulations per position, many positions searched in parallel."""
    cols = []
    for i in range(0, len(bench), batch):
        chunk = bench[i:i + batch]
        roots = [mcts.Node(r["cur"], r["mask"]) for r in chunk]
        mcts.search(roots, evaluator, sims, c_puct=c_puct)
        cols.extend(mcts.choose_move(rt, 0.0) for rt in roots)
    return accuracy_summary(bench, cols)


def minimax_reference():
    """Minimax accuracy by depth from experiments/results (for the charts)."""
    import csv
    p = os.path.join(ROOT, "experiments", "results", "accuracy_accuracy_summary.csv")
    out = {}
    if os.path.isfile(p):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["bucket"] == "all":
                    out["d" + r["depth"]] = {"optimal": float(r["optimal"]),
                                             "keeps_nl": float(r["keeps_nl"]),
                                             "ms": float(r["mean_ms"])}
    return out


# ------------------------------------------------------------- tactics
def load_tactics():
    with open(os.path.join(ROOT, "experiments", "tactics_positions.json"), encoding="utf-8") as f:
        return json.load(f)["positions"]


def tactics_eval(agent):
    """agent: object with choose(board).  Returns share solved per category."""
    res = {}
    for p in load_tactics():
        b = Board.from_moves(p["moves"])
        col = agent.choose(b)
        cat = res.setdefault(p["category"], [0, 0])
        cat[0] += int(col in p["solution_cols"])
        cat[1] += 1
    out = {k: round(v[0] / v[1], 4) for k, v in res.items()}
    out["all"] = round(sum(v[0] for v in res.values()) / sum(v[1] for v in res.values()), 4)
    return out


# -------------------------------------------------------------- matches
def make_agent(spec):
    """Picklable agent spec -> agent.
      ("minimax", depth)                        project minimax, hand-crafted eval
      ("nn", model_path, "policy")              policy head only
      ("nn", model_path, "mcts", sims)          network + MCTS
      ("nn", model_path, "minimax", depth)      minimax with the network's value as eval
      ("random",)"""
    kind = spec[0]
    if kind == "minimax":
        from engine.agents import MinimaxAgent
        return MinimaxAgent(int(spec[1]))
    if kind == "random":
        from engine.agents import RandomAgent
        return RandomAgent(seed=spec[1] if len(spec) > 1 else None)
    if kind == "nn":
        from .agents import MCTSAgent, NNMinimaxAgent, PolicyAgent
        from .common import load_evaluator
        ev = _cached_evaluator(spec[1], load_evaluator)
        if spec[2] == "policy":
            return PolicyAgent(ev)
        if spec[2] == "mcts":
            return MCTSAgent(ev, sims=int(spec[3]))
        if spec[2] == "minimax":
            return NNMinimaxAgent(ev, int(spec[3]))
    raise ValueError("bad agent spec %r" % (spec,))


_EV_CACHE = {}


def _cached_evaluator(path, loader):
    key = (path, os.path.getmtime(path))
    if key not in _EV_CACHE:
        _EV_CACHE.clear()
        _EV_CACHE[key] = loader(path, "cpu")
    return _EV_CACHE[key]


def spec_name(spec):
    if spec[0] == "minimax":
        return "minimax_d%d" % spec[1]
    if spec[0] == "random":
        return "random"
    if spec[2] == "policy":
        return "nn_policy"
    if spec[2] == "mcts":
        return "nn_mcts%d" % spec[3]
    return "nn_minimax_d%d" % spec[3]


def random_openings(n, seed, min_len=2, max_len=4):
    """Same generator as experiments/common.py (distinct up to mirroring)."""
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        b = Board()
        for _ in range(rng.randint(min_len, max_len)):
            b.drop(rng.choice(b.legal_moves()))
        g = b.to_grid()
        k = min(tuple(map(tuple, g)), tuple(tuple(r[::-1]) for r in g))
        if k in seen:
            continue
        seen.add(k)
        out.append(b.to_move_string())
    return out


def play_one(task):
    """task = (spec_a, spec_b, opening, a_first).  Returns a result dict."""
    spec_a, spec_b, opening, a_first = task
    a, b = make_agent(spec_a), make_agent(spec_b)
    board = Board.from_moves(opening)
    side = {1: a, -1: b} if a_first else {1: b, -1: a}
    t_a = t_b = 0.0
    n_a = n_b = 0
    while True:
        ag = side[board.to_move]
        t0 = time.perf_counter()
        col = ag.choose(board)
        dt = time.perf_counter() - t0
        if ag is a:
            t_a += dt; n_a += 1
        else:
            t_b += dt; n_b += 1
        board.drop(col)
        w = board.check_win()
        if w or board.is_draw():
            break
    a_color = 1 if a_first else -1
    score = 0.5 if w == 0 else (1.0 if w == a_color else 0.0)
    return {"a": spec_name(spec_a), "b": spec_name(spec_b), "opening": opening,
            "a_first": a_first, "score_a": score, "plies": board.moves_played,
            "moves": board.to_move_string(),
            "a_ms": round(1000 * t_a / max(n_a, 1), 2), "b_ms": round(1000 * t_b / max(n_b, 1), 2)}


def run_tasks(tasks, workers):
    if workers <= 1:
        return [play_one(t) for t in tasks]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        return pool.map(play_one, tasks, chunksize=1)


def match(spec_a, spec_b, openings, workers=1):
    tasks = [(spec_a, spec_b, o, af) for o in openings for af in (True, False)]
    games = run_tasks(tasks, workers)
    return summarize_match(games), games


def summarize_match(games):
    w = sum(g["score_a"] == 1.0 for g in games)
    d = sum(g["score_a"] == 0.5 for g in games)
    l = sum(g["score_a"] == 0.0 for g in games)
    n = len(games)
    return {"games": n, "win": w, "draw": d, "loss": l,
            "score": round((w + 0.5 * d) / n, 4) if n else None,
            "a_ms": round(sum(g["a_ms"] for g in games) / n, 2) if n else None,
            "b_ms": round(sum(g["b_ms"] for g in games) / n, 2) if n else None}


# ------------------------------------------------------------------ Elo
def minimax_ladder_elo():
    """Bradley-Terry Elo of minimax d1..d8 from experiments/results/elo.json
    (2,800 games, d1 anchored at 1000)."""
    p = os.path.join(ROOT, "experiments", "results", "elo.json")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return {"minimax_" + k: v["elo"] for k, v in d.get("ratings", {}).items()}


def elo_vs_fixed(results):
    """Maximum-likelihood Elo of one player from games against opponents with
    KNOWN ratings.  results: list of (opponent_elo, score_sum, games).
    Clamped to [opp_min - 800, opp_max + 800] when the player wins/loses
    everything (the MLE is infinite then)."""
    results = [r for r in results if r[2] > 0]
    if not results:
        return None
    lo = min(r[0] for r in results) - 800
    hi = max(r[0] for r in results) + 800

    def grad(R):  # d logL / dR  (decreasing in R)
        g = 0.0
        for opp, s, n in results:
            p = 1.0 / (1.0 + 10 ** ((opp - R) / 400.0))
            g += s - n * p
        return g
    if grad(hi) > 0:
        return hi
    if grad(lo) < 0:
        return lo
    for _ in range(100):
        mid = (lo + hi) / 2
        if grad(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 1)
