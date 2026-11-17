"""Fixed gauntlet + speed measurement for sweep students (engine.board only, no arena/).

The student plays its policy head with no search (argmax over legal columns, centre
tie-break - nn.agents.PolicyAgent).  Opponents (all deterministic):
  mm_d1, mm_d2, mm_d4, mm_d6   engine minimax, hand heuristic
  mm_d4_evo60                  engine minimax depth 4, evolved weights of runs/evo_d2 gen 60
  distill_az_2x32              runs/distill_az_2x32/best.npz, policy only
  sup_6x64                     runs/sup_6x64/best.npz, policy only
Openings: nn.bench.random_openings(N_OPENINGS, OPENING_SEED) (2-4 random plies,
distinct up to mirroring); each opening is played twice with colours swapped.
Because every player is deterministic, a (student, opponent, opening, colour) game
always has the same result: the CI below is over openings, not over randomness.
"""

import math
import multiprocessing as mp
import os
import time

import numpy as np

from nn.common import ROOT

N_OPENINGS = 40
OPENING_SEED = 4150
MAX_WORKERS = 8
OPPONENTS = ("mm_d1", "mm_d2", "mm_d4", "mm_d6", "mm_d4_evo60", "distill_az_2x32", "sup_6x64")
_NETS = {"distill_az_2x32": "runs/distill_az_2x32/best.npz", "sup_6x64": "runs/sup_6x64/best.npz"}

_CACHE = {}


def openings(n=N_OPENINGS, seed=OPENING_SEED):
    from nn.bench import random_openings
    return random_openings(n, seed)


def _single_thread():
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"


def make_player(spec):
    """'net:<path>' | 'mm_dN' | 'mm_d4_evo60' | 'distill_az_2x32' | 'sup_6x64' -> agent."""
    if spec in _CACHE:
        return _CACHE[spec]
    if spec.startswith("net:") or spec in _NETS:
        from nn.agents import PolicyAgent
        from nn.numpy_net import NumpyNet
        p = spec[4:] if spec.startswith("net:") else _NETS[spec]
        ag = PolicyAgent(NumpyNet(p if os.path.isabs(p) else os.path.join(ROOT, p)), name=spec)
    elif spec == "mm_d4_evo60":
        from engine.agents import MinimaxAgent
        from research.teachers import evo_weights
        from rl.features import WeightedEvaluator
        ag = MinimaxAgent(4, evaluator=WeightedEvaluator(evo_weights(60)))
    elif spec.startswith("mm_d"):
        from engine.agents import MinimaxAgent
        ag = MinimaxAgent(int(spec[4:]))
    else:
        raise ValueError(spec)
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[spec] = ag
    return ag


def play(task):
    """task = (student_spec, opponent_spec, opening, student_first) -> score for the student."""
    from engine.board import Board
    s_spec, o_spec, opening, s_first = task
    stu, opp = make_player(s_spec), make_player(o_spec)
    b = Board.from_moves(opening)
    # side to move after the opening is b.to_move; the student is +1 if it plays first
    s_colour = 1 if s_first else -1
    while True:
        ag = stu if b.to_move == s_colour else opp
        b.drop(ag.choose(b))
        w = b.check_win()
        if w or b.is_draw():
            break
    return 0.5 if w == 0 else (1.0 if w == s_colour else 0.0)


def _init_worker():
    _single_thread()


def ci95(scores):
    """Normal-approximation 95 % CI of a mean score (games are not independent: each
    opening is played twice - treat as a rough interval)."""
    n = len(scores)
    if n == 0:
        return None
    m = float(np.mean(scores))
    sd = float(np.std(scores, ddof=1)) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n)
    return [round(max(0.0, m - h), 4), round(min(1.0, m + h), 4)]


def gauntlet(model_path, workers=MAX_WORKERS, opponents=OPPONENTS, n_openings=N_OPENINGS):
    _single_thread()
    ops = openings(n_openings)
    spec = "net:" + model_path
    tasks = [(spec, o, op, f) for o in opponents for op in ops for f in (True, False)]
    t0 = time.time()
    workers = max(1, min(int(workers), MAX_WORKERS))
    if workers == 1:
        res = [play(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            res = pool.map(play, tasks, chunksize=2)
    out = {}
    allsc = []
    for o in opponents:
        sc = [r for t, r in zip(tasks, res) if t[1] == o]
        allsc += sc
        out[o] = {"games": len(sc), "win": sum(s == 1.0 for s in sc), "draw": sum(s == 0.5 for s in sc),
                  "loss": sum(s == 0.0 for s in sc), "score": round(float(np.mean(sc)), 4), "ci95": ci95(sc)}
    return {"opponents": out, "mean_score": round(float(np.mean([v["score"] for v in out.values()])), 4),
            "games": len(allsc), "ci95": ci95(allsc), "openings": n_openings,
            "opening_seed": OPENING_SEED, "sec": round(time.time() - t0, 1)}


def ms_per_position(model_path, n=300):
    """Median ms of one single-position forward pass (numpy, single thread)."""
    _single_thread()
    from engine.board import Board
    from nn.numpy_net import NumpyNet
    net = NumpyNet(model_path)
    b = Board.from_moves("4453")
    cur, mask = [b.current_bits()], [b.mask]
    for _ in range(20):
        net.evaluate(cur, mask)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        net.evaluate(cur, mask)
        ts.append(time.perf_counter() - t0)
    return round(1000 * float(np.median(ts)), 4)
