"""Opponent's best reply after a network's move, via Pons' solver (bounded use).

For position i and the net's chosen column c: play c, and if the game is not over,
solve the successor (opponent to move) with tools/solver.py.  The label is the set of
the opponent's optimal columns.  Runs in <= `workers` processes, results cached by the
successor's move string.
"""

import multiprocessing as mp
import os

import numpy as np

from engine.board import Board
from nn.encode import bits_to_grid

_SOLVER = None


def _init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    global _SOLVER
    from tools.solver import Solver
    _SOLVER = Solver(timeout=10)


def _solve(ms):
    from tools.solver import SolverError
    try:
        return ms, _SOLVER.solve_all(ms)
    except SolverError:
        return ms, None


def successor_strings(cur, mask, cols):
    """Move strings of the successor positions (None if the move ends the game)."""
    me = bits_to_grid(np.asarray(cur, dtype=np.uint64))
    th = bits_to_grid(np.asarray(cur, dtype=np.uint64) ^ np.asarray(mask, dtype=np.uint64))
    out = []
    for i in range(len(cols)):
        first = me[i].sum() == th[i].sum()
        g = (me[i].astype(int) - th[i].astype(int)) * (1 if first else -1)
        b = Board.from_grid(g.tolist())
        b.drop(int(cols[i]))
        if b.check_win() or b.is_full():
            out.append(None)
        else:
            out.append(b.to_move_string())
    return out


def opponent_best_replies(cur, mask, cols, workers=4, cache=None):
    """Returns (opt (N,7) bool, ok (N,) bool, n_solved_new)."""
    cache = {} if cache is None else cache
    strs = successor_strings(cur, mask, cols)
    todo = sorted({s for s in strs if s is not None and s not in cache})
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init) as pool:
            for ms, sc in pool.imap_unordered(_solve, todo, chunksize=16):
                cache[ms] = sc
    opt = np.zeros((len(strs), 7), bool)
    ok = np.zeros(len(strs), bool)
    for i, s in enumerate(strs):
        if s is None or cache.get(s) is None:
            continue
        sc = cache[s]
        best = max(x for x in sc if x is not None)
        opt[i] = [x is not None and x == best for x in sc]
        ok[i] = True
    return opt, ok, len(todo)
