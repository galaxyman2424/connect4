"""Exact endgame for the net's search: positions late in the game are solved by the C alpha-beta.

Measured on the 3,000-position benchmark (C minimax = engine/cminimax, python -m engine.cbuild):
    depth 24 from ply 16:  plies 16-19 99.8% optimal (46 ms mean, <1 s max), plies 20-30 100%
while the 8x128 score net alone gets 95.8% (16-19) and 98.6-99.2% (20-30).  So the net's
remaining mistakes after ply 16 can simply be removed by searching to the end.

Two ways it is used:
  * ExactTailEvaluator(net, from_ply)  wraps any evaluator for MCTS.  Positions with
    ply >= from_ply get their value from an exact search (win in p plies -> 1 - p/100,
    loss -> -(1 - p/100), draw -> 0) and are marked exact, so mcts.search() treats them as
    solved leaves (no further expansion).  The net still gives the move priors.
  * HybridAgent(net, root_from=16)  plays the exact search's move once the game reaches
    ply root_from, and MCTS (with exact leaves) or the net's policy before that.

Only exact results are used: a search that cannot reach the end of the game within
``max_depth`` and does not prove a win/loss falls back to the net's value.
"""

import time

import numpy as np

from engine.minimax import MATE_THRESHOLD, WIN

N_CELLS = 42


def _popcount(x):
    return bin(int(x)).count("1")


def exact_value(cur, mask, max_depth=26):
    """(value in [-1, 1], column, exact?) for the side to move, by the C (or Python) alpha-beta."""
    from engine.board import Board
    b = Board.from_bits(int(cur), int(mask)) if hasattr(Board, "from_bits") else _board_from_bits(int(cur), int(mask))
    ply = _popcount(mask)
    depth = min(max_depth, N_CELLS - ply)
    from engine import minimax
    r = minimax.search(b, depth)
    if abs(r.score) >= MATE_THRESHOLD:
        p = WIN - abs(r.score)
        v = (1.0 - p / 100.0) * (1 if r.score > 0 else -1)
        return v, r.col, True
    if depth >= N_CELLS - ply:            # searched to the end of every line: no win -> draw
        return 0.0, r.col, True
    return None, r.col, False


def _board_from_bits(cur, mask):
    """engine.board.Board with side-to-move stones ``cur`` and all stones ``mask``."""
    from engine.board import Board
    b = Board()
    ply = _popcount(mask)
    first = ply % 2 == 0                  # side to move is player 1
    opp = cur ^ mask
    b.p1, b.p2 = (cur, opp) if first else (opp, cur)
    b.mask = mask
    b.to_move = 1 if first else -1
    b.moves_played = ply
    if hasattr(b, "history"):
        b.history = []
    return b


class ExactTailEvaluator:
    def __init__(self, evaluator, from_ply=20, max_depth=26, cache=True):
        from .agents import CachedEvaluator
        self.ev = CachedEvaluator(evaluator) if cache else evaluator
        self.meta = getattr(evaluator, "meta", {})
        self.from_ply = int(from_ply)
        self.max_depth = int(max_depth)
        self.exact_cache = {}
        self.last_exact = None
        self.solved = 0
        self.solve_ms = 0.0

    def evaluate(self, curs, masks):
        pri, val, wdl = self.ev.evaluate(curs, masks)
        val = np.array(val, dtype=np.float64)
        exact = np.zeros(len(val), bool)
        for i, (c, m) in enumerate(zip(curs, masks)):
            if _popcount(m) < self.from_ply:
                continue
            key = (int(c), int(m))
            got = self.exact_cache.get(key)
            if got is None:
                t0 = time.perf_counter()
                v, _, ok = exact_value(c, m, self.max_depth)
                self.solve_ms += (time.perf_counter() - t0) * 1000
                self.solved += 1
                got = (v, ok)
                self.exact_cache[key] = got
            if got[1]:
                val[i] = got[0]
                exact[i] = True
        self.last_exact = exact
        return pri, val, wdl


class HybridAgent:
    """Exact search from ply ``root_from``; before that MCTS (sims > 0, with exact leaves from
    ply ``leaf_from``) or the net's policy (sims = 0)."""

    def __init__(self, evaluator, root_from=16, sims=200, leaf_from=20, max_depth=26, name=None):
        from .agents import MCTSAgent, PolicyAgent
        self.root_from, self.max_depth = int(root_from), int(max_depth)
        self.tail = ExactTailEvaluator(evaluator, leaf_from, max_depth)
        self.inner = MCTSAgent(self.tail, sims=sims, cache=False) if sims else PolicyAgent(evaluator)
        self.name = name or "nn_hybrid%d_%s" % (root_from, ("mcts%d" % sims) if sims else "policy")
        self.last_result = None
        self.last_info = {}

    def choose(self, board):
        if board.moves_played >= self.root_from:
            from engine import minimax
            from engine.minimax import SearchResult
            t0 = time.perf_counter()
            r = minimax.search(board, min(self.max_depth, N_CELLS - board.moves_played))
            self.last_result = SearchResult(col=r.col, score=r.score, nodes=r.nodes,
                                            ms=(time.perf_counter() - t0) * 1000, depth_reached=r.depth_reached)
            self.last_info = {"mode": "exact_search", "depth": r.depth_reached}
            return r.col
        col = self.inner.choose(board)
        self.last_result = self.inner.last_result
        self.last_info = dict(getattr(self.inner, "last_info", {}), mode="net")
        return col
