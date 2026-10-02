"""Agents: RandomAgent, MinimaxAgent(depth, epsilon), SolverAgent, slider levels."""

import os
import random
import shutil
import subprocess
import time

from .board import CENTER_ORDER, COLS
from .minimax import SearchResult, search

# level -> (search depth, random-move probability)  [project statement 4.2]
LEVELS = {1: (1, 0.5), 2: (1, 0.2), 3: (2, 0.1), 4: (3, 0), 5: (4, 0),
          6: (5, 0), 7: (6, 0), 8: (7, 0), 9: (8, 0), 10: (20, 0)}
# Level 10 = iterative deepening up to depth 20 with a time limit (seconds).
# 1.9 s (not 2.0) leaves headroom for HTTP/JSON overhead so a move still
# arrives in < ~2 s.
LEVEL_TIME_LIMITS = {10: 1.9}


class RandomAgent:
    """Uniformly random legal move (seeded)."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.name = "random"
        self.last_result = None

    def choose(self, board):
        moves = board.legal_moves()
        if not moves:
            raise ValueError("no legal moves")
        col = self.rng.choice(moves)
        self.last_result = SearchResult(col=col, score=0.0, nodes=0, ms=0.0,
                                        depth_reached=0, random=True)
        return col


class MinimaxAgent:
    """Negamax/alpha-beta agent. With probability ``epsilon`` it plays a
    uniformly random legal move instead (seeded RNG; ``last_result.random``
    is then True, nodes=0, depth_reached=0, score=0.0)."""

    def __init__(self, depth, epsilon=0.0, time_limit=None, seed=None,
                 use_tt=True, ordering=True):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.depth = depth
        self.epsilon = epsilon
        self.time_limit = time_limit
        self.use_tt = use_tt
        self.ordering = ordering
        self.rng = random.Random(seed)
        self.last_result = None
        self.name = "d%d" % depth + ("_e%g" % epsilon if epsilon else "") + \
            ("_t%gs" % time_limit if time_limit else "")

    def choose(self, board):
        moves = board.legal_moves()
        if not moves:
            raise ValueError("no legal moves")
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            t0 = time.perf_counter()
            col = self.rng.choice(moves)
            self.last_result = SearchResult(
                col=col, score=0.0, nodes=0, ms=(time.perf_counter() - t0) * 1000.0,
                depth_reached=0, random=True)
            return col
        self.last_result = search(board, self.depth, time_limit=self.time_limit,
                                  use_tt=self.use_tt, ordering=self.ordering)
        return self.last_result.col


def agent_for_level(level, seed=None):
    """Slider level 1..10 -> MinimaxAgent."""
    if level not in LEVELS:
        raise ValueError("level must be an int in 1..10")
    depth, eps = LEVELS[level]
    return MinimaxAgent(depth, epsilon=eps, time_limit=LEVEL_TIME_LIMITS.get(level),
                        seed=seed)


# ---------------------------------------------------------------- solver
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOLVER = os.path.join(_ROOT, "tools", "pons", "c4solver")
DEFAULT_BOOK = os.path.join(_ROOT, "tools", "pons", "7x6.book")
INVALID_MOVE = -1000


class SolverUnavailable(RuntimeError):
    """Raised when the Pons solver binary cannot be found or run."""


class SolverAgent:
    """Perfect-play agent wrapping Pascal Pons' solver (tools/pons/c4solver).

    The binary reads one Pons move string per line on stdin ("4453",
    1-indexed columns) and, with ``-a``, prints the line followed by 7
    scores (side-to-move perspective; positive = win, larger = sooner;
    -1000 = column not playable).  Without the opening book (7x6.book next
    to the binary) early positions (< ~12 stones) can take minutes to solve.

    scores(board)        -> list of 7 ints or None (illegal column)
    scores_many(boards)  -> batched version (one process for all positions)
    choose(board)        -> best column, center-first among ties
    """

    def __init__(self, path=None, weak=False, timeout=120.0, book=None):
        path = path or os.environ.get("C4SOLVER") or DEFAULT_SOLVER
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            found = shutil.which("c4solver")
            if found is None:
                raise SolverUnavailable(
                    "Pons solver binary not found at %s (build it in tools/pons, "
                    "or set C4SOLVER=/path/to/c4solver)" % path)
            path = found
        self.path = path
        self.weak = weak
        self.timeout = timeout
        if book is None and os.path.isfile(DEFAULT_BOOK):
            book = DEFAULT_BOOK
        self.book = book
        self.name = "solver"
        self.last_result = None
        self.last_scores = None

    def _cmd(self):
        cmd = [self.path, "-a"]
        if self.weak:
            cmd.append("-w")
        if self.book:
            cmd += ["-b", self.book]
        return cmd

    def scores_many(self, boards):
        lines = []
        for b in boards:
            if b.is_terminal():
                raise ValueError("cannot solve a finished game")
            lines.append(b.to_move_string())
        try:
            proc = subprocess.run(self._cmd(), input="\n".join(lines) + "\n",
                                  capture_output=True, text=True,
                                  timeout=self.timeout,
                                  cwd=os.path.dirname(self.path))
        except (OSError, subprocess.SubprocessError) as e:
            raise SolverUnavailable("failed to run solver: %s" % e) from e
        out = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if len(out) != len(lines):
            raise SolverUnavailable("solver returned %d lines for %d positions; stderr: %s"
                                    % (len(out), len(lines), proc.stderr.strip()[:500]))
        result = []
        for ln in out:
            parts = ln.split()
            nums = [int(x) for x in parts[-COLS:]]
            result.append([None if v == INVALID_MOVE else v for v in nums])
        return result

    def scores(self, board):
        return self.scores_many([board])[0]

    def choose(self, board):
        t0 = time.perf_counter()
        sc = self.scores(board)
        self.last_scores = sc
        best = max(v for v in sc if v is not None)
        col = next(c for c in CENTER_ORDER if sc[c] == best)
        self.last_result = SearchResult(col=col, score=float(best), nodes=0,
                                        ms=(time.perf_counter() - t0) * 1000.0,
                                        depth_reached=42 - board.moves_played)
        return col
