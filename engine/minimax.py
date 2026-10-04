"""Negamax with alpha-beta pruning over bitboards.

search(board, depth, time_limit=None, use_tt=True, ordering=True) -> SearchResult

Semantics (identical for every combination of use_tt / ordering, so the
experiments can switch them freely):

* Value = depth-limited negamax value. A move that makes four in a row at
  ply p (root move = ply 1) scores WIN - p for the side making it, so faster
  wins and slower losses are preferred. A full board is 0. At depth 0 the
  static heuristic (engine.heuristic) is applied from the side-to-move view.
* Only *exact* (value-preserving) shortcuts are used:
    - if the side to move has an immediate win, return WIN - (ply+1);
    - with depth >= 2: if the opponent threatens two immediate wins, or every
      move either ignores a threat or plays under an opponent winning cell,
      return -(WIN - (ply+2)); otherwise only the non-losing moves are searched.
  These give the same value plain negamax would compute at that depth.
* Root column = the most central column among those with the maximal value
  (center-first tie-break), regardless of search order. Non-first root moves
  are searched with alpha = best-1 so ties are detected exactly.
* Transposition table: fresh per search() call; an entry only produces a
  cutoff when it was stored at the SAME remaining depth (in a fixed-depth
  search a position always appears at the same remaining depth, so this is
  exactly as strong as plain TT within one iteration, and it guarantees that
  TT on/off returns the same score and column). Entries from shallower
  iterative-deepening iterations are used only for move ordering.
  Bounds are stored as EXACT / LOWER / UPPER.
* nodes counts every negamax call (including the root).
* time_limit: iterative deepening up to ``depth``; the result of the last
  fully completed depth is returned (a partially searched depth is discarded).
* Iterative deepening (with TT) is also used for plain fixed-depth searches
  because TT best-moves from depth d-1 order depth d; it stops early once a
  forced win/loss is proven (deeper search cannot change that result).
"""

import time
from dataclasses import dataclass

from .board import (BOARD_MASK, BOTTOM_MASK, COLUMN_MASK, CENTER_ORDER,
                    CENTER_RANK, COLS, ROWS, H1, winning_cells_mask, alignment)
from .heuristic import evaluate_bits

WIN = 1_000_000
INF = 10 ** 9
MATE_THRESHOLD = WIN - 100   # |score| >= this  ->  forced win/loss found

_EXACT, _LOWER, _UPPER = 0, 1, 2
_FULL = BOARD_MASK


@dataclass
class SearchResult:
    col: int
    score: float
    nodes: int
    ms: float
    depth_reached: int
    random: bool = False      # True when the agent played a random (epsilon) move
    timed_out: bool = False   # True when the time limit cut an iteration short

    def to_dict(self):
        return {"col": self.col, "score": self.score, "nodes": self.nodes,
                "ms": self.ms, "depth": self.depth_reached, "random": self.random}


class _Timeout(Exception):
    pass


def is_win_score(score):
    return abs(score) >= MATE_THRESHOLD


def plies_to_end(score):
    """For a win/loss score, number of plies until the game ends."""
    return int(WIN - abs(score))


def search(board, depth, time_limit=None, use_tt=True, ordering=True, evaluator=None):
    """Pick a move for the side to move on ``board`` (not mutated).

    ``evaluator``: optional leaf evaluation ``f(me_bits, opp_bits) -> number``
    (side-to-move perspective) used at depth 0 instead of the hand-crafted
    heuristic.  It must stay well below ``MATE_THRESHOLD`` in absolute value.
    The neural-network experiments (nn/agents.py) plug a value network in here."""
    if board.is_terminal():
        raise ValueError("cannot search a finished game")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    t0 = time.perf_counter()
    deadline = (t0 + time_limit) if time_limit else None
    s = _Searcher(use_tt, ordering, deadline, evaluator)
    cur = board.current_bits()
    mask = board.mask
    max_depth = min(depth, ROWS * COLS - board.moves_played)

    best = None
    reached = 0
    timed_out = False
    if not use_tt and not time_limit:
        best = s.root(cur, mask, max_depth, -1)
        reached = max_depth
    else:
        prev = -1
        for d in range(1, max_depth + 1):
            try:
                res = s.root(cur, mask, d, prev, allow_abort=(d > 1))
            except _Timeout:
                timed_out = True
                break
            best, reached = res, d
            prev = res[0]
            if is_win_score(res[1]):
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
    ms = (time.perf_counter() - t0) * 1000.0
    return SearchResult(col=best[0], score=float(best[1]), nodes=s.nodes, ms=ms,
                        depth_reached=reached, timed_out=timed_out)


class _Searcher:
    def __init__(self, use_tt, ordering, deadline, evaluator=None):
        self.nodes = 0
        self.evaluate = evaluator or evaluate_bits
        self.tt = {} if use_tt else None
        self.ordering = ordering
        self.deadline = deadline
        self.abort_ok = False
        self.killers = [[-1, -1] for _ in range(ROWS * COLS + 2)]
        self.history = [0] * (COLS * H1)
        self.static = CENTER_ORDER if ordering else tuple(range(COLS))

    # ------------------------------------------------------------------ root
    def root(self, cur, mask, depth, prev_best, allow_abort=False):
        """Returns (col, score). Center-first tie-break among equal values."""
        self.abort_ok = allow_abort
        self.nodes += 1
        opp = cur ^ mask
        possible = (mask + BOTTOM_MASK) & BOARD_MASK
        wins = winning_cells_mask(cur, mask) & possible
        if wins:
            return (_first_col(wins), WIN - 1)
        if depth >= 2:
            ow = winning_cells_mask(opp, mask)
            forced = possible & ow
            if forced:
                if forced & (forced - 1):
                    return (_first_col(forced), -(WIN - 2))
                possible = forced
            safe = possible & ~(ow >> 1)
            if not safe:
                return (_first_col(possible), -(WIN - 2))
            possible = safe
        cols = [c for c in self.static if possible & COLUMN_MASK[c]]
        if self.ordering and prev_best in cols:
            cols.remove(prev_best)
            cols.insert(0, prev_best)
        best = -INF
        best_col = cols[0]
        neg = self.negamax
        for i, c in enumerate(cols):
            mv = possible & COLUMN_MASK[c]
            nm = mask | mv
            if nm == _FULL:
                v = 0
            elif i == 0:
                v = -neg(opp, nm, depth - 1, -INF, INF, 1)
            else:
                v = -neg(opp, nm, depth - 1, -INF, -(best - 1), 1)
            if v > best or (v == best and CENTER_RANK[c] < CENTER_RANK[best_col]):
                best = v
                best_col = c
        return (best_col, best)

    # -------------------------------------------------------------- negamax
    def negamax(self, cur, mask, depth, alpha, beta, ply):
        self.nodes += 1
        if self.deadline is not None and not (self.nodes & 1023) and self.abort_ok \
                and time.perf_counter() >= self.deadline:
            raise _Timeout()
        if mask == _FULL:
            return 0
        opp = cur ^ mask
        if depth == 0:
            return self.evaluate(cur, opp)
        possible = (mask + BOTTOM_MASK) & BOARD_MASK
        if winning_cells_mask(cur, mask) & possible:
            return WIN - ply - 1
        if depth >= 2:
            ow = winning_cells_mask(opp, mask)
            forced = possible & ow
            if forced:
                if forced & (forced - 1):
                    return -(WIN - ply - 2)
                possible = forced
            possible &= ~(ow >> 1)
            if not possible:
                return -(WIN - ply - 2)

        alpha_orig = alpha
        beta_orig = beta
        tt = self.tt
        tt_move = -1
        key = cur + mask
        if tt is not None:
            e = tt.get(key)
            if e is not None:
                if e[0] == depth:
                    flag = e[1]
                    val = e[2]
                    if flag == _EXACT:
                        return val
                    if flag == _LOWER:
                        if val > alpha:
                            alpha = val
                    elif val < beta:
                        beta = val
                    if alpha >= beta:
                        return val
                tt_move = e[3]

        # ---- move ordering
        if self.ordering:
            hist = self.history
            killers = self.killers[ply]
            scored = []
            for c in CENTER_ORDER:
                mv = possible & COLUMN_MASK[c]
                if mv:
                    if c == tt_move:
                        k = 1 << 40
                    elif c == killers[0]:
                        k = 1 << 39
                    elif c == killers[1]:
                        k = 1 << 38
                    else:
                        k = hist[mv.bit_length() - 1]
                    scored.append((k, -CENTER_RANK[c], c, mv))
            scored.sort(reverse=True)
            moves = [(c, mv) for _, _, c, mv in scored]
        else:
            moves = [(c, possible & COLUMN_MASK[c]) for c in range(COLS)
                     if possible & COLUMN_MASK[c]]

        best = -INF
        best_col = moves[0][0]
        d1 = depth - 1
        p1 = ply + 1
        for c, mv in moves:
            v = -self.negamax(opp, mask | mv, d1, -beta, -alpha, p1)
            if v > best:
                best = v
                best_col = c
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        if self.ordering:
                            ks = self.killers[ply]
                            if ks[0] != c:
                                ks[1] = ks[0]
                                ks[0] = c
                            self.history[mv.bit_length() - 1] += depth * depth
                        break

        if tt is not None:
            if best <= alpha_orig:
                flag = _UPPER
            elif best >= beta_orig:
                flag = _LOWER
            else:
                flag = _EXACT
            tt[key] = (depth, flag, best, best_col)
        return best


def _first_col(bits):
    """Most central column that has a bit set in ``bits``."""
    for c in CENTER_ORDER:
        if bits & COLUMN_MASK[c]:
            return c
    raise ValueError("no column")


def plain_negamax(board, depth):
    """Reference implementation: plain depth-limited negamax WITHOUT pruning,
    TT or shortcuts (exponential; use only for tiny depths in tests).
    Returns (col, score) with the same scoring and center-first tie-break."""
    cur = board.current_bits()
    mask = board.mask

    def rec(cur, mask, depth, ply):
        if mask == _FULL:
            return 0
        opp = cur ^ mask
        if depth == 0:
            return evaluate_bits(cur, opp)
        best = -INF
        for c in range(COLS):
            if mask & (1 << (c * H1 + ROWS - 1)):
                continue
            mv = (mask + (1 << (c * H1))) & COLUMN_MASK[c]
            if alignment(cur | mv):
                v = WIN - ply - 1
            else:
                v = -rec(opp, mask | mv, depth - 1, ply + 1)
            best = max(best, v)
        return best

    best, best_col = -INF, None
    for c in CENTER_ORDER:
        if mask & (1 << (c * H1 + ROWS - 1)):
            continue
        mv = (mask + (1 << (c * H1))) & COLUMN_MASK[c]
        if alignment(cur | mv):
            v = WIN - 1
        else:
            v = -rec(cur ^ mask, mask | mv, depth - 1, 1)
        if v > best:
            best, best_col = v, c
    return best_col, best
