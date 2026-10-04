"""Batched PUCT Monte Carlo Tree Search (the search used by AlphaZero).

One simulation = walk down the tree from the root, at each node picking the
move with the highest

    Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))

(Q = average value of that move so far, P = the network's prior for it,
N = visit counts), until we reach a position we have not evaluated yet. The
network evaluates it once (prior for its moves + value), and that value is
backed up the path, flipping sign every ply (what is good for me is bad for
you). Wins/draws inside the tree are scored exactly (+1 / 0), so MCTS sees
forced wins without any hand-written rules.

After ``sims`` simulations the move is chosen from the root visit counts
(most visited, or sampled with a temperature during self-play), and the visit
distribution is the policy training target.

Batching: ``search()`` takes a list of roots (one per game) and runs the
simulations for all of them in lock-step, so each network call evaluates up
to len(roots) positions at once. That keeps the GPU busy during self-play.

Positions are (cur, mask) bitboard pairs, see nn/encode.py.
"""

import math

import numpy as np

from engine.board import BOARD_MASK, BOTTOM_COL, CENTER_ORDER, COLUMN_MASK, TOP_MASK, alignment

COLS = 7
_CENTER_RANK = {c: i for i, c in enumerate(CENTER_ORDER)}


class Node:
    __slots__ = ("cur", "mask", "terminal", "expanded", "legal", "P", "P_raw",
                 "N", "W", "children", "n", "value")

    def __init__(self, cur, mask, terminal=None):
        self.cur = cur
        self.mask = mask
        self.terminal = terminal      # None, or exact value for the side to move (-1 lost, 0 draw)
        self.expanded = False
        self.legal = None
        self.P = None
        self.P_raw = None
        self.N = [0] * COLS
        self.W = [0.0] * COLS
        self.children = [None] * COLS
        self.n = 0                    # total visits through this node's edges
        self.value = 0.0              # network value of this node (side to move)

    def q(self):
        """Average backed-up value of the root from the side-to-move view."""
        return sum(self.W) / self.n if self.n else self.value

    def child(self, a):
        ch = self.children[a]
        if ch is None:
            mv = (self.mask + BOTTOM_COL[a]) & COLUMN_MASK[a]
            me = self.cur | mv
            nmask = self.mask | mv
            if alignment(me):
                t = -1.0                  # the player who moved just won -> side to move lost
            elif nmask == BOARD_MASK:
                t = 0.0
            else:
                t = None
            ch = Node(me ^ nmask, nmask, t)
            self.children[a] = ch
        return ch


def root_from_board(board):
    """engine.board.Board -> Node (raises if the game is over)."""
    if board.is_terminal():
        raise ValueError("game is over")
    return Node(board.current_bits(), board.mask)


def legal_columns(mask):
    return [c for c in range(COLS) if not mask & TOP_MASK[c]]


def _set_priors(node, pri, value):
    legal = legal_columns(node.mask)
    p = [0.0] * COLS
    s = 0.0
    for c in legal:
        s += float(pri[c])
    if s <= 0:
        for c in legal:
            p[c] = 1.0 / len(legal)
    else:
        for c in legal:
            p[c] = float(pri[c]) / s
    node.legal = legal
    node.P = p
    node.P_raw = list(p)
    node.value = float(value)
    node.expanded = True


def _select(node, c_puct, fpu):
    sq = c_puct * math.sqrt(node.n if node.n > 0 else 1)
    best_a = -1
    best = -1e18
    N, W, P = node.N, node.W, node.P
    for a in node.legal:
        n = N[a]
        u = (W[a] / n if n else fpu) + sq * P[a] / (1 + n)
        if u > best:
            best = u
            best_a = a
    return best_a


def _backup(path, v):
    """v = value for the side to move at the end of ``path``."""
    val = -v
    for node, a in reversed(path):
        node.N[a] += 1
        node.W[a] += val
        node.n += 1
        val = -val


def add_dirichlet(node, rng, alpha, eps):
    """Root exploration noise (self-play only)."""
    if not node.expanded or len(node.legal) < 2:
        return
    noise = rng.dirichlet([alpha] * len(node.legal))
    for k, c in enumerate(node.legal):
        node.P[c] = (1 - eps) * node.P_raw[c] + eps * float(noise[k])


def expand_roots(roots, evaluator):
    need = [r for r in roots if not r.expanded and r.terminal is None]
    if need:
        pri, val, _ = evaluator.evaluate([r.cur for r in need], [r.mask for r in need])
        for r, p, v in zip(need, pri, val):
            _set_priors(r, p, v)


def search(roots, evaluator, sims, c_puct=1.5, fpu=0.0):
    """Run ``sims`` simulations on every root (in place).  Returns the number
    of network evaluations made."""
    expand_roots(roots, evaluator)
    evals = 0
    live = [r for r in roots if r.terminal is None]
    for _ in range(sims):
        leaves = []
        paths = []
        for root in live:
            node = root
            path = []
            while True:
                a = _select(node, c_puct, fpu)
                path.append((node, a))
                ch = node.child(a)
                if ch.terminal is not None:
                    _backup(path, ch.terminal)
                    break
                if not ch.expanded:
                    leaves.append(ch)
                    paths.append(path)
                    break
                node = ch
        if leaves:
            pri, val, _ = evaluator.evaluate([l.cur for l in leaves], [l.mask for l in leaves])
            evals += len(leaves)
            for leaf, p, v, path in zip(leaves, pri, val, paths):
                _set_priors(leaf, p, v)
                _backup(path, float(v))
    return evals


def visit_policy(node):
    """Visit-count distribution over the 7 columns (training target)."""
    tot = sum(node.N)
    if tot == 0:
        return [1.0 / len(node.legal) if c in node.legal else 0.0 for c in range(COLS)]
    return [n / tot for n in node.N]


def choose_move(node, temperature=0.0, rng=None):
    """temperature 0: most visited (center-first tie-break); else sample
    proportional to N^(1/T)."""
    if temperature <= 1e-6 or rng is None:
        best = max(node.N[c] for c in node.legal)
        return min((c for c in node.legal if node.N[c] == best), key=lambda c: _CENTER_RANK[c])
    w = np.array([node.N[c] ** (1.0 / temperature) for c in node.legal], dtype=np.float64)
    if w.sum() <= 0:
        return node.legal[int(rng.integers(len(node.legal)))]
    return node.legal[int(rng.choice(len(node.legal), p=w / w.sum()))]


def principal_variation(node, max_len=8):
    pv = []
    while node is not None and node.expanded and node.n > 0 and len(pv) < max_len:
        a = max(node.legal, key=lambda c: (node.N[c], -_CENTER_RANK[c]))
        pv.append(a)
        node = node.children[a]
    return pv
