"""Many Connect 4 games at once, on numpy uint64 bitboards.

Same bit layout as engine/board.py (column c owns bits c*7..c*7+6, bit c*7+6
is an always-empty sentinel).  A position is (cur, mask): stones of the side
to move, and all stones.  Everything here works on arrays of positions, so a
training loop can step 1,024 games with a handful of numpy calls.
"""

import numpy as np

from engine.board import BOARD_MASK, BOTTOM_MASK, COLS, H1, ROWS

U = np.uint64
_BOTTOM_COL = np.array([1 << (c * H1) for c in range(COLS)], dtype=U)
_COLUMN_MASK = np.array([((1 << ROWS) - 1) << (c * H1) for c in range(COLS)], dtype=U)
_TOP_BITS = np.array([c * H1 + ROWS - 1 for c in range(COLS)], dtype=U)
_BOARD = U(BOARD_MASK)
_BOTTOM = U(BOTTOM_MASK)
_SHIFTS = [(U(H1), U(2 * H1)), (U(H1 - 1), U(2 * (H1 - 1))),
           (U(H1 + 1), U(2 * (H1 + 1))), (U(1), U(2))]


def alignment(pos):
    """Bool array: does each bitboard contain four in a row?"""
    pos = np.asarray(pos, dtype=U)
    out = np.zeros(pos.shape, dtype=bool)
    for s1, s2 in _SHIFTS:
        m = pos & (pos >> s1)
        out |= (m & (m >> s2)) != 0
    return out


def legal(masks):
    """(B,) masks -> (B, 7) bool: column not full."""
    m = np.asarray(masks, dtype=U)
    return ((m[:, None] >> _TOP_BITS[None, :]) & U(1)) == 0


def move_bits(masks, cols):
    """Bit of the stone that would land in column cols[i] of masks[i]."""
    c = np.asarray(cols, dtype=np.int64)
    return (np.asarray(masks, dtype=U) + _BOTTOM_COL[c]) & _COLUMN_MASK[c]


def play(curs, masks, cols):
    """Play cols for the side to move.  Returns (next_cur, next_mask, won, full):
    next_cur is the OPPONENT's stones (they move next), won = the mover made
    four in a row, full = board is now full (draw unless won)."""
    curs = np.asarray(curs, dtype=U)
    masks = np.asarray(masks, dtype=U)
    mv = move_bits(masks, cols)
    me = curs | mv
    nmask = masks | mv
    return me ^ nmask, nmask, alignment(me), nmask == _BOARD


def afterstates(curs, masks):
    """All 7 moves of every position at once.
    Returns mover (B,7), mask (B,7), legal (B,7), won (B,7), full (B,7);
    mover = stones of the player who just moved (after the move)."""
    curs = np.asarray(curs, dtype=U)
    masks = np.asarray(masks, dtype=U)
    lg = legal(masks)
    mv = (masks[:, None] + _BOTTOM_COL[None, :]) & _COLUMN_MASK[None, :]
    mover = curs[:, None] | mv
    nmask = masks[:, None] | mv
    won = alignment(mover) & lg
    full = (nmask == _BOARD) & lg
    return mover, nmask, lg, won, full


def winning_moves(curs, masks):
    """(B,7) bool: playing this column wins immediately."""
    return afterstates(curs, masks)[3]


def popcount(x):
    x = np.asarray(x, dtype=U)
    if hasattr(np, "bitwise_count"):                       # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int64)
    return np.array([int(v).bit_count() for v in x.ravel()], dtype=np.int64).reshape(x.shape)


def mirror(bb):
    """Mirror bitboards left-right (column c -> 6 - c)."""
    bb = np.asarray(bb, dtype=U)
    out = np.zeros_like(bb)
    colmask = U((1 << H1) - 1)
    for c in range(COLS):
        out |= ((bb >> U(c * H1)) & colmask) << U((COLS - 1 - c) * H1)
    return out


def masked_sample(probs, legal_mask, rng):
    """Sample one legal column per row from (B,7) probabilities."""
    p = np.where(legal_mask, probs, 0.0)
    s = p.sum(1, keepdims=True)
    p = np.where(s > 0, p / np.where(s > 0, s, 1), legal_mask / legal_mask.sum(1, keepdims=True))
    u = rng.random((p.shape[0], 1))
    idx = (np.cumsum(p, 1) < u).sum(1)
    idx = np.minimum(idx, COLS - 1)
    # guard against rounding landing on an illegal column
    bad = ~legal_mask[np.arange(len(idx)), idx]
    if bad.any():
        idx[bad] = np.argmax(legal_mask[bad], 1)
    return idx


def random_legal(legal_mask, rng):
    return masked_sample(np.ones(legal_mask.shape), legal_mask, rng)


def argmax_center(values, legal_mask):
    """Argmax over legal columns, ties broken toward the centre (like the engine)."""
    order = np.array([3, 2, 4, 1, 5, 0, 6])
    v = np.where(legal_mask, values, -np.inf)[:, order]
    return order[np.argmax(v, 1)]


class VecGames:
    """B independent games in lock-step.  Finished games restart automatically
    (from an empty board or a short random opening) when reset_done() is called.

    Attributes: cur, mask (uint64 arrays), ply (int array), and per game the
    first player is the one to move when ply is even."""

    def __init__(self, n, rng, random_opening=(0, 0)):
        self.n = n
        self.rng = rng
        self.random_opening = random_opening
        self.cur = np.zeros(n, dtype=U)
        self.mask = np.zeros(n, dtype=U)
        self.ply = np.zeros(n, dtype=np.int64)
        self.reset(np.arange(n))

    def reset(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            return
        self.cur[idx] = 0
        self.mask[idx] = 0
        self.ply[idx] = 0
        lo, hi = self.random_opening
        if hi > 0:
            k = self.rng.integers(lo, hi + 1, size=len(idx))
            for step in range(hi):
                act = idx[k > step]
                if len(act) == 0:
                    break
                cols = random_legal(legal(self.mask[act]), self.rng)
                nc, nm, won, full = play(self.cur[act], self.mask[act], cols)
                ok = ~(won | full)            # never end a game inside the opening
                act, nc, nm = act[ok], nc[ok], nm[ok]
                self.cur[act], self.mask[act] = nc, nm
                self.ply[act] += 1

    def legal(self):
        return legal(self.mask)

    def step(self, cols, idx=None):
        """Play cols in games idx (default: all).  Returns (won, full) for those games;
        state advances to the opponent's view."""
        if idx is None:
            idx = np.arange(self.n)
        nc, nm, won, full = play(self.cur[idx], self.mask[idx], cols)
        self.cur[idx], self.mask[idx] = nc, nm
        self.ply[idx] += 1
        return won, full


def grids(curs, masks):
    """(B,) positions -> (B, 2, 6, 7) uint8 planes [me, them] (row 0 = top)."""
    from nn.encode import bits_to_grid
    curs = np.asarray(curs, dtype=U)
    masks = np.asarray(masks, dtype=U)
    return np.stack([bits_to_grid(curs), bits_to_grid(curs ^ masks)], 1)
