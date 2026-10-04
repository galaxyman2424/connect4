"""Board -> network input.

A position is stored everywhere in nn/ as two Python ints (or uint64 arrays):
  cur   bitboard of the stones of the side to move
  mask  bitboard of all stones
using the engine's bit layout (engine/board.py: column c owns bits c*7..c*7+6,
bit c*7+h = height h from the bottom, bit c*7+6 is an always-empty sentinel).

Input planes (4 x 6 x 7, row 0 = TOP row, like the website grid):
  0  stones of the side to move           ("me")
  1  stones of the opponent               ("them")
  2  all ones                             (lets 3x3 convs see the board edge)
  3  all ones if the side to move is the  (Connect 4 strategy depends on
     first player, else all zeros          odd/even rows, so the net must know
                                           which colour it is playing)
Encoding from the side-to-move's point of view means one network plays both
colours and the value is always "how good is this for the player to move".
"""

import numpy as np

ROWS, COLS, H1 = 6, 7, 7
PLANES = 4

# _BIT[r, c] = bit index of grid cell (row r from the top, column c)
_BIT = np.array([[c * H1 + (ROWS - 1 - r) for c in range(COLS)] for r in range(ROWS)],
                dtype=np.uint64)
_TOP_BITS = np.array([c * H1 + ROWS - 1 for c in range(COLS)], dtype=np.uint64)


def to_u64(values):
    """List of Python ints -> uint64 array."""
    return np.array(values, dtype=np.uint64)


def bits_to_grid(bb):
    """uint64 array (B,) -> uint8 array (B, 6, 7) of 0/1 cells."""
    bb = np.asarray(bb, dtype=np.uint64)
    return ((bb[:, None, None] >> _BIT[None, :, :]) & np.uint64(1)).astype(np.uint8)


def encode(curs, masks):
    """Batch encode.  curs, masks: sequences of ints or uint64 arrays (B,).
    Returns float32 array (B, 4, 6, 7)."""
    cur = np.asarray(curs, dtype=np.uint64) if not isinstance(curs, np.ndarray) else curs.astype(np.uint64)
    mask = np.asarray(masks, dtype=np.uint64) if not isinstance(masks, np.ndarray) else masks.astype(np.uint64)
    me = bits_to_grid(cur)
    them = bits_to_grid(cur ^ mask)
    b = me.shape[0]
    x = np.empty((b, PLANES, ROWS, COLS), dtype=np.float32)
    x[:, 0] = me
    x[:, 1] = them
    x[:, 2] = 1.0
    first = me.reshape(b, -1).sum(1) == them.reshape(b, -1).sum(1)   # equal counts -> P1 to move
    x[:, 3] = first[:, None, None].astype(np.float32)
    return x


def legal_mask(masks):
    """uint64 masks (B,) -> bool array (B, 7): column not full."""
    m = np.asarray(masks, dtype=np.uint64)
    return ((m[:, None] >> _TOP_BITS[None, :]) & np.uint64(1)) == 0


def mirror_planes(x):
    """Left-right mirror of encoded planes (B, C, 6, 7)."""
    return x[..., ::-1].copy()


def mirror_policy(p):
    """Left-right mirror of policy vectors (B, 7)."""
    return p[..., ::-1].copy()


def mirror_bits(bb):
    """Mirror bitboards (uint64 array) left-right: column c -> 6 - c."""
    bb = np.asarray(bb, dtype=np.uint64)
    out = np.zeros_like(bb)
    colmask = np.uint64((1 << H1) - 1)
    for c in range(COLS):
        col = (bb >> np.uint64(c * H1)) & colmask
        out |= col << np.uint64((COLS - 1 - c) * H1)
    return out


def board_to_bits(board):
    """engine.board.Board -> (cur, mask) ints for the side to move."""
    return board.current_bits(), board.mask
