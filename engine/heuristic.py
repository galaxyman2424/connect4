"""Static evaluation: 4-cell window scoring, computed bit-parallel.

For every 4-cell window (69 on a 6x7 board) that contains no opponent stone,
we count the side's stones in it:
  3 own + 1 empty  -> open three   (THREE)
  2 own + 2 empty  -> open two     (TWO)
The same is done for the opponent; the opponent's open threes are the
"threat penalty" (OPP_THREE) and open twos OPP_TWO. Every own stone in the
center column adds CENTER (opponent center stones subtract it).

All 69 windows of one direction are processed at once: for direction step d
the bitboard bit s stands for the window {s, s+d, s+2d, s+3d}; a bit-sliced
adder over the four shifted stone boards gives "exactly 2" / "exactly 3"
masks, and int.bit_count() counts them. The sentinel row of the bitboard
guarantees windows never wrap across columns.

The evaluation is deterministic and does NOT look at wins (terminal
positions are handled by the search).
"""

from .board import BOARD_MASK, COLUMN_MASK, H1

CENTER = 3
THREE = 5
TWO = 2
OPP_THREE = 4   # penalty
OPP_TWO = 2     # penalty

CENTER_MASK = COLUMN_MASK[3]
_DIRS = (1, H1, H1 - 1, H1 + 1)  # vertical, horizontal, two diagonals


def _counts(me, free):
    """(threes, twos) of windows lying entirely in ``free`` (no opponent
    stones) with exactly 3 / exactly 2 stones of ``me``."""
    threes = 0
    twos = 0
    for d in _DIRS:
        d2 = d + d
        d3 = d2 + d
        a = free & (free >> d) & (free >> d2) & (free >> d3)
        if not a:
            continue
        m0 = me
        m1 = me >> d
        m2 = me >> d2
        m3 = me >> d3
        s1 = m0 ^ m1
        c1 = m0 & m1
        s2 = m2 ^ m3
        c2 = m2 & m3
        low = s1 ^ s2
        mid = c1 ^ c2 ^ (s1 & s2)
        x = a & mid
        threes += (x & low).bit_count()
        twos += (x & ~low).bit_count()
    return threes, twos


def evaluate_bits(me, opp):
    """Score from the perspective of the player owning bitboard ``me``."""
    t, w = _counts(me, BOARD_MASK & ~opp)
    ot, ow = _counts(opp, BOARD_MASK & ~me)
    return (CENTER * ((me & CENTER_MASK).bit_count() - (opp & CENTER_MASK).bit_count())
            + THREE * t + TWO * w - OPP_THREE * ot - OPP_TWO * ow)


def evaluate(board, player):
    """Heuristic score of ``board`` from ``player``'s (+1/-1) perspective."""
    if player == 1:
        return float(evaluate_bits(board.p1, board.p2))
    if player == -1:
        return float(evaluate_bits(board.p2, board.p1))
    raise ValueError("player must be +1 or -1")


def evaluate_windows_reference(board, player):
    """Slow, obvious implementation (loops over the 69 windows) used by the
    tests to check evaluate()."""
    grid = board.to_grid()
    score = 0
    for r in range(6):
        if grid[r][3] == player:
            score += CENTER
        elif grid[r][3] == -player:
            score -= CENTER
    windows = []
    for r in range(6):
        for c in range(7):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(0 <= rr < 6 and 0 <= cc < 7 for rr, cc in cells):
                    windows.append([grid[rr][cc] for rr, cc in cells])
    for w in windows:
        mine, theirs = w.count(player), w.count(-player)
        if theirs == 0 and mine == 3:
            score += THREE
        elif theirs == 0 and mine == 2:
            score += TWO
        elif mine == 0 and theirs == 3:
            score -= OPP_THREE
        elif mine == 0 and theirs == 2:
            score -= OPP_TWO
    return float(score)
