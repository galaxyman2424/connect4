"""Connect 4 board backed by bitboards (pure Python, Python 3.10+).

Bit layout (same as Pascal Pons' solver): column ``c`` owns bits
``c*7 .. c*7+6``; bit ``c*7 + h`` is the cell at height ``h`` (0 = bottom row)
of column ``c``.  Bit ``c*7 + 6`` is an always-empty sentinel that stops
shifted line patterns from wrapping between columns.

Public grid format (JSON / API): list of 6 rows x 7 ints, row 0 = TOP row,
values +1 (player who moved first), -1 (second player), 0 (empty).
"""

ROWS = 6
COLS = 7
H1 = ROWS + 1  # bits per column including the sentinel

# --- precomputed masks ------------------------------------------------------
BOTTOM_MASK = 0
for _c in range(COLS):
    BOTTOM_MASK |= 1 << (_c * H1)
BOARD_MASK = BOTTOM_MASK * ((1 << ROWS) - 1)          # every playable cell
COLUMN_MASK = [((1 << ROWS) - 1) << (c * H1) for c in range(COLS)]
TOP_MASK = [1 << (ROWS - 1 + c * H1) for c in range(COLS)]
BOTTOM_COL = [1 << (c * H1) for c in range(COLS)]
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)                  # center-first ordering
CENTER_RANK = {c: i for i, c in enumerate(CENTER_ORDER)}


def alignment(pos):
    """True if bitboard ``pos`` contains four in a row."""
    m = pos & (pos >> H1)            # horizontal
    if m & (m >> (2 * H1)):
        return True
    m = pos & (pos >> (H1 - 1))      # diagonal "\" (down-right)
    if m & (m >> (2 * (H1 - 1))):
        return True
    m = pos & (pos >> (H1 + 1))      # diagonal "/" (up-right)
    if m & (m >> (2 * (H1 + 1))):
        return True
    m = pos & (pos >> 1)             # vertical
    if m & (m >> 2):
        return True
    return False


def winning_cells_mask(pos, mask):
    """Bitboard of EMPTY cells that would complete four-in-a-row for ``pos``
    (whether or not they are currently playable)."""
    r = (pos << 1) & (pos << 2) & (pos << 3)                  # vertical
    for p in (H1, H1 - 1, H1 + 1):
        p2 = (pos << p) & (pos << (2 * p))
        r |= p2 & (pos << (3 * p))
        r |= p2 & (pos >> p)
        p2 = (pos >> p) & (pos >> (2 * p))
        r |= p2 & (pos << p)
        r |= p2 & (pos >> (3 * p))
    return r & (BOARD_MASK ^ mask)


def _bit(row_from_bottom, col):
    return 1 << (col * H1 + row_from_bottom)


class Board:
    """Mutable Connect 4 position.

    Attributes (read-only by convention):
      p1, p2   bitboards of player +1 and player -1 stones
      mask     p1 | p2
      to_move  +1 or -1
    """

    __slots__ = ("p1", "p2", "mask", "to_move", "moves_played", "history")

    def __init__(self):
        self.p1 = 0
        self.p2 = 0
        self.mask = 0
        self.to_move = 1
        self.moves_played = 0
        self.history = []  # columns dropped since construction (for undo)

    # ------------------------------------------------------------- builders
    @classmethod
    def from_grid(cls, grid, to_move=None):
        """Build from a 6x7 grid (row 0 = top). Validates shape, values,
        gravity (no floating stones), piece counts and that at most one
        player has a four. ``to_move`` is inferred when None (+1 if the
        piece counts are equal, else -1); if given it must match the counts.
        Raises ValueError on any invalid grid."""
        if not isinstance(grid, (list, tuple)) or len(grid) != ROWS:
            raise ValueError("grid must have %d rows" % ROWS)
        b = cls()
        n1 = n2 = 0
        for r, row in enumerate(grid):
            if not isinstance(row, (list, tuple)) or len(row) != COLS:
                raise ValueError("each grid row must have %d columns" % COLS)
            h = ROWS - 1 - r  # height from bottom
            for c, v in enumerate(row):
                if isinstance(v, bool) or v not in (-1, 0, 1):
                    raise ValueError("grid values must be -1, 0 or +1")
                if v == 1:
                    b.p1 |= _bit(h, c)
                    n1 += 1
                elif v == -1:
                    b.p2 |= _bit(h, c)
                    n2 += 1
        b.mask = b.p1 | b.p2
        # gravity: every column must be a contiguous stack from the bottom
        for c in range(COLS):
            col = (b.mask >> (c * H1)) & ((1 << ROWS) - 1)
            if col & (col + 1):
                raise ValueError("floating stone in column %d" % c)
        if n1 - n2 not in (0, 1):
            raise ValueError("invalid piece counts (+1: %d, -1: %d)" % (n1, n2))
        inferred = 1 if n1 == n2 else -1
        if to_move is None:
            to_move = inferred
        elif to_move != inferred:
            raise ValueError("to_move=%r inconsistent with piece counts" % (to_move,))
        if alignment(b.p1) and alignment(b.p2):
            raise ValueError("both players have four in a row")
        b.to_move = to_move
        b.moves_played = n1 + n2
        return b

    @classmethod
    def from_moves(cls, moves):
        """Build from a Pons-style move string of 1-indexed columns, e.g. "4453".
        Also accepts an iterable of 0-indexed ints."""
        b = cls()
        if isinstance(moves, str):
            for ch in moves.strip():
                if ch < "1" or ch > "7":
                    raise ValueError("bad move character %r" % ch)
                b.drop(ord(ch) - ord("1"))
        else:
            for c in moves:
                b.drop(int(c))
        return b

    def copy(self):
        b = Board.__new__(Board)
        b.p1 = self.p1
        b.p2 = self.p2
        b.mask = self.mask
        b.to_move = self.to_move
        b.moves_played = self.moves_played
        b.history = list(self.history)
        return b

    # ------------------------------------------------------------ accessors
    def current_bits(self):
        """Bitboard of the side to move."""
        return self.p1 if self.to_move == 1 else self.p2

    def to_grid(self):
        grid = []
        for r in range(ROWS):
            h = ROWS - 1 - r
            row = []
            for c in range(COLS):
                bit = _bit(h, c)
                row.append(1 if self.p1 & bit else (-1 if self.p2 & bit else 0))
            grid.append(row)
        return grid

    def key(self):
        """Hashable, unique key for the position (independent of history)."""
        return (self.p1, self.p2)

    def height(self, col):
        return ((self.mask >> (col * H1)) & ((1 << ROWS) - 1)).bit_length()

    def can_play(self, col):
        return 0 <= col < COLS and not (self.mask & TOP_MASK[col])

    def legal_moves(self):
        """Columns that are not full, left to right. Empty once someone has won."""
        if self.check_win():
            return []
        m = self.mask
        return [c for c in range(COLS) if not (m & TOP_MASK[c])]

    # ------------------------------------------------------------- mutation
    def drop(self, col):
        """Drop a stone for the side to move into ``col`` (0-indexed).
        Raises ValueError if the column is out of range / full or the game is over."""
        if not isinstance(col, int) or isinstance(col, bool) or not 0 <= col < COLS:
            raise ValueError("column must be an int in 0..6, got %r" % (col,))
        if self.mask & TOP_MASK[col]:
            raise ValueError("column %d is full" % col)
        if self.check_win():
            raise ValueError("game is already over")
        move = (self.mask + BOTTOM_COL[col]) & COLUMN_MASK[col]
        if self.to_move == 1:
            self.p1 |= move
        else:
            self.p2 |= move
        self.mask |= move
        self.to_move = -self.to_move
        self.moves_played += 1
        self.history.append(col)
        return self

    def undo(self):
        """Revert the last drop made on this object. Raises IndexError if none."""
        if not self.history:
            raise IndexError("no move to undo")
        col = self.history.pop()
        colbits = self.mask & COLUMN_MASK[col]
        top = 1 << (colbits.bit_length() - 1)
        self.p1 &= ~top
        self.p2 &= ~top
        self.mask &= ~top
        self.to_move = -self.to_move
        self.moves_played -= 1
        return col

    # ---------------------------------------------------------------- state
    def check_win(self):
        """+1 / -1 if that player has four in a row, else 0."""
        if alignment(self.p1):
            return 1
        if alignment(self.p2):
            return -1
        return 0

    def is_full(self):
        return self.mask == BOARD_MASK

    def is_draw(self):
        return self.mask == BOARD_MASK and not self.check_win()

    def is_terminal(self):
        return self.mask == BOARD_MASK or self.check_win() != 0

    def winning_cells(self):
        """(row, col) cells of one winning line (row 0 = top), or []."""
        for pos in (self.p1, self.p2):
            if not alignment(pos):
                continue
            for c in range(COLS):
                for h in range(ROWS):
                    for dc, dh in ((1, 0), (0, 1), (1, 1), (1, -1)):
                        cells = [(h + i * dh, c + i * dc) for i in range(4)]
                        if all(0 <= hh < ROWS and 0 <= cc < COLS and pos & _bit(hh, cc)
                               for hh, cc in cells):
                            return [(ROWS - 1 - hh, cc) for hh, cc in cells]
        return []

    def to_move_string(self):
        """A Pons-style move string (1-indexed) that reaches this position.

        Uses the recorded history when the board was built from the empty
        position; otherwise reconstructs some legal move order (alternating
        colours, respecting gravity). Raises ValueError if impossible.
        """
        if len(self.history) == self.moves_played:
            return "".join(str(c + 1) for c in self.history)
        seq = _reconstruct(self.p1, self.p2, self.mask, -self.to_move, {})
        if seq is None:
            raise ValueError("position is not reachable by a legal move order")
        return "".join(str(c + 1) for c in seq)

    # -------------------------------------------------------------- display
    def __str__(self):
        sym = {1: "X", -1: "O", 0: "."}
        lines = [" ".join(sym[v] for v in row) for row in self.to_grid()]
        lines.append(" ".join(str(c + 1) for c in range(COLS)))
        return "\n".join(lines)

    def __repr__(self):
        return "Board(moves_played=%d, to_move=%+d)" % (self.moves_played, self.to_move)

    def __eq__(self, other):
        return isinstance(other, Board) and self.p1 == other.p1 and \
            self.p2 == other.p2 and self.to_move == other.to_move

    def __hash__(self):
        return hash((self.p1, self.p2))


def _reconstruct(p1, p2, mask, last_mover, memo):
    """Return list of 0-indexed columns from the empty board to (p1, p2)."""
    if mask == 0:
        return []
    key = (p1, p2)
    if key in memo:
        return None
    memo[key] = True
    stones = p1 if last_mover == 1 else p2
    for c in CENTER_ORDER:
        colbits = mask & COLUMN_MASK[c]
        if not colbits:
            continue
        top = 1 << (colbits.bit_length() - 1)
        if not stones & top:
            continue
        n1, n2 = (p1 & ~top, p2) if last_mover == 1 else (p1, p2 & ~top)
        if alignment(n1) or alignment(n2):
            continue  # the game would already have ended before this move
        sub = _reconstruct(n1, n2, mask & ~top, -last_mover, memo)
        if sub is not None:
            sub.append(c)
            return sub
    return None
