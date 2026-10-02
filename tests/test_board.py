import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board, ROWS, COLS  # noqa: E402
from engine.heuristic import evaluate, evaluate_windows_reference  # noqa: E402


def empty_grid():
    return [[0] * COLS for _ in range(ROWS)]


# ------------------------------------------------------------------ basics
def test_empty_board():
    b = Board()
    assert b.to_move == 1
    assert b.moves_played == 0
    assert b.legal_moves() == list(range(7))
    assert b.check_win() == 0
    assert not b.is_draw()
    assert not b.is_terminal()
    assert b.to_grid() == empty_grid()


def test_drop_goes_to_bottom_and_alternates():
    b = Board()
    b.drop(3)
    b.drop(3)
    g = b.to_grid()
    assert g[5][3] == 1      # row 0 = top, so the bottom row is index 5
    assert g[4][3] == -1
    assert b.to_move == 1
    assert b.moves_played == 2


# ------------------------------------------------------------------ wins
def test_horizontal_win():
    b = Board.from_moves("1122334")  # X: 1,2,3,4 on bottom row
    assert b.check_win() == 1
    assert b.is_terminal()
    assert sorted(b.winning_cells()) == [(5, 0), (5, 1), (5, 2), (5, 3)]


def test_horizontal_win_right_edge():
    b = Board.from_moves("4455667")
    assert b.check_win() == 1
    assert sorted(b.winning_cells()) == [(5, 3), (5, 4), (5, 5), (5, 6)]


def test_vertical_win():
    b = Board.from_moves("1212121")
    assert b.check_win() == 1
    assert sorted(b.winning_cells()) == [(2, 0), (3, 0), (4, 0), (5, 0)]


def test_vertical_win_second_player_top_of_column():
    # O stacks four in column 7 on top of an X stone -> rows 1..4
    b = Board.from_moves("77172717")
    assert b.check_win() == -1
    assert sorted(b.winning_cells()) == [(1, 6), (2, 6), (3, 6), (4, 6)]


def _diag_grid(mirror=False):
    X, O = 1, -1
    cells = {(5, 0): X, (4, 1): X, (3, 2): X, (2, 3): X,          # the diagonal
             (5, 1): O, (5, 2): X, (4, 2): O, (5, 3): O, (4, 3): X, (3, 3): O,
             (5, 4): O}
    g = empty_grid()
    for (r, c), v in cells.items():
        g[r][6 - c if mirror else c] = v
    return g


def test_diagonal_up_right_win():
    b = Board.from_grid(_diag_grid())
    assert b.check_win() == 1
    assert sorted(b.winning_cells()) == [(2, 3), (3, 2), (4, 1), (5, 0)]


def test_diagonal_down_right_win():
    b = Board.from_grid(_diag_grid(mirror=True))
    assert b.check_win() == 1
    assert sorted(b.winning_cells()) == [(2, 3), (3, 4), (4, 5), (5, 6)]


def test_diagonal_win_by_dropping():
    # same diagonal reached by real moves; the last move completes it
    b = Board.from_grid(_diag_grid())
    seq = b.to_move_string()
    b2 = Board.from_moves(seq[:-1])
    assert b2.check_win() == 0
    b2.drop(int(seq[-1]) - 1)
    assert b2.check_win() == 1


def test_no_wrap_between_columns():
    # X on top 3 cells of column 1 and bottom cell of column 2 are adjacent
    # bits in a naive 6-bit-per-column layout: must NOT count as a win.
    g = empty_grid()
    col0 = [-1, 1, -1, 1, 1, 1]  # heights 0..5 (bottom..top)
    for h, v in enumerate(col0):
        g[5 - h][0] = v
    g[5][1] = 1
    g[5][2] = -1
    g[5][3] = -1
    b = Board.from_grid(g)
    assert b.check_win() == 0


@pytest.mark.parametrize("seed", range(20))
def test_check_win_matches_bruteforce(seed):
    rng = random.Random(seed)
    b = Board()
    while not b.is_terminal():
        b.drop(rng.choice(b.legal_moves()))
        g = b.to_grid()
        expect = 0
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + i * dr, c + i * dc) for i in range(4)]
                    if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                        vals = {g[rr][cc] for rr, cc in cells}
                        if len(vals) == 1 and 0 not in vals:
                            expect = vals.pop()
        assert b.check_win() == expect
    if b.check_win():
        cells = b.winning_cells()
        g = b.to_grid()
        assert len(cells) == 4 and len({g[r][c] for r, c in cells}) == 1


# ------------------------------------------------------ full / illegal moves
def test_full_column_is_illegal():
    b = Board.from_moves("444444")
    assert 3 not in b.legal_moves()
    assert b.legal_moves() == [0, 1, 2, 4, 5, 6]
    with pytest.raises(ValueError):
        b.drop(3)
    assert b.moves_played == 6


@pytest.mark.parametrize("bad", [-1, 7, 100, "3", 2.0, None, True])
def test_out_of_range_column(bad):
    with pytest.raises(ValueError):
        Board().drop(bad)


def test_no_moves_after_win():
    b = Board.from_moves("1212121")
    assert b.legal_moves() == []
    with pytest.raises(ValueError):
        b.drop(4)


# ------------------------------------------------------------------ draws
# A drawn final position (from a random game): 21 stones each, no four.
DRAW_GRID = [
    [-1, -1, 1, -1, 1, -1, 1],
    [-1, 1, 1, 1, -1, 1, 1],
    [-1, 1, -1, 1, -1, 1, -1],
    [1, -1, 1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1, 1, -1],
    [1, 1, -1, 1, -1, 1, -1],
]


def test_draw_detection():
    b = Board.from_grid(DRAW_GRID)
    assert b.check_win() == 0
    assert b.is_draw()
    assert b.is_terminal()
    assert b.legal_moves() == []
    assert b.moves_played == 42


def test_draw_by_playing_out():
    seq = Board.from_grid(DRAW_GRID).to_move_string()
    b = Board.from_moves(seq[:-1])
    assert not b.is_draw() and not b.is_terminal()
    assert len(b.legal_moves()) == 1
    b.drop(b.legal_moves()[0])
    assert b.to_grid() == DRAW_GRID
    assert b.is_draw()
    b.undo()
    assert not b.is_draw()


def test_not_draw_when_last_move_wins():
    b = Board.from_moves("1212121")
    assert not b.is_draw()


# ------------------------------------------------------------------ undo
def test_undo_restores_everything():
    rng = random.Random(5)
    b = Board()
    snapshots = []
    while not b.is_terminal():
        snapshots.append((b.to_grid(), b.to_move, b.moves_played, b.key()))
        b.drop(rng.choice(b.legal_moves()))
    while snapshots:
        b.undo()
        g, tm, n, k = snapshots.pop()
        assert (b.to_grid(), b.to_move, b.moves_played, b.key()) == (g, tm, n, k)
    with pytest.raises(IndexError):
        b.undo()


def test_undo_returns_column_and_reopens_full_column():
    b = Board.from_moves("444444")
    assert b.undo() == 3
    assert 3 in b.legal_moves()


# ------------------------------------------------- grid round trip / builders
@pytest.mark.parametrize("seed", range(10))
def test_grid_round_trip(seed):
    rng = random.Random(seed)
    b = Board()
    for _ in range(rng.randint(0, 30)):
        if b.is_terminal():
            break
        b.drop(rng.choice(b.legal_moves()))
    g = b.to_grid()
    b2 = Board.from_grid(g)
    assert b2.to_grid() == g
    assert b2.to_move == b.to_move
    assert b2.moves_played == b.moves_played
    assert b2.key() == b.key()
    assert b2 == b
    # reconstructed move string reaches the same position
    assert Board.from_moves(b2.to_move_string()).key() == b.key()


def test_from_grid_infers_to_move():
    g = empty_grid()
    assert Board.from_grid(g).to_move == 1
    g[5][3] = 1
    assert Board.from_grid(g).to_move == -1
    g[5][4] = -1
    assert Board.from_grid(g).to_move == 1


@pytest.mark.parametrize("mutate", [
    lambda g: g[0].__setitem__(3, 1),                  # floating stone
    lambda g: (g[5].__setitem__(0, -1)),               # O moved first
    lambda g: (g[5].__setitem__(0, 1), g[5].__setitem__(1, 1)),  # two X, no O
    lambda g: g[5].__setitem__(0, 2),                  # bad value
    lambda g: g.pop(),                                 # wrong shape
    lambda g: g[2].append(0),                          # wrong row length
])
def test_from_grid_rejects_invalid(mutate):
    g = empty_grid()
    mutate(g)
    with pytest.raises(ValueError):
        Board.from_grid(g)


def test_from_grid_rejects_wrong_to_move():
    with pytest.raises(ValueError):
        Board.from_grid(empty_grid(), to_move=-1)


def test_from_moves():
    b = Board.from_moves("4453")
    g = b.to_grid()
    assert g[5][3] == 1 and g[4][3] == -1 and g[5][4] == 1 and g[5][2] == -1
    assert b.to_move == 1 and b.moves_played == 4
    assert b.to_move_string() == "4453"
    assert Board.from_moves([3, 3, 4, 2]).key() == b.key()
    with pytest.raises(ValueError):
        Board.from_moves("4483")


def test_copy_is_independent():
    b = Board.from_moves("44")
    c = b.copy()
    c.drop(0)
    assert b.moves_played == 2 and c.moves_played == 3


# ------------------------------------------------------------- heuristic
@pytest.mark.parametrize("seed", range(30))
def test_heuristic_matches_reference(seed):
    rng = random.Random(seed)
    b = Board()
    for _ in range(rng.randint(0, 35)):
        if b.is_terminal():
            break
        b.drop(rng.choice(b.legal_moves()))
    for p in (1, -1):
        assert evaluate(b, p) == evaluate_windows_reference(b, p)
