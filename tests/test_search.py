import os
import random
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import Board  # noqa: E402
from engine.minimax import search, plain_negamax, WIN, is_win_score  # noqa: E402
from engine.agents import (RandomAgent, MinimaxAgent, SolverAgent,  # noqa: E402
                           SolverUnavailable, LEVELS, LEVEL_TIME_LIMITS,
                           agent_for_level)


def random_position(rng, lo=0, hi=30):
    while True:
        b = Board()
        for _ in range(rng.randint(lo, hi)):
            if b.is_terminal():
                break
            b.drop(rng.choice(b.legal_moves()))
        if not b.is_terminal():
            return b


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 6])
def test_takes_win_in_1_horizontal(depth):
    b = Board.from_moves("112233")
    r = search(b, depth)
    assert r.col == 3
    assert r.score == WIN - 1


@pytest.mark.parametrize("depth", [1, 2, 5])
def test_takes_win_in_1_vertical(depth):
    b = Board.from_moves("121212")
    r = search(b, depth)
    assert r.col == 0 and r.score == WIN - 1


@pytest.mark.parametrize("depth", [1, 3, 4])
def test_takes_win_in_1_second_player(depth):
    b = Board.from_moves("1727376")   # O has 7,7,7 vertically; X just played 6
    assert b.to_move == -1
    r = search(b, depth)
    assert r.col == 6 and r.score == WIN - 1


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_takes_diagonal_win_in_1(depth):
    # X diagonal (c0,h0),(c1,h1),(c2,h2) + missing (c3,h3)
    g = [[0] * 7 for _ in range(6)]
    X, O = 1, -1
    for (r, c), v in {(5, 0): X, (4, 1): X, (3, 2): X, (5, 1): O, (5, 2): X,
                      (4, 2): O, (5, 3): O, (4, 3): X, (3, 3): O, (5, 6): O}.items():
        g[r][c] = v
    b = Board.from_grid(g)
    assert b.to_move == 1
    r = search(b, depth)
    assert r.col == 3 and r.score == WIN - 1


def test_prefers_win_over_block():
    # X: 1,2,3 on bottom (wins at 4); O: 7,7,7 (wins at 7). X to move.
    b = Board.from_moves("17273")
    b.drop(6)          # O's third stone in column 7
    assert b.to_move == 1
    for d in (1, 2, 4, 6):
        assert search(b, d).col == 3


@pytest.mark.parametrize("depth", [2, 3, 4, 5, 6, 8])
def test_blocks_opponent_win_in_1(depth):
    b = Board.from_moves("11223")     # X: 1,2,3 bottom row; O to move
    assert b.to_move == -1
    r = search(b, depth)
    assert r.col == 3


@pytest.mark.parametrize("depth", [2, 3, 4, 7])
def test_blocks_vertical_threat(depth):
    b = Board.from_moves("41424")     # X has col 4 at h0,h1,h2; O to move
    assert b.to_move == -1
    assert search(b, depth).col == 3


def test_faster_win_preferred_and_loss_scores():
    # a double threat: X wins in 3 plies at best
    b = Board.from_moves("4455")      # X to move: playing 3 or 6 creates _XXX_ (open both sides)
    r = search(b, 4)
    assert r.score == WIN - 3
    assert r.col in (2, 5)
    # side to move facing that double threat loses at ply 2
    b.drop(r.col)
    r2 = search(b, 4)
    assert r2.score == -(WIN - 2)


def test_draw_scores_zero():
    g = [
        [0, -1, 1, -1, 1, -1, 1],
        [-1, 1, 1, 1, -1, 1, 1],
        [-1, 1, -1, 1, -1, 1, -1],
        [1, -1, 1, -1, -1, -1, 1],
        [1, -1, -1, -1, 1, 1, -1],
        [1, 1, -1, 1, -1, 1, -1],
    ]
    b = Board.from_grid(g)
    r = search(b, 5)
    assert r.col == 0 and r.score == 0


def test_search_does_not_mutate_and_is_deterministic():
    b = Board.from_moves("44536")
    before = (b.key(), b.to_move, b.moves_played, list(b.history))
    r1 = search(b, 6)
    r2 = search(b, 6)
    assert (b.key(), b.to_move, b.moves_played, list(b.history)) == before
    assert (r1.col, r1.score, r1.nodes) == (r2.col, r2.score, r2.nodes)


def test_terminal_and_bad_depth_raise():
    with pytest.raises(ValueError):
        search(Board.from_moves("1212121"), 3)
    with pytest.raises(ValueError):
        search(Board(), 0)


# --------------------------------------------- correctness of the pruning
@pytest.mark.parametrize("seed", range(40))
def test_matches_plain_negamax(seed):
    """Alpha-beta + TT + shortcuts == plain exhaustive negamax (depth <= 3)."""
    rng = random.Random(seed)
    b = random_position(rng)
    for d in (1, 2, 3):
        pc, ps = plain_negamax(b, d)
        for use_tt in (True, False):
            for ordering in (True, False):
                r = search(b, d, use_tt=use_tt, ordering=ordering)
                assert r.score == ps
                if ps > -(WIN - 3):        # in lost positions any move is "best"
                    assert r.col == pc


@pytest.mark.parametrize("seed", range(25))
def test_tt_on_off_same_result(seed):
    rng = random.Random(1000 + seed)
    b = random_position(rng, 0, 24)
    for d in (4, 5, 6, 7):
        a = search(b, d, use_tt=True)
        c = search(b, d, use_tt=False)
        e = search(b, d, use_tt=False, ordering=False)
        assert (a.col, a.score) == (c.col, c.score) == (e.col, e.score), (b.to_move_string(), d)


def test_tt_reduces_nodes_at_depth_8():
    b = Board.from_moves("4453")
    a = search(b, 8, use_tt=True)
    c = search(b, 8, use_tt=False)
    assert (a.col, a.score) == (c.col, c.score)
    assert a.nodes < c.nodes


def test_node_counter_counts_calls():
    b = Board()
    r = search(b, 1, use_tt=False)
    assert r.nodes == 8           # root + 7 children
    r2 = search(b, 2, use_tt=False)
    assert r2.nodes > r.nodes


# ------------------------------------------------------------ performance
@pytest.mark.parametrize("moves", ["", "4453", "445344345624", "4453443456243333"])
def test_depth_8_is_fast(moves):
    r = search(Board.from_moves(moves), 8)
    assert r.ms < 2000, r


def test_time_limit_respected():
    t0 = time.perf_counter()
    r = search(Board(), 30, time_limit=0.3)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.6
    assert r.depth_reached >= 4
    assert 0 <= r.col < 7


# ---------------------------------------------------------------- agents
def play_game(first, second, opening=()):
    b = Board()
    for c in opening:
        b.drop(c)
    agents = {1: first, -1: second}
    while not b.is_terminal():
        b.drop(agents[b.to_move].choose(b))
    return b.check_win()


def test_depth4_beats_random():
    wins = 0
    games = 40
    for g in range(games):
        bot = MinimaxAgent(4)
        rnd = RandomAgent(seed=g)
        if g % 2 == 0:
            wins += play_game(bot, rnd) == 1
        else:
            wins += play_game(rnd, bot) == -1
    assert wins >= 0.95 * games, wins


def test_epsilon_one_is_random_and_flagged():
    a = MinimaxAgent(3, epsilon=1.0, seed=1)
    b = Board()
    cols = set()
    for _ in range(30):
        c = a.choose(b)
        assert a.last_result.random is True and a.last_result.nodes == 0
        cols.add(c)
    assert len(cols) > 1


def test_epsilon_seeded_reproducible():
    def run(seed):
        a = MinimaxAgent(2, epsilon=0.5, seed=seed)
        r = RandomAgent(seed=99)
        b = Board()
        moves = []
        while not b.is_terminal():
            c = (a if b.to_move == 1 else r).choose(b)
            moves.append(c)
            b.drop(c)
        return moves
    assert run(5) == run(5)


def test_levels_and_agent_for_level():
    assert LEVELS == {1: (1, 0.5), 2: (1, 0.2), 3: (2, 0.1), 4: (3, 0), 5: (4, 0),
                      6: (5, 0), 7: (6, 0), 8: (7, 0), 9: (8, 0), 10: (20, 0)}
    for lvl in range(1, 11):
        a = agent_for_level(lvl, seed=0)
        assert a.depth == LEVELS[lvl][0] and a.epsilon == LEVELS[lvl][1]
    assert agent_for_level(10).time_limit == LEVEL_TIME_LIMITS[10]
    assert agent_for_level(9).time_limit is None
    with pytest.raises(ValueError):
        agent_for_level(11)


def test_agent_result_fields():
    a = agent_for_level(5)
    b = Board.from_moves("44")
    c = a.choose(b)
    r = a.last_result
    assert r.col == c and r.depth_reached == 4 and r.nodes > 0 and r.ms >= 0
    assert r.random is False
    d = r.to_dict()
    assert set(d) == {"col", "score", "nodes", "ms", "depth", "random"}


# ---------------------------------------------------------------- solver
def _solver_or_skip():
    try:
        return SolverAgent()
    except SolverUnavailable:
        pytest.skip("Pons solver binary not built")


def test_solver_agent_late_position():
    s = _solver_or_skip()
    b = Board.from_moves("445344345624333377771111")   # 24 stones: fast to solve
    sc = s.scores(b)
    assert len(sc) == 7
    col = s.choose(b)
    assert sc[col] == max(v for v in sc if v is not None)


def test_solver_agent_takes_win():
    s = _solver_or_skip()
    b2 = Board.from_moves("112233")
    # immediate win: no need for the book
    assert s.choose(b2) == 3 and s.last_scores[3] == (43 - 6) // 2
