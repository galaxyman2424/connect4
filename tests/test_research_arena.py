"""Fast unit tests for research/arena_analysis.py (no solver, no real data needed)."""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import arena_analysis as A  # noqa: E402


def _game(a, b, oid, a_first, score_a, opening="44", moves=None):
    moves = moves or opening + "1212121"
    return {"a": a, "b": b, "opening_id": oid, "opening": opening, "a_first": a_first, "winner": "",
            "score_a": score_a, "plies": len(moves), "moves": moves, "a_ms": 1.0, "b_ms": 2.0,
            "a_max_ms": 1.0, "b_max_ms": 2.0, "forfeit": ""}


def _pool(p_ab=0.75, p_bc=0.75, p_ac=0.9, per_pair=400):
    """Deterministic synthetic pool: exact fractions of wins in each pairing (no draws)."""
    games = []
    for (x, y, p) in (("A", "B", p_ab), ("B", "C", p_bc), ("A", "C", p_ac)):
        wins = int(round(p * per_pair))
        for k in range(per_pair):
            games.append(_game(x, y, k // 2, k % 2, 1.0 if k < wins else 0.0))
    return games


def test_bradley_terry_recovers_known_gaps():
    # consistent BT strengths: A-B = B-C = 400*log10(3) ~ 190.8; A-C = 381.7 -> p = 0.9
    d = 400 * math.log10(3)
    p_ac = 1 / (1 + 10 ** (-2 * d / 400))
    res, boot = A.elo_bootstrap(_pool(0.75, 0.75, p_ac), ["A", "B", "C"], "C", 1000.0, reps=50, seed=0)
    assert res["C"][0] == pytest.approx(1000.0)
    assert res["B"][0] - res["C"][0] == pytest.approx(d, abs=4)
    assert res["A"][0] - res["B"][0] == pytest.approx(d, abs=4)
    for p in "AB":
        assert res[p][1] <= res[p][0] <= res[p][2]
    assert boot.shape == (50, 3)
    assert np.allclose(boot[:, 2], 1000.0)               # anchor fixed in every resample


def test_bootstrap_diff_ci():
    res, boot = A.elo_bootstrap(_pool(), ["A", "B", "C"], "C", 1000.0, reps=200, seed=0)
    d = A.diff_ci(boot, ["A", "B", "C"], "A", "B", res)
    assert d["lo"] <= d["diff"] <= d["hi"]
    assert d["p_a_better"] > 0.9


def test_head_to_head_counts_and_colours():
    g = [_game("X", "Y", 0, 1, 1.0), _game("X", "Y", 0, 0, 0.5),       # X first: win; X second: draw
         _game("Y", "X", 1, 1, 1.0), _game("Y", "X", 1, 0, 0.0),       # Y first wins; X first wins
         _game("X", "Z", 0, 1, 0.0)]                                    # other pairing: ignored
    h = A.head_to_head(g, "X", "Y", reps=100)
    assert (h["games"], h["w"], h["d"], h["l"]) == (4, 2, 1, 1)
    assert h["score"] == pytest.approx(2.5 / 4)
    assert h["first"]["games"] == 2 and h["first"]["w"] == 2
    assert h["second"]["games"] == 2 and h["second"]["d"] == 1 and h["second"]["l"] == 1
    assert h["lo"] <= h["score"] <= h["hi"]


def test_moves_by_and_players():
    g = _game("X", "Y", 0, 1, 1.0, opening="44", moves="44123")       # plies 2,3,4 after the opening
    assert A.players_of(g) == ("X", "Y")
    assert A.moves_by(g, "X") == 2 and A.moves_by(g, "Y") == 1
    g2 = _game("X", "Y", 0, 1, 1.0, opening="443", moves="4431")
    assert A.moves_by(g2, "Y") == 1 and A.moves_by(g2, "X") == 0


def test_linreg_exact_line():
    xs = [0, 1, 2, 3, 4, 5]
    r = A.linreg(xs, [2 * x + 1 for x in xs], reps=200)
    assert r["slope"] == pytest.approx(2.0)
    assert r["intercept"] == pytest.approx(1.0)
    assert r["r2"] == pytest.approx(1.0)
    assert r["n"] == 6
    assert r["slope_ci"][0] == pytest.approx(2.0) and r["slope_ci"][1] == pytest.approx(2.0)


def test_linreg_degenerate():
    assert A.linreg([1, 2], [3, 4])["slope"] is None
    assert A.linreg([1, 1, 1], [3, 4, 5])["slope"] is None


def test_mirror_canon_lookup():
    assert A.mirror("1247") == "7641"
    cache = {}
    key, flip = A.canon("71")
    assert key == "17" and flip
    cache["17"] = [1, 2, 3, 4, 5, 6, 7]
    assert A.lookup(cache, "71") == [7, 6, 5, 4, 3, 2, 1]
    assert A.lookup(cache, "17") == [1, 2, 3, 4, 5, 6, 7]
    assert A.lookup(cache, "44") is None


def test_move_error_and_expected():
    sc = [-2, -1, 0, 3, None, 0, -1]
    assert A.move_error(sc, 3) == (1, 1)          # optimal
    assert A.move_error(sc, 2) == (1, 0)          # win -> draw
    assert A.move_error(sc, 0) == (1, -1)         # win -> loss
    # opening with first player to move (even length) and a winning move -> first player wins
    assert A.value_to_p1_expected("44", sc) == 1.0
    # odd length: side to move is the second player
    assert A.value_to_p1_expected("443", sc) == 0.0
    assert A.value_to_p1_expected("443", [0, -1, None, -2, -1, -1, -3]) == 0.5
