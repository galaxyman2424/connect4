"""C minimax (engine/cminimax) == Python minimax: same column, score and node count."""
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import cminimax_ffi as C  # noqa: E402
from engine.board import Board  # noqa: E402
from engine.heuristic import evaluate_bits  # noqa: E402

if C._load() is None:
    try:
        from engine.cbuild import build
        build(verbose=False)
        C._ERR = None
    except SystemExit:
        pass
pytestmark = pytest.mark.skipif(C._load() is None, reason="C minimax not built (python -m engine.cbuild)")


def _boards(n, seed, max_ply=32):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = Board()
        for _ in range(rng.randint(0, max_ply)):
            if b.is_terminal():
                break
            b.drop(rng.choice(b.legal_moves()))
        if not b.is_terminal():
            out.append(b)
    return out


def _python(board, depth, **kw):
    from engine import minimax
    old = os.environ.get("C4_MINIMAX_C")
    os.environ["C4_MINIMAX_C"] = "0"
    try:
        return minimax.search(board, depth, **kw)
    finally:
        if old is None:
            del os.environ["C4_MINIMAX_C"]
        else:
            os.environ["C4_MINIMAX_C"] = old


def test_evaluate_identical():
    for b in _boards(400, 3):
        me, opp = b.current_bits(), b.current_bits() ^ b.mask
        assert C.evaluate_bits(me, opp) == evaluate_bits(me, opp)


@pytest.mark.parametrize("use_tt,ordering", [(True, True), (False, True), (True, False), (False, False)])
def test_search_identical(use_tt, ordering):
    for i, b in enumerate(_boards(120, 7)):
        d = 1 + i % 7
        p = _python(b, d, use_tt=use_tt, ordering=ordering)
        c = C.search(b, d, use_tt=use_tt, ordering=ordering)
        assert (c.col, c.score, c.nodes, c.depth_reached) == (p.col, p.score, p.nodes, p.depth_reached), (b.to_move_string(), d)


def test_deep_and_endgame_identical():
    for b in _boards(25, 11, max_ply=36):
        p = _python(b, 10)
        c = C.search(b, 10)
        assert (c.col, c.score, c.nodes) == (p.col, p.score, p.nodes)


def test_dispatch_and_time_limit():
    from engine import minimax
    b = Board.from_moves("4453")
    r = minimax.search(b, 30, time_limit=0.05)
    assert 0 <= r.col < 7 and r.depth_reached >= 1
    assert minimax.search(b, 6).nodes == _python(b, 6).nodes
