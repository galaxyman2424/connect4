"""C MCTS tree (engine/cminimax/mcts.c via nn/cmcts.py) == Python tree (nn/mcts.py)."""
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.board import Board  # noqa: E402
from nn import cmcts, mcts  # noqa: E402
from nn.agents import MCTSAgent, random_init_numpy_net  # noqa: E402

if cmcts._load() is None:
    try:
        from engine.cbuild import build
        build(verbose=False)
        cmcts._ERR = None
    except SystemExit:
        pass
pytestmark = pytest.mark.skipif(cmcts._load() is None, reason="C library not built (python -m engine.cbuild)")


def _boards(n, seed, lo=0, hi=36):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = Board()
        for _ in range(rng.randint(lo, hi)):
            if b.is_terminal():
                break
            b.drop(rng.choice(b.legal_moves()))
        if not b.is_terminal():
            out.append(b)
    return out


def _run(boards, ev, sims, use_c, c_puct=1.5):
    os.environ["C4_MCTS_TREE_C"] = "1" if use_c else "0"
    try:
        roots = [mcts.root_from_board(b) for b in boards]
        mcts.search(roots, ev, sims, c_puct=c_puct)
        return roots
    finally:
        os.environ.pop("C4_MCTS_TREE_C", None)


@pytest.mark.parametrize("sims,c_puct", [(1, 1.5), (50, 1.5), (300, 1.5), (120, 4.0)])
def test_same_tree_statistics(sims, c_puct):
    net = random_init_numpy_net(2, 16, seed=sims)
    boards = _boards(40, sims)
    py = _run(boards, net, sims, False, c_puct)
    cc = _run(boards, net, sims, True, c_puct)
    for a, b in zip(py, cc):
        assert a.N == b.N and a.n == b.n
        assert a.W == pytest.approx(b.W, abs=1e-12) and a.P == pytest.approx(b.P, abs=1e-15)
        assert mcts.choose_move(a) == mcts.choose_move(b)
        assert mcts.principal_variation(a, 42) == mcts.principal_variation(b, 42)
        assert a.q() == pytest.approx(b.q(), abs=1e-12)


def test_exact_leaves_same():
    from nn.exact_tail import ExactTailEvaluator
    net = random_init_numpy_net(2, 16, seed=9)
    boards = _boards(12, 2, lo=24, hi=30)
    py = _run(boards, ExactTailEvaluator(net, from_ply=30), 80, False)
    cc = _run(boards, ExactTailEvaluator(net, from_ply=30), 80, True)
    assert [r.N for r in py] == [r.N for r in cc]


def test_agent_game_same_moves():
    net = random_init_numpy_net(2, 16, seed=4)

    def game(use_c):
        os.environ["C4_MCTS_TREE_C"] = "1" if use_c else "0"
        try:
            a, b = MCTSAgent(net, sims=60), MCTSAgent(net, sims=60)
            bd, mv = Board.from_moves("4"), []
            while not bd.is_terminal():
                c = (a if bd.to_move == 1 else b).choose(bd)
                mv.append(c)
                bd.drop(c)
            return mv
        finally:
            os.environ.pop("C4_MCTS_TREE_C", None)
    assert game(False) == game(True)


def test_noise_roots_stay_python():
    import numpy as np
    net = random_init_numpy_net(2, 16, seed=1)
    root = mcts.root_from_board(Board.from_moves("44"))
    mcts.expand_roots([root], net)
    mcts.add_dirichlet(root, np.random.default_rng(0), 0.3, 0.25)
    mcts.search([root], net, 30)                 # expanded root -> Python path, still works
    assert root.c_pv is None and sum(root.N) == 30
