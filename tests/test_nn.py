"""Tests for nn/ (numpy parts always; PyTorch parts only if torch is installed)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.board import Board  # noqa: E402
from engine.minimax import search  # noqa: E402
from nn import mcts  # noqa: E402
from nn.agents import MCTSAgent, NNMinimaxAgent, PolicyAgent, random_init_numpy_net  # noqa: E402
from nn.encode import encode, legal_mask, mirror_bits  # noqa: E402


def test_encode_planes():
    b = Board.from_moves("4453")          # P1 to move (4 stones)
    x = encode([b.current_bits()], [b.mask])[0]
    g = b.to_grid()
    me = np.array([[1 if v == 1 else 0 for v in r] for r in g])
    them = np.array([[1 if v == -1 else 0 for v in r] for r in g])
    assert (x[0] == me).all() and (x[1] == them).all()
    assert (x[2] == 1).all() and (x[3] == 1).all()       # first player to move
    b.drop(0)
    x = encode([b.current_bits()], [b.mask])[0]
    assert (x[3] == 0).all()                              # now second player to move
    assert x[0].sum() == 2 and x[1].sum() == 3            # side-to-move view


def test_legal_mask_and_mirror():
    b = Board.from_moves("444444")
    assert legal_mask([b.mask])[0].tolist() == [True, True, True, False, True, True, True]
    b2 = Board.from_moves("1")
    m = mirror_bits(np.array([b2.mask], dtype=np.uint64))[0]
    assert int(m) == Board.from_moves("7").mask


def test_numpy_net_shapes_and_probabilities():
    net = random_init_numpy_net()
    b = Board.from_moves("444444")
    pri, val, wdl = net.evaluate([b.current_bits(), 0], [b.mask, 0])
    assert pri.shape == (2, 7) and val.shape == (2,) and wdl.shape == (2, 3)
    assert abs(pri[0].sum() - 1) < 1e-5 and pri[0][3] == 0        # full column gets no probability
    assert np.all(np.abs(val) <= 1)


class _Uniform:
    def evaluate(self, curs, masks):
        n = len(curs)
        return np.full((n, 7), 1 / 7), np.zeros(n), np.full((n, 3), 1 / 3)


def test_mcts_finds_immediate_win_and_block():
    ev = _Uniform()
    win = Board.from_moves("445566")        # P1 to move, 4-5-6 on the bottom row: 3 or 7 wins
    root = mcts.root_from_board(win)
    mcts.search([root], ev, 200)
    assert mcts.choose_move(root) in (2, 6)
    block = Board.from_moves("47556")       # P2 to move; P1 threatens only column 3
    root = mcts.root_from_board(block)
    mcts.search([root], ev, 400)
    assert mcts.choose_move(root) == 2


def test_mcts_batch_and_visit_policy():
    ev = _Uniform()
    roots = [mcts.Node(0, 0) for _ in range(5)]
    mcts.search(roots, ev, 30)
    for r in roots:
        assert sum(r.N) == 30
        assert abs(sum(mcts.visit_policy(r)) - 1) < 1e-9


def test_agents_play_legal_moves():
    net = random_init_numpy_net()
    b = Board.from_moves("4453526163")
    for ag in (PolicyAgent(net), MCTSAgent(net, sims=20), NNMinimaxAgent(net, 2)):
        c = ag.choose(b)
        assert c in b.legal_moves()
        assert ag.last_result.col == c


def test_minimax_evaluator_hook_default_unchanged():
    b = Board.from_moves("4453526163")
    from engine.heuristic import evaluate_bits
    r1 = search(b, 4)
    r2 = search(b, 4, evaluator=evaluate_bits)
    assert (r1.col, r1.score) == (r2.col, r2.score)
    r3 = search(b, 3, evaluator=lambda me, opp: 0.0)       # flat evaluation still a legal search
    assert r3.col in b.legal_moves()


def test_elo_vs_fixed():
    from nn.bench import elo_vs_fixed
    assert abs(elo_vs_fixed([(1000, 5, 10)]) - 1000) < 1
    assert elo_vs_fixed([(1000, 9, 10)]) > 1300
    assert elo_vs_fixed([(1000, 10, 10)]) == 1800         # clamped when undefeated


def test_accuracy_summary():
    from nn.bench import accuracy_summary, load_accuracy_bench
    rows = load_accuracy_bench(limit=50)
    perfect = [min(r["optimal"]) for r in rows]
    s = accuracy_summary(rows, perfect)
    assert s["optimal"] == 1.0 and s["keeps_nl"] == 1.0


def test_export_matches_torch():
    torch = pytest.importorskip("torch")
    from nn.export import export_model
    from nn.model import build
    from nn.numpy_net import NumpyNet
    torch.manual_seed(0)
    m = build({"blocks": 2, "channels": 16})
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.2, 0.2)
    m.eval()
    path = os.path.join(ROOT, "runs", "_test_export.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        export_model(m, path)
        net = NumpyNet(path)
        b = Board.from_moves("4453526163")
        x = encode([b.current_bits(), 0], [b.mask, 0])
        with torch.no_grad():
            p, v = m(torch.from_numpy(x))
        p2, v2 = net.forward(x)
        assert np.abs(p.numpy() - p2).max() < 1e-4
        assert np.abs(v.numpy() - v2).max() < 1e-4
    finally:
        if os.path.exists(path):
            os.remove(path)
