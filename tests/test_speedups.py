"""Search / evaluation speed-ups must not change any move:
  * MCTSAgent network cache (nn/agents.CachedEvaluator)
  * batched depth-1 leaves in NN-minimax (engine/minimax prefetch hook)
  * torch backend for .npz files (nn/torch_net.py) == NumpyNet, incl. score head + mirror
  * PCA input projection (inproj.*) identical in torch model, numpy and torch_net
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.board import Board  # noqa: E402
from nn.agents import MCTSAgent, NNMinimaxAgent, random_init_numpy_net  # noqa: E402

def _valid(moves):
    try:
        return not Board.from_moves(moves).is_terminal()
    except ValueError:
        return False


POSITIONS = [m for m in ("", "4", "44", "4453", "435", "334455", "11223344", "4444333", "1234567",
                         "44455563", "2345432", "4455667712") if _valid(m)]


def _play(agent_a, agent_b, opening="44", max_plies=14):
    b = Board.from_moves(opening)
    out = []
    side = {1: agent_a, -1: agent_b}
    while not b.is_terminal() and len(out) < max_plies:
        c = side[b.to_move].choose(b)
        out.append(c)
        b.drop(c)
    return out


def test_mcts_cache_same_moves_fewer_calls():
    net = random_init_numpy_net(2, 16, seed=3)
    a1, b1 = MCTSAgent(net, sims=40, cache=False), MCTSAgent(net, sims=40, cache=False)
    a2, b2 = MCTSAgent(net, sims=40), MCTSAgent(net, sims=40)
    assert _play(a1, b1) == _play(a2, b2)
    assert a2.ev.computed < a2.ev.asked


def test_nn_minimax_prefetch_same_moves():
    net = random_init_numpy_net(2, 16, seed=5)

    class NoPrefetch:
        def __init__(self, ev):
            self.ev = ev

        def evaluate(self, c, m):         # one position per call -> batch size never changes
            outs = [self.ev.evaluate([ci], [mi]) for ci, mi in zip(c, m)]
            return tuple(np.concatenate([o[k] for o in outs]) for k in range(3))

    one = NoPrefetch(net)
    for d in (1, 2, 3):
        fast = NNMinimaxAgent(one, d)
        slow = NNMinimaxAgent(one, d)
        slow.evaluator.prefetch = None        # attribute shadows the method -> hook disabled
        slow_nodes = []
        for mv in POSITIONS:
            b = Board.from_moves(mv)
            if b.is_terminal():
                continue
            assert fast.choose(b) == slow.choose(b), (d, mv)
            assert fast.last_result.score == pytest.approx(slow.last_result.score, abs=1e-6)
            slow_nodes.append(slow.last_result.nodes)
        if d >= 2:
            assert fast.evaluator.calls < slow.evaluator.calls


def _score_npz(tmp_path, pca=None, seed=0):
    torch = pytest.importorskip("torch")
    from nn.export import export_model
    from nn.model import build
    torch.manual_seed(seed)
    cfg = {"blocks": 2, "channels": 16, "score_head": True}
    if pca:
        cfg["pca_k"] = pca
    m = build(cfg)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
    if pca:
        rng = np.random.default_rng(0)
        q, _ = np.linalg.qr(rng.normal(size=(84, pca)))
        m.set_input_projection(rng.uniform(0, 0.5, 84).astype(np.float32), q.astype(np.float32))
    m.eval()
    path = str(tmp_path / ("m%s.npz" % (pca or "")))
    export_model(m, path, {"play_head": "combo", "play_temp": 0.5, "play_policy_weight": 2.0,
                           "play_mirror": True})
    return m, path


@pytest.mark.parametrize("pca", [None, 20])
def test_torch_backend_matches_numpy(tmp_path, pca):
    torch = pytest.importorskip("torch")
    from nn.common import load_evaluator
    from nn.encode import encode
    m, path = _score_npz(tmp_path, pca)
    boards = [Board.from_moves(s) for s in POSITIONS]
    curs, masks = [b.current_bits() for b in boards], [b.mask for b in boards]
    npv = load_evaluator(path, "cpu", backend="numpy")
    tov = load_evaluator(path, "cpu", backend="torch")
    p1, v1, w1 = npv.evaluate(curs, masks)
    p2, v2, w2 = tov.evaluate(curs, masks)
    assert np.allclose(p1, p2, atol=1e-4) and np.allclose(v1, v2, atol=1e-4) and np.allclose(w1, w2, atol=1e-4)
    assert (p1.argmax(1) == p2.argmax(1)).all()
    # and the numpy export equals the torch training model (projection included)
    x = encode(curs, masks)
    with torch.no_grad():
        pt, vt, st = m.forward_all(torch.from_numpy(x))
    pn, vn, sn = npv.forward_all(x)
    assert np.allclose(pt.numpy(), pn, atol=1e-4) and np.allclose(st.numpy() * 10.0, sn, atol=1e-3)


def test_backend_env_switch(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from nn.common import load_evaluator
    from nn.torch_net import TorchNpzEvaluator
    _, path = _score_npz(tmp_path)
    monkeypatch.setenv("C4_NN_BACKEND", "torch")
    assert isinstance(load_evaluator(path, "cpu"), TorchNpzEvaluator)
    monkeypatch.setenv("C4_NN_BACKEND", "numpy")
    assert not isinstance(load_evaluator(path, "cpu"), TorchNpzEvaluator)


def test_wilson_ci():
    from nn.bench import wilson_ci
    lo, hi = wilson_ci(2925, 3000)
    assert lo < 0.975 < hi and 0.004 < (hi - lo) / 2 < 0.007
    assert wilson_ci(0, 0) == (None, None)


def test_fit_input_pca_orthonormal():
    from nn.train_score import fit_input_pca
    rng = np.random.default_rng(0)
    grids = rng.integers(0, 2, (500, 2, 6, 7)).astype(np.uint8)
    info = fit_input_pca(grids, 16)
    U = info["components"]
    assert U.shape == (84, 16) and np.allclose(U.T @ U, np.eye(16), atol=1e-4)
    assert 0 < info["explained_variance"] < 1


def test_pca_lda_labels():
    from research.pca_lda import labels
    sc = np.array([[-128, 1, 0, 0, -2, 3, 3], [0, 0, 0, 0, 0, 0, -128], [-5, -1, -1, -2, -3, -4, -5]], np.int8)
    y, cls = labels(sc, "outcome")
    assert list(y) == [0, 1, 2] and cls == ["win", "draw", "loss"]
    y, _ = labels(sc, "bestcol")
    assert list(y) == [5, 3, 2]          # most central optimal column


def test_exact_tail_and_hybrid():
    from engine.minimax import search
    from nn import mcts
    from nn.exact_tail import ExactTailEvaluator, HybridAgent, _board_from_bits, exact_value
    import random
    rng = random.Random(4)
    n = 0
    while n < 15:
        b = Board()
        for _ in range(rng.randint(30, 36)):
            if b.is_terminal():
                break
            b.drop(rng.choice(b.legal_moves()))
        if b.is_terminal():
            continue
        n += 1
        bb = _board_from_bits(b.current_bits(), b.mask)
        assert (bb.p1, bb.p2, bb.to_move, bb.moves_played) == (b.p1, b.p2, b.to_move, b.moves_played)
        v, col, ok = exact_value(b.current_bits(), b.mask)
        full = search(b, 42 - b.moves_played)
        assert ok and col == full.col and (v > 0) == (full.score > 0) and (v < 0) == (full.score < 0)
    net = random_init_numpy_net(2, 16, seed=1)
    tail = ExactTailEvaluator(net, from_ply=30)
    root = mcts.root_from_board(Board.from_moves("4444333322"))   # ply 10: leaves never reach ply 30
    mcts.search([root], tail, 30)
    late = None
    while late is None:                                              # a random position at ply 26
        b = Board()
        for _ in range(26):
            if b.is_terminal():
                break
            b.drop(rng.choice(b.legal_moves()))
        if not b.is_terminal() and b.moves_played == 26:
            late = b
    if True:
        r2 = mcts.root_from_board(late)
        mcts.search([r2], ExactTailEvaluator(net, from_ply=27), 40)
        assert any(ch is not None and ch.terminal is not None for ch in r2.children)
        ag = HybridAgent(net, root_from=20, sims=20)
        assert ag.choose(late) in late.legal_moves() and ag.last_info["mode"] == "exact_search"


def test_teacher_targets_shapes(tmp_path):
    pytest.importorskip("torch")
    import torch
    from nn.train_score import teacher_targets
    _, path = _score_npz(tmp_path)
    rng = np.random.default_rng(0)
    D = {"grids": rng.integers(0, 2, (50, 2, 6, 7)).astype(np.uint8),
         "first": rng.integers(0, 2, 50).astype(bool), "legal": np.ones((50, 7), bool)}
    D["legal"][:, 0] = False
    tp, ts, n_sc = teacher_targets([path, path], D, torch.device("cpu"))
    assert tp.shape == (50, 7) and ts.shape == (50, 7) and n_sc == 2
    assert np.allclose(tp.sum(1), 1, atol=1e-5) and (tp[:, 0] == 0).all() and (ts[:, 0] == 0).all()
