"""Native net (engine/cminimax/net.c via nn/native.py) == NumpyNet, and native MCTS == C/Python tree."""
import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.board import Board  # noqa: E402
from nn import mcts, native  # noqa: E402
from nn.agents import MCTSAgent  # noqa: E402

if native._load() is None:
    try:
        from engine.cbuild import build
        build(verbose=False)
        native._ERR = None
    except SystemExit:
        pass
pytestmark = pytest.mark.skipif(native._load() is None, reason="C library not built (python -m engine.cbuild)")


def _boards(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = Board()
        for _ in range(rng.randint(0, 34)):
            if b.is_terminal():
                break
            b.drop(rng.choice(b.legal_moves()))
        if not b.is_terminal():
            out.append(b)
    return out


def _npz(tmp_path, score=True, pca=0, meta=None, blocks=2, channels=16, seed=0):
    """A random exported net written with numpy only (no torch needed, e.g. on the Windows PC).
    Same file layout as nn/export.py (BatchNorm already folded)."""
    import json
    rng = np.random.default_rng(seed)
    C = channels
    a = {"stem.w": rng.normal(0, .2, (C, 4, 3, 3)), "stem.b": rng.normal(0, .1, C)}
    for i in range(blocks):
        for k in ("c1", "c2"):
            a["b%d.%s.w" % (i, k)] = rng.normal(0, .05, (C, C, 3, 3))
            a["b%d.%s.b" % (i, k)] = rng.normal(0, .05, C)
    a["pol.w"], a["pol.b"] = rng.normal(0, .2, (2, C)), rng.normal(0, .1, 2)
    a["pol_fc.w"], a["pol_fc.b"] = rng.normal(0, .1, (7, 84)), rng.normal(0, .1, 7)
    a["val.w"], a["val.b"] = rng.normal(0, .2, (4, C)), rng.normal(0, .1, 4)
    a["val_fc1.w"], a["val_fc1.b"] = rng.normal(0, .1, (32, 168)), rng.normal(0, .1, 32)
    a["val_fc2.w"], a["val_fc2.b"] = rng.normal(0, .1, (3, 32)), rng.normal(0, .1, 3)
    cfg = {"blocks": blocks, "channels": C, "policy_channels": 2, "value_channels": 4, "value_hidden": 32}
    if score:
        a["sc.w"], a["sc.b"] = rng.normal(0, .2, (8, C)), rng.normal(0, .1, 8)
        a["sc_fc1.w"], a["sc_fc1.b"] = rng.normal(0, .1, (48, 336)), rng.normal(0, .1, 48)
        a["sc_fc2.w"], a["sc_fc2.b"] = rng.normal(0, .2, (7, 48)), rng.normal(0, .1, 7)
        cfg.update(score_head=True, score_channels=8, score_hidden=48)
    if pca:
        q, _ = np.linalg.qr(rng.normal(size=(84, pca)))
        a["inproj.mean"], a["inproj.U"] = rng.uniform(0, .5, 84), q
        cfg["pca_k"] = pca
    a = {k: np.asarray(v, np.float32) for k, v in a.items()}
    a["config_json"] = np.array(json.dumps(cfg))
    a["meta_json"] = np.array(json.dumps(meta or {}))
    p = str(tmp_path / ("n%d_%d_%d_%d.npz" % (int(score), pca, blocks, seed)))
    np.savez(p, **a)
    return p


CASES = [
    dict(score=False),
    dict(score=True, meta={"play_head": "score", "play_temp": 0.5, "play_mirror": True}),
    dict(score=True, meta={"play_head": "combo", "play_temp": 0.5, "play_policy_weight": 2.0, "play_mirror": True}),
    dict(score=True, pca=20, meta={"play_head": "combo", "play_temp": 0.5, "play_policy_weight": 1.0, "play_mirror": True}),
    dict(score=True, blocks=3, channels=24, meta={"play_head": "policy"}),
]


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("value_source", ["wdl", "score"])
def test_native_matches_numpy(tmp_path, monkeypatch, case, value_source):
    from nn.numpy_net import NumpyNet
    monkeypatch.setenv("C4_VALUE_SOURCE", value_source)
    path = _npz(tmp_path, **CASES[case])
    boards = _boards(37, case)          # 37 > one 8-position chunk: exercises the chunking
    curs, masks = [b.current_bits() for b in boards], [b.mask for b in boards]
    p1, v1, w1 = NumpyNet(path).evaluate(curs, masks)
    p2, v2, w2 = native.NativeNet(path).evaluate(curs, masks)
    assert np.allclose(p1, p2, atol=2e-5) and np.allclose(v1, v2, atol=2e-5) and np.allclose(w1, w2, atol=2e-5)
    assert (p1.argmax(1) == p2.argmax(1)).all()


def test_native_search_and_agent(tmp_path):
    from nn.numpy_net import NumpyNet
    path = _npz(tmp_path, score=True, meta={"play_head": "combo", "play_temp": 0.5, "play_policy_weight": 2.0,
                                             "play_mirror": True})
    npn, nat = NumpyNet(path), native.NativeNet(path)
    boards = _boards(30, 5)
    r1 = [mcts.root_from_board(b) for b in boards]
    r2 = [mcts.root_from_board(b) for b in boards]
    mcts.search(r1, npn, 150)
    mcts.search(r2, nat, 150)          # fully native
    same = sum(a.N == b.N for a, b in zip(r1, r2))
    assert same >= 28                  # float32 sums differ at ~1e-6: an exact visit tie can flip rarely
    assert sum(mcts.choose_move(a) == mcts.choose_move(b) for a, b in zip(r1, r2)) >= 29

    def game(ev):
        a, b, bd, mv = MCTSAgent(ev, sims=80), MCTSAgent(ev, sims=80), Board.from_moves("4"), []
        while not bd.is_terminal():
            c = (a if bd.to_move == 1 else b).choose(bd)
            mv.append(c)
            bd.drop(c)
        return mv, a
    m1, _ = game(npn)
    m2, ag = game(nat)
    assert m1 == m2
    asked, computed = ag.ev.native_cache.stats()
    assert 0 < computed < asked


def test_backend_switch(tmp_path):
    from nn.common import load_evaluator
    path = _npz(tmp_path, score=False)
    assert type(load_evaluator(path, "cpu", backend="native")).__name__ == "NativeNet"
