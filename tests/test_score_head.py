"""Score-head nets (nn/train_score.py): export matches PyTorch, old callers still work."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")

from engine.board import Board  # noqa: E402
from nn.encode import encode  # noqa: E402
from nn.export import export_model  # noqa: E402
from nn.model import SCORE_SCALE, build  # noqa: E402
from nn.numpy_net import NumpyNet  # noqa: E402


def _net():
    torch.manual_seed(0)
    m = build({"blocks": 2, "channels": 16, "score_head": True})
    for mod in m.modules():                       # non-trivial BatchNorm stats so folding is tested
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
    return m.eval()


def test_export_score_head_matches_torch(tmp_path):
    m = _net()
    path = str(tmp_path / "m.npz")
    export_model(m, path, {"play_head": "score", "play_temp": 0.5})
    boards = [Board.from_moves(s) for s in ("", "4", "4453", "1525364"[:6], "3344556")]
    x = encode([b.current_bits() for b in boards], [b.mask for b in boards])
    with torch.no_grad():
        p, v, s = m.forward_all(torch.from_numpy(x))
    n = NumpyNet(path)
    p2, v2, s2 = n.forward_all(x)
    assert np.allclose(p.numpy(), p2, atol=1e-4) and np.allclose(v.numpy(), v2, atol=1e-4)
    assert np.allclose(s.numpy() * SCORE_SCALE, s2, atol=1e-3)
    # forward() keeps the old two-output contract
    assert len(m(torch.from_numpy(x))) == 2 and len(n.forward(x)) == 2


def test_play_head_score_picks_argmax_of_scores(tmp_path):
    m = _net()
    path = str(tmp_path / "m.npz")
    export_model(m, path, {"play_head": "score", "play_temp": 0.5, "play_mirror": False})
    n = NumpyNet(path)
    b = Board.from_moves("4453")
    pri, val, wdl = n.evaluate([b.current_bits()], [b.mask])
    _, _, s = n.forward_all(encode([b.current_bits()], [b.mask]))
    assert int(np.argmax(pri[0])) == int(np.argmax(s[0])) and abs(pri[0].sum() - 1) < 1e-6


def test_old_net_without_score_head_unchanged(tmp_path):
    torch.manual_seed(1)
    m = build({"blocks": 1, "channels": 8}).eval()
    path = str(tmp_path / "old.npz")
    export_model(m, path, {"play_head": "score"})      # meta asks for score, but there is no head
    n = NumpyNet(path)
    assert not n.has_score and n.play_head == "policy"
