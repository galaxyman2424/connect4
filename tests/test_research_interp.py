"""Fast tests for research/interp*.py (CPU only)."""

import os
import random

import numpy as np
import pytest
import torch

from engine.board import Board, winning_cells_mask
from nn.encode import encode
from research.interp_core import (TorchNet, compute_concepts, flip_planes, linear_cka,
                                  train_probes, win_cells_bits)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(ROOT, "runs", "sup_6x64", "best.npz")
SMALL = os.path.join(ROOT, "runs", "distill_az_2x32", "best.npz")


def _pos(moves):
    b = Board.from_moves(moves)
    return b.current_bits(), b.mask


def _concepts(moves_list):
    cur, mask = zip(*[_pos(m) for m in moves_list])
    cur = np.array(cur, dtype=np.uint64)
    mask = np.array(mask, dtype=np.uint64)
    # fake scores: only legality matters for the board concepts
    legal = np.array([[Board.from_moves(m).can_play(c) for c in range(7)] for m in moves_list])
    scores = np.where(legal, 0, -128)
    return compute_concepts(cur, mask, scores)


def _random_positions(n, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = Board()
        for _ in range(rng.randrange(0, 30)):
            if b.is_terminal():
                break
            b.drop(rng.choice(b.legal_moves()))
        if not b.is_terminal():
            out.append(b)
    return out


def _nets():
    return [p for p in (NET, SMALL) if os.path.exists(p)]


@pytest.mark.skipif(not _nets(), reason="no exported network")
def test_forward_matches_numpy_reference():
    boards = _random_positions(24)
    x = encode([b.current_bits() for b in boards], [b.mask for b in boards])
    for path in _nets():
        net = TorchNet(path, "cpu")
        l, v, acts = net.run(torch.tensor(x), capture=tuple(net.layers))
        lr, vr = net.ref.forward(x)
        assert np.abs(l.numpy() - lr).max() < 1e-4
        assert np.abs(v.numpy() - vr).max() < 1e-4
        assert acts["stem"].shape == (24, net.channels, 6, 7)
        assert acts["pol_conv"].shape[1:] == (2, 6, 7) and acts["val_hidden"].shape == (24, 64)


@pytest.mark.skipif(not _nets(), reason="no exported network")
def test_block_skip_changes_outputs_and_zero_branch_equals_identity():
    boards = _random_positions(16, seed=1)
    x = torch.tensor(encode([b.current_bits() for b in boards], [b.mask for b in boards]))
    net = TorchNet(_nets()[0], "cpu")
    l0, _, _ = net.run(x)
    l1, _, _ = net.run(x, block_mode={1: "identity"})
    l2, _, _ = net.run(x, block_mode={1: "zero_branch"})
    assert (l0 - l1).abs().max() > 1e-3
    assert torch.allclose(l1, l2, atol=1e-6)        # block inputs are post-ReLU
    # editing a layer to itself is a no-op
    l3, _, _ = net.run(x, edit={"stem": lambda h: h})
    assert torch.allclose(l0, l3)


def test_win_cells_matches_engine():
    for b in _random_positions(60, seed=2):
        for pos in (b.current_bits(), b.current_bits() ^ b.mask):
            ref = winning_cells_mask(pos, b.mask)
            got = int(win_cells_bits(np.array([pos], np.uint64), np.array([b.mask], np.uint64))[0])
            assert got == ref


def test_concepts_hand_built():
    # x: cols 1,2,3 on the bottom row (0-idx 0,1,2), o: 5,5,6 (0-idx 4,4,5); x to move
    c, extra = _concepts(["152536", "15253"])
    # position 0: x to move, wins at column 3
    assert c["win_now"][0].tolist() == [False, False, False, True, False, False, False]
    assert not c["opp_win_now"][0].any()
    assert c["heights"][0].tolist() == [1, 1, 1, 0, 2, 1, 0]
    assert c["landing_row"][0].tolist() == [4, 4, 4, 5, 3, 4, 5]       # row from the TOP
    assert c["ply"][0] == 6 and c["first_to_move"][0] == 1
    assert c["phase"][0] == 0
    # position 1: o to move, must block column 3 (x threatens 0,1,2 + 3)
    assert c["opp_win_now"][1].tolist() == [False, False, False, True, False, False, False]
    assert not c["win_now"][1].any()
    assert c["first_to_move"][1] == 0
    # x's bottom-row threat cell (col 3, row 1 from bottom = odd) counts as odd for its owner
    assert c["own_odd_thr"][0] >= 1 and c["opp_odd_thr"][1] >= 1


def test_double_threat():
    # x: 2,3,4 on the bottom row with both 1 and 5 open -> two playable winning cells
    # "2737477": x plays 2,3,4,7 and o plays 7,7,7 -> 7 stones, o to move, and x threatens
    # both ends of its bottom-row three (0-idx columns 0 and 4)
    c, _ = _concepts(["2737477"])
    assert c["opp_double"][0] == 1
    assert c["opp_win_now"][0].tolist() == [True, False, False, False, True, False, False]


def test_flip_planes():
    cur, mask = _pos("4455")
    x = encode([cur], [mask])
    y = flip_planes(x)
    assert (y[:, 0] == x[:, 1]).all() and (y[:, 1] == x[:, 0]).all()
    assert (y[:, 3] == 1 - x[:, 3]).all()


def test_probe_helper_separable_and_shuffled():
    rng = np.random.default_rng(0)
    n, d = 3000, 20
    Xn = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d)
    ybin = (Xn @ w > 0)[:, None]
    ycat = np.argmax(Xn[:, :3], 1)
    yreg = (Xn @ w).astype(np.float32)
    X = torch.tensor(Xn)
    heads = [{"id": "b", "kind": "multibin", "K": None, "y": ybin},
             {"id": "c", "kind": "cat", "K": 3, "y": ycat},
             {"id": "r", "kind": "reg", "K": None, "y": yreg}]
    tr, va, te = np.arange(2000), np.arange(2000, 2500), np.arange(2500, 3000)
    s, _ = train_probes(X, heads, tr, va, te, rounds=5)
    assert s["b"] > 0.98 and s["c"] > 0.9 and s["r"] > 0.98
    sh, _ = train_probes(X, heads, tr, va, te, rounds=5, shuffle=True)
    assert abs(sh["b"] - 0.5) < 0.15 and sh["r"] < 0.2


def test_cka_bounds():
    rng = np.random.default_rng(1)
    A = torch.tensor(rng.standard_normal((500, 30)))
    assert abs(linear_cka(A, A) - 1) < 1e-6
    Q, _ = np.linalg.qr(rng.standard_normal((30, 30)))
    assert abs(linear_cka(A, A @ torch.tensor(Q)) - 1) < 1e-6        # rotation invariant
    assert linear_cka(A, torch.tensor(rng.standard_normal((500, 30)))) < 0.2
