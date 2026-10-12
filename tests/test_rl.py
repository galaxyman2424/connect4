"""Tests for rl/ (the five extra learning methods).  numpy parts always;
PyTorch parts only if torch is installed.  ~10 s."""
import json
import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.board import Board  # noqa: E402
from engine.heuristic import evaluate_bits  # noqa: E402
from rl import vecenv as V  # noqa: E402
from rl.evolve import CMAES, board_from_bits  # noqa: E402
from rl.features import HAND_WEIGHTS, WeightedEvaluator  # noqa: E402
from rl.ntuple import NTupleNet, lambda_returns, random_tuples  # noqa: E402
from rl.ppo import gae  # noqa: E402


def random_boards(n, seed=0, max_moves=30):
    r = random.Random(seed)
    out = []
    while len(out) < n:
        b = Board()
        for _ in range(r.randint(0, max_moves)):
            if b.is_terminal():
                break
            b.drop(r.choice(b.legal_moves()))
        if not b.is_terminal():
            out.append(b)
    return out


def test_vecenv_matches_board():
    rng = np.random.default_rng(0)
    g = V.VecGames(48, rng)
    boards = [Board() for _ in range(48)]
    for _ in range(42):
        alive = [i for i in range(48) if not boards[i].is_terminal()]
        if not alive:
            break
        idx = np.array(alive)
        lg = g.legal()[idx]
        cols = V.random_legal(lg, rng)
        for k, i in enumerate(alive):
            assert lg[k].tolist() == [c in boards[i].legal_moves() for c in range(7)]
            boards[i].drop(int(cols[k]))
        won, full = g.step(cols, idx)
        for k, i in enumerate(alive):
            assert bool(won[k]) == (boards[i].check_win() != 0)
            assert bool(full[k]) == boards[i].is_full()
            assert int(g.cur[i]) == boards[i].current_bits() and int(g.mask[i]) == boards[i].mask


def test_winning_moves_and_mirror():
    b = Board.from_moves("4455")          # P1 to move: 3 or 6 make three-in-a-row, not a win yet
    b.drop(2)                             # P1 plays col 3 -> X X X on the bottom (cols 3,4,5)
    b.drop(0)                             # P2 elsewhere; P1 to move with a double threat
    wm = V.winning_moves(np.array([b.current_bits()], np.uint64), np.array([b.mask], np.uint64))[0]
    assert wm[1] and wm[5]
    m = V.mirror(np.array([Board.from_moves("1").mask], np.uint64))[0]
    assert int(m) == Board.from_moves("7").mask


def test_weighted_evaluator_reproduces_hand_heuristic():
    ev = WeightedEvaluator(HAND_WEIGHTS)
    ev_slow = WeightedEvaluator(list(HAND_WEIGHTS[:5]) + [0, 0, 0, 0, 1e-12])    # forces the full feature path
    for b in random_boards(200):
        me, opp = b.current_bits(), b.current_bits() ^ b.mask
        assert ev(me, opp) == evaluate_bits(me, opp)
        assert abs(ev_slow(me, opp) - evaluate_bits(me, opp)) < 1e-6


def test_cmaes_minimises_a_quadratic():
    target = np.array([3.0, -2.0, 1.0, 0.5])
    es = CMAES(np.zeros(4), 1.0, popsize=10, seed=0)
    for _ in range(80):
        xs = es.ask()
        es.tell(xs, [-float(((x - target) ** 2).sum()) for x in xs])
    assert np.abs(es.mean - target).max() < 0.05
    es2 = CMAES.from_state(json.loads(json.dumps(es.state())), seed=1)
    assert np.allclose(es2.mean, es.mean)


def test_board_from_bits_roundtrip():
    for b in random_boards(50, seed=3):
        b2 = board_from_bits(b.current_bits(), b.mask)
        assert b2 == b and b2.moves_played == b.moves_played


def test_ntuple_symmetry_and_values():
    net = NTupleNet(random_tuples(12, 6, seed=1))
    net.lut[:] = np.random.default_rng(0).normal(0, 0.1, net.lut.size).astype(np.float32)
    boards = random_boards(30, seed=5)
    mover = np.array([b.current_bits() ^ b.mask for b in boards], np.uint64)    # player who just moved
    mask = np.array([b.mask for b in boards], np.uint64)
    v = net.value(mover, mask)
    vm = net.value(V.mirror(mover), V.mirror(mask))
    assert np.allclose(v, vm)                       # tuple + mirrored tuple share the table
    assert (np.abs(v) < 1).all()
    vals, lg = net.move_values(np.array([boards[0].current_bits()], np.uint64), np.array([boards[0].mask], np.uint64))
    assert np.isinf(vals[0][~lg[0]]).all()


def test_ntuple_takes_immediate_win(tmp_path):
    from rl.agents import NTupleAgent
    net = NTupleNet(random_tuples(8, 6, seed=2))
    b = Board.from_moves("121212")            # P1 to move, wins in column 1
    assert NTupleAgent(net, 0).choose(b) == 0
    p = str(tmp_path / "nt.npz")
    net.save(p, {"games": 5})
    net2 = NTupleNet.load(p)
    assert net2.meta["games"] == 5 and np.array_equal(net2.tuples, net.tuples)
    assert NTupleAgent(net2, 2).choose(Board.from_moves("4455")) in range(7)


def test_lambda_returns():
    # two non-terminal afterstates then a winning move by the player of afterstate 1
    g = lambda_returns(np.array([0.1, 0.2]), 1.0, lam=0.0)
    assert np.allclose(g, [-0.2, -1.0])                    # TD(0): negated next value
    g = lambda_returns(np.array([0.1, 0.2]), 1.0, lam=1.0)
    assert np.allclose(g, [1.0, -1.0])                     # Monte Carlo: +-final, alternating
    g = lambda_returns(np.array([0.1, 0.2]), 1.0, lam=1.0, gamma=0.5)
    assert np.allclose(g, [0.25, -0.5])


def test_gae():
    adv = gae(np.array([0.0, 0.5]), 1.0, lam=1.0)
    assert np.allclose(adv, [1.0, 0.5])                    # lambda=1: return - value
    adv = gae(np.array([0.0, 0.5]), 1.0, lam=0.0)
    assert np.allclose(adv, [0.5, 0.5])                    # one-step TD errors


def test_evolved_agent_and_bench_specs(tmp_path):
    from nn import bench
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"weights": list(HAND_WEIGHTS)}))
    a = bench.make_agent(("weights", str(p), 2))
    h = bench.make_agent(("minimax", 2))
    for b in random_boards(10, seed=9):
        assert a.choose(b) == h.choose(b)                  # same weights -> same moves
    assert bench.spec_name(("weights", str(p), 2)) == "evo_d2"


def test_cost_file_accumulates(tmp_path):
    from rl.meter import CostMeter, write_cost
    seg = {"wall_sec": 3600, "cpu_core_sec": 7200, "gpu_sec": 3600, "gpu_wh": 100, "cpu_wh": 50, "energy_wh": 150,
           "cpu_wh_source": "estimate"}
    write_cost(str(tmp_path), seg)
    d = write_cost(str(tmp_path), dict(seg, id="x"))
    d = write_cost(str(tmp_path), dict(seg, id="x"))        # same id replaces, does not add
    assert len(d["segments"]) == 2
    assert d["total"]["energy_kwh"] == 0.3 and d["total"]["cpu_core_hours"] == 4.0
    m = CostMeter(uses_gpu=False, interval=0.05).start()
    sum(i * i for i in range(200000))
    s = m.stop()
    assert s["wall_sec"] >= 0 and s["cpu_core_sec"] > 0 and s["gpu_wh"] == 0


# ------------------------------------------------------------- torch parts
torch = pytest.importorskip("torch")


def test_folded_torch_net_matches_numpy(tmp_path):
    from nn.export import export_model
    from nn.model import build
    from nn.numpy_net import NumpyNet
    from rl.torchutil import load_gpu_evaluator
    m = build({"blocks": 1, "channels": 8})
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 1.5)
    m.eval()
    p = str(tmp_path / "m.npz")
    export_model(m, p)
    boards = random_boards(20, seed=4)
    cur = [b.current_bits() for b in boards]
    mask = [b.mask for b in boards]
    pa, va, _ = NumpyNet(p).evaluate(cur, mask)
    pb, vb, _ = load_gpu_evaluator(p, "cpu").evaluate(cur, mask)
    assert np.abs(pa - pb).max() < 1e-4 and np.abs(va - vb).max() < 1e-4
