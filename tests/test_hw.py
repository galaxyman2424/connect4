"""Tests for nn/hw.py (worker counts, GPU selection) and nn/gpu_server.py.

The GPU server is exercised with device="cpu" so it runs anywhere; on the
training PC the same code runs with device="cuda:0".
"""
import os
import random
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.board import Board  # noqa: E402
from nn import hw  # noqa: E402


def _positions(n=40, seed=0):
    rng = random.Random(seed)
    curs, masks = [], []
    while len(curs) < n:
        b = Board()
        for _ in range(rng.randint(0, 20)):
            if b.is_terminal():
                break
            b.drop(rng.choice(b.legal_moves()))
        if not b.is_terminal():
            curs.append(b.current_bits())
            masks.append(b.mask)
    return curs, masks


def test_default_workers_uses_every_thread(monkeypatch):
    monkeypatch.delenv("C4_WORKERS", raising=False)
    assert hw.default_workers() == hw.cpu_threads()
    assert hw.default_workers(reserve=1) == max(1, hw.cpu_threads() - 1)
    monkeypatch.setenv("C4_WORKERS", "5")
    assert hw.default_workers() == 5


FAKE_GPUS = [{"index": 0, "name": "NVIDIA GeForce GTX 750 Ti", "cc": 5.0, "mem_mib": 2048, "free_mib": 2000},
             {"index": 1, "name": "NVIDIA GeForce RTX 3050", "cc": 8.6, "mem_mib": 8192, "free_mib": 8000}]


def _configure(monkeypatch, cvd):
    monkeypatch.setattr(hw, "_CONFIGURED", None)
    monkeypatch.setattr(hw, "nvidia_gpus", lambda: [dict(g) for g in FAKE_GPUS])
    monkeypatch.setattr(hw, "torch_min_cc", lambda: 7.5)
    monkeypatch.delenv("C4_NO_GPU_FIX", raising=False)
    if cvd is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    return hw.configure_cuda(verbose=False)


def test_configure_cuda_repairs_the_750ti_setting(monkeypatch):
    # the old runbook: PCI order + device 0 = the GTX 750 Ti -> PyTorch had no usable GPU
    info = _configure(monkeypatch, "0")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "750 Ti" in info["note"] and "RTX 3050" in info["note"]


def test_configure_cuda_picks_the_3050_when_unset(monkeypatch):
    _configure(monkeypatch, None)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_configure_cuda_keeps_a_good_setting(monkeypatch):
    _configure(monkeypatch, "1")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    _configure(monkeypatch, "0,1")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"           # unusable one dropped


def _exported_npz(torch):
    from nn.export import export_model
    from nn.model import build
    torch.manual_seed(0)
    model = build({"blocks": 2, "channels": 16, "value_hidden": 16})
    with torch.no_grad():                     # non-trivial BatchNorm statistics
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.2, 0.2)
                m.running_var.uniform_(0.5, 1.5)
    model.eval()
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    export_model(model, path)
    return path


def test_folded_net_matches_numpy_net():
    torch = pytest.importorskip("torch")
    from nn.common import TorchEvaluator
    from nn.gpu_server import load_torch_model
    from nn.numpy_net import NumpyNet
    path = _exported_npz(torch)
    try:
        curs, masks = _positions()
        a = NumpyNet(path).evaluate(curs, masks)
        b = TorchEvaluator(load_torch_model(path, torch.device("cpu")), torch.device("cpu")).evaluate(curs, masks)
        for x, y in zip(a, b):
            assert np.allclose(x, y, atol=1e-4)
    finally:
        os.remove(path)


def _remote_eval(args):
    curs, masks = args
    from nn import gpu_server
    return gpu_server.remote_evaluator("m").evaluate(curs, masks)


def test_inference_server_matches_local_evaluation():
    torch = pytest.importorskip("torch")
    from nn import gpu_server
    from nn.numpy_net import NumpyNet
    path = _exported_npz(torch)
    srv = gpu_server.InferenceServer(2, device="cpu").start()
    try:
        srv.load("m", path)
        curs, masks = _positions(60)
        chunks = [(curs[i:i + 15], masks[i:i + 15]) for i in range(0, 60, 15)]
        pool = hw.spawn_pool(2, gpu_server.init_worker, (srv.handle(),))
        try:
            parts = pool.map(_remote_eval, chunks)
        finally:
            pool.terminate()
        ref = NumpyNet(path).evaluate(curs, masks)
        got = [np.concatenate([p[k] for p in parts]) for k in range(3)]
        for x, y in zip(ref, got):
            assert np.allclose(x, y, atol=1e-4)
        st = srv.stop()
        assert st["positions"] == 60 and st["calls"] >= 1
    finally:
        srv.stop()
        os.remove(path)


def test_selfplay_worker_through_server():
    torch = pytest.importorskip("torch")
    from nn import gpu_server
    from nn.model import build, save_checkpoint
    from nn.train_selfplay import selfplay_worker
    d = tempfile.mkdtemp()
    ckpt = os.path.join(d, "latest.pt")
    torch.manual_seed(0)
    save_checkpoint(ckpt, build({"blocks": 1, "channels": 8, "value_hidden": 8}))
    srv = gpu_server.InferenceServer(2, device="cpu").start()
    try:
        srv.load("selfplay", ckpt, amp=True)
        task = {"ckpt": ckpt, "device": "cpu", "games": 3, "parallel": 3, "sims": 8, "c_puct": 1.5,
                "temp_moves": 4, "dir_alpha": 1.0, "dir_eps": 0.25}
        pool = hw.spawn_pool(2, gpu_server.init_worker, (srv.handle(),))
        try:
            parts = pool.map(selfplay_worker, [dict(task, seed=1), dict(task, seed=2)])
        finally:
            pool.terminate()
        for p in parts:
            assert len(p["lengths"]) == 3
            assert len(p["z"]) == sum(p["lengths"]) and p["pi"].shape == (len(p["z"]), 7)
            assert np.allclose(p["pi"].sum(1), 1.0, atol=1e-5)
    finally:
        srv.stop()


def test_run_tasks_same_games_with_and_without_server():
    torch = pytest.importorskip("torch")
    from nn import bench
    path = _exported_npz(torch)
    try:
        spec = ("nn", path, "mcts", 12)
        tasks = [(spec, ("minimax", 2), o, af) for o in ("44", "435") for af in (True, False)]
        a = bench.run_tasks(tasks, 2, nn_device="cpu")
        b = bench.run_tasks(tasks, 2, nn_device="server-cpu")
        assert [g["moves"] for g in a] == [g["moves"] for g in b]
    finally:
        os.remove(path)
