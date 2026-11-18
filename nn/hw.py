"""Hardware helpers shared by every script: how many CPU workers, which GPU.

    python -m nn.hw              # what this machine has and what the scripts will use
    python -m nn.hw --bench      # time self-play on CPU workers vs the GPU server (training PC)

CPU
  default_workers()  every logical CPU thread (16 on the Ryzen 7 5800X).  The
                     C4_WORKERS environment variable overrides it; every
                     script's --workers flag overrides both.
  spawn_pool()       a multiprocessing pool whose workers each use ONE thread
                     for numpy/torch, so N workers = N busy cores and no
                     oversubscription.

GPU
  configure_cuda()   makes PyTorch see only GPUs it can actually run on.  On
                     the training PC that means the RTX 3050 (compute 8.6) and
                     not the GTX 750 Ti (compute 5.0; this PyTorch build needs
                     >= 7.5).  Why this exists: with
                         CUDA_DEVICE_ORDER=PCI_BUS_ID  CUDA_VISIBLE_DEVICES=0
                     (what the runbook used to say) device 0 IS the 750 Ti, so
                     PyTorch had no usable GPU and az_6x64 generations 1-53
                     and sup_6x64 trained on the CPU.  If CUDA_VISIBLE_DEVICES
                     points only at unusable GPUs, this switches it to the
                     best usable one and prints a warning saying so.
                     Must run before the first CUDA call; nn.common.pick_device()
                     calls it, so every trainer gets it automatically.
"""

import os
import shutil
import subprocess
import sys

_CONFIGURED = None


# ------------------------------------------------------------------ CPU
def cpu_threads():
    """Logical CPU threads this process may use."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, n)


def default_workers(reserve=0):
    """Worker processes for parallel games: all logical threads (minus ``reserve``).
    C4_WORKERS=<n> overrides."""
    env = os.environ.get("C4_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, cpu_threads() - reserve)


def single_thread_env():
    """Make child processes use one numpy/BLAS/torch thread each (inherited at spawn)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def single_thread_torch():
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001  (torch missing is fine)
        pass


def spawn_pool(workers, initializer=None, initargs=()):
    """multiprocessing 'spawn' pool, one thread per worker."""
    import multiprocessing as mp
    single_thread_env()
    return mp.get_context("spawn").Pool(workers, initializer=initializer, initargs=initargs)


# ------------------------------------------------------------------ GPU
def nvidia_gpus():
    """GPUs as nvidia-smi lists them (PCI bus order):
    [{"index", "name", "cc", "mem_mib", "free_mib"}].  [] without nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    cmd = [exe, "--query-gpu=index,name,compute_cap,memory.total,memory.free", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5:
            continue
        try:
            gpus.append({"index": int(p[0]), "name": p[1], "cc": float(p[2]),
                         "mem_mib": int(float(p[3])), "free_mib": int(float(p[4]))})
        except ValueError:
            continue
    return gpus


def torch_min_cc():
    """Lowest compute capability the installed PyTorch has kernels for (7.5 for
    torch 2.14 + CUDA 13), or None if torch / CUDA support is missing."""
    try:
        import torch
        arches = torch.cuda.get_arch_list()
    except Exception:  # noqa: BLE001
        return None
    ccs = [int(a.split("_")[1]) for a in arches if a.startswith("sm_") and a.split("_")[1].isdigit()]
    return min(ccs) / 10.0 if ccs else None


def _say(msg):
    print("[hw] " + msg, file=sys.stderr, flush=True)


def configure_cuda(verbose=True):
    """Restrict CUDA_VISIBLE_DEVICES to GPUs PyTorch can use (see module doc).
    Returns {"gpus", "usable", "visible", "note"}; safe to call many times."""
    global _CONFIGURED
    if _CONFIGURED is not None:
        return _CONFIGURED
    info = {"gpus": [], "usable": [], "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "note": ""}
    _CONFIGURED = info
    if os.environ.get("C4_NO_GPU_FIX") == "1":
        info["note"] = "C4_NO_GPU_FIX=1: left CUDA_VISIBLE_DEVICES alone"
        return info
    gpus = nvidia_gpus()
    info["gpus"] = gpus
    if not gpus:
        return info
    min_cc = torch_min_cc()
    if min_cc is None:
        return info
    try:
        import torch
        if torch.cuda.is_initialized():
            info["note"] = "CUDA already initialised; could not change the visible GPUs"
            return info
    except Exception:  # noqa: BLE001
        pass
    usable = [g for g in gpus if g["cc"] >= min_cc]
    info["usable"] = usable
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"          # indices = nvidia-smi indices
    best = max(usable, key=lambda g: g["mem_mib"]) if usable else None
    cur = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    by_idx = {g["index"]: g for g in gpus}
    if cur and all(x.strip().isdigit() for x in cur.split(",")):
        chosen = [by_idx.get(int(x)) for x in cur.split(",")]
        chosen = [g for g in chosen if g]
        ok = [g for g in chosen if g["cc"] >= min_cc]
        if ok and len(ok) == len(chosen):
            info["note"] = "CUDA_VISIBLE_DEVICES=%s (%s)" % (cur, ", ".join(g["name"] for g in ok))
        elif ok:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g["index"]) for g in ok)
            info["note"] = "dropped unusable GPUs from CUDA_VISIBLE_DEVICES=%s" % cur
        elif best:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best["index"])
            info["note"] = ("CUDA_VISIBLE_DEVICES=%s is %s (compute %.1f), which this PyTorch cannot use "
                            "(needs >= %.1f). Using GPU %d (%s) instead. To stop this warning: "
                            "unset CUDA_VISIBLE_DEVICES, or set it to %d."
                            % (cur, " + ".join(g["name"] for g in chosen) or "no GPU", chosen[0]["cc"] if chosen else 0,
                               min_cc, best["index"], best["name"], best["index"]))
            if verbose:
                _say("WARNING: " + info["note"])
        else:
            info["note"] = "no GPU with compute >= %.1f: running on the CPU" % min_cc
    elif not cur:
        if best:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best["index"])
            info["note"] = "using GPU %d (%s, %d MiB)" % (best["index"], best["name"], best["mem_mib"])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            info["note"] = ("no GPU this PyTorch can use (needs compute >= %.1f; found %s): running on the CPU"
                            % (min_cc, ", ".join("%s %.1f" % (g["name"], g["cc"]) for g in gpus)))
            if verbose:
                _say("WARNING: " + info["note"])
    info["visible"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info


def resolve_nn_device(requested):
    """--nn-device value -> device string for the GPU server, or "cpu".
    "cuda"/"cuda:N" fall back to "cpu" (with a warning) when PyTorch has no usable GPU.
    "server-cpu" runs the server on the CPU (tests only)."""
    req = (requested or "cpu").strip()
    if req == "server-cpu":
        return "cpu", True
    if not req.startswith("cuda"):
        return "cpu", False
    try:
        from .common import pick_device
        dev = pick_device(req if ":" in req else "auto")
    except Exception as e:  # noqa: BLE001  (no torch installed)
        _say("WARNING: --nn-device %s needs PyTorch (%s); using numpy on the CPU" % (req, e))
        return "cpu", False
    if dev.type != "cuda":
        _say("WARNING: --nn-device %s but no usable GPU; using numpy on the CPU" % req)
        return "cpu", False
    try:
        import torch
        torch.zeros(1, device=dev)
    except Exception as e:  # noqa: BLE001
        _say("WARNING: cannot use %s (%s); using numpy on the CPU" % (dev, e))
        return "cpu", False
    return str(dev), True


def gpu_memory_mib(device):
    """(allocated peak, reserved) MiB on a torch device, or (None, None)."""
    try:
        import torch
        if getattr(device, "type", str(device)).startswith("cuda"):
            return (round(torch.cuda.max_memory_allocated(device) / 2 ** 20),
                    round(torch.cuda.memory_reserved(device) / 2 ** 20))
    except Exception:  # noqa: BLE001
        pass
    return None, None


# ------------------------------------------------------------------ CLI
def report():
    print("CPU: %d logical threads -> default --workers %d%s" % (
        cpu_threads(), default_workers(), "  (C4_WORKERS is set)" if os.environ.get("C4_WORKERS") else ""))
    gpus = nvidia_gpus()
    if not gpus:
        print("GPU: nvidia-smi not found (CPU only)")
        return
    min_cc = torch_min_cc()
    print("GPUs (nvidia-smi / PCI order); this PyTorch needs compute >= %s:" % (min_cc or "?"))
    for g in gpus:
        ok = min_cc is not None and g["cc"] >= min_cc
        print("  %d  %-28s compute %.1f  %5d MiB (%d free)  %s" % (
            g["index"], g["name"], g["cc"], g["mem_mib"], g["free_mib"], "usable" if ok else "NOT usable"))
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    info = configure_cuda(verbose=False)
    print("CUDA_VISIBLE_DEVICES before: %r -> scripts will use: %r" % (before, info["visible"]))
    if info["note"]:
        print("  " + info["note"])
    try:
        import torch
        if torch.cuda.is_available():
            print("PyTorch sees: %s" % ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
        else:
            print("PyTorch sees no GPU")
    except ImportError:
        print("PyTorch not installed (fine for the website)")


def bench(n_games=200, sims=64):
    """Self-play throughput on this machine: the old way (each worker loads its
    own model; 8 workers on a GPU) vs the GPU server (all threads but one).
    Uses a freshly initialised 6x64 network, so it measures speed only."""
    import tempfile
    import time
    from . import gpu_server
    from .common import pick_device
    from .model import build, save_checkpoint
    from .train_selfplay import selfplay_worker
    dev = str(pick_device("auto"))
    ckpt = os.path.join(tempfile.mkdtemp(), "bench.pt")
    save_checkpoint(ckpt, build({"blocks": 6, "channels": 64}))
    n = default_workers()
    base = {"ckpt": ckpt, "parallel": 128, "sims": sims, "c_puct": 1.5, "temp_moves": 12,
            "dir_alpha": 1.0, "dir_eps": 0.25, "device": dev}

    def tasks(k, games):
        per = [games // k + (1 if i < games % k else 0) for i in range(k)]
        return [dict(base, games=g, seed=i) for i, g in enumerate(per) if g]

    print("self-play benchmark: %d games x %d sims per move, network 6x64, device %s" % (n_games, sims, dev))
    modes = [("local", min(n, 8) if dev.startswith("cuda") else n)]
    if dev.startswith("cuda"):
        modes.append(("server", max(1, n - 1)))
    for mode, k in modes:
        srv = None
        if mode == "server":
            srv = gpu_server.InferenceServer(k, device=dev).start()
            srv.load("selfplay", ckpt, amp=True)
            pool = spawn_pool(k, gpu_server.init_worker, (srv.handle(),))
        else:
            pool = spawn_pool(k)
        try:
            pool.map(selfplay_worker, tasks(k, k))                  # warm-up: load models, CUDA init
            t0 = time.time()
            parts = pool.map(selfplay_worker, tasks(k, n_games))
            dt = time.time() - t0
        finally:
            pool.terminate()
            stats = srv.stop() if srv else None
        pos = sum(len(p["z"]) for p in parts)
        extra = ""
        if stats:
            extra = "  avg GPU batch %.0f positions, server peak VRAM %s MiB" % (stats["avg_batch"], stats["peak_mib"])
        print("  %-6s %2d workers: %6.1f s  (%.0f games/min, %.0f positions/s)%s" % (
            mode, k, dt, 60 * n_games / dt, pos / dt, extra))


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        report()
