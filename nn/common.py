"""Shared helpers for nn/: paths, run folders, metrics log, device choice."""

import datetime
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RUNS = os.path.join(ROOT, "runs")            # one folder per training run
NN_DATA = os.path.join(ROOT, "nn_data")      # generated datasets (git-ignored)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "experiments", "results")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------ runs
class Run:
    """A training run folder:

    runs/<name>/
      config.json      every setting used (+ machine info)   [committed]
      metrics.jsonl    one JSON object per line, appended     [committed]
      model.npz        latest exported weights (numpy)        [committed]
      best.npz         best weights by the run's own criterion[committed]
      eval.json        written by nn/evaluate.py              [committed]
      notes.md         your own notes for the presentation    [committed]
      checkpoints/     .pt files, replay buffer               [git-ignored]
    """

    def __init__(self, name, create=True):
        self.name = name
        self.dir = os.path.join(RUNS, name)
        self.ckpt_dir = os.path.join(self.dir, "checkpoints")
        if create:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def exists(self):
        return os.path.isfile(self.path("config.json"))

    def write_config(self, cfg):
        with open(self.path("config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def read_config(self):
        with open(self.path("config.json"), encoding="utf-8") as f:
            return json.load(f)

    def log_metrics(self, record):
        record = dict(record)
        record.setdefault("time", now_iso())
        with open(self.path("metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_metrics(self):
        out = []
        p = self.path("metrics.jsonl")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        return out


def machine_info():
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return info


# --------------------------------------------------------------- devices
def pick_device(requested="auto"):
    """'auto' (or None) -> the CUDA GPU with the most memory among those this
    PyTorch build can actually run a kernel on (so an unsupported old card,
    like the GTX 750 Ti that used to sit next to the RTX 3050, is skipped),
    else CPU.  'cpu', 'cuda', 'cuda:1' are honoured as given."""
    import torch
    if requested and requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best, best_mem = None, -1
    for i in range(torch.cuda.device_count()):
        try:
            (torch.ones(8, device="cuda:%d" % i) * 2).sum().item()   # fails if torch has no kernels for it
            mem = torch.cuda.get_device_properties(i).total_memory
        except Exception:
            continue
        if mem > best_mem:
            best, best_mem = i, mem
    if best is None:
        log("WARNING: CUDA reports %d GPU(s) but none can run this PyTorch build - using the CPU"
            % torch.cuda.device_count())
        return torch.device("cpu")
    return torch.device("cuda:%d" % best)


class TorchEvaluator:
    """Batched evaluate(curs, masks) on a PyTorch model (GPU or CPU).
    Same return format as numpy_net.NumpyNet.evaluate."""

    def __init__(self, model, device, amp=False):
        import torch
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.amp = amp and device.type == "cuda"

    def evaluate(self, curs, masks):
        import numpy as np
        from .encode import encode, legal_mask
        torch = self.torch
        x = torch.from_numpy(encode(curs, masks)).to(self.device, non_blocking=True)
        legal = torch.from_numpy(legal_mask(masks)).to(self.device)
        with torch.inference_mode():
            if self.amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    p, v = self.model(x)
            else:
                p, v = self.model(x)
            p = p.float().masked_fill(~legal, float("-inf"))
            pri = torch.softmax(p, 1)
            wdl = torch.softmax(v.float(), 1)
            val = wdl[:, 0] - wdl[:, 2]
        return (pri.cpu().numpy().astype(np.float64), val.cpu().numpy().astype(np.float64),
                wdl.cpu().numpy())


class EnsembleEvaluator:
    """Average of several evaluators: priors, values and WDL are the mean over the nets
    (each net uses its own playing head and mirror setting)."""

    def __init__(self, evaluators):
        self.evs = list(evaluators)

    def evaluate(self, curs, masks):
        import numpy as np
        outs = [e.evaluate(curs, masks) for e in self.evs]
        pri = np.mean([o[0] for o in outs], 0)
        pri = pri / pri.sum(1, keepdims=True)
        return pri, np.mean([o[1] for o in outs], 0), np.mean([o[2] for o in outs], 0)


def nn_backend():
    """How .npz files are run: 'native' (C, engine/cminimax/net.c), 'torch' or 'numpy'.

    C is the standard: with no C4_NN_BACKEND set, the native C net is used whenever the
    library is built (python -m engine.cbuild), else numpy.  All three give the same moves
    (python -m nn.native --bench).  C4_NN_BACKEND=numpy|torch|native forces one
    (nn.evaluate --backend torch sets it, so worker processes follow)."""
    b = os.environ.get("C4_NN_BACKEND", "").strip().lower()
    if b in ("numpy", "torch", "native"):
        return b
    try:
        from .native import available
        return "native" if available() else "numpy"
    except Exception:
        return "numpy"


def load_evaluator(path, device="auto", backend=None):
    """Model file -> object with .evaluate().  .npz -> numpy (no torch needed), or torch
    (nn/torch_net.py: same moves, 2-5x faster on CPU, much faster on a GPU) when
    backend='torch' / C4_NN_BACKEND=torch; .pt -> torch on the chosen device.
    'a.npz+b.npz+c.npz' -> EnsembleEvaluator."""
    if "+" in path:
        return EnsembleEvaluator(load_evaluator(p, device, backend) for p in path.split("+"))
    if path.endswith(".npz"):
        if (backend or nn_backend()) == "native":
            from .native import NativeNet, available
            if available():
                return NativeNet(path)
            log("WARNING: native net not built (python -m engine.cbuild) - using numpy")
        if (backend or nn_backend()) == "torch":
            from .torch_net import TorchNpzEvaluator
            threads = os.environ.get("C4_NN_THREADS")
            return TorchNpzEvaluator(path, device, threads=int(threads) if threads else None)
        from .numpy_net import NumpyNet
        return NumpyNet(path)
    from .model import load_checkpoint
    dev = pick_device(device)
    model, _ = load_checkpoint(path, dev)
    return TorchEvaluator(model, dev)
