"""One process owns the GPU; every CPU core runs game/search workers that send it positions.

Why
  Tree search (MCTS) and game bookkeeping are plain Python, so they need CPU
  cores.  The network wants big batches on the GPU.  The old self-play gave
  every worker its own copy of the model on the GPU: each CUDA context costs
  about 0.5 GB, so 8 workers + the trainer peaked at ~5 GB of the RTX 3050's
  8 GB and 16 workers would not fit.  With this server:
    * every CPU thread runs a worker (15 workers + this server on 16 threads),
    * there is one CUDA context, and
    * each GPU call evaluates the pending positions of ALL workers together
      (self-play: up to workers x --parallel, e.g. 15 x 128 = 1,920 positions).

Use
    srv = InferenceServer(n_clients=15, device="cuda:0").start()
    srv.load("selfplay", "runs/x/checkpoints/latest.pt", amp=True)   # or a .npz
    pool = hw.spawn_pool(15, init_worker, (srv.handle(),))
    ...  # inside a worker:  ev = remote_evaluator("selfplay");  ev.evaluate(curs, masks)
    stats = srv.stop()

remote_evaluator() returns an object with the same evaluate(curs, masks) ->
(priors, values, wdl) as NumpyNet / TorchEvaluator, so MCTS, the agents and
the benchmarks work unchanged.
"""

import os
import queue
import time

import numpy as np

_CLIENT = {}          # inside a worker process: {"req", "resp", "cid"}
_RESP_TIMEOUT = float(os.environ.get("C4_GPU_SERVER_TIMEOUT", "600"))


# ================================================================ model on GPU
def load_torch_model(path, device):
    """.pt checkpoint or exported .npz -> eval-mode torch module on ``device``."""
    if path.endswith(".npz"):
        return FoldedNet.from_npz(path).to(device).eval()
    from .model import load_checkpoint
    model, _ = load_checkpoint(path, device)
    return model


def _folded_net_class():
    import torch
    import torch.nn.functional as F

    class _FoldedNet(torch.nn.Module):
        """The exported model.npz (BatchNorm already folded into the convs) as a
        torch module, i.e. numpy_net.NumpyNet on the GPU.  Same maths, same
        weights, so it plays the same moves as the website's numpy version
        (up to float rounding)."""

        def __init__(self, arrays, config):
            super().__init__()
            self.config = config
            self.blocks = int(config["blocks"])
            for k, v in arrays.items():
                self.register_buffer(k.replace(".", "_"), torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)))

        @classmethod
        def from_npz(cls, path):
            import json
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files if not k.endswith("_json")}
                config = json.loads(str(z["config_json"]))
            return cls(arrays, config)

        def w(self, name):
            return getattr(self, name.replace(".", "_"))

        def forward(self, x):
            w = self.w
            h = F.relu(F.conv2d(x, w("stem.w"), w("stem.b"), padding=1))
            for i in range(self.blocks):
                y = F.relu(F.conv2d(h, w("b%d.c1.w" % i), w("b%d.c1.b" % i), padding=1))
                y = F.conv2d(y, w("b%d.c2.w" % i), w("b%d.c2.b" % i), padding=1)
                h = F.relu(h + y)
            p = F.relu(F.conv2d(h, w("pol.w")[:, :, None, None], w("pol.b"))).flatten(1)
            p = F.linear(p, w("pol_fc.w"), w("pol_fc.b"))
            v = F.relu(F.conv2d(h, w("val.w")[:, :, None, None], w("val.b"))).flatten(1)
            v = F.relu(F.linear(v, w("val_fc1.w"), w("val_fc1.b")))
            v = F.linear(v, w("val_fc2.w"), w("val_fc2.b"))
            return p, v
    return _FoldedNet


class FoldedNet:
    """Lazy accessor so importing this module does not import torch."""

    @staticmethod
    def from_npz(path):
        return _folded_net_class().from_npz(path)


# ===================================================================== server
def _serve(req_q, resp_qs, ctrl_q, device, max_wait):
    import torch
    from .common import TorchEvaluator
    from .hw import single_thread_torch
    single_thread_torch()
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    models = {}                       # key -> TorchEvaluator
    n = len(resp_qs)
    last_seen = {}                    # client id -> time of its last request
    stats = {"calls": 0, "positions": 0, "requests": 0}
    deferred = []

    def control(msg):
        kind = msg[0]
        if kind == "load":
            _, key, path, amp = msg
            try:
                models[key] = TorchEvaluator(load_torch_model(path, dev), dev, amp=amp)
                ctrl_q.put(("ok", key))
            except Exception as e:  # noqa: BLE001
                ctrl_q.put(("error", "%s: %s" % (type(e).__name__, e)))
        elif kind == "stats":
            peak = round(torch.cuda.max_memory_reserved(dev) / 2 ** 20) if dev.type == "cuda" else None
            ctrl_q.put(("stats", dict(stats, peak_mib=peak,
                                      avg_batch=stats["positions"] / max(stats["calls"], 1))))
        return kind != "stop"

    running = True
    while running:
        msg = deferred.pop(0) if deferred else req_q.get()
        if msg[0] != "eval":
            running = control(msg)
            continue
        batch = [msg]
        now = time.perf_counter()
        last_seen[msg[1]] = now
        # wait (briefly) until every recently active worker has a request in
        active = sum(1 for t in last_seen.values() if now - t < 0.25) or 1
        waiting = {msg[1]}
        deadline = now + max_wait
        while len(waiting) < min(active, n):
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            try:
                m = req_q.get(timeout=left)
            except queue.Empty:
                break
            if m[0] != "eval":
                deferred.append(m)
                break
            batch.append(m)
            waiting.add(m[1])
            last_seen[m[1]] = time.perf_counter()
        # one forward pass per model over everything that arrived
        by_key = {}
        for m in batch:
            by_key.setdefault(m[2], []).append(m)
        for key, ms in by_key.items():
            ev = models.get(key)
            if ev is None:
                for m in ms:
                    resp_qs[m[1]].put(("err", "model %r not loaded on the GPU server" % key))
                continue
            try:
                curs = np.concatenate([m[3] for m in ms])
                masks = np.concatenate([m[4] for m in ms])
                pri, val, wdl = ev.evaluate(curs, masks)
            except Exception as e:  # noqa: BLE001
                for m in ms:
                    resp_qs[m[1]].put(("err", "%s: %s" % (type(e).__name__, e)))
                continue
            stats["calls"] += 1
            stats["positions"] += len(curs)
            stats["requests"] += len(ms)
            i = 0
            pri, val, wdl = pri.astype(np.float32), val.astype(np.float32), wdl.astype(np.float32)
            for m in ms:
                k = len(m[3])
                resp_qs[m[1]].put(("ok", pri[i:i + k], val[i:i + k], wdl[i:i + k]))
                i += k


class InferenceServer:
    """Owns the GPU process.  ``n_clients`` = number of worker processes that will
    call remote_evaluator() (each needs its own reply queue)."""

    def __init__(self, n_clients, device="cuda", max_wait_ms=2.0):
        import multiprocessing as mp
        self.ctx = mp.get_context("spawn")
        self.n = int(n_clients)
        self.device = str(device)
        self.max_wait = max_wait_ms / 1000.0
        self.req = self.ctx.Queue()
        self.resp = [self.ctx.Queue() for _ in range(self.n)]
        self.ctrl = self.ctx.Queue()
        self.counter = self.ctx.Value("i", 0)
        self.proc = None

    def start(self):
        self.proc = self.ctx.Process(target=_serve, args=(self.req, self.resp, self.ctrl, self.device, self.max_wait),
                                     daemon=True, name="gpu-server")
        self.proc.start()
        return self

    def handle(self):
        """Picklable bundle for init_worker (pass through a Pool's initargs)."""
        return (self.req, self.resp, self.counter)

    def load(self, key, path, amp=False, timeout=300):
        """(Re)load model ``key`` from a .pt or .npz file; blocks until it is on the GPU."""
        self.req.put(("load", key, os.path.abspath(path), bool(amp)))
        reply = self._ctrl(timeout)
        if reply[0] != "ok":
            raise RuntimeError("GPU server could not load %s: %s" % (path, reply[1]))

    def stats(self, timeout=30):
        self.req.put(("stats",))
        r = self._ctrl(timeout)
        return r[1] if r[0] == "stats" else {}

    def _ctrl(self, timeout):
        t_end = time.time() + timeout
        while True:
            try:
                return self.ctrl.get(timeout=1.0)
            except queue.Empty:
                if self.proc is not None and not self.proc.is_alive():
                    raise RuntimeError("GPU server process died (exit code %s)" % self.proc.exitcode)
                if time.time() > t_end:
                    raise RuntimeError("GPU server did not answer within %d s" % timeout)

    def stop(self):
        st = {}
        if self.proc is not None and self.proc.is_alive():
            try:
                st = self.stats()
            except RuntimeError:
                st = {}
            self.req.put(("stop",))
            self.proc.join(timeout=10)
            if self.proc.is_alive():
                self.proc.terminate()
        self.proc = None
        return st

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ===================================================================== worker
def init_worker(handle, then=None, then_args=()):
    """Pool initializer: connect this worker to the server (and optionally run
    another initializer afterwards)."""
    req, resp, counter = handle
    with counter.get_lock():
        cid = counter.value % len(resp)
        counter.value += 1
    _CLIENT.update(req=req, resp=resp[cid], cid=cid)
    from .hw import single_thread_torch
    single_thread_torch()
    if then is not None:
        then(*then_args)


def connected():
    return bool(_CLIENT)


class RemoteEvaluator:
    def __init__(self, key):
        if not _CLIENT:
            raise RuntimeError("this process is not connected to a GPU server (init_worker not run)")
        self.key = key
        self.req, self.resp, self.cid = _CLIENT["req"], _CLIENT["resp"], _CLIENT["cid"]

    def evaluate(self, curs, masks):
        curs = np.ascontiguousarray(curs, dtype=np.uint64)
        masks = np.ascontiguousarray(masks, dtype=np.uint64)
        self.req.put(("eval", self.cid, self.key, curs, masks))
        try:
            msg = self.resp.get(timeout=_RESP_TIMEOUT)
        except queue.Empty:
            raise RuntimeError("GPU server did not answer within %d s" % _RESP_TIMEOUT) from None
        if msg[0] != "ok":
            raise RuntimeError(msg[1])
        _, pri, val, wdl = msg
        return pri.astype(np.float64), val.astype(np.float64), wdl


def remote_evaluator(key):
    return RemoteEvaluator(key)
