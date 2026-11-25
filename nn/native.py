"""Native (C) network + fully native MCTS: no Python between simulations.

engine/cminimax/net.c runs an exported model.npz with the same maths as nn/numpy_net.py
(PCA projection, score head, mirror averaging, play head, value source), and runs whole MCTS
searches with the C tree of mcts.c, evaluating the leaves itself through a position cache.

    ev = NativeNet("runs/score_8x128_s0_r1/best.npz")   # .evaluate() like NumpyNet
    load_evaluator(path, backend="native")               # same thing; or C4_NN_BACKEND=native
    MCTSAgent(ev, sims=200)                              # uses the fully native search automatically

Build: python -m engine.cbuild.   Benchmark: python -m nn.native --bench
"""

import ctypes
import json
import os
import time

import numpy as np

_LIB = None
_ERR = None
PLAY_HEADS = {"policy": 0, "score": 1, "combo": 2}


def _load():
    global _LIB, _ERR
    if _LIB is not None or _ERR is not None:
        return _LIB
    try:
        from engine.cbuild import lib_path
        p = lib_path()
        if not os.path.isfile(p):
            _ERR = "not built (python -m engine.cbuild)"
            return None
        lib = ctypes.CDLL(p)
        if not hasattr(lib, "c4n_create"):
            _ERR = "library built before the native net existed: rerun python -m engine.cbuild"
            return None
        vp, u64p, dp, ip, fp = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float))
        lib.c4n_create.argtypes = [ip, fp, fp, ctypes.c_longlong]
        lib.c4n_create.restype = vp
        lib.c4n_free.argtypes = [vp]
        lib.c4n_evaluate.argtypes = [vp, ctypes.c_int, u64p, u64p, dp, dp, fp]
        lib.c4n_evaluate.restype = ctypes.c_int
        lib.c4n_cache_create.argtypes = [ctypes.c_int]
        lib.c4n_cache_create.restype = vp
        lib.c4n_cache_free.argtypes = [vp]
        lib.c4n_cache_stats.argtypes = [vp, ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong)]
        lib.c4n_search.argtypes = [vp, vp, ctypes.c_int, u64p, u64p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                   ip, dp, dp, dp, ip, ip, ip]
        lib.c4n_search.restype = ctypes.c_int
        _LIB = lib
    except OSError as e:
        _ERR = str(e)
    return _LIB


def available():
    return _load() is not None


def load_error():
    _load()
    return _ERR


def _p(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


def _pack(w, cfg, has_proj, has_score):
    order = []
    if has_proj:
        order += ["inproj.mean", "inproj.U"]
    order += ["stem.w", "stem.b"]
    for i in range(int(cfg["blocks"])):
        order += ["b%d.c1.w" % i, "b%d.c1.b" % i, "b%d.c2.w" % i, "b%d.c2.b" % i]
    order += ["pol.w", "pol.b", "pol_fc.w", "pol_fc.b", "val.w", "val.b", "val_fc1.w", "val_fc1.b",
              "val_fc2.w", "val_fc2.b"]
    if has_score:
        order += ["sc.w", "sc.b", "sc_fc1.w", "sc_fc1.b", "sc_fc2.w", "sc_fc2.b"]
    return np.ascontiguousarray(np.concatenate([np.ascontiguousarray(w[k], np.float32).ravel() for k in order]))


class NativeNet:
    """Drop-in for NumpyNet (same .evaluate / .meta / .play_head ...), computed in C."""

    def __init__(self, path):
        lib = _load()
        if lib is None:
            raise RuntimeError("native net unavailable: %s" % _ERR)
        from .numpy_net import value_source
        with np.load(path, allow_pickle=False) as z:
            w = {k: z[k] for k in z.files if not k.endswith("_json")}
            self.config = json.loads(str(z["config_json"]))
            self.meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        self.path = path
        self.has_score = "sc.w" in w
        has_proj = "inproj.U" in w
        self.play_head = self.meta.get("play_head", "policy") if self.has_score else "policy"
        self.play_temp = float(self.meta.get("play_temp", 0.5))
        self.play_policy_weight = float(self.meta.get("play_policy_weight", 0.0))
        self.value_source = value_source()
        C = int(w["stem.w"].shape[0])
        cfg = np.array([int(self.config["blocks"]), C, w["pol.w"].shape[0], w["val.w"].shape[0], w["val_fc1.w"].shape[0],
                        int(self.has_score), w["sc.w"].shape[0] if self.has_score else 0,
                        w["sc_fc1.w"].shape[0] if self.has_score else 0,
                        w["inproj.U"].shape[1] if has_proj else 0, PLAY_HEADS[self.play_head],
                        int(bool(self.meta.get("play_mirror"))), int(self.value_source == "score" and self.has_score)],
                       dtype=np.int32)
        play = np.array([self.play_temp, self.play_policy_weight], np.float32)
        buf = _pack(w, self.config, has_proj, self.has_score)
        self._lib = lib
        self._h = lib.c4n_create(_p(cfg, ctypes.c_int), _p(play, ctypes.c_float), _p(buf, ctypes.c_float), buf.size)
        if not self._h:
            raise RuntimeError("c4n_create failed (unexpected weight layout in %s)" % path)

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.c4n_free(h)
            self._h = None

    def evaluate(self, curs, masks):
        cur = np.ascontiguousarray(np.asarray(curs, dtype=np.uint64))
        mask = np.ascontiguousarray(np.asarray(masks, dtype=np.uint64))
        n = len(cur)
        pri, val, wdl = np.empty((n, 7)), np.empty(n), np.empty((n, 3), np.float32)
        if n and self._lib.c4n_evaluate(self._h, n, _p(cur, ctypes.c_uint64), _p(mask, ctypes.c_uint64),
                                        _p(pri, ctypes.c_double), _p(val, ctypes.c_double), _p(wdl, ctypes.c_float)):
            raise MemoryError("c4n_evaluate failed")
        return pri, val, wdl

    def new_cache(self, log2cap=18):
        return NativeCache(self._lib, log2cap)

    def search(self, roots, sims, c_puct=1.5, fpu=0.0, cache=None):
        """Fully native search on fresh mcts.Node roots; fills them like nn/cmcts.search_c."""
        from .mcts import legal_columns
        n = len(roots)
        cur = np.array([r.cur for r in roots], np.uint64)
        mask = np.array([r.mask for r in roots], np.uint64)
        N, W, P = np.zeros((n, 7), np.int32), np.zeros((n, 7)), np.zeros((n, 7))
        val, vis = np.zeros(n), np.zeros(n, np.int32)
        pv, md = np.zeros((n, 42), np.int32), np.zeros(n, np.int32)
        rc = self._lib.c4n_search(self._h, cache._h if cache else None, n, _p(cur, ctypes.c_uint64),
                                  _p(mask, ctypes.c_uint64), int(sims), float(c_puct), float(fpu),
                                  _p(N, ctypes.c_int), _p(W, ctypes.c_double), _p(P, ctypes.c_double),
                                  _p(val, ctypes.c_double), _p(vis, ctypes.c_int), _p(pv, ctypes.c_int),
                                  _p(md, ctypes.c_int))
        if rc:
            raise MemoryError("c4n_search failed")
        for i, r in enumerate(roots):
            r.N, r.W, r.P = N[i].tolist(), W[i].tolist(), P[i].tolist()
            r.P_raw = list(r.P)
            r.n, r.value = int(vis[i]), float(val[i])
            r.legal = legal_columns(r.mask)
            r.expanded = True
            r.c_pv = [int(x) for x in pv[i] if x >= 0]
            r.c_max_depth = int(md[i])
        return sims * n


class NativeCache:
    def __init__(self, lib, log2cap=18):
        self._lib = lib
        self._h = lib.c4n_cache_create(int(log2cap))

    def stats(self):
        a, c = ctypes.c_longlong(), ctypes.c_longlong()
        self._lib.c4n_cache_stats(self._h, ctypes.byref(a), ctypes.byref(c))
        return a.value, c.value

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            self._lib.c4n_cache_free(h)
            self._h = None


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--runs", default="sup_6x64,score_8x128_s0_r1")
    ap.add_argument("--model", default="best.npz")
    ap.add_argument("--games", type=int, default=2, help="self-play games per net for the agent timing")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--positions", type=int, default=3000, help="benchmark positions for the policy check")
    a = ap.parse_args()
    from engine.board import Board
    from . import bench as B
    from .agents import MCTSAgent
    from .common import RUNS, load_evaluator
    rows = B.stratified(B.load_accuracy_bench(), a.positions, seed=1)
    for run in a.runs.split(","):
        path = os.path.join(RUNS, run, a.model)
        print("== %s" % run)
        nets = {"numpy": load_evaluator(path, "cpu", backend="numpy"), "native": NativeNet(path)}
        try:
            nets["torch"] = load_evaluator(path, "cpu", backend="torch")
        except Exception:  # noqa: BLE001
            pass
        x1 = [rows[0]["cur"]], [rows[0]["mask"]]
        ref = None
        for name, ev in nets.items():
            ev.evaluate(*x1)
            t = time.perf_counter()
            for _ in range(200):
                ev.evaluate(*x1)
            t1 = (time.perf_counter() - t) / 200 * 1000
            t = time.perf_counter()
            pri, val, _ = ev.evaluate([r["cur"] for r in rows], [r["mask"] for r in rows])
            tb = time.perf_counter() - t
            cols = [max([c for c in range(7) if r["scores"][c] is not None], key=lambda c: p[c]) for r, p in zip(rows, pri)]
            acc = B.accuracy_summary(rows, cols)["optimal"]
            if ref is None:
                ref = (pri, val, cols)
            same = sum(x == y for x, y in zip(cols, ref[2]))
            print("  %-7s 1 position %6.2f ms   %d positions %6.2f s   optimal %.4f   same move as numpy %d/%d"
                  "   max |prior diff| %.1e" % (name, t1, len(rows), tb, acc, same, len(rows), np.abs(pri - ref[0]).max()))
        for name in [k for k in ("numpy", "torch", "native") if k in nets]:
            t, moves = time.perf_counter(), []
            for g in range(a.games):
                ag, b, mv = MCTSAgent(nets[name], sims=a.sims), Board.from_moves(str(g + 3)), []
                while not b.is_terminal():
                    c = ag.choose(b)
                    mv.append(c)
                    b.drop(c)
                moves.append(mv)
            dt = time.perf_counter() - t
            n_moves = sum(len(m) for m in moves)
            print("  MCTSAgent %d sims, %-6s: %d games %6.2f s  (%6.1f ms/move)  moves %s" % (
                a.sims, name, a.games, dt, 1000 * dt / n_moves, "|".join("".join(map(str, m)) for m in moves)))


if __name__ == "__main__":
    main()
