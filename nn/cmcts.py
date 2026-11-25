"""PUCT tree in C (engine/cminimax/mcts.c), network in Python.

nn.mcts.search() uses this automatically when the C library is built (python -m engine.cbuild)
and the roots are fresh (not yet expanded: every play / benchmark search; self-play roots
with Dirichlet noise keep the Python tree).  C4_MCTS_TREE_C=0 forces the Python tree.

Per simulation step the C side walks every tree to its next unevaluated leaf, Python
evaluates that batch of leaves with one network call, and C backs the values up.  Same
arithmetic and tie-breaks as nn/mcts.py, so visit counts and moves are identical
(tests/test_cmcts.py).  After the search the root statistics are copied back into the
Python Node objects, so choose_move / visit_policy / q() / principal_variation work as before.

    python -m nn.cmcts --bench                 # Python tree vs C tree, same net, same positions
    python -m nn.cmcts --bench --run score_8x128_s0_r1 --backend torch --positions 512 --sims 200
"""

import ctypes
import os
import time

import numpy as np

_LIB = None
_ERR = None


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
        if not hasattr(lib, "c4m_create"):
            _ERR = "library built before the MCTS tree existed: rerun python -m engine.cbuild"
            return None
        vp, u64p, dp, ip = ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double), \
            ctypes.POINTER(ctypes.c_int)
        lib.c4m_create.argtypes = [ctypes.c_int, u64p, u64p, ctypes.c_double, ctypes.c_double]
        lib.c4m_create.restype = vp
        lib.c4m_free.argtypes = [vp]
        lib.c4m_collect.argtypes = [vp, u64p, u64p]
        lib.c4m_collect.restype = ctypes.c_int
        lib.c4m_apply.argtypes = [vp, dp, dp, ctypes.POINTER(ctypes.c_uint8)]
        lib.c4m_root_stats.argtypes = [vp, ctypes.c_int, ip, dp, dp, ip, dp]
        lib.c4m_pv.argtypes = [vp, ctypes.c_int, ip, ctypes.c_int]
        lib.c4m_pv.restype = ctypes.c_int
        lib.c4m_max_depth.argtypes = [vp, ctypes.c_int]
        lib.c4m_max_depth.restype = ctypes.c_int
        lib.c4m_node_count.argtypes = [vp]
        lib.c4m_node_count.restype = ctypes.c_int
        _LIB = lib
    except OSError as e:
        _ERR = str(e)
    return _LIB


def available():
    return os.environ.get("C4_MCTS_TREE_C", "1") != "0" and _load() is not None


def load_error():
    _load()
    return _ERR


def _ptr(a, t):
    return a.ctypes.data_as(ctypes.POINTER(t))


def search_c(roots, evaluator, sims, c_puct=1.5, fpu=0.0):
    """Same contract as nn.mcts.search for fresh roots: runs ``sims`` simulations on every
    root and fills the roots' N / W / P / n / value / legal (+ C-side PV and depth)."""
    lib = _load()
    n = len(roots)
    cur = np.array([r.cur for r in roots], dtype=np.uint64)
    mask = np.array([r.mask for r in roots], dtype=np.uint64)
    h = lib.c4m_create(n, _ptr(cur, ctypes.c_uint64), _ptr(mask, ctypes.c_uint64), float(c_puct), float(fpu))
    if not h:
        raise MemoryError("c4m_create failed")
    out_c = np.zeros(max(n, 1), np.uint64)
    out_m = np.zeros(max(n, 1), np.uint64)
    evals = 0
    try:
        for step in range(sims + 1):                    # step 0 = expand the roots
            k = lib.c4m_collect(h, _ptr(out_c, ctypes.c_uint64), _ptr(out_m, ctypes.c_uint64))
            if k < 0:
                raise MemoryError("MCTS tree out of memory")
            if k == 0:
                lib.c4m_apply(h, None, None, None)
                continue
            curs, masks = out_c[:k].tolist(), out_m[:k].tolist()
            pri, val, _ = evaluator.evaluate(curs, masks)
            if step:
                evals += k
            pri = np.ascontiguousarray(pri, dtype=np.float64).reshape(k, 7)
            val = np.ascontiguousarray(val, dtype=np.float64).reshape(k)
            ex = getattr(evaluator, "last_exact", None) if step else None
            exa = np.ascontiguousarray(ex, dtype=np.uint8) if ex is not None else None
            lib.c4m_apply(h, _ptr(pri, ctypes.c_double), _ptr(val, ctypes.c_double),
                          _ptr(exa, ctypes.c_uint8) if exa is not None else None)
        N = (ctypes.c_int * 7)()
        W = (ctypes.c_double * 7)()
        P = (ctypes.c_double * 7)()
        nn_, v = ctypes.c_int(), ctypes.c_double()
        pv = (ctypes.c_int * 42)()
        from .mcts import legal_columns
        for i, r in enumerate(roots):
            lib.c4m_root_stats(h, i, N, W, P, ctypes.byref(nn_), ctypes.byref(v))
            r.N, r.W, r.P = list(N), list(W), list(P)
            r.P_raw = list(r.P)
            r.n, r.value = nn_.value, v.value
            r.legal = legal_columns(r.mask)
            r.expanded = True
            k = lib.c4m_pv(h, i, pv, 42)
            r.c_pv = list(pv[:k])
            r.c_max_depth = lib.c4m_max_depth(h, i)
    finally:
        lib.c4m_free(h)
    return evals


def bench(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--run", default="sup_6x64")
    ap.add_argument("--model", default="best.npz")
    ap.add_argument("--backend", default="numpy", choices=("numpy", "torch"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--positions", type=int, default=256)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args(argv)
    from . import bench as B
    from . import mcts
    from .agents import CachedEvaluator
    from .common import RUNS, load_evaluator
    path = a.model if os.path.isfile(a.model) else os.path.join(RUNS, a.run, a.model)
    net = load_evaluator(path, a.device, backend=a.backend)
    rows = B.stratified(B.load_accuracy_bench(), a.positions, seed=3)

    class Timed:
        """Evaluator wrapper that measures time inside the network (so tree time = total - net)."""

        def __init__(self, ev):
            self.ev, self.sec, self.meta = ev, 0.0, getattr(ev, "meta", {})

        def evaluate(self, c, m):
            t = time.perf_counter()
            out = self.ev.evaluate(c, m)
            self.sec += time.perf_counter() - t
            return out

    def run(use_c, ev):
        os.environ["C4_MCTS_TREE_C"] = "1" if use_c else "0"
        cols, t0 = [], time.perf_counter()
        for i in range(0, len(rows), a.batch):
            roots = [mcts.Node(r["cur"], r["mask"]) for r in rows[i:i + a.batch]]
            mcts.search(roots, ev, a.sims)
            cols += [mcts.choose_move(rt) for rt in roots]
        return cols, time.perf_counter() - t0

    print("%d positions x %d sims, %s (%s), batch %d" % (len(rows), a.sims, a.run, a.backend, a.batch))
    res = {}
    for label, use_c in (("python tree", False), ("C tree", True)):
        tev = Timed(CachedEvaluator(net))     # same cache on both sides: net time is comparable
        cols, dt = run(use_c, tev)
        res[label] = (cols, dt, tev.sec)
        acc = B.accuracy_summary(rows, cols)["optimal"]
        print("  %-12s total %7.2fs   net %7.2fs   tree %7.2fs   (%.3f ms/sim tree)   optimal %.4f" % (
            label, dt, tev.sec, dt - tev.sec, 1000 * (dt - tev.sec) / (len(rows) * a.sims), acc))
    same = sum(x == y for x, y in zip(res["python tree"][0], res["C tree"][0]))
    tp, tc = (res[k][1] - res[k][2] for k in ("python tree", "C tree"))
    print("  identical moves %d/%d;  tree %.0fx faster;  whole search %.2fx faster" % (
        same, len(rows), tp / max(tc, 1e-9), res["python tree"][1] / max(res["C tree"][1], 1e-9)))
    os.environ.pop("C4_MCTS_TREE_C", None)


if __name__ == "__main__":
    bench()
