"""ctypes binding of the C minimax (engine/cminimax/minimax.c; build: python -m engine.cbuild).

engine.minimax.search() calls search() here automatically when the library is built and the
search uses the hand-crafted heuristic (no custom evaluator); C4_MINIMAX_C=0 turns it off.
Results (column, score, node count) are identical to the Python search.
"""

import ctypes
import os
import time

_LIB = None
_ERR = None


def _load():
    global _LIB, _ERR
    if _LIB is not None or _ERR is not None:
        return _LIB
    try:
        from .cbuild import lib_path
        p = lib_path()
        if not os.path.isfile(p):
            _ERR = "not built (python -m engine.cbuild)"
            return None
        lib = ctypes.CDLL(p)
        u64, i64p, ip = ctypes.c_uint64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int)
        lib.c4_search.argtypes = [u64, u64, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                  ctypes.c_int, i64p, i64p, ip, ip]
        lib.c4_search.restype = ctypes.c_int
        lib.c4_evaluate.argtypes = [u64, u64]
        lib.c4_evaluate.restype = ctypes.c_int64
        _LIB = lib
    except OSError as e:
        _ERR = str(e)
    return _LIB


def available():
    if os.environ.get("C4_MINIMAX_C", "1") == "0":
        return False
    return _load() is not None


def load_error():
    _load()
    return _ERR


def evaluate_bits(me, opp):
    return int(_load().c4_evaluate(me, opp))


def search(board, depth, time_limit=None, use_tt=True, ordering=True):
    from .minimax import SearchResult
    lib = _load()
    if board.is_terminal():
        raise ValueError("cannot search a finished game")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    score, nodes = ctypes.c_int64(), ctypes.c_int64()
    reached, timed = ctypes.c_int(), ctypes.c_int()
    t0 = time.perf_counter()
    col = lib.c4_search(board.current_bits(), board.mask, board.moves_played, int(depth),
                        float(time_limit or 0.0), int(bool(use_tt)), int(bool(ordering)),
                        ctypes.byref(score), ctypes.byref(nodes), ctypes.byref(reached), ctypes.byref(timed))
    if col < 0:
        raise RuntimeError("C minimax failed (code %d)" % col)
    return SearchResult(col=col, score=float(score.value), nodes=int(nodes.value),
                        ms=(time.perf_counter() - t0) * 1000.0, depth_reached=int(reached.value),
                        timed_out=bool(timed.value))
