"""Build the C minimax (engine/cminimax/minimax.c) into a shared library next to it.

    python -m engine.cbuild            # once per machine (Linux: gcc/clang; Windows: gcc from
                                       # MinGW/MSYS2, clang, or MSVC 'cl' in a Developer prompt)
    python -m engine.cbuild --check    # build, then compare with the Python search and time both

After building, engine.minimax.search() uses the C version automatically whenever it can
give the identical answer (hand-crafted heuristic, no custom evaluator).  Turn it off with
C4_MINIMAX_C=0.  The library is git-ignored (it is machine-specific).
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cminimax")
SRC = os.path.join(HERE, "minimax.c")


def lib_path():
    if sys.platform.startswith("win"):
        return os.path.join(HERE, "c4minimax.dll")
    if sys.platform == "darwin":
        return os.path.join(HERE, "libc4minimax.dylib")
    return os.path.join(HERE, "libc4minimax.so")


def build(verbose=True):
    out = lib_path()
    tmp = out + ".tmp"
    tries = []
    if sys.platform.startswith("win"):
        for cc in ("gcc", "clang"):
            if shutil.which(cc):
                tries.append([cc, "-O3", "-shared", "-o", tmp, SRC])
        if shutil.which("cl"):
            tries.append(["cl", "/nologo", "/O2", "/LD", SRC, "/Fe" + tmp, "/Fo" + os.path.join(HERE, "minimax.obj")])
    else:
        for cc in (os.environ.get("CC"), "cc", "gcc", "clang"):
            if cc and shutil.which(cc):
                tries.append([cc, "-O3", "-march=native", "-fPIC", "-shared", "-o", tmp, SRC])
                tries.append([cc, "-O3", "-fPIC", "-shared", "-o", tmp, SRC])
    if not tries:
        raise SystemExit("no C compiler found (Linux: sudo apt install build-essential; "
                         "Windows: install MSYS2/MinGW gcc or use an MSVC Developer prompt)")
    last = None
    for cmd in tries:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        if r.returncode == 0 and os.path.isfile(tmp):
            os.replace(tmp, out)
            if verbose:
                print("built %s with: %s" % (out, " ".join(cmd[:4])))
            return out
        last = (cmd, r.stdout + r.stderr)
    raise SystemExit("C build failed:\n%s\n%s" % (" ".join(last[0]), last[1]))


def check(n=300, depths=(2, 4, 6, 8)):
    import random
    import time
    os.environ["C4_MINIMAX_C"] = "0"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import minimax
    from engine.board import Board
    from engine import cminimax_ffi as C
    if C._load() is None:
        raise SystemExit("library did not load: %s" % C.load_error())
    rng = random.Random(1)
    boards = []
    while len(boards) < n:
        b = Board()
        for _ in range(rng.randint(0, 30)):
            if b.is_terminal():
                break
            b.drop(rng.choice(b.legal_moves()))
        if not b.is_terminal():
            boards.append(b)
    for d in depths:
        t = time.perf_counter()
        py = [minimax.search(b, d) for b in boards]
        tp = time.perf_counter() - t
        t = time.perf_counter()
        cc = [C.search(b, d) for b in boards]
        tc = time.perf_counter() - t
        same = sum(p.col == c.col and p.score == c.score and p.nodes == c.nodes for p, c in zip(py, cc))
        print("depth %d: identical %d/%d   python %.2fs   C %.3fs   %.0fx faster" % (d, same, n, tp, tc, tp / max(tc, 1e-9)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    build()
    if a.check:
        check()


if __name__ == "__main__":
    main()
