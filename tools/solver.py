"""Python wrapper around Pascal Pons' Connect 4 solver (tools/pons/c4solver).

Move strings use 1-indexed columns ("4453..."), as the solver expects; returned
score lists / column sets are 0-indexed (index 0 = leftmost column).

Score convention (for the side to move): 0 = draw, >0 = win, <0 = loss.
|score| = 22 - (number of stones the winner has placed by the end of the game),
i.e. an immediate win with n stones already on the board scores (43 - n) // 2;
a loss is the negative.  Bigger positive = faster win; more negative = faster loss.
Full columns -> None.  Python 3.10 compatible.
"""
import os
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIN = os.path.join(HERE, "pons", "c4solver")
DEFAULT_BOOK = os.path.join(HERE, "pons", "7x6.book")
INVALID = -1000
WIDTH, HEIGHT = 7, 6


class SolverError(RuntimeError):
    pass


def validate(moves):
    """Raise SolverError if the 1-indexed move string is illegal or the game is already won/over.
    (The solver prints nothing on stdout for invalid input, so we must pre-check or we would block.)"""
    heights = [0] * WIDTH
    grid = {}
    for ply, ch in enumerate(moves):
        if ch not in "1234567":
            raise SolverError("bad char %r in %r" % (ch, moves))
        c = int(ch) - 1
        if heights[c] >= HEIGHT:
            raise SolverError("column full at ply %d in %r" % (ply + 1, moves))
        r = heights[c]
        heights[c] += 1
        p = ply % 2
        grid[(c, r)] = p
        for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
            n = 1
            for sgn in (1, -1):
                cc, rr = c + sgn * dc, r + sgn * dr
                while grid.get((cc, rr)) == p:
                    n += 1
                    cc += sgn * dc
                    rr += sgn * dr
            if n >= 4:
                raise SolverError("game already won at ply %d in %r" % (ply + 1, moves))
    if len(moves) >= WIDTH * HEIGHT:
        raise SolverError("board full")


class Solver:
    def __init__(self, binary=DEFAULT_BIN, book=None, timeout=None, analyze=True, weak=False):
        """analyze=True: solve_all gives 7 column scores (-a).  analyze=False: use solve() for a single
        position score.  weak=True (-w): only the sign of the score is meaningful (win/draw/loss), much faster."""
        self.analyze = analyze
        args = [binary] + (["-a"] if analyze else []) + (["-w"] if weak else [])
        if book is None and os.path.exists(DEFAULT_BOOK):
            book = DEFAULT_BOOK
        if book:
            args += ["-b", book]
        self.timeout = timeout
        self._args = args
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        self.proc = subprocess.Popen(self._args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def solve(self, moves):
        """Score of the position for the side to move (needs analyze=False)."""
        assert not self.analyze, "construct Solver(analyze=False) to use solve()"
        return self.solve_all(moves)

    def solve_all(self, moves):
        """moves: str of 1-indexed columns (e.g. "4453"). Returns list of 7 scores (None = full col).
        Raises SolverError for an invalid/already-won sequence or on timeout (solver restarted)."""
        moves = "".join(str(m) for m in moves)
        validate(moves)
        timer = None
        with self._lock:
            if self.timeout:
                timer = threading.Timer(self.timeout, self.proc.kill)
                timer.start()
            try:
                self.proc.stdin.write(moves + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
            except (BrokenPipeError, ValueError):
                line = ""
            finally:
                if timer:
                    timer.cancel()
            if not line:
                self._start()
                raise SolverError("solver died/timed out on %r" % moves)
            parts = line.split()
            if not self.analyze:
                if len(parts) != 2 or parts[0] != moves:
                    raise SolverError("bad solver reply %r" % line)
                return int(parts[1])
            if moves == "" and len(parts) == 7:      # empty board: the solver echoes no move string
                parts = [""] + parts
            if len(parts) != 8 or parts[0] != moves:
                raise SolverError("invalid position %r (solver said %r)" % (moves, line.strip()))
            return [None if int(x) == INVALID else int(x) for x in parts[1:]]


def optimal_columns(moves, solver=None):
    """0-indexed set of columns achieving the max score."""
    s = (solver or _default()).solve_all(moves)
    best = max(x for x in s if x is not None)
    return {i for i, x in enumerate(s) if x == best}


_DEFAULT = None


def _default():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Solver()
    return _DEFAULT


def solve_all(moves):
    return _default().solve_all(moves)


if __name__ == "__main__":
    import time
    t0 = time.time()
    with Solver() as s:
        # win in 1: x has 1,2,3 on the bottom row, x to move at ply 6 -> col 4 (index 3) wins now
        sc = s.solve_all("152536")
        assert sc[3] == (43 - 6) // 2 and sc[3] == max(sc), sc
        assert optimal_columns("152536", s) == {3}
        # full column -> None
        sc = s.solve_all("111111" "2222")
        assert sc[0] is None, sc
        # known 8-ply position
        assert s.solve_all("43214321") == [0, 4, 4, 4, -1, -1, -1]
        assert optimal_columns("43214321", s) == {1, 2, 3}
        try:
            s.solve_all("1111111")
            raise AssertionError("expected error")
        except SolverError:
            pass
        # solver still alive after error
        assert s.solve_all("43214321")[1] == 4
        print("self-test OK (%.2fs)" % (time.time() - t0))
