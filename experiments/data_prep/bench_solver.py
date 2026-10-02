"""Benchmark solver time on random non-terminal positions of a given ply."""
import os, random, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import solver as S

def rand_pos(ply, rng):
    while True:
        m = ""
        ok = True
        for _ in range(ply):
            legal = [c for c in "1234567" if m.count(c) < 6]
            m += rng.choice(legal)
            try:
                S.validate(m)
            except S.SolverError:
                ok = False
                break
        if ok:
            return m

if __name__ == "__main__":
    rng = random.Random(1)
    with S.Solver(timeout=60) as s:
        for ply in (8, 12, 16, 20):
            ts = []
            for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 10):
                m = rand_pos(ply, rng)
                t = time.time()
                try:
                    s.solve_all(m)
                except S.SolverError:
                    pass
                ts.append(time.time() - t)
            print("ply %2d: n=%d mean=%.3fs median=%.3fs max=%.3fs" % (ply, len(ts), statistics.mean(ts), statistics.median(ts), max(ts)), flush=True)
