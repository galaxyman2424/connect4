"""Generate the labeled accuracy set: data/accuracy_positions.csv

Positions come from random legal play (50%) and heuristic play (50%: take an immediate win,
else block an immediate loss, else centre-biased random). Ply 8-30, stratified into six ply
buckets, no winner yet, board not full, deduplicated (exact move-sequence AND board, so
transpositions are not repeated). Each is labeled with the 7 Pons column scores.
Positions where the solver needs > 5 s are skipped (so hard early positions are slightly
under-represented). Columns in the CSV: moves (1-indexed cols), ply, scores (JSON list, null=full
column), optimal_cols (JSON list, 0-indexed), source, bucket.

Usage: python make_accuracy_set.py [--n 3000] [--workers 2] [--seed 12345]
"""
import argparse, json, multiprocessing as mp, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import solver as S

BUCKETS = [(8, 11), (12, 15), (16, 19), (20, 23), (24, 27), (28, 30)]
W, H = 7, 6

def board_key(moves):
    heights = [0] * W; cols = [[] for _ in range(W)]
    for i, ch in enumerate(moves):
        c = int(ch) - 1; cols[c].append(i % 2)
    k = tuple(tuple(c) for c in cols)
    return min(k, k[::-1])  # mirror-equivalent positions count as duplicates

def wins_with(moves, c):
    try:
        S.validate(moves + str(c + 1)); return False
    except S.SolverError as e:
        return "already won" in str(e)

def legal(moves):
    return [c for c in range(W) if moves.count(str(c + 1)) < H]

def gen_one(ply, heuristic, rng):
    m = ""
    for _ in range(ply):
        lg = legal(m)
        if not lg: return None
        c = None
        if heuristic:
            # immediate win for side to move
            wins = [x for x in lg if wins_with(m, x)]
            if wins:
                c = rng.choice(wins)
            else:
                # block: play a column where the opponent would otherwise win
                threats = [x for x in lg if _opp_wins(m, x)]
                if threats:
                    c = rng.choice(threats)
                else:
                    c = rng.choices(lg, weights=[4 - abs(x - 3) for x in lg])[0]
        else:
            c = rng.choice(lg)
        m += str(c + 1)
        try:
            S.validate(m)
        except S.SolverError:
            return None   # someone won (or invalid) before reaching target ply
    return m

def _opp_wins(m, x):
    """Would the opponent (of the side to move) win by playing column x? Test by placing a stone
    for the opponent: emulate by playing x for a passed turn using a dummy-free board evaluation."""
    heights = [0] * W; grid = {}
    for i, ch in enumerate(m):
        c = int(ch) - 1; grid[(c, heights[c])] = i % 2; heights[c] += 1
    if heights[x] >= H: return False
    p = 1 - (len(m) % 2); c, r = x, heights[x]
    for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
        n = 1
        for sgn in (1, -1):
            cc, rr = c + sgn * dc, r + sgn * dr
            while grid.get((cc, rr)) == p:
                n += 1; cc += sgn * dc; rr += sgn * dr
        if n >= 4: return True
    return False

_solver = None
def _solve(m):
    global _solver
    if _solver is None:
        _solver = S.Solver(timeout=5)
    t = time.time()
    try:
        sc = _solver.solve_all(m)
    except S.SolverError:
        return m, None, time.time() - t
    return m, sc, time.time() - t

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    quota = [a.n // len(BUCKETS) + (1 if i < a.n % len(BUCKETS) else 0) for i in range(len(BUCKETS))]
    got = [dict() for _ in BUCKETS]   # moves -> (scores, source)
    seen = set(); skipped = [0] * len(BUCKETS); tsum = [0.0] * len(BUCKETS)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        rnd = 0
        while any(len(got[i]) < quota[i] for i in range(len(BUCKETS))) and rnd < 6:
            rnd += 1; cands = []
            for bi, (lo, hi) in enumerate(BUCKETS):
                need = quota[bi] - len(got[bi])
                if need <= 0: continue
                target = int(need * 1.15) + 5; tries = 0
                while target > 0 and tries < 200000:
                    tries += 1
                    ply = rng.randint(lo, hi); heur = rng.random() < 0.5
                    m = gen_one(ply, heur, rng)
                    if m is None: continue
                    k = board_key(m)
                    if k in seen: continue
                    seen.add(k); cands.append((bi, m, "heuristic" if heur else "random")); target -= 1
            info = {m: (bi, src) for bi, m, src in cands}
            for m, sc, dt in pool.imap_unordered(_solve, [m for _, m, _ in cands], chunksize=4):
                bi, src = info[m]
                tsum[bi] += dt
                if sc is None: skipped[bi] += 1; continue
                if len(got[bi]) < quota[bi]: got[bi][m] = (sc, src)
            print("round %d done %.0fs: %s skipped=%s" % (rnd, time.time() - t0, [len(g) for g in got], skipped), flush=True)
    out = os.path.join(ROOT, "data", "accuracy_positions.csv")
    rows = []
    for bi, g in enumerate(got):
        for m, (sc, src) in g.items():
            best = max(x for x in sc if x is not None)
            rows.append((len(m), m, sc, [i for i, x in enumerate(sc) if x == best], src, bi))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(out, "w") as f:
        f.write("moves,ply,scores,optimal_cols,source,bucket\n")
        for ply, m, sc, oc, src, bi in rows:
            f.write('%s,%d,"%s","%s",%s,%d-%d\n' % (m, ply, json.dumps(sc), json.dumps(oc), src, *BUCKETS[bi]))
    print("wrote", out, len(rows), "rows; per bucket", [len(g) for g in got], "skipped(>5s)", skipped,
          "solve-time sum per bucket", [round(x) for x in tsum])
