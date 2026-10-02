"""Regenerate the UCI Connect-4 dataset (67,557 positions) and label with Pons' solver.

UCI description: all legal 8-ply positions, neither player has won, next move not forced.
Empirically (verified, exactly 67,557): distinct 8-ply boards, no winner, NEITHER side has an
immediate winning move available, deduplicated up to left-right mirror symmetry.
Label = outcome for the first player (x) under perfect play: win / loss / draw.

Usage: python label_uci.py [--workers 2]   (resumable; writes data/uci/_shard*.tsv, then merges)
"""
import argparse, multiprocessing as mp, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, HERE)
import solver as S
from enum_uci import enumerate_positions, can_win, W, H

OUT = os.path.join(ROOT, "data", "uci")

def positions():
    lv = enumerate_positions(8)
    best = {}
    for pos, seq in lv.items():
        if can_win(pos, 0) or can_win(pos, 1):
            continue
        key = min(pos, pos[::-1])
        if key == pos:
            best[key] = seq
        elif key not in best:
            best[key] = "".join(str(8 - int(ch)) for ch in seq)  # mirror the sequence -> key position
    items = sorted(best.items(), key=lambda kv: kv[1])
    return items

def grid_cells(pos):
    cells = []
    for c in range(W):
        for r in range(H):
            cells.append("b" if r >= len(pos[c]) else ("x" if pos[c][r] == 0 else "o"))
    return cells

def work(args):
    shard, seqs = args
    path = os.path.join(OUT, "_shard%d.tsv" % shard)
    done = set()
    if os.path.exists(path):
        done = {l.split("\t")[0] for l in open(path) if l.endswith("\n")}
    t0 = time.time(); n = 0
    with S.Solver(analyze=False, weak=True, timeout=600) as s, open(path, "a") as f:
        for m in seqs:
            if m in done: continue
            v = s.solve(m)
            f.write("%s\t%d\n" % (m, v)); f.flush(); n += 1
            if n % 500 == 0:
                print("shard %d: %d/%d  %.1fs" % (shard, n + len(done), len(seqs), time.time() - t0), flush=True)
    return path

if __name__ == "__main__":
    import random
    from collections import Counter
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="label only this many (random order, seed 0); 0 = all 67,557")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    items = positions()
    print("positions:", len(items), flush=True)
    assert len(items) == 67557
    seqs = [m for _, m in items]
    order = seqs[:]; random.Random(0).shuffle(order)       # fixed random order => prefix = random sample
    todo = order[:a.limit] if a.limit else order
    k = a.workers
    if not a.merge_only:
        with mp.Pool(k) as p:
            p.map(work, [(i, todo[i::k]) for i in range(k)], chunksize=1)
    res = {}
    for f in os.listdir(OUT):
        if f.startswith("_shard"):
            for l in open(os.path.join(OUT, f)):
                if l.endswith("\n"):
                    m, v = l.split(); res[m] = int(v)
    lab = lambda v: "win" if v > 0 else ("loss" if v < 0 else "draw")
    with open(os.path.join(OUT, "positions.csv"), "w") as fc:
        fc.write("moves,label," + ",".join("c%d" % i for i in range(42)) + "\n")
        for pos, m in items:
            fc.write("%s,%s,%s\n" % (m, lab(res[m]) if m in res else "", ",".join(grid_cells(pos))))
    # connect-4.data in the original UCI line format, only when every position is labeled
    if len(res) == len(items):
        with open(os.path.join(OUT, "connect-4.data"), "w") as fd:
            for pos, m in items:
                fd.write(",".join(grid_cells(pos)) + "," + lab(res[m]) + "\n")
    print("labeled %d/%d" % (len(res), len(items)), Counter(lab(v) for v in res.values()))
