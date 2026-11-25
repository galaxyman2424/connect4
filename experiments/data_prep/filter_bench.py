"""Remove training-set positions from a benchmark csv and re-balance its ply buckets.

    python experiments/data_prep/filter_bench.py --bench data/accuracy_positions_60k_raw.csv \
        --exclude nn_data/solver_3m.npz,nn_data/solver_300k.npz,nn_data/hard_8x128_s0.npz \
        --per-bucket 8333 --out data/accuracy_positions_50k.csv

Why: a big benchmark made on another machine (or before new training data existed) may
contain positions some net was trained on, which would inflate its accuracy.  This drops
every position (and its mirror image) that occurs in any --exclude file (.npz datasets in
nn/make_dataset.py format, or other benchmark .csv files), then keeps the same number of
positions in every ply bucket (--per-bucket, default = the smallest bucket after filtering),
so the buckets stay equally weighted like in the 3,000-position benchmark.

Prints how many positions each exclude file removed per bucket: a big overlap in plies 8-11
is expected (few distinct early positions exist), almost none after ply 20.
"""

import argparse
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_accuracy_set import BUCKETS, _moves_key, exclusion_keys  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--exclude", default="", help="comma list of .npz datasets / .csv benchmarks")
    ap.add_argument("--per-bucket", type=int, default=0, help="positions kept per ply bucket (0 = smallest bucket)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.bench, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys()) if rows else []
    keys = [_moves_key(r["moves"]) for r in rows]
    drop = [False] * len(rows)
    names = ["%d-%d" % b for b in BUCKETS]
    for path in [p for p in a.exclude.split(",") if p]:
        if not os.path.isfile(path):
            print("WARNING: not found, skipped: %s" % path)
            continue
        ex = exclusion_keys(path)
        per = {n: 0 for n in names}
        for i, k in enumerate(keys):
            if not drop[i] and k in ex:
                drop[i] = True
                per[rows[i]["bucket"]] = per.get(rows[i]["bucket"], 0) + 1
        print("%-40s removed %6d  %s" % (path, sum(per.values()), per))
    kept = {n: [r for r, d in zip(rows, drop) if not d and r["bucket"] == n] for n in names}
    sizes = {n: len(v) for n, v in kept.items()}
    k = a.per_bucket or min(sizes.values())
    short = {n: s for n, s in sizes.items() if s < k}
    if short:
        print("WARNING: buckets with fewer than %d positions keep all they have: %s" % (k, short))
    rng = random.Random(a.seed)
    out = []
    for n in names:
        v = kept[n]
        out += v if len(v) <= k else rng.sample(v, k)
    out.sort(key=lambda r: (int(r["ply"]), r["moves"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    print("wrote %s: %d positions, per bucket %s (after filtering: %s)" % (
        a.out, len(out), {n: min(k, s) for n, s in sizes.items()}, sizes))


if __name__ == "__main__":
    main()
