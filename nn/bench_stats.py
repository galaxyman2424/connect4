"""How precise is a benchmark accuracy, and is net A really better than net B?

    python -m nn.bench_stats --need                         # positions needed for a given precision
    python -m nn.bench_stats --runs score_8x128_s0_r1        # 95% CI per ply bucket, policy head
    python -m nn.bench_stats --runs score_8x128_s0_r1,score_8x128_s1_r1 --model best.npz
    python -m nn.bench_stats --runs a,b --bench data/accuracy_positions_20k.csv --backend torch
    python -m nn.bench_stats --runs a,b --out experiments/results/bench_stats_a_b.json

For every run: accuracy ("optimal" = one of the solver's best columns, same definition as
nn.evaluate) with a 95% Wilson interval, overall and per ply bucket.  For every PAIR of runs:
the positions only A gets right / only B gets right, the exact McNemar p-value and a paired
bootstrap 95% interval of the accuracy difference.  Paired testing is what makes 3,000
positions usable for comparing two strong nets: their mistakes overlap, so only the
positions where they disagree carry information.

--need prints the half-width of the 95% interval at several accuracies and benchmark sizes,
and how many positions a given half-width needs (normal approximation, n = 1.96^2 p(1-p)/h^2).
"""

import argparse
import json
import os

import numpy as np

from . import bench
from .common import RUNS, load_evaluator, log


def need_table(accs=(0.90, 0.95, 0.975, 0.99), sizes=(3000, 10000, 20000, 50000), halfs=(0.005, 0.0025, 0.001)):
    rows = []
    for p in accs:
        r = {"accuracy": p, "half_width_points": {}, "positions_for_half_width": {}}
        for n in sizes:
            lo, hi = bench.wilson_ci(round(p * n), n)
            r["half_width_points"][n] = round(50 * (hi - lo), 2)
        for h in halfs:
            r["positions_for_half_width"]["+-%g" % (100 * h)] = int(np.ceil(1.96 ** 2 * p * (1 - p) / h ** 2))
        rows.append(r)
    return rows


def correct_vector(model_path, rows, backend, device):
    ev = load_evaluator(model_path, device, backend=backend)
    cols = []
    for i in range(0, len(rows), 1024):
        ch = rows[i:i + 1024]
        pri, _, _ = ev.evaluate([r["cur"] for r in ch], [r["mask"] for r in ch])
        for r, p in zip(ch, pri):
            legal = [c for c in range(7) if r["scores"][c] is not None]
            cols.append(max(legal, key=lambda c: p[c]))
    return np.array([c in r["optimal"] for c, r in zip(cols, rows)]), cols


def paired(a, b, boot=10000, seed=0):
    from experiments import stats
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    rng = np.random.default_rng(seed)
    n = len(a)
    d = a.astype(np.int8) - b.astype(np.int8)
    diffs = np.concatenate([d[rng.integers(0, n, (k, n), dtype=np.int32)].mean(1)
                            for k in [500] * (boot // 500) + ([boot % 500] if boot % 500 else [])])
    return {"only_a": only_a, "only_b": only_b, "diff": round(float(d.mean()), 4),
            "diff_ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            "mcnemar_p": round(stats.mcnemar(only_a, only_b), 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="", help="comma list of run names (or .npz paths; 'a.npz+b.npz' = ensemble)")
    ap.add_argument("--model", default="best.npz", help="file inside each runs/<run>/")
    ap.add_argument("--bench", default="", help="benchmark csv (default data/accuracy_positions.csv)")
    ap.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--need", action="store_true", help="print the precision / sample-size table")
    ap.add_argument("--out", default="", help="also write the result as json here")
    a = ap.parse_args()

    out = {}
    if a.need or not a.runs:
        out["need"] = need_table()
        print("95% interval half-width (accuracy points) by benchmark size:")
        print("  accuracy  " + "  ".join("%7d" % n for n in (3000, 10000, 20000, 50000)) +
              "   positions for +-0.5 / +-0.25 / +-0.1 points")
        for r in out["need"]:
            print("  %6.1f%%  " % (100 * r["accuracy"]) +
                  "  ".join("%7.2f" % v for v in r["half_width_points"].values()) + "   " +
                  " / ".join("%d" % v for v in r["positions_for_half_width"].values()))
    if a.runs:
        rows = bench.load_accuracy_bench(a.bench or None)
        out["benchmark"] = os.path.relpath(bench.bench_path(a.bench or None), bench.ROOT).replace("\\", "/")
        out["positions"] = len(rows)
        vecs, out["runs"] = {}, {}
        for name in [r for r in a.runs.split(",") if r]:
            parts = [p if os.path.isfile(p) else os.path.join(RUNS, p, a.model) for p in name.split("+")]
            for p in parts:
                if not os.path.isfile(p):
                    raise SystemExit("model not found: %s" % p)
            log("scoring %s on %d positions ..." % (name, len(rows)))
            v, cols = correct_vector("+".join(parts), rows, a.backend, a.device)
            vecs[name] = v
            summ = bench.accuracy_summary(rows, cols)
            out["runs"][name] = summ
            print("%-28s optimal %.4f  95%% CI %.4f-%.4f (+-%.2f pts)  keeps %.4f" % (
                name, summ["optimal"], summ["optimal_ci95"][0], summ["optimal_ci95"][1],
                50 * (summ["optimal_ci95"][1] - summ["optimal_ci95"][0]), summ["keeps_nl"]))
            for bk, s in summ["by_bucket"].items():
                print("    ply %-6s n=%4d  %.4f  (%.4f-%.4f)" % (bk, s["n"], s["optimal"], *s["optimal_ci95"]))
        names = list(vecs)
        out["pairs"] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = paired(vecs[names[i]], vecs[names[j]])
                r.update(a=names[i], b=names[j])
                out["pairs"].append(r)
                print("%s vs %s: diff %+.4f (95%% CI %+.4f..%+.4f)  only-A %d  only-B %d  McNemar p=%.4g" % (
                    names[i], names[j], r["diff"], r["diff_ci95"][0], r["diff_ci95"][1],
                    r["only_a"], r["only_b"], r["mcnemar_p"]))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
