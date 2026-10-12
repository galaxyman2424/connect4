"""One standard evaluation of every learning method -> table + charts.

    python -m rl.compare                         # finds every run in runs/ automatically
    python -m rl.compare --name final --openings 20 --positions 3000
    python -m rl.compare --add "Evolved d6=weights:runs/evo_d2/weights.json:6@evo_d2"
    python -m rl.compare --list                  # show what would be evaluated, then stop

For every entry (a way of playing + the run that trained it):
  strength   games vs our minimax at --depths from --openings random openings
             x 2 colours -> Elo on the minimax ladder (depth 1 = 1000), the
             same scale as the AlphaZero and supervised runs
  accuracy   share of --positions held-out positions where it picks a move the
             perfect solver rates best (optimal) / keeps the win-draw result
  speed      milliseconds per move (on the CPU, numpy, like the website)
  cost       from runs/<run>/cost.json (rl.meter): wall hours, GPU hours, CPU
             core-hours, energy (kWh) and its price.  Distilled students also
             get "cost incl. teachers" (a student is only as cheap as the
             teachers it copied).  Runs without cost.json fall back to the
             elapsed time in metrics.jsonl (energy unknown).

Entries found automatically (per run folder, by its config "phase"):
  supervised   policy, + MCTS 200            selfplay  policy, + MCTS 200
  evolution    evolved weights in minimax at --search-depth (+ the hand-crafted
               weights at the same depth as the reference)
  ntuple       greedy (no search), + minimax at --search-depth
  dqn          policy (argmax Q)             ppo       policy, + MCTS 200
  distill      student policy (best.npz)

Output: experiments/results/methods/<name>/summary.json, summary.csv,
elo_vs_cost.png, elo_vs_speed.png.  The Lab page's "Method comparison" tab
reads summary.json.  Results are cached per entry (same spec + same model
file = not re-run); --force re-runs everything.
"""

import argparse
import csv
import json
import os
import random
import time

from nn import bench
from nn.common import RESULTS, RUNS, log, now_iso

from .evalhook import ladder, spec_accuracy

OUT = os.path.join(RESULTS, "methods")
PHASE_LABEL = {"supervised": "Supervised (solver labels)", "selfplay": "AlphaZero self-play",
               "evolution": "Evolution (CMA-ES)", "ntuple": "TD(λ) n-tuple", "dqn": "DQN",
               "ppo": "PPO self-play", "distill": "Distillation"}


def parse_list(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def _cfg(run):
    try:
        with open(os.path.join(RUNS, run, "config.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def rel(p):
    return os.path.relpath(p, os.path.dirname(RUNS)).replace("\\", "/")


def auto_entries(search_depth, mcts_sims):
    out = []
    if not os.path.isdir(RUNS):
        return out
    hand_added = False
    for run in sorted(os.listdir(RUNS)):
        cfg = _cfg(run)
        if not cfg:
            continue
        ph = cfg.get("phase")
        d = os.path.join(RUNS, run)
        model = os.path.join(d, "best.npz") if os.path.isfile(os.path.join(d, "best.npz")) else os.path.join(d, "model.npz")
        name = PHASE_LABEL.get(ph, ph)
        if ph in ("supervised", "selfplay", "ppo") and os.path.isfile(model):
            out.append({"label": "%s: policy only [%s]" % (name, run), "method": ph, "run": run, "search": "none",
                        "spec": ["nn", rel(model), "policy"]})
            out.append({"label": "%s + MCTS %d [%s]" % (name, mcts_sims, run), "method": ph, "run": run,
                        "search": "mcts%d" % mcts_sims, "spec": ["nn", rel(model), "mcts", mcts_sims]})
        elif ph in ("dqn", "distill") and os.path.isfile(model):
            out.append({"label": "%s: policy only [%s]" % (name, run), "method": ph, "run": run, "search": "none",
                        "spec": ["nn", rel(model), "policy"]})
        elif ph == "evolution" and os.path.isfile(os.path.join(d, "weights.json")):
            out.append({"label": "%s weights, minimax d%d [%s]" % (name, search_depth, run), "method": ph, "run": run,
                        "search": "minimax d%d" % search_depth,
                        "spec": ["weights", rel(os.path.join(d, "weights.json")), search_depth]})
            if not hand_added:
                out.append({"label": "Hand-crafted weights, minimax d%d (reference)" % search_depth, "method": "hand",
                            "run": None, "search": "minimax d%d" % search_depth, "spec": ["minimax", search_depth]})
                hand_added = True
        elif ph == "ntuple" and os.path.isfile(os.path.join(d, "ntuple.npz")):
            p = rel(os.path.join(d, "ntuple.npz"))
            out.append({"label": "%s: greedy, no search [%s]" % (name, run), "method": ph, "run": run,
                        "search": "none", "spec": ["ntuple", p, 0]})
            out.append({"label": "%s + minimax d%d [%s]" % (name, search_depth, run), "method": ph, "run": run,
                        "search": "minimax d%d" % search_depth, "spec": ["ntuple", p, search_depth]})
    return out


def parse_add(s):
    """'label=kind:path:arg[:arg]@run' -> entry."""
    label, rest = s.split("=", 1)
    run = None
    if "@" in rest:
        rest, run = rest.rsplit("@", 1)
    parts = rest.split(":")
    spec = [parts[0]] + [int(x) if x.isdigit() else x for x in parts[1:]]
    cfg = _cfg(run) if run else None
    return {"label": label.strip(), "method": (cfg or {}).get("phase", "custom"), "run": run,
            "search": "custom", "spec": spec}


def run_cost(run, seen=None):
    """Training cost of a run (+ its teachers for distillation)."""
    if not run:
        return None
    seen = seen or set()
    if run in seen:
        return None
    seen.add(run)
    d = os.path.join(RUNS, run)
    cost = None
    try:
        with open(os.path.join(d, "cost.json"), encoding="utf-8") as f:
            cost = dict(json.load(f)["total"])
            cost["source"] = "cost.json"
    except (OSError, ValueError, KeyError):
        m = bench_metrics(run)
        if m:
            el = max((r.get("elapsed_sec") or 0) for r in m)
            cost = {"wall_hours": round(el / 3600, 3), "gpu_hours": None, "cpu_core_hours": None,
                    "energy_kwh": None, "energy_cost_usd": None, "source": "metrics elapsed_sec (no rl.meter)"}
    cfg = _cfg(run) or {}
    if cfg.get("phase") == "distill" and cost:
        inc = dict(cost)
        for t in cfg.get("teachers", []):
            parts = t.split(":")
            if len(parts) > 1 and "runs/" in parts[1].replace("\\", "/"):
                trun = parts[1].replace("\\", "/").split("runs/")[1].split("/")[0]
                tc = run_cost(trun, seen)
                if tc:
                    for k in ("wall_hours", "gpu_hours", "cpu_core_hours", "energy_kwh", "energy_cost_usd"):
                        if inc.get(k) is not None and tc.get(k) is not None:
                            inc[k] = round(inc[k] + tc[k], 4)
        cost["including_teachers"] = inc
    return cost


def bench_metrics(run):
    from nn.common import Run
    try:
        return Run(run, create=False).read_metrics()
    except OSError:
        return []


def model_size(entry):
    spec = entry["spec"]
    root = os.path.dirname(RUNS)
    if spec[0] in ("nn", "ntuple", "weights"):
        p = os.path.join(root, spec[1])
        return os.path.getsize(p) if os.path.isfile(p) else None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="main", help="results go to experiments/results/methods/<name>/")
    ap.add_argument("--add", action="append", default=[], help='extra entry "label=spec@run" (see top)')
    ap.add_argument("--only", default="", help="comma list of phases to include (default: all)")
    ap.add_argument("--no-auto", action="store_true", help="only the --add entries")
    ap.add_argument("--search-depth", type=int, default=4, help="minimax depth for evolved weights / n-tuple")
    ap.add_argument("--mcts-sims", type=int, default=200)
    ap.add_argument("--depths", default="1,2,4,6,8", help="minimax opponents for the Elo")
    ap.add_argument("--openings", type=int, default=10)
    ap.add_argument("--positions", type=int, default=1000, help="benchmark positions for accuracy and speed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    entries = [] if a.no_auto else auto_entries(a.search_depth, a.mcts_sims)
    if a.only:
        keep = set(a.only.split(","))
        entries = [e for e in entries if e["method"] in keep or e["method"] == "hand"]
    entries += [parse_add(s) for s in a.add]
    if not entries:
        raise SystemExit("nothing to compare: no trained runs in runs/ (and no --add)")
    if a.list:
        for e in entries:
            print("%-60s %s" % (e["label"], e["spec"]))
        return

    out_dir = os.path.join(OUT, a.name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    old = {}
    if os.path.isfile(path) and not a.force:
        with open(path, encoding="utf-8") as f:
            old = {e["key"]: e for e in json.load(f).get("entries", [])}
    depths = parse_list(a.depths)
    rows = bench.load_accuracy_bench()
    sub = random.Random(5).sample(rows, min(a.positions, len(rows)))
    root = os.path.dirname(RUNS)
    results = []

    def save():
        doc = {"name": a.name, "created": now_iso(),
               "settings": {"depths": depths, "openings": a.openings, "positions": len(sub),
                            "search_depth": a.search_depth, "mcts_sims": a.mcts_sims},
               "ladder_elo": bench.minimax_ladder_elo(), "entries": results}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

    for e in entries:
        spec = tuple(e["spec"])
        f = os.path.join(root, spec[1]) if spec[0] in ("nn", "ntuple", "weights") else None
        key = json.dumps([e["label"], list(spec), os.path.getmtime(f) if f and os.path.isfile(f) else 0,
                          depths, a.openings, len(sub)])
        if key in old:
            r = old[key]
            r["cost"] = run_cost(e["run"])                    # cost may have grown (resumed training)
            results.append(r)
            log("cached: %s" % e["label"])
            continue
        t0 = time.time()
        log("%s ..." % e["label"])
        abs_spec = tuple([spec[0], os.path.join(root, spec[1])] + list(spec[2:])) if f else spec
        acc = spec_accuracy(abs_spec, sub, a.workers)
        lad = ladder(abs_spec, depths, a.openings, a.workers)
        r = dict(e)
        r.update({"key": key, "elo": lad["elo"], "vs_minimax": lad["vs"], "optimal": acc["optimal"],
                  "keeps_nl": acc["keeps_nl"], "ms_per_move": acc["ms"], "model_bytes": model_size(e),
                  "cost": run_cost(e["run"]), "eval_sec": round(time.time() - t0, 1)})
        cfg = _cfg(e["run"]) if e["run"] else None
        r["params"] = (cfg or {}).get("params")
        results.append(r)
        log("  Elo %s  optimal %.3f  %.1f ms/move  [%.0fs]" % (r["elo"], r["optimal"], r["ms_per_move"], r["eval_sec"]))
        save()
    save()

    # ---- table
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["label", "method", "run", "search", "elo", "optimal", "keeps_nl", "ms_per_move", "params",
                    "wall_hours", "gpu_hours", "cpu_core_hours", "energy_kwh", "energy_cost_usd",
                    "energy_kwh_incl_teachers"])
        for r in results:
            c = r.get("cost") or {}
            w.writerow([r["label"], r["method"], r["run"], r["search"], r["elo"], r["optimal"], r["keeps_nl"],
                        r["ms_per_move"], r.get("params"), c.get("wall_hours"), c.get("gpu_hours"),
                        c.get("cpu_core_hours"), c.get("energy_kwh"), c.get("energy_cost_usd"),
                        (c.get("including_teachers") or {}).get("energy_kwh")])
    print()
    print("%-58s %7s %8s %9s %9s %9s" % ("entry", "Elo", "optimal", "ms/move", "train h", "kWh"))
    for r in sorted(results, key=lambda r: -(r["elo"] or 0)):
        c = r.get("cost") or {}
        print("%-58s %7s %8.3f %9.1f %9s %9s" % (r["label"][:58], r["elo"], r["optimal"], r["ms_per_move"],
                                                c.get("wall_hours", "-"), c.get("energy_kwh", "-")))
    plots(results, out_dir)
    log("wrote %s (+ summary.csv, charts). Lab page: Method comparison tab." % path)


def plots(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed - skipping the PNG charts (the Lab page draws them anyway)")
        return
    methods = sorted({r["method"] for r in results})
    cmap = plt.get_cmap("tab10")
    color = {m: cmap(i % 10) for i, m in enumerate(methods)}
    for xkey, fname, xlabel in (("energy_kwh", "elo_vs_cost.png", "training energy (kWh, log scale)"),
                                ("wall_hours", "elo_vs_time.png", "training time (hours, log scale)"),
                                ("ms_per_move", "elo_vs_speed.png", "milliseconds per move (log scale)")):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for r in results:
            c = r.get("cost") or {}
            x = r.get("ms_per_move") if xkey == "ms_per_move" else c.get(xkey)
            if x is None or r.get("elo") is None or x <= 0:
                continue
            ax.scatter([x], [r["elo"]], color=color[r["method"]], s=60,
                       marker="o" if r["search"] == "none" else "^")
            ax.annotate(r["label"].split(" [")[0], (x, r["elo"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Elo (minimax depth 1 = 1000)")
        ax.grid(alpha=0.3)
        handles = [plt.Line2D([], [], color=color[m], marker="o", ls="", label=PHASE_LABEL.get(m, m)) for m in methods]
        handles += [plt.Line2D([], [], color="grey", marker="o", ls="", label="no search"),
                    plt.Line2D([], [], color="grey", marker="^", ls="", label="with search")]
        ax.legend(handles=handles, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
