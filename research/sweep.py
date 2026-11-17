"""Supervised size x teacher sweep: queue, training, evaluation, sweep.json.

    python -m research.sweep --plan                 # show the queue and what is done
    python -m research.sweep --run                  # run everything not finished (resumable)
    python -m research.sweep --run --only 'rs_A_.*' # only runs whose name matches the regex
    python -m research.sweep --analyse              # rebuild sweep.json from finished runs
    nohup setsid .venv/bin/python -m research.sweep --run > runs/rs_sweep.log 2>&1 &

Groups (all students = nn.model.C4Net trained supervised on the SAME 300k positions of
nn_data/solver_300k.npz with the same fixed train/val split; only teacher and size differ):
  A  size ladder on solver (Pons) labels: 1x16 2x32 3x48 4x64 6x64 6x96 8x128 (+ seeds)
  B  teacher comparison at 2x32 and 6x64: every teacher in research/teachers.py
  C  distillation ladder: soft targets of sup_6x64 and of the new 8x128 (rs_A_pons_8x128)
     -> students 1x16 2x32 3x48 4x64
A run is finished when runs/<name>/research.json exists.  Training-done-but-not-evaluated
runs (train.json, no research.json) are only evaluated on resume; a run folder with
neither is incomplete and is deleted and retrained.  Failures are recorded and skipped;
the next --run retries them.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"          # this process measures single-thread numpy speed

from nn.common import ROOT, RUNS, log, now_iso  # noqa: E402

OUT_DIR = os.path.join(ROOT, "experiments", "results", "research")
OUT_JSON = os.path.join(OUT_DIR, "sweep.json")
OUT_CSV = os.path.join(OUT_DIR, "sweep_runs.csv")
LOCK = os.path.join(RUNS, "rs_sweep.lock")

LADDER = [(1, 16), (2, 32), (3, 48), (4, 64), (6, 96), (8, 128)]
SIZES_A = [(1, 16), (2, 32), (3, 48), (4, 64), (6, 64), (6, 96), (8, 128)]
SIZES_B = [(2, 32), (6, 64)]
SIZES_C = [(1, 16), (2, 32), (3, 48), (4, 64)]
TEACHERS_B = ["heur_hand", "heur_evo_g10", "heur_evo_g20", "heur_evo_g40", "heur_evo_g60",
              "mm_d4_hand", "mm_d4_evo60", "ppo_net", "ppo_from_sup"]
TEACHERS_C = ["sup6x64", "sup8x128"]
# Epochs per run, the same for every size: measured on the RTX 3050 (2-epoch timing runs,
# epoch incl. benchmark) 1x16 ~2 s, 2x32 ~2 s, 6x64 ~7 s, 8x128 ~25 s per epoch, so 40 epochs
# for the whole queue is ~2.5-3 h of GPU time + ~30 s evaluation per run.
EPOCHS = {"default": 40}


def epochs_for(b, c):
    return EPOCHS.get("%dx%d" % (b, c), EPOCHS["default"])


def run_name(group, teacher, b, c, seed=0):
    return "rs_%s_%s_%dx%d%s" % (group, teacher, b, c, "" if seed == 0 else "_s%d" % seed)


def plan(extra_seeds=True, optional=True):
    """Ordered queue: A (seed 0) -> B -> C -> extra seeds -> optional 10x128."""
    q = []

    def add(group, teacher, b, c, seed=0, opt=False):
        name = run_name(group, teacher, b, c, seed)
        if not any(r["name"] == name for r in q):
            q.append({"name": name, "group": group, "teacher": teacher, "blocks": b, "channels": c,
                      "seed": seed, "epochs": epochs_for(b, c), "optional": opt})
    for b, c in SIZES_A:
        add("A", "pons", b, c)
    for t in TEACHERS_B:
        for b, c in SIZES_B:
            add("B", t, b, c)
    for t in TEACHERS_C:
        for b, c in SIZES_C:
            add("C", t, b, c)
    if extra_seeds:
        for b, c in SIZES_A:
            if b * c <= 6 * 64:                      # sizes <= 4x64, plus the 6x64 reference
                add("A", "pons", b, c, seed=1)
        for t in TEACHERS_B:                         # teacher comparison noise at 2x32
            add("B", t, 2, 32, seed=1)
    if optional:
        add("A", "pons", 10, 128, opt=True)
    return q


def param_count(b, c):
    """Exact C4Net parameter count (no torch needed)."""
    conv = lambda ci, co, k: ci * co * k * k                     # noqa: E731
    p = conv(4, c, 3) + 2 * c                                    # stem conv + BN
    p += b * 2 * (conv(c, c, 3) + 2 * c)                         # residual blocks
    p += conv(c, 2, 1) + 2 * 2 + (2 * 42 * 7 + 7)                # policy head
    p += conv(c, 4, 1) + 2 * 4 + (4 * 42 * 64 + 64) + (64 * 3 + 3)   # value head
    return p


# ------------------------------------------------------------------ state
def run_dir(name):
    return os.path.join(RUNS, name)


def run_state(name):
    d = run_dir(name)
    if os.path.isfile(os.path.join(d, "research.json")):
        return "done"
    if os.path.isfile(os.path.join(d, "research_failed.json")):
        return "failed"
    if os.path.isfile(os.path.join(d, "train.json")):
        return "trained"
    if os.path.isdir(d):
        return "partial"
    return "queued"


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ------------------------------------------------------------------ one run
def teacher_ready(teacher):
    from research.teachers import NET_TEACHERS
    if teacher in NET_TEACHERS:
        return os.path.isfile(os.path.join(ROOT, NET_TEACHERS[teacher]))
    return True


def execute(spec, train_fn=None, eval_fn=None):
    """Train (unless already trained) and evaluate one run; writes research.json."""
    name = spec["name"]
    d = run_dir(name)
    st = run_state(name)
    if st in ("partial", "failed"):
        shutil.rmtree(d, ignore_errors=True)
    if run_state(name) != "trained":
        tr = (train_fn or _train)(spec)
        write_json(os.path.join(d, "train.json"), tr)
    tr = read_json(os.path.join(d, "train.json"))
    ev = (eval_fn or _evaluate)(spec)
    res = dict(spec)
    res.update({"params": tr["params"], "train": tr, "train_sec": tr["train_sec"],
                "gpu_sec": tr["gpu_sec"], "device": tr.get("device_name"),
                "teacher_agreement": tr["val_teacher_agree"], "best_epoch": tr["best_epoch"]})
    res.update(ev)
    res["status"] = "done"
    res["finished"] = now_iso()
    write_json(os.path.join(d, "research.json"), res)
    return res


def _train(spec):
    from research.sweep_train import train
    from research.teachers import ensure_labels, load_labels
    ensure_labels(spec["teacher"])
    labels = load_labels(spec["teacher"])
    return train(spec["name"], spec["teacher"], labels, spec["blocks"], spec["channels"], spec["seed"],
                 spec["epochs"], extra_cfg={"group": spec["group"], "sweep": "research.sweep"}, logf=log)


def _evaluate(spec):
    from nn import bench
    from nn.numpy_net import NumpyNet
    from research.sweep_gauntlet import gauntlet, ms_per_position
    path = os.path.join(run_dir(spec["name"]), "best.npz")
    t0 = time.time()
    pb = bench.policy_eval(NumpyNet(path), bench.load_accuracy_bench())
    b = {k: pb[k] for k in ("optimal", "keeps_nl", "value_acc", "n", "by_bucket", "value_acc_by_bucket")}
    ms = ms_per_position(path)
    g = gauntlet(path)
    return {"bench": b, "ms_per_pos": ms, "gauntlet": g, "eval_sec": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ sweep.json
def teachers_table():
    from research.teachers import DEFAULT_TEACHERS, label_path, parse
    out = []
    for t in DEFAULT_TEACHERS + ["sup8x128"]:
        p = label_path(t)
        meta = {}
        if os.path.isfile(p):
            import numpy as np
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["meta_json"]))
        s = parse(t)
        out.append({"id": t, "label": s["label"], "kind": s["kind"], "search_depth": s["search_depth"],
                    "gen": s.get("gen"), "params": _net_params(s), "cached": bool(meta),
                    "bench_optimal": meta.get("bench_optimal"), "bench_keeps_nl": meta.get("bench_keeps_nl"),
                    "bench_by_bucket": meta.get("bench_by_bucket"),
                    "agree_pons": meta.get("agree_pons"), "label_sec": meta.get("label_sec"),
                    "label_ms_per_pos": meta.get("label_ms_per_pos"), "workers": meta.get("workers"),
                    "single_ms_per_pos": meta.get("single_ms_per_pos"), "value_sigma": meta.get("value_sigma"),
                    "mate_share": meta.get("mate_share"), "ties_mean": meta.get("ties_mean"),
                    "device": meta.get("device")})
    return out


def _net_params(spec):
    if spec["kind"] != "net":
        return None
    p = os.path.join(ROOT, spec["path"])
    if not os.path.isfile(p):
        return None
    import numpy as np
    with np.load(p, allow_pickle=False) as z:
        cfg = json.loads(str(z["config_json"]))
    return param_count(int(cfg["blocks"]), int(cfg["channels"]))


def collect_runs(queue, running=None):
    out = []
    for spec in queue:
        st = run_state(spec["name"])
        r = {k: v for k, v in spec.items() if not k.startswith("_")}
        r["params"] = param_count(spec["blocks"], spec["channels"])
        if st == "done":
            r.update(read_json(os.path.join(run_dir(spec["name"]), "research.json"), {}))
        elif st == "failed":
            r.update(read_json(os.path.join(run_dir(spec["name"]), "research_failed.json"), {}))
            r["status"] = "failed"
        else:
            r["status"] = "running" if spec["name"] == running else "queued"
        r.pop("train", None)
        out.append(r)
    return out


def estimate_sec(spec, done):
    """Seconds for a queued run: same-size finished runs scaled by epochs, else a
    log-log interpolation in params over finished runs, else 300 s."""
    import math
    same = [r for r in done if (r["blocks"], r["channels"]) == (spec["blocks"], spec["channels"])]
    tot = lambda r: r.get("train_sec", 0) + r.get("eval_sec", 0)   # noqa: E731
    if same:
        return sum(tot(r) * spec["epochs"] / max(r["epochs"], 1) for r in same) / len(same)
    pts = sorted({(r["params"], tot(r) / max(r["epochs"], 1)) for r in done})
    if not pts:
        return 300.0
    p = param_count(spec["blocks"], spec["channels"])
    if len(pts) == 1:
        return pts[0][1] * spec["epochs"]
    xs = [math.log(x) for x, _ in pts]
    ys = [math.log(max(y, 1e-3)) for _, y in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
    slope = max(slope, 0.0)
    return math.exp(my + slope * (math.log(p) - mx)) * spec["epochs"]


def build_json(queue, running=None, started=None, extra_status=None):
    from research.sweep_analysis import analyse, findings
    from research.sweep_gauntlet import N_OPENINGS, OPENING_SEED, OPPONENTS
    runs = collect_runs(queue, running)
    done = [r for r in runs if r["status"] == "done"]
    left = [r for r in runs if r["status"] in ("queued", "running")]
    eta_sec = sum(estimate_sec(r, done) for r in left)
    counts = {k: sum(r["status"] == k for r in runs) for k in ("queued", "running", "done", "failed")}
    status = {"state": "running" if running else ("done" if not left else "idle"), "counts": counts,
              "total": len(runs), "running": running, "started": started, "updated": now_iso(),
              "eta_sec": round(eta_sec), "eta_clock": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                                  time.localtime(time.time() + eta_sec))}
    status.update(extra_status or {})
    teachers = teachers_table()
    an = analyse(runs, teachers)
    obj = {"id": "sweep", "title": "Supervised sweep: student size x teacher (Connect 4 C4Net)",
           "generated": now_iso(), "status": status, "notes": NOTES + [
               "Epochs per run: %s." % json.dumps(EPOCHS),
               "Gauntlet: %d openings (nn.bench.random_openings seed %d, 2-4 plies) x 2 colours vs %s." % (
                   N_OPENINGS, OPENING_SEED, ", ".join(OPPONENTS))],
           "sizes": [{"blocks": b, "channels": c, "params": param_count(b, c)} for b, c in
                     SIZES_A + [(10, 128)]],
           "teachers": teachers, "runs": runs, "analysis": an, "findings": findings(runs, teachers, an)}
    write_json(OUT_JSON, obj)
    _write_csv(runs)
    return obj


def _write_csv(runs):
    import csv
    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["name", "group", "teacher", "blocks", "channels", "params", "seed", "epochs", "status",
            "bench_optimal", "bench_keeps_nl", "bench_value_acc", "gauntlet_mean", "teacher_agreement",
            "ms_per_pos", "train_sec", "gpu_sec"]
    with open(OUT_CSV + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            b = r.get("bench") or {}
            g = r.get("gauntlet") or {}
            w.writerow([r.get("name"), r.get("group"), r.get("teacher"), r.get("blocks"), r.get("channels"),
                        r.get("params"), r.get("seed"), r.get("epochs"), r.get("status"), b.get("optimal"),
                        b.get("keeps_nl"), b.get("value_acc"), g.get("mean_score"), r.get("teacher_agreement"),
                        r.get("ms_per_pos"), r.get("train_sec"), r.get("gpu_sec")])
    os.replace(OUT_CSV + ".tmp", OUT_CSV)


NOTES = [
    "All students are nn.model.C4Net (residual tower, blocks x channels, 7 policy logits, WDL head) trained "
    "supervised on the same 300,000 positions of nn_data/solver_300k.npz with the same fixed 95/5 train/val "
    "split (split seed 0 = the split of runs/sup_6x64); only teacher and size differ.",
    "Training recipe = nn/train_supervised.py: batch 1024, AdamW lr 2e-3 wd 1e-4, OneCycle, mirror "
    "augmentation, bf16 autocast on the RTX 3050, loss = CE(teacher policy) + 1.0 x CE(teacher WDL). "
    "best.npz = lowest validation loss against the TEACHER's labels (solver labels are never used to "
    "select a non-solver student).",
    "Heuristic teachers (heur_*): for each legal column play it and score the resulting position with the "
    "heuristic from the mover's view; a move completing four scores WIN-1 (taken); no other rule (no forced "
    "block). This equals engine.minimax.search(depth=1). Policy target = uniform over the columns tied for the "
    "best score (hard target, same shape as the Pons target). Value target = tanh(s/sigma) of the best score "
    "(sigma = std of that teacher's non-mate best scores; +-1 for proven wins/losses) mapped to "
    "(W,D,L) = (max(v,0), 1-|v|, max(-v,0)): a monotone squashing, not a calibrated probability.",
    "Minimax teachers (mm_d4_*): each legal column scored by depth-4 negamax (engine.minimax.search to depth 3 "
    "after the move); same target rules as the heuristic teachers. mm_d4_hand = slider level 5 "
    "(depth 4, epsilon 0).",
    "Network teachers (ppo_net, ppo_from_sup, sup6x64, sup8x128): soft policy = the network's softmax over "
    "legal columns and its WDL output, policy head only, no search/MCTS. Group C (distillation) uses these "
    "soft targets on the fixed 300k positions instead of rl/distill.py (which generates its own positions).",
    "Teacher bench numbers: the teacher's own move on the 3,000 held-out solver-scored positions "
    "(data/accuracy_positions.csv); agree_pons = share of the 300k positions where the teacher's move is "
    "Pons-optimal. Student bench = policy head only, no search.",
    "'6^2 x 64^2' = 36 blocks x 4096 channels would be about 36 * 2 * 9 * 4096^2 = 10.9 billion parameters: "
    "not trainable on this hardware (RTX 3050 8 GB). 8x128 (2.4 M parameters) is the top rung of the 1:16 "
    "ladder; 10x128 is an optional extra.",
    "Single seed per run unless a _s1 twin exists; benchmark n = 3,000 positions (standard error of an "
    "optimal-rate near 0.85 is about 0.0065); gauntlet games are deterministic given the opening.",
]


# ------------------------------------------------------------------ driver
def _locked():
    info = read_json(LOCK)
    if not info:
        return False
    try:
        os.kill(int(info["pid"]), 0)
        return True
    except (OSError, ValueError, KeyError):
        return False


def run_queue(queue, only=None, train_fn=None, eval_fn=None, write=True, retry_failed=True):
    """Run every unfinished run of ``queue`` (matching ``only``) in order.  Returns
    {name: final state}."""
    if _locked():
        raise SystemExit("another sweep driver is running (%s)" % LOCK)
    write_json(LOCK, {"pid": os.getpid(), "started": now_iso()})
    started = now_iso()
    pending = [s for s in queue if (not only or re.search(only, s["name"]))]
    deferred = 0
    try:
        while True:
            todo = [s for s in pending if run_state(s["name"]) != "done"
                    and (retry_failed or run_state(s["name"]) != "failed")]
            todo = [s for s in todo if s.get("_tried", 0) == 0]
            if not todo:
                break
            ready = [s for s in todo if teacher_ready(s["teacher"])]
            if not ready:
                for s in todo:
                    s["_tried"] = 1
                    log("SKIP %s: teacher %s not available yet" % (s["name"], s["teacher"]))
                deferred += len(todo)
                break
            spec = ready[0]
            spec["_tried"] = 1
            clean = {k: v for k, v in spec.items() if not k.startswith("_")}
            if write:
                build_json(queue, running=spec["name"], started=started)
            log("=== %s (%s, teacher %s, %dx%d, seed %d, %d epochs)" % (
                spec["name"], spec["group"], spec["teacher"], spec["blocks"], spec["channels"],
                spec["seed"], spec["epochs"]))
            t0 = time.time()
            try:
                res = execute(clean, train_fn, eval_fn)
                log("=== %s done in %.0fs: bench optimal %s keeps %s gauntlet %s" % (
                    spec["name"], time.time() - t0, res.get("bench", {}).get("optimal"),
                    res.get("bench", {}).get("keeps_nl"), res.get("gauntlet", {}).get("mean_score")))
            except Exception as e:      # keep going with the rest of the queue
                log("FAILED %s: %r" % (spec["name"], e))
                traceback.print_exc()
                os.makedirs(run_dir(spec["name"]), exist_ok=True)
                write_json(os.path.join(run_dir(spec["name"]), "research_failed.json"),
                           {"status": "failed", "error": repr(e), "traceback": traceback.format_exc(),
                            "failed": now_iso()})
            if write:
                build_json(queue, running=None, started=started)
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass
        if write:
            build_json(queue, running=None, started=started)
    return {s["name"]: run_state(s["name"]) for s in pending}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--only", default=None, help="regex on run names (with --run or alone)")
    ap.add_argument("--epochs", type=int, default=None, help="override every run's epochs (testing)")
    ap.add_argument("--no-extra-seeds", action="store_true")
    ap.add_argument("--no-optional", action="store_true")
    a = ap.parse_args()
    queue = plan(extra_seeds=not a.no_extra_seeds, optional=not a.no_optional)
    if a.epochs:
        for s in queue:
            s["epochs"] = a.epochs
    if a.run or (a.only and not a.plan and not a.analyse):
        res = run_queue(queue, a.only)
        log("finished: %s" % json.dumps({k: sum(v == k for v in res.values()) for k in set(res.values())}))
    elif a.analyse:
        running = started = None
        if _locked():          # a driver is active: keep sweep.json's status "running"
            info = read_json(LOCK, {})
            started = info.get("started")
            running = next((s["name"] for s in queue if run_state(s["name"]) in ("partial", "trained")), None)
        obj = build_json(queue, running=running, started=started)
        log("wrote %s (%d runs, %d done, %d findings)" % (OUT_JSON, len(obj["runs"]),
                                                          obj["status"]["counts"]["done"], len(obj["findings"])))
    else:
        done = []
        for i, s in enumerate(queue):
            st = run_state(s["name"])
            r = read_json(os.path.join(run_dir(s["name"]), "research.json")) if st == "done" else None
            if r:
                done.append(r)
            print("%2d %-30s %s %-12s %2dx%-3d %9d params  seed %d  %2d ep  %-8s %s" % (
                i + 1, s["name"], s["group"], s["teacher"], s["blocks"], s["channels"],
                param_count(s["blocks"], s["channels"]), s["seed"], s["epochs"], st,
                "" if not r else "opt %.4f  gauntlet %.3f" % (r["bench"]["optimal"], r["gauntlet"]["mean_score"])))
        left = [s for s in queue if run_state(s["name"]) != "done"]
        eta = sum(estimate_sec(s, done) for s in left)
        print("%d runs, %d left, ETA %.1f h" % (len(queue), len(left), eta / 3600))


if __name__ == "__main__":
    sys.exit(main())
