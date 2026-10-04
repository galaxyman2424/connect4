"""JSON endpoints behind the Lab page (/lab) and the neural-net opponent.

Everything is read from files the experiment scripts write, so the page
always shows whatever is in the repo right now:

  runs/<run>/config.json, metrics.jsonl, eval.json, notes.md   (nn/ training)
  experiments/results/arena/<name>/summary.json                (arena/)
  experiments/results/*.csv, elo.json                          (minimax reference)

GET /api/lab/runs              list of training runs (+ a short summary each)
GET /api/lab/run/<run>         config, metrics, eval, notes of one run
GET /api/lab/arena             list of tournaments
GET /api/lab/arena/<name>      summary.json of one tournament
GET /api/lab/reference         minimax baselines (accuracy, Elo ladder, tactics, speed)
GET /api/nn/models             runs that have an exported model.npz (play page menu)
"""

import csv
import json
import os
import re

from flask import Blueprint, abort, jsonify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
RESULTS = os.path.join(ROOT, "experiments", "results")
ARENA = os.path.join(RESULTS, "arena")
SAFE = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")

bp = Blueprint("lab", __name__)


def _safe_dir(base, name):
    if not SAFE.match(name or "") or name.startswith("."):
        abort(404)
    p = os.path.join(base, name)
    if not os.path.isdir(p):
        abort(404)
    return p


def _json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _metrics(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass          # a line being written right now
    except OSError:
        pass
    return out


def _run_summary(name):
    d = os.path.join(RUNS, name)
    cfg = _json(os.path.join(d, "config.json"), {})
    m = _metrics(os.path.join(d, "metrics.jsonl"))
    last = m[-1] if m else {}
    best = max((r.get("bench_optimal") or 0 for r in m), default=None)
    elos = [r["elo"] for r in m if r.get("elo") is not None]
    ev = _json(os.path.join(d, "eval.json"))
    return {"name": name, "phase": cfg.get("phase"), "created": cfg.get("created"),
            "model": cfg.get("model"), "params": cfg.get("params"), "device": cfg.get("device"),
            "points": len(m), "last": {k: last.get(k) for k in ("epoch", "gen", "total_games", "elapsed_sec",
                                                                   "bench_optimal", "bench_value_acc", "time")},
            "best_bench_optimal": best, "last_elo": elos[-1] if elos else None,
            "has_model": os.path.isfile(os.path.join(d, "model.npz")),
            "has_eval": ev is not None,
            "eval_elo": (ev or {}).get("ladder", {}).get("elo")}


@bp.get("/api/lab/runs")
def runs():
    out = []
    if os.path.isdir(RUNS):
        for name in sorted(os.listdir(RUNS)):
            if SAFE.match(name) and os.path.isfile(os.path.join(RUNS, name, "config.json")):
                out.append(_run_summary(name))
    out.sort(key=lambda r: r.get("created") or "", reverse=True)
    return jsonify(out)


@bp.get("/api/lab/run/<name>")
def run(name):
    d = _safe_dir(RUNS, name)
    ev = _json(os.path.join(d, "eval.json"))
    if ev and "ladder" in ev:
        ev["ladder"] = {k: v for k, v in ev["ladder"].items() if k != "games"}   # keep the payload small
    notes = ""
    try:
        with open(os.path.join(d, "notes.md"), encoding="utf-8") as f:
            notes = f.read()
    except OSError:
        pass
    return jsonify({"name": name, "config": _json(os.path.join(d, "config.json"), {}),
                    "metrics": _metrics(os.path.join(d, "metrics.jsonl")), "eval": ev, "notes": notes})


@bp.get("/api/lab/arena")
def arenas():
    out = []
    if os.path.isdir(ARENA):
        for name in sorted(os.listdir(ARENA)):
            s = _json(os.path.join(ARENA, name, "summary.json")) if SAFE.match(name) else None
            if s:
                out.append({"name": name, "created": s.get("meta", {}).get("created"),
                            "bots": len(s.get("bots", [])), "games": s.get("games")})
    out.sort(key=lambda r: r.get("created") or "", reverse=True)       # newest first ...
    out.sort(key=lambda r: r["name"] != "main")                         # ... but "main" on top
    return jsonify(out)


@bp.get("/api/lab/arena/<name>")
def arena(name):
    d = _safe_dir(ARENA, name)
    s = _json(os.path.join(d, "summary.json"))
    if s is None:
        abort(404)
    return jsonify(s)


def _csv(name):
    try:
        with open(os.path.join(RESULTS, name), newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


@bp.get("/api/lab/reference")
def reference():
    acc = {}
    for r in _csv("accuracy_accuracy_summary.csv"):
        acc.setdefault("d" + r["depth"], {})[r["bucket"]] = {
            "optimal": float(r["optimal"]), "keeps_nl": float(r["keeps_nl"] or 0), "ms": float(r["mean_ms"])}
    tac = {}
    for r in _csv("tactics_summary.csv"):
        if r["agent"].startswith("d"):
            tac.setdefault(r["agent"], {})[r["category"]] = float(r["share_solved"])
    elo = _json(os.path.join(RESULTS, "elo.json"), {}).get("ratings", {})
    return jsonify({"accuracy": acc, "tactics": tac, "elo": elo})


@bp.get("/api/nn/models")
def models():
    out = []
    if os.path.isdir(RUNS):
        for name in sorted(os.listdir(RUNS)):
            p = os.path.join(RUNS, name, "model.npz")
            if SAFE.match(name) and os.path.isfile(p):
                s = _run_summary(name)
                out.append({"run": name, "phase": s["phase"], "best_bench_optimal": s["best_bench_optimal"],
                            "elo": s["eval_elo"] or s["last_elo"], "points": s["points"]})
    return jsonify(out)


# ------------------------------------------------------------ NN opponent
_NETS = {}


def nn_evaluator(run_name):
    """Cached NumpyNet for runs/<run>/model.npz (reloaded when the file changes)."""
    if not SAFE.match(run_name or ""):
        raise ValueError("bad run name")
    p = os.path.join(RUNS, run_name, "model.npz")
    if not os.path.isfile(p):
        raise ValueError("no model for run %r" % run_name)
    key = (p, os.path.getmtime(p))
    if _NETS.get(run_name, (None,))[0] != key:
        from nn.numpy_net import NumpyNet
        _NETS[run_name] = (key, NumpyNet(p))
    return _NETS[run_name][1]
