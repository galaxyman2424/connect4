"""JSON endpoints behind the Lab page (/lab) and the neural-net opponent.

Everything is read from files the experiment scripts write, so the page
always shows whatever is in the repo right now:

  runs/<run>/config.json, metrics.jsonl, eval.json, notes.md   (nn/ training)
  experiments/results/arena/<name>/summary.json                (arena/)
  experiments/results/*.csv, elo.json                          (minimax reference)

GET /api/lab/runs              list of training runs (+ a short summary each)
GET /api/lab/run/<run>         config, metrics, eval, notes of one run
GET /api/lab/evals             every nn.evaluate result (eval.json, eval_<tag>.json) + top-8 nets still missing one
GET /api/lab/eval/<run>/<f>    one of those files
GET /api/lab/compare           top-8 arena nets joined with benchmark, training time, hypothesis, ladder
GET /api/lab/arena             list of tournaments
GET /api/lab/arena/<name>      summary.json of one tournament
GET /api/lab/reference         minimax baselines (accuracy, Elo ladder, tactics, speed)
GET /api/lab/methods           list of method comparisons (rl/compare.py)
GET /api/lab/methods/<name>    summary.json of one comparison
GET /api/nn/models             runs that have an exported model.npz (play page menu)

Runs from rl/ (evolution, ntuple, dqn, ppo, distill) use the same files plus
cost.json (training cost from rl/meter.py), which is included when present.
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
METHODS = os.path.join(RESULTS, "methods")
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


RESEARCH = os.path.join(RESULTS, "research")
TOP_ARENA = "top8_o100_mcts_fixed"          # the arena run whose nets are "the top 8"
EVAL_FILE = re.compile(r"^eval(_[A-Za-z0-9_\-]{1,40})?\.json$")
_MEMO = {}


def _cached_json(path):
    """json file, re-read only when it changes."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    hit = _MEMO.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    v = _json(path)
    _MEMO[path] = (mt, v)
    return v


def _bench_all():
    """experiments/results/research/bench_all.json -> {run: net row} (every net on the 3,000 positions)."""
    b = _cached_json(os.path.join(RESEARCH, "bench_all.json")) or {}
    return {n["run"]: n for n in b.get("nets", [])}, b.get("meta", {})


def _top_arena_name():
    if os.path.isfile(os.path.join(ARENA, TOP_ARENA, "summary.json")):
        return TOP_ARENA
    cands = []
    for n in os.listdir(ARENA) if os.path.isdir(ARENA) else []:
        s = _cached_json(os.path.join(ARENA, n, "summary.json")) if n.startswith("top8") else None
        if s:
            cands.append(((s.get("meta") or {}).get("created") or "", n))
    return max(cands)[1] if cands else None


def _model_runs(model):
    """'runs/a/best.npz+runs/b/best.npz' -> ['a', 'b']."""
    out = []
    for p in (model or "").replace("\\", "/").split("+"):
        if "runs/" in p:
            out.append(p.split("runs/", 1)[1].split("/")[0])
    return out


def _top8():
    """Nets in the top-8 arena run: [{bot, runs, mode, label}] (Pons and Bruneton are not ours)."""
    name = _top_arena_name()
    s = _cached_json(os.path.join(ARENA, name, "summary.json")) if name else None
    out = []
    for bot in (s or {}).get("bots", []):
        inf = s.get("bot_info", {}).get(bot, {})
        if inf.get("group") != "ours":
            continue
        model = (inf.get("params") or {}).get("model", "")
        out.append({"bot": bot, "runs": _model_runs(model), "model": model, "label": inf.get("label"),
                    "mode": "mcts" + bot.rsplit("_mcts", 1)[1] if "_mcts" in bot else "policy"})
    return out


def _top8_runs():
    rs = set()
    for p in _top8():
        rs.update(p["runs"])
    return rs


FAMILY_BY_PHASE = {"selfplay": "self-play (AlphaZero)", "dqn": "self-play RL (PPO / DQN)",
                   "ppo": "self-play RL (PPO / DQN)", "distill": "net teacher (distillation)",
                   "evolution": "evolved heuristic (CMA-ES)", "ntuple": "TD(λ) n-tuple",
                   "supervised": "perfect labels (Pons)",
                   "supervised_score": "perfect labels + score head (Pons, 3M)"}


def _size(cfg):
    mc = cfg.get("model") or {}
    if mc.get("blocks"):
        return "%dx%d" % (mc["blocks"], mc.get("channels") or 0)
    if mc.get("kind") == "n-tuple":
        return "n-tuple"
    if mc.get("kind"):
        return "heuristic weights"
    return "?"


def _elapsed(run):
    """Wall seconds of one run (cost.json, else the last elapsed_sec in metrics.jsonl)."""
    d = os.path.join(RUNS, run)
    c = (_json(os.path.join(d, "cost.json")) or {}).get("total") or {}
    if c.get("wall_sec") is not None:
        return c["wall_sec"], c.get("energy_wh")
    m = _metrics(os.path.join(d, "metrics.jsonl"))
    el = max((r.get("elapsed_sec") or 0 for r in m), default=0)
    return (el or None), None


def _train_cost(run, seen=None):
    """Training time of a run INCLUDING the run it was started from (--init) and, for distillation,
    its teachers: a fine-tuned or distilled net is only as cheap as everything it was built on."""
    seen = set() if seen is None else seen
    if run in seen or not os.path.isfile(os.path.join(RUNS, run, "config.json")):
        return {"sec": 0, "chain": [], "energy_wh": None}
    seen.add(run)
    cfg = _json(os.path.join(RUNS, run, "config.json"), {})
    sec, wh = _elapsed(run)
    out = {"sec": sec or 0, "own_sec": sec, "chain": [run], "energy_wh": wh}
    parents = []
    init = ((cfg.get("args") or {}).get("init") or (cfg.get("init") or {}).get("from") or "")
    parents += _model_runs(init)
    for t in cfg.get("teachers", []) or []:
        parents += _model_runs(str(t).split(":")[1] if ":" in str(t) else str(t))
    for p in parents:
        c = _train_cost(p, seen)
        if not c["chain"]:
            continue
        out["sec"] += c["sec"]
        out["chain"] += c["chain"]
        out["energy_wh"] = None if out["energy_wh"] is None or c["energy_wh"] is None else out["energy_wh"] + c["energy_wh"]
    return out


def _eval_files(run):
    d = os.path.join(RUNS, run)
    try:
        return sorted(f for f in os.listdir(d) if EVAL_FILE.match(f))
    except OSError:
        return []


def _run_summary(name):
    d = os.path.join(RUNS, name)
    cfg = _json(os.path.join(d, "config.json"), {})
    m = _metrics(os.path.join(d, "metrics.jsonl"))
    last = m[-1] if m else {}
    best = max((r.get("bench_optimal") or 0 for r in m), default=None)
    elos = [r["elo"] for r in m if r.get("elo") is not None]
    ev = _json(os.path.join(d, "eval.json"))
    cost = (_json(os.path.join(d, "cost.json")) or {}).get("total")
    bench, _ = _bench_all()
    b = bench.get(name) or {}
    return {"name": name, "phase": cfg.get("phase"), "cost": cost, "created": cfg.get("created"),
            "model": cfg.get("model"), "params": cfg.get("params"), "device": cfg.get("device"),
            "size": _size(cfg), "family": b.get("family") or FAMILY_BY_PHASE.get(cfg.get("phase"), cfg.get("phase")),
            "teacher": b.get("teacher_label"), "sweep": b.get("sweep"), "seed": b.get("seed"),
            "bench": {k: b.get(k) for k in ("policy_optimal", "played_optimal", "value_acc", "play_head")} if b else None,
            "top8": name in _top8_runs(), "eval_files": _eval_files(name),
            "points": len(m), "last": {k: last.get(k) for k in ("epoch", "gen", "total_games", "elapsed_sec",
                                                                   "bench_optimal", "bench_value_acc", "time",
                                                                   "energy_wh")},
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
    cost = _json(os.path.join(d, "cost.json"))
    if cost:
        for seg in cost.get("segments", []):
            seg.pop("timeline", None)
    return jsonify({"name": name, "config": _json(os.path.join(d, "config.json"), {}),
                    "metrics": _metrics(os.path.join(d, "metrics.jsonl")), "eval": ev, "notes": notes,
                    "cost": cost})


def _eval_label(run, f, ev):
    model = (ev or {}).get("model") or ""
    rs = _model_runs(model)
    if len(rs) > 1:
        return "Ensemble of %d (%s)" % (len(rs), ", ".join(rs))
    return run if f == "eval.json" else "%s (%s)" % (run, f[5:-5])


@bp.get("/api/lab/evals")
def evals():
    return jsonify(_evals_data())


def _evals_data():
    """Every nn.evaluate result: runs/<run>/eval.json and runs/<run>/eval_<tag>.json (ensembles)."""
    top = _top8()
    top_models = {}
    for p in top:
        top_models.setdefault("+".join(p["runs"]), []).append(p["bot"])
    out = []
    if os.path.isdir(RUNS):
        for run in sorted(os.listdir(RUNS)):
            if not SAFE.match(run):
                continue
            for f in _eval_files(run):
                ev = _json(os.path.join(RUNS, run, f))
                if not ev:
                    continue
                rs = _model_runs(ev.get("model")) or [run]
                h = ev.get("hypothesis") or {}
                out.append({"run": run, "file": f, "label": _eval_label(run, f, ev), "model": ev.get("model"),
                            "runs": rs, "created": ev.get("created"), "top8": "+".join(rs) in top_models,
                            "arena_bots": top_models.get("+".join(rs), []),
                            "has_hypothesis": bool(h.get("rows")), "verdict": h.get("verdict"),
                            "sections": [k for k in ("accuracy", "tactics", "ladder", "hypothesis") if k in ev]})
    # nets that are in the top-8 arena but have no evaluation yet (the page prints the command)
    have = {"+".join(e["runs"]) for e in out if e["has_hypothesis"]}
    missing, seen = [], set()
    for p in top:
        k = "+".join(p["runs"])
        if k not in have and k not in seen:
            seen.add(k)
            missing.append({"runs": p["runs"], "model": p["model"], "bot": p["bot"], "label": p["label"]})
    out.sort(key=lambda e: (not e["top8"], e["label"]))
    return {"evals": out, "missing": missing, "arena": _top_arena_name()}


@bp.get("/api/lab/eval/<run>/<fname>")
def eval_file(run, fname):
    d = _safe_dir(RUNS, run)
    if not EVAL_FILE.match(fname or ""):
        abort(404)
    ev = _json(os.path.join(d, fname))
    if ev is None:
        abort(404)
    if "ladder" in ev:
        ev["ladder"] = {k: v for k, v in ev["ladder"].items() if k != "games"}
    return jsonify(ev)


def _ensemble_bench(runs):
    """experiments/results/score/*.json written by nn.eval_score for exactly these runs -> ENSEMBLE row."""
    d = os.path.join(RESULTS, "score")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        j = _cached_json(os.path.join(d, f)) if f.endswith(".json") else None
        if j and sorted(j.get("runs") or []) == sorted(runs) and "ENSEMBLE" in (j.get("results") or {}):
            e = j["results"]["ENSEMBLE"]
            best = max(e.items(), key=lambda kv: kv[1].get("optimal") or 0)
            return {"file": "experiments/results/score/" + f, "policy": e.get("policy"),
                    "best_head": best[0], "best": best[1], "heads": e}
    return None


@bp.get("/api/lab/compare")
def compare():
    """Everything the 'Method comparison' tab needs, joined per net:
    arena (top-8 run: Elo, score, vs Pons, ms/move), the 3,000-position benchmark of every net
    (research/bench_all.json), training time incl. init/teacher runs, nn.evaluate hypothesis
    verdicts, and the minimax-ladder comparison (methods/top8 if it was run, else methods/main)."""
    bench, bmeta = _bench_all()
    aname = _top_arena_name()
    s = _cached_json(os.path.join(ARENA, aname, "summary.json")) if aname else None
    players = []
    if s:
        info, P, E, W = s.get("bot_info", {}), s.get("per_bot", {}), s.get("elo", {}), s.get("wdl", {})
        pons = next((b for b in s["bots"] if b.startswith("pons")), None)
        top = {p["bot"]: p for p in _top8()}
        for bot in s["bots"]:
            inf, p = info.get(bot, {}), P.get(bot, {})
            t = top.get(bot) or {"runs": [], "mode": None, "model": None}
            vp = (W.get(bot) or {}).get(pons) if pons and bot != pons else None
            seen = set()
            cost = {"sec": 0, "chain": [], "energy_wh": 0}
            for r in t["runs"]:
                c = _train_cost(r, seen)
                cost["sec"] += c["sec"]
                cost["chain"] += c["chain"]
                cost["energy_wh"] = None if cost["energy_wh"] is None or c["energy_wh"] is None else cost["energy_wh"] + c["energy_wh"]
            if not t["runs"]:
                cost = None
            cfg = _json(os.path.join(RUNS, t["runs"][0], "config.json"), {}) if t["runs"] else {}
            bn = [bench.get(r) for r in t["runs"] if bench.get(r)]
            ens = _ensemble_bench(t["runs"]) if len(t["runs"]) > 1 else None
            players.append({
                "bot": bot, "label": inf.get("label") or bot, "group": inf.get("group"), "runs": t["runs"],
                "mode": t["mode"], "model": t["model"], "phase": cfg.get("phase"),
                "family": (bn[0]["family"] if bn else FAMILY_BY_PHASE.get(cfg.get("phase"))) if t["runs"] else None,
                "size": _size(cfg) if t["runs"] else None, "params": cfg.get("params"),
                "elo": E.get(bot), "score": p.get("score"), "wdl": [p.get("wins"), p.get("draws"), p.get("losses")],
                "avg_ms": p.get("avg_ms"), "max_ms": p.get("max_ms"),
                "vs_pons": ({"w": vp[0], "d": vp[1], "l": vp[2], "score": (vp[0] + 0.5 * vp[1]) / max(1, sum(vp))}
                            if vp else None),
                "bench": bn, "ensemble_bench": ens, "cost": cost})
    nets = []
    for r, n in bench.items():
        cfg = _json(os.path.join(RUNS, r, "config.json"), {})
        c = _train_cost(r)
        nets.append({k: n.get(k) for k in ("run", "family", "teacher_label", "sweep", "blocks", "channels", "params",
                                             "seed", "policy_optimal", "played_optimal", "policy_keeps",
                                             "value_acc", "value_acc_W", "value_acc_D", "value_acc_L",
                                             "policy_optimal_by_ply", "play_head")})
        nets[-1]["phase"] = cfg.get("phase")
        nets[-1]["train_sec"] = c["sec"] or None
        nets[-1]["top8"] = r in _top8_runs()
    evs = {}
    for e in _evals_data()["evals"]:
        if e["has_hypothesis"]:
            ev = _json(os.path.join(RUNS, e["run"], e["file"]), {})
            evs["+".join(e["runs"])] = {"file": e["run"] + "/" + e["file"], "verdict": e["verdict"],
                                        "rows": [{"depth": h["depth"], "score": h["head_to_head"]["score"],
                                                  "supported": h["supported"], "nn_ms": h.get("nn_ms"),
                                                  "nn_optimal": h["nn_accuracy"]["optimal"],
                                                  "hand_optimal": h["hand_accuracy"]["optimal"]}
                                                 for h in ev.get("hypothesis", {}).get("rows", [])],
                                        "ladder_elo": (ev.get("ladder") or {}).get("elo")}
    lad_name = "top8" if os.path.isfile(os.path.join(METHODS, "top8", "summary.json")) else "main"
    lad = _json(os.path.join(METHODS, lad_name, "summary.json"), {}) or {}
    ladder = [{k: e.get(k) for k in ("label", "method", "run", "search", "spec", "elo", "optimal", "keeps_nl",
                                      "ms_per_move", "params")} for e in lad.get("entries", [])]
    return jsonify({"arena": {"name": aname, "games": (s or {}).get("games"), "meta": {k: (s or {}).get("meta", {}).get(k) for k in ("created", "seconds", "workers")},
                              "openings": len((s or {}).get("meta", {}).get("openings") or []) or (s or {}).get("meta", {}).get("openings"),
                              "anchor": (s or {}).get("elo_anchor")},
                    "players": players, "nets": nets, "bench_meta": bmeta, "evals": evs,
                    "ladder": {"name": lad_name, "created": lad.get("created"), "settings": lad.get("settings"),
                               "entries": ladder}})


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


@bp.get("/api/lab/methods")
def methods():
    out = []
    if os.path.isdir(METHODS):
        for name in sorted(os.listdir(METHODS)):
            s = _json(os.path.join(METHODS, name, "summary.json")) if SAFE.match(name) else None
            if s:
                out.append({"name": name, "created": s.get("created"), "entries": len(s.get("entries", []))})
    out.sort(key=lambda r: r.get("created") or "", reverse=True)
    out.sort(key=lambda r: r["name"] != "main")
    return jsonify(out)


@bp.get("/api/lab/methods/<name>")
def method_summary(name):
    d = _safe_dir(METHODS, name)
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
    if not isinstance(run_name, str) or not SAFE.match(run_name) or run_name.startswith("."):
        raise ValueError("bad run name")
    p = os.path.join(RUNS, run_name, "model.npz")
    if not os.path.isfile(p):
        raise ValueError("no model for run %r" % run_name)
    key = (p, os.path.getmtime(p))
    if _NETS.get(run_name, (None,))[0] != key:
        from nn.numpy_net import NumpyNet
        _NETS[run_name] = (key, NumpyNet(p))
    return _NETS[run_name][1]
