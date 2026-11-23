"""MCP server for the Connect 4 project (local only, git-ignored).

Gives an MCP client (Claude Code, Claude Desktop, etc.) read access to the
project's runs, arena results, research results and docs, plus live tools:
run a trained net on a position, ask the Pons solver, ask minimax, and score
a net on the 3,000-position benchmark.

Run (stdio):
    .venv/bin/python mcp_server/server.py

Moves are always a string of 1-indexed columns, e.g. "4453" (same format as
the solver and data/accuracy_positions.csv). Columns in outputs are 0-indexed
unless the field name says "_1idx".
"""

import contextlib
import glob
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
# one BLAS thread: same fix as the arena hang
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("connect4-project")

RUNS = os.path.join(ROOT, "runs")
DOCS = os.path.join(ROOT, "docs")
RESULTS = os.path.join(ROOT, "experiments", "results")
_evaluators = {}
_solver = None


# ------------------------------------------------------------- helpers
@contextlib.contextmanager
def _quiet():
    """Keep project prints (nn.common.log) off stdout: stdout carries the MCP protocol."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_name(name):
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("bad name: %r" % name)
    return name


def _board(moves):
    from engine.board import Board
    moves = "".join(str(m) for m in (moves or "")).strip()
    if moves and not moves.isdigit():
        raise ValueError("moves must be digits 1-7, e.g. '4453'")
    return Board.from_moves(moves), moves


def _grid_text(board):
    sym = {1: "X", -1: "O", 0: "."}
    rows = [" ".join(sym[int(v)] for v in row) for row in board.to_grid()]
    return "\n".join(rows + ["1 2 3 4 5 6 7"])


def _model_path(run, model):
    run = _safe_name(run)
    p = os.path.join(RUNS, run, _safe_name(model))
    if not os.path.isfile(p):
        raise ValueError("no %s in runs/%s (see list_runs)" % (model, run))
    return p


def _evaluator(run, model="model.npz"):
    key = (run, model)
    if key not in _evaluators:
        from nn.common import load_evaluator
        with _quiet():
            _evaluators[key] = load_evaluator(_model_path(run, model), "auto")
    return _evaluators[key]


def _solver_obj():
    global _solver
    if _solver is None:
        from tools.solver import Solver
        _solver = Solver(timeout=30)
    return _solver


def _eval_net(ev, curs, masks, mirror):
    import numpy as np
    from nn.encode import mirror_bits
    pri, val, wdl = ev.evaluate(curs, masks)
    if mirror:
        mc = mirror_bits(np.array(curs, dtype=np.uint64))
        mm = mirror_bits(np.array(masks, dtype=np.uint64))
        p2, _, w2 = ev.evaluate(mc, mm)
        pri = (pri + p2[:, ::-1]) / 2
        wdl = (wdl + w2) / 2
        val = wdl[:, 0] - wdl[:, 2]
    return pri, val, wdl


def _sign(x):
    return (x > 0) - (x < 0)


# --------------------------------------------------------------- tools
@mcp.tool()
def list_runs() -> list:
    """List every training run in runs/ with phase, size, params and its
    final benchmark accuracy (policy alone, and MCTS if eval.json exists)."""
    out = []
    for d in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, d)
        if not os.path.isdir(p) or d.startswith("_"):
            continue
        row = {"run": d, "models": sorted(f for f in os.listdir(p) if f.endswith((".npz", ".pt")))}
        cfg_p = os.path.join(p, "config.json")
        if os.path.isfile(cfg_p):
            cfg = _read_json(cfg_p)
            row.update({"phase": cfg.get("phase"), "model": cfg.get("model"),
                        "params": cfg.get("params"), "data": (cfg.get("args") or {}).get("data"),
                        "train_positions": cfg.get("train_positions")})
        met_p = os.path.join(p, "metrics.jsonl")
        if os.path.isfile(met_p):
            last = None
            with open(met_p, encoding="utf-8") as f:
                for line in f:
                    if '"bench_optimal"' in line:
                        last = line
            if last:
                m = json.loads(last)
                row["last_epoch"] = m.get("epoch")
                row["bench_optimal"] = m.get("bench_optimal")
                row["bench_keeps_nl"] = m.get("bench_keeps_nl")
        ev_p = os.path.join(p, "eval.json")
        if os.path.isfile(ev_p):
            acc = _read_json(ev_p).get("accuracy", {})
            row["eval_optimal"] = {k: v.get("optimal") for k, v in acc.items()
                                   if isinstance(v, dict) and "optimal" in v}
        out.append(row)
    return out


@mcp.tool()
def run_metrics(run: str, last_n: int = 5) -> dict:
    """config.json plus the last `last_n` epoch records of runs/<run>/metrics.jsonl
    (train/val losses, val accuracy, benchmark accuracy per ply bucket)."""
    p = os.path.join(RUNS, _safe_name(run))
    if not os.path.isdir(p):
        raise ValueError("unknown run %r" % run)
    out = {"run": run}
    if os.path.isfile(os.path.join(p, "config.json")):
        cfg = _read_json(os.path.join(p, "config.json"))
        cfg.pop("machine", None)
        out["config"] = cfg
    recs = []
    mp_ = os.path.join(p, "metrics.jsonl")
    if os.path.isfile(mp_):
        with open(mp_, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
    out["n_records"] = len(recs)
    out["last"] = recs[-max(1, int(last_n)):]
    return out


@mcp.tool()
def run_eval(run: str) -> dict:
    """Full runs/<run>/eval.json (accuracy, value head, tactics, ladder, hypothesis)."""
    p = os.path.join(RUNS, _safe_name(run), "eval.json")
    if not os.path.isfile(p):
        raise ValueError("runs/%s has no eval.json (python -m nn.evaluate --run %s)" % (run, run))
    return _read_json(p)


@mcp.tool()
def solve_position(moves: str = "") -> dict:
    """Pascal Pons' exact solver on a position: score of each column
    (positive = side to move wins; larger = faster win; None = full column),
    the optimal columns (0-indexed) and the game-theoretic result."""
    b, mv = _board(moves)
    if b.is_terminal():
        return {"moves": mv, "terminal": True, "winner": b.check_win()}
    t0 = time.perf_counter()
    sc = _solver_obj().solve_all(mv)
    legal = [c for c in range(7) if sc[c] is not None]
    best = max(sc[c] for c in legal)
    return {"moves": mv, "scores": sc, "best": best,
            "optimal_cols": [c for c in legal if sc[c] == best],
            "result": "win" if best > 0 else "draw" if best == 0 else "loss",
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


@mcp.tool()
def analyze_position(moves: str = "", run: str = "sup_6x64", model: str = "model.npz",
                     mirror: bool = True, with_solver: bool = True) -> dict:
    """Run a trained net on a position: policy over columns, WDL value, its move.
    mirror=True averages the net with its left-right mirror (test-time augmentation).
    with_solver=True also returns the Pons solver's scores, so you can see if the net is right."""
    b, mv = _board(moves)
    if b.is_terminal():
        return {"moves": mv, "board": _grid_text(b), "terminal": True, "winner": b.check_win()}
    ev = _evaluator(run, model)
    pri, val, wdl = _eval_net(ev, [b.current_bits()], [b.mask], mirror)
    legal = b.legal_moves()
    col = max(legal, key=lambda c: pri[0][c])
    out = {"moves": mv, "ply": len(mv), "to_move": "X (first)" if b.to_move == 1 else "O (second)",
           "board": _grid_text(b), "run": run, "mirror": mirror,
           "policy": [round(float(x), 4) for x in pri[0]],
           "wdl": {"win": round(float(wdl[0][0]), 4), "draw": round(float(wdl[0][1]), 4),
                   "loss": round(float(wdl[0][2]), 4)},
           "value": round(float(val[0]), 4), "net_move": col, "net_move_1idx": col + 1}
    if with_solver:
        s = solve_position(mv)
        out["solver"] = s
        out["net_move_optimal"] = col in s["optimal_cols"]
        out["net_move_keeps_result"] = s["scores"][col] is not None and \
            _sign(s["scores"][col]) == _sign(s["best"])
    return out


@mcp.tool()
def minimax_move(moves: str = "", depth: int = 4) -> dict:
    """The project's own negamax/alpha-beta (hand-crafted heuristic) at `depth` (1-12)."""
    from engine.agents import MinimaxAgent
    b, mv = _board(moves)
    if b.is_terminal():
        return {"moves": mv, "terminal": True}
    depth = max(1, min(int(depth), 12))
    ag = MinimaxAgent(depth)
    t0 = time.perf_counter()
    with _quiet():
        col = ag.choose(b)
    r = ag.last_result
    return {"moves": mv, "depth": depth, "col": col, "col_1idx": col + 1,
            "score": getattr(r, "score", None), "nodes": getattr(r, "nodes", None),
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


@mcp.tool()
def bench_accuracy(run: str = "sup_6x64", model: str = "model.npz", mirror: bool = False,
                   limit: int = 0) -> dict:
    """Score a net's policy head (no search) on the 3,000-position held-out benchmark
    (data/accuracy_positions.csv). 'optimal' = picked one of the solver's best columns;
    'keeps_nl' = in non-lost positions, kept the game result. mirror=True adds
    left-right test-time augmentation. limit>0 uses only the first N positions."""
    from nn import bench
    rows = bench.load_accuracy_bench(limit=limit or None)
    ev = _evaluator(run, model)
    t0 = time.time()
    cols = []
    for i in range(0, len(rows), 512):
        ch = rows[i:i + 512]
        pri, _, _ = _eval_net(ev, [r["cur"] for r in ch], [r["mask"] for r in ch], mirror)
        for r, p in zip(ch, pri):
            legal = [c for c in range(7) if r["scores"][c] is not None]
            cols.append(max(legal, key=lambda c: p[c]))
    res = bench.accuracy_summary(rows, cols)
    res.update({"run": run, "model": model, "mirror": mirror, "seconds": round(time.time() - t0, 1)})
    return res


@mcp.tool()
def arena_summary(name: str = "top15", top: int = 30) -> dict:
    """Elo table and per-bot record from experiments/results/arena/<name>/summary.json
    (names: main, part3, top10, top15, ... see list_results)."""
    p = os.path.join(RESULTS, "arena", _safe_name(name), "summary.json")
    if not os.path.isfile(p):
        raise ValueError("no arena run %r" % name)
    s = _read_json(p)
    elo = sorted(s.get("elo", {}).items(), key=lambda kv: -kv[1]["elo"])[:max(1, int(top))]
    per = s.get("per_bot", {})
    table = [{"bot": k, **v, "score": per.get(k, {}).get("score"), "games": per.get(k, {}).get("games"),
              "avg_ms": per.get(k, {}).get("avg_ms")} for k, v in elo]
    return {"name": name, "games": s.get("games"), "forfeits": len(s.get("forfeits") or []),
            "elo_anchor": "%s=%s" % (s.get("elo_anchor"), s.get("elo_anchor_value")),
            "first_player_score": s.get("first_player_score"), "table": table,
            "meta": {k: v for k, v in (s.get("meta") or {}).items() if k != "openings"}}


@mcp.tool()
def head_to_head(bot_a: str, bot_b: str, name: str = "top15") -> dict:
    """Wins/draws/losses of bot_a against bot_b in an arena run."""
    p = os.path.join(RESULTS, "arena", _safe_name(name), "summary.json")
    if not os.path.isfile(p):
        raise ValueError("no arena run %r" % name)
    wdl = _read_json(p).get("wdl", {}).get(bot_a, {}).get(bot_b)
    if wdl is None:
        raise ValueError("no record for %s vs %s in %s" % (bot_a, bot_b, name))
    w, d, l = wdl
    return {"bot_a": bot_a, "bot_b": bot_b, "arena": name, "wins": w, "draws": d, "losses": l,
            "score": round((w + 0.5 * d) / max(1, w + d + l), 4)}


@mcp.tool()
def list_results() -> dict:
    """Result files under experiments/results (arena runs, research json, csv summaries)."""
    files = sorted(os.path.relpath(p, RESULTS).replace("\\", "/")
                   for p in glob.glob(os.path.join(RESULTS, "**", "*"), recursive=True)
                   if os.path.isfile(p) and not p.endswith(".old"))
    return {"arena_runs": sorted(os.listdir(os.path.join(RESULTS, "arena"))), "files": files}


@mcp.tool()
def read_result(path: str, max_chars: int = 20000) -> str:
    """Read a file under experiments/results (path relative to it, e.g.
    'research/sweep.json' or 'elo.json'). Truncated to max_chars."""
    full = os.path.realpath(os.path.join(RESULTS, path))
    if not full.startswith(os.path.realpath(RESULTS) + os.sep) or not os.path.isfile(full):
        raise ValueError("not a file under experiments/results: %r" % path)
    with open(full, encoding="utf-8", errors="replace") as f:
        txt = f.read(max_chars + 1)
    return txt if len(txt) <= max_chars else txt[:max_chars] + "\n...[truncated]"


@mcp.tool()
def list_docs() -> list:
    """Markdown docs in docs/ and the repo root (root ones are prefixed '../')."""
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(DOCS, "*.md"))] +
                  ["../" + os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "*.md"))])


@mcp.tool()
def read_doc(name: str, max_chars: int = 40000) -> str:
    """Read a doc from docs/ (e.g. 'RESEARCH.md') or the repo root ('../README.md')."""
    full = os.path.realpath(os.path.join(DOCS, name))
    if not full.startswith(os.path.realpath(ROOT) + os.sep) or not full.endswith(".md") \
            or not os.path.isfile(full):
        raise ValueError("unknown doc %r (see list_docs)" % name)
    with open(full, encoding="utf-8", errors="replace") as f:
        txt = f.read(max_chars + 1)
    return txt if len(txt) <= max_chars else txt[:max_chars] + "\n...[truncated]"


@mcp.resource("connect4://research-notes")
def research_notes() -> str:
    """docs/RESEARCH.md"""
    return read_doc("RESEARCH.md")


if __name__ == "__main__":
    mcp.run()
