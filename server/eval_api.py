"""Live evaluation bar for the play page (chess-style).

POST /api/eval   {"moves": "4453", "source": "nn:sup_6x64" | "solver"}
GET  /api/eval/sources                 which evaluators this machine can run

The position is given as its move history (1-indexed columns, like the Pons solver),
so the reply can also say how fast a forced result is.  Everything is reported from
RED's point of view (red = +1 = the player who moved first), so the bar does not flip
when the side to move changes:

  bar     0..1, share of the bar that is red (0.5 = level)
  label   short text on the bar: "+0.42" for a net, "R5" / "Y5" (red / yellow forces a win in 5 of its own moves) / "0.0" for the solver
  text    one-line description
  value   -1..1 for red (net: P(win) - P(loss); solver: +1 / 0 / -1)
  wdl     [win, draw, loss] for red (net only)
  columns solver only: the 7 Pons scores for the side to move (None = full column),
          so the page can grade the move that is played next

Net evaluation averages the position and its mirror image (measured +1-1.5 points of
accuracy on the 3,000-position benchmark, docs/RESEARCH.md).  The solver is used only
when tools/pons/c4solver(.exe) exists; a position it cannot solve in SOLVER_TIMEOUT
seconds falls back to the default net.
"""

import os
import threading

import numpy as np
from flask import Blueprint, jsonify, request

from engine.board import Board, COLS
from server.lab_api import RUNS, SAFE, nn_evaluator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PONS = os.path.join(ROOT, "tools", "pons")
SOLVER_TIMEOUT = 4.0
# strongest policy+value nets first; the first one that exists is the default
PREFERRED_NETS = ("rs_A_pons_8x128", "rs_A_pons_10x128", "sup_6x64", "rs_A_pons_6x64")

bp = Blueprint("eval", __name__)
_solver = None
_solver_lock = threading.Lock()
_solver_failed = False


def _solver_binary():
    for name in ("c4solver", "c4solver.exe"):
        p = os.path.join(PONS, name)
        if os.path.isfile(p):
            return p
    return None


def get_solver():
    """Shared Solver (analyze mode) or None when the binary is missing / cannot start."""
    global _solver, _solver_failed
    if _solver is not None or _solver_failed:
        return _solver
    with _solver_lock:
        if _solver is None and not _solver_failed:
            b = _solver_binary()
            if not b:
                _solver_failed = True
                return None
            try:
                from tools.solver import Solver
                _solver = Solver(binary=b, timeout=SOLVER_TIMEOUT, analyze=True)
            except Exception:
                _solver_failed = True
    return _solver


def net_runs():
    if not os.path.isdir(RUNS):
        return []
    have = [n for n in sorted(os.listdir(RUNS))
            if SAFE.match(n) and os.path.isfile(os.path.join(RUNS, n, "model.npz"))]
    first = [n for n in PREFERRED_NETS if n in have]
    return first + [n for n in have if n not in first]


def board_from_moves(moves):
    if not isinstance(moves, str) or len(moves) > 42 or any(ch not in "1234567" for ch in moves):
        raise ValueError("moves must be a string of column digits 1-7 (at most 42)")
    b = Board()
    for i, ch in enumerate(moves):
        if b.is_terminal():
            raise ValueError("game was already over before move %d" % (i + 1))
        c = int(ch) - 1
        if c not in b.legal_moves():
            raise ValueError("column %s is full at move %d" % (ch, i + 1))
        b.drop(c)
    return b


def _terminal(board):
    w = board.check_win()
    if w:
        return {"bar": 1.0 if w == 1 else 0.0, "value": float(w), "label": "1-0" if w == 1 else "0-1",
                "text": ("Red" if w == 1 else "Yellow") + " has won", "result": "red" if w == 1 else "yellow"}
    return {"bar": 0.5, "value": 0.0, "label": "½-½", "text": "Draw: the board is full", "result": "draw"}


def eval_net(run, board):
    from nn.encode import mirror_bits, to_u64
    ev = nn_evaluator(run)
    cur, mask = board.current_bits(), board.mask
    curs = np.concatenate([to_u64([cur]), mirror_bits(to_u64([cur]))])
    masks = np.concatenate([to_u64([mask]), mirror_bits(to_u64([mask]))])
    _, _, wdl = ev.evaluate(curs, masks)
    w, d, l = (float(x) for x in wdl.mean(axis=0))           # for the side to move
    if board.to_move != 1:                                    # report for red
        w, l = l, w
    v = w - l
    pct = round(100 * (w + 0.5 * d))
    return {"bar": 0.5 + 0.5 * v, "value": v, "wdl": [round(w, 4), round(d, 4), round(l, 4)],
            "label": "%+.2f" % v if abs(v) >= 0.005 else "0.00",
            "text": "%s thinks red scores about %d%% (win %d%%, draw %d%%, loss %d%%)"
                    % (run, pct, round(100 * w), round(100 * d), round(100 * l))}


def eval_solver(solver, moves, board):
    cols = solver.solve_all(moves)
    s = max(x for x in cols if x is not None)                 # side to move, Pons convention
    red = s if board.to_move == 1 else -s
    n = len(moves)
    out = {"columns": cols, "score": red}
    if red == 0:
        out.update(bar=0.5, value=0.0, label="0.0", text="Perfect play from here is a draw")
        return out
    # the winner has 22 - |s| stones on the board when the game ends (tools/solver.py)
    stones = 22 - abs(s)
    if red > 0:                                               # red moves on odd plies
        plies = (2 * stones - 1) - n
        winner_moves = stones - (n + 1) // 2
    else:
        plies = 2 * stones - n
        winner_moves = stones - n // 2
    who = "Red" if red > 0 else "Yellow"
    # any forced win fills most of the bar; faster wins fill more (like a mate score)
    frac = 0.30 + 0.20 * min(abs(s), 21) / 21.0
    out.update(bar=0.5 + (frac if red > 0 else -frac), value=1.0 if red > 0 else -1.0,
               label=("R" if red > 0 else "Y") + str(winner_moves),
               text="%s wins with perfect play in %d more move%s of their own (%d pl%s)"
                    % (who, winner_moves, "" if winner_moves == 1 else "s", plies, "y" if plies == 1 else "ies"))
    return out


@bp.get("/api/eval/sources")
def eval_sources():
    nets = net_runs()
    return jsonify({"solver": bool(_solver_binary()), "nets": nets, "default": "nn:" + nets[0] if nets else
                    ("solver" if _solver_binary() else None)})


@bp.post("/api/eval")
def api_eval():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "expected JSON object"}), 400
    moves = data.get("moves", "")
    try:
        board = board_from_moves(moves)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    src = data.get("source") or ""
    base = {"ply": len(moves), "to_move": board.to_move}
    if board.is_terminal():
        return jsonify(dict(base, source=src or "rules", **_terminal(board)))
    nets = net_runs()
    note = None
    if src == "solver":
        solver = get_solver()
        if solver is not None:
            try:
                return jsonify(dict(base, source="solver", **eval_solver(solver, moves, board)))
            except Exception as e:                            # timeout / crash: fall back to a net
                note = "solver failed (%s); showing the net instead" % str(e)[:80]
        else:
            note = "solver binary not found on this machine; showing the net instead"
        src = "nn:" + nets[0] if nets else ""
    if not src.startswith("nn:"):
        src = "nn:" + nets[0] if nets else ""
    run = src[3:]
    if not run:
        return jsonify({"error": "no evaluator available (no runs/*/model.npz and no solver)"}), 503
    try:
        out = eval_net(run, board)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if note:
        out["note"] = note
    return jsonify(dict(base, source="nn:" + run, **out))
