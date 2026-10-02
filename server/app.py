"""Flask server for the Connect 4 bot.  Run: python server/app.py (from root or server/)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

from engine.board import Board, ROWS, COLS  # noqa: E402
from engine.agents import agent_for_level  # noqa: E402

WEB_DIR = os.path.join(ROOT, "web")
app = Flask(__name__, static_folder=None)


class BadRequest(Exception):
    pass


def parse_board(grid):
    """Validate a grid and return an engine Board. Raises BadRequest."""
    if not isinstance(grid, list) or len(grid) != ROWS:
        raise BadRequest("board must be a list of %d rows" % ROWS)
    for row in grid:
        if not isinstance(row, list) or len(row) != COLS:
            raise BadRequest("each row must have %d cells" % COLS)
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int) or v not in (-1, 0, 1):
                raise BadRequest("cells must be -1, 0 or 1")
    n1 = sum(v == 1 for row in grid for v in row)
    n2 = sum(v == -1 for row in grid for v in row)
    if n1 not in (n2, n2 + 1):
        raise BadRequest("illegal piece counts (+1 moves first, so counts must be equal or +1 one ahead)")
    for c in range(COLS):
        seen_empty = False
        for r in range(ROWS - 1, -1, -1):  # bottom to top
            if grid[r][c] == 0:
                seen_empty = True
            elif seen_empty:
                raise BadRequest("floating piece in column %d" % (c + 1))
    board = Board.from_grid(grid)
    if board.is_terminal():
        raise BadRequest("game is already over")
    return board


def status_of(board):
    w = board.check_win()
    if w:
        return {"over": True, "winner": w, "draw": False,
                "cells": [list(x) for x in board.winning_cells()]}
    if board.is_draw():
        return {"over": True, "winner": 0, "draw": True, "cells": []}
    return {"over": False, "winner": 0, "draw": False, "cells": []}


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:name>")
def static_files(name):
    if name in ("app.js", "style.css"):
        return send_from_directory(WEB_DIR, name)
    return jsonify({"error": "not found"}), 404


@app.post("/api/move")
def api_move():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "expected JSON object"}), 400
    level = data.get("level", 5)
    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 10:
        return jsonify({"error": "level must be an integer 1-10"}), 400
    try:
        board = parse_board(data.get("board"))
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400

    agent = agent_for_level(level)
    col = agent.choose(board)
    res = getattr(agent, "last_result", None)
    is_random = bool(getattr(res, "random", False))
    after = board.copy()
    mover = after.to_move
    after.drop(col)
    out = {
        "col": int(col),
        "score": float(getattr(res, "score", 0) or 0),
        "nodes": int(getattr(res, "nodes", 0) or 0),
        "ms": float(getattr(res, "ms", 0) or 0),
        "depth": int(getattr(res, "depth_reached", 0) or 0),
        "random": is_random,
        "player": mover,
        "status": status_of(after),
    }
    return jsonify(out)


@app.post("/api/status")
def api_status():
    data = request.get_json(silent=True) or {}
    grid = data.get("board")
    try:
        # same validation, but finished games are allowed here
        if not isinstance(grid, list) or len(grid) != ROWS or any(
                not isinstance(r, list) or len(r) != COLS for r in grid):
            raise BadRequest("bad board shape")
        board = Board.from_grid(grid)
    except (BadRequest, ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(status_of(board))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
