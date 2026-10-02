# Shared engine API contract (the website, tests and experiments all code against this)

Code must run on **Python 3.10+** — no 3.11+ syntax/features. Pure Python for engine (no numpy needed in engine).
Project layout mirrors the repo layout in connect4-project-statement.md.

## engine/board.py
- ROWS=6, COLS=7. Board representation for the API/JSON: list of 6 lists of 7 ints, row 0 = TOP row, values +1 / -1 / 0.
  +1 = player who moved first in that game ("P1"), -1 = second player.
- class Board:
  - Board() empty; Board.from_grid(grid, to_move=None) (to_move inferred from piece counts if None: +1 if counts equal else -1)
  - .to_grid() -> list[list[int]]; .to_move -> +1/-1; .moves_played -> int
  - .legal_moves() -> list[int] (columns not full)
  - .drop(col) mutates (raises ValueError if illegal); .undo() reverts last drop
  - .check_win() -> +1/-1/0 (winner of current position), .is_draw() -> bool (full, no winner), .is_terminal()
  - .winning_cells() -> list[(row,col)] of a winning line or []
  - .copy(); .key() hashable position key; Board.from_moves("4453") convenience (1-indexed column chars, Pons format)
  - internal representation free (bitboard recommended for speed).

## engine/heuristic.py
- evaluate(board, player) -> float, score from `player`'s perspective (window scoring per statement).

## engine/minimax.py
- negamax search: search(board, depth, time_limit=None, use_tt=True, ordering=True) -> SearchResult
  SearchResult fields: col:int, score:float, nodes:int, ms:float, depth_reached:int
  Score from the side-to-move perspective; wins scored as large (e.g. 1e6 - ply) so faster wins preferred.

## engine/agents.py
- RandomAgent(seed=None).choose(board) -> int
- MinimaxAgent(depth, epsilon=0.0, time_limit=None, seed=None).choose(board) -> int; .last_result -> SearchResult (None-ish/defaults if random move)
- SolverAgent (optional; wraps a Pons solver binary if present at tools/pons/c4solver)
- LEVELS = {1:(1,0.5),2:(1,0.2),3:(2,0.1),4:(3,0),5:(4,0),6:(5,0),7:(6,0),8:(7,0),9:(8,0),10:(20,0)} with level 10 time_limit≈2.0s (iterative deepening)
- agent_for_level(level, seed=None) -> MinimaxAgent

## server/app.py (Flask)
- GET / -> web/index.html (static files from web/)
- POST /api/move JSON {board: grid, level: int} -> {col, score, nodes, ms, depth, random: bool} ; error 400 on bad board / game over
- Run: `python server/app.py` -> http://localhost:5000
