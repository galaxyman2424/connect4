"""Shared helpers for the experiment scripts (Python 3.10+).

Every script in experiments/ is run from the project root, e.g.
``python experiments/tournament.py``.  This module puts the project root on
sys.path so ``engine`` and ``tools`` import, and provides:

* paths (RESULTS, FIGURES, DATA)
* agent specs: "random", "d<n>" (fixed-depth minimax), "L<n>" (slider level)
* random openings (2-4 plies) and a single game runner
* a multiprocessing map that also works on Windows (spawn)
* data loaders for the solver-labeled sets
"""

import csv
import json
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.agents import LEVELS, LEVEL_TIME_LIMITS, MinimaxAgent, RandomAgent, agent_for_level  # noqa: E402
from engine.board import COLS, Board  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")
DATA = os.path.join(ROOT, "data")
SOLVER_BIN = os.path.join(ROOT, "tools", "pons", "c4solver")


def ensure_dirs():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)


def default_workers():
    return max(1, os.cpu_count() or 1)


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ agents
def level_equivalent(spec):
    """Slider levels 4-9 are plain fixed-depth bots (epsilon 0, no time limit):
    L4 == d3 ... L9 == d8.  Returns the canonical spec so identical bots share
    game results."""
    if spec.startswith("L"):
        lvl = int(spec[1:])
        depth, eps = LEVELS[lvl]
        if eps == 0 and lvl not in LEVEL_TIME_LIMITS:
            return "d%d" % depth
    return spec


def make_agent(spec, seed=None):
    if spec == "random":
        return RandomAgent(seed=seed)
    if spec.startswith("d"):
        return MinimaxAgent(int(spec[1:]), seed=seed)
    if spec.startswith("L"):
        return agent_for_level(int(spec[1:]), seed=seed)
    raise ValueError("unknown agent spec %r" % spec)


def spec_sort_key(spec):
    if spec == "random":
        return (0, 0)
    return ({"d": 1, "L": 2}[spec[0]], int(spec[1:]))


# ---------------------------------------------------------------- openings
def mirror_moves(moves):
    return "".join(str(8 - int(c)) for c in moves)


def canonical_key(board):
    """Position key invariant under left-right mirroring."""
    g = board.to_grid()
    a = tuple(tuple(r) for r in g)
    b = tuple(tuple(r[::-1]) for r in g)
    return min(a, b)


def random_openings(n, seed, min_len=2, max_len=4):
    """``n`` distinct random openings of min_len..max_len plies (Pons move
    strings).  Distinct = different position up to left-right mirroring.
    With <= 4 plies each side has at most 2 stones, so no opening can
    contain a win or an immediate threat; no tactical filtering is needed."""
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < n:
        length = rng.randint(min_len, max_len)
        b = Board()
        for _ in range(length):
            b.drop(rng.choice(b.legal_moves()))
        k = canonical_key(b)
        if k in seen:
            continue
        seen.add(k)
        out.append(b.to_move_string())
    return out


# ------------------------------------------------------------------ games
def play_game(a_spec, b_spec, opening, a_first, seed):
    """Play one game.  ``a_first``: agent A plays P1 (+1, the colour that
    made the first opening move).  Returns a dict of per-game results."""
    board = Board.from_moves(opening)
    sa = random.Random("%s|A|%s" % (seed, a_spec)).randrange(2 ** 31)
    sb = random.Random("%s|B|%s" % (seed, b_spec)).randrange(2 ** 31)
    agents = {1 if a_first else -1: ("a", make_agent(a_spec, sa)),
              -1 if a_first else 1: ("b", make_agent(b_spec, sb))}
    stats = {"a": [], "b": []}
    moves = list(opening)
    winner = 0
    while True:
        side, agent = agents[board.to_move]
        col = agent.choose(board)
        r = agent.last_result
        stats[side].append((r.ms, r.nodes, r.depth_reached, bool(r.random)))
        board.drop(col)
        moves.append(str(col + 1))
        w = board.check_win()
        if w:
            winner = w
            break
        if board.is_draw():
            break
    if winner == 0:
        win_name = "draw"
    else:
        win_name = agents[winner][0]
    row = {"a": a_spec, "b": b_spec, "opening": opening,
           "first": a_spec if a_first else b_spec,
           "winner": {"a": a_spec, "b": b_spec, "draw": "draw"}[win_name],
           "score_a": {"a": 1.0, "b": 0.0, "draw": 0.5}[win_name],
           "plies": board.moves_played, "moves": "".join(moves)}
    for side in ("a", "b"):
        st = [s for s in stats[side] if not s[3]]          # searched moves only
        n_all = len(stats[side])
        row["%s_moves" % side] = n_all
        row["%s_random_moves" % side] = n_all - len(st)
        row["%s_ms" % side] = round(sum(s[0] for s in st) / len(st), 3) if st else 0.0
        row["%s_max_ms" % side] = round(max(s[0] for s in st), 3) if st else 0.0
        row["%s_nodes" % side] = round(sum(s[1] for s in st) / len(st), 1) if st else 0.0
        row["%s_depth" % side] = round(sum(s[2] for s in st) / len(st), 2) if st else 0.0
    return row


# ----------------------------------------------------------- parallel map
def pmap(func, tasks, workers, label="", every=None):
    """Ordered results of func(task) using a process pool (spawn-safe:
    ``func`` must be a module-level function).  Prints progress."""
    tasks = list(tasks)
    n = len(tasks)
    every = every or max(1, n // 20)
    t0 = time.time()
    out = [None] * n
    if workers <= 1:
        for i, t in enumerate(tasks):
            out[i] = func(t)
            if (i + 1) % every == 0 or i + 1 == n:
                log("  %s %d/%d  %.0fs" % (label, i + 1, n, time.time() - t0))
        return out
    with mp.Pool(workers) as pool:
        for k, (i, res) in enumerate(pool.imap_unordered(_indexed, [(func, i, t) for i, t in enumerate(tasks)])):
            out[i] = res
            if (k + 1) % every == 0 or k + 1 == n:
                log("  %s %d/%d  %.0fs" % (label, k + 1, n, time.time() - t0))
    return out


def _indexed(arg):
    func, i, t = arg
    return i, func(t)


# ------------------------------------------------------------------ data
def load_accuracy_positions(path=None):
    """data/accuracy_positions.csv -> list of dicts with parsed fields."""
    path = path or os.path.join(DATA, "accuracy_positions.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"moves": r["moves"], "ply": int(r["ply"]),
                         "scores": json.loads(r["scores"]),
                         "optimal_cols": json.loads(r["optimal_cols"]),
                         "source": r.get("source", ""), "bucket": r.get("bucket", "")})
    return rows


def outcome(score):
    """Pons score -> 'win' / 'draw' / 'loss' for the side to move."""
    if score is None:
        return None
    return "win" if score > 0 else ("loss" if score < 0 else "draw")


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def solver_available():
    return os.path.isfile(SOLVER_BIN) and os.access(SOLVER_BIN, os.X_OK)


def board_from_grid_any(grid):
    """Board from a 6x7 grid assumed row0=top; if that violates gravity try
    row0=bottom.  Returns (board, flipped) or (None, None)."""
    try:
        return Board.from_grid(grid), False
    except ValueError:
        pass
    try:
        return Board.from_grid(list(reversed(grid))), True
    except ValueError:
        return None, None


__all__ = ["ROOT", "RESULTS", "FIGURES", "DATA", "COLS", "Board", "LEVELS"]
