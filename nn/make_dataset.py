"""Phase A data: Connect 4 positions labelled with perfect play.

    python -m nn.make_dataset --positions 300000 --workers 14 --out nn_data/solver_300k.npz

1. Generate positions by playing many games with a mix of move pickers
   (uniformly random moves and the project's minimax at depths 1-4 with some
   randomness), so the data covers both sloppy and sensible play, every
   game phase (plies 0-41) and both colours.
2. Remove duplicates (a position and its left-right mirror count once) and
   remove every position of the held-out benchmark
   data/accuracy_positions.csv (and its mirror), so evaluation stays honest.
3. Label each position with Pascal Pons' solver (tools/pons/c4solver -a with
   the opening book): the exact game-theoretic score of all 7 columns.
   From that we get the training targets:
     policy target = uniform over the optimal columns
     value target  = Win / Draw / Loss for the side to move under perfect play

Needs the solver binary (cd tools/pons && make c4solver) and the opening book
tools/pons/7x6.book (python data/download_data.py --book).  Positions the
solver cannot finish within --timeout seconds are skipped and counted.

Output .npz: cur, mask (uint64 bitboards, side-to-move view, see
nn/encode.py), scores (int8, 7 columns, -128 = full column), ply (uint8).
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from engine.agents import MinimaxAgent
from engine.board import Board

from .common import NN_DATA, ROOT, log

ILLEGAL = -128


def canon(board):
    g = board.to_grid()
    a = tuple(map(tuple, g))
    b = tuple(tuple(r[::-1]) for r in g)
    return min(a, b)


def gen_game(seed):
    """One game with a random mix of move pickers; returns move strings of
    every non-terminal position (including the empty board)."""
    rng = random.Random(seed)
    style = rng.random()
    p_random = rng.choice([0.0, 0.1, 0.25, 0.5, 1.0]) if style < 0.85 else 1.0
    depths = (rng.choice([1, 2, 3, 4]), rng.choice([1, 2, 3, 4]))
    bots = {1: MinimaxAgent(depths[0], seed=rng.randrange(1 << 30)),
            -1: MinimaxAgent(depths[1], seed=rng.randrange(1 << 30))}
    b = Board()
    out = []
    while not b.is_terminal():
        out.append(b.to_move_string())
        if rng.random() < p_random:
            col = rng.choice(b.legal_moves())
        else:
            col = bots[b.to_move].choose(b)
        b.drop(col)
    return out


def _label_worker(args):
    chunk, timeout = args
    from tools.solver import Solver, SolverError
    out = []
    with Solver(timeout=timeout) as s:
        for mv in chunk:
            try:
                out.append((mv, s.solve_all(mv)))
            except SolverError:
                out.append((mv, None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=300_000, help="target number of unique positions")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per position before skipping")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(NN_DATA, "solver_positions.npz"))
    a = ap.parse_args()

    from experiments.common import SOLVER_BIN
    if not (os.path.isfile(SOLVER_BIN) and os.access(SOLVER_BIN, os.X_OK)):
        raise SystemExit("solver binary not found at %s - build it: cd tools/pons && make c4solver" % SOLVER_BIN)
    if not os.path.isfile(os.path.join(ROOT, "tools", "pons", "7x6.book")):
        log("WARNING: no opening book (tools/pons/7x6.book); early positions will be very slow. "
            "Get it with: python data/download_data.py --book")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()

    # held-out benchmark positions (and mirrors) are excluded
    from .bench import load_accuracy_bench
    held = {canon(Board.from_moves(r["moves"])) for r in load_accuracy_bench()}

    # ---- 1+2: generate unique positions
    seen = set()
    moves = []
    ctx = mp.get_context("spawn")
    seed = a.seed * 1_000_003
    batch = 200
    with ctx.Pool(a.workers) as pool:
        while len(moves) < a.positions:
            games = pool.map(gen_game, range(seed, seed + batch * a.workers))
            seed += batch * a.workers
            for g in games:
                for mv in g:
                    k = canon(Board.from_moves(mv))
                    if k in seen or k in held:
                        continue
                    seen.add(k)
                    moves.append(mv)
            log("generated %d unique positions" % len(moves))
    random.Random(a.seed).shuffle(moves)
    moves = moves[:a.positions]

    # ---- 3: label with the solver
    log("labelling %d positions with the solver on %d workers ..." % (len(moves), a.workers))
    chunks = [moves[i:i + 500] for i in range(0, len(moves), 500)]
    labelled, skipped = [], 0
    with ctx.Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(_label_worker, [(c, a.timeout) for c in chunks])):
            for mv, sc in res:
                if sc is None:
                    skipped += 1
                else:
                    labelled.append((mv, sc))
            if (k + 1) % 20 == 0 or k + 1 == len(chunks):
                log("  labelled %d / %d  (skipped %d)  %.0fs" % (len(labelled), len(moves), skipped, time.time() - t0))

    n = len(labelled)
    cur = np.zeros(n, np.uint64)
    mask = np.zeros(n, np.uint64)
    scores = np.full((n, 7), ILLEGAL, np.int8)
    ply = np.zeros(n, np.uint8)
    for i, (mv, sc) in enumerate(labelled):
        b = Board.from_moves(mv)
        cur[i] = b.current_bits()
        mask[i] = b.mask
        ply[i] = b.moves_played
        for c, v in enumerate(sc):
            if v is not None:
                scores[i, c] = v
    best = scores.max(1)
    meta = {"positions": n, "skipped_timeout": skipped, "seed": a.seed, "timeout": a.timeout,
            "seconds": round(time.time() - t0, 1),
            "outcome_counts": {"win": int((best > 0).sum()), "draw": int((best == 0).sum()),
                               "loss": int((best < 0).sum())},
            "ply_hist": np.bincount(ply, minlength=42).tolist(),
            "excluded_benchmark_positions": len(held)}
    np.savez_compressed(a.out, cur=cur, mask=mask, scores=scores, ply=ply,
                        meta_json=np.array(json.dumps(meta)))
    with open(os.path.splitext(a.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log("wrote %s: %d positions (%d skipped), outcomes %s, %.0fs" % (
        a.out, n, skipped, meta["outcome_counts"], time.time() - t0))


if __name__ == "__main__":
    main()
