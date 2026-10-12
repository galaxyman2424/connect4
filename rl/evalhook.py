"""Shared evaluation for the rl/ trainers and rl/compare.py.

Uses exactly the protocol of nn/bench.py, so the numbers line up with the
minimax and neural-net results already on the Lab page:
  * accuracy vs perfect play on the 3,000 held-out solver-labelled positions
  * games vs our minimax at several depths from random 2-4 ply openings
    (each opening played with both colours) -> Elo on the minimax ladder
    (experiments/results/elo.json, depth 1 = 1000)
Any agent spec understood by nn.bench.make_agent works, including the new
("weights", path, depth) and ("ntuple", path, depth).
"""

import time

from nn import bench


def ladder(spec, depths, n_openings, workers, seed=4150):
    """Games of `spec` vs minimax at each depth -> {"vs": {...}, "elo": float}."""
    openings = bench.random_openings(n_openings, seed=seed)
    tasks = [(spec, ("minimax", d), o, af) for d in depths for o in openings for af in (True, False)]
    games = bench.run_tasks(tasks, workers)
    lad = bench.minimax_ladder_elo()
    vs, fixed = {}, []
    for d in depths:
        s = bench.summarize_match([g for g in games if g["b"] == "minimax_d%d" % d])
        vs["d%d" % d] = s
        if "minimax_d%d" % d in lad:
            fixed.append((lad["minimax_d%d" % d], s["win"] + 0.5 * s["draw"], s["games"]))
    return {"vs": vs, "elo": bench.elo_vs_fixed(fixed), "games": games, "openings": len(openings)}


def _acc_task(task):
    spec, rows = task
    from engine.board import Board
    agent = bench.make_agent(spec)
    cols, ms = [], []
    for r in rows:
        b = Board.from_moves(r["moves"])
        t0 = time.perf_counter()
        cols.append(agent.choose(b))
        ms.append((time.perf_counter() - t0) * 1000)
    return cols, ms


def spec_accuracy(spec, rows, workers=1):
    """Accuracy of any agent spec on bench rows (one choose() per position).
    Returns accuracy_summary(...) + mean ms per move."""
    if workers <= 1 or len(rows) < 40:
        cols, ms = _acc_task((spec, rows))
    else:
        import multiprocessing as mp
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        k = max(1, len(rows) // (workers * 4))
        chunks = [rows[i:i + k] for i in range(0, len(rows), k)]
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_acc_task, [(spec, c) for c in chunks])
        cols = sum((p[0] for p in parts), [])
        ms = sum((p[1] for p in parts), [])
    res = bench.accuracy_summary(rows, cols)
    res["ms"] = round(sum(ms) / max(len(ms), 1), 3)
    return res
