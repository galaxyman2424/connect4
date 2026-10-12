"""Evolutionary tuning of the minimax heuristic (CMA-ES).  CPU only, no gradients.

    python -m rl.evolve --run evo_d2 --generations 60                  # fitness = games
    python -m rl.evolve --run evo_acc --fitness accuracy --data nn_data/solver_300k.npz
    python -m rl.evolve --run evo_d2 --resume --generations 100

What it does
  The heuristic at the leaves of our minimax is a weighted sum of board
  features (rl/features.py: the 5 hand-crafted ones + 5 new ones such as
  odd/even threats).  Instead of choosing the weights by hand, CMA-ES
  (Covariance Matrix Adaptation Evolution Strategy, Hansen 2001) searches for
  them:
    1. sample a population of weight vectors around the current mean
    2. score each one (its "fitness")
    3. move the mean toward the best ones, and adapt the step size and the
       shape of the sampling distribution (the covariance matrix) so that
       directions that keep paying off get explored more
  Fitness options:
    games     (default) each candidate plays every opening of this generation
              with both colours against (a) the hand-crafted minimax and (b)
              the current mean, at --depth.  Openings change every generation
              so the weights don't overfit a fixed set.  No labels needed.
    accuracy  share of --fit-positions solver-labelled training positions
              (nn/make_dataset.py output; the 3,000 benchmark positions are
              never in there) where minimax with these weights picks an optimal
              move.  Less noisy but needs perfect-play labels.

Trade-offs: needs no GPU, gradients or differentiable model, and the result
is 10 numbers you can read and explain.  But every fitness evaluation is a
batch of full games, so it is sample-hungry and noisy, and it can only
re-weight the features a human designed - it can't discover new ones.

Outputs: runs/<run>/weights.json (the mean, used as the result),
runs/<run>/checkpoints/cma.json (for --resume), metrics.jsonl, cost.json.
Use the result anywhere with the agent spec ("weights", "runs/<run>/weights.json", depth).
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np

from nn.common import Run, log, machine_info, now_iso

from .features import HAND_WEIGHTS, N_FEATURES, NAMES, WEIGHT_LIMIT
from .meter import CostMeter


# =================================================================== CMA-ES
class CMAES:
    """Plain (mu/mu_w, lambda)-CMA-ES following Hansen's tutorial (arXiv:1604.00772).
    Maximises fitness."""

    def __init__(self, x0, sigma0, popsize=None, seed=0):
        n = len(x0)
        self.n = n
        self.lam = popsize or 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / float((self.w ** 2).sum())
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean = np.array(x0, dtype=float)
        self.sigma = float(sigma0)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.gen = 0
        self.rng = np.random.default_rng(seed)

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        y = (z * self.D) @ self.B.T
        return self.mean + self.sigma * y

    def tell(self, xs, fitness):
        self.gen += 1
        order = np.argsort(-np.asarray(fitness))[: self.mu]
        xold = self.mean.copy()
        y = (np.asarray(xs)[order] - xold) / self.sigma
        self.mean = xold + self.sigma * (self.w @ y)
        ymean = (self.mean - xold) / self.sigma
        inv_sqrt_c = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (inv_sqrt_c @ ymean)
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean
        rank_mu = (y.T * self.w) @ y
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = float(min(self.sigma, 50.0))
        self.C = (self.C + self.C.T) / 2
        d2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(d2, 1e-20))

    def state(self):
        return {"mean": self.mean.tolist(), "sigma": self.sigma, "pc": self.pc.tolist(), "ps": self.ps.tolist(),
                "C": self.C.tolist(), "gen": self.gen, "lam": self.lam}

    @classmethod
    def from_state(cls, s, seed):
        o = cls(s["mean"], s["sigma"], s["lam"], seed)
        o.pc, o.ps, o.C, o.gen = np.array(s["pc"]), np.array(s["ps"]), np.array(s["C"]), s["gen"]
        d2, o.B = np.linalg.eigh(o.C)
        o.D = np.sqrt(np.maximum(d2, 1e-20))
        return o


# ================================================================== fitness
def _play(agent_a, agent_b, opening, a_first):
    from engine.board import Board
    board = Board.from_moves(opening)
    side = {1: agent_a, -1: agent_b} if a_first else {1: agent_b, -1: agent_a}
    while True:
        board.drop(side[board.to_move].choose(board))
        w = board.check_win()
        if w or board.is_draw():
            break
    a_color = 1 if a_first else -1
    if w == 0:
        return 0.5
    # small tie-breaker: faster wins / slower losses score a little better
    bonus = 0.002 * (42 - board.moves_played)
    return 1.0 + bonus if w == a_color else 0.0 - bonus


def games_fitness(task):
    """task = (weights, opponents, openings, depth) -> mean score of weights."""
    from engine.agents import MinimaxAgent
    from .agents import EvolvedMinimaxAgent
    w, opponents, openings, depth = task
    me = EvolvedMinimaxAgent(w, depth)
    total, n = 0.0, 0
    for opp in opponents:
        other = MinimaxAgent(depth) if opp is None else EvolvedMinimaxAgent(opp, depth)
        for o in openings:
            for af in (True, False):
                total += _play(me, other, o, af)
                n += 1
    return total / n


def board_from_bits(cur, mask):
    from engine.board import Board
    cur, mask = int(cur), int(mask)
    b = Board()
    n = mask.bit_count()
    b.to_move = 1 if n % 2 == 0 else -1
    if b.to_move == 1:
        b.p1, b.p2 = cur, cur ^ mask
    else:
        b.p2, b.p1 = cur, cur ^ mask
    b.mask, b.moves_played = mask, n
    return b


def accuracy_fitness(task):
    """task = (weights, positions [(cur, mask, optimal_cols)], depth) -> share optimal."""
    from .agents import EvolvedMinimaxAgent
    w, positions, depth = task
    me = EvolvedMinimaxAgent(w, depth)
    ok = 0
    for cur, mask, opt in positions:
        ok += me.choose(board_from_bits(cur, mask)) in opt
    return ok / len(positions)


def load_fit_positions(path):
    z = np.load(path)
    scores = z["scores"].astype(np.int16)
    legal = scores != -128
    best = np.where(legal, scores, -1000).max(1)
    opt = (scores == best[:, None]) & legal
    return z["cur"], z["mask"], opt


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=14, help="candidates per generation (ideally = --workers)")
    ap.add_argument("--sigma", type=float, default=2.0, help="initial step size")
    ap.add_argument("--init", choices=["hand", "zeros", "random"], default="hand",
                    help="start from the hand-crafted weights, all zeros, or random")
    ap.add_argument("--depth", type=int, default=2, help="minimax depth used during evolution")
    ap.add_argument("--fitness", choices=["games", "accuracy"], default="games")
    ap.add_argument("--openings", type=int, default=12, help="games fitness: new random openings per generation")
    ap.add_argument("--data", default=None, help="accuracy fitness: nn_data/*.npz from nn.make_dataset")
    ap.add_argument("--fit-positions", type=int, default=400, help="accuracy fitness: positions per generation")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-depths", default="1,2,4,6")
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--bench-positions", type=int, default=600, help="benchmark subset for the accuracy curve")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    from nn import bench
    from .evalhook import ladder, spec_accuracy

    run = Run(a.run)
    ckpt = os.path.join(run.ckpt_dir, "cma.json")
    wpath = run.path("weights.json")
    if run.exists() and not a.resume:
        raise SystemExit("runs/%s exists - use --resume or another --run" % a.run)
    if a.resume and os.path.isfile(ckpt):
        with open(ckpt, encoding="utf-8") as f:
            st = json.load(f)
        prev = run.read_config()["args"]
        for k in ("depth", "fitness", "openings", "data", "fit_positions", "popsize"):
            setattr(a, k, prev.get(k, getattr(a, k)))
        es = CMAES.from_state(st["cma"], a.seed + st["cma"]["gen"])
        best_ever = st.get("best_ever")
        log("resuming %s at generation %d (sigma %.3f)" % (a.run, es.gen, es.sigma))
    else:
        if a.init == "hand":
            x0 = np.array(HAND_WEIGHTS, dtype=float)
        elif a.init == "zeros":
            x0 = np.zeros(N_FEATURES)
        else:
            x0 = np.random.default_rng(a.seed).normal(0, 3, N_FEATURES)
        es = CMAES(x0, a.sigma, a.popsize, a.seed)
        best_ever = None
        run.write_config({"phase": "evolution", "created": now_iso(), "args": vars(a), "device": "cpu",
                          "model": {"kind": "weighted heuristic", "features": list(NAMES),
                                    "hand_weights": list(HAND_WEIGHTS), "search_depth": a.depth},
                          "params": N_FEATURES, "machine": machine_info()})
    if a.fitness == "accuracy":
        if not a.data:
            raise SystemExit("--fitness accuracy needs --data nn_data/<file>.npz (python -m nn.make_dataset)")
        fit_cur, fit_mask, fit_opt = load_fit_positions(a.data)

    meter = CostMeter(uses_gpu=False, label="rl.evolve " + a.run).start()
    rows = bench.load_accuracy_bench()
    bench_sub = random.Random(11).sample(rows, min(a.bench_positions, len(rows)))
    depths = [int(x) for x in a.eval_depths.split(",") if x]
    import multiprocessing as mp
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    pool = mp.get_context("spawn").Pool(a.workers)
    t_run = time.time()

    def save_weights(mean, gen):
        with open(wpath, "w", encoding="utf-8") as f:
            json.dump({"weights": [round(float(x), 4) for x in mean], "features": list(NAMES),
                       "hand_weights": list(HAND_WEIGHTS), "search_depth": a.depth, "gen": gen,
                       "fitness": a.fitness, "run": a.run, "saved": now_iso()}, f, indent=2)

    try:
        while es.gen < a.generations:
            t0 = time.time()
            gen = es.gen + 1
            xs = np.clip(es.ask(), -WEIGHT_LIMIT, WEIGHT_LIMIT)
            if a.fitness == "games":
                openings = bench.random_openings(a.openings, seed=a.seed * 7919 + gen)
                opponents = [None, es.mean.tolist()]                     # hand-crafted + current mean
                tasks = [(x.tolist(), opponents, openings, a.depth) for x in xs]
                fit = pool.map(games_fitness, tasks, chunksize=1)
            else:
                rng = np.random.default_rng(a.seed * 7919 + gen)
                idx = rng.choice(len(fit_cur), size=min(a.fit_positions, len(fit_cur)), replace=False)
                pos = [(int(fit_cur[i]), int(fit_mask[i]), set(np.nonzero(fit_opt[i])[0].tolist())) for i in idx]
                fit = pool.map(accuracy_fitness, [(x.tolist(), pos, a.depth) for x in xs], chunksize=1)
            fit = np.array(fit)
            ib = int(np.argmax(fit))
            if best_ever is None or fit[ib] > best_ever["fitness"]:
                best_ever = {"fitness": float(fit[ib]), "weights": xs[ib].tolist(), "gen": gen}
            es.tell(xs, fit)
            es.mean = np.clip(es.mean, -WEIGHT_LIMIT, WEIGHT_LIMIT)
            t_fit = time.time() - t0
            save_weights(es.mean, gen)
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump({"cma": es.state(), "best_ever": best_ever}, f)

            rec = {"type": "generation", "phase": "evolution", "gen": gen, "sigma": round(es.sigma, 4),
                   "fitness_best": round(float(fit.max()), 4), "fitness_mean": round(float(fit.mean()), 4),
                   "fitness_worst": round(float(fit.min()), 4),
                   "weights": {n: round(float(v), 3) for n, v in zip(NAMES, es.mean)},
                   "games": len(xs) * (2 * a.openings * 2 if a.fitness == "games" else 0),
                   "train_sec": round(t_fit, 1)}
            rec["total_games"] = sum(r.get("games", 0) for r in run.read_metrics()) + rec["games"]
            t1 = time.time()
            if a.eval_every and (gen % a.eval_every == 0 or gen == a.generations or gen == 1):
                spec = ("weights", wpath, a.depth)
                acc = spec_accuracy(spec, bench_sub, a.workers)
                rec["bench_optimal"], rec["bench_keeps_nl"], rec["bench_ms"] = acc["optimal"], acc["keeps_nl"], acc["ms"]
                rec["bench_positions"] = len(bench_sub)
                lad = ladder(spec, depths, a.eval_openings, a.workers)
                rec["vs_minimax"], rec["elo"] = lad["vs"], lad["elo"]
            rec["eval_sec"] = round(time.time() - t1, 1)
            rec["elapsed_sec"] = round(time.time() - t_run, 1)
            rec.update(meter.snapshot())
            run.log_metrics(rec)
            meter.write(run.dir)
            msg = "gen %d/%d  fitness best %.3f mean %.3f  sigma %.3f  [%.0fs]\n      weights %s" % (
                gen, a.generations, rec["fitness_best"], rec["fitness_mean"], es.sigma, t_fit,
                "  ".join("%s %.2f" % (n, v) for n, v in zip(NAMES, es.mean)))
            if "elo" in rec:
                msg += "\n      bench optimal %.3f at depth %d   vs minimax: %s   Elo~%s" % (
                    rec["bench_optimal"], a.depth,
                    "  ".join("%s %.2f" % (k, v["score"]) for k, v in rec["vs_minimax"].items()), rec["elo"])
            log(msg)
    finally:
        pool.terminate()
    meter.stop()
    meter.write(run.dir)
    log("done. weights in %s. Next: python -m rl.compare" % wpath)


if __name__ == "__main__":
    main()
