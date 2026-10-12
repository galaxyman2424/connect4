"""TD(lambda) learning with an n-tuple network - self-play, CPU only, no deep learning.

    python -m rl.ntuple --run td_ntuple --games 2000000
    python -m rl.ntuple --run td_ntuple --resume --games 4000000     # train longer

The idea (Lucas 2008; Thill, Koch & Konen 2012, "Reinforcement learning with
n-tuples on the game Connect-4", which reached near-perfect play this way):

* An n-tuple is a fixed group of n board cells (here: a random walk of 8
  neighbouring cells).  Each cell is in one of 4 states - empty and not yet
  playable, empty and playable now, my stone, opponent's stone - so a tuple
  sees one of 4^8 = 65,536 patterns, and owns a lookup table (LUT) with one
  weight per pattern.
* The value of a position is tanh(sum over tuples of LUT[tuple][pattern]),
  read from the view of the player who just moved ("afterstate" value).
  Every tuple is also read on the mirrored board with the same LUT
  (left-right symmetry), and there are separate LUT sets for "I moved first"
  and "I moved second" (odd/even threat strategy differs).
  Default: 70 tuples x 65,536 patterns x 2 = 9.2 M weights, but only 140 are
  touched per position, so learning and play are fast.
* The agent plays greedily: for every legal column it looks up the value of
  the position after the move (an immediate win is worth exactly +1) and
  picks the best; with probability epsilon it plays a random move instead
  (exploration).
* TD(lambda): after each game, every afterstate's value is moved toward the
  lambda-return built from the values of the positions that followed it
  (negated each ply: good for you is bad for me) and the final result
  (+1 win / 0 draw / -1 loss), discounted by --gamma per ply so that a slow
  loss is better than a quick one.  The step on the weights is alpha x
  (target - value), a cross-entropy-style update: with the usual tanh
  derivative factor every value saturated at +-1 in our tests (the first
  player wins most early self-play games), after which nothing changed.
  Learning from its own predictions, not only
  from the final result, is what makes TD learning sample-efficient.

Trade-offs vs. the deep-learning methods: no GPU, tiny compute per move, but
the "features" (which cells form a tuple) are fixed random patterns and the
table is big (~37 MB) - it memorises patterns instead of generalising like a
convolution does.  On its own it does no search; rl/agents.py also plugs its
value into our minimax (spec ("ntuple", path, depth)) to combine both.

Outputs: runs/<run>/ntuple.npz (the model), config.json, metrics.jsonl,
cost.json.  Lab page: accuracy vs perfect play, value accuracy, Elo.
"""

import argparse
import math
import os
import time

import numpy as np

from nn.common import Run, log, machine_info, now_iso

from . import vecenv as V
from .meter import CostMeter

ROWS, COLS, H1 = 6, 7, 7
N_CELLS = ROWS * COLS
_CELLBITS = np.array([c * H1 + h for c in range(COLS) for h in range(ROWS)], dtype=np.uint64)  # cell i = c*6+h
_BOTTOM = np.uint64(V.BOTTOM_MASK)
_BOARD = np.uint64(V.BOARD_MASK)


def random_tuples(n_tuples, length, seed):
    """Random walks of `length` distinct, 8-neighbour-adjacent cells."""
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n_tuples:
        c, h = int(rng.integers(COLS)), int(rng.integers(ROWS))
        cells = [(c, h)]
        tries = 0
        while len(cells) < length and tries < 200:
            tries += 1
            dc, dh = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
            cc, hh = cells[-1][0] + dc, cells[-1][1] + dh
            if 0 <= cc < COLS and 0 <= hh < ROWS and (cc, hh) not in cells:
                cells.append((cc, hh))
        if len(cells) < length:
            continue
        key = tuple(sorted(cells))
        mkey = tuple(sorted((COLS - 1 - c, h) for c, h in cells))
        if key in seen or mkey in seen:
            continue
        seen.add(key)
        out.append([c * ROWS + h for c, h in cells])
    return np.array(out, dtype=np.int64)


def mirror_tuples(tuples):
    c, h = tuples // ROWS, tuples % ROWS
    return (COLS - 1 - c) * ROWS + h


class NTupleNet:
    def __init__(self, tuples, lut=None):
        self.tuples = np.asarray(tuples, dtype=np.int64)
        self.T, self.n = self.tuples.shape
        self.S = 4 ** self.n
        self.both = np.concatenate([self.tuples, mirror_tuples(self.tuples)])     # (2T, n)
        self.pow = (4 ** np.arange(self.n)).astype(np.int64)
        self.t_off = np.concatenate([np.arange(self.T), np.arange(self.T)]) * self.S  # (2T,)
        if lut is None:
            lut = np.zeros(2 * self.T * self.S, dtype=np.float32)
        self.lut = lut.reshape(-1)
        self.meta = {}

    # ------------------------------------------------------------ indexing
    def indices(self, mover, mask):
        """Afterstates (mover bits, mask) arrays (N,) -> flat LUT indices (N, 2T)."""
        mover = np.asarray(mover, dtype=np.uint64)
        mask = np.asarray(mask, dtype=np.uint64)
        opp = mover ^ mask
        play = (mask + _BOTTOM) & _BOARD
        sh = _CELLBITS[None, :]
        code = (((mover[:, None] >> sh) & np.uint64(1)) * np.uint64(2)
                + ((opp[:, None] >> sh) & np.uint64(1)) * np.uint64(3)
                + ((play[:, None] >> sh) & np.uint64(1))).astype(np.int64)          # (N, 42) in 0..3
        idx = code[:, self.both] @ self.pow                                          # (N, 2T)
        first = V.popcount(mover) > V.popcount(opp)                                  # mover started the game
        par = first.astype(np.int64) * (self.T * self.S)
        return idx + self.t_off[None, :] + par[:, None]

    def value_idx(self, flat):
        return np.tanh(self.lut[flat].sum(1, dtype=np.float64))

    def value(self, mover, mask):
        """Value for the player who just moved, in (-1, 1)."""
        if len(np.atleast_1d(mover)) == 0:
            return np.zeros(0)
        return self.value_idx(self.indices(mover, mask))

    def move_values(self, curs, masks):
        """(B,) positions with side to move -> (B,7) values of each move for the mover:
        exact +1 for an immediate win, 0 for filling the board, -inf for illegal."""
        mover, nmask, lg, won, full = V.afterstates(curs, masks)
        out = np.full(mover.shape, -np.inf)
        need = lg & ~won & ~full
        if need.any():
            out[need] = self.value(mover[need], nmask[need])
        out[full & ~won] = 0.0
        out[won] = 1.0
        return out, lg

    # ---------------------------------------------------------------- io
    def save(self, path, meta=None):
        tmp = path + ".tmp.npz"
        np.savez(tmp, tuples=self.tuples, lut=self.lut.astype(np.float32),
                 meta_json=np.array(__import__("json").dumps(meta or {})))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        import json
        with np.load(path, allow_pickle=False) as z:
            net = cls(z["tuples"], z["lut"].astype(np.float32))
            net.meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        net.path = path
        return net


# ================================================================ benchmark
def bench_eval(net, rows):
    """Greedy policy + value accuracy on nn.bench positions (same metrics as the nets)."""
    from nn import bench
    curs = np.array([r["cur"] for r in rows], dtype=np.uint64)
    masks = np.array([r["mask"] for r in rows], dtype=np.uint64)
    vals, lg = net.move_values(curs, masks)
    cols = V.argmax_center(vals, lg).tolist()
    res = bench.accuracy_summary(rows, cols)
    vstm = np.where(lg, vals, -np.inf).max(1)
    pred = np.where(vstm > 1 / 3, 0, np.where(vstm < -1 / 3, 2, 1))
    truth = np.array([r["outcome"] for r in rows])
    res["value_acc"] = round(float((pred == truth).mean()), 4)
    return res


# ================================================================== training
def lambda_returns(values, final, lam, gamma=1.0):
    """values[p] = V(s_p) for the non-terminal afterstates p = 0..L-2 (each from its
    own mover's view); final = exact result of the last move for ITS mover (+1/0).
    Returns G_p = -gamma [(1-lam) V(s_{p+1}) + lam G_{p+1}], with V(s_{L-1}) = G_{L-1} = final.
    gamma < 1 makes a quick win worth more than a slow one and a slow loss less bad
    than a quick one, so the player still prefers blocking when it is losing anyway."""
    L1 = len(values)
    G = np.empty(L1)
    nxt_v, nxt_g = final, final
    for p in range(L1 - 1, -1, -1):
        G[p] = -gamma * ((1 - lam) * nxt_v + lam * nxt_g)
        nxt_v, nxt_g = values[p], G[p]
    return G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--games", type=int, default=2_000_000, help="total self-play games")
    ap.add_argument("--games-per-gen", type=int, default=50_000, help="log/save every this many games")
    ap.add_argument("--tuples", type=int, default=70)
    ap.add_argument("--length", type=int, default=8, help="cells per tuple (LUT size 4^length)")
    ap.add_argument("--alpha", type=float, default=0.004, help="learning rate per weight at the start")
    ap.add_argument("--alpha-final", type=float, default=0.002)
    ap.add_argument("--lam", type=float, default=0.5, help="TD(lambda) trace decay (0 = TD(0), 1 = Monte Carlo)")
    ap.add_argument("--gamma", type=float, default=0.98, help="discount per ply (1 = none)")
    ap.add_argument("--eps", type=float, default=0.1, help="exploration at the start")
    ap.add_argument("--eps-final", type=float, default=0.02)
    ap.add_argument("--parallel", type=int, default=1024, help="games played in lock-step")
    ap.add_argument("--eval-every", type=int, default=4, help="Elo games every N generations (0 = never)")
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--eval-depths", default="2,4,6")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    from nn import bench
    from .evalhook import ladder

    run = Run(a.run)
    path = run.path("ntuple.npz")
    if run.exists() and not a.resume:
        raise SystemExit("runs/%s exists - use --resume or another --run" % a.run)
    rng = np.random.default_rng(a.seed)
    if a.resume and os.path.isfile(path):
        net = NTupleNet.load(path)
        done_games = int(net.meta.get("games", 0))
        gen = int(net.meta.get("gen", 0))
        log("resuming %s at %d games" % (a.run, done_games))
        rng = np.random.default_rng(a.seed + gen)
    else:
        net = NTupleNet(random_tuples(a.tuples, a.length, a.seed))
        done_games, gen = 0, 0
        run.write_config({"phase": "ntuple", "created": now_iso(), "args": vars(a), "device": "cpu",
                          "model": {"kind": "n-tuple", "tuples": net.T, "length": net.n,
                                    "lut_entries": int(net.lut.size)},
                          "params": int(net.lut.size), "machine": machine_info()})
        net.save(path, {"games": 0, "gen": 0})
    log("n-tuple net: %d tuples x %d cells, %.1f M weights (%.0f MB)" % (
        net.T, net.n, net.lut.size / 1e6, net.lut.nbytes / 1e6))

    meter = CostMeter(uses_gpu=False, label="rl.ntuple " + a.run).start()
    rows = bench.load_accuracy_bench()
    depths = [int(x) for x in a.eval_depths.split(",") if x]
    B = a.parallel
    games = V.VecGames(B, rng)
    traj = [[] for _ in range(B)]             # per game: flat index arrays of non-terminal afterstates
    t_run = time.time()

    while done_games < a.games:
        gen += 1
        t0 = time.time()
        target = min(a.games, done_games + a.games_per_gen)
        n_fin, lengths, res = 0, [], []
        td_abs, td_n = 0.0, 0
        while done_games + n_fin < target:
            frac = (done_games + n_fin) / max(1, a.games)
            eps = a.eps + (a.eps_final - a.eps) * frac
            alpha = a.alpha * (a.alpha_final / a.alpha) ** frac
            vals, lg = net.move_values(games.cur, games.mask)
            cols = V.argmax_center(vals, lg)
            explore = rng.random(B) < eps
            if explore.any():
                cols[explore] = V.random_legal(lg[explore], rng)
            mv = V.move_bits(games.mask, cols)
            mover = games.cur | mv
            nmask = games.mask | mv
            won, full = games.step(cols)
            done = won | full
            live = np.nonzero(~done)[0]
            if len(live):
                flat = net.indices(mover[live], nmask[live])
                for k, g in enumerate(live):
                    traj[g].append(flat[k])
            fin = np.nonzero(done)[0]
            if len(fin):
                # ---- TD(lambda) update for every finished game
                allf, segs = [], []
                for g in fin:
                    if traj[g]:
                        segs.append((len(allf), len(traj[g]), 1.0 if won[g] else 0.0))
                        allf.extend(traj[g])
                if allf:
                    F = np.stack(allf)
                    Vs = net.value_idx(F)
                    G = np.empty(len(allf))
                    for s0, L, final in segs:
                        G[s0:s0 + L] = lambda_returns(Vs[s0:s0 + L], final, a.lam, a.gamma)
                    # cross-entropy-style step on the pre-tanh sum: (G - V), WITHOUT the tanh
                    # derivative (1 - V^2), which vanishes once values saturate near +-1 and
                    # freezes learning (every move then looks equally lost/won)
                    delta = (G - Vs) * alpha
                    td_abs += float(np.abs(G - Vs).sum())
                    td_n += len(G)
                    np.add.at(net.lut, F.ravel(), np.repeat(delta, F.shape[1]).astype(np.float32))
                for g in fin:
                    lengths.append(int(games.ply[g]))
                    # winner: last mover; ply is now the number of stones
                    res.append(0 if not won[g] else (1 if games.ply[g] % 2 == 1 else -1))
                    traj[g] = []
                n_fin += len(fin)
                games.reset(fin)
        done_games += n_fin
        t_train = time.time() - t0

        net.save(path, {"games": done_games, "gen": gen, "run": a.run, "exported": now_iso()})
        t1 = time.time()
        pb = bench_eval(net, rows)
        n = len(res)
        rec = {"type": "generation", "phase": "ntuple", "gen": gen, "games": n, "total_games": done_games,
               "eps": round(eps, 4), "alpha": round(alpha, 6),
               "p1_win": round(sum(r == 1 for r in res) / n, 4), "draw": round(sum(r == 0 for r in res) / n, 4),
               "p2_win": round(sum(r == -1 for r in res) / n, 4), "avg_length": round(float(np.mean(lengths)), 2),
               "td_error": round(td_abs / max(td_n, 1), 5),
               "bench_optimal": pb["optimal"], "bench_keeps_nl": pb["keeps_nl"], "bench_value_acc": pb["value_acc"],
               "bench_by_bucket": pb["by_bucket"], "train_sec": round(t_train, 1),
               "games_per_sec": round(n / max(t_train, 1e-9), 1)}
        if a.eval_every and (gen % a.eval_every == 0 or done_games >= a.games):
            lad = ladder(("ntuple", path, 0), depths, a.eval_openings, a.workers)
            rec["vs_minimax"], rec["elo"] = lad["vs"], lad["elo"]
        rec["eval_sec"] = round(time.time() - t1, 1)
        rec["elapsed_sec"] = round(time.time() - t_run, 1)
        rec.update(meter.snapshot())
        run.log_metrics(rec)
        meter.write(run.dir)
        msg = "gen %d  %d games (%.0f/s)  P1 %.0f%% draw %.0f%% len %.1f  td_err %.3f  eps %.3f  " \
              "bench optimal %.3f value %.3f" % (gen, done_games, rec["games_per_sec"], 100 * rec["p1_win"],
                                                100 * rec["draw"], rec["avg_length"], rec["td_error"], eps,
                                                pb["optimal"], pb["value_acc"])
        if "elo" in rec:
            msg += "\n      vs minimax: " + "  ".join("%s %.2f" % (k, v["score"]) for k, v in rec["vs_minimax"].items())
            msg += "   Elo~%s" % rec["elo"]
        log(msg)
    meter.stop()
    meter.write(run.dir)
    log("done. Next: python -m rl.compare (or play it: spec ntuple:%s:4 in the arena)" % path)


if __name__ == "__main__":
    main()
