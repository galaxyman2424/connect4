"""Phase B: AlphaZero-style reinforcement learning from self-play (no teacher).

    python -m nn.train_selfplay --run az_6x64 --preset standard
    python -m nn.train_selfplay --run az_6x64 --resume          # continue after a stop

The network starts with random weights and knows only the rules. Each
generation:

  1. SELF-PLAY  --games games of the current network against itself. Every
     move is chosen by MCTS (--sims simulations) guided by the network.
     Exploration: Dirichlet noise on the root priors, and moves are sampled
     in proportion to visit counts for the first --temp-moves plies (then the
     most-visited move is played).
     Each position is stored with
       pi = MCTS visit distribution        (policy target: "what search found")
       z  = final result for the player to move (win / draw / loss)
  2. TRAIN      --steps gradient steps on random mini-batches from a replay
     buffer of the most recent --buffer positions (with mirror augmentation).
     Loss = cross-entropy(policy, pi) + cross-entropy(WDL, z).
  3. EXPORT     runs/<run>/model.npz (+ checkpoints/latest.pt, buffer.npz).
  4. MEASURE    every generation: held-out accuracy vs perfect play (policy
     head) and value accuracy; every --eval-every generations: MCTS accuracy
     and matches against the minimax bot at --eval-depths, from which an
     Elo on the existing minimax ladder (experiments/results/elo.json) is
     estimated.
  All numbers go to runs/<run>/metrics.jsonl, which the Lab page charts.

The search makes the network's own moves better than its raw policy; training
the policy toward the search result and the value toward real outcomes makes
the next search better still. That loop is the whole idea of AlphaZero.

Presets (games/gen x generations, wall-clock on a Ryzen 7 5800X + RTX 3050 is
an estimate - check the first generations' timing in the log):
  quick     200 x 15, 64 sims       smoke test, ~10-15 min
  standard  1000 x 100, 128 sims    ~100k games, a few hours
  long      2000 x 200, 200 sims    ~400k games, overnight+
Explicit flags override the preset.
"""

import argparse
import math
import multiprocessing as mp
import os
import time

import numpy as np

from .common import Run, log, machine_info, now_iso, pick_device

PRESETS = {
    "quick": dict(games=200, generations=15, sims=64, buffer=150_000, eval_every=5, eval_openings=6),
    "standard": dict(games=1000, generations=100, sims=128, buffer=600_000, eval_every=5, eval_openings=10),
    "long": dict(games=2000, generations=200, sims=200, buffer=1_200_000, eval_every=10, eval_openings=20),
}
DEFAULTS = dict(games=1000, generations=100, sims=128, buffer=600_000, eval_every=5, eval_openings=10)


# =============================================================== self-play
_W = {}


def _worker_model(ckpt, device):
    """Load (and cache) the current network inside a self-play worker."""
    import torch
    from .common import TorchEvaluator
    from .model import load_checkpoint
    key = (ckpt, os.path.getmtime(ckpt), device)
    if _W.get("key") != key:
        torch.set_num_threads(1)
        dev = torch.device(device)
        model, _ = load_checkpoint(ckpt, dev)
        _W["ev"] = TorchEvaluator(model, dev, amp=(dev.type == "cuda"))
        _W["key"] = key
    return _W["ev"]


def selfplay_worker(task):
    """Play task['games'] self-play games; returns training arrays + stats."""
    from . import mcts
    ev = _worker_model(task["ckpt"], task["device"])
    rng = np.random.default_rng(task["seed"])
    n_total, par = task["games"], max(1, min(task["parallel"], task["games"]))
    sims, temp_moves = task["sims"], task["temp_moves"]
    started = 0
    games = []

    def new_game():
        return {"root": mcts.Node(0, 0), "ply": 0, "hist": [], "first": None}

    live = []
    while started < n_total and len(live) < par:
        live.append(new_game())
        started += 1
    out_cur, out_mask, out_pi, out_z, out_ply = [], [], [], [], []
    lengths, results, first_moves = [], [], []
    while live:
        roots = [g["root"] for g in live]
        mcts.expand_roots(roots, ev)
        for r in roots:
            mcts.add_dirichlet(r, rng, task["dir_alpha"], task["dir_eps"])
        mcts.search(roots, ev, sims, c_puct=task["c_puct"])
        nxt = []
        for g in live:
            root = g["root"]
            pi = mcts.visit_policy(root)
            temp = 1.0 if g["ply"] < temp_moves else 0.0
            a = mcts.choose_move(root, temp, rng)
            g["hist"].append((root.cur, root.mask, pi, g["ply"]))
            if g["first"] is None:
                g["first"] = a
            child = root.child(a)
            g["ply"] += 1
            if child.terminal is not None:
                # winner = player who just moved (if not a draw)
                if child.terminal == 0.0:
                    res_p1 = 0
                else:
                    res_p1 = 1 if g["ply"] % 2 == 1 else -1
                for cur, mask, p, ply in g["hist"]:
                    stm = 1 if ply % 2 == 0 else -1
                    z = 1 if res_p1 == 0 else (0 if res_p1 == stm else 2)   # 0 win,1 draw,2 loss
                    out_cur.append(cur); out_mask.append(mask); out_pi.append(p)
                    out_z.append(z); out_ply.append(ply)
                lengths.append(g["ply"])
                results.append(res_p1)
                first_moves.append(g["first"])
                if started < n_total:
                    nxt.append(new_game())
                    started += 1
            else:
                g["root"] = child
                nxt.append(g)
        live = nxt
    return {"cur": np.array(out_cur, dtype=np.uint64), "mask": np.array(out_mask, dtype=np.uint64),
            "pi": np.array(out_pi, dtype=np.float32), "z": np.array(out_z, dtype=np.int64),
            "ply": np.array(out_ply, dtype=np.uint8), "lengths": lengths, "results": results,
            "first_moves": first_moves}


# ================================================================= buffer
class Replay:
    def __init__(self, capacity):
        self.cap = capacity
        self.cur = np.zeros(0, np.uint64)
        self.mask = np.zeros(0, np.uint64)
        self.pi = np.zeros((0, 7), np.float32)
        self.z = np.zeros(0, np.int64)

    def add(self, d):
        self.cur = np.concatenate([self.cur, d["cur"]])[-self.cap:]
        self.mask = np.concatenate([self.mask, d["mask"]])[-self.cap:]
        self.pi = np.concatenate([self.pi, d["pi"]])[-self.cap:]
        self.z = np.concatenate([self.z, d["z"]])[-self.cap:]

    def __len__(self):
        return len(self.cur)

    def save(self, path):
        tmp = path + ".tmp.npz"
        np.savez(tmp, cur=self.cur, mask=self.mask, pi=self.pi, z=self.z)
        os.replace(tmp, path)

    def load(self, path):
        z = np.load(path)
        self.cur, self.mask, self.pi, self.z = z["cur"], z["mask"], z["pi"], z["z"]

    def to_device(self, device):
        import torch
        from .encode import bits_to_grid
        n = len(self)
        me = bits_to_grid(self.cur)
        them = bits_to_grid(self.cur ^ self.mask)
        first = me.reshape(n, -1).sum(1) == them.reshape(n, -1).sum(1)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
        return {"grids": t(np.stack([me, them], 1)), "first": t(first), "pi": t(self.pi), "z": t(self.z)}


def train_steps(model, opt, D, steps, batch, device, value_weight):
    import torch
    import torch.nn.functional as F
    model.train()
    n = D["z"].shape[0]
    sums = np.zeros(3)
    use_amp = device.type == "cuda"
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        g = D["grids"][idx].float()
        b = g.shape[0]
        first = D["first"][idx].float().view(b, 1, 1, 1).expand(b, 1, 6, 7)
        x = torch.cat([g, torch.ones((b, 1, 6, 7), device=device), first], 1)
        pi = D["pi"][idx]
        mirror = torch.rand(b, device=device) < 0.5
        x = torch.where(mirror.view(b, 1, 1, 1), x.flip(3), x)
        pi = torch.where(mirror.view(b, 1), pi.flip(1), pi)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            p_logits, v_logits = model(x)
        lp = -(pi * F.log_softmax(p_logits.float(), 1)).sum(1).mean()
        lv = F.cross_entropy(v_logits.float(), D["z"][idx])
        loss = lp + value_weight * lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sums += [loss.item(), lp.item(), lv.item()]
    model.eval()
    return sums / max(steps, 1)


# =================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--resume", action="store_true", help="continue runs/<run> from its last generation")
    ap.add_argument("--init", default=None, help="start from these weights (.pt), e.g. a supervised run")
    ap.add_argument("--generations", type=int)
    ap.add_argument("--games", type=int, help="self-play games per generation")
    ap.add_argument("--sims", type=int, help="MCTS simulations per move in self-play")
    ap.add_argument("--buffer", type=int, help="replay buffer size (positions)")
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--eval-openings", type=int, help="openings per minimax depth (x2 colours)")
    ap.add_argument("--eval-depths", default="2,4,6")
    ap.add_argument("--eval-sims", type=int, default=100, help="MCTS sims for evaluation games/accuracy")
    ap.add_argument("--reuse", type=float, default=4.0,
                    help="train steps per gen = new positions * reuse / batch")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=1e-4, help="lr decays (log-linearly) to this by the last generation")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temp-moves", type=int, default=12, help="sample moves (temperature 1) for this many plies")
    ap.add_argument("--dir-alpha", type=float, default=1.0)
    ap.add_argument("--dir-eps", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--parallel", type=int, default=128, help="games played in lock-step per worker (GPU batch size)")
    ap.add_argument("--device", default="auto", help="training device")
    ap.add_argument("--selfplay-device", default="auto", help="device for self-play workers (auto = same GPU)")
    ap.add_argument("--match-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import torch
    from . import bench
    from .common import TorchEvaluator
    from .export import export_model
    from .model import build, count_params, load_checkpoint, save_checkpoint

    run = Run(a.run)
    if run.exists() and not a.resume:
        raise SystemExit("runs/%s exists - use --resume to continue it or pick another --run" % a.run)
    if a.resume and run.exists():
        prev = run.read_config()["args"]
        for k, v in prev.items():               # keep the original settings unless overridden
            if getattr(a, k, None) is None:
                setattr(a, k, v)
    base = dict(DEFAULTS)
    base.update(PRESETS.get(a.preset or "", {}))
    for k, v in base.items():
        if getattr(a, k) is None:
            setattr(a, k, v)

    device = pick_device(a.device)
    sp_device = str(pick_device(a.selfplay_device))
    torch.manual_seed(a.seed)
    depths = [int(x) for x in a.eval_depths.split(",") if x]

    latest = os.path.join(run.ckpt_dir, "latest.pt")
    buf_path = os.path.join(run.ckpt_dir, "buffer.npz")
    replay = Replay(a.buffer)
    start_gen, total_games, total_pos = 1, 0, 0
    if a.resume and os.path.isfile(latest):
        model, obj = load_checkpoint(latest, device)
        start_gen = obj.get("gen", 0) + 1
        total_games, total_pos = obj.get("total_games", 0), obj.get("total_positions", 0)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        if "optimizer" in obj:
            opt.load_state_dict(obj["optimizer"])
        if os.path.isfile(buf_path):
            replay.load(buf_path)
        log("resuming %s at generation %d (%d games so far, buffer %d)" % (a.run, start_gen, total_games, len(replay)))
    else:
        if a.init:
            model, _ = load_checkpoint(a.init, device)
            log("initialised from %s" % a.init)
        else:
            model = build({"blocks": a.blocks, "channels": a.channels}).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        run.write_config({"phase": "selfplay", "created": now_iso(), "args": vars(a),
                          "device": str(device), "selfplay_device": sp_device,
                          "model": model.config, "params": count_params(model),
                          "machine": machine_info()})
        save_checkpoint(latest, model, {"gen": 0, "total_games": 0, "total_positions": 0})
        export_model(model.eval(), run.path("model.npz"), {"run": a.run, "gen": 0})
    model.eval()
    log("run %s: %s, %d params, self-play on %s with %d workers x %d parallel games" % (
        a.run, device, count_params(model), sp_device, a.workers, a.parallel))

    bench_rows = bench.load_accuracy_bench()
    bench_sub = bench_rows[::3]                      # 1,000 positions for the (slower) MCTS check
    ladder = bench.minimax_ladder_elo()
    openings = bench.random_openings(a.eval_openings, seed=4150)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.workers)
    t_run = time.time()
    try:
        for gen in range(start_gen, a.generations + 1):
            t0 = time.time()
            # learning rate for this generation (log-linear decay)
            frac = (gen - 1) / max(1, a.generations - 1)
            lr = a.lr * (a.lr_final / a.lr) ** frac
            for gparam in opt.param_groups:
                gparam["lr"] = lr

            # ---- 1. self-play
            per = [a.games // a.workers + (1 if i < a.games % a.workers else 0) for i in range(a.workers)]
            tasks = [{"ckpt": latest, "device": sp_device, "games": n, "parallel": a.parallel,
                      "sims": a.sims, "c_puct": a.c_puct, "temp_moves": a.temp_moves,
                      "dir_alpha": a.dir_alpha, "dir_eps": a.dir_eps,
                      "seed": a.seed * 1_000_003 + gen * 1009 + i} for i, n in enumerate(per) if n > 0]
            parts = pool.map(selfplay_worker, tasks)
            new = {k: np.concatenate([p[k] for p in parts]) for k in ("cur", "mask", "pi", "z", "ply")}
            lengths = sum((p["lengths"] for p in parts), [])
            results = sum((p["results"] for p in parts), [])
            firsts = sum((p["first_moves"] for p in parts), [])
            replay.add(new)
            t_sp = time.time() - t0
            total_games += len(lengths)
            total_pos += len(new["z"])

            # ---- 2. train
            t1 = time.time()
            steps = max(1, int(math.ceil(len(new["z"]) * a.reuse / a.batch)))
            D = replay.to_device(device)
            losses = train_steps(model, opt, D, steps, a.batch, device, a.value_weight)
            del D
            t_tr = time.time() - t1

            # ---- 3. export
            save_checkpoint(latest, model, {"gen": gen, "total_games": total_games,
                                            "total_positions": total_pos, "optimizer": opt.state_dict()})
            if gen % 10 == 0:
                save_checkpoint(os.path.join(run.ckpt_dir, "gen_%04d.pt" % gen), model, {"gen": gen})
            replay.save(buf_path)
            meta = {"run": a.run, "phase": "selfplay", "gen": gen, "total_games": total_games,
                    "exported": now_iso()}
            export_model(model, run.path("model.npz"), meta)

            # ---- 4. measure
            t2 = time.time()
            ev = TorchEvaluator(model, device)
            pb = bench.policy_eval(ev, bench_rows)
            pri, val, wdl = ev.evaluate([0], [0])
            n = len(results)
            rec = {"type": "generation", "phase": "selfplay", "gen": gen, "lr": round(lr, 7),
                   "games": n, "positions": int(len(new["z"])), "buffer": len(replay),
                   "total_games": total_games, "total_positions": total_pos,
                   "p1_win": round(sum(r == 1 for r in results) / n, 4),
                   "draw": round(sum(r == 0 for r in results) / n, 4),
                   "p2_win": round(sum(r == -1 for r in results) / n, 4),
                   "avg_length": round(float(np.mean(lengths)), 2),
                   "first_move_hist": np.bincount(firsts, minlength=7).tolist(),
                   "train_steps": steps, "train_loss": round(float(losses[0]), 5),
                   "policy_loss": round(float(losses[1]), 5), "value_loss": round(float(losses[2]), 5),
                   "bench_optimal": pb["optimal"], "bench_keeps_nl": pb["keeps_nl"],
                   "bench_value_acc": pb["value_acc"], "bench_by_bucket": pb["by_bucket"],
                   "empty_board_policy": [round(float(x), 4) for x in pri[0]],
                   "empty_board_wdl": [round(float(x), 4) for x in wdl[0]],
                   "selfplay_sec": round(t_sp, 1), "train_sec": round(t_tr, 1)}
            if a.eval_every and (gen % a.eval_every == 0 or gen == a.generations):
                mb = bench.mcts_eval(ev, bench_sub, a.eval_sims)
                rec["bench_mcts_sims"] = a.eval_sims
                rec["bench_mcts_optimal"] = mb["optimal"]
                rec["bench_mcts_keeps_nl"] = mb["keeps_nl"]
                vs, fixed = {}, []
                me = ("nn", run.path("model.npz"), "mcts", a.eval_sims)
                tasks = [(me, ("minimax", d), o, af) for d in depths for o in openings for af in (True, False)]
                games = bench.run_tasks(tasks, a.match_workers)
                for d in depths:
                    gs = [g for g in games if g["b"] == "minimax_d%d" % d]
                    s = bench.summarize_match(gs)
                    vs["d%d" % d] = s
                    if "minimax_d%d" % d in ladder:
                        fixed.append((ladder["minimax_d%d" % d], s["win"] + 0.5 * s["draw"], s["games"]))
                rec["vs_minimax"] = vs
                rec["elo"] = bench.elo_vs_fixed(fixed)
            rec["eval_sec"] = round(time.time() - t2, 1)
            rec["elapsed_sec"] = round(time.time() - t_run, 1)
            run.log_metrics(rec)
            msg = "gen %d/%d  games %d (P1 %.0f%% draw %.0f%%, len %.1f)  loss %.3f (p %.3f v %.3f)  " \
                  "bench optimal %.3f value %.3f  [self-play %.0fs, train %.0fs/%d steps, eval %.0fs]" % (
                      gen, a.generations, n, 100 * rec["p1_win"], 100 * rec["draw"], rec["avg_length"],
                      losses[0], losses[1], losses[2], pb["optimal"], pb["value_acc"], t_sp, t_tr, steps,
                      rec["eval_sec"])
            if "vs_minimax" in rec:
                msg += "\n      vs minimax: " + "  ".join("%s %.2f" % (k, v["score"]) for k, v in rec["vs_minimax"].items())
                msg += "   Elo~%s   MCTS optimal %.3f" % (rec["elo"], rec["bench_mcts_optimal"])
            log(msg)
    finally:
        pool.terminate()
    log("done. Next: python -m nn.evaluate --run %s" % a.run)


if __name__ == "__main__":
    main()
