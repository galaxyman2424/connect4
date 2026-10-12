"""Deep Q-learning (double DQN) self-play, with the project's ResNet.

    python -m rl.dqn --run dqn_6x64 --iterations 400
    python -m rl.dqn --run dqn_from_sup --init runs/sup_6x64/checkpoints/best.pt     # warm start
    python -m rl.dqn --run dqn_6x64 --resume --iterations 600

Idea
  Q(s, a) = how good it is for the player to move to play column a in
  position s, in [-1, 1] (+1 = certain win).  The network's 7-output policy
  head is used as the Q head (tanh of its outputs).  Learning rule (Bellman,
  negamax form for a two-player zero-sum game):

      target(s, a) = +1                                   if a wins
                   =  0                                   if a fills the board
                   = -max_a' Q_target(s', a')             otherwise

  s' is the position after a, with the OPPONENT to move, so its best value
  is bad for us (hence the minus).  "Double" DQN: the online network picks
  a', the slower-moving target network scores it - this curbs the
  over-estimation that plain max causes.

  Experience: --parallel games are played at once by the current network
  against itself, epsilon-greedy (random move with probability epsilon,
  decaying from 1.0 to --eps-final).  Every move is stored in a replay buffer
  (--buffer transitions); training samples random mini-batches from it, so
  consecutive, highly correlated moves are not learned from back to back.
  Left-right mirroring doubles the data.  The value head is trained on the
  final game result as a side task, so the exported model also works inside
  MCTS and the NN-minimax hypothesis test.

Trade-offs vs AlphaZero: no tree search while training, so each step is
cheap and the GPU does almost all the work; but it learns only from its own
one-step predictions (bootstrapping), which is unstable in two-player games
(the target moves as the opponent - itself - changes) and needs many more
games.  vs PPO: off-policy, so old experience is reused many times (more
sample-efficient), but it learns values, not a policy, and only plays
greedily.

Outputs: runs/<run>/model.npz (policy head = Q-values; PolicyAgent plays
argmax Q), checkpoints/latest.pt, metrics.jsonl, cost.json.
"""

import argparse
import copy
import os
import time

import numpy as np

from nn.common import Run, log, machine_info, now_iso, pick_device

from . import vecenv as V
from .meter import CostMeter


class Replay:
    def __init__(self, cap):
        self.cap = cap
        self.cur = np.zeros(cap, np.uint64)
        self.mask = np.zeros(cap, np.uint64)
        self.ncur = np.zeros(cap, np.uint64)
        self.nmask = np.zeros(cap, np.uint64)
        self.act = np.zeros(cap, np.int64)
        self.rew = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, bool)
        self.z = np.full(cap, -1, np.int64)       # WDL outcome for the mover (filled when the game ends)
        self.n = 0
        self.ptr = 0

    def add(self, cur, mask, act, rew, done, ncur, nmask):
        k = len(cur)
        idx = (self.ptr + np.arange(k)) % self.cap
        self.cur[idx], self.mask[idx], self.act[idx] = cur, mask, act
        self.rew[idx], self.done[idx], self.ncur[idx], self.nmask[idx] = rew, done, ncur, nmask
        self.z[idx] = -1
        self.ptr = (self.ptr + k) % self.cap
        self.n = min(self.cap, self.n + k)
        return idx

    def sample(self, k, rng):
        return rng.integers(0, self.n, size=k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="start from these weights (.pt), e.g. a supervised run")
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--steps-per-iter", type=int, default=100, help="lock-step moves per iteration")
    ap.add_argument("--parallel", type=int, default=256, help="games played at once")
    ap.add_argument("--updates-per-step", type=int, default=2, help="gradient steps per lock-step move")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--warmup", type=int, default=20_000, help="transitions before training starts")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--target-every", type=int, default=1000, help="copy online -> target every N updates")
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-final", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=float, default=0.4, help="fraction of the run over which epsilon decays")
    ap.add_argument("--value-weight", type=float, default=0.5, help="weight of the auxiliary WDL loss")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-openings", type=int, default=10)
    ap.add_argument("--eval-depths", default="2,4,6")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from nn import bench
    from nn.common import TorchEvaluator
    from nn.export import export_model
    from nn.model import count_params, load_checkpoint, save_checkpoint
    from .evalhook import ladder
    from .torchutil import amp_ctx, build_or_init, legal_t, masked_logits, planes

    run = Run(a.run)
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    latest = os.path.join(run.ckpt_dir, "latest.pt")
    if run.exists() and not a.resume:
        raise SystemExit("runs/%s exists - use --resume or another --run" % a.run)
    start_it, total_games, total_updates = 1, 0, 0
    if a.resume and os.path.isfile(latest):
        model, obj = load_checkpoint(latest, device)
        start_it = obj.get("iteration", 0) + 1
        total_games, total_updates = obj.get("total_games", 0), obj.get("total_updates", 0)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        if "optimizer" in obj:
            opt.load_state_dict(obj["optimizer"])
        log("resuming %s at iteration %d (replay buffer starts empty again)" % (a.run, start_it))
    else:
        model = build_or_init(a.blocks, a.channels, a.init, device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        run.write_config({"phase": "dqn", "created": now_iso(), "args": vars(a), "device": str(device),
                          "model": model.config, "params": count_params(model), "machine": machine_info()})
    target = copy.deepcopy(model).eval()
    model.train()
    log("run %s on %s: %d params, %d games in parallel" % (a.run, device, count_params(model), a.parallel))

    meter = CostMeter(uses_gpu=device.type == "cuda", label="rl.dqn " + a.run).start()
    rng = np.random.default_rng(a.seed + start_it)
    rows = bench.load_accuracy_bench()
    depths = [int(x) for x in a.eval_depths.split(",") if x]
    buf = Replay(a.buffer)
    B = a.parallel
    games = V.VecGames(B, rng, random_opening=(0, 2))
    slots = [[] for _ in range(B)]
    total_steps = a.iterations * a.steps_per_iter
    t_run = time.time()

    def q_values(net, curs, masks):
        with torch.no_grad(), amp_ctx(device):
            p, _ = net(planes(curs, masks, device))
        return torch.tanh(p.float())

    for it in range(start_it, a.iterations + 1):
        t0 = time.time()
        n_games, lengths, results = 0, [], []
        sums, n_upd = np.zeros(4), 0
        for step in range(a.steps_per_iter):
            g_step = (it - 1) * a.steps_per_iter + step
            frac = min(1.0, g_step / max(1, a.eps_decay * total_steps))
            eps = a.eps_start + (a.eps_final - a.eps_start) * frac
            # ---- act
            lg = games.legal()
            model.eval()
            q = q_values(model, games.cur, games.mask).cpu().numpy()
            model.train()
            cols = V.argmax_center(q, lg)
            explore = rng.random(B) < eps
            if explore.any():
                cols[explore] = V.random_legal(lg[explore], rng)
            cur0, mask0 = games.cur.copy(), games.mask.copy()
            won, full = games.step(cols)
            done = won | full
            idx = buf.add(cur0, mask0, cols, won.astype(np.float32), done, games.cur.copy(), games.mask.copy())
            for g in range(B):
                slots[g].append(idx[g])
            fin = np.nonzero(done)[0]
            for g in fin:
                s = slots[g]
                L = len(s)
                for k, slot in enumerate(s):
                    dist = L - 1 - k                    # plies before the end; even = same player as last mover
                    buf.z[slot] = 1 if not won[g] else (0 if dist % 2 == 0 else 2)
                slots[g] = []
                lengths.append(int(games.ply[g]))
                results.append(0 if not won[g] else (1 if games.ply[g] % 2 == 1 else -1))
            n_games += len(fin)
            games.reset(fin)
            # ---- learn
            if buf.n < a.warmup:
                continue
            for _ in range(a.updates_per_step):
                bi = buf.sample(a.batch, rng)
                x = planes(buf.cur[bi], buf.mask[bi], device)
                xn = planes(buf.ncur[bi], buf.nmask[bi], device)
                lg_n = legal_t(buf.nmask[bi], device)
                act = torch.from_numpy(buf.act[bi]).to(device)
                rew = torch.from_numpy(buf.rew[bi]).to(device)
                dn = torch.from_numpy(buf.done[bi].astype(np.float32)).to(device)
                z = torch.from_numpy(buf.z[bi]).to(device)
                mir = torch.rand(len(bi), device=device) < 0.5
                x = torch.where(mir.view(-1, 1, 1, 1), x.flip(3), x)
                xn = torch.where(mir.view(-1, 1, 1, 1), xn.flip(3), xn)
                lg_n = torch.where(mir.view(-1, 1), lg_n.flip(1), lg_n)
                act = torch.where(mir, 6 - act, act)
                with torch.no_grad(), amp_ctx(device):
                    pn_online, _ = model(xn)
                    pn_target, _ = target(xn)
                a_star = masked_logits(pn_online, lg_n).argmax(1)
                q_next = torch.tanh(pn_target.float()).gather(1, a_star[:, None]).squeeze(1)
                y = rew + (1 - dn) * (-q_next)
                with amp_ctx(device):
                    p, v = model(x)
                q_sa = torch.tanh(p.float()).gather(1, act[:, None]).squeeze(1)
                lq = F.smooth_l1_loss(q_sa, y)
                known = z >= 0
                lv = F.cross_entropy(v.float()[known], z[known]) if known.any() else torch.zeros((), device=device)
                loss = lq + a.value_weight * lv
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_updates += 1
                n_upd += 1
                sums += [loss.item(), lq.item(), lv.item(), q_sa.detach().abs().mean().item()]
                if total_updates % a.target_every == 0:
                    target.load_state_dict(model.state_dict())
        total_games += n_games
        t_train = time.time() - t0

        model.eval()
        save_checkpoint(latest, model, {"iteration": it, "total_games": total_games,
                                        "total_updates": total_updates, "optimizer": opt.state_dict()})
        meta = {"run": a.run, "phase": "dqn", "gen": it, "total_games": total_games, "exported": now_iso(),
                "policy_head": "q_values"}
        export_model(model, run.path("model.npz"), meta)
        t1 = time.time()
        ev = TorchEvaluator(model, device)
        pb = bench.policy_eval(ev, rows)
        _, _, wdl0 = ev.evaluate([0], [0])
        m = sums / max(n_upd, 1)
        n = max(len(results), 1)
        rec = {"type": "generation", "phase": "dqn", "gen": it, "games": len(results), "total_games": total_games,
               "updates": n_upd, "total_updates": total_updates, "eps": round(eps, 4), "buffer": buf.n,
               "train_loss": round(float(m[0]), 5), "q_loss": round(float(m[1]), 5), "value_loss": round(float(m[2]), 5),
               "q_abs_mean": round(float(m[3]), 4),
               "p1_win": round(sum(r == 1 for r in results) / n, 4), "draw": round(sum(r == 0 for r in results) / n, 4),
               "p2_win": round(sum(r == -1 for r in results) / n, 4),
               "avg_length": round(float(np.mean(lengths)), 2) if lengths else None,
               "bench_optimal": pb["optimal"], "bench_keeps_nl": pb["keeps_nl"], "bench_value_acc": pb["value_acc"],
               "bench_by_bucket": pb["by_bucket"], "empty_board_wdl": [round(float(x), 4) for x in wdl0[0]],
               "train_sec": round(t_train, 1)}
        if a.eval_every and (it % a.eval_every == 0 or it == a.iterations):
            lad = ladder(("nn", run.path("model.npz"), "policy"), depths, a.eval_openings, a.workers)
            rec["vs_minimax"], rec["elo"] = lad["vs"], lad["elo"]
        rec["eval_sec"] = round(time.time() - t1, 1)
        rec["elapsed_sec"] = round(time.time() - t_run, 1)
        rec.update(meter.snapshot())
        run.log_metrics(rec)
        meter.write(run.dir)
        msg = "iter %d/%d  games %d  eps %.3f  loss %.4f (q %.4f v %.3f) |Q| %.2f  len %s  bench optimal %.3f value %.3f  [%.0fs]" % (
            it, a.iterations, total_games, eps, m[0], m[1], m[2], m[3], rec["avg_length"], pb["optimal"],
            pb["value_acc"], t_train)
        if "elo" in rec:
            msg += "\n      vs minimax: " + "  ".join("%s %.2f" % (k, v["score"]) for k, v in rec["vs_minimax"].items())
            msg += "   Elo~%s" % rec["elo"]
        log(msg)
        model.train()
    meter.stop()
    meter.write(run.dir)
    log("done. Next: python -m nn.evaluate --run %s   and   python -m rl.compare" % a.run)


if __name__ == "__main__":
    main()
