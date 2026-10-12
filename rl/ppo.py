"""PPO (Proximal Policy Optimization) self-play - a policy-gradient method.

    python -m rl.ppo --run ppo_6x64 --iterations 300
    python -m rl.ppo --run ppo_from_sup --init runs/sup_6x64/checkpoints/best.pt   # warm start
    python -m rl.ppo --run ppo_6x64 --resume --iterations 500

Idea
  The policy head IS the player: it outputs a probability for each column and
  moves are sampled from it.  After a batch of games, every move gets an
  "advantage" = how much better the game went than the value head (critic)
  expected at that point (GAE, Schulman et al. 2015).  The policy gradient
  makes moves with positive advantage more likely and the others less likely.
  PPO (Schulman et al. 2017) limits how far one update can move the policy
  (the probability ratio new/old is clipped to [1-0.2, 1+0.2]), which keeps
  training stable; each batch is reused for a few epochs.
  Loss = clipped policy loss + value_weight x CE(WDL head, game result)
         - entropy_coef x entropy (keeps exploring).

  Opponents: with probability --pool-prob a game is played against a frozen
  past copy of the network (a snapshot is saved every --snapshot-every
  iterations, the last --pool-size are kept), otherwise against itself.
  Playing old selves too stops the policy from going round in circles
  (beating only its latest self while forgetting how to beat older styles).

Trade-offs: no search and no replay buffer - simple, the trained policy plays
instantly - but on-policy: every game is used for a few epochs and thrown
away, so it needs a lot of games, and a policy trained without search is
usually weaker than AlphaZero's search-trained one at equal compute.  The
exported model has both heads, so it can still be used with MCTS afterwards
(spec nn:runs/<run>/model.npz:mcts:200) - combining policy gradient with
search at play time.

Outputs: runs/<run>/model.npz, checkpoints/latest.pt, metrics.jsonl, cost.json.
"""

import argparse
import copy
import os
import time

import numpy as np

from nn.common import Run, log, machine_info, now_iso, pick_device

from . import vecenv as V
from .meter import CostMeter


def gae(seq_values, final_reward, lam, gamma=1.0):
    """One player's decisions in one game: values V_0..V_{T-1}, reward only at the
    end (+1 win / 0 draw / -1 loss).  Returns advantages A_0..A_{T-1}."""
    T = len(seq_values)
    adv = np.zeros(T)
    nxt_v, nxt_a = 0.0, 0.0
    for t in range(T - 1, -1, -1):
        r = final_reward if t == T - 1 else 0.0
        d = r + gamma * nxt_v - seq_values[t]
        nxt_a = d + gamma * lam * nxt_a
        adv[t] = nxt_a
        nxt_v = seq_values[t]
    return adv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="start from these weights (.pt), e.g. a supervised run")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--games", type=int, default=2048, help="games per iteration")
    ap.add_argument("--parallel", type=int, default=1024, help="games played at once")
    ap.add_argument("--epochs", type=int, default=4, help="passes over each batch")
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--target-kl", type=float, default=0.03, help="stop the epochs early above this KL")
    ap.add_argument("--pool-prob", type=float, default=0.3, help="share of games vs a past snapshot")
    ap.add_argument("--pool-size", type=int, default=10)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=10)
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
    from .torchutil import amp_ctx, build_or_init, legal_t, masked_log_softmax, planes

    run = Run(a.run)
    device = pick_device(a.device)
    torch.manual_seed(a.seed)
    latest = os.path.join(run.ckpt_dir, "latest.pt")
    pool_path = os.path.join(run.ckpt_dir, "pool.pt")
    if run.exists() and not a.resume:
        raise SystemExit("runs/%s exists - use --resume or another --run" % a.run)
    start_it, total_games, pool = 1, 0, []
    if a.resume and os.path.isfile(latest):
        model, obj = load_checkpoint(latest, device)
        start_it = obj.get("iteration", 0) + 1
        total_games = obj.get("total_games", 0)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        if "optimizer" in obj:
            opt.load_state_dict(obj["optimizer"])
        if os.path.isfile(pool_path):
            pool = torch.load(pool_path, map_location="cpu", weights_only=True)
        log("resuming %s at iteration %d (%d snapshots in the pool)" % (a.run, start_it, len(pool)))
    else:
        model = build_or_init(a.blocks, a.channels, a.init, device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        run.write_config({"phase": "ppo", "created": now_iso(), "args": vars(a), "device": str(device),
                          "model": model.config, "params": count_params(model), "machine": machine_info()})
    opp_model = copy.deepcopy(model).eval()
    log("run %s on %s: %d params" % (a.run, device, count_params(model)))

    meter = CostMeter(uses_gpu=device.type == "cuda", label="rl.ppo " + a.run).start()
    rng = np.random.default_rng(a.seed + start_it)
    rows = bench.load_accuracy_bench()
    depths = [int(x) for x in a.eval_depths.split(",") if x]
    t_run = time.time()

    def policy(net, curs, masks):
        with torch.no_grad(), amp_ctx(device):
            p, v = net(planes(curs, masks, device))
        lg = legal_t(masks, device)
        logp = masked_log_softmax(p, lg)
        wdl = torch.softmax(v.float(), 1)
        return logp.exp().cpu().numpy(), logp.cpu().numpy(), (wdl[:, 0] - wdl[:, 2]).cpu().numpy()

    for it in range(start_it, a.iterations + 1):
        t0 = time.time()
        model.eval()
        # ------------------------------------------------------------ rollouts
        D = {k: [] for k in ("cur", "mask", "act", "logp", "adv", "z", "val")}
        lengths, results, vs_pool = [], [], []
        played = 0
        while played < a.games:
            B = min(a.parallel, a.games - played)
            g = V.VecGames(B, rng, random_opening=(0, 2))
            use_pool = (rng.random() < a.pool_prob) and len(pool) > 0
            if use_pool:
                opp_model.load_state_dict(pool[int(rng.integers(len(pool)))])
                learner = np.where(rng.random(B) < 0.5, 1, -1)          # colour the learner plays
            else:
                learner = np.zeros(B, dtype=np.int64)                   # self-play: learn from both sides
            start_ply = g.ply.copy()
            steps = []                                                  # (game, colour, cur, mask, act, logp, val)
            alive = np.ones(B, dtype=bool)
            won_by = np.zeros(B, dtype=np.int64)                        # +1 first player, -1 second, 0 draw
            while alive.any():
                idx = np.nonzero(alive)[0]
                colour = np.where(g.ply[idx] % 2 == 0, 1, -1)
                mine = (learner[idx] == 0) | (learner[idx] == colour)
                cols = np.zeros(len(idx), dtype=np.int64)
                if mine.any():
                    li = idx[mine]
                    pr, lp, val = policy(model, g.cur[li], g.mask[li])
                    c = V.masked_sample(pr, V.legal(g.mask[li]), rng)
                    cols[mine] = c
                    for k, gi in enumerate(li):
                        steps.append((gi, colour[mine][k], g.cur[gi], g.mask[gi], c[k], lp[k, c[k]], val[k]))
                if (~mine).any():
                    oi = idx[~mine]
                    pr, _, _ = policy(opp_model, g.cur[oi], g.mask[oi])
                    cols[~mine] = V.masked_sample(pr, V.legal(g.mask[oi]), rng)
                won, full = g.step(cols, idx)
                done = won | full
                for k in np.nonzero(done)[0]:
                    gi = idx[k]
                    alive[gi] = False
                    won_by[gi] = 0 if not won[k] else (1 if g.ply[gi] % 2 == 1 else -1)
                    lengths.append(int(g.ply[gi]))
                    results.append(int(won_by[gi]))
            played += B
            if use_pool:
                vs_pool += [(1.0 if won_by[i] == learner[i] else 0.5 if won_by[i] == 0 else 0.0) for i in range(B)]
            # ---- advantages per (game, colour) sequence
            seqs = {}
            for s in steps:
                seqs.setdefault((s[0], s[1]), []).append(s)
            for (gi, col), seq in seqs.items():
                res = 0.0 if won_by[gi] == 0 else (1.0 if won_by[gi] == col else -1.0)
                adv = gae(np.array([s[6] for s in seq]), res, a.lam)
                zc = 1 if res == 0 else (0 if res > 0 else 2)
                for s, ad in zip(seq, adv):
                    D["cur"].append(s[2]); D["mask"].append(s[3]); D["act"].append(s[4])
                    D["logp"].append(s[5]); D["val"].append(s[6]); D["adv"].append(ad); D["z"].append(zc)
        total_games += played
        t_roll = time.time() - t0

        # ------------------------------------------------------------- update
        t1 = time.time()
        model.train()
        N = len(D["act"])
        cur = np.array(D["cur"], dtype=np.uint64)
        mask = np.array(D["mask"], dtype=np.uint64)
        X = planes(cur, mask, device)
        LG = legal_t(mask, device)
        ACT = torch.tensor(D["act"], device=device)
        OLD = torch.tensor(D["logp"], device=device, dtype=torch.float32)
        adv = np.array(D["adv"], dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ADV = torch.from_numpy(adv).to(device)
        Z = torch.tensor(D["z"], device=device)
        stats, n_mb, stopped = np.zeros(6), 0, False
        for ep in range(a.epochs):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, a.minibatch):
                mb = perm[i:i + a.minibatch]
                with amp_ctx(device):
                    p, v = model(X[mb])
                logp_all = masked_log_softmax(p, LG[mb])
                logp = logp_all.gather(1, ACT[mb][:, None]).squeeze(1)
                ratio = torch.exp(logp - OLD[mb])
                s1 = ratio * ADV[mb]
                s2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * ADV[mb]
                lpol = -torch.min(s1, s2).mean()
                lval = F.cross_entropy(v.float(), Z[mb])
                ent = -(logp_all.exp() * logp_all).sum(1).mean()
                loss = lpol + a.value_weight * lval - a.entropy * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    kl = (OLD[mb] - logp).mean().item()
                    clipfrac = ((ratio - 1).abs() > a.clip).float().mean().item()
                stats += [loss.item(), lpol.item(), lval.item(), ent.item(), kl, clipfrac]
                n_mb += 1
            if a.target_kl and kl > a.target_kl:
                stopped = True
                break
        del X, LG
        t_upd = time.time() - t1
        model.eval()
        if it % a.snapshot_every == 0:
            pool.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            pool = pool[-a.pool_size:]
            torch.save(pool, pool_path)
        save_checkpoint(latest, model, {"iteration": it, "total_games": total_games, "optimizer": opt.state_dict()})
        export_model(model, run.path("model.npz"), {"run": a.run, "phase": "ppo", "gen": it,
                                                    "total_games": total_games, "exported": now_iso()})
        t2 = time.time()
        ev = TorchEvaluator(model, device)
        pb = bench.policy_eval(ev, rows)
        pri, _, wdl0 = ev.evaluate([0], [0])
        m = stats / max(n_mb, 1)
        n = max(len(results), 1)
        rec = {"type": "generation", "phase": "ppo", "gen": it, "games": played, "total_games": total_games,
               "positions": N, "train_loss": round(float(m[0]), 5), "policy_loss": round(float(m[1]), 5),
               "value_loss": round(float(m[2]), 5), "entropy": round(float(m[3]), 4), "approx_kl": round(float(m[4]), 5),
               "clip_frac": round(float(m[5]), 4), "early_stop": stopped,
               "p1_win": round(sum(r == 1 for r in results) / n, 4), "draw": round(sum(r == 0 for r in results) / n, 4),
               "p2_win": round(sum(r == -1 for r in results) / n, 4), "avg_length": round(float(np.mean(lengths)), 2),
               "score_vs_pool": round(float(np.mean(vs_pool)), 4) if vs_pool else None, "pool": len(pool),
               "bench_optimal": pb["optimal"], "bench_keeps_nl": pb["keeps_nl"], "bench_value_acc": pb["value_acc"],
               "bench_by_bucket": pb["by_bucket"], "empty_board_policy": [round(float(x), 4) for x in pri[0]],
               "empty_board_wdl": [round(float(x), 4) for x in wdl0[0]],
               "selfplay_sec": round(t_roll, 1), "train_sec": round(t_upd, 1)}
        if a.eval_every and (it % a.eval_every == 0 or it == a.iterations):
            lad = ladder(("nn", run.path("model.npz"), "policy"), depths, a.eval_openings, a.workers)
            rec["vs_minimax"], rec["elo"] = lad["vs"], lad["elo"]
        rec["eval_sec"] = round(time.time() - t2, 1)
        rec["elapsed_sec"] = round(time.time() - t_run, 1)
        rec.update(meter.snapshot())
        run.log_metrics(rec)
        meter.write(run.dir)
        msg = "iter %d/%d  games %d  len %.1f  P1 %.0f%%  loss %.3f (pi %.3f v %.3f) ent %.2f kl %.4f  " \
              "bench optimal %.3f value %.3f  [rollout %.0fs, update %.0fs]" % (
                  it, a.iterations, total_games, rec["avg_length"], 100 * rec["p1_win"], m[0], m[1], m[2], m[3], m[4],
                  pb["optimal"], pb["value_acc"], t_roll, t_upd)
        if "elo" in rec:
            msg += "\n      vs minimax: " + "  ".join("%s %.2f" % (k, v["score"]) for k, v in rec["vs_minimax"].items())
            msg += "   Elo~%s" % rec["elo"]
        log(msg)
    meter.stop()
    meter.write(run.dir)
    log("done. Next: python -m nn.evaluate --run %s   and   python -m rl.compare" % a.run)


if __name__ == "__main__":
    main()
