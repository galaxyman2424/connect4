"""Phase A: supervised training on perfect-play labels.

    python -m nn.train_supervised --data nn_data/solver_positions.npz --run sup_6x64 --epochs 30

The network sees a position and is trained to predict
  * policy: which columns are optimal (target = uniform over the solver's
    best columns)                                   -> cross-entropy
  * value:  win / draw / loss for the side to move under perfect play
                                                    -> cross-entropy
Loss = policy_loss + value_weight * value_loss, AdamW, cosine learning-rate
decay, random left-right mirroring as data augmentation (Connect 4 is
symmetric, so a mirrored position with a mirrored target is just as true).

Each epoch it logs to runs/<run>/metrics.jsonl:
  train/val losses, val policy accuracy (argmax is an optimal move), val
  value accuracy, and the held-out benchmark (data/accuracy_positions.csv):
  bench_optimal, bench_keeps_nl, bench_value_acc (+ MCTS accuracy every
  --mcts-every epochs).  It exports runs/<run>/model.npz every epoch and
  best.npz for the epoch with the lowest validation loss.
"""

import argparse
import math
import os
import time

import numpy as np

from .common import NN_DATA, Run, log, machine_info, now_iso, pick_device

ILLEGAL = -128


def load_data(path, val_frac, seed):
    from .encode import bits_to_grid
    z = np.load(path)
    cur, mask, scores = z["cur"], z["mask"], z["scores"].astype(np.int16)
    n = len(cur)
    legal = scores != ILLEGAL
    best = np.where(legal, scores, -1000).max(1)
    opt = (scores == best[:, None]) & legal
    pol = opt / opt.sum(1, keepdims=True)
    val = np.where(best > 0, 0, np.where(best == 0, 1, 2)).astype(np.int64)
    me = bits_to_grid(cur)
    them = bits_to_grid(cur ^ mask)
    grids = np.stack([me, them], 1)                                    # (N,2,6,7) uint8
    first = me.reshape(n, -1).sum(1) == them.reshape(n, -1).sum(1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nv = int(n * val_frac)
    return {"grids": grids, "first": first, "pol": pol.astype(np.float32), "val": val,
            "legal": legal, "opt": opt, "ply": z["ply"]}, perm[nv:], perm[:nv]


def to_device(d, device):
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in d.items()}


def make_batch(D, idx, mirror):
    """Build network input on the device from pre-encoded grids."""
    import torch
    g = D["grids"][idx].float()                                        # (B,2,6,7)
    b = g.shape[0]
    ones = torch.ones((b, 1, 6, 7), device=g.device)
    first = D["first"][idx].float().view(b, 1, 1, 1).expand(b, 1, 6, 7)
    x = torch.cat([g, ones, first], 1)
    pol = D["pol"][idx]
    legal = D["legal"][idx]
    opt = D["opt"][idx]
    if mirror is not None:
        x = torch.where(mirror.view(b, 1, 1, 1), x.flip(3), x)
        pol = torch.where(mirror.view(b, 1), pol.flip(1), pol)
        legal = torch.where(mirror.view(b, 1), legal.flip(1), legal)
        opt = torch.where(mirror.view(b, 1), opt.flip(1), opt)
    return x, pol, D["val"][idx], legal, opt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(NN_DATA, "solver_positions.npz"))
    ap.add_argument("--run", required=True, help="run name -> runs/<run>/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--mcts-every", type=int, default=5, help="MCTS benchmark every N epochs (0 = never)")
    ap.add_argument("--mcts-sims", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from . import bench
    from .common import TorchEvaluator
    from .export import export_model
    from .model import build, count_params, save_checkpoint

    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    run = Run(a.run)
    if run.exists():
        raise SystemExit("runs/%s already exists - choose another --run name" % a.run)
    D, tr, va = load_data(a.data, a.val_frac, a.seed)
    model = build({"blocks": a.blocks, "channels": a.channels}).to(device)
    cfg = {"phase": "supervised", "created": now_iso(), "args": vars(a), "device": str(device),
           "model": model.config, "params": count_params(model), "train_positions": len(tr),
           "val_positions": len(va), "machine": machine_info()}
    run.write_config(cfg)
    log("run %s on %s: %d params, %d train / %d val positions" % (a.run, device, cfg["params"], len(tr), len(va)))

    Dd = to_device(D, device)
    tr_t = torch.from_numpy(tr).to(device)
    va_t = torch.from_numpy(va).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = math.ceil(len(tr) / a.batch)
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)
    use_amp = device.type == "cuda"
    bench_rows = bench.load_accuracy_bench()
    best_val = float("inf")
    step = 0
    t_start = time.time()

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=device)]
        sums = np.zeros(3)
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            mirror = torch.rand(len(idx), device=device) < 0.5
            x, pol, val, _, _ = make_batch(Dd, idx, mirror)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                p_logits, v_logits = model(x)
            lp = -(pol * F.log_softmax(p_logits.float(), 1)).sum(1).mean()
            lv = F.cross_entropy(v_logits.float(), val)
            loss = lp + a.value_weight * lv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            sums += [loss.item() * len(idx), lp.item() * len(idx), lv.item() * len(idx)]
        sums /= len(perm)

        # ---- validation
        model.eval()
        vs = np.zeros(3)
        pacc = vacc = 0
        with torch.inference_mode():
            for i in range(0, len(va_t), 4096):
                idx = va_t[i:i + 4096]
                x, pol, val, legal, optm = make_batch(Dd, idx, None)
                p_logits, v_logits = model(x)
                lp = -(pol * F.log_softmax(p_logits, 1)).sum(1)
                lv = F.cross_entropy(v_logits, val, reduction="none")
                vs += [(lp + a.value_weight * lv).sum().item(), lp.sum().item(), lv.sum().item()]
                choice = p_logits.masked_fill(~legal, float("-inf")).argmax(1)
                pacc += optm.gather(1, choice[:, None]).sum().item()
                vacc += (v_logits.argmax(1) == val).sum().item()
        nv = max(len(va_t), 1)
        vs /= nv
        ev = TorchEvaluator(model, device)
        pb = bench.policy_eval(ev, bench_rows)
        rec = {"type": "epoch", "phase": "supervised", "epoch": epoch, "step": step,
               "lr": sched.get_last_lr()[0], "positions_seen": step * a.batch,
               "train_loss": round(sums[0], 5), "train_policy_loss": round(sums[1], 5),
               "train_value_loss": round(sums[2], 5),
               "val_loss": round(vs[0], 5), "val_policy_loss": round(vs[1], 5), "val_value_loss": round(vs[2], 5),
               "val_policy_acc": round(pacc / nv, 4), "val_value_acc": round(vacc / nv, 4),
               "bench_optimal": pb["optimal"], "bench_keeps_nl": pb["keeps_nl"],
               "bench_value_acc": pb["value_acc"], "bench_by_bucket": pb["by_bucket"],
               "epoch_sec": round(time.time() - t0, 1), "elapsed_sec": round(time.time() - t_start, 1)}
        if a.mcts_every and (epoch % a.mcts_every == 0 or epoch == a.epochs):
            mb = bench.mcts_eval(ev, bench_rows, a.mcts_sims)
            rec["bench_mcts_sims"] = a.mcts_sims
            rec["bench_mcts_optimal"] = mb["optimal"]
            rec["bench_mcts_keeps_nl"] = mb["keeps_nl"]
        model.train()
        run.log_metrics(rec)
        meta = {"run": a.run, "phase": "supervised", "epoch": epoch, "val_loss": rec["val_loss"],
                "bench_optimal": rec["bench_optimal"], "exported": now_iso()}
        save_checkpoint(os.path.join(run.ckpt_dir, "latest.pt"), model,
                        {"epoch": epoch, "step": step, "optimizer": opt.state_dict()})
        export_model(model.eval(), run.path("model.npz"), meta)
        if vs[0] < best_val:
            best_val = vs[0]
            export_model(model, run.path("best.npz"), meta)
            save_checkpoint(os.path.join(run.ckpt_dir, "best.pt"), model, {"epoch": epoch})
        model.train()
        log("epoch %d/%d  loss %.4f (p %.4f v %.4f)  val %.4f  val-policy %.3f  val-value %.3f  "
            "bench optimal %.3f keeps %.3f value %.3f  %.0fs" % (
                epoch, a.epochs, sums[0], sums[1], sums[2], vs[0], rec["val_policy_acc"],
                rec["val_value_acc"], pb["optimal"], pb["keeps_nl"], pb["value_acc"], rec["epoch_sec"]))
    log("done. Next: python -m nn.evaluate --run %s" % a.run)


if __name__ == "__main__":
    main()
