"""Supervised training of one sweep student on one teacher's cached labels.

Used by research/sweep.py.  The loop mirrors nn/train_supervised.py on purpose
(same encoding, mirror augmentation, AdamW + OneCycle, batch 1024, lr 2e-3, wd 1e-4,
bf16 autocast on CUDA, best.npz = lowest validation loss) with two differences:

* targets come from a teacher label file (research/teachers.py), so the policy and
  WDL targets may be soft: loss = CE(teacher policy, student policy)
  + value_weight * CE(teacher WDL, student WDL).  For the ``pons`` teacher the targets
  are exactly the ones train_supervised builds, so the loss is identical.
* the train/validation split is FIXED (split seed 0, 5 % validation) for every run,
  whatever the training seed - it is the split runs/sup_6x64 used, so the 6x64
  network used as a distillation teacher never saw the validation positions either.

Selection of best.npz uses the validation loss AGAINST THE TEACHER, never the solver
labels, so a non-solver teacher's student is not tuned on perfect-play information.
"""

import math
import os
import time

import numpy as np

from nn.common import NN_DATA, Run, machine_info, now_iso

DATASET = os.path.join(NN_DATA, "solver_300k.npz")
ILLEGAL = -128
SPLIT_SEED = 0
VAL_FRAC = 0.05

_DATA = {}


def load_base(path=DATASET):
    """Positions + solver information shared by every run (cached in-process)."""
    if path in _DATA:
        return _DATA[path]
    from nn.encode import bits_to_grid
    z = np.load(path)
    cur, mask, scores = z["cur"], z["mask"], z["scores"].astype(np.int16)
    n = len(cur)
    legal = scores != ILLEGAL
    best = np.where(legal, scores, -1000).max(1)
    opt = (scores == best[:, None]) & legal
    outcome = np.where(best > 0, 0, np.where(best == 0, 1, 2)).astype(np.int64)
    me = bits_to_grid(cur)
    them = bits_to_grid(cur ^ mask)
    grids = np.stack([me, them], 1)
    first = me.reshape(n, -1).sum(1) == them.reshape(n, -1).sum(1)
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    nv = int(n * VAL_FRAC)
    base = {"grids": grids, "first": first, "legal": legal, "opt": opt, "outcome": outcome,
            "ply": z["ply"], "train": perm[nv:], "val": perm[:nv], "n": n}
    _DATA[path] = base
    return base


def teacher_best_set(pol):
    """Columns the teacher rates best (ties for hard targets, argmax for soft ones)."""
    return pol >= (pol.max(1, keepdims=True) - 1e-6)


def _batch(D, idx, mirror):
    import torch
    g = D["grids"][idx].float()
    b = g.shape[0]
    ones = torch.ones((b, 1, 6, 7), device=g.device)
    first = D["first"][idx].float().view(b, 1, 1, 1).expand(b, 1, 6, 7)
    x = torch.cat([g, ones, first], 1)
    out = {k: D[k][idx] for k in ("pol", "wdl", "legal", "opt", "tbest")}
    out["outcome"] = D["outcome"][idx]
    if mirror is not None:
        x = torch.where(mirror.view(b, 1, 1, 1), x.flip(3), x)
        for k in ("pol", "legal", "opt", "tbest"):
            out[k] = torch.where(mirror.view(b, 1), out[k].flip(1), out[k])
    return x, out


def train(name, teacher_id, labels, blocks, channels, seed, epochs, extra_cfg=None,
          batch=1024, lr=2e-3, weight_decay=1e-4, value_weight=1.0, device="auto",
          logf=print, data_path=DATASET):
    """Train runs/<name>.  Returns a dict with timing and final validation numbers."""
    import torch
    import torch.nn.functional as F
    from nn import bench
    from nn.common import TorchEvaluator, pick_device
    from nn.export import export_model
    from nn.model import build, count_params, save_checkpoint

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = pick_device(device)
    base = load_base(data_path)
    run = Run(name)
    model = build({"blocks": blocks, "channels": channels}).to(dev)
    params = count_params(model)
    cfg = {"phase": "supervised", "created": now_iso(), "device": str(dev),
           "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
           "args": {"data": os.path.relpath(data_path, os.path.dirname(NN_DATA)), "run": name,
                    "epochs": epochs, "batch": batch, "lr": lr, "weight_decay": weight_decay,
                    "value_weight": value_weight, "blocks": blocks, "channels": channels,
                    "val_frac": VAL_FRAC, "split_seed": SPLIT_SEED, "seed": seed,
                    "teacher": teacher_id, "mcts_every": 0},
           "model": model.config, "params": params, "train_positions": int(len(base["train"])),
           "val_positions": int(len(base["val"])), "machine": machine_info(),
           "research": dict(extra_cfg or {}, teacher=teacher_id)}
    run.write_config(cfg)
    logf("  train %s on %s (%s): %d params, %d epochs" % (name, dev, cfg["device_name"], params, epochs))

    pol = labels["pol"].astype(np.float32)
    D = {"grids": base["grids"], "first": base["first"], "legal": base["legal"], "opt": base["opt"],
         "outcome": base["outcome"], "pol": pol, "wdl": labels["wdl"].astype(np.float32),
         "tbest": teacher_best_set(pol) & base["legal"]}
    Dd = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in D.items()}
    tr_t = torch.from_numpy(base["train"]).to(dev)
    va_t = torch.from_numpy(base["val"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(tr_t) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_per_epoch * epochs,
                                                pct_start=0.1)
    use_amp = dev.type == "cuda"
    bench_rows = bench.load_accuracy_bench()
    best_val, best_rec = float("inf"), None
    step = 0
    t_start = time.time()
    train_only = 0.0

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        sums = torch.zeros(3, device=dev)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            mirror = torch.rand(len(idx), device=dev) < 0.5
            x, t = _batch(Dd, idx, mirror)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                p_logits, v_logits = model(x)
            lp = -(t["pol"] * F.log_softmax(p_logits.float(), 1)).sum(1).mean()
            lv = -(t["wdl"] * F.log_softmax(v_logits.float(), 1)).sum(1).mean()
            loss = lp + value_weight * lv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            sums += torch.stack([loss.detach(), lp.detach(), lv.detach()]) * len(idx)
        sync()
        train_only += time.time() - t0
        sums = (sums / len(perm)).cpu().numpy()

        model.eval()
        vs = np.zeros(3)
        pacc = vacc = tagree = 0
        with torch.inference_mode():
            for i in range(0, len(va_t), 8192):
                idx = va_t[i:i + 8192]
                x, t = _batch(Dd, idx, None)
                p_logits, v_logits = model(x)
                lp = -(t["pol"] * F.log_softmax(p_logits, 1)).sum(1)
                lv = -(t["wdl"] * F.log_softmax(v_logits, 1)).sum(1)
                vs += [(lp + value_weight * lv).sum().item(), lp.sum().item(), lv.sum().item()]
                choice = p_logits.masked_fill(~t["legal"], float("-inf")).argmax(1)[:, None]
                pacc += t["opt"].gather(1, choice).sum().item()
                tagree += t["tbest"].gather(1, choice).sum().item()
                vacc += (v_logits.argmax(1) == t["outcome"]).sum().item()
        nv = max(len(va_t), 1)
        vs /= nv
        pb = bench.policy_eval(TorchEvaluator(model, dev), bench_rows)
        rec = {"type": "epoch", "phase": "supervised", "epoch": epoch, "step": step,
               "lr": sched.get_last_lr()[0], "positions_seen": step * batch,
               "train_loss": round(float(sums[0]), 5), "train_policy_loss": round(float(sums[1]), 5),
               "train_value_loss": round(float(sums[2]), 5),
               "val_loss": round(vs[0], 5), "val_policy_loss": round(vs[1], 5),
               "val_value_loss": round(vs[2], 5),
               "val_policy_acc": round(pacc / nv, 4), "val_value_acc": round(vacc / nv, 4),
               "val_teacher_agree": round(tagree / nv, 4),
               "bench_optimal": pb["optimal"], "bench_keeps_nl": pb["keeps_nl"],
               "bench_value_acc": pb["value_acc"], "bench_by_bucket": pb["by_bucket"],
               "epoch_sec": round(time.time() - t0, 1), "elapsed_sec": round(time.time() - t_start, 1)}
        run.log_metrics(rec)
        meta = {"run": name, "phase": "supervised", "epoch": epoch, "val_loss": rec["val_loss"],
                "bench_optimal": rec["bench_optimal"], "teacher": teacher_id, "exported": now_iso()}
        export_model(model.eval(), run.path("model.npz"), meta)
        if vs[0] < best_val:
            best_val, best_rec = vs[0], rec
            export_model(model, run.path("best.npz"), meta)
            save_checkpoint(os.path.join(run.ckpt_dir, "best.pt"), model, {"epoch": epoch})
        logf("    %s ep %d/%d loss %.4f val %.4f pons-acc %.3f teacher-agree %.3f bench %.3f/%.3f %.0fs" % (
            name, epoch, epochs, sums[0], vs[0], rec["val_policy_acc"], rec["val_teacher_agree"],
            pb["optimal"], pb["keeps_nl"], rec["epoch_sec"]))
    total = time.time() - t_start
    return {"params": params, "device": str(dev), "device_name": cfg["device_name"],
            "train_sec": round(total, 1), "train_loop_sec": round(train_only, 1),
            "gpu_sec": round(train_only, 1) if dev.type == "cuda" else 0.0,
            "best_epoch": best_rec["epoch"], "best_val_loss": best_rec["val_loss"],
            "val_teacher_agree": best_rec["val_teacher_agree"],
            "val_pons_acc": best_rec["val_policy_acc"], "val_value_acc": best_rec["val_value_acc"],
            "epochs": epochs, "finished": now_iso()}
