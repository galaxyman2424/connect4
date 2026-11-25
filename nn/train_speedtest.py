"""Measure training throughput on THIS GPU for each --speed level and batch size.

    python -m nn.train_speedtest                         # 8x128 score net, ~3-5 min on the RTX 3050
    python -m nn.train_speedtest --blocks 6 --channels 64 --steps 300
    python -m nn.train_speedtest --batches 1024,4096 --speeds off,safe,max

Runs the real nn/train_score.py training step (same losses, bf16 autocast, AdamW + one-cycle,
EMA, random mirroring) on random synthetic positions held on the GPU, so only the GPU work
is timed.  "off+sync" is the OLD loop (5 .item() GPU syncs per step); the others keep the
loss sums on the GPU (what the training scripts do now).  Prints positions/second; the
fastest row is what to use (--speed and --batch of nn.train_score / nn.train_supervised).
A bigger --batch usually wants a proportionally bigger --lr (e.g. 4096 -> --lr 4e-3 to 6e-3);
check that BENCH accuracy does not drop before switching for good.
Writes experiments/results/speedtest_<blocks>x<channels>.json.
"""

import argparse
import json
import os
import time

import numpy as np

from .common import RESULTS, log, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--batches", default="1024,2048,4096")
    ap.add_argument("--speeds", default="off+sync,off,safe,max")
    ap.add_argument("--steps", type=int, default=200, help="timed steps per configuration (after warm-up)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    import torch
    from . import fasttrain
    from .model import build
    from .train_score import losses

    device = pick_device(a.device)
    if device.type != "cuda":
        log("WARNING: no CUDA GPU - numbers are CPU numbers and --speed has no effect")
    n = 200_000
    rng = np.random.default_rng(0)
    grids = torch.from_numpy(rng.integers(0, 2, (n, 2, 6, 7), dtype=np.uint8)).to(device)
    first = torch.from_numpy(rng.integers(0, 2, n).astype(bool)).to(device)
    sc = torch.from_numpy(rng.integers(-18, 19, (n, 7)).astype(np.float32)).to(device)
    legal = torch.ones((n, 7), dtype=torch.bool, device=device)
    val = torch.from_numpy(rng.integers(0, 3, n)).to(device)
    wts = torch.ones(n, device=device)

    class A:  # loss weights as in nn.train_score defaults
        policy_temp = rank_temp = 1.0
        value_weight, score_weight, rank_weight = 0.5, 1.0, 1.0

    results = []
    for speed in [s for s in a.speeds.split(",") if s]:
        level = "off" if speed == "off+sync" else speed
        for batch in [int(b) for b in a.batches.split(",") if b]:
            torch.manual_seed(0)
            # reset global flags so each level is measured on its own
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            fasttrain.configure(level, device)
            model = build({"blocks": a.blocks, "channels": a.channels, "score_head": True}).to(device)
            model, fwd = fasttrain.prepare_model(level, model, device, "forward_all")
            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
            ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            total = a.warmup + a.steps
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total + 1, pct_start=0.05)
            model.train()
            sums = torch.zeros(5, device=device)
            t0 = None
            try:
                for step in range(total):
                    if step == a.warmup:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                    idx = torch.randint(0, n, (batch,), device=device)
                    g = grids[idx].float()
                    m = torch.rand(batch, device=device) < 0.5
                    x = torch.cat([g, torch.ones((batch, 1, 6, 7), device=device),
                                   first[idx].float().view(batch, 1, 1, 1).expand(batch, 1, 6, 7)], 1)
                    x = torch.where(m.view(batch, 1, 1, 1), x.flip(3), x)
                    x = fasttrain.to_input(level, x)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        loss, parts = losses(model, x, sc[idx], legal[idx], val[idx], wts[idx], A, fwd=fwd)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    sched.step()
                    ema.update_parameters(model)
                    if speed == "off+sync":
                        _ = [loss.item()] + parts.tolist()            # old loop: syncs every step
                    else:
                        sums += torch.cat([loss.detach().float().view(1), parts.float()])
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                r = {"speed": speed, "batch": batch, "steps": a.steps, "sec": round(dt, 2),
                     "positions_per_sec": round(a.steps * batch / dt),
                     "peak_mib": round(torch.cuda.max_memory_allocated(device) / 2 ** 20) if device.type == "cuda" else None}
            except Exception as e:  # noqa: BLE001
                r = {"speed": speed, "batch": batch, "error": "%s: %s" % (type(e).__name__, e)}
            results.append(r)
            log("%-9s batch %5d  %s" % (speed, batch, ("%9d positions/s  (%.1fs, peak %s MiB)" % (
                r["positions_per_sec"], r["sec"], r["peak_mib"])) if "error" not in r else r["error"]))
            del model, ema, opt
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
    ok = [r for r in results if "error" not in r]
    if ok:
        base = next((r for r in ok if r["speed"] == "off+sync" and r["batch"] == 1024), ok[0])
        best = max(ok, key=lambda r: r["positions_per_sec"])
        log("fastest: --speed %s --batch %d = %.2fx the old loop at batch 1024 (%d positions/s)" % (
            "off" if best["speed"] == "off+sync" else best["speed"], best["batch"],
            best["positions_per_sec"] / base["positions_per_sec"], best["positions_per_sec"]))
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "speedtest_%dx%d.json" % (a.blocks, a.channels))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "torch": torch.__version__, "results": results}, f, indent=2)
    log("wrote %s" % out)


if __name__ == "__main__":
    main()
