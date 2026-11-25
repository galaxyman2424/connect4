"""Supervised training with a per-column SCORE head (aim: 95%+ optimal moves).

    python -m nn.train_score --data nn_data/solver_3m.npz --run score_8x128_s0 --blocks 8 --channels 128
    python -m nn.train_score --data nn_data/solver_3m.npz --hard nn_data/hard_r1.npz \
        --init runs/score_8x128_s0/checkpoints/best.pt --run score_8x128_s0_r1 --epochs 12 --lr 7e-4

Why: the old nets (nn/train_supervised.py) learn "uniform over the best columns" plus
win/draw/loss.  Nothing tells them how FAST a win is, and at 93% optimal most remaining
errors are "right result, slower win" (docs/ACCURACY_95.md).  The solver labels already
contain all 7 exact scores, so this script trains three heads on them:

  policy  cross-entropy to softmax(score / --policy-temp) over legal columns
          (soft target: the best column gets most mass, faster wins rank above slower ones)
  value   win / draw / loss, as before (used by MCTS and the evaluation bar)
  score   7 outputs = the Pons score of each column, two losses:
            smooth-L1 regression on score / 10
            listwise cross-entropy softmax(pred / tau) vs softmax(score / tau): gets the ORDER right

Also: random left-right mirroring, AdamW + one-cycle cosine LR, an EMA copy of the weights
(evaluated and exported instead of the raw weights), optional hard positions (--hard, weighted),
and --init to start from any checkpoint (an old net without a score head works: the new head
starts from scratch).

Which head plays is chosen on VALIDATION positions of plies 8-30 (never on the benchmark):
policy / score / score + w * log(policy).  The choice goes into model.npz meta, so the
website, arena and MCTS use it automatically (nn/numpy_net.py), with mirror averaging on.

Each epoch logs to runs/<run>/metrics.jsonl the benchmark (data/accuracy_positions.csv,
held out) for every head, with and without mirror averaging.  best.npz = best validation
"optimal" of the chosen head.
"""

import argparse
import copy
import math
import os
import time

import numpy as np

from .common import NN_DATA, Run, log, machine_info, now_iso, pick_device
from .fasttrain import SPEEDS

ILLEGAL = -128
SCORE_SCALE = 10.0
COMBO_W = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)     # score + w * log(policy), w in Pons points


# ------------------------------------------------------------------ data
def _load_npz(path):
    z = np.load(path)
    return z["cur"].astype(np.uint64), z["mask"].astype(np.uint64), z["scores"].astype(np.int8), z["ply"].astype(np.uint8)


def load_data(paths, hard_paths, hard_weight, val_frac, val_max, seed):
    """Concatenate datasets, drop duplicate positions (hard copies win), split train/val."""
    from .encode import bits_to_grid
    parts = []
    for p in paths:
        parts.append(_load_npz(p) + (np.ones(len(np.load(p)["cur"]), np.float32),))
    for p in hard_paths:
        c, m, s, pl = _load_npz(p)
        parts.append((c, m, s, pl, np.full(len(c), hard_weight, np.float32)))
    cur = np.concatenate([q[0] for q in parts])
    mask = np.concatenate([q[1] for q in parts])
    scores = np.concatenate([q[2] for q in parts]).astype(np.int16)
    ply = np.concatenate([q[3] for q in parts])
    w = np.concatenate([q[4] for q in parts])
    # dedupe on (cur, mask); keep the highest weight
    key = np.stack([cur, mask], 1).view([("c", np.uint64), ("m", np.uint64)]).ravel()
    order = np.lexsort((-w, mask, cur))
    k_sorted = key[order]
    first = np.ones(len(order), bool)
    first[1:] = k_sorted[1:] != k_sorted[:-1]
    keep = np.sort(order[first])
    n_dup = len(cur) - len(keep)
    cur, mask, scores, ply, w = cur[keep], mask[keep], scores[keep], ply[keep], w[keep]
    n = len(cur)

    legal = scores != ILLEGAL
    best = np.where(legal, scores, -1000).max(1)
    opt = (scores == best[:, None]) & legal
    val = np.where(best > 0, 0, np.where(best == 0, 1, 2)).astype(np.int64)
    me = bits_to_grid(cur)
    them = bits_to_grid(cur ^ mask)
    grids = np.stack([me, them], 1)
    firstp = me.reshape(n, -1).sum(1) == them.reshape(n, -1).sum(1)
    # validation bucket = a fixed hash of the position, so every seed, every dataset size and
    # every hard-mining round holds out the SAME positions (no leakage when using --init)
    from .encode import mirror_bits
    mc, mm = mirror_bits(cur), mirror_bits(mask)
    flip = (mc < cur) | ((mc == cur) & (mm < mask))          # hash the canonical (mirror-free) form
    kc, km = np.where(flip, mc, cur), np.where(flip, mm, mask)
    with np.errstate(over="ignore"):
        h = (kc * np.uint64(0x9E3779B97F4A7C15)) ^ (km * np.uint64(0xC2B2AE3D27D4EB4F))
    in_val = (h >> np.uint64(40)) % np.uint64(10000) < np.uint64(int(val_frac * 10000))
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(~in_val)
    rng.shuffle(tr)
    va = np.flatnonzero(in_val & (w == 1.0))      # measured on the normal distribution only
    if len(va) > val_max:
        va = np.sort(np.random.default_rng(0).choice(va, val_max, replace=False))
    D = {"grids": grids, "first": firstp, "scores": np.where(legal, scores, 0).astype(np.float32),
         "legal": legal, "opt": opt, "val": val, "w": w, "ply": ply.astype(np.int64)}
    info = {"positions": n, "duplicates_dropped": int(n_dup), "hard_positions": int((w > 1).sum()),
            "train": len(tr), "val": len(va)}
    return D, tr, va, info


def fit_input_pca(grids, k, seed=0, max_rows=500_000):
    """PCA of the 84 stone cells (planes me/them, flattened like nn/encode.py) of the
    training positions.  Mirrored copies are added because training mirrors at random.
    Returns mean (84,), components U (84, k) and the explained variance."""
    rng = np.random.default_rng(seed)
    g = grids if len(grids) <= max_rows else grids[np.sort(rng.choice(len(grids), max_rows, replace=False))]
    X = np.concatenate([g.reshape(len(g), -1), g[..., ::-1].reshape(len(g), -1)]).astype(np.float64)
    mean = X.mean(0)
    Xc = X - mean
    cov = Xc.T @ Xc / (len(Xc) - 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    k = int(min(max(k, 1), X.shape[1]))
    return {"k": k, "mean": mean.astype(np.float32), "components": evecs[:, :k].astype(np.float32),
            "explained_variance": round(float(evals[:k].sum() / evals.sum()), 5),
            "explained_by_component": [round(float(e / evals.sum()), 5) for e in evals[:k]],
            "fit_positions": int(len(g))}


def teacher_targets(paths, D, device, batch=8192):
    """Ensemble teacher outputs for every position of D: mean over nets (each mirror-averaged)
    of the policy-head probabilities over legal columns and of the score head (Pons units).
    Teachers are exported .npz files (score-head nets; older nets give policy only)."""
    import torch
    from .torch_net import TorchNpzNet
    nets = [TorchNpzNet(p, device) for p in paths]
    n = len(D["grids"])
    t_pol = np.zeros((n, 7), np.float32)
    t_sc = np.zeros((n, 7), np.float32)
    n_sc = sum(1 for t in nets if t.has_score)
    g_all = torch.from_numpy(D["grids"])
    f_all = torch.from_numpy(D["first"])
    l_all = torch.from_numpy(D["legal"])
    with torch.inference_mode():
        for i in range(0, n, batch):
            g = g_all[i:i + batch].to(device).float()
            b = g.shape[0]
            x = torch.cat([g, torch.ones((b, 1, 6, 7), device=device),
                           f_all[i:i + batch].to(device).float().view(b, 1, 1, 1).expand(b, 1, 6, 7)], 1)
            legal = l_all[i:i + batch].to(device)
            pol = torch.zeros((b, 7), device=device)
            sc = torch.zeros((b, 7), device=device)
            for t in nets:
                p1, _, s1 = t.forward_all(x)
                p2, _, s2 = t.forward_all(x.flip(3))
                logit = (p1 + p2.flip(1)) / 2
                pol += torch.softmax(logit.masked_fill(~legal, float("-inf")), 1)
                if s1 is not None:
                    sc += (s1 + s2.flip(1)) / 2
            t_pol[i:i + b] = (pol / len(nets)).cpu().numpy()
            if n_sc:
                t_sc[i:i + b] = (sc / n_sc).masked_fill(~legal, 0).cpu().numpy()
    return t_pol, t_sc, n_sc


def to_device(D, device):
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in D.items()}


def make_batch(D, idx, mirror):
    import torch
    g = D["grids"][idx].float()
    b = g.shape[0]
    ones = torch.ones((b, 1, 6, 7), device=g.device)
    first = D["first"][idx].float().view(b, 1, 1, 1).expand(b, 1, 6, 7)
    x = torch.cat([g, ones, first], 1)
    sc, legal, opt = D["scores"][idx], D["legal"][idx], D["opt"][idx]
    if mirror is not None:
        m4, m2 = mirror.view(b, 1, 1, 1), mirror.view(b, 1)
        x = torch.where(m4, x.flip(3), x)
        sc = torch.where(m2, sc.flip(1), sc)
        legal = torch.where(m2, legal.flip(1), legal)
        opt = torch.where(m2, opt.flip(1), opt)
    return x, sc, legal, opt, D["val"][idx], D["w"][idx]


def soft_target(sc, legal, temp):
    import torch
    z = (sc / temp).masked_fill(~legal, float("-inf"))
    return torch.softmax(z, 1)


def losses(model, x, sc, legal, val, w, a, fwd=None, teacher=None):
    """(weighted total loss, detached tensor [policy, value, score, rank]) - no GPU sync.
    teacher = (policy probs (B,7), scores in Pons units (B,7)) of the --teacher ensemble:
    adds distill_weight * (cross-entropy to the teacher policy + score regression to the
    teacher's scores) - knowledge distillation of several nets into this one."""
    import torch
    import torch.nn.functional as F
    p, v, s = (fwd or model.forward_all)(x)
    p, v, s = p.float(), v.float(), s.float()
    lp_all = F.log_softmax(p.masked_fill(~legal, -1e9), 1)
    tgt = soft_target(sc, legal, a.policy_temp)
    lp = -(tgt * lp_all.masked_fill(~legal, 0)).sum(1)
    lv = F.cross_entropy(v, val, reduction="none")
    reg = F.smooth_l1_loss(s, sc / SCORE_SCALE, reduction="none", beta=0.1)
    lreg = (reg * legal).sum(1) / legal.sum(1).clamp(min=1)
    ls_all = F.log_softmax((s * SCORE_SCALE / a.rank_temp).masked_fill(~legal, -1e9), 1)
    lrank = -(soft_target(sc, legal, a.rank_temp) * ls_all.masked_fill(~legal, 0)).sum(1)
    per = lp + a.value_weight * lv + a.score_weight * lreg + a.rank_weight * lrank
    if teacher is not None:
        t_pol, t_sc = teacher
        ld_p = -(t_pol * lp_all.masked_fill(~legal, 0)).sum(1)
        ld_s = (F.smooth_l1_loss(s, t_sc / SCORE_SCALE, reduction="none", beta=0.1) * legal).sum(1) \
            / legal.sum(1).clamp(min=1)
        per = per + a.distill_weight * (ld_p + a.score_weight * ld_s)
    wsum = w.sum()
    parts = torch.stack([(t * w).sum() / wsum for t in (lp, lv, lreg, lrank)]).detach()
    return (per * w).sum() / wsum, parts


# ------------------------------------------------------------------ evaluation
def head_logits(model, x, legal, mirror=True):
    """policy log-probs and scores (Pons units), optionally mirror-averaged."""
    import torch
    import torch.nn.functional as F

    def run(xx):
        p, _, s = model.forward_all(xx)
        return p.float(), s.float() * SCORE_SCALE
    p, s = run(x)
    if mirror:
        p2, s2 = run(x.flip(3))
        p, s = (p + p2.flip(1)) / 2, (s + s2.flip(1)) / 2
    lp = F.log_softmax(p.masked_fill(~legal, -1e9), 1)
    return lp, s


def choices(lp, s, legal):
    """dict head-name -> chosen column (B,)"""
    out = {"policy": lp.argmax(1), "score": s.masked_fill(~legal, -1e9).argmax(1)}
    for wgt in COMBO_W[1:]:
        out["combo%g" % wgt] = (s + wgt * lp).masked_fill(~legal, -1e9).argmax(1)
    return out


def eval_positions(model, x, legal, opt, batch=4096, mirror=True):
    """share of positions where each head picks an optimal column."""
    import torch
    acc = {}
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            sl = slice(i, i + batch)
            lp, s = head_logits(model, x[sl], legal[sl], mirror)
            for k, c in choices(lp, s, legal[sl]).items():
                acc[k] = acc.get(k, 0) + opt[sl].gather(1, c[:, None]).sum().item()
    return {k: v / max(len(x), 1) for k, v in acc.items()}


class Bench:
    """The 3,000 held-out positions as GPU tensors, with every column's solver score."""

    def __init__(self, device):
        import torch
        from . import bench
        from .encode import encode, legal_mask
        rows = bench.load_accuracy_bench()
        self.rows = rows
        self.x = torch.from_numpy(encode([r["cur"] for r in rows], [r["mask"] for r in rows])).to(device)
        self.legal = torch.from_numpy(legal_mask(np.array([r["mask"] for r in rows], np.uint64))).to(device)
        sc = np.array([[(-1000 if v is None else v) for v in r["scores"]] for r in rows], np.int64)
        self.sc = torch.from_numpy(sc).to(device)
        best = sc.max(1)
        self.best = torch.from_numpy(best).to(device)
        self.opt = torch.from_numpy(sc == best[:, None]).to(device)
        self.bucket = [r["bucket"] for r in rows]

    def evaluate(self, model, mirror=True):
        import torch
        sg = lambda t: (t > 0).long() - (t < 0).long()  # noqa: E731
        out = {}
        with torch.inference_mode():
            lp, s = head_logits(model, self.x, self.legal, mirror)
            for k, c in choices(lp, s, self.legal).items():
                got = self.sc.gather(1, c[:, None]).squeeze(1)
                nl = self.best >= 0
                out[k] = {"optimal": round(self.opt.gather(1, c[:, None]).float().mean().item(), 4),
                          "keeps_nl": round((sg(got) == sg(self.best))[nl].float().mean().item(), 4)}
        return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(NN_DATA, "solver_3m.npz"), help="comma list of .npz (nn/make_dataset.py format)")
    ap.add_argument("--hard", default="", help="comma list of hard-position .npz (nn/mine_hard.py)")
    ap.add_argument("--hard-weight", type=float, default=2.0)
    ap.add_argument("--run", required=True)
    ap.add_argument("--init", default="", help="start from this checkpoint (.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--policy-temp", type=float, default=1.0, help="policy target softmax(score / T)")
    ap.add_argument("--rank-temp", type=float, default=1.0)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--score-weight", type=float, default=1.0)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay per step (0 = off)")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-max", type=int, default=60000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    ap.add_argument("--speed", choices=SPEEDS, default="safe",
                    help="GPU speed settings (nn/fasttrain.py): off | safe (default) | max = + channels_last + torch.compile")
    ap.add_argument("--teacher", default="",
                    help="distil an ensemble into this net: 'runs/a/best.npz+runs/b/best.npz+runs/c/best.npz'")
    ap.add_argument("--distill-weight", type=float, default=1.0,
                    help="weight of the teacher losses (the Pons-label losses keep weight 1)")
    ap.add_argument("--pca", type=int, default=0,
                    help="PCA input projection with K components fitted on the training positions "
                         "(nn/model.py; research/pca_lda.py runs a sweep). 0 = off")
    a = ap.parse_args()

    import torch
    from .export import export_model
    from .model import build, count_params, load_checkpoint, save_checkpoint

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = pick_device(a.device)
    run = Run(a.run)
    if run.exists():
        raise SystemExit("runs/%s already exists - choose another --run name" % a.run)

    paths = [p for p in a.data.split(",") if p]
    hard = [p for p in a.hard.split(",") if p]
    for p in paths + hard:
        if not os.path.isfile(p):
            raise SystemExit("missing dataset %s" % p)
    D, tr, va, dinfo = load_data(paths, hard, a.hard_weight, a.val_frac, a.val_max, a.seed)

    cfg_model = {"blocks": a.blocks, "channels": a.channels, "score_head": True}
    if a.pca:
        cfg_model["pca_k"] = a.pca
    init_note = None
    if a.init:
        old, _ = load_checkpoint(a.init, "cpu")
        cfg_model.update({k: v for k, v in old.config.items() if k in ("blocks", "channels", "policy_channels",
                                                                    "value_channels", "value_hidden")})
        model = build(cfg_model)
        missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
        init_note = {"from": a.init, "new_layers": sorted({k.split(".")[0] for k in missing})}
        del old
    else:
        model = build(cfg_model)
    pca_info = None
    if a.pca:
        pca_info = fit_input_pca(D["grids"][tr], a.pca, a.seed)
        model.set_input_projection(pca_info.pop("mean"), pca_info.pop("components"))
        log("PCA input projection: %d components keep %.1f%% of the variance of the 84 stone cells"
            % (a.pca, 100 * pca_info["explained_variance"]))
    model = model.to(device)
    from . import fasttrain
    speed_info = fasttrain.configure(a.speed, device)
    model, train_fwd = fasttrain.prepare_model(a.speed, model, device, "forward_all")

    ema = None
    if a.ema:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    cfg = {"phase": "supervised_score", "created": now_iso(), "args": vars(a), "device": str(device),
           "model": model.config, "params": count_params(model), "data": dinfo, "init": init_note,
           "speed": speed_info, "pca": pca_info, "machine": machine_info()}
    run.write_config(cfg)
    log("run %s on %s: %d params; %s" % (a.run, device, cfg["params"], dinfo))

    teacher_info = None
    if a.teacher:
        tpaths = [p for p in a.teacher.split("+") if p]
        for p in tpaths:
            if not os.path.isfile(p):
                raise SystemExit("teacher not found: %s" % p)
        t0t = time.time()
        D["t_pol"], D["t_sc"], n_sc = teacher_targets(tpaths, D, device)
        teacher_info = {"nets": tpaths, "with_score_head": n_sc, "weight": a.distill_weight,
                        "seconds": round(time.time() - t0t, 1)}
        if not n_sc:
            D["t_sc"] = D["scores"].copy()       # no teacher scores: keep the Pons scores as target
        log("teacher targets from %d nets for %d positions in %.0fs" % (len(tpaths), len(D["grids"]), time.time() - t0t))
        cfg["teacher"] = teacher_info
        run.write_config(cfg)
    Dd = to_device(D, device)
    tr_t = torch.from_numpy(tr).to(device)
    va_t = torch.from_numpy(va).to(device)
    # validation tensors restricted to plies 8-30 (the benchmark's range) for choosing the head
    va_mid = va_t[(Dd["ply"][va_t] >= 8) & (Dd["ply"][va_t] <= 30)]
    xv, _, lv_, ov_, _, _ = make_batch(Dd, va_mid, None)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = math.ceil(len(tr) / a.batch)
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    use_amp = device.type == "cuda"
    bench = Bench(device)
    best_metric = -1.0
    step = 0
    t_start = time.time()

    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=device)]
        sums_t, nb = torch.zeros(5, device=device), 0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            mirror = torch.rand(len(idx), device=device) < 0.5
            x, sc, legal, _, val, w = make_batch(Dd, idx, mirror)
            x = fasttrain.to_input(a.speed, x)
            tch = None
            if teacher_info is not None:
                m2 = mirror.view(-1, 1)
                tp, ts = Dd["t_pol"][idx], Dd["t_sc"][idx]
                tch = (torch.where(m2, tp.flip(1), tp), torch.where(m2, ts.flip(1), ts))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, parts = losses(model, x, sc, legal, val, w, a, fwd=train_fwd, teacher=tch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            step += 1
            nb += 1
            sums_t += torch.cat([loss.detach().float().view(1), parts.float()])
            if a.max_steps and step >= a.max_steps:
                break
        sums = sums_t.cpu().numpy().astype(np.float64) / max(nb, 1)     # one GPU sync per epoch

        net = ema.module if ema is not None else model
        net.eval()
        vacc = eval_positions(net, xv, lv_, ov_)
        head = max(vacc, key=lambda k: (vacc[k], k == "policy"))
        bm = bench.evaluate(net, mirror=True)
        bm_nomirror = bench.evaluate(net, mirror=False)
        rec = {"type": "epoch", "phase": "supervised_score", "epoch": epoch, "step": step,
               "lr": sched.get_last_lr()[0], "positions_seen": step * a.batch,
               "train_loss": round(sums[0], 5), "train_policy_loss": round(sums[1], 5),
               "train_value_loss": round(sums[2], 5), "train_score_loss": round(sums[3], 5),
               "train_rank_loss": round(sums[4], 5),
               "val_optimal_by_head": {k: round(v, 4) for k, v in vacc.items()}, "chosen_head": head,
               "bench_optimal": bm[head]["optimal"], "bench_keeps_nl": bm[head]["keeps_nl"],
               "bench_by_head_mirror": bm, "bench_by_head_no_mirror": bm_nomirror,
               "epoch_sec": round(time.time() - t0, 1), "elapsed_sec": round(time.time() - t_start, 1)}
        run.log_metrics(rec)
        if head == "policy":
            play = {"play_head": "policy"}
        elif head == "score":
            play = {"play_head": "score", "play_temp": 0.5}
        else:
            wgt = float(head[5:])
            play = {"play_head": "combo", "play_temp": 0.5, "play_policy_weight": wgt / 0.5}
        meta = dict(play, run=a.run, phase="supervised_score", epoch=epoch, play_mirror=True,
                    val_optimal=round(vacc[head], 4), bench_optimal=bm[head]["optimal"], exported=now_iso())
        save_checkpoint(os.path.join(run.ckpt_dir, "latest.pt"), net,
                        {"epoch": epoch, "step": step})
        export_model(net, run.path("model.npz"), meta)
        if vacc[head] > best_metric:
            best_metric = vacc[head]
            export_model(net, run.path("best.npz"), meta)
            save_checkpoint(os.path.join(run.ckpt_dir, "best.pt"), copy.deepcopy(net).cpu(), {"epoch": epoch})
        log("epoch %d/%d  loss %.4f (p %.3f v %.3f s %.3f r %.3f)  val-opt[%s] %.4f  "
            "BENCH %s %.4f (policy %.4f, score %.4f; no mirror %.4f)  keeps %.4f  %.0fs" % (
                epoch, a.epochs, *sums, head, vacc[head], head, bm[head]["optimal"], bm["policy"]["optimal"],
                bm["score"]["optimal"], bm_nomirror[head]["optimal"], bm[head]["keeps_nl"], rec["epoch_sec"]))
        if a.max_steps and step >= a.max_steps:
            break
    log("done. best val optimal %.4f. Next: python -m nn.eval_score --runs %s" % (best_metric, a.run))


if __name__ == "__main__":
    main()
