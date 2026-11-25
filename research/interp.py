"""Mechanistic interpretability of the C4Net networks ("which layer knows what").

    python -m research.interp                                  # all models, defaults
    python -m research.interp --models sup_6x64 az_6x64 --n 50000 --opp-n 4000
    python -m research.interp --quick                           # small smoke run

Writes experiments/results/research/interp.json (+ interp_channels.json with the heavy
per-channel arrays).  Experiments:
 1. linear concept probes per layer (flat, spatially pooled, shuffled-label control)
 2. channel census (mean maps, row/column selectivity, content, phase, correlations, labels)
 3. causal checks: block knock-out, channel-group ablation vs random-channel null,
    top single channels, head weight attribution, probe-direction patching
 4. cross-model comparison (probe matrices, linear CKA)
Everything is computed from a fixed-seed subset of nn_data/solver_300k.npz.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from nn.encode import encode  # noqa: E402

from .interp_core import (TorchNet, chance_score, compute_concepts, flip_planes,  # noqa: E402
                          layer_acts, linear_cka, run_batched, train_probes)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "experiments", "results", "research")
DATA = os.path.join(ROOT, "nn_data", "solver_300k.npz")
BENCH = os.path.join(ROOT, "data", "accuracy_positions.csv")
MODELS = {
    "sup_6x64": "runs/sup_6x64/best.npz",
    "az_6x64": "runs/az_6x64/model.npz",
    "distill_az_2x32": "runs/distill_az_2x32/best.npz",
    "ppo_6x64": "runs/ppo_6x64/model.npz",
    # added 2026-11-24: the rest of the nets in the top8_o100_mcts_fixed arena
    "ppo_from_sup": "runs/ppo_from_sup/model.npz",
    "score_8x128_s0": "runs/score_8x128_s0_r1/best.npz",
    "score_8x128_s1": "runs/score_8x128_s1_r1/best.npz",
    "score_8x128_s2": "runs/score_8x128_s2_r1/best.npz",
}
ARENA_MODELS = ["sup_6x64", "az_6x64", "ppo_from_sup", "ppo_6x64", "distill_az_2x32",
                "score_8x128_s0", "score_8x128_s1", "score_8x128_s2"]
PRIMARY = "sup_6x64"

# thresholds (stated in notes)
FIRST_FRAC = 0.9        # first layer reaching 90% of the best normalised skill
POOL_FRAC = 0.8         # pooled probe keeps >= 80% of the best flat normalised skill
DEAD_ACTIVE = 0.01      # channel > 0 on < 1% of (position, cell) samples
THR_R = 0.3             # |r| with threat cells (empty cells only)
OCC_R = 0.5             # |r| with mine / theirs / empty cells
POS_FRAC = 0.25         # variance share of the mean map in total activation variance
SEL = 0.6               # row or column share of the mean map's variance
PHASE_R = 0.6           # |r| of pooled activation with ply
NULL_DRAWS = 30
SMALL_D = 400           # probes on <= this many features run on the CPU

# id, label, group, kind, K
CONCEPTS = [
    ("heights", "Column heights (stones per column, 7 outputs)", "geometry", "reg", None),
    ("landing_row", "Landing row per column (7-way per column, incl. full)", "geometry", "multicat", 7),
    ("ply", "Ply = number of stones (regression)", "geometry", "reg", None),
    ("first_to_move", "Side to move is the first player (parity)", "geometry", "multibin", None),
    ("win_now", "Own immediate win, per column", "tactics", "multibin", None),
    ("opp_win_now", "Opponent's immediate win per column (= must-block columns; also opp-move (c))", "tactics", "multibin", None),
    ("own_threats", "Count of own threat squares (empty cells completing four)", "tactics", "reg", None),
    ("opp_threats", "Count of opponent threat squares", "tactics", "reg", None),
    ("own_double", "Own double threat (2 playable wins or stacked threats)", "tactics", "multibin", None),
    ("opp_double", "Opponent double threat", "tactics", "multibin", None),
    ("own_odd_thr", "Own threat squares on odd rows (1,3,5 from bottom)", "tactics", "reg", None),
    ("own_even_thr", "Own threat squares on even rows (2,4,6 from bottom)", "tactics", "reg", None),
    ("opp_odd_thr", "Opponent threat squares on odd rows", "tactics", "reg", None),
    ("opp_even_thr", "Opponent threat squares on even rows", "tactics", "reg", None),
    ("opt_set", "Solver-optimal column set (multi-label)", "own_move", "multibin", None),
    ("net_move", "The net's own argmax move", "own_move", "cat", 7),
    ("outcome", "Solver outcome W/D/L", "own_move", "cat", 3),
    ("score", "Solver score of the position", "own_move", "reg", None),
    ("opp_reply", "Opponent's solver-optimal reply after the net's move (subset)", "opp_move", "multibin", None),
    ("opp_net_move", "Net's argmax if the opponent were to move now (colour-flipped input)", "opp_move", "cat", 7),
    ("phase", "Phase: early (ply<12) / mid / late (ply>=24)", "phase", "cat", 3),
]
METRIC = {"multibin": "auc", "cat": "acc", "multicat": "acc", "reg": "r2"}
SUBSET_CONCEPTS = {"opp_reply"}


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def rnd(x, d=4):
    if isinstance(x, dict):
        return {k: rnd(v, d) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, d) for v in x]
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), d)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if x != x else round(x, d)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def norm_skill(s, chance):
    if s is None or chance is None or s != s:
        return float("nan")
    return (s - chance) / max(1e-9, 1 - chance)


# ============================================================================ data
def load_data(n, seed):
    z = np.load(DATA)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(z["ply"]), n, replace=False))
    idx = idx[rng.permutation(n)]
    return z["cur"][idx], z["mask"][idx], z["scores"][idx].astype(np.int64), z["ply"][idx].astype(np.int64)


def splits(n):
    ntr, nva = int(0.7 * n), int(0.1 * n)
    return np.arange(ntr), np.arange(ntr, ntr + nva), np.arange(ntr + nva, n)


def concept_labels(c):
    """Shape every label for the probe trainer (binary -> (N,1))."""
    out = {}
    for cid, _, _, kind, _ in CONCEPTS:
        if cid not in c:
            continue
        y = c[cid]
        if kind == "multibin" and y.ndim == 1:
            y = y[:, None]
        out[cid] = y
    return out


# ============================================================================ forward checks
def verify_forward(net, X):
    lt, vt = run_batched(net, X[:512])
    lr, vr = net.ref.forward(X[:512])
    return float(max(np.abs(lt.cpu().numpy() - lr).max(), np.abs(vt.cpu().numpy() - vr).max()))


def bench_agreement(net):
    from engine.board import Board
    curs, masks, opts = [], [], []
    with open(BENCH, newline="") as f:
        for r in csv.DictReader(f):
            b = Board.from_moves(r["moves"])
            curs.append(b.current_bits())
            masks.append(b.mask)
            opts.append(set(json.loads(r["optimal_cols"])))
    x = encode(curs, masks)
    from nn.encode import legal_mask
    lg, _ = run_batched(net, x)
    lg = lg.cpu().numpy()
    lg[~legal_mask(np.array(masks, dtype=np.uint64))] = -np.inf
    a = lg.argmax(1)
    return float(np.mean([a[i] in opts[i] for i in range(len(a))])), len(a)


# ============================================================================ eval context
class Ctx:
    """Labels of the evaluation positions as torch tensors, for ablation metrics."""

    def __init__(self, c, extra, sel, dev):
        f = extra["features"]
        T = lambda a: torch.as_tensor(np.asarray(a)[sel], device=dev)  # noqa: E731
        self.legal = T(extra["legal"])
        self.opt = T(extra["opt"])
        self.win = T(c["win_now"])
        self.oppwin = T(c["opp_win_now"])
        self.outcome = T(c["outcome"])
        self.phase = T(c["phase"])
        n_ow = self.oppwin.sum(1)
        has_win = self.win.any(1)
        self.blockpos = (~has_win) & (n_ow == 1)
        self.blockcol = self.oppwin.float().argmax(1)
        self.groups = {
            "all": torch.ones_like(has_win),
            "early": self.phase == 0, "mid": self.phase == 1, "late": self.phase == 2,
            "own_win": has_win, "must_block": (~has_win) & (n_ow >= 1), "quiet": (~has_win) & (n_ow == 0),
        }
        del f


@torch.no_grad()
def metrics(ctx, logits, wdl, ref_logits, ref_wdl):
    lg = logits.masked_fill(~ctx.legal, float("-inf"))
    rl = ref_logits.masked_fill(~ctx.legal, float("-inf"))
    a = lg.argmax(1)
    ar = torch.arange(len(a), device=a.device)
    agree = ctx.opt[ar, a].float()
    vacc = (wdl.argmax(1) == ctx.outcome).float()
    lp, lq = F.log_softmax(lg, 1), F.log_softmax(rl, 1)
    kl = torch.where(ctx.legal, lq.exp() * (lq - lp), torch.zeros_like(lp)).sum(1)
    pw, rw = F.softmax(wdl, 1), F.softmax(ref_wdl, 1)
    dv = ((pw[:, 0] - pw[:, 2]) - (rw[:, 0] - rw[:, 2])).abs()
    has_win = ctx.groups["own_win"].float()
    blk = ctx.blockpos.float()
    missw = (~ctx.win[ar, a]).float() * has_win
    missb = (a != ctx.blockcol).float() * blk
    if not hasattr(ctx, "_G"):
        ctx._names = list(ctx.groups)
        ctx._G = torch.stack([ctx.groups[g].float() for g in ctx._names])          # (G,N)
    V = torch.stack([torch.ones_like(agree), agree, vacc, kl, dv, has_win, missw, blk, missb], 1)
    S = (ctx._G.double() @ V.double()).cpu().numpy()                               # one transfer
    out = {}
    for gi, g in enumerate(ctx._names):
        n, sa, sv, sk, sd, nw, smw, nb, smb = S[gi]
        if n == 0:
            continue
        d = {"n": int(n), "agree": sa / n, "value_acc": sv / n, "kl": sk / n, "dv": sd / n}
        if nw > 0:
            d["missed_win"] = smw / nw
        if nb > 0:
            d["missed_block"] = smb / nb
        out[g] = d
    return out


def delta(m, base):
    """Ablated minus intact for every metric present in both."""
    return {g: {k: (v - base[g][k]) if k not in ("n",) and k in base[g] else v for k, v in d.items()}
            for g, d in m.items() if g in base}


# ============================================================================ 1. probes
def probe_model(net, X, labels, kinds, tr, va, te, sub, args):
    """labels: {cid: y}; sub: (tr_s, va_s, te_s) indices for SUBSET_CONCEPTS (or None)."""
    res = {cid: {} for cid in labels}
    full_heads = [{"id": c, "kind": kinds[c][0], "K": kinds[c][1], "y": labels[c]}
                  for c in labels if c not in SUBSET_CONCEPTS]
    sub_heads = [{"id": c, "kind": kinds[c][0], "K": kinds[c][1], "y": labels[c]}
                 for c in labels if c in SUBSET_CONCEPTS]
    for layer in ["input"] + net.layers:
        t0 = time.time()
        A = layer_acts(net, X, layer)
        if A.shape[1] <= SMALL_D:          # small problems are latency-bound on a shared GPU: use the CPU
            A = A.float().cpu()
        P = layer_acts(net, X, layer, pooled=True).float().cpu() if layer not in ("val_hidden",) else None
        flat, _ = train_probes(A, full_heads, tr, va, te, rounds=args.rounds, seed=1)
        shuf, _ = train_probes(A, full_heads, tr, va, te, rounds=args.rounds, seed=2, shuffle=True)
        pool = train_probes(P, full_heads, tr, va, te, rounds=args.rounds, seed=3)[0] if P is not None else {}
        if sub_heads and sub is not None:
            ts, vs, es = sub
            A = A.float().cpu()
            f2, _ = train_probes(A, sub_heads, ts, vs, es, rounds=args.rounds, seed=1, l2=1e-3)
            s2, _ = train_probes(A, sub_heads, ts, vs, es, rounds=args.rounds, seed=2, l2=1e-3, shuffle=True)
            p2 = train_probes(P, sub_heads, ts, vs, es, rounds=args.rounds, seed=3, l2=1e-3)[0] if P is not None else {}
            flat.update(f2)
            shuf.update(s2)
            pool.update(p2)
        for cid in labels:
            if cid in flat:
                res[cid][layer] = {"flat": flat[cid], "pooled": pool.get(cid), "shuffled": shuf[cid]}
        del A, P
        log("  probes %s: %.1fs" % (layer, time.time() - t0))
    return res


def summarise_probes(probes, layer_order, chance):
    out = {}
    for cid, per in probes.items():
        ch = chance[cid]
        layers = [l for l in layer_order if l in per]
        if not layers:
            continue
        flat = {l: per[l]["flat"] for l in layers}
        best_layer = max(layers, key=lambda l: -1e9 if flat[l] != flat[l] else flat[l])
        best = flat[best_layer]
        nb = norm_skill(best, ch)
        first = next((l for l in layers if norm_skill(flat[l], ch) >= FIRST_FRAC * nb), best_layer)
        pooled = [per[l]["pooled"] for l in layers if per[l]["pooled"] is not None]
        pbest = max(pooled) if pooled else float("nan")
        out[cid] = {"first_layer": first, "best_layer": best_layer, "best": best,
                    "input_baseline": flat.get("input"), "majority": ch,
                    "pooled_best": pbest,
                    "survives_pooling": bool(norm_skill(pbest, ch) >= POOL_FRAC * nb) if nb > 0 else False,
                    "shuffled_max": max(per[l]["shuffled"] for l in layers),
                    "norm_best": nb}
    return out


# ============================================================================ 2. census
@torch.no_grad()
def census(net, X, c, extra, sel):
    dev = net.device
    f = extra["features"]
    G = lambda k: torch.as_tensor(f[k][sel].reshape(len(sel), -1), device=dev).float()  # noqa: E731
    cellf = {k: G(k) for k in ("me", "them", "empty", "playable", "own_thr", "opp_thr", "own_imm", "opp_imm")}
    empty = cellf["empty"]
    scal = {
        "ply": c["ply"][sel], "value": 1.0 - c["outcome"][sel].astype(np.float32),     # +1 win, 0 draw, -1 loss
        "win_now_any": c["win_now"][sel].any(1), "must_block_any": c["opp_win_now"][sel].any(1),
        "own_threats": c["own_threats"][sel], "opp_threats": c["opp_threats"][sel],
    }
    scal = {k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=dev) for k, v in scal.items()}
    phase = torch.as_tensor(c["phase"][sel], device=dev)
    content = torch.full(empty.shape, 1, device=dev, dtype=torch.long)     # 1 = empty, not playable
    content[cellf["playable"].bool()] = 0
    content[cellf["me"].bool()] = 2
    content[cellf["them"].bool()] = 3
    Xs = X[sel]
    out = {}
    for layer in net.stream_layers:
        A = layer_acts(net, Xs, layer, dtype=torch.float32).view(len(sel), net.channels, -1)  # (N,C,42)
        N, C, S = A.shape
        M = A.mean(0)                                                    # (C,42)
        Mg = M.view(C, 6, 7)
        g = Mg.mean((1, 2), keepdim=True)
        re = Mg.mean(2, keepdim=True) - g
        ce = Mg.mean(1, keepdim=True) - g
        ss_tot = ((Mg - g) ** 2).sum((1, 2))
        ss_row = (re ** 2).sum((1, 2)) * 7
        ss_col = (ce ** 2).sum((1, 2)) * 6
        var_tot = A.permute(1, 0, 2).reshape(C, -1).var(1, unbiased=False)
        pos_frac = (ss_tot / 42) / (var_tot + 1e-12)
        active = (A > 0).float().mean((0, 2))

        def corr(Fm, mask=None):
            Fm = Fm[:, None, :]
            Mk = torch.ones_like(Fm) if mask is None else mask[:, None, :]
            n = Mk.sum()
            sa = (A * Mk).sum((0, 2))
            saa = (A * A * Mk).sum((0, 2))
            sf = (Fm * Mk).sum()
            sff = (Fm * Fm * Mk).sum()
            saf = (A * Fm * Mk).sum((0, 2))
            cov = saf / n - sa / n * sf / n
            va = saa / n - (sa / n) ** 2
            vf = sff / n - (sf / n) ** 2
            return cov / (va.clamp_min(1e-12) * vf.clamp_min(1e-12)).sqrt()

        cc = {"cell_mine": corr(cellf["me"]), "cell_theirs": corr(cellf["them"]),
              "cell_empty": corr(cellf["empty"]), "cell_playable": corr(cellf["playable"]),
              "own_thr_cell": corr(cellf["own_thr"], empty), "opp_thr_cell": corr(cellf["opp_thr"], empty),
              "own_win_cell": corr(cellf["own_imm"], empty), "opp_win_cell": corr(cellf["opp_imm"], empty)}
        pooled = A.mean(2)                                                # (N,C)
        pc = pooled - pooled.mean(0)
        for k, v in scal.items():
            vc = v - v.mean()
            cc[k] = (pc * vc[:, None]).mean(0) / (pc.std(0, unbiased=False).clamp_min(1e-9) * vc.std(unbiased=False).clamp_min(1e-9))
        ph = torch.stack([pooled[phase == p].mean(0) for p in range(3)], 1)   # (C,3)
        cont = torch.stack([(A * (content == k)[:, None, :]).sum((0, 2)) / (content == k).sum().clamp_min(1)
                            for k in range(4)], 1)                        # (C,4)
        # content eta^2 = between-category variance / total variance over (n, cell)
        fr = torch.stack([(content == k).float().mean() for k in range(4)])
        grand = A.mean((0, 2))
        eta = ((cont - grand[:, None]) ** 2 * fr[None, :]).sum(1) / var_tot.clamp_min(1e-12)
        rows = []
        cm = {k: v.cpu().numpy() for k, v in cc.items()}
        for ch in range(C):
            dead = bool(active[ch] < DEAD_ACTIVE)
            rs = float(ss_row[ch] / ss_tot[ch]) if ss_tot[ch] > 1e-12 else 0.0
            cs = float(ss_col[ch] / ss_tot[ch]) if ss_tot[ch] > 1e-12 else 0.0
            pf = float(pos_frac[ch])
            r = {k: float(np.nan_to_num(v[ch])) for k, v in cm.items()}
            top = max(r, key=lambda k: abs(r[k]))
            # label: dead first; else the first candidate in a fixed priority order
            cand = {}
            own_r = max(r["own_thr_cell"], r["own_win_cell"])
            opp_r = max(r["opp_thr_cell"], r["opp_win_cell"])
            if own_r >= THR_R and own_r > opp_r:
                cand["own-threat"] = own_r / THR_R
            if opp_r >= THR_R and opp_r > own_r:
                cand["opponent-threat"] = opp_r / THR_R
            occ = max(abs(r["cell_mine"]), abs(r["cell_theirs"]), abs(r["cell_empty"]))
            if occ >= OCC_R:
                cand["occupancy copy"] = occ / OCC_R
            if pf >= POS_FRAC and cs >= SEL:
                cand["column detector"] = min(pf / POS_FRAC, cs / SEL)
            if pf >= POS_FRAC and rs >= SEL:
                cand["row/height detector"] = min(pf / POS_FRAC, rs / SEL)
            if abs(r["ply"]) >= PHASE_R:
                cand["phase: late" if r["ply"] > 0 else "phase: early"] = abs(r["ply"]) / PHASE_R
            order = ["own-threat", "opponent-threat", "occupancy copy", "column detector",
                     "row/height detector", "phase: early", "phase: late"]
            label = "dead" if dead else next((k for k in order if k in cand), "mixed")
            if label in ("own-threat", "opponent-threat") and "own-threat" in cand and "opponent-threat" in cand:
                label = max(("own-threat", "opponent-threat"), key=cand.get)
            e, m_, l_ = [float(v) for v in ph[ch].cpu()]
            rows.append({
                "ch": ch, "label": label, "row_sel": rs, "col_sel": cs, "pos_frac": pf,
                "active": float(active[ch]), "phase": [e, m_, l_],
                "phase_idx": (l_ - e) / (abs(l_) + abs(e) + 1e-6),
                "content": [float(v) for v in cont[ch].cpu()], "content_eta2": float(eta[ch]),
                "top_concept": top, "top_corr": r[top], "corr": r, "dead": dead,
                "_map": Mg[ch].cpu().numpy(),
            })
        out[layer] = rows
        del A
    return out


# ============================================================================ 3. ablations
@torch.no_grad()
def ablations(net, X, ctx, cen, sel, rng):
    dev = net.device
    Xs = torch.as_tensor(X[sel], device=dev)
    ref_l, ref_v = run_batched(net, Xs)
    base = metrics(ctx, ref_l, ref_v, ref_l, ref_v)
    # means of the stream and of the residual branches on the eval set (batched)
    means = {}
    for s in range(0, len(Xs), 2048):
        _, _, acts = net.run(Xs[s:s + 2048], capture=tuple(net.stream_layers), capture_branch=True)
        for k, v in acts.items():
            means[k] = means.get(k, 0) + v.sum(0)
        del acts
    means = {k: v / len(Xs) for k, v in means.items()}
    blocks = []
    for k in range(1, net.blocks + 1):
        for mode in ("identity", "zero_branch", "mean_branch"):
            bm = {k: ("mean_branch", means["block%d_branch" % k]) if mode == "mean_branch" else mode}
            l, v = run_batched(net, Xs, block_mode=bm)
            d = delta(metrics(ctx, l, v, ref_l, ref_v), base)
            blocks.append({"layer": "block%d" % k, "mode": mode, "overall": d["all"],
                           "by_phase": {p: d[p] for p in ("early", "mid", "late") if p in d},
                           "by_type": {t: d[t] for t in ("own_win", "must_block", "quiet") if t in d}})

    def ablate(layer, chans, mode):
        idx = torch.as_tensor(chans, device=dev, dtype=torch.long)
        mm = means[layer]

        def fn(h):
            h = h.clone()
            h[:, idx] = 0.0 if mode == "zero" else mm[idx].unsqueeze(0)
            return h
        l, v = run_batched(net, Xs, edit={layer: fn})
        return delta(metrics(ctx, l, v, ref_l, ref_v), base)

    def pick(d):
        a = d["all"]
        return {"agree": a["agree"], "value_acc": a["value_acc"], "kl": a["kl"],
                "missed_win": a.get("missed_win"), "missed_block": a.get("missed_block")}

    groups = []
    for layer in net.stream_layers:
        rows = cen[layer]
        alive = [r["ch"] for r in rows if not r["dead"]]
        labels = sorted({r["label"] for r in rows if r["label"] not in ("dead",)})
        for lab in labels:
            chans = [r["ch"] for r in rows if r["label"] == lab]
            if not chans:
                continue
            pool = [c for c in alive if c not in chans]
            if len(pool) < len(chans):
                pool = alive
            for mode in ("zero", "mean"):
                eff = pick(ablate(layer, chans, mode))
                null = [pick(ablate(layer, list(rng.choice(pool, len(chans), replace=False)), mode))
                        for _ in range(NULL_DRAWS)]
                ns = {}
                for key in ("agree", "value_acc", "kl", "missed_win", "missed_block"):
                    vals = np.array([n[key] for n in null if n[key] is not None], float)
                    if eff[key] is None or len(vals) == 0:
                        continue
                    ns[key] = {"mean": vals.mean(), "std": vals.std(),
                               "pct": float((vals < eff[key]).mean()),
                               "z": (eff[key] - vals.mean()) / (vals.std() + 1e-9)}
                spec = None
                if eff["missed_block"] is not None and eff["missed_win"] is not None:
                    diffs = np.array([n["missed_block"] - n["missed_win"] for n in null])
                    e = eff["missed_block"] - eff["missed_win"]
                    spec = {"block_minus_win": e, "null_mean": diffs.mean(), "null_std": diffs.std(),
                            "pct": float((diffs < e).mean())}
                groups.append({"layer": layer, "group": lab, "n_channels": len(chans), "mode": mode,
                               "effect": eff, "null": ns, "specificity": spec})
    # single channels (mean ablation)
    single = {}
    for layer in net.stream_layers:
        per = []
        for ch in range(net.channels):
            d = ablate(layer, [ch], "mean")
            per.append({"kl": d["all"]["kl"], "dv": d["all"]["dv"], "agree": d["all"]["agree"],
                        "value_acc": d["all"]["value_acc"],
                        "agree_phase": [d[p]["agree"] for p in ("early", "mid", "late")],
                        "value_acc_phase": [d[p]["value_acc"] for p in ("early", "mid", "late")],
                        "missed_win": d["all"].get("missed_win"), "missed_block": d["all"].get("missed_block")})
        single[layer] = per
    return base, blocks, groups, single


def top_channels(single, cen, k=8):
    flat = [(layer, ch, s) for layer, per in single.items() for ch, s in enumerate(per)]

    def row(layer, ch, s):
        return {"layer": layer, "ch": ch, "label": cen[layer][ch]["label"], "kl": s["kl"], "dv": s["dv"],
                "d_agree": s["agree"], "d_value_acc": s["value_acc"]}
    out = {"policy": [row(*t) for t in sorted(flat, key=lambda t: -t[2]["kl"])[:k]],
           "value": [row(*t) for t in sorted(flat, key=lambda t: -t[2]["dv"])[:k]], "by_phase": {}}
    for i, p in enumerate(("early", "mid", "late")):
        out["by_phase"][p] = {
            "policy": [dict(row(*t), d_agree_phase=t[2]["agree_phase"][i])
                       for t in sorted(flat, key=lambda t: t[2]["agree_phase"][i])[:k]],
            "value": [dict(row(*t), d_value_acc_phase=t[2]["value_acc_phase"][i])
                      for t in sorted(flat, key=lambda t: t[2]["value_acc_phase"][i])[:k]]}
    return out


def head_attribution(net):
    w = net.ref.w
    pc = w["pol.w"].shape[0]
    pol = w["pol_fc.w"].reshape(7, pc, 6, 7)
    colmass = np.abs(pol).sum((1, 2))                                # (7 outputs, 7 board columns)
    locality = [float(colmass[j, j] / colmass[j].mean()) for j in range(7)]
    own_frac = [float(colmass[j, j] / colmass[j].sum()) for j in range(7)]
    vc = w["val.w"].shape[0]
    eff = ((w["val_fc2.w"][0] - w["val_fc2.w"][2]) @ w["val_fc1.w"]).reshape(vc, 6, 7)
    return {"policy_fc": pol, "policy_col_mass": colmass, "policy_locality": locality,
            "policy_own_col_frac": own_frac,
            "value_fc": eff, "value_row_mass": np.abs(eff).sum((0, 2)), "value_col_mass": np.abs(eff).sum((0, 1)),
            "value_note": "value_fc = (W2[win]-W2[loss]) @ W1 reshaped (value_conv_channels,6,7): linear "
                          "win-minus-loss read-out ignoring the hidden ReLU"}


# ============================================================================ patching
@torch.no_grad()
def patching(net, X, labels, tr, va, te, layer, ctx, rng, n_rand=20):
    dev = net.device
    with torch.enable_grad():
        A = layer_acts(net, X, layer)
        _, wts = train_probes(A, [{"id": "opp_win_now", "kind": "multibin", "K": None, "y": labels["opp_win_now"]}],
                              tr, va, te, rounds=10, seed=4, return_weights=("opp_win_now",))
    W = wts["opp_win_now"][0]                                          # (D,7)
    Ate = A[torch.as_tensor(te, device=dev)].float()
    del A
    Xs = torch.as_tensor(X[te], device=dev)
    ref_l, _ = run_batched(net, Xs)
    ow = labels["opp_win_now"][te]
    shape = (net.channels, 6, 7)
    res = []
    for j in range(7):
        d = W[:, j] / W[:, j].norm()
        proj = Ate @ d
        pos = torch.as_tensor(ow[:, j], device=dev)
        legal_j = ctx.legal[:, j]
        neg = (~pos) & legal_j
        if int(pos.sum()) < 20:
            continue
        alpha = float(proj[pos].mean() - proj[neg].mean())
        quiet = ctx.groups["quiet"] & legal_j
        blk = ctx.blockpos & (ctx.blockcol == j)

        def effect(direction, sign, msk):
            v = (sign * alpha * direction).view(shape)
            l, _, _ = net.run(Xs[msk], edit={layer: lambda h: h + v})
            lr = ref_l[msk]
            dl = l - lr
            others = torch.cat([dl[:, :j], dl[:, j + 1:]], 1).mean(1)
            lg = l.masked_fill(~ctx.legal[msk], float("-inf"))
            rl = lr.masked_fill(~ctx.legal[msk], float("-inf"))
            dp = F.softmax(lg, 1)[:, j] - F.softmax(rl, 1)[:, j]
            return {"d_logit_rel": float((dl[:, j] - others).mean()), "d_prob": float(dp.mean()),
                    "to_j": float((lg.argmax(1) == j).float().mean() - (rl.argmax(1) == j).float().mean())}

        row = {"col": j, "alpha": alpha, "n_add": int(quiet.sum()), "n_sub": int(blk.sum())}
        for name, sign, msk in (("add", 1.0, quiet), ("subtract", -1.0, blk)):
            if int(msk.sum()) < 10:
                continue
            e = effect(d, sign, msk)
            rnd_e = []
            for _ in range(n_rand):
                r = torch.as_tensor(rng.standard_normal(d.shape[0]), device=dev, dtype=torch.float32)
                rnd_e.append(effect(r / r.norm(), sign, msk))
            rv = np.array([x["d_logit_rel"] for x in rnd_e])
            rp = np.array([x["d_prob"] for x in rnd_e])
            row[name] = dict(e, rand_logit_mean=rv.mean(), rand_logit_std=rv.std(),
                             z=(e["d_logit_rel"] - rv.mean()) / (rv.std() + 1e-9),
                             rand_prob_mean=rp.mean(), rand_prob_std=rp.std())
        res.append(row)
    return {"layer": layer, "concept": "opp_win_now", "random_draws": n_rand, "per_column": res}


# ============================================================================ 4. cross-model
@torch.no_grad()
def cross_model(nets, X, sel, summaries, kinds):
    out = {"cka": {}, "concept_diffs": {}}
    if PRIMARY not in nets:
        return out
    a = nets[PRIMARY]
    Xs = X[sel]
    acts_a = {l: layer_acts(a, Xs, l) for l in ["input"] + a.layers}
    for mid, b in nets.items():
        if mid == PRIMARY:
            continue
        lb = ["input"] + b.layers
        mat = []
        for la in acts_a:
            row = []
            for l2 in lb:
                B = layer_acts(b, Xs, l2)
                row.append(linear_cka(acts_a[la].float(), B.float()))
                del B
            mat.append(row)
        out["cka"][mid] = {"rows": list(acts_a), "cols": lb, "matrix": mat,
                           "diag": {l: mat[i][lb.index(l)] for i, l in enumerate(acts_a) if l in lb}}
        diffs = []
        for cid, s in summaries[PRIMARY].items():
            if cid not in summaries[mid]:
                continue
            na, nb = s["norm_best"], summaries[mid][cid]["norm_best"]
            diffs.append({"concept": cid, "primary_best": s["best"], "other_best": summaries[mid][cid]["best"],
                          "primary_norm": na, "other_norm": nb, "diff_norm": na - nb})
        out["concept_diffs"][mid] = sorted(diffs, key=lambda d: -abs(d["diff_norm"]))
    return out


# ============================================================================ findings
def make_findings(R):
    F_ = []
    ps = R["probe_summary"]
    pr = R["probes"]
    lab = {c["id"]: c["label"] for c in R["concepts"]}
    met = {c["id"]: c["metric"] for c in R["concepts"]}

    def desc(m, cid):
        s = ps[m][cid]
        return "%s %s=%.3f at %s (first >=90%% of best skill: %s; raw-input probe %.3f; chance %.3f; shuffled-label max %.3f; pooled best %.3f)" % (
            cid, met[cid], s["best"], s["best_layer"], s["first_layer"], s["input_baseline"], s["majority"],
            s["shuffled_max"], s["pooled_best"])

    m = PRIMARY if PRIMARY in ps else next(iter(ps))
    S = ps[m]
    groups = [("geometry", ["heights", "landing_row", "ply", "first_to_move"], "Board geometry (row/column)"),
              ("tactics", ["win_now", "opp_win_now", "own_threats", "opp_threats", "own_double", "opp_double"],
               "Own vs opponent threats"),
              ("zugzwang", ["own_odd_thr", "own_even_thr", "opp_odd_thr", "opp_even_thr"], "Odd/even threat squares"),
              ("own_move", ["opt_set", "net_move", "outcome", "score"], "Own move and evaluation"),
              ("opp_move", ["opp_reply", "opp_net_move", "opp_win_now"], "Opponent's move"),
              ("phase", ["phase", "ply"], "Game phase")]
    for gid, cids, title in groups:
        cids = [c for c in cids if c in S]
        if not cids:
            continue
        lines = [desc(m, c) for c in cids]
        gain = {c: S[c]["best"] - (S[c]["input_baseline"] or 0) for c in cids}
        top = max(cids, key=lambda c: gain[c])
        claim = "%s: %s; largest gain over a probe on the raw input is %s (+%.3f, %s %.3f vs %.3f)." % (
            m, "; ".join("%s best at %s (%s %.3f)" % (c, S[c]["best_layer"], met[c], S[c]["best"]) for c in cids),
            top, gain[top], met[top], S[top]["best"], S[top]["input_baseline"])
        pooled_ok = [c for c in cids if S[c]["survives_pooling"]]
        F_.append({"id": "interp-probe-" + gid, "title": title, "claim": claim,
                   "evidence": " | ".join(lines),
                   "caveat": "Linear probes show decodability, not use. Survives spatial averaging (>=%d%% of best skill): %s." % (
                       int(POOL_FRAC * 100), ", ".join(pooled_ok) or "none")})
    # blocks
    for mid, rows in R["ablation"]["blocks"].items():
        rows = [r for r in rows if r["mode"] == "mean_branch"]
        if not rows:
            continue
        worst = min(rows, key=lambda r: r["overall"]["agree"])
        ph = {p: min(rows, key=lambda r: r["by_phase"].get(p, {"agree": 0})["agree"]) for p in ("early", "late")}
        ident = [r for r in R["ablation"]["blocks"][mid] if r["mode"] == "identity"]
        F_.append({"id": "interp-blocks-" + mid, "title": "Which blocks matter (%s)" % mid,
                   "claim": "Mean-ablating the residual branch of %s costs the most solver agreement overall (%+.3f); "
                            "early-game (ply<12) most affected by %s (%+.3f), late-game (ply>=24) by %s (%+.3f)." % (
                                worst["layer"], worst["overall"]["agree"], ph["early"]["layer"],
                                ph["early"]["by_phase"]["early"]["agree"], ph["late"]["layer"],
                                ph["late"]["by_phase"]["late"]["agree"]),
                   "evidence": "identity knock-out agreement deltas: " + ", ".join(
                       "%s %+.3f" % (r["layer"], r["overall"]["agree"]) for r in ident),
                   "caveat": "zero_branch equals identity exactly (block inputs are post-ReLU, so ReLU(h+0)=h); "
                             "deltas are on %d held-out positions." % rows[0]["overall"]["n"]})
    # group specificity
    for mid, rows in R["ablation"]["groups"].items():
        sel = [r for r in rows if r["mode"] == "mean" and r["group"] in ("opponent-threat", "own-threat")
               and r["specificity"] is not None]
        if not sel:
            F_.append({"id": "interp-groups-" + mid, "title": "Threat-channel ablation (%s)" % mid,
                       "claim": "No channel met the threat-channel rule (|r|>=%.1f with threat cells), so no threat-group ablation was possible." % THR_R,
                       "evidence": "", "caveat": ""})
            continue
        best = sorted(sel, key=lambda r: -abs(r["specificity"]["block_minus_win"] - r["specificity"]["null_mean"]))[:4]
        ev = "; ".join("%s %s (%d ch, mean-abl): d_missed_block %+.3f, d_missed_win %+.3f, block-minus-win %+.3f vs null %.3f+-%.3f (pct %.2f)" % (
            r["layer"], r["group"], r["n_channels"], r["effect"]["missed_block"], r["effect"]["missed_win"],
            r["specificity"]["block_minus_win"], r["specificity"]["null_mean"], r["specificity"]["null_std"],
            r["specificity"]["pct"]) for r in best)
        opp = [r for r in sel if r["group"] == "opponent-threat"]
        own = [r for r in sel if r["group"] == "own-threat"]
        opp_sig = sum(r["specificity"]["pct"] >= 0.95 for r in opp)
        own_sig = sum(r["specificity"]["pct"] <= 0.05 for r in own)
        F_.append({"id": "interp-groups-" + mid, "title": "Threat-channel ablation specificity (%s)" % mid,
                   "claim": "Opponent-threat groups whose ablation raises missed blocks more than missed wins beyond 95%% of %d random-channel draws: %d of %d layers; own-threat groups with the reverse (<=5th pct): %d of %d." % (
                       NULL_DRAWS, opp_sig, len(opp), own_sig, len(own)),
                   "evidence": ev,
                   "caveat": "Groups come from correlational labels; percentiles are against %d random equal-size sets of live channels in the same layer." % NULL_DRAWS})
    # census
    for mid, per in R["channel_counts"].items():
        tot = {}
        for layer, cnt in per.items():
            for k, v in cnt.items():
                tot[k] = tot.get(k, 0) + v
        F_.append({"id": "interp-census-" + mid, "title": "Channel census (%s)" % mid,
                   "claim": "Across stem+blocks: " + ", ".join("%s %d" % (k, v) for k, v in sorted(tot.items(), key=lambda t: -t[1])) + ".",
                   "evidence": "; ".join("%s: %s" % (l, ", ".join("%s %d" % kv for kv in sorted(c.items(), key=lambda t: -t[1]))) for l, c in per.items()),
                   "caveat": "Labels are rule-based summaries of correlations/mean maps (thresholds in notes), not proofs of function."})
    # heads
    for mid, h in R["heads"].items():
        F_.append({"id": "interp-head-" + mid, "title": "Policy head column locality (%s)" % mid,
                   "claim": "Weight mass of column j's policy logit on board column j relative to the per-column average: %s (1.0 = no locality)." % (
                       ", ".join("%.2f" % v for v in h["policy_locality"])),
                   "evidence": "own-column share of |w|: " + ", ".join("%.2f" % v for v in h["policy_own_col_frac"]),
                   "caveat": "Weights only; the policy conv channels already mix information from the whole board."})
    # patching
    for mid, p in R["patching"].items():
        cols = [r for r in p["per_column"] if "add" in r]
        if not cols:
            continue
        za = [r["add"]["z"] for r in cols]
        zs = [r["subtract"]["z"] for r in cols if "subtract" in r]
        F_.append({"id": "interp-patch-" + mid, "title": "Probe-direction patching (%s)" % mid,
                   "claim": "Adding the 'opponent wins at column j' probe direction at %s changes column j's relative logit by %s (z vs %d random directions: mean %.1f); subtracting it on must-block-at-j positions: z mean %s." % (
                       p["layer"], ", ".join("%+.2f" % r["add"]["d_logit_rel"] for r in cols), p["random_draws"],
                       float(np.mean(za)), "%.1f" % float(np.mean(zs)) if zs else "n/a"),
                   "evidence": "; ".join("col %d: add d_prob %+.3f, sub d_prob %s" % (
                       r["col"], r["add"]["d_prob"], "%+.3f" % r["subtract"]["d_prob"] if "subtract" in r else "n/a") for r in cols),
                   "caveat": "Step size = gap between class means along the direction; random directions have equal norm."})
    # cross-model
    cm = R["cross_model"]
    for mid, ck in cm.get("cka", {}).items():
        F_.append({"id": "interp-cka-" + mid, "title": "CKA %s vs %s" % (PRIMARY, mid),
                   "claim": "Linear CKA of matching layers: " + ", ".join("%s %.2f" % kv for kv in ck["diag"].items()) + ".",
                   "evidence": "full layer x layer matrix in cross_model.cka.%s" % mid,
                   "caveat": "CKA is invariant to rotations of channel space but not to spatial permutations."})
    for mid, diffs in cm.get("concept_diffs", {}).items():
        big = [d for d in diffs if abs(d["diff_norm"]) >= 0.15]
        F_.append({"id": "interp-diff-" + mid, "title": "Concepts %s has and %s lacks (or vice versa)" % (PRIMARY, mid),
                   "claim": ("Concepts whose best normalised probe skill differs by >=0.15: " +
                             ", ".join("%s (%s %.3f vs %.3f)" % (d["concept"], met[d["concept"]], d["primary_best"], d["other_best"]) for d in big))
                   if big else "No concept's best normalised probe skill differs by >=0.15.",
                   "evidence": ", ".join("%s %+.3f" % (d["concept"], d["diff_norm"]) for d in diffs[:8]),
                   "caveat": "net_move / opp_net_move / opp_reply labels are model-specific."})
    # concepts already linear in the raw input (probe on input reaches >= 0.98 normalised skill)
    triv = [c for c, v in S.items() if norm_skill(v["input_baseline"], v["majority"]) >= 0.98]
    F_.append({"id": "interp-trivial", "title": "Concepts that need no computation",
               "claim": "%s: %s are already decodable from the raw input planes by a linear probe (input score >= 98%% of perfect "
                        "normalised skill), so their layer profile only shows how long the network keeps them, not where it computes them." % (
                            m, ", ".join("%s (%s %.3f)" % (c, met[c], S[c]["input_baseline"]) for c in triv)),
               "evidence": "; ".join("%s at stem/%s: %s" % (c, "block%d" % max(1, len([l for l in pr[m][c] if l.startswith("block")])),
                                                            ", ".join("%.2f" % pr[m][c][l]["flat"] for l in ("stem", "block1", "block%d" % len([l for l in pr[m][c] if l.startswith("block")])) if l in pr[m][c]))
                                     for c in triv),
               "caveat": "Pooled (position-independent) versions of heights / landing_row are much weaker; see probe_summary.pooled_best."})
    # opponent-move concepts versus their input baseline, all models
    ev = []
    for mm, sm in ps.items():
        for c in ("opp_reply", "opp_net_move", "net_move"):
            if c in sm:
                ev.append("%s %s: best %.3f at %s vs input %.3f, shuffled max %.3f" % (
                    mm, c, sm[c]["best"], sm[c]["best_layer"], sm[c]["input_baseline"], sm[c]["shuffled_max"]))
    if "opp_reply" in S:
        o = S["opp_reply"]
        F_.append({"id": "interp-oppmove", "title": "Own move vs opponent's move",
                   "claim": "%s: its own move is linearly decodable (net_move acc %.3f at %s vs %.3f on raw input); the solver's best opponent reply after that move is weak "
                            "(opp_reply AUC %.3f at %s vs %.3f on raw input, %.3f shuffled max); the net's own choice on the colour-flipped board reaches acc %.3f at %s (input %.3f)." % (
                                m, S["net_move"]["best"], S["net_move"]["best_layer"], S["net_move"]["input_baseline"],
                                o["best"], o["best_layer"], o["input_baseline"], o["shuffled_max"],
                                S["opp_net_move"]["best"], S["opp_net_move"]["best_layer"], S["opp_net_move"]["input_baseline"]),
                   "evidence": " | ".join(ev),
                   "caveat": "opp_reply uses ~3k solved positions per model (see checks), so its probe is data-limited; for az/2x32/ppo no layer beats the raw input on it."})
    # single most important threat channel: largest missed-block effect of an opponent-threat group
    for mm, rows in R["ablation"]["groups"].items():
        cand = [r for r in rows if r["mode"] == "mean" and r["group"] == "opponent-threat" and r["effect"].get("missed_block") is not None]
        cand_w = [r for r in rows if r["mode"] == "mean" and r["group"] == "own-threat" and r["effect"].get("missed_win") is not None]
        parts = []
        if cand:
            r = max(cand, key=lambda r: r["effect"]["missed_block"])
            chs = [c["ch"] for c in R["channels"][mm][r["layer"]] if c["label"] == "opponent-threat"]
            parts.append("opponent-threat channel(s) %s at %s: missed-block %+.3f vs missed-win %+.3f (null pct %.2f for missed-block)" % (
                chs, r["layer"], r["effect"]["missed_block"], r["effect"]["missed_win"], r["null"]["missed_block"]["pct"]))
        if cand_w:
            r = max(cand_w, key=lambda r: r["effect"]["missed_win"])
            chs = [c["ch"] for c in R["channels"][mm][r["layer"]] if c["label"] == "own-threat"]
            parts.append("own-threat channels %s at %s: missed-win %+.3f vs missed-block %+.3f (null pct %.2f for missed-win)" % (
                chs, r["layer"], r["effect"]["missed_win"], r["effect"]["missed_block"], r["null"]["missed_win"]["pct"]))
        if parts:
            F_.append({"id": "interp-threatch-" + mm, "title": "Strongest threat-channel ablation (%s)" % mm,
                       "claim": "Mean-ablating " + "; ".join(parts) + ".",
                       "evidence": "ablation.groups.%s (mode mean)" % mm,
                       "caveat": "Single-channel groups are small; compare with the %d-draw random null in the same entry." % NULL_DRAWS})
    del lab
    return F_


# ============================================================================ main
def main(argv=None):
    global NULL_DRAWS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--n", type=int, default=50000, help="positions (70/10/20 train/val/test)")
    ap.add_argument("--opp-n", type=int, default=4000, help="positions solved for the opponent-reply concept")
    ap.add_argument("--rounds", type=int, default=10, help="probe L-BFGS rounds of 10 iterations")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-solver", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "interp.json"))
    a = ap.parse_args(argv)
    if a.quick:
        a.n, a.opp_n, a.rounds, NULL_DRAWS = 4000, 200, 4, 3
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    rng = np.random.default_rng(a.seed)

    cur, mask, scores, ply = load_data(a.n, a.seed)
    X = encode(cur, mask)
    c, extra = compute_concepts(cur, mask, scores, ply)
    tr, va, te = splits(a.n)
    base_labels = concept_labels(c)
    kinds = {cid: (kind, K) for cid, _, _, kind, K in CONCEPTS}
    log("data: %d positions (%d/%d/%d), device %s" % (a.n, len(tr), len(va), len(te), dev))

    R = {"id": "interp", "title": "Mechanistic interpretability: which layer knows what",
         "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": [], "models": [],
         "concepts": [], "probes": {}, "probe_summary": {}, "channels": {}, "channel_counts": {},
         "ablation": {"blocks": {}, "groups": {}, "top_channels": {}, "intact": {}},
         "heads": {}, "patching": {}, "cross_model": {}, "checks": {}, "prevalence": {}}
    heavy = {"channels": {}, "single_channel_ablation": {}, "heads": {}}
    nets, summaries = {}, {}
    for cid, label, group, kind, K in CONCEPTS:
        R["concepts"].append({"id": cid, "label": label, "group": group, "metric": METRIC[kind],
                              "kind": kind, "baseline": None})
    for cid in ("win_now", "opp_win_now", "own_double", "opp_double", "first_to_move"):
        y = base_labels[cid]
        R["prevalence"][cid] = float(y.any(1).mean())
    R["prevalence"]["phase"] = np.bincount(c["phase"], minlength=3) / len(c["phase"])
    R["prevalence"]["outcome_wdl"] = np.bincount(c["outcome"], minlength=3) / len(c["outcome"])
    solver_cache = {}

    for mid in a.models:
        t0 = time.time()
        path = os.path.join(ROOT, MODELS.get(mid, mid))
        net = TorchNet(path, dev)
        nets[mid] = net
        err = verify_forward(net, X)
        assert err < 1e-3, "torch forward disagrees with NumpyNet (%g)" % err
        bacc, bn = bench_agreement(net)
        R["checks"][mid] = {"forward_max_abs_diff_vs_numpy": err, "bench_agreement": bacc, "bench_n": bn}
        log("%s: forward check %.2e, benchmark agreement %.4f (%d)" % (mid, err, bacc, bn))
        R["models"].append({"id": mid, "run": MODELS.get(mid, mid), "blocks": net.blocks,
                            "channels": net.channels, "params": net.params,
                            "layers": ["input"] + net.layers})
        legal = extra["legal"]
        lg = run_batched(net, X)[0].cpu().numpy()
        lg[~legal] = -np.inf
        lgf = run_batched(net, flip_planes(X))[0].cpu().numpy()
        lgf[~legal] = -np.inf
        labels = dict(base_labels)
        labels["net_move"] = lg.argmax(1)
        labels["opp_net_move"] = lgf.argmax(1)
        sub = None
        if not a.no_solver and a.opp_n > 0:
            from .interp_solver import opponent_best_replies
            k_tr, k_va, k_te = int(0.7 * a.opp_n), int(0.1 * a.opp_n), int(0.2 * a.opp_n)
            sidx = np.concatenate([tr[:k_tr], va[:k_va], te[:k_te]])
            ts = time.time()
            opt, ok, nnew = opponent_best_replies(cur[sidx], mask[sidx], labels["net_move"][sidx],
                                                  workers=4, cache=solver_cache)
            y = np.zeros((a.n, 7), bool)
            y[sidx] = opt
            labels["opp_reply"] = y
            okset = set(sidx[ok].tolist())
            sub = tuple(np.array([i for i in s if i in okset]) for s in (tr[:k_tr], va[:k_va], te[:k_te]))
            R["checks"][mid]["opp_reply"] = {"requested": int(len(sidx)), "labelled": int(ok.sum()),
                                             "game_over_after_move_or_timeout": int((~ok).sum()),
                                             "new_solver_calls": nnew, "seconds": time.time() - ts,
                                             "train": len(sub[0]), "val": len(sub[1]), "test": len(sub[2])}
            log("  opp_reply: %d labelled of %d (%.0fs)" % (ok.sum(), len(sidx), time.time() - ts))
        chance = {}
        for cid, y in labels.items():
            kind, K = kinds[cid]
            if cid in SUBSET_CONCEPTS:
                chance[cid] = chance_score(kind, y[sub[0]], y[sub[2]], K)
            else:
                chance[cid] = chance_score(kind, y[tr], y[te], K)
        probes = probe_model(net, X, labels, kinds, tr, va, te, sub, a)
        R["probes"][mid] = probes
        summaries[mid] = summarise_probes(probes, ["input"] + net.layers, chance)
        R["probe_summary"][mid] = summaries[mid]
        log("  probes done %.0fs" % (time.time() - t0))

        cen = census(net, X, c, extra, te)
        maps = {}
        R["channels"][mid] = {}
        R["channel_counts"][mid] = {}
        for layer, rows in cen.items():
            maps[layer] = [{"map": r["_map"], "row": r["_map"].mean(1), "col": r["_map"].mean(0)} for r in rows]
            R["channels"][mid][layer] = [{k: v for k, v in r.items() if k != "_map"} for r in rows]
            cnt = {}
            for r in rows:
                cnt[r["label"]] = cnt.get(r["label"], 0) + 1
            R["channel_counts"][mid][layer] = cnt
        heavy["channels"][mid] = maps
        log("  census done %.0fs" % (time.time() - t0))

        ctx = Ctx(c, extra, te, dev)
        base, blocks, groups, single = ablations(net, X, ctx, cen, te, rng)
        R["ablation"]["intact"][mid] = base
        R["ablation"]["blocks"][mid] = blocks
        R["ablation"]["groups"][mid] = groups
        R["ablation"]["top_channels"][mid] = top_channels(single, cen)
        heavy["single_channel_ablation"][mid] = single
        log("  ablations done %.0fs" % (time.time() - t0))

        h = head_attribution(net)
        R["heads"][mid] = {k: v for k, v in h.items()}
        stream = [l for l in net.stream_layers if l in probes["opp_win_now"]]
        pl = max(stream, key=lambda l: probes["opp_win_now"][l]["flat"])
        R["patching"][mid] = patching(net, X, labels, tr, va, te, pl, ctx, rng)
        log("  %s done in %.0fs" % (mid, time.time() - t0))

    R["cross_model"] = cross_model(nets, X, te[:5000], summaries, kinds)
    for cobj in R["concepts"]:
        if PRIMARY in summaries and cobj["id"] in summaries[PRIMARY]:
            cobj["baseline"] = summaries[PRIMARY][cobj["id"]]["majority"]
    R["notes"] = [
        "Positions: fixed-seed (seed %d) random subset of nn_data/solver_300k.npz, n=%d split 70/10/20 into train/val/test (%d/%d/%d). "
        "All probe scores are on the test split; ablation/census/patching use the test split." % (a.seed, a.n, len(tr), len(va), len(te)),
        "Row convention: row 0 = TOP row of the board (as in nn/encode.py and the website grid). 'Odd/even rows' are counted 1..6 from the BOTTOM (zugzwang convention).",
        "Input planes: me, opponent, ones, first-player-to-move flag; everything is from the side-to-move's perspective.",
        "Forward pass re-implemented in torch from the exported .npz (BatchNorm folded); max |diff| vs nn/numpy_net.py is in checks.",
        "Probes: linear (logistic / softmax / least-squares) on standardised features, full-batch L-BFGS from zero init, L2 1e-4 (1e-3 for the small opp_reply subset), %d rounds x 10 iterations; "
        "each head keeps the test score of its best validation epoch. flat = all C*6*7 activations; pooled = per-channel spatial mean; "
        "shuffled = same probe on flat features with permuted train+val labels (test labels real)." % a.rounds,
        "Metrics: auc = ROC AUC averaged over columns (binary concepts are single-column), acc = accuracy (multicat: mean over columns), r2 = R^2 averaged over outputs. "
        "'majority' = best constant predictor (auc 0.5).",
        "Normalised skill = (score - chance)/(1 - chance). first_layer = first layer (order input, stem, blocks, pol_conv, val_conv, val_hidden) reaching %.0f%% of the best normalised skill; "
        "survives_pooling = best pooled skill >= %.0f%% of best flat skill." % (FIRST_FRAC * 100, POOL_FRAC * 100),
        "opp_reply: the net's argmax move is played and the successor solved with Pons' solver; label = the opponent's optimal set. "
        "Requested opp-n=%d per model (70/10/20 from the same splits); positions where the net's move ends the game are dropped (see checks.<model>.opp_reply)." % a.opp_n,
        "opp_net_move: the same net's argmax on the colour-flipped input (planes 0/1 swapped, plane 3 toggled) - an off-distribution input. "
        "opp_win_now doubles as opp-move concept (c): the opponent's immediate-win columns = must-block columns.",
        "Threat square = empty cell that would complete four for that side (playable or not). Double threat = 2+ playable immediate wins, or a playable threat with a second threat directly above it.",
        "Channel labels (stem and block outputs, test split): dead if active (>0) on <%.0f%% of (position,cell) samples; candidates own-/opponent-threat if Pearson r with that side's threat or immediate-win cells, computed over EMPTY cells only, >= %.1f and exceeds the other side's; "
        "occupancy copy if |r| with mine/theirs/empty cells >= %.1f; column (row/height) detector if the mean 6x7 map holds >= %.0f%% of total activation variance and column (row) main effects hold >= %.0f%% of the map's variance; "
        "phase: early/late if |r| of the spatially pooled activation with ply >= %.1f (note: ply = stone count, so this also catches channels that simply grow with the number of stones). "
        "Priority when several apply: threat > occupancy > column > row > phase; otherwise 'mixed'." % (
            DEAD_ACTIVE * 100, THR_R, OCC_R, POS_FRAC * 100, SEL * 100, PHASE_R),
        "Channel content means: [empty playable, empty not playable, mine, theirs]. phase = mean pooled activation [early, mid, late]; phase_idx = (late-early)/(|late|+|early|).",
        "Ablation metrics: agree = argmax (legal) in solver-optimal set; value_acc = argmax WDL == solver outcome; kl = KL(intact || ablated) policy over legal columns; dv = mean |change in P(win)-P(loss)|; "
        "missed_win over positions with an own immediate win; missed_block over positions with no own win and exactly one opponent win column. Reported as ablated minus intact.",
        "Block modes: identity (block skipped), zero_branch (residual branch zeroed: identical to identity because block inputs are post-ReLU), mean_branch (branch replaced by its mean over the test split).",
        "Group ablation: channels of one census label in one layer set to 0 (zero) or their per-cell test-split mean (mean); null = %d random equal-size sets of live channels from the same layer (excluding the group when possible). pct = fraction of null draws below the effect." % NULL_DRAWS,
        "Single-channel ablation uses mean ablation; per-channel arrays in interp_channels.json.",
        "Patching: probe direction for opp_win_now column j (logistic weights, raw-activation units, normalised) at the stream layer where that probe is best; step = gap between positive/negative class means along it; "
        "'add' on quiet positions (no immediate win for either side, column j legal), 'subtract' on positions where j is the unique must-block column; d_logit_rel = change in logit j minus mean change of other logits; 20 random unit directions with the same step.",
        "CKA: linear CKA on %d test positions between every layer of %s and every layer of each other model." % (min(5000, len(te)), PRIMARY),
        "params = parameters of the exported .npz (BatchNorm folded into convolutions).",
        "Runtime: %.0f s total." % (time.time() - t_start),
    ]
    R["findings"] = make_findings(R)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rnd(R), f, separators=(",", ":"))
    hp = os.path.join(os.path.dirname(a.out), os.path.basename(a.out).replace(".json", "_channels.json"))
    with open(hp, "w") as f:
        json.dump(rnd(heavy, 3), f, separators=(",", ":"))
    log("wrote %s (%.0f KB) and %s (%.0f KB) in %.0fs" % (a.out, os.path.getsize(a.out) / 1e3, hp,
                                                          os.path.getsize(hp) / 1e3, time.time() - t_start))


if __name__ == "__main__":
    main()
