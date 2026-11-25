"""PCA input reduction and LDA-focused training data, each tested with nn.evaluate.

Everything writes to experiments/results/research/pca_lda/ and trains into runs/<name>/,
so the nets show up on the Lab page like any other run.  The training data is
nn_data/solver_300k.npz by default (fast: ~7 s/epoch for 6x64 on the RTX 3050); pass
--data nn_data/solver_3m.npz for the full-size version.

PCA  (does the net need the whole board, or does a low-rank view of it do?)
    python -m research.pca_lda pca-variance                 # CPU, seconds: variance kept per K
    python -m research.pca_lda pca-sweep                    # print the train + evaluate commands
    python -m research.pca_lda pca-sweep --go               # run them (resumable, GPU)
    python -m research.pca_lda pca-summary                  # table + chart of accuracy vs K
  Each net is nn/train_score.py with --pca K: the 84 stone cells are replaced by their rank-K
  PCA reconstruction before the conv stem (nn/model.py).  K=0 is the normal net (control).
  The projection is stored in model.npz, so nn.evaluate, the arena and the website run
  these nets unchanged.

LDA  (which positions are linearly hard, and does focusing training on them help?)
    python -m research.pca_lda lda-analyze                  # CPU, ~1 min: how separable are
                                                            #   outcome / best column, which cells matter
    python -m research.pca_lda lda-analyze --run score_8x128_s0_r1   # + the same on the net's trunk features
    python -m research.pca_lda lda-focus                    # CPU: write the focus + control subsets
    python -m research.pca_lda lda-sweep --go               # GPU: baseline / focus / control nets + nn.evaluate
    python -m research.pca_lda lda-summary
  lda-focus fits LDA (classes = win/draw/loss under perfect play, or best column) on the
  training positions and keeps the --frac positions where LDA is LEAST sure (smallest gap
  between its top two class posteriors; --select misclassified keeps the ones it gets
  wrong, --select class:draw keeps one class).  They are written in nn/mine_hard.py format,
  so nn.train_score --hard <file> --hard-weight W up-weights them.  A random subset of the
  same size is the control: if focus beats control, the LDA selection itself mattered,
  not just "extra weight on some positions".
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

from nn.common import NN_DATA, ROOT, RUNS, log, now_iso

OUT = os.path.join(ROOT, "experiments", "results", "research", "pca_lda")
DEFAULT_DATA = os.path.join(NN_DATA, "solver_300k.npz")
ILLEGAL = -128
OUTCOMES = ("win", "draw", "loss")


# ====================================================================== data
def load(path, max_rows=0, seed=0):
    z = np.load(path)
    cur, mask, scores, ply = z["cur"].astype(np.uint64), z["mask"].astype(np.uint64), z["scores"], z["ply"]
    if max_rows and len(cur) > max_rows:
        idx = np.sort(np.random.default_rng(seed).choice(len(cur), max_rows, replace=False))
        cur, mask, scores, ply = cur[idx], mask[idx], scores[idx], ply[idx]
    return cur, mask, scores, ply


def labels(scores, kind):
    sc = scores.astype(np.int16)
    legal = sc != ILLEGAL
    best = np.where(legal, sc, -1000).max(1)
    if kind == "outcome":
        return np.where(best > 0, 0, np.where(best == 0, 1, 2)), list(OUTCOMES)
    if kind == "bestcol":       # most central optimal column (unique label per position)
        order = (3, 2, 4, 1, 5, 0, 6)
        opt = (sc == best[:, None]) & legal
        lab = np.full(len(sc), -1)
        for c in order[::-1]:
            lab[opt[:, c]] = c
        return lab, ["col%d" % (c + 1) for c in range(7)]
    raise ValueError(kind)


def cell_features(cur, mask):
    """84 stone cells (me, them; row 0 = top, like nn/encode.py) + 'I moved first' flag."""
    from nn.encode import bits_to_grid
    me, them = bits_to_grid(cur), bits_to_grid(cur ^ mask)
    n = len(cur)
    first = me.reshape(n, -1).sum(1) == them.reshape(n, -1).sum(1)
    return np.concatenate([me.reshape(n, -1), them.reshape(n, -1), first[:, None]], 1).astype(np.float32)


def hand_features(cur, mask, ply):
    """The 10 heuristic features of rl/features.py (open threes, threats, odd/even threats...) + ply."""
    from rl.features import features
    f = np.array([features(int(c), int(c ^ m)) for c, m in zip(cur, mask)], np.float32)
    return np.concatenate([f, ply[:, None].astype(np.float32)], 1)


def hand_names():
    from rl.features import NAMES
    return list(NAMES) + ["ply"]


def trunk_features(npz_path, cur, mask, device="auto", batch=4096):
    """Output of the net's residual trunk: per-channel mean over the board + per-column means
    (C + 7C numbers per position).  Shows how linearly separable the net has made things."""
    import torch
    import torch.nn.functional as F
    from nn.encode import encode
    from nn.torch_net import TorchNpzNet
    from nn.common import pick_device
    net = TorchNpzNet(npz_path, pick_device(device))
    w = net.w
    out = []
    with torch.inference_mode():
        for i in range(0, len(cur), batch):
            x = torch.from_numpy(encode(cur[i:i + batch], mask[i:i + batch])).to(net.device)
            if net.has_proj:
                b = x.shape[0]
                z = x[:, :2].reshape(b, -1)
                z = (z - w["inproj.mean"]) @ w["inproj.U"] @ w["inproj.U"].T + w["inproj.mean"]
                x = torch.cat([z.reshape(b, 2, 6, 7), x[:, 2:]], 1)
            h = F.relu(F.conv2d(x, w["stem.w"], w["stem.b"], padding=1))
            for k in range(net.blocks):
                y = F.relu(F.conv2d(h, w["b%d.c1.w" % k], w["b%d.c1.b" % k], padding=1))
                h = F.relu(h + F.conv2d(y, w["b%d.c2.w" % k], w["b%d.c2.b" % k], padding=1))
            g = h.mean((2, 3))
            col = h.mean(2).flatten(1)
            out.append(torch.cat([g, col], 1).float().cpu().numpy())
    return np.concatenate(out)


def save_json(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    log("wrote %s" % os.path.relpath(p, ROOT))
    return p


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run_cmd(cmd, log_path):
    """Run a python -m command, tee to log_path; returns exit code."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log("$ " + " ".join(cmd))
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("\n# %s  %s\n" % (now_iso(), " ".join(cmd)))
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        return p.wait()


def train_and_eval(name, train_args, eval_args, go):
    """Shared queue step: train runs/<name> (unless best.npz exists) then nn.evaluate it."""
    py = sys.executable
    train = [py, "-m", "nn.train_score", "--run", name] + train_args
    ev = [py, "-m", "nn.evaluate", "--run", name, "--model", "best.npz"] + eval_args
    if not go:
        print(" ".join(["python"] + train[1:]))
        print(" ".join(["python"] + ev[1:]))
        return True
    logp = os.path.join(RUNS, "pca_lda_logs", name + ".log")
    rd = os.path.join(RUNS, name)
    if os.path.isfile(os.path.join(rd, "eval.json")):
        log("%s: done (eval.json exists)" % name)
        return True
    if not os.path.isfile(os.path.join(rd, ".done")):
        if os.path.isdir(rd):      # unfinished earlier attempt: park it (runs/_* is git-ignored)
            park = os.path.join(RUNS, "_unfinished_%s_%d" % (name, int(time.time())))
            os.rename(rd, park)
            log("%s: moved an unfinished earlier attempt to %s" % (name, os.path.relpath(park, ROOT)))
        if run_cmd(train, logp) != 0:
            log("%s: TRAINING FAILED, see %s" % (name, logp))
            return False
        open(os.path.join(rd, ".done"), "w").close()
    if run_cmd(ev, logp) != 0:
        log("%s: EVALUATION FAILED, see %s" % (name, logp))
        return False
    return True


def eval_args(a):
    out = ["--backend", a.backend, "--sims", a.sims]
    if not a.full_eval:
        out += ["--skip", "ladder,hypothesis"]
    return out


def collect(names):
    rows = []
    for n in names:
        p = os.path.join(RUNS, n, "eval.json")
        if not os.path.isfile(p):
            rows.append({"run": n, "status": "missing"})
            continue
        with open(p, encoding="utf-8") as f:
            e = json.load(f)
        acc = e.get("accuracy", {})
        pol = acc.get("nn_policy", {})
        r = {"run": n, "status": "ok", "policy_optimal": pol.get("optimal"),
             "policy_optimal_ci95": pol.get("optimal_ci95"), "policy_keeps_nl": pol.get("keeps_nl"),
             "value_acc": (e.get("value") or {}).get("accuracy")}
        for k, v in acc.items():
            if k.startswith("nn_mcts") and isinstance(v, dict):
                r[k + "_optimal"] = v.get("optimal")
        tac = (e.get("tactics") or {}).get("nn_policy")
        if tac:
            r["tactics_policy"] = tac.get("all")
        lad = e.get("ladder") or {}
        if lad.get("elo") is not None:
            r["ladder_elo"] = lad["elo"]
        cfg_p = os.path.join(RUNS, n, "config.json")
        if os.path.isfile(cfg_p):
            with open(cfg_p, encoding="utf-8") as f:
                cfg = json.load(f)
            r["params"] = cfg.get("params")
            if cfg.get("pca"):
                r["pca_k"] = cfg["pca"]["k"]
                r["pca_explained_variance"] = cfg["pca"]["explained_variance"]
            r["train_minutes"] = None
        mp = os.path.join(RUNS, n, "metrics.jsonl")
        if os.path.isfile(mp):
            with open(mp, encoding="utf-8") as f:
                last = [json.loads(l) for l in f if l.strip()]
            if last:
                r["train_minutes"] = round(last[-1].get("elapsed_sec", 0) / 60, 1)
        rows.append(r)
    return rows


def write_table(name, rows):
    os.makedirs(OUT, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    p = os.path.join(OUT, name)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    log("wrote %s" % os.path.relpath(p, ROOT))


# ======================================================================= PCA
def cmd_pca_variance(a):
    from nn.train_score import fit_input_pca
    from nn.encode import bits_to_grid
    cur, mask, _, _ = load(a.data, a.max_rows)
    grids = np.stack([bits_to_grid(cur), bits_to_grid(cur ^ mask)], 1)
    info = fit_input_pca(grids, 84)
    cum = np.cumsum(info["explained_by_component"]).tolist()
    ks = [k for k in (4, 8, 12, 16, 24, 32, 48, 64, 84)]
    res = {"data": os.path.relpath(a.data, ROOT), "positions": int(len(cur)),
           "cumulative_explained": [round(c, 5) for c in cum],
           "at_k": {k: round(cum[k - 1], 4) for k in ks},
           "k_for": {"%d%%" % p: int(np.searchsorted(cum, p / 100) + 1) for p in (50, 75, 90, 95, 99)},
           "note": "variance of the 84 stone cells (me + them planes, mirrored copies included)"}
    print("components needed:", res["k_for"])
    print("variance kept at K:", res["at_k"])
    save_json("pca_variance.json", res)
    # first components as boards (what PCA thinks the main directions are)
    plt = _plt()
    from nn.train_score import fit_input_pca as fit
    U = fit(grids, 8)["components"]
    fig, axs = plt.subplots(2, 8, figsize=(16, 4.4))
    for j in range(8):
        for r, nm in enumerate(("me", "them")):
            v = U[r * 42:(r + 1) * 42, j].reshape(6, 7)
            m = np.abs(U[:, j]).max()
            axs[r, j].imshow(v, cmap="RdBu_r", vmin=-m, vmax=m)
            axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
            if r == 0:
                axs[r, j].set_title("PC%d (%.1f%%)" % (j + 1, 100 * info["explained_by_component"][j]))
            if j == 0:
                axs[r, j].set_ylabel(nm)
    fig.suptitle("First 8 principal components of the stone cells (top row of each board = top of the board)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pca_components.png"), dpi=110)
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, 85), cum)
    ax.set_xlabel("components K"); ax.set_ylabel("variance kept"); ax.grid(alpha=.3)
    ax.set_title("PCA of the 84 stone cells")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "pca_variance.png"), dpi=110)
    log("wrote pca_components.png, pca_variance.png")


def pca_names(a):
    b, c = a.size.split("x")
    out = []
    for s in [int(x) for x in a.seeds.split(",") if x]:
        for k in [int(x) for x in a.ks.split(",") if x != ""]:
            out.append(("pca_k%d_%s_s%d" % (k, a.size, s), k, s, b, c))
    return out


def cmd_pca_sweep(a):
    if not a.go:
        print("# PCA sweep: %d nets. Run on the GPU PC (or add --go to run them one after another):" % len(pca_names(a)))
    ok = True
    for name, k, s, b, c in pca_names(a):
        targs = ["--data", os.path.relpath(a.data, ROOT), "--blocks", b, "--channels", c,
                 "--epochs", str(a.epochs), "--seed", str(s)] + (["--pca", str(k)] if k else [])
        ok &= train_and_eval(name, targs + a.train_args.split(), eval_args(a), a.go)
    if a.go:
        cmd_pca_summary(a)
    return 0 if ok else 1


def cmd_pca_summary(a):
    names = [n for n, *_ in pca_names(a)]
    rows = collect(names)
    for r, (_, k, s, _, _) in zip(rows, pca_names(a)):
        r["k"] = k if k else 84
        r["projection"] = bool(k)
        r["seed"] = s
    write_table("pca_summary.csv", rows)
    save_json("pca_summary.json", {"created": now_iso(), "data": os.path.relpath(a.data, ROOT), "size": a.size,
                                   "rows": rows})
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for proj in (True, False):
            pts = sorted((r["k"], r["policy_optimal"], r["policy_optimal_ci95"]) for r in ok if r["projection"] == proj)
            if not pts:
                continue
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            err = [[p[1] - p[2][0] for p in pts], [p[2][1] - p[1] for p in pts]] if all(p[2] for p in pts) else None
            ax.errorbar(x, y, yerr=err, marker="o", ls="-" if proj else "", capsize=3,
                        label="PCA input, K components" if proj else "no projection (control)")
        ax.set_xlabel("K"); ax.set_ylabel("policy optimal on the benchmark"); ax.grid(alpha=.3); ax.legend()
        ax.set_title("Accuracy vs PCA input rank (%s)" % a.size)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "pca_accuracy.png"), dpi=110)
        log("wrote pca_accuracy.png")
        for r in sorted(ok, key=lambda r: r["k"]):
            print("K=%-3s %-24s optimal %.4f %s" % (r["k"] if r["projection"] else "-", r["run"], r["policy_optimal"],
                                                    r.get("policy_optimal_ci95")))


# ======================================================================= LDA
def features_for(kind, cur, mask, ply, run=None, device="auto"):
    if kind == "cells":
        return cell_features(cur, mask), ["me_r%dc%d" % (r, c) for r in range(6) for c in range(7)] + \
            ["them_r%dc%d" % (r, c) for r in range(6) for c in range(7)] + ["first"]
    if kind == "hand":
        return hand_features(cur, mask, ply), hand_names()
    if kind == "trunk":
        p = run if run.endswith(".npz") else os.path.join(RUNS, run, "best.npz")
        f = trunk_features(p, cur, mask, device)
        return f, ["trunk%d" % i for i in range(f.shape[1])]
    raise ValueError(kind)


def cmd_lda_analyze(a):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import cross_val_score
    cur, mask, scores, ply = load(a.data, a.max_rows, a.seed)
    kinds = ["cells", "hand"] + (["trunk"] if a.run else [])
    res = {"created": now_iso(), "data": os.path.relpath(a.data, ROOT), "positions": int(len(cur)),
           "run": a.run, "results": {}}
    plt = _plt()
    for lab_kind in ("outcome", "bestcol"):
        y, cls = labels(scores, lab_kind)
        maj = float(np.bincount(y).max() / len(y))
        for fk in kinds:
            t0 = time.time()
            X, names = features_for(fk, cur, mask, ply, a.run, a.device)
            lda = LDA(solver="svd")
            cv = cross_val_score(LDA(solver="svd"), X, y, cv=5, n_jobs=1)
            lda.fit(X, y)
            Z = lda.transform(X)
            r = {"labels": lab_kind, "classes": cls, "features": fk, "dims": int(X.shape[1]),
                 "cv_accuracy": round(float(cv.mean()), 4), "cv_std": round(float(cv.std()), 4),
                 "majority_baseline": round(maj, 4),
                 "explained_variance_ratio": [round(float(v), 4) for v in lda.explained_variance_ratio_],
                 "seconds": round(time.time() - t0, 1)}
            if fk == "hand":
                coef = lda.scalings_[:, 0] / (np.abs(lda.scalings_[:, 0]).max() + 1e-12)
                r["ld1_weights"] = {n: round(float(c), 3) for n, c in zip(names, coef)}
            res["results"]["%s/%s" % (lab_kind, fk)] = r
            log("LDA %-8s on %-5s (%4d dims): CV accuracy %.3f (majority class %.3f), LD variance %s" % (
                lab_kind, fk, X.shape[1], r["cv_accuracy"], maj, r["explained_variance_ratio"][:3]))
            # scatter of the first two discriminants
            if Z.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(5.5, 5))
                sub = np.random.default_rng(0).choice(len(Z), min(20000, len(Z)), replace=False)
                for ci, cn in enumerate(cls):
                    m = y[sub] == ci
                    if m.any():
                        ax.scatter(Z[sub][m, 0], Z[sub][m, 1], s=2, alpha=.35, label=cn)
                ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.legend(markerscale=6, fontsize=8)
                ax.set_title("LDA: %s from %s features (CV acc %.2f)" % (lab_kind, fk, r["cv_accuracy"]))
                fig.tight_layout()
                fig.savefig(os.path.join(OUT, "lda_%s_%s.png" % (lab_kind, fk)), dpi=110)
                plt.close(fig)
            if fk == "cells":   # LD1 as a board: which cells push towards each class
                v = lda.scalings_[:, 0]
                m = np.abs(v[:84]).max()
                fig, axs = plt.subplots(1, 2, figsize=(7, 3))
                for k, nm in enumerate(("me", "them")):
                    axs[k].imshow(v[k * 42:(k + 1) * 42].reshape(6, 7), cmap="RdBu_r", vmin=-m, vmax=m)
                    axs[k].set_title("LD1 weight, %s stones" % nm); axs[k].set_xticks(range(7))
                    axs[k].set_xticklabels(range(1, 8)); axs[k].set_yticks([])
                fig.suptitle("LDA %s: first discriminant on the board" % lab_kind)
                fig.tight_layout()
                fig.savefig(os.path.join(OUT, "lda_%s_cells_ld1_board.png" % lab_kind), dpi=110)
                plt.close(fig)
    save_json("lda_analysis.json", res)


def cmd_lda_focus(a):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    cur, mask, scores, ply = load(a.data)
    y, cls = labels(scores, a.label)
    X = np.concatenate([features_for(k, cur, mask, ply)[0] for k in a.features.split("+")], 1)
    fit_idx = np.random.default_rng(a.seed).choice(len(X), min(len(X), a.max_rows or len(X)), replace=False)
    lda = LDA(solver="svd").fit(X[fit_idx], y[fit_idx])
    post = lda.predict_proba(X)
    top2 = np.sort(post, 1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    pred = post.argmax(1)
    n_keep = int(round(a.frac * len(X)))
    if a.select == "uncertain":
        keep = np.argsort(margin, kind="stable")[:n_keep]
    elif a.select == "misclassified":
        keep = np.flatnonzero(pred != y)
        if len(keep) > n_keep:
            keep = keep[np.argsort(margin[keep], kind="stable")[:n_keep]]
    elif a.select.startswith("class:"):
        want = cls.index(a.select.split(":", 1)[1])
        keep = np.flatnonzero(y == want)
    else:
        raise SystemExit("--select must be uncertain, misclassified or class:<name>")
    keep = np.sort(keep)
    rng = np.random.default_rng(a.seed + 1)
    ctrl = np.sort(rng.choice(len(X), len(keep), replace=False))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ctrl_out = a.out.replace(".npz", "_control.npz")
    for path, idx in ((a.out, keep), (ctrl_out, ctrl)):
        np.savez_compressed(path, cur=cur[idx], mask=mask[idx], scores=scores[idx], ply=ply[idx],
                            meta_json=np.array(json.dumps({"from": os.path.relpath(a.data, ROOT), "select": a.select,
                                                           "label": a.label, "features": a.features,
                                                           "control": path == ctrl_out})))
    def desc(idx):
        yy = y[idx]
        return {"positions": int(len(idx)), "class_share": {c: round(float((yy == i).mean()), 4) for i, c in enumerate(cls)},
                "mean_ply": round(float(ply[idx].mean()), 2),
                "lda_accuracy_on_subset": round(float((pred[idx] == yy).mean()), 4),
                "mean_margin": round(float(margin[idx].mean()), 4)}
    info = {"created": now_iso(), "data": os.path.relpath(a.data, ROOT), "label": a.label, "features": a.features,
            "select": a.select, "frac": a.frac, "lda_train_accuracy": round(float((pred == y).mean()), 4),
            "focus_file": os.path.relpath(a.out, ROOT), "control_file": os.path.relpath(ctrl_out, ROOT),
            "all": desc(np.arange(len(X))), "focus": desc(keep), "control": desc(ctrl)}
    save_json("lda_focus_%s.json" % re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(os.path.basename(a.out))[0]), info)
    log("focus: %s" % info["focus"])
    log("control: %s" % info["control"])
    print("next: python -m research.pca_lda lda-sweep --focus %s" % os.path.relpath(a.out, ROOT))


def lda_jobs(a):
    b, c = a.size.split("x")
    tag = os.path.splitext(os.path.basename(a.focus))[0]
    common = ["--data", os.path.relpath(a.data, ROOT), "--blocks", b, "--channels", c,
              "--epochs", str(a.epochs), "--seed", str(a.seed)]
    ctrl = a.focus.replace(".npz", "_control.npz")
    return [("lda_base_%s_s%d" % (a.size, a.seed), common),
            ("lda_%s_w%g_%s_s%d" % (tag, a.hard_weight, a.size, a.seed),
             common + ["--hard", os.path.relpath(a.focus, ROOT), "--hard-weight", str(a.hard_weight)]),
            ("lda_%s_control_w%g_%s_s%d" % (tag, a.hard_weight, a.size, a.seed),
             common + ["--hard", os.path.relpath(ctrl, ROOT), "--hard-weight", str(a.hard_weight)])]


def cmd_lda_sweep(a):
    if not (os.path.isfile(a.focus) and os.path.isfile(a.focus.replace(".npz", "_control.npz"))):
        raise SystemExit("make the subsets first: python -m research.pca_lda lda-focus --out %s" % a.focus)
    if not a.go:
        print("# LDA focus experiment: baseline / LDA-focused / random control (add --go to run):")
    ok = True
    for name, targs in lda_jobs(a):
        ok &= train_and_eval(name, targs + a.train_args.split(), eval_args(a), a.go)
    if a.go:
        cmd_lda_summary(a)
    return 0 if ok else 1


def cmd_lda_summary(a):
    jobs = lda_jobs(a)
    rows = collect([n for n, _ in jobs])
    for r, role in zip(rows, ("baseline", "lda_focus", "random_control")):
        r["role"] = role
    write_table("lda_summary.csv", rows)
    out = {"created": now_iso(), "focus": os.path.relpath(a.focus, ROOT), "rows": rows}
    ok = {r["role"]: r for r in rows if r["status"] == "ok"}
    if "lda_focus" in ok and "random_control" in ok:
        from nn.bench_stats import correct_vector, paired
        from nn import bench
        rws = bench.load_accuracy_bench()
        va, _ = correct_vector(os.path.join(RUNS, ok["lda_focus"]["run"], "best.npz"), rws, a.backend, "auto")
        vb, _ = correct_vector(os.path.join(RUNS, ok["random_control"]["run"], "best.npz"), rws, a.backend, "auto")
        out["focus_vs_control"] = paired(va, vb)
        log("focus vs control: %s" % out["focus_vs_control"])
    save_json("lda_summary.json", out)
    for r in rows:
        print("%-16s %-40s %s" % (r.get("role"), r["run"], r.get("policy_optimal")))


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, gpu=False):
        p.add_argument("--data", default=DEFAULT_DATA)
        p.add_argument("--seed", type=int, default=0)
        if gpu:
            p.add_argument("--size", default="6x64", help="BLOCKSxCHANNELS of the nets")
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument("--go", action="store_true", help="run the queue (default: print the commands)")
            p.add_argument("--backend", default="torch", choices=("numpy", "torch"), help="nn.evaluate backend")
            p.add_argument("--sims", default="50,200", help="nn.evaluate MCTS budgets")
            p.add_argument("--full-eval", action="store_true", help="also the ladder + hypothesis sections (slow)")
            p.add_argument("--train-args", default="", help="extra nn.train_score arguments, e.g. '--batch 2048 --speed max'")

    for nm in ("pca-sweep", "pca-summary"):
        p = sub.add_parser(nm)
        common(p, gpu=True)
        p.add_argument("--ks", default="0,8,16,24,32,48,64,84", help="0 = no projection (control)")
        p.add_argument("--seeds", default="0")
    p = sub.add_parser("pca-variance")
    common(p)
    p.add_argument("--max-rows", type=int, default=300_000)

    p = sub.add_parser("lda-analyze")
    common(p)
    p.add_argument("--max-rows", type=int, default=100_000)
    p.add_argument("--run", default="", help="also LDA on this net's trunk features (run name or .npz)")
    p.add_argument("--device", default="auto")

    p = sub.add_parser("lda-focus")
    common(p)
    p.add_argument("--label", default="outcome", choices=("outcome", "bestcol"))
    p.add_argument("--features", default="cells+hand", help="cells, hand or cells+hand")
    p.add_argument("--select", default="uncertain", help="uncertain | misclassified | class:<win|draw|loss|colN>")
    p.add_argument("--frac", type=float, default=0.2, help="share of the training positions to keep")
    p.add_argument("--max-rows", type=int, default=0, help="fit LDA on at most this many positions (0 = all)")
    p.add_argument("--out", default=os.path.join(NN_DATA, "lda_focus.npz"))

    for nm in ("lda-sweep", "lda-summary"):
        p = sub.add_parser(nm)
        common(p, gpu=True)
        p.add_argument("--focus", default=os.path.join(NN_DATA, "lda_focus.npz"))
        p.add_argument("--hard-weight", type=float, default=2.0)

    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for k in ("data", "focus", "out"):
        if getattr(a, k, None) and not os.path.isabs(getattr(a, k)):
            setattr(a, k, os.path.join(ROOT, getattr(a, k)))
    if a.cmd in ("pca-variance", "lda-analyze", "lda-focus") and not os.path.isfile(a.data):
        raise SystemExit("dataset not found: %s (python -m nn.make_dataset --positions 300000 --out %s)" % (
            a.data, os.path.relpath(a.data, ROOT)))
    fn = {"pca-variance": cmd_pca_variance, "pca-sweep": cmd_pca_sweep, "pca-summary": cmd_pca_summary,
          "lda-analyze": cmd_lda_analyze, "lda-focus": cmd_lda_focus, "lda-sweep": cmd_lda_sweep,
          "lda-summary": cmd_lda_summary}[a.cmd]
    sys.exit(fn(a) or 0)


if __name__ == "__main__":
    main()
