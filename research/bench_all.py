"""Policy + value (+ score) head accuracy of EVERY trained net on one common benchmark.

    python -m research.bench_all

Benchmark: data/accuracy_positions.csv (3,000 held-out positions, plies 8-30, every column
solved by Pons).  For each run in runs/ that has best.npz or model.npz (the file the arena
plays is used: best.npz if present, else model.npz):

  policy_optimal   argmax of the raw policy head (legal columns, no mirror) is solver-optimal
  policy_keeps     ... keeps the game-theoretic result (win stays win, draw stays draw),
                   over the positions that are not already lost (same as bench_keeps_nl)
  played_optimal   the move the net actually plays with its own settings (play_head, mirror)
  value_acc        argmax of the W/D/L head == solver outcome of the position
  value_acc_W/D/L  the same, split by true outcome
  value_brier      multi-class Brier score of the W/D/L probabilities (lower = better)
  value_corr       Pearson r of P(win)-P(loss) with the true outcome (+1/0/-1)
  score_mae        score-head nets only: mean |predicted - Pons score| over legal columns
  score_sign_acc   score-head nets only: sign of the predicted column score is right

plus the run's own training-time validation numbers (last line of metrics.jsonl), which use
each run's own validation split and are NOT comparable between runs.

Writes experiments/results/research/bench_all.json and bench_all.csv.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import csv  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from engine.board import Board  # noqa: E402
from nn.encode import encode, legal_mask  # noqa: E402
from nn.numpy_net import NumpyNet, _softmax  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
BENCH = os.path.join(ROOT, "data", "accuracy_positions.csv")
OUT = os.path.join(ROOT, "experiments", "results", "research", "bench_all")

TEACHER_LABEL = {
    "pons": "Pons solver (perfect labels)",
    "heur_hand": "hand heuristic, no search",
    "heur_evo_g10": "evolved heuristic gen 10, no search",
    "heur_evo_g20": "evolved heuristic gen 20, no search",
    "heur_evo_g40": "evolved heuristic gen 40, no search",
    "heur_evo_g60": "evolved heuristic gen 60, no search",
    "mm_d4_hand": "minimax depth 4 + hand heuristic",
    "mm_d4_evo60": "minimax depth 4 + evolved heuristic",
    "ppo_net": "PPO self-play net",
    "ppo_from_sup": "PPO net initialised from supervised",
    "sup6x64": "distilled from sup 6x64 net",
    "sup8x128": "distilled from sup 8x128 net",
}
# label family = what produced the training targets
FAMILY_ORDER = ["perfect labels (Pons)", "perfect labels + score head (Pons, 3M)", "net teacher (distillation)",
                "search teacher (minimax d4)", "heuristic teacher (no search)", "RL-net teacher",
                "self-play (AlphaZero)", "self-play RL (PPO / DQN)"]


def describe(run, cfg):
    a = cfg.get("args", {})
    m = re.match(r"rs_([ABC])_(.+)_(\d+x\d+)(_s\d)?$", run)
    if m:
        t = m.group(2)
        fam = ("perfect labels (Pons)" if t == "pons" else
               "net teacher (distillation)" if t.startswith("sup") else
               "search teacher (minimax d4)" if t.startswith("mm_") else
               "heuristic teacher (no search)" if t.startswith("heur") else "RL-net teacher")
        return fam, t, TEACHER_LABEL.get(t, t), "sweep " + m.group(1)
    if run.startswith("score_8x128"):
        return ("perfect labels + score head (Pons, 3M)", "pons_3m",
                "Pons solver, 3M positions, score head" + (" + hard positions" if run.endswith("_r1") else ""), "score")
    if run in ("sup_6x64", "pilot_cpu_sup"):
        return "perfect labels (Pons)", "pons", TEACHER_LABEL["pons"], "main"
    if run in ("az_6x64", "pilot_cpu_az"):
        return "self-play (AlphaZero)", "selfplay_az", "AlphaZero self-play + MCTS targets", "main"
    if run.startswith("distill_az"):
        return "net teacher (distillation)", "az_mcts", "distilled from az_6x64 + MCTS100", "main"
    if run.startswith("distill_ens"):
        return "net teacher (distillation)", "ensemble", "distilled from a teacher ensemble", "main"
    if run.startswith("ppo_from_sup"):
        return "self-play RL (PPO / DQN)", "ppo_from_sup", "PPO self-play, initialised from sup_6x64", "main"
    if run.startswith("ppo"):
        return "self-play RL (PPO / DQN)", "ppo", "PPO self-play from scratch", "main"
    if run.startswith("dqn"):
        return "self-play RL (PPO / DQN)", "dqn", "DQN self-play", "main"
    return "other", a.get("teacher", "?"), a.get("teacher", "?"), "?"


def load_bench():
    rows = list(csv.DictReader(open(BENCH, newline="")))
    curs, masks, sc = [], [], []
    for r in rows:
        b = Board.from_moves(r["moves"]) if hasattr(Board, "from_moves") else None
        if b is None:
            b = Board()
            for ch in r["moves"]:
                b.drop(int(ch) - 1)
        curs.append(b.current_bits())
        masks.append(b.mask)
        sc.append([np.nan if v is None else v for v in json.loads(r["scores"])])
    sc = np.array(sc, dtype=float)
    return (np.array(curs, dtype=np.uint64), np.array(masks, dtype=np.uint64), sc,
            np.array([int(r["ply"]) for r in rows]))


def last_metrics(run):
    p = os.path.join(RUNS, run, "metrics.jsonl")
    last = None
    if os.path.isfile(p):
        for line in open(p):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") in ("epoch", "generation"):
                last = d
    if not last:
        return {}
    keep = ("val_policy_acc", "val_value_acc", "val_teacher_agree", "val_teacher_agreement", "val_policy_loss",
            "val_value_loss", "bench_optimal", "bench_keeps_nl", "bench_value_acc", "epoch", "gen")
    out = {k: last[k] for k in keep if k in last}
    if "val_optimal_by_head" in last:
        out["val_policy_acc"] = last["val_optimal_by_head"].get("policy")
        out["val_score_head_acc"] = last["val_optimal_by_head"].get("score")
    return out


def main():
    t0 = time.time()
    cur, mask, sc, ply = load_bench()
    legal = legal_mask(mask)
    best = np.nanmax(sc, 1)
    opt = (sc == best[:, None])
    keeps = np.sign(np.nan_to_num(sc, nan=-99)) == np.sign(best)[:, None]
    outcome = np.where(best > 0, 0, np.where(best == 0, 1, 2))        # W D L index
    truth_v = np.sign(best)
    nl = best >= 0                                                   # not-lost: a move can throw the result
    X = encode(cur, mask)
    res = []
    for run in sorted(os.listdir(RUNS)):
        d = os.path.join(RUNS, run)
        if run.startswith("_") or not os.path.isdir(d):
            continue
        f = "best.npz" if os.path.isfile(os.path.join(d, "best.npz")) else "model.npz"
        if not os.path.isfile(os.path.join(d, f)):
            continue
        cfg = json.load(open(os.path.join(d, "config.json"))) if os.path.isfile(os.path.join(d, "config.json")) else {}
        net = NumpyNet(os.path.join(d, f))
        pl, wl, s = [], [], []
        for i in range(0, len(X), 1000):
            p_, w_, s_ = net.forward_all(X[i:i + 1000])
            pl.append(p_)
            wl.append(w_)
            s.append(s_)
        p = np.concatenate(pl)
        w = _softmax(np.concatenate(wl))
        p[~legal] = -np.inf
        mv = p.argmax(1)
        played, _, _ = net.evaluate(cur, mask)
        pmv = played.argmax(1)
        ix = np.arange(len(mv))
        v = w[:, 0] - w[:, 2]
        onehot = np.eye(3)[outcome]
        fam, tkey, tlab, sweep = describe(run, cfg)
        r = {"run": run, "file": f, "family": fam, "teacher": tkey, "teacher_label": tlab, "sweep": sweep,
             "blocks": net.blocks, "channels": int(net.w["stem.w"].shape[0]),
             "params": int(sum(x.size for x in net.w.values())),
             "seed": cfg.get("args", {}).get("seed"),
             "policy_optimal": float(opt[ix, mv].mean()), "policy_keeps": float(keeps[ix, mv][nl].mean()),
             "played_optimal": float(opt[ix, pmv].mean()), "played_keeps": float(keeps[ix, pmv][nl].mean()),
             "play_head": net.play_head, "play_mirror": bool(net.meta.get("play_mirror")),
             "value_acc": float((w.argmax(1) == outcome).mean()),
             "value_brier": float(((w - onehot) ** 2).sum(1).mean()),
             "value_corr": float(np.corrcoef(v, truth_v)[0, 1])}
        for k, name in enumerate("WDL"):
            sel = outcome == k
            r["value_acc_" + name] = float((w[sel].argmax(1) == k).mean())
        if s[0] is not None:
            S = np.concatenate(s)
            ok = legal & ~np.isnan(sc)
            r["score_mae"] = float(np.abs(S - np.nan_to_num(sc))[ok].mean())
            r["score_sign_acc"] = float((np.sign(np.round(S)) == np.sign(np.nan_to_num(sc)))[ok].mean())
            r["score_head_optimal"] = float(opt[ix, np.where(legal, S, -np.inf).argmax(1)].mean())
        # by phase
        r["policy_optimal_by_ply"] = {}
        for lo, hi in ((8, 15), (16, 23), (24, 30)):
            sel = (ply >= lo) & (ply <= hi)
            r["policy_optimal_by_ply"]["%d-%d" % (lo, hi)] = float(opt[ix[sel], mv[sel]].mean())
        r["training_val"] = last_metrics(run)
        res.append(r)
        print("%-28s %-40s opt %.3f keeps %.3f played %.3f value %.3f" % (
            run, fam[:40], r["policy_optimal"], r["policy_keeps"], r["played_optimal"], r["value_acc"]), flush=True)
    res.sort(key=lambda r: (FAMILY_ORDER.index(r["family"]) if r["family"] in FAMILY_ORDER else 99,
                            r["teacher"], -r["policy_optimal"]))
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "benchmark": "data/accuracy_positions.csv",
            "n": int(len(X)), "outcome_counts": {k: int((outcome == i).sum()) for i, k in enumerate("WDL")},
            "family_order": FAMILY_ORDER, "seconds": round(time.time() - t0, 1)}
    json.dump({"meta": meta, "nets": res}, open(OUT + ".json", "w"), indent=1)
    cols = ["run", "family", "teacher_label", "blocks", "channels", "params", "seed", "policy_optimal",
            "policy_keeps", "played_optimal", "value_acc", "value_acc_W", "value_acc_D", "value_acc_L",
            "value_brier", "value_corr", "score_mae", "score_sign_acc"]
    with open(OUT + ".csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols + ["train_val_policy_acc", "train_val_value_acc"])
        for r in res:
            wr.writerow([r.get(c, "") for c in cols] + [r["training_val"].get("val_policy_acc", ""),
                                                         r["training_val"].get("val_value_acc", "")])
    print("wrote %s.json/.csv (%d nets, %.0fs)" % (OUT, len(res), time.time() - t0))


if __name__ == "__main__":
    main()
