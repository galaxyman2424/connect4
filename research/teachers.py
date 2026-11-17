"""Teachers for the supervised sweep (research/sweep.py).

A *teacher* labels a fixed set of positions (nn_data/solver_300k.npz) with
  pol   (N,7) float32  policy target (rows sum to 1, zero on full columns)
  wdl   (N,3) float32  value target: P(win), P(draw), P(loss) for the side to move
  top   (N,)  int8     the single column the teacher itself would play
  scores(N,7) float32  the teacher's raw per-column number (NaN = full column)
and the labels are cached in nn_data/rs_labels_<teacher>.npz.

    python -m research.teachers --list
    python -m research.teachers --label all            # label + cache every teacher (<= 8 workers)
    python -m research.teachers --label heur_hand --limit 2000 --no-cache

Teacher ids
  pons            Pascal Pons' solver scores already in the dataset.  Target exactly as
                  nn/train_supervised.py: policy = uniform over the optimal columns,
                  value = one-hot win/draw/loss under perfect play.
  heur_hand       the hand-written heuristic ALONE (no search).
  heur_evo_gN     the same with the evolved weight vector of generation N of runs/evo_d2
                  (N = 1..60; generation 0 is heur_hand).
  mm_d4_hand      minimax depth 4 with the hand heuristic (= difficulty slider level 5).
  mm_d4_evoN      minimax depth 4 with the generation-N evolved weights.
  ppo_net, ppo_from_sup, sup6x64, sup8x128 ...   a trained network (policy head, no search).
  net:<path.npz>  any exported network.

Design decisions (the same rule for every teacher of a kind)
* "Heuristic alone" = for every legal column: play it and score the resulting position
  with the heuristic from the mover's point of view.  The heuristic never looks at
  four-in-a-row (engine/heuristic.py says terminal positions are the search's job), so the
  one game rule we add is: a move that completes four scores WIN-1 and is therefore taken.
  Nothing else is added - in particular NO "must block" rule; if the heuristic does not see
  the opponent's threat, the teacher walks into it.  This is exactly what
  engine.minimax.search(board, depth=1) computes (tests check the equality).
* Policy target of heuristic / minimax teachers = uniform over the columns that share the
  teacher's best score (the same shape as the Pons target).  `top` = the most central of
  them, which is the column the agent plays.
* Value target of heuristic / minimax teachers: s = the teacher's value of its best move
  (side to move), v = +-1 for a proven win/loss (|s| >= MATE_THRESHOLD) else tanh(s / sigma)
  with sigma = the standard deviation of that teacher's non-mate values over the labelled
  set (so the target does not depend on the scale of the weight vector), and
  (W, D, L) = (max(v,0), 1-|v|, max(-v,0)) - the mapping rl/distill.py uses for its
  n-tuple teacher.  A monotone squashing, NOT a calibrated probability.
* Network teachers: pol = the network's softmax over legal columns (soft), wdl = its WDL
  output, top = argmax.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from engine.board import (BOARD_MASK, BOTTOM_COL, CENTER_ORDER, COLS, COLUMN_MASK,
                          TOP_MASK, Board, alignment)
from engine.heuristic import evaluate_bits
from engine.minimax import MATE_THRESHOLD, WIN, search
from nn.common import NN_DATA, ROOT, RUNS, log

DATASET = os.path.join(NN_DATA, "solver_300k.npz")
ILLEGAL = -128
MAX_WORKERS = 8
EVO_RUN = "evo_d2"
NET_TEACHERS = {
    "ppo_net": "runs/ppo_6x64/model.npz",
    "ppo_from_sup": "runs/ppo_from_sup/model.npz",
    "sup6x64": "runs/sup_6x64/best.npz",
    "sup8x128": "runs/rs_A_pons_8x128/best.npz",        # exists once the sweep has trained it
}
DEFAULT_TEACHERS = ["pons", "heur_hand", "heur_evo_g10", "heur_evo_g20", "heur_evo_g40",
                    "heur_evo_g60", "mm_d4_hand", "mm_d4_evo60", "ppo_net", "ppo_from_sup",
                    "sup6x64"]


# ------------------------------------------------------------------ specs
def evo_weights(gen, run=EVO_RUN):
    """Weight vector (10 numbers) of generation ``gen`` of runs/<run>; gen 0 = hand weights.
    Read from metrics.jsonl (3 decimals, the precision the run logged)."""
    from rl.features import HAND_WEIGHTS, NAMES
    if gen == 0:
        return [float(x) for x in HAND_WEIGHTS]
    with open(os.path.join(RUNS, run, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("gen") == gen and "weights" in r:
                return [float(r["weights"][k]) for k in NAMES]
    raise KeyError("generation %d not found in runs/%s/metrics.jsonl" % (gen, run))


def parse(tid):
    """Teacher id -> spec dict {id, kind, label, search_depth, weights|path, gen}."""
    if tid == "pons":
        return {"id": tid, "kind": "solver", "label": "Pons solver (perfect play)",
                "search_depth": None, "gen": None}
    if tid == "heur_hand":
        return {"id": tid, "kind": "heuristic", "label": "hand heuristic, no search",
                "search_depth": 0, "weights": None, "gen": 0}
    if tid.startswith("heur_evo_g"):
        g = int(tid[len("heur_evo_g"):])
        return {"id": tid, "kind": "heuristic", "label": "evolved heuristic gen %d, no search" % g,
                "search_depth": 0, "weights": evo_weights(g), "gen": g}
    if tid.startswith("mm_d"):
        d, w = tid[4:].split("_", 1)
        d = int(d)
        if w == "hand":
            return {"id": tid, "kind": "minimax", "search_depth": d, "weights": None, "gen": 0,
                    "label": "minimax depth %d, hand heuristic%s" % (d, " (slider 5)" if d == 4 else "")}
        g = int(w[3:])
        return {"id": tid, "kind": "minimax", "search_depth": d, "weights": evo_weights(g), "gen": g,
                "label": "minimax depth %d, evolved heuristic gen %d" % (d, g)}
    path = NET_TEACHERS.get(tid) or (tid[4:] if tid.startswith("net:") else None)
    if path:
        return {"id": tid.replace("net:", "net_").replace("/", "_").replace(".npz", ""),
                "kind": "net", "label": "network %s (policy head, no search)" % path,
                "search_depth": 0, "path": path, "gen": None}
    raise ValueError("unknown teacher %r" % tid)


def label_path(tid):
    return os.path.join(NN_DATA, "rs_labels_%s.npz" % parse(tid)["id"])


def _evaluator(weights):
    if weights is None:
        return evaluate_bits
    from rl.features import WeightedEvaluator
    return WeightedEvaluator(weights)


# ------------------------------------------------- per-position move scores
def board_from_bits(cur, mask):
    """engine Board for (stones of the side to move, all stones)."""
    cur, mask = int(cur), int(mask)
    opp = cur ^ mask
    b = Board.__new__(Board)
    first = cur.bit_count() == opp.bit_count()
    b.p1, b.p2 = (cur, opp) if first else (opp, cur)
    b.mask = mask
    b.to_move = 1 if first else -1
    b.moves_played = mask.bit_count()
    b.history = []
    return b


def heuristic_move_scores(cur, mask, evaluator=evaluate_bits):
    """Heuristic alone: list of 7 scores (None = full column), mover's point of view."""
    cur, mask = int(cur), int(mask)
    opp = cur ^ mask
    out = [None] * COLS
    for c in range(COLS):
        if mask & TOP_MASK[c]:
            continue
        mv = (mask + BOTTOM_COL[c]) & COLUMN_MASK[c]
        if alignment(cur | mv):
            out[c] = float(WIN - 1)
        elif (mask | mv) == BOARD_MASK:
            out[c] = 0.0
        else:
            out[c] = -float(evaluator(opp, cur | mv))
    return out


def minimax_move_scores(cur, mask, depth, evaluator=None):
    """Depth-``depth`` negamax value of every legal column (None = full column), on the
    scale of engine.minimax (win at root ply p = WIN - p).  Column c is searched to
    depth-1 from the position after c with the engine's own search()."""
    cur, mask = int(cur), int(mask)
    opp = cur ^ mask
    out = [None] * COLS
    for c in range(COLS):
        if mask & TOP_MASK[c]:
            continue
        mv = (mask + BOTTOM_COL[c]) & COLUMN_MASK[c]
        if alignment(cur | mv):
            out[c] = float(WIN - 1)
        elif (mask | mv) == BOARD_MASK:
            out[c] = 0.0
        elif depth <= 1:
            out[c] = -float((evaluator or evaluate_bits)(opp, cur | mv))
        else:
            s = search(board_from_bits(opp, mask | mv), depth - 1, evaluator=evaluator).score
            if abs(s) >= MATE_THRESHOLD:          # child's "ply q" is the root's ply q+1
                s = math.copysign(abs(s) - 1, s)
            out[c] = -float(s)
    return out


def move_scores(spec, cur, mask):
    ev = _evaluator(spec.get("weights"))
    if spec["kind"] == "heuristic":
        return heuristic_move_scores(cur, mask, ev)
    if spec["kind"] == "minimax":
        return minimax_move_scores(cur, mask, spec["search_depth"],
                                   None if spec.get("weights") is None else ev)
    raise ValueError("move_scores is for heuristic/minimax teachers")


def best_col(scores):
    """Most central column with the maximal score."""
    best = max(s for s in scores if s is not None)
    return next(c for c in CENTER_ORDER if scores[c] is not None and scores[c] == best)


class ScoreAgent:
    """Plays a heuristic/minimax teacher exactly as it labels (engine.agents interface)."""

    def __init__(self, spec):
        self.spec = spec
        self.name = spec["id"]
        self.last_result = None

    def choose(self, board):
        return best_col(move_scores(self.spec, board.current_bits(), board.mask))


def _chunk_scores(args):
    spec, cur, mask = args
    out = np.full((len(cur), COLS), np.nan, np.float64)
    for i in range(len(cur)):
        for c, s in enumerate(move_scores(spec, cur[i], mask[i])):
            if s is not None:
                out[i, c] = s
    return out


def raw_scores(spec, cur, mask, workers=MAX_WORKERS, chunk=1000):
    """(N,7) float64 move scores, NaN on full columns.  workers<=1 -> in this process."""
    workers = max(1, min(int(workers), MAX_WORKERS))
    tasks = [(spec, cur[i:i + chunk], mask[i:i + chunk]) for i in range(0, len(cur), chunk)]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"           # inherited by the spawned workers (no oversubscription)
    if workers == 1 or len(tasks) == 1:
        parts = [_chunk_scores(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_chunk_scores, tasks, chunksize=1)
    return np.concatenate(parts) if parts else np.zeros((0, COLS))


# ------------------------------------------------------------ score -> targets
_RANK = np.array([CENTER_ORDER.index(c) for c in range(COLS)])


def targets_from_scores(scores, sigma=None):
    """scores (N,7) with NaN = illegal -> pol, wdl, top, sigma (see module docstring)."""
    legal = ~np.isnan(scores)
    best = np.where(legal, scores, -np.inf).max(1)
    tie = legal & (scores == best[:, None])
    pol = (tie / tie.sum(1, keepdims=True)).astype(np.float32)
    top = np.where(tie, _RANK[None, :], 99).argmin(1).astype(np.int8)
    mate = np.abs(best) >= MATE_THRESHOLD
    if sigma is None:
        sigma = float(best[~mate].std()) if (~mate).sum() > 1 else 1.0
        sigma = sigma if sigma > 1e-9 else 1.0
    v = np.where(mate, np.sign(best), np.tanh(np.where(mate, 0.0, best) / sigma))
    wdl = np.stack([np.maximum(v, 0), 1 - np.abs(v), np.maximum(-v, 0)], 1).astype(np.float32)
    return pol, wdl, top, sigma


def pons_targets(scores):
    """Exactly nn/train_supervised.load_data: uniform over optimal columns + one-hot WDL."""
    sc = scores.astype(np.int16)
    legal = sc != ILLEGAL
    best = np.where(legal, sc, -1000).max(1)
    opt = (sc == best[:, None]) & legal
    pol = (opt / opt.sum(1, keepdims=True)).astype(np.float32)
    cls = np.where(best > 0, 0, np.where(best == 0, 1, 2))
    wdl = np.eye(3, dtype=np.float32)[cls]
    top = np.where(opt, _RANK[None, :], 99).argmin(1).astype(np.int8)
    return pol, wdl, top, opt


def net_targets(path, cur, mask, device="auto", batch=4096):
    """Network teacher: soft policy over legal columns + its WDL.  GPU if available."""
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    where = "numpy/cpu"
    try:
        if device == "cpu":
            raise RuntimeError("cpu requested")
        from rl.torchutil import load_gpu_evaluator
        ev = load_gpu_evaluator(full, device)
        where = str(ev.device)
    except Exception:
        from nn.numpy_net import NumpyNet
        ev = NumpyNet(full)
    n = len(cur)
    pol = np.zeros((n, COLS), np.float32)
    wdl = np.zeros((n, 3), np.float32)
    for i in range(0, n, batch):
        p, _, w = ev.evaluate(cur[i:i + batch], mask[i:i + batch])
        pol[i:i + batch], wdl[i:i + batch] = p, w
    return pol, wdl, pol.argmax(1).astype(np.int8), where


# ------------------------------------------------------------------ labelling
def label_positions(tid, cur, mask, pons_scores=None, workers=MAX_WORKERS, sigma=None, device="auto"):
    """Label positions with teacher ``tid``.  Returns dict(pol, wdl, top, scores, meta)."""
    spec = parse(tid)
    cur = np.asarray(cur, dtype=np.uint64)
    mask = np.asarray(mask, dtype=np.uint64)
    t0 = time.time()
    meta = {"teacher": spec["id"], "kind": spec["kind"], "label": spec["label"],
            "search_depth": spec["search_depth"], "gen": spec.get("gen"),
            "weights": spec.get("weights"), "path": spec.get("path"), "n": int(len(cur))}
    if spec["kind"] == "solver":
        if pons_scores is None:
            raise ValueError("the pons teacher needs the dataset's solver scores")
        pol, wdl, top, _ = pons_targets(pons_scores)
        sc = np.where(pons_scores == ILLEGAL, np.nan, pons_scores.astype(np.float32))
        meta["workers"] = 0
    elif spec["kind"] == "net":
        pol, wdl, top, where = net_targets(spec["path"], cur, mask, device)
        sc = pol.copy()
        meta["workers"] = 0
        meta["device"] = where
    else:
        w = max(1, min(int(workers), MAX_WORKERS))
        raw = raw_scores(spec, [int(x) for x in cur], [int(x) for x in mask], w)
        pol, wdl, top, sigma = targets_from_scores(raw, sigma)
        sc = raw.astype(np.float32)
        meta["workers"] = w
        meta["value_sigma"] = round(sigma, 4)
        meta["mate_share"] = round(float((np.abs(np.nanmax(raw, 1)) >= MATE_THRESHOLD).mean()), 4)
    sec = time.time() - t0
    meta["label_sec"] = round(sec, 2)
    meta["label_ms_per_pos"] = round(1000 * sec / max(len(cur), 1), 4)
    meta["ties_mean"] = round(float((pol > 0).sum(1).mean()), 3) if spec["kind"] != "net" else None
    return {"pol": pol, "wdl": wdl, "top": top, "scores": sc.astype(np.float32), "meta": meta}


def load_dataset(path=DATASET, limit=None):
    z = np.load(path)
    cur, mask, scores, ply = z["cur"], z["mask"], z["scores"], z["ply"]
    if limit:
        cur, mask, scores, ply = cur[:limit], mask[:limit], scores[:limit], ply[:limit]
    return cur, mask, scores, ply


def bench_quality(tid, sigma=None, device="auto"):
    """The teacher's own quality on the 3,000 held-out benchmark positions (single process,
    so the time is the single-core labelling cost)."""
    from nn import bench
    rows = bench.load_accuracy_bench()
    spec = parse(tid)
    if spec["kind"] == "solver":
        return {"n": len(rows), "optimal": 1.0, "keeps_nl": 1.0, "single_ms_per_pos": None,
                "note": "the benchmark is scored by this solver, so 1.0 by definition"}
    t0 = time.perf_counter()
    lab = label_positions(tid, [r["cur"] for r in rows], [r["mask"] for r in rows],
                          workers=1, sigma=sigma, device=device)
    ms = 1000 * (time.perf_counter() - t0) / len(rows)
    res = bench.accuracy_summary(rows, [int(c) for c in lab["top"]])
    res["single_ms_per_pos"] = round(ms, 4)
    return res


def ensure_labels(tid, data=DATASET, workers=MAX_WORKERS, force=False, device="auto"):
    """Label the whole dataset with ``tid`` (cached).  Returns (path, meta)."""
    path = label_path(tid)
    if os.path.isfile(path) and not force:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"]))
        spec = parse(tid)
        src = spec.get("path")
        stale = src and os.path.getmtime(os.path.join(ROOT, src)) > os.path.getmtime(path)
        if not stale:
            return path, meta
    cur, mask, scores, _ = load_dataset(data)
    log("labelling %d positions with %s ..." % (len(cur), tid))
    lab = label_positions(tid, cur, mask, pons_scores=scores, workers=workers, device=device)
    meta = lab["meta"]
    _, _, _, opt = pons_targets(scores)
    meta["agree_pons"] = round(float(opt[np.arange(len(cur)), lab["top"]].mean()), 4)
    q = bench_quality(tid, sigma=meta.get("value_sigma"), device=device)
    meta["bench_optimal"], meta["bench_keeps_nl"] = q["optimal"], q["keeps_nl"]
    meta["bench_by_bucket"] = q.get("by_bucket")
    meta["single_ms_per_pos"] = q.get("single_ms_per_pos")
    meta["dataset"] = os.path.relpath(data, ROOT)
    os.makedirs(NN_DATA, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, pol=lab["pol"], wdl=lab["wdl"], top=lab["top"], scores=lab["scores"],
             meta_json=np.array(json.dumps(meta)))
    os.replace(tmp, path)
    log("  %s: %.1fs (%.3f ms/pos wall, %d workers)  agree_pons %.4f  bench optimal %.4f keeps %.4f" % (
        tid, meta["label_sec"], meta["label_ms_per_pos"], meta["workers"], meta["agree_pons"],
        meta["bench_optimal"], meta["bench_keeps_nl"]))
    return path, meta


def load_labels(tid):
    with np.load(label_path(tid), allow_pickle=False) as z:
        return {"pol": z["pol"], "wdl": z["wdl"], "top": z["top"],
                "meta": json.loads(str(z["meta_json"]))}


def teacher_agent(tid):
    """The teacher as a player (for the gauntlet).  None for the solver."""
    spec = parse(tid)
    if spec["kind"] == "solver":
        return None
    if spec["kind"] == "net":
        from nn.agents import PolicyAgent
        from nn.numpy_net import NumpyNet
        return PolicyAgent(NumpyNet(os.path.join(ROOT, spec["path"])), name=spec["id"])
    return ScoreAgent(spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--label", nargs="*", default=None, help="teacher ids, or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="only the first N positions (implies --no-cache)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    a = ap.parse_args()
    if a.list or a.label is None:
        for t in DEFAULT_TEACHERS + ["sup8x128"]:
            s = parse(t)
            print("%-14s %-9s %-55s cached=%s" % (t, s["kind"], s["label"], os.path.isfile(label_path(t))))
        return
    ids = DEFAULT_TEACHERS if a.label in ([], ["all"]) else a.label
    for t in ids:
        if a.limit or a.no_cache:
            cur, mask, scores, _ = load_dataset(limit=a.limit)
            lab = label_positions(t, cur, mask, pons_scores=scores, workers=a.workers)
            _, _, _, opt = pons_targets(scores)
            agree = float(opt[np.arange(len(cur)), lab["top"]].mean())
            log("%s: %d positions %.2fs (%.3f ms/pos wall)  agree_pons %.4f  ties %s  W/D/L mean %s" % (
                t, len(cur), lab["meta"]["label_sec"], lab["meta"]["label_ms_per_pos"], agree,
                lab["meta"]["ties_mean"], np.round(lab["wdl"].mean(0), 3).tolist()))
        else:
            ensure_labels(t, workers=a.workers, force=a.force)


if __name__ == "__main__":
    main()
