"""Embedding-neighbor experiment (statement section 7, project dossier).

    python experiments/embedding.py                       # corpus = self-play games from results.csv
    python experiments/embedding.py --corpus uci          # UCI 8-ply positions (after data/download_data.py --uci)
    python experiments/embedding.py --corpus both

Query: a fixed mid-game position after 8 plies (``--query``, Pons move string,
default 44334552), then ONE controlled change: add one move (default: the
engine's depth-10 choice; ``--change`` to pick a column).

Corpus: every distinct non-terminal position (ply >= 4) occurring in the
tournament games (experiments/results/results.csv) and/or the UCI positions.

Embeddings (both compared with cosine similarity, sklearn NearestNeighbors):
  features  hand-crafted, standardized over the corpus (z-scores):
            phase (ply/42), side to move, per player: stones in the center
            column, center-weighted stone sum, open twos, open threes (4-cell
            windows with no opponent stone), playable immediate threats, all
            threat cells, threat cells on odd / even rows (Allis' zugzwang
            parity), plus the 7 column heights.
  raw       the flattened 42-cell grid (+1 / -1 / 0), the dossier's baseline.

"Similarity != same best move" is measured twice:
  1. demo: for the query (before and after the change) the best column of each
     of its top-5 neighbors (solver if available, else depth-10 engine) is
     compared with the query's best column;
  2. systematically on the 3,000 solver-labeled positions of
     data/accuracy_positions.csv (leave-one-out top-5 neighbors within that set):
     share of neighbors sharing an optimal column / the same outcome
     (win/draw/loss for the side to move) with the query, against random
     pairs from the same ply bucket.

Outputs: figures/embedding_neighbors.png, figures/embedding_neighbors_raw.png,
results/embedding_neighbors.json, results/embedding_label_agreement.csv
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, FIGURES, RESULTS, Board, board_from_grid_any, ensure_dirs,  # noqa: E402
                    load_accuracy_positions, log, outcome, read_csv, solver_available, write_csv,
                    write_json)
from engine.board import BOARD_MASK, BOTTOM_MASK, COLUMN_MASK, winning_cells_mask  # noqa: E402
from engine.heuristic import _counts  # noqa: E402
from engine.minimax import search  # noqa: E402

COLW = (0, 1, 2, 3, 2, 1, 0)
ODD_ROWS = sum(1 << (c * 7 + h) for c in range(7) for h in (0, 2, 4))  # rows 1,3,5 (1-indexed)
EVEN_ROWS = sum(1 << (c * 7 + h) for c in range(7) for h in (1, 3, 5))
FEATURE_NAMES = (["phase", "to_move"] +
                 ["%s_%s" % (p, f) for p in ("p1", "p2") for f in
                  ("center", "center_w", "open2", "open3", "threats_now", "threat_cells",
                   "threats_odd", "threats_even")] +
                 ["height_%d" % c for c in range(7)])


def features(b):
    """Hand-crafted feature vector (absolute colours: p1 = first player)."""
    mask = b.mask
    possible = (mask + BOTTOM_MASK) & BOARD_MASK
    f = [b.moves_played / 42.0, float(b.to_move)]
    for me, opp in ((b.p1, b.p2), (b.p2, b.p1)):
        threes, twos = _counts(me, BOARD_MASK & ~opp)
        cells = winning_cells_mask(me, mask)
        cw = sum(COLW[c] * (me & COLUMN_MASK[c]).bit_count() for c in range(7))
        f += [(me & COLUMN_MASK[3]).bit_count(), cw, twos, threes,
              (cells & possible).bit_count(), cells.bit_count(),
              (cells & ODD_ROWS).bit_count(), (cells & EVEN_ROWS).bit_count()]
    f += [b.height(c) for c in range(7)]
    return f


def raw(b):
    return [v for row in b.to_grid() for v in row]


# ------------------------------------------------------------------ corpus
def selfplay_corpus(path, min_ply=4):
    if not os.path.exists(path):
        sys.exit("%s not found - run experiments/tournament.py first" % path)
    seen = {}
    for r in read_csv(path):
        mv = r["moves"]
        b = Board()
        for i, ch in enumerate(mv):
            b.drop(int(ch) - 1)
            if b.check_win() or b.is_draw():
                break
            if b.moves_played >= min_ply and b.key() not in seen:
                seen[b.key()] = mv[:i + 1]
    return [("selfplay", m) for m in seen.values()]


def uci_corpus():
    """Real UCI file (data/download_data.py --uci) if present, else the locally
    regenerated, equivalent 67,557-position set data/uci/positions.csv."""
    path = os.path.join(DATA, "uci", "uci_positions.csv")
    regen = os.path.join(DATA, "uci", "positions.csv")
    out = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                b, _ = board_from_grid_any(json.loads(r["grid"]))
                if b is not None:
                    out.append(("uci:" + r["label"], b))
        return out
    if os.path.exists(regen):
        log("using regenerated UCI positions %s (run data/download_data.py --uci for the real file)" % regen)
        with open(regen, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.append(("uci-regen", Board.from_moves(r["moves"])))
        return out
    sys.exit("%s not found - run `python data/download_data.py --uci` first" % path)


# ------------------------------------------------------------------ helpers
class BestMove:
    """Best column(s) of a position: solver if available, else depth-10 engine."""

    def __init__(self, use_solver, timeout):
        self.solver = None
        if use_solver and solver_available():
            from tools.solver import Solver
            self.solver = Solver(timeout=timeout)

    def __call__(self, b):
        if self.solver is not None:
            try:
                sc = self.solver.solve_all(b.to_move_string())
                best = max(x for x in sc if x is not None)
                return sorted(c for c in range(7) if sc[c] == best), "solver", sc
            except Exception:   # timeout -> engine fallback
                pass
        r = search(b, 10)
        return [r.col], "engine-d10", None


def draw_board(ax, b, title, highlight=None, color_title="black", mark_col=None):
    import matplotlib.patches as patches
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(patches.FancyBboxPatch((-0.5, -0.5), 7, 6, boxstyle="round,pad=0.02",
                                        fc="#2b4c7e", ec="none"))
    grid = b.to_grid()
    for r in range(6):
        for c in range(7):
            v = grid[r][c]
            fc = {1: "#e69f00", -1: "#56b4e9", 0: "white"}[v]
            ax.add_patch(patches.Circle((c, 5 - r), 0.4, fc=fc, ec="#1a2f4f", lw=0.8))
    if mark_col is not None:   # the controlled change: the stone just added
        ax.add_patch(patches.Circle((mark_col, b.height(mark_col) - 1), 0.12, fc="black", ec="none"))
    if highlight is not None:
        h = b.height(highlight)
        ax.add_patch(patches.Circle((highlight, h), 0.4, fc="none", ec="#cc0000", lw=2.2, ls="--"))
    ax.set_title(title, fontsize=8.5, color=color_title)


def neighbors(nn, X, q, k, exclude_key, keys):
    dist, idx = nn.kneighbors(q.reshape(1, -1), n_neighbors=k + 5)
    out = []
    for d, i in zip(dist[0], idx[0]):
        if keys[i] == exclude_key:
            continue
        out.append((int(i), 1.0 - float(d)))
        if len(out) == k:
            break
    return out


# -------------------------------------------------------------- systematic
def label_agreement(k, seed):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    rows = load_accuracy_positions()
    boards = [Board.from_moves(r["moves"]) for r in rows]
    outs = [outcome(max(s for s in r["scores"] if s is not None)) for r in rows]
    opt = [set(r["optimal_cols"]) for r in rows]
    embs = {"features": StandardScaler().fit_transform(np.array([features(b) for b in boards], float)),
            "raw": np.array([raw(b) for b in boards], float)}
    res = []
    rng = random.Random(seed)
    for name, X in embs.items():
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
        _, idx = nn.kneighbors(X)
        same_opt = same_out = dply = 0
        n = 0
        for i in range(len(rows)):
            nb = [j for j in idx[i] if j != i][:k]
            for j in nb:
                same_opt += bool(opt[i] & opt[j])
                same_out += outs[i] == outs[j]
                dply += abs(rows[i]["ply"] - rows[j]["ply"])
                n += 1
        res.append({"embedding": name, "pairs": n, "share_optimal_col_overlap": round(same_opt / n, 4),
                    "share_same_outcome": round(same_out / n, 4), "mean_abs_ply_diff": round(dply / n, 2)})
    # baselines: random partner from the same ply bucket, and from anywhere
    by_bucket = {}
    for i, r in enumerate(rows):
        by_bucket.setdefault(r["bucket"], []).append(i)
    for name, pick in (("random_same_bucket", lambda i: rng.choice(by_bucket[rows[i]["bucket"]])),
                       ("random_any", lambda i: rng.randrange(len(rows)))):
        same_opt = same_out = dply = n = 0
        for i in range(len(rows)):
            for _ in range(k):
                j = pick(i)
                while j == i:
                    j = pick(i)
                same_opt += bool(opt[i] & opt[j])
                same_out += outs[i] == outs[j]
                dply += abs(rows[i]["ply"] - rows[j]["ply"])
                n += 1
        res.append({"embedding": name, "pairs": n, "share_optimal_col_overlap": round(same_opt / n, 4),
                    "share_same_outcome": round(same_out / n, 4), "mean_abs_ply_diff": round(dply / n, 2)})
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=["selfplay", "uci", "both"], default="selfplay")
    ap.add_argument("--games", default=os.path.join(RESULTS, "results.csv"))
    ap.add_argument("--query", default="44334552")
    ap.add_argument("--change", type=int, default=None,
                    help="0-indexed column of the added move (default: engine depth-10 choice)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-solver", action="store_true")
    ap.add_argument("--solver-timeout", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    ensure_dirs()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    corpus = []
    if args.corpus in ("selfplay", "both"):
        corpus += [(src, Board.from_moves(m)) for src, m in selfplay_corpus(args.games)]
    if args.corpus in ("uci", "both"):
        corpus += uci_corpus()
    keys = [b.key() for _, b in corpus]
    log("corpus: %d positions (%s), built in %.0f s" % (len(corpus), args.corpus, time.time() - t0))

    q0 = Board.from_moves(args.query)
    change = args.change if args.change is not None else search(q0, 10).col
    q1 = q0.copy()
    q1.drop(change)
    queries = [("before", q0), ("after (+col %d)" % (change + 1), q1)]

    F = np.array([features(b) for _, b in corpus], float)
    scaler = StandardScaler().fit(F)
    embs = {"features": (scaler.transform(F), lambda b: scaler.transform(np.array([features(b)], float))[0]),
            "raw": (np.array([raw(b) for _, b in corpus], float), lambda b: np.array(raw(b), float))}
    best = BestMove(not args.no_solver, args.solver_timeout)
    cache = {}

    def bestcols(b):
        if b.key() not in cache:
            cache[b.key()] = best(b)
        return cache[b.key()]

    report = {"query": args.query, "change_col_1indexed": change + 1,
              "corpus": args.corpus, "corpus_size": len(corpus), "k": args.k, "embeddings": {}}
    for ename, (X, embed) in embs.items():
        nn = NearestNeighbors(metric="cosine").fit(X)
        fig, axes = plt.subplots(2, args.k + 1, figsize=(2.3 * (args.k + 1), 5.6))
        rep = {}
        sets = []
        for row, (qname, qb) in enumerate(queries):
            qcols, qsrc, _ = bestcols(qb)
            nbs = neighbors(nn, X, embed(qb), args.k, qb.key(), keys)
            sets.append({keys[i] for i, _ in nbs})
            draw_board(axes[row][0], qb, "QUERY %s\nply %d, best col %s (%s)" % (
                qname, qb.moves_played, "/".join(str(c + 1) for c in qcols), qsrc),
                highlight=qcols[0], color_title="#cc0000" if row else "black",
                mark_col=change if row else None)
            entries = []
            for j, (i, sim) in enumerate(nbs):
                src, b = corpus[i]
                cols, csrc, _ = bestcols(b)
                same = bool(set(cols) & set(qcols))
                entries.append({"rank": j + 1, "moves": b.to_move_string(), "source": src,
                                "ply": b.moves_played, "cosine": round(sim, 4),
                                "best_cols_1indexed": [c + 1 for c in cols], "label_source": csrc,
                                "same_best_move_as_query": same})
                draw_board(axes[row][j + 1], b, "#%d  cos %.3f  ply %d\nbest col %s - %s" % (
                    j + 1, sim, b.moves_played, "/".join(str(c + 1) for c in cols),
                    "SAME" if same else "different"), highlight=cols[0])
            rep[qname] = {"query_best_cols_1indexed": [c + 1 for c in qcols], "label_source": qsrc,
                          "neighbors": entries,
                          "share_same_best_move": round(sum(e["same_best_move_as_query"]
                                                            for e in entries) / len(entries), 3)}
        rep["top_k_overlap_before_after"] = len(sets[0] & sets[1])
        report["embeddings"][ename] = rep
        fig.suptitle("Embedding neighbors (%s embedding, cosine) - query %s, then one added move "
                     "\n(orange = first player, blue = second player; dashed ring = best column; black dot = the added move)" % (ename, args.query), fontsize=10)
        fig.tight_layout()
        out = os.path.join(FIGURES, "embedding_neighbors%s.png" % ("" if ename == "features" else "_raw"))
        fig.savefig(out, dpi=130)
        plt.close(fig)
        log("%s: wrote %s" % (ename, out))
        for qname in rep:
            if qname.startswith("top"):
                continue
            log("  %-16s share of top-%d with same best move: %.2f" % (
                qname, args.k, rep[qname]["share_same_best_move"]))
        log("  top-%d overlap before/after change: %d" % (args.k, rep["top_k_overlap_before_after"]))

    log("label agreement on the solver-labeled set ...")
    agree = label_agreement(args.k, args.seed)
    write_csv(os.path.join(RESULTS, "embedding_label_agreement.csv"), agree)
    report["label_agreement_accuracy_set"] = agree
    report["feature_names"] = FEATURE_NAMES
    write_json(os.path.join(RESULTS, "embedding_neighbors.json"), report)
    for a in agree:
        log("  %-20s optimal-col overlap %.3f  same outcome %.3f  |dply| %.2f" % (
            a["embedding"], a["share_optimal_col_overlap"], a["share_same_outcome"], a["mean_abs_ply_diff"]))
    if best.solver is not None:
        best.solver.close()
    log("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
