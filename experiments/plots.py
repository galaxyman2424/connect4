"""All analysis charts, drawn from experiments/results/*.  Used by
analysis.ipynb; can also be run headless:

    python experiments/plots.py        # writes every figure it has data for

One style for every chart: Okabe-Ito colour-blind-safe palette, titled,
labelled axes, light grid, PNG at 150 dpi in experiments/figures/.
"""

import json
import os
import sys

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, FIGURES, RESULTS, ROOT, ensure_dirs  # noqa: E402,F401

OKABE = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
         "blue": "#0072B2", "red": "#D55E00", "purple": "#CC79A7", "black": "#000000",
         "grey": "#999999"}
SEQ = [OKABE[k] for k in ("blue", "orange", "green", "red", "purple", "sky", "yellow", "black")]
ODD_C, EVEN_C = OKABE["orange"], OKABE["blue"]


def style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=SEQ)})


def _r(name):
    return os.path.join(RESULTS, name)


def _save(fig, name):
    path = os.path.join(FIGURES, name)
    fig.savefig(path)
    return path


def have(*names):
    return all(os.path.exists(_r(n)) for n in names)


# --------------------------------------------------------------- tournament
def winrate_heatmap():
    m = pd.read_csv(_r("tournament_winrate.csv"), index_col=0)
    players = list(m.index)
    vals = m[players].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(vals, cmap="RdBu", vmin=0, vmax=1)
    for i in range(len(players)):
        for j in range(len(players)):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, "%.2f" % vals[i, j], ha="center", va="center", fontsize=8.5,
                        color="white" if abs(vals[i, j] - 0.5) > 0.3 else "black")
    ax.set_xticks(range(len(players)), [p.replace("d", "depth ") for p in players], rotation=45)
    ax.set_yticks(range(len(players)), [p.replace("d", "depth ") for p in players])
    ax.set_xlabel("opponent (column)")
    ax.set_ylabel("bot (row)")
    ax.grid(False)
    ax.set_title("Round robin: score of row vs column\n(100 games per pairing, draw = 0.5)")
    fig.colorbar(im, ax=ax, label="score (win rate, draws = 0.5)", shrink=0.85)
    return _save(fig, "winrate_heatmap.png"), fig


def elo_plot():
    with open(_r("elo.json")) as f:
        e = json.load(f)["ratings"]
    ds = sorted(e, key=lambda p: int(p[1:]))
    x = [int(p[1:]) for p in ds]
    y = np.array([e[p]["elo"] for p in ds])
    lo = np.array([e[p]["lo"] for p in ds])
    hi = np.array([e[p]["hi"] for p in ds])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, y, "-", color=OKABE["grey"], lw=1, zorder=1)
    for xi, yi, l, h in zip(x, y, lo, hi):
        c = ODD_C if xi % 2 else EVEN_C
        ax.errorbar(xi, yi, yerr=[[yi - l], [h - yi]], fmt="o", color=c, capsize=4, ms=7, zorder=2)
    ax.plot([], [], "o", color=ODD_C, label="odd depth")
    ax.plot([], [], "o", color=EVEN_C, label="even depth")
    ax.set_xlabel("search depth (plies)")
    ax.set_ylabel("Elo (depth 1 = 1000)")
    ax.set_title("Bradley-Terry Elo vs search depth (95% bootstrap CI)")
    ax.set_xticks(x)
    ax.legend(loc="lower right")
    return _save(fig, "elo_vs_depth.png"), fig


def slider_plot():
    s = pd.read_csv(_r("slider_summary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, opp, title in ((axes[0], True, "Each level vs RandomAgent"),
                           (axes[1], False, "Each level vs the level below")):
        sub = s[(s.opponent == "random") == opp].copy()
        sub["lv"] = sub.level.str[1:].astype(int)
        sub = sub.sort_values("lv")
        if opp:
            sc, lo, hi = sub.score, sub.ci_lo, sub.ci_hi
            labels = list(sub.lv)
        else:   # rows are "L_k vs L_k+1" from L_k's side -> show the higher level's score
            sc, lo, hi = 1 - sub.score, 1 - sub.ci_hi, 1 - sub.ci_lo
            labels = ["%d vs %d" % (v + 1, v) for v in sub.lv]
        xs = np.arange(len(sub))
        ax.bar(xs, sc, color=OKABE["blue"] if opp else OKABE["orange"], alpha=0.85)
        ax.errorbar(xs, sc, yerr=[sc - lo, hi - sc], fmt="none", color="black", capsize=3, lw=1)
        for xi, n in zip(xs, sub.games):
            ax.text(xi, 1.06, "n=%d" % n, ha="center", va="bottom", fontsize=7)
        ax.axhline(0.5, color=OKABE["grey"], ls="--", lw=1)
        ax.set_xticks(xs, labels)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("slider level" if opp else "higher level vs lower level")
        ax.set_ylabel("score of the %s level (draw = 0.5)" % ("" if opp else "higher"))
        ax.set_title(title)
    fig.suptitle("Difficulty slider: higher levels should score more (95% Wilson CI)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "slider_levels.png"), fig


def responsiveness_plot():
    r = pd.read_csv(_r("responsiveness.csv"))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(r.level, r.mean_ms, color=OKABE["sky"], label="mean")
    ax.plot(r.level, r.max_ms, "D", color=OKABE["red"], label="max over test positions")
    ax.axhline(2000, color="black", ls="--", lw=1, label="2 s target")
    ax.set_yscale("log")
    ax.set_xticks(r.level)
    ax.set_xlabel("slider level")
    ax.set_ylabel("ms per move (log scale)")
    ax.set_title("Responsiveness: time per bot move by level")
    ax.legend(loc="upper left")
    return _save(fig, "responsiveness.png"), fig


# ----------------------------------------------------------------- accuracy
def accuracy_plot():
    s = pd.read_csv(_r("accuracy_accuracy_summary.csv"))
    a = s[s.bucket == "all"].sort_values("depth")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    ax.errorbar(a.depth, a.optimal, yerr=[a.optimal - a.opt_lo, a.opt_hi - a.optimal], fmt="o-",
                color=OKABE["blue"], capsize=3, label="picks an optimal column")
    ax.errorbar(a.depth, a.keeps_nl, yerr=[a.keeps_nl - a.keeps_nl_lo, a.keeps_nl_hi - a.keeps_nl],
                fmt="s-", color=OKABE["green"], capsize=3,
                label="keeps the result (not-lost positions)")
    if have("accuracy_accuracy_subset_summary.csv"):
        sb = pd.read_csv(_r("accuracy_accuracy_subset_summary.csv"))
        sb = sb[sb.bucket == "all"].sort_values("depth")
        ax.plot(sb.depth, sb.optimal, "o:", color=OKABE["blue"], alpha=0.5, mfc="white",
                label="optimal, %d-position subset (to depth %d)" % (sb.n.iloc[0], sb.depth.max()))
    ax.axhline(a.rand_optimal.iloc[0], color=OKABE["blue"], ls="--", lw=1, alpha=0.6)
    ax.text(1, a.rand_optimal.iloc[0] + 0.01, "random move: optimal", fontsize=8, color=OKABE["blue"])
    ax.axhline(float(a.rand_keeps_nl.iloc[0]), color=OKABE["green"], ls="--", lw=1, alpha=0.6)
    ax.text(1, float(a.rand_keeps_nl.iloc[0]) + 0.01, "random move: keeps result", fontsize=8,
            color=OKABE["green"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("search depth")
    ax.set_ylabel("share of positions")
    ax.set_title("Accuracy vs perfect play (%d solver-labeled positions)" % a.n.iloc[0])
    ax.set_xticks(list(a.depth))
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    b = s[s.bucket != "all"]
    for i, (bk, g) in enumerate(sorted(b.groupby("bucket"), key=lambda t: int(t[0].split("-")[0]))):
        g = g.sort_values("depth")
        ax.plot(g.depth, g.optimal, "o-", color=SEQ[i % len(SEQ)], label="ply %s" % bk, ms=4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("search depth")
    ax.set_ylabel("share picking an optimal column")
    ax.set_title("Accuracy by game phase (ply bucket)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    return _save(fig, "accuracy_vs_depth.png"), fig


def tactics_plot():
    s = pd.read_csv(_r("tactics_summary.csv"))
    s = s[s.agent.str.startswith("d")]
    names = {"win1": "win in 1", "block": "must block", "win2": "win in 2 (3 plies)",
             "win3": "win in 3 (5 plies)", "all": "all"}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i, cat in enumerate(["win1", "block", "win2", "win3", "all"]):
        g = s[s.category == cat].sort_values("depth")
        ax.plot(g.depth + (i - 2) * 0.04, g.share_solved, "o-" if cat != "all" else "--",
                color=SEQ[i] if cat != "all" else "black", label="%s (n=%d)" % (names[cat], g.n.iloc[0]),
                ms=5, lw=1.5 if cat != "all" else 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("search depth")
    ax.set_ylabel("share solved")
    ax.set_title("Tactics suite: share solved vs depth (solver-verified)")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, "tactics_vs_depth.png"), fig


# --------------------------------------------------------------------- cost
def _cost_open():
    s = pd.read_csv(_r("cost_summary.csv"))
    return s[s.subset == "open"] if "subset" in s else s


def cost_plot():
    s = _cost_open()
    names = {"full": "TT + ordering (shipped)", "no_tt": "ordering only (no TT)",
             "no_order": "TT only (no ordering)", "plain": "plain alpha-beta"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for i, (cfg, lab) in enumerate(names.items()):
        g = s[(s.config == cfg) & (s.complete == 1)].sort_values("depth")
        if g.empty:
            continue
        axes[0].plot(g.depth, g.mean_nodes, "o-", color=SEQ[i], label=lab, ms=4)
        axes[1].plot(g.depth, g.mean_ms, "o-", color=SEQ[i], label=lab, ms=4)
    for ax, yl, t in ((axes[0], "positions searched per move (mean)", "Nodes per move vs depth"),
                      (axes[1], "ms per move (mean)", "Time per move vs depth")):
        ax.set_yscale("log")
        ax.set_xlabel("search depth")
        ax.set_ylabel(yl + ", log scale")
        ax.set_title(t)
        ax.set_xticks(range(1, int(s.depth.max()) + 1))
    axes[1].axhline(2000, color="black", ls="--", lw=1)
    axes[1].text(1, 2200, "2 s", fontsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Cost of strength: %d fixed positions the search does not solve by depth 10\n"
                 "(a point is drawn only if every position finished under the time cap)"
                 % s[s.config == "full"].n.max(), fontweight="bold")
    fig.tight_layout()
    return _save(fig, "cost_vs_depth.png"), fig


def strength_vs_cost_plot():
    """Elo against ms per move: diminishing returns."""
    with open(_r("elo.json")) as f:
        e = json.load(f)["ratings"]
    s = _cost_open()
    s = s[s.config == "full"].set_index("depth")
    ds = sorted(e, key=lambda p: int(p[1:]))
    x = [s.loc[int(p[1:]), "mean_ms"] for p in ds]
    y = [e[p]["elo"] for p in ds]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x, y, "-", color=OKABE["grey"], lw=1)
    for xi, yi, p in zip(x, y, ds):
        ax.plot(xi, yi, "o", color=ODD_C if int(p[1:]) % 2 else EVEN_C, ms=7)
        ax.annotate("d%s" % p[1:], (xi, yi), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("mean ms per move on unsolved positions (log scale)")
    ax.set_ylabel("Elo (depth 1 = 1000)")
    ax.set_title("Strength bought per unit of compute")
    return _save(fig, "elo_vs_cost.png"), fig


# ------------------------------------------------------------------ odd/even
def oddeven_plot():
    with open(_r("oddeven.json")) as f:
        o = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ev = o.get("root_eval_by_depth", {})
    if ev:
        d = sorted(int(k) for k in ev)
        v = [ev[str(k)]["mean"] for k in d]
        axes[0].plot(d, v, "-", color=OKABE["grey"], lw=1)
        axes[0].scatter(d, v, c=[ODD_C if k % 2 else EVEN_C for k in d], s=45, zorder=3)
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_xlabel("search depth")
        axes[0].set_ylabel("mean root score (heuristic units)")
        axes[0].set_title("Root evaluation zig-zags with parity")
    st = o.get("strength_step", [])
    if st:
        d = [s["to_depth"] for s in st]
        g = [s["elo_gain"] for s in st]
        axes[1].bar(d, g, color=[ODD_C if k % 2 else EVEN_C for k in d],
                    yerr=[[s["elo_gain"] - s["lo"] for s in st], [s["hi"] - s["elo_gain"] for s in st]],
                    capsize=3)
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_xlabel("step d-1 -> d (bar at d)")
        axes[1].set_ylabel("Elo gain")
        axes[1].set_title("Elo gained by one extra ply")
    ac = o.get("accuracy_step", [])
    if ac:
        d = [s["to_depth"] for s in ac]
        axes[2].bar(d, [100 * s["delta_optimal"] for s in ac],
                    color=[ODD_C if k % 2 else EVEN_C for k in d])
        axes[2].axhline(0, color="black", lw=0.8)
        axes[2].set_xlabel("step d-1 -> d (bar at d)")
        axes[2].set_ylabel("change in optimal-move rate (pp)")
        axes[2].set_title("Accuracy gained by one extra ply")
    for ax in axes:
        ax.plot([], [], "s", color=ODD_C, label="odd depth (bot moves last)")
        ax.plot([], [], "s", color=EVEN_C, label="even depth (opponent moves last)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Odd vs even depth (horizon effect)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "odd_even.png"), fig


ALL = [("tournament_winrate.csv", winrate_heatmap), ("elo.json", elo_plot),
       ("slider_summary.csv", slider_plot), ("responsiveness.csv", responsiveness_plot),
       ("accuracy_accuracy_summary.csv", accuracy_plot), ("tactics_summary.csv", tactics_plot),
       ("cost_summary.csv", cost_plot), ("oddeven.json", oddeven_plot)]


def main():
    ensure_dirs()
    style()
    for need, fn in ALL:
        if have(need):
            path, fig = fn()
            plt.close(fig)
            print("wrote", path)
        else:
            print("skip %s (missing %s)" % (fn.__name__, need))
    if have("elo.json", "cost_summary.csv"):
        path, fig = strength_vs_cost_plot()
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
