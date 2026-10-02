"""Odd vs even search depth (statement 4.3 #5, optional) - derived from the
tournament and accuracy results (no new searches).

    python experiments/oddeven.py      # after tournament.py and accuracy.py

At odd depth the last searched ply is the bot's own move, at even depth the
opponent's.  A static evaluation at the horizon is therefore taken right
after our move (odd: optimistic - the opponent's reply is not seen) or right
after the opponent's move (even: pessimistic).  We look for this in:
  1. root evaluation: mean root score (non-proven positions) per depth
     on the accuracy set - should zig-zag (odd high, even low);
  2. strength steps: Elo gain and head-to-head score of d+1 over d, split
     by whether d+1 is odd or even;
  3. accuracy steps: change in optimal-move rate from d to d+1, by parity.

Output: results/oddeven.json (+ printed table).  The figure is drawn by
plots.py / analysis.ipynb.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS, log, read_csv, write_json  # noqa: E402


def main():
    out = {}
    moves_path = os.path.join(RESULTS, "accuracy_accuracy_moves.csv")
    if os.path.exists(moves_path):
        ev = {}
        for r in read_csv(moves_path):
            e = float(r["eval"])
            if abs(e) < 900000:           # skip proven wins/losses
                ev.setdefault(int(r["depth"]), []).append(e)
        out["root_eval_by_depth"] = {d: {"mean": round(sum(v) / len(v), 3), "n": len(v)}
                                     for d, v in sorted(ev.items())}
        summ = [s for s in read_csv(os.path.join(RESULTS, "accuracy_accuracy_summary.csv"))
                if s["bucket"] == "all"]
        acc = {int(s["depth"]): float(s["optimal"]) for s in summ}
        keep = {int(s["depth"]): float(s["keeps_nl"]) for s in summ}
        out["accuracy_step"] = [{"to_depth": d, "parity": "odd" if d % 2 else "even",
                                 "delta_optimal": round(acc[d] - acc[d - 1], 4),
                                 "delta_keeps_nl": round(keep[d] - keep[d - 1], 4)}
                                for d in sorted(acc) if d - 1 in acc]
    elo_path = os.path.join(RESULTS, "elo.json")
    if os.path.exists(elo_path):
        with open(elo_path) as f:
            elo = json.load(f)
        steps = []
        adj = {r["deeper"]: r for r in read_csv(os.path.join(RESULTS, "tournament_adjacent.csv"))}
        for k, v in elo["step_gain"].items():
            hi = k.split("->")[1]
            d = int(hi[1:])
            a = adj.get(hi, {})
            steps.append({"to_depth": d, "parity": "odd" if d % 2 else "even",
                          "elo_gain": v["gain"], "lo": v["lo"], "hi": v["hi"],
                          "h2h_score": float(a["score"]) if a else None})
        out["strength_step"] = steps
    for key in ("strength_step", "accuracy_step"):
        if key in out:
            for par in ("odd", "even"):
                vals = [s for s in out[key] if s["parity"] == par]
                field = "elo_gain" if key == "strength_step" else "delta_optimal"
                if vals:
                    out.setdefault("parity_means", {})["%s_%s_mean_%s" % (key, par, field)] = \
                        round(sum(s[field] for s in vals) / len(vals), 4)
    write_json(os.path.join(RESULTS, "oddeven.json"), out)
    log(json.dumps(out, indent=1))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
