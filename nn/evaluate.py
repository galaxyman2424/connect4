"""Full evaluation of a trained network -> runs/<run>/eval.json (shown on the Lab page).

    python -m nn.evaluate --run az_6x64                  # uses runs/az_6x64/model.npz
    python -m nn.evaluate --run sup_6x64 --model best.npz
    python -m nn.evaluate --run az_6x64 --quick          # smaller, ~5 min smoke version

Sections
1. accuracy   held-out accuracy vs perfect play (3,000 solver-labelled
              positions): policy head alone and MCTS at each --sims, next to
              minimax depth 1-10 from experiments/results.
2. value      WDL head vs the solver's true outcome (+ confusion matrix).
3. tactics    the 52-position tactics suite (win in 1/2/3, forced block).
4. ladder     MCTS player vs minimax d1..d8 (--openings openings x 2 colours
              each) -> Elo on the same scale as experiments/results/elo.json
              (minimax d1 = 1000).
5. hypothesis the dossier hypothesis: minimax at depth d with the NETWORK's
              value as its evaluation vs minimax at the same depth with the
              HAND-CRAFTED heuristic, for each --hyp-depths:
                head-to-head score (+ exact binomial test on decisive games),
                accuracy on --hyp-positions benchmark positions (paired,
                McNemar test), tactics (win-in-1 and block must stay 100%),
                and time per move.
              Success criterion (from the dossier): the NN version scores
              >= 50% at the same depth, still takes wins-in-1 and makes forced
              blocks at least as often as the hand-crafted version at that
              depth, and stays under 2 s per move.
"""

import argparse
import json
import os
import time

from . import bench
from .common import Run, load_evaluator, log, now_iso


def parse_list(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def _acc_task(task):
    """Accuracy of one minimax configuration on a list of bench positions."""
    spec, rows = task
    agent = bench.make_agent(spec)
    from engine.board import Board
    cols, ms = [], []
    for r in rows:
        b = Board.from_moves(r["moves"])
        t0 = time.perf_counter()
        cols.append(agent.choose(b))
        ms.append((time.perf_counter() - t0) * 1000)
    return cols, ms


def _par_accuracy(spec, rows, workers):
    """Split rows over workers; returns (cols, ms) in order."""
    if workers <= 1:
        return _acc_task((spec, rows))
    import multiprocessing as mp
    k = max(1, len(rows) // (workers * 4))
    chunks = [rows[i:i + k] for i in range(0, len(rows), k)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_acc_task, [(spec, c) for c in chunks])
    cols = sum((p[0] for p in parts), [])
    ms = sum((p[1] for p in parts), [])
    return cols, ms


def tactics_reference():
    import csv
    p = os.path.join(bench.ROOT, "experiments", "results", "tactics_summary.csv")
    out = {}
    if os.path.isfile(p):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["agent"].startswith("d"):
                    out.setdefault(r["agent"].replace("d", "minimax_d"), {})[r["category"]] = float(r["share_solved"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="model.npz", help="file inside runs/<run>/ (or a path)")
    ap.add_argument("--sims", default="50,200,800", help="MCTS simulation budgets to test")
    ap.add_argument("--ladder-sims", type=int, default=200, help="MCTS sims for the minimax ladder games")
    ap.add_argument("--depths", default="1-8", help="minimax depths for the ladder")
    ap.add_argument("--openings", type=int, default=20)
    ap.add_argument("--hyp-depths", default="1,2,3,4")
    ap.add_argument("--hyp-openings", type=int, default=20)
    ap.add_argument("--hyp-positions", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--quick", action="store_true", help="small version for a smoke test")
    ap.add_argument("--skip", default="", help="comma list of sections to skip: accuracy,tactics,ladder,hypothesis")
    a = ap.parse_args()
    if a.quick:
        a.sims, a.depths, a.openings = "50", "1,2,4", 4
        a.hyp_depths, a.hyp_openings, a.hyp_positions, a.ladder_sims = "1,2", 4, 60, 50
    skip = set(a.skip.split(","))

    run = Run(a.run, create=False)
    model_path = a.model if os.path.isfile(a.model) else run.path(a.model)
    if not os.path.isfile(model_path):
        raise SystemExit("model not found: %s" % model_path)
    ev = load_evaluator(model_path, "auto")
    meta = getattr(ev, "meta", {})
    out_path = run.path("eval.json")
    out = {"run": a.run, "model": os.path.relpath(model_path, bench.ROOT).replace("\\", "/"),
           "model_meta": meta, "created": now_iso(), "settings": vars(a)}
    if os.path.isfile(out_path):                 # keep sections we skip this time
        with open(out_path, encoding="utf-8") as f:
            old = json.load(f)
        section_of = {"accuracy": "accuracy", "value": "accuracy", "tactics": "tactics",
                      "ladder": "ladder", "hypothesis": "hypothesis"}
        for k, sec in section_of.items():
            if k in old and sec in skip:
                out[k] = old[k]

    def save():
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    rows = bench.load_accuracy_bench()
    t0 = time.time()

    # 1+2 ---------------------------------------------------------------
    if "accuracy" not in skip:
        log("accuracy vs perfect play (%d positions) ..." % len(rows))
        pol = bench.policy_eval(ev, rows)
        acc = {"nn_policy": {k: pol[k] for k in ("n", "optimal", "keeps_nl", "by_bucket")}}
        for s in parse_list(a.sims):
            r = bench.mcts_eval(ev, rows, s)
            acc["nn_mcts%d" % s] = r
            log("  MCTS %d sims: optimal %.3f keeps %.3f" % (s, r["optimal"], r["keeps_nl"]))
        acc["minimax_reference"] = bench.minimax_reference()
        out["accuracy"] = acc
        out["value"] = {"accuracy": pol["value_acc"], "by_bucket": pol["value_acc_by_bucket"],
                        "confusion": pol["value_confusion"], "labels": ["win", "draw", "loss"]}
        log("  policy optimal %.3f keeps %.3f, value accuracy %.3f" % (pol["optimal"], pol["keeps_nl"], pol["value_acc"]))
        save()

    # 3 -----------------------------------------------------------------
    if "tactics" not in skip:
        from .agents import MCTSAgent, PolicyAgent
        tac = {"nn_policy": bench.tactics_eval(PolicyAgent(ev))}
        for s in parse_list(a.sims):
            tac["nn_mcts%d" % s] = bench.tactics_eval(MCTSAgent(ev, sims=s))
        tac["reference"] = tactics_reference()
        out["tactics"] = tac
        log("tactics: " + ", ".join("%s %.2f" % (k, v["all"]) for k, v in tac.items() if k != "reference"))
        save()

    # 4 -----------------------------------------------------------------
    if "ladder" not in skip:
        depths = parse_list(a.depths)
        openings = bench.random_openings(a.openings, seed=2026)
        me = ("nn", model_path, "mcts", a.ladder_sims)
        log("ladder: nn_mcts%d vs minimax d%s, %d games each ..." % (a.ladder_sims, depths, 2 * len(openings)))
        tasks = [(me, ("minimax", d), o, af) for d in depths for o in openings for af in (True, False)]
        games = bench.run_tasks(tasks, a.workers)
        ladder = bench.minimax_ladder_elo()
        res, fixed = {}, []
        for d in depths:
            s = bench.summarize_match([g for g in games if g["b"] == "minimax_d%d" % d])
            res["minimax_d%d" % d] = s
            if "minimax_d%d" % d in ladder:
                fixed.append((ladder["minimax_d%d" % d], s["win"] + 0.5 * s["draw"], s["games"]))
            log("  vs d%d: %d-%d-%d (score %.2f)" % (d, s["win"], s["draw"], s["loss"], s["score"]))
        out["ladder"] = {"player": "nn_mcts%d" % a.ladder_sims, "vs": res,
                         "elo": bench.elo_vs_fixed(fixed), "ladder_elo": ladder,
                         "openings": len(openings), "games": games}
        log("  Elo on the minimax ladder (d1 = 1000): %s" % out["ladder"]["elo"])
        save()

    # 5 -----------------------------------------------------------------
    if "hypothesis" not in skip:
        import random
        from experiments import stats
        hdepths = parse_list(a.hyp_depths)
        sub = random.Random(7).sample(rows, min(a.hyp_positions, len(rows)))
        openings = bench.random_openings(a.hyp_openings, seed=77)
        hyp = {"depths": hdepths, "positions": len(sub), "openings": len(openings), "rows": []}
        for d in hdepths:
            log("hypothesis depth %d: NN-eval minimax vs hand-crafted minimax ..." % d)
            nn_spec, hand_spec = ("nn", model_path, "minimax", d), ("minimax", d)
            m, games = bench.match(nn_spec, hand_spec, openings, a.workers)
            nn_cols, nn_ms = _par_accuracy(nn_spec, sub, a.workers)
            hd_cols, hd_ms = _par_accuracy(hand_spec, sub, a.workers)
            nn_acc = bench.accuracy_summary(sub, nn_cols)
            hd_acc = bench.accuracy_summary(sub, hd_cols)
            only_nn = sum(nn_cols[i] in sub[i]["optimal"] and hd_cols[i] not in sub[i]["optimal"] for i in range(len(sub)))
            only_hd = sum(hd_cols[i] in sub[i]["optimal"] and nn_cols[i] not in sub[i]["optimal"] for i in range(len(sub)))
            from .agents import NNMinimaxAgent
            nn_tac = bench.tactics_eval(NNMinimaxAgent(ev, d))
            from engine.agents import MinimaxAgent
            hd_tac = bench.tactics_eval(MinimaxAgent(d))
            row = {"depth": d, "head_to_head": m,
                   "binom_p": round(stats.binom_test(m["win"], m["loss"]), 4),
                   "nn_accuracy": {k: nn_acc[k] for k in ("optimal", "keeps_nl")},
                   "hand_accuracy": {k: hd_acc[k] for k in ("optimal", "keeps_nl")},
                   "mcnemar_only_nn": only_nn, "mcnemar_only_hand": only_hd,
                   "mcnemar_p": round(stats.mcnemar(only_nn, only_hd), 4),
                   "nn_tactics": nn_tac, "hand_tactics": hd_tac,
                   "nn_ms": round(sum(nn_ms) / len(nn_ms), 2), "hand_ms": round(sum(hd_ms) / len(hd_ms), 2)}
            ok_score = m["score"] >= 0.5
            # "still blocks and takes obvious wins": at least as often as the hand-crafted
            # heuristic at the same depth (depth-1 minimax cannot see blocks at all, so
            # demanding 100% there would test the depth, not the evaluation)
            ok_tac = nn_tac.get("win1", 0) >= hd_tac.get("win1", 0) and nn_tac.get("block", 0) >= hd_tac.get("block", 0)
            ok_time = row["nn_ms"] <= 2000
            row["criteria"] = {"score_at_least_50pct": ok_score, "wins_and_blocks_at_least_as_hand": ok_tac,
                               "fast_enough_2s": ok_time}
            row["supported"] = ok_score and ok_tac and ok_time
            hyp["rows"].append(row)
            log("  d%d: NN-eval scores %.2f vs hand (%d-%d-%d, p=%.3g); optimal %.3f vs %.3f; "
                "tactics %.2f vs %s; %.0f ms vs %.0f ms/move" % (
                    d, m["score"], m["win"], m["draw"], m["loss"], row["binom_p"],
                    nn_acc["optimal"], hd_acc["optimal"], nn_tac["all"], hd_tac.get("all"),
                    row["nn_ms"], row["hand_ms"]))
            out["hypothesis"] = hyp
            save()
        n_ok = sum(r["supported"] for r in hyp["rows"])
        hyp["verdict"] = ("supported at %d of %d depths (%s)" % (
            n_ok, len(hyp["rows"]), ", ".join("d%d %s" % (r["depth"], "yes" if r["supported"] else "no")
                                           for r in hyp["rows"])))
        save()
    out["seconds"] = round(time.time() - t0, 1)
    save()
    log("wrote %s (%.0fs)" % (out_path, out["seconds"]))


if __name__ == "__main__":
    main()
