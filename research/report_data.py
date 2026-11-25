"""Flatten the 2026-11-24 analyses into CSVs for the report page.

    python -m research.report_data

Reads experiments/results/research/{arena_vs_pons,bench_all,interp_arena,interp_bruneton}.json
and writes experiments/results/research/report/*.csv, plus all of those tables in one file,
experiments/results/research/report.json, which the website's /research page reads
(tabs "Nets vs Pons", "Inside all arena nets", "All nets benchmarked").
"""

import csv
import json
import os
import time

from scipy import stats as sstats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "experiments", "results", "research")
OUT = os.path.join(RES, "report")
PONS = "pons_perfect"
SHORT = {"pons_perfect": "Pons solver", "nn_sup_6x64_mcts200": "sup 6x64 + MCTS200",
         "nn_az_6x64_mcts200": "az 6x64 + MCTS200", "nn_ppo_from_sup_policy": "ppo_from_sup (policy)",
         "bruneton_az": "Bruneton AlphaZero", "nn_ppo_6x64_mcts200": "ppo 6x64 + MCTS200",
         "nn_distill_az_2x32_policy": "distill_az 2x32 (policy)",
         "nn_ens3_score_8x128_policy": "3-net score ensemble (policy)",
         "nn_ens3_score_8x128_mcts200": "3-net score ensemble + MCTS200",
         "nn_score_8x128_s0_r1_mcts200": "score 8x128 s0 + MCTS200"}
MODEL_BOT = {"sup_6x64": "nn_sup_6x64_mcts200", "az_6x64": "nn_az_6x64_mcts200",
             "ppo_from_sup": "nn_ppo_from_sup_policy", "ppo_6x64": "nn_ppo_6x64_mcts200",
             "distill_az_2x32": "nn_distill_az_2x32_policy",
             "score_8x128_s0": "nn_score_8x128_s0_r1_mcts200 / ensemble member",
             "score_8x128_s1": "ensemble member", "score_8x128_s2": "ensemble member",
             "bruneton_az": "bruneton_az"}


TABLES = {}


def _typed(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) and "e" not in str(v).lower() else f
    except (TypeError, ValueError):
        return v


def write(name, rows, cols):
    TABLES[name[:-4]] = [{k: _typed(r.get(k)) for k in cols} for r in rows]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    print("wrote", name, len(rows))


def arena():
    A = json.load(open(os.path.join(RES, "arena_vs_pons.json")))
    S = json.load(open(os.path.join(ROOT, "experiments", "results", "arena", A["run"], "summary.json")))
    rows = []
    for b in A["bots"]:
        c, pb, e = A["vs_common_field"][b], A["per_bot"][b], S["elo"][b]
        r = {"bot": b, "name": SHORT.get(b, b), "elo": e["elo"], "elo_lo": e["lo"], "elo_hi": e["hi"],
             "score_all": pb["score"], "strength_weighted": A["strength_weighted_score"][b],
             "elo_top_group": A["elo_top_group_only"].get(b),
             "score_field": c["score"], "par_field": c["par"], "gifts_field": c["gifts"],
             "own_losses_field": c["own_losses"], "above_par_field": c["above_par"],
             "opp_error_rate_field": c["opp_error_rate"], "opp_only_move_field": c["opp_only_move_share"],
             "opp_safe_moves_field": c["opp_mean_safe_moves"], "own_error_rate": pb["own_error_rate"]}
        if b == PONS:
            r.update({"h2h_w": "", "h2h_score": "", "elo_from_pons_h2h": e["elo"], "elo_pred_vs_pons": ""})
        else:
            rec = A["records"][b]
            h = rec["actual_vs_pons"]
            r.update({"h2h_w": h["w"], "h2h_d": h["d"], "h2h_l": h["l"], "h2h_score": h["score"],
                      "h2h_p": h["sign_test_p"], "elo_pred_vs_pons": rec["elo_predicted_score_vs_pons"],
                      "elo_from_pons_h2h": A["elo_from_h2h_vs_pons_only"][b]})
        rows.append(r)
    write("records.csv", rows, ["bot", "name", "elo", "elo_lo", "elo_hi", "score_all", "h2h_w", "h2h_d", "h2h_l",
                                "h2h_score", "h2h_p", "elo_pred_vs_pons", "elo_from_pons_h2h", "elo_top_group",
                                "strength_weighted", "score_field", "par_field", "gifts_field", "own_losses_field",
                                "above_par_field", "opp_error_rate_field", "opp_only_move_field",
                                "opp_safe_moves_field", "own_error_rate"])
    pr = []
    for b in [PONS, "nn_ens3_score_8x128_mcts200", "nn_score_8x128_s0_r1_mcts200", "nn_ens3_score_8x128_policy",
              "nn_sup_6x64_mcts200", "nn_az_6x64_mcts200"]:
        for x in A["common_field"]:
            if x == b:
                continue
            p = A["pair"][b][x]
            pr.append({"bot": SHORT[b], "opponent": SHORT[x], "key": SHORT[b] + "|" + SHORT[x], "score": p["score"],
                       "par": p["par"], "gifts": p["gifts"], "own_losses": p["own_losses"],
                       "opp_error_rate": p["opp_error_rate"], "opp_only_move_share": p["opp_only_move_share"]})
    write("pairs.csv", pr, ["key", "bot", "opponent", "score", "par", "gifts", "own_losses", "opp_error_rate",
                            "opp_only_move_share"])
    mr, ex = [], []
    for o in ["nn_ens3_score_8x128_mcts200", "nn_score_8x128_s0_r1_mcts200", "nn_ens3_score_8x128_policy",
              "nn_sup_6x64_mcts200", "nn_az_6x64_mcts200"]:
        m = A["matched_vs_pons"][o]
        tot = {"better": 0, "worse": 0, "same": 0, "n": 0}
        for st in ("won", "drawn", "lost"):
            v = m["by_state_at_divergence"].get(st)
            if not v:
                continue
            tot["better"] += v["other_better"]
            tot["worse"] += v["other_worse"]
            tot["same"] += v["same"]
            tot["n"] += v["n"]
            mr.append({"key": SHORT[o] + "|" + st, "net": SHORT[o], "state": st, "n": v["n"],
                       "net_better": v["other_better"], "same": v["same"], "net_worse": v["other_worse"],
                       "net_move_also_keeps_value": v["other_move_also_safe"], "mean_score_diff": v["mean_score_diff"],
                       "opp_err_after_pons": v["opp_err_rate_after_strong"], "opp_err_after_net": v["opp_err_rate_after_other"],
                       "only_move_after_pons": v["only_move_share_after_strong"],
                       "only_move_after_net": v["only_move_share_after_other"]})
        p = sstats.binomtest(tot["better"], tot["better"] + tot["worse"]).pvalue
        mr.append({"key": SHORT[o] + "|all", "net": SHORT[o], "state": "all", "n": tot["n"],
                   "net_better": tot["better"], "same": tot["same"], "net_worse": tot["worse"],
                   "sign_p": round(float(p), 6), "pairs": m["pairs"], "identical_games": m["identical_games"]})
        if o == "nn_ens3_score_8x128_mcts200":
            sp = lambda g: " ".join(g)  # noqa: E731  (spaced so the strings are not read as numbers)
            for e in m["examples"]:
                if e["strong_score"] == e["other_score"]:
                    continue
                ex.append({"id": "ex%02d" % len(ex), "opponent": SHORT[e["opponent"]], "opening": e["opening"],
                           "position": sp(e["position"]),
                           "state": e["state"], "pons_move": e["strong_move"], "net_move": e["other_move"],
                           "pons_score": e["strong_score"], "net_score": e["other_score"],
                           "pons_game": sp(e["strong_game"]), "net_game": sp(e["other_game"])})
    write("matched.csv", mr, ["key", "net", "state", "n", "net_better", "same", "net_worse", "net_move_also_keeps_value",
                              "mean_score_diff", "opp_err_after_pons", "opp_err_after_net", "only_move_after_pons",
                              "only_move_after_net", "sign_p", "pairs", "identical_games"])
    write("examples.csv", ex[:20], ["id", "opponent", "opening", "position", "state", "pons_move", "net_move", "pons_score",
                                    "net_score", "pons_game", "net_game"])


def nets():
    B = json.load(open(os.path.join(RES, "bench_all.json")))
    rows = []
    for r in B["nets"]:
        t = r["training_val"]
        rows.append(dict(r, size="%dx%d" % (r["blocks"], r["channels"]),
                         train_val_policy_acc=t.get("val_policy_acc"), train_val_value_acc=t.get("val_value_acc"),
                         train_val_teacher_agree=t.get("val_teacher_agree", t.get("val_teacher_agreement")),
                         opt_8_15=r["policy_optimal_by_ply"]["8-15"], opt_16_23=r["policy_optimal_by_ply"]["16-23"],
                         opt_24_30=r["policy_optimal_by_ply"]["24-30"]))
    write("nets.csv", rows, ["run", "family", "teacher_label", "sweep", "size", "blocks", "channels", "params", "seed",
                             "policy_optimal", "policy_keeps", "played_optimal", "value_acc", "value_acc_W",
                             "value_acc_D", "value_acc_L", "value_brier", "value_corr", "score_mae", "score_sign_acc",
                             "score_head_optimal", "opt_8_15", "opt_16_23", "opt_24_30", "train_val_policy_acc",
                             "train_val_value_acc", "train_val_teacher_agree"])
    fams = {}
    for r in rows:
        f = fams.setdefault(r["family"], [])
        f.append(r)
    fr = []
    for f, rs in fams.items():
        best = max(rs, key=lambda r: r["policy_optimal"])
        fr.append({"family": f, "nets": len(rs), "best_run": best["run"], "best_policy_optimal": best["policy_optimal"],
                   "best_value_acc": max(r["value_acc"] for r in rs),
                   "mean_policy_optimal": round(sum(r["policy_optimal"] for r in rs) / len(rs), 4)})
    fr.sort(key=lambda r: -r["best_policy_optimal"])
    write("families.csv", fr, ["family", "nets", "best_run", "best_policy_optimal", "best_value_acc", "mean_policy_optimal"])


def interp():
    I = json.load(open(os.path.join(RES, "interp_arena.json")))
    Bz = json.load(open(os.path.join(RES, "interp_bruneton.json")))
    concepts = {c["id"]: c for c in I["concepts"]}
    pr, sm, ko, ch, mi = [], [], [], [], []
    for m in I["models"]:
        mid = m["id"]
        layers = m["layers"]
        summ = I["probe_summary"][mid]
        for cid, per in I["probes"][mid].items():
            ch_ = summ[cid]["majority"] if cid in summ else None
            for li, l in enumerate(layers):
                if l not in per:
                    continue
                v = per[l]["flat"]
                sk = None if v is None or ch_ is None else round((v - ch_) / max(1e-9, 1 - ch_), 4)
                pr.append({"key": "%s|%s|%s" % (mid, cid, l), "model": mid, "concept": cid,
                           "concept_label": concepts[cid]["label"], "group": concepts[cid]["group"],
                           "metric": concepts[cid]["metric"], "layer": l, "layer_index": li,
                           "flat": v, "pooled": per[l].get("pooled"), "shuffled": per[l].get("shuffled"),
                           "skill": sk})
            if cid in summ:
                s = summ[cid]
                sm.append({"key": mid + "|" + cid, "model": mid, "concept": cid, "group": concepts[cid]["group"],
                           "first_layer": s["first_layer"], "best_layer": s["best_layer"], "best": s["best"],
                           "norm_best": s["norm_best"], "input_baseline": s["input_baseline"],
                           "survives_pooling": s["survives_pooling"]})
        for b in I["ablation"]["blocks"][mid]:
            if b["mode"] != "identity":
                continue
            o = b["overall"]
            ko.append({"key": mid + "|" + b["layer"], "model": mid, "block": b["layer"],
                       "d_agree": o["agree"], "d_value_acc": o["value_acc"], "kl": o["kl"],
                       "d_missed_win": o.get("missed_win"), "d_missed_block": o.get("missed_block"),
                       "d_agree_early": b["by_phase"]["early"]["agree"], "d_agree_mid": b["by_phase"]["mid"]["agree"],
                       "d_agree_late": b["by_phase"]["late"]["agree"],
                       "d_agree_own_win": b["by_type"]["own_win"]["agree"],
                       "d_agree_must_block": b["by_type"]["must_block"]["agree"],
                       "d_agree_quiet": b["by_type"]["quiet"]["agree"]})
        for layer, cnt in I["channel_counts"][mid].items():
            for lab, n in cnt.items():
                ch.append({"key": "%s|%s|%s" % (mid, layer, lab), "model": mid, "layer": layer, "label": lab, "count": n})
        it = I["ablation"]["intact"][mid]["all"]
        pz = [abs(c_["add"]["z"]) for c_ in I["patching"][mid].get("per_column", []) if c_.get("add")]
        mi.append({"model": mid, "arena_bot": MODEL_BOT.get(mid, ""), "blocks": m["blocks"], "channels": m["channels"],
                   "params": m["params"], "intact_agree": it["agree"], "intact_value_acc": it["value_acc"],
                   "patch_layer": I["patching"][mid].get("layer"),
                   "patch_mean_abs_z": round(sum(pz) / len(pz), 2) if pz else None})
    # bruneton
    for cid, per in Bz["probes"].items():
        for li, l in enumerate(Bz["layers"]):
            d = per[l]
            pr.append({"key": "bruneton_az|%s|%s" % (cid, l), "model": "bruneton_az", "concept": cid,
                       "concept_label": concepts[cid]["label"], "group": concepts[cid]["group"],
                       "metric": concepts[cid]["metric"], "layer": l, "layer_index": li, "flat": d["flat"],
                       "shuffled": d["shuffled"], "skill": d["skill"]})
        sk = {l: per[l]["skill"] for l in Bz["layers"]}
        best_l = max(sk, key=lambda l: sk[l] if sk[l] == sk[l] else -9)
        first = next(l for l in Bz["layers"] if sk[l] >= 0.9 * sk[best_l])
        sm.append({"key": "bruneton_az|" + cid, "model": "bruneton_az", "concept": cid, "group": concepts[cid]["group"],
                   "first_layer": first, "best_layer": best_l, "best": per[best_l]["flat"], "norm_best": sk[best_l],
                   "input_baseline": per["input"]["flat"]})
    ko.append({"key": "bruneton_az|block1", "model": "bruneton_az", "block": "block1",
               "d_agree": round(Bz["block1_skipped"]["agree"] - Bz["intact"]["agree"], 4),
               "d_value_acc": round(Bz["block1_skipped"]["value_acc_thirds"] - Bz["intact"]["value_acc_thirds"], 4)})
    mi.append({"model": "bruneton_az", "arena_bot": "bruneton_az", "blocks": 1, "channels": 256, "params": Bz["params"],
               "intact_agree": Bz["intact"]["agree"], "intact_value_acc": Bz["intact"]["value_acc_thirds"]})
    write("probes.csv", pr, ["key", "model", "concept", "concept_label", "group", "metric", "layer", "layer_index",
                             "flat", "pooled", "shuffled", "skill"])
    write("probe_summary.csv", sm, ["key", "model", "concept", "group", "first_layer", "best_layer", "best",
                                    "norm_best", "input_baseline", "survives_pooling"])
    write("knockouts.csv", ko, ["key", "model", "block", "d_agree", "d_value_acc", "kl", "d_missed_win",
                                "d_missed_block", "d_agree_early", "d_agree_mid", "d_agree_late", "d_agree_own_win",
                                "d_agree_must_block", "d_agree_quiet"])
    write("channels.csv", ch, ["key", "model", "layer", "label", "count"])
    write("models.csv", mi, ["model", "arena_bot", "blocks", "channels", "params", "intact_agree", "intact_value_acc",
                             "patch_layer", "patch_mean_abs_z"])


def next_turn():
    p = os.path.join(RES, "arena_next_turn.json")
    if not os.path.isfile(p):
        return
    N = json.load(open(p))
    rows = []
    for net, d in N.items():
        for st, v in d.items():
            rows.append(dict(v, key=SHORT[net] + "|" + st, net=SHORT[net], state=st))
    write("next_turn.csv", rows, ["key", "net", "state", "n", "mean_safe_pons", "mean_safe_net", "only_move_pons",
                                  "only_move_net", "next_move_err_pons", "next_move_err_net"])


if __name__ == "__main__":
    arena()
    next_turn()
    nets()
    if os.path.isfile(os.path.join(RES, "interp_arena.json")):
        interp()
    A = json.load(open(os.path.join(RES, "arena_vs_pons.json")))
    out = {"id": "report", "title": "Nets vs Pons, inside all arena nets, every net benchmarked",
           "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "arena_run": A["run"], "games": A["games"],
           "positions_solved": A["positions"], "common_field": A["common_field"], "tables": TABLES}
    tmp = os.path.join(RES, ".report.json.part")
    with open(tmp, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(RES, "report.json"))
    print("wrote report.json", os.path.getsize(os.path.join(RES, "report.json")))
