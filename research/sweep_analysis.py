"""Analysis of the supervised sweep (research/sweep.py) -> sweep.json "analysis" + "findings".

Pure functions of the run / teacher tables, recomputed every time sweep.json is written,
so partial results are always consistent.  Only finished runs (status "done") count.
"""

import math

import numpy as np

SMALL_N = 4
BOOT = 2000


def _done(runs):
    return [r for r in runs if r.get("status") == "done" and r.get("bench")]


def _opt(r):
    return r["bench"]["optimal"]


def _gscore(r):
    return (r.get("gauntlet") or {}).get("mean_score")


def regress(x, y, boot=BOOT, seed=0):
    """OLS y = a + b x.  Returns slope, intercept, r2, n and a bootstrap 95 % CI of the
    slope (resampling points; None when n < 3)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 2 or np.ptp(x) == 0:
        return {"n": int(n), "slope": None, "intercept": None, "r2": None, "slope_ci95": None,
                "small_n": True}
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    ci = None
    if n >= 3:
        rng = np.random.default_rng(seed)
        sl = []
        for _ in range(boot):
            i = rng.integers(0, n, n)
            if np.ptp(x[i]) == 0:
                continue
            sl.append(np.polyfit(x[i], y[i], 1)[0])
        if sl:
            ci = [round(float(np.percentile(sl, 2.5)), 5), round(float(np.percentile(sl, 97.5)), 5)]
    return {"n": int(n), "slope": round(float(b), 5), "intercept": round(float(a), 5),
            "r2": None if r2 is None else round(r2, 4), "slope_ci95": ci, "small_n": n < SMALL_N}


def _series(rs, key):
    pts = [(math.log10(r["params"]), key(r)) for r in rs if key(r) is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def size_regressions(runs):
    """Per teacher (all sizes, all seeds) and per distillation ladder."""
    out = {}
    for t in sorted({r["teacher"] for r in runs}):
        rs = [r for r in runs if r["teacher"] == t]
        if len(rs) < 2:
            continue
        ent = {"teacher": t, "groups": sorted({r["group"] for r in rs}),
               "points": [{"name": r["name"], "params": r["params"], "log10_params": round(math.log10(r["params"]), 4),
                           "optimal": _opt(r), "gauntlet": _gscore(r)} for r in rs]}
        ent["optimal_vs_log10_params"] = regress(*_series(rs, _opt))
        ent["gauntlet_vs_log10_params"] = regress(*_series(rs, _gscore))
        out[t] = ent
    return out


def _by(runs, teacher, b, c, seed=0):
    for r in runs:
        if r["teacher"] == teacher and r["blocks"] == b and r["channels"] == c and r.get("seed", 0) == seed:
            return r
    return None


def teacher_vs_student(runs, teachers):
    tq = {t["id"]: t for t in teachers}
    out = {}
    for (b, c) in sorted({(r["blocks"], r["channels"]) for r in runs}):
        rows = []
        for t in tq:
            r = _by(runs, t, b, c)
            if r is None or tq[t].get("bench_optimal") is None:
                continue
            rows.append({"teacher": t, "kind": tq[t]["kind"], "teacher_optimal": tq[t]["bench_optimal"],
                         "teacher_keeps_nl": tq[t]["bench_keeps_nl"], "student_optimal": _opt(r),
                         "student_keeps_nl": r["bench"]["keeps_nl"],
                         "delta_optimal": round(_opt(r) - tq[t]["bench_optimal"], 4),
                         "delta_keeps_nl": round(r["bench"]["keeps_nl"] - tq[t]["bench_keeps_nl"], 4),
                         "student_gauntlet": _gscore(r), "teacher_agreement": r.get("teacher_agreement")})
        if len(rows) < 2:
            continue
        x = [q["teacher_optimal"] for q in rows]
        y = [q["student_optimal"] for q in rows]
        ent = {"size": "%dx%d" % (b, c), "rows": rows, "student_vs_teacher": regress(x, y)}
        if len(rows) >= 3:
            ent["pearson_r"] = round(float(np.corrcoef(x, y)[0, 1]), 4)
        out["%dx%d" % (b, c)] = ent
    return out


def evo_curve(runs, teachers):
    tq = {t["id"]: t for t in teachers}
    ids = [("heur_hand", 0), ("heur_evo_g10", 10), ("heur_evo_g20", 20), ("heur_evo_g40", 40),
           ("heur_evo_g60", 60)]
    rows = []
    for tid, g in ids + [("mm_d4_hand", 0), ("mm_d4_evo60", 60)]:
        t = tq.get(tid, {})
        row = {"teacher": tid, "gen": g, "search_depth": t.get("search_depth"),
               "teacher_optimal": t.get("bench_optimal"), "teacher_keeps_nl": t.get("bench_keeps_nl"),
               "teacher_agree_pons": t.get("agree_pons")}
        for b, c in ((2, 32), (6, 64)):
            r = _by(runs, tid, b, c)
            row["student_%dx%d_optimal" % (b, c)] = _opt(r) if r else None
            row["student_%dx%d_gauntlet" % (b, c)] = _gscore(r) if r else None
        rows.append(row)
    nos = [r for r in rows[:5] if r["teacher_optimal"] is not None]
    reg = {"teacher_optimal_vs_gen": regress([r["gen"] for r in nos], [r["teacher_optimal"] for r in nos])}
    for s in ("2x32", "6x64"):
        pts = [r for r in rows[:5] if r["student_%s_optimal" % s] is not None]
        reg["student_%s_optimal_vs_gen" % s] = regress([r["gen"] for r in pts],
                                                       [r["student_%s_optimal" % s] for r in pts])
    return {"rows": rows, "regressions": reg}


def search_vs_nosearch(runs, teachers):
    tq = {t["id"]: t for t in teachers}
    out = []
    for heur, nos, mm in (("hand", "heur_hand", "mm_d4_hand"), ("evolved gen 60", "heur_evo_g60", "mm_d4_evo60")):
        row = {"heuristic": heur, "no_search": nos, "search_d4": mm}
        for k, t in (("no_search", nos), ("search_d4", mm)):
            row[k + "_teacher_optimal"] = tq.get(t, {}).get("bench_optimal")
            for b, c in ((2, 32), (6, 64)):
                r = _by(runs, t, b, c)
                row["%s_student_%dx%d_optimal" % (k, b, c)] = _opt(r) if r else None
                row["%s_student_%dx%d_gauntlet" % (k, b, c)] = _gscore(r) if r else None
        out.append(row)
    return out


def distill_vs_direct(runs, teachers):
    tq = {t["id"]: t for t in teachers}
    big = {"sup6x64": tq.get("sup6x64", {}), "sup8x128": tq.get("sup8x128", {})}
    out = []
    for b, c in ((1, 16), (2, 32), (3, 48), (4, 64)):
        row = {"size": "%dx%d" % (b, c)}
        p = None
        for route, t in (("direct", "pons"), ("from_6x64", "sup6x64"), ("from_8x128", "sup8x128")):
            r = _by(runs, t, b, c)
            if r:
                p = r["params"]
            row[route + "_optimal"] = _opt(r) if r else None
            row[route + "_keeps_nl"] = r["bench"]["keeps_nl"] if r else None
            row[route + "_gauntlet"] = _gscore(r) if r else None
            if t != "pons" and r and big[t].get("bench_optimal") and big[t].get("params"):
                row[route + "_retained"] = round(_opt(r) / big[t]["bench_optimal"], 4)
                row[route + "_compression"] = round(big[t]["params"] / r["params"], 2)
        row["params"] = p
        out.append(row)
    return {"rows": out, "teachers": {k: {"params": v.get("params"), "bench_optimal": v.get("bench_optimal")}
                                      for k, v in big.items()}}


def pareto(runs, xkey):
    pts = [r for r in runs if r.get(xkey) is not None]
    front = []
    for r in sorted(pts, key=lambda r: (r[xkey], -_opt(r))):
        if not front or _opt(r) > _opt(front[-1]):
            front.append(r)
    return [{"name": r["name"], xkey: r[xkey], "optimal": _opt(r)} for r in front]


def seed_noise(runs):
    out = []
    for r in runs:
        if r.get("seed", 0) != 0:
            r0 = _by(runs, r["teacher"], r["blocks"], r["channels"], 0)
            if r0:
                out.append({"size": "%dx%d" % (r["blocks"], r["channels"]), "teacher": r["teacher"],
                            "optimal_s0": _opt(r0), "optimal_s1": _opt(r),
                            "abs_diff_optimal": round(abs(_opt(r) - _opt(r0)), 4),
                            "gauntlet_s0": _gscore(r0), "gauntlet_s1": _gscore(r),
                            "abs_diff_gauntlet": None if _gscore(r) is None or _gscore(r0) is None
                            else round(abs(_gscore(r) - _gscore(r0)), 4)})
    return out


def analyse(runs, teachers):
    d = _done(runs)
    return {"n_done": len(d),
            "size_regression": size_regressions(d),
            "teacher_vs_student": teacher_vs_student(d, teachers),
            "evo_curve": evo_curve(d, teachers),
            "search_vs_nosearch": search_vs_nosearch(d, teachers),
            "distill_vs_direct": distill_vs_direct(d, teachers),
            "pareto_params": pareto(d, "params"),
            "pareto_ms": pareto(d, "ms_per_pos"),
            "seed_noise": seed_noise(d)}


# ------------------------------------------------------------------ findings
def _f(fid, title, claim, evidence, caveat):
    return {"id": "sweep-" + fid, "title": title, "claim": claim, "evidence": evidence, "caveat": caveat}


def findings(runs, teachers, an):
    """Templated findings, only when their supporting runs exist."""
    out = []
    reg = an["size_regression"].get("pons")
    if reg and reg["optimal_vs_log10_params"]["slope"] is not None and len(reg["points"]) >= 3:
        rr = reg["optimal_vs_log10_params"]
        pts = sorted(reg["points"], key=lambda p: p["params"])
        out.append(_f("size-pons", "Benchmark accuracy vs network size (solver labels)",
                      "On solver labels the optimal-move rate rises by %.3f per tenfold increase in parameters "
                      "(R^2 %.2f, n=%d), from %.3f at %d params to %.3f at %d params." % (
                          rr["slope"], rr["r2"] or 0, rr["n"], pts[0]["optimal"], pts[0]["params"],
                          pts[-1]["optimal"], pts[-1]["params"]),
                      "analysis.size_regression.pons; runs rs_A_pons_*",
                      "One training schedule for all sizes; bootstrap slope CI %s%s." % (
                          rr["slope_ci95"], " (small n)" if rr["small_n"] else "")))
    tvs = an["teacher_vs_student"]
    for size, ent in tvs.items():
        rows = [r for r in ent["rows"] if r["kind"] in ("heuristic", "minimax")]
        beat = [r for r in rows if r["delta_optimal"] > 0]
        if len(rows) >= 2:
            best = max(rows, key=lambda r: r["delta_optimal"])
            out.append(_f("exceed-" + size, "Do %s students exceed their search-free / search teachers?" % size,
                          "%d of %d heuristic/minimax teachers were exceeded in benchmark optimal-rate by their "
                          "%s student; largest gap %+.3f (%s: teacher %.3f, student %.3f)." % (
                              len(beat), len(rows), size, best["delta_optimal"], best["teacher"],
                              best["teacher_optimal"], best["student_optimal"]),
                          "analysis.teacher_vs_student.%s" % size,
                          "Single seed; 3,000 benchmark positions (SE about 0.007)."))
        if "pearson_r" in ent:
            out.append(_f("teacher-quality-" + size, "Teacher quality predicts student quality (%s)" % size,
                          "Across %d teachers at %s, student optimal-rate vs teacher optimal-rate has Pearson r = "
                          "%.2f, slope %.2f." % (len(ent["rows"]), size, ent["pearson_r"],
                                                 ent["student_vs_teacher"]["slope"] or 0),
                          "analysis.teacher_vs_student.%s" % size,
                          "Teachers differ in kind (hard vs soft targets, value-target definition)."))
    evo = an["evo_curve"]["rows"]
    have = [r for r in evo[:5] if r["teacher_optimal"] is not None and r["student_2x32_optimal"] is not None]
    if len(have) >= 3:
        out.append(_f("evo-gen", "Length of evolved training vs teacher and student quality",
                      "No-search evolved-heuristic teachers at generations %s score %s optimal; their 2x32 "
                      "students score %s." % (
                          ",".join(str(r["gen"]) for r in have),
                          ", ".join("%.3f" % r["teacher_optimal"] for r in have),
                          ", ".join("%.3f" % r["student_2x32_optimal"] for r in have)),
                      "analysis.evo_curve",
                      "Weights were evolved with depth-2 search (runs/evo_d2) and are used here without search."))
    for row in an["search_vs_nosearch"]:
        a, b = row.get("no_search_student_2x32_optimal"), row.get("search_d4_student_2x32_optimal")
        if a is not None and b is not None:
            out.append(_f("search-" + row["no_search"], "Search vs no-search teacher (%s heuristic)" % row["heuristic"],
                          "Teacher optimal-rate %.3f without search vs %.3f with depth-4 search; 2x32 students "
                          "%.3f vs %.3f." % (row["no_search_teacher_optimal"], row["search_d4_teacher_optimal"], a, b),
                          "analysis.search_vs_nosearch", "Single seed per student."))
    for row in an["distill_vs_direct"]["rows"]:
        vals = {k: row.get(k + "_optimal") for k in ("direct", "from_6x64", "from_8x128")}
        if sum(v is not None for v in vals.values()) >= 2:
            txt = ", ".join("%s %.3f" % (k.replace("_", " "), v) for k, v in vals.items() if v is not None)
            out.append(_f("distill-" + row["size"], "Distilled vs direct at %s" % row["size"],
                          "At %s (%s params) the benchmark optimal-rate is: %s." % (row["size"], row["params"], txt),
                          "analysis.distill_vs_direct", "Single seed; distillation targets are policy-only soft labels."))
    sn = an["seed_noise"]
    if sn:
        m = max(s["abs_diff_optimal"] for s in sn)
        out.append(_f("seed-noise", "Seed-to-seed noise",
                      "Across %d size(s) trained with two seeds the optimal-rate differs by at most %.4f." % (len(sn), m),
                      "analysis.seed_noise", "Two seeds only."))
    return out
