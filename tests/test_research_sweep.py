"""Fast tests for research/teachers.py, research/sweep.py and research/sweep_analysis.py."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.board import Board  # noqa: E402
from engine.minimax import MATE_THRESHOLD, WIN, search  # noqa: E402
from research import sweep, sweep_analysis, teachers  # noqa: E402


def _bits(moves):
    b = Board.from_moves(moves)
    return b.current_bits(), b.mask


# ------------------------------------------------------------------ teachers
def test_heuristic_teacher_takes_immediate_win():
    # P1 has three in column 1 (moves "1212 12" -> P1 stones at col 0 heights 0-2), P1 to move
    cur, mask = _bits("121212")
    sc = teachers.heuristic_move_scores(cur, mask)
    assert sc[0] == WIN - 1
    assert teachers.best_col(sc) == 0
    pol, wdl, top, _ = teachers.targets_from_scores(np.array([[np.nan if s is None else s for s in sc]]))
    assert top[0] == 0 and pol[0, 0] == 1.0
    assert wdl[0].tolist() == [1.0, 0.0, 0.0]          # proven win -> W


def test_heuristic_teacher_equals_depth1_search():
    for mv in ("", "4", "4453", "44444", "123456712", "443322"):
        b = Board.from_moves(mv)
        sc = teachers.heuristic_move_scores(b.current_bits(), b.mask)
        res = search(b, 1)
        assert teachers.best_col(sc) == res.col
        assert max(s for s in sc if s is not None) == res.score


def test_minimax_teacher_blocks_forced_threat():
    # P2 (to move) must block column 0 where P1 has three stacked
    cur, mask = _bits("12121")
    sc = teachers.minimax_move_scores(cur, mask, 4)
    assert teachers.best_col(sc) == 0
    others = [s for c, s in enumerate(sc) if c != 0 and s is not None]
    assert all(s <= -MATE_THRESHOLD for s in others)    # everything else loses at once
    # and the depth-4 choice equals the engine's own depth-4 search (slider level 5)
    for mv in ("4453", "44444", "3344"):
        b = Board.from_moves(mv)
        assert teachers.best_col(teachers.minimax_move_scores(b.current_bits(), b.mask, 4)) == search(b, 4).col


def test_label_positions_shapes_and_targets():
    pos = [_bits(m) for m in ("", "4", "121212", "12121", "4453")]
    cur = [p[0] for p in pos]
    mask = [p[1] for p in pos]
    for tid in ("heur_hand", "heur_evo_g60", "mm_d4_hand"):
        lab = teachers.label_positions(tid, cur, mask, workers=1)
        assert lab["pol"].shape == (5, 7) and lab["wdl"].shape == (5, 3)
        assert np.allclose(lab["pol"].sum(1), 1) and np.allclose(lab["wdl"].sum(1), 1)
        assert lab["top"][2] == 0                       # immediate win taken
    lab = teachers.label_positions("mm_d4_hand", cur, mask, workers=1)
    assert lab["top"][3] == 0                           # forced block found by search


def test_pons_targets_match_train_supervised():
    sc = np.array([[-128, 1, 3, 3, -2, 0, -128], [0, 0, -1, -128, -3, 0, -1]], dtype=np.int8)
    pol, wdl, top, opt = teachers.pons_targets(sc)
    assert np.allclose(pol[0], [0, 0, .5, .5, 0, 0, 0]) and wdl[0].tolist() == [1, 0, 0]
    assert np.allclose(pol[1], [1 / 3, 1 / 3, 0, 0, 0, 1 / 3, 0]) and wdl[1].tolist() == [0, 1, 0]
    assert top[0] == 3 and top[1] == 1                  # most central optimal column


def test_evo_weights_generation_zero_is_hand():
    from rl.features import HAND_WEIGHTS
    assert teachers.evo_weights(0) == [float(x) for x in HAND_WEIGHTS]


# ------------------------------------------------------------------ sweep queue
def test_param_count_matches_model():
    torch = pytest.importorskip("torch")  # noqa: F841
    from nn.model import build, count_params
    for b, c in ((1, 16), (2, 32), (6, 64)):
        assert sweep.param_count(b, c) == count_params(build({"blocks": b, "channels": c}))


def test_plan_order_and_names():
    q = sweep.plan()
    names = [r["name"] for r in q]
    assert len(names) == len(set(names))
    groups = [r["group"] for r in q if r["seed"] == 0 and not r["optional"]]
    assert groups == sorted(groups)                     # A before B before C
    assert names.index("rs_A_pons_8x128") < names.index("rs_C_sup8x128_1x16")


def test_queue_resume_with_fake_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "RUNS", str(tmp_path))
    monkeypatch.setattr(sweep, "LOCK", str(tmp_path / "lock.json"))
    calls = []

    def fake_train(spec):
        calls.append(("train", spec["name"]))
        if spec["name"].endswith("bad"):
            raise RuntimeError("boom")
        os.makedirs(sweep.run_dir(spec["name"]), exist_ok=True)
        return {"params": 10, "train_sec": 1.0, "gpu_sec": 0.0, "device_name": "cpu",
                "val_teacher_agree": 0.5, "best_epoch": 1}

    def fake_eval(spec):
        calls.append(("eval", spec["name"]))
        return {"bench": {"optimal": 0.5, "keeps_nl": 0.6}, "gauntlet": {"mean_score": 0.4}, "ms_per_pos": 0.1,
                "eval_sec": 0.1}

    q = [{"name": n, "group": "A", "teacher": "pons", "blocks": 1, "channels": 16, "seed": 0, "epochs": 1,
          "optional": False} for n in ("rs_t1", "rs_bad", "rs_t2")]
    res = sweep.run_queue([dict(s) for s in q], train_fn=fake_train, eval_fn=fake_eval, write=False)
    assert res == {"rs_t1": "done", "rs_bad": "failed", "rs_t2": "done"}
    calls.clear()
    # resume: finished runs are skipped, the failed one is retried
    res = sweep.run_queue([dict(s) for s in q], train_fn=fake_train, eval_fn=fake_eval, write=False)
    assert [c for c in calls if c[0] == "train"] == [("train", "rs_bad")]
    # trained-but-not-evaluated run: only evaluated
    os.remove(os.path.join(sweep.run_dir("rs_t2"), "research.json"))
    calls.clear()
    sweep.run_queue([dict(s) for s in q if s["name"] == "rs_t2"], train_fn=fake_train, eval_fn=fake_eval,
                    write=False)
    assert calls == [("eval", "rs_t2")]
    r = json.load(open(os.path.join(sweep.run_dir("rs_t2"), "research.json")))
    assert r["status"] == "done" and r["bench"]["optimal"] == 0.5
    assert not os.path.exists(sweep.LOCK)


# ------------------------------------------------------------------ analysis
def test_regression_on_synthetic_data():
    x = np.log10([1e4, 3e4, 1e5, 3e5, 1e6])
    y = 0.1 + 0.15 * x
    r = sweep_analysis.regress(x, y)
    assert abs(r["slope"] - 0.15) < 1e-6 and abs(r["intercept"] - 0.1) < 1e-6 and r["r2"] == 1.0
    assert r["slope_ci95"][0] == pytest.approx(0.15, abs=1e-6) and not r["small_n"]
    r2 = sweep_analysis.regress([1, 2], [1, 3])
    assert r2["slope"] == pytest.approx(2) and r2["slope_ci95"] is None and r2["small_n"]
    rng = np.random.default_rng(1)
    x = np.linspace(4, 6.5, 12)
    y = 0.2 * x + rng.normal(0, 0.02, 12)
    r3 = sweep_analysis.regress(x, y)
    assert r3["slope_ci95"][0] < 0.2 < r3["slope_ci95"][1]


def test_analyse_and_findings_on_fake_runs():
    def run(name, teacher, b, c, opt, g, group="A"):
        return {"name": name, "group": group, "teacher": teacher, "blocks": b, "channels": c, "seed": 0,
                "params": sweep.param_count(b, c), "status": "done", "bench": {"optimal": opt, "keeps_nl": opt},
                "gauntlet": {"mean_score": g}, "ms_per_pos": b * c / 100, "teacher_agreement": 0.8}
    runs = [run("a1", "pons", 1, 16, .6, .1), run("a2", "pons", 2, 32, .7, .2), run("a3", "pons", 6, 64, .85, .4),
            run("b1", "heur_hand", 2, 32, .75, .3, "B"), run("b2", "mm_d4_hand", 2, 32, .78, .35, "B"),
            run("c1", "sup6x64", 1, 16, .65, .15, "C"), {"name": "q", "status": "queued"}]
    teachers_tab = [{"id": "pons", "kind": "solver", "bench_optimal": 1.0, "bench_keeps_nl": 1.0},
                    {"id": "heur_hand", "kind": "heuristic", "bench_optimal": .73, "bench_keeps_nl": .85,
                     "search_depth": 0},
                    {"id": "mm_d4_hand", "kind": "minimax", "bench_optimal": .87, "bench_keeps_nl": .93,
                     "search_depth": 4},
                    {"id": "sup6x64", "kind": "net", "bench_optimal": .90, "bench_keeps_nl": .95, "params": 458338}]
    an = sweep_analysis.analyse(runs, teachers_tab)
    assert an["n_done"] == 6
    assert an["size_regression"]["pons"]["optimal_vs_log10_params"]["n"] == 3
    assert an["distill_vs_direct"]["rows"][0]["from_6x64_retained"] == round(.65 / .90, 4)
    assert [p["name"] for p in an["pareto_params"]][0] == "c1"
    f = sweep_analysis.findings(runs, teachers_tab, an)
    ids = {x["id"] for x in f}
    assert "sweep-size-pons" in ids and "sweep-exceed-2x32" in ids and "sweep-distill-1x16" in ids
    for x in f:
        assert set(x) == {"id", "title", "claim", "evidence", "caveat"}
