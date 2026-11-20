"""Research page endpoints (server/research_api.py), against a temporary results dir."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.app import app  # noqa: E402
import server.research_api as ra  # noqa: E402


def _write(d, name, obj):
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(obj, f)


ARENA = {"id": "arena", "title": "Arena test", "generated": "2026-01-01T00:00:00",
         "findings": [{"id": "arena-a", "title": "Finding A", "claim": "Claim A holds.", "evidence": "ev A", "caveat": "cav A"},
                      {"id": "arena-b", "title": "Finding B", "claim": "Claim B holds.", "evidence": "ev B", "caveat": "cav B"}],
         "best_nn": {"answer": {"needed_experiments": ["Run more openings."]}}}
SWEEP = {"id": "sweep", "title": "Sweep test", "generated": "2026-01-01T00:00:00",
         "status": {"state": "running", "counts": {"queued": 1, "running": 1, "done": 1, "failed": 0}, "total": 3},
         "runs": [{"name": "rs_A_pons_1x16", "group": "A", "teacher": "pons", "blocks": 1, "channels": 16, "params": 16994,
                   "status": "done", "bench": {"optimal": 0.787}, "gauntlet": {"mean_score": 0.33}}],
         "findings": [{"id": "sweep-size", "title": "Size helps", "claim": "Sweep claim.", "evidence": "e", "caveat": "c"}]}
ROSTER = {"id": "roster", "top15": [{"name": "pons_perfect", "label": "Pons solver", "family": "exact solver",
                                     "search_budget": "exact", "top15_elo": 1811.5, "top15_lo": 1775.1, "top15_hi": 1851.1,
                                     "part3_elo": 1788.5}], "dropped": []}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    res = tmp_path / "research"
    res.mkdir()
    docs = tmp_path / "docs"
    monkeypatch.setattr(ra, "RESULTS_DIR", str(res))
    monkeypatch.setattr(ra, "DOCS_DIR", str(docs))
    monkeypatch.setattr(ra, "ARENA_TOP15", str(tmp_path / "top15_summary.json"))
    ra._CACHE.clear()
    _write(str(res), "arena_analysis", ARENA)
    _write(str(res), "roster", ROSTER)
    app.config["TESTING"] = True
    return {"client": app.test_client(), "res": str(res), "docs": str(docs), "tmp": tmp_path}


def test_page_and_static(env):
    c = env["client"]
    r = c.get("/research")
    assert r.status_code == 200 and b"research.js" in r.data
    assert c.get("/research.js").status_code == 200
    assert c.get("/research.css").status_code == 200


def test_index_reports_existing_and_missing(env):
    d = env["client"].get("/api/research/index").get_json()["datasets"]
    assert d["arena_analysis"]["exists"] and d["arena_analysis"]["generated"] == "2026-01-01T00:00:00"
    assert d["arena_analysis"]["n_findings"] == 2
    assert d["roster"]["exists"]
    for k in ("sweep", "interp", "interp_channels", "arena_top15"):
        assert d[k]["exists"] is False
    _write(env["res"], "sweep", SWEEP)
    d = env["client"].get("/api/research/index").get_json()["datasets"]
    assert d["sweep"]["exists"] and d["sweep"]["status"]["counts"]["done"] == 1


def test_data_whitelist_and_404(env):
    c = env["client"]
    r = c.get("/api/research/data/arena_analysis")
    assert r.status_code == 200 and r.get_json()["title"] == "Arena test"
    r = c.get("/api/research/data/sweep")
    assert r.status_code == 404 and r.get_json()["error"]
    for bad in ("selection", "..%2Froster", "nope", "roster.json"):
        r = c.get("/api/research/data/" + bad)
        assert r.status_code == 404
    assert c.get("/api/research/data/../../server/app.py").status_code == 404


def test_data_mid_write_is_503(env):
    with open(os.path.join(env["res"], "sweep.json"), "w") as f:
        f.write('{"id": "sweep", "runs": [')
    assert env["client"].get("/api/research/data/sweep").status_code == 503


def test_arena_top15_served_from_its_own_path(env):
    with open(str(env["tmp"] / "top15_summary.json"), "w") as f:
        json.dump({"bots": ["a"], "elo": {"a": {"elo": 1409}}, "meta": {"created": "2026-01-02T00:00:00", "scheduled": 10, "recorded": 9}}, f)
    c = env["client"]
    assert c.get("/api/research/index").get_json()["datasets"]["arena_top15"]["generated"] == "2026-01-02T00:00:00"
    assert c.get("/api/research/data/arena_top15").get_json()["meta"]["recorded"] == 9


def test_selection_round_trip(env):
    c = env["client"]
    s = c.get("/api/research/selection").get_json()
    assert s["findings"] == {} and s["roster"] == {} and s["runs"] == {}
    body = {"findings": {"arena-a": {"include": True, "note": "use in intro"}}, "roster": {"pons_perfect": True}, "runs": {"rs_A_pons_1x16": True}}
    r = c.post("/api/research/selection", json=body)
    assert r.status_code == 200 and r.get_json()["updated"]
    s = c.get("/api/research/selection").get_json()
    assert s["findings"]["arena-a"] == {"include": True, "note": "use in intro"}
    assert s["roster"] == {"pons_perfect": True} and s["runs"] == {"rs_A_pons_1x16": True}
    # partial update keeps the other sections
    c.post("/api/research/selection", json={"runs": {}})
    s = c.get("/api/research/selection").get_json()
    assert s["runs"] == {} and s["roster"] == {"pons_perfect": True}
    assert os.path.isfile(os.path.join(env["res"], "selection.json"))
    assert not [f for f in os.listdir(env["res"]) if f.startswith(".tmp_")]


@pytest.mark.parametrize("body", [
    [1, 2],
    {"findings": []},
    {"findings": {"a": {"include": "yes"}}},
    {"findings": {"a": {"include": True, "note": 5}}},
    {"findings": {"a": {"include": True, "note": "x" * 6000}}},
    {"findings": {"a": {"include": True, "extra": 1}}},
    {"findings": {"bad key/..": {"include": True}}},
    {"roster": {"pons_perfect": 1}},
    {"runs": {"r": "true"}},
    {"other": {}},
])
def test_selection_validation(env, body):
    r = env["client"].post("/api/research/selection", json=body)
    assert r.status_code == 400
    assert env["client"].get("/api/research/selection").get_json()["findings"] == {}


def test_selection_too_large(env):
    big = {"findings": {"f%d" % i: {"include": True, "note": "x" * 900} for i in range(800)}}
    r = env["client"].post("/api/research/selection", json=big)
    assert r.status_code == 413


def test_outline_has_selected_and_omits_unselected(env):
    c = env["client"]
    _write(env["res"], "sweep", SWEEP)
    c.post("/api/research/selection", json={"findings": {"arena-a": {"include": True, "note": "my own remark"},
                                                         "arena-b": {"include": False, "note": "hidden"},
                                                         "sweep-size": {"include": True, "note": ""}},
                                            "roster": {"pons_perfect": True}, "runs": {"rs_A_pons_1x16": True}})
    md = ra.build_outline()
    assert "Finding A" in md and "Claim A holds." in md and "cav A" in md and "my own remark" in md
    assert "Finding B" not in md and "Claim B holds." not in md and "hidden" not in md
    assert "Size helps" in md and "Sweep claim." in md
    assert "Run more openings." in md                      # needed experiments
    assert "Pons solver" in md and "rs_A_pons_1x16" in md and "0.787" in md
    assert "Interpretability" in md and "not available yet" in md
    assert md.index("Finding A") < md.index("Size helps")  # arena section before sweep section


def test_outline_include_all_ignores_selection(env):
    _write(env["res"], "sweep", SWEEP)
    md = ra.build_outline(include_all=True)
    assert "Finding A" in md and "Finding B" in md and "Size helps" in md
    assert "pons_perfect" in md and "rs_A_pons_1x16" in md


def test_outline_routes_removed_and_cli_writes_docs(env):
    c = env["client"]
    assert c.get("/api/research/outline.md").status_code == 404
    assert c.post("/api/research/outline/save").status_code in (404, 405)
    from research.paper_outline import main
    main([])
    with open(os.path.join(env["docs"], "PAPER_OUTLINE.md"), encoding="utf-8") as f:
        text = f.read()
    assert "Finding A" in text and "Finding B" in text


def test_works_with_no_data_at_all(env):
    for n in ("arena_analysis", "roster"):
        os.remove(os.path.join(env["res"], n + ".json"))
    c = env["client"]
    assert all(not v["exists"] for v in c.get("/api/research/index").get_json()["datasets"].values())
    md = ra.build_outline()
    assert "Research questions" in md and "not available yet" in md


def test_corrupt_selection_file_is_ignored(env):
    with open(os.path.join(env["res"], "selection.json"), "w") as f:
        f.write("{not json")
    assert env["client"].get("/api/research/selection").get_json()["findings"] == {}
