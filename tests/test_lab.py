"""Lab page endpoints, the neural-net opponent and the arena plumbing."""
import json
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.app import app  # noqa: E402

RUN = "_pytest_run"


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture()
def fake_run():
    """A tiny run folder with a random numpy model and two metric lines."""
    import numpy as np
    from nn.agents import random_init_numpy_net
    d = os.path.join(ROOT, "runs", RUN)
    os.makedirs(d, exist_ok=True)
    net = random_init_numpy_net()
    arrays = dict(net.w)
    arrays["config_json"] = np.array(json.dumps(net.config))
    arrays["meta_json"] = np.array("{}")
    np.savez(os.path.join(d, "model.npz"), **arrays)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"phase": "selfplay", "created": "2000-01-01T00:00:00", "model": net.config}, f)
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"gen": 1, "bench_optimal": 0.4, "elo": 900}) + "\n")
        f.write(json.dumps({"gen": 2, "bench_optimal": 0.5}) + "\n{broken line\n")
    yield RUN
    shutil.rmtree(d, ignore_errors=True)


def test_lab_page_and_assets(client):
    assert client.get("/lab").status_code == 200
    assert client.get("/lab.js").status_code == 200
    assert client.get("/lab.css").status_code == 200


def test_reference(client):
    j = client.get("/api/lab/reference").get_json()
    assert "d1" in j["accuracy"] and "all" in j["accuracy"]["d1"]
    assert "d1" in j["elo"]


def test_runs_and_run(client, fake_run):
    runs = client.get("/api/lab/runs").get_json()
    r = [x for x in runs if x["name"] == fake_run][0]
    assert r["best_bench_optimal"] == 0.5 and r["last_elo"] == 900 and r["has_model"]
    d = client.get("/api/lab/run/" + fake_run).get_json()
    assert len(d["metrics"]) == 2                       # broken line skipped
    assert client.get("/api/lab/run/../server").status_code == 404
    assert client.get("/api/lab/run/nope_nope").status_code == 404


def test_nn_move(client, fake_run):
    models = client.get("/api/nn/models").get_json()
    assert any(m["run"] == fake_run for m in models)
    board = [[0] * 7 for _ in range(6)]
    r = client.post("/api/move", json={"board": board, "engine": "nn", "run": fake_run, "sims": 10})
    j = r.get_json()
    assert r.status_code == 200 and 0 <= j["col"] <= 6 and j["engine"] == "nn"
    r = client.post("/api/move", json={"board": board, "engine": "nn", "run": fake_run, "sims": 0})
    assert r.status_code == 200
    assert client.post("/api/move", json={"board": board, "engine": "nn", "run": "missing"}).status_code == 400
    assert client.post("/api/move", json={"board": board, "engine": "nn", "run": fake_run, "sims": -1}).status_code == 400
    assert client.post("/api/move", json={"board": board, "engine": "bogus"}).status_code == 400


def test_arena_summary_math():
    from arena.tournament import summarize
    ents = {"a": {"label": "A"}, "b": {"label": "B"}, "c": {"label": "C"}}
    games = []
    for i in range(6):
        games.append({"a": "a", "b": "b", "opening_id": i, "a_first": i % 2, "score_a": 1.0 if i < 4 else 0.0,
                      "a_ms": 1, "b_ms": 2, "a_max_ms": 1, "b_max_ms": 2, "forfeit": ""})
        games.append({"a": "b", "b": "c", "opening_id": i, "a_first": i % 2, "score_a": 0.5,
                      "a_ms": 1, "b_ms": 2, "a_max_ms": 1, "b_max_ms": 2, "forfeit": ""})
        games.append({"a": "a", "b": "c", "opening_id": i, "a_first": i % 2, "score_a": 1.0 if i < 5 else 0.5,
                      "a_ms": 1, "b_ms": 2, "a_max_ms": 1, "b_max_ms": 2, "forfeit": ""})
    s = summarize(games, ents, "b", boot=20)
    i, j = s["bots"].index("a"), s["bots"].index("b")
    assert abs(s["matrix"][i][j] - 4 / 6) < 1e-3
    assert s["per_bot"]["a"]["wins"] == 9
    assert s["elo"]["b"]["elo"] == 1000.0 and s["elo"]["a"]["elo"] > 1000


def test_arena_adapters_win_in_one():
    """Every adapter that is available must see an immediate win (checks the board translation)."""
    from arena import registry
    from arena.adapters import BotUnavailable
    from engine.board import Board
    b = Board.from_moves("445566")           # P1 to move, wins with column 3 or 7
    fast = {"kaggle_negamax": {}, "easyai_negamax": {}, "galli_minimax": {"galli_minimax.depth": "2"},
            "pons_perfect": {}, "ours_minimax_d4": {}}
    tested = 0
    for name, over in fast.items():
        try:
            bot = registry.build(registry.resolve([name], over)[name])
        except BotUnavailable:
            continue
        assert bot.choose(b.copy()) in (2, 6), name
        tested += 1
    assert tested >= 1
