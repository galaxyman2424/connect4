"""Evaluation bar endpoint (server/eval_api.py)."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from server.app import app  # noqa: E402
import server.eval_api as ea  # noqa: E402


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_sources(client):
    j = client.get("/api/eval/sources").get_json()
    assert isinstance(j["nets"], list) and isinstance(j["solver"], bool)


@pytest.mark.parametrize("moves", ["8", "4a", "1111111", "1" * 43, 5])
def test_bad_moves(client, moves):
    assert client.post("/api/eval", json={"moves": moves}).status_code == 400


def test_game_over_positions(client):
    j = client.post("/api/eval", json={"moves": "1212121"}).get_json()
    assert j["bar"] == 1.0 and j["result"] == "red"
    j = client.post("/api/eval", json={"moves": "2121212"}).get_json()   # 1212121 shifted: yellow can't win this way
    assert j["ply"] == 7


@pytest.mark.skipif(not ea._solver_binary(), reason="Pons solver not built")
def test_solver_labels(client):
    j = client.post("/api/eval", json={"moves": "", "source": "solver"}).get_json()
    assert j["source"] == "solver" and j["label"] == "R21" and j["bar"] > 0.5      # first player wins on move 21
    j = client.post("/api/eval", json={"moves": "152536", "source": "solver"}).get_json()
    assert j["label"] == "R1" and len(j["columns"]) == 7                           # red wins now in column 4
    j = client.post("/api/eval", json={"moves": "4453", "source": "solver"}).get_json()
    assert j["label"].startswith("Y") and j["bar"] < 0.5


@pytest.mark.skipif(not ea.net_runs(), reason="no trained net")
def test_net_is_reported_for_red(client):
    run = ea.net_runs()[0]
    a = client.post("/api/eval", json={"moves": "152536", "source": "nn:" + run}).get_json()
    b = client.post("/api/eval", json={"moves": "1525364", "source": "nn:" + run}).get_json()
    assert a["value"] > 0.5                       # red to move and wins now
    assert abs(sum(a["wdl"]) - 1) < 1e-3
    assert b["result"] == "red"
