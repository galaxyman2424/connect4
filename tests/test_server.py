import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.board import Board  # noqa: E402
from server.app import app  # noqa: E402


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


def empty():
    return [[0] * 7 for _ in range(6)]


MIDGAME = "4453526163"  # 10 plies, nobody has won


def post(client, board, level=3):
    return client.post("/api/move", json={"board": board, "level": level})


def test_index_and_static(client):
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/style.css").status_code == 200


def test_valid_move_empty_board(client):
    r = post(client, empty(), 4)
    assert r.status_code == 200
    j = r.get_json()
    for k in ("col", "score", "nodes", "ms", "depth", "random", "status"):
        assert k in j
    assert 0 <= j["col"] < 7
    assert j["status"]["over"] is False


def test_bot_takes_immediate_win_and_reports_status(client):
    g = empty()
    g[5][0] = g[5][1] = g[5][2] = 1
    g[5][5] = g[4][0] = g[4][1] = -1
    # +1: 3 pieces, -1: 3 pieces -> +1 to move, winning column is index 3
    j = post(client, g, 5).get_json()
    assert j["col"] == 3
    assert j["status"]["over"] is True and j["status"]["winner"] == 1
    assert len(j["status"]["cells"]) == 4


@pytest.mark.parametrize("bad", [
    None, [], [[0] * 7] * 5, [[0] * 6] * 6, "x",
    [[2] + [0] * 6] + [[0] * 7] * 5,
])
def test_invalid_shape_or_values(client, bad):
    r = post(client, bad)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_bad_piece_counts(client):
    g = empty()
    g[5][0] = g[5][1] = -1
    assert post(client, g).status_code == 400
    g = empty()
    g[5][0] = g[5][1] = g[5][2] = 1
    assert post(client, g).status_code == 400


def test_floating_piece(client):
    g = empty()
    g[3][3] = 1
    r = post(client, g)
    assert r.status_code == 400


def test_game_already_over(client):
    g = empty()
    g[5][0] = g[5][1] = g[5][2] = g[5][3] = 1
    g[4][0] = g[4][1] = g[4][2] = -1
    r = post(client, g)
    assert r.status_code == 400
    assert "over" in r.get_json()["error"]


def test_bad_level(client):
    for lvl in (0, 11, "a", None, True):
        assert client.post("/api/move", json={"board": empty(), "level": lvl}).status_code == 400


@pytest.mark.parametrize("level", range(1, 11))
def test_each_level_returns_legal_column(client, level):
    # column 3 filled almost to the top so legality is non-trivial
    b = Board.from_moves(MIDGAME)
    r = post(client, b.to_grid(), level)
    assert r.status_code == 200
    assert r.get_json()["col"] in b.legal_moves()


@pytest.mark.parametrize("level", [9, 10])
def test_high_level_is_fast(client, level):
    b = Board.from_moves(MIDGAME)
    t = time.time()
    r = post(client, b.to_grid(), level)
    dt = time.time() - t
    assert r.status_code == 200
    assert dt < 3.0, "level %d took %.2fs" % (level, dt)
