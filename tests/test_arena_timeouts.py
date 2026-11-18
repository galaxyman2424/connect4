"""The arena can never hang: misbehaving fake bots (sleep forever, ignore signals and
busy-loop, kill their own process, too slow for the game cap) must cost exactly
one forfeit each, every scheduled game must give exactly one games.csv row, and
the tournament must finish quickly.  Uses the test bots of arena/registry.py."""

import csv
import json
import os
import time

import pytest

from arena import registry, tournament

MODES = ("hang", "crash", "ok", "sleep", "busy", "slow", "threads")


@pytest.fixture(autouse=True)
def _test_bots(monkeypatch):
    monkeypatch.setenv("C4_ARENA_TEST", "1")          # spawned workers register the test bots too
    for m in MODES:
        monkeypatch.setitem(registry.BOTS, "test_" + m,
                            {"label": "test bot (%s)" % m, "group": "baseline", "params": {},
                             "make": (lambda mm: (lambda: registry._TestBot(mm)))(m)})


def _run(tmp_path, bots, *extra):
    t0 = time.time()
    rc = tournament.main(["--bots", ",".join(bots), "--openings", "1", "--workers", "4", "--boot", "5",
                          "--name", "t", "--out-root", str(tmp_path)] + list(extra))
    secs = time.time() - t0
    with open(tmp_path / "t" / "games.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rc, rows, secs


def _keys(rows):
    return [(r["a"], r["b"], r["opening_id"], r["a_first"]) for r in rows]


def test_misbehaving_bots_forfeit_and_tournament_finishes(tmp_path):
    bots = ["test_ok", "test_sleep", "test_hang", "test_busy", "test_crash"]
    rc, rows, secs = _run(tmp_path, bots, "--move-timeout", "1", "--kill-grace", "1", "--game-timeout", "60")
    assert rc == 0
    assert len(rows) == 10 * 2 and len(set(_keys(rows))) == 20       # 10 pairings x 2 colours, no duplicates
    assert secs < 120, secs
    for r in rows:
        bad = [x for x in (r["a"], r["b"]) if x != "test_ok"]
        assert r["forfeit"], r                                         # every game has a misbehaving bot
        loser = r["forfeit"].split(":")[0]
        assert loser in bad, r
        assert r["winner"] == (r["b"] if loser == r["a"] else r["a"])
        if "test_ok" in (r["a"], r["b"]):
            reason = r["forfeit"]
            if loser == "test_sleep":
                assert "move timeout" in reason and "killed" not in reason   # the in-worker alarm
            elif loser in ("test_hang", "test_busy"):
                assert "move timeout" in reason and "killed" in reason       # the parent's hard kill
            elif loser == "test_crash":
                assert "died" in reason
    s = json.load(open(tmp_path / "t" / "summary.json"))
    assert s["meta"]["scheduled"] == s["meta"]["recorded"] == 20 and s["meta"]["missing"] == 0
    assert s["per_bot"]["test_ok"]["wins"] == 8


def test_game_timeout(tmp_path):
    rc, rows, secs = _run(tmp_path, ["test_ok", "test_slow"], "--move-timeout", "5", "--game-timeout", "1",
                          "--kill-grace", "1")
    assert rc == 0 and len(rows) == 2 and secs < 60
    for r in rows:
        assert r["forfeit"].startswith("test_slow: game timeout"), r


def test_workers_are_single_threaded(tmp_path):
    before = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    rc, rows, _ = _run(tmp_path, ["test_ok", "test_threads"])
    assert rc == 0 and len(rows) == 2
    assert all(r["forfeit"] == "" for r in rows), [r["forfeit"] for r in rows]
    assert {k: os.environ.get(k) for k in before} == before          # the caller's environment is untouched


def test_resume_drops_partial_line_and_completes(tmp_path):
    bots = ["test_ok", "random"]
    rc, rows, _ = _run(tmp_path, bots, "--openings", "2")
    assert rc == 0 and len(rows) == 4
    path = tmp_path / "t" / "games.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]) + lines[-1][:20], encoding="utf-8")   # lose a game, half-written line
    rc, rows, _ = _run(tmp_path, bots, "--openings", "2", "--resume")
    assert rc == 0 and len(rows) == 4 and len(set(_keys(rows))) == 4


def test_check_complete_reports_missing_and_duplicates():
    tasks = tournament.schedule(["x", "y"], ["44", "43"], 1)
    rows = [{"a": t[0], "b": t[1], "opening_id": str(t[2]), "a_first": str(int(t[4]))} for t in tasks]
    assert tournament.check_complete(tasks, rows) == ([], [])
    missing, dupes = tournament.check_complete(tasks, rows[1:] + rows[2:3])
    assert missing == [tournament.game_key(tasks[0])] and dupes == [tournament.game_key(tasks[2])]


def test_rosters_and_anchor():
    top = registry.ROSTERS["top15"]
    assert top["bots"] == ["pons_perfect", "nn_sup_6x64_mcts200", "nn_az_6x64_mcts200", "nn_ppo_from_sup_policy",
                           "bruneton_az", "nn_ppo_6x64_mcts200", "nn_distill_az_2x32_policy",
                           "nn_distill_ens_2x32_policy", "alfo_mcts", "ours_minimax_d8", "evo_evo_d2_d4",
                           "ours_minimax_d4", "galli_minimax", "nn_ppo_6x64_policy", "nn_dqn_6x64_policy"]
    assert list(registry.resolve(top["bots"])) == top["bots"]          # aliases resolve to the same names
    assert tournament.parse_anchor(top["elo_anchor"]) == ("ours_minimax_d4", 1409.0)
    assert tournament.parse_anchor("kaggle_negamax") == ("kaggle_negamax", 1000.0)
    assert len(registry.ROSTERS["part3"]["bots"]) == 20
    for dropped in ("plkmo_az", "kaggle_negamax", "easyai_negamax"):
        assert dropped in registry.BOTS
