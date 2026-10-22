"""Round-robin tournament between Connect 4 bots.

    python -m arena.tournament --list
    python -m arena.tournament                                   # default line-up, 20 openings
    python -m arena.tournament --bots pons_perfect,galli_minimax,nn:runs/az_6x64/model.npz:mcts:200
    python -m arena.tournament --name quick --openings 4 --workers 2
    python -m arena.tournament --resume                          # finish an interrupted run

Protocol (same as experiments/tournament.py, so results are comparable):
* --openings random openings of 2-4 plies (distinct up to mirroring, fixed
  seed). Every pair of bots plays every opening TWICE, swapping who moves
  first, so neither bot profits from a lucky opening or from moving first.
  Openings matter because most bots are deterministic: without them, the
  same two bots would replay the identical game.
* Before each game Python's `random`, numpy's and torch's global RNGs are
  seeded from (seed, pairing, opening, colour) -> reproducible.
* An exception or an illegal move from a bot = it forfeits that game
  (recorded with the reason).
* A bot that takes longer than --move-timeout (default 300 s) on one move
  forfeits.  A watchdog in the parent also kills any worker whose bot has not
  moved for --watchdog seconds (default move-timeout + 60), or whose process
  died, and records a forfeit - so one stuck game can never hang the run.
* Games are appended to games.csv as they finish, so a stopped run can be
  resumed with --resume.  --summary-only writes summary.json from the games
  finished so far without playing any more.

Outputs in experiments/results/arena/<name>/:
  games.csv     one row per game (bots, opening, who started, result, moves, ms/move)
  summary.json  bots (source, licence, algorithm, settings), score matrix,
                W/D/L per pairing, overall score, Bradley-Terry Elo with
                95% bootstrap CIs, time per move, forfeits.  The Lab page
                (/lab) reads this file.
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import signal
import random
import sys
import time

import numpy as np

from engine.board import Board

from . import registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "experiments", "results", "arena")
FIELDS = ["a", "b", "opening_id", "opening", "a_first", "winner", "score_a", "plies", "moves",
          "a_ms", "b_ms", "a_max_ms", "b_max_ms", "forfeit", "seed"]

_BOTS = {}
_ENTRIES = {}


def _init(entries):
    for name, e in entries.items():
        if "make" not in e:                      # sent from the parent process: rebuild here
            full = registry.resolve([e["spec"]])[name]
            full["params"] = e["params"]
            e = full
        _ENTRIES[name] = e
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _bot(name):
    if name not in _BOTS:
        _BOTS[name] = registry.build(_ENTRIES[name])
    return _BOTS[name]


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


class MoveTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise MoveTimeout("no move within %ss" % os.environ.get("C4_MOVE_TIMEOUT"))


def play(task, beat=None):
    """Play one game.  beat(name) is called before every move (the supervisor's
    watchdog uses it to know which bot is thinking and since when)."""
    a, b, oid, opening, a_first, seed = task
    limit = float(os.environ.get("C4_MOVE_TIMEOUT", "0") or 0)
    use_alarm = limit > 0 and hasattr(signal, "setitimer")
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm)
    _seed_all(seed)
    board = Board.from_moves(opening)
    color = {a: 1 if a_first else -1, b: -1 if a_first else 1}
    who = {1: a if a_first else b, -1: b if a_first else a}
    times = {a: [], b: []}
    forfeit = ""
    winner = None
    while True:
        name = who[board.to_move]
        if beat is not None:
            beat(name, board.moves_played)
        t0 = time.perf_counter()
        try:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, limit)
            try:
                col = _bot(name).choose(board.copy())
            finally:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception as e:  # noqa: BLE001  (MoveTimeout included)
            forfeit = "%s: %s: %s" % (name, type(e).__name__, str(e)[:200])
            winner = b if name == a else a
            _BOTS.pop(name, None)          # rebuild that bot for its next game (it may be in a bad state)
            break
        times[name].append((time.perf_counter() - t0) * 1000)
        if not isinstance(col, (int, np.integer)) or int(col) not in board.legal_moves():
            forfeit = "%s: illegal move %r" % (name, col)
            winner = b if name == a else a
            break
        board.drop(int(col))
        w = board.check_win()
        if w:
            winner = who[w]
            break
        if board.is_draw():
            winner = "draw"
            break
    score_a = 0.5 if winner == "draw" else (1.0 if winner == a else 0.0)
    avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0.0  # noqa: E731
    return {"a": a, "b": b, "opening_id": oid, "opening": opening, "a_first": int(a_first),
            "winner": winner, "score_a": score_a, "plies": board.moves_played,
            "moves": board.to_move_string(), "a_ms": avg(times[a]), "b_ms": avg(times[b]),
            "a_max_ms": round(max(times[a]), 1) if times[a] else 0.0,
            "b_max_ms": round(max(times[b]), 1) if times[b] else 0.0,
            "forfeit": forfeit, "seed": seed, "_color_a": color[a]}


# ------------------------------------------------------- supervised workers
# multiprocessing.Pool waits forever if a worker dies (out of memory, segfault)
# or is stuck inside C code, where the per-move SIGALRM cannot interrupt it.
# Instead each worker reports every move to the parent; a worker that has not
# moved for --watchdog seconds (or has died) is killed and replaced, and the
# bot that was thinking forfeits that game.

def _sup_worker(wid, entries, task_q, res_q, hb):
    """hb = shared array [task idx (-1 idle), side thinking (0 = a, 1 = b), ply, time]:
    written directly to shared memory, so it is never lost if this process dies."""
    _init(entries)
    while True:
        item = task_q.get()
        if item is None:
            return
        idx, task = item
        hb[0], hb[1], hb[2], hb[3] = idx, -1, len(task[3]), time.time()

        def beat(name, ply, _a=task[0]):
            hb[1], hb[2], hb[3] = (0 if name == _a else 1), ply, time.time()
        try:
            g = play(task, beat=beat)
        except Exception as e:  # noqa: BLE001  - never lose a game silently
            g = _forfeit_game(task, task[0], "worker error: %s: %s" % (type(e).__name__, str(e)[:200]))
        res_q.put((idx, g))
        hb[0] = -1


def _forfeit_game(task, loser, reason, plies=None):
    a, b, oid, opening, a_first, seed = task
    winner = b if loser == a else a
    return {"a": a, "b": b, "opening_id": oid, "opening": opening, "a_first": int(a_first),
            "winner": winner, "score_a": 1.0 if winner == a else 0.0,
            "plies": plies if plies is not None else len(opening), "moves": opening,
            "a_ms": 0.0, "b_ms": 0.0, "a_max_ms": 0.0, "b_max_ms": 0.0,
            "forfeit": "%s: %s" % (loser, reason), "seed": seed}


def _first_mover(task):
    a, b, oid, opening, a_first, seed = task
    p1_to_move = len(opening) % 2 == 0
    return a if p1_to_move == bool(a_first) else b


def run_supervised(tasks, workers, entries, watchdog):
    """Yield finished games (dicts) like Pool.imap_unordered, but never hang."""
    import queue as _queue
    ctx = mp.get_context("spawn")
    task_q, res_q = ctx.Queue(), ctx.Queue()
    for i, t in enumerate(tasks):
        task_q.put((i, t))
    ent = entries_picklable(entries)
    procs, hbs, last_idx = {}, {}, {}

    def spawn(wid):
        hb = ctx.Array("d", [-1, -1, 0, time.time()], lock=False)
        p = ctx.Process(target=_sup_worker, args=(wid, ent, task_q, res_q, hb), daemon=True)
        p.start()
        procs[wid], hbs[wid] = p, hb

    n0 = min(workers, len(tasks))
    for w in range(n0):
        spawn(w)
        task_q.put(None)                     # one stop signal per worker, after all tasks
    next_wid = n0
    finished = set()
    while len(finished) < len(tasks):
        try:
            idx, g = res_q.get(timeout=5)
            if idx not in finished:
                finished.add(idx)
                yield g
            continue
        except _queue.Empty:
            pass
        now = time.time()
        for wid in list(procs):
            p, hb = procs[wid], hbs[wid]
            cur = int(hb[0])
            if cur >= 0:
                last_idx[wid] = cur
            stuck = cur >= 0 and watchdog > 0 and now - hb[3] > watchdog
            if p.is_alive() and not stuck:
                continue
            if not p.is_alive() and cur < 0 and last_idx.get(wid, -1) in finished | {-1}:
                procs.pop(wid)               # finished normally (got its stop signal)
                continue
            if p.is_alive():
                p.kill()
            p.join(timeout=10)
            procs.pop(wid)
            idx = cur if cur >= 0 else last_idx.get(wid, -1)
            if idx >= 0 and idx not in finished:
                task = tasks[idx]
                side = int(hb[1]) if cur >= 0 else -1
                loser = task[0] if side == 0 else task[1] if side == 1 else _first_mover(task)
                why = ("watchdog: no move within %ds, worker killed" % watchdog) if stuck else \
                      ("worker process died (exit code %s)" % p.exitcode)
                print("  [watchdog] %s vs %s, opening %d: %s - %s forfeits" % (task[0], task[1], task[2], why, loser),
                      flush=True)
                finished.add(idx)
                yield _forfeit_game(task, loser, why, int(hb[2]) if cur >= 0 else None)
            if len(finished) < len(tasks):   # replacement worker (needs its own stop signal)
                spawn(next_wid)
                task_q.put(None)
                next_wid += 1
    for p in procs.values():
        p.join(timeout=5)
        if p.is_alive():
            p.kill()


def random_openings(n, seed):
    from nn.bench import random_openings as ro
    return ro(n, seed)


def summarize(games, entries, anchor, boot):
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import stats
    names = list(entries)
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    S = np.zeros((k, k))
    N = np.zeros((k, k))
    wdl = {}
    for g in games:
        i, j = idx[g["a"]], idx[g["b"]]
        sa = float(g["score_a"])
        S[i, j] += sa
        S[j, i] += 1 - sa
        N[i, j] += 1
        N[j, i] += 1
        for x, y, s in ((g["a"], g["b"], sa), (g["b"], g["a"], 1 - sa)):
            r = wdl.setdefault(x, {}).setdefault(y, [0, 0, 0])
            r[0 if s == 1 else (1 if s == 0.5 else 2)] += 1
    matrix = [[(round(S[i, j] / N[i, j], 4) if N[i, j] else None) for j in range(k)] for i in range(k)]
    per_bot = {}
    for n in names:
        i = idx[n]
        ms = [float(g["a_ms"]) for g in games if g["a"] == n] + [float(g["b_ms"]) for g in games if g["b"] == n]
        mx = [float(g["a_max_ms"]) for g in games if g["a"] == n] + [float(g["b_max_ms"]) for g in games if g["b"] == n]
        played = int(N[i].sum())
        first = [g for g in games if (g["a"] == n and int(g["a_first"])) or (g["b"] == n and not int(g["a_first"]))]
        sf = sum(float(g["score_a"]) if g["a"] == n else 1 - float(g["score_a"]) for g in first)
        per_bot[n] = {"games": played, "score": round(S[i].sum() / played, 4) if played else None,
                      "wins": sum(v[0] for v in wdl.get(n, {}).values()),
                      "draws": sum(v[1] for v in wdl.get(n, {}).values()),
                      "losses": sum(v[2] for v in wdl.get(n, {}).values()),
                      "score_moving_first": round(sf / len(first), 4) if first else None,
                      "avg_ms": round(sum(ms) / len(ms), 1) if ms else None,
                      "max_ms": round(max(mx), 1) if mx else None,
                      "forfeits": sum(1 for g in games if g["forfeit"].startswith(n + ":"))}
    # Elo (Bradley-Terry); bots that lost or won EVERY game have no finite rating -> excluded, listed.
    rated = [n for n in names if per_bot[n]["games"] and 0 < per_bot[n]["score"] < 1]
    elo = {}
    if anchor not in rated and rated:
        anchor = rated[len(rated) // 2]
    if len(rated) >= 2:
        eg = [{"a": g["a"], "b": g["b"], "score_a": float(g["score_a"]),
               "cluster": "%s|%s|%s" % (min(g["a"], g["b"]), max(g["a"], g["b"]), g["opening_id"])}
              for g in games if g["a"] in rated and g["b"] in rated]
        res, _ = stats.elo_with_bootstrap(eg, rated, anchor, 1000.0, reps=boot, seed=0)
        elo = {n: {"elo": round(v[0], 1), "lo": round(v[1], 1), "hi": round(v[2], 1)} for n, v in res.items()}
    p1 = [float(g["score_a"]) if int(g["a_first"]) else 1 - float(g["score_a"]) for g in games]
    return {"bots": names, "matrix": matrix, "wdl": wdl, "per_bot": per_bot, "elo": elo,
            "elo_anchor": anchor if elo else None,
            "unrated": [n for n in names if n not in rated],
            "first_player_score": round(sum(p1) / len(p1), 4) if p1 else None,
            "games": len(games), "decisive": sum(1 for g in games if float(g["score_a"]) != 0.5),
            "forfeits": [g["forfeit"] for g in games if g["forfeit"]][:50]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bots", default=",".join(registry.DEFAULT_LINEUP),
                    help="comma list of bot names (see --list); add nets as nn:<path.npz>[:mcts:200|:policy|:minimax:4]")
    ap.add_argument("--add-nn", action="append", default=[], help="shortcut: also add this nn: spec")
    ap.add_argument("--set", action="append", default=[], metavar="BOT.PARAM=VALUE",
                    help="change a bot setting, e.g. --set plkmo_az.reads=200")
    ap.add_argument("--openings", type=int, default=20, help="openings per pairing (x2 colours)")
    ap.add_argument("--seed", type=int, default=4150)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--name", default="main", help="results go to experiments/results/arena/<name>/")
    ap.add_argument("--anchor", default="kaggle_negamax", help="bot fixed at Elo 1000")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for Elo CIs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--move-timeout", type=float, default=300.0,
                    help="seconds a bot may think about one move before it forfeits the game (0 = no limit)")
    ap.add_argument("--watchdog", type=float, default=None,
                    help="kill a worker whose bot has not moved for this many seconds, even if it is stuck in "
                         "C code or the process died; that bot forfeits (default: move-timeout + 60; 0 = off)")
    ap.add_argument("--summary-only", action="store_true",
                    help="play nothing: write summary.json from the games already in games.csv")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        from .adapters import BotUnavailable
        for n, e in registry.BOTS.items():
            try:
                registry.build(e)
                st = "ready"
            except BotUnavailable as ex:
                st = "NOT READY: %s" % ex
            except Exception as ex:  # noqa: BLE001
                st = "ERROR: %s: %s" % (type(ex).__name__, ex)
            print("%-16s %-42s %s" % (n, e["label"], st))
        print("\nplus any trained network: nn:runs/<run>/model.npz[:mcts:200|:policy|:minimax:4]")
        print("plus rl/ players: evo:runs/<run>/weights.json[:depth]  ntuple:runs/<run>/ntuple.npz[:depth]")
        return

    os.environ["C4_MOVE_TIMEOUT"] = str(a.move_timeout)        # read by play() in every worker
    if a.watchdog is None:
        a.watchdog = (a.move_timeout + 60) if a.move_timeout > 0 else 600
    names = [x.strip() for x in a.bots.split(",") if x.strip()] + a.add_nn
    overrides = dict(s.split("=", 1) for s in a.set)
    entries = registry.resolve(names, overrides)
    out_dir = os.path.join(OUT_ROOT, a.name)
    os.makedirs(out_dir, exist_ok=True)
    games_path = os.path.join(out_dir, "games.csv")

    if a.summary_only:
        if not os.path.isfile(games_path):
            sys.exit("no %s yet" % games_path)
        with open(games_path, newline="", encoding="utf-8") as f:
            done = [r for r in csv.DictReader(f) if r["a"] in entries and r["b"] in entries]
        print("summary from %d finished games (pairings with missing games just count fewer)" % len(done))
        write_summary(done, entries, a, out_dir, 0.0)
        return

    # check every bot can be built here before starting
    from .adapters import BotUnavailable
    for n, e in entries.items():
        try:
            registry.build(e)
        except BotUnavailable as ex:
            sys.exit("cannot run %s: %s" % (n, ex))

    openings = random_openings(a.openings, a.seed)
    order = list(entries)
    tasks = []
    for i, x in enumerate(order):
        for y in order[i + 1:]:
            for oi, op in enumerate(openings):
                for af in (True, False):
                    seed = random.Random("%d|%s|%s|%d|%d" % (a.seed, x, y, oi, af)).randrange(2 ** 31)
                    tasks.append((x, y, oi, op, af, seed))
    done = []
    if a.resume and os.path.isfile(games_path):
        with open(games_path, newline="", encoding="utf-8") as f:
            done = [r for r in csv.DictReader(f) if r["a"] in entries and r["b"] in entries]
        have = {(r["a"], r["b"], int(r["opening_id"]), bool(int(r["a_first"]))) for r in done}
        tasks = [t for t in tasks if (t[0], t[1], t[2], t[4]) not in have]
        print("resuming: %d games done, %d to go" % (len(done), len(tasks)))
    elif os.path.isfile(games_path):
        os.replace(games_path, games_path + ".old")
    # slow bots first so the pool stays busy at the end
    slow = {"plkmo_az": 5, "bruneton_az": 4, "ours_level10": 4, "pons_perfect": 1, "alfo_mcts": 3}
    tasks.sort(key=lambda t: -(slow.get(t[0], 1) + slow.get(t[1], 1)))
    print("%d bots, %d pairings, %d games (%d openings x 2 colours), %d workers -> %s" % (
        len(order), len(order) * (len(order) - 1) // 2, len(tasks), len(openings), a.workers, out_dir))

    t0 = time.time()
    new_file = not os.path.isfile(games_path)
    with open(games_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        if a.workers <= 1 and a.watchdog <= 0:
            _init(entries)
            it = map(play, tasks)
        else:
            it = run_supervised(tasks, max(1, a.workers), entries, a.watchdog)
        for k, g in enumerate(it, 1):
            w.writerow(g)
            f.flush()
            done.append({kk: str(v) for kk, v in g.items()})
            if k % max(1, len(tasks) // 50) == 0 or k == len(tasks):
                el = time.time() - t0
                print("  %d/%d games  %.0fs elapsed, ~%.0f min left" % (
                    k, len(tasks), el, el / k * (len(tasks) - k) / 60), flush=True)

    write_summary(done, entries, a, out_dir, time.time() - t0)


def write_summary(done, entries, a, out_dir, seconds):
    openings = random_openings(a.openings, a.seed)
    summ = summarize(done, entries, a.anchor, a.boot)
    summ["meta"] = {"name": a.name, "openings": openings, "seed": a.seed, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "seconds": round(seconds, 1), "workers": a.workers, "command": " ".join(sys.argv),
                    "machine": machine()}
    summ["bot_info"] = {n: {k: v for k, v in e.items() if k not in ("make", "spec")} for n, e in entries.items()}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print_table(summ)
    print("wrote", os.path.join(out_dir, "summary.json"))


def entries_picklable(entries):
    """Lambdas cannot be sent to spawned workers: send names + params, rebuild there."""
    return {n: {"spec": e["spec"], "params": e["params"]} for n, e in entries.items()}


def machine():
    import platform
    return {"platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count()}


def print_table(s):
    print("\n%-28s %6s %6s %5s %5s %5s %9s %8s" % ("bot", "Elo", "score", "W", "D", "L", "ms/move", "forfeit"))
    order = sorted(s["bots"], key=lambda n: -(s["per_bot"][n]["score"] or 0))
    for n in order:
        p = s["per_bot"][n]
        e = s["elo"].get(n, {}).get("elo", "-")
        print("%-28s %6s %6.3f %5d %5d %5d %9.1f %8d" % (n, e, p["score"] or 0, p["wins"], p["draws"],
                                                         p["losses"], p["avg_ms"] or 0, p["forfeits"]))
    print("first player scored %.3f overall" % (s["first_player_score"] or 0))


if __name__ == "__main__":
    main()
