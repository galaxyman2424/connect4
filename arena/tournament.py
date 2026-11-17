"""Round-robin tournament between Connect 4 bots.

    python -m arena.tournament --list
    python -m arena.tournament                                   # default line-up, 20 openings
    python -m arena.tournament --roster top15 --name phase4      # preset line-up (see --list)
    python -m arena.tournament --bots pons_perfect,galli_minimax,nn:runs/az_6x64/model.npz:mcts:200
    python -m arena.tournament --name quick --openings 4 --workers 2
    python -m arena.tournament --resume                          # finish an interrupted run

Protocol (same as experiments/tournament.py, so results are comparable):
* --openings random openings of 2-4 plies (distinct up to mirroring, fixed
  seed). Every pair of bots plays every opening TWICE, swapping who moves
  first, so neither bot profits from a lucky opening or from moving first.
* Before each game Python's `random`, numpy's and torch's global RNGs are
  seeded from (seed, pairing, opening, colour) -> reproducible.
* An exception or an illegal move from a bot = it forfeits that game
  (recorded in the `forfeit` column with the reason).

Time limits (a hang is impossible by construction):
* Every worker is a separate process limited to --threads (default 1) BLAS /
  OpenMP / torch threads.  The limit is put in the environment BEFORE the
  worker imports numpy or torch (OpenBLAS reads it only once, at load time).
  Without it each worker started 16 OpenBLAS threads, and 14 workers x 16
  spinning threads on 16 cores made the network bots 10-100x slower.
* --move-timeout (default 10 s): a bot that thinks longer on one move
  forfeits.  Enforced twice: SIGALRM inside the worker (clean, worker kept),
  and the parent kills the worker's whole process group if the move is not
  done --kill-grace seconds later (works even when the bot is stuck in C
  code or ignores signals).
* --game-timeout (default 300 s): hard wall-clock cap per game; the bot that
  used more thinking time in that game forfeits.
* --build-timeout (default 120 s): loading a bot (networks, solver) happens
  before the clock starts and has its own limit.
* A worker that dies (segfault, out of memory, os._exit) costs the bot that
  was thinking that game; the worker is replaced.
* The parent hands out games one at a time over a private pipe per worker,
  so it always knows which game each worker holds: every scheduled game
  yields exactly one row, and at the end the run checks scheduled ==
  recorded (loud error and exit code 3 otherwise).
* Games are appended to games.csv (flushed) as they finish; --resume
  continues a stopped run (a half-written last line is dropped).
  --summary-only writes summary.json from the games finished so far.

Outputs in experiments/results/arena/<name>/:
  games.csv     one row per game (bots, opening, who started, result, moves, ms/move, forfeit)
  summary.json  bots (source, licence, algorithm, settings), score matrix,
                W/D/L per pairing, overall score, Bradley-Terry Elo with
                95% bootstrap CIs, time per move, forfeits.  The Lab page
                (/lab) reads this file.
"""

import os

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def thread_env(n):
    return {v: str(int(n)) for v in _THREAD_VARS}


# Run as a script (`python -m arena.tournament`) or re-imported as __mp_main__ by a
# spawned worker: limit threads BEFORE numpy is imported below.  (When this module is
# merely imported, e.g. by tests, the caller's environment is left alone; the workers
# still get the limit, see spawn() in run_supervised.)
if __name__ in ("__main__", "__mp_main__"):
    os.environ.update(thread_env(os.environ.get("C4_ARENA_THREADS", "1")))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing.connection import wait as mp_wait  # noqa: E402

import numpy as np  # noqa: E402

from engine.board import Board  # noqa: E402

from . import registry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "experiments", "results", "arena")
FIELDS = ["a", "b", "opening_id", "opening", "a_first", "winner", "score_a", "plies", "moves",
          "a_ms", "b_ms", "a_max_ms", "b_max_ms", "forfeit", "seed"]

_BOTS = {}
_ENTRIES = {}
_TP_LIMIT = []          # keeps the threadpoolctl limiter alive


def _init(entries):
    for name, e in entries.items():
        if "make" not in e:                      # sent from the parent process: rebuild here
            full = registry.resolve([e["spec"]])[name]
            full["params"] = e["params"]
            e = full
        _ENTRIES[name] = e


def limit_threads(n=None):
    """Runtime thread limits on top of the environment variables (which are what
    really matters for OpenBLAS): threadpoolctl for every BLAS/OpenMP library loaded
    so far, and torch if it is loaded.  Called at worker start and after every bot
    build (a bot may load torch / MKL)."""
    n = int(n or os.environ.get("C4_ARENA_THREADS", "1") or 1)
    os.environ.update(thread_env(n))
    try:
        from threadpoolctl import threadpool_limits
        _TP_LIMIT[:] = [threadpool_limits(limits=n)]
    except Exception:  # noqa: BLE001  (threadpoolctl is optional)
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.get_num_threads() != n:
                torch.set_num_threads(n)
        except Exception:  # noqa: BLE001
            pass


def _bot(name):
    if name not in _BOTS:
        _BOTS[name] = registry.build(_ENTRIES[name])
        limit_threads()
    return _BOTS[name]


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:  # noqa: BLE001
            pass


class MoveTimeout(BaseException):
    """BaseException, so a bot's own `except Exception:` cannot swallow it."""


def _alarm(signum, frame):
    raise MoveTimeout()


# heartbeat array (shared memory: still readable after the worker died), all doubles
HB_IDX, HB_SIDE, HB_PHASE, HB_PLY, HB_T, HB_GAME_T0, HB_A_MS, HB_B_MS = range(8)
PH_IDLE, PH_BUILD, PH_THINK, PH_BETWEEN = 0, 1, 2, 3


def play(task, hb=None, move_timeout=0.0, game_timeout=0.0):
    """Play one game in this process.  hb: optional heartbeat array (see above) that
    the parent watches.  Timeouts in seconds, 0 = none."""
    a, b, oid, opening, a_first, seed = task
    use_alarm = (move_timeout > 0 or game_timeout > 0) and hasattr(signal, "setitimer")
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm)

    def hbset(d):
        if hb is not None:
            for k, v in d.items():
                hb[k] = v

    side_of = {a: 0, b: 1}
    times = {a: [], b: []}
    total = {a: 0.0, b: 0.0}
    board = Board.from_moves(opening)

    def result(winner, forfeit="", extra_ms=None):
        score_a = 0.5 if winner == "draw" else (1.0 if winner == a else 0.0)
        avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else 0.0  # noqa: E731
        mx = {n: max(times[n]) if times[n] else 0.0 for n in (a, b)}
        if extra_ms:                                  # the move that timed out / failed
            n, ms = extra_ms
            mx[n] = max(mx[n], ms)
        return {"a": a, "b": b, "opening_id": oid, "opening": opening, "a_first": int(a_first),
                "winner": winner, "score_a": score_a, "plies": board.moves_played,
                "moves": board.to_move_string(), "a_ms": avg(times[a]), "b_ms": avg(times[b]),
                "a_max_ms": round(mx[a], 1), "b_max_ms": round(mx[b], 1),
                "forfeit": forfeit, "seed": seed}

    # 1. build both bots (not on the game clock; the parent enforces --build-timeout)
    for name in (a, b):
        hbset({HB_SIDE: side_of[name], HB_PHASE: PH_BUILD, HB_T: time.time()})
        try:
            _bot(name)
        except Exception as e:  # noqa: BLE001
            _BOTS.pop(name, None)
            return result(b if name == a else a,
                          "%s: build error: %s: %s" % (name, type(e).__name__, str(e)[:200]))

    # 2. the game
    _seed_all(seed)
    who = {1: a if a_first else b, -1: b if a_first else a}
    g0 = time.time()
    hbset({HB_GAME_T0: g0, HB_A_MS: 0.0, HB_B_MS: 0.0})
    while True:
        name = who[board.to_move]
        other = b if name == a else a
        limit, why = move_timeout, "move"
        if game_timeout > 0:
            left = game_timeout - (time.time() - g0)
            if left <= 0:
                loser = name if total[name] >= total[other] else other
                return result(b if loser == a else a, "%s: game timeout: game exceeded %gs (it used %.1fs)"
                              % (loser, game_timeout, total[loser] / 1000))
            if limit <= 0 or left < limit:
                limit, why = left, "game"
        hbset({HB_SIDE: side_of[name], HB_PHASE: PH_THINK, HB_PLY: board.moves_played, HB_T: time.time()})
        t0 = time.perf_counter()
        err = None
        try:
            try:
                if use_alarm and limit > 0:
                    signal.setitimer(signal.ITIMER_REAL, limit)
                col = _bot(name).choose(board.copy())
            finally:
                if use_alarm:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        except MoveTimeout:
            err = "timeout"
        except Exception as e:  # noqa: BLE001
            err = "%s: %s" % (type(e).__name__, str(e)[:200])
        ms = (time.perf_counter() - t0) * 1000
        if err is None and limit > 0 and ms > limit * 1000 + 250:
            err = "timeout"                            # the alarm was swallowed somewhere
        if err is not None:
            _BOTS.pop(name, None)                      # rebuild that bot for its next game (bad state)
            if err != "timeout":
                return result(other, "%s: %s" % (name, err), (name, ms))
            total[name] += ms
            if why == "move":
                loser, reason = name, "%s: move timeout: no move within %gs" % (name, move_timeout)
            else:
                loser = name if total[name] >= total[other] else other
                reason = "%s: game timeout: game exceeded %gs (it used %.1fs)" % (
                    loser, game_timeout, total[loser] / 1000)
            return result(b if loser == a else a, reason, (name, ms))
        times[name].append(ms)
        total[name] += ms
        hbset({HB_PHASE: PH_BETWEEN, (HB_A_MS if name == a else HB_B_MS): total[name]})
        if not isinstance(col, (int, np.integer)) or int(col) not in board.legal_moves():
            return result(other, "%s: illegal move %r" % (name, col))
        board.drop(int(col))
        w = board.check_win()
        if w:
            return result(who[w])
        if board.is_draw():
            return result("draw")


# ------------------------------------------------------- supervised workers

def _worker_main(wid, entries, conn, hb, cfg):
    """One worker: receives (idx, task) over its own pipe, answers ("game", idx, row)."""
    try:
        os.setpgid(0, 0)          # own process group: the parent can kill it with its children (solver)
    except Exception:  # noqa: BLE001
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)       # Ctrl-C is handled by the parent
    limit_threads(cfg["threads"])
    _init(entries)
    conn.send(("ready", wid))
    while True:
        try:
            item = conn.recv()
        except (EOFError, OSError):
            return
        if item is None:
            return
        idx, task = item
        hb[HB_SIDE], hb[HB_PHASE], hb[HB_T], hb[HB_GAME_T0] = -1, PH_BETWEEN, time.time(), 0
        hb[HB_IDX] = idx
        try:
            g = play(task, hb, cfg["move_timeout"], cfg["game_timeout"])
        except BaseException as e:  # noqa: BLE001  - never lose a game silently
            g = _forfeit_game(task, task[0], "worker error: %s: %s" % (type(e).__name__, str(e)[:200]))
        hb[HB_PHASE] = PH_IDLE
        conn.send(("game", idx, g))
        hb[HB_IDX] = -1


def _forfeit_game(task, loser, reason, plies=None, loser_ms=0.0):
    a, b, oid, opening, a_first, seed = task
    winner = b if loser == a else a
    return {"a": a, "b": b, "opening_id": oid, "opening": opening, "a_first": int(a_first),
            "winner": winner, "score_a": 1.0 if winner == a else 0.0,
            "plies": plies if plies is not None else len(opening), "moves": opening,
            "a_ms": 0.0, "b_ms": 0.0,
            "a_max_ms": round(loser_ms, 1) if loser == a else 0.0,
            "b_max_ms": round(loser_ms, 1) if loser == b else 0.0,
            "forfeit": "%s: %s" % (loser, reason), "seed": seed}


def _first_mover(task):
    a, b, oid, opening, a_first, seed = task
    p1_to_move = len(opening) % 2 == 0
    return a if p1_to_move == bool(a_first) else b


def _kill(p):
    """Kill a worker and everything it started (it runs in its own process group)."""
    try:
        if p.pid and os.getpgid(p.pid) == p.pid:
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    try:
        p.kill()
    except Exception:  # noqa: BLE001
        pass
    p.join(timeout=10)


class _W:
    __slots__ = ("wid", "proc", "conn", "hb", "ready", "t_spawn", "idx", "t_assign", "dead")


def run_supervised(tasks, workers, entries, move_timeout=10.0, game_timeout=300.0, build_timeout=120.0,
                   kill_grace=5.0, threads=1, startup_timeout=180.0, log=print):
    """Play every task in worker processes; yields exactly one finished game (dict)
    per task, in finishing order.  Never hangs: see the module docstring."""
    ctx = mp.get_context("spawn")
    ent = entries_picklable(entries)
    cfg = {"move_timeout": float(move_timeout), "game_timeout": float(game_timeout), "threads": int(threads)}
    pending = list(range(len(tasks)))[::-1]          # pop() from the end = original order
    finished = set()
    retries = {}
    ws = {}
    state = {"next": 0, "startup_fail": 0}

    def spawn():
        wid = state["next"]
        state["next"] += 1
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        hb = ctx.Array("d", [-1, -1, PH_IDLE, 0, time.time(), 0, 0, 0], lock=False)
        p = ctx.Process(target=_worker_main, args=(wid, ent, child_conn, hb, cfg), daemon=True)
        keys = list(_THREAD_VARS) + ["C4_ARENA_THREADS"]
        saved = {k: os.environ.get(k) for k in keys}
        os.environ.update(thread_env(threads))       # inherited by the new interpreter, i.e. set before numpy loads
        os.environ["C4_ARENA_THREADS"] = str(int(threads))
        try:
            p.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        child_conn.close()
        w = _W()
        w.wid, w.proc, w.conn, w.hb, w.ready, w.t_spawn, w.idx, w.t_assign, w.dead = \
            wid, p, parent_conn, hb, False, time.time(), None, 0.0, False
        ws[wid] = w

    def retire(w, reason=None, loser=None, plies=None, loser_ms=0.0, requeue=False):
        """Kill worker w; its game (if any) becomes a forfeit row (returned) or is re-queued."""
        _kill(w.proc)
        try:
            w.conn.close()
        except Exception:  # noqa: BLE001
            pass
        ws.pop(w.wid, None)
        if w.idx is None or w.idx in finished:
            return None
        idx, task = w.idx, tasks[w.idx]
        if requeue:
            retries[idx] = retries.get(idx, 0) + 1
            if retries[idx] <= 2:
                log("  [supervisor] %s vs %s, opening %d: %s - game re-queued" % (task[0], task[1], task[2], reason))
                pending.append(idx)
                return None
            loser = _first_mover(task)
            reason = "%s (3 times), side unknown" % reason
        finished.add(idx)
        log("  [supervisor] %s vs %s, opening %d: %s - %s forfeits" % (task[0], task[1], task[2], reason, loser))
        return _forfeit_game(task, loser, reason, plies, loser_ms)

    def drain(w):
        """Read every message waiting from w; returns finished games.  Sets w.dead on EOF."""
        out = []
        while True:
            try:
                if not w.conn.poll():
                    break
                msg = w.conn.recv()
            except (EOFError, OSError):
                w.dead = True
                break
            if msg[0] == "ready":
                w.ready = True
                state["startup_fail"] = 0
            elif msg[0] == "game":
                _, idx, g = msg
                if w.idx == idx:
                    w.idx = None
                if idx not in finished:
                    finished.add(idx)
                    out.append(g)
        return out

    try:
        while len(finished) < len(tasks):
            busy = sum(w.idx is not None for w in ws.values())
            while pending and len(ws) < workers and len(ws) < len(pending) + busy:
                spawn()
            for w in list(ws.values()):                  # hand out games to idle, ready workers
                if w.ready and w.idx is None and pending:
                    idx = pending.pop()
                    try:
                        w.conn.send((idx, tasks[idx]))
                        w.idx, w.t_assign = idx, time.time()
                    except (OSError, EOFError, ValueError):
                        pending.append(idx)
                        w.dead = True
            if not ws and not pending and len(finished) < len(tasks):
                raise RuntimeError("supervisor lost track of %d games" % (len(tasks) - len(finished)))
            objs = {}
            for w in ws.values():
                objs[w.conn] = w
                objs[w.proc.sentinel] = w
            ready = mp_wait(list(objs), timeout=0.25) if objs else []
            for w in {objs[o].wid: objs[o] for o in ready}.values():
                for g in drain(w):
                    yield g
            # deaths (after draining, so a game sent just before exiting is not lost)
            for w in list(ws.values()):
                if not w.dead and w.proc.is_alive():
                    continue
                for g in drain(w):
                    yield g
                w.proc.join(timeout=1)
                code = w.proc.exitcode
                if not w.ready:
                    state["startup_fail"] += 1
                    if state["startup_fail"] >= 3:
                        raise RuntimeError("workers die during start-up (exit code %s); "
                                           "check: python -m arena.tournament --list" % code)
                hb = w.hb
                started = w.idx is not None and int(hb[HB_IDX]) == w.idx
                side, phase = (int(hb[HB_SIDE]), int(hb[HB_PHASE])) if started else (-1, PH_IDLE)
                if side in (0, 1) and phase in (PH_BUILD, PH_THINK):
                    what = "while loading the bot" if phase == PH_BUILD else "while thinking"
                    g = retire(w, "worker process died %s (exit code %s)" % (what, code), tasks[w.idx][side],
                               int(hb[HB_PLY]) if phase == PH_THINK else None, (time.time() - hb[HB_T]) * 1000)
                else:
                    g = retire(w, "worker process died outside a move (exit code %s)" % code, requeue=True)
                if g is not None:
                    yield g
            # deadlines, enforced here in the parent: work even if the worker is stuck in C code
            now = time.time()
            for w in list(ws.values()):
                hb = w.hb
                if not w.ready:
                    if now - w.t_spawn > startup_timeout:
                        state["startup_fail"] += 1
                        retire(w)
                        if state["startup_fail"] >= 3:
                            raise RuntimeError("workers do not start within %gs" % startup_timeout)
                    continue
                if w.idx is None:
                    continue
                task = tasks[w.idx]
                if int(hb[HB_IDX]) != w.idx:             # game not picked up yet
                    if now - w.t_assign > startup_timeout:
                        g = retire(w, "worker unresponsive", requeue=True)
                        if g is not None:
                            yield g
                    continue
                phase, side = int(hb[HB_PHASE]), int(hb[HB_SIDE])
                el = now - hb[HB_T]
                g = None
                if phase == PH_BUILD and side in (0, 1) and el > build_timeout:
                    g = retire(w, "build timeout: bot not loaded within %gs, worker killed" % build_timeout,
                               task[side])
                elif phase == PH_THINK and side in (0, 1) and move_timeout > 0 and el > move_timeout + kill_grace:
                    g = retire(w, "move timeout: no move within %gs (worker killed after %gs)"
                               % (move_timeout, move_timeout + kill_grace), task[side], int(hb[HB_PLY]), el * 1000)
                elif game_timeout > 0 and hb[HB_GAME_T0] > 0 and now - hb[HB_GAME_T0] > game_timeout + kill_grace:
                    tot = [hb[HB_A_MS], hb[HB_B_MS]]
                    if phase == PH_THINK and side in (0, 1):
                        tot[side] += el * 1000
                    ls = 0 if tot[0] >= tot[1] else 1
                    g = retire(w, "game timeout: game exceeded %gs (worker killed; it used %.1fs)"
                               % (game_timeout, tot[ls] / 1000), task[ls], int(hb[HB_PLY]))
                elif phase == PH_BETWEEN and el > max(60.0, move_timeout + kill_grace) + 60:
                    # the harness itself (not a bot) stuck between moves: should never happen
                    g = retire(w, "worker stuck between moves", requeue=True)
                if g is not None:
                    yield g
    finally:
        for w in list(ws.values()):
            try:
                w.conn.send(None)
            except Exception:  # noqa: BLE001
                pass
        t_end = time.time() + 5
        for w in list(ws.values()):
            w.proc.join(timeout=max(0.1, t_end - time.time()))
            if w.proc.is_alive():
                _kill(w.proc)
            try:
                w.conn.close()
            except Exception:  # noqa: BLE001
                pass
        ws.clear()


def random_openings(n, seed):
    from nn.bench import random_openings as ro
    return ro(n, seed)


def parse_anchor(s, default_value=1000.0):
    """'name' or 'name:value' -> (name, value)."""
    name, _, val = (s or "").partition(":")
    return name, float(val) if val else float(default_value)


def summarize(games, entries, anchor, boot, anchor_value=1000.0):
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
        anchor_value = 1000.0
    if len(rated) >= 2:
        eg = [{"a": g["a"], "b": g["b"], "score_a": float(g["score_a"]),
               "cluster": "%s|%s|%s" % (min(g["a"], g["b"]), max(g["a"], g["b"]), g["opening_id"])}
              for g in games if g["a"] in rated and g["b"] in rated]
        res, _ = stats.elo_with_bootstrap(eg, rated, anchor, float(anchor_value), reps=boot, seed=0)
        elo = {n: {"elo": round(v[0], 1), "lo": round(v[1], 1), "hi": round(v[2], 1)} for n, v in res.items()}
    p1 = [float(g["score_a"]) if int(g["a_first"]) else 1 - float(g["score_a"]) for g in games]
    return {"bots": names, "matrix": matrix, "wdl": wdl, "per_bot": per_bot, "elo": elo,
            "elo_anchor": anchor if elo else None,
            "elo_anchor_value": float(anchor_value) if elo else None,
            "unrated": [n for n in names if n not in rated],
            "first_player_score": round(sum(p1) / len(p1), 4) if p1 else None,
            "games": len(games), "decisive": sum(1 for g in games if float(g["score_a"]) != 0.5),
            "forfeits": [g["forfeit"] for g in games if g["forfeit"]][:50]}


# ------------------------------------------------------------- games.csv

def game_key(row):
    """(a, b, opening_id, a_first) of a task tuple or of a games.csv row."""
    if isinstance(row, tuple):
        return (row[0], row[1], int(row[2]), bool(row[4]))
    return (row["a"], row["b"], int(row["opening_id"]), bool(int(row["a_first"])))


def read_games(path):
    """Valid rows of games.csv (a half-written or corrupt line is skipped)."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if any(r.get(k) is None for k in FIELDS) or None in r:
                    continue
                game_key(r)
                float(r["score_a"])
                float(r["b_max_ms"])
                rows.append(r)
            except (ValueError, TypeError, KeyError):
                continue
    return rows


def rewrite_games(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def check_complete(tasks, rows):
    """-> (missing keys, duplicated keys) of the scheduled tasks among rows."""
    want = {game_key(t) for t in tasks}
    seen = {}
    for r in rows:
        k = game_key(r)
        if k in want:
            seen[k] = seen.get(k, 0) + 1
    missing = sorted(want - set(seen))
    dupes = sorted(k for k, c in seen.items() if c > 1)
    return missing, dupes


def schedule(order, openings, seed):
    tasks = []
    for i, x in enumerate(order):
        for y in order[i + 1:]:
            for oi, op in enumerate(openings):
                for af in (True, False):
                    s = random.Random("%d|%s|%s|%d|%d" % (seed, x, y, oi, af)).randrange(2 ** 31)
                    tasks.append((x, y, oi, op, af, s))
    return tasks


def _slowness(name):
    slow = {"plkmo_az": 5, "bruneton_az": 4, "ours_level10": 4, "alfo_mcts": 3, "pons_perfect": 2}
    if name in slow:
        return slow[name]
    return 3 if (name.startswith("nn_") and "_mcts" in name) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", choices=sorted(registry.ROSTERS),
                    help="preset line-up (see --list); --bots / --add-nn add to it")
    ap.add_argument("--bots", default=None,
                    help="comma list of bot names (see --list); add nets as nn:<path.npz>[:mcts:200|:policy|:minimax:4]"
                         " (default: the roster, else %s)" % ",".join(registry.DEFAULT_LINEUP))
    ap.add_argument("--add-nn", action="append", default=[], help="shortcut: also add this nn:/evo:/ntuple: spec")
    ap.add_argument("--set", action="append", default=[], metavar="BOT.PARAM=VALUE",
                    help="change a bot setting, e.g. --set plkmo_az.reads=200")
    ap.add_argument("--openings", type=int, default=20, help="openings per pairing (x2 colours)")
    ap.add_argument("--seed", type=int, default=4150)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("C4_ARENA_THREADS", "1") or 1),
                    help="BLAS/OpenMP/torch threads per worker (default 1; keep workers x threads <= cores)")
    ap.add_argument("--name", default="main", help="results go to experiments/results/arena/<name>/")
    ap.add_argument("--out-root", default=OUT_ROOT, help=argparse.SUPPRESS)
    ap.add_argument("--elo-anchor", "--anchor", dest="elo_anchor", default=None, metavar="NAME[:VALUE]",
                    help="bot whose Elo is fixed (default: the roster's anchor, else kaggle_negamax:1000)")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for Elo CIs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--move-timeout", type=float, default=10.0,
                    help="seconds a bot may think about one move before it forfeits the game (0 = no limit)")
    ap.add_argument("--game-timeout", type=float, default=300.0,
                    help="wall-clock cap per game; the bot that used more time forfeits (0 = no cap)")
    ap.add_argument("--build-timeout", type=float, default=120.0,
                    help="seconds a worker may take to load one bot (not on the game clock)")
    ap.add_argument("--kill-grace", type=float, default=5.0,
                    help="the parent kills a worker this many seconds after a move/game limit if the bot "
                         "has not stopped by itself (stuck in C code)")
    ap.add_argument("--watchdog", type=float, default=None, help=argparse.SUPPRESS)   # old flag: absolute kill time
    ap.add_argument("--summary-only", action="store_true",
                    help="play nothing: write summary.json from the games already in games.csv")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

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
        print("\nrosters (--roster NAME):")
        for rn, r in registry.ROSTERS.items():
            print("  %-6s %2d bots, Elo anchor %s:\n         %s" % (rn, len(r["bots"]), r.get("elo_anchor", "-"),
                                                                ", ".join(r["bots"])))
        return 0

    if a.watchdog is not None and a.watchdog > 0 and a.move_timeout > 0:
        a.kill_grace = max(1.0, a.watchdog - a.move_timeout)
    roster = registry.ROSTERS.get(a.roster) if a.roster else None
    base = list(roster["bots"]) if roster else []
    if a.bots is not None:
        base += [x.strip() for x in a.bots.split(",") if x.strip() and x.strip() not in base]
    elif not roster:
        base = list(registry.DEFAULT_LINEUP)
    names = base + [x for x in a.add_nn if x not in base]
    anchor_spec = a.elo_anchor or (roster or {}).get("elo_anchor") or "kaggle_negamax:1000"
    a.anchor, a.anchor_value = parse_anchor(anchor_spec)
    overrides = dict(s.split("=", 1) for s in a.set)
    entries = registry.resolve(names, overrides)
    out_dir = os.path.join(a.out_root, a.name)
    os.makedirs(out_dir, exist_ok=True)
    games_path = os.path.join(out_dir, "games.csv")
    openings = random_openings(a.openings, a.seed)
    all_tasks = schedule(list(entries), openings, a.seed)

    if a.summary_only:
        if not os.path.isfile(games_path):
            sys.exit("no %s yet" % games_path)
        done = [r for r in read_games(games_path) if r["a"] in entries and r["b"] in entries]
        print("summary from %d finished games (pairings with missing games just count fewer)" % len(done))
        write_summary(done, entries, a, out_dir, 0.0, all_tasks)
        return 0

    # check every bot can be built here before starting
    from .adapters import BotUnavailable
    for n, e in entries.items():
        try:
            registry.build(e)
        except BotUnavailable as ex:
            sys.exit("cannot run %s: %s" % (n, ex))

    tasks = list(all_tasks)
    if a.resume and os.path.isfile(games_path):
        seen, clean = set(), []
        for r in read_games(games_path):     # drop duplicates and broken lines, keep other bots' games
            k = game_key(r)
            if k not in seen:
                seen.add(k)
                clean.append(r)
        rewrite_games(games_path, clean)
        tasks = [t for t in tasks if game_key(t) not in seen]
        print("resuming: %d games done, %d to go" % (len(all_tasks) - len(tasks), len(tasks)))
    elif os.path.isfile(games_path):
        os.replace(games_path, games_path + ".old")
    tasks.sort(key=lambda t: -(_slowness(t[0]) + _slowness(t[1])))     # slow pairings first
    print("%d bots, %d pairings, %d games (%d openings x 2 colours), %d workers x %d threads, "
          "move limit %gs, game limit %gs, Elo anchor %s=%g -> %s" % (
              len(entries), len(entries) * (len(entries) - 1) // 2, len(tasks), len(openings), a.workers,
              a.threads, a.move_timeout, a.game_timeout, a.anchor, a.anchor_value, out_dir), flush=True)

    t0 = time.time()
    new_file = not os.path.isfile(games_path)
    with open(games_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
            f.flush()
        it = run_supervised(tasks, max(1, a.workers), entries, a.move_timeout, a.game_timeout,
                            a.build_timeout, a.kill_grace, a.threads, log=lambda s: print(s, flush=True))
        step = max(1, len(tasks) // 50)
        for k, g in enumerate(it, 1):
            w.writerow(g)
            f.flush()
            if k % step == 0 or k == len(tasks):
                el = time.time() - t0
                print("  %d/%d games  %.0fs elapsed, ~%.0f min left" % (
                    k, len(tasks), el, el / k * (len(tasks) - k) / 60), flush=True)

    done = [r for r in read_games(games_path) if r["a"] in entries and r["b"] in entries]
    ok = write_summary(done, entries, a, out_dir, time.time() - t0, all_tasks)
    return 0 if ok else 3


def write_summary(done, entries, a, out_dir, seconds, all_tasks):
    missing, dupes = check_complete(all_tasks, done)
    want = {game_key(t) for t in all_tasks}
    seen, rows = set(), []
    for r in done:                            # statistics: one row per scheduled game
        k = game_key(r)
        if k in want and k not in seen:
            seen.add(k)
            rows.append(r)
    summ = summarize(rows, entries, a.anchor, a.boot, a.anchor_value)
    summ["meta"] = {"name": a.name, "openings": random_openings(a.openings, a.seed), "seed": a.seed,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "seconds": round(seconds, 1), "workers": a.workers, "threads_per_worker": a.threads,
                    "move_timeout": a.move_timeout, "game_timeout": a.game_timeout,
                    "roster": a.roster, "elo_anchor": "%s:%g" % (a.anchor, a.anchor_value),
                    "scheduled": len(all_tasks), "recorded": len(rows), "missing": len(missing),
                    "duplicates": len(dupes), "command": " ".join(sys.argv), "machine": machine()}
    summ["bot_info"] = {n: {k: v for k, v in e.items() if k not in ("make", "spec")} for n, e in entries.items()}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    print_table(summ)
    print("wrote", os.path.join(out_dir, "summary.json"))
    print("scheduled %d games, recorded %d" % (len(all_tasks), len(rows)), flush=True)
    if missing or dupes:
        bar = "!" * 72
        print("\n%s\nERROR: games.csv does not match the schedule: %d missing, %d duplicated.\n"
              "  missing (first 10): %s\n  duplicated (first 10): %s\n"
              "  Run again with --resume to play the missing games.\n%s"
              % (bar, len(missing), len(dupes), missing[:10], dupes[:10], bar), file=sys.stderr, flush=True)
        return False
    return True


def entries_picklable(entries):
    """Lambdas cannot be sent to spawned workers: send names + params, rebuild there."""
    return {n: {"spec": e["spec"], "params": e["params"]} for n, e in entries.items()}


def machine():
    import platform
    return {"platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count()}


def print_table(s):
    print("\n%-28s %6s %6s %5s %5s %5s %9s %9s %8s" % ("bot", "Elo", "score", "W", "D", "L", "ms/move",
                                                     "max ms", "forfeit"))
    order = sorted(s["bots"], key=lambda n: -(s["per_bot"][n]["score"] or 0))
    for n in order:
        p = s["per_bot"][n]
        e = s["elo"].get(n, {}).get("elo", "-")
        print("%-28s %6s %6.3f %5d %5d %5d %9.1f %9.1f %8d" % (n, e, p["score"] or 0, p["wins"], p["draws"],
                                                               p["losses"], p["avg_ms"] or 0, p["max_ms"] or 0,
                                                               p["forfeits"]))
    print("first player scored %.3f overall" % (s["first_player_score"] or 0))


if __name__ == "__main__":
    sys.exit(main())
