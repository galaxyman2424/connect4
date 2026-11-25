"""The bot line-up: name -> description + how to build it.

Names are what you pass to `python -m arena.tournament --bots ...`.
Settings in brackets can be changed with --set, e.g.
`--set plkmo_az.reads=200 --set galli_minimax.depth=4`.
"""

import os

from . import adapters as A

GH = "https://github.com/"

BOTS = {
    # ------------------------------------------------------------ open source
    "pons_perfect": {
        "label": "Pons solver (perfect play)", "group": "open source",
        "author": "Pascal Pons", "url": GH + "PascalPons/connect4", "commit": "d6ba50d",
        "license": "AGPL-3.0", "lang": "C++",
        "algorithm": "Exact solver: negamax + alpha-beta on bitboards, transposition table, "
                     "iterative null-window search, opening book. Never makes a mistake.",
        "params": {}, "make": lambda: A.PonsSolver()},
    "bruneton_az": {
        "label": "AlphaZero (Bruneton et al.)", "group": "open source",
        "author": "J.-P. Bruneton, A. Douin, V. Reverdy",
        "url": GH + "jpbruneton/Alpha-Zero-algorithm-for-Connect-4-game", "commit": "6cd76be",
        "license": "BSD-3-Clause", "lang": "Python / PyTorch",
        "algorithm": "AlphaZero: ResNet (1 block x 256 filters, 1.2 M params) policy+value net "
                     "trained by self-play, PUCT MCTS. Pretrained weights from the repo.",
        "params": {"sims": 200}, "make": lambda sims=200: A.BrunetonAlphaZero(int(sims))},
    "plkmo_az": {
        "label": "AlphaZero (plkmo / AI Singapore)", "group": "open source",
        "author": "plkmo (Wee Tee Soh)", "url": GH + "plkmo/AlphaZero_Connect4", "commit": "01ff24a",
        "license": "Apache-2.0", "lang": "Python / PyTorch",
        "algorithm": "AlphaZero: ResNet (19 blocks x 128 channels) policy+value net trained by "
                     "self-play ('iter8' weights from the repo), PUCT MCTS with root noise.",
        "params": {"reads": 777}, "make": lambda reads=777: A.PlkmoAlphaZero(int(reads))},
    "alfo_mcts": {
        "label": "UCT MCTS, random roll-outs (Alfo5123)", "group": "open source",
        "author": "Alfredo de la Fuente", "url": GH + "Alfo5123/Connect4", "commit": "d371203",
        "license": "MIT", "lang": "Python 2 (run under 3)",
        "algorithm": "Classic Monte Carlo Tree Search (UCT) with uniformly random play-outs; "
                     "no heuristic, no neural network.",
        "params": {"iterations": 3000}, "make": lambda iterations=3000: A.AlfoMCTS(int(iterations))},
    "galli_minimax": {
        "label": "Minimax (Keith Galli tutorial)", "group": "open source",
        "author": "Keith Galli", "url": GH + "KeithGalli/Connect4-Python", "commit": "503c0b4",
        "license": "MIT", "lang": "Python / numpy",
        "algorithm": "Minimax + alpha-beta, window heuristic (center 3, three 5, two 2, "
                     "opponent three -4), no move ordering or transposition table.",
        "params": {"depth": 5}, "make": lambda depth=5: A.GalliMinimax(int(depth))},
    "easyai_negamax": {
        "label": "easyAI Negamax", "group": "open source",
        "author": "Zulko (Valentin Zulkower)", "url": GH + "Zulko/easyAI", "commit": "43e9462",
        "license": "MIT", "lang": "Python / numpy",
        "algorithm": "Negamax + alpha-beta from the easyAI library on its ConnectFour example; "
                     "scores only wins/losses (no positional heuristic).",
        "params": {"depth": 5}, "make": lambda depth=5: A.EasyAINegamax(int(depth))},
    "kaggle_negamax": {
        "label": "Kaggle ConnectX negamax", "group": "open source",
        "author": "Kaggle", "url": GH + "Kaggle/kaggle-environments", "commit": "eb8b5ef",
        "license": "Apache-2.0", "lang": "Python",
        "algorithm": "The built-in 'negamax' opponent of Kaggle's ConnectX competition: plain "
                     "negamax, depth 4, no pruning, leaf score = own stones adjacent to the cell.",
        "params": {}, "make": lambda: A.KaggleNegamax()},
    # ------------------------------------------------------------------ ours
    "ours_minimax_d4": {
        "label": "Our minimax, depth 4 (slider level 5)", "group": "ours",
        "author": "Connor Thelen", "url": "", "license": "", "lang": "Python",
        "algorithm": "Negamax + alpha-beta, transposition table, move ordering, iterative "
                     "deepening, bit-parallel window heuristic.",
        "params": {"depth": 4}, "make": lambda depth=4: _ours_minimax(int(depth))},
    "ours_minimax_d8": {
        "label": "Our minimax, depth 8 (slider level 9)", "group": "ours",
        "author": "Connor Thelen", "url": "", "license": "", "lang": "Python",
        "algorithm": "Same search as above, 8 plies deep.",
        "params": {"depth": 8}, "make": lambda depth=8: _ours_minimax(int(depth))},
    "ours_level10": {
        "label": "Our minimax, level 10 (~2 s/move)", "group": "ours",
        "author": "Connor Thelen", "url": "", "license": "", "lang": "Python",
        "algorithm": "Iterative deepening up to depth 20 with a 1.9 s time limit (the website's "
                     "strongest setting).",
        "params": {}, "make": lambda: A.Ours(_level(10), "ours_level10")},
    "random": {
        "label": "Random mover", "group": "baseline",
        "author": "", "url": "", "license": "", "lang": "Python",
        "algorithm": "Uniformly random legal move (sanity check).",
        "params": {}, "make": lambda: A.Ours(_random(), "random")},
}

DEFAULT_LINEUP = ["pons_perfect", "bruneton_az", "plkmo_az", "alfo_mcts", "galli_minimax",
                  "easyai_negamax", "kaggle_negamax", "ours_minimax_d4", "ours_minimax_d8"]

# Short names for our trained players (= the names the tournament gives them), so a
# line-up can be written as names only: `--bots nn_sup_6x64_mcts200,...`.
ALIASES = {
    "nn_az_6x64_mcts200": "nn:runs/az_6x64/model.npz:mcts:200",
    "nn_sup_6x64_mcts200": "nn:runs/sup_6x64/best.npz:mcts:200",
    "nn_ppo_6x64_policy": "nn:runs/ppo_6x64/model.npz:policy",
    "nn_ppo_6x64_mcts200": "nn:runs/ppo_6x64/model.npz:mcts:200",
    "nn_ppo_from_sup_policy": "nn:runs/ppo_from_sup/model.npz:policy",
    "nn_dqn_6x64_policy": "nn:runs/dqn_6x64/model.npz:policy",
    "nn_distill_az_2x32_policy": "nn:runs/distill_az_2x32/best.npz:policy",
    "nn_distill_ens_2x32_policy": "nn:runs/distill_ens_2x32/best.npz:policy",
    "evo_evo_d2_d4": "evo:runs/evo_d2/weights.json:4",
    "ntuple_td_ntuple_d0": "ntuple:runs/td_ntuple/ntuple.npz:0",
    "ntuple_td_ntuple_d4": "ntuple:runs/td_ntuple/ntuple.npz:4",
    # 3 score-head nets (nn/train_score.py) averaged, policy only (no search)
    "nn_ens3_score_8x128_policy": "nn:runs/score_8x128_s0_r1/best.npz+runs/score_8x128_s1_r1/best.npz"
                                  "+runs/score_8x128_s2_r1/best.npz:policy",
}

# Preset line-ups for `python -m arena.tournament --roster NAME`.
ROSTERS = {
    # the 20-bot field of experiments/results/arena/part3 (Elo anchor kaggle_negamax = 1000)
    "part3": {"bots": DEFAULT_LINEUP + [
        "nn_az_6x64_mcts200", "nn_sup_6x64_mcts200", "nn_ppo_6x64_policy", "nn_ppo_6x64_mcts200",
        "nn_ppo_from_sup_policy", "nn_dqn_6x64_policy", "nn_distill_az_2x32_policy",
        "nn_distill_ens_2x32_policy", "evo_evo_d2_d4", "ntuple_td_ntuple_d0", "ntuple_td_ntuple_d4"],
        "elo_anchor": "kaggle_negamax:1000"},
    # research phase: part3's top 15 by Elo (dropped: ntuple_td_ntuple_d4, easyai_negamax,
    # kaggle_negamax, ntuple_td_ntuple_d0, plkmo_az).  kaggle_negamax, the old anchor, is
    # gone, so ours_minimax_d4 is pinned to its part3 Elo (1409) to keep the scales comparable.
    "top15": {"bots": [
        "pons_perfect", "nn_sup_6x64_mcts200", "nn_az_6x64_mcts200", "nn_ppo_from_sup_policy", "bruneton_az",
        "nn_ppo_6x64_mcts200", "nn_distill_az_2x32_policy", "nn_distill_ens_2x32_policy", "alfo_mcts",
        "ours_minimax_d8", "evo_evo_d2_d4", "ours_minimax_d4", "galli_minimax", "nn_ppo_6x64_policy",
        "nn_dqn_6x64_policy"],
        "elo_anchor": "ours_minimax_d4:1409"},
    # the top 7 of the 100-opening top-15 run (top15_o100) + the 3-net score-head ensemble.
    # ours_minimax_d4 is gone, so bruneton_az is pinned at its top15_o100 Elo to keep the same scale.
    "top8": {"bots": [
        # same order as top15, so the existing top15_o100 games keep their (a, b) keys
        "pons_perfect", "nn_sup_6x64_mcts200", "nn_az_6x64_mcts200", "nn_ppo_from_sup_policy", "bruneton_az",
        "nn_ppo_6x64_mcts200", "nn_distill_az_2x32_policy", "nn_ens3_score_8x128_policy"],
        "elo_anchor": "bruneton_az:1609"},
}


def _ours_minimax(depth):
    from engine.agents import MinimaxAgent
    return A.Ours(MinimaxAgent(depth), "ours_minimax_d%d" % depth)


class _TestBot:
    """Misbehaving bots for tests/test_arena_timeouts.py only (C4_ARENA_TEST=1).
    ok: plays at once.  sleep: sleeps forever (the per-move alarm stops it).
    hang: ignores the alarm and sleeps (like a bot stuck in C code).
    busy: ignores the alarm and spins the CPU.  crash: kills its own process
    (like an out-of-memory kill).  slow: 0.3 s per move (for the game cap).
    threads: forfeits (raises) unless the worker's BLAS thread limit is 1."""

    def __init__(self, mode):
        self.mode = mode
        self.name = "test_" + mode
        self.last_result = None

    def choose(self, board):
        import signal
        import time
        if self.mode == "slow":
            time.sleep(0.3)
        if self.mode == "threads":
            from threadpoolctl import threadpool_info
            bad = [(i["internal_api"], i["num_threads"]) for i in threadpool_info() if i["num_threads"] != 1]
            if bad or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
                raise RuntimeError("thread limit not applied: %s" % bad)
        if board.moves_played >= 6:
            if self.mode == "sleep":
                while True:
                    time.sleep(1)
            if self.mode == "hang":
                signal.signal(signal.SIGALRM, signal.SIG_IGN)
                while True:
                    time.sleep(1)
            if self.mode == "busy":
                signal.signal(signal.SIGALRM, signal.SIG_IGN)
                x = 0
                while True:
                    x += 1
            if self.mode == "crash":
                os._exit(3)
        return board.legal_moves()[0]


if os.environ.get("C4_ARENA_TEST") == "1":
    for _m in ("hang", "crash", "ok", "sleep", "busy", "slow", "threads"):
        BOTS["test_" + _m] = {"label": "test bot (%s)" % _m, "group": "baseline", "params": {},
                              "make": (lambda m: (lambda: _TestBot(m)))(_m)}


def _level(level):
    from engine.agents import agent_for_level
    return agent_for_level(level)


def _random():
    from engine.agents import RandomAgent
    return RandomAgent()


def nn_entry(spec):
    """Dynamic entry for a trained network: 'nn:<model path>[:mode[:n]]'
    e.g. nn:runs/az_6x64/model.npz:mcts:200  or  nn:runs/sup_6x64/best.npz:policy"""
    parts = spec.split(":")
    path = parts[1]
    mode = parts[2] if len(parts) > 2 else "mcts"
    n = int(parts[3]) if len(parts) > 3 else (200 if mode == "mcts" else 16 if mode == "hybrid" else 4)
    sims = int(parts[4]) if mode == "hybrid" and len(parts) > 4 else 200     # nn:<path>:hybrid:<ply>[:<sims>]
    paths = path.split("+")                 # 'a.npz+b.npz+c.npz' = ensemble (averaged)
    runs = [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in paths]
    if len(runs) == 1:
        run = runs[0]
    else:
        common = os.path.commonprefix(runs).rstrip("_s") or "nets"
        run = "ens%d_%s" % (len(runs), common)
    short = {"mcts": "mcts%d" % n, "policy": "policy", "minimax": "minimax_d%d" % n,
             "hybrid": "hybrid%d_mcts%d" % (n, sims)}[mode]
    name = "nn_%s_%s" % (run, short)

    def make():
        from nn.bench import make_agent
        if mode == "mcts":
            ag = make_agent(("nn", path, "mcts", n))
        elif mode == "policy":
            ag = make_agent(("nn", path, "policy"))
        elif mode == "hybrid":
            ag = make_agent(("nn", path, "hybrid", n, sims))
        else:
            ag = make_agent(("nn", path, "minimax", n))
        return A.Ours(ag, name)
    label = "Our neural net (%s, %s)" % (run, short) if len(runs) == 1 else \
        "Ensemble of %d nets (%s; %s)" % (len(runs), ", ".join(runs), short)
    return name, {"label": label, "group": "ours",
                  "author": "Connor Thelen", "url": "", "license": "", "lang": "Python / numpy",
                  "algorithm": {"mcts": "Our ResNet policy+value network + PUCT MCTS (%d simulations)." % n,
                                "policy": "Our network's policy head alone (no search).",
                                "minimax": "Our minimax (depth %d) with the network's value as the "
                                           "evaluation function." % n,
                                "hybrid": "Our network + MCTS (%d sims, exact C alpha-beta values at late leaves) "
                                          "until ply %d, then exact C alpha-beta search." % (sims, n)}[mode],
                  "params": {"model": path}, "make": make}


def rl_entry(spec):
    """Dynamic entry for the non-network methods of rl/:
    'evo:<weights.json>[:depth]'    our minimax with evolved heuristic weights (default depth 4)
    'ntuple:<ntuple.npz>[:depth]'   TD n-tuple player; depth 0 = greedy, >0 = inside our minimax"""
    parts = spec.split(":")
    kind, path = parts[0], parts[1]
    depth = int(parts[2]) if len(parts) > 2 else (4 if kind == "evo" else 0)
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    name = "%s_%s_d%d" % (kind, run, depth)

    def make():
        from nn.bench import make_agent
        ag = make_agent(("weights" if kind == "evo" else "ntuple", path, depth))
        return A.Ours(ag, name)
    if kind == "evo":
        label = "Evolved heuristic (%s), minimax d%d" % (run, depth)
        algo = "Our minimax + alpha-beta (depth %d) with heuristic weights tuned by CMA-ES." % depth
    else:
        label = "TD n-tuple (%s)%s" % (run, ", minimax d%d" % depth if depth else ", greedy")
        algo = ("n-tuple network (lookup tables over 8-cell patterns) trained by TD(lambda) self-play; "
                + ("value used at the leaves of our minimax (depth %d)." % depth if depth else "greedy, no search."))
    return name, {"label": label, "group": "ours", "author": "Connor Thelen", "url": "", "license": "",
                  "lang": "Python / numpy", "algorithm": algo, "params": {"model": path}, "make": make}


def resolve(names, overrides=None):
    """names -> {name: entry}. overrides: {'bot.param': value}."""
    out = {}
    overrides = overrides or {}
    for n in names:
        n = ALIASES.get(n, n)
        if n.startswith("nn:"):
            key, entry = nn_entry(n)
        elif n.startswith("evo:") or n.startswith("ntuple:"):
            key, entry = rl_entry(n)
        elif n in BOTS:
            key, entry = n, dict(BOTS[n])
            params = dict(entry["params"])
            for k, v in overrides.items():
                b, _, p = k.partition(".")
                if b == n and p in params:
                    params[p] = type(params[p])(v)
            entry["params"] = params
        else:
            raise SystemExit("unknown bot %r (see --list)" % n)
        entry["spec"] = n
        out[key] = entry
    return out


def build(entry):
    p = {k: v for k, v in entry["params"].items() if k != "model"}
    return entry["make"](**p)
