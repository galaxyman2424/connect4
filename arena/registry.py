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


def _ours_minimax(depth):
    from engine.agents import MinimaxAgent
    return A.Ours(MinimaxAgent(depth), "ours_minimax_d%d" % depth)


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
    n = int(parts[3]) if len(parts) > 3 else (200 if mode == "mcts" else 4)
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    short = {"mcts": "mcts%d" % n, "policy": "policy", "minimax": "minimax_d%d" % n}[mode]
    name = "nn_%s_%s" % (run, short)

    def make():
        from nn.bench import make_agent
        if mode == "mcts":
            ag = make_agent(("nn", path, "mcts", n))
        elif mode == "policy":
            ag = make_agent(("nn", path, "policy"))
        else:
            ag = make_agent(("nn", path, "minimax", n))
        return A.Ours(ag, name)
    return name, {"label": "Our neural net (%s, %s)" % (run, short), "group": "ours",
                  "author": "Connor Thelen", "url": "", "license": "", "lang": "Python / numpy",
                  "algorithm": {"mcts": "Our ResNet policy+value network + PUCT MCTS (%d simulations)." % n,
                                "policy": "Our network's policy head alone (no search).",
                                "minimax": "Our minimax (depth %d) with the network's value as the "
                                           "evaluation function." % n}[mode],
                  "params": {"model": path}, "make": make}


def resolve(names, overrides=None):
    """names -> {name: entry}. overrides: {'bot.param': value}."""
    out = {}
    overrides = overrides or {}
    for n in names:
        if n.startswith("nn:"):
            key, entry = nn_entry(n)
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
