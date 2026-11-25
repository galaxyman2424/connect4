"""Agents that play with a trained network.  Same interface as engine.agents:
``choose(board) -> column`` and ``last_result`` (engine.minimax.SearchResult).

  PolicyAgent      no search at all: play the column the policy head likes most.
                   Measures what the network "knows" by intuition alone.
  MCTSAgent        PUCT tree search guided by the network (AlphaZero style).
  NNMinimaxAgent   the project's own negamax/alpha-beta search, with the
                   network's value head replacing the hand-crafted heuristic
                   at the leaves.  This is the dossier hypothesis test:
                   "a small neural network can learn a board evaluation that
                   plays as well as our hand-crafted minimax heuristic".

All three accept any evaluator with .evaluate(curs, masks) -> (priors,
values, wdl): numpy_net.NumpyNet (.npz) or common.TorchEvaluator (.pt).
"""

import time

import numpy as np

from engine.agents import MinimaxAgent
from engine.board import CENTER_ORDER
from engine.minimax import SearchResult

from . import mcts

_RANK = {c: i for i, c in enumerate(CENTER_ORDER)}


class PolicyAgent:
    def __init__(self, evaluator, name="nn_policy"):
        self.ev = evaluator
        self.name = name
        self.last_result = None
        self.last_info = {}

    def choose(self, board):
        t0 = time.perf_counter()
        pri, val, wdl = self.ev.evaluate([board.current_bits()], [board.mask])
        legal = board.legal_moves()
        best = max(legal, key=lambda c: (pri[0][c], -_RANK[c]))
        self.last_info = {"policy": [round(float(x), 4) for x in pri[0]],
                          "wdl": [round(float(x), 4) for x in wdl[0]]}
        self.last_result = SearchResult(col=best, score=float(val[0]), nodes=1,
                                        ms=(time.perf_counter() - t0) * 1000.0, depth_reached=0)
        return best


class CachedEvaluator:
    """Wraps an evaluator with a position -> (prior, value, wdl) cache.

    Within one game the MCTS trees of consecutive moves overlap heavily (and Connect 4 has
    many transpositions), so most leaves were already evaluated on an earlier move.
    Measured on a full 200-sim game with sup_6x64: 5,763 evaluations asked, 1,178 computed
    (4.9x fewer network calls).  The network is deterministic, so moves are unchanged."""

    def __init__(self, evaluator, max_size=400_000):
        self.ev = evaluator
        self.cache = {}
        self.max_size = max_size
        self.asked = 0
        self.computed = 0
        self.meta = getattr(evaluator, "meta", {})

    def evaluate(self, curs, masks):
        keys = list(zip((int(c) for c in curs), (int(m) for m in masks)))
        self.asked += len(keys)
        cache = self.cache
        need = [k for k in dict.fromkeys(keys) if k not in cache]
        if need:
            self.computed += len(need)
            if len(cache) + len(need) > self.max_size:
                cache.clear()
            pri, val, wdl = self.ev.evaluate([k[0] for k in need], [k[1] for k in need])
            for k, p, v, w in zip(need, pri, val, wdl):
                cache[k] = (p, v, w)
        got = [cache[k] for k in keys]
        return (np.array([g[0] for g in got]), np.array([g[1] for g in got]),
                np.array([g[2] for g in got]))


class MCTSAgent:
    """Deterministic at play time: no Dirichlet noise, most-visited move.
    cache=True (default) keeps a network-output cache for the agent's lifetime (one game in
    the arena / ladder), which removes ~80% of the network calls without changing moves.
    The ms/move it reports therefore drops too: set C4_MCTS_CACHE=0 to measure the old,
    uncached speed (e.g. to compare ms/move with arena runs made before the cache)."""

    def __init__(self, evaluator, sims=200, c_puct=1.5, name=None, cache=None):
        import os
        if cache is None:
            cache = os.environ.get("C4_MCTS_CACHE", "1") != "0"
        self.ev = CachedEvaluator(evaluator) if cache and not isinstance(evaluator, CachedEvaluator) else evaluator
        if cache and type(evaluator).__name__ == "NativeNet":      # nn/native.py: cache lives in C
            self.ev.native_cache = evaluator.new_cache()
        self.sims = sims
        self.c_puct = c_puct
        self.name = name or "nn_mcts%d" % sims
        self.last_result = None
        self.last_info = {}

    def choose(self, board):
        t0 = time.perf_counter()
        root = mcts.root_from_board(board)
        mcts.search([root], self.ev, self.sims, c_puct=self.c_puct)
        col = mcts.choose_move(root, 0.0)
        pv = mcts.principal_variation(root)
        self.last_info = {"visits": list(root.N), "prior": [round(x, 4) for x in root.P],
                          "q": round(root.q(), 4), "pv": pv}
        self.last_result = SearchResult(col=col, score=float(root.q()), nodes=self.sims,
                                        ms=(time.perf_counter() - t0) * 1000.0,
                                        depth_reached=len(pv))
        return col


class NNValueEvaluator:
    """Adapter: network value -> engine.minimax leaf evaluation f(me, opp).

    The value v = P(win) - P(loss) in [-1, 1] is multiplied by ``scale`` so it
    lives on a similar range to the hand-crafted heuristic (roughly +-100),
    far below the search's win scores (~1e6).  A cache avoids evaluating the
    same position twice (transpositions are common in Connect 4)."""

    def __init__(self, evaluator, scale=100.0, cache_size=500_000):
        self.ev = evaluator
        self.scale = scale
        self.cache = {}
        self.cache_size = cache_size
        self.calls = 0

    def __call__(self, me, opp):
        key = (me, opp)
        v = self.cache.get(key)
        if v is None:
            self.calls += 1
            _, val, _ = self.ev.evaluate([me], [me | opp])
            v = float(val[0]) * self.scale
            if len(self.cache) >= self.cache_size:
                self.cache.clear()
            self.cache[key] = v
        return v

    def prefetch(self, pairs):
        """engine.minimax calls this at depth-1 nodes with every child leaf (me, opp):
        one batched network call instead of up to 7 single-position calls."""
        need = [k for k in dict.fromkeys(pairs) if k not in self.cache]
        if len(need) < 2:
            return
        self.calls += 1
        _, val, _ = self.ev.evaluate([k[0] for k in need], [k[0] | k[1] for k in need])
        if len(self.cache) + len(need) >= self.cache_size:
            self.cache.clear()
        for k, v in zip(need, val):
            self.cache[k] = float(v) * self.scale


class NNMinimaxAgent(MinimaxAgent):
    def __init__(self, evaluator, depth, scale=100.0, seed=None, name=None):
        super().__init__(depth, seed=seed, evaluator=NNValueEvaluator(evaluator, scale))
        self.name = name or "nn_minimax_d%d" % depth


def load_agent(model_path, mode="mcts", sims=200, depth=4, device="auto"):
    """Convenience: model file + mode ('policy' | 'mcts' | 'minimax') -> agent."""
    from .common import load_evaluator
    ev = load_evaluator(model_path, device)
    if mode == "policy":
        return PolicyAgent(ev)
    if mode == "mcts":
        return MCTSAgent(ev, sims=sims)
    if mode == "minimax":
        return NNMinimaxAgent(ev, depth)
    raise ValueError("mode must be policy, mcts or minimax")


def random_init_numpy_net(blocks=2, channels=16, seed=0):
    """Tiny randomly initialised NumpyNet (for tests and smoke runs without torch)."""
    import json
    import os
    import tempfile
    from .numpy_net import NumpyNet
    rng = np.random.default_rng(seed)
    c = channels
    a = {"stem.w": rng.normal(0, 0.2, (c, 4, 3, 3)), "stem.b": np.zeros(c)}
    for i in range(blocks):
        for k in ("c1", "c2"):
            a["b%d.%s.w" % (i, k)] = rng.normal(0, 0.05, (c, c, 3, 3))
            a["b%d.%s.b" % (i, k)] = np.zeros(c)
    a["pol.w"] = rng.normal(0, 0.2, (2, c)); a["pol.b"] = np.zeros(2)
    a["pol_fc.w"] = rng.normal(0, 0.1, (7, 84)); a["pol_fc.b"] = np.zeros(7)
    a["val.w"] = rng.normal(0, 0.2, (4, c)); a["val.b"] = np.zeros(4)
    a["val_fc1.w"] = rng.normal(0, 0.1, (16, 168)); a["val_fc1.b"] = np.zeros(16)
    a["val_fc2.w"] = rng.normal(0, 0.1, (3, 16)); a["val_fc2.b"] = np.zeros(3)
    a = {k: v.astype(np.float32) for k, v in a.items()}
    a["config_json"] = np.array(json.dumps({"blocks": blocks, "channels": c, "policy_channels": 2,
                                            "value_channels": 4, "value_hidden": 16}))
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    np.savez(path, **a)
    net = NumpyNet(path)
    os.remove(path)
    return net
