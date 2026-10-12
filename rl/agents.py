"""Agents for the methods that are not neural networks.  Same interface as
engine.agents: choose(board) -> column, plus last_result (SearchResult).

  EvolvedMinimaxAgent(weights, depth)   our minimax with evolved heuristic weights
  NTupleAgent(net, depth)               depth 0: greedy on the n-tuple value;
                                        depth >= 1: our minimax with the n-tuple
                                        value at the leaves (learning + search)

make_agent(spec) builds them from picklable specs (used by nn.bench, the
arena and rl.compare):
  ("weights", path_to_weights.json, depth)
  ("ntuple",  path_to_ntuple.npz,   depth)
"""

import json
import os
import time

import numpy as np

from engine.agents import MinimaxAgent
from engine.minimax import SearchResult

from .features import WeightedEvaluator


def load_weights(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["weights"], d


class EvolvedMinimaxAgent(MinimaxAgent):
    def __init__(self, weights, depth, seed=None, name=None, time_limit=None):
        super().__init__(depth, seed=seed, time_limit=time_limit, evaluator=WeightedEvaluator(weights))
        self.name = name or "evo_d%d" % depth


class NTupleLeaf:
    """n-tuple value as engine.minimax leaf evaluation f(me, opp) (side to move = me).
    The n-tuple net scores afterstates for the player who just moved (= opp),
    so f = -V * scale.  Cached: Connect 4 has many transpositions."""

    def __init__(self, net, scale=100.0, cache_size=500_000):
        self.net = net
        self.scale = scale
        self.cache = {}
        self.cache_size = cache_size

    def __call__(self, me, opp):
        key = (me, opp)
        v = self.cache.get(key)
        if v is None:
            v = -float(self.net.value(np.array([opp], dtype=np.uint64),
                                      np.array([me | opp], dtype=np.uint64))[0]) * self.scale
            if len(self.cache) >= self.cache_size:
                self.cache.clear()
            self.cache[key] = v
        return v


class NTupleAgent:
    def __init__(self, net, depth=0, name=None):
        self.net = net
        self.depth = depth
        self.name = name or "ntuple_d%d" % depth
        self.last_result = None
        self._mm = MinimaxAgent(depth, evaluator=NTupleLeaf(net)) if depth >= 1 else None

    def choose(self, board):
        if self._mm is not None:
            col = self._mm.choose(board)
            self.last_result = self._mm.last_result
            return col
        from . import vecenv as V
        t0 = time.perf_counter()
        vals, lg = self.net.move_values(np.array([board.current_bits()], dtype=np.uint64),
                                        np.array([board.mask], dtype=np.uint64))
        col = int(V.argmax_center(vals, lg)[0])
        self.last_result = SearchResult(col=col, score=float(vals[0][col]), nodes=1,
                                        ms=(time.perf_counter() - t0) * 1000.0, depth_reached=1)
        return col


_CACHE = {}


def _cached(path, loader):
    key = (path, os.path.getmtime(path))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = loader(path)
    return _CACHE[key]


def make_agent(spec):
    kind = spec[0]
    if kind == "weights":
        w, _ = load_weights(spec[1])
        return EvolvedMinimaxAgent(w, int(spec[2]))
    if kind == "ntuple":
        from .ntuple import NTupleNet
        net = _cached(spec[1], NTupleNet.load)
        return NTupleAgent(net, int(spec[2]))
    raise ValueError("bad spec %r" % (spec,))


def spec_name(spec):
    if spec[0] == "weights":
        return "evo_d%d" % int(spec[2])
    if spec[0] == "ntuple":
        return "ntuple_d%d" % int(spec[2])
    raise ValueError("bad spec %r" % (spec,))
