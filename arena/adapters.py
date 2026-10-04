"""Adapters: make every bot answer the same question.

Each adapter has ``choose(board) -> column (0-6)`` where ``board`` is our
engine.board.Board, and translates our position into the bot's own board
format (orientation, piece codes, whose turn), calls the bot's ORIGINAL
search code, and translates the answer back. Nothing about how a bot thinks
is changed; the only edits to third-party code are listed per adapter:

  galli        exec only the function/constant definitions of
               connect4_with_ai.py (skip the pygame window + game loop that
               run at import time).
  kaggle       none (connectx.py is imported as is).
  easyai       none.
  alfo_mcts    the file is Python 2 with a Tkinter GUI: we exec the game +
               MCTS part only, expand tabs as Python 2 did (8 columns) and
               drop one debug `print` statement (Python 2 syntax).
  plkmo_az     their UCT_search calls `.cuda()`; on a machine without CUDA we
               make Tensor.cuda() a no-op so it runs on the CPU.
  bruneton_az  none.

Randomness: several bots break ties with Python's `random` / numpy's global
RNG. The tournament seeds both before every game, so results are
reproducible.
"""

import ast
import importlib.util
import math
import os
import sys
import time
import types

import numpy as np

from engine.board import CENTER_ORDER, COLS, ROWS, Board
from engine.minimax import SearchResult

HERE = os.path.dirname(os.path.abspath(__file__))
THIRD = os.path.join(HERE, "third_party")
_RANK = {c: i for i, c in enumerate(CENTER_ORDER)}


class BotUnavailable(RuntimeError):
    pass


def _need(key, *files):
    d = os.path.join(THIRD, key)
    for f in files:
        if not os.path.isfile(os.path.join(d, f)):
            raise BotUnavailable("%s not downloaded - run: python -m arena.fetch %s" % (key, key))
    return d


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _grid(board):
    """Our grid: row 0 = top, +1 = first player, -1 = second player."""
    return board.to_grid()


class _Base:
    name = "bot"

    def __init__(self):
        self.last_result = None

    def _done(self, col, t0, nodes=0, score=0.0):
        self.last_result = SearchResult(col=int(col), score=float(score), nodes=nodes,
                                        ms=(time.perf_counter() - t0) * 1000.0, depth_reached=0)
        return int(col)


# ------------------------------------------------------------------ Galli
class GalliMinimax(_Base):
    """Keith Galli, 'Connect4-Python' (MIT). Minimax + alpha-beta, depth 5 in
    the original game, window heuristic (center 3, three 5, two 2, opponent
    three -4) - the same family of heuristic as our engine/heuristic.py."""

    def __init__(self, depth=5):
        super().__init__()
        d = _need("galli", "connect4_with_ai.py")
        src = open(os.path.join(d, "connect4_with_ai.py"), encoding="utf-8").read()
        tree = ast.parse(src)
        keep = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                if any(n in ("pygame", "sys") for n in names):
                    continue
                keep.append(node)
            elif isinstance(node, ast.FunctionDef):
                if node.name != "draw_board":
                    keep.append(node)
            elif isinstance(node, ast.Assign):
                # module constants only (stop at the first statement that runs the game)
                if isinstance(node.value, ast.Call):
                    break
                keep.append(node)
        mod = ast.Module(body=keep, type_ignores=[])
        self.ns = {"__name__": "galli_connect4"}
        exec(compile(mod, os.path.join(d, "connect4_with_ai.py"), "exec"), self.ns)
        self.depth = depth
        self.name = "galli_minimax_d%d" % depth

    def choose(self, board):
        t0 = time.perf_counter()
        ns = self.ns
        me = board.to_move
        b = np.zeros((ROWS, COLS))
        for r, row in enumerate(_grid(board)):
            for c, v in enumerate(row):
                if v:
                    b[ROWS - 1 - r][c] = ns["AI_PIECE"] if v == me else ns["PLAYER_PIECE"]
        col, score = ns["minimax"](b, self.depth, -math.inf, math.inf, True)
        return self._done(col, t0, score=min(max(score, -1e6), 1e6))


# ----------------------------------------------------------------- Kaggle
class KaggleNegamax(_Base):
    """Kaggle 'kaggle-environments' ConnectX built-in negamax agent
    (Apache-2.0). Plain negamax, depth 4 hard-coded, no pruning; leaf score
    = how many own stones touch the cell (a 'clustering' heuristic)."""

    def __init__(self):
        super().__init__()
        d = _need("kaggle", "kaggle_environments/envs/connectx/connectx.py",
                  "kaggle_environments/envs/connectx/connectx.json")
        self.mod = _load_module("kaggle_connectx", os.path.join(d, "kaggle_environments", "envs",
                                                                "connectx", "connectx.py"))
        self.config = types.SimpleNamespace(columns=COLS, rows=ROWS, inarow=4)
        self.name = "kaggle_negamax"

    def choose(self, board):
        t0 = time.perf_counter()
        me = board.to_move
        flat = []
        for row in _grid(board):                       # row 0 = top, like ConnectX
            for v in row:
                flat.append(0 if v == 0 else (1 if v == me else 2))
        obs = types.SimpleNamespace(board=flat, mark=1)
        return self._done(self.mod.negamax_agent(obs, self.config), t0)


# ----------------------------------------------------------------- easyAI
class EasyAINegamax(_Base):
    """Zulko 'easyAI' (MIT): its Negamax (alpha-beta + optional transposition
    table) on its ConnectFour game. The game's scoring is only -100 for a lost
    position, else 0 - no positional heuristic, so it plays purely tactically
    (and picks the leftmost column among equal moves)."""

    def __init__(self, depth=5):
        super().__init__()
        d = _need("easyai", "easyAI/__init__.py", "easyAI/games/ConnectFour.py")
        if d not in sys.path:
            sys.path.insert(0, d)
        import easyAI  # noqa: F401
        self.easyAI = easyAI
        self.cf = _load_module("easyai_connectfour", os.path.join(d, "easyAI", "games", "ConnectFour.py"))
        self.depth = depth
        self.name = "easyai_negamax_d%d" % depth

    def choose(self, board):
        t0 = time.perf_counter()
        me = board.to_move
        arr = np.zeros((ROWS, COLS), dtype=int)       # easyAI: row 0 = bottom
        for r, row in enumerate(_grid(board)):
            for c, v in enumerate(row):
                if v:
                    arr[ROWS - 1 - r][c] = 1 if v == me else 2
        algo = self.easyAI.Negamax(self.depth)
        game = self.cf.ConnectFour([self.easyAI.AI_Player(algo), self.easyAI.AI_Player(algo)], board=arr)
        game.current_player = 1
        return self._done(algo(game), t0)


# ------------------------------------------------------------- Alfo MCTS
class AlfoMCTS(_Base):
    """Alfredo de la Fuente, 'Connect4' (MIT): classic UCT Monte Carlo Tree
    Search with uniformly random roll-outs (no neural network, no heuristic).
    Original settings: 3000 iterations, exploration factor 2.0."""

    def __init__(self, iterations=3000, factor=2.0):
        super().__init__()
        d = _need("alfo_mcts", "game.py")
        src = open(os.path.join(d, "game.py"), encoding="utf-8").read().expandtabs(8)
        src = src.split("## GUI Configuration")[0]
        lines = []
        for ln in src.splitlines():
            s = ln.strip()
            if s.startswith("from Tkinter") or s.startswith("import tkFont"):
                continue
            if s.startswith("print ["):           # Python 2 debug print in MTCS()
                ln = ln[:len(ln) - len(ln.lstrip())] + "pass"
            lines.append(ln)
        self.ns = {"__name__": "alfo_connect4"}
        exec(compile("\n".join(lines), os.path.join(d, "game.py"), "exec"), self.ns)
        self.iterations = iterations
        self.factor = factor
        self.name = "alfo_mcts%d" % iterations

    def choose(self, board):
        t0 = time.perf_counter()
        me = board.to_move
        g = [[0 if v == 0 else (1 if v == me else -1) for v in row] for row in _grid(board)]
        ns = self.ns
        root = ns["Node"](ns["Board"](g, [None, None]))
        best = ns["MTCS"](self.iterations, root, self.factor)
        return self._done(best.state.last_move[1], t0, nodes=self.iterations)


# ---------------------------------------------------- plkmo AlphaZero
class PlkmoAlphaZero(_Base):
    """plkmo 'AlphaZero_Connect4' (Apache-2.0; the code from the AI Singapore
    blog 'From-scratch implementation of AlphaZero for Connect4'): 19 residual
    blocks x 128 channels, pretrained network 'iter8' shipped in the repo,
    PUCT MCTS with 777 reads per move (their setting)."""

    def __init__(self, reads=777):
        super().__init__()
        d = _need("plkmo_az", "src/MCTS_c4.py", "src/model_data/c4_current_net_trained_iter8.pth.tar")
        src = os.path.join(d, "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        import torch
        self.torch = torch
        if torch.cuda.is_available():
            from nn.common import pick_device
            dev = pick_device("auto")
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            self.cuda = dev.type == "cuda"
        else:
            self.cuda = False
        if not self.cuda:
            torch.Tensor.cuda = lambda self_, *a, **k: self_      # run their .cuda() calls on CPU
        import alpha_net_c4
        import connect_board
        import MCTS_c4
        self.MCTS = MCTS_c4
        self.cboard = connect_board.board
        net = alpha_net_c4.ConnectNet()
        ck = torch.load(os.path.join(src, "model_data", "c4_current_net_trained_iter8.pth.tar"),
                        map_location="cpu", weights_only=True)
        net.load_state_dict(ck["state_dict"])
        if self.cuda:
            net.cuda()
        net.eval()
        self.net = net
        self.reads = reads
        self.name = "plkmo_az%d" % reads

    def choose(self, board):
        t0 = time.perf_counter()
        b = self.cboard()
        arr = np.zeros([6, 7]).astype(str)
        arr[arr == "0.0"] = " "
        for r, row in enumerate(_grid(board)):        # row 0 = top in both
            for c, v in enumerate(row):
                if v:
                    arr[r, c] = "O" if v == 1 else "X"   # "O" moves first in their game
        b.current_board = arr
        b.player = 0 if board.to_move == 1 else 1
        with self.torch.no_grad():
            root = self.MCTS.UCT_search(b, self.reads, self.net, 0.1)
        visits = root.child_number_visits
        legal = board.legal_moves()
        col = max(legal, key=lambda c: (visits[c], -_RANK[c]))
        return self._done(col, t0, nodes=self.reads)


# ------------------------------------------------- Bruneton AlphaZero
class BrunetonAlphaZero(_Base):
    """J.-P. Bruneton, A. Douin, V. Reverdy, 'Alpha-Zero-algorithm-for-Connect-4-game'
    (BSD-3): 1 residual block x 256 channels, pretrained 'best_model_resnet.pth'
    (the README calls it 'almost perfect'), PUCT MCTS, 200 simulations per move
    in their play script."""

    def __init__(self, sims=200):
        super().__init__()
        d = _need("bruneton_az", "ResNet.py", "MCTS_NN.py", "best_model_resnet.pth")
        if d not in sys.path:
            sys.path.insert(0, d)
        import torch
        n_threads = torch.get_num_threads()
        import Game_bitboard
        import MCTS_NN
        import ResNet
        self.Game = Game_bitboard.Game
        self.MCTS_NN = MCTS_NN
        net = ResNet.resnet18()
        torch.set_num_threads(n_threads)       # their constructor forces 1 thread globally
        net.load_state_dict(torch.load(os.path.join(d, "best_model_resnet.pth"), map_location="cpu",
                                       weights_only=True))
        net.eval()
        self.net = net
        self.torch = torch
        self.sims = sims
        self.name = "bruneton_az%d" % sims

    def choose(self, board):
        t0 = time.perf_counter()
        yellow = red = 0                               # their layout: bit 8*col + height
        for r, row in enumerate(_grid(board)):
            h = ROWS - 1 - r
            for c, v in enumerate(row):
                if v == 1:
                    yellow |= 1 << (8 * c + h)
                elif v == -1:
                    red |= 1 << (8 * c + h)
        state = [yellow, red, 1 if board.to_move == 1 else -1]   # yellow (player 1) moves first
        tree = self.MCTS_NN.MCTS_NN(self.net, use_dirichlet=False)
        root = tree.createNode(state)
        with self.torch.no_grad():
            for _ in range(self.sims):
                tree.simulate(root, cpuct=1)
        visits = np.array([ch.N for ch in root.children])
        best = np.random.choice(np.where(visits == visits.max())[0])   # their tie-break
        col = self.Game().convert_move_to_col_index(root.children[best].move)
        return self._done(col, t0, nodes=self.sims)


# ------------------------------------------------------------ Pons solver
class PonsSolver(_Base):
    """Pascal Pons' perfect-play solver (AGPL-3.0, tools/pons). Plays a
    move with the best game-theoretic score (fastest win / slowest loss),
    most central column among ties."""

    def __init__(self):
        super().__init__()
        from experiments.common import SOLVER_BIN
        if not (os.path.isfile(SOLVER_BIN) and os.access(SOLVER_BIN, os.X_OK)):
            raise BotUnavailable("Pons solver not built - run: cd tools/pons && make c4solver")
        from tools.solver import Solver
        self.solver = Solver(timeout=120)
        self.name = "pons_perfect"

    def choose(self, board):
        t0 = time.perf_counter()
        sc = self.solver.solve_all(board.to_move_string())
        best = max(v for v in sc if v is not None)
        col = min((c for c in range(COLS) if sc[c] == best), key=lambda c: _RANK[c])
        return self._done(col, t0, score=best)


# -------------------------------------------------------------- ours
class Ours(_Base):
    """Wraps one of our agents (engine.agents / nn.agents)."""

    def __init__(self, agent, name):
        super().__init__()
        self.agent = agent
        self.name = name

    def choose(self, board):
        col = self.agent.choose(board)
        self.last_result = self.agent.last_result
        return col
