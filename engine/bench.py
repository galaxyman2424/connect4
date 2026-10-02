"""Benchmark: ms and nodes per search depth.

    python -m engine.bench                 # depths 1-10, empty board + midgames
    python -m engine.bench --max-depth 8 --no-tt
    python -m engine.bench --levels        # time each slider level
"""

import argparse
import platform
import sys

from .board import Board
from .minimax import search
from .agents import LEVELS, agent_for_level

# (label, Pons move string); midgames come from depth-7 self-play after 4453
POSITIONS = [
    ("empty", ""),
    ("opening 4453", "4453"),
    ("midgame 12 ply", "445344345624"),
    ("midgame 16 ply", "4453443456243333"),
    ("midgame 20 ply", "44534434562433337777"),
]


def bench_depths(max_depth, use_tt, ordering, positions):
    print("Python %s  use_tt=%s ordering=%s" % (platform.python_version(), use_tt, ordering))
    print("%-16s %5s %10s %10s %6s %10s" % ("position", "depth", "ms", "nodes", "col", "score"))
    for label, moves in positions:
        b = Board.from_moves(moves)
        for d in range(1, max_depth + 1):
            r = search(b, d, use_tt=use_tt, ordering=ordering)
            print("%-16s %5d %10.1f %10d %6d %10.0f%s" % (
                label, d, r.ms, r.nodes, r.col, r.score,
                "" if r.depth_reached == d else "  (proven at depth %d)" % r.depth_reached))
        print()


def bench_levels(positions):
    print("%-16s %5s %10s %10s %6s %6s" % ("position", "level", "ms", "nodes", "depth", "rand"))
    for label, moves in positions:
        b = Board.from_moves(moves)
        for lvl in sorted(LEVELS):
            agent = agent_for_level(lvl, seed=0)
            agent.choose(b)
            r = agent.last_result
            print("%-16s %5d %10.1f %10d %6d %6s" % (label, lvl, r.ms, r.nodes,
                                                    r.depth_reached, r.random))
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--no-tt", action="store_true")
    ap.add_argument("--no-ordering", action="store_true")
    ap.add_argument("--levels", action="store_true", help="time every slider level")
    ap.add_argument("--moves", help="benchmark one extra position (Pons move string)")
    args = ap.parse_args(argv)
    positions = list(POSITIONS)
    if args.moves is not None:
        positions = [("custom " + args.moves, args.moves)]
    if args.levels:
        bench_levels(positions)
    else:
        bench_depths(args.max_depth, not args.no_tt, not args.no_ordering, positions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
