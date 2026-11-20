"""Write the paper outline as Markdown (replaces the old "Paper builder" tab on /research).

    python -m research.paper_outline              # every finding, top-15 bot and sweep run
    python -m research.paper_outline --selected   # only what selection.json marks as included
    python -m research.paper_outline --out path.md

Default output: docs/PAPER_OUTLINE.md, which is gitignored (a local working draft).
Re-run it after the research scripts update experiments/results/research/*.json.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from server.research_api import ROOT as _R, write_outline  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selected", action="store_true", help="only findings/bots/runs marked in selection.json")
    ap.add_argument("--out", default=None, help="output path (default docs/PAPER_OUTLINE.md)")
    a = ap.parse_args(argv)
    path, n = write_outline(a.out, include_all=not a.selected)
    print("wrote %s (%d bytes)" % (os.path.relpath(path, _R), n))


if __name__ == "__main__":
    main()
