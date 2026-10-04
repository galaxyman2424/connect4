"""Download the third-party bots, pinned to exact commits, into arena/third_party/.

    python -m arena.fetch            # everything (about 30 MB, mostly two network weight files)
    python -m arena.fetch galli kaggle
    python -m arena.fetch --force    # re-download

We download the original files unmodified (plus each project's licence) and
record where they came from in arena/third_party/<key>/SOURCE.json. The
folder is git-ignored: the code belongs to its authors and is fetched, not
re-published in our repo. Any adaptation needed to call a bot from our
arena lives in arena/adapters.py and is documented in docs/ARENA.md.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
THIRD = os.path.join(HERE, "third_party")

SOURCES = {
    "galli": {
        "repo": "KeithGalli/Connect4-Python",
        "commit": "503c0b4807001e7ea43a039cb234a4e55c4b226c",
        "files": ["connect4_with_ai.py", "LICENSE", "README.md"],
    },
    "kaggle": {
        "repo": "Kaggle/kaggle-environments",
        "commit": "eb8b5ef905cb066babf8bfcd776540d0e9dd8fa6",
        "files": ["kaggle_environments/envs/connectx/connectx.py",
                  "kaggle_environments/envs/connectx/connectx.json", "LICENSE"],
    },
    "easyai": {
        "repo": "Zulko/easyAI",
        "commit": "43e946283df91f565fd1655826ce9dbde6e829d7",
        "files": ["easyAI/__init__.py", "easyAI/TwoPlayerGame.py", "easyAI/Player.py",
                  "easyAI/AI/__init__.py", "easyAI/AI/DUAL.py", "easyAI/AI/DictTranspositionTable.py",
                  "easyAI/AI/HashTranspositionTable.py", "easyAI/AI/Hashes.py", "easyAI/AI/MTdriver.py",
                  "easyAI/AI/Negamax.py", "easyAI/AI/NonRecursiveNegamax.py", "easyAI/AI/SSS.py",
                  "easyAI/AI/TranspositionTable.py", "easyAI/AI/solving.py",
                  "easyAI/games/ConnectFour.py", "LICENCE.txt"],
    },
    "alfo_mcts": {
        "repo": "Alfo5123/Connect4",
        "commit": "d37120361cebb6d11ec3a40806f73c95c06778b2",
        "files": ["game.py", "LICENSE", "README.md"],
    },
    "plkmo_az": {
        "repo": "plkmo/AlphaZero_Connect4",
        "commit": "01ff24aae145ccd23e58f630aa49582cade49847",
        "files": ["src/alpha_net_c4.py", "src/connect_board.py", "src/encoder_decoder_c4.py",
                  "src/MCTS_c4.py", "src/model_data/c4_current_net_trained_iter8.pth.tar",
                  "LICENSE", "README.md"],
    },
    "bruneton_az": {
        "repo": "jpbruneton/Alpha-Zero-algorithm-for-Connect-4-game",
        "commit": "6cd76be879fe2603742df516822b4988411e3c71",
        "files": ["config.py", "Game_bitboard.py", "MCTS_NN.py", "ResNet.py",
                  "best_model_resnet.pth", "LICENSE", "README.md"],
    },
}


def raw_url(repo, commit, path):
    return "https://raw.githubusercontent.com/%s/%s/%s" % (repo, commit, path)


def fetch(key, force=False):
    src = SOURCES[key]
    dest = os.path.join(THIRD, key)
    marker = os.path.join(dest, "SOURCE.json")
    if os.path.isfile(marker) and not force:
        print("%-12s already present (use --force to re-download)" % key)
        return True
    os.makedirs(dest, exist_ok=True)
    hashes = {}
    for path in src["files"]:
        url = raw_url(src["repo"], src["commit"], path)
        out = os.path.join(dest, *path.split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001
            print("%-12s FAILED %s: %s" % (key, path, e))
            return False
        with open(out, "wb") as f:
            f.write(data)
        hashes[path] = hashlib.sha256(data).hexdigest()
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"repo": "https://github.com/" + src["repo"], "commit": src["commit"],
                   "files": hashes}, f, indent=2)
    print("%-12s ok (%s @ %s)" % (key, src["repo"], src["commit"][:8]))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help="subset of: " + ", ".join(SOURCES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    keys = a.keys or list(SOURCES)
    bad = [k for k in keys if k not in SOURCES]
    if bad:
        sys.exit("unknown: %s (choose from %s)" % (bad, list(SOURCES)))
    ok = all([fetch(k, a.force) for k in keys])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
