# data/

## Downloading the public datasets (run on your machine)
The build environment could not reach `archive.ics.uci.edu` or `huggingface.co`, so the public datasets are fetched by a
script you run yourself (Windows / macOS / Linux, Python 3.10+, from the project root):

    pip install -r data/requirements-data.txt
    python data/download_data.py --all          # or pick: --uci  --tonycwang 5000  --leon  --book
    # add --force to redo a step whose output already exists

* `--uci` -> `data/uci/connect-4.data` and `data/uci/uci_positions.csv` (columns `c0..c41` in UCI order a1..a6,b1..b6,... bottom-up with x/o/b, `label`, and `grid` = JSON 6x7 list, row 0 = top, +1 = x / first player, -1 = o; checks 67,557 rows). Handles zip/gzip/`.Z` (needs `unlzw3`).
* `--tonycwang N` -> `data/tonycwang_sample.csv` (`grid`, `scores`, `optimal_cols`); N rows streamed through a shuffle buffer. The HF schema was unknown when this was written: the script prints the row keys, and if it cannot map them it saves `data/tonycwang_sample_raw.jsonl` instead.
* `--leon` -> `data/leon/` (one file per folder, each <= 50 MB).
* `--book` -> `tools/pons/7x6.book` (optional solver opening book).
* Each step skips existing output and a failing step never stops the others (exit code 1 if any failed).

**Primary ground truth:** the 3,000 solver-labeled positions in `accuracy_positions.csv` (below) are already included in the repo and are
what the accuracy experiments use. The regenerated-UCI path (`label_uci.py`, section "uci/") is now optional and superseded by the real UCI download.

**Solver on Windows:** `tools/pons/c4solver` is a Linux binary. On Windows compile it yourself (MinGW `g++`, or WSL: `cd tools/pons && make`).
The solver is only needed to regenerate labels, not to use the committed data.

---

All labels here are produced locally with Pascal Pons' perfect-play solver (`tools/pons`, `tools/solver.py`).
Spec deviation: the spec (section 6.2) calls for the HF `TonyCWang/ConnectFour` dataset (solver-labeled, 7 column scores)
and the UCI file. `huggingface.co` and `archive.ics.uci.edu` are unreachable from the build environment, and no
GitHub mirror of `connect-4.data` was reachable either (code search is blocked, only the Pons repo is readable). We
therefore generated equivalent ground truth ourselves: same kind of labels (exact per-column solver scores), our own sample of positions.

## accuracy_positions.csv  (3,000 rows, ~230 KB, committed)
Columns: `moves` (1-indexed columns, first player moves first), `ply`, `scores` (JSON list of 7 Pons scores for the
side to move, `null` = full column), `optimal_cols` (JSON list of 0-indexed columns with the max score), `source`
(`random` or `heuristic`), `bucket` (ply bucket).
* Score convention: 0 draw, + win, - loss for the side to move; |score| = 22 - winner's stones at the end (higher = faster win). An immediate win at n stones on board = (43-n)//2.
* 500 positions in each ply bucket 8-11, 12-15, 16-19, 20-23, 24-27, 28-30. Non-terminal, no winner, deduplicated up to
  transposition and left-right mirror. Sources: 1,388 random legal play, 1,612 heuristic (take win / block / centre-biased).
* Positions needing > 5 s to solve were skipped (22 candidates in the 8-11 bucket), so the hardest early positions are slightly under-represented.
* Outcome of the position under perfect play: 1,900 win / 234 draw / 866 loss for the side to move. 1,928 positions have a unique optimal column.
* Regenerate: `python experiments/data_prep/make_accuracy_set.py --n 3000 --workers 2 --seed 12345` (~6 min on 2 cores; deterministic given the seed).

## uci/  (regenerated UCI Connect-4)
* `positions.csv` (git-ignored, ~9 MB): all **67,557** positions. Columns `moves` (one move sequence reaching the board, 1-indexed
  columns), `label` (win / loss / draw for the first player, **blank if not yet labeled**), `c0..c41`: 42-cell encoding in
  UCI order: columns a..g, rows 1..6 bottom-up (`c0..c5` = a1..a6, `c6..c11` = b1..b6, ...), values `x` = first player, `o` = second player, `b` = blank.
* Regeneration definition (verified by exact count = 67,557): all distinct legal 8-ply boards with no winner where neither player has
  an immediate winning move, deduplicated under left-right mirror symmetry (a mirror pair counts once, the representative is arbitrary, so
  the set matches UCI as a set of positions-up-to-symmetry but may contain the mirror image of some UCI rows).
* Labels: weak solve (`c4solver -w`), sign of the first player's score. Cost is ~0.65-1.2 s/position/core with no opening book, i.e. ~10-20 core-hours for all
  67,557, so only a random sample was labeled here (see counts below). `data/uci/connect-4.data` (original UCI line format
  `b,b,...,x,o,win`) is written only when all rows are labeled.
* Continue labeling (resumable): `python experiments/data_prep/label_uci.py --workers 2 --limit N` (the first N positions of a fixed random order, seed 0, so any prefix is a random sample); drop `--limit` for everything.
  `--merge-only` just rewrites `positions.csv` from the shard files `data/uci/_shard*.tsv`.
* Helper: `experiments/data_prep/enum_uci.py` enumerates positions and prints the counts (182,383 non-won 8-ply boards -> 134,934 with no immediate threat -> 67,557 up to mirror).
