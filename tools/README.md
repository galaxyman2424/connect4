# tools/

## pons/ - Pascal Pons' Connect 4 solver
Sources copied from https://github.com/PascalPons/connect4 (commit d6ba50d), **AGPL-3.0** (see `pons/LICENSE`).
Only change: `Solver.hpp` lets the transposition-table size be set with `-DC4_TABLE_SIZE=n` (default 24, unchanged),
and the Makefile has an extra `c4solver_big` target (2^28 table; measured no faster, kept for reference).

Build: `cd tools/pons && make c4solver` (g++ -O3, C++11). The binary `c4solver` is git-ignored.

CLI: reads one move sequence per line on stdin (1-indexed columns, e.g. `4453362211`), prints one line per input.
* `./c4solver`      -> `<moves> <score>`
* `./c4solver -a`   -> `<moves> s1 s2 ... s7` (score of playing each column; -1000 = column full)
* `-w` weak solver (only the sign of the score is right: win/draw/loss; faster). `-b file` opening book.
* Invalid input (illegal move, or a game that is already won) prints an error on **stderr** and **nothing on stdout**,
  so a naive reader blocks forever; `solver.py` pre-validates to avoid that.

Score convention (side to move): 0 = draw, positive = win, negative = loss. `|score| = 22 - (stones the winner has
placed when the game ends)`; an immediate win with n stones already on the board scores `(43 - n) // 2`
(e.g. 18 at ply 6, 21 on an empty board is the theoretical max). Larger = faster win / slower loss.
In `-a` mode the column score is from the point of view of the side that moves *now* (i.e. already negated child score).

Opening book: **no `7x6.book` is available** here. The repo contains no book file (only `OpeningBook.hpp`; the
`book` git tag has no data), the GitHub releases page/API is not reachable, and blog.gamesolver.org is blocked.
Without it, positions at ply <= ~10 are slow (see timings below); the solver warns "Unable to load opening book" on stderr.
If you obtain the book (~ 7x6.book from the solver's website), drop it at `tools/pons/7x6.book` and `solver.py` uses it
automatically.

## solver.py
`Solver(analyze=True, weak=False, timeout=None)` keeps one persistent `c4solver` subprocess.
* `solve_all(moves) -> list[7]` (None for full columns; raises `SolverError` for illegal/already-won sequences or a timeout, after which the subprocess is restarted)
* `optimal_columns(moves) -> set[int]` 0-indexed columns with max score
* `Solver(analyze=False).solve(moves)` -> single score; `weak=True` -> sign only.
* Module-level `solve_all(moves)` uses a shared default instance. `python tools/solver.py` runs the self-test.
Python 3.10 compatible, stdlib only. Not thread-safe across processes: create one `Solver` per worker process.
