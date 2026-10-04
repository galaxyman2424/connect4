# Bot arena: open-source Connect 4 bots, head to head

Goal: put real, independently written Connect 4 bots in one round-robin tournament (plus our
minimax and our neural network) and measure how often each beats each other.

## 1. The line-up

All third-party code is downloaded **unmodified, pinned to an exact commit**, by
`python -m arena.fetch`, into `arena/third_party/` (git-ignored: we don't re-publish other
people's code; `SOURCE.json` in each folder records the URL, commit and file hashes).

| name | bot | author / licence | algorithm | settings used |
|---|---|---|---|---|
| `pons_perfect` | [PascalPons/connect4](https://github.com/PascalPons/connect4) @ d6ba50d | Pascal Pons, AGPL-3.0 (C++) | exact solver: negamax + alpha-beta on bitboards, transposition table, null-window search, opening book | best score, centre-first ties |
| `bruneton_az` | [jpbruneton/Alpha-Zero-algorithm-for-Connect-4-game](https://github.com/jpbruneton/Alpha-Zero-algorithm-for-Connect-4-game) @ 6cd76be | Bruneton, Douin, Reverdy; BSD-3 | AlphaZero: 1 res-block x 256 filters (1.2M params), pretrained weights in repo ("almost perfect" per README), PUCT MCTS | 200 simulations (their play script) |
| `plkmo_az` | [plkmo/AlphaZero_Connect4](https://github.com/plkmo/AlphaZero_Connect4) @ 01ff24a | plkmo, Apache-2.0 | AlphaZero: 19 res-blocks x 128 ch, pretrained "iter8" net, PUCT MCTS with root noise | 777 reads (their setting) |
| `alfo_mcts` | [Alfo5123/Connect4](https://github.com/Alfo5123/Connect4) @ d371203 | Alfredo de la Fuente, MIT | classic UCT MCTS with uniformly random roll-outs | 3000 iterations, factor 2.0 (their GUI) |
| `galli_minimax` | [KeithGalli/Connect4-Python](https://github.com/KeithGalli/Connect4-Python) @ 503c0b4 | Keith Galli, MIT | minimax + alpha-beta, window heuristic (the popular YouTube tutorial) | depth 5 (their game) |
| `easyai_negamax` | [Zulko/easyAI](https://github.com/Zulko/easyAI) @ 43e9462 | Zulko, MIT | easyAI's Negamax + alpha-beta on its ConnectFour example; scores only win/loss | depth 5 (their example) |
| `kaggle_negamax` | [Kaggle/kaggle-environments](https://github.com/Kaggle/kaggle-environments) @ eb8b5ef | Kaggle, Apache-2.0 | ConnectX built-in "negamax" opponent: depth 4, no pruning, "adjacent own stones" leaf score | as shipped |
| `ours_minimax_d4` / `_d8` / `ours_level10` | this repo | | our negamax (TT, ordering, bit-parallel heuristic) | depth 4 / 8 / website level 10 |
| `nn:<model>:mcts:200` | this repo | | our trained network + MCTS | 200 sims |
| `random` | | | uniformly random legal move | sanity check |

Pons' solver is the ceiling: it never makes a mistake, so it can only lose from an opening
that's already lost for its side (see section 3).

## 2. Adapters (`arena/adapters.py`)

Every bot has its own board format: row 0 = top or bottom, pieces as 1/2, 1/-1, "O"/"X" or
bitboards, "AI" vs "player" instead of first/second. Each adapter translates our position
into the bot's format, calls the bot's **own search function**, and translates the column
back. The only changes to third-party code:

| bot | change, and why |
|---|---|
| galli | the file opens a pygame window and starts a game at import, so we parse it and run only its function and constant definitions |
| alfo_mcts | Python 2 + Tkinter: run only the game/MCTS part, expand tabs the way Python 2 did, and turn one debug `print [...]` (Python 2 syntax) into `pass` |
| plkmo_az | its search calls `.cuda()` unconditionally; on a machine without CUDA we make that a no-op so it runs on the CPU |
| others | none |

**Verification that the translation is right** (`tests/test_lab.py::test_arena_adapters_win_in_one`,
plus a manual run over the 52-position tactics suite): every bot is shown positions with an
immediate win. Kaggle, easyAI, Alfo's MCTS, Pons and ours take 13/13, and Bruneton's
AlphaZero takes 12/13. Two bots fell clearly short, and we traced both to the bots
themselves, not the adapters:
* **Galli** took 10/13 immediate wins. In the other 3 it played a move that *still wins
  later* (the solver confirms it), because its minimax scores every win the same
  (1e14), with no preference for faster ones.
* **plkmo AlphaZero** took only about 6/13 immediate wins, even with its full 777 reads. We
  rebuilt the same positions with plkmo's own `drop_piece` and got identical boards, so the
  translation is correct. Their MCTS backs up the *network's* value even at finished games
  instead of the true result, so it can't "see" a win its network misjudges.

Those are both real findings worth a sentence in the presentation.

## 3. Protocol (`arena/tournament.py`)

* **Openings:** `--openings` random 2-4 ply openings (default 20, distinct up to mirroring,
  fixed seed). Every pair plays every opening **twice, with colours swapped**. That's needed
  because most bots are deterministic: from the empty board two bots would replay one game
  forever. Colour swapping cancels first-move advantage within a pairing.
* **Randomness:** before each game Python's `random`, numpy's and torch's RNGs are seeded from
  (seed, pairing, opening, colour), so results are reproducible. One exception: Bruneton's
  MCTS re-seeds itself from the OS on every call (their code), so its tie-breaks aren't
  reproducible.
* **Forfeits:** an exception or illegal move loses that game and is listed.
* **Parallel and resumable:** `--workers` processes. Every finished game is appended to
  `games.csv`, and `--resume` skips what's done.
* **Statistics:**
  * score = (wins + draws/2) / games;
  * head-to-head matrix;
  * **Bradley-Terry Elo** (max likelihood), `kaggle_negamax` = 1000, with **95% bootstrap
    CIs**: 500 resamples of whole openings within each pairing, the same method as
    `experiments/tournament.py`;
  * time per move (mean, max).
  * Bots that won or lost *every* game have no finite rating; they're reported as
    "undefeated" or "never scored" instead.

Size: 9 bots -> 36 pairings x 40 games = 1,440 games (20 openings). Add our two networks:
55 pairings, 2,200 games. The slow bots are plkmo (777 reads, about 1-2 s/move on the GPU,
about 7 s on a CPU) and Bruneton (200 sims, about 0.6 s/move). With 14 workers on the
5800X, expect roughly 1-3 hours.

## 4. Running it

```bash
pip install -r arena/requirements.txt         # torch + tqdm for the two AlphaZero bots
python -m arena.fetch                         # download the bots
cd tools/pons && make c4solver && cd ../..    # Pons solver (+ tools/pons/7x6.book)
python -m arena.tournament --list             # check everything is "ready"
python -m arena.tournament --openings 20 --add-nn nn:runs/az_6x64/model.npz:mcts:200
```

Useful options: `--bots a,b,c` (custom line-up), `--set plkmo_az.reads=200` (change a
setting; it's recorded in the results), `--name x` (separate results folder), `--resume`.

Results: `experiments/results/arena/<name>/games.csv` and `summary.json`, shown on the
Lab page's **Bot arena** tab: leaderboard with Elo and CIs, Elo chart, strength vs time
per move, head-to-head heatmap, and a card per bot with source, licence and settings.

## 5. Interpreting

* **Score vs Elo:** score depends on who else is in the field; Elo adjusts for it.
  Overlapping CIs mean "not clearly different".
* **Speed vs strength:** search budgets differ wildly (Kaggle about 50 ms, plkmo about 1 s+), so
  the scatter plot matters as much as the ranking.
* **First-player score:** Connect 4 is a first-player win with perfect play; among
  imperfect bots the advantage is smaller.
* Everything is conditioned on the openings and settings used; say so when presenting.

## 6. Pilot run (CPU, reduced, for checking the setup only)

`experiments/results/arena/pilot_cpu/` is a small tournament run on a 2-core cloud VM:
4 openings x 2 colours per pairing (8 games per pair, 360 games), **plkmo_az at 200 reads
instead of 777**, and the tiny pilot network `runs/pilot_cpu_az` (not a real training run).
It shows the plumbing works; the confidence intervals are far too wide for conclusions.

| bot | score | Elo (Kaggle = 1000) | ms/move |
|---|---|---|---|
| Pons solver | 95% | 2436 | 48 |
| Bruneton AlphaZero (200 sims) | 76% | 2135 | 331 |
| Alfo UCT MCTS (3000 it.) | 65% | 2012 | 790 |
| ours minimax d4 | 61% | 1969 | 1 |
| ours minimax d8 | 60% | 1962 | 23 |
| Galli minimax d5 | 55% | 1903 | 139 |
| pilot NN + MCTS 200 | 52% | 1872 | 176 |
| easyAI Negamax d5 | 21% | 1244 | 106 |
| Kaggle negamax | 13% | 1000 | 30 |
| plkmo AlphaZero (200 reads) | 1% | 612 | 2090 |

Early observations to check in the full run:
* Pons lost 2 and drew 3 of 72: those games started from openings that are already lost or
  drawn for its side. That's the effect of the random openings, not a solver mistake.
* plkmo's pretrained net is very weak here, consistent with it missing immediate wins
  (section 2). The full run uses its own 777 reads.
* Depth 8 doesn't beat depth 4 in this tiny sample. The 2,800-game depth tournament
  (`experiments/results/elo.json`) does show d8 > d4, which is why the full arena uses
  20+ openings.

Delete the pilot folder once you have the full tournament (`rm -r experiments/results/arena/pilot_cpu`).
