# Connect 4 Bot — CSCI 4150 Intro to AI Final Project (RPI, Fall 2026)

A local website where you play Connect 4 against our own AI, plus everything we built around it to
answer one question: **what actually makes a Connect 4 bot strong, and what does that strength cost?**

The project grew in four parts:

1. **Classic search** — a minimax bot (negamax + alpha-beta, transposition table, iterative deepening,
   hand-crafted window heuristic) with a 10-level difficulty slider, and experiments measuring how
   strength, accuracy and cost change with search depth.
2. **Our own neural network** — a small ResNet trained two ways (supervised on perfect-play solver
   labels, and AlphaZero-style self-play with MCTS), and a round-robin **arena** against open-source bots.
3. **Five more learning methods** — evolution (CMA-ES), TD(λ) n-tuple networks, DQN, PPO self-play and
   knowledge distillation, each measured for strength *and* training cost (hours, kWh, $).
4. **Research phase** — a 48-run supervised sweep (model size × teacher), interpretability of the
   network's layers, and a push to raise move accuracy from ~91 % to **97.5 %** of perfect play.

No LLM is used anywhere in the bot. (We tried an LLM as a move agent early on: it hallucinated board
states and removed pieces that were already on the board, so we dropped the idea.)

**Team:** Connor Thelen, Maggie, Andrew

---

## Contents

- [Headline results](#headline-results)
- [Quick start](#quick-start)
- [The website](#the-website)
- [Part 1 — Minimax and the depth experiments](#part-1--minimax-and-the-depth-experiments)
- [Part 2 — Neural network and the bot arena](#part-2--neural-network-and-the-bot-arena)
- [Part 3 — Other learning methods and training cost](#part-3--other-learning-methods-and-training-cost)
- [Part 4 — Research phase](#part-4--research-phase)
- [Part 5 — Pushing move accuracy past 95 %](#part-5--pushing-move-accuracy-past-95-)
- [Hardware and lessons learned](#hardware-and-lessons-learned)
- [Caveats](#caveats)
- [Repository layout](#repository-layout)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Credits and licences](#credits-and-licences)

---

## Headline results

| | Result |
|---|---|
| Best bot overall (arena, 100 openings) | Pascal Pons' perfect solver, Elo 1877 |
| Best bot we trained | 3-net **score-head ensemble, no search**: Elo 1812, tied with our supervised net + 200 MCTS simulations (1810), at **~32 ms/move vs ~280 ms** |
| Our net vs open-source AlphaZero (Bruneton) | Our supervised 6x64 + MCTS200 is ~+200 Elo stronger in the 100-opening arena (1810 vs 1609) |
| Move accuracy vs perfect play (3,000 held-out positions) | minimax depth 10: 91.8 % · supervised 6x64 policy: 90.7 % · score-head 8x128 + 1-ply lookahead: **97.5 %** (99.3 % never changes the game result) |
| Why the supervised net beats our self-play net | Perfect solver labels give a much better policy prior (0.90 vs 0.73 optimal without search); 200 MCTS simulations close most of that gap in play. The lead (~26 Elo) is **not yet statistically significant** (sign test p = 0.14). |
| Cheapest strong method | PPO warm-started from the supervised net: 0.23 GPU-h, ~$0.004 of electricity |

Arena Elo is anchored on our minimax at depth 4 (= 1409) or Bruneton's AlphaZero (= 1609, 100-opening
runs); the method-comparison "ladder" Elo is a **different scale** (anchored on minimax depth 1 = 1000).

---

## Quick start

Python 3.10 or newer (the training PC runs 3.12).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # website + experiments (numpy, flask, ...)
python server/app.py               # open http://localhost:5000
```

The website only needs numpy — no PyTorch. Training needs `pip install -r requirements-train.txt`.
Every command in order is in [`COMMANDS.md`](COMMANDS.md).

```bash
python -m pytest -q                # full test suite, ~10 s; PyTorch tests skip if torch isn't installed
```

---

## The website

| Page | What it shows |
|---|---|
| `/` **Play** | Play against the bot. Pick a level (1 = very easy … 10 = strongest, ~2 s/move), choose who goes first, click a column. Under the board the bot explains itself ("searched N positions in M ms"). An **Opponent: Neural net** switch appears once a trained `runs/<run>/model.npz` exists. |
| `/lab` **Lab** | Neural-net training charts (`runs/<run>/metrics.jsonl`), the hypothesis test (`eval.json`), the bot arena (`experiments/results/arena/*/summary.json`), the method comparison (Elo, accuracy, ms/move, GPU-h, kWh, $), and a "How it works" tab. Pages show the exact command to run when there's no data yet. |
| `/research` **Research** | The paper-prep results: overview, the top-15 roster, "why is our net the best", the supervised sweep, interpretability, and a paper outline — choose which parts of the project to look at. |

Screenshots are in [`docs/screenshots/`](docs/screenshots/).

---

## Part 1 — Minimax and the depth experiments

**Engine** (`engine/`): bitboard board, negamax with alpha-beta, transposition table, iterative
deepening, move ordering, and a hand-crafted heuristic that scores every 4-cell window. The shared
engine API used by the website and experiments is in [`CONTRACT.md`](CONTRACT.md).

**Evaluation plan** (from the [project statement](connect4-project-statement.md)): does the site enforce
the rules and play full games without crashing, and how does strength change with depth?

| Experiment | Result |
|---|---|
| Round robin, depths 1–8 (2,800 games), Bradley–Terry Elo with bootstrap CIs | d1 1000 → d2 1272 → **d3 1207** → d4 1389 → … → d8 1500. Depth 3 is *weaker* than depth 2 — an odd/even horizon effect (`experiments/oddeven.py`). Gains flatten after depth 4 (+19 to +44 per ply). |
| Accuracy vs perfect play (3,000 solver-labelled positions, plies 8–30) | d1 72.9 % → d4 87.3 % → d8 90.5 % → d10 91.8 % optimal moves. |
| Cost | d1 0.09 ms/move → d4 1.3 ms → d10 62 ms (mean). |
| Difficulty slider (10 levels) | Every level beats a random player (L1 84 %, L4+ 100 %); in every adjacent pairing the higher level wins or ties (L7 vs L8 is even). |
| Tactics suite (52 positions: wins in 1/2/3, forced blocks) | Wins-in-1 are found at depth 1; every forced block from depth 2 (depth 1 misses 4 of 13). |
| Embedding-neighbour artifact | Board states embedded and compared with cosine nearest neighbours; shows that *structurally similar* positions often do **not** share a best move. |

Charts: `experiments/figures/` (accuracy, cost, Elo vs depth, Elo vs cost, odd/even, slider levels,
tactics, win-rate heatmap, embedding neighbours). All of them are produced by
`experiments/plots.py` or `experiments/analysis.ipynb`.

---

## Part 2 — Neural network and the bot arena

**Network** (`nn/`, details in [`docs/NEURAL_NET.md`](docs/NEURAL_NET.md)): a small ResNet (default
6 blocks × 64 channels, ~0.46 M parameters) with a 7-column policy head and a win/draw/loss value head.
Input is 4 planes (my stones, opponent stones, ones, "am I the first player"). Batched PUCT MCTS;
numpy-only inference of exported `model.npz` so the website doesn't need PyTorch.

- **Phase A — supervised** (`nn.make_dataset`, `nn.train_supervised`): positions labelled by Pascal
  Pons' perfect solver (benchmark positions held out). We sometimes called this "PPO — Pons perfect
  optimizer"; it is *not* the PPO reinforcement-learning method in Part 3.
- **Phase B — AlphaZero self-play** (`nn.train_selfplay --preset quick|standard|long`, resumable).
- **Evaluation** (`nn.evaluate`): accuracy vs perfect play, value accuracy, tactics, games vs minimax
  d1–d8 → Elo, and the **hypothesis test**: does the NN's value inside our minimax beat the hand-crafted
  heuristic at the same depth (≥ 50 % head-to-head, at least as many wins-in-1/forced blocks, < 2 s/move)?

**Arena** (`arena/`, protocol in [`docs/ARENA.md`](docs/ARENA.md)): round robin against open-source bots
pinned to exact commits — Pons solver, Bruneton AlphaZero, plkmo AlphaZero, Alfo UCT-MCTS, Keith Galli
minimax, easyAI negamax, Kaggle ConnectX negamax — plus our minimax and every network we trained.
Bradley–Terry Elo with cluster-bootstrap CIs over openings.

**Full 20-bot arena (part 3, 7,599 games)** — top of the table:

| Bot | Elo [95 % CI] |
|---|---|
| pons_perfect | 1789 [1741, 1838] |
| **nn_sup_6x64 + MCTS200 (ours)** | **1761** [1718, 1814] |
| nn_az_6x64 + MCTS200 (ours) | 1718 [1667, 1772] |
| nn_ppo_from_sup policy (ours) | 1591 |
| bruneton_az (open source) | 1581 |
| nn_ppo_6x64 + MCTS200 (ours) | 1572 |
| distilled 2x32 students (ours) | 1517–1543 |
| alfo_mcts / ours_minimax_d8 | 1454 / 1444 |
| ours_minimax_d4 (anchor) | 1409 |

At the bottom: both TD n-tuple bots, easyAI, Kaggle negamax and plkmo_az (which looks broken — its
MCTS backs up the network value at terminal states and misses immediate wins).

---

## Part 3 — Other learning methods and training cost

`rl/` adds five methods, all wired into the Lab page and the arena (see
[`docs/OTHER_METHODS.md`](docs/OTHER_METHODS.md)):

| Method | File | Idea |
|---|---|---|
| Evolution (CMA-ES) | `rl/evolve.py` | Tunes 10 heuristic weights (5 hand-crafted features + centre, threat cells, odd/even threats); fitness from games or solver accuracy. |
| TD(λ) n-tuple network | `rl/ntuple.py` | 70 random 8-cell tuples, mirror-shared lookup tables, CPU only. |
| DQN | `rl/dqn.py` | Double DQN self-play, negamax Bellman target, replay buffer, target net. |
| PPO self-play | `rl/ppo.py` | GAE, clipped ratio, entropy bonus, opponent pool of past snapshots. |
| Distillation | `rl/distill.py` | 2x32 student learns from one or more teachers (soft-label KL + WDL). |
| Cost meter | `rl/meter.py` | Wall time, CPU core-h, GPU-h, energy (nvidia-smi + RAPL/estimate), $ at $0.20/kWh. |
| Comparison | `rl/compare.py` | One evaluation of every run → `experiments/results/methods/main/`. |

**Method comparison** (ladder Elo, d1 = 1000; optimal = best-move rate on the 3,000-position set):

| Method | Search | Elo | Optimal | ms/move | Training cost |
|---|---|---|---|---|---|
| Supervised 6x64 | MCTS200 | **2180** | **0.940** | 291 | 0.76 h |
| Supervised 6x64 | none | 1528 | 0.907 | 2.9 | 0.76 h |
| PPO warm-started from supervised | MCTS200 | 1932 | 0.834 | 249 | 0.23 GPU-h, $0.004 |
| AlphaZero self-play 6x64 | MCTS200 | 1645 | 0.834 | 318 | 12.3 h, 2.3 kWh, $0.46 |
| AlphaZero self-play 6x64 | none | 1547 | 0.727 | 2.9 | |
| Distilled 2x32 (from AZ) | none | 1579 | 0.724 | 0.8 | 0.19 h (+ teacher) |
| PPO from scratch 6x64 | MCTS200 | 1523 | 0.738 | 320 | 0.20 GPU-h |
| Hand-crafted heuristic | minimax d4 | 1415 | 0.870 | 0.9 | — |
| Evolved heuristic (CMA-ES) | minimax d4 | 1377 | 0.872 | 1.6 | 0.01 h |
| DQN 6x64 | none | 1356 | 0.838 | 2.9 | 0.52 GPU-h |
| TD n-tuple | minimax d4 | 1177 | 0.823 | 8.0 | 0.46 h CPU |
| TD n-tuple | greedy | 696 | 0.589 | 0.2 | |

Takeaways: search helps most when the prior is already good (MCTS200 adds +653 ladder Elo to the
supervised net but only +97 to the self-play net); a supervised warm start makes PPO far better than
PPO from scratch; distillation keeps most of the strength at ~1/9 the parameters and ~3x the speed.
Combination experiments C1–C6 (learning + search, evolved vs hand weights, warm starts, ensemble
teachers) are described in `docs/OTHER_METHODS.md`.

---

## Part 4 — Research phase

Prepared for a research paper; full notes with every table and source in
[`docs/RESEARCH.md`](docs/RESEARCH.md), a generated outline in
[`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md), interactive view at `/research`. Code in `research/`,
results in `experiments/results/research/`.

**Roster:** the top 15 bots of the 7,599-game arena (`python -m arena.tournament --roster top15`).

**Arena hang — fixed.** Part 3's arena once waited ~59 h for a game that a dead worker never finished,
and MCTS nets took 67 s/move because each worker ran 16 BLAS threads. Now: 1 thread per worker, 10 s
move limit, 300 s game limit, forfeits recorded, scheduled-vs-recorded check, `--resume`. The clean
top-15 rerun finished 4,200/4,200 games with 0 forfeits in 47.5 min.

**Arena with 100 openings** (more openings → tighter statistics):

| Bot | Elo [95 % CI] | ms/move |
|---|---|---|
| pons_perfect | 1877 [1855, 1898] | 30 |
| **3-net score-head ensemble, no search (ours)** | **1812** [1792, 1832] | **32** |
| nn_sup_6x64 + MCTS200 (ours) | 1810 [1790, 1830] | 280 |
| nn_az_6x64 + MCTS200 (ours) | 1804 [1783, 1825] | 270 |
| nn_ppo_from_sup policy (ours) | 1610 | 2 |
| bruneton_az (anchor) | 1609 | 326 |
| nn_ppo_6x64 + MCTS200 (ours) | 1609 | 276 |
| nn_distill_az_2x32 policy (ours) | 1559 | 0.7 |

(`experiments/results/arena/top8_o100_mcts/`, 5,600 games, 0 forfeits. A full top-15 + ensemble
run of 24,000 games gave the same picture: ensemble 1828, tied with sup+MCTS200.)

**Why our supervised net is the best neural net**
- Supported: highest Elo of all neural bots; most accurate policy without search (~0.90 vs 0.73 for
  the self-play net); fewest result-changing blunders (sup 2.0 %, our AZ 2.8 %, Bruneton 5.8 %).
- Not yet established: its lead over our own AlphaZero net. With 100 openings: 93 wins, 34 draws,
  73 losses (score 0.55), sign test p = 0.14.
- Not perfect: it scores ~0.41 against the solver.

**Supervised sweep (48 runs on the RTX 3050)** — `research/sweep*.py`
- Size (on solver labels): 1x16 0.79 → 2x32 0.87 → 3x48 0.89 → 4x64 0.90 → 6x64 0.91, then flat
  through 6x96, 8x128, 10x128 (~0.91–0.92). Bigger stopped helping → data-limited, not capacity-limited.
- Teachers: student accuracy tracks teacher accuracy (r = 0.97–0.98). Heuristic alone 0.73–0.76,
  minimax depth 4 (slider 5) 0.87–0.88; students match their teacher and don't beat it.
- Evolved heuristic with no search: generation 10 is best (0.757); longer evolution doesn't help.
- Distilling from a big net vs training directly: differences ≤ 0.012, about the seed-to-seed range.

**Interpretability (4 nets, 50 k positions)** — `research/interp*.py`
- Row/column geometry is already linear in the raw input; the net keeps it rather than computing it.
- Own and opponent immediate threats become decodable in blocks 1–3.
- The net's own move becomes readable steadily through the blocks; the opponent's best reply barely does.
- No "early-game block" in the supervised net — blocks split by position type (block 2 wins, blocks
  5–6 blocks). The self-play net does lean on early blocks for early positions.
- Causal test: own-threat channels in block 3 are specific versus random channels; patching the
  "opponent wins at column j" direction changes the policy.

---

## Part 5 — Pushing move accuracy past 95 %

**Diagnosis of the ~91 % ceiling:** (1) only 285 k training positions — the 8x128 net overfits
(train policy loss 0.41 vs val 0.78); (2) the policy target is "uniform over best columns" and the value
head is WDL only, so nothing teaches the net *how fast* a win is. Of the best old net's misses, only
2.2 % changed the game result; the rest were slower wins / faster losses.

**Fix:** 3 M solver-labelled positions (2.7 h to label) and a new **per-column score head** that
predicts all 7 Pons scores, trained on 8x128 nets with 3 seeds and a hard-example mining round.

| Player (3,000 held-out positions) | Optimal | Keeps result |
|---|---|---|
| minimax depth 10 | 0.918 | 0.951 |
| supervised 6x64 policy | 0.903 | 0.947 |
| best old net (10x128 + mirror averaging) | 0.932 | 0.970 |
| old 6-net ensemble | 0.942 | 0.975 |
| score_8x128, single net (3 seeds, mirror) | 0.969–0.972 | ~0.99 |
| **score_8x128 + 1-ply score lookahead** | **0.975** | **0.993** |
| 3-net score-head ensemble | 0.974 | 0.990 |

Result-changing mistakes dropped from ~3 % to under 1 %. Hard-example mining and ensembling gave no
further gain on the benchmark; the next steps toward 99 % are search guided by the score head, or a
10x128 now that data is no longer the limit. Run folders `runs/score_8x128_*` and training logs in
`logs/` are committed; the score-head training code lives on the `eval-bar-and-accuracy` branch
(which also adds a live chess-style evaluation bar to the play page).

We also looked at LLM-driven weight editing (MIT "weight-space programming", Nov 2026) and judged it
unlikely to help: it fixes single localised bugs, while our errors are spread out and we already have
a perfect teacher for gradient descent.

---

## Hardware and lessons learned

Code is written on a Windows PC; training runs on a Linux Mint 22.3 PC (Ryzen 7 5800X, 16 GB RAM,
RTX 3050 8 GB + GTX 750 Ti, NVIDIA driver 580 / CUDA 13.0). See
[`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md).

- **The wrong GPU.** With `CUDA_DEVICE_ORDER=PCI_BUS_ID`, device 0 was the old GTX 750 Ti, so
  AlphaZero generations 1–53 silently ran on the CPU (~11.8 h) and only gens 54–100 on the RTX 3050
  (0.5 h). `nn/hw.py` now hides the 750 Ti and repairs `CUDA_VISIBLE_DEVICES`; `python -m nn.hw` prints
  a report (`--bench` to benchmark).
- **GPU batching.** `nn/gpu_server.py` lets one GPU process batch evaluations for all CPU self-play
  workers (15 workers + server by default).
- **Arena reading a model while it was being written.** The first arena ran during AZ training and
  loaded `model.npz` mid-overwrite; the arena now snapshots network files at start.
- **Thread oversubscription and missing timeouts** — see the arena-hang fix above.
- **TD n-tuple saturation.** The textbook tanh-derivative TD update pushed every value to ±1 and
  learning stopped; a cross-entropy-style step and γ = 0.98 per ply fixed it.

---

## Caveats

- Most runs are single seeds; the first arenas used 20 openings (two near-trivial). All players are
  deterministic, so repeated arenas mostly confirm, not resample.
- Ladder Elo above ~1700 is extrapolated (ladder top is minimax d8 = 1500); 2180 for sup+MCTS200 is not
  comparable to arena Elo.
- The A2 midterm deck's "12.3 GPU-h" for AlphaZero should read **11.8 h CPU + 0.5 h GPU**, and its
  "Ours: net (Oct 5) 1386" is a mix of early generations, not one snapshot.
- Interpretability test positions for the supervised net probably overlap its training set.
- The supervised-vs-self-play comparison differs in data, amount of training and checkpoint
  selection, not just the training signal.

---

## Repository layout

```
engine/       bitboard board, heuristic, minimax search, agents + slider levels
server/       Flask app: serves web/, /api/move, /api/status, Lab API, Research API
web/          Play page, Lab page, Research page (HTML/JS/CSS)
nn/           neural net: model, MCTS, numpy inference, datasets, supervised + self-play training,
              evaluation, hardware detection (hw.py), GPU batching server (gpu_server.py)
rl/           evolution, TD n-tuple, DQN, PPO, distillation, cost meter, method comparison
arena/        open-source bot adapters, fetch script, registry, round-robin tournament
research/     sweep, teachers, arena analysis, interpretability, paper outline
experiments/  depth tournament, tactics, accuracy, cost, odd/even, embedding, plots,
              analysis.ipynb; data_prep/; results/ (CSV/JSON); figures/ (PNG)
data/         solver-labelled 3,000-position accuracy set + public-dataset download script
tools/        Pascal Pons' perfect-play solver (C++ source, AGPL-3.0) + Python wrapper
runs/         one folder per training run (config, metrics, exported model, cost)
logs/         training / labelling / arena logs for the score-head work
tests/        pytest suite (rules, search, server, lab, nn, rl, hardware, arena, research)
docs/         guides below + screenshots/
```

| Doc | Contents |
|---|---|
| [`COMMANDS.md`](COMMANDS.md) | Every command in order (website, training, arena, moving results, other methods) |
| [`docs/NEURAL_NET.md`](docs/NEURAL_NET.md) | Network design, method, metrics |
| [`docs/OTHER_METHODS.md`](docs/OTHER_METHODS.md) | Evolution, n-tuple, DQN, PPO, distillation, cost, combinations |
| [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md) | Step-by-step for the Linux GPU PC |
| [`docs/ARENA.md`](docs/ARENA.md) | Open-source bots, tournament protocol |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | Research-phase findings, claims, threats to validity, next experiments |
| [`docs/PAPER_OUTLINE.md`](docs/PAPER_OUTLINE.md) | Generated paper outline with every number sourced |
| [`docs/PRESENTATION_GUIDE.md`](docs/PRESENTATION_GUIDE.md) | Story, slide plan, likely questions |

---

## Reproducing the experiments

Run from the project root. Every script takes `--help`, is seeded, uses all CPU cores
(`--workers N` to change), and writes CSV/JSON to `experiments/results/` and PNGs to
`experiments/figures/`. Times measured on a 2-core cloud VM.

| Step | Command | Time (2 cores) | Needs solver? |
|---|---|---|---|
| Round robin depths 1–8 + slider levels | `python experiments/tournament.py` | ~9 + ~10 min | no |
| Tactics suite (52 positions × depths 1–8) | `python experiments/tactics.py` | ~5 s | only with `--regenerate` |
| Accuracy vs perfect play (3,000 positions, depths 1–10) | `python experiments/accuracy.py` | ~3.5 min | no (labels are in the CSV) |
| Cost of strength + ablations + per-level ms | `python experiments/cost.py` | ~3.5 min | no |
| Odd vs even depth | `python experiments/oddeven.py` | 1 s | no |
| Embedding-neighbour artifact | `python experiments/embedding.py` | ~1 min | optional |
| All charts | `python experiments/plots.py` or run `experiments/analysis.ipynb` | ~15 s | no |

Neural nets, other methods and the arena (GPU PC; full details in `COMMANDS.md`):

```bash
pip install -r requirements-train.txt
python -m nn.hw                                              # check which GPU/threads will be used
python -m nn.make_dataset --positions 300000                 # needs tools/pons/c4solver + 7x6.book
python -m nn.train_supervised --run sup_6x64                 # Phase A: learn from perfect play
python -m nn.train_selfplay --run az_6x64 --preset standard  # Phase B: AlphaZero self-play
python -m nn.evaluate --run az_6x64                          # evaluation + hypothesis test
python -m rl.compare                                         # every method: Elo, accuracy, cost
python -m arena.fetch && python -m arena.tournament --roster top15
```

Quick tournament smoke run (doesn't overwrite committed results):
`python experiments/tournament.py --openings 5 --l10-openings 1 --out smoke/results.csv`.

### Public datasets (optional)

The committed experiments use our own solver-labelled set (`data/accuracy_positions.csv`, see
[`data/README.md`](data/README.md)). To also test on public data:

```bash
pip install -r data/requirements-data.txt
python data/download_data.py --all                 # or --uci / --tonycwang 5000 / --leon / --book
python experiments/accuracy.py --source tonycwang
python experiments/accuracy.py --source uci
python experiments/embedding.py --corpus uci
```

### The solver (Windows note)

`tools/pons/c4solver` is built from Pascal Pons' C++ solver (git-ignored; build it yourself). It is only
needed to create *new* labels; the website, tests and every other experiment run without it. On
macOS/Linux: `cd tools/pons && make c4solver`. On Windows use WSL — the scripts look for exactly
`tools/pons/c4solver`. Without the opening book (`7x6.book`, see `tools/README.md`) positions with fewer
than ~12 stones are slow to solve.

---

## Credits and licences

- Perfect-play solver: [Pascal Pons, connect4](https://github.com/PascalPons/connect4) (AGPL-3.0),
  used for training labels, grading and as an arena bot.
- Arena opponents are downloaded at pinned commits by `python -m arena.fetch` and are not
  redistributed here: Bruneton AlphaZero, plkmo AlphaZero, Alfo UCT-MCTS, Keith Galli minimax,
  easyAI negamax, Kaggle ConnectX negamax. See [`docs/ARENA.md`](docs/ARENA.md).
- Course: CSCI 4150 Introduction to Artificial Intelligence, Rensselaer Polytechnic Institute, Fall 2026.
