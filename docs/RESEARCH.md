# Research notes: small supervised Connect 4 nets, teachers, and what is inside them

Working paper-prep notes for the CSCI 4150 (Intro to AI) project. Written 2026-11-16 on branch `research-phase`.
This file is a map of the research phase and a source of numbers for the paper, not the paper.

How to read the numbers: every number below was copied from a data file or computed from one with a short script.
The source is named next to each table (file, then the JSON path or CSV column). Where the data do not support
a conclusion, the text says so next to the claim. The CSV/JSON files are the authority; if this file and a data
file disagree, the data file wins and this file is wrong.

## 1. Purpose and status

Purpose: decide what can honestly be claimed in a short research paper built on the existing Connect 4 project
(solver, minimax, supervised / AlphaZero / PPO / DQN / distilled nets, open-source bot arena), and record how every
claim was produced.

Status on 2026-11-16 (all four research experiments have finished and written their result files):

| Experiment | Files | When | Size | Command / note |
|---|---|---|---|---|
| Arena re-analysis | `experiments/results/research/arena_analysis.json`, `roster.json` | generated 2026-11-16T00:17:40 | 7,599 part3 games, 20 bots; Bradley-Terry Elo, 500-rep cluster bootstrap; every move solver-judged | `python -m research.arena_analysis` (analysis runtime 19.8 s) |
| Supervised sweep | `experiments/results/research/sweep.json`, `sweep_runs.csv` | started 2026-11-16T00:10:24, updated 2026-11-16T02:56:53 | 48 of 48 runs done, 0 failed, 0 queued (`status.counts`) | `python -m research.sweep --run` |
| Interpretability | `experiments/results/research/interp.json`, `interp_channels.json` | generated 2026-11-16 00:27:28 | 4 nets, 50,000 positions (10,000 test), runtime 4330 s (from interp.json notes) | `python -m research.interp` |
| Clean top-15 arena rerun | `experiments/results/arena/top15/summary.json`, `games.csv` | created 2026-11-16T01:07:55 | 4,200 of 4,200 games recorded, 0 missing, 0 duplicates, no forfeits; 2850.2 s wall, 6 workers x 1 thread | `python -m arena.tournament --roster top15 --name top15 --openings 20 --workers 6` |
| Original arena (part3) | `experiments/results/arena/part3/summary.json`, `games.csv` | created 2026-10-21T22:14:59 | 7,599 games of 7,600 scheduled (one lost), 14 workers, wall 43.86 h (arena_analysis.json -> health.wall_hours) | run once, 2026-10-19..21; see section 6 |

Source: the `source`, `status`, `generated` and `meta` fields of the files named in the table.

Branch and tree state (read-only `git` checks, 2026-11-16): branch `research-phase`, HEAD `bae85f2` (2026-11-15).
The working tree has many uncommitted changes (86 paths in `git status --short`), including modified `arena/tournament.py`,
`arena/registry.py`, `docs/ARENA.md`, and untracked `research/`, `experiments/results/research/`, `experiments/results/arena/top15/`
and all `runs/rs_*` folders. The results below were therefore produced by code that is not in any commit yet.
Commit before the paper numbers are frozen.

### Hardware (from the `config.json` / `meta.machine` fields)

| run | phase | device used | GPUs listed by torch | CPU threads | platform | torch |
|---|---|---|---|---|---|---|
| sup_6x64 | supervised | cpu | NVIDIA GeForce GTX 750 Ti | 16 | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 | 2.14.1+cu130 |
| az_6x64 | selfplay | cpu | NVIDIA GeForce GTX 750 Ti | 16 | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 | 2.14.1+cu130 |
| ppo_6x64 | ppo | cuda:0 | NVIDIA GeForce RTX 3050, NVIDIA GeForce GTX 750 Ti | 16 | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 | 2.14.1+cu130 |
| ppo_from_sup | ppo | cuda:0 | NVIDIA GeForce RTX 3050, NVIDIA GeForce GTX 750 Ti | 16 | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 | 2.14.1+cu130 |
| dqn_6x64 | dqn | cuda:0 | NVIDIA GeForce RTX 3050, NVIDIA GeForce GTX 750 Ti | 16 | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 | 2.14.1+cu130 |
| distill_az_2x32 | distill | cuda:0 | NVIDIA GeForce RTX 3050, NVIDIA GeForce GTX 750 Ti | 16 | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 | 2.14.1+cu130 |
| evo_d2 | evolution | cpu | NVIDIA GeForce RTX 3050, NVIDIA GeForce GTX 750 Ti | 16 | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 | 2.14.1+cu130 |
| td_ntuple | ntuple | cpu | NVIDIA GeForce RTX 3050, NVIDIA GeForce GTX 750 Ti | 16 | Linux-7.0.0-38-generic-x86_64-with-glibc2.39 | 2.14.1+cu130 |
| rs_* (all 48 sweep runs) | supervised | NVIDIA GeForce RTX 3050 | NVIDIA GeForce RTX 3050 | 16 | Linux-7.0.0-43-generic-x86_64-with-glibc2.39 | 2.14.1+cu130 |

Source: `runs/<run>/config.json` -> `phase`, `device` / `device_name`, `machine.*`. All 48 sweep runs share the same
device string (`NVIDIA GeForce RTX 3050`; one distinct value over `runs[].device` in sweep.json).
Tournaments: top15 `meta.machine` = Linux-7.0.0-43-generic-x86_64-with-glibc2.39, 16 CPUs; part3 `meta.machine` = Linux-7.0.0-38-generic-x86_64-with-glibc2.39, 16 CPUs.
The CPU model and RAM are not recorded in any result file; `docs/TRAINING_RUNBOOK.md` describes the machine as a Ryzen 7 5800X
(8 cores / 16 threads), 16 GB RAM, RTX 3050 8 GB (+ GTX 750 Ti, not used by PyTorch).

Two things to keep in mind about timing: (1) the original `sup_6x64` and `az_6x64` runs used `device: cpu` in their
configs, the sweep used the GPU, so training times are not comparable between them; (2) the sweep (00:10-02:56), the
arena analysis (00:17), interp (finished 00:27) and the top-15 rerun (01:07-01:55) overlapped in time on the same
16-thread machine (timestamps above), so every ms-per-move number measured during those windows carries load noise.

## 2. Research questions

The owner's two directions.

**A. Plain supervised nets (no self-play).** A student network learns to copy a teacher on a fixed set of positions.

- A1. How small can the network get before accuracy collapses? (size ladder)
- A2. What happens with different network shapes on a 1:16 blocks:channels ladder (1x16, 2x32, 3x48, 4x64, 6x96, 8x128; 6x64 is the
  project's original size and sits off the ladder; 10x128 is an optional extra)? Why is 36 blocks x 4096 channels ("6^2 x 64^2") not
  part of the ladder? Answer from `sweep.json -> notes`: it would have about 10.9 billion parameters
  (36 * 2 * 9 * 4096^2) and is not trainable on the RTX 3050 (8 GB). 8x128 (2,380,642 parameters, `sweep.json -> sizes`) is the top rung that was trained.
- A3. How much does the teacher matter? Teachers: the Pons solver (perfect labels); the hand-written heuristic alone with no search;
  the same heuristic with weights evolved for 10 / 20 / 40 / 60 generations, still with no search; minimax depth 4 on the hand heuristic
  (= difficulty slider level 5) and on the generation-60 heuristic; PPO-trained nets (`ppo_net`, `ppo_from_sup`).
- A4. Small vs big distillation: does a small student learn better from a trained big net (6x64 or 8x128) than directly from the solver labels?
- A5. Why is our network "the best neural net"? (Arena question: is the supervised 6x64 net with MCTS200 really ahead of the other
  neural nets, including the project's own AlphaZero net and an open-source AlphaZero net, and why?)

**B. Interpretability.** What do the layers and channels encode: board geometry (row/column), threats (own vs opponent),
own move vs the opponent's reply, game phase? Which blocks and channel groups does the output depend on (knock-out, group ablation,
direction patching), and how do nets trained differently compare (CKA)?

## 3. Method summary (enough for a methods section)

### 3.1 Data and metrics

- **Training positions.** `nn_data/solver_300k.json`: 300000 positions, seed 2026, each labelled by Pascal Pons' solver with the exact score
  of every column (labelling took 1287.0 s wall on 14 workers, 0 timeouts). Outcome for the side to move: win 185,507 (61.8 %), draw 20,403 (6.8 %), loss 94,090 (31.4 %).
  Positions come from games of random mixes of move pickers, deduplicated up to mirroring, with the benchmark positions removed (`docs/NEURAL_NET.md` section 5).
  Fixed split: 285,000 train / 15,000 validation (`runs/*/config.json -> train_positions, val_positions`), split seed 0.
- **Benchmark.** `data/accuracy_positions.csv`: 3,000 held-out solver-scored positions, plies 8-30, six buckets of 500 (8-11, 12-15, 16-19, 20-23, 24-27, 28-30);
  1,612 have `source = heuristic`, 1,388 `source = random`. 2,134 of the 3,000 are not already lost for the side to move (`n_not_lost` in sweep.json).
  The benchmark never enters training or checkpoint selection (`docs/NEURAL_NET.md`).
- **`optimal`** = share of benchmark positions where the chosen column is one of the solver's max-score columns.
  **`keeps_nl`** = among positions that are not already lost, share where the chosen column keeps the game-theoretic result (win stays win, draw stays draw);
  it is 1 minus the blunder rate. Definitions: `experiments/accuracy.py` module docstring. Both use the policy head alone (no search) unless a row says MCTS.
  `value_acc` = argmax of the WDL head equals the solver outcome.
- **Benchmark noise.** `sweep.json -> notes`: standard error of an `optimal` rate near 0.85 on n = 3,000 is about 0.0065.

### 3.2 Student training recipe (sweep, `nn/train_supervised.py` through `research/sweep_train.py`)

All students are `nn.model.C4Net` (residual tower of `blocks` x `channels`, 7 policy logits, WDL head), trained on the same 300,000 positions
with the same split; only teacher and size differ (`sweep.json -> notes`). Batch 1024, AdamW lr 2e-3, weight decay 1e-4, OneCycle schedule,
mirror augmentation, bf16 autocast, 40 epochs for every run (`EPOCHS = {"default": 40}`), loss = CE(teacher policy) + 1.0 x CE(teacher WDL).
`best.npz` = lowest validation loss against the TEACHER's labels; solver labels are never used to select a non-solver student.
Student benchmark numbers use the policy head only.

The project's original `sup_6x64` (the net in the arena) was trained earlier with the same recipe but 30 epochs, on CPU, with periodic MCTS evaluation
(`runs/sup_6x64/config.json`). Its `best.npz` is epoch 18 by validation loss (0.90716; `runs/sup_6x64/metrics.jsonl`); validation loss rises after
epoch 18 while benchmark accuracy stays near 0.90-0.91.

### 3.3 Teachers and target construction (`research/teachers.py` docstring, `sweep.json -> notes`)

| teacher id | what labels the positions | policy target | value target |
|---|---|---|---|
| `pons` | the solver | uniform over the optimal columns | one-hot win/draw/loss under perfect play |
| `heur_hand` | hand heuristic alone, no search: play each legal column, score the position from the mover's view; a move completing four scores WIN-1; no must-block rule. Equals `engine.minimax.search(depth=1)` | uniform over the columns tied for the best score (hard) | tanh(s / sigma) of the best score, sigma = std of non-mate best scores; +-1 for proven wins/losses; mapped to (W, D, L). Monotone squashing, not a calibrated probability |
| `heur_evo_g10/20/40/60` | same, with the weight vector of generation N of `runs/evo_d2` (evolved with depth-2 search and game results) | same | same |
| `mm_d4_hand`, `mm_d4_evo60` | depth-4 negamax (`engine.minimax.search`), hand weights / generation-60 weights. `mm_d4_hand` = slider level 5 | same as heuristic | same as heuristic |
| `ppo_net`, `ppo_from_sup` | the trained net `runs/ppo_6x64` / `runs/ppo_from_sup`, policy head only, no search | the net's softmax over legal columns (soft) | the net's WDL output |
| `sup6x64`, `sup8x128` | `runs/sup_6x64/best.npz`, `runs/rs_A_pons_8x128/best.npz`, policy head only | soft | soft |

`ppo_from_sup` was initialised from `sup_6x64`'s best checkpoint (`runs/ppo_from_sup/config.json -> args.init`).

### 3.4 Gauntlet (second sweep metric)

Each student plays 40 openings (`nn.bench.random_openings`, seed 4150, 2-4 plies) x 2 colours against 7 opponents (minimax d1, d2, d4, d6; `mm_d4_evo60`;
`distill_az_2x32`; `sup_6x64`): 560 games, policy head only, argmax with a centre tie-break (`research/sweep_gauntlet.py`). `mean_score` is the average over the 7 opponents. Every player is deterministic,
so a game is fully fixed by (student, opponent, opening, colour); the intervals in `runs[].gauntlet.ci95` are over openings, not over randomness, and there is no run-to-run randomness to average out.

### 3.5 Arena protocol (`docs/ARENA.md`, `arena/tournament.py`)

Round robin; 20 random 2-4 ply openings (seed 4150, distinct up to mirroring) x 2 colours per pairing, so 40 games per pairing. Per-move timeout 10 s,
per-game timeout 300 s, workers limited to one BLAS/OpenMP thread set before numpy is imported. Elo = maximum-likelihood Bradley-Terry; 95 % intervals from a cluster bootstrap
(cluster = the two colour-swapped games of one opening in one pairing), 500 resamples (`arena_analysis.json -> source.elo_method`). Differences between two bots use a separate 2,000-resample bootstrap (seed 1).
Part3: 20 bots, 190 pairings, 7,599 of 7,600 games. Top-15 rerun: 15 bots, 105 pairings, 4,200 games (`top15/summary.json -> meta`), anchor `ours_minimax_d4` = 1409.
Neural bots with search use 200 PUCT simulations, a fresh tree per move.
Solver-judged blunders: every move of every part3 game was classified with Pons' solver in weak mode (win/draw/loss exact) (`best_nn.decomposition.f_blunders.meta`).

### 3.6 Interpretability protocol (`interp.json -> notes`)

Positions: fixed-seed (2026) random subset of `solver_300k.npz`, n = 50,000, split 70/10/20 -> 35,000 / 5,000 / 10,000 (probes are scored on the 10,000 test positions).
Models: `sup_6x64` (`runs/sup_6x64/best.npz`), `az_6x64`, `distill_az_2x32` (`runs/distill_az_2x32/best.npz`), `ppo_6x64`.
Forward pass re-implemented in torch from the exported weights; maximum absolute difference to `nn/numpy_net.py` is 0.0 for all four models (`interp.json -> checks.<model>.forward_max_abs_diff_vs_numpy`).

- *Probes:* linear (logistic / softmax / least-squares) on standardised flat activations (`C*6*7` features), full-batch L-BFGS, L2 1e-4. Controls: raw-input probe (layer `input`),
  spatially pooled probe, and a shuffled-label probe (labels permuted on train+val, test labels real). `first_layer` = first layer reaching 90 % of the best normalised skill.
  Concepts (21): column heights, landing row, ply, side-to-move parity, own / opponent immediate wins, own / opponent threat-square counts and double threats, odd / even-row threat squares,
  solver-optimal set, the net's own move, outcome, solver score, opponent's solver-optimal reply after the net's move, the net's move on the colour-flipped board, phase.
- *Channel census:* rule-based labels (threat / occupancy copy / column / row detector / phase / dead / mixed) from correlations and mean maps; thresholds in `interp.py`.
- *Knock-out:* skip a residual block (`identity`; `zero_branch` is identical because inputs are post-ReLU) or replace its branch by its test-set mean (`mean_branch`); report
  agreement with solver-optimal moves, ablated minus intact, split by phase (early ply < 12, mid, late ply >= 24) and position type (own immediate win, must-block, quiet).
- *Group ablation:* all channels of one census label in one layer set to zero or to their mean, compared with 30 random equal-size sets of live channels in the same layer (`pct` = share of null draws below the effect).
- *Direction patching:* probe direction for "opponent wins at column j" at the layer where that probe is best, step = gap between class means, added on quiet positions and subtracted on must-block positions;
  compared with 20 random directions of equal norm.
- *CKA:* linear CKA on 5,000 test positions between every layer of `sup_6x64` and every layer of the other nets.

## 4. Results

### 4.1 The top-15 roster: part3 Elo vs clean-rerun Elo

| bot | family | part3 Elo [95% CI] (anchor kaggle_negamax = 1000, 20 bots) | part3-games refit (anchor ours_minimax_d4 = 1409, 4,199 games) | clean rerun Elo [95% CI] (anchor d4 = 1409, 4,200 games) | rank part3 | rank refit | rerun score | rerun avg ms/move |
|---|---|---|---|---|---|---|---|---|
| pons_perfect | exact solver | 1788.5 [1740.8, 1838.4] | 1811.5 | 1811.5 [1770.3, 1851.1] | 1 | 1 | 0.8125 | 36.6 |
| nn_sup_6x64_mcts200 | supervised NN (solver labels) + MCTS | 1760.9 [1717.9, 1814.0] | 1783.3 | 1785.6 [1751.6, 1824.6] | 2 | 2 | 0.7884 | 305.1 |
| nn_az_6x64_mcts200 | self-play NN (AlphaZero) + MCTS | 1718.2 [1667.4, 1772.1] | 1750.1 | 1750.9 [1714.1, 1792.4] | 3 | 3 | 0.7536 | 303.7 |
| nn_ppo_from_sup_policy | RL NN (PPO, initialised from supervised), no search | 1591.0 [1546.9, 1642.3] | 1625.1 | 1621.6 [1585.7, 1660.1] | 4 | 4 | 0.6018 | 2.5 |
| bruneton_az | self-play NN (AlphaZero, open source) + MCTS | 1581.2 [1535.1, 1631.9] | 1601.9 | 1599.5 [1566.4, 1637.9] | 5 | 5 | 0.5732 | 357.9 |
| nn_ppo_6x64_mcts200 | RL NN (PPO self-play) + MCTS | 1572.1 [1526.1, 1622.6] | 1600.9 | 1599.5 [1562.4, 1636.1] | 6 | 6 | 0.5732 | 349.9 |
| nn_distill_az_2x32_policy | distilled NN (teacher: az_6x64 + MCTS100), no search | 1542.5 [1499.1, 1591.2] | 1560.3 | 1556.9 [1518.6, 1594.6] | 7 | 7 | 0.5170 | 0.8 |
| nn_distill_ens_2x32_policy | distilled NN (teacher ensemble), no search | 1517.1 [1472.5, 1568.1] | 1533.6 | 1541.5 [1508.9, 1578.3] | 8 | 8 | 0.4964 | 0.8 |
| alfo_mcts | MCTS with random roll-outs (open source) | 1454.2 [1401.5, 1500.3] | 1460.3 | 1460.3 [1425.3, 1495.0] | 9 | 9 | 0.3884 | 609.7 |
| ours_minimax_d8 | minimax + hand heuristic | 1443.5 [1400.5, 1489.7] | 1448.6 | 1448.6 [1414.0, 1484.4] | 10 | 10 | 0.3732 | 18.8 |
| evo_evo_d2_d4 | minimax + evolved heuristic (CMA-ES) | 1411.4 [1367.7, 1460.3] | 1418.4 | 1418.4 [1377.1, 1458.5] | 11 | 12 | 0.3348 | 1.9 |
| ours_minimax_d4 | minimax + hand heuristic | 1409.0 [1364.9, 1454.0] | 1409.0 | 1409.0 [1409.0, 1409.0] | 12 | 13 | 0.3232 | 1.1 |
| galli_minimax | minimax + hand heuristic (open source) | 1399.8 [1358.2, 1450.2] | 1403.2 | 1403.2 [1366.2, 1439.9] | 13 | 14 | 0.3161 | 131.3 |
| nn_ppo_6x64_policy | RL NN (PPO self-play), no search | 1399.8 [1352.3, 1448.6] | 1440.3 | 1440.3 [1403.6, 1476.5] | 14 | 11 | 0.3625 | 2.5 |
| nn_dqn_6x64_policy | RL NN (DQN), no search | 1367.5 [1319.9, 1412.9] | 1377.8 | 1377.8 [1339.4, 1415.7] | 15 | 15 | 0.2857 | 2.5 |

Sources: part3 Elo = `roster.json -> top15[].part3_elo/part3_lo/part3_hi` (equal to `part3/summary.json -> elo`); refit = `roster.json -> top15[].top15_elo` (= `arena_analysis.json -> elo_top15`, part3 games among the 15 bots);
rerun Elo and score = `top15/summary.json -> elo`, `per_bot.score`; rerun avg ms = `top15/summary.json -> per_bot.avg_ms`.

What the table shows. The top four bots (`pons_perfect`, `nn_sup_6x64_mcts200`, `nn_az_6x64_mcts200`, `nn_ppo_from_sup_policy`) and the bottom bot keep their ranks in all three columns. `bruneton_az` and `nn_ppo_6x64_mcts200` are tied at 1599.5 in the rerun.
The middle of the table moves because Elo depends on the pool: `nn_ppo_6x64_policy` is rank 14 in the 20-bot part3 fit and rank 11 in the 15-bot fits (refit and rerun both 1440.3), and `evo_evo_d2_d4`, `ours_minimax_d4`, `galli_minimax` each drop one rank as a result.
The rerun played all 4,200 scheduled games with no forfeits and no missing or duplicate games (`top15/summary.json -> meta`), so the one part3 game that was lost (`bruneton_az` vs `nn_sup_6x64_mcts200`, opening 0) is replaced.
Elo values shift by a few points between the refit and the rerun, within the intervals. The anchor is `ours_minimax_d4` = 1409 in both fits; part3 values are anchored differently (kaggle_negamax = 1000), so compare ranks and differences, not absolute levels across columns.

Time per move: the MCTS200 nets averaged 305.1 ms (sup), 303.7 ms (az) per move in the rerun against 67422.8 and 65560.5 ms in part3 (`part3/summary.json -> per_bot.avg_ms`; also `best_nn.decomposition.g_time_confound`).
The earlier 67-second moves were a thread-oversubscription artefact (`docs/ARENA.md`, "Why part3 took 44 h"), now fixed in the tournament code. Since the fixed-simulation search plays the same moves regardless of speed, strength is not expected to change,
and the head-to-head results of the rerun do match part3 (next section).

**The rerun is mostly a replay, not an independent sample.** Matching the 4,199 games that exist in both runs by (pair, opening, who moved first) (computed from `part3/games.csv` and `top15/games.csv`):
the winner is the same in 4,134 games (98.5 %), the move list is identical in 3,944 (93.9 %). Among the 3,640 games without `bruneton_az` the move list is identical in 3,426; among the 559 games with it, in 518
(`docs/ARENA.md` notes that Bruneton's MCTS re-seeds from the OS, so its games are not reproducible). Same openings, same seed, deterministic bots: the rerun shows that the ranking is not an artefact of the thread-oversubscription / hang problems and it closes the missing game,
but it adds almost no new statistical information about the 20-opening noise. The small differences in non-Bruneton games are probably floating-point differences from the different BLAS thread count; that was not checked.

### 4.2 Why is the supervised net the best neural net?

"Best neural net" here means highest arena Elo among the neural-network bots (ours and open source), every bot playing every other from the same 20 openings x 2 colours
(`best_nn.sense`). Net under test: `nn_sup_6x64_mcts200` = `runs/sup_6x64/best.npz` (458,338 parameters) + PUCT MCTS with 200 simulations.

**Sup vs each rival, five analyses.** Elo differences are Bradley-Terry bootstrap intervals (positive = sup ahead). Head-to-head = 40 games (20 openings x 2 colours; 39 for Bruneton in part3 because one game was lost);
the score interval is a bootstrap over openings; p = exact two-sided sign test on decisive games.

| rival | sup W-D-L (part3) | sup score [CI] (part3) | sign test p | (1) part3, 20 bots | (2) part3 games, top-15 refit (500 reps) | (3) part3 without openings 18, 19 | (4) clean rerun, 20 openings | (5) clean rerun without 2 trivial openings |
|---|---|---|---|---|---|---|---|---|
| az_6x64 + MCTS200 (ours, self-play) | 23-2-15 | 0.600 [0.487, 0.725] | 0.2559 | +42.7 [+6.4, +81.8] | +33.2 [-9.3, +71.6] | +55.3 [+13.1, +98.3] | +34.7 [-3.5, +70.7] | +45.7 [+6.0, +89.1] |
| ppo_from_sup (policy only) | 22-8-10 | 0.650 [0.550, 0.750] | 0.0501 | +169.8 [+134.1, +207.0] | +158.2 [+122.1, +194.0] | +212.5 [+174.2, +254.2] | +164.1 [+128.2, +199.9] | +199.6 [+162.9, +240.9] |
| bruneton_az (open source AlphaZero) | 27-5-7 | 0.756 [0.637, 0.868] | 0.0008 | +179.6 [+143.2, +217.2] | +181.4 [+146.0, +217.7] | +222.9 [+184.5, +264.9] | +186.2 [+149.9, +220.7] | +224.0 [+184.5, +268.4] |
| pons_perfect (solver) | 13-7-20 | 0.412 [0.338, 0.475] | 0.2962 | -27.7 [-63.2, +8.4] | -28.2 [-65.4, +5.2] | -36.8 [-74.4, +1.5] | -25.9 [-61.0, +8.7] | -35.8 [-75.5, +2.6] |

Sources: (1) `best_nn.pairwise[].elo_diff`, head-to-head from `best_nn.pairwise[]`; (2) `elo_top15_meta.differences_vs_best` (500 resamples); (3) `elo_sensitivity_no_trivial_openings.differences` (2,000 resamples; excluded openings 18 and 19, ids of the openings `3344` and `4626`);
(4) and (5) computed for this document from `experiments/results/arena/top15/games.csv` with `research.arena_analysis.elo_bootstrap` / `diff_ci` (2,000 resamples, seed 1, anchor `ours_minimax_d4` = 1409; snippet in section 8). Columns (4) and (5) are **not stored in any repo JSON**.
Rerun head-to-head W-D-L computed from `top15/games.csv`: sup vs az 23-2-15, sup vs pons 13-7-20, sup vs bruneton 27-6-7, sup vs ppo_from_sup 22-8-10 (same as part3 apart from the extra Bruneton game; see the replay note in 4.1).

**The verdict as `arena_analysis.json -> best_nn.answer` states it** (copied; the numbers inside are the part3 ones):

- *supported*
  - Among the 19 non-solver bots of part3, nn_sup_6x64_mcts200 has the highest point-estimate Elo (1761, CI 1718-1814). Its lead is significant over every neural net except our own az_6x64 (see suggestive). Against the open-source AlphaZero net bruneton_az the gap is +180 Elo (bootstrap CI +143 to +217, P>0 = 1.000) and head-to-head 27-5-7 (score 0.756, CI 0.637-0.868).
  - Better than our PPO/DQN/distilled nets: sup - ppo_from_sup = +170 Elo (CI +134 to +207).
  - Its policy head alone is the most accurate move predictor we have: optimal 0.907 vs 0.727 (az), 0.786 (ppo_from_sup), 0.838 (dqn) on the same 1000 solver positions; with MCTS200 0.940 vs 0.834 (az). (Note dqn: high 'optimal' but low arena Elo - per-position accuracy alone does not rank bots.)
- *suggestive*
  - sup vs az (our self-play net): +43 Elo, bootstrap CI +6 to +82, P(sup>az) = 0.988; head-to-head 23-2-15, score 0.600 (CI 0.487-0.725), sign test p = 0.256. Top-15 pool: +33 (CI -9 to +72). Without the two near-trivial openings: +55 (CI +13 to +98). Not robust: the interval includes 0 in at least one analysis and/or the direct match is not significant, so 'better than our AlphaZero net' is NOT established; the full-pool gap comes partly from results against weaker bots.
  - Elo cannot separate sup from perfect play (sup - pons = -28, CI -63 to +8), but sup is not perfect: against pons (40 games, expected score 0.5 from the opening values) it scored 0.412 (13-7-20), i.e. it gave away 3.5 of 20 available points; az_6x64 gave away 5.5 and bruneton_az 11.5. The Elo tie only means the 20-opening pool cannot tell them apart (pons' Elo is capped by its lost openings).
  - Solver blunder rates (fraction of own moves that change the game-theoretic result): sup 0.0203, az 0.0282, bruneton 0.0584, ppo_from_sup 0.0562. These are over different positions (each bot reaches its own), so they are descriptive, not a controlled comparison.
  - Why (hypothesis): supervised training on perfect solver labels gives a better prior and value than 12.3 h of self-play at this size, and MCTS200 adds most where the prior is good (ladder +653 for sup vs +97 for az). Confounds: sup and az differ in data, training length, and the sup net was selected on bench_optimal (best.npz); one run each, no seeds.
- *needed_experiments*: kept in section 7.

**Re-check of "sup is better than our own AlphaZero net" against the clean rerun.**
- Full 20 openings: sup - az = +34.7 Elo, 95 % CI [-3.5, +70.7], P(sup > az) = 0.959. The interval includes 0. Part3 pool gave +42.7 [+6.4, +81.8] (P = 0.988) and the part3 top-15 refit +33.2 [-9.3, +71.6] (P = 0.944).
- Direct games in the rerun: 23 wins, 2 draws, 15 losses, score 0.600, bootstrap interval [0.4875, 0.7125], sign test p = 0.2559. Not significant.
- Excluding the two near-trivial openings (a post-hoc choice): +45.7 [+6.0, +89.1], P = 0.9855; part3 gave +55.3 [+13.1, +98.3]. The interval excludes 0 only after an exclusion that the data themselves suggested, so it is a sensitivity check, not a confirmation.
- All five point estimates are positive (+33.2 to +55.3), but only some intervals clear zero and the direct match does not. Wording that the data support: *the supervised net is ahead of our AlphaZero net by about 30 to 55 Elo in these 20 openings, a lead the data suggest but do not establish.*
- Lead over the other neural nets is clear in every analysis: `bruneton_az` +186.2 [+149.9, +220.7] and `ppo_from_sup` +164.1 [+128.2, +199.9] in the rerun.
- Versus the solver: sup - pons = -25.9 [-61.0, +8.7] in the rerun. The interval includes 0, so Elo does not separate them, but that is not evidence of equality: sup scored 0.4125 against pons head-to-head (13-7-20, sign test p = 0.2962), and the interval for az - pons, -60.6 [-96.5, -23.9], excludes 0.

**Why: what the data say about the mechanism.**

Policy quality on 1,000 solver-scored benchmark positions (methods table; `best_nn.decomposition.a_policy_quality.rows`). `ladder_elo` is the methods-table Elo from ~100 games per opponent against minimax d1-d8 and extrapolates far beyond that ladder for the strongest bots (see caveat).

| run | search | optimal | keeps_nl | ladder Elo | ms/move | params |
|---|---|---|---|---|---|---|
| sup_6x64 | none | 0.907 | 0.9472 | 1527.5 | 2.931 | 458,338 |
| sup_6x64 | mcts200 | 0.940 | 0.9837 | 2180.2 | 290.837 | 458,338 |
| az_6x64 | none | 0.727 | 0.8604 | 1547.3 | 2.943 | 458,338 |
| az_6x64 | mcts200 | 0.834 | 0.9783 | 1644.5 | 318.153 | 458,338 |
| ppo_6x64 | none | 0.602 | 0.7764 | 1415.3 | 2.956 | 458,338 |
| ppo_6x64 | mcts200 | 0.738 | 0.9241 | 1522.6 | 319.998 | 458,338 |
| ppo_from_sup | none | 0.786 | 0.9106 | 1601.4 | 2.942 | 458,338 |
| ppo_from_sup | mcts200 | 0.834 | 0.9675 | 1931.8 | 249.296 | 458,338 |
| dqn_6x64 | none | 0.838 | 0.9106 | 1356.3 | 2.910 | 458,338 |
| distill_az_2x32 | none | 0.724 | 0.8862 | 1578.9 | 0.822 | 50,146 |
| distill_ens_2x32 | none | 0.744 | 0.8848 | 1513.0 | 0.824 | 50,146 |
| td_ntuple | none | 0.589 | 0.7371 | 696.1 | 0.184 | 9,175,040 |

Training signal and cost (part3 arena Elo; `best_nn.decomposition.c_training_signal_and_cost`). `sup_6x64` has no `cost.json`: its wall time is `metrics elapsed_sec`, and label generation (1,287.0 s on 14 workers; 0.357 h) is listed separately in the file and not included in 0.757 h.

| bot | training signal | arena Elo | wall h | energy kWh | cost source |
|---|---|---|---|---|---|
| nn_sup_6x64_mcts200 | perfect solver labels (300k positions) | 1760.9 | 0.757 |  | metrics elapsed_sec (no rl.meter) |
| nn_az_6x64_mcts200 | AlphaZero self-play (100 gens x 1000 games) | 1718.2 | 12.309 | 2.3032 | cost.json |
| nn_ppo_6x64_mcts200 | PPO self-play (300 iters) | 1572.1 | 0.196 | 0.0185 | cost.json |
| nn_ppo_from_sup_policy | PPO fine-tune of sup_6x64 | 1591.0 | 0.225 | 0.0213 | cost.json |
| nn_dqn_6x64_policy | DQN self-play | 1367.5 | 0.522 | 0.0507 | cost.json |
| nn_distill_az_2x32_policy | distillation of az_6x64+MCTS100 | 1542.5 | 0.194 | 0.0127 | cost.json |
| nn_distill_ens_2x32_policy | distillation of an ensemble | 1517.1 | 0.197 | 0.0129 | cost.json |
| evo_evo_d2_d4 | CMA-ES on game results | 1411.4 | 0.010 | 0.0008 | cost.json |
| ntuple_td_ntuple_d4 | TD(lambda) self-play | 1275.4 | 0.457 | 0.0031 | cost.json |

Solver-judged moves in part3 (`best_nn.decomposition.f_blunders.per_bot`; weak-solver mode; each bot faces different positions, so descriptive only). error = share of own moves that change the game-theoretic result; threw win = share of moves in won positions that give the win away.

| bot | moves judged | error rate | threw-win rate | points lost / 100 moves |
|---|---|---|---|---|
| pons_perfect | 9253 | 0.00 % | 0.00 % | 0.00 |
| nn_sup_6x64_mcts200 | 9360 | 2.03 % | 1.99 % | 1.24 |
| nn_az_6x64_mcts200 | 9982 | 2.81 % | 2.28 % | 1.79 |
| nn_distill_az_2x32_policy | 9861 | 4.70 % | 5.28 % | 3.10 |
| nn_ppo_from_sup_policy | 10332 | 5.62 % | 6.73 % | 3.77 |
| nn_ppo_6x64_mcts200 | 10355 | 6.17 % | 7.14 % | 4.05 |
| bruneton_az | 10340 | 5.84 % | 6.64 % | 3.91 |
| ours_minimax_d8 | 9849 | 7.94 % | 14.10 % | 5.54 |
| nn_dqn_6x64_policy | 9809 | 9.30 % | 16.72 % | 6.55 |

Reading these together:
- *Supported:* the supervised net has the best per-position move quality of our nets (policy-only `optimal` 0.907 vs 0.727 for az, 0.786 ppo_from_sup; 1,000 positions), and the lowest solver-judged error rate of any neural bot in the arena (2.03 %, vs 2.81 % az, 5.84 % bruneton). Its lead over Bruneton's net and over our PPO / DQN / distilled nets holds in every analysis in the table above (Bruneton and `ppo_from_sup`); against the other nets the part3 Elo differences (`best_nn.pairwise[].elo_diff`) are: `ppo_6x64_mcts200` +188.7 [+155.3, +226.0], `distill_az_2x32` +218.4 [+184.5, +253.8], `distill_ens_2x32` +243.8 [+211.1, +280.1], `ppo_6x64` policy +361.0 [+324.5, +401.1], `dqn_6x64` policy +393.4 [+356.0, +436.2], all with intervals above 0.
- *Suggestive only:* why. The hypothesis in the data file is that perfect solver labels give a better prior and value than 12.3 h of self-play at this size (az wall time 12.309 h in the cost table), and that MCTS200 adds most where the prior is good (ladder gain +652.7 for sup vs +97.2 for az, `b_search_gain`).
  Confounds: sup and az differ in training data, training length and checkpoint selection; one training run each, no seeds; the ladder Elo values (2180.2 for sup + MCTS) are far outside the ladder they were measured on (max d8 = 1500) and are not comparable to arena Elo. Policy-only sup and az were never in the arena, so the search gain could only be measured in the arena for `ppo_6x64` (+172.3, CI [137.5, 207.8]).
- *Not supported:* that per-position accuracy alone ranks bots. `nn_dqn_6x64_policy` has `optimal` 0.838 but the lowest arena Elo among the top 15 (1367.5; methods-table accuracy 0.838 vs `ppo_from_sup` 0.786 which is 223.5 Elo higher). Of the cheap predictors of arena Elo, methods-table `elo_vs_keeps_nl` has R^2 0.594 (n = 12) and `optimal` 0.315; both rest on 8-20 points from one run per method with wide slope intervals (`predictor_ranking`, `regressions`).
- *Not supported:* any statement about equal wall-clock strength. All MCTS bots used 200 simulations; speed differs by orders of magnitude between bots and the part3 timings were corrupted by thread oversubscription (`g_time_confound`: 232x slowdown for the MCTS200 nets, 168x for the 6x64 policy nets, explained from code and not re-measured inside the arena). In the rerun, sup + MCTS200 averages 305.1 ms/move and `bruneton_az` 357.9 ms/move, so the rerun is roughly equal in time, but that is an observation, not a controlled equal-time condition.
- *Inconsistency to fix before quoting:* `best_nn.answer.suggestive` says the sup net "was selected on bench_optimal (best.npz)". The training code selects `best.npz` by lowest validation loss (`nn/train_supervised.py`, line "if vs[0] < best_val", and `docs/NEURAL_NET.md` section 5), and `runs/sup_6x64/metrics.jsonl` has the best validation loss at epoch 18 (benchmark accuracy there 0.9043, the same value `sweep.json -> teachers[sup6x64].bench_optimal` reports). Use "selected by validation loss".
- Two benchmark figures for the same net appear in the files: 0.907 (methods table, a 1,000-position subsample: `a_policy_quality.positions`) and 0.9043 (3,000 positions; `sweep.json`, `interp.json -> checks.sup_6x64.bench_agreement`). They are different samples of the same benchmark, not a contradiction.

### 4.3 Noise first: seed-to-seed and benchmark noise

Read this before any small gap below. Only 14 configurations were trained with two seeds (`seed` 0 and 1; same data and split, different initialisation / batch order). Everything else is a single run.

| teacher | size | optimal seed 0 | optimal seed 1 | abs diff | gauntlet seed 0 | gauntlet seed 1 | abs diff |
|---|---|---|---|---|---|---|---|
| pons | 1x16 | 0.7870 | 0.7943 | 0.0073 | 0.3268 | 0.3527 | 0.0259 |
| pons | 2x32 | 0.8730 | 0.8820 | 0.0090 | 0.5366 | 0.5438 | 0.0072 |
| pons | 3x48 | 0.8887 | 0.8897 | 0.0010 | 0.6447 | 0.6696 | 0.0249 |
| pons | 4x64 | 0.9037 | 0.8947 | 0.0090 | 0.7232 | 0.7036 | 0.0196 |
| pons | 6x64 | 0.9117 | 0.9083 | 0.0034 | 0.7411 | 0.7313 | 0.0098 |
| heur_hand | 2x32 | 0.7327 | 0.7310 | 0.0017 | 0.1884 | 0.1875 | 0.0009 |
| heur_evo_g10 | 2x32 | 0.7563 | 0.7533 | 0.0030 | 0.2116 | 0.2134 | 0.0018 |
| heur_evo_g20 | 2x32 | 0.7440 | 0.7457 | 0.0017 | 0.2268 | 0.2357 | 0.0089 |
| heur_evo_g40 | 2x32 | 0.7383 | 0.7390 | 0.0007 | 0.2214 | 0.2089 | 0.0125 |
| heur_evo_g60 | 2x32 | 0.7277 | 0.7377 | 0.0100 | 0.2241 | 0.2009 | 0.0232 |
| mm_d4_hand | 2x32 | 0.8463 | 0.8530 | 0.0067 | 0.4598 | 0.4616 | 0.0018 |
| mm_d4_evo60 | 2x32 | 0.8483 | 0.8503 | 0.0020 | 0.4554 | 0.4679 | 0.0125 |
| ppo_net | 2x32 | 0.6207 | 0.6153 | 0.0054 | 0.5098 | 0.4991 | 0.0107 |
| ppo_from_sup | 2x32 | 0.7967 | 0.7993 | 0.0026 | 0.6795 | 0.7000 | 0.0205 |

Source: `sweep.json -> analysis.seed_noise`.

Summary (computed from the 14 rows): `optimal` differs between two seeds by 0.0045 on average and at most 0.0100 (max also stated in finding `sweep-seed-noise`); the gauntlet score differs by 0.0129 on average and at most 0.0259.
With two seeds per configuration these are rough ranges, not confidence intervals. Together with the benchmark standard error (about 0.0065 near 0.85, `sweep.json -> notes`), the working rule used below is:
**gaps in `optimal` smaller than about 0.01, and gaps in gauntlet score smaller than about 0.025, are within noise.** The 14 configurations are mostly 2x32; for sizes above 6x64 there is no second seed at all.

### 4.4 Size ladder on solver labels (A)

| size | params | optimal s0 | optimal s1 | keeps_nl s0 | value_acc s0 | gauntlet s0 | gauntlet s1 | agreement with Pons s0 | ms/pos (numpy, 1 thread) | best epoch s0 | train s (GPU, s0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x16 | 16,994 | 0.7870 | 0.7943 | 0.8796 | 0.8093 | 0.3268 | 0.3527 | 0.7997 | 0.301 | 40 | 33 |
| 2x32 | 50,146 | 0.8730 | 0.8820 | 0.9396 | 0.8643 | 0.5366 | 0.5438 | 0.8804 | 0.544 | 40 | 63 |
| 3x48 | 138,722 | 0.8887 | 0.8897 | 0.9419 | 0.8797 | 0.6447 | 0.6696 | 0.9054 | 0.886 | 26 | 138 |
| 4x64 | 310,370 | 0.9037 | 0.8947 | 0.9531 | 0.8843 | 0.7232 | 0.7036 | 0.9173 | 1.322 | 27 | 209 |
| 6x64 (off ladder) | 458,338 | 0.9117 | 0.9083 | 0.9620 | 0.8933 | 0.7411 | 0.7313 | 0.9165 | 2.519 | 25 | 295 |
| 6x96 | 1,013,474 | 0.9150 | - | 0.9588 | 0.8947 | 0.7072 | - | 0.9159 | 3.922 | 17 | 633 |
| 8x128 | 2,380,642 | 0.9130 | - | 0.9588 | 0.9023 | 0.7732 | - | 0.9275 | 8.714 | 20 | 1331 |
| 10x128 (optional extra) | 2,971,490 | 0.9147 | - | 0.9611 | 0.9060 | 0.7866 | - | 0.9285 | 6.001 | 19 | 1053 |

Source: `sweep.json -> runs[]` (`bench.optimal`, `bench.keeps_nl`, `bench.value_acc`, `gauntlet.mean_score`, `teacher_agreement`, `ms_per_pos`, `best_epoch`, `train_sec`), runs `rs_A_pons_*`. Parameter counts: `sweep.json -> sizes` (the exported file of `sup_6x64` has 457,500 parameters with BatchNorm folded, `interp.json -> models`).
Gauntlet opponents include `sup_6x64`, so the gauntlet is capped by how often a student can beat a 6x64 net.

- Fit over all 13 solver-label runs (both seeds pooled, so points are not independent): `optimal` rises by 0.0499 per tenfold increase in parameters, R^2 0.7191, bootstrap slope interval [0.0208, 0.0750]; gauntlet slope 0.1902, R^2 0.8457, interval [0.1139, 0.2579]
  (`analysis.size_regression.pons`). A straight line in log10(params) is a poor description: the gain is concentrated below 4x64.
- Small end: 1x16 (16,994 parameters) already reaches `optimal` 0.787 / 0.7943 (two seeds) against 0.7287 for the hand heuristic alone and 0.873 for depth-4 minimax on the same benchmark (`teachers[]`). 2x32 (50,146 parameters) reaches 0.873 / 0.882, the level of depth-4 minimax, with a gauntlet score of 0.5366 / 0.5438.
- Large end: from 6x64 up the values are 0.9117 (6x64), 0.915 (6x96), 0.913 (8x128), 0.9147 (10x128). The spread (0.0033) is smaller than the seed-to-seed range in 4.3, so the data show a plateau at about 0.91 from 6x64 on, not a ranking among these four.
  The 6x64 second seed (0.9083) is 0.0034 below its first.
- The largest nets stop improving early (best epoch 25 for 6x64, 17 for 6x96, 20 for 8x128, 19 for 10x128, out of 40) and validation loss against the teacher stops improving after that (for the original `sup_6x64` it rises after epoch 18, section 3.2); an `optimal` plateau near 0.91 may therefore reflect the 285,000 training positions as much as network capacity. This was not tested (no run with more positions).
- The gauntlet separates the large sizes slightly more (6x64 0.7411, 6x96 0.7072, 8x128 0.7732, 10x128 0.7866), but 6x96 is below 6x64 although it is larger, one seed each; the differences are 0.03-0.05, a little above the seed range of 0.026, and are not interpreted.
- Cost: 8x128 takes 1,331 s of GPU training against 33.1 s for 1x16 (`train_sec`), and 8.7144 ms/position against 0.3011 ms (single-thread numpy; other jobs were running on the machine at times, section 1).
- Why 36x4096 is out of reach: see section 2 (about 10.9 billion parameters, `sweep.json -> notes`).

Quality / cost frontier over all 48 runs (`analysis.pareto_params`, `pareto_ms`, as computed by `research/sweep_analysis.py`; I did not re-derive the dominance rule):

| by parameters: run | params | optimal |
|---|---|---|
| rs_A_pons_1x16_s1 | 16,994 | 0.7943 |
| rs_C_sup6x64_2x32 | 50,146 | 0.8833 |
| rs_C_sup6x64_3x48 | 138,722 | 0.9003 |
| rs_C_sup6x64_4x64 | 310,370 | 0.9123 |
| rs_A_pons_6x96 | 1,013,474 | 0.9150 |

| by time per position: run | ms/pos | optimal |
|---|---|---|
| rs_A_pons_1x16_s1 | 0.3007 | 0.7943 |
| rs_A_pons_2x32_s1 | 0.5349 | 0.8820 |
| rs_C_sup6x64_2x32 | 0.5361 | 0.8833 |
| rs_C_sup6x64_3x48 | 0.8683 | 0.9003 |
| rs_A_pons_4x64 | 1.3223 | 0.9037 |
| rs_C_sup6x64_4x64 | 1.3306 | 0.9123 |
| rs_A_pons_6x96 | 3.9225 | 0.9150 |

The frontier mixes distilled students (`rs_C_sup6x64_*` at 2x32, 3x48, 4x64) with solver-label runs. The distilled students are ahead of the direct solver-label runs of the same size by at most 0.012 (section 4.8), which is near the noise level in 4.3, so the frontier's membership at those sizes is not robust.

### 4.5 Teacher comparison (B), including student-vs-teacher correlation

Teacher quality is measured on the same 3,000-position benchmark as the students (`teachers[].bench_optimal`; the teacher's own move, no search for the heuristics beyond what the row says). `agree_pons` = share of the 300,000 training positions where the teacher's move is Pons-optimal. Student columns: `optimal` of the student on the benchmark; "s1" = second seed.

| teacher | kind | teacher optimal | teacher agree_pons | 2x32 student optimal (s0) | 2x32 (s1) | 6x64 student optimal (s0) | 2x32 student - teacher | 6x64 student - teacher | 2x32 gauntlet | 6x64 gauntlet |
|---|---|---|---|---|---|---|---|---|---|---|
| pons | solver | 1.0000 | 1.0000 | 0.8730 | 0.8820 | 0.9117 | -0.1270 | -0.0883 | 0.5366 | 0.7411 |
| sup8x128 | net | 0.9130 | 0.9606 | 0.8693 | - | - | -0.0437 | - | 0.5723 | - |
| sup6x64 | net | 0.9043 | 0.9404 | 0.8833 | - | - | -0.0210 | - | 0.5411 | - |
| mm_d4_evo60 | minimax | 0.8777 | 0.8406 | 0.8483 | 0.8503 | 0.8650 | -0.0294 | -0.0127 | 0.4554 | 0.4804 |
| mm_d4_hand | minimax | 0.8730 | 0.8337 | 0.8463 | 0.8530 | 0.8673 | -0.0267 | -0.0057 | 0.4598 | 0.4884 |
| ppo_from_sup | net | 0.7900 | 0.7949 | 0.7967 | 0.7993 | 0.7993 | +0.0067 | +0.0093 | 0.6795 | 0.7420 |
| heur_evo_g10 | heuristic | 0.7573 | 0.7282 | 0.7563 | 0.7533 | 0.7580 | -0.0010 | +0.0007 | 0.2116 | 0.2125 |
| heur_evo_g20 | heuristic | 0.7457 | 0.7153 | 0.7440 | 0.7457 | 0.7500 | -0.0017 | +0.0043 | 0.2268 | 0.2295 |
| heur_evo_g60 | heuristic | 0.7363 | 0.7037 | 0.7277 | 0.7377 | 0.7357 | -0.0086 | -0.0006 | 0.2241 | 0.2205 |
| heur_evo_g40 | heuristic | 0.7350 | 0.7057 | 0.7383 | 0.7390 | 0.7387 | +0.0033 | +0.0037 | 0.2214 | 0.2321 |
| heur_hand | heuristic | 0.7287 | 0.6842 | 0.7327 | 0.7310 | 0.7367 | +0.0040 | +0.0080 | 0.1884 | 0.1804 |
| ppo_net | net | 0.6027 | 0.6010 | 0.6207 | 0.6153 | 0.6243 | +0.0180 | +0.0216 | 0.5098 | 0.5384 |

Source: `sweep.json -> teachers[]`, `runs[]`, `analysis.teacher_vs_student` (the "student - teacher" columns are the `delta_optimal` fields, recomputed here from the two numbers beside them). Teachers `sup6x64` / `sup8x128` were only used for the 2x32 student here (group C trained them at 1x16-4x64, see 4.8); the 6x64 student of a `sup*` teacher was not trained.

**Does teacher quality predict student quality?**
- Across all teachers of one student size the correlation is high: 2x32, n = 12 teachers, Pearson r = 0.9659, slope 0.7043 (95 % interval [0.5343, 0.8607]); 6x64, n = 10 teachers, r = 0.9824, slope 0.7610 [0.6445, 0.9281] (`analysis.teacher_vs_student.<size>.student_vs_teacher`, `pearson_r`).
  Caveat: this correlation is carried by the wide spread of teacher quality (0.60 for `ppo_net` to 1.0 for Pons) and by the heuristic teachers, whose students reproduce them almost exactly (agreement 0.94-1.00 with the teacher). It says that a student cannot be much better than a weak teacher; it does not say that improvements in a teacher of similar quality pass through.
- At 1x16, 3x48 and 4x64 only three teachers (Pons, `sup6x64`, `sup8x128`, all with `optimal` 0.90-1.0) were used, and the correlations are r = -0.12, -0.67, -0.25 (n = 3 each, slope intervals include or sit near 0). With n = 3 and a 0.1-wide teacher range these numbers carry no information; do not quote them.
- The teachers differ in kind (hard solver / heuristic targets vs soft network targets, three different value-target definitions), so the regression mixes effects; the caveat attached to every `sweep-teacher-quality-*` finding says the same.

**Do students exceed their teachers?** (`delta_optimal` > 0)
- Heuristic and minimax teachers, 2x32: 2 of 7 exceeded (largest +0.0040, `heur_hand`: 0.7287 teacher, 0.7327 student); 6x64: 4 of 7 exceeded, largest +0.0080. Both are inside the 0.01 noise level (4.3) and the benchmark standard error (about 0.007), so the data show students matching, not beating, these teachers (finding `sweep-exceed-2x32`, `sweep-exceed-6x64`).
- Network teacher `ppo_net` (the weakest teacher, `optimal` 0.6027): students are above it in all three runs: 2x32 +0.0180 (seed 0) and +0.0126 (seed 1: 0.6153), 6x64 +0.0216. Same sign three times and larger than the noise range, but no significance test was run and the teacher number comes from the same 3,000 positions. Suggestive: a student trained on the soft targets of a weak PPO net on solver-sampled positions can be slightly more accurate than its teacher.
  `ppo_from_sup` (teacher 0.7900): +0.0067 (2x32 seed 0), +0.0093 (2x32 seed 1, 0.7993), +0.0093 (6x64): within noise.
- Strong teachers (Pons, `sup6x64`, `sup8x128`, both minimax teachers) are not matched: students fall 0.004 to 0.213 below them in every run but one, the gap shrinking with student size (Pons: 2x32 -0.127, 6x64 -0.0883; `sup6x64` 1x16 -0.1206, 2x32 -0.021, 3x48 -0.0040, 4x64 +0.0080). The 4x64 student of `sup6x64` is 0.008 above its teacher: inside the noise range.
- So: no student clearly exceeds a teacher that is itself accurate. Where a student is above its teacher the teacher is weak, and the margin is a few hundredths at most.

**Ranking by `optimal` is not the ranking by gauntlet.** At 2x32 the students of `mm_d4_hand` / `mm_d4_evo60` have `optimal` 0.8463 / 0.8483 but gauntlet 0.4598 / 0.4554, while the student of `ppo_from_sup` has `optimal` 0.7967 and gauntlet 0.6795, and the student of `ppo_net` 0.6207 and 0.5098. The pattern repeats at 6x64 (`mm_d4_hand` 0.8673 / 0.4884; `ppo_from_sup` 0.7993 / 0.7420, see the table). Students of network teachers (trained from self-play or from a supervised net) win more games against the fixed opponents than students of minimax teachers that agree more often with the solver.
Same observation in the arena: `nn_dqn_6x64_policy` has a high `optimal` and the lowest arena Elo (4.2). Possible explanations (per-position accuracy counts positions weighted differently from games; students of network teachers reproduce the opponent set's own style because two gauntlet opponents are nets) were not tested.

### 4.6 Evolved-heuristic generations (10 / 20 / 40 / 60) as no-search teachers

| teacher | generation | search depth | teacher optimal | teacher keeps_nl | agree_pons | 2x32 student optimal | 2x32 gauntlet | 6x64 student optimal | 6x64 gauntlet |
|---|---|---|---|---|---|---|---|---|---|
| heur_hand | 0 | 0 | 0.7287 | 0.8529 | 0.6842 | 0.7327 | 0.1884 | 0.7367 | 0.1804 |
| heur_evo_g10 | 10 | 0 | 0.7573 | 0.8702 | 0.7282 | 0.7563 | 0.2116 | 0.7580 | 0.2125 |
| heur_evo_g20 | 20 | 0 | 0.7457 | 0.8646 | 0.7153 | 0.7440 | 0.2268 | 0.7500 | 0.2295 |
| heur_evo_g40 | 40 | 0 | 0.7350 | 0.8580 | 0.7057 | 0.7383 | 0.2214 | 0.7387 | 0.2321 |
| heur_evo_g60 | 60 | 0 | 0.7363 | 0.8580 | 0.7037 | 0.7277 | 0.2241 | 0.7357 | 0.2205 |
| mm_d4_hand | 0 | 4 | 0.8730 | 0.9349 | 0.8337 | 0.8463 | 0.4598 | 0.8673 | 0.4884 |
| mm_d4_evo60 | 60 | 4 | 0.8777 | 0.9358 | 0.8406 | 0.8483 | 0.4554 | 0.8650 | 0.4804 |

Source: `analysis.evo_curve.rows` (generation 0 = the hand heuristic, `heur_hand`). Regressions on the five no-search generations (`analysis.evo_curve.regressions`): teacher `optimal` vs generation: slope -0.00009 per generation, 95 % interval [-0.00074, 0.00085], R^2 0.0394; 2x32 student: slope -0.00023 [-0.00060, 0.00057]; 6x64 student: slope -0.00018 [-0.00064, 0.00067].

What this shows: the first 10 generations raise the no-search teacher from 0.7287 (hand) to 0.7573 (+0.0286, several times the benchmark standard error of about 0.0065; no paired test was run). After generation 10 there is no further gain: generations 20, 40, 60 are 0.7457, 0.7350, 0.7363, all within 0.0107 of each other and all below generation 10; the fitted slope over generations is indistinguishable from 0.
The 2x32 students follow their teachers within 0.0086 (second seeds in 4.3 differ by up to 0.01). Student `optimal` after generation 10 is therefore flat or slightly falling; the gauntlet score of the students is 0.1884 (hand), 0.2116, 0.2268, 0.2214, 0.2241: all evolved-weight students are above the hand one by 0.023 to 0.038 (the smallest gap equals the largest seed-to-seed gauntlet difference of 0.0232 seen for these teachers, so only the larger gaps are clear), and generations 10-60 differ by 0.0152 or less.
Evolved weights as a teacher thus help once (hand to generation 10) and not afterwards.
Caveats: the weight vectors were evolved with depth-2 search and game results as fitness (`runs/evo_d2/config.json`: depth 2, fitness games), and are used here without search, so they were never optimised for the no-search role. Single seed per 6x64 student, two seeds for 2x32. The teacher benchmark has one measurement per generation.
With search (depth 4) the generation-60 weights give `optimal` 0.8777 against 0.873 for the hand weights: +0.0047, inside the noise.

### 4.7 Search vs no-search teachers

| heuristic | teacher type | teacher id | teacher optimal | 2x32 student optimal | 2x32 gauntlet | 6x64 student optimal | 6x64 gauntlet |
|---|---|---|---|---|---|---|---|
| hand | no search | heur_hand | 0.7287 | 0.7327 | 0.1884 | 0.7367 | 0.1804 |
| hand | depth-4 minimax | mm_d4_hand | 0.8730 | 0.8463 | 0.4598 | 0.8673 | 0.4884 |
| evolved gen 60 | no search | heur_evo_g60 | 0.7363 | 0.7277 | 0.2241 | 0.7357 | 0.2205 |
| evolved gen 60 | depth-4 minimax | mm_d4_evo60 | 0.8777 | 0.8483 | 0.4554 | 0.8650 | 0.4804 |

Source: `analysis.search_vs_nosearch`; findings `sweep-search-heur_hand`, `sweep-search-heur_evo_g60`.

Supported: wrapping the same heuristic in depth-4 minimax before labelling raises the teacher `optimal` by +0.1443 (hand) and +0.1414 (generation 60), the 2x32 student by +0.1136 and +0.1206, and the 6x64 student by +0.1306 and +0.1293, all far above the noise level. Gauntlet scores of the students rise from about 0.18-0.22 to about 0.46-0.49.
The students do not fully reach the searching teacher (2x32: -0.0267 and -0.0294; 6x64: -0.0057 and -0.0127), but they recover most of the gain with a single forward pass (no search at play time).
Caveats: one seed per 6x64 student and two for 2x32 (the 2x32 seed-1 runs agree with seed 0 to within 0.0100 on `optimal`, 4.3); the two heuristics are the only searching teachers, so "search helps" is shown for depth 4 with this one family of evaluation functions. The value targets of heuristic/minimax teachers are squashed scores, not calibrated probabilities (section 6), and `value_acc` of these students is 0.66-0.71 against 0.86-0.91 for solver-label students from 2x32 to 10x128 (`runs[].bench.value_acc`).

### 4.8 Distillation: direct (solver labels) vs from 6x64 vs from 8x128

| student size | params | direct optimal (Pons labels, s0) | from sup 6x64 | from sup 8x128 | from 6x64 - direct | from 8x128 - direct | direct gauntlet | from 6x64 gauntlet | from 8x128 gauntlet | compression vs 6x64 (x params) | compression vs 8x128 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1x16 | 16,994 | 0.7870 | 0.7837 | 0.7940 | -0.0033 | +0.0070 | 0.3268 | 0.3143 | 0.3250 | 26.97 | 140.09 |
| 2x32 | 50,146 | 0.8730 | 0.8833 | 0.8693 | +0.0103 | -0.0037 | 0.5366 | 0.5411 | 0.5723 | 9.14 | 47.47 |
| 3x48 | 138,722 | 0.8887 | 0.9003 | 0.8903 | +0.0116 | +0.0016 | 0.6447 | 0.6616 | 0.6848 | 3.30 | 17.16 |
| 4x64 | 310,370 | 0.9037 | 0.9123 | 0.8990 | +0.0086 | -0.0047 | 0.7232 | 0.7393 | 0.7322 | 1.48 | 7.67 |

Source: `analysis.distill_vs_direct.rows` (the two difference columns are recomputed from the optimal columns beside them); teachers: `sup6x64` 458,338 parameters, `optimal` 0.9043; `sup8x128` 2,380,642 parameters, `optimal` 0.913 (`analysis.distill_vs_direct.teachers`). "Direct" = group A seed 0 (Pons labels, hard targets); distilled = group C (soft policy and WDL of the teacher network on the same 300,000 positions, policy head only, no search; section 3.3).

What this shows: distilling from the 6x64 net is ahead of direct training on `optimal` at 2x32, 3x48 and 4x64 by 0.0103, 0.0116 and 0.0086 and behind at 1x16 by 0.0033; distilling from the 8x128 net is ahead at 1x16 (+0.0070) and 3x48 (+0.0016) and behind at 2x32 (-0.0037) and 4x64 (-0.0047).
Differences are at most 0.0116, about the size of the seed-to-seed range in 4.3 (up to 0.01 for solver-label runs), with one seed per distilled run, so **no distillation advantage on `optimal` is established**, and none for the bigger 8x128 teacher. On the gauntlet the distilled 2x32 and 3x48 students from 8x128 score higher than direct (0.5723 vs 0.5366 and 0.6848 vs 0.6447; seed-1 direct values 0.5438 and 0.6696 leave gaps of 0.0285 and 0.0152, around the seed range of 0.025): suggestive at best.
The student that is smallest relative to its teacher (1x16 from 8x128: 140x fewer parameters) keeps `optimal` 0.794 against 0.913, i.e. `retained` 0.8697 of the teacher (`from_8x128_retained`); the 2x32 from 6x64 keeps 0.9768 of its teacher's `optimal` with 9.14x fewer parameters.
Caveats: the teacher was itself trained on the same solver labels, so the comparison is "hard solver labels" vs "soft labels of a net that learned from them", not distillation from an independent source; the direct runs have 40 epochs and the teacher `sup6x64` 30 epochs; the part3-arena distilled nets (`distill_az_2x32`, `distill_ens_2x32`) are a different experiment (teacher = AlphaZero net with MCTS100) and are 175.7 Elo below their teacher's MCTS200 sibling (`size_contrast.distill_vs_teacher`: -175.7, CI [-211.9, -139.9]; the actual teacher, az_6x64 + MCTS100, is not in the arena).

### 4.9 Interpretability: what the layers and channels encode

Models: `sup_6x64` (primary), `az_6x64`, `distill_az_2x32`, `ppo_6x64`. All numbers are on the 10,000 test positions (`interp.json -> notes`). Layers in order: input, stem, block1..block6 (blocks 1-2 for the 2x32 net), pol_conv, val_conv, val_hidden.
**Three caveats apply to the whole subsection.**
(a) Linear probes show that a concept is linearly decodable from a layer, not that the network uses it. The ablation and patching experiments below are the causal checks, and they too are limited (4.9.5).
(b) The positions are a random subset of `nn_data/solver_300k.npz`, the data that `sup_6x64` was trained on (285,000 of the 300,000 positions are its training split; the interp subset was drawn with a different seed and not checked against the split indices). The intact agreement with solver-optimal moves on these positions is 0.9383 for `sup_6x64` (`ablation.intact.sup_6x64.all.agree`) against 0.9043 on the held-out benchmark (`checks.sup_6x64.bench_agreement`), which fits with the net having seen most of these positions. This is an inference from the two numbers; I did not test it. The other nets were not trained on this file in the same way (their intact agreement is 0.7265 az, 0.7481 distill, 0.5976 ppo).
(c) The probe labels come from the solver and from board rules, the same source that trained `sup_6x64`. Concepts such as `outcome` and `score` are therefore partly what the supervised net was taught, and the contrast with the other nets (`interp-diff-*`) mixes "different architecture / training" with "trained on these labels".

#### 4.9.1 Row / column geometry (and phase) is already linear in the input

| concept (sup_6x64) | best probe score | best layer | probe on raw input | first layer with >= 90% of best skill | shuffled-label max | best pooled probe | survives pooling |
|---|---|---|---|---|---|---|---|
| heights | 0.9965 | input | 0.9965 | input | 0.0018 | 0.7389 | no |
| landing_row | 1.0000 | stem | 0.9998 | input | 0.2462 | 0.5273 | no |
| ply | 0.9996 | input | 0.9996 | input | 0.0084 | 1.0000 | yes |
| first_to_move | 1.0000 | input | 1.0000 | input | 0.7694 | 1.0000 | yes |
| phase | 0.9935 | input | 0.9935 | input | 0.5408 | 1.0000 | yes |

Source: `interp.json -> probe_summary.sup_6x64.<concept>` (`best`, `best_layer`, `input_baseline`, `first_layer`, `shuffled_max`, `pooled_best`, `survives_pooling`). Metrics: heights = R^2, landing_row / phase = accuracy, first_to_move = AUC, ply = R^2 (`concepts[]`).
Column heights, landing row, ply, side-to-move parity and phase can be read off the raw input planes by a linear probe with essentially the same score as at any later layer (0.9965 or higher for all but `phase`, 0.994), so their layer profile only shows how long the net keeps them; it does not show where they are computed (finding `interp-trivial`; the notes say the same of phase/ply: "a sanity check, not a discovery").
The position-independent (pooled) versions of heights and landing row are much weaker (0.7389, 0.527 best pooled), i.e. the representation keeps where things are, not only how many. Row/column detectors exist as channels (census below: 5 row/height detectors and 3 column detectors in the stem of `sup_6x64`).

#### 4.9.2 Threats are computed in the early blocks

| concept (sup_6x64) | best probe score | best layer | probe on raw input | first layer with >= 90% of best skill | shuffled-label max | best pooled probe | survives pooling |
|---|---|---|---|---|---|---|---|
| win_now | 1.0000 | block5 | 0.8568 | block1 | 0.4792 | 0.9746 | yes |
| opp_win_now | 0.9991 | block4 | 0.8411 | block1 | 0.4999 | 0.9658 | yes |
| own_threats | 0.8043 | block2 | 0.3447 | block2 | 0.0075 | 0.7096 | yes |
| opp_threats | 0.8252 | block3 | 0.3328 | block1 | 0.0048 | 0.7574 | yes |
| own_double | 0.9909 | block4 | 0.8488 | block1 | 0.5624 | 0.9888 | yes |
| opp_double | 0.9929 | block5 | 0.8131 | block1 | 0.5148 | 0.9950 | yes |
| own_odd_thr | 0.7302 | block2 | 0.2400 | block2 | 0.0101 | 0.4578 | no |
| own_even_thr | 0.7348 | block2 | 0.2506 | block2 | -0.0021 | 0.4605 | no |
| opp_odd_thr | 0.7612 | block3 | 0.2238 | block2 | -0.0038 | 0.4773 | no |
| opp_even_thr | 0.7719 | block3 | 0.2507 | block1 | 0.0067 | 0.4980 | no |

Same concepts across the four nets (best probe score and the layer where it is reached):

| concept | sup_6x64 | az_6x64 | distill_az_2x32 | ppo_6x64 |
|---|---|---|---|---|
| win_now | 1.000 (block5) | 0.982 (block4) | 0.979 (block2) | 0.971 (block3) |
| opp_win_now | 0.999 (block4) | 0.984 (block4) | 0.976 (block2) | 0.975 (block4) |
| own_threats | 0.804 (block2) | 0.649 (block4) | 0.640 (block1) | 0.611 (block1) |
| opp_threats | 0.825 (block3) | 0.696 (block2) | 0.664 (block2) | 0.645 (block2) |
| own_double | 0.991 (block4) | 0.951 (block6) | 0.949 (block2) | 0.936 (block3) |
| opp_double | 0.993 (block5) | 0.948 (block6) | 0.931 (block2) | 0.933 (block3) |

Source: `probe_summary.<model>.<concept>`. Metrics: win_now, opp_win_now, own_double, opp_double = AUC; own_threats, opp_threats and the odd/even rows = R^2.
Reading: in `sup_6x64`, "an opponent threat exists / is playable" (`opp_win_now`, `opp_double`, `opp_threats`) and "I can win now" (`win_now`, `own_double`) reach 90 % of their best skill at block1, and own threat-square counts at block2; best scores are at blocks 2-5 (0.804 R^2 for own_threats at block2, 0.825 for opp_threats at block3, AUC 1.000 / 0.999 for win_now / opp_win_now at blocks 5 / 4).
So threats are decodable after one or two residual blocks, and the best readout comes a few blocks later. Caveat: threats are partly linear in the input already (raw-input AUC 0.857 for win_now and 0.841 for opp_win_now, R^2 0.345 / 0.333 for threat counts), so the gain from layers is large for counts and moderate for the win/no-win concepts.
Odd/even-row threat squares (zugzwang-style) are decodable with R^2 0.73 to 0.77 at blocks 2-3 and the pooled versions do not survive spatial averaging, i.e. the net keeps them as spatial maps.
Other nets: all four nets decode own and opponent immediate wins with AUC 0.971 or higher and doubles with AUC 0.931 or higher somewhere in the stack; the threat counts are clearly better in `sup_6x64` than in the others (own_threats R^2 0.804 vs 0.649 az, 0.640 distill, 0.611 ppo, `interp-diff-*`), which fits, but is not shown to be caused by, supervised training on solver labels (caveat (c)).

#### 4.9.3 Own move vs the opponent's move

| concept | sup_6x64 | az_6x64 | distill_az_2x32 | ppo_6x64 |
|---|---|---|---|---|
| opt_set | 0.974 (pol_conv); raw input 0.719; shuffled max 0.522 | 0.861 (block6); raw input 0.719; shuffled max 0.528 | 0.842 (block2); raw input 0.719; shuffled max 0.523 | 0.825 (block4); raw input 0.719; shuffled max 0.513 |
| net_move | 0.947 (pol_conv); raw input 0.331; shuffled max 0.209 | 0.972 (pol_conv); raw input 0.393; shuffled max 0.211 | 0.890 (pol_conv); raw input 0.376; shuffled max 0.218 | 0.927 (pol_conv); raw input 0.428; shuffled max 0.210 |
| outcome | 0.949 (val_hidden); raw input 0.687; shuffled max 0.617 | 0.828 (block6); raw input 0.687; shuffled max 0.617 | 0.799 (val_hidden); raw input 0.686; shuffled max 0.617 | 0.785 (block6); raw input 0.687; shuffled max 0.617 |
| score | 0.851 (block6); raw input 0.188; shuffled max 0.012 | 0.553 (block6); raw input 0.188; shuffled max 0.026 | 0.491 (block2); raw input 0.186; shuffled max 0.008 | 0.433 (block3); raw input 0.187; shuffled max 0.001 |
| opp_reply | 0.723 (block6); raw input 0.691; shuffled max 0.520 | 0.684 (input); raw input 0.684; shuffled max 0.512 | 0.693 (input); raw input 0.693; shuffled max 0.521 | 0.687 (input); raw input 0.687; shuffled max 0.511 |
| opp_net_move | 0.731 (block6); raw input 0.376; shuffled max 0.185 | 0.653 (block6); raw input 0.360; shuffled max 0.218 | 0.663 (block2); raw input 0.373; shuffled max 0.218 | 0.634 (block4); raw input 0.400; shuffled max 0.216 |

Source: `probe_summary.<model>.<concept>`. Metrics: opt_set and opp_reply = AUC; net_move, outcome, opp_net_move = accuracy; score = R^2. `opp_reply` uses the net's argmax move, then the successor solved with Pons' solver; positions where the net's move ends the game are dropped. Labelled positions for this concept (`checks.<model>.opp_reply`): 4000 requested -> 3061 / 3390 / 3281 / 3562 labelled for sup / az / distill / ppo, of which 614 / 673 / 653 / 716 are in the test split.

Reading:
- The net's **own move** is linearly decodable from the policy conv layer (`net_move` accuracy 0.947 at `pol_conv` for sup, against 0.331 from the raw input; solver-optimal set AUC 0.974).
- The **opponent's best reply** after the net's move is a near-null result. For `sup_6x64`, AUC 0.7225 at block6 against 0.6906 from the raw input and 0.5196 for shuffled labels (and a test split of only 614 positions); for az, distill and ppo no layer beats the raw-input probe (best layer = `input`: 0.684, 0.693, 0.687). The small gain in sup (+0.032 AUC over the raw input) is not strong evidence that the net represents the opponent's reply; the probe is also data-limited.
- The **opponent's must-block threats** (`opp_win_now`, the opponent's immediate wins) are decodable very well (AUC 0.999 at block4 for sup, table in 4.9.2); this is the one opponent-side quantity the nets clearly carry. It is a threat, not a prediction of the opponent's reply.
- `opp_net_move` (what the net would play if the colour-flipped board were presented) reaches accuracy 0.731 at block6 for sup against 0.376 from the raw input (az 0.653 at block6, distill 0.663 at block2, ppo 0.634 at block4). This uses an off-distribution input (planes swapped and the first-player flag toggled), so it measures a computation on unusual boards, not a model of the opponent.
- Net answer to "does it model the opponent's move?": it clearly represents the opponent's immediate threats; it does not clearly represent the opponent's best reply, and the experiment that would show this (a larger labelled set; see section 7) has not been run.

#### 4.9.4 Block knock-outs, by game phase and by position type

Change in agreement with the solver-optimal move (ablated minus intact) when one residual block is skipped (`identity` mode). Phase: early = ply < 12, mid, late = ply >= 24. Position types: own = an own immediate win exists; block = the opponent has an immediate win that must be blocked; quiet = neither. Intact agreement is in the first row of each model.

**sup_6x64**

| block removed | all | early | mid | late | own win | must block | quiet | all, mean-ablation | missed win / missed block (identity) |
|---|---|---|---|---|---|---|---|---|---|
| intact (agreement) | 0.9383 | 0.9044 | 0.9393 | 0.9823 | 1.0000 | 1.0000 | 0.8904 |  |  |
| block1 | -0.456 | -0.445 | -0.484 | -0.393 | -0.644 | -0.563 | -0.338 | -0.502 | +0.644 / +0.661 |
| block2 | -0.254 | -0.246 | -0.272 | -0.216 | -0.444 | -0.255 | -0.174 | -0.293 | +0.444 / +0.296 |
| block3 | -0.152 | -0.143 | -0.155 | -0.156 | -0.130 | -0.155 | -0.160 | -0.187 | +0.130 / +0.167 |
| block4 | -0.070 | -0.066 | -0.074 | -0.064 | -0.060 | -0.048 | -0.082 | -0.094 | +0.060 / +0.037 |
| block5 | -0.046 | -0.045 | -0.047 | -0.045 | -0.011 | -0.034 | -0.065 | -0.057 | +0.011 / +0.030 |
| block6 | -0.059 | -0.072 | -0.061 | -0.036 | -0.008 | -0.059 | -0.081 | -0.062 | +0.008 / +0.072 |

**az_6x64**

| block removed | all | early | mid | late | own win | must block | quiet | all, mean-ablation | missed win / missed block (identity) |
|---|---|---|---|---|---|---|---|---|---|
| intact (agreement) | 0.7265 | 0.7177 | 0.7027 | 0.8057 | 0.6737 | 0.9019 | 0.6869 |  |  |
| block1 | -0.252 | -0.282 | -0.262 | -0.182 | -0.296 | -0.464 | -0.159 | -0.281 | +0.296 / +0.550 |
| block2 | -0.138 | -0.157 | -0.149 | -0.078 | -0.135 | -0.227 | -0.107 | -0.161 | +0.135 / +0.265 |
| block3 | -0.101 | -0.105 | -0.110 | -0.070 | -0.145 | -0.105 | -0.081 | -0.108 | +0.145 / +0.131 |
| block4 | -0.043 | -0.049 | -0.043 | -0.032 | -0.060 | -0.055 | -0.030 | -0.041 | +0.060 / +0.063 |
| block5 | -0.047 | -0.063 | -0.045 | -0.032 | -0.072 | -0.030 | -0.043 | -0.042 | +0.072 / +0.040 |
| block6 | -0.036 | -0.034 | -0.041 | -0.023 | -0.021 | -0.047 | -0.038 | -0.043 | +0.021 / +0.059 |

**distill_az_2x32**

| block removed | all | early | mid | late | own win | must block | quiet | all, mean-ablation | missed win / missed block (identity) |
|---|---|---|---|---|---|---|---|---|---|
| intact (agreement) | 0.7481 | 0.7469 | 0.7243 | 0.8166 | 0.7732 | 0.8768 | 0.6921 |  |  |
| block1 | -0.351 | -0.388 | -0.363 | -0.267 | -0.538 | -0.400 | -0.256 | -0.329 | +0.538 / +0.485 |
| block2 | -0.405 | -0.504 | -0.388 | -0.318 | -0.517 | -0.486 | -0.330 | -0.344 | +0.517 / +0.594 |

**ppo_6x64**

| block removed | all | early | mid | late | own win | must block | quiet | all, mean-ablation | missed win / missed block (identity) |
|---|---|---|---|---|---|---|---|---|---|
| intact (agreement) | 0.5976 | 0.5632 | 0.5588 | 0.7543 | 0.4792 | 0.7445 | 0.5958 |  |  |
| block1 | -0.168 | -0.155 | -0.180 | -0.150 | -0.277 | -0.187 | -0.115 | -0.164 | +0.277 / +0.232 |
| block2 | -0.057 | -0.050 | -0.060 | -0.061 | -0.038 | -0.104 | -0.049 | -0.084 | +0.038 / +0.126 |
| block3 | -0.059 | -0.040 | -0.062 | -0.078 | -0.111 | -0.028 | -0.048 | -0.066 | +0.111 / +0.038 |
| block4 | -0.082 | -0.069 | -0.093 | -0.069 | -0.116 | -0.097 | -0.062 | -0.059 | +0.116 / +0.124 |
| block5 | -0.028 | -0.026 | -0.032 | -0.021 | -0.052 | -0.014 | -0.023 | -0.024 | +0.052 / +0.018 |
| block6 | -0.023 | -0.015 | -0.028 | -0.019 | -0.037 | -0.012 | -0.021 | -0.024 | +0.037 / +0.014 |

Source: `interp.json -> ablation.blocks.<model>[]` (`mode` = identity, `by_phase`, `by_type`, `overall`) and `ablation.intact`. n = 10,000 test positions; early 2,667, mid 5,408, late 1,925; own win 2,381, must block 1,988, quiet 5,631 (`n` fields for `sup_6x64`).

What this shows (all four nets):
- **Block 1 is the most important block in the three 6x64 nets** (identity knock-out: sup -0.456, az -0.252, ppo -0.168 agreement). Block 2 is next for sup (-0.254) and az (-0.138); in the 2x32 distilled net block 2 (-0.405) is slightly worse than block 1 (-0.351). Importance falls with depth for sup (blocks 4-6: -0.070, -0.046, -0.059) and is small in the last blocks for all three 6x64 nets.
  In the PPO net the order is not monotone (block 4, -0.082, is larger than blocks 2 and 3: -0.057, -0.059).
- **By phase.** In `sup_6x64` the early / mid / late deltas at blocks 1-2 are of similar size (block 1: early -0.445, mid -0.484, late -0.393; block 2: -0.246, -0.272, -0.216), with late positions slightly less damaged. `az_6x64` and `distill_az_2x32` show larger early than late deltas (az block 1: -0.282 vs -0.182; distill block 2: -0.504 vs -0.318). In sup and az later blocks matter a little more for early positions (sup block 6: early -0.072, late -0.036; az block 5: -0.063 vs -0.032); not in ppo (block 4: -0.069 early and late).
  The picture does not support "different blocks for different phases": every phase depends mostly on blocks 1-2.
  Intact agreement differs by phase (sup 0.9044 early, 0.9393 mid, 0.9823 late), so deltas in late positions start from a ceiling.
- **By position type** (sup): removing block 1 destroys own-win moves (-0.644 = every missed win in `missed_win`) and must-block moves (-0.563) more than quiet positions (-0.338); block 2 still hurts wins (-0.444) and blocks (-0.255) more than quiet moves (-0.174). From block 3 on, effects are similar across types, and at blocks 5-6 quiet positions are the most affected (block 5: own win -0.011, must block -0.034, quiet -0.065; block 6: -0.008, -0.059, -0.081). Tactical moves (taking or blocking a win) therefore depend on blocks 1-3; the last blocks matter mainly for the quiet positions where the solver-optimal move is a finer choice.
  The same split holds in az (block 1: own win -0.296, must block -0.464, quiet -0.159) but there the must-block moves are hit harder than own wins. Intact `az_6x64` already misses its own immediate win in 32.6 % of own-win positions (`ablation.intact.az_6x64.all.missed_win` = 0.3263), `sup_6x64` in 0.0 %; the delta columns are therefore relative to very different baselines.
- Caveat: `identity` removes the residual branch only; later layers then see a distribution they were not trained on, so large deltas at block 1 partly reflect distribution shift, not a clean "function of block 1". Also the interp positions are mostly `sup_6x64` training positions (caveat (b) above).

#### 4.9.5 Channel-group ablations against the random-channel null

Channels labelled own-threat / opponent-threat by the census rule (|r| >= 0.3 with threat cells, empty cells only, exceeding the other side's; `interp.json -> notes`) are set to their test-set mean in one layer and compared with 30 random equal-size sets of live channels in the same layer. Effects are ablated minus intact. `pct` = share of the 30 random draws that fall below the effect (1.00 = larger than every draw). "d miss win" = change in the share of own-win positions where the net no longer plays the win; "d miss block" = same for must-block positions.

**sup_6x64** (mean ablation)

| layer | group | channels | d miss win | null d miss win (mean +- sd) | pct | d miss block | null d miss block | pct | d agreement |
|---|---|---|---|---|---|---|---|---|---|
| block1 | own-threat | 1 | +0.000 | 0.002 +- 0.005 | 0.50 | -0.003 | -0.000 +- 0.001 | 0.00 | -0.002 |
| block2 | own-threat | 5 | +0.025 | 0.005 +- 0.012 | 0.97 | -0.002 | -0.000 +- 0.004 | 0.27 | -0.019 |
| block3 | opponent-threat | 4 | +0.000 | 0.001 +- 0.001 | 0.00 | -0.015 | -0.000 +- 0.003 | 0.00 | -0.012 |
| block3 | own-threat | 6 | +0.083 | 0.002 +- 0.002 | 1.00 | -0.001 | -0.001 +- 0.005 | 0.43 | -0.031 |
| block4 | opponent-threat | 1 | +0.000 | 0.000 +- 0.000 | 0.00 | -0.004 | -0.001 +- 0.001 | 0.00 | -0.001 |
| block4 | own-threat | 2 | +0.039 | 0.000 +- 0.000 | 1.00 | +0.001 | -0.001 +- 0.002 | 0.77 | -0.009 |
| block5 | opponent-threat | 1 | +0.000 | 0.000 +- 0.000 | 0.00 | +0.001 | -0.001 +- 0.003 | 0.90 | -0.000 |
| block5 | own-threat | 3 | +0.006 | 0.000 +- 0.000 | 1.00 | -0.001 | -0.001 +- 0.003 | 0.17 | -0.003 |
| block6 | opponent-threat | 1 | +0.000 | 0.000 +- 0.000 | 0.00 | +0.002 | -0.000 +- 0.001 | 1.00 | +0.000 |
| block6 | own-threat | 2 | +0.000 | 0.000 +- 0.000 | 0.00 | +0.000 | -0.000 +- 0.001 | 0.60 | -0.001 |

**az_6x64** (mean ablation)

| layer | group | channels | d miss win | null d miss win (mean +- sd) | pct | d miss block | null d miss block | pct | d agreement |
|---|---|---|---|---|---|---|---|---|---|
| block2 | opponent-threat | 1 | +0.047 | 0.002 +- 0.009 | 1.00 | +0.128 | 0.002 +- 0.005 | 1.00 | -0.056 |
| block3 | opponent-threat | 1 | +0.014 | 0.002 +- 0.010 | 0.90 | +0.078 | 0.005 +- 0.006 | 1.00 | -0.022 |
| block4 | opponent-threat | 1 | -0.026 | -0.001 +- 0.004 | 0.00 | +0.031 | 0.001 +- 0.003 | 1.00 | +0.002 |
| block5 | opponent-threat | 1 | -0.009 | 0.001 +- 0.006 | 0.07 | +0.011 | 0.002 +- 0.004 | 1.00 | +0.003 |

Source: `interp.json -> ablation.groups.<model>[]` (`mode` = mean; `effect`, `null.<metric>.mean/std/pct`). The zero-ablation rows have the same pattern with larger effects (e.g. sup block 3 own-threat: d miss win +0.132, pct 1.00) and are in the file. Findings: `interp-groups-sup_6x64`, `interp-groups-az_6x64`, `interp-threatch-*`. For `distill_az_2x32` and `ppo_6x64` no channel met the threat-channel rule, so no threat-group ablation exists.

What this shows:
- **`sup_6x64`: own-threat channels are specific; opponent-threat channels are a null.** Ablating the own-threat channels at blocks 2-5 (5, 6, 2, 3 channels) raises the share of missed own wins (+0.025, +0.083, +0.039, +0.006; pct 0.97-1.00) while must-block behaviour does not change (d miss block between -0.002 and +0.001). Ablating the opponent-threat channels at blocks 3-6 does not raise missed blocks: d miss block is -0.015, -0.004, +0.001, +0.002 (the +0.001 / +0.002 at blocks 5 and 6 sit at pct 0.90 and 1.00 but are one channel and a change of 0.2 percentage points; the block-3 group reduces missed blocks, pct 0.00). The summary in `interp-groups-sup_6x64` ("opponent-threat groups ... 1 of 4 layers; own-threat groups with the reverse: 5 of 6") counts the block-6 channel as a pass; given its size I treat the opponent-threat result as null.
- **`az_6x64` shows the mirror pattern, but with single channels.** The census finds no own-threat channel in az (`channel_counts`), only one opponent-threat channel per layer at blocks 2-5, and ablating it raises missed blocks by +0.128, +0.078, +0.031, +0.011 (pct 1.00, 1.00, 1.00, 1.00) with smaller effects on missed wins (+0.047, +0.014, -0.026, -0.009). That is a specific must-block channel in az at blocks 2-4, the effect shrinking with depth.
- Overall agreement with the solver falls by at most 0.031 in sup (own-threat group, block 3, mean ablation; 0.045 under zero ablation) and by at most 0.056 in az (single channel, block 2; 0.080 under zero ablation) when a threat group is removed, far less than a block knock-out (4.9.4). The nets are redundant: missing channels are largely covered by others.
- Caveats: the groups are defined from correlations with threat cells, not from function; group sizes are small (1 to 6 channels), so the nulls are small random sets with 30 draws; "pct" is an empirical rank, not a p-value; the censuses call 334 of 448 sup channels "mixed" (below), so most of the network is unlabeled and the labelled groups are a minority.

#### 4.9.6 Direction patching

| model | layer | add: d logit (column j relative to others), range over 7 columns | add: mean z vs 20 random directions | add: d probability of column j | subtract: d logit rel. | subtract: mean z | subtract: d probability |
|---|---|---|---|---|---|---|---|
| sup_6x64 | block4 | +0.83 to +3.30 | 11.2 | +0.079 to +0.330 | -4.75 to -1.51 | -8.8 | -0.117 to -0.027 |
| az_6x64 | block4 | +0.10 to +0.29 | 6.7 | +0.012 to +0.028 | -0.41 to -0.05 | -4.6 | -0.070 to -0.012 |
| distill_az_2x32 | block2 | +0.12 to +0.52 | 5.9 | +0.015 to +0.042 | -0.73 to -0.19 | -7.6 | -0.119 to -0.029 |
| ppo_6x64 | block4 | +0.04 to +0.15 | 2.5 | +0.003 to +0.010 | -0.26 to -0.04 | -3.0 | -0.017 to -0.003 |

Source: `interp.json -> patching.<model>.per_column[]` (`add.d_logit_rel`, `add.z`, `add.d_prob`, `subtract.*`; z is the effect in standard deviations of the 20 random equal-norm directions; mean z recomputed over the 7 columns, matching the figures in `interp-patch-*`). Concept: `opp_win_now` ("opponent wins at column j", i.e. column j must be blocked). Adding the direction is done on quiet positions (e.g. 5340 positions for column 0 in sup), subtracting on positions where column j is the unique must-block column (140 for column 0).

What this shows: moving the activation along the probe direction for "the opponent threatens column j" changes the policy logit of column j in the expected direction in all four nets, and by more than a random direction of the same size (mean z 11.2 for sup, 6.7 az, 5.9 distill, 2.5 ppo on adding; -8.8, -4.6, -7.6, -3.0 on subtracting; the ppo values are modest). So the direction is not only decodable but used by the policy head to promote the blocking move (supported for the layer tested, one concept).
Effect sizes differ: in sup the logit of column j rises by +0.83 to +3.30 relative to the others, and the probability of column j by +0.079 to +0.330; in ppo the logit change is +0.04 to +0.15 and the probability change at most +0.010, small despite z = 2.5. Columns at the edge (0 and 6) respond most in every net.
Caveats: the direction is a probe weight vector (correlational) and the step is the gap between class means along it, so "how large a push" is chosen by the data; the null is 20 random directions per column; each net was patched at the layer where its probe was best (block 4 for sup, az and ppo; block 2 for distill), so layers are not matched across models. Only the must-block concept was patched; the opponent's reply and own-threat directions were not.

#### 4.9.7 Cross-model comparison (CKA)

Linear CKA between `sup_6x64` and each other net at the layer of the same name (row = `sup_6x64` layer, column = other net's layer; 5,000 test positions). 1.00 = identical up to rotation.

| layer | sup vs az_6x64 | sup vs ppo_6x64 | sup vs distill_az_2x32 |
|---|---|---|---|
| input | 1.00 | 1.00 | 1.00 |
| stem | 0.94 | 0.95 | 0.94 |
| block1 | 0.92 | 0.90 | 0.89 |
| block2 | 0.84 | 0.79 | 0.72 |
| block3 | 0.77 | 0.65 | - |
| block4 | 0.69 | 0.50 | - |
| block5 | 0.57 | 0.33 | - |
| block6 | 0.51 | 0.33 | - |
| pol_conv | 0.74 | 0.18 | 0.38 |
| val_conv | 0.48 | 0.37 | 0.40 |
| val_hidden | 0.46 | 0.36 | 0.46 |

Source: `interp.json -> cross_model.cka.<model>.matrix` (diagonal of the matrix; `rows` = sup_6x64 layers, `cols` = the other net's layers). Concept differences (`cross_model.concept_diffs`, threshold 0.15 in normalised probe skill): sup is ahead of az on outcome (0.949 vs 0.828), score (0.851 vs 0.553), opt_set (0.974 vs 0.861), own_even_thr (0.735 vs 0.573); the full lists are in findings `interp-diff-*`.

What this shows: the early layers are nearly the same in all nets (stem 0.94-0.95, block1 0.89-0.92) and similarity falls with depth (sup vs az: 0.69 at block 4, 0.51 at block 6; sup vs ppo 0.50 and 0.33). The heads (pol_conv 0.74 for az, 0.18 for ppo) and the value path (val_conv 0.37-0.48) differ more. The supervised net is more similar to the AlphaZero net than to the PPO net at the same depth.
Caveat: there is no seed baseline. Two supervised nets of the same size trained from different seeds (the sweep has `rs_A_pons_6x64` and `rs_A_pons_6x64_s1`) were not compared, so it is not known how much of the late-layer dissimilarity is training objective and how much is just a different initialisation; "CKA falls with depth" might also happen between two seeds. The nets also differ in training data.

#### 4.9.8 Channel census and policy-head locality

| census label | sup_6x64 | az_6x64 | distill_az_2x32 | ppo_6x64 |
|---|---|---|---|---|
| mixed | 334 | 318 | 44 | 330 |
| phase: late | 63 | 101 | 15 | 80 |
| phase: early | 14 | 7 | 6 | 7 |
| own-threat | 19 | 0 | 0 | 0 |
| opponent-threat | 7 | 4 | 0 | 0 |
| row/height detector | 5 | 16 | 25 | 22 |
| column detector | 3 | 0 | 1 | 2 |
| occupancy copy | 3 | 2 | 5 | 7 |
| dead | 0 | 0 | 0 | 0 |
| total channels (stem + blocks) | 448 | 448 | 96 | 448 |

`sup_6x64` by layer:

| layer | own-threat | opponent-threat | row/height | column | occupancy copy | phase: late | phase: early | mixed |
|---|---|---|---|---|---|---|---|---|
| stem | 0 | 0 | 5 | 3 | 3 | 25 | 3 | 25 |
| block1 | 1 | 0 | 0 | 0 | 0 | 22 | 1 | 40 |
| block2 | 5 | 0 | 0 | 0 | 0 | 8 | 2 | 49 |
| block3 | 6 | 4 | 0 | 0 | 0 | 4 | 3 | 47 |
| block4 | 2 | 1 | 0 | 0 | 0 | 2 | 3 | 56 |
| block5 | 3 | 1 | 0 | 0 | 0 | 1 | 1 | 58 |
| block6 | 2 | 1 | 0 | 0 | 0 | 1 | 1 | 59 |

Source: `interp.json -> channel_counts.<model>.<layer>` (summed over stem and blocks). Label rules are in `interp.json -> notes` and `research/interp.py` constants (own/opponent-threat |r| >= 0.3 over empty cells; occupancy copy |r| >= 0.5; column / row detector: mean map holds >= 25 % of the activation variance and the column (row) main effect >= 60 % of the map variance; phase |r| of the pooled activation with ply >= 0.6; priority threat > occupancy > column > row > phase; otherwise mixed).

What this shows: most channels (334 of 448 in `sup_6x64`) get no interpretable label. Where labels exist, `sup_6x64` has more own-threat channels (19, spread over blocks 1-6, most in blocks 2-3) than opponent-threat channels (7, blocks 3-6); the stem holds the row/height and column detectors and occupancy copies (5, 3, 3). Phase labels are the next largest group, but a "phase" channel only means that the pooled activation correlates with the stone count (ply = stone count), which is trivially present; do not count these as findings.
The self-play and PPO nets have more row/height detectors (az 16, ppo 22, distill 25) than the supervised net (5) and no own-threat channels; the 2x32 distilled net has proportionally the most position-specific channels (25 of 96). These are counts of rule-labelled channels in single networks (one seed each), not evidence about training methods in general.
Policy-head locality (`interp-head-*`): the weight mass of column j's policy logit on board column j relative to the per-column average is 2.31-2.95 for `sup_6x64` and 2.31-3.17 for `az_6x64` (1.0 = no locality), 0.63-1.91 for distill and 0.92-1.42 for ppo. Caveat: weights only, the policy conv layer already mixes the whole board.

## 5. What can and cannot be claimed in the paper

Status words: **supported** = the data show it and the main alternative explanations in the data are addressed; **suggestive** = the point estimates point that way but intervals, sample sizes or confounds leave it open; **not supported** = the data do not show it (or show the opposite). Section numbers refer to section 4.

| claim | status | evidence (section 4 reference and numbers) | caveat |
|---|---|---|---|
| Our supervised 6x64 net + MCTS200 has the highest arena Elo among all neural-network bots tested (ours and open source) | supported | 4.2: ahead of bruneton_az by +186.2 [+149.9, +220.7], of ppo_from_sup by +164.1 [+128.2, +199.9], in the clean rerun; same sign and clear intervals in all five analyses; direct 27-6-7 vs bruneton, sign test p = 0.0008 | 20 openings; one weight file per net; nominal equal simulations (200) but different code (c_puct 1.5 vs 1, tie-break) and Bruneton's net is 2.6x larger; no equal-time comparison |
| ... and ahead of our own AlphaZero net | suggestive, not established | 4.2: +34.7 [-3.5, +70.7] in the clean rerun (interval includes 0), +42.7 [+6.4, +81.8] in part3, +33.2 [-9.3, +71.6] top-15 refit; direct 23-2-15, p = 0.2559; only the post-hoc exclusion of 2 openings gives intervals above 0 (+45.7 [+6.0, +89.1] rerun) | one training run each; they differ in data, training length and checkpoint selection; the rerun replays the same games (98.5 % same winner) |
| The supervised net is statistically indistinguishable from perfect play (Pons) in Elo | suggestive (reads as "not distinguished", not "equal") | 4.2: sup - pons = -25.9 [-61.0, +8.7] in the rerun; but sup scored 0.4125 against pons (13-7-20, p = 0.2962) and pons - az excludes 0 | pons' Elo is capped by lost openings (8 of 20 openings are lost for the first player); the interval is wide and includes values up to +8.7 in favour of sup and -61.0 against |
| Supervised training on solver labels beats self-play at this network size and budget (reason for the lead) | suggestive (hypothesis) | 4.2: lower solver-judged error rate (2.03 % vs 2.81 % az), policy-only `optimal` 0.907 vs 0.727; az training took 12.309 h | one run each, no seeds; not matched for data/steps; checkpoint selection differs; search-gain evidence rests on methods-table ladder Elo that extrapolates (2180.2 vs ladder max d8 = 1500) |
| Per-position accuracy predicts game strength | not supported | 4.2, 4.5: DQN has `optimal` 0.838 but the lowest top-15 Elo; students of minimax teachers have higher `optimal` and lower gauntlet score than students of network teachers; R^2 of `optimal` vs arena Elo 0.315 (n = 12) | small n; different measurements; mechanism untested |
| A 17k-parameter net (1x16) already plays the solver-optimal move in about 79 % of benchmark positions; 2x32 (50k) about 87-88 %, at the level of depth-4 minimax | supported | 4.4: 1x16 0.787 / 0.7943 (two seeds), 2x32 0.873 / 0.882; mm_d4_hand teacher 0.873 | policy head only, 3,000 positions of plies 8-30, SE about 0.0065; not a claim about game strength (gauntlet 1x16 0.3268 / 0.3527, 2x32 0.5366 / 0.5438) |
| Accuracy plateaus at about 0.91 from 6x64 upward (6x64, 6x96, 8x128, 10x128 within 0.0033) | supported as a plateau; the cause is untested | 4.4: 0.9117, 0.9150, 0.9130, 0.9147; seed range up to 0.01 | single seed above 6x64; best epochs 17-25 of 40; the plateau may come from 285,000 training positions rather than capacity |
| Network shape on the 1:16 ladder matters up to about 4x64 and little after | suggestive | 4.4: 0.787 -> 0.873 -> 0.8887 -> 0.9037 -> 0.9117 (6x64, off ladder); 6x96 0.915, 8x128 0.913 | 6x96 and 8x128 have one seed each (1x16 to 4x64 and the 6x64 reference have two); 6x64 is not on the 1:16 ladder |
| 36 blocks x 4096 channels is out of reach on this hardware | supported | section 2: about 10.9 billion parameters (sweep notes) vs RTX 3050 8 GB | arithmetic from the notes, not a failed run |
| Teacher quality predicts student quality | supported as a range effect | 4.5: r = 0.966 (2x32, 12 teachers), 0.982 (6x64, 10 teachers), slopes 0.70, 0.76 | driven by the spread 0.60-1.0 and by heuristic teachers copied almost exactly; the three-teacher sizes give r = -0.12, -0.67, -0.25 and are uninformative; teachers differ in kind |
| Students exceed their teachers | not supported for accurate and heuristic teachers; suggestive for the weak PPO teacher | 4.5: heuristic/minimax teachers: at most +0.008 (noise); `ppo_net`: +0.0180, +0.0126, +0.0216 in three runs; strong teachers: students below by 0.004-0.213 in all but one run | no significance test; the weak teacher's number comes from the same 3,000 positions |
| Evolved heuristic weights make a better no-search teacher after the first 10 generations | suggestive for 0 -> 10 (+0.0286); not supported after 10 | 4.6: 0.7287, 0.7573, 0.7457, 0.7350, 0.7363; slope over generations -0.00009 [-0.00074, 0.00085] | weights evolved with depth-2 search; one benchmark measurement per generation |
| A searching teacher (depth 4) gives a much better student than the same heuristic without search | supported | 4.7: teacher +0.1443 / +0.1414, 2x32 student +0.1136 / +0.1206, 6x64 student +0.1306 / +0.1293, gauntlet 0.18-0.22 -> 0.46-0.49 | one search depth, two heuristics; squashed (uncalibrated) value targets |
| Distilling from a bigger net gives a better small net than learning from solver labels directly | not supported on `optimal`; suggestive on the gauntlet | 4.8: difference to direct at most 0.0116 (6x64 teacher) and 0.0070 (8x128 teacher); 8x128 teacher not better than 6x64 teacher; gauntlet gaps 0.0285 / 0.0152 for 2x32 / 3x48 from 8x128 | one seed per distilled run; the teacher was trained on the same labels |
| Row/column geometry, ply, parity and phase are already linearly decodable from the input | supported (trivial) | 4.9.1: probe scores 0.9965-1.0 at the input layer | nothing is "computed" for these; the layer profile only shows retention |
| Threat concepts become linearly decodable in the first residual blocks (block1-2) and are best read out at blocks 2-5 | supported for sup_6x64 (decodability) | 4.9.2: first layer at 90 % of best skill = block1 for 5 of 6 threat concepts, block2 for own_threats | decodable is not used; the raw input already gives AUC 0.84-0.86 for win/must-block concepts; one seed, positions largely seen in training |
| Own-threat channels in sup_6x64 are specifically needed for taking wins; opponent-threat channels are not needed for blocking | supported (own); null result (opponent) | 4.9.5: own-threat groups at blocks 2-5: d miss win +0.025, +0.083, +0.039, +0.006 (pct 0.97-1.00), d miss block about 0; opponent-threat groups: d miss block -0.015 to +0.002 | groups from correlational labels; 1-6 channels; 30-draw null; single net; az shows the opposite pattern (single opponent-threat channels matter at blocks 2-5) |
| The nets represent the opponent's best reply | not supported | 4.9.3: opp_reply AUC 0.7225 vs 0.6906 raw input, 0.5196 shuffled; no layer beats the raw input in az, distill, ppo | 614 test positions for sup; probe is data-limited |
| The nets represent the opponent's immediate threats and use them | supported | 4.9.2-4.9.3, 4.9.6: opp_win_now AUC 0.999 at block4; adding the probe direction raises column j's logit by +0.83 to +3.30 (sup), mean z 11.2 | one concept patched; ppo effect small (z 2.5) |
| Early blocks carry the decision, last blocks refine quiet positions; tactical moves depend on blocks 1-3 | supported for sup_6x64 (knock-out); phase-specific blocks not supported | 4.9.4: block 1 -0.456, block 2 -0.254, block 3 -0.152; own-win -0.644 at block 1 vs quiet -0.338; late blocks hit quiet positions more | identity knock-out causes distribution shift; positions largely in sup's training set |
| Supervised, self-play and PPO nets use different internal representations | suggestive | 4.9.7-4.9.8: CKA falls with depth (sup vs az 0.51 at block6, vs ppo 0.33); different channel label profiles (own-threat 19 / 0 / 0 in sup / az / ppo) | no seed baseline for CKA; one network per method; different data |
| Our MCTS nets are better than Bruneton's at equal wall-clock | not supported | 4.2: not tested; part3 timings invalid (232x slowdown); rerun gives 305.1 vs 357.9 ms/move as an observation only | no equal-time condition was run |
| plkmo_az (AlphaZero, 5.6 M parameters) shows what AlphaZero nets can do | not supported | section 6: lost 727 of 760 games, also to kaggle_negamax; misses about 6 of 13 immediate wins in ARENA.md checks | adapter or weights suspect; excluded from the top 15 |


## 6. Threats to validity

Each item names what it limits.

**Statistical**
1. **Single seeds.** Only 14 sweep configurations (5 solver-label sizes up to 6x64 and 9 teachers at 2x32) have a second seed; everything else, including all 6x96 / 8x128 / 10x128 runs, all group C (distillation) runs, all 6x64 teacher runs, and every net in the arena and interpretability study (one `sup_6x64`, one `az_6x64`, one `ppo_6x64`, one `distill_az_2x32`) is one run. Seed-to-seed range: `optimal` up to 0.0100, gauntlet up to 0.0259 (4.3). Any claim built on a gap smaller than that is not supported.
2. **20 openings, two near-trivial.** The arena uses 20 openings x 2 colours per pairing (40 games). The Elo interval for sup - az is driven by 20 opening clusters. Openings 18 and 19 (`3344`, `4626`) end within 9 plies in 77.9 % and 78.4 % of games (`e_where_decided.openings[].share_games_le_9_plies`) because the side to move has an immediate unstoppable threat; about 10 % of all games carry almost no information about skill (arena_analysis finding `arena-openings`). In part3, 31 of the 96 losses of `nn_sup_6x64_mcts200` are in these two openings (computed from `part3/games.csv`; matches the finding). Excluding them is a post-hoc choice and is reported as a sensitivity analysis only. First-player value of the 20 openings: 9 win, 3 draw, 8 loss, so even the solver cannot score above its opening values and its Elo is capped.
3. **Deterministic players in the gauntlet.** Each gauntlet game is fixed by (student, opponent, opening, colour). The 560 games per student are 40 openings x 2 colours x 7 opponents, all deterministic; intervals are over openings only, and the gauntlet says nothing about play against anything outside those 7 opponents. Two opponents are nets (`distill_az_2x32`, `sup_6x64`), which may favour students trained from similar signals.
4. **Benchmark standard error.** 3,000 positions, plies 8-30, from random and heuristic play (1,388 random, 1,612 heuristic): standard error about 0.0065 near 0.85. The benchmark is not the distribution of positions that strong players reach, and 866 of the 3,000 positions (3,000 minus the 2,134 `n_not_lost`) are already lost for the side to move, where `keeps_nl` is not defined.
5. **Rerun is a replay.** The clean top-15 rerun reuses openings and seeds; 98.5 % of its games have the same winner as in part3 (4.1). It confirms that the ranking is not an artefact of the hang / thread problem but it is not an independent replication.

**Measurement and design**
6. **Heuristic value target is not calibrated.** For heuristic and minimax teachers the value target is `tanh(s / sigma)` with sigma the standard deviation of that teacher's non-mate scores, mapped to (W, D, L); it is a monotone squashing, not a probability (`sweep.json -> notes`). The value head of those students (`value_acc` 0.66-0.71) is therefore not comparable to the solver-label students (0.81-0.91). The policy results are not affected directly, but the shared loss may be.
7. **Teacher and student are not independent in group C and in the arena.** `sup6x64` / `sup8x128` were trained on the same solver labels the direct students learn from (4.8). The distilled arena nets use `az_6x64 + MCTS100` as teacher and that teacher is not in the arena.
8. **Different training budgets and checkpoint rules.** The arena's `sup_6x64`: 30 epochs on 285,000 positions, `best.npz` chosen by validation loss (epoch 18). `az_6x64`: 100 generations x 1,000 games, final `model.npz`. Sweep students: 40 epochs. `arena_analysis.json` says sup was selected on `bench_optimal`; the code and metrics say validation loss (4.2). The sweep's `rs_A_pons_6x64` (0.9117) and the arena's `sup_6x64` (0.9043) are different training runs of the same size.
9. **Probe labels are correlational and the interpretability positions overlap sup's training data.** Probes show decodability, not use. Labels (threat squares, doubles) come from board rules and the solver, the same source that trained `sup_6x64`. The interp positions are a random subset of the 300,000 training positions (overlap inferred from agreement 0.9383 vs 0.9043 held-out; not checked against the split indices). Census labels are rule-based (334 of 448 `sup_6x64` channels are "mixed"; "phase" labels only track stone count). Group ablations use groups of 1-6 channels against a 30-draw null. Knock-outs use `identity`, which creates distribution shift.
10. **Other nets in the interpretability study are single runs from different pipelines.** `az_6x64` and `ppo_6x64` were not trained on the 300,000 positions; `distill_az_2x32` was trained on positions generated by its own pipeline. Differences between nets mix size (2x32 vs 6x64), objective and data.
11. **`plkmo_az` is likely broken.** It lost 727 of 760 games (score 0.0421), including to `kaggle_negamax`; `docs/ARENA.md` found it takes only about 6 of 13 immediate wins even with 777 reads (its MCTS backs up the network value at finished games). Adapter or weights are suspect and this was not investigated; it was dropped from the top 15 and says nothing about AlphaZero nets in general.
12. **Different Elo scales.** (a) The methods-table "ladder Elo" is measured against minimax depths 1-8 with about 100 games each and extrapolates beyond its ladder for strong nets (sup + MCTS200: 2180.2, ladder maximum d8 = 1500); (b) part3 arena Elo is anchored at `kaggle_negamax` = 1000 in a 20-bot pool; (c) the top-15 refit and rerun are anchored at `ours_minimax_d4` = 1409 in a 15-bot pool. Elo is pool-dependent (`nn_ppo_6x64_policy` moves from rank 14 to 11). Do not put numbers from different scales in one table without saying so.
13. **Part3 time-per-move confound and its fix.** In part3 the MCTS200 nets averaged about 67 s per move (vs 0.29 s in the methods table, about 232x) because OpenBLAS started 16 threads in each of 14 workers; the 6x64 policy nets averaged about 0.49 s vs 2.9 ms. The fix (thread limits set before numpy loads, timeouts and kill logic) is in the working tree; the rerun averages 305.1 ms (sup) and 303.7 ms (az) per move. Strength should not depend on speed because the search uses a fixed number of simulations, but no equal-wall-clock experiment has been run, and the explanation was inferred from code and not re-measured inside the arena (`g_time_confound`).
14. **Hardware concurrency.** The sweep, arena analysis, interp and the top-15 rerun ran partly at the same time on one machine (section 1); ms-per-position and ms-per-move numbers carry load noise.
15. **Code state.** The results were produced by a working tree with 86 uncommitted paths, including untracked `research/` (section 1). Commit and tag before freezing numbers.
16. **Equal budgets are only nominal.** Our MCTS nets and Bruneton's use 200 simulations, but c_puct (1.5 vs 1), tie-breaking (centre-first vs random) and network size (0.458 M vs 1.203 M parameters) differ; Bruneton's search re-seeds from the OS (not reproducible); `alfo_mcts` (3,000 random roll-outs) and `plkmo_az` (777 reads) use other budgets.


## 7. Next experiments, in priority order

The first group decides whether the central arena claim can be written without hedging; the later groups strengthen the sweep and interpretability results. Commands are written for `/home/connor/projects/introAI/connect4` with `.venv/bin/python`; I checked the flags against `--help`, but did not run any of them for this document. "Needs a code change" means no flag exists today.

| # | experiment | why (what in sections 4-6 it settles) | command / what to change |
|---|---|---|---|
| 1 | Commit and tag the current tree (research/, results, runs/rs_*, arena fixes) | threat 15: results come from an uncommitted tree | `git add research experiments/results/research experiments/results/arena/top15 ...` then commit; do this before any rerun that would change files |
| 2 | **Seed spread of the supervised net in the arena.** Put the two sweep 6x64 solver-label nets (`rs_A_pons_6x64`, `rs_A_pons_6x64_s1`) next to `sup_6x64`, `az_6x64`, Bruneton and Pons | the sup - az gap is +33 to +55 Elo; if two supervised seeds differ by that much, the "lead" is seed noise. Cheap: 6 bots, 15 pairings | `.venv/bin/python -m arena.tournament --bots pons_perfect,nn_sup_6x64_mcts200,nn_az_6x64_mcts200,bruneton_az --add-nn nn:runs/rs_A_pons_6x64/best.npz:mcts:200 --add-nn nn:runs/rs_A_pons_6x64_s1/best.npz:mcts:200 --openings 20 --workers 6 --elo-anchor pons_perfect:1000 --name seeds_sup` |
| 3 | **More openings for the sup / az / Bruneton / Pons sub-pool** (100-200 instead of 20) | `needed_experiments` #2 of `arena_analysis.json`: the Elo interval is driven by 20 opening clusters | `.venv/bin/python -m arena.tournament --bots pons_perfect,nn_sup_6x64_mcts200,nn_az_6x64_mcts200,bruneton_az --openings 100 --workers 6 --elo-anchor pons_perfect:1000 --name sub100`. No flag filters near-trivial openings; the opening generator would need a rule that drops positions with a forced win in <= 3 moves (needs a code change). Check that the first 20 openings are the same as before (seed 4150) |
| 4 | **Bruneton with more simulations, and equal-time conditions** | `needed_experiments` #1: nominal 200 simulations is not equal compute; Bruneton's net is 2.6x larger | `--set bruneton_az.sims=800` (and `1600`) in the same sub-pool run (the bot's `sims` setting is listed in `top15/summary.json -> bot_info.bruneton_az.params`). A true equal wall-clock condition needs a time-control option for MCTS bots (needs a code change) |
| 5 | **Policy-only sup and az, and the distillation teacher, in the arena** | `size_contrast.missing`: sup/az policy-only and az + MCTS100 were never in the arena, so the search gain and the student-vs-teacher gap cannot be read from Elo | `--add-nn nn:runs/sup_6x64/best.npz:policy --add-nn nn:runs/az_6x64/model.npz:policy --add-nn nn:runs/az_6x64/model.npz:mcts:100` on top of experiment 2's line-up |
| 6 | **Retrain the AlphaZero net with other seeds** (and ideally the same data budget as sup) | threat 1 and 8: one run each, different budgets; the 12.3 h cost per run (`cost.json`) is why this is later | `nohup .venv/bin/python -m nn.train_selfplay --run az_6x64_s1 --preset standard --seed 1 > runs/az_6x64_s1.log 2>&1 &` (the original used the `standard` preset: 100 generations x 1,000 games, `runs/az_6x64/config.json`) |
| 7 | **Second seeds for the sweep above 6x64 and for groups C and the 6x64 teacher runs** | 4.3/4.4/4.8: the plateau, the 6x96 vs 6x64 gauntlet gap, and the distillation differences (all <= 0.012) cannot be judged with one seed | In `research/sweep.py`, `plan()` adds seed 1 only for `b * c <= 6 * 64` and for the 2x32 teachers; add seed 1 for 6x96, 8x128, `TEACHERS_C` and the 6x64 teacher runs (needs a code change), then `.venv/bin/python -m research.sweep --run --only '_s1$'`. Training time of the large ones from this sweep: 6x96 633 s, 8x128 1,331 s (GPU) |
| 8 | **Does the plateau at 0.91 come from the data size?** Label more positions and train 6x64 and 8x128 on them | 4.4: best epochs 17-25 of 40 and a flat curve beyond 6x64 | `.venv/bin/python -m nn.make_dataset --positions 1000000 --workers 14 --seed 2027 --out nn_data/solver_1m.npz` (the 300,000-position file took 1,287 s on 14 workers), then point the sweep at the new file (needs a code change in `research/sweep.py` / `sweep_train.py`) |
| 9 | **Arena games for the sweep students** to test whether the `optimal`-vs-gauntlet disagreement (4.5) is real | the students of minimax teachers have higher `optimal` but lower gauntlet score than students of network teachers | add `--add-nn nn:runs/rs_B_mm_d4_hand_2x32/best.npz:policy --add-nn nn:runs/rs_B_ppo_from_sup_2x32/best.npz:policy --add-nn nn:runs/rs_A_pons_2x32/best.npz:policy --add-nn nn:runs/rs_B_ppo_net_2x32/best.npz:policy` to a small tournament (policy-only, about 1 ms per move) |
| 10 | **Interpretability on positions the nets have not seen**, and a larger `opp_reply` set | threat 9 / 4.9.3: sup's interp positions are mostly training positions; `opp_reply` has 614 test positions | `.venv/bin/python -m research.interp --models sup_6x64 az_6x64 --opp-n 20000 --out experiments/results/research_opp20k` runs the larger solved set into a separate folder (check the `--out` semantics first, it should not overwrite `interp.json`). Switching the position source to `data/accuracy_positions.csv` (3,000 held-out positions) needs a code change in `research/interp.py` (`BENCH` is already defined there) |
| 11 | **CKA seed baseline** | 4.9.7: without it "late layers differ" cannot be attributed to the training objective | add `rs_A_pons_6x64` and `rs_A_pons_6x64_s1` to the `MODELS` dict in `research/interp.py` (needs a code change), then `.venv/bin/python -m research.interp --models sup_6x64 rs_A_pons_6x64 rs_A_pons_6x64_s1 --out experiments/results/research_seedcka` |
| 12 | **Patch more concepts** (own-threat direction, own-win direction) and test the own-threat channel group in az-like and supervised nets of other sizes | 4.9.5-4.9.6: only the must-block direction was patched | extend the concept list used by the patching step in `research/interp.py` (needs a code change) |
| 13 | **Uncalibrated heuristic value targets** | threat 6: value targets of heuristic and minimax teachers are squashed scores | train heuristic-teacher students with the value loss weight set to 0 (policy only) and compare policy accuracy (the config has a `value_weight` argument; wiring it through `research/sweep_train.py` needs a code change) |
| 14 | **Evolve weights for the no-search role** | 4.6: the weights were evolved with depth-2 search and used without search | `rl/evolve.py` has a `depth` argument (`runs/evo_d2/config.json`); run with depth 0 (check `.venv/bin/python -m rl.evolve --help`), then relabel with `research.teachers` |
| 15 | **Randomise the gauntlet** (sampling or epsilon moves, more openings) | threat 3: deterministic players, 40 openings | needs a code change in `research/sweep_gauntlet.py` |
| 16 | **Find out why `plkmo_az` is weak, or drop it from the paper** | threat 11 | reproduce the repo's own demo with the same weights; compare with the adapter in `arena/adapters.py` |


## 8. Reproduction

Everything runs from the repo root; use the virtualenv interpreter.

```bash
cd /home/connor/projects/introAI/connect4
```

**Supervised sweep** (`research/sweep.py`; 48 runs, started 2026-11-16 00:10:24, last update 02:56:53, i.e. about 2 h 46 min, `sweep.json -> status`)

```bash
.venv/bin/python -m research.teachers --list
.venv/bin/python -m research.teachers --label all        # label + cache every teacher in nn_data/rs_labels_<teacher>.npz (<= 8 workers)
.venv/bin/python -m research.sweep --plan                 # show the queue and what is done
nohup setsid .venv/bin/python -m research.sweep --run > runs/rs_sweep.log 2>&1 &    # resumable; finished = runs/<name>/research.json exists
.venv/bin/python -m research.sweep --run --only 'rs_A_.*'  # only runs whose name matches the regex
.venv/bin/python -m research.sweep --analyse              # rebuild experiments/results/research/sweep.json and sweep_runs.csv
```

**Arena, top-15 tournament** (clean rerun; `top15/summary.json -> meta.command`: 20 openings, 6 workers, 1 thread per worker; 2,850.2 s)

```bash
.venv/bin/python -m arena.tournament --list               # every bot must say "ready"
.venv/bin/python -m arena.tournament --roster top15 --name top15 --openings 20 --workers 6
.venv/bin/python -m arena.tournament --roster top15 --name top15 --openings 20 --workers 6 --resume   # continue an interrupted run
```

Output: `experiments/results/arena/top15/games.csv` and `summary.json`. The original 20-bot run (`part3`) is not meant to be repeated (44 h); it was produced with the old, unfixed tournament code (section 6, item 13).

**Arena analysis** (`research/arena_analysis.py`; full run solves every position of every game, 15-25 minutes the first time, cached in `/tmp/c4_arena_solver_cache.json`)

```bash
.venv/bin/python -m research.arena_analysis                  # writes arena_analysis.json and roster.json
.venv/bin/python -m research.arena_analysis --no-solver      # skip the solver-based parts (fast)
```

The analysis reads `part3` only. The sup-vs-rival Elo differences for the clean rerun (columns 4 and 5 of the table in 4.2) are not produced by it; this snippet reproduces them (about 8 s):

```python
# run from the repo root:  .venv/bin/python - <<'EOF' ... EOF
import research.arena_analysis as A
g = A.load_games('experiments/results/arena/top15/games.csv')
for label, gg in [('all', g), ('no_trivial', [x for x in g if x['opening'] not in ('3344', '4626')])]:
    pt, boot = A.elo_bootstrap(gg, A.TOP15, 'ours_minimax_d4', 1409.0, reps=2000, seed=1)
    for y in ['nn_az_6x64_mcts200', 'pons_perfect', 'bruneton_az', 'nn_ppo_from_sup_policy']:
        print(label, A.diff_ci(boot, A.TOP15, 'nn_sup_6x64_mcts200', y, pt))
print(A.head_to_head(g, 'nn_sup_6x64_mcts200', 'nn_az_6x64_mcts200'))
```

**Interpretability** (`research/interp.py`; runtime 4,330 s, `interp.json -> notes`)

```bash
.venv/bin/python -m research.interp --models sup_6x64 az_6x64 distill_az_2x32 ppo_6x64 --n 50000 --opp-n 4000
.venv/bin/python -m research.interp --quick                   # small smoke run
```

Writes `interp.json` and `interp_channels.json`.

**Tests** (not re-run for this document)

```bash
.venv/bin/python -m pytest -q                                  # whole suite (README: ~255 tests, ~10 s; torch tests skipped without torch)
.venv/bin/python -m pytest -q tests/test_research_sweep.py tests/test_research_arena.py tests/test_research_interp.py tests/test_research_api.py tests/test_arena_timeouts.py
```

**The Research page**

```bash
.venv/bin/python server/app.py        # restart it so the new research_api results are served
# then open http://localhost:5000/research
```

Tabs: Overview, Roster, "Why ours is the best net", "Size & teachers", "Inside the net", "Paper builder" (`web/research.html`). The Roster tab's top-15 re-run card reads `experiments/results/arena/top15/summary.json`.

### File map

| path | what |
|---|---|
| `research/arena_analysis.py` | arena re-analysis: Bradley-Terry + cluster bootstrap, head-to-head, per-move solver judging, regressions, roster |
| `research/sweep.py`, `sweep_train.py`, `sweep_gauntlet.py`, `sweep_analysis.py` | sweep queue / training / gauntlet / analysis; `teachers.py` = the 12 teachers and their label cache |
| `research/interp.py`, `interp_core.py`, `interp_solver.py` | probes, census, ablation, patching, CKA; torch re-implementation of the nets; solver calls for `opp_reply` |
| `server/research_api.py`, `web/research.html`, `research.js`, `research.css` | the `/research` page and its API |
| `experiments/results/research/arena_analysis.json` | arena findings, `best_nn`, regressions, health |
| `experiments/results/research/roster.json` | top-15 roster and the five dropped bots |
| `experiments/results/research/sweep.json`, `sweep_runs.csv` | sweep notes, sizes, teachers, runs, analysis, findings (the CSV is one row per run) |
| `experiments/results/research/interp.json`, `interp_channels.json` | probes, census, ablation, patching, cross-model, findings; heavy per-channel arrays |
| `experiments/results/arena/top15/`, `part3/` | clean rerun and original tournament (`games.csv`, `summary.json`) |
| `runs/rs_*/` | one folder per sweep run: `config.json`, `metrics.jsonl`, `train.json`, `research.json`, `best.npz`, `model.npz` |
| `nn_data/solver_300k.npz`, `solver_300k.json`, `rs_labels_<teacher>.npz` | training positions with solver scores; label caches |
| `data/accuracy_positions.csv` | the 3,000-position held-out benchmark |

## 9. Suggested paper outline

**Title options** (pick one after the claims in section 5 are settled)
1. How small can a Connect 4 network be? Supervised learning from a perfect solver, and what the network learns
2. Copying a solver: network size, teacher quality and distillation for Connect 4 move prediction
3. A supervised Connect 4 net against AlphaZero-style self-play: strength, size and internal structure

**Abstract skeleton** (fill the blanks from the tables; do not add claims that section 5 marks as not supported)

> We study how well small residual networks learn Connect 4 from different teachers. Networks of ____ to ____ parameters are trained on ____ positions labelled by a perfect solver, by hand-written and evolved heuristics with and without depth-4 search, and by trained networks.
> A network with ____ parameters chooses a solver-optimal move in ____ % of ____ held-out positions; accuracy saturates at about ____ % from ____ parameters on. Students ____ (exceed / do not exceed) their teachers; a searching teacher improves the student by ____ points; distillation from a larger network ____ (improves / does not improve) over direct training (differences within ____ points, seed range ____).
> In a round-robin of ____ bots, the supervised network with 200-simulation MCTS reaches ____ Elo, ____ Elo (95 % CI ____ to ____) above an open-source AlphaZero network and ____ Elo (CI ____ to ____) relative to our own self-play network; the latter difference ____ (is / is not) statistically established.
> Linear probes, block and channel ablations and direction patching show that ____ is decodable from layer ____, that ____ channels are needed for ____, and that the opponent's best reply ____ (is / is not) represented. Limits: ____ seeds, ____ openings.

**Section list**

| section | content | where in this file | numbers / tables to take |
|---|---|---|---|
| 1 Introduction | question A and B; contribution list limited to supported claims | 2, 5 | none |
| 2 Related work / background | Connect 4 solved (Pons solver), AlphaZero, distillation, probing and CKA; the open-source bots | 3.5, 4.2 | cite the bots in `bot_info` (author, commit, licence) |
| 3 Methods | data, benchmark metrics, student recipe, teachers, gauntlet, arena protocol, interpretability protocol | 3 | table of teachers (3.3), protocol numbers (3.1-3.6) |
| 4 Supervised size ladder | size ladder, plateau, frontier | 4.3, 4.4 | table 4.4, seed-noise statement 4.3 |
| 5 Teachers | teacher comparison, evolved generations, search vs no search, distillation | 4.5-4.8 | tables 4.5, 4.6, 4.7, 4.8 |
| 6 Arena | roster, best net, what is and is not established | 4.1, 4.2 | tables 4.1 and the five-analysis table in 4.2 |
| 7 Inside the nets | geometry, threats, own vs opponent move, knock-outs, group ablation, patching, CKA | 4.9 | tables 4.9.2-4.9.7 |
| 8 Limitations | threats to validity | 6 | items 1-16 |
| 9 Conclusion | only supported claims | 5 | none |

**Figure list** (each figure maps to a card on the `/research` page, http://localhost:5000/research; draw the paper version from the same JSON)

| figure | content | `/research` tab -> card |
|---|---|---|
| 1 | optimal-move rate and gauntlet score vs parameters, with the OLS fit | Size & teachers -> "Benchmark quality vs network size" |
| 2 | quality / cost frontier | Size & teachers -> "Pareto front: quality vs cost" |
| 3 | student vs teacher accuracy, all teachers | Size & teachers -> "Teachers: own quality vs the students trained from them" |
| 4 | evolved generations 0-60 | Size & teachers -> "Evolved heuristic: generation -> teacher and student quality" |
| 5 | search vs no-search teachers | Size & teachers -> "Does a searching teacher make a better student?" |
| 6 | direct vs distilled | Size & teachers -> "Distillation: direct vs distilled from a bigger net" |
| 7 | top-15 Elo, part3 vs refit | Roster -> "Full pool vs top-15 refit"; clean rerun table: "Top-15 re-run (fixed arena): tournament health" |
| 8 | head-to-head heatmap | Roster -> "Head-to-head scores (top 15)" |
| 9 | sup vs each rival, and sup - az across analyses | Why ours is the best net -> "Head-to-head: ... vs each rival", "Elo differences (Bradley-Terry bootstrap)", "sup - az across the three analyses" |
| 10 | policy head alone vs + MCTS; accuracy by phase | Why ours is the best net -> "Policy head alone vs + MCTS", "sup_6x64 by game phase" |
| 11 | blunders and points lost against perfect play | Why ours is the best net -> "Blunders by game phase (solver-judged)", "Points lost against perfect play" |
| 12 | opening values and where games are decided | Why ours is the best net -> "Where games are decided: the 20 openings" |
| 13 | probe matrix: which layer knows what | Inside the net -> "Which layer knows what (linear probes)" |
| 14 | knock-outs by block, phase and position type; group ablations | Inside the net -> "Knock-outs: which blocks and channels play depends on" |
| 15 | patching | Inside the net -> "Probe-direction patching" |
| 16 | CKA and concept differences | Inside the net -> "Cross-model comparison" |
| 17 | channel census / explorer (optional) | Inside the net -> "Channel explorer", "Channel labels per layer" |

Findings text for the captions can be picked on the "Paper builder" tab (it lists the `findings[]` of the three JSON files with their claim, caveat and evidence). Several of those finding sentences are worded more strongly than the data allow or carry stale details (Appendix); use section 5 of this file for the final wording.

## Appendix: inconsistencies and loose ends found while writing this file

1. `arena_analysis.json -> best_nn.answer.suggestive` says the sup net "was selected on bench_optimal (best.npz)". The training code selects `best.npz` by lowest validation loss, and `runs/sup_6x64/metrics.jsonl` shows the best validation loss at epoch 18 (benchmark 0.9043 there). The sentence should say "validation loss".
2. Two sets of "top-15" Elo values exist: `roster.json -> top15[].top15_elo` / `arena_analysis.json -> elo_top15` (refit on the 4,199 part3 games among the 15 bots, 500 resamples) and `top15/summary.json -> elo` (rerun, 4,200 games). They differ by up to a few points (sup 1783.3 vs 1785.6, `nn_ppo_from_sup_policy` 1625.1 vs 1621.6) and have different intervals. Neither is wrong; state which one a number comes from.
3. The sup - az, sup - pons, sup - bruneton, sup - ppo_from_sup differences for the clean rerun (4.2 columns 4 and 5) exist only in this document and the snippet in section 8; `arena_analysis.json` does not read `top15/games.csv`. Consider adding the rerun to `research/arena_analysis.py`.
4. `interp.json -> findings[interp-groups-sup_6x64]` counts an opponent-threat pass in "1 of 4 layers"; that pass is a one-channel group at block 6 with an effect of +0.002, so the opponent-threat result in sup is effectively null (4.9.5).
5. Parameter counts: 458,338 (`sweep.json`, `config.json`) vs 457,500 (`interp.json -> models`, BatchNorm folded into the convolutions). Same network, two counting conventions.
6. Benchmark accuracy of `sup_6x64`: 0.907 (methods table, 1,000-position subsample) vs 0.9043 (3,000 positions). Different samples, not a contradiction, but the two numbers appear in different findings.
7. `sweep.json -> findings[sweep-size-pons]` reads "0.787 at 16994 params to 0.915 at 2971490 params"; the maximum is 0.9150 at 6x96 (1,013,474 parameters) and 10x128 is 0.9147. The curve is flat above 6x64, not rising to 10x128.
8. `sweep.json -> findings[sweep-teacher-quality-1x16 / 3x48 / 4x64]` report correlations from three teachers (r = -0.12, -0.67, -0.25). They are shown on the Paper builder tab and should not be quoted.
9. The notes in `sweep.json` give 40 epochs for every run; the arena's `sup_6x64` (and `sup6x64` teacher) was trained for 30 epochs on CPU, so group C compares 40-epoch students with a 30-epoch teacher.
10. The part3 `games.csv` has 7,599 games for 20 bots (190 pairings x 40 = 7,600); the missing game is `bruneton_az` vs `nn_sup_6x64_mcts200`, opening 0. The part3 head-to-head for that pair therefore has 39 games (27-5-7); the rerun has 40 (27-6-7).
11. The part3 run records `seconds: 0.0` in `meta` because it was summarised with `--summary-only` after a hang; the wall time (43.86 h) is reconstructed from a log line in `arena_analysis.json -> health`.
12. Not sourced: the CPU model and RAM of the training machine are not in any result file; they are taken from `docs/TRAINING_RUNBOOK.md` (Ryzen 7 5800X, 16 GB).
