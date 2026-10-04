# Neural network: design, method, and how to read the results

This is the technical write-up for part 2 of the project: training our own neural network
to play Connect 4. Step-by-step commands are in [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md).
How to present it is in [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md).

## 1. Questions

1. **Can a small neural network learn to evaluate Connect 4 positions?** (Phase A, supervised)
2. **Can it learn to play well with no teacher, only the rules and self-play?** (Phase B, AlphaZero)
3. **Dossier hypothesis:** *"A small neural network can learn a board evaluation that plays
   as well as our hand-crafted minimax heuristic."*
   * Success criterion (dossier): at the same search depth, the network-evaluated minimax wins
     at least as often as the hand-crafted one, and it still blocks and takes obvious wins.
   * Failure criterion (dossier): a lower win rate, missed one-move wins/blocks, or moves that
     take too long.
   * Our earlier "expected decision" was *Defer* (minimax already meets the core goal). These
     experiments are what lets us revisit that decision with data.

## 2. Representation

**Position** = two 64-bit integers, `cur` (stones of the player to move) and `mask` (all stones),
in the same bit layout as `engine/board.py`. Stored that way in datasets and the replay buffer
(16 bytes per position).

**Network input** (`nn/encode.py`): 4 planes of 6x7
| plane | content | why |
|---|---|---|
| 0 | my stones | side-to-move view, so one network plays both colours |
| 1 | opponent's stones | |
| 2 | all ones | lets 3x3 convolutions tell "edge of board" from "empty" |
| 3 | all ones if I am the first player | odd/even-row threats decide many Connect 4 endgames, and the network can't count stones reliably |

**Data augmentation:** Connect 4 is left-right symmetric, so every training batch mirrors a
random half of the positions (and their policy targets).

## 3. Network (`nn/model.py`)

A small AlphaZero-style residual CNN, default 6 blocks x 64 channels = **458k parameters**:

```
input 4x6x7
 -> 3x3 conv (64) + BatchNorm + ReLU
 -> 6 x residual block [3x3 conv, BN, ReLU, 3x3 conv, BN, + skip, ReLU]
 -> policy head: 1x1 conv (2) + BN + ReLU -> linear -> 7 logits     ("which column?")
 -> value head:  1x1 conv (4) + BN + ReLU -> linear 64 -> ReLU -> 3 logits (win / draw / loss)
```

* **WDL value head** instead of AlphaZero's single tanh. Solver labels and game results are
  naturally win/draw/loss, cross-entropy trains more smoothly, and we still get a scalar
  `v = P(win) - P(loss)` in [-1, 1] for search.
* **Deployment without PyTorch:** after every epoch/generation, `nn/export.py` folds BatchNorm
  into the convolutions and writes `runs/<run>/model.npz`. `nn/numpy_net.py` runs it with numpy
  only (tests check it matches PyTorch to 1e-4). That's what the website and arena use, so the
  Windows laptop never needs PyTorch. One position takes about 2 ms on a CPU, so a 200-simulation
  search takes about 0.4 s.

## 4. Search with the network (`nn/mcts.py`, `nn/agents.py`)

Three ways to play with a trained network:

| agent | what it does | used for |
|---|---|---|
| `PolicyAgent` | plays the policy head's top move, no search | "intuition only" accuracy |
| `MCTSAgent` | PUCT Monte Carlo Tree Search, AlphaZero style | the real player; website opponent |
| `NNMinimaxAgent` | **our own minimax**, with the network's value replacing the hand-crafted heuristic at the leaves | the hypothesis test |

**PUCT:** each simulation walks down the tree picking the move maximising
`Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))`, evaluates the new leaf once with the
network, and backs the value up the path (negated every ply). Wins and draws inside the tree
are scored exactly (+1/0), so forced wins are found without hand-written rules. The move
played is the most visited one. `c_puct = 1.5`; unvisited moves count as Q = 0.

**Batching:** `search()` runs many games' trees in lock-step and evaluates all their leaves in
one network call. This is what makes self-play fast on one GPU.

**Hook into minimax:** `engine/minimax.search()` gained an optional `evaluator=` argument
(default: the hand-crafted heuristic, so all old behaviour and the existing tests are unchanged).
`NNValueEvaluator` multiplies the network value by 100, which puts it on the heuristic's
scale and far below the search's win scores (about 1e6). It also caches positions, because
transpositions are common.

## 5. Phase A: supervised learning (`nn/make_dataset.py`, `nn/train_supervised.py`)

**Data.** Positions come from games played by a random mix of move pickers (uniformly random,
and our minimax at depths 1-4, with some random moves mixed in), so both sloppy and sensible
positions at every ply 0-41 appear. Duplicates and mirror images are removed. **Every
position of the held-out benchmark (and its mirror) is removed**, so the evaluation stays
honest. Each position is labelled by Pascal Pons' solver (`c4solver -a` + opening book) with the
exact score of every column.

**Targets.** Policy = uniform over the solver's optimal columns. Value = win/draw/loss for the
side to move under perfect play.

**Training.** AdamW (lr 2e-3, one-cycle schedule, weight decay 1e-4), batch 1024, bf16
autocast on the GPU, 5% validation split, loss = policy CE + value CE. Each epoch exports
`model.npz`, and `best.npz` is the epoch with the lowest validation loss. (We select on
validation, never on the benchmark.)

## 6. Phase B: AlphaZero self-play (`nn/train_selfplay.py`)

Each **generation**:
1. **Self-play.** `--games` games of the current network against itself (8 worker processes
   x 128 simultaneous games, all sharing the GPU). Every move: MCTS with `--sims` simulations,
   Dirichlet noise on the root (alpha 1.0, 25%), moves sampled in proportion to visits for the
   first 12 plies, then greedy. The search tree is reused between moves.
2. **Store** every position with `pi` (root visit distribution) and `z` (final result for the
   player to move). Replay buffer = last `--buffer` positions (600k in `standard`).
3. **Train** on random mirrored mini-batches. Steps = new positions x 4 / batch, so each
   position is seen about 4 times on average. Loss = CE(policy, pi) + CE(WDL, z). AdamW,
   lr decays from 1e-3 to 1e-4 over the planned generations.
4. **Export** `model.npz`. Save `checkpoints/latest.pt` + `buffer.npz` (for `--resume`) and
   a `gen_XXXX.pt` every 10 generations.
5. **Measure** (see section 7) and append one line to `metrics.jsonl`.

There's no gating ("is the new net better than the old one?"); AlphaZero dropped it too.
Strength is tracked against fixed minimax opponents instead.

| preset | games/gen x gens | sims | buffer | intended use |
|---|---|---|---|---|
| quick | 200 x 15 | 64 | 150k | smoke test |
| standard | 1000 x 100 | 128 | 600k | main run (~100k games) |
| long | 2000 x 200 | 200 | 1.2M | overnight-plus, strongest |

## 7. Measurements (identical definitions to the minimax experiments)

| metric | definition | where |
|---|---|---|
| `bench_optimal` | share of the 3,000 held-out solver-labelled positions (`data/accuracy_positions.csv`, plies 8-30) where the policy's top move is optimal | every epoch/gen |
| `bench_keeps_nl` | among positions not already lost, share where the move keeps the result (win stays win, draw stays draw) = 1 - blunder rate | every epoch/gen |
| `bench_value_acc` | value head's W/D/L call matches the solver (majority baseline = 63.3%) | every epoch/gen |
| `bench_mcts_optimal` | same as `bench_optimal`, but with MCTS (100 sims) | every `--eval-every` |
| `vs_minimax` | score vs minimax d2/d4/d6, 10 openings x 2 colours | every `--eval-every` |
| `elo` | max-likelihood Elo from those games against the **fixed ratings of the existing minimax ladder** (`experiments/results/elo.json`, d1 = 1000, d8 = 1500) | every `--eval-every` |
| `empty_board_wdl`, `first_move_hist` | what the net thinks of the empty board; where self-play games start | every gen |

`nn/evaluate.py` then does the full evaluation of a finished model and writes `eval.json`:
accuracy for policy and MCTS 50/200/800, value confusion matrix, the 52-position tactics
suite, 20 openings x 2 colours against each minimax depth 1-8 (with Elo), and the
**hypothesis test**: for depths 1-4,
* head to head: NN-evaluated minimax vs hand-crafted minimax, same depth, 20 openings x 2
  colours, with an exact binomial test on decisive games;
* accuracy of both on the same 300 benchmark positions (paired; McNemar test);
* tactics (win in 1, forced block) of both;
* time per move.

**Supported at depth d** if the NN version scores at least 50%, takes wins and makes blocks
at least as often as the hand-crafted version at that depth, and stays under 2 s per move.
(Note: hand-crafted depth-1 minimax only makes 69% of forced blocks, because it can't see
the opponent's reply. Demanding 100% at depth 1 would test the depth, not the evaluation.)

## 8. Reading the charts (Lab page, Neural net training tab)

* **Accuracy vs perfect play.** The headline chart. Dashed lines = minimax d1/d4/d8 on the same
  positions (d8 = 90.5% optimal). The question is where the blue (policy only) and orange (with
  MCTS) lines end up relative to those.
* **Training loss.** Phase A: watch for validation (dashed) rising while train falls, which is
  overfitting. Phase B: the targets move as the net improves, so a plateau or even a rise isn't
  a bug.
* **Value accuracy.** Must clearly beat 63% (always guessing "win") to mean anything.
* **Accuracy by game phase.** Minimax is weakest in the opening (the win is 20+ moves away). A
  network can be strong there, because it has seen thousands of games.
* **Elo over training / score vs minimax.** Strength in games, on the same Elo scale as the
  minimax depth tournament.
* **Self-play games.** Connect 4 is a first-player win under perfect play. As the network gets
  stronger, the first player should win its self-play games more and more often.
* **First-move strip / empty board.** Perfect play opens in the centre (column 4) and the empty
  board is a first-player win. Watching the network discover both is a nice slide.

## 9. Pipeline check (not results)

To verify the code end to end, tiny versions ran on a 2-core cloud VM without a GPU:

* supervised, 20k positions, 4x32 net, 6 epochs (30 s): policy 63.5% optimal, MCTS-50 78.8%,
  value 74%, Elo ~1310 on the minimax ladder;
* self-play, 4x32 net, 40 generations x 200 games x 48 sims (27 min): policy 44%, MCTS 74%,
  Elo ~1200, and it already learned to open in the centre (column 4 in ~90% of games).

These only show that the pipeline learns. Real numbers come from the GPU runs, which use
about 100x more games and a 5x bigger network.

## 10. Files

```
nn/encode.py nn/model.py nn/numpy_net.py nn/export.py nn/common.py
nn/mcts.py nn/agents.py nn/bench.py
nn/make_dataset.py nn/train_supervised.py nn/train_selfplay.py nn/evaluate.py
runs/<run>/...            see runs/README.md
tests/test_nn.py          encoding, numpy-vs-torch export, MCTS finds wins/blocks, agents, Elo
```
