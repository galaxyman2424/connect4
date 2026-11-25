# Faster search and evaluation, benchmark precision, PCA and LDA experiments

Added 2026-11-25. Commands to run are in [`COMMANDS.md`](../COMMANDS.md) part H.

## 1. Is the 3,000-position benchmark enough?

It's enough for one net's headline number, but too small to tell apart two nets that are
both at 97% or better.

95% Wilson interval half-width, in accuracy points (`python -m nn.bench_stats --need`):

| accuracy | 3,000 | 10,000 | 20,000 | 50,000 |
|---|---|---|---|---|
| 90% | ±1.07 | ±0.58 | ±0.41 | ±0.26 |
| 95% | ±0.78 | ±0.42 | ±0.30 | ±0.19 |
| 97.5% | ±0.56 | ±0.30 | ±0.22 | ±0.13 |
| 99% | ±0.36 | ±0.20 | ±0.14 | ±0.09 |

- **score_8x128_s0_r1: 97.20%, 95% CI 96.55 to 97.73.** Each 500-position ply bucket is
  about ±1.4 to ±2 points, so per-bucket differences under ~3 points are noise.
- **Comparing two nets.** Use the paired test (`nn.bench_stats --runs a,b`), not two
  overlapping intervals. Both nets see the same positions, so only the positions where they
  disagree count. Example: score_8x128_s0_r1 vs sup_6x64 gives a difference of +6.8 points
  (CI +5.8 to +7.8), with 230 positions only A gets right, 27 only B gets right, and
  p < 1e-30. Two top nets typically disagree on roughly 50-100 of the 3,000 positions, so
  McNemar needs a difference of about 0.6 points to reach p < 0.05. Single-vs-ensemble or
  seed-vs-seed gaps of 0.2-0.4 points need about 20,000 positions (detectable from roughly
  0.25 points).
- **Claiming 99%.** At 99% only about 30 of the 3,000 positions are errors. A 20k benchmark
  (about 200 errors) makes "99.0 ± 0.14" a solid claim.
- **Biases.** These hold at any benchmark size:
  - Positions are plies 8-30 only.
  - They come from random or simple heuristic play, not from strong games.
  - Positions the solver needed more than 5 s for are dropped, so hard early positions are
    slightly under-represented.
  - The errors are concentrated in plies 8-19 (score net: 94.8-95.8% there, 98.6-99.2% after
    ply 20).
- **Making a bigger benchmark.** Run `experiments/data_prep/make_accuracy_set.py --n 20000 --out
  data/accuracy_positions_20k.csv --exclude <every training .npz>`. It takes minutes on 14
  workers. `--exclude` is essential: positions the nets were trained on must not be in
  the benchmark. From then on, `nn.make_dataset` and `nn.mine_hard` automatically leave out every
  `data/accuracy_positions_*.csv`.
- **Using it.** Score a net on it with `nn.evaluate --bench data/accuracy_positions_20k.csv
  --out eval_bench20k.json` or `nn.bench_stats --bench ...`. `nn.evaluate` now also logs and
  stores `optimal_ci95` / `keeps_nl_ci95`.

## 2. Faster MCTS / minimax / evaluation (moves unchanged)

Everything below gives the same moves and the same numbers. Each was checked on the full
benchmark and in `tests/test_speedups.py`.

| what | where | measured |
|---|---|---|
| Network cache in `MCTSAgent`. Trees of consecutive moves overlap, and Connect 4 has many transpositions. | `nn/agents.py` `CachedEvaluator`, on by default | one 200-sim game: 5,763 evaluations asked, 1,178 computed. **11.1 s → 2.3 s** with the same moves |
| torch backend for `.npz` nets. It reproduces score head + combo + mirror exactly (the older `FoldedNet` torch paths only ran the policy head, so score nets played a different, weaker move there). | `nn/torch_net.py`; `nn.evaluate --backend torch`; `C4_NN_BACKEND=torch` for anything else | 8x128 policy accuracy on 3,000 positions: **22.2 s → 7.3 s** on 2 CPU cores, same 0.9720. A game with cache + torch: **11.1 s → 1.2 s**. The GPU is much faster again for the MCTS accuracy section. |
| Batched depth-1 leaves in NN-minimax | `engine/minimax.py` prefetch hook + `NNValueEvaluator.prefetch` | network calls ÷3.7. With torch: d2 **2.43 → 0.78 s**, d3 **6.5 → 2.25 s** per 50 positions. With numpy: only ~5% faster, because numpy's batched conv costs grow almost linearly. |
| Bigger MCTS batch in the accuracy section | `nn.evaluate --mcts-batch 1024` | fuller GPU batches |
| Hand-crafted minimax | none | already bitboard + TT + killers/history. A static-eval cache gave 0% (TT already covers it). The remaining big win would be PyPy or a C port, not a Python change. |

Notes:

- **ms/move goes down.** The cache makes MCTS nets' ms/move in the arena lower than in arena
  runs made before it. To reproduce the old timing (for example, for the cost/time charts),
  set `C4_MCTS_CACHE=0`.
- **Elo is not affected.** The nets play the same moves.

## 3. Training speed (and BatchNorm)

- **BatchNorm is already there.** Every conv in `C4Net` already has BatchNorm (stem, both
  convs of each block, all heads), and `nn/export.py` folds it into the conv weights, so
  inference pays nothing for it.
- **What BatchNorm does and doesn't buy.** BN helps training stability and allows a higher
  learning rate. It does not make a training step faster.
- **What `--speed` adds.** `nn.train_score` and `nn.train_supervised` now take `--speed`
  (`nn/fasttrain.py`):
  - `off`: the old behaviour.
  - `safe` (default): cudnn.benchmark + TF32.
  - `max`: `safe` + channels_last + torch.compile.
- **No per-step GPU sync.** All levels now keep the loss sums on the GPU. The old loops
  called `.item()` 3-5 times per batch, which forces a sync every step.
- **Measure on the 3050.** `python -m nn.train_speedtest` measures every level × batch size
  on your GPU (it can't be measured in the cloud workspace, which has no GPU). With a
  bigger `--batch`, raise `--lr` roughly in proportion and check that BENCH doesn't drop.

## 4. PCA input reduction

**What it tests.** Does the net need the whole board, or is a rank-K view of the 84 stone
cells enough?

**How it works.**
- `nn.train_score --pca K` fits PCA on the training positions (mirrored copies included).
- It then feeds the conv stem the rank-K reconstruction
  `x' = (x - mean) U Uᵀ + mean` in place of the raw cells.
- The projection is part of the model (`inproj.mean` / `inproj.U` buffers, exported into
  `model.npz`), so `nn.evaluate`, `nn.bench_stats`, the arena and the website run these nets
  like any other.
- `research/pca_lda.py pca-sweep` trains K = 0 (no projection, control), 8, 16, 24, 32, 48, 64
  and 84 (full rank = identity) at 6x64 on `solver_300k`, then runs `nn.evaluate --backend torch`
  on each. `pca-summary` makes `experiments/results/research/pca_lda/pca_summary.{csv,json}`
  and `pca_accuracy.png` (accuracy vs K with 95% CIs).
- `pca-variance` (CPU, seconds) shows how much variance each K keeps and draws the first
  8 components as boards.

**Caveat.** The projection is only implemented in `nn/model.py`, `nn/numpy_net.py` and
`nn/torch_net.py`. The interp scripts (`research/interp_core.py`) and the old
`FoldedNet` GPU paths don't know about it, so don't run PCA nets through those.

## 5. LDA: separating and focusing the training data

`research/pca_lda.py` has three LDA steps: `lda-analyze` (where the data separates),
`lda-focus` (pick a focus subset) and `lda-sweep` (train on it and compare).

**`lda-analyze`** (CPU, about a minute).
- Classes: game result (win/draw/loss) or best column.
- Features:
  - the 84 cells + colour;
  - the 10 hand-crafted features of `rl/features.py` + ply;
  - with `--run`, a trained net's trunk activations.
- It reports 5-fold CV accuracy against the majority-class baseline and the discriminant
  variance. It also draws LD1/LD2 scatter plots, LD1 as a board heat map, and LD1 weights
  on the hand features.
- The trunk features show how much more linearly separable the net has made the problem
  than the raw board. That ties into research direction 2 (what each part of the net
  encodes).

**`lda-focus`.**
- Fits LDA on the training positions.
- `--select uncertain` keeps the 20% of positions where LDA is least sure (smallest gap
  between its top two class probabilities). `--select misclassified` keeps those it gets
  wrong; `--select class:draw` keeps one class.
- It writes them in `nn.mine_hard` format, plus a random control subset of the same size.

**`lda-sweep --go`.**
- Trains three nets: baseline, baseline + focus (`--hard ... --hard-weight 2`) and
  baseline + random control (same weight).
- It runs `nn.evaluate` on each.
- `lda-summary` reports the paired focus-vs-control test. Focus beating control is the
  result that would show the LDA selection matters, rather than just up-weighting some
  positions.
