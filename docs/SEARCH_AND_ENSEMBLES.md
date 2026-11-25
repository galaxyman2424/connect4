# Search, bigger benchmark, ensembles → one net, C minimax

Added 2026-11-25 (second batch). Commands are in [`COMMANDS.md`](../COMMANDS.md) part I.
Numbers marked *(cloud)* were measured in a 2-core cloud workspace without a GPU.

## 1. A bigger benchmark (lower variance)

- **Generating it.**
  - `make_accuracy_set.py --n 60000 --out ... --exclude <training data>` writes the positions.
  - `filter_bench.py` removes any position (or its mirror) that appears in a training set,
    then trims every ply bucket back to the same size.
- **Using it.**
  - `nn.evaluate --bench data/accuracy_positions_50k.csv --mcts-sample 6000` runs the policy
    and value sections on all 50k positions, and MCTS on a ply-stratified 6k subset (MCTS
    costs about 1,000× more per position).
  - `nn.bench_stats` and `nn.search_sweep` take `--bench` as well.

95% interval half-width (accuracy points):

| benchmark | per 500/8.3k-position bucket at 95% | overall at 97.5% | overall at 99% |
|---|---|---|---|
| 3,000 (now) | ±1.9 | ±0.56 | ±0.36 |
| 50,000 | ±0.47 | ±0.14 | ±0.09 |

**First result** *(cloud: 60,000 positions, 10,000 per bucket, not yet filtered against the
training sets).*
- score_8x128_s0_r1 scores **96.85% (CI 96.70–96.98)**. That's 0.35 points below its
  3,000-position number (97.20), but still inside the old interval.
- The old set flattered the late buckets. Plies 20–23 come out at 97.4% here, vs 98.6% on
  the 3,000.
- Per-bucket results (± is the 95% interval):

| plies | 8–11 | 12–15 | 16–19 | 20–23 | 24–27 | 28–30 |
|---|---|---|---|---|---|---|
| optimal | 94.7% ±0.4 | 95.1% | 96.0% | 97.4% | 98.5% | 99.4% ±0.2 |

- sup_6x64 scores 90.52 ±0.23.
- Paired comparison: 4,448 positions only the score net gets right vs 654 only sup_6x64
  gets right.
- Filtering can only lower these numbers, since overlap with training data inflates a
  net's score.

## 2. What MCTS does here, and how far it looks

**One simulation** (`nn/mcts.py`):
1. **Select.** Start at the current position. At each node, take the move with the highest
   `Q(a) + c·P(a)·√N / (1 + N(a))`:
   - `Q` is the average result of that move so far, also written W/N (total value over visits);
   - `P` is the net's prior for that move;
   - `N` is the visit count.

   The first term exploits moves that have worked. The second explores moves the net likes
   but that haven't been tried much.
2. **Expand + evaluate.** At the first position not yet in the tree, call the net once. That
   gives priors for its moves (P) and a value for the position (v = P(win) − P(loss)).
3. **Back up.** Add v to every move on the path, flipping the sign each ply: good for me is
   bad for you.
4. **Repeat** `sims` times. Then play the most-visited root move.

There are no random rollouts: the net's value replaces playing games out. Won, lost and
drawn positions inside the tree get their exact value.

**How deep does 200 sims look?** Measured by `nn.search_sweep` *(cloud, 120 positions,
score_8x128_s0_r1)*:

| sims | main line (PV) length | 90th percentile | deepest node | root moves tried | optimal |
|---|---|---|---|---|---|
| 0 (policy) | – | – | – | – | 96.7% |
| 50 | 5.0 plies | 10 | 5.1 | 3.1 of ~7 | 98.3% |
| 200 | 6.6 plies | 14 | 7.0 | 3.2 | 98.3% |

So 200 sims follows the main line about 7 plies ahead (occasionally 14), but the tree is
very narrow. It mostly checks the net's top 3 moves and the opponent's best replies.

**More sims only helps if the net's priors or values are wrong somewhere the tree reaches.**
- The score net's priors are already right about 97% of the time.
- The value it backs up is WDL, which can't tell a fast win from a slow one. That's why
  sup_6x64 went 90.4 → 93.2 → 93.8% from 0 → 200 → 800 sims, but flattens out.

`nn.search_sweep` measures this properly on the GPU:
- **Sims budgets:** 0, 50, 200, 800, 1,600 (up to 3,200).
- **`--values wdl,score`:** the score-head value, `C4_VALUE_SOURCE=score`, prefers faster wins.
- **`--exact-from`:** exact values at late leaves.

**The big lever is exact search late in the game** (`nn/exact_tail.py`). On the 3,000
benchmark *(cloud)*:
- The C alpha-beta at depth 24 is **99.8–100% optimal from ply 16**. It averages 46 ms per
  position at plies 16–19 (worst case under 1 s) and is near-instant later.
- The net alone is 95.8% (plies 16–19) and 98.6–99.2% (plies 20–30).

| player | optimal | keeps result |
|---|---|---|
| score_8x128_s0_r1 policy | 97.20% | 98.92% |
| same + MCTS 200 (eval.json) | 97.87% | 99.72% |
| 3-net ensemble + MCTS 200 (eval_ens3.json) | 98.07% | 99.81% |
| **score net policy + exact C search from ply 16** | **98.43%** | 99.48% |
| … exact from ply 20 / 24 | 97.73% / 97.50% | 99.06% |
| *60k benchmark:* score net policy alone | 96.85% ±0.14 | 98.80% |
| *60k benchmark:* **policy + exact from ply 16** | **98.31% ±0.10** | 99.46% |

On the 60k set the exact search matched Pons on all 40,000 positions from ply 16 on
(100.00%), averaging about 10 ms each. That also checks the C minimax against the solver.

After that, every remaining error is in plies 8–15. Those are positions where no search
reaches the end of the game, so the heuristic or the net has to judge them.

**Next run on the GPU:** MCTS (or the ensemble) for plies 8–15 + exact from ply 16. That's
`nn.search_sweep --root-exact-from 16 --exact-from 20`, or the arena player
`nn:<model>:hybrid:16:200`.

**For the paper:** from ply 16 the search reaches the end of the game, so it is
exact like an endgame tablebase in chess. Report "net + exact endgame search" as its own
row, not as the net's accuracy.

## 3. Would training even harder on perfect play help? Rewards instead of move labels?

**Imitation is near its ceiling where it matters.**
- "Keeps the result" is already 98.9% without search and 99.7–99.8% with MCTS.
- Most of the remaining "optimal" misses are *right result, slower win*. Those don't
  change who wins.
- Training to 99% "optimal" mainly teaches the net to count moves to the end, which is a
  hard function of the board. More data, the hard-position rounds, the score head and
  ensembles are the right tools for that, and you already use all of them.

**Rewards (RL) instead of labels.** The reward for a game is win/draw/loss at the end, which
is far less information than the solver's exact score for all 7 moves at every position.
Your own runs show it: AlphaZero/PPO-from-scratch are clearly below supervised-from-Pons.
RL makes sense when there is no solver, and Connect 4 has one. You aren't missing a
better teacher.

**"Some lines are better than others" is real, but it's about the opponent, not the
game.** In a solved game, two moves with the same Pons score lead to the same result
against perfect play. Against a fallible opponent, one can be much better: a line where
only one reply holds is a trap, and a line where every reply holds is not. That is the
idea behind practical chess play (swindles). You can measure it exactly:
- for each move, ask Pons for the opponent's 7 replies and count how many keep their
  result;
- prefer moves that leave the opponent the fewest safe replies, as a tie-break among equal
  Pons scores or as an extra "trap" training target.

It would raise the score against the arena bots without changing benchmark accuracy.
That's a nice paper experiment; I can add it if you want it.

## 4. Three ensemble nets → one net

You can't average the weights. Independently trained nets (seeds 0/1/2) put the same
features in different channels, so averaging scrambles them. Averaging only works for
fine-tunes of the same starting net, like `s0` and `s0_r1`.

Use **knowledge distillation**: `nn.train_score --teacher a.npz+b.npz+c.npz`.
1. It runs the ensemble once over every training position (mirror-averaged): the policy
   probabilities and the 7 predicted scores.
2. It trains one net on the Pons labels and on those soft teacher targets:
   `loss = Pons losses + distill_weight × (CE to the teacher policy + score regression to
   the teacher scores)`.
3. The student keeps the score head and the usual head selection, so
   `nn.evaluate` / arena / website use it unchanged.

Try it at 8x128 (same cost as one ensemble member) and at 10x128 or 12x128 (more capacity
to absorb three nets). With a perfect teacher already in the loss, expect the student to
recover part of the ensemble's +0.3–0.4 points rather than all of it.

**"Classifying some nodes better".** The ensemble tells you exactly which positions are
uncertain: where the three nets disagree or their score predictions spread. Those are the
positions to:
- mine and up-weight (like `nn.mine_hard`, but chosen by disagreement, at no extra solver
  cost);
- give more search at play time (adaptive compute: MCTS only when the nets disagree);
- hand to exact search, which `exact_tail` already does for every position after ply 16.

## 5. C minimax: is it better?

**Speed.** `engine/cminimax/minimax.c` is the same algorithm as `engine/minimax.py`, built
with `python -m engine.cbuild`. It returns the identical column, score and node count
(`tests/test_cminimax.py`). It's used automatically when built; `C4_MINIMAX_C=0` turns it
off.

| depth | Python | C | speed-up |
|---|---|---|---|
| 4 | 0.80 ms | 0.05 ms | 17× |
| 8 | 22 ms | 0.26 ms | 83× |
| 10 | 62 ms/move | 0.32 ms/move | ~190× |

**Strength.** A faster search reaches deeper in the same time, but the heuristic is the
limit early in the game. On the benchmark *(cloud, C)*:

| depth | optimal | keeps | ms/move |
|---|---|---|---|
| 8 | 90.5% | 94.6% | 0.13 |
| 12 | 92.5% | 95.2% | 0.87 |
| 16 | 94.4% | 96.3% | 7.8 |
| 20 | 95.7% | 97.1% | 82 |

Even depth 20 stays below the net's policy alone (97.2% / 98.9%) because at plies 8–11 it
is only about 83%. Where it shines is the endgame (section 2). Use it there, and as the
fast ladder / arena minimax.

**Timing caveat.** Arena and ladder ms/move for `minimax_d*` bots drop about 100× with the
C version. Use `C4_MINIMAX_C=0` to reproduce the old timing charts.

## 6. Should the project switch languages?

No. Keep Python for orchestration and move only the hot loops into C or the GPU, the way
numpy and torch already do:

| part | where it runs now | bottleneck? |
|---|---|---|
| Pons solver, labels | C++ (Pons) | no |
| minimax / ladder / arena minimax bots | **C now** | no |
| net inference | numpy (website) / torch GPU (`--backend torch`) | no on GPU |
| MCTS tree walk | **C now** for fresh roots (`engine/cminimax/mcts.c`, section 7); Python for self-play roots with Dirichlet noise | no |
| benchmark position generation | Python | minutes, acceptable |
| website / Lab / research scripts | Flask + Python | no |

**More C is not needed.** The MCTS tree is now in C too (section 7). The Python tree cost
about 8–14 µs per simulation, not the 0.16 ms I estimated earlier (that figure included the
network). Rewriting everything (in C++ or Rust) would cost weeks and break the website, Lab,
arena adapters and research scripts, for no accuracy gain.

## 7. C MCTS tree

`engine/cminimax/mcts.c` (built into the same library by `python -m engine.cbuild`;
Python side `nn/cmcts.py`).
- **What runs where.** C does select / expand / back up. Python still runs the network:
  each step C walks every tree to its next leaf, Python evaluates that batch in one call,
  and C backs the values up.
- **Same results.** Same arithmetic and tie-breaks as `nn/mcts.py`, so visit counts, W, Q,
  the principal variation and moves are identical (`tests/test_cmcts.py`, including exact
  leaves and full agent games).
- **When it's used.** `nn.mcts.search` uses it automatically for fresh roots.
  `C4_MCTS_TREE_C=0` forces the Python tree. Self-play roots with Dirichlet noise stay in
  Python.

Measured *(cloud, 1 CPU thread)*:

| setup | Python tree | C tree | speed-up |
|---|---|---|---|
| instant "network", 512 positions × 200 sims (tree + glue only) | 0.81 s | 0.10 s | **8.1×** |
| instant "network", 512 positions × 800 sims | 5.55 s | 0.53 s | **10.5×** |
| real sup_6x64 on CPU numpy, 256 positions × 200 sims | 19.9 s (tree 0.67 s) | 19.0 s (tree 0.06 s) | tree 11×, total 1.04× |
| one game, `MCTSAgent` 200 sims (batch of 1) | 2.55 s | 2.49 s | 1.0× |

**What this means.**
- On CPU the network is 97% of the time, so the C tree changes almost nothing there.
- On a single root (the arena and website), the per-step call overhead is about the same as
  the Python tree's.
- It pays off where the network is fast and the batch is big: `nn.evaluate` and
  `nn.search_sweep` on the GPU. There the tree was about 6–7 ms of every step for 512
  positions at 800 sims, and is now under 1 ms.
- Measure it on the 3050 with
  `python -m nn.cmcts --bench --run score_8x128_s0_r1 --backend torch --positions 1024 --sims 800 --batch 1024`.
