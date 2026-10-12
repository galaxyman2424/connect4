# Part 3: five more ways to learn Connect 4, and what each one costs

Part 2 trained one ResNet two ways: **supervised** (copy the perfect solver) and **AlphaZero**
(self-play + MCTS). Part 3 adds five more methods in `rl/`, measures what each costs to train,
and combines them to show where they help each other. Every method writes the same
`runs/<run>/` files, so the Lab page shows them next to the Part 2 runs. The new
**Method comparison** tab puts them all on one table.

| method | what it learns | learns from | hardware | output |
|---|---|---|---|---|
| **Evolution (CMA-ES)** `rl/evolve.py` | 10 weights of the minimax heuristic | games between candidate weight vectors (no gradients) | CPU, all cores | `weights.json` |
| **TD(λ) n-tuple** `rl/ntuple.py` | a 9 M-entry lookup table over random 8-cell patterns | its own self-play, bootstrapping from its own predictions | 1 CPU core | `ntuple.npz` |
| **DQN** `rl/dqn.py` | Q(s, a) for every column, same ResNet | replayed self-play moves (off-policy, Bellman targets) | GPU | `model.npz` |
| **PPO** `rl/ppo.py` | a policy (move probabilities) + critic, same ResNet | fresh self-play games vs itself and past snapshots (on-policy) | GPU + CPU | `model.npz` |
| **Distillation** `rl/distill.py` | a small student net (2×32 by default) | soft labels from trained teachers (any mix of the above) | GPU | `model.npz` |

All network outputs are ordinary `model.npz` files. The website's neural-net opponent,
MCTS, `nn.evaluate` (with its hypothesis test) and the arena all work on them unchanged.

---

## The trade-offs

**Evolution (CMA-ES).** Samples a population of weight vectors around a mean, scores each by
playing games, then moves the mean toward the winners and reshapes the sampling distribution.
- **+** No gradients, labels or GPU. The result is 10 numbers you can read and explain
  ("it learned that threats on odd rows matter").
- **+** It improves the bot people already play against, at the same speed per move.
- **−** Sample-hungry and noisy: every fitness value is a batch of full games.
- **−** It can only re-weight features a human designed. It can't discover new ones, so its
  ceiling is the feature set.

**TD(λ) n-tuple network.** 70 random groups of 8 neighbouring cells, each with a lookup table
over its 4⁸ patterns. Value = tanh(sum of table entries). It learns by temporal-difference
updates from its own self-play (Thill, Koch & Konen 2012 reached near-perfect play this way
with about 10 M games).
- **+** CPU only, one core. Thousands of games per second, sub-millisecond moves.
- **+** Learns from its own predictions (bootstrapping), so it learns faster per game than
  waiting for final results.
- **−** The table memorizes patterns rather than generalizing like a convolution, so it is
  big (37 MB) and needs millions of games.
- **−** On its own it does no search. Played greedily, it misses tactics that depth-2 minimax
  sees.
- Implementation note: with the textbook tanh-derivative update, every value saturated at ±1
  in our tests (the first player wins most early self-play games). After that nothing could
  be learned. We use a cross-entropy-style step and a per-ply discount (γ = 0.98) so that a
  slow loss beats a quick one. This is a finding to mention in the presentation.

**DQN (double DQN).** Learns Q(s, a) with the negamax Bellman target
`r + (−max Q_target(s′, ·))`, from a replay buffer of self-play moves, with a slowly updated
target network.
- **+** Off-policy: every move is reused several times from the buffer, so it gets more from
  each game.
- **+** Training is GPU-bound and needs no search.
- **−** Bootstrapping against a moving opponent (itself) is unstable in two-player games.
  It needs many games and careful ε / target-update settings.
- **−** It learns values, not a policy, and plays greedily.

**PPO (policy gradient).** The policy head is the player. Moves that turned out better than
the critic expected (advantage, GAE) get more likely; the update is clipped so one batch
can't move the policy too far. An opponent pool of past snapshots stops it from cycling.
- **+** Simple, stable, and the trained policy moves instantly.
- **+** The exported net has both heads, so you can add MCTS at play time.
- **−** On-policy: each game is used for a few epochs and then discarded, so it is the most
  game-hungry.
- **−** Without search during training it usually ends up weaker than AlphaZero at equal
  compute.

**Distillation.** A small student copies teachers' move distributions (KL divergence) and
value estimates. A teacher with MCTS gives "what search found" as the target.
- **+** Very cheap to train: no self-play, the labels come from the teachers.
- **+** The student is many times smaller and faster, which suits the website.
- **−** It can at best match its teachers.
- **−** If a teacher uses search, labelling is the expensive step.
- **−** Its real cost includes training the teachers. The comparison table shows it both
  ways.

---

## Measuring training time and compute cost

Every `rl/` trainer measures itself with `rl/meter.py` and writes `runs/<run>/cost.json`:

| field | meaning |
|---|---|
| `wall_hours` | elapsed time |
| `cpu_core_hours` | CPU time of the trainer and all its worker processes (1 core busy for 1 h = 1) |
| `gpu_hours` | hours the run held the GPU (how cloud providers bill) |
| `gpu_wh` | GPU energy: `nvidia-smi` power draw, integrated over time |
| `cpu_wh` | CPU energy from RAPL if readable (usually needs root), otherwise **estimated** as core-hours × 6.5 W per busy thread (105 W ÷ 16 threads on the 5800X) |
| `energy_kwh`, `energy_cost_usd` | GPU + CPU energy and its price (default $0.20/kWh; set `C4_PRICE_PER_KWH`, e.g. to your actual utility rate) |

Each row in `metrics.jsonl` also carries the cumulative numbers, so the Lab page can chart
"Compute spent so far".

**Measuring the Part 2 trainers the same way.** Wrap any command:

```bash
python -m rl.meter --run sup_6x64_m -- python -m nn.train_supervised --data nn_data/solver_300k.npz --run sup_6x64_m --epochs 30
python -m rl.meter --run az_6x64 -- python -m nn.train_selfplay --run az_6x64 --resume --generations 100
```

**Runs that are already finished** (your `az_6x64`): write an estimate from their logged time:

```bash
python -m rl.meter --estimate --run az_6x64 --gpu-watts 100 --cpu-watts 90
```

Get the wattages by watching `nvidia-smi` (GPU) and, ideally, a wall power meter for a few
minutes of a `--resume`d run. The file is marked `"estimated": true`.

Pin the GPU first so only the RTX 3050 is measured (the meter reads `CUDA_VISIBLE_DEVICES`):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
```

---

## Commands (Linux PC, from the project folder, venv active)

Times are **rough guesses** for the Ryzen 7 5800X + RTX 3050. Check the per-generation timing
in the first lines of each log and scale. Every trainer can be stopped with Ctrl+C and
continued with `--resume` (except distillation, which is short).

### 0. Check everything runs (about 5 min)

```bash
python -m pytest -q tests/test_rl.py
python -m rl.evolve  --run smoke_evo --generations 2 --popsize 4 --openings 2 --depth 1 --eval-every 1 --eval-depths 1,2 --eval-openings 2 --bench-positions 40
python -m rl.ntuple  --run smoke_nt  --games 20000 --games-per-gen 10000 --eval-every 2 --eval-openings 2 --eval-depths 1
python -m rl.dqn     --run smoke_dqn --iterations 3 --steps-per-iter 30 --warmup 2000 --eval-every 3 --eval-openings 2 --eval-depths 1
python -m rl.ppo     --run smoke_ppo --iterations 3 --games 256 --eval-every 3 --eval-openings 2 --eval-depths 1
python -m rl.distill --run smoke_dist --teacher nn:runs/az_6x64/model.npz --positions 5000 --epochs 2
rm -r runs/smoke_*
```

### 1. Train each method (use `nohup ... &` so they survive closing the terminal)

The CPU-only methods can run at the same time as one GPU method.

```bash
# Evolution: ~20-40 min, all CPU cores
nohup python -m rl.evolve --run evo_d2 --generations 60 --depth 2 > runs/evo_d2.log 2>&1 &

# TD n-tuple: one CPU core, ~1-2 h for 3 M games; run it alongside a GPU job
nohup python -m rl.ntuple --run td_ntuple --games 3000000 > runs/td_ntuple.log 2>&1 &

# DQN: GPU, ~45-90 min for 400 iterations (~400k games)
nohup python -m rl.dqn --run dqn_6x64 --iterations 400 > runs/dqn_6x64.log 2>&1 &

# PPO: GPU, ~45-90 min for 300 iterations (~600k games)
nohup python -m rl.ppo --run ppo_6x64 --iterations 300 > runs/ppo_6x64.log 2>&1 &

tail -f runs/ppo_6x64.log          # watch any of them (Ctrl+C stops watching, not training)
```

Then evaluate the networks the same way as Part 2 (this adds the hypothesis test and tactics
for them):

```bash
python -m nn.evaluate --run dqn_6x64
python -m nn.evaluate --run ppo_6x64
```

### 2. Combinations: where the methods help each other

| # | question | what to run | where to look |
|---|---|---|---|
| C1 | **Learning + search.** Does adding search to a learned evaluator help? | nothing extra: `rl.compare` scores the n-tuple player greedy *and* inside minimax d4, and PPO/AZ policy alone *and* + MCTS 200 | Method comparison: compare each "no search" row with its "+ search" row |
| C2 | **Evolution improves the human heuristic.** | `evo_d2` above | Method comparison: evolved weights vs "hand-crafted weights, minimax d4 (reference)". The evolution run's "Score vs minimax" chart shows the head-to-head at equal depth |
| C3 | **Supervised warm start for RL.** Does knowing perfect-play moves first make RL cheaper? | `python -m rl.ppo --run ppo_from_sup --init runs/sup_6x64/checkpoints/best.pt --iterations 300` and `python -m rl.dqn --run dqn_from_sup --init runs/sup_6x64/checkpoints/best.pt --iterations 400` | Lab: accuracy/Elo curves of `ppo_6x64` vs `ppo_from_sup` (x axis = iterations, same budget) |
| C4 | **Evolution with solver labels vs with games.** Cheap labels vs self-generated signal | `python -m rl.evolve --run evo_acc --fitness accuracy --data nn_data/solver_300k.npz --generations 60` | Method comparison: `evo_d2` vs `evo_acc` (Elo and cost) |
| C5 | **Distil search into a tiny net.** | `python -m rl.distill --run distill_az_2x32 --teacher nn:runs/az_6x64/model.npz:mcts:100` | student vs teacher: Elo, ms/move, params |
| C6 | **Ensemble teacher**: AlphaZero + PPO + n-tuple together → one student | see below | `distill_ens_2x32` vs `distill_az_2x32`: does mixing teachers help, or does the weakest one hurt? |

```bash
# C6: ensemble distillation (weights: AZ+MCTS counts double)
python -m rl.distill --run distill_ens_2x32 \
    --teacher nn:runs/az_6x64/model.npz:mcts:100 \
    --teacher nn:runs/ppo_6x64/model.npz \
    --teacher ntuple:runs/td_ntuple/ntuple.npz \
    --teacher-weights 2,1,1

# same labels, bigger student (no teacher work: reuses labels.npz)
python -m rl.distill --run distill_ens_4x48 --labels runs/distill_ens_2x32/checkpoints/labels.npz --blocks 4 --channels 48
```

### 3. One table for everything

```bash
python -m rl.compare --list                     # what will be evaluated
python -m rl.compare                            # ~30-60 min with all cores (Elo vs minimax d1-d8, 1,000 positions)
python -m rl.compare --name final --openings 20 --positions 3000    # bigger, for the final slides
```

Writes `experiments/results/methods/<name>/summary.json` (Lab page, **Method comparison**
tab), `summary.csv`, and three PNGs: `elo_vs_cost.png`, `elo_vs_time.png`,
`elo_vs_speed.png`. Results are cached per model file, so re-running after training one more
method only evaluates the new one.

### 4. Into the arena with the open-source bots

```bash
python -m arena.tournament --openings 20 \
    --add-nn nn:runs/az_6x64/model.npz:mcts:200 \
    --add-nn nn:runs/ppo_6x64/model.npz:mcts:200 \
    --add-nn nn:runs/distill_az_2x32/best.npz:policy \
    --add-nn evo:runs/evo_d2/weights.json:6 \
    --add-nn ntuple:runs/td_ntuple/ntuple.npz:4
```

### 5. Bring everything to the website PC

```bash
git add runs/ experiments/results/methods/ experiments/results/arena/
git commit -m "Part 3: evolution, n-tuple, DQN, PPO, distillation + comparison"
git push
```

Large files: `ntuple.npz` is about 37 MB (fine for git; GitHub's limit is 100 MB per file).
`checkpoints/` stays git-ignored as before (`runs/*/checkpoints/`), which also covers the
new `labels.npz`, `pool.pt` and `cma.json` files.

---

## What to say in the presentation

Fill these in with your measured numbers; the expectations below come from the literature,
not from your runs.

1. **One chart answers "what does it cost?"**: Elo vs training energy (log x). Expect
   search-based training (AlphaZero) high and to the right, distillation near its teacher for
   a small extra cost, and the n-tuple and evolution at the cheap end.
2. **Learning vs search is a trade** (C1, "Strength vs thinking time"): you pay for search on
   every move at play time, and for learning once, up front.
3. **Combinations**: does a supervised warm start cut RL cost (C3)? Do the evolved weights
   beat the hand-tuned ones at equal depth (C2)? How close does a 2×32 student get to its
   teacher, and does an ensemble teacher help (C5/C6)?
4. **Negative results count too**: DQN's instability, PPO's appetite for games, and the
   n-tuple's saturation problem and the update change that fixed it.
