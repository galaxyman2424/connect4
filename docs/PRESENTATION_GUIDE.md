# Presentation guide: what we did, how to explain it, and what to show

Use this to explain the neural-network and arena work. Each section says what to say,
what to show (usually a Lab page chart), and where the numbers come from. Fill in the
`[ ]` blanks after your GPU runs. Every number on the Lab page is also in a JSON/CSV
file, listed next to it.

---

## The story in one paragraph

> We built a minimax bot with a hand-crafted heuristic and measured it (depth vs strength vs
> cost). Then we asked whether a neural network could *learn* what we hand-coded. First we
> trained one on perfect-play labels from a Connect 4 solver (can it learn evaluation at
> all?). Then we trained one from scratch with AlphaZero-style self-play (can it learn
> without a teacher?). We tested our dossier hypothesis by putting the network's
> evaluation inside our own minimax in place of the heuristic. Finally we ran a tournament
> against real open-source Connect 4 bots, from a YouTube-tutorial minimax to pretrained
> AlphaZero nets and a perfect solver, to see where everything stands.

---

## Steps we took (chronological; this is the "methods" slide)

1. **Baseline** (already done): negamax + alpha-beta + transposition table + move ordering,
   hand-crafted window heuristic, 10 difficulty levels. Measured with a depth tournament
   (Elo ladder d1 = 1000 ... d8 = 1500), accuracy vs perfect play on 3,000 solver-labelled
   positions (d8 = 90.5% optimal), and a tactics suite.
2. **Made the minimax evaluation pluggable** (`engine/minimax.search(..., evaluator=)`), so
   any function, including a neural network, can replace the heuristic. All 227 original
   tests still pass with the default.
3. **Built the network**: a small ResNet with policy and win/draw/loss heads (`nn/model.py`),
   a 4-plane input encoding (`nn/encode.py`), and numpy-only inference (`nn/numpy_net.py`)
   so the website doesn't need PyTorch.
4. **Built batched MCTS** (`nn/mcts.py`), the AlphaZero search, which runs many games at
   once to keep the GPU busy.
5. **Phase A, supervised:** generated [ ] positions, labelled each with the Pons solver,
   held out the benchmark positions, trained `sup_6x64` for [ ] epochs on the RTX 3050
   ([ ] minutes).
6. **Phase B, self-play:** trained `az_6x64` from random weights for [ ] generations / [ ]
   games ([ ] hours).
7. **Evaluated both** with `nn/evaluate.py`: accuracy, tactics, games vs minimax d1-d8, and
   the hypothesis test.
8. **Arena:** fetched 7 open-source bots pinned to exact commits, wrote adapters, checked
   them on win-in-1 positions, and ran a [ ]-bot round robin of [ ] games.
9. **Lab page** (`/lab`): every chart is drawn live from the result files.

Hardware for training: Ryzen 7 5800X, 16 GB RAM, RTX 3050 8 GB, Linux Mint 22.3,
PyTorch [version] / CUDA 13.0. (Recorded automatically in `runs/<run>/config.json` ->
`machine`.)

---

## Suggested slides

### 1. Problem and baseline (30 s)
Website + minimax recap. Show the existing `experiments/figures/elo_vs_depth.png` or the
minimax accuracy figure.

### 2. Why a neural network? (30 s)
The heuristic is our guess at what matters (centre, open threes, twos). Can a network learn
a better one from data? Two routes: learn from a perfect teacher, or learn from nothing.

### 3. The network (1 min), using the "How it works" tab
Input planes, ResNet, two heads. Say why the "am I first player" plane exists (odd/even
threats) and why there's a win/draw/loss head (labels are W/D/L).

### 4. Phase A: learning from perfect play (1 min)
**Show:** Lab, Neural net training, run `sup_6x64`, "Accuracy vs perfect play".
**Say:** "After [ ] epochs the network alone, with no search, picks a perfect move [ ]% of the
time on positions it never saw. Minimax at depth 8 gets 90.5%. With 200 MCTS simulations it
reaches [ ]%." Point at the dashed minimax lines.
**Numbers:** `runs/sup_6x64/eval.json` -> `accuracy`.

### 5. Phase B: learning from nothing (1-2 min). The fun slide.
**Show:** run `az_6x64`: "Where does it play first?" (it discovers the centre), "What it
thinks of the empty board" (it discovers the first player should win), "Strength (Elo)
over training".
**Say:** "Generation 1 plays randomly. By generation [ ] it opens in the centre in [ ]% of
games, and by the end it scores [ ]% against depth-8 minimax, Elo [ ] on our ladder,
after [ ] games of practice and no human knowledge beyond the rules."

### 6. Hypothesis test (1 min)
**Show:** "Hypothesis test" tab.
**Say:** state the hypothesis and the success criterion *from the dossier*, then the result
per depth: score, p-value, tactics, time. Give the verdict honestly. A "not supported"
or "only at depth X" is a fine result if you can explain it. The network is about 100x
slower per evaluation than the bit-parallel heuristic, so even a better evaluation may not
pay off at a fixed time budget.

### 7. Arena (1-2 min)
**Show:** "Bot arena" tab: leaderboard, Elo with CIs, strength vs time, heatmap.
**Say:** who the bots are (cards), the protocol (openings x both colours), where we land.
Mention the two findings from the adapter check (docs/ARENA.md section 2): Galli's minimax
doesn't prefer faster wins, and plkmo's AlphaZero misses immediate wins because its MCTS
trusts the network even at finished games.

### 8. Limitations and next steps (30 s)
* Elo depends on the field and the opening set; CIs show the uncertainty.
* The benchmark covers plies 8-30 only (solver cost).
* One training run per configuration (no seeds repeated); differences of a few % may be noise.
* Next: a bigger network or longer self-play, MCTS + network as the website's level 11,
  distilling the network into a faster evaluator.

### 9. Live demo (optional)
Play page -> Opponent: **Neural net** -> pick the run -> 200 simulations.

---

## Numbers to fill in (and where they are)

| number | file -> field |
|---|---|
| positions in dataset, outcome mix | `nn_data/<name>.json` |
| supervised: policy / MCTS accuracy | `runs/sup_6x64/eval.json` -> `accuracy.nn_policy.optimal`, `accuracy.nn_mcts200.optimal` |
| value accuracy | `eval.json` -> `value.accuracy` |
| self-play games, hours | last line of `runs/az_6x64/metrics.jsonl` -> `total_games`, `elapsed_sec` |
| Elo on the minimax ladder | `eval.json` -> `ladder.elo` |
| score vs minimax d8 | `eval.json` -> `ladder.vs.minimax_d8.score` |
| hypothesis verdict | `eval.json` -> `hypothesis.verdict` + `rows[]` |
| arena ranking | `experiments/results/arena/main/summary.json` -> `elo`, `per_bot` |
| training hardware/software | `runs/<run>/config.json` -> `machine` |

---

## Questions you may get (and answers)

**Isn't the supervised network just memorising the solver?** No. The 3,000 test positions
(and their mirror images) are removed from the training data, so all accuracy numbers are on
positions it never saw. Validation loss is tracked per epoch to catch overfitting.

**Why MCTS instead of minimax for the network?** MCTS uses the policy to decide where to
look, so it needs far fewer evaluations than full-width alpha-beta. A network evaluation
costs about 2 ms vs microseconds for the heuristic, so it has to be choosy. We also test the
network inside minimax; that's the hypothesis test.

**How is Elo computed?** For training progress: max-likelihood Elo from games against our
minimax at fixed depths, whose ratings come from the 2,800-game depth tournament (d1 = 1000).
For the arena: a Bradley-Terry fit of all games with bootstrap confidence intervals, resampling
whole openings.

**Why random openings?** Most bots are deterministic; without openings every pairing would
replay one game. Each opening is played with both colours to cancel first-move advantage.

**Is it fair to the open-source bots?** We use their own code and their own default
settings (listed per bot), and test that our translation layer is correct. Settings are
shown on the Lab page.

**Why does the AlphaZero run's loss not go down smoothly?** The targets (MCTS visit counts
and game results) change as the network improves; it's a moving target, unlike supervised
learning.

**Could you just use the solver?** Yes. It's perfect, and it's in the arena as the ceiling.
The point of the project is *learning* and *search*, and the solver needs an opening book
(MBs of precomputed results) to be fast.

---

## Lab notebook template (copy into `runs/<run>/notes.md`)

```markdown
# <run name>
Date / machine:
Command:
Why this run (what question):
Settings changed vs previous run:
Observations during training (time per gen, anything odd):
Result (accuracy / Elo / hypothesis):
What I'd try next:
```
