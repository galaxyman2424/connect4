# Getting the policy net above 95% optimal

*Written 2026-11-20 on branch `eval-bar-and-accuracy`. Metric: share of the 3,000 held-out
benchmark positions (plies 8-30) where the chosen column is one of the solver's best columns.*

## Where we are (measured today, `python -m experiments.ensemble_probe`, about 8 min on CPU)

| player | optimal | keeps result |
|---|---|---|
| rs_A_pons_10x128, policy | 0.916 | |
| rs_A_pons_10x128, policy + mirror | 0.932 | 0.970 |
| rs_A_pons_8x128, policy + mirror | 0.930 | 0.969 |
| **ensemble of 6 nets + mirror** (8x128, 10x128, 6x96, sup_6x64, 6x64, 6x64_s1) | **0.942** | 0.975 |
| ensemble value head, 1-ply lookahead | 0.935 | **0.980** |

Earlier (connect4-accuracy-ceiling): MCTS 800 on sup_6x64 0.938; mirror + 1-ply 0.936.

Two readings:
- Ensembling is the cheapest real gain we have: +1.0 point over the best single net, no training.
- The value head makes **fewer result-changing mistakes** (98.0% keeps the result) but
  **lower "optimal"**, because a win/draw/loss head cannot tell a fast win from a slow one.
  The missing ~5% is mostly "right result, wrong speed" (4.8% of moves at 93%), not blunders.

So the gap to 95% is a *label* problem plus a *data* problem, not a capacity problem
(sizes flatten at 6x64 and above; 8x128 overfits: train policy loss 0.41 vs val 0.78).

## Plan, in order of expected value per hour

1. **Score head (7 outputs = the solver's score for every column).** We already have these
   scores for every training position (Pons `-a`). Train with a cross-entropy over the 7 columns
   with targets `softmax(score / T)` (T around 1-2) plus a regression/ordinal loss on the scores
   themselves. Pick `argmax(score)`. This is the only change that targets the "wrong speed"
   errors, which are most of what is left. Expected: biggest single jump.
2. **10x more labelled positions** (285k to 3M; 300k took 21 min, so about 3.5 h of solver time
   on the 5800X). Sample from the same ply range as the benchmark (8-30) and from net self-play
   games, not only random play. Fixes the overfitting.
3. **Hard-example mining.** After a first model, run it on 1M fresh positions, keep the ones it
   gets wrong (solver tells us instantly), add them with weight 2-3, retrain. Repeat 2-3 rounds.
4. **Training recipe:** random left-right mirroring during training (not only at inference),
   cosine LR, weight decay 1e-4, EMA / SWA of weights over the last epochs, 3 seeds.
5. **Ensemble + mirror at inference** (already measured: +1 point). With 3 seeds of the
   score-head model this should hold.
6. **Shallow search on the score head** (2-ply alpha-beta or MCTS with the score head as the
   leaf value). Because the head knows win speed, search should raise "optimal", unlike today's
   WDL head.

Report for the paper: a table with each step added on top of the previous one, 3 seeds each.

## Should an LLM (e.g. Fable) edit the weights?

The paper you saw is MIT CSAIL's "weight-space programming" (Takehi, Teoh et al., Nov 2026):
an LLM read raw weights and patched them, e.g. removing a repetition direction in
Qwen3.5-0.8B and fixing a driving controller that got stuck on a ramp.

Why it is unlikely to help here:
- Their wins were **repairs of one localized bug**; the authors say it does poorly when the
  error is spread across the network, and on a model without that bug (Gemma 3 270M) it gave no
  established gain. Our nets have no single bug: the errors are distributed (data limit + a
  head that cannot express win speed).
- We have a **perfect, free teacher** (the solver). Gradient descent on solver labels already
  points every weight in the exact direction that reduces error; an LLM guessing edits to
  ~400k-2M floats cannot beat that signal.
- It was not compared against normal fine-tuning, and did not save compute.
- Earlier in this project an LLM playing moves hallucinated board states, and the plan was to
  keep LLMs out of the bots.

A version that *would* be a fair, small paper experiment: use the interpretability results
(block 3 own-threat channels, the "opponent wins at column j" direction that already moves the
policy when patched) and try **hand-designed / LLM-proposed localized edits** to fix one
measured failure class (e.g. missed blocks), then compare against fine-tuning on the same
positions with the same budget. Expect fine-tuning to win; the comparison itself is the result.

## Implemented (2026-11-20)

| file | what |
|---|---|
| `nn/model.py` | optional score head (7 outputs = Pons score per column); `forward()` unchanged |
| `nn/train_score.py` | soft policy target softmax(score/T), WDL, score regression + listwise ranking loss, mirroring, EMA, `--hard`, `--init`; picks the playing head on validation plies 8-30 |
| `nn/mine_hard.py` | batched GPU self-play of a trained net, solver labels, keeps what it gets wrong |
| `nn/eval_score.py` | benchmark per head, mirror, ensembles, 1-ply score lookahead |
| `nn/numpy_net.py`, `nn/export.py` | score head in model.npz; meta `play_head` + `play_mirror` so website/arena use the best head |
| `tools/train_95.sh` | the whole pipeline, resumable |

Validation split is a fixed hash of each (mirror-canonical) position, so seeds, dataset sizes
and hard-mining rounds all hold out the same positions.

**Smoke test, same 2x32 size and same 285k data as `rs_A_pons_2x32`, 40 epochs:**
old method 0.873, score-head method 0.898 with mirror (0.890 without). +2.5 points before any
extra data. That 2x32 net was only 0.829 optimal on positions from its own games
(`nn/mine_hard.py`), which is what the hard-mining round targets.

Commands: `COMMANDS.md`, section G.

## Results (pipeline finished 2026-11-21, `tools/train_95.sh`, 8x128, 3 seeds)

Data: 3,000,000 solver-labelled positions (2.7 h labelling, 0 timeouts; benchmark positions and
mirrors excluded). Training: 30 epochs, about 3.7 min/epoch on the RTX 3050.

| player (3,000 held-out positions, plies 8-30) | optimal | keeps result |
|---|---|---|
| best old net, rs_A_pons_10x128 policy + mirror | 0.932 | 0.970 |
| old 6-net ensemble + mirror | 0.942 | 0.975 |
| score_8x128_s0, policy head, **no** mirror | 0.966 | |
| score_8x128_s0 / s1 / s2, mirror, best head | 0.972 / 0.970 / 0.969 | 0.99 |
| after hard-position round (`_r1`) | 0.972 / 0.970 / 0.968 | 0.99 |
| best single net + 1-ply score lookahead (`look+`, s0_r1) | **0.975** | **0.993** |
| 3-net ensemble (r1 nets) | 0.974 | 0.990 |

Findings:
- **More data + score head did it:** 0.932 to about 0.970 for one net (+3.8 points). Result-changing
  mistakes fell from about 3% to under 1%.
- **The benchmark curve is flat from epoch ~16** (0.970-0.972) while training loss keeps falling.
- **Hard-position mining did not help** (within seed noise, -0.1 to +0.1). On positions from its own
  games the net was already 94.6% optimal (32k mistakes in 600k positions).
- **Ensembling no longer helps either** (0.974 vs 0.972-0.975 single): the three nets make
  mostly the same mistakes, so what is left is hard for this architecture, not noise.
- 1-ply lookahead with the score head now helps slightly (it hurt with the weak smoke-test net).
- Toward 99%: deeper search with the score head (MCTS / 2-ply), or a bigger net now that data is
  no longer the limit (one 10x128 run would test it).

The website's evaluation bar now defaults to `score_8x128_s0_r1`.
