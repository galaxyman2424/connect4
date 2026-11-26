# Commands cheat sheet

Every command you need, in order. Run them from the project folder (`Final Project/`, the
one with `server/`, `nn/`, `arena/` in it). More detail on training:
[`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md).

- **Part A** runs the website, on any computer (the Windows PC is fine; no GPU or PyTorch needed).
- **Part B** trains the neural network, on the Linux Mint PC with the RTX 3050.
- **Part C** runs the bot tournament.
- **Part D** moves results from the training PC to the website PC.
- **Part E** trains the five extra methods (evolution, TD n-tuple, DQN, PPO, distillation),
  measures their training cost and compares everything. Details:
  [`docs/OTHER_METHODS.md`](docs/OTHER_METHODS.md).

---

> **C is the standard backend.** After `python -m engine.cbuild` (once per machine), the network
> (engine/cminimax/net.c), the MCTS tree (mcts.c) and the hand-crafted minimax (minimax.c) all run
> in C by default, everywhere: website, arena, nn.evaluate, bench_stats. Same moves as numpy
> (`python -m nn.native --bench`). Without the library everything falls back to numpy.
> Force a backend with `C4_NN_BACKEND=numpy|torch|native`. New hot-path code goes in C first.

## A. Run the website (Windows)

### First time only

```powershell
cd "C:\Users\Connor\Desktop\csci4150\Final Project"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell refuses to run `activate`, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Every time

```powershell
cd "C:\Users\Connor\Desktop\csci4150\Final Project"
.venv\Scripts\activate
python server/app.py
```

Then open in a browser:

| page | URL |
|---|---|
| Play against the bot | http://localhost:5000 |
| Lab (training charts, hypothesis test, bot arena) | http://localhost:5000/lab |

Stop the server with `Ctrl+C`. On the play page, **Opponent → Neural net** shows up once a
`runs/<name>/model.npz` exists.

### Run the tests (optional)

```powershell
python -m pytest -q
```

---

## B. Train the neural network (Linux Mint, RTX 3050)

### First time only

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev build-essential

git clone https://github.com/galaxyman2424/connect4.git
cd connect4

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-train.txt -r arena/requirements.txt

# check that PyTorch sees the GPU (should print True and list the RTX 3050)
python -c "import torch; print(torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

# build the perfect-play solver and get its opening book
cd tools/pons && make c4solver && cd ../..
python data/download_data.py --book     # or copy tools/pons/7x6.book from the Windows PC
```

### Every new terminal

```bash
cd connect4
source .venv/bin/activate
# only needed if the PC has more than one GPU (e.g. an old GTX 750 Ti next to the RTX 3050):
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0           # the RTX 3050's number in `nvidia-smi -L`
```

`--device auto` (the default) uses the CUDA GPU with the most memory that this PyTorch build can
actually run, and falls back to the CPU (with a warning) if there is none. Check which device a
run used in `runs/<run>/config.json` (`"device"`).

### 1. Smoke test (about 10 min, checks that everything works)

```bash
python -m nn.make_dataset --positions 20000 --out nn_data/smoke.npz
python -m nn.train_supervised --data nn_data/smoke.npz --run smoke_sup --epochs 3
python -m nn.train_selfplay --run smoke_az --preset quick --generations 3
python -m nn.evaluate --run smoke_sup --quick
rm -r runs/smoke_*                      # clean up afterwards
```

### 2. Phase A: learn from perfect play (supervised)

```bash
python -m nn.make_dataset --positions 300000 --out nn_data/solver_300k.npz
python -m nn.train_supervised --data nn_data/solver_300k.npz --run sup_6x64 --epochs 30
python -m nn.evaluate --run sup_6x64 --model best.npz
```

### 3. Phase B: learn from nothing (AlphaZero self-play)

```bash
# runs for hours; nohup keeps it going if you close the terminal
nohup python -m nn.train_selfplay --run az_6x64 --preset standard > runs/az_6x64.log 2>&1 &
tail -f runs/az_6x64.log                # watch progress (Ctrl+C stops watching, not training)
```

| need to... | command |
|---|---|
| stop training | `pkill -f nn.train_selfplay` (progress is saved every generation) |
| continue after a stop | `python -m nn.train_selfplay --run az_6x64 --resume` |
| train longer | `python -m nn.train_selfplay --run az_6x64 --resume --generations 150` |
| evaluate when done | `python -m nn.evaluate --run az_6x64` |

Presets: `quick` (smoke test), `standard` (about 100k games), `long` (overnight-plus).

---

## C. Bot tournament (arena)

Run this on the Linux PC; it needs PyTorch and the solver.

```bash
python -m arena.fetch                   # download the open-source bots (first time only)
python -m arena.tournament --list       # every bot should say "ready"
python -m arena.tournament --openings 20 \
    --add-nn nn:runs/az_6x64/model.npz:mcts:200 \
    --add-nn nn:runs/sup_6x64/best.npz:mcts:200
```

To continue an interrupted tournament: add `--resume` to the same command.

---

## D. Move results to the website PC

On the Linux PC:

```bash
git add runs/ experiments/results/arena/
git commit -m "Training runs and arena results"
git push
```

On the Windows PC:

```powershell
git pull origin main
.venv\Scripts\activate
python server/app.py                    # then open http://localhost:5000/lab
```

When your real runs are in, delete the pilot data I made for testing:

```bash
rm -r runs/pilot_cpu_az runs/pilot_cpu_sup experiments/results/arena/pilot_cpu
```

On Windows (PowerShell): `Remove-Item -Recurse -Force runs\pilot_cpu_az, runs\pilot_cpu_sup, experiments\results\arena\pilot_cpu`

---

## E. More learning methods + training cost (Linux PC)

Same terminal setup as Part B (`source .venv/bin/activate`).
Each trainer measures its own time, CPU core-hours, GPU energy and cost into
`runs/<run>/cost.json`. Times below are rough guesses; check the log's first lines.

```bash
python -m pytest -q tests/test_rl.py            # quick check (~10 s)

# CPU-only methods (can run at the same time as one GPU job)
nohup python -m rl.evolve --run evo_d2 --generations 60 --depth 2 > runs/evo_d2.log 2>&1 &         # ~20-40 min
nohup python -m rl.ntuple --run td_ntuple --games 3000000 > runs/td_ntuple.log 2>&1 &             # 1 core, ~1-2 h

# GPU methods (one at a time)
nohup python -m rl.ppo --run ppo_6x64 --iterations 300 > runs/ppo_6x64.log 2>&1 &                 # ~45-90 min
nohup python -m rl.dqn --run dqn_6x64 --iterations 400 > runs/dqn_6x64.log 2>&1 &                 # ~45-90 min

# combinations
python -m rl.ppo --run ppo_from_sup --init runs/sup_6x64/checkpoints/best.pt --iterations 300     # supervised warm start
python -m rl.dqn --run dqn_from_sup --init runs/sup_6x64/checkpoints/best.pt --iterations 400
python -m rl.distill --run distill_az_2x32 --teacher nn:runs/az_6x64/model.npz:mcts:100           # search -> tiny net
python -m rl.distill --run distill_ens_2x32 --teacher nn:runs/az_6x64/model.npz:mcts:100 \
    --teacher nn:runs/ppo_6x64/model.npz --teacher ntuple:runs/td_ntuple/ntuple.npz --teacher-weights 2,1,1

# cost of runs trained before the meter existed (estimate from logged time x watts)
python -m rl.meter --estimate --run az_6x64 --gpu-watts 100 --cpu-watts 90
# or measure any command, e.g. a new supervised run:
python -m rl.meter --run sup_m -- python -m nn.train_supervised --data nn_data/solver_300k.npz --run sup_m --epochs 30

# compare everything: Elo, accuracy, ms/move, training hours, GPU-h, kWh, $
python -m rl.compare
```

| need to... | command |
|---|---|
| continue a stopped run | add `--resume` (and e.g. a bigger `--iterations` / `--games` / `--generations`) |
| evaluate a DQN/PPO/distilled net like Part 2 | `python -m nn.evaluate --run ppo_6x64` |
| see results | Lab page, **Method comparison** tab, and each run under **Neural net training** |
| add the new players to the arena | `--add-nn evo:runs/evo_d2/weights.json:6 --add-nn ntuple:runs/td_ntuple/ntuple.npz:4` |

Commit `runs/` and `experiments/results/methods/` like in Part D.

---

## F. Research phase extras (branch `eval-bar-and-accuracy`)

### Top-15 bots in a new arena run (Linux PC)

```bash
cd ~/projects/introAI/connect4 && source .venv/bin/activate
python -m arena.tournament --list --roster top15          # all 15 should say "ready"
# new run: 100 fresh openings (seed 2 = different from part3/top15), ~21,000 games, about 4 h on 6 workers
tmux new -s arena
python -m arena.tournament --roster top15 --name top15_o100 --openings 100 --seed 2 --workers 6
#   detach: Ctrl+b then d    re-attach: tmux attach -t arena
python -m arena.tournament --roster top15 --name top15_o100 --openings 100 --seed 2 --workers 6 --resume   # if it stopped
python -m arena.tournament --roster top15 --name top15_o100 --summary-only   # Elo from the games so far
```
Results: `experiments/results/arena/top15_o100/summary.json` (shows up on the Lab page).

### Paper outline (was the Paper builder tab)

```bash
python -m research.paper_outline              # every finding, top-15 bot and sweep run
python -m research.paper_outline --selected   # only what is ticked on /research
```
Writes `docs/PAPER_OUTLINE.md` (gitignored, local only).

### Evaluation bar / accuracy

- The play page shows a chess-style bar beside the board; pick the net or the Pons solver under the board.
- `python -m experiments.ensemble_probe` measures single nets vs an ensemble on the 3,000-position benchmark (about 8 min on CPU). Plan for 95%+: `docs/ACCURACY_95.md`.

---

## G. Score-head nets: aiming for 95%+ (Linux PC, branch `eval-bar-and-accuracy`)

Everything in one go (resumable; ~15-18 h):

```bash
cd ~/projects/introAI/connect4 && source .venv/bin/activate
tmux new -s c4train
bash tools/train_95.sh              # SIZE=6x64 or SEEDS="0" to make it shorter
#   detach: Ctrl+b then d    re-attach: tmux attach -t c4train    progress: tail -f logs/*.log
```

Or step by step:

```bash
# 1. 3M solver-labelled positions (~3.5 h, CPU)
python -m nn.make_dataset --positions 3000000 --workers 14 --seed 2027 --out nn_data/solver_3m.npz
# 2. score-head net (~5 min/epoch for 8x128 on the 3050); repeat with --seed 1, 2 for the ensemble
python -m nn.train_score --data nn_data/solver_3m.npz --run score_8x128_s0 --blocks 8 --channels 128 --epochs 30 --seed 0
# 3. positions it gets wrong in its own games (~1 h)
python -m nn.mine_hard --run score_8x128_s0 --positions 600000 --out nn_data/hard_8x128_s0.npz
# 4. fine-tune with them
python -m nn.train_score --data nn_data/solver_3m.npz --hard nn_data/hard_8x128_s0.npz --hard-weight 2 \
    --init runs/score_8x128_s0/checkpoints/best.pt --run score_8x128_s0_r1 --epochs 10 --lr 6e-4
# 5. benchmark (per head, + ensemble, + 1-ply lookahead)
python -m nn.eval_score --runs score_8x128_s0,score_8x128_s0_r1 --lookahead
python -m nn.eval_score --runs score_8x128_s0_r1,score_8x128_s1_r1,score_8x128_s2_r1 --lookahead
```

Watch `BENCH ...` in each epoch line. Quick start without the 3.5 h of labelling: train on
`nn_data/solver_300k.npz`, or warm-start from the old net with
`--init runs/rs_A_pons_8x128/checkpoints/best.pt` (its score head starts from scratch).

---

## H. Speed-ups, benchmark precision, PCA and LDA (Linux PC)

Background and measured numbers: [`docs/SPEED_PCA_LDA.md`](docs/SPEED_PCA_LDA.md).

```bash
cd ~/projects/introAI/connect4 && source .venv/bin/activate
python -m pytest -q tests/test_speedups.py          # ~5 s: speed-ups give identical moves

# 1. training speed on THIS GPU (~5 min each): which --speed / --batch is fastest
python -m nn.train_speedtest                         # 8x128 score net
python -m nn.train_speedtest --blocks 6 --channels 64
#    then e.g.  python -m nn.train_score ... --speed max --batch 2048 --lr 3e-3

# 2. faster evaluation: same numbers, torch on the GPU (--backend torch)
python -m nn.evaluate --run score_8x128_s0_r1 --model best.npz --backend torch --mcts-batch 1024
python -m nn.bench_stats --runs score_8x128_s0_r1,score_8x128_s1_r1,score_8x128_s2_r1 --backend torch
python -m nn.bench_stats --runs score_8x128_s0_r1,runs/score_8x128_s0_r1/best.npz+runs/score_8x128_s1_r1/best.npz+runs/score_8x128_s2_r1/best.npz --backend torch
#    arena with the project's nets on torch:  C4_NN_BACKEND=torch C4_NN_THREADS=1 python -m arena.tournament ...

# 3. a 20,000-position benchmark (CPU, minutes) that no training set contains
python experiments/data_prep/make_accuracy_set.py --n 20000 --workers 14 --seed 777 \
    --out data/accuracy_positions_20k.csv \
    --exclude data/accuracy_positions.csv,nn_data/solver_3m.npz,nn_data/solver_300k.npz,nn_data/hard_8x128_s0.npz,nn_data/hard_8x128_s1.npz,nn_data/hard_8x128_s2.npz
#    (add any other nn_data/*.npz you trained on; missing files are skipped with a warning)
python -m nn.bench_stats --runs score_8x128_s0_r1,score_8x128_s1_r1 --bench data/accuracy_positions_20k.csv --backend torch
python -m nn.evaluate --run score_8x128_s0_r1 --model best.npz --backend torch \
    --bench data/accuracy_positions_20k.csv --out eval_bench20k.json --skip ladder,hypothesis,tactics

# 4. PCA input reduction: 8 nets (6x64, solver_300k, 20 epochs) ~25-40 min GPU + evaluation
python -m research.pca_lda pca-variance                       # CPU, seconds
python -m research.pca_lda pca-sweep                          # prints the commands
tmux new -s pca
python -m research.pca_lda pca-sweep --go                     # resumable; logs: runs/pca_lda_logs/
python -m research.pca_lda pca-summary                        # experiments/results/research/pca_lda/
#    bigger version: --data nn_data/solver_3m.npz --size 8x128 --epochs 30 --seeds 0,1

# 5. LDA: how separable is the data, then focus training on the LDA-hard positions
python -m research.pca_lda lda-analyze --run score_8x128_s0_r1   # CPU ~1-2 min (+ trunk features on GPU)
python -m research.pca_lda lda-focus                             # nn_data/lda_focus.npz + _control.npz
python -m research.pca_lda lda-sweep --go                        # baseline / focus / control, ~15 min GPU
python -m research.pca_lda lda-summary                           # paired focus-vs-control test
#    variants: lda-focus --select misclassified | --select class:draw | --label bestcol | --frac 0.1
```

The MCP server (`mcp_server/`, git-ignored, so copy it to this PC yourself) can start all of
these as background jobs: see `mcp_server/README.md`.

---

## I. Bigger benchmark, search sweeps, hybrid, ensemble distillation, C minimax (Linux PC)

Background and measured numbers: [`docs/SEARCH_AND_ENSEMBLES.md`](docs/SEARCH_AND_ENSEMBLES.md).

```bash
cd ~/projects/introAI/connect4 && source .venv/bin/activate

# 0. C minimax + C MCTS tree (identical results); used automatically once built
python -m engine.cbuild --check
python -m nn.cmcts --bench --run score_8x128_s0_r1 --backend torch --positions 1024 --sims 800 --batch 1024
python -m nn.native --bench                     # native C net + MCTS vs numpy vs torch (same moves)
#    play with the native backend (arena / website / ladder):  C4_NN_BACKEND=native python -m arena.tournament ...
python -m pytest -q tests/test_cminimax.py tests/test_cmcts.py tests/test_native.py tests/test_speedups.py

# 1. 50,000-position benchmark (per-bucket CI +-0.5 instead of +-1.9), CPU ~10-20 min
python experiments/data_prep/make_accuracy_set.py --n 60000 --workers 14 --seed 20261125 \
    --out data/accuracy_positions_60k_raw.csv --exclude data/accuracy_positions.csv
python experiments/data_prep/filter_bench.py --bench data/accuracy_positions_60k_raw.csv \
    --exclude nn_data/solver_3m.npz,nn_data/solver_300k.npz,nn_data/hard_8x128_s0.npz,nn_data/hard_8x128_s1.npz,nn_data/hard_8x128_s2.npz \
    --per-bucket 8333 --out data/accuracy_positions_50k.csv
#    (data/accuracy_positions_60k_raw.csv may already be in the repo: then only run filter_bench)
python -m nn.bench_stats --runs score_8x128_s0_r1,score_8x128_s1_r1,score_8x128_s2_r1 --bench data/accuracy_positions_50k.csv --backend torch
python -m nn.evaluate --run score_8x128_s0_r1 --model best.npz --backend torch --bench data/accuracy_positions_50k.csv \
    --mcts-sample 6000 --sims 200,800 --out eval_bench50k.json --skip ladder,hypothesis

# 2. how much does search help (sims x value source, tree depth), GPU ~30-60 min each
python -m nn.search_sweep --run score_8x128_s0_r1 --sims 0,50,200,800,1600,3200 --values wdl,score --positions 3000
python -m nn.search_sweep --run score_8x128_s0_r1 --sims 0,200,800 --root-exact-from 16 --exact-from 20 --positions 3000
python -m nn.search_sweep --run ens3 --model runs/score_8x128_s0_r1/best.npz+runs/score_8x128_s1_r1/best.npz+runs/score_8x128_s2_r1/best.npz \
    --sims 0,200,800 --root-exact-from 16 --exact-from 20 --positions 3000
#    results: experiments/results/search_sweep_<run>[_exact20][_root16].json

# 3. hybrid player in the arena: net + MCTS until ply 16, exact C search after
python -m arena.tournament --roster top8 --name top8_hybrid --openings 100 --seed 2 --workers 6 \
    --add-nn nn:runs/score_8x128_s0_r1/best.npz:hybrid:16:200

# 4. three ensemble nets -> one net (knowledge distillation), GPU ~2.5 h each at 8x128
python -m nn.train_score --data nn_data/solver_3m.npz --hard nn_data/hard_8x128_s0.npz --hard-weight 2 \
    --run distill_ens3_8x128 --blocks 8 --channels 128 --epochs 30 --seed 7 \
    --teacher runs/score_8x128_s0_r1/best.npz+runs/score_8x128_s1_r1/best.npz+runs/score_8x128_s2_r1/best.npz
python -m nn.train_score --data nn_data/solver_3m.npz --run distill_ens3_10x128 --blocks 10 --channels 128 --epochs 30 --seed 7 \
    --teacher runs/score_8x128_s0_r1/best.npz+runs/score_8x128_s1_r1/best.npz+runs/score_8x128_s2_r1/best.npz
python -m nn.bench_stats --runs distill_ens3_8x128,score_8x128_s0_r1,runs/score_8x128_s0_r1/best.npz+runs/score_8x128_s1_r1/best.npz+runs/score_8x128_s2_r1/best.npz \
    --bench data/accuracy_positions_50k.csv --backend torch
```

---

## Where things end up

| what | where |
|---|---|
| training charts data | `runs/<run>/metrics.jsonl` |
| trained model (used by website + arena) | `runs/<run>/model.npz` |
| full evaluation + hypothesis test | `runs/<run>/eval.json` |
| your notes for the presentation | `runs/<run>/notes.md` |
| tournament results | `experiments/results/arena/<name>/summary.json` |
| training cost of a run | `runs/<run>/cost.json` |
| evolved weights / n-tuple table | `runs/<run>/weights.json`, `runs/<run>/ntuple.npz` |
| method comparison | `experiments/results/methods/<name>/summary.json` (+ csv, png charts) |
| PCA / LDA experiments | `experiments/results/research/pca_lda/` (+ runs `pca_k*`, `lda_*`; logs `runs/pca_lda_logs/`) |
| training speed test | `experiments/results/speedtest_<blocks>x<channels>.json` |
| search sweeps (sims, value source, tree depth, hybrid) | `experiments/results/search_sweep_<run>*.json` |
| bigger benchmark | `data/accuracy_positions_50k.csv` (made by `filter_bench.py`) |
| jobs started from the MCP server | `mcp_server/jobs/<id>/output.log` (git-ignored) |
