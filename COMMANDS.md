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
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0           # the RTX 3050's number in `nvidia-smi -L`
```

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

Same terminal setup as Part B (`source .venv/bin/activate`, `CUDA_VISIBLE_DEVICES=0`).
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
