# Training runbook: Linux Mint + RTX 3050

Copy-paste steps to go from a fresh clone on the training PC to charts on the Lab page.
Target machine: Ryzen 7 5800X (8 cores / 16 threads), 16 GB RAM, RTX 3050 8 GB (+ GTX 750 Ti),
Linux Mint 22.3, NVIDIA driver 580.173.02 (CUDA 13.0).

Run every command from the project root (`connect4/`) unless it says otherwise. Keep a
log of what you ran and when in `runs/<run>/notes.md`. The Lab page shows that file,
and it's the "steps I took" record for the presentation.

---

## 0. One-time setup (about 15 minutes)

```bash
sudo apt update
sudo apt install -y git python3-venv python3-dev build-essential

git clone https://github.com/galaxyman2424/connect4.git
cd connect4

python3 -m venv .venv
source .venv/bin/activate              # do this in every new terminal
pip install --upgrade pip
pip install -r requirements.txt -r requirements-train.txt -r arena/requirements.txt
```

`pip install torch` on Linux installs the CUDA build (torch 2.14 uses CUDA 13.0, which your
driver 580 supports). Check that PyTorch sees the GPUs:

```bash
nvidia-smi -L
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
python -c "from nn.common import pick_device; print(pick_device())"
```

`pick_device()` picks the CUDA GPU with the most memory that PyTorch can actually run on.
That's the RTX 3050. The GTX 750 Ti (Maxwell, 2014) isn't supported by current PyTorch builds,
so it's skipped. To pin the 3050 by hand (recommended, so nothing ever lands on the 750 Ti):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0          # the index nvidia-smi -L shows for the RTX 3050
```

Build the perfect-play solver and get its opening book (needed for Phase A data and for
the Pons bot in the arena):

```bash
cd tools/pons && make c4solver && cd ../..
python data/download_data.py --book    # or copy tools/pons/7x6.book over from the Windows PC (USB)
python tools/solver.py                 # self-test
```

Run the tests (about 15 s):

```bash
python -m pytest -q
```

---

## 1. Smoke test (about 10 minutes): does everything run?

```bash
python -m nn.make_dataset --positions 20000 --out nn_data/smoke.npz
python -m nn.train_supervised --data nn_data/smoke.npz --run smoke_sup --epochs 3
python -m nn.train_selfplay --run smoke_az --preset quick --generations 3
python -m nn.evaluate --run smoke_sup --quick
```

Each script prints one line per epoch/generation. Look at the self-play line's timing,
`[self-play Xs, train Ys, eval Zs]`: it tells you how long a full run will take.
Delete the smoke runs afterwards (`rm -r runs/smoke_*`) so they don't clutter the Lab page.

---

## 2. Phase A: supervised learning from perfect play

**Step 2a, data (about 10-20 min with 14 workers):**

```bash
python -m nn.make_dataset --positions 300000 --workers 14 --out nn_data/solver_300k.npz
```

This writes `nn_data/solver_300k.json` with outcome counts and a ply histogram. Note those
numbers in your notes. The 3,000 benchmark positions are automatically excluded.

**Step 2b, train (about 10-30 min on the RTX 3050):**

```bash
python -m nn.train_supervised --data nn_data/solver_300k.npz --run sup_6x64 --epochs 30
```

**Optional experiments** (each is a separate run, so you can compare them on the Lab page):

```bash
# bigger network
python -m nn.train_supervised --data nn_data/solver_300k.npz --run sup_10x128 --blocks 10 --channels 128 --epochs 30
# more data (generate 1M positions first)
python -m nn.make_dataset --positions 1000000 --out nn_data/solver_1m.npz
python -m nn.train_supervised --data nn_data/solver_1m.npz --run sup_6x64_1m --epochs 20
```

**Evaluate:**

```bash
python -m nn.evaluate --run sup_6x64 --model best.npz
```

---

## 3. Phase B: AlphaZero self-play (the "learn it from nothing" run)

```bash
python -m nn.train_selfplay --run az_6x64 --preset standard
```

* `standard` = 100 generations x 1,000 games, 128 MCTS simulations per move (about 100k games).
  Check the first generation's timing and multiply by 100. Every generation is saved, so
  you can stop at any time with Ctrl+C and continue later:

```bash
python -m nn.train_selfplay --run az_6x64 --resume
python -m nn.train_selfplay --run az_6x64 --resume --generations 150     # extend the run
```

* Leave it running overnight with `nohup` so closing the terminal doesn't kill it:

```bash
nohup python -m nn.train_selfplay --run az_6x64 --preset standard > runs/az_6x64.log 2>&1 &
tail -f runs/az_6x64.log
```

* Tuning knobs, if the GPU or CPU is idle:
  * `--workers` (default 8): self-play processes. Each one uses a CPU core for the tree
    search and ~0.4 GB of GPU memory. With 16 threads, 8-12 is reasonable. Watch `htop` and
    `nvidia-smi`.
  * `--parallel` (default 128): games each worker plays at once = GPU batch size.
  * `--sims`: more simulations means stronger self-play games but slower generations.
* **Optional comparison:** start AlphaZero from the supervised network instead of random
  weights: `--init runs/sup_6x64/checkpoints/best.pt --run az_from_sup`

**Evaluate when done** (the minimax ladder and hypothesis test use all CPU cores, about 30-60 min):

```bash
python -m nn.evaluate --run az_6x64
```

---

## 4. Bot arena (about 1-3 hours, unattended)

```bash
python -m arena.fetch                  # download the pinned open-source bots (~30 MB)
python -m arena.tournament --list      # every bot should say "ready"
python -m arena.tournament --openings 20 --workers 14 \
    --add-nn nn:runs/az_6x64/model.npz:mcts:200 \
    --add-nn nn:runs/sup_6x64/best.npz:mcts:200
```

`--resume` continues an interrupted tournament. Use `--name <x>` to keep several
tournaments side by side (they all appear in the Lab page's menu).

---

## 5. Bring the results to the presentation computer

```bash
git add runs/ experiments/results/arena/
git commit -m "Training runs + arena results"
git push
```

On the Windows PC: `git pull`, then `python server/app.py`, and open http://localhost:5000/lab.
The website only needs numpy, not PyTorch. The play page now has an **Opponent: Neural net**
switch that uses `runs/<run>/model.npz`.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `torch.cuda.is_available()` is False | `nvidia-smi` must work first; reinstall torch inside the venv (`pip install --force-reinstall torch`) |
| CUDA out of memory | lower `--workers` or `--parallel`; close other GPU programs |
| "no kernel image is available" | the job landed on the GTX 750 Ti: set `CUDA_VISIBLE_DEVICES` to the 3050 (step 0) |
| solver binary not found | `cd tools/pons && make c4solver` |
| make_dataset is very slow early on | the opening book is missing (`tools/pons/7x6.book`) |
| Lab page shows nothing | the run folder needs `config.json` + `metrics.jsonl`; click Refresh |
| self-play generation takes far longer than expected | `htop`: if the CPU isn't saturated, raise `--workers`; if the GPU is at 100%, lower `--sims` |
