# Connect 4 Bot (CSCI 4150 Intro to AI, final project)

A local website where you play Connect 4 against a minimax bot (negamax + alpha-beta, transposition
table, iterative deepening, hand-crafted window heuristic) with a 10-level difficulty slider, plus the
experiments that measure how much stronger the bot gets as it searches deeper and what that costs.

* Project statement: [`connect4-project-statement.md`](connect4-project-statement.md)
* Results write-up: [`docs/REPORT.md`](docs/REPORT.md)
* Engine API shared by the website and the experiments: [`CONTRACT.md`](CONTRACT.md)
* **Part 2: our own neural network + open-source bot arena**
  * [`docs/NEURAL_NET.md`](docs/NEURAL_NET.md): design, method, metrics
  * [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md): copy-paste steps for the GPU training PC
  * [`docs/ARENA.md`](docs/ARENA.md): the open-source bots, adapters, tournament protocol
  * [`docs/PRESENTATION_GUIDE.md`](docs/PRESENTATION_GUIDE.md): the story, slide plan, numbers to fill in, likely questions
  * Lab page: `python server/app.py`, then open http://localhost:5000/lab

```
engine/       board (bitboards), heuristic, minimax search, agents + slider levels
server/       Flask app: GET / serves web/, POST /api/move {board, level}
web/          index.html, app.js, style.css
tests/        pytest suite (rules, search, server)
experiments/  tournament, tactics, accuracy, cost, odd/even, embedding, analysis.ipynb
data/         solver-labeled accuracy set (committed) + dataset download script
tools/        Pascal Pons' perfect-play solver (C++ source, AGPL-3.0) + Python wrapper
docs/         REPORT.md, screenshots, NEURAL_NET.md, TRAINING_RUNBOOK.md, ARENA.md, PRESENTATION_GUIDE.md
nn/           neural network: model, MCTS, numpy inference, datasets, supervised + self-play training, evaluation
runs/         one folder per training run (metrics + exported model; shown on the Lab page)
arena/        open-source bot adapters, fetch script, round-robin tournament
web/lab.*     the Lab (data) page
```

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Play

```bash
python server/app.py               # then open http://localhost:5000
```

Pick a level (1 = very easy, 10 = strongest, about 2 s per move), choose who goes first, click a
column. Under the board the page shows the bot's reasoning ("searched N positions in M ms").

## Lab page and neural-net opponent

`python server/app.py`, then open http://localhost:5000/lab. Four tabs: **Neural net training**
(charts from `runs/<run>/metrics.jsonl`), **Hypothesis test** (`runs/<run>/eval.json`), **Bot
arena** (`experiments/results/arena/<name>/summary.json`) and **How it works**. Pages
show the exact command to run when there's no data yet. Once a run folder with a `model.npz`
exists, the play page gets an **Opponent: Neural net** switch. The website only needs numpy,
not PyTorch.

Train on a GPU machine: see [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md). In short:

```bash
pip install -r requirements-train.txt                      # PyTorch
python -m nn.make_dataset --positions 300000               # needs tools/pons/c4solver + 7x6.book
python -m nn.train_supervised --run sup_6x64               # Phase A: learn from perfect play
python -m nn.train_selfplay --run az_6x64 --preset standard  # Phase B: AlphaZero self-play
python -m nn.evaluate --run az_6x64                        # full evaluation + hypothesis test
python -m arena.fetch && python -m arena.tournament --add-nn nn:runs/az_6x64/model.npz:mcts:200
```

## Tests

```bash
python -m pytest -q                # 243 tests, ~15 s (the PyTorch test is skipped if torch isn't installed)
```

## Reproduce the experiments

Run from the project root. Every script takes `--help`, is seeded, uses all CPU cores
(`--workers N` to change), and writes CSV/JSON to `experiments/results/` and PNGs to
`experiments/figures/`. Times below were measured on a 2-core cloud VM (Python 3.13); a modern laptop
is usually faster.

| Step | Command | Time (2 cores) | Needs solver? |
|---|---|---|---|
| Round robin depths 1-8 (2,800 games) + slider levels (L10 games bounded) | `python experiments/tournament.py` | ~9 + ~10 min | no |
| Tactics suite (52 positions x depths 1-8) | `python experiments/tactics.py` | ~5 s | only with `--regenerate` |
| Accuracy vs perfect play (3,000 positions, depths 1-10) | `python experiments/accuracy.py` | ~3.5 min | no (labels are in the CSV) |
| Cost of strength + ablations + per-level ms | `python experiments/cost.py` | ~3.5 min | no |
| Odd vs even depth (derived) | `python experiments/oddeven.py` | 1 s | no |
| Embedding-neighbor artifact | `python experiments/embedding.py` | ~1 min | optional (best-move labels; falls back to depth-10 search) |
| All charts | `python experiments/plots.py` or run `experiments/analysis.ipynb` | ~15 s | no |

Execute the notebook non-interactively (saves outputs in place):

```bash
jupyter nbconvert --to notebook --execute --inplace experiments/analysis.ipynb
```

Quick smoke run of the tournament (a minute or two), written to a scratch folder so the committed results are not overwritten:
`python experiments/tournament.py --openings 5 --l10-openings 1 --out smoke/results.csv`.
The slider part reuses the depth games, so to replay only one part use `--parts depth` or `--parts slider`.

### Public datasets (optional)

The build environment could not reach Hugging Face or the UCI archive, so the committed experiments use
our own solver-labeled set (`data/accuracy_positions.csv`, see [`data/README.md`](data/README.md)).
On your own machine:

```bash
pip install -r data/requirements-data.txt
python data/download_data.py --all                 # or --uci / --tonycwang 5000 / --leon / --book
python experiments/accuracy.py --source tonycwang  # accuracy on the HF TonyCWang sample
python experiments/accuracy.py --source uci        # eval-sign vs UCI win/loss labels (weak check)
python experiments/embedding.py --corpus uci       # embedding corpus = UCI positions
```

If a file is missing the scripts say which download step to run.

### The solver (Windows note)

`tools/pons/c4solver` is a binary built from Pascal Pons' C++ solver (git-ignored; build it yourself). It is only needed to
create *new* labels (`tactics.py --regenerate`, `data/` regeneration scripts, best-move labels in
`embedding.py`). Every other experiment, the website and the tests run without it. On macOS/Linux
build it with `cd tools/pons && make c4solver`. On Windows a Linux build does not run, so it must be
compiled; the easiest route is WSL (`make c4solver` and run the label-generating step from WSL). A native
MinGW/MSVC build produces `c4solver.exe`, but `tools/solver.py` and `experiments/` look for exactly
`tools/pons/c4solver`, so stick to WSL unless you adjust that path. Without the opening book (`7x6.book`, see `tools/README.md`) positions with fewer than about
12 stones are slow to solve.
