# runs/

One folder per neural-network training run (created by `nn/train_supervised.py` or
`nn/train_selfplay.py`). The Lab page (`/lab`) lists every folder here.

| file | what | in git? |
|---|---|---|
| `config.json` | every setting, model size, machine (GPU, torch version) | yes |
| `metrics.jsonl` | one JSON line per epoch / generation (losses, accuracy, Elo, ...) | yes |
| `model.npz` | latest weights, numpy format (website + arena use this) | yes |
| `best.npz` | supervised runs: weights with the lowest validation loss | yes |
| `eval.json` | output of `python -m nn.evaluate --run <name>` | yes |
| `notes.md` | your own notes: why you ran it, what you saw (shown on the Lab page) | yes |
| `checkpoints/` | PyTorch `.pt` files + replay buffer (for `--resume`) | no (big) |

To show results on another computer: commit the run folder (git ignores `checkpoints/`),
push, pull on the other machine, start `python server/app.py`, open http://localhost:5000/lab.
