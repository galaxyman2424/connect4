# Connect 4 Bot — Project Statement

**Course:** CSCI 4150 Intro to Artificial Intelligence, Final Project
**Date:** October 2, 2026

---

## 1. Problem Statement

We will build a local website (served on `localhost`) where a human can play Connect 4 against an AI opponent. The AI chooses its moves with adversarial search: minimax with alpha-beta pruning and a hand-crafted board evaluation. A difficulty slider controls how strong the bot plays. Beyond the playable game, we will measure how much stronger the bot gets as its search depth increases.

## 2. Goals

1. **Playable bot with a difficulty slider.** A human can play a full game in the browser against a bot whose strength is set by a slider (levels 1–10).
2. **Depth vs. strength analysis.** Show, with numbers and charts, how much stronger a deeper-searching bot is than a shallower one, and what that strength costs in computation.

## 3. Input and Output

- **Input:** the current 6×7 board, encoded as a grid of `+1` / `-1` / `0`, plus whose turn it is. The human picks a column by clicking on the board.
- **Output:** the column the bot chooses, along with its evaluation score, the number of positions it searched, and how long the move took.

## 4. Approach

### 4.1 Architecture

A single Python engine is shared by the website and the experiments, so the bot that people play against is exactly the bot that we measure.

```
Final Project/
├── engine/
│   ├── board.py        # 6x7 grid, legal_moves, drop, undo, check_win, is_draw
│   ├── heuristic.py    # scores 4-cell windows: center column bonus, open 3s/2s, opponent threats
│   ├── minimax.py      # negamax + alpha-beta, center-first move ordering, depth parameter,
│   │                   #   node counter, optional transposition table and time limit
│   └── agents.py       # RandomAgent, MinimaxAgent(depth, epsilon), SolverAgent (optional)
├── server/
│   └── app.py          # Flask: GET / serves the page;
│                       #   POST /api/move {board, level} -> {col, score, nodes, ms}
├── web/
│   ├── index.html      # board, difficulty slider, New Game, "who goes first" toggle
│   ├── app.js          # click a column -> check move -> call /api/move -> animate the drop
│   └── style.css
├── experiments/
│   ├── tournament.py   # depth-vs-depth round robin -> results.csv
│   ├── accuracy.py     # bot's move vs. the perfect move on labeled positions
│   ├── tactics.py      # "win in 1" / "must block" test positions
│   └── analysis.ipynb  # heatmap, Elo, win rate vs. depth, nodes/time vs. depth
├── data/               # downloaded datasets (git-ignored)
└── tests/
    └── test_board.py   # wins in every direction, full columns, draws
```

**How a move flows:** the browser sends the board and the slider level to `/api/move`. The server turns the level into a `MinimaxAgent`, which runs `negamax(board, depth, α, β)` and returns the chosen column with its stats. The page then shows the reasoning, for example: *"Bot searched 48,210 positions in 210 ms."*

### 4.2 Difficulty Slider

Search depth alone doesn't make easy levels easy: even a depth-1 bot always takes an immediate win. So the slider combines search depth with a chance of playing a random move.

| Level | Search depth | Random-move chance |
|---|---|---|
| 1 | 1 | 50% |
| 2 | 1 | 20% |
| 3 | 2 | 10% |
| 4 | 3 | 0% |
| 5 | 4 | 0% |
| 6 | 5 | 0% |
| 7 | 6 | 0% |
| 8 | 7 | 0% |
| 9 | 8 | 0% |
| 10 | 9+ | 0%, with a ~2 s time limit |

**Performance:** in plain Python, depth 7–8 can take several seconds per move. Center-first move ordering and a transposition table should handle this. If moves are still too slow, we will switch the board to a bitboard representation (two 64-bit integers).

### 4.3 Depth vs. Strength Experiments

**Design constraint:** minimax is deterministic, so two fixed-depth bots play the same game every time. To get meaningful statistics, each game starts from a random opening of 2–4 moves. Each opening is played twice, with the bots swapping sides, to cancel out the first-move advantage.

1. **Round-robin tournament.** Bots at depths 1–8 play about 100 games per pairing.
   - *Outputs:* a win-rate heatmap (depth vs. depth), and Elo ratings plotted against depth.
2. **Accuracy against perfect play.** On a few thousand positions labeled by a solver, measure the percentage of moves where the bot picks an optimal column, at each depth.
3. **Tactics suite.** About 50 positions that require a win in 1, a forced block, or a win in 3. Report the share solved at each depth.
4. **Cost of strength.** Plot positions searched and milliseconds per move against depth, on a log scale. We expect cost to grow roughly exponentially while strength gains level off.
5. **Optional: odd vs. even depth.** Compare how bots behave when the deepest searched move belongs to the bot vs. the opponent (the horizon effect).

## 5. Evaluation

| Criterion | How it's measured | Success looks like |
|---|---|---|
| **Functionality** | Unit tests and manual play | Rules are enforced, wins and draws are detected, and a full game runs without crashing |
| **Slider works** | Each level plays against a random bot and against the other levels | Higher levels win more often; levels 1–2 can be beaten by a casual player |
| **Strength vs. depth** | Tournament, Elo, accuracy, tactics suite | Clear, statistically supported increase in strength with depth |
| **Tactical soundness** | Tactics suite | At depth ≥ 2, the bot always takes a win in 1 and always blocks an opponent's win in 1 |
| **Responsiveness** | ms per move | Under ~2 s per move at every slider level |

## 6. External Resources

### 6.1 Models (Hugging Face and others)

Hugging Face doesn't have a strong, ready-to-use Connect 4 model, so search remains our primary approach. We may use these as outside opponents or reference points:

| Resource | What it is | Planned use |
|---|---|---|
| [Lyte/QuadConnect2.5-0.5B](https://huggingface.co/Lyte/QuadConnect2.5-0.5B-v0.0.9b) | 0.5B-parameter language model (based on Qwen 2.5) trained to pick a column; its card reports ~14% accuracy | Optional "language model vs. search" comparison |
| [ClementBM/connectfour](https://huggingface.co/spaces/ClementBM/connectfour/tree/main/models) | Small ONNX neural net (673 kB); training method undocumented | Optional outside tournament opponent (loaded with `onnxruntime`) |
| Pascal Pons' Connect 4 solver (GitHub) | Perfect-play solver | Ceiling opponent and source of correct-move labels |
| [OpenEnv Connect4 environment](https://huggingface.co/docs/openenv/main/environments/connect4) | Reinforcement-learning training environment | Only if we later train a learned agent |

*Not used:* [kaiserbuffle/connect4_model_v2](https://huggingface.co/kaiserbuffle/connect4_model_v2) is a robot-arm control model that works from camera images, not a game-playing AI.

### 6.2 Datasets

| Dataset | Contents | Planned use |
|---|---|---|
| [UCI Connect-4](https://archive.ics.uci.edu/dataset/26/connect+4) | 67,557 positions after 8 moves, labeled win/loss/draw for the first player | Embedding-experiment corpus; quick accuracy check |
| [TonyCWang/ConnectFour](https://huggingface.co/datasets/TonyCWang/ConnectFour) | ~160M solver-labeled positions with a perfect-play score for each of the 7 columns | Accuracy experiment. The full set is 46–211 GB, so we will stream a subset with `streaming=True` |
| [Leon-LLM/Connect-Four-Datasets-Collection](https://huggingface.co/datasets/Leon-LLM/Connect-Four-Datasets-Collection) | ~460 MB of text in two folders (LC4N, SC4N), likely game move sequences (to be confirmed) | Opening variety; additional embedding corpus |
| Self-play games | Generated by `tournament.py` | Embedding corpus with labels we control |

## 7. Relation to the Embedding-Neighbor Experiment

Board states from the UCI dataset and from self-play will be the corpus for the embedding-neighbor artifact. As recorded in the dossier, embeddings are used **only** for "show similar positions" queries. Move selection stays with minimax and alpha-beta, because similar-looking boards don't necessarily share the same best move. The neural evaluation-function hypothesis remains **deferred**.

## 8. Milestones

| # | Milestone | Deliverable |
|---|---|---|
| 1 | Board engine | `board.py` passing all rule tests |
| 2 | Search | Heuristic and minimax; depth 4 beats a random bot almost every game |
| 3 | Website | Flask server and web page with the slider (**Goal 1 complete**) |
| 4 | Experiment harness | Tactics suite; tournament script with randomized openings |
| 5 | Ground-truth evaluation | UCI and TonyCWang data loaded; accuracy experiment |
| 6 | Analysis | Notebook charts and write-up (**Goal 2 complete**) |

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Deep searches are too slow in Python | Move ordering, transposition table, time-limited iterative deepening; bitboards if needed |
| Deterministic bots produce repeated games | Randomized openings, played twice with sides swapped |
| Large dataset is too big to download | Stream a fixed-size sample |
| Outside models are undocumented or weak | Treat them as optional comparisons, not core dependencies |
