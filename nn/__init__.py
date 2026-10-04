"""Neural-network Connect 4 player (CSCI 4150 final project, part 2).

Layout
  encode.py          board bitboards -> input planes (numpy), mirror augmentation
  model.py           PyTorch ResNet with a policy head (7 moves) and a WDL value head
  numpy_net.py       numpy-only inference of an exported model (.npz) - no torch needed
  evaluator.py       load a model (.npz or .pt) -> batched evaluate(curs, masks)
  mcts.py            batched PUCT Monte Carlo Tree Search (AlphaZero style)
  agents.py          PolicyAgent, MCTSAgent, NNMinimaxAgent (same interface as engine.agents)
  common.py          paths, run folders, metrics logging, GPU selection
  bench.py           benchmarks: accuracy vs perfect play, value accuracy, quick matches
  make_dataset.py    Phase A data: positions labelled by the Pons solver
  train_supervised.py  Phase A training (learn from perfect-play labels)
  train_selfplay.py  Phase B training (AlphaZero loop: self-play -> train -> evaluate)
  evaluate.py        full evaluation of a trained model + the "NN eval inside minimax" hypothesis test
  export.py          checkpoint (.pt) -> model.npz

Training needs PyTorch (see docs/NEURAL_NET.md); everything that only *uses*
a trained model (website, arena, evaluate.py with an .npz) needs only numpy.
"""
