"""Other ways to learn Connect 4 (CSCI 4150 final project, part 3).

Part 2 (nn/) trained one ResNet two ways: supervised from the perfect solver
and AlphaZero self-play.  This package adds five more learning methods so the
presentation can compare what each one costs and what it buys:

  evolve.py    Evolutionary tuning (CMA-ES) of the minimax heuristic's weights.
               No gradients, no labels: candidates play games, winners breed.
  ntuple.py    TD(lambda) with an n-tuple network: a huge lookup table over
               random 8-cell patterns, learned from self-play on the CPU only.
  dqn.py       Deep Q-learning (double DQN, replay buffer, target network) with
               the same ResNet; the policy head outputs Q-values.
  ppo.py       Proximal Policy Optimization (policy gradient) self-play with an
               opponent pool; actor = policy head, critic = value head.
  distill.py   Knowledge distillation: a small "student" net copies one or more
               trained "teachers" (any mix of the above, optionally + MCTS).

Shared pieces
  vecenv.py    many Connect 4 games stepped at once with numpy bitboards
  meter.py     training cost: wall clock, CPU core-hours, GPU power -> energy (Wh)
  evalhook.py  the same benchmarks as nn/ (accuracy vs perfect play, Elo on the
               minimax ladder) so every method lands on the same charts
  agents.py    agents for the non-network methods (evolved weights, n-tuple)
  compare.py   one standard evaluation of every method -> table + charts

Every trainer writes runs/<run>/{config.json, metrics.jsonl, cost.json} like
nn/ does, so the Lab page (/lab) shows these runs next to the existing ones.
DQN, PPO and distillation export the usual runs/<run>/model.npz, so the
website, MCTS, the NN-minimax hypothesis test and the arena work unchanged.
"""
