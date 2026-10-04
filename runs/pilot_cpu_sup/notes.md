# pilot_cpu_sup: pipeline check, NOT a real result
Made by Claude on a 2-core cloud VM without a GPU (3 Oct 2026) to check that supervised training works end to end.
Data: 20,000 solver-labelled positions (python -m nn.make_dataset --positions 20000). Net 4x32, 6 epochs, 30 s.
Evaluation: python -m nn.evaluate --quick (small settings).
Delete this folder once you have your own runs (rm -r runs/pilot_cpu_*).
