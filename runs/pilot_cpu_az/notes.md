# pilot_cpu_az: pipeline check, NOT a real result
Made by Claude on a 2-core cloud VM without a GPU (3 Oct 2026) to check that self-play training works end to end.
Command: python -m nn.train_selfplay --run pilot_az --games 200 --generations 40 --sims 48 --blocks 4 --channels 32 --workers 1 --parallel 100 --eval-every 5 --eval-openings 3 --eval-depths 1,2,4 --eval-sims 48
Tiny network (4x32, 87k params), 8,000 games total, 27 minutes.
What it shows: the loop learns (opens in the centre in ~90% of games by the end, Elo ~1200 on the minimax ladder),
but it is ~100x smaller than the real run. Delete this folder once you have your own runs (rm -r runs/pilot_cpu_*).
