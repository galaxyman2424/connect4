#!/usr/bin/env bash
# Full score-head pipeline (docs/ACCURACY_95.md).  Every stage is skipped when its output
# already exists, so after a crash or Ctrl+C just run it again.
#
#   bash tools/train_95.sh            # everything (about 15-18 h on the 5800X + RTX 3050)
#   SIZE=6x64 bash tools/train_95.sh  # smaller/faster nets
#   SEEDS="0" bash tools/train_95.sh  # one seed only
#
# Stages
#   1  3M solver-labelled positions               nn_data/solver_3m.npz        ~3.5 h CPU
#   2  score-head nets, one per seed              runs/score_<size>_s<seed>    ~2.5 h each GPU (8x128)
#   3  mine positions each net gets wrong         nn_data/hard_<size>_s<seed>.npz  ~1 h each CPU
#   4  fine-tune each net with its hard positions runs/score_<size>_s<seed>_r1 ~1 h each GPU
#   5  benchmark every net + the ensemble         experiments/results/score/*.json
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

SIZE=${SIZE:-8x128}
SEEDS=${SEEDS:-"0 1 2"}
POSITIONS=${POSITIONS:-3000000}
EPOCHS=${EPOCHS:-30}
WORKERS=${WORKERS:-14}
MINE=${MINE:-600000}
B=${SIZE%x*}; C=${SIZE#*x}
DATA=nn_data/solver_3m.npz
mkdir -p experiments/results/score logs

if [ ! -f "$DATA" ]; then
  echo "== 1. labelling $POSITIONS positions"
  python -m nn.make_dataset --positions "$POSITIONS" --workers "$WORKERS" --seed 2027 --out "$DATA" 2>&1 | tee logs/make_3m.log
fi

R1=""
for S in $SEEDS; do
  RUN=score_${SIZE}_s${S}
  if [ ! -f runs/$RUN/best.npz ] || [ ! -f runs/$RUN/.done ]; then
    echo "== 2. training $RUN"
    rm -rf runs/$RUN
    python -m nn.train_score --data "$DATA" --run "$RUN" --blocks "$B" --channels "$C" \
        --epochs "$EPOCHS" --seed "$S" 2>&1 | tee logs/$RUN.log
    touch runs/$RUN/.done
  fi
  HARD=nn_data/hard_${SIZE}_s${S}.npz
  if [ ! -f "$HARD" ]; then
    echo "== 3. mining hard positions for $RUN"
    python -m nn.mine_hard --run "$RUN" --positions "$MINE" --workers "$WORKERS" --seed $((100 + S)) \
        --out "$HARD" 2>&1 | tee logs/mine_$RUN.log
  fi
  RUN1=${RUN}_r1
  if [ ! -f runs/$RUN1/best.npz ] || [ ! -f runs/$RUN1/.done ]; then
    echo "== 4. fine-tuning $RUN1 with $HARD"
    rm -rf runs/$RUN1
    python -m nn.train_score --data "$DATA" --hard "$HARD" --hard-weight 2 --init runs/$RUN/checkpoints/best.pt \
        --run "$RUN1" --epochs 10 --lr 6e-4 --seed "$S" 2>&1 | tee logs/$RUN1.log
    touch runs/$RUN1/.done
  fi
  R1="$R1,$RUN1"
done
R1=${R1#,}

echo "== 5. benchmark"
for S in $SEEDS; do
  python -m nn.eval_score --runs score_${SIZE}_s${S},score_${SIZE}_s${S}_r1 --lookahead \
      --out experiments/results/score/${SIZE}_s${S}.json
done
python -m nn.eval_score --runs "$R1" --lookahead --out experiments/results/score/${SIZE}_ensemble.json
echo "done: see experiments/results/score/ and runs/score_${SIZE}_s*/metrics.jsonl"
