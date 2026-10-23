#!/usr/bin/env bash
# Per-fold hyperparameter search for one cohort: TRIALS Optuna trials per outer fold, scored by 3-fold patient-grouped
# cross-validation inside that fold's training patients (the outer fold is never seen). Folds run in parallel; their results
# are then written into HP_FILE under the cohort.
#   bash scripts/tune_hp.sh <cohort> <feats> <hp_file> [trials=30]
set -eu
COHORT=$1; FEATS=$2; HP_FILE=$3; TRIALS=${4:-30}; OUT=${OUT:-runs_tune}; GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
for k in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$((k % GPUS)) python -m glitr.train --features $FEATS --gt configs/gt_$COHORT.csv \
    --folds-file configs/folds_$COHORT.csv --out $OUT/$COHORT/full --seed 42 --tune-trials $TRIALS --only-fold $k &
done
wait
python - "$HP_FILE" "$COHORT" "$OUT/$COHORT/full/s42" <<'PY'
import json, os, sys
path, cohort, run = sys.argv[1:]
hp = json.load(open(path)) if os.path.exists(path) else {}
hp[cohort] = {f"fold{k}": json.load(open(os.path.join(run, f"hp_fold{k}.json"))) for k in range(1, 6)}
json.dump(hp, open(path, "w"), indent=1)
PY
