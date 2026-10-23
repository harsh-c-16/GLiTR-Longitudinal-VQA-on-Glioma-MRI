#!/usr/bin/env bash
# One variant (full model, or an ablation via NAME / ARGS, see README) for one cohort: 5 seeds (SEEDS overrides), JOBS at a time
# (default one per GPU), with the tuned hyperparameters in HP_FILE (default configs/hp.json).
#   [NAME=... ARGS=...] bash scripts/run_seeds.sh <cohort> <feats> [out=runs]
set -eu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
COHORT=$1; FEATS=$2; OUT=${3:-runs}; GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}; JOBS=${JOBS:-$GPUS}; NAME=${NAME:-full}; ARGS=${ARGS:-}
HP_FILE=${HP_FILE:-configs/hp.json}
export COHORT FEATS OUT HP_FILE NAME ARGS
i=0
for seed in ${SEEDS:-42 43 44 45 46}; do echo "$seed $((i++ % GPUS))"; done | xargs -P "$JOBS" -L 1 bash -c \
  'CUDA_VISIBLE_DEVICES=$1 python -m glitr.train --features $FEATS --gt configs/gt_$COHORT.csv --folds-file configs/folds_$COHORT.csv \
     --out $OUT/$COHORT/$NAME --seed $0 --hp-file $HP_FILE --cohort $COHORT $ARGS'
