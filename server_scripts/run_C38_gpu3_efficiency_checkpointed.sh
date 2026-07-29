#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "===== START C38_efficiency_gpu3_seed10_14 $(date) ====="
nice -n 5 python -u profile_c38_efficiency_checkpointed.py \
  --tasks sssp bottleneck \
  --variants copy_safe repair_only \
  --seeds 10 11 12 13 14 \
  --sizes 64 128 256 512 \
  --train_episodes 200 --test_episodes 50 --epochs 20 \
  --hidden 128 --batch 64 --eval_batch 2 \
  --train_delta_size 4 --test_delta_size 12 \
  --test_weight_hi 100 \
  --repair_weight 1.0 --critical_weight 0.0 \
  --copy_safe_weight 1.0 --copy_safe_pos_weight 20.0 \
  --copy_scale 5.0 \
  --out results/C38_efficiency_gpu3_seed10_14.json \
  --markdown C38_EFFICIENCY_gpu3_seed10_14.md \
  > logs/C38_efficiency_gpu3_seed10_14.log 2>&1
echo "===== DONE C38_efficiency_gpu3_seed10_14 $(date) ====="
