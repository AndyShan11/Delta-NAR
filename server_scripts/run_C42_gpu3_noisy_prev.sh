#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

echo "===== C42 GPU3 noisy-previous robustness start $(date) ====="
python -u robustness_c42_noisy_prev_checkpointed.py \
  --tasks sssp bottleneck \
  --variants copy_safe repair_only \
  --seeds 10 11 12 13 14 15 16 17 18 19 \
  --sizes 64 128 \
  --noise_rates 0.0 0.01 0.05 0.10 \
  --noise_mode both \
  --train_episodes 500 \
  --test_episodes 100 \
  --epochs 50 \
  --batch 64 \
  --eval_batch 16 \
  --hidden 128 \
  --train_delta_size 4 \
  --test_delta_size 12 \
  --repair_weight 1.0 \
  --copy_safe_weight 1.0 \
  --copy_safe_pos_weight 20.0 \
  --out results/C42_noisy_prev_gpu3_seed10_19.json \
  --markdown RESULTS_C42_NOISY_PREV_gpu3_seed10_19.md \
  > logs/C42_noisy_prev_gpu3_seed10_19.log 2>&1

echo "===== C42 GPU3 noisy-previous robustness done $(date) ====="
