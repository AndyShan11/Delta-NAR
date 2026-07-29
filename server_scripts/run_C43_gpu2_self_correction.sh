#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

echo "===== C43 GPU2 self-correction start $(date) ====="
python -u train_c43_self_correction_checkpointed.py \
  --tasks sssp bottleneck \
  --variants copy_safe self_correct \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --sizes 64 128 \
  --noise_rates 0.0 0.01 0.05 0.10 \
  --train_noise_rate 0.05 \
  --train_noise_mode both \
  --rollout_sizes 64 \
  --rollout_steps 60 \
  --rollout_episodes 40 \
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
  --out results/C43_self_correction_gpu2_seed0_9.json \
  --markdown RESULTS_C43_SELF_CORRECTION_gpu2_seed0_9.md \
  > logs/C43_self_correction_gpu2_seed0_9.log 2>&1
echo "===== C43 GPU2 self-correction done $(date) ====="
