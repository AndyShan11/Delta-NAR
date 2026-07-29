#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

echo "===== C48 2-day focused guard-profile GPU3 start $(date) ====="

python -u train_c44_focused_stress_checkpointed.py \
  --tasks sssp bottleneck \
  --train_modes copy_safe \
  --seeds 10 11 12 13 \
  --sizes 128 256 \
  --noise_rates 0.0 0.10 0.30 \
  --eval_suites er wide100 heavy_tail ba \
  --train_noise_rate 0.05 \
  --train_noise_mode both \
  --rollout_sizes 128 256 \
  --rollout_steps 120 \
  --rollout_episodes 30 \
  --train_episodes 500 \
  --test_episodes 40 \
  --epochs 50 \
  --batch 64 \
  --eval_batch 16 \
  --hidden 128 \
  --train_delta_size 4 \
  --test_delta_size 12 \
  --repair_weight 1.0 \
  --copy_safe_weight 1.0 \
  --copy_safe_pos_weight 20.0 \
  --cert_profiles \
    cu_m1:copy_unsafe:-1.0:0.0:1.0 \
    cu_0:copy_unsafe:0.0:0.0:1.0 \
    both_m1:both:-1.0:0.0:1.0 \
    both_0:both:0.0:0.0:1.0 \
  --cert_bonus 20.0 \
  --out results/C48_2d_guard_profiles_gpu3_seed10_13.json \
  --markdown RESULTS_C48_2D_GUARD_PROFILES_gpu3_seed10_13.md \
  > logs/C48_2d_guard_profiles_gpu3_seed10_13.log 2>&1

echo "===== C48 2-day focused guard-profile GPU3 done $(date) ====="
