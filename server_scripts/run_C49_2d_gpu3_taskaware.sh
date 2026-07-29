#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

echo "===== C49 2-day task-aware certificate routing GPU3 start $(date) ====="

python -u train_c44_focused_stress_checkpointed.py \
  --tasks sssp bottleneck \
  --train_modes copy_safe \
  --seeds 10 11 12 13 14 15 \
  --sizes 256 \
  --noise_rates 0.0 0.10 0.30 \
  --eval_suites er wide100 heavy_tail ba \
  --train_noise_rate 0.05 \
  --train_noise_mode both \
  --rollout_sizes 256 \
  --rollout_steps 120 \
  --rollout_episodes 30 \
  --train_episodes 500 \
  --test_episodes 50 \
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
  --cert_bonus 20.0 \
  --out results/C49_taskaware_gpu3_seed10_15.json \
  --markdown RESULTS_C49_TASKAWARE_gpu3_seed10_15.md \
  > logs/C49_taskaware_gpu3_seed10_15.log 2>&1

echo "===== C49 2-day task-aware certificate routing GPU3 done $(date) ====="
