#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

echo "===== C45 GPU3 multi-day stress-test push start $(date) ====="

echo "===== Stage 1: resume/complete C44-18H GPU3 $(date) ====="
bash server_scripts/run_C44_18h_gpu3_focused_stress.sh

echo "===== Stage 2: guarded certificate + fixed noisy deep audit GPU3 $(date) ====="
python -u train_c44_focused_stress_checkpointed.py \
  --tasks sssp bottleneck \
  --train_modes copy_safe fixed_noisy \
  --seeds 10 11 12 13 14 15 16 17 18 19 \
  --sizes 64 128 256 \
  --noise_rates 0.0 0.05 0.10 0.20 0.30 \
  --eval_suites er wide100 heavy_tail ba \
  --train_noise_rate 0.05 \
  --train_noise_mode both \
  --rollout_sizes 64 128 256 \
  --rollout_steps 120 \
  --rollout_episodes 80 \
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
  --cert_guard copy_unsafe \
  --cert_tol 1.0 \
  --cert_bonus 20.0 \
  --out results/C45_guarded_deep_gpu3_seed10_19.json \
  --markdown RESULTS_C45_GUARDED_DEEP_gpu3_seed10_19.md \
  > logs/C45_guarded_deep_gpu3_seed10_19.log 2>&1

echo "===== Stage 3: C46 large-size guarded stress GPU3 $(date) ====="
python -u train_c44_focused_stress_checkpointed.py \
  --tasks sssp bottleneck \
  --train_modes copy_safe fixed_noisy \
  --seeds 10 11 12 13 14 15 16 17 18 19 \
  --sizes 512 \
  --noise_rates 0.0 0.10 0.20 \
  --eval_suites er wide100 \
  --train_noise_rate 0.05 \
  --train_noise_mode both \
  --rollout_sizes 256 \
  --rollout_steps 120 \
  --rollout_episodes 40 \
  --train_episodes 500 \
  --test_episodes 40 \
  --epochs 50 \
  --batch 64 \
  --eval_batch 8 \
  --hidden 128 \
  --train_delta_size 4 \
  --test_delta_size 12 \
  --repair_weight 1.0 \
  --copy_safe_weight 1.0 \
  --copy_safe_pos_weight 20.0 \
  --cert_guard copy_unsafe \
  --cert_tol 1.0 \
  --cert_bonus 20.0 \
  --out results/C46_large_guarded_gpu3_seed10_19.json \
  --markdown RESULTS_C46_LARGE_GUARDED_gpu3_seed10_19.md \
  > logs/C46_large_guarded_gpu3_seed10_19.log 2>&1

echo "===== C45 GPU3 multi-day stress-test push done $(date) ====="
