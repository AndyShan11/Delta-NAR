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

run_task () {
  local name="$1"
  shift
  echo "===== START ${name} $(date) ====="
  nice -n 5 "$@" > "logs/${name}.log" 2>&1
  echo "===== DONE ${name} $(date) ====="
}

run_task C34_c3_size_gpu3_seed10_19 \
  python -u diagnose_repair_loss.py \
    --tasks sssp bottleneck \
    --seeds 10 11 12 13 14 15 16 17 18 19 \
    --sizes 256 512 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 2 \
    --train_delta_size 4 --test_delta_size 12 \
    --repair_weight 1.0 --critical_weight 0.0 \
    --use_copy_safe_gate --copy_safe_weight 1.0 --copy_safe_pos_weight 20.0 \
    --copy_scale 5.0 \
    --out results/C34_c3_size_gpu3_seed10_19.json \
    --markdown RESULTS_C34_C3_SIZE_gpu3_seed10_19.md

run_task C34_c4_wide_gpu3_seed10_19 \
  python -u diagnose_repair_loss.py \
    --tasks sssp bottleneck \
    --seeds 10 11 12 13 14 15 16 17 18 19 \
    --sizes 64 128 256 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --train_delta_size 4 --test_delta_size 12 \
    --test_weight_hi 100 \
    --repair_weight 1.0 --critical_weight 0.0 \
    --use_copy_safe_gate --copy_safe_weight 1.0 --copy_safe_pos_weight 20.0 \
    --copy_scale 5.0 \
    --out results/C34_c4_wide_gpu3_seed10_19.json \
    --markdown RESULTS_C34_C4_WIDE_gpu3_seed10_19.md

echo "===== GPU3 C34 11h copy-safe chain complete $(date) ====="
