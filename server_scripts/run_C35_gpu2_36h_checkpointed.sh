#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2

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

run_task C35_36h_c3_size_repaironly_gpu2_seed0_9 \
  python -u diagnose_repair_loss_checkpointed.py \
    --tasks sssp bottleneck \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --sizes 256 512 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 2 \
    --train_delta_size 4 --test_delta_size 12 \
    --repair_weight 1.0 --critical_weight 0.0 \
    --copy_safe_weight 0.0 --copy_safe_pos_weight 20.0 \
    --copy_scale 5.0 \
    --out results/C35_36h_c3_size_repaironly_gpu2_seed0_9.json \
    --markdown RESULTS_C35_36H_C3_SIZE_REPAIRONLY_gpu2_seed0_9.md

run_task C35_36h_c4_wide_repaironly_gpu2_seed0_9 \
  python -u diagnose_repair_loss_checkpointed.py \
    --tasks sssp bottleneck \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --sizes 64 128 256 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --train_delta_size 4 --test_delta_size 12 \
    --test_weight_hi 100 \
    --repair_weight 1.0 --critical_weight 0.0 \
    --copy_safe_weight 0.0 --copy_safe_pos_weight 20.0 \
    --copy_scale 5.0 \
    --out results/C35_36h_c4_wide_repaironly_gpu2_seed0_9.json \
    --markdown RESULTS_C35_36H_C4_WIDE_REPAIRONLY_gpu2_seed0_9.md

echo "===== GPU2 C35 36h repair-only ablation chain complete $(date) ====="
