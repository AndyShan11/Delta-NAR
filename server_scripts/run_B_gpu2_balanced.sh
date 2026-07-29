#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2

# Keep CPU pressure modest because other jobs may be using the host.
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

# Balanced GPU2 lane: SSSP default diagnostics, bottleneck hard diagnostics,
# OOD checks, and SSSP extreme.

run_task B01_sssp_k1_core_diag \
  python -u diagnose_a_metrics.py \
    --tasks sssp \
    --models DeltaNAR no-copy Naive-d Recomp+gate Recompute Attn \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --sizes 16 64 128 256 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --out results/B01_sssp_k1_core_diag.json \
    --markdown RESULTS_B01_sssp_k1_core_diag.md

run_task B08_bottleneck_k4_core_diag \
  python -u diagnose_a_metrics.py \
    --tasks bottleneck \
    --models DeltaNAR no-copy Naive-d Recomp+gate Recompute Attn \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --sizes 16 64 128 256 \
    --train_delta_size 4 --test_delta_size 4 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --out results/B08_bottleneck_k4_core_diag.json \
    --markdown RESULTS_B08_bottleneck_k4_core_diag.md

run_task B04_ood_topology_ba_sssp \
  python -u experiments.py \
    --tasks sssp \
    --models DeltaNAR no-copy Naive-d Recomp+gate Recompute Attn \
    --seeds 0 1 2 3 4 \
    --sizes 64 128 \
    --test_topology ba \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --out results/B04_ood_topology_ba_sssp.json

run_task B05_ood_weight_wide_sssp \
  python -u experiments.py \
    --tasks sssp \
    --models DeltaNAR no-copy Naive-d Recomp+gate Recompute Attn \
    --seeds 0 1 2 3 4 \
    --sizes 64 128 \
    --test_weight_hi 100 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --out results/B05_ood_weight_wide_sssp.json

run_task B12_ood_failure_sssp_bottleneck \
  python -u experiments.py \
    --tasks sssp bottleneck \
    --models DeltaNAR no-copy Naive-d Recompute \
    --seeds 0 1 2 3 4 \
    --sizes 16 64 128 \
    --test_delta_kinds edge_delete \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --out results/B12_ood_failure_sssp_bottleneck.json

run_task B06_sssp_extreme_512 \
  python -u experiments.py \
    --tasks sssp \
    --models DeltaNAR Recompute \
    --seeds 0 1 2 3 4 \
    --sizes 512 \
    --train_episodes 500 --test_episodes 60 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 1 \
    --steps 0 \
    --out results/B06_sssp_extreme_512.json

echo "===== GPU2 balanced B-line chain complete $(date) ====="
