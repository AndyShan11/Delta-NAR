#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=2

run_task () {
  local name="$1"
  shift
  echo "===== START ${name} $(date) ====="
  "$@" > "logs/${name}.log" 2>&1
  echo "===== DONE ${name} $(date) ====="
}

# GPU2 half of B-line server experiments.
# Focus: default single-edit diagnostics, bottleneck default, OOD, and SSSP extreme size.

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

run_task B02_sssp_triplet_diag \
  python -u diagnose_a_metrics.py \
    --tasks sssp \
    --models Triplet \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --sizes 16 32 64 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 32 --eval_batch 2 \
    --out results/B02_sssp_triplet_diag.json \
    --markdown RESULTS_B02_sssp_triplet_diag.md

run_task B03_bottleneck_k1_core_diag \
  python -u diagnose_a_metrics.py \
    --tasks bottleneck \
    --models DeltaNAR no-copy Naive-d Recomp+gate Recompute Attn \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --sizes 16 64 128 256 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --out results/B03_bottleneck_k1_core_diag.json \
    --markdown RESULTS_B03_bottleneck_k1_core_diag.md

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

echo "===== GPU2 B-line chain complete $(date) ====="
