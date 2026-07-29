#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3

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

# Balanced GPU3 lane: bottleneck default diagnostics, SSSP hard diagnostics,
# triplet capped runs, oracle variants, main DeltaNAR refresh, and bottleneck extreme.

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

run_task B07_sssp_k8_core_diag \
  python -u diagnose_a_metrics.py \
    --tasks sssp \
    --models DeltaNAR no-copy Naive-d Recomp+gate Recompute Attn \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --sizes 16 64 128 256 \
    --train_delta_size 8 --test_delta_size 8 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --out results/B07_sssp_k8_core_diag.json \
    --markdown RESULTS_B07_sssp_k8_core_diag.md

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

run_task B09_bottleneck_triplet_diag \
  python -u diagnose_a_metrics.py \
    --tasks bottleneck \
    --models Triplet \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --sizes 16 32 64 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 32 --eval_batch 2 \
    --out results/B09_bottleneck_triplet_diag.json \
    --markdown RESULTS_B09_bottleneck_triplet_diag.md

run_task B10_oracle_sssp_bottleneck \
  python -u diagnose_a_oracle.py \
    --tasks sssp bottleneck \
    --seeds 0 1 2 3 4 \
    --sizes 16 64 128 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --out results/B10_oracle_sssp_bottleneck.json \
    --markdown RESULTS_B10_oracle_sssp_bottleneck.md

run_task B11_main_deltanar_20seed \
  python -u experiments.py \
    --tasks sssp bottleneck \
    --models DeltaNAR \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --sizes 16 32 64 128 256 \
    --train_episodes 500 --test_episodes 100 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 4 \
    --out results/B11_main_deltanar_20seed.json

run_task B13_bottleneck_extreme_512_1024 \
  python -u experiments.py \
    --tasks bottleneck \
    --models DeltaNAR Recompute \
    --seeds 0 1 2 3 4 \
    --sizes 512 1024 \
    --train_episodes 500 --test_episodes 40 --epochs 50 \
    --hidden 128 --batch 64 --eval_batch 1 \
    --steps 32 \
    --out results/B13_bottleneck_extreme_512_1024.json

echo "===== GPU3 balanced B-line chain complete $(date) ====="
