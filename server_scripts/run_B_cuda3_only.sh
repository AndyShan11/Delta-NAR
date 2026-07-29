#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

# CUDA3-only mode. This runs the GPU2 split on CUDA3 first, then the GPU3 split
# on CUDA3. No command in this wrapper uses CUDA2.

tmp_script="/tmp/run_B_gpu2_as_cuda3_$$.sh"
sed 's/export CUDA_VISIBLE_DEVICES=2/export CUDA_VISIBLE_DEVICES=3/' \
  server_scripts/run_B_gpu2.sh > "${tmp_script}"
chmod +x "${tmp_script}"

echo "===== CUDA3-only phase 1: GPU2 split remapped to CUDA3 $(date) ====="
bash "${tmp_script}"
rm -f "${tmp_script}"

echo "===== CUDA3-only phase 2: GPU3 split on CUDA3 $(date) ====="
bash server_scripts/run_B_gpu3.sh

echo "===== CUDA3-only B-line chain complete $(date) ====="
