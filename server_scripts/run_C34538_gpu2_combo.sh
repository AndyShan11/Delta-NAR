#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

echo "===== GPU2 COMBO START C34 -> C35 -> C38 $(date) ====="

echo "===== GPU2 COMBO START C34 $(date) ====="
bash server_scripts/run_C34_gpu2_36h_checkpointed.sh > logs/C34_GPU2_36H_MASTER.log 2>&1
echo "===== GPU2 COMBO DONE C34 $(date) ====="

echo "===== GPU2 COMBO START C35 $(date) ====="
bash server_scripts/run_C35_gpu2_36h_checkpointed.sh > logs/C35_GPU2_36H_MASTER.log 2>&1
echo "===== GPU2 COMBO DONE C35 $(date) ====="

echo "===== GPU2 COMBO START C38 $(date) ====="
bash server_scripts/run_C38_gpu2_efficiency_checkpointed.sh > logs/C38_GPU2_EFFICIENCY_MASTER.log 2>&1
echo "===== GPU2 COMBO DONE C38 $(date) ====="

echo "===== GPU2 COMBO COMPLETE $(date) ====="
