#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/NAR}"
mkdir -p logs results

echo "===== GPU3 COMBO START C34 -> C35 -> C38 $(date) ====="

echo "===== GPU3 COMBO START C34 $(date) ====="
bash server_scripts/run_C34_gpu3_36h_checkpointed.sh > logs/C34_GPU3_36H_MASTER.log 2>&1
echo "===== GPU3 COMBO DONE C34 $(date) ====="

echo "===== GPU3 COMBO START C35 $(date) ====="
bash server_scripts/run_C35_gpu3_36h_checkpointed.sh > logs/C35_GPU3_36H_MASTER.log 2>&1
echo "===== GPU3 COMBO DONE C35 $(date) ====="

echo "===== GPU3 COMBO START C38 $(date) ====="
bash server_scripts/run_C38_gpu3_efficiency_checkpointed.sh > logs/C38_GPU3_EFFICIENCY_MASTER.log 2>&1
echo "===== GPU3 COMBO DONE C38 $(date) ====="

echo "===== GPU3 COMBO COMPLETE $(date) ====="
