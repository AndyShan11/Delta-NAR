"""Two-GPU ~12h experiment batch for CLRS-Δ. Launch per GPU:

  cd "${PROJECT_DIR:-$HOME/NAR}" && mkdir -p logs results
  CUDA_VISIBLE_DEVICES=2 nohup python -u run_batch.py --part gpu2 > logs/gpu2.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 nohup python -u run_batch.py --part gpu3 > logs/gpu3.log 2>&1 &
  tail -f logs/gpu2.log

Each sub-run is isolated (one crash does not stop the rest). Results -> results/*.json,
per-run stdout -> logs/*.log. gpu2 = SSSP-family + intensity/rollout; gpu3 = bottleneck-family
+ generalist/MST. Covers: strong baselines (Triplet/Recomp+gate), topology-OOD (BA/grid),
weight-OOD (heavy-tail / wide range), extreme size (512), all multi-seed.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)


def run(cmd, log):
    print(f"\n>>> [{time.strftime('%H:%M:%S')}] {log}\n    {cmd}", flush=True)
    t = time.time()
    try:
        with open(os.path.join(HERE, "logs", log), "w") as f:
            subprocess.run(cmd, shell=True, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    except Exception as e:  # never let one run kill the batch
        print(f"    ERROR: {e}", flush=True)
    print(f"    done in {time.time() - t:.0f}s", flush=True)


def E(arg, out, log):
    run(f"{PY} -u experiments.py {arg} --out results/{out}", log)


def Tr(script, arg, log):
    run(f"{PY} -u src/{script} {arg}", log)


def gpu2():
    # A. SSSP strong-baseline size grid (4 models to n=256; Triplet O(n^3) only to 64)
    E("--tasks sssp --models DeltaNAR Naive-d Recompute Recomp+gate --sizes 16 32 64 128 256 --eval_batch 6", "A_sssp.json", "A_sssp.log")
    E("--tasks sssp --models Triplet --sizes 16 32 64 --eval_batch 6", "A_sssp_triplet.json", "A_sssp_triplet.log")
    # B. topology OOD (train ER -> test BA / grid)
    E("--tasks sssp --models DeltaNAR Triplet Recompute --sizes 64 --test_topology ba", "B_sssp_ba.json", "B_sssp_ba.log")
    E("--tasks sssp --models DeltaNAR Triplet Recompute --sizes 64 --test_topology grid", "B_sssp_grid.json", "B_sssp_grid.log")
    # C. weight OOD (wide range / heavy tail)
    E("--tasks sssp --models DeltaNAR Recompute --sizes 64 --test_weight_hi 100", "C_sssp_whi100.json", "C_sssp_whi100.log")
    E("--tasks sssp --models DeltaNAR Recompute --sizes 64 --test_heavy_tail", "C_sssp_heavy.json", "C_sssp_heavy.log")
    # D. dynamic-intensity + rollout, multi-seed
    for s in range(5):
        Tr("train.py", f"--intensity --intensity_n 16 --seed {s} --epochs 50 --models DeltaNAR Naive-d Recompute", f"D_intensity_s{s}.log")
    for s in range(5):
        Tr("train.py", f"--rollout --rollout_n 64 --seed {s} --epochs 50 --models DeltaNAR Naive-d", f"D_rollout_s{s}.log")
    # E. extreme size (isolated last; may be slow)
    E("--tasks sssp --models DeltaNAR Recompute --sizes 512 --seeds 0 1 2 --eval_batch 2", "E_sssp_512.json", "E_sssp_512.log")


def gpu3():
    # A. bottleneck strong-baseline size grid
    E("--tasks bottleneck --models DeltaNAR Naive-d Recompute Recomp+gate --sizes 16 32 64 128 256 --eval_batch 6", "A_bot.json", "A_bot.log")
    E("--tasks bottleneck --models Triplet --sizes 16 32 64 --eval_batch 6", "A_bot_triplet.json", "A_bot_triplet.log")
    # B. topology OOD
    E("--tasks bottleneck --models DeltaNAR Triplet Recompute --sizes 64 --test_topology ba", "B_bot_ba.json", "B_bot_ba.log")
    E("--tasks bottleneck --models DeltaNAR Triplet Recompute --sizes 64 --test_topology grid", "B_bot_grid.json", "B_bot_grid.log")
    # D. generalist + MST, multi-seed
    for s in range(5):
        Tr("train_generalist.py", f"--seed {s} --epochs 50 --test_ns 16 32 64 128", f"D_gen_s{s}.log")
    for s in range(5):
        Tr("train_mst.py", f"--seed {s} --epochs 50 --test_ns 16 32 64", f"D_mst_s{s}.log")
    # E. extreme size
    E("--tasks bottleneck --models DeltaNAR Recompute --sizes 512 --seeds 0 1 2 --eval_batch 2", "E_bot_512.json", "E_bot_512.log")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["gpu2", "gpu3"])
    a = ap.parse_args()
    t0 = time.time()
    (gpu2 if a.part == "gpu2" else gpu3)()
    print(f"\n=== {a.part} ALL DONE in {(time.time() - t0) / 3600:.2f}h ===", flush=True)
