"""CUDA2/3 re-run of experiments #12-#14 (the blocks #12 stalled on). Launch:

  cd "${PROJECT_DIR:-$HOME/NAR}" && tar xzf NAR_v4.tar.gz && mkdir -p logs results
  CUDA_VISIBLE_DEVICES=2 nohup python -u run_batch3b.py --part gpu2 > logs/b3b_gpu2.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 nohup python -u run_batch3b.py --part gpu3 > logs/b3b_gpu3.log 2>&1 &
  tail -f logs/b3b_gpu2.log

#12 now uses fixed 32 MP steps (A4: DeltaNAR localises => few steps suffice), so
n=1024 no longer stalls. Isolated per-run. ~3h per GPU.
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
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
    print(f"    done in {time.time() - t:.0f}s", flush=True)


def E(a, o, l):
    run(f"{PY} -u experiments.py {a} --out results/{o}", l)


def Tr(s, a, l):
    run(f"{PY} -u src/{s} {a}", l)


def gpu2():
    # #14 main grid seeds->10 (SSSP)
    E("--tasks sssp --models DeltaNAR --seeds 0 1 2 3 4 5 6 7 8 9 --sizes 16 32 64 128 256 --eval_batch 6", "SEED14_sssp.json", "SEED14_sssp.log")
    # #12 extreme size with capped steps (SSSP): n=512, 1024
    E("--tasks sssp --models DeltaNAR Recompute --sizes 512 1024 --seeds 0 1 2 --eval_batch 2 --steps 32", "EXT12_sssp.json", "EXT12_sssp.log")


def gpu3():
    # #14 main grid seeds->10 (bottleneck)
    E("--tasks bottleneck --models DeltaNAR --seeds 0 1 2 3 4 5 6 7 8 9 --sizes 16 32 64 128 256 --eval_batch 6", "SEED14_bot.json", "SEED14_bot.log")
    # #13 long rollout (60 steps), 5 seeds
    for s in range(5):
        Tr("train.py", f"--rollout --rollout_n 64 --rollout_steps 60 --seed {s} --epochs 50 --models DeltaNAR Naive-d", f"ROLL13_s{s}.log")
    # #12 extreme size with capped steps (bottleneck)
    E("--tasks bottleneck --models DeltaNAR Recompute --sizes 512 1024 --seeds 0 1 2 --eval_batch 2 --steps 32", "EXT12_bot.json", "EXT12_bot.log")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["gpu2", "gpu3"])
    a = ap.parse_args()
    t0 = time.time()
    (gpu2 if a.part == "gpu2" else gpu3)()
    print(f"\n=== {a.part} DONE in {(time.time() - t0) / 3600:.2f}h ===", flush=True)
