"""CUDA3-only batch for experiments #7-#14. Launch:

  cd "${PROJECT_DIR:-$HOME/NAR}" && tar xzf NAR_v3.tar.gz && mkdir -p logs results
  CUDA_VISIBLE_DEVICES=3 nohup python -u run_batch3.py > logs/batch3.log 2>&1 &
  tail -f logs/batch3.log

Single GPU, isolated per-run (one crash won't stop the rest). Ordered by
value: ablation + generalist first, then baselines/OOD/extreme. Long (~15h);
results stream to results/*.json as each block finishes.
"""
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


# #7 component ablation: -copy / -max(min-only) / -gate(Naive-d) / -prev(Recompute)
E("--tasks sssp bottleneck --models DeltaNAR no-copy min-only Naive-d Recompute --sizes 16 32 64 128", "ABL7.json", "ABL7.log")
# #8 generalist 5-seed confirm
for s in range(5):
    Tr("train_generalist.py", f"--seed {s} --epochs 50 --test_ns 16 32 64 128", f"GEN8_s{s}.log")
# #9 attention baseline
E("--tasks sssp bottleneck --models Attn --sizes 16 32 64 128", "BASE9_attn.json", "BASE9_attn.log")
# #10 failure/cascade OOD: train mixed edits -> test edge deletions (failures)
E("--tasks sssp bottleneck --models DeltaNAR Recompute --sizes 16 64 --test_delta_kinds edge_delete", "OOD10_failure.json", "OOD10_failure.log")
# #11 cross-task transfer (SSSP -> bottleneck, zero/few-shot)
Tr("transfer.py", "--seeds 0 1 2", "TRANS11.log")
# #12 extreme size n=1024 (32x->64x)
E("--tasks sssp --models DeltaNAR Recompute --sizes 1024 --seeds 0 1 2 --eval_batch 2", "EXT12_1024.json", "EXT12_1024.log")
# #13 long rollout (60 steps) multi-seed
for s in range(5):
    Tr("train.py", f"--rollout --rollout_n 64 --rollout_steps 60 --seed {s} --epochs 50 --models DeltaNAR Naive-d", f"ROLL13_s{s}.log")
# #14 main grid seeds 5->10 (tighter +-std)
E("--tasks sssp bottleneck --models DeltaNAR --seeds 0 1 2 3 4 5 6 7 8 9 --sizes 16 32 64 128 256 --eval_batch 6", "SEED14.json", "SEED14.log")

print("\n=== BATCH3 ALL DONE ===", flush=True)
