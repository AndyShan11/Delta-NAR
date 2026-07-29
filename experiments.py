"""Formal multi-seed experiment grid for CLRS-Δ (run on the 2x2080Ti server).

Trains DeltaNAR / Naive-d / Recompute on SSSP (min-plus) and bottleneck (max-min)
across seeds and test sizes; reports graph/node accuracy as mean±std and saves
results.json. To split across the two GPUs, run two processes with disjoint
--seeds and CUDA_VISIBLE_DEVICES=0 / =1, then merge the JSONs.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import make_dataset  # noqa: E402
from src.dataset_bottleneck import make_dataset_bottleneck  # noqa: E402
from src.model import DeltaNAR, collate, compute_loss, evaluate  # noqa: E402

TASK_CFG = {  # task -> (dataset_fn, eval_semiring)
    "sssp": (make_dataset, "min_plus"),
    "bottleneck": (make_dataset_bottleneck, "max_min"),
}
CONFIGS = {
    "DeltaNAR":   dict(use_prev=True,  use_delta=True,  use_gate=True),
    "Naive-d":    dict(use_prev=True,  use_delta=True,  use_gate=False, use_copy=False),  # true -copy
    "Recompute":  dict(use_prev=False, use_delta=False, use_gate=False),
    "Recomp+gate": dict(use_prev=False, use_delta=False, use_gate=True),               # gated from-scratch
    "Triplet":    dict(use_prev=False, use_delta=False, use_gate=True, use_triplet=True),  # strong SOTA-style from-scratch
    "no-copy":    dict(use_prev=True,  use_delta=True,  use_gate=True, use_copy=False),    # ablation: -copy
    "min-only":   dict(use_prev=True,  use_delta=True,  use_gate=True, use_minmax=False), # ablation: -max agg
    "Attn":       dict(use_prev=False, use_delta=False, use_gate=True, use_attn=True),    # attention baseline
}


def chunks(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i:i + k]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def eval_at(model, mk, sem, n, n_ep, device, batch=32, topology="er", heavy_tail=False, whi=10, kinds=None):
    model.eval()
    samples = mk(n_ep, n, n_steps=5, seed0=900_000,  # fixed test set across seeds (fair)
                 topology=topology, heavy_tail=heavy_tail, weight_range=(1, whi), kinds=kinds)
    tot, cnt = None, 0
    for ch in chunks(samples, batch):
        b = collate(ch, device=device)
        m = evaluate(b, *model(b), semiring=sem)
        if tot is None:
            tot = {k: 0.0 for k in m}
        for k in tot:
            tot[k] += m[k] * len(ch)
        cnt += len(ch)
    return {k: tot[k] / cnt for k in tot}


def train_eval(task, cfg, seed, args, device):
    mk, sem = TASK_CFG[task]
    set_seed(seed)
    cm = 2.0 if task == "bottleneck" else 0.0  # conservative copy for harder semirings
    model = DeltaNAR(hidden=args.hidden, task="sssp", steps=(args.steps or None), copy_margin=cm, **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train = mk(args.train_episodes, args.train_n, n_steps=5, seed0=1 + seed * 7919)  # seed-varied train
    for _ in range(args.epochs):
        model.train()
        random.shuffle(train)
        for ch in chunks(train, args.batch):
            b = collate(ch, device=device)
            loss, _ = compute_loss(b, *model(b))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {n: eval_at(model, mk, sem, n, args.test_episodes, device, batch=args.eval_batch,
                       topology=args.test_topology, heavy_tail=args.test_heavy_tail,
                       whi=args.test_weight_hi, kinds=args.test_delta_kinds)
            for n in args.sizes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sssp", "bottleneck"])
    ap.add_argument("--models", nargs="+", default=list(CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--train_n", type=int, default=16)
    ap.add_argument("--train_episodes", type=int, default=500)
    ap.add_argument("--test_episodes", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=16)  # smaller for large n
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--test_topology", default="er")        # er | ba | grid  (train is always er)
    ap.add_argument("--test_heavy_tail", action="store_true")  # heavy-tailed test weights
    ap.add_argument("--test_weight_hi", type=int, default=10)   # test weight upper bound (train=10)
    ap.add_argument("--test_delta_kinds", nargs="+", default=None)  # e.g. edge_delete for failure-OOD
    ap.add_argument("--steps", type=int, default=0)  # 0 => MP steps = n; else fixed (e.g. 32) for huge n
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} tasks={args.tasks} models={args.models} "
          f"seeds={args.seeds} sizes={args.sizes}", flush=True)

    raw = {}  # raw[task][model][n][metric] = [over seeds]
    for task in args.tasks:
        raw[task] = {}
        for mname in args.models:
            raw[task][mname] = {n: {} for n in args.sizes}
            for seed in args.seeds:
                t = time.time()
                res = train_eval(task, CONFIGS[mname], seed, args, device)
                for n in args.sizes:
                    for k, v in res[n].items():
                        raw[task][mname][n].setdefault(k, []).append(v)
                print(f"[{task}/{mname}/seed{seed}] {time.time() - t:.0f}s", flush=True)

    agg = {}
    for task in raw:
        agg[task] = {}
        for mname in raw[task]:
            agg[task][mname] = {}
            for n in raw[task][mname]:
                agg[task][mname][n] = {
                    k: [float(np.mean(vs)), float(np.std(vs))]
                    for k, vs in raw[task][mname][n].items()
                }

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "agg": agg, "raw": raw}, f, indent=2)

    for task in agg:
        print(f"\n=== {task}: graph_acc_v  (mean±std over {len(args.seeds)} seeds) ===", flush=True)
        print("Model".ljust(11) + "".join(f"n={n}".ljust(16) for n in args.sizes))
        for mname in agg[task]:
            row = mname.ljust(11)
            for n in args.sizes:
                m, s = agg[task][mname][n]["graph_acc_v"]
                row += f"{m:.3f}±{s:.3f}".ljust(16)
            print(row)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
