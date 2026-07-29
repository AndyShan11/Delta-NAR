"""Generalist DeltaNAR: ONE processor trained jointly on SSSP (min-plus) and
bottleneck (max-min), distinguished only by a task embedding.

This is the strongest generality claim: a single set of weights performs
incremental reasoning for two different algebraic structures, and still holds
size-OOD on both. Compare against the single-task DeltaNAR numbers.
Run with the torch interpreter.
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import make_dataset  # noqa: E402
from src.dataset_bottleneck import make_dataset_bottleneck  # noqa: E402
from src.model import DeltaNAR, collate, compute_loss, evaluate  # noqa: E402

# task -> (dataset_fn, task_id, eval_semiring)
TASKS = {
    "sssp": (make_dataset, 0, "min_plus"),
    "bottleneck": (make_dataset_bottleneck, 1, "max_min"),
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
def eval_on(model, task, n, n_episodes, device, batch=32):
    mk, tid, sem = TASKS[task]
    model.eval()
    samples = mk(n_episodes, n, n_steps=5, seed0=900_000)
    tot, cnt = None, 0
    for ch in chunks(samples, batch):
        b = collate(ch, device=device, task_id=tid)
        m = evaluate(b, *model(b), semiring=sem)
        if tot is None:
            tot = {k: 0.0 for k in m}
        for k in tot:
            tot[k] += m[k] * len(ch)
        cnt += len(ch)
    return {k: tot[k] / cnt for k in tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=16)
    ap.add_argument("--test_ns", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--train_episodes", type=int, default=500)
    ap.add_argument("--test_episodes", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    print(f"[GENERALIST] device={device}  one processor for SSSP+bottleneck  epochs={args.epochs}")

    model = DeltaNAR(hidden=args.hidden, task="sssp", use_task_emb=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_sssp = make_dataset(args.train_episodes, args.train_n, n_steps=5, seed0=0)
    train_bot = make_dataset_bottleneck(args.train_episodes, args.train_n, n_steps=5, seed0=0)

    t0 = time.time()
    for _ in range(args.epochs):
        model.train()
        random.shuffle(train_sssp)
        random.shuffle(train_bot)
        batches = ([(ch, 0) for ch in chunks(train_sssp, args.batch)]
                   + [(ch, 1) for ch in chunks(train_bot, args.batch)])
        random.shuffle(batches)
        for ch, tid in batches:
            b = collate(ch, device=device, task_id=tid)
            loss, _ = compute_loss(b, *model(b))
            opt.zero_grad()
            loss.backward()
            opt.step()
    print(f"trained one generalist in {time.time() - t0:5.1f}s")

    print("\n=== GENERALIST (one model, both tasks): gAv / nAv / aF1 ===")
    header = "Task".ljust(12) + "".join(f"  n={n:<4d} gAv/nAv/aF1".ljust(28) for n in args.test_ns)
    print(header)
    print("-" * len(header))
    for task in TASKS:
        row = task.ljust(12)
        for n in args.test_ns:
            r = eval_on(model, task, n, args.test_episodes, device)
            row += f"  {r['graph_acc_v']:.3f}/{r['node_acc_v']:.3f}/{r['aff_f1']:.3f}".ljust(28)
        print(row)


if __name__ == "__main__":
    main()
