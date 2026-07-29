"""Train + evaluate DeltaNAR on the incremental-MST task (generality check).

If the copy-not-recompute mechanism is task-general, DeltaNAR should again hold
size-OOD on MST while Recompute collapses — using the SAME architecture, only a
different decoder. Run with the torch interpreter:
  & "D:\\dev\\python\\python3.12.4\\python.exe" E:\\NAR\\src\\train_mst.py
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_mst import make_dataset_mst  # noqa: E402
from src.model import DeltaNAR, collate_mst, compute_loss_mst, evaluate_mst  # noqa: E402

CONFIGS = {
    "DeltaNAR":  dict(use_prev=True,  use_delta=True,  use_gate=True),
    "Naive-d":   dict(use_prev=True,  use_delta=True,  use_gate=False),
    "Recompute": dict(use_prev=False, use_delta=False, use_gate=False),
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
def eval_on(model, n, n_episodes, device, batch=32, delta_size=1):
    model.eval()
    samples = make_dataset_mst(n_episodes, n, n_steps=4, seed0=900_000, delta_size=delta_size)
    tot, cnt = None, 0
    for ch in chunks(samples, batch):
        b = collate_mst(ch, device=device)
        m = evaluate_mst(b, *model(b))
        if tot is None:
            tot = {k: 0.0 for k in m}
        for k in tot:
            tot[k] += m[k] * len(ch)
        cnt += len(ch)
    return {k: tot[k] / cnt for k in tot}


def train_one(cfg, args, device):
    set_seed(args.seed)
    model = DeltaNAR(hidden=args.hidden, task="mst", **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train = make_dataset_mst(args.train_episodes, args.train_n, n_steps=4, seed0=0,
                             delta_size=args.train_delta)
    for _ in range(args.epochs):
        model.train()
        random.shuffle(train)
        for ch in chunks(train, args.batch):
            b = collate_mst(ch, device=device)
            loss, _ = compute_loss_mst(b, *model(b))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, {n: eval_on(model, n, args.test_episodes, device, delta_size=args.test_delta)
                   for n in args.test_ns}


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
    ap.add_argument("--models", nargs="+", default=list(CONFIGS))
    ap.add_argument("--train_delta", type=int, default=1)
    ap.add_argument("--test_delta", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MST] device={device}  train_n={args.train_n}  test_ns={args.test_ns}  epochs={args.epochs}")

    results = {}
    selected = {k: CONFIGS[k] for k in args.models}
    for name, cfg in selected.items():
        t = time.time()
        _, results[name] = train_one(cfg, args, device)
        print(f"[{name:10s}] trained in {time.time() - t:5.1f}s")

    print("\n=== MST: graph_acc / edge_acc / affected_F1 ===")
    header = "Model".ljust(11) + "".join(f"  n={n:<4d} gA/eA/aF1".ljust(28) for n in args.test_ns)
    print(header)
    print("-" * len(header))
    for name in selected:
        row = name.ljust(11)
        for n in args.test_ns:
            r = results[name][n]
            row += f"  {r['graph_acc']:.3f}/{r['edge_acc']:.3f}/{r['aff_f1']:.3f}".ljust(28)
        print(row)


if __name__ == "__main__":
    main()
