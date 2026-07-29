"""#11 cross-task transfer: train on SSSP (min-plus), evaluate on bottleneck
(max-min). Zero-shot (frozen) vs few-shot (short fine-tune). Shows how far the
incremental mechanism transfers across semirings with/without minimal adaptation."""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import make_dataset  # noqa: E402
from src.dataset_bottleneck import make_dataset_bottleneck  # noqa: E402
from src.model import DeltaNAR, collate, compute_loss, evaluate  # noqa: E402


def chunks(l, k):
    for i in range(0, len(l), k):
        yield l[i:i + k]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def ev_bottleneck(model, n, ne, dev):
    model.eval()
    s = make_dataset_bottleneck(ne, n, n_steps=5, seed0=900_000)
    tot, c = None, 0
    for ch in chunks(s, 16):
        b = collate(ch, device=dev)
        m = evaluate(b, *model(b), semiring="max_min")
        if tot is None:
            tot = {k: 0.0 for k in m}
        for k in tot:
            tot[k] += m[k] * len(ch)
        c += len(ch)
    return {k: tot[k] / c for k in tot}


def train(model, data, dev, epochs):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        model.train(); random.shuffle(data)
        for ch in chunks(data, 64):
            b = collate(ch, device=dev)
            loss, _ = compute_loss(b, *model(b))
            opt.zero_grad(); loss.backward(); opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ft_epochs", type=int, default=5)
    ap.add_argument("--out", default="results/TRANS11.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    res = {}
    for seed in a.seeds:
        set_seed(seed)
        model = DeltaNAR(task="sssp").to(dev)  # SSSP decoder, min-plus training
        train(model, make_dataset(500, 16, n_steps=5, seed0=1 + seed * 7919), dev, a.epochs)
        zs = {n: ev_bottleneck(model, n, 60, dev) for n in a.sizes}          # zero-shot on bottleneck
        train(model, make_dataset_bottleneck(100, 16, n_steps=5, seed0=2 + seed * 7919), dev, a.ft_epochs)
        fs = {n: ev_bottleneck(model, n, 60, dev) for n in a.sizes}          # after few-shot fine-tune
        res[seed] = {"zero_shot": zs, "few_shot": fs}
        print(f"[seed{seed}] bottleneck nAv@16 zero-shot={zs[16]['node_acc_v']:.3f} -> "
              f"few-shot={fs[16]['node_acc_v']:.3f}", flush=True)

    json.dump(res, open(a.out, "w"), indent=2)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
