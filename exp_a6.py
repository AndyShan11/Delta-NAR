"""A6: delta-type OOD. Train on weight-change edits -> test on edge add/delete."""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import make_dataset  # noqa: E402
from src.model import DeltaNAR, collate, compute_loss, evaluate  # noqa: E402


def chunks(l, k):
    for i in range(0, len(l), k):
        yield l[i:i + k]


def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


dev = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT = ["weight_increase", "weight_decrease"]
STRUCT = ["edge_add", "edge_delete"]


@torch.no_grad()
def ev(m, kinds, n=64, ne=60):
    m.eval(); tot = 0.0; c = 0
    for ch in chunks(make_dataset(ne, n, n_steps=5, seed0=900_000, kinds=kinds), 32):
        b = collate(ch, device=dev)
        tot += evaluate(b, *m(b))["node_acc_v"] * len(ch); c += len(ch)
    return tot / c


print(f"{'model':>10} {'in-dist(weight)':>16} {'OOD(add/del)':>14}")
for name, cfg in [("DeltaNAR", dict(use_prev=True, use_delta=True, use_gate=True)),
                  ("Recompute", dict(use_prev=False, use_delta=False, use_gate=False))]:
    seed(0)
    m = DeltaNAR(task="sssp", **cfg).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    tr = make_dataset(500, 16, n_steps=5, seed0=0, kinds=WEIGHT)  # train: weight changes only
    for _ in range(50):
        m.train(); random.shuffle(tr)
        for ch in chunks(tr, 64):
            b = collate(ch, device=dev)
            loss, _ = compute_loss(b, *m(b))
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"{name:>10} {ev(m, WEIGHT):>16.3f} {ev(m, STRUCT):>14.3f}")
