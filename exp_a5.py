"""A5: copy_scale / copy_margin sensitivity (robustness — no fine-tuning needed)."""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import make_dataset  # noqa: E402
from src.dataset_bottleneck import make_dataset_bottleneck  # noqa: E402
from src.model import DeltaNAR, collate, compute_loss, evaluate  # noqa: E402


def chunks(l, k):
    for i in range(0, len(l), k):
        yield l[i:i + k]


def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


dev = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def ev(m, mk, sem, n=64, ne=60):
    m.eval(); tot = 0.0; c = 0
    for ch in chunks(mk(ne, n, n_steps=5, seed0=900_000), 32):
        b = collate(ch, device=dev)
        tot += evaluate(b, *m(b), semiring=sem)["node_acc_v"] * len(ch); c += len(ch)
    return tot / c


def te(mk, sem, cs, cm):
    seed(0)
    m = DeltaNAR(task="sssp", copy_scale=cs, copy_margin=cm).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    tr = mk(400, 16, n_steps=5, seed0=0)
    for _ in range(40):
        m.train(); random.shuffle(tr)
        for ch in chunks(tr, 64):
            b = collate(ch, device=dev)
            loss, _ = compute_loss(b, *m(b))
            opt.zero_grad(); loss.backward(); opt.step()
    return ev(m, mk, sem)


print("SSSP node_acc@64 vs copy_scale (margin=0):")
for cs in [1, 2, 5, 10, 20]:
    print(f"  scale={cs:>2}: {te(make_dataset, 'min_plus', float(cs), 0.0):.3f}")
print("bottleneck node_acc@64 vs copy_margin (scale=5):")
for cm in [0, 1, 2, 3]:
    print(f"  margin={cm}: {te(make_dataset_bottleneck, 'max_min', 5.0, float(cm)):.3f}")
