"""A4: message-passing steps ablation (+ the +3 efficiency claim).

DeltaNAR localises the update, so it should work with very few message-passing
steps; from-scratch (Recompute) needs ~diameter(~n) steps to propagate. Few steps
== less compute, so this doubles as the sublinear-update efficiency evidence.
"""
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


@torch.no_grad()
def ev(m, n, ne=60):
    m.eval(); tot = 0.0; c = 0
    for ch in chunks(make_dataset(ne, n, n_steps=5, seed0=900_000), 32):
        b = collate(ch, device=dev)
        tot += evaluate(b, *m(b))["node_acc_v"] * len(ch); c += len(ch)
    return tot / c


print("node_acc_v (train_n=16); DeltaNAR localises => few steps suffice")
print(f"{'steps':>6} | {'DeltaNAR n16':>12} {'n64':>7} | {'Recompute n16':>13} {'n64':>7}")
for st in [2, 4, 8, 16]:
    out = {}
    for name, cfg in [("D", dict(use_prev=True, use_delta=True, use_gate=True)),
                      ("R", dict(use_prev=False, use_delta=False, use_gate=False))]:
        seed(0)
        m = DeltaNAR(steps=st, task="sssp", **cfg).to(dev)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        tr = make_dataset(400, 16, n_steps=5, seed0=0)
        for _ in range(40):
            m.train(); random.shuffle(tr)
            for ch in chunks(tr, 64):
                b = collate(ch, device=dev)
                loss, _ = compute_loss(b, *m(b))
                opt.zero_grad(); loss.backward(); opt.step()
        out[name] = (ev(m, 16), ev(m, 64))
    print(f"{st:>6} | {out['D'][0]:>12.3f} {out['D'][1]:>7.3f} | {out['R'][0]:>13.3f} {out['R'][1]:>7.3f}")
