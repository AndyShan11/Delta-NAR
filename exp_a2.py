"""A2: node-acc vs |A_t| + gate recall rho. Train DeltaNAR(SSSP); evaluate
per-sample across mixed delta sizes (|A_t| spans a range) to verify the bound's
|A_t|-dependence (acc should drop as |A_t| grows) and measure rho.
"""
import collections
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import make_dataset  # noqa: E402
from src.model import DeltaNAR, collate, compute_loss  # noqa: E402


def chunks(l, k):
    for i in range(0, len(l), k):
        yield l[i:i + k]


def bkt(a):
    return "0" if a == 0 else "1-2" if a <= 2 else "3-5" if a <= 5 else "6-10" if a <= 10 else "11+"


dev = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0); np.random.seed(0); torch.manual_seed(0)
model = DeltaNAR(task="sssp").to(dev)
opt = torch.optim.Adam(model.parameters(), 1e-3)
tr = make_dataset(500, 16, n_steps=5, seed0=0)
for _ in range(50):
    model.train(); random.shuffle(tr)
    for ch in chunks(tr, 64):
        b = collate(ch, device=dev)
        loss, _ = compute_loss(b, *model(b))
        opt.zero_grad(); loss.backward(); opt.step()
model.eval()

bk = collections.defaultdict(list)
rho = []
with torch.no_grad():
    for n in [16, 64]:
        for ds in [1, 2, 4, 8]:
            for s in make_dataset(40, n, n_steps=5, seed0=900_000 + ds * 1000, delta_size=ds):
                b = collate([s], device=dev)
                g, score, al = model(b)
                mask = (b["reach_next"] * (1 - b["is_src"])) > 0
                pred = score.argmax(-1)
                B, nn, _ = score.shape
                dt = b["g_next"] * b["scale"]
                wuv = torch.gather(b["W"], 1, pred.unsqueeze(1).expand(B, 1, nn)).squeeze(1)
                du = torch.gather(dt, 1, pred)
                valid = ((du + wuv - dt).abs() < 0.5) | (~mask)
                if mask.any():
                    bk[bkt(int(b["affected"].sum()))].append(valid[mask].float().mean().item())
                aff = b["affected"] > 0
                gh = al > 0
                tp = (gh & aff).sum().item()
                fn = ((~gh) & aff).sum().item()
                if tp + fn > 0:
                    rho.append(tp / (tp + fn))

print("|A_t| bucket -> mean node-acc (count):")
for k in ["0", "1-2", "3-5", "6-10", "11+"]:
    if k in bk:
        v = bk[k]
        print(f"  {k:>5}: node-acc={sum(v) / len(v):.3f}  (n={len(v)})")
print(f"\ngate recall rho = {sum(rho) / len(rho):.3f}  (over {len(rho)} samples)")
