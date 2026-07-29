"""A1: affected-region size |A_t| vs n (pure ground-truth, no training).

Verifies the THEORY assumption: a single local edit's |A_t| does NOT grow ~ n,
so |A_t|/n -> 0, which is *why* DeltaNAR node-acc rises with n (copy dominates).
"""
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dynamic_sssp import DynamicSSSP  # noqa: E402
from src.graphs import gen_er_digraph, sample_delta  # noqa: E402

KINDS = ["weight_increase", "weight_decrease", "edge_add", "edge_delete"]


def measure(n, p, n_seeds=200):
    A = []
    for s in range(n_seeds):
        rng = random.Random(s)
        adj = gen_er_digraph(n, p=p, seed=s)
        ds = DynamicSSSP(adj, 0, n)
        d = sample_delta(adj, rng.choice(KINDS), rng)
        if d is None:
            continue
        A.append(len(ds.apply_delta(d)["affected"]))
    return statistics.mean(A), statistics.pstdev(A), 100.0 * statistics.mean(A) / n


print("=== training distribution (ER p=0.3) ===")
print(f"{'n':>6}{'|A_t|':>9}{'+-std':>8}{'|A|/n %':>10}")
for n in [16, 32, 64, 128, 256]:
    m, s, pct = measure(n, 0.3)
    print(f"{n:>6}{m:>9.2f}{s:>8.2f}{pct:>9.1f}%")

print("\n=== fixed avg-degree ~6 (sparse, road-like) ===")
print(f"{'n':>6}{'|A_t|':>9}{'+-std':>8}{'|A|/n %':>10}")
for n in [16, 32, 64, 128, 256, 512, 1024]:
    p = min(0.5, 6.0 / (n - 1))
    m, s, pct = measure(n, p, n_seeds=150)
    print(f"{n:>6}{m:>9.2f}{s:>8.2f}{pct:>9.1f}%")
