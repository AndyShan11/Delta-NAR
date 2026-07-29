"""Validate that packed samples carry correct targets.

Run directly:  python tests/test_dataset.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import adj_to_matrices, make_episode  # noqa: E402
from src.sssp import dijkstra  # noqa: E402


def _matrices_to_adj(W, A, n):
    adj = {u: {} for u in range(n)}
    for u in range(n):
        for v in range(n):
            if A[u, v] > 0:
                adj[u][v] = W[u, v]
    return adj


def test_sample_targets_match_dijkstra():
    """g_next reconstructed from (W, A) must equal from-scratch Dijkstra."""
    for seed in range(80):
        n = random.Random(seed).randint(6, 22)
        for s in make_episode(n, n_steps=5, seed=seed):
            adj = _matrices_to_adj(s["W"], s["A"], n)
            dref, _ = dijkstra(adj, s["source"], n)
            assert np.array_equal(s["g_next"], np.array(dref, dtype=np.float64)), (
                f"target mismatch seed={seed}"
            )


def test_affected_covers_changed():
    """Every vertex whose distance changed must be in the affected mask."""
    for seed in range(80):
        n = random.Random(seed).randint(6, 22)
        for s in make_episode(n, n_steps=5, seed=seed):
            changed = s["g_next"] != s["g_prev"]
            assert np.all(s["affected"][changed] == 1.0), f"affected misses changed seed={seed}"


def test_shapes_consistent():
    for s in make_episode(16, n_steps=4, seed=7):
        n = s["n"]
        assert s["W"].shape == (n, n) and s["A"].shape == (n, n)
        for key in ("g_prev", "pred_prev", "g_next", "pred_next", "affected"):
            assert s[key].shape == (n,), f"{key} shape {s[key].shape}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"\nALL {len(tests)} TESTS PASS")
