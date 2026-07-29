"""Validate incremental-MST samples carry correct targets.

Run directly:  python tests/test_mst_dataset.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_mst import edgeset_to_mat, make_episode_mst  # noqa: E402
from src.mst import kruskal  # noqa: E402


def _mat_to_unadj(W, A, n):
    adj = {u: {} for u in range(n)}
    for u in range(n):
        for v in range(n):
            if A[u, v] > 0:
                adj[u][v] = W[u, v]
    return adj


def test_target_matches_kruskal():
    """mst_next reconstructed from (W, A) equals from-scratch Kruskal."""
    for seed in range(100):
        n = random.Random(seed).randint(5, 22)
        for s in make_episode_mst(n, n_steps=4, seed=seed):
            adj = _mat_to_unadj(s["W"], s["A"], n)
            mst, _ = kruskal(adj, n)
            assert np.array_equal(s["mst_next"], edgeset_to_mat(mst, n)), f"seed={seed}"


def test_mst_is_spanning_and_symmetric():
    for seed in range(100):
        n = random.Random(seed).randint(5, 22)
        for s in make_episode_mst(n, n_steps=4, seed=seed):
            M = s["mst_next"]
            assert np.array_equal(M, M.T), f"not symmetric seed={seed}"
            assert M.sum() == 2 * (n - 1), f"not spanning seed={seed}: {M.sum()}"


def test_affected_covers_flipped():
    """Every node incident to a flipped MST edge is marked affected."""
    for seed in range(100):
        n = random.Random(seed).randint(5, 22)
        for s in make_episode_mst(n, n_steps=4, seed=seed):
            flipped = np.logical_xor(s["prev_mst"] > 0, s["mst_next"] > 0)
            incident = flipped.any(axis=1)
            assert np.all(s["affected"][incident] == 1.0), f"affected misses flip seed={seed}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"\nALL {len(tests)} TESTS PASS")
