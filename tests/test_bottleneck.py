"""Cross-check widest-path ground truth against a Floyd-style bottleneck oracle.

Run directly:  python tests/test_bottleneck.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graphs import gen_er_digraph  # noqa: E402
from src.sssp import widest_path  # noqa: E402

NEG = float("-inf")


def floyd_bottleneck(adj, n):
    """Independent oracle: B[i][j] = max-bottleneck capacity i->j."""
    B = [[NEG] * n for _ in range(n)]
    for i in range(n):
        B[i][i] = float("inf")
    for u in adj:
        for v, w in adj[u].items():
            if w > B[u][v]:
                B[u][v] = w
    for k in range(n):
        Bk = B[k]
        for i in range(n):
            bik = B[i][k]
            if bik == NEG:
                continue
            Bi = B[i]
            for j in range(n):
                cand = bik if Bk[j] > bik else Bk[j]  # min(B[i][k], B[k][j])
                if cand > Bi[j]:
                    Bi[j] = cand
    return B


def test_widest_matches_floyd():
    for seed in range(150):
        n = random.Random(seed).randint(4, 18)
        adj = gen_er_digraph(n, p=0.3, seed=seed)
        width, _ = widest_path(adj, 0, n)
        assert width == floyd_bottleneck(adj, n)[0], f"widest != floyd seed={seed}"


def test_pred_tree_valid():
    for seed in range(150):
        n = random.Random(seed).randint(4, 18)
        adj = gen_er_digraph(n, p=0.3, seed=seed)
        width, pred = widest_path(adj, 0, n)
        for v in range(n):
            if v == 0 or width[v] == NEG:
                continue
            p = pred[v]
            assert p != -1, f"reachable v={v} has no pred (seed={seed})"
            expect = min(width[p], adj[p][v])
            assert width[v] == expect, f"bad pred bottleneck v={v} (seed={seed})"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"\nALL {len(tests)} TESTS PASS")
