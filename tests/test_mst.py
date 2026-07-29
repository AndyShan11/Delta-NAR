"""Cross-check MST ground truth: Kruskal must equal Prim under distinct weights.

Run directly:  python tests/test_mst.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mst import (_UF, apply_delta_un, gen_er_ungraph, kruskal,  # noqa: E402
                     prim, sample_delta_un)


def test_kruskal_equals_prim():
    for seed in range(400):
        n = random.Random(seed).randint(4, 25)
        adj = gen_er_ungraph(n, p=0.3, seed=seed)
        mk, tk = kruskal(adj, n)
        mp, tp = prim(adj, n)
        assert tk == tp, f"total-weight mismatch seed={seed}: {tk} vs {tp}"
        assert mk == mp, f"edge-set mismatch seed={seed}"


def test_spanning_tree_properties():
    for seed in range(150):
        n = random.Random(seed).randint(4, 25)
        adj = gen_er_ungraph(n, p=0.3, seed=seed)
        mst, _ = kruskal(adj, n)
        assert len(mst) == n - 1, f"not spanning seed={seed}: {len(mst)} edges"
        uf = _UF(n)
        for u, v in mst:
            assert uf.union(u, v), f"cycle in MST seed={seed}"  # union False => already linked
        assert len({uf.find(i) for i in range(n)}) == 1, f"MST not connected seed={seed}"


def test_delta_keeps_mst_consistent():
    for seed in range(250):
        rng = random.Random(3000 + seed)
        n = rng.randint(5, 25)
        cur = gen_er_ungraph(n, p=0.3, seed=seed)
        for step in range(4):
            d = sample_delta_un(cur, rng.choice(["weight", "add"]), rng)
            if d is None:
                continue
            cur = apply_delta_un(cur, d)
            mk, tk = kruskal(cur, n)
            mp, tp = prim(cur, n)
            assert tk == tp and mk == mp, f"post-delta MST mismatch seed={seed} step={step}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"\nALL {len(tests)} TESTS PASS")
