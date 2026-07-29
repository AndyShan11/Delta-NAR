"""Differential tests: the incremental engine must match from-scratch Dijkstra.

Run directly (no pytest required):  python tests/test_dynamic_sssp.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamic_sssp import DynamicSSSP  # noqa: E402
from src.graphs import (INF, apply_delta, apply_deltas, gen_er_digraph,  # noqa: E402
                        sample_delta, sample_deltas)
from src.sssp import dijkstra  # noqa: E402

KINDS = ["weight_increase", "weight_decrease", "edge_add", "edge_delete"]


def test_init_matches_dijkstra():
    """Initial full computation == Dijkstra on the same graph."""
    for seed in range(80):
        n = random.Random(seed).randint(5, 25)
        adj = gen_er_digraph(n, p=0.3, seed=seed)
        ds = DynamicSSSP(adj, 0, n)
        dref, _ = dijkstra(adj, 0, n)
        assert ds.distances() == dref, f"init mismatch seed={seed}"


def test_incremental_matches_recompute():
    """After each edit (and over sequences of edits), g == from-scratch Dijkstra."""
    for seed in range(300):
        rng = random.Random(1000 + seed)
        n = rng.randint(5, 25)
        adj = gen_er_digraph(n, p=0.3, seed=seed)
        ds = DynamicSSSP(adj, 0, n)
        cur = adj
        for step in range(6):
            kind = rng.choice(KINDS)
            delta = sample_delta(cur, kind, rng)
            if delta is None:
                continue
            cur = apply_delta(cur, delta)
            ds.apply_delta(delta)
            dref, _ = dijkstra(cur, 0, n)
            assert ds.distances() == dref, (
                f"incremental mismatch seed={seed} step={step} kind={kind}\n"
                f"got={ds.distances()}\nref={dref}"
            )


def test_multiedge_matches_recompute():
    """Simultaneous k-edge updates (dynamic intensity) match from-scratch Dijkstra."""
    for seed in range(200):
        rng = random.Random(7000 + seed)
        n = rng.randint(8, 25)
        adj = gen_er_digraph(n, p=0.3, seed=seed)
        ds = DynamicSSSP(adj, 0, n)
        cur = adj
        for step in range(4):
            k = rng.randint(2, 6)
            deltas = sample_deltas(cur, k, KINDS, rng)
            if not deltas:
                continue
            cur = apply_deltas(cur, deltas)
            ds.apply_deltas(deltas)
            dref, _ = dijkstra(cur, 0, n)
            assert ds.distances() == dref, (
                f"multi-edge mismatch seed={seed} step={step} k={len(deltas)}"
            )


def test_predecessor_tree_valid():
    """Recovered predecessors form a valid shortest-path tree."""
    for seed in range(80):
        n = random.Random(seed).randint(5, 25)
        adj = gen_er_digraph(n, p=0.3, seed=seed)
        ds = DynamicSSSP(adj, 0, n)
        g, pred = ds.distances(), ds.predecessors()
        for v in range(n):
            if v == 0 or g[v] >= INF:
                continue
            p = pred[v]
            assert p != -1, f"reachable v={v} has no predecessor (seed={seed})"
            assert g[p] + ds.adj[p][v] == g[v], f"bad pred edge v={v} (seed={seed})"


def test_unaffected_region_unchanged():
    """Output-sensitivity: vertices not in `changed` keep their old distance."""
    for seed in range(120):
        rng = random.Random(5000 + seed)
        n = rng.randint(5, 25)
        adj = gen_er_digraph(n, p=0.3, seed=seed)
        ds = DynamicSSSP(adj, 0, n)
        before = ds.distances()
        delta = sample_delta(adj, rng.choice(KINDS), rng)
        if delta is None:
            continue
        res = ds.apply_delta(delta)
        after = ds.distances()
        changed_set = set(res["changed"])
        for v in range(n):
            if v not in changed_set:
                assert after[v] == before[v], f"unchanged vertex moved v={v} seed={seed}"
            # every actually-changed vertex must have been visited (affected ⊇ changed)
        assert changed_set.issubset(set(res["affected"])), f"changed ⊄ affected seed={seed}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"\nALL {len(tests)} TESTS PASS")
