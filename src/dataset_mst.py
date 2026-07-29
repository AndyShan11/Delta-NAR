"""CLRS-Δ incremental-MST samples (second task, edge-membership target).

Each sample: post-edit undirected graph + previous MST + delta -> updated MST
(as a symmetric edge-membership mask) + affected nodes. Ground truth is from
scratch Kruskal (cross-checked by Prim in tests/test_mst.py). Edge representation
keeps the affected region local (one weight change swaps a single MST edge).
"""
import random

import numpy as np

from .mst import (apply_delta_un, gen_er_ungraph, kruskal, mst_parent,
                  sample_delta_un, undirected_edges)

MST_KINDS = ["weight", "add"]


def _distinct_weight(adj, direction, rng):
    """Smallest ('down') or largest ('up') unused positive-integer weight."""
    used = {w for u in adj for w in adj[u].values()}
    if direction == "down":
        w = 1
        while w in used:
            w += 1
        return w
    w = (max(used) if used else 0) + 1
    while w in used:
        w += 1
    return w


def _informative_delta(cur, mst, rng):
    """A weight edit likely to actually change the MST (so the model must update,
    not just copy). Decrease a non-tree edge (forces it in, ejecting the cycle-max
    edge) or increase a tree edge (may eject it)."""
    mst_list = list(mst)
    non_tree = [(u, v) for u, v, _ in undirected_edges(cur) if (u, v) not in mst]
    if rng.random() < 0.5 and non_tree:          # pull a non-tree edge IN (guaranteed swap)
        u, v = rng.choice(non_tree)
        return {"type": "weight", "u": u, "v": v, "old": cur[u][v], "new": _distinct_weight(cur, "down", rng)}
    if mst_list:                                  # push a tree edge towards ejection
        u, v = rng.choice(mst_list)
        return {"type": "weight", "u": u, "v": v, "old": cur[u][v], "new": _distinct_weight(cur, "up", rng)}
    return None


def un_matrices(adj, n):
    """Symmetric (W, A) from an undirected adjacency map."""
    W = np.zeros((n, n), dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    for u in adj:
        for v, w in adj[u].items():
            W[u, v] = w
            A[u, v] = 1.0
    return W, A


def edgeset_to_mat(edges, n):
    """Symmetric 0/1 membership matrix from a set of (u<v) edges."""
    M = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        M[u, v] = 1.0
        M[v, u] = 1.0
    return M


def _sample_deltas_un(cur, k, rng):
    deltas, used = [], set()
    for _ in range(k * 4):
        if len(deltas) >= k:
            break
        d = sample_delta_un(cur, rng.choice(MST_KINDS), rng)
        if d is None:
            continue
        key = (d["u"], d["v"])
        if key in used:
            continue
        used.add(key)
        deltas.append(d)
    return deltas


def make_episode_mst(n, p=0.3, n_steps=4, seed=0, root=0, delta_size=1):
    rng = random.Random(seed)
    cur = gen_er_ungraph(n, p, seed)
    cur_mst, _ = kruskal(cur, n)
    samples = []
    for _ in range(n_steps):
        d = _informative_delta(cur, cur_mst, rng)
        if d is None:
            continue
        deltas = [d]
        prev_mst = cur_mst
        prev_parent = np.array(mst_parent(prev_mst, n, root), dtype=np.int64)
        for d in deltas:
            cur = apply_delta_un(cur, d)
        new_mst, _ = kruskal(cur, n)

        flipped = prev_mst ^ new_mst  # symmetric diff of MST edge sets
        affected = np.zeros(n, dtype=np.float64)
        for u, v in flipped:
            affected[u] = 1.0
            affected[v] = 1.0
        for d in deltas:  # delta endpoints are always examined by an incremental MST update
            affected[d["u"]] = 1.0
            affected[d["v"]] = 1.0

        W, A = un_matrices(cur, n)
        samples.append(
            {
                "n": n,
                "root": root,
                "W": W,
                "A": A,
                "prev_mst": edgeset_to_mat(prev_mst, n),   # previous solution (edges)
                "prev_parent": prev_parent,                # rooted, for prev feature
                "deltas": deltas,
                "mst_next": edgeset_to_mat(new_mst, n),    # target MST membership
                "affected": affected,                      # gate target
                "size_affected": int(affected.sum()),
            }
        )
        cur_mst = new_mst
    return samples


def make_dataset_mst(n_episodes, n, n_steps=4, seed0=0, **kw):
    out = []
    for e in range(n_episodes):
        out.extend(make_episode_mst(n, n_steps=n_steps, seed=seed0 + e, **kw))
    return out
