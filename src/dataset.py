"""Build CLRS-Δ training samples from the incremental SSSP engine.

Each sample is one edit step: the model sees the POST-edit graph, the PREVIOUS
solution (distances + predecessors on the pre-edit graph), and the delta (where
the perturbation entered); it must predict the updated solution and the affected
region. Targets are produced by the verified DynamicSSSP engine, so a sample's
`g_next` always equals from-scratch Dijkstra on the post-edit graph (asserted in
tests/test_dataset.py).

Arrays use np.inf for unreachable distances; tensorisation / capping happens in
the model layer so this layer stays a pure, checkable data generator.
"""
import random

import numpy as np

from .dynamic_sssp import DynamicSSSP
from .graphs import (apply_delta, apply_deltas, gen_er_digraph, gen_graph,
                     sample_delta, sample_deltas)

# weight_increase and weight_decrease both reduce to an existing-edge "weight" edit.
TYPE_ID = {"weight": 0, "add": 1, "delete": 2}
DEFAULT_KINDS = ["weight_increase", "weight_decrease", "edge_add", "edge_delete"]


def adj_to_matrices(adj, n):
    """Adjacency map -> (W, A): weight matrix and 0/1 existence mask."""
    W = np.zeros((n, n), dtype=np.float64)
    A = np.zeros((n, n), dtype=np.float64)
    for u in adj:
        for v, w in adj[u].items():
            W[u, v] = w
            A[u, v] = 1.0
    return W, A


def make_episode(n, p=0.3, n_steps=4, kinds=None, seed=0, source=0, weight_range=(1, 10),
                 delta_size=1, topology="er", heavy_tail=False):
    """Generate one graph and a sequence of (possibly multi-edge) edits.

    delta_size = ‖δ‖ edges changed simultaneously each step.
    topology in {er, ba, grid}; heavy_tail toggles Pareto weights (for OOD).
    """
    kinds = kinds or DEFAULT_KINDS
    rng = random.Random(seed)
    adj = gen_graph(n, topology, p, weight_range, seed, heavy_tail)
    ds = DynamicSSSP(adj, source, n)
    cur = adj
    samples = []
    for _ in range(n_steps):
        deltas = sample_deltas(cur, delta_size, kinds, rng, weight_range)
        if not deltas:
            continue
        g_prev = np.array(ds.distances(), dtype=np.float64)
        pred_prev = np.array(ds.predecessors(), dtype=np.int64)

        cur = apply_deltas(cur, deltas)
        res = ds.apply_deltas(deltas)

        W, A = adj_to_matrices(cur, n)
        affected = np.zeros(n, dtype=np.float64)
        affected[res["affected"]] = 1.0
        samples.append(
            {
                "n": n,
                "source": source,
                "W": W,              # post-edit weights
                "A": A,              # post-edit existence mask
                "g_prev": g_prev,    # previous-solution distances
                "pred_prev": pred_prev,
                "deltas": deltas,    # list of edits this step (‖δ‖ = len(deltas))
                "g_next": np.array(ds.distances(), dtype=np.float64),  # target distances
                "pred_next": np.array(ds.predecessors(), dtype=np.int64),  # target pointers
                "affected": affected,  # gate target (LPA* work-set)
                "size_affected": int(affected.sum()),
            }
        )
    return samples


def make_dataset(n_episodes, n, n_steps=4, seed0=0, **kw):
    """Concatenate samples from many episodes (distinct seeds)."""
    out = []
    for e in range(n_episodes):
        out.extend(make_episode(n, n_steps=n_steps, seed=seed0 + e, **kw))
    return out
