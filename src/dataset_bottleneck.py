"""CLRS-Δ incremental widest/bottleneck-path samples (max-min semiring).

Same output structure as SSSP (per-node value + predecessor), so it reuses the
SSSP collate / decoder / loss. Ground truth is from-scratch widest_path; affected
nodes are those whose bottleneck value or predecessor changed, plus delta endpoints.
"""
import random

import numpy as np

from .dataset import DEFAULT_KINDS, adj_to_matrices
from .graphs import apply_deltas, gen_er_digraph, gen_graph, sample_deltas
from .sssp import widest_path


def make_episode_bottleneck(n, p=0.3, n_steps=4, kinds=None, seed=0, source=0,
                            weight_range=(1, 10), delta_size=1, topology="er", heavy_tail=False):
    kinds = kinds or DEFAULT_KINDS
    rng = random.Random(seed)
    cur = gen_graph(n, topology, p, weight_range, seed, heavy_tail)
    width_prev, pred_prev = widest_path(cur, source, n)
    samples = []
    for _ in range(n_steps):
        deltas = sample_deltas(cur, delta_size, kinds, rng, weight_range)
        if not deltas:
            continue
        wp = np.array(width_prev, dtype=np.float64)
        pp = np.array(pred_prev, dtype=np.int64)

        cur = apply_deltas(cur, deltas)
        width_next, pred_next = widest_path(cur, source, n)
        wn = np.array(width_next, dtype=np.float64)
        pn = np.array(pred_next, dtype=np.int64)

        affected = np.zeros(n, dtype=np.float64)
        affected[(wn != wp) | (pn != pp)] = 1.0
        for d in deltas:
            affected[d["u"]] = 1.0
            affected[d["v"]] = 1.0

        W, A = adj_to_matrices(cur, n)
        samples.append(
            {
                "n": n, "source": source, "W": W, "A": A,
                "g_prev": wp, "pred_prev": pp, "deltas": deltas,
                "g_next": wn, "pred_next": pn, "affected": affected,
                "size_affected": int(affected.sum()),
            }
        )
        width_prev, pred_prev = width_next, pred_next
    return samples


def make_dataset_bottleneck(n_episodes, n, n_steps=4, seed0=0, **kw):
    out = []
    for e in range(n_episodes):
        out.extend(make_episode_bottleneck(n, n_steps=n_steps, seed=seed0 + e, **kw))
    return out
