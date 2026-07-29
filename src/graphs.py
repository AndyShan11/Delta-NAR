"""Graph generation and edge deltas for the CLRS-Δ dynamic-SSSP benchmark.

Graphs are weighted directed graphs represented as adjacency maps:
    adj : dict[int, dict[int, int]]   # adj[u][v] = weight(u -> v)
All weights are positive integers (Dijkstra / LPA* require non-negative weights).
Functions here are pure standard library so the ground-truth pipeline is fully
unit-testable on the local machine without numpy/torch.
"""
import random

INF = float("inf")


def gen_er_digraph(n, p=0.3, weight_range=(1, 10), seed=None):
    """Erdős–Rényi directed graph G(n, p) with random positive integer weights.

    Returns adj: dict[u] -> dict[v] -> weight.
    """
    rng = random.Random(seed)
    lo, hi = weight_range
    adj = {u: {} for u in range(n)}
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                adj[u][v] = rng.randint(lo, hi)
    return adj


def edges(adj):
    """Flat list of (u, v, w) edges."""
    return [(u, v, w) for u in adj for v, w in adj[u].items()]


def sample_delta(adj, kind, rng, weight_range=(1, 10)):
    """Sample one edge edit of the given kind. Does NOT mutate `adj`.

    kind in {weight_increase, weight_decrease, edge_add, edge_delete}.
    Returns a delta dict {type, u, v, old?, new?} or None if not applicable.
    """
    lo, hi = weight_range
    elist = edges(adj)
    n = len(adj)

    if kind in ("weight_increase", "weight_decrease"):
        if not elist:
            return None
        u, v, w = rng.choice(elist)
        if kind == "weight_increase":
            new_w = w + rng.randint(1, hi)
        else:  # decrease, keep >= lo
            new_w = max(lo, w - rng.randint(1, max(1, w - lo)))
        return {"type": "weight", "u": u, "v": v, "old": w, "new": new_w}

    if kind == "edge_delete":
        if not elist:
            return None
        u, v, w = rng.choice(elist)
        return {"type": "delete", "u": u, "v": v, "old": w}

    if kind == "edge_add":
        for _ in range(100):
            u, v = rng.randrange(n), rng.randrange(n)
            if u != v and v not in adj[u]:
                return {"type": "add", "u": u, "v": v, "new": rng.randint(lo, hi)}
        return None  # graph nearly complete

    raise ValueError(f"unknown delta kind: {kind}")


def apply_delta(adj, delta):
    """Return a NEW adjacency map with `delta` applied (original untouched)."""
    new = {u: dict(d) for u, d in adj.items()}
    u, v, t = delta["u"], delta["v"], delta["type"]
    if t == "delete":
        new[u].pop(v, None)
    else:  # weight change or edge add
        new[u][v] = delta["new"]
    return new


def sample_deltas(adj, k, kinds, rng, weight_range=(1, 10)):
    """Sample up to k distinct-edge deltas (for dynamic-intensity ‖δ‖=k)."""
    deltas, used = [], set()
    for _ in range(k * 4):
        if len(deltas) >= k:
            break
        d = sample_delta(adj, rng.choice(kinds), rng, weight_range)
        if d is None:
            continue
        key = (d["u"], d["v"])
        if key in used:
            continue
        used.add(key)
        deltas.append(d)
    return deltas


def apply_deltas(adj, deltas):
    """Return a NEW adjacency map with all deltas applied simultaneously."""
    new = {u: dict(d) for u, d in adj.items()}
    for delta in deltas:
        u, v, t = delta["u"], delta["v"], delta["type"]
        if t == "delete":
            new[u].pop(v, None)
        else:
            new[u][v] = delta["new"]
    return new


# --- topology / weight variants for OOD generalization ---------------------

def _one_weight(rng, weight_range, heavy_tail):
    lo, hi = weight_range
    if heavy_tail:  # Pareto-tailed positive integers (capped)
        return min(hi * 20, lo + int(rng.paretovariate(1.3)))
    return rng.randint(lo, hi)


def gen_ba_digraph(n, m=2, weight_range=(1, 10), seed=None, heavy_tail=False):
    """Barabási–Albert scale-free graph, directed (each edge randomly oriented)."""
    rng = random.Random(seed)
    adj = {u: {} for u in range(n)}
    repeated = []
    for v in range(n):
        if v < max(1, m):
            for u in range(v):
                a, b = (u, v) if rng.random() < 0.5 else (v, u)
                adj[a][b] = _one_weight(rng, weight_range, heavy_tail)
                repeated += [u, v]
        else:
            chosen, tries = set(), 0
            while len(chosen) < m and tries < 100:
                u = rng.choice(repeated) if repeated else rng.randrange(v)
                if u != v:
                    chosen.add(u)
                tries += 1
            for u in chosen:
                a, b = (u, v) if rng.random() < 0.5 else (v, u)
                adj[a][b] = _one_weight(rng, weight_range, heavy_tail)
                repeated += [u, v]
    return adj


def gen_grid_digraph(n, weight_range=(1, 10), seed=None, heavy_tail=False):
    """2D grid on side*side<=n nodes (extra nodes isolated), bidirectional edges."""
    rng = random.Random(seed)
    side = max(1, int(n ** 0.5))
    adj = {u: {} for u in range(n)}
    for i in range(side):
        for j in range(side):
            u = i * side + j
            if j + 1 < side:
                w = i * side + (j + 1)
                adj[u][w] = _one_weight(rng, weight_range, heavy_tail)
                adj[w][u] = _one_weight(rng, weight_range, heavy_tail)
            if i + 1 < side:
                w = (i + 1) * side + j
                adj[u][w] = _one_weight(rng, weight_range, heavy_tail)
                adj[w][u] = _one_weight(rng, weight_range, heavy_tail)
    return adj


def gen_graph(n, topology="er", p=0.3, weight_range=(1, 10), seed=None, heavy_tail=False):
    """Dispatch graph generation by topology, with optional heavy-tailed weights."""
    if topology == "ba":
        return gen_ba_digraph(n, weight_range=weight_range, seed=seed, heavy_tail=heavy_tail)
    if topology == "grid":
        return gen_grid_digraph(n, weight_range=weight_range, seed=seed, heavy_tail=heavy_tail)
    # default: Erdos-Renyi (optionally heavy-tailed weights)
    rng = random.Random(seed)
    adj = {u: {} for u in range(n)}
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                adj[u][v] = _one_weight(rng, weight_range, heavy_tail)
    return adj
