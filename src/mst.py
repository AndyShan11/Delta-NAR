"""Minimum spanning tree for the CLRS-Δ incremental-MST task (second task).

Undirected weighted graphs as symmetric adjacency maps adj[u][v]=adj[v][u]=w.
Ground truth is computed from scratch by Kruskal; Prim cross-checks correctness.
DISTINCT edge weights => unique MST => well-defined edge-membership target.

Different optimisation objective from SSSP (minimise total weight, not distance
to a source) — used to show the incremental-reasoning mechanism is task-general.
"""
import heapq
import random

INF = float("inf")


class _UF:
    """Union-find with path halving."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def undirected_edges(adj):
    """Unique undirected edges as (u, v, w) with u < v."""
    return [(u, v, w) for u in adj for v, w in adj[u].items() if u < v]


def gen_er_ungraph(n, p=0.3, seed=None):
    """Connected ER undirected graph with DISTINCT integer weights.

    A random spanning tree is laid down first (guarantees connectivity), then
    extra edges are added with probability p.
    """
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    edge_set = set()
    for i in range(1, n):  # random spanning tree
        a, b = perm[i], perm[rng.randint(0, i - 1)]
        edge_set.add((min(a, b), max(a, b)))
    for u in range(n):  # extra edges
        for v in range(u + 1, n):
            if (u, v) not in edge_set and rng.random() < p:
                edge_set.add((u, v))
    edges = list(edge_set)
    weights = rng.sample(range(1, len(edges) * 10 + 10), k=len(edges))  # distinct
    adj = {u: {} for u in range(n)}
    for (u, v), w in zip(edges, weights):
        adj[u][v] = w
        adj[v][u] = w
    return adj


def kruskal(adj, n):
    """Returns (mst_edge_set {(u<v)}, total_weight)."""
    uf = _UF(n)
    mst, total = set(), 0
    for u, v, w in sorted(undirected_edges(adj), key=lambda e: e[2]):
        if uf.union(u, v):
            mst.add((u, v))
            total += w
    return mst, total


def prim(adj, n):
    """Independent MST algorithm for cross-checking Kruskal."""
    visited = [False] * n
    mst, total = set(), 0
    for start in range(n):  # loop handles disconnected graphs (forest)
        if visited[start]:
            continue
        pq = [(0, start, -1)]
        while pq:
            w, u, par = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True
            if par != -1:
                mst.add((min(u, par), max(u, par)))
                total += w
            for v, ww in adj[u].items():
                if not visited[v]:
                    heapq.heappush(pq, (ww, v, u))
    return mst, total


def mst_parent(mst, n, root=0):
    """Rooted-parent array of the MST (for the prev-solution feature). -1 at roots."""
    madj = {v: [] for v in range(n)}
    for u, v in mst:
        madj[u].append(v)
        madj[v].append(u)
    parent = [-1] * n
    seen = [False] * n
    for r in range(n):
        if seen[r]:
            continue
        seen[r] = True
        stack = [r]
        while stack:
            x = stack.pop()
            for y in madj[x]:
                if not seen[y]:
                    seen[y] = True
                    parent[y] = x
                    stack.append(y)
    return parent


def _unused_weight(adj, rng, hi=10_000_000):
    used = {w for u in adj for w in adj[u].values()}
    while True:
        w = rng.randint(1, hi)
        if w not in used:
            return w


def sample_delta_un(adj, kind, rng):
    """Sample an undirected edit (weight change or edge add) with a distinct weight."""
    n = len(adj)
    if kind == "weight":
        edges = undirected_edges(adj)
        if not edges:
            return None
        u, v, w = rng.choice(edges)
        return {"type": "weight", "u": u, "v": v, "old": w, "new": _unused_weight(adj, rng)}
    if kind == "add":
        for _ in range(100):
            u, v = rng.randrange(n), rng.randrange(n)
            if u != v and v not in adj[u]:
                return {"type": "add", "u": min(u, v), "v": max(u, v), "new": _unused_weight(adj, rng)}
        return None
    raise ValueError(kind)


def apply_delta_un(adj, delta):
    """Return a NEW symmetric adjacency map with the delta applied."""
    new = {u: dict(d) for u, d in adj.items()}
    u, v, nw = delta["u"], delta["v"], delta["new"]
    new[u][v] = nw
    new[v][u] = nw
    return new
