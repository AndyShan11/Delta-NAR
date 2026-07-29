"""Static single-source shortest path references (used as ground-truth oracle)."""
import heapq

from .graphs import INF


def dijkstra(adj, source, n):
    """Static SSSP via Dijkstra. Returns (dist, pred) as length-n lists.

    dist[v] = shortest distance source -> v (INF if unreachable).
    pred[v] = predecessor of v on a shortest path (-1 if none / source).
    Assumes non-negative weights.
    """
    dist = [INF] * n
    pred = [-1] * n
    dist[source] = 0
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, pred


def widest_path(adj, source, n):
    """Maximum-bottleneck (widest) path from source.

    A DIFFERENT semiring from SSSP: max-min instead of min-plus, but the same
    output structure (a value + a predecessor per node) — so it reuses DeltaNAR's
    pointer/value decoder and copy mechanism, testing whether the incremental
    mechanism is semiring-agnostic.

    width[v] = max over source->v paths of the minimum edge weight on the path.
    """
    width = [float("-inf")] * n
    width[source] = float("inf")
    pred = [-1] * n
    visited = [False] * n
    pq = [(-float("inf"), source)]  # max-heap via negation
    while pq:
        negw, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        wu = -negw
        for v, cap in adj[u].items():
            nb = cap if wu > cap else wu  # min(wu, cap) = bottleneck through u
            if nb > width[v]:
                width[v] = nb
                pred[v] = u
                heapq.heappush(pq, (-nb, v))
    return width, pred
