"""Incremental single-source shortest path (LPA*-style, heuristic h = 0).

This is the ground-truth engine for the CLRS-Δ dynamic-SSSP task. It maintains
the LPA* invariants g / rhs and, after an edge edit, recomputes ONLY the
locally-inconsistent vertices (Ramalingam–Reps style output-sensitive update),
emitting the recomputation trajectory used as hint supervision in DeltaNAR.

Correspondence to the three supervision signals (see DESIGN.md):
  * final answer            = g (distances) + recovered predecessors
  * dynamic hints           = the set of vertices that enter the priority queue
                              (the locally-inconsistent / affected region)
  * local recompute traject = the order in which vertices are popped/expanded
"""
import heapq

from .graphs import INF


class _PQ:
    """Min priority queue with lazy deletion + key updates.

    Heap entries are mutable lists [key, node, valid]; a stale entry is flagged
    valid=False and skipped. `node` ids are unique, so the bool `valid` field
    never participates in heap ordering.
    """

    def __init__(self):
        self.heap = []
        self.entry = {}  # node -> entry list currently considered valid

    def __contains__(self, node):
        return node in self.entry

    def insert(self, node, key):
        if node in self.entry:  # invalidate the previous entry for this node
            self.entry[node][2] = False
        e = [key, node, True]
        self.entry[node] = e
        heapq.heappush(self.heap, e)

    def remove(self, node):
        if node in self.entry:
            self.entry[node][2] = False
            del self.entry[node]

    def _clean(self):
        while self.heap and not self.heap[0][2]:
            heapq.heappop(self.heap)

    def empty(self):
        self._clean()
        return not self.heap

    def pop(self):
        self._clean()
        _, node, _ = heapq.heappop(self.heap)
        del self.entry[node]
        return node


class DynamicSSSP:
    """Maintain SSSP from a fixed source under a stream of edge edits."""

    def __init__(self, adj, source, n):
        self.n = n
        self.source = source
        # forward + reverse adjacency (reverse needed to recompute rhs of a vertex)
        self.adj = {u: dict(d) for u, d in adj.items()}
        self.radj = {v: {} for v in range(n)}
        for u in self.adj:
            for v, w in self.adj[u].items():
                self.radj[v][u] = w
        self.g = [INF] * n
        self.rhs = [INF] * n
        self.rhs[source] = 0
        self.U = _PQ()
        self.U.insert(source, 0)
        self._compute()  # initial full computation (equivalent to Dijkstra)

    # --- LPA* core -------------------------------------------------------
    def _key(self, v):
        return min(self.g[v], self.rhs[v])  # h = 0

    def _update_vertex(self, v):
        if v != self.source:
            best = INF
            for u, w in self.radj[v].items():
                if self.g[u] < INF:
                    cand = self.g[u] + w
                    if cand < best:
                        best = cand
            self.rhs[v] = best
        if v in self.U:
            self.U.remove(v)
        if self.g[v] != self.rhs[v]:
            self.U.insert(v, self._key(v))

    def _compute(self, record=None):
        """Process the queue until empty (full SSSP, no target early-stop)."""
        while not self.U.empty():
            u = self.U.pop()
            if record is not None:
                record.append({"node": u, "g_before": self.g[u], "rhs": self.rhs[u]})
            if self.g[u] > self.rhs[u]:  # overconsistent: distance improved
                self.g[u] = self.rhs[u]
                for s in self.adj[u]:
                    self._update_vertex(s)
            else:  # underconsistent: distance worsened -> reset and re-settle
                self.g[u] = INF
                self._update_vertex(u)
                for s in self.adj[u]:
                    self._update_vertex(s)

    # --- public API ------------------------------------------------------
    def apply_delta(self, delta):
        """Apply one edge edit and incrementally update.

        Returns {trajectory, affected, changed}:
          trajectory : list of pop steps (node + g/rhs at pop time)
          affected   : sorted vertices that entered the queue (hint target)
          changed    : vertices whose final distance actually changed
        """
        u, v, t = delta["u"], delta["v"], delta["type"]
        if t == "delete":
            self.adj[u].pop(v, None)
            self.radj[v].pop(u, None)
        else:  # 'weight' (existing edge) or 'add' (new edge)
            nw = delta["new"]
            self.adj[u][v] = nw
            self.radj[v][u] = nw

        g_before = list(self.g)
        record = []
        self._update_vertex(v)  # only head v's rhs can change directly
        self._compute(record=record)

        affected = sorted({step["node"] for step in record})
        changed = [i for i in range(self.n) if self.g[i] != g_before[i]]
        return {"trajectory": record, "affected": affected, "changed": changed}

    def apply_deltas(self, deltas):
        """Apply several simultaneous edge edits, then a single incremental update."""
        g_before = list(self.g)
        seeds = []
        for d in deltas:
            u, v, t = d["u"], d["v"], d["type"]
            if t == "delete":
                self.adj[u].pop(v, None)
                self.radj[v].pop(u, None)
            else:
                nw = d["new"]
                self.adj[u][v] = nw
                self.radj[v][u] = nw
            seeds.append(v)
        record = []
        for v in seeds:
            self._update_vertex(v)  # radj[v] already holds all new in-edges
        self._compute(record=record)
        affected = sorted({step["node"] for step in record})
        changed = [i for i in range(self.n) if self.g[i] != g_before[i]]
        return {"trajectory": record, "affected": affected, "changed": changed}

    def distances(self):
        return list(self.g)

    def predecessors(self):
        """Recover a shortest-path-tree predecessor for each reachable vertex."""
        pred = [-1] * self.n
        for v in range(self.n):
            if v == self.source or self.g[v] >= INF:
                continue
            for u, w in self.radj[v].items():
                if self.g[u] < INF and self.g[u] + w == self.g[v]:
                    pred[v] = u
                    break
        return pred
