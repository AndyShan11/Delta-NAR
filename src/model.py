"""DeltaNAR and ablation baselines for CLRS-Δ dynamic SSSP.

A single configurable module covers all three conditions in the decisive
experiment (see DESIGN.md):
  * DeltaNAR          : use_prev=True,  use_delta=True,  use_gate=True
  * Recompute-scratch : use_prev=False, use_delta=False, use_gate=False
  * Naive-Δ           : use_prev=True,  use_delta=True,  use_gate=False

Processor = min-plus (tropical) message passing, the canonical Bellman-Ford
alignment (Veličković et al., ICLR 2020) — used as a cited building block, not
claimed as a contribution. The novel piece is the Affected-Region Gate, whose
target is the LPA* work-set produced by the verified ground-truth engine.

Dense (B, n, n) tensors: simple and exact for the sizes we test; can be made
sparse later if n grows large.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import TYPE_ID


def collate(samples, scale=None, device="cpu", task_id=0):
    """Pack numpy samples sharing the same n into dense tensors.

    Distances are normalised by `scale`; unreachable (inf) -> scale (i.e. 1.0).
    `task_id` tags the batch for the generalist (0=sssp, 1=bottleneck, 2=mst).
    """
    n = samples[0]["n"]
    B = len(samples)
    if scale is None:
        scale = float(n * 10)  # ~ n * max_weight

    def z(*shape):
        return torch.zeros(*shape)

    W, A = z(B, n, n), z(B, n, n)
    g_prev, g_next = z(B, n), z(B, n)
    reach_next = z(B, n)
    pred_next = torch.zeros(B, n, dtype=torch.long)
    is_src, is_du, is_dv = z(B, n), z(B, n), z(B, n)
    delta_edge = z(B, n, n)
    delta_type = z(B, n, n, 3)
    affected = z(B, n)
    prev_tree = z(B, n, n)  # edges of the previous shortest-path tree (prev solution)

    for b, s in enumerate(samples):
        W[b] = torch.from_numpy(s["W"])
        A[b] = torch.from_numpy(s["A"])
        gp = np.where(np.isfinite(s["g_prev"]), s["g_prev"], scale)
        gn = np.where(np.isfinite(s["g_next"]), s["g_next"], scale)
        g_prev[b] = torch.from_numpy(gp)
        g_next[b] = torch.from_numpy(gn)
        reach_next[b] = torch.from_numpy(np.isfinite(s["g_next"]).astype(np.float64))
        pn = s["pred_next"].copy()
        pn[pn < 0] = np.where(pn < 0)[0]  # self-point where no predecessor (masked in loss)
        pred_next[b] = torch.from_numpy(pn)
        is_src[b, s["source"]] = 1.0
        for d in s["deltas"]:
            du, dv = d["u"], d["v"]
            is_du[b, du] = 1.0
            is_dv[b, dv] = 1.0
            delta_edge[b, du, dv] = 1.0
            delta_type[b, du, dv, TYPE_ID[d["type"]]] = 1.0
        affected[b] = torch.from_numpy(s["affected"])
        pp = s["pred_prev"]
        for v in range(n):
            if pp[v] >= 0:
                prev_tree[b, pp[v], v] = 1.0

    out = {
        "n": n, "scale": scale,
        "W": W, "A": A,
        "g_prev": (g_prev / scale).float(), "g_next": (g_next / scale).float(),
        "reach_next": reach_next.float(), "pred_next": pred_next,
        "is_src": is_src.float(), "is_du": is_du.float(), "is_dv": is_dv.float(),
        "delta_edge": delta_edge.float(), "delta_type": delta_type.float(),
        "prev_tree": prev_tree.float(), "affected": affected.float(),
        "task_id": torch.full((B,), task_id, dtype=torch.long),
    }
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}


class DeltaNAR(nn.Module):
    def __init__(self, hidden=128, steps=None, task="sssp",
                 use_prev=True, use_delta=True, use_gate=True, use_task_emb=False, copy_margin=0.0,
                 use_triplet=False, use_copy=True, use_minmax=True, use_attn=False, copy_scale=5.0,
                 use_copy_safe_gate=False):
        super().__init__()
        self.hidden = hidden
        self.steps = steps  # None -> use n at forward time
        self.task = task    # "sssp" (pointer + distance) or "mst" (edge membership)
        self.use_prev = use_prev
        self.use_delta = use_delta
        self.use_gate = use_gate
        self.use_task_emb = use_task_emb  # generalist: condition on task id
        self.task_emb = nn.Embedding(3, hidden)  # sssp / bottleneck / mst

        self.node_enc = nn.Linear(4, hidden)          # [g_prev, is_src, is_du, is_dv]
        self.edge_enc = nn.Linear(7, hidden)          # [W, A, delta_edge, prev_tree, delta_type(3)]
        self.msg = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.upd = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))  # h + [min,max]
        self.gate = nn.Linear(hidden, 1)              # per-step soft update gate
        self.dist_dec = nn.Linear(hidden, 1)
        self.affected_dec = nn.Linear(hidden, 1)
        self.copy_safe_dec = nn.Linear(hidden, 1)
        self.pred_score = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.edge_dec = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.copy_scale = copy_scale    # bias strength for copying the previous solution on unaffected nodes
        self.copy_margin = copy_margin  # conservative-copy margin; task-specific (SSSP 0, harder semirings >0)
        self.use_triplet = use_triplet  # CLRS-style triplet edge reasoning (for the strong from-scratch baseline)
        self.use_copy = use_copy        # ablation: turn off copy bias (keep gate/prev)
        self.use_minmax = use_minmax    # ablation: min-only vs min+max aggregation
        self.use_attn = use_attn        # baseline: attention aggregation instead of min/max
        self.use_copy_safe_gate = use_copy_safe_gate  # experimental: separate "safe to copy" gate
        self.last_copy_safe_logit = None
        self.attn = nn.Linear(hidden, 1)
        Ht = 8
        self.tri_qi = nn.Linear(hidden, Ht)
        self.tri_qj = nn.Linear(hidden, Ht)
        self.tri_qk = nn.Linear(hidden, Ht)
        self.tri_out = nn.Linear(Ht, hidden)

    def forward(self, batch):
        W, A = batch["W"], batch["A"]
        B, n, _ = W.shape
        H = self.hidden
        steps = self.steps or n

        g_prev = batch["g_prev"] if self.use_prev else torch.zeros_like(batch["g_prev"])
        is_du = batch["is_du"] if self.use_delta else torch.zeros_like(batch["is_du"])
        is_dv = batch["is_dv"] if self.use_delta else torch.zeros_like(batch["is_dv"])
        d_edge = batch["delta_edge"] if self.use_delta else torch.zeros_like(batch["delta_edge"])
        d_type = batch["delta_type"] if self.use_delta else torch.zeros_like(batch["delta_type"])
        p_tree = batch["prev_tree"] if self.use_prev else torch.zeros_like(batch["prev_tree"])

        node_feat = torch.stack([g_prev, batch["is_src"], is_du, is_dv], dim=-1)  # (B,n,4)
        edge_feat = torch.cat([W.unsqueeze(-1), A.unsqueeze(-1), d_edge.unsqueeze(-1),
                               p_tree.unsqueeze(-1), d_type], dim=-1)  # (B,n,n,7)

        h = self.node_enc(node_feat)        # (B,n,H)
        if self.use_task_emb:
            h = h + self.task_emb(batch["task_id"]).unsqueeze(1)  # condition on task
        he = self.edge_enc(edge_feat)       # (B,n,n,H)
        if self.use_triplet:  # triplet edge reasoning: edge(i,j) <- max_k psi(h_i,h_j,h_k)
            qi, qj, qk = self.tri_qi(h), self.tri_qj(h), self.tri_qk(h)  # (B,n,Ht)
            t = torch.relu(qi[:, :, None, None, :] + qj[:, None, :, None, :] + qk[:, None, None, :, :])
            he = he + self.tri_out(t.max(dim=3)[0])  # max over intermediate node k -> (B,n,n,H)

        in_mask = A > 0                          # edge u->v exists; (B,n,n)
        no_incoming = (in_mask.sum(dim=1) == 0)  # (B,n) vertex v has no in-edge
        nb = ~in_mask.unsqueeze(-1)

        for _ in range(steps):
            h_u = h.unsqueeze(2).expand(B, n, n, H)            # sender u, broadcast over v
            m = self.msg(torch.cat([h_u, he], dim=-1))         # message on edge u->v
            if self.use_attn:  # attention aggregation over senders (alternative strong baseline)
                sc = self.attn(m).squeeze(-1).masked_fill(~in_mask, -1e9)
                w = torch.softmax(sc, dim=1)
                a1 = (w.unsqueeze(-1) * m).sum(dim=1)
                a2 = a1
            else:
                a1 = m.masked_fill(nb, 1e9).min(dim=1)[0].masked_fill(no_incoming.unsqueeze(-1), 0.0)  # min-plus
                a2 = (m.masked_fill(nb, -1e9).max(dim=1)[0].masked_fill(no_incoming.unsqueeze(-1), 0.0)
                      if self.use_minmax else a1)  # max (off => min-only ablation)
            agg = torch.cat([a1, a2], dim=-1)            # (B,n,2H)
            h_new = self.upd(torch.cat([h, agg], dim=-1))
            if self.use_gate:
                gt = torch.sigmoid(self.gate(h_new))           # (B,n,1) localise update
                h = gt * h_new + (1 - gt) * h
            else:
                h = h_new

        affected_logit = self.affected_dec(h).squeeze(-1)      # (B,n) shared gate target
        copy_safe_logit = self.copy_safe_dec(h).squeeze(-1) if self.use_copy_safe_gate else None
        self.last_copy_safe_logit = copy_safe_logit
        h_v = h.unsqueeze(2).expand(B, n, n, H)
        h_u = h.unsqueeze(1).expand(B, n, n, H)

        if self.task == "sssp":
            g_pred = self.dist_dec(h).squeeze(-1)
            score = self.pred_score(torch.cat([h_v, h_u, he], dim=-1)).squeeze(-1)  # score[b,v,u]
            # "update, not recompute": unaffected vertices copy their previous predecessor.
            if self.use_copy and self.use_prev:
                margin = self.copy_margin
                if self.use_task_emb:  # generalist: per-task margin (bottleneck=task 1 needs conservative copy)
                    margin = (batch["task_id"] == 1).float().mul(2.0).view(-1, 1)
                if self.use_copy_safe_gate:
                    stay = torch.sigmoid(copy_safe_logit - margin).unsqueeze(-1)
                else:
                    stay = torch.sigmoid(-affected_logit - margin).unsqueeze(-1)  # conservative
                score = score + self.copy_scale * stay * batch["prev_tree"].transpose(1, 2)
            score = score.masked_fill(~(A.transpose(1, 2) > 0), -1e9)
            return g_pred, score, affected_logit

        # task == "mst": predict symmetric edge membership
        e = self.edge_dec(torch.cat([h_v, h_u, he], dim=-1)).squeeze(-1)  # (B,n,n)
        edge_logit = 0.5 * (e + e.transpose(1, 2))                        # symmetrise
        if self.use_copy and self.use_prev:
            stay = torch.sigmoid(-affected_logit - self.copy_margin)      # (B,n) conservative copy
            stay_e = stay.unsqueeze(2) * stay.unsqueeze(1)                # both endpoints unaffected
            edge_logit = edge_logit + self.copy_scale * stay_e * (2 * batch["prev_tree"] - 1)
        edge_logit = edge_logit.masked_fill(~(A > 0), -1e9)               # only existing edges
        return edge_logit, affected_logit


def compute_loss(batch, g_pred, score, affected_logit):
    reach = batch["reach_next"]
    not_src = 1.0 - batch["is_src"]
    # distance regression (all vertices; unreachable target = capped 1.0)
    dist_loss = F.mse_loss(g_pred, batch["g_next"])
    # predecessor pointer CE over reachable non-source vertices
    mask = (reach * not_src) > 0
    B, n, _ = score.shape
    if mask.any():
        pred_loss = F.cross_entropy(score[mask], batch["pred_next"][mask])
    else:
        pred_loss = score.new_tensor(0.0)
    # affected-region BCE (the gate target)
    pos_weight = affected_logit.new_tensor(10.0)  # affected vertices are sparse (~8%)
    aff_loss = F.binary_cross_entropy_with_logits(affected_logit, batch["affected"], pos_weight=pos_weight)
    return dist_loss + pred_loss + aff_loss, {
        "dist": dist_loss.item(), "pred": pred_loss.item(), "aff": aff_loss.item(),
    }


@torch.no_grad()
def evaluate(batch, g_pred, score, affected_logit, semiring="min_plus"):
    """Predecessor accuracy (exact match) and tie-tolerant validity.

    'exact'  : predicted predecessor equals the ground-truth pointer.
    'valid'  : predicted predecessor yields a correct shortest distance
               (dist[u] + w(u,v) == dist[v]) — fair under shortest-path ties.
    """
    reach = batch["reach_next"]
    not_src = 1.0 - batch["is_src"]
    mask = (reach * not_src) > 0                          # (B,n)
    pred_hat = score.argmax(dim=-1)                       # (B,n)
    B, n, _ = score.shape

    # exact-match accuracy
    node_acc = (pred_hat[mask] == batch["pred_next"][mask]).float().mean().item() if mask.any() else 1.0
    graph_acc = ((pred_hat == batch["pred_next"]) | (~mask)).all(dim=1).float().mean().item()

    # tie-tolerant validity using ground-truth distances
    dist_true = batch["g_next"] * batch["scale"]          # (B,n) de-normalised
    w_uv = torch.gather(batch["W"], 1, pred_hat.unsqueeze(1).expand(B, 1, n)).squeeze(1)
    dist_u = torch.gather(dist_true, 1, pred_hat)
    combined = torch.minimum(dist_u, w_uv) if semiring == "max_min" else dist_u + w_uv
    valid = ((combined - dist_true).abs() < 0.5) | (~mask)
    node_acc_v = valid[mask].float().mean().item() if mask.any() else 1.0
    graph_acc_v = valid.all(dim=1).float().mean().item()

    aff_f1 = _f1((affected_logit > 0).float(), batch["affected"])
    return {
        "node_acc": node_acc, "graph_acc": graph_acc,
        "node_acc_v": node_acc_v, "graph_acc_v": graph_acc_v, "aff_f1": aff_f1,
    }


def _f1(pred, tgt):
    tp = (pred * tgt).sum().item()
    fp = (pred * (1 - tgt)).sum().item()
    fn = ((1 - pred) * tgt).sum().item()
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0


# --- MST task (edge-membership) ------------------------------------------

def collate_mst(samples, device="cpu"):
    """Pack incremental-MST samples; reuses the shared input keys (prev_tree
    holds the previous MST membership; is_src marks the tree root)."""
    n = samples[0]["n"]
    B = len(samples)

    def z(*shape):
        return torch.zeros(*shape)

    W, A = z(B, n, n), z(B, n, n)
    prev_tree = z(B, n, n)          # previous MST membership (symmetric)
    mst_next = z(B, n, n)           # target MST membership
    is_src, is_du, is_dv = z(B, n), z(B, n), z(B, n)
    delta_edge = z(B, n, n)
    delta_type = z(B, n, n, 3)
    affected = z(B, n)

    for b, s in enumerate(samples):
        W[b] = torch.from_numpy(s["W"])
        A[b] = torch.from_numpy(s["A"])
        prev_tree[b] = torch.from_numpy(s["prev_mst"])
        mst_next[b] = torch.from_numpy(s["mst_next"])
        affected[b] = torch.from_numpy(s["affected"])
        is_src[b, s["root"]] = 1.0
        for d in s["deltas"]:
            du, dv = d["u"], d["v"]
            is_du[b, du] = 1.0
            is_dv[b, dv] = 1.0
            for (a, c) in ((du, dv), (dv, du)):  # undirected: both directions
                delta_edge[b, a, c] = 1.0
                delta_type[b, a, c, TYPE_ID[d["type"]]] = 1.0

    out = {
        "n": n, "scale": float(n),
        "W": W, "A": A, "prev_tree": prev_tree.float(), "mst_next": mst_next.float(),
        "g_prev": z(B, n),  # placeholder (MST has no distance feature)
        "is_src": is_src.float(), "is_du": is_du.float(), "is_dv": is_dv.float(),
        "delta_edge": delta_edge.float(), "delta_type": delta_type.float(),
        "affected": affected.float(),
    }
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}


def compute_loss_mst(batch, edge_logit, affected_logit):
    mask = batch["A"] > 0
    edge_loss = F.binary_cross_entropy_with_logits(edge_logit[mask], batch["mst_next"][mask])
    pos_weight = affected_logit.new_tensor(10.0)
    aff_loss = F.binary_cross_entropy_with_logits(affected_logit, batch["affected"], pos_weight=pos_weight)
    return edge_loss + aff_loss, {"edge": edge_loss.item(), "aff": aff_loss.item()}


@torch.no_grad()
def evaluate_mst(batch, edge_logit, affected_logit):
    """Edge-level accuracy and whole-MST (graph) accuracy on existing edges."""
    A = batch["A"]
    B = A.shape[0]
    triu = torch.triu(torch.ones_like(A), diagonal=1) > 0  # count each undirected edge once
    em = (A > 0) & triu
    pred = edge_logit > 0
    tgt = batch["mst_next"] > 0
    edge_acc = (pred[em] == tgt[em]).float().mean().item() if em.any() else 1.0
    correct = (pred == tgt) | (~em)
    graph_acc = correct.view(B, -1).all(dim=1).float().mean().item()
    aff_f1 = _f1((affected_logit > 0).float(), batch["affected"])
    return {"edge_acc": edge_acc, "graph_acc": graph_acc, "aff_f1": aff_f1}
