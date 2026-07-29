"""Repair-focused local diagnostics for the AHR stress setting.

This script intentionally does not replace the main experiment entry points.
It adds optional losses that overweight changed-output vertices and train the
affected gate against copy-unsafe vertices, then evaluates with the same
diagnostic metrics used by the A-line audits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_a_metrics import diagnostic_metrics, eval_at
from experiments import CONFIGS, TASK_CFG, chunks, set_seed
from src.model import DeltaNAR, collate, compute_loss


def mask_mean_loss(loss_vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return loss_vec[mask].mean()
    return loss_vec.new_tensor(0.0)


@torch.no_grad()
def prev_copy_valid(batch: dict, semiring: str) -> torch.Tensor:
    reach = batch["reach_next"]
    not_src = 1.0 - batch["is_src"]
    mask = (reach * not_src) > 0
    B, n, _ = batch["W"].shape
    prev_pred = batch["prev_tree"].argmax(dim=1)
    has_prev = batch["prev_tree"].sum(dim=1) > 0
    prev_pred = torch.where(has_prev, prev_pred, torch.arange(n, device=batch["W"].device).view(1, n))

    dist_true = batch["g_next"] * batch["scale"]
    prev_w = torch.gather(batch["W"], 1, prev_pred.unsqueeze(1).expand(B, 1, n)).squeeze(1)
    prev_dist_u = torch.gather(dist_true, 1, prev_pred)
    combined = torch.minimum(prev_dist_u, prev_w) if semiring == "max_min" else prev_dist_u + prev_w
    return (((combined - dist_true).abs() < 0.5) | (~mask))


def hard_masks(batch: dict, semiring: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reach = batch["reach_next"]
    not_src = 1.0 - batch["is_src"]
    mask = (reach * not_src) > 0

    prev_pred = batch["prev_tree"].argmax(dim=1)
    has_prev = batch["prev_tree"].sum(dim=1) > 0
    n = batch["W"].shape[1]
    prev_pred = torch.where(has_prev, prev_pred, torch.arange(n, device=batch["W"].device).view(1, n))

    prev_g = batch["g_prev"] * batch["scale"]
    next_g = batch["g_next"] * batch["scale"]
    changed_value = (prev_g - next_g).abs() >= 0.5
    changed_pred = prev_pred != batch["pred_next"]
    changed_output = mask & (changed_value | changed_pred)
    copy_critical = mask & ~prev_copy_valid(batch, semiring)
    return mask, changed_output, copy_critical


def masked_bce_with_logits(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        loss = F.binary_cross_entropy_with_logits(logit, target.float(), reduction="none")
        return loss[mask].mean()
    return logit.new_tensor(0.0)


def compute_repair_loss(batch: dict, g_pred, score, affected_logit, semiring: str, args,
                        copy_safe_logit: torch.Tensor | None = None):
    base_loss, parts = compute_loss(batch, g_pred, score, affected_logit)
    mask, changed_output, copy_critical = hard_masks(batch, semiring)
    B, n, _ = score.shape
    flat_score = score.reshape(B * n, n)
    flat_target = batch["pred_next"].reshape(B * n)
    ce = F.cross_entropy(flat_score, flat_target, reduction="none").reshape(B, n)

    repair_loss = mask_mean_loss(ce, changed_output)
    critical_loss = mask_mean_loss(ce, copy_critical)

    copy_safe_loss = affected_logit.new_tensor(0.0)
    if args.copy_safe_weight:
        if args.use_copy_safe_gate and copy_safe_logit is not None:
            copy_safe_target = (mask & ~copy_critical).float()
            copy_safe_loss = masked_bce_with_logits(copy_safe_logit, copy_safe_target, mask)
        else:
            copy_unsafe = copy_critical.float()
            pos_weight = affected_logit.new_tensor(args.copy_safe_pos_weight)
            copy_safe_loss = F.binary_cross_entropy_with_logits(
                affected_logit, copy_unsafe, pos_weight=pos_weight
            )

    loss = (
        base_loss
        + args.repair_weight * repair_loss
        + args.critical_weight * critical_loss
        + args.copy_safe_weight * copy_safe_loss
    )
    parts.update(
        {
            "repair": float(repair_loss.detach().cpu()),
            "critical": float(critical_loss.detach().cpu()),
            "copy_safe": float(copy_safe_loss.detach().cpu()),
        }
    )
    return loss, parts


def train_one(task: str, seed: int, args, device: str):
    mk, sem = TASK_CFG[task]
    set_seed(seed)
    copy_margin = args.copy_margin if args.copy_margin is not None else (2.0 if task == "bottleneck" else 0.0)
    cfg = dict(CONFIGS["DeltaNAR"])
    model = DeltaNAR(
        hidden=args.hidden,
        task="sssp",
        steps=(args.steps or None),
        copy_margin=copy_margin,
        copy_scale=args.copy_scale,
        use_copy_safe_gate=args.use_copy_safe_gate,
        **cfg,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train = mk(
        args.train_episodes,
        args.train_n,
        n_steps=5,
        seed0=1 + seed * 7919,
        delta_size=args.train_delta_size,
        kinds=args.train_delta_kinds,
    )
    for _ in range(args.epochs):
        model.train()
        random.shuffle(train)
        for ch in chunks(train, args.batch):
            b = collate(ch, device=device)
            outputs = model(b)
            loss, _ = compute_repair_loss(
                b, *outputs, semiring=sem, args=args, copy_safe_logit=model.last_copy_safe_logit
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {
        str(n): eval_at(
            model,
            mk,
            sem,
            n,
            args.test_episodes,
            device,
            args.eval_batch,
            topology=args.test_topology,
            heavy_tail=args.test_heavy_tail,
            whi=args.test_weight_hi,
            kinds=args.test_delta_kinds,
            delta_size=args.test_delta_size,
        )
        for n in args.sizes
    }


def aggregate(raw: dict) -> dict:
    agg = {}
    for task, by_size in raw.items():
        agg[task] = {"DeltaNAR+repair": {}}
        for n, by_metric in by_size.items():
            agg[task]["DeltaNAR+repair"][n] = {
                k: [float(np.mean(v)), float(np.std(v))]
                for k, v in by_metric.items()
            }
    return agg


def write_markdown(path: str, args, agg: dict) -> None:
    metrics = [
        "node_acc_v",
        "graph_acc_v",
        "changed_output_acc_v",
        "copy_critical_acc_v",
        "pure_copy_changed_acc_v",
        "gate_recall",
        "copy_critical_gate_recall",
        "changed_output_fraction",
        "copy_critical_fraction",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Repair-Focused AHR Diagnostic\n\n")
        f.write("Experimental local run; not a headline result unless repeated.\n\n")
        f.write("```json\n")
        f.write(json.dumps(vars(args), indent=2))
        f.write("\n```\n\n")
        for task, by_model in agg.items():
            f.write(f"## {task}\n\n")
            sizes = [str(n) for n in args.sizes]
            for metric in metrics:
                f.write(f"### {metric}\n\n")
                f.write("| Model | " + " | ".join(f"n={n}" for n in sizes) + " |\n")
                f.write("|---|" + "|".join("---:" for _ in sizes) + "|\n")
                for model_name, by_size in by_model.items():
                    cells = []
                    for n in sizes:
                        mean, std = by_size[n][metric]
                        cells.append(f"{mean:.3f} +/- {std:.3f}")
                    f.write(f"| {model_name} | " + " | ".join(cells) + " |\n")
                f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sssp"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--train_n", type=int, default=16)
    ap.add_argument("--train_episodes", type=int, default=120)
    ap.add_argument("--test_episodes", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/AHR_repair_loss.json")
    ap.add_argument("--markdown", default="RESULTS_AHR_REPAIR_LOSS.md")
    ap.add_argument("--test_topology", default="er")
    ap.add_argument("--test_heavy_tail", action="store_true")
    ap.add_argument("--test_weight_hi", type=int, default=10)
    ap.add_argument("--test_delta_kinds", nargs="+", default=None)
    ap.add_argument("--train_delta_kinds", nargs="+", default=None)
    ap.add_argument("--train_delta_size", type=int, default=4)
    ap.add_argument("--test_delta_size", type=int, default=12)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--repair_weight", type=float, default=1.0)
    ap.add_argument("--critical_weight", type=float, default=0.0)
    ap.add_argument("--copy_safe_weight", type=float, default=0.0)
    ap.add_argument("--copy_safe_pos_weight", type=float, default=20.0)
    ap.add_argument("--use_copy_safe_gate", action="store_true")
    ap.add_argument("--copy_margin", type=float, default=None)
    ap.add_argument("--copy_scale", type=float, default=5.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} tasks={args.tasks} seeds={args.seeds} sizes={args.sizes}", flush=True)
    raw = {}
    for task in args.tasks:
        raw[task] = {str(n): {} for n in args.sizes}
        for seed in args.seeds:
            t0 = time.time()
            res = train_one(task, seed, args, device)
            for n, metrics in res.items():
                for k, v in metrics.items():
                    raw[task][n].setdefault(k, []).append(v)
            print(f"[{task}/DeltaNAR+repair/seed{seed}] {time.time() - t0:.0f}s", flush=True)

    agg = aggregate(raw)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "agg": agg, "raw": raw}, f, indent=2)
    write_markdown(args.markdown, args, agg)
    print(f"saved -> {args.out}", flush=True)
    print(f"saved -> {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
