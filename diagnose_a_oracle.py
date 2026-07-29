"""Oracle-mask local diagnostics for DeltaNAR.

The goal is to separate three evaluation questions that aggregate metrics may conflate:

1. How good is the learned recompute branch without copy bias?
2. How much does the learned affected gate help the copy decoder?
3. What is the upper bound if the copy/recompute decision were oracle-perfect?

Small local runs only; server grids should reuse the same variant definitions.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from diagnose_a_metrics import diagnostic_metrics
from experiments import CONFIGS, TASK_CFG, chunks, set_seed
from src.model import DeltaNAR, collate, compute_loss


def valid_previous_copy(batch, semiring):
    mask = (batch["reach_next"] * (1.0 - batch["is_src"])) > 0
    B, n, _ = batch["W"].shape
    prev_pred = batch["prev_tree"].argmax(dim=1)
    has_prev = batch["prev_tree"].sum(dim=1) > 0
    prev_pred = torch.where(has_prev, prev_pred, torch.arange(n, device=batch["W"].device).view(1, n))

    dist_true = batch["g_next"] * batch["scale"]
    prev_w = torch.gather(batch["W"], 1, prev_pred.unsqueeze(1).expand(B, 1, n)).squeeze(1)
    prev_dist_u = torch.gather(dist_true, 1, prev_pred)
    prev_combined = torch.minimum(prev_dist_u, prev_w) if semiring == "max_min" else prev_dist_u + prev_w
    return (((prev_combined - dist_true).abs() < 0.5) | (~mask))


def apply_copy(raw_score, batch, stay, copy_scale):
    return raw_score + copy_scale * stay.unsqueeze(-1) * batch["prev_tree"].transpose(1, 2)


def pure_copy_score(raw_score, batch):
    score = raw_score.new_full(raw_score.shape, -1e9)
    score = score + 2e9 * batch["prev_tree"].transpose(1, 2)
    return score.masked_fill(~(batch["A"].transpose(1, 2) > 0), -1e9)


@torch.no_grad()
def eval_variants(model, mk, sem, n, args, device):
    model.eval()
    samples = mk(args.test_episodes, n, n_steps=5, seed0=900_000,
                 topology=args.test_topology, heavy_tail=args.test_heavy_tail,
                 weight_range=(1, args.test_weight_hi), kinds=args.test_delta_kinds,
                 delta_size=args.test_delta_size)
    variant_tot, cnt = {}, 0
    old_use_copy = model.use_copy
    model.use_copy = False
    try:
        for ch in chunks(samples, args.eval_batch):
            b = collate(ch, device=device)
            g_pred, raw_score, affected_logit = model(b)

            pred_stay = torch.sigmoid(-affected_logit - model.copy_margin)
            oracle_affected_stay = (b["affected"] <= 0.5).float()
            oracle_copy_safe_stay = valid_previous_copy(b, sem).float()

            variants = {
                "raw_no_copy": raw_score,
                "predicted_copy": apply_copy(raw_score, b, pred_stay, model.copy_scale),
                "oracle_affected_copy": apply_copy(raw_score, b, oracle_affected_stay, model.copy_scale),
                "oracle_copy_safe_upper": apply_copy(raw_score, b, oracle_copy_safe_stay, model.copy_scale),
                "pure_copy": pure_copy_score(raw_score, b),
            }
            for name, score in variants.items():
                metrics = diagnostic_metrics(b, g_pred, score, affected_logit, semiring=sem)
                variant_tot.setdefault(name, {k: 0.0 for k in metrics})
                for k, v in metrics.items():
                    variant_tot[name][k] += v * len(ch)
            cnt += len(ch)
    finally:
        model.use_copy = old_use_copy
    return {name: {k: v / cnt for k, v in metrics.items()} for name, metrics in variant_tot.items()}


def train_model(task, seed, args, device):
    mk, _ = TASK_CFG[task]
    set_seed(seed)
    copy_margin = 2.0 if task == "bottleneck" else 0.0
    cfg = dict(CONFIGS["DeltaNAR"])
    model = DeltaNAR(hidden=args.hidden, task="sssp", steps=(args.steps or None),
                     copy_margin=copy_margin, **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train = mk(args.train_episodes, args.train_n, n_steps=5, seed0=1 + seed * 7919,
               delta_size=args.train_delta_size, kinds=args.train_delta_kinds)
    for _ in range(args.epochs):
        model.train()
        random.shuffle(train)
        for ch in chunks(train, args.batch):
            b = collate(ch, device=device)
            loss, _ = compute_loss(b, *model(b))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def aggregate(raw):
    agg = {}
    for task, by_variant in raw.items():
        agg[task] = {}
        for variant, by_size in by_variant.items():
            agg[task][variant] = {}
            for n, by_metric in by_size.items():
                agg[task][variant][n] = {
                    k: [float(np.mean(v)), float(np.std(v))]
                    for k, v in by_metric.items()
                }
    return agg


def write_markdown(path, args, agg):
    metrics = [
        "node_acc_v", "graph_acc_v", "affected_node_acc_v", "changed_output_acc_v",
        "copy_critical_acc_v", "pure_copy_node_acc_v", "pure_copy_changed_acc_v",
        "gate_recall", "copy_critical_gate_recall", "changed_output_fraction",
        "copy_critical_fraction",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# A-line Oracle-Mask Diagnostics\n\n")
        f.write("Small local DeltaNAR-only diagnostic. Variants reuse the same trained model.\n\n")
        f.write("```json\n")
        f.write(json.dumps(vars(args), indent=2))
        f.write("\n```\n\n")
        for task, by_variant in agg.items():
            f.write(f"## {task}\n\n")
            for metric in metrics:
                sizes = [str(n) for n in args.sizes]
                f.write(f"### {metric}\n\n")
                f.write("| Variant | " + " | ".join(f"n={n}" for n in sizes) + " |\n")
                f.write("|---|" + "|".join("---:" for _ in sizes) + "|\n")
                for variant, by_size in by_variant.items():
                    cells = []
                    for n in sizes:
                        mean, std = by_size[n][metric]
                        cells.append(f"{mean:.3f} +/- {std:.3f}")
                    f.write(f"| {variant} | " + " | ".join(cells) + " |\n")
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
    ap.add_argument("--out", default="results/A_oracle_diagnostics.json")
    ap.add_argument("--markdown", default="RESULTS_A_ORACLE.md")
    ap.add_argument("--test_topology", default="er")
    ap.add_argument("--test_heavy_tail", action="store_true")
    ap.add_argument("--test_weight_hi", type=int, default=10)
    ap.add_argument("--test_delta_kinds", nargs="+", default=None)
    ap.add_argument("--train_delta_kinds", nargs="+", default=None)
    ap.add_argument("--train_delta_size", type=int, default=1)
    ap.add_argument("--test_delta_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} tasks={args.tasks} seeds={args.seeds} sizes={args.sizes}", flush=True)
    raw = {}
    for task in args.tasks:
        raw[task] = {}
        mk, sem = TASK_CFG[task]
        for seed in args.seeds:
            t0 = time.time()
            model = train_model(task, seed, args, device)
            for n in args.sizes:
                res = eval_variants(model, mk, sem, n, args, device)
                for variant, metrics in res.items():
                    raw[task].setdefault(variant, {}).setdefault(str(n), {})
                    for k, v in metrics.items():
                        raw[task][variant][str(n)].setdefault(k, []).append(v)
            print(f"[{task}/seed{seed}] {time.time() - t0:.0f}s", flush=True)

    agg = aggregate(raw)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "agg": agg, "raw": raw}, f, indent=2)
    write_markdown(args.markdown, args, agg)
    print(f"saved -> {args.out}", flush=True)
    print(f"saved -> {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
