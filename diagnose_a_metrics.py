"""Local A-line diagnostics for targeted metric validation.

This script trains small local runs and reports slices that the main experiment
grid does not currently expose: affected-only validity, changed-output validity,
copy-critical validity, pure-copy oracle quality, and gate recall on the regions
where copying is unsafe.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

from experiments import CONFIGS, TASK_CFG, chunks, set_seed
from src.model import DeltaNAR, collate, compute_loss, evaluate


def mean_or_one(x):
    # Return the vacuous-success value only for an actually empty slice.
    # ``x.any()`` is false for a nonempty all-false slice and would therefore
    # incorrectly report accuracy 1.0 exactly when every prediction fails.
    return x.float().mean().item() if x.numel() else 1.0


def safe_div(num, den):
    return float(num / den) if den else 1.0


def gate_counts(pred, target):
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    fn_rate = safe_div(fn, tp + fn)
    return precision, recall, f1, fn_rate, tp, fp, fn, tn


@torch.no_grad()
def diagnostic_metrics(batch, g_pred, score, affected_logit, semiring="min_plus"):
    base = evaluate(batch, g_pred, score, affected_logit, semiring=semiring)

    reach = batch["reach_next"]
    not_src = 1.0 - batch["is_src"]
    mask = (reach * not_src) > 0
    pred_hat = score.argmax(dim=-1)
    B, n, _ = score.shape

    dist_true = batch["g_next"] * batch["scale"]
    w_uv = torch.gather(batch["W"], 1, pred_hat.unsqueeze(1).expand(B, 1, n)).squeeze(1)
    dist_u = torch.gather(dist_true, 1, pred_hat)
    combined = torch.minimum(dist_u, w_uv) if semiring == "max_min" else dist_u + w_uv
    valid = (((combined - dist_true).abs() < 0.5) | (~mask))

    prev_pred = batch["prev_tree"].argmax(dim=1)
    has_prev = batch["prev_tree"].sum(dim=1) > 0
    prev_pred = torch.where(has_prev, prev_pred, torch.arange(n, device=score.device).view(1, n))
    prev_w = torch.gather(batch["W"], 1, prev_pred.unsqueeze(1).expand(B, 1, n)).squeeze(1)
    prev_dist_u = torch.gather(dist_true, 1, prev_pred)
    prev_combined = torch.minimum(prev_dist_u, prev_w) if semiring == "max_min" else prev_dist_u + prev_w
    prev_valid = (((prev_combined - dist_true).abs() < 0.5) | (~mask))

    prev_g = batch["g_prev"] * batch["scale"]
    next_g = batch["g_next"] * batch["scale"]
    changed_value = (prev_g - next_g).abs() >= 0.5
    changed_pred = prev_pred != batch["pred_next"]
    changed_output = mask & (changed_value | changed_pred)
    affected = mask & (batch["affected"] > 0.5)
    unaffected = mask & ~affected
    copy_critical = mask & ~prev_valid
    copy_safe = mask & prev_valid

    gate_pred = affected_logit > 0
    affected_precision, affected_recall, affected_f1, affected_fn_rate, *_ = gate_counts(
        gate_pred[mask], affected[mask]
    )
    cc_precision, cc_recall, cc_f1, cc_fn_rate, *_ = gate_counts(
        gate_pred[mask], copy_critical[mask]
    )

    out = dict(base)
    out.update({
        "affected_node_acc_v": mean_or_one(valid[affected]),
        "unaffected_node_acc_v": mean_or_one(valid[unaffected]),
        "changed_output_acc_v": mean_or_one(valid[changed_output]),
        "unchanged_output_acc_v": mean_or_one(valid[mask & ~changed_output]),
        "copy_critical_acc_v": mean_or_one(valid[copy_critical]),
        "copy_safe_acc_v": mean_or_one(valid[copy_safe]),
        "pure_copy_node_acc_v": mean_or_one(prev_valid[mask]),
        "pure_copy_changed_acc_v": mean_or_one(prev_valid[changed_output]),
        "pure_copy_critical_acc_v": mean_or_one(prev_valid[copy_critical]),
        "affected_fraction": affected.float().sum().item() / max(mask.float().sum().item(), 1.0),
        "changed_output_fraction": changed_output.float().sum().item() / max(mask.float().sum().item(), 1.0),
        "copy_critical_fraction": copy_critical.float().sum().item() / max(mask.float().sum().item(), 1.0),
        "gate_precision": affected_precision,
        "gate_recall": affected_recall,
        "gate_f1_diag": affected_f1,
        "gate_fn_rate": affected_fn_rate,
        "copy_critical_gate_precision": cc_precision,
        "copy_critical_gate_recall": cc_recall,
        "copy_critical_gate_f1": cc_f1,
        "copy_critical_gate_fn_rate": cc_fn_rate,
        "pred_unaffected_rate": ((~gate_pred) & mask).float().sum().item() / max(mask.float().sum().item(), 1.0),
    })
    return out


@torch.no_grad()
def eval_at(model, mk, sem, n, n_ep, device, batch_size, topology="er", heavy_tail=False,
            whi=10, kinds=None, delta_size=1):
    model.eval()
    samples = mk(n_ep, n, n_steps=5, seed0=900_000,
                 topology=topology, heavy_tail=heavy_tail, weight_range=(1, whi),
                 kinds=kinds, delta_size=delta_size)
    tot, cnt = None, 0
    for ch in chunks(samples, batch_size):
        b = collate(ch, device=device)
        metrics = diagnostic_metrics(b, *model(b), semiring=sem)
        if tot is None:
            tot = {k: 0.0 for k in metrics}
        for k, v in metrics.items():
            tot[k] += v * len(ch)
        cnt += len(ch)
    return {k: tot[k] / cnt for k in tot}


def train_one(task, model_name, seed, args, device):
    mk, sem = TASK_CFG[task]
    cfg = CONFIGS[model_name]
    set_seed(seed)
    copy_margin = 2.0 if task == "bottleneck" else 0.0
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
    return {
        str(n): eval_at(model, mk, sem, n, args.test_episodes, device, args.eval_batch,
                        topology=args.test_topology, heavy_tail=args.test_heavy_tail,
                        whi=args.test_weight_hi, kinds=args.test_delta_kinds,
                        delta_size=args.test_delta_size)
        for n in args.sizes
    }


def aggregate(raw):
    agg = {}
    for task, by_model in raw.items():
        agg[task] = {}
        for model_name, by_size in by_model.items():
            agg[task][model_name] = {}
            for n, by_metric in by_size.items():
                agg[task][model_name][n] = {
                    k: [float(np.mean(v)), float(np.std(v))]
                    for k, v in by_metric.items()
                }
    return agg


def write_markdown(path, args, agg):
    metrics = [
        "node_acc_v", "graph_acc_v", "affected_node_acc_v", "changed_output_acc_v",
        "copy_critical_acc_v", "pure_copy_node_acc_v", "pure_copy_changed_acc_v",
        "gate_recall", "copy_critical_gate_recall", "affected_fraction",
        "changed_output_fraction", "copy_critical_fraction",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# A-line Local Diagnostic Metrics\n\n")
        f.write("Small local diagnostic run for targeted metric validation. ")
        f.write("Do not treat these as final server-scale paper numbers.\n\n")
        f.write("```json\n")
        f.write(json.dumps(vars(args), indent=2))
        f.write("\n```\n\n")
        for task, by_model in agg.items():
            f.write(f"## {task}\n\n")
            for metric in metrics:
                f.write(f"### {metric}\n\n")
                sizes = [str(n) for n in args.sizes]
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
    ap.add_argument("--models", nargs="+", default=["DeltaNAR", "no-copy", "Naive-d", "Recompute"])
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
    ap.add_argument("--out", default="results/A_local_diagnostics.json")
    ap.add_argument("--markdown", default="RESULTS_AUDIT.md")
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
    print(f"device={device} tasks={args.tasks} models={args.models} seeds={args.seeds} sizes={args.sizes}", flush=True)
    raw = {}
    for task in args.tasks:
        raw[task] = {}
        for model_name in args.models:
            raw[task][model_name] = {str(n): {} for n in args.sizes}
            for seed in args.seeds:
                t0 = time.time()
                res = train_one(task, model_name, seed, args, device)
                for n, metrics in res.items():
                    for k, v in metrics.items():
                        raw[task][model_name][n].setdefault(k, []).append(v)
                print(f"[{task}/{model_name}/seed{seed}] {time.time() - t0:.0f}s", flush=True)

    agg = aggregate(raw)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "agg": agg, "raw": raw}, f, indent=2)
    write_markdown(args.markdown, args, agg)
    print(f"saved -> {args.out}", flush=True)
    print(f"saved -> {args.markdown}", flush=True)


if __name__ == "__main__":
    main()
