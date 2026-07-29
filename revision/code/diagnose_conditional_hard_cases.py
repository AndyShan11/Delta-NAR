"""Train unchanged models and evaluate only edits that invalidate old parents."""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from diagnose_a_metrics import (
    CONFIGS,
    TASK_CFG,
    DeltaNAR,
    chunks,
    collate,
    compute_loss,
    diagnostic_metrics,
    set_seed,
)


def critical_count(sample: dict, semiring: str) -> int:
    n = int(sample["n"])
    source = int(sample["source"])
    distance = sample["g_next"]
    previous = sample["pred_prev"]
    weights = sample["W"]
    adjacency = sample["A"] > 0
    count = 0
    for vertex in range(n):
        if vertex == source or not np.isfinite(distance[vertex]):
            continue
        parent = int(previous[vertex])
        if parent < 0 or not adjacency[parent, vertex]:
            count += 1
            continue
        combined = (
            min(distance[parent], weights[parent, vertex])
            if semiring == "max_min"
            else distance[parent] + weights[parent, vertex]
        )
        if not np.isfinite(combined) or abs(combined - distance[vertex]) >= 0.5:
            count += 1
    return count


def build_conditioned_samples(
    maker,
    semiring: str,
    n: int,
    target: int,
    seed0: int,
    delta_size: int,
    min_critical: int,
    max_candidates: int,
) -> tuple[list[dict], dict]:
    accepted: list[dict] = []
    candidates = 0
    episode_seed = seed0
    while len(accepted) < target and candidates < max_candidates:
        batch = maker(
            20,
            n,
            n_steps=5,
            seed0=episode_seed,
            delta_size=delta_size,
            weight_range=(1, 10),
        )
        episode_seed += 20
        for sample in batch:
            candidates += 1
            count = critical_count(sample, semiring)
            if count >= min_critical:
                sample = dict(sample)
                sample["_critical_count"] = count
                accepted.append(sample)
                if len(accepted) >= target:
                    break
            if candidates >= max_candidates:
                break
    if len(accepted) < target:
        raise RuntimeError(
            f"only {len(accepted)} accepted of {candidates} candidates "
            f"for n={n}, delta_size={delta_size}"
        )
    return accepted, {
        "accepted": len(accepted),
        "candidates": candidates,
        "acceptance_rate": len(accepted) / candidates,
        "mean_critical_nodes": float(
            np.mean([sample["_critical_count"] for sample in accepted])
        ),
    }


@torch.no_grad()
def evaluate_samples(model, samples, semiring, device, batch_size):
    model.eval()
    totals = None
    count = 0
    for chunk in chunks(samples, batch_size):
        batch = collate(chunk, device=device)
        metrics = diagnostic_metrics(batch, *model(batch), semiring=semiring)
        if totals is None:
            totals = {key: 0.0 for key in metrics}
        for key, value in metrics.items():
            totals[key] += float(value) * len(chunk)
        count += len(chunk)
    return {key: value / count for key, value in totals.items()}


def train_model(task, model_name, seed, args, device):
    maker, _ = TASK_CFG[task]
    set_seed(seed)
    model = DeltaNAR(
        hidden=args.hidden,
        task="sssp",
        steps=(args.steps or None),
        copy_margin=2.0 if task == "bottleneck" else 0.0,
        **CONFIGS[model_name],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    training = maker(
        args.train_episodes,
        args.train_n,
        n_steps=5,
        seed0=1 + seed * 7919,
        delta_size=args.train_delta_size,
    )
    for _ in range(args.epochs):
        model.train()
        random.shuffle(training)
        for chunk in chunks(training, args.batch):
            batch = collate(chunk, device=device)
            loss, _ = compute_loss(batch, *model(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.factorized_msg = True
    return model


def aggregate(raw):
    output = {}
    for task, by_model in raw.items():
        output[task] = {}
        for model, by_size in by_model.items():
            output[task][model] = {}
            for size, by_metric in by_size.items():
                output[task][model][size] = {
                    metric: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "n": len(values),
                    }
                    for metric, values in by_metric.items()
                }
    return output


def write_markdown(path, args, aggregate_result, sampling):
    metrics = (
        "node_acc_v",
        "graph_acc_v",
        "affected_node_acc_v",
        "changed_output_acc_v",
        "copy_critical_acc_v",
        "pure_copy_node_acc_v",
        "pure_copy_changed_acc_v",
        "pure_copy_critical_acc_v",
        "gate_recall",
        "copy_critical_gate_recall",
        "copy_critical_fraction",
    )
    lines = [
        "# Conditional Copy-Failure Evaluation",
        "",
        "Every evaluated graph contains at least the requested number of nodes for "
        "which the previous predecessor is invalid on the post-edit graph.",
        "",
        "```json",
        json.dumps(vars(args), indent=2),
        "```",
        "",
        "## Sampling",
        "",
        "| Task | n | Accepted | Candidates | Acceptance | Mean critical nodes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, by_size in sampling.items():
        for size, row in by_size.items():
            lines.append(
                f"| {task} | {size} | {row['accepted']} | {row['candidates']} "
                f"| {row['acceptance_rate']:.4f} | {row['mean_critical_nodes']:.2f} |"
            )
    for task, by_model in aggregate_result.items():
        lines += ["", f"## {task}", ""]
        for metric in metrics:
            lines += [
                f"### {metric}",
                "",
                "| Model | "
                + " | ".join(f"n={size}" for size in map(str, args.sizes))
                + " |",
                "|---|" + "|".join("---:" for _ in args.sizes) + "|",
            ]
            for model, by_size in by_model.items():
                values = []
                for size in map(str, args.sizes):
                    row = by_size[size][metric]
                    values.append(f"{row['mean']:.4f} +/- {row['std']:.4f}")
                lines.append(f"| {model} | " + " | ".join(values) + " |")
            lines.append("")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["sssp"])
    parser.add_argument(
        "--models", nargs="+", default=["DeltaNAR", "Naive-d", "Recompute"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--target-samples", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--min-critical", type=int, default=1)
    parser.add_argument("--test-delta-size", type=int, default=12)
    parser.add_argument("--train-delta-size", type=int, default=1)
    parser.add_argument("--train-n", type=int, default=16)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conditioned = {}
    sampling = {}
    for task in args.tasks:
        maker, semiring = TASK_CFG[task]
        conditioned[task] = {}
        sampling[task] = {}
        for n in args.sizes:
            samples, stats = build_conditioned_samples(
                maker,
                semiring,
                n,
                args.target_samples,
                910000 + n,
                args.test_delta_size,
                args.min_critical,
                args.max_candidates,
            )
            conditioned[task][str(n)] = samples
            sampling[task][str(n)] = stats
            print(task, n, stats, flush=True)

    raw = {}
    started = time.time()
    for task in args.tasks:
        _, semiring = TASK_CFG[task]
        raw[task] = {}
        for model_name in args.models:
            raw[task][model_name] = {str(n): {} for n in args.sizes}
            for seed in args.seeds:
                model = train_model(task, model_name, seed, args, device)
                for n in args.sizes:
                    metrics = evaluate_samples(
                        model,
                        conditioned[task][str(n)],
                        semiring,
                        device,
                        args.eval_batch,
                    )
                    for key, value in metrics.items():
                        raw[task][model_name][str(n)].setdefault(key, []).append(value)
                print(task, model_name, seed, f"{time.time() - started:.1f}s", flush=True)

    result = {
        "args": vars(args),
        "sampling": sampling,
        "raw": raw,
        "aggregate": aggregate(raw),
        "elapsed_seconds": time.time() - started,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    write_markdown(args.markdown, args, result["aggregate"], sampling)


if __name__ == "__main__":
    main()
