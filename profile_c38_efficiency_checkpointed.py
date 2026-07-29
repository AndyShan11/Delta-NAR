"""Checkpointed C38 efficiency profiling for copy-safe hard repair.

This script measures evaluation latency and peak CUDA memory across graph sizes
for copy-safe repair and repair-only variants. It is intentionally lighter than
C34/C35: the output is meant for scalability/efficiency figures, not headline
accuracy.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from diagnose_a_metrics import eval_at
from diagnose_repair_loss import compute_repair_loss
from experiments import CONFIGS, TASK_CFG, chunks, set_seed
from src.model import DeltaNAR, collate


VARIANT_LABELS = {
    "copy_safe": "copy-safe repair",
    "repair_only": "repair-only",
}


def aggregate(raw: dict) -> dict:
    agg = {}
    for variant, by_task in raw.items():
        agg[variant] = {}
        for task, by_size in by_task.items():
            agg[variant][task] = {}
            for size, by_metric in by_size.items():
                agg[variant][task][size] = {
                    metric: [float(np.mean(values)), float(np.std(values)), len(values)]
                    for metric, values in by_metric.items()
                    if values
                }
    return agg


def load_existing(path: Path) -> tuple[dict, set[tuple[str, str, int]], list[dict]]:
    if not path.exists():
        return {}, set(), []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    completed_rows = data.get("completed", [])
    completed = {
        (str(row["variant"]), str(row["task"]), int(row["seed"]))
        for row in completed_rows
    }
    return data.get("raw", {}), completed, completed_rows


def pm(pair: list[float], digits: int = 3) -> str:
    return f"{pair[0]:.{digits}f} +/- {pair[1]:.{digits}f} (n={int(pair[2])})"


def write_markdown(path: Path, args, agg: dict) -> None:
    metrics = [
        ("eval_seconds", "Evaluation seconds"),
        ("eval_samples_per_sec", "Evaluation samples/sec"),
        ("peak_memory_mb", "Peak CUDA memory MB"),
        ("node_acc_v", "Node validity"),
        ("graph_acc_v", "Graph validity"),
        ("changed_output_acc_v", "Changed-output validity"),
        ("copy_critical_acc_v", "Copy-critical validity"),
    ]
    lines = [
        "# C38 Efficiency Profile\n",
        "Checkpointed efficiency/scalability profile for hard-repair copy-safe "
        "and repair-only variants.\n",
        "```json",
        json.dumps(vars(args), indent=2),
        "```\n",
    ]
    for task in args.tasks:
        lines.append(f"## {task}\n")
        for metric, title in metrics:
            lines.append(f"### {title}\n")
            lines.append("| Variant | " + " | ".join(f"n={n}" for n in args.sizes) + " |")
            lines.append("|---|" + "|".join("---:" for _ in args.sizes) + "|")
            for variant in args.variants:
                by_size = agg.get(variant, {}).get(task, {})
                cells = []
                for n in args.sizes:
                    pair = by_size.get(str(n), {}).get(metric)
                    cells.append(pm(pair, 2 if "seconds" in metric or "memory" in metric else 3) if pair else "-")
                lines.append(f"| {VARIANT_LABELS[variant]} | " + " | ".join(cells) + " |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, markdown: Path, args, raw: dict, completed_rows: list[dict]) -> None:
    agg = aggregate(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "completed": completed_rows,
                "agg": agg,
                "raw": raw,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(markdown, args, agg)
    print(f"checkpoint -> {path}", flush=True)
    print(f"checkpoint -> {markdown}", flush=True)


def train_model(task: str, seed: int, variant: str, args, device: str):
    mk, sem = TASK_CFG[task]
    set_seed(seed)
    copy_margin = args.copy_margin if args.copy_margin is not None else (2.0 if task == "bottleneck" else 0.0)
    cfg = dict(CONFIGS["DeltaNAR"])
    use_copy_safe = variant == "copy_safe"
    model = DeltaNAR(
        hidden=args.hidden,
        task="sssp",
        steps=(args.steps or None),
        copy_margin=copy_margin,
        copy_scale=args.copy_scale,
        use_copy_safe_gate=use_copy_safe,
        **cfg,
    ).to(device)
    train_args = copy.copy(args)
    train_args.use_copy_safe_gate = use_copy_safe
    train_args.copy_safe_weight = args.copy_safe_weight if use_copy_safe else 0.0
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train = mk(
        args.train_episodes,
        args.train_n,
        n_steps=5,
        seed0=1 + seed * 7919,
        delta_size=args.train_delta_size,
        kinds=args.train_delta_kinds,
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.epochs):
        model.train()
        random.shuffle(train)
        for ch in chunks(train, args.batch):
            b = collate(ch, device=device)
            outputs = model(b)
            loss, _ = compute_repair_loss(
                b, *outputs, semiring=sem, args=train_args, copy_safe_logit=model.last_copy_safe_logit
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.time() - t0
    train_peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
    return model, mk, sem, train_seconds, train_peak


def profile_one(task: str, seed: int, variant: str, args, device: str) -> dict:
    model, mk, sem, train_seconds, train_peak = train_model(task, seed, variant, args, device)
    by_size = {}
    for size in args.sizes:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        metrics = eval_at(
            model,
            mk,
            sem,
            size,
            args.test_episodes,
            device,
            args.eval_batch,
            topology=args.test_topology,
            heavy_tail=args.test_heavy_tail,
            whi=args.test_weight_hi,
            kinds=args.test_delta_kinds,
            delta_size=args.test_delta_size,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        eval_seconds = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0.0
        metrics.update(
            {
                "train_seconds": train_seconds,
                "train_peak_memory_mb": train_peak,
                "eval_seconds": eval_seconds,
                "eval_samples_per_sec": args.test_episodes / max(eval_seconds, 1e-9),
                "peak_memory_mb": peak,
            }
        )
        by_size[str(size)] = metrics
    return by_size


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sssp", "bottleneck"])
    ap.add_argument("--variants", nargs="+", default=["copy_safe", "repair_only"], choices=list(VARIANT_LABELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--train_n", type=int, default=16)
    ap.add_argument("--train_episodes", type=int, default=200)
    ap.add_argument("--test_episodes", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/C38_efficiency_checkpointed.json")
    ap.add_argument("--markdown", default="C38_EFFICIENCY_PROFILE.md")
    ap.add_argument("--test_topology", default="er")
    ap.add_argument("--test_heavy_tail", action="store_true")
    ap.add_argument("--test_weight_hi", type=int, default=100)
    ap.add_argument("--test_delta_kinds", nargs="+", default=None)
    ap.add_argument("--train_delta_kinds", nargs="+", default=None)
    ap.add_argument("--train_delta_size", type=int, default=4)
    ap.add_argument("--test_delta_size", type=int, default=12)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--repair_weight", type=float, default=1.0)
    ap.add_argument("--critical_weight", type=float, default=0.0)
    ap.add_argument("--copy_safe_weight", type=float, default=1.0)
    ap.add_argument("--copy_safe_pos_weight", type=float, default=20.0)
    ap.add_argument("--copy_margin", type=float, default=None)
    ap.add_argument("--copy_scale", type=float, default=5.0)
    return ap


def main() -> None:
    args = parser().parse_args()
    out = Path(args.out)
    markdown = Path(args.markdown)
    raw, completed, completed_rows = load_existing(out)
    for variant in args.variants:
        raw.setdefault(variant, {})
        for task in args.tasks:
            raw[variant].setdefault(task, {str(n): {} for n in args.sizes})
            for n in args.sizes:
                raw[variant][task].setdefault(str(n), {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} variants={args.variants} tasks={args.tasks} "
        f"seeds={args.seeds} sizes={args.sizes} completed={len(completed)}",
        flush=True,
    )
    for variant in args.variants:
        for task in args.tasks:
            for seed in args.seeds:
                key = (variant, task, seed)
                if key in completed:
                    print(f"[skip {variant}/{task}/seed{seed}] already checkpointed", flush=True)
                    continue
                t0 = time.time()
                by_size = profile_one(task, seed, variant, args, device)
                for size, metrics in by_size.items():
                    for metric, value in metrics.items():
                        raw[variant][task][size].setdefault(metric, []).append(value)
                elapsed = time.time() - t0
                completed.add(key)
                completed_rows.append(
                    {"variant": variant, "task": task, "seed": seed, "elapsed_sec": elapsed}
                )
                print(f"[{variant}/{task}/seed{seed}] {elapsed:.0f}s", flush=True)
                save_checkpoint(out, markdown, args, raw, completed_rows)

    save_checkpoint(out, markdown, args, raw, completed_rows)
    print("C38 efficiency profiling complete", flush=True)


if __name__ == "__main__":
    main()
