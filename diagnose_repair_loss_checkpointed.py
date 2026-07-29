"""Checkpointed variant of diagnose_repair_loss.py for long server runs.

The original diagnostic writes JSON only after all tasks and seeds finish.
This wrapper preserves the same training/evaluation semantics, but flushes a
usable result file after every completed task/seed pair and skips completed
pairs when resumed with the same output path.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from diagnose_repair_loss import train_one, write_markdown


def aggregate(raw: dict) -> dict:
    agg = {}
    for task, by_size in raw.items():
        task_out = {}
        model_out = {}
        for size, by_metric in by_size.items():
            metric_out = {}
            for metric, values in by_metric.items():
                if values:
                    metric_out[metric] = [float(np.mean(values)), float(np.std(values))]
            if metric_out:
                model_out[size] = metric_out
        if model_out:
            task_out["DeltaNAR+repair"] = model_out
            agg[task] = task_out
    return agg


def load_existing(path: Path) -> tuple[dict, set[tuple[str, int]], list[dict]]:
    if not path.exists():
        return {}, set(), []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    completed_rows = data.get("completed", [])
    completed = {(str(row["task"]), int(row["seed"])) for row in completed_rows}
    return data.get("raw", {}), completed, completed_rows


def save_checkpoint(path: Path, markdown: Path, args, raw: dict, completed_rows: list[dict]) -> None:
    agg = aggregate(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "completed": completed_rows,
                "agg": agg,
                "raw": raw,
            },
            f,
            indent=2,
        )
    write_markdown(str(markdown), args, agg)
    print(f"checkpoint -> {path}", flush=True)
    print(f"checkpoint -> {markdown}", flush=True)


def parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--out", default="results/AHR_repair_loss_checkpointed.json")
    ap.add_argument("--markdown", default="RESULTS_AHR_REPAIR_LOSS_CHECKPOINTED.md")
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
    return ap


def main() -> None:
    args = parser().parse_args()
    out = Path(args.out)
    markdown = Path(args.markdown)
    raw, completed, completed_rows = load_existing(out)
    for task in args.tasks:
        raw.setdefault(task, {str(n): {} for n in args.sizes})
        for n in args.sizes:
            raw[task].setdefault(str(n), {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} tasks={args.tasks} seeds={args.seeds} sizes={args.sizes} "
        f"completed={len(completed)}",
        flush=True,
    )
    for task in args.tasks:
        for seed in args.seeds:
            key = (task, seed)
            if key in completed:
                print(f"[skip {task}/seed{seed}] already checkpointed", flush=True)
                continue
            t0 = time.time()
            res = train_one(task, seed, args, device)
            for n, metrics in res.items():
                for metric, value in metrics.items():
                    raw[task][n].setdefault(metric, []).append(value)
            elapsed = time.time() - t0
            completed.add(key)
            completed_rows.append({"task": task, "seed": seed, "elapsed_sec": elapsed})
            print(f"[{task}/DeltaNAR+repair/seed{seed}] {elapsed:.0f}s", flush=True)
            save_checkpoint(out, markdown, args, raw, completed_rows)

    save_checkpoint(out, markdown, args, raw, completed_rows)
    print("checkpointed diagnostic complete", flush=True)


if __name__ == "__main__":
    main()
