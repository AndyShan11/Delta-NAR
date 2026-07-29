"""Merge and summarize C42 noisy-previous-solution result shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANT_LABELS = {
    "copy_safe": "DeltaNAR+copy-safe repair",
    "repair_only": "DeltaNAR+repair-only",
}

METRICS = [
    ("node_acc_v", "Node validity"),
    ("graph_acc_v", "Graph validity"),
    ("changed_output_acc_v", "Changed-output validity"),
    ("copy_critical_acc_v", "Copy-critical validity"),
    ("pure_copy_node_acc_v", "Pure-copy node validity"),
    ("pure_copy_changed_acc_v", "Pure-copy changed validity"),
    ("gate_recall", "Affected gate recall"),
    ("copy_critical_gate_recall", "Copy-critical gate recall"),
    ("changed_output_fraction", "Changed-output fraction"),
    ("copy_critical_fraction", "Copy-critical fraction"),
]


def merge_raw(inputs: list[Path]) -> tuple[dict, list[dict], dict]:
    raw: dict = {}
    completed: list[dict] = []
    args0 = {}
    for path in inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not args0:
            args0 = data.get("args", {})
        completed.extend(data.get("completed", []))
        for variant, by_task in data.get("raw", {}).items():
            raw.setdefault(variant, {})
            for task, by_noise in by_task.items():
                raw[variant].setdefault(task, {})
                for noise, by_size in by_noise.items():
                    raw[variant][task].setdefault(noise, {})
                    for size, by_metric in by_size.items():
                        raw[variant][task][noise].setdefault(size, {})
                        for metric, values in by_metric.items():
                            raw[variant][task][noise][size].setdefault(metric, []).extend(values)
    return raw, completed, args0


def aggregate(raw: dict) -> dict:
    agg = {}
    for variant, by_task in raw.items():
        agg[variant] = {}
        for task, by_noise in by_task.items():
            agg[variant][task] = {}
            for noise, by_size in by_noise.items():
                agg[variant][task][noise] = {}
                for size, by_metric in by_size.items():
                    agg[variant][task][noise][size] = {
                        metric: [float(np.mean(values)), float(np.std(values)), len(values)]
                        for metric, values in by_metric.items()
                        if values
                    }
    return agg


def pm(pair: list[float] | None, digits: int = 3) -> str:
    if not pair:
        return "-"
    return f"{pair[0]:.{digits}f} +/- {pair[1]:.{digits}f} (n={int(pair[2])})"


def infer_order(args0: dict, agg: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    variants = args0.get("variants") or list(agg.keys())
    tasks = args0.get("tasks") or sorted({t for by_task in agg.values() for t in by_task})
    noises = [f"{float(x):.4g}" for x in args0.get("noise_rates", [])]
    if not noises:
        noises = sorted({n for by_task in agg.values() for by_noise in by_task.values() for n in by_noise}, key=float)
    sizes = [str(x) for x in args0.get("sizes", [])]
    if not sizes:
        sizes = sorted(
            {s for by_task in agg.values() for by_noise in by_task.values() for by_size in by_noise.values() for s in by_size},
            key=int,
        )
    return variants, tasks, noises, sizes


def write_markdown(path: Path, args0: dict, agg: dict) -> None:
    variants, tasks, noises, sizes = infer_order(args0, agg)
    lines = [
        "# C42 Noisy Previous-Solution Robustness\n",
        "Merged server summary for evaluation-time previous-solution corruption. "
        "Training is unchanged; only the previous-solution input is corrupted "
        "during evaluation.\n",
        "```json",
        json.dumps(args0, indent=2),
        "```\n",
    ]
    for task in tasks:
        lines.append(f"## {task}\n")
        for metric, title in METRICS:
            lines.append(f"### {title}\n")
            lines.append("| Variant | noise | " + " | ".join(f"n={n}" for n in sizes) + " |")
            lines.append("|---|---|" + "|".join("---:" for _ in sizes) + "|")
            for variant in variants:
                for noise in noises:
                    cells = []
                    for size in sizes:
                        pair = agg.get(variant, {}).get(task, {}).get(noise, {}).get(size, {}).get(metric)
                        cells.append(pm(pair))
                    lines.append(
                        f"| {VARIANT_LABELS.get(variant, variant)} | {noise} | "
                        + " | ".join(cells)
                        + " |"
                    )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="results/C42_noisy_prev_20seed_merged.json")
    ap.add_argument("--markdown", default="C42_NOISY_PREV_SERVER_SUMMARY.md")
    args = ap.parse_args()

    raw, completed, args0 = merge_raw([Path(p) for p in args.inputs])
    agg = aggregate(raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"args": args0, "completed": completed, "agg": agg, "raw": raw}, indent=2),
        encoding="utf-8",
    )
    write_markdown(Path(args.markdown), args0, agg)
    print(f"saved -> {args.out}")
    print(f"saved -> {args.markdown}")


if __name__ == "__main__":
    main()
