"""Merge checkpointed C38 efficiency profile results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
VARIANT_LABELS = {
    "copy_safe": "copy-safe repair",
    "repair_only": "repair-only",
}
METRICS = [
    ("eval_seconds", "Evaluation seconds"),
    ("eval_samples_per_sec", "Evaluation samples/sec"),
    ("peak_memory_mb", "Peak CUDA memory MB"),
    ("node_acc_v", "Node validity"),
    ("graph_acc_v", "Graph validity"),
    ("changed_output_acc_v", "Changed-output validity"),
    ("copy_critical_acc_v", "Copy-critical validity"),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_raw(parts: list[dict]) -> dict:
    merged: dict = {}
    for part in parts:
        for variant, by_task in part.get("raw", {}).items():
            variant_out = merged.setdefault(variant, {})
            for task, by_size in by_task.items():
                task_out = variant_out.setdefault(task, {})
                for size, by_metric in by_size.items():
                    size_out = task_out.setdefault(size, {})
                    for metric, values in by_metric.items():
                        size_out.setdefault(metric, []).extend(values)
    return merged


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


def display_source(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def pm(pair: list[float], digits: int = 3) -> str:
    return f"{pair[0]:.{digits}f} +/- {pair[1]:.{digits}f} (n={int(pair[2])})"


def write_markdown(path: Path, sources: list[Path], agg: dict) -> None:
    tasks = sorted({task for by_task in agg.values() for task in by_task})
    variants = [v for v in ["copy_safe", "repair_only"] if v in agg] + [
        v for v in sorted(agg) if v not in {"copy_safe", "repair_only"}
    ]
    lines = [
        "# C38 Efficiency Server Summary\n",
        "Merged C38 efficiency profile for copy-safe repair versus repair-only.\n",
        "## Sources\n",
    ]
    lines.extend(f"- `{display_source(p)}`" for p in sources)
    lines.append("")
    for task in tasks:
        sizes = sorted(
            {size for variant in variants for size in agg.get(variant, {}).get(task, {})},
            key=lambda x: int(x),
        )
        lines.append(f"## {task}\n")
        for metric, title in METRICS:
            if not any(metric in agg.get(v, {}).get(task, {}).get(size, {}) for v in variants for size in sizes):
                continue
            lines.append(f"### {title}\n")
            lines.append("| Variant | " + " | ".join(f"n={size}" for size in sizes) + " |")
            lines.append("|---|" + "|".join("---:" for _ in sizes) + "|")
            for variant in variants:
                cells = []
                for size in sizes:
                    pair = agg.get(variant, {}).get(task, {}).get(size, {}).get(metric)
                    cells.append(pm(pair, 2 if "seconds" in metric or "memory" in metric else 3) if pair else "-")
                lines.append(f"| {VARIANT_LABELS.get(variant, variant)} | " + " | ".join(cells) + " |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", required=True)
    args = ap.parse_args()

    inputs = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.inputs]
    raw = merge_raw([load_json(p) for p in inputs])
    agg = aggregate(raw)

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"sources": [display_source(p) for p in inputs], "agg": agg, "raw": raw}, indent=2),
        encoding="utf-8",
    )
    md = Path(args.markdown) if Path(args.markdown).is_absolute() else ROOT / args.markdown
    write_markdown(md, inputs, agg)
    print(f"wrote {out}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
