"""Merge checkpointed C34 copy-safe gate server results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_LABEL = "DeltaNAR+copy-safe repair"
METRICS = [
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_raw(parts: list[dict]) -> dict:
    merged: dict = {}
    for part in parts:
        for task, by_size in part.get("raw", {}).items():
            task_out = merged.setdefault(task, {})
            for size, by_metric in by_size.items():
                size_out = task_out.setdefault(size, {})
                for metric, values in by_metric.items():
                    size_out.setdefault(metric, []).extend(values)
    return merged


def aggregate(raw: dict, model_label: str) -> dict:
    agg = {}
    for task, by_size in raw.items():
        model = {}
        for size, by_metric in by_size.items():
            metrics = {
                metric: [float(np.mean(values)), float(np.std(values)), len(values)]
                for metric, values in by_metric.items()
                if values
            }
            if metrics:
                model[size] = metrics
        if model:
            agg[task] = {model_label: model}
    return agg


def display_source(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def pm(pair: list[float]) -> str:
    return f"{pair[0]:.3f} +/- {pair[1]:.3f} (n={int(pair[2])})"


def write_markdown(path: Path, sources: list[Path], agg: dict, model_label: str) -> None:
    lines = [
        "# C34 Copy-Safe Hard-Repair Server Summary\n",
        "Merged checkpointed C34 run for the copy-safe gate under hard-repair "
        "size extrapolation and wide-weight OOD stress.\n",
        "## Sources\n",
    ]
    lines.extend(f"- `{display_source(p)}`" for p in sources)
    lines.append("")
    for task, by_model in agg.items():
        lines.append(f"## {task}\n")
        by_size = by_model[model_label]
        sizes = sorted(by_size, key=lambda x: int(x))
        for metric in METRICS:
            if any(metric in by_size[size] for size in sizes):
                lines.append(f"### {metric}\n")
                lines.append("| Model | " + " | ".join(f"n={size}" for size in sizes) + " |")
                lines.append("|---|" + "|".join("---:" for _ in sizes) + "|")
                cells = [pm(by_size[size][metric]) if metric in by_size[size] else "-" for size in sizes]
                lines.append(f"| {model_label} | " + " | ".join(cells) + " |")
                lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    args = ap.parse_args()

    inputs = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.inputs]
    parts = [load_json(p) for p in inputs]
    raw = merge_raw(parts)
    agg = aggregate(raw, args.model_label)

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"sources": [display_source(p) for p in inputs], "agg": agg, "raw": raw}, indent=2),
        encoding="utf-8",
    )

    md = Path(args.markdown) if Path(args.markdown).is_absolute() else ROOT / args.markdown
    write_markdown(md, inputs, agg, args.model_label)
    print(f"wrote {out}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
