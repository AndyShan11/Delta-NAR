"""Merge C44 focused stress-test result shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANT_ORDER = ["copy_safe", "copy_safe_cert", "fixed_noisy", "fixed_noisy_cert"]
VARIANT_LABELS = {
    "copy_safe": "copy-safe",
    "copy_safe_cert": "copy-safe + cert-repair",
    "fixed_noisy": "fixed-target noisy",
    "fixed_noisy_cert": "fixed-target noisy + cert-repair",
}


def merge(inputs: list[Path]) -> tuple[dict, dict, list[dict], dict]:
    raw: dict = {}
    rollout_raw: dict = {}
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
        for variant, by_size in data.get("rollout_raw", {}).items():
            rollout_raw.setdefault(variant, {})
            for size, by_step in by_size.items():
                rollout_raw[variant].setdefault(size, {})
                for step, values in by_step.items():
                    rollout_raw[variant][size].setdefault(step, []).extend(values)
    return raw, rollout_raw, completed, args0


def aggregate(raw: dict) -> dict:
    out = {}
    for variant, by_task in raw.items():
        out[variant] = {}
        for task, by_noise in by_task.items():
            out[variant][task] = {}
            for noise, by_size in by_noise.items():
                out[variant][task][noise] = {}
                for size, by_metric in by_size.items():
                    out[variant][task][noise][size] = {
                        metric: [float(np.mean(values)), float(np.std(values)), len(values)]
                        for metric, values in by_metric.items()
                        if values
                    }
    return out


def aggregate_rollout(raw: dict) -> dict:
    out = {}
    for variant, by_size in raw.items():
        out[variant] = {}
        for size, by_step in by_size.items():
            out[variant][size] = {
                step: [float(np.mean(values)), float(np.std(values)), len(values)]
                for step, values in by_step.items()
                if values
            }
    return out


def pm(pair: list[float] | None) -> str:
    if not pair:
        return "-"
    return f"{pair[0]:.3f} +/- {pair[1]:.3f} (n={int(pair[2])})"


def write_markdown(path: Path, args0: dict, agg: dict, rollout_agg: dict) -> None:
    variants = [v for v in VARIANT_ORDER if v in agg] + [v for v in agg if v not in VARIANT_ORDER]
    tasks = sorted({task for by_task in agg.values() for task in by_task})
    noises = [f"{float(x):.4g}" for x in args0.get("noise_rates", [])]
    sizes = [str(x) for x in args0.get("sizes", [])]
    lines = [
        "# C44 Focused Stress-Test Server Summary\n",
        "Merged checkpointed results for fixed-target noisy-memory training and "
        "inference-time certificate repair.\n",
        "```json",
        json.dumps(args0, indent=2),
        "```\n",
    ]
    for task in tasks:
        lines.append(f"## {task}\n")
        for metric in [
            "node_acc_v",
            "graph_acc_v",
            "changed_output_acc_v",
            "copy_critical_acc_v",
            "copy_critical_gate_recall",
            "cert_repair_rate",
        ]:
            lines.append(f"### {metric}\n")
            lines.append("| Variant | noise | " + " | ".join(f"n={s}" for s in sizes) + " |")
            lines.append("|---|---|" + "|".join("---:" for _ in sizes) + "|")
            for variant in variants:
                for noise in noises:
                    cells = [pm(agg.get(variant, {}).get(task, {}).get(noise, {}).get(size, {}).get(metric)) for size in sizes]
                    lines.append(f"| {VARIANT_LABELS.get(variant, variant)} | {noise} | " + " | ".join(cells) + " |")
            lines.append("")
    if rollout_agg:
        lines.append("## sssp rollout\n")
        rollout_sizes = [str(x) for x in args0.get("rollout_sizes", [])]
        steps = [str(t) for t in range(int(args0.get("rollout_steps", 0)))]
        lines.append("| Variant | n | " + " | ".join(f"t={t}" for t in steps) + " |")
        lines.append("|---|---|" + "|".join("---:" for _ in steps) + "|")
        for variant in variants:
            for size in rollout_sizes:
                cells = [pm(rollout_agg.get(variant, {}).get(size, {}).get(step)) for step in steps]
                lines.append(f"| {VARIANT_LABELS.get(variant, variant)} | {size} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="results/C44_focused_stress_20seed_merged.json")
    ap.add_argument("--markdown", default="C44_FOCUSED_STRESS_SERVER_SUMMARY.md")
    args = ap.parse_args()
    raw, rollout_raw, completed, args0 = merge([Path(p) for p in args.inputs])
    agg = aggregate(raw)
    rollout_agg = aggregate_rollout(rollout_raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "args": args0,
                "completed": completed,
                "agg": agg,
                "rollout_agg": rollout_agg,
                "raw": raw,
                "rollout_raw": rollout_raw,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(Path(args.markdown), args0, agg, rollout_agg)
    print(f"saved -> {args.out}")
    print(f"saved -> {args.markdown}")


if __name__ == "__main__":
    main()
