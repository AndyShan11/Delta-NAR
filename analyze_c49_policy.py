"""Analyze C49 task-aware certificate-routing results.

C49 asks whether the local certificate signal can be made useful by a scoped
policy: keep raw copy-safe on SSSP, where guarded repair hurts, and use guarded
repair on bottleneck, where it helps. This script does not merge shards; run
`analyze_c44_results.py` first, then pass the merged JSON here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = {
    "sssp": ["sssp", "sssp_wide100", "sssp_heavy_tail", "sssp_ba"],
    "bottleneck": ["bottleneck", "bottleneck_wide100", "bottleneck_heavy_tail", "bottleneck_ba"],
}

METRICS = ["node_acc_v", "changed_output_acc_v", "copy_critical_acc_v", "graph_acc_v"]


def get(agg: dict, variant: str, task: str, noise: str, size: str, metric: str):
    return agg.get(variant, {}).get(task, {}).get(noise, {}).get(size, {}).get(metric)


def pm(x) -> str:
    if not x:
        return "-"
    return f"{x[0]:.3f} +/- {x[1]:.3f} (n={int(x[2])})"


def delta(a, b) -> str:
    if not a or not b:
        return "-"
    return f"{a[0] - b[0]:+.3f}"


def avg_delta(agg: dict, variant: str, tasks: list[str], noise: str, size: str, metric: str) -> float | None:
    vals = []
    for task in tasks:
        a = get(agg, variant, task, noise, size, metric)
        b = get(agg, "copy_safe", task, noise, size, metric)
        if a and b:
            vals.append(a[0] - b[0])
    if not vals:
        return None
    return sum(vals) / len(vals)


def write_markdown(path: Path, data: dict, size: str, noise: str, guard_variants: list[str]) -> None:
    agg = data["agg"]
    lines = [
        "# C49 Task-Aware Certificate Routing Summary\n",
        "C49 evaluates whether thresholded local certificate repair is useful when "
        "scoped by task: raw copy-safe is retained for SSSP, while guarded repair "
        "is considered for bottleneck. This is a diagnostic policy analysis, not "
        "a claim that graph-level self-correction is solved.\n",
        "```json",
        json.dumps(data.get("args", {}), indent=2),
        "```\n",
        f"## Main slice: n={size}, noisy previous memory={noise}\n",
    ]
    for guard in guard_variants:
        lines.append(f"### Guard profile `{guard}`\n")
        lines.append("| Family | avg node delta | avg changed delta | avg copy-critical delta |")
        lines.append("|---|---:|---:|---:|")
        for family, tasks in TASKS.items():
            row = [family]
            for metric in ["node_acc_v", "changed_output_acc_v", "copy_critical_acc_v"]:
                d = avg_delta(agg, guard, tasks, noise, size, metric)
                row.append("-" if d is None else f"{d:+.3f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines.append("## Representative per-suite metrics\n")
    reps = ["sssp", "sssp_heavy_tail", "bottleneck", "bottleneck_heavy_tail", "bottleneck_wide100", "bottleneck_ba"]
    for task in reps:
        lines.append(f"### {task}\n")
        lines.append("| Variant | node | node delta | changed | changed delta | copy-critical | copy-critical delta |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for variant in ["copy_safe"] + guard_variants:
            row = [variant]
            for metric in ["node_acc_v", "changed_output_acc_v", "copy_critical_acc_v"]:
                x = get(agg, variant, task, noise, size, metric)
                b = get(agg, "copy_safe", task, noise, size, metric)
                row.extend([pm(x), delta(x, b)])
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--markdown", default="C49_TASK_AWARE_POLICY_SUMMARY.md")
    ap.add_argument("--size", default="256")
    ap.add_argument("--noise", default="0.3")
    ap.add_argument("--guards", nargs="+", default=["copy_safe_cu_m1", "copy_safe_cu_0"])
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    write_markdown(Path(args.markdown), data, args.size, args.noise, args.guards)
    print(f"saved -> {args.markdown}")


if __name__ == "__main__":
    main()
