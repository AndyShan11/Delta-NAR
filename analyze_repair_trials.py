"""Summarize exploratory repair-loss trials.

These are local design probes for the AHR weakness. They should not be treated
as main paper results unless repeated and promoted into the main experiment
suite.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = ROOT / "REPAIR_LOSS_EXPERIMENTS.md"


def load_agg(name: str) -> dict:
    with (RESULTS / name).open("r", encoding="utf-8") as f:
        return json.load(f)["agg"]["sssp"]["DeltaNAR+repair"]["64"]


def load_base_seed0() -> dict:
    with (RESULTS / "AHR_sssp_train4_test12_3seed.json").open("r", encoding="utf-8") as f:
        return json.load(f)["raw"]["sssp"]["DeltaNAR"]["64"]


def cell(d: dict, metric: str, raw: bool = False) -> str:
    if raw:
        return f"{d[metric][0]:.3f}"
    return f"{d[metric][0]:.3f} +/- {d[metric][1]:.3f}"


def main() -> None:
    rows = [
        (
            "baseline scale5 margin0 seed0",
            load_base_seed0(),
            True,
            "Original DeltaNAR AHR seed0.",
        ),
        (
            "copy-safe gate, 3 seeds",
            load_agg("AHR_copysafegate_sssp_3seed.json"),
            False,
            "Best changed-output repair gain, but full-node and graph accuracy drop.",
        ),
        (
            "copy-safe gate seed0",
            load_agg("AHR_copysafegate_sssp_seed0.json"),
            False,
            "Strong single-seed changed gain; confirmed directionally by 3-seed run.",
        ),
        (
            "copy_margin=1, 3 seeds",
            load_agg("AHR_copymargin1_sssp_3seed.json"),
            False,
            "Raises changed/copy-critical on average, but graph drops and gate recall is unstable.",
        ),
        (
            "copy_margin=1 seed0",
            load_agg("AHR_copymargin1_sssp_seed0.json"),
            False,
            "Best single-seed calibration result; not stable enough alone.",
        ),
        (
            "copy_margin=0.5 seed0",
            load_agg("AHR_copymargin05_sssp_seed0.json"),
            False,
            "Intermediate margin performs worse than baseline.",
        ),
        (
            "copy_margin=-1 seed0",
            load_agg("AHR_copymargin_neg1_sssp_seed0.json"),
            False,
            "More aggressive copy gives mild changed gain but graph loss.",
        ),
        (
            "repair rw=1 cw=0.5",
            load_agg("AHR_repairloss_sssp_rw1_cw05_seed0.json"),
            False,
            "Raises changed-output but damages full-node and graph accuracy.",
        ),
        (
            "repair rw=0.25",
            load_agg("AHR_repairloss_sssp_rw025_seed0.json"),
            False,
            "Smaller repair weight still loses full-node/graph accuracy.",
        ),
        (
            "copy_scale=10",
            load_agg("AHR_copyscale10_sssp_seed0.json"),
            False,
            "Stronger copy bias does not improve changed-output repair.",
        ),
    ]
    metrics = [
        ("node_acc_v", "node"),
        ("graph_acc_v", "graph"),
        ("changed_output_acc_v", "changed"),
        ("copy_critical_acc_v", "copy-critical"),
        ("gate_recall", "gate recall"),
    ]
    lines = [
        "# Repair-Loss Exploratory Experiments\n",
        "Local SSSP AHR probes on seed 0, train4/test12, n=64. These are design "
        "diagnostics, not promoted paper results.\n",
        "| Variant | " + " | ".join(label for _, label in metrics) + " | Note |",
        "|---|" + "|".join("---:" for _ in metrics) + "|---|",
    ]
    for name, data, raw, note in rows:
        vals = [cell(data, metric, raw=raw) for metric, _ in metrics]
        lines.append("| " + " | ".join([name, *vals, note]) + " |")
    lines.append(
        "\nConclusion: the independent copy-safe gate is the most promising local "
        "direction because it raises changed-output accuracy from the 3-seed "
        "baseline 0.376 to 0.477, but it still trades away full-node and graph "
        "accuracy. Naive repair-focused CE has the same trade-off more severely. "
        "A larger copy margin gives some average changed/copy-critical gain, but "
        "graph accuracy drops and gate recall becomes unstable. The next useful "
        "change should combine copy-safety calibration with a repair branch that "
        "does not damage the copied region."
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
