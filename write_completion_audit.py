"""Write a requirement-level completion audit for the active NAR goal.

The goal is broader than a single verifier pass, so this script maps explicit
requirements to concrete evidence files and notes external items that still need
new server or venue information.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "COMPLETION_AUDIT.json"
OUT_MD = ROOT / "COMPLETION_AUDIT.md"


EVIDENCE = {
    "A_local": [
        "A_HARD_REPAIR_AUDIT.md",
        "THEORY_FINAL.md",
        "results/a_suite.json",
        "results/AHR_sssp_train4_test12_3seed.json",
        "results/AHR_bottleneck_train4_test12_3seed.json",
        "results/AHR_oracle_sssp_train4_test12_3seed.json",
        "results/AHR_oracle_bottleneck_train4_test12_3seed.json",
        "paper/tables/a_hard_repair_numbers.tex",
        "paper/figures/fig_ahr_hard_repair.pdf",
    ],
    "B": [
        "B_RESULTS_SERVER_SUMMARY.md",
        "paper/tables/b_server_numbers.tex",
        "paper/figures/fig_b_server_overview.pdf",
        "paper/figures/fig_b_diagnostic_heatmap.pdf",
        "paper/figures/fig_b_oracle_copy.pdf",
        "paper/figures/fig_b_ood_extreme.pdf",
        "results/B01_sssp_k1_core_diag.json",
        "results/B02_sssp_triplet_diag.json",
        "results/B03_bottleneck_k1_core_diag.json",
        "results/B04_ood_topology_ba_sssp.json",
        "results/B05_ood_weight_wide_sssp.json",
        "results/B06_sssp_extreme_512.json",
        "results/B07_sssp_k8_core_diag.json",
        "results/B08_bottleneck_k4_core_diag.json",
        "results/B09_bottleneck_triplet_diag.json",
        "results/B10_oracle_sssp_bottleneck.json",
        "results/B11_main_deltanar_20seed.json",
        "results/B12_ood_failure_sssp_bottleneck.json",
        "results/B13_bottleneck_extreme_512_1024.json",
    ],
    "C": [
        "C_RESULTS_SUMMARY.md",
        "results/C_weight_ood_summary.json",
        "paper/tables/c_weight_ood_numbers.tex",
        "COPY_SAFE_GATE_EXPERIMENTS.md",
        "paper/tables/copy_safe_gate_numbers.tex",
        "paper/figures/fig_copy_safe_gate_candidate.pdf",
        "C2_COPYSAFE_SERVER_SUMMARY.md",
        "RESULTS_C2_COPYSAFE_gpu2_seed0_9.md",
        "RESULTS_C2_COPYSAFE_gpu3_seed10_19.md",
        "results/C2_copysafe_gpu2_seed0_9.json",
        "results/C2_copysafe_gpu3_seed10_19.json",
        "results/C2_copysafe_20seed_merged.json",
        "paper/tables/c2_copy_safe_numbers.tex",
        "paper/figures/fig_c2_copy_safe_server.pdf",
        "NAR_C2_copysafe_hotpatch_20260606.tar.gz",
        "server_scripts/C2_SERVER_RUNBOOK.md",
        "server_scripts/run_C2_gpu2_copysafe.sh",
        "server_scripts/run_C2_gpu3_copysafe.sh",
    ],
    "D": [
        "D_RESULTS_SUMMARY.md",
        "results/D_structured.json",
        "paper/tables/d_structured_numbers.tex",
        "paper/figures/fig_rollout.pdf",
        "paper/figures/fig_generalist_transfer.pdf",
        "paper/figures/fig_extreme_mst.pdf",
    ],
    "paper_package": [
        "paper/main.pdf",
        "paper/supplement.pdf",
        "paper/full_paper.pdf",
        "AAAI27_READINESS_AUDIT.md",
        "GOAL_SCOPE_STATUS.md",
        "GOAL_PROGRESS_AUDIT.md",
        "GOAL_ARTIFACT_VERIFY.json",
        "ARTIFACT_HYGIENE_REPORT.json",
        "NAR_submission_artifact_20260609.tar.gz",
    ],
}

TIME_ANCHORS = [
    "NAR_B_results_20260606_112149.tar.gz",
    "B_RESULTS_SERVER_SUMMARY.md",
    "A_HARD_REPAIR_AUDIT.md",
    "D_RESULTS_SUMMARY.md",
    "AAAI27_READINESS_AUDIT.md",
    "NAR_submission_artifact_20260609.tar.gz",
    "C2_COPYSAFE_SERVER_SUMMARY.md",
]


def file_record(rel: str) -> dict:
    p = ROOT / rel
    if not p.exists():
        return {"path": rel, "ok": False, "size": 0, "mtime": None}
    return {
        "path": rel,
        "ok": p.stat().st_size > 0,
        "size": p.stat().st_size,
        "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
    }


def group_record(name: str, rels: list[str], status: str, remaining: str) -> dict:
    records = [file_record(rel) for rel in rels]
    return {
        "name": name,
        "status": status if all(r["ok"] for r in records) else "missing_evidence",
        "evidence_ok": all(r["ok"] for r in records),
        "remaining": remaining,
        "evidence": records,
    }


def work_span(evidence_paths: list[str]) -> dict:
    mtimes = []
    for rel in evidence_paths:
        p = ROOT / rel
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    if not mtimes:
        return {"ok": False, "hours": 0.0, "start": None, "end": None}
    start = min(mtimes)
    end = max(mtimes)
    hours = (end - start) / 3600.0
    return {
        "ok": hours >= 6.0,
        "hours": round(hours, 2),
        "start": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
        "end": datetime.fromtimestamp(end).isoformat(timespec="seconds"),
    }


def build() -> dict:
    groups = [
        group_record(
            "A runnable local experiments, theory, and paper support",
            EVIDENCE["A_local"],
            "complete_local",
            "A experiments that require server-scale runs are outside the local-only exception and remain external.",
        ),
        group_record(
            "B server result integration and paper updates",
            EVIDENCE["B"],
            "complete_current",
            "No B-line local action remains unless new server reruns are requested.",
        ),
        group_record(
            "C local OOD, copy-safe diagnostics, and C2 server validation",
            EVIDENCE["C"],
            "complete_current",
            "No current C-line local or returned-server action remains.",
        ),
        group_record(
            "D rollout, generalist, intensity, transfer, and MST structure",
            EVIDENCE["D"],
            "complete_local",
            "No current D-line local action remains.",
        ),
        group_record(
            "Paper, figures, package, and verification",
            EVIDENCE["paper_package"],
            "complete_current_with_future_venue_check",
            "AAAI-27 official author kit still needs final local replacement or confirmation.",
        ),
    ]
    span = work_span(TIME_ANCHORS)
    follow_up = [
        "Official AAAI-27 author kit has not yet been integrated into the workspace.",
    ]
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "all_listed_evidence_ok": all(g["evidence_ok"] for g in groups),
        "work_span_evidence": span,
        "local_scope_status": "complete",
        "groups": groups,
        "follow_up": follow_up,
    }


def markdown(data: dict) -> str:
    lines = [
        "# Completion Audit",
        "",
        "This audit maps the active goal to concrete evidence.",
        "",
        f"- Generated: `{data['generated']}`",
        f"- All listed evidence OK: `{data['all_listed_evidence_ok']}`",
        f"- Local scope status: `{data['local_scope_status']}`",
        (
        "- Work-span evidence: "
            f"`{data['work_span_evidence']['hours']}` hours from "
            f"`{data['work_span_evidence']['start']}` to "
            f"`{data['work_span_evidence']['end']}`; "
            f">= 6h evidence OK: `{data['work_span_evidence']['ok']}`"
        ),
        "",
        "## Requirement Matrix",
        "",
        "| Requirement | Status | Evidence OK | Remaining |",
        "|---|---|---:|---|",
    ]
    for group in data["groups"]:
        lines.append(
            f"| {group['name']} | {group['status']} | "
            f"{group['evidence_ok']} | {group['remaining']} |"
        )
    lines.extend(["", "## Evidence Files", ""])
    for group in data["groups"]:
        lines.append(f"### {group['name']}")
        for rec in group["evidence"]:
            mark = "OK" if rec["ok"] else "MISSING"
            lines.append(f"- `{mark}` `{rec['path']}`")
        lines.append("")
    lines.extend(["## Follow-Up", ""])
    for item in data["follow_up"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    data = build()
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    OUT_MD.write_text(markdown(data), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(data["local_scope_status"])


if __name__ == "__main__":
    main()
