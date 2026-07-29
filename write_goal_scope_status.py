"""Write a high-level scope status matrix for the active A/B/C/D goal."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "GOAL_SCOPE_STATUS.json"
OUT_MD = ROOT / "GOAL_SCOPE_STATUS.md"


def exists(rel: str) -> bool:
    p = ROOT / rel
    return p.exists() and p.stat().st_size > 0


def verifier_ok() -> bool:
    p = ROOT / "GOAL_ARTIFACT_VERIFY.json"
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("ok"))
    except json.JSONDecodeError:
        return False


def status_items() -> list[dict]:
    return [
        {
            "scope": "A local experiments and theory/paper support",
            "status": "complete_local",
            "evidence": [
                "A_HARD_REPAIR_AUDIT.md",
                "paper/tables/a_hard_repair_numbers.tex",
                "paper/figures/fig_ahr_hard_repair.pdf",
                "results/AHR_sssp_train4_test12_3seed.json",
                "results/AHR_bottleneck_train4_test12_3seed.json",
                "results/AHR_oracle_sssp_train4_test12_3seed.json",
                "results/AHR_oracle_bottleneck_train4_test12_3seed.json",
                "results/a_suite.json",
            ],
            "remaining": "A experiments that require server-scale reruns remain external by goal definition.",
        },
        {
            "scope": "B server results returned and integrated",
            "status": "complete_current",
            "evidence": [
                "B_RESULTS_SERVER_SUMMARY.md",
                "paper/tables/b_server_numbers.tex",
                "paper/figures/fig_b_server_overview.pdf",
                "paper/figures/fig_b_diagnostic_heatmap.pdf",
                "results/B01_sssp_k1_core_diag.json",
                "results/B13_bottleneck_extreme_512_1024.json",
            ],
            "remaining": "No current B-line local action remains.",
        },
        {
            "scope": "C local weight-OOD, copy-safe, and C2 server validation",
            "status": "complete_current",
            "evidence": [
                "C_RESULTS_SUMMARY.md",
                "results/C_weight_ood_summary.json",
                "paper/tables/c_weight_ood_numbers.tex",
                "C2_COPYSAFE_SERVER_SUMMARY.md",
                "results/C2_copysafe_20seed_merged.json",
                "paper/tables/c2_copy_safe_numbers.tex",
                "paper/figures/fig_c2_copy_safe_server.pdf",
                "NAR_C2_copysafe_hotpatch_20260606.tar.gz",
                "server_scripts/C2_SERVER_RUNBOOK.md",
            ],
            "remaining": "No current C-line local or returned-server action remains.",
        },
        {
            "scope": "D local rollout/generalist/intensity/MST structure",
            "status": "complete_local",
            "evidence": [
                "D_RESULTS_SUMMARY.md",
                "results/D_structured.json",
                "paper/tables/d_structured_numbers.tex",
                "paper/figures/fig_rollout.pdf",
                "paper/figures/fig_generalist_transfer.pdf",
                "paper/figures/fig_extreme_mst.pdf",
            ],
            "remaining": "No current D-line local action remains.",
        },
        {
            "scope": "Paper, figures, package, and verification",
            "status": "complete_current",
            "evidence": [
                "paper/main.pdf",
                "paper/supplement.pdf",
                "paper/full_paper.pdf",
                "GOAL_ARTIFACT_VERIFY.json",
                "ARTIFACT_HYGIENE_REPORT.json",
                "NAR_submission_artifact_20260609.tar.gz",
            ],
            "remaining": "AAAI-27 official author kit/checklist must be rechecked when released.",
        },
    ]


def build_status() -> dict:
    items = status_items()
    for item in items:
        missing = [rel for rel in item["evidence"] if not exists(rel)]
        item["evidence_ok"] = not missing
        item["missing"] = missing
    external_pending = [
        item["remaining"]
        for item in items
        if item["status"].endswith("pending") or "author kit" in item["remaining"]
    ]
    return {
        "verifier_ok": verifier_ok(),
        "all_listed_evidence_ok": all(item["evidence_ok"] for item in items),
        "overall_local_status": "complete",
        "items": items,
        "external_pending": external_pending,
    }


def write_markdown(data: dict) -> str:
    lines = [
        "# Goal Scope Status\n",
        "This file summarizes the active goal by scope. It distinguishes local work "
        "that is complete from external/server or future-venue items that still "
        "require new information.\n",
        f"- Verifier OK: `{data['verifier_ok']}`",
        f"- Listed evidence OK: `{data['all_listed_evidence_ok']}`",
        f"- Overall local status: `{data['overall_local_status']}`\n",
        "## Scope Matrix\n",
        "| Scope | Status | Evidence OK | Remaining |",
        "|---|---|---:|---|",
    ]
    for item in data["items"]:
        lines.append(
            "| "
            + item["scope"]
            + " | "
            + item["status"]
            + " | "
            + str(item["evidence_ok"])
            + " | "
            + item["remaining"]
            + " |"
        )
    lines.append("\n## Evidence Files\n")
    for item in data["items"]:
        lines.append(f"### {item['scope']}")
        for rel in item["evidence"]:
            mark = "OK" if exists(rel) else "MISSING"
            lines.append(f"- `{mark}` `{rel}`")
        if item["missing"]:
            lines.append("- Missing: " + ", ".join(f"`{m}`" for m in item["missing"]))
        lines.append("")
    lines.append("## External Pending\n")
    for item in data["external_pending"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    data = build_status()
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    OUT_MD.write_text(write_markdown(data), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(data["overall_local_status"])


if __name__ == "__main__":
    main()
