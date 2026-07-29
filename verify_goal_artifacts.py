"""Verify the current A/B/C/D artifact set for the active AAAI-27 work goal.

This is an evidence collector, not a claim that the whole long-running goal is
complete. It checks the local files that should exist after the A/B/C/D work and
writes a compact JSON + Markdown audit.
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

try:
    import fitz  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    fitz = None

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - environment-dependent fallback
    PdfReader = None


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "GOAL_ARTIFACT_VERIFY.json"
OUT_MD = ROOT / "GOAL_PROGRESS_AUDIT.md"

HARD_LOG_PATTERNS = [
    r"^!",
    r"Undefined control sequence",
    r"LaTeX Error",
    r"Citation.*undefined",
    r"Reference.*undefined",
    r"Overfull \\hbox",
]


REQUIRED_FILES = [
    "RESULTS.md",
    "A_HARD_REPAIR_AUDIT.md",
    "B_RESULTS_SERVER_SUMMARY.md",
    "C_RESULTS_SUMMARY.md",
    "D_RESULTS_SUMMARY.md",
    "COPY_SAFE_GATE_EXPERIMENTS.md",
    "C2_COPYSAFE_SERVER_SUMMARY.md",
    "C34_36H_C3_SIZE_SERVER_SUMMARY.md",
    "C34_36H_C4_WIDE_SERVER_SUMMARY.md",
    "C35_36H_C3_SIZE_REPAIRONLY_SERVER_SUMMARY.md",
    "C35_36H_C4_WIDE_REPAIRONLY_SERVER_SUMMARY.md",
    "C38_EFFICIENCY_SERVER_SUMMARY.md",
    "C34538_INTEGRATION_SUMMARY.md",
    "C42_NOISY_PREV_SERVER_SUMMARY.md",
    "C42_INTEGRATION_SUMMARY.md",
    "C43_SELF_CORRECTION_SERVER_SUMMARY.md",
    "C43_INTEGRATION_SUMMARY.md",
    "C44_18H_FOCUSED_STRESS_SERVER_SUMMARY.md",
    "C44_18H_COMPLETED_SERVER_SUMMARY.md",
    "C44_PARTIAL_INTEGRATION_SUMMARY.md",
    "C45_GUARDED_DEEP_SERVER_SUMMARY.md",
    "C45_RETRIEVAL_AUDIT_20260623.md",
    "C44_C45_INTEGRATION_SUMMARY_20260623.md",
    "C48_2D_GUARD_PROFILES_SERVER_SUMMARY.md",
    "C48_2D_FOCUSED_INTEGRATION_SUMMARY.md",
    "C49_TASKAWARE_SERVER_SUMMARY.md",
    "C49_TASKAWARE_POLICY_SUMMARY.md",
    "C49_TASKAWARE_INTEGRATION_SUMMARY.md",
    "RESULTS_C2_COPYSAFE_gpu2_seed0_9.md",
    "RESULTS_C2_COPYSAFE_gpu3_seed10_19.md",
    "RESULTS_C42_NOISY_PREV_gpu2_seed0_9.md",
    "RESULTS_C42_NOISY_PREV_gpu3_seed10_19.md",
    "RESULTS_C43_SELF_CORRECTION_gpu2_seed0_9.md",
    "RESULTS_C43_SELF_CORRECTION_gpu3_seed10_19.md",
    "RESULTS_C44_18H_FOCUSED_STRESS_gpu2_seed0_9.md",
    "RESULTS_C44_18H_FOCUSED_STRESS_gpu3_seed10_19.md",
    "RESULTS_C45_GUARDED_DEEP_gpu2_seed0_9.md",
    "RESULTS_C45_GUARDED_DEEP_gpu3_seed10_19.md",
    "RESULTS_C48_2D_GUARD_PROFILES_gpu2_seed0_3.md",
    "RESULTS_C48_2D_GUARD_PROFILES_gpu3_seed10_13.md",
    "RESULTS_C49_TASKAWARE_gpu2_seed0_5.md",
    "RESULTS_C49_TASKAWARE_gpu3_seed10_15.md",
    "REPAIR_LOSS_EXPERIMENTS.md",
    "ARTIFACT_MANIFEST.md",
    "AAAI27_READINESS_AUDIT.md",
    "COMPLETION_AUDIT.md",
    "COMPLETION_AUDIT.json",
    "GOAL_SCOPE_STATUS.md",
    "GOAL_SCOPE_STATUS.json",
    "analyze_b_results.py",
    "analyze_ahr_results.py",
    "analyze_c_results.py",
    "analyze_d_results.py",
    "analyze_copy_safe_gate.py",
    "analyze_c2_results.py",
    "analyze_c34_results.py",
    "analyze_c38_results.py",
    "analyze_c42_results.py",
    "analyze_c43_results.py",
    "train_c43_self_correction_checkpointed.py",
    "analyze_c44_results.py",
    "train_c44_focused_stress_checkpointed.py",
    "write_c34538_tables.py",
    "write_goal_scope_status.py",
    "write_completion_audit.py",
    "check_artifact_hygiene.py",
    "diagnose_repair_loss.py",
    "ARTIFACT_HYGIENE_REPORT.json",
    "paper/main.pdf",
    "paper/supplement.pdf",
    "paper/full_paper.pdf",
    "paper/tables/a_hard_repair_numbers.tex",
    "paper/tables/b_server_numbers.tex",
    "paper/tables/c_weight_ood_numbers.tex",
    "paper/tables/c2_copy_safe_numbers.tex",
    "paper/tables/c34538_stress_repair_numbers.tex",
    "paper/tables/c38_efficiency_numbers.tex",
    "paper/tables/copy_safe_gate_numbers.tex",
    "paper/tables/d_structured_numbers.tex",
    "paper/figures/fig_size_ood.pdf",
    "paper/figures/fig_b_server_overview.pdf",
    "paper/figures/fig_ahr_hard_repair.pdf",
    "paper/figures/fig_copy_safe_gate_candidate.pdf",
    "paper/figures/fig_c2_copy_safe_server.pdf",
    "paper/figures/fig_c34538_stress_repair.pdf",
    "paper/figures/fig_c38_efficiency.pdf",
    "paper/figures/fig_generalist_transfer.pdf",
    "paper/figures/fig_extreme_mst.pdf",
    "results/C_weight_ood_summary.json",
    "results/C2_copysafe_gpu2_seed0_9.json",
    "results/C2_copysafe_gpu3_seed10_19.json",
    "results/C2_copysafe_20seed_merged.json",
    "results/C34_36h_c3_size_20seed_merged.json",
    "results/C34_36h_c4_wide_20seed_merged.json",
    "results/C35_36h_c3_size_repaironly_20seed_merged.json",
    "results/C35_36h_c4_wide_repaironly_20seed_merged.json",
    "results/C38_efficiency_10seed_merged.json",
    "results/C42_noisy_prev_gpu2_seed0_9.json",
    "results/C42_noisy_prev_gpu3_seed10_19.json",
    "results/C42_noisy_prev_20seed_merged.json",
    "results/C43_self_correction_gpu2_seed0_9.json",
    "results/C43_self_correction_gpu3_seed10_19.json",
    "results/C43_self_correction_20seed_merged.json",
    "results/C44_focused_stress18h_gpu2_seed0_9.json",
    "results/C44_focused_stress18h_gpu3_seed10_19.json",
    "results/C44_focused_stress18h_20seed_merged.json",
    "results/C44_focused_stress18h_completed_20seed_merged.json",
    "results/C45_guarded_deep_gpu2_seed0_9.json",
    "results/C45_guarded_deep_gpu3_seed10_19.json",
    "results/C45_guarded_deep_20seed_merged.json",
    "results/C48_2d_guard_profiles_gpu2_seed0_3.json",
    "results/C48_2d_guard_profiles_gpu3_seed10_13.json",
    "results/C48_2d_guard_profiles_8seed_merged.json",
    "results/C49_taskaware_gpu2_seed0_5.json",
    "results/C49_taskaware_gpu3_seed10_15.json",
    "results/C49_taskaware_12seed_merged.json",
    "results/D_structured.json",
    "NAR_C2_copysafe_hotpatch_20260606.tar.gz",
    "NAR_C34538_combo_hotpatch_20260610.tar.gz",
    "NAR_C42_noisy_prev_hotpatch_20260612.tar.gz",
    "NAR_C42_noisy_prev_results_20260614_121446.tar.gz",
    "NAR_C43_self_correction_hotpatch_20260614.tar.gz",
    "NAR_C43_self_correction_results_20260616_115903.tar.gz",
    "NAR_C44_focused_stress18h_results_20260617_122748.tar.gz",
    "NAR_C45_6d_extended_stress_hotpatch_20260617.tar.gz",
    "NAR_C45_6d_extended_stress_results_20260623_160741.tar.gz",
    "NAR_submission_artifact_20260623.tar.gz",
    "server_scripts/C2_SERVER_RUNBOOK.md",
    "server_scripts/run_C2_gpu2_copysafe.sh",
    "server_scripts/run_C2_gpu3_copysafe.sh",
    "server_scripts/C42_NOISY_PREV_RUNBOOK.md",
    "server_scripts/run_C42_gpu2_noisy_prev.sh",
    "server_scripts/run_C42_gpu3_noisy_prev.sh",
    "server_scripts/C43_SELF_CORRECTION_RUNBOOK.md",
    "server_scripts/run_C43_gpu2_self_correction.sh",
    "server_scripts/run_C43_gpu3_self_correction.sh",
    "server_scripts/C45_6D_EXTENDED_STRESS_RUNBOOK.md",
    "server_scripts/run_C45_6d_gpu2_extended_stress.sh",
    "server_scripts/run_C45_6d_gpu3_extended_stress.sh",
]

B_FILES = [
    "B01_sssp_k1_core_diag.json",
    "B02_sssp_triplet_diag.json",
    "B03_bottleneck_k1_core_diag.json",
    "B04_ood_topology_ba_sssp.json",
    "B05_ood_weight_wide_sssp.json",
    "B06_sssp_extreme_512.json",
    "B07_sssp_k8_core_diag.json",
    "B08_bottleneck_k4_core_diag.json",
    "B09_bottleneck_triplet_diag.json",
    "B10_oracle_sssp_bottleneck.json",
    "B11_main_deltanar_20seed.json",
    "B12_ood_failure_sssp_bottleneck.json",
    "B13_bottleneck_extreme_512_1024.json",
]

AHR_FILES = [
    "AHR_sssp_train4_test12_3seed.json",
    "AHR_bottleneck_train4_test12_3seed.json",
    "AHR_oracle_sssp_train4_test12_3seed.json",
    "AHR_oracle_bottleneck_train4_test12_3seed.json",
    "AHR_sssp_train12_test12_3seed.json",
    "AHR_bottleneck_train12_test12_3seed.json",
    "AHR_copysafegate_sssp_3seed.json",
    "AHR_copysafegate_bottleneck_3seed.json",
]

C2_TAR_MEMBERS = {
    "src/model.py",
    "diagnose_repair_loss.py",
    "analyze_c2_results.py",
    "server_scripts/run_C2_gpu2_copysafe.sh",
    "server_scripts/run_C2_gpu3_copysafe.sh",
    "server_scripts/C2_SERVER_RUNBOOK.md",
}


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})

    def require_file(self, rel: str) -> None:
        p = ROOT / rel
        self.add(f"file exists: {rel}", p.exists() and p.stat().st_size > 0, f"{p.stat().st_size if p.exists() else 'missing'}")

    @property
    def ok(self) -> bool:
        return all(c["ok"] for c in self.checks)


def load_json(rel: str) -> dict:
    with (ROOT / rel).open("r", encoding="utf-8") as f:
        return json.load(f)


def has_path(data: dict, path: list[str]) -> bool:
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def check_json_path(audit: Audit, rel: str, path: list[str]) -> None:
    try:
        data = load_json(rel)
        audit.add(f"json path: {rel}::{'.'.join(path)}", has_path(data, path))
    except Exception as exc:  # noqa: BLE001
        audit.add(f"json path: {rel}::{'.'.join(path)}", False, repr(exc))


def check_pdf_pages(audit: Audit, rel: str, expected: int) -> None:
    try:
        if fitz is not None:
            pages = len(fitz.open(ROOT / rel))
        elif PdfReader is not None:
            pages = len(PdfReader(str(ROOT / rel)).pages)
        else:
            raise RuntimeError("neither fitz nor pypdf is available")
        audit.add(f"pdf pages: {rel}", pages == expected, f"got {pages}, expected {expected}")
    except Exception as exc:  # noqa: BLE001
        audit.add(f"pdf pages: {rel}", False, repr(exc))


def check_logs(audit: Audit) -> None:
    rx = re.compile("|".join(f"(?:{p})" for p in HARD_LOG_PATTERNS), re.MULTILINE)
    for rel in ["paper/main.log", "paper/supplement.log", "paper/full_paper.log"]:
        p = ROOT / rel
        if not p.exists():
            audit.add(f"log hard-error scan: {rel}", False, "missing")
            continue
        text = p.read_text(errors="ignore")
        match = rx.search(text)
        audit.add(f"log hard-error scan: {rel}", match is None, match.group(0) if match else "clean")


def check_tex_macros(audit: Audit) -> None:
    macros = {
        "paper/tables/a_hard_repair_numbers.tex": r"\AHRModelRows",
        "paper/tables/b_server_numbers.tex": r"\BServerMainRows",
        "paper/tables/c_weight_ood_numbers.tex": r"\CWeightOODRows",
        "paper/tables/copy_safe_gate_numbers.tex": r"\CopySafeGateRows",
        "paper/tables/c2_copy_safe_numbers.tex": r"\CtwoCopySafeRows",
        "paper/tables/c34538_stress_repair_numbers.tex": r"\CStressRepairRows",
        "paper/tables/c38_efficiency_numbers.tex": r"\CEfficiencyRows",
        "paper/tables/d_structured_numbers.tex": r"\DIntensityRows",
    }
    for rel, macro in macros.items():
        p = ROOT / rel
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        audit.add(f"tex macro: {macro}", macro in text, rel)


def check_c2_tar(audit: Audit) -> None:
    rel = "NAR_C2_copysafe_hotpatch_20260606.tar.gz"
    p = ROOT / rel
    if not p.exists():
        audit.add("C2 hotpatch tar members", False, "missing tarball")
        return
    try:
        with tarfile.open(p, "r:gz") as tf:
            members = set(tf.getnames())
        missing = sorted(C2_TAR_MEMBERS - members)
        audit.add("C2 hotpatch tar members", not missing, "missing: " + ", ".join(missing) if missing else "complete")
    except Exception as exc:  # noqa: BLE001
        audit.add("C2 hotpatch tar members", False, repr(exc))


def check_temp_files(audit: Audit) -> None:
    leftovers = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "paper").glob("_render_*.png"))
    audit.add("no paper render temp PNGs", not leftovers, ", ".join(leftovers) if leftovers else "clean")


def check_submission_package(audit: Audit) -> None:
    rel = "NAR_submission_artifact_20260623.tar.gz"
    p = ROOT / rel
    if not p.exists():
        audit.add("submission package contents", False, "missing package")
        return
    bad_prefixes = ("/_A_", "/_c2_", "/_copy_safe_smoke", "/_repair_smoke", "/_smoke")
    bad_names = ("__pycache__", "_server_B_unpack")
    required = {
        "NAR/paper/main.pdf",
        "NAR/paper/supplement.pdf",
        "NAR/paper/full_paper.pdf",
        "NAR/results/C_weight_ood_summary.json",
        "NAR/results/C2_copysafe_20seed_merged.json",
        "NAR/results/C34_36h_c3_size_20seed_merged.json",
        "NAR/results/C34_36h_c4_wide_20seed_merged.json",
        "NAR/results/C35_36h_c3_size_repaironly_20seed_merged.json",
        "NAR/results/C35_36h_c4_wide_repaironly_20seed_merged.json",
        "NAR/results/C38_efficiency_10seed_merged.json",
        "NAR/results/C42_noisy_prev_20seed_merged.json",
        "NAR/results/C43_self_correction_20seed_merged.json",
        "NAR/results/C44_focused_stress18h_20seed_merged.json",
        "NAR/results/C44_focused_stress18h_completed_20seed_merged.json",
        "NAR/results/C45_guarded_deep_20seed_merged.json",
        "NAR/results/C48_2d_guard_profiles_8seed_merged.json",
        "NAR/results/C49_taskaware_12seed_merged.json",
        "NAR/results/D_structured.json",
        "NAR/C2_COPYSAFE_SERVER_SUMMARY.md",
        "NAR/C34_36H_C3_SIZE_SERVER_SUMMARY.md",
        "NAR/C35_36H_C3_SIZE_REPAIRONLY_SERVER_SUMMARY.md",
        "NAR/C38_EFFICIENCY_SERVER_SUMMARY.md",
        "NAR/C34538_INTEGRATION_SUMMARY.md",
        "NAR/C42_NOISY_PREV_SERVER_SUMMARY.md",
        "NAR/C42_INTEGRATION_SUMMARY.md",
        "NAR/C43_SELF_CORRECTION_SERVER_SUMMARY.md",
        "NAR/C43_INTEGRATION_SUMMARY.md",
        "NAR/C44_18H_FOCUSED_STRESS_SERVER_SUMMARY.md",
        "NAR/C44_18H_COMPLETED_SERVER_SUMMARY.md",
        "NAR/C44_PARTIAL_INTEGRATION_SUMMARY.md",
        "NAR/C45_GUARDED_DEEP_SERVER_SUMMARY.md",
        "NAR/C44_C45_INTEGRATION_SUMMARY_20260623.md",
        "NAR/C48_2D_GUARD_PROFILES_SERVER_SUMMARY.md",
        "NAR/C48_2D_FOCUSED_INTEGRATION_SUMMARY.md",
        "NAR/C49_TASKAWARE_SERVER_SUMMARY.md",
        "NAR/C49_TASKAWARE_POLICY_SUMMARY.md",
        "NAR/C49_TASKAWARE_INTEGRATION_SUMMARY.md",
        "NAR/NAR_C2_copysafe_hotpatch_20260606.tar.gz",
        "NAR/NAR_C34538_combo_hotpatch_20260610.tar.gz",
        "NAR/NAR_C42_noisy_prev_hotpatch_20260612.tar.gz",
        "NAR/NAR_C42_noisy_prev_results_20260614_121446.tar.gz",
        "NAR/NAR_C43_self_correction_hotpatch_20260614.tar.gz",
        "NAR/NAR_C43_self_correction_results_20260616_115903.tar.gz",
        "NAR/NAR_C44_focused_stress18h_results_20260617_122748.tar.gz",
        "NAR/NAR_C45_6d_extended_stress_hotpatch_20260617.tar.gz",
        "NAR/NAR_C45_6d_extended_stress_results_20260623_160741.tar.gz",
        "NAR/COMPLETION_AUDIT.md",
        "NAR/COMPLETION_AUDIT.json",
        "NAR/GOAL_SCOPE_STATUS.md",
        "NAR/GOAL_SCOPE_STATUS.json",
        "NAR/SUBMISSION_MANIFEST.tsv",
    }
    try:
        with tarfile.open(p, "r:gz") as tf:
            members = set(tf.getnames())
        missing = sorted(required - members)
        bad = sorted(
            name
            for name in members
            if any(part in name for part in bad_names)
            or any(name.startswith("NAR" + prefix) for prefix in bad_prefixes)
            or name.endswith((".aux", ".toc", ".pyc"))
            or name in {"NAR/nar_results.tar.gz", "NAR/NAR_B_results_20260606_112149.tar.gz"}
        )
        ok = not missing and not bad
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if bad:
            detail.append("bad: " + ", ".join(bad[:8]))
        audit.add("submission package contents", ok, "; ".join(detail) if detail else "clean")
    except Exception as exc:  # noqa: BLE001
        audit.add("submission package contents", False, repr(exc))


def check_hygiene_report(audit: Audit) -> None:
    rel = "ARTIFACT_HYGIENE_REPORT.json"
    p = ROOT / rel
    if not p.exists():
        audit.add("artifact hygiene report", False, "missing")
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        audit.add("artifact hygiene report", bool(data.get("ok")), f"{len(data.get('issues', []))} issues")
    except Exception as exc:  # noqa: BLE001
        audit.add("artifact hygiene report", False, repr(exc))


def check_completion_audit(audit: Audit) -> None:
    rel = "COMPLETION_AUDIT.json"
    p = ROOT / rel
    if not p.exists():
        audit.add("completion audit status", False, "missing")
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ok = (
            data.get("all_listed_evidence_ok") is True
            and data.get("work_span_evidence", {}).get("ok") is True
            and data.get("local_scope_status") == "complete"
        )
        hours = data.get("work_span_evidence", {}).get("hours")
        audit.add("completion audit status", ok, f"work-span hours: {hours}")
    except Exception as exc:  # noqa: BLE001
        audit.add("completion audit status", False, repr(exc))


def build_markdown(result: dict) -> str:
    lines = [
        "# Goal Progress Audit\n",
        "This audit verifies the current local A/B/C/D artifact set. It is an "
        "evidence snapshot for the active A/B/C/D goal.\n",
        f"Overall verifier status: **{'PASS' if result['ok'] else 'FAIL'}**\n",
        "## Checked Evidence\n",
    ]
    for check in result["checks"]:
        mark = "PASS" if check["ok"] else "FAIL"
        detail = f" -- {check['detail']}" if check.get("detail") else ""
        lines.append(f"- `{mark}` {check['name']}{detail}")
    lines.extend(
        [
            "\n## Remaining Non-Local / Follow-Up Items\n",
            "- The AAAI-27 official template and checklist should be rechecked when the venue releases the final author kit.",
            "- Final submission packaging should exclude smoke artifacts such as `_A_*` and `_c2_*` files.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    audit = Audit()
    for rel in REQUIRED_FILES:
        audit.require_file(rel)
    for name in B_FILES:
        audit.require_file(f"results/{name}")
    for name in AHR_FILES:
        audit.require_file(f"results/{name}")

    check_json_path(audit, "results/B11_main_deltanar_20seed.json", ["agg", "sssp", "DeltaNAR", "256", "node_acc_v"])
    check_json_path(audit, "results/B11_main_deltanar_20seed.json", ["agg", "bottleneck", "DeltaNAR", "256", "node_acc_v"])
    check_json_path(audit, "results/AHR_sssp_train4_test12_3seed.json", ["agg", "sssp", "DeltaNAR", "64", "changed_output_acc_v"])
    check_json_path(audit, "results/AHR_bottleneck_train4_test12_3seed.json", ["agg", "bottleneck", "DeltaNAR", "64", "changed_output_acc_v"])
    check_json_path(audit, "results/C_weight_ood_summary.json", ["agg", "DeltaNAR", "wide"])
    check_json_path(audit, "results/C2_copysafe_20seed_merged.json", ["agg", "sssp", "DeltaNAR+copy-safe", "64", "changed_output_acc_v"])
    check_json_path(audit, "results/C2_copysafe_20seed_merged.json", ["agg", "bottleneck", "DeltaNAR+copy-safe", "64", "graph_acc_v"])
    check_json_path(audit, "results/C34_36h_c3_size_20seed_merged.json", ["agg", "sssp", "DeltaNAR+copy-safe repair", "512", "node_acc_v"])
    check_json_path(audit, "results/C34_36h_c3_size_20seed_merged.json", ["agg", "bottleneck", "DeltaNAR+copy-safe repair", "512", "node_acc_v"])
    check_json_path(audit, "results/C35_36h_c3_size_repaironly_20seed_merged.json", ["agg", "sssp", "DeltaNAR+repair-only", "512", "node_acc_v"])
    check_json_path(audit, "results/C38_efficiency_10seed_merged.json", ["agg", "copy_safe", "sssp", "512", "eval_seconds"])
    check_json_path(audit, "results/C42_noisy_prev_20seed_merged.json", ["agg", "copy_safe", "sssp", "0.1", "128", "node_acc_v"])
    check_json_path(audit, "results/C42_noisy_prev_20seed_merged.json", ["agg", "copy_safe", "bottleneck", "0.1", "128", "node_acc_v"])
    check_json_path(audit, "results/C43_self_correction_20seed_merged.json", ["agg", "copy_safe", "sssp", "0.1", "128", "node_acc_v"])
    check_json_path(audit, "results/C43_self_correction_20seed_merged.json", ["agg", "self_correct", "bottleneck", "0.1", "64", "node_acc_v"])
    check_json_path(audit, "results/C43_self_correction_20seed_merged.json", ["rollout_agg", "copy_safe", "64", "59"])
    check_json_path(audit, "results/C44_focused_stress18h_20seed_merged.json", ["agg", "copy_safe", "sssp", "0.2", "256", "node_acc_v"])
    check_json_path(audit, "results/C44_focused_stress18h_20seed_merged.json", ["agg", "copy_safe_cert", "bottleneck", "0.2", "256", "graph_acc_v"])
    check_json_path(audit, "results/C44_focused_stress18h_20seed_merged.json", ["rollout_agg", "copy_safe", "128", "79"])
    check_json_path(audit, "results/C44_focused_stress18h_completed_20seed_merged.json", ["agg", "fixed_noisy", "sssp", "0.2", "256", "node_acc_v"])
    check_json_path(audit, "results/C44_focused_stress18h_completed_20seed_merged.json", ["agg", "fixed_noisy_cert", "bottleneck", "0.2", "256", "graph_acc_v"])
    check_json_path(audit, "results/C44_focused_stress18h_completed_20seed_merged.json", ["rollout_agg", "fixed_noisy", "128", "79"])
    check_json_path(audit, "results/C45_guarded_deep_20seed_merged.json", ["agg", "copy_safe_cert", "bottleneck", "0.3", "256", "changed_output_acc_v"])
    check_json_path(audit, "results/C45_guarded_deep_20seed_merged.json", ["agg", "copy_safe_cert", "sssp_heavy_tail", "0.3", "256", "node_acc_v"])
    check_json_path(audit, "results/C45_guarded_deep_20seed_merged.json", ["rollout_agg", "copy_safe", "256", "119"])
    check_json_path(audit, "results/C48_2d_guard_profiles_8seed_merged.json", ["agg", "copy_safe_cu_m1", "sssp", "0.3", "256", "node_acc_v"])
    check_json_path(audit, "results/C48_2d_guard_profiles_8seed_merged.json", ["agg", "copy_safe_cu_0", "bottleneck", "0.3", "256", "changed_output_acc_v"])
    check_json_path(audit, "results/C48_2d_guard_profiles_8seed_merged.json", ["rollout_agg", "copy_safe", "256", "119"])
    check_json_path(audit, "results/C49_taskaware_12seed_merged.json", ["agg", "copy_safe", "sssp", "0.3", "256", "node_acc_v"])
    check_json_path(audit, "results/C49_taskaware_12seed_merged.json", ["agg", "copy_safe_cu_0", "bottleneck", "0.3", "256", "changed_output_acc_v"])
    check_json_path(audit, "results/C49_taskaware_12seed_merged.json", ["rollout_agg", "copy_safe", "256", "119"])
    check_json_path(audit, "results/D_structured.json", ["intensity_paper", "agg", "DeltaNAR", "8", "node_acc_v"])
    check_json_path(audit, "results/D_structured.json", ["rollout", "agg", "DeltaNAR", "mean"])
    check_json_path(audit, "results/D_structured.json", ["mst", "agg", "DeltaNAR", "64", "edge_acc"])

    check_pdf_pages(audit, "paper/main.pdf", 8)
    check_pdf_pages(audit, "paper/supplement.pdf", 38)
    check_pdf_pages(audit, "paper/full_paper.pdf", 46)
    check_logs(audit)
    check_tex_macros(audit)
    check_c2_tar(audit)
    check_temp_files(audit)
    check_submission_package(audit)
    check_hygiene_report(audit)
    check_completion_audit(audit)

    result = {"ok": audit.ok, "checks": audit.checks}
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    OUT_MD.write_text(build_markdown(result), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print("PASS" if audit.ok else "FAIL")


if __name__ == "__main__":
    main()
