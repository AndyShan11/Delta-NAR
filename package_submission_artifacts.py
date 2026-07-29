"""Build a clean submission/reproducibility artifact bundle.

The working directory contains useful smoke outputs and old local tarballs. This
script creates a curated archive for review/reproduction without those temporary
files. It does not delete anything from the workspace.
"""

from __future__ import annotations

import argparse
import tarfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULT_NAME = "NAR_submission_artifact_20260623.tar.gz"

ROOT_FILES = [
    "README.md",
    "REPRODUCIBILITY.md",
    "RESULTS.md",
    "ARTIFACT_MANIFEST.md",
    "AAAI27_READINESS_AUDIT.md",
    "COMPLETION_AUDIT.md",
    "COMPLETION_AUDIT.json",
    "GOAL_PROGRESS_AUDIT.md",
    "GOAL_SCOPE_STATUS.md",
    "GOAL_SCOPE_STATUS.json",
    "GOAL_ARTIFACT_VERIFY.json",
    "A_HARD_REPAIR_AUDIT.md",
    "B_RESULTS_SERVER_SUMMARY.md",
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
    "C44_C45_INTEGRATION_SUMMARY_20260623.md",
    "C48_2D_GUARD_PROFILES_SERVER_SUMMARY.md",
    "C48_2D_FOCUSED_INTEGRATION_SUMMARY.md",
    "C49_TASKAWARE_SERVER_SUMMARY.md",
    "C49_TASKAWARE_POLICY_SUMMARY.md",
    "C49_TASKAWARE_INTEGRATION_SUMMARY.md",
    "C_RESULTS_SUMMARY.md",
    "D_RESULTS_SUMMARY.md",
    "COPY_SAFE_GATE_EXPERIMENTS.md",
    "REPAIR_LOSS_EXPERIMENTS.md",
    "THEORY_FINAL.md",
    "NOVELTY_CLAIMS.md",
    "RELATED_WORK_AUDIT.md",
    "BASELINE_CONFIG_AUDIT.md",
    "DESIGN.md",
    "NAR_C2_copysafe_hotpatch_20260606.tar.gz",
    "NAR_C34538_combo_hotpatch_20260610.tar.gz",
    "NAR_C42_noisy_prev_hotpatch_20260612.tar.gz",
    "NAR_C42_noisy_prev_results_20260614_121446.tar.gz",
    "NAR_C43_self_correction_hotpatch_20260614.tar.gz",
    "NAR_C43_self_correction_results_20260616_115903.tar.gz",
    "NAR_C44_focused_stress18h_results_20260617_122748.tar.gz",
    "NAR_C45_6d_extended_stress_hotpatch_20260617.tar.gz",
    "NAR_C45_6d_extended_stress_results_20260623_160741.tar.gz",
]

ROOT_GLOBS = [
    "*.py",
    "RESULTS_AHR_*.md",
    "RESULTS_B*.md",
    "RESULTS_C2_COPYSAFE_*.md",
    "RESULTS_C34_36H_*.md",
    "RESULTS_C35_36H_*.md",
    "RESULTS_C42_NOISY_PREV_*.md",
    "RESULTS_C43_SELF_CORRECTION_*.md",
    "RESULTS_C44_18H_FOCUSED_STRESS_*.md",
    "RESULTS_C45_GUARDED_DEEP_*.md",
    "RESULTS_C48_2D_GUARD_PROFILES_*.md",
    "RESULTS_C49_TASKAWARE_*.md",
    "C38_EFFICIENCY_*.md",
]

DIRS = [
    "src",
    "tests",
    "paper",
    "results",
    "logs",
    "server_scripts",
]

EXCLUDE_PREFIXES = (
    "_A_",
    "_c2_",
    "_copy_safe_smoke",
    "_repair_smoke",
    "_smoke",
    "_s",
)

EXCLUDE_SUFFIXES = (
    ".pyc",
    ".pyo",
)

EXCLUDE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    DEFAULT_NAME,
    "C34_11H_SERVER_RUNBOOK.md",
    "C34_36H_CHECKPOINTED_RUNBOOK.md",
    "C34538_COMBO_RUNBOOK.md",
}

EXCLUDE_EXTS = {
    ".aux",
    ".out",
    ".toc",
    ".fls",
    ".fdb_latexmk",
    ".synctex.gz",
}

KEEP_TARBALLS = {
    "NAR_C2_copysafe_hotpatch_20260606.tar.gz",
    "NAR_C34538_combo_hotpatch_20260610.tar.gz",
    "NAR_C42_noisy_prev_hotpatch_20260612.tar.gz",
    "NAR_C42_noisy_prev_results_20260614_121446.tar.gz",
    "NAR_C43_self_correction_hotpatch_20260614.tar.gz",
    "NAR_C43_self_correction_results_20260616_115903.tar.gz",
    "NAR_C44_focused_stress18h_results_20260617_122748.tar.gz",
    "NAR_C45_6d_extended_stress_hotpatch_20260617.tar.gz",
    "NAR_C45_6d_extended_stress_results_20260623_160741.tar.gz",
}


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    name = path.name
    if len(parts) >= 2 and parts[0] == "server_scripts" and path.suffix == ".md":
        return True
    if any(part in EXCLUDE_NAMES for part in parts):
        return True
    if any(name.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return True
    if any(part.startswith("_server_B_unpack") or part.startswith("_server_C2_unpack") for part in parts):
        return True
    if name.startswith("_render_"):
        return True
    if name.endswith(EXCLUDE_SUFFIXES):
        return True
    if path.suffix in EXCLUDE_EXTS:
        return True
    if path.suffixes[-2:] == [".tar", ".gz"] or name.endswith(".tar.gz"):
        return name not in KEEP_TARBALLS
    return False


def collect_files() -> list[Path]:
    files: set[Path] = set()
    for rel in ROOT_FILES:
        p = ROOT / rel
        if p.exists() and not is_excluded(p):
            files.add(p)
    for pattern in ROOT_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file() and not is_excluded(p):
                files.add(p)
    for rel_dir in DIRS:
        d = ROOT / rel_dir
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and not is_excluded(p):
                files.add(p)
    return sorted(files, key=lambda p: str(p.relative_to(ROOT)).lower())


def build_manifest(files: list[Path]) -> str:
    lines = [
        "NAR submission artifact bundle",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"File count: {len(files)}",
        "",
    ]
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        lines.append(f"{rel}\t{p.stat().st_size}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_NAME)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = collect_files()
    manifest = build_manifest(files)
    out = ROOT / args.out
    if args.dry_run:
        print(manifest)
        return

    if out.exists():
        out.unlink()
    with tarfile.open(out, "w:gz") as tf:
        for p in files:
            tf.add(p, arcname=f"NAR/{p.relative_to(ROOT).as_posix()}")
        info = tarfile.TarInfo("NAR/SUBMISSION_MANIFEST.tsv")
        data = manifest.encode("utf-8")
        info.size = len(data)
        info.mtime = datetime.now().timestamp()
        tf.addfile(info, fileobj=__import__("io").BytesIO(data))
    print(f"wrote {out}")
    print(f"files {len(files)}")


if __name__ == "__main__":
    main()
