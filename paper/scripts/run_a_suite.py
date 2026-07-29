"""Run and archive the local theory-validation A-suite.

This wrapper executes exp_a1.py, exp_a2.py, exp_a4.py, exp_a5.py, and exp_a6.py,
captures their stdout in logs/a_suite.log, and writes parsed metrics to
results/a_suite.json.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "logs" / "a_suite.log"
OUT = ROOT / "results" / "a_suite.json"
SCRIPTS = ["exp_a1.py", "exp_a2.py", "exp_a4.py", "exp_a5.py", "exp_a6.py"]


def run_script(name: str):
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT / name)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=7200,
    )
    elapsed = time.time() - t0
    return {
        "script": name,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_a1(txt: str):
    sections = {}
    current = None
    for line in txt.splitlines():
        if "training distribution" in line:
            current = "train_er_p03"
            sections[current] = {}
            continue
        if "fixed avg-degree" in line:
            current = "fixed_avg_degree_6"
            sections[current] = {}
            continue
        m = re.match(r"\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)%", line)
        if current and m:
            n, mean, std, pct = m.groups()
            sections[current][n] = {
                "affected_mean": float(mean),
                "affected_std": float(std),
                "affected_pct": float(pct),
            }
    return sections


def parse_a2(txt: str):
    buckets = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(0|1-2|3-5|6-10|11\+): node-acc=([0-9.]+)\s+\(n=(\d+)\)", line)
        if m:
            b, acc, count = m.groups()
            buckets[b] = {"node_acc": float(acc), "count": int(count)}
    rho = None
    m = re.search(r"gate recall rho = ([0-9.]+)\s+\(over (\d+) samples\)", txt)
    if m:
        rho = {"mean": float(m.group(1)), "samples": int(m.group(2))}
    return {"buckets": buckets, "gate_recall": rho}


def parse_a4(txt: str):
    rows = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)\s+\|\s+([0-9.]+)\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+([0-9.]+)", line)
        if m:
            st, d16, d64, r16, r64 = m.groups()
            rows[st] = {
                "DeltaNAR_n16": float(d16),
                "DeltaNAR_n64": float(d64),
                "Recompute_n16": float(r16),
                "Recompute_n64": float(r64),
            }
    return rows


def parse_a5(txt: str):
    scale, margin = {}, {}
    mode = None
    for line in txt.splitlines():
        if "copy_scale" in line:
            mode = "scale"
        elif "copy_margin" in line:
            mode = "margin"
        m = re.search(r"(scale|margin)=\s*(\d+):\s+([0-9.]+)", line)
        if m and mode:
            _, k, v = m.groups()
            (scale if mode == "scale" else margin)[k] = float(v)
    return {"copy_scale": scale, "copy_margin": margin}


def parse_a6(txt: str):
    rows = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(DeltaNAR|Recompute)\s+([0-9.]+)\s+([0-9.]+)", line)
        if m:
            model, in_dist, ood = m.groups()
            rows[model] = {"in_dist_weight": float(in_dist), "ood_add_delete": float(ood)}
    return rows


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    with open(LOG, "w", encoding="utf-8") as log:
        for script in SCRIPTS:
            print(f"[A-SUITE] running {script}", flush=True)
            res = run_script(script)
            runs.append(res)
            log.write(f"\n\n===== {script} | rc={res['returncode']} | {res['elapsed_sec']:.1f}s =====\n")
            log.write(res["stdout"])
            if res["stderr"]:
                log.write("\n--- STDERR ---\n")
                log.write(res["stderr"])
            log.flush()
            if res["returncode"] != 0:
                print(f"[A-SUITE] {script} failed with rc={res['returncode']}", flush=True)
                break

    by_script = {r["script"]: r for r in runs}
    parsed = {
        "runs": [
            {"script": r["script"], "returncode": r["returncode"], "elapsed_sec": r["elapsed_sec"]}
            for r in runs
        ],
    }
    if "exp_a1.py" in by_script:
        parsed["a1"] = parse_a1(by_script["exp_a1.py"]["stdout"])
    if "exp_a2.py" in by_script:
        parsed["a2"] = parse_a2(by_script["exp_a2.py"]["stdout"])
    if "exp_a4.py" in by_script:
        parsed["a4"] = parse_a4(by_script["exp_a4.py"]["stdout"])
    if "exp_a5.py" in by_script:
        parsed["a5"] = parse_a5(by_script["exp_a5.py"]["stdout"])
    if "exp_a6.py" in by_script:
        parsed["a6"] = parse_a6(by_script["exp_a6.py"]["stdout"])

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, sort_keys=True)
    print(f"[A-SUITE] wrote {OUT}")
    print(f"[A-SUITE] wrote {LOG}")


if __name__ == "__main__":
    main()
