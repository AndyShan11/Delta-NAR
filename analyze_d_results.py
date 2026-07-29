"""Parse D-line logs into structured reproducible artifacts.

D-line experiments were originally recorded as compact stdout tables. This
script turns rollout, generalist, dynamic-intensity, transfer, and MST records
into JSON/Markdown so figures and supplement tables do not depend on hand-copied
numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
SERVER_LOGS = ROOT / "_server_B_unpack" / "logs"
RESULTS = ROOT / "results"

OUT_JSON = RESULTS / "D_structured.json"
OUT_MD = ROOT / "D_RESULTS_SUMMARY.md"
OUT_TEX = ROOT / "paper" / "tables" / "d_structured_numbers.tex"


def floats(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+\.\d+", line)]


def source_glob(preferred_dir: Path, pattern: str, fallback_dir: Path = LOGS) -> list[Path]:
    preferred = sorted(preferred_dir.glob(pattern))
    if preferred:
        return preferred
    return sorted(fallback_dir.glob(pattern))


def mean_std(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=float)
    return [float(arr.mean()), float(arr.std())]


def nested_metric_agg(raw: dict) -> dict:
    out = {}
    for a, by_b in raw.items():
        out[a] = {}
        for b, by_metric in by_b.items():
            out[a][b] = {metric: mean_std(vals) for metric, vals in by_metric.items()}
    return out


def parse_rollout(paths: list[Path]) -> dict:
    raw: dict[str, list[list[float]]] = {"DeltaNAR": [], "Naive-d": []}
    for path in paths:
        in_rollout = False
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith("=== rollout"):
                in_rollout = True
                continue
            if not in_rollout:
                continue
            for model in list(raw):
                if line.startswith(model):
                    raw[model].append(floats(line))
    agg = {}
    for model, rows in raw.items():
        if not rows:
            continue
        arr = np.array(rows, dtype=float)
        agg[model] = {
            "mean": [float(x) for x in arr.mean(axis=0)],
            "std": [float(x) for x in arr.std(axis=0)],
        }
    return {"sources": [str(p.relative_to(ROOT)) for p in paths], "raw": raw, "agg": agg}


def parse_generalist(paths: list[Path]) -> dict:
    raw: dict[str, dict[str, dict[str, list[float]]]] = {
        "sssp": {},
        "bottleneck": {},
    }
    sizes = ["16", "32", "64", "128"]
    for path in paths:
        for line in path.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("sssp") or stripped.startswith("bottleneck"):
                task = stripped.split()[0]
                vals = floats(stripped)
                if len(vals) != 12:
                    continue
                for i, size in enumerate(sizes):
                    g, n, f1 = vals[3 * i : 3 * i + 3]
                    by_metric = raw[task].setdefault(size, {"graph_acc_v": [], "node_acc_v": [], "aff_f1": []})
                    by_metric["graph_acc_v"].append(g)
                    by_metric["node_acc_v"].append(n)
                    by_metric["aff_f1"].append(f1)
    return {
        "sources": [str(p.relative_to(ROOT)) for p in paths],
        "raw": raw,
        "agg": nested_metric_agg(raw),
    }


def parse_intensity(paths: list[Path]) -> dict:
    raw: dict[str, dict[str, dict[str, list[float]]]] = {}
    ks = ["1", "2", "4", "8"]
    for path in paths:
        in_intensity = False
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith("=== dynamic-intensity"):
                in_intensity = True
                continue
            if not in_intensity:
                continue
            stripped = line.strip()
            if not any(stripped.startswith(m) for m in ["DeltaNAR", "Naive-d", "Recompute"]):
                continue
            model = stripped.split()[0]
            vals = floats(stripped)
            if len(vals) < 8:
                continue
            for i, k in enumerate(ks):
                g, n = vals[2 * i : 2 * i + 2]
                by_metric = raw.setdefault(model, {}).setdefault(k, {"graph_acc_v": [], "node_acc_v": []})
                by_metric["graph_acc_v"].append(g)
                by_metric["node_acc_v"].append(n)
    return {
        "sources": [str(p.relative_to(ROOT)) for p in paths],
        "raw": raw,
        "agg": nested_metric_agg(raw),
    }


def parse_mst(paths: list[Path]) -> dict:
    raw: dict[str, dict[str, dict[str, list[float]]]] = {}
    sizes = ["16", "32", "64"]
    for path in paths:
        for line in path.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if not any(stripped.startswith(m) for m in ["DeltaNAR", "Naive-d", "Recompute"]):
                continue
            model = stripped.split()[0]
            vals = floats(stripped)
            if len(vals) != 9:
                continue
            for i, size in enumerate(sizes):
                g, e, f1 = vals[3 * i : 3 * i + 3]
                by_metric = raw.setdefault(model, {}).setdefault(size, {"graph_acc": [], "edge_acc": [], "aff_f1": []})
                by_metric["graph_acc"].append(g)
                by_metric["edge_acc"].append(e)
                by_metric["aff_f1"].append(f1)
    return {
        "sources": [str(p.relative_to(ROOT)) for p in paths],
        "raw": raw,
        "agg": nested_metric_agg(raw),
    }


def parse_transfer(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw: dict[str, dict[str, dict[str, list[float]]]] = {"zero_shot": {}, "few_shot": {}}
    for _seed, by_mode in data.items():
        for mode, by_size in by_mode.items():
            for size, metrics in by_size.items():
                by_metric = raw.setdefault(mode, {}).setdefault(size, {})
                for metric, value in metrics.items():
                    by_metric.setdefault(metric, []).append(float(value))
    return {
        "sources": [str(path.relative_to(ROOT))],
        "raw": raw,
        "agg": nested_metric_agg(raw),
    }


def make_paper_intensity(local: dict, b07_path: Path) -> dict:
    agg = json.loads(json.dumps(local["agg"]))
    sources = list(local["sources"])
    if b07_path.exists():
        b07 = json.loads(b07_path.read_text(encoding="utf-8"))
        sources.append(str(b07_path.relative_to(ROOT)))
        for model in ["DeltaNAR", "Naive-d", "Recompute"]:
            if model in b07["agg"]["sssp"]:
                agg.setdefault(model, {})["8"] = {
                    "node_acc_v": b07["agg"]["sssp"][model]["16"]["node_acc_v"],
                    "graph_acc_v": b07["agg"]["sssp"][model]["16"]["graph_acc_v"],
                }
    return {
        "sources": sources,
        "agg": agg,
        "note": "k=1/2/4 from local D_intensity logs; k=8 from server B07 matched k=8 rerun.",
    }


def pm(pair: list[float]) -> str:
    return f"{pair[0]:.3f} +/- {pair[1]:.3f}"


def tex_num(pair: list[float]) -> str:
    return f"{pair[0]:.3f}"


def write_latex(data: dict) -> None:
    intensity = data["intensity_paper"]["agg"]
    labels = {"DeltaNAR": r"\deltanar", "Naive-d": "Prev+Delta", "Recompute": "Recompute"}
    tex = [
        "% Auto-generated by analyze_d_results.py; do not edit numbers by hand.",
        r"\newcommand{\DIntensityRows}{%",
    ]
    for model in ["DeltaNAR", "Naive-d", "Recompute"]:
        cells = [tex_num(intensity[model][k]["node_acc_v"]) for k in ["1", "2", "4", "8"]]
        tex.append(labels[model] + " & " + " & ".join(cells) + r" \\")
    tex.append("}")
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")


def write_markdown(data: dict) -> None:
    lines = [
        "# D-Line Structured Results\n",
        "Structured parse of rollout, generalist, dynamic-intensity, transfer, and "
        "MST experiments. Values are mean +/- population std over parsed seeds.\n",
    ]

    rollout = data["rollout"]["agg"]
    lines.append("## Rollout\n")
    lines.append("| Model | t0 | t9 | t59 |")
    lines.append("|---|---:|---:|---:|")
    for model in ["DeltaNAR", "Naive-d"]:
        mean = rollout[model]["mean"]
        std = rollout[model]["std"]
        idxs = [0, 9, len(mean) - 1]
        cells = [f"{mean[i]:.3f} +/- {std[i]:.3f}" for i in idxs]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

    generalist = data["generalist"]["agg"]
    lines.append("## Generalist Node Accuracy\n")
    lines.append("| Task | n=16 | n=32 | n=64 | n=128 |")
    lines.append("|---|---:|---:|---:|---:|")
    for task in ["sssp", "bottleneck"]:
        cells = [pm(generalist[task][size]["node_acc_v"]) for size in ["16", "32", "64", "128"]]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines.append("")

    transfer = data["transfer"]["agg"]
    lines.append("## Transfer Node Accuracy\n")
    lines.append("| Mode | n=16 | n=64 |")
    lines.append("|---|---:|---:|")
    for mode in ["zero_shot", "few_shot"]:
        cells = [pm(transfer[mode][size]["node_acc_v"]) for size in ["16", "64"]]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines.append("")

    intensity = data["intensity_local"]["agg"]
    lines.append("## Dynamic Intensity Node Accuracy\n")
    lines.append("| Model | k=1 | k=2 | k=4 | k=8 |")
    lines.append("|---|---:|---:|---:|---:|")
    for model in ["DeltaNAR", "Naive-d", "Recompute"]:
        cells = [pm(intensity[model][k]["node_acc_v"]) for k in ["1", "2", "4", "8"]]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

    paper_intensity = data["intensity_paper"]["agg"]
    lines.append("## Paper Dynamic Intensity Node Accuracy\n")
    lines.append(data["intensity_paper"]["note"] + "\n")
    lines.append("| Model | k=1 | k=2 | k=4 | k=8 |")
    lines.append("|---|---:|---:|---:|---:|")
    for model in ["DeltaNAR", "Naive-d", "Recompute"]:
        cells = [pm(paper_intensity[model][k]["node_acc_v"]) for k in ["1", "2", "4", "8"]]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

    mst = data["mst"]["agg"]
    lines.append("## MST\n")
    for metric, label in [("edge_acc", "Edge Accuracy"), ("graph_acc", "Graph Accuracy")]:
        lines.append(f"### {label}\n")
        lines.append("| Model | n=16 | n=32 | n=64 |")
        lines.append("|---|---:|---:|---:|")
        for model in ["DeltaNAR", "Naive-d", "Recompute"]:
            cells = [pm(mst[model][size][metric]) for size in ["16", "32", "64"]]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rollout_paths = source_glob(SERVER_LOGS, "ROLL13_s*.log")
    generalist_paths = source_glob(SERVER_LOGS, "GEN8_s*.log")
    intensity_paths = source_glob(LOGS, "D_intensity_s*.log")
    mst_paths = source_glob(LOGS, "D_mst_s*.log")
    transfer_path = RESULTS / "TRANS11.json"

    intensity_local = parse_intensity(intensity_paths)
    data = {
        "rollout": parse_rollout(rollout_paths),
        "generalist": parse_generalist(generalist_paths),
        "intensity_local": intensity_local,
        "intensity_paper": make_paper_intensity(intensity_local, RESULTS / "B07_sssp_k8_core_diag.json"),
        "mst": parse_mst(mst_paths),
        "transfer": parse_transfer(transfer_path),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_markdown(data)
    write_latex(data)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
