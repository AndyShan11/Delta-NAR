"""Generate paper figures from reproducible result artifacts.

Run from the repository root:
  D:/dev/python/python3.12.4/python.exe paper/scripts/make_figures.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
FIGS = ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "DeltaNAR": "#2f6fbb",
    "no-copy": "#d55e00",
    "Naive-d": "#7f7f7f",
    "Recomp+gate": "#cc79a7",
    "Recompute": "#b22222",
    "Attn": "#009e73",
    "Triplet": "#6a3d9a",
    "Oracle": "#0072b2",
    "PureCopy": "#56b4e9",
    "copy-safe": "#009e73",
    "self-correct": "#6a3d9a",
}

LABELS = {
    "DeltaNAR": "DeltaNAR",
    "no-copy": "no-copy",
    "Naive-d": "Prev+Delta",
    "Recomp+gate": "Recomp+gate",
    "Recompute": "Recompute",
    "Attn": "Attn",
    "Triplet": "Triplet",
    "raw_no_copy": "raw no-copy",
    "predicted_copy": "predicted copy",
    "oracle_affected_copy": "oracle affected",
    "oracle_copy_safe_upper": "oracle copy-safe",
    "pure_copy": "pure copy",
    "copy-safe": "copy-safe",
    "self-correct": "self-correct",
}


def load_json(name: str):
    with open(RESULTS / name, "r", encoding="utf-8") as f:
        return json.load(f)


def agg_metric(name: str, task: str, model: str, metric: str, sizes):
    d = load_json(name)["agg"][task][model]
    ys, es = [], []
    for n in sizes:
        m, s = d[str(n)][metric]
        ys.append(m)
        es.append(s)
    return np.array(ys), np.array(es)


def maybe_metric(files, task, model, metric, sizes):
    out = {}
    for fn in files:
        p = RESULTS / fn
        if not p.exists():
            continue
        d = load_json(fn)["agg"].get(task, {})
        if model not in d:
            continue
        for n, md in d[model].items():
            if int(n) in sizes and metric in md:
                out[int(n)] = md[metric]
    ys, es = [], []
    for n in sizes:
        if n in out:
            ys.append(out[n][0])
            es.append(out[n][1])
        else:
            ys.append(np.nan)
            es.append(0.0)
    return np.array(ys), np.array(es)


def save(fig, name, tight: bool = True):
    if tight:
        fig.tight_layout()
    output = FIGS / name
    fig.savefig(output, bbox_inches="tight")
    if output.suffix.lower() == ".pdf":
        png_output = output.with_name(f"{output.stem}_aaai27.png")
        fig.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def metric_value(name: str, task: str, model: str, size: int, metric: str):
    val = load_json(name)["agg"][task][model][str(size)][metric]
    return val[0], val[1]


def load_d_structured():
    p = RESULTS / "D_structured.json"
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def fig_size_ood():
    sizes = [16, 32, 64, 128, 256]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.2), sharey=True)
    panels = [
        ("SSSP", "sssp", "B01_sssp_k1_core_diag.json", "B02_sssp_triplet_diag.json"),
        ("Bottleneck", "bottleneck", "B03_bottleneck_k1_core_diag.json", "B09_bottleneck_triplet_diag.json"),
    ]
    for ax, (title, task, base_file, triplet_file) in zip(axs, panels):
        y, e = agg_metric("B11_main_deltanar_20seed.json", task, "DeltaNAR", "node_acc_v", sizes)
        ax.errorbar(sizes, y, yerr=e, marker="o", linewidth=2.4, color=COLORS["DeltaNAR"], label="DeltaNAR (20 seeds)")
        for model in ["Naive-d", "Recomp+gate", "Recompute", "Attn"]:
            y, e = maybe_metric([base_file], task, model, "node_acc_v", sizes)
            ax.plot(sizes, y, marker="o", linewidth=1.55, color=COLORS[model], label=LABELS[model])
        y, e = maybe_metric([triplet_file], task, "Triplet", "node_acc_v", sizes)
        ax.plot(sizes, y, marker="s", linewidth=1.55, color=COLORS["Triplet"], label="Triplet")
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes], rotation=0)
        ax.set_ylim(0.0, 1.04)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("test graph size n")
    axs[0].set_ylabel("node_acc_v")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=3, fontsize=7, frameon=False)
    fig.subplots_adjust(top=0.76, wspace=0.12)
    save(fig, "fig_size_ood.pdf", tight=False)


def fig_ablation():
    sizes = [16, 32, 64, 128]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, task in zip(axs, ["sssp", "bottleneck"]):
        for model, color in [
            ("DeltaNAR", "#2563eb"),
            ("no-copy", "#dc2626"),
            ("min-only", "#059669"),
            ("Recompute", "#64748b"),
        ]:
            y, _ = agg_metric("ABL7.json", task, model, "node_acc_v", sizes)
            ax.plot(sizes, y, marker="o", linewidth=2 if model == "DeltaNAR" else 1.5, color=color, label=model)
        ax.set_title(task.upper() if task == "sssp" else "Bottleneck")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("test graph size n")
    axs[0].set_ylabel("node_acc_v")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, fontsize=7, frameon=False)
    fig.subplots_adjust(top=0.78, wspace=0.12)
    save(fig, "fig_ablation.pdf", tight=False)


def fig_ood():
    labels = ["ER", "BA", "Grid", "Wide weights", "Heavy tail", "Edge delete"]
    delta = [
        agg_metric("A_sssp.json", "sssp", "DeltaNAR", "node_acc_v", [64])[0][0],
        agg_metric("B_sssp_ba.json", "sssp", "DeltaNAR", "node_acc_v", [64])[0][0],
        agg_metric("B_sssp_grid.json", "sssp", "DeltaNAR", "node_acc_v", [64])[0][0],
        agg_metric("C_sssp_whi100.json", "sssp", "DeltaNAR", "node_acc_v", [64])[0][0],
        agg_metric("C_sssp_heavy.json", "sssp", "DeltaNAR", "node_acc_v", [64])[0][0],
        agg_metric("OOD10_failure.json", "sssp", "DeltaNAR", "node_acc_v", [64])[0][0],
    ]
    recomp = [
        agg_metric("A_sssp.json", "sssp", "Recompute", "node_acc_v", [64])[0][0],
        agg_metric("B_sssp_ba.json", "sssp", "Recompute", "node_acc_v", [64])[0][0],
        agg_metric("B_sssp_grid.json", "sssp", "Recompute", "node_acc_v", [64])[0][0],
        agg_metric("C_sssp_whi100.json", "sssp", "Recompute", "node_acc_v", [64])[0][0],
        agg_metric("C_sssp_heavy.json", "sssp", "Recompute", "node_acc_v", [64])[0][0],
        agg_metric("OOD10_failure.json", "sssp", "Recompute", "node_acc_v", [64])[0][0],
    ]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.bar(x - w / 2, delta, width=w, color="#2563eb", label="DeltaNAR")
    ax.bar(x + w / 2, recomp, width=w, color="#dc2626", label="Recompute")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("node_acc_v @ n=64")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, fontsize=8)
    save(fig, "fig_ood.pdf")


def parse_a_suite():
    p = RESULTS / "a_suite.json"
    if p.exists():
        return json.load(open(p, "r", encoding="utf-8"))
    return {
        "a1": {
            "train_er_p03": {
                "16": {"affected_mean": 8.32, "affected_pct": 52.0},
                "32": {"affected_mean": 5.92, "affected_pct": 18.5},
                "64": {"affected_mean": 3.20, "affected_pct": 5.0},
                "128": {"affected_mean": 1.28, "affected_pct": 1.0},
                "256": {"affected_mean": 0.01, "affected_pct": 0.1},
            }
        },
        "a2": {
            "buckets": {
                "0": {"node_acc": 0.998},
                "1-2": {"node_acc": 0.970},
                "3-5": {"node_acc": 0.922},
                "6-10": {"node_acc": 0.882},
                "11+": {"node_acc": 0.793},
            }
        },
    }


def fig_theory():
    data = parse_a_suite()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    a1 = data["a1"]["train_er_p03"]
    sizes = sorted(int(k) for k in a1)
    pct = [a1[str(n)]["affected_pct"] for n in sizes]
    axs[0].plot(sizes, pct, marker="o", color="#2563eb")
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(sizes)
    axs[0].set_xticklabels([str(s) for s in sizes])
    axs[0].set_ylabel("|A_t| / n (%)")
    axs[0].set_xlabel("graph size n")
    axs[0].set_title("Affected fraction shrinks")
    axs[0].grid(True, axis="y", alpha=0.25)

    buckets = ["0", "1-2", "3-5", "6-10", "11+"]
    acc = [data["a2"]["buckets"][b]["node_acc"] for b in buckets]
    axs[1].bar(np.arange(len(buckets)), acc, color="#f97316")
    axs[1].set_ylim(0.72, 1.01)
    axs[1].set_xticks(np.arange(len(buckets)))
    axs[1].set_xticklabels(buckets)
    axs[1].set_xlabel("|A_t| bucket")
    axs[1].set_ylabel("node_acc_v")
    axs[1].set_title("Accuracy tracks affected size")
    axs[1].grid(True, axis="y", alpha=0.25)
    save(fig, "fig_theory_a_suite.pdf")


def parse_rollout():
    c43 = RESULTS / "C43_self_correction_20seed_merged.json"
    if c43.exists():
        d = json.load(open(c43, "r", encoding="utf-8"))["rollout_agg"]
        out = {}
        for src, label in [("copy_safe", "copy-safe"), ("self_correct", "self-correct")]:
            steps = sorted((int(k) for k in d[src]["64"]), key=int)
            mu = np.array([d[src]["64"][str(t)][0] for t in steps], dtype=float)
            sd = np.array([d[src]["64"][str(t)][1] for t in steps], dtype=float)
            out[label] = np.array([mu, sd], dtype=float)
        return out
    d = load_d_structured()
    if d and "rollout" in d:
        return {
            model: np.array([stats["mean"], stats["std"]], dtype=float)
            for model, stats in d["rollout"]["agg"].items()
        }
    vals = {"DeltaNAR": [], "Naive-d": []}
    for p in sorted(LOGS.glob("ROLL13_s*.log")):
        lines = p.read_text(errors="ignore").splitlines()
        in_roll = False
        for line in lines:
            if line.startswith("=== rollout"):
                in_roll = True
                continue
            if not in_roll:
                continue
            for model in vals:
                if line.startswith(model):
                    nums = [float(x) for x in re.findall(r"\d\.\d{3}", line)]
                    vals[model].append(nums)
    return {k: np.array(v, dtype=float) for k, v in vals.items() if v}


def fig_rollout():
    data = parse_rollout()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    if "copy-safe" in data:
        series = [("copy-safe", COLORS["copy-safe"]), ("self-correct", COLORS["self-correct"])]
    else:
        series = [("DeltaNAR", "#2563eb"), ("Naive-d", "#dc2626")]
    for model, color in series:
        arr = data[model]
        if arr.ndim == 2 and arr.shape[0] == 2:
            mu, sd = arr[0], arr[1]
        else:
            mu, sd = arr.mean(axis=0), arr.std(axis=0)
        t = np.arange(len(mu))
        ax.plot(t, mu, color=color, linewidth=2, label=LABELS.get(model, model))
        ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("node_acc_v @ n=64")
    ax.set_ylim(0, 1.04)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "fig_rollout.pdf")


def fig_generalist_transfer():
    d = load_d_structured()
    sizes = np.array([16, 32, 64, 128])
    if d:
        gen = d["generalist"]["agg"]
        gen_sssp = np.array([gen["sssp"][str(n)]["node_acc_v"][0] for n in sizes])
        gen_bot = np.array([gen["bottleneck"][str(n)]["node_acc_v"][0] for n in sizes])
        transfer_raw = d["transfer"]["raw"]
        transfer = {
            "zero-shot": transfer_raw["zero_shot"]["64"]["node_acc_v"],
            "few-shot": transfer_raw["few_shot"]["64"]["node_acc_v"],
        }
    else:
        gen_sssp = np.array([0.978, 0.986, 0.993, 0.997])
        gen_bot = np.array([0.956, 0.955, 0.961, 0.965])
        transfer = {
            "zero-shot": [0.413, 0.580, 0.444],
            "few-shot": [0.991, 0.993, 0.990],
        }
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axs[0].plot(sizes, gen_sssp, marker="o", color="#2563eb", label="SSSP")
    axs[0].plot(sizes, gen_bot, marker="o", color="#f97316", label="Bottleneck")
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks(sizes)
    axs[0].set_xticklabels([str(s) for s in sizes])
    axs[0].set_ylim(0.9, 1.01)
    axs[0].set_title("Generalist", pad=32)
    axs[0].set_xlabel("test graph size n")
    axs[0].set_ylabel("node_acc_v")
    axs[0].grid(True, axis="y", alpha=0.25)
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  frameon=False, fontsize=8)

    x = np.arange(3)
    axs[1].bar(x - 0.18, transfer["zero-shot"], width=0.36, color="#94a3b8", label="zero-shot")
    axs[1].bar(x + 0.18, transfer["few-shot"], width=0.36, color="#2563eb", label="few-shot")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(["seed0", "seed1", "seed2"])
    axs[1].set_ylim(0, 1.04)
    axs[1].set_title("SSSP -> Bottleneck", pad=32)
    axs[1].set_ylabel("node_acc_v @ n=64")
    axs[1].grid(True, axis="y", alpha=0.25)
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  frameon=False, fontsize=8)
    save(fig, "fig_generalist_transfer.pdf")


def fig_extreme_mst():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    labels = ["SSSP\n512", "Bottleneck\n512", "Bottleneck\n1024"]
    delta = [0.998, 0.937, 0.956]
    recomp = [0.029, 0.123, 0.107]
    x = np.arange(len(labels))
    w = 0.36
    axs[0].bar(x - w / 2, delta, width=w, color="#2563eb", label="DeltaNAR")
    axs[0].bar(x + w / 2, recomp, width=w, color="#dc2626", label="Recompute")
    axs[0].set_ylim(0, 1.04)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels)
    axs[0].set_ylabel("node_acc_v")
    axs[0].set_title("Extreme size", pad=32)
    axs[0].grid(True, axis="y", alpha=0.25)
    axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  frameon=False, fontsize=8)

    sizes = np.array([16, 32, 64])
    d = load_d_structured()
    if d:
        mst = d["mst"]["agg"]["DeltaNAR"]
        edge = np.array([mst[str(n)]["edge_acc"][0] for n in sizes])
        graph = np.array([mst[str(n)]["graph_acc"][0] for n in sizes])
    else:
        edge = np.array([0.983, 0.989, 0.938])
        graph = np.array([0.278, 0.125, 0.0])
    axs[1].plot(sizes, edge, marker="o", color="#2563eb", label="edge acc.")
    axs[1].plot(sizes, graph, marker="o", color="#f97316", label="graph acc.")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xticks(sizes)
    axs[1].set_xticklabels([str(s) for s in sizes])
    axs[1].set_ylim(0, 1.04)
    axs[1].set_title("MST difficulty", pad=32)
    axs[1].set_xlabel("test graph size n")
    axs[1].grid(True, axis="y", alpha=0.25)
    axs[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
                  frameon=False, fontsize=8)
    save(fig, "fig_extreme_mst.pdf")


def fig_b_server_overview():
    sizes = [16, 64, 128, 256]
    panels = [
        ("SSSP, one edit", "B01_sssp_k1_core_diag.json", "sssp", ["DeltaNAR", "Naive-d", "Recomp+gate", "Recompute", "Attn"]),
        ("Bottleneck, one edit", "B03_bottleneck_k1_core_diag.json", "bottleneck", ["DeltaNAR", "Naive-d", "Recomp+gate", "Recompute", "Attn"]),
        ("SSSP, eight edits", "B07_sssp_k8_core_diag.json", "sssp", ["DeltaNAR", "no-copy", "Recompute", "Attn"]),
        ("Bottleneck, four edits", "B08_bottleneck_k4_core_diag.json", "bottleneck", ["DeltaNAR", "no-copy", "Recompute", "Attn"]),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(8.8, 5.6), sharey=True)
    for ax, (title, fn, task, models) in zip(axs.ravel(), panels):
        for model in models:
            y, e = agg_metric(fn, task, model, "node_acc_v", sizes)
            ax.errorbar(
                sizes,
                y,
                yerr=e,
                marker="o",
                linewidth=2.2 if model == "DeltaNAR" else 1.35,
                markersize=4,
                color=COLORS[model],
                label=LABELS[model],
                alpha=1.0 if model == "DeltaNAR" else 0.9,
            )
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_ylim(0, 1.04)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlabel("test graph size n")
    axs[0, 0].set_ylabel("node_acc_v")
    axs[1, 0].set_ylabel("node_acc_v")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=5, frameon=False, fontsize=8)
    save(fig, "fig_b_server_overview.pdf")


def fig_b_diagnostic_heatmap():
    specs = [
        ("SSSP k=1", "B01_sssp_k1_core_diag.json", "sssp", 256),
        ("SSSP k=8", "B07_sssp_k8_core_diag.json", "sssp", 256),
        ("Bottleneck k=1", "B03_bottleneck_k1_core_diag.json", "bottleneck", 256),
        ("Bottleneck k=4", "B08_bottleneck_k4_core_diag.json", "bottleneck", 256),
    ]
    models = ["DeltaNAR", "no-copy", "Naive-d", "Recomp+gate", "Recompute", "Attn"]
    metrics = [
        ("node_acc_v", "node"),
        ("graph_acc_v", "graph"),
        ("changed_output_acc_v", "changed"),
        ("copy_critical_acc_v", "copy-critical"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(10.4, 3.2), sharey=True)
    for ax, (title, fn, task, size) in zip(axs, specs):
        data = np.zeros((len(models), len(metrics)))
        for i, model in enumerate(models):
            for j, (metric, _) in enumerate(metrics):
                data[i, j] = metric_value(fn, task, model, size, metric)[0]
        im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([m[1] for m in metrics], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels([LABELS[m] for m in models])
        for i in range(len(models)):
            for j in range(len(metrics)):
                txt_color = "white" if data[i, j] < 0.55 else "black"
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6.6, color=txt_color)
    fig.subplots_adjust(right=0.88, wspace=0.18)
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.84, pad=0.015)
    cbar.set_label("accuracy")
    save(fig, "fig_b_diagnostic_heatmap.pdf", tight=False)


def fig_b_oracle_copy():
    variants = ["raw_no_copy", "predicted_copy", "oracle_affected_copy", "oracle_copy_safe_upper", "pure_copy"]
    panels = [("SSSP", "sssp", 128), ("Bottleneck", "bottleneck", 128)]
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
    for ax, (title, task, size) in zip(axs, panels):
        node = [metric_value("B10_oracle_sssp_bottleneck.json", task, v, size, "node_acc_v")[0] for v in variants]
        changed = [metric_value("B10_oracle_sssp_bottleneck.json", task, v, size, "changed_output_acc_v")[0] for v in variants]
        x = np.arange(len(variants))
        ax.bar(x - 0.18, node, width=0.36, color="#2f6fbb", label="node")
        ax.bar(x + 0.18, changed, width=0.36, color="#e69f00", label="changed")
        ax.set_title(f"{title}, n={size}")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[v] for v in variants], rotation=28, ha="right")
        ax.set_ylim(0, 1.04)
        ax.grid(True, axis="y", alpha=0.25)
    axs[0].set_ylabel("accuracy")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=8)
    save(fig, "fig_b_oracle_copy.pdf")


def fig_b_ood_extreme():
    labels = ["BA", "wide\nweights", "edge\ndelete", "extreme\n512"]
    delta = [
        metric_value("B04_ood_topology_ba_sssp.json", "sssp", "DeltaNAR", 64, "node_acc_v")[0],
        metric_value("B05_ood_weight_wide_sssp.json", "sssp", "DeltaNAR", 64, "node_acc_v")[0],
        metric_value("B12_ood_failure_sssp_bottleneck.json", "sssp", "DeltaNAR", 64, "node_acc_v")[0],
        metric_value("B06_sssp_extreme_512.json", "sssp", "DeltaNAR", 512, "node_acc_v")[0],
    ]
    recomp = [
        metric_value("B04_ood_topology_ba_sssp.json", "sssp", "Recompute", 64, "node_acc_v")[0],
        metric_value("B05_ood_weight_wide_sssp.json", "sssp", "Recompute", 64, "node_acc_v")[0],
        metric_value("B12_ood_failure_sssp_bottleneck.json", "sssp", "Recompute", 64, "node_acc_v")[0],
        metric_value("B06_sssp_extreme_512.json", "sssp", "Recompute", 512, "node_acc_v")[0],
    ]
    bot_labels = ["edge\ndelete", "extreme\n512", "extreme\n1024"]
    bot_delta = [
        metric_value("B12_ood_failure_sssp_bottleneck.json", "bottleneck", "DeltaNAR", 64, "node_acc_v")[0],
        metric_value("B13_bottleneck_extreme_512_1024.json", "bottleneck", "DeltaNAR", 512, "node_acc_v")[0],
        metric_value("B13_bottleneck_extreme_512_1024.json", "bottleneck", "DeltaNAR", 1024, "node_acc_v")[0],
    ]
    bot_recomp = [
        metric_value("B12_ood_failure_sssp_bottleneck.json", "bottleneck", "Recompute", 64, "node_acc_v")[0],
        metric_value("B13_bottleneck_extreme_512_1024.json", "bottleneck", "Recompute", 512, "node_acc_v")[0],
        metric_value("B13_bottleneck_extreme_512_1024.json", "bottleneck", "Recompute", 1024, "node_acc_v")[0],
    ]
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, title, labs, a, b in [
        (axs[0], "SSSP stress", labels, delta, recomp),
        (axs[1], "Bottleneck stress", bot_labels, bot_delta, bot_recomp),
    ]:
        x = np.arange(len(labs))
        w = 0.36
        ax.bar(x - w / 2, a, width=w, color=COLORS["DeltaNAR"], label="DeltaNAR")
        ax.bar(x + w / 2, b, width=w, color=COLORS["Recompute"], label="Recompute")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
        ax.set_ylim(0, 1.04)
        ax.grid(True, axis="y", alpha=0.25)
    axs[0].set_ylabel("node_acc_v")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=8)
    save(fig, "fig_b_ood_extreme.pdf")


def fig_ahr_hard_repair():
    files = {
        "sssp": ("AHR_sssp_train4_test12_3seed.json", "AHR_oracle_sssp_train4_test12_3seed.json", "SSSP"),
        "bottleneck": ("AHR_bottleneck_train4_test12_3seed.json", "AHR_oracle_bottleneck_train4_test12_3seed.json", "Bottleneck"),
    }
    if any(not (RESULTS / f).exists() for pair in files.values() for f in pair[:2]):
        return
    sizes = [16, 32, 64]
    fig, axs = plt.subplots(2, 3, figsize=(10.4, 5.7), sharex="col")
    variants = ["raw_no_copy", "predicted_copy", "oracle_affected_copy", "oracle_copy_safe_upper", "pure_copy"]
    for row, (task, (hard_file, oracle_file, title)) in enumerate(files.items()):
        for model in ["DeltaNAR", "no-copy", "Naive-d", "Recompute"]:
            y, e = agg_metric(hard_file, task, model, "node_acc_v", sizes)
            axs[row, 0].errorbar(sizes, y, yerr=e, marker="o", linewidth=2 if model == "DeltaNAR" else 1.25,
                                 color=COLORS[model], label=LABELS[model])
            y, e = agg_metric(hard_file, task, model, "changed_output_acc_v", sizes)
            axs[row, 1].errorbar(sizes, y, yerr=e, marker="o", linewidth=2 if model == "DeltaNAR" else 1.25,
                                 color=COLORS[model], label=LABELS[model])
        vals = [metric_value(oracle_file, task, v, 64, "changed_output_acc_v")[0] for v in variants]
        x = np.arange(len(variants))
        axs[row, 2].bar(x, vals, color=["#9ca3af", COLORS["DeltaNAR"], "#56b4e9", "#0072b2", "#e69f00"])
        axs[row, 0].set_ylabel(f"{title}\nnode_acc_v")
        axs[row, 1].set_ylabel("changed acc.")
        axs[row, 2].set_ylabel("changed acc.")
        for col, panel_title in enumerate(["Full-node accuracy", "Changed-output repair", "Oracle copy @ n=64"]):
            axs[row, col].set_title(panel_title if row == 0 else "")
            axs[row, col].set_ylim(0, 1.04)
            axs[row, col].grid(True, axis="y", alpha=0.25)
        for col in [0, 1]:
            axs[row, col].set_xscale("log", base=2)
            axs[row, col].set_xticks(sizes)
            axs[row, col].set_xticklabels([str(s) for s in sizes])
            axs[row, col].set_xlabel("test graph size n")
        axs[row, 2].set_xticks(x)
        axs[row, 2].set_xticklabels([LABELS[v] for v in variants], rotation=32, ha="right")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.36, 1.03), ncol=4, frameon=False, fontsize=8)
    save(fig, "fig_ahr_hard_repair.pdf")


def fig_copy_safe_gate_candidate():
    files = {
        "sssp": ("AHR_sssp_train4_test12_3seed.json", "AHR_copysafegate_sssp_3seed.json", "SSSP"),
        "bottleneck": (
            "AHR_bottleneck_train4_test12_3seed.json",
            "AHR_copysafegate_bottleneck_3seed.json",
            "Bottleneck",
        ),
    }
    if any(not (RESULTS / f).exists() for pair in files.values() for f in pair[:2]):
        return
    metrics = [
        ("node_acc_v", "node"),
        ("graph_acc_v", "graph"),
        ("changed_output_acc_v", "changed"),
        ("copy_critical_acc_v", "copy-critical"),
        ("gate_recall", "gate recall"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.3), sharey=True)
    for ax, (task, (base_file, safe_file, title)) in zip(axs, files.items()):
        base = [metric_value(base_file, task, "DeltaNAR", 64, metric)[0] for metric, _ in metrics]
        base_err = [metric_value(base_file, task, "DeltaNAR", 64, metric)[1] for metric, _ in metrics]
        safe = [metric_value(safe_file, task, "DeltaNAR+repair", 64, metric)[0] for metric, _ in metrics]
        safe_err = [metric_value(safe_file, task, "DeltaNAR+repair", 64, metric)[1] for metric, _ in metrics]
        x = np.arange(len(metrics))
        w = 0.36
        ax.bar(x - w / 2, base, width=w, yerr=base_err, color=COLORS["DeltaNAR"], label="baseline")
        ax.bar(x + w / 2, safe, width=w, yerr=safe_err, color="#009e73", label="copy-safe gate")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], rotation=28, ha="right")
        ax.set_ylim(0, 1.04)
        ax.grid(True, axis="y", alpha=0.25)
    axs[0].set_ylabel("accuracy @ n=64")
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=8)
    save(fig, "fig_copy_safe_gate_candidate.pdf")


def fig_c2_copy_safe_server():
    fn = "C2_copysafe_20seed_merged.json"
    if not (RESULTS / fn).exists():
        return
    sizes = [16, 32, 64, 128]
    tasks = [("sssp", "SSSP"), ("bottleneck", "Bottleneck")]
    fig, axs = plt.subplots(2, 2, figsize=(8.4, 5.4), sharex=True, sharey=True)
    for row, (task, title) in enumerate(tasks):
        for metric, label, color, marker in [
            ("node_acc_v", "node", COLORS["copy-safe"], "o"),
            ("graph_acc_v", "graph", "#d55e00", "s"),
        ]:
            y, e = agg_metric(fn, task, "DeltaNAR+copy-safe", metric, sizes)
            axs[row, 0].errorbar(sizes, y, yerr=e, marker=marker, lw=2.0, capsize=3, color=color, label=label)
        for metric, label, color, marker in [
            ("changed_output_acc_v", "changed", "#cc79a7", "o"),
            ("copy_critical_acc_v", "copy-critical", "#0072b2", "^"),
        ]:
            y, e = agg_metric(fn, task, "DeltaNAR+copy-safe", metric, sizes)
            axs[row, 1].errorbar(sizes, y, yerr=e, marker=marker, lw=2.0, capsize=3, color=color, label=label)
        axs[row, 0].set_title(f"{title}: overall")
        axs[row, 1].set_title(f"{title}: repair slices")
        for ax in axs[row]:
            ax.set_ylim(0, 1.04)
            ax.set_xticks(sizes)
            ax.grid(True, alpha=0.25)
    axs[0, 0].set_ylabel("accuracy")
    axs[1, 0].set_ylabel("accuracy")
    axs[1, 0].set_xlabel("test graph size n")
    axs[1, 1].set_xlabel("test graph size n")
    h0, l0 = axs[0, 0].get_legend_handles_labels()
    h1, l1 = axs[0, 1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, fontsize=8, frameon=False)
    fig.subplots_adjust(top=0.88, hspace=0.28, wspace=0.12)
    save(fig, "fig_c2_copy_safe_server.pdf", tight=False)


def _pair3(data, path):
    cur = data
    for key in path:
        cur = cur[key]
    return float(cur[0]), float(cur[1])


def fig_c34538_stress_repair():
    needed = [
        "C34_36h_c3_size_20seed_merged.json",
        "C34_36h_c4_wide_20seed_merged.json",
        "C35_36h_c3_size_repaironly_20seed_merged.json",
        "C35_36h_c4_wide_repaironly_20seed_merged.json",
    ]
    if any(not (RESULTS / fn).exists() for fn in needed):
        return
    c34_c3 = load_json(needed[0])
    c34_c4 = load_json(needed[1])
    c35_c3 = load_json(needed[2])
    c35_c4 = load_json(needed[3])
    specs = [
        ("Size extrapolation", c34_c3, c35_c3, [256, 512]),
        ("Wide-weight OOD", c34_c4, c35_c4, [64, 128, 256]),
    ]
    tasks = [("sssp", "SSSP"), ("bottleneck", "Bottleneck")]
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 5.8), sharey=True)
    safe_model = "DeltaNAR+copy-safe repair"
    repair_model = "DeltaNAR+repair-only"
    for col, (setting, safe, repair, sizes) in enumerate(specs):
        for row, (task, label) in enumerate(tasks):
            ax = axs[row, col]
            x = np.arange(len(sizes))
            w = 0.36
            safe_y, safe_e, rep_y, rep_e = [], [], [], []
            for n in sizes:
                m, s = _pair3(safe, ["agg", task, safe_model, str(n), "node_acc_v"])
                safe_y.append(m)
                safe_e.append(s)
                m, s = _pair3(repair, ["agg", task, repair_model, str(n), "node_acc_v"])
                rep_y.append(m)
                rep_e.append(s)
            ax.bar(x - w / 2, safe_y, yerr=safe_e, width=w, capsize=2, color=COLORS["copy-safe"],
                   label="copy-safe")
            ax.bar(x + w / 2, rep_y, yerr=rep_e, width=w, capsize=2, color="#d55e00",
                   label="repair-only")
            ax.set_title(f"{label}: {setting}")
            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in sizes])
            ax.set_ylim(0, 1.04)
            ax.grid(True, axis="y", alpha=0.25)
            if row == 1:
                ax.set_xlabel("test graph size n")
            if col == 0:
                ax.set_ylabel("node_acc_v")
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_size(9)
            ax.tick_params(axis="both", labelsize=9)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=2, fontsize=9, frameon=False)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.08, right=0.98,
                        hspace=0.34, wspace=0.14)
    save(fig, "fig_c34538_stress_repair.pdf", tight=False)


def fig_c38_efficiency():
    fn = "C38_efficiency_10seed_merged.json"
    if not (RESULTS / fn).exists():
        return
    data = load_json(fn)
    sizes = [64, 128, 256, 512]
    tasks = [("sssp", "SSSP"), ("bottleneck", "Bottleneck")]
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.2))
    for task, label in tasks:
        y, e = [], []
        for n in sizes:
            m, s = _pair3(data, ["agg", "copy_safe", task, str(n), "eval_seconds"])
            y.append(m)
            e.append(s)
        axs[0].errorbar(sizes, y, yerr=e, marker="o", lw=2.0, capsize=3, label=label)
        y, e = [], []
        for n in sizes:
            m, s = _pair3(data, ["agg", "copy_safe", task, str(n), "peak_memory_mb"])
            y.append(m)
            e.append(s)
        axs[1].errorbar(sizes, y, yerr=e, marker="o", lw=2.0, capsize=3, label=label)
    axs[0].set_title("Evaluation time")
    axs[1].set_title("Peak CUDA memory")
    axs[0].set_ylabel("seconds / 50 examples")
    axs[1].set_ylabel("MB")
    for ax in axs:
        ax.set_xlabel("test graph size n")
        ax.set_xticks(sizes)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=8)
    save(fig, "fig_c38_efficiency.pdf")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig_size_ood()
    fig_ablation()
    fig_ood()
    fig_theory()
    fig_rollout()
    fig_generalist_transfer()
    fig_extreme_mst()
    fig_b_server_overview()
    fig_b_diagnostic_heatmap()
    fig_b_oracle_copy()
    fig_b_ood_extreme()
    fig_ahr_hard_repair()
    fig_copy_safe_gate_candidate()
    fig_c2_copy_safe_server()
    fig_c34538_stress_repair()
    fig_c38_efficiency()
    print(f"Wrote figures to {FIGS}")
