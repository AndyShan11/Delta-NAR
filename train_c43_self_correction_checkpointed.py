"""C43 self-correction training for noisy previous-solution memory.

This runner tests whether DeltaNAR can learn not only to copy safe memory, but
also to distrust stale memory. The self-correcting variant injects noise into
the previous-solution input during training and keeps the copy-safe loss target
defined against that corrupted memory. Evaluation reports the C42-style noisy
previous-solution sweep and an SSSP rollout curve.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from diagnose_a_metrics import diagnostic_metrics
from diagnose_repair_loss import compute_repair_loss
from experiments import CONFIGS, TASK_CFG, chunks, set_seed
from robustness_c42_noisy_prev_checkpointed import corrupt_previous_solution, eval_at_noisy
from src.dataset import DEFAULT_KINDS, adj_to_matrices
from src.dynamic_sssp import DynamicSSSP
from src.graphs import apply_delta, gen_er_digraph, sample_delta
from src.model import DeltaNAR, collate, evaluate


VARIANT_LABELS = {
    "copy_safe": "copy-safe clean-train",
    "self_correct": "copy-safe self-correct",
}


def aggregate_metric(raw: dict) -> dict:
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


def load_existing(path: Path) -> tuple[dict, dict, set[tuple[str, str, int]], list[dict]]:
    if not path.exists():
        return {}, {}, set(), []
    data = json.loads(path.read_text(encoding="utf-8"))
    completed_rows = data.get("completed", [])
    completed = {
        (str(row["variant"]), str(row["task"]), int(row["seed"]))
        for row in completed_rows
    }
    return data.get("raw", {}), data.get("rollout_raw", {}), completed, completed_rows


def pm(pair: list[float] | None, digits: int = 3) -> str:
    if not pair:
        return "-"
    return f"{pair[0]:.{digits}f} +/- {pair[1]:.{digits}f} (n={int(pair[2])})"


def write_markdown(path: Path, args, agg: dict, rollout_agg: dict) -> None:
    lines = [
        "# C43 Self-Correction Training\n",
        "Checkpointed audit for training-time noisy previous-solution memory. "
        "The self-correct variant corrupts previous memory during training and "
        "keeps the copy-safe target tied to the corrupted witness.\n",
        "```json",
        json.dumps(vars(args), indent=2),
        "```\n",
    ]
    metrics = [
        ("node_acc_v", "Node validity"),
        ("graph_acc_v", "Graph validity"),
        ("changed_output_acc_v", "Changed-output validity"),
        ("copy_critical_gate_recall", "Copy-critical gate recall"),
    ]
    for task in args.tasks:
        lines.append(f"## {task} noisy-memory sweep\n")
        for metric, title in metrics:
            lines.append(f"### {title}\n")
            lines.append("| Variant | noise | " + " | ".join(f"n={n}" for n in args.sizes) + " |")
            lines.append("|---|---|" + "|".join("---:" for _ in args.sizes) + "|")
            for variant in args.variants:
                for noise in args.noise_rates:
                    noise_key = f"{noise:.4g}"
                    cells = []
                    for size in args.sizes:
                        pair = agg.get(variant, {}).get(task, {}).get(noise_key, {}).get(str(size), {}).get(metric)
                        cells.append(pm(pair))
                    lines.append(f"| {VARIANT_LABELS[variant]} | {noise_key} | " + " | ".join(cells) + " |")
            lines.append("")

    if rollout_agg:
        lines.append("## SSSP rollout\n")
        steps = [str(t) for t in range(args.rollout_steps)]
        lines.append("| Variant | n | " + " | ".join(f"t={t}" for t in steps) + " |")
        lines.append("|---|---|" + "|".join("---:" for _ in steps) + "|")
        for variant in args.variants:
            for size in args.rollout_sizes:
                cells = [pm(rollout_agg.get(variant, {}).get(str(size), {}).get(t), 3) for t in steps]
                lines.append(f"| {VARIANT_LABELS[variant]} | {size} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, markdown: Path, args, raw: dict, rollout_raw: dict, completed_rows: list[dict]) -> None:
    agg = aggregate_metric(raw)
    rollout_agg = aggregate_rollout(rollout_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "completed": completed_rows,
                "agg": agg,
                "rollout_agg": rollout_agg,
                "raw": raw,
                "rollout_raw": rollout_raw,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(markdown, args, agg, rollout_agg)
    print(f"checkpoint -> {path}", flush=True)
    print(f"checkpoint -> {markdown}", flush=True)


def train_model(task: str, seed: int, variant: str, args, device: str):
    mk, sem = TASK_CFG[task]
    set_seed(seed)
    copy_margin = args.copy_margin if args.copy_margin is not None else (2.0 if task == "bottleneck" else 0.0)
    cfg = dict(CONFIGS["DeltaNAR"])
    model = DeltaNAR(
        hidden=args.hidden,
        task="sssp",
        steps=(args.steps or None),
        copy_margin=copy_margin,
        copy_scale=args.copy_scale,
        use_copy_safe_gate=True,
        **cfg,
    ).to(device)
    train_args = copy.copy(args)
    train_args.use_copy_safe_gate = True
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train = mk(
        args.train_episodes,
        args.train_n,
        n_steps=5,
        seed0=1 + seed * 7919,
        delta_size=args.train_delta_size,
        kinds=args.train_delta_kinds,
    )
    use_noisy_train = variant == "self_correct"
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        for batch_id, ch in enumerate(chunks(train, args.batch)):
            b = collate(ch, device=device)
            if use_noisy_train and args.train_noise_rate > 0:
                b = corrupt_previous_solution(
                    b,
                    rate=args.train_noise_rate,
                    mode=args.train_noise_mode,
                    seed=args.noise_seed + seed * 100_003 + epoch * 10_007 + batch_id,
                )
            outputs = model(b)
            loss, _ = compute_repair_loss(
                b, *outputs, semiring=sem, args=train_args, copy_safe_logit=model.last_copy_safe_logit
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, mk, sem


@torch.no_grad()
def rollout_sssp(model, n: int, n_episodes: int, n_steps: int, device: str, seed0: int = 700_000) -> list[float]:
    model.eval()
    per_step = [[] for _ in range(n_steps)]
    for episode in range(n_episodes):
        rng = random.Random(seed0 + episode)
        adj = gen_er_digraph(n, seed=seed0 + episode)
        ds = DynamicSSSP(adj, 0, n)
        cur = adj
        g_run = np.array(ds.distances(), dtype=np.float64)
        pred_run = np.array(ds.predecessors(), dtype=np.int64)
        for t in range(n_steps):
            delta = sample_delta(cur, rng.choice(DEFAULT_KINDS), rng)
            if delta is None:
                continue
            cur = apply_delta(cur, delta)
            ds.apply_delta(delta)
            W, A = adj_to_matrices(cur, n)
            sample = {
                "n": n,
                "source": 0,
                "W": W,
                "A": A,
                "g_prev": g_run,
                "pred_prev": pred_run,
                "deltas": [delta],
                "g_next": np.array(ds.distances(), dtype=np.float64),
                "pred_next": np.array(ds.predecessors(), dtype=np.int64),
                "affected": np.zeros(n, dtype=np.float64),
            }
            b = collate([sample], device=device)
            g_pred, score, aff = model(b)
            per_step[t].append(evaluate(b, g_pred, score, aff)["node_acc_v"])
            pred_run = score.argmax(-1)[0].detach().cpu().numpy().astype(np.int64)
            g_run = g_pred[0].detach().cpu().numpy() * b["scale"]
    return [float(np.mean(values)) if values else float("nan") for values in per_step]


def run_one(task: str, seed: int, variant: str, args, device: str) -> tuple[dict, dict]:
    model, mk, sem = train_model(task, seed, variant, args, device)
    by_noise = {}
    for noise_rate in args.noise_rates:
        noise_key = f"{noise_rate:.4g}"
        by_noise[noise_key] = {}
        for size in args.sizes:
            by_noise[noise_key][str(size)] = eval_at_noisy(
                model,
                mk,
                sem,
                size,
                args.test_episodes,
                device,
                args.eval_batch,
                noise_rate=noise_rate,
                noise_mode=args.noise_mode,
                noise_seed=args.noise_seed + seed * 100_003,
                topology=args.test_topology,
                heavy_tail=args.test_heavy_tail,
                whi=args.test_weight_hi,
                kinds=args.test_delta_kinds,
                delta_size=args.test_delta_size,
            )
    rollout = {}
    if task == "sssp" and args.rollout_episodes > 0:
        for size in args.rollout_sizes:
            curve = rollout_sssp(model, size, args.rollout_episodes, args.rollout_steps, device)
            rollout[str(size)] = {str(t): value for t, value in enumerate(curve)}
    return by_noise, rollout


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sssp", "bottleneck"])
    ap.add_argument("--variants", nargs="+", default=["copy_safe", "self_correct"], choices=list(VARIANT_LABELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--noise_rates", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.10])
    ap.add_argument("--noise_mode", choices=["pred", "value", "both"], default="both")
    ap.add_argument("--train_noise_rate", type=float, default=0.05)
    ap.add_argument("--train_noise_mode", choices=["pred", "value", "both"], default="both")
    ap.add_argument("--noise_seed", type=int, default=434343)
    ap.add_argument("--rollout_sizes", type=int, nargs="+", default=[64])
    ap.add_argument("--rollout_steps", type=int, default=60)
    ap.add_argument("--rollout_episodes", type=int, default=40)
    ap.add_argument("--train_n", type=int, default=16)
    ap.add_argument("--train_episodes", type=int, default=500)
    ap.add_argument("--test_episodes", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/C43_self_correction_checkpointed.json")
    ap.add_argument("--markdown", default="C43_SELF_CORRECTION_SUMMARY.md")
    ap.add_argument("--test_topology", default="er")
    ap.add_argument("--test_heavy_tail", action="store_true")
    ap.add_argument("--test_weight_hi", type=int, default=10)
    ap.add_argument("--test_delta_kinds", nargs="+", default=None)
    ap.add_argument("--train_delta_kinds", nargs="+", default=None)
    ap.add_argument("--train_delta_size", type=int, default=4)
    ap.add_argument("--test_delta_size", type=int, default=12)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--repair_weight", type=float, default=1.0)
    ap.add_argument("--critical_weight", type=float, default=0.0)
    ap.add_argument("--copy_safe_weight", type=float, default=1.0)
    ap.add_argument("--copy_safe_pos_weight", type=float, default=20.0)
    ap.add_argument("--copy_margin", type=float, default=None)
    ap.add_argument("--copy_scale", type=float, default=5.0)
    return ap


def main() -> None:
    args = parser().parse_args()
    out = Path(args.out)
    markdown = Path(args.markdown)
    raw, rollout_raw, completed, completed_rows = load_existing(out)
    for variant in args.variants:
        raw.setdefault(variant, {})
        rollout_raw.setdefault(variant, {})
        for size in args.rollout_sizes:
            rollout_raw[variant].setdefault(str(size), {str(t): [] for t in range(args.rollout_steps)})
        for task in args.tasks:
            raw[variant].setdefault(task, {})
            for noise_rate in args.noise_rates:
                noise_key = f"{noise_rate:.4g}"
                raw[variant][task].setdefault(noise_key, {str(n): {} for n in args.sizes})
                for size in args.sizes:
                    raw[variant][task][noise_key].setdefault(str(size), {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} variants={args.variants} tasks={args.tasks} seeds={args.seeds} "
        f"sizes={args.sizes} train_noise={args.train_noise_rate} completed={len(completed)}",
        flush=True,
    )
    for variant in args.variants:
        for task in args.tasks:
            for seed in args.seeds:
                key = (variant, task, seed)
                if key in completed:
                    print(f"[skip {variant}/{task}/seed{seed}] already checkpointed", flush=True)
                    continue
                t0 = time.time()
                by_noise, rollout = run_one(task, seed, variant, args, device)
                for noise_key, by_size in by_noise.items():
                    for size, metrics in by_size.items():
                        for metric, value in metrics.items():
                            raw[variant][task][noise_key][size].setdefault(metric, []).append(value)
                for size, by_step in rollout.items():
                    rollout_raw[variant].setdefault(size, {str(t): [] for t in range(args.rollout_steps)})
                    for step, value in by_step.items():
                        rollout_raw[variant][size].setdefault(step, []).append(value)
                elapsed = time.time() - t0
                completed.add(key)
                completed_rows.append(
                    {"variant": variant, "task": task, "seed": seed, "elapsed_sec": elapsed}
                )
                print(f"[{variant}/{task}/seed{seed}] {elapsed:.0f}s", flush=True)
                save_checkpoint(out, markdown, args, raw, rollout_raw, completed_rows)

    save_checkpoint(out, markdown, args, raw, rollout_raw, completed_rows)
    print("C43 self-correction training complete", flush=True)


if __name__ == "__main__":
    main()
