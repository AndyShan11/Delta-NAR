"""C42 robustness audit for imperfect previous solutions.

The main DeltaNAR contract assumes that y_{t-1} is exact at the beginning of a
one-step update. This checkpointed diagnostic keeps training unchanged, then
corrupts the previous-solution input at evaluation time. It measures how much
copy-based variants rely on exact memory, and flushes JSON/Markdown after every
completed variant/task/seed pair so server jobs can be resumed safely.
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
from src.model import DeltaNAR, collate


VARIANT_LABELS = {
    "copy_safe": "DeltaNAR+copy-safe repair",
    "repair_only": "DeltaNAR+repair-only",
}


def aggregate(raw: dict) -> dict:
    agg = {}
    for variant, by_task in raw.items():
        agg[variant] = {}
        for task, by_noise in by_task.items():
            agg[variant][task] = {}
            for noise, by_size in by_noise.items():
                agg[variant][task][noise] = {}
                for size, by_metric in by_size.items():
                    agg[variant][task][noise][size] = {
                        metric: [float(np.mean(values)), float(np.std(values)), len(values)]
                        for metric, values in by_metric.items()
                        if values
                    }
    return agg


def load_existing(path: Path) -> tuple[dict, set[tuple[str, str, int]], list[dict]]:
    if not path.exists():
        return {}, set(), []
    data = json.loads(path.read_text(encoding="utf-8"))
    completed_rows = data.get("completed", [])
    completed = {
        (str(row["variant"]), str(row["task"]), int(row["seed"]))
        for row in completed_rows
    }
    return data.get("raw", {}), completed, completed_rows


def pm(pair: list[float] | None, digits: int = 3) -> str:
    if not pair:
        return "-"
    return f"{pair[0]:.{digits}f} +/- {pair[1]:.{digits}f} (n={int(pair[2])})"


def write_markdown(path: Path, args, agg: dict) -> None:
    metrics = [
        ("node_acc_v", "Node validity"),
        ("graph_acc_v", "Graph validity"),
        ("changed_output_acc_v", "Changed-output validity"),
        ("copy_critical_acc_v", "Copy-critical validity"),
        ("pure_copy_node_acc_v", "Pure-copy node validity"),
        ("pure_copy_changed_acc_v", "Pure-copy changed validity"),
        ("gate_recall", "Affected gate recall"),
        ("copy_critical_gate_recall", "Copy-critical gate recall"),
        ("changed_output_fraction", "Changed-output fraction"),
        ("copy_critical_fraction", "Copy-critical fraction"),
    ]
    lines = [
        "# C42 Noisy Previous-Solution Robustness\n",
        "Checkpointed evaluation-time audit for robustness to imperfect "
        "previous-solution inputs. Training is unchanged; only `g_prev` and "
        "`pred_prev`/`prev_tree` are corrupted during evaluation.\n",
        "```json",
        json.dumps(vars(args), indent=2),
        "```\n",
    ]
    for task in args.tasks:
        lines.append(f"## {task}\n")
        for metric, title in metrics:
            lines.append(f"### {title}\n")
            header = ["Variant", "noise"] + [f"n={n}" for n in args.sizes]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|---|---|" + "|".join("---:" for _ in args.sizes) + "|")
            for variant in args.variants:
                for noise in args.noise_rates:
                    noise_key = f"{noise:.4g}"
                    cells = []
                    for n in args.sizes:
                        pair = agg.get(variant, {}).get(task, {}).get(noise_key, {}).get(str(n), {}).get(metric)
                        cells.append(pm(pair))
                    lines.append(
                        f"| {VARIANT_LABELS[variant]} | {noise_key} | "
                        + " | ".join(cells)
                        + " |"
                    )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_checkpoint(path: Path, markdown: Path, args, raw: dict, completed_rows: list[dict]) -> None:
    agg = aggregate(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "completed": completed_rows,
                "agg": agg,
                "raw": raw,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(markdown, args, agg)
    print(f"checkpoint -> {path}", flush=True)
    print(f"checkpoint -> {markdown}", flush=True)


def corrupt_previous_solution(batch: dict, rate: float, mode: str, seed: int) -> dict:
    """Return a shallow-copied batch with noisy previous-solution fields.

    The corruption is evaluation-only. We sample reachable non-source vertices,
    replace their previous predecessor with a random incoming edge when possible,
    and optionally replace their previous scalar value with another node's value.
    """
    if rate <= 0:
        return batch
    out = dict(batch)
    prev_tree = batch["prev_tree"].clone()
    g_prev = batch["g_prev"].clone()
    A = batch["A"]
    mask = ((batch["reach_next"] * (1.0 - batch["is_src"])) > 0).detach().cpu().numpy()
    B, n, _ = prev_tree.shape
    rng = np.random.default_rng(seed)

    do_pred = mode in {"pred", "both"}
    do_value = mode in {"value", "both"}
    for b in range(B):
        selected = np.flatnonzero(mask[b] & (rng.random(n) < rate))
        for v in selected:
            if do_pred:
                incoming = torch.nonzero(A[b, :, v] > 0, as_tuple=False).flatten().detach().cpu().numpy()
                if incoming.size:
                    old = int(prev_tree[b, :, v].argmax().item())
                    choices = incoming[incoming != old]
                    u = int(rng.choice(choices if choices.size else incoming))
                    prev_tree[b, :, v] = 0.0
                    prev_tree[b, u, v] = 1.0
            if do_value:
                donor = int(rng.integers(0, n))
                g_prev[b, v] = batch["g_prev"][b, donor]

    out["prev_tree"] = prev_tree
    out["g_prev"] = g_prev
    return out


@torch.no_grad()
def eval_at_noisy(
    model,
    mk,
    sem,
    n: int,
    n_ep: int,
    device: str,
    batch_size: int,
    noise_rate: float,
    noise_mode: str,
    noise_seed: int,
    topology: str = "er",
    heavy_tail: bool = False,
    whi: int = 10,
    kinds=None,
    delta_size: int = 1,
) -> dict:
    model.eval()
    samples = mk(
        n_ep,
        n,
        n_steps=5,
        seed0=900_000,
        topology=topology,
        heavy_tail=heavy_tail,
        weight_range=(1, whi),
        kinds=kinds,
        delta_size=delta_size,
    )
    tot, cnt = None, 0
    for chunk_id, ch in enumerate(chunks(samples, batch_size)):
        b = collate(ch, device=device)
        b = corrupt_previous_solution(
            b,
            rate=noise_rate,
            mode=noise_mode,
            seed=noise_seed + 1009 * chunk_id + 9176 * n,
        )
        metrics = diagnostic_metrics(b, *model(b), semiring=sem)
        if tot is None:
            tot = {k: 0.0 for k in metrics}
        for k, v in metrics.items():
            tot[k] += v * len(ch)
        cnt += len(ch)
    return {k: tot[k] / cnt for k in tot}


def train_model(task: str, seed: int, variant: str, args, device: str):
    mk, sem = TASK_CFG[task]
    set_seed(seed)
    copy_margin = args.copy_margin if args.copy_margin is not None else (2.0 if task == "bottleneck" else 0.0)
    cfg = dict(CONFIGS["DeltaNAR"])
    use_copy_safe = variant == "copy_safe"
    model = DeltaNAR(
        hidden=args.hidden,
        task="sssp",
        steps=(args.steps or None),
        copy_margin=copy_margin,
        copy_scale=args.copy_scale,
        use_copy_safe_gate=use_copy_safe,
        **cfg,
    ).to(device)
    train_args = copy.copy(args)
    train_args.use_copy_safe_gate = use_copy_safe
    train_args.copy_safe_weight = args.copy_safe_weight if use_copy_safe else 0.0
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train = mk(
        args.train_episodes,
        args.train_n,
        n_steps=5,
        seed0=1 + seed * 7919,
        delta_size=args.train_delta_size,
        kinds=args.train_delta_kinds,
    )
    for _ in range(args.epochs):
        model.train()
        random.shuffle(train)
        for ch in chunks(train, args.batch):
            b = collate(ch, device=device)
            outputs = model(b)
            loss, _ = compute_repair_loss(
                b, *outputs, semiring=sem, args=train_args, copy_safe_logit=model.last_copy_safe_logit
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, mk, sem


def run_one(task: str, seed: int, variant: str, args, device: str) -> dict:
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
    return by_noise


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sssp", "bottleneck"])
    ap.add_argument("--variants", nargs="+", default=["copy_safe", "repair_only"], choices=list(VARIANT_LABELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--noise_rates", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.10])
    ap.add_argument("--noise_mode", choices=["pred", "value", "both"], default="both")
    ap.add_argument("--noise_seed", type=int, default=424242)
    ap.add_argument("--train_n", type=int, default=16)
    ap.add_argument("--train_episodes", type=int, default=500)
    ap.add_argument("--test_episodes", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/C42_noisy_prev_checkpointed.json")
    ap.add_argument("--markdown", default="C42_NOISY_PREV_SUMMARY.md")
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
    raw, completed, completed_rows = load_existing(out)
    for variant in args.variants:
        raw.setdefault(variant, {})
        for task in args.tasks:
            raw[variant].setdefault(task, {})
            for noise_rate in args.noise_rates:
                noise_key = f"{noise_rate:.4g}"
                raw[variant][task].setdefault(noise_key, {str(n): {} for n in args.sizes})
                for n in args.sizes:
                    raw[variant][task][noise_key].setdefault(str(n), {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} variants={args.variants} tasks={args.tasks} "
        f"seeds={args.seeds} sizes={args.sizes} noise_rates={args.noise_rates} "
        f"completed={len(completed)}",
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
                by_noise = run_one(task, seed, variant, args, device)
                for noise_key, by_size in by_noise.items():
                    for size, metrics in by_size.items():
                        for metric, value in metrics.items():
                            raw[variant][task][noise_key][size].setdefault(metric, []).append(value)
                elapsed = time.time() - t0
                completed.add(key)
                completed_rows.append(
                    {"variant": variant, "task": task, "seed": seed, "elapsed_sec": elapsed}
                )
                print(f"[{variant}/{task}/seed{seed}] {elapsed:.0f}s", flush=True)
                save_checkpoint(out, markdown, args, raw, completed_rows)

    save_checkpoint(out, markdown, args, raw, completed_rows)
    print("C42 noisy previous-solution robustness complete", flush=True)


if __name__ == "__main__":
    main()
