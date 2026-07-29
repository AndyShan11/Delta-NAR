"""C44 focused stress-test audit.

This runner keeps the C43 budget but changes the self-correction question from
"corrupt memory and learn against corrupted copy targets" to two more plausible
mechanisms:

1. fixed_noisy: corrupt previous memory during the model forward pass, but keep
   repair/copy-safe targets tied to the clean oracle previous solution.
2. certificate repair: at evaluation/rollout time, reject a predicted witness
   whose local predicted-value certificate is inconsistent and choose the
   locally most consistent predecessor.

Each training mode is evaluated both raw and with certificate repair, yielding
four reported variants while training only two models per task/seed.
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
from robustness_c42_noisy_prev_checkpointed import corrupt_previous_solution
from src.dataset import DEFAULT_KINDS, adj_to_matrices
from src.dynamic_sssp import DynamicSSSP
from src.graphs import apply_delta, gen_er_digraph, sample_delta
from src.model import DeltaNAR, collate, evaluate


TRAIN_LABELS = {
    "copy_safe": "copy-safe clean-train",
    "fixed_noisy": "fixed-target noisy-train",
}

VARIANT_LABELS = {
    "copy_safe": "copy-safe",
    "copy_safe_cert": "copy-safe + cert-repair",
    "fixed_noisy": "fixed-target noisy",
    "fixed_noisy_cert": "fixed-target noisy + cert-repair",
}


def parse_cert_profiles(profile_specs: list[str] | None, args) -> list[dict]:
    """Return raw plus one or more certificate profiles.

    Specs use `name:guard:copy_threshold:affected_threshold:tol`. The default
    reproduces the historical raw/cert behavior.
    """
    profiles = [{"name": "", "cert": False}]
    if not profile_specs:
        profiles.append(
            {
                "name": "cert",
                "cert": True,
                "guard": args.cert_guard,
                "copy_threshold": args.cert_copy_safe_threshold,
                "affected_threshold": args.cert_affected_threshold,
                "tol": args.cert_tol,
            }
        )
        return profiles
    for spec in profile_specs:
        parts = spec.split(":")
        if len(parts) != 5:
            raise ValueError(
                "--cert_profiles entries must be name:guard:copy_threshold:affected_threshold:tol"
            )
        name, guard, copy_threshold, affected_threshold, tol = parts
        if guard not in {"all", "affected", "copy_unsafe", "both"}:
            raise ValueError(f"unknown cert guard in profile {spec!r}: {guard}")
        if not name or not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"profile name must be alphanumeric/underscore/hyphen: {name!r}")
        profiles.append(
            {
                "name": name.replace("-", "_"),
                "cert": True,
                "guard": guard,
                "copy_threshold": float(copy_threshold),
                "affected_threshold": float(affected_threshold),
                "tol": float(tol),
            }
        )
    return profiles


def output_variant(train_mode: str, profile: dict) -> str:
    return train_mode if not profile["cert"] else f"{train_mode}_{profile['name']}"


def args_for_profile(args, profile: dict):
    prof_args = copy.copy(args)
    if profile["cert"]:
        prof_args.cert_guard = profile["guard"]
        prof_args.cert_copy_safe_threshold = profile["copy_threshold"]
        prof_args.cert_affected_threshold = profile["affected_threshold"]
        prof_args.cert_tol = profile["tol"]
    return prof_args


def suite_config(name: str, args) -> dict:
    if name == "er":
        return {
            "task_suffix": "",
            "topology": args.test_topology,
            "heavy_tail": args.test_heavy_tail,
            "whi": args.test_weight_hi,
        }
    if name == "wide100":
        return {"task_suffix": "_wide100", "topology": "er", "heavy_tail": False, "whi": 100}
    if name == "heavy_tail":
        return {"task_suffix": "_heavy_tail", "topology": "er", "heavy_tail": True, "whi": args.test_weight_hi}
    if name == "ba":
        return {"task_suffix": "_ba", "topology": "ba", "heavy_tail": False, "whi": args.test_weight_hi}
    raise ValueError(f"unknown eval suite: {name}")


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
        (str(row["train_mode"]), str(row["task"]), int(row["seed"]))
        for row in completed_rows
    }
    return data.get("raw", {}), data.get("rollout_raw", {}), completed, completed_rows


def pm(pair: list[float] | None, digits: int = 3) -> str:
    if not pair:
        return "-"
    return f"{pair[0]:.{digits}f} +/- {pair[1]:.{digits}f} (n={int(pair[2])})"


def write_markdown(path: Path, args, agg: dict, rollout_agg: dict) -> None:
    lines = [
        "# C44 Focused Stress-Test Audit\n",
        "Checkpointed audit for fixed-target noisy-memory training and "
        "inference-time local certificate repair.\n",
        "```json",
        json.dumps(vars(args), indent=2),
        "```\n",
    ]
    metrics = [
        ("node_acc_v", "Node validity"),
        ("graph_acc_v", "Graph validity"),
        ("changed_output_acc_v", "Changed-output validity"),
        ("copy_critical_acc_v", "Copy-critical validity"),
        ("copy_critical_gate_recall", "Copy-critical gate recall"),
        ("cert_repair_rate", "Certificate repair rate"),
    ]
    variants = sorted(agg.keys(), key=lambda v: list(VARIANT_LABELS).index(v) if v in VARIANT_LABELS else 99)
    task_keys = sorted({task for by_task in agg.values() for task in by_task})
    for task in task_keys:
        lines.append(f"## {task} noisy-memory sweep\n")
        for metric, title in metrics:
            lines.append(f"### {title}\n")
            lines.append("| Variant | noise | " + " | ".join(f"n={n}" for n in args.sizes) + " |")
            lines.append("|---|---|" + "|".join("---:" for _ in args.sizes) + "|")
            for variant in variants:
                for noise in args.noise_rates:
                    noise_key = f"{noise:.4g}"
                    cells = []
                    for size in args.sizes:
                        pair = agg.get(variant, {}).get(task, {}).get(noise_key, {}).get(str(size), {}).get(metric)
                        cells.append(pm(pair))
                    lines.append(f"| {VARIANT_LABELS.get(variant, variant)} | {noise_key} | " + " | ".join(cells) + " |")
            lines.append("")

    if rollout_agg:
        lines.append("## SSSP rollout\n")
        steps = [str(t) for t in range(args.rollout_steps)]
        lines.append("| Variant | n | " + " | ".join(f"t={t}" for t in steps) + " |")
        lines.append("|---|---|" + "|".join("---:" for _ in steps) + "|")
        for variant in variants:
            for size in args.rollout_sizes:
                cells = [pm(rollout_agg.get(variant, {}).get(str(size), {}).get(t), 3) for t in steps]
                lines.append(f"| {VARIANT_LABELS.get(variant, variant)} | {size} | " + " | ".join(cells) + " |")
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


@torch.no_grad()
def certificate_repair_scores(
    batch: dict,
    g_pred: torch.Tensor,
    score: torch.Tensor,
    affected_logit: torch.Tensor,
    copy_safe_logit: torch.Tensor | None,
    semiring: str,
    args,
) -> tuple[torch.Tensor, dict]:
    if not args.cert_repair:
        return score, {"cert_repair_rate": 0.0}
    B, n, _ = score.shape
    g = g_pred * batch["scale"]
    g_u = g.unsqueeze(1).expand(B, n, n)
    g_v = g.unsqueeze(2)
    w_vu = batch["W"].transpose(1, 2)
    candidate = batch["A"].transpose(1, 2) > 0
    if semiring == "max_min":
        residual = (torch.minimum(g_u, w_vu) - g_v).abs()
    else:
        residual = (g_u + w_vu - g_v).abs()
    residual = residual.masked_fill(~candidate, float("inf"))
    pred = score.argmax(dim=-1)
    chosen_residual = residual.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
    best_residual, best = residual.min(dim=-1)
    reachable = (batch["reach_next"] * (1.0 - batch["is_src"])) > 0
    bad = reachable & torch.isfinite(best_residual) & (chosen_residual > args.cert_tol)
    if args.cert_guard == "affected":
        bad = bad & (affected_logit > args.cert_affected_threshold)
    elif args.cert_guard == "copy_unsafe":
        if copy_safe_logit is None:
            bad = bad & (affected_logit > args.cert_affected_threshold)
        else:
            bad = bad & (copy_safe_logit < args.cert_copy_safe_threshold)
    elif args.cert_guard == "both":
        if copy_safe_logit is None:
            bad = bad & (affected_logit > args.cert_affected_threshold)
        else:
            bad = bad & (affected_logit > args.cert_affected_threshold) & (copy_safe_logit < args.cert_copy_safe_threshold)
    elif args.cert_guard != "all":
        raise ValueError(f"unknown cert_guard: {args.cert_guard}")
    if not bad.any():
        return score, {"cert_repair_rate": 0.0}
    repaired = score.clone()
    row_max = repaired.max(dim=-1).values + args.cert_bonus
    b_idx, v_idx = torch.nonzero(bad, as_tuple=True)
    repaired[b_idx, v_idx, best[b_idx, v_idx]] = row_max[b_idx, v_idx]
    denom = reachable.float().sum().item()
    rate = bad.float().sum().item() / max(denom, 1.0)
    return repaired, {"cert_repair_rate": rate}


def train_model(task: str, seed: int, train_mode: str, args, device: str):
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
    use_noisy_train = train_mode == "fixed_noisy"
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        for batch_id, ch in enumerate(chunks(train, args.batch)):
            clean = collate(ch, device=device)
            model_batch = clean
            if use_noisy_train and args.train_noise_rate > 0:
                model_batch = corrupt_previous_solution(
                    clean,
                    rate=args.train_noise_rate,
                    mode=args.train_noise_mode,
                    seed=args.noise_seed + seed * 100_003 + epoch * 10_007 + batch_id,
                )
            outputs = model(model_batch)
            loss, _ = compute_repair_loss(
                clean,
                *outputs,
                semiring=sem,
                args=train_args,
                copy_safe_logit=model.last_copy_safe_logit,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, mk, sem


@torch.no_grad()
def eval_at_c44(
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
    cert: bool,
    args,
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
    old_cert = args.cert_repair
    args.cert_repair = cert
    try:
        for chunk_id, ch in enumerate(chunks(samples, batch_size)):
            b = collate(ch, device=device)
            b = corrupt_previous_solution(
                b,
                rate=noise_rate,
                mode=noise_mode,
                seed=noise_seed + 1009 * chunk_id + 9176 * n,
            )
            g_pred, score, affected_logit = model(b)
            score, cert_stats = certificate_repair_scores(
                b, g_pred, score, affected_logit, model.last_copy_safe_logit, sem, args
            )
            metrics = diagnostic_metrics(b, g_pred, score, affected_logit, semiring=sem)
            metrics.update(cert_stats)
            if tot is None:
                tot = {k: 0.0 for k in metrics}
            for k, v in metrics.items():
                tot[k] += v * len(ch)
            cnt += len(ch)
    finally:
        args.cert_repair = old_cert
    return {k: tot[k] / cnt for k in tot}


@torch.no_grad()
def rollout_sssp_c44(model, n: int, n_episodes: int, n_steps: int, device: str, cert: bool, args, seed0: int = 700_000) -> list[float]:
    model.eval()
    per_step = [[] for _ in range(n_steps)]
    old_cert = args.cert_repair
    args.cert_repair = cert
    try:
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
                score, _ = certificate_repair_scores(
                    b, g_pred, score, aff, model.last_copy_safe_logit, "min_plus", args
                )
                per_step[t].append(evaluate(b, g_pred, score, aff)["node_acc_v"])
                pred_run = score.argmax(-1)[0].detach().cpu().numpy().astype(np.int64)
                g_run = g_pred[0].detach().cpu().numpy() * b["scale"]
    finally:
        args.cert_repair = old_cert
    return [float(np.mean(values)) if values else float("nan") for values in per_step]


def run_one(task: str, seed: int, train_mode: str, args, device: str) -> tuple[dict, dict]:
    model, mk, sem = train_model(task, seed, train_mode, args, device)
    by_variant = {}
    rollout_by_variant = {}
    for profile in parse_cert_profiles(args.cert_profiles, args):
        cert = bool(profile["cert"])
        prof_args = args_for_profile(args, profile)
        variant = output_variant(train_mode, profile)
        by_variant[variant] = {}
        for suite in args.eval_suites:
            cfg = suite_config(suite, args)
            task_key = f"{task}{cfg['task_suffix']}"
            by_variant[variant].setdefault(task_key, {})
            for noise_rate in args.noise_rates:
                noise_key = f"{noise_rate:.4g}"
                by_variant[variant][task_key].setdefault(noise_key, {})
                for size in args.sizes:
                    by_variant[variant][task_key][noise_key][str(size)] = eval_at_c44(
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
                        cert=cert,
                        args=prof_args,
                        topology=cfg["topology"],
                        heavy_tail=cfg["heavy_tail"],
                        whi=cfg["whi"],
                        kinds=args.test_delta_kinds,
                        delta_size=args.test_delta_size,
                    )
        if task == "sssp" and args.rollout_episodes > 0:
            rollout_by_variant[variant] = {}
            for size in args.rollout_sizes:
                curve = rollout_sssp_c44(model, size, args.rollout_episodes, args.rollout_steps, device, cert, prof_args)
                rollout_by_variant[variant][str(size)] = {str(t): value for t, value in enumerate(curve)}
    return by_variant, rollout_by_variant


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["sssp", "bottleneck"])
    ap.add_argument("--train_modes", nargs="+", default=["copy_safe", "fixed_noisy"], choices=list(TRAIN_LABELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--noise_rates", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.10])
    ap.add_argument("--eval_suites", nargs="+", default=["er"], choices=["er", "wide100", "heavy_tail", "ba"])
    ap.add_argument("--noise_mode", choices=["pred", "value", "both"], default="both")
    ap.add_argument("--train_noise_rate", type=float, default=0.05)
    ap.add_argument("--train_noise_mode", choices=["pred", "value", "both"], default="both")
    ap.add_argument("--noise_seed", type=int, default=444444)
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
    ap.add_argument("--out", default="results/C44_focused_stress_checkpointed.json")
    ap.add_argument("--markdown", default="RESULTS_C44_FOCUSED_STRESS.md")
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
    ap.add_argument("--cert_repair", action="store_true")
    ap.add_argument("--cert_tol", type=float, default=1.0)
    ap.add_argument("--cert_bonus", type=float, default=20.0)
    ap.add_argument("--cert_guard", choices=["all", "affected", "copy_unsafe", "both"], default="all")
    ap.add_argument("--cert_copy_safe_threshold", type=float, default=0.0)
    ap.add_argument("--cert_affected_threshold", type=float, default=0.0)
    ap.add_argument(
        "--cert_profiles",
        nargs="+",
        default=None,
        help="Optional cert profiles: name:guard:copy_threshold:affected_threshold:tol",
    )
    return ap


def main() -> None:
    args = parser().parse_args()
    out = Path(args.out)
    markdown = Path(args.markdown)
    raw, rollout_raw, completed, completed_rows = load_existing(out)
    cert_profiles = parse_cert_profiles(args.cert_profiles, args)
    for train_mode in args.train_modes:
        for profile in cert_profiles:
            variant = output_variant(train_mode, profile)
            raw.setdefault(variant, {})
            rollout_raw.setdefault(variant, {})
            for size in args.rollout_sizes:
                rollout_raw[variant].setdefault(str(size), {str(t): [] for t in range(args.rollout_steps)})
            for task in args.tasks:
                for suite in args.eval_suites:
                    cfg = suite_config(suite, args)
                    task_key = f"{task}{cfg['task_suffix']}"
                    raw[variant].setdefault(task_key, {})
                    for noise_rate in args.noise_rates:
                        noise_key = f"{noise_rate:.4g}"
                        raw[variant][task_key].setdefault(noise_key, {str(n): {} for n in args.sizes})
                        for size in args.sizes:
                            raw[variant][task_key][noise_key].setdefault(str(size), {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device} train_modes={args.train_modes} tasks={args.tasks} seeds={args.seeds} "
        f"sizes={args.sizes} train_noise={args.train_noise_rate} cert_tol={args.cert_tol} "
        f"completed={len(completed)}",
        flush=True,
    )
    for train_mode in args.train_modes:
        for task in args.tasks:
            for seed in args.seeds:
                key = (train_mode, task, seed)
                if key in completed:
                    print(f"[skip {train_mode}/{task}/seed{seed}] already checkpointed", flush=True)
                    continue
                t0 = time.time()
                by_variant, rollout_by_variant = run_one(task, seed, train_mode, args, device)
                for variant, by_task in by_variant.items():
                    for task_key, by_noise in by_task.items():
                        for noise_key, by_size in by_noise.items():
                            for size, metrics in by_size.items():
                                for metric, value in metrics.items():
                                    raw[variant][task_key][noise_key][size].setdefault(metric, []).append(value)
                for variant, by_size in rollout_by_variant.items():
                    for size, by_step in by_size.items():
                        rollout_raw[variant].setdefault(size, {str(t): [] for t in range(args.rollout_steps)})
                        for step, value in by_step.items():
                            rollout_raw[variant][size].setdefault(step, []).append(value)
                elapsed = time.time() - t0
                completed.add(key)
                completed_rows.append(
                    {"train_mode": train_mode, "task": task, "seed": seed, "elapsed_sec": elapsed}
                )
                print(f"[{train_mode}/{task}/seed{seed}] {elapsed:.0f}s", flush=True)
                save_checkpoint(out, markdown, args, raw, rollout_raw, completed_rows)

    save_checkpoint(out, markdown, args, raw, rollout_raw, completed_rows)
    print("C44 focused stress-test audit complete", flush=True)


if __name__ == "__main__":
    main()
