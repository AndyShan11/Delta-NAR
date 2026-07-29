"""Train + evaluate DeltaNAR and ablations on CLRS-Δ dynamic SSSP.

Decisive smoke test: train on n=16, test in-distribution (n=16) and OOD (n=64).
Core hypothesis: DeltaNAR (prev-solution + delta + affected-region gate) keeps
OOD graph-accuracy high, while Recompute-from-scratch collapses on large graphs.

Run (must use the torch interpreter, NOT the default MinGW python):
  & "D:\\dev\\python\\python3.12.4\\python.exe" E:\\NAR\\src\\train.py
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (DEFAULT_KINDS, TYPE_ID, adj_to_matrices,  # noqa: E402
                         make_dataset)
from src.dynamic_sssp import DynamicSSSP  # noqa: E402
from src.graphs import apply_delta, gen_er_digraph, sample_delta  # noqa: E402
from src.model import DeltaNAR, collate, compute_loss, evaluate  # noqa: E402

CONFIGS = {
    "DeltaNAR":  dict(use_prev=True,  use_delta=True,  use_gate=True),
    "Naive-d":   dict(use_prev=True,  use_delta=True,  use_gate=False, use_copy=False),  # true -copy
    "Recompute": dict(use_prev=False, use_delta=False, use_gate=False),
}


def chunks(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i:i + k]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def eval_on(model, n, n_episodes, device, batch=32, delta_size=1):
    model.eval()
    samples = make_dataset(n_episodes, n, n_steps=5, seed0=900_000, delta_size=delta_size)
    tot, cnt = None, 0
    for ch in chunks(samples, batch):
        b = collate(ch, device=device)            # scale defaults to n*10 (per-n normalised)
        m = evaluate(b, *model(b))
        if tot is None:
            tot = {k: 0.0 for k in m}
        for k in tot:
            tot[k] += m[k] * len(ch)
        cnt += len(ch)
    return {k: tot[k] / cnt for k in tot}


@torch.no_grad()
def rollout_eval(model, n, n_episodes, n_steps, device, seed0=700_000):
    """Feed the model's OWN previous prediction as next-step prev (no teacher forcing).

    Returns per-step tie-tolerant node accuracy; reveals whether error compounds.
    """
    model.eval()
    per_step = [[] for _ in range(n_steps)]
    for e in range(n_episodes):
        rng = random.Random(seed0 + e)
        adj = gen_er_digraph(n, seed=seed0 + e)
        ds = DynamicSSSP(adj, 0, n)  # independent ground-truth engine
        cur = adj
        g_run = np.array(ds.distances(), dtype=np.float64)        # model's running solution
        pred_run = np.array(ds.predecessors(), dtype=np.int64)
        for t in range(n_steps):
            delta = sample_delta(cur, rng.choice(DEFAULT_KINDS), rng)
            if delta is None:
                continue
            cur = apply_delta(cur, delta)
            ds.apply_delta(delta)
            W, A = adj_to_matrices(cur, n)
            sample = {
                "n": n, "source": 0, "W": W, "A": A,
                "g_prev": g_run, "pred_prev": pred_run,
                "deltas": [delta],
                "g_next": np.array(ds.distances(), dtype=np.float64),
                "pred_next": np.array(ds.predecessors(), dtype=np.int64),
                "affected": np.zeros(n, dtype=np.float64),  # unused by forward
            }
            b = collate([sample], device=device)
            g_pred, score, aff = model(b)
            per_step[t].append(evaluate(b, g_pred, score, aff)["node_acc_v"])
            pred_run = score.argmax(-1)[0].cpu().numpy().astype(np.int64)  # roll own prediction
            g_run = g_pred[0].cpu().numpy() * b["scale"]
    return [float(np.mean(s)) if s else float("nan") for s in per_step]


def train_one(cfg, args, device):
    set_seed(args.seed)
    model = DeltaNAR(hidden=args.hidden, **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train = make_dataset(args.train_episodes, args.train_n, n_steps=5, seed0=0,
                         delta_size=args.train_delta)
    for _ in range(args.epochs):
        model.train()
        random.shuffle(train)
        for ch in chunks(train, args.batch):
            b = collate(ch, device=device)
            loss, _ = compute_loss(b, *model(b))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, {n: eval_on(model, n, args.test_episodes, device, delta_size=args.test_delta)
                   for n in args.test_ns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=16)
    ap.add_argument("--test_ns", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--train_episodes", type=int, default=500)
    ap.add_argument("--test_episodes", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--models", nargs="+", default=list(CONFIGS))
    ap.add_argument("--rollout", action="store_true")
    ap.add_argument("--rollout_n", type=int, default=64)
    ap.add_argument("--rollout_steps", type=int, default=10)
    ap.add_argument("--rollout_episodes", type=int, default=40)
    ap.add_argument("--train_delta", type=int, default=1)
    ap.add_argument("--test_delta", type=int, default=1)
    ap.add_argument("--intensity", action="store_true")
    ap.add_argument("--intensity_n", type=int, default=16)
    ap.add_argument("--intensity_deltas", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  train_n={args.train_n}  test_ns={args.test_ns}  "
          f"epochs={args.epochs}  hidden={args.hidden}")

    results, models = {}, {}
    selected = {k: CONFIGS[k] for k in args.models}
    for name, cfg in selected.items():
        t = time.time()
        models[name], results[name] = train_one(cfg, args, device)
        print(f"[{name:10s}] trained in {time.time() - t:5.1f}s")

    # gAx=graph_acc exact, gAv=graph_acc valid(tie-tolerant), nAv=node_acc valid, aF1=affected F1
    print("\n=== gAx / gAv / nAv / aF1  (x=exact match, v=tie-tolerant) ===")
    header = "Model".ljust(11) + "".join(f"  n={n:<4d} gAx/gAv/nAv/aF1".ljust(34) for n in args.test_ns)
    print(header)
    print("-" * len(header))
    for name in selected:
        row = name.ljust(11)
        for n in args.test_ns:
            r = results[name][n]
            row += f"  {r['graph_acc']:.3f}/{r['graph_acc_v']:.3f}/{r['node_acc_v']:.3f}/{r['aff_f1']:.3f}".ljust(34)
        print(row)

    if args.rollout:
        print(f"\n=== rollout node_acc_v @ n={args.rollout_n} "
              f"(model feeds its OWN prediction forward, {args.rollout_steps} steps) ===")
        print("model      " + "".join(f"  t{t:<3d}" for t in range(args.rollout_steps)))
        for name in selected:
            curve = rollout_eval(models[name], args.rollout_n, args.rollout_episodes,
                                 args.rollout_steps, device)
            print(name.ljust(11) + "".join(f" {c:.3f}" for c in curve))

    if args.intensity:
        print(f"\n=== dynamic-intensity OOD @ n={args.intensity_n} "
              f"(trained ‖δ‖={args.train_delta}; gAv/nAv per test ‖δ‖) ===")
        print("model      " + "".join(f"‖d‖={k:<2d} gAv/nAv".ljust(20) for k in args.intensity_deltas))
        for name in selected:
            row = name.ljust(11)
            for k in args.intensity_deltas:
                r = eval_on(models[name], args.intensity_n, args.test_episodes, device, delta_size=k)
                row += f"{r['graph_acc_v']:.3f}/{r['node_acc_v']:.3f}".ljust(20)
            print(row)


if __name__ == "__main__":
    main()
