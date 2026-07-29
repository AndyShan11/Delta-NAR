"""Evaluate an official static DEAR Bellman-Ford checkpoint on CLRS-Delta graphs.

The adapter supplies only the post-edit weighted graph and source node. It does
not expose the previous solution or edit metadata, so DEAR remains a
from-scratch recomputation baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch, Data

from datasets.constants import _DATASET_ROOTS
from models.algorithm_processor import LitAlgorithmProcessor


def delta_data(sample: dict, position_rng: np.random.Generator, weight_hi: float) -> Data:
    n = int(sample["n"])
    adjacency = torch.from_numpy(sample["A"]).bool()
    edge_index = adjacency.nonzero().T.contiguous()
    edge_index, _ = torch_geometric.utils.add_remaining_self_loops(
        edge_index, num_nodes=n
    )
    edge_index = torch_geometric.utils.coalesce(edge_index)

    weights = torch.from_numpy(sample["W"]).float() / float(weight_hi)
    position = np.empty(n, dtype=np.float32)
    position[position_rng.permutation(n)] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    source = torch.zeros(n, dtype=torch.float32)
    source[int(sample["source"])] = 1.0
    predecessor = torch.from_numpy(sample["pred_next"].copy()).long()
    predecessor[predecessor < 0] = torch.arange(n)[predecessor < 0]

    return Data(
        pos=torch.from_numpy(position),
        s=source,
        edge_index=edge_index,
        A=weights[edge_index[0], edge_index[1]],
        num_nodes=n,
        lengths=torch.tensor([2.0]),
        pi_h_index_temporal=torch.arange(n, dtype=torch.long).view(1, n),
        d_temporal=torch.zeros((n, 1), dtype=torch.float32),
        msk_temporal=torch.zeros((n, 1), dtype=torch.float32),
        pi_index=predecessor,
    )


def make_samples(
    task: str,
    delta_code_root: str,
    episodes: int,
    n: int,
    seed0: int,
    delta_size: int,
    weight_hi: int,
) -> list[dict]:
    sys.path.insert(0, delta_code_root)
    if task == "sssp":
        from src.dataset import make_dataset

        maker = make_dataset
    else:
        from src.dataset_bottleneck import make_dataset_bottleneck

        maker = make_dataset_bottleneck
    return maker(
        episodes,
        n,
        n_steps=5,
        seed0=seed0,
        delta_size=delta_size,
        weight_range=(1, weight_hi),
    )


@torch.no_grad()
def predict_predecessors(model, batch: Batch) -> torch.Tensor:
    algorithm = model.algorithms["bellman_ford"]
    (
        _,
        _,
        output_logits,
        _,
        _,
        _,
    ) = algorithm.fwd_step(batch, 0, hint_mode="none")
    converted = type(algorithm.algorithm_module).convert_logits_to_outputs(
        algorithm.dataset.spec,
        output_logits,
        batch.edge_index[0],
        batch.edge_index[1],
        batch.num_nodes,
        batch.batch,
        include_probabilities=False,
    )["output"]
    return converted["pi_index"]


def evaluate_batch(samples: list[dict], predictions: torch.Tensor) -> dict[str, float]:
    offset = 0
    node_valid = []
    graph_valid = []
    exact = []
    changed_valid = []
    copy_critical_valid = []
    for sample in samples:
        n = int(sample["n"])
        pred = predictions[offset : offset + n].detach().cpu().numpy().astype(np.int64)
        offset += n
        target = sample["pred_next"]
        distance = sample["g_next"]
        previous = sample["pred_prev"]
        weights = sample["W"]
        source = int(sample["source"])
        reachable = np.isfinite(distance)
        mask = reachable.copy()
        mask[source] = False
        vertices = np.arange(n)
        pred = np.clip(pred, 0, n - 1)
        valid = np.zeros(n, dtype=bool)
        valid[~mask] = True
        valid[mask] = (
            np.abs(distance[pred[mask]] + weights[pred[mask], vertices[mask]] - distance[mask])
            < 0.5
        )
        previous_valid = np.zeros(n, dtype=bool)
        previous_valid[~mask] = True
        usable_previous = mask & (previous >= 0)
        previous_valid[usable_previous] = (
            np.abs(
                distance[previous[usable_previous]]
                + weights[previous[usable_previous], vertices[usable_previous]]
                - distance[usable_previous]
            )
            < 0.5
        )
        changed = mask & (
            (np.abs(sample["g_prev"] - sample["g_next"]) >= 0.5)
            | (sample["pred_prev"] != sample["pred_next"])
        )
        critical = mask & ~previous_valid
        node_valid.append(float(valid[mask].mean()) if mask.any() else 1.0)
        graph_valid.append(float(valid[mask].all()))
        exact.append(float((pred[mask] == target[mask]).mean()) if mask.any() else 1.0)
        if changed.any():
            changed_valid.append(float(valid[changed].mean()))
        if critical.any():
            copy_critical_valid.append(float(valid[critical].mean()))
    return {
        "node_acc_v": float(np.mean(node_valid)),
        "graph_acc_v": float(np.mean(graph_valid)),
        "exact_predecessor_acc": float(np.mean(exact)),
        "changed_output_acc_v": float(np.mean(changed_valid)) if changed_valid else None,
        "copy_critical_acc_v": (
            float(np.mean(copy_critical_valid)) if copy_critical_valid else None
        ),
        "graphs": len(samples),
        "graphs_with_changed_output": len(changed_valid),
        "graphs_with_copy_critical_node": len(copy_critical_valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--delta-code-root", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=900000)
    parser.add_argument("--delta-size", type=int, default=1)
    parser.add_argument("--weight-hi", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitAlgorithmProcessor.load_from_checkpoint(
        args.checkpoint,
        dataset_root=_DATASET_ROOTS["bellman_ford"],
        strict=False,
    ).to(device)
    model.eval()

    results = {}
    started = time.time()
    for n in args.sizes:
        samples = make_samples(
            "sssp",
            args.delta_code_root,
            args.episodes,
            n,
            args.seed0,
            args.delta_size,
            args.weight_hi,
        )
        position_rng = np.random.default_rng(args.seed0 + n)
        totals: dict[str, list[float]] = {}
        for start in range(0, len(samples), args.batch_size):
            chunk = samples[start : start + args.batch_size]
            batch = Batch.from_data_list(
                [delta_data(s, position_rng, args.weight_hi) for s in chunk],
                follow_batch=["edge_index"],
            ).to(device)
            pred = predict_predecessors(model, batch)
            metrics = evaluate_batch(chunk, pred)
            for key, value in metrics.items():
                if value is not None:
                    totals.setdefault(key, []).append(float(value))
        results[str(n)] = {
            key: float(np.mean(values)) for key, values in totals.items()
        }
        print(n, results[str(n)], flush=True)

    Path(args.output).write_text(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": args.checkpoint,
                "protocol": "official static DEAR; post-edit graph only",
                "args": vars(args),
                "results": results,
                "elapsed_seconds": time.time() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
