"""Train official DEAR as a matched post-edit-graph recomputation baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from clrs import Location, Stage, Type

os.environ.setdefault("WANDB_MODE", "disabled")

from datasets import constants as dataset_constants
from datasets._configs import CONFIGS
from datasets.specs import SPECS
from train_reasoner import train


DELTA_CODE_ROOT = ""
MODEL_SEED = 0
MATCHED_BELLMAN_FORD_SPEC = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
    "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
    "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
    "pi_h_index_temporal": (Stage.HINT, Location.NODE, Type.POINTER),
    "d_temporal": (Stage.HINT, Location.NODE, Type.SCALAR),
    "msk_temporal": (Stage.HINT, Location.NODE, Type.MASK),
    "pi_index": (Stage.OUTPUT, Location.NODE, Type.POINTER),
}


def to_dear_data(sample: dict, position_seed: int, weight_hi: float = 10.0) -> Data:
    n = int(sample["n"])
    adjacency = torch.from_numpy(sample["A"]).bool()
    edge_index = adjacency.nonzero().T.contiguous()
    edge_index, _ = torch_geometric.utils.add_remaining_self_loops(
        edge_index, num_nodes=n
    )
    edge_index = torch_geometric.utils.coalesce(edge_index)
    weights = torch.from_numpy(sample["W"]).float() / weight_hi

    rng = np.random.default_rng(position_seed)
    pos = np.empty(n, dtype=np.float32)
    pos[rng.permutation(n)] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    source = torch.zeros(n, dtype=torch.float32)
    source[int(sample["source"])] = 1.0
    predecessor = torch.from_numpy(sample["pred_next"].copy()).long()
    predecessor[predecessor < 0] = torch.arange(n)[predecessor < 0]

    return Data(
        pos=torch.from_numpy(pos),
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


class DeltaPostEditDataset(Dataset):
    """DEAR-compatible dataset containing only post-edit graph inputs."""

    def __init__(
        self,
        root: str,
        num_nodes: int,
        num_samples: int,
        algorithm: str = "bellman_ford",
        split: str = "train",
        **_: object,
    ) -> None:
        super().__init__(root=None)
        if algorithm != "bellman_ford":
            raise ValueError(algorithm)
        sys.path.insert(0, DELTA_CODE_ROOT)
        from src.dataset import make_dataset

        if split == "train":
            seed0 = 1 + MODEL_SEED * 7919
        elif split == "val":
            seed0 = 800000
        else:
            seed0 = 900000
        episodes = math.ceil(num_samples / 5)
        samples = make_dataset(
            episodes,
            num_nodes,
            n_steps=5,
            seed0=seed0,
            delta_size=1,
            weight_range=(1, 10),
        )[:num_samples]
        self.items = [
            to_dear_data(sample, position_seed=seed0 * 7 + i)
            for i, sample in enumerate(samples)
        ]
        self.spec = MATCHED_BELLMAN_FORD_SPEC
        self.attribute_list = list(self.spec)
        self.split = split
        self.algorithm = algorithm
        self.num_nodes = num_nodes
        self.num_samples = num_samples

    def len(self) -> int:
        return len(self.items)

    def get(self, idx: int) -> Data:
        return self.items[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--delta-code-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=100)
    parser.add_argument("--test-samples", type=int, default=100)
    args = parser.parse_args()

    global DELTA_CODE_ROOT, MODEL_SEED
    DELTA_CODE_ROOT = args.delta_code_root
    MODEL_SEED = args.seed
    dataset_constants._DATASET_SPECS["bellman_ford"]["dataclass"] = DeltaPostEditDataset
    dataset_constants._DATASET_SPECS["bellman_ford"]["rootdir"] = "delta-post-edit"
    CONFIGS["bellman_ford"] = {
        "train": {"num_samples": args.train_samples, "num_nodes": 16},
        "val": {"num_samples": args.val_samples, "num_nodes": 16},
        "test": {"num_samples": args.test_samples, "num_nodes": 64},
    }

    train_args = {
        "--help": False,
        "--xavier-on-scalars": False,
        "--biased-gate": False,
        "--update-edges-hidden": False,
        "--use-LSTM": False,
        "--patience": None,
        "--model-name": f"dear_delta_matched_bellman_ford_seed{args.seed}",
        "--processors": ["MPNN"],
        "--RPHWM": False,
        "--gradient-clip-val": 1,
        "--algorithms": ["bellman_ford"],
        "--seed": args.seed,
        "--config-path": args.config,
        "--agentify": None,
    }
    started = time.time()
    status = "ok"
    error = None
    try:
        train(None, args=train_args)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        Path(args.output).write_text(
            json.dumps(
                {
                    "status": status,
                    "error": error,
                    "elapsed_seconds": time.time() - started,
                    "seed": args.seed,
                    "protocol": "official DEAR architecture trained on matched post-edit graphs",
                    "inputs": "post-edit weighted graph and source only",
                    "previous_solution_visible": False,
                    "edit_metadata_visible": False,
                    "train_samples": args.train_samples,
                    "train_n": 16,
                    "test_n": 64,
                    "epochs": 100,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
