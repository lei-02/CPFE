"""Deterministic one-sample physics patch training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from pfem_cpfe.mesh.voxel import single_element_problem
from pfem_cpfe.operator.features import build_node_features
from pfem_cpfe.operator.transolver import TransolverCPFE
from pfem_cpfe.training.loss import relative_equilibrium_loss


@dataclass(frozen=True)
class PatchTrainingConfig:
    output: Path
    steps: int = 200
    learning_rate: float = 2.0e-4
    seed: int = 20260915
    hidden: int = 32
    layers: int = 2
    heads: int = 4
    slices: int = 8


def train_patch(config: PatchTrainingConfig) -> dict[str, float | bool]:
    """Overfit one unlabeled elastic CPFE patch and persist model/metrics."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    problem = single_element_problem(
        seed=5,
        axial_displacement=1.0e-6,
        dt=1.0e-3,
    )
    features = build_node_features(problem, problem.states, load_step=1.0)
    batch = {
        "problem": problem,
        "state_n": problem.states,
        "xyz": torch.tensor(problem.nodes[None], dtype=torch.float64),
        "features": torch.tensor(features[None], dtype=torch.float64),
        "mask": torch.ones((1, problem.nodes.shape[0]), dtype=torch.bool),
        "u_n": torch.zeros((problem.nodes.shape[0], 3), dtype=torch.float64),
        "output_scale": 1.0e-6,
    }
    model = TransolverCPFE(
        feature_dim=features.shape[1],
        hidden=config.hidden,
        layers=config.layers,
        heads=config.heads,
        slices=config.slices,
    ).double()
    torch.nn.init.zeros_(model.output.weight)
    torch.nn.init.zeros_(model.output.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    initial = None
    history = []
    for _ in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = relative_equilibrium_loss(model, batch)
        if initial is None:
            initial = metrics["relative_residual"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        history.append(metrics["relative_residual"])

    with torch.no_grad():
        _, final_metrics = relative_equilibrium_loss(model, batch)
    final = final_metrics["relative_residual"]
    history.append(final)

    config.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_dim": features.shape[1],
            "config": {
                "hidden": config.hidden,
                "layers": config.layers,
                "heads": config.heads,
                "slices": config.slices,
            },
        },
        config.output / "model.pt",
    )
    metrics = {
        "initial_relative_residual": float(initial),
        "final_relative_residual": float(final),
        "uses_solution_labels": False,
    }
    (config.output / "metrics.json").write_text(
        json.dumps({**metrics, "history": history}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics
