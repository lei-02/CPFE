from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from pfem_cpfe.mesh.voxel import single_element_problem
from pfem_cpfe.operator.features import build_node_features
from pfem_cpfe.operator.transolver import TransolverCPFE


def _training_case(axial_displacement: float):
    problem = single_element_problem(
        seed=5,
        axial_displacement=axial_displacement,
        dt=1.0e-3,
    )
    features = build_node_features(problem, problem.states, load_step=1.0)
    return {
        "problem": problem,
        "state_n": problem.states,
        "xyz": torch.tensor(problem.nodes[None], dtype=torch.float64),
        "features": torch.tensor(features[None], dtype=torch.float64),
        "mask": torch.ones((1, problem.nodes.shape[0]), dtype=torch.bool),
        "u_n": torch.zeros((problem.nodes.shape[0], 3), dtype=torch.float64),
        "output_scale": max(abs(axial_displacement), 1.0e-6),
    }


@pytest.fixture
def unlabeled_case():
    return _training_case(1.0e-6)


@pytest.fixture
def equilibrated_case():
    return _training_case(0.0)


@pytest.fixture
def model(unlabeled_case):
    torch.manual_seed(29)
    return TransolverCPFE(
        feature_dim=unlabeled_case["features"].shape[-1],
        hidden=16,
        layers=1,
        heads=4,
        slices=4,
    ).double()


@pytest.fixture
def identity_predictor():
    class IdentityPredictor(nn.Module):
        def forward(self, xyz, features, mask):
            return torch.zeros(
                (*xyz.shape[:2], 3), dtype=xyz.dtype, device=xyz.device
            )

    return IdentityPredictor()
