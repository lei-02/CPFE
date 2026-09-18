from dataclasses import dataclass

import numpy as np
import pytest
import torch

from pfem_cpfe.mesh.voxel import eight_grain_problem, single_element_problem
from pfem_cpfe.operator.features import build_node_features


@dataclass(frozen=True)
class NodeBatch:
    xyz: torch.Tensor
    features: torch.Tensor
    mask: torch.Tensor


@pytest.fixture
def problem():
    return single_element_problem(seed=11, axial_displacement=0.002)


@pytest.fixture
def raw_du(problem):
    rng = np.random.default_rng(17)
    return rng.normal(scale=1.0e-3, size=(problem.nodes.shape[0], 3))


@pytest.fixture
def node_batch():
    problems = [
        single_element_problem(seed=3),
        eight_grain_problem(seed=20260915),
    ]
    node_count = max(problem.nodes.shape[0] for problem in problems)
    feature_dim = build_node_features(problems[0], problems[0].states, 0.25).shape[1]
    xyz = np.zeros((2, node_count, 3), dtype=np.float32)
    features = np.zeros((2, node_count, feature_dim), dtype=np.float32)
    mask = np.zeros((2, node_count), dtype=np.bool_)
    for batch, current in enumerate(problems):
        count = current.nodes.shape[0]
        xyz[batch, :count] = current.nodes
        features[batch, :count] = build_node_features(
            current, current.states, 0.25
        )
        mask[batch, :count] = True
    return NodeBatch(
        xyz=torch.from_numpy(xyz),
        features=torch.from_numpy(features),
        mask=torch.from_numpy(mask),
    )
