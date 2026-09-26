import numpy as np
import torch

from pfem_cpfe.operator.features import (
    FeatureLayout,
    build_node_features,
    enforce_displacement_bc,
)


def test_fixed_feature_layout_contains_all_cpfe_prompt_groups(problem):
    layout = FeatureLayout.default()
    features = build_node_features(problem, problem.states, load_step=0.4)

    assert features.shape == (problem.nodes.shape[0], layout.width)
    np.testing.assert_allclose(
        features[:, layout.dt], np.log10(problem.dt / 1.0e-3)
    )
    assert np.max(np.abs(features)) < 100.0
    assert layout.orientation.stop - layout.orientation.start == 9
    assert layout.fp.stop - layout.fp.start == 9
    assert layout.slip.stop - layout.slip.start == 12
    assert layout.ssd.stop - layout.ssd.start == 12
    assert layout.crss.stop - layout.crss.start == 12
    assert layout.material.stop - layout.material.start >= 14
    assert np.all((features[:, layout.coordinates] >= 0.0) & (features[:, layout.coordinates] <= 1.0))


def test_prompt_projection_does_not_mutate_committed_state(problem):
    before = problem.states.ssd.copy()

    features = build_node_features(problem, problem.states, load_step=0.5)

    assert np.all(np.isfinite(features))
    np.testing.assert_array_equal(problem.states.ssd, before)


def test_boundary_projection_is_exact_for_numpy(problem, raw_du):
    displacement = enforce_displacement_bc(raw_du, problem)
    flattened = displacement.reshape(-1)

    np.testing.assert_allclose(
        flattened[problem.fixed_dofs],
        problem.prescribed_increment[problem.fixed_dofs],
    )


def test_boundary_projection_is_exact_and_differentiable_for_torch(problem, raw_du):
    raw = torch.tensor(raw_du, dtype=torch.float64, requires_grad=True)

    displacement = enforce_displacement_bc(raw, problem)
    displacement.sum().backward()

    flattened = displacement.detach().reshape(-1)
    expected = torch.tensor(problem.prescribed_increment.tolist(), dtype=raw.dtype)
    fixed = torch.tensor(problem.fixed_dofs.tolist(), dtype=torch.int64)
    torch.testing.assert_close(flattened[fixed], expected[fixed])
    assert raw.grad is not None
    assert torch.all(raw.grad.reshape(-1)[fixed] == 0.0)
