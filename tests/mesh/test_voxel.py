import numpy as np

from pfem_cpfe.elements.c3d8 import c3d8_kinematics
from pfem_cpfe.mesh.voxel import eight_grain_problem, single_element_problem


def test_eight_grain_mesh_is_reproducible():
    first = eight_grain_problem(seed=20260915)
    second = eight_grain_problem(seed=20260915)

    assert first.nodes.shape == (125, 3)
    assert first.elements.shape == (64, 8)
    assert set(np.unique(first.grain_ids)) == set(range(8))
    np.testing.assert_array_equal(first.grain_ids, second.grain_ids)
    np.testing.assert_allclose(first.orientations, second.orientations)


def test_seeded_grain_orientations_are_proper_rotations():
    problem = eight_grain_problem(seed=20260915)

    np.testing.assert_allclose(
        problem.orientations @ problem.orientations.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), (8, 3, 3)),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(np.linalg.det(problem.orientations), 1.0, atol=1.0e-13)


def test_voxel_connectivity_uses_positive_c3d8_order():
    problem = eight_grain_problem(seed=20260915)

    for connectivity in problem.elements:
        kinematics = c3d8_kinematics(
            problem.nodes[connectivity], np.zeros((8, 3))
        )
        assert np.all(kinematics.reference_jacobian_determinants > 0.0)


def test_axial_boundary_data_has_bottom_fixity_and_top_displacement():
    problem = eight_grain_problem(seed=20260915, axial_displacement=0.005)
    coordinates = problem.nodes
    bottom = np.flatnonzero(np.isclose(coordinates[:, 2], 0.0))
    top = np.flatnonzero(np.isclose(coordinates[:, 2], 1.0))
    prescribed = dict(zip(problem.prescribed_dofs.tolist(), problem.prescribed_values.tolist()))

    assert all(prescribed[3 * node + 2] == 0.0 for node in bottom)
    assert all(prescribed[3 * node + 2] == 0.005 for node in top)
    assert prescribed[0] == prescribed[1] == 0.0


def test_single_element_problem_has_one_state_block():
    problem = single_element_problem(seed=7)

    assert problem.elements.shape == (1, 8)
    assert problem.states.ssd.shape == (1, 8, 12)
