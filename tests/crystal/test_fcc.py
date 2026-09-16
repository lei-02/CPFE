import numpy as np

from pfem_cpfe.crystal.fcc import (
    fcc_slip_systems,
    project_rotation,
    rotation_from_rodrigues,
    schmid_tensors,
)


def test_fcc_has_twelve_normalized_orthogonal_systems():
    directions, normals = fcc_slip_systems()

    assert directions.shape == normals.shape == (12, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    np.testing.assert_allclose(
        np.sum(directions * normals, axis=1),
        0.0,
        atol=1e-14,
    )


def test_fcc_order_starts_with_oxford_phase_two_entries():
    directions, normals = fcc_slip_systems()

    np.testing.assert_allclose(directions[0], [1 / np.sqrt(2), -1 / np.sqrt(2), 0])
    np.testing.assert_allclose(normals[0], np.ones(3) / np.sqrt(3))
    np.testing.assert_allclose(directions[3], [1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    np.testing.assert_allclose(normals[3], [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)])


def test_orientation_projection_returns_proper_rotation():
    rotation = project_rotation(np.eye(3) + 1e-4 * np.ones((3, 3)))

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) > 0.0
    assert schmid_tensors(rotation).shape == (12, 3, 3)


def test_rodrigues_rotation_vector_rotates_about_its_axis():
    rotation = rotation_from_rodrigues([0.0, 0.0, np.pi / 2])

    np.testing.assert_allclose(rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-14)
