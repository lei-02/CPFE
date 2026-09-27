import numpy as np
import pytest

from pfem_cpfe.elements.c3d8 import (
    c3d8_kinematics,
    c3d8_quadrature,
    c3d8_response,
    c3d8_shape,
)


def test_shape_partition_and_zero_gradient_sum():
    shape, gradient = c3d8_shape(np.array([0.13, -0.21, 0.37]))

    np.testing.assert_allclose(shape.sum(), 1.0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=1.0e-14)


def test_quadrature_has_eight_symmetric_unit_weights():
    points, weights = c3d8_quadrature()

    assert points.shape == (8, 3)
    np.testing.assert_array_equal(weights, np.ones(8))
    np.testing.assert_allclose(points.sum(axis=0), 0.0, atol=1.0e-15)


def test_unit_cube_affine_displacement_has_constant_F(unit_cube):
    deformation_gradient = np.array(
        [[0.01, 0.002, 0.0], [0.0, -0.003, 0.0], [0.0, 0.0, 0.004]]
    )
    displacement = unit_cube.nodes @ deformation_gradient.T

    kinematics = c3d8_kinematics(unit_cube.nodes, displacement)

    np.testing.assert_allclose(
        kinematics.F,
        np.broadcast_to(np.eye(3) + deformation_gradient, (8, 3, 3)),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(kinematics.weights.sum(), 1.0, atol=1.0e-14)


def test_rigid_translation_produces_zero_internal_force(unit_cube):
    displacement = np.tile([0.1, -0.2, 0.3], (8, 1))

    result = c3d8_response(
        unit_cube.nodes,
        displacement,
        unit_cube.states,
        dt=1.0e-3,
        temperature=293.15,
        parameters=unit_cube.parameters,
    )

    assert result.converged
    np.testing.assert_allclose(result.internal_force, 0.0, atol=1.0e-9)
    assert result.tangent.shape == (24, 24)
    assert len(result.trial_states) == 8


def test_element_tangent_matches_internal_force_directional_derivative(unit_cube):
    affine = np.diag([1.0e-8, -2.0e-9, 3.0e-9])
    displacement = unit_cube.nodes @ affine.T
    direction = np.arange(24, dtype=np.float64).reshape(8, 3) - 11.5
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-8

    base = c3d8_response(
        unit_cube.nodes,
        displacement,
        unit_cube.states,
        dt=1.0e-3,
        temperature=293.15,
        parameters=unit_cube.parameters,
    )
    plus = c3d8_response(
        unit_cube.nodes,
        displacement + epsilon * direction,
        unit_cube.states,
        dt=1.0e-3,
        temperature=293.15,
        parameters=unit_cube.parameters,
    )
    minus = c3d8_response(
        unit_cube.nodes,
        displacement - epsilon * direction,
        unit_cube.states,
        dt=1.0e-3,
        temperature=293.15,
        parameters=unit_cube.parameters,
    )
    finite_difference = (plus.internal_force - minus.internal_force) / (2.0 * epsilon)

    np.testing.assert_allclose(
        base.tangent @ direction.ravel(),
        finite_difference,
        rtol=5.0e-5,
        atol=2.0e-5,
    )


def test_inverted_current_element_is_rejected(unit_cube):
    displacement = np.zeros_like(unit_cube.nodes)
    displacement[4:, 2] = -2.0

    with pytest.raises(ValueError, match="current Jacobian"):
        c3d8_kinematics(unit_cube.nodes, displacement)
