"""Oxford/Abaqus-ordered C3D8 element with full 2 x 2 x 2 integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pfem_cpfe.contracts import Case2Parameters, MaterialState
from pfem_cpfe.materials.case2_numpy import case2_step_numpy


FloatArray = NDArray[np.float64]
_NODE_SIGNS = np.array(
    [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]
)


@dataclass(frozen=True)
class C3D8Kinematics:
    """Kinematic values at the eight full-integration points."""

    F: FloatArray
    weights: FloatArray
    gradients: FloatArray
    reference_jacobian_determinants: FloatArray
    current_jacobian_determinants: FloatArray


@dataclass(frozen=True)
class C3D8Response:
    """Uncommitted element force, tangent, and integration-point states."""

    F: FloatArray
    weights: FloatArray
    internal_force: FloatArray
    tangent: FloatArray
    trial_states: tuple[MaterialState, ...]
    converged: bool
    reason: str


def c3d8_quadrature() -> tuple[FloatArray, FloatArray]:
    """Return the eight Gauss points and natural-space unit weights."""

    coordinate = 1.0 / np.sqrt(3.0)
    points = _NODE_SIGNS * coordinate
    return points.copy(), np.ones(8)


def c3d8_shape(xi: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Evaluate trilinear shape functions and natural derivatives."""

    natural = np.asarray(xi, dtype=np.float64)
    if natural.shape != (3,):
        raise ValueError("C3D8 natural coordinate must have three components")
    factors = 1.0 + _NODE_SIGNS * natural
    shape = 0.125 * np.prod(factors, axis=1)
    gradient = np.empty((8, 3))
    gradient[:, 0] = (
        0.125 * _NODE_SIGNS[:, 0] * factors[:, 1] * factors[:, 2]
    )
    gradient[:, 1] = (
        0.125 * factors[:, 0] * _NODE_SIGNS[:, 1] * factors[:, 2]
    )
    gradient[:, 2] = (
        0.125 * factors[:, 0] * factors[:, 1] * _NODE_SIGNS[:, 2]
    )
    return shape, gradient


def c3d8_kinematics(Xe: ArrayLike, ue: ArrayLike) -> C3D8Kinematics:
    """Compute reference gradients, weights, and deformation gradients."""

    nodes = np.asarray(Xe, dtype=np.float64)
    displacement = np.asarray(ue, dtype=np.float64)
    if nodes.shape != (8, 3) or displacement.shape != (8, 3):
        raise ValueError("C3D8 coordinates and displacements must have shape (8, 3)")
    points, natural_weights = c3d8_quadrature()
    deformation_gradients = np.empty((8, 3, 3))
    gradients = np.empty((8, 8, 3))
    weights = np.empty(8)
    reference_determinants = np.empty(8)
    current_determinants = np.empty(8)
    current_nodes = nodes + displacement

    for point_index, point in enumerate(points):
        _, natural_gradient = c3d8_shape(point)
        reference_jacobian = nodes.T @ natural_gradient
        reference_determinant = float(np.linalg.det(reference_jacobian))
        if not np.isfinite(reference_determinant) or reference_determinant <= 0.0:
            raise ValueError(
                f"C3D8 reference Jacobian is nonpositive at point {point_index}"
            )
        current_jacobian = current_nodes.T @ natural_gradient
        current_determinant = float(np.linalg.det(current_jacobian))
        if not np.isfinite(current_determinant) or current_determinant <= 0.0:
            raise ValueError(
                f"C3D8 current Jacobian is nonpositive at point {point_index}"
            )
        reference_gradient = natural_gradient @ np.linalg.inv(reference_jacobian)
        displacement_gradient = displacement.T @ reference_gradient
        deformation_gradients[point_index] = np.eye(3) + displacement_gradient
        gradients[point_index] = reference_gradient
        reference_determinants[point_index] = reference_determinant
        current_determinants[point_index] = current_determinant
        weights[point_index] = natural_weights[point_index] * reference_determinant

    return C3D8Kinematics(
        F=deformation_gradients,
        weights=weights,
        gradients=gradients,
        reference_jacobian_determinants=reference_determinants,
        current_jacobian_determinants=current_determinants,
    )


def _strain_displacement(gradient: FloatArray) -> FloatArray:
    matrix = np.zeros((6, 24))
    for node, (dx, dy, dz) in enumerate(gradient):
        column = 3 * node
        matrix[0, column] = dx
        matrix[1, column + 1] = dy
        matrix[2, column + 2] = dz
        matrix[3, column] = dy
        matrix[3, column + 1] = dx
        matrix[4, column] = dz
        matrix[4, column + 2] = dx
        matrix[5, column + 1] = dz
        matrix[5, column + 2] = dy
    return matrix


def c3d8_response(
    Xe: ArrayLike,
    ue: ArrayLike,
    states_n: Sequence[MaterialState],
    *,
    dt: float,
    temperature: float,
    parameters: Case2Parameters,
    ue_n: ArrayLike | None = None,
) -> C3D8Response:
    """Assemble an uncommitted C3D8 constitutive response."""

    if len(states_n) != 8:
        raise ValueError("C3D8 requires exactly eight integration-point states")
    current = c3d8_kinematics(Xe, ue)
    previous_displacement = (
        np.zeros((8, 3)) if ue_n is None else np.asarray(ue_n, dtype=np.float64)
    )
    previous = c3d8_kinematics(Xe, previous_displacement)
    internal_force = np.zeros(24)
    tangent = np.zeros((24, 24))
    trial_states: list[MaterialState] = []

    for point in range(8):
        trial = case2_step_numpy(
            previous.F[point],
            current.F[point],
            dt,
            temperature,
            parameters,
            states_n[point],
        )
        if not trial.converged:
            return C3D8Response(
                F=current.F,
                weights=current.weights,
                internal_force=np.zeros(24),
                tangent=np.zeros((24, 24)),
                trial_states=tuple(states_n),
                converged=False,
                reason=f"integration point {point}: {trial.reason}",
            )
        strain_displacement = _strain_displacement(current.gradients[point])
        weight = current.weights[point]
        internal_force += strain_displacement.T @ trial.stress * weight
        tangent += strain_displacement.T @ trial.tangent @ strain_displacement * weight
        trial_states.append(trial.state)

    return C3D8Response(
        F=current.F,
        weights=current.weights,
        internal_force=internal_force,
        tangent=tangent,
        trial_states=tuple(trial_states),
        converged=True,
        reason="",
    )
