"""Oxford-ordered FCC slip systems and rotation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]

_FCC_DIRECTIONS = np.array(
    [
        [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, -1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, -1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

_FCC_NORMALS = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
    ],
    dtype=np.float64,
)


def fcc_slip_systems() -> tuple[FloatArray, FloatArray]:
    """Return normalized ``<110>{111}`` systems in Oxford phase-2 order."""

    directions = _FCC_DIRECTIONS / np.linalg.norm(_FCC_DIRECTIONS, axis=1)[:, None]
    normals = _FCC_NORMALS / np.linalg.norm(_FCC_NORMALS, axis=1)[:, None]
    return directions, normals


def project_rotation(matrix: ArrayLike) -> FloatArray:
    """Project a 3-by-3 matrix to the nearest proper rotation."""

    candidate = np.asarray(matrix, dtype=np.float64)
    if candidate.shape != (3, 3):
        raise ValueError(f"Expected a 3-by-3 matrix, got {candidate.shape}")
    left, _, right_transpose = np.linalg.svd(candidate)
    if np.linalg.det(left @ right_transpose) < 0.0:
        left[:, -1] *= -1.0
    return left @ right_transpose


def rotation_from_rodrigues(vector: ArrayLike) -> FloatArray:
    """Convert a rotation vector to a proper rotation with Rodrigues' formula."""

    rotation_vector = np.asarray(vector, dtype=np.float64)
    if rotation_vector.shape != (3,):
        raise ValueError(f"Expected a three-component rotation vector, got {rotation_vector.shape}")
    angle = float(np.linalg.norm(rotation_vector))
    if angle == 0.0:
        return np.eye(3)
    axis = rotation_vector / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def schmid_tensors(rotation: ArrayLike) -> FloatArray:
    """Return ``s tensor m`` in the sample frame for each FCC system."""

    proper_rotation = project_rotation(rotation)
    crystal_directions, crystal_normals = fcc_slip_systems()
    directions = crystal_directions @ proper_rotation.T
    normals = crystal_normals @ proper_rotation.T
    return np.einsum("si,sj->sij", directions, normals)
