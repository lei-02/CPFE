"""Seeded single-crystal and eight-grain C3D8 voxel problems."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pfem_cpfe.contracts import Case2Parameters, MaterialState
from pfem_cpfe.solver.assembly import CPFEProblem, GlobalMaterialState


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _voxel_mesh(cells: tuple[int, int, int]) -> tuple[FloatArray, IntArray]:
    nx, ny, nz = cells
    if min(cells) <= 0:
        raise ValueError("Voxel cell counts must be positive")
    nodes = np.array(
        [
            (i / nx, j / ny, k / nz)
            for k in range(nz + 1)
            for j in range(ny + 1)
            for i in range(nx + 1)
        ],
        dtype=np.float64,
    )

    def node(i: int, j: int, k: int) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append(
                    (
                        node(i, j, k),
                        node(i + 1, j, k),
                        node(i + 1, j + 1, k),
                        node(i, j + 1, k),
                        node(i, j, k + 1),
                        node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1),
                        node(i, j + 1, k + 1),
                    )
                )
    return nodes, np.asarray(elements, dtype=np.int64)


def _uniform_rotations(rng: np.random.Generator, count: int) -> FloatArray:
    values = rng.random((count, 3))
    u1, u2, u3 = values.T
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    rotations = np.empty((count, 3, 3))
    rotations[:, 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    rotations[:, 0, 1] = 2.0 * (qx * qy - qz * qw)
    rotations[:, 0, 2] = 2.0 * (qx * qz + qy * qw)
    rotations[:, 1, 0] = 2.0 * (qx * qy + qz * qw)
    rotations[:, 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    rotations[:, 1, 2] = 2.0 * (qy * qz - qx * qw)
    rotations[:, 2, 0] = 2.0 * (qx * qz - qy * qw)
    rotations[:, 2, 1] = 2.0 * (qy * qz + qx * qw)
    rotations[:, 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    return rotations


def _states_for_grains(
    grain_ids: IntArray,
    orientations: FloatArray,
    parameters: Case2Parameters,
) -> GlobalMaterialState:
    return GlobalMaterialState(
        tuple(
            tuple(
                MaterialState.initial(parameters).copied(
                    orientation=orientations[int(grain_ids[element])]
                )
                for _ in range(8)
            )
            for element in range(grain_ids.size)
        )
    )


def _boundary_data(
    nodes: FloatArray, axial_displacement: float
) -> tuple[IntArray, FloatArray]:
    bottom = np.flatnonzero(np.isclose(nodes[:, 2], 0.0))
    top = np.flatnonzero(np.isclose(nodes[:, 2], 1.0))
    origin = int(np.argmin(np.linalg.norm(nodes, axis=1)))
    x_anchor_candidates = np.flatnonzero(
        np.isclose(nodes[:, 1], 0.0) & np.isclose(nodes[:, 2], 0.0)
    )
    x_anchor = int(x_anchor_candidates[np.argmax(nodes[x_anchor_candidates, 0])])
    values: dict[int, float] = {}
    for node in bottom:
        values[3 * int(node) + 2] = 0.0
    for node in top:
        values[3 * int(node) + 2] = axial_displacement
    values[3 * origin] = 0.0
    values[3 * origin + 1] = 0.0
    values[3 * x_anchor + 1] = 0.0
    dofs = np.array(sorted(values), dtype=np.int64)
    prescribed = np.array([values[int(dof)] for dof in dofs], dtype=np.float64)
    return dofs, prescribed


def _problem(
    nodes: FloatArray,
    elements: IntArray,
    grain_ids: IntArray,
    orientations: FloatArray,
    *,
    axial_displacement: float,
    dt: float,
    temperature: float,
) -> CPFEProblem:
    parameters = Case2Parameters.default()
    states = _states_for_grains(grain_ids, orientations, parameters)
    prescribed_dofs, prescribed_values = _boundary_data(
        nodes, axial_displacement
    )
    ndof = 3 * nodes.shape[0]
    return CPFEProblem(
        nodes=nodes,
        elements=elements,
        parameters=parameters,
        states=states,
        prescribed_dofs=prescribed_dofs,
        prescribed_values=prescribed_values,
        previous_u=np.zeros(ndof),
        external_force=np.zeros(ndof),
        dt=dt,
        temperature=temperature,
        grain_ids=grain_ids,
        orientations=orientations,
    )


def single_element_problem(
    seed: int = 20260915,
    *,
    axial_displacement: float = 0.005,
    dt: float = 1.0e-8,
    temperature: float = 293.15,
) -> CPFEProblem:
    """Create one deterministically oriented C3D8 crystal."""

    nodes, elements = _voxel_mesh((1, 1, 1))
    rng = np.random.default_rng(seed)
    orientations = _uniform_rotations(rng, 1)
    return _problem(
        nodes,
        elements,
        np.zeros(1, dtype=np.int64),
        orientations,
        axial_displacement=axial_displacement,
        dt=dt,
        temperature=temperature,
    )


def eight_grain_problem(
    seed: int = 20260915,
    cells: tuple[int, int, int] = (4, 4, 4),
    *,
    axial_displacement: float = 0.005,
    dt: float = 1.0e-8,
    temperature: float = 293.15,
) -> CPFEProblem:
    """Create a reproducible seeded-Voronoi eight-grain voxel aggregate."""

    nodes, elements = _voxel_mesh(cells)
    centroids = nodes[elements].mean(axis=1)
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        seeds = 0.05 + 0.90 * rng.random((8, 3))
        orientations = _uniform_rotations(rng, 8)
        distances = np.sum((centroids[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
        grain_ids = np.argmin(distances, axis=1).astype(np.int64)
        if np.unique(grain_ids).size == 8:
            break
    else:
        raise RuntimeError("Could not construct a nonempty deterministic grain partition")
    return _problem(
        nodes,
        elements,
        grain_ids,
        orientations,
        axial_displacement=axial_displacement,
        dt=dt,
        temperature=temperature,
    )
