"""Global sparse assembly for standalone C3D8 CPFE problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix

from pfem_cpfe.contracts import Case2Parameters, MaterialState
from pfem_cpfe.elements.c3d8 import c3d8_response


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _readonly(value: ArrayLike, dtype=np.float64):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GlobalMaterialState:
    """Committed material states arranged by element and integration point."""

    points: tuple[tuple[MaterialState, ...], ...]

    def __post_init__(self) -> None:
        normalized = tuple(tuple(point for point in element) for element in self.points)
        if any(len(element) != 8 for element in normalized):
            raise ValueError("Every C3D8 element must contain eight material states")
        object.__setattr__(self, "points", normalized)

    @classmethod
    def initial(
        cls, element_count: int, parameters: Case2Parameters
    ) -> GlobalMaterialState:
        return cls(
            tuple(
                tuple(MaterialState.initial(parameters) for _ in range(8))
                for _ in range(element_count)
            )
        )

    @property
    def element_count(self) -> int:
        return len(self.points)

    def element(self, index: int) -> tuple[MaterialState, ...]:
        return self.points[index]

    def copied(self) -> GlobalMaterialState:
        return GlobalMaterialState(
            tuple(
                tuple(point.copied() for point in element) for element in self.points
            )
        )

    def _stack(self, name: str) -> FloatArray:
        if not self.points:
            return np.empty((0, 8, 0))
        return np.stack(
            [np.stack([getattr(point, name) for point in element]) for element in self.points]
        )

    @property
    def fp(self) -> FloatArray:
        return self._stack("fp")

    @property
    def orientation(self) -> FloatArray:
        return self._stack("orientation")

    @property
    def stress(self) -> FloatArray:
        return self._stack("stress")

    @property
    def ssd(self) -> FloatArray:
        return self._stack("ssd")

    @property
    def crss(self) -> FloatArray:
        return self._stack("crss")

    @property
    def slip(self) -> FloatArray:
        return self._stack("slip")

    @property
    def slip_rate(self) -> FloatArray:
        return self._stack("slip_rate")


@dataclass(frozen=True)
class CPFEProblem:
    """Mesh, load, constitutive, and committed-step data for one increment."""

    nodes: FloatArray
    elements: IntArray
    parameters: Case2Parameters
    states: GlobalMaterialState
    prescribed_dofs: IntArray
    prescribed_values: FloatArray
    previous_u: FloatArray
    external_force: FloatArray
    dt: float
    temperature: float
    grain_ids: IntArray | None = None
    orientations: FloatArray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", _readonly(self.nodes))
        object.__setattr__(self, "elements", _readonly(self.elements, np.int64))
        object.__setattr__(
            self, "prescribed_dofs", _readonly(self.prescribed_dofs, np.int64)
        )
        object.__setattr__(
            self, "prescribed_values", _readonly(self.prescribed_values)
        )
        object.__setattr__(self, "previous_u", _readonly(self.previous_u))
        object.__setattr__(self, "external_force", _readonly(self.external_force))
        if self.grain_ids is not None:
            object.__setattr__(self, "grain_ids", _readonly(self.grain_ids, np.int64))
        if self.orientations is not None:
            object.__setattr__(self, "orientations", _readonly(self.orientations))
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("Problem nodes must have shape (node_count, 3)")
        if self.elements.ndim != 2 or self.elements.shape[1] != 8:
            raise ValueError("Problem elements must have shape (element_count, 8)")
        if self.states.element_count != self.elements.shape[0]:
            raise ValueError("Material-state element count does not match mesh")
        if self.grain_ids is not None and self.grain_ids.shape != (
            self.elements.shape[0],
        ):
            raise ValueError("Grain IDs must contain one entry per element")
        if self.orientations is not None and (
            self.orientations.ndim != 3 or self.orientations.shape[1:] != (3, 3)
        ):
            raise ValueError("Grain orientations must have shape (grain_count, 3, 3)")
        if self.prescribed_dofs.shape != self.prescribed_values.shape:
            raise ValueError("Prescribed DOFs and values must have matching shapes")
        if self.previous_u.shape != (self.ndof,) or self.external_force.shape != (
            self.ndof,
        ):
            raise ValueError("Global vectors do not match the problem DOF count")
        if np.any(self.prescribed_dofs < 0) or np.any(
            self.prescribed_dofs >= self.ndof
        ):
            raise ValueError("A prescribed DOF is outside the global vector")
        if np.unique(self.prescribed_dofs).size != self.prescribed_dofs.size:
            raise ValueError("Prescribed DOFs must be unique")
        if self.dt <= 0.0 or self.temperature <= 0.0:
            raise ValueError("Time increment and temperature must be positive")

    @property
    def ndof(self) -> int:
        return 3 * self.nodes.shape[0]

    @property
    def free_dofs(self) -> IntArray:
        mask = np.ones(self.ndof, dtype=np.bool_)
        mask[self.prescribed_dofs] = False
        return np.flatnonzero(mask)

    @property
    def fixed_dofs(self) -> IntArray:
        """Alias used by operator-side boundary projection."""

        return self.prescribed_dofs

    @property
    def prescribed_increment(self) -> FloatArray:
        """Full global vector of displacement increments at constrained DOFs."""

        values = np.zeros(self.ndof)
        values[self.prescribed_dofs] = (
            self.prescribed_values - self.previous_u[self.prescribed_dofs]
        )
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class GlobalTrial:
    """One uncommitted global residual/tangent evaluation."""

    internal_force: FloatArray
    residual: FloatArray
    tangent: csr_matrix
    states: GlobalMaterialState
    converged: bool
    reason: str


def _element_dofs(connectivity: IntArray) -> IntArray:
    return np.array(
        [3 * int(node) + component for node in connectivity for component in range(3)],
        dtype=np.int64,
    )


def assemble_trial(
    problem: CPFEProblem,
    u_trial: ArrayLike,
    state_n: GlobalMaterialState,
) -> GlobalTrial:
    """Assemble an uncommitted global force, residual, and CSR tangent."""

    displacement = np.asarray(u_trial, dtype=np.float64)
    if displacement.shape != (problem.ndof,):
        raise ValueError("Trial displacement has the wrong global size")
    if state_n.element_count != problem.elements.shape[0]:
        raise ValueError("Trial state element count does not match the problem")

    internal_force = np.zeros(problem.ndof)
    row_parts: list[IntArray] = []
    column_parts: list[IntArray] = []
    data_parts: list[FloatArray] = []
    trial_points: list[tuple[MaterialState, ...]] = []
    current_nodes = displacement.reshape((-1, 3))
    previous_nodes = problem.previous_u.reshape((-1, 3))

    for element_index, connectivity in enumerate(problem.elements):
        dofs = _element_dofs(connectivity)
        try:
            response = c3d8_response(
                problem.nodes[connectivity],
                current_nodes[connectivity],
                state_n.element(element_index),
                dt=problem.dt,
                temperature=problem.temperature,
                parameters=problem.parameters,
                ue_n=previous_nodes[connectivity],
            )
        except ValueError as error:
            zero = csr_matrix((problem.ndof, problem.ndof))
            return GlobalTrial(
                internal_force=np.zeros(problem.ndof),
                residual=np.zeros(problem.ndof),
                tangent=zero,
                states=state_n.copied(),
                converged=False,
                reason=f"element {element_index}: {error}",
            )
        if not response.converged:
            zero = csr_matrix((problem.ndof, problem.ndof))
            return GlobalTrial(
                internal_force=np.zeros(problem.ndof),
                residual=np.zeros(problem.ndof),
                tangent=zero,
                states=state_n.copied(),
                converged=False,
                reason=f"element {element_index}: {response.reason}",
            )
        internal_force[dofs] += response.internal_force
        row_parts.append(np.repeat(dofs, 24))
        column_parts.append(np.tile(dofs, 24))
        data_parts.append(response.tangent.ravel())
        trial_points.append(response.trial_states)

    if data_parts:
        tangent = coo_matrix(
            (
                np.concatenate(data_parts),
                (np.concatenate(row_parts), np.concatenate(column_parts)),
            ),
            shape=(problem.ndof, problem.ndof),
        ).tocsr()
    else:
        tangent = csr_matrix((problem.ndof, problem.ndof))
    residual = internal_force - problem.external_force
    return GlobalTrial(
        internal_force=internal_force,
        residual=residual,
        tangent=tangent,
        states=GlobalMaterialState(tuple(trial_points)),
        converged=True,
        reason="",
    )
