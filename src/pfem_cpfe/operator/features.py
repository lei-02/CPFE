"""Fixed CPFE node prompts and exact displacement-boundary projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pfem_cpfe.crystal.fcc import project_rotation
from pfem_cpfe.elements.c3d8 import c3d8_kinematics
from pfem_cpfe.solver.assembly import CPFEProblem, GlobalMaterialState


@dataclass(frozen=True)
class FeatureLayout:
    """Stable column layout for every Transolver CPFE prompt."""

    coordinates: slice
    prescribed_mask: slice
    prescribed_values: slice
    orientation: slice
    fp: slice
    slip: slice
    ssd: slice
    crss: slice
    temperature: slice
    dt: slice
    load_fraction: slice
    material: slice
    material_names: tuple[str, ...]

    @property
    def width(self) -> int:
        return self.material.stop

    @classmethod
    def default(cls) -> FeatureLayout:
        cursor = 0

        def take(width: int) -> slice:
            nonlocal cursor
            result = slice(cursor, cursor + width)
            cursor += width
            return result

        material_names = (
            "c11",
            "c12",
            "c44",
            "gamma0",
            "inner_exponent",
            "outer_exponent",
            "activation_energy_octahedral",
            "boltzmann",
            "rho0",
            "tauc0",
            "burgers",
            "k1",
            "k2",
            "geometric_factor",
        )
        return cls(
            coordinates=take(3),
            prescribed_mask=take(3),
            prescribed_values=take(3),
            orientation=take(9),
            fp=take(9),
            slip=take(12),
            ssd=take(12),
            crss=take(12),
            temperature=take(1),
            dt=take(1),
            load_fraction=take(1),
            material=take(len(material_names)),
            material_names=material_names,
        )


def _project_state_to_nodes(
    problem: CPFEProblem, state_n: GlobalMaterialState
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_count = problem.nodes.shape[0]
    accumulators = {
        "orientation": np.zeros((node_count, 3, 3)),
        "fp": np.zeros((node_count, 3, 3)),
        "slip": np.zeros((node_count, 12)),
        "ssd": np.zeros((node_count, 12)),
        "crss": np.zeros((node_count, 12)),
    }
    weights = np.zeros(node_count)
    for element_index, connectivity in enumerate(problem.elements):
        element_state = state_n.element(element_index)
        volume = float(
            c3d8_kinematics(
                problem.nodes[connectivity], np.zeros((8, 3))
            ).weights.sum()
        )
        nodal_weight = volume / 8.0
        values = {
            name: np.mean(
                np.stack([getattr(point, name) for point in element_state]), axis=0
            )
            for name in accumulators
        }
        for node in connectivity:
            node_index = int(node)
            weights[node_index] += nodal_weight
            for name, accumulator in accumulators.items():
                accumulator[node_index] += nodal_weight * values[name]
    if np.any(weights <= 0.0):
        raise ValueError("Every operator node must be adjacent to an element")
    for accumulator in accumulators.values():
        accumulator /= weights.reshape((-1,) + (1,) * (accumulator.ndim - 1))
    for node in range(node_count):
        accumulators["orientation"][node] = project_rotation(
            accumulators["orientation"][node]
        )
        determinant = float(np.linalg.det(accumulators["fp"][node]))
        if determinant > 0.0:
            accumulators["fp"][node] /= determinant ** (1.0 / 3.0)
    return (
        accumulators["orientation"],
        accumulators["fp"],
        accumulators["slip"],
        accumulators["ssd"],
        accumulators["crss"],
    )


def _material_prompt(problem: CPFEProblem) -> np.ndarray:
    parameters = problem.parameters
    return np.array(
        [
            parameters.c11 / 1.0e5,
            parameters.c12 / 1.0e5,
            parameters.c44 / 1.0e5,
            parameters.gamma0 / 1.0e6,
            parameters.inner_exponent,
            parameters.outer_exponent,
            parameters.activation_energy_octahedral / 1.0e-19,
            parameters.boltzmann / 1.0e-23,
            parameters.rho0,
            parameters.tauc0 / 1.0e2,
            parameters.burgers / 1.0e-4,
            parameters.k1,
            parameters.k2 / 10.0,
            parameters.geometric_factor,
        ],
        dtype=np.float64,
    )


def build_node_features(
    problem: CPFEProblem,
    state_n: GlobalMaterialState,
    load_step: float,
) -> np.ndarray:
    """Volume-project committed integration-point history into node prompts."""

    if state_n.element_count != problem.elements.shape[0]:
        raise ValueError("Prompt state does not match the problem mesh")
    layout = FeatureLayout.default()
    features = np.zeros((problem.nodes.shape[0], layout.width), dtype=np.float64)
    minimum = problem.nodes.min(axis=0)
    extent = problem.nodes.max(axis=0) - minimum
    safe_extent = np.where(extent > 0.0, extent, 1.0)
    features[:, layout.coordinates] = (problem.nodes - minimum) / safe_extent
    mask = np.zeros(problem.ndof)
    mask[problem.prescribed_dofs] = 1.0
    characteristic_length = max(float(np.max(extent)), 1.0)
    features[:, layout.prescribed_mask] = mask.reshape((-1, 3))
    features[:, layout.prescribed_values] = (
        problem.prescribed_increment.reshape((-1, 3)) / characteristic_length
    )
    orientation, fp, slip, ssd, crss = _project_state_to_nodes(problem, state_n)
    features[:, layout.orientation] = orientation.reshape((-1, 9))
    features[:, layout.fp] = fp.reshape((-1, 9))
    features[:, layout.slip] = slip
    features[:, layout.ssd] = ssd / max(problem.parameters.rho0, 1.0e-12)
    features[:, layout.crss] = crss / max(problem.parameters.tauc0, 1.0e-12)
    features[:, layout.temperature] = problem.temperature / 1.0e3
    features[:, layout.dt] = np.log10(
        max(problem.dt, np.finfo(np.float64).tiny) / 1.0e-3
    )
    features[:, layout.load_fraction] = float(load_step)
    features[:, layout.material] = _material_prompt(problem)
    return features


def enforce_displacement_bc(raw_du, problem: CPFEProblem):
    """Project a nodal displacement increment onto exact essential BCs."""

    original_shape = raw_du.shape
    if int(np.prod(original_shape)) != problem.ndof:
        raise ValueError("Operator displacement has the wrong number of DOFs")
    if isinstance(raw_du, torch.Tensor):
        flat = raw_du.reshape(-1)
        mask = torch.zeros(
            problem.ndof, dtype=torch.bool, device=raw_du.device
        )
        indices = torch.tensor(
            problem.prescribed_dofs.tolist(),
            dtype=torch.int64,
            device=raw_du.device,
        )
        mask[indices] = True
        prescribed = torch.tensor(
            problem.prescribed_increment.tolist(),
            dtype=raw_du.dtype,
            device=raw_du.device,
        )
        return torch.where(mask, prescribed, flat).reshape(original_shape)
    array = np.asarray(raw_du)
    flat = array.reshape(-1).copy()
    flat[problem.prescribed_dofs] = problem.prescribed_increment[
        problem.prescribed_dofs
    ]
    return flat.reshape(original_shape)
