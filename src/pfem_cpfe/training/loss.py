"""Differentiable discrete-equilibrium loss without solution labels."""

from __future__ import annotations

import numpy as np
import torch

from pfem_cpfe.elements.c3d8 import c3d8_kinematics
from pfem_cpfe.materials.case2_torch import TorchMaterialState, case2_step_torch
from pfem_cpfe.operator.features import enforce_displacement_bc


def _element_constants(problem, dtype, device):
    gradients = []
    weights = []
    for connectivity in problem.elements:
        kinematics = c3d8_kinematics(
            problem.nodes[connectivity], np.zeros((8, 3))
        )
        gradients.append(kinematics.gradients)
        weights.append(kinematics.weights)
    return (
        torch.tensor(np.stack(gradients), dtype=dtype, device=device),
        torch.tensor(np.stack(weights), dtype=dtype, device=device),
        torch.tensor(problem.elements.tolist(), dtype=torch.int64, device=device),
    )


def _torch_state(state_n, dtype, device) -> TorchMaterialState:
    def tensor(value):
        return torch.tensor(value, dtype=dtype, device=device).flatten(0, 1)

    return TorchMaterialState(
        fp=tensor(state_n.fp),
        orientation=tensor(state_n.orientation),
        stress=tensor(state_n.stress),
        ssd=tensor(state_n.ssd),
        crss=tensor(state_n.crss),
        slip=tensor(state_n.slip),
        slip_rate=tensor(state_n.slip_rate),
    )


def _deformation_gradients(displacement, connectivity, gradients):
    element_displacement = displacement[connectivity]
    displacement_gradient = torch.einsum(
        "eai,egaj->egij", element_displacement, gradients
    )
    identity = torch.eye(
        3, dtype=displacement.dtype, device=displacement.device
    )
    return identity + displacement_gradient


def _stress_tensors(stress):
    row0 = torch.stack((stress[..., 0], stress[..., 3], stress[..., 4]), dim=-1)
    row1 = torch.stack((stress[..., 3], stress[..., 1], stress[..., 5]), dim=-1)
    row2 = torch.stack((stress[..., 4], stress[..., 5], stress[..., 2]), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _internal_force(stress, gradients, weights, connectivity, node_count):
    stress_tensor = _stress_tensors(stress)
    point_forces = torch.einsum(
        "egij,egaj,eg->egai", stress_tensor, gradients, weights
    )
    element_forces = point_forces.sum(dim=1)
    internal = torch.zeros(
        (node_count, 3), dtype=stress.dtype, device=stress.device
    )
    for local_node in range(8):
        internal = internal.index_add(
            0, connectivity[:, local_node], element_forces[:, local_node]
        )
    return internal.reshape(-1)


def relative_equilibrium_loss(model, batch):
    """Return unlabeled free-DOF equilibrium loss and diagnostic metrics."""

    problem = batch["problem"]
    xyz = batch["xyz"]
    features = batch["features"]
    mask = batch["mask"]
    if xyz.shape[0] != 1:
        raise ValueError("The current explicit FE loss accepts one mesh per call")
    raw_increment = model(xyz, features, mask)[0]
    raw_increment = raw_increment * float(batch.get("output_scale", 1.0))
    increment = enforce_displacement_bc(raw_increment, problem)
    previous_displacement = batch["u_n"]
    current_displacement = previous_displacement + increment
    gradients, weights, connectivity = _element_constants(
        problem, xyz.dtype, xyz.device
    )
    F_n = _deformation_gradients(
        previous_displacement, connectivity, gradients
    )
    F_np1 = _deformation_gradients(
        current_displacement, connectivity, gradients
    )
    torch_state = _torch_state(batch["state_n"], xyz.dtype, xyz.device)
    material = case2_step_torch(
        F_n.flatten(0, 1),
        F_np1.flatten(0, 1),
        problem.dt,
        problem.temperature,
        problem.parameters,
        torch_state,
    )
    stress = material.stress.reshape(problem.elements.shape[0], 8, 6)
    internal = _internal_force(
        stress,
        gradients,
        weights,
        connectivity,
        problem.nodes.shape[0],
    )
    external = torch.tensor(
        problem.external_force.tolist(), dtype=xyz.dtype, device=xyz.device
    )
    residual = internal - external
    free = torch.tensor(
        problem.free_dofs.tolist(), dtype=torch.int64, device=xyz.device
    )
    free_residual = residual[free]
    prescribed_scale = float(
        np.max(np.abs(problem.prescribed_increment), initial=0.0)
    )
    force_scale = max(
        float(np.linalg.norm(problem.external_force)),
        problem.parameters.c11 * prescribed_scale,
        1.0,
    )
    relative_residual = torch.sum(free_residual.square()) / (
        force_scale * force_scale + 1.0e-12
    )

    determinant = torch.linalg.det(F_np1)
    determinant_penalty = torch.relu(1.0e-8 - determinant).square().mean()
    negative_ssd = torch.relu(-material.state.ssd).square().mean()
    orientation = material.state.orientation
    identity = torch.eye(3, dtype=xyz.dtype, device=xyz.device)
    orthogonality = (
        orientation.transpose(1, 2) @ orientation - identity
    ).square().mean()
    orientation_determinant = (
        torch.linalg.det(orientation) - 1.0
    ).square().mean()
    state_penalty = (
        determinant_penalty + negative_ssd + orthogonality + orientation_determinant
    )
    invalid_fraction = (~material.converged).to(xyz.dtype).mean()
    loss = relative_residual + state_penalty + 100.0 * invalid_fraction
    metrics = {
        "relative_residual": float(relative_residual.detach().cpu()),
        "state_penalty": float(state_penalty.detach().cpu()),
        "invalid_fraction": float(invalid_fraction.detach().cpu()),
    }
    return loss, metrics
