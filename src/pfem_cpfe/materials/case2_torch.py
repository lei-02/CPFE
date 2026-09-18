"""Differentiable batched Oxford CASE2 material-point integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pfem_cpfe.contracts import Case2Parameters


@dataclass(frozen=True)
class TorchMaterialState:
    """Batched constitutive state stored entirely as Torch tensors."""

    fp: torch.Tensor
    orientation: torch.Tensor
    stress: torch.Tensor
    ssd: torch.Tensor
    crss: torch.Tensor
    slip: torch.Tensor
    slip_rate: torch.Tensor


@dataclass(frozen=True)
class TorchTrialResult:
    """Uncommitted batched response with per-point status."""

    stress: torch.Tensor
    tangent: torch.Tensor
    state: TorchMaterialState
    converged: torch.Tensor
    iterations: torch.Tensor
    reason_codes: torch.Tensor


_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_FCC_DIRECTIONS = (
    (1.0, -1.0, 0.0),
    (0.0, 1.0, -1.0),
    (1.0, 0.0, -1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, -1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, -1.0),
    (1.0, -1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
)
_FCC_NORMALS = (
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0),
)
_FCC_SCREW_SYSTEMS = (0, 1, 2, 3, 5, 7)


def _batch_scalar(value, reference: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim == 0:
        result = result.expand(reference.shape[0])
    if result.shape != (reference.shape[0],):
        raise ValueError("Expected one scalar value per material point")
    return result


def _systems(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    directions = torch.tensor(
        _FCC_DIRECTIONS, dtype=reference.dtype, device=reference.device
    )
    normals = torch.tensor(_FCC_NORMALS, dtype=reference.dtype, device=reference.device)
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    normals = normals / torch.linalg.vector_norm(normals, dim=1, keepdim=True)
    return directions, normals


def _strain_to_voigt(tensor: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            tensor[:, 0, 0],
            tensor[:, 1, 1],
            tensor[:, 2, 2],
            2.0 * tensor[:, 0, 1],
            2.0 * tensor[:, 0, 2],
            2.0 * tensor[:, 1, 2],
        ),
        dim=1,
    )


def _cubic_stiffness(
    parameters: Case2Parameters, rotation: torch.Tensor
) -> torch.Tensor:
    crystal = torch.zeros(
        (3, 3, 3, 3), dtype=rotation.dtype, device=rotation.device
    )
    for i in range(3):
        crystal[i, i, i, i] = parameters.c11
        for j in range(3):
            if i != j:
                crystal[i, i, j, j] = parameters.c12
    for i in range(3):
        for j in range(i + 1, 3):
            crystal[i, j, i, j] = parameters.c44
            crystal[i, j, j, i] = parameters.c44
            crystal[j, i, i, j] = parameters.c44
            crystal[j, i, j, i] = parameters.c44
    sample = torch.einsum(
        "bai,bcj,bdk,bel,ijkl->bacde",
        rotation,
        rotation,
        rotation,
        rotation,
        crystal,
    )
    rows = []
    for i, j in _VOIGT_PAIRS:
        rows.append(
            torch.stack(
                [sample[:, i, j, k, l] for k, l in _VOIGT_PAIRS], dim=1
            )
        )
    return torch.stack(rows, dim=1)


def _schmid_tensors(rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    directions, normals = _systems(rotation)
    sample_directions = torch.einsum("bij,sj->bsi", rotation, directions)
    sample_normals = torch.einsum("bij,sj->bsi", rotation, normals)
    tensors = sample_directions[:, :, :, None] * sample_normals[:, :, None, :]
    vectors = torch.stack(
        (
            tensors[:, :, 0, 0],
            tensors[:, :, 1, 1],
            tensors[:, :, 2, 2],
            tensors[:, :, 0, 1] + tensors[:, :, 1, 0],
            tensors[:, :, 0, 2] + tensors[:, :, 2, 0],
            tensors[:, :, 1, 2] + tensors[:, :, 2, 1],
        ),
        dim=2,
    )
    return tensors, vectors


def _density_operators(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    directions, normals = _systems(reference)
    edge_lines = torch.linalg.cross(directions, normals, dim=1)
    index = torch.tensor(_FCC_SCREW_SYSTEMS, device=reference.device)
    screw_lines = directions[index]
    lines = torch.cat((edge_lines, screw_lines), dim=0)
    forest_projection = torch.abs(normals @ lines.T)
    equivalent = torch.abs(screw_lines @ directions.T) > 0.99
    slip_to_screw = equivalent.to(reference.dtype) * 0.5
    return forest_projection, slip_to_screw


def _forest_density(
    ssd: torch.Tensor,
    forest_projection: torch.Tensor,
    slip_to_screw: torch.Tensor,
) -> torch.Tensor:
    nslip = ssd.shape[1]
    screw_ssd = torch.einsum("ks,bs->bk", slip_to_screw, ssd)
    return torch.einsum("sj,bj->bs", forest_projection[:, :nslip], ssd) + torch.einsum(
        "sk,bk->bs", forest_projection[:, nslip:], screw_ssd
    )


def _slip_resistance(
    forest: torch.Tensor, parameters: Case2Parameters
) -> torch.Tensor:
    return (
        parameters.tauc0
        + parameters.c44
        * parameters.burgers
        * parameters.geometric_factor
        * torch.sqrt(forest)
    )


def _slip_rate_and_derivative(
    resolved: torch.Tensor,
    resistance: torch.Tensor,
    temperature: torch.Tensor,
    parameters: Case2Parameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = torch.abs(resolved) / resistance
    safe_ratio = torch.clamp(ratio, min=torch.finfo(ratio.dtype).tiny)
    remaining = torch.clamp(
        1.0 - safe_ratio**parameters.inner_exponent, min=0.0
    )
    barrier = remaining**parameters.outer_exponent
    activated_rate = torch.sign(resolved) * parameters.gamma0 * torch.exp(
        -parameters.activation_energy_octahedral
        / parameters.boltzmann
        / temperature[:, None]
        * barrier
    )
    rate = torch.where(
        ratio >= 1.0,
        torch.sign(resolved) * parameters.gamma0,
        activated_rate,
    )
    derivative = (
        torch.abs(rate)
        * parameters.activation_energy_octahedral
        / parameters.boltzmann
        / temperature[:, None]
        * parameters.outer_exponent
        * remaining ** (parameters.outer_exponent - 1.0)
        * parameters.inner_exponent
        / resistance
        * safe_ratio ** (parameters.inner_exponent - 1.0)
    )
    active = (ratio > 0.0) & (ratio < 1.0) & (
        torch.abs(rate) > parameters.small_number
    )
    rate = torch.where(active | (ratio >= 1.0), rate, torch.zeros_like(rate))
    derivative = torch.where(active, derivative, torch.zeros_like(derivative))
    return rate, derivative


def _solve_stress(
    trial_stress: torch.Tensor,
    initial_stress: torch.Tensor,
    stiffness: torch.Tensor,
    schmid: torch.Tensor,
    schmid_vectors: torch.Tensor,
    resistance: torch.Tensor,
    dt: torch.Tensor,
    temperature: torch.Tensor,
    parameters: Case2Parameters,
    active: torch.Tensor,
):
    batch = trial_stress.shape[0]
    stress = initial_stress
    tangent = stiffness
    slip_rate = torch.zeros_like(resistance)
    plastic_velocity = torch.zeros_like(schmid[:, 0])
    plastic_rate = torch.zeros_like(trial_stress)
    done = ~active
    iterations = torch.zeros(batch, dtype=torch.int64, device=stress.device)
    identity = torch.eye(6, dtype=stress.dtype, device=stress.device).expand(batch, -1, -1)

    for _ in range(200):
        update = ~done
        resolved = torch.einsum("bsi,bi->bs", schmid_vectors, stress)
        candidate_rate, derivative = _slip_rate_and_derivative(
            resolved, resistance, temperature, parameters
        )
        candidate_velocity = torch.einsum("bs,bsij->bij", candidate_rate, schmid)
        plastic_stretch = 0.5 * (candidate_velocity + candidate_velocity.transpose(1, 2))
        candidate_plastic_rate = _strain_to_voigt(plastic_stretch)
        plastic_increment = candidate_plastic_rate * dt[:, None]
        plastic_jacobian = dt[:, None, None] * torch.einsum(
            "bs,bsi,bsj->bij", derivative, schmid_vectors, schmid_vectors
        )
        residual_jacobian = identity + stiffness @ plastic_jacobian
        inverse_jacobian = torch.linalg.inv(residual_jacobian)
        candidate_tangent = inverse_jacobian @ stiffness
        residual = trial_stress - stress - torch.einsum(
            "bij,bj->bi", stiffness, plastic_increment
        )
        candidate_stress = stress + torch.einsum(
            "bij,bj->bi", inverse_jacobian, residual
        )
        stress = torch.where(update[:, None], candidate_stress, stress)
        tangent = torch.where(update[:, None, None], candidate_tangent, tangent)
        slip_rate = torch.where(update[:, None], candidate_rate, slip_rate)
        plastic_velocity = torch.where(
            update[:, None, None], candidate_velocity, plastic_velocity
        )
        plastic_rate = torch.where(
            update[:, None], candidate_plastic_rate, plastic_rate
        )
        iterations = iterations + update.to(torch.int64)
        done = done | (update & (torch.linalg.vector_norm(residual, dim=1) < 1.0e-8))
        if bool(torch.all(done).detach().cpu()):
            break
    return (
        stress,
        slip_rate,
        plastic_velocity,
        plastic_rate,
        tangent,
        done,
        iterations,
    )


def _select(mask: torch.Tensor, candidate: torch.Tensor, fallback: torch.Tensor):
    while mask.ndim < candidate.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, candidate, fallback)


def case2_step_torch(
    F_n: torch.Tensor,
    F_np1: torch.Tensor,
    dt,
    temperature,
    parameters: Case2Parameters,
    state_n: TorchMaterialState,
) -> TorchTrialResult:
    """Advance a batch of uncommitted CASE2 material-point increments."""

    if F_n.ndim != 3 or F_n.shape[1:] != (3, 3) or F_np1.shape != F_n.shape:
        raise ValueError("Deformation gradients must have shape (batch, 3, 3)")
    batch = F_n.shape[0]
    dt_values = _batch_scalar(dt, F_n)
    temperature_values = _batch_scalar(temperature, F_n)
    identity3 = torch.eye(3, dtype=F_n.dtype, device=F_n.device).expand(batch, -1, -1)
    determinant = torch.linalg.det(F_np1)
    input_valid = (
        torch.isfinite(F_n).all(dim=(1, 2))
        & torch.isfinite(F_np1).all(dim=(1, 2))
        & (determinant > parameters.small_number)
        & (dt_values > 0.0)
        & (temperature_values > 0.0)
    )
    safe_deformation = _select(input_valid, F_np1, identity3)
    safe_previous = _select(input_valid, F_n, identity3)
    safe_dt = torch.where(input_valid, dt_values, torch.ones_like(dt_values))
    safe_temperature = torch.where(
        input_valid, temperature_values, torch.ones_like(temperature_values)
    )

    inverse_deformation = torch.linalg.inv(safe_deformation)
    velocity_gradient = (
        (safe_deformation - safe_previous) / safe_dt[:, None, None]
    ) @ inverse_deformation
    strain_tensor = 0.5 * (velocity_gradient + velocity_gradient.transpose(1, 2))
    strain_increment = _strain_to_voigt(strain_tensor) * safe_dt[:, None]
    stiffness = _cubic_stiffness(parameters, state_n.orientation)
    trial_stress = state_n.stress + torch.einsum(
        "bij,bj->bi", stiffness, strain_increment
    )
    schmid, schmid_vectors = _schmid_tensors(state_n.orientation)
    forest_projection, slip_to_screw = _density_operators(F_n)
    old_forest = _forest_density(state_n.ssd, forest_projection, slip_to_screw)
    resistance = _slip_resistance(old_forest, parameters)

    stress = state_n.stress
    tangent = stiffness
    slip_rate = torch.zeros_like(state_n.slip_rate)
    plastic_velocity = torch.zeros_like(F_n)
    new_ssd = state_n.ssd
    outer_done = ~input_valid
    local_failed = torch.zeros(batch, dtype=torch.bool, device=F_n.device)
    state_failed = torch.zeros_like(local_failed)
    total_iterations = torch.zeros(batch, dtype=torch.int64, device=F_n.device)

    for _ in range(200):
        active = input_valid & ~outer_done & ~local_failed & ~state_failed
        if not bool(torch.any(active).detach().cpu()):
            break
        (
            candidate_stress,
            candidate_rate,
            candidate_velocity,
            _,
            candidate_tangent,
            inner_done,
            inner_iterations,
        ) = _solve_stress(
            trial_stress,
            stress,
            stiffness,
            schmid,
            schmid_vectors,
            resistance,
            safe_dt,
            safe_temperature,
            parameters,
            active,
        )
        total_iterations = total_iterations + inner_iterations
        inner_failed = active & ~inner_done
        local_failed = local_failed | inner_failed
        update = active & inner_done
        candidate_ssd = state_n.ssd + (
            parameters.k1 / parameters.burgers * torch.sqrt(old_forest)
            - parameters.k2 * state_n.ssd
        ) * torch.abs(candidate_rate) * safe_dt[:, None]
        finite_state = torch.isfinite(candidate_ssd).all(dim=1) & torch.all(
            candidate_ssd >= 0.0, dim=1
        )
        state_failed = state_failed | (update & ~finite_state)
        update = update & finite_state
        candidate_forest = _forest_density(
            candidate_ssd, forest_projection, slip_to_screw
        )
        candidate_resistance = _slip_resistance(candidate_forest, parameters)
        resistance_change = torch.amax(
            torch.abs(candidate_resistance - resistance), dim=1
        )
        stress = _select(update, candidate_stress, stress)
        tangent = _select(update, candidate_tangent, tangent)
        slip_rate = _select(update, candidate_rate, slip_rate)
        plastic_velocity = _select(update, candidate_velocity, plastic_velocity)
        new_ssd = _select(update, candidate_ssd, new_ssd)
        resistance = _select(update, candidate_resistance, resistance)
        outer_done = outer_done | (update & (resistance_change < 1.0e-4))

    constitutive_converged = input_valid & outer_done & ~local_failed & ~state_failed
    plastic_increment = (
        inverse_deformation @ plastic_velocity @ safe_deformation
    ) * safe_dt[:, None, None]
    fp_candidate = state_n.fp @ (identity3 + plastic_increment)
    fp_determinant = torch.linalg.det(fp_candidate)
    safe_fp_determinant = torch.where(
        fp_determinant > parameters.small_number,
        fp_determinant,
        torch.ones_like(fp_determinant),
    )
    fp_candidate = fp_candidate / safe_fp_determinant[:, None, None] ** (1.0 / 3.0)
    elastic_velocity = velocity_gradient - plastic_velocity
    spin = 0.5 * (elastic_velocity - elastic_velocity.transpose(1, 2))
    orientation_candidate = (identity3 + spin * safe_dt[:, None, None]) @ state_n.orientation
    slip_candidate = state_n.slip + slip_rate * safe_dt[:, None]
    post_finite = (
        torch.isfinite(stress).all(dim=1)
        & torch.isfinite(tangent).all(dim=(1, 2))
        & torch.isfinite(fp_candidate).all(dim=(1, 2))
        & torch.isfinite(orientation_candidate).all(dim=(1, 2))
        & torch.isfinite(resistance).all(dim=1)
        & (fp_determinant > parameters.small_number)
    )
    converged = constitutive_converged & post_finite

    final_stress = _select(converged, stress, state_n.stress)
    final_tangent = _select(converged, tangent, torch.zeros_like(tangent))
    final_state = TorchMaterialState(
        fp=_select(converged, fp_candidate, state_n.fp),
        orientation=_select(converged, orientation_candidate, state_n.orientation),
        stress=final_stress,
        ssd=_select(converged, new_ssd, state_n.ssd),
        crss=_select(converged, resistance, state_n.crss),
        slip=_select(converged, slip_candidate, state_n.slip),
        slip_rate=_select(converged, slip_rate, state_n.slip_rate),
    )
    reason_codes = torch.zeros(batch, dtype=torch.int64, device=F_n.device)
    reason_codes = torch.where(~input_valid, torch.ones_like(reason_codes), reason_codes)
    reason_codes = torch.where(local_failed, torch.full_like(reason_codes, 2), reason_codes)
    exhausted = input_valid & ~outer_done & ~local_failed & ~state_failed
    reason_codes = torch.where(exhausted, torch.full_like(reason_codes, 3), reason_codes)
    reason_codes = torch.where(
        constitutive_converged & ~post_finite,
        torch.full_like(reason_codes, 4),
        reason_codes,
    )
    reason_codes = torch.where(state_failed, torch.full_like(reason_codes, 4), reason_codes)
    return TorchTrialResult(
        stress=final_stress,
        tangent=final_tangent,
        state=final_state,
        converged=converged,
        iterations=total_iterations,
        reason_codes=reason_codes,
    )
