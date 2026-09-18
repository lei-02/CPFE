"""NumPy material-point integration for Oxford UMAT CASE2."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pfem_cpfe.contracts import (
    Case2Parameters,
    MaterialState,
    TrialResult,
)
from pfem_cpfe.crystal.fcc import fcc_slip_systems
from pfem_cpfe.materials.case2_laws import (
    case2_slip_resistance,
    kocks_mecking_increment,
    total_and_forest_density,
)


FloatArray = NDArray[np.float64]
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_FCC_SCREW_SYSTEMS = np.array([0, 1, 2, 3, 5, 7], dtype=np.int64)


def _strain_to_voigt(tensor: FloatArray) -> FloatArray:
    return np.array(
        [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            2.0 * tensor[0, 1],
            2.0 * tensor[0, 2],
            2.0 * tensor[1, 2],
        ]
    )


def _stress_to_tensor(vector: FloatArray) -> FloatArray:
    return np.array(
        [
            [vector[0], vector[3], vector[4]],
            [vector[3], vector[1], vector[5]],
            [vector[4], vector[5], vector[2]],
        ]
    )


def _inverse6(matrix: FloatArray, small_number: float) -> FloatArray:
    """Match Oxford's pivoted Gauss-Jordan inverse for the local Jacobian."""

    augmented = np.concatenate((matrix.copy(), np.eye(6)), axis=1)
    for column in range(6):
        pivot = column + int(np.argmax(np.abs(augmented[column:, column])))
        if abs(augmented[pivot, column]) <= small_number:
            raise RuntimeError("singular local stress Jacobian")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        pivot_value = augmented[column, column]
        augmented[column] /= pivot_value
        for row in range(6):
            if row != column:
                factor = augmented[row, column]
                augmented[row] -= factor * augmented[column]
    return augmented[:, 6:]


def _cubic_stiffness(parameters: Case2Parameters, rotation: FloatArray) -> FloatArray:
    crystal = np.zeros((3, 3, 3, 3))
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
    sample = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    value = 0.0
                    for ii in range(3):
                        for jj in range(3):
                            for kk in range(3):
                                for ll in range(3):
                                    value += (
                                        rotation[i, ii]
                                        * rotation[j, jj]
                                        * rotation[k, kk]
                                        * rotation[l, ll]
                                        * crystal[ii, jj, kk, ll]
                                    )
                    sample[i, j, k, l] = value
    stiffness = np.empty((6, 6))
    for row, (i, j) in enumerate(_VOIGT_PAIRS):
        for column, (k, l) in enumerate(_VOIGT_PAIRS):
            stiffness[row, column] = sample[i, j, k, l]
    return stiffness


def _density_operators() -> tuple[FloatArray, FloatArray]:
    directions, normals = fcc_slip_systems()
    edge_lines = np.cross(directions, normals)
    screw_lines = directions[_FCC_SCREW_SYSTEMS]
    line_directions = np.concatenate((edge_lines, screw_lines), axis=0)
    forest_projection = np.abs(normals @ line_directions.T)
    slip_to_screw = np.zeros((_FCC_SCREW_SYSTEMS.size, directions.shape[0]))
    for screw_row, system in enumerate(_FCC_SCREW_SYSTEMS):
        equivalent = np.abs(directions @ directions[system]) > 0.99
        slip_to_screw[screw_row, equivalent] = 0.5
    return forest_projection, slip_to_screw


def _schmid_vectors(tensors: FloatArray) -> FloatArray:
    vectors = np.empty((tensors.shape[0], 6))
    vectors[:, 0] = tensors[:, 0, 0]
    vectors[:, 1] = tensors[:, 1, 1]
    vectors[:, 2] = tensors[:, 2, 2]
    vectors[:, 3] = tensors[:, 0, 1] + tensors[:, 1, 0]
    vectors[:, 4] = tensors[:, 0, 2] + tensors[:, 2, 0]
    vectors[:, 5] = tensors[:, 1, 2] + tensors[:, 2, 1]
    return vectors


def _source_schmid_tensors(rotation: FloatArray) -> FloatArray:
    """Build Schmid tensors without the rotation projection absent in Oxford."""

    crystal_directions, crystal_normals = fcc_slip_systems()
    tensors = np.empty((crystal_directions.shape[0], 3, 3))
    for system in range(crystal_directions.shape[0]):
        direction = np.empty(3)
        normal = np.empty(3)
        for i in range(3):
            direction[i] = sum(
                rotation[i, j] * crystal_directions[system, j] for j in range(3)
            )
            normal[i] = sum(
                rotation[i, j] * crystal_normals[system, j] for j in range(3)
            )
        for i in range(3):
            for j in range(3):
                tensors[system, i, j] = direction[i] * normal[j]
    return tensors


def _slip_rate_and_derivative(
    resolved_shear: FloatArray,
    resistance: FloatArray,
    temperature: float,
    parameters: Case2Parameters,
) -> tuple[FloatArray, FloatArray]:
    ratio = np.abs(resolved_shear) / resistance
    signs = np.sign(resolved_shear)
    rates = np.zeros_like(ratio)
    derivatives = np.zeros_like(ratio)
    saturated = ratio >= 1.0
    rates[saturated] = signs[saturated] * parameters.gamma0
    activated = (ratio > 0.0) & (ratio < 1.0)
    active_ratio = ratio[activated]
    barrier = (1.0 - active_ratio**parameters.inner_exponent) ** parameters.outer_exponent
    rates[activated] = signs[activated] * parameters.gamma0 * np.exp(
        -parameters.activation_energy_octahedral
        / (parameters.boltzmann * temperature)
        * barrier
    )
    derivatives[activated] = (
        np.abs(rates[activated])
        * parameters.activation_energy_octahedral
        / (parameters.boltzmann * temperature)
        * parameters.outer_exponent
        * (1.0 - active_ratio**parameters.inner_exponent)
        ** (parameters.outer_exponent - 1.0)
        * parameters.inner_exponent
        / resistance[activated]
        * active_ratio ** (parameters.inner_exponent - 1.0)
    )
    negligible = np.abs(rates) <= parameters.small_number
    rates[negligible] = 0.0
    derivatives[negligible] = 0.0
    return rates, derivatives


def _solve_stress(
    trial_stress: FloatArray,
    initial_stress: FloatArray,
    stiffness: FloatArray,
    schmid: FloatArray,
    schmid_vectors: FloatArray,
    resistance: FloatArray,
    dt: float,
    temperature: float,
    parameters: Case2Parameters,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, int]:
    stress = initial_stress.copy()
    resolved = schmid_vectors @ stress
    tangent = stiffness.copy()
    plastic_velocity = np.zeros((3, 3))
    plastic_rate = np.zeros(6)
    for iteration in range(1, 201):
        slip_rate, derivative = _slip_rate_and_derivative(
            resolved, resistance, temperature, parameters
        )
        plastic_velocity = np.einsum("s,sij->ij", slip_rate, schmid)
        plastic_stretch = 0.5 * (plastic_velocity + plastic_velocity.T)
        plastic_rate = _strain_to_voigt(plastic_stretch)
        plastic_increment = plastic_rate * dt
        plastic_jacobian = dt * np.einsum(
            "s,si,sj->ij", derivative, schmid_vectors, schmid_vectors
        )
        residual_jacobian = np.eye(6) + stiffness @ plastic_jacobian
        residual = trial_stress - stress - stiffness @ plastic_increment
        inverse_residual_jacobian = _inverse6(
            residual_jacobian, parameters.small_number
        )
        tangent = inverse_residual_jacobian @ stiffness
        stress += inverse_residual_jacobian @ residual
        resolved = schmid_vectors @ stress
        if np.linalg.norm(residual) < 1.0e-8:
            return (
                stress,
                slip_rate,
                plastic_velocity,
                plastic_rate,
                tangent,
                iteration,
            )
    raise RuntimeError("local stress iteration exhausted")


def _failed_trial(state: MaterialState, reason: str, iterations: int = 0) -> TrialResult:
    return TrialResult(
        stress=state.stress,
        tangent=np.zeros((6, 6)),
        state=state,
        converged=False,
        iterations=iterations,
        reason=reason,
    )


def case2_step_numpy(
    F_n: ArrayLike,
    F_np1: ArrayLike,
    dt: float,
    temperature: float,
    parameters: Case2Parameters,
    state_n: MaterialState,
) -> TrialResult:
    """Advance one uncommitted Oxford CASE2 material-point increment."""

    previous_deformation = np.asarray(F_n, dtype=np.float64)
    deformation = np.asarray(F_np1, dtype=np.float64)
    if previous_deformation.shape != (3, 3) or deformation.shape != (3, 3):
        return _failed_trial(state_n, "deformation gradients must be 3-by-3")
    if dt <= 0.0:
        return _failed_trial(state_n, "time increment must be positive")
    if temperature <= 0.0:
        return _failed_trial(state_n, "temperature must be positive")
    if not np.all(np.isfinite(deformation)):
        return _failed_trial(state_n, "deformation gradient is nonfinite")
    determinant = float(np.linalg.det(deformation))
    if determinant <= parameters.small_number:
        return _failed_trial(state_n, "deformation gradient has nonpositive determinant")

    try:
        inverse_deformation = np.linalg.inv(deformation)
        velocity_gradient = (deformation - previous_deformation) / dt @ inverse_deformation
        strain_increment_tensor = 0.5 * (velocity_gradient + velocity_gradient.T) * dt
        strain_increment = _strain_to_voigt(strain_increment_tensor)
        stiffness = _cubic_stiffness(parameters, state_n.orientation)
        trial_stress = state_n.stress + stiffness @ strain_increment
        schmid = _source_schmid_tensors(state_n.orientation)
        schmid_vectors = _schmid_vectors(schmid)
        forest_projection, slip_to_screw = _density_operators()
        gnd = np.zeros(parameters.nslip + slip_to_screw.shape[0])
        base_forest = np.zeros(parameters.nslip)
        _, _, old_forest_density = total_and_forest_density(
            state_n.ssd,
            gnd,
            base_forest,
            forest_projection,
            slip_to_screw,
        )
        resistance = case2_slip_resistance(old_forest_density, parameters)

        total_iterations = 0
        slip_rate = np.zeros(parameters.nslip)
        plastic_velocity = np.zeros((3, 3))
        tangent = stiffness
        new_ssd = state_n.ssd.copy()
        stress = state_n.stress.copy()
        for _ in range(200):
            (
                stress,
                slip_rate,
                plastic_velocity,
                _,
                tangent,
                inner_iterations,
            ) = _solve_stress(
                trial_stress,
                stress,
                stiffness,
                schmid,
                schmid_vectors,
                resistance,
                dt,
                temperature,
                parameters,
            )
            total_iterations += inner_iterations
            new_ssd = state_n.ssd + kocks_mecking_increment(
                state_n.ssd,
                old_forest_density,
                slip_rate,
                dt,
                parameters,
            )
            if np.any(new_ssd < 0.0) or not np.all(np.isfinite(new_ssd)):
                return _failed_trial(state_n, "SSD update is invalid", total_iterations)
            _, _, new_forest_density = total_and_forest_density(
                new_ssd,
                gnd,
                base_forest,
                forest_projection,
                slip_to_screw,
            )
            new_resistance = case2_slip_resistance(new_forest_density, parameters)
            resistance_change = float(np.max(np.abs(new_resistance - resistance)))
            resistance = new_resistance
            if resistance_change < 1.0e-4:
                break
        else:
            return _failed_trial(state_n, "state iteration exhausted", total_iterations)

        plastic_increment_intermediate = (
            inverse_deformation @ plastic_velocity @ deformation
        ) * dt
        fp = state_n.fp @ (np.eye(3) + plastic_increment_intermediate)
        fp_determinant = float(np.linalg.det(fp))
        if fp_determinant <= parameters.small_number:
            return _failed_trial(state_n, "plastic deformation gradient is singular", total_iterations)
        fp /= fp_determinant ** (1.0 / 3.0)

        elastic_velocity = velocity_gradient - plastic_velocity
        elastic_spin = 0.5 * (elastic_velocity - elastic_velocity.T)
        orientation = (np.eye(3) + elastic_spin * dt) @ state_n.orientation
        new_state = state_n.copied(
            fp=fp,
            orientation=orientation,
            stress=stress,
            ssd=new_ssd,
            crss=resistance,
            slip=state_n.slip + slip_rate * dt,
            slip_rate=slip_rate,
        )
        if not all(
            np.all(np.isfinite(value))
            for value in (
                stress,
                tangent,
                fp,
                orientation,
                new_ssd,
                resistance,
            )
        ):
            return _failed_trial(state_n, "material-point result is nonfinite", total_iterations)
        return TrialResult(
            stress=stress,
            tangent=tangent,
            state=new_state,
            converged=True,
            iterations=total_iterations,
            reason="",
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return _failed_trial(state_n, str(error))
