"""Source-locked local laws used by Oxford UMAT CASE2."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pfem_cpfe.contracts import Case2Parameters


FloatArray = NDArray[np.float64]


def double_exponential_slip(
    resolved_shear: ArrayLike,
    backstress: ArrayLike,
    resistance: ArrayLike,
    temperature: float,
    parameters: Case2Parameters,
) -> FloatArray:
    """Evaluate Oxford ``doubleexpslip`` for octahedral CASE2 systems."""

    tau = np.asarray(resolved_shear, dtype=np.float64)
    back = np.asarray(backstress, dtype=np.float64)
    crss = np.asarray(resistance, dtype=np.float64)
    if tau.shape != back.shape or tau.shape != crss.shape:
        raise ValueError("Resolved shear, backstress, and resistance must have matching shapes")
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive")
    if np.any(crss <= 0.0):
        raise ValueError("Slip resistance must be positive")

    effective = tau - back
    ratio = np.abs(effective) / crss
    sign = np.sign(effective)
    rate = np.zeros_like(ratio)

    saturated = ratio >= 1.0
    rate[saturated] = sign[saturated] * parameters.gamma0

    activated = (ratio > 0.0) & (ratio < 1.0)
    activated_ratio = ratio[activated]
    barrier = (
        1.0 - activated_ratio**parameters.inner_exponent
    ) ** parameters.outer_exponent
    rate[activated] = sign[activated] * parameters.gamma0 * np.exp(
        -parameters.activation_energy_octahedral
        / (parameters.boltzmann * temperature)
        * barrier
    )
    rate[np.abs(rate) <= parameters.small_number] = 0.0
    return rate


def kocks_mecking_increment(
    ssd: ArrayLike,
    forest_density: ArrayLike,
    slip_rate: ArrayLike,
    dt: float,
    parameters: Case2Parameters,
) -> FloatArray:
    """Return the Oxford hardening-model-3 SSD increment."""

    density = np.asarray(ssd, dtype=np.float64)
    forest = np.asarray(forest_density, dtype=np.float64)
    rate = np.asarray(slip_rate, dtype=np.float64)
    if density.shape != forest.shape or density.shape != rate.shape:
        raise ValueError("SSD, forest density, and slip rate must have matching shapes")
    if parameters.burgers <= 0.0:
        raise ValueError("Burgers-vector magnitude must be positive")
    if dt < 0.0:
        raise ValueError("Time increment cannot be negative")
    if np.any(density < 0.0) or np.any(forest < 0.0):
        raise ValueError("Dislocation densities cannot be negative")
    return (
        parameters.k1 / parameters.burgers * np.sqrt(forest)
        - parameters.k2 * density
    ) * np.abs(rate) * dt


def total_and_forest_density(
    ssd: ArrayLike,
    gnd: ArrayLike,
    forest: ArrayLike,
    forest_projection: ArrayLike,
    slip_to_screw: ArrayLike,
) -> tuple[FloatArray, float, FloatArray]:
    """Reproduce Oxford ``totalandforest`` matrix operations."""

    ssd_array = np.asarray(ssd, dtype=np.float64)
    gnd_array = np.asarray(gnd, dtype=np.float64)
    forest_array = np.asarray(forest, dtype=np.float64)
    projection = np.asarray(forest_projection, dtype=np.float64)
    screw_map = np.asarray(slip_to_screw, dtype=np.float64)
    nslip = ssd_array.size
    nscrew = screw_map.shape[0]
    if ssd_array.shape != (nslip,) or forest_array.shape != (nslip,):
        raise ValueError("SSD and base forest density must be one-dimensional")
    if screw_map.shape != (nscrew, nslip):
        raise ValueError("Slip-to-screw map has an invalid shape")
    if gnd_array.shape != (nslip + nscrew,):
        raise ValueError("GND array must contain edge and screw components")
    if projection.shape != (nslip, nslip + nscrew):
        raise ValueError("Forest projection has an invalid shape")
    if np.any(ssd_array < 0.0) or np.any(forest_array < 0.0):
        raise ValueError("Dislocation densities cannot be negative")

    screw_ssd = screw_map @ ssd_array
    forest_density = (
        forest_array
        + projection @ np.abs(gnd_array)
        + projection[:, :nslip] @ ssd_array
        + projection[:, nslip:] @ screw_ssd
    )
    total_density = (
        ssd_array
        + np.abs(gnd_array[:nslip])
        + screw_map.T @ np.abs(gnd_array[nslip:])
    )
    scalar_total = float(np.linalg.norm(gnd_array) + np.sum(ssd_array))
    return total_density, scalar_total, forest_density


def case2_slip_resistance(
    forest_density: ArrayLike,
    parameters: Case2Parameters,
) -> FloatArray:
    """Return CASE2 CRSS without irradiation or substructure terms."""

    forest = np.asarray(forest_density, dtype=np.float64)
    if np.any(forest < 0.0):
        raise ValueError("Forest density cannot be negative")
    return (
        parameters.tauc0
        + parameters.c44
        * parameters.burgers
        * parameters.geometric_factor
        * np.sqrt(forest)
    )
