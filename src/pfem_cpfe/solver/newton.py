"""Safeguarded Newton solution with line search and recursive cutback."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from pfem_cpfe.solver.assembly import (
    CPFEProblem,
    GlobalMaterialState,
    assemble_trial,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NewtonControls:
    residual_rtol: float = 1.0e-9
    residual_atol: float = 1.0e-9
    correction_rtol: float = 1.0e-9
    correction_atol: float = 1.0e-12
    max_iterations: int = 30
    max_cutbacks: int = 4
    max_prescribed_increment: float = np.inf


@dataclass(frozen=True)
class IncrementResult:
    u: FloatArray
    states: GlobalMaterialState
    residual: FloatArray
    converged: bool
    iterations: int
    cutbacks: int
    reason: str
    timing: dict[str, float]


def _failed_result(
    problem: CPFEProblem,
    state_n: GlobalMaterialState,
    reason: str,
    *,
    iterations: int = 0,
    cutbacks: int = 0,
    elapsed: float = 0.0,
    detail: Mapping[str, float] | None = None,
) -> IncrementResult:
    timing = {
        "newton": elapsed,
        "material": 0.0,
        "assembly": 0.0,
        "linear_solve": 0.0,
    }
    if detail is not None:
        timing.update({name: float(value) for name, value in detail.items()})
        timing["newton"] = elapsed
    return IncrementResult(
        u=problem.previous_u.copy(),
        states=state_n.copied(),
        residual=np.full(problem.ndof, np.nan),
        converged=False,
        iterations=iterations,
        cutbacks=cutbacks,
        reason=reason,
        timing=timing,
    )


def _newton_once(
    problem: CPFEProblem,
    state_n: GlobalMaterialState,
    u0: ArrayLike,
    controls: NewtonControls,
) -> IncrementResult:
    start = perf_counter()
    displacement = np.asarray(u0, dtype=np.float64).copy()
    if displacement.shape != (problem.ndof,):
        raise ValueError("Initial displacement has the wrong global size")
    displacement[problem.prescribed_dofs] = problem.prescribed_values
    free = problem.free_dofs
    reference_residual = 1.0
    last_correction = 0.0
    residual = np.full(problem.ndof, np.nan)
    material_seconds = 0.0
    assembly_seconds = 0.0
    linear_solve_seconds = 0.0

    def record_trial(trial) -> None:
        nonlocal material_seconds, assembly_seconds
        material_seconds += trial.timing.get("material", 0.0)
        assembly_seconds += trial.timing.get("assembly", 0.0)

    def detailed_timing() -> dict[str, float]:
        return {
            "material": material_seconds,
            "assembly": assembly_seconds,
            "linear_solve": linear_solve_seconds,
        }

    for iteration in range(controls.max_iterations + 1):
        trial = assemble_trial(problem, displacement, state_n)
        record_trial(trial)
        if not trial.converged:
            return _failed_result(
                problem,
                state_n,
                trial.reason,
                iterations=iteration,
                elapsed=perf_counter() - start,
                detail=detailed_timing(),
            )
        residual = trial.residual
        free_residual = residual[free]
        residual_norm = float(np.linalg.norm(free_residual))
        if iteration == 0:
            reference_residual = max(residual_norm, 1.0)
        displacement_scale = max(
            float(np.linalg.norm(displacement[free])),
            float(
                np.linalg.norm(
                    problem.prescribed_values
                    - problem.previous_u[problem.prescribed_dofs]
                )
            ),
            1.0e-16,
        )
        residual_ok = residual_norm <= (
            controls.residual_atol + controls.residual_rtol * reference_residual
        )
        correction_ok = last_correction <= (
            controls.correction_atol + controls.correction_rtol * displacement_scale
        )
        if residual_ok and correction_ok:
            return IncrementResult(
                u=displacement.copy(),
                states=trial.states.copied(),
                residual=residual.copy(),
                converged=True,
                iterations=iteration,
                cutbacks=0,
                reason="",
                timing={
                    "newton": perf_counter() - start,
                    **detailed_timing(),
                },
            )
        if iteration == controls.max_iterations:
            break
        if free.size == 0:
            return _failed_result(
                problem,
                state_n,
                "nonzero residual with no free degrees of freedom",
                iterations=iteration,
                elapsed=perf_counter() - start,
                detail=detailed_timing(),
            )
        tangent_free = trial.tangent[free][:, free]
        try:
            linear_start = perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                correction = np.asarray(spsolve(tangent_free, -free_residual))
            linear_solve_seconds += perf_counter() - linear_start
        except (MatrixRankWarning, RuntimeError, ValueError) as error:
            linear_solve_seconds += perf_counter() - linear_start
            return _failed_result(
                problem,
                state_n,
                f"linear solve failed: {error}",
                iterations=iteration,
                elapsed=perf_counter() - start,
                detail=detailed_timing(),
            )
        if correction.shape != (free.size,) or not np.all(np.isfinite(correction)):
            return _failed_result(
                problem,
                state_n,
                "linear solve returned a nonfinite correction",
                iterations=iteration,
                elapsed=perf_counter() - start,
                detail=detailed_timing(),
            )

        accepted = False
        for factor in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
            candidate = displacement.copy()
            candidate[free] += factor * correction
            candidate[problem.prescribed_dofs] = problem.prescribed_values
            candidate_trial = assemble_trial(problem, candidate, state_n)
            record_trial(candidate_trial)
            if not candidate_trial.converged:
                continue
            candidate_norm = float(
                np.linalg.norm(candidate_trial.residual[free])
            )
            if candidate_norm < residual_norm or residual_norm == 0.0:
                displacement = candidate
                last_correction = factor * float(np.linalg.norm(correction))
                accepted = True
                break
        if not accepted:
            return _failed_result(
                problem,
                state_n,
                "line search failed to reduce the free residual",
                iterations=iteration + 1,
                elapsed=perf_counter() - start,
                detail=detailed_timing(),
            )

    return _failed_result(
        problem,
        state_n,
        "Newton iteration limit reached",
        iterations=controls.max_iterations,
        elapsed=perf_counter() - start,
        detail=detailed_timing(),
    )


def _increment_norm(problem: CPFEProblem) -> float:
    return float(
        np.linalg.norm(
            problem.prescribed_values - problem.previous_u[problem.prescribed_dofs]
        )
    )


def _solve_recursive(
    problem: CPFEProblem,
    state_n: GlobalMaterialState,
    u0: ArrayLike,
    controls: NewtonControls,
    depth: int,
) -> IncrementResult:
    force_cutback = _increment_norm(problem) > controls.max_prescribed_increment
    direct = None if force_cutback else _newton_once(problem, state_n, u0, controls)
    if direct is not None and direct.converged:
        return direct
    if depth >= controls.max_cutbacks:
        if direct is not None:
            return direct
        return _failed_result(
            problem,
            state_n,
            "prescribed increment exceeds cutback limit",
            cutbacks=depth,
        )

    previous_values = problem.previous_u[problem.prescribed_dofs]
    midpoint_values = 0.5 * (previous_values + problem.prescribed_values)
    midpoint_problem = replace(
        problem,
        prescribed_values=midpoint_values,
        dt=0.5 * problem.dt,
    )
    midpoint = _solve_recursive(
        midpoint_problem, state_n, u0, controls, depth + 1
    )
    if not midpoint.converged:
        return _failed_result(
            problem,
            state_n,
            f"cutback midpoint failed: {midpoint.reason}",
            iterations=midpoint.iterations,
            cutbacks=1 + midpoint.cutbacks,
            elapsed=midpoint.timing.get("newton", 0.0),
            detail=midpoint.timing,
        )
    endpoint_problem = replace(
        problem,
        previous_u=midpoint.u,
        dt=0.5 * problem.dt,
    )
    endpoint = _solve_recursive(
        endpoint_problem, midpoint.states, midpoint.u, controls, depth + 1
    )
    if not endpoint.converged:
        return _failed_result(
            problem,
            state_n,
            f"cutback endpoint failed: {endpoint.reason}",
            iterations=midpoint.iterations + endpoint.iterations,
            cutbacks=1 + midpoint.cutbacks + endpoint.cutbacks,
            elapsed=midpoint.timing.get("newton", 0.0)
            + endpoint.timing.get("newton", 0.0),
            detail={
                name: midpoint.timing.get(name, 0.0)
                + endpoint.timing.get(name, 0.0)
                for name in ("material", "assembly", "linear_solve")
            },
        )
    return IncrementResult(
        u=endpoint.u,
        states=endpoint.states,
        residual=endpoint.residual,
        converged=True,
        iterations=midpoint.iterations + endpoint.iterations,
        cutbacks=1 + midpoint.cutbacks + endpoint.cutbacks,
        reason="",
        timing={
            name: midpoint.timing.get(name, 0.0)
            + endpoint.timing.get(name, 0.0)
            for name in ("newton", "material", "assembly", "linear_solve")
        },
    )


def solve_increment(
    problem: CPFEProblem,
    state_n: GlobalMaterialState,
    u0: ArrayLike,
    controls: NewtonControls | None = None,
) -> IncrementResult:
    """Solve one increment and commit history only after global convergence."""

    settings = controls or NewtonControls()
    return _solve_recursive(problem, state_n, u0, settings, 0)
