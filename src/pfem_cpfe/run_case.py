"""Run deterministic standalone CPFE reference cases from zero initial guesses."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pfem_cpfe.elements.c3d8 import c3d8_kinematics
from pfem_cpfe.io.results import FullFieldResult, save_result
from pfem_cpfe.mesh.voxel import eight_grain_problem, single_element_problem
from pfem_cpfe.solver.newton import solve_increment


def _strain_field(problem, displacement: np.ndarray) -> np.ndarray:
    nodal = displacement.reshape((-1, 3))
    strain = np.empty((problem.elements.shape[0], 8, 6))
    identity = np.eye(3)
    for element, connectivity in enumerate(problem.elements):
        kinematics = c3d8_kinematics(
            problem.nodes[connectivity], nodal[connectivity]
        )
        for point, deformation in enumerate(kinematics.F):
            green = 0.5 * (deformation.T @ deformation - identity)
            strain[element, point] = (
                green[0, 0],
                green[1, 1],
                green[2, 2],
                2.0 * green[0, 1],
                2.0 * green[0, 2],
                2.0 * green[1, 2],
            )
    return strain


def run_case(case: str, increments: int, output: str | Path) -> FullFieldResult:
    if increments <= 0:
        raise ValueError("Increment count must be positive")
    if case == "eight-grain":
        base = eight_grain_problem()
    elif case == "single-crystal":
        base = single_element_problem()
    else:
        raise ValueError(f"Unknown case: {case}")

    start = perf_counter()
    displacement = base.previous_u.copy()
    states = base.states.copied()
    final_values = base.prescribed_values.copy()
    residual_history: list[float] = []
    iterations: list[int] = []
    newton_time = 0.0
    last_result = None
    for increment in range(1, increments + 1):
        target = final_values * (increment / increments)
        problem = replace(
            base,
            states=states,
            previous_u=displacement,
            prescribed_values=target,
        )
        last_result = solve_increment(
            problem, state_n=states, u0=displacement
        )
        iterations.append(last_result.iterations)
        newton_time += last_result.timing.get("newton", 0.0)
        free = problem.free_dofs
        residual_history.append(float(np.linalg.norm(last_result.residual[free])))
        if not last_result.converged:
            break
        displacement = last_result.u
        states = last_result.states

    if last_result is None:
        raise RuntimeError("No increment was attempted")
    elapsed = perf_counter() - start
    result = FullFieldResult(
        u=displacement.reshape((-1, 3)),
        stress=states.stress,
        strain=_strain_field(base, displacement),
        slip=states.slip,
        orientation=states.orientation,
        fp=states.fp,
        ssd=states.ssd,
        crss=states.crss,
        residual=last_result.residual,
        converged=last_result.converged and len(iterations) == increments,
        residual_history=np.asarray(residual_history),
        iterations=np.asarray(iterations, dtype=np.int64),
        timing={"newton": newton_time, "end_to_end": elapsed},
    )
    npz_path, json_path = save_result(output, result)
    print(
        json.dumps(
            {
                "converged": result.converged,
                "increments": len(iterations),
                "npz": str(npz_path),
                "metadata": str(json_path),
                "timing": result.timing,
            },
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("single-crystal", "eight-grain"), default="eight-grain"
    )
    parser.add_argument("--increments", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_case(arguments.case, arguments.increments, arguments.output)
    if not result.converged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
