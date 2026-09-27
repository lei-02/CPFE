from dataclasses import dataclass, replace

import numpy as np
import pytest

from pfem_cpfe.contracts import Case2Parameters
from pfem_cpfe.solver.assembly import CPFEProblem, GlobalMaterialState


@dataclass(frozen=True)
class SingleCrystalCase:
    problem: CPFEProblem
    states: GlobalMaterialState
    small_admissible_guess: np.ndarray

    def with_impossible_increment(self) -> CPFEProblem:
        values = self.problem.prescribed_values.copy()
        top_z = np.array([3 * node + 2 for node in (4, 5, 6, 7)])
        values[np.isin(self.problem.prescribed_dofs, top_z)] = -2.0
        return replace(self.problem, prescribed_values=values)


@pytest.fixture
def single_crystal_problem():
    nodes = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
    )
    elements = np.arange(8, dtype=np.int64)[None, :]
    parameters = Case2Parameters.default()
    states = GlobalMaterialState.initial(1, parameters)
    prescribed = np.array(
        sorted(
            {3 * node + 2 for node in (0, 1, 2, 3, 4, 5, 6, 7)}
            | {0, 1, 4}
        ),
        dtype=np.int64,
    )
    values = np.zeros(prescribed.size)
    top_z = np.array([3 * node + 2 for node in (4, 5, 6, 7)])
    values[np.isin(prescribed, top_z)] = 1.0e-7
    problem = CPFEProblem(
        nodes=nodes,
        elements=elements,
        parameters=parameters,
        states=states,
        prescribed_dofs=prescribed,
        prescribed_values=values,
        previous_u=np.zeros(24),
        external_force=np.zeros(24),
        dt=1.0e-3,
        temperature=293.15,
    )
    rng = np.random.default_rng(20260915)
    guess = rng.normal(scale=1.0e-9, size=24)
    guess[prescribed] = values
    return SingleCrystalCase(problem=problem, states=states, small_admissible_guess=guess)
