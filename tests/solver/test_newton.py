from dataclasses import replace

import numpy as np
from scipy.sparse import csr_matrix

from pfem_cpfe.solver.assembly import assemble_trial
from pfem_cpfe.solver.newton import NewtonControls, solve_increment


def test_global_assembly_returns_sparse_finite_trial(single_crystal_problem):
    case = single_crystal_problem
    u = np.zeros(case.problem.ndof)
    u[case.problem.prescribed_dofs] = case.problem.prescribed_values

    trial = assemble_trial(case.problem, u, case.states)

    assert trial.converged
    assert isinstance(trial.tangent, csr_matrix)
    assert trial.tangent.shape == (case.problem.ndof, case.problem.ndof)
    assert np.all(np.isfinite(trial.residual))
    assert trial.states.ssd.shape == (1, 8, 12)


def test_zero_and_perturbed_starts_converge_to_same_single_element_solution(
    single_crystal_problem,
):
    case = single_crystal_problem
    state_n = case.states.copied()

    zero = solve_increment(
        case.problem, state_n=state_n, u0=np.zeros(case.problem.ndof)
    )
    perturbed = solve_increment(
        case.problem, state_n=state_n, u0=case.small_admissible_guess
    )

    assert zero.converged and perturbed.converged
    np.testing.assert_allclose(zero.u, perturbed.u, rtol=1.0e-9, atol=1.0e-11)
    np.testing.assert_allclose(
        zero.states.ssd, perturbed.states.ssd, rtol=1.0e-8, atol=1.0e-10
    )


def test_prescribed_increment_can_be_bisected_and_committed_atomically(
    single_crystal_problem,
):
    case = single_crystal_problem
    controls = NewtonControls(max_cutbacks=1, max_prescribed_increment=1.1e-7)

    result = solve_increment(
        case.problem,
        state_n=case.states.copied(),
        u0=np.zeros(case.problem.ndof),
        controls=controls,
    )

    assert result.converged
    assert result.cutbacks == 1
    np.testing.assert_allclose(
        result.u[case.problem.prescribed_dofs], case.problem.prescribed_values
    )


def test_failed_increment_preserves_committed_history(single_crystal_problem):
    case = single_crystal_problem
    old = case.states.copied()
    impossible = case.with_impossible_increment()

    result = solve_increment(
        impossible,
        state_n=old,
        u0=np.zeros(impossible.ndof),
        controls=NewtonControls(max_cutbacks=2),
    )

    assert not result.converged
    np.testing.assert_array_equal(old.ssd, case.states.ssd)
    np.testing.assert_array_equal(result.states.ssd, old.ssd)
