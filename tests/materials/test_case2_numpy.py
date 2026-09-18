import numpy as np

from pfem_cpfe.contracts import Case2Parameters, MaterialState
from pfem_cpfe.materials.case2_numpy import case2_step_numpy


def test_identity_increment_preserves_state():
    parameters = Case2Parameters.default()
    state = MaterialState.initial(parameters)

    result = case2_step_numpy(
        np.eye(3),
        np.eye(3),
        1.0e-3,
        293.15,
        parameters,
        state,
    )

    assert result.converged
    np.testing.assert_allclose(result.stress, 0.0, atol=1.0e-12)
    np.testing.assert_array_equal(result.state.ssd, state.ssd)
    np.testing.assert_array_equal(result.state.fp, state.fp)


def test_small_uniaxial_increment_recovers_cubic_elastic_response():
    parameters = Case2Parameters.default()
    state = MaterialState.initial(parameters)
    deformation = np.diag([1.0 + 1.0e-8, 1.0, 1.0])

    result = case2_step_numpy(
        np.eye(3),
        deformation,
        1.0e-3,
        293.15,
        parameters,
        state,
    )

    strain_11 = (deformation[0, 0] - 1.0) / deformation[0, 0]
    np.testing.assert_allclose(
        result.stress,
        [parameters.c11 * strain_11, parameters.c12 * strain_11, parameters.c12 * strain_11, 0.0, 0.0, 0.0],
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_invalid_trial_leaves_input_state_unchanged():
    parameters = Case2Parameters.default()
    state = MaterialState.initial(parameters)
    before = state.ssd.copy()

    result = case2_step_numpy(
        np.eye(3),
        np.diag([-1.0, 1.0, 1.0]),
        1.0e-3,
        293.15,
        parameters,
        state,
    )

    assert not result.converged
    np.testing.assert_array_equal(state.ssd, before)


def test_elastic_algorithmic_tangent_matches_finite_difference():
    parameters = Case2Parameters.default()
    state = MaterialState.initial(parameters)
    base = np.diag([1.0 + 1.0e-8, 1.0, 1.0])
    perturbation = 1.0e-9

    result = case2_step_numpy(np.eye(3), base, 1.0e-3, 293.15, parameters, state)
    perturbed = base.copy()
    perturbed[0, 0] += perturbation
    perturbed_result = case2_step_numpy(
        np.eye(3), perturbed, 1.0e-3, 293.15, parameters, state
    )
    base_strain = (base[0, 0] - 1.0) / base[0, 0]
    perturbed_strain = (perturbed[0, 0] - 1.0) / perturbed[0, 0]
    finite_difference = (perturbed_result.stress - result.stress) / (
        perturbed_strain - base_strain
    )

    np.testing.assert_allclose(
        result.tangent[:, 0], finite_difference, rtol=5.0e-4, atol=1.0e-5
    )
