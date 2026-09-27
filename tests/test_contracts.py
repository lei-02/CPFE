import numpy as np
import pytest

from pfem_cpfe.contracts import (
    Case2Parameters,
    MaterialState,
    TrialResult,
    commit_trial,
)


def test_case2_defaults_match_oxford_source():
    parameters = Case2Parameters.default()

    assert parameters.nslip == 12
    assert parameters.k1 == 0.129
    assert parameters.k2 == 10.0
    assert parameters.burgers == 2.53e-4
    assert parameters.c11 == 289e3
    assert parameters.c12 == 124e3
    assert parameters.c44 == 83e3


def test_commit_returns_independent_state_without_mutating_old_state():
    parameters = Case2Parameters.default()
    old = MaterialState.initial(parameters)
    trial_state = old.copied(ssd=old.ssd + 0.1)
    trial = TrialResult.zeros(trial_state)

    new = commit_trial(old, trial)

    np.testing.assert_array_equal(old.ssd, np.full(12, 0.46))
    np.testing.assert_array_equal(new.ssd, np.full(12, 0.56))
    assert not np.shares_memory(old.ssd, new.ssd)


def test_failed_trial_cannot_be_committed():
    parameters = Case2Parameters.default()
    old = MaterialState.initial(parameters)
    failed = TrialResult.zeros(old, converged=False, reason="local iteration failed")

    with pytest.raises(ValueError, match="local iteration failed"):
        commit_trial(old, failed)
