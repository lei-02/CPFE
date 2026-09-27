import numpy as np

from pfem_cpfe.contracts import Case2Parameters, MaterialState
from pfem_cpfe.materials.case2_numpy import case2_step_numpy


def replay_numpy_backend(trace):
    parameters = Case2Parameters.default()
    state = MaterialState.initial(parameters)
    previous_deformation = np.eye(3)
    outputs = {
        "stress": [],
        "fp": [],
        "orientation": [],
        "slip": [],
        "ssd": [],
        "crss": [],
    }
    for index, deformation in enumerate(trace["F"]):
        if trace["history_start"][index]:
            state = MaterialState.initial(parameters)
            previous_deformation = np.eye(3)
        result = case2_step_numpy(
            previous_deformation,
            deformation,
            float(trace["dt"][index]),
            float(trace["temperature"]),
            parameters,
            state,
        )
        assert result.converged, f"trace step {index}: {result.reason}"
        state = result.state
        previous_deformation = deformation
        outputs["stress"].append(result.stress)
        outputs["fp"].append(state.fp)
        outputs["orientation"].append(state.orientation)
        outputs["slip"].append(state.slip)
        outputs["ssd"].append(state.ssd)
        outputs["crss"].append(state.crss)
    return {name: np.stack(values) for name, values in outputs.items()}


def test_numpy_backend_matches_oxford_history(oxford_trace):
    result = replay_numpy_backend(oxford_trace)

    np.testing.assert_allclose(result["stress"], oxford_trace["stress"], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(result["ssd"], oxford_trace["ssd"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(result["fp"], oxford_trace["fp"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result["crss"], oxford_trace["crss"], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(result["slip"], oxford_trace["slip"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(
        result["orientation"], oxford_trace["orientation"], rtol=1e-7, atol=1e-9
    )
