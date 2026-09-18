from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pfem_cpfe.contracts import Case2Parameters, MaterialState
from pfem_cpfe.materials.case2_numpy import case2_step_numpy
from pfem_cpfe.materials.case2_torch import (
    TorchMaterialState,
    case2_step_torch,
)


@dataclass(frozen=True)
class Case2Batch:
    torch_inputs: dict
    numpy_results: SimpleNamespace
    F_np1_double: torch.Tensor
    stress_function: object


@pytest.fixture(params=["cpu", "cuda"], ids=["cpu", "cuda"])
def case2_batch(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    parameters = Case2Parameters.default()
    F_n = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    F_np1 = F_n.copy()
    F_np1[0, 0, 0] = 1.001
    F_np1[1, 0, 1] = 0.002
    dt = 1.0e-8
    temperature = 293.15
    states = [MaterialState.initial(parameters) for _ in range(2)]
    results = [
        case2_step_numpy(F_n[index], F_np1[index], dt, temperature, parameters, states[index])
        for index in range(2)
    ]
    assert all(result.converged for result in results)
    numpy_results = SimpleNamespace(
        stress=np.stack([result.stress for result in results]),
        tangent=np.stack([result.tangent for result in results]),
        state=SimpleNamespace(
            fp=np.stack([result.state.fp for result in results]),
            orientation=np.stack([result.state.orientation for result in results]),
            stress=np.stack([result.state.stress for result in results]),
            ssd=np.stack([result.state.ssd for result in results]),
            crss=np.stack([result.state.crss for result in results]),
            slip=np.stack([result.state.slip for result in results]),
            slip_rate=np.stack([result.state.slip_rate for result in results]),
        ),
    )

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64, device=device)

    torch_state = TorchMaterialState(
        fp=tensor(np.stack([state.fp for state in states])),
        orientation=tensor(np.stack([state.orientation for state in states])),
        stress=tensor(np.stack([state.stress for state in states])),
        ssd=tensor(np.stack([state.ssd for state in states])),
        crss=tensor(np.stack([state.crss for state in states])),
        slip=tensor(np.stack([state.slip for state in states])),
        slip_rate=tensor(np.stack([state.slip_rate for state in states])),
    )
    torch_inputs = {
        "F_n": tensor(F_n),
        "F_np1": tensor(F_np1),
        "dt": dt,
        "temperature": temperature,
        "parameters": parameters,
        "state_n": torch_state,
    }

    def stress_function(deformation):
        inputs = dict(torch_inputs)
        inputs["F_np1"] = deformation
        return case2_step_torch(**inputs).stress.sum()

    return Case2Batch(
        torch_inputs=torch_inputs,
        numpy_results=numpy_results,
        F_np1_double=tensor(F_np1),
        stress_function=stress_function,
    )
