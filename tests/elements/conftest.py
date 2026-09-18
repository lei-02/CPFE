from dataclasses import dataclass

import numpy as np
import pytest

from pfem_cpfe.contracts import Case2Parameters, MaterialState


@dataclass(frozen=True)
class UnitCube:
    nodes: np.ndarray
    parameters: Case2Parameters
    states: tuple[MaterialState, ...]


@pytest.fixture
def unit_cube():
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
    parameters = Case2Parameters.default()
    states = tuple(MaterialState.initial(parameters) for _ in range(8))
    return UnitCube(nodes=nodes, parameters=parameters, states=states)
