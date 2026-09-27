import numpy as np
import pytest

from pfem_cpfe.io.results import FullFieldResult
from pfem_cpfe.mesh.voxel import eight_grain_problem


@pytest.fixture
def solved_eight_grain():
    problem = eight_grain_problem(seed=20260915)
    element_count = problem.elements.shape[0]
    point_shape = (element_count, 8)
    state = problem.states
    return FullFieldResult(
        u=np.zeros((problem.nodes.shape[0], 3)),
        stress=state.stress,
        strain=np.zeros(point_shape + (6,)),
        slip=state.slip,
        orientation=state.orientation,
        fp=state.fp,
        ssd=state.ssd,
        crss=state.crss,
        residual=np.zeros(problem.ndof),
        converged=True,
        residual_history=np.array([0.0]),
        iterations=np.array([0], dtype=np.int64),
        timing={"newton": 0.0, "end_to_end": 0.0},
    )
