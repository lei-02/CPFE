from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def oxford_trace():
    path = Path(__file__).parents[1] / "fixtures" / "oxford_case2_trace.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
