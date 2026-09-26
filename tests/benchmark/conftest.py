from __future__ import annotations

from pathlib import Path

import pytest

from pfem_cpfe.benchmark import BenchmarkConfig, run_benchmark
from pfem_cpfe.solver.newton import solve_increment


class ExactAdmissibleGuess:
    def __call__(self, problem, state_n, load_fraction, features):
        del load_fraction, features
        solution = solve_increment(problem, state_n, problem.previous_u)
        assert solution.converged
        return solution.u - problem.previous_u


@pytest.fixture(scope="session")
def benchmark_report(tmp_path_factory):
    output = Path(tmp_path_factory.mktemp("benchmark"))
    return run_benchmark(
        BenchmarkConfig(
            output=output,
            case="single-crystal",
            increments=2,
            repetitions=2,
            warmup_repetitions=1,
            cpu_threads=1,
        ),
        predictor=ExactAdmissibleGuess(),
    )
