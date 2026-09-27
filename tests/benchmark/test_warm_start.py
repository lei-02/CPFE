from __future__ import annotations

import json


def test_warm_and_zero_start_reach_same_solution(benchmark_report):
    assert benchmark_report.zero.converged
    assert benchmark_report.warm.converged
    assert benchmark_report.field_error("u") < 1.0e-7
    assert benchmark_report.field_error("stress") < 1.0e-6
    assert benchmark_report.field_error("ssd") < 1.0e-6
    assert benchmark_report.field_error("slip") < 1.0e-6
    assert benchmark_report.field_error("orientation") < 1.0e-6
    assert benchmark_report.field_error("fp") < 1.0e-6
    assert benchmark_report.field_error("crss") < 1.0e-6


def test_total_time_includes_inference_and_newton(benchmark_report):
    expected = (
        benchmark_report.warm.inference_seconds
        + benchmark_report.warm.newton_seconds
    )
    assert abs(benchmark_report.warm.total_seconds - expected) < 1.0e-12
    assert (
        benchmark_report.warm.inference_seconds
        == benchmark_report.warm.feature_seconds
        + benchmark_report.warm.operator_seconds
    )
    assert benchmark_report.zero.inference_seconds == 0.0


def test_report_records_repetitions_iterations_and_fallbacks(benchmark_report):
    assert len(benchmark_report.zero.repetitions) == 2
    assert len(benchmark_report.warm.repetitions) == 2
    assert benchmark_report.zero.iterations >= benchmark_report.warm.iterations
    assert benchmark_report.warm.fallbacks >= 0
    payload = json.loads(benchmark_report.report_path.read_text(encoding="utf-8"))
    assert payload["configuration"]["increments"] == 2
    assert payload["warm"]["fallbacks"] == benchmark_report.warm.fallbacks
    assert payload["field_errors"]["stress"] < 1.0e-6


def test_detailed_newton_timing_is_accounted_for(benchmark_report):
    for mode in (benchmark_report.zero, benchmark_report.warm):
        assert mode.material_seconds >= 0.0
        assert mode.assembly_seconds >= 0.0
        assert mode.linear_solve_seconds >= 0.0
        assert (
            mode.material_seconds
            + mode.assembly_seconds
            + mode.linear_solve_seconds
            <= mode.newton_seconds * (1.0 + 1.0e-9)
        )
