"""Fair zero-start versus PFEM-operator warm-start CPFE benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from pfem_cpfe.elements.c3d8 import c3d8_kinematics
from pfem_cpfe.io.results import FullFieldResult, save_result
from pfem_cpfe.mesh.voxel import eight_grain_problem, single_element_problem
from pfem_cpfe.operator.features import build_node_features, enforce_displacement_bc
from pfem_cpfe.operator.transolver import TransolverCPFE
from pfem_cpfe.run_case import _strain_field
from pfem_cpfe.solver.assembly import assemble_trial
from pfem_cpfe.solver.newton import NewtonControls, solve_increment


Predictor = Callable[[object, object, float, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BenchmarkConfig:
    output: Path
    case: str = "eight-grain"
    checkpoint: Path | None = None
    increments: int = 20
    repetitions: int = 5
    warmup_repetitions: int = 1
    cpu_threads: int = 1
    device: str = "cpu"
    controls: NewtonControls = NewtonControls()


@dataclass(frozen=True)
class RepetitionRecord:
    converged: bool
    feature_seconds: float
    operator_seconds: float
    inference_seconds: float
    material_seconds: float
    assembly_seconds: float
    linear_solve_seconds: float
    newton_seconds: float
    total_seconds: float
    iterations: int
    fallbacks: int


@dataclass(frozen=True)
class ModeSummary:
    converged: bool
    feature_seconds: float
    operator_seconds: float
    inference_seconds: float
    material_seconds: float
    assembly_seconds: float
    linear_solve_seconds: float
    newton_seconds: float
    total_seconds: float
    iterations: int
    fallbacks: int
    repetitions: tuple[RepetitionRecord, ...]
    result: FullFieldResult


@dataclass(frozen=True)
class BenchmarkReport:
    configuration: BenchmarkConfig
    zero: ModeSummary
    warm: ModeSummary
    field_errors: dict[str, float]
    report_path: Path

    def field_error(self, name: str) -> float:
        return self.field_errors[name]

    @property
    def speedup_ratio(self) -> float:
        if self.warm.total_seconds <= 0.0:
            return float("inf")
        return self.zero.total_seconds / self.warm.total_seconds


class CheckpointPredictor:
    """Frozen Transolver checkpoint returning dimensioned nodal increments."""

    def __init__(self, checkpoint: Path, device: str) -> None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        payload = torch.load(checkpoint, map_location=requested, weights_only=True)
        self.feature_dim = int(payload["feature_dim"])
        self.model = TransolverCPFE(
            feature_dim=self.feature_dim, **payload["config"]
        ).to(device=requested, dtype=torch.float64)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.device = requested

    def __call__(self, problem, state_n, load_fraction, features):
        del state_n, load_fraction
        if features.shape[1] != self.feature_dim:
            raise ValueError("Checkpoint feature layout does not match the problem")
        xyz = torch.tensor(
            np.array(problem.nodes[None], copy=True),
            dtype=torch.float64,
            device=self.device,
        )
        prompt = torch.as_tensor(
            features[None], dtype=torch.float64, device=self.device
        )
        mask = torch.ones(
            (1, problem.nodes.shape[0]), dtype=torch.bool, device=self.device
        )
        with torch.inference_mode():
            raw = self.model(xyz, prompt, mask)[0]
        scale = max(
            float(np.max(np.abs(problem.prescribed_increment), initial=0.0)),
            1.0e-12,
        )
        return raw.detach().cpu().numpy() * scale


def _synchronize(predictor: object | None) -> None:
    device = getattr(predictor, "device", None)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _base_problem(case: str):
    if case == "single-crystal":
        return single_element_problem()
    if case == "eight-grain":
        return eight_grain_problem()
    raise ValueError(f"Unknown benchmark case: {case}")


def _admissible_guess(problem, increment: np.ndarray) -> np.ndarray | None:
    values = np.asarray(increment, dtype=np.float64)
    if values.size != problem.ndof or not np.all(np.isfinite(values)):
        return None
    values = enforce_displacement_bc(values.reshape((-1, 3)), problem).reshape(-1)
    guess = problem.previous_u + values
    try:
        nodal = guess.reshape((-1, 3))
        for connectivity in problem.elements:
            c3d8_kinematics(problem.nodes[connectivity], nodal[connectivity])
    except ValueError:
        return None
    return guess


def _execute_once(config: BenchmarkConfig, predictor: Predictor | None):
    base = _base_problem(config.case)
    displacement = base.previous_u.copy()
    states = base.states.copied()
    final_values = base.prescribed_values.copy()
    feature_seconds = 0.0
    operator_seconds = 0.0
    newton_seconds = 0.0
    material_seconds = 0.0
    assembly_seconds = 0.0
    linear_solve_seconds = 0.0
    iterations = 0
    fallbacks = 0
    residual_history: list[float] = []
    iteration_history: list[int] = []
    last = None

    for increment_index in range(1, config.increments + 1):
        fraction = increment_index / config.increments
        problem = replace(
            base,
            states=states,
            previous_u=displacement,
            prescribed_values=final_values * fraction,
        )
        guess = displacement.copy()
        used_operator_guess = False
        if predictor is not None:
            feature_start = perf_counter()
            features = build_node_features(problem, states, fraction)
            feature_seconds += perf_counter() - feature_start
            _synchronize(predictor)
            operator_start = perf_counter()
            predicted_increment = predictor(problem, states, fraction, features)
            _synchronize(predictor)
            operator_seconds += perf_counter() - operator_start
            candidate = _admissible_guess(problem, predicted_increment)
            if candidate is None:
                fallbacks += 1
            else:
                validation_start = perf_counter()
                validation = assemble_trial(problem, candidate, states)
                newton_seconds += perf_counter() - validation_start
                material_seconds += validation.timing.get("material", 0.0)
                assembly_seconds += validation.timing.get("assembly", 0.0)
                if validation.converged:
                    guess = candidate
                    used_operator_guess = True
                else:
                    fallbacks += 1

        result = solve_increment(
            problem, state_n=states, u0=guess, controls=config.controls
        )
        attempts = [result]
        if predictor is not None and used_operator_guess and not result.converged:
            fallbacks += 1
            result = solve_increment(
                problem,
                state_n=states,
                u0=displacement,
                controls=config.controls,
            )
            attempts.append(result)
        for attempt in attempts:
            newton_seconds += attempt.timing.get("newton", 0.0)
            material_seconds += attempt.timing.get("material", 0.0)
            assembly_seconds += attempt.timing.get("assembly", 0.0)
            linear_solve_seconds += attempt.timing.get("linear_solve", 0.0)
            iterations += attempt.iterations
        iteration_history.append(result.iterations)
        last = result
        if not result.converged:
            break
        displacement = result.u
        states = result.states
        residual_history.append(float(np.linalg.norm(result.residual[problem.free_dofs])))

    if last is None:
        raise RuntimeError("Benchmark must contain at least one increment")
    inference_seconds = feature_seconds + operator_seconds
    total_seconds = inference_seconds + newton_seconds
    field = FullFieldResult(
        u=displacement.reshape((-1, 3)),
        stress=states.stress,
        strain=_strain_field(base, displacement),
        slip=states.slip,
        orientation=states.orientation,
        fp=states.fp,
        ssd=states.ssd,
        crss=states.crss,
        residual=last.residual,
        converged=last.converged and len(iteration_history) == config.increments,
        residual_history=np.asarray(residual_history),
        iterations=np.asarray(iteration_history, dtype=np.int64),
        timing={
            "feature": feature_seconds,
            "operator": operator_seconds,
            "inference": inference_seconds,
            "material": material_seconds,
            "assembly": assembly_seconds,
            "linear_solve": linear_solve_seconds,
            "newton": newton_seconds,
            "total": total_seconds,
        },
    )
    record = RepetitionRecord(
        converged=field.converged,
        feature_seconds=feature_seconds,
        operator_seconds=operator_seconds,
        inference_seconds=inference_seconds,
        material_seconds=material_seconds,
        assembly_seconds=assembly_seconds,
        linear_solve_seconds=linear_solve_seconds,
        newton_seconds=newton_seconds,
        total_seconds=total_seconds,
        iterations=iterations,
        fallbacks=fallbacks,
    )
    return record, field


def _summarize(records, result) -> ModeSummary:
    def middle(name: str) -> float:
        return float(median(getattr(record, name) for record in records))

    feature_seconds = middle("feature_seconds")
    operator_seconds = middle("operator_seconds")
    inference_seconds = feature_seconds + operator_seconds
    newton_seconds = middle("newton_seconds")
    return ModeSummary(
        converged=all(record.converged for record in records),
        feature_seconds=feature_seconds,
        operator_seconds=operator_seconds,
        inference_seconds=inference_seconds,
        material_seconds=middle("material_seconds"),
        assembly_seconds=middle("assembly_seconds"),
        linear_solve_seconds=middle("linear_solve_seconds"),
        newton_seconds=newton_seconds,
        total_seconds=inference_seconds + newton_seconds,
        iterations=int(median(record.iterations for record in records)),
        fallbacks=int(median(record.fallbacks for record in records)),
        repetitions=tuple(records),
        result=result,
    )


def _run_mode(config: BenchmarkConfig, predictor: Predictor | None) -> ModeSummary:
    for _ in range(config.warmup_repetitions):
        _execute_once(config, predictor)
    records = []
    result = None
    for _ in range(config.repetitions):
        record, result = _execute_once(config, predictor)
        records.append(record)
    assert result is not None
    return _summarize(records, result)


def _relative_field_error(zero: FullFieldResult, warm: FullFieldResult, name: str):
    reference = np.asarray(getattr(zero, name))
    difference = np.asarray(getattr(warm, name)) - reference
    return float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1.0e-12))


def _mode_payload(mode: ModeSummary) -> dict[str, object]:
    payload = asdict(mode)
    payload.pop("result")
    return payload


def run_benchmark(
    config: BenchmarkConfig, predictor: Predictor | None = None
) -> BenchmarkReport:
    """Run matched repeated solves and persist fields plus a JSON report."""

    if config.increments <= 0 or config.repetitions <= 0:
        raise ValueError("Increments and repetitions must be positive")
    if config.warmup_repetitions < 0 or config.cpu_threads <= 0:
        raise ValueError("Warm-up repetitions must be nonnegative and threads positive")
    config.output.mkdir(parents=True, exist_ok=True)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(config.cpu_threads)
    torch.set_num_threads(config.cpu_threads)
    try:
        torch.set_num_interop_threads(config.cpu_threads)
    except RuntimeError:
        pass
    if predictor is None:
        if config.checkpoint is None:
            raise ValueError("A checkpoint or explicit predictor is required")
        predictor = CheckpointPredictor(config.checkpoint, config.device)

    zero = _run_mode(config, None)
    warm = _run_mode(config, predictor)
    field_names = ("u", "stress", "ssd", "slip", "orientation", "fp", "crss")
    errors = {
        name: _relative_field_error(zero.result, warm.result, name)
        for name in field_names
    }
    save_result(config.output / "zero", zero.result)
    save_result(config.output / "warm", warm.result)
    report_path = config.output / "report.json"
    configuration = {
        "case": config.case,
        "checkpoint": None if config.checkpoint is None else str(config.checkpoint),
        "increments": config.increments,
        "repetitions": config.repetitions,
        "warmup_repetitions": config.warmup_repetitions,
        "cpu_threads": config.cpu_threads,
        "device": config.device,
    }
    payload = {
        "configuration": configuration,
        "zero": _mode_payload(zero),
        "warm": _mode_payload(warm),
        "field_errors": errors,
        "speedup_ratio": (
            float("inf")
            if warm.total_seconds <= 0.0
            else zero.total_seconds / warm.total_seconds
        ),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return BenchmarkReport(config, zero, warm, errors, report_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("single-crystal", "eight-grain"), default="eight-grain")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--increments", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_benchmark(BenchmarkConfig(**vars(arguments)))
    print(
        json.dumps(
            {
                "report": str(report.report_path),
                "zero_converged": report.zero.converged,
                "warm_converged": report.warm.converged,
                "speedup_ratio": report.speedup_ratio,
                "field_errors": report.field_errors,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
