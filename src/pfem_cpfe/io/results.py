"""Compressed full-field NPZ data with a human-readable JSON sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FullFieldResult:
    u: FloatArray
    stress: FloatArray
    strain: FloatArray
    slip: FloatArray
    orientation: FloatArray
    fp: FloatArray
    ssd: FloatArray
    crss: FloatArray
    residual: FloatArray
    converged: bool
    residual_history: FloatArray
    iterations: NDArray[np.int64]
    timing: Mapping[str, float]


def _paths(path: str | Path) -> tuple[Path, Path]:
    requested = Path(path)
    npz_path = requested if requested.suffix.lower() == ".npz" else requested.with_suffix(".npz")
    return npz_path, npz_path.with_suffix(".json")


def save_result(
    path: str | Path, result: FullFieldResult
) -> tuple[Path, Path]:
    """Persist required full fields and return the NPZ and JSON paths."""

    npz_path, json_path = _paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "u": np.asarray(result.u),
        "stress": np.asarray(result.stress),
        "strain": np.asarray(result.strain),
        "slip": np.asarray(result.slip),
        "orientation": np.asarray(result.orientation),
        "fp": np.asarray(result.fp),
        "ssd": np.asarray(result.ssd),
        "crss": np.asarray(result.crss),
        "residual": np.asarray(result.residual),
        "residual_history": np.asarray(result.residual_history),
        "iterations": np.asarray(result.iterations, dtype=np.int64),
    }
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        "schema_version": 1,
        "converged": bool(result.converged),
        "residual_history": arrays["residual_history"].tolist(),
        "iterations": arrays["iterations"].tolist(),
        "timing": {name: float(value) for name, value in result.timing.items()},
    }
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return npz_path, json_path


def load_result(path: str | Path) -> dict[str, object]:
    """Load full fields and sidecar metadata into one mapping."""

    npz_path, json_path = _paths(path)
    with np.load(npz_path, allow_pickle=False) as archive:
        result: dict[str, object] = {
            name: archive[name].copy() for name in archive.files
        }
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    result["schema_version"] = metadata["schema_version"]
    result["converged"] = metadata["converged"]
    result["timing"] = metadata["timing"]
    return result
