"""Generate deterministic Oxford CASE2 material-point history fixtures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


_HISTORY_NAMES = {1: "tension", 2: "compression", 3: "shear", 4: "unload", 5: "reload"}


def _read_float64(stream, count: int) -> np.ndarray:
    values = np.fromfile(stream, dtype="<f8", count=count)
    if values.size != count:
        raise RuntimeError("truncated Oxford reference trace")
    return values


def _read_int32(stream, count: int) -> np.ndarray:
    values = np.fromfile(stream, dtype="<i4", count=count)
    if values.size != count:
        raise RuntimeError("truncated Oxford reference trace")
    return values


def _read_matrix_history(stream, rows: int, columns: int, steps: int) -> np.ndarray:
    values = _read_float64(stream, rows * columns * steps)
    return values.reshape((rows, columns, steps), order="F").transpose(2, 0, 1)


def _read_vector_history(stream, width: int, steps: int) -> np.ndarray:
    values = _read_float64(stream, width * steps)
    return values.reshape((width, steps), order="F").T


def _run_reference(trace_path: Path) -> None:
    driver = Path(os.environ["OXFORD_REFERENCE_DRIVER"])
    environment = os.environ.copy()
    runtime = environment.get("OXFORD_GFORTRAN_BIN")
    if runtime:
        environment["PATH"] = runtime + os.pathsep + environment["PATH"]
    subprocess.run([driver, trace_path], check=True, env=environment)


def generate(output: Path) -> None:
    temperature = 293.15
    records: dict[str, list[np.ndarray]] = {
        "F": [],
        "dt": [],
        "history_start": [],
        "history_name": [],
        "stress": [],
        "fp": [],
        "orientation": [],
        "slip": [],
        "ssd": [],
        "crss": [],
    }
    with tempfile.TemporaryDirectory() as temporary:
        raw_trace = Path(temporary) / "oxford_case2_trace.bin"
        _run_reference(raw_trace)
        with raw_trace.open("rb") as stream:
            history_count = int(_read_int32(stream, 1)[0])
            for _ in range(history_count):
                history_id, count = (int(value) for value in _read_int32(stream, 2))
                name = _HISTORY_NAMES[history_id]
                history = _read_matrix_history(stream, 3, 3, count)
                dt = _read_float64(stream, count)
                stress = _read_vector_history(stream, 6, count)
                fp = _read_matrix_history(stream, 3, 3, count)
                orientation = _read_matrix_history(stream, 3, 3, count)
                slip = _read_vector_history(stream, 12, count)
                ssd = _read_vector_history(stream, 12, count)
                crss = _read_vector_history(stream, 12, count)
                status = _read_int32(stream, count)
                if np.any(status != 0):
                    index = int(np.flatnonzero(status != 0)[0])
                    raise RuntimeError(
                        f"Fortran CASE2 failed in {name} at step {index}: {status[index]}"
                    )
                records["F"].append(history)
                records["dt"].append(dt)
                starts = np.zeros(count, dtype=np.bool_)
                starts[0] = True
                records["history_start"].append(starts)
                records["history_name"].append(np.full(count, name, dtype="U16"))
                records["stress"].append(stress)
                records["fp"].append(fp)
                records["orientation"].append(orientation)
                records["slip"].append(slip)
                records["ssd"].append(ssd)
                records["crss"].append(crss)

    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in records.items()}
    arrays["temperature"] = np.array(temperature)
    np.savez_compressed(output, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.output)


if __name__ == "__main__":
    main()
