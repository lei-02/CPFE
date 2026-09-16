"""Shared immutable data contracts for constitutive and global solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _copied_readonly(value: ArrayLike) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Case2Parameters:
    """Oxford CASE2 constants in the UMAT's MPa-micrometre-second units."""

    nslip: int
    rho0: float
    tauc0: float
    burgers: float
    c11: float
    c12: float
    c44: float
    k1: float
    k2: float

    @classmethod
    def default(cls) -> Case2Parameters:
        return cls(
            nslip=12,
            rho0=0.46,
            tauc0=206.0,
            burgers=2.53e-4,
            c11=289e3,
            c12=124e3,
            c44=83e3,
            k1=0.129,
            k2=10.0,
        )


@dataclass(frozen=True)
class MaterialState:
    """Committed state at one integration point."""

    fp: FloatArray
    orientation: FloatArray
    stress: FloatArray
    ssd: FloatArray
    crss: FloatArray
    slip: FloatArray
    slip_rate: FloatArray

    def __post_init__(self) -> None:
        for name in (
            "fp",
            "orientation",
            "stress",
            "ssd",
            "crss",
            "slip",
            "slip_rate",
        ):
            object.__setattr__(self, name, _copied_readonly(getattr(self, name)))

    @classmethod
    def initial(cls, parameters: Case2Parameters) -> MaterialState:
        return cls(
            fp=np.eye(3),
            orientation=np.eye(3),
            stress=np.zeros(6),
            ssd=np.full(parameters.nslip, parameters.rho0),
            crss=np.full(parameters.nslip, parameters.tauc0),
            slip=np.zeros(parameters.nslip),
            slip_rate=np.zeros(parameters.nslip),
        )

    def copied(self, **changes: ArrayLike) -> MaterialState:
        values = {
            "fp": self.fp,
            "orientation": self.orientation,
            "stress": self.stress,
            "ssd": self.ssd,
            "crss": self.crss,
            "slip": self.slip,
            "slip_rate": self.slip_rate,
        }
        unknown = set(changes) - set(values)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown material-state fields: {names}")
        values.update(changes)
        return MaterialState(**values)


@dataclass(frozen=True)
class TrialResult:
    """Uncommitted response returned by a material-point update."""

    stress: FloatArray
    tangent: FloatArray
    state: MaterialState
    converged: bool
    iterations: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stress", _copied_readonly(self.stress))
        object.__setattr__(self, "tangent", _copied_readonly(self.tangent))

    @classmethod
    def zeros(
        cls,
        state: MaterialState,
        *,
        converged: bool = True,
        iterations: int = 0,
        reason: str = "",
    ) -> TrialResult:
        return cls(
            stress=np.zeros(6),
            tangent=np.zeros((6, 6)),
            state=state,
            converged=converged,
            iterations=iterations,
            reason=reason,
        )


def commit_trial(old: MaterialState, trial: TrialResult) -> MaterialState:
    """Return an independent committed state or reject a failed trial."""

    del old
    if not trial.converged:
        detail = trial.reason or "trial did not converge"
        raise ValueError(f"Cannot commit failed trial: {detail}")
    return trial.state.copied()
