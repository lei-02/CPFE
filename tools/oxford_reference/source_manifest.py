"""Inspect the supplied Oxford UMAT archive without modifying it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from zipfile import ZipFile


@dataclass(frozen=True)
class OxfordCase2Facts:
    iphase: int
    slip_model: int
    nslip: int
    hardening_model: int
    k1: float
    k2: float
    has_kocks_mecking_expression: bool


def _member_with_suffix(archive: ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        raise ValueError(f"Expected one archive member ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def _assignment(text: str, name: str) -> float:
    match = re.search(
        rf"^\s*{re.escape(name)}\s*=\s*([-+0-9.dDeE]+)",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Missing CASE2 assignment for {name}")
    return float(match.group(1).lower().replace("d", "e"))


def inspect_case2_archive(path: Path) -> OxfordCase2Facts:
    """Return source facts from the CASE2 and hardening branches."""

    with ZipFile(path) as archive:
        materials_name = _member_with_suffix(archive, "/usermaterials.f")
        hardening_name = _member_with_suffix(archive, "/hardening.f")
        materials = archive.read(materials_name).decode("latin1")
        hardening = archive.read(hardening_name).decode("latin1")

    case_match = re.search(
        r"\bcase\s*\(\s*2\s*\)(.*?)(?=\bcase\s*\(\s*3\s*\))",
        materials,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if case_match is None:
        raise ValueError("CASE2 block is absent from usermaterials.f")
    case2 = case_match.group(1)

    compact_hardening = re.sub(r"[\s+]", "", hardening).lower()
    kocks_expression = (
        "dssd(is)=(k1/b*dsqrt(rhofor(is))-k2*ssd(is))*"
        "absgammadot(is)*dt"
    )
    return OxfordCase2Facts(
        iphase=int(_assignment(case2, "iphase")),
        slip_model=int(_assignment(case2, "slipmodel")),
        nslip=int(_assignment(case2, "nslip")),
        hardening_model=int(_assignment(case2, "hardeningmodel")),
        k1=_assignment(case2, "hardeningparam(1)"),
        k2=_assignment(case2, "hardeningparam(2)"),
        has_kocks_mecking_expression=kocks_expression in compact_hardening,
    )
