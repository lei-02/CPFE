# PFEM-CPFE

This project transfers the PFEM Transolver approach to an incremental,
path-dependent crystal-plasticity finite-element solver. The operator predicts
only nodal displacement increments; the standalone physical solver computes
stress, strain, slip, lattice orientation, plastic deformation gradient, SSD,
and CRSS.

The first milestone targets Oxford UMAT CASE2, twelve FCC slip systems,
Kocks-Mecking hardening, fully integrated C3D8 elements, and a deterministic
eight-grain voxel aggregate. Abaqus is not used during formal prediction.

## Development environment

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
uv pip install --python .venv/Scripts/python.exe --reinstall-package torch --no-deps "torch==2.14.0+cu130" --index-url https://download.pytorch.org/whl/cu130
.venv/Scripts/python.exe -m pytest -v
```

The second install command selects the official CUDA 13.0 PyTorch wheel for
the NVIDIA development host. CPU-only hosts may omit that command.

The detailed approved design and implementation plan are in
`docs/superpowers/specs/` and `docs/superpowers/plans/`.
