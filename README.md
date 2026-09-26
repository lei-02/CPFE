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

## Reproduction workflow

All commands below run from the repository root in PowerShell. To verify the
supplied Oxford source contract without modifying the archive:

```powershell
$env:OXFORD_UMAT_ZIP = "C:\path\to\OXFORD-UMAT v3.3lsh.zip"
.venv/Scripts/python.exe -m pytest tests/reference/test_oxford_manifest.py -v
```

The checked-in parity fixture can be regenerated with the isolated Fortran
environment expected at `.fortran-reference/`:

```powershell
tools/oxford_reference/build_reference.ps1
.venv/Scripts/python.exe -m pytest tests/reference/test_case2_parity.py -v
```

Run the standalone zero-start reference solve without Abaqus:

```powershell
.venv/Scripts/python.exe -m pfem_cpfe.run_case --case eight-grain --increments 20 --output artifacts/zero-eight-grain
```

Train the initial Transolver patch using only discrete CPFE equilibrium (no
reference displacement labels):

```powershell
.venv/Scripts/python.exe -c "from pathlib import Path; from pfem_cpfe.training import PatchTrainingConfig, train_patch; print(train_patch(PatchTrainingConfig(output=Path('artifacts/patch'), steps=200)))"
```

Freeze that checkpoint and run matched zero-start and operator-warm-start
solves. The benchmark performs one discarded warm-up and reports the median of
five measured repetitions:

```powershell
.venv/Scripts/python.exe -m pfem_cpfe.benchmark --case eight-grain --checkpoint artifacts/patch/model.pt --increments 20 --repetitions 5 --output artifacts/milestone-1
```

`artifacts/milestone-1/report.json` records every repetition and the median
feature, operator, material, assembly, linear-solve, Newton, and total times.
`speedup_ratio` is zero-start total time divided by warm-start total time, so
only values greater than one demonstrate acceleration. `field_errors` must be
checked together with convergence, and `fallbacks` counts rejected or failed
operator guesses. A ratio below one or nonzero field error is retained as an
experimental result rather than relabeled as acceleration.

### Milestone 1 measured result

The deterministic eight-grain command above was run with 20 increments, one
discarded warm-up, five measured repetitions, and the one-sample patch
checkpoint. Both modes converged to the same fields (maximum reported relative
field error `3.16e-12`). The zero-start median component total was `284.831 s`;
the warm-start total was `293.841 s`, with `0.602 s` spent on feature
construction plus operator inference. Both modes required 143 Newton
iterations, and the warm path rejected five operator guesses per repetition.
The measured speedup ratio was therefore `0.9693`: this checkpoint was about
3.2% slower and does **not** demonstrate acceleration. The result identifies
broader physics-only training across multigrain plastic increments as the next
experimental need; optimizing the already-small network forward cost would
not address the observed bottleneck.

Run the complete verification suite with:

```powershell
.venv/Scripts/python.exe -m pytest -v
```
