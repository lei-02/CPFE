# PFEM-CPFE Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a minimal Abaqus-independent CASE2 CPFE solver, then train the PFEM Transolver with unlabeled physics loss to warm-start that solver on a single crystal and a deterministic eight-grain voxel aggregate.

**Architecture:** A source-locked Oxford CASE2 material-point kernel supplies stress, consistent tangent, and trial history at C3D8 integration points. A standalone incremental Newton solver provides the high-fidelity solution. A Transolver consumes node point-cloud features and predicts only the next displacement increment; explicit C3D8 differentiation and a differentiable CASE2 backend provide the physics loss, while the CPU solver remains the final authority.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyTorch, einops, pytest, NPZ/JSON outputs, and a separate GNU Fortran plus f2py reference environment.

**Spec:** `docs/superpowers/specs/2026-09-15-pfem-cpfe-design.md`

## Global Constraints

- Formal prediction must not call Abaqus.
- Transolver outputs only three nodal displacement-increment components.
- Stress, strain, slip, orientation, `Fp`, SSD, and CRSS are produced by the physical kernel after convergence.
- History is stored at integration points and is committed only after global convergence.
- The first milestone uses Oxford CASE2, FCC 12-slip Kocks-Mecking hardening, C3D8 full integration, monotonic displacement control, and a `4 x 4 x 4` eight-grain aggregate.
- The first milestone excludes GND, periodic boundaries, multiple hardening equations, dynamics, damage, and industrial-scale meshes.
- Physics training uses explicit finite-element shape-function differentiation and no CPFEM solution labels.
- Speed comparisons use identical mesh, increments, tolerances, hardware, threads, and constitutive backends, and report operator inference plus Newton time.
- The repository currently has no installed PyTorch or Fortran compiler; dependency downloads and compiler installation require approval during execution.
- Oxford parity is a hard gate. If a reference compiler cannot be installed, implementation may continue, but milestone completion must remain unclaimed until the reference traces are produced and compared.

## File Map

```text
pyproject.toml                         Package and dependency contract
.gitignore                             Local environments, caches, and generated artifacts
README.md                              Reproducible commands and result definitions
src/pfem_cpfe/contracts.py             Material, mesh, trial, and solver dataclasses
src/pfem_cpfe/crystal/fcc.py           FCC systems and orientation operations
src/pfem_cpfe/materials/case2_laws.py  CASE2 slip, hardening, and CRSS laws
src/pfem_cpfe/materials/case2_numpy.py CPU material-point integration
src/pfem_cpfe/materials/case2_torch.py Differentiable batched material integration
src/pfem_cpfe/elements/c3d8.py         Shape functions, quadrature, kinematics, element response
src/pfem_cpfe/solver/assembly.py       Global residual and tangent assembly
src/pfem_cpfe/solver/newton.py         Increment solve, line search, cutback, commit
src/pfem_cpfe/mesh/voxel.py            Single-element and eight-grain mesh generation
src/pfem_cpfe/io/results.py            Stable NPZ/JSON result schema
src/pfem_cpfe/run_case.py               Standalone zero-start CPFE command
src/pfem_cpfe/operator/transolver.py   PFEM physics-attention neural operator
src/pfem_cpfe/operator/features.py     Integration-point-to-node prompt construction
src/pfem_cpfe/training/loss.py         Discrete equilibrium physics loss
src/pfem_cpfe/training/train.py        Single-sample patch and unlabeled multi-case training
src/pfem_cpfe/benchmark.py             Zero-start versus warm-start evaluation
tools/oxford_reference/                Source manifest, Fortran adapter, and trace generator
tests/                                 Unit, parity, integration, and benchmark-contract tests
```

---

### Task 1: Reproducible Python Package and Test Harness

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `src/pfem_cpfe/__init__.py`
- Create: `tests/test_package.py`
- Create: `README.md`

**Interfaces:**
- Consumes: none
- Produces: importable `pfem_cpfe` package and the standard command `python -m pytest`

- [x] **Step 1: Write the failing package-import test**

```python
def test_package_exposes_version():
    import pfem_cpfe
    assert pfem_cpfe.__version__ == "0.1.0"
```

- [x] **Step 2: Run the test and verify the package is absent**

Run: `python -m pytest tests/test_package.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'pfem_cpfe'`.

- [x] **Step 3: Add the minimal package and dependency contract**

Use Python `>=3.10,<3.12`. Declare runtime dependencies `numpy>=1.26,<3`, `scipy>=1.12,<2`, `torch>=2.4,<3`, `einops>=0.8,<1`; declare `pytest>=8,<10` in the `dev` extra. Configure pytest with `pythonpath = ["src"]` and `testpaths = ["tests"]`. Set `__version__ = "0.1.0"`.

Ignore `.venv/`, `.fortran-reference/`, `artifacts/`, `__pycache__/`, `.pytest_cache/`, compiled extension files, and generated build directories. Keep the checked-in Oxford NPZ trace fixture trackable.

- [x] **Step 4: Create and verify the environment**

Run: `uv venv --python 3.11 .venv`

Run: `uv pip install --python .venv/Scripts/python.exe -e ".[dev]"`

Run: `.venv/Scripts/python.exe -m pytest tests/test_package.py -v`

Expected: PASS; `.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"` prints `True` on the RTX 4060 Ti host.

- [x] **Step 5: Commit**

```bash
git add pyproject.toml .python-version .gitignore README.md src/pfem_cpfe/__init__.py tests/test_package.py
git commit -m "build: initialize PFEM-CPFE package"
```

### Task 2: Immutable Material-State and Trial/Commit Contracts

**Files:**
- Create: `src/pfem_cpfe/contracts.py`
- Create: `tests/test_contracts.py`

**Interfaces:**
- Consumes: NumPy
- Produces: `Case2Parameters.default()`, `MaterialState.initial(params)`, `TrialResult`, `commit_trial(state, trial)`

- [x] **Step 1: Write tests for CASE2 defaults and rollback-safe state**

```python
import numpy as np
from pfem_cpfe.contracts import Case2Parameters, MaterialState, TrialResult, commit_trial

def test_case2_defaults_match_oxford_source():
    p = Case2Parameters.default()
    assert p.nslip == 12
    assert p.k1 == 0.129 and p.k2 == 10.0
    assert p.burgers == 2.53e-4
    assert p.c11 == 289e3 and p.c12 == 124e3 and p.c44 == 83e3

def test_trial_does_not_mutate_committed_state():
    p = Case2Parameters.default()
    old = MaterialState.initial(p)
    trial_state = old.copied(ssd=old.ssd + 0.1)
    trial = TrialResult.zeros(trial_state)
    new = commit_trial(old, trial)
    assert np.all(old.ssd == 0.46)
    assert np.all(new.ssd == 0.56)
    assert not np.shares_memory(old.ssd, new.ssd)
```

- [x] **Step 2: Run tests and verify missing contracts**

Run: `.venv/Scripts/python.exe -m pytest tests/test_contracts.py -v`

Expected: FAIL on importing `pfem_cpfe.contracts`.

- [x] **Step 3: Implement the typed contracts**

Use frozen dataclasses for `Case2Parameters` and `MaterialState`. Store `fp`, `orientation`, `stress`, `ssd`, `crss`, `slip`, and `slip_rate` as copied `float64` arrays. Define `TrialResult(stress, tangent, state, converged, iterations, reason)` and reject commit when `converged` is false.

- [x] **Step 4: Run contract tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_contracts.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/pfem_cpfe/contracts.py tests/test_contracts.py
git commit -m "feat: define CPFE state and trial contracts"
```

### Task 3: FCC Slip Systems and Orientation Kinematics

**Files:**
- Create: `src/pfem_cpfe/crystal/__init__.py`
- Create: `src/pfem_cpfe/crystal/fcc.py`
- Create: `tests/crystal/test_fcc.py`

**Interfaces:**
- Consumes: `MaterialState.orientation`
- Produces: `fcc_slip_systems() -> tuple[ndarray, ndarray]`, `schmid_tensors(rotation)`, `rotation_from_rodrigues(vector)`, `project_rotation(matrix)`

- [x] **Step 1: Write crystallography tests**

```python
import numpy as np
from pfem_cpfe.crystal.fcc import fcc_slip_systems, schmid_tensors, project_rotation

def test_fcc_has_twelve_normalized_orthogonal_systems():
    directions, normals = fcc_slip_systems()
    assert directions.shape == normals.shape == (12, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    np.testing.assert_allclose(np.sum(directions * normals, axis=1), 0.0, atol=1e-14)

def test_orientation_projection_returns_proper_rotation():
    r = project_rotation(np.eye(3) + 1e-4 * np.ones((3, 3)))
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) > 0.0
    assert schmid_tensors(r).shape == (12, 3, 3)
```

- [x] **Step 2: Run tests and verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/crystal/test_fcc.py -v`

Expected: FAIL because `fcc.py` is absent.

- [x] **Step 3: Implement source-ordered FCC systems**

Transcribe the 12 systems in the same order used by Oxford phase 2. Normalize each direction and plane normal, form `s tensor m`, and use an SVD polar projection with determinant correction for rotations.

- [x] **Step 4: Run crystallography tests**

Run: `.venv/Scripts/python.exe -m pytest tests/crystal/test_fcc.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/pfem_cpfe/crystal tests/crystal
git commit -m "feat: add Oxford-ordered FCC crystallography"
```

### Task 4: Source-Locked CASE2 Local Laws

**Files:**
- Create: `src/pfem_cpfe/materials/__init__.py`
- Create: `src/pfem_cpfe/materials/case2_laws.py`
- Create: `tools/oxford_reference/source_manifest.py`
- Create: `tests/materials/test_case2_laws.py`
- Create: `tests/reference/test_oxford_manifest.py`

**Interfaces:**
- Consumes: `Case2Parameters`, slip-system arrays, archived Oxford source
- Produces: `double_exponential_slip`, `kocks_mecking_increment`, `total_and_forest_density`, `case2_slip_resistance`

- [x] **Step 1: Write equation-level tests**

```python
import numpy as np
from pfem_cpfe.contracts import Case2Parameters
from pfem_cpfe.materials.case2_laws import double_exponential_slip, kocks_mecking_increment

def test_kocks_mecking_increment_matches_oxford_equation():
    p = Case2Parameters.default()
    rho_for = np.full(12, 0.81)
    ssd = np.full(12, 0.46)
    rate = np.linspace(0.01, 0.12, 12)
    dt = 0.02
    expected = (p.k1 / p.burgers * np.sqrt(rho_for) - p.k2 * ssd) * np.abs(rate) * dt
    np.testing.assert_allclose(kocks_mecking_increment(ssd, rho_for, rate, dt, p), expected)

def test_slip_rate_is_odd_and_saturates_at_reference_rate():
    p = Case2Parameters.default()
    tau = np.array([0.5 * 206.0, -0.5 * 206.0, 1.1 * 206.0])
    rate = double_exponential_slip(tau, np.zeros(3), np.full(3, 206.0), 293.15, p)
    np.testing.assert_allclose(rate[0], -rate[1])
    assert rate[2] == p.gamma0
```

- [x] **Step 2: Add a manifest test against the supplied ZIP**

The manifest must read the archive path from `OXFORD_UMAT_ZIP`, extract text without modifying the archive, and assert these source facts: `case(2)`, `iphase=2`, `nslip=12`, `hardeningmodel=3`, `k1=0.129`, `k2=10`, and the Kocks-Mecking expression from `hardening.f`. Mark this test `reference` and skip with an explicit reason only when the environment variable is absent.

- [x] **Step 3: Run tests and verify missing laws**

Run: `.venv/Scripts/python.exe -m pytest tests/materials/test_case2_laws.py tests/reference/test_oxford_manifest.py -v`

Expected: the equation tests FAIL before implementation; the manifest test PASS when `OXFORD_UMAT_ZIP` points to the supplied archive.

- [x] **Step 4: Implement NumPy laws exactly as source branches**

Implement the `ratio >= 1` saturation branch and the `0 < ratio < 1` thermally activated branch from `slip.f:306-460`. Implement the Kocks-Mecking branch from `hardening.f:278-317` and density projections from `crss.f:308-358`. Reject nonpositive temperature, Burgers vector, or CRSS.

- [x] **Step 5: Run tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/materials/test_case2_laws.py tests/reference/test_oxford_manifest.py -v`

Expected: PASS.

```bash
git add src/pfem_cpfe/materials tools/oxford_reference/source_manifest.py tests/materials tests/reference/test_oxford_manifest.py
git commit -m "feat: port source-locked CASE2 local laws"
```

### Task 5: CPU Material-Point Integrator and Oxford Reference Traces

**Files:**
- Create: `src/pfem_cpfe/materials/case2_numpy.py`
- Create: `tools/oxford_reference/case2_reference.f90`
- Create: `tools/oxford_reference/case2_trace_driver.f90`
- Create: `tools/oxford_reference/build_reference.ps1`
- Create: `tools/oxford_reference/generate_traces.py`
- Create: `tests/fixtures/oxford_case2_trace.npz`
- Create: `tests/materials/test_case2_numpy.py`
- Create: `tests/reference/test_case2_parity.py`
- Create: `tests/reference/conftest.py`

**Interfaces:**
- Consumes: `Case2Parameters`, `MaterialState`, Task 3 crystallography, Task 4 laws
- Produces: `case2_step_numpy(F_n, F_np1, dt, temperature, params, state_n) -> TrialResult`

- [x] **Step 1: Write zero, elastic, rollback, and tangent tests**

```python
import numpy as np
from pfem_cpfe.contracts import Case2Parameters, MaterialState
from pfem_cpfe.materials.case2_numpy import case2_step_numpy

def test_identity_increment_preserves_state():
    p = Case2Parameters.default()
    s = MaterialState.initial(p)
    out = case2_step_numpy(np.eye(3), np.eye(3), 1e-3, 293.15, p, s)
    assert out.converged
    np.testing.assert_allclose(out.stress, 0.0, atol=1e-10)
    np.testing.assert_allclose(out.state.ssd, s.ssd)
    np.testing.assert_allclose(out.state.fp, s.fp)

def test_invalid_trial_leaves_input_state_unchanged():
    p = Case2Parameters.default()
    s = MaterialState.initial(p)
    before = s.ssd.copy()
    out = case2_step_numpy(np.eye(3), np.diag([-1.0, 1.0, 1.0]), 1e-3, 293.15, p, s)
    assert not out.converged
    np.testing.assert_array_equal(s.ssd, before)
```

- [x] **Step 2: Run tests and verify the integrator is absent**

Run: `.venv/Scripts/python.exe -m pytest tests/materials/test_case2_numpy.py -v`

Expected: FAIL on importing `case2_numpy`.

- [x] **Step 3: Implement the source-faithful semi-implicit update**

Use multiplicative kinematics `F = Fe @ Fp`, Oxford-ordered Schmid tensors, the double-exponential slip law, and the outer CRSS/SSD fixed-point loop. Return an uncommitted `TrialResult`; fail with a reason string for nonpositive determinant, nonfinite state, negative SSD/CRSS, or local iteration exhaustion. Compute the algorithmic `6 x 6` tangent using the same local Jacobian terms as the Oxford inner loop, and retain a finite-difference tangent test with relative tolerance `5e-4`.

- [x] **Step 4: Build the independent Fortran trace generator**

`case2_reference.f90` must contain a single public `case2_history` entry point and source-faithful copies of the CASE2 branches, with no Abaqus include or global database calls. The Python-extension route was replaced with a pure Fortran trace executable because the available MinGW 5.3 Fortran compiler cannot link reliably against the current Python/UCRT C ABI. `build_reference.ps1` shall compile the standalone driver and then package its binary output:

```powershell
conda create -p .fortran-reference -c conda-forge python=3.11 numpy=1.26 setuptools=68.2.2 m2w64-gcc-fortran -y
tools\oxford_reference\build_reference.ps1
```

Generate deterministic tension, compression, shear, unload, and reload histories into `tests/fixtures/oxford_case2_trace.npz`, including every step's `F`, stress, `Fp`, orientation, 12 slips, 12 SSDs, and 12 CRSS values.

- [x] **Step 5: Write and run parity tests**

Define `tests/reference/conftest.py::oxford_trace` to load the named arrays from the NPZ fixture. Define `replay_numpy_backend` in `test_case2_parity.py` to start from `MaterialState.initial`, call `case2_step_numpy` sequentially for every recorded deformation gradient, assert convergence at every step, and stack its outputs.

```python
def test_numpy_backend_matches_oxford_history(oxford_trace):
    result = replay_numpy_backend(oxford_trace)
    np.testing.assert_allclose(result["stress"], oxford_trace["stress"], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(result["ssd"], oxford_trace["ssd"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(result["fp"], oxford_trace["fp"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result["crss"], oxford_trace["crss"], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(result["slip"], oxford_trace["slip"], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(result["orientation"], oxford_trace["orientation"], rtol=1e-7, atol=1e-9)
```

Run: `.venv/Scripts/python.exe -m pytest tests/materials/test_case2_numpy.py tests/reference/test_case2_parity.py -v`

Expected: PASS. If the compiler cannot be installed, record the failed build command and stop parity sign-off rather than weakening tolerances.

- [x] **Step 6: Commit**

```bash
git add src/pfem_cpfe/materials/case2_numpy.py tools/oxford_reference tests/fixtures/oxford_case2_trace.npz tests/materials/test_case2_numpy.py tests/reference/conftest.py tests/reference/test_case2_parity.py
git commit -m "feat: add verified CASE2 material-point integrator"
```

### Task 6: Differentiable Batched CASE2 Backend

**Files:**
- Create: `src/pfem_cpfe/materials/case2_torch.py`
- Create: `tests/materials/conftest.py`
- Create: `tests/materials/test_case2_torch.py`

**Interfaces:**
- Consumes: Task 5 update semantics and trace fixtures
- Produces: `case2_step_torch(F_n, F_np1, dt, temperature, params, state_n) -> TorchTrialResult`

- [x] **Step 1: Write NumPy parity and gradient tests**

In `tests/materials/conftest.py`, define `case2_batch` from a deterministic two-point batch: identity `F_n`, `F_np1[0,0]=1.001`, `F_np1[0,1]=0.002`, CASE2 parameters, and copied initial states. The fixture exposes matching Torch inputs, NumPy results, a double-precision deformation tensor, and a scalar stress closure for `gradcheck`. Parametrize parity over CPU and CUDA, skipping the CUDA case explicitly when unavailable.

```python
def test_torch_backend_matches_numpy_backend(case2_batch):
    torch_out = case2_step_torch(**case2_batch.torch_inputs)
    numpy_out = case2_batch.numpy_results
    np.testing.assert_allclose(torch_out.stress.detach().cpu(), numpy_out.stress, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(torch_out.state.ssd.detach().cpu(), numpy_out.state.ssd, rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(torch_out.state.fp.detach().cpu(), numpy_out.state.fp, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(torch_out.state.orientation.detach().cpu(), numpy_out.state.orientation, rtol=2e-6, atol=2e-7)

def test_stress_is_differentiable_with_respect_to_deformation(case2_batch):
    F = case2_batch.F_np1_double.requires_grad_(True)
    assert torch.autograd.gradcheck(case2_batch.stress_function, (F,), eps=1e-6, atol=2e-4, rtol=2e-3)
```

- [x] **Step 2: Run tests and verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/materials/test_case2_torch.py -v`

Expected: FAIL because `case2_torch` is absent.

- [x] **Step 3: Implement tensorized local integration**

Use `torch.float64` for parity tests and permit `float32` for training. Avoid NumPy conversions and in-place writes on tensors needed by autograd. Batch all integration points; return convergence masks and reason codes without silently dropping failed points.

- [x] **Step 4: Run parity and gradient tests on CPU and CUDA**

Run: `.venv/Scripts/python.exe -m pytest tests/materials/test_case2_torch.py -v`

Expected: CPU cases PASS and CUDA cases PASS on this host; CUDA cases are explicitly SKIPPED on hosts without CUDA.

- [x] **Step 5: Commit**

```bash
git add src/pfem_cpfe/materials/case2_torch.py tests/materials/conftest.py tests/materials/test_case2_torch.py
git commit -m "feat: add differentiable CASE2 backend"
```

### Task 7: C3D8 Full-Integration Element

**Files:**
- Create: `src/pfem_cpfe/elements/__init__.py`
- Create: `src/pfem_cpfe/elements/c3d8.py`
- Create: `tests/elements/conftest.py`
- Create: `tests/elements/test_c3d8.py`

**Interfaces:**
- Consumes: Task 5 material backend
- Produces: `c3d8_quadrature()`, `c3d8_shape(xi)`, `c3d8_kinematics(Xe, ue)`, `c3d8_response(Xe, ue, states_n, ...)`

- [x] **Step 1: Write partition, Jacobian, rigid-motion, and constant-strain tests**

In `tests/elements/conftest.py`, define `unit_cube` with the Oxford/Abaqus C3D8 node order `[(0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)]`.

```python
def test_shape_partition_and_zero_gradient_sum():
    N, dN = c3d8_shape(np.array([0.13, -0.21, 0.37]))
    np.testing.assert_allclose(N.sum(), 1.0)
    np.testing.assert_allclose(dN.sum(axis=0), 0.0, atol=1e-14)

def test_unit_cube_affine_displacement_has_constant_F(unit_cube):
    A = np.array([[0.01, 0.002, 0.0], [0.0, -0.003, 0.0], [0.0, 0.0, 0.004]])
    F = c3d8_kinematics(unit_cube.nodes, unit_cube.nodes @ A.T).F
    np.testing.assert_allclose(F, np.eye(3) + A, atol=1e-13)
```

- [x] **Step 2: Run tests and verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/elements/test_c3d8.py -v`

Expected: FAIL because `c3d8.py` is absent.

- [x] **Step 3: Implement 2 x 2 x 2 integration and element assembly**

Return eight deformation gradients, integration weights, element internal force, `24 x 24` tangent, and eight trial states. Reject inverted reference or current Jacobians. Use finite differences to compare the assembled element tangent against the internal-force derivative.

- [x] **Step 4: Run element tests**

Run: `.venv/Scripts/python.exe -m pytest tests/elements/test_c3d8.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/pfem_cpfe/elements tests/elements
git commit -m "feat: implement C3D8 crystal-plasticity element"
```

### Task 8: Global Assembly, Newton Solve, and Atomic History Commit

**Files:**
- Create: `src/pfem_cpfe/solver/__init__.py`
- Create: `src/pfem_cpfe/solver/assembly.py`
- Create: `src/pfem_cpfe/solver/newton.py`
- Create: `tests/solver/conftest.py`
- Create: `tests/solver/test_newton.py`

**Interfaces:**
- Consumes: mesh arrays, element response, committed integration-point states
- Produces: `assemble_trial(problem, u_trial, state_n) -> GlobalTrial`, `solve_increment(problem, state_n, u0, controls=None) -> IncrementResult`

- [x] **Step 1: Write convergence, equality, cutback, and rollback tests**

In `tests/solver/conftest.py`, define `single_crystal_problem` with a copied initial `MaterialState`, a one-element mesh, one small displacement-controlled increment, and both zero and deterministic admissible initial guesses.

```python
def test_zero_and_perturbed_starts_converge_to_same_single_element_solution(single_crystal_problem):
    state_n = single_crystal_problem.states.copied()
    zero = solve_increment(single_crystal_problem, state_n=state_n, u0=np.zeros(single_crystal_problem.ndof))
    perturbed = solve_increment(single_crystal_problem, state_n=state_n, u0=single_crystal_problem.small_admissible_guess)
    assert zero.converged and perturbed.converged
    np.testing.assert_allclose(zero.u, perturbed.u, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(zero.states.ssd, perturbed.states.ssd, rtol=1e-8, atol=1e-10)

def test_failed_increment_preserves_committed_history(single_crystal_problem):
    old = single_crystal_problem.states.copied()
    out = solve_increment(
        single_crystal_problem.with_impossible_increment(),
        state_n=old,
        u0=np.zeros(single_crystal_problem.ndof),
    )
    assert not out.converged
    np.testing.assert_array_equal(old.ssd, single_crystal_problem.states.ssd)
```

- [x] **Step 2: Run tests and verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/solver/test_newton.py -v`

Expected: FAIL because solver modules are absent.

- [x] **Step 3: Implement sparse assembly and safeguarded Newton**

Assemble a SciPy CSR tangent, eliminate prescribed DOFs, solve free increments with `scipy.sparse.linalg.spsolve`, and require both normalized residual and normalized correction tolerances. Backtrack line search over factors `1, 1/2, ..., 1/128`. On failure, recursively bisect the load increment up to `max_cutbacks`; commit integration-point states only after convergence.

- [x] **Step 4: Run solver tests**

Run: `.venv/Scripts/python.exe -m pytest tests/solver/test_newton.py -v`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/pfem_cpfe/solver tests/solver
git commit -m "feat: add standalone incremental CPFE solver"
```

### Task 9: Deterministic Single-Crystal and Eight-Grain Problems

**Files:**
- Create: `src/pfem_cpfe/mesh/__init__.py`
- Create: `src/pfem_cpfe/mesh/voxel.py`
- Create: `src/pfem_cpfe/io/__init__.py`
- Create: `src/pfem_cpfe/io/results.py`
- Create: `src/pfem_cpfe/run_case.py`
- Create: `tests/mesh/test_voxel.py`
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_eight_grain.py`

**Interfaces:**
- Consumes: contracts and Task 8 solver
- Produces: `single_element_problem(seed)`, `eight_grain_problem(seed=20260915, cells=(4,4,4))`, `save_result(path, result)`, `load_result(path)`

- [x] **Step 1: Write deterministic mesh and result-schema tests**

In `tests/integration/conftest.py`, define `solved_eight_grain` as a deterministic, finite `IncrementResult` with the exact full-field shapes of the `4 x 4 x 4` problem. This keeps the I/O schema test fast; the command-line baseline below performs the actual nonlinear solve.

```python
def test_eight_grain_mesh_is_reproducible():
    a = eight_grain_problem(seed=20260915)
    b = eight_grain_problem(seed=20260915)
    assert a.elements.shape == (64, 8)
    assert set(np.unique(a.grain_ids)) == set(range(8))
    np.testing.assert_array_equal(a.grain_ids, b.grain_ids)
    np.testing.assert_allclose(a.orientations, b.orientations)

def test_result_contains_required_full_fields(tmp_path, solved_eight_grain):
    path = tmp_path / "result.npz"
    save_result(path, solved_eight_grain)
    data = load_result(path)
    assert {"u", "stress", "strain", "slip", "orientation", "fp", "ssd", "crss", "residual", "timing"} <= set(data)
```

- [x] **Step 2: Run tests and verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/mesh/test_voxel.py tests/integration/test_eight_grain.py -v`

Expected: FAIL because mesh and I/O modules are absent.

- [x] **Step 3: Implement seeded Voronoi assignment and displacement control**

Create the unit-cube node lattice and 64 C3D8 connectivities. Draw eight interior seed points and eight uniform random rotations from a fixed NumPy generator, assign each element centroid to its nearest seed, and retry the deterministic seed sequence until every grain is nonempty. Apply the approved bottom/top axial displacement constraints and minimal rigid-body constraints.

Implement `save_result` as a stable compressed NPZ for arrays plus a sibling JSON file for schema version, convergence, residual history, iteration counts, and timing. Implement `pfem_cpfe.run_case` as the zero-start command that builds a named problem, advances every increment sequentially, commits history only on convergence, and writes both files.

- [x] **Step 4: Solve and persist the baseline**

Run: `.venv/Scripts/python.exe -m pfem_cpfe.run_case --case eight-grain --increments 20 --output artifacts/baseline-eight-grain`

Expected: final convergence is true; all required fields are finite; no SSD is negative; the JSON contains per-increment Newton iterations and timing categories.

- [x] **Step 5: Run tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/mesh/test_voxel.py tests/integration/test_eight_grain.py -v`

Expected: PASS.

```bash
git add src/pfem_cpfe/mesh src/pfem_cpfe/io src/pfem_cpfe/run_case.py tests/mesh tests/integration
git commit -m "feat: add deterministic eight-grain benchmark"
```

### Task 10: PFEM Transolver Backbone and Node Prompts

**Files:**
- Create: `src/pfem_cpfe/operator/__init__.py`
- Create: `src/pfem_cpfe/operator/transolver.py`
- Create: `src/pfem_cpfe/operator/features.py`
- Create: `tests/operator/conftest.py`
- Create: `tests/operator/test_transolver.py`
- Create: `tests/operator/test_features.py`

**Interfaces:**
- Consumes: PFEM-main physics-attention design, mesh, committed integration-point state
- Produces: `TransolverCPFE`, `FeatureLayout`, `build_node_features(problem, state_n, load_step)`, `enforce_displacement_bc(raw_du, problem)`

- [ ] **Step 1: Write shape, permutation, projection, and boundary tests**

In `tests/operator/conftest.py`, define `node_batch` from two padded deterministic meshes, and define `problem` plus `raw_du` so fixed and prescribed DOFs are known exactly.

```python
def test_transolver_outputs_only_nodal_displacement(node_batch):
    model = TransolverCPFE(feature_dim=node_batch.features.shape[-1], hidden=64, layers=2, heads=4, slices=16)
    out = model(node_batch.xyz, node_batch.features, node_batch.mask)
    assert out.shape == (*node_batch.xyz.shape[:2], 3)

def test_boundary_projection_is_exact(problem, raw_du):
    du = enforce_displacement_bc(raw_du, problem)
    np.testing.assert_allclose(du[problem.fixed_dofs], problem.prescribed_increment[problem.fixed_dofs])
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/operator -v`

Expected: FAIL because operator modules are absent.

- [ ] **Step 3: Port the PFEM physics-attention structure**

Implement point-to-token slicing, token self-attention, and token-to-point deslicing with complexity `O(N + S^2)`. Use `torch.nn.init.trunc_normal_` so the package does not require `timm`. Support masks for padded batches and CPU/CUDA without embedded `.cuda()` calls.

- [ ] **Step 4: Implement the fixed prompt layout**

Include normalized coordinates; prescribed-DOF mask and values; projected orientation matrix; projected `Fp`; 12 accumulated slips; 12 normalized SSD values; 12 normalized CRSS values; temperature; `dt`; load fraction; and CASE2 elastic, slip, and hardening parameters. Volume-average integration-point features to adjacent nodes only for prompting; keep committed physical state unchanged.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/Scripts/python.exe -m pytest tests/operator -v`

Expected: PASS, including a node-permutation equivariance check.

```bash
git add src/pfem_cpfe/operator tests/operator
git commit -m "feat: adapt PFEM Transolver to CPFE prompts"
```

### Task 11: Unlabeled Discrete-Equilibrium Physics Training

**Files:**
- Create: `src/pfem_cpfe/training/__init__.py`
- Create: `src/pfem_cpfe/training/loss.py`
- Create: `src/pfem_cpfe/training/train.py`
- Create: `tests/training/conftest.py`
- Create: `tests/training/test_physics_loss.py`
- Create: `tests/training/test_patch_training.py`

**Interfaces:**
- Consumes: Task 6 differentiable backend, Task 7 C3D8 kinematics, Task 10 operator
- Produces: `relative_equilibrium_loss(model, batch)`, `train_patch(config)`, checkpoint plus metrics JSON

- [ ] **Step 1: Write loss-contract tests**

In `tests/training/conftest.py`, define `unlabeled_case` with geometry, loads, boundary data, material parameters, and committed state only; define `equilibrated_case` from a zero-load rigidly constrained problem; and define `identity_predictor` to return its admissible zero increment. Define a small deterministic `model` fixture separately.

```python
def test_physics_loss_uses_no_solution_labels(unlabeled_case, model):
    assert "u_exact" not in unlabeled_case
    loss, metrics = relative_equilibrium_loss(model, unlabeled_case)
    assert loss.requires_grad
    assert set(metrics) == {"relative_residual", "state_penalty", "invalid_fraction"}

def test_exact_equilibrium_field_has_near_zero_loss(equilibrated_case, identity_predictor):
    loss, _ = relative_equilibrium_loss(identity_predictor, equilibrated_case)
    assert loss.item() < 1e-10
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_physics_loss.py -v`

Expected: FAIL because training modules are absent.

- [ ] **Step 3: Implement explicit FE residual loss**

Compute `F` at eight Gauss points from predicted nodal increments with C3D8 shape gradients, call the batched Torch CASE2 backend, assemble free-DOF internal minus external force, and divide its squared norm by `F_scale^2 + epsilon`. Apply displacement constraints exactly before kinematics. Add finite penalties only for nonpositive determinants, negative SSD, invalid rotations, and failed local points.

- [ ] **Step 4: Prove the one-sample PFEM patch test**

Train one single-crystal sample with fixed seed and save the initial/final residual. The test shall use a small model and assert at least a 100-fold residual reduction without accessing a reference displacement field.

Run: `.venv/Scripts/python.exe -m pytest tests/training/test_physics_loss.py tests/training/test_patch_training.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pfem_cpfe/training tests/training
git commit -m "feat: train Transolver from CPFE equilibrium"
```

### Task 12: Warm-Start Accuracy and End-to-End Timing Benchmark

**Files:**
- Create: `src/pfem_cpfe/benchmark.py`
- Create: `tests/benchmark/conftest.py`
- Create: `tests/benchmark/test_warm_start.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: zero-start solver, trained checkpoint, deterministic eight-grain problem
- Produces: `run_benchmark(config) -> BenchmarkReport`, `artifacts/<run>/report.json`, full-field NPZ files

- [ ] **Step 1: Write benchmark equivalence and accounting tests**

In `tests/benchmark/conftest.py`, define `benchmark_report` by running a two-increment single-element benchmark with an exact admissible-guess stub. It exercises both paths quickly while the milestone command below uses the trained operator and eight-grain problem.

```python
def test_warm_and_zero_start_reach_same_solution(benchmark_report):
    assert benchmark_report.zero.converged
    assert benchmark_report.warm.converged
    assert benchmark_report.field_error("u") < 1e-7
    assert benchmark_report.field_error("stress") < 1e-6
    assert benchmark_report.field_error("ssd") < 1e-6
    assert benchmark_report.field_error("slip") < 1e-6
    assert benchmark_report.field_error("orientation") < 1e-6
    assert benchmark_report.field_error("fp") < 1e-6
    assert benchmark_report.field_error("crss") < 1e-6

def test_total_time_includes_inference_and_newton(benchmark_report):
    expected = benchmark_report.warm.inference_seconds + benchmark_report.warm.newton_seconds
    assert abs(benchmark_report.warm.total_seconds - expected) < 1e-9
```

- [ ] **Step 2: Run tests and verify the report API is absent**

Run: `.venv/Scripts/python.exe -m pytest tests/benchmark/test_warm_start.py -v`

Expected: FAIL because the benchmark report is absent.

- [ ] **Step 3: Implement fair benchmark execution**

Warm the GPU once, synchronize CUDA around timings, pin CPU thread counts, use identical increments and nonlinear/linear tolerances, and run at least five repetitions after warm-up. Record median and individual times for feature construction, operator inference, material integration, assembly, linear solves, Newton total, and end-to-end total. Record every fallback from an invalid operator guess.

- [ ] **Step 4: Run the milestone benchmark**

Run: `.venv/Scripts/python.exe -m pfem_cpfe.benchmark --case eight-grain --checkpoint artifacts/patch/model.pt --increments 20 --repetitions 5 --output artifacts/milestone-1`

Expected: both modes converge to the specified tolerances and the report contains actual iteration and time ratios. Treat a ratio below one as an experimental failure, not as a software-test failure or a reason to hide the result.

- [ ] **Step 5: Run the full verification suite**

Run: `.venv/Scripts/python.exe -m pytest -v`

Expected: PASS, including Oxford trace parity, Torch gradient checks, C3D8 patch tests, rollback tests, single-sample physics training, and warm/zero final-field equivalence.

- [ ] **Step 6: Document exact reproduction commands and commit**

Add README commands for environment setup, Oxford trace generation, zero-start solve, patch training, warm-start solve, and report interpretation.

```bash
git add src/pfem_cpfe/benchmark.py tests/benchmark README.md
git commit -m "feat: benchmark PFEM warm starts for CPFE"
```

## Follow-On Plans

After Milestone 1 passes, create separate reviewed plans for: Oxford `gndmodel=1` with manufactured-Curl and mesh-convergence tests; periodic RVE constraints; broader unlabeled pretraining; frozen-weight material-parameter evaluation; and frozen-weight transfer to a second hardening law. These plans must preserve the same final-solution and end-to-end timing standards.
