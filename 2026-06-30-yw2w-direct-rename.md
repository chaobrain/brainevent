# yw2w Direct Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `yw2w` the only API, module, CUDA symbol, test, and documentation name for the per-synapse `y * w -> w-shaped output` operators.

**Architecture:** This is a direct rename of the existing `yw2y` implementation to `yw2w`. The CSR CUDA implementation, CSR/CSC Python wrappers, FCN pure-JAX wrapper, package exports, high-level sparse methods, tests, docs, and type-check baseline all move to the new spelling with no compatibility shims or aliases.

**Tech Stack:** Python, JAX, brainunit, pytest, CUDA FFI kernels, Sphinx docs, mypy baseline, conda environment `brainevent-jax10`.

---

## Non-Negotiable Requirements

- No legacy `yw2y` public API symbols remain in `brainevent`, `brainevent._csr`, or `brainevent._fcn`.
- No legacy `yw2y` modules remain in CSR or FCN.
- No compatibility alias assignments remain in implementation modules.
- No wrapper layer translates from `yw2y` to `yw2w`.
- `CSR.yw_to_w`, `CSR.yw_to_w_transposed`, `CSC.yw_to_w`, and `CSC.yw_to_w_transposed` call `csrmv_yw2w` directly.
- Fixed-connection high-level APIs call `fcnmv_yw2w` directly.
- CUDA FFI registration names and call targets use `csrmv_yw2w`.
- Do not create git commits unless the user explicitly asks.

## Current State

The repository still uses the old spelling. These files exist today and must be renamed or updated:

- Rename: `brainevent/_csr/yw2y.py` -> `brainevent/_csr/yw2w.py`
- Rename: `brainevent/_csr/yw2y.cu` -> `brainevent/_csr/yw2w.cu`
- Rename: `brainevent/_csr/yw2y_test.py` -> `brainevent/_csr/yw2w_test.py`
- Rename: `brainevent/_fcn/yw2y.py` -> `brainevent/_fcn/yw2w.py`
- Rename: `brainevent/_fcn/yw2y_test.py` -> `brainevent/_fcn/yw2w_test.py`
- Modify: `brainevent/_csr/__init__.py`
- Modify: `brainevent/_fcn/__init__.py`
- Modify: `brainevent/__init__.py`
- Modify: `brainevent/_csr/main.py`
- Modify: `brainevent/_fcn/main.py`
- Modify: `brainevent/_csr/cuda_int64_indptr_test.py`
- Modify: `brainevent/_fcn/main_test.py`
- Modify: `docs/reference/apis/operations.rst`
- Modify: `mypy-baseline.txt`
- Review only unless release notes are required: `changelog.md`

## Task 1: Rename Source and Test Files

- [ ] Rename the CSR implementation files.

```bash
mv brainevent/_csr/yw2y.py brainevent/_csr/yw2w.py
mv brainevent/_csr/yw2y.cu brainevent/_csr/yw2w.cu
mv brainevent/_csr/yw2y_test.py brainevent/_csr/yw2w_test.py
```

- [ ] Rename the FCN implementation and test files.

```bash
mv brainevent/_fcn/yw2y.py brainevent/_fcn/yw2w.py
mv brainevent/_fcn/yw2y_test.py brainevent/_fcn/yw2w_test.py
```

- [ ] Confirm the old module files are gone and the new module files exist.

```bash
test ! -e brainevent/_csr/yw2y.py
test ! -e brainevent/_csr/yw2y.cu
test ! -e brainevent/_csr/yw2y_test.py
test ! -e brainevent/_fcn/yw2y.py
test ! -e brainevent/_fcn/yw2y_test.py
test -e brainevent/_csr/yw2w.py
test -e brainevent/_csr/yw2w.cu
test -e brainevent/_csr/yw2w_test.py
test -e brainevent/_fcn/yw2w.py
test -e brainevent/_fcn/yw2w_test.py
```

Expected: all commands exit successfully with no output.

## Task 2: Rename CSR Python API and CUDA Wiring

**Files:**
- Modify: `brainevent/_csr/yw2w.py`
- Modify: `brainevent/_csr/yw2w.cu`

- [ ] In `brainevent/_csr/yw2w.py`, replace every `yw2y` token with `yw2w`, preserving case style for constants and class names where applicable.

Required public API:

```python
__all__ = [
    'csrmv_yw2w',
    'cscmv_yw2w',
    'csrmv_yw2w_p',
]
```

Required function and primitive names:

```python
def csrmv_yw2w(...):
    ...

def cscmv_yw2w(...):
    ...

def csrmv_yw2w_p_call(...):
    ...

csrmv_yw2w_p = XLACustomKernel(
    'csrmv_yw2w',
    ...
)
```

- [ ] Update the CUDA loader in `brainevent/_csr/yw2w.py` to load the renamed CUDA file and register the renamed target.

Required code shape:

```python
load_cuda_file(
    Path(__file__).parent.joinpath('yw2w.cu'),
    name='csrmv_yw2w',
)

kernel_name = f'csrmv_yw2w.csrmv_yw2w_nt_auto{wt_sfx}'
```

- [ ] Rename private helper functions in `brainevent/_csr/yw2w.py` so no private `yw2y` identifiers remain.

Required helper name pattern:

```python
def _csrmv_yw2w_numba_kernels(...):
def _csrmv_yw2w_cuda_kernel(...):
def _csrmv_yw2w_jax_kernel(...):
def _csrmv_yw2w_jvp_y(...):
def _csrmv_yw2w_jvp_w(...):
def _csrmv_yw2w_transpose_rule(...):
def _csrmv_yw2w_benchmark_data(...):
```

- [ ] In `brainevent/_csr/yw2w.cu`, replace CUDA macro, kernel, FFI function, and `// @BE` registration names from `YW2Y` / `yw2y` / `csrmv_yw2y` to `YW2W` / `yw2w` / `csrmv_yw2w`.

Required exported target examples:

```cpp
// @BE csrmv_yw2w_nt_row_thread_f32
// @BE csrmv_yw2w_nt_row_warp_f32
// @BE csrmv_yw2w_nt_nz_thread_f32
// @BE csrmv_yw2w_nt_auto_f32
// @BE csrmv_yw2w_nt_auto_f64
// @BE csrmv_yw2w_nt_auto_f16
// @BE csrmv_yw2w_nt_auto_bf16
```

- [ ] Run a focused source check for the renamed CSR implementation.

```bash
rg -n -S "yw2y|YW2Y|Yw2y" brainevent/_csr/yw2w.py brainevent/_csr/yw2w.cu
```

Expected: no output.

## Task 3: Rename FCN Python API

**Files:**
- Modify: `brainevent/_fcn/yw2w.py`

- [ ] In `brainevent/_fcn/yw2w.py`, replace every `yw2y` token with `yw2w`.

Required public API:

```python
__all__ = [
    'fcnmv_yw2w',
]
```

Required function name:

```python
def fcnmv_yw2w(...):
    ...
```

- [ ] Update FCN docstrings and examples to refer to `fcnmv_yw2w` and `csrmv_yw2w`.

- [ ] Run a focused source check for the renamed FCN implementation.

```bash
rg -n -S "yw2y|YW2Y|Yw2y" brainevent/_fcn/yw2w.py
```

Expected: no output.

## Task 4: Update Package Exports and High-Level Calls

**Files:**
- Modify: `brainevent/_csr/__init__.py`
- Modify: `brainevent/_fcn/__init__.py`
- Modify: `brainevent/__init__.py`
- Modify: `brainevent/_csr/main.py`
- Modify: `brainevent/_fcn/main.py`

- [ ] Update CSR package exports.

Required import:

```python
from .yw2w import csrmv_yw2w, cscmv_yw2w, csrmv_yw2w_p
```

Required `__all__` entries:

```python
'csrmv_yw2w', 'cscmv_yw2w', 'csrmv_yw2w_p',
```

- [ ] Update FCN package exports.

Required import:

```python
from .yw2w import fcnmv_yw2w
```

Required `__all__` entry:

```python
'fcnmv_yw2w',
```

- [ ] Update top-level exports in `brainevent/__init__.py` so imports and `__all__` expose only `yw2w` names.

Required CSR import entries:

```python
csrmv_yw2w, cscmv_yw2w, csrmv_yw2w_p,
```

Required FCN import entry:

```python
fcnmv_yw2w,
```

Required `__all__` entries:

```python
'csrmv_yw2w', 'cscmv_yw2w', 'csrmv_yw2w_p',
'fcnmv_yw2w',
```

- [ ] Update CSR/CSC high-level calls in `brainevent/_csr/main.py`.

Required import:

```python
from .yw2w import csrmv_yw2w
```

Required direct calls:

```python
return csrmv_yw2w(y_dim_arr, w_dim_arr, self.indices, self.indptr,
                  shape=self.shape, transpose=False, backend=self.backend)

return csrmv_yw2w(y_dim_arr, w_dim_arr, self.indices, self.indptr,
                  shape=self.shape, transpose=True, backend=self.backend)

return csrmv_yw2w(y_dim_arr, w_dim_arr, self.indices, self.indptr,
                  shape=self.shape[::-1], transpose=True, backend=self.backend)

return csrmv_yw2w(y_dim_arr, w_dim_arr, self.indices, self.indptr,
                  shape=self.shape[::-1], transpose=False, backend=self.backend)
```

- [ ] Update fixed-connection high-level calls in `brainevent/_fcn/main.py`.

Required import:

```python
from .yw2w import fcnmv_yw2w
```

Required direct calls:

```python
return fcnmv_yw2w(w, self.indices, y_dim_arr, shape=self._a_shape,
                  transpose=self._ell_transpose(False))

return fcnmv_yw2w(w, self.indices, y_dim_arr, shape=self._a_shape,
                  transpose=self._ell_transpose(True))
```

- [ ] Run a focused source check for exports and high-level calls.

```bash
rg -n -S "yw2y|YW2Y|Yw2y" \
  brainevent/_csr/__init__.py \
  brainevent/_fcn/__init__.py \
  brainevent/__init__.py \
  brainevent/_csr/main.py \
  brainevent/_fcn/main.py
```

Expected: no output.

## Task 5: Rename Tests and Test Expectations

**Files:**
- Modify: `brainevent/_csr/yw2w_test.py`
- Modify: `brainevent/_fcn/yw2w_test.py`
- Modify: `brainevent/_csr/cuda_int64_indptr_test.py`
- Modify: `brainevent/_fcn/main_test.py`

- [ ] In `brainevent/_csr/yw2w_test.py`, update imports, constants, class names, comments, skip reasons, and calls to use `yw2w`.

Required import:

```python
from brainevent._csr.yw2w import csrmv_yw2w, cscmv_yw2w, csrmv_yw2w_p
```

Required constant:

```python
CSRMV_YW2W_IMPLEMENTATIONS = tuple(csrmv_yw2w_p.available_backends(platform))
```

- [ ] In `brainevent/_fcn/yw2w_test.py`, update imports, test names, comments, and calls to use `fcnmv_yw2w`.

Required imports:

```python
from brainevent import fcnmv_yw2w
from brainevent._fcn.yw2w import fcnmv_yw2w as fcnmv_yw2w_module
```

- [ ] In `brainevent/_csr/cuda_int64_indptr_test.py`, update the CSR CUDA generator test to import the renamed module and assert renamed CUDA targets.

Required imports:

```python
import brainevent._csr.yw2w as yw2w_mod
from brainevent._csr.yw2w import csrmv_yw2w
```

Required expected load name:

```python
'csrmv_yw2w',
```

Required expected FFI call:

```python
'csrmv_yw2w.csrmv_yw2w_nt_auto_f32',
```

- [ ] Rename the int64-indptr GPU test function and calls.

Required function name:

```python
def test_yw2w_cuda_accepts_int64_indptr():
```

Required calls:

```python
got = csrmv_yw2w(y, weights, indices, indptr64, shape=(2, 3), backend='cuda_raw')
expected = csrmv_yw2w(y, weights, indices, indptr32, shape=(2, 3), backend='jax_raw')
```

- [ ] In `brainevent/_fcn/main_test.py`, rename `Test_Yw2y` to `Test_Yw2w` and update any comments containing `yw2y`.

- [ ] Run a focused source check for tests.

```bash
rg -n -S "yw2y|YW2Y|Yw2y" \
  brainevent/_csr/yw2w_test.py \
  brainevent/_fcn/yw2w_test.py \
  brainevent/_csr/cuda_int64_indptr_test.py \
  brainevent/_fcn/main_test.py
```

Expected: no output.

## Task 6: Update Documentation and Static Baselines

**Files:**
- Modify: `docs/reference/apis/operations.rst`
- Modify: `mypy-baseline.txt`
- Review: `changelog.md`

- [ ] Update API docs autosummary entries.

Required entries:

```rst
   csrmv_yw2w
   csrmv_yw2w_p
```

- [ ] Update `mypy-baseline.txt` paths from `brainevent/_csr/yw2y.py` to `brainevent/_csr/yw2w.py`. Preserve the existing error text unless running mypy proves the baseline has changed for another reason.

- [ ] Review `changelog.md`. Keep historical release notes unchanged unless the project policy requires unreleased rename notes in this file. Historical entries may legitimately mention `csrmv_yw2y` because they describe past releases.

- [ ] Run a focused source check for docs and baseline, excluding historical changelog entries.

```bash
rg -n -S "yw2y|YW2Y|Yw2y" docs mypy-baseline.txt -g '!docs/tutorials/**/*.ipynb'
```

Expected: no output.

## Task 7: Interface Absence and Presence Verification

- [ ] Run this interface check in the required conda environment.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate brainevent-jax10
python - <<'PY'
import importlib
import brainevent
import brainevent._csr as csr
import brainevent._fcn as fcn

old = 'yw' + '2' + 'y'
new = 'yw' + '2' + 'w'

for module_name in ('brainevent._csr.' + old, 'brainevent._fcn.' + old):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass
    else:
        raise SystemExit(f'{module_name} still imports')

old_attrs = [
    (brainevent, 'csrmv_' + old),
    (brainevent, 'cscmv_' + old),
    (brainevent, 'fcnmv_' + old),
    (csr, 'csrmv_' + old),
    (csr, 'cscmv_' + old),
    (fcn, 'fcnmv_' + old),
]
for module, attr in old_attrs:
    if hasattr(module, attr):
        raise SystemExit(f'{module.__name__}.{attr} still exists')

new_attrs = [
    (brainevent, 'csrmv_' + new),
    (brainevent, 'cscmv_' + new),
    (brainevent, 'csrmv_' + new + '_p'),
    (brainevent, 'fcnmv_' + new),
    (csr, 'csrmv_' + new),
    (csr, 'cscmv_' + new),
    (csr, 'csrmv_' + new + '_p'),
    (fcn, 'fcnmv_' + new),
]
for module, attr in new_attrs:
    if not hasattr(module, attr):
        raise SystemExit(f'{module.__name__}.{attr} is missing')

print('yw2w interface present; yw2y interface absent')
PY
```

Expected: `yw2w interface present; yw2y interface absent`.

## Task 8: Run Focused Tests

- [ ] Run the focused `yw2w` tests.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate brainevent-jax10
python -m pytest brainevent/_csr/yw2w_test.py brainevent/_fcn/yw2w_test.py -q
```

Expected: all focused tests pass.

- [ ] Run tests covering high-level `yw_to_w` dispatch and CUDA int64 generator expectations.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate brainevent-jax10
python -m pytest \
  brainevent/_csr/main_test.py \
  brainevent/_fcn/main_test.py \
  brainevent/_csr/cuda_int64_indptr_test.py \
  -q
```

Expected: all selected tests pass or GPU-only tests skip on non-GPU machines.

## Task 9: Residual Scan and Final Review

- [ ] Run a tracked-source residual scan. `changelog.md` is intentionally excluded because historical release notes may keep old names; all active code, tests, docs, and baselines should be clear.

```bash
rg -n -S "yw2y|YW2Y|Yw2y" \
  brainevent docs mypy-baseline.txt README.md examples pyproject.toml \
  -g '!docs/tutorials/**/*.ipynb'
```

Expected: no output.

- [ ] If generated cache files for deleted modules exist, remove only those generated cache files.

```bash
find brainevent -path '*/__pycache__/*yw2y*' -print -delete
```

Expected: prints deleted cache paths if any existed; otherwise no output.

- [ ] Run final formatting and worktree checks.

```bash
git diff --check
git status --short
```

Expected:
- `git diff --check` has no output.
- `git status --short` shows only the intentional renames and edits.

- [ ] Review the diff and confirm changes are naming/API-surface cleanup only, with no kernel math or high-level behavior changes.
