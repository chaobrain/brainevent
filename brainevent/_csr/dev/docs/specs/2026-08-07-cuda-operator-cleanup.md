# CUDA operator cleanup

Status: implemented
Scope: dead CUDA headers, unused header symbols, unreachable `@BE` kernel entry points,
dead Python dispatch paths, FFI module-name consistency.

## Background

`brainevent` ships 45 `.cu` files and 9 headers as *package data*; there is no compile step at
build time. The kernix toolchain (`brainevent/_op/kernix_*.py`) parses `// @BE <name>` annotations
at runtime, generates an XLA FFI wrapper per annotation, invokes `nvcc`, and registers each target
as `"<module>.<func>"`.

Consequently **every `@BE` annotation has a cost**: a generated FFI wrapper, a compiled symbol in
the `.so`, and a registered target on first lowering — whether or not the Python layer can ever
name it. Annotations that no Python code path can reach are pure overhead.

An audit of the whole CUDA surface found:

- 102 of 749 `@BE` entry points unreachable from Python.
- 2 orphan header files (never `#include`d anywhere).
- 8 unused symbols in the internal `cuda_common.h`.
- 2 dead Python dispatch paths in `_fcn/binary.py`.
- 3 FFI module names in `_jit_uniform` that break the `jit_<family>_` naming convention.

No entire `.cu` file is dead, and no Python-referenced target is dangling.

## Non-findings (checked, deliberately not acted on)

- **Wheel packaging of `include/*.h`.** Suspected missing from the wheel; verified against the
  installed `brainevent-0.1.2` `RECORD`, which lists all 9 headers. Not a bug.
- **`_dense/plasticity_binary_on_{pre,post}.cu`.** Untouched since 2026-03 and therefore look
  stale, but they are registered with `def_cuda_raw_kernel(..., asdefault=True)` and are the
  **default** GPU backend. Left alone.
- **`dispatch.h` / `check.h` public macros.** `BE_DISPATCH_FLOATING`, `BE_DISPATCH_INTEGRAL`,
  `BE_DISPATCH_ALL_TYPES`, `BE_CHECK`, `BE_CHECK_KERNEL_LAUNCH` are documented in
  `docs/reference/kernels/cpp-api.rst` and exercised by `_op/kernix_cpp_test.py` /
  `_op/kernix_dtypes_test.py`. `BE_DISPATCH_FLOATING_AND_HALF`, `BE_DISPATCH_COMPLEX`,
  `BE_DISPATCH_ALL` and `complex64_t`/`complex128_t` are unused but live in the same public
  header — retained to avoid a breaking change for downstream kernel authors.
- **`HIPBackend`** (`_op/kernix_compiler.py`) — intentional stub with implementation guidance in
  its docstring, publicly exported.
- **`load_cuda_dir`** — never called internally, but documented public API.

## Changes

### 1. Orphan headers deleted

| File | Lines | Rationale |
|---|---|---|
| `brainevent/include/brainevent/attrs.h` | 26 | Body was entirely a comment declaring itself a placeholder. Zero `#include`s repo-wide. |
| `brainevent/include/curand_common.h` | 94 | Zero `#include`s repo-wide. No `.cu` file uses cuRAND at all — the `_jit_*` RNG kernels do not need it. |

Both were byte-hashed into the compile cache key, so removal invalidates cached modules once.

### 2. Unused symbols removed from `cuda_common.h`

`warp_reduce_max_f32`, `warp_reduce_max_f64`, `warp_reduce_min_f32`, `warp_reduce_min_f64`,
`ACC_T_F16`, `ACC_T_BF16`, `ACC_T_F32`, `ACC_T_F64`.

All other helpers (`warp_reduce_sum_*`, `atomic_add_*`, `READ_*`, `WRITE_*`, `IS_ACTIVE_*`,
`ZERO_*`) remain — they are in heavy use.

### 3. Unreachable `@BE` entry points removed (749 → 647)

**Governing rule.** In the CSR/slice/dt2t files, the `_auto` dispatcher *internally launches* the
`__global__` strategy kernels. Removing an entry point therefore deletes only its **host FFI
wrapper** macro, that macro's instantiations, and its `// @BE` annotation. A `__global__` kernel is
removed only when nothing launches it afterward.

| File | Removed | Notes |
|---|---|---|
| `_csr/float_csrmv.cu` | 12 | `_auto` still launches thread/warp/block kernels |
| `_csr/float_csrmm.cu` | 8 | `_auto` still launches warp/block kernels |
| `_csr/binary_csrmv.cu` | 12 | `_auto` branches only thread/block, so `_csrmv_nt_warp_{homo,hetero}_kern` and their `DEFINE_` macros were also removed |
| `_csr/binary_csrmm.cu` | 8 | `_auto` uses warp and block |
| `_csr/dt2t.cu` | 6 | both mv and mm `_auto`s use all three strategies |
| `_csr/slice_csr_slice_rows.cu` | 32 | both fwd and grad `_auto`s use all strategies |
| `_fcn/binary_fcnmm.cu` | 16 | entire `araw` family removed, including its self-contained `__global__` kernels |
| `_fcn/binary_fcnmv.cu` | 8 | `_float` scatter variants |

Why each was unreachable:

- **`_nt_thread` / `_nt_warp` / `_nt_block`; slice `_thread` / `_warp` / `_block`; dt2t
  `_row_thread` / `_row_warp` / `_nz_thread`** — Python only ever composes the `_auto` target name
  (`_csr/float.py`, `_csr/binary.py`, `_csr/slice.py`, `_csr/dt2t.py`). The explicit strategy
  variants had no caller and no test.
- **`binary_fcnmm_araw_*`** — the dispatch that would have named them was a `'''...'''`
  pseudo-code block containing non-parsing Python (`else sraw:`).
- **`binary_fcnmv_scatter_*_float_*`** — `_fcn/binary.py` hardcodes the `_bool` suffix and coerces
  spikes with `u.math.asarray(spikes, dtype=bool)`, making the `_float` path permanently
  unreachable.

### 4. Dead Python paths removed in `_fcn/binary.py`

- The `'''...'''` pseudo-code block documenting the removed `araw` dispatch. It sat mid-function
  after executable statements (so it was not a docstring) and contained invalid Python.
- The phantom backend string `'SRAW_MM_kernel'` in both membership tests of
  `_binary_fcnmm_uses_raw_batch_first`. It is registered nowhere; passing it as `backend=` raises
  `KernelFallbackExhaustedError`, so the branch was unreachable. `'cuda_raw'` is retained.

### 5. `_jit_uniform` FFI module names aligned

`_jit_uniform/float.py` registered unprefixed FFI module names (`float_jitu`, `float_jitumv`,
`float_jitumm`) while `_jit_scalar` and `_jit_normal` use the `jit_<family>_` convention. The
kernix registry is process-global and a same-name/different-module clobber raises
`KernelRegistrationError`, so unprefixed names are a collision hazard. Renamed to
`jit_uniform_jitu`, `jit_uniform_jitumv`, `jit_uniform_jitumm`.

The `XLACustomKernel('float_jitu', ...)` **primitive** names were deliberately left unchanged —
they already match `_jit_scalar` (`float_jitsmv`) and `_jit_normal` (`float_jitnmv`), and renaming
them would alter jaxpr output and registry keys for no benefit.

## Edge cases considered

- **Orphaning a device kernel.** Removing a host wrapper can strand the `__global__` kernel it was
  the sole launcher of. Handled per-file; only `binary_csrmv.cu`'s warp kernels and the entire
  `araw` family were affected.
- **Over-removal.** `binary_fcnmm_gather_*` (48 entries) superficially resembles the dead variants
  but *is* named by Python — retained.
- **Cache invalidation.** Header deletion and the module rename change compile cache keys; the
  first GPU run after this change recompiles. Expected, not a regression.
- **`_auto` threshold semantics.** Removing explicit variants does not change which kernel `_auto`
  selects — the branch thresholds are untouched, so GPU numerics are unchanged.

## Verification

Completed:

1. **Annotation count** — `@BE` annotations dropped 749 → 647 (exactly 102 removed).
2. **Name-set diff vs HEAD** — 102 unique exported names removed, 0 added, and every removed name
   matched the intended dead-kernel pattern (no collateral removals).
3. **Template resolution** — all 51 Python `kernel_name` templates with a literal stem still
   resolve to a surviving exported kernel.
4. **No orphaned device code** — no `__global__` kernel is left without a launcher, no launcher
   references a deleted kernel, and no `DEFINE_`/`FFI_` macro is defined-but-unused or
   used-but-undefined across the 8 edited files.
5. **Residual references** — `attrs.h`, `curand_common`, `warp_reduce_max/min`, `ACC_T_`,
   `araw`/`ARAW` and `SRAW_MM_kernel` are unreferenced package-wide.
6. **Tests** — `pytest brainevent/` → 1574 passed, 0 failed, 201 skipped.
7. **Type gate** — reproduced the CI `type_check` job in an isolated venv with only
   `mypy==2.3.0` (CI deliberately installs no runtime deps): `Success: no issues found in 75
   source files`, exit 0. A local mypy run *with* runtime deps installed reports 204 pre-existing
   errors at lines untouched by this change; that configuration is not the gate.

8. **GPU validation** — run on an RTX 3080 Laptop (driver 596.49, CUDA 13.2) with jax 0.11.0 +
   jaxlib CUDA and nvcc 13.1, after deleting `~/.cache/brainevent/` so every kernel recompiled
   from source: `pytest brainevent/ -m ""` → **3773 passed, 0 failed, 29 skipped** (27m07s).
   This exercises the `cuda_raw` backend for every edited module against the `jax_raw` / `numba`
   reference paths.

### Regression caught during GPU validation

The first GPU run failed 35 CSR tests with `CompilationError` from nvcc. Cause: the initial
removal pass deleted only the *first* line of each backslash-continued
`DEFINE_CSRMV_NT_WARP_{HOMO,HETERO}(...)` instantiation in `binary_csrmv.cu`, leaving 16 dangling
argument fragments such as `READ_F32, WRITE_F32, warp_reduce_sum_f32, 0.0f)` that bound to the
following instantiation and corrupted it.

Nothing in the CPU pipeline could have caught this — `.cu` sources are never compiled unless a GPU
kernel is actually lowered, so the 1574-test CPU suite, the annotation cross-check, and the
macro-definition audit all passed against broken CUDA. The file was reverted and redone with a
removal pass that follows line continuations, and a dangling-fragment detector was run across all
45 `.cu` files to confirm no other file was affected.

**Takeaway for future `.cu` edits: any change to these sources must be validated on a GPU.** The
CPU suite proves nothing about them.

## Known follow-up (out of scope)

`csr_solve` is imported in `_csr/__init__.py`, absent from that module's `__all__`, not re-exported
from `brainevent/__init__.py`, yet listed in `docs/reference/apis/operations.rst` — a documented
API that is not importable from the package root.
