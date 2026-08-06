# Package-wide simplification spec

Status: in progress
Branch: `worktree-simplify-package-cleanup`

## Progress

Landed (each as its own commit):

- **Tier A** — A1–A8, A11, A12, A13, plus one bonus finding (a `TestCompactOnlyVector`
  class defined twice, the first shadowed and never executed).
- **Tier B** — B1 (`_dtype_sfx` ×31), B2/B3/B4 (the `_jit_*` chunking, mode and
  stride helpers).

Deliberately skipped, with reasons:

- **A9 (`_registered_platforms`)** — reported as always equal to `set(self._kernels)`.
  True on the happy path only: the field records *successful* lowering registration,
  so if `_register_fallback_lowering` raises, the two diverge and a retry would
  re-attempt registration. Removing it changes error-path behaviour for the sake of
  one set. Not worth it.
- **A10 (write-only FFI handler attributes)** — entangled with C2, which rekeys the
  same cache. Should be done together, not separately.
- **Pallas compat shim** — see "Adjudicated conflicts" below; needs JAX-version
  verification first.
- **`dtype_suffix` raising on unmapped dtypes** — recommended by the reviewer, but it
  changes dispatch behaviour rather than preserving it. Documented and pinned by test
  instead; raising is a follow-up decision, not a cleanup.

Not started: B5–B17, all of Tier C, all of Tier D.

Verification at this point: full fast suite 1574 passed / 0 failed / 201 skipped;
mypy 204 errors, down from the 208-error baseline (no regression).

## Motivation

A four-angle quality review (reuse, simplification, efficiency, altitude) of the
whole `brainevent` package (~53k lines of non-test source across ~60 modules)
produced 45 findings. This spec records them, tiers them by risk, and defines the
verification bar for landing them.

The review targeted **quality only** — reuse, simplification, efficiency, and
altitude. It did not hunt for correctness bugs. A handful of probable bugs
surfaced incidentally and are recorded in "Out of scope" below rather than fixed
here.

## Baseline (measured before any change)

- `mypy` (CI gate config): **208 errors in 23 files** (75 source files checked).
  The gate is *not* currently clean, so the bar for this work is "no new errors";
  reductions are a bonus.
- `import brainevent`: ~1.94 s cold; brainevent's own module bodies 0.95–1.28 s.
- `detect_cpp_toolchain()`: 8.4 ms/call, uncached.
- Header glob + SHA-256 per compile-cache lookup: ~75 ms, paid even on a cache hit.
- Re-lowering an identical CSR kernel: ~100–140 ms each time (full numba recompile).

## Verification bar

Per `AGENTS.md`:

- Full test suite must pass (CI runs `pytest -m ""`, including `slow` variants).
- `mypy` must not regress past the 208-error baseline.
- Meaningful tests for edge cases and critical paths, not trivial line coverage.
- Tests co-located as `foo_test.py` beside `foo.py` (suffix style, never a
  `tests/` directory, never a `test_*.py` prefix).

## Adjudicated conflicts

Three findings were contested between reviewers and were resolved by direct
inspection:

1. **`_dtype_sfx` (31 copies).** One reviewer declined to flag it, arguing each
   table maps to a different `.cu` symbol family. Verified false: all 31 bodies
   are byte-identical dtype→suffix maps (`np.dtype` vs `jnp.dtype` keys resolve to
   the same objects). Only the *base kernel name* differs, and that stays at the
   call site. `_jit_uniform/binary.py:32` already imports the table cross-module,
   proving it is shareable. **Verdict: consolidate.**
2. **`cdiv`.** One reviewer called it dead code to delete; another wanted it reused.
   Verified: genuinely unreferenced outside `_misc.py`, while three `_jit_*/csr.py`
   modules hand-roll its body as `_n_chunks`. **Verdict: keep and adopt it.**
3. **Pallas compat shim.** `pallas_triton_params` / `pallas_mosaic_tpu_params` are
   defined, exported in `__all__`, and used nowhere; the two real Pallas call sites
   pass `backend=impl_backend`. Whether wiring the shim up is correct depends on
   JAX-version behavior not yet verified. **Verdict: defer pending verification;
   do not delete and do not blind-wire.**

## Tier A — mechanical, no behavior change

| # | Site | Change |
|---|------|--------|
| A1 | `_misc.py:244–887` | Delete 9 never-called helpers (~443 lines): `is_known_type` (dup of live `_event/base.py:51`), `_block_csr_tocsr`, `_block_csr_tocoo`, `estimate_block_size`, `count_blocks`, `_count_blocks`, `_nonzero_blocks`, `_coordinate_index_dtype`. Keep `cdiv` (see A-adjacent B3). |
| A2 | package-wide | Remove 18 `ruff F401` unused imports and `F841` unused locals (`_dense/plasticity_binary.py:300,632`; `n_rows` in three `_jit_*/csr.py`). |
| A3 | `_jit_{normal,scalar,uniform}/csr.py` | `row_counts` is `device_get`'d twice; do it once. Removes a redundant device→host sync per conversion. |
| A4 | `_data.py:41` | `_buffer_registry.update(...)` is immediately followed by a loop that registers each name anyway. Delete. |
| A5 | `_csr/dt2t.py:341` | `_csrmv_dt2t_transpose_rule` is never registered; delete. |
| A6 | `_jit_uniform/binary.py:416` | `_spike_sfx` dict, sole occurrence in repo; delete. |
| A7 | `_csr/float.py:775–777` | `_hetero` arm is unreachable inside `if is_homo:`; inline the `_homo` literal. |
| A8 | `_op/main.py:437` | Invert `if ...: pass / else:` to drop a nesting level. |
| A9 | `_op/main.py:184,372` | `_registered_platforms` is always `set(self._kernels)`; delete the field. |
| A10 | `_op/numba_ffi.py:600`, `_op/numba_cuda_ffi.py:355,836` | 7 handler attributes are stored and never read; drop from the three `__init__` signatures (keep as locals for the memoisation key). |
| A11 | `_op/main.py:1217–1273` | `catch_errors` arms duplicate the timing call and the 13-line success record; unify under one `try`. |
| A12 | `_op/numba_ffi.py:97` | Header scan runs 1576 regex attempts with no early exit; add `break`, make lazy. |
| A13 | `_csr/float.py:868` | Stray `TODO` above a complete implementation; delete. |

## Tier B — consolidate duplicates

| # | Duplication | Target |
|---|-------------|--------|
| B1 | `_dtype_sfx` ×31 (+ `spk_suffix` inline ×11, `homo_suffix` ×82) | `dtype_suffix()` / `spike_suffix()` in `_op/util.py`, raising on unmapped dtype instead of silently falling back to `_f32` |
| B2 | `_normalize_chunk_size` ×4 | `_misc.py` (RNG-stream keying hazard is documented in only one copy) |
| B3 | `_is_static_zero`, `_n_chunks`, `_mode_infix` ×3 | `_misc.py`; `_n_chunks` reimplemented on existing `cdiv` |
| B4 | `_normalize_matrix_mode` ×3 | `_misc.py`; `MatrixMode` alias to `_typing.py` |
| B5 | Operator-forwarding block ×4 subclasses (~950 lines) | `DataRepresentation` in `_data.py`. Fixes drift: `%` currently works on `fcn`/`JITCMatrix` but not `csr`/`dense` |
| B6 | `_binary_op`/`_binary_rop` dispatcher ×20 | `DataRepresentation._elementwise_binary`. Fixes the `"mul with object of shape"` message copied verbatim into all 20 regardless of operator |
| B7 | CSR `*_p_call` validation preamble ×8 | `check_csr_matmul_shape()` in `_misc.py`, mirroring the existing `check_fixed_conn_num_shape` |
| B8 | `__matmul__`/`__rmatmul__` ×4 | Make `EventRepresentation.__matmul__`/`__rmatmul__` concrete |
| B9 | XLA FFI callback body ×3 | `_handle_metadata_extension()` + `_extract_buffers()` in `numba_ffi.py` |
| B10 | `_jax_x64_enabled` ×6 test modules (+3 variants) | One `enable_x64()` in `_test_util.py` |
| B11 | `_csr/main_test.py:43,59` | Import `gen_events`/`ones_like` from `_test_util.py` |
| B12 | `coo_to_csc_index` / `coo2csr`, each ×2 (np/jnp branches) | One `_group_by_axis` with the `mod = np if ... else jnp` selector already used at `_misc.py:1181` |
| B13 | `_op/benchmark.py:631–747` | `_format_hierarchical` delegates to `_format_vary_by` (verified byte-identical output) |
| B14 | `kernix_pipeline.py:269–319` vs `474–513` | One `_build_and_cache(...)` |
| B15 | `_csr/main.py:93–175` | Replace 7-way positional closure factory with module-level defs, mirroring `_fcn/main.py:102–129` |
| B16 | `_csr/main.py:58`, `_csr/binary.py:75` | Two interchangeable workspace types (frozen dataclass vs namedtuple) that convert into each other mid-autodiff; keep one |
| B17 | `_jit_*/main.py` R/C classes | Hoist 4 methods each onto the shared base using `type(self)` |

## Tier C — efficiency

| # | Site | Change | Measured cost today |
|---|------|--------|---------------------|
| C1 | `_op/main.py:459` | Memoize `entry.kernel_generator(...)` on its static kwargs | ~100–140 ms per re-lowering |
| C2 | `numba_ffi.py:715`, `numba_cuda_ffi.py:533,1004` | Drop shapes from the FFI signature (they are never read; shape comes from `buf_ptr.dims`) | One permanent FFI target + numba dispatcher per (kernel, shape), never evicted |
| C3 | `kernix_toolchain.py:634,743` | Cache toolchain detection, invalidated by the existing `set_nvcc_discovery`/`set_compute_capabilities` | 8.4 ms/call |
| C4 | `kernix_pipeline.py:92`, `kernix_cache.py:132` | Cache header digests by `(path, mtime_ns, size)` | ~75 ms per lookup, paid on cache *hits* |
| C5 | `kernix_pipeline.py:346,382` | Memoize `.cu` source reads | per lowering |
| C6 | `kernix_pipeline.py:50` | Defer `CompilationCache` `mkdir` out of import | creates a dir on `import brainevent`, breaks read-only filesystems |
| C7 | `__init__.py:18–140` | Lazy submodule imports via the existing PEP 562 `__getattr__` | ~570 ms for a user who only touches `CSR` |

## Tier D — architectural

These are design changes, not cleanups. Each needs its own validation and is
sequenced last; any that prove too risky to land safely will be written up as
follow-up specs rather than forced in.

| # | Finding | Deeper fix |
|---|---------|-----------|
| D1 | `_csr/main.py:977–2809` — `CSR`/`CSC` are a 900-line mirror, 24 methods each | Implement once on `CompressedSparseData` against `_is_row_major` / `_transposed_cls` hooks |
| D2 | `_jit_{normal,scalar,uniform}` are 71–97% textually identical | A `Distribution` descriptor parameterizing one generic family; the CPU seam (`get_numba_light_rng_funcs`) already proves it viable |
| D3 | Unit split/reattach at ~70 entry points (`split_mantissa_unit` ×82, `maybe_decimal` ×67) | A decorator on the existing `@namescope` seam |
| D4 | 191 hand-threaded `*_info=ShapeDtypeStruct(...)` kwargs across 23 files | `_register_fallback_lowering` computes them from the avals it already has |
| D5 | `_fcn/binary.py:896` — caller predicts the lowering-time backend to compute abstract output shape | Backend resolved once at bind time as a static param, or the cuda_raw generator emits its own transpose |
| D6 | `_op/main.py:453` — per-backend preconditions hardcoded in the dispatcher | `KernelEntry.precondition` supplied at registration |
| D7 | `def_kernel('jax_raw', ...)` ×75 | `def_jax_kernel` / `platform='*'` sentinel |
| D8 | `config.py` setters mutate globals read at lowering time, no invalidation | Resolve at bind time as static params, or setters clear caches |
| D9 | `_dense/binary.py:39–52` — CUDA wheel discovery reimplemented outside the toolchain layer | `kernix_toolchain.find_cuda_runtime_library(name)` |
| D10 | `_misc.py:1683` — bare `except Exception` turns a CUDA build failure into a silent CPU fallback | Cheap `gpu_available()` predicate; real toolchain errors propagate |

## Out of scope — probable correctness bugs

Recorded, not fixed here. These need reproducing tests first per `AGENTS.md` rule 4.

- `clen` int32/`ceil` normalization applied in `_jit_uniform/binary.py:741,1367` but
  not in `_jit_normal/binary.py:563,977` or `_jit_scalar/binary.py:704,1267`.
- `_is_static_zero_prob` short-circuit present in `_jit_normal` (3/3 entry points),
  partial in `_jit_uniform` (1/3), absent in `_jit_scalar` (0/3).
- `JITCScalarR * JITCScalarR` works; `JITCNormalR * JITCNormalR` raises.
- Version guards below the supported floor (`jax < 0.4.38`, `< 0.4.34`) are
  unreachable given `jax>=0.8.0`; `check_pallas_jax_version` guards `(0,7,1)` and can
  never raise. One of these is also a live mypy error (`_compatible_import.py:55`).
- `_pallas_random.py` (1,244 lines) is imported only by `__init__.py`; no `_jit_*`
  primitive registers a pallas backend.
