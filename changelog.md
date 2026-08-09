# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-08-09

A follow-up to `0.2.0` that removes the mv/mm split from the just-in-time
connectivity (JITC) families, restores the daily JAX compatibility matrix to
green, and brings the API reference back in sync with the public surface.

*JITC.* `0.2.0` exposed the fact that the mv and mm light kernels drew
*different* connectivity for the same `(prob, seed, shape)` by making the
difference explicit: a required `matrix_mode` keyword and `mat.mv` / `mat.mm`
materialization views. `0.2.1` removes the difference instead. Every JITC entry
point — scalar, normal and uniform, float and binary, dense, CSR and `dt2t` —
now draws the 32-lane mv matrix on every backend, so `matrix_mode` and the two
views are gone and `todense()` / `tocsr()` / `tocsc()` / `tocoo()` are
unambiguous again. Signatures return to their `0.1.2` shape (#190).

*Compatibility.* The Daily CI matrix now pins `0.10.0` alongside `0.8.0` and
`0.9.0`, and the operator test that broke on pinned JAX ≤ 0.9 has been made
independent of JAX's traceback-filtering behaviour.

**Requirements:** unchanged from `0.2.0` — Python ≥ 3.11, `jax` ≥ 0.8.0
(validated through 0.11.x), `brainunit` ≥ 0.0.8, `numpy` ≥ 2.0.

### ⚠️ Breaking changes & migration

`0.2.0` shipped the mv/mm split to PyPI, so the removal below is a breaking
change for code written against that release. Code written against `0.1.2` or
earlier needs no changes.

| `0.2.0` usage | `0.2.1` usage |
| --- | --- |
| `jits(w, prob, seed, shape=..., matrix_mode='mv')`, and likewise `jitn` / `jitu` | drop the keyword: `jits(w, prob, seed, shape=...)` |
| `jitsmm(..., matrix_mode='mm')`, `jitnmm`, `jitumm`, `binary_jitsmm`, `binary_jitnmm`, `binary_jitumm` | drop the keyword |
| `mat.mv.todense()` / `mat.mm.todense()` (and `.tocsr()`, `.tocsc()`, `.tocoo()`) | `mat.todense()` / `mat.tocsr()` / `mat.tocsc()` / `mat.tocoo()` |
| `from brainevent._typing import MatrixMode` | removed; there is no mode to annotate |

**Matrix-matrix JITC results recorded with `0.2.0` change.** The mm kernels
drew a 4-lane residue-class matrix; they now draw the 32-lane mv matrix, so
`jitsmm` / `jitnmm` / `jitumm` and their `binary_*` counterparts return
different — not merely reordered — values for the same `(weight, prob, seed,
shape, corder)`. Matrix-vector results are bit-identical to `0.2.0`. Re-record
any golden outputs captured from an mm path; as in `0.2.0`, seeds are not
portable across the change.

The upside is the invariant that motivated the work: for a given `(weight,
prob, seed, shape, corder)`, `jits`, `jitsmv`, `jitsmm`, `binary_jitsmv`,
`binary_jitsmm`, `jits_to_csr` and `jitsmv_dt2t` — and the `jitn` / `jitu`
equivalents — now materialize one matrix, identically on `numba` and
`cuda_raw`. A model that moves between the matrix-vector and matrix-matrix
paths, or between CPU and GPU, keeps its connectivity.

### Changed

- **JITC connectivity is mode-free.** `matrix_mode` is removed from every public
  and internal JITC entry point, and the mm generation path is folded onto the
  mv walk (`_LANE_STRIDE` = 32) in both the `numba` kernels and the CUDA
  sources (#190).
- **`todense()` / `tocsr()` / `tocsc()` / `tocoo()` work directly** on
  `JITCScalarR`/`C`, `JITCNormalR`/`C` and `JITCUniformR`/`C` again. In `0.2.0`
  they raised `NotImplementedError` and directed callers to `mat.mv` / `mat.mm`.
- **Daily CI covers every supported JAX minor.** The `jax-version` matrix is
  `[ "0.8.0", "0.9.0", "0.10.0", "" ]`; the empty entry continues to track the
  newest release.

### Fixed

- **Daily CI Tests failed on the pinned-JAX legs.**
  `test_f17_kernel_generator_failure_is_wrapped_with_alternatives` asserted that
  the kernel-generator exception was the *direct* `__cause__` of the raised
  `KernelCompilationError`. `jax._src.traceback_util.api_boundary` splices a
  synthetic frame (`UnfilteredStackTrace` on JAX 0.10,
  `JaxStackTraceBeforeTransformation` on 0.8/0.9) into the cause chain,
  demoting the real cause by one level; whether it does so depends on the
  default `jax_traceback_filtering` mode, which differs across JAX minors. The
  assertion now walks the whole chain, so it holds regardless of how many frames
  JAX inserts. Library behaviour was correct throughout — only the test was
  over-specific.

### Removed

- `matrix_mode` keyword, the `mat.mv` / `mat.mm` materialization views, the
  `MatrixMode` type alias, and the `NotImplementedError` guards that the split
  required (#190).
- The mm-specific CUDA generation kernels and their `numba` counterparts: the
  JITC `.cu` sources shed 3,439 lines against 468 added, since one drawn matrix
  needs one walk.

### Documentation

- **API reference re-synchronized with `brainevent.__all__`.** 41 stale
  `autosummary` entries were removed — symbols retired in `0.2.0`
  (`csr_solve`, `IndexedBinary1d` / `IndexedBinary2d`,
  `IndexedEventRepresentation`, `indexed_binary_dense*`, `binary_array_index`,
  `BenchmarkReport`, `register_cuda_kernels`) and 30 `lfsr*` / `get_numba_*`
  helpers that live in a private module and never resolved. 44 public exports
  gained a reference page, among them `BitPackedBinary`, `CompactBinary`,
  `bitpack`, `Dense`, `binary_csrm{v,m}_indexed`, the CSC and fixed-connectivity
  plasticity operators, the `CompilerBackend` hierarchy, the primitive-registry
  accessors and the hybrid-CSR scheduling knobs.
- **Exception hierarchy documented in one place.** `BrainEventError` and its
  nineteen subclasses now appear in `errors.rst`, grouped by subtree and
  preceded by the inheritance tree; `operator.rst` cross-references it instead
  of documenting four of them separately.
- **Two broken snippets fixed.** The quickstart and the E/I-network how-to
  constructed `JITCScalarR` and `FixedPostNumConn` from `num_pre=` / `num_post=`
  / `conn_num=` / `weight=` / `seed=` keywords that these constructors have not
  accepted since at least `0.1.2`, so both raised `TypeError` as written; the
  quickstart also built a `CSR` from undefined names. Both are rewritten against
  the real constructors and were executed end to end.
- **Deprecated aliases replaced throughout the prose docs and tutorials.**
  `FixedPreNumConn` → `FixedNumPerPost` and `FixedPostNumConn` →
  `FixedNumPerPre` (the mapping is crossed) across the explanation pages,
  how-to guides and the two data-structure notebooks; the removed
  `IndexedBinary1d` / `IndexedBinary2d` give way to `BitPackedBinary` /
  `CompactBinary`.
- `docs/specs/release-0.2.1.md` records the CI root cause and the reference audit.

### Internal

- `brainevent/__init___test.py` guards the reference against future drift: every
  `autosummary` entry must resolve on its `currentmodule`, every name in
  `brainevent.__all__` and `brainevent.config.__all__` must be documented exactly
  once, and deprecated aliases must stay undocumented.
- The full suite was re-run against pinned `jax` 0.8.0, 0.9.0 and 0.10.x before
  release.

## [0.2.0] - 2026-08-08

A correctness release spanning three layers of the stack.

*Data structures.* The just-in-time-connectivity (JITC) families now draw **the
same matrix on CPU and GPU**: the `numba` kernels were rebuilt on the CUDA
light-RNG walk, replacing the LFSR generators that silently produced different
connectivity per platform. The compressed-sparse representations validate their
structure at construction and support `int64` `indptr` for matrices whose `nnz`
exceeds the `int32` range, and a first-class `Dense` representation joins the
data-representation family (#179).

*Operator machinery.* A hardening pass over `brainevent._op` fixes the 19
defects catalogued in the 2026-07-16 operator-registration audit
(`dev/2026-07-16-op-registration-audit.md`): stale backend dispatch after runtime
backend switches, silently dropped JVP rules, incomplete compilation-cache keys,
order-dependent FFI target names, and incorrect `vmap` execution of `numba.cuda`
kernels, among others (#187).

*CUDA kernels.* A scan of the whole `.cu` tree fixes an out-of-bounds
shared-memory request that aborted the CUDA context for `float64`
`binary_csrmm`, widens 54 index expressions that wrapped past `INT32_MAX`,
restores the missing warp-per-row CSRMV dispatch tier (up to 3.4× on the row
lengths typical of sparse connectivity), and retires 154 unreachable kernel
entry points (#185, #188).

The retired `pararnn` subpackage is removed (#177).

**Requirements:** Python ≥ 3.11, `jax` ≥ 0.8.0 (validated through 0.11.x),
`brainunit` ≥ 0.0.8, `numpy` ≥ 2.0. (The `jax` ≥ 0.8 and `numpy` ≥ 2.0 floors
were raised during `0.1.2` by #171; the `0.1.2` notes below understate them.)

### ⚠️ Breaking changes & migration

| Old usage | New usage |
| --- | --- |
| `mat.todense()` / `.tocsr()` / `.tocsc()` / `.tocoo()` on any `JITCScalarR`/`C`, `JITCNormalR`/`C`, `JITCUniformR`/`C` | `mat.mv.todense()` (the matrix behind `mat @ vector`) or `mat.mm.todense()` (the matrix behind `mat @ matrix`); the bare calls now raise `NotImplementedError` |
| `jits(w, prob, seed, shape=...)` / `jitn(...)` / `jitu(...)` | `matrix_mode='mv'` or `matrix_mode='mm'` is now a **required** keyword |
| `numba`-backend JITC results computed with `0.1.x` | values change: the CPU kernels now draw the CUDA matrix. Re-record any CPU golden outputs; seeds are not portable across `0.1.x` → `0.2.0` |
| `CSR(...)` / `CSC(...)` with malformed `indices` / `indptr` | now raises `TypeError` / `ValueError` / `OverflowError` at construction instead of failing later inside a kernel |
| `CSR.fromdense(..., index_dtype=jnp.int64)` | raises — `indices` are always `int32`; use `indptr_dtype=` to control offset precision |
| `load_cuda_inline(..., replace=True)` / `force_rebuild=True` with *changed* source in a live process | raises `KernelRegistrationError`; register under a new `name=` to iterate on a kernel within one process (see _Changed_) |
| Explicit CUDA entry points `binary_csrm{v,m}_{nt_thread,nt_warp,nt_block,t_warp}_*`, `csr_slice_rows_{thread,warp,block}_*`, `binary_fcnmm_araw_*`, `binary_fcnmv_scatter_*_float_*`, `dt2t_{row_thread,row_warp,nz_thread}_*` | the auto-dispatching wrappers (`*_nt_auto*`, `*_hybrid`), which select the same device kernels internally |
| `import brainevent.pararnn` | removed, with no replacement |

Three of these deserve a note.

**JITC CPU/GPU parity changes CPU numbers.** In `0.1.x` the `numba` kernels
generated connectivity with an LFSR stream while the `cuda_raw` kernels used the
light-RNG chunk/lane walk, so the *same* `(prob, seed, shape)` described a
different matrix on each platform. The `numba` kernels now reproduce the CUDA
walk exactly, so a model moved between CPU and GPU keeps its connectivity — but
CPU results recorded against `0.1.x` will not reproduce. Only the drawn matrix
changed; the operator semantics did not.

**`mv` and `mm` are genuinely different matrices.** The light kernels walk 32
lanes for matrix-vector and 4 threads (AW-T4) for matrix-matrix, and the stride
is part of the drawn matrix. Bare materialization was therefore ambiguous and
silently returned the mv matrix even when the caller was about to use `mm`; it
now raises, and the `mat.mv` / `mat.mm` views select explicitly.

**The removed CUDA entry points were already unreachable from Python.** A
reachability pass over every `// @BE` annotation against every kernel-name
template in the package found that no Python path could name them; the
`__global__` kernels behind the auto-dispatched families are retained and still
launched internally.

### Added

- **`Dense` — an explicit dense data representation (`brainevent.Dense`).** The
  dense counterpart to `CSR` / `CSC` / `FixedNumPerPre` / `FixedNumPerPost`,
  holding the full weight matrix as its single pytree leaf while exposing the
  same representation contract: unit-aware `data`, `shape`, `backend`, named
  `buffers`, event-driven binary matmul dispatch (`@` with `BinaryArray` /
  `BitPackedBinary`), and the `update_dense_on_binary_pre` / `_post` plasticity
  helpers. It is a registered pytree, so it passes through `jax.jit` in the same
  style as the sparse families.
- **`int64` `indptr` support for `CSR` / `CSC`.** The constructors and
  `CSR.fromdense` take `indptr_dtype`: `"auto"` (default) keeps `int32` and
  promotes to `int64` only when `nnz` exceeds the `int32` range; an explicit
  `int32` raises `OverflowError` rather than truncating. `int64` offsets require
  `jax_enable_x64` — the library refuses with an actionable error instead of
  toggling the global config on your behalf, since JAX would otherwise silently
  downcast. `indices` stay `int32` in every case: they are secondary-axis
  coordinates bounded by the matrix dimension, so widening them would cost
  bandwidth for nothing.
- **Tunable CSR hybrid CUDA scheduler (`HybridConfig`, `get_hybrid_config`,
  `init_csr_config`).** The four hybrid kernels (`binary_csrmv_hybrid.cu`,
  `binary_csrmm_hybrid.cu`, and their `binary_indexed_*` siblings) expose
  `block_size`, `fixed_scatter_blocks`, `tpr_threshold`, and `task_nnz` as
  `-DBE_HYBRID_*` compile-time constants. `init_csr_config()` benchmarks
  candidate configurations by compiling the *production* kernel and persists the
  winner per GPU model to `<cache_dir>/csr_hybrid_config.json`; later processes
  pick it up through `get_hybrid_config()`. Resolution order is
  `$BRAINEVENT_CSR_HYBRID_CONFIG` (a JSON object, for CI and one-off overrides)
  → the per-`device_kind` entry in the cache file → the defaults baked into the
  `.cu` sources. The same function sizes the host-side task workspace, so the
  compiled `.so` and the Python allocation can no longer drift apart.
  `init_csr_config` is GPU-only, never runs automatically, and must not be
  called inside a JIT closure.
- **`'cublas'` GPU backend for `binary_densemv` / `binary_densemm`.** A cuBLAS
  dense path (`float32` weights, `bool` spikes) alongside the event-driven
  `cuda_raw` default and the `jax_raw` reference; useful as a dense-throughput
  baseline at high spike rates. `libcublas` is located in the installed `nvidia`
  CUDA Python packages at load time.
- **`mat.mv` / `mat.mm` materialization views on every JITC family.** Each view
  exposes `todense` / `tocsr` / `tocsc` / `tocoo` for the matrix that mode
  actually uses. For column-oriented matrices (`JITCScalarC` and siblings) the
  view also applies the swapped generation shape, so the dense form matches the
  matvec — the light kernels' `chunk_size` depends on `shape[1]`, which made the
  old direct materialization shape-inconsistent.
- **`numba` CPU kernels for the JITC CSR and `dt2t` paths.** `jits_csr_count` /
  `jits_csr_fill` and their `jitn_*` / `jitu_*` siblings, plus the fused
  `jitsmv_dt2t` / `jitnmv_dt2t` / `jitumv_dt2t` fill primitives, previously had
  CUDA-only backends; `.tocsr()` and the eligibility-trace operators now run on
  CPU.
- **Light-RNG helpers in the numba random utilities** — `light_rng_uniform01`,
  `light_rng_normal01`, and `get_numba_light_rng_funcs()`, the CUDA-compatible
  `(seed, row, col)` weight hashes and the njit dispatch table backing the
  kernels above. Listed in the *Utilities* API reference.
- **`matrix_mode` on the JITC CSR materialization entry points** (`jits_to_csr`,
  `jitn_to_csr`, `jitu_to_csr`, defaulting to `'mv'`).
- **jax 0.11.x is now a validated version (#182).** The numba XLA FFI bridge
  raises its validated ceiling (`_MAX_VALIDATED_JAX`) from `0.10` to `0.11`, so
  installing `brainevent` alongside jax 0.11 no longer emits the "untested jax"
  `RuntimeWarning`. jaxlib 0.11 reports the same `XLA_FFI_API` version (`0.3`) as
  0.10, meaning the hand-mirrored `ffi.h` struct layout is unchanged; the full
  test suite passes on jax 0.11.0 on both the CPU and CUDA backends. The
  `jax>=0.8.0` floor is unchanged.

### Fixed

- **`binary_csrmm` no longer aborts the CUDA context with `float64` weights
  (#188).** The CSRMM non-transpose block kernels stage one accumulator per
  `(strip, lane)` pair — 8 strips × 32 lanes — but requested only
  `8 * sizeof(ACC_T)` bytes of dynamic shared memory, a 32× under-request. The
  1 KiB overrun of the 32-bit instantiations goes unnoticed on sm_86 because the
  per-block shared-memory window is rounded up; `float64` needs 2 KiB and faults,
  so `binary_csrmm` with `float64` weights and `avg_nnz > 512` (where `nt_auto`
  selects the block kernel) died with `CUDA_ERROR_ILLEGAL_ADDRESS`. Fixed across
  26 instantiations in `binary_csrmm.cu` and `binary_indexed_csrmm.cu`.
- **CSRMM and JIT index arithmetic no longer wraps past `INT32_MAX` (#188).**
  The CSRMM kernels computed `B[indices[j] * n + c]` and `C[row * n + c]` in
  32-bit arithmetic; since `B` is usually a `bool`/`int8` event matrix, 2 GiB of
  allocation is enough to cross the boundary and read out of bounds. The JIT
  connectivity families had the same defect in
  `chunk_counts[row * n_chunks + chunk_id]`, reachable once the indexed buffer
  passes ~8.6 GB. Widened 30 CSRMM subscripts and 24 JIT chunk sites to
  `size_t`. Shared-memory subscripts with a literal `* 32` are deliberately
  unchanged — they are bounded by the block size.
- **Backend switches now take effect immediately (#187).**
  `XLACustomKernel.set_default`, `brainevent.config.set_backend`, and
  `clear_backends` invalidate JAX's dispatch and executable caches
  (`jax.clear_caches()`) whenever the effective setting changes. Previously,
  eager calls and warm `jax.jit` functions kept executing the previously selected
  backend. Note the invalidation is process-global: the next call of every jitted
  function recompiles.
- **`defjvp` rejects mismatched rule arity (#187).** Registering a number of JVP
  rules different from the primitive's number of inputs now raises `ValueError`
  at differentiation time instead of silently dropping trailing gradients
  (previously `zip` truncation produced wrong, silent results). A multi-result
  JVP rule returning a bare array instead of a sequence now raises `TypeError`.
  One latent in-tree mismatch (`binary_fcnmm_p`: four rules for three inputs) was
  corrected.
- **`vmap` over `numba.cuda` kernels computes correct results (#187).** Batched
  calls now execute one kernel launch per batch slice with the kernel's original
  launch configuration, instead of reusing the launch grid of the unbatched shape
  over folded buffers (which silently corrupted any kernel that couples rows,
  e.g. stencils and reductions). Kernels wrapped with an explicit `grid=` cannot
  be batched; combining `grid=` with `vmap_method=` raises `ValueError` at wrap
  time. Only one `vmap` level is supported: nested `vmap` now raises a clear
  error instead of returning uninitialized memory for all but the first slice.
- **Compilation-cache keys cover everything that affects codegen (#187).** The
  kernix (inline C++/CUDA) cache key now includes the resolved `FunctionSpec`s
  and the content of user-provided extra include headers (key schema v2 — old
  cache entries are recompiled once, not misused). The `numba` CPU FFI memo no
  longer keys on array shapes, so one kernel serves all shapes of the same dtype
  signature.
- **FFI target names are content-derived (#187).** CPU and CUDA numba kernels
  register under a fingerprint of the kernel's bytecode, constants, closure
  values, and referenced globals rather than a process-order counter, making
  `jax.export` artifacts stable across processes. Kernels whose content cannot be
  fingerprinted deterministically fall back to per-process counter names.
- Unknown, packed sub-byte (`S1`–`S4`, `U1`–`U4`, `F4E2M1FN`), and FP8 buffer
  dtypes now raise a descriptive `ValueError` instead of being reinterpreted as
  raw bytes; `bfloat16` is rejected explicitly on the numba paths. XLA FFI
  extension chains are walked fully, and FFI error objects are destroyed after
  use (#187).
- CUDA output buffers for kernels that accumulate are zero-filled on XLA's stream
  (previously uninitialized memory could leak into results). Transient CUDA probe
  failures no longer permanently disable the `numba.cuda` backend for the process
  (#187).
- Kernel construction/compile failures during lowering now raise
  `KernelCompilationError` (with the original exception as `__cause__` and the
  remaining registered backends listed); calling a kernel on a platform with no
  registered backend raises `KernelFallbackExhaustedError` naming the platforms
  that are registered. Both are exported from `brainevent` (#187).

### Performance

- **Restored the warp-per-row CSRMV tier (#188).** `float_csrmv.cu` documents a
  three-tier row mapping (thread / warp / block), but the three binary CSRMV
  dispatchers only had two — every row length from 16 to 512, the normal range
  for sparse neural connectivity, ran the thread-per-row kernel, where the 32
  lanes of a warp each walk a different row and every `indices[j]` load is
  uncoalesced. Measured on sm_86 (m = k = 65536, f32 hetero, bool spikes),
  thread vs warp: 0.088 vs 0.043 ms at `avg_nnz` 32, 0.209 vs 0.076 at 64, 0.950
  vs 0.277 at 256. Thresholds follow the crossovers: thread below 16, warp to
  512, block above.
- **Packed every one-warp-per-block launch into 256-thread grid-strided blocks
  (#188).** A block holds a scheduler slot regardless of its size, so
  `<<<m, 32>>>` wasted 7/8 of it. Applied to the CSRMV warp kernels (3.3× at
  `avg_nnz` 8, 2.5× at 32, parity from 128), the `float_csrmm` warp kernels
  (2.05× at 2, 1.19× at 64), and the `csr_slice_rows` / `dt2t` row-warp kernels.
  The last group is not a uniform win: it gains 2.5× below `avg_nnz` 32 and
  regresses ~20% near 512, which was taken deliberately because real sparse
  connectivity sits well below 256 non-zeros per row. No `<<<..., 32>>>` launch
  remains in the tree.

### Changed

- **JITC `numba` kernels rebuilt on the CUDA light-RNG walk (#179).** The dense,
  matvec, matmat, CSR, and `dt2t` generators across `_jit_scalar`, `_jit_normal`,
  and `_jit_uniform` now share the CUDA `light_rng_init_wpr` /
  `stationary_initial_q` initialization, lane strides, and chunking, and sample
  weights with the same `(seed, row, col)` hash, so `numba` and `cuda_raw`
  materialize bit-identical matrices. See the *Breaking changes* note above.
- **`CSR` / `CSC` validate their structure at construction (#179).** `indices`
  must be integral, non-negative, in bounds for the secondary axis, and are
  coerced to `int32`; `indptr` must be 1-D, `int32`/`int64`, of length *primary
  dimension + 1*, start at `0`, be monotonically non-decreasing, and end at
  `nnz`. Value checks are host-side and therefore skipped under tracers, where
  only the static dtype and shape invariants are enforced. Structure-preserving
  paths (`with_data`, `transpose`, `apply`, `tree_unflatten`, data-only binary
  ops) reuse the already-validated structure, so they add no host readback inside
  `jax.jit` / `jax.vmap`.
- **`FixedNumPerPre` / `FixedNumPerPost` coerce their connection indices to
  `int32`** (#179), matching the compressed-sparse families; bounds are still
  validated by the existing invalid-index check.
- **Re-registering an FFI target with different content now raises
  `KernelRegistrationError` on every platform (#187)** (including
  `load_cuda_inline(..., replace=True)` / `force_rebuild=True` with changed
  source). Live re-pointing of an already-registered XLA FFI target is not
  supported by JAX (CPU raises; CUDA silently keeps the old handler), so
  brainevent refuses deterministically instead of silently dispatching stale code
  — register under a new `name=` to iterate on a kernel within one process.
  Re-registration of *unchanged* source (e.g. `force_rebuild=True` twice) is an
  idempotent no-op: registration identity is the deterministic compilation cache
  key, not the compiler's output bytes.
- Registering a second primitive under an existing name emits a `UserWarning`
  (the new registration still wins, as before) (#187).
- **Duplicated internals consolidated (#183).** The dtype→CUDA-suffix table had
  31 verbatim copies across 22 files and now lives in `_op/util.py` as
  `dtype_suffix()` / `spike_suffix()` (the lenient `'_f32'` fallback is preserved,
  documented, and tested). The JIT families' `_normalize_chunk_size`,
  `_normalize_matrix_mode`, `_MV_STRIDE` / `_MM_STRIDE`, `_is_static_zero`,
  `_n_chunks`, and `_mode_infix` move to `_misc.py`, and `MatrixMode` to
  `_typing.py`. This duplication was the riskiest kind: `chunk_size` participates
  in the RNG stream keying, so a divergent default would not raise — it would
  silently make one operator draw a different connectivity matrix than its
  siblings.
- **CSR binary task capacity is single-sourced (#179).** The host-side workspace
  sizing moved out of `_csr/main.py` into `hybrid_config.hybrid_task_capacity`,
  the same function that emits the kernel's compile flags.
- The three `_jit_uniform` FFI module names are prefixed `jit_uniform_*` to match
  `_jit_scalar` and `_jit_normal` (#185); the registry is process-global and
  unprefixed names risked a `KernelRegistrationError` clobber. The
  `XLACustomKernel` primitive names are deliberately unchanged.
- **`CONTRIBUTING.md` rewritten (#182).** It previously described *BrainPy* and
  linked to a page that returns HTTP 404. It is now a self-contained `brainevent`
  guide covering development setup, the test/mypy/pre-commit gates, docs builds,
  code style, the pull request checklist, and GPU kernel contributions.
- **`SECURITY.md` rewritten (#182).** Vulnerability reports now go through GitHub
  private vulnerability reporting or email instead of public issues, and the
  policy documents supported versions, response targets, and the trust boundary
  around the runtime C++/CUDA compilation APIs (`load_cpp_inline`,
  `load_cuda_inline`, and friends).
- **`CODE_OF_CONDUCT.md` upgraded** from Contributor Covenant 2.1 to 3.0 (#182).
- **`.gitattributes` expanded** to cover the header, reStructuredText, notebook,
  YAML, TOML and image file types actually present in the tree, with
  language-aware diff drivers, explicit binary markers, GitHub
  language-statistics hints, and `export-ignore` rules for development-only
  infrastructure (#182).

### Removed

- **The `brainevent.pararnn` subpackage (#177).** The parallel-RNN training
  module — diagonal GRU/LSTM cells, the Newton solver, the parallel-prefix
  reduce, and their fused CUDA kernels — is deleted along with its tests and
  benchmark. It was never re-exported from `brainevent/__init__.py` and nothing
  else in the package imported it, so the top-level API is unaffected;
  `import brainevent.pararnn` no longer resolves.
- **154 unreachable CUDA entry points (#185, #188).** Every `// @BE` annotation
  costs a generated XLA FFI wrapper, an nvcc compile, and a registration at first
  lowering, whether or not Python can name it. Two reachability passes removed
  102 and then 52 entry points — the explicit `_nt_thread` / `_nt_warp` /
  `_nt_block` and `t_warp` CSR families, the `csr_slice_rows` tier wrappers, the
  `binary_fcnmm_araw_*` family, the float `binary_fcnmv_scatter_*` variants, and
  the `dt2t` tier wrappers. Device kernels reachable through an `_auto`
  dispatcher were kept; the genuinely orphaned ones went with their wrappers,
  which also retired a below-sm_70 compile trap (`atomicAdd` called directly on
  `__half*` / `__nv_bfloat16*` instead of the arch-guarded helpers).
- **Orphan headers and dead helpers (#185, #183).**
  `include/brainevent/attrs.h` (a self-declared placeholder) and
  `include/curand_common.h` (no `.cu` file uses cuRAND) are deleted, along with
  the unused `cuda_common.h` symbols `warp_reduce_max/min_f32/f64` and `ACC_T_*`.
  On the Python side, nine never-called `_misc.py` helpers (~412 lines: the
  block-sparse subsystem, `_coordinate_index_dtype`, and the `is_known_type`
  duplicate), the never-registered `_csrmv_dt2t_transpose_rule`, the phantom
  `'SRAW_MM_kernel'` backend string, and 24 unused imports are removed.

### Internal

- **Every test is now co-located with the module it tests (#184, #186).** Rule 11
  of `AGENTS.md` — each module `foo.py` keeps its tests in a sibling
  `foo_test.py`, no `tests/` directory and no `test_*.py` prefix — is applied to
  the remaining violations: `_csr/test_util.py` (which matched pytest's default
  collection glob) is renamed `_csr/_test_util.py`, and eleven orphan test files
  are merged or split into their target modules. The only shipping-code change is
  that the `__getattr__` deprecation shim moves from `__init__.py` to a new
  `brainevent/_deprecation.py`, storing rename targets as name strings resolved
  against a caller-supplied namespace rather than live objects; public behaviour
  is unchanged. Three module-level `pytestmark = pytest.mark.slow` declarations
  became per-item decorators, which would otherwise have dropped 152 fast tests
  out of the default lane.
- `CLAUDE.md` is renamed `AGENTS.md`, with a `CLAUDE.md` stub importing it for
  backward compatibility (#181).
- Bumped `mypy` from 2.1.0 to 2.3.0 (#175).
- Test-suite fixes for the new CSR initialization module (#180).


## [0.1.2] - 2026-07-03

A consolidation release. Three threads land together: the `DT2T` / `DT_to_T`
naming convention is folded into a single, consistently-cased `dt2t` name;
batched (`mm`) variants of the `dt2t` operators are added for D-RTRL
eligibility traces; and the GPU cuSPARSE SpMV/SpMM backends are consolidated
under one `cusparse` name. Alongside these, several GPU-only autodiff and
output-shape defects in the event-driven CSR and fixed-connection-number
kernels are fixed.

**Requirements:** unchanged from `0.1.1` — Python ≥ 3.11, `jax` ≥ 0.5.0,
`brainunit` ≥ 0.0.8, `numpy`, `absl-py`.

### ⚠️ Breaking changes & migration

No compatibility aliases are kept for this release — update call sites
directly:

| Old name | New name |
| --- | --- |
| `csrmv_DT2T` / `cscmv_DT2T` / `csrmv_DT2T_p` | `csrmv_dt2t` / `cscmv_dt2t` / `csrmv_dt2t_p` |
| `jitn_DT2T` / `jits_DT2T` / `jitu_DT2T` | `jitnmv_dt2t` / `jitsmv_dt2t` / `jitumv_dt2t` |
| `fcnmv_DT2T` | `fcnmv_dt2t` |
| `DataRepresentation.DT_to_T` / `DT_to_T_transposed` (and every backend override on `CSR`, `CSC`, `FixedNumPerPost`, `JITCScalar*`, `JITCNormal*`, `JITCUniform*`) | `.dt2t` / `.dt2t_transposed` |
| `jitn_DT2T_fill_p` / `jits_DT2T_fill_p` / `jitu_DT2T_fill_p` / `jitn_csr_fill_p` / `jits_csr_fill_p` / `jitu_csr_fill_p` | removed from the public API (see _Removed_ below) |
| `binary_csrmv` / `binary_csrmm` GPU backend `'JAX_cusparse'` | `'cusparse'` |
| `binary_csrmv` / `binary_csrmm` GPU backend `'BCOO_cusparse'` | removed — use `'cusparse'` or the default `'cuda_raw'` |

### Added

- Batched (`mm`) variants of the per-synapse `dt2t` operators, implementing
  the batched `Dᵗ εᵗ⁻¹` term of the D-RTRL eligibility-trace update
  `εᵗ ≈ Dᵗ εᵗ⁻¹ + diag(D_fᵗ) ⊗ xᵗ`. Both operands carry a shared leading
  batch axis: `y` holds the per-neuron factor `Dᵗ` with shape
  `(n_batch, n_hidden)` and the weight operand holds the per-synapse trace
  `εᵗ⁻¹` with shape `(n_batch, ...)`.
  - `csrmm_dt2t` / `cscmm_dt2t` / `csrmm_dt2t_p` — CSR/CSC layouts;
    `w` is `(n_batch, nse)` and the output matches `w`. The primitive ships
    `numba` (CPU), `cuda_raw` (GPU, default; batched row-thread/row-warp/
    nz-thread kernels auto-dispatched on `avg_nnz`, non-transpose path), and
    `jax_raw` (CPU/GPU/TPU) kernels plus JVP rules.
  - `fcnmm_dt2t` — fixed-connection-number (ELL) layout; `weights` is
    `(n_batch, rows, n_conn)` (or a size-1 homogeneous value) and the
    output is `(n_batch, rows, n_conn)`. Pure JAX, fully differentiable.

### Changed

- **`DT2T` renamed to `dt2t` across the public API**, and the JIT-connectivity
  variants additionally gain an `mv` infix matching their `jitnmv`/`jitsmv`/
  `jitumv` siblings. Every function using the `DT2T` naming convention —
  `csrmv_DT2T`, `cscmv_DT2T`, `csrmv_DT2T_p`, `fcnmv_DT2T` — is now spelled
  with the lowercase `dt2t` suffix; `jitn_DT2T` / `jits_DT2T` / `jitu_DT2T`
  become `jitnmv_dt2t` / `jitsmv_dt2t` / `jitumv_dt2t`. Purely a rename;
  behavior is unchanged.
- **`DataRepresentation.DT_to_T` / `DT_to_T_transposed` renamed to `.dt2t` /
  `.dt2t_transposed`.** These are the per-synapse `y`-to-`W`-shaped
  conversion methods declared on the base `DataRepresentation` contract and
  overridden by every concrete representation that supports them directly
  (`CSR`, `CSC`, `FixedNumPerPre`, `FixedNumPerPost`, `JITCScalarR`/`C`,
  `JITCNormalR`/`C`, `JITCUniformR`/`C`); the `JITCMatrix` base class's
  `UnsupportedOperationError` fallback is renamed identically. Purely a
  rename; behavior and signatures are unchanged.
- **GPU cuSPARSE backend for `binary_csrmv` / `binary_csrmm` renamed
  `'JAX_cusparse'` → `'cusparse'`.** The `jax.experimental.sparse`-backed
  SpMV/SpMM kernels are now selected with `backend='cusparse'`. The default
  GPU backend remains `'cuda_raw'`, so code that does not pin a backend is
  unaffected.

### Removed

- **`*_fill_p` fill-primitive exports.** `jitn_csr_fill_p`, `jits_csr_fill_p`,
  `jitu_csr_fill_p`, `jitnmv_dt2t_p`, `jitsmv_dt2t_p`, and `jitumv_dt2t_p`
  (renamed from `jitn_DT2T_fill_p` / `jits_DT2T_fill_p` / `jitu_DT2T_fill_p`)
  are no longer re-exported from `brainevent.jit_normal`/`jit_scalar`/
  `jit_uniform` or top-level `brainevent`. They were internal
  `XLACustomKernel` primitives backing `jitnmv_dt2t`/`jitsmv_dt2t`/
  `jitumv_dt2t` and `.tocsr()`, never meant to be called directly; they
  remain defined in their respective submodules.
- **`'BCOO_cusparse'` GPU backend for `binary_csrmv` / `binary_csrmm`.** The
  redundant BCOO/BCSR-based cuSPARSE kernel path is removed; the equivalent
  `jax.experimental.sparse` path remains available as `backend='cusparse'`.

### Fixed

- **Gradients of `binary_csrmv` / `binary_csrmm` no longer fail on GPU-only
  backends.** The autodiff (JVP / transpose) rules form tangents and
  cotangents with the *float* `csrmv` / `csrmm` primitive while forwarding the
  binary primitive's backend name; a GPU-only backend such as `'cusparse'` is
  not registered on the float primitive, so the backward pass raised
  `KernelFallbackExhaustedError`. The rules now fall back to automatic backend
  selection whenever the float primitive cannot service the requested backend,
  leaving `'cuda_raw'` / `'jax_raw'` / `'numba'` behaviour unchanged.
- **`binary_fcnmm` returns its documented logical shape on `cuda_raw`.** With
  `transpose=True` on the `cuda_raw` backend the high-level wrapper leaked the
  kernel's internal "batch-first" `(n, num_post)` layout instead of the
  documented `(num_post, n)` shape that every other backend already returns.
  The wrapper now normalises the output, matching the dense reference and the
  `jax_raw` path.

## [0.1.1] - 2026-06-18

A maintenance release focused on the correctness and cross-version compatibility
of the JAX custom-operator / FFI layer. There are no public API changes and no
new deprecations; code written against `0.1.0` runs unchanged.

**Requirements:** unchanged from `0.1.0` — Python ≥ 3.11, `jax` ≥ 0.5.0,
`brainunit` ≥ 0.0.8, `numpy`, `absl-py`.

### Fixed

- **Hardened the JAX custom-op / FFI layer against silent wrong answers and
  process crashes (#164).** An audit of `brainevent/_op` and the C++/CUDA FFI
  headers (`brainevent/include`) fixed ~30 defects, most of which produced
  silently-incorrect output or killed the host process instead of raising a
  clean Python error. Notable fixes:
  - numba CPU/CUDA callbacks no longer swallow exceptions and return a NULL
    `OkStatus` (which left the output buffer uninitialized); they now build a
    real `XLA_FFI_Error*` so JAX raises.
  - `fp16` / `bf16` / `complex` dtypes are handled via raw byte-views and
    `ml_dtypes.bfloat16` instead of indexing a `None` entry in the dtype map.
  - The GPU callback binds the XLA-assigned device before allocating device
    arrays and streams, closing a multi-GPU data race.
  - `BE_CHECK` / `BE_CUDA_CHECK` raise C++ exceptions that propagate as
    `xla::ffi::Error` rather than calling `abort()` and sending `SIGABRT` to the
    host process.
  - FFI targets are memoized per `(kernel, shapes, dtypes, platform, launch
    config)` instead of being re-registered on every call, and the compile cache
    key now incorporates header byte-contents, the jaxlib version, and the
    include paths so any header edit triggers a rebuild.
- **Corrected `indptr` and CSC construction (#166).** Fixes index-pointer and
  CSC building along with related dtype handling, covered by new regression
  tests.
- **numba FFI bridge now works on `jax`/`jaxlib` 0.7–0.9, not only 0.10+ (#167).**
  The XLA FFI metadata handshake reported a hardcoded API version (`0.3`), which
  only the jaxlib bundled with `jax` 0.10+ accepts. Older jaxlib builds advertise
  a lower framework version (`0.1` for 0.7/0.8, `0.2` for 0.9) and rejected every
  numba CPU/`numba_cuda` kernel registration with an
  `INVALID_ARGUMENT … incompatible API version` error, failing ~180 tests on
  those versions. The bridge now detects the installed jaxlib's FFI API version
  from its bundled `xla/ffi/api/c_api.h` header and reports that, so registration
  succeeds across the supported `jax >= 0.5` range.
- **Restored compatibility with newer JAX (#168).** Recent JAX removed the
  public `jax.interpreters.batching.not_mapped` symbol, breaking the
  unpinned-JAX CI job with an `AttributeError`. The all-unbatched branch of
  `general_batching_rule` now returns a bare `None` batch dimension, which every
  supported JAX (0.7.2+) treats identically.
- **Aligned the JITC test suite with the `saiunit` ≥ 0.4 unit contract (#165).**
  `saiunit` ≥ 0.4 correctly rejects a unit-bearing relative tolerance; the
  JIT-connectivity unit tests now pass `rtol` as a dimensionless value and keep
  the physical unit only on `atol`, matching the documented `allclose` contract.

### Internal

- Bumped `codecov/codecov-action` from 6 to 7 (#163).

## [0.1.0] - 2026-06-07

First stable feature release of `BrainEvent` on PyPI. It consolidates the
event-driven data structures (binary / bit-packed / compact events; `CSR` / `CSC`,
fixed-number connectivity, and just-in-time connectivity matrices) behind a
single, uniform API, ships inline type information, and retires the legacy names
accumulated during the `0.0.x` series.

> **Not to be confused with the historical `V0.1.0` git tag** (2025-05-02), which
> was tagged on GitHub but never published to PyPI. The PyPI line ran
> `0.0.1.postN` → … → `0.0.7`; this `0.1.0` is the first `0.1.0` distributed on
> PyPI. See the `[V0.1.0]` section below for the historical note.

**Requirements:** Python ≥ 3.11, `jax` ≥ 0.5.0, `brainunit` ≥ 0.0.8, `numpy`,
`absl-py`.

### ⚠️ Breaking changes & migration

This release standardizes naming, but **retains a backward-compatibility shim** so
every public name exported by v0.0.7 stays importable (see _Deprecated_ below).
Renamed symbols forward to their replacement with a `DeprecationWarning`; names
whose underlying functionality was removed raise an `AttributeError` that names the
replacement. Recommended updates:

| Deprecated / changed name | Replacement / migration |
| --- | --- |
| `EventArray` | `BinaryArray` |
| `JITCHomoR` / `JITCHomoC` | `JITCScalarR` / `JITCScalarC` |
| `FixedPostNumConn` / `FixedPreNumConn` | `FixedNumPerPre` / `FixedNumPerPost` |
| `FixedNumConn.to_csr` / `to_csc` / `to_dense` | `tocsr` / `tocsc` / `todense` |
| `csr_on_pre`, `csr2csc_on_post`, `dense_on_pre`, `dense_on_post` | `update_csr_on_binary_pre`, `update_csc_on_binary_post`, `update_dense_on_binary_pre`, `update_dense_on_binary_post` |
| `EllLayout` / `CscLayout` | (removed — use the canonical representations) |
| `COO` sparse class & operators | `CSR` / `CSC` (+ `coo2csr` and the `*_index` helpers) |
| `CSC.__getitem__(i)` → column `i` | now returns **row** `i`; use `csc.transpose()[i]` or `csc.todense()[:, i]` for the old result |
| `JITCScalar*` / `JITCNormal*` / `JITCUniform*` `.fromdense` / `yw_to_w` / `update_on_*` | materialize with `.tocsr()` first, then operate |

`import brainevent` no longer pulls in `brainstate`.

### Added

- **Uniform common-API contract on `DataRepresentation`**: every concrete data
  representation now exposes (or deliberately refuses) a single conversion and
  neural-plasticity surface — `fromdense`, `todense`, `tocoo`, `tocsr`, `tocsc`,
  `yw_to_w`, `yw_to_w_transposed`, `update_on_pre`, `update_on_post`. The base
  class declares stubs so a missing override fails loudly rather than silently
  inheriting an unrelated implementation (#161).
- **Format conversions** `tocsr` / `tocsc` / `tocoo` for `CSR`, `CSC`,
  `FixedNumPerPre`, `FixedNumPerPost`, and the JIT-connectivity matrices (the
  latter materialize eagerly via `tocsr` and delegate the rest). CSR/CSC
  conversions are `jax.jit`-safe (#153, #161).
- **`FixedNumPerPre.fromdense` / `FixedNumPerPost.fromdense`**: build a
  fixed-num-connection matrix from a dense array. With `num_conn=None` the dense
  matrix must have a uniform per-row (per-column) non-zero count; passing
  `num_conn` pads short rows with in-range zero-weight sentinels and raises
  `ValueError` on overflow. Physical units are preserved (#161).
- **Sparse row slicing** for `CSR`, `CSC`, `FixedNumPerPre`, and `FixedNumPerPost`:
  a dense `__getitem__` returning row(s) of the logical matrix `W` with full NumPy
  index semantics (`int` / `list` / `tuple` / `array` / Python `slice`, negative-index
  wrapping, concrete out-of-bounds raising `IndexError`), plus a sparse
  `slice_rows(index)` returning `W[rows, :]`
  (`CSR`→`CSR`, `CSC`→`CSC`, `FixedNumPerPre`→`FixedNumPerPre`, `FixedNumPerPost`→`CSR`).
  `FixedNumPerPre.slice_rows` is `jax.jit`-safe; the other `slice_rows` paths have a
  data-dependent number of non-zeros and must run outside `jax.jit` (#145).
- **`UnsupportedOperationError`** (subclass of `BrainEventError`): raised when an
  operation is structurally meaningless for a representation, distinct from
  `NotImplementedError`. The JIT-connectivity matrices (`JITCScalar*`,
  `JITCNormal*`, `JITCUniform*`) raise it for `fromdense`, `yw_to_w`,
  `yw_to_w_transposed`, `update_on_pre`, and `update_on_post`, pointing callers
  to `.tocsr()` for a materialized, plastic representation (#161).
- **PEP 561 inline type information**: ships a `py.typed` marker so downstream
  type checkers consume `brainevent`'s annotations. Public-API type hints and
  NumPy-style docstrings were completed across the package, guarded by a mypy
  CI ratchet (#151).

### Changed

- **`FixedNumConn` conversion methods renamed to the no-underscore canonical
  form** (scipy / `saiunit` convention): `to_csr` → `tocsr`, `to_csc` → `tocsc`,
  `to_dense` → `todense`. **Breaking** — no aliases are kept (#148, #161).
- **`CSC.__getitem__` now returns row `i` of `W`** (NumPy semantics) instead of
  column `i`. **Breaking** for code relying on the previous column-indexing
  behavior (#145).
- **`brainstate` dropped from the core import path**: importing `brainevent` no
  longer imports `brainstate`, removing it as an implicit runtime dependency of
  the core package (#159).
- **Documentation reorganized into the Diátaxis structure** (tutorials / how-to /
  reference / explanation); the README was updated to match the current public
  API (#149, #152, #155).
- **Internal CSR / JIT kernel layout**: `_jit_conn_csr` split into per-distribution
  submodules, with JIT-matrix `.tocsr()` backed by dedicated CPU / CUDA operators
  (#153, #160).

### Deprecated

- **Backward-compatibility shim for every v0.0.7 public name.** A module-level
  `__getattr__` keeps the entire v0.0.7 import surface resolvable. Renamed symbols
  emit a `DeprecationWarning` and forward to their replacement (slated for removal
  in a future major release):
  `EventArray` → `BinaryArray`;
  `JITCHomoR` / `JITCHomoC` → `JITCScalarR` / `JITCScalarC`;
  `FixedPostNumConn` / `FixedPreNumConn` → `FixedNumPerPre` / `FixedNumPerPost`;
  `csr_on_pre` / `csr2csc_on_post` / `dense_on_pre` / `dense_on_post` → the
  corresponding `update_*_on_binary_*` functions. Names whose functionality was
  removed — the `COO` class & operators, the `bitpack_` / `compact_` FCN kernels,
  and `EllLayout` / `CscLayout` — raise an `AttributeError` that names the
  replacement instead of failing silently.

### Removed

- **`COO` sparse format class and its operators** removed; accessing them now
  raises a guided `AttributeError`. Use `CSR` / `CSC` together with the `coo2csr`
  helper and the `*_index` conversion utilities (`csr_to_coo_index`,
  `coo_to_csc_index`, `csr_to_csc_index`, `csc_to_csr_index`) for index
  manipulation (#124).
- **Explicit `bitpack_` / `compact_` FCN kernels** removed; they were unified into
  `fcnmv` / `fcnmm`, which dispatch on the input event type. Wrap spikes with
  `BitPackedBinary` / `CompactBinary` and call `fcnmv` / `fcnmm`.
- **`FixedNumConn.to_csr` / `to_csc` / `to_dense`** (added and renamed within the
  0.1.0 cycle, never shipped in a release) standardized to `tocsr` / `tocsc` /
  `todense` (#148, #161).
- **cuSPARSE-based CSR SpMV / SpMM kernel implementations** removed in favor of
  the native CUDA / JAX kernel paths (internal; no public-API change).

## [0.0.7] - 2026-03-12

### Added

- **CUDA kernel compilation pipeline (`cuda_raw` backend)**: Native nvcc-based compilation system. Compile `.cu` files on-the-fly with source-hash caching, automatic XLA FFI registration, and multi-dtype dispatch (f16, bf16, f32, f64). Key APIs: `load_cuda_file`, `load_cuda_inline`, `load_cuda_dir`, `load_cpp_file`, `load_cpp_inline` (#88)
- **BitPacked binary event representations**: `BitPackedBinary` compresses 32 spike values into a single uint32 word (32x memory reduction). `CompactBinary` combines bitpacking with stream compaction to skip inactive rows in scatter kernels. Factory methods: `BitPackedBinary.from_array(x)`, `CompactBinary.from_array(x)`, and standalone `bitpack()` utility (#97)
- **BitPack FCN kernels**: `bitpack_binary_fcnmv`, `bitpack_binary_fcnmm`, `compact_binary_fcnmv`, `compact_binary_fcnmm` with both Numba CPU and CUDA GPU backends for event-driven matmul on packed spike representations (#97)
- **Parallel RNN training (`brainevent.pararnn`)**: O(log T) parallel training via Newton's method and parallel prefix reduction. Includes `parallel_rnn()` single-function API, `AutoRNNCell` with automatic Jacobian structure detection (diagonal, block-diagonal, dense), pre-built cells (`GRUDiagMH`, `LSTMCIFGDiagMH`), fused CUDA kernels for GRU/LSTM forward and backward passes, and configurable Newton solver (#85)
- **Warp kernel support** for CSR matrix-vector multiplication and various binary/sparse operations across COO, CSR, Dense, and FCN modules (#86)
- **Shared CUDA headers** (`brainevent/include/`): `common.h` (`BE::Tensor`, `BE::DType`, error-check macros), `cuda_common.h` (warp reductions, dtype macros, atomics), `dispatch.h` (type dispatch macros) for consistent CUDA kernel development
- **CUDA compilation diagnostics**: `print_diagnostics()`, `get_cache_dir()`, `set_cache_dir()`, `clear_cache()` for cache management; `CompiledModule`, `register_ffi_target`, `list_registered_targets` for FFI target management
- Tutorials for custom GPU operators with Warp and Numba CUDA (#83)

### Changed

- **CUDA raw as default GPU backend**: All operations (COO, CSR, Dense, FCN, JIT*) now default to `cuda_raw` backend on GPU, with automatic fallback to numba/pallas when CUDA is unavailable (#94)
- **Namespace migration**: `brainevent.kernix` namespace moved into `brainevent._op` and re-exported directly under `brainevent.*` (e.g., `brainevent.load_cuda_file`). Old `kernix` namespace removed (#96)
- **Backend rename**: `"tvmffi"` backend renamed to `"cuda_raw"` throughout the codebase (#87, #96)
- **Versioned cache directory**: Compiled kernel cache moved from `~/.cache/brainevent/` to `~/.cache/brainevent/<version>/` to prevent cross-version incompatibilities
- **FCN kernel launch optimization**: Scatter/gather kernels switched from block-per-row (`<<<n_pre, 256>>>`) to thread-per-row (`<<<ceil(n_pre/256), 256>>>`) strategy for moderate n_conn (33–512), yielding up to 6.4x speedup on COBA benchmarks (#84, #97)
- **FCN interface streamlining**: Unified `fcnmv`/`fcnmm` dispatch to optimal kernel based on input type (dense, bitpacked, or compact) (#96)
- **JAX >= 0.9.1 compatibility**: Added JAX Zero init helper and refactored JVP utilities for forward compatibility (#93)
- **JIT/CSR CUDA module splitting**: Reorganized CUDA kernel files for JIT and CSR operations into separate modules with updated Warp kernel implementations (#86)

### Removed

- `sparse_float` module and all related operations
- `IndexedBinary1d`, `IndexedBinary2d`, `IndexedSpFloat1d`, `IndexedSpFloat2d` classes (replaced by bitpack/compact representations)
- `brainevent.kernix` namespace (absorbed into `brainevent._op`, re-exported at top level)
- `ell_mv` function (superseded by FCN operations)

### Fixed

- **Binary FCN CUDA kernel correctness**: Fixed kernel launch parameter issues causing incorrect results in scatter/gather operations (#87)
- **Warp tile operation bug in JIT modules**: Cooperative tile ops produced diagonal-like output when launch dimensions < 32; replaced with scalar loops (#86)
- **CSR matrix-vector multiplication tolerance**: Enhanced assertion tolerance for numerical stability in tests

## [0.0.6] - 2026-02-14

### Added

- **`DataRepresentation` base class** with buffer registry for mutable named state on sparse matrices (`register_buffer`, `set_buffer`, `buffers`), plus `JITCMatrix` with full operator overloading (`__mul__`, `__add__`, `apply`, `apply2`, etc.) (#81)
- **CSR/CSC row slicing** via `csr_slice_rows` with full autodiff support (JVP, transpose, batching) and three backends (numba, warp, pallas); enables `csr[row_indices]` and `csc[col_indices]` indexing (#80)
- **SDDMM helpers** (`sddmm_indices`, `sddmm_coo_indices`, `sddmm_bcoo`) for Sampled Dense-Dense Matrix Multiplication built on `jax.experimental.sparse` (#75)
- **Primitive registry** (`get_registry`, `get_primitives_by_tags`, `get_all_primitive_names`) with automatic registration of all `XLACustomKernel` instances (#65)
- **User backend configuration** (`brainevent/config.py`) with JSON persistence, per-primitive default backend selection, Numba threading config, and LFSR algorithm selection (#65, #74)
- **CLI tool** (`brainevent benchmark-performance`) for automated benchmarking across backends with tabular output and automatic optimal-default persistence (#65)
- **Configurable LFSR RNG** for both Numba (`_numba_random.py`) and Pallas (`_pallas_random.py`) with three algorithm families: LFSR88 (~2^88 period), LFSR113 (~2^113 period), LFSR128 (~2^128 period) (#74)
- **TPU backend support** for CSR operations (#72)
- **Event representation classes**: `IndexedBinary1d/2d`, `IndexedSpFloat1d/2d` for indexed subsets of events, with `binary_array_index()` extraction function
- **Fixed-connection matmul helpers** (`binary_fcnmv/mm`, `fcnmv/mm`) and JITC matmul helpers for scalar/normal/uniform connectivity (#61)
- **`namescope` JAX decorator** for per-backend JIT compilation caching (#62)
- **Custom error types**: `KernelNotAvailableError`, `KernelCompilationError`, `KernelFallbackExhaustedError`, `KernelExecutionError`
- Tutorial on BinaryArray usage and optimization techniques (#64)

### Changed

- **Major codebase restructuring**: flat modules reorganized into coherent subpackages (`_coo/`, `_csr/`, `_dense/`, `_fcn/`, `_jit_scalar/`, `_jit_normal/`, `_jit_uniform/`, `_event/`) (#59, #69)
- **Consistent function naming convention** across all operations: `binary_*mv/mm`, `*mv/mm`, `update_*_on_binary_pre/post`, with `_p` suffix for raw primitives (#62)
- **`EventArray` renamed to `BinaryArray`** across the entire codebase (backward-compatible alias retained)
- **JITC class renames**: `JITCHomoR/C` → `JITCScalarR/C`; module renames `_jitc_homo` → `_jit_scalar`, `_jitc_normal` → `_jit_normal`, `_jitc_uniform` → `_jit_uniform`
- **Pallas RNG class renames**: `LFSR88RNG` → `PallasLFSR88RNG`, `LFSR113RNG` → `PallasLFSR113RNG`; new factory `PallasLFSRRNG(seed)`
- **Plasticity function renames**: `csr_on_pre` → `update_csr_on_binary_pre`, `coo_on_pre` → `update_coo_on_binary_pre`, etc. (backward-compatible aliases for CSR/dense variants)
- **Configuration system**: replaced `_config.py` singleton with `config.py` module using JSON file persistence
- `XLACustomKernel` enhanced with `def_tags()`, `def_benchmark_data()`, `benchmark()`, `available_backends()`, `set_default()`, and `KernelEntry` dataclass
- `csrmv_yw2y` moved to its own module `_csr/yw2y.py` (#79)
- Unified sparse-float dense matmul operations across all formats (#77)
- Project description updated to "Enabling Event-driven Computation in CPU/GPU/TPU"
- Added Python 3.14 support; dropped Python 3.10 from classifiers
- Core dependency `jax>=0.5.0` now explicitly required

### Fixed

- **Pallas GPU `binary_densemm` kernel corruption**: `pl.ds()` out-of-bounds reads when `block_dim > m` corrupted adjacent GPU memory; fixed with scalar `pl.program_id()` indexing and `jnp.where` instead of `jax.lax.cond` (#71)
- **Warp tile operation bug**: cooperative tile ops (`tile_load`, `tile_store`, `tile_atomic_add`) produced diagonal-like output when launch dimensions < 32 threads; replaced with scalar loops in `_jit_normal/float.py` (#71)
- **Backend passthrough in AD rules**: JVP/transpose/batching rules now correctly forward `backend=` parameter to `*_p_call()` functions, preventing silent use of wrong backend for tangent computation (#72)
- Fixed-connection matmul return values (#62)
- Bool-to-float conversion added in `binary_densemm_p_call` before passing to primitive (#71)

### Removed

- `BlockCSR` class and `_block_csr` module
- `BlockELL` class and `_block_ell` module
- `BaseArray`, `BinaryArrayIndex`, `MaskedFloat`, `MaskedFloatIndex` classes (replaced by new event representations)
- `GPUKernelChoice`, `pallas_kernel`, `warp_kernel` from `_op`
- `_primitives.py` module (replaced by `_registry.py`)

## [0.0.5] - 2025-12-25

### Added
- SDDMM (Sampled Dense-Dense Matrix Multiplication) functionality with COO indices
- Numba FFI backend for CPU custom kernels (#56)
- Warp FFI backend for GPU custom kernels (#56)
- STDP (Spike-Timing-Dependent Plasticity) tutorial documentation (#53)

### Changed
- Refactored package layout and module organization (#56)
- Updated package structure for improved modularity
- Refactored binary and float implementation modules

### Removed
- Original BrainPy content that was deprecated (#55)

### Fixed
- Updated image source in README to use raw.githubusercontent.com for proper display

## [0.0.4] - 2025-08-07

### Added
- Centralized primitives registry module for managing JAX primitives (#45)
- BlockCSR class with matrix multiplication, transpose, and other methods (#42, #47)
- Synaptic weight update operations for sparse matrices in COO, CSR, and CSC formats (#44)
- Sparse indexed arrays: `BinaryArrayIndex` and `MaskedFloatIndex` classes (#43)
- `__hash__` method to ArrayBase for supporting hashable arguments (#46)
- Weighted sparse matrix-vector multiplication `csrmv_yw2y` for CSR/CSC (#41)
- Diagonal position handling and updates for CSR/CSC matrices (#40)
- CSR/CSC sparse solve operations (#36)
- Support for warp-lang 1.9.0+ (#52)
- Daily CI workflow for improved testing coverage (#27)

### Changed
- Refactored BaseArray from classes to pure functions (#43)
- Updated BlockCSR methods for improved clarity and performance (#47)
- Enhanced type hints throughout the codebase (#27)
- Improved weight and dtype checking with relaxed test tolerances (#35, #37)
- Updated EINet class to use brainpy and braintools
- Updated logo and branding (#50)

### Fixed
- CSR solve test tolerances for numerical stability (#37)
- CI configuration to use development requirements for CPU installation

## [V0.1.0] - 2025-05-02 — GitHub tag only, never published to PyPI

> **Historical note:** The `V0.1.0` git tag was published on GitHub on 2025-05-02
> but was **never released to PyPI**. The PyPI distribution line continued as
> `0.0.1.postN` → `0.0.2` … `0.0.7`; the first `0.1.0` published to PyPI is the
> entry dated 2026-06-07 at the top of this file. This section is retained for
> historical accuracy.

### Added
- Just-In-Time Connectivity (JITC) matrix operators for CSR format (#18)
  - `JITCHomoR`, `JITCHomoC`: Homogeneous weight matrices
  - `JITCNormalR`, `JITCNormalC`: Normal distribution weight matrices
  - `JITCUniformR`, `JITCUniformC`: Uniform distribution weight matrices
- Pallas kernel implementations for GPU/TPU backends (#28, #30)
- Tiled Pallas kernels for JITC operators (#30)
- JVP/transpose rules for JITC `todense()` operations on random matrices (#29)
- Fixed connection number matrix operations (#25, #31)
  - `FixedPostNumConn`: Fixed number of post-synaptic connections
  - `FixedPreNumConn`: Fixed number of pre-synaptic connections
- BinaryArray and MaskedFloat classes with optimized dense/sparse operations (#34)
- Event-driven dense matrix operations (#24)
- COO (Coordinate) sparse matrix implementation with spmv and spmm operators (#7, #15)
- CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column) implementations (#26)
- Load-balanced CSR/CSC classes (`CSR_LB`, `CSC_LB`) for improved performance (#11)
- Lazy-loading for 'nn' submodule (#16)
- Enhanced CSR implementation with Pallas and improved benchmarks (#26)

### Changed
- Unified kernel API with direct functions instead of classes (#33)
- Unified configuration management with Config singleton (#32)
- Improved GPU/TPU backend selection for JITC operators (#28)
- Refactored COO and CSR implementations with new type aliases for readability (#14)
- Integrated general batching rule for all operator implementations (#13)
- Enhanced BinaryArray with additional built-in functions (#5, #24)
- Restructured brainevent module documentation (#21)
- Improved code formatting and replaced deprecated references (#22)

### Added - Infrastructure
- Compatibility layer for JAX version handling and custom call registration (#12)
- Development dependencies: absl-py for enhanced functionality
- DOI badge from Zenodo (10.5281/zenodo.15324450)

### Removed
- Deprecated code for improved JAX compatibility (#19)
- Unnecessary files from project structure

### Fixed
- Event handling and linear computation for improved performance and readability (#17)
- Updated documentation and CI configuration (#20)

## [0.0.1] - Initial Release

### Added
- Initial project structure and setup
- Basic CSR matrix operations
- CSR float tests
- CSRMM (CSR Matrix-Matrix multiplication) VJP and JVP rules (#1)
- Basic BinaryArray implementation
- FixedPostNumConn event and float implementations (#4)
- BinaryArray built-in functions
- CSR spmv gradient computation (#5)
- README and project documentation (#3, #6)

### Changed
- Upgraded project structure (#2)
- Updated FixedPostNumConn implementation (#4, #5)

---

## Version Comparison Links

- [0.2.1](https://github.com/chaobrain/brainevent/compare/v0.2.0...v0.2.1)
- [0.2.0](https://github.com/chaobrain/brainevent/compare/v0.1.2...v0.2.0)
- [0.1.2](https://github.com/chaobrain/brainevent/compare/v0.1.1...v0.1.2)
- [0.1.1](https://github.com/chaobrain/brainevent/compare/v0.1.0...v0.1.1)
- [0.1.0](https://github.com/chaobrain/brainevent/compare/v0.0.7...v0.1.0)
- [0.0.7](https://github.com/chaobrain/brainevent/compare/v0.0.6...v0.0.7)
- [0.0.6](https://github.com/chaobrain/brainevent/compare/v0.0.5...v0.0.6)
- [0.0.5](https://github.com/chaobrain/brainevent/compare/V0.0.4...v0.0.5)
- [0.0.4](https://github.com/chaobrain/brainevent/compare/V0.1.0...V0.0.4)
- [V0.1.0 — GitHub tag, 2025-05-02, never published to PyPI](https://github.com/chaobrain/brainevent/releases/tag/V0.1.0)
