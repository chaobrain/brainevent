# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

A hardening pass over the custom-operator registration machinery
(`brainevent._op`), fixing the 19 defects catalogued in the 2026-07-16
operator-registration audit (`dev/2026-07-16-op-registration-audit.md`):
stale backend dispatch after runtime backend switches, silently dropped
JVP rules, incomplete compilation-cache keys, order-dependent FFI target
names, and incorrect `vmap` execution of `numba.cuda` kernels, among
others.

### Added

- **jax 0.11.x is now a validated version.** The numba XLA FFI bridge raises its
  validated ceiling (`_MAX_VALIDATED_JAX`) from `0.10` to `0.11`, so installing
  `brainevent` alongside jax 0.11 no longer emits the "untested jax" `RuntimeWarning`.
  jaxlib 0.11 reports the same `XLA_FFI_API` version (`0.3`) as 0.10, meaning the
  hand-mirrored `ffi.h` struct layout is unchanged; the full test suite passes on
  jax 0.11.0 on both the CPU and CUDA backends. The `jax>=0.8.0` floor is unchanged.

### Fixed

- **Backend switches now take effect immediately.**
  `XLACustomKernel.set_default`, `brainevent.config.set_backend`, and
  `clear_backends` invalidate JAX's dispatch and executable caches
  (`jax.clear_caches()`) whenever the effective setting changes.
  Previously, eager calls and warm `jax.jit` functions kept executing the
  previously selected backend. Note the invalidation is process-global:
  the next call of every jitted function recompiles.
- **`defjvp` rejects mismatched rule arity.** Registering a number of JVP
  rules different from the primitive's number of inputs now raises
  `ValueError` at differentiation time instead of silently dropping
  trailing gradients (previously `zip` truncation produced wrong, silent
  results). A multi-result JVP rule returning a bare array instead of a
  sequence now raises `TypeError`. One latent in-tree mismatch
  (`binary_fcnmm_p`: four rules for three inputs) was corrected.
- **`vmap` over `numba.cuda` kernels computes correct results.** Batched
  calls now execute one kernel launch per batch slice with the kernel's
  original launch configuration, instead of reusing the launch grid of
  the unbatched shape over folded buffers (which silently corrupted any
  kernel that couples rows, e.g. stencils and reductions). Kernels wrapped
  with an explicit `grid=` cannot be batched; combining `grid=` with
  `vmap_method=` raises `ValueError` at wrap time. Only one `vmap` level is
  supported: nested `vmap` now raises a clear error instead of returning
  uninitialized memory for all but the first slice.
- **Compilation-cache keys cover everything that affects codegen.** The
  kernix (inline C++/CUDA) cache key now includes the resolved
  `FunctionSpec`s and the content of user-provided extra include headers
  (key schema v2 — old cache entries are recompiled once, not misused).
  The `numba` CPU FFI memo no longer keys on array shapes, so one kernel
  serves all shapes of the same dtype signature.
- **FFI target names are content-derived.** CPU and CUDA numba kernels
  register under a fingerprint of the kernel's bytecode, constants,
  closure values, and referenced globals rather than a process-order
  counter, making `jax.export` artifacts stable across processes. Kernels
  whose content cannot be fingerprinted deterministically fall back to
  per-process counter names.
- Unknown, packed sub-byte (`S1`–`S4`, `U1`–`U4`, `F4E2M1FN`), and FP8
  buffer dtypes now raise a descriptive `ValueError` instead of being
  reinterpreted as raw bytes; `bfloat16` is rejected explicitly on the
  numba paths. XLA FFI extension chains are walked fully, and FFI error
  objects are destroyed after use.
- CUDA output buffers for kernels that accumulate are zero-filled on
  XLA's stream (previously uninitialized memory could leak into results).
  Transient CUDA probe failures no longer permanently disable the
  `numba.cuda` backend for the process.
- Kernel construction/compile failures during lowering now raise
  `KernelCompilationError` (with the original exception as `__cause__`
  and the remaining registered backends listed); calling a kernel on a
  platform with no registered backend raises
  `KernelFallbackExhaustedError` naming the platforms that are
  registered. Both are exported from `brainevent`.

### Changed

- **Re-registering an FFI target with different content now raises
  `KernelRegistrationError` on every platform** (including
  `load_cuda_inline(..., replace=True)` / `force_rebuild=True` with
  changed source). Live re-pointing of an already-registered XLA FFI
  target is not supported by JAX (CPU raises; CUDA silently keeps the old
  handler), so brainevent refuses deterministically instead of silently
  dispatching stale code — register under a new `name=` to iterate on a
  kernel within one process. Re-registration of *unchanged* source (e.g.
  `force_rebuild=True` twice) is an idempotent no-op: registration identity
  is the deterministic compilation cache key, not the compiler's output
  bytes.
- Registering a second primitive under an existing name emits a
  `UserWarning` (the new registration still wins, as before).
- **`CONTRIBUTING.md` rewritten.** It previously described *BrainPy* and linked to a
  page that returns HTTP 404. It is now a self-contained `brainevent` guide covering
  development setup, the test/mypy/pre-commit gates, docs builds, code style, the pull
  request checklist, and GPU kernel contributions.
- **`SECURITY.md` rewritten.** Vulnerability reports now go through GitHub private
  vulnerability reporting or email instead of public issues, and the policy documents
  supported versions, response targets, and the trust boundary around the runtime
  C++/CUDA compilation APIs (`load_cpp_inline`, `load_cuda_inline`, and friends).
- **`CODE_OF_CONDUCT.md` upgraded** from Contributor Covenant 2.1 to 3.0.
- **`.gitattributes` expanded** to cover the header, reStructuredText, notebook, YAML,
  TOML and image file types actually present in the tree, with language-aware diff
  drivers, explicit binary markers, GitHub language-statistics hints, and
  `export-ignore` rules for development-only infrastructure.

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

- [0.1.1](https://github.com/chaobrain/brainevent/compare/v0.1.0...v0.1.1)
- [0.1.0](https://github.com/chaobrain/brainevent/compare/v0.0.7...v0.1.0)
- [0.0.7](https://github.com/chaobrain/brainevent/compare/v0.0.6...v0.0.7)
- [0.0.6](https://github.com/chaobrain/brainevent/compare/v0.0.5...v0.0.6)
- [0.0.5](https://github.com/chaobrain/brainevent/compare/V0.0.4...v0.0.5)
- [0.0.4](https://github.com/chaobrain/brainevent/compare/V0.1.0...V0.0.4)
- [V0.1.0 — GitHub tag, 2025-05-02, never published to PyPI](https://github.com/chaobrain/brainevent/releases/tag/V0.1.0)
