# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes yet._

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

- [0.1.0](https://github.com/chaobrain/brainevent/compare/v0.0.7...v0.1.0)
- [0.0.7](https://github.com/chaobrain/brainevent/compare/v0.0.6...v0.0.7)
- [0.0.6](https://github.com/chaobrain/brainevent/compare/v0.0.5...v0.0.6)
- [0.0.5](https://github.com/chaobrain/brainevent/compare/V0.0.4...v0.0.5)
- [0.0.4](https://github.com/chaobrain/brainevent/compare/V0.1.0...V0.0.4)
- [V0.1.0 — GitHub tag, 2025-05-02, never published to PyPI](https://github.com/chaobrain/brainevent/releases/tag/V0.1.0)
