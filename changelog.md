# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Fixed-connection matmul helpers** (`binary_fcnmv/mm`, `fcnmv/mm`, `spfloat_fcnmv/mm`) and JITC matmul helpers for scalar/normal/uniform connectivity (#61)
- **`namescope` JAX decorator** for per-backend JIT compilation caching (#62)
- **Custom error types**: `KernelNotAvailableError`, `KernelCompilationError`, `KernelFallbackExhaustedError`, `KernelExecutionError`
- Tutorial on BinaryArray usage and optimization techniques (#64)

### Changed

- **Major codebase restructuring**: flat modules reorganized into coherent subpackages (`_coo/`, `_csr/`, `_dense/`, `_fcn/`, `_jit_scalar/`, `_jit_normal/`, `_jit_uniform/`, `_event/`) (#59, #69)
- **Consistent function naming convention** across all operations: `binary_*mv/mm`, `*mv/mm`, `spfloat_*mv/mm`, `update_*_on_binary_pre/post`, with `_p` suffix for raw primitives (#62)
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

## [0.1.0] - 2025-05-02

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

- [0.0.6](https://github.com/chaobrain/brainevent/compare/v0.0.5...v0.0.6)
- [0.0.5](https://github.com/chaobrain/brainevent/compare/V0.0.4...v0.0.5)
- [0.0.4](https://github.com/chaobrain/brainevent/compare/V0.1.0...V0.0.4)
- [0.1.0](https://github.com/chaobrain/brainevent/releases/tag/V0.1.0)
