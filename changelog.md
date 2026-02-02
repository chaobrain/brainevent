# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - Version 0.0.5

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
- Enhanced EventArray with additional built-in functions (#5, #24)
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
- Basic EventArray implementation
- FixedPostNumConn event and float implementations (#4)
- EventArray built-in functions
- CSR spmv gradient computation (#5)
- README and project documentation (#3, #6)

### Changed
- Upgraded project structure (#2)
- Updated FixedPostNumConn implementation (#4, #5)

---

## Version Comparison Links

- [Unreleased](https://github.com/chaobrain/brainevent/compare/V0.0.4...HEAD)
- [0.0.4](https://github.com/chaobrain/brainevent/compare/V0.1.0...V0.0.4)
- [0.1.0](https://github.com/chaobrain/brainevent/releases/tag/V0.1.0)
