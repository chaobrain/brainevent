# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BrainEvent is a Python library for event-driven sparse computation in spiking neural networks, targeting CPU/GPU/TPU via
JAX. It provides specialized sparse matrix formats and operations optimized for binary (spike) events and synaptic
weight updates.

## Build & Test Commands

```bash
# Install for development (CPU)
pip install -r requirements-dev-cpu.txt && pip install . --no-cache-dir

# Install for development (GPU)
pip install -r requirements-dev-gpu.txt && pip install ".[cuda13]" --no-cache-dir

# Run all tests
pytest brainevent/

# Run tests for a specific module
pytest brainevent/_csr/

# Run a single test file
pytest brainevent/_csr/binary_test.py

# Run a single test
pytest brainevent/_csr/binary_test.py::test_function_name

# Linting (pre-commit hooks: end-of-file-fixer, debug-statements, trailing-whitespace, flake8)
pre-commit run --all
```

## Architecture

### Computation Backends

Each operation supports multiple backends selected at runtime:

- **numba** — CPU JIT compilation (default on CPU)
- **pallas** — JAX Pallas kernels for GPU/TPU

Backend config is persisted per-primitive in `~/.config/brainevent/defaults.json` (Linux). See `brainevent/config.py`.

### Sparse Matrix Formats

Each format lives in its own subpackage under `brainevent/`:

| Subpackage      | Format                                | Class(es)                                             |
|-----------------|---------------------------------------|-------------------------------------------------------|
| `_coo/`         | Coordinate                            | `COO`                                                 |
| `_csr/`         | Compressed Sparse Row/Col             | `CSR`, `CSC`                                          |
| `_dense/`       | Dense matrix                          | (no class, operations only)                           |
| `_fcn/`         | Fixed Number Connectivity             | `FixedNumConn`, `FixedPreNumConn`, `FixedPostNumConn` |
| `_jit_scalar/`  | JITC with scalar weights              | `JITCScalarR`, `JITCScalarC`                          |
| `_jit_normal/`  | JITC with normal-distributed weights  | `JITCNormalR`, `JITCNormalC`                          |
| `_jit_uniform/` | JITC with uniform-distributed weights | `JITCUniformR`, `JITCUniformC`                        |

### Module Internal Structure (repeated per format)

Each subpackage follows a consistent layout:

- `main.py` — Sparse matrix class definition (JAX pytree, `tree_flatten`/`tree_unflatten`)
- `binary.py` — Event-driven (binary spike) matmul kernels (`binary_*mv`, `binary_*mm`)
- `float.py` — Float-valued matmul kernels (`*mv`, `*mm`)
- `plasticity_binary.py` — Synaptic plasticity / weight update rules (`update_*_on_binary_pre/post`)
- `sparse_float.py` — Sparse float operations (where applicable)
- `test_util.py` — Reference (naive) implementations for validation
- `*_test.py` — Tests

### Function Naming Convention

- `binary_*mv` / `binary_*mm` — Event-driven (binary) matrix-vector / matrix-matrix multiply
- `*mv` / `*mm` — Float-valued matrix-vector / matrix-matrix multiply
- `spfloat_*mv` / `spfloat_*mm` — Sparse float operations
- `update_*_on_binary_pre` / `update_*_on_binary_post` — Plasticity weight updates
- `*_p` suffix — The raw JAX primitive version (lower-level); the unsuffixed version is the user-facing wrapper

### Class Hierarchy

```
brainunit.sparse.SparseMatrix  (sets self.shape)
  └── DataRepresentation       (adds buffer registry for mutable state)
        └── JITCMatrix          (adds unitary/binary ops, apply/apply2)
              ├── JITCScalarR/C
              ├── JITCNormalR/C
              └── JITCUniformR/C
```

`COO`, `CSR`, `CSC` extend `DataRepresentation` directly.

### Custom Kernel Registration (`_op/`)

- `_op/main.py` — `XLACustomKernel` base class for registering JAX primitives with multiple backend implementations (
  `KernelEntry` per backend)
- `_op/util.py` — Helpers: `defjvp`, `general_batching_rule`, type conversions
- `_op/numba_ffi.py` / `numba_cuda_ffi.py` — Numba CPU/CUDA FFI kernel registration

### Event Representations (`_event/`)

- `BinaryArray` — Wraps boolean arrays representing spikes
- `SparseFloat` — Sparse float event representation
- `Indexed*` variants — For pre-sliced/indexed subsets of events

### Key Files

- `brainevent/_compatible_import.py` — JAX version compatibility shims
- `brainevent/_misc.py` — Index conversion utilities (`csr_to_coo_index`, etc.), `COOInfo` namedtuple
- `brainevent/_pallas_random.py` — LFSR-based RNG for Pallas GPU kernels
- `brainevent/_registry.py` — Global primitive registry, lookup by name or tags

## GPU Kernel Pitfalls

- **Backend passthrough**: JVP/transpose/batching rules must forward `backend=` to `*_p_call()` functions, otherwise
  tangent computation may silently use the wrong backend.

- **TVM FFI — NEVER dereference `data_ptr()` on the host**: `TensorView::data_ptr()` returns a **GPU device memory
  pointer**. Dereferencing it from C++ host code (inside a TVM FFI entry function) causes an immediate SIGSEGV. The
  common offender is:
  ```c
  // WRONG — causes SIGSEGV: dereferences a GPU pointer from the CPU host
  bool is_homo = (weights.ndim() == 1);
  float homo_w = is_homo ? *static_cast<const float*>(weights.data_ptr()) : 0.0f;
  ```
  **The fix**: only read host-safe *metadata* (`ndim()`, `size(0)`, etc.) on the host; pass the raw device pointer to
  the kernel and let GPU threads read from it:
  ```c
  // CORRECT — metadata is host-safe; device ptr is passed to kernel, not dereferenced
  int is_homo = (weights.ndim() == 1) ? 1 : 0;          // metadata: OK on host
  const float* d_weights = static_cast<const float*>(weights.data_ptr()); // device ptr
  my_kernel<<<grid, block, shm, stream>>>(d_weights, ..., is_homo);
  // Inside the kernel, GPU threads read: weights[0] (homo) or weights[row*n_conn+k] (hetero)
  ```

- **NVRTC (TVM FFI) — no static `__shared__` in `__device__` functions**: TVM FFI uses NVRTC to JIT-compile CUDA code.
  NVRTC does **not** correctly handle `__shared__` variables declared inside `__device__` (non-kernel) functions — this
  produces a segfault at kernel launch time. **Always use `extern __shared__` (dynamic shared memory) in the
  calling `__global__` kernel.** Never write patterns like:
  ```c
  // WRONG — causes segfault with NVRTC/TVM FFI
  __device__ float block_reduce(float val) {
      __shared__ float smem[32];  // static __shared__ in __device__ function
      ...
  }
  ```
  Instead, inline the reduction directly in the `__global__` kernel using `extern __shared__`:
  ```c
  // CORRECT — allocate shared mem in kernel, use extern __shared__
  __global__ void my_kern(...) {
      extern __shared__ float smem_red[];  // allocated at launch: 32*sizeof(float)
      int lane = threadIdx.x & 31, warpid = threadIdx.x >> 5;
      val = warp_reduce_sum(val);
      if (lane == 0) smem_red[warpid] = val;
      __syncthreads();
      int n_warps = (blockDim.x + 31) >> 5;
      val = (threadIdx.x < n_warps) ? smem_red[lane] : 0.0f;
      if (warpid == 0) val = warp_reduce_sum(val);
  }
  // Launch with: <<<grid, 256, 32*sizeof(float), stream>>>
  ```

- **Existing TVM FFI bugs (jax_tvm_ffi<=0.1.2)**: 
  Float64 is not handled during dtype mapping (https://github.com/NVIDIA/jax-tvm-ffi/issues/13).
  This means that if you use float64 weights with GPU backends, the TVM FFI will fail to compile 
  the kernel due to an unsupported dtype. However, for jax_tvm_ffi>0.1.2, this will be fixed.


- **TVM FFI entry-point discovery is regex-based on raw source text**: `register_tvm_cuda_from_file` discovers FFI entry
  points by scanning the `.cu` source for `^void\s+(\w+)\s*\(` at column 0 with `tvm::ffi::TensorView` parameters.
  **C preprocessor macros are NOT expanded** — the scanner sees the raw text. If you use macros to generate FFI entry
  points (e.g. `FFI_GATHER_AUTO(_f32_bool, ...)`), the scanner will not find them. Two solutions:
  1. **Annotation comments** (preferred): place `// @tvm_ffi function_name` before each macro invocation:
     ```c
     // @tvm_ffi binary_densemv_gather_auto_f32_bool
     FFI_GATHER_AUTO(_f32_bool, float, int8_t, 32 * sizeof(float))
     ```
  2. **Explicit functions**: write the `void function_name(tvm::ffi::TensorView ...)` signatures directly (verbose but
     always works).

  Both mechanisms coexist — the parser merges results from explicit functions and `@tvm_ffi` annotations, deduplicating
  automatically. Existing `.cu` files with explicit functions continue to work unchanged.

- **Multi-dtype CUDA kernel pattern**: To support multiple weight dtypes (f16, bf16, f32, f64) in a single `.cu` file,
  use parameterized macros with `READ_W`/`WRITE_W` conversion functions and an `ACC_T` accumulator type:
  ```c
  #define READ_F16(x)   __half2float(x)
  #define WRITE_F16(x)  __float2half(x)

  #define DEFINE_GATHER_WARP(SUFFIX, SPIKE_T, IS_ACTIVE, WEIGHT_T, ACC_T, \
                             READ_W, WRITE_W, WARP_RED, ACC_ZERO) ...

  // Instantiate: f16 accumulates in f32 for stability
  DEFINE_GATHER_WARP(_f16_bool, int8_t, IS_ACTIVE_BOOL, __half, float,
                     READ_F16, WRITE_F16, warp_reduce_sum_f32, 0.0f)
  ```
  Float16 (`__half`) and bfloat16 (`__nv_bfloat16`) must accumulate in float32 for numerical stability. Float64
  (`double`) uses its own `warp_reduce_sum_f64` and accumulates natively. The Python-side dtype dispatch uses:
  ```python
  _dtype_sfx = {
      jnp.dtype('float16'): '_f16', jnp.dtype('float32'): '_f32',
      jnp.dtype('float64'): '_f64', jnp.dtype('bfloat16'): '_bf16',
  }
  wt_sfx = _dtype_sfx.get(jnp.dtype(kwargs['weight_info'].dtype), '_f32')
  ```

- **`extern __shared__` with multi-dtype block reduction**: When the accumulator type varies across dtype
  instantiations, use `extern __shared__ char _smem_bytes[]` with `reinterpret_cast<ACC_T*>` instead of declaring
  `extern __shared__ float/double`. This works universally across all accumulator types and avoids type conflicts
  in macro-generated kernels. Launch with the correct shared memory size: `32 * sizeof(ACC_T)`.


## Dev Script Path Fix

When running benchmark/dev scripts directly (e.g. `python dev/fcn/benchmark_fcnmv.py`), Python adds the **script's
directory** to `sys.path[0]` — not the project root. This causes Python to import the **installed** `brainevent` from
site-packages instead of the development version, silently hiding any local changes.

Fix: add this block at the top of every script in a `dev/` subdirectory:

```python
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
```

Or equivalently run with: `PYTHONPATH=/path/to/project python dev/subdir/script.py`

## Benchmarking Primitives

Prefer using the built-in `.benchmark()` method on `XLACustomKernel` primitives (e.g. `fcnmv_p`, `fcnmm_p`) rather than
writing custom timing loops. The pattern:

1. Define a data generator that `yield`s or returns `BenchmarkConfig` instances:
   ```python
   from brainevent import BenchmarkConfig

   def _my_benchmark_data(*, platform):
       for n in [1000, 5000, 10000]:
           ...
           yield BenchmarkConfig(
               name=f"NT,homo,{n}",
               args=(weights, indices, vector),
               kernel_kwargs={'shape': (n, n), 'transpose': False},
               data_kwargs={'n': n},
           )
   ```
2. Register it and call `.benchmark()`:
   ```python
   fcnmv_p.def_benchmark_data(_my_benchmark_data)
   result = fcnmv_p.benchmark(platform='gpu', n_warmup=10, n_runs=100, verbose=True)
   result.print(group_by='label', highlight_best=True)
   ```

This handles warmup, timing, cross-backend comparison, and tabular display automatically. See
`dev/fcn/benchmark_fcnmv.py` for a complete example.

## CUDA Source File Conventions (`.cu` files)

### File layout

Each `.cu` file must follow this structure, in order:

1. **License header** — Apache 2.0 using `//` C-style line comments (not `#`):
   ```c
   // Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
   //
   // Licensed under the Apache License, Version 2.0 (the "License");
   // ...
   // ==============================================================================
   ```
2. **Docstring block** — `/* ... */` block comment summarising the public Python API, parameters, and behaviour.
3. **CUDA kernel source** — `__device__`, `__global__`, and TVM FFI `void` entry functions.

### Keeping CUDA out of Python

Never embed large CUDA source strings inline in Python files. Instead:

- Store kernels in a co-located `.cu` file (e.g. `brainevent/_fcn/fcnmv.cu`).
- Load at runtime: `Path(__file__).parent.joinpath('fcnmv.cu').read_text()`.
- This keeps Python files readable, allows the CUDA to be edited/compiled independently, and is bundled in wheels via
  `pyproject.toml` (see below).


## Linter

A linter runs on file save and may revert changes. When making many edits to a single file, prefer writing the entire
file at once rather than incremental edits.
