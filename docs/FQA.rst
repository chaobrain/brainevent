Frequently Asked Questions
==========================

.. contents:: Questions
   :local:
   :depth: 1


Which CUDA backend should I use: ``brainevent.kernix`` or ``jax-tvm-ffi``?
--------------------------------------------------------------------------

BrainEvent provides two ways to define custom CUDA kernels:

- **brainevent.kernix** — compiles CUDA source via
  ``nvcc`` and registers the result as an XLA FFI target.
- **jax-tvm-ffi** — JIT-compiles CUDA source at
  runtime via NVRTC using the TVM FFI infrastructure.

**Use** ``kernix`` **for all new kernels.**  The table below explains why.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - ``kernix``
     - ``jax-tvm-ffi``
   * - Compiler
     - ``nvcc`` (full CUDA toolkit)
     - NVRTC (runtime compilation)
   * - float64 support
     - Yes
     - Broken in ``jax_tvm_ffi <= 0.1.2`` (`upstream issue <https://github.com/NVIDIA/jax-tvm-ffi/issues/13>`_)
   * - ``__shared__`` in ``__device__`` functions
     - Supported
     - Causes segfault under NVRTC — must use ``extern __shared__`` in ``__global__`` kernels only
   * - Host-side ``data_ptr()`` access
     - N/A (XLA FFI buffers)
     - SIGSEGV if dereferenced on host (GPU pointer)
   * - Disk cache
     - Yes — hash-based, survives process restarts
     - No (TVM may cache PTX internally, but no persistent artifact cache)
   * - CUDA Graph support
     - Yes (``allow_cuda_graph=True``)
     - Yes (``allow_cuda_graph=True``)
   * - External dependency
     - None beyond CUDA toolkit
     - ``jax-tvm-ffi`` + ``tvm_ffi.cpp``
   * - nvcc at deployment
     - Required
     - Not required (NVRTC ships with CUDA runtime)
   * - Kernel language features
     - Full C++17 + CUDA
     - NVRTC subset (no separate compilation, limited headers)

**Why** ``kernix`` **wins for BrainEvent:**

1. **Known TVM FFI bugs have caused real issues** in this codebase: the
   float64 dtype-mapping bug, SIGSEGV from host-side ``data_ptr()``
   dereference, and segfaults from ``static __shared__`` inside
   ``__device__`` functions.  These restrictions do not exist with ``nvcc``.

2. **No extra dependency.**  ``jax-tvm-ffi`` and ``tvm_ffi.cpp`` are external
   packages with their own versioning risk.  ``kernix`` only needs the
   CUDA toolkit that GPU users already have.

3. **Disk cache.**  ``kernix`` caches compiled ``.so`` files keyed by a
   hash of the source, architecture, and compiler flags.  Subsequent imports
   are instant — no recompilation.

4. **Full language support.**  Complex kernels (templates, cooperative groups,
   multi-dtype macros, rich ``__shared__`` usage) work without workarounds.

**When** ``jax-tvm-ffi`` **is acceptable:**

The only practical advantage of TVM FFI is that end users do not need ``nvcc``
at runtime, only the CUDA runtime library.  If you are writing a simple kernel
and deployment without the full CUDA toolkit is a hard requirement, TVM FFI
remains an option — but be aware of its restrictions.

Existing kernels in BrainEvent that already use TVM FFI can be migrated to
``kernix`` opportunistically when those files are touched.


How do I write a new CUDA kernel with ``kernix``?
--------------------------------------------------------

Place the kernel in a co-located ``.cu`` file and load it at import time:

.. code-block:: python

   # my_module/my_kernels.py
   from pathlib import Path
   from brainevent.kernix import load_cuda_file

   _module = load_cuda_file(
       Path(__file__).parent / "my_kernels.cu",
       target_prefix="my_module.my_kernels",
   )

Annotate each entry point in the ``.cu`` file with ``// @BE``:

.. code-block:: cuda

   // @BE my_kernel arg arg ret stream
   void my_kernel(const BE::Tensor& input,
                  const BE::Tensor& weights,
                  BE::Tensor& output,
                  cudaStream_t stream) {
       // kernel launch code
   }

``load_cuda_file`` compiles the kernel on first use, caches the ``.so`` to
disk, and registers it as a JAX FFI target.  Subsequent imports skip
compilation entirely.

See the tutorials under ``docs/tutorial/`` for complete examples.
