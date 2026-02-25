Compiler Options
================

``brainevent.kernix`` exposes nvcc optimization options directly through
``load_cuda_inline`` (and ``load_cuda_file`` / ``load_cuda_dir``, which
forward all keyword arguments).

Optimization Level
------------------

The ``optimization_level`` parameter controls the ``-O<n>`` flag passed to
nvcc.  It applies to both host-side C++ code (via the underlying host
compiler) and device-side PTX generation.

.. code-block:: python

   from brainevent import kernix

   # Default: -O3 (recommended for production)
   mod = kernix.load_cuda_inline(
       name="my_kernels",
       cuda_sources=CUDA_SRC,
       functions={"my_func": ["arg", "ret", "stream"]},
   )

   # Debug build: -O0 (preserves variable values for cuda-gdb / Nsight)
   mod = kernix.load_cuda_inline(
       ...,
       optimization_level=0,
   )

   # Explicit production build: -O3
   mod = kernix.load_cuda_inline(
       ...,
       optimization_level=3,
   )

Each value corresponds directly to an nvcc ``-O`` flag:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Level
     - Effect
   * - ``0``
     - No optimization. Useful for debugging with cuda-gdb or Nsight Compute.
   * - ``1``
     - Basic optimizations.
   * - ``2``
     - Standard optimizations.
   * - ``3``
     - Aggressive optimizations, including auto-vectorization and loop
       unrolling (default).

Fast Math
---------

The ``use_fast_math`` flag passes ``--use_fast_math`` to nvcc.  This is a
compound flag that enables several device-code optimizations:

- ``-ftz=true`` — flush denormal floats to zero
- ``-prec-div=false`` — approximate (faster) division
- ``-prec-sqrt=false`` — approximate (faster) square root
- ``-fmad=true`` — fused multiply-add (FMA) contraction

These trade IEEE 754 compliance for speed.  Typical speed-up is **10–30 %**
on floating-point-heavy kernels (GEMM, reductions, activations).

.. code-block:: python

   # Enable fast math — safe for most ML kernels
   mod = kernix.load_cuda_inline(
       name="my_fast_kernels",
       cuda_sources=CUDA_SRC,
       functions={"my_func": ["arg", "ret", "stream"]},
       use_fast_math=True,
   )

.. warning::

   ``use_fast_math`` can change numerical results.  Division and square root
   may differ from IEEE 754 by a few ULPs.  Denormals are flushed to zero,
   which can cause underflow for very small values.  Validate your kernel
   results before enabling this in production.

Extra Flags
-----------

For flags not covered by the named parameters, use ``extra_cuda_cflags``:

.. code-block:: python

   mod = kernix.load_cuda_inline(
       ...,
       extra_cuda_cflags=[
           "--generate-line-info",         # source-line info for Nsight profiling
           "-maxrregcount=64",             # cap register usage to raise occupancy
           "--ptxas-options=-v",           # verbose PTX assembler stats
           "-Xcompiler", "-march=native",  # native CPU optimisation for host code
       ],
   )

CUDA Graph Support
------------------

`CUDA Graphs <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs>`_
(called *command buffers* in XLA) let the GPU driver record a sequence of
kernel launches once and replay the recording cheaply, eliminating CPU-side
kernel-launch overhead on every call.  JAX captures CUDA graphs automatically
for kernels registered with the ``COMMAND_BUFFER_COMPATIBLE`` XLA trait.

Pass ``allow_cuda_graph=True`` to opt in (this is the **default**):

.. code-block:: python

   mod = kernix.load_cuda_inline(
       name="my_kernels",
       cuda_sources=CUDA_SRC,
       functions={"my_func": ["arg", "ret", "stream"]},
       allow_cuda_graph=True,   # default — no need to pass explicitly
   )

Opt out only for kernels with host-side side effects during replay:

.. code-block:: python

   mod = kernix.load_cuda_inline(
       ...,
       allow_cuda_graph=False,
   )

.. warning::

   Set ``allow_cuda_graph=False`` if the kernel has **host-side side effects
   during replay**: dynamic memory allocation (``cudaMalloc``), host callbacks,
   or non-deterministic resource usage.  Plain element-wise, reduction, and
   GEMM kernels are all safe with the default ``True``.

Combining All Options
---------------------

.. code-block:: python

   mod = kernix.load_cuda_inline(
       name="peak_perf",
       cuda_sources=CUDA_SRC,
       functions={"my_func": ["arg", "ret", "stream"]},
       optimization_level=3,    # -O3 (default)
       use_fast_math=True,      # ~10-30% faster FP ops, relaxed precision
       # allow_cuda_graph=True  # default — no need to pass explicitly
   )

Caching Behaviour
-----------------

``optimization_level`` and ``use_fast_math`` are part of the cache key.
Changing them for the same source triggers a recompilation and stores a
separate cached binary — no need for ``force_rebuild=True``.

``allow_cuda_graph`` is a registration-only flag and does **not** affect
the cache key or trigger a rebuild.
