The custom-kernel architecture
==============================

``brainevent`` lets you write performance-critical operators in plain C++ (CPU) or CUDA
(GPU) and call them from JAX with **zero boilerplate**. This page explains how that bridge
is put together; for the step-by-step recipe see :doc:`/how-to/compile-raw-cuda-cpp`, and for
the full reference see the :doc:`/reference/index` *Custom kernels* section.


The problem it solves
---------------------

Calling a native kernel from JAX normally means writing an XLA FFI wrapper by hand: declaring
the buffer types, decoding scalar attributes, registering the target, and managing
compilation. That boilerplate is repetitive and error-prone. ``brainevent`` generates it for
you from a compact description of each function's arguments.


How it fits together
--------------------

.. code-block:: python

   import brainevent

   mod = brainevent.load_cuda_inline(
       name="my_kernels",
       cuda_sources=CUDA_SRC,
       functions={"vector_add": ["arg", "arg", "ret", "stream"]},
   )

   # Call from JAX
   result = jax.ffi.ffi_call("my_kernels.vector_add", out_spec)(a, b)

Four pieces cooperate:

1. **The arg_spec** — a small token list (``"arg"``, ``"ret"``, ``"stream"``, ``"attr.*"``)
   that describes each function's parameters. It is the contract between your C++ signature
   and the generated wrapper. See :doc:`/reference/kernels/arg-spec`.
2. **The wrapper generator** — reads the arg_spec (or infers it from the C++ signature) and
   emits the XLA FFI binding code, so you never write it.
3. **The compiler driver** — invokes ``nvcc``/the host C++ compiler with the right flags and
   produces a shared library. Optimization level, fast math, and extra flags are configurable
   (:doc:`/reference/kernels/compiler-options`).
4. **The cache + registrar** — keys compiled artifacts by a SHA-256 of the source, flags,
   architecture, and version, so recompilation is skipped on subsequent runs, and registers
   the compiled functions as JAX FFI targets (:doc:`/reference/kernels/caching`).


Key properties
--------------

- **Zero boilerplate** — write standard CUDA/C++ and call it from JAX.
- **Automatic FFI wrapper generation** — no manual XLA FFI binding code.
- **Multi-platform** — CUDA (GPU) and C++ (CPU) from the same workflow.
- **Smart caching** — SHA-256-based compilation cache that survives process restarts.
- **Auto-registration** — compiled functions become JAX FFI targets automatically;
  re-importing the same module is a safe no-op.


Where this sits relative to the higher-level decorators
-------------------------------------------------------

Raw C++/CUDA is the lowest-level extension path. For many operators the higher-level
decorators (Numba for CPU, Numba-CUDA and Warp for GPU) are more convenient and require no
separate compiler — see the :doc:`/tutorials/custom-operators/01_numba`,
:doc:`/tutorials/custom-operators/02_numba_cuda`, and
:doc:`/tutorials/custom-operators/03_warp` tutorials. Reach for raw C++/CUDA when you need
full control over the kernel or want to reuse existing native code.
