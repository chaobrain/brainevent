CUDA/C++ Compilation API
========================

**Seamless bridge between native C++/CUDA kernels and JAX via XLA FFI.**

Write kernels in plain C++ (CPU) or CUDA (GPU), and call them from JAX with
zero boilerplate.  ``brainevent`` handles compilation, XLA FFI wrapper
generation, caching, and registration automatically.

.. code-block:: python

   import brainevent

   mod = brainevent.load_cuda_inline(
       name="my_kernels",
       cuda_sources=CUDA_SRC,
       functions={"vector_add": ["arg", "arg", "ret", "stream"]},
   )

   # Call from JAX
   result = jax.ffi.ffi_call("my_kernels.vector_add", out_spec)(a, b)


Key Features
------------

- **Zero boilerplate** — write standard CUDA/C++ and call it from JAX
- **Automatic FFI wrapper generation** — no manual XLA FFI binding code
- **Multi-platform** — CUDA (GPU) and C++ (CPU)
- **Smart caching** — SHA-256-based compilation cache, survives process restarts
- **Ninja parallel builds** — fast multi-file compilation when ninja is available
- **Auto-registration** — compiled functions are automatically registered as
  JAX FFI targets; re-importing the same module is a no-op

.. toctree::
   :maxdepth: 1

   quickstart
   arg_spec
   caching
   compiler_options
   cpp_api
   api
