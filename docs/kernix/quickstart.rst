Quick Start
===========

CUDA (GPU)
----------

Write a CUDA kernel, compile it, and call it from JAX:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from brainevent import kernix

   CUDA_SRC = r"""
   #include <cuda_runtime.h>
   #include "brainevent/common.h"

   __global__ void add_kernel(const float* a, const float* b, float* out, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) out[idx] = a[idx] + b[idx];
   }

   // @BE vector_add arg arg ret stream
   void vector_add(const BE::Tensor a, const BE::Tensor b,
                   BE::Tensor out, int64_t stream) {
       int n = a.numel();
       add_kernel<<<(n+255)/256, 256, 0, (cudaStream_t)stream>>>(
           static_cast<const float*>(a.data_ptr()),
           static_cast<const float*>(b.data_ptr()),
           static_cast<float*>(out.data_ptr()), n);
   }
   """

   # Compile and register in one call
   mod = kernix.load_cuda_inline(
       name="my_kernels",
       cuda_sources=CUDA_SRC,
       functions={"vector_add": ["arg", "arg", "ret", "stream"]},
   )

   # Call from JAX
   a = jnp.ones(1024, dtype=jnp.float32)
   b = jnp.full(1024, 2.0, dtype=jnp.float32)

   result = jax.ffi.ffi_call(
       "my_kernels.vector_add",
       jax.ShapeDtypeStruct(a.shape, a.dtype),
   )(a, b)

   print(result)  # [3. 3. 3. ... 3.]

The ``functions`` dict maps each function name to its :doc:`arg_spec <arg_spec>`
token list.  ``kernix`` auto-generates the XLA FFI wrapper and registers the
function as ``"my_kernels.vector_add"``.

.. tip::

   Instead of a ``functions`` dict you can annotate entry points directly in
   the CUDA source with ``// @BE``:

   .. code-block:: cuda

      // @BE vector_add arg arg ret stream
      void vector_add(const BE::Tensor a, const BE::Tensor b,
                      BE::Tensor out, int64_t stream) { ... }

   Then pass ``functions=None`` (the default) and kernix discovers them
   automatically.

CPU (C++)
---------

CPU kernels work the same way but use ``load_cpp_inline`` and don't need CUDA:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from brainevent import kernix

   CPP_SRC = r"""
   #include "brainevent/common.h"

   void add_one(const BE::Tensor x, BE::Tensor y) {
       int n = x.numel();
       const float* in_ptr = static_cast<const float*>(x.data_ptr());
       float* out_ptr = static_cast<float*>(y.data_ptr());
       for (int i = 0; i < n; ++i) out_ptr[i] = in_ptr[i] + 1.0f;
   }
   """

   # Auto-detects arg_spec from C++ signature (const -> arg, non-const -> ret)
   mod = kernix.load_cpp_inline(
       name="my_cpu_ops",
       cpp_sources=CPP_SRC,
       functions=["add_one"],   # list form: auto-detect arg_spec
   )

   cpu = jax.devices("cpu")[0]
   x = jax.device_put(jnp.array([1.0, 2.0, 3.0]), cpu)

   result = jax.ffi.ffi_call(
       "my_cpu_ops.add_one",
       jax.ShapeDtypeStruct(x.shape, x.dtype),
       vmap_method="broadcast_all",
   )(x)

   print(result)  # [2. 3. 4.]

For CPU functions you can pass a **list** of function names instead of a dict.
kernix will parse the C++ signatures automatically: ``const BE::Tensor``
parameters become ``"arg"`` tokens and non-const ``BE::Tensor`` parameters
become ``"ret"`` tokens.

Using ``@jax.jit``
------------------

All registered FFI targets work seamlessly inside ``@jax.jit``:

.. code-block:: python

   @jax.jit
   def add_jit(x, y):
       return jax.ffi.ffi_call(
           "my_kernels.vector_add",
           jax.ShapeDtypeStruct(x.shape, x.dtype),
       )(x, y)

   result = add_jit(a, b)

Loading from Files
------------------

Instead of inline source strings, compile directly from files on disk:

.. code-block:: python

   # Single file — name defaults to the file stem
   mod = kernix.load_cuda_file("kernels/my_kernel.cu")

   # Explicit functions dict if not using // @BE annotations
   mod = kernix.load_cuda_file(
       "kernels/my_kernel.cu",
       functions={"my_func": ["arg", "ret", "stream"]},
   )

   # Entire directory (uses ninja for parallel compilation when available)
   mod = kernix.load_cuda_dir(
       "kernels/",
       functions={"func_a": ["arg", "ret", "stream"],
                  "func_b": ["arg", "arg", "ret", "stream"]},
   )
