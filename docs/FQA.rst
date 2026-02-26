Frequently Asked Questions
==========================

.. contents:: Questions
   :local:
   :depth: 1


Which CUDA backend should I use?
--------------------------------

Use ``brainevent`` for all custom CUDA kernels.

It compiles CUDA sources via ``nvcc``, registers XLA FFI targets, and caches
compiled artifacts on disk for fast reloads.


How do I write a new CUDA kernel?
----------------------------------

Place the kernel in a co-located ``.cu`` file and load it at import time:

.. code-block:: python

   # my_module/my_kernels.py
   from pathlib import Path
   from brainevent import load_cuda_file

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
                  int64_t stream) {
       // kernel launch code
   }

``load_cuda_file`` compiles the kernel on first use, caches the ``.so`` to
disk, and registers it as a JAX FFI target. Subsequent imports skip
recompilation.
