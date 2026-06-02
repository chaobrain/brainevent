Frequently Asked Questions
==========================

.. contents:: Questions
   :local:
   :depth: 1


Do I need a GPU to use ``brainevent``?
--------------------------------------

No. Event-driven array and sparse-matrix operations run on CPU, GPU, and TPU. A GPU (and a
host C++ compiler) is only required when you compile **custom** C++/CUDA kernels. See
:doc:`/getting-started/installation`.


Do I need to install the CUDA Toolkit separately?
-------------------------------------------------

No. Installing ``jax[cuda12]`` or ``jax[cuda13]`` pulls in the ``nvidia-*`` pip packages,
which already bundle ``nvcc``, ``ptxas``, and the CUDA runtime and headers. You still need the
NVIDIA **driver** and a host **C++ compiler** (``g++``/``clang++``). Details in
:doc:`/getting-started/installation`.


How does the event-driven optimization actually work?
------------------------------------------------------

When you multiply a :class:`~brainevent.BinaryArray` by a connectivity structure,
``brainevent`` dispatches a kernel that iterates only over the **active spike indices** and
accumulates their contributions. Work scales with the number of spikes, not the size of the
matrix. See :doc:`event-driven-computation`.


Which connectivity format should I use?
---------------------------------------

- Explicit, reusable sparsity → :class:`~brainevent.CSR` / :class:`~brainevent.CSC`.
- Large random connectivity → JITC (memory independent of synapse count).
- Fixed number of connections per neuron → :class:`~brainevent.FixedPreNumConn` /
  :class:`~brainevent.FixedPostNumConn`.

See :doc:`/how-to/data-structures/choosing-a-sparse-format` and :doc:`sparse-formats`.


Can I learn or inspect individual JITC weights?
-----------------------------------------------

No. JITC connectivity is regenerated from a seed inside the kernel and never materialised, so
individual weights are not addressable. Use an explicit :class:`~brainevent.CSR` matrix when
you need plastic or inspectable weights. See :doc:`jit-connectivity`.


Are computations reproducible?
------------------------------

Yes. JITC connectivity is fully determined by its ``seed`` — the same seed reproduces the
same matrix across processes and devices. Combine with JAX's own PRNG-key discipline for
end-to-end reproducibility.


Can I attach physical units to weights?
---------------------------------------

Yes. ``brainevent`` integrates with `BrainUnit <https://github.com/chaobrain/brainunit>`_, so
weights and currents can carry units and be dimensionally checked, with no runtime cost. See
:doc:`/how-to/data-structures/unit-aware-computation`.


Which custom-kernel backend should I use?
-----------------------------------------

- **Numba** (CPU) / **Numba-CUDA**, **Warp** (GPU) — convenient, decorator-based, no separate
  compiler step.
- **Raw C++/CUDA** — maximum control, or to reuse existing native code.

For raw CUDA kernels, ``brainevent`` compiles your source via ``nvcc``, registers XLA FFI
targets, and caches compiled artifacts on disk for fast reloads. See
:doc:`custom-kernel-architecture` and :doc:`/how-to/building-extending/compile-raw-cuda-cpp`.


How do I ship a custom CUDA kernel with my project?
---------------------------------------------------

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

``load_cuda_file`` compiles the kernel on first use, caches the ``.so`` to disk, and registers
it as a JAX FFI target. Subsequent imports skip recompilation (see
:doc:`/reference/kernels/caching`).
