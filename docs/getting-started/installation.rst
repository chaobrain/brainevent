Installation
============

``brainevent`` is a pure-Python package built on `JAX <https://docs.jax.dev/>`_. Pick the
install that matches your hardware.

.. tab-set::

   .. tab-item:: CPU

      .. code-block:: bash

         pip install -U brainevent[cpu]

   .. tab-item:: GPU (CUDA)

      .. code-block:: bash

         # CUDA 12
         pip install -U brainevent[cuda12]

         # CUDA 13
         pip install -U brainevent[cuda13]

   .. tab-item:: TPU

      .. code-block:: bash

         pip install -U brainevent[tpu]

To install the whole `BrainX <https://brainx.chaobrain.com/>`_ ecosystem (``brainevent``
bundled with compatible modeling packages) in one step:

.. code-block:: bash

   pip install -U BrainX


Verifying the installation
--------------------------

.. code-block:: python

   import brainevent
   import jax

   print(brainevent.__version__)
   print(jax.devices())          # confirm the expected CPU / GPU / TPU backend


GPU compilation dependencies
----------------------------

The first time a kernel runs on a GPU, ``brainevent`` compiles its CUDA source on the fly.
This needs three things:

1. **NVIDIA driver** (provides ``libcuda`` and ``nvidia-smi``) — a system-level requirement
   for any GPU workload.
2. **``jax[cuda12]`` or ``jax[cuda13]``** — installing it pulls in the ``nvidia-*`` pip
   packages, which already bundle ``nvcc``/``ptxas``/the CUDA runtime/headers. **A separate
   system CUDA Toolkit is therefore not required.**
3. **A host C++ compiler (``g++``/``clang++``)** — pip does not provide one. Install it via
   ``conda install -c conda-forge gxx``, ``sudo apt-get install g++``, or
   ``sudo dnf install gcc-c++``.

.. note::

   Pure event-driven array and sparse-matrix operations work out of the box on every
   backend. The C++ compiler is only needed when you compile **custom** C++/CUDA kernels
   (see :doc:`/how-to/compile-raw-cuda-cpp`).


Optional toolchain configuration
---------------------------------

For non-standard CUDA setups, these knobs control kernel compilation:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Setting
     - Effect
   * - ``brainevent.config.prefer_system_nvcc()``
     - Prefer the system ``PATH`` ``nvcc`` instead of the pip-bundled one (pip is the default).
   * - ``BRAINEVENT_NVCC_PREFER=pip|system``
     - Same choice via environment variable.
   * - ``BRAINEVENT_NVCC_PATH`` / ``CUDA_HOME`` / ``CXX``
     - Point at a specific ``nvcc``, CUDA install, or host compiler.
   * - ``BRAINEVENT_ALLOW_UNSUPPORTED_COMPILER=1``
     - Force compilation when the host gcc is newer than nvcc officially supports.
   * - ``BRAINEVENT_COMPUTE_CAPABILITIES=8.6,8.0``
     - Skip ``nvidia-smi`` auto-detection and target these architectures.
   * - ``BRAINEVENT_TOOLCHAIN_DEBUG=1``
     - Append a "toolchain snapshot" to every toolchain error for easier debugging.

Once installed, head to the :doc:`quickstart`.
