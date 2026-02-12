Custom Kernel Framework
=======================

.. currentmodule:: brainevent
.. automodule:: brainevent
   :no-index:


Custom Kernel
-------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   XLACustomKernel
   KernelEntry


Benchmarking
------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   BenchmarkResult
   BenchmarkReport

.. autosummary::
   :toctree: generated/

   benchmark_function


CPU Kernel (Numba)
------------------

.. autosummary::
   :toctree: generated/

   numba_kernel


GPU Kernel (Numba CUDA)
-----------------------

.. autosummary::
   :toctree: generated/

   numba_cuda_kernel
   numba_cuda_callable


GPU/TPU Kernel (Pallas)
-----------------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   PallasLFSR88RNG
   PallasLFSR113RNG
   PallasLFSR128RNG

.. autosummary::
   :toctree: generated/

   PallasLFSRRNG
   get_pallas_lfsr_rng_class


Helpers
-------

.. autosummary::
   :toctree: generated/

   defjvp
   general_batching_rule
   jaxtype_to_warptype
   jaxinfo_to_warpinfo

