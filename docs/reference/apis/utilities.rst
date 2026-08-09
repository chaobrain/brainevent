Utility Functions
=================

.. currentmodule:: brainevent
.. automodule:: brainevent
   :no-index:


Index Conversion
----------------

.. autosummary::
   :toctree: generated/

   csr_to_coo_index
   csr_to_csc_index
   csc_to_csr_index
   coo_to_csc_index
   coo2csr


GPU/TPU Random Number Generator
-------------------------------

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


Hybrid CSR Scheduling
---------------------

Tuning knobs for the hybrid CSR kernels, which pick a per-row execution tier
from the connectivity statistics.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   HybridConfig

.. autosummary::
   :toctree: generated/

   get_hybrid_config
   init_csr_config


Benchmarking
------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   BenchmarkConfig
   BenchmarkRecord
   BenchmarkResult

.. autosummary::
   :toctree: generated/

   benchmark_function


Kernel Helpers
--------------

.. autosummary::
   :toctree: generated/

   defjvp
   general_batching_rule
   jaxtype_to_warptype
   jaxinfo_to_warpinfo
