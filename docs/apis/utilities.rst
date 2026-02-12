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
   coo_to_csc_index
   csr_to_csc_index
   binary_array_index


Kernel Helpers
--------------

.. autosummary::
   :toctree: generated/

   register_cuda_kernels
   defjvp
   general_batching_rule
   jaxtype_to_warptype
   jaxinfo_to_warpinfo


GPU/TPU Random Number Generators
--------------------------------

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


Numba RNG LFSR88
----------------

.. autosummary::
   :toctree: generated/

   lfsr88_seed
   lfsr88_next_key
   lfsr88_rand
   lfsr88_randint
   lfsr88_randn
   lfsr88_uniform
   lfsr88_normal
   lfsr88_random_integers


Numba RNG LFSR113
-----------------

.. autosummary::
   :toctree: generated/

   lfsr113_seed
   lfsr113_next_key
   lfsr113_rand
   lfsr113_randint
   lfsr113_randn
   lfsr113_uniform
   lfsr113_normal
   lfsr113_random_integers


Numba RNG LFSR128
-----------------

.. autosummary::
   :toctree: generated/

   lfsr128_seed
   lfsr128_next_key
   lfsr128_rand
   lfsr128_randint
   lfsr128_randn
   lfsr128_uniform
   lfsr128_normal
   lfsr128_random_integers


Numba RNG Dispatch
------------------

Used internally by JIT kernel generators to resolve the correct
LFSR functions at kernel-generation time.

.. autosummary::
   :toctree: generated/

   get_numba_lfsr_seed
   get_numba_lfsr_random_integers
   get_numba_lfsr_uniform
   get_numba_lfsr_normal
   get_numba_lfsr_funcs
