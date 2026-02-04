Operation Customization Routines
================================

.. currentmodule:: brainevent
.. automodule:: brainevent
   :no-index:


Define JAX Primitive
--------------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    XLACustomKernel



Define JAX Kernel on CPU/GPU/TPU
--------------------------------


CPU kernel definition using Numba.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    numba_kernel


GPU kernel definition using Numba CUDA.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    numba_cuda_kernel



GPU/TPU kernel definition using JAX Pallas.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    LFSR88RNG
    LFSR113RNG
    LFSR128RNG



Kernel definition helper functions.

.. autosummary::
   :toctree: generated/

    defjvp
    general_batching_rule
    jaxtype_to_warptype
    jaxinfo_to_warpinfo

