Operation Customization Routines
================================

.. currentmodule:: brainevent
.. automodule:: brainevent


Define JAX Primitive
--------------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    XLACustomKernel
    GPUKernelChoice



Define JAX Kernel on CPU/GPU/TPU
--------------------------------


CPU kernel definition using Numba.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    numba_kernel


GPU kernel definition using Warp.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    warp_kernel
    jaxtype_to_warptype
    jaxinfo_to_warpinfo



GPU/TPU kernel definition using JAX Pallas.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    pallas_kernel
    LFSR88RNG
    LFSR113RNG
    LFSR128RNG




Kernel definition helper functions.

.. autosummary::
   :toctree: generated/

    defjvp
    general_batching_rule

