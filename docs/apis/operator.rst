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

    NumbaKernelGenerator
    numba_kernel
    set_numba_environ
    numba_environ_context



GPU kernel definition using Warp.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    WarpKernelGenerator
    warp_kernel
    dtype_to_warp_type



GPU/TPU kernel definition using JAX Pallas.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    PallasKernelGenerator
    LFSR88RNG
    LFSR113RNG
    LFSR128RNG




Kernel definition helper functions.

.. autosummary::
   :toctree: generated/

    defjvp
    general_batching_rule

