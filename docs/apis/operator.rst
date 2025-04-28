Operation Customization Routines
================================

.. currentmodule:: brainevent
.. automodule:: brainevent


Define JAX Primitive
--------------------

.. autosummary::
   :toctree: generated/

    XLACustomKernel



Define JAX Kernel on CPU/GPU/TPU
--------------------------------

.. autosummary::
   :toctree: generated/

    NumbaKernelGenerator
    WarpKernelGenerator
    PallasKernelGenerator


JAX Kernel Definition Helpers
--------------------------------

.. autosummary::
   :toctree: generated/

    defjvp
    general_batching_rule


Numba Kernel Helper
-------------------

.. autosummary::
   :toctree: generated/

    set_numba_environ
    numba_environ_context



Warp Kernel Helper
-------------------

.. autosummary::
   :toctree: generated/

    dtype_to_warp_type

