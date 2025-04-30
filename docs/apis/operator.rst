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



Define JAX Kernel on CPU/GPU/TPU
--------------------------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

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


Pallas Kernel Helper
-------------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

    LFSR88
    LFSR113
    LFSR128

