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


CPU Kernel via Numba
-------------------

.. autosummary::
   :toctree: generated/

   numba_kernel


GPU Kernel via Numba CUDA
-------------------------

.. autosummary::
   :toctree: generated/

   numba_cuda_kernel
   numba_cuda_callable


GPU Kernel via CUDA Source
--------------------------

.. autosummary::
   :toctree: generated/

   load_cuda_inline
   load_cuda_file
   load_cuda_dir

CPU Kernel via C++ Source
-------------------------

.. autosummary::
   :toctree: generated/

   load_cpp_inline
   load_cpp_file


Runtime
-------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   CompiledModule

.. autosummary::
   :toctree: generated/

   register_ffi_target
   list_registered_targets


Cache Utilities
---------------

.. autosummary::
   :toctree: generated/

   clear_cache
   set_cache_dir
   get_cache_dir


Diagnostics
-----------

.. autosummary::
   :toctree: generated/

   print_diagnostics


Exceptions
----------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   BrainEventError
   CompilationError
   KernelToolchainError
   KernelRegistrationError
