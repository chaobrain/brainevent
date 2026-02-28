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
------------------

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

.. autofunction:: brainevent.load_cuda_inline

.. autofunction:: brainevent.load_cuda_file

.. autofunction:: brainevent.load_cuda_dir

CPU Kernel vis C++ Source
-------------------------

.. autofunction:: brainevent.load_cpp_inline

.. autofunction:: brainevent.load_cpp_file


Runtime
-------

.. autoclass:: brainevent.CompiledModule
   :members:
   :undoc-members:

.. autofunction:: brainevent.register_ffi_target

.. autofunction:: brainevent.list_registered_targets


Cache Utilities
---------------

.. autofunction:: brainevent.clear_cache

.. autofunction:: brainevent.set_cache_dir

.. autofunction:: brainevent.get_cache_dir


Diagnostics
-----------

.. autofunction:: brainevent.print_diagnostics


Exceptions
----------

.. autoclass:: brainevent.BrainEventError
   :members:
   :show-inheritance:

.. autoclass:: brainevent.CompilationError
   :members:
   :show-inheritance:

.. autoclass:: brainevent.KernelToolchainError
   :members:
   :show-inheritance:

.. autoclass:: brainevent.KernelRegistrationError
   :members:
   :show-inheritance:
