API Reference
=============

Compilation — CUDA
------------------

.. autofunction:: brainevent.kernix.load_cuda_inline

.. autofunction:: brainevent.kernix.load_cuda_file

.. autofunction:: brainevent.kernix.load_cuda_dir

Compilation — CPU / C++
------------------------

.. autofunction:: brainevent.kernix.load_cpp_inline

.. autofunction:: brainevent.kernix.load_cpp_file

Runtime
-------

.. autoclass:: brainevent.kernix.CompiledModule
   :members:
   :undoc-members:

.. autofunction:: brainevent.kernix.register_ffi_target

.. autofunction:: brainevent.kernix.list_registered_targets

Cache Utilities
---------------

.. autofunction:: brainevent.kernix.clear_cache

.. autofunction:: brainevent.kernix.set_cache_dir

.. autofunction:: brainevent.kernix.get_cache_dir

Diagnostics
-----------

.. autofunction:: brainevent.kernix.print_diagnostics

Exceptions
----------

.. autoclass:: brainevent._error.BrainEventError
   :members:
   :show-inheritance:

.. autoclass:: brainevent._error.CompilationError
   :members:
   :show-inheritance:

.. autoclass:: brainevent._error.KernelToolchainError
   :members:
   :show-inheritance:

.. autoclass:: brainevent._error.KernelRegistrationError
   :members:
   :show-inheritance:
