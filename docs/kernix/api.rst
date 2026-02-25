API Reference
=============

Compilation — CUDA
------------------

.. autofunction:: brainevent.load_cuda_inline

.. autofunction:: brainevent.load_cuda_file

.. autofunction:: brainevent.load_cuda_dir

Compilation — CPU / C++
------------------------

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
