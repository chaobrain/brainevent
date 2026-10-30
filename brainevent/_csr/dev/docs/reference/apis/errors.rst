Error Classes
=============

.. currentmodule:: brainevent
.. automodule:: brainevent
   :no-index:

Every exception raised by ``brainevent`` derives from :class:`BrainEventError`,
so a single ``except brainevent.BrainEventError`` clause catches all of them.
The hierarchy is::

    BrainEventError
    ├── MathError
    ├── UnsupportedOperationError
    ├── BenchmarkDataFnNotProvidedError
    └── KernelError
        ├── KernelNotAvailableError
        ├── KernelFallbackExhaustedError
        ├── KernelExecutionError
        ├── KernelRegistrationError
        ├── KernelLoadError
        ├── CUDANotInstalledError
        ├── KernelCompilationError
        │   └── CompilationError
        │       └── HostCompilerIncompatibleError
        └── KernelToolchainError
            ├── NvccNotFoundError
            ├── HeaderNotFoundError
            ├── HostCompilerNotFoundError
            ├── GpuArchDetectionError
            └── UnsupportedArchError


Base
----

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   BrainEventError
   KernelError


Array and Operation Errors
--------------------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   MathError
   UnsupportedOperationError
   BenchmarkDataFnNotProvidedError


Kernel Dispatch and Execution Errors
------------------------------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   KernelNotAvailableError
   KernelFallbackExhaustedError
   KernelExecutionError
   KernelRegistrationError
   KernelLoadError


Kernel Compilation Errors
-------------------------

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   KernelCompilationError
   CompilationError
   HostCompilerIncompatibleError


Toolchain Errors
----------------

Raised when the CUDA / C++ toolchain needed to build a kernel is missing or
unusable. Each carries a stable ``E-`` diagnostic code in its message.

.. autosummary::
   :toctree: generated/
   :template: classtemplate.rst

   KernelToolchainError
   CUDANotInstalledError
   NvccNotFoundError
   HeaderNotFoundError
   HostCompilerNotFoundError
   GpuArchDetectionError
   UnsupportedArchError
