# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-


__all__ = [
    'MathError',
    'KernelNotAvailableError',
    'KernelCompilationError',
    'KernelFallbackExhaustedError',
    'KernelExecutionError',
    'BenchmarkDataFnNotProvidedError',
    'TVMFFINotInstalledError',
    'TVMModuleAlreadyRegisteredError',
]


class MathError(Exception):
    """Base exception for mathematical errors in brainevent operations.

    Raised when a mathematical operation fails due to invalid inputs,
    numerical issues, or constraint violations in sparse matrix or
    event-driven computations.

    Parameters
    ----------
    message : str
        A human-readable description of the mathematical error.

    See Also
    --------
    KernelExecutionError : Exception for runtime kernel failures.

    Notes
    -----
    This is the base exception for all math-related errors in brainevent.
    Catch this exception to handle any mathematical failure generically.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._error import MathError
        >>> raise MathError("Matrix dimensions are incompatible")  # doctest: +SKIP
        Traceback (most recent call last):
            ...
        brainevent.MathError: Matrix dimensions are incompatible
    """
    __module__ = 'brainevent'


class KernelNotAvailableError(Exception):
    """Raised when a requested kernel backend is not installed or is version-incompatible.

    This exception signals that a specific backend (e.g., Warp, Pallas,
    Triton) was requested but could not be loaded, either because the
    package is not installed or because the installed version does not
    meet the minimum requirements.

    Parameters
    ----------
    message : str
        A human-readable description indicating which backend is
        unavailable and how to install or upgrade it.

    See Also
    --------
    KernelFallbackExhaustedError : Raised when no fallback backends
        remain after all alternatives have been tried.
    KernelCompilationError : Raised when a backend is available but
        the kernel fails to compile.

    Notes
    -----
    This exception is typically raised during the lowering phase when
    JAX attempts to compile the computation graph and discovers that the
    selected backend cannot be loaded.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._error import KernelNotAvailableError
        >>> raise KernelNotAvailableError(
        ...     "Pallas is not available on this platform."
        ... )  # doctest: +SKIP
    """
    __module__ = 'brainevent'


class KernelCompilationError(Exception):
    """Raised when a kernel fails to compile on the target backend.

    This exception indicates that the backend is available but the
    kernel source could not be compiled, for example due to unsupported
    operations, shape mismatches, or backend-specific limitations.

    Parameters
    ----------
    message : str
        A human-readable description of the compilation failure,
        including the backend name and the underlying error details.

    See Also
    --------
    KernelNotAvailableError : Raised when the backend itself is missing.
    KernelExecutionError : Raised when a compiled kernel fails at runtime.

    Notes
    -----
    Compilation errors are backend-specific. For example, Pallas kernels
    may fail if unsupported JAX operations (like ``dynamic_slice`` on
    Triton) are used, while Warp kernels may fail on type mismatches.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._error import KernelCompilationError
        >>> raise KernelCompilationError(
        ...     "Pallas kernel compilation failed: dynamic_slice not supported on Triton"
        ... )  # doctest: +SKIP
    """
    __module__ = 'brainevent'


class KernelFallbackExhaustedError(Exception):
    """Raised when all fallback kernel backends have been exhausted.

    This exception is raised by :class:`~brainevent._op.main.XLACustomKernel`
    when no registered backend can handle the requested operation on the
    current platform, either because no backends are registered or
    because the explicitly requested backend is not available.

    Parameters
    ----------
    message : str
        A human-readable description listing the primitive name, the
        platform, and the backends that were attempted.

    See Also
    --------
    KernelNotAvailableError : Raised for a single missing backend.
    XLACustomKernel : The kernel manager that raises this exception
        during dispatch.

    Notes
    -----
    The error message typically includes the list of available backends
    for the platform so that the user can switch to an alternative via
    the ``backend=`` keyword argument or :meth:`XLACustomKernel.set_default`.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._error import KernelFallbackExhaustedError
        >>> raise KernelFallbackExhaustedError(
        ...     "No kernels registered for platform 'tpu' in primitive 'csrmv'."
        ... )  # doctest: +SKIP
    """
    __module__ = 'brainevent'


class KernelExecutionError(Exception):
    """Raised when a compiled kernel fails during execution at runtime.

    This exception wraps runtime errors that occur after a kernel has
    been successfully compiled and dispatched, such as out-of-bounds
    memory access, numerical overflow, or device-side assertions. The
    error message includes information about available alternative
    backends.

    Parameters
    ----------
    message : str
        A human-readable description of the runtime failure, typically
        including the backend name, the operation, and suggestions for
        alternative backends.

    See Also
    --------
    KernelCompilationError : Raised when the kernel fails to compile
        rather than at runtime.
    KernelFallbackExhaustedError : Raised when no backends are available
        at all.

    Notes
    -----
    When this exception is raised, the user may be able to work around
    the issue by switching to a different backend via the ``backend=``
    keyword argument in the operation call.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._error import KernelExecutionError
        >>> raise KernelExecutionError(
        ...     "Warp kernel 'csrmv' failed. Try backend='pallas' instead."
        ... )  # doctest: +SKIP
    """
    __module__ = 'brainevent'


class TVMFFINotInstalledError(Exception):
    """Raised when a TVM FFI operation is requested but the package is not installed.

    This exception is raised by :func:`~brainevent._op.util.register_tvm_cuda_kernels`
    when ``jax_tvm_ffi`` or ``tvm_ffi.cpp`` is not available in the current
    environment.

    Parameters
    ----------
    message : str
        A human-readable description indicating that TVM FFI is missing
        and how to install it.

    See Also
    --------
    TVMModuleAlreadyRegisteredError : Raised when the same module name is
        registered more than once.
    KernelNotAvailableError : General exception for unavailable backends.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._error import TVMFFINotInstalledError
        >>> raise TVMFFINotInstalledError(
        ...     "jax_tvm_ffi is not installed. Install with: pip install jax-tvm-ffi"
        ... )  # doctest: +SKIP
    """
    __module__ = 'brainevent'


class TVMModuleAlreadyRegisteredError(Exception):
    """Raised when a TVM CUDA module name is registered more than once.

    :func:`~brainevent._op.util.register_tvm_cuda_kernels` maintains a
    per-process cache of compiled module names.  Attempting to register
    the same *module* name a second time raises this exception so that
    accidental double-registration is caught early rather than silently
    overwriting existing kernels.

    Parameters
    ----------
    message : str
        A human-readable description including the duplicate module name.

    See Also
    --------
    TVMFFINotInstalledError : Raised when TVM FFI is not installed.
    register_tvm_cuda_kernels : The function that raises this exception.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._error import TVMModuleAlreadyRegisteredError
        >>> raise TVMModuleAlreadyRegisteredError(
        ...     "TVM CUDA module 'my_kernels' has already been registered."
        ... )  # doctest: +SKIP
    """
    __module__ = 'brainevent'


class BenchmarkDataFnNotProvidedError(Exception):
    """Raised when ``benchmark()`` is called but no data function has been registered.

    :meth:`~brainevent._op.main.XLACustomKernel.benchmark` requires a
    benchmark data generator to be registered via
    :meth:`~brainevent._op.main.XLACustomKernel.def_benchmark_data`.
    This exception is raised if that function is missing so that the
    caller receives a clear, actionable error instead of a silent
    fallback.

    Parameters
    ----------
    message : str
        A description indicating which primitive is missing its data
        function and how to fix the problem.

    See Also
    --------
    XLACustomKernel.def_benchmark_data : Register a benchmark data
        generator for a primitive.
    XLACustomKernel.benchmark : The method that raises this exception.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._error import BenchmarkDataFnNotProvidedError
        >>> raise BenchmarkDataFnNotProvidedError(
        ...     "No benchmark data function registered for 'csrmv'. "
        ...     "Use def_benchmark_data() to register one."
        ... )  # doctest: +SKIP
    """
    __module__ = 'brainevent'
