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

import ctypes
import importlib.util
import threading
import traceback
from ctypes import c_void_p, c_size_t, POINTER, CFUNCTYPE, Structure
from typing import Callable, Dict, Tuple, Union

import jax
import numpy as np

from .numba_ffi import (
    XLA_FFI_Extension_Type,
    XLA_FFI_Extension_Base,
    XLA_FFI_Metadata_Extension,
    XLA_FFI_CallFrame,
    XLA_FFI_Buffer,
    _XLA_FFI_DTYPE_TO_NUMPY,
    _normalize_shapes_and_dtypes,
)
from .util import OutType, abstract_arguments

__all__ = [
    'numba_cuda_kernel',
    'numba_cuda_callable',
]

numba_cuda_installed = importlib.util.find_spec('numba') is not None

# Try to import numba.cuda - will fail gracefully if CUDA is not available
if numba_cuda_installed:
    try:
        from numba import cuda

        # Check if CUDA is actually available
        if not cuda.is_available():
            numba_cuda_installed = False
    except:
        numba_cuda_installed = False

_NUMBA_CUDA_FFI_HANDLES: Dict[str, object] = {}
_CUDA_FFI_CALLBACK_COUNTER = 0
_CUDA_FFI_CALLBACK_LOCK = threading.Lock()

# The typed FFI callback signature: void* fn(XLA_FFI_CallFrame*)
_CUDA_FFI_CALLBACK_TYPE = CFUNCTYPE(c_void_p, POINTER(XLA_FFI_CallFrame))


# ---------------------------------------------------------------------------
# XLA FFI API structures for CUDA stream extraction
# (Based on XLA's C API: xla/ffi/api/c_api.h)
# ---------------------------------------------------------------------------

class XLA_FFI_Api_Version(Structure):
    """XLA FFI API version structure.

    Mirrors the ``XLA_FFI_Api_Version`` struct from XLA's C API header
    (``xla/ffi/api/c_api.h``).  Used internally to interpret version
    information from the XLA FFI API pointer.
    """
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
    ]


class XLA_FFI_Stream_Get_Args(Structure):
    """Arguments for the ``XLA_FFI_Stream_Get`` function.

    Mirrors the ``XLA_FFI_Stream_Get_Args`` struct from XLA's C API.
    The ``ctx`` field is set to the execution context from the call
    frame, and ``stream`` is populated by XLA with the active CUDA
    stream pointer upon return.
    """
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("ctx", c_void_p),  # XLA_FFI_ExecutionContext*
        ("stream", c_void_p),  # Output: cudaStream_t
    ]


# Function pointer type: XLA_FFI_Error* (*XLA_FFI_Stream_Get)(XLA_FFI_Stream_Get_Args*)
_XLA_FFI_Stream_Get_Fn = CFUNCTYPE(c_void_p, POINTER(XLA_FFI_Stream_Get_Args))


class XLA_FFI_Api(Structure):
    """Partial mirror of the ``XLA_FFI_Api`` structure from XLA's C API.

    Only fields up to and including ``XLA_FFI_Stream_Get`` are declared;
    later fields are not needed for CUDA stream extraction and are
    omitted.

    See Also
    --------
    _get_stream_from_callframe : Uses this structure to extract the
        CUDA stream from an XLA call frame.
    """
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("api_version", XLA_FFI_Api_Version),  # Embedded struct, not pointer
        ("internal_api", c_void_p),
        ("XLA_FFI_Error_Create", c_void_p),
        ("XLA_FFI_Error_GetMessage", c_void_p),
        ("XLA_FFI_Error_Destroy", c_void_p),
        ("XLA_FFI_Handler_Register", c_void_p),
        ("XLA_FFI_Stream_Get", _XLA_FFI_Stream_Get_Fn),
        # ... other fields not needed for stream extraction
    ]


def _get_stream_from_callframe(call_frame) -> int:
    """Extract the CUDA stream pointer from an XLA FFI call frame.

    Calls the ``XLA_FFI_Stream_Get`` function exposed by XLA's API
    pointer to obtain the ``cudaStream_t`` associated with the current
    execution context.

    Parameters
    ----------
    call_frame : XLA_FFI_CallFrame
        The call frame structure passed to the FFI callback by XLA.

    Returns
    -------
    int
        The CUDA stream pointer (``cudaStream_t``) as a Python integer.

    Notes
    -----
    This function is internal and should not be called directly.  It is
    used by :class:`NumbaCudaFfiHandler` and
    :class:`NumbaCudaCallableHandler` to obtain the XLA-managed CUDA
    stream for zero-copy kernel launches.
    """
    api = call_frame.api
    # Prepare stream get arguments
    stream_args = XLA_FFI_Stream_Get_Args()
    stream_args.struct_size = ctypes.sizeof(XLA_FFI_Stream_Get_Args)
    stream_args.extension_start = POINTER(XLA_FFI_Extension_Base)()
    stream_args.ctx = call_frame.ctx
    stream_args.stream = None

    # Call XLA's stream getter
    api_ptr = ctypes.cast(api, POINTER(XLA_FFI_Api))
    api_ptr.contents.XLA_FFI_Stream_Get(stream_args)

    return stream_args.stream


def _numba_stream_from_ptr(stream_ptr: int):
    """Create a Numba CUDA stream from a raw ``cudaStream_t`` pointer.

    Parameters
    ----------
    stream_ptr : int
        The ``cudaStream_t`` pointer as a Python integer (e.g.,
        obtained from :func:`_get_stream_from_callframe`).

    Returns
    -------
    numba.cuda.cudadrv.driver.Stream
        A Numba CUDA stream object wrapping the given pointer.  Kernel
        launches on this stream will execute on XLA's CUDA stream.
    """
    return cuda.external_stream(stream_ptr)


def _device_array_from_buffer(data_ptr: int, shape: Tuple[int, ...], dtype: np.dtype):
    """Create a Numba CUDA device array from a raw device memory pointer.

    Uses the ``__cuda_array_interface__`` protocol for zero-copy access
    to device memory owned by XLA.

    Parameters
    ----------
    data_ptr : int
        The device memory pointer as a Python integer.
    shape : tuple of int
        The shape of the array.
    dtype : numpy.dtype
        The element data type.

    Returns
    -------
    numba.cuda.cudadrv.devicearray.DeviceNDArray
        A Numba CUDA device array that wraps the given device memory
        without copying.

    Notes
    -----
    The returned array does **not** own the underlying memory.  The
    caller must ensure that the memory remains valid for the lifetime
    of the array.
    """

    class DevicePointerWrapper:
        """Wrapper class that implements __cuda_array_interface__ protocol."""

        def __init__(self, ptr, arr_shape, arr_dtype):
            self._ptr = ptr
            self._shape = arr_shape
            self._dtype = arr_dtype

        @property
        def __cuda_array_interface__(self):
            return {
                'shape': self._shape,
                'typestr': self._dtype.str,
                'data': (self._ptr, False),  # (ptr, read_only)
                'version': 3,
            }

    wrapper = DevicePointerWrapper(data_ptr, shape, dtype)
    return cuda.as_cuda_array(wrapper)


def _compute_launch_config(
    launch_dims: Union[int, Tuple[int, ...]],
    threads_per_block: int = 256,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Compute CUDA grid and block dimensions from total launch dimensions.

    Automatically determines an appropriate grid/block decomposition
    for 1-D, 2-D, or 3-D kernel launches given the total number of
    threads desired along each axis.

    Parameters
    ----------
    launch_dims : int or tuple of int
        Total number of threads to launch along each axis.  An ``int``
        is treated as a 1-D launch.  Tuples of length 2 or 3 produce
        2-D or 3-D launches respectively.
    threads_per_block : int, optional
        Maximum number of threads per block for 1-D launches.  Default
        is ``256``.  For 2-D and 3-D launches, fixed block sizes are
        used (16x16 and 8x8x4 respectively).

    Returns
    -------
    grid : tuple of int
        Grid dimensions (number of blocks per axis).
    block : tuple of int
        Block dimensions (number of threads per block per axis).

    Raises
    ------
    ValueError
        If *launch_dims* has more than 3 dimensions.

    Examples
    --------
    .. code-block:: python

        >>> grid, block = _compute_launch_config(1024)
        >>> grid
        (4,)
        >>> block
        (256,)

        >>> grid, block = _compute_launch_config((64, 64))
        >>> grid
        (4, 4)
        >>> block
        (16, 16)
    """
    if isinstance(launch_dims, int):
        launch_dims = (launch_dims,)

    if len(launch_dims) == 1:
        total = launch_dims[0]
        block = (min(threads_per_block, total),)
        grid = ((total + block[0] - 1) // block[0],)
    elif len(launch_dims) == 2:
        # For 2D, use a square-ish block
        block_x = min(16, launch_dims[0])
        block_y = min(16, launch_dims[1])
        block = (block_x, block_y)
        grid = (
            (launch_dims[0] + block[0] - 1) // block[0],
            (launch_dims[1] + block[1] - 1) // block[1],
        )
    elif len(launch_dims) == 3:
        # For 3D
        block_x = min(8, launch_dims[0])
        block_y = min(8, launch_dims[1])
        block_z = min(4, launch_dims[2])
        block = (block_x, block_y, block_z)
        grid = (
            (launch_dims[0] + block[0] - 1) // block[0],
            (launch_dims[1] + block[1] - 1) // block[1],
            (launch_dims[2] + block[2] - 1) // block[2],
        )
    else:
        raise ValueError(f"launch_dims must have 1-3 dimensions, got {len(launch_dims)}")

    return grid, block


class NumbaCudaFfiHandler:
    """Typed FFI handler that bridges XLA's typed FFI protocol to a single Numba CUDA kernel.

    This handler registers a single ``@cuda.jit`` kernel as an XLA FFI
    target with fixed grid and block dimensions.  When XLA invokes the
    FFI callback during execution, the handler extracts input/output
    device arrays and the CUDA stream from the call frame, then launches
    the kernel on that stream.

    Parameters
    ----------
    name : str
        Unique FFI target name used for registration with
        ``jax.ffi.register_ffi_target``.
    kernel : numba.cuda.compiler.CUDADispatcher
        The compiled Numba CUDA kernel (from ``@cuda.jit``).
    input_shapes : tuple of tuple of int
        Expected shapes of the input buffers.
    input_dtypes : tuple of numpy.dtype
        Expected data types of the input buffers.
    output_shapes : tuple of tuple of int
        Expected shapes of the output buffers.
    output_dtypes : tuple of numpy.dtype
        Expected data types of the output buffers.
    grid : tuple of int
        Grid dimensions for the kernel launch.
    block : tuple of int
        Block dimensions for the kernel launch.
    shared_mem : int, optional
        Dynamic shared memory size in bytes.  Default is ``0``.

    See Also
    --------
    numba_cuda_kernel : High-level API for creating a JAX-callable from
        a single Numba CUDA kernel.
    NumbaCudaCallableHandler : Handler for arbitrary multi-kernel Python
        callables.

    Notes
    -----
    The handler object must be kept alive (stored in a module-level
    dictionary) to prevent garbage collection of the ctypes callback,
    which would cause a segmentation fault when XLA tries to invoke it.
    """

    def __init__(
        self,
        name: str,
        kernel,
        input_shapes: Tuple[Tuple[int, ...], ...],
        input_dtypes: Tuple[np.dtype, ...],
        output_shapes: Tuple[Tuple[int, ...], ...],
        output_dtypes: Tuple[np.dtype, ...],
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
        shared_mem: int = 0,
    ):
        self.name = name
        self.kernel = kernel
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.output_shapes = output_shapes
        self.output_dtypes = output_dtypes
        self.grid = grid
        self.block = block
        self.shared_mem = shared_mem

        # Create the ctypes callback - must be stored as an attribute to prevent GC
        self._callback = _CUDA_FFI_CALLBACK_TYPE(self._ffi_callback)

        # Register as an FFI target for CUDA platform
        capsule = jax.ffi.pycapsule(ctypes.cast(self._callback, c_void_p).value)
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA")

    def _ffi_callback(self, call_frame_ptr):
        """Typed FFI callback invoked by XLA during kernel execution.

        Extracts input and output device arrays from the call frame,
        obtains the CUDA stream, and launches the Numba CUDA kernel.
        Also handles XLA metadata extension queries (API version and
        traits).

        Parameters
        ----------
        call_frame_ptr : ctypes.POINTER(XLA_FFI_CallFrame)
            Pointer to the XLA FFI call frame.

        Returns
        -------
        None
            Returns ``None`` to indicate success to XLA.
        """
        try:
            call_frame = call_frame_ptr.contents

            # Check for metadata query extension
            ext_ptr = call_frame.extension_start
            if ext_ptr:
                ext = ext_ptr.contents
                if ext.type == int(XLA_FFI_Extension_Type.Metadata):
                    metadata_ext = ctypes.cast(
                        ext_ptr, POINTER(XLA_FFI_Metadata_Extension)
                    ).contents
                    metadata = metadata_ext.metadata.contents
                    metadata.api_version.major_version = 0
                    metadata.api_version.minor_version = 1
                    metadata.traits = 0  # not command-buffer-compatible
                    return None  # success

            # Extract input buffers as CUDA device arrays
            n_inputs = call_frame.args.size
            input_arrays = []
            for i in range(n_inputs):
                buf_ptr = ctypes.cast(
                    call_frame.args.args[i], POINTER(XLA_FFI_Buffer)
                ).contents
                shape = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                dtype = _XLA_FFI_DTYPE_TO_NUMPY.get(buf_ptr.dtype, self.input_dtypes[i])
                input_arrays.append(_device_array_from_buffer(buf_ptr.data, shape, dtype))

            # Extract output buffers as CUDA device arrays
            n_outputs = call_frame.rets.size
            output_arrays = []
            for i in range(n_outputs):
                buf_ptr = ctypes.cast(
                    call_frame.rets.rets[i], POINTER(XLA_FFI_Buffer)
                ).contents
                shape = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                dtype = _XLA_FFI_DTYPE_TO_NUMPY.get(buf_ptr.dtype, self.output_dtypes[i])
                output_arrays.append(_device_array_from_buffer(buf_ptr.data, shape, dtype))

            # Extract XLA's CUDA stream and launch kernel on it
            stream_ptr = _get_stream_from_callframe(call_frame)
            # Use XLA's stream - no synchronization needed
            stream = _numba_stream_from_ptr(stream_ptr)
            with _CUDA_FFI_CALLBACK_LOCK:
                self.kernel[self.grid, self.block, stream, self.shared_mem](*input_arrays, *output_arrays)

        except Exception:
            traceback.print_exc()

        return None  # success


def _register_numba_cuda_ffi_target(
    kernel,
    input_shapes: Tuple[Tuple[int, ...], ...],
    input_dtypes: Tuple[np.dtype, ...],
    output_shapes: Tuple[Tuple[int, ...], ...],
    output_dtypes: Tuple[np.dtype, ...],
    grid: Tuple[int, ...],
    block: Tuple[int, ...],
    shared_mem: int = 0,
):
    """Register a Numba CUDA kernel as an XLA typed FFI target.

    Creates a :class:`NumbaCudaFfiHandler` that wraps the kernel and
    registers it with ``jax.ffi.register_ffi_target``.  The handler is
    stored in a module-level dictionary to prevent garbage collection.

    Parameters
    ----------
    kernel : numba.cuda.compiler.CUDADispatcher
        The compiled Numba CUDA kernel (from ``@cuda.jit``).
    input_shapes : tuple of tuple of int
        Shapes of the input buffers.
    input_dtypes : tuple of numpy.dtype
        Data types of the input buffers.
    output_shapes : tuple of tuple of int
        Shapes of the output buffers.
    output_dtypes : tuple of numpy.dtype
        Data types of the output buffers.
    grid : tuple of int
        Grid dimensions for the kernel launch.
    block : tuple of int
        Block dimensions for the kernel launch.
    shared_mem : int, optional
        Dynamic shared memory size in bytes.  Default is ``0``.

    Returns
    -------
    target_name : str
        The unique FFI target name assigned to this kernel.
    out_types : tuple of jax.ShapeDtypeStruct
        Output type specifications for use with ``jax.ffi.ffi_call``.

    Raises
    ------
    ImportError
        If Numba with CUDA support is not available.

    See Also
    --------
    NumbaCudaFfiHandler : The handler class created by this function.
    numba_cuda_kernel : High-level user-facing API.
    """
    global _CUDA_FFI_CALLBACK_COUNTER

    if not numba_cuda_installed:
        raise ImportError(
            'Numba with CUDA support is required to compile the GPU kernel. '
            'Please install numba and ensure CUDA is available.'
        )

    target_name = f'brainevent_numba_cuda_ffi_{_CUDA_FFI_CALLBACK_COUNTER}'
    _CUDA_FFI_CALLBACK_COUNTER += 1

    handler = NumbaCudaFfiHandler(
        name=target_name,
        kernel=kernel,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        block=block,
        shared_mem=shared_mem,
    )

    # Keep the handler alive to prevent GC of ctypes callback
    _NUMBA_CUDA_FFI_HANDLES[target_name] = handler

    out_types = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(output_shapes, output_dtypes)
    )
    return target_name, out_types


def numba_cuda_kernel(
    kernel,
    outs: OutType,
    *,
    grid: Union[int, Tuple[int, ...], None] = None,
    block: Union[int, Tuple[int, ...], None] = None,
    launch_dims: Union[int, Tuple[int, ...], None] = None,
    threads_per_block: int = 256,
    shared_mem: int = 0,
    vmap_method: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
):
    """Create a JAX-callable function from a single Numba CUDA kernel.

    Wraps a Numba CUDA kernel (decorated with ``@cuda.jit``) so that it
    can be called from JAX on GPU.  The kernel operates on device memory
    directly with zero-copy access via XLA's typed FFI protocol.

    Either ``(grid, block)`` or ``launch_dims`` must be specified to
    configure the CUDA launch.  When ``launch_dims`` is used, the grid
    and block dimensions are computed automatically.

    Parameters
    ----------
    kernel : numba.cuda.compiler.CUDADispatcher
        A Numba CUDA kernel function decorated with ``@cuda.jit``.
    outs : OutType
        Output specification.  A single ``jax.ShapeDtypeStruct`` or a
        sequence/pytree of them for multiple outputs.
    grid : int or tuple of int or None, optional
        Grid dimensions for the kernel launch.  Must be specified
        together with *block*.  Mutually exclusive with *launch_dims*.
    block : int or tuple of int or None, optional
        Block dimensions for the kernel launch.  Must be specified
        together with *grid*.
    launch_dims : int or tuple of int or None, optional
        Total number of threads to launch.  Grid and block are computed
        automatically.  Mutually exclusive with *(grid, block)*.
    threads_per_block : int, optional
        Number of threads per block when using *launch_dims*.  Default
        is ``256``.
    shared_mem : int, optional
        Dynamic shared memory size in bytes.  Default is ``0``.
    vmap_method : str or None, optional
        Method to use for ``jax.vmap``.  Passed directly to
        ``jax.ffi.ffi_call``.
    input_output_aliases : dict of int to int or None, optional
        Mapping from input index to output index for in-place
        operations.  Passed directly to ``jax.ffi.ffi_call``.

    Returns
    -------
    callable
        A function that takes JAX arrays as inputs and returns JAX
        arrays as outputs.  The function can be used inside
        ``jax.jit``-compiled code.

    Raises
    ------
    ImportError
        If Numba with CUDA support is not available.
    ValueError
        If neither ``(grid, block)`` nor ``launch_dims`` is specified.
    AssertionError
        If *kernel* is not a ``numba.cuda.dispatcher.CUDADispatcher``.

    See Also
    --------
    numba_cuda_callable : Wrap an arbitrary Python callable that
        launches multiple Numba CUDA kernels.
    XLACustomKernel.def_numba_cuda_kernel : Register a Numba CUDA
        kernel with an ``XLACustomKernel``.

    Notes
    -----
    Each call to the returned function registers a **new** FFI target.
    For performance-critical inner loops, consider caching the returned
    callable.

    Examples
    --------
    .. code-block:: python

        >>> from numba import cuda
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> @cuda.jit
        ... def add_kernel(x, y, out):
        ...     i = cuda.grid(1)
        ...     if i < out.size:
        ...         out[i] = x[i] + y[i]
        >>>
        >>> # Option 1: Explicit grid/block
        >>> kernel_fn = numba_cuda_kernel(
        ...     add_kernel,
        ...     outs=jax.ShapeDtypeStruct((1024,), jnp.float32),
        ...     grid=4,
        ...     block=256,
        ... )
        >>>
        >>> # Option 2: Auto grid from launch_dims
        >>> kernel_fn = numba_cuda_kernel(
        ...     add_kernel,
        ...     outs=jax.ShapeDtypeStruct((1024,), jnp.float32),
        ...     launch_dims=1024,
        ... )
        >>>
        >>> @jax.jit
        ... def f(a, b):
        ...     return kernel_fn(a, b)
    """

    if not numba_cuda_installed:
        raise ImportError(
            'Numba with CUDA support is required to compile the GPU kernel. '
            'Please install numba and ensure CUDA is available.'
        )

    from numba.cuda.dispatcher import CUDADispatcher

    # Validate kernel type
    assert isinstance(kernel, CUDADispatcher), (
        f'The kernel must be a Numba CUDA JIT-compiled function (from @cuda.jit), '
        f'but got {type(kernel).__name__}.'
    )

    # Compute grid and block dimensions
    if grid is not None and block is not None:
        # Explicit grid/block specified
        if isinstance(grid, int):
            grid = (grid,)
        if isinstance(block, int):
            block = (block,)
        grid = tuple(grid)
        block = tuple(block)
    elif launch_dims is not None:
        # Compute from launch_dims
        grid, block = _compute_launch_config(launch_dims, threads_per_block)
    else:
        raise ValueError(
            "Either (grid, block) or launch_dims must be specified for kernel launch configuration."
        )

    # Output information
    out_info, out_treedef = abstract_arguments(outs)
    output_shapes, output_dtypes = _normalize_shapes_and_dtypes(
        tuple(out.shape for out in out_info),
        tuple(out.dtype for out in out_info),
        'output',
    )

    def call(*ins):
        """Invoke the registered Numba CUDA kernel through XLA FFI.

        Parameters
        ----------
        *ins : jax.Array
            Input arrays on GPU device.

        Returns
        -------
        result
            Output array(s) matching the ``outs`` specification.
        """
        # Input information
        in_info, _ = abstract_arguments(ins)
        input_shapes, input_dtypes = _normalize_shapes_and_dtypes(
            tuple(inp.shape for inp in in_info),
            tuple(inp.dtype for inp in in_info),
            'input',
        )

        # Register FFI target
        target_name, out_types = _register_numba_cuda_ffi_target(
            kernel,
            input_shapes,
            input_dtypes,
            output_shapes,
            output_dtypes,
            grid,
            block,
            shared_mem,
        )

        # Call FFI with typed FFI protocol
        result = jax.ffi.ffi_call(
            target_name,
            out_types,
            input_output_aliases=input_output_aliases,
            vmap_method=vmap_method,
        )(*ins)

        return jax.tree.unflatten(out_treedef, result)

    return call


# ===========================================================================
# numba_cuda_callable: Multi-kernel callable wrapper
# ===========================================================================

_NUMBA_CUDA_CALLABLE_HANDLES: Dict[str, object] = {}
_CUDA_CALLABLE_CALLBACK_COUNTER = 0
_CUDA_CALLABLE_LOCK = threading.Lock()

# The typed FFI callback signature: void* fn(XLA_FFI_CallFrame*)
_CUDA_CALLABLE_CALLBACK_TYPE = CFUNCTYPE(c_void_p, POINTER(XLA_FFI_CallFrame))


class NumbaCudaCallableHandler:
    """Typed FFI handler for arbitrary Python callables that launch Numba CUDA kernels.

    Unlike :class:`NumbaCudaFfiHandler` (which wraps a **single**
    ``@cuda.jit`` kernel with a fixed grid/block), this handler invokes
    a plain Python function and passes it Numba device arrays together
    with a Numba CUDA stream so the function can launch an arbitrary
    number of kernels, allocate temporary device memory, and perform
    multi-step GPU computations.

    Parameters
    ----------
    name : str
        Unique FFI target name for registration with
        ``jax.ffi.register_ffi_target``.
    func : callable
        The Python function to invoke.  Its signature must be
        ``func(in1, in2, ..., out1, out2, ..., stream)`` where each
        ``in*`` and ``out*`` is a Numba CUDA device array and ``stream``
        is a Numba CUDA stream.
    num_inputs : int
        Number of input buffers expected.
    num_outputs : int
        Number of output buffers expected.
    input_dtypes : tuple of numpy.dtype
        Expected data types of the input buffers.
    output_shapes : tuple of tuple of int
        Expected shapes of the output buffers.
    output_dtypes : tuple of numpy.dtype
        Expected data types of the output buffers.

    See Also
    --------
    numba_cuda_callable : High-level API for creating a JAX-callable
        from an arbitrary Python function.
    NumbaCudaFfiHandler : Handler for a single Numba CUDA kernel.

    Notes
    -----
    The handler object must be kept alive (stored in a module-level
    dictionary) to prevent garbage collection of the ctypes callback.
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        num_inputs: int,
        num_outputs: int,
        input_dtypes: Tuple[np.dtype, ...],
        output_shapes: Tuple[Tuple[int, ...], ...],
        output_dtypes: Tuple[np.dtype, ...],
    ):
        self.name = name
        self.func = func
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_dtypes = input_dtypes
        self.output_shapes = output_shapes
        self.output_dtypes = output_dtypes

        # Create the ctypes callback -- must be kept alive to prevent GC
        self._callback = _CUDA_CALLABLE_CALLBACK_TYPE(self._ffi_callback)

        # Register as an FFI target for CUDA platform
        capsule = jax.ffi.pycapsule(ctypes.cast(self._callback, c_void_p).value)
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA")

    def _ffi_callback(self, call_frame_ptr):
        """Typed FFI callback invoked by XLA during execution.

        Extracts input and output device arrays and the CUDA stream
        from the call frame, then calls the user-provided Python
        function.  Also handles XLA metadata extension queries.

        Parameters
        ----------
        call_frame_ptr : ctypes.POINTER(XLA_FFI_CallFrame)
            Pointer to the XLA FFI call frame.

        Returns
        -------
        None
            Returns ``None`` to indicate success to XLA.
        """
        try:
            call_frame = call_frame_ptr.contents

            # Handle metadata extension query (API version / traits)
            ext_ptr = call_frame.extension_start
            if ext_ptr:
                ext = ext_ptr.contents
                if ext.type == int(XLA_FFI_Extension_Type.Metadata):
                    metadata_ext = ctypes.cast(
                        ext_ptr, POINTER(XLA_FFI_Metadata_Extension)
                    ).contents
                    metadata = metadata_ext.metadata.contents
                    metadata.api_version.major_version = 0
                    metadata.api_version.minor_version = 1
                    metadata.traits = 0  # not command-buffer-compatible
                    return None  # success

            # Extract input buffers as Numba CUDA device arrays
            n_inputs = call_frame.args.size
            input_arrays = []
            for i in range(n_inputs):
                buf_ptr = ctypes.cast(
                    call_frame.args.args[i], POINTER(XLA_FFI_Buffer)
                ).contents
                shape = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                dtype = _XLA_FFI_DTYPE_TO_NUMPY.get(buf_ptr.dtype)
                if dtype is None and i < len(self.input_dtypes):
                    dtype = self.input_dtypes[i]
                elif dtype is None:
                    dtype = np.dtype(np.float32)
                input_arrays.append(_device_array_from_buffer(buf_ptr.data, shape, dtype))

            # Extract output buffers as Numba CUDA device arrays
            n_outputs = call_frame.rets.size
            output_arrays = []
            for i in range(n_outputs):
                buf_ptr = ctypes.cast(
                    call_frame.rets.rets[i], POINTER(XLA_FFI_Buffer)
                ).contents
                shape = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                dtype = _XLA_FFI_DTYPE_TO_NUMPY.get(buf_ptr.dtype)
                if dtype is None and i < len(self.output_dtypes):
                    dtype = self.output_dtypes[i]
                elif dtype is None:
                    dtype = np.dtype(np.float32)
                output_arrays.append(_device_array_from_buffer(buf_ptr.data, shape, dtype))

            # Extract XLA's CUDA stream and create Numba stream wrapper
            stream_ptr = _get_stream_from_callframe(call_frame)
            stream = _numba_stream_from_ptr(stream_ptr)

            # Call the user function
            # Signature: func(in1, in2, ..., out1, out2, ..., stream)
            with _CUDA_CALLABLE_LOCK:
                self.func(*input_arrays, *output_arrays, stream)

        except Exception:
            traceback.print_exc()

        return None  # success


def _register_numba_cuda_callable_target(
    func: Callable,
    num_inputs: int,
    num_outputs: int,
    input_dtypes: Tuple[np.dtype, ...],
    output_shapes: Tuple[Tuple[int, ...], ...],
    output_dtypes: Tuple[np.dtype, ...],
):
    """Register a Python callable as an XLA typed FFI target for CUDA.

    Creates a :class:`NumbaCudaCallableHandler` and registers it with
    ``jax.ffi.register_ffi_target``.  The handler is stored in a
    module-level dictionary to prevent garbage collection.

    Parameters
    ----------
    func : callable
        The Python function to wrap.  Its signature must be
        ``func(in1, ..., out1, ..., stream)``.
    num_inputs : int
        Number of input buffers.
    num_outputs : int
        Number of output buffers.
    input_dtypes : tuple of numpy.dtype
        Data types of the input buffers.
    output_shapes : tuple of tuple of int
        Shapes of the output buffers.
    output_dtypes : tuple of numpy.dtype
        Data types of the output buffers.

    Returns
    -------
    target_name : str
        The unique FFI target name assigned to this callable.
    out_types : tuple of jax.ShapeDtypeStruct
        Output type specifications for use with ``jax.ffi.ffi_call``.

    Raises
    ------
    ImportError
        If Numba with CUDA support is not available.

    See Also
    --------
    NumbaCudaCallableHandler : The handler class created by this
        function.
    numba_cuda_callable : High-level user-facing API.
    """
    global _CUDA_CALLABLE_CALLBACK_COUNTER

    if not numba_cuda_installed:
        raise ImportError(
            'Numba with CUDA support is required. '
            'Please install numba and ensure CUDA is available.'
        )

    target_name = f'brainevent_numba_cuda_callable_{_CUDA_CALLABLE_CALLBACK_COUNTER}'
    _CUDA_CALLABLE_CALLBACK_COUNTER += 1

    handler = NumbaCudaCallableHandler(
        name=target_name,
        func=func,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )

    # Keep the handler alive to prevent GC of the ctypes callback
    _NUMBA_CUDA_CALLABLE_HANDLES[target_name] = handler

    out_types = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(output_shapes, output_dtypes)
    )
    return target_name, out_types


def numba_cuda_callable(
    func: Callable,
    outs: OutType,
    *,
    vmap_method: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
):
    """Create a JAX-callable from a Python function that launches Numba CUDA kernels.

    Unlike :func:`numba_cuda_kernel` (which wraps a single
    ``@cuda.jit`` kernel), this function wraps an **arbitrary** Python
    callable.  The callable receives Numba CUDA device arrays for inputs
    and outputs, plus a Numba CUDA stream, and may launch any number of
    kernels, allocate temporary device memory, or perform multi-step GPU
    computations.

    The wrapped function must have the signature::

        func(input_1, input_2, ..., output_1, output_2, ..., stream)

    where every ``input_*`` and ``output_*`` is a Numba CUDA device
    array and ``stream`` is a Numba CUDA stream obtained from XLA.

    Parameters
    ----------
    func : callable
        A Python function with the signature described above.
    outs : OutType
        Output specification.  A single ``jax.ShapeDtypeStruct`` or a
        sequence/pytree of them for multiple outputs.
    vmap_method : str or None, optional
        How to handle ``jax.vmap``.  Passed directly to
        ``jax.ffi.ffi_call``.
    input_output_aliases : dict of int to int or None, optional
        Mapping from input index to output index for in-place
        operations.  Passed directly to ``jax.ffi.ffi_call``.

    Returns
    -------
    callable
        A function that takes JAX arrays as inputs and returns JAX
        arrays as outputs.  The function can be used inside
        ``jax.jit``-compiled code.

    Raises
    ------
    ImportError
        If Numba with CUDA support is not available.
    TypeError
        If *func* is not callable.
    ValueError
        If any input array is a 0-d (scalar) array, which is not
        supported by Numba CUDA device arrays.

    See Also
    --------
    numba_cuda_kernel : Wrap a single ``@cuda.jit`` kernel with fixed
        grid/block configuration.
    XLACustomKernel.def_numba_cuda_kernel : Register a Numba CUDA
        kernel with an ``XLACustomKernel``.

    Notes
    -----
    Each call to the returned function registers a new FFI target.  For
    performance-critical inner loops, consider caching the returned
    callable.

    Scalar (0-d) inputs are not supported because Numba CUDA cannot
    create device arrays from 0-d buffers.  Wrap scalar values in 1-d
    arrays (e.g., ``jnp.array([value])``) before passing them.

    Examples
    --------
    .. code-block:: python

        >>> from numba import cuda
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> @cuda.jit
        ... def add_kernel(x, y, temp, n):
        ...     i = cuda.grid(1)
        ...     if i < n:
        ...         temp[i] = x[i] + y[i]
        >>>
        >>> @cuda.jit
        ... def scale_kernel(temp, out, scale, n):
        ...     i = cuda.grid(1)
        ...     if i < n:
        ...         out[i] = temp[i] * scale
        >>>
        >>> def my_op(x, y, out, stream):
        ...     n = x.shape[0]
        ...     temp = cuda.device_array(n, dtype=x.dtype)
        ...     threads = 256
        ...     blocks = (n + threads - 1) // threads
        ...     add_kernel[blocks, threads, stream](x, y, temp, n)
        ...     scale_kernel[blocks, threads, stream](temp, out, 2.0, n)
        >>>
        >>> fn = numba_cuda_callable(
        ...     my_op,
        ...     outs=jax.ShapeDtypeStruct((1024,), jnp.float32),
        ... )
        >>>
        >>> @jax.jit
        ... def f(a, b):
        ...     return fn(a, b)
    """

    if not numba_cuda_installed:
        raise ImportError(
            'Numba with CUDA support is required to use numba_cuda_callable. '
            'Please install numba and ensure CUDA is available.'
        )

    if not callable(func):
        raise TypeError(
            f'func must be callable, but got {type(func).__name__}.'
        )

    # Output information
    out_info, out_treedef = abstract_arguments(outs)
    output_shapes, output_dtypes = _normalize_shapes_and_dtypes(
        tuple(out.shape for out in out_info),
        tuple(out.dtype for out in out_info),
        'output',
    )
    num_outputs = len(out_info)

    def call(*inputs):
        """Invoke the registered callable through XLA FFI.

        Parameters
        ----------
        *inputs : jax.Array
            Input arrays on GPU device.

        Returns
        -------
        result
            Output array(s) matching the ``outs`` specification.
        """
        inputs = jax.tree.map(jax.numpy.array, inputs)

        # Reject scalar (0-d) inputs — Numba CUDA kernels cannot operate on 0-d device arrays
        for i, inp in enumerate(jax.tree.leaves(inputs)):
            if jax.numpy.ndim(inp) == 0:
                raise ValueError(
                    f"numba_cuda_callable does not support 0-d (scalar) array inputs, "
                    f"but input {i} has shape (). "
                    f"Wrap scalars in a 1-d array, e.g. jnp.array([value])."
                )

        # -- collect input metadata --------------------------------------------
        in_info, _ = abstract_arguments(inputs)
        input_dtypes = tuple(np.dtype(inp.dtype) for inp in in_info)

        # -- register the FFI target -------------------------------------------
        target_name, out_types = _register_numba_cuda_callable_target(
            func,
            num_inputs=len(inputs),
            num_outputs=num_outputs,
            input_dtypes=input_dtypes,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )

        # -- invoke via jax.ffi.ffi_call ---------------------------------------
        result = jax.ffi.ffi_call(
            target_name,
            out_types,
            input_output_aliases=input_output_aliases,
            vmap_method=vmap_method,
        )(*inputs)

        return jax.tree.unflatten(out_treedef, result)

    return call
