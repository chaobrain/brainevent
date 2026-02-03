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
from typing import Dict, Sequence, Tuple, Union

import jax
import numpy as np

from .numba_ffi import (
    XLA_FFI_Extension_Type,
    XLA_FFI_Extension_Base,
    XLA_FFI_Metadata_Extension,
    XLA_FFI_CallFrame,
    XLA_FFI_Buffer,
    _XLA_FFI_DTYPE_TO_NUMPY,
    _ensure_sequence,
    _normalize_shapes_and_dtypes,
)
from .util import OutType, abstract_arguments

__all__ = [
    'numba_cuda_kernel',
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
    """XLA FFI API version structure."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
    ]


class XLA_FFI_Stream_Get_Args(Structure):
    """Arguments for XLA_FFI_Stream_Get function."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("ctx", c_void_p),  # XLA_FFI_ExecutionContext*
        ("stream", c_void_p),  # Output: cudaStream_t
    ]


# Function pointer type: XLA_FFI_Error* (*XLA_FFI_Stream_Get)(XLA_FFI_Stream_Get_Args*)
_XLA_FFI_Stream_Get_Fn = CFUNCTYPE(c_void_p, POINTER(XLA_FFI_Stream_Get_Args))


class XLA_FFI_Api(Structure):
    """XLA FFI API structure with fields up to XLA_FFI_Stream_Get.

    This matches the layout from XLA's c_api.h header file.
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
    """Extract CUDA stream pointer from XLA FFI call frame.

    Args:
        call_frame: The XLA_FFI_CallFrame structure.

    Returns:
        The CUDA stream pointer as an integer, or 0 if extraction fails.
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
    """Create a Numba CUDA stream from a raw CUDA stream pointer.

    Args:
        stream_ptr: The cudaStream_t pointer as an integer.

    Returns:
        A Numba CUDA stream object that wraps the given stream.
    """
    return cuda.external_stream(stream_ptr)


def _device_array_from_buffer(data_ptr: int, shape: Tuple[int, ...], dtype: np.dtype):
    """Create a Numba CUDA device array from a raw device pointer.

    Uses the __cuda_array_interface__ protocol for zero-copy access to device memory.

    Args:
        data_ptr: The device memory pointer as an integer.
        shape: The shape of the array.
        dtype: The numpy dtype of the array elements.

    Returns:
        A Numba CUDA device array that wraps the given device memory.
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
    """Compute grid and block dimensions from total launch dimensions.

    Args:
        launch_dims: Total number of threads to launch. Can be an int for 1D,
            or a tuple for multi-dimensional launches.
        threads_per_block: Number of threads per block (default 256).

    Returns:
        A tuple of (grid, block) dimensions.
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
    """Typed FFI handler that bridges XLA's typed FFI protocol to a Numba CUDA kernel."""

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
        """Typed FFI callback matching XLA_FFI_Handler signature."""
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
    """Register a Numba CUDA kernel as an XLA FFI target.

    Args:
        kernel: The Numba CUDA kernel (from @cuda.jit).
        input_shapes: Tuple of input shapes.
        input_dtypes: Tuple of input numpy dtypes.
        output_shapes: Tuple of output shapes.
        output_dtypes: Tuple of output numpy dtypes.
        grid: Grid dimensions for kernel launch.
        block: Block dimensions for kernel launch.
        shared_mem: Dynamic shared memory size in bytes.

    Returns:
        Tuple of (target_name, out_types) for use with jax.ffi.ffi_call.
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
    """Create a JAX-callable function from a Numba CUDA kernel.

    This function wraps a Numba CUDA kernel (decorated with @cuda.jit) so it can
    be called from JAX on GPU. The kernel operates on device memory directly with
    zero-copy access.

    Args:
        kernel: A Numba CUDA kernel function decorated with @cuda.jit.
        outs: Output specification. Can be a single jax.ShapeDtypeStruct or a
            sequence of them for multiple outputs.
        grid: Grid dimensions for kernel launch. Can be an int for 1D or a tuple
            for multi-dimensional grids. Either (grid, block) or launch_dims must
            be specified.
        block: Block dimensions for kernel launch. Can be an int for 1D or a tuple
            for multi-dimensional blocks.
        launch_dims: Total number of threads to launch. Alternative to specifying
            grid and block directly. Grid and block will be computed automatically.
        threads_per_block: Number of threads per block when using launch_dims.
            Default is 256.
        shared_mem: Dynamic shared memory size in bytes. Default is 0.
        vmap_method: The method to use for vmapping this kernel. See JAX documentation
            for jax.ffi.ffi_call for details.
        input_output_aliases: A dictionary mapping input indices to output indices
            for in-place operations. See JAX documentation for details.

    Returns:
        A callable that takes JAX arrays as inputs and returns JAX arrays as outputs.

    Example:
        >>> from numba import cuda
        >>> import jax.numpy as jnp
        >>>
        >>> @cuda.jit
        ... def add_kernel(x, y, out):
        ...     i = cuda.grid(1)
        ...     if i < out.size:
        ...         out[i] = x[i] + y[i]
        >>>
        >>> # Option 1: Explicit grid/block
        >>> kernel = numba_cuda_kernel(
        ...     add_kernel,
        ...     outs=jax.ShapeDtypeStruct((1024,), jnp.float32),
        ...     grid=4,
        ...     block=256,
        ... )
        >>>
        >>> # Option 2: Auto grid from launch_dims
        >>> kernel = numba_cuda_kernel(
        ...     add_kernel,
        ...     outs=jax.ShapeDtypeStruct((1024,), jnp.float32),
        ...     launch_dims=1024,
        ... )
        >>>
        >>> # Use in JAX
        >>> @jax.jit
        ... def f(a, b):
        ...     return kernel(a, b)

    Raises:
        ImportError: If Numba CUDA is not available.
        ValueError: If neither (grid, block) nor launch_dims is specified.
        AssertionError: If kernel is not a Numba CUDA dispatcher.
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

    # Output information - track if single output for unpacking later
    single_output = not isinstance(outs, Sequence)
    outs_seq = _ensure_sequence(outs)
    output_shapes, output_dtypes = _normalize_shapes_and_dtypes(
        tuple(out.shape for out in outs_seq),
        tuple(out.dtype for out in outs_seq),
        'output',
    )

    def call(*ins):
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

        # Unpack single output
        if single_output:
            return result[0]
        return result

    return call
