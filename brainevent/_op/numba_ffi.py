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
import enum
import importlib.util
import threading
import traceback
from ctypes import c_void_p, c_int, c_int64, c_uint32, c_size_t, POINTER, Structure, CFUNCTYPE
from typing import Dict, Sequence, Tuple

import jax
import numpy as np

from .util import OutType, abstract_arguments

__all__ = [
    'numba_kernel',
]

numba_installed = importlib.util.find_spec('numba') is not None
_NUMBA_CPU_FFI_HANDLES: Dict[str, object] = {}
_FFI_CALLBACK_COUNTER = 0
_FFI_CALLBACK_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# XLA FFI ctypes structure definitions
# (minimal set matching the XLA C header at jaxlib/include/xla/ffi/api/c_api.h
#  and warp._src.jax_experimental.xla_ffi)
# ---------------------------------------------------------------------------


class XLA_FFI_Extension_Type(enum.IntEnum):
    """XLA FFI extension type enumeration."""
    Metadata = 1


# Forward-declared for self-referencing pointer
class XLA_FFI_Extension_Base(Structure):
    """Base ctypes struct for XLA FFI extension chain nodes."""
    pass


XLA_FFI_Extension_Base._fields_ = [
    ("struct_size", c_size_t),
    ("type", c_int),  # XLA_FFI_Extension_Type
    ("next", POINTER(XLA_FFI_Extension_Base)),
]


class XLA_FFI_Api_Version(Structure):
    """ctypes struct for XLA FFI API version information."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("major_version", c_int),
        ("minor_version", c_int),
    ]


class XLA_FFI_Handler_TraitsBits(enum.IntEnum):
    """Bit flags for XLA FFI handler traits."""
    COMMAND_BUFFER_COMPATIBLE = 1 << 0


class XLA_FFI_Metadata(Structure):
    """ctypes struct for XLA FFI handler metadata."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("api_version", XLA_FFI_Api_Version),
        ("traits", c_uint32),
    ]


class XLA_FFI_Metadata_Extension(Structure):
    """ctypes struct for XLA FFI metadata extension node."""
    _fields_ = [
        ("extension_base", XLA_FFI_Extension_Base),
        ("metadata", POINTER(XLA_FFI_Metadata)),
    ]


class XLA_FFI_Buffer(Structure):
    """ctypes struct for an XLA FFI buffer (tensor argument or return value)."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("dtype", c_int),  # XLA_FFI_DataType
        ("data", c_void_p),
        ("rank", c_int64),
        ("dims", POINTER(c_int64)),
    ]


class XLA_FFI_Args(Structure):
    """ctypes struct for XLA FFI call-frame input arguments."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("size", c_int64),
        ("types", POINTER(c_int)),  # XLA_FFI_ArgType*
        ("args", POINTER(c_void_p)),
    ]


class XLA_FFI_Rets(Structure):
    """ctypes struct for XLA FFI call-frame return values."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("size", c_int64),
        ("types", POINTER(c_int)),  # XLA_FFI_RetType*
        ("rets", POINTER(c_void_p)),
    ]


class XLA_FFI_Attrs(Structure):
    """ctypes struct for XLA FFI call-frame attributes."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("size", c_int64),
        ("types", POINTER(c_int)),  # XLA_FFI_AttrType*
        ("names", POINTER(c_void_p)),  # XLA_FFI_ByteSpan**
        ("attrs", POINTER(c_void_p)),
    ]


class XLA_FFI_CallFrame(Structure):
    """ctypes struct for an XLA FFI call frame passed to handler callbacks."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("api", c_void_p),  # const XLA_FFI_Api*
        ("ctx", c_void_p),  # XLA_FFI_ExecutionContext*
        ("stage", c_int),  # XLA_FFI_ExecutionStage
        ("args", XLA_FFI_Args),
        ("rets", XLA_FFI_Rets),
        ("attrs", XLA_FFI_Attrs),
        ("future", c_void_p),  # XLA_FFI_Future*
    ]


# XLA FFI dtype enum -> numpy dtype mapping
# (from xla/ffi/api/c_api.h XLA_FFI_DataType)
_XLA_FFI_DTYPE_TO_NUMPY = {
    1: np.dtype(np.bool_),  # PRED
    2: np.dtype(np.int8),  # S8
    3: np.dtype(np.int16),  # S16
    4: np.dtype(np.int32),  # S32
    5: np.dtype(np.int64),  # S64
    6: np.dtype(np.uint8),  # U8
    7: np.dtype(np.uint16),  # U16
    8: np.dtype(np.uint32),  # U32
    9: np.dtype(np.uint64),  # U64
    10: np.dtype(np.float16),  # F16
    11: np.dtype(np.float32),  # F32
    12: np.dtype(np.float64),  # F64
    15: np.dtype(np.complex64),  # C64
    16: np.dtype(np.bfloat16) if hasattr(np, 'bfloat16') else None,  # BF16
    18: np.dtype(np.complex128),  # C128
}


def _numpy_from_buffer(data_ptr, shape, dtype):
    """Create a zero-copy numpy array from a raw data pointer."""
    size = 1
    for dim in shape:
        size *= dim
    if size == 0:
        return np.empty(shape, dtype=dtype)
    c_type = np.ctypeslib.as_ctypes_type(dtype)
    buffer = (c_type * size).from_address(data_ptr)
    return np.ctypeslib.as_array(buffer).reshape(shape)


# The typed FFI callback signature: void* fn(XLA_FFI_CallFrame*)
_FFI_CALLBACK_TYPE = CFUNCTYPE(c_void_p, POINTER(XLA_FFI_CallFrame))


class NumbaCpuFfiHandler:
    """Typed FFI handler that bridges XLA's typed FFI protocol to a numba kernel."""

    def __init__(
        self,
        name: str,
        kernel,
        input_shapes: Tuple[Tuple[int, ...], ...],
        input_dtypes: Tuple[np.dtype, ...],
        output_shapes: Tuple[Tuple[int, ...], ...],
        output_dtypes: Tuple[np.dtype, ...],
    ):
        self.name = name
        self.kernel = kernel
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.output_shapes = output_shapes
        self.output_dtypes = output_dtypes

        # Create the ctypes callback - must be stored as an attribute to prevent GC
        self._callback = _FFI_CALLBACK_TYPE(self._ffi_callback)

        # Register as an FFI target (typed FFI api_version=1 is the default in JAX 0.9+)
        capsule = jax.ffi.pycapsule(ctypes.cast(self._callback, c_void_p).value)
        jax.ffi.register_ffi_target(name, capsule, platform="cpu")

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

            # Extract input buffers
            n_inputs = call_frame.args.size
            input_arrays = []
            for i in range(n_inputs):
                buf_ptr = ctypes.cast(
                    call_frame.args.args[i], POINTER(XLA_FFI_Buffer)
                ).contents
                shape = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                dtype = _XLA_FFI_DTYPE_TO_NUMPY.get(buf_ptr.dtype, self.input_dtypes[i])
                input_arrays.append(_numpy_from_buffer(buf_ptr.data, shape, dtype))

            # Extract output buffers
            n_outputs = call_frame.rets.size
            output_arrays = []
            for i in range(n_outputs):
                buf_ptr = ctypes.cast(
                    call_frame.rets.rets[i], POINTER(XLA_FFI_Buffer)
                ).contents
                shape = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                dtype = _XLA_FFI_DTYPE_TO_NUMPY.get(buf_ptr.dtype, self.output_dtypes[i])
                output_arrays.append(_numpy_from_buffer(buf_ptr.data, shape, dtype))

            # Call the numba kernel
            with _FFI_CALLBACK_LOCK:
                self.kernel(*input_arrays, *output_arrays)

        except Exception:
            traceback.print_exc()

        return None  # success


def _ensure_sequence(outs: OutType):
    if isinstance(outs, Sequence):
        return tuple(outs)
    return (outs,)


def _normalize_shapes_and_dtypes(
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[object],
    kind: str,
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[np.dtype, ...]]:
    if len(shapes) != len(dtypes):
        raise ValueError(f'Number of {kind} shapes ({len(shapes)}) must match number of dtypes ({len(dtypes)}).')
    normalized_shapes = tuple(tuple(int(dim) for dim in shape) for shape in shapes)
    normalized_dtypes = tuple(np.dtype(dtype) for dtype in dtypes)
    return normalized_shapes, normalized_dtypes


def _register_numba_cpu_ffi_target(
    kernel,
    input_shapes: Tuple[Tuple[int, ...], ...],
    input_dtypes: Tuple[np.dtype, ...],
    output_shapes: Tuple[Tuple[int, ...], ...],
    output_dtypes: Tuple[np.dtype, ...],
):
    global _FFI_CALLBACK_COUNTER

    if not numba_installed:
        raise ImportError('Numba is required to compile the CPU kernel for the custom operator.')

    target_name = f'brainevent_numba_ffi_{_FFI_CALLBACK_COUNTER}'
    _FFI_CALLBACK_COUNTER += 1

    handler = NumbaCpuFfiHandler(
        name=target_name,
        kernel=kernel,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
    )

    # Keep the handler alive to prevent GC of ctypes callback
    _NUMBA_CPU_FFI_HANDLES[target_name] = handler

    out_types = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(output_shapes, output_dtypes)
    )
    return target_name, out_types


def numba_kernel(
    kernel,
    outs: OutType,
    *,
    vmap_method: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
):
    """Create a JAX-callable function from a Numba CPU kernel.

    Wraps a Numba JIT-compiled CPU kernel (decorated with ``@numba.njit``)
    so it can be called from JAX on CPU.  The kernel operates on memory
    directly through the XLA FFI (Foreign Function Interface).

    Parameters
    ----------
    kernel : callable
        A Numba CPU kernel function decorated with ``@numba.njit``.
    outs : jax.ShapeDtypeStruct or sequence of jax.ShapeDtypeStruct
        Output specification.  A single struct for one output, or a
        sequence for multiple outputs.
    vmap_method : str or None, optional
        The method to use for vmapping this kernel.  See JAX documentation
        for ``jax.ffi.ffi_call``.  Default is ``None``.
    input_output_aliases : dict of int to int or None, optional
        Mapping from input indices to output indices for in-place
        operations.  Default is ``None``.

    Returns
    -------
    callable
        A function that takes JAX arrays as inputs and returns JAX
        arrays as outputs.  Compatible with ``jax.jit`` and other
        transformations.

    Examples
    --------
    .. code-block:: python

        >>> import numba
        >>> import jax.numpy as jnp
        >>> import jax
        >>>
        >>> @numba.njit
        ... def add_kernel(x, y, out):
        ...     for i in range(out.size):
        ...         out[i] = x[i] + y[i]
        >>>
        >>> kernel = numba_kernel(
        ...     add_kernel,
        ...     outs=jax.ShapeDtypeStruct((64,), jnp.float32),
        ... )
        >>>
        >>> a = jnp.arange(64, dtype=jnp.float32)
        >>> b = jnp.ones(64, dtype=jnp.float32)
        >>> result = kernel(a, b)
        >>>
        >>> # Multiple outputs
        >>> @numba.njit
        ... def split_kernel(x, out1, out2):
        ...     for i in range(out1.size):
        ...         out1[i] = x[i] * 2
        ...         out2[i] = x[i] * 3
        >>>
        >>> kernel = numba_kernel(
        ...     split_kernel,
        ...     outs=[
        ...         jax.ShapeDtypeStruct((64,), jnp.float32),
        ...         jax.ShapeDtypeStruct((64,), jnp.float32),
        ...     ],
        ... )
        >>> out1, out2 = kernel(x)
        >>>
        >>> # Use with jax.jit
        >>> @jax.jit
        ... def f(a, b):
        ...     return kernel(a, b)
        >>>
        >>> # Use parallel Numba
        >>> @numba.njit(parallel=True)
        ... def parallel_add_kernel(x, y, out):
        ...     for i in numba.prange(out.size):
        ...         out[i] = x[i] + y[i]

    Raises:
        ImportError: If Numba is not installed.
        AssertionError: If kernel is not a Numba CPU dispatcher.

    Note:
        The Numba kernel function should:
        - Accept input arrays followed by output arrays as arguments
        - Write results directly to the output arrays
        - Not return any values (outputs are written in-place)
    """
    if not numba_installed:
        raise ImportError('Numba is required to compile the CPU kernel for the custom operator.')

    from numba.core.registry import CPUDispatcher

    # output information
    out_info, out_treedef = abstract_arguments(outs)
    output_shapes, output_dtypes = _normalize_shapes_and_dtypes(
        tuple(out.shape for out in out_info),
        tuple(out.dtype for out in out_info),
        'output',
    )
    assert isinstance(kernel, CPUDispatcher), 'The kernel must be a Numba JIT-compiled function.'

    def call(*ins):
        # input information
        in_info, _ = abstract_arguments(ins)
        input_shapes, input_dtypes = _normalize_shapes_and_dtypes(
            tuple(inp.shape for inp in in_info),
            tuple(inp.dtype for inp in in_info),
            'input',
        )

        # register FFI target
        target_name, out_types = _register_numba_cpu_ffi_target(
            kernel, input_shapes, input_dtypes, output_shapes, output_dtypes,
        )

        # call FFI with typed FFI protocol
        results = jax.ffi.ffi_call(
            target_name,
            out_types,
            input_output_aliases=input_output_aliases,
            vmap_method=vmap_method,
        )(*ins)
        return jax.tree.unflatten(out_treedef, results)

    return call
