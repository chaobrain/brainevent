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
from ctypes import c_void_p, c_int, c_int32, c_int64, c_uint32, c_size_t, c_char_p, POINTER, Structure, CFUNCTYPE
from typing import Callable, Dict, Sequence, Tuple

import jax
import ml_dtypes
import numpy as np

from .util import OutType, abstract_arguments

__all__ = [
    'numba_kernel',
]

numba_installed = importlib.util.find_spec('numba') is not None
_NUMBA_CPU_FFI_HANDLES: Dict[str, object] = {}
# Maps a kernel/shape/dtype signature to an already-registered FFI target name so
# repeated eager calls reuse one registration instead of leaking one per call (H1).
_NUMBA_CPU_FFI_TARGETS: Dict[tuple, str] = {}
_FFI_CALLBACK_COUNTER = 0
# Serializes target registration (trace/lowering time, distinct from execution).
_REGISTRATION_LOCK = threading.Lock()

# XLA FFI API version implemented by this bridge (jaxlib c_api.h
# XLA_FFI_API_MAJOR / XLA_FFI_API_MINOR).  Reported back during the metadata
# handshake so XLA does not reject the handler on a version mismatch.
XLA_FFI_API_MAJOR = 0
XLA_FFI_API_MINOR = 3


class XLA_FFI_Error_Code(enum.IntEnum):
    """Subset of XLA FFI error codes (mirrors ``absl::StatusCode``)."""
    OK = 0
    INVALID_ARGUMENT = 3
    UNIMPLEMENTED = 12
    INTERNAL = 13


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


class XLA_FFI_TypeId(Structure):
    """ctypes struct for an XLA FFI type identifier (state type id)."""
    _fields_ = [
        ("type_id", c_int64),
    ]


class XLA_FFI_Metadata(Structure):
    """ctypes struct for XLA FFI handler metadata.

    Mirrors ``struct XLA_FFI_Metadata`` in jaxlib ``c_api.h``: the trailing
    ``state_type_id`` field must be present or XLA reads/writes past the end of
    the struct during the metadata handshake.
    """
    _fields_ = [
        ("struct_size", c_size_t),
        ("api_version", XLA_FFI_Api_Version),
        ("traits", c_uint32),
        ("state_type_id", XLA_FFI_TypeId),
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


class XLA_FFI_Error_Create_Args(Structure):
    """ctypes struct for arguments to ``XLA_FFI_Error_Create``."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("message", c_char_p),
        ("errc", c_int),  # XLA_FFI_Error_Code
    ]


class XLA_FFI_Stream_Get_Args(Structure):
    """ctypes struct for arguments to ``XLA_FFI_Stream_Get``."""
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("ctx", c_void_p),  # XLA_FFI_ExecutionContext*
        ("stream", c_void_p),  # void* (out)
    ]


class XLA_FFI_DeviceOrdinal_Get_Args(Structure):
    """ctypes struct for arguments to ``XLA_FFI_DeviceOrdinal_Get``.

    Mirrors ``struct XLA_FFI_DeviceOrdinal_Get_Args`` in jaxlib ``c_api.h``.
    ``device_ordinal`` is an out parameter: XLA writes the ordinal of the
    device this call executes on so the GPU bridge can bind the matching CUDA
    context before building device arrays (C3).
    """
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("ctx", c_void_p),  # XLA_FFI_ExecutionContext*
        ("device_ordinal", c_int32),  # int32_t (out)
    ]


# Function-pointer signatures for the API entries these bridges actually call.
XLA_FFI_Error_Create_Func = CFUNCTYPE(c_void_p, POINTER(XLA_FFI_Error_Create_Args))
XLA_FFI_Stream_Get_Func = CFUNCTYPE(c_void_p, POINTER(XLA_FFI_Stream_Get_Args))
XLA_FFI_DeviceOrdinal_Get_Func = CFUNCTYPE(c_void_p, POINTER(XLA_FFI_DeviceOrdinal_Get_Args))


class XLA_FFI_Api(Structure):
    """ctypes view of the XLA FFI API dispatch table (jaxlib ``c_api.h``).

    Every ``_XLA_FFI_API_STRUCT_FIELD`` entry is a function pointer, so the
    table can be modelled as a flat sequence of pointers.  The two entries this
    bridge invokes (``XLA_FFI_Error_Create`` and ``XLA_FFI_Stream_Get``) are
    typed as :class:`ctypes.CFUNCTYPE`; the rest are opaque ``c_void_p`` of the
    same width so the field offsets stay exact.
    """
    _fields_ = [
        ("struct_size", c_size_t),
        ("extension_start", POINTER(XLA_FFI_Extension_Base)),
        ("api_version", XLA_FFI_Api_Version),
        ("internal_api", c_void_p),
        ("XLA_FFI_Error_Create", XLA_FFI_Error_Create_Func),
        ("XLA_FFI_Error_GetMessage", c_void_p),
        ("XLA_FFI_Error_Destroy", c_void_p),
        ("XLA_FFI_Handler_Register", c_void_p),
        ("XLA_FFI_Stream_Get", XLA_FFI_Stream_Get_Func),
        ("XLA_FFI_Type_Register", c_void_p),
        ("XLA_FFI_ExecutionContext_Get", c_void_p),
        ("XLA_FFI_State_Set", c_void_p),
        ("XLA_FFI_State_Get", c_void_p),
        ("XLA_FFI_DeviceMemory_Allocate", c_void_p),
        ("XLA_FFI_DeviceMemory_Free", c_void_p),
        ("XLA_FFI_ThreadPool_Schedule", c_void_p),
        ("XLA_FFI_ThreadPool_NumThreads", c_void_p),
        ("XLA_FFI_Future_Create", c_void_p),
        ("XLA_FFI_Future_SetAvailable", c_void_p),
        ("XLA_FFI_Future_SetError", c_void_p),
        ("XLA_FFI_RunId_Get", c_void_p),
        ("XLA_FFI_DeviceOrdinal_Get", XLA_FFI_DeviceOrdinal_Get_Func),
    ]


def make_ffi_error(api_ptr, code, message: str):
    """Build an ``XLA_FFI_Error*`` to return from a failed FFI callback.

    A typed FFI callback signals failure by returning a non-null
    ``XLA_FFI_Error*``.  Returning ``None`` (a null pointer) is XLA's *Ok*
    status, so an exception that is merely printed and swallowed is reported to
    XLA as success — producing silent wrong answers.  This constructs a real
    error object via the API table referenced by the call frame.

    Parameters
    ----------
    api_ptr : int or None
        Value of ``XLA_FFI_CallFrame.api`` (address of the ``XLA_FFI_Api``
        table).  ``None``/0 disables error creation.
    code : XLA_FFI_Error_Code or int
        Status code to attach to the error.
    message : str
        Human-readable message surfaced to the JAX caller.

    Returns
    -------
    int or None
        The ``XLA_FFI_Error*`` pointer value, or ``None`` when the API table is
        unavailable (in which case XLA falls back to Ok status).
    """
    if not api_ptr:
        return None
    api = ctypes.cast(api_ptr, POINTER(XLA_FFI_Api)).contents
    args = XLA_FFI_Error_Create_Args(
        struct_size=ctypes.sizeof(XLA_FFI_Error_Create_Args),
        extension_start=None,
        message=message.encode('utf-8'),
        errc=int(code),
    )
    return api.XLA_FFI_Error_Create(ctypes.byref(args))


def get_xla_stream(api_ptr, ctx) -> int:
    """Return the CUDA stream XLA assigned to this FFI call.

    A GPU FFI handler must launch its kernel on the stream XLA hands it, not on
    a stream of its own; otherwise the kernel races XLA's own work on the device
    (use-before-write / write-after-read hazards).  ``XLA_FFI_Stream_Get``
    returns an ``XLA_FFI_Error*`` whose non-null value previously went unchecked
    — a failed lookup would silently yield a null/garbage stream.

    Parameters
    ----------
    api_ptr : int
        Value of ``XLA_FFI_CallFrame.api`` (address of the ``XLA_FFI_Api``).
    ctx : int
        Value of ``XLA_FFI_CallFrame.ctx`` (the execution context).

    Returns
    -------
    int
        The CUDA stream handle as an integer (``0`` is the legal default
        stream).

    Raises
    ------
    RuntimeError
        If the API table is unavailable or ``XLA_FFI_Stream_Get`` reports an
        error.  Raised (not swallowed) so the surrounding callback converts it
        into a real ``XLA_FFI_Error`` rather than returning a wrong answer.
    """
    if not api_ptr:
        raise RuntimeError('XLA FFI API table pointer is null; cannot resolve the CUDA stream.')
    api = ctypes.cast(api_ptr, POINTER(XLA_FFI_Api)).contents
    args = XLA_FFI_Stream_Get_Args(
        struct_size=ctypes.sizeof(XLA_FFI_Stream_Get_Args),
        extension_start=None,
        ctx=ctx,
        stream=None,
    )
    err = api.XLA_FFI_Stream_Get(ctypes.byref(args))
    if err:
        raise RuntimeError('XLA_FFI_Stream_Get reported an error while resolving the CUDA stream.')
    # ``stream`` may legitimately be 0 (the default stream); only a hard lookup
    # failure (signalled by *err*) is an error.  Normalise None -> 0.
    return int(args.stream) if args.stream else 0


def get_device_ordinal(api_ptr, ctx):
    """Return the device ordinal XLA placed this FFI call on, or ``None``.

    Used by the GPU bridge to bind the matching CUDA context
    (``with cuda.gpus[ordinal]:``) before constructing device arrays, so a
    multi-GPU execution builds arrays and launches on the device that actually
    owns the buffers (C3).  Returns ``None`` when the ordinal cannot be queried
    (older jaxlib that does not populate ``XLA_FFI_DeviceOrdinal_Get``), letting
    the caller fall back to numba's current device.

    Parameters
    ----------
    api_ptr : int
        Value of ``XLA_FFI_CallFrame.api``.
    ctx : int
        Value of ``XLA_FFI_CallFrame.ctx``.

    Returns
    -------
    int or None
        The device ordinal, or ``None`` if it is unavailable.
    """
    if not api_ptr:
        return None
    try:
        api = ctypes.cast(api_ptr, POINTER(XLA_FFI_Api)).contents
        fn = api.XLA_FFI_DeviceOrdinal_Get
        if not fn:
            return None
        args = XLA_FFI_DeviceOrdinal_Get_Args(
            struct_size=ctypes.sizeof(XLA_FFI_DeviceOrdinal_Get_Args),
            extension_start=None,
            ctx=ctx,
            device_ordinal=-1,
        )
        err = fn(ctypes.byref(args))
        if err or args.device_ordinal < 0:
            return None
        return int(args.device_ordinal)
    except Exception:  # noqa: BLE001 - ordinal is best-effort; fall back to current device
        return None


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
    16: np.dtype(ml_dtypes.bfloat16),  # BF16
    18: np.dtype(np.complex128),  # C128
}


def resolve_buffer_dtype(dtype_code: int, fallback: np.dtype) -> np.dtype:
    """Resolve an XLA FFI dtype enum to a numpy dtype.

    Distinguishes a *known-but-unsupported* dtype code (raise, never silently
    reinterpret the bytes) from an *unknown* code (use the caller's abstract
    fallback).  Replaces ``dict.get(code, fallback)``, which returned the stored
    ``None`` instead of *fallback* when a code was present-but-unmapped.

    Parameters
    ----------
    dtype_code : int
        ``XLA_FFI_Buffer.dtype`` value (an ``XLA_FFI_DataType`` enum).
    fallback : numpy.dtype
        Dtype to use when *dtype_code* is not in the table (the abstract dtype
        JAX assigned to the buffer).

    Returns
    -------
    numpy.dtype
        The resolved element dtype.

    Raises
    ------
    TypeError
        If *dtype_code* is known to the table but maps to no numpy dtype.
    """
    if dtype_code in _XLA_FFI_DTYPE_TO_NUMPY:
        resolved = _XLA_FFI_DTYPE_TO_NUMPY[dtype_code]
        if resolved is None:
            raise TypeError(
                f'XLA FFI dtype code {dtype_code} has no numpy representation in this build.'
            )
        return resolved
    return np.dtype(fallback)


def _numpy_from_buffer(data_ptr, shape, dtype):
    """Create a numpy array viewing a raw XLA buffer pointer.

    Reconstructs the array from a raw-byte view (``np.frombuffer`` over a
    ``ctypes`` char array) rather than ``numpy.ctypeslib.as_ctypes_type``, which
    raises for ``float16``, ``bfloat16``, ``complex64`` and ``complex128``.  The
    byte view is exact for every fixed-width dtype and remains writable, so a
    numba kernel can write results straight into the output buffer.

    Parameters
    ----------
    data_ptr : int
        Address of the buffer's first byte.
    shape : tuple of int
        Logical shape of the array.
    dtype : numpy.dtype
        Element type used to interpret the raw bytes.

    Returns
    -------
    numpy.ndarray
        Array of *shape* and *dtype* sharing memory with the buffer, or a fresh
        empty array when the buffer holds zero elements.
    """
    dtype = np.dtype(dtype)
    size = 1
    for dim in shape:
        size *= int(dim)
    if size == 0:
        return np.empty(shape, dtype=dtype)
    nbytes = size * dtype.itemsize
    raw = (ctypes.c_char * nbytes).from_address(data_ptr)
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


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
                    metadata.api_version.major_version = XLA_FFI_API_MAJOR
                    metadata.api_version.minor_version = XLA_FFI_API_MINOR
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
                dtype = resolve_buffer_dtype(buf_ptr.dtype, self.input_dtypes[i])
                input_arrays.append(_numpy_from_buffer(buf_ptr.data, shape, dtype))

            # Extract output buffers
            n_outputs = call_frame.rets.size
            output_arrays = []
            for i in range(n_outputs):
                buf_ptr = ctypes.cast(
                    call_frame.rets.rets[i], POINTER(XLA_FFI_Buffer)
                ).contents
                shape = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                dtype = resolve_buffer_dtype(buf_ptr.dtype, self.output_dtypes[i])
                output_arrays.append(_numpy_from_buffer(buf_ptr.data, shape, dtype))

            # Call the numba kernel.  No global lock is held: every call works
            # only on its own call-local input/output arrays, so concurrent XLA
            # worker threads cannot race here (H7).
            self.kernel(*input_arrays, *output_arrays)

        except Exception as exc:  # noqa: BLE001 - surfaced to XLA as an FFI error
            traceback.print_exc()
            try:
                api_ptr = call_frame_ptr.contents.api
            except Exception:
                api_ptr = None
            return make_ffi_error(
                api_ptr,
                XLA_FFI_Error_Code.INTERNAL,
                f'Numba CPU kernel {self.name!r} raised '
                f'{type(exc).__name__}: {exc}',
            )

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

    out_types = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(output_shapes, output_dtypes)
    )

    # Reuse an existing registration for an identical kernel/signature.  Every
    # eager call would otherwise mint a fresh target name and register a new FFI
    # handler, leaking one handler (and ctypes callback) per call (H1).  The
    # cached handler keeps *kernel* alive, so ``id(kernel)`` cannot be recycled
    # while the entry lives.
    signature = (id(kernel), input_shapes, input_dtypes, output_shapes, output_dtypes)
    with _REGISTRATION_LOCK:
        cached_name = _NUMBA_CPU_FFI_TARGETS.get(signature)
        if cached_name is not None:
            return cached_name, out_types

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

        # Keep the handler alive to prevent GC of ctypes callback.
        _NUMBA_CPU_FFI_HANDLES[target_name] = handler
        _NUMBA_CPU_FFI_TARGETS[signature] = target_name

    return target_name, out_types


def numba_kernel(
    kernel: Callable,
    outs: OutType,
    *,
    vmap_method: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
) -> Callable:
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

    Raises
    ------
    ImportError
        If Numba is not installed.
    AssertionError
        If *kernel* is not a Numba CPU dispatcher.

    Notes
    -----
    The Numba kernel function should:

    - Accept input arrays followed by output arrays as arguments.
    - Write results directly to the output arrays.
    - Not return any values (outputs are written in-place).

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
    # Use an explicit ``raise`` rather than ``assert`` so the check survives
    # ``python -O`` (which strips assertions) (L14).
    if not isinstance(kernel, CPUDispatcher):
        raise TypeError('The kernel must be a Numba JIT-compiled function (numba.njit).')

    # Pin row-major layouts so XLA is contractually required to hand the handler
    # C-contiguous buffers; the callback reshapes by ``dims`` only and cannot
    # recover a non-default layout from ``XLA_FFI_Buffer`` (M4).  ``ffi_call``
    # takes layouts in major-to-minor order, so row-major is the natural
    # ``range(ndim)`` (it reverses these to XLA's minor-to-major internally).
    output_layouts = tuple(tuple(range(len(out.shape))) for out in out_info)

    def call(*ins):
        # input information
        in_info, _ = abstract_arguments(ins)
        input_shapes, input_dtypes = _normalize_shapes_and_dtypes(
            tuple(inp.shape for inp in in_info),
            tuple(inp.dtype for inp in in_info),
            'input',
        )
        input_layouts = tuple(tuple(range(len(shape))) for shape in input_shapes)

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
            input_layouts=list(input_layouts),
            output_layouts=list(output_layouts),
        )(*ins)
        return jax.tree.unflatten(out_treedef, results)

    return call
