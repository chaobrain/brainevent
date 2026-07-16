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

import contextlib
import ctypes
import importlib.util
import threading
import traceback
from ctypes import c_void_p, POINTER, CFUNCTYPE
from typing import Callable, Dict, Optional, Tuple, Union

import jax
import numpy as np

from brainevent._error import KernelRegistrationError
from .ffi_naming import kernel_content_fingerprint
from .numba_ffi import (
    XLA_FFI_API_MAJOR,
    XLA_FFI_API_MINOR,
    XLA_FFI_Error_Code,
    XLA_FFI_Extension_Type,
    XLA_FFI_Metadata_Extension,
    XLA_FFI_CallFrame,
    XLA_FFI_Buffer,
    make_ffi_error,
    resolve_buffer_dtype,
    get_xla_stream,
    get_device_ordinal,
    _normalize_shapes_and_dtypes,
    _warn_if_untested_jax,
)
from .util import OutType, abstract_arguments

__all__ = [
    'numba_cuda_kernel',
    'numba_cuda_callable',
]

numba_cuda_installed = importlib.util.find_spec('numba') is not None

# Cached lazy import, initialized by import_numba_cuda() on first use.
cuda = None


def import_numba_cuda():
    """Import ``numba.cuda`` lazily and validate CUDA availability.

    Returns
    -------
    module
        The imported ``numba.cuda`` module.

    Raises
    ------
    ImportError
        If numba is not importable, or if numba is installed but CUDA is not
        currently available (device not present or the driver raised).

    Notes
    -----
    Only a genuine :class:`ImportError` (numba itself is not importable) poisons
    the module-level ``numba_cuda_installed`` flag; a *transient* CUDA failure
    (e.g. ``CUDA_ERROR_NOT_INITIALIZED`` inside a forked worker, or
    ``cuda.is_available()`` raising) leaves the flag ``True`` so a later call in
    a healthy context can succeed (F19).  The flag therefore means "numba is
    importable", never "CUDA worked once".
    """
    global cuda, numba_cuda_installed
    if cuda is not None:
        return cuda
    if not numba_cuda_installed:
        raise ImportError(
            'Numba with CUDA support is required. '
            'Please install numba and ensure CUDA is available.'
        )
    try:
        from numba import cuda as _cuda
    except ImportError as exc:
        # Genuine import failure: numba (or its CUDA target) is not installed.
        # This is the *only* condition that poisons the availability flag.
        numba_cuda_installed = False
        raise ImportError(
            'Numba with CUDA support is required. '
            'Please install numba and ensure CUDA is available.'
        ) from exc

    # numba imported cleanly; probe the runtime WITHOUT poisoning the flag so a
    # transient CUDA error does not permanently disable the backend (F19).
    try:
        available = _cuda.is_available()
    except Exception as exc:  # noqa: BLE001 - transient CUDA/driver error
        raise ImportError(
            'Numba is installed but CUDA is not currently available: '
            f'{type(exc).__name__}: {exc}. This may be transient (e.g. an '
            'uninitialised CUDA context in a forked worker); retry in a '
            'healthy CUDA context.'
        ) from exc
    if not available:
        raise ImportError(
            'Numba is installed but no CUDA device is available on this machine.'
        )
    cuda = _cuda
    return cuda

_NUMBA_CUDA_FFI_HANDLES: Dict[str, object] = {}
# Maps a kernel/dtype/launch-mode/out-size signature to an already-registered
# FFI target so repeated eager calls reuse one registration instead of leaking
# a fresh handler (and ctypes callback) per call (H1/F8).
_NUMBA_CUDA_FFI_TARGETS: Dict[tuple, str] = {}
# Maps a content-derived target *name* to the fingerprint it was registered
# under (F14).  Two textually identical kernels (even freshly redefined after a
# module reload) reuse the same registration; a name whose stored fingerprint
# differs from a new one raises rather than silently rebinding the target.
_NUMBA_CUDA_FFI_NAME_FINGERPRINTS: Dict[str, Optional[str]] = {}
# Pins kernel objects memoized in ``_NUMBA_CUDA_FFI_TARGETS`` via the
# fingerprint-reuse path.  Those kernels share the first registration's handler
# (which pins only the FIRST kernel), so without this pin such a kernel could
# be garbage-collected and its ``id`` recycled by a different kernel, which
# would then wrongly hit the memo and dispatch to the old handler.
_NUMBA_CUDA_FFI_KERNEL_PINS: Dict[int, object] = {}
_CUDA_FFI_CALLBACK_COUNTER = 0
# Serializes target registration (trace/lowering time).  There is deliberately
# no per-launch lock: each callback operates only on its own call-local device
# arrays and on XLA's stream, so concurrent launches cannot race (L15).
_CUDA_REGISTRATION_LOCK = threading.Lock()

# The typed FFI callback signature: void* fn(XLA_FFI_CallFrame*)
_CUDA_FFI_CALLBACK_TYPE = CFUNCTYPE(c_void_p, POINTER(XLA_FFI_CallFrame))


# ---------------------------------------------------------------------------
# Device-context binding
#
# The XLA FFI ctypes structures, the stream getter and the device-ordinal
# getter live in ``numba_ffi`` (the single source of truth for the FFI ABI).
# This bridge imports ``get_xla_stream`` / ``get_device_ordinal`` from there
# and only adds the numba-CUDA-specific device-context helper below.
# ---------------------------------------------------------------------------

def _device_context(ordinal):
    """Return a context manager binding numba.cuda to device *ordinal*.

    XLA may place an FFI call on any visible GPU; the device arrays and the
    stream must be constructed on *that* device's context, not on whatever
    device numba currently has selected (C3).  Entering ``cuda.gpus[ordinal]``
    pushes the matching device context for the duration of the launch.

    Parameters
    ----------
    ordinal : int or None
        Device ordinal reported by ``XLA_FFI_DeviceOrdinal_Get``.  ``None``
        (older jaxlib that does not expose the ordinal) yields a
        :class:`contextlib.nullcontext`, falling back to numba's current
        device.

    Returns
    -------
    context manager
        Binds the requested device on ``__enter__`` and restores the previous
        device on ``__exit__``.
    """
    if ordinal is None:
        return contextlib.nullcontext()
    try:
        return import_numba_cuda().gpus[ordinal]
    except Exception:  # noqa: BLE001 - unknown ordinal -> keep current device
        return contextlib.nullcontext()


def _numba_stream_from_ptr(stream_ptr: int):
    """Create a Numba CUDA stream from a raw ``cudaStream_t`` pointer.

    Parameters
    ----------
    stream_ptr : int
        The ``cudaStream_t`` pointer as a Python integer (e.g.,
        obtained from :func:`brainevent._op.numba_ffi.get_xla_stream`).

    Returns
    -------
    numba.cuda.cudadrv.driver.Stream
        A Numba CUDA stream object wrapping the given pointer.  Kernel
        launches on this stream will execute on XLA's CUDA stream.
    """
    return import_numba_cuda().external_stream(stream_ptr)


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

    A zero-element buffer is materialised as a fresh empty device array
    rather than wrapped, because XLA may hand a null pointer for an empty
    buffer and ``as_cuda_array`` rejects a null base pointer (M3).
    """
    dtype = np.dtype(dtype)
    shape = tuple(int(d) for d in shape)
    size = 1
    for d in shape:
        size *= d
    if size == 0:
        return import_numba_cuda().device_array(shape, dtype=dtype)

    class DevicePointerWrapper:
        """Wrapper class that implements __cuda_array_interface__ protocol."""

        def __init__(self, ptr, arr_shape, arr_dtype):
            self._ptr = ptr
            self._shape = arr_shape
            self._dtype = arr_dtype

        @property
        def __cuda_array_interface__(self):
            # ``strides`` is ``None`` to declare the buffer C-contiguous; the
            # ffi_call ``input_layouts``/``output_layouts`` make XLA honour
            # this (M4), so the row-major reshape from ``shape`` is exact.
            return {
                'shape': self._shape,
                'typestr': self._dtype.str,
                'data': (self._ptr, False),  # (ptr, read_only)
                'strides': None,
                'version': 3,
            }

    wrapper = DevicePointerWrapper(data_ptr, shape, dtype)
    return import_numba_cuda().as_cuda_array(wrapper)


def _zero_fill_on_stream(device_array, stream) -> None:
    """Asynchronously zero every byte of *device_array* on *stream*.

    Used when a kernel launch is skipped for a degenerate (zero) grid/block but
    an output buffer is non-empty (F9): returning success without touching the
    buffer would hand XLA uninitialised device memory.  A byte-wise memset to
    ``0`` yields ``0`` for every fixed-width numeric dtype (IEEE ``+0.0``,
    integer ``0``, boolean ``False``), so no per-dtype special-casing is needed.

    Parameters
    ----------
    device_array : numba.cuda.cudadrv.devicearray.DeviceNDArray
        The output device array to clear.  Must be C-contiguous (the bridge
        only ever builds contiguous wrappers).
    stream : numba.cuda.cudadrv.driver.Stream
        The XLA-provided CUDA stream to enqueue the memset on, preserving the
        async ordering XLA expects.

    Notes
    -----
    ``numba.cuda.cudadrv.driver.device_memset`` is the cheapest correct
    mechanism numba exposes for a stream-ordered clear: it issues a single
    ``cuMemsetD8Async`` rather than compiling and launching a fill kernel.
    """
    from numba.cuda.cudadrv.driver import device_memset

    nbytes = int(device_array.size) * device_array.dtype.itemsize
    if nbytes == 0:
        return
    device_memset(device_array, 0, nbytes, stream=stream)


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
        If *launch_dims* has zero or more than 3 dimensions, contains a
        negative extent, or if *threads_per_block* is not positive.

    Notes
    -----
    A zero extent along an axis is allowed and yields a grid of ``0`` blocks
    along that axis (an empty launch); the per-axis block size is clamped to a
    minimum of ``1`` so the grid computation never divides by zero (M3).

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
    launch_dims = tuple(int(d) for d in launch_dims)

    n = len(launch_dims)
    if n < 1 or n > 3:
        raise ValueError(f"launch_dims must have 1-3 dimensions, got {n}")
    if any(d < 0 for d in launch_dims):
        raise ValueError(f"launch_dims extents must be non-negative, got {launch_dims}")
    if threads_per_block < 1:
        raise ValueError(f"threads_per_block must be positive, got {threads_per_block}")

    # Per-axis caps: 1-D uses the configurable budget, 2-D a 16x16 tile, 3-D an
    # 8x8x4 tile.  ``max(1, ...)`` guards a zero extent so the grid division
    # below never divides by zero; a zero extent then produces a 0-block grid.
    caps = {1: (threads_per_block,), 2: (16, 16), 3: (8, 8, 4)}[n]
    block = tuple(max(1, min(cap, dim)) for cap, dim in zip(caps, launch_dims))
    grid = tuple((dim + blk - 1) // blk for dim, blk in zip(launch_dims, block))

    return grid, block


class NumbaCudaFfiHandler:
    """Typed FFI handler that bridges XLA's typed FFI protocol to a single Numba CUDA kernel.

    This handler registers a single ``@cuda.jit`` kernel as an XLA FFI
    target.  When XLA invokes the FFI callback during execution, the
    handler extracts input/output device arrays and the CUDA stream from
    the call frame, computes the launch configuration from the stored
    *launch policy*, and launches the kernel on that stream.

    Rather than freezing a concrete grid/block at construction, the handler
    stores the *launch policy* so it can adapt to ``jax.vmap`` (F5).  Under vmap
    the forwarded ``vmap_method`` hands the callback buffers with exactly one
    extra leading batch axis.  The callback detects that by *rank* (runtime rank
    of output 0 equals its abstract rank + 1) and launches the unbatched kernel
    **once per batch slice** on the XLA stream, passing a zero-copy view of each
    slice.  Per-slice launches (rather than a single rescaled launch over a
    flattened buffer) are correct for *any* kernel, including ones whose rows are
    coupled (stencils, reductions, atomics); a flattened launch would read/write
    across batch boundaries.  ``B`` slices enqueued on one stream run in order,
    so no extra synchronisation is needed.

    Parameters
    ----------
    name : str
        Unique FFI target name used for registration with
        ``jax.ffi.register_ffi_target``.
    kernel : numba.cuda.compiler.CUDADispatcher
        The compiled Numba CUDA kernel (from ``@cuda.jit``).
    input_dtypes : tuple of numpy.dtype
        Trace-time input dtypes, used only as the fallback for
        :func:`resolve_buffer_dtype` (the runtime dtype code is authoritative).
    output_dtypes : tuple of numpy.dtype
        Trace-time output dtypes, used only as the resolver fallback.
    abstract_out_shapes : tuple of tuple of int
        The *unbatched* (abstract) shape of each output.  vmap is detected by
        comparing the runtime rank of output 0 against ``len(abstract_out_shapes[0])``.
    launch_mode : tuple
        The launch policy.  Either ``('launch_dims', launch_dims, threads_per_block)``
        (grid/block computed from the *unbatched* dims per launch) or
        ``('explicit', grid, block)`` (fixed grid/block; vmap forbidden).
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
        input_dtypes: Tuple[np.dtype, ...],
        output_dtypes: Tuple[np.dtype, ...],
        abstract_out_shapes: Tuple[Tuple[int, ...], ...],
        launch_mode: tuple,
        shared_mem: int = 0,
    ):
        self.name = name
        self.kernel = kernel
        self.input_dtypes = input_dtypes
        self.output_dtypes = output_dtypes
        self.abstract_out_shapes = abstract_out_shapes
        self.launch_mode = launch_mode
        self.shared_mem = shared_mem

        # Create the ctypes callback - must be stored as an attribute to prevent GC
        self._callback = _CUDA_FFI_CALLBACK_TYPE(self._ffi_callback)

        # Register as an FFI target for CUDA platform
        _warn_if_untested_jax()
        capsule = jax.ffi.pycapsule(ctypes.cast(self._callback, c_void_p).value)
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA")

        # Self-pin (F7): XLA now holds a raw function pointer into
        # ``self._callback``; ``self`` must never be collected while the
        # registration is live, even for direct construction that bypasses
        # the module-level factory. Re-pinning the same name is idempotent.
        _NUMBA_CUDA_FFI_HANDLES[name] = self

    def _ffi_callback(self, call_frame_ptr):
        """Typed FFI callback invoked by XLA during kernel execution.

        Extracts input and output device arrays from the call frame, derives the
        vmap batch factor and launch configuration from the stored launch
        policy, obtains the CUDA stream, and launches the Numba CUDA kernel.
        Also handles XLA metadata extension queries (API version and traits).

        Parameters
        ----------
        call_frame_ptr : ctypes.POINTER(XLA_FFI_CallFrame)
            Pointer to the XLA FFI call frame.

        Returns
        -------
        None or int
            ``None`` (XLA OkStatus) on success, or an ``XLA_FFI_Error*``
            pointer (as an integer) when the launch raised or a batched call was
            rejected, so the failure surfaces to the JAX caller instead of being
            reported as success (C1/F5).

        Notes
        -----
        Under ``jax.vmap`` the ``expand_dims`` / ``broadcast_all`` methods hand
        every buffer exactly one extra leading axis (size ``B`` for a mapped
        operand, size ``1`` for an operand broadcast in from ``in_axes=None``).
        vmap is detected by *rank*: the runtime rank of output 0 equals its
        abstract rank + 1.  The callback then launches the *unbatched* kernel
        once per batch slice ``b`` on the XLA stream, passing ``arr[b]`` for a
        buffer whose leading dim is ``B`` and ``arr[0]`` for one whose leading
        dim is ``1`` (both are zero-copy contiguous views).  Per-slice launches
        are correct for coupled-row kernels (stencils, reductions, atomics),
        which a single flattened launch would corrupt across batch boundaries.
        No vmap (equal ranks) is the pre-existing single launch, bit-identical.
        When the (unbatched) launch config is degenerate (a zero grid/block) the
        launch is skipped and every non-empty output is zero-filled once on the
        stream so no uninitialised memory is returned (F9).
        """
        try:
            call_frame = call_frame_ptr.contents

            # Metadata query: walk the whole extension chain (a future jaxlib may
            # prepend other nodes before the metadata node) (F19).
            ext_ptr = call_frame.extension_start
            while ext_ptr:
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
                ext_ptr = ext.next

            api_ptr = call_frame.api
            ctx = call_frame.ctx

            # Bind the GPU XLA placed this call on before building any device
            # array or stream, so they reference the correct device (C3).
            ordinal = get_device_ordinal(api_ptr, ctx)
            with _device_context(ordinal):
                # --- read raw output dims and detect vmap by RANK --------------
                n_outputs = call_frame.rets.size
                out_bufs = []
                for i in range(n_outputs):
                    buf_ptr = ctypes.cast(
                        call_frame.rets.rets[i], POINTER(XLA_FFI_Buffer)
                    ).contents
                    dims = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                    out_bufs.append((buf_ptr, dims))

                out0_dims = out_bufs[0][1]
                abstract0_shape = self.abstract_out_shapes[0]
                abstract0_rank = len(abstract0_shape)
                # Each vmap level adds exactly one leading batch axis; rank-based
                # detection (unlike a size ratio) correctly handles batch == 1.
                # Only a single level is supported: with two or more extra axes
                # the per-slice reconstruction below would treat the call as
                # unbatched and return garbage for every slice but the first,
                # so refuse loudly instead (nested-vmap users should wrap the
                # outer level with vmap_method='sequential').
                extra_axes = len(out0_dims) - abstract0_rank
                if extra_axes not in (0, 1):
                    return make_ffi_error(
                        api_ptr,
                        XLA_FFI_Error_Code.INVALID_ARGUMENT,
                        f'Numba CUDA kernel {self.name!r}: runtime output rank '
                        f'{len(out0_dims)} differs from the abstract rank '
                        f'{abstract0_rank} by {extra_axes} leading axes; only one '
                        f'level of vmap is supported (nested vmap adds one axis '
                        f'per level). Apply outer levels with '
                        f"vmap_method='sequential' or flatten batch axes before "
                        f'calling.',
                    )
                vmapped = (extra_axes == 1)

                batch = out0_dims[0] if vmapped else 1
                if vmapped:
                    # Defensive: the leading axis must account for the whole size
                    # difference between the runtime and abstract output.
                    abstract0_size = 1
                    for d in abstract0_shape:
                        abstract0_size *= d
                    runtime0_size = 1
                    for d in out0_dims:
                        runtime0_size *= d
                    if abstract0_size * batch != runtime0_size:
                        return make_ffi_error(
                            api_ptr,
                            XLA_FFI_Error_Code.INTERNAL,
                            f'Numba CUDA kernel {self.name!r}: output rank implies a '
                            f'vmap batch axis of {batch}, but runtime size {runtime0_size} '
                            f'!= batch * abstract size ({batch} * {abstract0_size}).',
                        )

                # --- resolve the *unbatched* launch configuration --------------
                mode = self.launch_mode[0]
                if mode == 'explicit':
                    if vmapped:
                        return make_ffi_error(
                            api_ptr,
                            XLA_FFI_Error_Code.INTERNAL,
                            f'Numba CUDA kernel {self.name!r} was registered with an '
                            f'explicit grid/block and cannot be vmapped; register with '
                            f'launch_dims for batched execution.',
                        )
                    grid, block = self.launch_mode[1], self.launch_mode[2]
                else:  # 'launch_dims' -- unbatched config (no scaling)
                    launch_dims, threads_per_block = self.launch_mode[1], self.launch_mode[2]
                    grid, block = _compute_launch_config(launch_dims, threads_per_block)

                degenerate = (0 in grid) or (0 in block)

                # --- build the (possibly batched) device arrays ----------------
                n_inputs = call_frame.args.size
                input_arrays = []
                for i in range(n_inputs):
                    buf_ptr = ctypes.cast(
                        call_frame.args.args[i], POINTER(XLA_FFI_Buffer)
                    ).contents
                    dims = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                    fallback = self.input_dtypes[i] if i < len(self.input_dtypes) else np.dtype(np.float32)
                    dtype = resolve_buffer_dtype(buf_ptr.dtype, fallback)
                    input_arrays.append(_device_array_from_buffer(buf_ptr.data, dims, dtype))

                output_arrays = []
                for i in range(n_outputs):
                    buf_ptr, dims = out_bufs[i]
                    fallback = self.output_dtypes[i] if i < len(self.output_dtypes) else np.dtype(np.float32)
                    dtype = resolve_buffer_dtype(buf_ptr.dtype, fallback)
                    output_arrays.append(_device_array_from_buffer(buf_ptr.data, dims, dtype))

                # Extract XLA's CUDA stream (checked: a failed lookup raises
                # rather than yielding a null/garbage stream) and launch on it.
                stream_ptr = get_xla_stream(api_ptr, ctx)
                stream = _numba_stream_from_ptr(stream_ptr)

                if degenerate:
                    # Skip a degenerate launch (a zero grid/block dimension is a
                    # driver error, and an empty problem has no work); zero-fill
                    # every non-empty output once so nothing is returned
                    # uninitialised (F9/M3).
                    for arr in output_arrays:
                        if arr.size > 0:
                            _zero_fill_on_stream(arr, stream)
                elif not vmapped:
                    # Pre-existing single-launch path (bit-identical).
                    self.kernel[grid, block, stream, self.shared_mem](*input_arrays, *output_arrays)
                else:
                    # vmap: every buffer must carry a leading axis of size B or 1.
                    def _slice_selector(arrays, kind):
                        selectors = []
                        for j, arr in enumerate(arrays):
                            lead = arr.shape[0] if arr.ndim >= 1 else 1
                            if lead == batch:
                                selectors.append((arr, True))   # slice arr[b]
                            elif lead == 1:
                                selectors.append((arr, False))  # broadcast arr[0]
                            else:
                                return None, make_ffi_error(
                                    api_ptr,
                                    XLA_FFI_Error_Code.INTERNAL,
                                    f'Numba CUDA kernel {self.name!r}: {kind} {j} has '
                                    f'leading dim {lead}, which is neither the vmap batch '
                                    f'{batch} nor 1. Retry with vmap_method="broadcast_all".',
                                )
                        return selectors, None

                    in_sel, err = _slice_selector(input_arrays, 'input')
                    if err is not None:
                        return err
                    out_sel, err = _slice_selector(output_arrays, 'output')
                    if err is not None:
                        return err

                    # B slices enqueued on one stream run in order; each slice is
                    # a zero-copy view, so the kernel sees unbatched-rank arrays.
                    for b in range(batch):
                        in_slices = [arr[b] if per_batch else arr[0] for arr, per_batch in in_sel]
                        out_slices = [arr[b] if per_batch else arr[0] for arr, per_batch in out_sel]
                        self.kernel[grid, block, stream, self.shared_mem](*in_slices, *out_slices)

        except Exception as exc:  # noqa: BLE001 - surfaced to XLA as an FFI error
            traceback.print_exc()
            try:
                err_api_ptr = call_frame_ptr.contents.api
            except Exception:
                err_api_ptr = None
            return make_ffi_error(
                err_api_ptr,
                XLA_FFI_Error_Code.INTERNAL,
                f'Numba CUDA kernel {self.name!r} raised '
                f'{type(exc).__name__}: {exc}',
            )

        return None  # success


def _register_numba_cuda_ffi_target(
    kernel,
    input_dtypes: Tuple[np.dtype, ...],
    output_shapes: Tuple[Tuple[int, ...], ...],
    output_dtypes: Tuple[np.dtype, ...],
    launch_mode: tuple,
    shared_mem: int = 0,
):
    """Register (or reuse) a Numba CUDA kernel as an XLA typed FFI target.

    Creates a :class:`NumbaCudaFfiHandler` that wraps the kernel and registers
    it with ``jax.ffi.register_ffi_target``.  The handler is stored in a
    module-level dictionary to prevent garbage collection.

    Parameters
    ----------
    kernel : numba.cuda.compiler.CUDADispatcher
        The compiled Numba CUDA kernel (from ``@cuda.jit``).
    input_dtypes : tuple of numpy.dtype
        Data types of the input buffers (resolver fallback only).
    output_shapes : tuple of tuple of int
        Abstract (unbatched) shapes of the output buffers.
    output_dtypes : tuple of numpy.dtype
        Data types of the output buffers.
    launch_mode : tuple
        The launch policy: ``('launch_dims', launch_dims, threads_per_block)``
        or ``('explicit', grid, block)``.
    shared_mem : int, optional
        Dynamic shared memory size in bytes.  Default is ``0``.

    Returns
    -------
    target_name : str
        The FFI target name assigned to this kernel.
    out_types : tuple of jax.ShapeDtypeStruct
        Output type specifications for use with ``jax.ffi.ffi_call``.

    Raises
    ------
    ImportError
        If Numba with CUDA support is not available.
    KernelRegistrationError
        If a content-derived name is already registered under a *different*
        fingerprint (an astronomically unlikely sha256 collision).

    See Also
    --------
    NumbaCudaFfiHandler : The handler class created by this function.
    numba_cuda_kernel : High-level user-facing API.

    Notes
    -----
    Two caches are layered here (mirroring the CPU path in
    :mod:`brainevent._op.numba_ffi`):

    1. A fast in-process memo keyed on
       ``(id(kernel), input_dtypes, output_dtypes, abstract_out_shapes,
       launch_mode, shared_mem)`` — per-call *input shapes* and the runtime
       (batched) output shapes are excluded (the callback re-derives them from
       ``buf_ptr.dims``), so a kernel called with many distinct shapes registers
       a single target instead of leaking one per shape (F8).  The *abstract*
       output shapes stay in the key because the callback needs their rank to
       detect vmap, and because one kernel function may be wrapped twice with
       different ``outs`` — those must map to distinct handlers.
    2. A content-derived name (F14):
       ``brainevent_numba_cuda_ffi_{fingerprint}`` where *fingerprint* is
       :func:`kernel_content_fingerprint` over the kernel content plus the same
       discriminators as the key.  This makes the name and the key one-to-one,
       so two freshly redefined but byte-identical kernels (e.g. after a module
       reload — different ``id`` yet same content) reuse the one registration.
       ``None`` (an unserialisable closure) falls back to the legacy per-process
       counter for that kernel only (cross-process name stability lost, never
       correctness).
    """
    global _CUDA_FFI_CALLBACK_COUNTER

    import_numba_cuda()

    out_types = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(output_shapes, output_dtypes)
    )
    abstract_out_shapes = tuple(
        tuple(int(d) for d in shape) for shape in output_shapes
    )

    # Discriminators shared by BOTH the key and the fingerprint so a name maps
    # one-to-one to a handler (see Notes).  ``input_dtypes``/``output_dtypes`` are
    # kept because they feed the resolver fallback; folding them into the
    # fingerprint too keeps distinct-dtype kernels on distinct targets.  The
    # abstract output *shapes* (not just sizes) are stored so the callback can
    # detect vmap by rank and so two wrappers with the same total size but
    # different out shapes never share a handler holding the wrong shape.
    discriminators = (input_dtypes, output_dtypes, abstract_out_shapes, launch_mode, shared_mem)
    signature = (id(kernel),) + discriminators

    with _CUDA_REGISTRATION_LOCK:
        cached_name = _NUMBA_CUDA_FFI_TARGETS.get(signature)
        if cached_name is not None:
            return cached_name, out_types

        fingerprint = kernel_content_fingerprint(kernel, extra=discriminators)
        if fingerprint is not None:
            target_name = f'brainevent_numba_cuda_ffi_{fingerprint}'
            existing_fingerprint = _NUMBA_CUDA_FFI_NAME_FINGERPRINTS.get(target_name)
            if existing_fingerprint is not None:
                if existing_fingerprint == fingerprint:
                    # Same content already registered under this name (e.g. a
                    # module reload re-ran this call site with a freshly defined
                    # but byte-identical kernel) -- reuse without re-registering.
                    # Pin *this* kernel: the shared handler keeps only the FIRST
                    # kernel alive, and the memo entry is keyed on this one's id.
                    _NUMBA_CUDA_FFI_KERNEL_PINS[id(kernel)] = kernel
                    _NUMBA_CUDA_FFI_TARGETS[signature] = target_name
                    return target_name, out_types
                raise KernelRegistrationError(
                    f'FFI target name {target_name!r} is already registered for a kernel '
                    f'with a different content fingerprint ({existing_fingerprint!r} != '
                    f'{fingerprint!r}). This is a sha256 collision between two distinct '
                    f'kernel contents and should be astronomically unlikely; if it happens, '
                    f'please report it at https://github.com/chaobrain/brainevent/issues.'
                )
        else:
            # Unserializable closure/global: fall back to a per-process unique
            # counter name for this kernel only (loses cross-process name
            # stability, never correctness).
            target_name = f'brainevent_numba_cuda_ffi_{_CUDA_FFI_CALLBACK_COUNTER}'
            _CUDA_FFI_CALLBACK_COUNTER += 1

        handler = NumbaCudaFfiHandler(
            name=target_name,
            kernel=kernel,
            input_dtypes=input_dtypes,
            output_dtypes=output_dtypes,
            abstract_out_shapes=abstract_out_shapes,
            launch_mode=launch_mode,
            shared_mem=shared_mem,
        )

        # Keep the handler alive to prevent GC of ctypes callback
        _NUMBA_CUDA_FFI_HANDLES[target_name] = handler
        _NUMBA_CUDA_FFI_NAME_FINGERPRINTS[target_name] = fingerprint
        _NUMBA_CUDA_FFI_TARGETS[signature] = target_name

    return target_name, out_types


def numba_cuda_kernel(
    kernel: Callable,
    outs: OutType,
    *,
    grid: Union[int, Tuple[int, ...], None] = None,
    block: Union[int, Tuple[int, ...], None] = None,
    launch_dims: Union[int, Tuple[int, ...], None] = None,
    threads_per_block: int = 256,
    shared_mem: int = 0,
    vmap_method: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
) -> Callable:
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
        If the launch configuration is invalid: neither ``(grid, block)`` nor
        ``launch_dims`` given; only one of ``grid``/``block`` given; both
        ``(grid, block)`` and ``launch_dims`` given; ``vmap_method`` combined
        with an explicit ``grid``/``block``; or an input/output dtype is
        ``bfloat16`` (which numba CUDA cannot launch).
    TypeError
        If *kernel* is not a ``numba.cuda.dispatcher.CUDADispatcher``.

    See Also
    --------
    numba_cuda_callable : Wrap an arbitrary Python callable that
        launches multiple Numba CUDA kernels.
    XLACustomKernel.def_numba_cuda_kernel : Register a Numba CUDA
        kernel with an ``XLACustomKernel``.

    Notes
    -----
    ``grid``/``block`` and ``launch_dims`` are mutually exclusive and one is
    required; the two members of ``grid``/``block`` must be given together.

    Under ``jax.vmap`` the ``launch_dims`` path is batch-aware: the callback
    detects the added batch axis by rank and launches the unbatched kernel once
    per batch slice on the stream, so every batched element is computed
    correctly — including coupled-row kernels such as stencils or reductions
    (F5).  Explicit ``grid``/``block`` kernels cannot be sliced safely, so
    combining them with ``vmap_method`` raises here at wrap time.

    Registrations are memoised by ``(kernel, dtypes, abstract-out-shapes,
    launch-mode, shared_mem)`` — per-call shapes are excluded, so repeated calls
    with different input shapes reuse a single FFI target instead of leaking one
    handler per shape (F8).  Target names are content-derived (F14).

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

    # --- validate the launch configuration (pure; before importing numba so
    #     obvious config errors fail fast even without a CUDA runtime) ---------
    explicit = grid is not None or block is not None
    if explicit and launch_dims is not None:
        raise ValueError(
            "Specify either (grid, block) or launch_dims for the kernel launch "
            "configuration, not both."
        )
    if explicit and (grid is None or block is None):
        raise ValueError(
            "grid and block must be specified together; got "
            f"grid={grid!r}, block={block!r}."
        )
    if not explicit and launch_dims is None:
        raise ValueError(
            "Either (grid, block) or launch_dims must be specified for kernel "
            "launch configuration."
        )
    if explicit and vmap_method is not None:
        raise ValueError(
            "Explicit grid/block kernels cannot be vmapped: a fixed grid cannot "
            "be adapted per batch slice. Use launch_dims together with "
            "vmap_method for batched execution."
        )

    # Build the launch policy (F5): store the *policy*, not a frozen grid/block.
    launch_mode: tuple
    if grid is not None and block is not None:
        grid_t = (grid,) if isinstance(grid, int) else tuple(int(g) for g in grid)
        block_t = (block,) if isinstance(block, int) else tuple(int(b) for b in block)
        launch_mode = ('explicit', grid_t, block_t)
    else:
        # The only remaining case after the validation above; the guard keeps
        # the type-checker happy and is defensive (survives ``python -O``).
        if launch_dims is None:  # pragma: no cover - unreachable after validation
            raise ValueError(
                "Either (grid, block) or launch_dims must be specified for kernel "
                "launch configuration."
            )
        dims_t = (launch_dims,) if isinstance(launch_dims, int) else tuple(int(d) for d in launch_dims)
        # Validate the launch dims eagerly so bad dims raise at wrap time; the
        # callback recomputes the same (unbatched) config per launch.
        _compute_launch_config(dims_t, threads_per_block)
        launch_mode = ('launch_dims', dims_t, int(threads_per_block))

    # Output information
    out_info, out_treedef = abstract_arguments(outs)
    output_shapes, output_dtypes = _normalize_shapes_and_dtypes(
        tuple(out.shape for out in out_info),
        tuple(out.dtype for out in out_info),
        'output',
    )

    # Reject bfloat16 outputs explicitly: numba CUDA cannot launch bf16 kernels,
    # so make the rejection intentional rather than an accidental numba failure
    # deep in the launch (F19).
    for dt in output_dtypes:
        if np.dtype(dt).name == 'bfloat16':
            raise ValueError(
                "numba CUDA cannot launch bfloat16 (bf16) kernels; output dtype "
                "bfloat16 is not supported. Cast the output to float16 or float32."
            )

    import_numba_cuda()

    from numba.cuda.dispatcher import CUDADispatcher

    # Validate kernel type.  Use an explicit ``raise`` rather than ``assert`` so
    # the check survives ``python -O`` (which strips assertions) (L14).
    if not isinstance(kernel, CUDADispatcher):
        raise TypeError(
            f'The kernel must be a Numba CUDA JIT-compiled function (from @cuda.jit), '
            f'but got {type(kernel).__name__}.'
        )
    # Pin row-major layouts so XLA hands the handler C-contiguous device
    # buffers; the callback wraps them by ``dims`` only and cannot recover a
    # non-default layout from ``XLA_FFI_Buffer`` (M4).  ``ffi_call`` takes
    # layouts major-to-minor, so row-major is ``range(ndim)``.
    output_layouts = tuple(tuple(range(len(out.shape))) for out in out_info)

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

        # Reject 0-d (scalar) inputs at trace time with a clear error, mirroring
        # numba_cuda_callable: numba CUDA cannot build device arrays from 0-d
        # buffers, and the run-time failure would otherwise be an opaque INTERNAL
        # FFI error (F19).
        for i, shape in enumerate(input_shapes):
            if len(shape) == 0:
                raise ValueError(
                    f"numba_cuda_kernel does not support 0-d (scalar) array inputs, "
                    f"but input {i} has shape (). Wrap scalars in a 1-d array, "
                    f"e.g. jnp.array([value])."
                )

        # Reject bfloat16 inputs explicitly (numba CUDA cannot launch bf16) (F19).
        for i, dt in enumerate(input_dtypes):
            if np.dtype(dt).name == 'bfloat16':
                raise ValueError(
                    f"numba CUDA cannot launch bfloat16 (bf16) kernels; input {i} "
                    f"has dtype bfloat16, which is not supported. Cast the input to "
                    f"float16 or float32."
                )

        input_layouts = tuple(tuple(range(len(shape))) for shape in input_shapes)

        # Register FFI target
        target_name, out_types = _register_numba_cuda_ffi_target(
            kernel,
            input_dtypes,
            output_shapes,
            output_dtypes,
            launch_mode,
            shared_mem,
        )

        # Call FFI with typed FFI protocol
        result = jax.ffi.ffi_call(
            target_name,
            out_types,
            input_output_aliases=input_output_aliases,
            vmap_method=vmap_method,
            input_layouts=list(input_layouts),
            output_layouts=list(output_layouts),
        )(*ins)

        return jax.tree.unflatten(out_treedef, result)

    return call


# ===========================================================================
# numba_cuda_callable: Multi-kernel callable wrapper
# ===========================================================================

_NUMBA_CUDA_CALLABLE_HANDLES: Dict[str, object] = {}
# Maps a func/io-count/shape/dtype signature to an already-registered target so
# repeated eager calls reuse one registration instead of leaking per call (H1).
_NUMBA_CUDA_CALLABLE_TARGETS: Dict[tuple, str] = {}
# Content-derived name -> fingerprint map (F14), see the kernel path for details.
_NUMBA_CUDA_CALLABLE_NAME_FINGERPRINTS: Dict[str, Optional[str]] = {}
# Pins callables memoized via the fingerprint-reuse path (see
# ``_NUMBA_CUDA_FFI_KERNEL_PINS`` for the id-recycling hazard this prevents).
_NUMBA_CUDA_CALLABLE_PINS: Dict[int, object] = {}
_CUDA_CALLABLE_CALLBACK_COUNTER = 0

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
        _warn_if_untested_jax()
        capsule = jax.ffi.pycapsule(ctypes.cast(self._callback, c_void_p).value)
        jax.ffi.register_ffi_target(name, capsule, platform="CUDA")

        # Self-pin (F7): XLA now holds a raw function pointer into
        # ``self._callback``; ``self`` must never be collected while the
        # registration is live, even for direct construction that bypasses
        # the module-level factory. Re-pinning the same name is idempotent.
        _NUMBA_CUDA_CALLABLE_HANDLES[name] = self

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
        None or int
            ``None`` (XLA OkStatus) on success, or an ``XLA_FFI_Error*``
            pointer (as an integer) when the user function raised, so the
            failure surfaces to the JAX caller instead of being reported as
            success (C1).
        """
        try:
            call_frame = call_frame_ptr.contents

            # Handle metadata extension query (API version / traits).  Walk the
            # whole extension chain via ``ext.next`` (a future jaxlib may prepend
            # other nodes before the metadata node) (F19).
            ext_ptr = call_frame.extension_start
            while ext_ptr:
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
                ext_ptr = ext.next

            api_ptr = call_frame.api
            ctx = call_frame.ctx

            # Bind the GPU XLA placed this call on before building any device
            # array or stream, so they reference the correct device (C3).
            ordinal = get_device_ordinal(api_ptr, ctx)
            with _device_context(ordinal):
                # Extract input buffers.  ``resolve_buffer_dtype`` raises on a
                # known-but-unsupported dtype rather than silently mis-decoding
                # it, and uses the abstract fallback for an unknown code (C2).
                n_inputs = call_frame.args.size
                input_arrays = []
                for i in range(n_inputs):
                    buf_ptr = ctypes.cast(
                        call_frame.args.args[i], POINTER(XLA_FFI_Buffer)
                    ).contents
                    shape = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                    fallback = self.input_dtypes[i] if i < len(self.input_dtypes) else np.dtype(np.float32)
                    dtype = resolve_buffer_dtype(buf_ptr.dtype, fallback)
                    input_arrays.append(_device_array_from_buffer(buf_ptr.data, shape, dtype))

                # Extract output buffers as Numba CUDA device arrays
                n_outputs = call_frame.rets.size
                output_arrays = []
                for i in range(n_outputs):
                    buf_ptr = ctypes.cast(
                        call_frame.rets.rets[i], POINTER(XLA_FFI_Buffer)
                    ).contents
                    shape = tuple(buf_ptr.dims[d] for d in range(buf_ptr.rank))
                    fallback = self.output_dtypes[i] if i < len(self.output_dtypes) else np.dtype(np.float32)
                    dtype = resolve_buffer_dtype(buf_ptr.dtype, fallback)
                    output_arrays.append(_device_array_from_buffer(buf_ptr.data, shape, dtype))

                # Extract XLA's CUDA stream (checked) and create Numba wrapper.
                stream_ptr = get_xla_stream(api_ptr, ctx)
                stream = _numba_stream_from_ptr(stream_ptr)

                # Call the user function
                # Signature: func(in1, in2, ..., out1, out2, ..., stream)
                self.func(*input_arrays, *output_arrays, stream)

        except Exception as exc:  # noqa: BLE001 - surfaced to XLA as an FFI error
            traceback.print_exc()
            try:
                err_api_ptr = call_frame_ptr.contents.api
            except Exception:
                err_api_ptr = None
            return make_ffi_error(
                err_api_ptr,
                XLA_FFI_Error_Code.INTERNAL,
                f'Numba CUDA callable {self.name!r} raised '
                f'{type(exc).__name__}: {exc}',
            )

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

    import_numba_cuda()

    out_types = tuple(
        jax.ShapeDtypeStruct(shape, dtype)
        for shape, dtype in zip(output_shapes, output_dtypes)
    )

    # Reuse an existing registration for an identical func/signature so repeated
    # eager calls do not each leak a handler and ctypes callback (H1).  The
    # cached handler keeps *func* alive, so ``id(func)`` cannot be recycled.  A
    # content-derived name (F14) additionally lets two freshly redefined but
    # byte-identical callables share the one registration.
    discriminators = (num_inputs, num_outputs, input_dtypes, output_shapes, output_dtypes)
    signature = (id(func),) + discriminators
    with _CUDA_REGISTRATION_LOCK:
        cached_name = _NUMBA_CUDA_CALLABLE_TARGETS.get(signature)
        if cached_name is not None:
            return cached_name, out_types

        fingerprint = kernel_content_fingerprint(func, extra=discriminators)
        if fingerprint is not None:
            target_name = f'brainevent_numba_cuda_callable_{fingerprint}'
            existing_fingerprint = _NUMBA_CUDA_CALLABLE_NAME_FINGERPRINTS.get(target_name)
            if existing_fingerprint is not None:
                if existing_fingerprint == fingerprint:
                    # Pin *this* func: the shared handler keeps only the FIRST
                    # func alive, and the memo entry is keyed on this one's id.
                    _NUMBA_CUDA_CALLABLE_PINS[id(func)] = func
                    _NUMBA_CUDA_CALLABLE_TARGETS[signature] = target_name
                    return target_name, out_types
                raise KernelRegistrationError(
                    f'FFI target name {target_name!r} is already registered for a callable '
                    f'with a different content fingerprint ({existing_fingerprint!r} != '
                    f'{fingerprint!r}). This is a sha256 collision between two distinct '
                    f'callable contents and should be astronomically unlikely; if it '
                    f'happens, please report it at '
                    f'https://github.com/chaobrain/brainevent/issues.'
                )
        else:
            # Unserializable closure/global: per-process counter fallback for
            # this callable only (loses cross-process name stability).
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
        _NUMBA_CUDA_CALLABLE_NAME_FINGERPRINTS[target_name] = fingerprint
        _NUMBA_CUDA_CALLABLE_TARGETS[signature] = target_name

    return target_name, out_types


def numba_cuda_callable(
    func: Callable,
    outs: OutType,
    *,
    vmap_method: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
) -> Callable:
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
    Registrations are memoised by ``(func, io-counts, shapes, dtypes)``:
    repeated calls with an identical signature reuse a single FFI target
    instead of leaking one handler per call (H1).

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

    .. warning::

        A ``temp`` array allocated with ``cuda.device_array(...)`` inside the
        callable is managed by numba: it is *deallocated when the Python object
        is garbage-collected*, which numba enqueues on numba's own default
        stream, not on the XLA ``stream`` the kernels run on.  Keep every
        reference to such temporaries alive until all kernels that use them have
        been enqueued (as above, ``temp`` stays in scope for the whole
        function), and do not rely on the deallocation being ordered against the
        XLA stream.  For large or performance-critical temporaries, prefer
        allocating them on ``stream`` and holding the reference for the callable's
        lifetime.
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

    import_numba_cuda()

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
    # Pin row-major layouts so XLA hands C-contiguous device buffers (M4).
    output_layouts = tuple(tuple(range(len(out.shape))) for out in out_info)

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
        # ``asarray`` (not ``array``) normalises dtype/container without forcing
        # a copy, preserving any ``input_output_aliases`` donation (F13).
        inputs = jax.tree.map(jax.numpy.asarray, inputs)

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
        input_layouts = tuple(tuple(range(len(inp.shape))) for inp in in_info)

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
            input_layouts=list(input_layouts),
            output_layouts=list(output_layouts),
        )(*inputs)

        return jax.tree.unflatten(out_treedef, result)

    return call
