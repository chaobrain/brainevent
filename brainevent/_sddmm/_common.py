# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Backend metadata, primitive registration and cuSPARSE execution shared by sampled-product leaves."""

import importlib.util
import re
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Iterable, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import match_varying_axes
from brainevent._misc import _cusparse_accepts_nse
from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._op.kernix_toolchain import _nvidia_roots
from brainevent._op.util import CUDA_VALUE_DTYPES, cuda_event_abi, dtype_suffix, event_factor

if TYPE_CHECKING:
    from brainevent._csr.config import CSRKernelSpec

Mode = Literal['float', 'binary', 'weighted']
_INT32_MAX = 2**31 - 1
# Binary float64 kernels also read int8 and float64 events natively; every
# other event dtype follows the shared ``_bool`` / ``_float`` event ABI.
_F64_NATIVE_EVENTS = {np.dtype(np.int8): '_int8', np.dtype(np.float64): '_double'}


@dataclass(frozen=True)
class _CSRBackend:
    """Carry the forward CSR backend selection through gradient dispatch."""

    primitive: 'XLACustomKernel'
    requested: str | None
    config_spec: 'CSRKernelSpec | None' = None


def native_event_target(prefix: str, value_dtype, events: jax.Array, *,
                        weighted: bool) -> tuple[str, jax.Array]:
    """Select a native leaf's FFI symbol and its event operand.

    The symbol is ``prefix`` plus the value suffix, and for binary kernels the
    event suffix. Weighted kernels read events in the value dtype; binary
    kernels follow :func:`~brainevent._op.util.cuda_event_abi`, plus native
    int8/float64 events for float64 values.
    """
    value_dtype = np.dtype(value_dtype)
    if weighted:
        return f'{prefix}{dtype_suffix(value_dtype)}', events.astype(value_dtype)
    extra = _F64_NATIVE_EVENTS.get(np.dtype(events.dtype)) if value_dtype == np.float64 else None
    if extra is not None:
        return f'{prefix}{dtype_suffix(value_dtype)}{extra}', events
    abi = cuda_event_abi(events.dtype)
    return f'{prefix}{dtype_suffix(value_dtype)}{abi.suffix}', abi.prepare(events)


def bind_events(primitive: XLACustomKernel, events: jax.Array, ct: jax.Array, indices: jax.Array,
                indptr: jax.Array, *, transpose: bool, weighted: bool,
                backend: 'str | None | _CSRBackend', **params) -> jax.Array:
    """Bind normalized CSR events to a leaf primitive; validation belongs to the dispatcher.

    A :class:`_CSRBackend` source binds ``auto``, which resolves the source
    CSR primitive's backend on the actual lowering platform.
    """
    source = backend.primitive if isinstance(backend, _CSRBackend) else None
    return primitive(
        events, ct, indices, indptr,
        outs=[jax.ShapeDtypeStruct(indices.shape, ct.dtype)],
        transpose=transpose, weighted=weighted,
        backend='auto' if source is not None else backend, csr_primitive=source,
        ct_info=jax.ShapeDtypeStruct(ct.shape, ct.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        **params,
    )[0]


def event_primitive(name: str, tag: str, cuda_kernel: Callable, jax_kernel: Callable,
                    native_dims: Callable[[tuple[int, ...]], bool]) -> XLACustomKernel:
    """Register a leaf's CSR event primitive with its JAX, CUDA and cuSPARSE kernels.

    GPU ``auto`` uses cuSPARSE when the source CSR primitive selects it, CUDA
    for transposed products whose cotangent shape fits ``native_dims`` and
    JAX otherwise. CPU and TPU always use JAX.
    """
    def jax_generator(**kwargs):
        return partial(jax_kernel, **kwargs)

    def auto_gpu(*, transpose: bool, csr_primitive=None, **kwargs):
        backend = None if csr_primitive is None else csr_primitive._resolve_backend('gpu')
        if backend == 'cusparse':
            return _cusparse_event_kernel(transpose=transpose, **kwargs)
        if transpose and backend in (None, 'cuda_raw') and native_dims(kwargs['ct_info'].shape):
            return cuda_kernel(transpose=transpose, **kwargs)
        return partial(jax_kernel, transpose=transpose, **kwargs)

    primitive = XLACustomKernel(name)
    primitive.def_cuda_raw_kernel(cuda_kernel)
    primitive.def_kernel('cusparse', 'gpu', _cusparse_event_kernel)
    primitive.def_kernel('auto', 'gpu', auto_gpu, asdefault=True)
    for platform in ('cpu', 'gpu', 'tpu'):
        primitive.def_kernel('jax_raw', platform, jax_generator)
    for platform in ('cpu', 'tpu'):
        primitive.def_kernel('auto', platform, jax_generator)
    primitive.def_tags('csr', tag)
    return primitive


def check_jax_backend(backend: str | None | _CSRBackend) -> None:
    """Validate a leaf's backend selection."""
    if isinstance(backend, _CSRBackend):
        return
    if backend not in (None, 'auto', 'jax_raw', 'cuda_raw', 'cusparse'):
        raise ValueError(f'unsupported sampled-product backend: {backend!r}')


def _jaxlib_cuda_major() -> int | None:
    """Return the CUDA major version of the running jaxlib GPU plugin, if known."""
    try:
        version = jax.devices('gpu')[0].client.platform_version  # e.g. 'cuda 12080'
    except Exception:  # noqa: BLE001 - no GPU client: fall back to the installed plugins
        version = ''
    match = re.search(r'cuda\s+(\d+)', version)
    if match:
        number = int(match.group(1))
        return number // 1000 if number >= 1000 else number
    majors = {
        major for major in range(11, 20)
        if importlib.util.find_spec(f'jax_cuda{major}_plugin') is not None
        or _find_spec(f'jax_plugins.xla_cuda{major}') is not None
    }
    return majors.pop() if len(majors) == 1 else None


def _find_spec(name: str):
    """``importlib.util.find_spec`` that treats a missing parent package as absent."""
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None


def _cusparse_wheel_library(roots: Iterable[str], cuda_major: int | None) -> Path | None:
    """Pick the NVIDIA wheel cuSPARSE library built for ``cuda_major``.

    CUDA 13 wheels install ``nvidia/cu13/lib``; CUDA 12 wheels install
    ``nvidia/cusparse/lib`` (or ``nvidia/cu12/lib``). Sorting alone would put
    ``cu13`` before ``cusparse`` and link a CUDA 13 library into a CUDA 12
    process. With a known major and no matching wheel, return ``None`` so the
    toolkit library is used instead; with an unknown major, keep the first.
    """
    candidates = sorted(
        candidate
        for root in roots
        for candidate in Path(root).glob('cu*/lib/libcusparse.so.*')
        if candidate.is_file()
    )
    if cuda_major is None:
        return candidates[0] if candidates else None
    wanted = {f'cu{cuda_major}'} | ({'cusparse'} if cuda_major <= 12 else set())
    return next((c for c in candidates if c.parent.parent.name in wanted), None)


@lru_cache(maxsize=1)
def _load_cusparse() -> str:
    """Load the shared SDDMM FFI against wheel or toolkit cuSPARSE libraries."""
    name = 'csr_sddmm_cusparse'
    link_flags = ['-lcusparse']
    include_paths = []
    roots = _nvidia_roots()
    library = _cusparse_wheel_library(roots, _jaxlib_cuda_major()) if roots else None
    if library is not None:
        link_flags = [str(library), f'-rpath={library.parent}']
        include_paths = [str(library.parent.parent / 'include')]
    load_cuda_file(
        Path(__file__).parent / '_cuda' / 'cusparse_sddmm.cu', name=name,
        extra_ldflags=link_flags, extra_include_paths=include_paths,
        allow_cuda_graph=False,
    )
    return name


def _cusparse_sampled(left: jax.Array, right: jax.Array, indices: jax.Array,
                      indptr: jax.Array | None, *, mode: Mode, transpose: bool) -> jax.Array:
    """Execute a CSR sampled product, isolating inactive nonfinite events."""
    if indptr is None:
        raise NotImplementedError('cusparse sampled products require CSR indices and indptr')
    if jax.default_backend() != 'gpu':
        raise ValueError('cusparse sampled products require a GPU')
    if mode != 'float':
        events, ct = (left, right.T) if transpose else (right.T, left)
        factors = event_factor(events, dtype=ct.dtype, weighted=mode == 'weighted')
        left, right = (factors, ct.T) if transpose else (ct, factors.T)
    dtype = jnp.result_type(left, right)
    if dtype not in CUDA_VALUE_DTYPES:
        raise TypeError('cusparse SDDMM operands must promote to float16, bfloat16, float32 or float64')
    if not indices.size or not left.shape[1]:
        return match_varying_axes(jnp.zeros(indices.shape, dtype), left, right, indices, indptr)
    if not _cusparse_accepts_nse(indices.size, (left.shape[0], right.shape[1])):
        raise ValueError('cusparse SDDMM requires nnz <= rows * cols; use jax_raw for larger duplicate patterns')
    compute_dtype = jnp.dtype(jnp.float64 if dtype == jnp.float64 else jnp.float32)

    # cuSPARSE needs one index type for offsets and columns. Narrow the
    # rows + 1 offsets instead of widening the nnz columns whenever every
    # offset (at most nnz) fits in int32.
    if indptr.dtype == jnp.int64 and indices.size <= _INT32_MAX:
        offsets, columns = indptr.astype(jnp.int32), indices
    else:
        offsets, columns = indptr, indices.astype(indptr.dtype)

    def product():
        module = _load_cusparse()
        result = jax.ffi.ffi_call(
            f'{module}.cusparse_sddmm{dtype_suffix(compute_dtype)}',
            jax.ShapeDtypeStruct(indices.shape, compute_dtype),
            input_output_aliases={4: 0}, vmap_method='sequential',
        )(left.astype(compute_dtype), right.T.astype(compute_dtype), columns, offsets,
          jnp.zeros(indices.shape, compute_dtype))
        return match_varying_axes(result.astype(dtype), left, right, indices, indptr)

    if mode == 'float':
        return product()

    def selected_product():
        # The matrix leaf's exact JAX selection (float32 sums for half types).
        from brainevent._sddmm._mm.sddmm import _jax_kernel
        result = _jax_kernel(events, ct, indices, indptr, transpose=transpose,
                             weighted=mode == 'weighted')[0]
        return match_varying_axes(result, left, right, indices, indptr)

    # cuSPARSE sums factor * ct over the batch, so an inactive event meeting a
    # NaN/Inf cotangent would poison its slot. Masking per slot would need the
    # gathers cuSPARSE avoids, so only nonfinite cotangents take the exact JAX
    # selection. The predicate is computed on device and does not block host
    # dispatch.
    return jax.lax.cond(jnp.all(jnp.isfinite(ct)), product, selected_product)


def _cusparse_event_kernel(*, transpose: bool, weighted: bool, **kwargs):
    """Build the common GPU event lowering for the vector and matrix leaves."""
    def kernel(events, ct, indices, indptr):
        if events.ndim == 1:
            events, ct = events[:, None], ct[:, None]
        left, right = (events, ct.T) if transpose else (ct, events.T)
        return (_cusparse_sampled(left, right, indices, indptr,
                                  mode='weighted' if weighted else 'binary', transpose=transpose),)
    return kernel
