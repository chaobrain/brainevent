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

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import Tracer
from brainevent._op import (
    XLACustomKernel,
    numba_kernel,
    general_batching_rule,
    load_cuda_file,
)


def _compact_1d_jax(x, jax_impl: bool = False):
    """JIT-compatible 1D stream compaction using prefix sum.

    Parameters
    ----------
    x : jax.Array
        1D binary array of shape ``(n,)``.

    Returns
    -------
    active_ids : jax.Array
        Shape ``(n,)``, int32.  First ``n_active`` entries are valid indices.
    n_active : jax.Array
        Shape ``(1,)``, int32.  Number of active elements.
    """
    if jax_impl:
        is_active = (x != 0).astype(jnp.int32)
        n_active = jnp.sum(is_active, dtype=jnp.int32).reshape(1)
        positions = jnp.cumsum(is_active) - 1
        n = x.shape[0]
        arange = jnp.arange(n, dtype=jnp.int32)
        # Use n as out-of-bounds position for inactive elements to avoid
        # nondeterministic overwrites from duplicate positions on GPU.
        safe_positions = jnp.where(is_active.astype(jnp.bool_), positions, n)
        active_ids = jnp.zeros(n, dtype=jnp.int32)
        active_ids = active_ids.at[safe_positions].set(arange)
        return active_ids, n_active
    else:
        return binary_1d_array_index_p_call(x)


def _compact_2d_jax(x, jax_impl: bool = False):
    """JIT-compatible 2D stream compaction along axis=0 (feature axis).

    A row is active if ANY element in that row is non-zero.
    Uses CUDA kernel when available, falls back to JAX prefix-sum.

    Parameters
    ----------
    x : jax.Array
        2D binary array of shape ``(n, batch_size)``.

    Returns
    -------
    active_ids : jax.Array
        Shape ``(n,)``, int32.
    n_active : jax.Array
        Shape ``(1,)``, int32.
    """
    if jax_impl:
        is_active = jnp.any(x != 0, axis=1).astype(jnp.int32)
        n_active = jnp.sum(is_active, dtype=jnp.int32).reshape(1)
        positions = jnp.cumsum(is_active) - 1
        n = x.shape[0]
        arange = jnp.arange(n, dtype=jnp.int32)
        # Use n as out-of-bounds position for inactive elements to avoid
        # nondeterministic overwrites from duplicate positions on GPU.
        safe_positions = jnp.where(is_active.astype(jnp.bool_), positions, n)
        active_ids = jnp.zeros(n, dtype=jnp.int32)
        active_ids = active_ids.at[safe_positions].set(arange)
        return active_ids, n_active
    else:
        return binary_2d_compact_only_p_call(x)


# ==============================================================================
# 2D compaction-only primitive (no bitpack)
# ==============================================================================

def _binary_2d_compact_only_cuda_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact.cu'),
        name='compact',
    )

    is_bool = (spikes_info.dtype == jnp.bool_)
    kernel_name = ('compact.compact_2d_only_bool'
                   if is_bool else
                   'compact.compact_2d_only_float')
    out_info = kwargs['outs']

    def kernel(spikes):
        if not is_bool and spikes.dtype != jnp.float32:
            spikes = spikes.astype(jnp.float32)
        return jax.ffi.ffi_call(kernel_name, out_info)(spikes)

    return kernel


def _binary_2d_compact_only_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    n_pre, n_batch = spikes_info.shape

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, active_ids, n_active):
            n_act = 0
            for i in range(n_pre):
                row_active = False
                for j in range(n_batch):
                    if spikes[i, j]:
                        row_active = True
                        break
                if row_active:
                    active_ids[n_act] = i
                    n_act += 1
            n_active[0] = n_act
    else:
        @numba.njit(fastmath=True)
        def mv(spikes, active_ids, n_active):
            n_act = 0
            for i in range(n_pre):
                row_active = False
                for j in range(n_batch):
                    if spikes[i, j] != 0.:
                        row_active = True
                        break
                if row_active:
                    active_ids[n_act] = i
                    n_act += 1
            n_active[0] = n_act

    def kernel(spikes):
        return numba_kernel(mv, outs=kwargs['outs'])(spikes)

    return kernel


def _binary_2d_compact_only_batching(args, axes, **kwargs):
    return general_batching_rule(binary_2d_compact_only_p, args, axes, **kwargs)


def binary_2d_compact_only_p_call(spikes, *, backend: Optional[str] = None):
    """Row-level compaction for 2D binary array (no bitpack).

    Parameters
    ----------
    spikes : jax.Array
        2D boolean or numeric array of shape ``(n_pre, n_batch)``.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    active_ids : jax.Array
        Int32 array of shape ``(n_pre,)`` with active row indices.
    n_active : jax.Array
        Int32 array of shape ``(1,)`` with the count of active rows.
    """
    n_pre = spikes.shape[0]
    active_ids_info = jax.ShapeDtypeStruct([n_pre], jnp.int32)
    n_active_info = jax.ShapeDtypeStruct([1], jnp.int32)
    return binary_2d_compact_only_p(
        spikes,
        outs=[active_ids_info, n_active_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        active_ids_info=active_ids_info,
        n_active_info=n_active_info,
        backend=backend,
    )


binary_2d_compact_only_p = XLACustomKernel(
    'binary_2d_compact_only',
    doc="""Lightweight 2D row-level compaction (no bitpack).

Identifies rows with at least one non-zero element in a 2D binary array.
Much lighter than the fused bitpack+compaction kernel when bitpack data
is already available.
""",
)
binary_2d_compact_only_p.def_numba_kernel(_binary_2d_compact_only_numba_kernel)
binary_2d_compact_only_p.def_cuda_raw_kernel(_binary_2d_compact_only_cuda_kernel)
binary_2d_compact_only_p.def_batching_rule(_binary_2d_compact_only_batching)
binary_2d_compact_only_p.def_call(binary_2d_compact_only_p_call)
binary_2d_compact_only_p.def_tags('event', 'binary')


# ==============================================================================
# 1D stream compaction primitive
# ==============================================================================

def _binary_1d_array_index_cuda_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact.cu'),
        name='compact',
    )

    is_bool = (spikes_info.dtype == jnp.bool_)
    kernel_name = ('compact.compact_1d_bool'
                   if is_bool else
                   'compact.compact_1d_float')
    out_info = kwargs['outs']

    def kernel(spikes):
        if not is_bool and spikes.dtype != jnp.float32:
            spikes = spikes.astype(jnp.float32)
        return jax.ffi.ffi_call(kernel_name, out_info)(spikes)

    return kernel


def _binary_1d_array_index_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    n = spikes_info.shape[0]

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, active_ids, n_active):
            n_act = 0
            for i in range(n):
                if spikes[i]:
                    active_ids[n_act] = i
                    n_act += 1
            n_active[0] = n_act
    else:
        @numba.njit(fastmath=True)
        def mv(spikes, active_ids, n_active):
            n_act = 0
            for i in range(n):
                if spikes[i] != 0.:
                    active_ids[n_act] = i
                    n_act += 1
            n_active[0] = n_act

    def kernel(spikes):
        return numba_kernel(mv, outs=kwargs['outs'])(spikes)

    return kernel


def _binary_1d_array_index_batching(args, axes, **kwargs):
    ax_s, = axes
    if ax_s is not None:
        spikes = args[0]
        if ax_s != 0:
            spikes = jnp.moveaxis(spikes, ax_s, 0)
        # (batch, n) → (n, batch) → 2D row-level compaction
        spikes_2d = spikes.swapaxes(0, 1)
        active_ids, n_active = binary_2d_compact_only_p_call(spikes_2d)
        # Merged result: rows active in ANY batch element.
        return (active_ids, n_active), (None, None)
    else:
        return general_batching_rule(binary_1d_array_index_p, args, axes, **kwargs)


def binary_1d_array_index_p_call(spikes, *, backend: Optional[str] = None):
    """Stream compaction for 1D binary array.

    Parameters
    ----------
    spikes : jax.Array
        1D boolean or numeric array of shape ``(n,)``.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    active_ids : jax.Array
        Int32 array of shape ``(n,)`` with active element indices.
    n_active : jax.Array
        Int32 array of shape ``(1,)`` with the count of active elements.
    """
    n = spikes.shape[0]
    active_ids_info = jax.ShapeDtypeStruct([n], jnp.int32)
    n_active_info = jax.ShapeDtypeStruct([1], jnp.int32)
    return binary_1d_array_index_p(
        spikes,
        outs=[active_ids_info, n_active_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        active_ids_info=active_ids_info,
        n_active_info=n_active_info,
        backend=backend,
    )


binary_1d_array_index_p = XLACustomKernel(
    'binary_1d_array_index',
    doc="""1D stream compaction for binary arrays.

Extracts indices of all non-zero elements in a 1D binary array.
Uses __ballot_sync + atomicAdd CUDA kernel for GPU, Numba for CPU.
""",
)
binary_1d_array_index_p.def_numba_kernel(_binary_1d_array_index_numba_kernel)
binary_1d_array_index_p.def_cuda_raw_kernel(_binary_1d_array_index_cuda_kernel)
binary_1d_array_index_p.def_batching_rule(_binary_1d_array_index_batching)
binary_1d_array_index_p.def_call(binary_1d_array_index_p_call)
binary_1d_array_index_p.def_tags('event', 'binary')


# ==============================================================================
# 2D fused bitpack + compaction primitive
# ==============================================================================

def _binary_2d_array_index_cuda_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact.cu'),
        name='compact',
    )

    is_bool = (spikes_info.dtype == jnp.bool_)
    kernel_name = ('compact.fused_bitpack_compact_2d_bool'
                   if is_bool else
                   'compact.fused_bitpack_compact_2d_float')
    out_info = kwargs['outs']

    def kernel(spikes):
        if not is_bool and spikes.dtype != jnp.float32:
            spikes = spikes.astype(jnp.float32)
        return jax.ffi.ffi_call(kernel_name, out_info)(spikes)

    return kernel


def _binary_2d_array_index_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba
    import math

    n_pre, n_batch = spikes_info.shape
    n_batch_packed = math.ceil(n_batch / 32)

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, packed, active_ids, n_active):
            n_act = 0
            for i in range(n_pre):
                row_active = False
                for w in range(n_batch_packed):
                    word = numba.uint32(0)
                    for bit in range(32):
                        col = w * 32 + bit
                        if col < n_batch and spikes[i, col]:
                            word |= numba.uint32(1) << numba.uint32(bit)
                    packed[i, w] = word
                    if word != 0:
                        row_active = True
                if row_active:
                    active_ids[n_act] = i
                    n_act += 1
            n_active[0] = n_act
    else:
        @numba.njit(fastmath=True)
        def mv(spikes, packed, active_ids, n_active):
            n_act = 0
            for i in range(n_pre):
                row_active = False
                for w in range(n_batch_packed):
                    word = numba.uint32(0)
                    for bit in range(32):
                        col = w * 32 + bit
                        if col < n_batch and spikes[i, col] != 0.:
                            word |= numba.uint32(1) << numba.uint32(bit)
                    packed[i, w] = word
                    if word != 0:
                        row_active = True
                if row_active:
                    active_ids[n_act] = i
                    n_act += 1
            n_active[0] = n_act

    def kernel(spikes):
        return numba_kernel(mv, outs=kwargs['outs'])(spikes)

    return kernel


def _binary_2d_array_index_batching(args, axes, **kwargs):
    return general_batching_rule(binary_2d_array_index_p, args, axes, **kwargs)


def binary_2d_array_index_p_call(spikes, *, backend: Optional[str] = None):
    """Fused bitpack + row-level compaction for 2D binary array.

    Parameters
    ----------
    spikes : jax.Array
        2D boolean or numeric array of shape ``(n_pre, n_batch)``.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    packed : jax.Array
        Uint32 array of shape ``(n_pre, n_batch_packed)`` with bit-packed data.
    active_ids : jax.Array
        Int32 array of shape ``(n_pre,)`` with active row indices.
    n_active : jax.Array
        Int32 array of shape ``(1,)`` with the count of active rows.
    """
    import math

    n_pre, n_batch = spikes.shape
    n_batch_packed = math.ceil(n_batch / 32)
    packed_info = jax.ShapeDtypeStruct([n_pre, n_batch_packed], jnp.uint32)
    active_ids_info = jax.ShapeDtypeStruct([n_pre], jnp.int32)
    n_active_info = jax.ShapeDtypeStruct([1], jnp.int32)
    return binary_2d_array_index_p(
        spikes,
        outs=[packed_info, active_ids_info, n_active_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        packed_info=packed_info,
        active_ids_info=active_ids_info,
        n_active_info=n_active_info,
        backend=backend,
    )


binary_2d_array_index_p = XLACustomKernel(
    'binary_2d_array_index',
    doc="""Fused 2D bitpack + row-level compaction for binary arrays.

Packs 32 binary values into each uint32 word along axis=1,
and identifies rows with at least one non-zero element.
""",
)
binary_2d_array_index_p.def_numba_kernel(_binary_2d_array_index_numba_kernel)
binary_2d_array_index_p.def_cuda_raw_kernel(_binary_2d_array_index_cuda_kernel)
binary_2d_array_index_p.def_batching_rule(_binary_2d_array_index_batching)
binary_2d_array_index_p.def_call(binary_2d_array_index_p_call)
binary_2d_array_index_p.def_tags('event', 'binary')


# ==============================================================================
# 2D Pair-Stream Encoding Primitive
# ==============================================================================

def _binary_2d_pair_stream_encode_cuda_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact.cu'),
        name='compact',
    )

    is_bool = (spikes_info.dtype == jnp.bool_)
    kernel_name = ('compact.pair_stream_encode_2d_bool'
                   if is_bool else
                   'compact.pair_stream_encode_2d_float')
    out_info = kwargs['outs']

    def kernel(spikes):
        if not is_bool and spikes.dtype != jnp.float32:
            spikes = spikes.astype(jnp.float32)
        return jax.ffi.ffi_call(kernel_name, out_info)(spikes)

    return kernel


def _binary_2d_pair_stream_encode_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    n_src, n_batch = spikes_info.shape

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, pair_stream, n_pairs):
            pair_stream[:] = 0
            write_pos = 0
            for row in range(n_src):
                for col in range(n_batch):
                    if spikes[row, col]:
                        pair_stream[write_pos, 0] = row
                        pair_stream[write_pos, 1] = col
                        write_pos += 1
            n_pairs[0] = write_pos
    else:
        @numba.njit(fastmath=True)
        def mv(spikes, pair_stream, n_pairs):
            pair_stream[:] = 0
            write_pos = 0
            for row in range(n_src):
                for col in range(n_batch):
                    if spikes[row, col] != 0.:
                        pair_stream[write_pos, 0] = row
                        pair_stream[write_pos, 1] = col
                        write_pos += 1
            n_pairs[0] = write_pos

    def kernel(spikes):
        return numba_kernel(mv, outs=kwargs['outs'])(spikes)

    return kernel


def _binary_2d_pair_stream_encode_batching(args, axes, **kwargs):
    return general_batching_rule(binary_2d_pair_stream_encode_p, args, axes, **kwargs)


def binary_2d_pair_stream_encode_p_call(
    spikes,
    *,
    backend: Optional[str] = None,
):
    """Encode a dense 2D binary matrix into a compact `(row, col)` pair stream.

    Parameters
    ----------
    spikes : jax.Array
        2D boolean or numeric array of shape ``(n_src, n_batch)``.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    pair_stream : jax.Array
        Int32 array of shape ``(n_src * n_batch, 2)``. Only the first
        ``n_pairs`` rows are valid and store zero-based ``(row, col)`` pairs.
        CUDA emits a compact active-pair stream; valid-pair order is not
        guaranteed to be row-major.
    n_pairs : jax.Array
        Int32 array of shape ``(1,)`` containing the number of valid pairs.
    """
    if spikes.ndim != 2:
        raise ValueError(f'`spikes` must be 2D, got {spikes.ndim}D.')

    n_src, n_batch = spikes.shape
    pair_stream_info = jax.ShapeDtypeStruct([n_src * n_batch, 2], jnp.int32)
    n_pairs_info = jax.ShapeDtypeStruct([1], jnp.int32)
    return binary_2d_pair_stream_encode_p(
        spikes,
        outs=[pair_stream_info, n_pairs_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        pair_stream_info=pair_stream_info,
        n_pairs_info=n_pairs_info,
        backend=backend,
    )


binary_2d_pair_stream_encode_p = XLACustomKernel(
    'binary_2d_pair_stream_encode',
    doc="""Encode a dense 2D binary matrix into a compact `(row, col)` pair stream.

Produces a static-capacity int32 buffer with shape ``(n_src * n_batch, 2)``
plus a scalar valid-length output. Only the leading ``n_pairs`` rows are used.
""",
)
binary_2d_pair_stream_encode_p.def_numba_kernel(_binary_2d_pair_stream_encode_numba_kernel)
binary_2d_pair_stream_encode_p.def_cuda_raw_kernel(_binary_2d_pair_stream_encode_cuda_kernel)
binary_2d_pair_stream_encode_p.def_batching_rule(_binary_2d_pair_stream_encode_batching)
binary_2d_pair_stream_encode_p.def_call(binary_2d_pair_stream_encode_p_call)
binary_2d_pair_stream_encode_p.def_tags('event', 'binary', 'streaming', 'pair')


# ==============================================================================
# 2D Row-Sparse Encoding Primitive
# ==============================================================================

def _binary_2d_row_sparse_encode_cuda_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    row_size: int,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact.cu'),
        name='compact',
    )

    is_bool = (spikes_info.dtype == jnp.bool_)
    kernel_name = ('compact.row_sparse_encode_2d_bool'
                   if is_bool else
                   'compact.row_sparse_encode_2d_float')
    out_info = kwargs['outs']

    def kernel(spikes):
        if not is_bool and spikes.dtype != jnp.float32:
            spikes = spikes.astype(jnp.float32)
        return jax.ffi.ffi_call(kernel_name, out_info)(spikes)

    return kernel


def _binary_2d_row_sparse_encode_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    row_size: int,
    **kwargs
):
    import numba

    n_src, n_batch = spikes_info.shape

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, spike_indices):
            spike_indices[:] = 0
            for row in range(n_src):
                write_pos = 0
                for col in range(n_batch):
                    if spikes[row, col]:
                        spike_indices[row, write_pos] = col + 1
                        write_pos += 1
    else:
        @numba.njit(fastmath=True)
        def mv(spikes, spike_indices):
            spike_indices[:] = 0
            for row in range(n_src):
                write_pos = 0
                for col in range(n_batch):
                    if spikes[row, col] != 0.:
                        spike_indices[row, write_pos] = col + 1
                        write_pos += 1

    def kernel(spikes):
        return numba_kernel(mv, outs=kwargs['outs'])(spikes)

    return kernel


def _binary_2d_row_sparse_encode_batching(args, axes, **kwargs):
    return general_batching_rule(binary_2d_row_sparse_encode_p, args, axes, **kwargs)


def _validate_binary_2d_row_sparse_capacity(spikes, row_size: int):
    if spikes.shape[0] == 0:
        return

    def _raise_if_overflow(max_row_nnz):
        max_row_nnz = int(max_row_nnz)
        if max_row_nnz > row_size:
            raise ValueError(
                f'`row_size={row_size}` is too small for the input spikes; '
                f'max row NNZ is {max_row_nnz}.'
            )

    if isinstance(spikes, Tracer):
        # JAX 0.9.x can raise effect-lowering errors for ordered debug callbacks
        # when this validation runs inside the structural custom-kernel path.
        # Keep eager validation for concrete inputs and skip tracer-time checks.
        return

    max_row_nnz = int(np.max(np.sum(np.asarray(spikes) != 0, axis=1, dtype=np.int32), initial=0))
    _raise_if_overflow(max_row_nnz)


def binary_2d_row_sparse_encode_p_call(
    spikes,
    *,
    row_size: int,
    backend: Optional[str] = None,
):
    """Encode a dense 2D binary matrix into a fixed-width FCN spike layout.

    Parameters
    ----------
    spikes : jax.Array
        2D boolean or numeric array of shape ``(n_src, n_batch)``.
    row_size : int
        Fixed number of spike slots allocated per row.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    spike_indices : jax.Array
        Int32 array of shape ``(n_src, row_size)`` storing 1-based active
        batch-column ids for each row, compacted to the front and zero-padded.
    """
    if row_size <= 0:
        raise ValueError(f'`row_size` must be positive, got {row_size}.')
    n_src, n_batch = spikes.shape
    if row_size > n_batch:
        raise ValueError(f'`row_size` must be <= n_batch={n_batch}, got {row_size}.')

    _validate_binary_2d_row_sparse_capacity(spikes, row_size)

    spike_indices_info = jax.ShapeDtypeStruct([n_src, row_size], jnp.int32)
    return binary_2d_row_sparse_encode_p(
        spikes,
        outs=[spike_indices_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        spike_indices_info=spike_indices_info,
        row_size=row_size,
        backend=backend,
    )


binary_2d_row_sparse_encode_p = XLACustomKernel(
    'binary_2d_row_sparse_encode',
    doc="""Encode a dense 2D binary matrix into a fixed-width FCN spike layout.

Produces a fixed-width int32 matrix of 1-based active batch-column ids,
one row per source neuron, with zeros used as padding.
""",
)
binary_2d_row_sparse_encode_p.def_numba_kernel(_binary_2d_row_sparse_encode_numba_kernel)
binary_2d_row_sparse_encode_p.def_cuda_raw_kernel(_binary_2d_row_sparse_encode_cuda_kernel)
binary_2d_row_sparse_encode_p.def_batching_rule(_binary_2d_row_sparse_encode_batching)
binary_2d_row_sparse_encode_p.def_call(binary_2d_row_sparse_encode_p_call)
binary_2d_row_sparse_encode_p.def_tags('event', 'binary')


# ==============================================================================
# 2D Dense-to-CSR Encoding (binary, values omitted)
# ==============================================================================

def _binary_2d_csr_row_count_cuda_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact.cu'),
        name='compact',
    )

    is_bool = (spikes_info.dtype == jnp.bool_)
    kernel_name = ('compact.csr_row_count_2d_bool'
                   if is_bool else
                   'compact.csr_row_count_2d_float')
    out_info = kwargs['outs']

    def kernel(spikes):
        if not is_bool and spikes.dtype != jnp.float32:
            spikes = spikes.astype(jnp.float32)
        return jax.ffi.ffi_call(kernel_name, out_info)(spikes)

    return kernel


def _binary_2d_csr_row_count_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    n_src, n_batch = spikes_info.shape

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, row_counts):
            for row in range(n_src):
                count = 0
                for col in range(n_batch):
                    if spikes[row, col]:
                        count += 1
                row_counts[row] = count
    else:
        @numba.njit(fastmath=True)
        def mv(spikes, row_counts):
            for row in range(n_src):
                count = 0
                for col in range(n_batch):
                    if spikes[row, col] != 0.:
                        count += 1
                row_counts[row] = count

    def kernel(spikes):
        return numba_kernel(mv, outs=kwargs['outs'])(spikes)

    return kernel


def _binary_2d_csr_row_count_batching(args, axes, **kwargs):
    return general_batching_rule(binary_2d_csr_row_count_p, args, axes, **kwargs)


def binary_2d_csr_row_count_p_call(
    spikes,
    *,
    backend: Optional[str] = None,
):
    """Count non-zero elements row-wise for a dense 2D binary matrix."""
    if spikes.ndim != 2:
        raise ValueError(f'`spikes` must be 2D, got {spikes.ndim}D.')
    row_counts_info = jax.ShapeDtypeStruct([spikes.shape[0]], jnp.int32)
    return binary_2d_csr_row_count_p(
        spikes,
        outs=[row_counts_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        row_counts_info=row_counts_info,
        backend=backend,
    )


binary_2d_csr_row_count_p = XLACustomKernel(
    'binary_2d_csr_row_count',
    doc="""Count row-wise NNZ for a dense 2D binary matrix.""",
)
binary_2d_csr_row_count_p.def_numba_kernel(_binary_2d_csr_row_count_numba_kernel)
binary_2d_csr_row_count_p.def_cuda_raw_kernel(_binary_2d_csr_row_count_cuda_kernel)
binary_2d_csr_row_count_p.def_batching_rule(_binary_2d_csr_row_count_batching)
binary_2d_csr_row_count_p.def_call(binary_2d_csr_row_count_p_call)
binary_2d_csr_row_count_p.def_tags('event', 'binary', 'csr')


def _binary_2d_csr_fill_cuda_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact.cu'),
        name='compact',
    )

    is_bool = (spikes_info.dtype == jnp.bool_)
    kernel_name = ('compact.csr_fill_2d_bool'
                   if is_bool else
                   'compact.csr_fill_2d_float')
    out_info = kwargs['outs']

    def kernel(spikes, indptr):
        if not is_bool and spikes.dtype != jnp.float32:
            spikes = spikes.astype(jnp.float32)
        return jax.ffi.ffi_call(kernel_name, out_info)(spikes, indptr)

    return kernel


def _binary_2d_csr_fill_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    n_src, n_batch = spikes_info.shape

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, indptr, indices):
            indices[:] = 0
            for row in range(n_src):
                write_pos = indptr[row]
                for col in range(n_batch):
                    if spikes[row, col]:
                        indices[write_pos] = col
                        write_pos += 1
    else:
        @numba.njit(fastmath=True)
        def mv(spikes, indptr, indices):
            indices[:] = 0
            for row in range(n_src):
                write_pos = indptr[row]
                for col in range(n_batch):
                    if spikes[row, col] != 0.:
                        indices[write_pos] = col
                        write_pos += 1

    def kernel(spikes, indptr):
        return numba_kernel(mv, outs=kwargs['outs'])(spikes, indptr)

    return kernel


def _binary_2d_csr_fill_batching(args, axes, **kwargs):
    return general_batching_rule(binary_2d_csr_fill_p, args, axes, **kwargs)


def binary_2d_csr_fill_p_call(
    spikes,
    indptr,
    *,
    backend: Optional[str] = None,
):
    """Fill a flat CSR index buffer from dense 2D binary spikes and row pointers."""
    if spikes.ndim != 2:
        raise ValueError(f'`spikes` must be 2D, got {spikes.ndim}D.')
    if indptr.ndim != 1:
        raise ValueError(f'`indptr` must be 1D, got {indptr.ndim}D.')
    if indptr.shape[0] != spikes.shape[0] + 1:
        raise ValueError(
            f'`indptr.shape[0]` must equal spikes.shape[0] + 1 = {spikes.shape[0] + 1}, '
            f'got {indptr.shape[0]}.'
        )

    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices_info = jax.ShapeDtypeStruct([spikes.shape[0] * spikes.shape[1]], jnp.int32)
    return binary_2d_csr_fill_p(
        spikes,
        indptr,
        outs=[indices_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        indices_info=indices_info,
        backend=backend,
    )


binary_2d_csr_fill_p = XLACustomKernel(
    'binary_2d_csr_fill',
    doc="""Fill a flat CSR column-index buffer using precomputed row pointers.""",
)
binary_2d_csr_fill_p.def_numba_kernel(_binary_2d_csr_fill_numba_kernel)
binary_2d_csr_fill_p.def_cuda_raw_kernel(_binary_2d_csr_fill_cuda_kernel)
binary_2d_csr_fill_p.def_batching_rule(_binary_2d_csr_fill_batching)
binary_2d_csr_fill_p.def_call(binary_2d_csr_fill_p_call)
binary_2d_csr_fill_p.def_tags('event', 'binary', 'csr')


def binary_2d_csr_encode_p_call(
    spikes,
    *,
    backend: Optional[str] = None,
):
    """Encode a dense 2D binary matrix into a CSR-like event structure.

    Returns a static-capacity CSR buffer suitable for ``jax.jit``:

    - ``indices`` has shape ``(n_src * n_batch,)`` and stores valid column
      ids in ``indices[:indptr[-1]]``.
    - ``indptr`` has shape ``(n_src + 1,)`` and is a standard CSR row-pointer
      array with ``indptr[0] == 0``.

    ``indices`` tail elements beyond ``indptr[-1]`` are zero-filled.
    """
    if spikes.ndim != 2:
        raise ValueError(f'`spikes` must be 2D, got {spikes.ndim}D.')

    row_counts, = binary_2d_csr_row_count_p_call(
        spikes,
        backend=backend,
    )
    row_counts = jnp.asarray(row_counts, dtype=jnp.int32)
    indptr = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(row_counts, dtype=jnp.int32)],
        axis=0,
    )
    indices, = binary_2d_csr_fill_p_call(
        spikes,
        indptr,
        backend=backend,
    )
    return indices, indptr
