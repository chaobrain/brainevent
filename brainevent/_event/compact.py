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
from jax.interpreters import ad

from brainevent._misc import namescope
from brainevent._op import (
    XLACustomKernel,
    numba_kernel,
    general_batching_rule,
    jaxinfo_to_warpinfo,
    load_cuda_file,
    BenchmarkConfig,
)



def _compact_1d_jax(x):
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


def _compact_2d_jax(x):
    """JIT-compatible 2D stream compaction along axis=0 (feature axis).

    A row is active if ANY element in that row is non-zero.

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


@namescope
def binary_array_index(spikes, *, backend: Optional[str] = None):
    """Extract indices and count of non-zero elements from a binary spike array.

    Scans a 1-D or 2-D array and returns the positions of all non-zero
    entries together with the total count.

    Parameters
    ----------
    spikes : jax.Array
        A 1-D or 2-D boolean or numeric array.  Non-zero entries are
        treated as active events.
    backend : str or None, optional
        Compute backend for the extraction kernel (e.g. ``'numba'``,
        ``'pallas'``).  ``None`` selects the default.

    Returns
    -------
    For 1-D input:
        indices : jax.Array
            Int32 array containing the positions of non-zero elements.
            Only the first ``count[0]`` entries are valid.
        count : jax.Array
            Int32 array of shape ``(1,)`` with the number of non-zero elements.

    For 2-D input:
        packed : jax.Array
            Uint32 array of shape ``(n_pre, ceil(n_batch/32))`` with bit-packed rows.
        active_ids : jax.Array
            Int32 array of shape ``(n_pre,)`` with indices of active rows.
            Only the first ``n_active[0]`` entries are valid.
        n_active : jax.Array
            Int32 array of shape ``(1,)`` with the number of active rows.

    Raises
    ------
    ValueError
        If *spikes* has more than 2 dimensions.

    See Also
    --------
    """
    if spikes.ndim == 1:
        indices, count = binary_1d_array_index_p_call(spikes, backend=backend)
        return indices, count
    elif spikes.ndim == 2:
        packed, active_ids, n_active = binary_2d_array_index_p_call(spikes, backend=backend)
        return packed, active_ids, n_active
    else:
        raise ValueError("Only 1D and 2D binary arrays are supported for index extraction.")


# ==============================================================================
# 1D stream compaction primitive
# ==============================================================================

def _binary_1d_array_index_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, indices, count):
            idx = 0
            for i in range(spikes.shape[0]):
                if spikes[i]:
                    indices[idx] = i
                    idx += 1
            count[0] = idx
    else:
        @numba.njit(fastmath=True)
        def mv(spikes, indices, count):
            idx = 0
            for i in range(spikes.shape[0]):
                if spikes[i] != 0.:
                    indices[idx] = i
                    idx += 1
            count[0] = idx

    def kernel(spikes):
        return numba_kernel(mv, outs=kwargs['outs'])(spikes)

    return kernel


def _binary_1d_array_index_pallas_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    BLOCK_SIZE = 64

    def _raw_kernel(
        spikes_ref,
        indices_ref,
        count_ref,
    ):
        pid = pl.program_id(0)
        start = pid * BLOCK_SIZE
        idxs = start + jnp.arange(0, BLOCK_SIZE)

        # Check valid indices
        valid_mask = idxs < spikes_ref.shape[0]

        # Load values using direct indexing
        if spikes_info.dtype == jnp.bool_:
            x_vals = spikes_ref[idxs]
            value_mask = x_vals
        else:
            x_vals = spikes_ref[idxs]
            value_mask = x_vals != 0.0

        # Apply valid mask
        combined_mask = valid_mask & value_mask

        # Count non-zero elements in this block
        total_in_block = jnp.sum(combined_mask.astype(jnp.int32))

        # Atomically reserve space in global count
        base_pos = atomic_add(count_ref, (0,), total_in_block, mask=combined_mask[0:1])
        prefix_offsets = jnp.cumsum(combined_mask) - combined_mask

        # Calculate write positions
        write_positions = base_pos + prefix_offsets

        # Store indices using direct assignment
        indices_ref[write_positions] = jnp.where(combined_mask, idxs, 0)

    def kernel(spikes, indices, count):
        dim = spikes_info.shape[0]
        if dim == 0:
            return jnp.zeros_like(indices), jnp.zeros_like(count)

        num_blocks = pl.cdiv(spikes_info.shape[0], BLOCK_SIZE)
        fn = pl.pallas_call(_raw_kernel, grid=(num_blocks,), out_shape=kwargs['outs'], backend='triton')
        indices, count = fn(spikes, indices, count)

        # Keep deterministic ordering consistent with numba/CPU implementation.
        valid_mask = jnp.arange(dim) < count[0]
        sentinel = jnp.asarray(dim, dtype=indices.dtype)
        sorted_indices = jnp.sort(jnp.where(valid_mask, indices, sentinel))
        indices = jnp.where(valid_mask, sorted_indices, 0)
        return indices, count

    return kernel


def _binary_1d_array_index_cuda_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact.cu'),
        name='compact',
    )

    is_bool = (spikes_info.dtype == jnp.bool_)
    kernel_name = 'compact.compact_1d_bool' if is_bool else 'compact.compact_1d_float'
    out_info = kwargs['outs']

    def kernel(spikes):
        if not is_bool and spikes.dtype != jnp.float32:
            spikes = spikes.astype(jnp.float32)
        return jax.ffi.ffi_call(kernel_name, out_info)(spikes)

    return kernel


def _binary_1d_array_index_jvp_spikes(spikes_dot, spikes, **kwargs):
    return binary_1d_array_index_p_call(spikes_dot, backend=kwargs['backend'], )


def _binary_1d_array_index_transpose_rule(ct, spikes, indices, count, **kwargs):
    ct_indices, ct_count = ct
    if ad.is_undefined_primal(spikes):
        if type(ct_indices) is ad.Zero and type(ct_count) is ad.Zero:
            ct_spikes = ad.Zero(spikes)
        else:
            # Gradient: sum of gradients at indexed positions
            ct_spikes = jnp.zeros_like(spikes)
            if type(ct_indices) is not ad.Zero:
                valid_count = count if type(count) is not ad.Zero else 0
                if type(ct_indices) is not ad.Zero:
                    ct_spikes = ct_spikes.at[ct_indices].add(1.0)
        return ct_spikes, indices, count
    else:
        return spikes, indices, count


def _binary_1d_array_index_batching(args, axes, **kwargs):
    return general_batching_rule(binary_1d_array_index_p, args, axes, **kwargs)


def _binary_1d_array_index_benchmark_data(*, platform):
    n = 1000
    configs = []
    for bool_event in (True, False):
        if bool_event:
            spikes = jnp.asarray(np.random.rand(n) > 0.9, dtype=jnp.bool_)
        else:
            spikes = jnp.asarray(
                np.where(np.random.rand(n) > 0.9, np.random.rand(n), 0.0),
                dtype=jnp.float32,
            )
        name = "bool" if bool_event else "float"
        configs.append(BenchmarkConfig(name, (spikes,)))
    return configs


def binary_1d_array_index_p_call(spikes, *, backend: Optional[str] = None):
    """Dispatch the 1-D binary index extraction primitive.

    Parameters
    ----------
    spikes : jax.Array
        A 1-D boolean or numeric array.
    backend : str or None, optional
        Compute backend (e.g. ``'numba'``, ``'pallas'``, ``'cuda_raw'``).

    Returns
    -------
    indices : jax.Array
        Int32 array of shape ``(n,)`` with non-zero positions.
    count : jax.Array
        Int32 array of shape ``(1,)`` with the count of non-zero elements.
    """
    indices_info = jax.ShapeDtypeStruct([spikes.shape[0]], jnp.int32)
    count_info = jax.ShapeDtypeStruct([1], jnp.int32)
    return binary_1d_array_index_p(
        spikes,
        outs=[indices_info, count_info],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=indices_info,
        count_info=count_info,
        backend=backend,
    )


binary_1d_array_index_p = XLACustomKernel(
    'binary_1d_array_index',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_array_index``.

This ``XLACustomKernel`` instance dispatches the binary 1D array indexing
operation to registered backends (``numba``, ``pallas``, ``cuda_raw``),
using runtime shape/dtype metadata provided by the high-level wrapper.

Extracts indices of non-zero elements from a binary event array, enabling
efficient event-driven processing by identifying which elements are active.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_1d_array_index_p.available_backends(platform)``,
and the default backend can be configured with ``binary_1d_array_index_p.set_default(platform, backend)``.

See Also
--------
binary_array_index : High-level user-facing function wrapper.
"""
)


def _binary_1d_array_index_warp_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    spikes_warp_info = jaxinfo_to_warpinfo(spikes_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    count_warp_info = jaxinfo_to_warpinfo(count_info)

    if spikes_info.dtype == jnp.bool_:
        @warp.kernel
        def mv(
            spikes: spikes_warp_info,
            indices: indices_warp_info,
            count: count_warp_info,
        ):
            i_col_block = warp.tid()
            if spikes[i_col_block]:
                idx = warp.atomic_add(count, 0, 1)
                indices[idx] = i_col_block

    else:
        @warp.kernel
        def mv(
            spikes: spikes_warp_info,
            indices: indices_warp_info,
            count: count_warp_info,
        ):
            i_col_block = warp.tid()
            if spikes[i_col_block] != 0.:
                idx = warp.atomic_add(count, 0, 1)
                indices[idx] = i_col_block

    def kernel(spikes):
        dim = spikes_info.shape[0]
        if dim == 0:
            return (
                jnp.zeros(indices_info.shape, dtype=indices_info.dtype),
                jnp.zeros(count_info.shape, dtype=count_info.dtype),
            )

        indices = jnp.zeros(indices_info.shape, dtype=indices_info.dtype)
        count = jnp.zeros(count_info.shape, dtype=count_info.dtype)
        fn = jax_kernel(
            mv,
            launch_dims=[dim],
            num_outputs=2,
            in_out_argnames=['indices', 'count'],
        )
        indices, count = fn(spikes, indices, count)

        # atomic_add-based writes are nondeterministic in order; enforce ascending
        # order on the valid prefix so all backends match the reference behavior.
        valid_mask = jnp.arange(dim) < count[0]
        sentinel = jnp.asarray(dim, dtype=indices.dtype)
        sorted_indices = jnp.sort(jnp.where(valid_mask, indices, sentinel))
        indices = jnp.where(valid_mask, sorted_indices, 0)
        return indices, count

    return kernel

binary_1d_array_index_p.def_numba_kernel(_binary_1d_array_index_numba_kernel)
binary_1d_array_index_p.def_warp_kernel(_binary_1d_array_index_warp_kernel)
binary_1d_array_index_p.def_pallas_kernel('gpu', _binary_1d_array_index_pallas_kernel)
binary_1d_array_index_p.def_cuda_raw_kernel(_binary_1d_array_index_cuda_kernel)
binary_1d_array_index_p.def_jvp_rule2(_binary_1d_array_index_jvp_spikes)
binary_1d_array_index_p.def_transpose_rule(_binary_1d_array_index_transpose_rule)
binary_1d_array_index_p.def_batching_rule(_binary_1d_array_index_batching)
binary_1d_array_index_p.def_call(binary_1d_array_index_p_call)
binary_1d_array_index_p.def_tags('event', 'binary')
binary_1d_array_index_p.def_benchmark_data(_binary_1d_array_index_benchmark_data)


# ==============================================================================
# 2D fused bit-pack + row-level compaction primitive
# ==============================================================================

def _ceil_div(a, b):
    return (a + b - 1) // b


def _binary_2d_array_index_numba_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    n_pre, n_batch = spikes_info.shape
    n_batch_packed = _ceil_div(n_batch, 32)

    if spikes_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def mv(spikes, packed, active_ids, n_active):
            n_act = 0
            for i in range(n_pre):
                row_active = False
                for w in range(n_batch_packed):
                    word = numba.types.uint32(0)
                    for b in range(32):
                        col = w * 32 + b
                        if col < n_batch and spikes[i, col]:
                            word |= numba.types.uint32(1) << numba.types.uint32(b)
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
                    word = numba.types.uint32(0)
                    for b in range(32):
                        col = w * 32 + b
                        if col < n_batch and spikes[i, col] != 0.:
                            word |= numba.types.uint32(1) << numba.types.uint32(b)
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


def _binary_2d_array_index_jvp_spikes(spikes_dot, spikes, **kwargs):
    return binary_2d_array_index_p_call(spikes_dot, backend=kwargs['backend'])


def _binary_2d_array_index_transpose_rule(ct, spikes, packed, active_ids, n_active, **kwargs):
    ct_packed, ct_active_ids, ct_n_active = ct
    if ad.is_undefined_primal(spikes):
        if (type(ct_packed) is ad.Zero and
                type(ct_active_ids) is ad.Zero and
                type(ct_n_active) is ad.Zero):
            ct_spikes = ad.Zero(spikes)
        else:
            ct_spikes = jnp.zeros_like(spikes)
        return ct_spikes, packed, active_ids, n_active
    else:
        return spikes, packed, active_ids, n_active


def _binary_2d_array_index_batching(args, axes, **kwargs):
    return general_batching_rule(binary_2d_array_index_p, args, axes, **kwargs)


def _binary_2d_array_index_benchmark_data(*, platform):
    configs = []
    for n_pre, n_batch in [(100, 64), (1000, 128), (10000, 256)]:
        for bool_event in (True, False):
            if bool_event:
                spikes = jnp.asarray(np.random.rand(n_pre, n_batch) > 0.99, dtype=jnp.bool_)
            else:
                spikes = jnp.asarray(
                    np.where(np.random.rand(n_pre, n_batch) > 0.99,
                             np.random.rand(n_pre, n_batch), 0.0),
                    dtype=jnp.float32,
                )
            dtype_name = "bool" if bool_event else "float"
            configs.append(BenchmarkConfig(f"{dtype_name}_{n_pre}x{n_batch}", (spikes,)))
    return configs


def binary_2d_array_index_p_call(spikes, *, backend: Optional[str] = None):
    """Dispatch the 2-D binary fused bit-pack + compaction primitive.

    Computes both a bit-packed representation and a list of active (non-zero)
    row indices from a 2-D binary array.

    Parameters
    ----------
    spikes : jax.Array
        A 2-D boolean or numeric array of shape ``(n_pre, n_batch)``.
    backend : str or None, optional
        Compute backend (e.g. ``'numba'``, ``'cuda_raw'``).

    Returns
    -------
    packed : jax.Array
        Uint32 array of shape ``(n_pre, n_batch_packed)`` where
        ``n_batch_packed = ceil(n_batch / 32)``.  Bit ``b`` of
        ``packed[i, w]`` corresponds to ``spikes[i, w*32 + b]``.
    active_ids : jax.Array
        Int32 array of shape ``(n_pre,)`` containing row indices with at
        least one non-zero column.  Only the first ``n_active[0]`` entries
        are valid.
    n_active : jax.Array
        Int32 array of shape ``(1,)`` with the count of active rows.
    """
    n_pre, n_batch = spikes.shape
    n_batch_packed = _ceil_div(n_batch, 32)

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
    doc="""
Low-level XLA custom-kernel primitive for 2-D binary array indexing.

This ``XLACustomKernel`` instance dispatches the fused bit-pack +
row-level compaction operation to registered backends (``numba``,
``cuda_raw``).

Given a 2-D binary array of shape ``(n_pre, n_batch)``, it produces:
  1. A bit-packed representation ``packed`` of shape ``(n_pre, ceil(n_batch/32))``
     where each uint32 word packs 32 batch columns.
  2. An ``active_ids`` array listing row indices that contain at least one
     non-zero element.
  3. A scalar ``n_active`` counting the number of active rows.

See Also
--------
binary_array_index : High-level user-facing function wrapper.
binary_1d_array_index_p : 1-D variant.
"""
)

binary_2d_array_index_p.def_numba_kernel(_binary_2d_array_index_numba_kernel)
binary_2d_array_index_p.def_cuda_raw_kernel(_binary_2d_array_index_cuda_kernel)
binary_2d_array_index_p.def_jvp_rule2(_binary_2d_array_index_jvp_spikes)
binary_2d_array_index_p.def_transpose_rule(_binary_2d_array_index_transpose_rule)
binary_2d_array_index_p.def_batching_rule(_binary_2d_array_index_batching)
binary_2d_array_index_p.def_call(binary_2d_array_index_p_call)
binary_2d_array_index_p.def_tags('event', 'binary')
binary_2d_array_index_p.def_benchmark_data(_binary_2d_array_index_benchmark_data)
