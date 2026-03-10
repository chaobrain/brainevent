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
