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


from pathlib import Path
from typing import Tuple, Union

import brainunit as u
import jax
import jax.numpy as jnp
from jax.interpreters import ad

from brainevent._misc import namescope
from brainevent._op import XLACustomKernel, general_batching_rule, load_cuda_file, numba_kernel
from brainevent.config import get_numba_parallel
from .float import fcnmv, fcnmm

__all__ = [
    'compact_binary_fcnmv',
    'compact_binary_fcnmv_p',
    'compact_binary_fcnmm',
    'compact_binary_fcnmm_p',
]


# ===========================================================================
# compact_binary_fcnmv — Matrix-Vector product with compact binary spikes
# ===========================================================================

@namescope(static_argnames=['shape', 'transpose'])
def compact_binary_fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    packed: jax.Array,
    active_ids: jax.Array,
    n_active: jax.Array,
    spikes: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven sparse matrix--vector product with compact binary spikes.

    Computes ``y = W @ s`` (or ``y = W^T @ s`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``s`` is represented as a ``CompactBinary`` (bitpack + compaction).

    In gather mode (transpose=False), the packed uint32 spike data is used
    for efficient bit-level spike checking.

    In scatter mode (transpose=True), the active_ids + n_active data is used
    to iterate only over active (spiking) rows, skipping all inactive rows.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    packed : jax.Array
        Bit-packed spike vector of dtype uint32.  Shape ``(ceil(n/32),)``
        where ``n`` is the number of neurons along the spike dimension.
    active_ids : jax.Array
        Int32 array of active element indices, shape ``(n_orig,)``.
        Only the first ``n_active`` entries are valid.
    n_active : jax.Array
        Int32 array of shape ``(1,)`` containing the number of active elements.
    spikes : jax.Array
        Original binary spike vector (bool or float).  Used only for
        autodiff (JVP / transpose); not passed to the CUDA kernel.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool, optional
        If ``False`` (default), compute ``W @ s`` (gather mode).
        If ``True``, compute ``W^T @ s`` (scatter mode).

    Returns
    -------
    jax.Array or u.Quantity
        Result vector.
    """
    weights, w_unit = u.split_mantissa_unit(weights)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = compact_binary_fcnmv_p_call(
        weights,
        indices,
        packed,
        active_ids,
        n_active,
        spikes,
        shape=shape,
        transpose=transpose,
    )[0]
    return u.maybe_decimal(r * w_unit)


# ---------------------------------------------------------------------------
# CUDA kernel
# ---------------------------------------------------------------------------

def _compact_binary_fcnmv_cuda_kernel(
    transpose: bool,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact_binary_fcnmv.cu'),
        name='fcn_compact_binary_mv',
    )

    out_info = kwargs['outs']
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'

    if transpose:
        kernel_name = f'fcn_compact_binary_mv.compact_binary_fcnmv_scatter{mode_sfx}{sfx}'
    else:
        kernel_name = f'fcn_compact_binary_mv.compact_binary_fcnmv_gather{mode_sfx}{sfx}'

    def kernel(weights, indices, packed, active_ids, n_active, spikes):
        return jax.ffi.ffi_call(
            kernel_name, out_info
        )(weights, indices, packed, active_ids, n_active)

    return kernel


# ---------------------------------------------------------------------------
# Numba CPU kernel
# ---------------------------------------------------------------------------

def _compact_binary_fcnmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba
    import numpy as np

    if transpose:
        # Scatter: iterate only over active_ids rows
        if weight_info.size == 1:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, active_ids, n_active, posts):
                posts[:] = 0.
                n_act = n_active[0]
                w = weights[0]
                n_conn = indices.shape[1]
                for a in range(n_act):
                    i = active_ids[a]
                    for k in range(n_conn):
                        posts[indices[i, k]] += w
        else:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, active_ids, n_active, posts):
                posts[:] = 0.
                n_act = n_active[0]
                n_conn = indices.shape[1]
                for a in range(n_act):
                    i = active_ids[a]
                    for k in range(n_conn):
                        posts[indices[i, k]] += weights[i, k]

        def kernel(weights, indices, packed, active_ids, n_active, spikes):
            return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, active_ids, n_active)

    else:
        # Gather: same as bitpack_binary — use packed for bit checking
        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, packed, posts):
                w = weights[0]
                n_pre = indices.shape[0]
                n_conn = indices.shape[1]
                for i in numba.prange(n_pre):
                    r = 0.
                    for k in range(n_conn):
                        idx = indices[i, k]
                        word = packed[idx >> 5]
                        bit = np.uint32(1) << np.uint32(idx & 31)
                        if word & bit:
                            r += w
                    posts[i] = r
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, packed, posts):
                n_pre = indices.shape[0]
                n_conn = indices.shape[1]
                for i in numba.prange(n_pre):
                    r = 0.
                    for k in range(n_conn):
                        idx = indices[i, k]
                        word = packed[idx >> 5]
                        bit = np.uint32(1) << np.uint32(idx & 31)
                        if word & bit:
                            r += weights[i, k]
                    posts[i] = r

        def kernel(weights, indices, packed, active_ids, n_active, spikes):
            return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, packed)

    return kernel


# ---------------------------------------------------------------------------
# JVP rules
# ---------------------------------------------------------------------------

def _compact_binary_fcnmv_jvp_weights(
    w_dot, weights, indices, packed, active_ids, n_active, spikes,
    *, shape, transpose, **kwargs
):
    return compact_binary_fcnmv_p_call(
        w_dot, indices, packed, active_ids, n_active, spikes,
        shape=shape, transpose=transpose,
    )


def _compact_binary_fcnmv_jvp_spikes(
    spk_dot, weights, indices, packed, active_ids, n_active, spikes,
    *, shape, transpose, **kwargs
):
    return fcnmv(weights, indices, spk_dot, shape=shape, transpose=transpose),


# ---------------------------------------------------------------------------
# Transpose rule
# ---------------------------------------------------------------------------

def _compact_binary_fcnmv_transpose_rule(
    ct, weights, indices, packed, active_ids, n_active, spikes,
    *, shape, transpose, weight_info, **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(packed):
        raise ValueError("Cannot transpose with respect to packed data.")
    if ad.is_undefined_primal(active_ids):
        raise ValueError("Cannot transpose with respect to active_ids.")
    if ad.is_undefined_primal(n_active):
        raise ValueError("Cannot transpose with respect to n_active.")

    ct = ct[0]

    homo = weight_info.size == 1

    # Gradient w.r.t. spikes
    if ad.is_undefined_primal(spikes):
        if type(ct) is ad.Zero:
            ct_spk = ad.Zero(spikes)
        else:
            ct_spk = fcnmv(weights, indices, ct, shape=shape, transpose=not transpose)
        return weights, indices, packed, active_ids, n_active, ct_spk

    # Gradient w.r.t. weights
    if type(ct) is ad.Zero:
        ct_gmax = ad.Zero(weights)
    elif homo:
        ct_gmax = compact_binary_fcnmv_p_call(
            jnp.asarray(1., dtype=weight_info.dtype),
            indices, packed, active_ids, n_active, spikes,
            shape=shape, transpose=transpose,
        )[0]
        ct_gmax = jnp.inner(ct, ct_gmax).reshape(*weight_info.shape)
    else:
        if transpose:
            ct_gmax = jax.vmap(lambda v, ind: v * ct[ind])(spikes, indices)
        else:
            ct_gmax = jax.vmap(lambda c, ind: c * spikes[ind])(ct, indices)
    return ct_gmax, indices, packed, active_ids, n_active, spikes


# ---------------------------------------------------------------------------
# Batching rule
# ---------------------------------------------------------------------------

def _compact_binary_fcnmv_batching(args, axes, **kwargs):
    # args = (weights, indices, packed, active_ids, n_active, spikes)
    # Promote MV → MM when packed (arg2) and spikes (arg5) are batched.
    # Reuse per-batch packed (transposed). Compute compaction only for scatter.
    ax_w, ax_i, ax_p, ax_a, ax_n, ax_s = axes
    if (ax_w is None and ax_i is None and ax_p is not None and ax_s is not None):
        packed = args[2] if ax_p == 0 else jnp.moveaxis(args[2], ax_p, 0)
        spikes = args[5] if ax_s == 0 else jnp.moveaxis(args[5], ax_s, 0)

        # packed: (batch, n_words) → (n_words, batch) — reuse, pack_axis=0
        reused_packed = packed.swapaxes(0, 1)
        # spikes: (batch, n_source) → (n_source, batch) for MM layout
        M = spikes.swapaxes(0, 1)
        n_source = M.shape[0]

        if kwargs['transpose']:
            # Scatter mode: needs active_ids for row-skipping.
            # Use lightweight CUDA compaction-only kernel.
            from brainevent._event.compact import binary_2d_compact_only_p_call
            new_active_ids, new_n_active = binary_2d_compact_only_p_call(M)
        else:
            # Gather mode: active_ids/n_active unused by kernel — skip compaction.
            new_active_ids = jnp.zeros(n_source, dtype=jnp.int32)
            new_n_active = jnp.zeros(1, dtype=jnp.int32)

        r = compact_binary_fcnmm_p_call(
            args[0], args[1], reused_packed, new_active_ids, new_n_active, M,
            shape=kwargs['shape'], transpose=kwargs['transpose'], pack_axis=0,
        )
        return r, [1]
    else:
        return general_batching_rule(compact_binary_fcnmv_p, args, axes, **kwargs)


# ---------------------------------------------------------------------------
# Primitive call wrapper
# ---------------------------------------------------------------------------

def compact_binary_fcnmv_p_call(
    weights: jax.Array,
    indices: jax.Array,
    packed: jax.Array,
    active_ids: jax.Array,
    n_active: jax.Array,
    spikes: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for compact binary fcnmv.

    Parameters
    ----------
    weights : jax.Array
        Weight values.
    indices : jax.Array
        Index array of shape ``(num_pre, num_conn)``.
    packed : jax.Array
        Bit-packed spike vector (uint32).
    active_ids : jax.Array
        Active element indices (int32).
    n_active : jax.Array
        Active element count (int32, shape ``(1,)``).
    spikes : jax.Array
        Original spike vector (for autodiff only).
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape.
    transpose : bool, optional
        Gather (False) or scatter (True) mode.

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result vector.
    """
    n_pre, n_post = shape
    if weights.ndim == 0:
        weights = jnp.expand_dims(weights, 0)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    if transpose:
        out = jax.ShapeDtypeStruct((n_post,), weights.dtype)
    else:
        out = jax.ShapeDtypeStruct((n_pre,), weights.dtype)
    return compact_binary_fcnmv_p(
        weights,
        indices,
        packed,
        active_ids,
        n_active,
        spikes,
        outs=[out],
        shape=shape,
        transpose=transpose,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        packed_info=jax.ShapeDtypeStruct(packed.shape, packed.dtype),
    )


# ---------------------------------------------------------------------------
# Primitive registration
# ---------------------------------------------------------------------------

compact_binary_fcnmv_p = XLACustomKernel(
    'compact_binary_fcnmv',
    doc="""
Low-level XLA custom-kernel primitive for ``compact_binary_fcnmv``.

Performs event-driven sparse matrix-vector product using CompactBinary
event representation (bitpack + stream compaction).

The primitive takes six positional arguments: ``(weights, indices,
packed, active_ids, n_active, spikes)``.

- ``packed`` (uint32) is used by the gather CUDA kernel for fast bit-extraction.
- ``active_ids`` and ``n_active`` are used by the scatter CUDA kernel to iterate
  only over active rows.
- ``spikes`` (the original bool array) carries gradient information through
  JVP and transpose rules — it is NOT passed to the CUDA kernel.
""",
)
compact_binary_fcnmv_p.def_numba_kernel(_compact_binary_fcnmv_numba_kernel)
compact_binary_fcnmv_p.def_cuda_raw_kernel(_compact_binary_fcnmv_cuda_kernel, asdefault=True)
compact_binary_fcnmv_p.def_jvp_rule2(
    _compact_binary_fcnmv_jvp_weights,  # arg 0: weights
    None,  # arg 1: indices (not differentiable)
    None,  # arg 2: packed (not differentiable)
    None,  # arg 3: active_ids (not differentiable)
    None,  # arg 4: n_active (not differentiable)
    _compact_binary_fcnmv_jvp_spikes,  # arg 5: spikes (differentiable)
)
compact_binary_fcnmv_p.def_transpose_rule(_compact_binary_fcnmv_transpose_rule)
compact_binary_fcnmv_p.def_batching_rule(_compact_binary_fcnmv_batching)
compact_binary_fcnmv_p.def_call(compact_binary_fcnmv_p_call)
compact_binary_fcnmv_p.def_tags('fcn', 'binary', 'compact', 'bitpack')


# ===========================================================================
# compact_binary_fcnmm — Matrix-Matrix product with compact binary spikes
# ===========================================================================

@namescope(static_argnames=['shape', 'transpose', 'pack_axis'])
def compact_binary_fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    packed: jax.Array,
    active_ids: jax.Array,
    n_active: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    pack_axis: int = 1,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven sparse matrix--matrix product with compact binary spikes.

    Computes ``Y = W @ M`` (or ``Y = W^T @ M`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix in fixed-connection-number format
    and ``M`` is represented as a ``CompactBinary`` (bitpack + compaction).

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape ``(1,)`` for homogeneous or
        ``(num_pre, num_conn)`` for heterogeneous weights.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    packed : jax.Array
        Bit-packed spike matrix (uint32).
    active_ids : jax.Array
        Int32 array of active row indices, shape ``(n_orig,)``.
    n_active : jax.Array
        Int32 array of shape ``(1,)`` with active row count.
    matrix : jax.Array
        Original binary spike matrix (bool or float).  Used only for
        autodiff; not passed to the CUDA kernel.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the dense weight matrix.
    transpose : bool, optional
        If ``False`` (default), compute ``W @ M`` (gather mode).
        If ``True``, compute ``W^T @ M`` (scatter mode).
    pack_axis : int, optional
        Axis along which ``matrix`` was packed.  Default is ``1``.

    Returns
    -------
    jax.Array or u.Quantity
        Result matrix.
    """
    weights, w_unit = u.split_mantissa_unit(weights)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = compact_binary_fcnmm_p_call(
        weights,
        indices,
        packed,
        active_ids,
        n_active,
        matrix,
        shape=shape,
        transpose=transpose,
        pack_axis=pack_axis,
    )[0]
    return u.maybe_decimal(r * w_unit)


# ---------------------------------------------------------------------------
# CUDA kernel for fcnmm
# ---------------------------------------------------------------------------

def _compact_binary_fcnmm_cuda_kernel(
    transpose: bool,
    pack_axis: int,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('compact_binary_fcnmm.cu'),
        name='fcn_compact_binary_mm',
    )

    out_info = kwargs['outs']
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    axis_sfx = f'_a{pack_axis}'
    dir_sfx = '_scatter' if transpose else '_gather'
    kernel_name = f'fcn_compact_binary_mm.compact_binary_fcnmm{dir_sfx}{mode_sfx}{axis_sfx}{sfx}'

    def kernel(weights, indices, packed, active_ids, n_active, matrix):
        return jax.ffi.ffi_call(kernel_name, out_info)(
            weights, indices, packed, active_ids, n_active
        )

    return kernel


# ---------------------------------------------------------------------------
# Numba CPU kernel for fcnmm
# ---------------------------------------------------------------------------

def _compact_binary_fcnmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    pack_axis: int,
    **kwargs
):
    import numba
    import numpy as np

    homo = weight_info.size == 1

    if transpose:
        # Scatter mode: iterate over active rows, check packed for batch column activity
        if pack_axis == 1:
            # packed shape: (n_source, ceil(n_batch/32))
            if homo:
                @numba.njit(fastmath=True)
                def ell_mm(weights, indices, packed, active_ids, n_active, posts):
                    posts[:] = 0.
                    w = weights[0]
                    n_act = n_active[0]
                    n_conn = indices.shape[1]
                    n_batch = posts.shape[1]
                    for a in range(n_act):
                        i = active_ids[a]
                        for j in range(n_batch):
                            word = packed[i, j >> 5]
                            bit = np.uint32(1) << np.uint32(j & 31)
                            if word & bit:
                                for k in range(n_conn):
                                    posts[indices[i, k], j] += w
            else:
                @numba.njit(fastmath=True)
                def ell_mm(weights, indices, packed, active_ids, n_active, posts):
                    posts[:] = 0.
                    n_act = n_active[0]
                    n_conn = indices.shape[1]
                    n_batch = posts.shape[1]
                    for a in range(n_act):
                        i = active_ids[a]
                        for j in range(n_batch):
                            word = packed[i, j >> 5]
                            bit = np.uint32(1) << np.uint32(j & 31)
                            if word & bit:
                                for k in range(n_conn):
                                    posts[indices[i, k], j] += weights[i, k]
        else:
            # pack_axis == 0
            # packed shape: (ceil(n_source/32), n_batch)
            if homo:
                @numba.njit(fastmath=True)
                def ell_mm(weights, indices, packed, active_ids, n_active, posts):
                    posts[:] = 0.
                    w = weights[0]
                    n_act = n_active[0]
                    n_conn = indices.shape[1]
                    n_batch = posts.shape[1]
                    for a in range(n_act):
                        i = active_ids[a]
                        word_idx = i >> 5
                        bit = np.uint32(1) << np.uint32(i & 31)
                        for j in range(n_batch):
                            if packed[word_idx, j] & bit:
                                for k in range(n_conn):
                                    posts[indices[i, k], j] += w
            else:
                @numba.njit(fastmath=True)
                def ell_mm(weights, indices, packed, active_ids, n_active, posts):
                    posts[:] = 0.
                    n_act = n_active[0]
                    n_conn = indices.shape[1]
                    n_batch = posts.shape[1]
                    for a in range(n_act):
                        i = active_ids[a]
                        word_idx = i >> 5
                        bit = np.uint32(1) << np.uint32(i & 31)
                        for j in range(n_batch):
                            if packed[word_idx, j] & bit:
                                for k in range(n_conn):
                                    posts[indices[i, k], j] += weights[i, k]

        def kernel(weights, indices, packed, active_ids, n_active, matrix):
            return numba_kernel(ell_mm, outs=kwargs['outs'])(
                weights, indices, packed, active_ids, n_active
            )

    else:
        # Gather mode: same as bitpack_binary — active_ids unused
        if pack_axis == 1:
            # packed shape: (n_source, ceil(n_batch/32))
            if homo:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mm(weights, indices, packed, posts):
                    w = weights[0]
                    n_pre = indices.shape[0]
                    n_conn = indices.shape[1]
                    n_batch = posts.shape[1]
                    for i in numba.prange(n_pre):
                        for j in range(n_batch):
                            r = 0.
                            for k in range(n_conn):
                                idx = indices[i, k]
                                word = packed[idx, j >> 5]
                                bit = np.uint32(1) << np.uint32(j & 31)
                                if word & bit:
                                    r += w
                            posts[i, j] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mm(weights, indices, packed, posts):
                    n_pre = indices.shape[0]
                    n_conn = indices.shape[1]
                    n_batch = posts.shape[1]
                    for i in numba.prange(n_pre):
                        for j in range(n_batch):
                            r = 0.
                            for k in range(n_conn):
                                idx = indices[i, k]
                                word = packed[idx, j >> 5]
                                bit = np.uint32(1) << np.uint32(j & 31)
                                if word & bit:
                                    r += weights[i, k]
                            posts[i, j] = r
        else:
            # pack_axis == 0
            # packed shape: (ceil(n_source/32), n_batch)
            if homo:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mm(weights, indices, packed, posts):
                    w = weights[0]
                    n_pre = indices.shape[0]
                    n_conn = indices.shape[1]
                    n_batch = posts.shape[1]
                    for i in numba.prange(n_pre):
                        for j in range(n_batch):
                            r = 0.
                            for k in range(n_conn):
                                idx = indices[i, k]
                                word = packed[idx >> 5, j]
                                bit = np.uint32(1) << np.uint32(idx & 31)
                                if word & bit:
                                    r += w
                            posts[i, j] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mm(weights, indices, packed, posts):
                    n_pre = indices.shape[0]
                    n_conn = indices.shape[1]
                    n_batch = posts.shape[1]
                    for i in numba.prange(n_pre):
                        for j in range(n_batch):
                            r = 0.
                            for k in range(n_conn):
                                idx = indices[i, k]
                                word = packed[idx >> 5, j]
                                bit = np.uint32(1) << np.uint32(idx & 31)
                                if word & bit:
                                    r += weights[i, k]
                            posts[i, j] = r

        def kernel(weights, indices, packed, active_ids, n_active, matrix):
            return numba_kernel(ell_mm, outs=kwargs['outs'])(weights, indices, packed)

    return kernel


# ---------------------------------------------------------------------------
# JVP rules for fcnmm
# ---------------------------------------------------------------------------

def _compact_binary_fcnmm_jvp_weights(
    w_dot, weights, indices, packed, active_ids, n_active, matrix,
    *, shape, transpose, pack_axis, **kwargs
):
    return compact_binary_fcnmm_p_call(
        w_dot, indices, packed, active_ids, n_active, matrix,
        shape=shape, transpose=transpose, pack_axis=pack_axis,
    )


def _compact_binary_fcnmm_jvp_matrix(
    m_dot, weights, indices, packed, active_ids, n_active, matrix,
    *, shape, transpose, **kwargs
):
    return fcnmm(weights, indices, m_dot, shape=shape, transpose=transpose),


# ---------------------------------------------------------------------------
# Transpose rule for fcnmm
# ---------------------------------------------------------------------------

def _compact_binary_fcnmm_transpose_rule(
    ct, weights, indices, packed, active_ids, n_active, matrix,
    *, shape, transpose, weight_info, pack_axis, **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(packed):
        raise ValueError("Cannot transpose with respect to packed data.")
    if ad.is_undefined_primal(active_ids):
        raise ValueError("Cannot transpose with respect to active_ids.")
    if ad.is_undefined_primal(n_active):
        raise ValueError("Cannot transpose with respect to n_active.")

    ct = ct[0]
    homo = weight_info.size == 1

    # Gradient w.r.t. matrix
    if ad.is_undefined_primal(matrix):
        if type(ct) is ad.Zero:
            ct_mat = ad.Zero(matrix)
        else:
            ct_mat = fcnmm(weights, indices, ct, shape=shape, transpose=not transpose)
        return weights, indices, packed, active_ids, n_active, ct_mat

    # Gradient w.r.t. weights
    if type(ct) is ad.Zero:
        ct_weight = ad.Zero(weights)
    elif homo:
        ct_weight = compact_binary_fcnmm_p_call(
            jnp.ones([1], dtype=weight_info.dtype), indices,
            packed, active_ids, n_active, matrix,
            shape=shape, transpose=transpose, pack_axis=pack_axis,
        )[0]
        ct_weight = jnp.sum(ct * ct_weight).reshape(*weight_info.shape)
    else:
        if transpose:
            ct_weight = jax.vmap(lambda mat, ind: ct[ind] @ mat)(matrix, indices)
        else:
            ct_weight = jax.vmap(lambda c, ind: (matrix[ind] @ c))(ct, indices)
    return ct_weight, indices, packed, active_ids, n_active, matrix


# ---------------------------------------------------------------------------
# Batching rule for fcnmm
# ---------------------------------------------------------------------------

def _compact_binary_fcnmm_batching_base_fn(args, pack_axis, axis=1, **kwargs):
    from brainevent._event.compact import binary_2d_array_index_p_call

    # args = (weights, indices, packed, active_ids, n_active, matrix)
    assert args[5].ndim == 3, 'Batching requires 3D matrix input.'
    matrix = args[5]
    m, maybe_batch1, maybe_batch2 = matrix.shape
    M = matrix.reshape(m, maybe_batch1 * maybe_batch2)
    # Fused CUDA kernel: bitpack + compaction in one launch
    new_packed, new_active_ids, new_n_active = binary_2d_array_index_p_call(M)
    r = compact_binary_fcnmm_p_call(
        args[0],
        args[1],
        new_packed,
        new_active_ids,
        new_n_active,
        M,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        pack_axis=1,
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _compact_binary_fcnmm_batching(args, axes, **kwargs):
    pack_axis = kwargs.get('pack_axis', 1)

    if tuple(axes) == (None, None, 0, 0, 0, 0):
        assert args[5].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[5] = jnp.transpose(args[5], (1, 0, 2))
        return _compact_binary_fcnmm_batching_base_fn(
            args, pack_axis, shape=kwargs['shape'], transpose=kwargs['transpose'])

    elif tuple(axes) == (None, None, 1, 1, 1, 1):
        return _compact_binary_fcnmm_batching_base_fn(
            args, pack_axis, shape=kwargs['shape'], transpose=kwargs['transpose'])

    elif tuple(axes) == (None, None, 2, 2, 2, 2):
        return _compact_binary_fcnmm_batching_base_fn(
            args, pack_axis, axis=2, shape=kwargs['shape'], transpose=kwargs['transpose'])

    else:
        return general_batching_rule(
            compact_binary_fcnmm_p, args, axes,
            shape=kwargs['shape'], transpose=kwargs['transpose']
        )


# ---------------------------------------------------------------------------
# Primitive call wrapper for fcnmm
# ---------------------------------------------------------------------------

def compact_binary_fcnmm_p_call(
    weights: jax.Array,
    indices: jax.Array,
    packed: jax.Array,
    active_ids: jax.Array,
    n_active: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    pack_axis: int = 1,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for compact binary fcnmm.

    Parameters
    ----------
    weights : jax.Array
        Weight values.
    indices : jax.Array
        Index array of shape ``(num_pre, num_conn)``.
    packed : jax.Array
        Bit-packed spike matrix (uint32).
    active_ids : jax.Array
        Active row indices (int32).
    n_active : jax.Array
        Active row count (int32, shape ``(1,)``).
    matrix : jax.Array
        Original spike matrix (for autodiff only).
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape.
    transpose : bool, optional
        Gather (False) or scatter (True) mode.
    pack_axis : int, optional
        Axis along which matrix was packed.

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result matrix.
    """
    n_pre, n_post = shape
    n_batch = matrix.shape[1]
    if weights.ndim == 0:
        weights = jnp.expand_dims(weights, 0)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    if transpose:
        out = jax.ShapeDtypeStruct((n_post, n_batch), weights.dtype)
    else:
        out = jax.ShapeDtypeStruct((n_pre, n_batch), weights.dtype)
    return compact_binary_fcnmm_p(
        weights,
        indices,
        packed,
        active_ids,
        n_active,
        matrix,
        outs=[out],
        shape=shape,
        transpose=transpose,
        pack_axis=pack_axis,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        packed_info=jax.ShapeDtypeStruct(packed.shape, packed.dtype),
    )


# ---------------------------------------------------------------------------
# Primitive registration for fcnmm
# ---------------------------------------------------------------------------

compact_binary_fcnmm_p = XLACustomKernel(
    'compact_binary_fcnmm',
    doc="""
Low-level XLA custom-kernel primitive for ``compact_binary_fcnmm``.

Performs event-driven sparse matrix-matrix product using CompactBinary
event representation (bitpack + stream compaction).

The primitive takes six positional arguments: ``(weights, indices,
packed, active_ids, n_active, matrix)``.

- ``packed`` (uint32) is used by the gather CUDA kernel for bit-extraction.
- ``active_ids`` and ``n_active`` are used by the scatter CUDA kernel.
- ``matrix`` (the original bool array) carries gradient information — NOT
  passed to the CUDA kernel.
""",
)
compact_binary_fcnmm_p.def_numba_kernel(_compact_binary_fcnmm_numba_kernel)
compact_binary_fcnmm_p.def_cuda_raw_kernel(_compact_binary_fcnmm_cuda_kernel, asdefault=True)
compact_binary_fcnmm_p.def_jvp_rule2(
    _compact_binary_fcnmm_jvp_weights,  # arg 0: weights
    None,  # arg 1: indices
    None,  # arg 2: packed
    None,  # arg 3: active_ids
    None,  # arg 4: n_active
    _compact_binary_fcnmm_jvp_matrix,  # arg 5: matrix
)
compact_binary_fcnmm_p.def_transpose_rule(_compact_binary_fcnmm_transpose_rule)
compact_binary_fcnmm_p.def_batching_rule(_compact_binary_fcnmm_batching)
compact_binary_fcnmm_p.def_call(compact_binary_fcnmm_p_call)
compact_binary_fcnmm_p.def_tags('fcn', 'binary', 'compact', 'bitpack')
