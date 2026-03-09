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
from brainevent._op import XLACustomKernel, general_batching_rule, numba_kernel, load_cuda_file
from brainevent.config import get_numba_parallel
from .float import fcnmv, fcnmm

__all__ = [
    'bitpack_binary_fcnmv',
    'bitpack_binary_fcnmv_p',
    'bitpack_binary_fcnmm',
    'bitpack_binary_fcnmm_p',
]


@namescope(static_argnames=['shape', 'transpose', 'pack_axis'])
def bitpack_binary_fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    packed: jax.Array,
    spikes: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    pack_axis: int = 0,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven sparse matrix--vector product with bit-packed binary spikes.

    Computes ``y = W @ s`` (or ``y = W^T @ s`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``s`` is a bit-packed binary spike vector (uint32 words,
    32 spikes per word).

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
    spikes : jax.Array
        Original binary spike vector (bool or float).  Used only for
        autodiff (JVP / transpose); not passed to the CUDA kernel.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool, optional
        If ``False`` (default), compute ``W @ s`` (gather mode).
        If ``True``, compute ``W^T @ s`` (scatter mode).
    pack_axis : int, optional
        Axis along which ``spikes`` was packed.  Default is ``0``.

    Returns
    -------
    jax.Array or u.Quantity
        Result vector.
    """
    weights, w_unit = u.split_mantissa_unit(weights)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = bitpack_binary_fcnmv_p_call(
        weights,
        indices,
        packed,
        spikes,
        shape=shape,
        transpose=transpose,
        pack_axis=pack_axis,
    )[0]
    return u.maybe_decimal(r * w_unit)


# ---------------------------------------------------------------------------
# CUDA kernel
# ---------------------------------------------------------------------------

def _bitpack_binary_fcnmv_cuda_kernel(
    transpose: bool,
    pack_axis: int,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('bitpack_binary_fcnmv.cu'),
        name='fcn_bitpack_binary_mv',
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
        kernel_name = f'fcn_bitpack_binary_mv.bitpack_binary_fcnmv_scatter{mode_sfx}{sfx}'
    else:
        kernel_name = f'fcn_bitpack_binary_mv.bitpack_binary_fcnmv_gather{mode_sfx}{sfx}'

    def kernel(weights, indices, packed, spikes):
        return jax.ffi.ffi_call(
            kernel_name, out_info
        )(weights, indices, packed, pack_axis=pack_axis)

    return kernel


# ---------------------------------------------------------------------------
# Numba CPU kernel
# ---------------------------------------------------------------------------

def _bitpack_binary_fcnmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba
    import numpy as np

    if transpose:
        # Scatter: for each pre-neuron i, if bit i is active in packed,
        #          output[indices[i,k]] += weights[...]
        if weight_info.size == 1:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, packed, spikes, posts):
                posts[:] = 0.
                w = weights[0]
                n_pre = indices.shape[0]
                n_conn = indices.shape[1]
                for i in range(n_pre):
                    word = packed[i >> 5]
                    bit = np.uint32(1) << np.uint32(i & 31)
                    if word & bit:
                        for k in range(n_conn):
                            posts[indices[i, k]] += w
        else:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, packed, spikes, posts):
                posts[:] = 0.
                n_pre = indices.shape[0]
                n_conn = indices.shape[1]
                for i in range(n_pre):
                    word = packed[i >> 5]
                    bit = np.uint32(1) << np.uint32(i & 31)
                    if word & bit:
                        for k in range(n_conn):
                            posts[indices[i, k]] += weights[i, k]
    else:
        # Gather: for each pre-neuron i,
        #         output[i] = sum_k weights[...] * is_active(packed, indices[i,k])
        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, packed, spikes, posts):
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
            def ell_mv(weights, indices, packed, spikes, posts):
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

    def kernel(weights, indices, packed, spikes):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, packed, spikes)

    return kernel


# ---------------------------------------------------------------------------
# JVP rules
# ---------------------------------------------------------------------------

def _bitpack_binary_fcnmv_jvp_weights(
    w_dot, weights, indices, packed, spikes, *, shape, transpose, pack_axis, **kwargs
):
    return bitpack_binary_fcnmv_p_call(
        w_dot, indices, packed, spikes, shape=shape, transpose=transpose, pack_axis=pack_axis,
    )


def _bitpack_binary_fcnmv_jvp_spikes(
    spk_dot, weights, indices, packed, spikes, *, shape, transpose, **kwargs
):
    return fcnmv(weights, indices, spk_dot, shape=shape, transpose=transpose),


# ---------------------------------------------------------------------------
# Transpose rule
# ---------------------------------------------------------------------------

def _bitpack_binary_fcnmv_transpose_rule(
    ct, weights, indices, packed, spikes, *, shape, transpose, weight_info, pack_axis, **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(packed):
        raise ValueError("Cannot transpose with respect to packed data.")

    ct = ct[0]

    homo = weight_info.size == 1

    # Gradient w.r.t. spikes
    if ad.is_undefined_primal(spikes):
        if type(ct) is ad.Zero:
            ct_spk = ad.Zero(spikes)
        else:
            ct_spk = fcnmv(weights, indices, ct, shape=shape, transpose=not transpose)
        return weights, indices, packed, ct_spk

    # Gradient w.r.t. weights
    if type(ct) is ad.Zero:
        ct_gmax = ad.Zero(weights)
    elif homo:
        ct_gmax = bitpack_binary_fcnmv_p_call(
            jnp.asarray(1., dtype=weight_info.dtype),
            indices, packed, spikes,
            shape=shape, transpose=transpose, pack_axis=pack_axis,
        )[0]
        ct_gmax = jnp.inner(ct, ct_gmax).reshape(*weight_info.shape)
    else:
        if transpose:
            ct_gmax = jax.vmap(lambda v, ind: v * ct[ind])(spikes, indices)
        else:
            ct_gmax = jax.vmap(lambda c, ind: c * spikes[ind])(ct, indices)
    return ct_gmax, indices, packed, spikes


# ---------------------------------------------------------------------------
# Batching rule
# ---------------------------------------------------------------------------

def _bitpack_binary_fcnmv_batching(args, axes, **kwargs):
    # When spikes and packed are batched (axis 0), promote MV → MM
    if tuple(axes) == (None, None, 0, 0):
        # packed: (batch, n_words), spikes: (batch, n_source) → transpose for MM layout
        r = bitpack_binary_fcnmm_p_call(
            args[0], args[1],
            args[2].T,  # packed: (n_words, batch) → pack_axis=0 MM layout
            args[3].T,  # matrix: (n_source, batch) → MM layout
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            pack_axis=0,
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, 1):
        r = bitpack_binary_fcnmm_p_call(
            args[0], args[1],
            args[2],  # packed: (n_words, batch) — already pack_axis=0 layout
            args[3],  # matrix: (n_source, batch)
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            pack_axis=0,
        )
        return r, [1]
    else:
        return general_batching_rule(bitpack_binary_fcnmv_p, args, axes, **kwargs)


# ---------------------------------------------------------------------------
# Primitive call wrapper
# ---------------------------------------------------------------------------

def bitpack_binary_fcnmv_p_call(
    weights: jax.Array,
    indices: jax.Array,
    packed: jax.Array,
    spikes: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    pack_axis: int = 0,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for bit-packed binary fcnmv.

    Parameters
    ----------
    weights : jax.Array
        Weight values.
    indices : jax.Array
        Index array of shape ``(num_pre, num_conn)``.
    packed : jax.Array
        Bit-packed spike vector (uint32).
    spikes : jax.Array
        Original spike vector (for autodiff only).
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape.
    transpose : bool, optional
        Gather (False) or scatter (True) mode.
    pack_axis : int, optional
        Axis along which spikes were packed.

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
    return bitpack_binary_fcnmv_p(
        weights,
        indices,
        packed,
        spikes,
        outs=[out],
        shape=shape,
        transpose=transpose,
        pack_axis=pack_axis,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        packed_info=jax.ShapeDtypeStruct(packed.shape, packed.dtype),
    )


# ---------------------------------------------------------------------------
# Primitive registration
# ---------------------------------------------------------------------------

bitpack_binary_fcnmv_p = XLACustomKernel(
    'bitpack_binary_fcnmv',
    doc="""
Low-level XLA custom-kernel primitive for ``bitpack_binary_fcnmv``.

Performs event-driven sparse matrix-vector product using bit-packed
binary spike vectors for improved cache utilisation on GPU.

The primitive takes four positional arguments: ``(weights, indices,
packed, spikes)``.  ``packed`` (uint32) is used by the CUDA kernel for
fast bit-extraction.  ``spikes`` (the original bool array) carries
gradient information through JVP and transpose rules — it is NOT passed
to the CUDA kernel.
""",
)
bitpack_binary_fcnmv_p.def_numba_kernel(_bitpack_binary_fcnmv_numba_kernel)
bitpack_binary_fcnmv_p.def_cuda_raw_kernel(_bitpack_binary_fcnmv_cuda_kernel, asdefault=True)
bitpack_binary_fcnmv_p.def_jvp_rule2(
    _bitpack_binary_fcnmv_jvp_weights,  # arg 0: weights
    None,  # arg 1: indices (not differentiable)
    None,  # arg 2: packed (not differentiable)
    _bitpack_binary_fcnmv_jvp_spikes,  # arg 3: spikes (differentiable)
)
bitpack_binary_fcnmv_p.def_transpose_rule(_bitpack_binary_fcnmv_transpose_rule)
bitpack_binary_fcnmv_p.def_batching_rule(_bitpack_binary_fcnmv_batching)
bitpack_binary_fcnmv_p.def_call(bitpack_binary_fcnmv_p_call)
bitpack_binary_fcnmv_p.def_tags('fcn', 'binary', 'bitpack')


# ===========================================================================
# bitpack_binary_fcnmm — Matrix-Matrix product with pre-packed binary spikes
# ===========================================================================

@namescope(static_argnames=['shape', 'transpose', 'pack_axis'])
def bitpack_binary_fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    packed: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    pack_axis: int = 1,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven sparse matrix--matrix product with bit-packed binary spikes.

    Computes ``Y = W @ M`` (or ``Y = W^T @ M`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix in fixed-connection-number format
    and ``M`` is a pre-packed binary event matrix (uint32 words, 32 spikes
    per word).

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape ``(1,)`` for homogeneous or
        ``(num_pre, num_conn)`` for heterogeneous weights.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    packed : jax.Array
        Bit-packed spike matrix (uint32).

        - ``pack_axis=1``: shape ``(n_source, ceil(n_batch/32))``
        - ``pack_axis=0``: shape ``(ceil(n_source/32), n_batch)``
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
    r = bitpack_binary_fcnmm_p_call(
        weights,
        indices,
        packed,
        matrix,
        shape=shape,
        transpose=transpose,
        pack_axis=pack_axis,
    )[0]
    return u.maybe_decimal(r * w_unit)


# ---------------------------------------------------------------------------
# CUDA kernel for fcnmm
# ---------------------------------------------------------------------------

def _bitpack_binary_fcnmm_cuda_kernel(
    transpose: bool,
    pack_axis: int,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('bitpack_binary_fcnmm.cu'),
        name='fcn_bitpack_binary_mm',
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
    kernel_name = f'fcn_bitpack_binary_mm.bitpack_binary_fcnmm{dir_sfx}{mode_sfx}{axis_sfx}{sfx}'

    def kernel(weights, indices, packed, matrix):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, packed)

    return kernel


# ---------------------------------------------------------------------------
# JVP rules for fcnmm
# ---------------------------------------------------------------------------

def _bitpack_binary_fcnmm_jvp_weights(
    w_dot, weights, indices, packed, matrix, *, shape, transpose, pack_axis, **kwargs
):
    return bitpack_binary_fcnmm_p_call(
        w_dot, indices, packed, matrix, shape=shape, transpose=transpose, pack_axis=pack_axis,
    )


def _bitpack_binary_fcnmm_jvp_matrix(
    m_dot, weights, indices, packed, matrix, *, shape, transpose, **kwargs
):
    return fcnmm(weights, indices, m_dot, shape=shape, transpose=transpose),


# ---------------------------------------------------------------------------
# Transpose rule for fcnmm
# ---------------------------------------------------------------------------

def _bitpack_binary_fcnmm_transpose_rule(
    ct, weights, indices, packed, matrix, *, shape, transpose, weight_info, pack_axis, **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(packed):
        raise ValueError("Cannot transpose with respect to packed data.")

    ct = ct[0]
    homo = weight_info.size == 1

    # Gradient w.r.t. matrix
    if ad.is_undefined_primal(matrix):
        if type(ct) is ad.Zero:
            ct_mat = ad.Zero(matrix)
        else:
            ct_mat = fcnmm(weights, indices, ct, shape=shape, transpose=not transpose)
        return weights, indices, packed, ct_mat

    # Gradient w.r.t. weights
    if type(ct) is ad.Zero:
        ct_weight = ad.Zero(weights)
    elif homo:
        ct_weight = bitpack_binary_fcnmm_p_call(
            jnp.ones([1], dtype=weight_info.dtype),
            indices, packed, matrix,
            shape=shape, transpose=transpose, pack_axis=pack_axis,
        )[0]
        ct_weight = jnp.sum(ct * ct_weight).reshape(*weight_info.shape)
    else:
        if transpose:
            ct_weight = jax.vmap(lambda mat, ind: ct[ind] @ mat)(matrix, indices)
        else:
            ct_weight = jax.vmap(lambda c, ind: (matrix[ind] @ c))(ct, indices)
    return ct_weight, indices, packed, matrix


# ---------------------------------------------------------------------------
# Batching rule for fcnmm
# ---------------------------------------------------------------------------

def _bitpack_binary_fcnmm_batching(args, axes, **kwargs):
    return general_batching_rule(bitpack_binary_fcnmm_p, args, axes, **kwargs)


# ---------------------------------------------------------------------------
# Primitive call wrapper for fcnmm
# ---------------------------------------------------------------------------

def bitpack_binary_fcnmm_p_call(
    weights: jax.Array,
    indices: jax.Array,
    packed: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    pack_axis: int = 1,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for bit-packed binary fcnmm.

    Parameters
    ----------
    weights : jax.Array
        Weight values.
    indices : jax.Array
        Index array of shape ``(num_pre, num_conn)``.
    packed : jax.Array
        Bit-packed spike matrix (uint32).
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
    return bitpack_binary_fcnmm_p(
        weights,
        indices,
        packed,
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

bitpack_binary_fcnmm_p = XLACustomKernel(
    'bitpack_binary_fcnmm',
    doc="""
Low-level XLA custom-kernel primitive for ``bitpack_binary_fcnmm``.

Performs event-driven sparse matrix-matrix product using pre-packed
binary spike matrices for improved cache utilisation on GPU.

The primitive takes four positional arguments: ``(weights, indices,
packed, matrix)``.  ``packed`` (uint32) is used by the CUDA kernel for
fast bit-extraction.  ``matrix`` (the original bool array) carries
gradient information through JVP and transpose rules — it is NOT passed
to the CUDA kernel.
""",
)
bitpack_binary_fcnmm_p.def_cuda_raw_kernel(_bitpack_binary_fcnmm_cuda_kernel, asdefault=True)
bitpack_binary_fcnmm_p.def_jvp_rule2(
    _bitpack_binary_fcnmm_jvp_weights,  # arg 0: weights
    None,  # arg 1: indices
    None,  # arg 2: packed
    _bitpack_binary_fcnmm_jvp_matrix,  # arg 3: matrix
)
bitpack_binary_fcnmm_p.def_transpose_rule(_bitpack_binary_fcnmm_transpose_rule)
bitpack_binary_fcnmm_p.def_batching_rule(_bitpack_binary_fcnmm_batching)
bitpack_binary_fcnmm_p.def_call(bitpack_binary_fcnmm_p_call)
bitpack_binary_fcnmm_p.def_tags('fcn', 'binary', 'bitpack')
