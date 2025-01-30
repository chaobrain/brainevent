# Copyright 2024- BrainPy Ecosystem Limited. All Rights Reserved.
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

from typing import Callable, Union, Sequence

import warnings
from functools import partial

import jax
import numpy as np
from jax import core, numpy as jnp
from jax.interpreters import ad, mlir
from jaxlib import gpu_sparse
import brainunit as u

from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator

Kernel = Callable

__all__ = [
    "_coo_matvec",
    "_coo_matmat",
]


def _coo_matvec(
    data: Union[jax.Array, u.Quantity],
    row: jax.Array,
    col: jax.Array,
    v: jax.Array,
    *,
    shape: Sequence[int],
    transpose: bool = False
) -> Union[jax.Array, u.Quantity]:
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = coomv_p_call(data, row, col, v, shape=shape, transpose=transpose)[0]
    return u.maybe_decimal(res * unitd * unitv)

def _coo_matmat(
    data: Union[jax.Array, u.Quantity],
    row: jax.Array,
    col: jax.Array,
    B: jax.Array,
    *,
    shape: Sequence[int],
    transpose: bool = False
):
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = coomm_p_call(data, row, col, B, shape=shape, transpose=transpose)[0]
    return u.maybe_decimal(res * (unitd * unitb))

def coomv_cpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    match (transpose, weight_info.size):
        # transpose=True, homogeneous
        case (True, 1):
            def mv(weights, row, col, v, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    posts[col[i]] += w * v[row[i]]
        # transpose=True, heterogeneous
        case (True, _):
            def mv(weights, row, col, v, _, posts):
                for i in numba.prange(row.shape[0]):
                    posts[col[i]] += weights[i] * v[row[i]]
        # transpose=False, homogeneous
        case (False, 1):
            def mv(weights, row, col, v, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    posts[row[i]] += w * v[col[i]]
        # transpose=False, heterogeneous
        case (False, _):
            def mv(weights, row, col, v, _, posts):
                for i in numba.prange(row.shape[0]):
                    posts[row[i]] += weights[i] * v[col[i]]

    mv = numba.njit(**numba_environ.setting)(mv)
    return mv

def coomv_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    row_dtype = dtype_to_warp_type(row_info.dtype)
    col_dtype = dtype_to_warp_type(col_info.dtype)
    vector_dtype = dtype_to_warp_type(vector_info.dtype)


    match (transpose, weight_info.size):
        # transpose=True, homogeneous
        case (True, 1):
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                w = weights[0]
                posts[col[i]] += w * v[row[i]]
        # transpose=True, heterogeneous
        case (True, _):
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                posts[col[i]] += weights[i] * v[row[i]]
        # transpose=False, homogeneous
        case (False, 1):
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                w = weights[0]
                posts[row[i]] += w * v[col[i]]
        # transpose=False, heterogeneous
        case (False, _):
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                posts[row[i]] += weights[i] * v[col[i]]

    mv = warp.kernel(mv)
    return mv

def coomv_jvp_v(
    v_dot,
    data,
    row,
    col,
    v,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    # return coomv_p_call(
    #     data,
    #     row,
    #     col,
    #     v_dot,
    #     shape=shape,
    #     transpose=transpose,
    # )
    return [
        _coo_matvec(
            data,
            row,
            col,
            v_dot,
            shape=shape,
            transpose=transpose
        )
    ]

def coomv_jvp_weights(
    data_dot,
    data,
    row,
    col,
    v,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return coomv_p_call(
        data_dot,
        row,
        col,
        v,
        shape=shape,
        transpose=transpose,
    )

def coomv_transpose_rule(
    ct,
    data,
    row,
    col,
    v,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    assert not ad.is_undefined_primal(row)
    assert not ad.is_undefined_primal(col)
    ct = ct[0]

    if ad.is_undefined_primal(v):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(v)
        else:
            ct_events = _coo_matvec(
                data,
                row,
                col,
                ct,
                shape=shape,
                transpose=not transpose
            )
        return data, row, col, ct_events, _
    else:
        v = jnp.asarray(v)
        if data.aval.shape[0] == 1: # scalar
            ct_values = coomv_p_call(
                jnp.ones(1, dtype=data.aval.dtype),
                row,
                col,
                v,
                shape=shape,
                transpose=transpose,
            )[0]
            ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
        else:
            ct_values = v[row] * ct[col] if transpose else v[col] * ct[row]
        return ct_values, row, col, v, _

def coomv_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose']
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose']
        )
        return r, [1]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven COO matrix-vector product.")

def coomv_p_call(
    weights,
    row,
    col,
    v,
    *,
    shape: Sequence[int],
    transpose: bool,
):
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weights.dtype)
    )

    return coomv_p(
        weights,
        row,
        col,
        v,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
        vector_info=jax.ShapeDtypeStruct(v.shape, v.dtype),

    )

coomv_p = XLACustomKernel(
    'coomv',
    cpu_kernel=NumbaKernelGenerator(coomv_cpu_kernel_generator, input_output_aliases={4:0}),
    gpu_kernel=WarpKernelGenerator(
        coomv_gpu_kernel_generator,
        dim=lambda row_info, **kwargs: (
            row_info.shape[0]
        ),
        input_output_aliases={4:0}
    )
)
coomv_p.defjvp(coomv_jvp_weights, None, None, coomv_jvp_v)
coomv_p.def_transpose_rule(coomv_transpose_rule)
coomv_p.def_batching_rule(coomv_batching)


def coomm_cpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba

    match (transpose, weight_info.size):
        # transpose=True, homogeneous
        case (True, 1):
            def mm(weights, row, col, B, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    posts[col[i], :] += w * B[row[i], :]
        # transpose=True, heterogeneous
        case (True, _):
            def mm(weights, row, col, B, _, posts):
                for i in numba.prange(row.shape[0]):
                    posts[col[i], :] += weights[i] * B[row[i], :]
        # transpose=False, homogeneous
        case (False, 1):
            def mm(weights, row, col, B, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    posts[row[i], :] += w * B[col[i], :]
        # transpose=False, heterogeneous
        case (False, _):
            def mm(weights, row, col, B, _, posts):
                for i in numba.prange(row.shape[0]):
                    posts[row[i], :] += weights[i] * B[col[i], :]

    mm = numba.njit(**numba_environ.setting)(mm)
    return mm

def coomm_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    matrix_dtype = dtype_to_warp_type(matrix_info.dtype)
    row_dtype = dtype_to_warp_type(row_info.dtype)
    col_dtype = dtype_to_warp_type(col_info.dtype)

    match (transpose, weight_info.size):
        # transpose=True, weight.size==1
        case (True, 1):
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=matrix_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                posts[col[i], :] += w * B[row[i], :]
        # transpose=True, weight.size!=1
        case (True, _):
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=matrix_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i = warp.tid()
                posts[col[i], :] += weights[i] * B[row[i], :]
        # transpose=False, weight.size==1
        case (False, 1):
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=matrix_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                posts[row[i], :] += w * B[col[i], :]
        # transpose=False, weight.size!=1
        case (False, _):
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=matrix_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i = warp.tid()
                posts[row[i], :] += weights[i] * B[col[i], :]

    mm = warp.kernel(mm)
    return mm


def coomm_jvp_left(
    data_dot,
    data,
    row,
    col,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return coomm_p_call(
        data_dot,
        row,
        col,
        B,
        shape=shape,
        transpose=transpose
    )

def coomm_jvp_right(
    B_dot,
    data,
    row,
    col,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return coomm_p_call(
        data,
        row,
        col,
        B_dot,
        shape=shape,
        transpose=transpose
    )

def coomm_transpose_rule(
    ct,
    data,
    row,
    col,
    B,
    _,
    *,
    shape,
    transpose
):
    assert not ad.is_undefined_primal(row)
    assert not ad.is_undefined_primal(col)
    # TODO: Can optimize transpose rule if data is homogenous?
    if ad.is_undefined_primal(B):
        dB = _coo_matmat(data, row, col, ct, shape=shape, transpose=not transpose)
        return data, row, col, dB, _
    else:
        B = jnp.asarray(B)
        d_data = (ct[row] * B[col]).sum(1)
        return d_data, row, col, B, _

def coomm_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = coomm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = coomm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 2, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = coomm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven COO matrix-vector product.")

def coomm_p_call(
    weights,
    row,
    col,
    B,
    *,
    shape: Sequence[int],
    transpose: bool,
):
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weights.dtype)
    )
    return coomm_p(
        weights,
        row,
        col,
        B,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
    )

coomm_p = XLACustomKernel(
    'coomm',
    cpu_kernel=NumbaKernelGenerator(
        coomm_cpu_kernel_generator,
        input_output_aliases={4:0}
    ),
    gpu_kernel=WarpKernelGenerator(
        coomm_gpu_kernel_generator,
        dim=lambda row_info, **kwargs: (
            row_info.shape[0]
        ),
        input_output_aliases={4: 0}
    )
)
coomm_p.defjvp(coomm_jvp_left, None, None, coomm_jvp_right)
coomm_p.def_transpose_rule(coomm_transpose_rule)
coomm_p.def_batching_rule(coomm_batching)