# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainunit.sparse._csr import _csr_to_coo
from jax.interpreters import ad

from ._coo_float_impl import _coo_matvec, _coo_matmat
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator

Kernel = Callable

def _event_coo_matvec(
    data: Union[jax.Array, u.Quantity],
    row: jax.Array,
    col: jax.Array,
    v: jax.Array,
    *,
    shape: Sequence[int],
    transpose: bool = False,
    float_as_event: bool = True
) -> Union[jax.Array, u.Quantity]:
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = event_coomv_p_call(
        data, row, col, v,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))

def _event_coo_matmat(
data: Union[jax.Array, u.Quantity],
    row: jax.Array,
    col: jax.Array,
    B: jax.Array,
    *,
    shape: Sequence[int],
    transpose: bool = False,
    float_as_event: bool = True
) -> Union[jax.Array, u.Quantity]:
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = event_coomm_p_call(
        data, row, col, B,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def event_coomv_cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    match (transpose, weight_info.size, vector_info.dtype, float_as_event):

        # transpose=True, homogeneous
        case (True, 1, jnp.bool_, _):
            # bool
            def mv(weights, row, col, v, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    if v[row[i]]:
                        posts[col[i]] += w
        case (True, 1, _, True):
            # float_as_event
            def mv(weights, row, col, v, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    if v[row[i]] != 0.:
                        posts[col[i]] += w
        case (True, 1, _, False):
            # other
            def mv(weights, row, col, v, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    if v[row[i]] != 0.:
                        posts[col[i]] += w * v[row[i]]

        # transpose=True, heterogeneous
        case (True, _, jnp.bool_, _):
            # bool
            def mv(weights, row, col, v, _, posts):
                for i in numba.prange(row.shape[0]):
                    if v[row[i]]:
                        posts[col[i]] += weights[i]
        case (True, _, _, True):
            # float_as_event
            def mv(weights, row, col, v, _, posts):
                for i in numba.prange(row.shape[0]):
                    if v[row[i]] != 0.:
                        posts[col[i]] += weights[i]
        case (True, _, _, False):
            # other
            def mv(weights, row, col, v, _, posts):
                for i in numba.prange(row.shape[0]):
                    if v[row[i]] != 0.:
                        posts[col[i]] += weights[i] * v[row[i]]

        # transpose=False, homogeneous
        case (False, 1, jnp.bool_, _):
            # bool
            def mv(weights, row, col, v, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    if v[col[i]]:
                        posts[row[i]] += w
        case (False, 1, _, True):
            # float_as_event
            def mv(weights, row, col, v, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    if v[col[i]] != 0.:
                        posts[row[i]] += w
        case (False, 1, _, False):
            # other
            def mv(weights, row, col, v, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    if v[col[i]] != 0.:
                        posts[row[i]] += w * v[col[i]]

        # transpose=False, heterogeneous
        case (False, _, jnp.bool_, _):
            # bool
            def mv(weights, row, col, v, _, posts):
                for i in numba.prange(row.shape[0]):
                    if v[col[i]]:
                        posts[row[i]] += weights[i]
        case (False, _, _, True):
            # float_as_event
            def mv(weights, row, col, v, _, posts):
                for i in numba.prange(row.shape[0]):
                    if v[col[i]] != 0.:
                        posts[row[i]] += weights[i]
        case (False, _, _, False):
            # other
            def mv(weights, row, col, v, _, posts):
                for i in numba.prange(row.shape[0]):
                    if v[col[i]] != 0.:
                        posts[row[i]] += weights[i] * v[col[i]]

    # 统一添加装饰器
    mv = numba.njit(**numba_environ.setting)(mv)

    return mv

def event_coomv_gpu_kernel_generator(
    float_as_event: bool,
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
    spike_dtype = dtype_to_warp_type(vector_info.dtype)

    match (transpose, weight_info.size, vector_info.dtype, float_as_event):

        # transpose=True, homogeneous
        case(True, 1, jnp.bool_, _):
            # bool
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                if v[row[i]]:
                    posts[col[i]] += w
        case(True, 1, _, True):
            # float_as_event
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                if v[row[i]] != 0.:
                    posts[col[i]] += w
        case(True, 1, _, False):
            # other
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                if v[row[i]] != 0.:
                    posts[col[i]] += w * v[row[i]]

        # transpose=True, heterogeneous
        case(True, _, jnp.bool_, _):
            # bool
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                if v[row[i]]:
                    posts[col[i]] += weights[i]
        case(True, _, _, True):
            # float_as_event
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                if v[row[i]] != 0.:
                    posts[col[i]] += weights[i]
        case(True, _, _, False):
            # other
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                if v[row[i]] != 0.:
                    posts[col[i]] += weights[i] * v[row[i]]

        # transpose=False, homogeneous
        case(False, 1, jnp.bool_, _):
            # bool
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                if v[col[i]]:
                    posts[row[i]] += w
        case(False, 1, _, True):
            # float_as_event
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                if v[col[i]] != 0.:
                    posts[row[i]] += w
        case(False, 1, _, False):
            # other
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                if v[col[i]] != 0.:
                    posts[row[i]] += w * v[col[i]]

        # transpose=False, heterogeneous
        case(False, _, jnp.bool_, _):
            # bool
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                if v[col[i]]:
                    posts[row[i]] += weights[i]
        case(False, _, _, True):
            # float_as_event
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                if v[col[i]] != 0.:
                    posts[row[i]] += weights[i]
        case(False, _, _, False):
            # other
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                v: warp.array1d(dtype=spike_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                if v[col[i]] != 0.:
                    posts[row[i]] += weights[i] * v[col[i]]
    mv = warp.kernel(mv)
    return mv

def event_coomv_jvp_v(
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

def event_coomv_jvp_weights(
    data_dot,
    data,
    row,
    col,
    v,
    _,
    *,
    shape,
    transpose,
    float_as_event,
    **kwargs
):
    return event_coomv_p_call(
        data_dot,
        row,
        col,
        v,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event
    )

def event_coomv_transpose_rule(
    ct,
    data,
    row,
    col,
    events,
    _,
    *,
    shape,
    float_as_event,
    transpose,
    **kwargs
):
    assert not ad.is_undefined_primal(row)
    assert not ad.is_undefined_primal(col)
    ct = ct[0]

    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events)
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
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = event_coomv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    row,
                    col,
                    events,
                    shape=shape,
                    transpose=transpose,
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:
                ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
        return ct_values, row, col, events, _

def event_coomv_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = event_coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event'],
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = event_coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event'],
        )
        return r, [1]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven COO matrix-vector product.")

def event_coomv_p_call(
    weights,
    row,
    col,
    v,
    *,
    shape: Sequence[int],
    transpose: bool,
    float_as_event: bool,
):
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weights.dtype)
    )

    return event_coomv_p(
        weights,
        row,
        col,
        v,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        float_as_event=float_as_event,
        shape=shape,
        transpose=transpose,
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
    )

event_coomv_p = XLACustomKernel(
    'event_coomv',
    cpu_kernel=NumbaKernelGenerator(event_coomv_cpu_kernel_generator, input_output_aliases={4:0}),
    gpu_kernel=WarpKernelGenerator(
        event_coomv_gpu_kernel_generator,
        # TODO: check if dim param is correct
        dim=lambda row_info, vector_info, transpose, **kwargs: (
            vector_info.shape[0] if transpose else row_info.shape[0] - 1
        ),
        input_output_aliases={4:0}
    ),
)
event_coomv_p.defjvp(event_coomv_jvp_weights, None, None, event_coomv_jvp_v)
event_coomv_p.def_transpose_rule(event_coomv_transpose_rule)
event_coomv_p.def_batching_rule(event_coomv_batching)

def event_coomm_cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    match (transpose, weight_info.size, vector_info.dtype, float_as_event):

        # transpose=True, homogeneous
        case (True, 1, jnp.bool_, _):
            # bool
            def mm(weights, row, col, B, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[row[i], j]:
                            posts[col[i], j] += w
        case (True, 1, _, True):
            # float_as_event
            def mm(weights, row, col, B, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[row[i], j] != 0.:
                            posts[col[i], j] += w
        case (True, 1, _, False):
            # other
            def mm(weights, row, col, B, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[row[i], j] != 0.:
                            posts[col[i], j] += w * B[row[i], j]

        # transpose=True, heterogeneous
        case (True, _, jnp.bool_, _):
            # bool
            def mm(weights, row, col, B, _, posts):
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[row[i], j]:
                            posts[col[i], j] += weights[i]
        case (True, _, _, True):
            # float_as_event
            def mm(weights, row, col, B, _, posts):
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[row[i], j] != 0.:
                            posts[col[i], j] += weights[i]
        case (True, _, _, False):
            # other
            def mm(weights, row, col, B, _, posts):
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[row[i], j] != 0.:
                            posts[col[i], j] += weights[i] * B[row[i], j]

        # transpose=False, homogeneous
        case (False, 1, jnp.bool_, _):
            # bool
            def mm(weights, row, col, B, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[col[i], j]:
                            posts[row[i], j] += w
        case (False, 1, _, True):
            # float_as_event
            def mm(weights, row, col, B, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[col[i], j] != 0.:
                            posts[row[i], j] += w
        case (False, 1, _, False):
            # other
            def mm(weights, row, col, B, _, posts):
                w = weights[0]
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[col[i], j] != 0.:
                            posts[row[i], j] += w * B[col[i], j]

        # transpose=False, heterogeneous
        case (False, _, jnp.bool_, _):
            # bool
            def mm(weights, row, col, B, _, posts):
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[col[i], j]:
                            posts[row[i], j] += weights[i]
        case (False, _, _, True):
            # float_as_event
            def mm(weights, row, col, B, _, posts):
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[col[i], j] != 0.:
                            posts[row[i], j] += weights[i]
        case (False, _, _, False):
            # other
            def mm(weights, row, col, B, _, posts):
                for i in numba.prange(row.shape[0]):
                    for j in numba.prange(B.shape[1]):
                        if B[col[i], j] != 0.:
                            posts[row[i], j] += weights[i] * B[col[i], j]

    # 统一添加装饰器
    mm = numba.njit(**numba_environ.setting)(mm)

    return mm


def event_coomm_gpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    row_dtype = dtype_to_warp_type(row_info.dtype)
    col_dtype = dtype_to_warp_type(col_info.dtype)
    spike_dtype = dtype_to_warp_type(matrix_info.dtype)

    match (transpose, weight_info.size, matrix_info.dtype, float_as_event):

        # transpose=True, homogeneous
        case(True, 1, jnp.bool_, _):
            # bool
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[row[i], j]:
                    posts[col[i], :] += w
        case(True, 1, _, True):
            # float_as_event
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[row[i], j] != 0.:
                    posts[col[i], :] += w
        case(True, 1, _, False):
            # other
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[row[i], j] != 0.:
                    posts[col[i], :] += w * B[row[i], j]

        # transpose=True, heterogeneous
        case(True, _, jnp.bool_, _):
            # bool
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                if B[row[i], j]:
                    posts[col[i], :] += weights[i]
        case(True, _, _, True):
            # float_as_event
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                if B[row[i], j] != 0.:
                    posts[col[i], :] += weights[i]
        case(True, _, _, False):
            # other
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                if B[row[i], j] != 0.:
                    posts[col[i], :] += weights[i] * B[row[i], j]

        # transpose=False, homogeneous
        case(False, 1, jnp.bool_, _):
            # bool
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[col[i], j]:
                    posts[row[i], :] += w
        case(False, 1, _, True):
            # float_as_event
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[col[i], j] != 0.:
                    posts[row[i], :] += w
        case(False, 1, _, False):
            # other
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[col[i], j] != 0.:
                    posts[row[i], :] += w * B[col[i], j]

        # transpose=False, heterogeneous
        case(False, _, jnp.bool_, _):
            # bool
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                if B[col[i], j]:
                    posts[row[i], :] += weights[i]
        case(False, _, _, True):
            # float_as_event
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                if B[col[i], j] != 0.:
                    posts[row[i], :] += weights[i]
        case(False, _, _, False):
            # other
            def mm(
                weights: warp.array1d(dtype=weight_dtype),
                row: warp.array1d(dtype=row_dtype),
                col: warp.array1d(dtype=col_dtype),
                B: warp.array2d(dtype=spike_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype)
            ):
                i, j = warp.tid()
                if B[col[i], j] != 0.:
                    posts[row[i], :] += weights[i] * B[col[i], j]
                
    mm = warp.kernel(mm)
    return mm

def event_coomm_jvp_left(
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
    return [
        _coo_matmat(
            data_dot,
            row,
            col,
            B,
            shape=shape,
            transpose=transpose
        )
    ]

def event_coomm_jvp_right(
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
    return [
        _coo_matmat(
            data,
            row,
            col,
            B_dot,
            shape=shape,
            transpose=transpose
        )
    ]

def event_coomm_transpose_rule(
    ct,
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

def event_coomm_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = event_coomm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, batch_size, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = event_coomm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 2, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = event_coomm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        r = jnp.reshape(r[0], [r[0].shape[0], n, batch_size])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven CSR matrix-vector product.")


def event_coomm_p_call(
    weights,
    row,
    col,
    B,
    *,
    shape: Sequence[int],
    transpose: bool,
    float_as_event: bool,
):
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weights.dtype)
    )
    return event_coomm_p(
        weights,
        row,
        col,
        B,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )

event_coomm_p = XLACustomKernel(
    'event_coomm',
    cpu_kernel=NumbaKernelGenerator(
        event_coomm_cpu_kernel_generator,
        input_output_aliases={4: 0}
    ),
    gpu_kernel=WarpKernelGenerator(
        event_coomm_gpu_kernel_generator,
        # TODO: check if dim param is correct
        dim=lambda matrix_info, row_info, transpose, **kwargs: (
            tuple(reversed(matrix_info.shape))
            if transpose else
            [matrix_info.shape[1], row_info.shape[0] - 1]
        ),
        input_output_aliases={4: 0}
    )
)
event_coomm_p.defjvp(event_coomm_jvp_left, None, None, event_coomm_jvp_right)
event_coomm_p.def_transpose_rule(event_coomm_transpose_rule)
event_coomm_p.def_batching_rule(event_coomm_batching)