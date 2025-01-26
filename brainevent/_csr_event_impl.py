# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


from typing import Callable, Union, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainunit.sparse._csr import _csr_to_coo
from jax.interpreters import ad

from ._csr_float_impl import _csr_matvec, _csr_matmat
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator


def _event_csr_matvec(
    data: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    indptr: jax.Array,
    v: Union[jax.Array, u.Quantity],
    *,
    shape: Sequence[int],
    transpose: bool = False,
    float_as_event: bool = True,
) -> Union[jax.Array, u.Quantity]:
    """
    Product of CSR sparse matrix and a dense vector.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      v : array of shape ``(shape[0] if transpose else shape[1],)``
        and dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      y : array of shape ``(shape[1] if transpose else shape[0],)`` representing
        the matrix vector product.
    """
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = event_csrmv_p_call(
        data, indices, indptr, v,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


def _event_csr_matmat(
    data: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    indptr: jax.Array,
    B: Union[jax.Array, u.Quantity],
    *,
    shape: Sequence[int],
    transpose: bool = False,
    float_as_event: bool = True,
) -> Union[jax.Array, u.Quantity]:
    """
    Product of CSR sparse matrix and a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
        dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
      C : array of shape ``(shape[1] if transpose else shape[0], cols)``
        representing the matrix-matrix product.
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = event_csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


Kernel = Callable


def event_csrmv_cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

            elif float_as_event:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

            else:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    w = weights[0]
                    for i in range(v.shape[0]):
                        sp = v[i]
                        if sp != 0.:
                            wsp = w * sp
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += wsp

        else:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    w = weights[0]
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += w
                        posts[i] = r

            elif float_as_event:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    w = weights[0]
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += w
                        posts[i] = r

            else:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    w = weights[0]
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            c = v[indices[j]]
                            if c != 0.:
                                r += w * c
                        posts[i] = r

    else:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

            elif float_as_event:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    for i in range(v.shape[0]):
                        if v[i] != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

            else:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    for i in range(v.shape[0]):
                        sp = v[i]
                        if sp != 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j] * sp

        else:
            if vector_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += weights[j]
                        posts[i] = r

            elif float_as_event:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] != 0.:
                                r += weights[j]
                        posts[i] = r

            else:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, v, _, posts):
                    for i in range(indptr.shape[0] - 1):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            c = v[indices[j]]
                            if c != 0.:
                                r += weights[j] * c
                        posts[i] = r

    return mv


def event_csrmv_gpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    indices_dtype = dtype_to_warp_type(indices_info.dtype)
    indptr_dtype = dtype_to_warp_type(indptr_info.dtype)
    spike_dtype = dtype_to_warp_type(vector_info.dtype)

    if weight_info.size == 1:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[i]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += w

            elif float_as_event:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[i] != 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += w

            else:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    w = weights[0]
                    sp = v[i]
                    if sp != 0.:
                        wsp = w * sp
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += wsp

        else:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]]:
                            r += w
                    posts[i] = r

            elif float_as_event:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]] != 0.:
                            r += w
                    posts[i] = r

            else:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        c = v[indices[j]]
                        if c != 0.:
                            r += w * c
                    posts[i] = r

    else:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    if v[i]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j]

            elif float_as_event:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    if v[i] != 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j]

            else:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    sp = v[i]
                    if sp != 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j] * sp

        else:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]]:
                            r += weights[j]
                    posts[i] = r

            elif float_as_event:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]] != 0.:
                            r += weights[j]
                    posts[i] = r

            else:
                @warp.kernel
                def mv(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    v: warp.array1d(dtype=spike_dtype),
                    _: warp.array1d(dtype=weight_dtype),
                    posts: warp.array1d(dtype=weight_dtype),
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        c = v[indices[j]]
                        if c != 0.:
                            r += weights[j] * c
                    posts[i] = r

    return mv


def event_csrmv_jvp_v(
    v_dot,
    data,
    indices,
    indptr,
    v,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return [
        _csr_matvec(
            data,
            indices,
            indptr,
            v_dot,
            shape=shape,
            transpose=transpose
        )
    ]


def event_csrmv_jvp_weights(
    data_dot,
    data,
    indices,
    indptr,
    v,
    _,
    *,
    shape,
    transpose,
    float_as_event,
    **kwargs
):
    return event_csrmv_p_call(
        data_dot,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
    )


def event_csrmv_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    events,
    _,
    *,
    shape,
    float_as_event,
    transpose,
    **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events)
        else:
            ct_events = _csr_matvec(
                data,
                indices,
                indptr,
                ct,
                shape=shape,
                transpose=not transpose
            )
        return data, indices, indptr, ct_events, _
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = event_csrmv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    indices,
                    indptr,
                    events,
                    shape=shape,
                    transpose=transpose,
                    float_as_event=float_as_event,
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:  # heterogeneous values
                row, col = _csr_to_coo(indices, indptr)
                ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
        return ct_values, indices, indptr, events, _


def event_csrmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = event_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = event_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event']
        )
        return r, [1]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven CSR matrix-vector product.")


def event_csrmv_p_call(
    weights,
    indices,
    indptr,
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
    return event_csrmv_p(
        weights,
        indices,
        indptr,
        v,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        float_as_event=float_as_event,
        shape=shape,
        transpose=transpose,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
    )


event_csrmv_p = XLACustomKernel(
    'event_csrmv',
    cpu_kernel=NumbaKernelGenerator(event_csrmv_cpu_kernel_generator, input_output_aliases={4: 0}),
    gpu_kernel=WarpKernelGenerator(
        event_csrmv_gpu_kernel_generator,
        dim=lambda indptr_info, vector_info, transpose, **kwargs: (
            vector_info.shape[0] if transpose else indptr_info.shape[0] - 1
        ),
        input_output_aliases={4: 0}
    ),
)
event_csrmv_p.defjvp(event_csrmv_jvp_weights, None, None, event_csrmv_jvp_v)
event_csrmv_p.def_transpose_rule(event_csrmv_transpose_rule)
event_csrmv_p.def_batching_rule(event_csrmv_batching)


def event_csrmm_cpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            # csr.T @ B

            if vector_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def mv(weights, indices, indptr, B, _, posts):
                    w = weights[0]
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

            elif float_as_event:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def mv(weights, indices, indptr, B, _, posts):
                    B = B != 0.
                    w = weights[0]
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

            else:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def mv(weights, indices, indptr, B, _, posts):
                    w = weights[0]
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            sp = B[i, k]
                            if sp != 0.:
                                wsp = w * sp
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += wsp

        else:
            # csr @ B
            if vector_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, B, _, posts):
                    w = weights[0]
                    for i in range(indptr.shape[0] - 1):
                        r = np.zeros(B.shape[1], dtype=weights.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            index = indices[j]
                            for k in range(B.shape[1]):
                                if B[index, k]:
                                    r[k] += w
                        posts[i] = r

            elif float_as_event:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, B, _, posts):
                    w = weights[0]
                    B = B != 0.
                    for i in range(indptr.shape[0] - 1):
                        r = np.zeros(B.shape[1], dtype=weights.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            index = indices[j]
                            for k in range(B.shape[1]):
                                if B[index, k]:
                                    r[k] += w
                        posts[i] = r

            else:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, B, _, posts):
                    w = weights[0]
                    for i in range(indptr.shape[0] - 1):
                        for k in range(B.shape[1]):
                            r = 0.
                            for j in range(indptr[i], indptr[i + 1]):
                                c = B[indices[j], k]
                                if c != 0.:
                                    r += w * c
                            posts[i, k] = r

    else:
        if transpose:
            # csr.T @ B

            if vector_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def mv(weights, indices, indptr, B, _, posts):
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

            elif float_as_event:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def mv(weights, indices, indptr, B, _, posts):
                    B = B != 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

            else:
                @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
                def mv(weights, indices, indptr, B, _, posts):
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            sp = B[i, k]
                            if sp != 0.:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j] * sp

        else:
            # csr @ B

            if vector_info.dtype == jnp.bool_:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, B, _, posts):
                    for i in range(indptr.shape[0] - 1):
                        for k in range(B.shape[1]):
                            r = 0.
                            for j in range(indptr[i], indptr[i + 1]):
                                if B[indices[j], k]:
                                    r += weights[j]
                            posts[i, k] = r

            elif float_as_event:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, B, _, posts):
                    B = B != 0.
                    for i in range(indptr.shape[0] - 1):
                        for k in range(B.shape[1]):
                            r = 0.
                            for j in range(indptr[i], indptr[i + 1]):
                                if B[indices[j], k]:
                                    r += weights[j]
                            posts[i, k] = r

            else:
                @numba.njit(**numba_environ.setting)
                def mv(weights, indices, indptr, B, _, posts):
                    for i in range(indptr.shape[0] - 1):
                        for k in range(B.shape[1]):
                            r = 0.
                            for j in range(indptr[i], indptr[i + 1]):
                                c = B[indices[j], k]
                                if c != 0.:
                                    r += weights[j] * c
                            posts[i, k] = r

    return mv


def event_csrmm_gpu_kernel_generator(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    spike_dtype = dtype_to_warp_type(vector_info.dtype)
    indices_dtype = dtype_to_warp_type(indices_info.dtype)
    indptr_dtype = dtype_to_warp_type(indptr_info.dtype)

    if weight_info.size == 1:
        if transpose:
            # csr.T @ B

            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    if B[i, k]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += w

            elif float_as_event:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    if B[i, k] != 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += w

            else:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    sp = B[i, k]
                    if sp != 0.:
                        wsp = w * sp
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += wsp

        else:
            # csr @ B
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k]:
                            r += w
                    posts[i, k] = r

            elif float_as_event:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k] != 0.:
                            r += w
                    posts[i, k] = r

            else:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        c = B[index, k]
                        if c != 0.:
                            r += w * c
                    posts[i, k] = r

    else:
        if transpose:
            # csr.T @ B

            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    if B[i, k]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += weights[j]

            elif float_as_event:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    if B[i, k] != 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += weights[j]

            else:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    sp = B[i, k]
                    if sp != 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += weights[j] * sp

        else:
            # csr @ B

            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k]:
                            r += weights[j]
                    posts[i, k] = r

            elif float_as_event:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k] != 0.:
                            r += weights[j]
                    posts[i, k] = r

            else:
                @warp.kernel
                def mm(
                    weights: warp.array1d(dtype=weight_dtype),
                    indices: warp.array1d(dtype=indices_dtype),
                    indptr: warp.array1d(dtype=indptr_dtype),
                    B: warp.array2d(dtype=spike_dtype),
                    _: warp.array2d(dtype=weight_dtype),
                    posts: warp.array2d(dtype=weight_dtype),
                ):
                    k, i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        c = B[index, k]
                        if c != 0.:
                            r += weights[j] * c
                    posts[i, k] = r

    return mm


def csrmm_jvp_left(
    data_dot,
    data,
    indices,
    indptr,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return [
        _csr_matmat(
            data_dot,
            indices,
            indptr,
            B,
            shape=shape,
            transpose=transpose
        )
    ]


def csrmm_jvp_right(
    B_dot,
    data,
    indices,
    indptr,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return [
        _csr_matmat(
            data,
            indices,
            indptr,
            B_dot,
            shape=shape,
            transpose=transpose
        )
    ]


def csrmm_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    B,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)

    if ad.is_undefined_primal(B):
        dB = _csr_matmat(data, indices, indptr, ct, shape=shape, transpose=not transpose)
        return data, indices, indptr, dB, _
    else:
        B = jnp.asarray(B)
        row, col = _csr_to_coo(indices, indptr)
        d_data = (ct[row] * B[col]).sum(axis=1)
        return d_data, indices, indptr, B, _


def event_csrmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = event_csrmm_p_call(
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
        r = event_csrmm_p_call(
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
        r = event_csrmm_p_call(
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


def event_csrmm_p_call(
    weights,
    indices,
    indptr,
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
    return event_csrmm_p(
        weights,
        indices,
        indptr,
        B,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


event_csrmm_p = XLACustomKernel(
    'event_csrmm',
    cpu_kernel=NumbaKernelGenerator(
        event_csrmm_cpu_kernel_generator,
        input_output_aliases={4: 0}
    ),
    gpu_kernel=WarpKernelGenerator(
        event_csrmm_gpu_kernel_generator,
        dim=lambda vector_info, indptr_info, transpose, **kwargs: (
            tuple(reversed(vector_info.shape))
            if transpose else
            [vector_info.shape[1], indptr_info.shape[0] - 1]
        ),
        input_output_aliases={4: 0}
    ),
)
event_csrmm_p.defjvp(csrmm_jvp_left, None, None, csrmm_jvp_right)
event_csrmm_p.def_transpose_rule(csrmm_transpose_rule)
event_csrmm_p.def_batching_rule(event_csrmm_batching)
