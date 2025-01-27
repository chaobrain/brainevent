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
from brainunit.sparse._csr import _csr_to_coo
from jax.interpreters import ad

from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator

Kernel = Callable


def _csr_matvec(
    data: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    indptr: jax.Array,
    id: jax.Array,
    v: Union[jax.Array, u.Quantity],
    *,
    shape: Sequence[int],
    transpose: bool = False
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
    res = csrmv_p_call(data, indices, indptr, id, v, shape=shape, transpose=transpose)[0]
    return u.maybe_decimal(res * unitd * unitv)


def _csr_matmat(
    data: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    indptr: jax.Array,
    B: Union[jax.Array, u.Quantity],
    *,
    shape: Sequence[int],
    transpose: bool = False,
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
    res = csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def csrmv_cpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            @numba.njit(**numba_environ.setting)
            def mv(weights, indices, indptr, id, v, _, posts):
                w = weights[0]
                for i in range(v.shape[0]):
                    wsp = w * v[i]
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += wsp


        else:
            @numba.njit(**numba_environ.setting)
            def mv(weights, indices, indptr, id, v, _, posts):
                w = weights[0]
                for i in range(indptr.shape[0] - 1):
                    r = 0.
                    for j in range(indptr[i], indptr[i + 1]):
                        r += w * v[indices[j]]
                    posts[i] = r

    else:
        if transpose:
            @numba.njit(**numba_environ.setting)
            def mv(weights, indices, indptr, id, v, _, posts):
                for i in range(v.shape[0]):
                    sp = v[i]
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += weights[j] * sp

        else:
            @numba.njit(**numba_environ.setting)
            def mv(weights, indices, indptr, id, v, _, posts):
                for i in range(indptr.shape[0] - 1):
                    r = 0.
                    for j in range(indptr[i], indptr[i + 1]):
                        r += weights[j] * v[indices[j]]
                    posts[i] = r

    return mv


def csrmv_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    id_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    indices_dtype = dtype_to_warp_type(indices_info.dtype)
    indptr_dtype = dtype_to_warp_type(indptr_info.dtype)
    vector_dtype = dtype_to_warp_type(vector_info.dtype)
    id_dtype = dtype_to_warp_type(id_info.dtype)

    indices_shape = indices_info.shape[0]

    if weight_info.size == 1:
        if transpose:
            @warp.kernel
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                id: warp.array1d(dtype=id_info),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                lborder = i * 1024
                rborder = min(lborder + 1024, indices_shape)
                w = weights[0]
                for k in range(id[i],id[i+1]+1):
                    sp = v[k]
                    if sp != 0.:
                        wsp = w * sp
                        posl = max(indptr[k], lborder)
                        posr = min(indptr[k+1], rborder)
                        for j in range(posl, posr):
                            warp.atomic_add(posts,indices[j],wsp)

        else:
            @warp.kernel
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                id: warp.array1d(dtype=id_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                lborder = i * 1024
                rborder = min(lborder + 1024, indices_shape)
                w = weights[0]
                for k in range(id[i],id[i+1]+1):
                    r = weights.dtype(0.)
                    posl = max(indptr[k], lborder)
                    posr = min(indptr[k+1], rborder)
                    for j in range(posl, posr):
                        c = v[indices[j]]
                        if c != 0.:
                            r += w * c
                    warp.atomic_add(posts,k,r)

    else:
        if transpose:
            @warp.kernel
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                id: warp.array1d(dtype=id_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                lborder = i * 1024
                rborder = min(lborder + 1024, indices_shape)
                for k in range(id[i], id[i+1]+1):
                    sp = v[k]
                    if sp != 0.:
                        posl = max(indptr[k], lborder)
                        posr = min(indptr[k+1], rborder)
                        for j in range(posl, posr):
                            warp.atomic_add(posts,indices[j],weights[j]*sp)

        else:
            @warp.kernel
            def mv(
                weights: warp.array1d(dtype=weight_dtype),
                indices: warp.array1d(dtype=indices_dtype),
                indptr: warp.array1d(dtype=indptr_dtype),
                id: warp.array1d(dtype=id_dtype),
                v: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                i = warp.tid()
                lborder = i * 1024
                rborder = min(lborder + 1024, indices_shape)
                for k in range(id[i], id[i+1]+1):
                    r = weights.dtype(0.)
                    posl = max(indptr[k], lborder)
                    posr = min(indptr[k+1], rborder)
                    for j in range(posl, posr):
                        c = v[indices[j]]
                        if c != 0.:
                            r += weights[j] * c
                    warp.atomic_add(posts,k,r)

    return mv


def csrmv_jvp_v(
    v_dot,
    data,
    indices,
    indptr,
    id,
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
            id,
            v_dot,
            shape=shape,
            transpose=transpose
        )
    ]


def csrmv_jvp_weights(
    data_dot,
    data,
    indices,
    indptr,
    id,
    v,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return csrmv_p_call(
        data_dot,
        indices,
        indptr,
        id,
        v,
        shape=shape,
        transpose=transpose,
    )


def csrmv_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    id,
    vector,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(vector):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(vector)
        else:
            ct_events = _csr_matvec(
                data,
                indices,
                indptr,
                id,
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
                ct_values = csrmv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    indices,
                    indptr,
                    id,
                    vector,
                    shape=shape,
                    transpose=transpose,
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:  # heterogeneous values
                row, col = _csr_to_coo(indices, indptr)
                ct_values = vector[row] * ct[col] if transpose else vector[col] * ct[row]
        return ct_values, indices, indptr, vector, _


def csrmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[4].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[4].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1, None):
        assert args[4].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven CSR matrix-vector product.")


def csrmv_p_call(
    weights,
    indices,
    indptr,
    id,
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
    return csrmv_p(
        weights,
        indices,
        indptr,
        id,
        v,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        id_info=jax.ShapeDtypeStruct(id.shape, id.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
    )


csrmv_p = XLACustomKernel(
    'csrmv',
    cpu_kernel=NumbaKernelGenerator(csrmv_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        csrmv_gpu_kernel_generator,
        dim=lambda id_info, **kwargs: (
            id_info.shape[0] - 1
        ),
        input_output_aliases={5: 0}
    ),
)
csrmv_p.defjvp(csrmv_jvp_weights, None, None, csrmv_jvp_v)
csrmv_p.def_transpose_rule(csrmv_transpose_rule)
csrmv_p.def_batching_rule(csrmv_batching)


def csrmm_cpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
) -> Kernel:
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            # csr.T @ B
            @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
            def mm(weights, indices, indptr, B, _, posts):
                w = weights[0]
                for k in numba.prange(B.shape[1]):
                    for i in range(B.shape[0]):
                        wsp = w * B[i, k]
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += wsp

        else:
            # csr @ B
            @numba.njit(**numba_environ.setting)
            def mm(weights, indices, indptr, B, _, posts):
                w = weights[0]
                for i in range(indptr.shape[0] - 1):
                    for k in range(B.shape[1]):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            r += w * B[indices[j], k]
                        posts[i, k] = r

    else:
        if transpose:
            # csr.T @ B
            @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
            def mm(weights, indices, indptr, B, _, posts):
                for k in numba.prange(B.shape[1]):
                    for i in range(B.shape[0]):
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += weights[j] * B[i, k]

        else:
            # csr @ B
            @numba.njit(**numba_environ.setting)
            def mm(weights, indices, indptr, B, _, posts):
                for i in range(indptr.shape[0] - 1):
                    for k in range(B.shape[1]):
                        r = 0.
                        for j in range(indptr[i], indptr[i + 1]):
                            r += weights[j] * B[indices[j], k]
                        posts[i, k] = r

    return mm


def csrmm_gpu_kernel_generator(
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
                wsp = w * B[i, k]
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j], k] += wsp

        else:
            # csr @ B
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
                    r += w * B[index, k]
                posts[i, k] = r

    else:
        if transpose:
            # csr.T @ B
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
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j], k] += weights[j] * sp

        else:
            # csr @ B

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
                    r += weights[j] * B[index, k]
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


def csrmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = csrmm_p_call(
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
        m, batch_size, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = csrmm_p_call(
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
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], n, batch_size])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven CSR matrix-vector product.")


def csrmm_p_call(
    weights,
    indices,
    indptr,
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
    return csrmm_p(
        weights,
        indices,
        indptr,
        B,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


csrmm_p = XLACustomKernel(
    'csrmm',
    cpu_kernel=NumbaKernelGenerator(
        csrmm_cpu_kernel_generator,
        input_output_aliases={4: 0}
    ),
    gpu_kernel=WarpKernelGenerator(
        csrmm_gpu_kernel_generator,
        dim=lambda vector_info, indptr_info, transpose, **kwargs: (
            tuple(reversed(vector_info.shape))
            if transpose else
            [vector_info.shape[1], indptr_info.shape[0] - 1]
        ),
        input_output_aliases={4: 0}
    ),
)
csrmm_p.defjvp(csrmm_jvp_left, None, None, csrmm_jvp_right)
csrmm_p.def_transpose_rule(csrmm_transpose_rule)
csrmm_p.def_batching_rule(csrmm_batching)
