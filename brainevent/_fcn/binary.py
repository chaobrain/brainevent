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


from typing import Optional, Tuple, Union

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, check_fixed_conn_num_shape, namescope
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._typing import MatrixShape
from .float import fcnmv_p_call, fcnmm_p_call

__all__ = [
    'binary_fcnmv',
    'binary_fcnmv_p',
    'binary_fcnmm',
    'binary_fcnmm_p',
]


@namescope(static_argnames=['shape', 'transpose'])
def binary_fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    spikes: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
) -> Union[jax.Array, u.Quantity]:
    weights, w_unit = u.split_mantissa_unit(weights)
    spikes, v_unit = u.split_mantissa_unit(spikes)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = binary_fcnmv_p_call(
        weights,
        indices,
        spikes,
        shape=shape,
        transpose=transpose,
    )[0]
    return u.maybe_decimal(r * v_unit * w_unit)


def _binary_fcnmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i] > 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w
        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i] > 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]

    else:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(parallel=True, fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    w = weights[0]
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spikes[index]:
                                r += w
                        posts[i] = r
            else:
                @numba.njit(parallel=True, fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    spk_bool = spikes > 0.
                    w = weights[0]
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spk_bool[index]:
                                r += w
                        posts[i] = r
        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(parallel=True, fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spikes[index]:
                                r += weights[i, j]
                        posts[i] = r
            else:
                @numba.njit(parallel=True, fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    spk_bool = spikes > 0.
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spk_bool[index]:
                                r += weights[i, j]
                        posts[i] = r

    def kernel(weights, indices, spikes):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, spikes)

    return kernel


def _binary_fcnmv_warp_kernel(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    spike_warp_info = jaxinfo_to_warpinfo(spike_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @warp.kernel
                def ell_mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    spikes: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    w = weights[0]
                    if spikes[i]:
                        for j in range(indices.shape[1]):
                            warp.atomic_add(posts, indices[i, j], w)
            else:
                @warp.kernel
                def ell_mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    spikes: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    w = weights[0]
                    if spikes[i] > 0.:
                        for j in range(indices.shape[1]):
                            warp.atomic_add(posts, indices[i, j], w)
        else:
            if spike_info.dtype == jnp.bool_:
                @warp.kernel
                def ell_mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    spikes: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    if spikes[i]:
                        for j in range(indices.shape[1]):
                            warp.atomic_add(posts, indices[i, j], weights[i, j])
            else:
                @warp.kernel
                def ell_mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    spikes: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    if spikes[i] > 0.:
                        for j in range(indices.shape[1]):
                            warp.atomic_add(posts, indices[i, j], weights[i, j])

        def kernel(weights, indices, spikes):
            out_info = kwargs['outs'][0]
            dim = spike_info.shape[0]
            fn = jax_kernel(ell_mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, spikes, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @warp.kernel
                def ell_mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    spikes: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):
                        if spikes[indices[i, j]]:
                            r += w
                    posts[i] = r
            else:
                @warp.kernel
                def ell_mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    spikes: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):
                        if spikes[indices[i, j]] > 0.:
                            r += w
                    posts[i] = r
        else:
            if spike_info.dtype == jnp.bool_:
                @warp.kernel
                def ell_mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    spikes: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):
                        if spikes[indices[i, j]]:
                            r += weights[i, j]
                    posts[i] = r
            else:
                @warp.kernel
                def ell_mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    spikes: spike_warp_info,
                    posts: out_warp_info
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indices.shape[1]):
                        if spikes[indices[i, j]] > 0.:
                            r += weights[i, j]
                    posts[i] = r

        def kernel(weights, indices, spikes):
            out_info = kwargs['outs'][0]
            dim = indices_info.shape[0]
            fn = jax_kernel(ell_mv, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, spikes)

    return kernel


def _binary_fcnmv_pallas_kernel(
    transpose: int,
    shape: Tuple[int, int],
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    if len(shape) > 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    block_dim = generate_block_dim(indices_info.shape[1], maximum=256)

    if transpose:
        # Sparse Matrix: [k, m]
        # vector: [k]

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            vector_ref,  # [n_pre]
            _,
            out_ref,  # [n_post]
        ):
            i_row = pl.program_id(0)
            vector = vector_ref[i_row]

            @pl.when(vector > 0. if vector_ref.dtype > jnp.bool_ else vector)
            def run():
                if homo:
                    wv = weight_ref[0]
                    homo_data = jnp.ones(block_dim, dtype=weight_info.dtype) * wv

                def loop_fn(i_col_block, _):
                    i_col = i_col_block * block_dim
                    mask = i_col + jnp.arange(block_dim) < n_conn
                    ind = index_ref[i_row, pl.dslice(i_col, block_dim)]
                    ind = jnp.where(mask, ind, 0)
                    if homo:
                        data = homo_data
                    else:
                        data = weight_ref[i_row, pl.dslice(i_col, block_dim)]
                        data = jnp.where(mask, data, 0.0)
                    atomic_add(out_ref, ind, data, mask=mask)

                jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre,),
                input_output_aliases={3: 0},
                out_shape=kwargs['outs']
            )
            out_info = kwargs['outs'][0]
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)
            return fn(weights, indices, vector, placeholder)

    else:
        # Sparse Matrix: [m, k]
        # vector: [k]

        def _raw_kernel(
            weight_ref,  # [1]
            index_ref,  # [n_pre, n_conn]
            vector_ref,  # [n_post]
            _,
            out_ref,  # [n_pre]
        ):
            i_row = pl.program_id(0)

            def loop_fn(i_col_block, out):
                i_col = i_col_block * block_dim
                mask = i_col + jnp.arange(block_dim) < n_conn
                ind = index_ref[i_row, pl.dslice(i_col, block_dim)]
                ind = jnp.where(mask, ind, 0)
                vec = vector_ref[ind]
                vec = jnp.where(mask, vec, 0.0 if vector_ref.dtype > jnp.bool_ else False)
                if homo:
                    return out + jnp.sum(jnp.asarray(vec, dtype=out_ref.dtype))
                else:
                    weight = weight_ref[i_row, pl.dslice(i_col, block_dim)]
                    weight = jnp.where(mask, weight, 0.0)
                    if vector_ref.dtype == jnp.bool_:
                        weight = jnp.where(vec, weight, 0.)
                    else:
                        weight = jnp.where(vec > 0., weight, 0.)
                    return out + jnp.sum(weight)

            i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
            if homo:
                i_row_sum = i_row_sum * weight_ref[0]
            out_ref[i_row] = i_row_sum

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(_raw_kernel, grid=(n_pre,), out_shape=kwargs['outs'])
            return fn(weights, indices, vector)

    return kernel


def _binary_fcnmv_jvp_spikes(spk_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return fcnmv_p_call(weights, indices, spk_dot, shape=shape, transpose=transpose)


def _binary_fcnmv_jvp_weights(w_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return binary_fcnmv_p_call(w_dot, indices, spikes, shape=shape, transpose=transpose)


def _binary_fcnmv_transpose_rule(ct, weights, indices, spikes, *, shape, transpose, weight_info, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(spikes):
        if type(ct) is ad.Zero:
            ct_spk = ad.Zero(spikes)
        else:
            ct_spk = fcnmv_p_call(weights, indices, ct, shape=shape, transpose=not transpose)[0]
        return weights, indices, ct_spk

    else:
        # ∂L/∂w = ∂L/∂y * ∂y/∂w
        if type(ct) is ad.Zero:
            ct_gmax = ad.Zero(weights)
        elif homo:
            # scalar
            ct_gmax = binary_fcnmv_p_call(
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                spikes,
                shape=shape,
                transpose=transpose,
            )
            ct_gmax = jnp.inner(ct, ct_gmax[0]).reshape(*weight_info.shape)
        else:
            if transpose:
                ct_gmax = jax.vmap(lambda v, ind: v * ct[ind])(spikes, indices)
            else:
                ct_gmax = jax.vmap(lambda c, ind: c * spikes[ind])(ct, indices)
        return ct_gmax, indices, spikes


def _binary_fcnmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_fcnmm_p_call(
            args[0],
            args[1],
            args[2].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_fcnmm_p_call(
            args[0],
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]
    else:
        return general_batching_rule(binary_fcnmv_p, args, axes, **kwargs)


def _binary_fcnmv_benchmark_data(*, platform):
    import numpy as _np
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                n_conn = max(1, int(n_post * prob))
                indices = jnp.asarray(_np.random.randint(0, n_post, (n_pre, n_conn), dtype=_np.int32))
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.ones((n_pre, n_conn), dtype=dtype)
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    spikes = jnp.asarray(_np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    spikes = jnp.asarray(_np.random.rand(v_size), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, indices, spikes),
                        {'shape': (n_pre, n_post), 'transpose': transpose}
                    )
                )
    return configs


def binary_fcnmv_p_call(
    weights: jax.Array,
    indices: jax.Array,
    spikes: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, spikes, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return binary_fcnmv_p(
        weights,
        indices,
        spikes,
        outs=[out],
        shape=shape,
        transpose=transpose,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        backend=backend,
    )


binary_fcnmv_p = XLACustomKernel('binary_fcnmv', )
binary_fcnmv_p.def_numba_kernel(_binary_fcnmv_numba_kernel)
binary_fcnmv_p.def_warp_kernel(_binary_fcnmv_warp_kernel)
binary_fcnmv_p.def_pallas_kernel('gpu', _binary_fcnmv_pallas_kernel)
binary_fcnmv_p.def_pallas_kernel('tpu', _binary_fcnmv_pallas_kernel)
binary_fcnmv_p.def_jvp_rule2(_binary_fcnmv_jvp_weights, None, _binary_fcnmv_jvp_spikes, None)
binary_fcnmv_p.def_transpose_rule(_binary_fcnmv_transpose_rule)
binary_fcnmv_p.def_batching_rule(_binary_fcnmv_batching)
binary_fcnmv_p.def_call(binary_fcnmv_p_call)
binary_fcnmv_p.def_tags('fcn', 'binary')
binary_fcnmv_p.def_benchmark_data(_binary_fcnmv_benchmark_data)


@namescope(static_argnames=['shape', 'transpose'])
def binary_fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Union[jax.Array, u.Quantity]:
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = binary_fcnmm_p_call(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
    )[0]
    return u.maybe_decimal(r * m_unit * w_unit)


def _binary_fcnmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # fixed pre connection number
        #
        # CSR: [k, m]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i_k in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i_k])
                        for i_conn in range(indices.shape[1]):
                            posts[indices[i_k, i_conn], nonzero] += w
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i_k in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i_k] > 0.)
                        for i_conn in range(indices.shape[1]):
                            posts[indices[i_k, i_conn], nonzero] += w
        else:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    for i in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i])
                        for j in range(indices.shape[1]):
                            posts[indices[i, j], nonzero] += weights[i, j]
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    for i in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i] > 0.)
                        for j in range(indices.shape[1]):
                            posts[indices[i, j], nonzero] += weights[i, j]

    else:
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(parallel=True, fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    w = weights[0]
                    for i_m in numba.prange(indices.shape[0]):
                        posts[i_m] = w * np.sum(matrix[indices[i_m]], axis=0)
            else:
                @numba.njit(parallel=True, fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    w = weights[0]
                    for i_m in numba.prange(indices.shape[0]):
                        events = matrix[indices[i_m]] > 0.
                        posts[i_m] = w * np.sum(events, axis=0)
        else:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(parallel=True, fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    for i_m in numba.prange(indices.shape[0]):
                        posts[i_m] = weights[i_m] @ (matrix[indices[i_m]]).astype(weights.dtype)
            else:
                @numba.njit(parallel=True, fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    for i_m in numba.prange(indices.shape[0]):
                        events = (matrix[indices[i_m]] > 0.).astype(weights.dtype)
                        posts[i_m] = weights[i_m] @ events

    def kernel(weights, indices, matrix):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, matrix)

    return kernel


def _binary_fcnmm_pallas_kernel(
    shape: MatrixShape,
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    if len(shape) > 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    block_k = generate_block_dim(indices_info.shape[1], maximum=128)
    block_n = generate_block_dim(matrix_info.shape[1], maximum=128)

    if transpose:
        #
        # fixed pre connection number
        #
        # - CSR: [k, m]
        # - matrix: [k, n]
        #

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            matrix_ref,  # [k, n]
            _,
            out_ref,  # [n_pre, n]
        ):
            i_k = pl.program_id(0)
            i_n_block = pl.program_id(1)
            i_n_start = i_n_block * block_n
            i_n_mask = i_n_start + jnp.arange(block_n) < matrix_ref.shape[1]
            if homo:
                weight = jnp.full(block_k, weight_ref[0])

            def loop_fn(i_index_block, _):
                i_index_start = i_index_block * block_k
                i_index_mask = i_index_start + jnp.arange(block_k) < n_conn
                ind = index_ref[i_k, pl.dslice(i_index_start, block_k)]
                ind = jnp.where(i_index_mask, ind, 0)
                mat = matrix_ref[i_k, pl.dslice(i_n_start, block_n)]
                mat = jnp.where(i_n_mask, mat, 0.0)
                mat = jnp.asarray(mat, dtype=weight_ref.dtype)
                if homo:
                    A = weight
                else:
                    A = weight_ref[i_k, pl.dslice(i_index_start, block_k)]
                    A = jnp.where(i_index_mask, A, 0.0)
                data = A[:, None] * mat[None, :]
                atomic_add(out_ref, (ind, pl.dslice(i_n_start, block_n)), data,
                           mask=i_index_mask[:, None] & i_n_mask[None, :])

            jax.lax.fori_loop(0, pl.cdiv(n_conn, block_k), loop_fn, None)

        def kernel(weights, indices, matrix, _):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre, pl.cdiv(matrix_info.shape[1], block_n)),
                input_output_aliases={3: 0},
                out_shape=kwargs['outs']
            )
            out_info = kwargs['outs'][0]
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)
            return fn(weights, indices, matrix, placeholder)

    else:

        #
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            matrix_ref,  # [k, n]
            _,
            out_ref,  # [n_pre, n]
        ):
            i_m = pl.program_id(0)
            i_n_block = pl.program_id(1)
            i_n_start = i_n_block * block_n
            i_n_mask = i_n_start + jnp.arange(block_n) < matrix_ref.shape[1]

            def loop_fn(i_k_block, out):
                i_k_start = i_k_block * block_k
                i_k_mask = i_k_start + jnp.arange(block_k) < n_conn
                ind = index_ref[i_m, pl.dslice(i_k_start, block_k)]
                ind = jnp.where(i_k_mask, ind, 0)
                mat = matrix_ref[ind, pl.dslice(i_n_start, block_n)]
                mat = jnp.where(i_k_mask[:, None] & i_n_mask[None, :], mat, 0.0)
                if homo:
                    inc = mat.sum(axis=0)
                else:
                    mat = jnp.asarray(mat, dtype=weight_ref.dtype)
                    weight = weight_ref[i_m, pl.dslice(i_k_start, block_k)]
                    weight = jnp.where(i_k_mask, weight, 0.0)
                    inc = (weight[:, None] * mat).sum(axis=0)
                return out + inc

            final_out = jax.lax.fori_loop(
                0,
                pl.cdiv(n_conn, block_k),
                loop_fn,
                jnp.zeros(block_n, dtype=matrix_ref.dtype)
            )
            if homo:
                final_out = final_out * weight_ref[0]
            out_ref[i_m, pl.dslice(i_n_start, block_n)] = jnp.where(i_n_mask, final_out, 0.0)

        def kernel(weights, indices, matrix, _):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre, pl.cdiv(matrix_info.shape[1], block_n)),
                out_shape=kwargs['outs']
            )
            return fn(weights, indices, matrix, _)

    return kernel


def _binary_fcnmm_jvp_matrix(matrix_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return fcnmm_p_call(weights, indices, matrix_dot, shape=shape, transpose=transpose)


def _binary_fcnmm_jvp_weights(weights_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return binary_fcnmm_p_call(weights_dot, indices, matrix, shape=shape, transpose=transpose)


def _binary_fcnmm_transpose_rule(ct, weights, indices, matrix, *, shape, transpose, weight_info, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(matrix):
        if type(ct) is ad.Zero:
            ct_vector = ad.Zero(matrix)

        else:
            ct_vector = fcnmm_p_call(
                weights,
                indices,
                ct,
                shape=shape,
                transpose=not transpose
            )[0]

        return weights, indices, ct_vector
    else:
        # ∂L/∂w = ∂L/∂y * ∂y/∂w
        if type(ct) is ad.Zero:
            ct_weight = ad.Zero(weights)

        elif homo:
            ct_weight = binary_fcnmm_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                matrix,
                shape=shape,
                transpose=transpose,
            )[0]
            ct_weight = jnp.sum(ct * ct_weight).reshape(*weight_info.shape)

        else:
            if transpose:
                # inputs: [k, n] @ [k, n_conn]
                # ct: [m, n]
                ct_weight = jax.vmap(lambda mat, ind: ct[ind] @ mat)(matrix, indices)
            else:
                # inputs: [m, n] @ [m, n_conn]
                # ct: [k, n]
                ct_weight = jax.vmap(lambda c, ind: (matrix[ind] @ c))(ct, indices)
        return ct_weight, indices, matrix


def _batching_base_fn(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_fcnmm_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _binary_fcnmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 1):
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 2):
        return _batching_base_fn(args, axis=2, **kwargs)

    else:
        return general_batching_rule(binary_fcnmm_p, args, axes, **kwargs)


def _binary_fcnmm_benchmark_data(*, platform):
    import numpy as _np
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                n_conn = max(1, int(n_post * prob))
                indices = jnp.asarray(_np.random.randint(0, n_post, (n_pre, n_conn), dtype=_np.int32))
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.ones((n_pre, n_conn), dtype=dtype)
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    matrix = jnp.asarray(_np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    matrix = jnp.asarray(_np.random.rand(b_rows, 10), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, indices, matrix),
                        {'shape': (n_pre, n_post), 'transpose': transpose}
                    )
                )
    return configs


def binary_fcnmm_p_call(
    weights: jax.Array,
    indices: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Perform a sparse matrix-matrix multiplication with fixed connection number.

    This function multiplies a sparse weight matrix against a dense matrix, where the
    sparse matrix is represented in a format with a fixed number of connections per row.
    Depending on the transpose flag, it handles either fixed pre-connections (transpose=True)
    or fixed post-connections (transpose=False).

    Args:
        weights: The weight values for the sparse connections. Can be either a JAX array
                 or a Quantity object. For homogeneous weights, this can be a scalar.
        indices: The indices array specifying the sparse matrix pattern. For transpose=True,
                 shape should be [n_pre, n_conn], otherwise [n_post, n_conn].
        matrix: The dense matrix to multiply with. Can be either a JAX array or a Quantity object.
        shape: A tuple of (n_pre, n_post) specifying the dimensions of the sparse weight matrix.
        transpose: If True, performs computation for fixed pre connections.
                  If False, performs computation for fixed post connections.

    Returns:
        A tuple containing a single element: the resulting matrix after multiplication,
        which will have the same type (JAX array or Quantity) as the inputs.

    Note:
        The transpose=True implementation uses an optimized kernel, while transpose=False
        uses a JAX-based implementation.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return binary_fcnmm_p(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=[out],
        backend=backend,
    )


binary_fcnmm_p = XLACustomKernel('binary_fcnmm')
binary_fcnmm_p.def_numba_kernel(_binary_fcnmm_numba_kernel)
binary_fcnmm_p.def_pallas_kernel('gpu', _binary_fcnmm_pallas_kernel)
binary_fcnmm_p.def_pallas_kernel('tpu', _binary_fcnmm_pallas_kernel)
binary_fcnmm_p.def_jvp_rule2(_binary_fcnmm_jvp_weights, None, _binary_fcnmm_jvp_matrix, None)
binary_fcnmm_p.def_transpose_rule(_binary_fcnmm_transpose_rule)
binary_fcnmm_p.def_batching_rule(_binary_fcnmm_batching)
binary_fcnmm_p.def_call(binary_fcnmm_p_call)
binary_fcnmm_p.def_tags('fcn', 'binary')
binary_fcnmm_p.def_benchmark_data(_binary_fcnmm_benchmark_data)
