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

from typing import Union, Tuple, Sequence

import brainunit as u
import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_warp import WarpKernelGenerator, dtype_to_warp_type


def fixed_post_num_mv_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    if transpose:
        # fixed pre connection number
        if jnp.size(weight_info) == 1:
            @numba.njit(**numba_environ.numba_setting)
            def ell_mv(weights, indices, vector, _, posts):
                w = weights[0]
                for i in range(vector.shape[0]):
                    wv = w * vector[i]
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += wv

        else:
            @numba.njit(**numba_environ.numba_setting)
            def ell_mv(weights, indices, vector, _, posts):
                for i in range(vector.shape[0]):
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += weights[i, j] * vector[i]

    else:
        # fixed post connection number

        if jnp.size(weight_info) == 1:
            @numba.njit(**numba_environ.numba_setting)
            def ell_mv(weights, indices, vector, _, posts):
                w = weights[0]
                for i in range(indices.shape[0]):
                    posts[i] = w * np.sum(vector[indices[i]])

        else:
            @numba.njit(**numba_environ.numba_setting)
            def ell_mv(weights, indices, vector, _, posts):
                for i in range(indices.shape[0]):
                    posts[i] = np.sum(weights[i] * vector[indices[i]])

    return ell_mv


def fixed_post_num_mv_warp_kernel_generator(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    vector_dtype = dtype_to_warp_type(vector_info.dtype)
    indices_dtype = dtype_to_warp_type(indices_info.dtype)

    if transpose:
        # fixed pre connection number
        if jnp.size(weight_info) == 1:
            @warp.kernel
            def ell_mv(
                weights: warp.array2d(dtype=weight_dtype),
                indices: warp.array2d(dtype=indices_dtype),
                vector: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                wv = w * vector[i]
                for j in range(indices.shape[1]):
                    posts[indices[i, j]] += wv

        else:
            @warp.kernel
            def ell_mv(
                weights: warp.array2d(dtype=weight_dtype),
                indices: warp.array2d(dtype=indices_dtype),
                vector: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                for j in range(indices.shape[1]):
                    posts[indices[i, j]] += weights[i, j] * vector[i]

    else:
        # fixed post connection number

        if jnp.size(weight_info) == 1:
            @warp.kernel
            def ell_mv(
                weights: warp.array2d(dtype=weight_dtype),
                indices: warp.array2d(dtype=indices_dtype),
                vector: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                w = weights[0]
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += vector[indices[i, j]]
                posts[i] = w * r

        else:
            @warp.kernel
            def ell_mv(
                weights: warp.array2d(dtype=weight_dtype),
                indices: warp.array2d(dtype=indices_dtype),
                vector: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += weights[i, j] * vector[indices[i, j]]
                posts[i] = r

    return ell_mv


def fixed_post_num_mv_pallas_kernel_generator(
    block_size: int,
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    if transpose:
        n_pre, n_post = shape
    else:
        n_post, n_pre = shape
    n_conn = indices_info.shape[1]

    homo = jnp.size(weight_info) == 1

    if transpose:
        if homo:
            def _kernel(ind_ref, vec_ref, _, out_ref):
                # 每个block 处理 [block_size] 大小的vector
                # 每个block 处理 [block_size, block_size] 大小的indices 和 weights

                # -------------------------------
                # vec_ref: [block_size]
                # ind_ref: [block_size, block_size]
                # out_ref: [n_post]

                r_pid = pl.program_id(0)
                c_start = pl.program_id(1) * block_size
                mask = jnp.arange(block_size) + c_start
                row_length = jnp.minimum(n_pre - r_pid * block_size, block_size)

                def body_fn(j, _):
                    y = vec_ref[j] * jnp.ones(block_size, dtype=weight_info.dtype)
                    ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                    pl.atomic_add(out_ref, ind, y, mask=mask)

                jax.lax.fori_loop(0, row_length, body_fn, None)

            # heterogeneous weights
            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
                ],
                in_specs=[
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # ind_ref
                    pl.BlockSpec((block_size,), lambda i, j: i),  # vec_ref
                    pl.BlockSpec((n_post,), lambda i, j: 0),  # out_ref
                ],
                grid=(
                    pl.cdiv(n_pre, block_size),
                    pl.cdiv(n_conn, block_size),
                ),
                input_output_aliases={2: 0},
                interpret=False
            )
            return lambda weight, indices, vector, _: kernel(vector, indices, _) * weight

        else:
            def _kernel(w_ref, ind_ref, vec_ref, _, out_ref):
                # 每个block 处理 [block_size] 大小的vector
                # 每个block 处理 [block_size, n_conn] 大小的indices 和 weights

                # -------------------------------
                # vec_ref: [block_size]
                # ind_ref: [block_size, block_size]
                # w_ref: [block_size, block_size]
                # out_ref: [n_post]

                r_pid = pl.program_id(0)
                c_start = pl.program_id(1) * block_size
                mask = jnp.arange(block_size) + c_start
                row_length = jnp.minimum(n_pre - r_pid * block_size, block_size)

                def body_fn(j, _):
                    w = pl.load(w_ref, (j, pl.dslice(None)), mask=mask)
                    y = w * vec_ref[j]
                    ind = pl.load(ind_ref, (j, pl.dslice(None)), mask=mask)
                    pl.atomic_add(out_ref, ind, y, mask=mask)

                jax.lax.fori_loop(0, row_length, body_fn, None)

            # heterogeneous weights
            kernel = pl.pallas_call(
                _kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((n_post,), weight_info.dtype),
                ],
                in_specs=[
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # w_ref
                    pl.BlockSpec((block_size, block_size), lambda i, j: (i, j)),  # ind_ref
                    pl.BlockSpec((block_size,), lambda i, j: i),  # vec_ref
                    pl.BlockSpec((n_post,), lambda i: 0)  # out_ref
                ],
                grid=(
                    pl.cdiv(n_pre, block_size),
                    pl.cdiv(n_conn, block_size),
                ),
                input_output_aliases={3: 0},
                interpret=False
            )
            return lambda weight, indices, vector, _: kernel(vector, indices, weight, _)

    else:
        raise NotImplementedError


def fixed_post_num_mv_jvp_spikes(
    spk_dot,
    weights,
    indices,
    spikes,
    *,
    n_post,
    block_size,
    **kwargs
):
    return fixed_post_num_mv_p_call(
        spk_dot,
        weights,
        indices,
        n_post=n_post,
        block_size=block_size,
    )


def fixed_post_num_mv_jvp_weights(
    w_dot,
    weights,
    indices,
    vector,
    *,
    block_size, n_post, **kwargs
):
    return fixed_post_num_mv_p_call(
        vector,
        w_dot,
        indices,
        block_size=block_size,
        n_post=n_post,
    )


def fixed_post_num_mv_transpose_rule(
    ct, weights, indices, vector,
    *,
    n_post, block_size, weight_info, **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(vector):
        if homo:
            # homogeneous weight
            ct_spk = jax.vmap(lambda idx: jnp.sum(ct[idx] * weights))(indices)
        else:
            # heterogeneous weight
            ct_spk = jax.vmap(lambda idx, w: jnp.inner(ct[idx], w))(indices, weights)
        return (ad.Zero(vector) if type(ct) is ad.Zero else ct_spk), weights, indices

    else:
        # ∂L/∂w = ∂L/∂y * ∂y/∂w
        if homo:
            # scalar
            ct_gmax = fixed_post_num_mv_p_call(
                vector,
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                block_size=block_size,
                n_post=n_post,
            )
            ct_gmax = jnp.inner(ct, ct_gmax[0])
        else:
            ct_gmax = jax.vmap(lambda vec, one_ind: ct[one_ind] * vec)(vector, indices)
        return vector, (ad.Zero(weights) if type(ct) is ad.Zero else ct_gmax), indices


def fixed_post_num_mv_p_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    block_size: int = None
):
    assert weights.ndim == 2, 'weight dim should be 2'
    assert weights.shape[0] == shape[0], f'Pre size mismatch, got {weights.shape[0]} != {shape[0]}'
    n_pre, n_post = shape
    if transpose:
        out = jax.ShapeDtypeStruct([n_post], weights.dtype)
        assert vector.shape[0] == n_pre, f'When transpose, vector shape should be {n_pre}, got {vector.shape[0]}'
    else:
        out = jax.ShapeDtypeStruct([n_pre], weights.dtype)
        assert vector.shape[0] == n_post, f'When not transpose, vector shape should be {n_post}, got {vector.shape[0]}'

    n_conn = indices.shape[1]
    if block_size is None:
        # which is used for TPU/GPU kernel written in JAX pallas
        if n_conn <= 32:
            block_size = 32
        elif n_conn <= 64:
            block_size = 64
        elif n_conn <= 128:
            block_size = 128
        elif n_conn <= 256:
            block_size = 256
        else:
            block_size = 128

    weights, w_unit = u.split_mantissa_unit(weights)
    vector, v_unit = u.split_mantissa_unit(vector)

    r = fixed_post_num_mv_p(
        weights,
        indices,
        vector,
        jnp.zeros(out.shape, out.dtype),
        transpose=transpose,
        shape=shape,
        block_size=block_size,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=out
    )
    return [u.maybe_decimal(r * v_unit * w_unit)]


fixed_post_num_mv_p = XLACustomKernel(
    'fixed_post_num_mv',
    cpu_kernel=NumbaKernelGenerator(fixed_post_num_mv_numba_kernel_generator, input_output_aliases={2: 0}),
    gpu_kernel=WarpKernelGenerator(
        fixed_post_num_mv_pallas_kernel_generator,
        dim=lambda transpose, indices_info, vecto_infor, **kwargs: (
            vecto_infor.shape[0] if transpose else indices_info.shape[0]
        ),
        input_output_aliases={2: 0}
    ),
    tpu_kernel=PallasKernelGenerator(fixed_post_num_mv_pallas_kernel_generator),
)
fixed_post_num_mv_p.defjvp(fixed_post_num_mv_jvp_weights, None, fixed_post_num_mv_jvp_spikes)
fixed_post_num_mv_p.def_transpose_rule(fixed_post_num_mv_transpose_rule)
