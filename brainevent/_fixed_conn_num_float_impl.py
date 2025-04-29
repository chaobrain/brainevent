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
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from ._compatible_import import pallas as pl
from ._fixed_conn_num_misc import generate_block_dim, check_shape
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_warp import dtype_to_warp_type


def _fixed_post_num_mv_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    if transpose:
        # fixed pre connection number
        if jnp.size(weight_info) == 1:
            @numba.njit(**numba_environ.setting)
            def ell_mv(weights, indices, vector, _, posts):
                w = weights[0]
                for i in range(vector.shape[0]):
                    wv = w * vector[i]
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += wv

        else:
            @numba.njit(**numba_environ.setting)
            def ell_mv(weights, indices, vector, _, posts):
                for i in range(vector.shape[0]):
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += weights[i, j] * vector[i]

    else:
        # fixed post connection number

        if jnp.size(weight_info) == 1:
            @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
            def ell_mv(weights, indices, vector, _, posts):
                w = weights[0]
                for i in range(indices.shape[0]):
                    posts[i] = w * np.sum(vector[indices[i]])

        else:
            @numba.njit(**numba_environ.setting, parallel=numba_environ.parallel)
            def ell_mv(weights, indices, vector, _, posts):
                for i in range(indices.shape[0]):
                    posts[i] = np.sum(weights[i] * vector[indices[i]])

    return ell_mv


def _fixed_post_num_mv_warp_kernel_generator(
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


def _fixed_post_num_mv_pallas_kernel_generator(
    block_dim: int,
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    assert transpose, "TPU backend does not support fixed post connection number"
    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = jnp.size(weight_info) == 1

    if homo:
        def _ell_mv_kernel_homo(
            vector_ref,  # [n_pre]
            index_ref,  # [n_pre, n_conn]
            _,
            out_ref,  # [n_post]
        ):
            i_row = pl.program_id(0)
            vector = vector_ref[i_row]

            def loop_fn(i, _):
                i = i * block_dim
                mask = i + jnp.arange(block_dim) < n_conn
                ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                data = jnp.ones(block_dim, dtype=weight_info.dtype) * vector
                pl.atomic_add(out_ref, ind, data, mask=mask)

            jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

        # homogenous weights
        def kernel(weight, indices, spikes, out):
            fn = pl.pallas_call(
                _ell_mv_kernel_homo,
                out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                grid=(n_pre,),
                input_output_aliases={2: 0},
            )
            return [fn(spikes, indices, out) * weight]

    else:
        def _ell_mv_kernel_heter(
            vector_ref,  # [n_pre]
            index_ref,  # [n_pre, n_conn]
            weight_ref,  # [n_pre, n_conn]
            _,
            out_ref,  # [n_post]
        ):
            i_row = pl.program_id(0)
            vector = vector_ref[i_row]

            def loop_fn(i, _):
                i = i * block_dim
                mask = i + jnp.arange(block_dim) < n_conn
                ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                weight = pl.load(weight_ref, (i_row, pl.dslice(i, block_dim)), mask=mask) * vector
                pl.atomic_add(out_ref, ind, weight, mask=mask)

            jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

        # heterogeneous weights
        def kernel(weight, indices, spikes, out):
            fn = pl.pallas_call(
                _ell_mv_kernel_heter,
                out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                grid=(n_pre,),
                input_output_aliases={3: 0},
            )
            return [fn(spikes, indices, weight, out)]

    return kernel


def _fixed_post_num_mv_jvp_vector(
    spk_dot,
    weights,
    indices,
    spikes,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return fixed_post_num_mv_p_call(
        weights,
        indices,
        spk_dot,
        shape=shape,
        transpose=transpose,
    )


def _fixed_post_num_mv_jvp_weights(
    w_dot,
    weights,
    indices,
    vector,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return fixed_post_num_mv_p_call(
        w_dot,
        indices,
        vector,
        shape=shape,
        transpose=transpose,
    )


def _fixed_post_num_mv_transpose_rule(
    ct,
    weights,
    indices,
    vector,
    _,
    *,
    shape,
    transpose,
    weight_info,
    **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # ∂L/∂spk = ∂L/∂y * ∂y/∂spk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(vector):
        if type(ct) is ad.Zero:
            ct_vector = ad.Zero(vector)
        else:
            ct_vector = fixed_post_num_mv_p_call(
                weights,
                indices,
                ct,
                shape=shape,
                transpose=not transpose
            )[0]
        return weights, indices, ct_vector, _
    else:
        # ∂L/∂w = ∂L/∂y * ∂y/∂w
        if type(ct) is ad.Zero:
            ct_weight = ad.Zero(weights)
        elif homo:
            ct_weight = fixed_post_num_mv_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                vector,
                shape=shape,
                transpose=transpose
            )[0]
            ct_weight = jnp.inner(ct, ct_weight).reshape(*weight_info.shape)

        else:
            if transpose:
                ct_weight = jax.vmap(lambda v, ind: v * ct[ind])(vector, indices)
            else:
                ct_weight = jax.vmap(lambda c, ind: c * vector[ind])(ct, indices)
        return ct_weight, indices, vector, _


def _warp_fixed_post_num_mv_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    assert transpose, "Warp backend does not support non-transpose mode."
    out, weights, n_pre, n_post = check_shape(weights, indices, vector, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    vector, v_unit = u.split_mantissa_unit(vector)

    r = fixed_post_num_mv_p(
        weights,
        indices,
        vector,
        jnp.zeros(out.shape, out.dtype),
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=out
    )
    return (u.maybe_decimal(r * v_unit * w_unit),)


def _jax_fixed_post_num_mv_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    assert not transpose, "JAX backend does not support transpose mode."
    out, weights, n_pre, n_post = check_shape(
        weights, indices, vector, shape, transpose,
        require_scalar_weight=True,
    )
    scalar_weight = weights.ndim == 0
    if scalar_weight:
        return jax.vmap(lambda ind: weights * u.math.sum(vector[ind]))(indices),
    else:
        return jax.vmap(lambda w, ind: u.math.sum(w * vector[ind]))(weights, indices),


def fixed_post_num_mv_p_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    if transpose:
        return _warp_fixed_post_num_mv_call(
            weights,
            indices,
            vector,
            shape=shape,
            transpose=transpose
        )
    else:
        return _jax_fixed_post_num_mv_call(
            weights,
            indices,
            vector,
            shape=shape,
            transpose=transpose
        )


fixed_post_num_mv_p = XLACustomKernel(
    'fixed_post_num_mv',
    cpu_kernel=NumbaKernelGenerator(
        _fixed_post_num_mv_numba_kernel_generator,
        input_output_aliases={3: 0}
    ),
    # gpu_kernel=WarpKernelGenerator(
    #     _fixed_post_num_mv_warp_kernel_generator,
    #     dim=lambda transpose, indices_info, vector_info, **kwargs: (
    #         vector_info.shape[0]
    #         if transpose else
    #         indices_info.shape[0]
    #     ),
    #     input_output_aliases={3: 0}
    # ),
    gpu_kernel=PallasKernelGenerator(
        _fixed_post_num_mv_pallas_kernel_generator,
        block_dim=generate_block_dim,
        input_output_aliases={3: 0}
    ),
    tpu_kernel=PallasKernelGenerator(
        _fixed_post_num_mv_pallas_kernel_generator,
        block_dim=generate_block_dim,
        input_output_aliases={3: 0}
    ),
)
fixed_post_num_mv_p.defjvp(
    _fixed_post_num_mv_jvp_weights,
    None,
    _fixed_post_num_mv_jvp_vector,
    None,
)
fixed_post_num_mv_p.def_transpose_rule(_fixed_post_num_mv_transpose_rule)
