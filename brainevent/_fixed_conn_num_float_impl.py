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
from ._config import numba_environ, config
from ._misc import generate_block_dim, check_fixed_conn_num_shape
from ._xla_custom_op import XLACustomKernel, KernelChoice
from ._xla_custom_op_numba import NumbaKernelGenerator
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator


def _fixed_num_mv_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    if transpose:
        # fixed pre connection number
        if jnp.size(weight_info) == 1:
            def ell_mv(weights, indices, vector, _, posts):
                w = weights[0]
                for i in range(vector.shape[0]):
                    wv = w * vector[i]
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += wv

        else:
            def ell_mv(weights, indices, vector, _, posts):
                for i in range(vector.shape[0]):
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += weights[i, j] * vector[i]

    else:
        # fixed post connection number

        if jnp.size(weight_info) == 1:
            def ell_mv(weights, indices, vector, _, posts):
                w = weights[0]
                for i in range(indices.shape[0]):
                    posts[i] = w * np.sum(vector[indices[i]])

        else:
            def ell_mv(weights, indices, vector, _, posts):
                for i in range(indices.shape[0]):
                    posts[i] = np.sum(weights[i] * vector[indices[i]])

    return numba_environ.jit_fn(ell_mv)


def _fixed_num_mv_warp_kernel_generator(
    WARP_TILE_SIZE: int,
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
        # Sparse Matrix: [k, m]
        # vector: [k]

        # fixed pre connection number
        if jnp.size(weight_info) == 1:
            def ell_mv(
                weights: warp.array2d(dtype=weight_dtype),
                indices: warp.array2d(dtype=indices_dtype),
                vector: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i_k = warp.tid()
                w = weights[0]
                wv = w * vector[i_k]
                # index = warp.tile_load(indices[i_k], WARP_TILE_SIZE)
                # warp.tile_atomic_add(posts, index, wv)
                for j in range(indices.shape[1]):
                    posts[indices[i_k, j]] += wv

        else:
            def ell_mv(
                weights: warp.array2d(dtype=weight_dtype),
                indices: warp.array2d(dtype=indices_dtype),
                vector: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i = warp.tid()
                v = vector[i]

                # index = warp.tile_load(indices[i_k], WARP_TILE_SIZE)
                # weight = warp.tile_load(weights[i_k], WARP_TILE_SIZE)
                # warp.tile_atomic_add(posts, index, weight * v)

                for j in range(indices.shape[1]):
                    posts[indices[i, j]] += weights[i, j] * v

    else:
        # fixed post connection number
        # Sparse Matrix: [m, k]
        # vector: [k]

        if jnp.size(weight_info) == 1:
            def ell_mv(
                weights: warp.array2d(dtype=weight_dtype),
                indices: warp.array2d(dtype=indices_dtype),
                vector: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i_m = warp.tid()
                w = weights[0]

                # index = warp.tile_load(indices[i_m], WARP_TILE_SIZE)
                # vec = warp.tile_load(vector, index)
                # posts[i_m] = w * warp.tile_sum(vec)

                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += vector[indices[i_m, j]]
                posts[i_m] = w * r

        else:
            def ell_mv(
                weights: warp.array2d(dtype=weight_dtype),
                indices: warp.array2d(dtype=indices_dtype),
                vector: warp.array1d(dtype=vector_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype)
            ):
                i_m = warp.tid()

                # index = warp.tile_load(indices[i_m], WARP_TILE_SIZE)
                # vec = warp.tile_load(vector, index)
                # wei = warp.tile_load(weights[i_m], WARP_TILE_SIZE)
                # posts[i_m] = warp.tile_sum(vec * wei)

                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += weights[i_m, j] * vector[indices[i_m, j]]
                posts[i_m] = r

    return warp.kernel(ell_mv)


def _fixed_num_mv_pallas_kernel_generator(
    block_dim: int,
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = jnp.size(weight_info) == 1

    if transpose:

        # Sparse Matrix: [k, m]
        # vector: [k]

        if homo:
            def _ell_mv_kernel_homo(
                weight_ref,  # [1]
                index_ref,  # [n_pre, n_conn]
                vector_ref,  # [n_pre]
                _,
                out_ref,  # [n_post]
            ):
                i_row = pl.program_id(0)
                wv = vector_ref[i_row] * weight_ref[0]

                def loop_fn(i, _):
                    i = i * block_dim
                    mask = i + jnp.arange(block_dim) < n_conn
                    ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                    data = jnp.ones(block_dim, dtype=weight_info.dtype) * wv
                    pl.atomic_add(out_ref, ind, data, mask=mask)

                jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

            # homogenous weights
            def kernel(weight, indices, spikes, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_homo,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={3: 0},
                )
                return [fn(weight, indices, spikes, out)]

        else:
            def _ell_mv_kernel_heter(
                weight_ref,  # [n_pre, n_conn]
                index_ref,  # [n_pre, n_conn]
                vector_ref,  # [n_pre]
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
            def kernel(weight, indices, vector, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_heter,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={3: 0},
                )
                return [fn(weight, indices, vector, out)]

    else:

        # Sparse Matrix: [m, k]
        # vector: [k]

        if homo:
            def _ell_mv_kernel_homo(
                weight_ref,  # [1]
                index_ref,  # [n_pre, n_conn]
                vector_ref,  # [n_post]
                _,
                out_ref,  # [n_pre]
            ):
                i_row = pl.program_id(0)
                weight = weight_ref[0]

                def loop_fn(i, sum_):
                    i = i * block_dim
                    mask = i + jnp.arange(block_dim) < n_conn
                    ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                    vec = pl.load(vector_ref, ind, mask=mask)
                    return sum_ + jnp.sum(vec)

                i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
                out_ref[i_row] = i_row_sum * weight

            # homogenous weights
            def kernel(weight, indices, vector, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_homo,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={3: 0},
                )
                return [fn(weight, indices, vector, out)]

        else:
            def _ell_mv_kernel_heter(
                weight_ref,  # [n_pre, n_conn]
                index_ref,  # [n_pre, n_conn]
                vector_ref,  # [n_post]
                _,
                out_ref,  # [n_pre]
            ):
                i_row = pl.program_id(0)

                def loop_fn(i, sum_):
                    i = i * block_dim
                    mask = i + jnp.arange(block_dim) < n_conn
                    ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                    weight = pl.load(weight_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                    vector = pl.load(vector_ref, ind, mask=mask)
                    return sum_ + jnp.sum(weight * vector)

                i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
                out_ref[i_row] = i_row_sum

            # heterogeneous weights
            def kernel(weight, indices, vector, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_heter,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={3: 0},
                )
                return [fn(weight, indices, vector, out)]

    return kernel


def _fixed_num_mv_jvp_vector(
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
    return fixed_num_mv_p_call(
        weights,
        indices,
        spk_dot,
        shape=shape,
        transpose=transpose,
    )


def _fixed_num_mv_jvp_weights(
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
    return fixed_num_mv_p_call(
        w_dot,
        indices,
        vector,
        shape=shape,
        transpose=transpose,
    )


def _fixed_num_mv_transpose_rule(
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
            ct_vector = fixed_num_mv_p_call(
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
            ct_weight = fixed_num_mv_p_call(
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


def _warp_fixed_num_mv_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    assert transpose, "Customized operator does not support non-transpose mode."
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, vector, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    vector, v_unit = u.split_mantissa_unit(vector)

    r = fixed_num_mv_p.call(
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


def _jax_fixed_num_mv_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    assert not transpose, "JAX backend does not support transpose mode."
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(
        weights, indices, vector, shape, transpose, require_scalar_weight=True,
    )
    scalar_weight = weights.ndim == 0
    if scalar_weight:
        return jax.vmap(lambda ind: weights * u.math.sum(vector[ind]))(indices),
    else:
        return jax.vmap(lambda w, ind: u.math.sum(w * vector[ind]))(weights, indices),


def fixed_num_mv_p_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    """Perform a sparse matrix-vector multiplication with fixed connection number.

    This function multiplies a sparse weight matrix against a dense vector, where the
    sparse matrix is represented in a format with a fixed number of connections per row.
    Depending on the transpose flag, it routes to either a GPU/TPU optimized implementation
    (transpose=True) or a JAX-based implementation (transpose=False).

    Args:
        weights: The weight values for the sparse connections. Can be either a JAX array
                 or a Quantity object. For homogeneous weights, this can be a scalar.
        indices: The indices array specifying the sparse matrix pattern. For transpose=True,
                 shape should be [n_pre, n_conn], otherwise [n_post, n_conn].
        vector: The dense vector to multiply with. Can be either a JAX array or a Quantity object.
        shape: A tuple of (n_pre, n_post) specifying the dimensions of the sparse weight matrix.
        transpose: If True, performs computation for fixed pre connections using optimized kernels.
                  If False, performs computation for fixed post connections using JAX implementation.

    Returns:
        A tuple containing a single element: the resulting vector after multiplication,
        which will have the same type (JAX array or Quantity) as the inputs.
    """
    if transpose:
        return _warp_fixed_num_mv_call(
            weights,
            indices,
            vector,
            shape=shape,
            transpose=transpose
        )
    else:
        return _jax_fixed_num_mv_call(
            weights,
            indices,
            vector,
            shape=shape,
            transpose=transpose
        )


def _mv_gpu_kernel():
    if config.gpu_kernel_use_warp:
        return WarpKernelGenerator(
            _fixed_num_mv_warp_kernel_generator,
            dim=lambda transpose, indices_info, vector_info, **kwargs: (
                vector_info.shape[0]
                if transpose else
                indices_info.shape[0]
            ),
            input_output_aliases={3: 0}
        )
    else:
        return PallasKernelGenerator(
            _fixed_num_mv_pallas_kernel_generator,
            block_dim=lambda indices_info, **kwargs: generate_block_dim(indices_info.shape[1]),
            input_output_aliases={3: 0}
        )


fixed_num_mv_p = XLACustomKernel('fixed_num_mv')
fixed_num_mv_p.def_cpu_kernel(NumbaKernelGenerator(_fixed_num_mv_numba_kernel_generator, input_output_aliases={3: 0}))
fixed_num_mv_p.def_gpu_kernel(KernelChoice(_mv_gpu_kernel))
fixed_num_mv_p.def_tpu_kernel(
    PallasKernelGenerator(
        _fixed_num_mv_pallas_kernel_generator,
        block_dim=lambda indices_info, **kwargs: generate_block_dim(indices_info.shape[1]),
        input_output_aliases={3: 0}
    )
)
fixed_num_mv_p.defjvp(_fixed_num_mv_jvp_weights, None, _fixed_num_mv_jvp_vector, None)
fixed_num_mv_p.def_transpose_rule(_fixed_num_mv_transpose_rule)


def _fixed_num_mm_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    if transpose:

        # fixed pre connection number
        #
        # CSR: [k, m]
        # matrix: [k, n]
        #

        if jnp.size(weight_info) == 1:
            def ell_mv(weights, indices, matrix, _, posts):
                w = weights[0]
                for i_k in range(matrix.shape[0]):
                    wv = w * matrix[i_k]
                    for i_conn in range(indices.shape[1]):
                        posts[indices[i_k, i_conn]] += wv

        else:
            def ell_mv(weights, indices, vector, _, posts):
                for i in range(vector.shape[0]):
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += weights[i, j] * vector[i]

    else:
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        if jnp.size(weight_info) == 1:
            def ell_mv(weights, indices, matrix, _, posts):
                w = weights[0]
                for i_m in range(indices.shape[0]):
                    posts[i_m] = w * np.sum(matrix[indices[i_m]], axis=0)

        else:
            def ell_mv(weights, indices, matrix, _, posts):
                for i_m in range(indices.shape[0]):
                    posts[i_m] = weights[i_m] @ matrix[indices[i_m]]

    return numba_environ.jit_fn(ell_mv)


def _fixed_num_mm_warp_kernel_generator(
    transpose: bool,
    WRAP_TILE_SIZE: int,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    vector_dtype = dtype_to_warp_type(vector_info.dtype)
    indices_dtype = dtype_to_warp_type(indices_info.dtype)

    raise NotImplementedError

    # fixed pre connection number

    # CSR: [k, m]
    # matrix: [k, n]
    #

    if jnp.size(weight_info) == 1:
        def ell_mv(
            weights: warp.array2d(dtype=weight_dtype),
            indices: warp.array2d(dtype=indices_dtype),
            matrix: warp.array2d(dtype=vector_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype)
        ):
            i = warp.tid()
            w = weights[0]
            wv = w * matrix[i]
            for j in range(indices.shape[1]):
                posts[indices[i, j]] += wv

    else:
        def ell_mv(
            weights: warp.array2d(dtype=weight_dtype),
            indices: warp.array2d(dtype=indices_dtype),
            matrix: warp.array2d(dtype=vector_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype)
        ):
            i = warp.tid()
            for j in range(indices.shape[1]):
                posts[indices[i, j]] += weights[i, j] * matrix[i]

    return warp.kernel(ell_mv)


def _fixed_num_mm_pallas_kernel_generator(
    block_dim: int,
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = jnp.size(weight_info) == 1

    raise NotImplementedError

    if transpose:

        #
        # fixed pre connection number
        #
        # CSR: [k, m]
        # matrix: [k, n]
        #

        if homo:
            def _ell_mv_kernel_homo(
                weight_ref,  # [1]
                index_ref,  # [n_pre, n_conn]
                matrix_ref,  # [n_pre, k]
                _,
                out_ref,  # [n_post, k]
            ):
                i_conn = pl.program_id(0)
                i_n = pl.program_id(1)
                vector = matrix_ref[i_conn]

                def loop_fn(i, _):
                    i = i * block_dim
                    mask = i + jnp.arange(block_dim) < n_conn
                    ind = pl.load(index_ref, (i_conn, pl.dslice(i, block_dim)), mask=mask)
                    data = jnp.ones(block_dim, dtype=weight_info.dtype) * vector
                    pl.atomic_add(out_ref, ind, data, mask=mask)

                jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

            # homogenous weights
            def kernel(weight, indices, vector, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_homo,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={2: 0},
                )
                return [fn(weight, indices, vector, out)]

        else:
            def _ell_mv_kernel_heter(
                weight_ref,  # [n_pre, n_conn]
                index_ref,  # [n_pre, n_conn]
                vector_ref,  # [n_pre, k]
                _,
                out_ref,  # [n_post, k]
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
            def kernel(weight, indices, vector, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_heter,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={3: 0},
                )
                return [fn(weight, indices, vector, out)]

    else:

        #
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        if homo:
            def _ell_mv_kernel_homo(
                weight_ref,  # [1]
                index_ref,  # [n_pre, n_conn]
                matrix_ref,  # [k, n]
                _,
                out_ref,  # [n_pre, n]
            ):
                i_m = pl.program_id(0)
                i_n = pl.program_id(1)
                vector = matrix_ref[i_m]

                def loop_fn(i, _):
                    i = i * block_dim
                    mask = i + jnp.arange(block_dim) < n_conn
                    ind = pl.load(index_ref, (i_m, pl.dslice(i, block_dim)), mask=mask)
                    data = jnp.ones(block_dim, dtype=weight_info.dtype) * vector
                    pl.atomic_add(out_ref, ind, data, mask=mask)

                jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

            # homogenous weights
            def kernel(weight, indices, vector, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_homo,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={2: 0},
                )
                return [fn(weight, indices, vector, out)]

        else:
            def _ell_mv_kernel_heter(
                weight_ref,  # [n_pre, n_conn]
                index_ref,  # [n_pre, n_conn]
                matrix_ref,  # [k, n]
                _,
                out_ref,  # [n_pre, n]
            ):
                i_row = pl.program_id(0)
                vector = matrix_ref[i_row]

                def loop_fn(i, _):
                    i = i * block_dim
                    mask = i + jnp.arange(block_dim) < n_conn
                    ind = pl.load(index_ref, (i_row, pl.dslice(i, block_dim)), mask=mask)
                    weight = pl.load(weight_ref, (i_row, pl.dslice(i, block_dim)), mask=mask) * vector
                    pl.atomic_add(out_ref, ind, weight, mask=mask)

                jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

            # heterogeneous weights
            def kernel(weight, indices, vector, out):
                fn = pl.pallas_call(
                    _ell_mv_kernel_heter,
                    out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
                    grid=(n_pre,),
                    input_output_aliases={3: 0},
                )
                return [fn(weight, indices, vector, out)]

    return kernel


def _fixed_num_mm_jvp_matrix(
    matrix_dot,
    weights,
    indices,
    matrix,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return fixed_num_mm_p_call(
        weights,
        indices,
        matrix_dot,
        shape=shape,
        transpose=transpose,
    )


def _fixed_num_mm_jvp_weights(
    weights_dot,
    weights,
    indices,
    matrix,
    _,
    *,
    shape,
    transpose,
    **kwargs
):
    return fixed_num_mm_p_call(
        weights_dot,
        indices,
        matrix,
        shape=shape,
        transpose=transpose,
    )


def _fixed_num_mm_transpose_rule(
    ct,
    weights,
    indices,
    matrix,
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
    if ad.is_undefined_primal(matrix):
        if type(ct) is ad.Zero:
            ct_vector = ad.Zero(matrix)
        else:
            ct_vector = fixed_num_mv_p_call(
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
            ct_weight = fixed_num_mv_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                matrix,
                shape=shape,
                transpose=transpose
            )[0]
            ct_weight = jnp.inner(ct, ct_weight).reshape(*weight_info.shape)

        else:
            if transpose:
                ct_weight = jax.vmap(lambda v, ind: v * ct[ind])(matrix, indices)
            else:
                ct_weight = jax.vmap(lambda c, ind: c * matrix[ind])(ct, indices)
        return ct_weight, indices, matrix, _


def _warp_fixed_num_mm_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    assert transpose, "Customized operator does not support non-transpose mode."
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)

    r = fixed_num_mm_p.call(
        weights,
        indices,
        matrix,
        jnp.zeros(out.shape, out.dtype),
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=out,
        WRAP_TILE_SIZE=matrix.shape[1],
    )
    return (u.maybe_decimal(r * m_unit * w_unit),)


def _jax_fixed_num_mm_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    assert not transpose, "JAX backend does not support transpose mode."
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(
        weights, indices, matrix, shape, transpose, require_scalar_weight=True,
    )
    scalar_weight = weights.ndim == 0
    if scalar_weight:
        return jax.vmap(lambda ind: weights * u.math.sum(matrix[ind]))(indices),
    else:
        return jax.vmap(lambda w, ind: u.math.sum(w * matrix[ind]))(weights, indices),


def fixed_num_mm_p_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
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
    if transpose:
        return _warp_fixed_num_mm_call(
            weights,
            indices,
            matrix,
            shape=shape,
            transpose=transpose
        )
    else:
        return _jax_fixed_num_mm_call(
            weights,
            indices,
            matrix,
            shape=shape,
            transpose=transpose
        )


def _mm_gpu_kernel():
    if config.gpu_kernel_use_warp:
        return WarpKernelGenerator(
            _fixed_num_mv_warp_kernel_generator,
            dim=lambda transpose, indices_info, vector_info, **kwargs: (
                vector_info.shape[0]
                if transpose else
                indices_info.shape[0]
            ),
            input_output_aliases={3: 0}
        )
    else:
        return PallasKernelGenerator(
            _fixed_num_mm_pallas_kernel_generator,
            block_dim=lambda indices_info, **kwargs: generate_block_dim(indices_info.shape[1]),
            input_output_aliases={3: 0}
        )


fixed_num_mm_p = XLACustomKernel('fixed_num_mm')
fixed_num_mm_p.def_cpu_kernel(
    NumbaKernelGenerator(
        _fixed_num_mm_numba_kernel_generator,
        input_output_aliases={3: 0}
    )
)
fixed_num_mm_p.def_gpu_kernel(KernelChoice(_mm_gpu_kernel))
fixed_num_mm_p.def_tpu_kernel(
    PallasKernelGenerator(
        _fixed_num_mm_pallas_kernel_generator,
        block_dim=lambda indices_info, **kwargs: generate_block_dim(indices_info.shape[1]),
        input_output_aliases={3: 0}
    )
)
fixed_num_mm_p.defjvp(_fixed_num_mm_jvp_weights, None, _fixed_num_mm_jvp_matrix, None)
fixed_num_mm_p.def_transpose_rule(_fixed_num_mm_transpose_rule)
