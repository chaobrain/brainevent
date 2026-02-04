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

from typing import Union, Tuple, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, check_fixed_conn_num_shape, namescoped_jit
from brainevent._op import general_batching_rule, XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo


def _fixed_num_mv_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # fixed pre connection number
        if jnp.size(weight_info) == 1:
            @numba.njit(fastmath=True, cache=True)
            def ell_mv(weights, indices, vector, _, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(vector.shape[0]):
                    wv = w * vector[i]
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += wv
        else:
            @numba.njit(fastmath=True, cache=True)
            def ell_mv(weights, indices, vector, _, posts):
                posts[:] = 0.
                for i in range(vector.shape[0]):
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += weights[i, j] * vector[i]

    else:
        # fixed post connection number
        if jnp.size(weight_info) == 1:
            @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
            def ell_mv(weights, indices, vector, _, posts):
                w = weights[0]
                for i in numba.prange(indices.shape[0]):
                    posts[i] = w * np.sum(vector[indices[i]])
        else:
            @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
            def ell_mv(weights, indices, vector, _, posts):
                for i in numba.prange(indices.shape[0]):
                    posts[i] = np.sum(weights[i] * vector[indices[i]])

    def kernel(weights, indices, vector, _):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, vector, _)

    return kernel


def _fixed_num_mv_warp_kernel_generator(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    vector_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # Sparse Matrix: [k, m]
        # vector: [k]

        # fixed pre connection number
        if jnp.size(weight_info) == 1:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                vector: vector_warp_info,
                posts: out_warp_info
            ):
                i_k = warp.tid()
                w = weights[0]
                wv = w * vector[i_k]
                for j in range(indices.shape[1]):
                    warp.atomic_add(posts, indices[i_k, j], wv)
        else:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                vector: vector_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                v = vector[i]
                for j in range(indices.shape[1]):
                    warp.atomic_add(posts, indices[i, j], weights[i, j] * v)

        def kernel(weights, indices, vector, _):
            out_info = kwargs['outs'][0]
            dim = vector_info.shape[0]
            fn = jax_kernel(ell_mv, launch_dims=dim, num_outputs=0, in_out_argnames=['posts'])
            return fn(weights, indices, vector, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        # fixed post connection number
        # Sparse Matrix: [m, k]
        # vector: [k]

        if jnp.size(weight_info) == 1:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                vector: vector_warp_info,
                posts: out_warp_info
            ):
                i_m = warp.tid()
                w = weights[0]
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += vector[indices[i_m, j]]
                posts[i_m] = w * r
        else:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                vector: vector_warp_info,
                posts: out_warp_info
            ):
                i_m = warp.tid()
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += weights[i_m, j] * vector[indices[i_m, j]]
                posts[i_m] = r

        def kernel(weights, indices, vector, _):
            out_info = kwargs['outs'][0]
            dim = indices_info.shape[0]
            fn = jax_kernel(ell_mv, launch_dims=dim, num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, vector)

    return kernel


def _fixed_num_mv_pallas_kernel_generator(
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = jnp.size(weight_info) == 1
    block_dim = generate_block_dim(indices_info.shape[1], maximum=128)

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
            if homo:
                wv = vector * weight_ref[0]
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
                    data = jnp.where(mask, data * vector, 0.0)
                atomic_add(out_ref, ind, data, mask=mask)

            jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

        def kernel(weights, indices, vector, _):
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
                vec = jnp.where(mask, vec, 0.0)
                if homo:
                    return out + jnp.sum(vec)
                else:
                    weight = weight_ref[i_row, pl.dslice(i_col, block_dim)]
                    weight = jnp.where(mask, weight, 0.0)
                    return out + jnp.sum(weight * vec)

            i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
            if homo:
                i_row_sum = i_row_sum * weight_ref[0]
            out_ref[i_row] = i_row_sum

        def kernel(weights, indices, vector, _):
            fn = pl.pallas_call(_raw_kernel, grid=(n_pre,), out_shape=kwargs['outs'])
            return fn(weights, indices, vector, _)

    return kernel


def _fixed_num_mv_jvp_vector(spk_dot, weights, indices, spikes, _, *, shape, transpose, **kwargs):
    return fixed_num_mv_p_call(weights, indices, spk_dot, shape=shape, transpose=transpose)


def _fixed_num_mv_jvp_weights(w_dot, weights, indices, vector, _, *, shape, transpose, **kwargs):
    return fixed_num_mv_p_call(w_dot, indices, vector, shape=shape, transpose=transpose)


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


@namescoped_jit(static_argnames=("shape", "transpose"))
def _warp_fixed_num_mv_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
) -> Tuple[Union[jax.Array, u.Quantity]]:
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, vector, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    vector, v_unit = u.split_mantissa_unit(vector)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    r = fixed_num_mv_p(
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


def _fixed_num_mv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = fixed_num_mm_p_call(
            args[0],
            args[1],
            args[2].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = fixed_num_mm_p_call(
            args[0],
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]
    else:
        return general_batching_rule(fixed_num_mv_p, args, axes, **kwargs)


def fixed_num_mv_p_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
):
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
    return _warp_fixed_num_mv_call(
        weights,
        indices,
        vector,
        shape=shape,
        transpose=transpose
    )
    if transpose:
        pass
    else:
        return _jax_fixed_num_mv_call(
            weights,
            indices,
            vector,
            shape=shape,
            transpose=transpose
        )


fixed_num_mv_p = XLACustomKernel('fixed_num_mv')
fixed_num_mv_p.def_numba_kernel(_fixed_num_mv_numba_kernel_generator)
fixed_num_mv_p.def_warp_kernel(_fixed_num_mv_warp_kernel_generator)
fixed_num_mv_p.def_pallas_kernel('gpu', _fixed_num_mv_pallas_kernel_generator)
fixed_num_mv_p.def_pallas_kernel('tpu', _fixed_num_mv_pallas_kernel_generator)
fixed_num_mv_p.def_jvp_rule2(_fixed_num_mv_jvp_weights, None, _fixed_num_mv_jvp_vector, None)
fixed_num_mv_p.def_transpose_rule(_fixed_num_mv_transpose_rule)
fixed_num_mv_p.def_batching_rule(_fixed_num_mv_batching)
fixed_num_mv_p.def_call(fixed_num_mv_p_call)


def _fixed_num_mm_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
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

        if jnp.size(weight_info) == 1:
            @numba.njit(fastmath=True, cache=True)
            def ell_mv(weights, indices, matrix, _, posts):
                posts[:] = 0.
                w = weights[0]
                for i_k in range(matrix.shape[0]):
                    wv = w * matrix[i_k]
                    for i_conn in range(indices.shape[1]):
                        posts[indices[i_k, i_conn]] += wv
        else:
            @numba.njit(fastmath=True, cache=True)
            def ell_mv(weights, indices, vector, _, posts):
                posts[:] = 0.
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
            @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
            def ell_mv(weights, indices, matrix, _, posts):
                w = weights[0]
                for i_m in numba.prange(indices.shape[0]):
                    posts[i_m] = w * np.sum(matrix[indices[i_m]], axis=0)
        else:
            @numba.njit(parallel=True, fastmath=True, nogil=True, cache=True)
            def ell_mv(weights, indices, matrix, _, posts):
                for i_m in numba.prange(indices.shape[0]):
                    posts[i_m] = weights[i_m] @ matrix[indices[i_m]]

    def kernel(weights, indices, matrix, _):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, matrix, _)

    return kernel


def _fixed_num_mm_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    # Warp kernel for MM not yet implemented
    raise NotImplementedError


def _fixed_num_mm_pallas_kernel_generator(
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = jnp.size(weight_info) == 1
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


def _fixed_num_mm_jvp_matrix(matrix_dot, weights, indices, matrix, _, *, shape, transpose, **kwargs):
    return fixed_num_mm_p_call(weights, indices, matrix_dot, shape=shape, transpose=transpose)


def _fixed_num_mm_jvp_weights(weights_dot, weights, indices, matrix, _, *, shape, transpose, **kwargs):
    return fixed_num_mm_p_call(weights_dot, indices, matrix, shape=shape, transpose=transpose)


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
            ct_vector = fixed_num_mm_p_call(
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
            ct_weight = fixed_num_mm_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                matrix,
                shape=shape,
                transpose=transpose
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
        return ct_weight, indices, matrix, _


def _batching_base_fn(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = fixed_num_mm_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _fixed_num_mm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 1, None):
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 2, None):
        return _batching_base_fn(args, axis=2, **kwargs)

    else:
        return general_batching_rule(fixed_num_mm_p, args, axes, **kwargs)


@namescoped_jit(static_argnames=("shape", "transpose"))
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
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    r = fixed_num_mm_p(
        weights,
        indices,
        matrix,
        jnp.zeros(out.shape, out.dtype),
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=out,
    )
    return (u.maybe_decimal(r * m_unit * w_unit),)


fixed_num_mm_p = XLACustomKernel('fixed_num_mm')
fixed_num_mm_p.def_numba_kernel(_fixed_num_mm_numba_kernel_generator)
fixed_num_mm_p.def_pallas_kernel('gpu', _fixed_num_mm_pallas_kernel_generator)
fixed_num_mm_p.def_pallas_kernel('tpu', _fixed_num_mm_pallas_kernel_generator)
fixed_num_mm_p.def_jvp_rule2(_fixed_num_mm_jvp_weights, None, _fixed_num_mm_jvp_matrix, None)
fixed_num_mm_p.def_transpose_rule(_fixed_num_mm_transpose_rule)
fixed_num_mm_p.def_batching_rule(_fixed_num_mm_batching)
fixed_num_mm_p.def_call(fixed_num_mm_p_call)
