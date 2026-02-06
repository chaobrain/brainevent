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


from typing import Sequence, Optional

import brainunit as u
import jax
import jax.numpy as jnp
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, namescope
from brainevent._op import jaxinfo_to_warpinfo, numba_kernel, XLACustomKernel, general_batching_rule
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Row, Col, MatrixShape
from .float import coomv, coomm

__all__ = [
    'binary_coomv',
    'binary_coomv_p',
    'binary_coomm',
    'binary_coomm_p',
]


def binary_coomv(
    data: Data,
    row: Row,
    col: Col,
    v: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    float_as_event: bool = True,
    backend: Optional[str] = None,
) -> Data:
    """
    Perform a COO sparse matrix-vector multiplication with event-based inputs.

    This function multiplies a sparse matrix in COO format by a dense vector,
    where the vector is treated as containing binary events (spikes).

    Args:
        data: The non-zero values of the sparse matrix.
        row: The row indices of the non-zero values.
        col: The column indices of the non-zero values.
        v: The dense vector to multiply (treated as events).
        shape: The shape of the sparse matrix (rows, cols).
        transpose: If True, multiply by the transposed matrix.
        float_as_event: If True, treat non-zero floats as binary events.
        backend: Optional backend to use ('numba', 'warp', 'pallas', etc.).

    Returns:
        The result of the matrix-vector multiplication.
    """
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = binary_coomv_p_call(
        data, row, col, v,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


@namescope(static_argnames=("shape", "transpose", "float_as_event"))
def binary_coomm(
    data: Data,
    row: Row,
    col: Col,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    float_as_event: bool = True,
    backend: Optional[str] = None,
) -> Data:
    """
    Perform a COO sparse matrix-matrix multiplication with event-based inputs.

    This function multiplies a sparse matrix in COO format by a dense matrix,
    where the matrix B is treated as containing binary events (spikes).

    Args:
        data: The non-zero values of the sparse matrix.
        row: The row indices of the non-zero values.
        col: The column indices of the non-zero values.
        B: The dense matrix to multiply (treated as events).
        shape: The shape of the sparse matrix (rows, cols).
        transpose: If True, multiply by the transposed matrix.
        float_as_event: If True, treat non-zero floats as binary events.
        backend: Optional backend to use ('numba', 'warp', 'pallas', etc.).

    Returns:
        The result of the matrix-matrix multiplication.
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = binary_coomm_p_call(
        data, row, col, B,
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


# =============================================================================
# COO Matrix-Vector Multiplication (coomv)
# =============================================================================


def _coomv_numba_kernel(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    match (transpose, weight_info.size, vector_info.dtype, float_as_event):

        # transpose=True, homogeneous
        case (True, 1, jnp.bool_, _):
            # bool
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    if v[row[i]]:
                        posts[col[i]] += w

        case (True, 1, _, True):
            # float_as_event
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    if v[row[i]] > 0.:
                        posts[col[i]] += w

        case (True, 1, _, False):
            # other
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    if v[row[i]] > 0.:
                        posts[col[i]] += w * v[row[i]]

        # transpose=True, heterogeneous
        case (True, _, jnp.bool_, _):
            # bool
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    if v[row[i]]:
                        posts[col[i]] += weights[i]

        case (True, _, _, True):
            # float_as_event
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    if v[row[i]] > 0.:
                        posts[col[i]] += weights[i]

        case (True, _, _, False):
            # other
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    if v[row[i]] > 0.:
                        posts[col[i]] += weights[i] * v[row[i]]

        # transpose=False, homogeneous
        case (False, 1, jnp.bool_, _):
            # bool
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    if v[col[i]]:
                        posts[row[i]] += w

        case (False, 1, _, True):
            # float_as_event
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    if v[col[i]] > 0.:
                        posts[row[i]] += w

        case (False, 1, _, False):
            # other
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    if v[col[i]] > 0.:
                        posts[row[i]] += w * v[col[i]]

        # transpose=False, heterogeneous
        case (False, _, jnp.bool_, _):
            # bool
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    if v[col[i]]:
                        posts[row[i]] += weights[i]

        case (False, _, _, True):
            # float_as_event
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    if v[col[i]] > 0.:
                        posts[row[i]] += weights[i]

        case (False, _, _, False):
            # other
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    if v[col[i]] > 0.:
                        posts[row[i]] += weights[i] * v[col[i]]

    def kernel(weights, row, col, v):
        return numba_kernel(mv, outs=kwargs['outs'])(weights, row, col, v)

    return kernel


def _coomv_warp_kernel(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    row_warp_info = jaxinfo_to_warpinfo(row_info)
    col_warp_info = jaxinfo_to_warpinfo(col_info)
    spike_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    match (transpose, weight_info.size, vector_info.dtype, float_as_event):

        # transpose=True, homogeneous
        case (True, 1, jnp.bool_, _):
            # bool
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                w = weights[0]
                if v[row[i]]:
                    posts[col[i]] += w

        case (True, 1, _, True):
            # float_as_event
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                w = weights[0]
                if v[row[i]] > 0.:
                    posts[col[i]] += w

        case (True, 1, _, False):
            # other
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                w = weights[0]
                if v[row[i]] > 0.:
                    posts[col[i]] += w * v[row[i]]

        # transpose=True, heterogeneous
        case (True, _, jnp.bool_, _):
            # bool
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                if v[row[i]]:
                    posts[col[i]] += weights[i]

        case (True, _, _, True):
            # float_as_event
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                if v[row[i]] > 0.:
                    posts[col[i]] += weights[i]

        case (True, _, _, False):
            # other
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                if v[row[i]] > 0.:
                    posts[col[i]] += weights[i] * v[row[i]]

        # transpose=False, homogeneous
        case (False, 1, jnp.bool_, _):
            # bool
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                w = weights[0]
                if v[col[i]]:
                    posts[row[i]] += w

        case (False, 1, _, True):
            # float_as_event
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                w = weights[0]
                if v[col[i]] > 0.:
                    posts[row[i]] += w

        case (False, 1, _, False):
            # other
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                w = weights[0]
                if v[col[i]] > 0.:
                    posts[row[i]] += w * v[col[i]]

        # transpose=False, heterogeneous
        case (False, _, jnp.bool_, _):
            # bool
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                if v[col[i]]:
                    posts[row[i]] += weights[i]

        case (False, _, _, True):
            # float_as_event
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                if v[col[i]] > 0.:
                    posts[row[i]] += weights[i]

        case (False, _, _, False):
            # other
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                v: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                if v[col[i]] > 0.:
                    posts[row[i]] += weights[i] * v[col[i]]

    dim = row_info.shape[0]
    out_info = kwargs['outs'][0]

    def kernel(weights, row, col, v):
        fn = jax_kernel(mv, launch_dims=dim, num_outputs=1, in_out_argnames=['posts'])
        return fn(weights, row, col, v, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _coomv_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    nnz = row_info.shape[0]
    block_dim = generate_block_dim(nnz)
    block_dim = 32 if block_dim < 32 else block_dim

    if transpose:
        if weight_info.size == 1:
            # coo.T @ v (homogeneous weights)
            #
            # coo: [m, k]
            # v: [m]
            # result: [k]
            #
            def mv(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                vector_ref,  # [m]
                _,  # [k]
                posts_ref,  # [k]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]
                events = vector_ref[rows]

                if vector_ref.dtype == jnp.bool_:
                    event_mask = mask & events
                else:
                    event_mask = mask & (events > 0.)

                data = jnp.ones((block_dim,), dtype=posts_ref.dtype) * data_ref[0]
                atomic_add(posts_ref, cols, data, mask=event_mask)

        else:
            # coo.T @ v (heterogeneous weights)
            #
            # coo: [m, k]
            # v: [m]
            # result: [k]
            #
            def mv(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                vector_ref,  # [m]
                _,  # [k]
                posts_ref,  # [k]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]
                weights = data_ref[pl.dslice(i_start, block_dim)]
                events = vector_ref[rows]

                if vector_ref.dtype == jnp.bool_:
                    event_mask = mask & events
                else:
                    event_mask = mask & (events > 0.)

                data = jnp.asarray(weights, dtype=posts_ref.dtype)
                atomic_add(posts_ref, cols, data, mask=event_mask)

        def kernel(data, row, col, vector):
            fn = pl.pallas_call(
                mv,
                grid=(pl.cdiv(nnz, block_dim),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs']
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, vector, posts)

        return kernel

    else:
        # coo @ v (non-transpose)
        # Also needs atomic operations due to COO race conditions
        if weight_info.size == 1:
            # coo @ v (homogeneous weights)
            #
            # coo: [m, k]
            # v: [k]
            # result: [m]
            #
            def mv(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                vector_ref,  # [k]
                _,  # [m]
                posts_ref,  # [m]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]
                events = vector_ref[cols]

                if vector_ref.dtype == jnp.bool_:
                    event_mask = mask & events
                else:
                    event_mask = mask & (events > 0.)

                data = jnp.ones((block_dim,), dtype=posts_ref.dtype) * data_ref[0]
                atomic_add(posts_ref, rows, data, mask=event_mask)

        else:
            # coo @ v (heterogeneous weights)
            #
            # coo: [m, k]
            # v: [k]
            # result: [m]
            #
            def mv(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                vector_ref,  # [k]
                _,  # [m]
                posts_ref,  # [m]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]
                weights = data_ref[pl.dslice(i_start, block_dim)]
                events = vector_ref[cols]

                if vector_ref.dtype == jnp.bool_:
                    event_mask = mask & events
                else:
                    event_mask = mask & (events > 0.)

                data = jnp.asarray(weights, dtype=posts_ref.dtype)
                atomic_add(posts_ref, rows, data, mask=event_mask)

        def kernel(data, row, col, vector):
            fn = pl.pallas_call(
                mv,
                grid=(pl.cdiv(nnz, block_dim),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs']
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, vector, posts)

        return kernel


def _coomv_jvp_vector(v_dot, data, row, col, v, *, shape, transpose, **kwargs):
    return [coomv(data, row, col, v_dot, shape=shape, transpose=transpose)]


def _coomv_jvp_weights(data_dot, data, row, col, v, *, shape, transpose, float_as_event, **kwargs):
    return binary_coomv_p_call(data_dot, row, col, v, shape=shape, transpose=transpose, float_as_event=float_as_event)


def _coomv_transpose_rule(
    ct,
    data,
    row,
    col,
    events,
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
            ct_events = coomv(
                data,
                row,
                col,
                ct,
                shape=shape,
                transpose=not transpose
            )
        return data, row, col, ct_events
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = binary_coomv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    row,
                    col,
                    events,
                    shape=shape,
                    transpose=transpose,
                    float_as_event=float_as_event
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:
                ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
        return ct_values, row, col, events


def _coomv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            float_as_event=kwargs['float_as_event'],
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_coomm_p_call(
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
        return general_batching_rule(binary_coomv_p_call, args, axes, **kwargs)


def binary_coomv_p_call(
    weights,
    row,
    col,
    v,
    *,
    shape: Sequence[int],
    transpose: bool,
    float_as_event: bool = True,
    backend: Optional[str] = None,
):
    """
    Perform a custom sparse matrix-vector multiplication operation.

    This function takes a sparse matrix represented in COO (Coordinate) format
    and a dense vector, then performs a matrix-vector multiplication.

    Parameters
    ----------
    weights : jax.Array
        The non-zero values of the sparse matrix.
    row : jax.Array
        The row indices of the non-zero values in the sparse matrix.
    col : jax.Array
        The column indices of the non-zero values in the sparse matrix.
    v : jax.Array
        The dense vector to multiply with the sparse matrix.
    shape : Sequence[int]
        The shape of the sparse matrix.
    transpose : bool
        Whether to transpose the sparse matrix before multiplication.
    float_as_event : bool
        Treat floating-point values as events.
    backend : str, optional
        Backend to use for computation.

    Returns
    -------
    jax.Array
        The result of the sparse matrix-vector multiplication.
    """
    # Convert scalar weights to a single-element array
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    # Determine the output shape based on whether the sparse matrix is transposed
    out_info = (
        # If transposed, the output shape is [shape[1]]
        jax.ShapeDtypeStruct([shape[1]], weights.dtype)
        if transpose else
        # If not transposed, the output shape is [shape[0]]
        jax.ShapeDtypeStruct([shape[0]], weights.dtype)
    )

    # Call the custom kernel with the provided arguments and output information
    return binary_coomv_p(
        weights,
        row,
        col,
        v,
        outs=[out_info],
        float_as_event=float_as_event,
        shape=shape,
        transpose=transpose,
        backend=backend,
        # Provide shape and dtype information for row indices
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        # Provide shape and dtype information for column indices
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
        # Provide shape and dtype information for non-zero values
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and dtype information for the dense vector
        vector_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
    )


binary_coomv_p = XLACustomKernel('binary_coomv')
binary_coomv_p.def_numba_kernel(_coomv_numba_kernel)
binary_coomv_p.def_warp_kernel(_coomv_warp_kernel)
binary_coomv_p.def_pallas_kernel('gpu', _coomv_pallas_gpu_kernel)
binary_coomv_p.def_pallas_kernel('tpu', _coomv_pallas_gpu_kernel)
binary_coomv_p.def_jvp_rule2(_coomv_jvp_weights, None, None, _coomv_jvp_vector)
binary_coomv_p.def_transpose_rule(_coomv_transpose_rule)
binary_coomv_p.def_batching_rule(_coomv_batching)
binary_coomv_p.def_call(binary_coomv_p_call)


# =============================================================================
# COO Matrix-Matrix Multiplication (coomm)
# =============================================================================


def _coomm_numba_kernel(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    match (transpose, weight_info.size, matrix_info.dtype, float_as_event):

        # transpose=True, homogeneous
        case (True, 1, jnp.bool_, _):
            # bool
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[row[i], j]:
                            posts[col[i], j] += w

        case (True, 1, _, True):
            # float_as_event
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[row[i], j] > 0.:
                            posts[col[i], j] += w

        case (True, 1, _, False):
            # other
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[row[i], j] > 0.:
                            posts[col[i], j] += w * B[row[i], j]

        # transpose=True, heterogeneous
        case (True, _, jnp.bool_, _):
            # bool
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[row[i], j]:
                            posts[col[i], j] += weights[i]

        case (True, _, _, True):
            # float_as_event
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[row[i], j] > 0.:
                            posts[col[i], j] += weights[i]

        case (True, _, _, False):
            # other
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[row[i], j] > 0.:
                            posts[col[i], j] += weights[i] * B[row[i], j]

        # transpose=False, homogeneous
        case (False, 1, jnp.bool_, _):
            # bool
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[col[i], j]:
                            posts[row[i], j] += w

        case (False, 1, _, True):
            # float_as_event
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[col[i], j] > 0.:
                            posts[row[i], j] += w

        case (False, 1, _, False):
            # other
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[col[i], j] > 0.:
                            posts[row[i], j] += w * B[col[i], j]

        # transpose=False, heterogeneous
        case (False, _, jnp.bool_, _):
            # bool
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[col[i], j]:
                            posts[row[i], j] += weights[i]

        case (False, _, _, True):
            # float_as_event
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[col[i], j] > 0.:
                            posts[row[i], j] += weights[i]

        case (False, _, _, False):
            # other
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    for j in range(B.shape[1]):
                        if B[col[i], j] > 0.:
                            posts[row[i], j] += weights[i] * B[col[i], j]

    def kernel(weights, row, col, B):
        return numba_kernel(mm, outs=kwargs['outs'])(weights, row, col, B)

    return kernel


def _coomm_warp_kernel(
    float_as_event: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    row_warp_info = jaxinfo_to_warpinfo(row_info)
    col_warp_info = jaxinfo_to_warpinfo(col_info)
    spike_warp_info = jaxinfo_to_warpinfo(matrix_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    match (transpose, weight_info.size, matrix_info.dtype, float_as_event):

        # transpose=True, homogeneous
        case (True, 1, jnp.bool_, _):
            # bool
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[row[i], j]:
                    posts[col[i], j] += w

        case (True, 1, _, True):
            # float_as_event
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[row[i], j] > 0.:
                    posts[col[i], j] += w

        case (True, 1, _, False):
            # other
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[row[i], j] > 0.:
                    posts[col[i], j] += w * B[row[i], j]

        # transpose=True, heterogeneous
        case (True, _, jnp.bool_, _):
            # bool
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                if B[row[i], j]:
                    posts[col[i], j] += weights[i]

        case (True, _, _, True):
            # float_as_event
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                if B[row[i], j] > 0.:
                    posts[col[i], j] += weights[i]

        case (True, _, _, False):
            # other
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                if B[row[i], j] > 0.:
                    posts[col[i], j] += weights[i] * B[row[i], j]

        # transpose=False, homogeneous
        case (False, 1, jnp.bool_, _):
            # bool
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[col[i], j]:
                    posts[row[i], j] += w

        case (False, 1, _, True):
            # float_as_event
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[col[i], j] > 0.:
                    posts[row[i], j] += w

        case (False, 1, _, False):
            # other
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                w = weights[0]
                if B[col[i], j] > 0.:
                    posts[row[i], j] += w * B[col[i], j]

        # transpose=False, heterogeneous
        case (False, _, jnp.bool_, _):
            # bool
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                if B[col[i], j]:
                    posts[row[i], j] += weights[i]

        case (False, _, _, True):
            # float_as_event
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                if B[col[i], j] > 0.:
                    posts[row[i], j] += weights[i]

        case (False, _, _, False):
            # other
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: spike_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                if B[col[i], j] > 0.:
                    posts[row[i], j] += weights[i] * B[col[i], j]

    dim = (row_info.shape[0], matrix_info.shape[1])
    out_info = kwargs['outs'][0]

    def kernel(weights, row, col, B):
        fn = jax_kernel(mm, launch_dims=dim, num_outputs=1, in_out_argnames=['posts'])
        return fn(weights, row, col, B, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _coomm_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    nnz = row_info.shape[0]
    n = matrix_info.shape[1]
    block_dim = generate_block_dim(nnz)
    block_dim = 32 if block_dim < 32 else block_dim
    block_dim_n = generate_block_dim(n, 512)

    if transpose:
        if weight_info.size == 1:
            # coo.T @ B (homogeneous weights)
            #
            # coo: [m, k]
            # B: [m, n]
            # result: [k, n]
            #
            def mm(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                B_ref,  # [m, n]
                _,  # [k, n]
                posts_ref,  # [k, n]
            ):
                i = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]

                def loop_fn(idx, _):
                    row_idx = rows[idx]
                    col_idx = cols[idx]
                    events = B_ref[row_idx, pl.dslice(i_col_start, block_dim_n)]

                    @pl.when(mask[idx])
                    def process():
                        if B_ref.dtype == jnp.bool_:
                            event_mask = col_mask & events
                        else:
                            event_mask = col_mask & (events > 0.)
                        data = jnp.ones((block_dim_n,), dtype=posts_ref.dtype) * data_ref[0]
                        atomic_add(posts_ref, (col_idx, pl.dslice(i_col_start, block_dim_n)), data, mask=event_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        else:
            # coo.T @ B (heterogeneous weights)
            #
            # coo: [m, k]
            # B: [m, n]
            # result: [k, n]
            #
            def mm(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                B_ref,  # [m, n]
                _,  # [k, n]
                posts_ref,  # [k, n]
            ):
                i = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]
                weights = data_ref[pl.dslice(i_start, block_dim)]

                def loop_fn(idx, _):
                    row_idx = rows[idx]
                    col_idx = cols[idx]
                    w = weights[idx]
                    events = B_ref[row_idx, pl.dslice(i_col_start, block_dim_n)]

                    @pl.when(mask[idx])
                    def process():
                        if B_ref.dtype == jnp.bool_:
                            event_mask = col_mask & events
                        else:
                            event_mask = col_mask & (events > 0.)
                        data = jnp.ones((block_dim_n,), dtype=posts_ref.dtype) * w
                        atomic_add(posts_ref, (col_idx, pl.dslice(i_col_start, block_dim_n)), data, mask=event_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        def kernel(data, row, col, B):
            fn = pl.pallas_call(
                mm,
                grid=(pl.cdiv(nnz, block_dim), pl.cdiv(n, block_dim_n)),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs']
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, B, posts)

        return kernel

    else:
        # coo @ B (non-transpose)
        if weight_info.size == 1:
            # coo @ B (homogeneous weights)
            #
            # coo: [m, k]
            # B: [k, n]
            # result: [m, n]
            #
            def mm(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                B_ref,  # [k, n]
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]

                def loop_fn(idx, _):
                    row_idx = rows[idx]
                    col_idx = cols[idx]
                    events = B_ref[col_idx, pl.dslice(i_col_start, block_dim_n)]

                    @pl.when(mask[idx])
                    def process():
                        if B_ref.dtype == jnp.bool_:
                            event_mask = col_mask & events
                        else:
                            event_mask = col_mask & (events > 0.)
                        data = jnp.ones((block_dim_n,), dtype=posts_ref.dtype) * data_ref[0]
                        atomic_add(posts_ref, (row_idx, pl.dslice(i_col_start, block_dim_n)), data, mask=event_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        else:
            # coo @ B (heterogeneous weights)
            #
            # coo: [m, k]
            # B: [k, n]
            # result: [m, n]
            #
            def mm(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                B_ref,  # [k, n]
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]
                weights = data_ref[pl.dslice(i_start, block_dim)]

                def loop_fn(idx, _):
                    row_idx = rows[idx]
                    col_idx = cols[idx]
                    w = weights[idx]
                    events = B_ref[col_idx, pl.dslice(i_col_start, block_dim_n)]

                    @pl.when(mask[idx])
                    def process():
                        if B_ref.dtype == jnp.bool_:
                            event_mask = col_mask & events
                        else:
                            event_mask = col_mask & (events > 0.)
                        data = jnp.ones((block_dim_n,), dtype=posts_ref.dtype) * w
                        atomic_add(posts_ref, (row_idx, pl.dslice(i_col_start, block_dim_n)), data, mask=event_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        def kernel(data, row, col, B):
            fn = pl.pallas_call(
                mm,
                grid=(pl.cdiv(nnz, block_dim), pl.cdiv(n, block_dim_n)),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs']
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, B, posts)

        return kernel


def _coomm_jvp_left(data_dot, data, row, col, B, *, shape, transpose, **kwargs):
    return [coomm(data_dot, row, col, B, shape=shape, transpose=transpose)]


def _coomm_jvp_right(B_dot, data, row, col, B, *, shape, transpose, **kwargs):
    return [coomm(data, row, col, B_dot, shape=shape, transpose=transpose)]


def _coomm_transpose_rule(
    ct,
    data,
    row,
    col,
    B,
    *,
    shape,
    transpose,
    **kwargs
):
    assert not ad.is_undefined_primal(row)
    assert not ad.is_undefined_primal(col)
    ct = ct[0]

    if ad.is_undefined_primal(B):
        if type(ct) is ad.Zero:
            dB = ad.Zero(B)
        else:
            dB = coomm(data, row, col, ct, shape=shape, transpose=not transpose)
        return data, row, col, dB
    else:
        if type(ct) is ad.Zero:
            d_data = ad.Zero(data)
        else:
            d_data = sddmm_coo_indices(ct, B, row, col).data
        return d_data, row, col, B


def _coomm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = binary_coomm_p_call(
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

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, batch_size, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = binary_coomm_p_call(
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

    elif tuple(axes) == (None, None, None, 2):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = binary_coomm_p_call(
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
        return general_batching_rule(binary_coomm_p_call, args, axes, **kwargs)


def binary_coomm_p_call(
    weights,
    row,
    col,
    B,
    *,
    shape: Sequence[int],
    transpose: bool,
    float_as_event: bool = True,
    backend: Optional[str] = None,
):
    """
    Perform a custom sparse matrix-matrix multiplication operation.

    This function takes a sparse matrix represented in COO (Coordinate) format
    and a dense matrix, then performs a matrix-matrix multiplication.

    Parameters
    ----------
    weights : jax.Array
        The non-zero values of the sparse matrix.
    row : jax.Array
        The row indices of the non-zero values in the sparse matrix.
    col : jax.Array
        The column indices of the non-zero values in the sparse matrix.
    B : jax.Array
        The dense matrix to multiply with the sparse matrix.
    shape : Sequence[int]
        The shape of the sparse matrix.
    transpose : bool
        Whether to transpose the sparse matrix before multiplication.
    float_as_event : bool
        Treat floating-point values as events.
    backend : str, optional
        Backend to use for computation.

    Returns
    -------
    jax.Array
        The result of the sparse matrix-matrix multiplication.
    """
    # Convert scalar weights to a single-element array
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    # Determine the output shape based on whether the sparse matrix is transposed
    out_info = (
        # If transposed, the output shape is [shape[1], B.shape[1]]
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weights.dtype)
        if transpose else
        # If not transposed, the output shape is [shape[0], B.shape[1]]
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weights.dtype)
    )
    # Call the custom kernel with the provided arguments and output information
    return binary_coomm_p(
        weights,
        row,
        col,
        B,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        float_as_event=float_as_event,
        backend=backend,
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


binary_coomm_p = XLACustomKernel('binary_coomm')
binary_coomm_p.def_numba_kernel(_coomm_numba_kernel)
binary_coomm_p.def_warp_kernel(_coomm_warp_kernel)
binary_coomm_p.def_pallas_kernel('gpu', _coomm_pallas_gpu_kernel)
binary_coomm_p.def_pallas_kernel('tpu', _coomm_pallas_gpu_kernel)
binary_coomm_p.def_jvp_rule2(_coomm_jvp_left, None, None, _coomm_jvp_right)
binary_coomm_p.def_transpose_rule(_coomm_transpose_rule)
binary_coomm_p.def_batching_rule(_coomm_batching)
binary_coomm_p.def_call(binary_coomm_p_call)
