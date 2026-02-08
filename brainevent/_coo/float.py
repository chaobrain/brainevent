# Copyright 2024- BrainX Ecosystem Limited. All Rights Reserved.
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
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, namescope
from brainevent._op import jaxinfo_to_warpinfo, XLACustomKernel, general_batching_rule, numba_kernel
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Row, Col, MatrixShape

__all__ = [
    "coomv",
    "coomv_p",
    "coomm",
    "coomm_p",
]


@namescope(static_argnames=("shape", "transpose", "sorted_by_output"))
def coomv(
    data: Data,
    row: Row,
    col: Col,
    vector: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    sorted_by_output: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """
    Perform COO sparse matrix-vector multiplication.

    Args:
        data: The non-zero values of the sparse matrix.
        row: The row indices of the non-zero values.
        col: The column indices of the non-zero values.
        vector: The dense vector to multiply.
        shape: The shape of the sparse matrix (rows, cols).
        transpose: If True, multiply by the transposed matrix.
        sorted_by_output: If True, indices are sorted by the output dimension
            (row for non-transpose, col for transpose), enabling fewer atomics
            in the Warp backend.

    Returns:
        The result of the matrix-vector multiplication.
    """
    data, unitd = u.split_mantissa_unit(data)
    vector, unitv = u.split_mantissa_unit(vector)
    res = coomv_p_call(
        data,
        row,
        col,
        vector,
        shape=shape,
        transpose=transpose,
        sorted_by_output=sorted_by_output,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose"))
def coomm(
    data: Data,
    row: Row,
    col: Col,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
):
    """
    Perform COO sparse matrix-matrix multiplication.

    Args:
        data: The non-zero values of the sparse matrix.
        row: The row indices of the non-zero values.
        col: The column indices of the non-zero values.
        B: The dense matrix to multiply.
        shape: The shape of the sparse matrix (rows, cols).
        transpose: If True, multiply by the transposed matrix.

    Returns:
        The result of the matrix-matrix multiplication.
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = coomm_p_call(data, row, col, B, shape=shape, transpose=transpose, backend=backend)[0]
    return u.maybe_decimal(res * (unitd * unitb))


# =============================================================================
# COO Matrix-Vector Multiplication (coomv)
# =============================================================================


def _coomv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    posts[col[i]] += w * v[row[i]]
        else:
            # transpose=True, heterogeneous
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    posts[col[i]] += weights[i] * v[row[i]]
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    posts[row[i]] += w * v[col[i]]
        else:
            # transpose=False, heterogeneous
            @numba.njit(fastmath=True)
            def mv(weights, row, col, v, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    posts[row[i]] += weights[i] * v[col[i]]

    def kernel(weights, row, col, row_ptr, v):
        # row_ptr is unused for numba backend; present for signature parity
        return numba_kernel(mv, outs=kwargs['outs'])(weights, row, col, v)

    return kernel


def _coomv_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    col_info: jax.ShapeDtypeStruct,
    transpose: bool,
    row_ptr_info: jax.ShapeDtypeStruct,
    sorted_by_output: bool,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    row_warp_info = jaxinfo_to_warpinfo(row_info)
    col_warp_info = jaxinfo_to_warpinfo(col_info)
    vector_warp_info = jaxinfo_to_warpinfo(vector_info)
    row_ptr_warp_info = jaxinfo_to_warpinfo(row_ptr_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    # ------------------------------------------------------------------
    # Sorted path: one thread per output index, no atomics needed.
    # ------------------------------------------------------------------
    if sorted_by_output:
        if transpose:
            if weight_info.size == 1:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    row_ptr: row_ptr_warp_info,
                    v: vector_warp_info,
                    posts: out_warp_info,
                ):
                    o = warp.tid()
                    start = row_ptr[o]
                    end = row_ptr[o + 1]
                    w = weights[0]
                    acc = float(0.0)
                    for idx in range(start, end):
                        acc += w * v[row[idx]]
                    posts[o] = acc
            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    row_ptr: row_ptr_warp_info,
                    v: vector_warp_info,
                    posts: out_warp_info,
                ):
                    o = warp.tid()
                    start = row_ptr[o]
                    end = row_ptr[o + 1]
                    acc = float(0.0)
                    for idx in range(start, end):
                        acc += weights[idx] * v[row[idx]]
                    posts[o] = acc
        else:
            if weight_info.size == 1:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    row_ptr: row_ptr_warp_info,
                    v: vector_warp_info,
                    posts: out_warp_info,
                ):
                    o = warp.tid()
                    start = row_ptr[o]
                    end = row_ptr[o + 1]
                    w = weights[0]
                    acc = float(0.0)
                    for idx in range(start, end):
                        acc += w * v[col[idx]]
                    posts[o] = acc
            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    row: row_warp_info,
                    col: col_warp_info,
                    row_ptr: row_ptr_warp_info,
                    v: vector_warp_info,
                    posts: out_warp_info,
                ):
                    o = warp.tid()
                    start = row_ptr[o]
                    end = row_ptr[o + 1]
                    acc = float(0.0)
                    for idx in range(start, end):
                        acc += weights[idx] * v[col[idx]]
                    posts[o] = acc

        dim = kwargs['outs'][0].shape[0]

        def kernel(weights, row, col, row_ptr, v):
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, row, col, row_ptr, v, jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype))

        return kernel

    # Process multiple nnz per thread to cut the number of global atomics. Each thread
    # walks a small, contiguous chunk and locally accumulates runs of identical output
    # indices before issuing a single atomic_add. This preserves correctness for
    # duplicate indices while reducing contention on popular rows/cols.
    WORK_PER_THREAD = 4

    # ------------------------------------------------------------------
    # Unsorted path: chunked processing + atomics.
    # ------------------------------------------------------------------
    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                row_ptr: row_ptr_warp_info,
                v: vector_warp_info,
                posts: out_warp_info,
            ):
                tid = warp.tid()
                start = tid * WORK_PER_THREAD
                nnz = row.shape[0]
                if start >= nnz:
                    return

                w = weights[0]
                acc = float(0.0)
                dst = int(0)

                for k in range(WORK_PER_THREAD):
                    idx = start + k
                    if idx >= nnz:
                        break
                    dst_idx = col[idx]
                    val = w * v[row[idx]]

                    if k == 0:
                        dst = dst_idx
                        acc = val
                    else:
                        if dst_idx == dst:
                            acc += val
                        else:
                            warp.atomic_add(posts, dst, acc)
                            dst = dst_idx
                            acc = val

                warp.atomic_add(posts, dst, acc)
        else:
            # transpose=True, heterogeneous
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                row_ptr: row_ptr_warp_info,
                v: vector_warp_info,
                posts: out_warp_info,
            ):
                tid = warp.tid()
                start = tid * WORK_PER_THREAD
                nnz = row.shape[0]
                if start >= nnz:
                    return

                acc = float(0.0)
                dst = int(0)

                for k in range(WORK_PER_THREAD):
                    idx = start + k
                    if idx >= nnz:
                        break
                    dst_idx = col[idx]
                    val = weights[idx] * v[row[idx]]

                    if k == 0:
                        dst = dst_idx
                        acc = val
                    else:
                        if dst_idx == dst:
                            acc += val
                        else:
                            warp.atomic_add(posts, dst, acc)
                            dst = dst_idx
                            acc = val

                warp.atomic_add(posts, dst, acc)
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                row_ptr: row_ptr_warp_info,
                v: vector_warp_info,
                posts: out_warp_info,
            ):
                tid = warp.tid()
                start = tid * WORK_PER_THREAD
                nnz = row.shape[0]
                if start >= nnz:
                    return

                w = weights[0]
                acc = float(0.0)
                dst = int(0)

                for k in range(WORK_PER_THREAD):
                    idx = start + k
                    if idx >= nnz:
                        break
                    dst_idx = row[idx]
                    val = w * v[col[idx]]

                    if k == 0:
                        dst = dst_idx
                        acc = val
                    else:
                        if dst_idx == dst:
                            acc += val
                        else:
                            warp.atomic_add(posts, dst, acc)
                            dst = dst_idx
                            acc = val

                warp.atomic_add(posts, dst, acc)
        else:
            # transpose=False, heterogeneous
            @warp.kernel
            def mv(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                row_ptr: row_ptr_warp_info,
                v: vector_warp_info,
                posts: out_warp_info,
            ):
                tid = warp.tid()
                start = tid * WORK_PER_THREAD
                nnz = row.shape[0]
                if start >= nnz:
                    return

                acc = float(0.0)
                dst = int(0)

                for k in range(WORK_PER_THREAD):
                    idx = start + k
                    if idx >= nnz:
                        break
                    dst_idx = row[idx]
                    val = weights[idx] * v[col[idx]]

                    if k == 0:
                        dst = dst_idx
                        acc = val
                    else:
                        if dst_idx == dst:
                            acc += val
                        else:
                            warp.atomic_add(posts, dst, acc)
                            dst = dst_idx
                            acc = val

                warp.atomic_add(posts, dst, acc)

    dim = (row_info.shape[0] + WORK_PER_THREAD - 1) // WORK_PER_THREAD
    out_info = kwargs['outs'][0]

    def kernel(weights, row, col, row_ptr, v):
        fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
        return fn(weights, row, col, row_ptr, v, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _coomv_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    row_info: jax.ShapeDtypeStruct,
    row_ptr_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    sorted_by_output: bool,
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
            def mv(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                row_ptr_ref,  # [out+1]
                vector_ref,  # [m]
                _,  # [k]
                posts_ref,  # [k]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]
                vals = vector_ref[rows]
                data = jnp.asarray(vals, dtype=posts_ref.dtype) * data_ref[0]
                atomic_add(posts_ref, cols, data, mask=mask)

        else:
            # coo.T @ v (heterogeneous weights)
            def mv(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                row_ptr_ref,  # [out+1]
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
                vals = vector_ref[rows]
                data = jnp.asarray(vals, dtype=posts_ref.dtype) * weights
                atomic_add(posts_ref, cols, data, mask=mask)

        def kernel(data, row, col, row_ptr, vector):
            fn = pl.pallas_call(
                mv,
                grid=(pl.cdiv(nnz, block_dim),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs']
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, row_ptr, vector, posts)

        return kernel

    else:
        # coo @ v (non-transpose)
        if weight_info.size == 1:
            # coo @ v (homogeneous weights)
            def mv(
                data_ref,  # [1]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                row_ptr_ref,  # [out+1]
                vector_ref,  # [k]
                _,  # [m]
                posts_ref,  # [m]
            ):
                i = pl.program_id(0)
                i_start = i * block_dim
                mask = i_start + jnp.arange(block_dim) < nnz
                rows = row_ref[pl.dslice(i_start, block_dim)]
                cols = col_ref[pl.dslice(i_start, block_dim)]
                vals = vector_ref[cols]
                data = jnp.asarray(vals, dtype=posts_ref.dtype) * data_ref[0]
                atomic_add(posts_ref, rows, data, mask=mask)

        else:
            # coo @ v (heterogeneous weights)
            def mv(
                data_ref,  # [nnz]
                row_ref,  # [nnz]
                col_ref,  # [nnz]
                row_ptr_ref,  # [out+1]
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
                vals = vector_ref[cols]
                data = jnp.asarray(vals, dtype=posts_ref.dtype) * weights
                atomic_add(posts_ref, rows, data, mask=mask)

        def kernel(data, row, col, row_ptr, vector):
            fn = pl.pallas_call(
                mv,
                grid=(pl.cdiv(nnz, block_dim),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs']
            )
            posts = jnp.zeros(kwargs['outs'][0].shape, dtype=kwargs['outs'][0].dtype)
            return fn(data, row, col, row_ptr, vector, posts)

        return kernel


def _coomv_jvp_vector(vector_dot, data, row, col, vector, *, shape, transpose, **kwargs):
    return [coomv(data, row, col, vector_dot, shape=shape, transpose=transpose,
                  sorted_by_output=kwargs.get('sorted_by_output', False))]


def _coomv_jvp_weights(data_dot, data, row, col, vector, *, shape, transpose, **kwargs):
    return coomv_p_call(
        data_dot, row, col, vector,
        shape=shape,
        transpose=transpose,
        sorted_by_output=kwargs.get('sorted_by_output', False)
    )


def _coomv_transpose_rule(ct, data, row, col, v, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(row)
    assert not ad.is_undefined_primal(col)
    ct = ct[0]

    if ad.is_undefined_primal(v):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(v)
        else:
            ct_events = coomv(
                data,
                row,
                col,
                ct,
                shape=shape,
                transpose=not transpose,
                sorted_by_output=kwargs.get('sorted_by_output', False)
            )
        return data, row, col, ct_events
    else:
        v = jnp.asarray(v)
        if data.aval.shape[0] == 1:  # scalar
            ct_values = coomv_p_call(
                jnp.ones(1, dtype=data.aval.dtype),
                row,
                col,
                v,
                shape=shape,
                transpose=transpose,
                sorted_by_output=kwargs.get('sorted_by_output', False),
            )[0]
            ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
        else:
            ct_values = v[row] * ct[col] if transpose else v[col] * ct[row]
        return ct_values, row, col, v


def _coomv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            sorted_by_output=kwargs.get('sorted_by_output', False)
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = coomm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            sorted_by_output=kwargs.get('sorted_by_output', False)
        )
        return r, [1]

    else:
        return general_batching_rule(coomv_p_call, args, axes, **kwargs)


def _coomv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            nnz = max(1, int(n_pre * n_post * prob))
            row = np.random.randint(0, n_pre, nnz, dtype=np.int32)
            col = np.random.randint(0, n_post, nnz, dtype=np.int32)
            weights = jnp.ones(1, dtype=dtype) if homo else jnp.asarray(np.random.randn(nnz), dtype=dtype)
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, jnp.asarray(row), jnp.asarray(col), vector),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def coomv_p_call(
    weights,
    row,
    col,
    v,
    *,
    shape: Sequence[int],
    transpose: bool,
    sorted_by_output: bool = False,
    backend: Optional[str] = None,
):
    """
    Perform a COO sparse matrix-vector multiplication.

    Parameters
    ----------
    weights : jax.Array
        The non-zero values of the sparse matrix.
    row : jax.Array
        The row indices of the non-zero values.
    col : jax.Array
        The column indices of the non-zero values.
    v : jax.Array
        The dense vector to multiply.
    shape : Sequence[int]
        The shape of the sparse matrix.
    transpose : bool
        Whether to transpose the sparse matrix.
    sorted_by_output : bool
        If True, (row, col) are sorted by the output dimension (row for
        non-transpose, col for transpose), enabling a no-atomic Warp kernel.
    backend : str, optional
        Backend to use for computation.

    Returns
    -------
    jax.Array
        The result of the matrix-vector multiplication.
    """
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    index_dtype = row.dtype
    out_dim = shape[1] if transpose else shape[0]
    # Row pointer for fast, atomic-free Warp path; inexpensive histogram otherwise.
    counts = jnp.bincount(col if transpose else row, length=out_dim)
    row_ptr = jnp.concatenate(
        [jnp.asarray([0], dtype=index_dtype), jnp.cumsum(counts, dtype=index_dtype)]
    )

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weights.dtype)
    )

    return coomv_p(
        weights,
        row,
        col,
        row_ptr,
        v,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        sorted_by_output=sorted_by_output,
        backend=backend,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
        row_ptr_info=jax.ShapeDtypeStruct(row_ptr.shape, row_ptr.dtype),
        vector_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
    )


coomv_p = XLACustomKernel('coomv')
coomv_p.def_numba_kernel(_coomv_numba_kernel)
coomv_p.def_warp_kernel(_coomv_warp_kernel)
coomv_p.def_pallas_kernel('gpu', _coomv_pallas_gpu_kernel)
coomv_p.def_jvp_rule2(_coomv_jvp_weights, None, None, _coomv_jvp_vector)
coomv_p.def_transpose_rule(_coomv_transpose_rule)
coomv_p.def_batching_rule(_coomv_batching)
coomv_p.def_call(coomv_p_call)
coomv_p.def_tags('coo', 'float')
coomv_p.def_benchmark_data(_coomv_benchmark_data)


# =============================================================================
# COO Matrix-Matrix Multiplication (coomm)
# =============================================================================


def _coomm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            # transpose=True, homogeneous
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    posts[col[i], :] += w * B[row[i], :]
        else:
            # transpose=True, heterogeneous
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    posts[col[i], :] += weights[i] * B[row[i], :]
    else:
        if weight_info.size == 1:
            # transpose=False, homogeneous
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(row.shape[0]):
                    posts[row[i], :] += w * B[col[i], :]
        else:
            # transpose=False, heterogeneous
            @numba.njit(fastmath=True)
            def mm(weights, row, col, B, posts):
                posts[:] = 0.
                for i in range(row.shape[0]):
                    posts[row[i], :] += weights[i] * B[col[i], :]

    def kernel(weights, row, col, B):
        return numba_kernel(mm, outs=kwargs['outs'])(weights, row, col, B)

    return kernel


def _coomm_warp_kernel(
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
    matrix_warp_info = jaxinfo_to_warpinfo(matrix_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        if weight_info.size == 1:
            # transpose=True, weight.size==1
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: matrix_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                w = weights[0]
                posts[col[i], j] += w * B[row[i], j]
        else:
            # transpose=True, weight.size!=1
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: matrix_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                posts[col[i], j] += weights[i] * B[row[i], j]
    else:
        if weight_info.size == 1:
            # transpose=False, weight.size==1
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: matrix_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
                w = weights[0]
                posts[row[i], j] += w * B[col[i], j]
        else:
            # transpose=False, weight.size!=1
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                row: row_warp_info,
                col: col_warp_info,
                B: matrix_warp_info,
                posts: out_warp_info
            ):
                i, j = warp.tid()
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
                    vals = B_ref[row_idx, pl.dslice(i_col_start, block_dim_n)]

                    @pl.when(mask[idx])
                    def process():
                        data = jnp.asarray(vals, dtype=posts_ref.dtype) * data_ref[0]
                        atomic_add(posts_ref, (col_idx, pl.dslice(i_col_start, block_dim_n)), data, mask=col_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        else:
            # coo.T @ B (heterogeneous weights)
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
                    vals = B_ref[row_idx, pl.dslice(i_col_start, block_dim_n)]

                    @pl.when(mask[idx])
                    def process():
                        data = jnp.asarray(vals, dtype=posts_ref.dtype) * w
                        atomic_add(posts_ref, (col_idx, pl.dslice(i_col_start, block_dim_n)), data, mask=col_mask)

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
                    vals = B_ref[col_idx, pl.dslice(i_col_start, block_dim_n)]

                    @pl.when(mask[idx])
                    def process():
                        data = jnp.asarray(vals, dtype=posts_ref.dtype) * data_ref[0]
                        atomic_add(posts_ref, (row_idx, pl.dslice(i_col_start, block_dim_n)), data, mask=col_mask)

                jax.lax.fori_loop(0, block_dim, loop_fn, None)

        else:
            # coo @ B (heterogeneous weights)
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
                    vals = B_ref[col_idx, pl.dslice(i_col_start, block_dim_n)]

                    @pl.when(mask[idx])
                    def process():
                        data = jnp.asarray(vals, dtype=posts_ref.dtype) * w
                        atomic_add(posts_ref, (row_idx, pl.dslice(i_col_start, block_dim_n)), data, mask=col_mask)

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
    return coomm_p_call(data_dot, row, col, B, shape=shape, transpose=transpose)


def _coomm_jvp_right(B_dot, data, row, col, B, *, shape, transpose, **kwargs):
    return coomm_p_call(data, row, col, B_dot, shape=shape, transpose=transpose)


def _coomm_transpose_rule(ct, data, row, col, B, *, shape, transpose, **kwargs):
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

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, batch_size, n = args[3].shape
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

    elif tuple(axes) == (None, None, None, 2):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = coomm_p_call(
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
        return general_batching_rule(coomm_p_call, args, axes, **kwargs)


def _coomm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            nnz = max(1, int(n_pre * n_post * prob))
            row = np.random.randint(0, n_pre, nnz, dtype=np.int32)
            col = np.random.randint(0, n_post, nnz, dtype=np.int32)
            weights = jnp.ones(1, dtype=dtype) if homo else jnp.asarray(np.random.randn(nnz), dtype=dtype)
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(BenchmarkConfig(name, (weights, jnp.asarray(row), jnp.asarray(col), B), {
                'shape': (n_pre, n_post), 'transpose': transpose
            }))
    return configs


def coomm_p_call(
    weights,
    row,
    col,
    B,
    *,
    shape: Sequence[int],
    transpose: bool,
    sorted_by_output: bool = False,
    backend: Optional[str] = None,
):
    """
    Perform a COO sparse matrix-matrix multiplication.

    Parameters
    ----------
    weights : jax.Array
        The non-zero values of the sparse matrix.
    row : jax.Array
        The row indices of the non-zero values.
    col : jax.Array
        The column indices of the non-zero values.
    B : jax.Array
        The dense matrix to multiply.
    shape : Sequence[int]
        The shape of the sparse matrix.
    transpose : bool
        Whether to transpose the sparse matrix.
    sorted_by_output : bool
        Unused placeholder to keep signature aligned with coomv batching paths.
    backend : str, optional
        Backend to use for computation.

    Returns
    -------
    jax.Array
        The result of the matrix-matrix multiplication.
    """
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

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
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        sorted_by_output=sorted_by_output,
        backend=backend,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        row_info=jax.ShapeDtypeStruct(row.shape, row.dtype),
        col_info=jax.ShapeDtypeStruct(col.shape, col.dtype),
    )


coomm_p = XLACustomKernel('coomm')
coomm_p.def_numba_kernel(_coomm_numba_kernel)
coomm_p.def_warp_kernel(_coomm_warp_kernel)
coomm_p.def_pallas_kernel('gpu', _coomm_pallas_gpu_kernel)
coomm_p.def_pallas_kernel('tpu', _coomm_pallas_gpu_kernel)
coomm_p.def_jvp_rule2(_coomm_jvp_left, None, None, _coomm_jvp_right)
coomm_p.def_transpose_rule(_coomm_transpose_rule)
coomm_p.def_batching_rule(_coomm_batching)
coomm_p.def_call(coomm_p_call)
coomm_p.def_tags('coo', 'float')
coomm_p.def_benchmark_data(_coomm_benchmark_data)
