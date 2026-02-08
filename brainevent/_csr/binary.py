# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from typing import Optional, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._config import get_numba_parallel
from brainevent._misc import _csr_to_coo, generate_block_dim, namescope
from brainevent._op import jaxinfo_to_warpinfo, numba_kernel, XLACustomKernel, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._sddmm import sddmm_coo_indices
from brainevent._typing import Data, Indptr, Index, MatrixShape
from .float import csrmv, csrmm

__all__ = [
    'binary_csrmv',
    'binary_csrmv_p',
    'binary_csrmm',
    'binary_csrmm_p',
]


@namescope(static_argnames=("shape", "transpose"))
def binary_csrmv(
    data: Data,
    indices: Index,
    indptr: Indptr,
    v: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
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
    res = binary_csrmv_p_call(
        data,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


@namescope(static_argnames=("shape", "transpose"))
def binary_csrmm(
    data: Data,
    indices: Index,
    indptr: Indptr,
    B: Data,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
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
    res = binary_csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def _csrmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba
    if weight_info.size == 1:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                # Cannot parallelize due to race condition on posts[indices[j]]
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(v.shape[0]):
                        if v[i] > 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += w

        else:
            if vector_info.dtype == jnp.bool_:
                # Can parallelize by row
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += w
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    w = weights[0]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] > 0.:
                                r += w
                        posts[i] = r

    else:
        if transpose:
            if vector_info.dtype == jnp.bool_:
                # Cannot parallelize due to race condition
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i]:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

            else:
                @numba.njit(fastmath=True)
                def mv(weights, indices, indptr, v, posts):
                    posts[:] = 0.
                    for i in range(v.shape[0]):
                        if v[i] > 0.:
                            for j in range(indptr[i], indptr[i + 1]):
                                posts[indices[j]] += weights[j]

        else:
            if vector_info.dtype == jnp.bool_:
                # Can parallelize by row
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]]:
                                r += weights[j]
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mv(weights, indices, indptr, v, posts):
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = 0.0
                        for j in range(indptr[i], indptr[i + 1]):
                            if v[indices[j]] > 0.:
                                r += weights[j]
                        posts[i] = r

    def kernel(weights, indices, indptr, vector):
        return numba_kernel(mv, outs=kwargs['outs'])(weights, indices, indptr, vector)

    return kernel


def _csrmv_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    indptr_warp_info = jaxinfo_to_warpinfo(indptr_info)
    spike_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        if weight_info.size == 1:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[i]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += w

            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    w = weights[0]
                    if v[i] > 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += w


        else:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    if v[i]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j]

            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    if v[i] > 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j]] += weights[j]

        def kernel(weights, indices, indptr, v):
            out_info = (
                jax.ShapeDtypeStruct([shape[1]], weights.dtype)
                if transpose else
                jax.ShapeDtypeStruct([shape[0]], weights.dtype)
            )
            dim = vector_info.shape[0] if transpose else indptr_info.shape[0] - 1
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, indptr, v, jnp.zeros(out_info.shape, out_info.dtype))


    else:
        if weight_info.size == 1:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]]:
                            r += w
                    posts[i] = r

            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]] > 0.:
                            r += w
                    posts[i] = r

        else:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]]:
                            r += weights[j]
                    posts[i] = r

            else:
                @warp.kernel
                def mv(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    v: spike_warp_info,
                    posts: out_warp_info,
                ):
                    i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        if v[indices[j]] > 0.:
                            r += weights[j]
                    posts[i] = r

        def kernel(weights, indices, indptr, v):
            out_info = (
                jax.ShapeDtypeStruct([shape[1]], weights.dtype)
                if transpose else
                jax.ShapeDtypeStruct([shape[0]], weights.dtype)
            )
            dim = vector_info.shape[0] if transpose else indptr_info.shape[0] - 1
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, indptr, v)

    return kernel


def _csrmv_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl

    m, k = shape
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, shape[1] if transpose else shape[0]))
    block_dim = block_dim // 2
    block_dim = 32 if block_dim < 32 else block_dim

    if weight_info.size == 1:
        # csr @ B (homogeneous weights)
        #
        # csr: [m, k]
        # B: [k]
        # result: [m]
        #
        def mm(
            data_ref,  # [1]
            indices_ref,  # [nse]
            indptr_ref,  # [m + 1]
            vector_ref,  # [k]
            posts_ref,  # [m]
        ):
            i_row = pl.program_id(0)
            row_start = indptr_ref[i_row]
            row_end = indptr_ref[i_row + 1]
            row_nnz = row_end - row_start
            num_blocks = (row_nnz + block_dim - 1) // block_dim
            val_A = data_ref[0]

            def loop_fn(index, sum_):
                offset = row_start + index * block_dim
                mask = offset + jnp.arange(block_dim) < row_end
                cols = indices_ref[pl.dslice(offset, block_dim)]
                events = vector_ref[cols]
                if vector_ref.dtype == jnp.bool_:
                    events = jnp.asarray(events & mask, dtype=posts_ref.dtype)
                else:
                    events = jnp.where(
                        (events > 0.) & mask,
                        jnp.ones(events.shape, dtype=posts_ref.dtype),
                        0.
                    )
                sum_ += val_A * jnp.sum(events)
                return sum_

            i_row_sum = jax.lax.fori_loop(
                0,
                num_blocks,
                loop_fn,
                jnp.asarray(0., dtype=posts_ref.dtype)
            )
            posts_ref[i_row] = i_row_sum

    else:
        # csr @ B (heterogeneous weights)
        #
        # csr: [m, k]
        # B: [k]
        # result: [m]
        #
        def mm(
            data_ref,  # [nse]
            indices_ref,  # [nse]
            indptr_ref,  # [m + 1]
            vector_ref,  # [k]
            posts_ref,  # [m]
        ):
            i_row = pl.program_id(0)
            row_start = indptr_ref[i_row]
            row_end = indptr_ref[i_row + 1]
            row_nnz = row_end - row_start
            num_blocks = (row_nnz + block_dim - 1) // block_dim

            def loop_fn(index, sum_):
                offset = row_start + index * block_dim
                mask = offset + jnp.arange(block_dim) < row_end
                cols = indices_ref[pl.dslice(offset, block_dim)]
                val_A = data_ref[pl.dslice(offset, block_dim)]
                events = vector_ref[cols]
                if vector_ref.dtype == jnp.bool_:
                    events = jnp.asarray(events & mask, dtype=posts_ref.dtype)
                else:
                    events = jnp.where((events > 0.) & mask, jnp.ones(events.shape, dtype=posts_ref.dtype), 0.)
                sum_ += jnp.sum(val_A * events)
                return sum_

            i_row_sum = jax.lax.fori_loop(
                0,
                num_blocks,
                loop_fn,
                jnp.asarray(0., dtype=posts_ref.dtype)
            )
            posts_ref[i_row] = i_row_sum

    def kernel(data, indices, indptr, vector):
        fn = pl.pallas_call(mm, grid=(m,), out_shape=kwargs['outs'])
        return fn(data, indices, indptr, vector)

    return kernel


def _csrmv_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    m, k = shape
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, shape[1] if transpose else shape[0]))
    block_dim = block_dim // 2
    block_dim = 32 if block_dim < 32 else block_dim

    if transpose:
        if weight_info.size == 1:
            # csr.T @ B (homogeneous weights)
            #
            # csr: [m, k]
            # B: [m]
            # result: [k]
            #
            def mm(
                data_ref,  # [1]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                vector_ref,  # [m]
                _,  # [k]
                posts_ref,  # [k]
            ):
                i_row = pl.program_id(0)
                row_start = indptr_ref[i_row]
                row_end = indptr_ref[i_row + 1]
                row_nnz = row_end - row_start
                num_blocks = (row_nnz + block_dim - 1) // block_dim
                event = vector_ref[i_row]
                data = jnp.ones((block_dim,), dtype=posts_ref.dtype) * data_ref[0]

                @pl.when(event if vector_ref.dtype == jnp.bool_ else event > 0.)
                def event_processing():
                    def loop_fn(index, _):
                        offset = row_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < row_end
                        cols = indices_ref[pl.dslice(offset, block_dim)]
                        atomic_add(posts_ref, cols, data, mask=mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

        else:
            # csr.T @ B (heterogeneous weights)
            #
            # csr: [m, k]
            # B: [m]
            # result: [k]
            #
            def mm(
                data_ref,  # [nse]
                indices_ref,  # [nse]
                indptr_ref,  # [m + 1]
                vector_ref,  # [m]
                _,  # [k]
                posts_ref,  # [k]
            ):
                i_row = pl.program_id(0)
                row_start = indptr_ref[i_row]
                row_end = indptr_ref[i_row + 1]
                row_nnz = row_end - row_start
                num_blocks = (row_nnz + block_dim - 1) // block_dim
                event = vector_ref[i_row]

                @pl.when(event if vector_ref.dtype == jnp.bool_ else event > 0.)
                def event_processing():
                    def loop_fn(index, _):
                        offset = row_start + index * block_dim
                        mask = offset + jnp.arange(block_dim) < row_end
                        cols = indices_ref[pl.dslice(offset, block_dim)]
                        weights = data_ref[pl.dslice(offset, block_dim)]
                        data = jnp.asarray(weights, dtype=posts_ref.dtype)
                        atomic_add(posts_ref, cols, data, mask=mask)

                    jax.lax.fori_loop(0, num_blocks, loop_fn, None)

        def kernel(data, indices, indptr, vector):
            fn = pl.pallas_call(mm, grid=(m,), input_output_aliases={4: 0}, out_shape=kwargs['outs'])
            out = kwargs['outs'][0]
            return fn(data, indices, indptr, vector, jnp.zeros(out.shape, dtype=out.dtype))

        return kernel
    else:
        return _csrmv_pallas_kernel(
            weight_info=weight_info, indices_info=indices_info, shape=shape, transpose=transpose, **kwargs
        )


def _csrmv_jvp_v(v_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    return [csrmv(data, indices, indptr, v_dot, shape=shape, transpose=transpose)]


def _csrmv_jvp_weights(data_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
    return binary_csrmv_p_call(data_dot, indices, indptr, v, shape=shape, transpose=transpose)


def _csrmv_transpose_rule(ct, data, indices, indptr, events, *, shape, transpose, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events)
        else:
            ct_events = csrmv(data, indices, indptr, ct, shape=shape, transpose=not transpose)
        return data, indices, indptr, ct_events
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = binary_csrmv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    indices,
                    indptr,
                    events,
                    shape=shape,
                    transpose=transpose,
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:  # heterogeneous values
                row, col = _csr_to_coo(indices, indptr)
                ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
        return ct_values, indices, indptr, events


def _csrmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )
        return r, [1]

    else:
        return general_batching_rule(binary_csrmv_p, args, axes, **kwargs)


def _binary_csrmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                n_conn = max(1, int(n_post * prob))
                indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
                indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
                weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(n_pre * n_conn, dtype=dtype)
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    vector = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    vector = jnp.asarray(np.random.rand(v_size), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, indices, jnp.asarray(indptr), vector),
                        {'shape': (n_pre, n_post), 'transpose': transpose}
                    )
                )
    return configs


def binary_csrmv_p_call(
    weights,
    indices,
    indptr,
    vector,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
):
    """
    Perform a call to the event CSR matrix-vector multiplication custom operation.

    This function prepares the inputs and calls the binary_csrmv_p custom operation
    to perform matrix-vector multiplication using a CSR (Compressed Sparse Row) format.

    Args:
        weights (jax.Array): Non-zero elements of the CSR sparse matrix.
        indices (jax.Array): Column indices of non-zero elements in the CSR sparse matrix.
        indptr (jax.Array): Index pointers of the CSR sparse matrix, indicating the start of each row.
        vector (jax.Array): The dense vector to be multiplied with the sparse matrix.
        shape (Sequence[int]): A sequence of length 2, representing the shape of the sparse matrix.
        transpose (bool): Whether to transpose the sparse matrix before multiplication.
        backend (str, optional): Backend to use for computation.

    Returns:
        jax.Array: The result of the matrix-vector multiplication.
    """
    assert indices.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indices must be int32 or int64."
    assert indptr.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indptr must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert indptr.dtype == indices.dtype, "Indices and indptr must have the same dtype."
    if transpose:
        assert shape[0] == vector.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == vector.shape[0], "Shape mismatch for non-transpose operation."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    # Check if weights is a scalar. If so, convert it to a one-dimensional array.
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    # Determine the output shape and data type based on whether the sparse matrix is transposed.
    out_info = (
        # If transpose is True, the output shape is (shape[1],).
        jax.ShapeDtypeStruct([shape[1]], weights.dtype)
        if transpose else
        # If transpose is False, the output shape is (shape[0],).
        jax.ShapeDtypeStruct([shape[0]], weights.dtype)
    )
    # Call the binary_csrmv_p custom operation to perform the matrix-vector multiplication.
    return binary_csrmv_p(
        weights,
        indices,
        indptr,
        vector,
        # Initialize a zero vector with the output shape and data type.
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for v.
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
    )


binary_csrmv_p = XLACustomKernel('binary_csrmv')
binary_csrmv_p.def_numba_kernel(_csrmv_numba_kernel)
binary_csrmv_p.def_warp_kernel(_csrmv_warp_kernel)
binary_csrmv_p.def_pallas_kernel('gpu', _csrmv_pallas_gpu_kernel)
binary_csrmv_p.def_pallas_kernel('tpu', _csrmv_pallas_gpu_kernel)
binary_csrmv_p.def_jvp_rule2(_csrmv_jvp_weights, None, None, _csrmv_jvp_v)
binary_csrmv_p.def_transpose_rule(_csrmv_transpose_rule)
binary_csrmv_p.def_batching_rule(_csrmv_batching)
binary_csrmv_p.def_call(binary_csrmv_p_call)
binary_csrmv_p.def_tags('csr', 'binary')
binary_csrmv_p.def_benchmark_data(_binary_csrmv_benchmark_data)


def _csrmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if weight_info.size == 1:
        if transpose:
            #
            # csr.T @ B
            #
            # [k, m] @ [k, n]
            #
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    B = B > 0.
                    w = weights[0]
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += w

        else:
            # csr @ B
            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    w = weights[0]
                    posts[:] = 0.
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(B.shape[1], dtype=weights.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            index = indices[j]
                            for k in range(B.shape[1]):
                                if B[index, k]:
                                    r[k] += w
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    w = weights[0]
                    B = B > 0.
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(B.shape[1], dtype=weights.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            index = indices[j]
                            for k in range(B.shape[1]):
                                if B[index, k]:
                                    r[k] += w
                        posts[i] = r

    else:
        if transpose:
            # csr.T @ B

            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    B = B > 0.
                    posts[:] = 0.
                    for k in numba.prange(B.shape[1]):
                        for i in range(B.shape[0]):
                            if B[i, k]:
                                for j in range(indptr[i], indptr[i + 1]):
                                    posts[indices[j], k] += weights[j]

        else:
            # csr @ B
            # Fixed: Changed range to prange for parallelization

            if vector_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            w = weights[j]
                            B_row = B[col_idx]  # Load entire row once (cache-friendly)
                            for k in range(n_cols):
                                if B_row[k]:
                                    r[k] += w
                        posts[i] = r

            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def mm(weights, indices, indptr, B, posts):
                    n_cols = B.shape[1]
                    for i in numba.prange(indptr.shape[0] - 1):
                        r = np.zeros(n_cols, dtype=posts.dtype)
                        for j in range(indptr[i], indptr[i + 1]):
                            col_idx = indices[j]
                            w = weights[j]
                            for k in range(n_cols):
                                if B[col_idx, k] > 0.:
                                    r[k] += w
                        posts[i] = r

    def kernel(weights, indices, indptr, B):
        return numba_kernel(mm, kwargs['outs'])(weights, indices, indptr, B)

    return kernel


def _csrmm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    indptr_warp_info = jaxinfo_to_warpinfo(indptr_info)
    spike_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # csr.T @ B
        if weight_info.size == 1:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    if B[i, k]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += w

            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    if B[i, k] > 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += w

        else:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    if B[i, k]:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += weights[j]

            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    if B[i, k] > 0.:
                        for j in range(indptr[i], indptr[i + 1]):
                            posts[indices[j], k] += weights[j]

        def kernel(weights, indices, indptr, B):
            n = vector_info.shape[1]
            out_info = jax.ShapeDtypeStruct([shape[1], n], weights.dtype)
            dim = tuple(reversed(vector_info.shape))  # (n, m)
            fn = jax_kernel(mm, launch_dims=dim, num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, indptr, B, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        # csr @ B
        if weight_info.size == 1:
            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k]:
                            r += w
                    posts[i, k] = r

            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    w = weights[0]
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k] > 0.:
                            r += w
                    posts[i, k] = r

        else:
            # csr @ B

            if vector_info.dtype == jnp.bool_:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k]:
                            r += weights[j]
                    posts[i, k] = r

            else:
                @warp.kernel
                def mm(
                    weights: weight_warp_info,
                    indices: indices_warp_info,
                    indptr: indptr_warp_info,
                    B: spike_warp_info,
                    posts: out_warp_info,
                ):
                    k, i = warp.tid()
                    r = weights.dtype(0.)
                    for j in range(indptr[i], indptr[i + 1]):
                        index = indices[j]
                        if B[index, k] > 0.:
                            r += weights[j]
                    posts[i, k] = r

        def kernel(weights, indices, indptr, B):
            n = vector_info.shape[1]
            out_info = jax.ShapeDtypeStruct([shape[0], n], weights.dtype)
            dim = (vector_info.shape[1], indptr_info.shape[0] - 1)
            fn = jax_kernel(mm, launch_dims=dim, num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, indptr, B)

    return kernel


def _csrmm_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs
):
    from jax.experimental import pallas as pl

    m, k = shape
    n = vector_info.shape[1]

    # Block dimension for output columns
    block_dim_n = generate_block_dim(n, 512)

    # Block dimension for non-zeros (csrmv pattern)
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, m))
    block_dim = block_dim // 2
    block_dim = 32 if block_dim < 32 else block_dim

    if weight_info.size == 1:
        #
        # Gustavson algorithm: Sparse matrix–matrix multiplication is performed in a row-wise fashion.
        #
        # Each nonzero value in a row is multiplied by the nonzero values corresponding to the column index.
        # These values are summed and stored in a temporary row buffer based on their column indices.

        # csr @ B (homogeneous weights)
        #
        # csr: [m, k]
        # B: [k, n]
        # result: [m, n]
        #
        def mm(
            data_ref,  # [1]
            indices_ref,  # [nse]
            indptr_ref,  # [m + 1]
            B_ref,  # [k, n]
            posts_ref,  # [m, n]
        ):
            i_row = pl.program_id(0)
            i_n = pl.program_id(1)
            i_col_start = i_n * block_dim_n
            col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

            row_start = indptr_ref[i_row]
            row_end = indptr_ref[i_row + 1]
            row_nnz = row_end - row_start
            num_blocks = (row_nnz + block_dim - 1) // block_dim
            val_A = data_ref[0]

            def loop_fn(index, sum_):
                offset = row_start + index * block_dim
                nnz_mask = offset + jnp.arange(block_dim) < row_end

                cols = indices_ref[pl.dslice(offset, block_dim)]
                events = B_ref[cols, pl.dslice(i_col_start, block_dim_n)]
                events = jnp.asarray(events, dtype=posts_ref.dtype)

                contribution = jnp.sum(jnp.where(nnz_mask[:, None], events, 0.), axis=0)
                sum_ += val_A * contribution
                return sum_

            i_row_sum = jax.lax.fori_loop(
                0, num_blocks, loop_fn,
                jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
            )
            posts_ref[i_row, pl.dslice(i_col_start, block_dim_n)] = jnp.where(
                col_mask, i_row_sum, 0.
            )

    else:
        #
        # Gustavson algorithm: Sparse matrix–matrix multiplication is performed in a row-wise fashion.
        #
        # Each nonzero value in a row is multiplied by the nonzero values corresponding to the column index.
        # These values are summed and stored in a temporary row buffer based on their column indices.

        # csr @ B (heterogeneous weights)
        #
        # csr: [m, k]
        # B: [k, n]
        # result: [m, n]
        #
        def mm(
            data_ref,  # [nse]
            indices_ref,  # [nse]
            indptr_ref,  # [m + 1]
            B_ref,  # [k, n]
            posts_ref,  # [m, n]
        ):
            i_row = pl.program_id(0)
            i_n = pl.program_id(1)
            i_col_start = i_n * block_dim_n
            col_mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]

            row_start = indptr_ref[i_row]
            row_end = indptr_ref[i_row + 1]
            row_nnz = row_end - row_start
            num_blocks = (row_nnz + block_dim - 1) // block_dim

            def loop_fn(index, sum_):
                offset = row_start + index * block_dim
                nnz_mask = offset + jnp.arange(block_dim) < row_end

                cols = indices_ref[pl.dslice(offset, block_dim)]
                val_A = data_ref[pl.dslice(offset, block_dim)]
                events = B_ref[cols, pl.dslice(i_col_start, block_dim_n)]
                events = jnp.asarray(events, dtype=posts_ref.dtype)

                weighted = val_A[:, None] * events
                contribution = jnp.sum(jnp.where(nnz_mask[:, None], weighted, 0.), axis=0)
                sum_ += contribution
                return sum_

            i_row_sum = jax.lax.fori_loop(
                0, num_blocks, loop_fn,
                jnp.zeros([block_dim_n], dtype=posts_ref.dtype)
            )
            posts_ref[i_row, pl.dslice(i_col_start, block_dim_n)] = jnp.where(
                col_mask, i_row_sum, 0.
            )

    def kernel(data, indices, indptr, B):
        fn = pl.pallas_call(
            mm,
            grid=(m, pl.cdiv(n, block_dim_n)),
            out_shape=kwargs['outs']
        )
        return fn(data, indices, indptr, B)

    return kernel


def _csrmm_pallas_gpu_kernel(
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl

    m, k = shape
    n = vector_info.shape[1]

    # Block dimension for output columns
    block_dim_n = generate_block_dim(n, 512)

    # Block dimension for non-zeros (csrmv pattern)
    block_dim = generate_block_dim(pl.cdiv(indices_info.size, m))
    block_dim = block_dim // 2
    block_dim = 32 if block_dim < 32 else block_dim

    if transpose:
        if weight_info.size == 1:
            # csr.T @ B (homogeneous weights)
            #
            # csr: [m, k]
            # B: [m, n]
            # result: [k, n]
            #
            def mm(
                data_ref,  # [1]
                indices_ref,  # [nse]
                indptr_ref,  # [k + 1]
                B_ref,  # [k, n]
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i_k = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                col_start = indptr_ref[i_k]
                col_end = indptr_ref[i_k + 1]
                mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                events = pl.load(B_ref, (i_k, pl.dslice(i_col_start, block_dim_n)), mask=mask)
                if B_ref.dtype == jnp.bool_:
                    mask = mask & events
                    val = jnp.where(events, data_ref[0], 0.)
                else:
                    mask = mask & (events > 0.)
                    val = events * data_ref[0]

                def loop_fn(index, _):
                    i_row = indices_ref[index]
                    pl.atomic_add(posts_ref, (i_row, pl.dslice(i_col_start, block_dim_n)), val, mask=mask)

                jax.lax.fori_loop(col_start, col_end, loop_fn, None, )

        else:
            # csr.T @ B (heterogeneous weights)
            #
            # csr: [m, k]
            # B: [m, n]
            # result: [k, n]
            #
            def mm(
                data_ref,  # [nse]
                indices_ref,  # [nse]
                indptr_ref,  # [k + 1]
                B_ref,  # [k, n]
                _,  # [m, n]
                posts_ref,  # [m, n]
            ):
                i_k = pl.program_id(0)
                i_n = pl.program_id(1)
                i_col_start = i_n * block_dim_n
                mask = (i_col_start + jnp.arange(block_dim_n)) < B_ref.shape[1]
                col_start = indptr_ref[i_k]
                col_end = indptr_ref[i_k + 1]
                events = pl.load(B_ref, (i_k, pl.dslice(i_col_start, block_dim_n)), mask=mask)
                if B_ref.dtype == jnp.bool_:
                    mask = mask & events
                else:
                    mask = mask & (events > 0.)

                def loop_fn(index, _):
                    i_row = indices_ref[index]
                    val_A = data_ref[index]
                    if B_ref.dtype == jnp.bool_:
                        val = jnp.where(events, val_A, 0.)
                    else:
                        val = events * val_A
                    pl.atomic_add(posts_ref, (i_row, pl.dslice(i_col_start, block_dim_n)), val, mask=mask)

                jax.lax.fori_loop(col_start, col_end, loop_fn, None, )

        def kernel(data, indices, indptr, B):
            out_info = kwargs['outs'][0]
            fn = pl.pallas_call(
                mm,
                grid=(m, pl.cdiv(n, block_dim_n)),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs']
            )
            posts = jnp.zeros(out_info.shape, dtype=out_info.dtype)
            return fn(data, indices, indptr, B, posts)

        return kernel
    else:
        return _csrmm_pallas_kernel(
            weight_info=weight_info, indices_info=indices_info, vector_info=vector_info, shape=shape, **kwargs
        )


def _csrmm_jvp_data(data_dot, data, indices, indptr, B, *, shape, transpose, **kwargs):
    return [csrmm(data_dot, indices, indptr, B, shape=shape, transpose=transpose)]


def _csrmm_jvp_B(B_dot, data, indices, indptr, B, *, shape, transpose, **kwargs):
    return [csrmm(data, indices, indptr, B_dot, shape=shape, transpose=transpose)]


def _csrmm_transpose_rule(ct, data, indices, indptr, B, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)

    if ad.is_undefined_primal(B):
        dB = csrmm(data, indices, indptr, ct, shape=shape, transpose=not transpose)
        return data, indices, indptr, dB
    else:
        B = jnp.asarray(B)
        if data.aval.shape[0] == 1:  # scalar
            r = binary_csrmm_p_call(
                jnp.ones(1, dtype=data.aval.dtype),
                indices,
                indptr,
                B,
                shape=shape,
                transpose=transpose,
            )[0]
            return jnp.expand_dims(jnp.sum(r * ct), axis=0), indices, indptr, B
        else:
            row, col = _csr_to_coo(indices, indptr)
            if transpose:
                d_data = sddmm_coo_indices(B, ct.T, row, col).data
            else:
                d_data = sddmm_coo_indices(B, ct.T, col, row).data
            return d_data, indices, indptr, B


def _csrmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 1):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, batch_size, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )[0]
        r = jnp.reshape(r, [r.shape[0], batch_size, n])
        return [r], [1]

    elif tuple(axes) == (None, None, None, 2):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        m, n, batch_size = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
        )[0]
        r = jnp.reshape(r, [r.shape[0], n, batch_size])
        return [r], [2]

    else:
        return general_batching_rule(binary_csrmm_p, args, axes, **kwargs)


def _binary_csrmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                n_conn = max(1, int(n_post * prob))
                indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
                indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
                weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(n_pre * n_conn, dtype=dtype)
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    B = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    B = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, indices, jnp.asarray(indptr), B),
                        {'shape': (n_pre, n_post), 'transpose': transpose}
                    )
                )
    return configs


def binary_csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
):
    """
    Perform a call to the event CSR matrix-matrix multiplication custom operation.

    Args:
        weights (jax.Array): Non-zero elements of the CSR sparse matrix.
        indices (jax.Array): Column indices of non-zero elements in the CSR sparse matrix.
        indptr (jax.Array): Index pointers of the CSR sparse matrix, indicating the start of each row.
        B (jax.Array): A dense matrix.
        shape (Sequence[int]): A sequence of length 2, representing the shape of the sparse matrix.
        transpose (bool): A boolean indicating whether to transpose the sparse matrix before multiplication.
        backend (str, optional): Backend to use for computation.

    Returns:
        jax.Array: The result of the matrix-matrix multiplication.
    """
    assert indices.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indices must be int32 or int64."
    assert indptr.dtype in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64], "Indptr must be int32 or int64."
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert indptr.dtype == indices.dtype, "Indices and indptr must have the same dtype."
    if transpose:
        assert shape[0] == B.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == B.shape[0], f"Shape mismatch for non-transpose operation. {shape[1]} != {B.shape[0]}"
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    # Check if weights is a scalar. If so, convert it to a one-dimensional array.
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    # Determine the output shape and data type based on whether the sparse matrix is transposed.
    out_info = (
        # If transpose is True, the output shape is (shape[1], B.shape[1]).
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weights.dtype)
        if transpose else
        # If transpose is False, the output shape is (shape[0], B.shape[1]).
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weights.dtype)
    )
    # Call the binary_csrmm_p custom operation to perform the matrix-matrix multiplication.
    return binary_csrmm_p(
        weights,
        indices,
        indptr,
        B,
        outs=(out_info,),
        shape=shape,
        transpose=transpose,
        backend=backend,
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for B.
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
    )


binary_csrmm_p = XLACustomKernel('binary_csrmm')
binary_csrmm_p.def_numba_kernel(_csrmm_numba_kernel)
binary_csrmm_p.def_warp_kernel(_csrmm_warp_kernel)
binary_csrmm_p.def_pallas_kernel('gpu', _csrmm_pallas_gpu_kernel)
binary_csrmm_p.def_pallas_kernel('tpu', _csrmm_pallas_gpu_kernel)
binary_csrmm_p.def_jvp_rule2(_csrmm_jvp_data, None, None, _csrmm_jvp_B)
binary_csrmm_p.def_transpose_rule(_csrmm_transpose_rule)
binary_csrmm_p.def_batching_rule(_csrmm_batching)
binary_csrmm_p.def_call(binary_csrmm_p_call)
binary_csrmm_p.def_tags('csr', 'binary')
binary_csrmm_p.def_benchmark_data(_binary_csrmm_benchmark_data)
