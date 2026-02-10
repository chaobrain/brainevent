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

from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._config import get_numba_parallel
from brainevent._misc import cdiv, generate_block_dim, namescope
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig

__all__ = [
    'indexed_binary_densemv',
    'indexed_binary_densemv_p',
    'indexed_binary_densemm',
    'indexed_binary_densemm_p',
]


# ==============================================================================
# Unified indexed binary dense matrix-vector product (indexed_binary_densemv)
# ==============================================================================
#
# transpose=False: weights[m,k] columns selected by indices -> out[m]  (old indexed_dbmv)
# transpose=True:  weights[k,n] rows selected by indices -> out[n]    (old indexed_bdvm)
#
# Argument order is always (weights, binary_index).


@namescope(static_argnames=['transpose'])
def indexed_binary_densemv(weights, binary_index, *, transpose, backend: Optional[str] = None):
    """
    Performs indexed binary dense matrix-vector multiplication.

    When ``transpose=False``, computes ``weights[m,k] @ binary_index -> out[m]``
    by summing the columns of ``weights`` selected by the spike indices.

    When ``transpose=True``, computes ``binary_index @ weights[k,n] -> out[n]``
    by summing the rows of ``weights`` selected by the spike indices.

    Parameters
    ----------
    weights : ndarray or compatible
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``. Can be a ``brainunit`` quantity.
    binary_index : BinaryArray
        An object representing a binary vector in sparse format with attributes:
        ``value``, ``spike_indices``, ``spike_count``.
    transpose : bool
        If False, compute ``weights @ binary``. If True, compute ``binary @ weights``.
    backend : str, optional
        Backend to use for the computation.

    Returns
    -------
    result : ndarray or compatible
        Result vector. Shape ``(m,)`` when ``transpose=False``,
        or ``(n,)`` when ``transpose=True``.
    """
    weight_val, wunit = u.split_mantissa_unit(weights)
    spikes = binary_index.value
    indices = binary_index.spike_indices
    count = binary_index.spike_count
    r = indexed_binary_densemv_p_call(spikes, indices, count, weight_val, transpose=transpose, backend=backend)
    return u.maybe_decimal(r[0] * wunit)


def _mv_numba_kernel(transpose: bool, **kwargs):
    import numba

    if transpose:
        # weights[k,n], select rows by indices -> out[n]
        @numba.njit(fastmath=True)
        def kernel(indices, count, weights, out):
            out[:] = 0.
            nnz = min(count[0], indices.shape[0])
            for i in range(nnz):
                idx = indices[i]
                if 0 <= idx < weights.shape[0]:
                    out += weights[idx]
    else:
        # weights[m,k], select columns by indices -> out[m]
        @numba.njit(fastmath=True)
        def kernel(indices, count, weights, out):
            out[:] = 0.
            nnz = min(count[0], indices.shape[0])
            for i in range(nnz):
                idx = indices[i]
                if 0 <= idx < weights.shape[1]:
                    out += weights[:, idx]

    def run(spikes, indices, count, weights):
        return numba_kernel(kernel, outs=kwargs['outs'])(indices, count, weights)

    return run


def _mv_warp_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    count_warp_info = jaxinfo_to_warpinfo(count_info)
    weight_warp_info = jaxinfo_to_warpinfo(weights_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # weights[k,n], select rows -> out[n]
        n = weights_info.shape[1]

        @warp.kernel
        def kernel(
            indices: indices_warp_info,
            count: count_warp_info,
            weights: weight_warp_info,
            out: out_warp_info,
        ):
            j = warp.tid()
            r = weights.dtype(0.)
            nnz = count[0]
            max_i = indices.shape[0]
            if nnz > max_i:
                nnz = max_i
            for i in range(nnz):
                idx = indices[i]
                if 0 <= idx < weights.shape[0]:
                    r += weights[idx, j]
            out[j] = r

        def run(spikes, indices, count, weights):
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=[n], num_outputs=1, output_dims={'out': out_info.shape})
            return fn(indices, count, weights)
    else:
        # weights[m,k], select columns -> out[m]
        m = weights_info.shape[0]

        @warp.kernel
        def kernel(
            indices: indices_warp_info,
            count: count_warp_info,
            weights: weight_warp_info,
            out: out_warp_info,
        ):
            i = warp.tid()
            r = weights.dtype(0.)
            nnz = count[0]
            max_k = indices.shape[0]
            if nnz > max_k:
                nnz = max_k
            for k in range(nnz):
                idx = indices[k]
                if 0 <= idx < weights.shape[1]:
                    r += weights[i, idx]
            out[i] = r

        def run(spikes, indices, count, weights):
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=[m], num_outputs=1, output_dims={'out': out_info.shape})
            return fn(indices, count, weights)

    return run


def _mv_pallas_kernel(
    weights_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl

    if transpose:
        # weights[k,n], select rows -> out[n]
        block_dim = generate_block_dim(weights_info.shape[1], maximum=128)

        def kernel(
            indices_ref,  # [n_neuron]
            count_ref,  # [1]
            weights_ref,  # [k, n]
            out_ref,  # [n]
        ):
            i_block = pl.program_id(0)
            col_start = i_block * block_dim
            cols = col_start + jnp.arange(block_dim)
            mask = cols < weights_ref.shape[1]
            safe_cols = jnp.where(mask, cols, 0)
            count = jnp.minimum(count_ref[0], indices_ref.shape[0])

            def fn(i, temp):
                i_row = indices_ref[i]
                valid = (i_row >= 0) & (i_row < weights_ref.shape[0])
                i_row = jnp.where(valid, i_row, 0)
                weight_row = jnp.where(mask & valid, weights_ref[i_row, safe_cols], 0.0)
                return temp + weight_row

            out = jax.lax.fori_loop(0, count, fn, jnp.zeros([block_dim], dtype=weights_ref.dtype))
            out_ref[safe_cols] = jnp.where(mask, out, 0.0)

        def run(spikes, indices, count, weights):
            fn = pl.pallas_call(kernel, grid=(cdiv(weights_info.shape[1], block_dim),), out_shape=kwargs['outs'])
            return fn(indices, count, weights)
    else:
        # weights[m,k], select columns -> out[m]
        block_dim = generate_block_dim(weights_info.shape[0], maximum=128)

        def kernel(
            indices_ref,  # [n_neuron]
            count_ref,  # [1]
            weights_ref,  # [m, k]
            out_ref,  # [m]
        ):
            i_block = pl.program_id(0)
            row_start = i_block * block_dim
            rows = row_start + jnp.arange(block_dim)
            mask = rows < weights_ref.shape[0]
            safe_rows = jnp.where(mask, rows, 0)
            count = jnp.minimum(count_ref[0], indices_ref.shape[0])

            def fn(i, temp):
                i_col = indices_ref[i]
                valid = (i_col >= 0) & (i_col < weights_ref.shape[1])
                i_col = jnp.where(valid, i_col, 0)
                weight_col = jnp.where(mask & valid, weights_ref[safe_rows, i_col], 0.0)
                return temp + weight_col

            out = jax.lax.fori_loop(0, count, fn, jnp.zeros([block_dim], dtype=weights_ref.dtype))
            out_ref[safe_rows] = jnp.where(mask, out, 0.0)

        def run(spikes, indices, count, weights):
            fn = pl.pallas_call(kernel, grid=(cdiv(weights_info.shape[0], block_dim),), out_shape=kwargs['outs'])
            return fn(indices, count, weights)

    return run


def _mv_jvp_spikes(spikes_dot, spikes, indices, count, weights, *, transpose, **kwargs):
    if transpose:
        return [jnp.zeros((weights.shape[1],), dtype=weights.dtype)]
    else:
        return [jnp.zeros((weights.shape[0],), dtype=weights.dtype)]


def _mv_jvp_weights(weights_dot, spikes, indices, count, weights, *, transpose, **kwargs):
    return indexed_binary_densemv_p_call(
        spikes, indices, count, weights_dot, transpose=transpose, backend=kwargs['backend'],
    )


def _mv_transpose(ct, spikes, indices, count, weights, *, transpose, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to indices.")
    if ad.is_undefined_primal(count):
        raise ValueError("Cannot transpose with respect to count.")
    ct = ct[0]
    if ad.is_undefined_primal(spikes):
        return ad.Zero(spikes), indices, count, weights
    if ad.is_undefined_primal(weights):
        if type(ct) is ad.Zero:
            return spikes, indices, count, ad.Zero(weights)
        mask = jnp.arange(indices.shape[0]) < count[0]
        if transpose:
            # kernel sums rows: ct_weights[indices[i]] += ct
            updates = jnp.where(mask[:, None], ct, 0.0)
            zeros = jnp.zeros(weights.aval.shape, dtype=weights.aval.dtype)
            ct_weights = zeros.at[indices].add(updates)
        else:
            # kernel sums columns: ct_weights[:, indices[i]] += ct
            updates = jnp.where(mask[:, None], ct, 0.0)
            zeros = jnp.zeros(weights.aval.shape, dtype=weights.aval.dtype)
            ct_weights = zeros.at[:, indices].add(updates)
        return spikes, indices, count, ct_weights
    raise ValueError("Cannot transpose with respect to both spikes and weights.")


def _mv_batching(args, axes, *, transpose, **kwargs):
    if axes == (None, None, None, 0):
        spikes, indices, count, weights = args
        mask = jnp.arange(indices.shape[0]) < count[0]
        if transpose:
            # weights batched: [batch, k, n], select rows -> [batch, n]
            gathered = jnp.take(weights, indices, axis=1)
            updates = jnp.where(mask[None, :, None], gathered, 0.0)
            r = updates.sum(axis=1)
        else:
            # weights batched: [batch, m, k], select columns -> [batch, m]
            gathered = jnp.take(weights, indices, axis=2)
            updates = jnp.where(mask[None, None, :], gathered, 0.0)
            r = updates.sum(axis=2)
        return [r], [0]
    return general_batching_rule(indexed_binary_densemv_p, args, axes, transpose=transpose, **kwargs)


def _mv_benchmark_data(*, platform):
    n_input, n_output = 1000, 1000
    n_spikes = 100
    dtype = jnp.float32
    spikes = jnp.ones(n_input, dtype=dtype)
    indices = jnp.asarray(np.random.choice(n_input, n_spikes, replace=False).astype(np.int32))
    count = jnp.asarray([n_spikes], dtype=jnp.int32)
    weights = jnp.asarray(np.random.randn(n_input, n_output), dtype=dtype)
    return [
        BenchmarkConfig("default", (spikes, indices, count, weights)),
    ]


def indexed_binary_densemv_p_call(spikes, indices, count, weights, *, transpose, backend: Optional[str] = None):
    assert spikes.ndim == 1, "spikes should be 1D (n_spikes,)"
    assert indices.ndim == 1, "indices should be 1D (n_spikes,)"
    assert count.ndim == 1 and count.shape[0] == 1, "count should be 1D (1,)"
    assert weights.ndim == 2, "weights should be 2D"
    if transpose:
        # weights[k,n], select rows by indices -> out[n]
        assert spikes.shape[0] == weights.shape[0], (
            f"spikes and weights dimension mismatch, "
            f"got {spikes.shape} and {weights.shape}"
        )
        out_shape = jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
    else:
        # weights[m,k], select columns by indices -> out[m]
        assert spikes.shape[0] == weights.shape[1], (
            f"spikes and weights dimension mismatch, "
            f"got {spikes.shape} and {weights.shape}"
        )
        out_shape = jax.ShapeDtypeStruct([weights.shape[0]], weights.dtype)
    return indexed_binary_densemv_p(
        spikes,
        indices,
        count,
        weights,
        outs=[out_shape],
        transpose=transpose,
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        count_info=jax.ShapeDtypeStruct(count.shape, count.dtype),
        weights_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        backend=backend,
    )


indexed_binary_densemv_p = XLACustomKernel('indexed_binary_densemv')
indexed_binary_densemv_p.def_numba_kernel(_mv_numba_kernel)
indexed_binary_densemv_p.def_warp_kernel(_mv_warp_kernel)
indexed_binary_densemv_p.def_pallas_kernel('gpu', _mv_pallas_kernel)
indexed_binary_densemv_p.def_pallas_kernel('tpu', _mv_pallas_kernel)
indexed_binary_densemv_p.def_jvp_rule2(_mv_jvp_spikes, None, None, _mv_jvp_weights)
indexed_binary_densemv_p.def_transpose_rule(_mv_transpose)
indexed_binary_densemv_p.def_batching_rule(_mv_batching)
indexed_binary_densemv_p.def_call(indexed_binary_densemv_p_call)
indexed_binary_densemv_p.def_tags('dense', 'indexed_binary')
indexed_binary_densemv_p.def_benchmark_data(_mv_benchmark_data)


# ==============================================================================
# Unified indexed binary dense matrix-matrix product (indexed_binary_densemm)
# ==============================================================================
#
# transpose=False: weights[m,k] columns selected by indices -> out[batch, m]  (old indexed_dbmm)
# transpose=True:  weights[k,n] rows selected by indices -> out[batch, n]    (old indexed_bdmm)
#
# Argument order is always (weights, binary_arr).


@namescope(static_argnames=['transpose'])
def indexed_binary_densemm(weights, binary_arr, *, transpose, backend: Optional[str] = None):
    """
    Performs indexed binary dense matrix-matrix multiplication (batched).

    When ``transpose=False``, computes ``weights[m,k] @ binary_arr -> out[batch, m]``
    by summing the columns of ``weights`` selected by each batch's spike indices.

    When ``transpose=True``, computes ``binary_arr @ weights[k,n] -> out[batch, n]``
    by summing the rows of ``weights`` selected by each batch's spike indices.

    Parameters
    ----------
    weights : ndarray or compatible
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``. Can be a ``brainunit`` quantity.
    binary_arr : BinaryArray
        An object representing a batch of binary vectors in sparse format with attributes:
        ``value`` (batch, n_spikes), ``spike_indices`` (batch, n_spikes),
        ``spike_count`` (batch,).
    transpose : bool
        If False, compute ``weights @ binary``. If True, compute ``binary @ weights``.
    backend : str, optional
        Backend to use for the computation.

    Returns
    -------
    result : ndarray or compatible
        Result matrix. Shape ``(batch, m)`` when ``transpose=False``,
        or ``(batch, n)`` when ``transpose=True``.
    """
    weights, wunit = u.split_mantissa_unit(weights)
    spikes = binary_arr.value
    indices = binary_arr.spike_indices
    count = binary_arr.spike_count
    r = indexed_binary_densemm_p_call(spikes, indices, count, weights, transpose=transpose, backend=backend)
    return u.maybe_decimal(r[0] * wunit)


def _mm_numba_kernel(transpose: bool, **kwargs):
    import numba

    if transpose:
        # weights[k,n], select rows by indices -> out[batch, n]
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(indices, count, weights, out):
            for i_row in numba.prange(indices.shape[0]):
                temp = np.zeros(weights.shape[1], dtype=weights.dtype)
                nnz = min(count[i_row], indices.shape[1])
                for i_col in range(nnz):
                    idx = indices[i_row, i_col]
                    if 0 <= idx < weights.shape[0]:
                        temp += weights[idx]
                out[i_row] = temp
    else:
        # weights[m,k], select columns by indices -> out[batch, m]
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(indices, count, weights, out):
            for i_row in numba.prange(indices.shape[0]):
                temp = np.zeros(weights.shape[0], dtype=weights.dtype)
                nnz = min(count[i_row], indices.shape[1])
                for i_col in range(nnz):
                    idx = indices[i_row, i_col]
                    if 0 <= idx < weights.shape[1]:
                        temp += weights[:, idx]
                out[i_row] = temp

    def run(spikes, indices, count, weights):
        return numba_kernel(kernel, outs=kwargs['outs'])(indices, count, weights)

    return run


def _mm_warp_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    batch = indices_info.shape[0]
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    count_warp_info = jaxinfo_to_warpinfo(count_info)
    weight_warp_info = jaxinfo_to_warpinfo(weights_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # weights[k,n], select rows -> out[batch, n]
        n_out = weights_info.shape[1]

        @warp.kernel
        def kernel(
            indices: indices_warp_info,
            count: count_warp_info,
            weights: weight_warp_info,
            out: out_warp_info,
        ):
            i_row, j = warp.tid()
            r = weights.dtype(0.)
            nnz = count[i_row]
            max_k = indices.shape[1]
            if nnz > max_k:
                nnz = max_k
            for k in range(nnz):
                idx = indices[i_row, k]
                if 0 <= idx < weights.shape[0]:
                    r += weights[idx, j]
            out[i_row, j] = r

        def run(spikes, indices, count, weights):
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=(batch, n_out), num_outputs=1, output_dims={'out': out_info.shape})
            return fn(indices, count, weights)
    else:
        # weights[m,k], select columns -> out[batch, m]
        n_out = weights_info.shape[0]

        @warp.kernel
        def kernel(
            indices: indices_warp_info,
            count: count_warp_info,
            weights: weight_warp_info,
            out: out_warp_info,
        ):
            i_row, j = warp.tid()
            r = weights.dtype(0.)
            nnz = count[i_row]
            max_k = indices.shape[1]
            if nnz > max_k:
                nnz = max_k
            for k in range(nnz):
                idx = indices[i_row, k]
                if 0 <= idx < weights.shape[1]:
                    r += weights[j, idx]
            out[i_row, j] = r

        def run(spikes, indices, count, weights):
            out_info = kwargs['outs'][0]
            fn = jax_kernel(kernel, launch_dims=(batch, n_out), num_outputs=1, output_dims={'out': out_info.shape})
            return fn(indices, count, weights)

    return run


def _mm_pallas_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    from jax.experimental import pallas as pl

    if transpose:
        # weights[k,n], select rows -> out[batch, n]
        block_dim = generate_block_dim(weights_info.shape[1], maximum=128)

        def kernel(
            indices_ref,  # [batch, n_spikes]
            count_ref,  # [batch]
            weights_ref,  # [k, n]
            out_ref,  # [batch, n]
        ):
            i_row = pl.program_id(0)
            i_block = pl.program_id(1)
            col_start = i_block * block_dim
            cols = col_start + jnp.arange(block_dim)
            mask = cols < weights_ref.shape[1]
            safe_cols = jnp.where(mask, cols, 0)
            count = jnp.minimum(count_ref[i_row], indices_ref.shape[1])

            def fn(i_index, temp):
                idx = indices_ref[i_row, i_index]
                valid = (idx >= 0) & (idx < weights_ref.shape[0])
                idx = jnp.where(valid, idx, 0)
                weight_row = jnp.where(mask & valid, weights_ref[idx, safe_cols], 0.0)
                return temp + weight_row

            out = jax.lax.fori_loop(0, count, fn, jnp.zeros([block_dim], dtype=weights_ref.dtype))
            out_ref[i_row, safe_cols] = jnp.where(mask, out, 0.0)

        def run(spikes, indices, count, weights):
            grid = (spikes_info.shape[0], cdiv(weights_info.shape[1], block_dim))
            fn = pl.pallas_call(kernel, grid=grid, out_shape=kwargs['outs'])
            return fn(indices, count, weights)
    else:
        # weights[m,k], select columns -> out[batch, m]
        block_dim = generate_block_dim(weights_info.shape[0], maximum=128)

        def kernel(
            indices_ref,  # [batch, n_spikes]
            count_ref,  # [batch]
            weights_ref,  # [m, k]
            out_ref,  # [batch, m]
        ):
            i_row = pl.program_id(0)
            i_block = pl.program_id(1)
            row_start = i_block * block_dim
            rows = row_start + jnp.arange(block_dim)
            mask = rows < weights_ref.shape[0]
            safe_rows = jnp.where(mask, rows, 0)
            count = jnp.minimum(count_ref[i_row], indices_ref.shape[1])

            def fn(i_index, temp):
                idx = indices_ref[i_row, i_index]
                valid = (idx >= 0) & (idx < weights_ref.shape[1])
                idx = jnp.where(valid, idx, 0)
                weight_col = jnp.where(mask & valid, weights_ref[safe_rows, idx], 0.0)
                return temp + weight_col

            out = jax.lax.fori_loop(0, count, fn, jnp.zeros([block_dim], dtype=weights_ref.dtype))
            out_ref[i_row, safe_rows] = jnp.where(mask, out, 0.0)

        def run(spikes, indices, count, weights):
            grid = (spikes_info.shape[0], cdiv(weights_info.shape[0], block_dim))
            fn = pl.pallas_call(kernel, grid=grid, out_shape=kwargs['outs'])
            return fn(indices, count, weights)

    return run


def _mm_jvp_spikes(spikes_dot, spikes, indices, count, weights, *, transpose, **kwargs):
    if transpose:
        return [jnp.zeros((spikes.shape[0], weights.shape[1]), dtype=weights.dtype)]
    else:
        return [jnp.zeros((spikes.shape[0], weights.shape[0]), dtype=weights.dtype)]


def _mm_jvp_weights(weights_dot, spikes, indices, count, weights, *, transpose, **kwargs):
    return indexed_binary_densemm_p_call(spikes, indices, count, weights_dot, transpose=transpose)


def _mm_transpose(ct, spikes, indices, count, weights, *, transpose, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to indices.")
    if ad.is_undefined_primal(count):
        raise ValueError("Cannot transpose with respect to count.")
    ct = ct[0]
    if ad.is_undefined_primal(spikes):
        return ad.Zero(spikes), indices, count, weights
    if ad.is_undefined_primal(weights):
        if type(ct) is ad.Zero:
            return spikes, indices, count, ad.Zero(weights)
        if transpose:
            # kernel sums rows: ct_weights[indices[b,i]] += ct[b]
            mask = jnp.arange(indices.shape[1])[None, :] < count[:, None]
            updates = jnp.where(mask[:, :, None], ct[:, None, :], 0.0)
            zeros = jnp.zeros(weights.aval.shape, dtype=weights.aval.dtype)
            ct_weights = zeros.at[indices].add(updates)
        else:
            # kernel sums columns: ct_weights[:, indices[b,i]] += ct[b]
            mask = jnp.arange(indices.shape[1])[None, :] < count[:, None]
            updates = jnp.where(mask[:, :, None], ct[:, None, :], 0.0)
            zeros = jnp.zeros(weights.aval.shape, dtype=weights.aval.dtype)
            ct_weights = zeros.at[:, indices].add(updates)
        return spikes, indices, count, ct_weights
    raise ValueError("Cannot transpose with respect to both spikes and weights.")


def _mm_batching(args, axes, *, transpose, **kwargs):
    return general_batching_rule(indexed_binary_densemm_p, args, axes, transpose=transpose, **kwargs)


def _mm_benchmark_data(*, platform):
    batch_size, n_input, n_output = 32, 1000, 1000
    n_spikes = 100
    dtype = jnp.float32
    spikes = jnp.ones((batch_size, n_input), dtype=dtype)
    indices = jnp.asarray(
        np.stack([np.random.choice(n_input, n_spikes, replace=False)
                  for _ in range(batch_size)]).astype(np.int32)
    )
    count = jnp.full((batch_size,), n_spikes, dtype=jnp.int32)
    weights = jnp.asarray(np.random.randn(n_input, n_output), dtype=dtype)
    return [
        BenchmarkConfig("default", (spikes, indices, count, weights)),
    ]


def indexed_binary_densemm_p_call(spikes, indices, count, weights, *, transpose, backend: Optional[str] = None):
    assert spikes.ndim == 2, "spikes should be 2D (batch_size, n_spikes)"
    assert indices.ndim == 2, "indices should be 2D (batch_size, n_spikes)"
    assert count.ndim == 1 and count.shape[0] == spikes.shape[0], "count should be 1D (batch_size,)"
    assert weights.ndim == 2, "weights should be 2D"
    if transpose:
        # weights[k,n], select rows -> out[batch, n]
        assert spikes.shape[1] == weights.shape[0], (
            f"spikes and weights dimension mismatch, "
            f"got {spikes.shape} and {weights.shape}"
        )
        out_shape = jax.ShapeDtypeStruct([spikes.shape[0], weights.shape[1]], weights.dtype)
    else:
        # weights[m,k], select columns -> out[batch, m]
        assert spikes.shape[1] == weights.shape[1], (
            f"spikes and weights dimension mismatch, "
            f"got {spikes.shape} and {weights.shape}"
        )
        out_shape = jax.ShapeDtypeStruct([spikes.shape[0], weights.shape[0]], weights.dtype)
    return indexed_binary_densemm_p(
        spikes,
        indices,
        count,
        weights,
        outs=[out_shape],
        transpose=transpose,
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        count_info=jax.ShapeDtypeStruct(count.shape, count.dtype),
        weights_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        backend=backend,
    )


indexed_binary_densemm_p = XLACustomKernel('indexed_binary_densemm')
indexed_binary_densemm_p.def_numba_kernel(_mm_numba_kernel)
indexed_binary_densemm_p.def_warp_kernel(_mm_warp_kernel)
indexed_binary_densemm_p.def_pallas_kernel('gpu', _mm_pallas_kernel)
indexed_binary_densemm_p.def_pallas_kernel('tpu', _mm_pallas_kernel)
indexed_binary_densemm_p.def_jvp_rule2(_mm_jvp_spikes, None, None, _mm_jvp_weights)
indexed_binary_densemm_p.def_transpose_rule(_mm_transpose)
indexed_binary_densemm_p.def_batching_rule(_mm_batching)
indexed_binary_densemm_p.def_call(indexed_binary_densemm_p_call)
indexed_binary_densemm_p.def_tags('dense', 'indexed_binary')
indexed_binary_densemm_p.def_benchmark_data(_mm_benchmark_data)
