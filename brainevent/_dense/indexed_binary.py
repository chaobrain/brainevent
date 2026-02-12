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

from brainevent._misc import cdiv, generate_block_dim, namescope
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent.config import get_numba_parallel

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
    Perform indexed binary dense matrix-vector multiplication.

    Accumulates rows or columns of a dense weight matrix selected by
    sparse spike indices. This is an event-driven operation where only
    the weight entries corresponding to active spikes contribute to the
    output.

    When ``transpose=False``, sums the columns of ``weights`` at the
    positions given by ``binary_index.spike_indices``, producing a vector
    of length ``m``.

    When ``transpose=True``, sums the rows of ``weights`` at the
    positions given by ``binary_index.spike_indices``, producing a vector
    of length ``n``.

    Parameters
    ----------
    weights : array_like
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``. Can be a ``brainunit``
        quantity.
    binary_index : BinaryArray
        An object representing a binary vector in sparse format with
        attributes:

        - ``value`` : array of shape ``(k,)`` with spike values.
        - ``spike_indices`` : 1-D integer array of active spike indices.
        - ``spike_count`` : 1-D array of shape ``(1,)`` giving the number
          of valid entries in ``spike_indices``.
    transpose : bool
        If False, accumulate selected columns of ``weights``.
        If True, accumulate selected rows of ``weights``.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``, ``'warp'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : array_like
        Result vector. Shape ``(m,)`` when ``transpose=False``,
        or ``(n,)`` when ``transpose=True``. Carries the unit of
        ``weights`` if applicable.

    Raises
    ------
    AssertionError
        If the spike dimension does not match the corresponding weight
        dimension.

    See Also
    --------
    indexed_binary_densemm : Batched (matrix-matrix) variant.
    indexed_binary_densemv_p_call : Low-level primitive call.

    Notes
    -----
    Only the first ``spike_count`` entries of ``spike_indices`` are used.
    Indices outside the valid range of the weight matrix dimension are
    silently ignored.

    When ``transpose=False``, the operation computes:

    ``out[i] = sum_{p=0}^{count-1} W[i, indices[p]]``

    where ``indices`` is the array of active spike positions and ``count``
    is the number of valid entries.

    When ``transpose=True``, the operation computes:

    ``out[j] = sum_{p=0}^{count-1} W[indices[p], j]``

    This indexed form avoids iterating over the full spike vector and is
    efficient when the number of active spikes is much smaller than the
    total spike dimension ``k``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._dense.indexed_binary import indexed_binary_densemv
        >>> weights = jnp.ones((4, 3), dtype=jnp.float32)
        >>> # Assuming binary_index has spike_indices=[0, 2], spike_count=[2]
        >>> # transpose=False sums columns 0 and 2 of weights -> shape (4,)
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
            fn = pl.pallas_call(kernel, grid=(cdiv(weights_info.shape[1], block_dim),), out_shape=kwargs['outs'],
                                backend='triton')
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
            fn = pl.pallas_call(kernel, grid=(cdiv(weights_info.shape[0], block_dim),), out_shape=kwargs['outs'],
                                backend='triton')
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
    """
    Low-level primitive call for indexed binary dense matrix-vector multiplication.

    This function validates input shapes, constructs the output shape
    descriptor, and invokes the ``indexed_binary_densemv_p`` JAX primitive.
    Unlike :func:`indexed_binary_densemv`, this function operates on raw
    numerical arrays without ``brainunit`` unit handling and accepts the
    sparse index components directly rather than a ``BinaryArray`` object.

    Parameters
    ----------
    spikes : jax.Array
        Spike values array with shape ``(k,)``.
    indices : jax.Array
        Integer array of active spike indices with shape ``(n_spikes,)``.
    count : jax.Array
        Integer array of shape ``(1,)`` indicating the number of valid
        entries in ``indices``.
    weights : jax.Array
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``.
    transpose : bool
        If False, accumulate selected columns of ``weights`` producing
        shape ``(m,)``. If True, accumulate selected rows of ``weights``
        producing shape ``(n,)``.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``, ``'warp'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : list of jax.Array
        A single-element list containing the result vector.

    Raises
    ------
    AssertionError
        If ``spikes`` is not 1-D, ``indices`` is not 1-D, ``count`` does
        not have shape ``(1,)``, ``weights`` is not 2-D, or the spike
        dimension does not match the corresponding weight dimension.

    See Also
    --------
    indexed_binary_densemv : High-level function with unit handling.

    Notes
    -----
    This is the low-level entry point that bypasses unit handling and
    accepts the sparse index components directly. The mathematical
    operation is identical to :func:`indexed_binary_densemv`:

    When ``transpose=False``:

    ``out[i] = sum_{p=0}^{count-1} weights[i, indices[p]]``

    When ``transpose=True``:

    ``out[j] = sum_{p=0}^{count-1} weights[indices[p], j]``

    The function returns a single-element list to conform to the JAX
    primitive output convention.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._dense.indexed_binary import indexed_binary_densemv_p_call
        >>> spikes = jnp.ones(5, dtype=jnp.float32)
        >>> indices = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> count = jnp.array([3], dtype=jnp.int32)
        >>> weights = jnp.ones((5, 3), dtype=jnp.float32)
        >>> indexed_binary_densemv_p_call(spikes, indices, count, weights, transpose=True)
    """
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
"""
Low-level XLA custom-kernel primitive for ``indexed_binary_densemv``.

This ``XLACustomKernel`` instance dispatches the ``indexed_binary_densemv`` operation
to the backend registered below (for example ``numba``, ``warp``, and
``pallas``), using runtime shape/dtype metadata provided by the high-level
wrapper.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation
integrates correctly with ``jit``, ``vmap``, and autodiff.
"""
indexed_binary_densemv_p.def_numba_kernel(_mv_numba_kernel)
indexed_binary_densemv_p.def_warp_kernel(_mv_warp_kernel)
indexed_binary_densemv_p.def_pallas_kernel('gpu', _mv_pallas_kernel)
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
    Perform batched indexed binary dense matrix-matrix multiplication.

    For each sample in the batch, accumulates rows or columns of a dense
    weight matrix selected by that sample's sparse spike indices.

    When ``transpose=False``, for each batch element ``b``, sums the
    columns of ``weights`` at positions given by
    ``binary_arr.spike_indices[b]``, producing an output row of length
    ``m``.

    When ``transpose=True``, for each batch element ``b``, sums the rows
    of ``weights`` at positions given by
    ``binary_arr.spike_indices[b]``, producing an output row of length
    ``n``.

    Parameters
    ----------
    weights : array_like
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``. Can be a ``brainunit``
        quantity.
    binary_arr : BinaryArray
        An object representing a batch of binary vectors in sparse format
        with attributes:

        - ``value`` : array of shape ``(batch, k)`` with spike values.
        - ``spike_indices`` : integer array of shape ``(batch, n_spikes)``
          with active spike indices per sample.
        - ``spike_count`` : integer array of shape ``(batch,)`` giving the
          number of valid entries per sample.
    transpose : bool
        If False, accumulate selected columns of ``weights``.
        If True, accumulate selected rows of ``weights``.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``, ``'warp'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : array_like
        Result matrix. Shape ``(batch, m)`` when ``transpose=False``,
        or ``(batch, n)`` when ``transpose=True``. Carries the unit of
        ``weights`` if applicable.

    Raises
    ------
    AssertionError
        If the spike dimension does not match the corresponding weight
        dimension.

    See Also
    --------
    indexed_binary_densemv : Unbatched (matrix-vector) variant.
    indexed_binary_densemm_p_call : Low-level primitive call.

    Notes
    -----
    For each batch element, only the first ``spike_count[b]`` entries of
    ``spike_indices[b]`` are used. Indices outside the valid range of the
    weight matrix dimension are silently ignored.

    When ``transpose=False``, the operation computes for each batch
    element ``b``:

    ``out[b, i] = sum_{p=0}^{count[b]-1} W[i, indices[b, p]]``

    When ``transpose=True``, the operation computes for each batch
    element ``b``:

    ``out[b, j] = sum_{p=0}^{count[b]-1} W[indices[b, p], j]``

    This indexed form avoids iterating over the full spike vector and is
    efficient when the number of active spikes per sample is much smaller
    than the total spike dimension ``k``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._dense.indexed_binary import indexed_binary_densemm
        >>> weights = jnp.ones((4, 3), dtype=jnp.float32)
        >>> # Assuming binary_arr has spike_indices=[[0,2],[1,2]], spike_count=[2,2]
        >>> # transpose=False sums columns for each batch -> shape (2, 4)
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
            fn = pl.pallas_call(kernel, grid=grid, out_shape=kwargs['outs'], backend='triton')
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
            fn = pl.pallas_call(kernel, grid=grid, out_shape=kwargs['outs'], backend='triton')
            return fn(indices, count, weights)

    return run


def _mm_jvp_spikes(spikes_dot, spikes, indices, count, weights, *, transpose, **kwargs):
    if transpose:
        return [jnp.zeros((spikes.shape[0], weights.shape[1]), dtype=weights.dtype)]
    else:
        return [jnp.zeros((spikes.shape[0], weights.shape[0]), dtype=weights.dtype)]


def _mm_jvp_weights(weights_dot, spikes, indices, count, weights, *, transpose, **kwargs):
    return indexed_binary_densemm_p_call(spikes, indices, count, weights_dot,
                                         transpose=transpose, backend=kwargs['backend'])


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
    """
    Low-level primitive call for batched indexed binary dense matrix-matrix multiplication.

    This function validates input shapes, constructs the output shape
    descriptor, and invokes the ``indexed_binary_densemm_p`` JAX primitive.
    Unlike :func:`indexed_binary_densemm`, this function operates on raw
    numerical arrays without ``brainunit`` unit handling and accepts the
    sparse index components directly rather than a ``BinaryArray`` object.

    Parameters
    ----------
    spikes : jax.Array
        Spike values array with shape ``(batch, k)``.
    indices : jax.Array
        Integer array of active spike indices with shape
        ``(batch, n_spikes)``.
    count : jax.Array
        Integer array of shape ``(batch,)`` indicating the number of valid
        entries in ``indices`` for each batch element.
    weights : jax.Array
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``.
    transpose : bool
        If False, accumulate selected columns of ``weights`` producing
        shape ``(batch, m)``. If True, accumulate selected rows of
        ``weights`` producing shape ``(batch, n)``.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``, ``'warp'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : list of jax.Array
        A single-element list containing the result matrix with shape
        ``(batch, m)`` or ``(batch, n)``.

    Raises
    ------
    AssertionError
        If ``spikes`` is not 2-D, ``indices`` is not 2-D, ``count`` shape
        does not match the batch size, ``weights`` is not 2-D, or the spike
        dimension does not match the corresponding weight dimension.

    See Also
    --------
    indexed_binary_densemm : High-level function with unit handling.

    Notes
    -----
    This is the low-level entry point that bypasses unit handling and
    accepts the sparse index components directly. The mathematical
    operation is identical to :func:`indexed_binary_densemm`:

    When ``transpose=False`` for each batch element ``b``:

    ``out[b, i] = sum_{p=0}^{count[b]-1} weights[i, indices[b, p]]``

    When ``transpose=True`` for each batch element ``b``:

    ``out[b, j] = sum_{p=0}^{count[b]-1} weights[indices[b, p], j]``

    The function returns a single-element list to conform to the JAX
    primitive output convention.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._dense.indexed_binary import indexed_binary_densemm_p_call
        >>> spikes = jnp.ones((2, 5), dtype=jnp.float32)
        >>> indices = jnp.array([[0, 2, 4, 0, 0],
        ...                      [1, 3, 0, 0, 0]], dtype=jnp.int32)
        >>> count = jnp.array([3, 2], dtype=jnp.int32)
        >>> weights = jnp.ones((5, 3), dtype=jnp.float32)
        >>> indexed_binary_densemm_p_call(spikes, indices, count, weights, transpose=True)
    """
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
indexed_binary_densemm_p.def_jvp_rule2(_mm_jvp_spikes, None, None, _mm_jvp_weights)
indexed_binary_densemm_p.def_transpose_rule(_mm_transpose)
indexed_binary_densemm_p.def_batching_rule(_mm_batching)
indexed_binary_densemm_p.def_call(indexed_binary_densemm_p_call)
indexed_binary_densemm_p.def_tags('dense', 'indexed_binary')
indexed_binary_densemm_p.def_benchmark_data(_mm_benchmark_data)
indexed_binary_densemm_p.__doc__ = """
Low-level XLA custom-kernel primitive for ``indexed_binary_densemm``.

This ``XLACustomKernel`` instance dispatches the ``indexed_binary_densemm`` operation
to the backend registered below (for example ``numba``, ``warp``, and
``pallas``), using runtime shape/dtype metadata provided by the high-level
wrapper.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation
integrates correctly with ``jit``, ``vmap``, and autodiff.
"""
