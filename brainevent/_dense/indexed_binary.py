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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import cdiv, generate_block_dim, namescope
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule

__all__ = [
    'indexed_bv_dm',
    'indexed_bv_dm_p',
    'indexed_dm_bv',
    'indexed_dm_bm',
    'indexed_bm_dm',
    'indexed_bm_dm_p',
]


@namescope
def indexed_bv_dm(binary_index, weights):
    """
    Computes the dot product between a binary vector (in sparse format) and a dense matrix.

    The binary vector is represented by `binary_arr`, which contains the spike values,
    their indices, and the count of spikes. The dense matrix is given by `weights`.
    The function multiplies the selected rows of the dense matrix (as indicated by the
    spike indices) and sums them, then applies the unit scaling.

    Parameters
    ----------
    binary_index : IndexedBinary
        An object representing a binary vector in sparse format. It must have the attributes:
        - value: the spike values (typically all ones for binary)
        - spike_indices: indices of nonzero (spike) elements
        - spike_count: number of spikes (nonzero elements)
    weights : ndarray or compatible
        A dense matrix of shape (N, M), where N is the number of possible indices and M is the output dimension.
        Maybe a unit-aware array.

    Returns
    -------
    result : ndarray or compatible
        The result of the dot product, with the same dtype and unit as the input weights.

    Notes
    -----
    This function supports custom CPU and GPU kernels for efficient computation.
    The binary vector is assumed to be sparse, and only the rows of the dense matrix
    corresponding to the spike indices are summed.

    Examples
    --------
    >>> # Suppose binary_arr has spike_indices = [0, 2], spike_count = 2
    >>> # and weights is a (3, 4) matrix
    >>> result = indexed_bv_dm(binary_index, weights)
    """
    weight_val, wunit = u.split_mantissa_unit(weights)
    spikes = binary_index.value
    indices = binary_index.spike_indices
    count = binary_index.spike_count
    r = indexed_bvdm_p_call(spikes, indices, count, weight_val)
    return u.maybe_decimal(r[0] * wunit)


def _binary_vec_dot_dense_mat_numba_kernel(**kwargs):
    import numba

    @numba.njit(fastmath=True)
    def kernel(spikes, indices, count, weights, out):
        out[:] = 0.
        for i in range(count[0]):
            out += weights[indices[i]]

    def run(spikes, indices, count, weights):
        return numba_kernel(kernel, outs=kwargs['outs'])(spikes, indices, count, weights)

    return run


def _binary_vec_dot_dense_mat_warp_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    n = weights_info.shape[1]
    spike_warp_info = jaxinfo_to_warpinfo(spikes_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    count_warp_info = jaxinfo_to_warpinfo(count_info)
    weight_warp_info = jaxinfo_to_warpinfo(weights_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    @warp.kernel
    def kernel(
        spikes: spike_warp_info,
        indices: indices_warp_info,
        count: count_warp_info,
        weights: weight_warp_info,
        out: out_warp_info,
    ):
        j = warp.tid()
        r = weights.dtype(0.)
        for i in range(count[0]):
            r += weights[indices[i], j]
        out[j] = r

    def run(spikes, indices, count, weights):
        out_info = kwargs['outs'][0]
        fn = jax_kernel(kernel, launch_dims=[n], num_outputs=1, output_dims={'out': out_info.shape})
        return fn(spikes, indices, count, weights)

    return run


def _binary_vec_dot_dense_mat_pallas_kernel(
    weights_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    block_dim = generate_block_dim(weights_info.shape[1], maximum=128)

    def kernel(
        spikes_ref,  # [n_neuron]
        indices_ref,  # [n_neuron]
        count_ref,  # [1]
        weights_ref,  # [n_neuron, n_output]
        out_ref,  # [n_output]
    ):
        i_block = pl.program_id(0)
        col_start = i_block * block_dim
        cols = col_start + jnp.arange(block_dim)
        mask = cols < weights_ref.shape[1]
        safe_cols = jnp.where(mask, cols, 0)

        def fn(i, temp):
            i_row = indices_ref[i]
            weight_row = plt.load(weights_ref[i_row, safe_cols])
            weight_row = jnp.where(mask, weight_row, 0.0)
            return temp + weight_row

        out = jax.lax.fori_loop(
            0,
            count_ref[0],
            fn,
            jnp.zeros([block_dim], dtype=weights_ref.dtype)
        )
        plt.store(out_ref[safe_cols], out, mask=mask)

    def run(spikes, indices, count, weights):
        fn = pl.pallas_call(kernel, grid=(cdiv(weights_info.shape[1], block_dim),), out_shape=kwargs['outs'])
        return fn(spikes, indices, count, weights)

    return run


def _binary_vec_dot_dense_mat_jvp_spikes(spikes_dot, spikes, indices, count, weights, **kwargs):
    return [jnp.zeros((weights.shape[1],), dtype=weights.dtype)]


def _binary_vec_dot_dense_mat_jvp_weights(weights_dot, spikes, indices, count, weights, **kwargs):
    return indexed_bvdm_p_call(spikes, indices, count, weights_dot)


def _binary_vec_dot_dense_mat_transpose(ct, spikes, indices, count, weights, **kwargs):
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
        updates = jnp.where(mask[:, None], ct, 0.0)
        zeros = jnp.zeros(weights.aval.shape, dtype=weights.aval.dtype)
        ct_weights = zeros.at[indices].add(updates)
        return spikes, indices, count, ct_weights
    raise ValueError("Cannot transpose with respect to both spikes and weights.")


def _binary_vec_dot_dense_mat_batching(args, axes, **kwargs):
    if axes == (None, None, None, 0):
        spikes, indices, count, weights = args
        mask = jnp.arange(indices.shape[0]) < count[0]
        gathered = jnp.take(weights, indices, axis=1)
        updates = jnp.where(mask[None, :, None], gathered, 0.0)
        r = updates.sum(axis=1)
        return [r], [0]
    return general_batching_rule(indexed_bv_dm_p, args, axes, **kwargs)


def indexed_bvdm_p_call(spikes, indices, count, weights):
    assert spikes.ndim == 1, "spikes should be 1D (n_spikes,)"
    assert indices.ndim == 1, "indices should be 1D (n_spikes,)"
    assert count.ndim == 1 and count.shape[0] == 1, "count should be 1D (1,)"
    assert weights.ndim == 2, "weights should be 2D (n_input, n_output)"
    assert spikes.shape[0] == weights.shape[0], (
        f"spikes and weights dimension mismatch, "
        f"got {spikes.shape} and {weights.shape}"
    )
    return indexed_bv_dm_p(
        spikes,
        indices,
        count,
        weights,
        outs=[jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        count_info=jax.ShapeDtypeStruct(count.shape, count.dtype),
        weights_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


indexed_bv_dm_p = XLACustomKernel('binary_vec_dot_dense_matrix')
indexed_bv_dm_p.def_numba_kernel(_binary_vec_dot_dense_mat_numba_kernel)
indexed_bv_dm_p.def_warp_kernel(_binary_vec_dot_dense_mat_warp_kernel)
indexed_bv_dm_p.def_pallas_kernel('gpu', _binary_vec_dot_dense_mat_pallas_kernel)
indexed_bv_dm_p.def_pallas_kernel('tpu', _binary_vec_dot_dense_mat_pallas_kernel)
indexed_bv_dm_p.def_jvp_rule2(_binary_vec_dot_dense_mat_jvp_spikes, None, None, _binary_vec_dot_dense_mat_jvp_weights)
indexed_bv_dm_p.def_transpose_rule(_binary_vec_dot_dense_mat_transpose)
indexed_bv_dm_p.def_batching_rule(_binary_vec_dot_dense_mat_batching)
indexed_bv_dm_p.def_call(indexed_bvdm_p_call)
indexed_bv_dm_p.def_tags('dense', 'indexed_binary')


def _indexed_bv_dm_benchmark_data(*, platform):
    import numpy as _np
    n_input, n_output = 1000, 1000
    n_spikes = 100
    dtype = jnp.float32
    spikes = jnp.ones(n_input, dtype=dtype)
    indices = jnp.asarray(_np.random.choice(n_input, n_spikes, replace=False).astype(_np.int32))
    count = jnp.asarray([n_spikes], dtype=jnp.int32)
    weights = jnp.asarray(_np.random.randn(n_input, n_output), dtype=dtype)
    return [
        ("default", (spikes, indices, count, weights), {}),
    ]


indexed_bv_dm_p.def_benchmark_data(_indexed_bv_dm_benchmark_data)


@namescope
def indexed_dm_bv(weights, binary_arr):
    """
    Computes the dot product between a dense matrix and a binary vector (in sparse format).

    The binary vector is represented by `binary_arr`, which contains the spike values,
    their indices, and the count of spikes. The dense matrix is given by `weights`.
    The function multiplies the selected columns of the dense matrix (as indicated by the
    spike indices) and sums them, then applies the unit scaling.

    Parameters
    ----------
    weights : ndarray or compatible
        A dense matrix of shape (N, M), where N is the input dimension and M is the output dimension.
        May be a unit-aware array.
    binary_arr : IndexedBinary
        An object representing a binary vector in sparse format. It must have the attributes:
        - value: the spike values (typically all ones for binary)
        - spike_indices: indices of nonzero (spike) elements
        - spike_count: number of spikes (nonzero elements)

    Returns
    -------
    result : ndarray or compatible
        The result of the dot product, with the same dtype and unit as the input weights.

    Notes
    -----
    This function is designed to support custom CPU and GPU kernels for efficient computation.
    The binary vector is assumed to be sparse, and only the columns of the dense matrix
    corresponding to the spike indices are summed.

    Examples
    --------
    >>> # Suppose binary_arr has spike_indices = [1, 3], spike_count = 2
    >>> # and weights is a (5, 4) matrix
    >>> result = indexed_dm_bv(weights, binary_arr)
    """
    return indexed_bv_dm(binary_arr, weights.T)


@namescope
def indexed_bm_dm(binary_arr, weights):
    """
    Computes the dot product between a batch of binary vectors (in sparse format) and a dense matrix.

    Each binary vector in the batch is represented by `binary_arr`, which contains the spike values,
    their indices, and the count of spikes for each vector. The dense matrix is given by `weights`.
    The function multiplies the selected rows of the dense matrix (as indicated by the spike indices
    for each vector in the batch) and sums them, then applies the unit scaling.

    Parameters
    ----------
    binary_arr : IndexedBinary
        An object representing a batch of binary vectors in sparse format. It must have the attributes:
        - value: the spike values (typically all ones for binary), shape (batch_size, n_spikes)
        - spike_indices: indices of nonzero (spike) elements, shape (batch_size, n_spikes)
        - spike_count: number of spikes (nonzero elements) for each vector, shape (batch_size,)
    weights : ndarray or compatible
        A dense matrix of shape (N, M), where N is the input dimension and M is the output dimension.
        May be a unit-aware array.

    Returns
    -------
    result : ndarray or compatible
        The result of the dot product for each vector in the batch, with shape (batch_size, M)
        and the same dtype and unit as the input weights.

    Notes
    -----
    This function is designed to support custom CPU and GPU kernels for efficient computation.
    The binary vectors are assumed to be sparse, and only the rows of the dense matrix
    corresponding to the spike indices are summed for each vector in the batch.

    Examples
    --------
    >>> # Suppose binary_arr has spike_indices = [[0, 2], [1, 3]], spike_count = [2, 2]
    >>> # and weights is a (4, 5) matrix
    >>> result = indexed_bm_dm(binary_arr, weights)
    """
    weights, wunit = u.split_mantissa_unit(weights)
    spikes = binary_arr.value
    indices = binary_arr.spike_indices
    count = binary_arr.spike_count
    r = indexed_bmdm_p_call(spikes, indices, count, weights)
    return u.maybe_decimal(r[0] * wunit)


def _binary_mat_dot_dense_mat_numba_kernel(**kwargs):
    import numba

    @numba.njit(parallel=True, fastmath=True, nogil=True)
    def kernel(spikes, indices, count, weights, out):
        for i_row in numba.prange(indices.shape[0]):
            temp = np.zeros(weights.shape[1], dtype=weights.dtype)
            for i_col in range(count[i_row]):
                temp += weights[indices[i_row, i_col]]
            out[i_row] = temp

    def run(spikes, indices, count, weights):
        return numba_kernel(kernel, outs=kwargs['outs'])(spikes, indices, count, weights)

    return run


def _binary_mat_dot_dense_mat_warp_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    batch = indices_info.shape[0]
    n_out = weights_info.shape[1]
    spike_warp_info = jaxinfo_to_warpinfo(spikes_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    count_warp_info = jaxinfo_to_warpinfo(count_info)
    weight_warp_info = jaxinfo_to_warpinfo(weights_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    @warp.kernel
    def kernel(
        spikes: spike_warp_info,
        indices: indices_warp_info,
        count: count_warp_info,
        weights: weight_warp_info,
        out: out_warp_info,
    ):
        i_row = warp.tid()
        for j in range(n_out):
            r = weights.dtype(0.)
            for k in range(count[i_row]):
                r += weights[indices[i_row, k], j]
            out[i_row, j] = r

    def run(spikes, indices, count, weights):
        out_info = kwargs['outs'][0]
        fn = jax_kernel(kernel, launch_dims=[batch], num_outputs=1, output_dims={'out': out_info.shape})
        return fn(spikes, indices, count, weights)

    return run


def _binary_mat_dot_dense_mat_pallas_kernel(
    spikes_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt

    block_dim = generate_block_dim(weights_info.shape[1], maximum=128)

    def kernel(
        spikes_ref,  # [batch, n_spikes]
        indices_ref,  # [batch, n_spikes]
        count_ref,  # [batch]
        weights_ref,  # [n_input, n_output]
        out_ref,  # [batch, n_output]
    ):
        i_row = pl.program_id(0)
        i_block = pl.program_id(1)
        col_start = i_block * block_dim
        cols = col_start + jnp.arange(block_dim)
        mask = cols < weights_ref.shape[1]
        safe_cols = jnp.where(mask, cols, 0)
        count = count_ref[i_row]

        def fn(i_index, temp):
            idx = indices_ref[i_row, i_index]
            weight_row = plt.load(weights_ref[idx, safe_cols])
            weight_row = jnp.where(mask, weight_row, 0.0)
            return temp + weight_row

        out = jax.lax.fori_loop(
            0,
            count,
            fn,
            jnp.zeros([block_dim], dtype=weights_ref.dtype)
        )
        plt.store(out_ref[i_row, safe_cols], out, mask=mask)

    def run(spikes, indices, count, weights):
        fn = pl.pallas_call(
            kernel,
            grid=(spikes_info.shape[0], cdiv(weights_info.shape[1], block_dim)),
            out_shape=kwargs['outs'],
        )
        return fn(spikes, indices, count, weights)

    return run


def _binary_mat_dot_dense_mat_jvp_spikes(spikes_dot, spikes, indices, count, weights, **kwargs):
    return [jnp.zeros((spikes.shape[0], weights.shape[1]), dtype=weights.dtype)]


def _binary_mat_dot_dense_mat_jvp_weights(weights_dot, spikes, indices, count, weights, **kwargs):
    return indexed_bmdm_p_call(spikes, indices, count, weights_dot)


def _binary_mat_dot_dense_mat_transpose(ct, spikes, indices, count, weights, **kwargs):
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
        mask = jnp.arange(indices.shape[1])[None, :] < count[:, None]
        updates = jnp.where(mask[:, :, None], ct[:, None, :], 0.0)
        zeros = jnp.zeros(weights.aval.shape, dtype=weights.aval.dtype)
        ct_weights = zeros.at[indices].add(updates)
        return spikes, indices, count, ct_weights
    raise ValueError("Cannot transpose with respect to both spikes and weights.")


def _binary_mat_dot_dense_mat_batching(args, axes, **kwargs):
    return general_batching_rule(indexed_bm_dm_p, args, axes, **kwargs)


def indexed_bmdm_p_call(spikes, indices, count, weights):
    assert spikes.ndim == 2, "spikes should be 2D (batch_size, n_spikes)"
    assert indices.ndim == 2, "indices should be 2D (batch_size, n_spikes)"
    assert count.ndim == 1 and count.shape[0] == spikes.shape[0], "count should be 1D (batch_size,)"
    assert weights.ndim == 2, "weights should be 2D (n_input, n_output)"
    assert spikes.shape[1] == weights.shape[0], (f"spikes and weights dimension mismatch, "
                                                 f"got {spikes.shape} and {weights.shape}")
    return indexed_bm_dm_p(
        spikes,
        indices,
        count,
        weights,
        outs=[jax.ShapeDtypeStruct([spikes.shape[0], weights.shape[1]], weights.dtype)],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        count_info=jax.ShapeDtypeStruct(count.shape, count.dtype),
        weights_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


indexed_bm_dm_p = XLACustomKernel('binary_mat_dot_dense_matrix')
indexed_bm_dm_p.def_numba_kernel(_binary_mat_dot_dense_mat_numba_kernel)
indexed_bm_dm_p.def_warp_kernel(_binary_mat_dot_dense_mat_warp_kernel)
indexed_bm_dm_p.def_pallas_kernel('gpu', _binary_mat_dot_dense_mat_pallas_kernel)
indexed_bm_dm_p.def_pallas_kernel('tpu', _binary_mat_dot_dense_mat_pallas_kernel)
indexed_bm_dm_p.def_jvp_rule2(_binary_mat_dot_dense_mat_jvp_spikes, None, None, _binary_mat_dot_dense_mat_jvp_weights)
indexed_bm_dm_p.def_transpose_rule(_binary_mat_dot_dense_mat_transpose)
indexed_bm_dm_p.def_batching_rule(_binary_mat_dot_dense_mat_batching)
indexed_bm_dm_p.def_call(indexed_bmdm_p_call)
indexed_bm_dm_p.def_tags('dense', 'indexed_binary')


def _indexed_bm_dm_benchmark_data(*, platform):
    import numpy as _np
    batch_size, n_input, n_output = 32, 1000, 1000
    n_spikes = 100
    dtype = jnp.float32
    spikes = jnp.ones((batch_size, n_input), dtype=dtype)
    indices = jnp.asarray(
        _np.stack([_np.random.choice(n_input, n_spikes, replace=False) for _ in range(batch_size)]).astype(_np.int32)
    )
    count = jnp.full((batch_size,), n_spikes, dtype=jnp.int32)
    weights = jnp.asarray(_np.random.randn(n_input, n_output), dtype=dtype)
    return [
        ("default", (spikes, indices, count, weights), {}),
    ]


indexed_bm_dm_p.def_benchmark_data(_indexed_bm_dm_benchmark_data)


@namescope
def indexed_dm_bm(weights, binary_arr):
    weight_val, wunit = u.split_mantissa_unit(weights)
    spikes = binary_arr.value
    indices = binary_arr.spike_indices
    count = binary_arr.spike_count
    return u.maybe_decimal(
        indexed_bmdm_p_call(spikes, indices, count, weight_val) * wunit
    )
