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

from brainevent._compatible_import import pallas as pl
from brainevent._misc import cdiv, generate_block_dim
from brainevent._op.main import XLACustomKernel
from brainevent._op.op_numba import numba_kernel
from brainevent._op.op_pallas import pallas_kernel
from brainevent._op.op_warp import warp_kernel, jaxinfo_to_warpinfo, jaxtype_to_warptype


def binary_vec_dot_dense_mat(binary_index, weights):
    """
    Computes the dot product between a binary vector (in sparse format) and a dense matrix.

    The binary vector is represented by `binary_arr`, which contains the spike values,
    their indices, and the count of spikes. The dense matrix is given by `weights`.
    The function multiplies the selected rows of the dense matrix (as indicated by the
    spike indices) and sums them, then applies the unit scaling.

    Parameters
    ----------
    binary_index : BinaryArrayIndex
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
    >>> result = binary_vec_dot_dense_mat(binary_index, weights)
    """
    weight_val, wunit = u.split_mantissa_unit(weights)
    spikes = binary_index.value
    indices = binary_index.spike_indices
    count = binary_index.spike_count
    r = _binary_vec_dot_dense_mat_p_call(spikes, indices, count, weight_val)
    return u.maybe_decimal(r[0] * wunit)


def _binary_vec_dot_dense_mat_numba_kernel_generator(
    **kwargs
):
    def _kernel(spikes, indices, count, weights, out):
        out[:] = 0.
        for i in range(count[0]):
            out += weights[indices[i]]

    return numba_kernel(_kernel)


def _binary_vec_dot_dense_mat_warp_kernel_generator(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel
    block_dim = generate_block_dim(weights_info.shape[1], maximum=128)
    weight_dtype = jaxtype_to_warptype(weights_info.dtype)

    def kernel(
        spikes: jaxinfo_to_warpinfo(spikes_info),
        indices: jaxinfo_to_warpinfo(indices_info),
        count: jaxinfo_to_warpinfo(count_info),
        weights: jaxinfo_to_warpinfo(weights_info),
        out: warp.array1d(dtype=weight_dtype),
    ):
        i_col_block = warp.tid()
        temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
        for j in range(count[0]):
            temp += warp.tile_load(weights[indices[j]], shape=(block_dim,), offset=(i_col_block * block_dim,))
        warp.tile_store(out, temp, offset=(i_col_block * block_dim,))

    return warp_kernel(
        kernel,
        tile=(cdiv(weights_info.shape[1], block_dim),),
        block_dim=block_dim,
    )


def _binary_vec_dot_dense_mat_pallas_kernel_generator(
    weights_info: jax.ShapeDtypeStruct,
    **kwargs
):
    block_dim = generate_block_dim(weights_info.shape[1], maximum=128)

    def kernel(
        spikes_ref,  # [n_neuron]
        indices_ref,  # [n_neuron]
        count_ref,  # [1]
        weights_ref,  # [n_neuron, n_output]
        o_ref,  # [n_output]
    ):
        i_block = pl.program_id(0)
        i_start = i_block * block_dim
        mask = i_start + jnp.arange(block_dim) < weights_ref.shape[1]

        def fn(i, temp):
            i_col = indices_ref[i]
            temp += pl.load(weights_ref, (i_col, pl.dslice(i_start, block_dim)), mask=mask)
            return temp

        out = jax.lax.fori_loop(0, count_ref[0], fn, jnp.zeros([block_dim], dtype=weights_ref.dtype))
        pl.store(o_ref, pl.dslice(i_start, block_dim), out, mask=mask)

    return pallas_kernel(kernel, outs=kwargs['outs'], tile=(cdiv(weights_info.shape[1], block_dim),))


def _binary_vec_dot_dense_mat_jvp_spikes(spikes_dot, spikes, indices, count, weights, **kwargs):
    return [spikes_dot @ weights]


def _binary_vec_dot_dense_mat_jvp_weights(weights_dot, spikes, indices, count, weights, **kwargs):
    return _binary_vec_dot_dense_mat_p_call(spikes, indices, count, weights_dot)


def _binary_vec_dot_dense_mat_transpose(ct, spikes, indices, count, weights, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to indices.")
    if ad.is_undefined_primal(count):
        raise ValueError("Cannot transpose with respect to count.")
    if ad.is_undefined_primal(spikes):
        return weights @ ct[0], indices, count, weights
    elif ad.is_undefined_primal(weights):
        return spikes, indices, count, jnp.outer(spikes, ct[0])
    else:
        raise ValueError("Cannot transpose with respect to both spikes and weights.")


def _binary_vec_dot_dense_mat_batching():
    pass


def _binary_vec_dot_dense_mat_p_call(spikes, indices, count, weights):
    assert spikes.ndim == 1, "spikes should be 1D (n_spikes,)"
    assert indices.ndim == 1, "indices should be 1D (n_spikes,)"
    assert count.ndim == 1 and count.shape[0] == 1, "count should be 1D (1,)"
    assert weights.ndim == 2, "weights should be 2D (n_input, n_output)"
    assert spikes.shape[0] == weights.shape[0], (f"spikes and weights dimension mismatch, "
                                                 f"got {spikes.shape} and {weights.shape}")
    return _binary_vec_dot_dense_mat_p(
        spikes,
        indices,
        count,
        weights,
        outs=[
            jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
        ],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        count_info=jax.ShapeDtypeStruct(count.shape, count.dtype),
        weights_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


_binary_vec_dot_dense_mat_p = XLACustomKernel('binary_vec_dot_dense_matrix')
_binary_vec_dot_dense_mat_p.def_cpu_kernel(_binary_vec_dot_dense_mat_numba_kernel_generator)
_binary_vec_dot_dense_mat_p.def_gpu_kernel(
    warp=_binary_vec_dot_dense_mat_warp_kernel_generator,
    pallas=_binary_vec_dot_dense_mat_pallas_kernel_generator,
    default='warp'
)
_binary_vec_dot_dense_mat_p.def_tpu_kernel(_binary_vec_dot_dense_mat_pallas_kernel_generator)
_binary_vec_dot_dense_mat_p.def_jvp_rule2(
    _binary_vec_dot_dense_mat_jvp_spikes,
    None, None,
    _binary_vec_dot_dense_mat_jvp_weights,
)
_binary_vec_dot_dense_mat_p.def_transpose_rule(_binary_vec_dot_dense_mat_transpose)


def dense_mat_dot_binary_vec(weights, binary_arr):
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
    binary_arr : BinaryArrayIndex
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
    >>> result = dense_mat_dot_binary_vec(weights, binary_arr)
    """
    return binary_vec_dot_dense_mat(binary_arr, weights.T)


def binary_mat_dot_dense_mat(binary_arr, weights):
    """
    Computes the dot product between a batch of binary vectors (in sparse format) and a dense matrix.

    Each binary vector in the batch is represented by `binary_arr`, which contains the spike values,
    their indices, and the count of spikes for each vector. The dense matrix is given by `weights`.
    The function multiplies the selected rows of the dense matrix (as indicated by the spike indices
    for each vector in the batch) and sums them, then applies the unit scaling.

    Parameters
    ----------
    binary_arr : BinaryArrayIndex
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
    >>> result = binary_mat_dot_dense_mat(binary_arr, weights)
    """
    weights, wunit = u.split_mantissa_unit(weights)
    spikes = binary_arr.value
    indices = binary_arr.spike_indices
    count = binary_arr.spike_count
    r = _binary_mat_dot_dense_mat_p_call(spikes, indices, count, weights)
    return u.maybe_decimal(r[0] * wunit)


def _binary_mat_dot_dense_mat_numba_kernel_generator(
    **kwargs
):
    def _kernel(spikes, indices, count, weights, out):
        for i_row in range(indices.shape[0]):
            temp = np.zeros(weights.shape[1], dtype=weights.dtype)
            for i_col in range(count[i_row]):
                temp += weights[indices[i_row, i_col]]
            out[i_row] = temp

    return numba_kernel(_kernel, parallel=True)


def _binary_mat_dot_dense_mat_warp_kernel_generator(
    spikes_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    count_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp  # pylint: disable=import-outside-toplevel
    block_dim = generate_block_dim(weights_info.shape[1], maximum=128)
    weight_dtype = jaxtype_to_warptype(weights_info.dtype)

    def kernel(
        spikes: jaxinfo_to_warpinfo(spikes_info),
        indices: jaxinfo_to_warpinfo(indices_info),
        count: jaxinfo_to_warpinfo(count_info),
        weights: jaxinfo_to_warpinfo(weights_info),
        out: warp.array2d(dtype=weight_dtype),
    ):
        i_row_block, i_col_block = warp.tid()
        temp = warp.tile_zeros(shape=(block_dim,), dtype=weight_dtype)
        n_count = count[i_row_block]
        for j in range(n_count):
            temp += warp.tile_load(
                weights[indices[i_row_block, j]],
                shape=(block_dim,),
                offset=(i_col_block * block_dim,)
            )
        warp.tile_store(out[i_row_block], temp, offset=(i_col_block * block_dim,))

    return warp_kernel(
        kernel,
        tile=(indices_info.shape[0], cdiv(weights_info.shape[1], block_dim),),
        block_dim=block_dim,
    )


def _binary_mat_dot_dense_mat_pallas_kernel_generator(
    spikes_info: jax.ShapeDtypeStruct,
    weights_info: jax.ShapeDtypeStruct,
    **kwargs
):
    block_dim = generate_block_dim(weights_info.shape[1], maximum=128)

    def kernel(
        spikes_ref,  # [n_neuron]
        indices_ref,  # [n_neuron]
        count_ref,  # [1]
        weights_ref,  # [n_neuron, n_output]
        o_ref,  # [n_output]
    ):
        i_row = pl.program_id(0)
        i_block = pl.program_id(1)
        i_start = i_block * block_dim
        col_mask = (i_start + jnp.arange(block_dim) < weights_ref.shape[1])[None, :]
        count = count_ref[i_row]

        def fn(i_index_block, temp):
            i_index = i_index_block * block_dim
            ind_mask = i_index + jnp.arange(block_dim) < count
            indices = pl.load(indices_ref, pl.dslice(i_index, block_dim), mask=ind_mask)
            weight = pl.load(weights_ref, (indices, pl.dslice(i_start, block_dim)), mask=ind_mask[:, None] & col_mask)
            temp += weight.sum(axis=0)
            return temp

        out = jax.lax.fori_loop(
            0, (count + block_dim - 1) // block_dim, fn,
            jnp.zeros([block_dim], dtype=weights_ref.dtype)
        )
        pl.store(o_ref, (i_row, pl.dslice(i_start, block_dim)), out, mask=col_mask)

    return pallas_kernel(
        kernel,
        outs=kwargs['outs'],
        tile=(spikes_info.shape[0], cdiv(weights_info.shape[1], block_dim))
    )


def _binary_mat_dot_dense_mat_jvp_spikes(spikes_dot, spikes, indices, count, weights, **kwargs):
    return [spikes_dot @ weights]


def _binary_mat_dot_dense_mat_jvp_weights(weights_dot, spikes, indices, count, weights, **kwargs):
    return _binary_mat_dot_dense_mat_p_call(spikes, indices, count, weights_dot)


def _binary_mat_dot_dense_mat_transpose(ct, spikes, indices, count, weights, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to indices.")
    if ad.is_undefined_primal(count):
        raise ValueError("Cannot transpose with respect to count.")
    if ad.is_undefined_primal(spikes):
        return ct[0] @ weights.T, indices, count, weights
    elif ad.is_undefined_primal(weights):
        # return spikes, indices, count, dense_mat_dot_binary_mat()
        return spikes, indices, count, spikes.T @ ct[0]
    else:
        raise ValueError("Cannot transpose with respect to both spikes and weights.")


def _binary_mat_dot_dense_mat_p_call(spikes, indices, count, weights):
    assert spikes.ndim == 2, "spikes should be 2D (batch_size, n_spikes)"
    assert indices.ndim == 2, "indices should be 2D (batch_size, n_spikes)"
    assert count.ndim == 1 and count.shape[0] == spikes.shape[0], "count should be 1D (batch_size,)"
    assert weights.ndim == 2, "weights should be 2D (n_input, n_output)"
    assert spikes.shape[1] == weights.shape[0], (f"spikes and weights dimension mismatch, "
                                                 f"got {spikes.shape} and {weights.shape}")
    return _binary_mat_dot_dense_mat_p(
        spikes,
        indices,
        count,
        weights,
        outs=[
            jax.ShapeDtypeStruct([spikes.shape[0], weights.shape[1]], weights.dtype)
        ],
        spikes_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        count_info=jax.ShapeDtypeStruct(count.shape, count.dtype),
        weights_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
    )


_binary_mat_dot_dense_mat_p = XLACustomKernel('binary_mat_dot_dense_matrix')
_binary_mat_dot_dense_mat_p.def_cpu_kernel(_binary_mat_dot_dense_mat_numba_kernel_generator)
_binary_mat_dot_dense_mat_p.def_gpu_kernel(
    warp=_binary_mat_dot_dense_mat_warp_kernel_generator,
    pallas=_binary_mat_dot_dense_mat_pallas_kernel_generator,
    default='pallas'
)
_binary_mat_dot_dense_mat_p.def_tpu_kernel(_binary_mat_dot_dense_mat_pallas_kernel_generator)
_binary_mat_dot_dense_mat_p.def_jvp_rule2(
    _binary_mat_dot_dense_mat_jvp_spikes,
    None, None,
    _binary_mat_dot_dense_mat_jvp_weights,
)
_binary_mat_dot_dense_mat_p.def_transpose_rule(_binary_mat_dot_dense_mat_transpose)


def dense_mat_dot_binary_mat(weights, binary_arr):
    weight_val, wunit = u.split_mantissa_unit(weights)
    spikes = binary_arr.value
    indices = binary_arr.spike_indices
    count = binary_arr.spike_count
    raise ValueError
