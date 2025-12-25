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

from typing import Union, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._compatible_import import pallas as pl
from brainevent._misc import generate_block_dim
from brainevent._typing import MatrixShape
from brainevent._op.main import XLACustomKernel
from brainevent._op.op_numba import numba_kernel
from brainevent._op.op_pallas import pallas_kernel


def csr_on_pre(
    weight: Union[u.Quantity, jax.Array],
    indices: Union[np.ndarray, jax.Array],
    indptr: Union[np.ndarray, jax.Array],
    pre_spike: jax.Array,
    post_trace: Union[u.Quantity, jax.Array],
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
    *,
    shape: MatrixShape,
):
    """Updates synaptic weights in CSR format based on presynaptic spike events and postsynaptic traces.

    This function implements a plasticity rule for sparse connectivity matrices where
    presynaptic spikes trigger weight updates modulated by postsynaptic trace values.
    The weight matrix is stored in Compressed Sparse Row (CSR) format.

    Specifically, for each presynaptic neuron, if it spikes ``pre_spike[i]`` is True,
    the weights of all synapses originating from that neuron are updated by adding the
    corresponding postsynaptic trace values ``post_trace[indices[index: index_end]]`` to
    the weights ``weight[index: index_end]``.

    Args:
        weight: Sparse synaptic weight array in CSR format, shape (n_nonzero,).
        indices: Column indices array of the CSR format, shape (n_nonzero,).
        indptr: Row pointers array of the CSR format, shape (n_rows + 1,).
        pre_spike: Binary/boolean array indicating presynaptic spike events, shape (n_pre,).
        post_trace: Postsynaptic trace values, shape (n_post,).
        w_min: Optional lower bound for weight clipping. Must have same units as weight.
        w_max: Optional upper bound for weight clipping. Must have same units as weight.
        shape: Tuple specifying the full matrix shape as (n_pre, n_post).

    Returns:
        Updated weight array with the same shape and units as the input weight.

    Note:
        The function handles unit conversion internally, ensuring that the post_trace
        is converted to the same unit as the weight before computation.
    """
    weight, wunit = u.split_mantissa_unit(weight)
    post_trace = u.Quantity(post_trace).to(wunit).mantissa
    weight = u.maybe_decimal(
        _csr_on_pre_prim_call(
            weight, indices, indptr, pre_spike, post_trace, shape=shape
        )[0] * wunit
    )
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _csr_on_pre_numba_kernel_generator(**kwargs):
    def kernel(weight, indices, indptr, pre_spike, post_trace, out_w):
        for i in range(pre_spike.shape[0]):
            if pre_spike[i]:
                i_start = indptr[i]
                i_end = indptr[i + 1]
                out_w[i_start: i_end] += post_trace[indices[i_start: i_end]]

    return numba_kernel(kernel, input_output_aliases={0: 0})


def _csr_on_pre_pallas_kernel_generator(weight_info, shape, **kwargs):
    block_dim = generate_block_dim(weight_info.shape[1], 512)

    def kernel(weight_ref, indices_ref, indptr_ref, spike_ref, trace_ref, out_w_ref):
        i_row = pl.program_id(0)
        i_col_start = indptr_ref[i_row]
        i_col_end = indptr_ref[i_row + 1]

        @pl.when(spike_ref[i_row] if spike_ref.dtype == jnp.bool_ else spike_ref[i_row] != 0)
        def run():
            def loop_fn(i_block, _):
                i_start = i_block * block_dim + i_col_start
                mask = jax.numpy.arange(block_dim) + i_start < i_col_end
                post_ids = pl.load(indices_ref, pl.dslice(i_start, block_dim), mask=mask)
                post_trace = pl.load(trace_ref, post_ids, mask=mask)
                pl.store(
                    out_w_ref,
                    pl.dslice(i_start, block_dim),
                    pl.load(out_w_ref, pl.dslice(i_start, block_dim), mask=mask) + post_trace,
                    mask=mask,
                )

            jax.lax.fori_loop(0, pl.cdiv(i_col_end - i_col_start, block_dim), loop_fn, None)

    return pallas_kernel(kernel, input_output_aliases={0: 0}, tile=(shape[0],))


def _csr_on_pre_prim_call(weight, indices, indptr, pre_spike, post_trace, *, shape):
    assert weight.ndim == 1, 'dense_one_pre only support 1D weight.'
    assert pre_spike.ndim == 1, 'pre_spike should be 1D.'
    assert post_trace.ndim == 1, 'post_trace should be 1D.'
    assert shape[0] == pre_spike.shape[0], f'pre_spike shape {pre_spike.shape} does not match with shape {shape}.'
    assert shape[1] == post_trace.shape[0], f'post_trace shape {post_trace.shape} does not match with shape {shape}.'
    assert weight.shape[0] == indices.shape[0], (
        f'weight shape {weight.shape}, indices shape {indices.shape}, indptr shape {indptr.shape} do not match.'
    )
    return _csr_on_pre_prim(
        weight, indices, indptr, pre_spike, post_trace,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        shape=shape,
    )


_csr_on_pre_prim = XLACustomKernel('csr_on_pre')
_csr_on_pre_prim.def_cpu_kernel(_csr_on_pre_numba_kernel_generator)
_csr_on_pre_prim.def_gpu_kernel(pallas=_csr_on_pre_pallas_kernel_generator)
_csr_on_pre_prim.def_tpu_kernel(_csr_on_pre_pallas_kernel_generator)


def csr2csc_on_post(
    weight: Union[u.Quantity, jax.Array],
    indices: Union[np.ndarray, jax.Array],
    indptr: Union[np.ndarray, jax.Array],
    weight_indices: Union[np.ndarray, jax.Array],
    pre_trace: Union[u.Quantity, jax.Array],
    post_spike: jax.Array,
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
    *,
    shape: MatrixShape,
):
    """Updates synaptic weights in CSC format based on postsynaptic spike events and presynaptic traces.

    This function implements a plasticity rule for sparse connectivity matrices where
    postsynaptic spikes trigger weight updates modulated by presynaptic trace values.
    The weight matrix is stored in Compressed Sparse Column (CSC) format.

    Specifically, for each postsynaptic neuron, if it spikes ``post_spike[i]`` is True,
    the weights of all synapses targeting that neuron are updated by adding the
    corresponding presynaptic trace values ``pre_trace[indices[index: index_end]]`` to
    the weights ``weight[index: index_end]``.

    Args:
        weight: Sparse synaptic weight array in CSC format, shape (n_nonzero,).
        indices: Row indices array of the CSC format, shape (n_nonzero,).
        indptr: Column pointers array of the CSC format, shape (n_cols + 1,).
        weight_indices: Array of weight indices corresponding to the synapses, shape (n_nonzero,).
        pre_trace: Presynaptic trace values, shape (n_pre,).
        post_spike: Binary/boolean array indicating postsynaptic spike events, shape (n_post,).
        w_min: Optional lower bound for weight clipping. Must have same units as weight.
        w_max: Optional upper bound for weight clipping. Must have same units as weight.
        shape: Tuple specifying the full matrix shape as (n_pre, n_post).

    Returns:
        Updated weight array with the same shape and units as the input weight.

    Note:
        The function handles unit conversion internally, ensuring that the pre_trace
        is converted to the same unit as the weight before computation.
    """
    weight, wunit = u.split_mantissa_unit(weight)
    pre_trace = u.Quantity(pre_trace).to(wunit).mantissa
    weight = u.maybe_decimal(
        _csr2csc_on_post_prim_call(
            weight, indices, indptr, weight_indices, pre_trace, post_spike, shape=shape
        )[0] * wunit
    )
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _csr2csc_on_post_numba_kernel_generator(**kwargs):
    def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike, out_w):
        for i in range(post_spike.shape[0]):
            if post_spike[i]:
                index = indptr[i]
                index_end = indptr[i + 1]
                weight_ids = weight_indices[index: index_end]
                pre_ids = indices[index: index_end]
                out_w[weight_ids] += pre_trace[pre_ids]

    return numba_kernel(kernel, parallel=False, input_output_aliases={0: 0})


def _csr2csc_on_post_pallas_kernel_generator(weight_info, shape, **kwargs):
    block_dim = generate_block_dim(weight_info.shape[0], 512)

    def kernel(weight_ref, indices_ref, indptr_ref, weight_indices_ref, trace_ref, spike_ref, out_w_ref):
        i_col = pl.program_id(0)
        i_row_start = indptr_ref[i_col]
        i_row_end = indptr_ref[i_col + 1]

        @pl.when(spike_ref[i_col] if spike_ref.dtype == jnp.bool_ else spike_ref[i_col] != 0)
        def run():
            def loop_fn(i_block, _):
                i_start = i_block * block_dim + i_row_start
                mask = jax.numpy.arange(block_dim) + i_start < i_row_end
                pre_ids = pl.load(indices_ref, pl.dslice(i_start, block_dim), mask=mask)
                weight_ids = pl.load(weight_indices_ref, pl.dslice(i_start, block_dim), mask=mask)
                pre_trace = pl.load(trace_ref, pre_ids, mask=mask)
                pl.store(
                    out_w_ref,
                    weight_ids,
                    pl.load(out_w_ref, weight_ids, mask=mask) + pre_trace,
                    mask=mask,
                )

            jax.lax.fori_loop(0, pl.cdiv(i_row_end - i_row_start, block_dim), loop_fn, None)

    return pallas_kernel(kernel, input_output_aliases={0: 0}, tile=(shape[1],))


def _csr2csc_on_post_prim_call(weight, indices, indptr, weight_indices, pre_trace, post_spike, *, shape):
    assert weight.ndim == 1, 'dense_one_post only support 1D weight.'
    assert post_spike.ndim == 1, 'post_spike should be 1D.'
    assert pre_trace.ndim == 1, 'pre_trace should be 1D.'
    assert shape[1] == post_spike.shape[0], f'post_spike shape {post_spike.shape} does not match with shape {shape}.'
    assert shape[0] == pre_trace.shape[0], f'pre_trace shape {pre_trace.shape} does not match with shape {shape}.'
    assert weight.shape == weight_indices.shape == indices.shape, (
        f'weight shape {weight.shape}, weight_indices shape {weight_indices.shape}, '
        f'indices shape {indices.shape}, indptr shape {indptr.shape} do not match.'
    )
    return _csr2csc_on_post_prim(
        weight, indices, indptr, weight_indices, pre_trace, post_spike,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        shape=shape,
    )


_csr2csc_on_post_prim = XLACustomKernel('csr2csc_on_post')
_csr2csc_on_post_prim.def_cpu_kernel(_csr2csc_on_post_numba_kernel_generator)
_csr2csc_on_post_prim.def_gpu_kernel(pallas=_csr2csc_on_post_pallas_kernel_generator)
_csr2csc_on_post_prim.def_tpu_kernel(_csr2csc_on_post_pallas_kernel_generator)
