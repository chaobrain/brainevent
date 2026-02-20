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
import numbers
from functools import partial
from pathlib import Path
from typing import Union, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._misc import namescope
from brainevent._op import XLACustomKernel, numba_kernel, register_tvm_cuda_from_file
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._typing import MatrixShape

__all__ = [
    'update_csr_on_binary_pre',
    'update_csr_on_binary_pre_p',
    'update_csr_on_binary_post',
    'update_csr_on_binary_post_p',
]


@namescope(static_argnames=['shape'])
def update_csr_on_binary_pre(
    weight: Union[u.Quantity, jax.Array, numbers.Number],
    indices: Union[np.ndarray, jax.Array],
    indptr: Union[np.ndarray, jax.Array],
    pre_spike: jax.Array,
    post_trace: Union[u.Quantity, jax.Array],
    w_min: Optional[Union[u.Quantity, jax.Array, numbers.Number]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array, numbers.Number]] = None,
    *,
    shape: MatrixShape,
    backend: Optional[str] = None,
):
    """Update CSR synaptic weights triggered by presynaptic binary spike events.

    Implements a spike-timing-dependent plasticity (STDP) rule for sparse
    connectivity matrices stored in Compressed Sparse Row (CSR) format. For each
    presynaptic neuron ``i`` that fires (``pre_spike[i]`` is ``True`` or nonzero),
    the weights of all outgoing synapses from that neuron are updated by adding the
    corresponding postsynaptic trace values:

    ``weight[indptr[i]:indptr[i+1]] += post_trace[indices[indptr[i]:indptr[i+1]]]``

    After the update, weights are optionally clipped to ``[w_min, w_max]``.

    Parameters
    ----------
    weight : jax.Array, Quantity, or number
        Sparse synaptic weight array in CSR data format, with shape ``(nse,)``
        where ``nse`` is the number of stored elements. May carry physical units
        via ``brainunit.Quantity``.
    indices : ndarray or jax.Array
        Column indices array of the CSR format, with shape ``(nse,)`` and integer
        dtype.
    indptr : ndarray or jax.Array
        Row pointer array of the CSR format, with shape ``(n_pre + 1,)`` and
        integer dtype.
    pre_spike : jax.Array
        Binary or boolean array indicating presynaptic spike events, with shape
        ``(n_pre,)``. If boolean, ``True`` indicates a spike. If float, any
        nonzero value indicates a spike.
    post_trace : jax.Array or Quantity
        Postsynaptic eligibility trace values, with shape ``(n_post,)``. Must be
        compatible in units with ``weight``.
    w_min : jax.Array, Quantity, number, or None, optional
        Lower bound for weight clipping. Must have the same units as ``weight``.
        If ``None``, no lower bound is applied.
    w_max : jax.Array, Quantity, number, or None, optional
        Upper bound for weight clipping. Must have the same units as ``weight``.
        If ``None``, no upper bound is applied.
    shape : tuple of int
        Full matrix shape as ``(n_pre, n_post)``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'pallas'``, or
        ``None`` for automatic selection.

    Returns
    -------
    jax.Array or Quantity
        Updated weight array with the same shape ``(nse,)`` and units as the
        input ``weight``.

    Raises
    ------
    AssertionError
        If ``weight`` is not 1-D, if ``pre_spike`` is not 1-D, if ``post_trace``
        is not 1-D, if ``shape[0] != pre_spike.shape[0]``,
        ``shape[1] != post_trace.shape[0]``, or if
        ``weight.shape[0] != indices.shape[0]``.  These checks are performed by
        the underlying :func:`csr_on_pre_prim_call`.

    See Also
    --------
    update_csr_on_binary_post : Post-synaptic-spike-triggered weight update.
    update_csr_on_binary_pre_p : Low-level XLA custom kernel primitive for this
        operation.

    Notes
    -----
    This function implements the **pre-synaptic** component of an additive
    spike-timing-dependent plasticity (STDP) rule.  In the standard pair-based
    STDP formulation the weight matrix ``W`` with shape ``(n_pre, n_post)`` is
    stored in CSR format.  When presynaptic neuron ``j`` fires, the update for
    every synapse ``(i, j)`` that exists in the sparsity pattern is:

    ``W[i, j] <- W[i, j] + post_trace[i]``

    After the additive update, weights are clipped element-wise:

    ``W[i, j] <- clip(W[i, j], w_min, w_max)``

    Here ``post_trace`` is an eligibility trace that typically decays
    exponentially between postsynaptic spikes, so synapses that experienced a
    recent postsynaptic spike receive a larger update.

    The function internally converts ``post_trace`` to the same unit as ``weight``
    before performing arithmetic, so mixed-unit inputs are supported as long as
    the units are dimensionally compatible.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.plasticity_binary import update_csr_on_binary_pre
        >>> weight = jnp.array([0.5, 0.3, 0.8, 0.2], dtype=jnp.float32)
        >>> indices = jnp.array([0, 1, 0, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> pre_spike = jnp.array([True, False])
        >>> post_trace = jnp.array([0.1, 0.2, 0.05], dtype=jnp.float32)
        >>> updated = update_csr_on_binary_pre(
        ...     weight, indices, indptr, pre_spike, post_trace,
        ...     shape=(2, 3),
        ... )
    """
    weight, wunit = u.split_mantissa_unit(weight)
    post_trace = u.Quantity(post_trace).to(wunit).mantissa
    weight = u.maybe_decimal(
        csr_on_pre_prim_call(
            weight, indices, indptr, pre_spike, post_trace, shape=shape, backend=backend
        )[0] * wunit
    )
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _csr_on_pre_numba_kernel_generator(
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    if spike_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def kernel(weight, indices, indptr, pre_spike, post_trace, out_w):
            for i in range(pre_spike.shape[0]):
                if pre_spike[i]:
                    i_start = indptr[i]
                    i_end = indptr[i + 1]
                    out_w[i_start: i_end] += post_trace[indices[i_start: i_end]]
    else:
        @numba.njit(fastmath=True)
        def kernel(weight, indices, indptr, pre_spike, post_trace, out_w):
            for i in range(pre_spike.shape[0]):
                if pre_spike[i] != 0.:
                    i_start = indptr[i]
                    i_end = indptr[i + 1]
                    out_w[i_start: i_end] += post_trace[indices[i_start: i_end]]

    def fn(weight, indices, indptr, pre_spike, post_trace):
        return numba_kernel(kernel, outs=kwargs['outs'], input_output_aliases={0: 0})(
            weight, indices, indptr, pre_spike, post_trace
        )

    return fn


def _csr_on_pre_pallas_kernel_generator(
    impl_backend,
    spike_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs
):
    from jax.experimental import pallas as pl

    if spike_info.dtype == jnp.bool_:
        def kernel_fn(weight_ref, indices_ref, indptr_ref, spike_ref, trace_ref, _, out_w_ref):
            i_row = pl.program_id(0)
            i_col_start = indptr_ref[i_row]
            i_col_end = indptr_ref[i_row + 1]

            @pl.when(spike_ref[i_row])
            def run():
                def loop_fn(j, _):
                    idx = i_col_start + j
                    post_id = indices_ref[idx]
                    out_w_ref[idx] = out_w_ref[idx] + trace_ref[post_id]

                jax.lax.fori_loop(0, i_col_end - i_col_start, loop_fn, None)
    else:
        def kernel_fn(weight_ref, indices_ref, indptr_ref, spike_ref, trace_ref, _, out_w_ref):
            i_row = pl.program_id(0)
            i_col_start = indptr_ref[i_row]
            i_col_end = indptr_ref[i_row + 1]

            @pl.when(spike_ref[i_row] != 0.)
            def run():
                def loop_fn(j, _):
                    idx = i_col_start + j
                    post_id = indices_ref[idx]
                    out_w_ref[idx] = out_w_ref[idx] + trace_ref[post_id]

                jax.lax.fori_loop(0, i_col_end - i_col_start, loop_fn, None)

    def kernel(weight, indices, indptr, pre_spike, post_trace):
        fn = pl.pallas_call(
            kernel_fn,
            grid=(shape[0],),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs'],
            backend=impl_backend,
        )
        return fn(weight, indices, indptr, pre_spike, post_trace, weight)

    return kernel


def _csr_on_pre_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        n_conn = max(1, int(n_post * prob))
        indptr = np.arange(n_pre + 1, dtype=np.int32) * n_conn
        indices = np.random.randint(0, n_post, (n_pre * n_conn,), dtype=np.int32)
        weight = jnp.ones(n_pre * n_conn, dtype=dtype)
        if bool_event:
            pre_spike = jnp.asarray(np.random.rand(n_pre) > 0.5, dtype=jnp.bool_)
        else:
            pre_spike = jnp.asarray(np.random.rand(n_pre), dtype=dtype)
        post_trace = jnp.asarray(np.random.randn(n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append(
            BenchmarkConfig(
                name,
                (weight, indices, jnp.asarray(indptr), pre_spike, post_trace),
                {'shape': (n_pre, n_post)}
            )
        )
    return configs


def _csr_on_pre_jax_kernel(
    spike_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs,
):
    """Pure-JAX kernel for CSR pre-synaptic plasticity update (all platforms)."""
    n_pre, n_post = shape
    is_bool = (spike_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size

    def kernel(weight, indices, indptr, pre_spike, post_trace):
        row_ids = jnp.repeat(
            jnp.arange(n_pre, dtype=indptr.dtype),
            jnp.diff(indptr),
            total_repeat_length=nse,
        )
        if is_bool:
            active = pre_spike[row_ids]
        else:
            active = pre_spike[row_ids] != 0.
        delta = jnp.where(active, post_trace[indices], jnp.zeros_like(post_trace[indices]))
        return [weight + delta]

    return kernel


def _csr_on_pre_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs,
):
    """TVM FFI CUDA kernel for CSR pre-synaptic plasticity update.

    Dispatches to ``update_csr_on_pre{wt_sfx}{spk_sfx}`` compiled from
    ``plasticity_binary_on_pre.cu``.  The auto-variant selects among
    thread-per-row, warp-per-row, and block-per-row sub-kernels at runtime
    based on avg_nnz = nse / n_pre.

    Only int32 index arrays are supported.  Callers with int64 indices should
    explicitly select ``backend='pallas'`` or ``backend='jax'``.
    """
    if indices_info.dtype == jnp.int64:
        raise TypeError(
            "update_csr_on_binary_pre: the 'tvmffi' backend only supports "
            "int32 index arrays (indices / indptr).  "
            "Use backend='pallas' or backend='jax' for int64 indices."
        )

    register_tvm_cuda_from_file(
        module='csr_plasticity_on_pre',
        source=Path(__file__).parent.joinpath('plasticity_binary_on_pre.cu'),
    )

    out_info = kwargs['outs']
    spk_suffix = '_bool' if spike_info.dtype == jnp.bool_ else '_float'

    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    kernel_name = f'csr_plasticity_on_pre.update_csr_on_pre{wt_sfx}{spk_suffix}'

    def kernel(weight, indices, indptr, pre_spike, post_trace):
        return jax.ffi.ffi_call(
            kernel_name,
            out_info,
            input_output_aliases={0: 0},
        )(weight, indices, indptr, pre_spike, post_trace)

    return kernel


def csr_on_pre_prim_call(weight, indices, indptr, pre_spike, post_trace, *, shape, backend: Optional[str] = None):
    """Invoke the low-level XLA custom kernel for presynaptic plasticity updates.

    Validates input shapes and dimensions, then dispatches to
    :data:`update_csr_on_binary_pre_p` with the appropriate metadata. This is the
    direct primitive call without unit handling or weight clipping; most users
    should prefer :func:`update_csr_on_binary_pre`.

    Parameters
    ----------
    weight : jax.Array
        Sparse synaptic weight data array, with shape ``(nse,)`` where ``nse`` is
        the number of stored elements.
    indices : jax.Array
        Column indices array of the CSR format, with shape ``(nse,)``.
    indptr : jax.Array
        Row pointer array of the CSR format, with shape ``(n_pre + 1,)``.
    pre_spike : jax.Array
        Binary or boolean presynaptic spike array, with shape ``(n_pre,)``.
    post_trace : jax.Array
        Postsynaptic trace values, with shape ``(n_post,)``.
    shape : tuple of int
        Full matrix shape as ``(n_pre, n_post)``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'pallas'``, or
        ``None`` for automatic selection.

    Returns
    -------
    tuple of jax.Array
        A single-element tuple containing the updated weight array with shape
        ``(nse,)``.

    Raises
    ------
    AssertionError
        If ``weight`` is not 1-D, if ``pre_spike`` is not 1-D, if ``post_trace``
        is not 1-D, if dimension sizes in ``shape`` do not match
        ``pre_spike.shape[0]`` and ``post_trace.shape[0]``, or if
        ``weight.shape[0] != indices.shape[0]``.

    See Also
    --------
    update_csr_on_binary_pre : High-level wrapper with unit handling and clipping.

    Notes
    -----
    This function operates on unitless mantissa arrays. All physical-unit
    handling and weight clipping are performed by the caller
    (:func:`update_csr_on_binary_pre`).  The additive update rule applied per
    row ``i`` where ``pre_spike[i]`` is active is:

    ``weight[indptr[i] : indptr[i+1]] += post_trace[indices[indptr[i] : indptr[i+1]]]``

    The function constructs :class:`jax.ShapeDtypeStruct` metadata for each
    operand and forwards the call to the ``XLACustomKernel`` instance
    :data:`update_csr_on_binary_pre_p`.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.plasticity_binary import csr_on_pre_prim_call
        >>> weight = jnp.array([0.5, 0.3, 0.8, 0.2], dtype=jnp.float32)
        >>> indices = jnp.array([0, 1, 0, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> pre_spike = jnp.array([True, False])
        >>> post_trace = jnp.array([0.1, 0.2, 0.05], dtype=jnp.float32)
        >>> (updated,) = csr_on_pre_prim_call(
        ...     weight, indices, indptr, pre_spike, post_trace,
        ...     shape=(2, 3),
        ... )
    """
    assert weight.ndim == 1, 'dense_one_pre only support 1D weight.'
    assert pre_spike.ndim == 1, 'pre_spike should be 1D.'
    assert post_trace.ndim == 1, 'post_trace should be 1D.'
    assert shape[0] == pre_spike.shape[0], f'pre_spike shape {pre_spike.shape} does not match with shape {shape}.'
    assert shape[1] == post_trace.shape[0], f'post_trace shape {post_trace.shape} does not match with shape {shape}.'
    assert weight.shape[0] == indices.shape[0], (
        f'weight shape {weight.shape}, indices shape {indices.shape}, indptr shape {indptr.shape} do not match.'
    )
    return update_csr_on_binary_pre_p(
        weight, indices, indptr, pre_spike, post_trace,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        spike_info=jax.ShapeDtypeStruct(pre_spike.shape, pre_spike.dtype),
        trace_info=jax.ShapeDtypeStruct(post_trace.shape, post_trace.dtype),
        backend=backend,
    )


update_csr_on_binary_pre_p = XLACustomKernel(
    'binary_csr_plast',
    doc="""
Low-level XLA custom-kernel primitive for ``update_csr_on_binary_pre``.

This ``XLACustomKernel`` instance dispatches the CSR weight update for pre-synaptic binary plasticity
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

For each presynaptic neuron that fires, updates all outgoing synaptic weights
by adding the corresponding postsynaptic trace values, implementing the pre-synaptic
component of spike-timing-dependent plasticity (STDP).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``update_csr_on_binary_pre_p.available_backends(platform)``,
and the default backend can be configured with ``update_csr_on_binary_pre_p.set_default(platform, backend)``.

See Also
--------
update_csr_on_binary_pre : High-level user-facing function wrapper.
"""
)
update_csr_on_binary_pre_p.def_numba_kernel(_csr_on_pre_numba_kernel_generator)
update_csr_on_binary_pre_p.def_pallas_kernel('gpu', partial(_csr_on_pre_pallas_kernel_generator, 'triton'))
update_csr_on_binary_pre_p.def_pallas_kernel('tpu', partial(_csr_on_pre_pallas_kernel_generator, 'mosaic_tpu'))
update_csr_on_binary_pre_p.def_tvmffi_kernel('gpu', _csr_on_pre_cuda_kernel)
update_csr_on_binary_pre_p.def_kernel('jax_raw', 'cpu', _csr_on_pre_jax_kernel)
update_csr_on_binary_pre_p.def_kernel('jax_raw', 'gpu', _csr_on_pre_jax_kernel)
update_csr_on_binary_pre_p.def_kernel('jax_raw', 'tpu', _csr_on_pre_jax_kernel)
update_csr_on_binary_pre_p.def_tags('csr', 'plasticity')
update_csr_on_binary_pre_p.def_benchmark_data(_csr_on_pre_benchmark_data)
update_csr_on_binary_pre_p.def_call(csr_on_pre_prim_call)


@namescope(static_argnames=['shape'])
def update_csr_on_binary_post(
    weight: Union[u.Quantity, jax.Array],
    indices: Union[np.ndarray, jax.Array],
    indptr: Union[np.ndarray, jax.Array],
    weight_indices: Union[np.ndarray, jax.Array],
    pre_trace: Union[u.Quantity, jax.Array],
    post_spike: jax.Array,
    w_min: Optional[Union[u.Quantity, jax.Array, numbers.Number]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array, numbers.Number]] = None,
    *,
    shape: MatrixShape,
    backend: Optional[str] = None,
):
    """Update CSR synaptic weights triggered by postsynaptic binary spike events.

    Implements a spike-timing-dependent plasticity (STDP) rule for sparse
    connectivity stored in CSC (Compressed Sparse Column) layout. For each
    postsynaptic neuron ``j`` that fires (``post_spike[j]`` is ``True`` or
    nonzero), the weights of all incoming synapses to that neuron are updated
    by adding the corresponding presynaptic trace values:

    ``weight[weight_indices[indptr[j]:indptr[j+1]]] += pre_trace[indices[indptr[j]:indptr[j+1]]]``

    The CSC structure (``indices``, ``indptr``) indexes by postsynaptic neuron,
    while ``weight_indices`` maps back to the original CSR weight positions.
    After the update, weights are optionally clipped to ``[w_min, w_max]``.

    Parameters
    ----------
    weight : jax.Array or Quantity
        Sparse synaptic weight array, with shape ``(nse,)`` where ``nse`` is the
        number of stored elements. May carry physical units via
        ``brainunit.Quantity``.
    indices : ndarray or jax.Array
        Row indices array of the CSC format, with shape ``(nse,)`` and integer
        dtype.
    indptr : ndarray or jax.Array
        Column pointer array of the CSC format, with shape ``(n_post + 1,)`` and
        integer dtype.
    weight_indices : ndarray or jax.Array
        Mapping from CSC element positions to CSR weight positions, with shape
        ``(nse,)`` and integer dtype.
    pre_trace : jax.Array or Quantity
        Presynaptic eligibility trace values, with shape ``(n_pre,)``. Must be
        compatible in units with ``weight``.
    post_spike : jax.Array
        Binary or boolean array indicating postsynaptic spike events, with shape
        ``(n_post,)``. If boolean, ``True`` indicates a spike. If float, any
        nonzero value indicates a spike.
    w_min : jax.Array, Quantity, number, or None, optional
        Lower bound for weight clipping. Must have the same units as ``weight``.
        If ``None``, no lower bound is applied.
    w_max : jax.Array, Quantity, number, or None, optional
        Upper bound for weight clipping. Must have the same units as ``weight``.
        If ``None``, no upper bound is applied.
    shape : tuple of int
        Full matrix shape as ``(n_pre, n_post)``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'pallas'``, or
        ``None`` for automatic selection.

    Returns
    -------
    jax.Array or Quantity
        Updated weight array with the same shape ``(nse,)`` and units as the
        input ``weight``.

    Raises
    ------
    AssertionError
        If ``weight`` is not 1-D, if ``post_spike`` is not 1-D, if ``pre_trace``
        is not 1-D, if ``shape[1] != post_spike.shape[0]``,
        ``shape[0] != pre_trace.shape[0]``, or if ``weight.shape``,
        ``weight_indices.shape``, and ``indices.shape`` are not all equal.
        These checks are performed by the underlying
        :func:`csr2csc_on_post_prim_call`.

    See Also
    --------
    update_csr_on_binary_pre : Pre-synaptic-spike-triggered weight update.
    update_csr_on_binary_post_p : Low-level XLA custom kernel primitive for this
        operation.

    Notes
    -----
    This function implements the **post-synaptic** component of an additive
    spike-timing-dependent plasticity (STDP) rule.  In the standard pair-based
    STDP formulation the weight matrix ``W`` with shape ``(n_pre, n_post)`` is
    stored in CSR format.  When postsynaptic neuron ``i`` fires, the update for
    every synapse ``(i, j)`` that exists in the sparsity pattern is:

    ``W[i, j] <- W[i, j] + pre_trace[j]``

    After the additive update, weights are clipped element-wise:

    ``W[i, j] <- clip(W[i, j], w_min, w_max)``

    Here ``pre_trace`` is an eligibility trace that typically decays
    exponentially between presynaptic spikes, so synapses whose presynaptic
    neuron fired recently receive a larger update.

    The function internally converts ``pre_trace`` to the same unit as ``weight``
    before performing arithmetic, so mixed-unit inputs are supported as long as
    the units are dimensionally compatible.

    The CSC layout is used so that iterating over postsynaptic spikes efficiently
    gathers all incoming synapses. The ``weight_indices`` array allows writing
    the updated values back to the correct positions in the original CSR weight
    array.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.plasticity_binary import update_csr_on_binary_post
        >>> weight = jnp.array([0.5, 0.3, 0.8, 0.2], dtype=jnp.float32)
        >>> indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> weight_indices = jnp.array([0, 2, 1, 3], dtype=jnp.int32)
        >>> pre_trace = jnp.array([0.1, -0.05], dtype=jnp.float32)
        >>> post_spike = jnp.array([True, False])
        >>> updated = update_csr_on_binary_post(
        ...     weight, indices, indptr, weight_indices,
        ...     pre_trace, post_spike, shape=(2, 2),
        ... )
    """
    weight, wunit = u.split_mantissa_unit(weight)
    pre_trace = u.Quantity(pre_trace).to(wunit).mantissa
    weight = u.maybe_decimal(
        csr2csc_on_post_prim_call(
            weight, indices, indptr, weight_indices, pre_trace, post_spike, shape=shape, backend=backend
        )[0] * wunit
    )
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _csr2csc_on_post_numba_kernel_generator(
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    # Note: Cannot parallelize due to potential race conditions when updating out_w[weight_ids]
    if spike_info.dtype == jnp.bool_:
        @numba.njit(fastmath=True)
        def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike, out_w):
            for i in range(post_spike.shape[0]):
                if post_spike[i]:
                    index = indptr[i]
                    index_end = indptr[i + 1]
                    weight_ids = weight_indices[index: index_end]
                    pre_ids = indices[index: index_end]
                    out_w[weight_ids] += pre_trace[pre_ids]
    else:
        @numba.njit(fastmath=True)
        def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike, out_w):
            for i in range(post_spike.shape[0]):
                if post_spike[i] != 0.:
                    index = indptr[i]
                    index_end = indptr[i + 1]
                    weight_ids = weight_indices[index: index_end]
                    pre_ids = indices[index: index_end]
                    out_w[weight_ids] += pre_trace[pre_ids]

    def fn(weight, indices, indptr, weight_indices, pre_trace, post_spike):
        return numba_kernel(kernel, outs=kwargs['outs'], input_output_aliases={0: 0})(
            weight, indices, indptr, weight_indices, pre_trace, post_spike
        )

    return fn


def _csr2csc_on_post_pallas_kernel_generator(
    impl_backend,
    spike_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs
):
    from jax.experimental import pallas as pl

    if spike_info.dtype == jnp.bool_:
        def kernel_fn(weight_ref, indices_ref, indptr_ref, weight_indices_ref, trace_ref, spike_ref, _, out_w_ref):
            i_col = pl.program_id(0)
            i_row_start = indptr_ref[i_col]
            i_row_end = indptr_ref[i_col + 1]

            @pl.when(spike_ref[i_col])
            def run():
                def loop_fn(j, _):
                    idx = i_row_start + j
                    pre_id = indices_ref[idx]
                    w_id = weight_indices_ref[idx]
                    out_w_ref[w_id] = out_w_ref[w_id] + trace_ref[pre_id]

                jax.lax.fori_loop(0, i_row_end - i_row_start, loop_fn, None)
    else:
        def kernel_fn(weight_ref, indices_ref, indptr_ref, weight_indices_ref, trace_ref, spike_ref, _, out_w_ref):
            i_col = pl.program_id(0)
            i_row_start = indptr_ref[i_col]
            i_row_end = indptr_ref[i_col + 1]

            @pl.when(spike_ref[i_col] != 0.)
            def run():
                def loop_fn(j, _):
                    idx = i_row_start + j
                    pre_id = indices_ref[idx]
                    w_id = weight_indices_ref[idx]
                    out_w_ref[w_id] = out_w_ref[w_id] + trace_ref[pre_id]

                jax.lax.fori_loop(0, i_row_end - i_row_start, loop_fn, None)

    def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike):
        fn = pl.pallas_call(
            kernel_fn, grid=(shape[1],), input_output_aliases={6: 0}, out_shape=kwargs['outs'], backend=impl_backend
        )
        return fn(weight, indices, indptr, weight_indices, pre_trace, post_spike, weight)

    return kernel


def _csr2csc_on_post_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        n_conn = max(1, int(n_post * prob))
        indptr = np.arange(n_post + 1, dtype=np.int32) * n_conn
        indices = np.random.randint(0, n_pre, (n_post * n_conn,), dtype=np.int32)
        weight = jnp.ones(n_post * n_conn, dtype=dtype)
        weight_indices = jnp.asarray(
            np.random.randint(0, n_post * n_conn, (n_post * n_conn,), dtype=np.int32)
        )
        pre_trace = jnp.asarray(np.random.randn(n_pre), dtype=dtype)
        if bool_event:
            post_spike = jnp.asarray(np.random.rand(n_post) > 0.5, dtype=jnp.bool_)
        else:
            post_spike = jnp.asarray(np.random.rand(n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append(
            BenchmarkConfig(
                name,
                (weight, indices, jnp.asarray(indptr), weight_indices, pre_trace, post_spike),
                {'shape': (n_pre, n_post)}
            )
        )
    return configs


def _csr2csc_on_post_jax_kernel(
    spike_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    **kwargs,
):
    """Pure-JAX kernel for CSR post-synaptic plasticity update (all platforms)."""
    n_pre, n_post = shape
    is_bool = (spike_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size

    def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike):
        col_ids = jnp.repeat(
            jnp.arange(n_post, dtype=indptr.dtype),
            jnp.diff(indptr),
            total_repeat_length=nse,
        )
        if is_bool:
            active = post_spike[col_ids]
        else:
            active = post_spike[col_ids] != 0.
        delta = jnp.where(active, pre_trace[indices], jnp.zeros_like(pre_trace[indices]))
        return [weight.at[weight_indices].add(delta)]

    return kernel


def _csr2csc_on_post_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs,
):
    """TVM FFI CUDA kernel for CSR post-synaptic plasticity update.

    Dispatches to ``update_csr_on_post{wt_sfx}{spk_sfx}`` compiled from
    ``plasticity_binary_on_post.cu``.  The auto-variant selects among
    thread-per-column, warp-per-column, and block-per-column sub-kernels at
    runtime based on avg_nnz = nse / n_post.

    Uses atomicAdd for scattered writes to the weight array.  Since
    weight_indices is injective by CSC construction, no actual race conditions
    occur; atomicAdd provides a safety guarantee against malformed inputs.

    Only int32 index arrays are supported.  Callers with int64 indices should
    explicitly select ``backend='pallas'`` or ``backend='jax'``.
    """
    if indices_info.dtype == jnp.int64:
        raise TypeError(
            "update_csr_on_binary_post: the 'tvmffi' backend only supports "
            "int32 index arrays (indices / indptr / weight_indices).  "
            "Use backend='pallas' or backend='jax' for int64 indices."
        )

    register_tvm_cuda_from_file(
        module='csr_plasticity_on_post',
        source=Path(__file__).parent.joinpath('plasticity_binary_on_post.cu'),
    )

    out_info = kwargs['outs']
    spk_suffix = '_bool' if spike_info.dtype == jnp.bool_ else '_float'

    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    kernel_name = f'csr_plasticity_on_post.update_csr_on_post{wt_sfx}{spk_suffix}'

    def kernel(weight, indices, indptr, weight_indices, pre_trace, post_spike):
        return jax.ffi.ffi_call(
            kernel_name,
            out_info,
            input_output_aliases={0: 0},
        )(weight, indices, indptr, weight_indices, pre_trace, post_spike)

    return kernel


def csr2csc_on_post_prim_call(weight, indices, indptr, weight_indices, pre_trace, post_spike, *, shape,
                              backend: Optional[str] = None):
    """Invoke the low-level XLA custom kernel for postsynaptic plasticity updates.

    Validates input shapes and dimensions, then dispatches to
    :data:`update_csr_on_binary_post_p` with the appropriate metadata. This is the
    direct primitive call without unit handling or weight clipping; most users
    should prefer :func:`update_csr_on_binary_post`.

    Parameters
    ----------
    weight : jax.Array
        Sparse synaptic weight data array, with shape ``(nse,)`` where ``nse`` is
        the number of stored elements.
    indices : jax.Array
        Row indices array of the CSC format, with shape ``(nse,)``.
    indptr : jax.Array
        Column pointer array of the CSC format, with shape ``(n_post + 1,)``.
    weight_indices : jax.Array
        Mapping from CSC element positions to CSR weight positions, with shape
        ``(nse,)`` and integer dtype.
    pre_trace : jax.Array
        Presynaptic trace values, with shape ``(n_pre,)``.
    post_spike : jax.Array
        Binary or boolean postsynaptic spike array, with shape ``(n_post,)``.
    shape : tuple of int
        Full matrix shape as ``(n_pre, n_post)``.
    backend : str or None, optional
        Compute backend to use. One of ``'numba'``, ``'pallas'``, or
        ``None`` for automatic selection.

    Returns
    -------
    tuple of jax.Array
        A single-element tuple containing the updated weight array with shape
        ``(nse,)``.

    Raises
    ------
    AssertionError
        If ``weight`` is not 1-D, if ``post_spike`` is not 1-D, if ``pre_trace``
        is not 1-D, if dimension sizes in ``shape`` do not match
        ``post_spike.shape[0]`` and ``pre_trace.shape[0]``, or if
        ``weight.shape``, ``weight_indices.shape``, and ``indices.shape`` are not
        all equal.

    See Also
    --------
    update_csr_on_binary_post : High-level wrapper with unit handling and clipping.

    Notes
    -----
    This function operates on unitless mantissa arrays. All physical-unit
    handling and weight clipping are performed by the caller
    (:func:`update_csr_on_binary_post`).  The CSC layout allows efficient
    iteration over postsynaptic neurons; for each postsynaptic neuron ``j``
    where ``post_spike[j]`` is active the update is:

    ``weight[weight_indices[indptr[j] : indptr[j+1]]] += pre_trace[indices[indptr[j] : indptr[j+1]]]``

    The function constructs :class:`jax.ShapeDtypeStruct` metadata for each
    operand and forwards the call to the ``XLACustomKernel`` instance
    :data:`update_csr_on_binary_post_p`.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._csr.plasticity_binary import csr2csc_on_post_prim_call
        >>> weight = jnp.array([0.5, 0.3, 0.8, 0.2], dtype=jnp.float32)
        >>> indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> weight_indices = jnp.array([0, 2, 1, 3], dtype=jnp.int32)
        >>> pre_trace = jnp.array([0.1, -0.05], dtype=jnp.float32)
        >>> post_spike = jnp.array([True, False])
        >>> (updated,) = csr2csc_on_post_prim_call(
        ...     weight, indices, indptr, weight_indices,
        ...     pre_trace, post_spike, shape=(2, 2),
        ... )
    """
    assert weight.ndim == 1, 'dense_one_post only support 1D weight.'
    assert post_spike.ndim == 1, 'post_spike should be 1D.'
    assert pre_trace.ndim == 1, 'pre_trace should be 1D.'
    assert shape[1] == post_spike.shape[0], f'post_spike shape {post_spike.shape} does not match with shape {shape}.'
    assert shape[0] == pre_trace.shape[0], f'pre_trace shape {pre_trace.shape} does not match with shape {shape}.'
    assert weight.shape == weight_indices.shape == indices.shape, (
        f'weight shape {weight.shape}, weight_indices shape {weight_indices.shape}, '
        f'indices shape {indices.shape}, indptr shape {indptr.shape} do not match.'
    )
    return update_csr_on_binary_post_p(
        weight, indices, indptr, weight_indices, pre_trace, post_spike,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_indices_info=jax.ShapeDtypeStruct(weight_indices.shape, weight_indices.dtype),
        trace_info=jax.ShapeDtypeStruct(pre_trace.shape, pre_trace.dtype),
        spike_info=jax.ShapeDtypeStruct(post_spike.shape, post_spike.dtype),
        backend=backend,
    )


update_csr_on_binary_post_p = XLACustomKernel(
    'csr2csc_on_post',
    doc="""
Low-level XLA custom-kernel primitive for ``update_csr_on_binary_post``.

This ``XLACustomKernel`` instance dispatches the CSR weight update for post-synaptic binary plasticity (CSR to CSC conversion)
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

For each postsynaptic neuron that fires, updates all incoming synaptic weights
by adding the corresponding presynaptic trace values, implementing the post-synaptic
component of spike-timing-dependent plasticity (STDP). Uses a CSC structure internally
for efficient iteration over postsynaptic spikes.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``update_csr_on_binary_post_p.available_backends(platform)``,
and the default backend can be configured with ``update_csr_on_binary_post_p.set_default(platform, backend)``.

See Also
--------
update_csr_on_binary_post : High-level user-facing function wrapper.
"""
)
update_csr_on_binary_post_p.def_numba_kernel(_csr2csc_on_post_numba_kernel_generator)
update_csr_on_binary_post_p.def_pallas_kernel('gpu', partial(_csr2csc_on_post_pallas_kernel_generator, 'triton'))
update_csr_on_binary_post_p.def_pallas_kernel('tpu', partial(_csr2csc_on_post_pallas_kernel_generator, 'mosaic_tpu'))
update_csr_on_binary_post_p.def_tvmffi_kernel('gpu', _csr2csc_on_post_cuda_kernel)
update_csr_on_binary_post_p.def_kernel('jax_raw', 'cpu', _csr2csc_on_post_jax_kernel)
update_csr_on_binary_post_p.def_kernel('jax_raw', 'gpu', _csr2csc_on_post_jax_kernel)
update_csr_on_binary_post_p.def_kernel('jax_raw', 'tpu', _csr2csc_on_post_jax_kernel)
update_csr_on_binary_post_p.def_tags('csr', 'plasticity')
update_csr_on_binary_post_p.def_benchmark_data(_csr2csc_on_post_benchmark_data)
update_csr_on_binary_post_p.def_call(csr2csc_on_post_prim_call)
