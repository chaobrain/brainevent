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

from pathlib import Path
from typing import Union, Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._misc import generate_block_dim, namescope
from brainevent._op import XLACustomKernel, numba_kernel, register_tvm_cuda_from_file
from brainevent._op.benchmark import BenchmarkConfig

__all__ = [
    'update_coo_on_binary_pre',
    'update_coo_on_binary_pre_p',
    'update_coo_on_binary_post',
    'update_coo_on_binary_post_p',
]


@namescope
def update_coo_on_binary_pre(
    weight: Union[u.Quantity, jax.Array],
    pre_ids: jax.Array,
    post_ids: jax.Array,
    pre_spike: jax.Array,
    post_trace: Union[u.Quantity, jax.Array],
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
    *,
    backend: Optional[str] = None,
):
    """
    Update synaptic weights in COO format driven by presynaptic spike events.

    For each synapse *i* stored in COO format, if the presynaptic neuron fires
    (``pre_spike[pre_ids[i]]`` is nonzero), the weight is updated according to:

    ``weight[i] = weight[i] + post_trace[post_ids[i]]``

    After the additive update, the result is clipped to ``[w_min, w_max]`` when
    the bounds are provided. Physical units attached to ``weight`` and
    ``post_trace`` are handled transparently via ``brainunit``.

    Parameters
    ----------
    weight : jax.Array or brainunit.Quantity
        Sparse synaptic weight values stored in COO format, shape
        ``(n_synapses,)``.
    pre_ids : jax.Array
        Presynaptic neuron index for every synapse, shape ``(n_synapses,)``.
    post_ids : jax.Array
        Postsynaptic neuron index for every synapse, shape ``(n_synapses,)``.
    pre_spike : jax.Array
        Binary or boolean array indicating which presynaptic neurons fired,
        shape ``(n_pre,)``. Non-boolean arrays are treated as active when the
        value is nonzero.
    post_trace : jax.Array or brainunit.Quantity
        Trace values accumulated at each postsynaptic neuron, shape
        ``(n_post,)``. Converted to the same unit as *weight* before the
        update.
    w_min : jax.Array or brainunit.Quantity or None, optional
        Lower bound for weight clipping. Must carry the same unit as *weight*
        when units are used. Default is ``None`` (no lower bound).
    w_max : jax.Array or brainunit.Quantity or None, optional
        Upper bound for weight clipping. Must carry the same unit as *weight*
        when units are used. Default is ``None`` (no upper bound).
    backend : str or None, optional
        Compute backend to use for the underlying kernel. Accepted values
        depend on the platform (e.g., ``'numba'``, ``'pallas'``).
        When ``None``, the default backend for the current platform is used.

    Returns
    -------
    jax.Array or brainunit.Quantity
        Updated weight array with the same shape and unit as the input
        *weight*, after the additive plasticity update and optional clipping.

    Raises
    ------
    AssertionError
        If *weight*, *pre_ids*, or *post_ids* do not all have matching 1-D
        shapes, or if *pre_spike* / *post_trace* are not 1-D.

    See Also
    --------
    update_coo_on_binary_post : Analogous update driven by postsynaptic spikes.
    update_coo_on_binary_pre_p : Low-level XLA custom-kernel primitive used
        internally.

    Notes
    -----
    This operation is the **pre-synaptic** half of a spike-timing-dependent
    plasticity (STDP) rule expressed in COO sparse format.  In the standard
    pair-based STDP formulation, when presynaptic neuron ``j`` fires the
    update for every synapse ``(i, j)`` that exists in the connectivity is:

    ``W[i, j] <- W[i, j] + post_trace[i]``

    After the additive update, weights are clipped element-wise:

    ``W[i, j] <- clip(W[i, j], w_min, w_max)``

    Here ``post_trace`` is an eligibility trace that typically decays
    exponentially between postsynaptic spikes, so synapses that experienced a
    recent postsynaptic spike receive a larger update.

    In COO storage the loop iterates over every stored synapse index ``s``:
    if ``pre_spike[pre_ids[s]]`` is active, then
    ``weight[s] += post_trace[post_ids[s]]``.

    The kernel is dispatched through ``update_coo_on_binary_pre_p``, an
    :class:`~brainevent._op.XLACustomKernel` instance that selects among
    Numba (CPU) and Pallas/Triton (GPU) implementations
    according to *backend* and the runtime platform.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._coo.plasticity_binary import update_coo_on_binary_pre
        >>> weight = jnp.array([0.5, 0.3, 0.8])
        >>> pre_ids = jnp.array([0, 1, 0])
        >>> post_ids = jnp.array([1, 0, 2])
        >>> pre_spike = jnp.array([True, False])
        >>> post_trace = jnp.array([0.1, 0.2, 0.05])
        >>> new_w = update_coo_on_binary_pre(
        ...     weight, pre_ids, post_ids, pre_spike, post_trace,
        ...     w_min=0.0, w_max=1.0,
        ... )
    """
    weight, wunit = u.split_mantissa_unit(weight)
    post_trace = u.Quantity(post_trace).to(wunit).mantissa
    weight = u.maybe_decimal(
        _coo_on_pre_prim_call(
            weight, pre_ids, post_ids, pre_spike, post_trace,
            backend=backend
        )[0] * wunit
    )
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _coo_on_pre_numba_kernel(**kwargs):
    import numba

    @numba.njit(fastmath=True)
    def kernel(weight, pre_ids, post_ids, pre_spike, post_trace, out_w):
        out_w[:] = weight[:]
        for i in range(weight.shape[0]):
            if pre_spike[pre_ids[i]]:
                out_w[i] += post_trace[post_ids[i]]

    def run(weight, pre_ids, post_ids, pre_spike, post_trace):
        return numba_kernel(kernel, outs=kwargs['outs'])(weight, pre_ids, post_ids, pre_spike, post_trace)

    return run


def _coo_on_pre_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl

    n_syn = weight_info.shape[0]
    block_dim = generate_block_dim(n_syn, 512)
    block_dim = 32 if block_dim < 32 else block_dim

    if spike_info.dtype == jnp.bool_:
        def kernel(weight_ref, pre_ids_ref, post_ids_ref, spike_ref, trace_ref, out_w_ref):
            i = pl.program_id(0)
            i_start = i * block_dim
            mask = i_start + jnp.arange(block_dim) < n_syn
            pre_ids = pre_ids_ref[pl.ds(i_start, block_dim)]
            post_ids = post_ids_ref[pl.ds(i_start, block_dim)]
            safe_pre_ids = jnp.where(mask, pre_ids, 0)
            spikes = spike_ref[safe_pre_ids]
            active = mask & spikes
            safe_post_ids = jnp.where(active, post_ids, 0)
            traces = trace_ref[safe_post_ids]
            old_w = out_w_ref[pl.ds(i_start, block_dim)]
            new_w = jnp.where(active, old_w + traces, old_w)
            out_w_ref[pl.ds(i_start, block_dim)] = new_w
    else:
        def kernel(weight_ref, pre_ids_ref, post_ids_ref, spike_ref, trace_ref, out_w_ref):
            i = pl.program_id(0)
            i_start = i * block_dim
            mask = i_start + jnp.arange(block_dim) < n_syn
            pre_ids = pre_ids_ref[pl.ds(i_start, block_dim)]
            post_ids = post_ids_ref[pl.ds(i_start, block_dim)]
            safe_pre_ids = jnp.where(mask, pre_ids, 0)
            spikes = spike_ref[safe_pre_ids]
            active = mask & (spikes != 0.)
            safe_post_ids = jnp.where(active, post_ids, 0)
            traces = trace_ref[safe_post_ids]
            old_w = out_w_ref[pl.ds(i_start, block_dim)]
            new_w = jnp.where(active, old_w + traces, old_w)
            out_w_ref[pl.ds(i_start, block_dim)] = new_w

    def run(weight, pre_ids, post_ids, pre_spike, post_trace):
        if n_syn == 0:
            return (weight,)
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(n_syn, block_dim),),
            input_output_aliases={0: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(weight, pre_ids, post_ids, pre_spike, post_trace)

    return run


def _coo_on_pre_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    **kwargs,
):
    """Pure-JAX kernel for presynaptic COO plasticity update (all platforms)."""
    is_bool = (spike_info.dtype == jnp.bool_)

    def kernel(weight, pre_ids, post_ids, pre_spike, post_trace):
        if is_bool:
            active = pre_spike[pre_ids]
        else:
            active = pre_spike[pre_ids] != 0.
        delta = jnp.where(active, post_trace[post_ids], jnp.zeros_like(post_trace[post_ids]))
        return [weight + delta]

    return kernel


def _coo_on_pre_cuda_kernel(weight_info, spike_info, pre_ids_info, **kwargs):
    """TVM FFI CUDA kernel for presynaptic COO plasticity update.

    Dispatches to ``update_coo_on_pre{wt_sfx}{spk_sfx}`` compiled from
    ``plasticity_binary.cu``.

    Only int32 index dtype is supported.  Callers with int64 pre_ids should
    explicitly select ``backend='pallas'`` or ``backend='jax'``.
    """
    if pre_ids_info.dtype == jnp.int64:
        raise TypeError(
            "update_coo_on_binary_pre: the 'tvmffi' backend only supports "
            "int32 index arrays (pre_ids / post_ids).  "
            "Use backend='pallas' or backend='jax' for int64 indices."
        )

    register_tvm_cuda_from_file(
        module='coo_plasticity_binary',
        source=Path(__file__).parent.joinpath('plasticity_binary.cu'),
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
    kernel_name = f'coo_plasticity_binary.update_coo_on_pre{wt_sfx}{spk_suffix}'

    def kernel(weight, pre_ids, post_ids, pre_spike, post_trace):
        return jax.ffi.ffi_call(
            kernel_name,
            out_info,
            input_output_aliases={0: 0},
        )(weight, pre_ids, post_ids, pre_spike, post_trace)

    return kernel


def _coo_on_pre_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        nnz = max(1, int(n_pre * n_post * prob))
        pre_ids = np.random.randint(0, n_pre, nnz, dtype=np.int32)
        post_ids = np.random.randint(0, n_post, nnz, dtype=np.int32)
        weight = jnp.ones(nnz, dtype=dtype)
        if bool_event:
            pre_spike = jnp.asarray(np.random.rand(n_pre) > 0.5, dtype=jnp.bool_)
        else:
            pre_spike = jnp.asarray(np.random.rand(n_pre), dtype=dtype)
        post_trace = jnp.asarray(np.random.randn(n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append(
            BenchmarkConfig(name, (weight, jnp.asarray(pre_ids), jnp.asarray(post_ids), pre_spike, post_trace)))
    return configs


def _coo_on_pre_prim_call(
    weight,
    pre_ids,
    post_ids,
    pre_spike,
    post_trace,
    *,
    backend: Optional[str] = None
):
    """
    Validate inputs and dispatch the presynaptic COO plasticity primitive.

    This is the low-level call wrapper around
    :data:`update_coo_on_binary_pre_p`. It performs shape and dimensionality
    checks on every operand, constructs the required
    :class:`jax.ShapeDtypeStruct` metadata, and forwards the call to the
    ``XLACustomKernel`` instance which selects the appropriate backend
    kernel.

    Parameters
    ----------
    weight : jax.Array
        Unitless weight mantissa array, shape ``(n_synapses,)``.
    pre_ids : jax.Array
        Presynaptic neuron indices, shape ``(n_synapses,)``.
    post_ids : jax.Array
        Postsynaptic neuron indices, shape ``(n_synapses,)``.
    pre_spike : jax.Array
        Binary spike indicator for presynaptic neurons, shape ``(n_pre,)``.
    post_trace : jax.Array
        Unitless trace mantissa for postsynaptic neurons, shape ``(n_post,)``.
    backend : str or None, optional
        Backend override forwarded to the kernel dispatcher. When ``None``,
        the platform default is used.

    Returns
    -------
    tuple of jax.Array
        A single-element tuple ``(updated_weight,)`` where
        ``updated_weight`` has the same shape and dtype as *weight*.

    Raises
    ------
    AssertionError
        If *weight* is not 1-D, if the shapes of *weight*, *pre_ids*, and
        *post_ids* do not match, or if *pre_spike* / *post_trace* are not
        1-D.

    See Also
    --------
    update_coo_on_binary_pre : High-level wrapper that handles units and
        clipping before calling this function.
    update_coo_on_binary_pre_p : The ``XLACustomKernel`` instance invoked
        by this function.

    Notes
    -----
    Callers are expected to strip physical units from *weight* and
    *post_trace* before calling this function. Unit handling is performed by
    :func:`update_coo_on_binary_pre`.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._coo.plasticity_binary import _coo_on_pre_prim_call
        >>> w = jnp.array([0.5, 0.3, 0.8])
        >>> pre = jnp.array([0, 1, 0])
        >>> post = jnp.array([1, 0, 2])
        >>> spike = jnp.array([True, False])
        >>> trace = jnp.array([0.1, 0.2, 0.05])
        >>> (new_w,) = _coo_on_pre_prim_call(w, pre, post, spike, trace)
    """
    assert weight.ndim == 1, 'coo_on_pre only supports 1D weight.'
    assert weight.shape == pre_ids.shape == post_ids.shape, (
        f'weight shape ({weight.shape}), '
        f'pre_ids shape ({pre_ids.shape}), '
        f'and post_ids shape ({post_ids.shape}) '
        'should all match.'
    )
    assert pre_spike.ndim == 1, 'pre_spike should be 1D.'
    assert post_trace.ndim == 1, 'post_trace should be 1D.'
    return update_coo_on_binary_pre_p(
        weight, pre_ids, post_ids, pre_spike, post_trace,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        pre_ids_info=jax.ShapeDtypeStruct(pre_ids.shape, pre_ids.dtype),
        post_ids_info=jax.ShapeDtypeStruct(post_ids.shape, post_ids.dtype),
        spike_info=jax.ShapeDtypeStruct(pre_spike.shape, pre_spike.dtype),
        trace_info=jax.ShapeDtypeStruct(post_trace.shape, post_trace.dtype),
        backend=backend,
    )


update_coo_on_binary_pre_p = XLACustomKernel(
    'coo_on_pre',
    doc="""
Low-level XLA custom-kernel primitive for ``update_coo_on_binary_pre``.

This ``XLACustomKernel`` instance dispatches the COO weight update for
pre-synaptic binary plasticity operation to registered backends (``numba``,
``pallas``), using runtime shape/dtype metadata provided by the
high-level wrapper.

For each synapse ``i`` in COO format, if the presynaptic neuron fires
(``pre_spike[pre_ids[i]]`` is nonzero), the weight is updated as
``weight[i] += post_trace[post_ids[i]]``. This implements the presynaptic
half of a spike-timing-dependent plasticity (STDP) rule, where the update
magnitude depends on the postsynaptic eligibility trace.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``update_coo_on_binary_pre_p.available_backends(platform)``,
and the default backend can be configured with ``update_coo_on_binary_pre_p.set_default(platform, backend)``.

See Also
--------
update_coo_on_binary_pre : High-level user-facing function wrapper.
"""
)
update_coo_on_binary_pre_p.def_numba_kernel(_coo_on_pre_numba_kernel)
update_coo_on_binary_pre_p.def_pallas_kernel('gpu', _coo_on_pre_pallas_kernel)
update_coo_on_binary_pre_p.def_tvmffi_kernel('gpu', _coo_on_pre_cuda_kernel)
update_coo_on_binary_pre_p.def_kernel('jax_raw', 'cpu', _coo_on_pre_jax_kernel)
update_coo_on_binary_pre_p.def_kernel('jax_raw', 'gpu', _coo_on_pre_jax_kernel)
update_coo_on_binary_pre_p.def_kernel('jax_raw', 'tpu', _coo_on_pre_jax_kernel)
update_coo_on_binary_pre_p.def_call(_coo_on_pre_prim_call)
update_coo_on_binary_pre_p.def_tags('coo', 'plasticity')
update_coo_on_binary_pre_p.def_benchmark_data(_coo_on_pre_benchmark_data)


# =============================================================================
# COO On-Post Plasticity
# =============================================================================


@namescope
def update_coo_on_binary_post(
    weight: Union[u.Quantity, jax.Array],
    pre_ids: jax.Array,
    post_ids: jax.Array,
    pre_trace: Union[u.Quantity, jax.Array],
    post_spike: jax.Array,
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
    *,
    backend: Optional[str] = None,
):
    """
    Update synaptic weights in COO format driven by postsynaptic spike events.

    For each synapse *i* stored in COO format, if the postsynaptic neuron
    fires (``post_spike[post_ids[i]]`` is nonzero), the weight is updated
    according to:

    ``weight[i] = weight[i] + pre_trace[pre_ids[i]]``

    After the additive update, the result is clipped to ``[w_min, w_max]``
    when the bounds are provided. Physical units attached to ``weight`` and
    ``pre_trace`` are handled transparently via ``brainunit``.

    Parameters
    ----------
    weight : jax.Array or brainunit.Quantity
        Sparse synaptic weight values stored in COO format, shape
        ``(n_synapses,)``.
    pre_ids : jax.Array
        Presynaptic neuron index for every synapse, shape ``(n_synapses,)``.
    post_ids : jax.Array
        Postsynaptic neuron index for every synapse, shape ``(n_synapses,)``.
    pre_trace : jax.Array or brainunit.Quantity
        Trace values accumulated at each presynaptic neuron, shape
        ``(n_pre,)``. Converted to the same unit as *weight* before the
        update.
    post_spike : jax.Array
        Binary or boolean array indicating which postsynaptic neurons fired,
        shape ``(n_post,)``. Non-boolean arrays are treated as active when
        the value is nonzero.
    w_min : jax.Array or brainunit.Quantity or None, optional
        Lower bound for weight clipping. Must carry the same unit as *weight*
        when units are used. Default is ``None`` (no lower bound).
    w_max : jax.Array or brainunit.Quantity or None, optional
        Upper bound for weight clipping. Must carry the same unit as *weight*
        when units are used. Default is ``None`` (no upper bound).
    backend : str or None, optional
        Compute backend to use for the underlying kernel. Accepted values
        depend on the platform (e.g., ``'numba'``, ``'pallas'``).
        When ``None``, the default backend for the current platform is used.

    Returns
    -------
    jax.Array or brainunit.Quantity
        Updated weight array with the same shape and unit as the input
        *weight*, after the additive plasticity update and optional clipping.

    Raises
    ------
    AssertionError
        If *weight*, *pre_ids*, or *post_ids* do not all have matching 1-D
        shapes, or if *pre_trace* / *post_spike* are not 1-D.

    See Also
    --------
    update_coo_on_binary_pre : Analogous update driven by presynaptic spikes.
    update_coo_on_binary_post_p : Low-level XLA custom-kernel primitive used
        internally.

    Notes
    -----
    This operation is the **post-synaptic** half of a spike-timing-dependent
    plasticity (STDP) rule expressed in COO sparse format.  In the standard
    pair-based STDP formulation, when postsynaptic neuron ``i`` fires the
    update for every synapse ``(i, j)`` that exists in the connectivity is:

    ``W[i, j] <- W[i, j] + pre_trace[j]``

    After the additive update, weights are clipped element-wise:

    ``W[i, j] <- clip(W[i, j], w_min, w_max)``

    Here ``pre_trace`` is an eligibility trace that typically decays
    exponentially between presynaptic spikes, so synapses whose presynaptic
    neuron fired recently receive a larger update.

    In COO storage the loop iterates over every stored synapse index ``s``:
    if ``post_spike[post_ids[s]]`` is active, then
    ``weight[s] += pre_trace[pre_ids[s]]``.

    The kernel is dispatched through ``update_coo_on_binary_post_p``, an
    :class:`~brainevent._op.XLACustomKernel` instance that selects among
    Numba (CPU) and Pallas/Triton (GPU) implementations
    according to *backend* and the runtime platform.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._coo.plasticity_binary import update_coo_on_binary_post
        >>> weight = jnp.array([0.5, 0.3, 0.8])
        >>> pre_ids = jnp.array([0, 1, 0])
        >>> post_ids = jnp.array([1, 0, 2])
        >>> pre_trace = jnp.array([0.1, 0.2])
        >>> post_spike = jnp.array([True, False, True])
        >>> new_w = update_coo_on_binary_post(
        ...     weight, pre_ids, post_ids, pre_trace, post_spike,
        ...     w_min=0.0, w_max=1.0,
        ... )
    """
    weight, wunit = u.split_mantissa_unit(weight)
    pre_trace = u.Quantity(pre_trace).to(wunit).mantissa
    weight = u.maybe_decimal(
        _coo_on_post_prim_call(
            weight, pre_ids, post_ids, pre_trace, post_spike, backend=backend
        )[0] * wunit
    )
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _coo_on_post_numba_kernel(**kwargs):
    import numba

    @numba.njit(fastmath=True)
    def kernel(weight, pre_ids, post_ids, pre_trace, post_spike, out_w):
        out_w[:] = weight[:]
        for i in range(weight.shape[0]):
            if post_spike[post_ids[i]]:
                out_w[i] += pre_trace[pre_ids[i]]

    def run(weight, pre_ids, post_ids, pre_trace, post_spike):
        return numba_kernel(kernel, outs=kwargs['outs'])(weight, pre_ids, post_ids, pre_trace, post_spike)

    return run


def _coo_on_post_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl

    n_syn = weight_info.shape[0]
    block_dim = generate_block_dim(n_syn, 512)
    block_dim = 32 if block_dim < 32 else block_dim

    if spike_info.dtype == jnp.bool_:
        def kernel(weight_ref, pre_ids_ref, post_ids_ref, trace_ref, spike_ref, out_w_ref):
            i = pl.program_id(0)
            i_start = i * block_dim
            mask = i_start + jnp.arange(block_dim) < n_syn
            pre_ids = pre_ids_ref[pl.ds(i_start, block_dim)]
            post_ids = post_ids_ref[pl.ds(i_start, block_dim)]
            safe_post_ids = jnp.where(mask, post_ids, 0)
            spikes = spike_ref[safe_post_ids]
            active = mask & spikes
            safe_pre_ids = jnp.where(active, pre_ids, 0)
            traces = trace_ref[safe_pre_ids]
            old_w = out_w_ref[pl.ds(i_start, block_dim)]
            new_w = jnp.where(active, old_w + traces, old_w)
            out_w_ref[pl.ds(i_start, block_dim)] = new_w
    else:
        def kernel(weight_ref, pre_ids_ref, post_ids_ref, trace_ref, spike_ref, out_w_ref):
            i = pl.program_id(0)
            i_start = i * block_dim
            mask = i_start + jnp.arange(block_dim) < n_syn
            pre_ids = pre_ids_ref[pl.ds(i_start, block_dim)]
            post_ids = post_ids_ref[pl.ds(i_start, block_dim)]
            safe_post_ids = jnp.where(mask, post_ids, 0)
            spikes = spike_ref[safe_post_ids]
            active = mask & (spikes != 0.)
            safe_pre_ids = jnp.where(active, pre_ids, 0)
            traces = trace_ref[safe_pre_ids]
            old_w = out_w_ref[pl.ds(i_start, block_dim)]
            new_w = jnp.where(active, old_w + traces, old_w)
            out_w_ref[pl.ds(i_start, block_dim)] = new_w

    def run(weight, pre_ids, post_ids, pre_trace, post_spike):
        if n_syn == 0:
            return (weight,)
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(n_syn, block_dim),),
            input_output_aliases={0: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(weight, pre_ids, post_ids, pre_trace, post_spike)

    return run


def _coo_on_post_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    **kwargs,
):
    """Pure-JAX kernel for postsynaptic COO plasticity update (all platforms)."""
    is_bool = (spike_info.dtype == jnp.bool_)

    def kernel(weight, pre_ids, post_ids, pre_trace, post_spike):
        if is_bool:
            active = post_spike[post_ids]
        else:
            active = post_spike[post_ids] != 0.
        delta = jnp.where(active, pre_trace[pre_ids], jnp.zeros_like(pre_trace[pre_ids]))
        return [weight + delta]

    return kernel


def _coo_on_post_cuda_kernel(weight_info, spike_info, pre_ids_info, **kwargs):
    """TVM FFI CUDA kernel for postsynaptic COO plasticity update.

    Dispatches to ``update_coo_on_post{wt_sfx}{spk_sfx}`` compiled from
    ``plasticity_binary.cu``.

    Only int32 index dtype is supported.  Callers with int64 post_ids should
    explicitly select ``backend='pallas'`` or ``backend='jax'``.
    """
    if pre_ids_info.dtype == jnp.int64:
        raise TypeError(
            "update_coo_on_binary_post: the 'tvmffi' backend only supports "
            "int32 index arrays (pre_ids / post_ids).  "
            "Use backend='pallas' or backend='jax' for int64 indices."
        )

    register_tvm_cuda_from_file(
        module='coo_plasticity_binary',
        source=Path(__file__).parent.joinpath('plasticity_binary.cu'),
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
    kernel_name = f'coo_plasticity_binary.update_coo_on_post{wt_sfx}{spk_suffix}'

    def kernel(weight, pre_ids, post_ids, pre_trace, post_spike):
        return jax.ffi.ffi_call(
            kernel_name,
            out_info,
            input_output_aliases={0: 0},
        )(weight, pre_ids, post_ids, pre_trace, post_spike)

    return kernel


def _coo_on_post_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        nnz = max(1, int(n_pre * n_post * prob))
        pre_ids = np.random.randint(0, n_pre, nnz, dtype=np.int32)
        post_ids = np.random.randint(0, n_post, nnz, dtype=np.int32)
        weight = jnp.ones(nnz, dtype=dtype)
        pre_trace = jnp.asarray(np.random.randn(n_pre), dtype=dtype)
        if bool_event:
            post_spike = jnp.asarray(np.random.rand(n_post) > 0.5, dtype=jnp.bool_)
        else:
            post_spike = jnp.asarray(np.random.rand(n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append(
            BenchmarkConfig(name, (weight, jnp.asarray(pre_ids), jnp.asarray(post_ids), pre_trace, post_spike))
        )
    return configs


def _coo_on_post_prim_call(
    weight,
    pre_ids,
    post_ids,
    pre_trace,
    post_spike,
    *,
    backend: Optional[str] = None
):
    """
    Validate inputs and dispatch the postsynaptic COO plasticity primitive.

    This is the low-level call wrapper around
    :data:`update_coo_on_binary_post_p`. It performs shape and dimensionality
    checks on every operand, constructs the required
    :class:`jax.ShapeDtypeStruct` metadata, and forwards the call to the
    ``XLACustomKernel`` instance which selects the appropriate backend
    kernel.

    Parameters
    ----------
    weight : jax.Array
        Unitless weight mantissa array, shape ``(n_synapses,)``.
    pre_ids : jax.Array
        Presynaptic neuron indices, shape ``(n_synapses,)``.
    post_ids : jax.Array
        Postsynaptic neuron indices, shape ``(n_synapses,)``.
    pre_trace : jax.Array
        Unitless trace mantissa for presynaptic neurons, shape ``(n_pre,)``.
    post_spike : jax.Array
        Binary spike indicator for postsynaptic neurons, shape ``(n_post,)``.
    backend : str or None, optional
        Backend override forwarded to the kernel dispatcher. When ``None``,
        the platform default is used.

    Returns
    -------
    tuple of jax.Array
        A single-element tuple ``(updated_weight,)`` where
        ``updated_weight`` has the same shape and dtype as *weight*.

    Raises
    ------
    AssertionError
        If *weight* is not 1-D, if the shapes of *weight*, *pre_ids*, and
        *post_ids* do not match, or if *pre_trace* / *post_spike* are not
        1-D.

    See Also
    --------
    update_coo_on_binary_post : High-level wrapper that handles units and
        clipping before calling this function.
    update_coo_on_binary_post_p : The ``XLACustomKernel`` instance invoked
        by this function.

    Notes
    -----
    Callers are expected to strip physical units from *weight* and
    *pre_trace* before calling this function. Unit handling is performed by
    :func:`update_coo_on_binary_post`.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._coo.plasticity_binary import _coo_on_post_prim_call
        >>> w = jnp.array([0.5, 0.3, 0.8])
        >>> pre = jnp.array([0, 1, 0])
        >>> post = jnp.array([1, 0, 2])
        >>> trace = jnp.array([0.1, 0.2])
        >>> spike = jnp.array([True, False, True])
        >>> (new_w,) = _coo_on_post_prim_call(w, pre, post, trace, spike)
    """
    assert weight.ndim == 1, 'coo_on_post only supports 1D weight.'
    assert weight.shape == pre_ids.shape == post_ids.shape, (
        f'weight shape ({weight.shape}), '
        f'pre_ids shape ({pre_ids.shape}), '
        f'and post_ids shape ({post_ids.shape}) '
        'should all match.'
    )
    assert pre_trace.ndim == 1, 'pre_trace should be 1D.'
    assert post_spike.ndim == 1, 'post_spike should be 1D.'
    return update_coo_on_binary_post_p(
        weight, pre_ids, post_ids, pre_trace, post_spike,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        pre_ids_info=jax.ShapeDtypeStruct(pre_ids.shape, pre_ids.dtype),
        post_ids_info=jax.ShapeDtypeStruct(post_ids.shape, post_ids.dtype),
        trace_info=jax.ShapeDtypeStruct(pre_trace.shape, pre_trace.dtype),
        spike_info=jax.ShapeDtypeStruct(post_spike.shape, post_spike.dtype),
        backend=backend,
    )


update_coo_on_binary_post_p = XLACustomKernel(
    'coo_on_post',
    doc="""
Low-level XLA custom-kernel primitive for ``update_coo_on_binary_post``.

This ``XLACustomKernel`` instance dispatches the COO weight update for
post-synaptic binary plasticity operation to registered backends (``numba``,
``pallas``), using runtime shape/dtype metadata provided by the
high-level wrapper.

For each synapse ``i`` in COO format, if the postsynaptic neuron fires
(``post_spike[post_ids[i]]`` is nonzero), the weight is updated as
``weight[i] += pre_trace[pre_ids[i]]``. This implements the postsynaptic
half of a spike-timing-dependent plasticity (STDP) rule, where the update
magnitude depends on the presynaptic eligibility trace.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``update_coo_on_binary_post_p.available_backends(platform)``,
and the default backend can be configured with ``update_coo_on_binary_post_p.set_default(platform, backend)``.

See Also
--------
update_coo_on_binary_post : High-level user-facing function wrapper.
"""
)
update_coo_on_binary_post_p.def_numba_kernel(_coo_on_post_numba_kernel)
update_coo_on_binary_post_p.def_pallas_kernel('gpu', _coo_on_post_pallas_kernel)
update_coo_on_binary_post_p.def_tvmffi_kernel('gpu', _coo_on_post_cuda_kernel)
update_coo_on_binary_post_p.def_kernel('jax_raw', 'cpu', _coo_on_post_jax_kernel)
update_coo_on_binary_post_p.def_kernel('jax_raw', 'gpu', _coo_on_post_jax_kernel)
update_coo_on_binary_post_p.def_kernel('jax_raw', 'tpu', _coo_on_post_jax_kernel)
update_coo_on_binary_post_p.def_call(_coo_on_post_prim_call)
update_coo_on_binary_post_p.def_tags('coo', 'plasticity')
update_coo_on_binary_post_p.def_benchmark_data(_coo_on_post_benchmark_data)
