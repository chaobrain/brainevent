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

from pathlib import Path
from typing import Optional, Union

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, namescope
from brainevent._op import (
    XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig,
    register_tvm_cuda_from_file, jaxinfo_to_warpinfo,
)
from brainevent.config import get_numba_parallel

__all__ = [
    'update_dense_on_binary_pre',
    'update_dense_on_binary_pre_p',
    'update_dense_on_binary_post',
    'update_dense_on_binary_post_p'
]


@namescope
def update_dense_on_binary_pre(
    weight: Union[u.Quantity, jax.Array],
    pre_spike: jax.Array,
    post_trace: Union[u.Quantity, jax.Array],
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
    *,
    backend: Optional[str] = None,
):
    """
    Update synaptic weights based on presynaptic spike events and postsynaptic traces.

    Implements a plasticity rule where presynaptic spikes trigger weight
    updates modulated by postsynaptic trace values. For each presynaptic
    neuron ``i`` that fires, the update is:

    ``weight[i, :] += post_trace``

    The result is optionally clipped to ``[w_min, w_max]``.

    Parameters
    ----------
    weight : array_like or Quantity
        Synaptic weight matrix of shape ``(n_pre, n_post)``. Can be a
        ``brainunit`` quantity.
    pre_spike : jax.Array
        Binary or boolean array indicating presynaptic spike events,
        with shape ``(n_pre,)``.
    post_trace : array_like or Quantity
        Postsynaptic trace values with shape ``(n_post,)``. Must be
        convertible to the same unit as ``weight``.
    w_min : array_like, Quantity, or None, optional
        Lower bound for weight clipping. Must have the same units as
        ``weight``. If ``None``, no lower bound is applied.
    w_max : array_like, Quantity, or None, optional
        Upper bound for weight clipping. Must have the same units as
        ``weight``. If ``None``, no upper bound is applied.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : array_like or Quantity
        Updated weight matrix with the same shape and units as the input
        ``weight``.

    Raises
    ------
    AssertionError
        If ``weight`` is not 2-D, ``pre_spike`` is not 1-D,
        ``post_trace`` is not 1-D, or the dimensions do not match
        (``weight.shape[0] != pre_spike.shape[0]`` or
        ``weight.shape[1] != post_trace.shape[0]``).

    See Also
    --------
    update_dense_on_binary_post : Post-synaptic variant of this plasticity rule.

    Notes
    -----
    This implements a pre-synaptic spike-triggered plasticity rule. The
    weight update for each synapse ``(i, j)`` is:

    ``delta_W[i, j] = post_trace[j]`` if ``pre_spike[i]`` is active

    ``delta_W[i, j] = 0`` otherwise

    The updated weight matrix is then:

    ``W'[i, j] = clip(W[i, j] + delta_W[i, j], w_min, w_max)``

    where the clip operation is only applied when ``w_min`` or ``w_max``
    is not ``None``. This rule is commonly used in spike-timing-dependent
    plasticity (STDP) models, where the presynaptic spike arrival
    triggers potentiation or depression modulated by the postsynaptic
    trace.

    The function handles unit conversion internally, ensuring that
    ``post_trace`` is converted to the same unit as ``weight`` before
    computation.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> weight = jnp.zeros((3, 4), dtype=jnp.float32)
        >>> pre_spike = jnp.array([True, False, True])
        >>> post_trace = jnp.ones(4, dtype=jnp.float32) * 0.1
        >>> update_dense_on_binary_pre(weight, pre_spike, post_trace)
        Array([[0.1, 0.1, 0.1, 0.1],
               [0. , 0. , 0. , 0. ],
               [0.1, 0.1, 0.1, 0.1]], dtype=float32)
    """
    weight, wunit = u.split_mantissa_unit(weight)
    post_trace = u.Quantity(post_trace).to(wunit).mantissa
    weight = u.maybe_decimal(_dense_on_pre_prim_call(weight, pre_spike, post_trace, backend=backend)[0] * wunit)
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _dense_on_pre_numba_kernel(spike_info: jax.ShapeDtypeStruct, **kwargs):
    import numba

    if spike_info.dtype == jnp.bool_:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(weight, spike, trace, out_w):
            for i in numba.prange(spike.shape[0]):
                if spike[i]:
                    out_w[i] += trace

    else:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(weight, spike, trace, out_w):
            for i in numba.prange(spike.shape[0]):
                if spike[i] != 0.:
                    out_w[i] += trace

    def run(weight, spike, trace):
        return numba_kernel(kernel, outs=kwargs['outs'], input_output_aliases={0: 0})(weight, spike, trace)

    return run


def _dense_on_pre_pallas_kernel(weight_info, spike_info: jax.ShapeDtypeStruct, **kwargs):
    from jax.experimental import pallas as pl

    block_dim = generate_block_dim(weight_info.shape[1], 512)

    def kernel(weight_ref, spike_ref, trace_ref, out_w_ref):
        i_block = pl.program_id(0)
        col_start = i_block * block_dim
        mask = col_start + jnp.arange(block_dim) < weight_info.shape[1]
        trace_block = trace_ref[pl.ds(col_start, block_dim)]
        trace_block = jnp.where(mask, trace_block, 0.0).astype(out_w_ref.dtype)

        def loop_fn(i, _):
            spike = spike_ref[i]

            @pl.when(spike if spike_info.dtype == jnp.bool_ else spike != 0.)
            def run():
                row_val = out_w_ref[i, pl.ds(col_start, block_dim)]
                out_w_ref[i, pl.ds(col_start, block_dim)] = jnp.where(mask, row_val + trace_block, row_val)

        jax.lax.fori_loop(0, spike_ref.shape[0], loop_fn, None)

    def run(weight, spike, trace):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(weight_info.shape[1], block_dim),),
            input_output_aliases={0: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(weight, spike, trace)

    return run


def _dense_on_pre_cuda_kernel(weight_info, spike_info, **kwargs):
    register_tvm_cuda_from_file(
        module='dense_plasticity_on_pre',
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
    kernel_name = f'dense_plasticity_on_pre.update_dense_on_pre{wt_sfx}{spk_suffix}'

    def kernel(weight, spike, trace):
        return jax.ffi.ffi_call(
            kernel_name,
            out_info,
            input_output_aliases={0: 0},
        )(weight, spike, trace)

    return kernel


def _dense_on_pre_jax_kernel(**kwargs):
    def kernel(weight, spike, trace):
        s = spike.astype(weight.dtype) if spike.dtype == jnp.bool_ else spike
        return [weight + jnp.outer(s, trace)]

    return kernel


def _dense_on_pre_prim_call(weight, pre_spike, post_trace, backend: Optional[str] = None):
    """
    Low-level primitive call for pre-synaptic plasticity weight update.

    This function validates input shapes, constructs the output shape
    descriptor, and invokes the ``update_dense_on_binary_pre_p`` JAX
    primitive. Unlike :func:`update_dense_on_binary_pre`, this function
    operates on raw numerical arrays without ``brainunit`` unit handling
    or weight clipping.

    Parameters
    ----------
    weight : jax.Array
        Synaptic weight matrix of shape ``(n_pre, n_post)``.
    pre_spike : jax.Array
        Binary or boolean array of presynaptic spike events with shape
        ``(n_pre,)``.
    post_trace : jax.Array
        Postsynaptic trace values with shape ``(n_post,)``.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : list of jax.Array
        A single-element list containing the updated weight matrix with
        shape ``(n_pre, n_post)``.

    Raises
    ------
    AssertionError
        If ``weight`` is not 2-D, ``pre_spike`` is not 1-D,
        ``post_trace`` is not 1-D, or the dimensions do not match.

    See Also
    --------
    update_dense_on_binary_pre : High-level function with unit handling and clipping.

    Notes
    -----
    This is the low-level entry point that bypasses unit handling and
    weight clipping. The mathematical operation is:

    ``W'[i, j] = W[i, j] + post_trace[j]`` if ``pre_spike[i]`` is active

    ``W'[i, j] = W[i, j]`` otherwise

    No clipping is applied at this level; clipping is handled by the
    high-level :func:`update_dense_on_binary_pre` wrapper. The function
    returns a single-element list to conform to the JAX primitive output
    convention.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> weight = jnp.zeros((3, 4), dtype=jnp.float32)
        >>> pre_spike = jnp.array([True, False, True])
        >>> post_trace = jnp.ones(4, dtype=jnp.float32)
        >>> _dense_on_pre_prim_call(weight, pre_spike, post_trace)
    """
    assert weight.ndim == 2, f'dense_one_pre only support 2D weight. But got shape: {weight.shape}.'
    assert pre_spike.ndim == 1, f'pre_spike should be 1D, But got shape: {pre_spike.shape}.'
    assert post_trace.ndim == 1, f'post_trace should be 1D. But got shape: {post_trace.shape}.'
    assert weight.shape[0] == pre_spike.shape[0], (
        f'weight shape[0] ({weight.shape[0]}) should '
        f'match pre_spike shape[0] ({pre_spike.shape[0]}).'
    )
    assert weight.shape[1] == post_trace.shape[0], (
        f'weight shape[1] ({weight.shape[1]}) should '
        f'match post_trace shape[0] ({post_trace.shape[0]}).'
    )
    return update_dense_on_binary_pre_p(
        weight, pre_spike, post_trace,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        spike_info=jax.ShapeDtypeStruct(pre_spike.shape, pre_spike.dtype),
        trace_info=jax.ShapeDtypeStruct(post_trace.shape, post_trace.dtype),
        backend=backend,
    )


def _dense_on_pre_jvp_weight(w_dot, weight, pre_spike, post_trace, **kwargs):
    return [w_dot]


def _dense_on_pre_transpose_rule(ct, weight, pre_spike, post_trace, **kwargs):
    if ad.is_undefined_primal(pre_spike) or ad.is_undefined_primal(post_trace):
        raise ValueError("Cannot transpose with respect to pre_spike or post_trace.")
    ct = ct[0]
    if ad.is_undefined_primal(weight):
        return (ad.Zero(weight) if type(ct) is ad.Zero else ct), pre_spike, post_trace
    return weight, pre_spike, post_trace


def _update_dense_pre_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        weight = jnp.asarray(np.random.randn(n_pre, n_post), dtype=dtype)
        if bool_event:
            pre_spike = jnp.asarray(np.random.rand(n_pre) > 0.5, dtype=jnp.bool_)
        else:
            pre_spike = jnp.asarray(np.random.rand(n_pre), dtype=dtype)
        post_trace = jnp.asarray(np.random.randn(n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append(BenchmarkConfig(name, (weight, pre_spike, post_trace)))
    return configs


def _dense_on_pre_batching(args, axes, **kwargs):
    if axes == (0, None, None):
        weight, pre_spike, post_trace = args
        if pre_spike.dtype == jnp.bool_:
            mask = pre_spike.astype(weight.dtype)
        else:
            mask = (pre_spike != 0.).astype(weight.dtype)
        update = mask[:, None] * post_trace[None, :]
        return [weight + update[None, :, :]], [0]
    return general_batching_rule(update_dense_on_binary_pre_p, args, axes, **kwargs)


update_dense_on_binary_pre_p = XLACustomKernel(
    'dense_on_pre',
    doc="""
Low-level XLA custom-kernel primitive for ``update_dense_on_binary_pre``.

This ``XLACustomKernel`` instance dispatches the dense weight update for
pre-synaptic binary plasticity operation to registered backends (``numba``,
``pallas``), using runtime shape/dtype metadata provided by the high-level wrapper.

The operation updates synaptic weights based on presynaptic spike events and
postsynaptic trace values: for each presynaptic neuron ``i`` that fires,
``weight[i, :] += post_trace``.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``update_dense_on_binary_pre_p.available_backends(platform)``,
and the default backend can be configured with ``update_dense_on_binary_pre_p.set_default(platform, backend)``.

See Also
--------
update_dense_on_binary_pre : High-level user-facing function wrapper.
"""
)
def _dense_on_pre_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    trace_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    n_pre, n_post = weight_info.shape
    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    spike_warp_info = jaxinfo_to_warpinfo(spike_info)
    trace_warp_info = jaxinfo_to_warpinfo(trace_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if spike_info.dtype == jnp.bool_:
        @warp.kernel
        def kernel(weight: weight_warp_info,
                   spike: spike_warp_info,
                   trace: trace_warp_info,
                   out_w: out_warp_info):
            i, j = warp.tid()
            out_w[i, j] = weight[i, j] + trace[j] if spike[i] else weight[i, j]
    else:
        @warp.kernel
        def kernel(weight: weight_warp_info,
                   spike: spike_warp_info,
                   trace: trace_warp_info,
                   out_w: out_warp_info):
            i, j = warp.tid()
            out_w[i, j] = weight[i, j] + trace[j] if spike[i] != 0. else weight[i, j]

    def run(weight, spike, trace):
        out_info = kwargs['outs'][0]
        fn = jax_kernel(kernel, launch_dims=(n_pre, n_post), num_outputs=1, output_dims={'out_w': out_info.shape})
        return fn(weight, spike, trace)

    return run


def _dense_on_post_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    trace_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    n_pre, n_post = weight_info.shape
    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    trace_warp_info = jaxinfo_to_warpinfo(trace_info)
    spike_warp_info = jaxinfo_to_warpinfo(spike_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if spike_info.dtype == jnp.bool_:
        @warp.kernel
        def kernel(weight: weight_warp_info,
                   trace: trace_warp_info,
                   spike: spike_warp_info,
                   out_w: out_warp_info):
            i, j = warp.tid()
            out_w[i, j] = weight[i, j] + trace[i] if spike[j] else weight[i, j]
    else:
        @warp.kernel
        def kernel(weight: weight_warp_info,
                   trace: trace_warp_info,
                   spike: spike_warp_info,
                   out_w: out_warp_info):
            i, j = warp.tid()
            out_w[i, j] = weight[i, j] + trace[i] if spike[j] != 0. else weight[i, j]

    def run(weight, trace, spike):
        out_info = kwargs['outs'][0]
        fn = jax_kernel(kernel, launch_dims=(n_pre, n_post), num_outputs=1, output_dims={'out_w': out_info.shape})
        return fn(weight, trace, spike)

    return run

update_dense_on_binary_pre_p.def_numba_kernel(_dense_on_pre_numba_kernel)
update_dense_on_binary_pre_p.def_warp_kernel(_dense_on_pre_warp_kernel)
update_dense_on_binary_pre_p.def_pallas_kernel('gpu', _dense_on_pre_pallas_kernel)
update_dense_on_binary_pre_p.def_tvmffi_kernel('gpu', _dense_on_pre_cuda_kernel)
update_dense_on_binary_pre_p.def_kernel('jax_raw', 'cpu', _dense_on_pre_jax_kernel)
update_dense_on_binary_pre_p.def_kernel('jax_raw', 'gpu', _dense_on_pre_jax_kernel)
update_dense_on_binary_pre_p.def_kernel('jax_raw', 'tpu', _dense_on_pre_jax_kernel)
update_dense_on_binary_pre_p.def_jvp_rule2(_dense_on_pre_jvp_weight, None, None)
update_dense_on_binary_pre_p.def_transpose_rule(_dense_on_pre_transpose_rule)
update_dense_on_binary_pre_p.def_batching_rule(_dense_on_pre_batching)
update_dense_on_binary_pre_p.def_call(_dense_on_pre_prim_call)
update_dense_on_binary_pre_p.def_tags('dense', 'plasticity')
update_dense_on_binary_pre_p.def_benchmark_data(_update_dense_pre_benchmark_data)


@namescope
def update_dense_on_binary_post(
    weight: Union[u.Quantity, jax.Array],
    pre_trace: Union[u.Quantity, jax.Array],
    post_spike: jax.Array,
    w_min: Optional[Union[u.Quantity, jax.Array]] = None,
    w_max: Optional[Union[u.Quantity, jax.Array]] = None,
    *,
    backend: Optional[str] = None,
):
    """
    Update synaptic weights based on postsynaptic spike events and presynaptic traces.

    Implements a plasticity rule where postsynaptic spikes trigger weight
    updates modulated by presynaptic trace values. For each postsynaptic
    neuron ``j`` that fires, the update is:

    ``weight[:, j] += pre_trace``

    The result is optionally clipped to ``[w_min, w_max]``.

    Parameters
    ----------
    weight : array_like or Quantity
        Synaptic weight matrix of shape ``(n_pre, n_post)``. Can be a
        ``brainunit`` quantity.
    pre_trace : array_like or Quantity
        Presynaptic trace values with shape ``(n_pre,)``. Must be
        convertible to the same unit as ``weight``.
    post_spike : jax.Array
        Binary or boolean array indicating postsynaptic spike events,
        with shape ``(n_post,)``.
    w_min : array_like, Quantity, or None, optional
        Lower bound for weight clipping. Must have the same units as
        ``weight``. If ``None``, no lower bound is applied.
    w_max : array_like, Quantity, or None, optional
        Upper bound for weight clipping. Must have the same units as
        ``weight``. If ``None``, no upper bound is applied.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : array_like or Quantity
        Updated weight matrix with the same shape and units as the input
        ``weight``.

    Raises
    ------
    AssertionError
        If ``weight`` is not 2-D, ``pre_trace`` is not 1-D,
        ``post_spike`` is not 1-D, or the dimensions do not match
        (``weight.shape[0] != pre_trace.shape[0]`` or
        ``weight.shape[1] != post_spike.shape[0]``).

    See Also
    --------
    update_dense_on_binary_pre : Pre-synaptic variant of this plasticity rule.

    Notes
    -----
    This implements a post-synaptic spike-triggered plasticity rule. The
    weight update for each synapse ``(i, j)`` is:

    ``delta_W[i, j] = pre_trace[i]`` if ``post_spike[j]`` is active

    ``delta_W[i, j] = 0`` otherwise

    The updated weight matrix is then:

    ``W'[i, j] = clip(W[i, j] + delta_W[i, j], w_min, w_max)``

    where the clip operation is only applied when ``w_min`` or ``w_max``
    is not ``None``. This rule is commonly used in spike-timing-dependent
    plasticity (STDP) models, where the postsynaptic spike arrival
    triggers potentiation or depression modulated by the presynaptic
    trace.

    The function handles unit conversion internally, ensuring that
    ``pre_trace`` is converted to the same unit as ``weight`` before
    computation.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> weight = jnp.zeros((3, 4), dtype=jnp.float32)
        >>> pre_trace = jnp.ones(3, dtype=jnp.float32) * 0.1
        >>> post_spike = jnp.array([True, False, True, False])
        >>> update_dense_on_binary_post(weight, pre_trace, post_spike)
        Array([[0.1, 0. , 0.1, 0. ],
               [0.1, 0. , 0.1, 0. ],
               [0.1, 0. , 0.1, 0. ]], dtype=float32)
    """
    weight, wunit = u.split_mantissa_unit(weight)
    pre_trace = u.Quantity(pre_trace).to(wunit).mantissa
    weight = u.maybe_decimal(_dense_on_post_prim_call(weight, pre_trace, post_spike, backend=backend)[0] * wunit)
    weight = u.math.clip(weight, w_min, w_max)
    return weight


def _dense_on_post_numba_kernel(spike_info: jax.ShapeDtypeStruct, **kwargs):
    import numba

    if spike_info.dtype == jnp.bool_:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(weight, trace, spike, out_w):
            for i in numba.prange(spike.shape[0]):
                if spike[i]:
                    out_w[:, i] += trace

    else:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(weight, trace, spike, out_w):
            for i in numba.prange(spike.shape[0]):
                if spike[i] != 0.:
                    out_w[:, i] += trace

    def run(weight, trace, spike):
        return numba_kernel(kernel, outs=kwargs['outs'], input_output_aliases={0: 0})(weight, trace, spike)

    return run


def _dense_on_post_pallas_kernel(weight_info, spike_info: jax.ShapeDtypeStruct, **kwargs):
    from jax.experimental import pallas as pl

    block_dim = generate_block_dim(weight_info.shape[0], 512)

    def kernel(weight_ref, trace_ref, spike_ref, out_w_ref):
        i_block = pl.program_id(0)
        row_start = i_block * block_dim
        mask = row_start + jnp.arange(block_dim) < weight_info.shape[0]
        trace_block = trace_ref[pl.ds(row_start, block_dim)]
        trace_block = jnp.where(mask, trace_block, 0.0).astype(out_w_ref.dtype)

        def loop_fn(i, _):
            spike = spike_ref[i]

            @pl.when(spike if spike_info.dtype == jnp.bool_ else spike != 0.)
            def run():
                col_val = out_w_ref[pl.ds(row_start, block_dim), i]
                out_w_ref[pl.ds(row_start, block_dim), i] = jnp.where(mask, col_val + trace_block, col_val)

        jax.lax.fori_loop(0, spike_ref.shape[0], loop_fn, None)

    def run(weight, trace, spike):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(weight_info.shape[0], block_dim),),
            input_output_aliases={0: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(weight, trace, spike)

    return run


def _dense_on_post_cuda_kernel(weight_info, spike_info, **kwargs):
    register_tvm_cuda_from_file(
        module='dense_plasticity_on_post',
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
    kernel_name = f'dense_plasticity_on_post.update_dense_on_post{wt_sfx}{spk_suffix}'

    def kernel(weight, trace, spike):
        return jax.ffi.ffi_call(
            kernel_name,
            out_info,
            input_output_aliases={0: 0},
        )(weight, trace, spike)

    return kernel


def _dense_on_post_jax_kernel(**kwargs):
    def kernel(weight, trace, spike):
        s = spike.astype(weight.dtype) if spike.dtype == jnp.bool_ else spike
        return [weight + jnp.outer(trace, s)]

    return kernel


def _dense_on_post_prim_call(weight, pre_trace, post_spike, backend: Optional[str] = None):
    """
    Low-level primitive call for post-synaptic plasticity weight update.

    This function validates input shapes, constructs the output shape
    descriptor, and invokes the ``update_dense_on_binary_post_p`` JAX
    primitive. Unlike :func:`update_dense_on_binary_post`, this function
    operates on raw numerical arrays without ``brainunit`` unit handling
    or weight clipping.

    Parameters
    ----------
    weight : jax.Array
        Synaptic weight matrix of shape ``(n_pre, n_post)``.
    pre_trace : jax.Array
        Presynaptic trace values with shape ``(n_pre,)``.
    post_spike : jax.Array
        Binary or boolean array of postsynaptic spike events with shape
        ``(n_post,)``.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : list of jax.Array
        A single-element list containing the updated weight matrix with
        shape ``(n_pre, n_post)``.

    Raises
    ------
    AssertionError
        If ``weight`` is not 2-D, ``pre_trace`` is not 1-D,
        ``post_spike`` is not 1-D, or the dimensions do not match.

    See Also
    --------
    update_dense_on_binary_post : High-level function with unit handling and clipping.

    Notes
    -----
    This is the low-level entry point that bypasses unit handling and
    weight clipping. The mathematical operation is:

    ``W'[i, j] = W[i, j] + pre_trace[i]`` if ``post_spike[j]`` is active

    ``W'[i, j] = W[i, j]`` otherwise

    No clipping is applied at this level; clipping is handled by the
    high-level :func:`update_dense_on_binary_post` wrapper. The function
    returns a single-element list to conform to the JAX primitive output
    convention.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> weight = jnp.zeros((3, 4), dtype=jnp.float32)
        >>> pre_trace = jnp.ones(3, dtype=jnp.float32)
        >>> post_spike = jnp.array([True, False, True, False])
        >>> _dense_on_post_prim_call(weight, pre_trace, post_spike)
    """
    assert weight.ndim == 2, f'dense_one_pre only support 2D weight. But got shape: {weight.shape}.'
    assert pre_trace.ndim == 1, f'pre_trace should be 1D. But got shape: {pre_trace.shape}.'
    assert post_spike.ndim == 1, f'post_spike should be 1D. But got shape: {post_spike.shape}.'
    assert weight.shape[0] == pre_trace.shape[0], (f'weight shape[0] ({weight.shape[0]}) should '
                                                   f'match pre_trace shape[0] ({pre_trace.shape[0]}).')
    assert weight.shape[1] == post_spike.shape[0], (f'weight shape[1] ({weight.shape[1]}) should '
                                                    f'match post_spike shape[0] ({post_spike.shape[0]}).')
    return update_dense_on_binary_post_p(
        weight, pre_trace, post_spike,
        outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        spike_info=jax.ShapeDtypeStruct(post_spike.shape, post_spike.dtype),
        trace_info=jax.ShapeDtypeStruct(pre_trace.shape, pre_trace.dtype),
        backend=backend,
    )


def _dense_on_post_jvp_weight(w_dot, weight, pre_trace, post_spike, **kwargs):
    return [w_dot]


def _dense_on_post_transpose_rule(ct, weight, pre_trace, post_spike, **kwargs):
    if ad.is_undefined_primal(pre_trace) or ad.is_undefined_primal(post_spike):
        raise ValueError("Cannot transpose with respect to pre_trace or post_spike.")
    ct = ct[0]
    if ad.is_undefined_primal(weight):
        return (ad.Zero(weight) if type(ct) is ad.Zero else ct), pre_trace, post_spike
    return weight, pre_trace, post_spike


def _dense_on_post_batching(args, axes, **kwargs):
    if axes == (0, None, None):
        weight, pre_trace, post_spike = args
        if post_spike.dtype == jnp.bool_:
            mask = post_spike.astype(weight.dtype)
        else:
            mask = (post_spike != 0.).astype(weight.dtype)
        update = pre_trace[:, None] * mask[None, :]
        return [weight + update[None, :, :]], [0]
    return general_batching_rule(update_dense_on_binary_post_p, args, axes, **kwargs)


def _update_dense_post_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for bool_event in (True, False):
        weight = jnp.asarray(np.random.randn(n_pre, n_post), dtype=dtype)
        pre_trace = jnp.asarray(np.random.randn(n_pre), dtype=dtype)
        if bool_event:
            post_spike = jnp.asarray(np.random.rand(n_post) > 0.5, dtype=jnp.bool_)
        else:
            post_spike = jnp.asarray(np.random.rand(n_post), dtype=dtype)
        name = f"{'bool' if bool_event else 'float'}"
        configs.append(BenchmarkConfig(name, (weight, pre_trace, post_spike)))
    return configs


update_dense_on_binary_post_p = XLACustomKernel(
    'dense_on_post',
    doc="""
Low-level XLA custom-kernel primitive for ``update_dense_on_binary_post``.

This ``XLACustomKernel`` instance dispatches the dense weight update for
post-synaptic binary plasticity operation to registered backends (``numba``,
``pallas``), using runtime shape/dtype metadata provided by the high-level wrapper.

The operation updates synaptic weights based on postsynaptic spike events and
presynaptic trace values: for each postsynaptic neuron ``j`` that fires,
``weight[:, j] += pre_trace``.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``update_dense_on_binary_post_p.available_backends(platform)``,
and the default backend can be configured with ``update_dense_on_binary_post_p.set_default(platform, backend)``.

See Also
--------
update_dense_on_binary_post : High-level user-facing function wrapper.
"""
)
update_dense_on_binary_post_p.def_numba_kernel(_dense_on_post_numba_kernel)
update_dense_on_binary_post_p.def_warp_kernel(_dense_on_post_warp_kernel)
update_dense_on_binary_post_p.def_pallas_kernel('gpu', _dense_on_post_pallas_kernel)
update_dense_on_binary_post_p.def_tvmffi_kernel('gpu', _dense_on_post_cuda_kernel)
update_dense_on_binary_post_p.def_kernel('jax_raw', 'cpu', _dense_on_post_jax_kernel)
update_dense_on_binary_post_p.def_kernel('jax_raw', 'gpu', _dense_on_post_jax_kernel)
update_dense_on_binary_post_p.def_kernel('jax_raw', 'tpu', _dense_on_post_jax_kernel)
update_dense_on_binary_post_p.def_jvp_rule2(_dense_on_post_jvp_weight, None, None)
update_dense_on_binary_post_p.def_transpose_rule(_dense_on_post_transpose_rule)
update_dense_on_binary_post_p.def_batching_rule(_dense_on_post_batching)
update_dense_on_binary_post_p.def_call(_dense_on_post_prim_call)
update_dense_on_binary_post_p.def_tags('dense', 'plasticity')
update_dense_on_binary_post_p.def_benchmark_data(_update_dense_post_benchmark_data)
