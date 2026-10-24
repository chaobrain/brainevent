# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
from typing import Optional, Tuple, Union

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import check_fixed_conn_num_shape, namescope
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig, load_cuda_file
from brainevent.config import get_backend, get_numba_parallel
from .float import fcnmv, fcnmm
from brainevent._op.util import (
    active_sum, cuda_event_abi, dtype_suffix, event_active, event_factor, NUMBA_EVENT_FASTMATH, numba_event_mask,
    select_active,
)

__all__ = [
    'binary_fcnmv',
    'binary_fcnmv_p',
    'binary_fcnmm',
    'binary_fcnmm_p',
]


_EVENT_FASTMATH = NUMBA_EVENT_FASTMATH
_numba_event_mask = numba_event_mask


@namescope(static_argnames=['shape', 'transpose', 'weighted'])
def binary_fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    spikes: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
    weighted: bool = False,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven sparse matrix--vector product with fixed connection number.

    Computes ``y = W @ s`` (or ``y = W^T @ s`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``s`` is a binary spike vector.  Only the connections to
    spiking neurons contribute to the result, making this operation
    event-driven.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    spikes : jax.Array or u.Quantity
        Binary spike vector.  Entries are treated as active when ``True``
        (boolean) or ``!= 0`` (numeric; negative values and NaN are active).
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool, optional
        If ``False`` (default), compute ``W @ s`` (fixed post-synaptic
        connections).  If ``True``, compute ``W^T @ s`` (fixed
        pre-synaptic connections, scatter mode).
    backend : str or None, optional
        Execution backend override (``'numba'``,
        ``'pallas'``, ``'cuda_raw'``, or ``None`` for automatic selection).
    weighted : bool, optional
        If ``True``, multiply each weight by the event value instead of its
        activity mask (:class:`~brainevent.WeightedArray` semantics); zero
        events are still skipped. Defaults to ``False``.

    Returns
    -------
    jax.Array or u.Quantity
        Result vector.  Shape is ``(num_pre,)`` when ``transpose=False``
        or ``(num_post,)`` when ``transpose=True``.

    See Also
    --------
    binary_fcnmm : Event-driven sparse matrix--matrix product.
    fcnmv : Float (non-event-driven) sparse matrix--vector product.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    The event-driven matrix-vector product computes (when ``transpose=False``):

        ``y[i] = sum_{k=0}^{n_conn-1} weights[i, k] * 1_{spikes[indices[i, k]] active}``

    where "active" means ``True`` for boolean spikes or ``!= 0`` for numeric spikes.
    For homogeneous weights (``weights`` has shape ``(1,)``):

        ``y[i] = w * sum_{k=0}^{n_conn-1} 1_{spikes[indices[i, k]] active}``

    When ``transpose=True``, the scatter-mode product computes:

        ``y[indices[i, k]] += weights[i, k] * 1_{spikes[i] active}``    for all ``i, k``

    The event-driven formulation skips inactive neurons entirely, making the
    computation efficient when the spike vector is sparse.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.binary import binary_fcnmv
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)  # homogeneous
        >>> indices = jnp.array([[0, 1], [1, 2]])      # (2, 2)
        >>> spikes = jnp.array([True, False, True])
        >>> y = binary_fcnmv(weights, indices, spikes, shape=(2, 3))
        >>> print(y)
        [1. 1.]
    """
    weights, w_unit = u.split_mantissa_unit(weights)
    spikes, v_unit = u.split_mantissa_unit(spikes)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = binary_fcnmv_p_call(
        weights,
        indices,
        spikes,
        shape=shape,
        transpose=transpose,
        backend=backend,
        weighted=weighted,
    )[0]
    return u.maybe_decimal(r * v_unit * w_unit)


def _binary_fcnmv_spike_activity(
    spikes: jax.Array,
    dtype: jnp.dtype,
    weighted: bool = False,
) -> jax.Array:
    return event_factor(u.math.asarray(spikes), dtype=dtype, weighted=weighted)


def _ell_weighted_matvec_numba_kernel(weight_info, transpose: bool):
    import numba

    if transpose:
        if weight_info.size == 1:
            @numba.njit(fastmath=_EVENT_FASTMATH)
            def ell_mv(weights, indices, spikes, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0:
                        wsp = w * sp
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += wsp
        else:
            @numba.njit(fastmath=_EVENT_FASTMATH)
            def ell_mv(weights, indices, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0:
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += weights[i, j] * sp
    else:
        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def ell_mv(weights, indices, spikes, posts):
                w = weights[0]
                for i in numba.prange(indices.shape[0]):
                    r = 0.
                    hit = False
                    for j in range(indices.shape[1]):
                        sp = spikes[indices[i, j]]
                        if sp != 0:
                            r += sp
                            hit = True
                    # Scale only rows with an active event: ``w * 0`` is NaN
                    # for a NaN/Inf weight.
                    posts[i] = w * r if hit else 0.
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def ell_mv(weights, indices, spikes, posts):
                for i in numba.prange(indices.shape[0]):
                    r = 0.
                    for j in range(indices.shape[1]):
                        sp = spikes[indices[i, j]]
                        if sp != 0:
                            r += weights[i, j] * sp
                    posts[i] = r
    return ell_mv


def _ell_binary_matvec_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    transpose: bool,
    weighted: bool = False,
    **kwargs
):
    import numba

    # Numba cannot read float16/bfloat16; such events use the boolean kernel on
    # their mask (binary) or are read as float32 (weighted).
    mask_events = _numba_event_mask(spike_info.dtype)
    bool_events = spike_info.dtype == jnp.bool_ or mask_events

    if weighted:
        weighted_mv = _ell_weighted_matvec_numba_kernel(weight_info, transpose)

        def weighted_kernel(weights, indices, spikes):
            if mask_events:
                spikes = spikes.astype(jnp.float32)
            return numba_kernel(weighted_mv, outs=kwargs['outs'])(weights, indices, spikes)

        return weighted_kernel

    if transpose:
        if weight_info.size == 1:
            if bool_events:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w
            else:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i] != 0:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w
        else:
            if bool_events:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]
            else:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i] != 0:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]

    else:
        if weight_info.size == 1:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    w = weights[0]
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spikes[index]:
                                r += w
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    spk_bool = spikes != 0
                    w = weights[0]
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spk_bool[index]:
                                r += w
                        posts[i] = r
        else:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spikes[index]:
                                r += weights[i, j]
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    spk_bool = spikes != 0
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spk_bool[index]:
                                r += weights[i, j]
                        posts[i] = r

    def kernel(weights, indices, spikes):
        if mask_events:
            spikes = event_active(spikes)
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, spikes)

    return kernel


def _ell_binary_matvec_cuda_kernel(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    weighted: bool = False,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    weight_info = kwargs['weight_info']
    row_sfx = dtype_suffix(weight_info.dtype)
    row_homo = weight_info.size == 1

    if not transpose:
        raise NotImplementedError(
            'The CUDA transpose=False binary row-gather matvec was removed. '
            'For the unfavorable W @ s direction the FCN classes route through '
            'the perm-fused CSR kernel (brainevent.binary_csrmv_indexed); the '
            'dispatcher selects it automatically.'
        )

    # Scatter mode: if is_active(spikes[i]) -> output[indices[i,k]] += weights[i,k]
    # (times spikes[i] for weighted events).
    mode_sfx = '_homo' if row_homo else '_hetero'
    if weighted:
        # Weighted scatter kernels read the event buffer in the weight dtype.
        abi = cuda_event_abi(spike_info.dtype, weighted=True, value_dtype=weight_info.dtype)
        kernel_name = f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}{abi.suffix}{row_sfx}'

        def weighted_kernel(weights, indices, spikes):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, abi.prepare(spikes))

        return weighted_kernel

    kernel_name = f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_bool{row_sfx}'

    def kernel(weights, indices, spikes):
        # The binary scatter kernel only reads a bool event buffer.
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, event_active(spikes))

    return kernel


def _ell_binary_matvec_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    weighted: bool = False,
    **kwargs,
):
    """Pure JAX reference implementation for benchmarking comparison."""
    n_pre, n_post = shape

    def kernel(weights, indices, spikes):
        active = event_active(spikes)
        spk_f = _binary_fcnmv_spike_activity(spikes, weights.dtype, weighted)
        w = jnp.broadcast_to(weights, indices.shape)
        # Select, never multiply by a 0/1 mask: a NaN/Inf weight that only
        # meets inactive events contributes exactly 0.
        if transpose:
            # Scatter: y[indices[i,k]] += where(active[i], weights[i,k] * f[i], 0)
            terms = select_active(active[:, None], w * spk_f[:, None] if weighted else w)
            return jax.ops.segment_sum(terms.ravel(), indices.ravel(), num_segments=n_post),
        else:
            # Gather: y[i] = sum_k where(active[indices[i,k]], weights[i,k] * f[indices[i,k]], 0)
            terms = select_active(active[indices], w * spk_f[indices] if weighted else w)
            return jnp.sum(terms, axis=1),

    return kernel


def _binary_fcnmv_jvp_spikes(
    spk_dot,
    weights,
    indices,
    spikes,
    *,
    shape,
    transpose,
    **kwargs
):
    # The output has the weight dtype; the event tangent may be wider (e.g. float64).
    return fcnmv(weights, indices, spk_dot, shape=shape, transpose=transpose).astype(weights.dtype),


def _binary_fcnmv_jvp_weights(
    w_dot,
    weights,
    indices,
    spikes,
    *,
    shape,
    transpose,
    weighted=False,
    **kwargs
):
    return binary_fcnmv_p_call(
        w_dot,
        indices,
        spikes,
        shape=shape,
        transpose=transpose,
        backend=kwargs['backend'],
        weighted=weighted,
    )


def _binary_fcnmv_transpose_rule(
    ct,
    weights,
    indices,
    spikes,
    *,
    shape,
    transpose,
    weight_info,
    weighted=False,
    **kwargs
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # dL/dspk = dL/dy * dy/dspk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(spikes):
        if type(ct) is ad.Zero:
            ct_spk = ad.Zero(spikes)
        else:
            ct_spk = fcnmv(weights, indices, ct, shape=shape, transpose=not transpose)
            ct_spk = ct_spk.astype(spikes.aval.dtype)
        return weights, indices, ct_spk

    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_gmax = ad.Zero(weights)
        elif homo:
            # scalar
            ct_gmax = binary_fcnmv_p_call(
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                spikes,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
                weighted=weighted,
            )
            # <ct, r> with r = prim(1): outputs reached by no active event
            # (r == 0) contribute exactly 0, even where ``ct`` is NaN/Inf
            # (select, not ``ct * 0``).
            ct_gmax = active_sum(ct, ct_gmax[0], weighted=True).reshape(*weight_info.shape)
        else:
            active = event_active(spikes)
            spk_f = _binary_fcnmv_spike_activity(spikes, weight_info.dtype, weighted)
            # The cotangent of a weight that meets an inactive event is exactly 0.
            if transpose:
                ct_gmax = select_active(active[:, None], spk_f[:, None] * ct[indices])
            else:
                ct_gmax = select_active(active[indices], ct[:, None] * spk_f[indices])
    return ct_gmax, indices, spikes


def _binary_fcnmv_batching(args, axes, **kwargs):
    # The cuda_raw scatter matmat returns the raw batch-first layout
    # ``(batch, num_post)``; every other backend returns ``(out, batch)``.
    out_axis = 0 if _binary_fcnmm_uses_raw_batch_first(
        transpose=kwargs['transpose'], backend=kwargs['backend']
    ) else 1
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_fcnmm_p_call(
            args[0],
            args[1],
            args[2].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
            weighted=kwargs.get('weighted', False),
        )
        return r, [out_axis]
    elif tuple(axes) == (None, None, 1):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_fcnmm_p_call(
            args[0],
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
            weighted=kwargs.get('weighted', False),
        )
        return r, [out_axis]
    else:
        return general_batching_rule(binary_fcnmv_p, args, axes, **kwargs)


def _binary_fcnmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                n_conn = max(1, int(n_post * prob))
                indices = jnp.asarray(np.random.randint(0, n_post, (n_pre, n_conn), dtype=np.int32))
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.ones((n_pre, n_conn), dtype=dtype)
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    spikes = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    spikes = jnp.asarray(np.random.rand(v_size), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                yield BenchmarkConfig(
                    name,
                    (weights, indices, spikes),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )


def binary_fcnmv_p_call(
    weights: jax.Array,
    indices: jax.Array,
    spikes: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
    weighted: bool = False,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for the ELL event-driven sparse matrix--vector
    product with fixed connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba, jax, or CUDA) without performing any
    physical-unit bookkeeping.  It is typically called from
    :func:`binary_fcnmv` or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    spikes : jax.Array
        Binary spike vector (boolean or float).
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool, optional
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).  Default is ``False``.
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, ``'cuda_raw'``, or
        ``None``).
    weighted : bool, optional
        Multiply weights by the event value instead of its activity mask.
        Defaults to ``False``.

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result vector.

    See Also
    --------
    binary_fcnmv : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, spikes, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return binary_fcnmv_p(
        weights,
        indices,
        spikes,
        outs=[out],
        shape=shape,
        transpose=transpose,
        weighted=weighted,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        backend=backend,
    )


binary_fcnmv_p = XLACustomKernel(
    'ell_binary_matvec',
    doc="""
Low-level XLA custom-kernel primitive for the ELL ``binary_fcnmv``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven)
fixed-connection matrix-vector multiplication operation to registered backends
(``numba``, ``jax``, ``cuda_raw``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. The event-driven formulation only processes
active (spiking) neurons, skipping zero entries for efficiency.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

The primitive takes three positional arguments ``(weights, indices, spikes)``.
The CUDA ``transpose=False`` row-gather path was removed; the unfavorable
``W @ s`` direction routes through the perm-fused CSR kernel
``brainevent.binary_csrmv_indexed``.

Available backends can be queried with ``binary_fcnmv_p.available_backends(platform)``,
and the default backend can be configured with ``binary_fcnmv_p.set_default(platform, backend)``.

See Also
--------
binary_fcnmv : High-level user-facing function wrapper.
"""
)


binary_fcnmv_p.def_numba_kernel(_ell_binary_matvec_numba_kernel)
binary_fcnmv_p.def_cuda_raw_kernel(_ell_binary_matvec_cuda_kernel, asdefault=True)
binary_fcnmv_p.def_kernel('jax_raw', 'cpu', _ell_binary_matvec_jax_kernel)
binary_fcnmv_p.def_kernel('jax_raw', 'gpu', _ell_binary_matvec_jax_kernel)
binary_fcnmv_p.def_kernel('jax_raw', 'tpu', _ell_binary_matvec_jax_kernel)

binary_fcnmv_p.def_jvp_rule2(
    _binary_fcnmv_jvp_weights,
    None,
    _binary_fcnmv_jvp_spikes,
)
binary_fcnmv_p.def_transpose_rule(_binary_fcnmv_transpose_rule)
binary_fcnmv_p.def_batching_rule(_binary_fcnmv_batching)
binary_fcnmv_p.def_call(binary_fcnmv_p_call)
binary_fcnmv_p.def_tags('fcn', 'binary')
binary_fcnmv_p.def_benchmark_data(_binary_fcnmv_benchmark_data)


@namescope(static_argnames=['shape', 'transpose', 'weighted'])
def binary_fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
    weighted: bool = False,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven sparse matrix--matrix product with fixed connection number.

    Computes ``Y = W @ M`` (or ``Y = W^T @ M`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``M`` is a dense binary event matrix.  Only the connections
    to active (spiking) entries contribute to the result.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    matrix : jax.Array or u.Quantity
        Dense binary event matrix of shape ``(k, n)`` where ``k`` matches
        the appropriate sparse-matrix dimension.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool
        If ``False``, compute ``W @ M`` (fixed post-synaptic connections).
        If ``True``, compute ``W^T @ M`` (fixed pre-synaptic connections,
        scatter mode).
    backend : str or None, optional
        Execution backend override.
    weighted : bool, optional
        If ``True``, multiply each weight by the event value instead of its
        activity mask (:class:`~brainevent.WeightedArray` semantics); zero
        events are still skipped. Defaults to ``False``.

    Returns
    -------
    jax.Array or u.Quantity
        Result matrix of shape ``(num_pre, n)`` when ``transpose=False``
        or ``(num_post, n)`` when ``transpose=True``.

    See Also
    --------
    binary_fcnmv : Event-driven sparse matrix--vector product.
    fcnmm : Float (non-event-driven) sparse matrix--matrix product.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    The event-driven matrix-matrix product applies column by column.  When
    ``transpose=False`` (gather mode), for each output element:

        ``Y[i, j] = sum_{k=0}^{n_conn-1} weights[i, k] * 1_{M[indices[i, k], j] active}``

    where "active" means ``True`` for boolean entries or ``!= 0`` for numeric
    entries of ``M``.  For homogeneous weights (``weights`` has shape ``(1,)``):

        ``Y[i, j] = w * sum_{k=0}^{n_conn-1} 1_{M[indices[i, k], j] active}``

    When ``transpose=True`` (scatter mode), the computation distributes
    active entries to their target rows:

        ``Y[indices[i, k], j] += weights[i, k] * 1_{M[i, j] active}``    for all ``i, k, j``

    The event-driven formulation skips inactive entries of ``M``, making the
    computation efficient when the event matrix is sparse.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.binary import binary_fcnmm
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> matrix = jnp.array([[True, False],
        ...                     [False, True],
        ...                     [True, True]])
        >>> y = binary_fcnmm(weights, indices, matrix, shape=(2, 3), transpose=False)
        >>> print(y)
        [[1. 1.]
         [1. 2.]]
    """
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = binary_fcnmm_p_call(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        backend=backend,
        weighted=weighted,
    )[0]
    if _binary_fcnmm_uses_raw_batch_first(transpose=transpose, backend=backend):
        # The cuda_raw scatter kernel emits a raw "batch-first" ``(n, num_post)``
        # layout for efficiency; normalise it to the logical ``(num_post, n)``
        # shape promised by this function (and produced by the other backends).
        r = _maybe_transpose_to_expected(
            r, (shape[1], matrix.shape[1]), op_name='binary_fcnmm'
        )
    return u.maybe_decimal(r * m_unit * w_unit)


def _fcnmm_weighted_numba_kernel(weight_info, transpose: bool):
    import numba

    if transpose:
        # Scatter: posts[indices[i, j], c] += weights[i, j] * matrix[i, c]
        if weight_info.size == 1:
            @numba.njit(fastmath=_EVENT_FASTMATH)
            def ell_mm(weights, indices, matrix, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(matrix.shape[0]):
                    nonzero, = np.where(matrix[i] != 0)
                    vals = matrix[i][nonzero] * w
                    for j in range(indices.shape[1]):
                        posts[indices[i, j], nonzero] += vals
        else:
            @numba.njit(fastmath=_EVENT_FASTMATH)
            def ell_mm(weights, indices, matrix, posts):
                posts[:] = 0.
                for i in range(matrix.shape[0]):
                    nonzero, = np.where(matrix[i] != 0)
                    vals = matrix[i][nonzero]
                    for j in range(indices.shape[1]):
                        posts[indices[i, j], nonzero] += vals * weights[i, j]
    else:
        # Gather: posts[i, c] = sum_j weights[i, j] * matrix[indices[i, j], c]
        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def ell_mm(weights, indices, matrix, posts):
                w = weights[0]
                for i_m in numba.prange(indices.shape[0]):
                    out = np.zeros(posts.shape[1], dtype=posts.dtype)
                    hit = np.zeros(posts.shape[1], dtype=np.bool_)
                    for j in range(indices.shape[1]):
                        row = matrix[indices[i_m, j]]
                        for c in range(row.shape[0]):
                            if row[c] != 0:
                                out[c] += row[c]
                                hit[c] = True
                    # Scale only columns with an active event (``w * 0`` is NaN
                    # for a NaN/Inf weight).
                    for c in range(out.shape[0]):
                        posts[i_m, c] = out[c] * w if hit[c] else 0.
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
            def ell_mm(weights, indices, matrix, posts):
                for i_m in numba.prange(indices.shape[0]):
                    out = np.zeros(posts.shape[1], dtype=posts.dtype)
                    for j in range(indices.shape[1]):
                        row = matrix[indices[i_m, j]]
                        w = weights[i_m, j]
                        for c in range(row.shape[0]):
                            if row[c] != 0:
                                out[c] += w * row[c]
                    posts[i_m] = out
    return ell_mm


def _binary_fcnmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    transpose: bool,
    weighted: bool = False,
    **kwargs
):
    import numba

    # Numba cannot read float16/bfloat16; such events use the boolean kernel on
    # their mask (binary) or are read as float32 (weighted).
    mask_events = _numba_event_mask(matrix_info.dtype)
    bool_events = matrix_info.dtype == jnp.bool_ or mask_events

    if weighted:
        weighted_mm = _fcnmm_weighted_numba_kernel(weight_info, transpose)

        def weighted_kernel(weights, indices, matrix):
            if mask_events:
                matrix = matrix.astype(jnp.float32)
            return numba_kernel(weighted_mm, outs=kwargs['outs'])(weights, indices, matrix)

        return weighted_kernel

    if transpose:
        # fixed pre connection number
        #
        # CSR: [k, m]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            if bool_events:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i_k in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i_k])
                        for i_conn in range(indices.shape[1]):
                            posts[indices[i_k, i_conn], nonzero] += w
            else:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i_k in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i_k] != 0)
                        for i_conn in range(indices.shape[1]):
                            posts[indices[i_k, i_conn], nonzero] += w
        else:
            if bool_events:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    for i in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i])
                        for j in range(indices.shape[1]):
                            posts[indices[i, j], nonzero] += weights[i, j]
            else:
                @numba.njit(fastmath=_EVENT_FASTMATH)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    for i in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i] != 0)
                        for j in range(indices.shape[1]):
                            posts[indices[i, j], nonzero] += weights[i, j]

    else:
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        # Inactive events select a zero term instead of multiplying the weight
        # by 0 (``NaN * 0`` is NaN); homogeneous rows count active events and
        # scale by ``w`` only where the count is non-zero.
        if weight_info.size == 1:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    w = weights[0]
                    for i_m in numba.prange(indices.shape[0]):
                        count = np.zeros(matrix.shape[1], dtype=np.int64)
                        for j in range(indices.shape[1]):
                            row = matrix[indices[i_m, j]]
                            for c in range(row.shape[0]):
                                if row[c]:
                                    count[c] += 1
                        for c in range(count.shape[0]):
                            posts[i_m, c] = w * count[c] if count[c] != 0 else 0.
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    w = weights[0]
                    for i_m in numba.prange(indices.shape[0]):
                        count = np.zeros(matrix.shape[1], dtype=np.int64)
                        for j in range(indices.shape[1]):
                            row = matrix[indices[i_m, j]]
                            for c in range(row.shape[0]):
                                if row[c] != 0:
                                    count[c] += 1
                        for c in range(count.shape[0]):
                            posts[i_m, c] = w * count[c] if count[c] != 0 else 0.
        else:
            if bool_events:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    for i_m in numba.prange(indices.shape[0]):
                        out = np.zeros(posts.shape[1], dtype=posts.dtype)
                        for j in range(indices.shape[1]):
                            w = weights[i_m, j]
                            row = matrix[indices[i_m, j]]
                            for c in range(row.shape[0]):
                                if row[c]:
                                    out[c] += w
                        posts[i_m] = out
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=_EVENT_FASTMATH, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    for i_m in numba.prange(indices.shape[0]):
                        out = np.zeros(posts.shape[1], dtype=posts.dtype)
                        for j in range(indices.shape[1]):
                            w = weights[i_m, j]
                            row = matrix[indices[i_m, j]]
                            for c in range(row.shape[0]):
                                if row[c] != 0:
                                    out[c] += w
                        posts[i_m] = out

    def kernel(weights, indices, matrix):
        if mask_events:
            matrix = event_active(matrix)
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, matrix)

    return kernel


def _binary_fcnmm_cuda_kernel(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    weighted: bool = False,
    **kwargs
):
    sfx = dtype_suffix(weight_info.dtype)
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    # Binary: ``_float`` kernels read events of the *weight* dtype; every other
    # event dtype is converted to its bool activity mask inside the closure.
    # Weighted: ``_wv`` kernels read events cast to the weight dtype.
    abi = cuda_event_abi(
        matrix_info.dtype, native_float=weight_info.dtype,
        weighted=weighted, value_dtype=weight_info.dtype,
    )
    spike_sfx = abi.suffix
    is_bool_matrix = spike_sfx == '_bool'

    if transpose:
        return _binary_fcnmm_sraw_cuda_kernel(
            transpose, weight_info, matrix_info, indices_info, weighted=weighted, **kwargs
        )

    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmm.cu'),
        name='fcn_binary_mm',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    # Gather mode — use packed path for large matrices to fit in L2 cache
    n_rows = matrix_info.shape[0]
    n_batch = matrix_info.shape[1]
    elem_size = {
        np.dtype('bool'): 1, np.dtype('uint8'): 1,
        np.dtype('float16'): 2, np.dtype('bfloat16'): 2,
        np.dtype('float32'): 4, np.dtype('float64'): 8,
    }.get(np.dtype(jnp.bool_) if is_bool_matrix else np.dtype(matrix_info.dtype), 4)
    mat_bytes = n_rows * n_batch * elem_size
    # Use packed path when spike matrix exceeds ~1MB (half L2 on most GPUs).
    # Bit-packing only carries activity, so weighted events never use it.
    use_packed = (mat_bytes > 1024 * 1024) and not weighted

    if use_packed:
        # Step 1: pack spikes into uint32 bitmasks
        pack_sfx = '_bool' if is_bool_matrix else sfx
        pack_name = f'fcn_binary_mm.binary_fcnmm_pack{pack_sfx}'
        n_batch_words = (n_batch + 31) // 32
        packed_info = jax.ShapeDtypeStruct((n_rows, n_batch_words), jnp.uint32)
        # Step 2: run packed gather kernel
        size_sfx = '_warp' if n_conn <= 32 else '_basic'
        packed_gather_name = f'fcn_binary_mm.binary_fcnmm_gather_packed{mode_sfx}{size_sfx}{sfx}'

        def kernel(weights, indices, matrix):
            packed = jax.ffi.ffi_call(pack_name, packed_info)(abi.prepare(matrix))
            return jax.ffi.ffi_call(packed_gather_name, out_info)(weights, indices, packed)
    else:
        # Small matrix — use unpacked gather directly
        kernel_name = (
            f'fcn_binary_mm.binary_fcnmm_gather{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mm.binary_fcnmm_gather{mode_sfx}_basic{spike_sfx}{sfx}'
        )

        def kernel(weights, indices, matrix):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, abi.prepare(matrix))

    return kernel


def _binary_fcnmm_sraw_cuda_kernel(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    weighted: bool = False,
    **kwargs
):

    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmm.cu'),
        name='fcn_binary_mm',
    )

    sraw_out = jax.ShapeDtypeStruct(
        kwargs['outs'][0].shape,
        kwargs['outs'][0].dtype,
    )
    sfx = dtype_suffix(weight_info.dtype)
    mode_sfx = '_homo' if weight_info.size == 1 else '_hetero'
    abi = cuda_event_abi(
        matrix_info.dtype, native_float=weight_info.dtype,
        weighted=weighted, value_dtype=weight_info.dtype,
    )
    spike_sfx = abi.suffix

    kernel_name = (
        'fcn_binary_mm.'
        f'binary_fcnmm_sraw{mode_sfx}{spike_sfx}{sfx}'
    )

    def kernel(weights, indices, matrix):
        raw = jax.ffi.ffi_call(kernel_name, sraw_out)(weights, indices, abi.prepare(matrix))
        return raw,

    return kernel


def _binary_fcnmm_uses_raw_batch_first(*, transpose, backend, platform=None):
    if not transpose:
        return False
    if backend == 'cuda_raw':
        return True
    if backend is None:
        platform = jax.default_backend() if platform is None else platform
        if platform == 'gpu':
            gpu_backend = get_backend('gpu')
            return gpu_backend in (None, 'cuda_raw')
    return False


def _maybe_transpose_to_expected(value, expected_shape, *, op_name):
    if type(value) is ad.Zero:
        return value
    if tuple(value.shape) == tuple(expected_shape):
        return value
    if value.ndim == 2 and tuple(value.T.shape) == tuple(expected_shape):
        return value.T
    raise ValueError(
        f'{op_name} shape mismatch: got {value.shape}, expected {expected_shape}.'
    )


def _logical_transpose_true_shape_from_cotangent(ct, shape):
    if tuple(ct.shape)[0] == shape[1]:
        return tuple(ct.shape)
    return (shape[1], ct.shape[0])

def _binary_fcnmm_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    weighted: bool = False,
    **kwargs,
):
    """Pure JAX reference implementation for benchmarking comparison."""
    n_pre, n_post = shape

    def kernel(weights, indices, matrix):
        active = event_active(matrix)
        # Float activity (binary) or float values (weighted)
        mat_f = event_factor(matrix, dtype=weights.dtype, weighted=weighted)
        w = jnp.broadcast_to(weights, indices.shape)
        # Select, never multiply by a 0/1 mask: a NaN/Inf weight that only
        # meets inactive events contributes exactly 0.

        if transpose:
            # Scatter: Y[n_post, n_batch]; matrix M[n_pre, n_batch]
            # Y[indices[i,k], j] += where(active[i, j], weights[i,k] * mat_f[i, j], 0)
            idx_flat = indices.ravel()  # [n_pre * n_conn]
            act_rep = jnp.repeat(active, indices.shape[1], axis=0)  # [n_pre*n_conn, n_batch]
            w_flat = w.ravel()[:, None]  # [n_pre*n_conn, 1]
            vals = w_flat * jnp.repeat(mat_f, indices.shape[1], axis=0) if weighted else w_flat
            return jax.ops.segment_sum(select_active(act_rep, vals), idx_flat, num_segments=n_post),
        else:
            # Gather: Y[n_pre, n_batch]; matrix M[n_post, n_batch]
            # Y[i, j] = sum_k where(active[indices[i,k], j], weights[i,k] * mat_f[indices[i,k], j], 0)
            def row(w_row, ind):
                vals = w_row[:, None] * mat_f[ind] if weighted else w_row[:, None]
                return jnp.sum(select_active(active[ind], vals), axis=0)

            return jax.vmap(row)(w, indices),

    return kernel


def _binary_fcnmm_jvp_matrix(matrix_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    tangent = fcnmm(weights, indices, matrix_dot, shape=shape, transpose=transpose).astype(weights.dtype)
    if _binary_fcnmm_uses_raw_batch_first(transpose=transpose, backend=kwargs.get('backend')):
        expected_shape = (matrix.shape[1], shape[1])
        tangent = _maybe_transpose_to_expected(
            tangent, expected_shape, op_name='binary_fcnmm matrix JVP'
        )
    return tangent,


def _binary_fcnmm_jvp_weights(weights_dot, weights, indices, matrix, *, shape, transpose, weighted=False,
                              **kwargs):
    tangent = binary_fcnmm_p_call(
        weights_dot, indices, matrix, shape=shape, transpose=transpose, backend=kwargs['backend'],
        weighted=weighted,
    )[0]
    if _binary_fcnmm_uses_raw_batch_first(transpose=transpose, backend=kwargs.get('backend')):
        expected_shape = (matrix.shape[1], shape[1])
        tangent = _maybe_transpose_to_expected(
            tangent, expected_shape, op_name='binary_fcnmm weight JVP'
        )
    return tangent,


def _binary_fcnmm_transpose_rule(ct, weights, indices, matrix, *, shape, transpose, weight_info, weighted=False,
                                 **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]
    if (
        type(ct) is not ad.Zero
        and _binary_fcnmm_uses_raw_batch_first(transpose=transpose, backend=kwargs.get('backend'))
    ):
        logical_ct_shape = _logical_transpose_true_shape_from_cotangent(ct, shape)
        ct = _maybe_transpose_to_expected(
            ct, logical_ct_shape, op_name='binary_fcnmm transpose cotangent'
        )

    # dL/dspk = dL/dy * dy/dspk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(matrix):
        if type(ct) is ad.Zero:
            ct_vector = ad.Zero(matrix)

        else:
            ct_vector = fcnmm(weights, indices, ct, shape=shape, transpose=not transpose)
            ct_vector = ct_vector.astype(matrix.aval.dtype)

        return weights, indices, ct_vector
    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_weight = ad.Zero(weights)

        elif homo:
            ct_weight = binary_fcnmm_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                matrix,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
                weighted=weighted,
            )[0]
            ct_weight = _maybe_transpose_to_expected(
                ct_weight, ct.shape, op_name='binary_fcnmm transpose homo weight'
            )
            # <ct, r> with r = prim(1): outputs reached by no active event
            # (r == 0) contribute exactly 0, even where ``ct`` is NaN/Inf
            # (select, not ``ct * 0``).
            ct_weight = active_sum(ct, ct_weight, weighted=True).reshape(*weight_info.shape)

        else:
            active = event_active(matrix)
            mat_f = event_factor(matrix, dtype=ct.dtype, weighted=weighted)
            # dW[i, k] = sum_j where(active, f * ct, 0): the cotangent of a weight
            # meeting an inactive event is exactly 0.
            if transpose:
                # active/mat_f: [k, n]; ct: [m, n]; dW[i, c] sums over ct[indices[i, c]]
                ct_weight = jax.vmap(
                    lambda a, f, ind: jnp.sum(select_active(a[None, :], ct[ind] * f[None, :]), axis=1)
                )(active, mat_f, indices)
            else:
                # active/mat_f: [m, n]; ct: [k, n]; dW[i, c] gathers events at indices[i, c]
                ct_weight = jax.vmap(
                    lambda c, ind: jnp.sum(select_active(active[ind], mat_f[ind] * c[None, :]), axis=1)
                )(ct, indices)
        return ct_weight, indices, matrix


def _batching_base_fn(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_fcnmm_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
        weighted=kwargs.get('weighted', False),
    )
    raw = r[0]
    if _binary_fcnmm_uses_raw_batch_first(transpose=kwargs['transpose'], backend=kwargs['backend']):
        # cuda_raw scatter emits the raw batch-first layout ``(cols, num_post)``
        # for the primitive as well, so the batch axis precedes ``num_post``.
        r = jnp.reshape(raw, [maybe_batch1, maybe_batch2, raw.shape[1]])
        return [r], [axis - 1]
    r = jnp.reshape(raw, [raw.shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _binary_fcnmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 1):
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 2):
        return _batching_base_fn(args, axis=2, **kwargs)

    else:
        return general_batching_rule(binary_fcnmm_p, args, axes, **kwargs)


def _binary_fcnmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            for bool_event in (True, False):
                n_conn = max(1, int(n_post * prob))
                indices = jnp.asarray(np.random.randint(0, n_post, (n_pre, n_conn), dtype=np.int32))
                if homo:
                    weights = jnp.ones(1, dtype=dtype)
                else:
                    weights = jnp.ones((n_pre, n_conn), dtype=dtype)
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    matrix = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    matrix = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weights, indices, matrix),
                        {'shape': (n_pre, n_post), 'transpose': transpose}
                    )
                )
    return configs


def binary_fcnmm_p_call(
    weights: jax.Array,
    indices: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
    weighted: bool = False,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for event-driven sparse matrix--matrix product
    with fixed connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba or Pallas) without performing any physical-unit
    bookkeeping.  It is typically called from :func:`binary_fcnmm` or from
    autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    matrix : jax.Array
        Dense binary event matrix of shape ``(k, n)``.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, or ``None``).
    weighted : bool, optional
        Multiply weights by the event value instead of its activity mask.
        Defaults to ``False``.

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result matrix.

    See Also
    --------
    binary_fcnmm : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    if _binary_fcnmm_uses_raw_batch_first(transpose=transpose, backend=backend):
        out = jax.ShapeDtypeStruct((matrix.shape[1], n_post), weights.dtype)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return binary_fcnmm_p(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        weighted=weighted,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=[out],
        backend=backend,
    )


binary_fcnmm_p = XLACustomKernel(
    'binary_fcnmm',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_fcnmm``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven)
fixed-connection matrix-matrix multiplication operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. The event-driven formulation only processes
active (spiking) entries, skipping zero entries for efficiency.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_fcnmm_p.available_backends(platform)``,
and the default backend can be configured with ``binary_fcnmm_p.set_default(platform, backend)``.

See Also
--------
binary_fcnmm : High-level user-facing function wrapper.
"""
)
binary_fcnmm_p.def_numba_kernel(_binary_fcnmm_numba_kernel)
binary_fcnmm_p.def_cuda_raw_kernel(_binary_fcnmm_cuda_kernel, asdefault=True)
binary_fcnmm_p.def_kernel('jax_raw', 'cpu', _binary_fcnmm_jax_kernel)
binary_fcnmm_p.def_kernel('jax_raw', 'gpu', _binary_fcnmm_jax_kernel)
binary_fcnmm_p.def_kernel('jax_raw', 'tpu', _binary_fcnmm_jax_kernel)

binary_fcnmm_p.def_jvp_rule2(_binary_fcnmm_jvp_weights, None, _binary_fcnmm_jvp_matrix)
binary_fcnmm_p.def_transpose_rule(_binary_fcnmm_transpose_rule)
binary_fcnmm_p.def_batching_rule(_binary_fcnmm_batching)
binary_fcnmm_p.def_call(binary_fcnmm_p_call)
binary_fcnmm_p.def_tags('fcn', 'binary')
binary_fcnmm_p.def_benchmark_data(_binary_fcnmm_benchmark_data)
