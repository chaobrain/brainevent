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

__all__ = [
    'binary_fcnmv',
    'ell_binary_matvec_p',
    'csc_binary_matvec',
    'csc_binary_matvec_p',
    'csc_binary_matmat',
    'csc_binary_matmat_p',
    'binary_fcnmm',
    'ell_binary_matmat_p',
]


@namescope(static_argnames=['shape', 'transpose'])
def binary_fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    spikes: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
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
        (boolean) or ``> 0`` (float).
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

    where "active" means ``True`` for boolean spikes or ``> 0`` for float spikes.
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
    r = ell_binary_matvec_p_call(
        weights,
        indices,
        spikes,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * v_unit * w_unit)


def _binary_fcnmv_spike_activity(
    spikes: jax.Array,
    dtype: jnp.dtype,
) -> jax.Array:
    spikes = u.math.asarray(spikes)
    if spikes.dtype == jnp.bool_:
        return spikes.astype(dtype)
    return (spikes > 0).astype(dtype)


def _ell_binary_matvec_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i in range(spikes.shape[0]):
                        if spikes[i] > 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += w
        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i]:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, spikes, posts):
                    posts[:] = 0.
                    for i in range(spikes.shape[0]):
                        if spikes[i] > 0.:
                            for j in range(indices.shape[1]):
                                posts[indices[i, j]] += weights[i, j]

    else:
        if weight_info.size == 1:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
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
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    spk_bool = spikes > 0.
                    w = weights[0]
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spk_bool[index]:
                                r += w
                        posts[i] = r
        else:
            if spike_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spikes[index]:
                                r += weights[i, j]
                        posts[i] = r
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, spikes, posts):
                    spk_bool = spikes > 0.
                    for i in numba.prange(indices.shape[0]):
                        r = 0.
                        for j in range(indices.shape[1]):
                            index = indices[i, j]
                            if spk_bool[index]:
                                r += weights[i, j]
                        posts[i] = r

    def kernel(weights, indices, spikes):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, spikes)

    return kernel


def _ell_binary_matvec_cuda_kernel(
    transpose: bool,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv.cu'),
        name='fcn_binary_mv',
    )

    out_info = kwargs['outs']
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    row_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    row_homo = weight_info.size == 1

    if not transpose:
        raise NotImplementedError(
            'The CUDA transpose=False binary row-gather matvec was removed. '
            'Use the CSC column-scatter primitive (csc_binary_matvec) for '
            'W @ s on CUDA; the dispatcher selects it automatically.'
        )

    # Scatter mode: if is_active(spikes[i]) -> output[indices[i,k]] += weights[i,k]
    mode_sfx = '_homo' if row_homo else '_hetero'
    kernel_name = f'fcn_binary_mv.binary_fcnmv_scatter{mode_sfx}_bool{row_sfx}'

    def kernel(weights, indices, spikes):
        spikes = u.math.asarray(spikes, dtype=bool)
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, spikes)

    return kernel


def _ell_binary_matvec_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    """Pure JAX reference implementation for benchmarking comparison."""
    n_pre, n_post = shape

    def kernel(weights, indices, spikes):
        spk_f = _binary_fcnmv_spike_activity(spikes, weights.dtype)

        if transpose:
            # Scatter: y[indices[i,k]] += weights[i,k] * spk_f[i]
            masked = jnp.broadcast_to(spk_f[:, None] * weights, indices.shape)
            return jax.ops.segment_sum(masked.ravel(), indices.ravel(), num_segments=n_post),
        else:
            # Gather: y[i] = sum_k weights[i,k] * spk_f[indices[i,k]]
            if weights.size == 1:
                w = weights[0]
                return jax.vmap(lambda ind: w * jnp.sum(spk_f[ind]))(indices),
            else:
                return jax.vmap(lambda w, ind: jnp.sum(w * spk_f[ind]))(weights, indices),

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
    return fcnmv(weights, indices, spk_dot, shape=shape, transpose=transpose),


def _binary_fcnmv_jvp_weights(
    w_dot,
    weights,
    indices,
    spikes,
    *,
    shape,
    transpose,
    **kwargs
):
    return ell_binary_matvec_p_call(
        w_dot,
        indices,
        spikes,
        shape=shape,
        transpose=transpose,
        backend=kwargs['backend'],
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
        return weights, indices, ct_spk

    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_gmax = ad.Zero(weights)
        elif homo:
            # scalar
            ct_gmax = ell_binary_matvec_p_call(
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                spikes,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )
            ct_gmax = jnp.inner(ct, ct_gmax[0]).reshape(*weight_info.shape)
        else:
            spk_active = _binary_fcnmv_spike_activity(spikes, weight_info.dtype)
            if transpose:
                ct_gmax = jax.vmap(lambda v, ind: v * ct[ind])(spk_active, indices)
            else:
                ct_gmax = jax.vmap(lambda c, ind: c * spk_active[ind])(ct, indices)
    return ct_gmax, indices, spikes


def _binary_fcnmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = ell_binary_matmat_p_call(
            args[0],
            args[1],
            args[2].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = ell_binary_matmat_p_call(
            args[0],
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(ell_binary_matvec_p, args, axes, **kwargs)


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


def ell_binary_matvec_p_call(
    weights: jax.Array,
    indices: jax.Array,
    spikes: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
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
    return ell_binary_matvec_p(
        weights,
        indices,
        spikes,
        outs=[out],
        shape=shape,
        transpose=transpose,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        backend=backend,
    )


ell_binary_matvec_p = XLACustomKernel(
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
The CUDA ``transpose=False`` row-gather path was removed; use
``csc_binary_matvec`` for ``W @ s`` on CUDA.

Available backends can be queried with ``ell_binary_matvec_p.available_backends(platform)``,
and the default backend can be configured with ``ell_binary_matvec_p.set_default(platform, backend)``.

See Also
--------
binary_fcnmv : High-level user-facing function wrapper.
"""
)


ell_binary_matvec_p.def_numba_kernel(_ell_binary_matvec_numba_kernel)
ell_binary_matvec_p.def_cuda_raw_kernel(_ell_binary_matvec_cuda_kernel, asdefault=True)
ell_binary_matvec_p.def_kernel('jax_raw', 'cpu', _ell_binary_matvec_jax_kernel)
ell_binary_matvec_p.def_kernel('jax_raw', 'gpu', _ell_binary_matvec_jax_kernel)
ell_binary_matvec_p.def_kernel('jax_raw', 'tpu', _ell_binary_matvec_jax_kernel)

ell_binary_matvec_p.def_jvp_rule2(
    _binary_fcnmv_jvp_weights,
    None,
    _binary_fcnmv_jvp_spikes,
)
ell_binary_matvec_p.def_transpose_rule(_binary_fcnmv_transpose_rule)
ell_binary_matvec_p.def_batching_rule(_binary_fcnmv_batching)
ell_binary_matvec_p.def_call(ell_binary_matvec_p_call)
ell_binary_matvec_p.def_tags('fcn', 'binary')
ell_binary_matvec_p.def_benchmark_data(_binary_fcnmv_benchmark_data)


# ---------------------------------------------------------------------------
# CSC column-scatter primitive: y = W @ s (event-driven), W stored column-major
# ---------------------------------------------------------------------------


def _csc_binary_matvec_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    if weight_info.size == 1:
        if spike_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def csc_mv(weights, indices, indptr, spikes, posts):
                posts[:] = 0.
                w = weights[0]
                for col in range(spikes.shape[0]):
                    if spikes[col]:
                        for pos in range(indptr[col], indptr[col + 1]):
                            posts[indices[pos]] += w
        else:
            @numba.njit(fastmath=True)
            def csc_mv(weights, indices, indptr, spikes, posts):
                posts[:] = 0.
                w = weights[0]
                for col in range(spikes.shape[0]):
                    if spikes[col] > 0.:
                        for pos in range(indptr[col], indptr[col + 1]):
                            posts[indices[pos]] += w
    else:
        if spike_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def csc_mv(weights, indices, indptr, spikes, posts):
                posts[:] = 0.
                for col in range(spikes.shape[0]):
                    if spikes[col]:
                        for pos in range(indptr[col], indptr[col + 1]):
                            posts[indices[pos]] += weights[pos]
        else:
            @numba.njit(fastmath=True)
            def csc_mv(weights, indices, indptr, spikes, posts):
                posts[:] = 0.
                for col in range(spikes.shape[0]):
                    if spikes[col] > 0.:
                        for pos in range(indptr[col], indptr[col + 1]):
                            posts[indices[pos]] += weights[pos]

    def kernel(weights, indices, indptr, spikes):
        return numba_kernel(csc_mv, outs=kwargs['outs'])(weights, indices, indptr, spikes)

    return kernel


def _csc_binary_matvec_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmv_col_scatter.cu'),
        name='fcn_binary_mv_col_scatter',
    )

    out_info = kwargs['outs']
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16'
    }
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    mode_sfx = '_homo' if weight_info.size == 1 else '_hetero'
    kernel_name = f'fcn_binary_mv_col_scatter.binary_fcnmv_col_scatter{mode_sfx}_bool{sfx}'

    def kernel(weights, indices, indptr, spikes):
        spikes = u.math.asarray(spikes, dtype=bool)
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, indptr, spikes)

    return kernel


def _csc_binary_matvec_jax_kernel(
    shape: Tuple[int, int],
    **kwargs,
):
    """Pure JAX reference: y = W @ s via column scatter (segment_sum)."""
    n_pre, n_post = shape

    def kernel(weights, indices, indptr, spikes):
        spk_f = _binary_fcnmv_spike_activity(spikes, weights.dtype)
        col_ids = jnp.repeat(
            jnp.arange(indptr.shape[0] - 1),
            jnp.diff(indptr),
            total_repeat_length=indices.shape[0],
        )
        if weights.size == 1:
            contrib = spk_f[col_ids] * weights[0]
        else:
            contrib = spk_f[col_ids] * weights
        return jax.ops.segment_sum(contrib, indices, num_segments=n_pre),

    return kernel


def _csc_binary_matvec_jvp_spikes(
    spk_dot,
    weights,
    indices,
    indptr,
    spikes,
    *,
    shape,
    **kwargs
):
    # y = W @ s linear in s: dy = W @ s_dot (float segment-scatter).
    n_pre = shape[0]
    col_ids = jnp.repeat(
        jnp.arange(indptr.shape[0] - 1),
        jnp.diff(indptr),
        total_repeat_length=indices.shape[0],
    )
    if weights.size == 1:
        contrib = spk_dot[col_ids] * weights[0]
    else:
        contrib = spk_dot[col_ids] * weights
    return jax.ops.segment_sum(contrib, indices, num_segments=n_pre),


def _csc_binary_matvec_jvp_weights(
    w_dot,
    weights,
    indices,
    indptr,
    spikes,
    *,
    shape,
    **kwargs
):
    return csc_binary_matvec_p_call(
        w_dot,
        indices,
        indptr,
        spikes,
        shape=shape,
        backend=kwargs['backend'],
    )


def _csc_binary_matvec_transpose_rule(
    ct,
    weights,
    indices,
    indptr,
    spikes,
    *,
    shape,
    weight_info,
    **kwargs
):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse structure.")

    ct = ct[0]
    n_pre, n_post = shape
    homo = weight_info.size == 1
    col_ids = jnp.repeat(
        jnp.arange(indptr.shape[0] - 1),
        jnp.diff(indptr),
        total_repeat_length=indices.shape[0],
    )

    if ad.is_undefined_primal(spikes):
        # dL/ds = W^T @ ct: ds[col] = sum_{pos in col} weights[pos] * ct[indices[pos]]
        if type(ct) is ad.Zero:
            ct_spk = ad.Zero(spikes)
        else:
            w = weights[0] if homo else weights
            contrib = w * ct[indices]
            ct_spk = jax.ops.segment_sum(contrib, col_ids, num_segments=n_post)
        return weights, indices, indptr, ct_spk

    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_w = ad.Zero(weights)
        else:
            spk_active = _binary_fcnmv_spike_activity(spikes, weight_info.dtype)
            per_pos = ct[indices] * spk_active[col_ids]
            if homo:
                ct_w = jnp.sum(per_pos).reshape(*weight_info.shape)
            else:
                ct_w = per_pos
        return ct_w, indices, indptr, spikes


def _csc_binary_matvec_batching(args, axes, **kwargs):
    return general_batching_rule(csc_binary_matvec_p, args, axes, **kwargs)


def csc_binary_matvec_p_call(
    weights: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    spikes: jax.Array,
    *,
    shape: Tuple[int, int],
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for the CSC column-scatter event-driven matrix
    --vector product ``y = W @ s``.

    Parameters
    ----------
    weights : jax.Array
        Column-major non-zero weights.  Shape ``(1,)`` for homogeneous or
        ``(NNZ,)`` for heterogeneous weights.  Must be floating-point.
    indices : jax.Array
        Row-index array of shape ``(NNZ,)``.
    indptr : jax.Array
        Column pointer array of shape ``(num_post + 1,)``.
    spikes : jax.Array
        Binary spike vector of shape ``(num_post,)`` (boolean or float).
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    backend : str or None, optional
        Backend override (``'numba'``, ``'cuda_raw'``, or ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result vector ``(num_pre,)``.
    """
    n_pre, n_post = shape
    weights = u.math.asarray(weights)
    indices = u.math.asarray(indices)
    indptr = u.math.asarray(indptr)
    spikes = u.math.asarray(spikes)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    out = jax.ShapeDtypeStruct((n_pre,), weights.dtype)
    return csc_binary_matvec_p(
        weights,
        indices,
        indptr,
        spikes,
        outs=[out],
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        backend=backend,
    )


csc_binary_matvec_p = XLACustomKernel(
    'csc_binary_matvec',
    doc="""
Low-level XLA custom-kernel primitive for the CSC column-scatter
event-driven matrix-vector product ``y = W @ s``.

The matrix ``W`` is stored column-major (CSC) as
``(weights, indices, indptr)``.  For each active input column the kernel
scatter-adds its weights to the output rows.  CUDA provides the real
column-scatter kernel; numba and jax provide a ``segment_sum`` reference so
the path is testable on every platform.

The primitive takes four positional arguments
``(weights, indices, indptr, spikes)``.

See Also
--------
csc_binary_matvec : High-level user-facing function wrapper.
ell_binary_matvec_p : Row-major (ELL) peer primitive.
"""
)


csc_binary_matvec_p.def_numba_kernel(_csc_binary_matvec_numba_kernel)
csc_binary_matvec_p.def_cuda_raw_kernel(_csc_binary_matvec_cuda_kernel, asdefault=True)
csc_binary_matvec_p.def_kernel('jax_raw', 'cpu', _csc_binary_matvec_jax_kernel)
csc_binary_matvec_p.def_kernel('jax_raw', 'gpu', _csc_binary_matvec_jax_kernel)
csc_binary_matvec_p.def_kernel('jax_raw', 'tpu', _csc_binary_matvec_jax_kernel)

csc_binary_matvec_p.def_jvp_rule2(
    _csc_binary_matvec_jvp_weights,
    None,
    None,
    _csc_binary_matvec_jvp_spikes,
)
csc_binary_matvec_p.def_transpose_rule(_csc_binary_matvec_transpose_rule)
csc_binary_matvec_p.def_batching_rule(_csc_binary_matvec_batching)
csc_binary_matvec_p.def_call(csc_binary_matvec_p_call)
csc_binary_matvec_p.def_tags('fcn', 'binary')


@namescope(static_argnames=['shape'])
def csc_binary_matvec(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    indptr: jax.Array,
    spikes: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven CSC column-scatter sparse matrix--vector product ``y = W @ s``.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Column-major non-zero weights, shape ``(1,)`` (homogeneous) or
        ``(NNZ,)`` (heterogeneous).  Must have a floating-point dtype.
    indices : jax.Array
        Row-index array of shape ``(NNZ,)``.
    indptr : jax.Array
        Column pointer array of shape ``(num_post + 1,)``.
    spikes : jax.Array or u.Quantity
        Binary spike vector of shape ``(num_post,)``.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the dense weight matrix.
    backend : str or None, optional
        Execution backend override.

    Returns
    -------
    jax.Array or u.Quantity
        Result vector of shape ``(num_pre,)``.

    See Also
    --------
    binary_fcnmv : ELL event-driven matrix--vector product.
    """
    weights, w_unit = u.split_mantissa_unit(weights)
    spikes, v_unit = u.split_mantissa_unit(spikes)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = csc_binary_matvec_p_call(
        weights,
        indices,
        indptr,
        spikes,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * v_unit * w_unit)


# ---------------------------------------------------------------------------
# CSC column-scatter matmat primitive: Y = W @ M (event-driven), W stored
# column-major.  This is the matmat peer of ``csc_binary_matvec`` and serves
# the *unfavorable* FCN matmat direction: weights are pre-permuted into CSC
# order by the caller, so the kernel reads them positionally (no fused gather).
# ---------------------------------------------------------------------------


def _csc_binary_matmat_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import numba

    if weight_info.size == 1:
        if matrix_info.dtype == jnp.bool_:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def csc_mm(weights, indices, indptr, matrix, posts):
                w = weights[0]
                posts[:] = 0.
                for k in numba.prange(matrix.shape[1]):
                    for col in range(matrix.shape[0]):
                        if matrix[col, k]:
                            for pos in range(indptr[col], indptr[col + 1]):
                                posts[indices[pos], k] += w
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def csc_mm(weights, indices, indptr, matrix, posts):
                w = weights[0]
                posts[:] = 0.
                for k in numba.prange(matrix.shape[1]):
                    for col in range(matrix.shape[0]):
                        if matrix[col, k] > 0.:
                            for pos in range(indptr[col], indptr[col + 1]):
                                posts[indices[pos], k] += w
    else:
        if matrix_info.dtype == jnp.bool_:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def csc_mm(weights, indices, indptr, matrix, posts):
                posts[:] = 0.
                for k in numba.prange(matrix.shape[1]):
                    for col in range(matrix.shape[0]):
                        if matrix[col, k]:
                            for pos in range(indptr[col], indptr[col + 1]):
                                posts[indices[pos], k] += weights[pos]
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def csc_mm(weights, indices, indptr, matrix, posts):
                posts[:] = 0.
                for k in numba.prange(matrix.shape[1]):
                    for col in range(matrix.shape[0]):
                        if matrix[col, k] > 0.:
                            for pos in range(indptr[col], indptr[col + 1]):
                                posts[indices[pos], k] += weights[pos]

    def kernel(weights, indices, indptr, matrix):
        return numba_kernel(csc_mm, outs=kwargs['outs'])(weights, indices, indptr, matrix)

    return kernel


def _csc_binary_matmat_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmm_col_scatter.cu'),
        name='fcn_binary_mm_col_scatter',
    )

    out_info = kwargs['outs']
    n_pre = out_info[0].shape[0]
    n_batch = out_info[0].shape[1]
    out_dtype = out_info[0].dtype
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    mode_sfx = '_homo' if weight_info.size == 1 else '_hetero'
    spike_sfx = '_bool' if matrix_info.dtype == jnp.bool_ else '_float'
    kernel_name = f'fcn_binary_mm_col_scatter.binary_fcnmm_col_scatter{mode_sfx}{spike_sfx}{sfx}'

    # The .cu kernel works in a transposed physical layout: ``matrix_t`` is read
    # as ``(n_batch, num_post)`` and ``output`` written as ``(n_batch, num_pre)``.
    cu_out_info = (jax.ShapeDtypeStruct((n_batch, n_pre), out_dtype),)

    def kernel(weights, indices, indptr, matrix):
        matrix_t = matrix.T  # (n_batch, num_post)
        if matrix_info.dtype == jnp.bool_:
            matrix_t = u.math.asarray(matrix_t, dtype=bool)
        r = jax.ffi.ffi_call(kernel_name, cu_out_info)(weights, indices, indptr, matrix_t)
        return r.T,  # (num_pre, n_batch)

    return kernel


def _csc_binary_matmat_jax_kernel(
    shape: Tuple[int, int],
    **kwargs,
):
    """Pure JAX reference: Y = W @ M via column scatter (segment_sum)."""
    n_pre, n_post = shape

    def kernel(weights, indices, indptr, matrix):
        mat_f = _binary_fcnmv_spike_activity(matrix, weights.dtype)  # [n_post, n]
        col_ids = jnp.repeat(
            jnp.arange(indptr.shape[0] - 1),
            jnp.diff(indptr),
            total_repeat_length=indices.shape[0],
        )
        contrib = mat_f[col_ids]  # [NNZ, n]
        if weights.size == 1:
            contrib = contrib * weights[0]
        else:
            contrib = contrib * weights[:, None]
        return jax.ops.segment_sum(contrib, indices, num_segments=n_pre),

    return kernel


def _csc_binary_matmat_jvp_matrix(
    mat_dot,
    weights,
    indices,
    indptr,
    matrix,
    *,
    shape,
    **kwargs
):
    # Y = W @ M linear in M: dY = W @ M_dot (float column-scatter).
    n_pre = shape[0]
    col_ids = jnp.repeat(
        jnp.arange(indptr.shape[0] - 1),
        jnp.diff(indptr),
        total_repeat_length=indices.shape[0],
    )
    contrib = mat_dot[col_ids]
    if weights.size == 1:
        contrib = contrib * weights[0]
    else:
        contrib = contrib * weights[:, None]
    return jax.ops.segment_sum(contrib, indices, num_segments=n_pre),


def _csc_binary_matmat_jvp_weights(
    w_dot,
    weights,
    indices,
    indptr,
    matrix,
    *,
    shape,
    **kwargs
):
    return csc_binary_matmat_p_call(
        w_dot,
        indices,
        indptr,
        matrix,
        shape=shape,
        backend=kwargs['backend'],
    )


def _csc_binary_matmat_transpose_rule(
    ct,
    weights,
    indices,
    indptr,
    matrix,
    *,
    shape,
    weight_info,
    **kwargs
):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse structure.")

    ct = ct[0]
    n_pre, n_post = shape
    homo = weight_info.size == 1
    col_ids = jnp.repeat(
        jnp.arange(indptr.shape[0] - 1),
        jnp.diff(indptr),
        total_repeat_length=indices.shape[0],
    )

    if ad.is_undefined_primal(matrix):
        # dL/dM = W^T @ ct: dM[col, j] = sum_{pos in col} weights[pos] * ct[indices[pos], j]
        if type(ct) is ad.Zero:
            ct_mat = ad.Zero(matrix)
        else:
            w = weights[0] if homo else weights[:, None]
            contrib = w * ct[indices]  # [NNZ, n]
            ct_mat = jax.ops.segment_sum(contrib, col_ids, num_segments=n_post)
        return weights, indices, indptr, ct_mat

    else:
        # dL/dw: dL/dweights[pos] = sum_j ct[indices[pos], j] * M_active[col(pos), j]
        if type(ct) is ad.Zero:
            ct_w = ad.Zero(weights)
        else:
            spk_active = _binary_fcnmv_spike_activity(matrix, weight_info.dtype)
            per_pos = jnp.sum(ct[indices] * spk_active[col_ids], axis=1)  # [NNZ]
            if homo:
                ct_w = jnp.sum(per_pos).reshape(*weight_info.shape)
            else:
                ct_w = per_pos
        return ct_w, indices, indptr, matrix


def _csc_binary_matmat_batching(args, axes, **kwargs):
    return general_batching_rule(csc_binary_matmat_p, args, axes, **kwargs)


def csc_binary_matmat_p_call(
    weights: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for the CSC column-scatter event-driven matrix
    --matrix product ``Y = W @ M``.

    Parameters
    ----------
    weights : jax.Array
        Column-major non-zero weights, already permuted into CSC order.  Shape
        ``(1,)`` for homogeneous or ``(NNZ,)`` for heterogeneous weights.  Must
        be floating-point.
    indices : jax.Array
        Row-index array of shape ``(NNZ,)``.
    indptr : jax.Array
        Column pointer array of shape ``(num_post + 1,)``.
    matrix : jax.Array
        Dense binary event matrix of shape ``(num_post, n)`` (boolean or float).
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    backend : str or None, optional
        Backend override (``'numba'``, ``'cuda_raw'``, or ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result matrix ``(num_pre, n)``.
    """
    n_pre, n_post = shape
    weights = u.math.asarray(weights)
    indices = u.math.asarray(indices)
    indptr = u.math.asarray(indptr)
    matrix = u.math.asarray(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert matrix.ndim == 2, 'matrix must be 2D.'
    assert matrix.shape[0] == n_post, 'matrix row count must equal num_post.'
    out = jax.ShapeDtypeStruct((n_pre, matrix.shape[1]), weights.dtype)
    return csc_binary_matmat_p(
        weights,
        indices,
        indptr,
        matrix,
        outs=[out],
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        backend=backend,
    )


csc_binary_matmat_p = XLACustomKernel(
    'csc_binary_matmat',
    doc="""
Low-level XLA custom-kernel primitive for the CSC column-scatter
event-driven matrix-matrix product ``Y = W @ M``.

The matrix ``W`` is stored column-major (CSC) as
``(weights, indices, indptr)`` with ``weights`` already permuted into CSC
order.  For each active ``(column, batch)`` entry the kernel scatter-adds the
column's weights to the output rows.  CUDA provides the real column-scatter
kernel; numba and jax provide a ``segment_sum`` reference so the path is
testable on every platform.

The primitive takes four positional arguments
``(weights, indices, indptr, matrix)``.

See Also
--------
csc_binary_matmat : High-level user-facing function wrapper.
csc_binary_matvec : Matrix-vector peer primitive.
"""
)


csc_binary_matmat_p.def_numba_kernel(_csc_binary_matmat_numba_kernel)
csc_binary_matmat_p.def_cuda_raw_kernel(_csc_binary_matmat_cuda_kernel, asdefault=True)
csc_binary_matmat_p.def_kernel('jax_raw', 'cpu', _csc_binary_matmat_jax_kernel)
csc_binary_matmat_p.def_kernel('jax_raw', 'gpu', _csc_binary_matmat_jax_kernel)
csc_binary_matmat_p.def_kernel('jax_raw', 'tpu', _csc_binary_matmat_jax_kernel)

csc_binary_matmat_p.def_jvp_rule2(
    _csc_binary_matmat_jvp_weights,
    None,
    None,
    _csc_binary_matmat_jvp_matrix,
)
csc_binary_matmat_p.def_transpose_rule(_csc_binary_matmat_transpose_rule)
csc_binary_matmat_p.def_batching_rule(_csc_binary_matmat_batching)
csc_binary_matmat_p.def_call(csc_binary_matmat_p_call)
csc_binary_matmat_p.def_tags('fcn', 'binary')


@namescope(static_argnames=['shape'])
def csc_binary_matmat(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    indptr: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Event-driven CSC column-scatter sparse matrix--matrix product ``Y = W @ M``.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Column-major non-zero weights, already permuted into CSC order.  Shape
        ``(1,)`` (homogeneous) or ``(NNZ,)`` (heterogeneous).  Must have a
        floating-point dtype.
    indices : jax.Array
        Row-index array of shape ``(NNZ,)``.
    indptr : jax.Array
        Column pointer array of shape ``(num_post + 1,)``.
    matrix : jax.Array or u.Quantity
        Dense binary event matrix of shape ``(num_post, n)``.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the dense weight matrix.
    backend : str or None, optional
        Execution backend override.

    Returns
    -------
    jax.Array or u.Quantity
        Result matrix of shape ``(num_pre, n)``.

    See Also
    --------
    csc_binary_matvec : CSC column-scatter matrix--vector product.
    binary_fcnmm : ELL event-driven matrix--matrix product.
    """
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = csc_binary_matmat_p_call(
        weights,
        indices,
        indptr,
        matrix,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * m_unit * w_unit)


@namescope(static_argnames=['shape', 'transpose'])
def binary_fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
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

    where "active" means ``True`` for boolean entries or ``> 0`` for float
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
    r = ell_binary_matmat_p_call(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * m_unit * w_unit)


def _binary_fcnmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # fixed pre connection number
        #
        # CSR: [k, m]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i_k in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i_k])
                        for i_conn in range(indices.shape[1]):
                            posts[indices[i_k, i_conn], nonzero] += w
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    w = weights[0]
                    for i_k in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i_k] > 0.)
                        for i_conn in range(indices.shape[1]):
                            posts[indices[i_k, i_conn], nonzero] += w
        else:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    for i in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i])
                        for j in range(indices.shape[1]):
                            posts[indices[i, j], nonzero] += weights[i, j]
            else:
                @numba.njit(fastmath=True)
                def ell_mv(weights, indices, matrix, posts):
                    posts[:] = 0.
                    for i in range(matrix.shape[0]):
                        nonzero, = np.where(matrix[i] > 0.)
                        for j in range(indices.shape[1]):
                            posts[indices[i, j], nonzero] += weights[i, j]

    else:
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    w = weights[0]
                    for i_m in numba.prange(indices.shape[0]):
                        posts[i_m] = w * np.sum(matrix[indices[i_m]], axis=0)
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    w = weights[0]
                    for i_m in numba.prange(indices.shape[0]):
                        events = matrix[indices[i_m]] > 0.
                        posts[i_m] = w * np.sum(events, axis=0)
        else:
            if matrix_info.dtype == jnp.bool_:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    for i_m in numba.prange(indices.shape[0]):
                        posts[i_m] = weights[i_m] @ (matrix[indices[i_m]]).astype(weights.dtype)
            else:
                @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
                def ell_mv(weights, indices, matrix, posts):
                    for i_m in numba.prange(indices.shape[0]):
                        events = (matrix[indices[i_m]] > 0.).astype(weights.dtype)
                        posts[i_m] = weights[i_m] @ events

    def kernel(weights, indices, matrix):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, matrix)

    return kernel


def _binary_fcnmm_cuda_kernel(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_fcnmm.cu'),
        name='fcn_binary_mm',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    is_bool_matrix = (matrix_info.dtype == jnp.bool_)
    _dtype_sfx = {
        np.dtype('float16'): '_f16',
        np.dtype('float32'): '_f32',
        np.dtype('float64'): '_f64',
        np.dtype('bfloat16'): '_bf16'
    }
    sfx = _dtype_sfx.get(np.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'
    spike_sfx = '_bool' if is_bool_matrix else '_float'

    if transpose:
        # Scatter mode
        kernel_name = (
            f'fcn_binary_mm.binary_fcnmm_scatter{mode_sfx}_warp{spike_sfx}{sfx}'
            if n_conn <= 32
            else f'fcn_binary_mm.binary_fcnmm_scatter{mode_sfx}_basic{spike_sfx}{sfx}'
        )

        def kernel(weights, indices, matrix):
            return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, matrix)
    else:
        # Gather mode — use packed path for large matrices to fit in L2 cache
        n_rows = matrix_info.shape[0]
        n_batch = matrix_info.shape[1]
        elem_size = {
            np.dtype('bool'): 1, np.dtype('uint8'): 1,
            np.dtype('float16'): 2, np.dtype('bfloat16'): 2,
            np.dtype('float32'): 4, np.dtype('float64'): 8,
        }.get(np.dtype(matrix_info.dtype), 4)
        mat_bytes = n_rows * n_batch * elem_size
        # Use packed path when spike matrix exceeds ~1MB (half L2 on most GPUs)
        use_packed = (mat_bytes > 1024 * 1024)

        if use_packed:
            # Step 1: pack spikes into uint32 bitmasks
            pack_sfx = '_bool' if is_bool_matrix else sfx
            pack_name = f'fcn_binary_mm.ell_binary_matmat_pack{pack_sfx}'
            n_batch_words = (n_batch + 31) // 32
            packed_info = jax.ShapeDtypeStruct((n_rows, n_batch_words), jnp.uint32)
            # Step 2: run packed gather kernel
            size_sfx = '_warp' if n_conn <= 32 else '_basic'
            packed_gather_name = f'fcn_binary_mm.binary_fcnmm_gather_packed{mode_sfx}{size_sfx}{sfx}'

            def kernel(weights, indices, matrix):
                packed = jax.ffi.ffi_call(pack_name, packed_info)(matrix)
                return jax.ffi.ffi_call(packed_gather_name, out_info)(weights, indices, packed)
        else:
            # Small matrix — use unpacked gather directly
            kernel_name = (
                f'fcn_binary_mm.binary_fcnmm_gather{mode_sfx}_warp{spike_sfx}{sfx}'
                if n_conn <= 32
                else f'fcn_binary_mm.binary_fcnmm_gather{mode_sfx}_basic{spike_sfx}{sfx}'
            )

            def kernel(weights, indices, matrix):
                return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, matrix)

    return kernel


def _binary_fcnmm_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    """Pure JAX reference implementation for benchmarking comparison."""
    n_pre, n_post = shape

    def kernel(weights, indices, matrix):
        # Convert to float activity
        if matrix.dtype == jnp.bool_:
            mat_f = matrix.astype(weights.dtype)
        else:
            mat_f = (matrix > 0).astype(weights.dtype)

        if transpose:
            # Scatter: Y[n_post, n_batch]; matrix M[n_pre, n_batch]
            # Y[indices[i,k], j] += weights[i,k] * mat_f[i, j]
            idx_flat = indices.ravel()  # [n_pre * n_conn]
            mat_rep = jnp.repeat(mat_f, indices.shape[1], axis=0)  # [n_pre*n_conn, n_batch]
            if weights.size == 1:
                return jax.ops.segment_sum(
                    mat_rep * weights[0], idx_flat, num_segments=n_post
                ),
            else:
                w_flat = weights.ravel()[:, None]  # [n_pre*n_conn, 1]
                return jax.ops.segment_sum(
                    mat_rep * w_flat, idx_flat, num_segments=n_post
                ),
        else:
            # Gather: Y[n_pre, n_batch]; matrix M[n_post, n_batch]
            # Y[i, j] = sum_k weights[i,k] * mat_f[indices[i,k], j]
            if weights.size == 1:
                w = weights[0]
                return jax.vmap(lambda ind: w * jnp.sum(mat_f[ind], axis=0))(indices),
            else:
                return jax.vmap(lambda w_row, ind: w_row @ mat_f[ind])(weights, indices),

    return kernel


def _binary_fcnmm_jvp_matrix(matrix_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return fcnmm(weights, indices, matrix_dot, shape=shape, transpose=transpose),


def _binary_fcnmm_jvp_weights(weights_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return ell_binary_matmat_p_call(
        weights_dot, indices, matrix, shape=shape, transpose=transpose, backend=kwargs['backend']
    )


def _binary_fcnmm_transpose_rule(ct, weights, indices, matrix, *, shape, transpose, weight_info, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # dL/dspk = dL/dy * dy/dspk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(matrix):
        if type(ct) is ad.Zero:
            ct_vector = ad.Zero(matrix)

        else:
            ct_vector = fcnmm(weights, indices, ct, shape=shape, transpose=not transpose)

        return weights, indices, ct_vector
    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_weight = ad.Zero(weights)

        elif homo:
            ct_weight = ell_binary_matmat_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                matrix,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )[0]
            ct_weight = jnp.sum(ct * ct_weight).reshape(*weight_info.shape)

        else:
            if transpose:
                # inputs: [k, n] @ [k, n_conn]
                # ct: [m, n]
                ct_weight = jax.vmap(lambda mat, ind: ct[ind] @ mat)(matrix, indices)
            else:
                # inputs: [m, n] @ [m, n_conn]
                # ct: [k, n]
                ct_weight = jax.vmap(lambda c, ind: (matrix[ind] @ c))(ct, indices)
        return ct_weight, indices, matrix


def _batching_base_fn(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = ell_binary_matmat_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
    )
    raw = r[0]
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
        return general_batching_rule(ell_binary_matmat_p, args, axes, **kwargs)


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


def ell_binary_matmat_p_call(
    weights: jax.Array,
    indices: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
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

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result matrix.

    See Also
    --------
    binary_fcnmm : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return ell_binary_matmat_p(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=[out],
        backend=backend,
    )


ell_binary_matmat_p = XLACustomKernel(
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

Available backends can be queried with ``ell_binary_matmat_p.available_backends(platform)``,
and the default backend can be configured with ``ell_binary_matmat_p.set_default(platform, backend)``.

See Also
--------
binary_fcnmm : High-level user-facing function wrapper.
"""
)
ell_binary_matmat_p.def_numba_kernel(_binary_fcnmm_numba_kernel)
ell_binary_matmat_p.def_cuda_raw_kernel(_binary_fcnmm_cuda_kernel, asdefault=True)
ell_binary_matmat_p.def_kernel('jax_raw', 'cpu', _binary_fcnmm_jax_kernel)
ell_binary_matmat_p.def_kernel('jax_raw', 'gpu', _binary_fcnmm_jax_kernel)
ell_binary_matmat_p.def_kernel('jax_raw', 'tpu', _binary_fcnmm_jax_kernel)

ell_binary_matmat_p.def_jvp_rule2(_binary_fcnmm_jvp_weights, None, _binary_fcnmm_jvp_matrix, None)
ell_binary_matmat_p.def_transpose_rule(_binary_fcnmm_transpose_rule)
ell_binary_matmat_p.def_batching_rule(_binary_fcnmm_batching)
ell_binary_matmat_p.def_call(ell_binary_matmat_p_call)
ell_binary_matmat_p.def_tags('fcn', 'binary')
ell_binary_matmat_p.def_benchmark_data(_binary_fcnmm_benchmark_data)
