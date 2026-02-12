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

from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import cdiv, generate_block_dim, namescope
from brainevent._op import XLACustomKernel, general_batching_rule, jaxinfo_to_warpinfo, numba_kernel
from brainevent._op.benchmark import BenchmarkConfig
from brainevent.config import get_numba_parallel

__all__ = [
    'spfloat_densemv',
    'spfloat_densemv_p',
    'spfloat_densemv_p_call',
    'spfloat_densemm',
    'spfloat_densemm_p',
    'spfloat_densemm_p_call',
]


# ==============================================================================
# Unified sparse-float dense matrix-vector product (spfloat_densemv)
# ==============================================================================
#
# transpose=False: weights[m,k] @ spikes[k] -> out[m]  (old dsfmv)
# transpose=True:  spikes[k] @ weights[k,n] -> out[n]  (old sfdvm)
#
# Argument order is always (weights, spikes).


@namescope(static_argnames=['transpose'])
def spfloat_densemv(weights, spikes, *, transpose, backend: Optional[str] = None):
    """
    Perform dense matrix-vector multiplication with sparse-float spikes.

    When ``transpose=False``, computes ``weights[m, k] @ spikes[k] -> out[m]``
    (dense matrix times sparse-float vector).

    When ``transpose=True``, computes ``spikes[k] @ weights[k, n] -> out[n]``
    (sparse-float vector times dense matrix).

    Parameters
    ----------
    weights : array_like
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``. Can be a ``brainunit`` quantity.
    spikes : array_like
        The sparse-float vector with shape ``(k,)``. Nonzero entries
        contribute their float value (not just a binary indicator).
        Can be a ``brainunit`` quantity.
    transpose : bool
        If False, compute ``weights @ spikes``. If True, compute
        ``spikes @ weights``.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``, ``'warp'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : array_like
        Result vector. Shape ``(m,)`` when ``transpose=False``,
        or ``(n,)`` when ``transpose=True``. If inputs carry units, the
        result carries the product of the weight and spike units.

    Raises
    ------
    AssertionError
        If the inner dimensions of ``weights`` and ``spikes`` do not match.

    See Also
    --------
    dsfmv : Alias for ``spfloat_densemv(..., transpose=False)``.
    sfdvm : Alias for ``spfloat_densemv(..., transpose=True)``.
    spfloat_densemv_p_call : Low-level primitive call without unit handling.

    Notes
    -----
    The computation is event-driven: only the columns (or rows) of
    ``weights`` corresponding to nonzero entries of ``spikes`` are
    accumulated, with each weighted by the spike value.

    When ``transpose=False``, the operation computes:

    ``out[i] = sum_{j where s[j] != 0} W[i, j] * s[j]``

    When ``transpose=True``, the operation computes:

    ``out[j] = sum_{i where s[i] != 0} s[i] * W[i, j]``

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._dense.sparse_float import spfloat_densemv
        >>> weights = jnp.eye(3, dtype=jnp.float32)
        >>> spikes = jnp.array([2.0, 0.0, 3.0], dtype=jnp.float32)
        >>> spfloat_densemv(weights, spikes, transpose=False)
        Array([2., 0., 3.], dtype=float32)
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = spfloat_densemv_p_call(weight_val, spk_val, transpose=transpose, backend=backend)
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _spfloat_densemv_numba_kernel(transpose: bool, **kwargs):
    import numba

    if transpose:
        # spikes[k] @ weights[k,n] -> out[n]
        @numba.njit(fastmath=True)
        def kernel(weights, spikes, posts):
            posts[:] = 0.0
            for i in range(spikes.shape[0]):
                spk = spikes[i]
                if spk != 0.0:
                    posts += weights[i] * spk

    else:
        # weights[m,k] @ spikes[k] -> out[m]
        @numba.njit(fastmath=True)
        def kernel(weights, spikes, posts):
            posts[:] = 0.0
            for i in range(spikes.shape[0]):
                spk = spikes[i]
                if spk != 0.0:
                    posts += weights[:, i] * spk

    def run(weights, spikes):
        return numba_kernel(kernel, outs=kwargs['outs'])(weights, spikes)

    return run


def _spfloat_densemv_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    import warp
    from warp.jax_experimental import jax_kernel

    spike_length = spk_info.shape[0]
    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # spikes[k] @ weights[k,n] -> out[n]
        n = weight_info.shape[1]

        if spk_info.dtype == jnp.bool_:
            spk_float_info = jax.ShapeDtypeStruct(spk_info.shape, jnp.float32)
            spike_warp_info = jaxinfo_to_warpinfo(spk_float_info)

            @warp.kernel
            def kernel(
                weight_ref: weight_warp_info,
                spike_ref: spike_warp_info,
                out_ref: out_warp_info,
            ):
                j = warp.tid()
                r = weight_ref.dtype(0.0)
                for i in range(spike_length):
                    spk = spike_ref[i]
                    if spk != 0.0:
                        r += weight_ref[i, j] * spk
                out_ref[j] = r

            def run(weights, spikes):
                spikes = spikes.astype(jnp.float32)
                out_info = kwargs['outs'][0]
                fn = jax_kernel(kernel, launch_dims=[n], num_outputs=1, output_dims={'out_ref': out_info.shape})
                return fn(weights, spikes)

        else:
            spike_warp_info = jaxinfo_to_warpinfo(spk_info)

            @warp.kernel
            def kernel(
                weight_ref: weight_warp_info,
                spike_ref: spike_warp_info,
                out_ref: out_warp_info,
            ):
                j = warp.tid()
                r = weight_ref.dtype(0.0)
                for i in range(spike_length):
                    spk = spike_ref[i]
                    if spk != 0.0:
                        r += weight_ref[i, j] * spk
                out_ref[j] = r

            def run(weights, spikes):
                out_info = kwargs['outs'][0]
                fn = jax_kernel(kernel, launch_dims=[n], num_outputs=1, output_dims={'out_ref': out_info.shape})
                return fn(weights, spikes)

    else:
        # weights[m,k] @ spikes[k] -> out[m]
        m = weight_info.shape[0]

        if spk_info.dtype == jnp.bool_:
            spk_float_info = jax.ShapeDtypeStruct(spk_info.shape, jnp.float32)
            spike_warp_info = jaxinfo_to_warpinfo(spk_float_info)

            @warp.kernel
            def kernel(
                weight_ref: weight_warp_info,
                spike_ref: spike_warp_info,
                out_ref: out_warp_info,
            ):
                i_row = warp.tid()
                r = weight_ref.dtype(0.0)
                for j in range(spike_length):
                    spk = spike_ref[j]
                    if spk != 0.0:
                        r += weight_ref[i_row, j] * spk
                out_ref[i_row] = r

            def run(weights, spikes):
                spikes = spikes.astype(jnp.float32)
                out_info = kwargs['outs'][0]
                fn = jax_kernel(kernel, launch_dims=[m], num_outputs=1, output_dims={'out_ref': out_info.shape})
                return fn(weights, spikes)

        else:
            spike_warp_info = jaxinfo_to_warpinfo(spk_info)

            @warp.kernel
            def kernel(
                weight_ref: weight_warp_info,
                spike_ref: spike_warp_info,
                out_ref: out_warp_info,
            ):
                i_row = warp.tid()
                r = weight_ref.dtype(0.0)
                for j in range(spike_length):
                    spk = spike_ref[j]
                    if spk != 0.0:
                        r += weight_ref[i_row, j] * spk
                out_ref[i_row] = r

            def run(weights, spikes):
                out_info = kwargs['outs'][0]
                fn = jax_kernel(kernel, launch_dims=[m], num_outputs=1, output_dims={'out_ref': out_info.shape})
                return fn(weights, spikes)

    return run


def _spfloat_densemv_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    from jax.experimental import pallas as pl

    if transpose:
        # spikes[k] @ weights[k,n] -> out[n]
        n = weight_info.shape[1]
        block_dim = generate_block_dim(n, maximum=1024)

        def kernel(weight_ref, spike_ref, out_ref):
            i_col_block = pl.program_id(0)
            col_start = i_col_block * block_dim
            cols = col_start + jnp.arange(block_dim)
            mask = cols < weight_ref.shape[1]
            safe_cols = jnp.where(mask, cols, 0)

            def loop_fn(i_spike, temp):
                spike = spike_ref[i_spike]
                return jax.lax.cond(
                    spike != 0.0,
                    lambda out: out + jnp.where(
                        mask,
                        weight_ref[i_spike, safe_cols] * spike,
                        0.0,
                    ),
                    lambda out: out,
                    temp,
                )

            i_col_out = jax.lax.fori_loop(
                0, spike_ref.shape[0], loop_fn, jnp.zeros((block_dim,), dtype=weight_ref.dtype)
            )
            out_ref[safe_cols] = jnp.where(mask, i_col_out, 0.0)

        def run(weights, spikes):
            fn = pl.pallas_call(kernel, grid=(cdiv(n, block_dim),), out_shape=kwargs['outs'], backend='triton')
            return fn(weights, spikes)

    else:
        # weights[m,k] @ spikes[k] -> out[m]
        m = weight_info.shape[0]
        mat_block_dim = generate_block_dim(m, maximum=1024)

        def kernel(weight_ref, spike_ref, out_ref):
            i_row_block = pl.program_id(0)
            row_start = i_row_block * mat_block_dim
            rows = row_start + jnp.arange(mat_block_dim)
            mask = rows < weight_ref.shape[0]
            safe_rows = jnp.where(mask, rows, 0)

            def loop_fn(i_spike, temp):
                spike = spike_ref[i_spike]
                return jax.lax.cond(
                    spike != 0.0,
                    lambda out: out + jnp.where(
                        mask,
                        weight_ref[safe_rows, i_spike] * spike,
                        0.0,
                    ),
                    lambda out: out,
                    temp,
                )

            i_row_out = jax.lax.fori_loop(
                0, spike_ref.shape[0], loop_fn, jnp.zeros((mat_block_dim,), dtype=weight_ref.dtype)
            )
            out_ref[safe_rows] = jnp.where(mask, i_row_out, 0.0)

        def run(weights, spikes):
            fn = pl.pallas_call(kernel, grid=(cdiv(m, mat_block_dim),), out_shape=kwargs['outs'], backend='triton')
            return fn(weights, spikes)

    return run


def _spfloat_densemv_jvp_weights(w_dot, weights, spikes, *, transpose, **kwargs):
    return spfloat_densemv_p_call(w_dot, spikes, transpose=transpose, backend=kwargs['backend'])


def _spfloat_densemv_jvp_spikes(spk_dot, weights, spikes, *, transpose, **kwargs):
    if transpose:
        return [spk_dot @ weights]
    else:
        return [weights @ spk_dot]


def _spfloat_densemv_transpose_rule(ct, weights, spikes, *, transpose, **kwargs):
    ct = ct[0]
    if ad.is_undefined_primal(spikes):
        if transpose:
            ct_spikes = jnp.matmul(weights, ct)
        else:
            ct_spikes = jnp.matmul(ct, weights)
        return weights, (ad.Zero(spikes) if type(ct) is ad.Zero else ct_spikes)
    else:
        if transpose:
            ct_weights = jnp.outer(spikes, ct)
        else:
            ct_weights = jnp.outer(ct, spikes)
        return (ad.Zero(weights) if type(ct) is ad.Zero else ct_weights), spikes


def _spfloat_densemv_batching(args, axes, *, transpose, **kwargs):
    if transpose:
        # spikes[k] @ weights[k,n] -> out[n]
        if axes == (None, 0):
            r = spfloat_densemm(args[0], args[1], transpose=True)
            return [r], [0]
        if axes == (None, 1):
            r = spfloat_densemm(args[0], args[1].T, transpose=True)
            return [r], [0]
    else:
        # weights[m,k] @ spikes[k] -> out[m]
        if axes == (None, 0):
            r = args[0] @ args[1].T
            return [r], [1]
        if axes == (None, 1):
            r = args[0] @ args[1]
            return [r], [1]
    return general_batching_rule(spfloat_densemv_p, args, axes, transpose=transpose, **kwargs)


def _spfloat_densemv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    # transpose=False benchmark (dsfmv style)
    weights = jnp.asarray(np.random.randn(n_pre, n_post), dtype=dtype)
    spikes = jnp.asarray(np.random.randn(n_post), dtype=dtype)
    configs.append(BenchmarkConfig('transpose=False', (weights, spikes)))
    # transpose=True benchmark (sfdvm style)
    weights = jnp.asarray(np.random.randn(n_post, n_pre), dtype=dtype)
    spikes = jnp.asarray(np.random.randn(n_post), dtype=dtype)
    configs.append(BenchmarkConfig('transpose=True', (weights, spikes)))
    return configs


def spfloat_densemv_p_call(weights, spikes, *, transpose, backend: Optional[str] = None):
    """
    Low-level primitive call for sparse-float dense matrix-vector multiplication.

    This function validates input shapes, constructs the output shape
    descriptor, and invokes the ``spfloat_densemv_p`` JAX primitive. Unlike
    :func:`spfloat_densemv`, this function operates on raw numerical arrays
    without ``brainunit`` unit handling.

    Parameters
    ----------
    weights : jax.Array
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``.
    spikes : jax.Array
        The sparse-float vector with shape ``(k,)``.
    transpose : bool
        If False, compute ``weights @ spikes`` producing shape ``(m,)``.
        If True, compute ``spikes @ weights`` producing shape ``(n,)``.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``, ``'warp'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : list of jax.Array
        A single-element list containing the result vector. Shape ``(m,)``
        when ``transpose=False``, or ``(n,)`` when ``transpose=True``.

    Raises
    ------
    AssertionError
        If the inner dimensions of ``weights`` and ``spikes`` do not match.

    See Also
    --------
    spfloat_densemv : High-level function with unit handling.

    Notes
    -----
    This is the low-level entry point that bypasses unit handling. The
    mathematical operation is identical to :func:`spfloat_densemv`:

    When ``transpose=False``:

    ``out[i] = sum_{j where s[j] != 0} weights[i, j] * s[j]``

    When ``transpose=True``:

    ``out[j] = sum_{i where s[i] != 0} s[i] * weights[i, j]``

    The function returns a single-element list to conform to the JAX
    primitive output convention.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._dense.sparse_float import spfloat_densemv_p_call
        >>> weights = jnp.eye(3, dtype=jnp.float32)
        >>> spikes = jnp.array([2.0, 0.0, 3.0], dtype=jnp.float32)
        >>> spfloat_densemv_p_call(weights, spikes, transpose=False)
    """
    if transpose:
        # spikes[k] @ weights[k,n] -> out[n]
        assert spikes.shape[0] == weights.shape[0], (
            f'shapes {spikes.shape} and {weights.shape} not aligned: '
            f'{spikes.shape[0]} (dim 0) != {weights.shape[0]} (dim 0)'
        )
        out = jax.ShapeDtypeStruct([weights.shape[1]], weights.dtype)
    else:
        # weights[m,k] @ spikes[k] -> out[m]
        assert spikes.shape[0] == weights.shape[1], (
            f'spikes shape {spikes.shape} and weights shape {weights.shape} are not compatible'
        )
        out = jax.ShapeDtypeStruct([weights.shape[0]], weights.dtype)
    return spfloat_densemv_p(
        weights,
        spikes,
        outs=[out],
        transpose=transpose,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        backend=backend,
    )


spfloat_densemv_p = XLACustomKernel(
    'spfloat_densemv',
    doc="""
Low-level XLA custom-kernel primitive for ``spfloat_densemv``.

This ``XLACustomKernel`` instance dispatches the sparse-float dense matrix-vector
multiplication operation to registered backends (``numba``, ``warp``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

The primitive handles two transpose modes controlled by the ``transpose`` parameter:

- **transpose=False**: Computes ``weights[m, k] @ spikes[k] -> out[m]``
  (dense matrix multiplies sparse-float vector from the right).

- **transpose=True**: Computes ``spikes[k] @ weights[k, n] -> out[n]``
  (sparse-float vector multiplies dense matrix from the left).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``spfloat_densemv_p.available_backends(platform)``,
and the default backend can be configured with ``spfloat_densemv_p.set_default(platform, backend)``.

See Also
--------
spfloat_densemv : High-level user-facing function wrapper.
"""
)
"""
Low-level XLA custom-kernel primitive for ``spfloat_densemv``.

This ``XLACustomKernel`` instance dispatches the ``spfloat_densemv`` operation
to the backend registered below (for example ``numba``, ``warp``, and
``pallas``), using runtime shape/dtype metadata provided by the high-level
wrapper.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation
integrates correctly with ``jit``, ``vmap``, and autodiff.
"""
spfloat_densemv_p.def_numba_kernel(_spfloat_densemv_numba_kernel)
spfloat_densemv_p.def_warp_kernel(_spfloat_densemv_warp_kernel)
spfloat_densemv_p.def_pallas_kernel('gpu', _spfloat_densemv_pallas_kernel)
spfloat_densemv_p.def_jvp_rule2(_spfloat_densemv_jvp_weights, _spfloat_densemv_jvp_spikes)
spfloat_densemv_p.def_transpose_rule(_spfloat_densemv_transpose_rule)
spfloat_densemv_p.def_batching_rule(_spfloat_densemv_batching)
spfloat_densemv_p.def_call(spfloat_densemv_p_call)
spfloat_densemv_p.def_tags('dense', 'sparse_float')
spfloat_densemv_p.def_benchmark_data(_spfloat_densemv_benchmark_data)


# ==============================================================================
# Unified sparse-float dense matrix-matrix product (spfloat_densemm)
# ==============================================================================
#
# transpose=False: weights[m,k] @ spikes[k,n] -> out[m,n]  (old dsfmm)
# transpose=True:  spikes[m,k] @ weights[k,n] -> out[m,n]  (old sfdmm)
#
# Argument order is always (weights, spikes).


@namescope(static_argnames=['transpose'])
def spfloat_densemm(weights, spikes, *, transpose, backend: Optional[str] = None):
    """
    Perform dense matrix-matrix multiplication with sparse-float spikes.

    When ``transpose=False``, computes ``weights[m, k] @ spikes[k, n] -> out[m, n]``
    (dense matrix times sparse-float matrix).

    When ``transpose=True``, computes ``spikes[m, k] @ weights[k, n] -> out[m, n]``
    (sparse-float matrix times dense matrix).

    Parameters
    ----------
    weights : array_like
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``. Can be a ``brainunit`` quantity.
    spikes : array_like
        The sparse-float matrix. Shape ``(k, n)`` when ``transpose=False``,
        or ``(m, k)`` when ``transpose=True``. Can be boolean or float.
        Can be a ``brainunit`` quantity.
    transpose : bool
        If False, compute ``weights @ spikes`` (dense left, sparse right).
        If True, compute ``spikes @ weights`` (sparse left, dense right).
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``, ``'warp'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : array_like
        Result matrix with shape ``(m, n)``. If inputs carry units, the
        result carries the product of the weight and spike units.

    Raises
    ------
    AssertionError
        If the shared dimensions of ``weights`` and ``spikes`` do not match.

    See Also
    --------
    spfloat_densemv : Matrix-vector variant of sparse-float dense multiplication.
    spfloat_densemm_p_call : Low-level primitive call without unit handling.

    Notes
    -----
    When ``transpose=False``, the operation computes:

    ``out[i, j] = sum_{k where s[k, j] != 0} W[i, k] * s[k, j]``

    where ``W`` is the dense weight matrix and ``s`` is the sparse-float spike matrix.

    When ``transpose=True``, the operation computes:

    ``out[i, j] = sum_{k where s[i, k] != 0} s[i, k] * W[k, j]``

    where ``s`` is the sparse-float spike matrix and ``W`` is the dense weight matrix.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._dense.sparse_float import spfloat_densemm
        >>> weights = jnp.eye(3, dtype=jnp.float32)
        >>> spikes = jnp.array([[2.0, 0.0],
        ...                     [0.0, 3.0],
        ...                     [1.0, 0.0]], dtype=jnp.float32)
        >>> spfloat_densemm(weights, spikes, transpose=False)
        Array([[2., 0.],
               [0., 3.],
               [1., 0.]], dtype=float32)
    """
    with jax.ensure_compile_time_eval():
        weights = u.math.asarray(weights)
        spikes = u.math.asarray(spikes)
    weight_val, wunit = u.split_mantissa_unit(weights)
    spk_val, spkunit = u.split_mantissa_unit(spikes)
    r = spfloat_densemm_p_call(weight_val, spk_val, transpose=transpose, backend=backend)
    return u.maybe_decimal(r[0] * wunit * spkunit)


def _spfloat_densemm_numba_kernel(transpose: bool, **kwargs):
    import numba

    if transpose:
        # spikes[m,k] @ weights[k,n] -> out[m,n]
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(weights, spikes, posts):
            for i_m in numba.prange(spikes.shape[0]):
                out = np.zeros(weights.shape[1], dtype=posts.dtype)
                for i_n in range(weights.shape[1]):
                    r = 0.0
                    for i_k in range(spikes.shape[1]):
                        spk = spikes[i_m, i_k]
                        if spk != 0.0:
                            r += weights[i_k, i_n] * spk
                    out[i_n] = r
                posts[i_m] = out
    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
        @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
        def kernel(weights, spikes, posts):
            for i_n in numba.prange(spikes.shape[1]):
                out = np.zeros(weights.shape[0], dtype=weights.dtype)
                for i_k in range(spikes.shape[0]):
                    spk = spikes[i_k, i_n]
                    if spk != 0.0:
                        out += weights[:, i_k] * spk
                posts[:, i_n] = out

    def run(weights, spikes):
        return numba_kernel(kernel, outs=kwargs['outs'])(weights, spikes)

    return run


def _spfloat_densemm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # spikes[m,k] @ weights[k,n] -> out[m,n]
        m, k = spk_info.shape
        n = weight_info.shape[1]

        if spk_info.dtype == jnp.bool_:
            spk_float_info = jax.ShapeDtypeStruct(spk_info.shape, jnp.float32)
            spike_warp_info = jaxinfo_to_warpinfo(spk_float_info)

            @warp.kernel
            def kernel(
                weight_ref: weight_warp_info,
                spike_ref: spike_warp_info,
                out_ref: out_warp_info,
            ):
                i_m, i_n = warp.tid()
                r = weight_ref.dtype(0.0)
                for i_k in range(k):
                    spk = spike_ref[i_m, i_k]
                    if spk != 0.0:
                        r += weight_ref[i_k, i_n] * spk
                out_ref[i_m, i_n] = r

            def run(weights, spikes):
                spikes = spikes.astype(jnp.float32)
                out_info = kwargs['outs'][0]
                fn = jax_kernel(kernel, launch_dims=(m, n), num_outputs=1, output_dims={'out_ref': out_info.shape})
                return fn(weights, spikes)

        else:
            spike_warp_info = jaxinfo_to_warpinfo(spk_info)

            @warp.kernel
            def kernel(
                weight_ref: weight_warp_info,
                spike_ref: spike_warp_info,
                out_ref: out_warp_info,
            ):
                i_m, i_n = warp.tid()
                r = weight_ref.dtype(0.0)
                for i_k in range(k):
                    spk = spike_ref[i_m, i_k]
                    if spk != 0.0:
                        r += weight_ref[i_k, i_n] * spk
                out_ref[i_m, i_n] = r

            def run(weights, spikes):
                out_info = kwargs['outs'][0]
                fn = jax_kernel(kernel, launch_dims=(m, n), num_outputs=1, output_dims={'out_ref': out_info.shape})
                return fn(weights, spikes)

    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
        k = spk_info.shape[0]
        n = spk_info.shape[1]
        m = weight_info.shape[0]

        if spk_info.dtype == jnp.bool_:
            spk_float_info = jax.ShapeDtypeStruct(spk_info.shape, jnp.float32)
            spike_warp_info = jaxinfo_to_warpinfo(spk_float_info)

            @warp.kernel
            def kernel(
                weight_ref: weight_warp_info,
                spike_ref: spike_warp_info,
                out_ref: out_warp_info,
            ):
                i_m, i_n = warp.tid()
                r = weight_ref.dtype(0.0)
                for i_k in range(k):
                    spk = spike_ref[i_k, i_n]
                    if spk != 0.0:
                        r += weight_ref[i_m, i_k] * spk
                out_ref[i_m, i_n] = r

            def run(weights, spikes):
                spikes = spikes.astype(jnp.float32)
                out_info = kwargs['outs'][0]
                fn = jax_kernel(kernel, launch_dims=(m, n), num_outputs=1, output_dims={'out_ref': out_info.shape})
                return fn(weights, spikes)

        else:
            spike_warp_info = jaxinfo_to_warpinfo(spk_info)

            @warp.kernel
            def kernel(
                weight_ref: weight_warp_info,
                spike_ref: spike_warp_info,
                out_ref: out_warp_info,
            ):
                i_m, i_n = warp.tid()
                r = weight_ref.dtype(0.0)
                for i_k in range(k):
                    spk = spike_ref[i_k, i_n]
                    if spk != 0.0:
                        r += weight_ref[i_m, i_k] * spk
                out_ref[i_m, i_n] = r

            def run(weights, spikes):
                out_info = kwargs['outs'][0]
                fn = jax_kernel(kernel, launch_dims=(m, n), num_outputs=1, output_dims={'out_ref': out_info.shape})
                return fn(weights, spikes)

    return run


def _spfloat_densemm_pallas_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spk_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    from jax.experimental import pallas as pl

    if transpose:
        # spikes[m,k] @ weights[k,n] -> out[m,n]
        m = spk_info.shape[0]
        k, n = weight_info.shape
        block_dim = generate_block_dim(n, maximum=1024)

        def kernel(
            weight_ref,  # [k, n]
            spike_ref,  # [m, k]
            out_ref,  # [m, n]
        ):
            i_m = pl.program_id(0)
            i_n_block = pl.program_id(1)
            col_start = i_n_block * block_dim
            cols = col_start + jnp.arange(block_dim)
            mask = cols < n
            safe_cols = jnp.where(mask, cols, 0)

            def loop_fn(i_k, temp):
                spike = spike_ref[i_m, i_k]
                return jax.lax.cond(
                    spike != 0.0,
                    lambda out: out + jnp.where(
                        mask,
                        weight_ref[i_k, safe_cols] * spike,
                        0.0,
                    ),
                    lambda out: out,
                    temp,
                )

            final_out = jax.lax.fori_loop(0, k, loop_fn, jnp.zeros(block_dim, dtype=weight_ref.dtype))
            out_ref[i_m, safe_cols] = jnp.where(mask, final_out, 0.0)

        def run(weights, spikes):
            fn = pl.pallas_call(kernel, grid=(m, cdiv(n, block_dim)), out_shape=kwargs['outs'], backend='triton')
            return fn(weights, spikes)

    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
        k = spk_info.shape[0]
        n = spk_info.shape[1]
        m = weight_info.shape[0]
        block_dim = generate_block_dim(m, maximum=1024)

        def kernel(
            weight_ref,  # [m, k]
            spike_ref,  # [k, n]
            out_ref,  # [m, n]
        ):
            i_n = pl.program_id(0)
            i_m_block = pl.program_id(1)
            row_start = i_m_block * block_dim
            rows = row_start + jnp.arange(block_dim)
            mask = rows < m
            safe_rows = jnp.where(mask, rows, 0)

            def loop_fn(i_k, temp):
                spike = spike_ref[i_k, i_n]
                return jax.lax.cond(
                    spike != 0.0,
                    lambda out: out + jnp.where(
                        mask,
                        weight_ref[safe_rows, i_k] * spike,
                        0.0,
                    ),
                    lambda out: out,
                    temp,
                )

            final_out = jax.lax.fori_loop(0, k, loop_fn, jnp.zeros(block_dim, dtype=weight_ref.dtype))
            out_ref[safe_rows, i_n] = jnp.where(mask, final_out, 0.0)

        def run(weights, spikes):
            fn = pl.pallas_call(kernel, grid=(n, cdiv(m, block_dim)), out_shape=kwargs['outs'], backend='triton')
            return fn(weights, spikes)

    return run


def _spfloat_densemm_jvp_weights(w_dot, weights, spikes, *, transpose, **kwargs):
    return spfloat_densemm_p_call(w_dot, spikes, transpose=transpose, backend=kwargs['backend'])


def _spfloat_densemm_jvp_spikes(spk_dot, weights, spikes, *, transpose, **kwargs):
    if transpose:
        return [spk_dot @ weights]
    else:
        return [weights @ spk_dot]


def _spfloat_densemm_transpose_rule(ct, weights, spikes, *, transpose, **kwargs):
    ct = ct[0]
    if ad.is_undefined_primal(spikes):
        if transpose:
            # spikes[m,k] @ weights[k,n] -> out[m,n]
            # ct[m,n] @ weights.T[n,k] -> ct_spikes[m,k]
            ct_spikes = ct @ weights.T
        else:
            # weights[m,k] @ spikes[k,n] -> out[m,n]
            # weights.T[k,m] @ ct[m,n] -> ct_spikes[k,n]
            ct_spikes = weights.T @ ct
        return weights, (ad.Zero(spikes) if type(ct) is ad.Zero else ct_spikes)
    else:
        if transpose:
            # spikes[m,k] @ weights[k,n] -> out[m,n]
            # ct[m,n], spikes[m,k] -> ct_weights[k,n]
            # spikes.T[k,m] @ ct[m,n] -> ct_weights[k,n]
            ct_weights = spfloat_densemm(ct, spikes.T, transpose=True)
        else:
            # weights[m,k] @ spikes[k,n] -> out[m,n]
            # ct[m,n], spikes[k,n] -> ct_weights[m,k]
            # ct[m,n] @ spikes.T[n,k] -> ct_weights[m,k]
            ct_weights = spfloat_densemm(ct, spikes.T, transpose=False)
        return (ad.Zero(weights) if type(ct) is ad.Zero else ct_weights), spikes


def _spfloat_densemm_batching(args, axes, *, transpose, **kwargs):
    # Simplified batching rule using general_batching_rule as fallback
    return general_batching_rule(spfloat_densemm_p, args, axes, transpose=transpose, **kwargs)


def _spfloat_densemm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    # transpose=False benchmark (dsfmm style)
    weights = jnp.asarray(np.random.randn(n_pre, n_post), dtype=dtype)
    spikes = jnp.asarray(np.random.randn(n_post, 10), dtype=dtype)
    configs.append(BenchmarkConfig('transpose=False', (weights, spikes)))
    # transpose=True benchmark (sfdmm style)
    weights = jnp.asarray(np.random.randn(n_post, n_post), dtype=dtype)
    spikes = jnp.asarray(np.random.randn(10, n_post), dtype=dtype)
    configs.append(BenchmarkConfig('transpose=True', (weights, spikes)))
    return configs


def spfloat_densemm_p_call(weights, spikes, *, transpose, backend: Optional[str] = None):
    """
    Low-level primitive call for sparse-float dense matrix-matrix multiplication.

    This function validates input shapes, constructs the output shape
    descriptor, and invokes the ``spfloat_densemm_p`` JAX primitive. Unlike
    :func:`spfloat_densemm`, this function operates on raw numerical arrays
    without ``brainunit`` unit handling.

    Parameters
    ----------
    weights : jax.Array
        The weight matrix. Shape ``(m, k)`` when ``transpose=False``,
        or ``(k, n)`` when ``transpose=True``.
    spikes : jax.Array
        The sparse-float matrix. Shape ``(k, n)`` when ``transpose=False``,
        or ``(m, k)`` when ``transpose=True``.
    transpose : bool
        If False, compute ``weights @ spikes`` producing shape ``(m, n)``.
        If True, compute ``spikes @ weights`` producing shape ``(m, n)``.
    backend : str, optional
        Backend to use for the computation. One of ``'numba'``, ``'warp'``,
        ``'pallas'``, or ``None`` (auto-select).

    Returns
    -------
    result : list of jax.Array
        A single-element list containing the result matrix with shape
        ``(m, n)``.

    Raises
    ------
    AssertionError
        If the shared dimensions of ``weights`` and ``spikes`` do not match.

    See Also
    --------
    spfloat_densemm : High-level function with unit handling.

    Notes
    -----
    This is the low-level entry point that bypasses unit handling. The
    mathematical operation is identical to :func:`spfloat_densemm`:

    When ``transpose=False``:

    ``out[i, j] = sum_{k where s[k, j] != 0} weights[i, k] * s[k, j]``

    When ``transpose=True``:

    ``out[i, j] = sum_{k where s[i, k] != 0} s[i, k] * weights[k, j]``

    The function returns a single-element list to conform to the JAX
    primitive output convention.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._dense.sparse_float import spfloat_densemm_p_call
        >>> weights = jnp.eye(3, dtype=jnp.float32)
        >>> spikes = jnp.array([[2.0, 0.0],
        ...                     [0.0, 3.0],
        ...                     [1.0, 0.0]], dtype=jnp.float32)
        >>> spfloat_densemm_p_call(weights, spikes, transpose=False)
    """
    if transpose:
        # spikes[m,k] @ weights[k,n] -> out[m,n]
        assert spikes.shape[1] == weights.shape[0], (
            f'spikes shape {spikes.shape} and weights shape {weights.shape} do not match: '
            f'spikes.shape[1] ({spikes.shape[1]}) != weights.shape[0] ({weights.shape[0]})'
        )
        out = jax.ShapeDtypeStruct([spikes.shape[0], weights.shape[1]], weights.dtype)
    else:
        # weights[m,k] @ spikes[k,n] -> out[m,n]
        assert weights.shape[1] == spikes.shape[0], (
            f'weights.shape[1] ({weights.shape[1]}) != spikes.shape[0] ({spikes.shape[0]})'
            f', weights: {weights.shape}, spikes: {spikes.shape}'
        )
        out = jax.ShapeDtypeStruct([weights.shape[0], spikes.shape[1]], weights.dtype)
    return spfloat_densemm_p(
        weights,
        spikes,
        outs=[out],
        transpose=transpose,
        spk_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        backend=backend,
    )


spfloat_densemm_p = XLACustomKernel(
    'spfloat_densemm',
    doc="""
Low-level XLA custom-kernel primitive for ``spfloat_densemm``.

This ``XLACustomKernel`` instance dispatches the sparse-float dense matrix-matrix
multiplication operation to registered backends (``numba``, ``warp``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

The primitive handles two transpose modes controlled by the ``transpose`` parameter:

- **transpose=False**: Computes ``weights[m, k] @ spikes[k, n] -> out[m, n]``
  (dense matrix multiplies sparse-float matrix from the right).

- **transpose=True**: Computes ``spikes[m, k] @ weights[k, n] -> out[m, n]``
  (sparse-float matrix multiplies dense matrix from the left).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``spfloat_densemm_p.available_backends(platform)``,
and the default backend can be configured with ``spfloat_densemm_p.set_default(platform, backend)``.

See Also
--------
spfloat_densemm : High-level user-facing function wrapper.
"""
)
spfloat_densemm_p.def_numba_kernel(_spfloat_densemm_numba_kernel)
spfloat_densemm_p.def_warp_kernel(_spfloat_densemm_warp_kernel)
spfloat_densemm_p.def_pallas_kernel('gpu', _spfloat_densemm_pallas_kernel)
spfloat_densemm_p.def_jvp_rule2(_spfloat_densemm_jvp_weights, _spfloat_densemm_jvp_spikes)
spfloat_densemm_p.def_transpose_rule(_spfloat_densemm_transpose_rule)
spfloat_densemm_p.def_batching_rule(_spfloat_densemm_batching)
spfloat_densemm_p.def_call(spfloat_densemm_p_call)
spfloat_densemm_p.def_tags('dense', 'sparse_float')
spfloat_densemm_p.def_benchmark_data(_spfloat_densemm_benchmark_data)
