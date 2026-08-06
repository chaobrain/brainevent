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
from typing import Optional, Sequence

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._data import _initialize_seed, _initialize_conn_length
from brainevent._misc import namescope
from brainevent._numba_random import get_numba_light_rng_funcs
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig
from brainevent._op import load_cuda_file
from brainevent._typing import Data, MatrixShape
from .float import _normalize_chunk_size, _MV_STRIDE, _MM_STRIDE, jitumv_p_call, jitumm_p_call
from brainevent._op.util import dtype_suffix

__all__ = [
    "binary_jitumv",
    "binary_jitumv_p",
    "binary_jitumm",
    "binary_jitumm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitumv(
    w_low: Data,
    w_high: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    """
    Event-driven matrix-vector product with a JIT uniform connectivity matrix.

    Computes the product of a just-in-time generated sparse matrix with
    uniformly distributed weights and a binary event vector. Only non-zero
    (event) entries in ``vector`` contribute to the output, making this
    operation efficient for spike-based neural network simulations.

    The sparse matrix ``A`` of shape ``(m, n)`` is never materialized. Each
    entry ``A[i, j]`` is drawn from ``Uniform(w_low, w_high)`` with
    probability ``prob``, seeded by ``seed``.

    Parameters
    ----------
    w_low : Data
        Lower bound of the uniform weight distribution. Scalar value, optionally
        with physical units (``brainunit.Quantity``).
    w_high : Data
        Upper bound of the uniform weight distribution. Must have the same
        dimension (units) as ``w_low``.
    prob : float
        Connection probability in [0, 1]. Determines the fraction of
        non-zero entries in each row/column of the connectivity matrix.
    vector : Data
        Input event vector. Can be boolean or floating-point; non-zero entries
        are treated as active events. Length must match the appropriate matrix
        dimension (``n`` if ``transpose=False``, ``m`` if ``transpose=True``).
    seed : int, optional
        Random seed for reproducible connectivity patterns. If None, a random
        seed is generated at compile time.
    shape : MatrixShape
        Shape ``(m, n)`` of the logical connectivity matrix.
    transpose : bool, optional
        If True, compute ``A.T @ vector`` instead of ``A @ vector``.
        Default is False.
    corder : bool, optional
        Memory layout order for the connectivity generation. True for C-order
        (row-major), False for Fortran-order (column-major). Default is True.
    backend : str, optional
        Computation backend. One of ``'numba'`` or ``'pallas'``.
        If None, the default backend is used.

    Returns
    -------
    Data
        Result vector of length ``m`` (if ``transpose=False``) or ``n``
        (if ``transpose=True``). Carries the product of units from the weight
        and the vector if either has physical units.

    See Also
    --------
    binary_jitumm : Event-driven matrix-matrix variant.
    jitumv : Float (non-event) matrix-vector variant.

    Notes
    -----
    The connectivity matrix ``A`` of shape ``(m, n)`` follows the model:

        ``A[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Uniform(w_low, w_high)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent, both determined by ``seed``.

    The event-driven matrix-vector product computes:

        ``result[i] = sum_{j : vector[j] is active} A[i, j]``

    where "active" means ``True`` for boolean arrays or ``> 0`` for float arrays.
    Only positions where ``vector[j]`` is active contribute, making this efficient
    when the event vector is sparse. The full expansion is:

        ``result[i] = sum_{j} U[i, j] * B[i, j] * 1_{vector[j] active}``

    When ``transpose=True``, the operation becomes ``result = A^T @ vector``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_uniform.binary import binary_jitumv
        >>> events = jnp.array([True, False, True, True, False])
        >>> result = binary_jitumv(
        ...     0.1, 0.5, 0.2, events, seed=42,
        ...     shape=(3, 5), transpose=False, corder=True,
        ... )
        >>> result.shape
        (3,)
    """
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    seed = _initialize_seed(seed)
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = binary_jitumv_p_call(
        w_low,
        w_high,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitumm(
    w_low: Data,
    w_high: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    """
    Event-driven matrix-matrix product with a JIT uniform connectivity matrix.

    Computes the product of a just-in-time generated sparse matrix with
    uniformly distributed weights and a binary event matrix ``B``. Each column
    of ``B`` is treated as an independent event vector, and only non-zero
    entries contribute to the output.

    The sparse matrix ``A`` of shape ``(m, n)`` is never materialized. Each
    entry ``A[i, j]`` is drawn from ``Uniform(w_low, w_high)`` with
    probability ``prob``, seeded by ``seed``.

    Parameters
    ----------
    w_low : Data
        Lower bound of the uniform weight distribution. Scalar value, optionally
        with physical units (``brainunit.Quantity``).
    w_high : Data
        Upper bound of the uniform weight distribution. Must have the same
        dimension (units) as ``w_low``.
    prob : float
        Connection probability in [0, 1]. Determines the fraction of
        non-zero entries in the connectivity matrix.
    B : Data
        Input event matrix of shape ``(n, k)`` (if ``transpose=False``) or
        ``(m, k)`` (if ``transpose=True``). Can be boolean or floating-point;
        non-zero entries are treated as active events.
    seed : int, optional
        Random seed for reproducible connectivity patterns. If None, a random
        seed is generated at compile time.
    shape : MatrixShape
        Shape ``(m, n)`` of the logical connectivity matrix.
    transpose : bool, optional
        If True, compute ``A.T @ B`` instead of ``A @ B``.
        Default is False.
    corder : bool, optional
        Memory layout order for the connectivity generation. True for C-order
        (row-major), False for Fortran-order (column-major). Default is True.
    backend : str, optional
        Computation backend. One of ``'numba'`` or ``'pallas'``.
        If None, the default backend is used.

    Returns
    -------
    Data
        Result matrix of shape ``(m, k)`` (if ``transpose=False``) or
        ``(n, k)`` (if ``transpose=True``). Carries the product of units
        from the weight and ``B`` if either has physical units.

    See Also
    --------
    binary_jitumv : Event-driven matrix-vector variant.
    jitumm : Float (non-event) matrix-matrix variant.

    Notes
    -----
    The connectivity matrix ``A`` of shape ``(m, n)`` follows the model:

        ``A[i, j] = U[i, j] * B_conn[i, j]``

    where ``U[i, j] ~ Uniform(w_low, w_high)`` and ``B_conn[i, j] ~ Bernoulli(prob)``
    are independent, both determined by ``seed``.

    The event-driven matrix-matrix product computes:

        ``result[i, j] = sum_{k : B[k, j] is active} A[i, k]``

    where "active" means ``True`` for boolean arrays or ``> 0`` for float arrays.
    Each column of ``B`` is treated as an independent event vector. The full
    expansion is:

        ``result[i, j] = sum_{k} U[i, k] * B_conn[i, k] * 1_{B[k, j] active}``

    When ``transpose=True``, the operation becomes ``result = A^T @ B``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_uniform.binary import binary_jitumm
        >>> B = jnp.array([[True, False], [False, True], [True, True],
        ...                [False, False], [True, False]])
        >>> result = binary_jitumm(
        ...     0.1, 0.5, 0.2, B, seed=42,
        ...     shape=(3, 5), transpose=False, corder=True,
        ... )
        >>> result.shape
        (3, 2)
    """
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    seed = _initialize_seed(seed)
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = binary_jitumm_p_call(
        w_low,
        w_high,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


# Kernel generators for JIT connection SPMV

def _jitumv_numba_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for binary event JIT-uniform matrix-vector product.

    Parameters
    ----------
    vector_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input event vector.
    corder : bool, optional
        If True, iterate over output elements (columns) in the outer loop.
        If False, iterate over input elements (rows) in the outer loop.
        Default is True.
    **kwargs
        Additional keyword arguments, must include ``outs`` specifying
        output shape/dtype information.

    Returns
    -------
    callable
        A function ``kernel(w_low, w_high, clen, vector, seed)`` that
        executes the Numba-compiled kernel and returns the result.
    """
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MV_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)
    is_bool = np.dtype(vector_info.dtype) in (np.dtype('bool'), np.dtype('int8'))

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, vector, seed, posts):
            m = posts.shape[0]
            k = vector.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            span = w_high0 - w_low0
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                out = np.float64(0.)
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            if is_bool:
                                active = vector[j]
                            else:
                                active = vector[j] > 0.
                            if active:
                                u01 = _rng_uniform01(seed0, row, j)
                                out += w_low0 + u01 * span
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = out
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, vector, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = vector.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            span = w_high0 - w_low0
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                if is_bool:
                    active_row = vector[row]
                else:
                    active_row = vector[row] > 0.
                if not active_row:
                    continue
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            u01 = _rng_uniform01(seed0, row, j)
                            posts[j] += w_low0 + u01 * span
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, vector, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, vector, seed)

    return kernel


def _binary_jitumv_cuda_kernel(
    corder: bool = True,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_jitumv.cu'),
        name='jit_uniform_binary_jitumv',
    )
    wt_sfx = dtype_suffix(kwargs['w_low_info'].dtype)
    vector_info = kwargs['vector_info']
    variant = 'notrans' if corder else 'trans'
    kernel_name = f'jit_uniform_binary_jitumv.{variant}{wt_sfx}'
    k = int(vector_info.shape[0])
    n_words = (k + 31) // 32
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)
    is_bool = np.dtype(vector_info.dtype) in (np.dtype('bool'), np.dtype('int8'))

    def kernel(w_low, w_high, clen, vector, seed):
        spikes = vector.astype(jnp.int8) if is_bool else (vector > 0).astype(jnp.int8)
        packed = jax.ffi.ffi_call(
            'jit_uniform_binary_jitumv.pack_bool',
            jax.ShapeDtypeStruct((n_words,), jnp.uint32),
        )(spikes)
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w_low, w_high, clen, seed, packed,
            vector_size=np.int32(k), chunk_size=np.int32(chunk_size),
        )

    return kernel


def _jitumv_jvp_v(v_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the vector argument of the binary JIT-uniform matrix-vector product.

    Parameters
    ----------
    v_dot : jax.Array
        Tangent vector for the ``vector`` argument.
    w_low, w_high, clen, vector, seed : jax.Array
        Primal values of the primitive's arguments.
    shape : MatrixShape
        Shape of the connectivity matrix.
    transpose : bool
        Whether the transposed operation is used.
    corder : bool
        Memory layout order flag.
    **kwargs
        Additional keyword arguments including ``backend``.

    Returns
    -------
    list
        Single-element list containing the JVP result.
    """
    return jitumv_p_call(
        w_low, w_high, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitumv_jvp_wloc(w_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w_low`` argument of the binary JIT-uniform matrix-vector product.

    Parameters
    ----------
    w_dot : jax.Array
        Tangent vector for the ``w_low`` argument.
    w_low, w_high, clen, vector, seed : jax.Array
        Primal values of the primitive's arguments.
    shape : MatrixShape
        Shape of the connectivity matrix.
    transpose : bool
        Whether the transposed operation is used.
    corder : bool
        Memory layout order flag.
    **kwargs
        Additional keyword arguments including ``backend``.

    Returns
    -------
    list
        Single-element list containing the JVP result.
    """
    count_basis = binary_jitumv_p_call(
        w_dot, w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder,
        backend=kwargs['backend'],
    )[0]
    high_basis = binary_jitumv_p_call(
        0., w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder,
        backend=kwargs['backend'],
    )[0]
    return [count_basis - high_basis]


def _jitumv_jvp_wscale(w_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w_high`` argument of the binary JIT-uniform matrix-vector product.

    Parameters
    ----------
    w_dot : jax.Array
        Tangent vector for the ``w_high`` argument.
    w_low, w_high, clen, vector, seed : jax.Array
        Primal values of the primitive's arguments.
    shape : MatrixShape
        Shape of the connectivity matrix.
    transpose : bool
        Whether the transposed operation is used.
    corder : bool
        Memory layout order flag.
    **kwargs
        Additional keyword arguments including ``backend``.

    Returns
    -------
    list
        Single-element list containing the JVP result.
    """
    return binary_jitumv_p_call(
        0., w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitumv_transpose_rules(ct, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the binary JIT-uniform matrix-vector product.

    Implements the VJP transpose for differentiation through the primitive.
    Supports transposing with respect to the ``vector``, ``w_low``, or ``w_high``
    arguments.

    Parameters
    ----------
    ct : list
        Cotangent of the output.
    w_low, w_high, clen, vector, seed : jax.Array or ad.UndefinedPrimal
        Primal values or undefined primals of the primitive's arguments.
    shape : MatrixShape
        Shape of the connectivity matrix.
    transpose : bool
        Whether the transposed operation was used in the forward pass.
    corder : bool
        Memory layout order flag.
    **kwargs
        Additional keyword arguments including ``backend``.

    Returns
    -------
    tuple
        Cotangents for each input argument (w_low, w_high, clen, vector, seed).

    Raises
    ------
    NotImplementedError
        If the undefined primal is not ``vector``, ``w_low``, or ``w_high``.

    Notes
    -----
    For the weight bounds, the transpose uses an affine decomposition of the
    output with respect to ``w_low`` and ``w_high``:

        ``y = w_low * C + (w_high - w_low) * U``

    where ``U = y(0, 1)`` and ``C = y(1, 1)``.
    """
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    needs_w_low = ad.is_undefined_primal(w_low)
    needs_w_high = ad.is_undefined_primal(w_high)
    needs_vector = ad.is_undefined_primal(vector)
    dw_low = w_low
    dw_high = w_high
    d_vector = vector

    if needs_vector and (needs_w_low or needs_w_high):
        raise NotImplementedError(
            'Transpose rules for binary_jitumv do not support '
            'differentiating weight bounds and vector in the same transpose.'
        )
    if needs_vector:
        d_vector = jitumv_p_call(
            w_low,
            w_high,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            backend=kwargs['backend'],
        )[0]

    if needs_w_low or needs_w_high:
        ones = jnp.ones((1,), dtype=ct.dtype)
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        high_basis = binary_jitumv_p_call(
            zeros,
            ones,
            clen,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=kwargs['backend'],
        )[0]
        if needs_w_low:
            count_basis = binary_jitumv_p_call(
                ones,
                ones,
                clen,
                vector,
                seed,
                shape=shape,
                transpose=transpose,
                corder=corder,
                backend=kwargs['backend'],
            )[0]
            dw_low = jnp.expand_dims(jnp.sum(ct * (count_basis - high_basis)), axis=0)
        if needs_w_high:
            dw_high = jnp.expand_dims(jnp.sum(ct * high_basis), axis=0)

    if needs_vector or needs_w_low or needs_w_high:
        return dw_low, dw_high, clen, d_vector, seed

    raise NotImplementedError(
        f"Transpose rule for {ct} not implemented "
        f"for event-driven COO matrix-vector product."
    )


def _jitumv_batching(
    args,
    axes,
    **kwargs
):
    """
    Batching rule for the binary JIT-uniform matrix-vector product primitive.

    Handles ``vmap`` over the vector argument by promoting the operation to
    a matrix-matrix product (``binary_jitumm_p_call``).

    Parameters
    ----------
    args : tuple
        Batched arguments ``(w_low, w_high, clen, vector, seed)``.
    axes : tuple
        Batch axis for each argument (None means not batched).
    **kwargs
        Additional keyword arguments including ``shape``, ``transpose``,
        ``corder``, and ``backend``.

    Returns
    -------
    tuple
        A pair ``(results, out_axes)`` where ``results`` is the batched output
        and ``out_axes`` indicates the batch dimension of each output.
    """
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitumm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitumm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(binary_jitumv_p, args, axes, **kwargs)


def _binary_jitumv_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the binary JIT-uniform matrix-vector product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering different combinations
        of transpose, corder, and boolean/float event types.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                w_low = jnp.zeros(1, dtype=dtype)
                w_high = jnp.ones(1, dtype=dtype)
                clen = jnp.atleast_1d(jnp.asarray(np.ceil(2.0 / prob), dtype=jnp.int32))
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    vector = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    vector = jnp.asarray(np.random.rand(v_size), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(BenchmarkConfig(name, (w_low, w_high, clen, vector, seed), {
                    'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
                }))
    return configs


def binary_jitumv_p_call(
    w_low,
    w_high,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for the binary event JIT-uniform matrix-vector product.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``binary_jitumv_p`` XLA custom kernel. This function expects
    pre-processed arguments (mantissa-only arrays, connection length instead of
    probability).

    Parameters
    ----------
    w_low : jax.Array
        Lower weight bound as a 1-D array of shape ``(1,)`` with floating dtype.
    w_high : jax.Array
        Upper weight bound as a 1-D array of shape ``(1,)`` with the same dtype
        as ``w_low``.
    clen : jax.Array
        Connection length parameter as a 1-D array of shape ``(1,)``, derived
        from the connection probability via ``ceil(2 / prob)``.
    vector : jax.Array
        Input event vector, a 1-D array. Length must match ``shape[1]``
        (if ``transpose=False``) or ``shape[0]`` (if ``transpose=True``).
    seed : jax.Array
        Random seed as a 1-D array of shape ``(1,)``.
    shape : Sequence[int]
        Shape ``(m, n)`` of the logical connectivity matrix.
    transpose : bool
        If True, compute ``A.T @ vector``; otherwise compute ``A @ vector``.
    corder : bool
        Memory layout order flag for the connectivity generation.
    backend : str, optional
        Computation backend (``'numba'`` or ``'pallas'``).

    Returns
    -------
    tuple
        A single-element tuple containing the result array of shape
        ``(shape[0],)`` or ``(shape[1],)`` depending on ``transpose``.

    Raises
    ------
    AssertionError
        If any input shape, dtype, or dimension constraint is violated.

    See Also
    --------
    binary_jitumv : High-level wrapper with unit handling and seed initialization.
    """
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)
    assert jnp.issubdtype(w_low.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w_low.dtype == w_high.dtype, "w_low and w_high must have the same dtype."

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert w_low.shape == (1,), f"The weight shape should be (1,), but got {w_low.shape}."
    assert w_high.shape == (1,), f"The weight shape should be (1,), but got {w_high.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_low.dtype)
    )

    return binary_jitumv_p(
        w_low,
        w_high,
        clen,
        vector,
        seed,
        outs=[out_info],
        w_low_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        w_high_info=jax.ShapeDtypeStruct(w_high.shape, w_high.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


binary_jitumv_p = XLACustomKernel(
    'binary_jitumv',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_jitumv``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) JIT uniform connectivity
matrix-vector multiplication operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

In this operation, the connectivity matrix has weights uniformly distributed between specified
bounds, and the input vector is treated as binary events (spikes). Only active events
contribute to the output computation.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_jitumv_p.available_backends(platform)``,
and the default backend can be configured with ``binary_jitumv_p.set_default(platform, backend)``.

See Also
--------
binary_jitumv : High-level user-facing function wrapper.
"""
)
binary_jitumv_p.def_numba_kernel(_jitumv_numba_kernel_generator)
binary_jitumv_p.def_cuda_raw_kernel(_binary_jitumv_cuda_kernel, asdefault=True)
binary_jitumv_p.def_jvp_rule2(_jitumv_jvp_wloc, _jitumv_jvp_wscale, None, _jitumv_jvp_v, None)
binary_jitumv_p.def_transpose_rule(_jitumv_transpose_rules)
binary_jitumv_p.def_batching_rule(_jitumv_batching)
binary_jitumv_p.def_call(binary_jitumv_p_call)
binary_jitumv_p.def_tags('jit_uniform', 'binary')
binary_jitumv_p.def_benchmark_data(_binary_jitumv_benchmark_data)


def _jitumm_numba_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for binary event JIT-uniform matrix-matrix product.

    Parameters
    ----------
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input event matrix ``B``.
    corder : bool, optional
        If True, iterate over output rows in the outer loop. If False,
        iterate over ``B`` rows in the outer loop. Default is True.
    **kwargs
        Additional keyword arguments, must include ``outs`` specifying
        output shape/dtype information.

    Returns
    -------
    callable
        A function ``kernel(w_low, w_high, clen, B, seed)`` that
        executes the Numba-compiled kernel and returns the result.
    """
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_uniform01 = _rng['uniform01']

    stride = _MM_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)
    is_bool = np.dtype(B_info.dtype) in (np.dtype('bool'), np.dtype('int8'))

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, B, seed, posts):
            m = posts.shape[0]
            n = posts.shape[1]
            k = B.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            span = w_high0 - w_low0
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                out = np.zeros(n, dtype=posts.dtype)
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            u01 = _rng_uniform01(seed0, row, j)
                            w = w_low0 + u01 * span
                            for col in range(n):
                                if is_bool:
                                    active = B[j, col]
                                else:
                                    active = B[j, col] > 0.
                                if active:
                                    out[col] += w
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = out
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, B, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = B.shape[0]
            n = B.shape[1]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            span = w_high0 - w_low0
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= k:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > k:
                        chunk_end = k
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            j = chunk_start + local_j
                            u01 = _rng_uniform01(seed0, row, j)
                            w = w_low0 + u01 * span
                            for col in range(n):
                                if is_bool:
                                    active = B[row, col]
                                else:
                                    active = B[row, col] > 0.
                                if active:
                                    posts[j, col] += w
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(w_low, w_high, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, B, seed)

    return kernel


def _binary_jitumm_cuda_kernel(
    corder: bool = True,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_jitumm.cu'),
        name='jit_uniform_binary_jitumm',
    )
    wt_sfx = dtype_suffix(kwargs['w_low_info'].dtype)
    B_info = kwargs['B_info']
    variant = 'notrans' if corder else 'trans'
    kernel_name = f'jit_uniform_binary_jitumm.{variant}{wt_sfx}'

    out_info = kwargs['out_info']
    k_pack = int(B_info.shape[0])
    n = int(B_info.shape[1])
    n_words = (k_pack + 31) // 32
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)
    if corder:
        m_ffi, k_ffi = int(out_info.shape[0]), k_pack
    else:
        m_ffi, k_ffi = k_pack, int(out_info.shape[0])
    is_bool = np.dtype(B_info.dtype) in (np.dtype('bool'), np.dtype('int8'))

    def kernel(w_low, w_high, clen, B, seed):
        spikes = B.astype(jnp.int8) if is_bool else (B > 0).astype(jnp.int8)
        packed = jax.ffi.ffi_call(
            'jit_uniform_binary_jitumm.pack',
            jax.ShapeDtypeStruct((n, n_words), jnp.uint32),
        )(spikes, k=np.int32(k_pack), n=np.int32(n), n_words=np.int32(n_words))
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w_low, w_high, clen, seed, packed,
            m=np.int32(m_ffi), k=np.int32(k_ffi), n=np.int32(n),
            n_words=np.int32(n_words), chunk_size=np.int32(chunk_size),
        )

    return kernel


def _jitumm_jvp_wloc(w_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w_low`` argument of the binary JIT-uniform matrix-matrix product.

    Parameters
    ----------
    w_dot : jax.Array
        Tangent vector for the ``w_low`` argument.
    w_low, w_high, clen, B, seed : jax.Array
        Primal values of the primitive's arguments.
    shape : MatrixShape
        Shape of the connectivity matrix.
    transpose : bool
        Whether the transposed operation is used.
    corder : bool
        Memory layout order flag.
    **kwargs
        Additional keyword arguments including ``backend``.

    Returns
    -------
    list
        Single-element list containing the JVP result.
    """
    count_basis = binary_jitumm_p_call(
        w_dot, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder,
        backend=kwargs['backend'],
    )[0]
    high_basis = binary_jitumm_p_call(
        0., w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder,
        backend=kwargs['backend'],
    )[0]
    return [count_basis - high_basis]


def _jitumm_jvp_wscale(w_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w_high`` argument of the binary JIT-uniform matrix-matrix product.

    Parameters
    ----------
    w_dot : jax.Array
        Tangent vector for the ``w_high`` argument.
    w_low, w_high, clen, B, seed : jax.Array
        Primal values of the primitive's arguments.
    shape : MatrixShape
        Shape of the connectivity matrix.
    transpose : bool
        Whether the transposed operation is used.
    corder : bool
        Memory layout order flag.
    **kwargs
        Additional keyword arguments including ``backend``.

    Returns
    -------
    list
        Single-element list containing the JVP result.
    """
    return binary_jitumm_p_call(
        0., w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitumm_jvp_B(B_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``B`` argument of the binary JIT-uniform matrix-matrix product.

    Parameters
    ----------
    B_dot : jax.Array
        Tangent matrix for the ``B`` argument.
    w_low, w_high, clen, B, seed : jax.Array
        Primal values of the primitive's arguments.
    shape : MatrixShape
        Shape of the connectivity matrix.
    transpose : bool
        Whether the transposed operation is used.
    corder : bool
        Memory layout order flag.
    **kwargs
        Additional keyword arguments including ``backend``.

    Returns
    -------
    list
        Single-element list containing the JVP result.
    """
    return jitumm_p_call(
        w_low, w_high, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder,
        matrix_mode='mm', backend=kwargs['backend'],
    )


def _jitumm_transpose_rules(ct, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the binary JIT-uniform matrix-matrix product.

    Implements the VJP transpose for differentiation through the primitive.
    Supports transposing with respect to ``B``, ``w_low``, or ``w_high``.

    Parameters
    ----------
    ct : list
        Cotangent of the output.
    w_low, w_high, clen, B, seed : jax.Array or ad.UndefinedPrimal
        Primal values or undefined primals of the primitive's arguments.
    shape : MatrixShape
        Shape of the connectivity matrix.
    transpose : bool
        Whether the transposed operation was used in the forward pass.
    corder : bool
        Memory layout order flag.
    **kwargs
        Additional keyword arguments including ``backend``.

    Returns
    -------
    tuple
        Cotangents for each input argument (w_low, w_high, clen, B, seed).

    Raises
    ------
    NotImplementedError
        If the undefined primal is not ``B``, ``w_low``, or ``w_high``.

    Notes
    -----
    For the weight bounds, the transpose uses the same affine decomposition
    as in ``_jitumv_transpose_rules``:

        ``y = w_low * C + (w_high - w_low) * U``

    where ``U = y(0, 1)`` and ``C = y(1, 1)``.
    """
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    needs_w_low = ad.is_undefined_primal(w_low)
    needs_w_high = ad.is_undefined_primal(w_high)
    needs_B = ad.is_undefined_primal(B)
    dw_low = w_low
    dw_high = w_high
    dB = B

    if needs_B and (needs_w_low or needs_w_high):
        raise NotImplementedError(
            'Transpose rules for binary_jitumm do not support '
            'differentiating weight bounds and B in the same transpose.'
        )
    if needs_B:
        dB = jitumm_p_call(
            w_low,
            w_high,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            matrix_mode='mm',
            backend=kwargs['backend'],
        )[0]

    if needs_w_low or needs_w_high:
        ones = jnp.ones((1,), dtype=ct.dtype)
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        high_basis = binary_jitumm_p_call(
            zeros,
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=kwargs['backend'],
        )[0]
        if needs_w_low:
            count_basis = binary_jitumm_p_call(
                ones,
                ones,
                clen,
                B,
                seed,
                shape=shape,
                transpose=transpose,
                corder=corder,
                backend=kwargs['backend'],
            )[0]
            dw_low = jnp.expand_dims(jnp.sum(ct * (count_basis - high_basis)), axis=0)
        if needs_w_high:
            dw_high = jnp.expand_dims(jnp.sum(ct * high_basis), axis=0)

    if needs_B or needs_w_low or needs_w_high:
        return dw_low, dw_high, clen, dB, seed

    raise NotImplementedError(
        'Transpose rules for jitc_matmat_uniform not implemented for '
        'non-undefined primals.'
    )


def _batching_axis1(args, axis=1, **kwargs):
    """
    Helper for batching along axis 1 of the ``B`` matrix.

    Reshapes a 3-D batched ``B`` into a 2-D matrix, performs the matrix-matrix
    product, and reshapes the result back to 3-D.

    Parameters
    ----------
    args : tuple
        Batched arguments ``(w_low, w_high, clen, B, seed)``.
    axis : int, optional
        The output batch axis. Default is 1.
    **kwargs
        Additional keyword arguments including ``shape``, ``transpose``,
        ``corder``, and ``backend``.

    Returns
    -------
    tuple
        A pair ``(results, out_axes)`` where ``results`` is the batched output
        and ``out_axes`` indicates the batch dimension of each output.
    """
    assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[3].shape
    B = args[3].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_jitumm_p_call(
        args[0],
        args[1],
        args[2],
        B,
        args[4],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitumm_batching(args, axes, **kwargs):
    """
    Batching rule for the binary JIT-uniform matrix-matrix product primitive.

    Handles ``vmap`` over the ``B`` argument along different axes by
    reshaping and delegating to ``binary_jitumm_p_call``.

    Parameters
    ----------
    args : tuple
        Batched arguments ``(w_low, w_high, clen, B, seed)``.
    axes : tuple
        Batch axis for each argument (None means not batched).
    **kwargs
        Additional keyword arguments including ``shape``, ``transpose``,
        ``corder``, and ``backend``.

    Returns
    -------
    tuple
        A pair ``(results, out_axes)`` where ``results`` is the batched output
        and ``out_axes`` indicates the batch dimension of each output.
    """
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[3] = jnp.transpose(args[3], (1, 0, 2))
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, None, 1, None):
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, None, 2, None):
        return _batching_axis1(args, axis=2, **kwargs)

    else:
        return general_batching_rule(binary_jitumm_p, args, axes, **kwargs)


def _binary_jitumm_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the binary JIT-uniform matrix-matrix product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering different combinations
        of transpose, corder, and boolean/float event types.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                w_low = jnp.zeros(1, dtype=dtype)
                w_high = jnp.ones(1, dtype=dtype)
                clen = jnp.atleast_1d(jnp.asarray(np.ceil(2.0 / prob), dtype=jnp.int32))
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    B = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    B = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(BenchmarkConfig(name, (w_low, w_high, clen, B, seed), {
                    'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
                }))
    return configs


def binary_jitumm_p_call(
    w_low,
    w_high,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for the binary event JIT-uniform matrix-matrix product.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``binary_jitumm_p`` XLA custom kernel. This function expects
    pre-processed arguments (mantissa-only arrays, connection length instead of
    probability).

    Parameters
    ----------
    w_low : jax.Array
        Lower weight bound as a 1-D array of shape ``(1,)`` with floating dtype.
    w_high : jax.Array
        Upper weight bound as a 1-D array of shape ``(1,)`` with the same dtype
        as ``w_low``.
    clen : jax.Array
        Connection length parameter as a 1-D array of shape ``(1,)``, derived
        from the connection probability via ``ceil(2 / prob)``.
    B : jax.Array
        Input event matrix, a 2-D array of shape ``(n, k)`` (if
        ``transpose=False``) or ``(m, k)`` (if ``transpose=True``).
    seed : jax.Array
        Random seed as a 1-D array of shape ``(1,)``.
    shape : MatrixShape
        Shape ``(m, n)`` of the logical connectivity matrix.
    transpose : bool
        If True, compute ``A.T @ B``; otherwise compute ``A @ B``.
    corder : bool
        Memory layout order flag for the connectivity generation.
    backend : str, optional
        Computation backend (``'numba'`` or ``'pallas'``).

    Returns
    -------
    tuple
        A single-element tuple containing the result matrix of shape
        ``(m, k)`` or ``(n, k)`` depending on ``transpose``.

    Raises
    ------
    AssertionError
        If any input shape, dtype, or dimension constraint is violated.

    See Also
    --------
    binary_jitumm : High-level wrapper with unit handling and seed initialization.
    """
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert w_low.ndim == 1, "The weight should be a 1D array."
    assert w_high.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert w_low.shape == (1,), "The weight should be a scalar."
    assert w_high.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"
    assert jnp.issubdtype(w_low.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w_low.dtype == w_high.dtype, "w_low and w_high must have the same dtype."

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_low.dtype)
    )

    return binary_jitumm_p(
        w_low,
        w_high,
        clen,
        B,
        seed,
        outs=[out_info],
        w_low_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        w_high_info=jax.ShapeDtypeStruct(w_high.shape, w_high.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


binary_jitumm_p = XLACustomKernel(
    'binary_jitumm',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_jitumm``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) JIT uniform connectivity
matrix-matrix multiplication operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

In this operation, the connectivity matrix has weights uniformly distributed between specified
bounds, and the input matrix is treated as binary events (spikes). Each column of the input
is processed independently as an event vector.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_jitumm_p.available_backends(platform)``,
and the default backend can be configured with ``binary_jitumm_p.set_default(platform, backend)``.

See Also
--------
binary_jitumm : High-level user-facing function wrapper.
"""
)
binary_jitumm_p.def_numba_kernel(_jitumm_numba_kernel_generator)
binary_jitumm_p.def_cuda_raw_kernel(_binary_jitumm_cuda_kernel, asdefault=True)
binary_jitumm_p.def_jvp_rule2(_jitumm_jvp_wloc, _jitumm_jvp_wscale, None, _jitumm_jvp_B, None)
binary_jitumm_p.def_transpose_rule(_jitumm_transpose_rules)
binary_jitumm_p.def_batching_rule(_jitumm_batching)
binary_jitumm_p.def_call(binary_jitumm_p_call)
binary_jitumm_p.def_tags('jit_uniform', 'binary')
binary_jitumm_p.def_benchmark_data(_binary_jitumm_benchmark_data)
