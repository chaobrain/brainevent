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
from brainevent._misc import generate_block_dim, namescope
from brainevent._numba_random import get_numba_lfsr_seed, get_numba_lfsr_random_integers, get_numba_lfsr_uniform
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig, register_tvm_cuda_from_file, jaxinfo_to_warpinfo
from brainevent._pallas_random import get_pallas_lfsr_rng_class
from brainevent._typing import Data, MatrixShape
from .float import jitumv_p_call, jitumm_p_call, _dtype_sfx

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

    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_uniform = get_numba_lfsr_uniform()

    if corder:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, vector, seed, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_col in range(n_col):
                    state = _lfsr_seed(seed0 + i_col * n_row)
                    i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.asarray(0., dtype=posts.dtype)
                    while i_row < n_row:
                        w = _lfsr_uniform(state, w_low0, w_high0)
                        if vector[i_row]:
                            out += w
                        i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_col] = out
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, vector, seed, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_col in range(n_col):
                    state = _lfsr_seed(seed0 + i_col * n_row)
                    i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.asarray(0., dtype=posts.dtype)
                    while i_row < n_row:
                        w = _lfsr_uniform(state, w_low0, w_high0)
                        if vector[i_row] > 0.:
                            out += w
                        i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_col] = out


    else:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_row in range(num_row):
                    if vector[i_row]:
                        state = _lfsr_seed(seed0 + i_row * num_col)
                        i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while i_col < num_col:
                            w = _lfsr_uniform(state, w_low0, w_high0)
                            posts[i_col] += w
                            i_col += _lfsr_random_integers(state, 1, clen0 - 1)
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_row in range(num_row):
                    if vector[i_row] > 0.:
                        state = _lfsr_seed(seed0 + i_row * num_col)
                        i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while i_col < num_col:
                            w = _lfsr_uniform(state, w_low0, w_high0)
                            posts[i_col] += w
                            i_col += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(w_low, w_high, clen, vector, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, vector, seed)

    return kernel


def _jitumv_pallas_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Generate a Pallas Triton GPU kernel for binary event JIT-uniform matrix-vector product.

    Parameters
    ----------
    vector_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input event vector.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output array.
    corder : bool, optional
        If True, vectorize over output elements. If False, vectorize over
        input elements using atomic adds. Default is True.
    **kwargs
        Additional keyword arguments, must include ``outs`` specifying
        output shape/dtype information.

    Returns
    -------
    callable
        A function ``run(w_low, w_high, clen, vector, seed)`` that
        launches the Pallas kernel on GPU and returns the result.

    Notes
    -----
    Uses the globally configured LFSR RNG for random number generation
    within the Pallas kernel. The kernel is launched with a 1-D grid
    where each block processes ``block_size`` elements.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    block_size = generate_block_dim(dim, maximum=128)
    vector_is_bool = vector_info.dtype == jnp.bool_

    if corder:
        def kernel(w_low_ref, w_high_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_row = vector_ref.shape[0]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(_step, data):
                i_rows, i_row_mask, rng, res = data
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                v = vector_ref[safe_rows]
                v = jnp.where(i_row_mask, v, False if vector_is_bool else 0.)
                if not vector_is_bool:
                    v = v > 0.
                w = rng.uniform(w_low, w_high)
                res = jnp.where(v, res + w, res)
                i_rows += rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, res

            rng = _PallasLFSRRNG(seed + i_cols * num_row)
            i_rows = rng.random_integers(0, clen)
            i_row_mask = i_rows < num_row
            _, _, _, out = jax.lax.fori_loop(
                0, num_row, body,
                (i_rows, i_row_mask, rng, jnp.zeros(block_size, dtype=post_ref.dtype))
            )
            post_ref[i_cols] = jnp.where(i_col_mask, out, post_ref[i_cols])

    else:
        # Matches _jit_normal/binary.py corder=False:
        # vectorize over input rows, seed by i_rows, loop over output (i_cols).
        # Binary: only scatter w (via atomic_add) when vector element is event.
        def kernel(w_low_ref, w_high_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim
            safe_rows = jnp.where(i_row_mask, i_rows, 0)
            v = vector_ref[safe_rows]
            if vector_info.dtype != jnp.bool_:
                v = v > 0.
            # event_mask: only active lanes where the vector element is an event
            event_mask = i_row_mask & v

            def body(_step, data):
                i_cols, i_col_mask, rng = data
                w = rng.uniform(w_low, w_high)
                atomic_add(post_ref, (i_cols,), w, mask=event_mask & i_col_mask)
                i_cols += rng.random_integers(1, clen)
                return i_cols, i_cols < num_col, rng

            rng = _PallasLFSRRNG(seed + i_rows * num_col)
            i_cols = rng.random_integers(0, clen)
            i_col_mask = i_cols < num_col
            jax.lax.fori_loop(0, num_col, body, (i_cols, i_col_mask, rng))

    def run(w_low, w_high, clen, vector, seed):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(dim, block_size),),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_low, w_high, clen, vector, seed, placeholder)

    return run


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
    return binary_jitumv_p_call(
        w_dot, w_high, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


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
        w_low, w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
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
    if ad.is_undefined_primal(vector):
        r = jitumv_p_call(
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
        return w_low, w_high, clen, r, seed
    elif ad.is_undefined_primal(w_low):
        # With fixed connectivity/event masks for this primitive call:
        #   w_ij = w_low + (w_high - w_low) * u_ij
        # so the output is affine in (w_low, w_high):
        #   y = w_low * C + (w_high - w_low) * U
        # where:
        #   U = y(w_low=0, w_high=1)
        #   C = y(w_low=1, w_high=1)  (active connection counts)
        # For cotangent ct:
        #   dL/dw_high = <ct, U>
        #   dL/dw_low  = <ct, C - U>
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
        return dw_low, w_high, clen, vector, seed
    elif ad.is_undefined_primal(w_high):
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        ones = jnp.ones((1,), dtype=ct.dtype)
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
        dw_high = jnp.expand_dims(jnp.sum(ct * high_basis), axis=0)
        return w_low, dw_high, clen, vector, seed
    else:
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
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
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


_spike_sfx = {
    np.dtype('bool'): '_bool',
    np.dtype('int8'): '_bool',
    np.dtype('float32'): '_float',
    np.dtype('float16'): '_float',
    np.dtype('float64'): '_float',
    np.dtype('bfloat16'): '_float',
}


def _binary_jitumv_cuda_kernel(
    corder: bool,
    vector_info: jax.ShapeDtypeStruct,
    **kwargs
):
    register_tvm_cuda_from_file(
        module='binary_jitumv',
        source=Path(__file__).parent.joinpath('binary_jitumv.cu'),
    )
    wt_sfx = _dtype_sfx.get(np.dtype(kwargs['w_low_info'].dtype), '_f32')
    sp_sfx = _spike_sfx.get(np.dtype(vector_info.dtype), '_float')
    variant = 'gather' if corder else 'scatter'
    kernel_name = f'binary_jitumv.binary_jitumv_{variant}{wt_sfx}{sp_sfx}'

    def kernel(w_low, w_high, clen, vector, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(w_low, w_high, clen, seed, vector)

    return kernel


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
def _jitumv_warp_kernel_generator(
    w_low_info: jax.ShapeDtypeStruct,
    w_high_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Generate a Warp GPU kernel for binary event JIT-uniform matrix-vector product.

    Parameters
    ----------
    w_low_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the lower weight bound.
    w_high_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the upper weight bound.
    clen_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the connection length parameter.
    vector_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input event vector.
    seed_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the random seed.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output array.
    corder : bool, optional
        If True, each GPU thread handles one output element. If False, each
        thread handles one input element using atomic adds. Default is True.
    **kwargs
        Additional keyword arguments, must include ``outs`` specifying
        output shape/dtype information.

    Returns
    -------
    callable
        A function ``kernel(w_low, w_high, clen, vector, seed)`` that
        launches the Warp kernel on GPU and returns the result.
    """
    import warp
    from warp.jax_experimental import jax_kernel

    w_low_warp = jaxinfo_to_warpinfo(w_low_info)
    w_high_warp = jaxinfo_to_warpinfo(w_high_info)
    clen_warp = jaxinfo_to_warpinfo(clen_info)
    vector_warp = jaxinfo_to_warpinfo(vector_info)
    seed_warp = jaxinfo_to_warpinfo(seed_info)
    out_warp = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if corder:
        if vector_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: vector_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                num_row = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                r = float(0.0)
                state = warp.rand_init(seed0 + i_col * num_row)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    w = warp.randf(state) * w_diff + w_low0
                    r = warp.where(vector[i_row], r + w, r)
                    i_row += warp.randi(state, 1, clen0)
                posts[i_col] = r

        else:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: vector_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                num_row = vector.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                r = float(0.0)
                state = warp.rand_init(seed0 + i_col * num_row)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    w = warp.randf(state) * w_diff + w_low0
                    if vector[i_row] > float(0.0):
                        r += w
                    i_row += warp.randi(state, 1, clen0)
                posts[i_col] = r

    else:
        if vector_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: vector_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                num_col = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                v = vector[i_row]
                if v:
                    state = warp.rand_init(seed0 + i_row * num_col)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        w = warp.randf(state) * w_diff + w_low0
                        warp.atomic_add(posts, i_col, w)
                        i_col += warp.randi(state, 1, clen0)
        else:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                vector: vector_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                num_col = posts.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_row = warp.tid()
                v = vector[i_row] > 0.
                if v:
                    state = warp.rand_init(seed0 + i_row * num_col)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        w = warp.randf(state) * w_diff + w_low0
                        warp.atomic_add(posts, i_col, w)
                        i_col += warp.randi(state, 1, clen0)

    def kernel(w_low, w_high, clen, vector, seed):
        dim = out_info.shape[0] if corder else vector_info.shape[0]
        fn = jax_kernel(kernel_impl, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
        return fn(w_low, w_high, clen, vector, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _jitumm_warp_kernel_generator(
    w_low_info: jax.ShapeDtypeStruct,
    w_high_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Generate a Warp GPU kernel for binary event JIT-uniform matrix-matrix product.

    Parameters
    ----------
    w_low_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the lower weight bound.
    w_high_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the upper weight bound.
    clen_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the connection length parameter.
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input event matrix ``B``.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output matrix.
    seed_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the random seed.
    corder : bool, optional
        If True, each GPU thread handles one output row. If False, each
        thread handles one ``B`` row using atomic adds. Default is True.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    callable
        A function ``kernel(w_low, w_high, clen, B, seed)`` that
        launches the Warp kernel on GPU and returns the result.
    """
    import warp
    from warp.jax_experimental import jax_kernel

    w_low_warp = jaxinfo_to_warpinfo(w_low_info)
    w_high_warp = jaxinfo_to_warpinfo(w_high_info)
    clen_warp = jaxinfo_to_warpinfo(clen_info)
    B_warp = jaxinfo_to_warpinfo(B_info)
    seed_warp = jaxinfo_to_warpinfo(seed_info)
    out_warp = jaxinfo_to_warpinfo(out_info)

    if corder:
        # Each thread i_m generates one row of the JITC matrix and
        # multiplies it with B, accumulating into posts[i_m, :].
        if B_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                B: B_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                k = B.shape[0]
                n = B.shape[1]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m * k)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randf(state) * w_diff + w_low0
                    for j in range(n):
                        if B[i_k, j]:
                            posts[i_m, j] += w
                    i_k += warp.randi(state, 1, clen0)

        else:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                B: B_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                k = B.shape[0]
                n = B.shape[1]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m * k)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    w = warp.randf(state) * w_diff + w_low0
                    for j in range(n):
                        if B[i_k, j] > float(0.0):
                            posts[i_m, j] += w
                    i_k += warp.randi(state, 1, clen0)

    else:
        # Each thread i_k generates one column of the JITC matrix and
        # scatters B[i_k, :] scaled by weight into output rows via atomic adds.
        if B_info.dtype == jnp.bool_:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                B: B_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                m = posts.shape[0]
                n = B.shape[1]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k * m)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randf(state) * w_diff + w_low0
                    for j in range(n):
                        if B[i_k, j]:
                            warp.atomic_add(posts, i_m, j, w)
                    i_m += warp.randi(state, 1, clen0)

        else:
            @warp.kernel
            def kernel_impl(
                w_low: w_low_warp,
                w_high: w_high_warp,
                clen: clen_warp,
                B: B_warp,
                seed: seed_warp,
                posts: out_warp,
            ):
                m = posts.shape[0]
                n = B.shape[1]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                w_diff = w_high0 - w_low0
                clen0 = clen[0]
                seed0 = seed[0]
                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k * m)
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    w = warp.randf(state) * w_diff + w_low0
                    for j in range(n):
                        if B[i_k, j] > float(0.0):
                            warp.atomic_add(posts, i_m, j, w)
                    i_m += warp.randi(state, 1, clen0)

    def kernel(w_low, w_high, clen, B, seed):
        dim = out_info.shape[0] if corder else B_info.shape[0]
        fn = jax_kernel(kernel_impl, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
        return fn(w_low, w_high, clen, B, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel

binary_jitumv_p.def_numba_kernel(_jitumv_numba_kernel_generator)
binary_jitumv_p.def_warp_kernel(_jitumv_warp_kernel_generator)
binary_jitumv_p.def_pallas_kernel('gpu', _jitumv_pallas_kernel_generator)
binary_jitumv_p.def_tvmffi_kernel('gpu', _binary_jitumv_cuda_kernel)
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

    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_uniform = get_numba_lfsr_uniform()

    if corder:
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_m in range(m):
                    state = _lfsr_seed(seed0 + i_m * k)
                    i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n, dtype=posts.dtype)
                    while i_k < k:
                        w = _lfsr_uniform(state, w_low0, w_high0)
                        for j in range(B.shape[1]):
                            if B[i_k, j]:
                                out[j] += w
                        i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_m] = out
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_m in range(m):
                    state = _lfsr_seed(seed0 + i_m * k)
                    i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n, dtype=posts.dtype)
                    while i_k < k:
                        w = _lfsr_uniform(state, w_low0, w_high0)
                        for j in range(B.shape[1]):
                            if B[i_k, j] > 0.:
                                out[j] += w
                        i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_m] = out


    else:
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_k in range(k):
                    state = _lfsr_seed(seed0 + i_k * m)
                    indices = np.where(B[i_k])[0]
                    i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                    while i_m < m:
                        w = _lfsr_uniform(state, w_low0, w_high0)
                        posts[i_m, indices] += w
                        i_m += _lfsr_random_integers(state, 1, clen0 - 1)
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(w_low, w_high, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                w_low0 = w_low[0]
                w_high0 = w_high[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_k in range(k):
                    state = _lfsr_seed(seed0 + i_k * m)
                    indices = np.where(B[i_k] > 0.)[0]
                    i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                    while i_m < m:
                        w = _lfsr_uniform(state, w_low0, w_high0)
                        posts[i_m, indices] += w
                        i_m += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(w_low, w_high, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, B, seed)

    return kernel


def _jitumm_pallas_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Pallas GPU kernel for binary event matmat with uniform-distributed JITC matrix.

    Matches _jitc_mm_normal_pallas_kernel_generator in _jit_normal/binary.py:
    - Grid: (row_or_k_blocks, B_cols) --- each block processes one B column
    - corder=True:  vectorize over output rows, seed by i_rows, loop over k
    - corder=False: vectorize over k, seed by i_ks, loop over output rows

    Parameters
    ----------
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input event matrix ``B``.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output matrix.
    corder : bool, optional
        If True, vectorize over output rows. If False, vectorize over the
        shared dimension ``k`` using atomic adds. Default is True.
    **kwargs
        Additional keyword arguments, must include ``outs`` specifying
        output shape/dtype information.

    Returns
    -------
    callable
        A function ``run(w_low, w_high, clen, B, seed)`` that
        launches the Pallas kernel on GPU and returns the result.

    Notes
    -----
    Uses the globally configured LFSR RNG for random number generation
    within the Pallas kernel. The kernel is launched with a 2-D grid of
    ``(row_or_k_blocks, B_cols)``.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    B_cols = B_info.shape[1]

    if corder:
        # Match _jit_normal/binary.py corder=True:
        # Grid: (row_blocks, B_cols). Each block processes one B column.
        out_rows = out_info.shape[0]
        row_block = generate_block_dim(out_rows, maximum=128)
        grid = (pl.cdiv(out_rows, row_block), B_cols)

        def kernel(w_low_ref, w_high_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            k = B_ref.shape[0]
            w_low0 = w_low_ref[0]
            w_high0 = w_high_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            col_j = pl.program_id(1)

            i_rows = i_row_block * row_block + jnp.arange(row_block)
            i_row_mask = i_rows < out_rows
            safe_rows = jnp.where(i_row_mask, i_rows, 0)

            rng = _PallasLFSRRNG(seed0 + i_rows * k)
            i_cols = rng.random_integers(0, clen0)
            i_col_mask = i_cols < k

            out = jnp.zeros(row_block, dtype=post_ref.dtype)

            def body(_step, data):
                i_cols, i_col_mask, rng, out = data
                w = rng.uniform(w_low0, w_high0)
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                b_vals = B_ref[safe_cols, col_j]
                # Binary thresholding: treat b_vals as events
                if B_ref.dtype == jnp.bool_:
                    events = jnp.asarray(b_vals, dtype=out.dtype)
                else:
                    events = jnp.where(b_vals > 0., 1., 0.)
                out += jnp.where(i_col_mask & i_row_mask, w * events, 0.)
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < k, rng, out

            _, _, _, out = jax.lax.fori_loop(0, k, body, (i_cols, i_col_mask, rng, out))
            atomic_add(post_ref, (safe_rows, col_j), out, mask=i_row_mask)

    else:
        # Match _jit_normal/binary.py corder=False:
        # Grid: (k_blocks, B_cols). Each block processes one B column.
        k_dim = B_info.shape[0]
        k_block = generate_block_dim(k_dim, maximum=128)
        grid = (pl.cdiv(k_dim, k_block), B_cols)

        def kernel(w_low_ref, w_high_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            m = post_ref.shape[0]
            w_low0 = w_low_ref[0]
            w_high0 = w_high_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_k_block = pl.program_id(0)
            col_j = pl.program_id(1)

            i_ks = i_k_block * k_block + jnp.arange(k_block)
            i_k_mask = i_ks < k_dim
            safe_ks = jnp.where(i_k_mask, i_ks, 0)

            # Preload B values for this column and apply binary thresholding
            b_vals = B_ref[safe_ks, col_j]
            if B_ref.dtype == jnp.bool_:
                b_events = jnp.asarray(b_vals, dtype=post_ref.dtype)
            else:
                b_events = jnp.where(b_vals > 0., 1., 0.)
            b_events = jnp.where(i_k_mask, b_events, 0.)

            rng = _PallasLFSRRNG(seed0 + i_ks * m)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < m

            def body(_step, data):
                i_rows, i_row_mask, rng = data
                w = rng.uniform(w_low0, w_high0)
                vals = jnp.where(i_k_mask & i_row_mask, w * b_events, 0.)
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                atomic_add(post_ref, (safe_rows, col_j), vals,
                           mask=i_k_mask & i_row_mask)
                i_rows += rng.random_integers(1, clen0)
                return i_rows, i_rows < m, rng

            jax.lax.fori_loop(0, m, body, (i_rows, i_row_mask, rng))

    def run(w_low, w_high, clen, B, seed):
        fn = pl.pallas_call(
            kernel,
            grid=grid,
            input_output_aliases={5: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_low, w_high, clen, B, seed, placeholder)

    return run


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
    return binary_jitumm_p_call(
        w_dot, w_high, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


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
        w_low, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
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
        w_low, w_high, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
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
    if ad.is_undefined_primal(B):
        r = jitumm_p_call(
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
        return w_low, w_high, clen, r, seed
    elif ad.is_undefined_primal(w_low):
        # Same affine decomposition as in _jitumv_transpose_rules:
        # y = w_low * C + (w_high - w_low) * U.
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
        return dw_low, w_high, clen, B, seed
    elif ad.is_undefined_primal(w_high):
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        ones = jnp.ones((1,), dtype=ct.dtype)
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
        dw_high = jnp.expand_dims(jnp.sum(ct * high_basis), axis=0)
        return w_low, dw_high, clen, B, seed
    else:
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
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
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


def _binary_jitumm_cuda_kernel(
    corder: bool,
    B_info: jax.ShapeDtypeStruct,
    **kwargs
):
    register_tvm_cuda_from_file(
        module='binary_jitumm',
        source=Path(__file__).parent.joinpath('binary_jitumm.cu'),
    )
    wt_sfx = _dtype_sfx.get(np.dtype(kwargs['w_low_info'].dtype), '_f32')
    sp_sfx = _spike_sfx.get(np.dtype(B_info.dtype), '_float')
    variant = 'gather' if corder else 'scatter'
    kernel_name = f'binary_jitumm.binary_jitumm_{variant}{wt_sfx}{sp_sfx}'

    def kernel(w_low, w_high, clen, B, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(w_low, w_high, clen, seed, B)

    return kernel


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
binary_jitumm_p.def_warp_kernel(_jitumm_warp_kernel_generator)
binary_jitumm_p.def_pallas_kernel('gpu', _jitumm_pallas_kernel_generator)
binary_jitumm_p.def_tvmffi_kernel('gpu', _binary_jitumm_cuda_kernel)
binary_jitumm_p.def_jvp_rule2(_jitumm_jvp_wloc, _jitumm_jvp_wscale, None, _jitumm_jvp_B, None)
binary_jitumm_p.def_transpose_rule(_jitumm_transpose_rules)
binary_jitumm_p.def_batching_rule(_jitumm_batching)
binary_jitumm_p.def_call(binary_jitumm_p_call)
binary_jitumm_p.def_tags('jit_uniform', 'binary')
binary_jitumm_p.def_benchmark_data(_binary_jitumm_benchmark_data)
