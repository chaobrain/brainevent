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
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._data import _initialize_seed, _initialize_conn_length
from brainevent._misc import generate_block_dim, namescope
from brainevent._numba_random import get_numba_lfsr_seed, get_numba_lfsr_random_integers, get_numba_lfsr_uniform
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig
from brainevent._pallas_random import get_pallas_lfsr_rng_class
from brainevent._typing import Data, MatrixShape

__all__ = [
    "jitu",
    "jitu_p",
    "jitumv",
    "jitumv_p",
    "jitumm",
    "jitumm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitu(
    w_low: Data,
    w_high: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    """
    Materialize a JIT uniform connectivity matrix as a dense array.

    Generates a dense matrix where each entry is drawn from
    ``Uniform(w_low, w_high)`` at positions determined by the connection
    probability ``prob`` and random seed ``seed``. All other entries are zero.

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
        non-zero entries in the generated matrix.
    seed : int
        Random seed for reproducible connectivity and weight generation.
    shape : MatrixShape
        Shape ``(m, n)`` of the output matrix.
    transpose : bool, optional
        If True, generate the transposed matrix of shape ``(n, m)``.
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
        Dense matrix of shape ``(m, n)`` (or ``(n, m)`` if ``transpose=True``)
        with uniformly distributed weights at connected positions and zeros
        elsewhere. Carries physical units if ``w_low`` has units.

    See Also
    --------
    jitumv : Matrix-vector product without materializing the matrix.

    Notes
    -----
    Each entry ``A[i, j]`` of the generated matrix follows the model:

        ``A[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Uniform(w_low, w_high)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables. Equivalently:

    - ``A[i, j] ~ Uniform(w_low, w_high)`` with probability ``prob``
    - ``A[i, j] = 0`` with probability ``1 - prob``

    The expected value of each entry is:

        ``E[A[i, j]] = prob * (w_low + w_high) / 2``

    The connectivity pattern and uniform variates are determined by ``seed`` and
    ``prob``. Using the same ``seed`` always produces the same matrix.

    This function materializes the full dense matrix. For implicit (non-materialized)
    matrix-vector products, use :func:`jitumv` or :func:`jitumm` instead.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_uniform.float import jitu
        >>> dense = jitu(0.1, 0.5, 0.2, seed=42, shape=(4, 6))
        >>> dense.shape
        (4, 6)
    """
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    clen = _initialize_conn_length(prob)
    res = jitu_p_call(
        w_low,
        w_high,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd)


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitumv(
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
    Float matrix-vector product with a JIT uniform connectivity matrix.

    Computes the product of a just-in-time generated sparse matrix with
    uniformly distributed weights and a dense vector. Unlike the binary
    variant, this function uses the full floating-point values of the vector
    elements.

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
        Input dense vector. Length must match the appropriate matrix
        dimension (``n`` if ``transpose=False``, ``m`` if ``transpose=True``).
        Optionally with physical units.
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
    jitumm : Matrix-matrix variant.
    binary_jitumv : Event-driven (binary) variant.

    Notes
    -----
    The connectivity matrix ``A`` of shape ``(m, n)`` follows the model:

        ``A[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Uniform(w_low, w_high)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent, both determined by ``seed``.

    The float matrix-vector product computes:

        ``result[i] = sum_{j=0}^{n-1} A[i, j] * vector[j]``

    Unlike the binary variant (:func:`binary_jitumv`), this uses the full
    floating-point values of ``vector`` rather than treating them as binary events.

    When ``transpose=True``, the operation becomes ``result = A^T @ vector``:

        ``result[j] = sum_{i=0}^{m-1} A[i, j] * vector[i]``

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_uniform.float import jitumv
        >>> vec = jnp.ones(5)
        >>> result = jitumv(0.1, 0.5, 0.2, vec, seed=42, shape=(3, 5))
        >>> result.shape
        (3,)
    """
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    seed = _initialize_seed(seed)
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = jitumv_p_call(
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
def jitumm(
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
    Float matrix-matrix product with a JIT uniform connectivity matrix.

    Computes the product of a just-in-time generated sparse matrix with
    uniformly distributed weights and a dense matrix ``B``. Unlike the binary
    variant, this function uses the full floating-point values of ``B``.

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
        Input dense matrix of shape ``(n, k)`` (if ``transpose=False``) or
        ``(m, k)`` (if ``transpose=True``). Optionally with physical units.
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
    jitumv : Matrix-vector variant.
    binary_jitumm : Event-driven (binary) variant.

    Notes
    -----
    The connectivity matrix ``A`` of shape ``(m, n)`` follows the model:

        ``A[i, j] = U[i, j] * B_conn[i, j]``

    where ``U[i, j] ~ Uniform(w_low, w_high)`` and ``B_conn[i, j] ~ Bernoulli(prob)``
    are independent, both determined by ``seed``.

    The float matrix-matrix product computes:

        ``result[i, j] = sum_{k=0}^{n-1} A[i, k] * B[k, j]``

    Unlike the binary variant (:func:`binary_jitumm`), this uses the full
    floating-point values of ``B`` rather than treating them as binary events.

    When ``transpose=True``, the operation becomes ``result = A^T @ B``:

        ``result[j, l] = sum_{i=0}^{m-1} A[i, j] * B[i, l]``

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_uniform.float import jitumm
        >>> B = jnp.ones((5, 3))
        >>> result = jitumm(0.1, 0.5, 0.2, B, seed=42, shape=(4, 5))
        >>> result.shape
        (4, 3)
    """
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    seed = _initialize_seed(seed)
    w_low, unitd = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unitd).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = jitumm_p_call(
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


def _jitu_numba_kernel_generator(
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for materializing a JIT uniform connectivity matrix.

    Parameters
    ----------
    corder : bool, optional
        If True, iterate over rows in the outer loop. If False, iterate
        over columns in the outer loop. Default is True.
    **kwargs
        Additional keyword arguments, must include ``outs`` specifying
        output shape/dtype information.

    Returns
    -------
    callable
        A function ``kernel(w_low, w_high, clen, seed)`` that executes
        the Numba-compiled kernel and returns the dense matrix.
    """
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_uniform = get_numba_lfsr_uniform()

    if corder:
        # JIT matrix.T
        # - JIT matrix shape = [m, n]
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[1]
            n = posts.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_row in range(n):
                state = _lfsr_seed(seed0 + i_row * m)
                i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_col < m:
                    posts[i_row, i_col] = _lfsr_uniform(state, w_low0, w_high0)
                    i_col += _lfsr_random_integers(state, 1, clen0 - 1)


    else:
        # JIT matrix.T
        # - JIT matrix shape = [m, n]
        @numba.njit(fastmath=True)
        def kernel_impl(w_low, w_high, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[1]
            n = posts.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_col in range(m):
                state = _lfsr_seed(seed0 + i_col * n)
                i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_row < n:
                    posts[i_row, i_col] = _lfsr_uniform(state, w_low0, w_high0)
                    i_row += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(w_low, w_high, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w_low, w_high, clen, seed)

    return kernel


def _jitu_pallas_kernel_generator(
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Generate a Pallas Triton GPU kernel for materializing a JIT uniform connectivity matrix.

    Parameters
    ----------
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output matrix.
    corder : bool, optional
        If True, vectorize over rows. If False, vectorize over columns.
        Default is True.
    **kwargs
        Additional keyword arguments, must include ``outs`` specifying
        output shape/dtype information.

    Returns
    -------
    callable
        A function ``run(w_low, w_high, clen, seed)`` that launches the
        Pallas kernel on GPU and returns the dense matrix.

    Notes
    -----
    Uses the globally configured LFSR RNG for random number
    generation within the Pallas kernel. The kernel is launched with a 1-D
    grid where each block processes ``block_size`` rows or columns.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    dim = out_info.shape[0] if corder else out_info.shape[1]
    block_size = generate_block_dim(dim, maximum=128)

    if corder:
        def kernel(w_low_ref, w_high_ref, clen_ref, seed_ref, _, post_ref):
            m = post_ref.shape[1]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim
            safe_rows = jnp.where(i_row_mask, i_rows, 0)

            def body(data):
                i_cols, i_col_mask, rng = data
                val = rng.uniform(w_low, w_high)
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                atomic_add(post_ref, (safe_rows, safe_cols), val, mask=i_row_mask & i_col_mask)
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < m, rng

            rng = _PallasLFSRRNG(seed0 + i_rows * m)
            i_cols = rng.random_integers(0, clen0)
            i_col_mask = i_cols < m
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng)
            )

    else:
        def kernel(w_low_ref, w_high_ref, clen_ref, seed_ref, _, post_ref):
            n = post_ref.shape[0]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim
            safe_cols = jnp.where(i_col_mask, i_cols, 0)

            def body(data):
                i_rows, i_row_mask, rng = data
                val = rng.uniform(w_low, w_high)
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                atomic_add(post_ref, (safe_rows, safe_cols), val, mask=i_row_mask & i_col_mask)
                i_rows = i_rows + rng.random_integers(1, clen0)
                return i_rows, i_rows < n, rng

            rng = _PallasLFSRRNG(seed0 + i_cols * n)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < n
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng)
            )

    def run(w_low, w_high, clen, seed):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(dim, block_size),),
            input_output_aliases={4: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_low, w_high, clen, seed, placeholder)

    return run


def _jitu_jvp_wlow(w_low_dot, w_low, w_high, clen, seed, *, shape, transpose: bool, corder: bool, **kwargs):
    """
    JVP rule for the ``w_low`` argument of the JIT-uniform dense matrix generation.

    Parameters
    ----------
    w_low_dot : jax.Array
        Tangent vector for the ``w_low`` argument.
    w_low, w_high, clen, seed : jax.Array
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

    Notes
    -----
    The derivative with respect to ``w_low`` is ``w_low_dot - jitu(0, w_low_dot)``,
    reflecting the affine structure ``A = w_low + (w_high - w_low) * U``.
    """
    res = jitu_p_call(
        0., w_low_dot, clen, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )[0]
    return [w_low_dot - res]


def _jitu_jvp_whigh(w_high_dot, w_low, w_high, clen, seed, *, shape, transpose: bool, corder: bool, **kwargs):
    """
    JVP rule for the ``w_high`` argument of the JIT-uniform dense matrix generation.

    Parameters
    ----------
    w_high_dot : jax.Array
        Tangent vector for the ``w_high`` argument.
    w_low, w_high, clen, seed : jax.Array
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
    return jitu_p_call(
        0., w_high_dot, clen, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _wlow_tranpose(ct, seed, clen, **kwargs):
    """
    Compute the transpose contribution from ``w_low`` for the dense matrix primitive.

    Parameters
    ----------
    ct : jax.Array
        Cotangent array.
    seed : jax.Array
        Random seed.
    clen : jax.Array
        Connection length parameter.
    **kwargs
        Keyword arguments passed to ``jitu_p_call`` including ``shape``,
        ``transpose``, ``corder``, and ``backend``.

    Returns
    -------
    jax.Array
        Scalar cotangent for ``w_low``, computed as ``sum(ct * (1 - U))``
        where ``U = jitu(0, 1)``.

    Notes
    -----
    Uses the affine decomposition ``A = (1 - U) * w_low + U * w_high``
    where ``U = jitu(0, 1)`` represents the uniform random fractions.
    """
    # JITC * (high - low) + low
    forward = jitu_p_call(0., 1., clen, seed, **kwargs)[0]
    return jnp.expand_dims((ct * (-forward + 1.)).sum(), axis=0)


def _whigh_tranpose(ct, seed, clen, **kwargs):
    """
    Compute the transpose contribution from ``w_high`` for the dense matrix primitive.

    Parameters
    ----------
    ct : jax.Array
        Cotangent array.
    seed : jax.Array
        Random seed.
    clen : jax.Array
        Connection length parameter.
    **kwargs
        Keyword arguments passed to ``jitu_p_call`` including ``shape``,
        ``transpose``, ``corder``, and ``backend``.

    Returns
    -------
    jax.Array
        Scalar cotangent for ``w_high``, computed as ``sum(ct * U)``
        where ``U = jitu(0, 1)``.

    Notes
    -----
    Uses the affine decomposition ``A = (1 - U) * w_low + U * w_high``
    where ``U = jitu(0, 1)`` represents the uniform random fractions.
    """
    # JITC * (high - low) + low
    forward = jitu_p_call(0., 1., clen, seed, **kwargs)[0]
    return jnp.expand_dims((ct * forward).sum(), axis=0)


def _jitu_transpose(ct, w_low, w_high, clen, seed, *, shape, transpose: bool, corder: bool, **kwargs):
    """
    Transpose (adjoint) rule for the JIT-uniform dense matrix generation.

    Parameters
    ----------
    ct : list
        Cotangent of the output.
    w_low, w_high, clen, seed : jax.Array or ad.UndefinedPrimal
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
        Cotangents for each input argument ``(w_low, w_high, clen, seed)``.

    Raises
    ------
    NotImplementedError
        If the undefined primal is not ``w_low`` or ``w_high``.
    """
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(w_low):
        dwlow = _wlow_tranpose(
            ct,
            seed,
            clen,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=kwargs['backend'],
        )
        return (dwlow, w_high, clen, seed)
    elif ad.is_undefined_primal(w_high):
        dwhigh = _whigh_tranpose(
            ct,
            seed,
            clen,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=kwargs['backend'],
        )
        return (w_low, dwhigh, clen, seed)
    else:
        raise NotImplementedError(
            'JITC matrix transpose is only implemented for the w_low and w_high arguments.'
        )


def _jitu_batching(args, axes, **kwargs):
    """
    Batching rule for the JIT-uniform dense matrix generation primitive.

    Parameters
    ----------
    args : tuple
        Batched arguments ``(w_low, w_high, clen, seed)``.
    axes : tuple
        Batch axis for each argument (None means not batched).
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    tuple
        A pair ``(results, out_axes)`` where ``results`` is the batched output
        and ``out_axes`` indicates the batch dimension of each output.
    """
    return general_batching_rule(jitu_p, args, axes, **kwargs)


def _jitu_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the JIT-uniform dense matrix generation.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering different combinations
        of transpose and corder.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_low = jnp.zeros(1, dtype=dtype)
            w_high = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (w_low, w_high, clen, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jitu_p_call(
    w_low,
    w_high,
    clen,
    seed,
    *,
    shape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for materializing a JIT uniform connectivity matrix.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``jitu_p`` XLA custom kernel. This function expects pre-processed
    arguments (mantissa-only arrays, connection length instead of probability).

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
    seed : jax.Array
        Random seed as a 1-D array of shape ``(1,)``.
    shape : MatrixShape
        Shape ``(m, n)`` of the output matrix.
    transpose : bool
        If True, the output shape is reversed to ``(n, m)``.
    corder : bool
        Memory layout order flag for the connectivity generation.
    backend : str, optional
        Computation backend (``'numba'`` or ``'pallas'``).

    Returns
    -------
    tuple
        A single-element tuple containing the dense matrix.

    Raises
    ------
    AssertionError
        If any input shape or dtype constraint is violated.

    See Also
    --------
    jitu : High-level wrapper with unit handling.
    """
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    assert jnp.issubdtype(w_low.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w_low.dtype == w_high.dtype, "w_low and w_high must have the same dtype."

    out_info = (
        jax.ShapeDtypeStruct(shape[::-1], dtype=w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct(shape, dtype=w_low.dtype)
    )

    return jitu_p(
        w_low,
        w_high,
        clen,
        seed,
        outs=[out_info],
        w_low_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        w_high_info=jax.ShapeDtypeStruct(w_high.shape, w_high.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


jitu_p = XLACustomKernel(
    'float_jitu',
    doc="""
Low-level XLA custom-kernel primitive for ``jitu``.

This ``XLACustomKernel`` instance dispatches the JIT uniform connectivity matrix generation
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

This operation generates a sparse connectivity matrix where weights are uniformly distributed
between specified lower and upper bounds. The connectivity pattern is generated on-the-fly
using a deterministic PRNG seeded by the provided seed value.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``jitu_p.available_backends(platform)``,
and the default backend can be configured with ``jitu_p.set_default(platform, backend)``.

See Also
--------
jitu : High-level user-facing function wrapper.
"""
)
jitu_p.def_numba_kernel(_jitu_numba_kernel_generator)
jitu_p.def_pallas_kernel('gpu', _jitu_pallas_kernel_generator)
jitu_p.def_jvp_rule2(_jitu_jvp_wlow, _jitu_jvp_whigh, None, None)
jitu_p.def_transpose_rule(_jitu_transpose)
jitu_p.def_batching_rule(_jitu_batching)
jitu_p.def_call(jitu_p_call)
jitu_p.def_tags('jit_uniform', 'float')
jitu_p.def_benchmark_data(_jitu_benchmark_data)


# Kernel generators for JIT connection SPMV

def _jitumv_numba_kernel_generator(
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for float JIT-uniform matrix-vector product.

    Parameters
    ----------
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
                out = np.asarray(0., dtype=vector.dtype)
                while i_row < n_row:
                    out += vector[i_row] * _lfsr_uniform(state, w_low0, w_high0)
                    i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                posts[i_col] = out


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
                state = _lfsr_seed(seed0 + i_row * num_col)
                v = vector[i_row]
                i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_col < num_col:
                    posts[i_col] += v * _lfsr_uniform(state, w_low0, w_high0)
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
    Generate a Pallas Triton GPU kernel for float JIT-uniform matrix-vector product.

    Parameters
    ----------
    vector_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input vector.
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
    Uses the globally configured LFSR RNG for random number
    generation within the Pallas kernel. The kernel is launched with a 1-D
    grid where each block processes ``block_size`` elements.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    block_size = generate_block_dim(dim, maximum=128)

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

            def body(data):
                i_rows, i_row_mask, rng, out = data
                v = jnp.where(i_row_mask, vector_ref[i_rows], 0.)
                out += v * rng.uniform(w_low, w_high)
                i_rows += rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, out

            rng = _PallasLFSRRNG(seed + i_cols * num_row)
            i_rows = rng.random_integers(0, clen)
            i_row_mask = i_rows < num_row
            out = jnp.zeros(block_size, dtype=post_ref.dtype)
            out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng, out)
            )[-1]
            post_ref[i_cols] = jnp.where(i_col_mask, out, post_ref[i_cols])

    else:
        def kernel(w_low_ref, w_high_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            w_low = w_low_ref[0]
            w_high = w_high_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim
            vector = jnp.where(i_row_mask, vector_ref[i_rows], 0.)

            def body(data):
                i_cols, i_col_mask, rng = data
                atomic_add(post_ref, (i_cols,), vector * rng.uniform(w_low, w_high), mask=i_row_mask & i_col_mask)
                i_cols += rng.random_integers(1, clen)
                return i_cols, i_cols < num_col, rng

            rng = _PallasLFSRRNG(seed + i_rows * num_col)
            i_cols = rng.random_integers(0, clen)
            i_col_mask = i_cols < num_col
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng)
            )

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
    JVP rule for the vector argument of the float JIT-uniform matrix-vector product.

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


def _jitumv_jvp_wlow(w_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w_low`` argument of the float JIT-uniform matrix-vector product.

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
    return jitumv_p_call(
        w_dot, w_high, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitumv_jvp_whigh(w_dot, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w_high`` argument of the float JIT-uniform matrix-vector product.

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
    return jitumv_p_call(
        w_low, w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitumv_transpose_rules(ct, w_low, w_high, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the float JIT-uniform matrix-vector product.

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
    For the weight bounds, the transpose uses an affine decomposition:

        ``y = w_low * C(v) + (w_high - w_low) * U(v)``

    where ``U(v) = y(0, 1)`` and ``C(v) = y(1, 1)``. The cotangents are:

    - ``dL/dw_high = <ct, U(v)>``
    - ``dL/dw_low  = <ct, C(v) - U(v)>``
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
        # Fix the sampled connectivity and RNG stream (same `clen/seed/shape/transpose/corder`).
        # For each active entry:
        #   w_ij = w_low + (w_high - w_low) * u_ij,  u_ij in [0, 1).
        # The linear map output is therefore affine in (w_low, w_high):
        #   y = w_low * C(v) + (w_high - w_low) * U(v),
        # where
        #   U(v) = y(0, 1)  and  C(v) = y(1, 1).
        # Given cotangent ct, with inner product <a, b> = sum(a * b):
        #   dL/dw_high = <ct, U(v)>
        #   dL/dw_low  = <ct, C(v) - U(v)>.
        ones = jnp.ones((1,), dtype=ct.dtype)
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        u_basis = jitumv_p_call(
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
        c_basis = jitumv_p_call(
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
        dw_low = jnp.expand_dims(jnp.sum(ct * (c_basis - u_basis)), axis=0)
        return dw_low, w_high, clen, vector, seed
    elif ad.is_undefined_primal(w_high):
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        ones = jnp.ones((1,), dtype=ct.dtype)
        u_basis = jitumv_p_call(
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
        dw_high = jnp.expand_dims(jnp.sum(ct * u_basis), axis=0)
        return w_low, dw_high, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitumv_batching(args, axes, **kwargs):
    """
    Batching rule for the float JIT-uniform matrix-vector product primitive.

    Handles ``vmap`` over the vector argument by promoting the operation to
    a matrix-matrix product (``jitumm_p_call``).

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
        r = jitumm_p_call(
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
        r = jitumm_p_call(
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
        return general_batching_rule(jitumv_p, args, axes, **kwargs)


def _jitumv_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the float JIT-uniform matrix-vector product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering different combinations
        of transpose and corder.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_low = jnp.zeros(1, dtype=dtype)
            w_high = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w_low, w_high, clen, vector, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                )
            )
    return configs


def jitumv_p_call(
    w_low,
    w_high,
    clen,
    vector,
    seed,
    *,
    shape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for the float JIT-uniform matrix-vector product.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``jitumv_p`` XLA custom kernel. This function expects pre-processed
    arguments (mantissa-only arrays, connection length instead of probability).

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
        Input dense vector, a 1-D array. Length must match ``shape[1]``
        (if ``transpose=False``) or ``shape[0]`` (if ``transpose=True``).
    seed : jax.Array
        Random seed as a 1-D array of shape ``(1,)``.
    shape : MatrixShape
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
    jitumv : High-level wrapper with unit handling and seed initialization.
    """
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert w_low.shape == (1,), f"The weight shape should be (1,), but got {w_low.shape}."
    assert w_high.shape == (1,), f"The weight shape should be (1,), but got {w_high.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."
    assert jnp.issubdtype(w_low.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w_low.dtype == w_high.dtype, "w_low and w_high must have the same dtype."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_low.dtype)
    )

    return jitumv_p(
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


jitumv_p = XLACustomKernel(
    'float_jitumv',
    doc="""
Low-level XLA custom-kernel primitive for ``jitumv``.

This ``XLACustomKernel`` instance dispatches the JIT uniform connectivity matrix-vector
multiplication with floating-point weights operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided by
the high-level wrapper.

In this operation, the connectivity matrix has weights uniformly distributed between
specified bounds, and the input vector contains floating-point values. The operation
computes a standard matrix-vector product without event-driven sparsity.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``jitumv_p.available_backends(platform)``,
and the default backend can be configured with ``jitumv_p.set_default(platform, backend)``.

See Also
--------
jitumv : High-level user-facing function wrapper.
"""
)
jitumv_p.def_numba_kernel(_jitumv_numba_kernel_generator)
jitumv_p.def_pallas_kernel('gpu', _jitumv_pallas_kernel_generator)
jitumv_p.def_jvp_rule2(_jitumv_jvp_wlow, _jitumv_jvp_whigh, None, _jitumv_jvp_v, None)
jitumv_p.def_transpose_rule(_jitumv_transpose_rules)
jitumv_p.def_batching_rule(_jitumv_batching)
jitumv_p.def_call(jitumv_p_call)
jitumv_p.def_tags('jit_uniform', 'float')
jitumv_p.def_benchmark_data(_jitumv_benchmark_data)


def _jitumm_numba_kernel_generator(
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for float JIT-uniform matrix-matrix product.

    Parameters
    ----------
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
        # JIT Matrix.T @ B
        # - JIT matrix: [k, m]
        # - B: [k, n]
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
                out = np.zeros(n, dtype=B.dtype)
                while i_k < k:
                    out += B[i_k] * _lfsr_uniform(state, w_low0, w_high0)
                    i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                posts[i_m] = out


    else:
        # JIT Matrix.T @ B
        # - JIT matrix: [k, m]
        # - B: [k, n]
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
                out = B[i_k]
                i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_m < m:
                    posts[i_m] += out * _lfsr_uniform(state, w_low0, w_high0)
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
    Generate a Pallas Triton GPU kernel for float JIT-uniform matrix-matrix product.

    Parameters
    ----------
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input matrix ``B``.
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
        A function ``run(w_low, w_high, clen, B, seed)`` that launches
        the Pallas kernel on GPU and returns the result.

    Notes
    -----
    The grid is 2-D: ``(row_or_k_blocks, B_cols)``. Each kernel block
    processes one column of ``B`` using vector RNG identical to the
    ``_jit_normal`` implementation.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    B_cols = B_info.shape[1]

    if corder:
        # Match _jit_normal float.py _jitnmm_pallas corder=True:
        # Grid: (row_blocks, B_cols). Each kernel block processes one B column,
        # using vector RNG identical to jitn. Accumulates into 1D local array.
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
            col_j = pl.program_id(1)  # scalar B column index

            # Row indices --- VECTOR
            i_rows = i_row_block * row_block + jnp.arange(row_block)
            i_row_mask = i_rows < out_rows
            safe_rows = jnp.where(i_row_mask, i_rows, 0)

            rng = _PallasLFSRRNG(seed0 + i_rows * k)
            i_cols = rng.random_integers(0, clen0)  # [row_block]
            i_col_mask = i_cols < k

            out = jnp.zeros(row_block, dtype=post_ref.dtype)

            def body(data):
                i_cols, i_col_mask, rng, out = data
                w = rng.uniform(w_low0, w_high0)  # [row_block]
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                b_vals = B_ref[safe_cols, col_j]  # [row_block] vector gather
                out += jnp.where(i_col_mask & i_row_mask, w * b_vals, 0.)
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < k, rng, out

            _, _, _, out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng, out)
            )
            atomic_add(post_ref, (safe_rows, col_j), out, mask=i_row_mask)

    else:
        # Match _jit_normal float.py _jitnmm_pallas corder=False:
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
            col_j = pl.program_id(1)  # scalar B column index

            i_ks = i_k_block * k_block + jnp.arange(k_block)
            i_k_mask = i_ks < k_dim
            safe_ks = jnp.where(i_k_mask, i_ks, 0)

            # Preload B values for this column (1D vector gather)
            b_vals = B_ref[safe_ks, col_j]  # [k_block]

            rng = _PallasLFSRRNG(seed0 + i_ks * m)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < m

            def body(data):
                i_rows, i_row_mask, rng = data
                w = rng.uniform(w_low0, w_high0)  # [k_block]
                vals = jnp.where(i_k_mask & i_row_mask, w * b_vals, 0.)
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                atomic_add(post_ref, (safe_rows, col_j), vals,
                           mask=i_k_mask & i_row_mask)
                i_rows += rng.random_integers(1, clen0)
                return i_rows, i_rows < m, rng

            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng)
            )

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


def _jitumm_jvp_wlow(w_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w_low`` argument of the float JIT-uniform matrix-matrix product.

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
    return jitumm_p_call(
        w_dot, w_high, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitumm_jvp_whigh(w_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w_high`` argument of the float JIT-uniform matrix-matrix product.

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
    return jitumm_p_call(
        w_low, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitumm_jvp_B(B_dot, w_low, w_high, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``B`` argument of the float JIT-uniform matrix-matrix product.

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
    Transpose (adjoint) rule for the float JIT-uniform matrix-matrix product.

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
    For the weight bounds, the transpose uses the affine decomposition:

        ``Y = w_low * C(B) + (w_high - w_low) * U(B)``

    where ``U(B) = Y(0, 1)`` and ``C(B) = Y(1, 1)``. The cotangents are:

    - ``dL/dw_high = <ct, U(B)>``
    - ``dL/dw_low  = <ct, C(B) - U(B)>``
    """
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        dB = jitumm_p_call(
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
        return w_low, w_high, clen, dB, seed
    elif ad.is_undefined_primal(w_low):
        # Same affine decomposition as _jitumv_transpose_rules, now for matrix right operand B:
        #   Y = w_low * C(B) + (w_high - w_low) * U(B),
        #   U(B) = Y(0, 1), C(B) = Y(1, 1).
        # Hence:
        #   dL/dw_high = <ct, U(B)>
        #   dL/dw_low  = <ct, C(B) - U(B)>.
        ones = jnp.ones((1,), dtype=ct.dtype)
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        u_basis = jitumm_p_call(
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
        c_basis = jitumm_p_call(
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
        dw_low = jnp.expand_dims(jnp.sum(ct * (c_basis - u_basis)), axis=0)
        return dw_low, w_high, clen, B, seed
    elif ad.is_undefined_primal(w_high):
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        ones = jnp.ones((1,), dtype=ct.dtype)
        u_basis = jitumm_p_call(
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
        dw_high = jnp.expand_dims(jnp.sum(ct * u_basis), axis=0)
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
    r = jitumm_p_call(
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
    Batching rule for the float JIT-uniform matrix-matrix product primitive.

    Handles ``vmap`` over the ``B`` argument along different axes by
    reshaping and delegating to ``jitumm_p_call``.

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
        return general_batching_rule(jitumm_p, args, axes, **kwargs)


def _jitumm_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the float JIT-uniform matrix-matrix product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering different combinations
        of transpose and corder.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_low = jnp.zeros(1, dtype=dtype)
            w_high = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w_low, w_high, clen, B, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                )
            )
    return configs


def jitumm_p_call(
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
    Low-level primitive call for the float JIT-uniform matrix-matrix product.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``jitumm_p`` XLA custom kernel. This function expects pre-processed
    arguments (mantissa-only arrays, connection length instead of probability).

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
        Input dense matrix, a 2-D array of shape ``(n, k)`` (if
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
    jitumm : High-level wrapper with unit handling and seed initialization.
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

    return jitumm_p(
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
        TITLE_SIZE=B.shape[1],
        backend=backend,
    )


jitumm_p = XLACustomKernel(
    'float_jitumm',
    doc="""
Low-level XLA custom-kernel primitive for ``jitumm``.

This ``XLACustomKernel`` instance dispatches the JIT uniform connectivity matrix-matrix
multiplication with floating-point weights operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided by
the high-level wrapper.

In this operation, the connectivity matrix has weights uniformly distributed between
specified bounds, and the input matrix contains floating-point values. Each column of
the input is processed independently in a standard matrix-matrix product.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``jitumm_p.available_backends(platform)``,
and the default backend can be configured with ``jitumm_p.set_default(platform, backend)``.

See Also
--------
jitumm : High-level user-facing function wrapper.
"""
)
jitumm_p.def_numba_kernel(_jitumm_numba_kernel_generator)
jitumm_p.def_pallas_kernel('gpu', _jitumm_pallas_kernel_generator)
jitumm_p.def_jvp_rule2(_jitumm_jvp_wlow, _jitumm_jvp_whigh, None, _jitumm_jvp_B, None)
jitumm_p.def_transpose_rule(_jitumm_transpose_rules)
jitumm_p.def_batching_rule(_jitumm_batching)
jitumm_p.def_call(jitumm_p_call)
jitumm_p.def_tags('jit_uniform', 'float')
jitumm_p.def_benchmark_data(_jitumm_benchmark_data)
