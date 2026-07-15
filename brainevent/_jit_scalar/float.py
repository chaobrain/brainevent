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
from typing import Literal, Optional
import warnings

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._data import _initialize_seed, _initialize_conn_length
from brainevent._misc import namescope
from brainevent._numba_random import get_numba_lfsr_seed, get_numba_lfsr_random_integers, get_numba_lfsr_uniform
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig
from brainevent._op import load_cuda_file
from brainevent._typing import Data, MatrixShape

MatrixMode = Literal['mv', 'mm']

__all__ = [
    "jits",
    "jits_p",
    "jitsmv",
    "jitsmv_p",
    "jitsmm",
    "jitsmm_p",
]


def _normalize_matrix_mode(matrix_mode: str) -> MatrixMode:
    if matrix_mode not in ('mv', 'mm'):
        raise ValueError(f"matrix_mode must be 'mv' or 'mm', got {matrix_mode!r}.")
    return matrix_mode


def _normalize_chunk_size(n_cols: int, chunk_size: Optional[int], target_chunks: int) -> int:
    if chunk_size is None:
        target_chunks = int(target_chunks)
        if target_chunks <= 0:
            raise ValueError("target_chunks must be positive")
        chunk_size = max(1, (int(n_cols) + target_chunks - 1) // target_chunks)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return chunk_size


def _warn_corder_ignored(corder: bool) -> None:
    warnings.warn(
        "corder is ignored by the light JIT scalar implementation.",
        UserWarning,
        stacklevel=3,
    )


def _light_options(kwargs):
    return {
        'chunk_size': kwargs.get('chunk_size', None),
        'target_chunks': kwargs.get('target_chunks', 4),
    }


@namescope(name="brainevent.jits", static_argnames=("shape", "transpose", "corder", "matrix_mode", "chunk_size", "target_chunks"))
def _jits_impl(
    w0: Data,
    w1: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    matrix_mode: MatrixMode = 'mv',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    """
    Materialize a JIT scalar connectivity matrix as a dense array.

    Generates a dense matrix where each entry is drawn from
    ``Scalar(w0, w1)`` at positions determined by the connection
    probability ``prob`` and random seed ``seed``. All other entries are zero.

    Parameters
    ----------
    w0 : Data
        Lower bound of the scalar weight distribution. Scalar value, optionally
        with physical units (``brainunit.Quantity``).
    w1 : Data
        Upper bound of the scalar weight distribution. Must have the same
        dimension (units) as ``w0``.
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
        with scalarly distributed weights at connected positions and zeros
        elsewhere. Carries physical units if ``w0`` has units.

    See Also
    --------
    jitsmv : Matrix-vector product without materializing the matrix.

    Notes
    -----
    Each entry ``A[i, j]`` of the generated matrix follows the model:

        ``A[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Scalar(w0, w1)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent random variables. Equivalently:

    - ``A[i, j] ~ Scalar(w0, w1)`` with probability ``prob``
    - ``A[i, j] = 0`` with probability ``1 - prob``

    The expected value of each entry is:

        ``E[A[i, j]] = prob * (w0 + w1) / 2``

    The connectivity pattern and scalar variates are determined by ``seed`` and
    ``prob``. Using the same ``seed`` always produces the same matrix.

    This function materializes the full dense matrix. For implicit (non-materialized)
    matrix-vector products, use :func:`jitsmv` or :func:`jitsmm` instead.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_scalar.float import jits
        >>> dense = jits(0.1, 0.5, 0.2, seed=42, shape=(4, 6))
        >>> dense.shape
        (4, 6)
    """
    u.fail_for_dimension_mismatch(w0, w1, "w0 and w1 must have the same dimension.")
    w0, unitd = u.split_mantissa_unit(w0)
    w1 = u.Quantity(w1).to(unitd).mantissa
    clen = _initialize_conn_length(prob)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    res = jits_p_call(
        w0,
        w1,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd)


def jits(
    weight: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    matrix_mode: MatrixMode = 'mv',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    _warn_corder_ignored(corder)
    w0 = weight
    w1 = weight
    return _jits_impl(
        w0,
        w1,
        prob,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


jits.__doc__ = _jits_impl.__doc__


@namescope(name="brainevent.jitsmv", static_argnames=("shape", "transpose", "corder", "chunk_size", "target_chunks"))
def _jitsmv_impl(
    w0: Data,
    w1: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    """
    Float matrix-vector product with a JIT scalar connectivity matrix.

    Computes the product of a just-in-time generated sparse matrix with
    scalarly distributed weights and a dense vector. Unlike the binary
    variant, this function uses the full floating-point values of the vector
    elements.

    The sparse matrix ``A`` of shape ``(m, n)`` is never materialized. Each
    entry ``A[i, j]`` is drawn from ``Scalar(w0, w1)`` with
    probability ``prob``, seeded by ``seed``.

    Parameters
    ----------
    w0 : Data
        Lower bound of the scalar weight distribution. Scalar value, optionally
        with physical units (``brainunit.Quantity``).
    w1 : Data
        Upper bound of the scalar weight distribution. Must have the same
        dimension (units) as ``w0``.
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
    jitsmm : Matrix-matrix variant.
    binary_jitsmv : Event-driven (binary) variant.

    Notes
    -----
    The connectivity matrix ``A`` of shape ``(m, n)`` follows the model:

        ``A[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Scalar(w0, w1)`` and ``B[i, j] ~ Bernoulli(prob)``
    are independent, both determined by ``seed``.

    The float matrix-vector product computes:

        ``result[i] = sum_{j=0}^{n-1} A[i, j] * vector[j]``

    Unlike the binary variant (:func:`binary_jitsmv`), this uses the full
    floating-point values of ``vector`` rather than treating them as binary events.

    When ``transpose=True``, the operation becomes ``result = A^T @ vector``:

        ``result[j] = sum_{i=0}^{m-1} A[i, j] * vector[i]``

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_scalar.float import jitsmv
        >>> vec = jnp.ones(5)
        >>> result = jitsmv(0.1, 0.5, 0.2, vec, seed=42, shape=(3, 5))
        >>> result.shape
        (3,)
    """
    u.fail_for_dimension_mismatch(w0, w1, "w0 and w1 must have the same dimension.")
    seed = _initialize_seed(seed)
    w0, unitd = u.split_mantissa_unit(w0)
    w1 = u.Quantity(w1).to(unitd).mantissa
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = jitsmv_p_call(
        w0,
        w1,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


def jitsmv(
    weight: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    _warn_corder_ignored(corder)
    w0 = weight
    w1 = weight
    return _jitsmv_impl(
        w0,
        w1,
        prob,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


jitsmv.__doc__ = _jitsmv_impl.__doc__


@namescope(name="brainevent.jitsmm", static_argnames=("shape", "transpose", "corder", "chunk_size", "target_chunks"))
def _jitsmm_impl(
    w0: Data,
    w1: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    """
    Float matrix-matrix product with a JIT scalar connectivity matrix.

    Computes the product of a just-in-time generated sparse matrix with
    scalarly distributed weights and a dense matrix ``B``. Unlike the binary
    variant, this function uses the full floating-point values of ``B``.

    The sparse matrix ``A`` of shape ``(m, n)`` is never materialized. Each
    entry ``A[i, j]`` is drawn from ``Scalar(w0, w1)`` with
    probability ``prob``, seeded by ``seed``.

    Parameters
    ----------
    w0 : Data
        Lower bound of the scalar weight distribution. Scalar value, optionally
        with physical units (``brainunit.Quantity``).
    w1 : Data
        Upper bound of the scalar weight distribution. Must have the same
        dimension (units) as ``w0``.
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
    jitsmv : Matrix-vector variant.
    binary_jitsmm : Event-driven (binary) variant.

    Notes
    -----
    The connectivity matrix ``A`` of shape ``(m, n)`` follows the model:

        ``A[i, j] = U[i, j] * B_conn[i, j]``

    where ``U[i, j] ~ Scalar(w0, w1)`` and ``B_conn[i, j] ~ Bernoulli(prob)``
    are independent, both determined by ``seed``.

    The float matrix-matrix product computes:

        ``result[i, j] = sum_{k=0}^{n-1} A[i, k] * B[k, j]``

    Unlike the binary variant (:func:`binary_jitsmm`), this uses the full
    floating-point values of ``B`` rather than treating them as binary events.

    When ``transpose=True``, the operation becomes ``result = A^T @ B``:

        ``result[j, l] = sum_{i=0}^{m-1} A[i, j] * B[i, l]``

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_scalar.float import jitsmm
        >>> B = jnp.ones((5, 3))
        >>> result = jitsmm(0.1, 0.5, 0.2, B, seed=42, shape=(4, 5))
        >>> result.shape
        (4, 3)
    """
    u.fail_for_dimension_mismatch(w0, w1, "w0 and w1 must have the same dimension.")
    seed = _initialize_seed(seed)
    w0, unitd = u.split_mantissa_unit(w0)
    w1 = u.Quantity(w1).to(unitd).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = jitsmm_p_call(
        w0,
        w1,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        matrix_mode='mm',
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def jitsmm(
    weight: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    _warn_corder_ignored(corder)
    w0 = weight
    w1 = weight
    return _jitsmm_impl(
        w0,
        w1,
        prob,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


jitsmm.__doc__ = _jitsmm_impl.__doc__


def _jits_numba_kernel_generator(
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for materializing a JIT scalar connectivity matrix.

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
        A function ``kernel(w0, w1, clen, seed)`` that executes
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
        def kernel_impl(w0, w1, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[1]
            n = posts.shape[0]
            w00 = w0[0]
            w10 = w1[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_row in range(n):
                state = _lfsr_seed(seed0 + i_row * m)
                i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_col < m:
                    posts[i_row, i_col] = _lfsr_uniform(state, w00, w10)
                    i_col += _lfsr_random_integers(state, 1, clen0 - 1)


    else:
        # JIT matrix.T
        # - JIT matrix shape = [m, n]
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[1]
            n = posts.shape[0]
            w00 = w0[0]
            w10 = w1[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_col in range(m):
                state = _lfsr_seed(seed0 + i_col * n)
                i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_row < n:
                    posts[i_row, i_col] = _lfsr_uniform(state, w00, w10)
                    i_row += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(w0, w1, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w0, w1, clen, seed)

    return kernel


_dtype_sfx = {
    np.dtype('float16'): '_f16',
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
    np.dtype('bfloat16'): '_bf16',
}


def _jits_cuda_kernel(
    corder: bool = True,
    shape: MatrixShape = None,
    transpose: bool = False,
    matrix_mode: MatrixMode = 'mv',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs
):
    del corder
    if np.dtype(kwargs['w0_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light float jits CUDA currently supports float32 weights only")

    matrix_mode = _normalize_matrix_mode(matrix_mode)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jits.cu'),
        name='float_jits',
    )
    kernel_name = (
        'float_jits.jits_mv_f32'
        if matrix_mode == 'mv'
        else 'float_jits.jits_mm_aw_t4_f32'
    )

    def kernel(w0, w1, clen, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w0,
            w1,
            clen,
            seed,
            n_rows=np.int32(n_rows),
            n_cols=np.int32(n_cols),
            transpose=np.int32(int(transpose)),
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jits_jvp_wlow(
    w0_dot, w0, w1, clen, seed, *,
    shape, transpose: bool, corder: bool, matrix_mode: MatrixMode = 'mv', **kwargs
):
    """
    JVP rule for the ``w0`` argument of the JIT-scalar dense matrix generation.

    Parameters
    ----------
    w0_dot : jax.Array
        Tangent vector for the ``w0`` argument.
    w0, w1, clen, seed : jax.Array
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
    The derivative with respect to ``w0`` is ``w0_dot - jits(0, w0_dot)``,
    reflecting the affine structure ``A = w0 + (w1 - w0) * U``.
    """
    res = jits_p_call(
        0., w0_dot, clen, seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )[0]
    return [w0_dot - res]


def _jits_jvp_whigh(
    w1_dot, w0, w1, clen, seed, *,
    shape, transpose: bool, corder: bool, matrix_mode: MatrixMode = 'mv', **kwargs
):
    """
    JVP rule for the ``w1`` argument of the JIT-scalar dense matrix generation.

    Parameters
    ----------
    w1_dot : jax.Array
        Tangent vector for the ``w1`` argument.
    w0, w1, clen, seed : jax.Array
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
    return jits_p_call(
        0., w1_dot, clen, seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _wlow_tranpose(ct, seed, clen, **kwargs):
    """
    Compute the transpose contribution from ``w0`` for the dense matrix primitive.

    Parameters
    ----------
    ct : jax.Array
        Cotangent array.
    seed : jax.Array
        Random seed.
    clen : jax.Array
        Connection length parameter.
    **kwargs
        Keyword arguments passed to ``jits_p_call`` including ``shape``,
        ``transpose``, ``corder``, and ``backend``.

    Returns
    -------
    jax.Array
        Scalar cotangent for ``w0``, computed as ``sum(ct * (1 - U))``
        where ``U = jits(0, 1)``.

    Notes
    -----
    Uses the affine decomposition ``A = (1 - U) * w0 + U * w1``
    where ``U = jits(0, 1)`` represents the scalar random fractions.
    """
    # JITC * (high - low) + low
    forward = jits_p_call(0., 1., clen, seed, **kwargs)[0]
    return jnp.expand_dims((ct * (-forward + 1.)).sum(), axis=0)


def _whigh_tranpose(ct, seed, clen, **kwargs):
    """
    Compute the transpose contribution from ``w1`` for the dense matrix primitive.

    Parameters
    ----------
    ct : jax.Array
        Cotangent array.
    seed : jax.Array
        Random seed.
    clen : jax.Array
        Connection length parameter.
    **kwargs
        Keyword arguments passed to ``jits_p_call`` including ``shape``,
        ``transpose``, ``corder``, and ``backend``.

    Returns
    -------
    jax.Array
        Scalar cotangent for ``w1``, computed as ``sum(ct * U)``
        where ``U = jits(0, 1)``.

    Notes
    -----
    Uses the affine decomposition ``A = (1 - U) * w0 + U * w1``
    where ``U = jits(0, 1)`` represents the scalar random fractions.
    """
    # JITC * (high - low) + low
    forward = jits_p_call(0., 1., clen, seed, **kwargs)[0]
    return jnp.expand_dims((ct * forward).sum(), axis=0)


def _jits_transpose(
    ct, w0, w1, clen, seed, *,
    shape, transpose: bool, corder: bool, matrix_mode: MatrixMode = 'mv', **kwargs
):
    """
    Transpose (adjoint) rule for the JIT-scalar dense matrix generation.

    Parameters
    ----------
    ct : list
        Cotangent of the output.
    w0, w1, clen, seed : jax.Array or ad.UndefinedPrimal
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
        Cotangents for each input argument ``(w0, w1, clen, seed)``.

    Raises
    ------
    NotImplementedError
        If the undefined primal is not ``w0`` or ``w1``.
    """
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(w0):
        dwlow = _wlow_tranpose(
            ct,
            seed,
            clen,
            shape=shape,
            transpose=transpose,
            corder=corder,
            matrix_mode=matrix_mode,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return (dwlow, w1, clen, seed)
    elif ad.is_undefined_primal(w1):
        dwhigh = _whigh_tranpose(
            ct,
            seed,
            clen,
            shape=shape,
            transpose=transpose,
            corder=corder,
            matrix_mode=matrix_mode,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return (w0, dwhigh, clen, seed)
    else:
        raise NotImplementedError(
            'JITC matrix transpose is only implemented for the w0 and w1 arguments.'
        )


def _jits_batching(args, axes, **kwargs):
    """
    Batching rule for the JIT-scalar dense matrix generation primitive.

    Parameters
    ----------
    args : tuple
        Batched arguments ``(w0, w1, clen, seed)``.
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
    return general_batching_rule(jits_p, args, axes, **kwargs)


def _jits_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the JIT-scalar dense matrix generation.

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
            w0 = jnp.zeros(1, dtype=dtype)
            w1 = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (w0, w1, clen, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jits_p_call(
    w0,
    w1,
    clen,
    seed,
    *,
    shape,
    transpose: bool,
    corder: bool,
    matrix_mode: MatrixMode = 'mv',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for materializing a JIT scalar connectivity matrix.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``jits_p`` XLA custom kernel. This function expects pre-processed
    arguments (mantissa-only arrays, connection length instead of probability).

    Parameters
    ----------
    w0 : jax.Array
        Lower weight bound as a 1-D array of shape ``(1,)`` with floating dtype.
    w1 : jax.Array
        Upper weight bound as a 1-D array of shape ``(1,)`` with the same dtype
        as ``w0``.
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
    jits : High-level wrapper with unit handling.
    """
    w0 = jnp.atleast_1d(w0)
    w1 = jnp.atleast_1d(w1)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    assert jnp.issubdtype(w0.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w0.dtype == w1.dtype, "w0 and w1 must have the same dtype."
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    _warn_corder_ignored(corder)
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)

    out_info = (
        jax.ShapeDtypeStruct(shape[::-1], dtype=w0.dtype)
        if transpose else
        jax.ShapeDtypeStruct(shape, dtype=w0.dtype)
    )

    return jits_p(
        w0,
        w1,
        clen,
        seed,
        outs=[out_info],
        w0_info=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        w1_info=jax.ShapeDtypeStruct(w1.shape, w1.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        backend=backend,
    )


jits_p = XLACustomKernel(
    'float_jits',
    doc="""
Low-level XLA custom-kernel primitive for ``jits``.

This ``XLACustomKernel`` instance dispatches the JIT scalar connectivity matrix generation
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

This operation generates a sparse connectivity matrix where weights are scalarly distributed
between specified lower and upper bounds. The connectivity pattern is generated on-the-fly
using a deterministic PRNG seeded by the provided seed value.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``jits_p.available_backends(platform)``,
and the default backend can be configured with ``jits_p.set_default(platform, backend)``.

See Also
--------
jits : High-level user-facing function wrapper.
"""
)
jits_p.def_numba_kernel(_jits_numba_kernel_generator)
jits_p.def_cuda_raw_kernel(_jits_cuda_kernel, asdefault=True)
jits_p.def_jvp_rule2(_jits_jvp_wlow, _jits_jvp_whigh, None, None)
jits_p.def_transpose_rule(_jits_transpose)
jits_p.def_batching_rule(_jits_batching)
jits_p.def_call(jits_p_call)
jits_p.def_tags('jit_scalar', 'float')
jits_p.def_benchmark_data(_jits_benchmark_data)


# Kernel generators for JIT connection SPMV

def _jitsmv_numba_kernel_generator(
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for float JIT-scalar matrix-vector product.

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
        A function ``kernel(w0, w1, clen, vector, seed)`` that
        executes the Numba-compiled kernel and returns the result.
    """
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_uniform = get_numba_lfsr_uniform()

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, vector, seed, posts):
            n_col = posts.shape[0]
            n_row = vector.shape[0]
            w00 = w0[0]
            w10 = w1[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_col in range(n_col):
                state = _lfsr_seed(seed0 + i_col * n_row)
                i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                out = np.asarray(0., dtype=vector.dtype)
                while i_row < n_row:
                    out += vector[i_row] * _lfsr_uniform(state, w00, w10)
                    i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                posts[i_col] = out


    else:
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, vector, seed, posts):
            posts[:] = 0.
            num_col = posts.shape[0]
            num_row = vector.shape[0]
            w00 = w0[0]
            w10 = w1[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_row in range(num_row):
                state = _lfsr_seed(seed0 + i_row * num_col)
                v = vector[i_row]
                i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_col < num_col:
                    posts[i_col] += v * _lfsr_uniform(state, w00, w10)
                    i_col += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(w0, w1, clen, vector, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w0, w1, clen, vector, seed)

    return kernel


def _jitsmv_cuda_kernel(
    corder: bool = True,
    transpose: bool = False,
    shape: MatrixShape = None,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs
):
    del corder
    if np.dtype(kwargs['w0_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light float jitsmv CUDA currently supports float32 weights only")

    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jitsmv.cu'),
        name='float_jitsmv',
    )
    variant = 'scatter' if transpose else 'gather'
    kernel_name = f'float_jitsmv.jitsmv_{variant}_f32'

    def kernel(w0, w1, clen, vector, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w0,
            w1,
            clen,
            seed,
            vector,
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jitsmv_jvp_v(v_dot, w0, w1, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the vector argument of the float JIT-scalar matrix-vector product.

    Parameters
    ----------
    v_dot : jax.Array
        Tangent vector for the ``vector`` argument.
    w0, w1, clen, vector, seed : jax.Array
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
    return jitsmv_p_call(
        w0, w1, clen, v_dot, seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmv_jvp_wlow(w_dot, w0, w1, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w0`` argument of the float JIT-scalar matrix-vector product.

    Parameters
    ----------
    w_dot : jax.Array
        Tangent vector for the ``w0`` argument.
    w0, w1, clen, vector, seed : jax.Array
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
    return jitsmv_p_call(
        w_dot, w1, clen, vector, seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmv_jvp_whigh(w_dot, w0, w1, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w1`` argument of the float JIT-scalar matrix-vector product.

    Parameters
    ----------
    w_dot : jax.Array
        Tangent vector for the ``w1`` argument.
    w0, w1, clen, vector, seed : jax.Array
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
    return jitsmv_p_call(
        w0, w_dot, clen, vector, seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmv_transpose_rules(ct, w0, w1, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the float JIT-scalar matrix-vector product.

    Implements the VJP transpose for differentiation through the primitive.
    Supports transposing with respect to the ``vector``, ``w0``, or ``w1``
    arguments.

    Parameters
    ----------
    ct : list
        Cotangent of the output.
    w0, w1, clen, vector, seed : jax.Array or ad.UndefinedPrimal
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
        Cotangents for each input argument (w0, w1, clen, vector, seed).

    Raises
    ------
    NotImplementedError
        If the undefined primal is not ``vector``, ``w0``, or ``w1``.

    Notes
    -----
    For the weight bounds, the transpose uses an affine decomposition:

        ``y = w0 * C(v) + (w1 - w0) * U(v)``

    where ``U(v) = y(0, 1)`` and ``C(v) = y(1, 1)``. The cotangents are:

    - ``dL/dw1 = <ct, U(v)>``
    - ``dL/dw0  = <ct, C(v) - U(v)>``
    """
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        r = jitsmv_p_call(
            w0,
            w1,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        return w0, w1, clen, r, seed
    elif ad.is_undefined_primal(w0):
        # Fix the sampled connectivity and RNG stream (same `clen/seed/shape/transpose/corder`).
        # For each active entry:
        #   w_ij = w0 + (w1 - w0) * u_ij,  u_ij in [0, 1).
        # The linear map output is therefore affine in (w0, w1):
        #   y = w0 * C(v) + (w1 - w0) * U(v),
        # where
        #   U(v) = y(0, 1)  and  C(v) = y(1, 1).
        # Given cotangent ct, with inner product <a, b> = sum(a * b):
        #   dL/dw1 = <ct, U(v)>
        #   dL/dw0  = <ct, C(v) - U(v)>.
        ones = jnp.ones((1,), dtype=ct.dtype)
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        u_basis = jitsmv_p_call(
            zeros,
            ones,
            clen,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        c_basis = jitsmv_p_call(
            ones,
            ones,
            clen,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        dw0 = jnp.expand_dims(jnp.sum(ct * (c_basis - u_basis)), axis=0)
        return dw0, w1, clen, vector, seed
    elif ad.is_undefined_primal(w1):
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        ones = jnp.ones((1,), dtype=ct.dtype)
        u_basis = jitsmv_p_call(
            zeros,
            ones,
            clen,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        dw1 = jnp.expand_dims(jnp.sum(ct * u_basis), axis=0)
        return w0, dw1, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitsmv_batching(args, axes, **kwargs):
    """
    Batching rule for the float JIT-scalar matrix-vector product primitive.

    Handles ``vmap`` over the vector argument by promoting the operation to
    a matrix-matrix product (``jitsmm_p_call``).

    Parameters
    ----------
    args : tuple
        Batched arguments ``(w0, w1, clen, vector, seed)``.
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
        r = jitsmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            matrix_mode='mv',
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitsmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            matrix_mode='mv',
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(jitsmv_p, args, axes, **kwargs)


def _jitsmv_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the float JIT-scalar matrix-vector product.

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
            w0 = jnp.zeros(1, dtype=dtype)
            w1 = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w0, w1, clen, vector, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                )
            )
    return configs


def jitsmv_p_call(
    w0,
    w1,
    clen,
    vector,
    seed,
    *,
    shape,
    transpose: bool,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for the float JIT-scalar matrix-vector product.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``jitsmv_p`` XLA custom kernel. This function expects pre-processed
    arguments (mantissa-only arrays, connection length instead of probability).

    Parameters
    ----------
    w0 : jax.Array
        Lower weight bound as a 1-D array of shape ``(1,)`` with floating dtype.
    w1 : jax.Array
        Upper weight bound as a 1-D array of shape ``(1,)`` with the same dtype
        as ``w0``.
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
    jitsmv : High-level wrapper with unit handling and seed initialization.
    """
    w0 = jnp.atleast_1d(w0)
    w1 = jnp.atleast_1d(w1)
    clen = jnp.atleast_1d(clen)
    _warn_corder_ignored(corder)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert w0.shape == (1,), f"The weight shape should be (1,), but got {w0.shape}."
    assert w1.shape == (1,), f"The weight shape should be (1,), but got {w1.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."
    assert jnp.issubdtype(w0.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w0.dtype == w1.dtype, "w0 and w1 must have the same dtype."
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w0.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w0.dtype)
    )

    return jitsmv_p(
        w0,
        w1,
        clen,
        vector,
        seed,
        outs=[out_info],
        w0_info=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        w1_info=jax.ShapeDtypeStruct(w1.shape, w1.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        backend=backend,
    )


jitsmv_p = XLACustomKernel(
    'float_jitsmv',
    doc="""
Low-level XLA custom-kernel primitive for ``jitsmv``.

This ``XLACustomKernel`` instance dispatches the JIT scalar connectivity matrix-vector
multiplication with floating-point weights operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided by
the high-level wrapper.

In this operation, the connectivity matrix has weights scalarly distributed between
specified bounds, and the input vector contains floating-point values. The operation
computes a standard matrix-vector product without event-driven sparsity.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``jitsmv_p.available_backends(platform)``,
and the default backend can be configured with ``jitsmv_p.set_default(platform, backend)``.

See Also
--------
jitsmv : High-level user-facing function wrapper.
"""
)
jitsmv_p.def_numba_kernel(_jitsmv_numba_kernel_generator)
jitsmv_p.def_cuda_raw_kernel(_jitsmv_cuda_kernel, asdefault=True)
jitsmv_p.def_jvp_rule2(_jitsmv_jvp_wlow, _jitsmv_jvp_whigh, None, _jitsmv_jvp_v, None)
jitsmv_p.def_transpose_rule(_jitsmv_transpose_rules)
jitsmv_p.def_batching_rule(_jitsmv_batching)
jitsmv_p.def_call(jitsmv_p_call)
jitsmv_p.def_tags('jit_scalar', 'float')
jitsmv_p.def_benchmark_data(_jitsmv_benchmark_data)


def _jitsmm_numba_kernel_generator(
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for float JIT-scalar matrix-matrix product.

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
        A function ``kernel(w0, w1, clen, B, seed)`` that
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
        def kernel_impl(w0, w1, clen, B, seed, posts):
            m = posts.shape[0]
            n = posts.shape[1]
            k = B.shape[0]
            w00 = w0[0]
            w10 = w1[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for i_m in range(m):
                state = _lfsr_seed(seed0 + i_m * k)
                i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                out = np.zeros(n, dtype=B.dtype)
                while i_k < k:
                    out += B[i_k] * _lfsr_uniform(state, w00, w10)
                    i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                posts[i_m] = out


    else:
        # JIT Matrix.T @ B
        # - JIT matrix: [k, m]
        # - B: [k, n]
        @numba.njit(fastmath=True)
        def kernel_impl(w0, w1, clen, B, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]
            k = B.shape[0]
            w00 = w0[0]
            w10 = w1[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for i_k in range(k):
                state = _lfsr_seed(seed0 + i_k * m)
                out = B[i_k]
                i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_m < m:
                    posts[i_m] += out * _lfsr_uniform(state, w00, w10)
                    i_m += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(w0, w1, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(w0, w1, clen, B, seed)

    return kernel


def _jitsmm_cuda_kernel(
    corder: bool = True,
    B_info: jax.ShapeDtypeStruct = None,
    transpose: bool = False,
    shape: MatrixShape = None,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mm',
    **kwargs
):
    del corder
    if np.dtype(kwargs['w0_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light float jitsmm CUDA currently supports float32 weights only")

    matrix_mode = _normalize_matrix_mode(matrix_mode)
    m = int(shape[0])
    k = int(shape[1])
    n = int(B_info.shape[1])
    chunk_size_value = _normalize_chunk_size(k, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jitsmm.cu'),
        name='float_jitsmm',
    )
    variant = 'scatter' if transpose else 'gather'
    prefix = 'jitsmm_mv' if matrix_mode == 'mv' else 'jitsmm'
    kernel_name = f'float_jitsmm.{prefix}_{variant}_f32'

    def kernel(w0, w1, clen, B, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w0,
            w1,
            clen,
            seed,
            B,
            m=np.int32(m),
            k=np.int32(k),
            n=np.int32(n),
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jitsmm_jvp_wlow(
    w_dot, w0, w1, clen, B, seed, *,
    shape, transpose, corder, matrix_mode: MatrixMode = 'mm', **kwargs
):
    """
    JVP rule for the ``w0`` argument of the float JIT-scalar matrix-matrix product.

    Parameters
    ----------
    w_dot : jax.Array
        Tangent vector for the ``w0`` argument.
    w0, w1, clen, B, seed : jax.Array
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
    return jitsmm_p_call(
        w_dot, w1, clen, B, seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmm_jvp_whigh(
    w_dot, w0, w1, clen, B, seed, *,
    shape, transpose, corder, matrix_mode: MatrixMode = 'mm', **kwargs
):
    """
    JVP rule for the ``w1`` argument of the float JIT-scalar matrix-matrix product.

    Parameters
    ----------
    w_dot : jax.Array
        Tangent vector for the ``w1`` argument.
    w0, w1, clen, B, seed : jax.Array
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
    return jitsmm_p_call(
        w0, w_dot, clen, B, seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmm_jvp_B(
    B_dot, w0, w1, clen, B, seed, *,
    shape, transpose, corder, matrix_mode: MatrixMode = 'mm', **kwargs
):
    """
    JVP rule for the ``B`` argument of the float JIT-scalar matrix-matrix product.

    Parameters
    ----------
    B_dot : jax.Array
        Tangent matrix for the ``B`` argument.
    w0, w1, clen, B, seed : jax.Array
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
    return jitsmm_p_call(
        w0, w1, clen, B_dot, seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmm_transpose_rules(
    ct, w0, w1, clen, B, seed, *,
    shape, transpose, corder, matrix_mode: MatrixMode = 'mm', **kwargs
):
    """
    Transpose (adjoint) rule for the float JIT-scalar matrix-matrix product.

    Implements the VJP transpose for differentiation through the primitive.
    Supports transposing with respect to ``B``, ``w0``, or ``w1``.

    Parameters
    ----------
    ct : list
        Cotangent of the output.
    w0, w1, clen, B, seed : jax.Array or ad.UndefinedPrimal
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
        Cotangents for each input argument (w0, w1, clen, B, seed).

    Raises
    ------
    NotImplementedError
        If the undefined primal is not ``B``, ``w0``, or ``w1``.

    Notes
    -----
    For the weight bounds, the transpose uses the affine decomposition:

        ``Y = w0 * C(B) + (w1 - w0) * U(B)``

    where ``U(B) = Y(0, 1)`` and ``C(B) = Y(1, 1)``. The cotangents are:

    - ``dL/dw1 = <ct, U(B)>``
    - ``dL/dw0  = <ct, C(B) - U(B)>``
    """
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        dB = jitsmm_p_call(
            w0,
            w1,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            matrix_mode=matrix_mode,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        return w0, w1, clen, dB, seed
    elif ad.is_undefined_primal(w0):
        # Same affine decomposition as _jitsmv_transpose_rules, now for matrix right operand B:
        #   Y = w0 * C(B) + (w1 - w0) * U(B),
        #   U(B) = Y(0, 1), C(B) = Y(1, 1).
        # Hence:
        #   dL/dw1 = <ct, U(B)>
        #   dL/dw0  = <ct, C(B) - U(B)>.
        ones = jnp.ones((1,), dtype=ct.dtype)
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        u_basis = jitsmm_p_call(
            zeros,
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            matrix_mode=matrix_mode,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        c_basis = jitsmm_p_call(
            ones,
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            matrix_mode=matrix_mode,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        dw0 = jnp.expand_dims(jnp.sum(ct * (c_basis - u_basis)), axis=0)
        return dw0, w1, clen, B, seed
    elif ad.is_undefined_primal(w1):
        zeros = jnp.zeros((1,), dtype=ct.dtype)
        ones = jnp.ones((1,), dtype=ct.dtype)
        u_basis = jitsmm_p_call(
            zeros,
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            matrix_mode=matrix_mode,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        dw1 = jnp.expand_dims(jnp.sum(ct * u_basis), axis=0)
        return w0, dw1, clen, B, seed
    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_scalar not implemented for '
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
        Batched arguments ``(w0, w1, clen, B, seed)``.
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
    r = jitsmm_p_call(
        args[0],
        args[1],
        args[2],
        B,
        args[4],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
        matrix_mode=kwargs.get('matrix_mode', 'mm'),
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitsmm_batching(args, axes, **kwargs):
    """
    Batching rule for the float JIT-scalar matrix-matrix product primitive.

    Handles ``vmap`` over the ``B`` argument along different axes by
    reshaping and delegating to ``jitsmm_p_call``.

    Parameters
    ----------
    args : tuple
        Batched arguments ``(w0, w1, clen, B, seed)``.
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
        return general_batching_rule(jitsmm_p, args, axes, **kwargs)


def _jitsmm_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the float JIT-scalar matrix-matrix product.

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
            w0 = jnp.zeros(1, dtype=dtype)
            w1 = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w0, w1, clen, B, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                )
            )
    return configs


def jitsmm_p_call(
    w0,
    w1,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mm',
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for the float JIT-scalar matrix-matrix product.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``jitsmm_p`` XLA custom kernel. This function expects pre-processed
    arguments (mantissa-only arrays, connection length instead of probability).

    Parameters
    ----------
    w0 : jax.Array
        Lower weight bound as a 1-D array of shape ``(1,)`` with floating dtype.
    w1 : jax.Array
        Upper weight bound as a 1-D array of shape ``(1,)`` with the same dtype
        as ``w0``.
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
    jitsmm : High-level wrapper with unit handling and seed initialization.
    """
    w0 = jnp.atleast_1d(w0)
    w1 = jnp.atleast_1d(w1)
    clen = jnp.atleast_1d(clen)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    _warn_corder_ignored(corder)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert w0.ndim == 1, "The weight should be a 1D array."
    assert w1.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert w0.shape == (1,), "The weight should be a scalar."
    assert w1.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"
    assert jnp.issubdtype(w0.dtype, jnp.floating), 'Weights must be a floating-point type.'
    assert w0.dtype == w1.dtype, "w0 and w1 must have the same dtype."
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w0.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w0.dtype)
    )

    return jitsmm_p(
        w0,
        w1,
        clen,
        B,
        seed,
        outs=[out_info],
        w0_info=jax.ShapeDtypeStruct(w0.shape, w0.dtype),
        w1_info=jax.ShapeDtypeStruct(w1.shape, w1.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        chunk_size=chunk_size_value,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
        TITLE_SIZE=B.shape[1],
        backend=backend,
    )


jitsmm_p = XLACustomKernel(
    'float_jitsmm',
    doc="""
Low-level XLA custom-kernel primitive for ``jitsmm``.

This ``XLACustomKernel`` instance dispatches the JIT scalar connectivity matrix-matrix
multiplication with floating-point weights operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided by
the high-level wrapper.

In this operation, the connectivity matrix has weights scalarly distributed between
specified bounds, and the input matrix contains floating-point values. Each column of
the input is processed independently in a standard matrix-matrix product.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``jitsmm_p.available_backends(platform)``,
and the default backend can be configured with ``jitsmm_p.set_default(platform, backend)``.

See Also
--------
jitsmm : High-level user-facing function wrapper.
"""
)
jitsmm_p.def_numba_kernel(_jitsmm_numba_kernel_generator)
jitsmm_p.def_cuda_raw_kernel(_jitsmm_cuda_kernel, asdefault=True)
jitsmm_p.def_jvp_rule2(_jitsmm_jvp_wlow, _jitsmm_jvp_whigh, None, _jitsmm_jvp_B, None)
jitsmm_p.def_transpose_rule(_jitsmm_transpose_rules)
jitsmm_p.def_batching_rule(_jitsmm_batching)
jitsmm_p.def_call(jitsmm_p_call)
jitsmm_p.def_tags('jit_scalar', 'float')
jitsmm_p.def_benchmark_data(_jitsmm_benchmark_data)
