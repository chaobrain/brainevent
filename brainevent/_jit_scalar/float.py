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


def _normalize_matrix_mode(matrix_mode: MatrixMode) -> MatrixMode:
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


def _warn_corder_deprecated(corder: Optional[bool]) -> None:
    if corder is None:
        return
    warnings.warn(
        "corder is deprecated and ignored by the light JIT scalar implementation.",
        FutureWarning,
        stacklevel=3,
    )


def _light_options(kwargs):
    return {
        'chunk_size': kwargs.get('chunk_size', None),
        'target_chunks': kwargs.get('target_chunks', 4),
    }


@namescope(name="brainevent.jits", static_argnames=("shape", "transpose", "matrix_mode", "chunk_size", "target_chunks"))
def _jits_impl(
    weight: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
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
    weight, unitd = u.split_mantissa_unit(weight)
    clen = _initialize_conn_length(prob)
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    res = jits_p_call(
        weight,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
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
    corder: Optional[bool] = None,
    matrix_mode: MatrixMode = 'mv',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    _warn_corder_deprecated(corder)
    return _jits_impl(
        weight,
        prob,
        seed,
        shape=shape,
        transpose=transpose,
        matrix_mode=matrix_mode,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


jits.__doc__ = _jits_impl.__doc__


@namescope(name="brainevent.jitsmv", static_argnames=("shape", "transpose", "chunk_size", "target_chunks"))
def _jitsmv_impl(
    weight: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
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
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = jitsmv_p_call(
        weight,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
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
    corder: Optional[bool] = None,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    _warn_corder_deprecated(corder)
    return _jitsmv_impl(
        weight,
        prob,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


jitsmv.__doc__ = _jitsmv_impl.__doc__


@namescope(name="brainevent.jitsmm", static_argnames=("shape", "transpose", "matrix_mode", "chunk_size", "target_chunks"))
def _jitsmm_impl(
    weight: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    matrix_mode: MatrixMode = 'mm',
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
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = jitsmm_p_call(
        weight,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        matrix_mode=matrix_mode,
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
    corder: Optional[bool] = None,
    matrix_mode: MatrixMode = 'mm',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
) -> Data:
    _warn_corder_deprecated(corder)
    return _jitsmm_impl(
        weight,
        prob,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        matrix_mode=matrix_mode,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


jitsmm.__doc__ = _jitsmm_impl.__doc__


def _jits_numba_kernel_generator(
    transpose: bool = False,
    **kwargs
):
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()

    if transpose:
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, posts):
            posts[:] = 0.
            n_rows = posts.shape[1]
            n_cols = posts.shape[0]
            w = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for row in range(n_rows):
                state = _lfsr_seed(seed0 + row * n_cols)
                col = _lfsr_random_integers(state, 0, clen0 - 1)
                while col < n_cols:
                    posts[col, row] = w
                    col += _lfsr_random_integers(state, 1, clen0 - 1)
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, posts):
            posts[:] = 0.
            n_rows = posts.shape[0]
            n_cols = posts.shape[1]
            w = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for row in range(n_rows):
                state = _lfsr_seed(seed0 + row * n_cols)
                col = _lfsr_random_integers(state, 0, clen0 - 1)
                while col < n_cols:
                    posts[row, col] = w
                    col += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(weight, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, seed)

    return kernel


_dtype_sfx = {
    np.dtype('float16'): '_f16',
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
    np.dtype('bfloat16'): '_bf16',
}


def _jits_cuda_kernel(
    shape: MatrixShape,
    transpose: bool = False,
    matrix_mode: MatrixMode = 'mv',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs
):
    if np.dtype(kwargs['weight_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light float jits CUDA currently supports float32 weights only")
    assert shape is not None, "shape must be provided"

    matrix_mode = _normalize_matrix_mode(matrix_mode)
    n_rows, n_cols = int(shape[0]), int(shape[1])
    chunk_size_value = _normalize_chunk_size(n_cols, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jits.cu'),
        name='float_jits',
    )
    if matrix_mode == 'mv':
        kernel_name = (
            'float_jits.jits_mv_trans_f32'
            if transpose
            else 'float_jits.jits_mv_notrans_f32'
        )
    else:
        kernel_name = (
            'float_jits.jits_mm_aw_t4_trans_f32'
            if transpose
            else 'float_jits.jits_mm_aw_t4_notrans_f32'
        )

    def kernel(weight, clen, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            weight,
            clen,
            seed,
            n_rows=np.int32(n_rows),
            n_cols=np.int32(n_cols),
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jits_jvp_weight(
    weight_dot, weight, clen, seed, *,
    shape, transpose: bool, matrix_mode: MatrixMode = 'mv', **kwargs
):
    return jits_p_call(
        weight_dot, clen, seed,
        shape=shape,
        transpose=transpose,
        matrix_mode=matrix_mode,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _weight_transpose(ct, seed, clen, **kwargs):
    ones = jnp.ones((1,), dtype=ct.dtype)
    mask = jits_p_call(ones, clen, seed, **kwargs)[0]
    return jnp.expand_dims((ct * mask).sum(), axis=0)


def _jits_transpose(
    ct, weight, clen, seed, *,
    shape, transpose: bool, matrix_mode: MatrixMode = 'mv', **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(weight):
        dweight = _weight_transpose(
            ct,
            seed,
            clen,
            shape=shape,
            transpose=transpose,
            matrix_mode=matrix_mode,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return (dweight, clen, seed)
    else:
        raise NotImplementedError(
            'JITC scalar matrix transpose is only implemented for the weight argument.'
        )


def _jits_batching(args, axes, **kwargs):
    return general_batching_rule(jits_p, args, axes, **kwargs)


def _jits_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        weight = jnp.ones(1, dtype=dtype)
        clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
        seed = jnp.asarray(42, dtype=jnp.uint32)
        name = f"{'T' if transpose else 'NT'}"
        configs.append(BenchmarkConfig(name, (weight, clen, seed), {
            'shape': (n_pre, n_post), 'transpose': transpose
        }))
    return configs


def jits_p_call(
    weight,
    clen,
    seed,
    *,
    shape,
    transpose: bool,
    matrix_mode: MatrixMode = 'mv',
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    assert jnp.issubdtype(weight.dtype, jnp.floating), 'Weights must be a floating-point type.'
    matrix_mode = _normalize_matrix_mode(matrix_mode)
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)

    out_info = (
        jax.ShapeDtypeStruct(shape[::-1], dtype=weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct(shape, dtype=weight.dtype)
    )

    return jits_p(
        weight,
        clen,
        seed,
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
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
jits_p.def_jvp_rule2(_jits_jvp_weight, None, None)
jits_p.def_transpose_rule(_jits_transpose)
jits_p.def_batching_rule(_jits_batching)
jits_p.def_call(jits_p_call)
jits_p.def_tags('jit_scalar', 'float')
jits_p.def_benchmark_data(_jits_benchmark_data)


# Kernel generators for JIT connection SPMV

def _jitsmv_numba_kernel_generator(
    transpose: bool = False,
    **kwargs
):
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()

    if transpose:
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, vector, seed, posts):
            posts[:] = 0.
            n_rows = vector.shape[0]
            n_cols = posts.shape[0]
            w = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for row in range(n_rows):
                state = _lfsr_seed(seed0 + row * n_cols)
                v = vector[row]
                col = _lfsr_random_integers(state, 0, clen0 - 1)
                while col < n_cols:
                    posts[col] += v * w
                    col += _lfsr_random_integers(state, 1, clen0 - 1)
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, vector, seed, posts):
            n_rows = posts.shape[0]
            n_cols = vector.shape[0]
            w = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for row in range(n_rows):
                state = _lfsr_seed(seed0 + row * n_cols)
                col = _lfsr_random_integers(state, 0, clen0 - 1)
                out = np.asarray(0., dtype=vector.dtype)
                while col < n_cols:
                    out += vector[col] * w
                    col += _lfsr_random_integers(state, 1, clen0 - 1)
                posts[row] = out

    def kernel(weight, clen, vector, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, vector, seed)

    return kernel


def _jitsmv_cuda_kernel(
    shape: MatrixShape,
    transpose: bool = False,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs
):
    if np.dtype(kwargs['weight_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light float jitsmv CUDA currently supports float32 weights only")
    assert shape is not None, "shape must be provided"

    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jitsmv.cu'),
        name='float_jitsmv',
    )
    variant = 'trans' if transpose else 'notrans'
    kernel_name = f'float_jitsmv.jitsmv_{variant}_f32'

    def kernel(weight, clen, vector, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            weight,
            clen,
            seed,
            vector,
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jitsmv_jvp_v(v_dot, weight, clen, vector, seed, *, shape, transpose, **kwargs):
    return jitsmv_p_call(
        weight, clen, v_dot, seed,
        shape=shape,
        transpose=transpose,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmv_jvp_weight(w_dot, weight, clen, vector, seed, *, shape, transpose, **kwargs):
    return jitsmv_p_call(
        w_dot, clen, vector, seed,
        shape=shape,
        transpose=transpose,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmv_transpose_rules(ct, weight, clen, vector, seed, *, shape, transpose, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        r = jitsmv_p_call(
            weight,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        return weight, clen, r, seed
    elif ad.is_undefined_primal(weight):
        ones = jnp.ones((1,), dtype=ct.dtype)
        basis = jitsmv_p_call(
            ones,
            clen,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        dweight = jnp.expand_dims(jnp.sum(ct * basis), axis=0)
        return dweight, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitsmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitsmm_p_call(
            args[0],
            args[1],
            args[2].T,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            matrix_mode='mv',
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitsmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            matrix_mode='mv',
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(jitsmv_p, args, axes, **kwargs)


def _jitsmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        weight = jnp.ones(1, dtype=dtype)
        clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
        v_size = n_post if not transpose else n_pre
        vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
        seed = jnp.asarray(42, dtype=jnp.uint32)
        name = f"{'T' if transpose else 'NT'}"
        configs.append(
            BenchmarkConfig(
                name,
                (weight, clen, vector, seed),
                {'shape': (n_pre, n_post), 'transpose': transpose}
            )
        )
    return configs


def jitsmv_p_call(
    weight,
    clen,
    vector,
    seed,
    *,
    shape,
    transpose: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert weight.shape == (1,), f"The weight shape should be (1,), but got {weight.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."
    assert jnp.issubdtype(weight.dtype, jnp.floating), 'Weights must be a floating-point type.'
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weight.dtype)
    )

    return jitsmv_p(
        weight,
        clen,
        vector,
        seed,
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
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
jitsmv_p.def_jvp_rule2(_jitsmv_jvp_weight, None, _jitsmv_jvp_v, None)
jitsmv_p.def_transpose_rule(_jitsmv_transpose_rules)
jitsmv_p.def_batching_rule(_jitsmv_batching)
jitsmv_p.def_call(jitsmv_p_call)
jitsmv_p.def_tags('jit_scalar', 'float')
jitsmv_p.def_benchmark_data(_jitsmv_benchmark_data)


def _jitsmm_numba_kernel_generator(
    transpose: bool = False,
    **kwargs
):
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()

    if transpose:
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, B, seed, posts):
            posts[:] = 0.
            n_rows = B.shape[0]
            n_cols = posts.shape[0]
            w = weight[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for row in range(n_rows):
                state = _lfsr_seed(seed0 + row * n_cols)
                out = B[row]
                col = _lfsr_random_integers(state, 0, clen0 - 1)
                while col < n_cols:
                    posts[col] += out * w
                    col += _lfsr_random_integers(state, 1, clen0 - 1)
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, B, seed, posts):
            n_rows = posts.shape[0]
            n_cols = B.shape[0]
            n_batch = B.shape[1]
            w = weight[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for row in range(n_rows):
                state = _lfsr_seed(seed0 + row * n_cols)
                col = _lfsr_random_integers(state, 0, clen0 - 1)
                out = np.zeros(n_batch, dtype=B.dtype)
                while col < n_cols:
                    out += B[col] * w
                    col += _lfsr_random_integers(state, 1, clen0 - 1)
                posts[row] = out

    def kernel(weight, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, B, seed)

    return kernel


def _jitsmm_cuda_kernel(
    shape: MatrixShape,
    B_info: Optional[jax.ShapeDtypeStruct] = None,
    transpose: bool = False,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mm',
    **kwargs
):
    if np.dtype(kwargs['weight_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light float jitsmm CUDA currently supports float32 weights only")
    assert shape is not None, "shape must be provided"
    assert B_info is not None, "B_info must be provided"

    matrix_mode = _normalize_matrix_mode(matrix_mode)
    m = int(shape[0])
    k = int(shape[1])
    n = int(B_info.shape[1])
    chunk_size_value = _normalize_chunk_size(k, chunk_size, target_chunks)
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jitsmm.cu'),
        name='float_jitsmm',
    )
    variant = 'trans' if transpose else 'notrans'
    prefix = 'jitsmm_mv' if matrix_mode == 'mv' else 'jitsmm'
    kernel_name = f'float_jitsmm.{prefix}_{variant}_f32'

    def kernel(weight, clen, B, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            weight,
            clen,
            seed,
            B,
            m=np.int32(m),
            k=np.int32(k),
            n=np.int32(n),
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jitsmm_jvp_weight(
    w_dot, weight, clen, B, seed, *,
    shape, transpose, matrix_mode: MatrixMode = 'mm', **kwargs
):
    return jitsmm_p_call(
        w_dot, clen, B, seed,
        shape=shape,
        transpose=transpose,
        matrix_mode=matrix_mode,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmm_jvp_B(
    B_dot, weight, clen, B, seed, *,
    shape, transpose, matrix_mode: MatrixMode = 'mm', **kwargs
):
    return jitsmm_p_call(
        weight, clen, B_dot, seed,
        shape=shape,
        transpose=transpose,
        matrix_mode=matrix_mode,
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )


def _jitsmm_transpose_rules(
    ct, weight, clen, B, seed, *,
    shape, transpose, matrix_mode: MatrixMode = 'mm', **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        dB = jitsmm_p_call(
            weight,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            matrix_mode=matrix_mode,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        return weight, clen, dB, seed
    elif ad.is_undefined_primal(weight):
        ones = jnp.ones((1,), dtype=ct.dtype)
        basis = jitsmm_p_call(
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            matrix_mode=matrix_mode,
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )[0]
        dweight = jnp.expand_dims(jnp.sum(ct * basis), axis=0)
        return dweight, clen, B, seed
    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_scalar not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = jitsmm_p_call(
        args[0],
        args[1],
        B,
        args[3],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        matrix_mode=kwargs.get('matrix_mode', 'mm'),
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitsmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, 1, None):
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, 2, None):
        return _batching_axis1(args, axis=2, **kwargs)

    else:
        return general_batching_rule(jitsmm_p, args, axes, **kwargs)


def _jitsmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        weight = jnp.ones(1, dtype=dtype)
        clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
        b_rows = n_post if not transpose else n_pre
        B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
        seed = jnp.asarray(42, dtype=jnp.uint32)
        name = f"{'T' if transpose else 'NT'}"
        configs.append(
            BenchmarkConfig(
                name,
                (weight, clen, B, seed),
                {'shape': (n_pre, n_post), 'transpose': transpose}
            )
        )
    return configs


def jitsmm_p_call(
    weight,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    matrix_mode: MatrixMode = 'mm',
    backend: Optional[str] = None,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    matrix_mode = _normalize_matrix_mode(matrix_mode)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert weight.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert weight.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"
    assert jnp.issubdtype(weight.dtype, jnp.floating), 'Weights must be a floating-point type.'
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weight.dtype)
    )

    return jitsmm_p(
        weight,
        clen,
        B,
        seed,
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
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
jitsmm_p.def_jvp_rule2(_jitsmm_jvp_weight, None, _jitsmm_jvp_B, None)
jitsmm_p.def_transpose_rule(_jitsmm_transpose_rules)
jitsmm_p.def_batching_rule(_jitsmm_batching)
jitsmm_p.def_call(jitsmm_p_call)
jitsmm_p.def_tags('jit_scalar', 'float')
jitsmm_p.def_benchmark_data(_jitsmm_benchmark_data)
