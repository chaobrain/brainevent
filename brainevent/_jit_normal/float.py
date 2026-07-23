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

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._compatible_import import Tracer
from brainevent._data import _initialize_seed, _initialize_conn_length
from brainevent._misc import namescope
from brainevent._numba_random import get_numba_light_rng_funcs
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig
from brainevent._op import load_cuda_file
from brainevent._typing import Data, MatrixShape

__all__ = [
    "jitn",
    "jitn_p",
    "jitnmv",
    "jitnmv_p",
    "jitnmm",
    "jitnmm_p",
]

MatrixMode = Literal['mv', 'mm']

_dtype_sfx = {
    np.dtype('float16'): '_f16',
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
    np.dtype('bfloat16'): '_bf16',
}


def _normalize_matrix_mode(matrix_mode: MatrixMode) -> MatrixMode:
    if matrix_mode not in ('mv', 'mm'):
        raise ValueError(f"matrix_mode must be 'mv' or 'mm', got {matrix_mode!r}.")
    return matrix_mode


def _normalize_chunk_size(n_cols, chunk_size, target_chunks=4):
    if chunk_size is None:
        target_chunks = int(target_chunks)
        if target_chunks <= 0:
            raise ValueError("target_chunks must be positive")
        chunk_size = max(1, (int(n_cols) + target_chunks - 1) // target_chunks)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return chunk_size


def _is_static_zero_prob(prob: float, *, op_name: str) -> bool:
    if isinstance(prob, Tracer):
        return False
    prob_arr = np.asarray(prob)
    if prob_arr.size != 1:
        raise ValueError(f"{op_name}: prob must be a scalar, but got shape {prob_arr.shape}.")
    prob_scalar = float(prob_arr.item())
    if not np.isfinite(prob_scalar):
        raise ValueError(f"{op_name}: prob must be finite, but got {prob_scalar}.")
    if not (0. <= prob_scalar <= 1.):
        raise ValueError(f"{op_name}: prob must be in [0, 1], but got {prob_scalar}.")
    return prob_scalar == 0.


_MV_STRIDE = 32
_MM_STRIDE = 4


@namescope(static_argnames=("shape", "transpose", "corder", "matrix_mode"))
def jitn(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    matrix_mode: MatrixMode,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    """Materialise a JIT normally-distributed random connectivity matrix.

    Generates a dense matrix of shape *shape* (or its transpose) where each
    element is drawn from ``Normal(w_loc, w_scale)`` with independent
    Bernoulli masking at probability *prob*.

    Parameters
    ----------
    w_loc : scalar or Quantity
        Mean of the normal weight distribution.
    w_scale : scalar or Quantity
        Standard deviation of the normal weight distribution.  Must have
        the same physical unit as *w_loc*.
    prob : float
        Connection probability in ``[0, 1]``.
    seed : int
        RNG seed for reproducible connectivity.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.
    transpose : bool, optional
        If ``True``, return the transposed matrix of shape
        ``(n_post, n_pre)``.  Default is ``False``.
    corder : bool, optional
        If ``True`` (default), iterate in column-major order internally.
    backend : str or None, optional
        Compute backend (e.g. ``'numba'``, ``'pallas'``).

    Returns
    -------
    jax.Array or Quantity
        Dense matrix of shape ``shape`` (or ``shape[::-1]`` when
        *transpose* is ``True``).

    Raises
    ------
    ValueError
        If ``prob`` is not a scalar, is not finite, or is outside ``[0, 1]``.

    See Also
    --------
    jitnmv : Matrix-vector multiply without materialising the matrix.
    jitnmm : Matrix-matrix multiply without materialising the matrix.
    jits : Scalar-weight variant (all non-zeros share one weight).
    jitu : Uniform-weight variant.

    Notes
    -----
    Each entry ``W[i, j]`` of the generated matrix follows the model:

        ``W[i, j] = N(w_loc, w_scale) * B[i, j]``

    where ``N(w_loc, w_scale)`` is a draw from a normal distribution and
    ``B[i, j] ~ Bernoulli(prob)`` is a binary mask.  Equivalently:

    - ``W[i, j] ~ Normal(w_loc, w_scale)`` with probability ``prob``
    - ``W[i, j] = 0`` with probability ``1 - prob``

    The expected value of each entry is ``E[W[i, j]] = prob * w_loc``.

    The connectivity pattern and normal variates are fully determined by
    ``seed`` and ``prob``.  Using the same ``seed`` always produces the
    same matrix.

    This function materialises the full dense matrix.  For implicit
    (non-materialised) products, use :func:`jitnmv` or :func:`jitnmm`.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_normal.float import jitn
        >>> W = jitn(0.0, 1.0, prob=0.1, seed=42, shape=(100, 50))
        >>> W.shape
        (100, 50)
    """
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    out_dtype = jnp.asarray(w_loc).dtype
    if _is_static_zero_prob(prob, op_name="jitn"):
        out_shape = shape[::-1] if transpose else shape
        return u.maybe_decimal(jnp.zeros(out_shape, dtype=out_dtype) * unitd)
    clen = _initialize_conn_length(prob)
    res = jitn_p_call(
        w_loc,
        w_scale,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd)


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitnmv(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    """JIT normally-distributed matrix-vector product.

    Computes ``W @ v`` (or ``W.T @ v``) where ``W`` is a random matrix
    with entries drawn from ``Normal(w_loc, w_scale)`` masked by
    Bernoulli(*prob*), without materialising ``W``.

    Parameters
    ----------
    w_loc : scalar or Quantity
        Mean of the normal weight distribution.
    w_scale : scalar or Quantity
        Standard deviation.  Must share units with *w_loc*.
    prob : float
        Connection probability in ``[0, 1]``.
    vector : jax.Array or Quantity
        Input vector of shape ``(k,)`` where ``k`` equals ``shape[0]``
        when *transpose* is ``True``, or ``shape[1]`` otherwise.
    seed : int or None, optional
        RNG seed.  ``None`` generates a random seed.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.
    transpose : bool, optional
        If ``True``, multiply by the transpose.  Default is ``False``.
    corder : bool, optional
        Column-major iteration order.  Default is ``True``.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    jax.Array or Quantity
        Output vector.  Shape is ``(shape[1],)`` when *transpose* is
        ``True``, or ``(shape[0],)`` otherwise.

    Raises
    ------
    ValueError
        If ``prob`` is not a scalar, is not finite, or is outside ``[0, 1]``.
    ValueError
        If *vector* is not 1-D or its length does not match the matrix shape.

    See Also
    --------
    jitn : Materialise the full matrix.
    jitnmm : Matrix-matrix multiply variant.

    Notes
    -----
    The connectivity matrix ``W`` of shape ``(m, n)`` follows the model:

        ``W[i, j] = N(w_loc, w_scale) * B[i, j]``

    where ``N(w_loc, w_scale)`` is a normal draw and
    ``B[i, j] ~ Bernoulli(prob)`` is a binary mask, both determined by
    ``seed``.

    The matrix-vector product computes:

        ``y[i] = sum_{j=0}^{n-1} W[i, j] * v[j]``

    When ``transpose=True``, the operation becomes ``y = W^T @ v``:

        ``y[j] = sum_{i=0}^{m-1} W[i, j] * v[i]``

    The matrix is never materialised; weights are generated and consumed
    on the fly, avoiding ``O(m * n)`` memory.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_normal.float import jitnmv
        >>> v = jnp.ones(50)
        >>> y = jitnmv(0.0, 1.0, 0.1, v, seed=42, shape=(100, 50))
        >>> y.shape
        (100,)
    """
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    seed = _initialize_seed(seed)
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    vector, unitv = u.split_mantissa_unit(vector)
    out_dtype = jnp.asarray(w_loc).dtype
    if _is_static_zero_prob(prob, op_name="jitnmv"):
        if vector.ndim != 1:
            raise ValueError(f"jitnmv: vector must be 1D, but got {vector.ndim}D.")
        expected = shape[0] if transpose else shape[1]
        if vector.shape[0] != expected:
            raise ValueError(
                f"jitnmv: shape mismatch, got matrix shape {shape} and vector shape {vector.shape}."
            )
        out_size = shape[1] if transpose else shape[0]
        return u.maybe_decimal(jnp.zeros((out_size,), dtype=out_dtype) * unitd * unitv)
    clen = _initialize_conn_length(prob)
    res = jitnmv_p_call(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose", "corder", "matrix_mode"))
def jitnmm(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    matrix_mode: MatrixMode = 'mm',
    backend: Optional[str] = None,
) -> Data:
    """JIT normally-distributed matrix-matrix product.

    Computes ``W @ B`` (or ``W.T @ B``) where ``W`` is a random matrix
    with entries drawn from ``Normal(w_loc, w_scale)`` masked by
    Bernoulli(*prob*), without materialising ``W``.

    Parameters
    ----------
    w_loc : scalar or Quantity
        Mean of the normal weight distribution.
    w_scale : scalar or Quantity
        Standard deviation.  Must share units with *w_loc*.
    prob : float
        Connection probability in ``[0, 1]``.
    B : jax.Array or Quantity
        Right-hand matrix of shape ``(k, n)`` where ``k`` equals
        ``shape[0]`` when *transpose* is ``True``, or ``shape[1]``
        otherwise.
    seed : int or None, optional
        RNG seed.  ``None`` generates a random seed.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.
    transpose : bool, optional
        If ``True``, multiply by the transpose.  Default is ``False``.
    corder : bool, optional
        Column-major iteration order.  Default is ``True``.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    jax.Array or Quantity
        Output matrix of shape ``(shape[1], n)`` when *transpose* is
        ``True``, or ``(shape[0], n)`` otherwise.

    Raises
    ------
    ValueError
        If ``prob`` is not a scalar, is not finite, or is outside ``[0, 1]``.
    ValueError
        If *B* is not 2-D or its leading dimension does not match the
        matrix shape.

    See Also
    --------
    jitn : Materialise the full matrix.
    jitnmv : Matrix-vector multiply variant.

    Notes
    -----
    The connectivity matrix ``W`` of shape ``(m, n)`` follows the model:

        ``W[i, j] = N(w_loc, w_scale) * B_mask[i, j]``

    where ``N(w_loc, w_scale)`` is a normal draw and
    ``B_mask[i, j] ~ Bernoulli(prob)`` is a binary mask, both determined
    by ``seed``.

    The matrix-matrix product computes:

        ``Y[i, c] = sum_{j=0}^{n-1} W[i, j] * B[j, c]``

    When ``transpose=True``, the operation becomes ``Y = W^T @ B``:

        ``Y[j, c] = sum_{i=0}^{m-1} W[i, j] * B[i, c]``

    The matrix ``W`` is never materialised; weights are generated and
    consumed on the fly.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_normal.float import jitnmm
        >>> B = jnp.ones((50, 10))
        >>> Y = jitnmm(0.0, 1.0, 0.1, B, seed=42, shape=(100, 50))
        >>> Y.shape
        (100, 10)
    """
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    seed = _initialize_seed(seed)
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    B, unitB = u.split_mantissa_unit(B)
    out_dtype = jnp.asarray(w_loc).dtype
    if _is_static_zero_prob(prob, op_name="jitnmm"):
        if B.ndim != 2:
            raise ValueError(f"jitnmm: B must be 2D, but got {B.ndim}D.")
        expected = shape[0] if transpose else shape[1]
        if B.shape[0] != expected:
            raise ValueError(
                f"jitnmm: shape mismatch, got matrix shape {shape} and B shape {B.shape}."
            )
        out_shape = (shape[1], B.shape[1]) if transpose else (shape[0], B.shape[1])
        return u.maybe_decimal(jnp.zeros(out_shape, dtype=out_dtype) * unitd * unitB)
    clen = _initialize_conn_length(prob)
    res = jitnmm_p_call(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=matrix_mode,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitn_numba_kernel_generator(
    corder: bool = True,
    matrix_mode: MatrixMode = 'mv',
    transpose: bool = False,
    **kwargs
):
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_normal01 = _rng['normal01']

    stride = _MV_STRIDE if _normalize_matrix_mode(matrix_mode) == 'mv' else _MM_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    if corder:
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, seed, posts):
            posts[:] = 0.
            n_rows = posts.shape[0]
            n_cols = posts.shape[1]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (n_cols + chunk_size - 1) // chunk_size
            for row in range(n_rows):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= n_cols:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > n_cols:
                        chunk_end = n_cols
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            col = chunk_start + local_j
                            n01 = _rng_normal01(seed0, row, col)
                            posts[row, col] = w_loc0 + n01 * w_scale0
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    else:
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, seed, posts):
            posts[:] = 0.
            n_rows = posts.shape[1]
            n_cols = posts.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (n_cols + chunk_size - 1) // chunk_size
            for row in range(n_rows):
                for chunk_id in range(n_chunks):
                    chunk_start = chunk_id * chunk_size
                    if chunk_start >= n_cols:
                        break
                    chunk_end = chunk_start + chunk_size
                    if chunk_end > n_cols:
                        chunk_end = n_cols
                    chunk_width = chunk_end - chunk_start
                    for lane in range(stride):
                        state = _rng_init(seed0, row, chunk_id, lane)
                        q, state = _rng_initial_q(state, cl)
                        local_j = lane + stride * int(q)
                        while local_j < chunk_width:
                            col = chunk_start + local_j
                            n01 = _rng_normal01(seed0, row, col)
                            posts[col, row] = w_loc0 + n01 * w_scale0
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def run(w_loc, w_scale, clen, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, seed)

    return run


def _jitn_cuda_kernel(
    corder: bool = True,
    matrix_mode: MatrixMode = 'mv',
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jitn.cu'),
        name='jit_normal_jitn',
    )
    sfx = _dtype_sfx.get(np.dtype(kwargs['w_loc_info'].dtype), '_f32')
    mode = 'mv' if _normalize_matrix_mode(matrix_mode) == 'mv' else 'mm_aw_t4'
    direction = 'notrans' if corder else 'trans'
    kernel_name = f'jit_normal_jitn.jitn_{mode}_{direction}{sfx}'
    out_shape = tuple(int(s) for s in kwargs['out_info'].shape)
    n_rows, n_cols = out_shape if corder else out_shape[::-1]
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    def kernel(w_loc, w_scale, clen, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w_loc, w_scale, clen, seed,
            n_rows=np.int32(n_rows), n_cols=np.int32(n_cols),
            chunk_size=np.int32(chunk_size),
        )

    return kernel


def _jitn_jvp_wlow(
    w_loc_dot, w_loc, w_scale, clen, seed, *,
    shape, transpose: bool, corder: bool, backend, **kwargs
):
    return jitn_p_call(
        w_loc_dot, 0., clen, seed, shape=shape, transpose=transpose, corder=corder,
        matrix_mode=kwargs['matrix_mode'], backend=backend,
    )


def _jitn_jvp_whigh(
    w_scale_dot, w_loc, w_scale, clen, seed, *,
    shape, transpose: bool, corder: bool, backend, **kwargs
):
    return jitn_p_call(
        0., w_scale_dot, clen, seed, shape=shape, transpose=transpose, corder=corder,
        matrix_mode=kwargs['matrix_mode'], backend=backend,
    )


def _jitn_transpose(
    ct, w_loc, w_scale, clen, seed, *,
    shape, transpose: bool, corder: bool, backend, **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(w_loc):
        forward = jitn_p_call(
            1., 0., clen, seed, shape=shape, transpose=transpose, corder=corder,
            matrix_mode=kwargs['matrix_mode'], backend=backend,
        )[0]
        dwlow = jnp.expand_dims((ct * forward).sum(), axis=0)
        return (dwlow, w_scale, clen, seed)

    elif ad.is_undefined_primal(w_scale):
        forward = jitn_p_call(
            0., 1., clen, seed, shape=shape, transpose=transpose, corder=corder,
            matrix_mode=kwargs['matrix_mode'], backend=backend,
        )[0]
        dwscale = jnp.expand_dims((ct * forward).sum(), axis=0)
        return (w_loc, dwscale, clen, seed)

    else:
        raise NotImplementedError(
            'JITC matrix transpose is only implemented for the w_low and w_scale arguments.'
        )


def _jitn_batching(args, axes, **kwargs):
    return general_batching_rule(jitn_p, args, axes, **kwargs)


def _jitn_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_loc = jnp.ones(1, dtype=dtype)
            w_scale = jnp.ones(1, dtype=dtype) * 0.1
            clen = jnp.atleast_1d(jnp.asarray(np.ceil(2.0 / prob), dtype=jnp.int32))
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (w_loc, w_scale, clen, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder,
                'matrix_mode': 'mv'
            }))
    return configs


def jitn_p_call(
    w_loc, w_scale, clen, seed, *, shape, transpose: bool, corder: bool,
    matrix_mode: MatrixMode, backend
):
    """Dispatch the JIT normal matrix materialisation primitive.

    Parameters
    ----------
    w_loc : jax.Array
        Weight mean, shape ``(1,)``.
    w_scale : jax.Array
        Weight standard deviation, shape ``(1,)``.
    clen : jax.Array
        Connection length derived from *prob*, shape ``(1,)``.
    seed : jax.Array
        RNG seed, shape ``(1,)``.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.
    transpose : bool
        Whether to transpose the output.
    corder : bool
        Column-major iteration order.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    tuple of jax.Array
        Single-element tuple containing the materialised matrix.

    See Also
    --------
    jitn : High-level wrapper with unit support.

    Notes
    -----
    Scalar inputs (``w_loc``, ``w_scale``, ``clen``, ``seed``) are
    automatically promoted to 1-D arrays of shape ``(1,)``.

    The generated matrix has entry model:

        ``W[i, j] = Normal(w_loc[0], w_scale[0]) * Bernoulli(prob)``

    where ``prob`` is implicitly encoded through ``clen = 2 / prob``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_normal.float import jitn_p_call
        >>> out = jitn_p_call(
        ...     jnp.array([0.0]), jnp.array([1.0]),
        ...     jnp.array([20.0]), jnp.array([42]),
        ...     shape=(10, 5), transpose=False, corder=True)
    """
    seed = _initialize_seed(seed)
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)

    out_info = (
        jax.ShapeDtypeStruct(shape[::-1], dtype=w_loc.dtype)
        if transpose else
        jax.ShapeDtypeStruct(shape, dtype=w_loc.dtype)
    )

    return jitn_p(
        w_loc,
        w_scale,
        clen,
        seed,
        outs=[out_info],
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=_normalize_matrix_mode(matrix_mode),
        backend=backend,
    )


jitn_p = XLACustomKernel(
    'float_jitn',
    doc="""
Low-level XLA custom-kernel primitive for ``jitn``.

This ``XLACustomKernel`` instance dispatches the JIT normal connectivity matrix generation
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

This operation generates a sparse connectivity matrix where weights are normally distributed
with specified mean and standard deviation. The connectivity pattern is generated on-the-fly
using a deterministic PRNG seeded by the provided seed value.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``jitn_p.available_backends(platform)``,
and the default backend can be configured with ``jitn_p.set_default(platform, backend)``.

See Also
--------
jitn : High-level user-facing function wrapper.
"""
)
jitn_p.def_numba_kernel(_jitn_numba_kernel_generator)
jitn_p.def_cuda_raw_kernel(_jitn_cuda_kernel, asdefault=True)
jitn_p.def_jvp_rule2(_jitn_jvp_wlow, _jitn_jvp_whigh, None, None)
jitn_p.def_transpose_rule(_jitn_transpose)
jitn_p.def_batching_rule(_jitn_batching)
jitn_p.def_call(jitn_p_call)
jitn_p.def_tags('jit_normal', 'float')
jitn_p.def_benchmark_data(_jitn_benchmark_data)


def _jitnmv_numba_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_normal01 = _rng['normal01']

    stride = _MV_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    if corder:
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, vector, seed, posts):
            m = posts.shape[0]
            k = vector.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                out = np.asarray(0., dtype=vector.dtype)
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
                            col = chunk_start + local_j
                            n01 = _rng_normal01(seed0, row, col)
                            out += vector[col] * (w_loc0 + n01 * w_scale0)
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = out

    else:
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, vector, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = vector.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                v = vector[row]
                if v == 0.:
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
                            col = chunk_start + local_j
                            n01 = _rng_normal01(seed0, row, col)
                            posts[col] += v * (w_loc0 + n01 * w_scale0)
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def run(w_loc, w_scale, clen, vector, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, vector, seed)

    return run


def _jitnmv_cuda_kernel(
    corder: bool = True,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jitnmv.cu'),
        name='jit_normal_jitnmv',
    )
    sfx = _dtype_sfx.get(np.dtype(kwargs['w_loc_info'].dtype), '_f32')
    variant = 'notrans' if corder else 'trans'
    kernel_name = f'jit_normal_jitnmv.jitnmv_{variant}{sfx}'
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    def kernel(w_loc, w_scale, clen, vector, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w_loc, w_scale, clen, seed, vector, chunk_size=np.int32(chunk_size))

    return kernel


def _jitnmv_jvp_v(v_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, backend, **kwargs):
    return jitnmv_p_call(
        w_loc, w_scale, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=backend
    )


def _jitnmv_jvp_wloc(w_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, backend, **kwargs):
    return jitnmv_p_call(
        w_dot, w_scale, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=backend
    )


def _jitnmv_jvp_wscale(w_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, backend, **kwargs):
    return jitnmv_p_call(
        w_loc, w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=backend
    )


def _jitnmv_transpose_rules(
    ct, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, backend, **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        # Gradient w.r.t. vector: d(loss)/d(v) = M^T @ ct
        r = jitnmv_p_call(
            w_loc,
            w_scale,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            backend=backend,
        )[0]
        return w_loc, w_scale, clen, r, seed
    elif ad.is_undefined_primal(w_loc):
        # M = (w_loc + w_scale * Z) * mask
        # d(M @ v)/d(w_loc) = mask @ v
        # d(loss)/d(w_loc) = ct^T @ (mask @ v) = (mask^T @ ct) . v
        # mask^T @ ct = jitnmv(1., 0., ...) with transposed shape
        r = jitnmv_p_call(
            1., 0., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=backend,
        )[0]
        dw_loc = jnp.expand_dims(jnp.sum(r * vector), axis=0)
        return dw_loc, w_scale, clen, vector, seed
    elif ad.is_undefined_primal(w_scale):
        # M = (w_loc + w_scale * Z) * mask
        # d(M @ v)/d(w_scale) = (Z * mask) @ v
        # d(loss)/d(w_scale) = ct^T @ ((Z * mask) @ v) = ((Z * mask)^T @ ct) . v
        # (Z * mask)^T @ ct = jitnmv(0., 1., ...) with transposed shape
        r = jitnmv_p_call(
            0., 1., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=backend,
        )[0]
        dw_scale = jnp.expand_dims(jnp.sum(r * vector), axis=0)
        return w_loc, dw_scale, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for jitnmv not implemented "
            f"when none of vector/w_loc/w_scale is an undefined primal."
        )


def _jitnmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitnmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            matrix_mode='mm',
            backend=kwargs.get('backend'),
        )
        return r, [1]
    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitnmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            matrix_mode='mm',
            backend=kwargs.get('backend'),
        )
        return r, [1]
    else:
        return general_batching_rule(
            jitnmv_p,
            args,
            axes,
            **kwargs,
        )


def _jitnmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_loc = jnp.ones(1, dtype=dtype)
            w_scale = jnp.ones(1, dtype=dtype) * 0.1
            clen = jnp.atleast_1d(jnp.asarray(np.ceil(2.0 / prob), dtype=jnp.int32))
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w_loc, w_scale, clen, vector, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                )
            )
    return configs


def jitnmv_p_call(
    w_loc,
    w_scale,
    clen,
    vector,
    seed,
    *,
    shape,
    transpose: bool,
    corder: bool,
    backend,
):
    """Dispatch the JIT normal matrix-vector multiply primitive.

    Parameters
    ----------
    w_loc : jax.Array
        Weight mean, shape ``(1,)``.
    w_scale : jax.Array
        Weight standard deviation, shape ``(1,)``.
    clen : jax.Array
        Connection length, shape ``(1,)``.
    vector : jax.Array
        Input 1-D vector.
    seed : jax.Array
        RNG seed, shape ``(1,)``.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.
    transpose : bool
        Whether to use the transposed matrix.
    corder : bool
        Column-major iteration order.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    tuple of jax.Array
        Single-element tuple containing the output vector.

    Raises
    ------
    AssertionError
        If ``shape`` is not length 2, or if ``w_loc``, ``w_scale``,
        ``clen``, ``seed`` do not have shape ``(1,)``, or if ``vector``
        is not 1-D, or if the matrix shape and vector length are
        incompatible.

    See Also
    --------
    jitnmv : High-level wrapper with unit support.

    Notes
    -----
    The product is computed without materialising the full matrix:

        ``y[i] = sum_{j} Normal(w_loc, w_scale) * Bernoulli(prob) * v[j]``

    Each weight is generated on the fly using a deterministic PRNG seeded
    by ``seed``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_normal.float import jitnmv_p_call
        >>> out = jitnmv_p_call(
        ...     jnp.array([0.0]), jnp.array([1.0]),
        ...     jnp.array([20.0]), jnp.ones(5), jnp.array([42]),
        ...     shape=(10, 5), transpose=False, corder=True)
    """
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert w_loc.shape == (1,), f"The weight shape should be (1,), but got {w_loc.shape}."
    assert w_scale.shape == (1,), f"The weight shape should be (1,), but got {w_scale.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_loc.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_loc.dtype)
    )

    return jitnmv_p(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        outs=[out_info],
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


jitnmv_p = XLACustomKernel(
    'float_jitnmv',
    doc="""
Low-level XLA custom-kernel primitive for ``jitnmv``.

This ``XLACustomKernel`` instance dispatches the JIT normal connectivity matrix-vector
multiplication with floating-point weights operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided by
the high-level wrapper.

In this operation, the connectivity matrix has weights normally distributed with specified
mean and standard deviation, and the input vector contains floating-point values. The
operation computes a standard matrix-vector product without event-driven sparsity.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``jitnmv_p.available_backends(platform)``,
and the default backend can be configured with ``jitnmv_p.set_default(platform, backend)``.

See Also
--------
jitnmv : High-level user-facing function wrapper.
"""
)
jitnmv_p.def_numba_kernel(_jitnmv_numba_kernel_generator)
jitnmv_p.def_cuda_raw_kernel(_jitnmv_cuda_kernel, asdefault=True)
jitnmv_p.def_jvp_rule2(_jitnmv_jvp_wloc, _jitnmv_jvp_wscale, None, _jitnmv_jvp_v, None)
jitnmv_p.def_transpose_rule(_jitnmv_transpose_rules)
jitnmv_p.def_batching_rule(_jitnmv_batching)
jitnmv_p.def_call(jitnmv_p_call)
jitnmv_p.def_tags('jit_normal', 'float')
jitnmv_p.def_benchmark_data(_jitnmv_benchmark_data)


def _jitnmm_numba_kernel_generator(
    corder: bool = True,
    matrix_mode: MatrixMode = 'mm',
    **kwargs
):
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']
    _rng_normal01 = _rng['normal01']

    stride = _MV_STRIDE if _normalize_matrix_mode(matrix_mode) == 'mv' else _MM_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    if corder:
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, B, seed, posts):
            m = posts.shape[0]
            n = posts.shape[1]
            k = B.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                out = np.zeros(n, dtype=B.dtype)
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
                            col = chunk_start + local_j
                            n01 = _rng_normal01(seed0, row, col)
                            out += B[col] * (w_loc0 + n01 * w_scale0)
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = out

    else:
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, B, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = B.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                out = B[row]
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
                            col = chunk_start + local_j
                            n01 = _rng_normal01(seed0, row, col)
                            posts[col] += out * (w_loc0 + n01 * w_scale0)
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def run(w_loc, w_scale, clen, B, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, B, seed)

    return run


def _jitnmm_cuda_kernel(
    corder: bool = True,
    matrix_mode: MatrixMode = 'mm',
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jitnmm.cu'),
        name='jit_normal_jitnmm',
    )
    sfx = _dtype_sfx.get(np.dtype(kwargs['w_loc_info'].dtype), '_f32')
    prefix = 'jitnmm_mv' if _normalize_matrix_mode(matrix_mode) == 'mv' else 'jitnmm'
    variant = 'notrans' if corder else 'trans'
    kernel_name = f'jit_normal_jitnmm.{prefix}_{variant}{sfx}'

    out_info = kwargs['out_info']
    B_info = kwargs['B_info']
    k_walk = int(B_info.shape[0])
    n = int(B_info.shape[1])
    if corder:
        m_ffi, k_ffi = int(out_info.shape[0]), k_walk
    else:
        m_ffi, k_ffi = k_walk, int(out_info.shape[0])
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    def kernel(w_loc, w_scale, clen, B, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            w_loc, w_scale, clen, seed, B,
            m=np.int32(m_ffi), k=np.int32(k_ffi), n=np.int32(n),
            chunk_size=np.int32(chunk_size),
        )

    return kernel


def _jitnmm_jvp_wloc(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend, **kwargs):
    return jitnmm_p_call(
        w_dot, w_scale, clen, B, seed, shape=shape, transpose=transpose, corder=corder,
        matrix_mode=kwargs['matrix_mode'], backend=backend,
    )


def _jitnmm_jvp_wscale(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend, **kwargs):
    return jitnmm_p_call(
        w_loc, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder,
        matrix_mode=kwargs['matrix_mode'], backend=backend,
    )


def _jitnmm_jvp_B(B_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend, **kwargs):
    return jitnmm_p_call(
        w_loc, w_scale, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder,
        matrix_mode=kwargs['matrix_mode'], backend=backend,
    )


def _jitnmm_transpose_rules(ct, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend, **kwargs):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        r = jitnmm_p_call(
            w_loc,
            w_scale,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            matrix_mode=kwargs['matrix_mode'],
            backend=backend,
        )[0]
        return w_loc, w_scale, clen, r, seed
    elif ad.is_undefined_primal(w_loc):
        # M = (w_loc + w_scale * Z) * mask
        # d(M @ B)/d(w_loc) = mask @ B
        # d(loss)/d(w_loc) = sum((mask^T @ ct) * B)
        r = jitnmm_p_call(
            1., 0., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            matrix_mode=kwargs['matrix_mode'],
            backend=backend,
        )[0]
        dw_loc = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return dw_loc, w_scale, clen, B, seed
    elif ad.is_undefined_primal(w_scale):
        # M = (w_loc + w_scale * Z) * mask
        # d(M @ B)/d(w_scale) = (Z * mask) @ B
        # d(loss)/d(w_scale) = sum(((Z*mask)^T @ ct) * B)
        r = jitnmm_p_call(
            0., 1., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            matrix_mode=kwargs['matrix_mode'],
            backend=backend,
        )[0]
        dw_scale = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return w_loc, dw_scale, clen, B, seed
    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_normal not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[3].shape
    B = args[3].reshape(m, maybe_batch1 * maybe_batch2)
    r = jitnmm_p_call(
        args[0],
        args[1],
        args[2],
        B,
        args[4],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
        matrix_mode=kwargs['matrix_mode'],
        backend=kwargs.get('backend'),
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitnmm_batching(args, axes, **kwargs):
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
        return general_batching_rule(jitnmm_p, args, axes, **kwargs)


def _jitnmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            w_loc = jnp.ones(1, dtype=dtype)
            w_scale = jnp.ones(1, dtype=dtype) * 0.1
            clen = jnp.atleast_1d(jnp.asarray(np.ceil(2.0 / prob), dtype=jnp.int32))
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w_loc, w_scale, clen, B, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder,
                     'matrix_mode': 'mm'}
                )
            )
    return configs


def jitnmm_p_call(
    w_loc, w_scale, clen, B, seed, *,
    shape: MatrixShape, transpose: bool, corder: bool,
    matrix_mode: MatrixMode = 'mm',
    backend: Optional[str] = None,
):
    """Dispatch the JIT normal matrix-matrix multiply primitive.

    Parameters
    ----------
    w_loc : jax.Array
        Weight mean, shape ``(1,)``.
    w_scale : jax.Array
        Weight standard deviation, shape ``(1,)``.
    clen : jax.Array
        Connection length, shape ``(1,)``.
    B : jax.Array
        Right-hand 2-D matrix.
    seed : jax.Array
        RNG seed, shape ``(1,)``.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.
    transpose : bool
        Whether to use the transposed matrix.
    corder : bool
        Column-major iteration order.
    backend : str or None, optional
        Compute backend.

    Returns
    -------
    tuple of jax.Array
        Single-element tuple containing the output matrix.

    Raises
    ------
    AssertionError
        If ``shape`` is not length 2, or if ``B`` is not 2-D, or if
        ``w_loc``, ``w_scale``, ``clen``, ``seed`` do not have the
        expected shapes, or if the matrix dimensions are incompatible.

    See Also
    --------
    jitnmm : High-level wrapper with unit support.

    Notes
    -----
    The product is computed without materialising the full matrix:

        ``Y[i, c] = sum_{j} Normal(w_loc, w_scale) * Bernoulli(prob) * B[j, c]``

    Each weight is generated on the fly using a deterministic PRNG seeded
    by ``seed``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_normal.float import jitnmm_p_call
        >>> out = jitnmm_p_call(
        ...     jnp.array([0.0]), jnp.array([1.0]),
        ...     jnp.array([20.0]), jnp.ones((5, 3)), jnp.array([42]),
        ...     shape=(10, 5), transpose=False, corder=True)
    """
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert w_loc.ndim == 1, "The weight should be a 1D array."
    assert w_scale.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert w_loc.shape == (1,), "The weight should be a scalar."
    assert w_scale.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_loc.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_loc.dtype)
    )

    return jitnmm_p(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        outs=[out_info],
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=_normalize_matrix_mode(matrix_mode),
        backend=backend,
    )


jitnmm_p = XLACustomKernel(
    'float_jitnmm',
    doc="""
Low-level XLA custom-kernel primitive for ``jitnmm``.

This ``XLACustomKernel`` instance dispatches the JIT normal connectivity matrix-matrix
multiplication with floating-point weights operation to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided by
the high-level wrapper.

In this operation, the connectivity matrix has weights normally distributed with specified
mean and standard deviation, and the input matrix contains floating-point values. Each
column of the input is processed independently in a standard matrix-matrix product.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``jitnmm_p.available_backends(platform)``,
and the default backend can be configured with ``jitnmm_p.set_default(platform, backend)``.

See Also
--------
jitnmm : High-level user-facing function wrapper.
"""
)
jitnmm_p.def_numba_kernel(_jitnmm_numba_kernel_generator)
jitnmm_p.def_cuda_raw_kernel(_jitnmm_cuda_kernel, asdefault=True)
jitnmm_p.def_jvp_rule2(_jitnmm_jvp_wloc, _jitnmm_jvp_wscale, None, _jitnmm_jvp_B, None)
jitnmm_p.def_transpose_rule(_jitnmm_transpose_rules)
jitnmm_p.def_batching_rule(_jitnmm_batching)
jitnmm_p.def_call(jitnmm_p_call)
jitnmm_p.def_tags('jit_normal', 'float')
jitnmm_p.def_benchmark_data(_jitnmm_benchmark_data)
