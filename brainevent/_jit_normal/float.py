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
from typing import Optional

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._compatible_import import Tracer
from brainevent._data import _initialize_seed, _initialize_conn_length
from brainevent._misc import generate_block_dim, namescope
from brainevent._numba_random import get_numba_lfsr_seed, get_numba_lfsr_random_integers, get_numba_lfsr_normal
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig, register_tvm_cuda_from_file
from brainevent._pallas_random import get_pallas_lfsr_rng_class
from brainevent._typing import Data, MatrixShape

__all__ = [
    "jitn",
    "jitn_p",
    "jitnmv",
    "jitnmv_p",
    "jitnmm",
    "jitnmm_p",
]


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


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitn(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
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


@namescope(static_argnames=("shape", "transpose", "corder"))
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
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitn_numba_kernel_generator(corder: bool = True, **kwargs):
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_normal = get_numba_lfsr_normal()

    if corder:
        # JIT matrix
        # - JIT matrix shape = [m, n]
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]
            n = posts.shape[1]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for i_row in range(m):
                state = _lfsr_seed(seed0 + i_row * n)
                i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_col < n:
                    posts[i_row, i_col] = _lfsr_normal(state, w_loc0, w_scale0)
                    i_col += _lfsr_random_integers(state, 1, clen0 - 1)

    else:
        # JIT matrix
        # - JIT matrix shape = [m, n]
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]
            n = posts.shape[1]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_col in range(n):
                state = _lfsr_seed(seed0 + i_col * m)
                i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_row < m:
                    posts[i_row, i_col] = _lfsr_normal(state, w_loc0, w_scale0)
                    i_row += _lfsr_random_integers(state, 1, clen0 - 1)

    def run(w_loc, w_scale, clen, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, seed)

    return run


def _jitn_pallas_kernel_generator(
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl
    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    dim = out_info.shape[0] if corder else out_info.shape[1]
    block_size = generate_block_dim(dim, maximum=128)
    n_block = pl.cdiv(dim, block_size)

    if corder:
        def kernel(w_loc_ref, w_scale_ref, clen_ref, seed_ref, _, post_ref):
            m = post_ref.shape[1]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim

            def body(_step, data):
                i_cols, i_col_mask, rng = data
                val = rng.normal(w_loc, w_scale)
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                post_ref[i_rows, safe_cols] = jnp.where(i_row_mask & i_col_mask, val, post_ref[i_rows, safe_cols])
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < m, rng

            rng = _PallasLFSRRNG(seed0 + i_rows * m)
            i_cols = rng.random_integers(0, clen0)
            i_col_mask = i_cols < m
            jax.lax.fori_loop(0, m, body, (i_cols, i_col_mask, rng))

    else:
        def kernel(w_loc_ref, w_scale_ref, clen_ref, seed_ref, _, post_ref):
            n = post_ref.shape[0]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(_step, data):
                i_rows, i_row_mask, rng = data
                val = rng.normal(w_loc, w_scale)
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                post_ref[safe_rows, i_cols] = jnp.where(i_row_mask & i_col_mask, val, post_ref[safe_rows, i_cols])
                i_rows = i_rows + rng.random_integers(1, clen0)
                return i_rows, i_rows < n, rng

            rng = _PallasLFSRRNG(seed0 + i_cols * n)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < n
            jax.lax.fori_loop(0, n, body, (i_rows, i_row_mask, rng))

    def run(w_loc, w_scale, clen, seed):
        fn = pl.pallas_call(kernel, grid=(n_block,), input_output_aliases={4: 0}, out_shape=kwargs['outs'],
                            backend='triton')
        out = kwargs['outs'][0]
        placeholder = jnp.zeros(out.shape, out.dtype)
        return fn(w_loc, w_scale, clen, seed, placeholder)

    return run


def _jitn_jvp_wlow(w_loc_dot, w_loc, w_scale, clen, seed, *, out_info, **kwargs):
    out = jnp.ones_like(out_info) * w_loc_dot
    return [out]


def _jitn_jvp_whigh(
    w_scale_dot, w_loc, w_scale, clen, seed, *,
    shape, transpose: bool, corder: bool, backend, **kwargs
):
    return jitn_p_call(
        0., w_scale_dot, clen, seed, shape=shape, transpose=transpose, corder=corder, backend=backend
    )


def _jitn_transpose(
    ct, w_loc, w_scale, clen, seed, *,
    shape, transpose: bool, corder: bool, backend, **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(w_loc):
        dwlow = jnp.expand_dims(ct.sum(), axis=0)
        return (dwlow, w_scale, clen, seed)

    elif ad.is_undefined_primal(w_scale):
        forward = jitn_p_call(
            0., 1., clen, seed, shape=shape, transpose=transpose, corder=corder, backend=backend
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
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (w_loc, w_scale, clen, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jitn_p_call(
    w_loc, w_scale, clen, seed, *, shape, transpose: bool, corder: bool, backend
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
        backend=backend,
    )


_dtype_sfx = {
    np.dtype('float16'): '_f16',
    np.dtype('float32'): '_f32',
    np.dtype('float64'): '_f64',
    np.dtype('bfloat16'): '_bf16',
}


def _jitn_cuda_kernel(
    corder: bool = True,
    **kwargs
):
    register_tvm_cuda_from_file(module='jit_normal', source=Path(__file__).parent.joinpath('jit_normal.cu'))
    sfx = _dtype_sfx.get(np.dtype(kwargs['w_loc_info'].dtype), '_f32')
    variant = 'corder_true' if corder else 'corder_false'
    kernel_name = f'jit_normal.jitn_{variant}{sfx}'

    def kernel(w_loc, w_scale, clen, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(w_loc, w_scale, clen, seed)

    return kernel


def _jitnmv_cuda_kernel(
    corder: bool = True,
    **kwargs
):
    register_tvm_cuda_from_file(module='jit_normal', source=Path(__file__).parent.joinpath('jit_normal.cu'))
    sfx = _dtype_sfx.get(np.dtype(kwargs['w_loc_info'].dtype), '_f32')
    variant = 'gather' if corder else 'scatter'
    kernel_name = f'jit_normal.jitnmv_{variant}{sfx}'

    def kernel(w_loc, w_scale, clen, vector, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(w_loc, w_scale, clen, seed, vector)

    return kernel


def _jitnmm_cuda_kernel(
    corder: bool = True,
    **kwargs
):
    register_tvm_cuda_from_file(module='jit_normal', source=Path(__file__).parent.joinpath('jit_normal.cu'))
    sfx = _dtype_sfx.get(np.dtype(kwargs['w_loc_info'].dtype), '_f32')
    variant = 'gather' if corder else 'scatter'
    kernel_name = f'jit_normal.jitnmm_{variant}{sfx}'

    def kernel(w_loc, w_scale, clen, B, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(w_loc, w_scale, clen, seed, B)

    return kernel


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
jitn_p.def_pallas_kernel('gpu', _jitn_pallas_kernel_generator)
jitn_p.def_tvmffi_kernel('gpu', _jitn_cuda_kernel)
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
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_normal = get_numba_lfsr_normal()

    if corder:
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, vector, seed, posts):
            posts[:] = 0.
            num_row = posts.shape[0]
            num_col = vector.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for i_row in range(num_row):
                state = _lfsr_seed(seed0 + i_row * num_col)
                i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                out = np.asarray(0., dtype=vector.dtype)
                while i_col < num_col:
                    out += vector[i_col] * _lfsr_normal(state, w_loc0, w_scale0)
                    i_col += _lfsr_random_integers(state, 1, clen0 - 1)
                posts[i_row] = out

    else:
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, vector, seed, posts):
            posts[:] = 0.
            num_row = posts.shape[0]
            num_col = vector.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_col in range(num_col):
                state = _lfsr_seed(seed0 + i_col * num_row)
                v = vector[i_col]
                i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_row < num_row:
                    posts[i_row] += v * _lfsr_normal(state, w_loc0, w_scale0)
                    i_row += _lfsr_random_integers(state, 1, clen0 - 1)

    def run(w_loc, w_scale, clen, vector, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, vector, seed)

    return run


def _jitnmv_pallas_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]
    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    block_size = generate_block_dim(dim, maximum=128)

    if corder:
        def kernel(w_loc_ref, w_scale_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_row = vector_ref.shape[0]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(_step, data):
                i_rows, i_row_mask, rng, out = data
                v = jnp.where(i_row_mask, vector_ref[i_rows], 0.)
                out += v * rng.normal(w_loc, w_scale)
                i_rows += rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, out

            rng = _PallasLFSRRNG(seed + i_cols * num_row)
            i_rows = rng.random_integers(0, clen)
            i_row_mask = i_rows < num_row
            out = jnp.zeros(block_size, dtype=post_ref.dtype)
            out = jax.lax.fori_loop(0, num_row, body, (i_rows, i_row_mask, rng, out))[-1]
            post_ref[i_cols] = jnp.where(i_col_mask, out, post_ref[i_cols])

    else:
        def kernel(w_loc_ref, w_scale_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim
            vector = jnp.where(i_row_mask, vector_ref[i_rows], 0.)

            def body(_step, data):
                i_cols, i_col_mask, rng = data
                atomic_add(post_ref, (i_cols,), vector * rng.normal(w_loc, w_scale), mask=i_row_mask & i_col_mask)
                i_cols += rng.random_integers(1, clen)
                return i_cols, i_cols < num_col, rng

            rng = _PallasLFSRRNG(seed + i_rows * num_col)
            i_cols = rng.random_integers(0, clen)
            i_col_mask = i_cols < num_col
            jax.lax.fori_loop(0, num_col, body, (i_cols, i_col_mask, rng))

    def run(w_loc, w_scale, clen, vector, seed):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(dim, block_size),),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_loc, w_scale, clen, vector, seed, placeholder)

    return run


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
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
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
jitnmv_p.def_pallas_kernel('gpu', _jitnmv_pallas_kernel_generator)
jitnmv_p.def_tvmffi_kernel('gpu', _jitnmv_cuda_kernel)
jitnmv_p.def_jvp_rule2(_jitnmv_jvp_wloc, _jitnmv_jvp_wscale, None, _jitnmv_jvp_v, None)
jitnmv_p.def_transpose_rule(_jitnmv_transpose_rules)
jitnmv_p.def_batching_rule(_jitnmv_batching)
jitnmv_p.def_call(jitnmv_p_call)
jitnmv_p.def_tags('jit_normal', 'float')
jitnmv_p.def_benchmark_data(_jitnmv_benchmark_data)


def _jitnmm_numba_kernel_generator(
    corder: bool = True,
    **kwargs
):
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_normal = get_numba_lfsr_normal()

    if corder:
        # JIT Matrix @ B
        # - JIT matrix: [m, k]
        # - B: [k, n]
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, B, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]
            n = posts.shape[1]
            k = B.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for i_m in range(m):
                state = _lfsr_seed(seed0 + i_m * k)
                i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                out = np.zeros(n, dtype=B.dtype)
                while i_k < k:
                    out += B[i_k] * _lfsr_normal(state, w_loc0, w_scale0)
                    i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                posts[i_m] = out

    else:
        # JIT Matrix @ B
        # - JIT matrix: [m, k]
        # - B: [k, n]
        @numba.njit(fastmath=True)
        def kernel(w_loc, w_scale, clen, B, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]
            k = B.shape[0]
            w_loc0 = w_loc[0]
            w_scale0 = w_scale[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for i_k in range(k):
                state = _lfsr_seed(seed0 + i_k * m)
                out = B[i_k]
                i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                while i_m < m:
                    posts[i_m] += out * _lfsr_normal(state, w_loc0, w_scale0)
                    i_m += _lfsr_random_integers(state, 1, clen0 - 1)

    def run(w_loc, w_scale, clen, B, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, B, seed)

    return run


def _jitnmm_pallas_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]
    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    B_cols = B_info.shape[1]

    if corder:
        # Match jitn corder=True: vectorize over rows with identical RNG seeding.
        # Grid: (row_blocks, B_cols). Each kernel block processes one B column,
        # using vector RNG identical to jitn. Accumulates into 1D local array
        # to avoid 2D local arrays (dynamic_slice unsupported in Pallas Triton).
        out_rows = out_info.shape[0]
        row_block = generate_block_dim(out_rows, maximum=128)
        grid = (pl.cdiv(out_rows, row_block), B_cols)

        def kernel(w_loc_ref, w_scale_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            k = B_ref.shape[0]
            w_loc0 = w_loc_ref[0]
            w_scale0 = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            col_j = pl.program_id(1)  # scalar B column index

            # Row indices — VECTOR, matching jitn exactly
            i_rows = i_row_block * row_block + jnp.arange(row_block)
            i_row_mask = i_rows < out_rows
            safe_rows = jnp.where(i_row_mask, i_rows, 0)

            # Seed each (row, col_j) thread with an additional column stride.
            rng = _PallasLFSRRNG(seed0 + i_rows * k)
            i_cols = rng.random_integers(0, clen0)  # [row_block]
            i_col_mask = i_cols < k

            out = jnp.zeros(row_block, dtype=post_ref.dtype)

            def body(_step, data):
                i_cols, i_col_mask, rng, out = data
                w = rng.normal(w_loc0, w_scale0)  # [row_block]
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                b_vals = B_ref[safe_cols, col_j]  # [row_block] vector gather
                out += jnp.where(i_col_mask & i_row_mask, w * b_vals, 0.)
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < k, rng, out

            _, _, _, out = jax.lax.fori_loop(0, k, body, (i_cols, i_col_mask, rng, out))
            atomic_add(post_ref, (safe_rows, col_j), out, mask=i_row_mask)

    else:
        # Match jitn corder=False: vectorize over k-dim columns with identical
        # RNG seeding. Grid: (k_blocks, B_cols). Each block processes one B column.
        k_dim = B_info.shape[0]
        k_block = generate_block_dim(k_dim, maximum=128)
        grid = (pl.cdiv(k_dim, k_block), B_cols)

        def kernel(w_loc_ref, w_scale_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            m = post_ref.shape[0]
            w_loc0 = w_loc_ref[0]
            w_scale0 = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_k_block = pl.program_id(0)
            col_j = pl.program_id(1)  # scalar B column index

            i_ks = i_k_block * k_block + jnp.arange(k_block)
            i_k_mask = i_ks < k_dim
            safe_ks = jnp.where(i_k_mask, i_ks, 0)

            # Preload B values for this column (1D vector gather)
            b_vals = B_ref[safe_ks, col_j]  # [k_block]

            # Seed each (k, col_j) thread with an additional column stride.
            rng = _PallasLFSRRNG(seed0 + i_ks * m)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < m

            def body(_step, data):
                i_rows, i_row_mask, rng = data
                w = rng.normal(w_loc0, w_scale0)  # [k_block]
                vals = jnp.where(i_k_mask & i_row_mask, w * b_vals, 0.)
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                atomic_add(post_ref, (safe_rows, col_j), vals,
                           mask=i_k_mask & i_row_mask)
                i_rows += rng.random_integers(1, clen0)
                return i_rows, i_rows < m, rng

            jax.lax.fori_loop(0, m, body, (i_rows, i_row_mask, rng))

    def run(w_loc, w_scale, clen, B, seed):
        fn = pl.pallas_call(
            kernel,
            grid=grid,
            input_output_aliases={5: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_loc, w_scale, clen, B, seed, placeholder)

    return run


def _jitnmm_jvp_wloc(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend, **kwargs):
    return jitnmm_p_call(
        w_dot, w_scale, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=backend
    )


def _jitnmm_jvp_wscale(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend, **kwargs):
    return jitnmm_p_call(w_loc, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=backend)


def _jitnmm_jvp_B(B_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, backend, **kwargs):
    return jitnmm_p_call(
        w_loc, w_scale, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=backend
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
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (w_loc, w_scale, clen, B, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                )
            )
    return configs


def jitnmm_p_call(
    w_loc, w_scale, clen, B, seed, *,
    shape: MatrixShape, transpose: bool, corder: bool,
    backend
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
jitnmm_p.def_pallas_kernel('gpu', _jitnmm_pallas_kernel_generator)
jitnmm_p.def_tvmffi_kernel('gpu', _jitnmm_cuda_kernel)
jitnmm_p.def_jvp_rule2(_jitnmm_jvp_wloc, _jitnmm_jvp_wscale, None, _jitnmm_jvp_B, None)
jitnmm_p.def_transpose_rule(_jitnmm_transpose_rules)
jitnmm_p.def_batching_rule(_jitnmm_batching)
jitnmm_p.def_call(jitnmm_p_call)
jitnmm_p.def_tags('jit_normal', 'float')
jitnmm_p.def_benchmark_data(_jitnmm_benchmark_data)
