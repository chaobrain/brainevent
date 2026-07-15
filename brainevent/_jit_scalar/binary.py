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
from .float import jitsmv_p_call, jitsmm_p_call, _dtype_sfx

__all__ = [
    "binary_jitsmv",
    "binary_jitsmv_p",
    "binary_jitsmm",
    "binary_jitsmm_p",
]


def _warn_corder_ignored(corder: bool) -> None:
    warnings.warn(
        "corder is ignored by the light JIT scalar implementation.",
        UserWarning,
        stacklevel=3,
    )


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


def _light_options(kwargs):
    return {
        'chunk_size': kwargs.get('chunk_size', None),
        'target_chunks': kwargs.get('target_chunks', 4),
    }


@namescope(name="brainevent.binary_jitsmv", static_argnames=("shape", "transpose", "corder", "chunk_size", "target_chunks"))
def _binary_jitsmv_impl(
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
    """
    Event-driven matrix-vector product with a JIT scalar connectivity matrix.

    Computes the product of a just-in-time generated sparse matrix with
    scalarly distributed weights and a binary event vector. Only non-zero
    (event) entries in ``vector`` contribute to the output, making this
    operation efficient for spike-based neural network simulations.

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
    binary_jitsmm : Event-driven matrix-matrix variant.
    jitsmv : Float (non-event) matrix-vector variant.

    Notes
    -----
    The connectivity matrix ``A`` of shape ``(m, n)`` follows the model:

        ``A[i, j] = U[i, j] * B[i, j]``

    where ``U[i, j] ~ Scalar(w0, w1)`` and ``B[i, j] ~ Bernoulli(prob)``
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
        >>> from brainevent._jit_scalar.binary import binary_jitsmv
        >>> events = jnp.array([True, False, True, True, False])
        >>> result = binary_jitsmv(
        ...     0.1, 0.5, 0.2, events, seed=42,
        ...     shape=(3, 5), transpose=False, corder=True,
        ... )
        >>> result.shape
        (3,)
    """
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = binary_jitsmv_p_call(
        weight,
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


def binary_jitsmv(
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
    return _binary_jitsmv_impl(
        weight,
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


binary_jitsmv.__doc__ = _binary_jitsmv_impl.__doc__


@namescope(name="brainevent.binary_jitsmm", static_argnames=("shape", "transpose", "corder", "chunk_size", "target_chunks"))
def _binary_jitsmm_impl(
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
    """
    Event-driven matrix-matrix product with a JIT scalar connectivity matrix.

    Computes the product of a just-in-time generated sparse matrix with
    scalarly distributed weights and a binary event matrix ``B``. Each column
    of ``B`` is treated as an independent event vector, and only non-zero
    entries contribute to the output.

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
    binary_jitsmv : Event-driven matrix-vector variant.
    jitsmm : Float (non-event) matrix-matrix variant.

    Notes
    -----
    The connectivity matrix ``A`` of shape ``(m, n)`` follows the model:

        ``A[i, j] = U[i, j] * B_conn[i, j]``

    where ``U[i, j] ~ Scalar(w0, w1)`` and ``B_conn[i, j] ~ Bernoulli(prob)``
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
        >>> from brainevent._jit_scalar.binary import binary_jitsmm
        >>> B = jnp.array([[True, False], [False, True], [True, True],
        ...                [False, False], [True, False]])
        >>> result = binary_jitsmm(
        ...     0.1, 0.5, 0.2, B, seed=42,
        ...     shape=(3, 5), transpose=False, corder=True,
        ... )
        >>> result.shape
        (3, 2)
    """
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = binary_jitsmm_p_call(
        weight,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def binary_jitsmm(
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
    return _binary_jitsmm_impl(
        weight,
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


binary_jitsmm.__doc__ = _binary_jitsmm_impl.__doc__


# Kernel generators for JIT connection SPMV

def _jitsmv_numba_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for binary event JIT-scalar matrix-vector product.

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
        A function ``kernel(w0, w1, clen, vector, seed)`` that
        executes the Numba-compiled kernel and returns the result.
    """
    import numba

    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_uniform = get_numba_lfsr_uniform()

    if corder:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                w = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_col in range(n_col):
                    state = _lfsr_seed(seed0 + i_col * n_row)
                    i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.asarray(0., dtype=posts.dtype)
                    while i_row < n_row:
                        if vector[i_row]:
                            out += w
                        i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_col] = out
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                w = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_col in range(n_col):
                    state = _lfsr_seed(seed0 + i_col * n_row)
                    i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.asarray(0., dtype=posts.dtype)
                    while i_row < n_row:
                        if vector[i_row] > 0.:
                            out += w
                        i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_col] = out


    else:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                w = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_row in range(num_row):
                    if vector[i_row]:
                        state = _lfsr_seed(seed0 + i_row * num_col)
                        i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while i_col < num_col:
                            posts[i_col] += w
                            i_col += _lfsr_random_integers(state, 1, clen0 - 1)
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                w = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_row in range(num_row):
                    if vector[i_row] > 0.:
                        state = _lfsr_seed(seed0 + i_row * num_col)
                        i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while i_col < num_col:
                            posts[i_col] += w
                            i_col += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(weight, clen, vector, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, vector, seed)

    return kernel


_spike_sfx = {
    np.dtype('bool'): '_bool',
    np.dtype('int8'): '_bool',
    np.dtype('float32'): '_float',
    np.dtype('float16'): '_float',
    np.dtype('float64'): '_float',
    np.dtype('bfloat16'): '_float',
}


def _binary_jitsmv_cuda_kernel(
    corder: bool,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs
):
    del corder
    if np.dtype(kwargs['weight_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light binary_jitsmv currently supports float32 weights only")

    load_cuda_file(
        Path(__file__).parent.joinpath('binary_jitsmv.cu'),
        name='binary_jitsmv',
    )
    event_size = int(vector_info.shape[0])
    packed_words = (event_size + 31) // 32
    packed_info = jax.ShapeDtypeStruct((packed_words,), jnp.uint32)
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)
    compute_name = 'binary_jitsmv.scatter_f32' if transpose else 'binary_jitsmv.gather_f32'

    def kernel(weight, clen, vector, seed):
        active = vector if vector.dtype == jnp.bool_ else (vector > 0).astype(jnp.int8)
        packed = jax.ffi.ffi_call(
            'binary_jitsmv.pack_bool',
            packed_info,
        )(active)
        return jax.ffi.ffi_call(
            compute_name,
            kwargs['outs'],
        )(
            weight,
            clen,
            seed,
            packed,
            vector_size=np.int32(event_size),
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jitsmv_jvp_v(v_dot, weight, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the vector argument of the binary JIT-scalar matrix-vector product.

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
        weight, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmv_jvp_weight(w_dot, weight, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w0`` argument of the binary JIT-scalar matrix-vector product.

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
    return binary_jitsmv_p_call(
        w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder,
        **_light_options(kwargs), backend=kwargs['backend'],
    )


def _jitsmv_transpose_rules(ct, weight, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the binary JIT-scalar matrix-vector product.

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
    For the weight bounds, the transpose uses an affine decomposition of the
    output with respect to ``w0`` and ``w1``:

        ``y = w0 * C + (w1 - w0) * U``

    where ``U = y(0, 1)`` and ``C = y(1, 1)``.
    """
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
            corder=not corder,
            backend=kwargs['backend'],
        )[0]
        return weight, clen, r, seed
    elif ad.is_undefined_primal(weight):
        ones = jnp.ones((1,), dtype=ct.dtype)
        basis = binary_jitsmv_p_call(
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
        dweight = jnp.expand_dims(jnp.sum(ct * basis), axis=0)
        return dweight, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitsmv_batching(
    args,
    axes,
    **kwargs
):
    """
    Batching rule for the binary JIT-scalar matrix-vector product primitive.

    Handles ``vmap`` over the vector argument by promoting the operation to
    a matrix-matrix product (``binary_jitsmm_p_call``).

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
    if tuple(axes) == (None, None, 0, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitsmm_p_call(
            args[0],
            args[1],
            args[2].T,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitsmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            **_light_options(kwargs),
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(binary_jitsmv_p, args, axes, **kwargs)


def _binary_jitsmv_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the binary JIT-scalar matrix-vector product.

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
                weight = jnp.ones(1, dtype=dtype)
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    vector = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    vector = jnp.asarray(np.random.rand(v_size), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(BenchmarkConfig(name, (weight, clen, vector, seed), {
                    'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
                }))
    return configs


def binary_jitsmv_p_call(
    weight,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for the binary event JIT-scalar matrix-vector product.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``binary_jitsmv_p`` XLA custom kernel. This function expects
    pre-processed arguments (mantissa-only arrays, connection length instead of
    probability).

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
    binary_jitsmv : High-level wrapper with unit handling and seed initialization.
    """
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    _warn_corder_ignored(corder)
    assert jnp.issubdtype(weight.dtype, jnp.floating), 'Weights must be a floating-point type.'
    if np.dtype(weight.dtype) != np.dtype('float32'):
        raise NotImplementedError("light binary_jitsmv currently supports float32 weights only")

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert weight.shape == (1,), f"The weight shape should be (1,), but got {weight.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weight.dtype)
    )

    return binary_jitsmv_p(
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
        corder=corder,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


binary_jitsmv_p = XLACustomKernel(
    'binary_jitsmv',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_jitsmv``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) JIT scalar connectivity
matrix-vector multiplication operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

In this operation, the connectivity matrix has weights scalarly distributed between specified
bounds, and the input vector is treated as binary events (spikes). Only active events
contribute to the output computation.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_jitsmv_p.available_backends(platform)``,
and the default backend can be configured with ``binary_jitsmv_p.set_default(platform, backend)``.

See Also
--------
binary_jitsmv : High-level user-facing function wrapper.
"""
)
binary_jitsmv_p.def_cuda_raw_kernel(_binary_jitsmv_cuda_kernel, asdefault=True)
binary_jitsmv_p.def_jvp_rule2(_jitsmv_jvp_weight, None, _jitsmv_jvp_v, None)
binary_jitsmv_p.def_transpose_rule(_jitsmv_transpose_rules)
binary_jitsmv_p.def_batching_rule(_jitsmv_batching)
binary_jitsmv_p.def_call(binary_jitsmv_p_call)
binary_jitsmv_p.def_tags('jit_scalar', 'binary')
binary_jitsmv_p.def_benchmark_data(_binary_jitsmv_benchmark_data)


def _jitsmm_numba_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Generate a Numba CPU kernel for binary event JIT-scalar matrix-matrix product.

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
        A function ``kernel(w0, w1, clen, B, seed)`` that
        executes the Numba-compiled kernel and returns the result.
    """
    import numba

    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_uniform = get_numba_lfsr_uniform()

    if corder:
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                w = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_m in range(m):
                    state = _lfsr_seed(seed0 + i_m * k)
                    i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n, dtype=posts.dtype)
                    while i_k < k:
                        for j in range(B.shape[1]):
                            if B[i_k, j]:
                                out[j] += w
                        i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_m] = out
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                w = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_m in range(m):
                    state = _lfsr_seed(seed0 + i_m * k)
                    i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n, dtype=posts.dtype)
                    while i_k < k:
                        for j in range(B.shape[1]):
                            if B[i_k, j] > 0.:
                                out[j] += w
                        i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_m] = out


    else:
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                w = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_k in range(k):
                    state = _lfsr_seed(seed0 + i_k * m)
                    indices = np.where(B[i_k])[0]
                    i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                    while i_m < m:
                        posts[i_m, indices] += w
                        i_m += _lfsr_random_integers(state, 1, clen0 - 1)
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                w = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_k in range(k):
                    state = _lfsr_seed(seed0 + i_k * m)
                    indices = np.where(B[i_k] > 0.)[0]
                    i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                    while i_m < m:
                        posts[i_m, indices] += w
                        i_m += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(weight, clen, B, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, B, seed)

    return kernel


def _binary_jitsmm_cuda_kernel(
    corder: bool,
    B_info: jax.ShapeDtypeStruct,
    transpose: bool,
    shape: MatrixShape,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    **kwargs
):
    del corder
    if np.dtype(kwargs['weight_info'].dtype) != np.dtype('float32'):
        raise NotImplementedError("light binary_jitsmm currently supports float32 weights only")
    if int(B_info.shape[1]) > 32:
        raise NotImplementedError("light binary_jitsmm currently supports at most 32 columns")

    load_cuda_file(
        Path(__file__).parent.joinpath('binary_jitsmm.cu'),
        name='binary_jitsmm',
    )
    event_rows = int(B_info.shape[0])
    n_cols = int(B_info.shape[1])
    n_words = (event_rows + 31) // 32
    packed_info = jax.ShapeDtypeStruct((n_cols, n_words), jnp.uint32)
    chunk_size_value = _normalize_chunk_size(int(shape[1]), chunk_size, target_chunks)
    compute_name = 'binary_jitsmm.scatter_f32' if transpose else 'binary_jitsmm.gather_f32'

    def kernel(weight, clen, B, seed):
        active = B if B.dtype == jnp.bool_ else (B > 0).astype(jnp.int8)
        packed = jax.ffi.ffi_call(
            'binary_jitsmm.pack',
            packed_info,
        )(
            active,
            k=np.int32(event_rows),
            n=np.int32(n_cols),
            n_words=np.int32(n_words),
        )
        return jax.ffi.ffi_call(
            compute_name,
            kwargs['outs'],
        )(
            weight,
            clen,
            seed,
            packed,
            m=np.int32(shape[0]),
            k=np.int32(shape[1]),
            n=np.int32(n_cols),
            n_words=np.int32(n_words),
            chunk_size=np.int32(chunk_size_value),
        )

    return kernel


def _jitsmm_jvp_weight(w_dot, weight, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``w0`` argument of the binary JIT-scalar matrix-matrix product.

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
    return binary_jitsmm_p_call(
        w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder,
        **_light_options(kwargs), backend=kwargs['backend'],
    )


def _jitsmm_jvp_B(B_dot, weight, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the ``B`` argument of the binary JIT-scalar matrix-matrix product.

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
        weight, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmm_transpose_rules(ct, weight, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the binary JIT-scalar matrix-matrix product.

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
    For the weight bounds, the transpose uses the same affine decomposition
    as in ``_jitsmv_transpose_rules``:

        ``y = w0 * C + (w1 - w0) * U``

    where ``U = y(0, 1)`` and ``C = y(1, 1)``.
    """
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        r = jitsmm_p_call(
            weight,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            backend=kwargs['backend'],
        )[0]
        return weight, clen, r, seed
    elif ad.is_undefined_primal(weight):
        ones = jnp.ones((1,), dtype=ct.dtype)
        basis = binary_jitsmm_p_call(
            ones,
            clen,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
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
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_jitsmm_p_call(
        args[0],
        args[1],
        B,
        args[3],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
        **_light_options(kwargs),
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitsmm_batching(args, axes, **kwargs):
    """
    Batching rule for the binary JIT-scalar matrix-matrix product primitive.

    Handles ``vmap`` over the ``B`` argument along different axes by
    reshaping and delegating to ``binary_jitsmm_p_call``.

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
        return general_batching_rule(binary_jitsmm_p, args, axes, **kwargs)


def _binary_jitsmm_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the binary JIT-scalar matrix-matrix product.

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
                weight = jnp.ones(1, dtype=dtype)
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    B = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    B = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(BenchmarkConfig(name, (weight, clen, B, seed), {
                    'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
                }))
    return configs


def binary_jitsmm_p_call(
    weight,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
    chunk_size: Optional[int] = None,
    target_chunks: int = 4,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for the binary event JIT-scalar matrix-matrix product.

    Validates input shapes and dtypes, constructs output metadata, and invokes
    the ``binary_jitsmm_p`` XLA custom kernel. This function expects
    pre-processed arguments (mantissa-only arrays, connection length instead of
    probability).

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
    binary_jitsmm : High-level wrapper with unit handling and seed initialization.
    """
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)
    _warn_corder_ignored(corder)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert weight.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert weight.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if B.shape[1] > 32:
        raise NotImplementedError("light binary_jitsmm currently supports at most 32 columns")
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"
    assert jnp.issubdtype(weight.dtype, jnp.floating), 'Weights must be a floating-point type.'
    if np.dtype(weight.dtype) != np.dtype('float32'):
        raise NotImplementedError("light binary_jitsmm currently supports float32 weights only")

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weight.dtype)
    )

    return binary_jitsmm_p(
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
        corder=corder,
        chunk_size=chunk_size,
        target_chunks=target_chunks,
        backend=backend,
    )


binary_jitsmm_p = XLACustomKernel(
    'binary_jitsmm',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_jitsmm``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) JIT scalar connectivity
matrix-matrix multiplication operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

In this operation, the connectivity matrix has weights scalarly distributed between specified
bounds, and the input matrix is treated as binary events (spikes). Each column of the input
is processed independently as an event vector.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_jitsmm_p.available_backends(platform)``,
and the default backend can be configured with ``binary_jitsmm_p.set_default(platform, backend)``.

See Also
--------
binary_jitsmm : High-level user-facing function wrapper.
"""
)
binary_jitsmm_p.def_cuda_raw_kernel(_binary_jitsmm_cuda_kernel, asdefault=True)
binary_jitsmm_p.def_jvp_rule2(_jitsmm_jvp_weight, None, _jitsmm_jvp_B, None)
binary_jitsmm_p.def_transpose_rule(_jitsmm_transpose_rules)
binary_jitsmm_p.def_batching_rule(_jitsmm_batching)
binary_jitsmm_p.def_call(binary_jitsmm_p_call)
binary_jitsmm_p.def_tags('jit_scalar', 'binary')
binary_jitsmm_p.def_benchmark_data(_binary_jitsmm_benchmark_data)
