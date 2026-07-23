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
from typing import Literal, Optional, Sequence

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

__all__ = [
    "jits",
    "jits_p",
    "jitsmv",
    "jitsmv_p",
    "jitsmm",
    "jitsmm_p",
]

MatrixMode = Literal['mv', 'mm']


def _normalize_matrix_mode(matrix_mode: MatrixMode) -> MatrixMode:
    """Validate the ``mv``/``mm`` materialization mode.

    ``mv`` uses the 32-lane kernels (matches ``jitsmv`` and the mv CSR
    materialization); ``mm`` uses the 4-thread AW-T4 kernels (matches ``jitsmm``
    and the mm CSR materialization).  The two draw *different* connectivity
    matrices on CUDA, so the mode must be chosen explicitly by the caller.
    """
    if matrix_mode not in ('mv', 'mm'):
        raise ValueError(f"matrix_mode must be 'mv' or 'mm', got {matrix_mode!r}.")
    return matrix_mode


def _normalize_chunk_size(n_cols, chunk_size, target_chunks=4):
    """Chunk width for the light-RNG connectivity walk.

    ``chunk_size`` participates in the RNG stream keying, so every operator that
    must draw the *same* matrix (``jits``/``jitsmv``/``jitsmm``, the binary
    kernels, and the CSR materialization) has to chunk identically: default to
    ``ceil(shape[1] / target_chunks)`` with ``target_chunks=4``.
    """
    if chunk_size is None:
        target_chunks = int(target_chunks)
        if target_chunks <= 0:
            raise ValueError("target_chunks must be positive")
        chunk_size = max(1, (int(n_cols) + target_chunks - 1) // target_chunks)
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return chunk_size


@namescope(static_argnames=("shape", "transpose", "corder", "matrix_mode"))
def jits(
    weight: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    matrix_mode: MatrixMode,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    r"""Generate a homogeneous sparse random matrix on-the-fly.

    ``matrix_mode`` (``'mv'`` or ``'mm'``) is **required**: the mv and mm light
    kernels draw different matrices on CUDA, so the dense form is only well
    defined once a mode is chosen. ``mat.mv.todense()`` / ``mat.mm.todense()``
    pick these for you.

    This function creates a sparse random matrix where all non-zero values are set
    to the same homogeneous weight. Instead of storing the full matrix in memory,
    this function efficiently represents it in a form that can be used with JAX
    transformations including jit(), vmap(), grad() and pmap().

    Parameters
    ----------
    weight : Data
        The value to use for all non-zero entries in the matrix. Can be a scalar,
        an Array, ndarray, or a Quantity with units.
    prob : float
        Connection probability for the matrix (between 0 and 1). Determines the
        sparsity of the generated matrix.
    seed : int
        Random seed for reproducible matrix generation.
    shape : MatrixShape
        The shape of the matrix as a tuple (num_rows, num_cols).
    transpose : bool, default=False
        If True, return the transposed random matrix.
    corder : bool, default=True
        Controls whether the parallelization order is oriented along the matrix columns:
        - True: Sampling index along collum dimension
        - False: Sampling index along row dimension
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    Data
        The generated sparse random matrix with the specified shape. If `transpose`
        is True, the matrix is transposed, and the output shape is ``shape``.
        Otherwise, the output shape is ``(shape[1], shape[0])``.

    Raises
    ------
    ValueError
        If ``prob`` is not a scalar, is not finite, or is outside ``[0, 1]``.

    See Also
    --------
    jitsmv : Matrix-vector product with JIT-generated scalar matrix.
    jitsmm : Matrix-matrix product with JIT-generated scalar matrix.

    Notes
    -----
    The matrix ``W`` is defined element-wise as:

    ``W[i, j] = w * B[i, j]``

    where ``w`` is the scalar weight and ``B[i, j] ~ Bernoulli(prob)`` is a
    binary mask fully determined by the seed. The mask is generated using a
    deterministic PRNG that, for a given ``(seed, i, j)`` triple, always
    produces the same outcome.

    The expected number of non-zeros is ``prob * m * n`` where ``(m, n)`` is
    the matrix shape. The connection length parameter ``clen = 2 / prob``
    controls the average stride between successive non-zero entries during
    the sampling loop.

    When using ``corder=True`` (default), the matrix generated with
    ``transpose=True`` will generally be different from the transpose of the
    matrix generated with ``transpose=False``. Set ``corder=False`` if exact
    correspondence between these two cases is required.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> import brainunit as u
        >>> from brainevent._jit_scalar.float import jits
        >>> # Generate a 1000x500 sparse matrix with 10% connection probability
        >>> matrix = jits(0.01, prob=0.1, seed=42, shape=(1000, 500))
        >>> matrix.shape  # (1000, 500)
        >>> # With units
        >>> matrix_u = jits(0.01 * u.mA, prob=0.1, seed=42, shape=(1000, 500))
    """
    weight, unitd = u.split_mantissa_unit(weight)
    clen = _initialize_conn_length(prob)
    res = jits_p_call(
        weight,
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
def jitsmv(
    weight: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    r"""
    Perform the :math:`y=M@v` or :math:`y=M.T@v` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).

        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``corder=True``, with the sacrifice of
        the speed compared with ``corder=False``.

    Parameters
    ----------
    weight: Array, ndarray, Quantity, float
        The value of the random matrix.
    prob: float
        The connection probability.
    vector: Array, ndarray, Quantity
        The vector.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    corder : bool, default=True
        Controls whether the parallelization order is oriented along the matrix columns:
        - True: Sampling index along collum dimension
        - False: Sampling index along row dimension
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    out: Array, ndarray, Quantity
        The output of :math:`y = M @ v` if ``transpose=False``,
        or the output of :math:`y = M^T @ v` if ``transpose=True``.

    Raises
    ------
    ValueError
        If ``prob`` is not a scalar, is not finite, or is outside ``[0, 1]``.
    AssertionError
        If the matrix shape and vector length are incompatible.

    See Also
    --------
    jits : Generate the full JIT scalar matrix as a dense array.
    jitsmm : Matrix-matrix product with JIT-generated scalar matrix.
    binary_jitsmv : Event-driven (binary) variant of this operation.

    Notes
    -----
    The operation computes:

    ``y[i] = sum_{j in C(i)} w * v[j]``

    where ``w`` is the scalar weight, ``v`` is the input vector, and
    ``C(i)`` is the deterministic random connection set for row ``i``
    (determined by the seed and connection probability). This is equivalent
    to ``y = M @ v`` where ``M[i, j] = w * Bernoulli(prob)``.

    The weight ``w`` and vector ``v`` may carry physical units from
    ``brainunit``; the output will have the product of their units.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_scalar.float import jitsmv
        >>> v = jnp.ones(50)
        >>> result = jitsmv(0.01, 0.1, v, seed=42, shape=(100, 50))
        >>> result.shape  # (100,)
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
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose", "corder", "matrix_mode"))
def jitsmm(
    weight: Data,
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
    r"""
    Perform the :math:`y=M@B` or :math:`y=M.T@B` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@B`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``corder=True``, with the sacrifice of
        the speed compared with ``corder=False``.

    Parameters
    ----------
    weight: Array, ndarray, Quantity, float
        The value of the random matrix.
    prob: float
        The connection probability.
    B: Array, ndarray, Quantity
        The matrix.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    corder : bool, default=True
        Controls whether the parallelization order is oriented along the matrix columns:
        - True: Sampling index along collum dimension
        - False: Sampling index along row dimension
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    out: Array, ndarray
        The output of :math:`y = M @ B` if ``transpose=False``,
        or the output of :math:`y = M^T @ B` if ``transpose=True``.

    Raises
    ------
    ValueError
        If ``prob`` is not a scalar, is not finite, or is outside ``[0, 1]``.
    AssertionError
        If the matrix shape and input matrix ``B`` dimensions are incompatible.

    See Also
    --------
    jits : Generate the full JIT scalar matrix as a dense array.
    jitsmv : Matrix-vector product with JIT-generated scalar matrix.
    binary_jitsmm : Event-driven (binary) variant of this operation.

    Notes
    -----
    The operation computes:

    ``Y[i, k] = sum_{j in C(i)} w * B[j, k]``

    where ``w`` is the scalar weight, ``B`` is the input matrix, and
    ``C(i)`` is the deterministic random connection set for row ``i``.
    This is equivalent to ``Y = M @ B`` where
    ``M[i, j] = w * Bernoulli(prob)``.

    This is mathematically equivalent to performing ``jitsmv`` for each
    column of ``B``, but is implemented more efficiently as a single kernel.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_scalar.float import jitsmm
        >>> B = jnp.ones((50, 10))
        >>> result = jitsmm(0.01, 0.1, B, seed=42, shape=(100, 50))
        >>> result.shape  # (100, 10)
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
        corder=corder,
        matrix_mode=matrix_mode,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


#: Residue-class stride of the light-RNG walk. The ``mv`` kernels mirror the
#: 32-lane CUDA kernels; the ``mm`` (AW-T4) kernels mirror the 4-thread CUDA
#: kernels. The stride is part of the drawn matrix, so ``matrix_mode='mv'`` and
#: ``matrix_mode='mm'`` sample *different* connectivity -- exactly as on CUDA.
_MV_STRIDE = 32
_MM_STRIDE = 4


def _jitc_homo_matrix_numba_kernel(
    corder: bool = True,
    matrix_mode: MatrixMode = 'mv',
    transpose: bool = False,
    **kwargs
):
    """
    Build a Numba CPU kernel for generating a dense JIT scalar connectivity matrix.

    The connectivity is regenerated with the same ``light_rng`` chunk/lane walk
    as the CUDA kernel in ``float_jits.cu``, so the numba and CUDA backends
    materialize a bit-identical matrix. As on CUDA, ``corder`` selects the
    ``notrans``/``trans`` write layout while the ``matrix_mode`` (``'mv'`` /
    ``'mm'``) selects the residue-class stride (32 / 4).

    Parameters
    ----------
    corder : bool, default=True
        Selects the notrans (``True``) / trans (``False``) generation orientation.
    matrix_mode : {'mv', 'mm'}
        Selects the light-RNG stride (mv=32, mm=4); the two draw different matrices.
    transpose : bool
        Whether the primitive output is transposed (only affects the output shape;
        the write layout is governed by ``corder``, matching the CUDA dispatch).
    **kwargs
        Additional keyword arguments; must include ``'outs'``, ``'out_info'`` and
        ``'shape'``.

    Returns
    -------
    callable
        A kernel function with signature ``(weight, clen, seed) -> tuple``.
    """
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']

    stride = _MV_STRIDE if _normalize_matrix_mode(matrix_mode) == 'mv' else _MM_STRIDE
    # ``chunk_size`` keys the RNG stream and follows the CUDA/CSR convention
    # (chunk over ``shape[1]`` with ``target_chunks=4``); ``n_chunks`` is derived
    # from the walk dimension inside the kernel, as CUDA does.
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    if corder:
        # notrans: seed by output row, walk output columns, write posts[row, col].
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, posts):
            posts[:] = 0.
            n_rows = posts.shape[0]
            n_cols = posts.shape[1]
            weight0 = weight[0]
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
                            posts[row, chunk_start + local_j] = weight0
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
    else:
        # trans: seed by (swapped) row, walk columns, write posts[col, row].
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, posts):
            posts[:] = 0.
            n_rows = posts.shape[1]
            n_cols = posts.shape[0]
            weight0 = weight[0]
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
                            posts[chunk_start + local_j, row] = weight0
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

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
    corder: bool = True,
    matrix_mode: MatrixMode = 'mv',
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jits.cu'),
        name='jit_scalar_jits',
    )
    sfx = _dtype_sfx.get(np.dtype(kwargs['weight_info'].dtype), '_f32')
    mode = 'mv' if _normalize_matrix_mode(matrix_mode) == 'mv' else 'mm_aw_t4'
    # ``corder`` selects the kernel (as in binary.py); the write layout is baked
    # into notrans/trans and the output shape encodes ``transpose``.
    direction = 'notrans' if corder else 'trans'
    kernel_name = f'jit_scalar_jits.jits_{mode}_{direction}{sfx}'
    # The kernel always seeds by row and walks n_cols (notrans writes row-major,
    # trans writes transposed). Choose (n_rows, n_cols) so the filled output has
    # ``out_info``'s shape and the drawn matrix matches jitsmv/jitsmm — hence
    # chunk over shape[1], identical to the vector/matrix kernels.
    out_shape = tuple(int(s) for s in kwargs['out_info'].shape)
    n_rows, n_cols = out_shape if corder else out_shape[::-1]
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    def kernel(weight, clen, seed):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            weight, clen, seed,
            n_rows=np.int32(n_rows), n_cols=np.int32(n_cols),
            chunk_size=np.int32(chunk_size),
        )

    return kernel


def _jitc_homo_matrix_jvp_weight(weight_dot, weight, clen, seed, *, shape, transpose: bool, corder: bool, **kwargs):
    """
    JVP rule for the weight argument of the JIT scalar matrix generation primitive.

    Parameters
    ----------
    weight_dot : jax.Array
        The tangent vector for the weight.
    weight, clen, seed : jax.Array
        Primal values of the primitive inputs.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the matrix is transposed.
    corder : bool
        Column-order flag.
    **kwargs
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jits_p_call(
        weight_dot, clen, seed, shape=shape, transpose=transpose, corder=corder,
        matrix_mode=kwargs['matrix_mode'], backend=kwargs['backend'],
    )


def _jitc_homo_matrix_transpose(ct, weight, clen, seed, *, shape, transpose: bool, corder: bool, **kwargs):
    """
    Transpose (adjoint) rule for the JIT scalar matrix generation primitive.

    Computes the weight gradient by generating a unit-weight matrix and
    performing a sum-of-products with the cotangent.

    Parameters
    ----------
    ct : tuple
        The cotangent values from the output.
    weight, clen, seed : jax.Array
        Primal or undefined-primal values.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the forward pass generated a transposed matrix.
    corder : bool
        Column-order flag used in the forward pass.
    **kwargs
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        Cotangent values for ``(weight, clen, seed)``.

    Raises
    ------
    NotImplementedError
        If ``weight`` is not the undefined primal.
    """
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(weight):
        forward = jits_p_call(
            1., clen, seed, shape=shape, transpose=transpose, corder=corder,
            matrix_mode=kwargs['matrix_mode'], backend=kwargs['backend'],
        )[0]
        dw = jnp.expand_dims((ct * forward).sum(), axis=0)
        return (dw, clen, seed)

    else:
        raise NotImplementedError('JITC matrix transpose is only implemented for the weight arguments.')


def _jitc_homo_matrix_batching(args, axes, **kwargs):
    """
    Batching (vmap) rule for the JIT scalar matrix generation primitive.

    When vectorizing over the weight dimension, generates the matrix once with
    unit weight and then scales by each batched weight value.

    Parameters
    ----------
    args : tuple
        The batched arguments ``(weight, clen, seed)``.
    axes : tuple
        The batch axes for each argument.
    **kwargs
        Keyword arguments including ``'shape'``, ``'transpose'``, ``'corder'``,
        and ``'backend'``.

    Returns
    -------
    tuple
        A 2-tuple of ``(results, out_axes)``.
    """
    if tuple(axes)[1:] == (None, None):
        # vmap on weight data
        r = jits_p_call(
            jnp.asarray([1.], dtype=args[0].dtype),
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            matrix_mode=kwargs['matrix_mode'],
            backend=kwargs['backend'],
        )[0]
        weight = args[0]
        axis = axes[0]
        r = jax.vmap(lambda w: r * w, in_axes=axis, out_axes=axis)(weight)
        return [r], [axis]
    else:
        return general_batching_rule(jits_p, args, axes, **kwargs)


def _jits_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the JIT scalar matrix generation primitive.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering combinations of
        transpose and corder.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            weight = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(np.ceil(2.0 / prob), dtype=jnp.int32))
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weight, clen, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder,
                     'matrix_mode': 'mv'}
                )
            )
    return configs


def jits_p_call(
    weight,
    clen,
    seed,
    *,
    shape,
    transpose: bool,
    corder: bool,
    matrix_mode: MatrixMode,
    backend: Optional[str] = None,
):
    r"""
    Low-level implementation function for generating a JIT scalar connectivity matrix.

    This function prepares inputs and calls the XLA custom kernel primitive that
    generates a dense matrix with homogeneous weight values at stochastically
    determined positions. The connectivity pattern is produced on-the-fly using the
    provided seed and connection length parameter.

    Parameters
    ----------
    weight : jax.Array or float
        Scalar weight value for non-zero connections. Will be converted to at
        least a 1-D array internally.
    clen : jax.Array or float
        Connection length parameter (approximately ``2 / prob``). Will be
        converted to at least a 1-D array internally.
    seed : jax.Array or int
        Random seed for reproducible matrix generation. Will be converted to
        at least a 1-D array internally.
    shape : tuple of int
        The shape of the matrix as ``(num_rows, num_cols)``.
    transpose : bool
        If True, generate the transposed matrix (shape is reversed).
    corder : bool
        Column-order flag controlling the parallelization strategy.
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    tuple
        A tuple containing the generated dense matrix. If ``transpose=False``,
        the output shape is ``shape``; if ``transpose=True``, the output shape
        is ``shape[::-1]``.

    Notes
    -----
    This is an internal implementation function. Use the higher-level ``jits``
    for a user-friendly interface with unit handling.

    See Also
    --------
    jits : High-level function with unit handling.
    """
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)

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
        corder=corder,
        matrix_mode=_normalize_matrix_mode(matrix_mode),
        backend=backend,
    )


jits_p = XLACustomKernel(
    'float_jitc_homo_matrix',
    doc="""
Low-level XLA custom-kernel primitive for ``jits``.

This ``XLACustomKernel`` instance dispatches the JIT scalar connectivity matrix generation
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

This operation generates a sparse connectivity matrix where all non-zero weights are set
to the same scalar value. The connectivity pattern is generated on-the-fly using a
deterministic PRNG seeded by the provided seed value.

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
jits_p.def_numba_kernel(_jitc_homo_matrix_numba_kernel)
jits_p.def_cuda_raw_kernel(_jits_cuda_kernel, asdefault=True)
jits_p.def_jvp_rule2(_jitc_homo_matrix_jvp_weight, None, None)
jits_p.def_transpose_rule(_jitc_homo_matrix_transpose)
jits_p.def_batching_rule(_jitc_homo_matrix_batching)
jits_p.def_call(jits_p_call)
jits_p.def_tags('jit_scalar', 'float')
jits_p.def_benchmark_data(_jits_benchmark_data)


def _jitsmv_numba_kernel(
    corder: bool = True,
    **kwargs
):
    """
    Build a Numba CPU kernel for float JIT scalar matrix-vector product.

    Parameters
    ----------
    corder : bool, default=True
        If True, iterate over columns (output dimension) as the outer loop,
        accumulating the weighted sum of input vector elements. If False,
        iterate over rows (input dimension) as the outer loop, scattering
        weighted values to the output via accumulation.
    **kwargs
        Additional keyword arguments, must include ``'outs'`` specifying
        the output shape/dtype information.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, vector, seed, _) -> tuple``.
    """
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']

    stride = _MV_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    if corder:
        # notrans: m = output rows, k = vector length (walk); gather w * v[j].
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, vector, seed, posts):
            m = posts.shape[0]
            k = vector.shape[0]
            weight0 = weight[0]
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
                            out += vector[chunk_start + local_j]
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = out * weight0
    else:
        # trans: m = vector length, k = output length (walk); scatter w * v[row].
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, vector, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = vector.shape[0]
            weight0 = weight[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                v = vector[row]
                if v == 0.:
                    continue
                vw = v * weight0
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
                            posts[chunk_start + local_j] += vw
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(weight, clen, vector, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, vector, seed)

    return kernel


def _jitsmv_cuda_kernel(
    corder: bool = True,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jitsmv.cu'),
        name='jit_scalar_jitsmv',
    )
    sfx = _dtype_sfx.get(np.dtype(kwargs['weight_info'].dtype), '_f32')
    # corder selects the kernel (gather -> notrans, scatter -> trans); transpose
    # only sets the output shape (m/k are derived from the tensor sizes).
    variant = 'notrans' if corder else 'trans'
    kernel_name = f'jit_scalar_jitsmv.jitsmv_{variant}{sfx}'
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    def kernel(weight, clen, vector, seed, _):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            weight, clen, seed, vector, chunk_size=np.int32(chunk_size))

    return kernel


def _jitsmv_jvp_v(v_dot, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the vector argument of the float JIT scalar matrix-vector product.

    Parameters
    ----------
    v_dot : jax.Array
        The tangent vector for the input vector.
    weight, clen, vector, seed, _ : jax.Array
        Primal values of the primitive inputs.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the matrix is transposed.
    corder : bool
        Column-order flag.
    **kwargs
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jitsmv_p_call(
        weight, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmv_jvp_weights(w_dot, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the weight argument of the float JIT scalar matrix-vector product.

    Parameters
    ----------
    w_dot : jax.Array
        The tangent vector for the weight.
    weight, clen, vector, seed, _ : jax.Array
        Primal values of the primitive inputs.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the matrix is transposed.
    corder : bool
        Column-order flag.
    **kwargs
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jitsmv_p_call(
        w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmv_transpose_rules(ct, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the float JIT scalar matrix-vector product.

    Implements the VJP backward pass. When the vector is the undefined primal,
    computes the cotangent by running the transpose of the forward pass. When the
    weight is the undefined primal, computes the weight gradient via a
    sum-of-products reduction.

    Parameters
    ----------
    ct : tuple
        The cotangent values from the output.
    weight, clen, vector, seed, _ : jax.Array
        Primal or undefined-primal values.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the forward pass used a transposed matrix.
    corder : bool
        Column-order flag used in the forward pass.
    **kwargs
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        Cotangent values for ``(weight, clen, vector, seed, _)``.

    Raises
    ------
    NotImplementedError
        If neither ``vector`` nor ``weight`` is the undefined primal.
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
        return weight, clen, r, seed, _
    elif ad.is_undefined_primal(weight):
        row = jitsmv_p_call(
            jnp.ones((1,), dtype=ct.dtype),
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw = jnp.sum(row * vector, keepdims=True)
        return dw, clen, vector, seed, _
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
    Batching (vmap) rule for the float JIT scalar matrix-vector product.

    Handles vectorized mapping over the input vector dimension by dispatching
    to the matrix-matrix product primitive ``jitsmm_p_call``.

    Parameters
    ----------
    args : tuple
        The batched arguments ``(weight, clen, vector, seed, _)``.
    axes : tuple
        The batch axes for each argument.
    **kwargs
        Keyword arguments including ``'shape'``, ``'transpose'``, ``'corder'``,
        and ``'backend'``.

    Returns
    -------
    tuple
        A 2-tuple of ``(results, out_axes)``.
    """
    if tuple(axes) == (None, None, 0, None, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitsmm_p_call(
            args[0],
            args[1],
            args[2].T,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            matrix_mode='mm',
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitsmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            matrix_mode='mm',
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(jitsmv_p, args, axes, **kwargs)


def _jitsmv_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the float JIT scalar matrix-vector product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering combinations of
        transpose and corder.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            weight = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(np.ceil(2.0 / prob), dtype=jnp.int32))
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (weight, clen, vector, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
            }))
    return configs


def jitsmv_p_call(
    weight,
    clen,
    vector,
    seed,
    *,
    shape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    r"""
    Low-level implementation function for just-in-time generated sparse matrix-vector multiplication
    with homogeneous weight values.

    This function prepares inputs and calls the XLA custom kernel primitive for matrix-vector
    multiplication with a sparsely connected matrix that is generated on-the-fly during execution.
    It handles necessary type conversions and array formatting before passing to the underlying
    primitive operation.

    Parameters
    ----------
    weight : Array, float
        Scalar weight value for non-zero connections in the randomly generated matrix.
        Will be converted to at least 1D array internally.
    clen : Array, float
        Connection length parameter (approximately 2/connection_probability).
        Controls the sparsity of the generated matrix.
    vector : Array
        Input vector for multiplication. Shape must be compatible with the matrix shape.
    seed : int, Array
        Random seed for reproducible matrix generation.
    shape : Sequence[int]
        The shape of the implicit matrix as a tuple (num_rows, num_cols).
    transpose : bool, default=False
        If True, perform ``y = M^T @ vector`` instead of ``y = M @ vector``.
    corder : bool, default=True
        Controls the parallelization strategy:
        - True: Parallelize along output dimension (typically faster)
        - False: Parallelize along input dimension (ensures reproducibility between
                 transposed operations, but may be slower)
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    tuple
        A tuple containing the output array from the primitive operation.
        The output shape is determined by the matrix shape and transpose flag:
        - If ``transpose=False``: output shape is ``(shape[0],)``
        - If ``transpose=True``: output shape is ``(shape[1],)``

    Notes
    -----
    This function is intended as an internal implementation detail and is used by the
    higher-level ``jitsmv`` function, which properly handles units and provides
    a more user-friendly interface.

    The operation is implemented as an XLA custom kernel to achieve high performance on
    both CPU and GPU. The primitive supports JAX transformations including grad, vmap, and jit.

    When using ``corder=True`` (default), the generated matrix ``M`` when ``transpose=False``
    will generally be different from the implicitly generated ``M^T`` when ``transpose=True``.
    Set ``corder=False`` if exact correspondence between ``M`` and ``M^T`` is required.

    See Also
    --------
    jitsmv : High-level function with unit handling.
    """

    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)

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

    return jitsmv_p(
        weight,
        clen,
        vector,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
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

In this operation, the connectivity matrix has all weights set to a single scalar value,
and the input vector contains floating-point values. The operation computes a standard
matrix-vector product without event-driven sparsity.

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
jitsmv_p.def_numba_kernel(_jitsmv_numba_kernel)
jitsmv_p.def_cuda_raw_kernel(_jitsmv_cuda_kernel, asdefault=True)
jitsmv_p.def_jvp_rule2(_jitsmv_jvp_weights, None, _jitsmv_jvp_v, None, None)
jitsmv_p.def_transpose_rule(_jitsmv_transpose_rules)
jitsmv_p.def_batching_rule(_jitsmv_batching)
jitsmv_p.def_call(jitsmv_p_call)
jitsmv_p.def_tags('jit_scalar', 'float')
jitsmv_p.def_benchmark_data(_jitsmv_benchmark_data)


def _jitsmm_numba_kernel(
    corder: bool = True,
    **kwargs
):
    """
    Build a Numba CPU kernel for float JIT scalar matrix-matrix product.

    Parameters
    ----------
    corder : bool, default=True
        If True, iterate over output rows as the outer loop, gathering from
        the input matrix ``B``. If False, iterate over input rows as the
        outer loop, scattering weighted row values to the output.
    **kwargs
        Additional keyword arguments, must include ``'outs'`` specifying
        the output shape/dtype information.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, B, seed, _) -> tuple``.
    """
    import numba
    _rng = get_numba_light_rng_funcs()
    _rng_init = _rng['init']
    _rng_next = _rng['next']
    _rng_bounded = _rng['bounded']
    _rng_initial_q = _rng['initial_q']

    stride = _MM_STRIDE
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    if corder:
        # notrans: Y = M @ B. m = output rows, k = B rows (walk), n = columns.
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, B, seed, posts):
            m = posts.shape[0]
            n = posts.shape[1]
            k = B.shape[0]
            weight0 = weight[0]
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
                            out += B[chunk_start + local_j]
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)
                posts[row] = out * weight0
    else:
        # trans: Y = M^T @ B. m = B rows (walk source), k = output rows (walk).
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, B, seed, posts):
            posts[:] = 0.
            k = posts.shape[0]
            m = B.shape[0]
            weight0 = weight[0]
            seed0 = np.uint32(seed[0])
            cl = np.uint32(clen[0])
            if cl < np.uint32(2):
                cl = np.uint32(2)
            n_chunks = (k + chunk_size - 1) // chunk_size
            for row in range(m):
                out = B[row] * weight0
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
                            posts[chunk_start + local_j] += out
                            state = _rng_next(state)
                            q = q + np.uint32(1) + _rng_bounded(state, cl - np.uint32(1))
                            local_j = lane + stride * int(q)

    def kernel(weight, clen, B, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, B, seed)

    return kernel


def _jitsmm_cuda_kernel(
    corder: bool = True,
    matrix_mode: MatrixMode = 'mm',
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('float_jitsmm.cu'),
        name='jit_scalar_jitsmm',
    )
    sfx = _dtype_sfx.get(np.dtype(kwargs['weight_info'].dtype), '_f32')
    # 'mm' -> AW-T4 (4-thread) kernels (native matmat); 'mv' -> 32-lane kernels
    # (draw the same matrix as jitsmv, used by vmap(jitsmv)).
    prefix = 'jitsmm_mv' if _normalize_matrix_mode(matrix_mode) == 'mv' else 'jitsmm'
    variant = 'notrans' if corder else 'trans'
    kernel_name = f'jit_scalar_jitsmm.{prefix}_{variant}{sfx}'

    out_info = kwargs['out_info']
    B_info = kwargs['B_info']
    k_walk = int(B_info.shape[0])          # rows of B == the walk dimension
    n = int(B_info.shape[1])
    # notrans: output has m rows and walks k = B-rows; trans swaps the two roles.
    if corder:
        m_ffi, k_ffi = int(out_info.shape[0]), k_walk
    else:
        m_ffi, k_ffi = k_walk, int(out_info.shape[0])
    chunk_size = _normalize_chunk_size(int(kwargs['shape'][1]), None)

    def kernel(weight, clen, B, seed, _):
        return jax.ffi.ffi_call(kernel_name, kwargs['outs'])(
            weight, clen, seed, B,
            m=np.int32(m_ffi), k=np.int32(k_ffi), n=np.int32(n),
            chunk_size=np.int32(chunk_size))

    return kernel


def _jitsmm_jvp_w(w_dot, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the weight argument of the float JIT scalar matrix-matrix product.

    Parameters
    ----------
    w_dot : jax.Array
        The tangent vector for the weight.
    weight, clen, B, seed, _ : jax.Array
        Primal values of the primitive inputs.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the matrix is transposed.
    corder : bool
        Column-order flag.
    **kwargs
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jitsmm_p_call(
        w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder,
        matrix_mode=kwargs['matrix_mode'], backend=kwargs['backend'],
    )


def _jitsmm_jvp_B(B_dot, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the matrix ``B`` argument of the float JIT scalar matrix-matrix product.

    Parameters
    ----------
    B_dot : jax.Array
        The tangent matrix for ``B``.
    weight, clen, B, seed, _ : jax.Array
        Primal values of the primitive inputs.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the matrix is transposed.
    corder : bool
        Column-order flag.
    **kwargs
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jitsmm_p_call(
        weight, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder,
        matrix_mode=kwargs['matrix_mode'], backend=kwargs['backend'],
    )


def _jitsmm_transpose_rules(ct, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the float JIT scalar matrix-matrix product.

    Implements the VJP backward pass for ``jitsmm_p``. When ``B`` is the
    undefined primal, computes the cotangent by running the transpose of the
    forward pass. When ``weight`` is the undefined primal, computes the weight
    gradient via a sum-of-products reduction.

    Parameters
    ----------
    ct : tuple
        The cotangent values from the output.
    weight, clen, B, seed, _ : jax.Array
        Primal or undefined-primal values.
    shape : tuple of int
        The matrix shape.
    transpose : bool
        Whether the forward pass used a transposed matrix.
    corder : bool
        Column-order flag used in the forward pass.
    **kwargs
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        Cotangent values for ``(weight, clen, B, seed, _)``.

    Raises
    ------
    NotImplementedError
        If neither ``B`` nor ``weight`` is the undefined primal.
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
            matrix_mode=kwargs['matrix_mode'],
            backend=kwargs['backend'],
        )[0]

        return weight, clen, r, seed, _

    elif ad.is_undefined_primal(weight):
        r = jitsmm_p_call(
            jnp.ones((1,), dtype=ct.dtype),
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            matrix_mode=kwargs['matrix_mode'],
            backend=kwargs['backend'],
        )[0]
        dw = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return dw, clen, B, seed, _

    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_homo not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    """
    Helper for batching over axis 1 (or 2) of a 3-D input matrix ``B``.

    Reshapes the 3-D input into a 2-D matrix, runs the matrix-matrix primitive,
    and reshapes the result back to 3-D.

    Parameters
    ----------
    args : tuple
        The batched arguments ``(weight, clen, B, seed, _)``, where ``B`` is 3-D.
    axis : int, default=1
        The batch axis in the output.
    **kwargs
        Keyword arguments including ``'shape'``, ``'transpose'``, ``'corder'``,
        and ``'backend'``.

    Returns
    -------
    tuple
        A 2-tuple of ``([result_3d], [axis])``.
    """
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
        corder=kwargs['corder'],
        matrix_mode=kwargs['matrix_mode'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitsmm_batching(args, axes, **kwargs):
    """
    Batching (vmap) rule for the float JIT scalar matrix-matrix product.

    Handles vectorized mapping over various axes of the input matrix ``B``
    by reshaping and dispatching to ``jitsmm_p_call``.

    Parameters
    ----------
    args : tuple
        The batched arguments ``(weight, clen, B, seed, _)``.
    axes : tuple
        The batch axes for each argument.
    **kwargs
        Keyword arguments including ``'shape'``, ``'transpose'``, ``'corder'``,
        and ``'backend'``.

    Returns
    -------
    tuple
        A 2-tuple of ``(results, out_axes)``.
    """
    if tuple(axes) == (None, None, 0, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, 1, None, None):
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, 2, None, None):
        return _batching_axis1(args, axis=2, **kwargs)

    else:
        return general_batching_rule(jitsmm_p, args, axes, **kwargs)


def _jitsmm_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the float JIT scalar matrix-matrix product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering combinations of
        transpose and corder.
    """
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            weight = jnp.ones(1, dtype=dtype)
            clen = jnp.atleast_1d(jnp.asarray(np.ceil(2.0 / prob), dtype=jnp.int32))
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (weight, clen, B, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder,
                'matrix_mode': 'mm'
            }))
    return configs


def jitsmm_p_call(
    weight,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
    matrix_mode: MatrixMode = 'mm',
    backend: Optional[str] = None,
):
    r"""
    Low-level implementation function for float JIT scalar matrix-matrix multiplication.

    This function prepares inputs and calls the XLA custom kernel primitive for matrix-matrix
    multiplication where the JIT connectivity matrix has a homogeneous scalar weight and the
    input is a dense float matrix ``B``. The connectivity pattern is generated on-the-fly
    during execution using the provided seed and connection length.

    Parameters
    ----------
    weight : jax.Array
        Scalar weight value for non-zero connections, as a 1-D array of shape ``(1,)``.
    clen : jax.Array
        Connection length parameter (approximately ``2 / prob``), as a 1-D array
        of shape ``(1,)``.
    B : jax.Array
        Input dense matrix of shape ``(k, n)`` where ``k`` must match the
        appropriate dimension of the JIT matrix (determined by ``transpose``).
    seed : jax.Array
        Random seed as a 1-D array of shape ``(1,)``.
    shape : MatrixShape
        The shape of the implicit JIT matrix as ``(num_rows, num_cols)``.
    transpose : bool
        If True, compute ``M^T @ B``; otherwise compute ``M @ B``.
    corder : bool
        Column-order flag controlling the parallelization strategy.
    backend : str or None, optional
        The computation backend to use. If ``None``, the default backend is
        selected automatically.

    Returns
    -------
    tuple
        A tuple containing the output matrix from the primitive operation:
        - If ``transpose=False``: output shape is ``(shape[0], B.shape[1])``
        - If ``transpose=True``: output shape is ``(shape[1], B.shape[1])``

    Notes
    -----
    This is an internal implementation function. Use the higher-level
    ``jitsmm`` for a user-friendly interface with unit handling.

    See Also
    --------
    jitsmm : High-level function with unit handling.
    """
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)

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
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode=_normalize_matrix_mode(matrix_mode),
        TITLE_SIZE=B.shape[1],  # Assuming B is [k, n], we want to process n columns at once
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

In this operation, the connectivity matrix has all weights set to a single scalar value,
and the input matrix contains floating-point values. Each column of the input is
processed independently in a standard matrix-matrix product.

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
jitsmm_p.def_numba_kernel(_jitsmm_numba_kernel)
jitsmm_p.def_cuda_raw_kernel(_jitsmm_cuda_kernel, asdefault=True)
jitsmm_p.def_jvp_rule2(_jitsmm_jvp_w, None, _jitsmm_jvp_B, None, None)
jitsmm_p.def_transpose_rule(_jitsmm_transpose_rules)
jitsmm_p.def_batching_rule(_jitsmm_batching)
jitsmm_p.def_call(jitsmm_p_call)
jitsmm_p.def_tags('jit_scalar', 'float')
jitsmm_p.def_benchmark_data(_jitsmm_benchmark_data)
