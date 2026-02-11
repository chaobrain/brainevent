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

from typing import Optional, Sequence

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._jitc_matrix import _initialize_seed, _initialize_conn_length
from brainevent._misc import generate_block_dim, namescope
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._numba_random import lfsr88_seed, lfsr88_random_integers
from brainevent._pallas_random import PallasLFSR88RNG
from brainevent._typing import Data, MatrixShape


__all__ = [
    "jits",
    "jits_p",
    "jitsmv",
    "jitsmv_p",
    "jitsmm",
    "jitsmm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def jits(
    weight: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    r"""Generate a homogeneous sparse random matrix on-the-fly.

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


@namescope(static_argnames=("shape", "transpose", "corder"))
def jitsmm(
    weight: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
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
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitc_homo_matrix_numba_kernel(
    corder: bool = True,
    **kwargs
):
    """
    Build a Numba CPU kernel for generating a dense JIT scalar connectivity matrix.

    Parameters
    ----------
    corder : bool, default=True
        If True, iterate over rows as the outer loop (each row samples column
        indices). If False, iterate over columns as the outer loop (each column
        samples row indices).
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'`` specifying
        the output shape/dtype information.

    Returns
    -------
    callable
        A kernel function with signature ``(weight, clen, seed) -> tuple``.
    """
    import numba

    if corder:
        # JIT matrix.T - JIT matrix shape = [m, n]
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[1]
            n = posts.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_row in range(n):
                state = lfsr88_seed(seed0 + i_row * m)
                i_col = lfsr88_random_integers(state, 0, clen0 - 1)
                while i_col < m:
                    posts[i_row, i_col] = weight0
                    i_col += lfsr88_random_integers(state, 1, clen0 - 1)
    else:
        # JIT matrix.T - JIT matrix shape = [m, n]
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, seed, posts):
            posts[:] = 0.
            m = posts.shape[1]
            n = posts.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_col in range(m):
                state = lfsr88_seed(seed0 + i_col * n)
                i_row = lfsr88_random_integers(state, 0, clen0 - 1)
                while i_row < n:
                    posts[i_row, i_col] = weight0
                    i_row += lfsr88_random_integers(state, 1, clen0 - 1)

    def kernel(weight, clen, seed):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, seed)

    return kernel


def _jitc_homo_matrix_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Build a Warp GPU kernel for generating a dense JIT scalar connectivity matrix.

    Parameters
    ----------
    weight_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the weight array.
    clen_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the connection length array.
    seed_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the random seed.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output matrix.
    corder : bool, default=True
        If True, each GPU thread handles one row. If False, each thread
        handles one column.
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'``.

    Returns
    -------
    callable
        A kernel function with signature ``(weight, clen, seed) -> tuple``.
    """
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if corder:
        # JIT matrix.T - JIT matrix shape = [m, n]
        @warp.kernel
        def mat(
            weight: weight_warp_info,
            clen: clen_warp_info,
            seed: seed_warp_info,
            posts: out_warp_info,
        ):
            m = posts.shape[1]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            i_row = warp.tid()
            state = warp.rand_init(seed0 + i_row * m)
            i_col = warp.randi(state, 0, clen0)
            while i_col < m:
                posts[i_row, i_col] = weight0
                i_col += warp.randi(state, 1, clen0)

        def kernel(weight, clen, seed):
            dim = out_info.shape[0]
            fn = jax_kernel(mat, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weight, clen, seed, jnp.zeros(out_info.shape, out_info.dtype))
    else:
        # JIT matrix.T - JIT matrix shape = [m, n]
        @warp.kernel
        def mat(
            weight: weight_warp_info,
            clen: clen_warp_info,
            seed: seed_warp_info,
            posts: out_warp_info,
        ):
            n = posts.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            i_col = warp.tid()
            state = warp.rand_init(seed0 + i_col * n)
            i_row = warp.randi(state, 0, clen0)
            while i_row < n:
                posts[i_row, i_col] = weight0
                i_row += warp.randi(state, 1, clen0)

        def kernel(weight, clen, seed):
            dim = out_info.shape[1]
            fn = jax_kernel(mat, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weight, clen, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _jitc_homo_matrix_pallas_kernel(
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Build a Pallas (Triton) GPU kernel for generating a dense JIT scalar connectivity matrix.

    Parameters
    ----------
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output matrix.
    corder : bool, default=True
        If True, each Pallas program handles one row. If False, each program
        handles one column.
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'``.

    Returns
    -------
    callable
        A kernel function with signature ``(weight, clen, seed) -> tuple``.
    """
    from jax.experimental import pallas as pl

    dim = out_info.shape[0] if corder else out_info.shape[1]

    if corder:
        def pallas_kernel_fn(weight_ref, clen_ref, seed_ref, _, post_ref):
            m = post_ref.shape[1]
            weight = weight_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row = pl.program_id(0)

            def body(data):
                i_col, rng = data
                post_ref[i_row, i_col] = weight
                i_col = i_col + rng.random_integers(1, clen0)
                return i_col, rng

            rng = PallasLFSR88RNG(seed0 + i_row * m)
            jax.lax.while_loop(
                lambda data: data[0] < m,
                body,
                (rng.random_integers(0, clen0), rng)
            )
    else:
        def pallas_kernel_fn(weight_ref, clen_ref, seed_ref, _, post_ref):
            n = post_ref.shape[0]
            weight = weight_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_col = pl.program_id(0)

            def body(data):
                i_row, rng = data
                post_ref[i_row, i_col] = weight
                i_row = i_row + rng.random_integers(1, clen0)
                return i_row, rng

            rng = PallasLFSR88RNG(seed0 + i_col * n)
            jax.lax.while_loop(
                lambda data: data[0] < n,
                body,
                (rng.random_integers(0, clen0), rng)
            )

    def kernel(weight, clen, seed):
        fn = pl.pallas_call(
            pallas_kernel_fn,
            grid=(dim,),
            input_output_aliases={3: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        out = kwargs['outs'][0]
        return fn(weight, clen, seed, jnp.zeros(out.shape, out.dtype))

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
    **kwargs : dict
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jits_p_call(
        weight_dot, clen, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
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
    **kwargs : dict
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
            1., clen, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
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
    **kwargs : dict
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
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weight, clen, seed),
                    {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
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
    jitsmv_p_call : Low-level matrix-vector product variant.
    jitsmm_p_call : Low-level matrix-matrix product variant.
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
        backend=backend,
    )


jits_p = XLACustomKernel('float_jitc_homo_matrix')
jits_p.def_numba_kernel(_jitc_homo_matrix_numba_kernel)
jits_p.def_warp_kernel(_jitc_homo_matrix_warp_kernel)
jits_p.def_pallas_kernel('gpu', _jitc_homo_matrix_pallas_kernel)
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
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'`` specifying
        the output shape/dtype information.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, vector, seed, _) -> tuple``.
    """
    import numba

    if corder:
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, vector, seed, posts):
            n_col = posts.shape[0]
            n_row = vector.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_col in range(n_col):
                state = lfsr88_seed(seed0 + i_col * n_row)
                i_row = lfsr88_random_integers(state, 0, clen0 - 1)
                out = np.float64(0.)
                while i_row < n_row:
                    out += vector[i_row]
                    i_row += lfsr88_random_integers(state, 1, clen0 - 1)
                posts[i_col] = out * weight0
    else:
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, vector, seed, posts):
            posts[:] = 0.
            num_col = posts.shape[0]
            num_row = vector.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            for i_row in range(num_row):
                state = lfsr88_seed(seed0 + i_row * num_col)
                v = vector[i_row] * weight0
                i_col = lfsr88_random_integers(state, 0, clen0 - 1)
                while i_col < num_col:
                    posts[i_col] += v
                    i_col += lfsr88_random_integers(state, 1, clen0 - 1)

    def kernel(weight, clen, vector, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, vector, seed)

    return kernel


def _jitsmv_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Build a Warp GPU kernel for float JIT scalar matrix-vector product.

    Parameters
    ----------
    weight_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the weight array.
    clen_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the connection length array.
    vector_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input vector.
    seed_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the random seed.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output array.
    corder : bool, default=True
        If True, each GPU thread handles one output element.
        If False, each thread handles one input element using atomic additions.
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'``.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, vector, seed, _) -> tuple``.
    """
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    vector_warp_info = jaxinfo_to_warpinfo(vector_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if corder:
        @warp.kernel
        def mv(
            weight: weight_warp_info,
            clen: clen_warp_info,
            vector: vector_warp_info,
            seed: seed_warp_info,
            posts: out_warp_info,
        ):
            num_row = vector.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            i_col = warp.tid()
            r = weight.dtype(0.)
            state = warp.rand_init(seed0 + i_col * num_row)
            i_row = warp.randi(state, 0, clen0)
            while i_row < num_row:
                r += vector[i_row]
                i_row += warp.randi(state, 1, clen0)
            posts[i_col] = r * weight0

    else:
        @warp.kernel
        def mv(
            weight: weight_warp_info,
            clen: clen_warp_info,
            vector: vector_warp_info,
            seed: seed_warp_info,
            posts: out_warp_info,
        ):
            num_col = posts.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            i_row = warp.tid()
            v = vector[i_row] * weight0
            state = warp.rand_init(seed0 + i_row * num_col)
            i_col = warp.randi(state, 0, clen0)
            while i_col < num_col:
                warp.atomic_add(posts, i_col, v)
                i_col += warp.randi(state, 1, clen0)

    def kernel(weight, clen, vector, seed, _):
        dim = out_info.shape[0] if corder else vector_info.shape[0]
        fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
        return fn(weight, clen, vector, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _jitsmv_pallas_kernel(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Build a Pallas (Triton) GPU kernel for float JIT scalar matrix-vector product.

    Parameters
    ----------
    vector_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input vector.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output array.
    corder : bool, default=True
        If True, vectorize over output elements. If False, vectorize over
        input elements with atomic additions.
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'``.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, vector, seed, _) -> tuple``.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    dim = (out_info.shape[0] if corder else vector_info.shape[0])

    if corder:
        block_size = int(np.gcd(generate_block_dim(dim, maximum=128), dim))

        def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_row = vector_ref.shape[0]
            weight = weight_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(data):
                i_rows, i_row_mask, rng, out = data
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                v = vector_ref[safe_rows]
                v = jnp.where(i_row_mask, v, 0.0)
                out += v * weight
                i_rows += rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, out

            rng = PallasLFSR88RNG(seed + i_cols * num_row)
            i_rows = rng.random_integers(0, clen)
            i_row_mask = i_rows < num_row
            out = jnp.zeros(block_size, dtype=post_ref.dtype)
            out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng, out)
            )[-1]
            post_ref[i_cols] = jnp.where(i_col_mask, out, post_ref[i_cols])

        def run(weights, clen, vector, seed, _):
            fn = pl.pallas_call(
                kernel,
                grid=(pl.cdiv(dim, block_size),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            return fn(weights, clen, vector, seed, _)

        return run

    else:
        if corder:
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_row = vector_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_col = pl.program_id(0)

                def body(data):
                    i, rng, res = data
                    res += vector_ref[i] * weight
                    i += rng.random_integers(1, clen0)
                    return i, rng, res

                rng = PallasLFSR88RNG(seed0 + i_col * num_row)
                _, _, r = jax.lax.while_loop(
                    lambda data: data[0] < num_row,
                    body,
                    (rng.random_integers(0, clen0), rng, 0.0)
                )
                post_ref[i_col] = r

        else:
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_col = post_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]
                seed0 = seed_ref[0]
                i_row = pl.program_id(0)
                v = vector_ref[i_row] * weight

                def body(data):
                    i, rng = data
                    atomic_add(post_ref, (i,), v)
                    i += rng.random_integers(1, clen0)
                    return i, rng

                rng = PallasLFSR88RNG(seed0 + i_row * num_col)
                jax.lax.while_loop(
                    lambda data: data[0] < num_col,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

        def run(weights, clen, vector, seed, _):
            fn = pl.pallas_call(
                kernel,
                grid=(dim,),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            return fn(weights, clen, vector, seed, _)

        return run


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
    **kwargs : dict
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
    **kwargs : dict
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
    **kwargs : dict
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
    **kwargs : dict
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
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
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
    jitsmm_p_call : Low-level matrix-matrix product variant.
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


jitsmv_p = XLACustomKernel('float_jitsmv')
jitsmv_p.def_numba_kernel(_jitsmv_numba_kernel)
jitsmv_p.def_warp_kernel(_jitsmv_warp_kernel)
jitsmv_p.def_pallas_kernel('gpu', _jitsmv_pallas_kernel)
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
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'`` specifying
        the output shape/dtype information.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, B, seed, _) -> tuple``.
    """
    import numba

    if corder:
        # JIT Matrix.T @ B - JIT matrix: [k, m], B: [k, n]
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, B, seed, posts):
            m = posts.shape[0]
            n = posts.shape[1]
            k = B.shape[0]
            weight0 = weight[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for i_m in range(m):
                state = lfsr88_seed(seed0 + i_m * k)
                i_k = lfsr88_random_integers(state, 0, clen0 - 1)
                out = np.zeros(n, dtype=B.dtype)
                while i_k < k:
                    out += B[i_k]
                    i_k += lfsr88_random_integers(state, 1, clen0 - 1)
                posts[i_m] = out * weight0
    else:
        # JIT Matrix.T @ B - JIT matrix: [k, m], B: [k, n]
        @numba.njit(fastmath=True)
        def kernel_impl(weight, clen, B, seed, posts):
            posts[:] = 0.
            m = posts.shape[0]
            k = B.shape[0]
            weight0 = weight[0]
            seed0 = seed[0]
            clen0 = clen[0]
            for i_k in range(k):
                state = lfsr88_seed(seed0 + i_k * m)
                out = B[i_k] * weight0
                i_m = lfsr88_random_integers(state, 0, clen0 - 1)
                while i_m < m:
                    posts[i_m] += out
                    i_m += lfsr88_random_integers(state, 1, clen0 - 1)

    def kernel(weight, clen, B, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, B, seed)

    return kernel


def _jitsmm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    TITLE_SIZE: int,
    corder: bool = True,
    **kwargs
):
    """
    Build a Warp GPU kernel for float JIT scalar matrix-matrix product.

    Parameters
    ----------
    weight_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the weight array.
    clen_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the connection length array.
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input matrix ``B``.
    seed_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the random seed.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output matrix.
    TITLE_SIZE : int
        Number of columns in ``B``, used for inner loop bounds.
    corder : bool, default=True
        If True, each GPU thread handles one output row.
        If False, each thread handles one input row using atomic additions.
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'``.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, B, seed, _) -> tuple``.
    """
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    clen_warp_info = jaxinfo_to_warpinfo(clen_info)
    B_warp_info = jaxinfo_to_warpinfo(B_info)
    seed_warp_info = jaxinfo_to_warpinfo(seed_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if corder:
        # JIT Matrix.T @ B
        @warp.kernel
        def mm(
            weight: weight_warp_info,
            clen: clen_warp_info,
            B: B_warp_info,
            seed: seed_warp_info,
            posts: out_warp_info,
        ):
            k = B.shape[0]
            n = B.shape[1]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            i_m = warp.tid()
            state = warp.rand_init(seed0 + i_m * k)
            i_k = warp.randi(state, 0, clen0)
            while i_k < k:
                for j in range(n):
                    posts[i_m, j] += B[i_k, j] * weight0
                i_k += warp.randi(state, 1, clen0)

        def kernel(weight, clen, B, seed, _):
            dim = out_info.shape[0]
            fn = jax_kernel(mm, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weight, clen, B, seed, jnp.zeros(out_info.shape, out_info.dtype))
    else:
        # JIT Matrix.T @ B
        @warp.kernel
        def mm(
            weight: weight_warp_info,
            clen: clen_warp_info,
            B: B_warp_info,
            seed: seed_warp_info,
            posts: out_warp_info,
        ):
            m = posts.shape[0]
            n = B.shape[1]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            i_k = warp.tid()
            state = warp.rand_init(seed0 + i_k * m)
            i_m = warp.randi(state, 0, clen0)
            while i_m < m:
                for j in range(n):
                    warp.atomic_add(posts, i_m, j, B[i_k, j] * weight0)
                i_m += warp.randi(state, 1, clen0)

        def kernel(weight, clen, B, seed, _):
            dim = B_info.shape[0]
            fn = jax_kernel(mm, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weight, clen, B, seed, jnp.zeros(out_info.shape, out_info.dtype))

    return kernel


def _jitsmm_pallas_kernel(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Build a Pallas (Triton) GPU kernel for float JIT scalar matrix-matrix product.

    Parameters
    ----------
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input matrix ``B``.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output matrix.
    corder : bool, default=True
        If True, vectorize over output rows. If False, use scalar indexing
        over input rows with atomic additions.
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'``.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, B, seed, _) -> tuple``.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    B_cols = B_info.shape[1]

    if corder:
        # Match jits corder=True RNG pattern:
        # seed by output row, loop over B rows (k), vectorized over output rows.
        out_rows = out_info.shape[0]
        row_block = int(np.gcd(generate_block_dim(out_rows, maximum=128), out_rows))
        grid = (pl.cdiv(out_rows, row_block), B_cols)

        def kernel(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            k = B_ref.shape[0]
            weight0 = weight_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            col_j = pl.program_id(1)

            i_rows = i_row_block * row_block + jnp.arange(row_block)
            i_row_mask = i_rows < out_rows
            safe_rows = jnp.where(i_row_mask, i_rows, 0)

            rng = PallasLFSR88RNG(seed0 + i_rows * k)
            i_cols = rng.random_integers(0, clen0)
            i_col_mask = i_cols < k
            out = jnp.zeros(row_block, dtype=post_ref.dtype)

            def body(data):
                i_cols, i_col_mask, rng, out = data
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                b_vals = B_ref[safe_cols, col_j]
                out += jnp.where(i_col_mask & i_row_mask, b_vals * weight0, 0.)
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < k, rng, out

            _, _, _, out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1] & i_row_mask) > 0,
                body,
                (i_cols, i_col_mask, rng, out)
            )
            atomic_add(post_ref, (safe_rows, col_j), out, mask=i_row_mask)

    else:
        # Match jits corder=False RNG pattern:
        # seed by B row index k, loop over output rows, scalar per-k thread.
        k_dim = B_info.shape[0]
        grid = (k_dim, B_cols)

        def kernel(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            m = post_ref.shape[0]
            weight0 = weight_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_k = pl.program_id(0)
            col_j = pl.program_id(1)
            val = B_ref[i_k, col_j] * weight0

            rng = PallasLFSR88RNG(seed0 + i_k * m)
            i_rows = rng.random_integers(0, clen0)

            def body(data):
                i_rows, rng = data
                atomic_add(post_ref, (i_rows, col_j), val)
                i_rows += rng.random_integers(1, clen0)
                return i_rows, rng

            jax.lax.while_loop(
                lambda data: data[0] < m,
                body,
                (i_rows, rng)
            )

    def run(weights, clen, B, seed, _):
        fn = pl.pallas_call(
            kernel,
            grid=grid,
            input_output_aliases={4: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(weights, clen, B, seed, _)

    return run


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
    **kwargs : dict
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jitsmm_p_call(
        w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
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
    **kwargs : dict
        Must contain ``'backend'``.

    Returns
    -------
    tuple
        The JVP result as a tuple of arrays.
    """
    return jitsmm_p_call(
        weight, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
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
    **kwargs : dict
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
    **kwargs : dict
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
    **kwargs : dict
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
            clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            seed = jnp.asarray(42, dtype=jnp.uint32)
            name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'}"
            configs.append(BenchmarkConfig(name, (weight, clen, B, seed), {
                'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
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
    jitsmv_p_call : Low-level matrix-vector product variant.
    jits_p_call : Low-level matrix generation variant.
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
        TITLE_SIZE=B.shape[1],  # Assuming B is [k, n], we want to process n columns at once
        backend=backend,
    )


jitsmm_p = XLACustomKernel('float_jitsmm')
jitsmm_p.def_numba_kernel(_jitsmm_numba_kernel)
jitsmm_p.def_warp_kernel(_jitsmm_warp_kernel)
jitsmm_p.def_pallas_kernel('gpu', _jitsmm_pallas_kernel)
jitsmm_p.def_jvp_rule2(_jitsmm_jvp_w, None, _jitsmm_jvp_B, None, None)
jitsmm_p.def_transpose_rule(_jitsmm_transpose_rules)
jitsmm_p.def_batching_rule(_jitsmm_batching)
jitsmm_p.def_call(jitsmm_p_call)
jitsmm_p.def_tags('jit_scalar', 'float')
jitsmm_p.def_benchmark_data(_jitsmm_benchmark_data)
