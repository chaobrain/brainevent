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
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._jitc_matrix import _initialize_seed, _initialize_conn_length
from brainevent._misc import generate_block_dim, namescope
from brainevent._numba_random import get_numba_lfsr_seed, get_numba_lfsr_random_integers
from brainevent._op import XLACustomKernel, numba_kernel, jaxinfo_to_warpinfo, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._pallas_random import get_pallas_lfsr_rng_class
from brainevent._typing import Data, MatrixShape
from .float import jitsmv_p_call, jitsmm_p_call

__all__ = [
    "binary_jitsmv",
    "binary_jitsmv_p",
    "binary_jitsmm",
    "binary_jitsmm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitsmv(
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
    binary_jitsmm : Event-driven matrix-matrix multiplication with scalar weight.
    jitsmv : Float matrix-vector multiplication with scalar weight.

    Notes
    -----
    This function computes an event-driven (spike-based) matrix-vector product
    where the connectivity matrix ``M`` has the structure:

    ``M[i, j] = w * Bernoulli(prob)``

    and the input ``vector`` is treated as a binary event vector (spikes).
    The output for each element is:

    ``y[i] = sum_{j in C(i)} w * spike[j]``

    where ``C(i)`` is the deterministic random connection set for row ``i``
    (determined by the seed), and ``spike[j]`` is 1 if ``vector[j]`` is
    True (for boolean) or ``> 0`` (for float).

    Since the input is binary, the operation reduces to counting the number
    of active (spiking) presynaptic neurons that connect to each postsynaptic
    neuron, then scaling by ``w``:

    ``y[i] = w * |{j in C(i) : spike[j] = 1}|``

    The matrix is never materialized in memory. The connectivity pattern is
    regenerated on-the-fly using the seed and connection length parameter
    ``clen = 2 / prob``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_scalar.binary import binary_jitsmv
        >>> weight = 0.5
        >>> events = jnp.array([True, False, True, True, False])
        >>> result = binary_jitsmv(weight, 0.5, events, seed=42,
        ...                        shape=(3, 5))
        >>> result.shape  # (3,)
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
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitsmm(
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
    binary_jitsmv : Event-driven matrix-vector multiplication with scalar weight.
    jitsmm : Float matrix-matrix multiplication with scalar weight.

    Notes
    -----
    This function computes an event-driven (spike-based) matrix-matrix product
    where the connectivity matrix ``M`` has the structure:

    ``M[i, j] = w * Bernoulli(prob)``

    and the input matrix ``B`` is treated as a binary event matrix (each column
    is a spike vector). For each output element:

    ``Y[i, k] = sum_{j in C(i)} w * spike[j, k]``

    where ``C(i)`` is the deterministic random connection set for row ``i``
    and ``spike[j, k]`` is 1 if ``B[j, k]`` is True (for boolean) or ``> 0``
    (for float).

    This is equivalent to performing ``binary_jitsmv`` independently for each
    column of ``B``, but is implemented more efficiently as a single kernel.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_scalar.binary import binary_jitsmm
        >>> weight = 0.5
        >>> B = jnp.array([[True, False], [False, True], [True, True],
        ...                [False, False], [True, False]])
        >>> result = binary_jitsmm(weight, 0.5, B, seed=42,
        ...                        shape=(3, 5))
        >>> result.shape  # (3, 2)
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
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitsmv_numba_kernel(
    corder: bool,
    vector_info: jax.ShapeDtypeStruct,
    **kwargs
):
    """
    Build a Numba CPU kernel for binary event-driven scalar JIT matrix-vector product.

    Parameters
    ----------
    corder : bool
        If True, iterate over columns (output dimension) as the outer loop.
        If False, iterate over rows (input dimension) as the outer loop.
    vector_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input vector. When
        ``vector_info.dtype == jnp.bool_``, the kernel uses boolean comparisons;
        otherwise it uses ``> 0`` comparisons.
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
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()

    if corder:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_col in range(n_col):
                    state = _lfsr_seed(seed0 + i_col * n_row)
                    i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.float64(0.)
                    while i_row < n_row:
                        if vector[i_row]:
                            out += 1.0
                        i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_col] = out * weight0

        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_col in range(n_col):
                    state = _lfsr_seed(seed0 + i_col * n_row)
                    i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.float64(0.)
                    while i_row < n_row:
                        if vector[i_row] > 0:
                            out += 1.0
                        i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_col] = out * weight0

    else:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_row in range(num_row):
                    if vector[i_row]:
                        state = _lfsr_seed(seed0 + i_row * num_col)
                        i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while i_col < num_col:
                            posts[i_col] += weight0
                            i_col += _lfsr_random_integers(state, 1, clen0 - 1)

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
                    if vector[i_row] > 0.:
                        state = _lfsr_seed(seed0 + i_row * num_col)
                        i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while i_col < num_col:
                            posts[i_col] += weight0
                            i_col += _lfsr_random_integers(state, 1, clen0 - 1)

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
    Build a Warp GPU kernel for binary event-driven scalar JIT matrix-vector product.

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
        If True, each GPU thread handles one output element (column-order).
        If False, each thread handles one input element (row-order) using
        atomic additions.
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
        if vector_info.dtype == jnp.bool_:
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
                    if vector[i_row]:
                        r += weight.dtype(1.)
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
                num_row = vector.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                i_col = warp.tid()
                r = weight.dtype(0.)
                state = warp.rand_init(seed0 + i_col * num_row)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    if vector[i_row] > vector.dtype(0.):
                        r += weight.dtype(1.)
                    i_row += warp.randi(state, 1, clen0)
                posts[i_col] = r * weight0

        def kernel(weight, clen, vector, seed, _):
            dim = out_info.shape[0]
            fn = jax_kernel(mv, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weight, clen, vector, seed)
    else:
        if vector_info.dtype == jnp.bool_:
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
                if vector[i_row]:
                    state = warp.rand_init(seed0 + i_row * num_col)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        warp.atomic_add(posts, i_col, weight0)
                        i_col += warp.randi(state, 1, clen0)
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
                if vector[i_row] > vector.dtype(0.):
                    state = warp.rand_init(seed0 + i_row * num_col)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        warp.atomic_add(posts, i_col, weight0)
                        i_col += warp.randi(state, 1, clen0)

        def kernel(weight, clen, vector, seed, _):
            dim = vector_info.shape[0]
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
    Build a Pallas (Triton) GPU kernel for binary event-driven scalar JIT matrix-vector product.

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

    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    dim = out_info.shape[0] if corder else vector_info.shape[0]
    block_size = generate_block_dim(dim, maximum=128)
    vector_is_bool = vector_info.dtype == jnp.bool_

    if corder:
        def pallas_kernel_fn(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_row = vector_ref.shape[0]
            weight = weight_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(data):
                i_rows, i_row_mask, rng, res = data
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                v = vector_ref[safe_rows]
                v = jnp.where(i_row_mask, v, False if vector_is_bool else 0.)
                if not vector_is_bool:
                    v = v > 0.
                res = jnp.where(v, res + weight, res)
                i_rows = i_rows + rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, res

            rng = _PallasLFSRRNG(seed + i_cols * num_row)
            i_rows = rng.random_integers(0, clen)
            i_row_mask = i_rows < num_row
            out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng, jnp.zeros(block_size, dtype=post_ref.dtype))
            )[-1]
            post_ref[i_cols] = jnp.where(i_col_mask, out, post_ref[i_cols])

        def kernel(weight, clen, vector, seed, _):
            fn = pl.pallas_call(
                pallas_kernel_fn,
                grid=(pl.cdiv(dim, block_size),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            return fn(weight, clen, vector, seed, _)
    else:
        def pallas_kernel_fn(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            weight = weight_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim
            safe_rows = jnp.where(i_row_mask, i_rows, 0)
            v = vector_ref[safe_rows]
            if vector_ref.dtype != jnp.bool_:
                v = v > 0.
            event_mask = i_row_mask & v

            def body(data):
                i_cols, i_col_mask, rng = data
                vals = jnp.full(block_size, weight, dtype=post_ref.dtype)
                atomic_add(post_ref, (i_cols,), vals, mask=event_mask & i_col_mask)
                i_cols = i_cols + rng.random_integers(1, clen0)
                return i_cols, i_cols < num_col, rng

            rng = _PallasLFSRRNG(seed0 + i_rows * num_col)
            i_cols = rng.random_integers(0, clen0)
            i_col_mask = i_cols < num_col
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng)
            )

        def kernel(weight, clen, vector, seed, _):
            fn = pl.pallas_call(
                pallas_kernel_fn,
                grid=(pl.cdiv(dim, block_size),),
                input_output_aliases={4: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            return fn(weight, clen, vector, seed, _)

    return kernel


def _jitsmv_jvp_v(v_dot, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the vector argument of the binary JIT scalar matrix-vector product.

    Computes the Jacobian-vector product with respect to the input vector by
    delegating to the float ``jitsmv_p_call`` with the tangent ``v_dot``.

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
    JVP rule for the weight argument of the binary JIT scalar matrix-vector product.

    Computes the Jacobian-vector product with respect to the weight by
    delegating to ``binary_jitsmv_p_call`` with the tangent ``w_dot``.

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
    return binary_jitsmv_p_call(
        w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmv_transpose_rules(ct, weight, clen, vector, seed, _, *, shape, transpose, corder, **kwargs):
    """
    Transpose (adjoint) rule for the binary JIT scalar matrix-vector product.

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
        dw = jnp.sum(row * vector, keepdims=True).reshape(weight.aval.shape)
        return dw, clen, vector, seed, _
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitsmv_batching(args, axes, **kwargs):
    """
    Batching (vmap) rule for the binary JIT scalar matrix-vector product.

    Handles vectorized mapping over the input vector dimension by dispatching
    to the matrix-matrix product primitive ``binary_jitsmm_p_call``.

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
        r = binary_jitsmm_p_call(
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
        r = binary_jitsmm_p_call(
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
        return general_batching_rule(binary_jitsmv_p, args, axes, **kwargs)


def _jitsmv_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the binary JIT scalar matrix-vector product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering combinations of
        transpose, corder, and boolean vs. float event vectors.
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
                configs.append(
                    BenchmarkConfig(
                        name,
                        (weight, clen, vector, seed),
                        {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                    )
                )
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
    higher-level ``binary_jitsmv`` function, which properly handles units and provides
    a more user-friendly interface.

    The operation is implemented as an XLA custom kernel to achieve high performance on
    both CPU and GPU. The primitive supports JAX transformations including grad, vmap, and jit.

    When using ``corder=True`` (default), the generated matrix ``M`` when ``transpose=False``
    will generally be different from the implicitly generated ``M^T`` when ``transpose=True``.
    Set ``corder=False`` if exact correspondence between ``M`` and ``M^T`` is required.

    See Also
    --------
    binary_jitsmv : High-level function with unit handling.
    binary_jitsmm_p_call : Low-level matrix-matrix variant.
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

    return binary_jitsmv_p(
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


binary_jitsmv_p = XLACustomKernel('binary_jitsmv')
binary_jitsmv_p.def_numba_kernel(_jitsmv_numba_kernel)
binary_jitsmv_p.def_warp_kernel(_jitsmv_warp_kernel)
binary_jitsmv_p.def_pallas_kernel('gpu', _jitsmv_pallas_kernel)
binary_jitsmv_p.def_jvp_rule2(_jitsmv_jvp_weights, None, _jitsmv_jvp_v, None, None)
binary_jitsmv_p.def_transpose_rule(_jitsmv_transpose_rules)
binary_jitsmv_p.def_batching_rule(_jitsmv_batching)
binary_jitsmv_p.def_call(binary_jitsmv_p_call)
binary_jitsmv_p.def_tags('jit_scalar', 'binary')
binary_jitsmv_p.def_benchmark_data(_jitsmv_benchmark_data)


def _jitsmm_numba_kernel(
    corder: bool,
    B_info: jax.ShapeDtypeStruct,
    **kwargs
):
    """
    Build a Numba CPU kernel for binary event-driven scalar JIT matrix-matrix product.

    Parameters
    ----------
    corder : bool
        If True, iterate over output rows as the outer loop.
        If False, iterate over input rows as the outer loop with accumulation.
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input matrix ``B``. When
        ``B_info.dtype == jnp.bool_``, the kernel uses boolean comparisons;
        otherwise it uses ``> 0`` comparisons.
    **kwargs : dict
        Additional keyword arguments, must include ``'outs'``.

    Returns
    -------
    callable
        A kernel function with signature
        ``(weight, clen, B, seed, _) -> tuple``.
    """
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()

    if corder:
        # JIT Matrix.T @ B
        # - JIT matrix: [k, m]
        # - B: [k, n]
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_m in range(m):
                    state = _lfsr_seed(seed0 + i_m * k)
                    i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n, dtype=weight.dtype)
                    while i_k < k:
                        for j in range(B.shape[1]):
                            if B[i_k, j]:
                                out[j] += 1.0
                        i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_m] = out * weight0
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_m in range(m):
                    state = _lfsr_seed(seed0 + i_m * k)
                    i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n, dtype=weight.dtype)
                    while i_k < k:
                        for j in range(B.shape[1]):
                            if B[i_k, j] > 0.:
                                out[j] += 1.0
                        i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_m] = out * weight0
    else:
        # JIT Matrix.T @ B
        # - JIT matrix: [k, m]
        # - B: [k, n]
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_k in range(k):
                    state = _lfsr_seed(seed0 + i_k * m)
                    indices = np.where(B[i_k])[0]
                    i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                    while i_m < m:
                        posts[i_m, indices] += weight0
                        i_m += _lfsr_random_integers(state, 1, clen0 - 1)
        else:
            @numba.njit(fastmath=True)
            def kernel_impl(weight, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                weight0 = weight[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_k in range(k):
                    state = _lfsr_seed(seed0 + i_k * m)
                    indices = np.where(B[i_k] > 0.)[0]
                    i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                    while i_m < m:
                        posts[i_m, indices] += weight0
                        i_m += _lfsr_random_integers(state, 1, clen0 - 1)

    def kernel(weight, clen, B, seed, _):
        return numba_kernel(kernel_impl, outs=kwargs['outs'])(weight, clen, B, seed)

    return kernel


def _jitsmm_warp_kernel(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    TITLE_SIZE: int,
    corder: bool = True,
    **kwargs
):
    """
    Build a Warp GPU kernel for binary event-driven scalar JIT matrix-matrix product.

    Parameters
    ----------
    weight_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the weight array.
    clen_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the connection length array.
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input matrix ``B``.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output matrix.
    seed_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the random seed.
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
        if B_info.dtype == jnp.bool_:
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
                        if B[i_k, j]:
                            posts[i_m, j] += weight0
                    i_k += warp.randi(state, 1, clen0)
        else:
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
                        if B[i_k, j] > float(0.0):
                            posts[i_m, j] += weight0
                    i_k += warp.randi(state, 1, clen0)

        def kernel(weight, clen, B, seed, _):
            dim = out_info.shape[0]
            fn = jax_kernel(mm, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weight, clen, B, seed, jnp.zeros(out_info.shape, out_info.dtype))
    else:
        # JIT Matrix.T @ B (corder=False)
        if B_info.dtype == jnp.bool_:
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
                        if B[i_k, j]:
                            warp.atomic_add(posts, i_m, j, weight0)
                    i_m += warp.randi(state, 1, clen0)
        else:
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
                        if B[i_k, j] > float(0.0):
                            warp.atomic_add(posts, i_m, j, weight0)
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
    Build a Pallas (Triton) GPU kernel for binary event-driven scalar JIT matrix-matrix product.

    Parameters
    ----------
    B_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the input matrix ``B``.
    out_info : jax.ShapeDtypeStruct
        Shape and dtype metadata for the output matrix.
    corder : bool, default=True
        If True, vectorize over output rows. If False, vectorize over
        input rows with atomic additions.
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

    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    B_cols = B_info.shape[1]

    if corder:
        # Match jits corder=True RNG pattern:
        # seed by output row, loop over B rows (k), vectorized over output rows.
        out_rows = out_info.shape[0]
        row_block = generate_block_dim(out_rows, maximum=128)
        grid = (pl.cdiv(out_rows, row_block), B_cols)

        def pallas_kernel_fn(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            k = B_ref.shape[0]
            weight = weight_ref[0]
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

            def body(data):
                i_cols, i_col_mask, rng, out = data
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                events = B_ref[safe_cols, col_j]
                if B_ref.dtype != jnp.bool_:
                    events = events > 0.
                out = jnp.where(i_col_mask & i_row_mask & events, out + weight, out)
                i_cols = i_cols + rng.random_integers(1, clen0)
                return i_cols, i_cols < k, rng, out

            _, _, _, out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng, out)
            )
            atomic_add(post_ref, (safe_rows, col_j), out, mask=i_row_mask)
    else:
        # Match jits corder=False RNG pattern:
        # seed by B row index k, loop over output rows, vectorized over k.
        k_dim = B_info.shape[0]
        k_block = generate_block_dim(k_dim, maximum=128)
        grid = (pl.cdiv(k_dim, k_block), B_cols)

        def pallas_kernel_fn(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            m = post_ref.shape[0]
            weight = weight_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_k_block = pl.program_id(0)
            col_j = pl.program_id(1)

            i_ks = i_k_block * k_block + jnp.arange(k_block)
            i_k_mask = i_ks < k_dim
            safe_ks = jnp.where(i_k_mask, i_ks, 0)
            events = B_ref[safe_ks, col_j]
            if B_ref.dtype != jnp.bool_:
                events = events > 0.
            rng = _PallasLFSRRNG(seed0 + i_ks * m)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < m

            def body(data):
                i_rows, i_row_mask, rng = data
                vals = jnp.where(i_k_mask & i_row_mask & events, weight, 0.)
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                atomic_add(post_ref, (safe_rows, col_j), vals, mask=i_k_mask & i_row_mask)
                i_rows = i_rows + rng.random_integers(1, clen0)
                return i_rows, i_rows < m, rng

            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng)
            )

    def kernel(weight, clen, B, seed, _):
        fn = pl.pallas_call(
            pallas_kernel_fn,
            grid=grid,
            input_output_aliases={4: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        return fn(weight, clen, B, seed, _)

    return kernel


def _jitsmm_jvp_w(w_dot, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the weight argument of the binary JIT scalar matrix-matrix product.

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
    return binary_jitsmm_p_call(
        w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitsmm_jvp_B(B_dot, weight, clen, B, seed, _, *, shape, transpose, corder, **kwargs):
    """
    JVP rule for the matrix ``B`` argument of the binary JIT scalar matrix-matrix product.

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
    Transpose (adjoint) rule for the binary JIT scalar matrix-matrix product.

    Implements the VJP backward pass for ``binary_jitsmm_p``. When ``B`` is the
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
        dw = jnp.sum(r * B, keepdims=True).reshape(weight.aval.shape)
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
    r = binary_jitsmm_p_call(
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
    Batching (vmap) rule for the binary JIT scalar matrix-matrix product.

    Handles vectorized mapping over various axes of the input matrix ``B``
    by reshaping and dispatching to ``binary_jitsmm_p_call``.

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
        return general_batching_rule(binary_jitsmm_p, args, axes, **kwargs)


def _jitsmm_benchmark_data(*, platform):
    """
    Generate benchmark configurations for the binary JIT scalar matrix-matrix product.

    Parameters
    ----------
    platform : str
        The target platform (e.g., ``'cpu'``, ``'gpu'``).

    Returns
    -------
    list of BenchmarkConfig
        A list of benchmark configurations covering combinations of
        transpose, corder, and boolean vs. float event matrices.
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
    backend: Optional[str] = None,
):
    r"""
    Low-level implementation function for binary event-driven JIT scalar matrix-matrix multiplication.

    This function prepares inputs and calls the XLA custom kernel primitive for matrix-matrix
    multiplication where the input matrix ``B`` is treated as a binary event matrix and the
    JIT connectivity matrix has a homogeneous scalar weight. The connectivity pattern is
    generated on-the-fly during execution using the provided seed and connection length.

    Parameters
    ----------
    weight : jax.Array
        Scalar weight value for non-zero connections, as a 1-D array of shape ``(1,)``.
    clen : jax.Array
        Connection length parameter (approximately ``2 / prob``), as a 1-D array
        of shape ``(1,)``.
    B : jax.Array
        Input binary event matrix of shape ``(k, n)`` where ``k`` must match the
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
    ``binary_jitsmm`` for a user-friendly interface with unit handling.

    See Also
    --------
    binary_jitsmm : High-level function with unit handling.
    binary_jitsmv_p_call : Low-level matrix-vector variant.
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

    return binary_jitsmm_p(
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


binary_jitsmm_p = XLACustomKernel('binary_jitsmm')
binary_jitsmm_p.def_numba_kernel(_jitsmm_numba_kernel)
binary_jitsmm_p.def_warp_kernel(_jitsmm_warp_kernel)
binary_jitsmm_p.def_pallas_kernel('gpu', _jitsmm_pallas_kernel)
binary_jitsmm_p.def_jvp_rule2(_jitsmm_jvp_w, None, _jitsmm_jvp_B, None, None)
binary_jitsmm_p.def_transpose_rule(_jitsmm_transpose_rules)
binary_jitsmm_p.def_batching_rule(_jitsmm_batching)
binary_jitsmm_p.def_call(binary_jitsmm_p_call)
binary_jitsmm_p.def_tags('jit_scalar', 'binary')
binary_jitsmm_p.def_benchmark_data(_jitsmm_benchmark_data)
