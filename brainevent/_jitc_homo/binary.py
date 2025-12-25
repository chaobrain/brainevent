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

from brainevent._compatible_import import pallas as pl
from brainevent._jitc_matrix import _initialize_seed, _initialize_conn_length
from brainevent._misc import generate_block_dim, namescoped_jit
from brainevent._pallas_random import LFSR88RNG
from brainevent._typing import Data, MatrixShape
from brainevent._op.main import XLACustomKernel
from brainevent._op.op_numba import numba_kernel
from brainevent._op.op_pallas import pallas_kernel
from brainevent._op.util import general_batching_rule
from brainevent._op.op_warp import jaxtype_to_warptype, warp_kernel
from .float import float_jitc_mv_homo_p_call, float_jitc_mm_homo_p_call

__all__ = [
    "binary_jitc_homo_matvec",
    "binary_jitc_homo_matmat",
]


@namescoped_jit(static_argnames=("shape", "transpose", "corder"))
def binary_jitc_homo_matvec(
    weight: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
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

    Returns
    -------
    out: Array, ndarray, Quantity
        The output of :math:`y = M @ v` if ``transpose=False``,
        or the output of :math:`y = M^T @ v` if ``transpose=True``.
    """

    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = binary_jitc_mv_homo_p_call(
        weight,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


@namescoped_jit(static_argnames=("shape", "transpose", "corder"))
def binary_jitc_homo_matmat(
    weight: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
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

    Returns
    -------
    out: Array, ndarray
        The output of :math:`y = M @ B` if ``transpose=False``,
        or the output of :math:`y = M^T @ B` if ``transpose=True``.
    """

    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = binary_jitc_mm_homo_p_call(
        weight,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitc_mv_homo_numba_kernel_generator(
    corder: bool,
    vector_info: jax.ShapeDtypeStruct,
    **kwargs
):
    if corder:
        if vector_info.dtype == jnp.bool_:
            def kernel(weight, clen, vector, seed, _, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                weight0 = weight[0]  # Homogeneous weight value
                clen0 = clen[0]  # Connection length (inverse of connection probability)
                seed0 = seed[0]  # Random seed
                np.random.seed(seed0)
                for i_col in range(n_col):
                    i_row = np.random.randint(0, clen0)
                    out = np.asarray(0., dtype=weight.dtype)
                    while i_row < n_row:
                        if vector[i_row]:
                            out += 1.0
                        i_row += np.random.randint(1, clen0)
                    posts[i_col] = out * weight0

        else:
            def kernel(weight, clen, vector, seed, _, posts):
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                weight0 = weight[0]  # Homogeneous weight value
                clen0 = clen[0]  # Connection length (inverse of connection probability)
                seed0 = seed[0]  # Random seed
                np.random.seed(seed0)
                for i_col in range(n_col):
                    i_row = np.random.randint(0, clen0)
                    out = np.asarray(0., dtype=weight.dtype)
                    while i_row < n_row:
                        if vector[i_row] != 0:
                            out += vector[i_row]
                        i_row += np.random.randint(1, clen0)
                    posts[i_col] = out * weight0

    else:
        if vector_info.dtype == jnp.bool_:
            def kernel(weight, clen, vector, seed, _, posts):
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                weight0 = weight[0]  # Homogeneous weight value applied to all connections
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                np.random.seed(seed0)
                for i_row in range(num_row):
                    is_event = vector[i_row]
                    i_col = np.random.randint(0, clen0)
                    while i_col < num_col:
                        if is_event:
                            posts[i_col] += weight0
                        i_col += np.random.randint(1, clen0)

        else:
            def kernel(weight, clen, vector, seed, _, posts):
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                weight0 = weight[0]  # Homogeneous weight value applied to all connections
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                np.random.seed(seed0)
                for i_row in range(num_row):
                    is_event = vector[i_row] != 0.
                    i_col = np.random.randint(0, clen0)
                    while i_col < num_col:
                        if is_event:
                            posts[i_col] += weight0
                        i_col += np.random.randint(1, clen0)

    return numba_kernel(kernel, parallel=False, input_output_aliases={4: 0})


def _jitc_mv_homo_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import warp

    weight_dtype = jaxtype_to_warptype(weight_info.dtype)
    clen_dtype = jaxtype_to_warptype(clen_info.dtype)
    v_dtype = jaxtype_to_warptype(vector_info.dtype)
    seed_dtype = jaxtype_to_warptype(seed_info.dtype)

    if corder:
        if vector_info.dtype == jnp.bool_:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                num_row = vector.shape[0]
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value
                i_col = warp.tid()
                r = float(0.0)
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    r = warp.where(vector[i_row], 1., 0.)
                    i_row += warp.randi(state, 1, clen0)
                posts[i_col] = r * weight0

        else:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                num_row = vector.shape[0]
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value
                i_col = warp.tid()
                r = float(0.0)
                state = warp.rand_init(seed0 + i_col)
                i_row = warp.randi(state, 0, clen0)
                while i_row < num_row:
                    r += vector[i_row]
                    i_row += warp.randi(state, 1, clen0)
                posts[i_col] = r * weight0

    else:
        if vector_info.dtype == jnp.bool_:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                num_col = posts.shape[0]
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value
                i_row = warp.tid()
                if vector[i_row]:
                    state = warp.rand_init(seed0 + i_row)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        posts[i_col] += weight0
                        i_col += warp.randi(state, 1, clen0)

        else:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                num_col = posts.shape[0]
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value
                i_row = warp.tid()
                if vector[i_row] != 0.:
                    state = warp.rand_init(seed0 + i_row)
                    i_col = warp.randi(state, 0, clen0)
                    while i_col < num_col:
                        posts[i_col] += weight0
                        i_col += warp.randi(state, 1, clen0)

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    return warp_kernel(kernel, dim=dim, input_output_aliases={4: 0})


def _jitc_mv_homo_pallas_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    tiled = True

    if corder:
        if tiled:
            block_size = generate_block_dim(dim, maximum=128)

            @pallas_kernel(
                outs=kwargs['outs'],
                tile=(pl.cdiv(dim, block_size),),
                input_output_aliases={4: 0},
            )
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_row = vector_ref.shape[0]
                weight = weight_ref[0]
                clen = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed = seed_ref[0]  # Base random seed value
                i_col_block = pl.program_id(0)
                i_cols = i_col_block * block_size + jnp.arange(block_size)
                i_col_mask = i_cols < dim

                def body(data):
                    i_rows, i_row_mask, rng, res = data
                    v = pl.load(vector_ref, i_rows, mask=i_row_mask)
                    if vector_ref.dtype != jnp.bool_:
                        v = v != 0.
                    res = jnp.where(v, res + weight, res)
                    i_rows += rng.random_integers(1, clen)
                    return i_rows, i_rows < num_row, rng, res

                rng = LFSR88RNG(seed + i_cols)
                i_rows = rng.random_integers(0, clen)
                i_row_mask = i_rows < num_row
                out = jax.lax.while_loop(
                    lambda data: jnp.sum(data[1]) > 0,
                    body,
                    (i_rows, i_row_mask, rng, jnp.zeros(block_size, dtype=post_ref.dtype))
                )[-1]
                pl.store(post_ref, i_cols, out, mask=i_col_mask)


        else:
            @pallas_kernel(
                outs=kwargs['outs'],
                tile=(dim,),
                input_output_aliases={4: 0},
            )
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_row = vector_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_col = pl.program_id(0)

                def body(data):
                    i, rng, res = data
                    if vector_ref.dtype == jnp.bool_:
                        res = jnp.where(vector_ref[i], res + weight, res)
                    else:
                        res = jnp.where(vector_ref[i] != 0., res + weight, res)
                    i += rng.random_integers(1, clen0)
                    return i, rng, res

                rng = LFSR88RNG(seed0 + i_col)
                _, _, r = jax.lax.while_loop(
                    lambda data: data[0] < num_row,
                    body,
                    (rng.random_integers(0, clen0), rng, 0.0)
                )
                post_ref[i_col] = r


    else:
        @pallas_kernel(
            outs=kwargs['outs'],
            tile=(dim,),
            input_output_aliases={4: 0},
        )
        def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            weight = weight_ref[0]
            clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
            seed0 = seed_ref[0]  # Base random seed value
            i_row = pl.program_id(0)
            v = vector_ref[i_row]

            @pl.when(v if v.dtype == jnp.bool_ else v != 0.)
            def run():
                def body(data):
                    i, rng = data
                    pl.atomic_add(post_ref, i, weight)
                    i += rng.random_integers(1, clen0)
                    return i, rng

                rng = LFSR88RNG(seed0 + i_row)
                jax.lax.while_loop(
                    lambda data: data[0] < num_col,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

    return kernel


def _jitc_mv_homo_jvp_v(
    v_dot,
    weight,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mv_homo_p_call(
        weight,
        clen,
        v_dot,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )


def _jitc_mv_homo_jvp_weights(
    w_dot,
    weight,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return binary_jitc_mv_homo_p_call(
        w_dot,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )


def _jitc_mv_homo_transpose_rules(
    ct,
    weight,
    clen,
    vector,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        r = float_jitc_mv_homo_p_call(
            weight,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder
        )[0]
        return weight, clen, r, seed, _
    elif ad.is_undefined_primal(weight):
        row = float_jitc_mv_homo_p_call(
            jnp.ones((1,), dtype=ct.dtype),
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder
        )[0]
        dw = jnp.sum(row * vector, keepdims=True).reshape(weight.aval.shape)
        return dw, clen, vector, seed, _
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitc_mv_homo_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitc_mm_homo_p_call(
            args[0],
            args[1],
            args[2].T,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitc_mm_homo_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )
        return r, [1]
    else:
        return general_batching_rule(binary_jitc_mv_homo_p, args, axes, **kwargs)


def binary_jitc_mv_homo_p_call(
    weight,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
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

    Returns
    -------
    tuple
        A tuple containing the output array from the primitive operation.
        The output shape is determined by the matrix shape and transpose flag:
        - If ``transpose=False``: output shape is (shape[0],)
        - If ``transpose=True``: output shape is (shape[1],)

    Notes
    -----
    This function is intended as an internal implementation detail and is used by the
    higher-level `jitc_matvec_homo` function, which properly handles units and provides
    a more user-friendly interface.

    The operation is implemented as an XLA custom kernel to achieve high performance on
    both CPU and GPU. The primitive supports JAX transformations including grad, vmap, and jit.

    When using ``corder=True`` (default), the generated matrix $M$ when ``transpose=False``
    will generally be different from the implicitly generated $M^T$ when ``transpose=True``.
    Set ``corder=False`` if exact correspondence between $M$ and $M^T$ is required.
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

    return binary_jitc_mv_homo_p(
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
    )


binary_jitc_mv_homo_p = XLACustomKernel('binary_jitc_mv_homo')
binary_jitc_mv_homo_p.def_cpu_kernel(_jitc_mv_homo_numba_kernel_generator)
binary_jitc_mv_homo_p.def_gpu_kernel(warp=_jitc_mv_homo_warp_kernel_generator,
                                     pallas=_jitc_mv_homo_pallas_kernel_generator,
                                     default='pallas')
binary_jitc_mv_homo_p.def_tpu_kernel(_jitc_mv_homo_pallas_kernel_generator)
binary_jitc_mv_homo_p.def_jvp_rule2(_jitc_mv_homo_jvp_weights, None, _jitc_mv_homo_jvp_v, None, None)
binary_jitc_mv_homo_p.def_transpose_rule(_jitc_mv_homo_transpose_rules)
binary_jitc_mv_homo_p.def_batching_rule(_jitc_mv_homo_batching)


def _jitc_mm_homo_numba_kernel_generator(
    transpose: bool,
    corder: bool,
    B_info: jax.ShapeDtypeStruct,
    **kwargs
):
    if corder:

        if transpose:
            # JIT Matrix.T @ B
            #
            # - JIT matrix: [k, m]
            # - B: [k, n]

            if B_info.dtype == jnp.bool_:
                def kernel(weight, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                    n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                    k = B.shape[0]  # Number of rows in B (rows in M)

                    weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)

                    for i_m in range(m):
                        i_k = np.random.randint(0, clen0)
                        out = np.zeros(n, dtype=weight.dtype)
                        while i_k < k:
                            # out += B[i_k]
                            for j in range(B.shape[1]):
                                if B[i_k, j]:
                                    out[j] += 1.0
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out * weight0

            else:
                def kernel(weight, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                    n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                    k = B.shape[0]  # Number of rows in B (rows in M)

                    weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)

                    for i_m in range(m):
                        i_k = np.random.randint(0, clen0)
                        out = np.zeros(n, dtype=weight.dtype)
                        while i_k < k:
                            # out += B[i_k]
                            for j in range(B.shape[1]):
                                if B[i_k, j] != 0.:
                                    out[j] += 1.0
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out * weight0

        else:
            # JIT Matrix @ B
            #
            # - JIT matrix: [m, k]
            # - B: [k, n]

            if B_info.dtype == jnp.bool_:
                def kernel(weight, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                    n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                    k = B.shape[0]  # Number of rows in B (columns in M)

                    weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed for reproducibility

                    for i_m in range(m):
                        i_k = np.random.randint(0, clen0)
                        out = np.zeros(n, dtype=weight.dtype)
                        while i_k < k:
                            # out += B[i_k]
                            for j in range(B.shape[1]):
                                if B[i_k, j]:
                                    out[j] += 1.0
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out * weight0
            else:
                def kernel(weight, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                    n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                    k = B.shape[0]  # Number of rows in B (columns in M)

                    weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed for reproducibility

                    for i_m in range(m):
                        i_k = np.random.randint(0, clen0)
                        out = np.zeros(n, dtype=weight.dtype)
                        while i_k < k:
                            # out += B[i_k]
                            for j in range(B.shape[1]):
                                if B[i_k, j] != 0.:
                                    out[j] += 1.0
                            i_k += np.random.randint(1, clen0)
                        posts[i_m] = out * weight0

    else:
        if transpose:
            # JIT Matrix.T @ B
            #
            # - JIT matrix: [k, m]
            # - B: [k, n]
            if B_info.dtype == jnp.bool_:
                def kernel(weight, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                    k = B.shape[0]  # Number of rows in B (rows in M)

                    weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed

                    for i_k in range(k):
                        indices = np.where(B[i_k])[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            posts[i_m, indices] += weight0
                            i_m += np.random.randint(1, clen0)
                    # for i_k in range(k):
                    #     out = B[i_k] * weight0
                    #     i_m = np.random.randint(0, clen0)
                    #     while i_m < m:
                    #         posts[i_m] += out
                    #         i_m += np.random.randint(1, clen0)

            else:
                def kernel(weight, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                    k = B.shape[0]  # Number of rows in B (rows in M)

                    weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed

                    for i_k in range(k):
                        indices = np.where(B[i_k] != 0.)[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            posts[i_m, indices] += weight0
                            i_m += np.random.randint(1, clen0)
                    # for i_k in range(k):
                    #     out = B[i_k] * weight0
                    #     i_m = np.random.randint(0, clen0)
                    #     while i_m < m:
                    #         posts[i_m] += out
                    #         i_m += np.random.randint(1, clen0)

        else:
            # JIT Matrix @ B
            #
            # - JIT matrix: [m, k]
            # - B: [k, n]
            if B_info.dtype == jnp.bool_:
                def kernel(weight, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                    k = B.shape[0]  # Number of rows in B (columns in M)

                    weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed

                    for i_k in range(k):
                        indices = np.where(B[i_k])[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            posts[i_m, indices] += weight0
                            i_m += np.random.randint(1, clen0)
                        # out = B[i_k] * weight0
                        # i_m = np.random.randint(0, clen0)
                        # while i_m < m:
                        #     posts[i_m] += out
                        #     i_m += np.random.randint(1, clen0)

            else:
                def kernel(weight, clen, B, seed, _, posts):
                    m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                    k = B.shape[0]  # Number of rows in B (columns in M)

                    weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                    seed0 = seed[0]  # Random seed for reproducible matrix generation
                    clen0 = clen[0]  # Connection length parameter (controls sparsity)
                    np.random.seed(seed0)  # Initialize random number generator with seed

                    for i_k in range(k):
                        indices = np.where(B[i_k] != 0.)[0]
                        i_m = np.random.randint(0, clen0)
                        while i_m < m:
                            posts[i_m, indices] += weight0
                            i_m += np.random.randint(1, clen0)
                        # out = B[i_k] * weight0
                        # i_m = np.random.randint(0, clen0)
                        # while i_m < m:
                        #     posts[i_m] += out
                        #     i_m += np.random.randint(1, clen0)

    return numba_kernel(kernel, parallel=False, input_output_aliases={4: 0})


def _jitc_mm_homo_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    TITLE_SIZE: int,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    import warp

    weight_dtype = jaxtype_to_warptype(weight_info.dtype)
    clen_dtype = jaxtype_to_warptype(clen_info.dtype)
    B_dtype = jaxtype_to_warptype(B_info.dtype)
    seed_dtype = jaxtype_to_warptype(seed_info.dtype)

    if corder:
        # JIT Matrix.T @ B

        if B_info.dtype == jnp.bool_:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                k = B.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]
                B = warp.array(B, dtype=weight_dtype)

                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m)

                out = warp.tile_zeros(TITLE_SIZE, dtype=weight.dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    out += warp.tile_astype(warp.tile_load(B[i_k], TITLE_SIZE), dtype=weight_dtype)
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out * weight0)

        else:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                k = B.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]

                i_m = warp.tid()
                state = warp.rand_init(seed0 + i_m)

                out = warp.tile_zeros(TITLE_SIZE, dtype=weight.dtype)
                i_k = warp.randi(state, 0, clen0)
                while i_k < k:
                    out += warp.tile_load(B[i_k], TITLE_SIZE)
                    i_k += warp.randi(state, 1, clen0)
                warp.tile_store(posts[i_m], out * weight0)


    else:
        # JIT Matrix.T @ B
        if B_info.dtype == jnp.bool_:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                m = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]

                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k)

                out = warp.tile_astype(warp.tile_load(B[i_k], TITLE_SIZE), dtype=weight_dtype) * weight0
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    warp.tile_atomic_add(posts[i_m], out)
                    i_m += warp.randi(state, 1, clen0)

        else:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                B: warp.array2d(dtype=B_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                m = posts.shape[0]
                weight0 = weight[0]
                clen0 = clen[0]
                seed0 = seed[0]

                i_k = warp.tid()
                state = warp.rand_init(seed0 + i_k)

                out = warp.tile_load(B[i_k], TITLE_SIZE) * weight0
                i_m = warp.randi(state, 0, clen0)
                while i_m < m:
                    warp.tile_atomic_add(posts[i_m], out)
                    i_m += warp.randi(state, 1, clen0)

    tile = (out_info.shape[0] if corder else B_info.shape[0])
    return warp_kernel(kernel, tile=tile, input_output_aliases={4: 0}, block_dim=256)


def _jitc_mm_homo_pallas_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    block_dim = generate_block_dim(B_info.shape[1], maximum=1024)

    if corder:
        # JIT Matrix.T @ B
        # - JIT matrix: [k, m]
        # - B: [k, n]
        def kernel(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            k = B_ref.shape[0]
            weight = weight_ref[0]
            clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
            seed0 = seed_ref[0]  # Base random seed value
            i_m = pl.program_id(0)
            i_n_block = pl.program_id(1)
            i_n_start = block_dim * i_n_block
            mask = i_n_start + jnp.arange(block_dim) < B_info.shape[1]

            def body(data):
                i, rng, out = data
                events = pl.load(B_ref, (i, pl.dslice(i_n_start, block_dim)), mask=mask)
                if B_ref.dtype == jnp.bool_:
                    out = jnp.where(events, out + weight, out)
                else:
                    out += events * weight
                i += rng.random_integers(1, clen0)
                return i, rng, out

            rng = LFSR88RNG(seed0 + i_m)
            out = jnp.zeros(block_dim, dtype=post_ref.dtype)
            _, _, out = jax.lax.while_loop(
                lambda data: data[0] < k,
                body,
                (rng.random_integers(0, clen0), rng, out)
            )
            pl.store(post_ref, (i_m, pl.dslice(i_n_start, block_dim)), out, mask=mask)


    else:
        # JIT Matrix.T @ B
        # - JIT matrix: [k, m]
        # - B: [k, n]
        def kernel(weight_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            m = post_ref.shape[0]
            weight = weight_ref[0]
            clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
            seed0 = seed_ref[0]  # Base random seed value
            i_k = pl.program_id(0)
            i_n_block = pl.program_id(1)
            i_n_start = block_dim * i_n_block
            mask = i_n_start + jnp.arange(block_dim) < B_info.shape[1]

            B_block = pl.load(B_ref, (i_k, pl.dslice(i_n_start, block_dim)), mask=mask)
            out = jnp.asarray(B_block, dtype=post_ref.dtype) * weight

            def body(data):
                i, rng = data
                pl.atomic_add(post_ref, (i, pl.dslice(i_n_start, block_dim)), out, mask=mask)
                i += rng.random_integers(1, clen0)
                return i, rng

            rng = LFSR88RNG(seed0 + i_k)
            jax.lax.while_loop(
                lambda data: data[0] < m,
                body,
                (rng.random_integers(0, clen0), rng)
            )

    tile = (out_info.shape[0] if corder else B_info.shape[0])
    grid = (tile, pl.cdiv(B_info.shape[1], block_dim))

    return pallas_kernel(
        kernel,
        tile=grid,
        input_output_aliases={4: 0},
        outs=kwargs['outs'],
    )


def _jitc_mm_homo_jvp_w(
    w_dot,
    weight,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return binary_jitc_mm_homo_p_call(
        w_dot,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


def _jitc_mm_homo_jvp_B(
    B_dot,
    weight,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    return float_jitc_mm_homo_p_call(
        weight,
        clen,
        B_dot,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


def _jitc_mm_homo_transpose_rules(
    ct,
    weight,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    corder,
    **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        r = float_jitc_mm_homo_p_call(
            weight,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
        )[0]

        return weight, clen, r, seed, _

    elif ad.is_undefined_primal(weight):
        r = float_jitc_mm_homo_p_call(
            jnp.ones((1,), dtype=ct.dtype),
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder
        )[0]
        dw = jnp.sum(r * B, keepdims=True).reshape(weight.aval.shape)
        return dw, clen, B, seed, _

    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_homo not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_jitc_mm_homo_p_call(
        args[0],
        args[1],
        B,
        args[3],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitc_mm_homo_batching(args, axes, **kwargs):
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
        return general_batching_rule(binary_jitc_mm_homo_p, args, axes, **kwargs)


def binary_jitc_mm_homo_p_call(
    weight,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
):
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

    return binary_jitc_mm_homo_p(
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
    )


binary_jitc_mm_homo_p = XLACustomKernel('binary_jitc_mm_homo')
binary_jitc_mm_homo_p.def_cpu_kernel(_jitc_mm_homo_numba_kernel_generator)
binary_jitc_mm_homo_p.def_gpu_kernel(warp=_jitc_mm_homo_warp_kernel_generator,
                                     pallas=_jitc_mm_homo_pallas_kernel_generator,
                                     default='pallas')
binary_jitc_mm_homo_p.def_tpu_kernel(_jitc_mm_homo_pallas_kernel_generator)
binary_jitc_mm_homo_p.def_jvp_rule2(_jitc_mm_homo_jvp_w, None, _jitc_mm_homo_jvp_B, None, None)
binary_jitc_mm_homo_p.def_transpose_rule(_jitc_mm_homo_transpose_rules)
binary_jitc_mm_homo_p.def_batching_rule(_jitc_mm_homo_batching)
