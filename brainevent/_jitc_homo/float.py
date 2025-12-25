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

__all__ = [
    "float_jitc_homo_matrix",
    "float_jitc_homo_matvec",
    "float_jitc_homo_matmat",
]


@namescoped_jit(static_argnames=("shape", "transpose", "corder"))
def float_jitc_homo_matrix(
    weight: Data,
    prob: float,
    seed: int,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
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

    Returns
    -------
    Data
        The generated sparse random matrix with the specified shape. If `transpose`
        is True, the matrix is transposed, and the output shape is ``shape``.
        Otherwise, the output shape is ``(shape[1], shape[0])``.

    Notes
    -----
    The matrix is generated using a probabilistic sampling approach rather than
    explicitly storing all values. This allows efficient operations with very large
    sparse matrices that would otherwise be impractical to store in memory.

    When using corder=True (default), the matrix generated with transpose=True
    will generally be different from the transpose of the matrix generated with transpose=False.
    Set corder=False if exact correspondence between these two cases is required.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>>
    >>> # Generate a 1000x500 sparse matrix with 10% connection probability
    >>> rng_seed = 42
    >>> weight = 0.01  # All connections have this value
    >>> matrix = float_jitc_homo_matrix(weight, prob=0.1, seed=rng_seed,
    ...                           shape=(1000, 500))
    >>>
    >>> # With units
    >>> import brainunit as u
    >>> weight_with_units = 0.01 * u.mA
    >>> matrix_with_units = float_jitc_homo_matrix(weight_with_units, prob=0.1,
    ...                                      seed=rng_seed, shape=(1000, 500))
    """
    weight, unitd = u.split_mantissa_unit(weight)
    clen = _initialize_conn_length(prob)
    res = float_jitc_homo_matrix_p_call(
        weight,
        clen,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd)


@namescoped_jit(static_argnames=("shape", "transpose", "corder"))
def float_jitc_homo_matvec(
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
    res = float_jitc_mv_homo_p_call(
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
def float_jitc_homo_matmat(
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
    res = float_jitc_mm_homo_p_call(
        weight,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitc_homo_matrix_numba_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    if corder:
        # This means that the for loop is parallelized along the dimension of the output vector: ``post.shape[0]``.

        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(weight, clen, seed, _, posts):
                m = posts.shape[1]
                n = posts.shape[0]

                # Extract scalar values from input arrays
                weight0 = weight[0]  # Homogeneous weight value
                clen0 = clen[0]  # Connection length (inverse of connection probability)
                seed0 = seed[0]  # Random seed

                # Initialize the random number generator with the provided seed
                # This ensures reproducibility for the same seed value
                np.random.seed(seed0)

                # Process each output element (column in the matrix)
                for i_row in range(n):
                    # Generate first row index randomly - this determines where to start sampling
                    i_col = np.random.randint(0, clen0)

                    # Process all connected entries for this column
                    while i_col < m:
                        posts[i_row, i_col] = weight0

                        # Skip ahead to next connected row (sparse sampling)
                        # The random skip ensures proper connection probability
                        # Each skip distance is randomly determined to maintain the sparse pattern
                        i_col += np.random.randint(1, clen0)

        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(weight, clen, seed, _, posts):
                m = posts.shape[0]
                n = posts.shape[1]

                # Extract scalar values from input arrays for more efficient access in loops
                weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)

                # Initialize the random number generator with the provided seed
                # This ensures the "random" matrix is reproducible for the same seed value
                np.random.seed(seed0)

                # Process each output element (each row of the matrix)
                for i_row in range(m):
                    # Generate first column index randomly - this determines where to start sampling
                    i_col = np.random.randint(0, clen0)

                    # Process all connected entries for this row
                    while i_col < n:
                        # Set the current matrix element to the weight value
                        posts[i_row, i_col] = weight0

                        # Skip ahead to next connected column (sparse sampling)
                        # The random skip ensures proper connection probability
                        i_col += np.random.randint(1, clen0)

    else:
        # This means that the for loop is parallelized along the dimension of the vector: ``vector.shape[0]``.

        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(weight, clen, seed, _, posts):
                m = posts.shape[1]
                n = posts.shape[0]

                # Extract scalar values from input arrays for more efficient repeated access
                weight0 = weight[0]  # Homogeneous weight value applied to all connections
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation

                # Initialize the random number generator with the provided seed
                # This ensures reproducibility for the same seed value
                np.random.seed(seed0)

                # Process each column of the matrix sequentially
                for i_col in range(m):
                    # Generate first row index randomly - this determines where to start sampling in this column
                    i_row = np.random.randint(0, clen0)

                    # Process all connected entries for this column
                    while i_row < n:
                        # Set the current matrix element to the weight value
                        posts[i_row, i_col] = weight0

                        # Skip ahead to next connected row (sparse sampling)
                        # The random skip ensures proper connection probability
                        i_row += np.random.randint(1, clen0)

        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(weight, clen, seed, _, posts):
                m = posts.shape[0]  # Number of rows in the output matrix
                n = posts.shape[1]  # Number of columns in the output matrix

                # Extract scalar values from input arrays for more efficient access in loops
                weight0 = weight[0]  # Homogeneous weight value applied to all connections
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation

                # Initialize the random number generator with the provided seed
                # This ensures reproducibility for the same seed value
                np.random.seed(seed0)

                # Process each column of the matrix sequentially
                for i_col in range(n):
                    # Generate first row index randomly - this determines where to start sampling in this column
                    i_row = np.random.randint(0, clen0)

                    # Process all connected entries for this column
                    while i_row < m:
                        # Set the current matrix element to the weight value
                        posts[i_row, i_col] = weight0

                        # Skip ahead to next connected row (sparse sampling)
                        # The random skip ensures proper connection probability
                        i_row += np.random.randint(1, clen0)

    return numba_kernel(kernel, parallel=False, input_output_aliases={3: 0})


def _jitc_homo_matrix_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    r"""
    Generate the GPU kernel for the :func:`_jitc_matvec_homo` operation.
    """
    import warp

    weight_dtype = jaxtype_to_warptype(weight_info.dtype)
    clen_dtype = jaxtype_to_warptype(clen_info.dtype)
    seed_dtype = jaxtype_to_warptype(seed_info.dtype)

    if corder:

        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                m = posts.shape[1]

                # Extract scalar values from input arrays for more efficient access
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one output element
                i_row = warp.tid()

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_row)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_col = warp.randi(state, 0, clen0)

                # Process all connected entries for this output element
                while i_col < m:
                    posts[i_row, i_col] = weight0

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_col += warp.randi(state, 1, clen0)


        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                n = posts.shape[1]  # Get number of columns in the output matrix

                # Extract scalar values from input arrays for more efficient access
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one output element (one row of the matrix)
                i_row = warp.tid()

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_row)

                # Sample the first connected column using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_col = warp.randi(state, 0, clen0)

                # Process all connected entries for this output element (row)
                while i_col < n:
                    # Add contribution from the current connected element
                    posts[i_row, i_col] = weight0

                    # Skip ahead to next connected column using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_col += warp.randi(state, 1, clen0)

    else:

        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                n = posts.shape[0]

                # Extract scalar values from input arrays for more efficient access
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one input element (column)
                i_col = warp.tid()

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_col)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_row = warp.randi(state, 0, clen0)

                # Process all connected output positions for this input element
                while i_row < n:
                    # Set the current matrix element to the weight value
                    # For this transpose=True and corder=False case, we're setting elements column by column
                    posts[i_row, i_col] = weight0

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_row += warp.randi(state, 1, clen0)


        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array2d(dtype=weight_dtype),
                posts: warp.array2d(dtype=weight_dtype),
            ):
                m = posts.shape[0]

                # Extract scalar values from input arrays for more efficient access
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one input element (column)
                i_col = warp.tid()

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_col)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_row = warp.randi(state, 0, clen0)

                # Process all connected output positions for this input element
                while i_row < m:
                    # Set the current matrix element to the weight value
                    posts[i_row, i_col] = weight0

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_row += warp.randi(state, 1, clen0)

    dim = out_info.shape[0] if corder else out_info.shape[1]
    return warp_kernel(kernel, dim=dim, input_output_aliases={3: 0})


def _jitc_homo_matrix_pallas_kernel_generator(
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    dim = out_info.shape[0] if corder else out_info.shape[1]
    tiled = True

    if tiled:
        block_size = generate_block_dim(dim, maximum=128)
        if corder:
            def _raw_kernel(
                weight_ref,  # [1]
                clen_ref,  # [1]
                seed_ref,  # [1]
                _,  # [m, n]
                post_ref,  # [m, n]
            ):
                m = post_ref.shape[1]
                weight = weight_ref[0]
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_row_block = pl.program_id(0)
                i_rows = i_row_block * block_size + jnp.arange(block_size)
                i_row_mask = i_rows < dim

                def body(data):
                    i_cols, i_col_mask, rng = data
                    pl.store(post_ref, (i_rows, i_cols), jnp.full(block_size, weight), mask=i_row_mask & i_col_mask)
                    i_cols += rng.random_integers(1, clen0)
                    return i_cols, i_cols < m, rng

                rng = LFSR88RNG(seed0 + i_rows)
                i_cols = rng.random_integers(0, clen0)
                i_col_mask = i_cols < m
                jax.lax.while_loop(
                    lambda data: jnp.sum(data[1]) > 0,
                    body,
                    (i_cols, i_col_mask, rng)
                )

        else:
            def _raw_kernel(
                weight_ref,  # [1]
                clen_ref,  # [1]
                seed_ref,  # [1]
                _,  # [m, n]
                post_ref,  # [m, n]
            ):
                n = post_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_col_block = pl.program_id(0)
                i_cols = i_col_block * block_size + jnp.arange(block_size)
                i_col_mask = i_cols < dim

                def body(data):
                    i_rows, i_row_mask, rng = data
                    pl.store(post_ref, (i_rows, i_cols), jnp.full(block_size, weight), mask=i_row_mask & i_col_mask)
                    i_rows = i_rows + rng.random_integers(1, clen0)
                    return i_rows, i_rows < n, rng

                rng = LFSR88RNG(seed0 + i_cols)
                i_rows = rng.random_integers(0, clen0)
                i_row_mask = i_rows < n
                jax.lax.while_loop(
                    lambda data: jnp.sum(data[1]) > 0,
                    body,
                    (i_rows, i_row_mask, rng)
                )

        return pallas_kernel(
            _raw_kernel,
            outs=kwargs['outs'],
            tile=(pl.cdiv(dim, block_size),),
            input_output_aliases={3: 0}
        )

    else:
        if corder:
            def _raw_kernel(
                weight_ref,
                clen_ref,
                seed_ref,
                _,
                post_ref,
            ):
                m = post_ref.shape[1]
                weight0 = weight_ref[0]  # Homogeneous weight for all connections
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_row = pl.program_id(0)

                def body(data):
                    i, rng_ = data
                    post_ref[i_row, i] = weight0
                    i = i + rng_.random_integers(1, clen0)
                    return i, rng_

                rng = LFSR88RNG(seed0 + i_row)
                jax.lax.while_loop(
                    lambda data: data[0] < m,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

        else:
            def _raw_kernel(
                weight_ref,
                clen_ref,
                seed_ref,
                _,
                post_ref,
            ):
                n = post_ref.shape[0]
                weight0 = weight_ref[0]  # Homogeneous weight for all connections
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_col = pl.program_id(0)

                def body(data):
                    i_row, rng_ = data
                    post_ref[i_row, i_col] = weight0
                    i_row = i_row + rng_.random_integers(1, clen0)
                    return i_row, rng_

                rng = LFSR88RNG(seed0 + i_col)
                jax.lax.while_loop(
                    lambda data: data[0] < n,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

        return pallas_kernel(
            _raw_kernel,
            outs=kwargs['outs'],
            tile=(dim,),
            input_output_aliases={3: 0}
        )


def _jitc_homo_matrix_jvp_weight(
    weight_dot, weight, clen, seed, _, *,
    shape: Sequence[int], transpose: bool, corder: bool, **kwargs
):
    return float_jitc_homo_matrix_p_call(weight_dot, clen, seed, shape=shape, transpose=transpose, corder=corder)


def _jitc_homo_matrix_transpose(
    ct, weight, clen, seed, _, *, shape: Sequence[int], transpose: bool, corder: bool, **kwargs
):
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    ct = ct[0]
    if ad.is_undefined_primal(weight):
        forward = float_jitc_homo_matrix_p_call(
            1., clen, seed, shape=shape, transpose=transpose, corder=corder
        )[0]
        dw = jnp.expand_dims((ct * forward).sum(), axis=0)
        return (dw, clen, seed, _)

    else:
        raise NotImplementedError(
            'JITC matrix transpose is only implemented for the weight arguments.'
        )


def _jitc_homo_matrix_batching(args, axes, **kwargs):
    if tuple(axes)[1:] == (None, None, None):
        # vmap on weight data
        r = float_jitc_homo_matrix_p_call(
            jnp.asarray([1.], dtype=args[0].dtype),
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
        )[0]
        weight = args[0]
        axis = axes[0]
        r = jax.vmap(lambda w: r * w, in_axes=axis, out_axes=axis)(weight)
        return [r], [axis]
    else:
        return general_batching_rule(float_jitc_homo_matrix_p, args, axes, **kwargs)


def float_jitc_homo_matrix_p_call(
    weight,
    clen,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)
    seed = jnp.atleast_1d(seed)

    out_info = (
        jax.ShapeDtypeStruct(shape[::-1], dtype=weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct(shape, dtype=weight.dtype)
    )

    return float_jitc_homo_matrix_p(
        weight,
        clen,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
    )


float_jitc_homo_matrix_p = XLACustomKernel('float_jitc_homo_matrix')
float_jitc_homo_matrix_p.def_cpu_kernel(_jitc_homo_matrix_numba_kernel_generator)
float_jitc_homo_matrix_p.def_gpu_kernel(warp=_jitc_homo_matrix_warp_kernel_generator,
                                        pallas=_jitc_homo_matrix_pallas_kernel_generator,
                                        default='pallas')
float_jitc_homo_matrix_p.def_tpu_kernel(_jitc_homo_matrix_pallas_kernel_generator)
float_jitc_homo_matrix_p.def_jvp_rule2(_jitc_homo_matrix_jvp_weight, None, None)
float_jitc_homo_matrix_p.def_transpose_rule(_jitc_homo_matrix_transpose)
float_jitc_homo_matrix_p.def_batching_rule(_jitc_homo_matrix_batching)


def _jitc_mv_homo_numba_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_homo` operation.
    """

    if corder:
        # This means that the for loop is parallelized along the dimension of the output vector: ``post.shape[0]``.

        if transpose:
            def kernel(weight, clen, vector, seed, _, posts):
                # Output vector dimension = number of columns in the matrix
                n_col = posts.shape[0]

                # Input vector dimension = number of rows in the matrix
                n_row = vector.shape[0]

                # Extract scalar values from input arrays
                weight0 = weight[0]  # Homogeneous weight value
                clen0 = clen[0]  # Connection length (inverse of connection probability)
                seed0 = seed[0]  # Random seed

                # Initialize the random number generator with the provided seed
                # This ensures reproducibility for the same seed value
                np.random.seed(seed0)

                # Process each output element (column in the matrix)
                for i_col in range(n_col):
                    # Generate first row index randomly - this determines where to start sampling
                    i_row = np.random.randint(0, clen0)

                    # Initialize accumulator for this output element with proper dtype
                    out = np.asarray(0., dtype=vector.dtype)

                    # Process all connected entries for this column
                    while i_row < n_row:
                        # Add contribution from the current connected element
                        out += vector[i_row]

                        # Skip ahead to next connected row (sparse sampling)
                        # The random skip ensures proper connection probability
                        # Each skip distance is randomly determined to maintain the sparse pattern
                        i_row += np.random.randint(1, clen0)

                    # Scale accumulated sum by weight and store in output array
                    # All connections have the same homogeneous weight value
                    posts[i_col] = out * weight0

        else:
            def kernel(weight, clen, vector, seed, _, posts):
                # Output vector dimension = number of rows in the matrix
                # Each row in the matrix will produce one element in the output vector
                num_row = posts.shape[0]

                # Input vector dimension = number of columns in the matrix
                # The input vector must match the number of columns in our implicit matrix
                num_col = vector.shape[0]

                # Extract scalar values from input arrays for more efficient access in loops
                weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)

                # Initialize the random number generator with the provided seed
                # This ensures the "random" matrix is reproducible for the same seed value
                np.random.seed(seed0)

                # Process each output element (each row of the matrix)
                for i_row in range(num_row):
                    # Randomly determine the first column where this row has a connection
                    # This implements efficient sampling of a sparse pattern
                    i_col = np.random.randint(0, clen0)

                    # Initialize accumulator for the dot product result for this row
                    # Using input vector's dtype ensures proper numerical precision
                    out = np.asarray(0., dtype=vector.dtype)

                    # Process all connected entries for this row by skipping through columns
                    # This is the core sparse sampling algorithm - we only process columns
                    # that have connections rather than checking every possible column
                    while i_col < num_col:
                        # Add contribution from the current connected element
                        # For connected positions, we add the corresponding vector element
                        out += vector[i_col]

                        # Skip ahead to next connected column using geometric-like distribution
                        # The random skip distance models the sparse connectivity pattern
                        # where each position has approximately 1/clen0 probability of connection
                        i_col += np.random.randint(1, clen0)

                    # Scale accumulated sum by weight and store in output array
                    # All connections share the same homogeneous weight value
                    posts[i_row] = out * weight0

    else:
        # This means that the for loop is parallelized along the dimension of the vector: ``vector.shape[0]``.

        if transpose:
            def kernel(weight, clen, vector, seed, _, posts):
                # Output vector dimension = number of columns in the matrix
                # This is the dimension of the result vector in the vector @ matrix operation
                num_col = posts.shape[0]

                # Input vector dimension = number of rows in the matrix
                # The vector elements are processed one by one, with each contributing to multiple output elements
                num_row = vector.shape[0]

                # Extract scalar values from input arrays for more efficient repeated access
                weight0 = weight[0]  # Homogeneous weight value applied to all connections
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation

                # Initialize the random number generator with the provided seed
                # This ensures the "random" matrix is reproducible for the same seed value
                np.random.seed(seed0)

                # Process each input row (vector element) and distribute its value to connected columns
                # This implements the vector @ matrix operation one row at a time
                for i_row in range(num_row):
                    # Pre-multiply the input value by weight for efficiency
                    # This avoids multiplying inside the inner loop for each connection
                    v = vector[i_row] * weight0

                    # Sample the first connected column using random skipping
                    # This implements the sparse sampling - each row connects to ~num_col/clen0 columns on average
                    # Starting from a random position in [0,clen0) creates variability in connection patterns
                    i_col = np.random.randint(0, clen0)

                    # Continue sampling and accumulating while we haven't exceeded output dimension
                    # This loop processes all columns this particular row connects to
                    while i_col < num_col:
                        # Add this connection's contribution to the appropriate output element
                        # The output is accumulated as we process each input element's contributions
                        posts[i_col] += v

                        # Move to the next connected column using geometric-like skipping
                        # Each next connection is approximately clen0 positions away on average
                        # This creates a sparse pattern where only ~1/clen0 of all possible connections exist
                        i_col += np.random.randint(1, clen0)

        else:
            def kernel(weight, clen, vector, seed, _, posts):
                # Output vector dimension = number of rows in the matrix
                # This represents the first dimension of the matrix and the result vector's size
                num_row = posts.shape[0]

                # Input vector dimension = number of columns in the matrix
                # Each element of the input vector corresponds to a column in the matrix
                num_col = vector.shape[0]

                # Extract scalar values from input arrays for more efficient access in loops
                weight0 = weight[0]  # Homogeneous weight value applied to all connections
                clen0 = clen[0]  # Controls sparsity - higher values mean fewer connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation

                # Initialize the random number generator with the provided seed
                # This ensures the "random" matrix is reproducible for the same seed value
                np.random.seed(seed0)

                # Process each input element (each column of the matrix)
                # This implements the matrix @ vector operation one column at a time
                for i_col in range(num_col):
                    # Pre-multiply the input value by weight for efficiency
                    # This avoids multiplying inside the inner loop for each connection
                    v = vector[i_col] * weight0

                    # Sample the first connected row using random skipping
                    # This implements the sparse sampling - each column connects to ~num_row/clen0 rows on average
                    # Starting from a random position in [0,clen0) creates variability in connection patterns
                    i_row = np.random.randint(0, clen0)

                    # Continue sampling and accumulating while we haven't exceeded output dimension
                    # This loop processes all rows this particular column connects to
                    while i_row < num_row:
                        # Add this connection's contribution to the appropriate output element
                        # The output is accumulated as we process each column's contributions
                        posts[i_row] += v

                        # Move to the next connected row using geometric-like skipping
                        # Each next connection is approximately clen0 positions away on average
                        # This creates a sparse pattern where only ~1/clen0 of all possible connections exist
                        i_row += np.random.randint(1, clen0)

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
    r"""
    Generate the GPU kernel for the :func:`_jitc_matvec_homo` operation.
    """
    import warp

    weight_dtype = jaxtype_to_warptype(weight_info.dtype)
    clen_dtype = jaxtype_to_warptype(clen_info.dtype)
    v_dtype = jaxtype_to_warptype(vector_info.dtype)
    seed_dtype = jaxtype_to_warptype(seed_info.dtype)

    if corder:

        if transpose:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                # Input vector dimension (number of rows in the matrix)
                num_row = vector.shape[0]

                # Extract scalar values from input arrays for more efficient access
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one output element
                i_col = warp.tid()

                # Initialize accumulator for dot product calculation
                r = float(0.0)

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_col)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_row = warp.randi(state, 0, clen0)

                # Process all connected entries for this output element
                while i_row < num_row:
                    # Add contribution from the current connected element
                    r += vector[i_row]

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_row += warp.randi(state, 1, clen0)

                # Scale accumulated sum by weight and store in output array
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
                # Input vector dimension (number of columns in the matrix)
                num_col = vector.shape[0]

                # Extract scalar values from input arrays for more efficient access
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one output element (one row of the matrix)
                i_row = warp.tid()

                # Initialize accumulator for dot product calculation
                r = float(0.0)

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_row)

                # Sample the first connected column using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_col = warp.randi(state, 0, clen0)

                # Process all connected entries for this output element (row)
                while i_col < num_col:
                    # Add contribution from the current connected element
                    r += vector[i_col]

                    # Skip ahead to next connected column using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_col += warp.randi(state, 1, clen0)

                # Scale accumulated sum by weight and store in output array
                posts[i_row] = r * weight0
    else:

        if transpose:
            def kernel(
                weight: warp.array1d(dtype=weight_dtype),
                clen: warp.array1d(dtype=clen_dtype),
                vector: warp.array1d(dtype=v_dtype),
                seed: warp.array1d(dtype=seed_dtype),
                _: warp.array1d(dtype=weight_dtype),
                posts: warp.array1d(dtype=weight_dtype),
            ):
                # Output dimension (number of columns in the matrix)
                num_col = posts.shape[0]

                # Extract scalar values from input arrays for more efficient access
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one input element (row)
                i_row = warp.tid()

                # Pre-multiply the input value by weight for efficiency
                # This avoids multiplying inside the inner loop for each connection
                v = vector[i_row] * weight0

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_row)

                # Sample the first connected column using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_col = warp.randi(state, 0, clen0)

                # Process all connected output positions for this input element
                while i_col < num_col:
                    # Atomically add contribution to the appropriate output element
                    # Using atomic operation because multiple threads may update the same output element
                    posts[i_col] += v

                    # Skip ahead to next connected column using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
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
                # Output dimension (number of rows in the matrix)
                num_row = posts.shape[0]

                # Extract scalar values from input arrays for more efficient access
                weight0 = weight[0]  # Homogeneous weight for all connections
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                seed0 = seed[0]  # Base random seed value

                # Get thread ID - each thread processes one input element (column)
                i_col = warp.tid()

                # Pre-multiply the input value by weight for efficiency
                # This avoids multiplying inside the inner loop for each connection
                v = vector[i_col] * weight0

                # Initialize random state with base seed plus thread ID to ensure
                # different but reproducible random sequences across threads
                state = warp.rand_init(seed0 + i_col)

                # Sample the first connected row using random skipping
                # Start at a random position in [0, clen0) for variability in connection patterns
                i_row = warp.randi(state, 0, clen0)

                # Process all connected output positions for this input element
                while i_row < num_row:
                    # Atomically add contribution to the appropriate output element
                    # Using atomic operation because multiple threads may update the same output element
                    posts[i_row] += v

                    # Skip ahead to next connected row using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_row += warp.randi(state, 1, clen0)

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    return warp_kernel(kernel, input_output_aliases={4: 0}, dim=dim)


def _jitc_mv_homo_pallas_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    tiled = True

    if tiled:
        block_size = generate_block_dim(dim, maximum=128)

        if corder:
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_row = vector_ref.shape[0]
                weight = weight_ref[0]
                clen = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed = seed_ref[0]  # Base random seed value
                i_col_block = pl.program_id(0)
                i_cols = i_col_block * block_size + jnp.arange(block_size)
                i_col_mask = i_cols < dim

                def body(data):
                    i_rows, i_row_mask, rng, out = data
                    out += pl.load(vector_ref, i_rows, mask=i_row_mask) * weight
                    i_rows += rng.random_integers(1, clen)
                    return i_rows, i_rows < num_row, rng, out

                rng = LFSR88RNG(seed + i_cols)
                i_rows = rng.random_integers(0, clen)
                i_row_mask = i_rows < num_row
                out = jnp.zeros(block_size, dtype=post_ref.dtype)
                out = jax.lax.while_loop(
                    lambda data: jnp.sum(data[1]) > 0,
                    body,
                    (i_rows, i_row_mask, rng, out)
                )[-1]
                pl.store(post_ref, i_cols, out, mask=i_col_mask)

        else:
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_col = post_ref.shape[0]
                weight = weight_ref[0]
                clen = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed = seed_ref[0]  # Base random seed value
                i_row_block = pl.program_id(0)
                i_rows = i_row_block * block_size + jnp.arange(block_size)
                i_row_mask = i_rows < dim
                vector = pl.load(vector_ref, i_rows, mask=i_row_mask) * weight

                def body(data):
                    i_cols, i_col_mask, rng = data
                    pl.atomic_add(post_ref, i_cols, vector, mask=i_row_mask & i_col_mask)
                    i_cols += rng.random_integers(1, clen)
                    return i_cols, i_cols < num_col, rng

                rng = LFSR88RNG(seed + i_rows)
                i_cols = rng.random_integers(0, clen)
                i_col_mask = i_cols < num_col
                jax.lax.while_loop(
                    lambda data: jnp.sum(data[1]) > 0,
                    body,
                    (i_cols, i_col_mask, rng)
                )

        return pallas_kernel(
            kernel,
            outs=kwargs['outs'],
            tile=(pl.cdiv(dim, block_size),),
            input_output_aliases={4: 0},
        )


    else:
        if corder:
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_row = vector_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_col = pl.program_id(0)

                def body(data):
                    i, rng, res = data
                    res += vector_ref[i] * weight
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
            def kernel(weight_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
                num_col = post_ref.shape[0]
                weight = weight_ref[0]
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_row = pl.program_id(0)
                v = vector_ref[i_row] * weight

                def body(data):
                    i, rng = data
                    pl.atomic_add(post_ref, i, v)
                    i += rng.random_integers(1, clen0)
                    return i, rng

                rng = LFSR88RNG(seed0 + i_row)
                jax.lax.while_loop(
                    lambda data: data[0] < num_col,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

        return pallas_kernel(
            kernel,
            outs=kwargs['outs'],
            tile=(dim,),
            input_output_aliases={4: 0},
        )

    return final_kernel


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
    return float_jitc_mv_homo_p_call(
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
        dw = jnp.sum(row * vector, keepdims=True)
        return dw, clen, vector, seed, _
    else:
        raise NotImplementedError(
            f"Transpose rule for {ct} not implemented "
            f"for event-driven COO matrix-vector product."
        )


def _jitc_mv_homo_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = float_jitc_mm_homo_p_call(
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
        r = float_jitc_mm_homo_p_call(
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
        return general_batching_rule(float_jitc_mv_homo_p, args, axes, **kwargs)


def float_jitc_mv_homo_p_call(
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

    return float_jitc_mv_homo_p(
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


float_jitc_mv_homo_p = XLACustomKernel('float_jitc_mv_homo')
float_jitc_mv_homo_p.def_cpu_kernel(_jitc_mv_homo_numba_kernel_generator)
float_jitc_mv_homo_p.def_gpu_kernel(default='pallas',
                                    warp=_jitc_mv_homo_warp_kernel_generator,
                                    pallas=_jitc_mv_homo_pallas_kernel_generator, )
float_jitc_mv_homo_p.def_tpu_kernel(_jitc_mv_homo_pallas_kernel_generator)
float_jitc_mv_homo_p.def_jvp_rule2(
    _jitc_mv_homo_jvp_weights,
    None,
    _jitc_mv_homo_jvp_v,
    None,
    None
)
float_jitc_mv_homo_p.def_transpose_rule(_jitc_mv_homo_transpose_rules)
float_jitc_mv_homo_p.def_batching_rule(_jitc_mv_homo_batching)


def _jitc_mm_homo_numba_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    if corder:

        if transpose:
            # JIT Matrix.T @ B
            #
            # - JIT matrix: [k, m]
            # - B: [k, n]

            def kernel(weight, clen, B, seed, _, posts):
                m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                k = B.shape[0]  # Number of rows in B (rows in M)

                weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)

                for i_m in range(m):
                    # Start at a random position in [0, clen0) for variability in connection patterns
                    i_k = np.random.randint(0, clen0)

                    # Initialize accumulator for this output row with proper dtype
                    out = np.zeros(n, dtype=B.dtype)

                    # Process all connected entries for this output row
                    while i_k < k:
                        # Add contribution from the current connected input row
                        out += B[i_k]

                        # Skip ahead to next connected row using geometric-like distribution
                        # This creates sparse connectivity with ~1/clen0 connection probability
                        i_k += np.random.randint(1, clen0)

                    # Scale accumulated sum by weight and store in output array
                    posts[i_m] = out * weight0

        else:
            # JIT Matrix @ B
            #
            # - JIT matrix: [m, k]
            # - B: [k, n]

            def kernel(weight, clen, B, seed, _, posts):
                m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                n = posts.shape[1]  # Number of columns in output matrix (columns in B)
                k = B.shape[0]  # Number of rows in B (columns in M)

                weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)  # Initialize random number generator with seed for reproducibility

                for i_m in range(m):
                    # Start at a random position in [0, clen0) for variability in connection patterns
                    i_k = np.random.randint(0, clen0)

                    # Initialize accumulator for this output row with proper dtype
                    out = np.zeros(n, dtype=B.dtype)

                    # Process all connected entries for this output row
                    while i_k < k:
                        # Add contribution from the current connected input row
                        out += B[i_k]

                        # Skip ahead to next connected row using geometric-like distribution
                        # This creates sparse connectivity with ~1/clen0 connection probability
                        i_k += np.random.randint(1, clen0)

                    # Scale accumulated sum by weight and store in output array
                    posts[i_m] = out * weight0

    else:
        if transpose:
            # JIT Matrix.T @ B
            #
            # - JIT matrix: [k, m]
            # - B: [k, n]

            def kernel(weight, clen, B, seed, _, posts):
                m = posts.shape[0]  # Number of rows in output matrix (columns in M)
                k = B.shape[0]  # Number of rows in B (rows in M)

                weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)  # Initialize random number generator with seed

                # Process each input row sequentially
                for i_k in range(k):
                    # Pre-multiply the current row by weight for efficiency
                    out = B[i_k] * weight0

                    # Sample the first connected output row using random skipping
                    # Start at a random position in [0, clen0) for variability in connection patterns
                    i_m = np.random.randint(0, clen0)

                    # Process all connected output rows for this input row
                    while i_m < m:
                        # Add contribution to the connected output row
                        # Using += to accumulate results across all input rows
                        posts[i_m] += out

                        # Skip ahead to next connected output row using geometric-like distribution
                        # This creates sparse connectivity with ~1/clen0 connection probability
                        i_m += np.random.randint(1, clen0)

        else:
            # JIT Matrix @ B
            #
            # - JIT matrix: [m, k]
            # - B: [k, n]

            def kernel(weight, clen, B, seed, _, posts):
                m = posts.shape[0]  # Number of rows in output matrix (rows in M)
                k = B.shape[0]  # Number of rows in B (columns in M)

                weight0 = weight[0]  # Homogeneous weight value for all non-zero connections
                seed0 = seed[0]  # Random seed for reproducible matrix generation
                clen0 = clen[0]  # Connection length parameter (controls sparsity)
                np.random.seed(seed0)  # Initialize random number generator with seed

                # Process each input column sequentially
                for i_k in range(k):
                    # Pre-multiply the current row by weight for efficiency
                    out = B[i_k] * weight0

                    # Sample the first connected output row using random skipping
                    # Start at a random position in [0, clen0) for variability in connection patterns
                    i_m = np.random.randint(0, clen0)

                    # Process all connected output rows for this input column
                    while i_m < m:
                        # Add contribution to the connected output row
                        # Using += to accumulate results across all input columns
                        posts[i_m] += out

                        # Skip ahead to next connected output row using geometric-like distribution
                        # This creates sparse connectivity with ~1/clen0 connection probability
                        i_m += np.random.randint(1, clen0)

    return numba_kernel(kernel, parallel=False, input_output_aliases={4: 0})


def _jitc_mm_homo_warp_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
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
        if transpose:
            # JIT Matrix.T @ B
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
            # JIT Matrix @ B
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
        if transpose:
            # JIT Matrix.T @ B
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


        else:
            # JIT Matrix @ B
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
    kernel = warp_kernel(kernel, tile=tile, block_dim=256, input_output_aliases={4: 0})
    return kernel


def _jitc_mm_homo_pallas_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    block_dim = generate_block_dim(B_info.shape[1], maximum=1024)

    if corder:
        if transpose:
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
                    out += pl.load(B_ref, (i, pl.dslice(i_n_start, block_dim)), mask=mask) * weight
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
            # - JIT matrix: [m, k]
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
                    out += pl.load(B_ref, (i, pl.dslice(i_n_start, block_dim)), mask=mask) * weight
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
        if transpose:
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
                out = B_block * weight

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

        else:
            # JIT Matrix.T @ B
            # - JIT matrix: [m, k]
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
                out = B_block * weight

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
        outs=kwargs['outs']
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
    return float_jitc_mm_homo_p_call(
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
        dw = jnp.expand_dims(jnp.sum(r * B), axis=0)
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
    r = float_jitc_mm_homo_p_call(
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
        return general_batching_rule(float_jitc_mm_homo_p, args, axes, **kwargs)


def float_jitc_mm_homo_p_call(
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

    return float_jitc_mm_homo_p(
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


float_jitc_mm_homo_p = XLACustomKernel('float_jitc_mm_homo')
float_jitc_mm_homo_p.def_cpu_kernel(_jitc_mm_homo_numba_kernel_generator)
float_jitc_mm_homo_p.def_gpu_kernel(warp=_jitc_mm_homo_warp_kernel_generator,
                                    pallas=_jitc_mm_homo_pallas_kernel_generator,
                                    default='pallas', )
float_jitc_mm_homo_p.def_tpu_kernel(_jitc_mm_homo_pallas_kernel_generator)
float_jitc_mm_homo_p.def_jvp_rule2(
    _jitc_mm_homo_jvp_w,
    None,
    _jitc_mm_homo_jvp_B,
    None,
    None
)
float_jitc_mm_homo_p.def_transpose_rule(_jitc_mm_homo_transpose_rules)
float_jitc_mm_homo_p.def_batching_rule(_jitc_mm_homo_batching)
