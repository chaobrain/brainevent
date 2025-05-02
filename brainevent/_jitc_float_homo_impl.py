# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from ._compatible_import import pallas as pl
from ._config import numba_environ
from ._jitc_util import _initialize_seed, _initialize_conn_length
from ._pallas_random import LFSR88RNG
from ._typing import Data, MatrixShape
from ._xla_custom_op import XLACustomKernel, GPUKernelChoice
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_kernel
from ._xla_custom_op_pallas import PallasKernelGenerator
from ._xla_custom_op_util import general_batching_rule
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator

__all__ = [
    "float_jitc_homo_matrix",
    "float_jitc_homo_matvec",
    "float_jitc_homo_matmat",
]


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


def _jitc_homo_matrix_cpu_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    r"""
    Generate the CPU kernel for the :func:`_jitc_matvec_homo` operation.
    """

    if corder:
        # This means that the for loop is parallelized along the dimension of the output vector: ``post.shape[0]``.

        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

            def kernel(weight, clen, seed, _, posts):
                """
                CPU kernel function for generating a sparse random matrix with homogeneous weights.

                This function implements the matrix generation when transpose=True and corder=True.
                It creates a sparse matrix where non-zero entries have the same value (weight0) and
                are randomly distributed with an approximate probability of 1/clen0.

                The matrix is generated using a random sampling approach rather than explicitly storing
                all values. For each row in the output matrix, it determines the positions of non-zero
                entries using random skipping, which provides an efficient way to sample sparse connections.

                Mathematical model:
                    For each row i_row:
                        1. Start at a random column position i_col ∈ [0, clen0)
                        2. Set posts[i_row, i_col] = weight0
                        3. Skip ahead by a random interval in [1, clen0)
                        4. Repeat until i_col >= m

                This creates a sparse matrix with approximately 1/clen0 connection probability.

                Parameters
                ----------
                weight : ndarray
                    Single-element array containing the homogeneous weight value for connections
                clen : ndarray
                    Single-element array containing the connection length parameter (~1/connection_probability)
                seed : ndarray
                    Single-element array containing the random seed for reproducible matrix generation
                _ : ndarray
                    Unused placeholder parameter (required for API compatibility)
                posts : ndarray
                    Output array to store the generated matrix, shape (n, m)

                Notes
                -----
                - The matrix is generated row by row, with each row containing approximately m/clen0 non-zero entries
                - This transpose=True variant means we're generating the transpose of the regular matrix
                - When transpose=True and corder=True, we're generating rows of the transposed matrix
                """
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
                """
                CPU kernel function for generating a sparse random matrix with homogeneous weights.

                This function implements the matrix generation when transpose=False and corder=True.
                It creates a sparse matrix where non-zero entries have the same value (weight0) and
                are randomly distributed with an approximate probability of 1/clen0.

                The matrix is generated using a random sampling approach rather than explicitly storing
                all values. For each row in the output matrix, it determines the positions of non-zero
                entries using random skipping, which provides an efficient way to sample sparse connections.

                Mathematical model:
                    For each row i_row:
                        1. Start at a random column position i_col ∈ [0, clen0)
                        2. Set posts[i_row, i_col] = weight0
                        3. Skip ahead by a random interval in [1, clen0)
                        4. Repeat until i_col >= n

                This creates a sparse matrix with approximately 1/clen0 connection probability.

                Parameters
                ----------
                weight : ndarray
                    Single-element array containing the homogeneous weight value for connections
                clen : ndarray
                    Single-element array containing the connection length parameter (~1/connection_probability)
                seed : ndarray
                    Single-element array containing the random seed for reproducible matrix generation
                _ : ndarray
                    Unused placeholder parameter (required for API compatibility)
                posts : ndarray
                    Output array to store the generated matrix, shape (m, n)

                Notes
                -----
                - The matrix is generated row by row, with each row containing approximately n/clen0 non-zero entries
                - When transpose=False and corder=True, we're generating rows of the matrix directly
                - The sparsity pattern is determined by the random seed, ensuring reproducibility
                """
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
                """
                CPU kernel function for generating a sparse random matrix with homogeneous weights.

                This function implements the matrix generation when transpose=True and corder=False.
                It creates a sparse matrix where non-zero entries have the same value (weight0) and
                are randomly distributed with an approximate probability of 1/clen0.

                The matrix is generated using a column-major approach where each column is processed
                sequentially, and non-zero entries within each column are determined using random skipping.
                This approach ensures consistency between this matrix and its transpose when generated
                with transpose=False and corder=False.

                Mathematical model:
                    For each column i_col:
                        1. Start at a random row position i_row ∈ [0, clen0)
                        2. Set posts[i_row, i_col] = weight0
                        3. Skip ahead by a random interval in [1, clen0)
                        4. Repeat until i_row >= n

                This creates a sparse matrix with approximately 1/clen0 connection probability.

                Parameters
                ----------
                weight : ndarray
                    Single-element array containing the homogeneous weight value for connections
                clen : ndarray
                    Single-element array containing the connection length parameter (~1/connection_probability)
                seed : ndarray
                    Single-element array containing the random seed for reproducible matrix generation
                _ : ndarray
                    Unused placeholder parameter (required for API compatibility)
                posts : ndarray
                    Output array to store the generated matrix, shape (n, m)

                Notes
                -----
                - The matrix is generated column by column, with each column containing approximately n/clen0 non-zero entries
                - When transpose=True and corder=False, we're generating columns of the transposed matrix
                - This algorithm is complementary to the row-major approach used when corder=True
                """
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
                """
                CPU kernel function for generating a sparse random matrix with homogeneous weights.

                This function implements the matrix generation when transpose=False and corder=False.
                It creates a sparse matrix where non-zero entries have the same value (weight0) and
                are randomly distributed with an approximate probability of 1/clen0.

                The matrix is generated using a column-major approach where each column is processed
                sequentially, and non-zero entries within each column are determined using random skipping.
                This provides an efficient way to generate a sparse connectivity pattern and ensures
                consistency between this matrix and its transpose when generated with appropriate parameters.

                Mathematical model:
                    For each column i_col:
                        1. Start at a random row position i_row ∈ [0, clen0)
                        2. Set posts[i_row, i_col] = weight0
                        3. Skip ahead by a random interval in [1, clen0)
                        4. Repeat until i_row >= m

                This creates a sparse matrix with approximately 1/clen0 connection probability.

                Parameters
                ----------
                weight : ndarray
                    Single-element array containing the homogeneous weight value for connections
                clen : ndarray
                    Single-element array containing the connection length parameter (~1/connection_probability)
                seed : ndarray
                    Single-element array containing the random seed for reproducible matrix generation
                _ : ndarray
                    Unused placeholder parameter (required for API compatibility)
                posts : ndarray
                    Output array to store the generated matrix, shape (m, n)

                Notes
                -----
                - The matrix is generated column by column, with each column containing approximately m/clen0 non-zero entries
                - When transpose=False and corder=False, we're generating columns of the matrix directly
                - This approach ensures that the generated matrix is consistent with its transpose
                  when using transpose=True and corder=False
                """
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


def _jitc_homo_matrix_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    r"""
    Generate the GPU kernel for the :func:`_jitc_matvec_homo` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

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
                """
                GPU kernel function for generating a sparse random matrix with homogeneous weights.

                This function implements the matrix generation when ``transpose=True`` and ``corder=True``
                for GPU execution using Warp. It creates a sparse matrix where non-zero entries have
                the same value (``weight0``) and are randomly distributed with an approximate probability
                of ``1/clen0``.

                The sparse matrix generation uses a thread-parallel approach where each CUDA thread
                handles one column of the transposed matrix. Instead of explicitly storing all values,
                it uses random skipping to efficiently sample non-zero positions.

                Mathematical model:
                    For each column i_col (thread):
                        1. Start at a random row position i_row ∈ [0, clen0)
                        2. Set posts[i_col, i_row] = weight0
                        3. Skip ahead by a random interval in [1, clen0)
                        4. Repeat until i_row >= m

                This creates a sparse matrix with approximately ``1/clen0`` connection probability.

                Parameters
                ----------
                weight : warp.array1d
                    Single-element array containing the homogeneous weight value for connections
                clen : warp.array1d
                    Single-element array containing the connection length parameter (~1/connection_probability)
                seed : warp.array1d
                    Single-element array containing the random seed for reproducible matrix generation
                _ : warp.array2d
                    Unused placeholder parameter (required for API compatibility)
                posts : warp.array1d
                    Output array to store the generated matrix, shape (n, m) where n is implicit from threads

                Notes
                -----
                - Each CUDA thread processes one column (i_col) of the output matrix in parallel
                - The random state is initialized with the base seed plus thread ID to ensure
                  different but reproducible random sequences for each column
                - The sparsity pattern follows a geometric-like distribution for connection intervals
                """
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
                """
                GPU kernel function for generating a sparse random matrix with homogeneous weights.

                This function implements a sparse matrix generation algorithm for GPU execution using Warp.
                It creates a matrix where non-zero entries all have the same value (weight0) and are randomly
                distributed with connection probability approximately 1/clen0.

                The matrix is generated using a row-wise parallel approach where each CUDA thread
                processes one row of the output matrix. Each thread uses a deterministic random
                sampling method to determine which columns in its row should contain the weight value.

                Mathematical model:
                    For each row i_row (thread):
                        1. Initialize random state using seed0 + i_row
                        2. Start at a random column position i_col ∈ [0, clen0)
                        3. Set posts[i_row, i_col] = weight0
                        4. Skip ahead by a random interval in [1, clen0)
                        5. Repeat until i_col >= n

                This creates a sparse matrix with approximately 1/clen0 connection probability.

                Parameters
                ----------
                weight : warp.array1d
                    Single-element array containing the homogeneous weight value for connections
                clen : warp.array1d
                    Single-element array containing the connection length parameter (~1/connection_probability)
                seed : warp.array1d
                    Single-element array containing the random seed for reproducible matrix generation
                _ : warp.array2d
                    Unused placeholder parameter (required for API compatibility)
                posts : warp.array1d
                    Output array to store the generated matrix

                Notes
                -----
                - Each CUDA thread processes one row of the output matrix in parallel
                - The random state is initialized with the base seed plus thread ID to ensure
                  different but reproducible random sequences for each row
                - The sparsity pattern follows a geometric-like distribution for connection intervals
                - When using many threads, this approach efficiently generates large sparse matrices
                """
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
                """
                GPU kernel function for generating a sparse random matrix with homogeneous weights.

                This function implements the matrix generation when transpose=True and corder=False
                for GPU execution using Warp. It creates a sparse matrix where non-zero entries have
                the same value (weight0) and are randomly distributed with an approximate probability
                of 1/clen0.

                The sparse matrix is generated using a column-wise parallel approach where each CUDA
                thread processes one column of the matrix. Each thread uses a deterministic random
                sampling method to determine which rows in its column should receive the weight value.

                Mathematical model:
                    For each column i_col (thread):
                        1. Initialize random state using seed0 + i_col
                        2. Start at a random row position i_row ∈ [0, clen0)
                        3. Set posts[i_row, i_col] = weight0
                        4. Skip ahead by a random interval in [1, clen0)
                        5. Repeat until i_row >= n

                This creates a sparse matrix with approximately 1/clen0 connection probability and
                ensures consistency between this matrix and its transpose when appropriate parameters
                are used.

                Parameters
                ----------
                weight : warp.array1d
                    Single-element array containing the homogeneous weight value for connections
                clen : warp.array1d
                    Single-element array containing the connection length parameter (~1/connection_probability)
                seed : warp.array1d
                    Single-element array containing the random seed for reproducible matrix generation
                _ : warp.array2d
                    Unused placeholder parameter (required for API compatibility)
                posts : warp.array2d
                    Output array to store the generated matrix

                Notes
                -----
                - Each CUDA thread processes one column of the output matrix in parallel
                - The random state is initialized with the base seed plus thread ID to ensure
                  different but reproducible random sequences for each column
                - When transpose=True and corder=False, this kernel generates columns of the transposed matrix
                - This approach ensures that the generated matrix is consistent with its transpose
                  when using appropriate flags, improving reproducibility across operations
                """
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
                """
                GPU kernel function for generating a sparse random matrix with homogeneous weights.

                This function implements the matrix generation when transpose=False and corder=False
                for GPU execution using Warp. It creates a sparse matrix where non-zero entries have
                the same value (weight0) and are randomly distributed with an approximate probability
                of 1/clen0.

                The matrix is generated using a column-wise parallel approach where each CUDA thread
                processes one column of the matrix. Each thread uses a deterministic random sampling
                method to determine which rows in its column should receive the weight value.

                Mathematical model:
                    For each column i_col (thread):
                        1. Initialize random state using seed0 + i_col
                        2. Start at a random row position i_row ∈ [0, clen0)
                        3. Set posts[i_row, i_col] = weight0
                        4. Skip ahead by a random interval in [1, clen0)
                        5. Repeat until i_row >= m

                This creates a sparse matrix with approximately 1/clen0 connection probability and
                ensures consistency between this matrix and its transpose when appropriate parameters
                are used.

                Parameters
                ----------
                weight : warp.array1d
                    Single-element array containing the homogeneous weight value for connections
                clen : warp.array1d
                    Single-element array containing the connection length parameter (~1/connection_probability)
                seed : warp.array1d
                    Single-element array containing the random seed for reproducible matrix generation
                _ : warp.array2d
                    Unused placeholder parameter (required for API compatibility)
                posts : warp.array2d
                    Output array to store the generated matrix

                Notes
                -----
                - Each CUDA thread processes one column of the output matrix in parallel
                - The random state is initialized with the base seed plus thread ID to ensure
                  different but reproducible random sequences for each column
                - When corder=False, this approach ensures that the generated matrix is consistent with
                  its transpose when using transpose=True, improving reproducibility across operations
                - This implementation optimizes memory access patterns for GPU execution
                """
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

    return warp.kernel(kernel)


def _jitc_homo_matrix_pallas_kernel_generator(
    out_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    if corder:
        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

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
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def _raw_kernel(
                weight_ref,
                clen_ref,
                seed_ref,
                _,
                post_ref,
            ):
                n = post_ref.shape[1]  # Get number of columns in the output matrix
                weight0 = weight_ref[0]  # Homogeneous weight for all connections
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_row = pl.program_id(0)

                def body(data):
                    i_col, rng_ = data
                    post_ref[i_row, i_col] = weight0
                    i_col = i_col + rng_.random_integers(1, clen0)
                    return i_col, rng_

                rng = LFSR88RNG(seed0 + i_row)
                jax.lax.while_loop(
                    lambda data: data[0] < n,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

    else:
        if transpose:
            # JIT matrix.T
            #
            # - JIT matrix shape = [m, n]
            #

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
                    i_row = i_row + rng_.random_integers(0, clen0)
                    return i_row, rng_

                rng = LFSR88RNG(seed0 + i_col)
                jax.lax.while_loop(
                    lambda data: data[0] < n,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

                # rng = LFSR88(seed0 + i_col)
                # i_row = rng.random_integers(0, clen0)
                # while i_row < n:
                #     post_ref[i_row, i_col] = weight0
                #     i_row += rng.random_integers(0, clen0)


        else:
            # JIT matrix
            #
            # - JIT matrix shape = [m, n]
            #

            def _raw_kernel(
                weight_ref,
                clen_ref,
                seed_ref,
                _,
                post_ref,
            ):
                m = post_ref.shape[0]
                weight0 = weight_ref[0]  # Homogeneous weight for all connections
                clen0 = clen_ref[0]  # Connection length parameter (controls sparsity)
                seed0 = seed_ref[0]  # Base random seed value
                i_col = pl.program_id(0)

                def body(data):
                    i_row, rng_ = data
                    post_ref[i_row, i_col] = weight0
                    i_row = i_row + rng_.random_integers(0, clen0)
                    return i_row, rng_

                rng = LFSR88RNG(seed0 + i_col)
                jax.lax.while_loop(
                    lambda data: data[0] < m,
                    body,
                    (rng.random_integers(0, clen0), rng)
                )

                # rng = LFSR88(seed0 + i_col)
                # i_row = rng.random_integers(0, clen0)
                # while i_row < m:
                #     post_ref[i_row, i_col] = weight0
                #     i_row += rng.random_integers(0, clen0)

    dim = out_info.shape[0] if corder else out_info.shape[1]

    def kernel(weight, clen, seed, out):
        fn = pl.pallas_call(
            _raw_kernel,
            out_shape=jax.ShapeDtypeStruct(out.shape, out.dtype),
            grid=(dim,),
            input_output_aliases={3: 0},
        )
        return [fn(weight, clen, seed, out)]

    return kernel


def _jitc_homo_matrix_batching(
    args,
    axes,
    **kwargs
):
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
        return general_batching_rule(
            float_jitc_homo_matrix_p,
            args,
            axes,
            **kwargs,
        )


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
float_jitc_homo_matrix_p.def_cpu_kernel(NumbaKernelGenerator(_jitc_homo_matrix_cpu_kernel_generator))
float_jitc_homo_matrix_p.def_gpu_kernel(
    GPUKernelChoice(
        default='warp',
        warp_kernel=WarpKernelGenerator(
            _jitc_homo_matrix_gpu_kernel_generator,
            dim=lambda out_info, corder, **kwargs: out_info.shape[0] if corder else out_info.shape[1],
            input_output_aliases={3: 0}
        ),
        pallas_kernel=PallasKernelGenerator(_jitc_homo_matrix_pallas_kernel_generator)
    )
)
float_jitc_homo_matrix_p.def_tpu_kernel(PallasKernelGenerator(_jitc_homo_matrix_pallas_kernel_generator))
float_jitc_homo_matrix_p.def_batching_rule(_jitc_homo_matrix_batching)


# Kernel generators for JIT connection SPMV

def _jitc_mv_homo_cpu_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_homo` operation.
    """

    if corder:
        # This means that the for loop is parallelized along the dimension of the output vector: ``post.shape[0]``.

        if transpose:
            @numba_environ.jit_fn
            def kernel(weight, clen, vector, seed, _, posts):
                """
                Numba kernel implementation for matrix-vector multiplication where the matrix
                is generated on-the-fly with homogeneous weights.

                This kernel implements a vector-matrix multiplication where the matrix has a homogeneous weight
                value and is sparsely connected with probability `clen`. Instead of generating the entire
                matrix, connections are sampled using binomial distribution to identify non-zero entries.

                The kernel is optimized for the case where `transpose=True` and `corder=True`, meaning
                it represents the operation: vector @ matrix (or matrix.T @ vector if we consider the original matrix).

                Parameters
                ----------
                weight : ndarray
                    A scalar value stored as a single-element array, representing the homogeneous weight
                    for all connections in the matrix.
                clen : ndarray
                    A scalar value stored as a single-element array, representing the connection probability.
                    The value is approximately 2/p where p is the connection probability.
                vector : ndarray
                    The input vector to be multiplied with the randomly generated matrix.
                seed : ndarray
                    A scalar value stored as a single-element array, used as a seed for random number generation.
                _ : ndarray
                    Placeholder parameter (not used).
                posts : ndarray
                    Output array where the result will be stored.

                Notes
                -----
                The algorithm uses a sparse sampling approach to avoid explicitly generating the entire matrix:
                1. For each output column, it samples connected rows using binomial distribution
                2. Only the connected entries contribute to the output sum
                3. The final result is scaled by the weight value

                Implementation details:
                - The algorithm performs sparse matrix operations by randomly determining which entries contribute
                - For efficiency, we don't explicitly generate the matrix but directly sample connections
                - Each output element is computed by summing only the connected input vector elements
                - The algorithm uses a skip-ahead random sampling approach to find connected elements
                """
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
            @numba_environ.jit_fn
            def kernel(weight, clen, vector, seed, _, posts):
                """
                Numba kernel implementation for matrix-vector multiplication where the matrix
                is generated on-the-fly with homogeneous weights.

                This kernel implements a matrix-vector multiplication where the matrix has a homogeneous weight
                value and is sparsely connected with probability `clen`. Instead of generating the entire
                matrix, connections are sampled using binomial distribution to identify non-zero entries.

                The kernel is optimized for the case where `transpose=False` and `corder=True`, meaning
                it represents the operation: matrix @ vector (standard matrix-vector product).

                Parameters
                ----------
                weight : ndarray
                    A scalar value stored as a single-element array, representing the homogeneous weight
                    for all connections in the matrix.
                clen : ndarray
                    A scalar value stored as a single-element array, representing the connection probability.
                    It's approximately 2/p where p is the connection probability, determining how sparse the matrix is.
                vector : ndarray
                    The input vector to be multiplied with the randomly generated matrix.
                seed : ndarray
                    A scalar value stored as a single-element array, used as a seed for random number generation
                    to ensure reproducibility.
                _ : ndarray
                    Placeholder parameter (not used in this implementation).
                posts : ndarray
                    Output array where the result will be stored. Its shape determines the number of rows
                    in the implicit matrix.

                Notes
                -----
                The algorithm uses a sparse sampling approach to avoid explicitly generating the entire matrix:
                1. For each output row, it samples connected columns using random skipping
                2. Only the connected entries contribute to the output sum
                3. The final result is scaled by the weight value

                Implementation details:
                - Connection probability is controlled by clen - larger values produce sparser matrices
                - The sampling approach avoids storing the matrix which would require O(rows×cols) memory
                - Time complexity is O(rows × average_connections_per_row), which is much more efficient
                  for very sparse matrices than the O(rows × cols) of standard matrix multiplication
                - This algorithm can be viewed as stochastic matrix generation with fixed weight values
                """
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
            @numba_environ.jit_fn
            def kernel(weight, clen, vector, seed, _, posts):
                """
                Numba kernel implementation for matrix-vector multiplication where the matrix
                is generated on-the-fly with homogeneous weights.

                This kernel implements a vector-matrix multiplication where the matrix has a homogeneous weight
                value and is sparsely connected with probability ~1/clen. Instead of generating the entire
                matrix, connections are sampled using random skipping to efficiently identify non-zero entries.

                The kernel is optimized for the case where `transpose=True` and `corder=False`, meaning
                it processes the input vector elements one by one, accumulating their contributions to the output vector.
                This approach is particularly efficient for very sparse matrices.

                Parameters
                ----------
                weight : ndarray
                    A scalar value stored as a single-element array, representing the homogeneous weight
                    for all connections in the matrix.
                clen : ndarray
                    A scalar value stored as a single-element array, representing the inverse of connection probability.
                    Larger values result in sparser matrices (~1/clen is the connection probability).
                vector : ndarray
                    The input vector to be multiplied with the randomly generated matrix.
                seed : ndarray
                    A scalar value stored as a single-element array, used as a seed for random number generation
                    to ensure reproducible results.
                _ : ndarray
                    Placeholder parameter (not used in this implementation).
                posts : ndarray
                    Output array where the result will be stored. Its length determines the number of columns
                    in the implicit matrix.

                Notes
                -----
                The algorithm uses a row-based approach for vector @ matrix multiplication:
                1. For each input row (vector element), it applies the weight and samples connected columns
                2. Each row contributes to multiple columns based on the sparse connectivity pattern
                3. The algorithm efficiently skips zeros in the matrix using geometric-like distribution sampling

                Time complexity is O(num_rows × average_connections_per_row) rather than O(num_rows × num_cols)
                of standard matrix multiplication, making this approach much more efficient for sparse matrices.
                """
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
            @numba_environ.jit_fn
            def kernel(weight, clen, vector, seed, _, posts):
                """
                Numba kernel implementation for matrix-vector multiplication where the matrix
                is generated on-the-fly with homogeneous weights.

                This kernel implements a matrix-vector multiplication where the matrix has a homogeneous weight
                value and is sparsely connected with probability ~1/clen. Instead of generating the entire
                matrix, connections are sampled using random skipping to efficiently identify non-zero entries.

                The kernel is optimized for the case where `transpose=False` and `corder=False`, meaning
                it processes each input vector element (column) separately and accumulates its contributions to
                all connected rows in the output vector. This is particularly efficient for sparse matrices.

                Parameters
                ----------
                weight : ndarray
                    A scalar value stored as a single-element array, representing the homogeneous weight
                    for all connections in the matrix.
                clen : ndarray
                    A scalar value stored as a single-element array, representing the inverse of connection probability.
                    Larger values result in sparser matrices (~1/clen is the connection probability).
                vector : ndarray
                    The input vector to be multiplied with the randomly generated matrix.
                seed : ndarray
                    A scalar value stored as a single-element array, used as a seed for random number generation
                    to ensure reproducible results.
                _ : ndarray
                    Placeholder parameter (not used in this implementation).
                posts : ndarray
                    Output array where the result will be stored. Its shape determines the number of rows
                    in the implicit matrix.

                Notes
                -----
                The algorithm uses a column-centric approach for matrix @ vector multiplication:
                1. For each input element, it pre-multiplies by weight and samples connected rows
                2. Each column contributes to multiple rows based on the sparse connectivity pattern
                3. The algorithm efficiently skips zeros in the matrix using geometric-like distribution sampling

                Time complexity is O(num_cols × average_connections_per_col) rather than O(num_rows × num_cols)
                of standard matrix multiplication, making this approach much more efficient for sparse matrices.
                """
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


def _jitc_mv_homo_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    r"""
    Generate the GPU kernel for the :func:`_jitc_matvec_homo` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    v_dtype = dtype_to_warp_type(vector_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

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
                """
                WARP GPU kernel for sparse matrix-vector multiplication with on-the-fly matrix generation.

                Implements vector-matrix product where the matrix is generated implicitly with
                sparse random connectivity and homogeneous weight values. This kernel handles
                the transpose=True, corder=True case.

                Each GPU thread processes one output element (column of the transposed matrix).
                The kernel efficiently samples connected entries using a random skipping approach
                to avoid generating the full matrix.

                Parameters
                ----------
                weight : warp.array1d
                    Single-element array containing the homogeneous weight value
                clen : warp.array1d
                    Single-element array with connection length parameter (~2/connection_probability)
                vector : warp.array1d
                    Input vector to be multiplied with the implicit matrix
                seed : warp.array1d
                    Single-element array with random seed for reproducible matrix generation
                _ : warp.array1d
                    Unused placeholder parameter (required for API compatibility)
                posts : warp.array1d
                    Output array where results are stored, shape determines output dimension

                Notes
                -----
                The algorithm:
                1. Each thread computes one element of the output vector
                2. Uses thread ID to ensure parallel processing across output elements
                3. Samples connected entries using random state initialized with seed+thread_id
                4. Accumulates contributions from input vector elements at sampled positions
                5. Finally scales by the weight value and stores in output
                """
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
                """
                WARP GPU kernel for sparse matrix-vector multiplication with on-the-fly matrix generation.

                Implements matrix-vector product where the matrix is generated implicitly with
                sparse random connectivity and homogeneous weight values. This kernel handles
                the transpose=False, corder=True case.

                Each GPU thread processes one output element (row of the matrix). The kernel
                efficiently samples connected entries using a random skipping approach to avoid
                generating the full matrix.

                Parameters
                ----------
                weight : warp.array1d
                    Single-element array containing the homogeneous weight value for all connections
                clen : warp.array1d
                    Single-element array containing connection length parameter (~2/connection_probability)
                vector : warp.array1d
                    Input vector to be multiplied with the implicit matrix
                seed : warp.array1d
                    Single-element array with random seed for reproducible matrix generation
                _ : warp.array1d
                    Unused placeholder parameter (required for API compatibility)
                posts : warp.array1d
                    Output array where results are stored, shape determines output dimension

                Notes
                -----
                The algorithm:
                1. Each thread computes one element of the output vector (one matrix row)
                2. Thread ID identifies which row this thread is processing
                3. Uses random sampling with geometric-like skipping to find connected columns
                4. Accumulates weighted values from the input vector at connected positions
                5. Finally applies the weight and stores result in the output array
                """
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
                """
                WARP GPU kernel for sparse matrix-vector multiplication with on-the-fly matrix generation.

                Implements vector-matrix product where the matrix is generated implicitly with
                sparse random connectivity and homogeneous weight values. This kernel handles
                the transpose=True, corder=False case.

                Each GPU thread processes one input element (row) and distributes its contribution
                to all connected output elements. This approach is efficient for sparse matrices
                as it avoids generating the full matrix.

                Parameters
                ----------
                weight : warp.array1d
                    Single-element array containing the homogeneous weight value for all connections
                clen : warp.array1d
                    Single-element array containing connection length parameter (~2/connection_probability)
                vector : warp.array1d
                    Input vector to be multiplied with the implicit matrix
                seed : warp.array1d
                    Single-element array with random seed for reproducible matrix generation
                _ : warp.array1d
                    Unused placeholder parameter (required for API compatibility)
                posts : warp.array1d
                    Output array where results are stored, shape determines output dimension

                Notes
                -----
                The algorithm:
                1. Each thread processes one input element (one row in the matrix)
                2. Uses thread ID to determine which input element to process
                3. Pre-multiplies the input value by weight for efficiency
                4. Samples connected output positions using geometric-like distribution
                5. Updates the corresponding output elements using atomic additions
                6. This row-based approach efficiently handles sparse connectivity patterns
                """
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
                """
                WARP GPU kernel for sparse matrix-vector multiplication with on-the-fly matrix generation.

                Implements matrix-vector product where the matrix is generated implicitly with
                sparse random connectivity and homogeneous weight values. This kernel handles
                the transpose=False, corder=False case.

                Each GPU thread processes one input element (column) and distributes its contribution
                to all connected output elements. This approach is efficient for sparse matrices
                as it avoids generating the full matrix.

                Parameters
                ----------
                weight : warp.array1d
                    Single-element array containing the homogeneous weight value for all connections
                clen : warp.array1d
                    Single-element array containing connection length parameter (~2/connection_probability)
                vector : warp.array1d
                    Input vector to be multiplied with the implicit matrix
                seed : warp.array1d
                    Single-element array with random seed for reproducible matrix generation
                _ : warp.array1d
                    Unused placeholder parameter (required for API compatibility)
                posts : warp.array1d
                    Output array where results are stored, shape determines output dimension

                Notes
                -----
                The algorithm:
                1. Each thread processes one input element (one column in the matrix)
                2. Uses thread ID to determine which input element to process
                3. Pre-multiplies the input value by weight for efficiency
                4. Samples connected output positions using geometric-like distribution
                5. Updates the corresponding output elements using atomic additions
                6. This column-based approach efficiently handles sparse connectivity patterns
                """
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

    return warp.kernel(kernel)


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
        return general_batching_rule(
            float_jitc_mv_homo_p,
            args,
            axes,
            **kwargs,
        )


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


def _mv_gpu_kernel():
    return


float_jitc_mv_homo_p = XLACustomKernel('float_jitc_mv_homo')
float_jitc_mv_homo_p.def_cpu_kernel(NumbaKernelGenerator(_jitc_mv_homo_cpu_kernel_generator))
float_jitc_mv_homo_p.def_gpu_kernel(
    GPUKernelChoice(
        default='warp',
        warp_kernel=WarpKernelGenerator(
            _jitc_mv_homo_gpu_kernel_generator,
            dim=lambda out_info, vector_info, corder, **kwargs: (out_info.shape[0] if corder else vector_info.shape[0]),
            input_output_aliases={4: 0}
        )
    )
)
float_jitc_mv_homo_p.def_jvp_rule2(
    _jitc_mv_homo_jvp_weights,
    None,
    _jitc_mv_homo_jvp_v,
    None,
    None
)
float_jitc_mv_homo_p.def_transpose_rule(_jitc_mv_homo_transpose_rules)
float_jitc_mv_homo_p.def_batching_rule(_jitc_mv_homo_batching)


# Kernel generators for JIT connection SPMM

# jitc csrmm homo

def _jitc_mm_homo_cpu_kernel_generator(
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    r"""
    Generate the CPU kernel for the :func:`_jitc_matmat_homo` operation.
    """

    if corder:

        if transpose:
            # JIT Matrix.T @ B
            #
            # - JIT matrix: [k, m]
            # - B: [k, n]

            def kernel(weight, clen, B, seed, _, posts):
                r"""
                Numba kernel for sparse matrix-matrix multiplication with on-the-fly matrix generation.

                Implements the operation M^T @ B where M is a sparse matrix with homogeneous weight values
                generated just-in-time during computation. Instead of storing the full matrix, this function
                uses a probabilistic approach to sample connections for each output row.

                This kernel handles the transpose=True, corder=True case, processing each output row
                in sequence. This design enables efficient multiplication with very large sparse matrices
                that would be impractical to store in memory.

                The mathematical operation performed is:

                y_ij = \sum_{k} M_{ki} * B_{kj}

                Where M is implicitly defined with connection probability ~1/clen0.

                Parameters
                ----------
                weight : array_like
                    Single-element array containing the homogeneous weight value for all connections
                clen : array_like
                    Single-element array containing the connection length parameter (~2/connection_probability)
                B : ndarray
                    Input matrix to multiply with the transposed implicit sparse matrix, shape (k, n)
                seed : array_like
                    Single-element array with random seed for reproducible matrix generation
                _ : ndarray
                    Unused placeholder parameter (required for API compatibility)
                posts : ndarray
                    Output array where results are stored, shape (m, n)

                Notes
                -----
                The algorithm:
                1. For each output row i_m, initialize a zero vector of length n
                2. Sample connections to input rows (i_k) using geometric-like skipping
                3. For each sampled connection, add the corresponding row of B to the output
                4. Finally scale the accumulated sum by the weight value
                5. This row-wise approach is memory efficient for sparse connectivity patterns
                """
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
                r"""
                Numba kernel for sparse matrix-matrix multiplication with on-the-fly matrix generation.

                Implements the operation M @ B where M is a sparse matrix with homogeneous weight values
                generated just-in-time during computation. Instead of storing the full matrix, this function
                uses a probabilistic approach to sample connections for each output row.

                This kernel handles the transpose=False, corder=True case, processing each output row
                in sequence. This design enables efficient multiplication with very large sparse matrices
                that would be impractical to store in memory.

                The mathematical operation performed is:

                y_ij = \sum_{k} M_{ik} * B_{kj}

                Where M is implicitly defined with connection probability ~1/clen0.

                Parameters
                ----------
                weight : array_like
                    Single-element array containing the homogeneous weight value for all connections
                clen : array_like
                    Single-element array containing the connection length parameter (~2/connection_probability)
                B : ndarray
                    Input matrix to multiply with the implicit sparse matrix, shape (k, n)
                seed : array_like
                    Single-element array with random seed for reproducible matrix generation
                _ : ndarray
                    Unused placeholder parameter (required for API compatibility)
                posts : ndarray
                    Output array where results are stored, shape (m, n)

                Notes
                -----
                The algorithm:
                1. For each output row i_m, initialize a zero vector of length n
                2. Sample connections to input rows (i_k) using geometric-like skipping
                3. For each sampled connection, add the corresponding row of B to the output
                4. Finally scale the accumulated sum by the weight value
                5. This row-wise approach is memory efficient for sparse connectivity patterns
                """
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
                r"""
                Numba kernel for sparse matrix-matrix multiplication with on-the-fly matrix generation.

                Implements the operation M^T @ B where M is a sparse matrix with homogeneous weight values
                generated just-in-time during computation. Instead of storing the full matrix, this function
                uses a probabilistic approach to sample connections for each input row.

                This kernel handles the transpose=True, corder=False case, processing each input row
                in sequence. This approach ensures that the generated M^T is consistent with the M that would
                be generated with transpose=False, at the potential cost of reduced parallelism.

                The mathematical operation performed is:

                y_ij = \sum_{k} M_{ki} * B_{kj}

                Where M is implicitly defined with connection probability ~1/clen0.

                Parameters
                ----------
                weight : array_like
                    Single-element array containing the homogeneous weight value for all connections
                clen : array_like
                    Single-element array containing the connection length parameter (~2/connection_probability)
                B : ndarray
                    Input matrix to multiply with the transposed implicit sparse matrix, shape (k, n)
                seed : array_like
                    Single-element array with random seed for reproducible matrix generation
                _ : ndarray
                    Unused placeholder parameter (required for API compatibility)
                posts : ndarray
                    Output array where results are stored, shape (m, n)

                Notes
                -----
                The algorithm:
                1. For each input row i_k, pre-scale the row from B by the weight value
                2. Sample connections to output rows (i_m) using geometric-like skipping
                3. For each sampled connection, add the weighted row to the corresponding output row
                4. This transpose-compatible approach ensures consistency between M and M^T operations
                """
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
                r"""
                Numba kernel for sparse matrix-matrix multiplication with on-the-fly matrix generation.

                Implements the operation M @ B where M is a sparse matrix with homogeneous weight values
                generated just-in-time during computation. Instead of storing the full matrix, this function
                uses a probabilistic approach to sample connections for each input column.

                This kernel handles the transpose=False, corder=False case, processing each input
                column in sequence. This approach ensures that the generated matrix is consistent with its
                transpose when using transpose=True, improving reproducibility at the potential cost of
                reduced parallelism.

                The mathematical operation performed is:

                y_ij = \sum_{k} M_{ik} * B_{kj}

                Where M is implicitly defined with connection probability ~1/clen0.

                Parameters
                ----------
                weight : array_like
                    Single-element array containing the homogeneous weight value for all connections
                clen : array_like
                    Single-element array containing the connection length parameter (~2/connection_probability)
                B : ndarray
                    Input matrix to multiply with the implicit sparse matrix, shape (k, n)
                seed : array_like
                    Single-element array with random seed for reproducible matrix generation
                _ : ndarray
                    Unused placeholder parameter (required for API compatibility)
                posts : ndarray
                    Output array where results are stored, shape (m, n)

                Notes
                -----
                The algorithm:
                1. For each input column i_k, pre-scale the row from B by the weight value
                2. Sample connections to output rows (i_m) using geometric-like skipping
                3. For each sampled connection, add the weighted input to the corresponding output row
                4. This approach processes inputs sequentially and distributes each to multiple outputs
                """
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


def _jitc_mm_homo_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    TITLE_SIZE: int,
    transpose: bool = False,
    corder: bool = True,
    **kwargs
):
    r"""
    Generate the GPU kernel for the :func:`_jitc_matmat_homo` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    B_dtype = dtype_to_warp_type(B_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

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

    kernel = warp.kernel(kernel)
    return kernel


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
        dw = jnp.sum(r * B, keepdims=True)
        return dw, clen, B, seed, _

    else:
        raise NotImplementedError(
            'Transpose rules for jitc_matmat_homo not implemented for '
            'non-undefined primals.'
        )


def _batching_axis0(args, axes, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    batch_size, m, n = args[2].shape
    B = jnp.transpose(args[2], (1, 0, 2)).reshape(m, batch_size * n)
    r = float_jitc_mm_homo_p_call(
        args[0],
        args[1],
        B,
        args[3],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
    return [r], [1]


def _jitc_mm_homo_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        return _batching_axis0(args, axes, **kwargs)

    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_axis0(args, axes, **kwargs)

    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (2, 0, 1))
        return _batching_axis0(args, axes, **kwargs)

    else:
        return general_batching_rule(
            float_jitc_mm_homo_p,
            args,
            axes,
            **kwargs,
        )


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
float_jitc_mm_homo_p.def_cpu_kernel(NumbaKernelGenerator(_jitc_mm_homo_cpu_kernel_generator))
float_jitc_mm_homo_p.def_gpu_kernel(
    GPUKernelChoice(
        default='warp',
        warp_kernel=WarpKernelGenerator(
            _jitc_mm_homo_gpu_kernel_generator,
            tile=lambda out_info, B_info, corder, **kwargs: (out_info.shape[0] if corder else B_info.shape[0]),
            block_dim=256,
            input_output_aliases={4: 0}
        ),
    )
)
float_jitc_mm_homo_p.def_jvp_rule2(
    _jitc_mm_homo_jvp_w,
    None,
    _jitc_mm_homo_jvp_B,
    None,
    None
)
float_jitc_mm_homo_p.def_transpose_rule(_jitc_mm_homo_transpose_rules)
float_jitc_mm_homo_p.def_batching_rule(_jitc_mm_homo_batching)
