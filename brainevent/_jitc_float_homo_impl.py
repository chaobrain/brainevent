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

from ._jitc_util import _initialize_seed, _initialize_conn_length
from ._typing import Kernel, Data, MatrixShape
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator

__all__ = [
    "jitc_matvec_homo",
    "jitc_matmat_homo",
]


def jitc_matvec_homo(
    weight: Data,
    conn_prob: float,
    v: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@v` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.

    .. warning::

        This API may change in the future.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).

        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.

    Parameters
    ----------
    weight: Array, ndarray, Quantity, float
        The value of the random matrix.
    conn_prob: float
        The connection probability.
    v: Array, ndarray, Quantity
        The vector.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    outdim_parallel: bool
        Perform the parallel random generations along the out dimension or not.
        It can be used to set the just-in-time generated :math:M^T: is the same
        as the just-in-time generated :math:`M` when ``transpose=True``.

    Returns
    -------
    out: Array, ndarray, Quantity
        The output of :math:`y = M @ v`.
    """
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    v, unitv = u.split_mantissa_unit(v)
    clen = _initialize_conn_length(conn_prob)
    res = jitc_mv_homo_p_call(
        weight,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


def jitc_matmat_homo(
    weight: Data,
    conn_prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@B` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.
    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.
    .. warning::
        This API may change in the future.
    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@B`.
    .. note::
        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.
    Parameters
    ----------
    weight: Array, ndarray, Quantity, float
        The value of the random matrix.
    conn_prob: float
        The connection probability.
    B: Array, ndarray, Quantity
        The matrix.
    seed: int
        The random number generation seed.
    shape: tuple of int
        The matrix shape.
    transpose: bool
        Transpose the random matrix or not.
    outdim_parallel: bool
        Perform the parallel random generations along the out dimension or not.
        It can be used to set the just-in-time generated :math:M^T: is the same
        as the just-in-time generated :math:`M` when ``transpose=True``.
    Returns
    -------
    out: Array, ndarray
        The output of :math:`y = M @ B`.
    """
    seed = _initialize_seed(seed)
    weight, unitd = u.split_mantissa_unit(weight)
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(conn_prob)
    res = jitc_mm_homo_p_call(
        weight,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


# Kernel generators for JIT connection SPMV

def _jitc_mv_homo_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_homo` operation.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # This means that the for loop is parallelized along the dimension of the output vector: ``post.shape[0]``.

        if transpose:
            @numba.njit(**numba_environ.setting)
            def kernel(weight, clen, vector, seed, _, posts):
                """
                Numba kernel implementation for matrix-vector multiplication where the matrix
                is generated on-the-fly with homogeneous weights.

                This kernel implements a vector-matrix multiplication where the matrix has a homogeneous weight
                value and is sparsely connected with probability `clen`. Instead of generating the entire
                matrix, connections are sampled using binomial distribution to identify non-zero entries.

                The kernel is optimized for the case where `transpose=True` and `outdim_parallel=True`, meaning
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
            @numba.njit(**numba_environ.setting)
            def kernel(weight, clen, vector, seed, _, posts):
                """
                Numba kernel implementation for matrix-vector multiplication where the matrix
                is generated on-the-fly with homogeneous weights.

                This kernel implements a matrix-vector multiplication where the matrix has a homogeneous weight
                value and is sparsely connected with probability `clen`. Instead of generating the entire
                matrix, connections are sampled using binomial distribution to identify non-zero entries.

                The kernel is optimized for the case where `transpose=False` and `outdim_parallel=True`, meaning
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
            @numba.njit(**numba_environ.setting)
            def kernel(weight, clen, vector, seed, _, posts):
                """
                Numba kernel implementation for matrix-vector multiplication where the matrix
                is generated on-the-fly with homogeneous weights.

                This kernel implements a vector-matrix multiplication where the matrix has a homogeneous weight
                value and is sparsely connected with probability ~1/clen. Instead of generating the entire
                matrix, connections are sampled using random skipping to efficiently identify non-zero entries.

                The kernel is optimized for the case where `transpose=True` and `outdim_parallel=False`, meaning
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
            @numba.njit(**numba_environ.setting)
            def kernel(weight, clen, vector, seed, _, posts):
                """
                Numba kernel implementation for matrix-vector multiplication where the matrix
                is generated on-the-fly with homogeneous weights.

                This kernel implements a matrix-vector multiplication where the matrix has a homogeneous weight
                value and is sparsely connected with probability ~1/clen. Instead of generating the entire
                matrix, connections are sampled using random skipping to efficiently identify non-zero entries.

                The kernel is optimized for the case where `transpose=False` and `outdim_parallel=False`, meaning
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
    return kernel


def _jitc_mv_homo_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    v_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matvec_homo` operation.
    """
    import warp

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    v_dtype = dtype_to_warp_type(v_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:

        if transpose:
            @warp.kernel
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
                the transpose=True, outdim_parallel=True case.

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
            @warp.kernel
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
                the transpose=False, outdim_parallel=True case.

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
                i_col = warp.randu(state, 0, clen0)

                # Process all connected entries for this output element (row)
                while i_col < num_col:
                    # Add contribution from the current connected element
                    r += vector[i_col]

                    # Skip ahead to next connected column using geometric-like distribution
                    # This creates sparse connectivity with ~1/clen0 connection probability
                    i_col += warp.randu(state, 1, clen0)

                # Scale accumulated sum by weight and store in output array
                posts[i_row] = r * weight0
    else:

        if transpose:
            @warp.kernel
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
                the transpose=True, outdim_parallel=False case.

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

            @warp.kernel
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
                the transpose=False, outdim_parallel=False case.

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
    outdim_parallel,
    **kwargs
):
    return [
        jitc_matvec_homo(
            weight,
            clen,
            v_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


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
    outdim_parallel,
    **kwargs
):
    return jitc_mv_homo_p_call(
        w_dot,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
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
    outdim_parallel,
    **kwargs
):
    assert not ad.is_undefined_primal(weight)
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(vector)

    r = jitc_mv_homo_p_call(
        weight,
        clen,
        ct,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return weight, clen, r, seed, _


def _jitc_mv_homo_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            args[2].T,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven COO matrix-vector product.")


def jitc_mv_homo_p_call(
    weight,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weight.dtype)
    )

    return jitc_mv_homo_p(
        weight,
        clen,
        vector,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        v_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mv_homo_p = XLACustomKernel(
    'jitc_mv_homo',
    cpu_kernel=NumbaKernelGenerator(_jitc_mv_homo_cpu_kernel_generator, input_output_aliases={4: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mv_homo_gpu_kernel_generator,
        dim=lambda v_info, out_info, outdim_parallel, **kwargs: (
            out_info.shape[0] if outdim_parallel else
            v_info.shape[0]
        ),
        input_output_aliases={4: 0}
    )
)

jitc_mv_homo_p.defjvp(_jitc_mv_homo_jvp_weights, None, _jitc_mv_homo_jvp_v)
jitc_mv_homo_p.def_transpose_rule(_jitc_mv_homo_transpose_rules)
jitc_mv_homo_p.def_batching_rule(_jitc_mv_homo_batching)


# Kernel generators for JIT connection SPMM

# jitc csrmm homo

def _jitc_mm_homo_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matmat_homo` operation.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(weight, clen, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_rows):
                for i_col in range(num_cols):
                    connections = np.random.binomial(1, clen0, num_cols)

                    r = np.sum(B[i_row, :] * connections)
                    posts[i_row, i_col] = r * weight0

    else:
        # outdim_parallel=False
        # TODO: more checks on this kernel (random generation method)
        def kernel(weight, clen, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_cols):
                for i_row in range(num_rows):
                    r = B[i_row, :] * weight0
                    connections = np.random.binomial(1, clen0, num_cols)

                    posts += connections[:, np.newaxis] * r

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mm_homo_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matmat_homo` operation.
    """
    import warp

    @warp.func
    def _binomial_n1(state: warp.uint32, p: float) -> int:
        """
        Draw samples from a binomial distribution.
        """
        return 1 if warp.randf(state) < p else 0

    weight_dtype = dtype_to_warp_type(weight_info.dtype)
    clen_dtype = dtype_to_warp_type(clen_info.dtype)
    B_dtype = dtype_to_warp_type(B_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            weight: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            # num_rows, num_cols = posts.shape
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)

            while cursor < num_cols:
                if _binomial_n1(state, clen0) == 1:
                    r += B[i_row, cursor]
                cursor += 1
            posts[i_row, i_col] = r * weight0
    else:
        # outdim_parallel=False
        def kernel(
            weight: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)

            while cursor < num_cols:
                # posts[cursor, :] += r
                if _binomial_n1(state, clen0) == 1:
                    for j in range(posts.shape[1]):
                        posts[cursor, j] += B[i_row, j] * weight0
                cursor += 1

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
    outdim_parallel,
    **kwargs
):
    return jitc_mm_homo_p_call(
        w_dot,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
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
    outdim_parallel,
    **kwargs
):
    return [
        jitc_matmat_homo(
            weight,
            clen,
            B_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel,
        )
    ]


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
    outdim_parallel,
    **kwargs
):
    assert not ad.is_undefined_primal(weight)
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(B)

    r = jitc_mv_homo_p_call(
        weight,
        clen,
        ct,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )[0]

    return weight, clen, r, seed, _


def _jitc_mm_homo_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = jnp.transpose(args[2], (1, 0, 2)).reshape(m, batch_size * n)
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            B,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = args[2].reshape(m, batch_size * n)
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            B,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = args[2].reshape(m, batch_size * n)
        r = jitc_mm_homo_p_call(
            args[0],
            args[1],
            B,
            args[3],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape, batch_size, n])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for JIT connection CSR matrix-matrix product")


def jitc_mm_homo_p_call(
    weight,
    clen,
    B,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    weight = jnp.atleast_1d(weight)
    clen = jnp.atleast_1d(clen)

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], weight.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], weight.dtype)
    )

    return jitc_mm_homo_p(
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
        outdim_parallel=outdim_parallel,
    )


jitc_mm_homo_p = XLACustomKernel(
    'jitc_mm_homo',
    cpu_kernel=NumbaKernelGenerator(_jitc_mm_homo_cpu_kernel_generator, input_output_aliases={4: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mm_homo_gpu_kernel_generator,
        dim=lambda out_info, **kwargs: (
            out_info.shape[0], out_info.shape[1]
        ),
        input_output_aliases={4: 0}
    )
)

jitc_mm_homo_p.defjvp(_jitc_mm_homo_jvp_w, None, None, _jitc_mm_homo_jvp_B)
jitc_mm_homo_p.def_transpose_rule(_jitc_mm_homo_transpose_rules)
jitc_mm_homo_p.def_batching_rule(_jitc_mm_homo_batching)
