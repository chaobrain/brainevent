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

from ._typing import Kernel, Data, MatrixShape
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator

__all__ = [
    "jitc_matvec_homo",
    "jitc_matvec_uniform",
    "jitc_matvec_normal",
    "jitc_matmat_homo",
    "jitc_matmat_uniform",
    "jitc_matmat_normal",
]


def _initialize_seed(seed=None):
    """Initialize a random seed for JAX operations.

    This function ensures a consistent format for random seeds used in JAX operations.
    If no seed is provided, it generates a random integer between 0 and 10^8 at compile time,
    ensuring reproducibility within compiled functions.

    Parameters
    ----------
    seed : int or array-like, optional
        The random seed to use. If None, a random seed is generated.

    Returns
    -------
    jax.Array
        A JAX array containing the seed value(s) with int32 dtype, ensuring it's
        in a format compatible with JAX random operations.

    Notes
    -----
    The function uses `jax.ensure_compile_time_eval()` to guarantee that random 
    seed generation happens during compilation rather than during execution when 
    no seed is provided, which helps maintain consistency across multiple calls
    to a JIT-compiled function.
    """
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), (1,))
    return jnp.asarray(jnp.atleast_1d(seed), dtype=jnp.int32)


def _initialize_conn_length(conn_prob: float):
    """
    Convert connection probability to connection length parameter for sparse matrix generation.

    This function transforms a connection probability (proportion of non-zero entries)
    into a connection length parameter used by the sparse sampling algorithms.
    The connection length is approximately the inverse of the connection probability,
    scaled by a factor of 2 to ensure adequate sparsity in the generated matrices.

    The function ensures the calculation happens at compile time when used in JIT-compiled
    functions by using JAX's compile_time_eval context.

    Parameters
    ----------
    conn_prob : float
        The connection probability (between 0 and 1) representing the fraction
        of non-zero entries in the randomly generated matrix.

    Returns
    -------
    jax.Array
        A JAX array containing the connection length value as an int32,
        which is approximately 2/conn_prob.

    Notes
    -----
    The connection length parameter is used in the kernels to determine the
    average distance between sampled connections when generating sparse matrices.
    Larger values result in sparser matrices (fewer connections).
    """
    with jax.ensure_compile_time_eval():
        clen = jnp.ceil(1 / conn_prob) * 2
        clen = jnp.asarray(clen, dtype=jnp.int32)
    return clen


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


def jitc_matvec_uniform(
    w_low: Data,
    w_high: Data,
    conn_prob: float,
    v: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@v` operation,
    where :math:`M` is just-in-time randomly generated with a uniform distribution for its value.

    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.

    .. warning::

        This API may change in the future.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is sampled from a uniform
    distrubtion within the range of `w_low` and `w_high`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).

        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.

    Parameters
    ----------
    w_low: Array, ndarray, Quantity, float
        The lower boundary of the random matrix.
    w_high: Array, ndarray, Quantity, float
        The upper boundary of the random matrix.
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
    out: Array, ndarray
        The output of :math:`y = M @ v`.
    """
    seed = _initialize_seed(seed)
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    w_low, unit_w_low = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unit_w_low).mantissa
    v, unitv = u.split_mantissa_unit(v)
    clen = _initialize_conn_length(conn_prob)
    res = jitc_mv_uniform_p_call(
        w_low,
        w_high,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unit_w_low * unitv)


def jitc_matvec_normal(
    w_mu: Data,
    w_sigma: Data,
    conn_prob: float,
    v: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@v` operation,
    where :math:`M` is just-in-time randomly generated with a normal distribution for its value.

    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.

    .. warning::

        This API may change in the future.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is sampled from a normal distribution
    with mean `w_mu` and standard deviation `w_sigma`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).

        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.

    Parameters
    ----------
    w_mu: Array, ndarray, Quantity, float
        Mean (centre) of the distribution.
    w_sigma: Array, ndarray, Quantity, float
        Standard deviation (spread or "width") of the distribution. Must be non-negative.
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
    u.fail_for_dimension_mismatch(w_mu, w_sigma, "w_low and w_high must have the same dimension.")
    w_mu, unit_w_mu = u.split_mantissa_unit(w_mu)
    w_sigma = u.Quantity(w_sigma).to(unit_w_mu).mantissa
    v, unitv = u.split_mantissa_unit(v)
    clen = _initialize_conn_length(conn_prob)
    res = jitc_mv_normal_p_call(
        w_mu,
        w_sigma,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unit_w_mu * unitv)


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


def jitc_matmat_uniform(
    w_low: Data,
    w_high: Data,
    conn_prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@B` operation,
    where :math:`M` is just-in-time randomly generated with a uniform distribution for its value.
    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.
    .. warning::
        This API may change in the future.
    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is sampled from a uniform
    distrubtion within the range of `w_low` and `w_high`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@m`.
    .. note::
        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).            
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time   
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.
    Parameters
    ----------
    w_low: Array, ndarray, Quantity, float
        The lower boundary of the random matrix.
    w_high: Array, ndarray, Quantity, float
        The upper boundary of the random matrix.
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
    u.fail_for_dimension_mismatch(w_low, w_high, "w_low and w_high must have the same dimension.")
    w_low, unit_w_low = u.split_mantissa_unit(w_low)
    w_high = u.Quantity(w_high).to(unit_w_low).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(conn_prob)
    res = jitc_mm_uniform_p_call(
        w_low,
        w_high,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unit_w_low * unitB)


def jitc_matmat_normal(
    w_mu: Data,
    w_sigma: Data,
    conn_prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> Data:
    r"""Perform the :math:`y=M@m` operation,
    where :math:`M` is just-in-time randomly generated with a normal distribution for its value.
    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.
    .. warning::
        This API may change in the future.
    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is sampled from a normal distribution
    with mean `w_mu` and standard deviation `w_sigma`.
    When ``transpose=True``, we perform an operation of :math:`y=M^T@m`.
    .. note::
        Note that the just-in-time generated :math:`M` (`transpose=False`) is
        different from the generated :math:`M^T` (`transpose=True`).
        If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
        matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
        the speed compared with ``outdim_parallel=False``.
    Parameters
    ----------
    w_mu: float
        Mean (centre) of the distribution.
    w_sigma: float
        Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    conn_prob: float
        The connection probability.
    m: Array, ndarray
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
        The output of :math:`y = M @ m`.
    """
    seed = _initialize_seed(seed)
    u.fail_for_dimension_mismatch(w_mu, w_sigma, "w_low and w_high must have the same dimension.")
    w_mu, unit_w_mu = u.split_mantissa_unit(w_mu)
    w_sigma = u.Quantity(w_sigma).to(unit_w_mu).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(conn_prob)
    res = jitc_mm_normal_p_call(
        w_mu,
        w_sigma,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]
    return u.maybe_decimal(res * unit_w_mu * unitB)


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
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matvec_homo` operation.
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
    v_dtype = dtype_to_warp_type(v_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            weight: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_row = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + tid)
            i_col = int(0)

            while i_col < num_col:
                if _binomial_n1(state, clen0) == 1:
                    r += v[i_col]
                i_col += 1

            posts[i_row] = r * weight0
    else:
        def kernel(
            weight: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            weight0 = weight[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_col = warp.tid()

            col_v = v[i_col] * weight0
            state = warp.rand_init(seed0 + tid)
            i_row = int(0)

            for i_row in range(num_row):
                if _binomial_n1(state, clen0) == 1:
                    posts[i_row] += col_v
                i_row += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mv_homo_jvp_v(
    v_dot,
    weight,
    clen,
    v,
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
    v,
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
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_homo_transpose_rules(
    ct,
    weight,
    clen,
    v,
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
    assert ad.is_undefined_primal(v)

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
    v,
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
        v,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        v_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
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


# jitc csrmv uniform

def _jitc_mv_uniform_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_uniform` operation.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_low, w_high, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_row):
                connections = np.random.binomial(1, clen0, num_col)
                random_weights = np.random.uniform(w_low0, w_high0, num_col)

                posts[i_row] = np.sum(v * random_weights * connections)
    else:
        # outdim_parallel=False
        def kernel(w_low, w_high, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_col):
                connections = np.random.binomial(1, clen0, num_row)
                random_weights = np.random.uniform(w_low0, w_high0, num_row)

                posts += connections * random_weights * v[i_col]

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mv_uniform_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    v_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matvec_uniform` operation.
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
    v_dtype = dtype_to_warp_type(v_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            w_low: warp.array1d(dtype=weight_dtype),
            w_high: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_row = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + tid)
            i_col = int(0)

            while i_col < num_col:
                if _binomial_n1(state, clen0) == 1:
                    raw_v = warp.randf(state, w_low0, w_high0)
                    r += v[i_col] * raw_v
                i_col += 1

            posts[i_row] = r * weight0
    else:
        def kernel(
            w_low: warp.array1d(dtype=weight_dtype),
            w_high: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            i_col = warp.tid()

            col_v = v[i_col] * weight0
            state = warp.rand_init(seed0 + tid)
            i_row = int(0)

            for i_row in range(num_row):
                if _binomial_n1(state, clen0) == 1:
                    raw_v = warp.randf(state, w_low0, w_high0)
                    posts[i_row] += col_v * raw_v
                i_row += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mv_uniform_jvp_v(
    v_dot,
    w_low,
    w_high,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return [
        jitc_matvec_uniform(
            w_low,
            w_high,
            clen,
            v_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def _jitc_mv_uniform_jvp_w_low(
    w_low_dot,
    w_low,
    w_high,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mv_uniform_p_call(
        w_low_dot,
        w_high,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_uniform_jvp_w_high(
    w_high_dot,
    w_low,
    w_high,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mv_uniform_p_call(
        w_high_dot,
        w_high,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_uniform_transpose_rules(
    ct,
    w_low,
    w_high,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    assert not ad.is_undefined_primal(w_low)
    assert not ad.is_undefined_primal(w_high)
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(v)

    r = jitc_mv_uniform_p_call(
        w_low,
        w_high,
        clen,
        ct[0],
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_low, w_high, clen, r, seed, _


def _jitc_mv_uniform_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven COO matrix-vector product.")


def jitc_mv_uniform_p_call(
    w_low,
    w_high,
    clen,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_low.dtype)
    )

    return jitc_mv_uniform_p(
        w_low,
        w_high,
        clen,
        v,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        v_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mv_uniform_p = XLACustomKernel(
    'jitc_mv_uniform',
    cpu_kernel=NumbaKernelGenerator(_jitc_mv_uniform_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mv_uniform_gpu_kernel_generator,
        dim=lambda v_info, out_info, outdim_parallel, **kwargs: (
            out_info.shape[0] if outdim_parallel else
            v_info.shape[0]
        ),
        input_output_aliases={5: 0}
    )
)

jitc_mv_uniform_p.defjvp(_jitc_mv_uniform_jvp_w_low,
                         _jitc_mv_uniform_jvp_w_high,
                         None,
                         _jitc_mv_uniform_jvp_v)
jitc_mv_uniform_p.def_transpose_rule(_jitc_mv_uniform_transpose_rules)
jitc_mv_uniform_p.def_batching_rule(_jitc_mv_uniform_batching)


# jitc csrmv normal

def _jitc_mv_normal_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_normal` operation.
    """
    import numba  # pylint: disable=import-outside-toplevel

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_mu, w_sigma, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_row):
                connections = np.random.binomial(1, clen0, num_col)
                random_weights = np.random.normal(w_mu0, w_sigma0, num_col)

                posts[i_row] = np.sum(v * random_weights * connections)
    else:
        # outdim_parallel=False
        def kernel(w_mu, w_sigma, clen, v, seed, _, posts):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_col):
                connections = np.random.binomial(1, clen0, num_row)
                random_weights = np.random.normal(w_mu0, w_sigma0, num_row)

                posts += connections * random_weights * v[i_col]

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mv_normal_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    v_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matvec_normal` operation.
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
    v_dtype = dtype_to_warp_type(v_info.dtype)
    seed_dtype = dtype_to_warp_type(seed_info.dtype)

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(
            w_mu: warp.array1d(dtype=weight_dtype),
            w_sigma: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            i_row = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + tid)
            i_col = int(0)

            while i_col < num_col:
                if _binomial_n1(state, clen0) == 1:
                    raw_v = w_mu0 + w_sigma0 * warp.randn(state)
                    r += v[i_col] * raw_v
                i_col += 1

            posts[i_row] = r * weight0

    else:
        def kernel(
            w_mu: warp.array1d(dtype=weight_dtype),
            w_sigma: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            v: warp.array1d(dtype=v_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array1d(dtype=weight_dtype),
        ):
            num_row = posts.shape[0]
            num_col = v.shape[0]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]

            step = warp.max(warp.int32((num_row + 1) >> 5), 1)

            i_col = warp.tid()

            col_v = v[i_col] * weight0
            state = warp.rand_init(seed0 + tid)
            i_row = int(0)

            for i_row in range(num_row):
                if _binomial_n1(state, clen0) == 1:
                    raw_v = w_mu0 + w_sigma0 * warp.randn(state)
                    posts[i_row] += col_v * raw_v
                i_row += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mv_normal_jvp_v(
    v_dot,
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return [
        jitc_matvec_normal(
            w_mu,
            w_sigma,
            clen,
            v_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def _jitc_mv_normal_jvp_w_mu(
    w_mu_dot,
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mv_uniform_p_call(
        w_mu_dot,
        w_sigma,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_normal_jvp_w_sigma(
    w_sigma_dot,
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel,
    **kwargs
):
    return jitc_mv_uniform_p_call(
        w_mu,
        w_sigma_dot,
        clen,
        v,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )


def _jitc_mv_normal_transpose_rules(
    ct,
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    assert not ad.is_undefined_primal(w_mu)
    assert not ad.is_undefined_primal(w_sigma)
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(v)

    r = jitc_mv_uniform_p_call(
        w_mu,
        w_sigma,
        clen,
        ct[0],
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_mu, w_sigma, clen, r, seed, _


def _jitc_mv_normal_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1, None, None):
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel1']
        )
        return r, [1]
    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for event-driven COO matrix-vector product.")


def jitc_mv_normal_p_call(
    w_mu,
    w_sigma,
    clen,
    v,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_mu = jnp.atleast_1d(w_mu)
    w_sigma = jnp.atleast_1d(w_sigma)
    clen = jnp.atleast_1d(clen)

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_mu.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_mu.dtype)
    )

    return jitc_mv_uniform_p(
        w_mu,
        w_sigma,
        clen,
        v,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_mu.shape, w_mu.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        v_info=jax.ShapeDtypeStruct(v.shape, v.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mv_normal_p = XLACustomKernel(
    'jitc_mv_normal',
    cpu_kernel=NumbaKernelGenerator(_jitc_mv_normal_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mv_normal_gpu_kernel_generator,
        dim=lambda v_info, out_info, outdim_parallel, **kwargs: (
            out_info.shape[0] if outdim_parallel else
            v_info.shape[0]
        ),
        input_output_aliases={5: 0}
    )
)

jitc_mv_normal_p.defjvp(_jitc_mv_normal_jvp_w_mu,
                        _jitc_mv_normal_jvp_w_sigma,
                        None,
                        _jitc_mv_normal_jvp_v,
                        None)
jitc_mv_normal_p.def_transpose_rule(_jitc_mv_normal_transpose_rules)
jitc_mv_normal_p.def_batching_rule(_jitc_mv_normal_batching)


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


# jitc csrmm uniform

def _jitc_mm_uniform_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matmat_uniform` operation.
    """
    import numba

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_low, w_high, clen, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_rows):
                for i_col in range(num_cols):
                    connections = np.random.binomial(1, clen0, num_cols)
                    random_weights = np.random.uniform(w_low0, w_high0, num_cols)

                    posts[i_row, i_col] = np.sum(B[i_row, :] * random_weights * connections)

    else:
        # outdim_parallel=False
        # TODO: more checks on this kernel (random generation method)
        def kernel(w_low, w_high, clen, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_cols):
                for i_row in range(num_rows):
                    r = B[i_row, :]

                    connections = np.random.binomial(1, clen0, num_rows)
                    random_weights = np.random.uniform(w_low0, w_high0, num_rows)
                    effective_weights = connections * random_weights

                    posts += effective_weights[:, np.newaxis] * r

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mm_uniform_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matmat_uniform` operation.
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
            w_low: warp.array1d(dtype=weight_dtype),
            w_high: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)

            while cursor < num_cols:
                if _binomial_n1(state, clen0) == 1:
                    raw_v = warp.randf(state, w_low0, w_high0)
                    r += B[i_row, cursor] * raw_v
                cursor += 1
            posts[i_row, i_col] = r
    else:
        # outdim_parallel=False
        def kernel(
            w_low: warp.array1d(dtype=weight_dtype),
            w_high: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            w_low0 = w_low[0]
            w_high0 = w_high[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)
            while cursor < num_cols:
                if _binomial_n1(state, clen0) == 1:
                    for j in range(posts.shape[1]):
                        raw_v = warp.randf(state, w_low0, w_high0)
                        posts[cursor, j] += B[i_row, j] * raw_v
                cursor += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mm_uniform_jvp_w_low(
    w_low_dot,
    w_low,
    w_high,
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
    return jitc_mm_uniform_p_call(
        w_low_dot,
        w_high,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


def _jitc_mm_uniform_jvp_w_high(
    w_high_dot,
    w_low,
    w_high,
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
    return jitc_mm_uniform_p_call(
        w_low,
        w_high_dot,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


def _jitc_mm_uniform_jvp_B(
    B_dot,
    w_low,
    w_high,
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
        jitc_matmat_uniform(
            w_low,
            w_high,
            clen,
            B_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def _jitc_mm_uniform_transpose_rules(
    ct,
    w_low,
    w_high,
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
    assert not ad.is_undefined_primal(w_low)
    assert not ad.is_undefined_primal(w_high)
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(B)

    r = jitc_mm_uniform_p_call(
        w_low,
        w_high,
        clen,
        ct,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_low, w_high, clen, r, seed, _


def _jitc_mm_uniform_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = args[3].reshape(m, batch_size * n)
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = jitc_mm_uniform_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape, batch_size, n])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for JIT connection CSR matrix-matrix product")


def jitc_mm_uniform_p_call(
    w_low,
    w_high,
    clen,
    B,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_low = jnp.atleast_1d(w_low)
    w_high = jnp.atleast_1d(w_high)
    clen = jnp.atleast_1d(clen)

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_low.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_low.dtype)
    )

    return jitc_mm_uniform_p(
        w_low,
        w_high,
        clen,
        B,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_low.shape, w_low.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mm_uniform_p = XLACustomKernel(
    'jitc_mm_uniform',
    cpu_kernel=NumbaKernelGenerator(_jitc_mm_uniform_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mm_uniform_gpu_kernel_generator,
        dim=lambda out_info, **kwargs: (
            out_info.shape[0], out_info.shape[1]
        ),
        input_output_aliases={5: 0}
    )
)

jitc_mm_uniform_p.defjvp(_jitc_mm_uniform_jvp_w_low,
                         _jitc_mm_uniform_jvp_w_high,
                         None,
                         _jitc_mm_uniform_jvp_B)
jitc_mm_uniform_p.def_transpose_rule(_jitc_mm_uniform_transpose_rules)
jitc_mm_uniform_p.def_batching_rule(_jitc_mm_uniform_batching)


# jitc csrmm normal

def _jitc_mm_normal_cpu_kernel_generator(
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the CPU kernel for the :func:`_jitc_matmat_normal` operation.
    """
    import numba

    if outdim_parallel:
        # outdim_parallel=True
        def kernel(w_mu, w_sigma, clen, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_row in range(num_rows):
                for i_col in range(num_cols):
                    connections = np.random.binomial(1, clen0, num_cols)
                    random_weights = np.random.normal(w_mu0, w_sigma0, num_cols)

                    posts[i_row, i_col] = np.sum(B[i_row, :] * random_weights * connections)

    else:
        # outdim_parallel=False
        # TODO: more checks on this kernel (random generation method)
        def kernel(w_mu, w_sigma, clen, B, seed, _, posts):
            num_rows, num_cols = posts.shape
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]
            np.random.seed(seed0)

            for i_col in range(num_cols):
                for i_row in range(num_rows):
                    r = B[i_row, :]

                    connections = np.random.binomial(1, clen0, num_rows)
                    random_weights = np.random.normal(w_mu0, w_sigma0, num_rows)
                    effective_weights = connections * random_weights

                    posts += effective_weights[:, np.newaxis] * r

    kernel = numba.njit(**numba_environ.setting)(kernel)
    return kernel


def _jitc_mm_normal_gpu_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    clen_info: jax.ShapeDtypeStruct,
    B_info: jax.ShapeDtypeStruct,
    seed_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool = False,
    outdim_parallel: bool = True,
    **kwargs
) -> Kernel:
    r"""Generate the GPU kernel for the :func:`_jitc_matmat_normal` operation.
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
            w_mu: warp.array1d(dtype=weight_dtype),
            w_sigma: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            r = float(0.0)
            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)

            while cursor < num_cols:
                if _binomial_n1(state, clen0) == 1:
                    raw_v = w_mu0 + w_sigma0 * warp.randf(state)
                    r += B[i_row, cursor] * raw_v
                cursor += 1
            posts[i_row, i_col] = r
    else:
        # outdim_parallel=False
        def kernel(
            w_mu: warp.array1d(dtype=weight_dtype),
            w_sigma: warp.array1d(dtype=weight_dtype),
            clen: warp.array1d(dtype=clen_dtype),
            B: warp.array2d(dtype=B_dtype),
            seed: warp.array1d(dtype=seed_dtype),
            _: warp.array1d(dtype=weight_dtype),
            posts: warp.array2d(dtype=weight_dtype),
        ):
            num_rows = posts.shape[0]
            num_cols = posts.shape[1]
            w_mu0 = w_mu[0]
            w_sigma0 = w_sigma[0]
            clen0 = clen[0]
            seed0 = seed[0]

            i_row, i_col = warp.tid()

            state = warp.rand_init(seed0 + i_row * num_cols + i_col)

            cursor = int(0)
            while cursor < num_cols:
                if _binomial_n1(state, clen0) == 1:
                    for j in range(posts.shape[1]):
                        raw_v = w_mu0 + w_sigma0 * warp.randf(state)
                        posts[cursor, j] += B[i_row, j] * raw_v
                cursor += 1

    kernel = warp.kernel(kernel)
    return kernel


def _jitc_mm_normal_jvp_w_mu(
    w_mu_dot,
    w_mu,
    w_sigma,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    return jitc_mm_normal_p_call(
        w_mu_dot,
        w_sigma,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


def _jitc_mm_normal_jvp_w_sigma(
    w_sigma_dot,
    w_mu,
    w_sigma,
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
    return jitc_mm_normal_p_call(
        w_mu,
        w_sigma_dot,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


def _jitc_mm_normal_jvp_B(
    B_dot,
    w_mu,
    w_sigma,
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
        jitc_matmat_normal(
            w_mu,
            w_sigma,
            clen,
            B_dot,
            seed,
            shape=shape,
            transpose=transpose,
            outdim_parallel=outdim_parallel
        )
    ]


def _jitc_mm_normal_transpose_rules(
    ct,
    w_mu,
    w_sigma,
    clen,
    B,
    seed,
    _,
    *,
    shape,
    transpose,
    outdim_parallel
):
    assert not ad.is_undefined_primal(w_mu)
    assert not ad.is_undefined_primal(w_sigma)
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)
    assert ad.is_undefined_primal(B)

    r = jitc_mm_normal_p_call(
        w_mu,
        w_sigma,
        clen,
        ct,
        seed,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel
    )[0]

    return w_mu, w_sigma, clen, r, seed, _


def _jitc_mm_normal_batching(
    args,
    axes,
    **kwargs
):
    if tuple(axes) == (None, None, 0, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = jnp.transpose(args[3], (1, 0, 2)).reshape(m, batch_size * n)
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[2].shape
        B = args[3].reshape(m, batch_size * n)
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape[0], batch_size, n])
        return [r], [1]
    elif tuple(axes) == (None, None, 1, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        batch_size, m, n = args[3].shape
        B = args[3].reshape(m, batch_size * n)
        r = jitc_mm_normal_p_call(
            args[0],
            args[1],
            args[2],
            B,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            outdim_parallel=kwargs['outdim_parallel'],
        )
        r = jnp.reshape(r[0], [r[0].shape, batch_size, n])
        return [r], [2]

    else:
        raise NotImplementedError(f"Batching axes {axes} not implemented for JIT connection CSR matrix-matrix product")


def jitc_mm_normal_p_call(
    w_mu,
    w_sigma,
    clen,
    B,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    outdim_parallel: bool,
):
    w_mu = jnp.atleast_1d(w_mu)
    w_sigma = jnp.atleast_1d(w_sigma)
    clen = jnp.atleast_1d(clen)

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_mu.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_mu.dtype)
    )

    return jitc_mm_uniform_p(
        w_mu,
        w_sigma,
        clen,
        B,
        seed,
        jnp.zeros(out_info.shape, out_info.dtype),
        outs=[out_info],
        weight_info=jax.ShapeDtypeStruct(w_mu.shape, w_mu.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        outdim_parallel=outdim_parallel,
    )


jitc_mm_normal_p = XLACustomKernel(
    'jitc_mm_normal',
    cpu_kernel=NumbaKernelGenerator(_jitc_mm_normal_cpu_kernel_generator, input_output_aliases={5: 0}),
    gpu_kernel=WarpKernelGenerator(
        _jitc_mm_normal_gpu_kernel_generator,
        dim=lambda out_info, **kwargs: (
            out_info.shape[0], out_info.shape[1]
        ),
        input_output_aliases={5: 0}
    )
)

jitc_mm_normal_p.defjvp(_jitc_mm_normal_jvp_w_mu,
                        _jitc_mm_normal_jvp_w_sigma,
                        None,
                        _jitc_mm_normal_jvp_B)
jitc_mm_normal_p.def_transpose_rule(_jitc_mm_normal_transpose_rules)
jitc_mm_normal_p.def_batching_rule(_jitc_mm_normal_batching)
