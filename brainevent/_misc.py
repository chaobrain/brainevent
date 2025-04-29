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

from typing import Tuple, NamedTuple, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
from jax.experimental.sparse import csr_todense_p, coo_todense_p

from ._typing import MatrixShape, Data, Index


class COOInfo(NamedTuple):
    """
    A named tuple containing metadata for COO (Coordinate) format sparse matrices.

    COO format represents a sparse matrix using three arrays: data values, row indices,
    and column indices. This class stores shape and sorting information needed for
    sparse matrix operations.

    Attributes:
        shape: Sequence[int]
            The shape of the matrix as a sequence of integers (rows, columns).
        rows_sorted: bool, default=False
            Indicates whether the row indices are in sorted order.
        cols_sorted: bool, default=False
            Indicates whether the column indices are in sorted order within each row.
            Only relevant if ``rows_sorted`` is True.
    """
    shape: MatrixShape
    rows_sorted: bool = False
    cols_sorted: bool = False


def _coo_todense(
    data: Data,
    row: Index,
    col: Index,
    *,
    spinfo: COOInfo
) -> Data:
    """Convert CSR-format sparse matrix to a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      row : array of shape ``(nse,)``
      col : array of shape ``(nse,)`` and dtype ``row.dtype``
      spinfo : COOInfo object containing matrix metadata

    Returns:
      mat : array with specified shape and dtype matching ``data``
    """
    data, unit = u.split_mantissa_unit(data)
    r = coo_todense_p.bind(data, row, col, spinfo=spinfo)
    return u.maybe_decimal(r * unit)


@jax.jit
def _csr_to_coo(indices: jax.Array, indptr: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Given CSR (indices, indptr) return COO (row, col)"""
    return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices


def _csr_todense(
    data: Data,
    indices: Index,
    indptr: Index,
    *,
    shape: MatrixShape
) -> Data:
    """
    Convert CSR-format sparse matrix to a dense matrix.

    Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      shape : length-2 tuple representing the matrix shape

    Returns:
      mat : array with specified shape and dtype matching ``data``
    """
    data, unit = u.split_mantissa_unit(data)
    mat = csr_todense_p.bind(data, indices, indptr, shape=shape)
    return u.maybe_decimal(mat * unit)


def cdiv(m: int, n: int) -> int:
    """
    Calculate ceiling division of m by n (division rounded up to nearest integer).

    This is equivalent to math.ceil(m/n) but avoids floating-point operations.

    Args:
        m: Dividend (numerator)
        n: Divisor (denominator), must be positive

    Returns:
        The smallest integer k such that k ≥ m/n

    Examples:
        >>> cdiv(10, 3)  # 10/3 = 3.33... -> 4
        4
        >>> cdiv(9, 3)   # 9/3 = 3 -> 3
        3
    """
    if n <= 0:
        raise ValueError("Divisor must be positive")
    return (m + n - 1) // n


def generate_block_dim(n_conn: int) -> int:
    """
    Determines an appropriate block dimension based on the number of connections.

    This function selects a block size, typically a power of 2, based on the
    input `n_conn`. It seems intended for optimizing operations possibly
    related to parallel processing or memory access patterns where block
    sizes like 32, 64, 128, or 256 are common.

    Args:
        n_conn: An integer representing the number of connections or a similar
                metric influencing the desired block size.

    Returns:
        An integer representing the calculated block dimension. Returns 32, 64,
        128, or 256 based on `n_conn`, defaulting to 128 if `n_conn` exceeds 256.
    """
    if n_conn <= 32:
        block_size = 32
    elif n_conn <= 64:
        block_size = 64
    elif n_conn <= 128:
        block_size = 128
    elif n_conn <= 256:
        block_size = 256
    else:
        # Default or fallback block size for larger numbers of connections
        block_size = 128

    return block_size


def check_fixed_conn_num_shape(
    weights: jax.Array,
    indices: jax.Array,
    vector: jax.Array,
    shape: Sequence[int],
    transpose: bool,
    require_scalar_weight: bool = False
) -> Tuple[jax.ShapeDtypeStruct, jax.Array, int, int]:
    """
    Checks the shapes and dtypes of inputs for sparse operations.

    Validates the dimensions and consistency of weights, indices, and a vector
    involved in a sparse matrix operation (like SpMV or SpM^T V). It adjusts
    the weights array based on its dimensions and the `require_scalar_weight`
    flag. It also determines the expected output shape based on the transpose
    flag.

    Parameters
    ----------
    weights : jax.Array
        The weights associated with the sparse connections. Can be 2D (same shape
        as indices), 1D (scalar weight), or 0D (scalar weight).
    indices : jax.Array
        The indices of the connections, typically of shape (n_pre, n_conn),
        where n_conn is the number of connections per pre-synaptic neuron.
    vector : jax.Array
        The vector to be multiplied with the sparse matrix. Its shape depends
        on the `transpose` flag.
    shape : Sequence[int]
        A sequence of two integers `(n_pre, n_post)` representing the logical
        shape of the dense equivalent matrix.
    transpose : bool
        If True, checks shapes for the transposed operation (vector * Matrix).
        If False, checks shapes for the forward operation (Matrix * vector).
    require_scalar_weight : bool, optional
        If True and weights are 1D or 0D, ensures weights is treated as a
        scalar value. If False and weights are 0D, converts weights to a 1D
        array of size 1. Defaults to False.

    Returns
    -------
    out_struct : jax.ShapeDtypeStruct
        A ShapeDtypeStruct representing the expected shape and dtype of the
        output vector.
    weights : jax.Array
        The potentially modified weights array (e.g., scalar extracted from
        1D array if `require_scalar_weight` is True, or 0D converted to 1D).
    n_pre : int
        The number of pre-synaptic elements.
    n_post : int
        The number of post-synaptic elements.

    Raises
    ------
    ValueError
        If `weights` has dimensions other than 0, 1, or 2.
    AssertionError
        If shape inconsistencies are found between inputs (e.g., `weights`
        and `indices` shapes don't match when `weights` is 2D, `weights` is
        1D but not size 1, `indices` first dimension doesn't match `n_pre`,
        or `vector` shape doesn't match `n_pre` or `n_post` based on
        `transpose`).

    Examples
    --------

    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> n_pre, n_post, n_conn = 5, 10, 3
        >>> shape = (n_pre, n_post)
        >>> indices = jax.random.randint(key, (n_pre, n_conn), 0, n_post)
        >>> # Example 1: 2D weights, no transpose
        >>> weights_2d = jax.random.uniform(key, (n_pre, n_conn))
        >>> vector_post = jnp.ones(n_post)
        >>> out_struct, w, _, _ = check_fixed_conn_num_shape(weights_2d, indices, vector_post, shape, False)
        >>> print(out_struct)
        ShapeDtypeStruct(shape=(5,), dtype=float32)
        >>> print(w.shape)
        (5, 3)
        >>> # Example 2: Scalar weight (0D), transpose
        >>> weights_0d = jnp.array(0.5)
        >>> vector_pre = jnp.ones(n_pre)
        >>> out_struct, w, _, _ = check_fixed_conn_num_shape(weights_0d, indices, vector_pre, shape, True)
        >>> print(out_struct)
        ShapeDtypeStruct(shape=(10,), dtype=float32)
        >>> print(w.shape) # Converted to 1D array
        (1,)
        >>> # Example 3: Scalar weight (1D), require scalar, no transpose
        >>> weights_1d = jnp.array([0.7])
        >>> vector_post = jnp.ones(n_post)
        >>> out_struct, w, _, _ = check_fixed_conn_num_shape(weights_1d, indices, vector_post, shape, False, require_scalar_weight=True)
        >>> print(out_struct)
        ShapeDtypeStruct(shape=(5,), dtype=float32)
        >>> print(w.shape) # Kept as scalar
        ()
        >>> print(w)
        0.7
    """
    if weights.ndim == 2:
        assert weights.shape == indices.shape, (
            f'The shape of weights {weights.shape} and indices {indices.shape} '
            f'should be the same.'
        )
    elif weights.ndim == 1:
        assert weights.size == 1, (
            f'When weights is 1D, it should be a scalar (size 1), '
            f'got {weights.size}.'
        )
        if require_scalar_weight:
            # Extract the scalar value if required
            weights = weights[0]
        # Otherwise, keep it as a 1D array of size 1
    elif weights.ndim == 0:
        if not require_scalar_weight:
            # Convert scalar to 1D array if scalar is not explicitly required
            # This might be needed for broadcasting in some implementations
            weights = u.math.asarray([weights])
        # Otherwise, keep it as a 0D scalar
    else:
        raise ValueError(f'weight dim should be 2, 1, or 0, but got {weights.ndim}')

    assert indices.ndim == 2, f"Indices must be 2D, got {indices.ndim}"
    assert len(shape) == 2, f"Shape must have length 2, got {len(shape)}"
    n_pre, n_post = shape

    # Use indices.shape[0] for checking pre-synaptic dimension consistency
    assert indices.shape[0] == n_pre, (
        f'Pre size mismatch: indices.shape[0] ({indices.shape[0]}) '
        f'!= shape[0] ({n_pre})'
    )

    if transpose:
        # Operation: vector (n_pre) * Matrix (n_pre, n_post) -> out (n_post)
        out_struct = jax.ShapeDtypeStruct((n_post,), weights.dtype)
        assert vector.shape == (n_pre,), (
            f'When transpose=True, vector shape should be ({n_pre},), '
            f'got {vector.shape}'
        )
    else:
        # Operation: Matrix (n_pre, n_post) * vector (n_post) -> out (n_pre)
        out_struct = jax.ShapeDtypeStruct((n_pre,), weights.dtype)
        assert vector.shape == (n_post,), (
            f'When transpose=False, vector shape should be ({n_post},), '
            f'got {vector.shape}'
        )

    return out_struct, weights, n_pre, n_post
