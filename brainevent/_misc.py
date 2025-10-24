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

from functools import partial
from typing import Tuple, NamedTuple, Sequence, Union, Callable

import brainstate.environ
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import csr_todense_p, coo_todense_p

from ._event.base import BaseArray
from ._typing import MatrixShape, Data, Index


def is_known_type(x):
    return isinstance(x, (u.Quantity, jax.Array, np.ndarray, BaseArray))


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
    if data.size == 1:
        data = jnp.ones(row.shape, dtype=data.dtype) * data
    r = coo_todense_p.bind(data, row, col, spinfo=spinfo)
    return u.maybe_decimal(r * unit)


@jax.jit
def _csr_to_coo(
    indices: jax.Array,
    indptr: jax.Array
) -> Tuple[jax.Array, jax.Array]:
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
    if data.size == 1:
        data = jnp.ones(indices.shape, dtype=data.dtype) * data
    mat = csr_todense_p.bind(data, indices, indptr, shape=shape)
    return u.maybe_decimal(mat * unit)


def _block_csr_tocsr(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    shape: MatrixShape
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    n, m = data.shape[1:]
    N, M = shape
    n_block_rows = indptr.shape[0] - 1

    block_row_ids = jnp.repeat(jnp.arange(n_block_rows), jnp.diff(indptr))
    block_col_ids = indices

    block_i = jnp.arange(n)
    block_j = jnp.arange(m)
    ii, jj = jnp.meshgrid(block_i, block_j, indexing='ij')  # (n, m)

    row = (block_row_ids[:, None, None] * n + ii[None, :, :]).reshape(-1)
    col = (block_col_ids[:, None, None] * m + jj[None, :, :]).reshape(-1)
    val = data.reshape(-1)

    mask = val != 0
    row = row[mask]
    col = col[mask]
    val = val[mask]

    counts = jnp.bincount(row, length=N)
    csr_indptr = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(counts)])

    order = jnp.lexsort((col, row))  # row based sort
    csr_data = val[order]
    csr_indices = col[order]

    return csr_data, csr_indices, csr_indptr


@partial(jax.jit, static_argnames=["n", "m", "dense_shape_row", "nse"])
def _block_csr_tocoo(
    n: int,
    m: int,
    dense_shape_row: int,
    nse: int,
    indices: jax.Array,
    indptr: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    nrows = dense_shape_row // n
    delta_row_array = jnp.arange(n).repeat(m)
    delta_col_array = jnp.tile(jnp.arange(m), n)
    mini_block_nse = n * m

    def i_body(i_row, out):
        def j_body(x):
            i_block, i_row, val = x
            i_col = indices[i_block]
            start_row = i_row * n
            start_col = i_col * m
            val0 = jax.lax.dynamic_update_slice(val[0], start_row + delta_row_array, (i_block * mini_block_nse,))
            val1 = jax.lax.dynamic_update_slice(val[1], start_col + delta_col_array, (i_block * mini_block_nse,))
            val = (val0, val1)
            return (i_block + 1, i_row, val)

        return jax.lax.while_loop(lambda x: x[0] < indptr[x[1] + 1], j_body, (indptr[i_row], i_row, out))[-1]

    pre_ids, post_ids = jax.lax.fori_loop(
        0, nrows, i_body, (jnp.zeros(nse, dtype=jnp.int32), jnp.zeros(nse, dtype=jnp.int32))
    )
    return pre_ids, post_ids


def estimate_block_size(csr, efficiency: float = 0.7) -> Tuple[int, int]:
    """Attempt to determine the block_size of a CSR matrix

    Returns a block_size=(r,c) such that best match the efficiency setting
    """
    if csr.nse == 0:
        return (1, 1)

    if not 0 < efficiency < 1.0:
        raise ValueError('efficiency must satisfy 0.0 < efficiency < 1.0')

    high_efficiency = (1.0 + efficiency) / 2.0
    nse = float(csr.nse)
    N, M = csr.shape

    if N % 2 == 0 and M % 2 == 0:
        e22 = nse / (4 * count_blocks(csr, (2, 2)))
    else:
        e22 = 0.0

    if M % 3 == 0 and N % 3 == 0:
        e33 = nse / (9 * count_blocks(csr, (3, 3)))
    else:
        e33 = 0.0

    if e22 > high_efficiency and e33 > high_efficiency:
        e66 = nse / (36 * count_blocks(csr, (6, 6)))
        if e66 > efficiency:
            return (6, 6)
        else:
            return (3, 3)
    else:
        if M % 4 == 0 and N % 4 == 0:
            e44 = nse / (16 * count_blocks(csr, (4, 4)))
        else:
            e44 = 0.0

        if e44 > efficiency:
            return (4, 4)
        elif e33 > efficiency:
            return (3, 3)
        elif e22 > efficiency:
            return (2, 2)
        else:
            return (1, 1)


def _count_blocks(N, M, n, m, indptr, indices):
    mask = np.full(M // m + 1, -1, dtype=np.int32)
    n_blks = 0

    for i in range(N):
        bi = i // n
        for jj in range(indptr[i], indptr[i + 1]):
            bj = indices[jj] // m
            if mask[bj] != bi:
                mask[bj] = bi
                n_blks += 1

    return n_blks


def count_blocks(mat, block_size: Tuple[int, int]) -> int:
    """For a given block_size=(n,m) count the number of occupied
    blocks in a csr matrix
    """
    n, m = block_size
    if n < 1 or m < 1:
        raise ValueError('The block size n and m must be positive')

    return _count_blocks(mat.shape[0], mat.shape[1], n, m, mat.indptr, mat.indices)


def _nonzero_blocks(
    dense: jax.Array,
    block_size: Tuple[int, int]
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    N, M = dense.shape
    n, m = block_size
    n_block_rows = N // n
    n_block_cols = M // m
    blocks = dense.reshape(n_block_rows, n, n_block_cols, m)
    blocks = blocks.transpose(0, 2, 1, 3)
    blocks = blocks.reshape(-1, n, m)

    nonzero_blocks = []
    indices = []
    indptr = [0]
    for i, block in enumerate(blocks):
        if not jnp.all(block == 0):
            nonzero_blocks.append(block)
            indices.append(i % n_block_cols)
        if (i + 1) % n_block_cols == 0:
            indptr.append(len(nonzero_blocks))
    nonzero_blocks = jnp.array(nonzero_blocks)
    indices = jnp.array(indices)
    indptr = jnp.array(indptr, dtype=jnp.int32)

    return nonzero_blocks, indices, indptr


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


def generate_block_dim(
    n_conn: int,
    maximum: int = 256
) -> int:
    """
    Determines an appropriate block dimension based on the number of connections.

    This function selects a block size, typically a power of 2, based on the
    input `n_conn`. It seems intended for optimizing operations possibly
    related to parallel processing or memory access patterns where block
    sizes like 32, 64, 128, or 256 are common.

    Args:
        n_conn: An integer representing the number of connections or a similar
                metric influencing the desired block size.
        maximum: An optional integer specifying the maximum allowed block size.

    Returns:
        An integer representing the calculated block dimension. Returns 32, 64,
        128, or 256 based on `n_conn`, defaulting to 128 if `n_conn` exceeds 256.
    """
    if n_conn <= 32 <= maximum:
        block_size = 32
    elif n_conn <= 64 <= maximum:
        block_size = 64
    elif n_conn <= 128 <= maximum:
        block_size = 128
    elif n_conn <= 256 <= maximum:
        block_size = 256
    else:
        # Default or fallback block size for larger numbers of connections
        block_size = maximum

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
        if vector.ndim == 1:
            # Operation: vector (n_pre) * Matrix (n_pre, n_post) -> out (n_post)
            assert vector.shape == (n_pre,), (
                f'When transpose=True, vector shape should be ({n_pre},), '
                f'got {vector.shape}'
            )
            out_struct = jax.ShapeDtypeStruct((n_post,), weights.dtype)
        else:
            # Operation: Matrix (n_post, n_pre) * matrix (n_pre, k) -> out (n_post, k)

            # If vector is not 1D, it should be a 2D matrix with shape (n_pre, 1)
            assert vector.ndim == 2, (
                f'When transpose=True, vector should be 1D or 2D, '
                f'got {vector.ndim}D'
            )
            assert vector.shape[0] == n_pre, (
                f'When transpose=True, matrix shape should be (xx, {n_pre}), '
                f'got {vector.shape}'
            )
            out_struct = jax.ShapeDtypeStruct((n_post, vector.shape[1]), weights.dtype)
    else:
        if vector.ndim == 1:
            # Operation: Matrix (n_pre, n_post) * vector (n_post) -> out (n_pre)
            assert vector.shape == (n_post,), (
                f'When transpose=False, vector shape should be ({n_post},), '
                f'got {vector.shape}'
            )
            out_struct = jax.ShapeDtypeStruct((n_pre,), weights.dtype)
        else:
            # Operation: Matrix (n_pre, n_post) * matrix (n_post, k) -> out (n_pre, k)
            assert vector.ndim == 2, (
                f'When transpose=False, vector should be 1D or 2D, '
                f'got {vector.ndim}D'
            )
            assert vector.shape[0] == n_post, (
                f'When transpose=False, matrix shape should be ({n_post}, xx), '
                f'got {vector.shape}'
            )
            out_struct = jax.ShapeDtypeStruct((n_pre, vector.shape[1]), weights.dtype)

    return out_struct, weights, n_pre, n_post


def csr_to_coo_index(
    indptr: Union[jax.Array, np.ndarray],
    indices: Union[jax.Array, np.ndarray]
):
    """
    Converts CSR (Compressed Sparse Row) format indices to COO (Coordinate) format indices.

    This function transforms the CSR representation of a sparse matrix (given by indptr and
    indices) into the COO representation, which consists of explicit row and column indices
    for each non-zero element.

    Args:
        indptr: Union[jax.Array, np.ndarray]
            Row pointers array in CSR format. For a matrix with m rows, this has length m+1.
            Each element represents the starting position of a row in the indices array.

        indices: Union[jax.Array, np.ndarray]
            Column indices array in CSR format. Contains the column index for each non-zero
            element of the sparse matrix.

    Returns:
        Tuple[Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray]]:
            A tuple (pre_ids, post_ids) where:
            - pre_ids: Row indices in COO format
            - post_ids: Column indices in COO format (same as input indices)

    Notes:
        The function automatically determines whether to use NumPy or JAX based on the
        type of the input arrays. The computation is performed at compile time using
        jax.ensure_compile_time_eval().
    """
    with jax.ensure_compile_time_eval():
        mod = np if isinstance(indptr, np.ndarray) else jnp
        pre_ids = mod.repeat(mod.arange(indptr.size - 1), mod.diff(indptr))
        post_ids = indices
        return pre_ids, post_ids


def coo_to_csc_index(
    pre_ids: Union[jax.Array, np.ndarray],
    indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
):
    """
    Convert COO (Coordinate) format indices to CSC (Compressed Sparse Column) format.

    This function transforms a sparse matrix representation from Coordinate format
    (given by explicit row and column indices) to Compressed Sparse Column format.
    The implementation handles both NumPy and JAX arrays automatically.

    Parameters
    ----------
    pre_ids : Union[jax.Array, np.ndarray]
        Row indices array in COO format. Contains the row index for each non-zero
        element of the sparse matrix.

    indices : Union[jax.Array, np.ndarray]
        Column indices array in COO format. Contains the column index for each non-zero
        element of the sparse matrix.

    shape : Tuple[int, int]
        A tuple of (n_rows, n_cols) specifying the dimensions of the sparse matrix.
        Required as a keyword-only argument.

    Returns
    -------
    Tuple[Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray]]
        A tuple containing:
        - csc_indptr: Column pointers array in CSC format. For a matrix with n columns,
          this has length n+1. Each element represents the starting position of a column
          in the row indices array.
        - csc_indices: Row indices array in CSC format. Contains the row index for each
          non-zero element, sorted by column.
        - post_positions: Array of indices that can be used to reorder the data values
          from COO to CSC format.

    Notes
    -----
    The implementation automatically determines whether to use NumPy or JAX based on
    the type of input arrays. When using JAX arrays, computation is performed at
    compile time using jax.ensure_compile_time_eval().
    """
    n_post = shape[1]
    if isinstance(indices, np.ndarray) and isinstance(pre_ids, np.ndarray):
        # to maintain the original order of the elements with the same value
        new_post_position = np.argsort(indices)
        pre_ids_new = np.asarray(pre_ids[new_post_position], dtype=brainstate.environ.ditype())

        unique_post_ids, count = np.unique(indices, return_counts=True)
        post_count = np.zeros(n_post, dtype=brainstate.environ.ditype())
        post_count[unique_post_ids] = count

        indptr_new = np.insert(post_count.cumsum(), 0, 0)
        indptr_new = np.asarray(indptr_new, dtype=brainstate.environ.ditype())

    else:
        # to maintain the original order of the elements with the same value

        with jax.ensure_compile_time_eval():
            new_post_position = jnp.argsort(indices)
            pre_ids_new = jnp.asarray(pre_ids[new_post_position], dtype=brainstate.environ.ditype())

            unique_post_ids, count = jnp.unique(indices, return_counts=True)
            post_count = jnp.zeros(n_post, dtype=brainstate.environ.ditype())
            post_count = post_count.at[unique_post_ids].set(count)

            indptr_new = jnp.insert(post_count.cumsum(), 0, 0)
            indptr_new = jnp.asarray(indptr_new, dtype=brainstate.environ.ditype())

    return indptr_new, pre_ids_new, new_post_position


def csr_to_csc_index(
    csr_indptr: Union[jax.Array, np.ndarray],
    csr_indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
):
    """
    Convert CSR (Compressed Sparse Row) format indices to CSC (Compressed Sparse Column) format.

    This function transforms the sparse matrix representation from Compressed Sparse Row format
    to Compressed Sparse Column format by first converting to COO (Coordinate) format as an
    intermediate step.

    Parameters
    ----------
    csr_indptr : Union[jax.Array, np.ndarray]
        Row pointers array in CSR format. For a matrix with m rows, this has length m+1.
        Each element represents the starting position of a row in the indices array.

    csr_indices : Union[jax.Array, np.ndarray]
        Column indices array in CSR format. Contains the column index for each non-zero
        element of the sparse matrix.

    shape : Tuple[int, int]
        A tuple of (n_rows, n_cols) specifying the dimensions of the sparse matrix.
        Required as a keyword-only argument.

    Returns
    -------
    Tuple[Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray]]
        A tuple containing:
        - csc_indptr: Column pointers array in CSC format
        - csc_indices: Row indices array in CSC format
        - post_positions: Array of indices that can be used to reorder the data values
          from CSR to CSC format

    Raises
    ------
    AssertionError
        If shape is not a tuple/list, doesn't have exactly 2 dimensions, or contains
        non-positive dimensions.

    Notes
    -----
    The implementation automatically determines whether to use NumPy or JAX based on
    the type of input arrays.
    """
    assert isinstance(shape, (tuple, list)), "Shape must be a tuple or list"
    assert len(shape) == 2, "Shape must have exactly two dimensions (rows, columns)"
    assert shape[0] > 0 and shape[1] > 0, "Shape dimensions must be positive integers"
    pre_ids, post_ids = csr_to_coo_index(csr_indptr, csr_indices)
    csc_indptr, csc_indices, post_positions = coo_to_csc_index(pre_ids, post_ids, shape=shape)
    return csc_indptr, csc_indices, post_positions


def namescoped_jit(
    name: str = None,
    prefix: str = "brainevent",
    static_argnums: Tuple[int, ...] = (),
    static_argnames: Tuple[str, ...] = ()
):
    """Decorator that wraps a function with JAX's JIT compilation and sets its name.
    (For `brainstate.experimental.gdiist_bpu` module)

    Args:
        name: Optional name to set for the function. If None, uses the original function name.
        prefix: Prefix to add to function name if name is None.
        static_argnums: Tuple of positional argument indices to be treated as static.
        static_argnames: Tuple of keyword argument names to be treated as static.

    Returns:
        Decorated function with JAX JIT compilation applied.

    Example:
        @warp_jit_fun("my_function", static_argnums=(0,))
        def my_func(x, y):
            return x + y

        @warp_jit_fun(static_argnames=("shape", "transpose"))
        def my_func2(x, y, *, shape, transpose=False):
            return x + y
    """

    def decorator(fun: Callable):
        if name is not None:
            fun.__name__ = name
        else:
            fun.__name__ = f"{prefix}.{fun.__name__}"
        return jax.jit(fun, static_argnums=static_argnums, static_argnames=static_argnames)

    return decorator
