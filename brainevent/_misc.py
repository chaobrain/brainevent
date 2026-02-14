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
import functools
import inspect
from functools import partial
from typing import Tuple, NamedTuple, Sequence, Union, Callable, Optional

import brainstate.environ
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import csr_todense_p, coo_todense_p

from ._typing import MatrixShape, Data, Index


# -*- coding: utf-8 -*-


def is_known_type(x):
    """Check whether an object is a recognized array or event type.

    Determines if the input is an instance of one of the known numerical
    or event-representation types used throughout brainevent:
    :class:`brainunit.Quantity`, :class:`jax.Array`, :class:`numpy.ndarray`,
    or :class:`~brainevent._event.base.EventRepresentation`.

    Parameters
    ----------
    x : object
        The object to check.

    Returns
    -------
    bool
        ``True`` if ``x`` is an instance of a recognized type, ``False``
        otherwise.

    See Also
    --------
    COOInfo : Metadata type for COO sparse matrices.

    Notes
    -----
    This function is used internally for type dispatching in sparse
    matrix operations, ensuring that only recognized numerical types
    are passed to kernel functions.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._misc import is_known_type
        >>> is_known_type(jnp.array([1, 2, 3]))
        True
        >>> is_known_type("not an array")
        False
    """
    from ._event.base import EventRepresentation
    return isinstance(x, (u.Quantity, jax.Array, np.ndarray, EventRepresentation))


class COOInfo(NamedTuple):
    """Metadata for COO (Coordinate) format sparse matrices.

    COO format represents a sparse matrix using three arrays: data
    values, row indices, and column indices.  This named tuple stores
    the matrix shape and sorting information needed by sparse matrix
    operations.

    Parameters
    ----------
    shape : MatrixShape
        The shape of the matrix as a sequence of two integers
        ``(n_rows, n_cols)``.
    rows_sorted : bool, optional
        Whether the row indices are in sorted (non-decreasing) order.
        Defaults to ``False``.
    cols_sorted : bool, optional
        Whether the column indices are in sorted order within each row.
        Only meaningful when ``rows_sorted`` is ``True``.  Defaults to
        ``False``.

    See Also
    --------
    csr_to_coo_index : Convert CSR indices to COO format.
    coo_to_csc_index : Convert COO indices to CSC format.

    Notes
    -----
    This type is used as the ``spinfo`` parameter in JAX's
    ``coo_todense`` primitive binding and throughout brainevent's COO
    sparse matrix operations.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._misc import COOInfo
        >>> info = COOInfo(shape=(100, 200), rows_sorted=True)
        >>> info.shape
        (100, 200)
        >>> info.rows_sorted
        True
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
    """Convert a COO-format sparse matrix to a dense matrix.

    Parameters
    ----------
    data : array_like
        Data values of shape ``(nse,)``, where *nse* is the number of
        stored elements.
    row : array_like
        Row index array of shape ``(nse,)``.
    col : array_like
        Column index array of shape ``(nse,)`` with the same dtype as
        *row*.
    spinfo : COOInfo
        Metadata for the sparse matrix including ``shape``.

    Returns
    -------
    Data
        A dense array with shape ``spinfo.shape`` and dtype matching
        *data*.
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
    """Convert CSR index arrays to COO ``(row, col)`` arrays.

    Parameters
    ----------
    indices : jax.Array
        Column index array from CSR format.
    indptr : jax.Array
        Row pointer array from CSR format.

    Returns
    -------
    row : jax.Array
        Row indices in COO format.
    col : jax.Array
        Column indices in COO format (identical to *indices*).
    """
    return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices


def _csr_todense(
    data: Data,
    indices: Index,
    indptr: Index,
    *,
    shape: MatrixShape
) -> Data:
    """Convert a CSR-format sparse matrix to a dense matrix.

    Parameters
    ----------
    data : array_like
        Data values of shape ``(nse,)``, where *nse* is the number of
        stored elements.
    indices : array_like
        Column index array of shape ``(nse,)``.
    indptr : array_like
        Row pointer array of shape ``(shape[0] + 1,)`` with the same
        dtype as *indices*.
    shape : MatrixShape
        A length-2 tuple ``(n_rows, n_cols)`` representing the matrix
        shape.

    Returns
    -------
    Data
        A dense array with the given *shape* and dtype matching *data*.
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
    """Convert block CSR format to regular CSR format by expanding blocks.

    Takes a block-sparse CSR matrix where each stored element is an (n, m)
    dense block and converts it to a regular CSR matrix by expanding all
    blocks into individual scalar elements. Zero elements within blocks are
    dropped from the output.

    Parameters
    ----------
    data : jax.Array
        Block data array of shape ``(num_blocks, n, m)`` where each entry is
        an ``n × m`` dense block.
    indices : jax.Array
        Block column indices array of shape ``(num_blocks,)``.
    indptr : jax.Array
        Block row pointer array of shape ``(n_block_rows + 1,)``.
    shape : tuple of int
        Shape ``(N, M)`` of the full (expanded) matrix.

    Returns
    -------
    csr_data : jax.Array
        CSR data array containing non-zero scalar values.
    csr_indices : jax.Array
        CSR column indices for each stored element.
    csr_indptr : jax.Array
        CSR row pointer array of shape ``(N + 1,)``.

    Notes
    -----
    This function is used internally for converting block-compressed
    representations to standard CSR format. The output CSR matrix has the same
    logical shape ``(N, M)`` as specified by the ``shape`` parameter.
    """
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
    """Convert block CSR format to COO format by expanding blocks.

    Takes a block-sparse CSR matrix where each stored block has shape ``(n, m)``
    and expands it into COO coordinate format with separate row and column
    index arrays. This function assumes all blocks are fully dense (no internal
    zeros are dropped).

    Parameters
    ----------
    n : int
        Number of rows per block.
    m : int
        Number of columns per block.
    dense_shape_row : int
        Total number of rows ``N`` in the expanded matrix.
    nse : int
        Number of stored elements (non-zeros) in the output COO format.
        This should equal ``num_blocks * n * m``.
    indices : jax.Array
        Block column indices array of shape ``(num_blocks,)``.
    indptr : jax.Array
        Block row pointer array of shape ``(n_block_rows + 1,)``.

    Returns
    -------
    pre_ids : jax.Array
        Row indices (pre-synaptic IDs) for each stored element in COO format,
        shape ``(nse,)``.
    post_ids : jax.Array
        Column indices (post-synaptic IDs) for each stored element in COO
        format, shape ``(nse,)``.

    Notes
    -----
    This function is JIT-compiled with static arguments for block size and
    shape to enable efficient lowering. It uses a nested loop structure with
    ``jax.lax.fori_loop`` and ``jax.lax.while_loop`` for JAX compatibility.
    """
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
    """Estimate an appropriate block size for a CSR sparse matrix.

    Attempts to find the largest block size ``(r, c)`` where the fraction
    of non-zero entries to total entries within occupied blocks (the block
    efficiency) exceeds the given ``efficiency`` threshold. Candidate block
    sizes are drawn from the set ``{(1,1), (2,2), (3,3), (4,4), (6,6)}``.

    Parameters
    ----------
    csr : sparse matrix
        A CSR sparse matrix with attributes ``nse`` (number of stored
        elements), ``shape``, ``indptr``, and ``indices``.
    efficiency : float, optional
        Target efficiency threshold in the open interval ``(0, 1)``.
        A higher value requires denser blocks before a larger block size
        is chosen. Defaults to ``0.7``.

    Returns
    -------
    tuple of int
        A ``(block_rows, block_cols)`` tuple selected from the candidate
        set that best matches the efficiency criterion. Returns ``(1, 1)``
        if the matrix is empty or no larger block size meets the threshold.

    Raises
    ------
    ValueError
        If ``efficiency`` is not in the open interval ``(0, 1)``.

    See Also
    --------
    count_blocks : Count the number of occupied blocks for a given block size.

    Notes
    -----
    The algorithm first checks ``(2,2)`` and ``(3,3)`` blocks. If both
    exceed a high-efficiency bar (the midpoint between ``efficiency`` and
    ``1.0``), it considers ``(6,6)``. Otherwise it falls through
    ``(4,4)``, ``(3,3)``, ``(2,2)`` in order. A candidate block size
    is only considered if the matrix dimensions are evenly divisible by
    the block dimensions.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._misc import estimate_block_size
        >>> # Assuming `csr_mat` is a CSR sparse matrix:
        >>> block_size = estimate_block_size(csr_mat, efficiency=0.7)  # doctest: +SKIP
        >>> print(block_size)  # e.g. (2, 2) or (1, 1)
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
    """Count the number of unique blocks needed for a block-sparse representation.

    Given a CSR matrix and a target block size ``(n, m)``, counts how many
    distinct ``n × m`` blocks would be needed to represent the non-zero
    structure in block-sparse format.

    Parameters
    ----------
    N : int
        Number of rows in the full matrix.
    M : int
        Number of columns in the full matrix.
    n : int
        Number of rows per block.
    m : int
        Number of columns per block.
    indptr : array_like
        CSR row pointer array of shape ``(N + 1,)``.
    indices : array_like
        CSR column index array.

    Returns
    -------
    int
        The number of unique blocks required to cover all non-zero entries.

    Notes
    -----
    This function uses a marking array (``mask``) to track which blocks have
    already been counted in each block row, ensuring each block is counted
    only once even if multiple scalar elements from the CSR matrix fall
    within it.
    """
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
    """Count the number of occupied blocks in a CSR sparse matrix.

    For a given ``block_size = (n, m)``, counts how many ``n x m`` blocks
    in the matrix contain at least one non-zero entry.

    Parameters
    ----------
    mat : sparse matrix
        A CSR sparse matrix with attributes ``shape``, ``indptr``, and
        ``indices``.
    block_size : tuple of int
        A ``(block_rows, block_cols)`` tuple specifying the dimensions of
        each block. Both values must be positive integers.

    Returns
    -------
    int
        The number of ``block_size``-shaped blocks that contain at least
        one non-zero element.

    Raises
    ------
    ValueError
        If either component of ``block_size`` is less than 1.

    See Also
    --------
    estimate_block_size : Automatically choose a good block size for a CSR matrix.

    Notes
    -----
    The counting is performed using a row-sweep algorithm that tracks
    which block columns have been seen for each block row, using a
    mask array for O(1) lookup.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._misc import count_blocks
        >>> # Assuming `csr_mat` is a CSR sparse matrix:
        >>> n_blocks = count_blocks(csr_mat, (2, 2))  # doctest: +SKIP
        >>> print(n_blocks)
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
    """Compute the ceiling division of two positive integers.

    Returns the smallest integer ``k`` such that ``k >= m / n``, equivalent
    to ``math.ceil(m / n)`` but implemented using only integer arithmetic.

    Parameters
    ----------
    m : int
        The dividend (numerator).
    n : int
        The divisor (denominator). Must be a positive integer.

    Returns
    -------
    int
        The smallest integer ``k`` satisfying ``k * n >= m``.

    Raises
    ------
    ValueError
        If ``n`` is not positive.

    See Also
    --------
    generate_block_dim : Select a power-of-two block size for kernels.

    Notes
    -----
    The implementation uses the integer formula ``(m + n - 1) // n``
    which avoids floating-point rounding issues that ``math.ceil``
    could introduce for very large integers.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._misc import cdiv
        >>> cdiv(10, 3)
        4
        >>> cdiv(9, 3)
        3
        >>> cdiv(1, 1)
        1
    """
    if n <= 0:
        raise ValueError("Divisor must be positive")
    return (m + n - 1) // n


def generate_block_dim(
    n_conn: int,
    maximum: int = 256
) -> int:
    """Determine an appropriate block dimension for parallel kernel execution.

    Selects a power-of-two block size from the set ``{32, 64, 128, 256}``
    that is at least as large as ``n_conn`` and does not exceed ``maximum``.
    If ``n_conn`` exceeds all candidates, the ``maximum`` value is returned.

    Parameters
    ----------
    n_conn : int
        The number of connections (or similar workload metric) that the
        block must cover.
    maximum : int, optional
        The maximum allowed block size. Defaults to ``256``.

    Returns
    -------
    int
        A block dimension from ``{32, 64, 128, 256}`` or ``maximum`` if
        no candidate is large enough.

    See Also
    --------
    cdiv : Ceiling division helper.

    Notes
    -----
    Choosing a power-of-two block size aligned to the GPU warp size
    (32 threads) ensures efficient hardware utilization.  The smallest
    sufficient block size is chosen to minimize wasted threads when
    ``n_conn`` is small.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._misc import generate_block_dim
        >>> generate_block_dim(20)
        32
        >>> generate_block_dim(50)
        64
        >>> generate_block_dim(300)
        256
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
    """Validate input shapes for fixed-connection-number sparse operations.

    Checks the dimensions and consistency of weights, indices, and a vector
    involved in a sparse matrix operation (SpMV or transposed SpMV). Adjusts
    the weights array based on its dimensionality and the
    ``require_scalar_weight`` flag, and determines the expected output shape
    based on the ``transpose`` flag.

    Parameters
    ----------
    weights : jax.Array
        The weights associated with the sparse connections. Can be:

        - **2D** with shape ``(n_pre, n_conn)`` matching ``indices``,
        - **1D** with a single element (scalar weight), or
        - **0D** (scalar weight).
    indices : jax.Array
        Connection index array of shape ``(n_pre, n_conn)`` where each row
        contains the post-synaptic indices for one pre-synaptic element.
    vector : jax.Array
        The vector (or matrix) to multiply with the sparse connectivity.
        Shape depends on ``transpose``:

        - ``transpose=False``: shape ``(n_post,)`` or ``(n_post, k)``.
        - ``transpose=True``: shape ``(n_pre,)`` or ``(n_pre, k)``.
    shape : sequence of int
        A length-2 sequence ``(n_pre, n_post)`` giving the logical dense
        matrix shape.
    transpose : bool
        If ``True``, validate for the transposed operation
        ``vector @ Matrix -> (n_post,)``.
        If ``False``, validate for the forward operation
        ``Matrix @ vector -> (n_pre,)``.
    require_scalar_weight : bool, optional
        If ``True`` and weights are 1D of size 1, extract the scalar value.
        If ``False`` and weights are 0D, promote to a 1D array of size 1.
        Defaults to ``False``.

    Returns
    -------
    out_struct : jax.ShapeDtypeStruct
        Expected shape and dtype of the output.
    weights : jax.Array
        The (potentially modified) weights array.
    n_pre : int
        Number of pre-synaptic elements.
    n_post : int
        Number of post-synaptic elements.

    Raises
    ------
    ValueError
        If ``weights`` has a number of dimensions other than 0, 1, or 2.
    AssertionError
        If shape inconsistencies are found between inputs (e.g.,
        ``weights`` and ``indices`` shapes do not match when ``weights``
        is 2D, ``indices`` first dimension does not match ``n_pre``, or
        ``vector`` shape is incompatible with the specified operation).

    See Also
    --------
    csr_to_coo_index : Convert CSR indices to COO format.

    Notes
    -----
    This function is used as a validation and normalization step before
    dispatching to fixed-connection-number sparse kernels (e.g., in the
    ``_fcn`` and ``_jit_*`` modules).  It ensures that the weight,
    index, and vector dimensions are mutually consistent and prepares
    the output specification for the kernel.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> n_pre, n_post, n_conn = 5, 10, 3
        >>> shape = (n_pre, n_post)
        >>> indices = jax.random.randint(key, (n_pre, n_conn), 0, n_post)
        >>> weights_2d = jax.random.uniform(key, (n_pre, n_conn))
        >>> vector_post = jnp.ones(n_post)
        >>> out_struct, w, _, _ = check_fixed_conn_num_shape(
        ...     weights_2d, indices, vector_post, shape, False
        ... )
        >>> print(out_struct)
        ShapeDtypeStruct(shape=(5,), dtype=float32)
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
    """Convert CSR format index arrays to COO format index arrays.

    Transforms the Compressed Sparse Row representation of a sparse matrix
    (given by ``indptr`` and ``indices``) into the Coordinate representation,
    which uses explicit row and column index arrays for each non-zero element.

    Parameters
    ----------
    indptr : jax.Array or numpy.ndarray
        Row pointer array in CSR format. For a matrix with ``m`` rows, this
        has length ``m + 1``. Element ``indptr[i]`` gives the index into
        ``indices`` where row ``i`` starts, and ``indptr[i+1] - indptr[i]``
        is the number of non-zero entries in row ``i``.
    indices : jax.Array or numpy.ndarray
        Column index array in CSR format. Contains the column index for
        each non-zero element. Length equals the number of stored elements.

    Returns
    -------
    pre_ids : jax.Array or numpy.ndarray
        Row indices in COO format, with the same length as ``indices``.
    post_ids : jax.Array or numpy.ndarray
        Column indices in COO format (identical to the input ``indices``).

    See Also
    --------
    coo_to_csc_index : Convert COO indices to CSC format.
    csr_to_csc_index : Convert CSR indices directly to CSC format.

    Notes
    -----
    The function automatically selects NumPy or JAX operations based on
    the type of the input arrays. When JAX arrays are provided, the
    computation is wrapped in ``jax.ensure_compile_time_eval()`` so that
    it runs at trace time.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._misc import csr_to_coo_index
        >>> indptr = np.array([0, 2, 3, 5])
        >>> indices = np.array([0, 2, 1, 0, 3])
        >>> row_ids, col_ids = csr_to_coo_index(indptr, indices)
        >>> print(row_ids)
        [0 0 1 2 2]
        >>> print(col_ids)
        [0 2 1 0 3]
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
    """Convert COO format index arrays to CSC format.

    Transforms a sparse matrix representation from Coordinate (COO) format
    (explicit row and column index arrays) to Compressed Sparse Column (CSC)
    format. The implementation automatically selects NumPy or JAX operations
    based on the type of the input arrays.

    Parameters
    ----------
    pre_ids : jax.Array or numpy.ndarray
        Row index array in COO format. Contains the row index for each
        non-zero element.
    indices : jax.Array or numpy.ndarray
        Column index array in COO format. Contains the column index for
        each non-zero element.
    shape : tuple of int
        A ``(n_rows, n_cols)`` tuple specifying the dimensions of the
        sparse matrix. Keyword-only argument.

    Returns
    -------
    csc_indptr : jax.Array or numpy.ndarray
        Column pointer array in CSC format. For a matrix with ``n`` columns,
        this has length ``n + 1``. Element ``csc_indptr[j]`` gives the
        position in ``csc_indices`` where column ``j`` starts.
    csc_indices : jax.Array or numpy.ndarray
        Row index array in CSC format. Contains the row index for each
        non-zero element, ordered by column.
    post_positions : jax.Array or numpy.ndarray
        Permutation array that reorders data values from COO order to
        CSC order. If ``data`` is the COO data array, then
        ``data[post_positions]`` gives the values in CSC order.

    See Also
    --------
    csr_to_coo_index : Convert CSR indices to COO format.
    csr_to_csc_index : Convert CSR indices directly to CSC format.

    Notes
    -----
    When JAX arrays are provided, the computation is wrapped in
    ``jax.ensure_compile_time_eval()`` so that it executes at trace time
    rather than at runtime.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._misc import coo_to_csc_index
        >>> row_ids = np.array([0, 0, 1, 2, 2])
        >>> col_ids = np.array([0, 2, 1, 0, 3])
        >>> indptr, row_indices, perm = coo_to_csc_index(
        ...     row_ids, col_ids, shape=(3, 4)
        ... )
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
    """Convert CSR format index arrays to CSC format.

    Transforms the sparse matrix representation from Compressed Sparse Row
    (CSR) format to Compressed Sparse Column (CSC) format. Internally
    converts to COO format as an intermediate step via :func:`csr_to_coo_index`,
    then to CSC via :func:`coo_to_csc_index`.

    Parameters
    ----------
    csr_indptr : jax.Array or numpy.ndarray
        Row pointer array in CSR format. For a matrix with ``m`` rows, this
        has length ``m + 1``.
    csr_indices : jax.Array or numpy.ndarray
        Column index array in CSR format. Contains the column index for
        each non-zero element.
    shape : tuple of int
        A ``(n_rows, n_cols)`` tuple specifying the dimensions of the
        sparse matrix. Keyword-only argument.

    Returns
    -------
    csc_indptr : jax.Array or numpy.ndarray
        Column pointer array in CSC format.
    csc_indices : jax.Array or numpy.ndarray
        Row index array in CSC format.
    post_positions : jax.Array or numpy.ndarray
        Permutation array that reorders data values from CSR order to
        CSC order. If ``data`` is the CSR data array, then
        ``data[post_positions]`` gives the values in CSC order.

    Raises
    ------
    AssertionError
        If ``shape`` is not a tuple or list, does not have exactly two
        elements, or contains non-positive dimensions.

    See Also
    --------
    csr_to_coo_index : Convert CSR indices to COO indices.
    coo_to_csc_index : Convert COO indices to CSC indices.

    Notes
    -----
    The conversion is performed in two steps: CSR is first expanded to
    COO via :func:`csr_to_coo_index`, then the COO representation is
    sorted by column via :func:`coo_to_csc_index`.  The returned
    ``post_positions`` permutation array can be used to reorder a CSR
    data array into CSC order.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._misc import csr_to_csc_index
        >>> indptr = np.array([0, 2, 3, 5])
        >>> indices = np.array([0, 2, 1, 0, 3])
        >>> csc_indptr, csc_indices, perm = csr_to_csc_index(
        ...     indptr, indices, shape=(3, 4)
        ... )
    """
    assert isinstance(shape, (tuple, list)), "Shape must be a tuple or list"
    assert len(shape) == 2, "Shape must have exactly two dimensions (rows, columns)"
    assert shape[0] > 0 and shape[1] > 0, "Shape dimensions must be positive integers"
    pre_ids, post_ids = csr_to_coo_index(csr_indptr, csr_indices)
    csc_indptr, csc_indices, post_positions = coo_to_csc_index(pre_ids, post_ids, shape=shape)
    return csc_indptr, csc_indices, post_positions


class NameScope:
    """A callable that caches a separate JIT-compiled function per unique ``backend`` value.

    This enables efficient per-backend caching without relying on JAX's
    static argument mechanism. Each distinct ``backend`` keyword argument
    produces a separate JIT-compiled variant of the wrapped function, which
    is cached for reuse on subsequent calls.

    Parameters
    ----------
    fn : callable
        The function to wrap with per-backend JIT compilation.
    name : str or None, optional
        Display name for the function. If ``None``, a name is constructed
        from ``prefix`` and the function's ``__name__``.
    prefix : str, optional
        Prefix prepended to the function name when ``name`` is ``None``.
        Defaults to ``"brainevent"``.
    module : str, optional
        Value to set for ``__module__``. Defaults to ``"brainevent"``.
    static_argnums : sequence of int or int, optional
        Positional argument indices to treat as static (passed through to
        ``jax.jit``). Defaults to ``()``.
    static_argnames : sequence of str or str, optional
        Keyword argument names to treat as static (passed through to
        ``jax.jit``). Defaults to ``()``.

    See Also
    --------
    namescope : Decorator form that creates a ``NameScope`` instance.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._misc import NameScope
        >>> def my_kernel(x, y):
        ...     return x + y
        >>> ns = NameScope(my_kernel, name="brainevent.my_kernel")
        >>> result = ns(x, y, backend="pallas")  # doctest: +SKIP
    """

    def __init__(
        self,
        fn: Callable,
        name: Optional[str] = None,
        prefix: str = "brainevent",
        module: str = 'brainevent',
        static_argnums: Sequence[int] | int = (),
        static_argnames: Sequence[str] | str = (),
    ):
        self._fn = fn
        self._static_argnums = static_argnums
        self._static_argnames = static_argnames
        fn.__name__ = name if name is not None else f"{prefix}.{fn.__name__}"
        self._cache = {}  # backend -> jit_compiled_fn
        # Check whether the wrapped function accepts a 'backend' keyword.
        # True when the function either has an explicit backend parameter or accepts **kwargs.
        sig = inspect.signature(fn)
        self._has_backend = (
            'backend' in sig.parameters or
            any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        )
        # Copy function metadata
        self.__name__ = fn.__name__
        self.__qualname__ = getattr(fn, '__qualname__', self.__name__)
        self.__doc__ = fn.__doc__
        self.__module__ = module
        self.__wrapped__ = fn

    def _get_jit_fn(self, backend):
        if backend not in self._cache:
            fn = functools.partial(self._fn, backend=backend) if self._has_backend else self._fn
            self._cache[backend] = jax.jit(
                fn,
                static_argnums=self._static_argnums,
                static_argnames=self._static_argnames,
            )
        return self._cache[backend]

    def __call__(self, *args, **kwargs):
        backend = kwargs.pop('backend', None)
        jit_fn = self._get_jit_fn(backend)
        return jit_fn(*args, **kwargs)

    def __repr__(self):
        return f"<NameScope({self.__name__})>"


def namescope(
    fn: Callable = None,
    name: str = None,
    prefix: str = "brainevent",
    module: str = 'brainevent',
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = ()
):
    """Decorator that wraps a function with per-backend JIT compilation.

    Returns a :class:`NameScope` instance that caches a separate
    JIT-compiled variant of the decorated function for each unique
    ``backend`` keyword argument value.

    Parameters
    ----------
    fn : callable, optional
        The function to decorate. When ``None``, returns a decorator
        (allowing use with or without parentheses).
    name : str or None, optional
        Display name for the function. If ``None``, the name is derived
        from ``prefix`` and the function's ``__name__``.
    prefix : str, optional
        Prefix prepended to the function name when ``name`` is ``None``.
        Defaults to ``"brainevent"``.
    module : str, optional
        Value to set for ``__module__``. Defaults to ``"brainevent"``.
    static_argnums : sequence of int, optional
        Positional argument indices to treat as static (passed through to
        ``jax.jit``). Defaults to ``()``.
    static_argnames : sequence of str, optional
        Keyword argument names to treat as static (passed through to
        ``jax.jit``). Defaults to ``()``.

    Returns
    -------
    NameScope
        A ``NameScope`` instance wrapping the function with per-backend
        JIT caching. When used as a parameterized decorator (i.e.,
        ``fn`` is ``None``), returns a decorator function instead.

    See Also
    --------
    NameScope : The underlying class that implements per-backend JIT caching.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._misc import namescope
        >>> @namescope(static_argnums=(0,))
        ... def my_func(x, y):
        ...     return x + y
        >>> @namescope(static_argnames=("shape", "transpose"))
        ... def my_func2(x, y, *, shape, transpose=False):
        ...     return x + y
    """

    if fn is None:
        def decorator(fun: Callable):
            return NameScope(
                fun,
                name=name,
                prefix=prefix,
                module=module,
                static_argnums=static_argnums,
                static_argnames=static_argnames
            )

        return decorator

    else:
        return NameScope(
            fn,
            name=name,
            prefix=prefix,
            module=module,
            static_argnums=static_argnums,
            static_argnames=static_argnames
        )
