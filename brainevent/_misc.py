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

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import coo_todense_p

from ._typing import MatrixShape, Data, Index
from ._compatible_import import Tracer


# -*- coding: utf-8 -*-

_INT32_MAX = np.iinfo(np.int32).max


def _normalize_dtype(dtype):
    return np.dtype(dtype)


def _resolve_indptr_dtype(nse: int, requested="auto") -> np.dtype:
    """Resolve the row/column pointer dtype from an output nonzero count."""
    nse = int(nse)
    if nse < 0:
        raise ValueError(f"nse must be non-negative, got {nse}.")
    if requested == "auto":
        return np.dtype(np.int64 if nse > _INT32_MAX else np.int32)
    requested_dtype = _normalize_dtype(requested)
    if requested_dtype == np.dtype(np.int32):
        if nse > _INT32_MAX:
            raise OverflowError(
                f"indptr_dtype=int32 cannot represent nse={nse}; use indptr_dtype='auto' "
                "or indptr_dtype=int64 with jax_enable_x64 enabled."
            )
        return np.dtype(np.int32)
    if requested_dtype == np.dtype(np.int64):
        return np.dtype(np.int64)
    raise TypeError("indptr_dtype must be 'auto', int32, or int64.")


def _require_jax_x64_for_int64(dtype, context: str) -> None:
    """Fail before creating a JAX int64 array when x64 is disabled."""
    if _normalize_dtype(dtype) == np.dtype(np.int64) and not jax.config.jax_enable_x64:
        raise ValueError(
            f"{context} requires int64 indices, but jax_enable_x64 is False. "
            "Enable x64 before constructing large CSR/CSC structures."
        )


def _is_numpy_output(*arrays) -> bool:
    return all(isinstance(x, np.ndarray) for x in arrays)


def _as_output_index_array(values, dtype, *, output_is_numpy: bool, context: str):
    dtype = _normalize_dtype(dtype)
    arr = np.asarray(values, dtype=dtype)
    if output_is_numpy:
        return arr
    _require_jax_x64_for_int64(dtype, context)
    return jnp.asarray(arr, dtype=jnp.dtype(dtype))


def _as_int32_indices(indices, secondary_dim: int, context: str, *, output_is_numpy: Optional[bool] = None):
    """Validate compressed secondary indices and return them as int32."""
    secondary_dim = int(secondary_dim)
    if secondary_dim < 0:
        raise ValueError(f"{context}: secondary dimension must be non-negative, got {secondary_dim}.")
    if secondary_dim > _INT32_MAX + 1:
        raise OverflowError(
            f"{context}: secondary dimension {secondary_dim} cannot be represented by int32 indices."
        )
    indices_np = np.asarray(jax.device_get(indices))
    if not np.issubdtype(indices_np.dtype, np.integer):
        raise TypeError(f"{context}: indices must be an integer array, got dtype {indices_np.dtype}.")
    if indices_np.size:
        min_index = int(indices_np.min())
        max_index = int(indices_np.max())
        if min_index < 0:
            raise ValueError(f"{context}: indices must be non-negative.")
        if max_index >= secondary_dim:
            raise ValueError(
                f"{context}: index value {max_index} is out of bounds for secondary dimension {secondary_dim}."
            )
        if max_index > _INT32_MAX:
            raise OverflowError(f"{context}: index value {max_index} cannot be represented by int32.")
    if output_is_numpy is None:
        output_is_numpy = isinstance(indices, np.ndarray)
    return _as_output_index_array(indices_np, np.int32, output_is_numpy=output_is_numpy, context=context)


def _as_indptr(indptr, nse: int, indptr_dtype="auto", context: str = "CSR", *, output_is_numpy: Optional[bool] = None):
    """Validate and convert a compressed pointer array without silent truncation."""
    dtype = _resolve_indptr_dtype(nse, requested=indptr_dtype)
    indptr_np = np.asarray(jax.device_get(indptr))
    if not np.issubdtype(indptr_np.dtype, np.integer):
        raise TypeError(f"{context}: indptr must be an integer array, got dtype {indptr_np.dtype}.")
    if indptr_np.size and int(indptr_np.min()) < 0:
        raise ValueError(f"{context}: indptr values must be non-negative.")
    if dtype == np.dtype(np.int32) and indptr_np.size:
        max_offset = int(indptr_np.max())
        if max_offset > _INT32_MAX:
            raise OverflowError(
                f"{context}: indptr value {max_offset} cannot be represented by int32; "
                "use indptr_dtype=int64 with jax_enable_x64 enabled."
            )
    if output_is_numpy is None:
        output_is_numpy = isinstance(indptr, np.ndarray)
    return _as_output_index_array(indptr_np, dtype, output_is_numpy=output_is_numpy, context=context)


def _as_int32_cuda_offsets(offsets, context: str):
    """Return int32 offsets for CUDA kernels that do not support int64 offsets."""
    dtype = getattr(offsets, "dtype", None)
    if dtype is None:
        dtype = np.asarray(offsets).dtype
    dtype = _normalize_dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"{context}: offsets must be an integer array, got dtype {dtype}.")
    if dtype == np.dtype(np.int64):
        raise NotImplementedError(
            f"{context}: this CUDA path currently supports only int32 offsets; got int64. "
            "Large CSR/CSC structures that require int64 indptr are refused here instead of "
            "being truncated to int32."
        )
    if dtype != np.dtype(np.int32):
        offsets_np = np.asarray(jax.device_get(offsets))
        if offsets_np.size:
            min_offset = int(offsets_np.min())
            max_offset = int(offsets_np.max())
            if min_offset < 0:
                raise ValueError(f"{context}: offsets must be non-negative.")
            if max_offset > _INT32_MAX:
                raise OverflowError(
                    f"{context}: offset value {max_offset} cannot be represented by int32."
                )
        return jnp.asarray(offsets_np, dtype=jnp.int32)
    return jnp.asarray(offsets, dtype=jnp.int32)


def _check_compressed_structure(indices, indptr, shape: MatrixShape, format: str) -> None:
    """Validate CSR/CSC structure arrays after safe dtype conversion."""
    assert isinstance(shape, (tuple, list)), "Shape must be a tuple or list"
    assert len(shape) == 2, "Shape must have exactly two dimensions"
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"{format}: shape dimensions must be non-negative, got {shape}.")
    fmt = format.lower()
    if fmt == "csr":
        primary_dim, secondary_dim = n_rows, n_cols
    elif fmt == "csc":
        primary_dim, secondary_dim = n_cols, n_rows
    else:
        raise ValueError(f"Unknown compressed sparse format {format!r}.")
    indices_np = np.asarray(jax.device_get(indices))
    indptr_np = np.asarray(jax.device_get(indptr))
    if indices_np.ndim != 1:
        raise ValueError(f"{format}: indices must be 1D, got shape {indices_np.shape}.")
    if indptr_np.ndim != 1:
        raise ValueError(f"{format}: indptr must be 1D, got shape {indptr_np.shape}.")
    if indptr_np.size != primary_dim + 1:
        raise ValueError(
            f"{format}: indptr length must be primary dimension + 1 ({primary_dim + 1}), "
            f"got {indptr_np.size}."
        )
    if indptr_np.size == 0:
        raise ValueError(f"{format}: indptr must contain at least one element.")
    if int(indptr_np[0]) != 0:
        raise ValueError(f"{format}: indptr must start at 0.")
    if np.any(np.diff(indptr_np.astype(np.int64, copy=False)) < 0):
        raise ValueError(f"{format}: indptr must be non-decreasing.")
    if int(indptr_np[-1]) != int(indices_np.size):
        raise ValueError(
            f"{format}: indptr[-1] ({int(indptr_np[-1])}) must equal indices.size ({indices_np.size})."
        )
    _as_int32_indices(indices_np, secondary_dim, f"{format} structure", output_is_numpy=True)


def _coordinate_index_dtype(dtype):
    dtype = _normalize_dtype(dtype)
    return np.int32


def _offset_index_dtype(nse: int, preferred=None):
    return _resolve_indptr_dtype(nse, requested="auto").type


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


_CSR_SIGNED_INDEX_DTYPES = (jnp.dtype(jnp.int32), jnp.dtype(jnp.int64))


def _check_csr_structure_dtypes(indices, indptr) -> None:
    """Validate public CSR structure dtypes.

    CSR kernels use int32 column indices and int32/int64 row pointers.
    """
    indices_dtype = jnp.dtype(indices.dtype)
    indptr_dtype = jnp.dtype(indptr.dtype)
    assert indices_dtype == jnp.dtype(jnp.int32), (
        f"CSR kernels require indices with dtype int32; got indices dtype {indices_dtype}."
    )
    assert indptr_dtype in _CSR_SIGNED_INDEX_DTYPES, (
        f"CSR kernels require indptr with dtype int32 or int64; got indptr dtype {indptr_dtype}."
    )


def _check_csr_cuda_structure_dtypes(indices_info, indptr_info) -> None:
    """Validate the raw CUDA CSR ABI: int32 indices and int32/int64 indptr."""
    indices_dtype = jnp.dtype(indices_info.dtype)
    indptr_dtype = jnp.dtype(indptr_info.dtype)
    if indices_dtype != jnp.dtype(jnp.int32):
        raise TypeError(
            "CSR cuda_raw kernels require indices with dtype int32; "
            f"got indices dtype {indices_dtype}."
        )
    if indptr_dtype not in _CSR_SIGNED_INDEX_DTYPES:
        raise TypeError(
            "CSR cuda_raw kernels require indptr with dtype int32 or int64; "
            f"got indptr dtype {indptr_dtype}."
        )


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

    Notes
    -----
    Repeated ``(row, col)`` coordinates are **accumulated** (summed), matching
    the semantics of the CSR matmul kernels.  Materialisation is performed with
    an additive scatter (``jnp.zeros(...).at[row, col].add(data)``) so the result
    is correct on every backend -- unlike JAX's ``csr_todense`` primitive (which
    overwrites on duplicate columns) and the cuSPARSE ``coo_todense`` lowering
    (which assumes a canonical, duplicate-free matrix).  Canonical matrices are
    unaffected by the choice of reduction.
    """
    data, unit = u.split_mantissa_unit(data)
    if data.size == 1:
        data = jnp.ones(indices.shape, dtype=data.dtype) * data
    row, col = _csr_to_coo(indices, indptr)
    mat = jnp.zeros(shape, dtype=data.dtype).at[row, col].add(data)
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

    nse = int(val.size)
    offset_dtype = _resolve_indptr_dtype(nse, requested="auto")
    _require_jax_x64_for_int64(offset_dtype, "_block_csr_tocsr indptr")
    offset_jdtype = jnp.dtype(offset_dtype)

    counts = jnp.bincount(row.astype(offset_jdtype), length=N).astype(offset_jdtype)
    csr_indptr = jnp.concatenate(
        [jnp.zeros((1,), dtype=offset_jdtype), jnp.cumsum(counts, dtype=offset_jdtype)]
    )

    order = jnp.lexsort((col, row))  # row based sort
    csr_data = val[order]
    csr_indices = _as_int32_indices(
        col[order], M, "_block_csr_tocsr column indices", output_is_numpy=False
    )

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
    nse = len(nonzero_blocks)
    offset_dtype = _resolve_indptr_dtype(nse, requested="auto")
    nonzero_blocks = jnp.array(nonzero_blocks)
    indices = _as_int32_indices(
        np.asarray(indices), n_block_cols, "_nonzero_blocks indices", output_is_numpy=False
    )
    indptr = _as_output_index_array(
        np.asarray(indptr), offset_dtype, output_is_numpy=False, context="_nonzero_blocks indptr"
    )

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
    indptr_dtype="auto",
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
    n_pre, n_post = shape
    output_is_numpy = _is_numpy_output(pre_ids, indices)
    with jax.ensure_compile_time_eval():
        pre_ids_np = np.asarray(jax.device_get(pre_ids))
        post_ids_np = np.asarray(jax.device_get(indices))
        if pre_ids_np.shape != post_ids_np.shape:
            raise ValueError(
                f"coo_to_csc_index: pre_ids and indices must have the same shape, "
                f"got {pre_ids_np.shape} and {post_ids_np.shape}."
            )
        nse = int(post_ids_np.size)
        offset_dtype = _resolve_indptr_dtype(nse, requested=indptr_dtype)
        _as_int32_indices(pre_ids_np, n_pre, "coo_to_csc_index row ids", output_is_numpy=True)
        post_ids_i32 = _as_int32_indices(
            post_ids_np, n_post, "coo_to_csc_index column ids", output_is_numpy=True
        )

        new_post_position_np = np.argsort(post_ids_i32, kind='stable')
        pre_ids_new_np = pre_ids_np[new_post_position_np]
        unique_post_ids, count = np.unique(post_ids_i32, return_counts=True)
        post_count = np.zeros(n_post, dtype=offset_dtype)
        post_count[unique_post_ids] = count

        indptr_np = np.empty(n_post + 1, dtype=offset_dtype)
        indptr_np[0] = 0
        np.cumsum(post_count, dtype=offset_dtype, out=indptr_np[1:])

    indptr_new = _as_output_index_array(
        indptr_np, offset_dtype, output_is_numpy=output_is_numpy, context="coo_to_csc_index indptr"
    )
    pre_ids_new = _as_int32_indices(
        pre_ids_new_np, n_pre, "coo_to_csc_index row ids", output_is_numpy=output_is_numpy
    )
    new_post_position = _as_output_index_array(
        new_post_position_np, offset_dtype, output_is_numpy=output_is_numpy,
        context="coo_to_csc_index permutation",
    )
    return indptr_new, pre_ids_new, new_post_position


def coo2csr(
    row_ids: Union[jax.Array, np.ndarray],
    col_ids: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
    indptr_dtype="auto",
):
    """Convert COO format index arrays to CSR format.

    Transforms a sparse matrix representation from Coordinate (COO) format
    (explicit row and column index arrays) to Compressed Sparse Row (CSR)
    format. The implementation automatically selects NumPy or JAX operations
    based on the type of the input arrays.

    Parameters
    ----------
    row_ids : jax.Array or numpy.ndarray
        Row index array in COO format. Contains the row index for each
        non-zero element.
    col_ids : jax.Array or numpy.ndarray
        Column index array in COO format. Contains the column index for
        each non-zero element.
    shape : tuple of int
        A ``(n_rows, n_cols)`` tuple specifying the dimensions of the
        sparse matrix. Keyword-only argument.

    Returns
    -------
    csr_indptr : jax.Array or numpy.ndarray
        Row pointer array in CSR format. For a matrix with ``m`` rows, this
        has length ``m + 1``. Element ``csr_indptr[i]`` gives the position
        in ``csr_indices`` where row ``i`` starts.
    csr_indices : jax.Array or numpy.ndarray
        Column index array in CSR format. Contains the column index for each
        non-zero element, ordered by row.
    order : jax.Array or numpy.ndarray
        Permutation array that reorders data values from COO order to CSR
        order. If ``data`` is the COO data array, then ``data[order]`` gives
        the values in CSR order.

    See Also
    --------
    csr_to_coo_index : Convert CSR indices to COO format.
    coo_to_csc_index : Convert COO indices to CSC format.
    csr_to_csc_index : Convert CSR indices directly to CSC format.

    Notes
    -----
    When JAX arrays are provided, the computation is wrapped in
    ``jax.ensure_compile_time_eval()`` so that it executes at trace time
    rather than at runtime. Entries are grouped by row using a stable sort,
    so the relative order of elements within each row is preserved (the
    column indices within a row are not themselves sorted).

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._misc import coo2csr
        >>> row_ids = np.array([0, 2, 1, 0, 2])
        >>> col_ids = np.array([0, 3, 1, 2, 0])
        >>> indptr, indices, order = coo2csr(row_ids, col_ids, shape=(3, 4))
        >>> print(indptr)
        [0 2 3 5]
        >>> print(indices)
        [0 2 1 3 0]
    """
    n_pre, n_post = shape
    output_is_numpy = _is_numpy_output(row_ids, col_ids)
    with jax.ensure_compile_time_eval():
        row_ids_np = np.asarray(jax.device_get(row_ids))
        col_ids_np = np.asarray(jax.device_get(col_ids))
        if row_ids_np.shape != col_ids_np.shape:
            raise ValueError(
                f"coo2csr: row_ids and col_ids must have the same shape, "
                f"got {row_ids_np.shape} and {col_ids_np.shape}."
            )
        nse = int(col_ids_np.size)
        offset_dtype = _resolve_indptr_dtype(nse, requested=indptr_dtype)
        row_ids_i32 = _as_int32_indices(row_ids_np, n_pre, "coo2csr row ids", output_is_numpy=True)
        _as_int32_indices(col_ids_np, n_post, "coo2csr column ids", output_is_numpy=True)

        order_np = np.argsort(row_ids_i32, kind='stable')
        csr_indices_np = col_ids_np[order_np]
        unique_row_ids, count = np.unique(row_ids_i32, return_counts=True)
        row_count = np.zeros(n_pre, dtype=offset_dtype)
        row_count[unique_row_ids] = count

        csr_indptr_np = np.empty(n_pre + 1, dtype=offset_dtype)
        csr_indptr_np[0] = 0
        np.cumsum(row_count, dtype=offset_dtype, out=csr_indptr_np[1:])

    csr_indptr = _as_output_index_array(
        csr_indptr_np, offset_dtype, output_is_numpy=output_is_numpy, context="coo2csr indptr"
    )
    csr_indices = _as_int32_indices(
        csr_indices_np, n_post, "coo2csr column ids", output_is_numpy=output_is_numpy
    )
    order = _as_output_index_array(
        order_np, offset_dtype, output_is_numpy=output_is_numpy, context="coo2csr permutation"
    )
    return csr_indptr, csr_indices, order


def fixed_conn_num_csr_indptr(
    indices: Union[jax.Array, np.ndarray],
) -> Union[jax.Array, np.ndarray]:
    """Build the implicit CSR ``indptr`` for a fixed-connection matrix."""
    assert indices.ndim == 2, f'Indices must be 2D, got {indices.ndim}D.'
    n_pre, n_conn = indices.shape
    nse = int(n_pre) * int(n_conn)
    offset_dtype = _resolve_indptr_dtype(nse, requested="auto")
    indptr_np = np.arange(n_pre + 1, dtype=offset_dtype) * n_conn
    if isinstance(indices, np.ndarray):
        return indptr_np
    return _as_output_index_array(
        indptr_np, offset_dtype, output_is_numpy=False, context="fixed_conn_num_csr_indptr"
    )


def normalize_row_index(index, n_rows: int):
    """Normalize a row index/slice into an ``int32`` array for row slicing.

    Accepts ``int`` / ``list`` / ``tuple`` / ``np.ndarray`` / ``jax.Array`` and
    Python ``slice`` objects. A scalar ``int`` returns a 0-D array (so a dense
    slice yields a 1-D row); everything else returns a 1-D array. Negative
    indices wrap against ``n_rows`` (NumPy semantics). When the indices are
    concrete, out-of-bounds values raise :class:`IndexError`; traced indices
    (under ``jax.jit``) are left unchecked and the slice kernel zero-fills any
    out-of-bounds rows.

    Parameters
    ----------
    index : int, list, tuple, numpy.ndarray, jax.Array, or slice
        Row selector.
    n_rows : int
        Size of axis 0 of the logical matrix ``W``.

    Returns
    -------
    jax.Array
        ``int32`` row indices: 0-D for a scalar ``int``, otherwise 1-D.
    """
    if isinstance(index, slice):
        start, stop, step = index.indices(int(n_rows))
        return jnp.arange(start, stop, step, dtype=jnp.int32)

    arr = jnp.asarray(index)
    if not jnp.issubdtype(arr.dtype, jnp.integer):
        raise IndexError(f"Row index must be integer, got dtype {arr.dtype}.")
    arr = arr.astype(jnp.int32)
    arr = jnp.where(arr < 0, arr + n_rows, arr)

    if not isinstance(arr, Tracer):
        with jax.ensure_compile_time_eval():
            arr_np = np.asarray(arr)
        if arr_np.size and (arr_np.min() < 0 or arr_np.max() >= n_rows):
            raise IndexError(
                f"Row index out of bounds for axis 0 with size {n_rows}."
            )
    return arr


def build_sub_csr(data, indices, indptr, rows, n_cols: int):
    """Build the CSR arrays of ``W[rows, :]`` from a CSR-of-``W`` view.

    The output number of non-zeros depends on the *values* of ``indptr`` and
    ``rows``, so this requires concrete (non-traced) ``rows`` / ``indptr`` and
    must run outside ``jax.jit``. Homogeneous (size-1) ``data`` is returned
    unchanged (still size-1).

    Parameters
    ----------
    data : jax.Array
        CSR-view values, shape ``(nnz,)`` or ``(1,)`` (homogeneous).
    indices : jax.Array
        CSR-view column indices, shape ``(nnz,)``.
    indptr : jax.Array
        CSR-view row pointers, shape ``(n_rows + 1,)``.
    rows : jax.Array
        Concrete 1-D row indices to extract.
    n_cols : int
        Number of columns of ``W``.

    Returns
    -------
    new_data : jax.Array
    new_indices : jax.Array
    new_indptr : jax.Array
    shape : tuple of int
        ``(len(rows), n_cols)``.
    """
    if isinstance(rows, Tracer) or isinstance(indptr, Tracer):
        raise RuntimeError(
            "build_sub_csr requires concrete row indices and indptr; the sparse "
            "`slice_rows` for CSR/CSC/FixedNumPerPost has a data-dependent number "
            "of non-zeros and cannot run under jax.jit. Call it outside jit."
        )
    with jax.ensure_compile_time_eval():
        indptr_np = np.asarray(indptr)
        rows_np = np.asarray(rows).reshape(-1).astype(np.int64)
        starts = indptr_np[rows_np]
        ends = indptr_np[rows_np + 1]
        counts = (ends - starts).astype(np.int64)
        new_nse = int(counts.sum())
        offset_dtype = _resolve_indptr_dtype(new_nse, requested="auto")
        new_indptr_np = np.zeros(rows_np.shape[0] + 1, dtype=offset_dtype)
        np.cumsum(counts, dtype=offset_dtype, out=new_indptr_np[1:])
        if counts.sum() == 0:
            gather_np = np.zeros((0,), dtype=offset_dtype)
        else:
            gather_np = np.concatenate(
                [np.arange(s, e, dtype=offset_dtype) for s, e in zip(starts, ends)]
            )
    gather = _as_output_index_array(
        gather_np, offset_dtype, output_is_numpy=False, context="build_sub_csr gather"
    )
    new_indptr = _as_output_index_array(
        new_indptr_np, offset_dtype, output_is_numpy=False, context="build_sub_csr indptr"
    )
    new_indices = jnp.asarray(indices).reshape(-1)[gather]
    new_data = data if data.size == 1 else data.reshape(-1)[gather]
    return new_data, new_indices, new_indptr, (int(rows_np.shape[0]), int(n_cols))


def fixed_conn_num_csc_structure(
    indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
) -> Tuple[Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray]]:
    """Convert row-major FCN connectivity into compact CSC structure."""
    assert indices.ndim == 2, f'Indices must be 2D, got {indices.ndim}D.'
    n_pre, n_post = shape
    assert indices.shape[0] == n_pre, (
        f'Pre size mismatch: indices.shape[0] ({indices.shape[0]}) != shape[0] ({n_pre})'
    )

    csr_indptr = fixed_conn_num_csr_indptr(indices)
    flat_indices = indices.reshape(-1)
    if not isinstance(indices, Tracer):
        csc_indptr, csc_indices, perm = csr_to_csc_index(csr_indptr, flat_indices, shape=shape)
        if isinstance(indices, np.ndarray):
            return (
                np.asarray(csc_indptr),
                np.asarray(csc_indices, dtype=np.int32),
                np.asarray(perm),
            )
        return (
            csc_indptr,
            jnp.asarray(csc_indices, dtype=jnp.int32),
            perm,
        )

    nse = int(flat_indices.size)
    offset_dtype = _resolve_indptr_dtype(nse, requested="auto")
    _require_jax_x64_for_int64(offset_dtype, "fixed_conn_num_csc_structure")
    row_ids = jnp.repeat(jnp.arange(n_pre, dtype=jnp.int32), indices.shape[1])
    perm = jnp.argsort(flat_indices, stable=True).astype(jnp.dtype(offset_dtype))
    counts = jnp.bincount(flat_indices.astype(jnp.int32), length=n_post).astype(jnp.dtype(offset_dtype))
    indptr = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.dtype(offset_dtype)), jnp.cumsum(counts, dtype=jnp.dtype(offset_dtype))]
    )
    return indptr, row_ids[perm], perm


def fixed_conn_num_to_csc(
    weights: Union[jax.Array, np.ndarray],
    indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
) -> Tuple[Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray]]:
    """Build compact CSC mirrors for FCN weights and indices."""
    if weights.ndim == 0:
        weights = weights.reshape((1,))
    if weights.ndim == 1:
        assert weights.size == 1, (
            f'When weights is 1D, it should be a scalar (size 1), got {weights.size}.'
        )
    elif weights.ndim != 2:
        raise ValueError(f'weight dim should be 2, 1, or 0, but got {weights.ndim}')

    csc_indptr, csc_indices, perm = fixed_conn_num_csc_structure(indices, shape=shape)
    if weights.ndim == 1:
        return weights.reshape(1), csc_indices, csc_indptr

    return weights.reshape(-1)[perm], csc_indices, csc_indptr


def _csr_to_csc_index_via_coo(
    csr_indptr: Union[jax.Array, np.ndarray],
    csr_indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
    include_perm: bool = True,
    indptr_dtype="auto",
):
    """Convert CSR indices to CSC indices through the legacy COO path."""
    nse = int(getattr(csr_indices, 'size', len(csr_indices)))
    output_is_numpy = _is_numpy_output(csr_indptr, csr_indices)
    csr_indices = _as_int32_indices(
        csr_indices, shape[1], "csr_to_csc_index column indices", output_is_numpy=output_is_numpy
    )
    csr_indptr = _as_indptr(
        csr_indptr, nse, indptr_dtype, "csr_to_csc_index indptr", output_is_numpy=output_is_numpy
    )
    pre_ids, post_ids = csr_to_coo_index(csr_indptr, csr_indices)
    csc_indptr, csc_indices, post_positions = coo_to_csc_index(
        pre_ids, post_ids, shape=shape, indptr_dtype=indptr_dtype
    )
    if not include_perm:
        post_positions = None
    return csc_indptr, csc_indices, post_positions


def _csr_to_csc_index_numpy(
    csr_indptr: Union[jax.Array, np.ndarray],
    csr_indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
    include_perm: bool = True,
    indptr_dtype="auto",
):
    """Convert CSR indices to CSC on CPU with NumPy, then restore array type."""
    n_post = shape[1]
    nse = getattr(csr_indices, 'size', None)
    if nse is None:
        nse = len(csr_indices)
    nse = int(nse)
    offset_dtype = _resolve_indptr_dtype(nse, requested=indptr_dtype)
    output_is_numpy = _is_numpy_output(csr_indptr, csr_indices)

    csr_indptr_np = np.asarray(_as_indptr(
        csr_indptr, nse, indptr_dtype, "csr_to_csc_index indptr", output_is_numpy=True
    ))
    csr_indices_np = _as_int32_indices(
        csr_indices, n_post, "csr_to_csc_index column indices", output_is_numpy=True
    )

    counts = np.bincount(csr_indices_np, minlength=n_post).astype(offset_dtype, copy=False)
    csc_indptr_np = np.empty(n_post + 1, dtype=offset_dtype)
    csc_indptr_np[0] = 0
    np.cumsum(counts, dtype=offset_dtype, out=csc_indptr_np[1:])

    order_np = np.argsort(csr_indices_np, kind='stable')
    order_np = np.asarray(order_np, dtype=offset_dtype)
    csc_indices_np = np.searchsorted(csr_indptr_np, order_np, side='right') - 1
    csc_indices_np = _as_int32_indices(
        csc_indices_np, shape[0], "csr_to_csc_index row indices", output_is_numpy=True
    )
    perm_np = order_np if include_perm else None

    csc_indptr = _as_output_index_array(
        csc_indptr_np, offset_dtype, output_is_numpy=output_is_numpy,
        context="csr_to_csc_index indptr",
    )
    csc_indices = _as_int32_indices(
        csc_indices_np, shape[0], "csr_to_csc_index row indices", output_is_numpy=output_is_numpy
    )
    perm = None if perm_np is None else _as_output_index_array(
        perm_np, offset_dtype, output_is_numpy=output_is_numpy,
        context="csr_to_csc_index permutation",
    )
    return csc_indptr, csc_indices, perm


_CSR_TO_CSC_CUDA_MODULE = None


def _load_csr_to_csc_cuda_module():
    global _CSR_TO_CSC_CUDA_MODULE
    if _CSR_TO_CSC_CUDA_MODULE is None:
        from pathlib import Path

        from ._op import load_cuda_file

        _CSR_TO_CSC_CUDA_MODULE = load_cuda_file(
            Path(__file__).resolve().parent / "_csr" / "csr_to_csc.cu",
            name="csr_to_csc",
        )
    return _CSR_TO_CSC_CUDA_MODULE


def _csr_to_csc_index_gpu_column_block(
    csr_indptr: Union[jax.Array, np.ndarray],
    csr_indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
    include_perm: bool = True,
    column_block_size: int = 4096,
    indptr_dtype="auto",
):
    """Convert CSR indices to CSC using CUDA column blocks and CPU stitching."""
    n_post = shape[1]
    try:
        column_block_size = int(column_block_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("column_block_size must be a positive integer") from exc
    if column_block_size <= 0:
        raise ValueError("column_block_size must be a positive integer")

    nse = getattr(csr_indices, 'size', None)
    if nse is None:
        nse = len(csr_indices)
    nse = int(nse)
    offset_dtype = _resolve_indptr_dtype(nse, requested=indptr_dtype)
    output_is_numpy = _is_numpy_output(csr_indptr, csr_indices)

    try:
        gpu_device = jax.devices("gpu")[0]
        _load_csr_to_csc_cuda_module()
    except Exception:
        return _csr_to_csc_index_numpy(
            csr_indptr,
            csr_indices,
            shape=shape,
            include_perm=include_perm,
            indptr_dtype=indptr_dtype,
        )

    _require_jax_x64_for_int64(offset_dtype, "csr_to_csc_index gpu_column_block")
    try:
        csr_indices_i32 = _as_int32_indices(
            csr_indices, n_post, "csr_to_csc_index column indices", output_is_numpy=False
        )
        csr_indptr_checked = _as_indptr(
            csr_indptr, nse, offset_dtype, "csr_to_csc_index indptr", output_is_numpy=False
        )
        csr_indices_dev = jax.device_put(
            csr_indices_i32,
            gpu_device,
        )
        csr_indptr_dev = jax.device_put(
            csr_indptr_checked,
            gpu_device,
        )

        counts_dev = jax.ffi.ffi_call(
            "csr_to_csc.csr_to_csc_count",
            jax.ShapeDtypeStruct((n_post,), offset_dtype),
        )(csr_indices_dev, csr_indptr_dev)
        counts_np = np.asarray(counts_dev, dtype=offset_dtype)

        csc_indptr_np = np.empty(n_post + 1, dtype=offset_dtype)
        csc_indptr_np[0] = 0
        np.cumsum(counts_np, dtype=offset_dtype, out=csc_indptr_np[1:])

        if int(csc_indptr_np[-1]) != nse:
            raise RuntimeError(
                "CUDA CSR-to-CSC count produced an unexpected nnz total: "
                f"{int(csc_indptr_np[-1])} != {nse}"
            )

        csc_indices_np = np.empty(nse, dtype=np.int32)
        perm_np = np.empty(nse, dtype=offset_dtype) if include_perm else None

        for col_start in range(0, n_post, column_block_size):
            col_end = min(col_start + column_block_size, n_post)
            base = int(csc_indptr_np[col_start])
            end = int(csc_indptr_np[col_end])
            block_nnz = end - base
            block_ncols = col_end - col_start

            if block_nnz == 0:
                continue

            local_indptr_np = (
                csc_indptr_np[col_start:col_end + 1] -
                csc_indptr_np[col_start]
            ).astype(offset_dtype, copy=False)
            initial_pos_dev = jax.device_put(
                jnp.asarray(local_indptr_np[:-1], dtype=offset_dtype),
                gpu_device,
            )

            scratch_info = jax.ShapeDtypeStruct((block_ncols,), offset_dtype)
            rows_info = jax.ShapeDtypeStruct((block_nnz,), jnp.int32)
            perm_info = jax.ShapeDtypeStruct((block_nnz,), offset_dtype)
            _, local_rows_dev, local_perm_dev = jax.ffi.ffi_call(
                "csr_to_csc.csr_to_csc_fill_block",
                (scratch_info, rows_info, perm_info),
            )(
                csr_indices_dev,
                csr_indptr_dev,
                initial_pos_dev,
                col_start=np.int64(col_start),
                col_end=np.int64(col_end),
            )

            csc_indices_np[base:end] = np.asarray(local_rows_dev, dtype=np.int32)
            if perm_np is not None:
                perm_np[base:end] = np.asarray(local_perm_dev, dtype=offset_dtype)

        if output_is_numpy:
            return csc_indptr_np, csc_indices_np, perm_np

        csc_indptr = jax.device_put(csc_indptr_np, gpu_device)
        csc_indices = jax.device_put(csc_indices_np, gpu_device)
        perm = None if perm_np is None else jax.device_put(perm_np, gpu_device)
        return csc_indptr, csc_indices, perm
    except Exception:
        raise


def csr_to_csc_index(
    csr_indptr: Union[jax.Array, np.ndarray],
    csr_indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
    include_perm: bool = True,
    method: str = "gpu_column_block",
    column_block_size: int = 4096,
    indptr_dtype="auto",
):
    """Convert CSR format index arrays to CSC format.

    Transforms the sparse matrix representation from Compressed Sparse Row
    (CSR) format to Compressed Sparse Column (CSC) format. The default
    ``method="coo"`` preserves the legacy CSR -> COO -> CSC behavior.

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
    include_perm : bool, optional
        If ``True`` (default), return the permutation that maps CSC slots back
        to CSR data positions. If ``False``, return ``None`` for the third
        result while still constructing the CSC structure.
    method : {"coo", "numpy", "gpu_column_block"}, optional
        Conversion algorithm. ``"coo"`` expands CSR to COO first and then
        converts COO to CSC; ``"numpy"`` computes the structure on CPU with
        NumPy; ``"gpu_column_block"`` builds consecutive CSC column blocks
        with CUDA kernels and falls back to ``"numpy"`` when CUDA is not
        available.
    column_block_size : int, optional
        Number of CSC columns per CUDA block for ``method="gpu_column_block"``.

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
    The returned ``post_positions`` permutation array can be used to reorder a
    CSR data array into CSC order. The ``"coo"`` method preserves the public
    int32 index-helper contract, while ``"numpy"`` and ``"gpu_column_block"``
    may return int64 offset arrays when the input row pointer requires them.

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
    if method == "coo":
        csc_indptr, csc_indices, post_positions = _csr_to_csc_index_via_coo(
            csr_indptr,
            csr_indices,
            shape=shape,
            include_perm=include_perm,
            indptr_dtype=indptr_dtype,
        )
    elif method == "numpy":
        csc_indptr, csc_indices, post_positions = _csr_to_csc_index_numpy(
            csr_indptr,
            csr_indices,
            shape=shape,
            include_perm=include_perm,
            indptr_dtype=indptr_dtype,
        )
    elif method == "gpu_column_block":
        csc_indptr, csc_indices, post_positions = _csr_to_csc_index_gpu_column_block(
            csr_indptr,
            csr_indices,
            shape=shape,
            include_perm=include_perm,
            column_block_size=column_block_size,
            indptr_dtype=indptr_dtype,
        )
    else:
        raise ValueError(
            f"Unknown csr_to_csc_index method {method!r}; "
            f"expected 'coo', 'numpy', or 'gpu_column_block'."
        )
    return csc_indptr, csc_indices, post_positions


def csc_to_csr_index(
    csc_indptr: Union[jax.Array, np.ndarray],
    csc_indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
    include_perm: bool = True,
    indptr_dtype="auto",
):
    """Convert CSC format index arrays to CSR format.

    Inverse companion of :func:`csr_to_csc_index`.  A Compressed Sparse Column
    layout of a matrix ``W`` with shape ``(n_rows, n_cols)`` is, array for array,
    the Compressed Sparse Row layout of ``W.T`` with shape ``(n_cols, n_rows)``.
    Building the CSR structure of ``W`` therefore reduces to calling
    :func:`csr_to_csc_index` on the transposed interpretation.

    Parameters
    ----------
    csc_indptr : jax.Array or numpy.ndarray
        Column pointer array in CSC format.  For a matrix with ``n_cols``
        columns, this has length ``n_cols + 1``.
    csc_indices : jax.Array or numpy.ndarray
        Row index array in CSC format.  Contains the row index for each
        non-zero element, ordered by column.
    shape : tuple of int
        A ``(n_rows, n_cols)`` tuple giving the dimensions of the matrix the
        CSC arrays describe.  Keyword-only argument.
    include_perm : bool, optional
        If ``True`` (default), return the permutation that maps CSR slots back
        to CSC data positions. If ``False``, return ``None`` for the third
        result while still constructing the CSR structure.

    Returns
    -------
    csr_indptr : jax.Array or numpy.ndarray
        Row pointer array in CSR format.  Length ``n_rows + 1``.
    csr_indices : jax.Array or numpy.ndarray
        Column index array in CSR format.
    perm : jax.Array or numpy.ndarray
        Permutation array reordering data values from CSC order to CSR order.
        If ``data`` is the CSC data array, then ``data[perm]`` gives the values
        in CSR order.

    Raises
    ------
    AssertionError
        If ``shape`` is not a length-2 tuple/list of positive integers.

    See Also
    --------
    csr_to_csc_index : The forward CSR-to-CSC companion (mutual inverse).
    coo_to_csc_index : Convert COO indices to CSC indices.

    Notes
    -----
    Because the two helpers are mutual inverses on the same structure, the
    permutations they return compose to the identity:
    ``csr_perm[csc_perm] == arange(nse)``.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> from brainevent._misc import csr_to_csc_index, csc_to_csr_index
        >>> indptr = np.array([0, 2, 3, 5])
        >>> indices = np.array([0, 2, 1, 0, 3])
        >>> csc_indptr, csc_indices, _ = csr_to_csc_index(indptr, indices, shape=(3, 4))
        >>> csr_indptr, csr_indices, _ = csc_to_csr_index(
        ...     csc_indptr, csc_indices, shape=(3, 4)
        ... )
    """
    assert isinstance(shape, (tuple, list)), "Shape must be a tuple or list"
    assert len(shape) == 2, "Shape must have exactly two dimensions (rows, columns)"
    assert shape[0] > 0 and shape[1] > 0, "Shape dimensions must be positive integers"
    csr_indptr, csr_indices, perm = csr_to_csc_index(
        csc_indptr,
        csc_indices,
        shape=(shape[1], shape[0]),
        include_perm=include_perm,
        indptr_dtype=indptr_dtype,
        method="coo",
    )
    return csr_indptr, csr_indices, perm


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
        self._cache: dict[Optional[str], Callable] = {}  # backend -> jit_compiled_fn
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
    fn: Optional[Callable] = None,
    name: Optional[str] = None,
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
