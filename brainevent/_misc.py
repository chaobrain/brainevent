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


def _offset_index_dtype(nse: int, preferred=None):
    if preferred is not None and _normalize_dtype(preferred) == np.dtype(np.int64):
        return np.int64
    return np.int64 if int(nse) > _INT32_MAX else np.int32


# ---------------------------------------------------------------------------
# Unified index/offset dtype policy.
#
# ``indices`` are secondary-axis coordinates (CSR column / CSC row) and are
# *always* int32 -- if a coordinate cannot be represented in int32 that is an
# error, not a reason to widen. ``indptr``/offset arrays are cumulative ``nnz``
# offsets and auto-promote to int64 only when ``nnz`` exceeds the int32 range.
# Creating a JAX int64 array requires ``jax_enable_x64``; the library never
# toggles that global config on the user's behalf -- it raises instead.
# ---------------------------------------------------------------------------

def _resolve_indptr_dtype(nse, requested="auto"):
    """Resolve the dtype of an ``indptr``/offset array holding ``nse`` offsets.

    Parameters
    ----------
    nse : int
        Number of stored elements (the largest offset value).
    requested : {"auto"} or dtype-like, optional
        ``"auto"`` (default) selects int64 when ``nse > int32_max`` else int32.
        An explicit ``int32`` raises :class:`OverflowError` when ``nse`` cannot
        be represented. An explicit ``int64`` is always honoured (the caller is
        responsible for gating on ``jax_enable_x64`` via
        :func:`_require_jax_x64_for_int64`).

    Returns
    -------
    numpy.dtype
        Either ``int32`` or ``int64``.
    """
    nse = int(nse)
    if isinstance(requested, str):
        if requested != "auto":
            raise ValueError(
                f"indptr_dtype must be 'auto', int32, or int64; got {requested!r}."
            )
        return np.dtype(np.int64) if nse > _INT32_MAX else np.dtype(np.int32)
    requested_dtype = _normalize_dtype(requested)
    if requested_dtype == np.dtype(np.int32):
        if nse > _INT32_MAX:
            raise OverflowError(
                f"nnz={nse} exceeds the int32 range ({_INT32_MAX}); request "
                "indptr_dtype='auto' or int64 (int64 requires jax_enable_x64)."
            )
        return np.dtype(np.int32)
    if requested_dtype == np.dtype(np.int64):
        return np.dtype(np.int64)
    raise ValueError(
        f"indptr_dtype must be 'auto', int32, or int64; got {requested!r}."
    )


def _require_jax_x64_for_int64(dtype, context):
    """Raise if ``dtype`` is int64 but JAX x64 is disabled.

    Creating a JAX int64 array while ``jax_enable_x64=False`` silently downcasts
    to int32. Rather than mutate the global config, we refuse and tell the user
    to enable x64 at program startup.
    """
    if _normalize_dtype(dtype) == np.dtype(np.int64) and not jax.config.jax_enable_x64:
        raise ValueError(
            f"{context} requires an int64 array, but JAX x64 is disabled. Enable "
            "it at program startup via "
            "`jax.config.update('jax_enable_x64', True)` (or set the environment "
            "variable `JAX_ENABLE_X64=1`) before constructing large sparse "
            "structures. The library does not toggle this global config for you."
        )


def _as_int32_indices(indices, secondary_dim, context, check_values=True):
    """Validate and cast a CSR/CSC ``indices`` array to int32.

    ``indices`` are secondary-axis coordinates and are always int32. Under a JAX
    tracer only static dtype is enforced (int64 tracers are rejected); host value
    checks are skipped. When ``check_values`` is ``False`` (structure-preserving
    reconstruction of already-validated arrays) the concrete value checks are
    skipped and only a dtype coercion is performed.
    """
    arr = u.math.asarray(indices)
    dtype = jnp.dtype(arr.dtype)
    if not jnp.issubdtype(dtype, jnp.integer):
        raise TypeError(f"{context}: indices must be an integer array; got dtype {dtype}.")
    if secondary_dim is not None and int(secondary_dim) > _INT32_MAX + 1:
        raise OverflowError(
            f"{context}: secondary dimension {int(secondary_dim)} exceeds the "
            "int32-representable coordinate range."
        )
    if isinstance(arr, Tracer):
        if dtype == jnp.dtype(jnp.int64):
            raise TypeError(
                f"{context}: indices must be int32 (secondary-axis coordinates), "
                "but a traced int64 array was supplied."
            )
        return arr if dtype == jnp.dtype(jnp.int32) else arr.astype(jnp.int32)
    if check_values:
        host = np.asarray(jax.device_get(arr))
        if host.size:
            min_v = int(host.min())
            max_v = int(host.max())
            if min_v < 0:
                raise ValueError(f"{context}: indices must be non-negative; got minimum {min_v}.")
            if secondary_dim is not None and max_v >= int(secondary_dim):
                raise ValueError(
                    f"{context}: index {max_v} is out of bounds for secondary "
                    f"dimension {int(secondary_dim)}."
                )
            if max_v > _INT32_MAX:
                raise OverflowError(
                    f"{context}: index {max_v} exceeds the int32 range; "
                    "secondary-axis coordinates must fit int32."
                )
    return arr if dtype == jnp.dtype(jnp.int32) else arr.astype(jnp.int32)


def _as_indptr(indptr, nse, indptr_dtype, context):
    """Validate and cast an ``indptr``/offset array with safe precision.

    Resolves the target dtype from ``nse`` and the requested ``indptr_dtype``,
    gates int64 on ``jax_enable_x64``, and casts without silent int64->int32
    truncation (the target is int32 only when ``nse`` fits int32, and every
    offset is bounded by ``nse``).
    """
    arr = u.math.asarray(indptr)
    src_dtype = jnp.dtype(arr.dtype)
    if not jnp.issubdtype(src_dtype, jnp.integer):
        raise TypeError(f"{context}: indptr must be an integer array; got dtype {src_dtype}.")
    target = _resolve_indptr_dtype(nse, indptr_dtype)
    _require_jax_x64_for_int64(target, context)
    if src_dtype == jnp.dtype(target):
        return arr
    return arr.astype(target)


def _check_compressed_structure(indices, indptr, shape, format="csr", check_values=True):
    """Validate CSR/CSC structural invariants.

    Static checks (ndim, dtype, indptr length, x64 gating for int64 indptr) run
    for both concrete and traced arrays. Value checks (``indptr[0] == 0``,
    monotonicity, ``indptr[-1] == indices.size``) run only for concrete arrays
    and only when ``check_values`` is set.
    """
    fmt = format.lower()
    if fmt not in ("csr", "csc"):
        raise ValueError(f"format must be 'csr' or 'csc'; got {format!r}.")
    primary_dim = shape[0] if fmt == "csr" else shape[1]

    if indices.ndim != 1:
        raise ValueError(f"{fmt} indices must be a 1D array; got ndim {indices.ndim}.")
    if indptr.ndim != 1:
        raise ValueError(f"{fmt} indptr must be a 1D array; got ndim {indptr.ndim}.")
    idx_dtype = jnp.dtype(indices.dtype)
    ptr_dtype = jnp.dtype(indptr.dtype)
    if idx_dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"{fmt} indices must be int32; got {idx_dtype}.")
    if ptr_dtype not in (jnp.dtype(jnp.int32), jnp.dtype(jnp.int64)):
        raise TypeError(f"{fmt} indptr must be int32 or int64; got {ptr_dtype}.")
    _require_jax_x64_for_int64(ptr_dtype, f"{fmt} indptr")
    if indptr.shape[0] != int(primary_dim) + 1:
        raise ValueError(
            f"{fmt} indptr length must be primary dimension + 1 "
            f"({int(primary_dim) + 1}); got {indptr.shape[0]}."
        )

    if not check_values or isinstance(indices, Tracer) or isinstance(indptr, Tracer):
        return

    ptr_host = np.asarray(jax.device_get(indptr))
    if ptr_host.size and int(ptr_host[0]) != 0:
        raise ValueError(f"{fmt} indptr[0] must be 0; got {int(ptr_host[0])}.")
    if np.any(np.diff(ptr_host) < 0):
        raise ValueError(f"{fmt} indptr must be monotonically non-decreasing.")
    if int(ptr_host[-1]) != int(indices.shape[0]):
        raise ValueError(
            f"{fmt} indptr[-1] ({int(ptr_host[-1])}) must equal the number of "
            f"stored elements ({int(indices.shape[0])})."
        )


def _as_int32_cuda_offsets(offsets, context):
    """Guard int32-only CUDA/JITC offset ABIs.

    Raises :class:`NotImplementedError` when ``offsets`` is int64 rather than
    silently truncating to int32. int32 offsets pass through unchanged.
    """
    arr = u.math.asarray(offsets)
    if jnp.dtype(arr.dtype) == jnp.dtype(jnp.int64):
        raise NotImplementedError(
            f"{context}: int64 offset arrays are not supported by this CUDA "
            "kernel (int32 ABI). The structure exceeds the int32 offset range; "
            "an int64-capable path is required."
        )
    return arr if jnp.dtype(arr.dtype) == jnp.dtype(jnp.int32) else arr.astype(jnp.int32)


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

    ``indices`` are secondary-axis coordinates and must be int32. ``indptr`` is
    an offset array and may be int32 or int64 (auto-promoted when ``nnz``
    exceeds the int32 range).
    """
    indices_dtype = jnp.dtype(indices.dtype)
    indptr_dtype = jnp.dtype(indptr.dtype)
    assert indices_dtype == jnp.dtype(jnp.int32), (
        f"Indices must be int32 (secondary-axis coordinates); got {indices_dtype}."
    )
    assert indptr_dtype in _CSR_SIGNED_INDEX_DTYPES, (
        f"Indptr must be int32 or int64; got {indptr_dtype}."
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
    # ``indices`` (row coordinates) are always int32; ``indptr`` offsets and the
    # ``perm`` permutation index into the nnz array and follow the nnz-resolved
    # offset dtype (auto-promoting to int64, gated on ``jax_enable_x64``).
    coord_dtype = np.dtype(np.int32)
    nse = int(np.asarray(indices).size) if isinstance(indices, np.ndarray) else int(indices.size)
    offset_dtype = _resolve_indptr_dtype(nse, "auto")
    _require_jax_x64_for_int64(offset_dtype, "coo_to_csc_index")
    if isinstance(indices, np.ndarray) and isinstance(pre_ids, np.ndarray):
        # to maintain the original order of the elements with the same value
        new_post_position = np.asarray(np.argsort(indices, kind='stable'), dtype=offset_dtype)
        pre_ids_new = np.asarray(pre_ids[new_post_position], dtype=coord_dtype)

        unique_post_ids, count = np.unique(indices, return_counts=True)
        post_count = np.zeros(n_post, dtype=offset_dtype)
        post_count[unique_post_ids] = count

        indptr_new = np.insert(post_count.cumsum(), 0, 0)
        indptr_new = np.asarray(indptr_new, dtype=offset_dtype)

    else:
        # to maintain the original order of the elements with the same value

        with jax.ensure_compile_time_eval():
            new_post_position = jnp.asarray(jnp.argsort(indices, stable=True), dtype=offset_dtype)
            pre_ids_new = jnp.asarray(pre_ids[new_post_position], dtype=coord_dtype)

            unique_post_ids, count = jnp.unique(indices, return_counts=True)
            post_count = jnp.zeros(n_post, dtype=offset_dtype)
            post_count = post_count.at[unique_post_ids].set(count)

            indptr_new = jnp.insert(post_count.cumsum(), 0, 0)
            indptr_new = jnp.asarray(indptr_new, dtype=offset_dtype)

    return indptr_new, pre_ids_new, new_post_position


def coo2csr(
    row_ids: Union[jax.Array, np.ndarray],
    col_ids: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
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
    n_pre = shape[0]
    # ``indices`` (column coordinates) are always int32; ``indptr`` offsets and
    # the ``order`` permutation index into the nnz array and follow the
    # nnz-resolved offset dtype (auto-promoting to int64, gated on
    # ``jax_enable_x64``).
    coord_dtype = np.dtype(np.int32)
    nse = int(np.asarray(row_ids).size) if isinstance(row_ids, np.ndarray) else int(row_ids.size)
    offset_dtype = _resolve_indptr_dtype(nse, "auto")
    _require_jax_x64_for_int64(offset_dtype, "coo2csr")
    if isinstance(row_ids, np.ndarray) and isinstance(col_ids, np.ndarray):
        # stable sort keeps the original order of entries within each row
        order = np.asarray(np.argsort(row_ids, kind='stable'), dtype=offset_dtype)
        csr_indices = np.asarray(col_ids[order], dtype=coord_dtype)

        unique_row_ids, count = np.unique(row_ids, return_counts=True)
        row_count = np.zeros(n_pre, dtype=offset_dtype)
        row_count[unique_row_ids] = count

        csr_indptr = np.insert(row_count.cumsum(), 0, 0)
        csr_indptr = np.asarray(csr_indptr, dtype=offset_dtype)

    else:
        with jax.ensure_compile_time_eval():
            # stable sort keeps the original order of entries within each row
            order = jnp.asarray(jnp.argsort(row_ids, stable=True), dtype=offset_dtype)
            csr_indices = jnp.asarray(col_ids[order], dtype=coord_dtype)

            unique_row_ids, count = jnp.unique(row_ids, return_counts=True)
            row_count = jnp.zeros(n_pre, dtype=offset_dtype)
            row_count = row_count.at[unique_row_ids].set(count)

            csr_indptr = jnp.insert(row_count.cumsum(), 0, 0)
            csr_indptr = jnp.asarray(csr_indptr, dtype=offset_dtype)

    return csr_indptr, csr_indices, order


def fixed_conn_num_csr_indptr(
    indices: Union[jax.Array, np.ndarray],
) -> Union[jax.Array, np.ndarray]:
    """Build the implicit CSR ``indptr`` for a fixed-connection matrix.

    ``indices`` are always int32 (secondary-axis coordinates), but the ``indptr``
    offsets range up to ``n_pre * n_conn`` (the nnz), which may exceed the int32
    range for very large matrices. The offset dtype is therefore resolved from
    the nnz (auto-promoting to int64, gated on ``jax_enable_x64``) rather than
    inherited from ``indices.dtype``.
    """
    assert indices.ndim == 2, f'Indices must be 2D, got {indices.ndim}D.'
    n_pre, n_conn = indices.shape
    nse = int(n_pre) * int(n_conn)
    offset_dtype = _resolve_indptr_dtype(nse, "auto")
    _require_jax_x64_for_int64(offset_dtype, "fixed_conn_num_csr_indptr")
    if isinstance(indices, np.ndarray):
        return np.arange(n_pre + 1, dtype=offset_dtype) * n_conn
    return jnp.arange(n_pre + 1, dtype=offset_dtype) * n_conn


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
        new_indptr_np = np.zeros(rows_np.shape[0] + 1, dtype=indptr_np.dtype)
        np.cumsum(counts, out=new_indptr_np[1:])
        if counts.sum() == 0:
            gather_np = np.zeros((0,), dtype=np.int64)
        else:
            gather_np = np.concatenate(
                [np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)]
            )
    gather = jnp.asarray(gather_np)
    new_indptr = jnp.asarray(new_indptr_np)
    new_indices = jnp.asarray(indices).reshape(-1)[gather]
    new_data = data if data.size == 1 else data.reshape(-1)[gather]
    return new_data, new_indices, new_indptr, (int(rows_np.shape[0]), int(n_cols))


def fixed_conn_num_csc_structure(
    indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
) -> Tuple[Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray], Union[jax.Array, np.ndarray]]:
    """Convert row-major FCN connectivity into compact CSC structure.

    ``indices`` (row ids) are always int32, but the ``indptr`` offsets and the
    ``perm`` permutation index into the flattened nnz array, so they follow the
    nnz-resolved offset dtype (auto-promoting to int64, gated on
    ``jax_enable_x64``) rather than the int32 ``indices`` dtype.
    """
    assert indices.ndim == 2, f'Indices must be 2D, got {indices.ndim}D.'
    n_pre, n_post = shape
    assert indices.shape[0] == n_pre, (
        f'Pre size mismatch: indices.shape[0] ({indices.shape[0]}) != shape[0] ({n_pre})'
    )

    nse = int(indices.shape[0]) * int(indices.shape[1])
    offset_dtype = _resolve_indptr_dtype(nse, "auto")
    _require_jax_x64_for_int64(offset_dtype, "fixed_conn_num_csc_structure")
    coord_dtype = np.dtype(np.int32)

    csr_indptr = fixed_conn_num_csr_indptr(indices)
    flat_indices = indices.reshape(-1)
    if not isinstance(indices, Tracer):
        csc_indptr, csc_indices, perm = csr_to_csc_index(csr_indptr, flat_indices, shape=shape)
        if isinstance(indices, np.ndarray):
            return (
                np.asarray(csc_indptr, dtype=offset_dtype),
                np.asarray(csc_indices, dtype=coord_dtype),
                np.asarray(perm, dtype=offset_dtype),
            )
        return (
            jnp.asarray(csc_indptr, dtype=offset_dtype),
            jnp.asarray(csc_indices, dtype=coord_dtype),
            jnp.asarray(perm, dtype=offset_dtype),
        )

    row_ids = jnp.repeat(jnp.arange(n_pre, dtype=coord_dtype), indices.shape[1])
    perm = jnp.argsort(flat_indices, stable=True).astype(offset_dtype)
    counts = jnp.bincount(flat_indices, length=n_post).astype(offset_dtype)
    indptr = jnp.concatenate(
        [jnp.zeros(1, dtype=offset_dtype), jnp.cumsum(counts, dtype=offset_dtype)]
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
):
    """Convert CSR indices to CSC indices through the legacy COO path."""
    pre_ids, post_ids = csr_to_coo_index(csr_indptr, csr_indices)
    csc_indptr, csc_indices, post_positions = coo_to_csc_index(pre_ids, post_ids, shape=shape)
    if not include_perm:
        post_positions = None
    return csc_indptr, csc_indices, post_positions


def _csr_to_csc_index_numpy(
    csr_indptr: Union[jax.Array, np.ndarray],
    csr_indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
    include_perm: bool = True,
):
    """Convert CSR indices to CSC on CPU with NumPy, then restore array type."""
    n_post = shape[1]
    # indices are secondary-axis coordinates -> always int32.
    coord_dtype = np.dtype(np.int32)
    nse = getattr(csr_indices, 'size', None)
    if nse is None:
        nse = len(csr_indices)
    # offsets auto-promote to int64 only when nnz exceeds the int32 range.
    offset_dtype = _offset_index_dtype(nse)

    csr_indptr_np = np.asarray(csr_indptr)
    csr_indices_np = np.asarray(csr_indices)

    counts = np.bincount(csr_indices_np, minlength=n_post).astype(offset_dtype, copy=False)
    csc_indptr_np = np.empty(n_post + 1, dtype=offset_dtype)
    csc_indptr_np[0] = 0
    np.cumsum(counts, dtype=offset_dtype, out=csc_indptr_np[1:])

    order_np = np.argsort(csr_indices_np, kind='stable')
    order_np = np.asarray(order_np, dtype=offset_dtype)
    csc_indices_np = np.searchsorted(csr_indptr_np, order_np, side='right') - 1
    csc_indices_np = np.asarray(csc_indices_np, dtype=coord_dtype)
    perm_np = order_np if include_perm else None

    if isinstance(csr_indptr, np.ndarray) and isinstance(csr_indices, np.ndarray):
        return csc_indptr_np, csc_indices_np, perm_np

    # Creating int64 JAX arrays requires x64; refuse rather than toggle the
    # global config behind the user's back.
    _require_jax_x64_for_int64(offset_dtype, "csr_to_csc_index (numpy)")
    csc_indptr = jnp.asarray(csc_indptr_np)
    csc_indices = jnp.asarray(csc_indices_np)
    perm = None if perm_np is None else jnp.asarray(perm_np)
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
):
    """Convert CSR indices to CSC using CUDA column blocks and CPU stitching."""
    n_post = shape[1]
    try:
        column_block_size = int(column_block_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("column_block_size must be a positive integer") from exc
    if column_block_size <= 0:
        raise ValueError("column_block_size must be a positive integer")

    # indices are secondary-axis coordinates -> always int32.
    coord_dtype = np.dtype(np.int32)
    nse = getattr(csr_indices, 'size', None)
    if nse is None:
        nse = len(csr_indices)
    nse = int(nse)
    # offsets auto-promote to int64 only when nnz exceeds the int32 range.
    offset_dtype = _offset_index_dtype(nse)

    # Creating int64 JAX arrays requires x64; refuse rather than toggle the
    # global config behind the user's back.
    _require_jax_x64_for_int64(offset_dtype, "csr_to_csc_index (gpu_column_block)")

    try:
        gpu_device = jax.devices("gpu")[0]
        _load_csr_to_csc_cuda_module()
    except Exception:
        return _csr_to_csc_index_numpy(
            csr_indptr,
            csr_indices,
            shape=shape,
            include_perm=include_perm,
        )

    csr_indices_dev = jax.device_put(
        jnp.asarray(csr_indices, dtype=coord_dtype),
        gpu_device,
    )
    csr_indptr_dev = jax.device_put(
        jnp.asarray(csr_indptr, dtype=offset_dtype),
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

    csc_indices_np = np.empty(nse, dtype=coord_dtype)
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
        rows_info = jax.ShapeDtypeStruct((block_nnz,), coord_dtype)
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

        csc_indices_np[base:end] = np.asarray(local_rows_dev, dtype=coord_dtype)
        if perm_np is not None:
            perm_np[base:end] = np.asarray(local_perm_dev, dtype=offset_dtype)

    if isinstance(csr_indptr, np.ndarray) and isinstance(csr_indices, np.ndarray):
        return csc_indptr_np, csc_indices_np, perm_np

    csc_indptr = jax.device_put(csc_indptr_np, gpu_device)
    csc_indices = jax.device_put(csc_indices_np, gpu_device)
    perm = None if perm_np is None else jax.device_put(perm_np, gpu_device)
    return csc_indptr, csc_indices, perm


def csr_to_csc_index(
    csr_indptr: Union[jax.Array, np.ndarray],
    csr_indices: Union[jax.Array, np.ndarray],
    *,
    shape: Tuple[int, int],
    include_perm: bool = True,
    method: str = "coo",
    column_block_size: int = 4096,
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
    CSR data array into CSC order. Across all methods ``csc_indices`` are always
    int32 (secondary-axis coordinates), while ``csc_indptr`` / ``post_positions``
    auto-promote to int64 when the nnz exceeds the int32 range (gated on
    ``jax_enable_x64``).

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
        )
    elif method == "numpy":
        csc_indptr, csc_indices, post_positions = _csr_to_csc_index_numpy(
            csr_indptr,
            csr_indices,
            shape=shape,
            include_perm=include_perm,
        )
    elif method == "gpu_column_block":
        csc_indptr, csc_indices, post_positions = _csr_to_csc_index_gpu_column_block(
            csr_indptr,
            csr_indices,
            shape=shape,
            include_perm=include_perm,
            column_block_size=column_block_size,
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
