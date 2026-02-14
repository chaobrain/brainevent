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


import operator
from typing import Any, Tuple, Optional, Dict

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._csr import (
    CSR, binary_csrmv, binary_csrmm, csrmv, csrmm,
    update_csr_on_binary_pre, update_csr_on_binary_post
)
from brainevent._data import DataRepresentation
from brainevent._event import BinaryArray, SparseFloat, EventRepresentation
from brainevent._misc import _coo_todense, COOInfo
from brainevent._typing import MatrixShape, Data, Index
from .binary import binary_coomv, binary_coomm
from .float import coomv, coomm
from .plasticity_binary import (
    update_coo_on_binary_pre,
    update_coo_on_binary_post,
)

__all__ = [
    'COO',
]


@jax.tree_util.register_pytree_node_class
class COO(DataRepresentation):
    """
    Coordinate Format (COO) sparse matrix.

    This class represents a sparse matrix in coordinate format, where non-zero
    elements are stored as triplets (row, column, value).

    The class supports arithmetic with dense arrays and scalars, and sparse-dense
    matrix multiplication via ``@``. Sparse-sparse operations are intentionally
    limited and may raise ``NotImplementedError``.

    Attributes
    ----------
    data : jax.Array, Quantity
        Array of the non-zero values in the matrix.
    row : jax.Array
        Array of row indices for each non-zero element.
    col : jax.Array
        Array of column indices for each non-zero element.
    shape : tuple[int, int]
        Shape of the matrix (rows, columns).
    nse : int
        Number of stored elements (property).
    dtype : dtype
        Data type of the matrix elements (property).
    info : COOInfo
        Additional information about the matrix structure (property).
    _bufs : tuple
        Tuple of ``(data, row, col, ptr)`` arrays (property).
    rows_sorted : bool
        Whether row indices are sorted.
    cols_sorted : bool
        Whether column indices are sorted within each row.
    ptr : jax.Array or None
        Row (or column) pointer array computed when indices are sorted.

    Note
    -----
    This class is registered as a PyTree node for JAX, allowing it to be used
    with JAX transformations and compiled functions.
    """
    __module__ = 'brainevent'

    data: Data
    row: Index
    col: Index
    shape: MatrixShape
    rows_sorted: bool
    cols_sorted: bool
    ptr: Index | None

    def __init__(
        self,
        data,
        row=None,
        col=None,
        ptr=None,
        *,
        shape: MatrixShape,
        rows_sorted: bool = False,
        cols_sorted: bool = False,
        backend: Optional[str] = None,
        buffers: Optional[Dict] = None,
    ):
        """
        Initialize a COO matrix.

        Supports two calling conventions::

            # Tuple syntax (original)
            COO((data, row, col), shape=(m, n))

            # Positional-argument syntax
            COO(data, row, col, shape=(m, n))

        Parameters
        ----------
        data : array or Sequence
            Either a single array (the non-zero values) when ``row`` and
            ``col`` are also provided, or a sequence of three arrays
            ``(data, row, col)`` when used with the tuple syntax.
        row : array, optional
            Row indices for each non-zero element. Required when ``data``
            is the values array.
        col : array, optional
            Column indices for each non-zero element. Required when ``data``
            is the values array.
        ptr : array, optional
            Pre-computed row (or column) pointer array. If None and indices
            are sorted, it will be computed automatically.
        shape : Tuple[int, int]
            Shape of the matrix as ``(num_rows, num_columns)``.
        rows_sorted : bool, optional
            Whether row indices are sorted. Default is False.
        cols_sorted : bool, optional
            Whether column indices are sorted within each row. Default is False.
        """
        if row is None and col is None:
            # Tuple syntax: COO((data, row, col), shape=...)
            args = data
        else:
            # Positional syntax: COO(data, row, col, shape=...)
            args = (data, row, col)

        assert len(args) == 3, "Expected three arguments: data, row, col."
        self.data, self.row, self.col = map(u.math.asarray, args)
        self.rows_sorted = rows_sorted
        self.cols_sorted = cols_sorted

        if rows_sorted and cols_sorted:
            raise ValueError(
                f'Both rows_sorted and cols_sorted cannot be True for COO. '
                f'Received rows_sorted={rows_sorted}, cols_sorted={cols_sorted}.'
            )

        if ptr is None and (rows_sorted or cols_sorted):
            sorted_idx = self.row if rows_sorted else self.col
            length = shape[0] if rows_sorted else shape[1]
            counts = jnp.bincount(sorted_idx, length=length)
            ptr = jnp.concatenate([jnp.zeros(1, dtype=self.row.dtype), jnp.cumsum(counts, dtype=self.row.dtype)])
        self.ptr = ptr
        self.backend = backend

        super().__init__(args, shape=shape, buffers=buffers)

    @property
    def dtype(self):
        """Data type of the stored non-zero values.

        Returns
        -------
        numpy.dtype
            Element data type, e.g. ``jnp.float32``.
        """
        return self.data.dtype

    @property
    def nse(self):
        """Number of stored elements (non-zeros).

        Returns
        -------
        int
            Total count of explicitly stored entries.
        """
        return self.data.size

    @property
    def info(self):
        """Structural metadata for the COO matrix.

        Returns
        -------
        COOInfo
            A named tuple containing ``shape``, ``rows_sorted``, and
            ``cols_sorted``.
        """
        return COOInfo(shape=self.shape, rows_sorted=self.rows_sorted, cols_sorted=self.cols_sorted)

    def _csr_params(self):
        """Return (indices, indptr, csr_shape, flip_transpose) for CSR dispatch."""
        if self.rows_sorted:
            return self.col, self.ptr, self.shape, False
        else:  # cols_sorted
            return self.row, self.ptr, (self.shape[1], self.shape[0]), True

    @classmethod
    def fromdense(
        cls,
        mat: Data,
        *,
        nse: int | None = None,
        index_dtype: jax.typing.DTypeLike = np.int32,
        backend: Optional[str] = None,
    ) -> 'COO':
        """
        Create a COO (Coordinate Format) sparse matrix from a dense matrix.

        This method converts a dense matrix to a sparse COO representation.

        Parameters
        ----------
        mat : jax.Array
            The dense matrix to be converted to COO format.
        nse : int | None, optional
            The number of non-zero elements in the matrix. If None, it will be
            calculated from the input matrix. Default is None.
        index_dtype : jax.typing.DTypeLike, optional
            The data type to be used for the row and column indices.
            Default is np.int32.

        Returns
        -------
        COO
            A new COO sparse matrix object representing the input dense matrix.
        """
        if nse is None:
            nse = (u.get_mantissa(mat) != 0.).sum()
        coo = u.sparse.coo_fromdense(mat, nse=nse, index_dtype=index_dtype)
        return COO(coo.data, coo.row, coo.col, shape=coo.shape, backend=backend)

    def sort_indices(self) -> 'COO':
        """Return a copy of the COO matrix with sorted indices.

        The matrix is sorted by row indices and column indices per row.
        If self.rows_sorted is True, this returns ``self`` without a copy.
        """
        if self.rows_sorted:
            return self
        data, unit = u.split_mantissa_unit(self.data)
        row, col, data = jax.lax.sort((self.row, self.col, data), num_keys=2)
        return COO(
            u.maybe_decimal(data * unit),
            row,
            col,
            shape=self.shape,
            rows_sorted=True,
            buffers=self.buffers,
            backend=self.backend,
        )

    def with_data(self, data: Data) -> 'COO':
        """
        Create a new COO matrix with the same structure but different data.

        This method returns a new COO matrix with the same row and column indices
        as the current matrix, but with new data values.

        Parameters
        ----------
        data : jax.Array | u.Quantity
            The new data to be used in the COO matrix. Must have the same shape,
            dtype, and unit as the current matrix's data.

        Returns
        -------
        COO
            A new COO matrix with the provided data and the same structure as
            the current matrix.

        Raises
        -------
        AssertionError
            If the shape, dtype, or unit of the new data doesn't match the
            current matrix's data.
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return COO(
            data, self.row, self.col, self.ptr,
            shape=self.shape,
            rows_sorted=self.rows_sorted,
            cols_sorted=self.cols_sorted,
            buffers=self.buffers,
            backend=self.backend,
        )

    def todense(self) -> Data:
        """
        Convert the COO matrix to a dense array.

        Returns
        -------
        jax.Array
            A dense representation of the COO matrix.
        """
        return _coo_todense(self.data, self.row, self.col, spinfo=self.info)

    def tocsr(self) -> 'CSR':
        """
        Convert the COO matrix to CSR (Compressed Sparse Row) format.

        Returns
        -------
        CSR
            A CSR matrix containing the same data as the original COO matrix.
        """
        dense = self.todense()
        return CSR.fromdense(dense, nse=self.nse)

    @property
    def T(self):
        """
        Get the transpose of the COO matrix.

        Returns
        -------
        COO
            The transposed COO matrix.
        """
        return self.transpose()

    def transpose(self, axes: Tuple[int, ...] | None = None) -> 'COO':
        """
        Transpose the COO matrix.

        Parameters
        ----------
        axes : Tuple[int, ...] | None, optional
            The axes to transpose over. Currently not implemented and will
            raise a NotImplementedError if provided.

        Returns
        -------
        COO
            The transposed COO matrix.

        Raises
        -------
        NotImplementedError
            If axes argument is provided.
        """
        if axes is not None:
            raise NotImplementedError("axes argument to transpose()")
        return COO(
            self.data, self.col, self.row, self.ptr,
            shape=self.shape[::-1],
            rows_sorted=self.cols_sorted,
            cols_sorted=self.rows_sorted,
            buffers=self.buffers,
            backend=self.backend,
        )

    def tree_flatten(self) -> Tuple[
        Tuple[jax.Array | u.Quantity,], dict[str, Any]
    ]:
        """
        Flatten the COO matrix for JAX transformations.

        This method is used by JAX to serialize the COO matrix object.

        Returns
        -------
        Tuple[Tuple[jax.Array | u.Quantity,], dict[str, Any]]
            A tuple containing:
            - A tuple with the matrix data.
            - A dictionary with auxiliary data (shape, sorting information, row and column indices).
        """
        aux = dict(
            shape=self.shape,
            rows_sorted=self.rows_sorted,
            cols_sorted=self.cols_sorted,
            ptr=self.ptr,
            row=self.row,
            col=self.col,
            backend=self.backend,
        )
        return (self.data,), (aux, self.buffers)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a COO matrix from flattened data.

        This class method is used by JAX to deserialize the COO matrix object.

        Parameters
        ----------
        aux_data : dict
            Auxiliary data containing shape, sorting information, and row and column indices.
        children : tuple
            A tuple containing the matrix data.

        Returns
        -------
        COO
            The reconstructed COO matrix.

        Raises
        -------
        ValueError
            If the auxiliary data doesn't contain the expected keys.
        """
        obj = object.__new__(cls)
        obj.data, = children
        aux_data, buffer = aux_data
        obj._buffer_registry = set(buffer.keys())
        for k, v in aux_data.items():
            setattr(obj, k, v)
        for k, v in buffer.items():
            setattr(obj, k, v)
        return obj

    def update_on_pre(
        self,
        pre_events: EventRepresentation,
        post_trace: Data,
        w_min: Optional[Data] = None,
        w_max: Optional[Data] = None,
        inplace: bool = False,
        backend: Optional[str] = None,
    ):
        """Update synaptic weights based on pre-synaptic events (pre-spike rule).

        Applies a spike-timing-dependent plasticity (STDP) update to the
        stored weights.  For each non-zero element ``(i, j)`` in the COO
        matrix whose pre-synaptic neuron ``j`` is active in *pre_events*,
        the weight is updated according to the post-synaptic trace value.

        Parameters
        ----------
        pre_events : EventRepresentation
            Pre-synaptic spike events.  Currently only ``BinaryArray`` is
            supported.
        post_trace : jax.Array or Quantity
            Post-synaptic eligibility trace, shape ``(num_post,)``.
        w_min : jax.Array or Quantity or None, optional
            Minimum weight bound.  ``None`` means no lower clipping.
        w_max : jax.Array or Quantity or None, optional
            Maximum weight bound.  ``None`` means no upper clipping.
        inplace : bool, optional
            If ``True``, mutate ``self.data`` in place and return ``None``.
            If ``False`` (default), return the updated weight array.
        backend : str or None, optional
            Compute backend (e.g. ``'numba'``, ``'warp'``).

        Returns
        -------
        jax.Array or Quantity or None
            Updated weight data, or ``None`` when *inplace* is ``True``.

        Raises
        ------
        NotImplementedError
            If *pre_events* is not a ``BinaryArray``, or if the matrix uses
            ``cols_sorted`` pointers (not yet supported).

        See Also
        --------
        update_on_post : Post-synaptic plasticity update.
        update_coo_on_binary_pre : Underlying COO pre-spike kernel.
        """
        if isinstance(pre_events, BinaryArray):
            _backend = backend or self.backend
            if self.ptr is not None:
                if self.rows_sorted:
                    data = update_csr_on_binary_pre(
                        self.data, self.col, self.ptr, pre_events.value, post_trace, w_min, w_max,
                        shape=self.shape, backend=_backend
                    )
                elif self.cols_sorted:
                    raise NotImplementedError("CSR dispatch for cols_sorted COO matrices is not yet implemented")
                else:
                    raise NotImplementedError("CSR dispatch requires rows_sorted or cols_sorted to be True")
            else:
                data = update_coo_on_binary_pre(
                    self.data, self.row, self.col, pre_events.value, post_trace, w_min, w_max, backend=_backend
                )

        else:
            raise NotImplementedError(
                f'update_on_pre is only implemented for BinaryArray pre_events, but got {type(pre_events)}'
            )

        if inplace:
            self.data = data
            return None
        return data

    def update_on_post(
        self,
        post_events: EventRepresentation,
        pre_trace: Data,
        w_min: Optional[Data] = None,
        w_max: Optional[Data] = None,
        inplace: bool = False,
        backend: Optional[str] = None,
    ):
        """Update synaptic weights based on post-synaptic events (post-spike rule).

        Applies a spike-timing-dependent plasticity (STDP) update to the
        stored weights.  For each non-zero element ``(i, j)`` whose
        post-synaptic neuron ``i`` is active in *post_events*, the weight
        is updated according to the pre-synaptic trace value.

        Parameters
        ----------
        post_events : EventRepresentation
            Post-synaptic spike events.  Currently only ``BinaryArray`` is
            supported.
        pre_trace : jax.Array or Quantity
            Pre-synaptic eligibility trace, shape ``(num_pre,)``.
        w_min : jax.Array or Quantity or None, optional
            Minimum weight bound.
        w_max : jax.Array or Quantity or None, optional
            Maximum weight bound.
        inplace : bool, optional
            If ``True``, mutate ``self.data`` in place and return ``None``.
        backend : str or None, optional
            Compute backend.

        Returns
        -------
        jax.Array or Quantity or None
            Updated weight data, or ``None`` when *inplace* is ``True``.

        Raises
        ------
        NotImplementedError
            If *post_events* is not a ``BinaryArray``, or if the matrix uses
            unsupported pointer layout.

        See Also
        --------
        update_on_pre : Pre-synaptic plasticity update.
        update_coo_on_binary_post : Underlying COO post-spike kernel.
        """
        if isinstance(post_events, BinaryArray):
            _backend = backend or self.backend
            if self.ptr is not None:
                if self.rows_sorted:
                    update_csr_on_binary_post
                # CSR dispatch for post-synaptic plasticity requires weight_indices
                # which is not available in COO format
                raise NotImplementedError(
                    "CSR dispatch for post-synaptic plasticity updates is not yet implemented for COO matrices. "
                    "Use a COO matrix without ptr (rows_sorted=False, cols_sorted=False) instead."
                )
            else:
                data = update_coo_on_binary_post(
                    self.data, self.row, self.col, post_events.value, pre_trace, w_min, w_max, backend=_backend
                )
        else:
            raise NotImplementedError(
                f'update_on_post is only implemented for BinaryArray post_events, but got {type(post_events)}'
            )

        if inplace:
            self.data = data
            return None
        return data

    def apply(self, fn):
        """
        Apply a function to the data and return a new sparse matrix with the same structure.

        Unlike :meth:`with_data`, which requires the new data to have the same
        shape, dtype, and unit, ``apply`` allows transformations that change
        dtype or unit.

        Parameters
        ----------
        fn : callable
            A function to apply to ``self.data``.

        Returns
        -------
        COO
            A new COO matrix with ``fn(self.data)`` and the same structure.
        """
        return COO(
            fn(self.data), self.row, self.col, self.ptr,
            shape=self.shape,
            rows_sorted=self.rows_sorted,
            cols_sorted=self.cols_sorted,
            buffers=self.buffers,
            backend=self.backend,
        )

    def __abs__(self):
        """Return element-wise absolute value, preserving COO structure.

        Returns
        -------
        COO
            A new COO matrix with ``abs(data)`` as the non-zero values.
        """
        return self.apply(operator.abs)

    def __neg__(self):
        """Return element-wise negation, preserving COO structure.

        Returns
        -------
        COO
            A new COO matrix with ``-data`` as the non-zero values.
        """
        return self.apply(operator.neg)

    def __pos__(self):
        """Return a copy of the COO matrix (unary positive).

        Returns
        -------
        COO
            A new COO matrix with the same data.
        """
        return self.apply(operator.pos)

    def _binary_op(self, other, op):
        if op in [operator.add, operator.sub]:
            other = u.math.asarray(other)
            dense = self.todense()
            return op(dense, other)

        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return COO(
                op(self.data, other), self.row, self.col, self.ptr,
                shape=self.shape,
                rows_sorted=self.rows_sorted,
                cols_sorted=self.cols_sorted,
                backend=self.backend,
                buffers=self.buffers,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            other = other[self.row, self.col]
            return COO(
                op(self.data, other), self.row, self.col, self.ptr,
                shape=self.shape,
                rows_sorted=self.rows_sorted,
                cols_sorted=self.cols_sorted,
                backend=self.backend,
                buffers=self.buffers,
            )
        else:
            raise NotImplementedError(f"{op.__name__} with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if op in [operator.add, operator.sub]:
            other = u.math.asarray(other)
            dense = self.todense()
            return op(other, dense)

        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return COO(
                op(other, self.data), self.row, self.col, self.ptr,
                shape=self.shape,
                rows_sorted=self.rows_sorted,
                cols_sorted=self.cols_sorted,
                backend=self.backend,
                buffers=self.buffers,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            other = other[self.row, self.col]
            return COO(
                op(other, self.data), self.row, self.col, self.ptr,
                shape=self.shape,
                rows_sorted=self.rows_sorted,
                cols_sorted=self.cols_sorted,
                backend=self.backend,
                buffers=self.buffers,
            )
        else:
            raise NotImplementedError(f"{op.__name__} with object of shape {other.shape}")

    def apply2(self, other, fn, *, reverse: bool = False):
        """
        Apply a binary function while preserving sparse structure semantics.

        Parameters
        ----------
        other : Any
            Right-hand operand for normal operations, or left-hand operand when
            ``reverse=True``.
        fn : callable
            Binary function from ``operator`` or a compatible callable.
        reverse : bool, optional
            If False, compute ``fn(self, other)`` semantics using ``_binary_op``.
            If True, compute ``fn(other, self)`` semantics using ``_binary_rop``.
            Defaults to False.

        Returns
        -------
        COO or Data
            Result of the operation.
        """
        if reverse:
            return self._binary_rop(other, fn)
        return self._binary_op(other, fn)

    def __mul__(self, other: Data) -> 'COO':
        """
        Perform element-wise multiplication of the COO matrix with another object.

        This method is called when the COO matrix is on the left side of the
        multiplication operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to be multiplied with the COO matrix.

        Returns
        -------
        COO
            A new COO matrix resulting from the element-wise multiplication.
        """
        return self.apply2(other, operator.mul)

    def __rmul__(self, other: Data) -> 'COO':
        """
        Perform right element-wise multiplication of the COO matrix with another object.

        This method is called when the COO matrix is on the right side of the
        multiplication operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to be multiplied with the COO matrix.

        Returns
        -------
        COO
            A new COO matrix resulting from the element-wise multiplication.
        """
        return self.apply2(other, operator.mul, reverse=True)

    def __truediv__(self, other: Data) -> 'COO':
        """Element-wise true division ``self / other``.

        Parameters
        ----------
        other : scalar or array_like
            Divisor.

        Returns
        -------
        COO
            Result with the same sparsity structure.
        """
        return self.apply2(other, operator.truediv)

    def __rtruediv__(self, other: Data) -> 'COO':
        """Element-wise true division ``other / self``.

        Parameters
        ----------
        other : scalar or array_like
            Dividend.

        Returns
        -------
        COO
            Result with the same sparsity structure.
        """
        return self.apply2(other, operator.truediv, reverse=True)

    def __add__(self, other: Data) -> 'COO':
        """Element-wise addition ``self + other``.

        Parameters
        ----------
        other : scalar or array_like
            Addend.

        Returns
        -------
        COO or jax.Array
            For scalar or shape-matched *other*, returns a COO.  For
            addition, the sparse matrix is first densified.
        """
        return self.apply2(other, operator.add)

    def __radd__(self, other: Data) -> 'COO':
        """Element-wise addition ``other + self``.

        Parameters
        ----------
        other : scalar or array_like
            Left addend.

        Returns
        -------
        COO or jax.Array
            Result of the addition.
        """
        return self.apply2(other, operator.add, reverse=True)

    def __sub__(self, other: Data) -> 'COO':
        """Element-wise subtraction ``self - other``.

        Parameters
        ----------
        other : scalar or array_like
            Subtrahend.

        Returns
        -------
        COO or jax.Array
            Result of the subtraction.
        """
        return self.apply2(other, operator.sub)

    def __rsub__(self, other: Data) -> 'COO':
        """Element-wise subtraction ``other - self``.

        Parameters
        ----------
        other : scalar or array_like
            Minuend.

        Returns
        -------
        COO or jax.Array
            Result of the subtraction.
        """
        return self.apply2(other, operator.sub, reverse=True)

    def __mod__(self, other: Data) -> 'COO':
        """Element-wise modulo ``self % other``.

        Parameters
        ----------
        other : scalar or array_like
            Divisor for the modulo operation.

        Returns
        -------
        COO
            Result with the same sparsity structure.
        """
        return self.apply2(other, operator.mod)

    def __rmod__(self, other: Data) -> 'COO':
        """Element-wise modulo ``other % self``.

        Parameters
        ----------
        other : scalar or array_like
            Dividend for the modulo operation.

        Returns
        -------
        COO
            Result with the same sparsity structure.
        """
        return self.apply2(other, operator.mod, reverse=True)

    def __matmul__(self, other: Data) -> Data:
        """
        Perform matrix multiplication (coo @ other).

        This method is called when the COO matrix is on the left side of the
        matrix multiplication operator.

        Parameters
        ----------
        other : jax.typing.ArrayLike
            The object to be multiplied with the COO matrix.

        Returns
        -------
        jax.Array | u.Quantity
            The result of the matrix multiplication.

        Raises
        -------
        NotImplementedError
            If the `other` object is a sparse matrix or has an unsupported shape.
        """
        # coo @ other
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        data = self.data

        # Dispatch to CSR when ptr is available
        if self.ptr is not None:
            indices, indptr, csr_shape, flip = self._csr_params()
            actual_transpose = False ^ flip

            if isinstance(other, BinaryArray):
                other = other.value
                if other.ndim == 1:
                    return binary_csrmv(
                        data, indices, indptr, other,
                        shape=csr_shape, transpose=actual_transpose, backend=self.backend
                    )
                elif other.ndim == 2:
                    return binary_csrmm(
                        data, indices, indptr, other,
                        shape=csr_shape, transpose=actual_transpose, backend=self.backend
                    )
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            elif isinstance(other, SparseFloat):
                other = other.value
                data, other = u.math.promote_dtypes(self.data, other)
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                other = u.math.asarray(other)
                data, other = u.math.promote_dtypes(self.data, other)
                if other.ndim == 1:
                    return csrmv(
                        data, indices, indptr, other, shape=csr_shape, transpose=actual_transpose,
                        backend=self.backend
                    )
                elif other.ndim == 2:
                    return csrmm(
                        data, indices, indptr, other, shape=csr_shape, transpose=actual_transpose,
                        backend=self.backend
                    )
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                return binary_coomv(data, self.row, self.col, other, shape=self.shape, backend=self.backend)
            elif other.ndim == 2:
                return binary_coomm(data, self.row, self.col, other, shape=self.shape, backend=self.backend)
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
        elif isinstance(other, SparseFloat):
            other = other.value
            data, other = u.math.promote_dtypes(self.data, other)
            raise NotImplementedError(f"matmul with object of shape {other.shape}")
        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return coomv(data, self.row, self.col, other, shape=self.shape, backend=self.backend)
            elif other.ndim == 2:
                return coomm(data, self.row, self.col, other, shape=self.shape, backend=self.backend)
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other: Data) -> Data:
        """
        Perform right matrix multiplication (other @ coo).

        This method is called when the COO matrix is on the right side of the
        matrix multiplication operator.

        Parameters
        ----------
        other : jax.typing.ArrayLike
            The object to be multiplied with the COO matrix.

        Returns
        -------
        jax.Array | u.Quantity
            The result of the matrix multiplication.

        Raises
        -------
        NotImplementedError
            If the `other` object is a sparse matrix or has an unsupported shape.
        """
        # other @ coo
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        # Dispatch to CSR when ptr is available
        if self.ptr is not None:
            indices, indptr, csr_shape, flip = self._csr_params()
            actual_transpose = True ^ flip

            if isinstance(other, BinaryArray):
                other = other.value
                if other.ndim == 1:
                    return binary_csrmv(
                        data, indices, indptr, other, shape=csr_shape, transpose=actual_transpose,
                        backend=self.backend
                    )
                elif other.ndim == 2:
                    other = other.T
                    r = binary_csrmm(
                        data, indices, indptr, other, shape=csr_shape, transpose=actual_transpose,
                        backend=self.backend
                    )
                    return r.T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            elif isinstance(other, SparseFloat):
                other = other.value
                data, other = u.math.promote_dtypes(self.data, other)
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                other = u.math.asarray(other)
                data, other = u.math.promote_dtypes(self.data, other)
                if other.ndim == 1:
                    return csrmv(
                        data, indices, indptr, other, shape=csr_shape, transpose=actual_transpose,
                        backend=self.backend
                    )
                elif other.ndim == 2:
                    other = other.T
                    r = csrmm(
                        data, indices, indptr, other, shape=csr_shape, transpose=actual_transpose,
                        backend=self.backend
                    )
                    return r.T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")

        if isinstance(other, BinaryArray):
            other = other.value
            if other.ndim == 1:
                return binary_coomv(
                    data, self.row, self.col, other, shape=self.shape, transpose=True,
                    backend=self.backend
                )
            elif other.ndim == 2:
                other = other.T
                r = binary_coomm(
                    data, self.row, self.col, other, shape=self.shape, transpose=True,
                    backend=self.backend
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        elif isinstance(other, SparseFloat):
            other = other.value
            data, other = u.math.promote_dtypes(self.data, other)
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return coomv(
                    data, self.row, self.col, other,
                    shape=self.shape, transpose=True, backend=self.backend
                )
            elif other.ndim == 2:
                other = other.T
                r = coomm(
                    data, self.row, self.col, other,
                    shape=self.shape, transpose=True, backend=self.backend
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
