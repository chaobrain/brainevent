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


from __future__ import annotations

import operator
from typing import Any, Tuple

import brainunit as u
import jax
import numpy as np

from brainevent._compatible_import import JAXSparse
from brainevent._event.binary import EventArray
from brainevent._misc import _coo_todense, COOInfo
from brainevent._typing import MatrixShape, Data, Index, Row, Col
from .binary import event_coo_matvec, event_coo_matmat
from .float import coo_matvec, coo_matmat

__all__ = [
    'COO',
]


@jax.tree_util.register_pytree_node_class
class COO(u.sparse.SparseMatrix):
    """
    Coordinate Format (COO) sparse matrix.

    This class represents a sparse matrix in coordinate format, where non-zero
    elements are stored as triplets (row, column, value).

    The class also supports various arithmetic operations (+, -, *, /, @, etc.)
    and comparisons with other COO matrices, dense arrays, and scalars.

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
    _info : COOInfo
        Additional information about the matrix structure (property).
    _bufs : tuple
        Tuple of (data, row, col) arrays (property).
    _rows_sorted : bool
        Whether row indices are sorted.
    _cols_sorted : bool
        Whether column indices are sorted within each row.

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
    nse = property(lambda self: self.data.size)
    dtype = property(lambda self: self.data.dtype)
    _info = property(
        lambda self: COOInfo(
            shape=self.shape,
            rows_sorted=self._rows_sorted,
            cols_sorted=self._cols_sorted)
    )
    _bufs = property(lambda self: (self.data, self.row, self.col))
    _rows_sorted: bool
    _cols_sorted: bool

    def __init__(
        self,
        args: Tuple[Data, Row, Col],
        *,
        shape: MatrixShape,
        rows_sorted: bool = False,
        cols_sorted: bool = False
    ):
        """
        Initialize a COO matrix.

        Parameters
        ----------
        args : Tuple[jax.Array | u.Quantity, jax.Array, jax.Array]
            Tuple containing (data, row indices, column indices).
        shape : Shape
            Shape of the matrix (rows, columns).
        rows_sorted : bool, optional
            Whether row indices are sorted. Default is False.
        cols_sorted : bool, optional
            Whether column indices are sorted within each row. Default is False.
        """
        self.data, self.row, self.col = map(u.math.asarray, args)
        self._rows_sorted = rows_sorted
        self._cols_sorted = cols_sorted
        super().__init__(args, shape=shape)

    @classmethod
    def fromdense(
        cls,
        mat: Data,
        *,
        nse: int | None = None,
        index_dtype: jax.typing.DTypeLike = np.int32
    ) -> COO:
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
        --------
        COO
            A new COO sparse matrix object representing the input dense matrix.
        """
        if nse is None:
            nse = (u.get_mantissa(mat) != 0.).sum()
        coo = u.sparse.coo_fromdense(mat, nse=nse, index_dtype=index_dtype)
        return COO((coo.data, coo.row, coo.col), shape=coo.shape)

    def _sort_indices(self) -> COO:
        """Return a copy of the COO matrix with sorted indices.

        The matrix is sorted by row indices and column indices per row.
        If self._rows_sorted is True, this returns ``self`` without a copy.
        """
        if self._rows_sorted:
            return self
        data, unit = u.split_mantissa_unit(self.data)
        row, col, data = jax.lax.sort((self.row, self.col, data), num_keys=2)
        return self.__class__(
            (
                u.maybe_decimal(u.Quantity(data, unit=unit)),
                row,
                col
            ),
            shape=self.shape,
            rows_sorted=True
        )

    def with_data(self, data: Data) -> COO:
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
        --------
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
        return COO((data, self.row, self.col), shape=self.shape)

    def todense(self) -> Data:
        """
        Convert the COO matrix to a dense array.

        Returns
        --------
        jax.Array
            A dense representation of the COO matrix.
        """
        return _coo_todense(self.data, self.row, self.col, spinfo=self._info)

    @property
    def T(self):
        """
        Get the transpose of the COO matrix.

        Returns
        --------
        COO
            The transposed COO matrix.
        """
        return self.transpose()

    def transpose(self, axes: Tuple[int, ...] | None = None) -> COO:
        """
        Transpose the COO matrix.

        Parameters
        ----------
        axes : Tuple[int, ...] | None, optional
            The axes to transpose over. Currently not implemented and will
            raise a NotImplementedError if provided.

        Returns
        --------
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
            (self.data, self.col, self.row),
            shape=self.shape[::-1],
            rows_sorted=self._cols_sorted,
            cols_sorted=self._rows_sorted
        )

    def tree_flatten(self) -> Tuple[
        Tuple[jax.Array | u.Quantity,], dict[str, Any]
    ]:
        """
        Flatten the COO matrix for JAX transformations.

        This method is used by JAX to serialize the COO matrix object.

        Returns
        --------
        Tuple[Tuple[jax.Array | u.Quantity,], dict[str, Any]]
            A tuple containing:
            - A tuple with the matrix data.
            - A dictionary with auxiliary data (shape, sorting information, row and column indices).
        """
        aux = self._info._asdict()
        aux['row'] = self.row
        aux['col'] = self.col
        return (self.data,), aux

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
        --------
        COO
            The reconstructed COO matrix.

        Raises
        -------
        ValueError
            If the auxiliary data doesn't contain the expected keys.
        """
        obj = object.__new__(cls)
        obj.data, = children
        if aux_data.keys() != {'shape', 'rows_sorted', 'cols_sorted', 'row', 'col'}:
            raise ValueError(f"COO.tree_unflatten: invalid {aux_data=}")
        obj.shape = aux_data['shape']
        obj._rows_sorted = aux_data['rows_sorted']
        obj._cols_sorted = aux_data['cols_sorted']
        obj.row = aux_data['row']
        obj.col = aux_data['col']
        return obj

    def _unitary_op(self, op):
        """
        Helper function for unary operations.

        Parameters
        ----------
        op : function
            The unary operation to apply to the data.

        Returns
        -------
        COO
            A new COO matrix with the operation applied to the data.
        """
        return COO(
            (
                op(self.data),
                self.row,
                self.col
            ),
            shape=self.shape,
            rows_sorted=self._rows_sorted,
            cols_sorted=self._cols_sorted
        )

    def __abs__(self):
        return self._unitary_op(operator.abs)

    def __neg__(self):
        return self._unitary_op(operator.neg)

    def __pos__(self):
        return self._unitary_op(operator.pos)

    def _binary_op(self, other, op):
        if isinstance(other, COO):
            if id(self.row) == id(other.row) and id(self.col) == id(other.col):
                return COO(
                    (
                        op(self.data, other.data),
                        self.row,
                        self.col
                    ),
                    shape=self.shape,
                    rows_sorted=self._rows_sorted,
                    cols_sorted=self._cols_sorted
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return COO(
                (
                    op(self.data, other),
                    self.row,
                    self.col
                ),
                shape=self.shape,
                rows_sorted=self._rows_sorted,
                cols_sorted=self._cols_sorted
            )
        elif other.ndim == 2 and other.shape == self.shape:
            other = other[self.row, self.col]
            return COO(
                (
                    op(self.data, other),
                    self.row,
                    self.col
                ),
                shape=self.shape,
                rows_sorted=self._rows_sorted,
                cols_sorted=self._cols_sorted
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, COO):
            if id(self.row) == id(other.row) and id(self.col) == id(other.col):
                return COO(
                    (
                        op(other.data, self.data),
                        self.row,
                        self.col
                    ),
                    shape=self.shape,
                    rows_sorted=self._rows_sorted,
                    cols_sorted=self._cols_sorted
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return COO(
                (
                    op(other, self.data),
                    self.row,
                    self.col
                ),
                shape=self.shape,
                rows_sorted=self._rows_sorted,
                cols_sorted=self._cols_sorted
            )
        elif other.ndim == 2 and other.shape == self.shape:
            other = other[self.row, self.col]
            return COO(
                (
                    op(other, self.data),
                    self.row,
                    self.col
                ),
                shape=self.shape,
                rows_sorted=self._rows_sorted,
                cols_sorted=self._cols_sorted
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: Data) -> COO:
        """
        Perform element-wise multiplication of the COO matrix with another object.

        This method is called when the COO matrix is on the left side of the
        multiplication operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to be multiplied with the COO matrix.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise multiplication.
        """
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Data) -> COO:
        """
        Perform right element-wise multiplication of the COO matrix with another object.

        This method is called when the COO matrix is on the right side of the
        multiplication operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to be multiplied with the COO matrix.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise multiplication.
        """
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: Data) -> COO:
        """
        Perform element-wise division of the COO matrix by another object.

        This method is called when the COO matrix is on the left side of the
        division operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to divide the COO matrix by.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise division.
        """
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: Data) -> COO:
        """
        Perform right element-wise division of the COO matrix by another object.

        This method is called when the COO matrix is on the right side of the
        division operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to divide the COO matrix by.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise division.
        """
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other: Data) -> COO:
        """
        Perform true division of the COO matrix by another object.

        This method is an alias for __div__.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to divide the COO matrix by.

        Returns
        --------
        COO
            A new COO matrix resulting from the true division.
        """
        return self.__div__(other)

    def __rtruediv__(self, other: Data) -> COO:
        """
        Perform right true division of the COO matrix by another object.

        This method is an alias for __rdiv__.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to divide the COO matrix by.

        Returns
        --------
        COO
            A new COO matrix resulting from the right true division.
        """
        return self.__rdiv__(other)

    def __add__(self, other: Data) -> COO:
        """
        Perform element-wise addition of the COO matrix with another object.

        This method is called when the COO matrix is on the left side of the
        addition operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to be added to the COO matrix.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise addition.
        """
        return self._binary_op(other, operator.add)

    def __radd__(self, other: Data) -> COO:
        """
        Perform right element-wise addition of the COO matrix with another object.

        This method is called when the COO matrix is on the right side of the
        addition operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to be added to the COO matrix.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise addition.
        """
        return self._binary_rop(other, operator.add)

    def __sub__(self, other: Data) -> COO:
        """
        Perform element-wise subtraction of another object from the COO matrix.

        This method is called when the COO matrix is on the left side of the
        subtraction operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to be subtracted from the COO matrix.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise subtraction.
        """
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: Data) -> COO:
        """
        Perform right element-wise subtraction of the COO matrix from another object.

        This method is called when the COO matrix is on the right side of the
        subtraction operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to subtract the COO matrix from.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise subtraction.
        """
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other: Data) -> COO:
        """
        Perform element-wise modulo operation of the COO matrix with another object.

        This method is called when the COO matrix is on the left side of the
        modulo operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to perform the modulo operation with the COO matrix.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise modulo operation.
        """
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other: Data) -> COO:
        """
        Perform right element-wise modulo operation of the COO matrix with another object.

        This method is called when the COO matrix is on the right side of the
        modulo operator.

        Parameters
        ----------
        other : jax.Array | u.Quantity
            The object to perform the modulo operation with the COO matrix.

        Returns
        --------
        COO
            A new COO matrix resulting from the element-wise modulo operation.
        """
        return self._binary_rop(other, operator.mod)

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
        --------
        jax.Array | u.Quantity
            The result of the matrix multiplication.

        Raises
        -------
        NotImplementedError
            If the `other` object is a sparse matrix or has an unsupported shape.
        """
        # coo @ other
        if isinstance(other, JAXSparse):
            # Raise an error if attempting matrix multiplication between two sparse objects
            raise NotImplementedError("matmul between two sparse objects.")

        # Get the data of the COO matrix
        data = self.data

        if isinstance(other, EventArray):
            # Extract the data from the BaseArray
            other = other.data
            if other.ndim == 1:
                # Perform matrix-vector multiplication with event data
                return event_coo_matvec(
                    data,
                    self.row,
                    self.col,
                    other,
                    shape=self.shape
                )
            elif other.ndim == 2:
                # Perform matrix-matrix multiplication with event data
                return event_coo_matmat(
                    data,
                    self.row,
                    self.col,
                    other,
                    shape=self.shape
                )
            else:
                # Raise an error if the shape of the other object is unsupported
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
        else:
            # Convert the other object to an appropriate array type
            other = u.math.asarray(other)
            # Promote the data types of the matrix and the other object
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                # Perform matrix-vector multiplication
                return coo_matvec(
                    data,
                    self.row,
                    self.col,
                    other,
                    shape=self.shape
                )
            elif other.ndim == 2:
                # Perform matrix-matrix multiplication
                return coo_matmat(
                    data,
                    self.row,
                    self.col,
                    other,
                    shape=self.shape
                )
            else:
                # Raise an error if the shape of the other object is unsupported
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
        --------
        jax.Array | u.Quantity
            The result of the matrix multiplication.

        Raises
        -------
        NotImplementedError
            If the `other` object is a sparse matrix or has an unsupported shape.
        """
        # other @ coo
        if isinstance(other, JAXSparse):
            # Raise an error if attempting matrix multiplication between two sparse objects
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, EventArray):
            # Extract the data from the BaseArray
            other = other.data
            if other.ndim == 1:
                # Perform matrix-vector multiplication with event data
                return event_coo_matvec(
                    data,
                    self.row,
                    self.col,
                    other,
                    shape=self.shape,
                    transpose=True
                )
            elif other.ndim == 2:
                # Transpose the other matrix for multiplication
                other = other.T
                # Perform matrix-matrix multiplication with event data
                r = event_coo_matmat(
                    data,
                    self.row,
                    self.col,
                    other,
                    shape=self.shape,
                    transpose=True
                )
                # Transpose the result back to the original orientation
                return r.T
            else:
                # Raise an error if the shape of the other object is unsupported
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
        else:
            # Convert the other object to an appropriate array type
            other = u.math.asarray(other)
            # Promote the data types of the matrix and the other object
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                # Perform matrix-vector multiplication
                return coo_matvec(
                    data,
                    self.row,
                    self.col,
                    other,
                    shape=self.shape,
                    transpose=True
                )
            elif other.ndim == 2:
                # Transpose the other matrix for multiplication
                other = other.T
                # Perform matrix-matrix multiplication
                r = coo_matmat(
                    data,
                    self.row,
                    self.col,
                    other,
                    shape=self.shape,
                    transpose=True
                )
                # Transpose the result back to the original orientation
                return r.T
            else:
                # Raise an error if the shape of the other object is unsupported
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
