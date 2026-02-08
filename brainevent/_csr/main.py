# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


import operator
from typing import Union, Sequence, Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._event import BinaryArray, SparseFloat
from brainevent._misc import _csr_to_coo, _csr_todense
from brainevent._typing import Data, Indptr, Index, MatrixShape
from .binary import binary_csrmv, binary_csrmm
from .diag_add import csr_diag_position_v2, csr_diag_add_v2
from .float import csrmv, csrmm, csrmv_yw2y
from .sparse_float import spfloat_csrmv, spfloat_csrmm
from .spsolve import csr_solve

__all__ = [
    'CSR',
    'CSC',
]


class BaseCLS(u.sparse.SparseMatrix):
    data: Data
    indices: Index
    indptr: Indptr
    shape: MatrixShape
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        data,
        indices=None,
        indptr=None,
        *,
        shape: MatrixShape
    ):
        """
        Initialize a compressed sparse matrix base instance.

        Supports two calling conventions::

            # Tuple syntax (original)
            CSR((data, indices, indptr), shape=(m, n))

            # Positional-argument syntax
            CSR(data, indices, indptr, shape=(m, n))

        Parameters
        ----------
        data : array or Sequence
            Either a single array (the ``data`` values) when ``indices`` and
            ``indptr`` are also provided, or a sequence of three arrays
            ``(data, indices, indptr)`` when used with the tuple syntax.
        indices : array, optional
            Secondary-axis indices for each stored element (column indices for
            CSR, row indices for CSC). Required when ``data`` is the
            data array.
        indptr : array, optional
            Primary-axis pointers indicating where each row/column starts in
            the data and indices arrays. Required when ``data`` is the
            data array.
        shape : Tuple[int, int]
            The shape of the matrix as ``(num_rows, num_columns)``.
        """
        if indices is None and indptr is None:
            # Tuple syntax: CSR((data, indices, indptr), shape=...)
            args = data
        else:
            # Positional syntax: CSR(data, indices, indptr, shape=...)
            args = (data, indices, indptr)

        assert len(args) == 3, "Expected three arguments: data, indices, indptr."
        self.data, self.indices, self.indptr = map(u.math.asarray, args)
        super().__init__(args, shape=shape)
        self.diag_positions = None

    def tree_flatten(self):
        aux = {
            'indices': self.indices,
            'indptr': self.indptr,
            'shape': self.shape,
            'diag_positions': self.diag_positions,
        }
        return (self.data,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.data, = children
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj

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
        CSR or CSC
            A new sparse matrix with ``fn(self.data)`` and the same structure.
        """
        raise NotImplementedError

    def __abs__(self):
        return self.apply(operator.abs)

    def __neg__(self):
        return self.apply(operator.neg)

    def __pos__(self):
        return self.apply(operator.pos)

    def _binary_op(self, other, op):
        raise NotImplementedError

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
        CSR or CSC or Data
            Result of the operation.
        """
        if reverse:
            return self._binary_rop(other, fn)
        return self._binary_op(other, fn)

    def __mul__(self, other: Data):
        return self.apply2(other, operator.mul)

    def __truediv__(self, other):
        return self.apply2(other, operator.truediv)

    def __add__(self, other):
        return self.apply2(other, operator.add)

    def __sub__(self, other):
        return self.apply2(other, operator.sub)

    def _binary_rop(self, other, op):
        raise NotImplementedError

    def __rmul__(self, other: Data):
        return self.apply2(other, operator.mul, reverse=True)

    def __rtruediv__(self, other):
        return self.apply2(other, operator.truediv, reverse=True)

    def __radd__(self, other):
        return self.apply2(other, operator.add, reverse=True)

    def __rsub__(self, other):
        return self.apply2(other, operator.sub, reverse=True)

    def yw_to_w(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity]
    ) -> Union[jax.Array, u.Quantity]:
        raise NotImplementedError

    def yw_to_w_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity]
    ) -> Union[jax.Array, u.Quantity]:
        raise NotImplementedError

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=jnp.int32):
        raise NotImplementedError

    def with_data(self, data: Data):
        raise NotImplementedError

    def todense(self) -> Union[jax.Array, u.Quantity]:
        raise NotImplementedError

    def tocoo(self):
        raise NotImplementedError

    def diag_add(self, other):
        """
        Add values to the matrix diagonal and return a new sparse matrix.

        This method adds the provided diagonal value to the diagonal elements of the
        sparse matrix represented in Compressed Sparse Row (CSR) format. If the diagonal
        positions have not been computed yet, it will first calculate them.

        Parameters
        ----------
        other : array-like
            The diagonal value to be added to the sparse matrix. It should be compatible
            with the data type of the matrix's non-zero elements.

        Raises
        ------
        AssertionError
            If `other` is an instance of `JAXSparse`, as this operation does not support
            `JAXSparse` objects.

        Notes
        -----
        - The diagonal positions are computed only once and cached in the `diag_positions`
          attribute of the matrix instance.
        - This method relies on `csr_diag_position_v2` to find diagonal positions and
          `csr_diag_add_v2` to perform the actual addition.
        """
        if self.diag_positions is None:
            self.diag_positions = csr_diag_position_v2(self.indptr, self.indices, self.shape)
        assert not isinstance(other, u.sparse.SparseMatrix), "diag_add does not support JAXSparse objects."
        return self.with_data(csr_diag_add_v2(self.data, self.diag_positions, other))

    def solve(self, b: Union[jax.Array, u.Quantity]) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system Ax = b where A is the sparse matrix.

        This method uses JAX's sparse solver to solve the equation Ax = b,
        where A is the current sparse matrix and b is the right-hand side vector.

        Parameters
        ----------
        b : array_like
            The right-hand side vector of the linear system.

        Returns
        -------
        x : jax.Array or u.Quantity
            The solution vector x that satisfies Ax = b.
        """
        raise NotImplementedError

    def _diag_pos(self, pos):
        self.diag_positions = pos
        return self


@jax.tree_util.register_pytree_node_class
class CSR(BaseCLS):
    """
    Event-driven and Unit-aware Compressed Sparse Row (CSR) matrix.

    This class represents a sparse matrix in CSR format, which is efficient for
    row-wise operations and matrix-vector multiplications. It is compatible with
    JAX's tree utilities and supports unit-aware computations.

    The class supports arithmetic with scalars and dense arrays, plus sparse-dense
    matrix multiplication via ``@``. Sparse-sparse operations are limited.

    Attributes
    ----------
    data : Data
        Array of the non-zero values in the matrix.
    indices : jax.Array
        Array of column indices for the non-zero values.
    indptr : jax.Array
        Array of row pointers indicating where each row starts in the data and indices arrays.
    shape : tuple[int, int]
        The shape of the matrix as (rows, columns).
    nse : int
        Number of stored elements (non-zero entries).
    dtype : dtype
        Data type of the matrix values.
    """
    __module__ = 'brainevent'

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=jnp.int32) -> 'CSR':
        """
        Create a CSR matrix from a dense matrix.

        This method converts a dense matrix to a Compressed Sparse Row (CSR) format.

        Parameters
        ----------
        mat : array_like
            The dense matrix to be converted to CSR format.
        nse : int, optional
            The number of non-zero elements in the matrix. If None, it will be
            calculated from the input matrix.
        index_dtype : dtype, optional
            The data type to be used for index arrays (default is jnp.int32).

        Returns
        -------
        CSR
            A new CSR matrix object created from the input dense matrix.
        """
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csr = u.sparse.csr_fromdense(mat, nse=nse, index_dtype=index_dtype)
        return CSR((csr.data, csr.indices, csr.indptr), shape=csr.shape)

    def with_data(self, data: Data) -> 'CSR':
        """
        Create a new CSR matrix with updated data while keeping the same structure.

        This method creates a new CSR matrix instance with the provided data,
        maintaining the original indices, indptr, and shape.

        Parameters
        ----------
        data : Data
            The new data array to replace the existing data in the CSR matrix.
            It must have the same shape, dtype, and unit as the original data.

        Returns
        -------
        CSR
            A new CSR matrix instance with updated data and the same structure as the original.

        Raises
        -------
        AssertionError
            If the shape, dtype, or unit of the new data doesn't match the original data.
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return CSR((data, self.indices, self.indptr), shape=self.shape)._diag_pos(self.diag_positions)

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the CSR matrix to a dense matrix.

        This method transforms the compressed sparse row (CSR) representation
        into a full dense matrix.

        Returns
        -------
        array_like
            A dense matrix representation of the CSR matrix.
        """
        return _csr_todense(self.data, self.indices, self.indptr, shape=self.shape)

    def tocoo(self):
        """
        Convert the CSR matrix to COO (Coordinate) format.

        This method transforms the Compressed Sparse Row (CSR) matrix into a COO matrix,
        which stores sparse data as a collection of (row, column, value) triplets.

        Returns
        -------
        COO
            A COO matrix containing the same data as the original CSR matrix.

        See Also
        --------
        _csr_to_coo : Internal function that converts CSR row/column indices to COO format
        COO : The Coordinate sparse matrix class

        Examples
        --------
        >>> csr_matrix = CSR((data, indices, indptr), shape=(3, 4))
        >>> coo_matrix = csr_matrix.tocoo()
        """
        from brainevent import COO
        pre_ids, post_ids = _csr_to_coo(self.indices, self.indptr)
        return COO((self.data, pre_ids, post_ids), shape=self.shape)

    def transpose(self, axes=None) -> 'CSC':
        """
        Transpose the CSR matrix.

        This method returns the transpose of the CSR matrix as a CSC matrix.

        Parameters
        ----------
        axes : None
            This parameter is not used and must be None. Included for compatibility
            with numpy's transpose function signature.

        Returns
        -------
        CSC
            The transpose of the CSR matrix as a CSC (Compressed Sparse Column) matrix.

        Raises
        -------
        AssertionError
            If axes is not None, as this implementation doesn't support custom axis ordering.
        """
        assert axes is None, "transpose does not support axes argument."
        return CSC((self.data, self.indices, self.indptr), shape=self.shape[::-1])._diag_pos(self.diag_positions)

    def apply(self, fn) -> 'CSR':
        return CSR(fn(self.data), self.indices, self.indptr, shape=self.shape)._diag_pos(self.diag_positions)

    def _binary_op(self, other, op) -> 'CSR':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(dense, other)

        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    (op(self.data, other.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )._diag_pos(self.diag_positions)
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )._diag_pos(self.diag_positions)

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )._diag_pos(self.diag_positions)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'CSR':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(other, dense)

        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    (op(other.data, self.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )._diag_pos(self.diag_positions)
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )._diag_pos(self.diag_positions)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )._diag_pos(self.diag_positions)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        # csr @ other
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, BinaryArray):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return binary_csrmv(self.data, self.indices, self.indptr, other, shape=self.shape)
                elif other.ndim == 2:
                    return binary_csrmm(self.data, self.indices, self.indptr, other, shape=self.shape)
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        elif isinstance(other, SparseFloat):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return spfloat_csrmv(self.data, self.indices, self.indptr, other, shape=self.shape)
                elif other.ndim == 2:
                    return spfloat_csrmm(self.data, self.indices, self.indptr, other, shape=self.shape)
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=False
                )
            elif other.ndim == 2:
                return csrmm(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=False
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        # other @ csr
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, BinaryArray):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return binary_csrmv(self.data, self.indices, self.indptr, other, shape=self.shape, transpose=True)
                elif other.ndim == 2:
                    other = other.T
                    r = binary_csrmm(self.data, self.indices, self.indptr, other, shape=self.shape, transpose=True)
                    return r.T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        elif isinstance(other, SparseFloat):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return spfloat_csrmv(
                        self.data, self.indices, self.indptr, other,
                        shape=self.shape,
                        transpose=True
                    )
                elif other.ndim == 2:
                    other = other.T
                    r = spfloat_csrmm(
                        self.data, self.indices, self.indptr, other,
                        shape=self.shape,
                        transpose=True
                    )
                    return r.T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=True
                )
            elif other.ndim == 2:
                other = other.T
                r = csrmm(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=True
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def solve(self, b: Union[jax.Array, u.Quantity], tol=1e-6, reorder=1) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system Ax = b where A is the sparse matrix.

        This method uses JAX's sparse solver to solve the equation Ax = b,
        where A is the current sparse matrix and b is the right-hand side vector.

        Parameters
        ----------
        b : array_like
            The right-hand side vector of the linear system.
        tol : Tolerance to decide if singular or not. Defaults to 1e-6.
        reorder : The reordering scheme to use to reduce fill-in. No reordering if
            ``reorder=0``. Otherwise, symrcm, symamd, or csrmetisnd (``reorder=1,2,3``),
            respectively. Defaults to symrcm.

        Returns
        -------
        x : jax.Array or u.Quantity
            The solution vector x that satisfies Ax = b.
        """
        assert self.shape[0] == b.shape[0], ("The number of rows in the matrix must match "
                                             "the size of the right-hand side vector b.")
        return csr_solve(self.data, self.indices, self.indptr, b)

    def yw_to_w(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Perform a specialized transformation from y-w space to w space using this sparse matrix.

        This method implements a matrix-vector product operation that is optimized for
        specific computational patterns in neural simulations. It efficiently computes
        a sparse matrix vector product where indices in the y dimension map to values
        in the w dimension.

        Parameters
        ----------
        y_dim_arr : jax.Array, np.ndarray, u.Quantity
            Array containing values in the y dimension (typically target/post-synaptic).
        w_dim_arr : jax.Array, np.ndarray, u.Quantity
            Array containing values in the w dimension (typically weights or connection values).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The resulting array after the sparse transformation operation.
            Maintains the same units as the input arrays if they have units.

        Notes
        -----
        This method is typically used in event-driven neural simulations to efficiently
        compute the effect of connections between neurons.
        """
        return csrmv_yw2y(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape, transpose=False)

    def yw_to_w_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Perform a transposed transformation from y-w space to w space using this sparse matrix.

        This method implements the transpose of the yw_to_w operation, computing a specialized
        matrix-vector product that is optimized for specific computational patterns in neural
        simulations. It efficiently handles the transposed mapping between the y dimension and
        the w dimension.

        Parameters
        ----------
        y_dim_arr : jax.Array, np.ndarray, u.Quantity
            Array containing values in the y dimension (typically target/post-synaptic).
        w_dim_arr : jax.Array, np.ndarray, u.Quantity
            Array containing values in the w dimension (typically weights or connection values).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The resulting array after the transposed sparse transformation operation.
            Maintains the same units as the input arrays if they have units.

        Notes
        -----
        This method computes the transpose of the yw_to_w operation, which can be useful
        for backpropagation or adjoint operations in neural simulations.
        """
        return csrmv_yw2y(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape, transpose=True)


@jax.tree_util.register_pytree_node_class
class CSC(BaseCLS):
    """
    Event-driven and Unit-aware Compressed Sparse Column (CSC) matrix.

    This class represents a sparse matrix in CSC format, which is efficient for
    column-wise operations. It is compatible with JAX's tree utilities and
    supports unit-aware computations.

    The class supports arithmetic with scalars and dense arrays, plus sparse-dense
    matrix multiplication via ``@``. Sparse-sparse operations are limited.

    Attributes
    ----------
    data : Data
        Array of the non-zero values in the matrix.
    indices : jax.Array
        Array of row indices for the non-zero values.
    indptr : jax.Array
        Array of column pointers indicating where each column starts in the data and indices arrays.
    shape : tuple[int, int]
        The shape of the matrix as (rows, columns).
    nse : int
        Number of stored elements (non-zero entries).
    dtype : dtype
        Data type of the matrix values.

    """
    __module__ = 'brainevent'

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=jnp.int32) -> 'CSC':
        """
        Create a CSC (Compressed Sparse Column) matrix from a dense matrix.

        This method converts a dense matrix to CSC format, which is an efficient
        storage format for sparse matrices.

        Parameters
        ----------
        mat : array_like
            The dense matrix to be converted to CSC format.
        nse : int, optional
            The number of non-zero elements in the matrix. If None, it will be
            calculated from the input matrix.
        index_dtype : dtype, optional
            The data type to be used for index arrays (default is jnp.int32).

        Returns
        -------
        CSC
            A new CSC matrix instance created from the input dense matrix.
        """
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csc = u.sparse.csr_fromdense(mat.T, nse=nse, index_dtype=index_dtype).T
        return CSC((csc.data, csc.indices, csc.indptr), shape=csc.shape)

    def with_data(self, data: Data) -> 'CSC':
        """
        Create a new CSC matrix with updated data while keeping the same structure.

        This method creates a new CSC matrix instance with the provided data,
        maintaining the original indices, indptr, and shape.

        Parameters
        ----------
        data : Data
            The new data array to replace the existing data in the CSC matrix.
            It must have the same shape, dtype, and unit as the original data.

        Returns
        -------
        CSC
            A new CSC matrix instance with updated data and the same structure as the original.

        Raises
        -------
        AssertionError
            If the shape, dtype, or unit of the new data doesn't match the original data.
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return CSC((data, self.indices, self.indptr), shape=self.shape)._diag_pos(self.diag_positions)

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the CSC matrix to a dense matrix.

        This method transforms the compressed sparse column (CSC) representation
        into a full dense matrix.

        Returns
        -------
        array_like
            A dense matrix representation of the CSC matrix.
        """
        return self.T.todense().T

    def tocoo(self):
        """
        Convert the CSC matrix to COO (Coordinate) format.

        This method transforms the Compressed Sparse Column (CSC) matrix into a COO matrix,
        which stores sparse data as a collection of (row, column, value) triplets.

        Returns
        -------
        COO
            A COO matrix containing the same data as the original CSC matrix.

        See Also
        --------
        _csr_to_coo : Internal function that converts CSC column/row indices to COO format
        COO : The Coordinate sparse matrix class

        Examples
        --------
        >>> csc_matrix = CSC((data, indices, indptr), shape=(3, 4))
        >>> coo_matrix = csc_matrix.tocoo()
        """
        from brainevent import COO
        post_ids, pre_ids = _csr_to_coo(self.indices, self.indptr)
        return COO((self.data, pre_ids, post_ids), shape=self.shape)

    def transpose(self, axes=None) -> 'CSR':
        """
        Transpose the CSC matrix.

        This method returns the transpose of the CSC matrix as a CSR matrix.

        Parameters
        ----------
        axes : None
            This parameter is not used and must be None. Included for compatibility
            with numpy's transpose function signature.

        Returns
        -------
        CSR
            The transpose of the CSC matrix as a CSR (Compressed Sparse Row) matrix.

        Raises
        -------
        AssertionError
            If axes is not None, as this implementation doesn't support custom axis ordering.
        """
        assert axes is None
        return CSR((self.data, self.indices, self.indptr), shape=self.shape[::-1])._diag_pos(self.diag_positions)

    def apply(self, fn) -> 'CSC':
        return CSC((fn(self.data), self.indices, self.indptr), shape=self.shape)._diag_pos(self.diag_positions)

    def _binary_op(self, other, op) -> 'CSC':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(dense, other)
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(self.data, other.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )._diag_pos(self.diag_positions)
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )._diag_pos(self.diag_positions)
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )._diag_pos(self.diag_positions)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'CSC':
        if op in [operator.add, operator.sub]:
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            other = u.math.asarray(other)
            return op(other, dense)
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(other.data, self.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )._diag_pos(self.diag_positions)
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )._diag_pos(self.diag_positions)
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )._diag_pos(self.diag_positions)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, BinaryArray):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return binary_csrmv(
                        data, self.indices, self.indptr, other,
                        shape=self.shape[::-1],
                        transpose=True
                    )
                elif other.ndim == 2:
                    return binary_csrmm(
                        data, self.indices, self.indptr, other,
                        shape=self.shape[::-1],
                        transpose=True
                    )
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        elif isinstance(other, SparseFloat):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return spfloat_csrmv(
                        data, self.indices, self.indptr, other,
                        shape=self.shape[::-1],
                        transpose=True
                    )
                elif other.ndim == 2:
                    return spfloat_csrmm(
                        data, self.indices, self.indptr, other,
                        shape=self.shape[::-1],
                        transpose=True
                    )
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        else:

            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=True
                )
            elif other.ndim == 2:
                return csrmm(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=True
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, BinaryArray):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return binary_csrmv(data, self.indices, self.indptr, other,
                                        shape=self.shape[::-1],
                                        transpose=False)
                elif other.ndim == 2:
                    return binary_csrmm(data, self.indices, self.indptr, other.T,
                                        shape=self.shape[::-1],
                                        transpose=False).T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        elif isinstance(other, SparseFloat):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return spfloat_csrmv(data, self.indices, self.indptr, other,
                                         shape=self.shape[::-1],
                                         transpose=False)
                elif other.ndim == 2:
                    return spfloat_csrmm(data, self.indices, self.indptr, other.T,
                                         shape=self.shape[::-1],
                                         transpose=False).T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=False
                )
            elif other.ndim == 2:
                other = other.T
                r = csrmm(
                    data,
                    self.indices,
                    self.indptr, other,
                    shape=self.shape[::-1],
                    transpose=False
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def solve(self, b: Union[jax.Array, u.Quantity], tol=1e-6, reorder=1) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system Ax = b where A is the sparse matrix.

        This method uses JAX's sparse solver to solve the equation Ax = b,
        where A is the current sparse matrix and b is the right-hand side vector.

        Parameters
        ----------
        b : array_like
            The right-hand side vector of the linear system.
        tol : Tolerance to decide if singular or not. Defaults to 1e-6.
        reorder : The reordering scheme to use to reduce fill-in. No reordering if
            ``reorder=0``. Otherwise, symrcm, symamd, or csrmetisnd (``reorder=1,2,3``),
            respectively. Defaults to symrcm.

        Returns
        -------
        x : jax.Array or u.Quantity
            The solution vector x that satisfies Ax = b.
        """
        assert self.shape[0] == b.shape[0], ("The number of rows in the matrix must match "
                                             "the size of the right-hand side vector b.")
        return self.T.solve(b, tol=tol, reorder=reorder)

    def yw_to_w(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Perform a specialized transformation from y-w space to w space using this sparse matrix.

        This method implements a matrix-vector product operation that is optimized for
        specific computational patterns in neural simulations. It efficiently computes
        a sparse matrix vector product where indices in the y dimension map to values
        in the w dimension, using the Compressed Sparse Column format.

        Parameters
        ----------
        y_dim_arr : jax.Array, np.ndarray, u.Quantity
            Array containing values in the y dimension (typically target/post-synaptic).
        w_dim_arr : jax.Array, np.ndarray, u.Quantity
            Array containing values in the w dimension (typically weights or connection values).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The resulting array after the sparse transformation operation.
            Maintains the same units as the input arrays if they have units.

        Notes
        -----
        This method is typically used in event-driven neural simulations to efficiently
        compute the effect of connections between neurons. Unlike the CSR implementation,
        this method uses a transposed operation with reversed shape to account for the
        column-oriented storage format.
        """
        return csrmv_yw2y(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape[::-1], transpose=True)

    def yw_to_w_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Perform a transposed transformation from y-w space to w space using this sparse matrix.

        This method implements the transpose of the yw_to_w operation, computing a specialized
        matrix-vector product that is optimized for specific computational patterns in neural
        simulations. It efficiently handles the transposed mapping between the y dimension and
        the w dimension using the Compressed Sparse Column format.

        Parameters
        ----------
        y_dim_arr : jax.Array, np.ndarray, u.Quantity
            Array containing values in the y dimension (typically target/post-synaptic).
        w_dim_arr : jax.Array, np.ndarray, u.Quantity
            Array containing values in the w dimension (typically weights or connection values).

        Returns
        -------
        Union[jax.Array, u.Quantity]
            The resulting array after the transposed sparse transformation operation.
            Maintains the same units as the input arrays if they have units.

        Notes
        -----
        This method computes the transpose of the yw_to_w operation, which can be useful
        for backpropagation or adjoint operations in neural simulations. Unlike the regular
        yw_to_w method, this transposed version uses transpose=False in the underlying
        implementation to compute the appropriate transposed operation.
        """
        return csrmv_yw2y(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape[::-1], transpose=False)
