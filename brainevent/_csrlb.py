# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


import importlib.util
from typing import Union, Sequence, Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from ._compatible_import import JAXSparse
from ._csr import CSR, CSC
from ._csrlb_event_impl import _event_csr_matvec, _event_csr_matmat
from ._csrlb_float_impl import _csr_matvec, _csr_matmat
from ._event import EventArray
from ._misc import _csr_to_coo

__all__ = [
    'CSR_LB',
    'CSC_LB',
]


def transform_to_id(n_conn, indptr):
    m = (n_conn - 1) // 32 + 1
    ids = np.zeros(m + 1, dtype=np.int32)
    pos = 0
    for k in range(0, indptr.shape[0] - 1):
        posl = pos
        pos = indptr[k + 1] // 32 + 1
        posr = pos
        for j in range(posl, posr):
            ids[j] = k
    ids[m] = indptr.shape[0] - 2
    return ids


if importlib.util.find_spec('numba'):
    import numba

    transform_to_id = numba.njit(transform_to_id)


@jax.tree_util.register_pytree_node_class
class CSR_LB(CSR):
    """
    Event-driven, Unit-aware, and Load-Balanced Compressed Sparse Row (CSR) matrix.

    This class represents a sparse matrix in CSR format, which is efficient for
    row-wise operations and matrix-vector multiplications. It is compatible with
    JAX's tree utilities and supports unit-aware computations.

    Attributes
    -----------
    data : Union[jax.Array, u.Quantity]
        Array of the non-zero values in the matrix.
    indices : jax.Array
        Array of column indices for the non-zero values.
    indptr : jax.Array
        Array of row pointers indicating where each row starts in the data and indices arrays.
    ids : jax.Array
        Array of integer IDs used for load balancing, derived from the `indptr` array.
    shape : tuple[int, int]
        The shape of the matrix as (rows, columns).
    nse : int
        Number of stored elements (non-zero entries).
    dtype : dtype
        Data type of the matrix values.
    """
    data: Union[jax.Array, u.Quantity]
    indices: jax.Array
    indptr: jax.Array
    ids: jax.Array
    shape: tuple[int, int]
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        args: Sequence[Union[jax.Array, np.ndarray, u.Quantity]],
        *,
        shape: Tuple[int, int]
    ):
        """
        Initialize a CSR (Compressed Sparse Row) matrix.

        This constructor creates a CSR matrix from the given arguments and shape.

        Parameters
        -----------
        args : Sequence[Union[jax.Array, np.ndarray, u.Quantity]]
            A sequence of three arrays representing the CSR matrix:
            - data: Contains the non-zero values of the matrix.
            - indices: Contains the column indices for each non-zero element.
            - indptr: Contains the row pointers indicating where each row starts in the data and indices arrays.
            - ids: (Optional) Array of integer IDs used for load balancing, derived from the `indptr` array.

        shape : Tuple[int, int]
            The shape of the matrix as a tuple of (num_rows, num_columns).
        """
        super().__init__(args[:3], shape=shape)
        if len(args) == 3:
            ids = transform_to_id(self.indices.size, np.asarray(self.indptr))
        else:
            ids = args[3]  # type: jax.Array | np.ndarray
        self.ids = jax.device_put(jnp.asarray(ids), device=next(iter(self.indptr.devices())))

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=jnp.int32) -> 'CSR_LB':
        """
        Create a CSR matrix from a dense matrix.
    
        This method converts a dense matrix to a Compressed Sparse Row (CSR) format.
    
        Parameters
        -----------
        mat : array_like
            The dense matrix to be converted to CSR format.
        nse : int, optional
            The number of non-zero elements in the matrix. If None, it will be
            calculated from the input matrix.
        index_dtype : dtype, optional
            The data type to be used for index arrays (default is jnp.int32).
    
        Returns
        --------
        CSR
            A new CSR matrix object created from the input dense matrix.
        """
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csr = u.sparse.csr_fromdense(mat, nse=nse, index_dtype=index_dtype)
        return CSR_LB((csr.data, csr.indices, csr.indptr), shape=csr.shape)

    @classmethod
    def fromcsr(cls, csr: CSR) -> 'CSR_LB':
        """
        Create a load-balanced CSR matrix from another CSR matrix.

        This method creates a new CSR matrix instance from an existing CSR matrix.

        Parameters
        -----------
        csr : CSR
            The source CSR matrix to be converted.

        Returns
        --------
        CSR_LB
            A new CSR matrix instance created from the input CSR matrix.
        """
        if isinstance(csr, CSR_LB):
            return cls((csr.data, csr.indices, csr.indptr, csr.ids), shape=csr.shape)
        assert isinstance(csr, CSR), "csr must be a CSR matrix"
        return cls((csr.data, csr.indices, csr.indptr), shape=csr.shape)

    def with_data(self, data: Union[jax.Array, u.Quantity]) -> 'CSR_LB':
        """
        Create a new CSR matrix with updated data while keeping the same structure.
    
        This method creates a new CSR matrix instance with the provided data,
        maintaining the original indices, indptr, and shape.
    
        Parameters
        -----------
        data : Union[jax.Array, u.Quantity]
            The new data array to replace the existing data in the CSR matrix.
            It must have the same shape, dtype, and unit as the original data.
    
        Returns
        --------
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
        return CSR_LB((data, self.indices, self.indptr, self.ids), shape=self.shape)

    def transpose(self, axes=None):
        """
        Transpose the CSR matrix.
    
        This method returns the transpose of the CSR matrix as a CSC matrix.
    
        Parameters
        -----------
        axes : None
            This parameter is not used and must be None. Included for compatibility
            with numpy's transpose function signature.
    
        Returns
        --------
        CSC_LB
            The transpose of the CSR matrix as a CSC (Compressed Sparse Column) matrix.
    
        Raises
        -------
        AssertionError
            If axes is not None, as this implementation doesn't support custom axis ordering.
        """
        assert axes is None, "transpose does not support axes argument."
        return CSC_LB((self.data, self.indices, self.indptr, self.ids), shape=self.shape[::-1])

    def _unitary_op(self, op):
        """
        Apply a unary operation to the data of the CSR matrix.

        This method is used internally to apply unary operations like abs, neg, etc.

        Parameters
        ----------
        op : function
            The unary operation to apply to the data array.

        Returns
        -------
        CSR_LB
            A new CSR matrix with the result of applying the unary operation to its data.
        """
        return CSR_LB((op(self.data), self.indices, self.indptr, self.ids), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR_LB((op(self.data, other.data), self.indices, self.indptr, self.ids), shape=self.shape)

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR_LB((op(self.data, other), self.indices, self.indptr, self.ids), shape=self.shape)

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR_LB((op(self.data, other), self.indices, self.indptr, self.ids), shape=self.shape)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR_LB((op(other.data, self.data), self.indices, self.indptr, self.ids), shape=self.shape)

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR_LB((op(other, self.data), self.indices, self.indptr, self.ids), shape=self.shape)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR_LB((op(other, self.data), self.indices, self.indptr, self.ids), shape=self.shape)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        # csr @ other
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                return _event_csr_matvec(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape
                )
            elif other.ndim == 2:
                return _event_csr_matmat(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return _csr_matvec(
                    data,
                    self.indices,
                    self.indptr,
                    self.ids,
                    other,
                    shape=self.shape
                )
            elif other.ndim == 2:
                return _csr_matmat(
                    data,
                    self.indices,
                    self.indptr,
                    self.ids,
                    other,
                    shape=self.shape
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        # other @ csr
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, EventArray):
            other = other.data
            if other.ndim == 1:
                return _event_csr_matvec(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=True
                )
            elif other.ndim == 2:
                other = other.T
                r = _event_csr_matmat(
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

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return _csr_matvec(
                    data,
                    self.indices,
                    self.indptr,
                    self.ids,
                    other,
                    shape=self.shape,
                    transpose=True
                )
            elif other.ndim == 2:
                other = other.T
                r = _csr_matmat(
                    data,
                    self.indices,
                    self.indptr,
                    self.ids,
                    other,
                    shape=self.shape,
                    transpose=True
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def tree_flatten(self):
        """
        Flatten the CSR matrix for JAX's tree utilities.
    
        This method is used by JAX's tree utilities to flatten the CSR matrix
        into a form suitable for transformation and reconstruction.
    
        Returns
        --------
        tuple
            A tuple containing two elements:
            - A tuple with the CSR matrix's data as the only element.
            - A tuple with the CSR matrix's indices, indptr, ids, and shape.
        """
        return (self.data,), (self.indices, self.indptr, self.ids, self.shape)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a CSR matrix from flattened data.
    
        This class method is used by JAX's tree utilities to reconstruct
        a CSR matrix from its flattened representation.
    
        Parameters
        -----------
        aux_data : tuple
            A tuple containing the CSR matrix's indices, indptr, ids, and shape.
        children : tuple
            A tuple containing the CSR matrix's data as its only element.
    
        Returns
        --------
        CSR
            A new CSR matrix instance reconstructed from the flattened data.
        """
        data, = children
        indices, indptr, ids, shape = aux_data
        return CSR_LB([data, indices, indptr, ids], shape=shape)


@jax.tree_util.register_pytree_node_class
class CSC_LB(CSC):
    """
    Event-driven, Unit-aware, and Load-Balanced Compressed Sparse Column (CSC) matrix.

    This class represents a sparse matrix in CSC format, which is efficient for
    column-wise operations and matrix-vector multiplications. It is compatible with
    JAX's tree utilities and supports unit-aware computations.

    Attributes
    -----------
    data : Union[jax.Array, u.Quantity]
        Array of the non-zero values in the matrix.
    indices : jax.Array
        Array of row indices for the non-zero values.
    indptr : jax.Array
        Array of column pointers indicating where each column starts in the data and indices arrays.
    ids : jax.Array
        Array of integer IDs used for load balancing, derived from the `indptr` array.
    shape : tuple[int, int]
        The shape of the matrix as (rows, columns).
    nse : int
        Number of stored elements (non-zero entries).
    dtype : dtype
        Data type of the matrix values.
    """
    data: Union[jax.Array, u.Quantity]
    indices: jax.Array
    indptr: jax.Array
    ids: jax.Array
    shape: tuple[int, int]
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        args: Sequence[Union[jax.Array, np.ndarray, u.Quantity]],
        *,
        shape: Tuple[int, int]
    ):
        """
        Initialize a CSC (Compressed Sparse Column) matrix.

        This constructor creates a CSC matrix from the given arguments and shape.

        Parameters
        ----------
        args : Sequence[Union[jax.Array, np.ndarray, u.Quantity]]
            A sequence of three arrays representing the CSC matrix:
            - data: Contains the non-zero values of the matrix.
            - indices: Contains the row indices for each non-zero element.
            - indptr: Contains the column pointers indicating where each column starts in the data and indices arrays.
            - ids: (Optional) Array of integer IDs used for load balancing, derived from the `indptr` array.

        shape : Tuple[int, int]
            The shape of the matrix as a tuple of (num_rows, num_columns).
        """
        super().__init__(args[:3], shape=shape)
        if len(args) == 3:
            ids = transform_to_id(self.indices.size, np.asarray(self.indptr))
        else:
            ids = args[3]  # type: jax.Array | np.ndarray
        self.ids = jnp.asarray(ids, device=self.indptr.device)

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=jnp.int32) -> 'CSC_LB':
        """
        Create a CSC (Compressed Sparse Column) matrix from a dense matrix.
    
        This method converts a dense matrix to CSC format, which is an efficient
        storage format for sparse matrices.
    
        Parameters
        -----------
        mat : array_like
            The dense matrix to be converted to CSC format.
        nse : int, optional
            The number of non-zero elements in the matrix. If None, it will be
            calculated from the input matrix.
        index_dtype : dtype, optional
            The data type to be used for index arrays (default is jnp.int32).
    
        Returns
        --------
        CSC_LB
            A new CSC matrix instance created from the input dense matrix.
        """
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csc = u.sparse.csr_fromdense(mat.T, nse=nse, index_dtype=index_dtype).T
        return CSC_LB((csc.data, csc.indices, csc.indptr), shape=csc.shape)

    def with_data(self, data: Union[jax.Array, u.Quantity]) -> 'CSC_LB':
        """
        Create a new CSC matrix with updated data while keeping the same structure.
    
        This method creates a new CSC matrix instance with the provided data,
        maintaining the original indices, indptr, and shape.
    
        Parameters
        -----------
        data : Union[jax.Array, u.Quantity]
            The new data array to replace the existing data in the CSC matrix.
            It must have the same shape, dtype, and unit as the original data.
    
        Returns
        --------
        CSC_LB
            A new CSC matrix instance with updated data and the same structure as the original.
    
        Raises
        -------
        AssertionError
            If the shape, dtype, or unit of the new data doesn't match the original data.
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return CSC_LB((data, self.indices, self.indptr, self.ids), shape=self.shape)

    def transpose(self, axes=None):
        """
        Transpose the CSC matrix.
    
        This method returns the transpose of the CSC matrix as a CSR matrix.
    
        Parameters
        -----------
        axes : None
            This parameter is not used and must be None. Included for compatibility
            with numpy's transpose function signature.
    
        Returns
        --------
        CSR
            The transpose of the CSC matrix as a CSR (Compressed Sparse Row) matrix.
    
        Raises
        -------
        AssertionError
            If axes is not None, as this implementation doesn't support custom axis ordering.
        """
        assert axes is None
        return CSR_LB((self.data, self.indices, self.indptr, self.ids), shape=self.shape[::-1])

    def _unitary_op(self, op):
        """
        Apply a unary operation to the data of the CSC matrix.

        This method is used internally to apply unary operations like abs, neg, etc.

        Parameters
        ----------
        op : function
            The unary operation to apply to the data array.

        Returns
        -------
        CSC_LB
            A new CSC matrix with the result of applying the unary operation to its data.
        """
        return CSC_LB((op(self.data), self.indices, self.indptr, self.ids), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, CSC_LB):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC_LB((op(self.data, other.data), self.indices, self.indptr, self.ids), shape=self.shape)

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC_LB((op(self.data, other), self.indices, self.indptr, self.ids), shape=self.shape)
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC_LB((op(self.data, other), self.indices, self.indptr, self.ids), shape=self.shape)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, CSC_LB):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC_LB((op(other.data, self.data), self.indices, self.indptr, self.ids), shape=self.shape)
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC_LB((op(other, self.data), self.indices, self.indptr, self.ids), shape=self.shape)
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC_LB((op(other, self.data), self.indices, self.indptr, self.ids), shape=self.shape)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, EventArray):
            other = other.value
            if other.ndim == 1:
                return _event_csr_matvec(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=True
                )
            elif other.ndim == 2:
                return _event_csr_matmat(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=True
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:

            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(data, other)
            if other.ndim == 1:
                return _csr_matvec(
                    data,
                    self.indices,
                    self.indptr,
                    self.ids,
                    other,
                    shape=self.shape[::-1],
                    transpose=True
                )
            elif other.ndim == 2:
                return _csr_matmat(
                    data,
                    self.indices,
                    self.indptr,
                    self.ids,
                    other,
                    shape=self.shape[::-1],
                    transpose=True
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, EventArray):
            other = other.value
            if other.ndim == 1:
                return _event_csr_matvec(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=False
                )
            elif other.ndim == 2:
                other = other.T
                r = _event_csr_matmat(
                    data,
                    self.indices,
                    self.indptr, other,
                    shape=self.shape[::-1],
                    transpose=False
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(data, other)
            if other.ndim == 1:
                return _csr_matvec(
                    data,
                    self.indices,
                    self.indptr,
                    self.ids,
                    other,
                    shape=self.shape[::-1],
                    transpose=False
                )
            elif other.ndim == 2:
                other = other.T
                r = _csr_matmat(
                    data,
                    self.indices,
                    self.indptr,
                    self.ids,
                    other,
                    shape=self.shape[::-1],
                    transpose=False
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def tree_flatten(self):
        """
        Flatten the CSC matrix for JAX's tree utilities.

        This method is used by JAX's tree utilities to flatten the CSC matrix
        into a form suitable for transformation and reconstruction.

        Returns
        -------
        tuple
            A tuple containing two elements:
            - A tuple with the CSC matrix's data as the only element.
            - A tuple with the CSC matrix's indices, indptr, ids, and shape.
        """
        return (self.data,), (self.indices, self.indptr, self.ids, self.shape)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a CSC matrix from flattened data.

        This class method is used by JAX's tree utilities to reconstruct
        a CSC matrix from its flattened representation.

        Parameters
        ----------
        aux_data : tuple
            A tuple containing the CSC matrix's indices, indptr, ids, and shape.
        children : tuple
            A tuple containing the CSC matrix's data as its only element.

        Returns
        -------
        CSC_LB
            A new CSC matrix instance reconstructed from the flattened data.
        """
        data, = children
        indices, indptr, ids, shape = aux_data
        return CSC_LB([data, indices, indptr, ids], shape=shape)
