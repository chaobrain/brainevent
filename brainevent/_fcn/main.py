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
from typing import Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._coo import COO
from brainevent._event.binary import BinaryArray
from brainevent._event.sparse_float import SparseFloat
from brainevent._misc import _coo_todense, COOInfo
from brainevent._typing import Data, MatrixShape, Index
from .binary import binary_fcnmv, binary_fcnmm
from .float import fcnmv, fcnmm
from .sparse_float import spfloat_fcnmv, spfloat_fcnmm

__all__ = [
    'FixedPostNumConn',
    'FixedPreNumConn',
]


def _validate_fixed_conn_indices(
    indices: Index,
    *,
    expected_rows: int,
    kind: str,
):
    if indices.ndim != 2:
        raise ValueError(f'{kind} indices must be 2D, got {indices.ndim}D.')
    if indices.shape[0] != expected_rows:
        raise ValueError(
            f'{kind} row number mismatch. '
            f'{indices.shape[0]} != {expected_rows}'
        )
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError(f'{kind} indices must be integer type, got {indices.dtype}.')


def _contains_invalid_indices(indices: Index, *, upper_bound: int) -> bool:
    with jax.ensure_compile_time_eval():
        indices_np = np.asarray(indices)
        if bool(np.any(indices_np < 0) or np.any(indices_np >= upper_bound)):
            raise ValueError(
                f'Found invalid indices in the connection matrix. '
                f'All indices must be in the range [0, {upper_bound - 1}]. '
                f'But found indices with min {indices_np.min()} and max {indices_np.max()}.'
            )


class FixedNumConn(u.sparse.SparseMatrix):
    """
    Base class for fixed number of connections.
    """
    data: Data
    indices: Index
    shape: MatrixShape

    def tree_flatten(self):
        aux = {
            'indices': self.indices,
            'shape': self.shape,
        }
        return (self.data,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.data, = children
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj

    def _unitary_op(self, op):
        raise NotImplementedError

    def apply(self, fn):
        """
        Apply a function to the value buffer and keep connectivity structure.

        Parameters
        ----------
        fn : callable
            A function applied to ``self.data``.

        Returns
        -------
        FixedNumConn
            A new matrix-like object with transformed values.
        """
        return self._unitary_op(fn)

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
        Apply a binary function while preserving fixed-connectivity semantics.

        Parameters
        ----------
        other : Any
            Right-hand operand for normal operations, or left-hand operand when
            ``reverse=True``.
        fn : callable
            Binary function from ``operator`` or a compatible callable.
        reverse : bool, optional
            If False, compute ``fn(self, other)`` via ``_binary_op``.
            If True, compute ``fn(other, self)`` via ``_binary_rop``.
            Defaults to False.

        Returns
        -------
        FixedNumConn or Data
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

    def __mod__(self, other):
        return self.apply2(other, operator.mod)

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

    def __rmod__(self, other):
        return self.apply2(other, operator.mod, reverse=True)


@jax.tree_util.register_pytree_node_class
class FixedPostNumConn(FixedNumConn):
    """
    Represents a sparse matrix with a fixed number of post-synaptic connections
    per pre-synaptic neuron.

    This format is efficient when each row (pre-synaptic neuron) in the
    logical matrix has the same number of non-zero entries (outgoing connections).
    It stores the matrix data and the corresponding post-synaptic indices in
    dense arrays.

    Attributes
    ----------
    data : jax.numpy.ndarray
        A 2D array containing the non-zero values (e.g., synaptic weights)
        of the sparse matrix. The shape is `(num_pre, num_conn)`, where
        `num_conn` is the fixed number of outgoing connections per
        pre-synaptic neuron. `data[i, k]` is the value of the connection
        from pre-synaptic neuron `i` to its k-th connected post-synaptic neuron.
    indices : jax.numpy.ndarray
        A 2D array containing the post-synaptic indices (column indices) for each
        connection stored in `data`. The shape is `(num_pre, num_conn)`.
        `indices[i, k]` is the index of the post-synaptic neuron corresponding
        to the value `data[i, k]`.
    shape : tuple[int, int]
        A tuple `(num_pre, num_post)` representing the logical shape of the
        dense equivalent matrix. `num_pre` is the total number of pre-synaptic
        neurons (rows), and `num_post` is the total number of post-synaptic
        neurons (columns).
    num_pre : int
        The number of pre-synaptic neurons (rows in the dense matrix).
        Equal to `indices.shape[0]` or `shape[0]`.
    num_conn : int
        The fixed number of post-synaptic connections per pre-synaptic neuron.
        Equal to `indices.shape[1]`.
    num_post : int
        The number of post-synaptic neurons (columns in the dense matrix).
        Equal to `shape[1]`.
    nse : int
        The total number of specified elements (non-zeros). Equal to
        `num_pre * num_conn`.
    dtype : jax.numpy.dtype
        The data type of the `data` array.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import FixedPostNumConn
        >>>
        >>> # Example: 2 pre-synaptic neurons, each connecting to 2 post-synaptic neurons.
        >>> # Total post-synaptic neurons = 3. Shape = (2, 3)
        >>> data = jnp.array([[1., 2.], [3., 4.]]) # Shape (num_pre=2, num_conn=2)
        >>> # Post-synaptic indices for each pre-synaptic neuron:
        >>> # Pre 0 connects to Post 0 and Post 1
        >>> # Pre 1 connects to Post 1 and Post 2
        >>> indices = jnp.array([[0, 1], [1, 2]]) # Shape (num_pre=2, num_conn=2)
        >>> shape = (2, 3) # (num_pre, num_post)
        >>>
        >>> mat = FixedPostNumConn((data, indices), shape=shape)
        >>>
        >>> print("Data:", mat.data)
        Data: [[1. 2.]
               [3. 4.]]
        >>> print("Indices:", mat.indices)
        Indices: [[0 1]
                  [1 2]]
        >>> print("Shape:", mat.shape)
        Shape: (2, 3)
        >>> print("Number of connections per pre-neuron:", mat.num_conn)
        Number of connections per pre-neuron: 2
        >>>
        >>> # Convert to dense matrix
        >>> dense_mat = mat.todense()
        >>> print("Dense matrix:\\n", dense_mat)
        Dense matrix:
         [[1. 2. 0.]
          [0. 3. 4.]]
        >>>
        >>> # Transpose to FixedPreNumConn
        >>> mat_t = mat.transpose()
        >>> print("Transposed shape:", mat_t.shape)
        Transposed shape: (3, 2)
        >>> print("Transposed data (same):", mat_t.data)
        Transposed data (same): [[1. 2.]
         [3. 4.]]
        >>> print("Transposed indices (reinterpreted):", mat_t.indices)
        Transposed indices (reinterpreted): [[0 1]
         [1 2]]
    """
    __module__ = 'brainevent'

    data: Data
    indices: Index
    shape: MatrixShape
    num_pre = property(lambda self: self.indices.shape[0])
    num_conn = property(lambda self: self.indices.shape[1])
    num_post = property(lambda self: self.shape[1])
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(self, data, indices=None, *, shape: MatrixShape):
        if indices is None:
            args = data
        else:
            args = (data, indices)
        self.data, self.indices = map(u.math.asarray, args)
        _validate_fixed_conn_indices(self.indices, expected_rows=shape[0], kind='Post-synaptic')
        if self.data.size != 1 and self.data.shape != self.indices.shape:
            raise ValueError(
                f"Data shape {self.data.shape} must match indices shape {self.indices.shape}. "
                f"But got {self.data.shape} != {self.indices.shape}"
            )
        super().__init__(args, shape=shape)

        _contains_invalid_indices(self.indices, upper_bound=self.shape[1])

    def with_data(self, data: Data) -> 'FixedPostNumConn':
        """
        Creates a new FixedPostNumConn instance with the same indices and shape but different data.

        Parameters
        ----------
        data : Data
            New data array with the same shape, dtype, and unit as ``self.data``.

        Returns
        -------
        FixedPostNumConn
            New matrix with the provided data and unchanged connectivity.

        Raises
        ------
        AssertionError
            If ``data`` shape, dtype, or unit differs from ``self.data``.
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedPostNumConn((data, self.indices), shape=self.shape)

    def todense(self):
        """
        Converts the FixedPostNumConn sparse matrix to a dense JAX NumPy array.

        This method first converts the internal representation to Coordinate (COO)
        format using `fixed_post_num_to_coo` to obtain the row and column indices
        corresponding to the stored data. Then, it uses these indices and the
        data to construct a dense matrix of the specified shape.

        Returns
        -------
        jax.Array or u.Quantity
            Dense matrix representation.

        Examples
        --------
            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPostNumConn
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.]])
            >>> indices = jnp.array([[0, 1], [1, 0]]) # post-synaptic indices
            >>> shape = (2, 2) # (num_pre, num_post)
            >>> mat = FixedPostNumConn((data, indices), shape=shape)
            >>>
            >>> dense_mat = mat.todense()
            >>> print(dense_mat)
            [[1. 2.]
             [4. 3.]]
        """
        pre_ids, post_ids, spinfo = fixed_post_num_to_coo(self)
        return _coo_todense(self.data.flatten(), pre_ids, post_ids, spinfo=spinfo)

    def tocoo(self) -> COO:
        """
        Converts the FixedPostNumConn sparse matrix to Coordinate (COO) format.

        This method generates the pre-synaptic (row) and post-synaptic (column)
        index arrays corresponding to the stored `data` array based on the
        `indices` (which store post-synaptic indices per pre-synaptic neuron).
        It then packages the `data`, `row` indices, and `col` indices into a
        `COO` sparse matrix object.

        Returns
        -------
        COO
            COO matrix representing the same sparse structure and values.

        Examples
        --------
            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPostNumConn
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.]])
            >>> indices = jnp.array([[0, 1], [1, 0]]) # post-synaptic indices
            >>> shape = (2, 2) # (num_pre, num_post)
            >>> mat = FixedPostNumConn((data, indices), shape=shape)
            >>>
            >>> coo_mat = mat.tocoo()
            >>> print("Data:", coo_mat.data)
            Data: [1. 2. 3. 4.]
            >>> print("Row Indices:", coo_mat.row)
            Row Indices: [0 0 1 1]
            >>> print("Column Indices:", coo_mat.col)
            Column Indices: [0 1 1 0]
            >>> print("Shape:", coo_mat.shape)
            Shape: (2, 2)
        """
        pre_ids, post_ids, spinfo = fixed_post_num_to_coo(self)
        return COO(
            (self.data.flatten(), pre_ids, post_ids),
            shape=self.shape,
            rows_sorted=spinfo.rows_sorted,
            cols_sorted=spinfo.cols_sorted,
        )

    def transpose(self, axes=None) -> 'FixedPreNumConn':
        """
        Transposes the matrix, returning a FixedPreNumConn representation.

        This operation swaps the dimensions of the matrix shape. The underlying
        `data` array remains the same. The `indices` array, which represents
        post-synaptic indices in FixedPostNumConn, is reinterpreted as
        pre-synaptic indices in the resulting FixedPreNumConn matrix.

        Notes
        -----
        The ``axes`` argument is not supported and must be ``None``.

        Parameters
        ----------
        axes : None, optional
            Included for compatibility with NumPy; must be ``None``.

        Returns
        -------
        FixedPreNumConn
            Transposed matrix view in fixed-pre format.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``.

        Examples
        --------
            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPostNumConn, FixedPreNumConn
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.]])
            >>> indices = jnp.array([[0, 1], [1, 0]]) # post-synaptic indices
            >>> shape = (2, 3) # (num_pre, num_post) - Example with non-square shape
            >>> mat = FixedPostNumConn((data, indices), shape=shape)
            >>>
            >>> mat_t = mat.transpose()
            >>> print(isinstance(mat_t, FixedPreNumConn))
            True
            >>> print("Transposed Shape:", mat_t.shape)
            Transposed Shape: (3, 2)
            >>> print("Transposed Data:", mat_t.data)
            Transposed Data: [[1. 2.]
             [3. 4.]]
            >>> # Note: indices are reinterpreted in FixedPreNumConn context
            >>> print("Transposed Indices:", mat_t.indices)
            Transposed Indices: [[0 1]
             [1 0]]
        """
        assert axes is None, "transpose does not support axes argument."
        # The indices array meaning changes:
        # In FixedPostNumConn: indices[i] are the post-synaptic targets for pre-synaptic neuron i.
        # In FixedPreNumConn: indices[j] are the pre-synaptic sources for post-synaptic neuron j.
        # When transposing, the roles of pre/post are swapped, so the same indices array
        # correctly represents the connections in the transposed view for FixedPreNumConn.
        return FixedPreNumConn((self.data, self.indices), shape=self.shape[::-1])

    def _unitary_op(self, op):
        return FixedPostNumConn((op(self.data), self.indices), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPostNumConn((op(self.data, other), self.indices), shape=self.shape)

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedPostNumConn((op(self.data, other), self.indices), shape=self.shape)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPostNumConn((op(other, self.data), self.indices), shape=self.shape)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedPostNumConn((op(other, self.data), self.indices,), shape=self.shape)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        # csr @ other
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, BinaryArray):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return binary_fcnmv(data, self.indices, other, shape=self.shape, transpose=False)
                elif other.ndim == 2:
                    return binary_fcnmm(data, self.indices, other, shape=self.shape, transpose=False)
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        elif isinstance(other, SparseFloat):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return spfloat_fcnmv(data, self.indices, other, shape=self.shape, transpose=False)
                elif other.ndim == 2:
                    return spfloat_fcnmm(data, self.indices, other, shape=self.shape, transpose=False)
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return fcnmv(data, self.indices, other, shape=self.shape, transpose=False)
            elif other.ndim == 2:
                return fcnmm(data, self.indices, other, shape=self.shape, transpose=False)
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        # other @ csr
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, BinaryArray):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return binary_fcnmv(data, self.indices, other, shape=self.shape, transpose=True)
                elif other.ndim == 2:
                    r = binary_fcnmm(data, self.indices, other.T, shape=self.shape, transpose=True)
                    return r.T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        elif isinstance(other, SparseFloat):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return spfloat_fcnmv(data, self.indices, other, shape=self.shape, transpose=True)
                elif other.ndim == 2:
                    r = spfloat_fcnmm(data, self.indices, other.T, shape=self.shape, transpose=True)
                    return r.T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return fcnmv(data, self.indices, other, shape=self.shape, transpose=True)
            elif other.ndim == 2:
                other = other.T
                r = fcnmm(data, self.indices, other, shape=self.shape, transpose=True)
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")


@jax.tree_util.register_pytree_node_class
class FixedPreNumConn(FixedNumConn):
    """
    Represents a sparse matrix with a fixed number of pre-synaptic connections
    per post-synaptic neuron.

    This format is efficient when each column (post-synaptic neuron) in the
    logical matrix has the same number of non-zero entries (incoming connections).
    It stores the matrix data and the corresponding pre-synaptic indices in
    dense arrays.

    Attributes
    ----------
    data : jax.numpy.ndarray
        A 2D array containing the non-zero values (e.g., synaptic weights)
        of the sparse matrix. The shape is `(num_post, num_conn)`, where
        `num_conn` is the fixed number of incoming connections per
        post-synaptic neuron. `data[j, k]` is the value of the connection
        from the k-th pre-synaptic neuron connected to post-synaptic neuron `j`.
    indices : jax.numpy.ndarray
        A 2D array containing the pre-synaptic indices (row indices) for each
        connection stored in `data`. The shape is `(num_post, num_conn)`.
        `indices[j, k]` is the index of the pre-synaptic neuron corresponding
        to the value `data[j, k]`.
    shape : tuple[int, int]
        A tuple `(num_pre, num_post)` representing the logical shape of the
        dense equivalent matrix. `num_pre` is the total number of pre-synaptic
        neurons (rows), and `num_post` is the total number of post-synaptic
        neurons (columns).
    num_conn : int
        The fixed number of pre-synaptic connections per post-synaptic neuron.
        Equal to `indices.shape[1]`.
    num_post : int
        The number of post-synaptic neurons (columns in the dense matrix).
        Equal to `indices.shape[0]` or `shape[1]`.
    num_pre : int
        The number of pre-synaptic neurons (rows in the dense matrix).
        Equal to `shape[0]`.
    nse : int
        The total number of specified elements (non-zeros). Equal to
        `num_post * num_conn`.
    dtype : jax.numpy.dtype
        The data type of the `data` array.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import FixedPreNumConn
        >>>
        >>> # Example: 3 post-synaptic neurons, each receiving from 2 pre-synaptic neurons.
        >>> # Total pre-synaptic neurons = 3. Shape = (3, 3)
        >>> data = jnp.array([[1., 2.], [3., 4.], [5., 6.]]) # Shape (num_post=3, num_conn=2)
        >>> # Pre-synaptic indices for each post-synaptic neuron:
        >>> # Post 0 receives from Pre 0 and Pre 1
        >>> # Post 1 receives from Pre 1 and Pre 0
        >>> # Post 2 receives from Pre 0 and Pre 2
        >>> indices = jnp.array([[0, 1], [1, 0], [0, 2]]) # Shape (num_post=3, num_conn=2)
        >>> shape = (3, 3) # (num_pre, num_post)
        >>>
        >>> mat = FixedPreNumConn((data, indices), shape=shape)
        >>>
        >>> print("Data:", mat.data)
        Data: [[1. 2.]
         [3. 4.]
         [5. 6.]]
        >>> print("Indices:", mat.indices)
        Indices: [[0 1]
         [1 0]
         [0 2]]
        >>> print("Shape:", mat.shape)
        Shape: (3, 3)
        >>> print("Number of connections per post-neuron:", mat.num_conn)
        Number of connections per post-neuron: 2
        >>>
        >>> # Convert to dense matrix
        >>> dense_mat = mat.todense()
        >>> print("Dense matrix:\\n", dense_mat)
        Dense matrix:
         [[1. 4. 5.]
          [2. 3. 0.]
          [0. 0. 6.]]
        >>>
        >>> # Transpose to FixedPostNumConn
        >>> mat_t = mat.transpose()
        >>> print("Transposed shape:", mat_t.shape)
        Transposed shape: (3, 3)
        >>> print("Transposed data (same):", mat_t.data)
        Transposed data (same): [[1. 2.]
         [3. 4.]
         [5. 6.]]
        >>> print("Transposed indices (reinterpreted):", mat_t.indices)
        Transposed indices (reinterpreted): [[0 1]
         [1 0]
         [0 2]]
    """
    __module__ = 'brainevent'

    data: Data
    indices: Index
    shape: MatrixShape
    num_conn = property(lambda self: self.indices.shape[1])
    num_post = property(lambda self: self.indices.shape[0])
    num_pre = property(lambda self: self.shape[0])
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(self, data, indices=None, *, shape: MatrixShape):
        if indices is None:
            args = data
        else:
            args = (data, indices)
        self.data, self.indices = map(u.math.asarray, args)
        _validate_fixed_conn_indices(
            self.indices,
            expected_rows=shape[1],
            kind='Pre-synaptic'
        )
        if self.data.size != 1 and self.data.shape != self.indices.shape:
            raise ValueError(
                f"Data shape {self.data.shape} must match indices shape {self.indices.shape}. "
                f"But got {self.data.shape} != {self.indices.shape}"
            )
        super().__init__(args, shape=shape)

        _contains_invalid_indices(self.indices, upper_bound=self.shape[0])

    def with_data(self, data: Data) -> 'FixedPreNumConn':
        """
        Creates a new FixedPreNumConn instance with the same indices and shape but different data.

        Parameters
        ----------
        data : Data
            New data array with the same shape, dtype, and unit as ``self.data``.

        Returns
        -------
        FixedPreNumConn
            New matrix with the provided data and unchanged connectivity.

        Raises
        ------
        AssertionError
            If ``data`` shape, dtype, or unit differs from ``self.data``.
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedPreNumConn((data, self.indices), shape=self.shape)

    def todense(self):
        """
        Converts the FixedPreNumConn sparse matrix to a dense JAX NumPy array.

        This method first converts the internal representation to Coordinate (COO)
        format using `fixed_pre_num_to_coo` to obtain the row and column indices
        corresponding to the stored data. Then, it uses these indices and the
        data to construct a dense matrix of the specified shape.

        Returns
        -------
        jax.Array or u.Quantity
            Dense matrix representation.

        Examples
        --------

        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPreNumConn
            >>>
            >>> # Example: 3 post-synaptic neurons, each receiving from 2 pre-synaptic neurons
            >>> data = jnp.array([[1., 2.], [3., 4.], [5., 6.]]) # Shape (num_post, num_conn)
            >>> indices = jnp.array([[0, 1], [1, 0], [0, 2]]) # pre-synaptic indices for each post-synaptic neuron
            >>> shape = (3, 3) # (num_pre, num_post)
            >>> mat = FixedPreNumConn((data, indices), shape=shape)
            >>>
            >>> dense_mat = mat.todense()
            >>> print(dense_mat)
            [[1. 4. 5.]
             [2. 3. 0.]
             [0. 0. 6.]]
        """
        pre_ids, post_ids, spinfo = fixed_pre_num_to_coo(self)
        return _coo_todense(self.data.flatten(), pre_ids, post_ids, spinfo=spinfo)

    def tocoo(self) -> COO:
        """
        Converts the FixedPreNumConn sparse matrix to Coordinate (COO) format.

        This method generates the pre-synaptic (row) and post-synaptic (column)
        index arrays corresponding to the stored `data` array based on the
        `indices` (which store pre-synaptic indices per post-synaptic neuron).
        It then packages the `data`, `row` indices, and `col` indices into a
        `COO` sparse matrix object.

        Returns
        -------
        COO
            COO matrix representing the same sparse structure and values.

        Examples
        --------

        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPreNumConn
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.], [5., 6.]]) # Shape (num_post, num_conn)
            >>> indices = jnp.array([[0, 1], [1, 0], [0, 2]]) # pre-synaptic indices
            >>> shape = (3, 3) # (num_pre, num_post)
            >>> mat = FixedPreNumConn((data, indices), shape=shape)
            >>>
            >>> coo_mat = mat.tocoo()
            >>> print("Data:", coo_mat.data)
            Data: [1. 2. 3. 4. 5. 6.]
            >>> print("Row Indices:", coo_mat.row) # Pre-synaptic indices
            Row Indices: [0 1 1 0 0 2]
            >>> print("Column Indices:", coo_mat.col) # Post-synaptic indices
            Column Indices: [0 0 1 1 2 2]
            >>> print("Shape:", coo_mat.shape)
            Shape: (3, 3)
        """
        pre_ids, post_ids, spinfo = fixed_pre_num_to_coo(self)
        return COO(
            (self.data.flatten(), pre_ids, post_ids),
            shape=self.shape,
            rows_sorted=spinfo.rows_sorted,
            cols_sorted=spinfo.cols_sorted,
        )

    def transpose(self, axes=None) -> FixedPostNumConn:
        """
        Transposes the matrix, returning a FixedPostNumConn representation.

        This operation swaps the dimensions of the matrix shape. The underlying
        `data` array remains the same. The `indices` array, which represents
        pre-synaptic indices in FixedPreNumConn, is reinterpreted as
        post-synaptic indices in the resulting FixedPostNumConn matrix.

        Notes
        -----
        The ``axes`` argument is not supported and must be ``None``.

        Parameters
        ----------
        axes : None, optional
            Included for compatibility with NumPy; must be ``None``.

        Returns
        -------
        FixedPostNumConn
            Transposed matrix view in fixed-post format.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``.

        Examples
        --------

        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPreNumConn, FixedPostNumConn
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.], [5., 6.]]) # Shape (num_post, num_conn)
            >>> indices = jnp.array([[0, 1], [1, 0], [0, 2]]) # pre-synaptic indices
            >>> shape = (3, 4) # (num_pre, num_post) - Example with non-square shape
            >>> mat = FixedPreNumConn((data, indices), shape=shape)
            >>>
            >>> mat_t = mat.transpose()
            >>> print(isinstance(mat_t, FixedPostNumConn))
            True
            >>> print("Transposed Shape:", mat_t.shape)
            Transposed Shape: (4, 3)
            >>> print("Transposed Data:", mat_t.data)
            Transposed Data: [[1. 2.]
             [3. 4.]
             [5. 6.]]
            >>> # Note: indices are reinterpreted in FixedPostNumConn context
            >>> print("Transposed Indices:", mat_t.indices)
            Transposed Indices: [[0 1]
             [1 0]
             [0 2]]
        """
        assert axes is None, "transpose does not support axes argument."
        # The indices array meaning changes:
        # In FixedPreNumConn: indices[j] are the pre-synaptic sources for post-synaptic neuron j.
        # In FixedPostNumConn: indices[i] are the post-synaptic targets for pre-synaptic neuron i.
        # When transposing, the roles of pre/post are swapped, so the same indices array
        # correctly represents the connections in the transposed view for FixedPostNumConn.
        return FixedPostNumConn((self.data, self.indices), shape=self.shape[::-1])

    def _unitary_op(self, op):
        return FixedPreNumConn((op(self.data), self.indices), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPreNumConn((op(self.data, other), self.indices), shape=self.shape)

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedPreNumConn((op(self.data, other), self.indices), shape=self.shape)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPreNumConn((op(other, self.data), self.indices), shape=self.shape)

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedPreNumConn((op(other, self.data), self.indices,), shape=self.shape)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        # csr @ other
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, BinaryArray):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return binary_fcnmv(data, self.indices, other, shape=self.shape[::-1], transpose=True)
                elif other.ndim == 2:
                    return binary_fcnmm(data, self.indices, other, shape=self.shape[::-1], transpose=True)
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        elif isinstance(other, SparseFloat):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return spfloat_fcnmv(data, self.indices, other, shape=self.shape[::-1], transpose=True)
                elif other.ndim == 2:
                    return spfloat_fcnmm(data, self.indices, other, shape=self.shape[::-1], transpose=True)
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return fcnmv(data, self.indices, other, shape=self.shape[::-1], transpose=True)
            elif other.ndim == 2:
                return fcnmm(data, self.indices, other, shape=self.shape[::-1], transpose=True)
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        # other @ csr
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        if isinstance(other, BinaryArray):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return binary_fcnmv(data, self.indices, other, shape=self.shape[::-1], transpose=False)
                elif other.ndim == 2:
                    r = binary_fcnmm(data, self.indices, other.T, shape=self.shape[::-1], transpose=False)
                    return r.T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        elif isinstance(other, SparseFloat):
            if not other.indexed:
                other = other.value
                if other.ndim == 1:
                    return spfloat_fcnmv(data, self.indices, other, shape=self.shape[::-1], transpose=False)
                elif other.ndim == 2:
                    r = spfloat_fcnmm(data, self.indices, other.T, shape=self.shape[::-1], transpose=False)
                    return r.T
                else:
                    raise NotImplementedError(f"matmul with object of shape {other.shape}")
            else:
                raise NotImplementedError

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return fcnmv(data, self.indices, other, shape=self.shape[::-1], transpose=False)
            elif other.ndim == 2:
                other = other.T
                r = fcnmm(data, self.indices, other, shape=self.shape[::-1], transpose=False)
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")


def fixed_post_num_to_coo(self: FixedPostNumConn):
    """
    Converts a FixedPostNumConn sparse matrix representation to COO format.

    In FixedPostNumConn, `indices` stores the post-synaptic indices for each
    pre-synaptic neuron. This function generates the corresponding pre-synaptic
    and post-synaptic index arrays needed for the COO format.

    Parameters
    ----------
    self : FixedPostNumConn
        Fixed-post sparse matrix.

    Returns
    -------
    tuple[jax.Array, jax.Array, COOInfo]
        ``(pre_ids, post_ids, spinfo)`` in COO-compatible form.
    """
    pre_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    post_ids = self.indices.flatten()
    spinfo = COOInfo(self.shape, rows_sorted=True, cols_sorted=False)
    return pre_ids, post_ids, spinfo


def fixed_pre_num_to_coo(self: FixedPreNumConn):
    """
    Converts a FixedPreNumConn sparse matrix representation to COO format.

    In FixedPreNumConn, `indices` stores the pre-synaptic indices for each
    post-synaptic neuron. This function generates the corresponding pre-synaptic
    and post-synaptic index arrays needed for the COO format.

    Parameters
    ----------
    self : FixedPreNumConn
        Fixed-pre sparse matrix.

    Returns
    -------
    tuple[jax.Array, jax.Array, COOInfo]
        ``(pre_ids, post_ids, spinfo)`` in COO-compatible form.
    """
    pre_ids = self.indices.flatten()
    post_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    spinfo = COOInfo(self.shape, rows_sorted=False, cols_sorted=True)
    return pre_ids, post_ids, spinfo
