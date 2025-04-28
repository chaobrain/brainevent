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

import operator
from typing import Tuple

import brainunit as u
import jax
import jax.numpy as jnp

from ._compatible_import import JAXSparse
from ._coo import COO
from ._event import EventArray
from ._fixed_conn_num_event_impl import event_fixed_post_num_mv_p_call
from ._fixed_conn_num_float_impl import fixed_post_num_mv_p_call
from ._misc import _coo_todense, COOInfo
from ._typing import Data, MatrixShape, Index

__all__ = [
    'FixedPostConnNum',
    'FixedPreConnNum',
]


class FixedConnNum(u.sparse.SparseMatrix):
    """
    Base class for fixed number of connections.
    """
    data: Data
    indices: Index
    shape: MatrixShape

    def tree_flatten(self):
        """
        Flattens the FixedConnNum object into its constituent parts for JAX PyTree processing.

        Returns:
            A tuple containing:
                - A tuple of children nodes (dynamic data, i.e., self.data).
                - A tuple of auxiliary data (static data, i.e., self.indices and self.shape).
        """
        return (self.data,), (self.indices, self.shape)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a FixedConnNum object from its flattened representation.

        Args:
            aux_data: A tuple containing the auxiliary data (indices, shape).
            children: A tuple containing the children nodes (data,).

        Returns:
            An instance of the FixedConnNum class reconstructed from the provided data.
        """
        data, = children
        indices, shape = aux_data
        return cls((data, indices), shape=shape)

    def _unitary_op(self, op):
        raise NotImplementedError

    def __abs__(self):
        return self._unitary_op(operator.abs)

    def __neg__(self):
        return self._unitary_op(operator.neg)

    def __pos__(self):
        return self._unitary_op(operator.pos)

    def _binary_op(self, other, op):
        raise NotImplementedError

    def __mul__(self, other: Data):
        return self._binary_op(other, operator.mul)

    def __div__(self, other: Data):
        return self._binary_op(other, operator.truediv)

    def __truediv__(self, other):
        return self.__div__(other)

    def __add__(self, other):
        return self._binary_op(other, operator.add)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub)

    def __mod__(self, other):
        return self._binary_op(other, operator.mod)

    def _binary_rop(self, other, op):
        raise NotImplementedError

    def __rmul__(self, other: Data):
        return self._binary_rop(other, operator.mul)

    def __rdiv__(self, other: Data):
        return self._binary_rop(other, operator.truediv)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __radd__(self, other):
        return self._binary_rop(other, operator.add)

    def __rsub__(self, other):
        return self._binary_rop(other, operator.sub)

    def __rmod__(self, other):
        return self._binary_rop(other, operator.mod)


@jax.tree_util.register_pytree_node_class
class FixedPostConnNum(FixedConnNum):
    """
    Fixed total number of postsynaptic neurons.
    """
    data: Data
    indices: Index
    shape: MatrixShape
    num_pre = property(lambda self: self.indices.shape[0])
    num_conn = property(lambda self: self.indices.shape[1])
    num_post = property(lambda self: self.shape[1])
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(self, args: Tuple[Data, Index], *, shape: MatrixShape):
        self.data, self.indices = map(u.math.asarray, args)
        assert self.indices.shape[0] == shape[0], \
            f'Pre-synaptic neuron number mismatch. {self.indices.shape[0]} != {shape[0]}'
        super().__init__(args, shape=shape)

    def with_data(self, data: Data) -> 'FixedPostConnNum':
        """
        Creates a new FixedPostConnNum instance with the same indices and shape but different data.

        Args:
            data: The new data array. Must have the same shape, dtype, and unit as the original data.

        Returns:
            A new FixedPostConnNum instance with the provided data.

        Raises:
            AssertionError: If the provided data does not match the shape, dtype, or unit of the original data.
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedPostConnNum((data, self.indices), shape=self.shape)

    def todense(self):
        """
        Converts the FixedPostConnNum sparse matrix to a dense JAX NumPy array.

        This method first converts the internal representation to Coordinate (COO)
        format using `fixed_post_num_to_coo` to obtain the row and column indices
        corresponding to the stored data. Then, it uses these indices and the
        data to construct a dense matrix of the specified shape.

        Returns:
            jax.numpy.ndarray: The dense matrix representation.

        Examples:
            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPostConnNum
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.]])
            >>> indices = jnp.array([[0, 1], [1, 0]]) # post-synaptic indices
            >>> shape = (2, 2) # (num_pre, num_post)
            >>> mat = FixedPostConnNum((data, indices), shape=shape)
            >>>
            >>> dense_mat = mat.todense()
            >>> print(dense_mat)
            [[1. 2.]
             [4. 3.]]
        """
        pre_ids, post_ids, spinfo = fixed_post_num_to_coo(self)
        return _coo_todense(self.data, pre_ids, post_ids, spinfo=spinfo)

    def tocoo(self) -> COO:
        """
        Converts the FixedPostConnNum sparse matrix to Coordinate (COO) format.

        This method generates the pre-synaptic (row) and post-synaptic (column)
        index arrays corresponding to the stored `data` array based on the
        `indices` (which store post-synaptic indices per pre-synaptic neuron).
        It then packages the `data`, `row` indices, and `col` indices into a
        `COO` sparse matrix object.

        Returns:
            COO: A COO sparse matrix object representing the same matrix.

        Examples:
            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPostConnNum
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.]])
            >>> indices = jnp.array([[0, 1], [1, 0]]) # post-synaptic indices
            >>> shape = (2, 2) # (num_pre, num_post)
            >>> mat = FixedPostConnNum((data, indices), shape=shape)
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
        return COO((self.data, (pre_ids, post_ids)), shape=self.shape, spinfo=spinfo)

    def transpose(self, axes=None) -> 'FixedPreConnNum':
        """
        Transposes the matrix, returning a FixedPreConnNum representation.

        This operation swaps the dimensions of the matrix shape. The underlying
        `data` array remains the same. The `indices` array, which represents
        post-synaptic indices in FixedPostConnNum, is reinterpreted as
        pre-synaptic indices in the resulting FixedPreConnNum matrix.

        Note:
            The `axes` argument is not supported and must be None.

        Args:
            axes: Must be None. Included for compatibility with NumPy's transpose
                  method signature but is not used.

        Returns:
            FixedPreConnNum: The transposed matrix.

        Raises:
            AssertionError: If `axes` is not None.

        Examples:
            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPostConnNum, FixedPreConnNum
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.]])
            >>> indices = jnp.array([[0, 1], [1, 0]]) # post-synaptic indices
            >>> shape = (2, 3) # (num_pre, num_post) - Example with non-square shape
            >>> mat = FixedPostConnNum((data, indices), shape=shape)
            >>>
            >>> mat_t = mat.transpose()
            >>> print(isinstance(mat_t, FixedPreConnNum))
            True
            >>> print("Transposed Shape:", mat_t.shape)
            Transposed Shape: (3, 2)
            >>> print("Transposed Data:", mat_t.data)
            Transposed Data: [[1. 2.]
             [3. 4.]]
            >>> # Note: indices are reinterpreted in FixedPreConnNum context
            >>> print("Transposed Indices:", mat_t.indices)
            Transposed Indices: [[0 1]
             [1 0]]
        """
        assert axes is None, "transpose does not support axes argument."
        # The indices array meaning changes:
        # In FixedPostConnNum: indices[i] are the post-synaptic targets for pre-synaptic neuron i.
        # In FixedPreConnNum: indices[j] are the pre-synaptic sources for post-synaptic neuron j.
        # When transposing, the roles of pre/post are swapped, so the same indices array
        # correctly represents the connections in the transposed view for FixedPreConnNum.
        return FixedPreConnNum((self.data, self.indices), shape=self.shape[::-1])

    def _unitary_op(self, op):
        return FixedPostConnNum((op(self.data), self.indices), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, FixedPostConnNum):
            if id(other.indices) == id(self.indices):
                return FixedPostConnNum((op(self.data, other.data), self.indices), shape=self.shape)

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPostConnNum((op(self.data, other), self.indices), shape=self.shape)

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedPostConnNum((op(self.data, other), self.indices), shape=self.shape)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, FixedPostConnNum):
            if id(other.indices) == id(self.indices):
                return FixedPostConnNum((op(other.data, self.data), self.indices), shape=self.shape)

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPostConnNum((op(other, self.data), self.indices), shape=self.shape)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedPostConnNum((op(other, self.data), self.indices,), shape=self.shape)
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
                return event_fixed_post_num_mv_p_call(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=False,
                    float_as_event=True,
                )[0]
            elif other.ndim == 2:
                return _event_perfect_ellmm(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=False,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return fixed_post_num_mv_p_call(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=False,
                )[0]
            elif other.ndim == 2:
                return _perfect_ellmm(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=False,
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
                return event_fixed_post_num_mv_p_call(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=True,
                )[0]
            elif other.ndim == 2:
                other = other.T
                r = _event_perfect_ellmm(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=True,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return fixed_post_num_mv_p_call(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=True,
                )[0]
            elif other.ndim == 2:
                other = other.T
                r = _perfect_ellmm(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=True,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")


@jax.tree_util.register_pytree_node_class
class FixedPreConnNum(FixedConnNum):
    """
    Fixed total number of pre-synaptic neurons.
    """
    data: Data
    indices: Index
    shape: MatrixShape
    num_conn = property(lambda self: self.indices.shape[1])
    num_post = property(lambda self: self.indices.shape[0])
    num_pre = property(lambda self: self.shape[0])
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(self, args: Tuple[Data, Index], *, shape: MatrixShape):
        self.data, self.indices = map(u.math.asarray, args)
        assert self.indices.shape[0] == shape[1], 'Post-synaptic neuron number mismatch.'
        super().__init__(args, shape=shape)

    def with_data(self, data: Data) -> 'FixedPreConnNum':
        """
        Creates a new FixedPreConnNum instance with the same indices and shape but different data.

        Args:
            data: The new data array. Must have the same shape, dtype, and unit as the original data.

        Returns:
            A new FixedPreConnNum instance with the provided data.

        Raises:
            AssertionError: If the provided data does not match the shape, dtype, or unit of the original data.
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedPreConnNum((data, self.indices), shape=self.shape)

    def todense(self):
        """
        Converts the FixedPreConnNum sparse matrix to a dense JAX NumPy array.

        This method first converts the internal representation to Coordinate (COO)
        format using `fixed_pre_num_to_coo` to obtain the row and column indices
        corresponding to the stored data. Then, it uses these indices and the
        data to construct a dense matrix of the specified shape.

        Returns:
            jax.numpy.ndarray: The dense matrix representation.

        Examples:
            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPreConnNum
            >>>
            >>> # Example: 3 post-synaptic neurons, each receiving from 2 pre-synaptic neurons
            >>> data = jnp.array([[1., 2.], [3., 4.], [5., 6.]]) # Shape (num_post, num_conn)
            >>> indices = jnp.array([[0, 1], [1, 0], [0, 2]]) # pre-synaptic indices for each post-synaptic neuron
            >>> shape = (3, 3) # (num_pre, num_post)
            >>> mat = FixedPreConnNum((data, indices), shape=shape)
            >>>
            >>> dense_mat = mat.todense()
            >>> print(dense_mat)
            [[1. 4. 5.]
             [2. 3. 0.]
             [0. 0. 6.]]
        """
        pre_ids, post_ids, spinfo = fixed_pre_num_to_coo(self)
        return _coo_todense(self.data, pre_ids, post_ids, spinfo=spinfo)

    def tocoo(self) -> COO:
        """
        Converts the FixedPreConnNum sparse matrix to Coordinate (COO) format.

        This method generates the pre-synaptic (row) and post-synaptic (column)
        index arrays corresponding to the stored `data` array based on the
        `indices` (which store pre-synaptic indices per post-synaptic neuron).
        It then packages the `data`, `row` indices, and `col` indices into a
        `COO` sparse matrix object.

        Returns:
            COO: A COO sparse matrix object representing the same matrix.

        Examples:
            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPreConnNum
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.], [5., 6.]]) # Shape (num_post, num_conn)
            >>> indices = jnp.array([[0, 1], [1, 0], [0, 2]]) # pre-synaptic indices
            >>> shape = (3, 3) # (num_pre, num_post)
            >>> mat = FixedPreConnNum((data, indices), shape=shape)
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
        return COO((self.data, (pre_ids, post_ids)), shape=self.shape, spinfo=spinfo)

    def transpose(self, axes=None) -> FixedPostConnNum:
        """
        Transposes the matrix, returning a FixedPostConnNum representation.

        This operation swaps the dimensions of the matrix shape. The underlying
        `data` array remains the same. The `indices` array, which represents
        pre-synaptic indices in FixedPreConnNum, is reinterpreted as
        post-synaptic indices in the resulting FixedPostConnNum matrix.

        Note:
            The `axes` argument is not supported and must be None.

        Args:
            axes: Must be None. Included for compatibility with NumPy's transpose
                  method signature but is not used.

        Returns:
            FixedPostConnNum: The transposed matrix.

        Raises:
            AssertionError: If `axes` is not None.

        Examples:
            >>> import jax.numpy as jnp
            >>> from brainevent import FixedPreConnNum, FixedPostConnNum
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.], [5., 6.]]) # Shape (num_post, num_conn)
            >>> indices = jnp.array([[0, 1], [1, 0], [0, 2]]) # pre-synaptic indices
            >>> shape = (3, 4) # (num_pre, num_post) - Example with non-square shape
            >>> mat = FixedPreConnNum((data, indices), shape=shape)
            >>>
            >>> mat_t = mat.transpose()
            >>> print(isinstance(mat_t, FixedPostConnNum))
            True
            >>> print("Transposed Shape:", mat_t.shape)
            Transposed Shape: (4, 3)
            >>> print("Transposed Data:", mat_t.data)
            Transposed Data: [[1. 2.]
             [3. 4.]
             [5. 6.]]
            >>> # Note: indices are reinterpreted in FixedPostConnNum context
            >>> print("Transposed Indices:", mat_t.indices)
            Transposed Indices: [[0 1]
             [1 0]
             [0 2]]
        """
        assert axes is None, "transpose does not support axes argument."
        # The indices array meaning changes:
        # In FixedPreConnNum: indices[j] are the pre-synaptic sources for post-synaptic neuron j.
        # In FixedPostConnNum: indices[i] are the post-synaptic targets for pre-synaptic neuron i.
        # When transposing, the roles of pre/post are swapped, so the same indices array
        # correctly represents the connections in the transposed view for FixedPostConnNum.
        return FixedPostConnNum((self.data, self.indices), shape=self.shape[::-1])

    def _unitary_op(self, op):
        return FixedPreConnNum((op(self.data), self.indices), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, FixedPreConnNum):
            if id(other.indices) == id(self.indices):
                return FixedPreConnNum((op(self.data, other.data), self.indices), shape=self.shape)

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPreConnNum((op(self.data, other), self.indices), shape=self.shape)

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedPreConnNum((op(self.data, other), self.indices), shape=self.shape)

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, FixedPreConnNum):
            if id(other.indices) == id(self.indices):
                return FixedPreConnNum((op(other.data, self.data), self.indices), shape=self.shape)

        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPreConnNum((op(other, self.data), self.indices), shape=self.shape)

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedPreConnNum((op(other, self.data), self.indices,), shape=self.shape)
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
                return _event_perfect_ellmv(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=False,
                )
            elif other.ndim == 2:
                return _event_perfect_ellmm(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=False,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return _perfect_ellmv(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=False,
                )
            elif other.ndim == 2:
                return _perfect_ellmm(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=False,
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
                return _event_perfect_ellmv(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=True,
                )
            elif other.ndim == 2:
                other = other.T
                r = _event_perfect_ellmm(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=True,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return _perfect_ellmv(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=True,
                )
            elif other.ndim == 2:
                other = other.T
                r = _perfect_ellmm(
                    data,
                    self.indices,
                    other,
                    shape=self.shape,
                    transpose=True,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")


def fixed_post_num_to_coo(self: FixedPostConnNum):
    """
    Converts a FixedPostConnNum sparse matrix representation to COO format.

    In FixedPostConnNum, `indices` stores the post-synaptic indices for each
    pre-synaptic neuron. This function generates the corresponding pre-synaptic
    and post-synaptic index arrays needed for the COO format.

    Args:
        self: The FixedPostConnNum instance.

    Returns:
        A tuple containing:
            - pre_ids (jax.numpy.ndarray): The array of pre-synaptic indices.
            - post_ids (jax.numpy.ndarray): The array of post-synaptic indices.
            - spinfo (COOInfo): Information about the COO matrix properties.
    """
    pre_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    post_ids = self.indices.flatten()
    spinfo = COOInfo(self.shape, rows_sorted=True, cols_sorted=False)
    return pre_ids, post_ids, spinfo


def fixed_pre_num_to_coo(self: FixedPreConnNum):
    """
    Converts a FixedPreConnNum sparse matrix representation to COO format.

    In FixedPreConnNum, `indices` stores the pre-synaptic indices for each
    post-synaptic neuron. This function generates the corresponding pre-synaptic
    and post-synaptic index arrays needed for the COO format.

    Args:
        self: The FixedPreConnNum instance.

    Returns:
        A tuple containing:
            - pre_ids (jax.numpy.ndarray): The array of pre-synaptic indices.
            - post_ids (jax.numpy.ndarray): The array of post-synaptic indices.
            - spinfo (COOInfo): Information about the COO matrix properties.
    """
    pre_ids = self.indices.flatten()
    post_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    spinfo = COOInfo(self.shape, rows_sorted=False, cols_sorted=True)
    return pre_ids, post_ids, spinfo
