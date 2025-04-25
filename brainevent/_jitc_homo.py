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

# -*- coding: utf-8 -*-


import operator
from typing import Union

import brainunit as u
import jax

from ._compatible_import import JAXSparse
from ._csr_event_impl import _event_csr_matvec, _event_csr_matmat
from ._csr_float_impl import _csr_matvec, _csr_matmat
from ._event import EventArray
from ._jitc import JITCMatrix
from ._misc import _csr_to_coo
from ._typing import MatrixShape

__all__ = [
    'JITRHomo',
    'JITCHomo',
]


class JITHomo(JITCMatrix):
    data: Union[jax.Array, u.Quantity]
    shape: MatrixShape
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        data: Union[jax.typing.ArrayLike, u.Quantity],
        seed: Union[int, jax.Array],
        *,
        shape: MatrixShape
    ):
        super().__init__((), shape=shape)
        self.data = u.math.asarray(data)
        self.seed = seed

    def with_data(self, data: Union[jax.typing.ArrayLike, u.Quantity]):
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return type(self)(data, self.seed, shape=self.shape)

    def tree_flatten(self):
        """
        Flattens the JITHomo object for JAX transformation compatibility.

        This method is part of JAX's pytree protocol that enables JAX transformations
        on custom classes. It separates the object into arrays that should be traced
        through JAX transformations (children) and auxiliary static data.

        Returns:
            tuple: A tuple with two elements:
                - A tuple of JAX-traceable arrays (only self.data in this case)
                - A dictionary of auxiliary data (shape, indices, and indptr)
        """
        return (self.data, self.seed), {"shape": self.shape, }

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a JITHomo object from flattened data.

        This method is part of JAX's pytree protocol that enables JAX transformations
        on custom classes. It rebuilds the JITHomo object from the flattened representation
        produced by tree_flatten.

        Args:
            aux_data (dict): Dictionary containing auxiliary static data (shape, indices, indptr)
            children (tuple): Tuple of JAX arrays that were transformed (contains only data)

        Returns:
            JITRHomo: Reconstructed JITHomo object

        Raises:
            ValueError: If the aux_data dictionary doesn't contain the expected keys
        """
        obj = object.__new__(cls)
        obj.data, obj.seed = children
        if aux_data.keys() != {'shape', }:
            raise ValueError("aux_data must contain 'shape', keys. But got: "
                             f"{aux_data.keys()}")
        obj.__dict__.update(**aux_data)
        return obj

    def _unitary_op(self, op):
        raise NotImplementedError("unitary operation not implemented.")

    def __abs__(self):
        return self._unitary_op(operator.abs)

    def __neg__(self):
        return self._unitary_op(operator.neg)

    def __pos__(self):
        return self._unitary_op(operator.pos)

    def _binary_op(self, other, op):
        raise NotImplementedError("binary operation not implemented.")

    def __mul__(self, other: Union[jax.Array, u.Quantity]):
        return self._binary_op(other, operator.mul)

    def __div__(self, other: Union[jax.Array, u.Quantity]):
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
        raise NotImplementedError("binary operation not implemented.")

    def __rmul__(self, other: Union[jax.Array, u.Quantity]):
        return self._binary_rop(other, operator.mul)

    def __rdiv__(self, other: Union[jax.Array, u.Quantity]):
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
class JITRHomo(JITHomo):
    """
    Just-In-Time connectivity Row-oriented Homogeneous matrix representation.

    This class represents a row-oriented homogeneous sparse matrix optimized for just-in-time
    connectivity. It stores a single homogeneous value that applies to all non-zero elements
    in the sparse matrix, along with indexing information to specify where these non-zero
    elements are located.

    The row-oriented structure follows the CSR (Compressed Sparse Row) format, making row-based
    operations more efficient than column-based ones.

    Attributes:
        data (Union[jax.Array, u.Quantity]): The single value used for all non-zero elements
        seed (Union[int, jax.Array]): Random seed used for initialization of the sparse structure
        shape (MatrixShape): The shape of the matrix as a tuple (rows, cols)
        dtype: The data type of the matrix elements (property inherited from parent)

    Examples:
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITRHomo
        >>>
        >>> # Create a homogeneous matrix with value 0.5 for all non-zero elements
        >>> homo_matrix = JITRHomo(0.5, seed=42, shape=(10, 10))
        >>>
        >>> # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = homo_matrix @ vec
        >>>
        >>> # Apply scalar operation
        >>> scaled = homo_matrix * 2.0

    Notes:
        - More memory-efficient than dense matrices for sparse connectivity patterns
        - Compatible with JAX transformations (jit, grad, vmap, etc.)
        - Well-suited for neural network connectivity matrices with uniform weights
    """

    def _unitary_op(self, op) -> 'JITRHomo':
        return JITRHomo(op(self.data), self.seed, shape=self.shape)

    def todense(self) -> Union[jax.Array, u.Quantity]:
        return _jitr_homo_todense(self.data, self.indices, self.indptr, shape=self.shape)

    def transpose(self, axes=None) -> 'JITCHomo':
        assert axes is None, "transpose does not support axes argument."
        return JITC_CSC((self.data, self.indices, self.indptr), shape=self.shape[::-1])

    def _binary_op(self, other, op) -> 'JITRHomo':
        if isinstance(other, JITRHomo):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return JITRHomo(
                    (op(self.data, other.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return JITRHomo(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return JITRHomo(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITRHomo':
        if isinstance(other, JITRHomo):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return JITRHomo(
                    (op(other.data, self.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return JITRHomo(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return JITRHomo(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
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
                    other,
                    shape=self.shape
                )
            elif other.ndim == 2:
                return _csr_matmat(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other) -> Union[jax.Array, u.Quantity]:
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
                    other,
                    shape=self.shape,
                    transpose=True
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")


@jax.tree_util.register_pytree_node_class
class JITCHomo(JITHomo):
    """
    Just-In-Time connectivity Column-oriented Homogeneous matrix representation.

    This class represents a column-oriented homogeneous sparse matrix optimized for JAX-based
    transformations. It stores a single homogeneous value that applies to all non-zero elements
    in the sparse matrix, along with indexing information to specify where these non-zero
    elements are located.

    The column-oriented structure follows the CSC (Compressed Sparse Column) format, making
    column-based operations more efficient than row-based ones. This is the transpose-oriented
    counterpart to :class:`JITRHomo`.

    Attributes:
        data (Union[jax.Array, u.Quantity]): The single value used for all non-zero elements
        seed (Union[int, jax.Array]): Random seed used for initialization of the sparse structure
        shape (MatrixShape): The shape of the matrix as a tuple (rows, cols)
        dtype: The data type of the matrix elements (property inherited from parent)

    Examples:
        >>> import jax
        >>> import brainunit as u
        >>> from brainevent import JITCHomo
        >>>
        >>> # Create a homogeneous matrix with value 0.5 for all non-zero elements
        >>> homo_matrix = JITCHomo(0.5, seed=42, shape=(10, 10))
        >>>
        >>> # Perform matrix-vector multiplication
        >>> vec = jax.numpy.ones(10)
        >>> result = homo_matrix @ vec
        >>>
        >>> # Apply scalar operation
        >>> scaled = homo_matrix * 2.0
        >>>
        >>> # Transpose to get a row-oriented matrix
        >>> row_matrix = homo_matrix.transpose()

    Notes:
        - Registered as a JAX pytree node for compatibility with JAX transformations
        - More efficient than JITRHomo for column slicing operations
        - Compatible with all standard mathematical operations
        - Well-suited for neural network connectivity matrices with uniform weights
    """

    def _unitary_op(self, op) -> 'JITCHomo':
        return JITCHomo(op(self.data), self.seed, shape=self.shape)

    def todense(self) -> Union[jax.Array, u.Quantity]:
        return _jitr_homo_todense(self.data, self.indices, self.indptr, shape=self.shape)

    def transpose(self, axes=None) -> 'JITRHomo':
        assert axes is None, "transpose does not support axes argument."
        return JITRHomo((self.data, self.indices, self.indptr), shape=self.shape[::-1])

    def _binary_op(self, other, op) -> 'JITCHomo':
        if isinstance(other, JITRHomo):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return JITRHomo(
                    (op(self.data, other.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return JITRHomo(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return JITRHomo(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'JITCHomo':
        if isinstance(other, JITRHomo):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return JITRHomo(
                    (op(other.data, self.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return JITRHomo(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return JITRHomo(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other) -> Union[jax.Array, u.Quantity]:
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
                    other,
                    shape=self.shape
                )
            elif other.ndim == 2:
                return _csr_matmat(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other) -> Union[jax.Array, u.Quantity]:
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
                    other,
                    shape=self.shape,
                    transpose=True
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")
