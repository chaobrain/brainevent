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


import operator
from typing import Union, Sequence, Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from brainunit.sparse._csr import _csr_to_coo, _csr_todense
from jax.experimental.sparse import CSR
from jax.experimental.sparse import JAXSparse

from ._csr_event_impl import _event_csr_matvec, _event_csr_matmat
from ._csr_float_impl import _csr_matvec, _csr_matmat
from ._event import EventArray

__all__ = [
    'CSR',
    'CSC',
]


# TODO: docstring needed to be improved
@jax.tree_util.register_pytree_node_class
class CSR(u.sparse.SparseMatrix):
    """
    Event-driven and Unit-aware CSR matrix.
    """
    data: Union[jax.Array, u.Quantity]
    indices: jax.Array
    indptr: jax.Array
    id: jax.Array
    shape: tuple[int, int]
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        args: Sequence[Union[jax.Array, np.ndarray, u.Quantity]],
        *,
        shape: Tuple[int, int]
    ):
        self.data, self.indices, self.indptr = map(u.math.asarray, args)
        m = (self.indices.shape[0] - 1) // 1024 + 1
        self.id = jnp.zeros(m, self.indices.dtype)
        sum = 0
        k = 0

        def cond_fun(carry):
            k, sum, indptr = carry
            return indptr[k + 1] <= sum

        def body_fun(carry):
            k, sum, indptr = carry
            return k + 1, sum, indptr

        for i in range(1, m):
            sum += 1024
            k = jax.lax.while_loop(cond_fun, body_fun, (k, sum, self.indptr))[0]
            self.id.at[i].set(k)
        super().__init__(args, shape=shape)

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=jnp.int32) -> 'CSR':
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csr = u.sparse.csr_fromdense(mat, nse=nse, index_dtype=index_dtype)
        return CSR((csr.data, csr.indices, csr.indptr), shape=csr.shape)

    def with_data(self, data: Union[jax.Array, u.Quantity]) -> 'CSR':
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return CSR((data, self.indices, self.indptr), shape=self.shape)

    def todense(self):
        return _csr_todense(self.data, self.indices, self.indptr, shape=self.shape)

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        assert axes is None, "transpose does not support axes argument."
        return CSC((self.data, self.indices, self.indptr), shape=self.shape[::-1])

    def __abs__(self):
        return CSR((abs(self.data), self.indices, self.indptr), shape=self.shape)

    def __neg__(self):
        return CSR((-self.data, self.indices, self.indptr), shape=self.shape)

    def __pos__(self):
        return CSR((self.data.__pos__(), self.indices, self.indptr), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    (op(self.data, other.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR(
                (op(self.data, other), self.indices, self.indptr),
                shape=self.shape
            )

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSR(
                    (op(other.data, self.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSR(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSR(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: Union[jax.Array, u.Quantity]) -> 'CSR':
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Union[jax.Array, u.Quantity]) -> 'CSR':
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: Union[jax.Array, u.Quantity]) -> 'CSR':
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: Union[jax.Array, u.Quantity]) -> 'CSR':
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> 'CSR':
        return self.__div__(other)

    def __rtruediv__(self, other) -> 'CSR':
        return self.__rdiv__(other)

    def __add__(self, other) -> 'CSR':
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> 'CSR':
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> 'CSR':
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> 'CSR':
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> 'CSR':
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> 'CSR':
        return self._binary_rop(other, operator.mod)

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
                    self.id,
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
                    self.id,
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

    def tree_flatten(self):
        return (self.data,), (self.indices, self.indptr, self.shape)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, = children
        indices, indptr, shape = aux_data
        return CSR([data, indices, indptr], shape=shape)


# TODO: docstring needed to be improved
@jax.tree_util.register_pytree_node_class
class CSC(u.sparse.SparseMatrix):
    """
    Event-driven and Unit-aware CSC matrix.

    """
    data: Union[jax.Array, u.Quantity]
    indices: jax.Array
    indptr: jax.Array
    id: jax.Array
    shape: tuple[int, int]
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        args: Sequence[Union[jax.Array, np.ndarray, u.Quantity]],
        *,
        shape: Tuple[int, int]
    ):
        self.data, self.indices, self.indptr = map(u.math.asarray, args)
        m = (self.indices.shape[0] - 1) // 1024 + 1
        self.id = jnp.zeros(m, self.indices.dtype)
        sum = 0
        k = 0

        def cond_fun(carry):
            k, sum, indptr = carry
            return indptr[k + 1] <= sum

        def body_fun(carry):
            k, sum, indptr = carry
            return k + 1, sum, indptr

        for i in range(1, m):
            sum += 1024
            k = jax.lax.while_loop(cond_fun, body_fun, (k, sum, self.indptr))[0]
            self.id.at[i].set(k)
        super().__init__(args, shape=shape)

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=jnp.int32) -> 'CSC':
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csc = u.sparse.csr_fromdense(mat.T, nse=nse, index_dtype=index_dtype).T
        return CSC((csc.data, csc.indices, csc.indptr), shape=csc.shape)

    @classmethod
    def _empty(cls, shape, *, dtype=None, index_dtype='int32'):
        """
        Create an empty CSC instance. Public method is sparse.empty().
        """
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError(f"CSC must have ndim=2; got {shape=}")
        data = jnp.empty(0, dtype)
        indices = jnp.empty(0, index_dtype)
        indptr = jnp.zeros(shape[1] + 1, index_dtype)
        return cls((data, indices, indptr), shape=shape)

    def with_data(self, data: Union[jax.Array, u.Quantity]) -> 'CSC':
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return CSC((data, self.indices, self.indptr), shape=self.shape)

    def todense(self):
        return self.T.todense().T

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        assert axes is None
        return CSR((self.data, self.indices, self.indptr), shape=self.shape[::-1])

    def __abs__(self):
        return CSC((abs(self.data), self.indices, self.indptr), shape=self.shape)

    def __neg__(self):
        return CSC((-self.data, self.indices, self.indptr), shape=self.shape)

    def __pos__(self):
        return CSC((self.data.__pos__(), self.indices, self.indptr), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(self.data, other.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(self.data, other),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return CSC(
                    (op(other.data, self.data),
                     self.indices,
                     self.indptr),
                    shape=self.shape
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return CSC(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return CSC(
                (op(other, self.data),
                 self.indices,
                 self.indptr),
                shape=self.shape
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: Union[jax.Array, u.Quantity]) -> 'CSC':
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Union[jax.Array, u.Quantity]) -> 'CSC':
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: Union[jax.Array, u.Quantity]) -> 'CSC':
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: Union[jax.Array, u.Quantity]) -> 'CSC':
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> 'CSC':
        return self.__div__(other)

    def __rtruediv__(self, other) -> 'CSC':
        return self.__rdiv__(other)

    def __add__(self, other) -> 'CSC':
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> 'CSC':
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> 'CSC':
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> 'CSC':
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> 'CSC':
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> 'CSC':
        return self._binary_rop(other, operator.mod)

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
                    self.id,
                    other,
                    shape=self.shape[::-1],
                    transpose=True
                )
            elif other.ndim == 2:
                return _csr_matmat(
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
                    self.indptr,
                    other,
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
                    self.id,
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
                    other,
                    shape=self.shape[::-1],
                    transpose=False
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def tree_flatten(self):
        return (self.data,), (self.indices, self.indptr, self.shape)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, = children
        indices, indptr, shape = aux_data
        return CSC([data, indices, indptr], shape=shape)
