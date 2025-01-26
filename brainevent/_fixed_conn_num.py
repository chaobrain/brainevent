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
from typing import Union, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
from brainunit.sparse._coo import _coo_todense, COOInfo
from jax.experimental.sparse import JAXSparse

from ._array import EventArray
from ._fixed_conn_num_event_impl import event_fixed_post_num_mv_p_call
from ._fixed_conn_num_float_impl import fixed_post_num_mv_p_call

__all__ = [
    'FixedPostNumConn',
    'FixedPreNumConn',
]


# TODO: docstring needed to be improved
@jax.tree_util.register_pytree_node_class
class FixedPostNumConn(u.sparse.SparseMatrix):
    """
    Fixed total number of postsynaptic neurons.
    """
    data: Union[jax.Array, u.Quantity]
    indices: jax.Array
    shape: tuple[int, int]
    num_pre = property(lambda self: self.indices.shape[0])
    num_conn = property(lambda self: self.indices.shape[1])
    num_post = property(lambda self: self.shape[1])
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(self, args, *, shape: Sequence[int]):
        self.data, self.indices = map(u.math.asarray, args)
        assert self.indices.shape[0] == shape[0], 'Pre-synaptic neuron number mismatch.'
        super().__init__(args, shape=shape)

    def with_data(self, data: Union[jax.Array, u.Quantity]) -> 'FixedPostNumConn':
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedPostNumConn((data, self.indices), shape=self.shape)

    def todense(self):
        """
        Convert the matrix to dense format.
        """
        pre_ids, post_ids, spinfo = fixed_post_num_to_coo(self)
        return _coo_todense(self.data, pre_ids, post_ids, spinfo=spinfo)

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None) -> 'FixedPreNumConn':
        """
        Transpose the matrix.
        """
        assert axes is None, "transpose does not support axes argument."
        return FixedPreNumConn(
            (self.data, self.indices),
            shape=self.shape[::-1],
        )

    def __abs__(self):
        return FixedPostNumConn((abs(self.data), self.indices), shape=self.shape)

    def __neg__(self):
        return FixedPostNumConn((-self.data, self.indices), shape=self.shape)

    def __pos__(self):
        return FixedPostNumConn((self.data.__pos__(), self.indices), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, FixedPostNumConn):
            if id(other.indices) == id(self.indices):
                return FixedPostNumConn(
                    (
                        op(self.data, other.data),
                        self.indices
                    ),
                    shape=self.shape,
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPostNumConn(
                (op(self.data, other), self.indices),
                shape=self.shape,
            )

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedPostNumConn(
                (op(self.data, other),
                 self.indices),
                shape=self.shape,
            )

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, FixedPostNumConn):
            if id(other.indices) == id(self.indices):
                return FixedPostNumConn(
                    (
                        op(other.data, self.data),
                        self.indices
                    ),
                    shape=self.shape,
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPostNumConn(
                (
                    op(other, self.data),
                    self.indices
                ),
                shape=self.shape,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedPostNumConn(
                (
                    op(other, self.data),
                    self.indices,
                ),
                shape=self.shape,
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: Union[jax.Array, u.Quantity]) -> 'FixedPostNumConn':
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Union[jax.Array, u.Quantity]) -> 'FixedPostNumConn':
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: Union[jax.Array, u.Quantity]) -> 'FixedPostNumConn':
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: Union[jax.Array, u.Quantity]) -> 'FixedPostNumConn':
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> 'FixedPostNumConn':
        return self.__div__(other)

    def __rtruediv__(self, other) -> 'FixedPostNumConn':
        return self.__rdiv__(other)

    def __add__(self, other) -> 'FixedPostNumConn':
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> 'FixedPostNumConn':
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> 'FixedPostNumConn':
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> 'FixedPostNumConn':
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> 'FixedPostNumConn':
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> 'FixedPostNumConn':
        return self._binary_rop(other, operator.mod)

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

    def tree_flatten(self):
        return (self.data,), (self.indices, self.shape)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, = children
        indices, shape = aux_data
        return FixedPostNumConn((data, indices), shape=shape)


# TODO: docstring needed to be improved
@jax.tree_util.register_pytree_node_class
class FixedPreNumConn(u.sparse.SparseMatrix):
    """
    Fixed total number of pre-synaptic neurons.
    """
    data: Union[jax.Array, u.Quantity]
    indices: jax.Array
    shape: tuple[int, int]
    num_conn = property(lambda self: self.indices.shape[1])
    num_post = property(lambda self: self.indices.shape[0])
    num_pre = property(lambda self: self.shape[0])
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(self, args, *, shape: Sequence[int]):
        self.data, self.indices = map(u.math.asarray, args)
        assert self.indices.shape[0] == shape[1], 'Post-synaptic neuron number mismatch.'
        super().__init__(args, shape=shape)

    def with_data(self, data: Union[jax.Array, u.Quantity]) -> 'FixedPreNumConn':
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedPreNumConn((data, self.indices), shape=self.shape)

    def todense(self):
        """
        Convert the matrix to dense format.
        """
        pre_ids, post_ids, spinfo = fixed_pre_num_to_coo(self)
        return _coo_todense(self.data, pre_ids, post_ids, spinfo=spinfo)

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None) -> 'FixedPostNumConn':
        """
        Transpose the matrix.
        """
        assert axes is None, "transpose does not support axes argument."
        return FixedPostNumConn(
            (self.data, self.indices),
            shape=self.shape[::-1],
        )

    def __abs__(self):
        return FixedPreNumConn((abs(self.data), self.indices), shape=self.shape)

    def __neg__(self):
        return FixedPreNumConn((-self.data, self.indices), shape=self.shape)

    def __pos__(self):
        return FixedPreNumConn((self.data.__pos__(), self.indices), shape=self.shape)

    def _binary_op(self, other, op):
        if isinstance(other, FixedPreNumConn):
            if id(other.indices) == id(self.indices):
                return FixedPreNumConn(
                    (
                        op(self.data, other.data),
                        self.indices
                    ),
                    shape=self.shape,
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPreNumConn(
                (op(self.data, other), self.indices),
                shape=self.shape,
            )

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedPreNumConn(
                (op(self.data, other),
                 self.indices),
                shape=self.shape,
            )

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, FixedPreNumConn):
            if id(other.indices) == id(self.indices):
                return FixedPreNumConn(
                    (
                        op(other.data, self.data),
                        self.indices
                    ),
                    shape=self.shape,
                )
        if isinstance(other, JAXSparse):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPreNumConn(
                (
                    op(other, self.data),
                    self.indices
                ),
                shape=self.shape,
            )
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedPreNumConn(
                (
                    op(other, self.data),
                    self.indices,
                ),
                shape=self.shape,
            )
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __mul__(self, other: Union[jax.Array, u.Quantity]) -> 'FixedPreNumConn':
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: Union[jax.Array, u.Quantity]) -> 'FixedPreNumConn':
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: Union[jax.Array, u.Quantity]) -> 'FixedPreNumConn':
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: Union[jax.Array, u.Quantity]) -> 'FixedPreNumConn':
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> 'FixedPreNumConn':
        return self.__div__(other)

    def __rtruediv__(self, other) -> 'FixedPreNumConn':
        return self.__rdiv__(other)

    def __add__(self, other) -> 'FixedPreNumConn':
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> 'FixedPreNumConn':
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> 'FixedPreNumConn':
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> 'FixedPreNumConn':
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> 'FixedPreNumConn':
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> 'FixedPreNumConn':
        return self._binary_rop(other, operator.mod)

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

    def tree_flatten(self):
        return (self.data,), (self.indices, self.shape)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        data, = children
        indices, shape = aux_data
        return FixedPreNumConn((data, indices), shape=shape)


def fixed_post_num_to_coo(self: FixedPostNumConn):
    pre_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    post_ids = self.indices.flatten()
    spinfo = COOInfo(self.shape, rows_sorted=True, cols_sorted=False)
    return pre_ids, post_ids, spinfo


def fixed_pre_num_to_coo(self: FixedPreNumConn):
    pre_ids = self.indices.flatten()
    post_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    spinfo = COOInfo(self.shape, rows_sorted=False, cols_sorted=True)
    return pre_ids, post_ids, spinfo
