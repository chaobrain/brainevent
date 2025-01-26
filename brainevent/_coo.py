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


from __future__ import annotations

import operator
from typing import Any, Tuple, Sequence, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainunit._base import Quantity, split_mantissa_unit, maybe_decimal, get_unit
from brainunit._sparse_base import SparseMatrix
from brainunit.math._fun_array_creation import asarray
from brainunit.math._fun_keep_unit import promote_dtypes
from jax import lax
from jax import tree_util
from jax._src.lax.lax import _const
from jax.experimental.sparse import JAXSparse

from ._array import EventArray
from ._coo_event_impl import _event_coo_matvec, _event_coo_matmat
from ._coo_float_impl import _coo_matvec, _coo_matmat

__all__ = [
    'COO', 'COOInfo',
]

Dtype = Any
Shape = tuple[int, ...]


class COOInfo(NamedTuple):
    shape: Shape
    rows_sorted: bool = False
    cols_sorted: bool = False


@tree_util.register_pytree_node_class
class COO(SparseMatrix):
    """Experimental COO matrix implemented in JAX.

    Note: this class has minimal compatibility with JAX transforms such as
    grad and autodiff, and offers very little functionality. In general you
    should prefer :class:`jax.experimental.sparse.BCOO`.

    Additionally, there are known failures in the case that `nse` is larger
    than the true number of nonzeros in the represented matrix. This situation
    is better handled in BCOO.
    """
    data: jax.Array
    row: jax.Array
    col: jax.Array
    shape: tuple[int, int]
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
        args: Tuple[jax.Array | Quantity, jax.Array, jax.Array],
        *,
        shape: Shape,
        rows_sorted: bool = False,
        cols_sorted: bool = False
    ):
        self.data, self.row, self.col = map(asarray, args)
        self._rows_sorted = rows_sorted
        self._cols_sorted = cols_sorted
        super().__init__(args, shape=shape)

    @classmethod
    def fromdense(
        cls,
        mat: jax.Array,
        *,
        nse: int | None = None,
        index_dtype: jax.typing.DTypeLike = np.int32
    ) -> COO:
        if nse is None:
            nse = (u.get_mantissa(mat) != 0.).sum()
        coo = u.sparse.coo_fromdense(mat, nse=nse, index_dtype=index_dtype)
        return COO(coo.data, coo.row, coo.col, shape=coo.shape)

    def _sort_indices(self) -> COO:
        """Return a copy of the COO matrix with sorted indices.

        The matrix is sorted by row indices and column indices per row.
        If self._rows_sorted is True, this returns ``self`` without a copy.
        """
        # TODO(jakevdp): would be benefit from lowering this to cusparse sort_rows utility?
        if self._rows_sorted:
            return self
        data, unit = split_mantissa_unit(self.data)
        row, col, data = lax.sort((self.row, self.col, data), num_keys=2)
        return self.__class__(
            (
                maybe_decimal(Quantity(data, unit=unit)),
                row,
                col
            ),
            shape=self.shape,
            rows_sorted=True
        )

    @classmethod
    def _empty(
        cls,
        shape: Sequence[int],
        *,
        dtype: jax.typing.DTypeLike | None = None,
        index_dtype: jax.typing.DTypeLike = 'int32'
    ) -> COO:
        """Create an empty COO instance. Public method is sparse.empty()."""
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError(f"COO must have ndim=2; got {shape=}")
        data = jnp.empty(0, dtype)
        row = col = jnp.empty(0, index_dtype)
        return cls(
            (data, row, col),
            shape=shape,
            rows_sorted=True,
            cols_sorted=True
        )

    @classmethod
    def _eye(
        cls,
        N: int,
        M: int,
        k: int,
        *,
        dtype: jax.typing.DTypeLike | None = None,
        index_dtype: jax.typing.DTypeLike = 'int32'
    ) -> COO:
        if k > 0:
            diag_size = min(N, M - k)
        else:
            diag_size = min(N + k, M)

        if diag_size <= 0:
            # if k is out of range, return an empty matrix.
            return cls._empty((N, M), dtype=dtype, index_dtype=index_dtype)

        data = jnp.ones(diag_size, dtype=dtype)
        idx = jnp.arange(diag_size, dtype=index_dtype)
        zero = _const(idx, 0)
        k = _const(idx, k)
        row = lax.sub(idx, lax.cond(k >= 0, lambda: zero, lambda: k))
        col = lax.add(idx, lax.cond(k <= 0, lambda: zero, lambda: k))
        return cls(
            (data, row, col),
            shape=(N, M),
            rows_sorted=True,
            cols_sorted=True
        )

    def with_data(self, data: jax.Array | Quantity) -> COO:
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert get_unit(data) == get_unit(self.data)
        return COO((data, self.row, self.col), shape=self.shape)

    def todense(self) -> jax.Array:
        return coo_todense(self)

    def transpose(self, axes: Tuple[int, ...] | None = None) -> COO:
        if axes is not None:
            raise NotImplementedError("axes argument to transpose()")
        return COO(
            (self.data, self.col, self.row),
            shape=self.shape[::-1],
            rows_sorted=self._cols_sorted,
            cols_sorted=self._rows_sorted
        )

    def tree_flatten(self) -> Tuple[
        Tuple[jax.Array | Quantity,], dict[str, Any]
    ]:
        aux = self._info._asdict()
        aux['row'] = self.row
        aux['col'] = self.col
        return (self.data,), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
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

    def __abs__(self):
        return COO(
            (self.data.__abs__(), self.row, self.col),
            shape=self.shape,
            rows_sorted=self._rows_sorted,
            cols_sorted=self._cols_sorted
        )

    def __neg__(self):
        return COO(
            (-self.data, self.row, self.col),
            shape=self.shape,
            rows_sorted=self._rows_sorted,
            cols_sorted=self._cols_sorted
        )

    def __pos__(self):
        return COO(
            (self.data.__pos__(), self.row, self.col),
            shape=self.shape,
            rows_sorted=self._rows_sorted,
            cols_sorted=self._cols_sorted
        )

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

        other = asarray(other)
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

        other = asarray(other)
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

    def __mul__(self, other: jax.Array | Quantity) -> COO:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: jax.Array | Quantity) -> COO:
        return self._binary_rop(other, operator.mul)

    def __div__(self, other: jax.Array | Quantity) -> COO:
        return self._binary_op(other, operator.truediv)

    def __rdiv__(self, other: jax.Array | Quantity) -> COO:
        return self._binary_rop(other, operator.truediv)

    def __truediv__(self, other) -> COO:
        return self.__div__(other)

    def __rtruediv__(self, other) -> COO:
        return self.__rdiv__(other)

    def __add__(self, other) -> COO:
        return self._binary_op(other, operator.add)

    def __radd__(self, other) -> COO:
        return self._binary_rop(other, operator.add)

    def __sub__(self, other) -> COO:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other) -> COO:
        return self._binary_rop(other, operator.sub)

    def __mod__(self, other) -> COO:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other) -> COO:
        return self._binary_rop(other, operator.mod)

    def __matmul__(
        self, other: jax.typing.ArrayLike
    ) -> jax.Array | Quantity:
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = asarray(other)
        data, other = promote_dtypes(self.data, other)
        self_promoted = COO(
            (
                data,
                self.row,
                self.col
            ),
            **self._info._asdict()
        )
        if other.ndim == 1:
            return coo_matvec(self_promoted, other)
        elif other.ndim == 2:
            return coo_matmat(self_promoted, other)
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(
        self,
        other: jax.typing.ArrayLike
    ) -> jax.Array | Quantity:
        if isinstance(other, JAXSparse):
            raise NotImplementedError("matmul between two sparse objects.")
        other = asarray(other)
        data, other = promote_dtypes(self.data, other)
        self_promoted = COO(
            (
                data,
                self.row,
                self.col
            ),
            **self._info._asdict()
        )
        if other.ndim == 1:
            return coo_matvec(self_promoted, other, transpose=True)
        elif other.ndim == 2:
            other = other.T
            return coo_matmat(self_promoted, other, transpose=True).T
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")
