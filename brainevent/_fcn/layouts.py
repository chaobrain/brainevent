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

import dataclasses
from typing import Tuple

import brainunit as u
import jax
import jax.numpy as jnp

from brainevent._misc import (
    COOInfo,
    _coo_todense,
    fixed_conn_num_to_csc,
    fixed_conn_num_csc_structure,
)

__all__ = [
    'EllLayout',
    'CscLayout',
    'CsrLayout',
    'ExecPlan',
    'resolve_matvec',
    'resolve_matmat',
]


@jax.tree_util.register_pytree_node_class
class EllLayout:
    """Fixed-connection ELL storage of a matrix ``A`` (``data`` / ``indices`` 2-D).

    ``axis == 0``: ``A == W`` (rows index pre-neurons, ``indices`` are post ids).
    ``axis == 1``: ``A == W^T`` (rows index post-neurons, ``indices`` are pre ids).
    The two are the same arrays under transpose, which is why a transpose only
    flips ``axis``.
    """

    def __init__(self, data, indices, axis: int):
        assert axis in (0, 1), f'axis must be 0 or 1, got {axis}.'
        self.data = u.math.asarray(data)
        self.indices = u.math.asarray(indices)
        self.axis = int(axis)

    @property
    def is_homogeneous(self) -> bool:
        return self.data.size == 1

    @property
    def num_conn(self) -> int:
        return self.indices.shape[1]

    def a_shape(self, *, shape: Tuple[int, int]) -> Tuple[int, int]:
        """Shape of the matrix ``A`` this ELL describes (``W`` or ``W^T``)."""
        return tuple(shape) if self.axis == 0 else tuple(shape)[::-1]

    def todense(self, *, shape: Tuple[int, int]):
        """Dense matrix in ``W``'s frame, shape ``(n_pre, n_post)``."""
        a_rows, a_cols = self.a_shape(shape=shape)
        row_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
        col_ids = self.indices.flatten()
        values, unit = u.split_mantissa_unit(self.data)
        if values.size == 1:
            flat = jnp.broadcast_to(values.reshape(()), (row_ids.size,))
        else:
            flat = values.reshape(-1)
        spinfo = COOInfo((a_rows, a_cols), rows_sorted=False, cols_sorted=False)
        dense = _coo_todense(flat, row_ids, col_ids, spinfo=spinfo)
        dense = dense if self.axis == 0 else dense.T
        return u.maybe_decimal(dense * unit)

    def to_csc(self, *, shape: Tuple[int, int]) -> 'CscLayout':
        """Build the CSC layout of ``A`` (capturing the ELL->CSC permutation)."""
        a_rows, a_cols = self.a_shape(shape=shape)
        values, unit = u.split_mantissa_unit(self.data)
        col_weights, csc_indices, csc_indptr = fixed_conn_num_to_csc(
            values, self.indices, shape=(a_rows, a_cols),
        )
        _, _, perm = fixed_conn_num_csc_structure(self.indices, shape=(a_rows, a_cols))
        col_weights = u.maybe_decimal(u.math.asarray(col_weights) * unit)
        return CscLayout(col_weights, csc_indices, csc_indptr, perm)

    def tree_flatten(self):
        return (self.data, self.indices), {'axis': self.axis}

    @classmethod
    def tree_unflatten(cls, aux, children):
        data, indices = children
        return cls(data, indices, axis=aux['axis'])


@jax.tree_util.register_pytree_node_class
class CscLayout:
    """Compressed-sparse-column storage of the primary ELL's matrix ``A``.

    ``perm`` maps the ELL flatten order to CSC order so values refresh without
    re-sorting. Structural arrays (``indices``, ``indptr``, ``perm``) are static
    integer leaves; only ``weights`` is differentiable.
    """

    def __init__(self, weights, indices, indptr, perm):
        self.weights = u.math.asarray(weights)
        self.indices = u.math.asarray(indices)
        self.indptr = u.math.asarray(indptr)
        self.perm = u.math.asarray(perm)

    @property
    def is_homogeneous(self) -> bool:
        return self.weights.size == 1

    def with_values_from_ell(self, ell_data) -> 'CscLayout':
        """Refresh weights from a (possibly new) ELL data array via ``perm``."""
        values, unit = u.split_mantissa_unit(ell_data)
        if values.size == 1:
            new_weights = u.maybe_decimal(values.reshape(1) * unit)
        else:
            new_weights = u.maybe_decimal(values.reshape(-1)[self.perm] * unit)
        return CscLayout(new_weights, self.indices, self.indptr, self.perm)

    def todense(self, *, shape: Tuple[int, int]):
        """Dense matrix of ``A``; ``shape`` is the A-shape ``(a_rows, a_cols)``."""
        a_rows, a_cols = shape
        col_ids = jnp.repeat(
            jnp.arange(self.indptr.shape[0] - 1),
            jnp.diff(self.indptr),
            total_repeat_length=self.indices.shape[0],
        )
        row_ids = self.indices
        weights, unit = u.split_mantissa_unit(self.weights)
        if weights.size == 1:
            flat = jnp.broadcast_to(weights.reshape(()), (row_ids.size,))
        else:
            flat = weights.reshape(-1)
        # Unsorted metadata forces the accumulate path so duplicate (row, col)
        # edges sum, matching the column-scatter kernel and EllLayout.todense.
        spinfo = COOInfo((a_rows, a_cols), rows_sorted=False, cols_sorted=False)
        dense = _coo_todense(flat, row_ids, col_ids, spinfo=spinfo)
        return u.maybe_decimal(dense * unit)

    def to_ell(self, *, a_shape, num_conn, axis) -> 'EllLayout':
        """Rebuild a fixed-connection ELL from CSC.

        Assumes each row of ``A`` has at most ``num_conn`` nonzeros, which holds
        for fixed-connection matrices. Only used for col-primary fallback paths.
        """
        a_rows, a_cols = a_shape
        dense = self.todense(shape=(a_rows, a_cols))
        dense_v, unit = u.split_mantissa_unit(dense)
        mask = dense_v != 0
        order = jnp.argsort(~mask, axis=1, stable=True)[:, :num_conn]
        new_indices = order.astype(self.indices.dtype)
        gathered = jnp.take_along_axis(dense_v, order, axis=1)
        new_data = u.maybe_decimal(gathered * unit)
        return EllLayout(new_data, new_indices, axis=axis)

    def tree_flatten(self):
        return (self.weights, self.indices, self.indptr, self.perm), {}

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


class CsrLayout:
    """Reserved peer of :class:`CscLayout`. Not implemented yet.

    A row-variable CSR layout would complete full row/column symmetry, but no
    kernel currently consumes it, so construction raises.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'CsrLayout is reserved for a future row-variable kernel.'
        )


@dataclasses.dataclass(frozen=True)
class ExecPlan:
    """Resolved execution choice for one matmul: which layout + how to call it."""
    format: str          # 'ell' or 'csc'
    transpose: bool      # ELL primitive transpose flag (ignored for csc)
    a_shape: Tuple[int, int]


def _a_shape(axis: int, shape: Tuple[int, int]) -> Tuple[int, int]:
    return tuple(shape) if axis == 0 else tuple(shape)[::-1]


def resolve_matvec(
    *,
    axis: int,
    transpose_W: bool,
    is_event: bool,
    backend_is_cuda: bool,
    shape: Tuple[int, int],
) -> ExecPlan:
    """Pick a layout + transpose flag for a logical matvec.

    ``transpose_W == False`` means ``W @ x``; ``True`` means ``W^T @ x``.
    The ELL primitive transpose flag is ``transpose_W XOR (axis == 1)``; a value
    of ``False`` is a gather (no CUDA kernel) so on CUDA event matvecs fall back
    to the CSC column-scatter.
    """
    a_shape = _a_shape(axis, shape)
    ell_transpose = bool(transpose_W) ^ (axis == 1)
    need_csc = is_event and (not ell_transpose) and backend_is_cuda
    if need_csc:
        return ExecPlan(format='csc', transpose=False, a_shape=a_shape)
    return ExecPlan(format='ell', transpose=ell_transpose, a_shape=a_shape)


def resolve_matmat(
    *,
    axis: int,
    transpose_W: bool,
    shape: Tuple[int, int],
) -> ExecPlan:
    """Pick a layout for a logical matmat. ``mm`` has no CSC kernel: always ELL."""
    a_shape = _a_shape(axis, shape)
    ell_transpose = bool(transpose_W) ^ (axis == 1)
    return ExecPlan(format='ell', transpose=ell_transpose, a_shape=a_shape)
