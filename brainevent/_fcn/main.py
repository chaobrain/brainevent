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
from typing import Optional

import brainunit as u
import jax

from brainevent._compatible_import import Tracer
from brainevent._data import DataRepresentation
from brainevent._event.binary import BinaryArray
from brainevent._misc import _coo_todense, COOInfo
from brainevent._typing import Data, MatrixShape, Index
from brainevent.config import get_backend
from .binary import binary_fcnmv, binary_fcnmm, csc_binary_matvec, ell_binary_matvec_p
from .float import fcnmv, fcnmm
from .layouts import EllLayout, CscLayout, resolve_matvec

__all__ = [
    'FixedNumConn',
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
    if not jax.numpy.issubdtype(indices.dtype, jax.numpy.integer):
        raise ValueError(f'{kind} indices must be integer type, got {indices.dtype}.')


def _contains_invalid_indices(indices: Index, *, upper_bound: int) -> bool:
    import numpy as np
    with jax.ensure_compile_time_eval():
        indices_np = np.asarray(indices)
        if bool(np.any(indices_np < 0) or np.any(indices_np >= upper_bound)):
            raise ValueError(
                f'Found invalid indices in the connection matrix. '
                f'All indices must be in the range [0, {upper_bound - 1}]. '
                f'But found indices with min {indices_np.min()} and max {indices_np.max()}.'
            )


def _ensure_fixed_conn_initialized_outside_jit(indices: Index, *, kind: str) -> None:
    if isinstance(indices, Tracer):
        raise RuntimeError(
            f'{kind} must be first constructed outside `jax.jit` / '
            '`brainstate.transform.jit`. Initialization validates connectivity '
            'and may materialize a CSC layout mirror from concrete indices. '
            'Construct the connection object before entering the jitted function, '
            'then reuse it inside JIT.'
        )


class FixedNumConn(DataRepresentation):
    """
    Unified, layout-aware sparse matrix with a fixed number of connections.

    A ``FixedNumConn`` represents a single logical weight matrix ``W`` of shape
    ``(num_pre, num_post)`` stored row-major in fixed-connection ELL format
    (``data`` / ``indices``).  The concrete subclasses fix the orientation:

    * :class:`FixedPostNumConn` (``axis == 0``): each pre-synaptic neuron has a
      fixed number of outgoing connections; ``indices`` are post-synaptic ids.
    * :class:`FixedPreNumConn` (``axis == 1``): each post-synaptic neuron has a
      fixed number of incoming connections; ``indices`` are pre-synaptic ids.

    Matrix products are routed through a small capability-aware dispatcher
    (:func:`brainevent._fcn.layouts.resolve_matvec`).  When an event-driven
    ``W @ s`` matvec lands on a CUDA backend -- where the row-gather kernel does
    not exist -- the dispatcher transparently builds and caches a column-major
    CSC mirror and runs the column-scatter kernel instead.  The mirror is built
    lazily on first need from concrete indices, so it must be triggered outside
    ``jax.jit``.

    Parameters
    ----------
    data : Data
        Non-zero values of the sparse matrix.
    indices : Index
        Integer index array that describes the connectivity pattern.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.

    See Also
    --------
    FixedPostNumConn : Concrete subclass for fixed post-synaptic connections.
    FixedPreNumConn : Concrete subclass for fixed pre-synaptic connections.
    """
    data: Data
    indices: Index
    shape: MatrixShape
    backend: Optional[str]
    # ELL axis (0 == fixed-post / row-major W, 1 == fixed-pre / W^T). Set by subclasses.
    axis: int = 0

    def __init__(self, *args, shape: MatrixShape, backend: Optional[str] = None):
        self.backend = backend
        self._csc = None
        super().__init__(*args, shape=shape)

    # ------------------------------------------------------------------ #
    # Layout / dispatch helpers
    # ------------------------------------------------------------------ #

    def _backend_is_cuda(self) -> bool:
        if self.backend is not None:
            return self.backend == 'cuda_raw'
        platform = jax.default_backend()
        global_backend = get_backend(platform)
        if global_backend is not None:
            return global_backend == 'cuda_raw'
        return ell_binary_matvec_p.get_default(platform) == 'cuda_raw'

    def _ell_plan(self, transpose_W: bool):
        a_shape = tuple(self.shape) if self.axis == 0 else tuple(self.shape)[::-1]
        ell_transpose = bool(transpose_W) ^ (self.axis == 1)
        return a_shape, ell_transpose

    def _ensure_csc(self) -> CscLayout:
        if self._csc is None:
            _ensure_fixed_conn_initialized_outside_jit(self.indices, kind=type(self).__name__)
            self._csc = EllLayout(self.data, self.indices, axis=self.axis).to_csc(shape=self.shape)
        return self._csc

    def _binary_matvec(self, s, transpose_W: bool):
        plan = resolve_matvec(
            axis=self.axis,
            transpose_W=transpose_W,
            is_event=True,
            backend_is_cuda=self._backend_is_cuda(),
            shape=self.shape,
        )
        if plan.format == 'csc':
            csc = self._ensure_csc()
            return csc_binary_matvec(
                csc.weights, csc.indices, csc.indptr, s,
                shape=plan.a_shape, backend=self.backend,
            )
        return binary_fcnmv(
            self.data, self.indices, s,
            shape=plan.a_shape, transpose=plan.transpose, backend=self.backend,
        )

    def _binary_matmat(self, matrix, transpose_W: bool):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        return binary_fcnmm(
            self.data, self.indices, matrix,
            shape=a_shape, transpose=ell_transpose, backend=self.backend,
        )

    def _float_matvec(self, vector, transpose_W: bool, data):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        return fcnmv(data, self.indices, vector, shape=a_shape, transpose=ell_transpose)

    def _float_matmat(self, matrix, transpose_W: bool, data):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        return fcnmm(data, self.indices, matrix, shape=a_shape, transpose=ell_transpose)

    def _dispatch(self, other, transpose_W: bool):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        if isinstance(other, BinaryArray):
            value = other.value
            if value.ndim == 1:
                return self._binary_matvec(value, transpose_W)
            elif value.ndim == 2:
                if transpose_W:
                    return self._binary_matmat(value.T, transpose_W).T
                return self._binary_matmat(value, transpose_W)
            else:
                raise NotImplementedError(f"matmul with object of shape {value.shape}")

        other = u.math.asarray(other)
        data, other = u.math.promote_dtypes(self.data, other)
        if other.ndim == 1:
            return self._float_matvec(other, transpose_W, data)
        elif other.ndim == 2:
            if transpose_W:
                return self._float_matmat(other.T, transpose_W, data).T
            return self._float_matmat(other, transpose_W, data)
        else:
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __matmul__(self, other):
        """Matrix multiplication ``self @ other`` (logical ``W @ other``)."""
        return self._dispatch(other, transpose_W=False)

    def __rmatmul__(self, other):
        """Reflected matrix multiplication ``other @ self`` (logical ``W^T @ other``)."""
        return self._dispatch(other, transpose_W=True)

    # ------------------------------------------------------------------ #
    # Pytree protocol
    # ------------------------------------------------------------------ #

    def tree_flatten(self):
        """Flatten into ``((data, indices), aux)``; the CSC mirror is a rebuildable cache."""
        aux = {'shape': self.shape, 'backend': self.backend}
        return (self.data, self.indices), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from pytree components. The CSC mirror cache resets to lazy."""
        obj = object.__new__(cls)
        obj.data, obj.indices = children
        obj.shape = aux_data['shape']
        obj.backend = aux_data['backend']
        obj._csc = None
        return obj

    # ------------------------------------------------------------------ #
    # Element-wise operators
    # ------------------------------------------------------------------ #

    def _unitary_op(self, op):
        raise NotImplementedError

    def apply(self, fn):
        """Apply ``fn`` to the value buffer while keeping connectivity structure."""
        return self._unitary_op(fn)

    def __abs__(self):
        """Element-wise absolute value, preserving connectivity."""
        return self.apply(operator.abs)

    def __neg__(self):
        """Element-wise negation, preserving connectivity."""
        return self.apply(operator.neg)

    def __pos__(self):
        """Element-wise positive (identity), preserving connectivity."""
        return self.apply(operator.pos)

    def _binary_op(self, other, op):
        raise NotImplementedError

    def apply2(self, other, fn, *, reverse: bool = False):
        """Apply a binary function while preserving fixed-connectivity semantics."""
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
    Sparse matrix with a fixed number of post-synaptic connections per
    pre-synaptic neuron (row-major ELL, ``axis == 0``).

    ``data`` and ``indices`` have shape ``(num_pre, num_conn)``; the equivalent
    dense matrix is ``W[i, indices[i, k]] = data[i, k]``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import FixedPostNumConn
        >>>
        >>> data = jnp.array([[1., 2.], [3., 4.]])
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> mat = FixedPostNumConn(data, indices, shape=(2, 3))
        >>> mat.shape
        (2, 3)
    """
    __module__ = 'brainevent'

    data: Data
    indices: Index
    shape: MatrixShape
    axis = 0
    num_pre = property(lambda self: self.indices.shape[0])
    num_conn = property(lambda self: self.indices.shape[1])
    num_post = property(lambda self: self.shape[1])
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        data,
        indices=None,
        *,
        shape: MatrixShape,
        backend: Optional[str] = None,
    ):
        if indices is None:
            args = data
        else:
            args = (data, indices)
        self.data, self.indices = map(u.math.asarray, args)
        _ensure_fixed_conn_initialized_outside_jit(self.indices, kind='FixedPostNumConn')
        _validate_fixed_conn_indices(self.indices, expected_rows=shape[0], kind='Post-synaptic')
        if self.data.size != 1 and self.data.shape != self.indices.shape:
            raise ValueError(
                f"Data shape {self.data.shape} must match indices shape {self.indices.shape}. "
                f"But got {self.data.shape} != {self.indices.shape}"
            )
        super().__init__((self.data, self.indices), shape=shape, backend=backend)
        _contains_invalid_indices(self.indices, upper_bound=self.shape[1])

    def with_data(self, data: Data) -> 'FixedPostNumConn':
        """Return a new matrix with the same connectivity and replaced values."""
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedPostNumConn((data, self.indices), shape=self.shape, backend=self.backend)

    def todense(self):
        """Convert to a dense matrix of shape ``(num_pre, num_post)``."""
        pre_ids, post_ids, spinfo = fixed_post_num_to_coo(self)
        return _coo_todense(self.data.flatten(), pre_ids, post_ids, spinfo=spinfo)

    def transpose(self, axes=None) -> 'FixedPreNumConn':
        """Transpose to a :class:`FixedPreNumConn` (O(1); reinterprets indices)."""
        assert axes is None, "transpose does not support axes argument."
        return FixedPreNumConn(
            (self.data, self.indices),
            shape=self.shape[::-1],
            backend=self.backend,
        )

    def _unitary_op(self, op):
        return FixedPostNumConn((op(self.data), self.indices), shape=self.shape, backend=self.backend)

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPostNumConn((op(self.data, other), self.indices), shape=self.shape, backend=self.backend)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedPostNumConn((op(self.data, other), self.indices), shape=self.shape, backend=self.backend)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPostNumConn((op(other, self.data), self.indices), shape=self.shape, backend=self.backend)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedPostNumConn((op(other, self.data), self.indices), shape=self.shape, backend=self.backend)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")


@jax.tree_util.register_pytree_node_class
class FixedPreNumConn(FixedNumConn):
    """
    Sparse matrix with a fixed number of pre-synaptic connections per
    post-synaptic neuron (``axis == 1``; stores ``W^T`` row-major).

    ``data`` and ``indices`` have shape ``(num_post, num_conn)``; the equivalent
    dense matrix is ``W[indices[j, k], j] = data[j, k]``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import FixedPreNumConn
        >>>
        >>> data = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
        >>> indices = jnp.array([[0, 1], [1, 0], [0, 2]])
        >>> mat = FixedPreNumConn(data, indices, shape=(3, 3))
        >>> mat.shape
        (3, 3)
    """
    __module__ = 'brainevent'

    data: Data
    indices: Index
    shape: MatrixShape
    axis = 1
    num_conn = property(lambda self: self.indices.shape[1])
    num_post = property(lambda self: self.indices.shape[0])
    num_pre = property(lambda self: self.shape[0])
    nse = property(lambda self: self.indices.size)
    dtype = property(lambda self: self.data.dtype)

    def __init__(
        self,
        data,
        indices=None,
        *,
        shape: MatrixShape,
        backend: Optional[str] = None,
    ):
        if indices is None:
            args = data
        else:
            args = (data, indices)
        self.data, self.indices = map(u.math.asarray, args)
        _ensure_fixed_conn_initialized_outside_jit(self.indices, kind='FixedPreNumConn')
        _validate_fixed_conn_indices(self.indices, expected_rows=shape[1], kind='Pre-synaptic')
        if self.data.size != 1 and self.data.shape != self.indices.shape:
            raise ValueError(
                f"Data shape {self.data.shape} must match indices shape {self.indices.shape}. "
                f"But got {self.data.shape} != {self.indices.shape}"
            )
        super().__init__((self.data, self.indices), shape=shape, backend=backend)
        _contains_invalid_indices(self.indices, upper_bound=self.shape[0])

    def with_data(self, data: Data) -> 'FixedPreNumConn':
        """Return a new matrix with the same connectivity and replaced values."""
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedPreNumConn((data, self.indices), shape=self.shape, backend=self.backend)

    def todense(self):
        """Convert to a dense matrix of shape ``(num_pre, num_post)``."""
        pre_ids, post_ids, spinfo = fixed_pre_num_to_coo(self)
        return _coo_todense(self.data.flatten(), pre_ids, post_ids, spinfo=spinfo)

    def transpose(self, axes=None) -> FixedPostNumConn:
        """Transpose to a :class:`FixedPostNumConn` (O(1); reinterprets indices)."""
        assert axes is None, "transpose does not support axes argument."
        return FixedPostNumConn(
            (self.data, self.indices),
            shape=self.shape[::-1],
            backend=self.backend,
        )

    def _unitary_op(self, op):
        return FixedPreNumConn((op(self.data), self.indices), shape=self.shape, backend=self.backend)

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPreNumConn((op(self.data, other), self.indices), shape=self.shape, backend=self.backend)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedPreNumConn((op(self.data, other), self.indices), shape=self.shape, backend=self.backend)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedPreNumConn((op(other, self.data), self.indices), shape=self.shape, backend=self.backend)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedPreNumConn((op(other, self.data), self.indices), shape=self.shape, backend=self.backend)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")


def fixed_post_num_to_coo(self: FixedPostNumConn):
    """Convert a :class:`FixedPostNumConn` to ``(pre_ids, post_ids, COOInfo)``."""
    import jax.numpy as jnp
    pre_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    post_ids = self.indices.flatten()
    # Keep COO metadata unsorted to preserve duplicate accumulation semantics
    # in coo_todense on GPU (sorted paths may overwrite duplicates).
    spinfo = COOInfo(self.shape, rows_sorted=False, cols_sorted=False)
    return pre_ids, post_ids, spinfo


def fixed_pre_num_to_coo(self: FixedPreNumConn):
    """Convert a :class:`FixedPreNumConn` to ``(pre_ids, post_ids, COOInfo)``."""
    import jax.numpy as jnp
    pre_ids = self.indices.flatten()
    post_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    # Keep COO metadata unsorted to preserve duplicate accumulation semantics
    # in coo_todense on GPU (sorted paths may overwrite duplicates).
    spinfo = COOInfo(self.shape, rows_sorted=False, cols_sorted=False)
    return pre_ids, post_ids, spinfo
