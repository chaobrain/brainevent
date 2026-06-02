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
from brainevent._misc import _coo_todense, COOInfo, coo2csr, fixed_conn_num_csc_structure
from brainevent._typing import Data, MatrixShape, Index
from .binary import binary_fcnmv, binary_fcnmm, csc_binary_matvec, csc_binary_matmat
from .float import fcnmv, fcnmm
from .yw2y import fcnmv_yw2y
from .plasticity_binary import (
    update_fixed_post_conn_on_binary_pre,
    update_fixed_post_conn_on_binary_post,
    update_fixed_pre_conn_on_binary_pre,
    update_fixed_pre_conn_on_binary_post,
)

__all__ = [
    'FixedNumConn',
    'FixedNumPerPre',
    'FixedNumPerPost',
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

    * :class:`FixedNumPerPre` (``axis == 0``): each pre-synaptic neuron has a
      fixed number of outgoing connections; ``indices`` are post-synaptic ids.
    * :class:`FixedNumPerPost` (``axis == 1``): each post-synaptic neuron has a
      fixed number of incoming connections; ``indices`` are pre-synaptic ids.

    Event-driven matrix-vector products follow the same favorable/unfavorable
    dispatch as :class:`brainevent.CSR` / :class:`brainevent.CSC`.  When the
    event vector indexes the ELL *stored* axis the product is a direct
    column-scatter (:func:`brainevent._fcn.binary.binary_fcnmv` with
    ``transpose=True``).  Otherwise the product would require a gather over every
    stored synapse; instead the structure is converted once to a column-major
    (CSC) view -- ``(indptr, indices, perm)`` built by
    :func:`brainevent._misc.fixed_conn_num_csc_structure` -- and the weights are
    permuted into CSC order so the same scatter kernel
    (:func:`brainevent._fcn.binary.csc_binary_matvec`) applies on every backend.
    The CSC view is built lazily on first need from concrete indices and cached
    in ``self._csc``, so it must be triggered outside ``jax.jit``.

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
    FixedNumPerPre : Concrete subclass for fixed post-synaptic connections.
    FixedNumPerPost : Concrete subclass for fixed pre-synaptic connections.
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

    def _ell_plan(self, transpose_W: bool):
        a_shape = tuple(self.shape) if self.axis == 0 else tuple(self.shape)[::-1]
        ell_transpose = bool(transpose_W) ^ (self.axis == 1)
        return a_shape, ell_transpose

    def _weight_indices(self):
        """Lazily build and cache the column-major (CSC) view of the structure.

        Returns the triple ``(csc_indptr, csc_indices, perm)`` for the ELL
        operand matrix (shape ``a_shape``), where ``perm`` maps each CSC slot to
        the position of its weight in the flattened ELL ``data``.  Built from
        concrete indices via :func:`fixed_conn_num_csc_structure`, so it must be
        triggered outside ``jax.jit``; the result is cached in ``self._csc``.
        """
        if self._csc is None:
            _ensure_fixed_conn_initialized_outside_jit(self.indices, kind=type(self).__name__)
            a_shape = tuple(self.shape) if self.axis == 0 else tuple(self.shape)[::-1]
            with jax.ensure_compile_time_eval():
                self._csc = fixed_conn_num_csc_structure(self.indices, shape=a_shape)
        return self._csc

    def _binary_matvec(self, s, transpose_W: bool):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        if ell_transpose:
            # Favorable: the event vector indexes the ELL stored axis -> direct
            # column-scatter over active events.
            return binary_fcnmv(
                self.data, self.indices, s,
                shape=a_shape, transpose=True, backend=self.backend,
            )
        # Unfavorable: convert to a column-major (CSC) view and permute weights
        # into CSC order so the scatter kernel applies on every backend.
        csc_indptr, csc_indices, perm = self._weight_indices()
        data = self.data
        weights = data.reshape(1) if data.size == 1 else data.reshape(-1)[perm]
        return csc_binary_matvec(
            weights, csc_indices, csc_indptr, s,
            shape=a_shape, backend=self.backend,
        )

    def _binary_matmat(self, matrix, transpose_W: bool):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        if ell_transpose:
            # Favorable: the event matrix indexes the ELL stored axis -> direct
            # column-scatter over active events.
            return binary_fcnmm(
                self.data, self.indices, matrix,
                shape=a_shape, transpose=True, backend=self.backend,
            )
        # Unfavorable: convert to a column-major (CSC) view and permute weights
        # into CSC order so the scatter matmat kernel applies on every backend --
        # parity with the matvec unfavorable path.
        csc_indptr, csc_indices, perm = self._weight_indices()
        data = self.data
        weights = data.reshape(1) if data.size == 1 else data.reshape(-1)[perm]
        return csc_binary_matmat(
            weights, csc_indices, csc_indptr, matrix,
            shape=a_shape, backend=self.backend,
        )

    def _float_matvec(self, vector, transpose_W: bool, data):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        return fcnmv(data, self.indices, vector, shape=a_shape, transpose=ell_transpose)

    def _float_matmat(self, matrix, transpose_W: bool, data):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        return fcnmm(data, self.indices, matrix, shape=a_shape, transpose=ell_transpose)

    # ------------------------------------------------------------------ #
    # Per-synapse y * w product (yw2y), parity with CSR / CSC
    # ------------------------------------------------------------------ #

    def yw_to_w(self, y_dim_arr, w_dim_arr=None):
        """Per-synapse ``w * y`` with ``y`` indexed by the row (pre) of ``W``.

        For every stored connection, returns ``w * y[row]`` where ``row`` is the
        pre-synaptic index of that connection, regardless of storage axis.  This
        is the fixed-connection analog of :meth:`brainevent.CSR.yw_to_w` and
        implements the ``yw_to_w`` protocol of :class:`brainunit.sparse.SparseMatrix`.

        Parameters
        ----------
        y_dim_arr : jax.Array or brainunit.Quantity
            Pre-synaptic (row) vector, sized ``shape[0]``.
        w_dim_arr : jax.Array or brainunit.Quantity, optional
            Per-synapse weights of shape ``indices.shape`` (or size-1).  Defaults
            to ``self.data``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result of shape ``self.indices.shape``.

        See Also
        --------
        yw_to_w_transposed : ``y`` indexed by the column (post) of ``W``.
        """
        w = self.data if w_dim_arr is None else w_dim_arr
        a_shape = tuple(self.shape) if self.axis == 0 else tuple(self.shape)[::-1]
        return fcnmv_yw2y(w, self.indices, y_dim_arr, shape=a_shape, transpose=(self.axis == 1))

    def yw_to_w_transposed(self, y_dim_arr, w_dim_arr=None):
        """Per-synapse ``w * y`` with ``y`` indexed by the column (post) of ``W``.

        Adjoint counterpart of :meth:`yw_to_w`: for every stored connection,
        returns ``w * y[col]`` where ``col`` is the post-synaptic index of that
        connection, regardless of storage axis.

        Parameters
        ----------
        y_dim_arr : jax.Array or brainunit.Quantity
            Post-synaptic (column) vector, sized ``shape[1]``.
        w_dim_arr : jax.Array or brainunit.Quantity, optional
            Per-synapse weights of shape ``indices.shape`` (or size-1).  Defaults
            to ``self.data``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result of shape ``self.indices.shape``.

        See Also
        --------
        yw_to_w : ``y`` indexed by the row (pre) of ``W``.
        """
        w = self.data if w_dim_arr is None else w_dim_arr
        a_shape = tuple(self.shape) if self.axis == 0 else tuple(self.shape)[::-1]
        return fcnmv_yw2y(w, self.indices, y_dim_arr, shape=a_shape, transpose=(self.axis == 0))

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
    # Format conversions
    # ------------------------------------------------------------------ #

    def _to_coo(self):
        """Return ``(pre_ids, post_ids, COOInfo)`` for the logical matrix ``W``.

        Dispatches on :attr:`axis` so both orientations describe the *same*
        logical matrix of shape ``(num_pre, num_post)``: ``pre_ids`` are row
        (pre-synaptic) indices and ``post_ids`` are column (post-synaptic)
        indices, each of length ``self.indices.size``.
        """
        if self.axis == 0:
            return fixed_post_num_to_coo(self)
        return fixed_pre_num_to_coo(self)

    def to_dense(self):
        """Convert to a dense ``(num_pre, num_post)`` matrix.

        Alias of :meth:`todense` provided for naming parity with
        :meth:`to_csr` and :meth:`to_csc`.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Dense matrix of shape ``self.shape``.  Stored entries that share a
            ``(row, column)`` coordinate (duplicate connections) are summed,
            matching :meth:`todense` and the event-driven matmul kernels.

        See Also
        --------
        todense : Underlying implementation.
        to_csr : Convert to Compressed Sparse Row format.
        to_csc : Convert to Compressed Sparse Column format.
        """
        return self.todense()

    def to_csr(self):
        """Convert to a :class:`brainevent.CSR` matrix of the logical matrix ``W``.

        The result is a Compressed Sparse Row view of the same logical weight
        matrix ``W`` of shape ``(num_pre, num_post)``, irrespective of the
        storage orientation (:class:`FixedNumPerPre` or
        :class:`FixedNumPerPost`).  Duplicate connections are preserved as
        repeated entries within a row (CSR matmul / :meth:`CSR.todense` sum
        them, matching :meth:`todense`).  Homogeneous (size-1) weights are kept
        as a single shared value.

        Returns
        -------
        CSR
            Equivalent matrix in CSR format with the same ``shape``, ``dtype``,
            unit, and ``backend`` as ``self``.

        Notes
        -----
        Building the CSR layout reorders the stored connections by row, which
        requires concrete indices.  Like the lazy CSC mirror, it must therefore
        run outside ``jax.jit`` / ``brainstate.transform.jit``; construct the
        connection and call this method before entering a jitted function.

        See Also
        --------
        to_csc : Convert to Compressed Sparse Column format.
        to_dense : Convert to a dense matrix.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import FixedNumPerPre
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.]])
            >>> indices = jnp.array([[0, 1], [1, 2]])
            >>> mat = FixedNumPerPre((data, indices), shape=(2, 3))
            >>> csr = mat.to_csr()
            >>> bool((csr.todense() == mat.todense()).all())
            True
        """
        from brainevent._csr import CSR  # local import avoids an import cycle

        _ensure_fixed_conn_initialized_outside_jit(self.indices, kind=type(self).__name__)
        pre_ids, post_ids, _ = self._to_coo()
        indptr, indices, order = coo2csr(pre_ids, post_ids, shape=self.shape)
        if self.data.size == 1:
            # Homogeneous weight: keep the single shared value.
            data = self.data.reshape(1)
        else:
            data = self.data.reshape(-1)[order]
        return CSR((data, indices, indptr), shape=self.shape, backend=self.backend)

    def to_csc(self):
        """Convert to a :class:`brainevent.CSC` matrix of the logical matrix ``W``.

        The result is a Compressed Sparse Column view of the same logical
        weight matrix ``W`` of shape ``(num_pre, num_post)``.  It is built by
        transposing to the opposite orientation, taking the CSR view of ``W^T``,
        and reinterpreting it as the (array-identical) CSC view of ``W`` -- so
        the same outside-``jit`` requirement as :meth:`to_csr` applies.

        Returns
        -------
        CSC
            Equivalent matrix in CSC format with the same ``shape``, ``dtype``,
            unit, and ``backend`` as ``self``.

        See Also
        --------
        to_csr : Convert to Compressed Sparse Row format.
        to_dense : Convert to a dense matrix.
        """
        return self.transpose().to_csr().transpose()

    # ------------------------------------------------------------------ #
    # Event-driven plasticity (STDP) updates
    # ------------------------------------------------------------------ #

    def _rebuild_with_data(self, new_data):
        """Structure-preserving rebuild that is safe under ``jax.jit``.

        Plasticity updates run inside the jitted simulation loop, so the result
        matrix must be reconstructible from traced arrays.  Unlike
        :meth:`with_data`, this bypasses the constructor's outside-``jit``
        validation (the connectivity is unchanged and was validated when ``self``
        was first built) by going through the registered pytree
        :meth:`tree_unflatten` path.
        """
        return type(self).tree_unflatten(
            {'shape': self.shape, 'backend': self.backend},
            (new_data, self.indices),
        )

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None):
        """Apply a pre-spike-triggered STDP update, returning a new matrix.

        For each firing pre neuron ``i`` every stored synapse is updated
        ``W[i, j] <- clip(W[i, j] + post_trace[j], w_min, w_max)``.  Favorable
        (row-driven) for :class:`FixedNumPerPre`, unfavorable (column-scan) for
        :class:`FixedNumPerPost`; dispatched on ``self.axis``.

        Parameters
        ----------
        pre_spike : jax.Array
            Pre-synaptic spikes, shape ``(shape[0],)``.
        post_trace : jax.Array or Quantity
            Post-synaptic trace, shape ``(shape[1],)``.
        w_min, w_max : jax.Array, Quantity, number, or None, optional
            Clip bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        FixedNumConn
            A new matrix of the same subclass with updated data and identical structure.

        See Also
        --------
        update_on_post : Post-spike-triggered counterpart.
        """
        if self.axis == 0:
            new = update_fixed_post_conn_on_binary_pre(
                self.data, self.indices, pre_spike, post_trace, w_min, w_max,
                shape=self.shape, backend=self.backend,
            )
        else:
            new = update_fixed_pre_conn_on_binary_pre(
                self.data, self.indices, pre_spike, post_trace, w_min, w_max,
                shape=self.shape, backend=self.backend,
            )
        return self._rebuild_with_data(new)

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None):
        """Apply a post-spike-triggered STDP update, returning a new matrix.

        For each firing post neuron ``j`` every stored synapse is updated
        ``W[i, j] <- clip(W[i, j] + pre_trace[i], w_min, w_max)``.  Unfavorable
        (column-scan) for :class:`FixedNumPerPre`, favorable (row-driven) for
        :class:`FixedNumPerPost`; dispatched on ``self.axis``.

        Parameters
        ----------
        pre_trace : jax.Array or Quantity
            Pre-synaptic trace, shape ``(shape[0],)``.
        post_spike : jax.Array
            Post-synaptic spikes, shape ``(shape[1],)``.
        w_min, w_max : jax.Array, Quantity, number, or None, optional
            Clip bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        FixedNumConn
            A new matrix of the same subclass with updated data and identical structure.

        See Also
        --------
        update_on_pre : Pre-spike-triggered counterpart.
        """
        if self.axis == 0:
            new = update_fixed_post_conn_on_binary_post(
                self.data, self.indices, pre_trace, post_spike, w_min, w_max,
                shape=self.shape, backend=self.backend,
            )
        else:
            new = update_fixed_pre_conn_on_binary_post(
                self.data, self.indices, pre_trace, post_spike, w_min, w_max,
                shape=self.shape, backend=self.backend,
            )
        return self._rebuild_with_data(new)

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
class FixedNumPerPre(FixedNumConn):
    """
    Sparse matrix with a fixed number of post-synaptic connections per
    pre-synaptic neuron (row-major ELL, ``axis == 0``).

    ``data`` and ``indices`` have shape ``(num_pre, num_conn)``; the equivalent
    dense matrix is ``W[i, indices[i, k]] = data[i, k]``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import FixedNumPerPre
        >>>
        >>> data = jnp.array([[1., 2.], [3., 4.]])
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> mat = FixedNumPerPre(data, indices, shape=(2, 3))
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
        _ensure_fixed_conn_initialized_outside_jit(self.indices, kind='FixedNumPerPre')
        _validate_fixed_conn_indices(self.indices, expected_rows=shape[0], kind='Post-synaptic')
        if self.data.size != 1 and self.data.shape != self.indices.shape:
            raise ValueError(
                f"Data shape {self.data.shape} must match indices shape {self.indices.shape}. "
                f"But got {self.data.shape} != {self.indices.shape}"
            )
        super().__init__((self.data, self.indices), shape=shape, backend=backend)
        _contains_invalid_indices(self.indices, upper_bound=self.shape[1])

    def with_data(self, data: Data) -> 'FixedNumPerPre':
        """Return a new matrix with the same connectivity and replaced values."""
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedNumPerPre((data, self.indices), shape=self.shape, backend=self.backend)

    def todense(self):
        """Convert to a dense matrix of shape ``(num_pre, num_post)``."""
        pre_ids, post_ids, spinfo = fixed_post_num_to_coo(self)
        return _coo_todense(self.data.flatten(), pre_ids, post_ids, spinfo=spinfo)

    def transpose(self, axes=None) -> 'FixedNumPerPost':
        """Transpose to a :class:`FixedNumPerPost` (O(1); reinterprets indices)."""
        assert axes is None, "transpose does not support axes argument."
        return FixedNumPerPost(
            (self.data, self.indices),
            shape=self.shape[::-1],
            backend=self.backend,
        )

    def _unitary_op(self, op):
        return FixedNumPerPre((op(self.data), self.indices), shape=self.shape, backend=self.backend)

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedNumPerPre((op(self.data, other), self.indices), shape=self.shape, backend=self.backend)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedNumPerPre((op(self.data, other), self.indices), shape=self.shape, backend=self.backend)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedNumPerPre((op(other, self.data), self.indices), shape=self.shape, backend=self.backend)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedNumPerPre((op(other, self.data), self.indices), shape=self.shape, backend=self.backend)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")


@jax.tree_util.register_pytree_node_class
class FixedNumPerPost(FixedNumConn):
    """
    Sparse matrix with a fixed number of pre-synaptic connections per
    post-synaptic neuron (``axis == 1``; stores ``W^T`` row-major).

    ``data`` and ``indices`` have shape ``(num_post, num_conn)``; the equivalent
    dense matrix is ``W[indices[j, k], j] = data[j, k]``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import FixedNumPerPost
        >>>
        >>> data = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
        >>> indices = jnp.array([[0, 1], [1, 0], [0, 2]])
        >>> mat = FixedNumPerPost(data, indices, shape=(3, 3))
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
        _ensure_fixed_conn_initialized_outside_jit(self.indices, kind='FixedNumPerPost')
        _validate_fixed_conn_indices(self.indices, expected_rows=shape[1], kind='Pre-synaptic')
        if self.data.size != 1 and self.data.shape != self.indices.shape:
            raise ValueError(
                f"Data shape {self.data.shape} must match indices shape {self.indices.shape}. "
                f"But got {self.data.shape} != {self.indices.shape}"
            )
        super().__init__((self.data, self.indices), shape=shape, backend=backend)
        _contains_invalid_indices(self.indices, upper_bound=self.shape[0])

    def with_data(self, data: Data) -> 'FixedNumPerPost':
        """Return a new matrix with the same connectivity and replaced values."""
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedNumPerPost((data, self.indices), shape=self.shape, backend=self.backend)

    def todense(self):
        """Convert to a dense matrix of shape ``(num_pre, num_post)``."""
        pre_ids, post_ids, spinfo = fixed_pre_num_to_coo(self)
        return _coo_todense(self.data.flatten(), pre_ids, post_ids, spinfo=spinfo)

    def transpose(self, axes=None) -> FixedNumPerPre:
        """Transpose to a :class:`FixedNumPerPre` (O(1); reinterprets indices)."""
        assert axes is None, "transpose does not support axes argument."
        return FixedNumPerPre(
            (self.data, self.indices),
            shape=self.shape[::-1],
            backend=self.backend,
        )

    def _unitary_op(self, op):
        return FixedNumPerPost((op(self.data), self.indices), shape=self.shape, backend=self.backend)

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedNumPerPost((op(self.data, other), self.indices), shape=self.shape, backend=self.backend)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedNumPerPost((op(self.data, other), self.indices), shape=self.shape, backend=self.backend)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedNumPerPost((op(other, self.data), self.indices), shape=self.shape, backend=self.backend)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedNumPerPost((op(other, self.data), self.indices), shape=self.shape, backend=self.backend)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")


def fixed_post_num_to_coo(self: FixedNumPerPre):
    """Convert a :class:`FixedNumPerPre` to ``(pre_ids, post_ids, COOInfo)``."""
    import jax.numpy as jnp
    pre_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    post_ids = self.indices.flatten()
    # Keep COO metadata unsorted to preserve duplicate accumulation semantics
    # in coo_todense on GPU (sorted paths may overwrite duplicates).
    spinfo = COOInfo(self.shape, rows_sorted=False, cols_sorted=False)
    return pre_ids, post_ids, spinfo


def fixed_pre_num_to_coo(self: FixedNumPerPost):
    """Convert a :class:`FixedNumPerPost` to ``(pre_ids, post_ids, COOInfo)``."""
    import jax.numpy as jnp
    pre_ids = self.indices.flatten()
    post_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    # Keep COO metadata unsorted to preserve duplicate accumulation semantics
    # in coo_todense on GPU (sorted paths may overwrite duplicates).
    spinfo = COOInfo(self.shape, rows_sorted=False, cols_sorted=False)
    return pre_ids, post_ids, spinfo
