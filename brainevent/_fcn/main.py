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
from typing import Optional, TYPE_CHECKING, cast

import brainunit as u
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from brainevent._csr.main import CSR

from brainevent._compatible_import import Tracer
from brainevent._csr.binary_indexed import binary_csrmv_indexed, binary_csrmm_indexed
from brainevent._csr.plasticity_binary import update_csr_on_binary_post
from brainevent._csr.slice import csr_slice_rows
from brainevent._data import DataRepresentation
from brainevent._event.binary import BinaryArray
from brainevent._misc import (
    _coo_todense, COOInfo, fixed_conn_num_csc_structure,
    fixed_conn_num_csr_indptr, normalize_row_index, build_sub_csr,
)
from brainevent._typing import Data, MatrixShape, Index
from .binary import binary_fcnmv, binary_fcnmm
from .float import fcnmv, fcnmm
from .plasticity_binary import (
    update_fixed_post_conn_on_binary_pre,
    update_fixed_pre_conn_on_binary_post,
)
from .yw2y import fcnmv_yw2y

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

    * :class:`FixedNumPerPre` (≡ :class:`brainevent.CSR`): each pre-synaptic
      neuron has a fixed number of outgoing connections; ``indices`` are
      post-synaptic ids.
    * :class:`FixedNumPerPost` (≡ :class:`brainevent.CSC`): each post-synaptic
      neuron has a fixed number of incoming connections; ``indices`` are
      pre-synaptic ids.

    Event-driven matrix-vector products follow the same favorable/unfavorable
    dispatch as :class:`brainevent.CSR` / :class:`brainevent.CSC`.  When the
    event vector indexes the ELL *stored* axis the product is a direct
    column-scatter (:func:`brainevent._fcn.binary.binary_fcnmv` with
    ``transpose=True``).  Otherwise the product would require a gather over every
    stored synapse; instead the structure is converted once to a column-major
    (CSC) view -- ``(indptr, indices, perm)`` built by
    :func:`brainevent._misc.fixed_conn_num_csc_structure` -- and the reused,
    perm-fused CSR kernel (:func:`brainevent.binary_csrmv_indexed`) reads
    ``data[perm[j]]`` so only active columns are touched.  The CSC view is built
    lazily on first need from concrete indices and cached in the ``'csc'`` buffer,
    so it must be triggered outside ``jax.jit``.

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

    def __init__(self, *args, shape: MatrixShape, backend: Optional[str] = None,
                 buffers: Optional[dict] = None):
        self.backend = backend
        super().__init__(*args, shape=shape, buffers=buffers)

    # ------------------------------------------------------------------ #
    # Orientation hooks (override per subclass; replace the old ``axis`` flag)
    # ------------------------------------------------------------------ #

    @property
    def _a_shape(self):
        """Stored CSR-structure shape ``S`` (``FixedNumPerPre`` ≡ CSR ``self.shape``;
        ``FixedNumPerPost`` ≡ CSC ``self.shape[::-1]``)."""
        raise NotImplementedError

    def _ell_transpose(self, transpose_W: bool) -> bool:
        """ELL transpose flag for the stored structure (favorable column-scatter ⇔ True)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Layout / dispatch helpers
    # ------------------------------------------------------------------ #

    def _ell_plan(self, transpose_W: bool):
        return self._a_shape, self._ell_transpose(transpose_W)

    def _weight_indices(self):
        """Lazily build and cache the column-major (CSC) view in ``buffers['csc']``.

        Returns the triple ``(csc_indptr, csc_indices, perm)`` for the ELL operand
        matrix of shape :attr:`_a_shape`, where ``perm`` maps each CSC slot to the
        position of its weight in the flattened ELL ``data``.  Built from concrete
        indices via :func:`fixed_conn_num_csc_structure`, so it must be triggered
        outside ``jax.jit``; cached in the buffer registry under ``'csc'``.
        """
        cached = self.buffers.get('csc')
        if cached is not None:
            return cached
        _ensure_fixed_conn_initialized_outside_jit(self.indices, kind=type(self).__name__)
        with jax.ensure_compile_time_eval():
            csc = fixed_conn_num_csc_structure(self.indices, shape=self._a_shape)
        self.register_buffer('csc', csc)
        return csc

    def build_weight_indices(self):
        """Eagerly build and cache the CSC mirror, returning a new instance.

        Parity with :meth:`brainevent.CSR.build_weight_indices`: builds the
        column-major ``(indptr, indices, perm)`` triple from concrete indices and
        stores it in the ``'csc'`` buffer of the returned matrix; the underlying
        ``data`` and ``indices`` arrays are shared (not copied).
        """
        _ensure_fixed_conn_initialized_outside_jit(self.indices, kind=type(self).__name__)
        with jax.ensure_compile_time_eval():
            csc = fixed_conn_num_csc_structure(self.indices, shape=self._a_shape)
        buffers = dict(self.buffers)
        buffers['csc'] = csc
        return type(self).tree_unflatten(
            ({'indices': self.indices, 'shape': self.shape, 'backend': self.backend}, buffers),
            (self.data,),
        )

    def _binary_matvec(self, s, transpose_W: bool):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        if ell_transpose:
            # Favorable: the event vector indexes the ELL stored axis -> direct
            # column-scatter over active events.
            return binary_fcnmv(
                self.data, self.indices, s,
                shape=a_shape, transpose=True, backend=self.backend,
            )
        # Unfavorable: traverse the cached CSC mirror with the reused, perm-fused
        # CSR kernel -- it reads ``data[perm[j]]`` so only active columns are
        # touched (no full-size weight gather). Same shape/transpose as CSR/CSC.
        csc_indptr, csc_indices, perm = self._weight_indices()
        return binary_csrmv_indexed(
            self.data.reshape(-1), csc_indices, csc_indptr, perm, s,
            shape=a_shape[::-1], transpose=True, backend=self.backend,
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
        # Unfavorable: perm-fused indexed matmat over the cached CSC mirror --
        # parity with the matvec unfavorable path.
        csc_indptr, csc_indices, perm = self._weight_indices()
        return binary_csrmm_indexed(
            self.data.reshape(-1), csc_indices, csc_indptr, perm, matrix,
            shape=a_shape[::-1], transpose=True, backend=self.backend,
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
        return fcnmv_yw2y(w, self.indices, y_dim_arr, shape=self._a_shape,
                          transpose=self._ell_transpose(False))

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
        return fcnmv_yw2y(w, self.indices, y_dim_arr, shape=self._a_shape,
                          transpose=self._ell_transpose(True))

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

        Both orientations describe the *same* logical matrix of shape
        ``(num_pre, num_post)``: ``pre_ids`` are row (pre-synaptic) indices and
        ``post_ids`` are column (post-synaptic) indices, each of length
        ``self.indices.size``.  Concrete behavior is defined per subclass
        (``FixedNumPerPre`` stores by pre, ``FixedNumPerPost`` by post).
        """
        raise NotImplementedError

    def _csr_index(self):
        """Return ``(indptr, indices, order)`` for the CSR view of ``W``.

        Builds the Compressed Sparse Row layout of the logical matrix ``W``
        directly from the stored ELL structure, without a generic COO-to-CSR
        conversion.  ``indptr`` has length ``num_pre + 1`` and ``indices`` are
        the column (post-synaptic) ids ordered by row.  ``order`` is a
        permutation mapping the flattened ELL ``data`` into CSR order, or
        ``None`` when the stored layout is already row-major and the data needs
        no reordering.  Concrete behavior is defined per subclass.
        """
        raise NotImplementedError

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

        # Building the CSR layout reads concrete indices (and, for the
        # column-major orientation, reorders connections by row), so it must run
        # outside jit. With the data-only pytree leaf, ``indices`` is static
        # (concrete even under jit), so the in-trace signal is the ``data`` leaf
        # becoming a tracer.
        if isinstance(self.data, Tracer) or isinstance(self.indices, Tracer):
            raise RuntimeError(
                f'{type(self).__name__}.to_csr() reorders connections by row and '
                'must be called outside `jax.jit` / `brainstate.transform.jit`. '
                'Convert the connection before entering the jitted function.'
            )
        indptr, indices, order = self._csr_index()
        if self.data.size == 1:
            # Homogeneous weight: keep the single shared value.
            data = self.data.reshape(1)
        elif order is None:
            # Row-major storage is already in CSR order; no permutation needed.
            data = self.data.reshape(-1)
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
        :meth:`tree_unflatten` path.  The cached ``buffers`` mirror is carried
        through (the structure is unchanged), so it survives the rebuild.
        """
        aux = {'indices': self.indices, 'shape': self.shape, 'backend': self.backend}
        return type(self).tree_unflatten((aux, self.buffers), (new_data,))

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None):
        """Apply a pre-spike-triggered STDP update, returning a new matrix.

        For each firing pre neuron ``i`` every stored synapse is updated
        ``W[i, j] <- clip(W[i, j] + post_trace[j], w_min, w_max)``.  Favorable
        (row-driven) for :class:`FixedNumPerPre`, unfavorable for
        :class:`FixedNumPerPost`.  Concrete behavior is defined per subclass.

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
        raise NotImplementedError

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None):
        """Apply a post-spike-triggered STDP update, returning a new matrix.

        For each firing post neuron ``j`` every stored synapse is updated
        ``W[i, j] <- clip(W[i, j] + pre_trace[i], w_min, w_max)``.  Unfavorable
        for :class:`FixedNumPerPre`, favorable (row-driven) for
        :class:`FixedNumPerPost`.  Concrete behavior is defined per subclass.

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
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Pytree protocol
    # ------------------------------------------------------------------ #

    def tree_flatten(self):
        """Flatten: ``data`` is the only leaf; ``indices``/``shape``/``backend`` and
        the rebuildable ``buffers`` mirror are static aux (mirrors CompressedSparseData)."""
        aux = {'indices': self.indices, 'shape': self.shape, 'backend': self.backend}
        return (self.data,), (aux, self.buffers)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from pytree components, restoring the buffer registry."""
        obj = object.__new__(cls)
        obj.data, = children
        aux, buffers = aux_data
        obj._buffer_registry = set(buffers.keys())
        for k, v in aux.items():
            setattr(obj, k, v)
        for k, v in buffers.items():
            setattr(obj, k, v)
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
    pre-synaptic neuron (row-major ELL; structurally equivalent to
    :class:`brainevent.CSR`).

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
        precompute_weight_indices: bool = False,
        buffers: Optional[dict] = None,
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
        super().__init__((self.data, self.indices), shape=shape, backend=backend, buffers=buffers)
        _contains_invalid_indices(self.indices, upper_bound=self.shape[1])
        if precompute_weight_indices:
            self._weight_indices()

    # FixedNumPerPre ≡ CSR: stored structure shape is ``self.shape``.
    @property
    def _a_shape(self):
        return tuple(self.shape)

    def _ell_transpose(self, transpose_W: bool) -> bool:
        return bool(transpose_W)

    def _to_coo(self):
        return fixed_post_num_to_coo(self)

    def _csr_index(self):
        # Row-major ELL is structurally a CSR of ``W``: a uniform row pointer
        # and the stored indices flattened in row order, with the ``data``
        # already in CSR order (no permutation).
        indptr = fixed_conn_num_csr_indptr(self.indices)
        indices = self.indices.reshape(-1)
        return indptr, indices, None

    def with_data(self, data: Data) -> 'FixedNumPerPre':
        """Return a new matrix with the same connectivity and replaced values."""
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedNumPerPre((data, self.indices), shape=self.shape,
                              backend=self.backend, buffers=self.buffers)

    def __getitem__(self, index):
        """Extract rows of ``W`` as a dense array (NumPy semantics).

        Row slicing is favorable for :class:`FixedNumPerPre`: the ELL is a CSR
        with an implicit uniform ``indptr``, fed straight to ``csr_slice_rows``.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0 (pre-synaptic). Negative indices wrap;
            Python slices are supported; concrete OOB indices raise ``IndexError``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            ``(num_post,)`` for a single int, otherwise ``(len(rows), num_post)``.
        """
        rows = normalize_row_index(index, self.shape[0])
        indptr = fixed_conn_num_csr_indptr(self.indices)
        flat_indices = self.indices.reshape(-1)
        flat_data = self.data if self.data.size == 1 else self.data.reshape(-1)
        return csr_slice_rows(
            flat_data, flat_indices, indptr, rows,
            shape=self.shape, backend=self.backend,
        )

    def slice_rows(self, index) -> 'FixedNumPerPre':
        """Return ``W[rows, :]`` as a new :class:`FixedNumPerPre`.

        Selecting pre-synaptic rows preserves the fixed-connection invariant
        (each selected row keeps its ``num_conn`` entries), so this is a static
        gather and is safe under ``jax.jit``.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0 (pre-synaptic).

        Returns
        -------
        FixedNumPerPre
            Sparse sub-matrix of shape ``(len(rows), num_post)``.
        """
        rows = jnp.atleast_1d(normalize_row_index(index, self.shape[0]))
        new_indices = self.indices[rows]
        data = cast(jax.Array, self.data)
        new_data = data if data.size == 1 else data[rows]
        k = new_indices.shape[0]
        # Structure-preserving build that bypasses the outside-jit constructor
        # guard (a validated subset of rows), mirroring `_rebuild_with_data`.
        # The CSC mirror is not carried over (the structure changed); it rebuilds
        # lazily on first need.
        aux = {'indices': new_indices, 'shape': (k, self.shape[1]), 'backend': self.backend}
        return FixedNumPerPre.tree_unflatten((aux, {}), (new_data,))

    def todense(self):
        """Convert to a dense matrix of shape ``(num_pre, num_post)``."""
        pre_ids, post_ids, spinfo = fixed_post_num_to_coo(self)
        return _coo_todense(self.data.flatten(), pre_ids, post_ids, spinfo=spinfo)

    def transpose(self, axes=None) -> 'FixedNumPerPost':
        """Transpose to a :class:`FixedNumPerPost` (O(1); reinterprets indices).

        Orientation flips, so the cached ``'csc'`` mirror is *not* carried over;
        the new matrix rebuilds its own mirror lazily on first need.
        """
        assert axes is None, "transpose does not support axes argument."
        return FixedNumPerPost(self.data, self.indices, shape=self.shape[::-1], backend=self.backend)

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None):
        """Pre-spike STDP (favorable, row-driven) -- mirrors :meth:`brainevent.CSR.update_on_pre`."""
        new = update_fixed_post_conn_on_binary_pre(
            self.data, self.indices, pre_spike, post_trace, w_min, w_max,
            shape=self.shape, backend=self.backend,
        )
        return self._rebuild_with_data(new)

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None):
        """Post-spike STDP (unfavorable) -- mirrors :meth:`brainevent.CSR.update_on_post` (perm-fused)."""
        csc_indptr, csc_indices, perm = self._weight_indices()
        new = update_csr_on_binary_post(
            self.data.reshape(-1), csc_indices, csc_indptr, perm,
            pre_trace, post_spike, w_min, w_max,
            shape=self._a_shape, backend=self.backend,
        )
        return self._rebuild_with_data(new.reshape(self.data.shape))

    def _unitary_op(self, op):
        return FixedNumPerPre(op(self.data), self.indices, shape=self.shape, backend=self.backend, buffers=self.buffers)

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedNumPerPre(op(self.data, other), self.indices, shape=self.shape,
                                  backend=self.backend, buffers=self.buffers)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedNumPerPre(op(self.data, other), self.indices, shape=self.shape,
                                  backend=self.backend, buffers=self.buffers)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedNumPerPre(op(other, self.data), self.indices, shape=self.shape,
                                  backend=self.backend, buffers=self.buffers)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            other = other[rows, cols]
            return FixedNumPerPre(op(other, self.data), self.indices, shape=self.shape,
                                  backend=self.backend, buffers=self.buffers)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")


@jax.tree_util.register_pytree_node_class
class FixedNumPerPost(FixedNumConn):
    """
    Sparse matrix with a fixed number of pre-synaptic connections per
    post-synaptic neuron (stores ``W^T`` row-major; structurally equivalent to
    :class:`brainevent.CSC`).

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
        precompute_weight_indices: bool = False,
        buffers: Optional[dict] = None,
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
        super().__init__((self.data, self.indices), shape=shape, backend=backend, buffers=buffers)
        _contains_invalid_indices(self.indices, upper_bound=self.shape[0])
        if precompute_weight_indices:
            self._weight_indices()

    # FixedNumPerPost ≡ CSC: stored structure shape is ``self.shape[::-1]``.
    @property
    def _a_shape(self):
        return tuple(self.shape)[::-1]

    def _ell_transpose(self, transpose_W: bool) -> bool:
        return not bool(transpose_W)

    def _to_coo(self):
        return fixed_pre_num_to_coo(self)

    def _csr_index(self):
        # Column-major ELL stores ``W^T`` row-major, so the compact CSC structure
        # of the stored operand (built and cached by ``_weight_indices``) is
        # exactly the CSR of ``W``; ``perm`` maps each CSR slot to the position
        # of its weight in the flattened ELL ``data``.
        csr_indptr, csr_indices, perm = self._weight_indices()
        return csr_indptr, csr_indices, perm

    def with_data(self, data: Data) -> 'FixedNumPerPost':
        """Return a new matrix with the same connectivity and replaced values."""
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return FixedNumPerPost((data, self.indices), shape=self.shape,
                               backend=self.backend, buffers=self.buffers)

    def __getitem__(self, index):
        """Extract rows of ``W`` as a dense array (NumPy semantics).

        Row slicing is the *unfavorable* direction for :class:`FixedNumPerPost`
        (which stores ``W`` column-major), so it reuses the cached CSR-of-``W``
        view from :meth:`_weight_indices` and the shared ``csr_slice_rows``
        kernel.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0 (pre-synaptic). Negative indices wrap;
            Python slices are supported; concrete OOB indices raise ``IndexError``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            ``(num_post,)`` for a single int, otherwise ``(len(rows), num_post)``.
        """
        rows = normalize_row_index(index, self.shape[0])
        csc_indptr, csc_indices, perm = self._weight_indices()
        weights = self.data if self.data.size == 1 else self.data.reshape(-1)[perm]
        return csr_slice_rows(
            weights, csc_indices, csc_indptr, rows,
            shape=self.shape, backend=self.backend,
        )

    def slice_rows(self, index) -> 'CSR':
        """Return ``W[rows, :]`` as a :class:`~brainevent.CSR` (outside ``jax.jit``).

        Selecting pre-synaptic rows breaks the fixed-per-post invariant (each
        post keeps a variable number of incoming pre), so the canonical
        row-major result is a :class:`CSR`. Built from the cached CSR-of-``W``
        view; the output non-zero count is data-dependent, so ``index`` must be
        concrete.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0 (pre-synaptic).

        Returns
        -------
        CSR
            Sparse sub-matrix of shape ``(len(rows), num_post)``.
        """
        from brainevent._csr.main import CSR  # local import avoids import cycle
        rows = jnp.atleast_1d(normalize_row_index(index, self.shape[0]))
        csc_indptr, csc_indices, perm = self._weight_indices()
        data = cast(jax.Array, self.data)
        weights = data if data.size == 1 else data.reshape(-1)[perm]
        new_data, new_indices, new_indptr, shape = build_sub_csr(
            weights, csc_indices, csc_indptr, rows, self.shape[1],
        )
        return CSR((new_data, new_indices, new_indptr), shape=shape, backend=self.backend)

    def todense(self):
        """Convert to a dense matrix of shape ``(num_pre, num_post)``."""
        pre_ids, post_ids, spinfo = fixed_pre_num_to_coo(self)
        return _coo_todense(self.data.flatten(), pre_ids, post_ids, spinfo=spinfo)

    def transpose(self, axes=None) -> FixedNumPerPre:
        """Transpose to a :class:`FixedNumPerPre` (O(1); reinterprets indices).

        Orientation flips, so the cached ``'csc'`` mirror is *not* carried over;
        the new matrix rebuilds its own mirror lazily on first need.
        """
        assert axes is None, "transpose does not support axes argument."
        return FixedNumPerPre(
            (self.data, self.indices),
            shape=self.shape[::-1],
            backend=self.backend,
        )

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None):
        """Pre-spike STDP (unfavorable) -- mirrors :meth:`brainevent.CSC.update_on_pre` (perm-fused)."""
        csc_indptr, csc_indices, perm = self._weight_indices()
        new = update_csr_on_binary_post(
            self.data.reshape(-1), csc_indices, csc_indptr, perm,
            post_trace, pre_spike, w_min, w_max,
            shape=self._a_shape, backend=self.backend,
        )
        return self._rebuild_with_data(new.reshape(self.data.shape))

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None):
        """Post-spike STDP (favorable, row-driven) -- mirrors :meth:`brainevent.CSC.update_on_post`."""
        new = update_fixed_pre_conn_on_binary_post(
            self.data, self.indices, pre_trace, post_spike, w_min, w_max,
            shape=self.shape, backend=self.backend,
        )
        return self._rebuild_with_data(new)

    def _unitary_op(self, op):
        return FixedNumPerPost((op(self.data), self.indices), shape=self.shape,
                               backend=self.backend, buffers=self.buffers)

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedNumPerPost((op(self.data, other), self.indices), shape=self.shape,
                                   backend=self.backend, buffers=self.buffers)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedNumPerPost((op(self.data, other), self.indices), shape=self.shape,
                                   backend=self.backend, buffers=self.buffers)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            return FixedNumPerPost((op(other, self.data), self.indices), shape=self.shape,
                                   backend=self.backend, buffers=self.buffers)
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            other = other[rows, cols]
            return FixedNumPerPost((op(other, self.data), self.indices), shape=self.shape,
                                   backend=self.backend, buffers=self.buffers)
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")


def fixed_post_num_to_coo(self: FixedNumPerPre):
    """Convert a :class:`FixedNumPerPre` to ``(pre_ids, post_ids, COOInfo)``."""
    pre_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    post_ids = self.indices.flatten()
    # Keep COO metadata unsorted to preserve duplicate accumulation semantics
    # in coo_todense on GPU (sorted paths may overwrite duplicates).
    spinfo = COOInfo(self.shape, rows_sorted=False, cols_sorted=False)
    return pre_ids, post_ids, spinfo


def fixed_pre_num_to_coo(self: FixedNumPerPost):
    """Convert a :class:`FixedNumPerPost` to ``(pre_ids, post_ids, COOInfo)``."""
    pre_ids = self.indices.flatten()
    post_ids = jnp.repeat(jnp.arange(self.indices.shape[0]), self.indices.shape[1])
    # Keep COO metadata unsorted to preserve duplicate accumulation semantics
    # in coo_todense on GPU (sorted paths may overwrite duplicates).
    spinfo = COOInfo(self.shape, rows_sorted=False, cols_sorted=False)
    return pre_ids, post_ids, spinfo
