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
from typing import Optional, TYPE_CHECKING, Union, cast

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from brainevent._csr.main import CSR

from brainevent._csr.binary_indexed import binary_csrmv_indexed, binary_csrmm_indexed
from brainevent._csr.plasticity_binary import update_csr_on_binary_post, _update_csr_on_binary_post_masked
from brainevent._csr.slice import csr_slice_rows
from brainevent._data import DataRepresentation
from brainevent._structure import StructureCache, is_concrete, nnz_hint, warn_mirror_fallback
from brainevent._event.dispatch import as_event_operand
from brainevent._misc import (
    _coo_todense, COOInfo, fixed_conn_num_csc_structure,
    fixed_conn_num_csr_indptr, normalize_row_index, build_sub_csr,
    _as_int32_indices,
)
from brainevent._typing import ArrayData, Data, MatrixShape, Index
from .binary import binary_fcnmv, binary_fcnmm
from .float import fcnmv, fcnmm
from .plasticity_binary import (
    update_fixed_post_conn_on_binary_pre,
    update_fixed_pre_conn_on_binary_post,
)
from .dt2t import fcnmv_dt2t

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


def _contains_invalid_indices(indices: Index, *, upper_bound: int) -> None:
    if not is_concrete(indices):
        return  # value checks need concrete indices; shapes/dtypes are checked statically
    with jax.ensure_compile_time_eval():
        indices_np = np.asarray(indices)
        if bool(np.any(indices_np < 0) or np.any(indices_np >= upper_bound)):
            raise ValueError(
                f'Found invalid indices in the connection matrix. '
                f'All indices must be in the range [0, {upper_bound - 1}]. '
                f'But found indices with min {indices_np.min()} and max {indices_np.max()}.'
            )


def _ell_from_rows(mantissa: np.ndarray, num_conn: Optional[int], *, kind: str):
    """Build fixed-connection ``(data, indices)`` from a row-major dense block.

    Each row of ``mantissa`` becomes one fixed-connection row: its non-zero
    columns are gathered (stable order) into ``indices`` and ``data``, padded to
    ``num_conn`` with a zero-weight sentinel at column ``0`` (which contributes
    nothing to the scatter kernels or :meth:`todense`).

    Parameters
    ----------
    mantissa : numpy.ndarray
        Unit-stripped dense block of shape ``(R, C)``.
    num_conn : int or None
        Target connections per row. ``None`` infers it and requires a uniform
        per-row non-zero count; otherwise pads/validates against this value.
    kind : str
        Subclass name, used in error messages.

    Returns
    -------
    data : numpy.ndarray
        Values of shape ``(R, num_conn)``.
    indices : numpy.ndarray
        Column ids of shape ``(R, num_conn)`` (``int32``).
    """
    mask = mantissa != 0
    nnz = mask.sum(axis=1)
    if num_conn is None:
        if mantissa.shape[0] == 0:
            num_conn = 0
        elif not bool((nnz == nnz[0]).all()):
            raise ValueError(
                f'{kind}.fromdense: rows have a non-uniform number of connections '
                f'(min {int(nnz.min())}, max {int(nnz.max())}). Pass num_conn= to '
                'pad to a fixed count, or use CSR.fromdense / CSC.fromdense for an '
                'irregular matrix.'
            )
        else:
            num_conn = int(nnz[0])
    else:
        num_conn = int(num_conn)
        if bool((nnz > num_conn).any()):
            raise ValueError(
                f'{kind}.fromdense: num_conn={num_conn} is too small; a row has '
                f'{int(nnz.max())} connections.'
            )
    # Stable argsort of ``~mask`` puts non-zero columns first, preserving column
    # order; take the first ``num_conn`` slots.
    order = np.argsort(~mask, axis=1, kind='stable')[:, :num_conn]
    selected = np.take_along_axis(mask, order, axis=1)
    data = np.where(selected, np.take_along_axis(mantissa, order, axis=1), 0).astype(mantissa.dtype)
    indices = np.where(selected, order, 0).astype(np.int32)
    return data, indices


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
    ``data[perm[j]]`` so only active columns are touched.  The CSC view (the
    *mirror*) is built lazily on first need from concrete indices and shared by
    every matrix with the same structure, including the transpose.  The matrix
    is a JAX pytree with leaves ``data``, ``indices`` and, once built, the three
    mirror arrays, so it can be constructed and transformed inside ``jax.jit``.
    A matrix whose indices are traced and that carries no mirror computes the
    unfavorable product with a gather instead, and warns once with
    :class:`brainevent.PerformanceWarning`; call :meth:`build_weight_indices`
    before ``jax.jit`` to keep it event-driven.  Differentiate with respect to
    ``M.data`` and rebuild with :meth:`with_data`.

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
    data: ArrayData
    indices: Index
    shape: MatrixShape
    backend: Optional[str]

    def __init__(self, *args, shape: MatrixShape, backend: Optional[str] = None,
                 buffers: Optional[dict] = None):
        self.backend = backend
        super().__init__(*args, shape=shape)
        self._cache = StructureCache()
        self._ingest_buffers(buffers)

    @classmethod
    def _from_parts(cls, data, indices, *, shape, backend=None, cache=None):
        """Assemble a matrix from trusted parts, without conversion or validation."""
        obj = object.__new__(cls)
        obj.data = data
        obj.indices = indices
        obj.shape = tuple(int(s) for s in shape)
        obj.backend = backend
        obj._buffer_registry = set()
        obj._cache = StructureCache() if cache is None else cache
        return obj

    def _same_structure(self, data):
        """A matrix of the same class and structure (shared caches) holding ``data``."""
        return type(self)._from_parts(data, self.indices, shape=self.shape, backend=self.backend, cache=self._cache)

    def _ingest_buffers(self, buffers) -> None:
        """Store the legacy ``'csc'`` mirror buffer."""
        for name, value in (buffers or {}).items():
            if value is None:
                continue
            if name != 'csc':
                raise ValueError(
                    f"{type(self).__name__} does not support buffer {name!r}; the only "
                    f"known buffer is 'csc'."
                )
            self._cache.set_mirror(value)

    def register_buffer(self, name, value=None):
        """Store the mirror under its legacy name ``'csc'`` (see :attr:`buffers`)."""
        self._ingest_buffers({name: value})

    def set_buffer(self, name, value):
        """Store the mirror under its legacy name ``'csc'`` (see :attr:`buffers`)."""
        self._ingest_buffers({name: value})

    @property
    def buffers(self) -> dict:
        """The cached mirror ``(indptr, indices, perm)`` under ``'csc'`` (a snapshot)."""
        return {} if self._cache.mirror is None else {'csc': self._cache.mirror}

    def sum(self, axis=None):
        """Return the sum of all logically represented matrix entries.

        Parameters
        ----------
        axis : int, sequence of int, or None, optional
            Axis or axes to reduce. Only ``None`` is currently supported.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Scalar sum of every fixed connection. A homogeneous size-one
            value is counted once per represented connection.

        Raises
        ------
        NotImplementedError
            If ``axis`` is not ``None``.
        """
        if axis is not None:
            raise NotImplementedError(
                f'{type(self).__name__}.sum with axis is not implemented.'
            )
        return self._sum_data()

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

    def _mirror(self):
        """Return the CSC mirror ``(indptr, indices, perm)``, or ``None`` if traced.

        The mirror is the column-major view of the stored structure (shape
        :attr:`_a_shape`); ``perm`` maps each CSC slot to the position of its
        weight in the flattened ELL ``data``. A concrete structure builds it
        once and caches it (also under an active trace, since the arrays are
        computed eagerly); a traced structure without a carried mirror returns
        ``None``.
        """
        cache = self._cache
        if cache.mirror is not None:
            return cache.mirror
        if not is_concrete(self.indices):
            return None
        with jax.ensure_compile_time_eval():
            mirror = fixed_conn_num_csc_structure(self.indices, shape=self._a_shape)
        return cache.set_mirror(mirror)

    def _weight_indices(self):
        """Return the CSC mirror, computing it (uncached) for a traced structure."""
        mirror = self._mirror()
        if mirror is None:
            mirror = fixed_conn_num_csc_structure(self.indices, shape=self._a_shape)
        return mirror

    def build_weight_indices(self):
        """Return a matrix sharing these arrays with the CSC mirror built.

        Parity with :meth:`brainevent.CSR.build_weight_indices`. The mirror
        drives the event-driven kernels of the unfavorable product direction
        and plasticity update. It is built lazily on first use when the indices
        are concrete, so call this before ``jax.jit`` when the matrix enters the
        jitted function as an argument.

        Returns
        -------
        FixedNumConn
            A matrix of the same class sharing ``data`` and ``indices``, with
            the mirror cached (and carried as pytree leaves).
        """
        if self._mirror() is not None:
            cache = self._cache
        else:
            cache = StructureCache()
            cache.set_mirror(self._weight_indices())
        return type(self)._from_parts(self.data, self.indices, shape=self.shape, backend=self.backend, cache=cache)

    def _gather_backend(self):
        """Backend for the mirror-free gather (``cuda_raw`` has no FCN gather kernel)."""
        if self.backend == 'cuda_raw' or (self.backend is None and jax.default_backend() == 'gpu'):
            return 'jax_raw'
        return self.backend

    def _binary_matvec(self, s, transpose_W: bool, weighted: bool = False):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        if ell_transpose:
            # Favorable: the event vector indexes the ELL stored axis -> direct
            # column-scatter over active events.
            return binary_fcnmv(
                self.data, self.indices, s,
                shape=a_shape, transpose=True, backend=self.backend, weighted=weighted,
            )
        # Unfavorable: traverse the cached CSC mirror with the reused, perm-fused
        # CSR kernel -- it reads ``data[perm[j]]`` so only active columns are
        # touched (no full-size weight gather). Same shape/transpose as CSR/CSC.
        mirror = self._mirror()
        if mirror is None:
            warn_mirror_fallback(type(self).__name__, 'the unfavorable event product')
            return binary_fcnmv(
                self.data, self.indices, s,
                shape=a_shape, transpose=False, backend=self._gather_backend(), weighted=weighted,
            )
        csc_indptr, csc_indices, perm = mirror
        return binary_csrmv_indexed(
            self.data.reshape(-1), csc_indices, csc_indptr, perm, s,
            shape=a_shape[::-1], transpose=True, backend=self.backend,
            weighted=weighted, max_row_nnz=self._cache.mirror_hint,
        )

    def _binary_matmat(self, matrix, transpose_W: bool, weighted: bool = False):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        if ell_transpose:
            # Favorable: the event matrix indexes the ELL stored axis -> direct
            # column-scatter over active events.
            return binary_fcnmm(
                self.data, self.indices, matrix,
                shape=a_shape, transpose=True, backend=self.backend, weighted=weighted,
            )
        # Unfavorable: perm-fused indexed matmat over the cached CSC mirror --
        # parity with the matvec unfavorable path.
        mirror = self._mirror()
        if mirror is None:
            warn_mirror_fallback(type(self).__name__, 'the unfavorable event product')
            return binary_fcnmm(
                self.data, self.indices, matrix,
                shape=a_shape, transpose=False, backend=self._gather_backend(), weighted=weighted,
            )
        csc_indptr, csc_indices, perm = mirror
        return binary_csrmm_indexed(
            self.data.reshape(-1), csc_indices, csc_indptr, perm, matrix,
            shape=a_shape[::-1], transpose=True, backend=self.backend,
            weighted=weighted, max_row_nnz=self._cache.mirror_hint,
        )

    def _float_matvec(self, vector, transpose_W: bool, data):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        return fcnmv(data, self.indices, vector, shape=a_shape, transpose=ell_transpose)

    def _float_matmat(self, matrix, transpose_W: bool, data):
        a_shape, ell_transpose = self._ell_plan(transpose_W)
        return fcnmm(data, self.indices, matrix, shape=a_shape, transpose=ell_transpose)

    # ------------------------------------------------------------------ #
    # Per-synapse y * w product (dt2t), parity with CSR / CSC
    # ------------------------------------------------------------------ #

    def dt2t(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Per-synapse ``w * y`` with ``y`` indexed by the row (pre) of ``W``.

        For every stored connection, returns ``w * y[row]`` where ``row`` is the
        pre-synaptic index of that connection, regardless of storage axis.  This
        is the fixed-connection analog of :meth:`brainevent.CSR.dt2t` and
        implements the ``dt2t`` protocol of :class:`brainevent.DataRepresentation`.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Pre-synaptic (row) vector, sized ``shape[0]``.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weights of shape ``indices.shape`` (or size-1).

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result of shape ``self.indices.shape``.

        See Also
        --------
        dt2t_transposed : ``y`` indexed by the column (post) of ``W``.
        """
        return fcnmv_dt2t(w_dim_arr, self.indices, y_dim_arr, shape=self._a_shape,
                          transpose=self._ell_transpose(False))

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Per-synapse ``w * y`` with ``y`` indexed by the column (post) of ``W``.

        Adjoint counterpart of :meth:`dt2t`: for every stored connection,
        returns ``w * y[col]`` where ``col`` is the post-synaptic index of that
        connection, regardless of storage axis.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Post-synaptic (column) vector, sized ``shape[1]``.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weights of shape ``indices.shape`` (or size-1).

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-synapse result of shape ``self.indices.shape``.

        See Also
        --------
        dt2t : ``y`` indexed by the row (pre) of ``W``.
        """
        return fcnmv_dt2t(w_dim_arr, self.indices, y_dim_arr, shape=self._a_shape,
                          transpose=self._ell_transpose(True))

    def _dispatch(self, other, transpose_W: bool):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        operand = as_event_operand(other)
        if operand is not None:
            value, weighted = operand
            if value.ndim == 1:
                return self._binary_matvec(value, transpose_W, weighted)
            elif value.ndim == 2:
                # ``_binary_matmat`` returns the logical ``(n_out, n_batch)`` layout.
                if transpose_W:
                    return self._binary_matmat(value.T, transpose_W, weighted).T
                return self._binary_matmat(value, transpose_W, weighted)
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

    def tocoo(self) -> u.sparse.COO:
        """Convert to coordinate (COO) format.

        Builds the COO of the logical matrix ``W`` of shape
        ``(num_pre, num_post)`` from the stored ELL structure, irrespective of
        orientation. Promotes the per-subclass ``_to_coo`` helper to the common
        conversion contract. Homogeneous (size-1) weights are broadcast to one
        entry per stored element.

        Returns
        -------
        brainunit.sparse.COO
            The same logical matrix in COO format, ``shape`` unchanged.

        Notes
        -----
        Building the coordinate layout reads concrete indices, so -- like
        :meth:`tocsr` -- it must run outside ``jax.jit``.

        See Also
        --------
        tocsr : Convert to Compressed Sparse Row format.
        tocsc : Convert to Compressed Sparse Column format.
        """
        pre_ids, post_ids, _ = self._to_coo()
        data = self.data.reshape(-1)
        if data.size != pre_ids.size:
            data = u.math.broadcast_to(data, pre_ids.shape)
        return u.sparse.COO((data, pre_ids, post_ids), shape=self.shape)

    def tocsr(self):
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
        tocsc : Convert to Compressed Sparse Column format.
        todense : Convert to a dense matrix.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import FixedNumPerPre
            >>>
            >>> data = jnp.array([[1., 2.], [3., 4.]])
            >>> indices = jnp.array([[0, 1], [1, 2]])
            >>> mat = FixedNumPerPre((data, indices), shape=(2, 3))
            >>> csr = mat.tocsr()
            >>> bool((csr.todense() == mat.todense()).all())
            True
        """
        from brainevent._csr import CSR  # local import avoids an import cycle

        indptr, indices, order = self._csr_index()
        if self.data.size == 1:
            # Homogeneous weight: keep the single shared value.
            data = self.data.reshape(1)
        elif order is None:
            # Row-major storage is already in CSR order; no permutation needed.
            data = self.data.reshape(-1)
        else:
            data = self.data.reshape(-1)[order]
        return CSR._from_parts(data, indices, indptr, shape=self.shape, backend=self.backend)

    def tocsc(self):
        """Convert to a :class:`brainevent.CSC` matrix of the logical matrix ``W``.

        The result is a Compressed Sparse Column view of the same logical
        weight matrix ``W`` of shape ``(num_pre, num_post)``.  It is built by
        transposing to the opposite orientation, taking the CSR view of ``W^T``,
        and reinterpreting it as the (array-identical) CSC view of ``W`` -- so
        the same outside-``jit`` requirement as :meth:`tocsr` applies.

        Returns
        -------
        CSC
            Equivalent matrix in CSC format with the same ``shape``, ``dtype``,
            unit, and ``backend`` as ``self``.

        See Also
        --------
        tocsr : Convert to Compressed Sparse Row format.
        todense : Convert to a dense matrix.
        """
        return self.transpose().tocsr().transpose()

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
        return self._same_structure(new_data)

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
        """Flatten into leaves ``(data, indices, *mirror)`` and hashable metadata."""
        mirror = self._cache.mirror or ()
        aux = (self.shape, self.backend, bool(mirror), self._cache.mirror_hint)
        return (self.data, self.indices, *mirror), aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct without validation (leaves may be tracers or placeholders)."""
        shape, backend, has_mirror, mirror_hint = aux_data
        expected = 2 + 3 * has_mirror
        if len(children) != expected:
            raise ValueError(f'{cls.__name__}.tree_unflatten expects {expected} leaves, got {len(children)}.')
        mirror = tuple(children[2:]) if has_mirror else None
        cache = StructureCache(mirror=mirror, mirror_hint=mirror_hint)
        return cls._from_parts(children[0], children[1], shape=shape, backend=backend, cache=cache)

    # ------------------------------------------------------------------ #
    # Element-wise operators
    # ------------------------------------------------------------------ #

    def _unitary_op(self, op):
        raise NotImplementedError

    def apply(self, fn):
        """Apply a function while keeping the connectivity structure.

        Parameters
        ----------
        fn : callable
            Function applied once to ``self.data``. It may change dtype or
            physical unit but must preserve the physical buffer shape.

        Returns
        -------
        FixedNumConn
            Same concrete fixed-connection representation with transformed
            values.

        Raises
        ------
        ValueError
            If the transformed value buffer does not have the same shape as
            ``self.data``.
        """
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

    data: ArrayData
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
        _validate_fixed_conn_indices(self.indices, expected_rows=shape[0], kind='Post-synaptic')
        # Connection indices are secondary-axis coordinates (post ids) and are
        # always stored int32; bounds are validated below by
        # ``_contains_invalid_indices`` so only a dtype coercion is needed here.
        self.indices = _as_int32_indices(
            self.indices, shape[1], 'FixedNumPerPre indices', check_values=False
        )
        if self.data.size != 1 and self.data.shape != self.indices.shape:
            raise ValueError(
                f"Data shape {self.data.shape} must match indices shape {self.indices.shape}. "
                f"But got {self.data.shape} != {self.indices.shape}"
            )
        super().__init__((self.data, self.indices), shape=shape, backend=backend, buffers=buffers)
        _contains_invalid_indices(self.indices, upper_bound=self.shape[1])
        if precompute_weight_indices and self._mirror() is None:
            # Traced indices: this fresh object is local to the trace.
            self._cache.set_mirror(self._weight_indices())

    # FixedNumPerPre ≡ CSR: stored structure shape is ``self.shape``.
    @property
    def _a_shape(self):
        return tuple(self.shape)

    def _ell_transpose(self, transpose_W: bool) -> bool:
        return bool(transpose_W)

    @classmethod
    def fromdense(cls, mat, *, num_conn=None, backend=None) -> 'FixedNumPerPre':
        """Construct from a dense matrix with a fixed number of connections per pre.

        Each pre-synaptic row of ``mat`` is encoded with ``num_conn`` outgoing
        connections (post-synaptic ids).

        Parameters
        ----------
        mat : jax.Array or brainunit.Quantity
            Dense ``(num_pre, num_post)`` matrix. Explicit zeros are treated as
            absent connections.
        num_conn : int, optional
            Connections per pre-synaptic neuron. If ``None`` (default), inferred
            from the matrix, which must then have a uniform per-row non-zero
            count; otherwise short rows are padded with a zero-weight sentinel
            and a row with more non-zeros than ``num_conn`` raises ``ValueError``.
        backend : str, optional
            Backend tag forwarded to the constructor.

        Returns
        -------
        FixedNumPerPre
            Encoded fixed-connection matrix, ``shape`` equal to ``mat.shape``.

        Raises
        ------
        ValueError
            If ``mat`` is not 2-D, if per-row counts are non-uniform and
            ``num_conn`` is omitted, or if a row exceeds ``num_conn``.

        See Also
        --------
        tocsr : Inverse-direction conversion to CSR.
        brainevent.CSR.fromdense : For irregular matrices.
        """
        mantissa = np.asarray(u.get_mantissa(mat))
        if mantissa.ndim != 2:
            raise ValueError(f"FixedNumPerPre.fromdense expects a 2-D matrix; got {mantissa.ndim}-D.")
        data, indices = _ell_from_rows(mantissa, num_conn, kind=cls.__name__)
        data_q = u.maybe_decimal(u.Quantity(jnp.asarray(data), unit=u.get_unit(mat)))
        return cls((data_q, jnp.asarray(indices)), shape=mantissa.shape, backend=backend)

    def _to_coo(self):
        return fixed_post_num_to_coo(self)

    def _csr_index(self):
        # Row-major ELL is structurally a CSR of ``W``: a uniform row pointer
        # and the stored indices flattened in row order, with the ``data``
        # already in CSR order (no permutation).
        indptr = fixed_conn_num_csr_indptr(self.indices)
        indices = self.indices.reshape(-1)
        return indptr, indices, None

    def with_data(self, data: ArrayData) -> 'FixedNumPerPre':
        """Return a new matrix with the same connectivity and replaced values."""
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return self._same_structure(data)

    def __getitem__(self, index):
        """Extract rows of ``W`` as a dense array.

        Row slicing is favorable for :class:`FixedNumPerPre`: the ELL is a CSR
        with an implicit uniform ``indptr``, fed straight to ``csr_slice_rows``.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0 (pre-synaptic). Negative indices wrap;
            Python slices are supported; concrete OOB indices raise ``IndexError``.
            A tuple is a list of rows: ``M[1, 2]`` returns rows 1 and 2, not the
            NumPy element ``(1, 2)``.

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
        # A validated subset of rows: no re-validation. The structure changed,
        # so the mirror is not carried over; it rebuilds lazily on first need.
        return FixedNumPerPre._from_parts(new_data, new_indices, shape=(k, self.shape[1]), backend=self.backend)

    def todense(self):
        """Convert to a dense matrix of shape ``(num_pre, num_post)``."""
        pre_ids, post_ids, spinfo = fixed_post_num_to_coo(self)
        return _coo_todense(self.data.flatten(), pre_ids, post_ids, spinfo=spinfo)

    def transpose(self, axes=None) -> 'FixedNumPerPost':
        """Transpose to a :class:`FixedNumPerPost` (O(1); reinterprets indices).

        Both orientations store the same structure (``_a_shape`` is unchanged),
        so the transpose shares the cached mirror.
        """
        assert axes is None, "transpose does not support axes argument."
        return FixedNumPerPost._from_parts(
            self.data, self.indices, shape=self.shape[::-1], backend=self.backend, cache=self._cache,
        )

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None):
        """Pre-spike STDP (favorable, row-driven) -- mirrors :meth:`brainevent.CSR.update_on_pre`."""
        new = update_fixed_post_conn_on_binary_pre(
            self.data, self.indices, pre_spike, post_trace, w_min, w_max,
            shape=self.shape, backend=self.backend,
        )
        return self._rebuild_with_data(new)

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None):
        """Post-spike STDP (unfavorable) -- mirrors :meth:`brainevent.CSR.update_on_post` (perm-fused)."""
        mirror = self._mirror()
        if mirror is None:
            warn_mirror_fallback('FixedNumPerPre', 'update_on_post')
            new = _update_csr_on_binary_post_masked(
                self.data.reshape(-1), self.indices.reshape(-1), fixed_conn_num_csr_indptr(self.indices),
                pre_trace, post_spike, w_min, w_max, shape=self._a_shape,
            )
        else:
            csc_indptr, csc_indices, perm = mirror
            new = update_csr_on_binary_post(
                self.data.reshape(-1), csc_indices, csc_indptr, perm,
                pre_trace, post_spike, w_min, w_max,
                shape=self._a_shape, backend=self.backend,
            )
        return self._rebuild_with_data(new.reshape(self.data.shape))

    def _unitary_op(self, op):
        return self._same_structure(self._apply_data(op))

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            other = other.reshape(())
            return self._same_structure(op(self.data, other))
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            # One gathered value per stored connection, laid out like ``indices``.
            other = other[rows, cols].reshape(self.indices.shape)
            return self._same_structure(op(self.data, other))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            other = other.reshape(())
            return self._same_structure(op(other, self.data))
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_post_num_to_coo(self)
            # One gathered value per stored connection, laid out like ``indices``.
            other = other[rows, cols].reshape(self.indices.shape)
            return self._same_structure(op(other, self.data))
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

    data: ArrayData
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
        _validate_fixed_conn_indices(self.indices, expected_rows=shape[1], kind='Pre-synaptic')
        # Connection indices are secondary-axis coordinates (pre ids) and are
        # always stored int32; bounds are validated below by
        # ``_contains_invalid_indices`` so only a dtype coercion is needed here.
        self.indices = _as_int32_indices(
            self.indices, shape[0], 'FixedNumPerPost indices', check_values=False
        )
        if self.data.size != 1 and self.data.shape != self.indices.shape:
            raise ValueError(
                f"Data shape {self.data.shape} must match indices shape {self.indices.shape}. "
                f"But got {self.data.shape} != {self.indices.shape}"
            )
        super().__init__((self.data, self.indices), shape=shape, backend=backend, buffers=buffers)
        _contains_invalid_indices(self.indices, upper_bound=self.shape[0])
        if precompute_weight_indices and self._mirror() is None:
            # Traced indices: this fresh object is local to the trace.
            self._cache.set_mirror(self._weight_indices())

    # FixedNumPerPost ≡ CSC: stored structure shape is ``self.shape[::-1]``.
    @property
    def _a_shape(self):
        return tuple(self.shape)[::-1]

    def _ell_transpose(self, transpose_W: bool) -> bool:
        return not bool(transpose_W)

    @classmethod
    def fromdense(cls, mat, *, num_conn=None, backend=None) -> 'FixedNumPerPost':
        """Construct from a dense matrix with a fixed number of connections per post.

        Each post-synaptic column of ``mat`` is encoded with ``num_conn``
        incoming connections (pre-synaptic ids).

        Parameters
        ----------
        mat : jax.Array or brainunit.Quantity
            Dense ``(num_pre, num_post)`` matrix. Explicit zeros are treated as
            absent connections.
        num_conn : int, optional
            Connections per post-synaptic neuron. If ``None`` (default), inferred
            from the matrix, which must then have a uniform per-column non-zero
            count; otherwise short columns are padded with a zero-weight sentinel
            and a column with more non-zeros than ``num_conn`` raises
            ``ValueError``.
        backend : str, optional
            Backend tag forwarded to the constructor.

        Returns
        -------
        FixedNumPerPost
            Encoded fixed-connection matrix, ``shape`` equal to ``mat.shape``.

        Raises
        ------
        ValueError
            If ``mat`` is not 2-D, if per-column counts are non-uniform and
            ``num_conn`` is omitted, or if a column exceeds ``num_conn``.

        See Also
        --------
        tocsc : Inverse-direction conversion to CSC.
        brainevent.CSC.fromdense : For irregular matrices.
        """
        mantissa = np.asarray(u.get_mantissa(mat))
        if mantissa.ndim != 2:
            raise ValueError(f"FixedNumPerPost.fromdense expects a 2-D matrix; got {mantissa.ndim}-D.")
        num_pre, num_post = mantissa.shape
        # Encode by post-synaptic column: each post row gathers its pre ids.
        data, indices = _ell_from_rows(mantissa.T, num_conn, kind=cls.__name__)
        data_q = u.maybe_decimal(u.Quantity(jnp.asarray(data), unit=u.get_unit(mat)))
        return cls((data_q, jnp.asarray(indices)), shape=(num_pre, num_post), backend=backend)

    def _to_coo(self):
        return fixed_pre_num_to_coo(self)

    def _csr_index(self):
        # Column-major ELL stores ``W^T`` row-major, so the compact CSC structure
        # of the stored operand (built and cached by ``_weight_indices``) is
        # exactly the CSR of ``W``; ``perm`` maps each CSR slot to the position
        # of its weight in the flattened ELL ``data``.
        csr_indptr, csr_indices, perm = self._weight_indices()
        return csr_indptr, csr_indices, perm

    def with_data(self, data: ArrayData) -> 'FixedNumPerPost':
        """Return a new matrix with the same connectivity and replaced values."""
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return self._same_structure(data)

    def __getitem__(self, index):
        """Extract rows of ``W`` as a dense array.

        Row slicing is the *unfavorable* direction for :class:`FixedNumPerPost`
        (which stores ``W`` column-major), so it reuses the cached CSR-of-``W``
        view from :meth:`_weight_indices` and the shared ``csr_slice_rows``
        kernel.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0 (pre-synaptic). Negative indices wrap;
            Python slices are supported; concrete OOB indices raise ``IndexError``.
            A tuple is a list of rows: ``M[1, 2]`` returns rows 1 and 2, not the
            NumPy element ``(1, 2)``.

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

        Both orientations store the same structure (``_a_shape`` is unchanged),
        so the transpose shares the cached mirror.
        """
        assert axes is None, "transpose does not support axes argument."
        return FixedNumPerPre._from_parts(
            self.data, self.indices, shape=self.shape[::-1], backend=self.backend, cache=self._cache,
        )

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None):
        """Pre-spike STDP (unfavorable) -- mirrors :meth:`brainevent.CSC.update_on_pre` (perm-fused)."""
        mirror = self._mirror()
        if mirror is None:
            warn_mirror_fallback('FixedNumPerPost', 'update_on_pre')
            new = _update_csr_on_binary_post_masked(
                self.data.reshape(-1), self.indices.reshape(-1), fixed_conn_num_csr_indptr(self.indices),
                post_trace, pre_spike, w_min, w_max, shape=self._a_shape,
            )
        else:
            csc_indptr, csc_indices, perm = mirror
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
        return self._same_structure(self._apply_data(op))

    def _binary_op(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            other = other.reshape(())
            return self._same_structure(op(self.data, other))
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            # One gathered value per stored connection, laid out like ``indices``.
            other = other[rows, cols].reshape(self.indices.shape)
            return self._same_structure(op(self.data, other))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op):
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            other = other.reshape(())
            return self._same_structure(op(other, self.data))
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols, _ = fixed_pre_num_to_coo(self)
            # One gathered value per stored connection, laid out like ``indices``.
            other = other[rows, cols].reshape(self.indices.shape)
            return self._same_structure(op(other, self.data))
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
