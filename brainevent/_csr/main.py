# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
import contextlib
from typing import Optional, Union, Dict, cast

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._data import DataRepresentation
from brainevent._event import as_event_operand
from brainevent._misc import (
    _csr_to_coo, _csr_todense, csr_to_csc_index, csr_to_csc_index_traced, normalize_row_index, build_sub_csr,
    _as_int32_indices, _as_indptr, _check_compressed_structure,
)
from brainevent._structure import StructureCache, is_concrete, nnz_hint, warn_mirror_fallback
from brainevent._typing import ArrayData, Data, Indptr, Index, MatrixShape
from .binary.binary import binary_csrmv, binary_csrmm
from .binary.binary_indexed import binary_csrmv_indexed, binary_csrmm_indexed
from .diag.diag_add import csr_diag_position, csr_diag_add
from .float.float import csrmv, csrmm
from .plasticity.plasticity_binary import (
    update_csr_on_binary_pre, update_csr_on_binary_post, _update_csr_on_binary_post_masked,
)
from .plasticity.plasticity_generic import csr_apply_on_pre, csr_apply_on_post
from .slice.slice import csr_slice_rows
from .solve.spsolve import csr_solve
from .dt2t.dt2t import csrmv_dt2t

__all__ = [
    'CSR',
    'CSC',
]


def _use_cuda_indexed_binary_route(backend: Optional[str]) -> bool:
    return backend == "cuda_raw"


class CompressedSparseData(DataRepresentation):
    """
    Abstract base class for compressed sparse matrix formats.

    ``CompressedSparseData`` provides the common interface shared by :class:`CSR` and
    :class:`CSC`. It inherits from ``brainunit.sparse.SparseMatrix`` and
    adds arithmetic operators, JAX pytree support, and helper methods for
    event-driven neural simulation.

    Subclasses must implement :meth:`apply`, :meth:`_binary_op`,
    :meth:`_binary_rop`, :meth:`todense`, :meth:`with_data`,
    :meth:`fromdense`, :meth:`dt2t`, and :meth:`dt2t_transposed`. Both
    concrete subclasses (:class:`CSR`, :class:`CSC`) also provide the common
    conversion contract :meth:`tocsr`, :meth:`tocsc`, and :meth:`tocoo`.

    Parameters
    ----------
    data : array_like or Sequence
        Non-zero values, or a length-3 sequence ``(data, indices, indptr)``.
    indices : array_like, optional
        Secondary-axis indices for each stored element.
    indptr : array_like, optional
        Primary-axis pointers into ``data`` and ``indices``.
    shape : tuple[int, int]
        Matrix shape ``(num_rows, num_columns)``.
    backend : str or None, optional
        Compute backend used by the matrix operations. Default ``None``.
    buffers : dict, optional
        Precomputed structure caches under their legacy names: the mirror
        layout (``'csc'`` for CSR, ``'csr'`` for CSC) and ``'diag_positions'``.
    precompute_weight_indices : bool, optional
        Build the mirror layout now (see :meth:`build_weight_indices`).
        Default ``False``.

    Attributes
    ----------
    data : Data
        Array of stored (non-zero) values.
    indices : Index
        Secondary-axis index array.
    indptr : Indptr
        Primary-axis pointer array.
    shape : tuple[int, int]
        Shape of the full matrix.

    Notes
    -----
    The matrix is a JAX pytree whose leaves are ``data``, ``indices``,
    ``indptr`` and, when cached, the three mirror arrays and the four
    ``diag_add`` plan arrays. The structure is therefore traced like any
    other input under ``jax.jit``: no index array is embedded in the compiled
    program, and two matrices with equal shapes share one compilation.
    Differentiate with respect to ``M.data`` and rebuild with
    :meth:`with_data`; ``jax.grad(f, allow_int=True)(M)`` also works and
    returns ``float0`` cotangents for the index leaves.

    See Also
    --------
    CSR : Compressed Sparse Row implementation.
    CSC : Compressed Sparse Column implementation.
    """

    data: ArrayData
    indices: Index
    indptr: Indptr
    shape: MatrixShape

    # Overridden by concrete subclasses (CSR -> 'csr', CSC -> 'csc'); selects
    # which shape axis is the secondary (coordinate) dimension for indices.
    _compressed_format = 'csr'
    # Legacy ``buffers`` name of the mirror layout (CSR -> 'csc', CSC -> 'csr').
    _mirror_key = 'csc'

    def __init__(
        self,
        data,
        indices=None,
        indptr=None,
        *,
        shape: MatrixShape,
        backend: Optional[str] = None,
        buffers: Optional[Dict] = None,
        indptr_dtype="auto",
        check_structure: bool = True,
        precompute_weight_indices: bool = False,
    ):
        if indices is None and indptr is None:
            # Tuple syntax: CSR((data, indices, indptr), shape=...)
            args = data
        else:
            # Positional syntax: CSR(data, indices, indptr, shape=...)
            args = (data, indices, indptr)

        assert len(args) == 3, "Expected three arguments: data, indices, indptr."
        data_arr, indices_arg, indptr_arg = args

        fmt = type(self)._compressed_format
        # ``indices`` are secondary-axis coordinates: CSR columns / CSC rows.
        secondary_dim = shape[1] if fmt == 'csr' else shape[0]
        context = f"{fmt.upper()} constructor"

        self.data = u.math.asarray(data_arr)
        # ``indices`` are always int32; validate values only when a full
        # structure check is requested (structure-preserving reconstructions
        # pass check_structure=False to skip redundant host readback).
        self.indices = _as_int32_indices(
            indices_arg, secondary_dim, context, check_values=check_structure
        )
        # ``indptr`` auto-promotes to int64 only when nnz exceeds int32 range.
        nse = self.indices.shape[0]
        self.indptr = _as_indptr(indptr_arg, nse, indptr_dtype, context)
        if check_structure:
            data_shape = tuple(getattr(self.data, 'shape', ()))
            if data_shape not in ((), (1,), (nse,)):
                raise ValueError(
                    f"{context}: data must be a scalar or have shape (1,) (homogeneous "
                    f"weight), or have shape (nse,) = ({nse},), got {data_shape}."
                )
            _check_compressed_structure(
                self.indices, self.indptr, shape, format=fmt, check_values=True
            )

        self.backend = backend
        super().__init__((self.data, self.indices, self.indptr), shape=shape)
        self._cache = StructureCache(primary_hint=nnz_hint(self.indptr))
        self._ingest_buffers(buffers)
        if precompute_weight_indices and self._mirror() is None:
            # Traced structure: this fresh object is local to the trace, so
            # its own cache may hold the traced mirror.
            self._cache.set_mirror(self._weight_indices())

    @classmethod
    def _from_parts(cls, data, indices, indptr, *, shape, backend=None, buffers=None, cache=None):
        """Assemble a matrix from trusted parts, without conversion or validation.

        Structure-preserving paths (``with_data``, ``transpose``, ``apply``,
        data-only arithmetic) pass the ``cache`` of the source matrix so the
        derived arrays stay shared; pytree ``tree_unflatten`` passes a cache
        rebuilt from its leaves. Leaves may be anything JAX puts in a pytree
        (tracers, ``None``, ``float0`` arrays, abstract values), so nothing
        here may inspect their values.
        """
        obj = object.__new__(cls)
        obj.data = data
        obj.indices = indices
        obj.indptr = indptr
        obj.shape = tuple(int(s) for s in shape)
        obj.backend = backend
        obj._buffer_registry = set()
        obj._cache = StructureCache(primary_hint=nnz_hint(indptr)) if cache is None else cache
        obj._ingest_buffers(buffers)
        return obj

    def _ingest_buffers(self, buffers) -> None:
        """Store legacy-named structure caches (mirror, ``diag_positions``)."""
        for name, value in (buffers or {}).items():
            if value is None:
                continue
            if name == self._mirror_key:
                self._cache.set_mirror(value)
            elif name == 'diag_positions':
                self._cache.diag_plan = tuple(value)
            else:
                raise ValueError(
                    f"{type(self).__name__} does not support buffer {name!r}; the known "
                    f"buffers are {self._mirror_key!r} and 'diag_positions'."
                )

    def register_buffer(self, name, value=None):
        """Store a structure cache under its legacy name (see :attr:`buffers`)."""
        self._ingest_buffers({name: value})

    def set_buffer(self, name, value):
        """Store a structure cache under its legacy name (see :attr:`buffers`)."""
        self._ingest_buffers({name: value})

    @property
    def buffers(self) -> Dict:
        """Structure caches under their legacy names (a read-only snapshot).

        Holds the mirror layout ``(indptr, indices, perm)`` under ``'csc'`` for
        a CSR matrix or ``'csr'`` for a CSC matrix, and the ``diag_add`` plan
        under ``'diag_positions'``, when they have been built.
        """
        out = {}
        if self._cache.mirror is not None:
            out[self._mirror_key] = self._cache.mirror
        if self._cache.diag_plan is not None:
            out['diag_positions'] = self._cache.diag_plan
        return out

    @property
    def diag_positions(self):
        """The cached ``diag_add`` plan (``AttributeError`` until it is built)."""
        plan = self._cache.diag_plan
        if plan is None:
            raise AttributeError('diag_positions')
        return plan

    @property
    def _storage_shape(self) -> MatrixShape:
        """Shape of the matrix whose CSR arrays are stored (``W`` or ``W.T``)."""
        rows, cols = self.shape
        return (rows, cols) if self._compressed_format == 'csr' else (cols, rows)

    def _mirror(self):
        """Return the mirror ``(indptr, indices, perm)``, or ``None`` if traced.

        A concrete structure builds the mirror once and caches it, even under
        an active trace (the arrays are computed eagerly, so they are not
        tracers). A traced structure without a carried mirror returns ``None``
        so the caller can take a mirror-free path.
        """
        cache = self._cache
        if cache.mirror is not None:
            return cache.mirror
        if not is_concrete(self.indices, self.indptr):
            return None
        with jax.ensure_compile_time_eval():
            mirror = csr_to_csc_index(self.indptr, self.indices, shape=self._storage_shape)
        return cache.set_mirror(mirror)

    def _weight_indices(self):
        """Return the mirror ``(indptr, indices, perm)``, computing it if needed.

        Unlike :meth:`_mirror` this always returns arrays: with a traced
        structure and no carried mirror it computes one with
        :func:`brainevent._misc.csr_to_csc_index_traced` (not cached).
        """
        mirror = self._mirror()
        if mirror is None:
            mirror = csr_to_csc_index_traced(self.indptr, self.indices, shape=self._storage_shape)
        return mirror

    def build_weight_indices(self):
        """Return a matrix sharing these arrays with the mirror layout built.

        The mirror is the other compression of the structure (column-major for
        CSR, row-major for CSC) plus the permutation back to the stored
        ``data`` order. It drives the event-driven kernels of the unfavorable
        product direction and the unfavorable plasticity update. It is built
        lazily on first use when the structure is concrete, so call this
        before ``jax.jit`` when the matrix enters the jitted function as an
        argument.

        Returns
        -------
        CSR or CSC
            A matrix of the same class sharing ``data``, ``indices`` and
            ``indptr``, with the mirror cached (and carried as pytree leaves).

        Examples
        --------
        .. code-block:: python

            >>> import jax
            >>> import jax.numpy as jnp
            >>> import brainevent
            >>> W = brainevent.CSR.fromdense(jnp.array([[1., 0.], [2., 3.]]))
            >>> W = W.build_weight_indices()
            >>> f = jax.jit(lambda M, s: M @ brainevent.BinaryArray(s))
            >>> y = f(W, jnp.array([True, False]))
        """
        mirror = self._mirror()
        if mirror is not None:
            cache = self._cache
        else:
            cache = StructureCache(primary_hint=self._cache.primary_hint, diag_plan=self._cache.diag_plan)
            cache.set_mirror(self._weight_indices())
        return type(self)._from_parts(
            self.data, self.indices, self.indptr, shape=self.shape, backend=self.backend, cache=cache,
        )

    def _same_structure(self, data):
        """A matrix of the same class and structure (shared caches) holding ``data``."""
        return type(self)._from_parts(
            data, self.indices, self.indptr, shape=self.shape, backend=self.backend, cache=self._cache,
        )

    @property
    def nse(self):
        """
        Number of stored elements in the sparse matrix.

        This counts all explicitly stored entries, including any stored
        zeros.  It equals ``self.indices.size``.

        Returns
        -------
        int
            The number of stored elements.
        """
        return self.indices.size

    @property
    def dtype(self):
        """
        Data type of the stored values.

        Returns
        -------
        numpy.dtype
            The dtype of ``self.data``.
        """
        return self.data.dtype

    def sum(self, axis=None):
        """Return the sum of all logically represented matrix entries.

        Parameters
        ----------
        axis : int, sequence of int, or None, optional
            Axis or axes to reduce. Only ``None`` is currently supported.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Scalar sum of every represented sparse entry. A homogeneous
            size-one value is counted once per stored entry.

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

    def _stored_event_product(self, x, weighted: bool, *, transpose: bool):
        """Event product with the stored CSR matrix ``S`` (``W`` for CSR, ``W.T`` for CSC).

        Computes ``S @ x`` (``transpose=False``) or ``S.T @ x``; ``x`` is a
        vector or a matrix whose columns are the batch. ``S.T @ x`` scatters
        the active events over the stored rows. ``S @ x`` gathers, except on
        ``cuda_raw``, which scatters over the cached mirror layout instead.
        Without a mirror and with a traced structure that falls back to the
        gather, with a one-time :class:`brainevent.PerformanceWarning`.
        """
        shape = self._storage_shape
        matrix = x.ndim == 2
        mirror = None
        if not transpose and _use_cuda_indexed_binary_route(self.backend):
            mirror = self._mirror()
            if mirror is None:
                warn_mirror_fallback(type(self).__name__, 'the unfavorable event product')
        if mirror is None:
            primary = binary_csrmm if matrix else binary_csrmv
            return primary(
                self.data, self.indices, self.indptr, x,
                shape=shape, transpose=transpose, backend=self.backend, weighted=weighted,
                max_row_nnz=self._cache.primary_hint,
            )
        mirror_indptr, mirror_indices, perm = mirror
        indexed = binary_csrmm_indexed if matrix else binary_csrmv_indexed
        return indexed(
            self.data, mirror_indices, mirror_indptr, perm, x,
            shape=(shape[1], shape[0]), transpose=True, backend=self.backend, weighted=weighted,
            max_row_nnz=self._cache.mirror_hint,
        )

    def tree_flatten(self):
        """
        Flatten this sparse matrix into JAX pytree leaves and auxiliary data.

        This method is part of the JAX pytree protocol. ``data``, ``indices``
        and ``indptr`` are leaves, followed by the cached mirror arrays and
        ``diag_add`` plan when present. The auxiliary data holds only hashable
        metadata, so structurally equal matrices share one jit cache entry.

        Returns
        -------
        children : tuple
            ``(data, indices, indptr, *mirror, *diag_plan)``.
        aux_data : tuple
            ``(shape, backend, has_mirror, has_diag_plan, primary_hint,
            mirror_hint)``.

        See Also
        --------
        tree_unflatten : Reconstruct a sparse matrix from flattened data.
        """
        cache = self._cache
        mirror = cache.mirror or ()
        plan = cache.diag_plan or ()
        children = (self.data, self.indices, self.indptr, *mirror, *plan)
        aux = (self.shape, self.backend, bool(mirror), bool(plan), cache.primary_hint, cache.mirror_hint)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a sparse matrix from JAX pytree leaves and auxiliary data.

        This is the inverse of :meth:`tree_flatten` and is called by JAX
        when unflattening a pytree (e.g., after ``jax.jit`` compilation).

        No validation or conversion is performed, so the leaves may be
        tracers, ``None``, ``float0`` arrays or other placeholders.

        Parameters
        ----------
        aux_data : tuple
            Auxiliary data produced by :meth:`tree_flatten`.
        children : tuple
            Leaves produced by :meth:`tree_flatten`.

        Returns
        -------
        CompressedSparseData
            A new instance of the sparse matrix class with restored
            attributes.

        See Also
        --------
        tree_flatten : Flatten a sparse matrix for JAX pytree handling.
        """
        shape, backend, has_mirror, has_plan, primary_hint, mirror_hint = aux_data
        expected = 3 + 3 * has_mirror + 4 * has_plan
        if len(children) != expected:
            raise ValueError(f'{cls.__name__}.tree_unflatten expects {expected} leaves, got {len(children)}.')
        data, indices, indptr = children[:3]
        mirror = tuple(children[3:6]) if has_mirror else None
        plan = tuple(children[3 + 3 * has_mirror:]) if has_plan else None
        cache = StructureCache(primary_hint, mirror, mirror_hint, plan)
        return cls._from_parts(data, indices, indptr, shape=shape, backend=backend, cache=cache)

    def apply(self, fn):
        """
        Apply a function to the data and return a new sparse matrix with the same structure.

        Unlike :meth:`with_data`, which requires the new data to have the same
        shape, dtype, and unit, ``apply`` allows transformations that change
        dtype or unit.

        Parameters
        ----------
        fn : callable
            A function to apply to ``self.data``.

        Returns
        -------
        CSR or CSC
            A new sparse matrix with ``fn(self.data)`` and the same structure.
        """
        raise NotImplementedError

    def __abs__(self):
        """
        Return a sparse matrix with element-wise absolute values.

        Computes ``abs(x)`` for every stored element *x* while preserving the
        sparsity structure (indices, indptr, and shape).

        Returns
        -------
        CSR or CSC
            A new sparse matrix whose data values are the absolute values of
            the original stored elements.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([-1.0, 2.0, -3.0]), indices, indptr, shape=(3, 3))
            >>> result = abs(csr)
        """
        return self.apply(operator.abs)

    def __neg__(self):
        """
        Return a sparse matrix with element-wise negation.

        Computes ``-x`` for every stored element *x* while preserving the
        sparsity structure.

        Returns
        -------
        CSR or CSC
            A new sparse matrix whose data values are the negation of the
            original stored elements.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 3.0]), indices, indptr, shape=(3, 3))
            >>> result = -csr
        """
        return self.apply(operator.neg)

    def __pos__(self):
        """
        Return a sparse matrix with the unary positive operator applied.

        Computes ``+x`` for every stored element *x*.  For numeric types this
        is typically the identity, but the operator may trigger type promotion
        for certain array-like objects.

        Returns
        -------
        CSR or CSC
            A new sparse matrix whose data values are ``+self.data``.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 3.0]), indices, indptr, shape=(3, 3))
            >>> result = +csr
        """
        return self.apply(operator.pos)

    def _binary_op(self, other, op):
        raise NotImplementedError

    def apply2(self, other, fn, *, reverse: bool = False):
        """
        Apply a binary function while preserving sparse structure semantics.

        Parameters
        ----------
        other : Any
            Right-hand operand for normal operations, or left-hand operand when
            ``reverse=True``.
        fn : callable
            Binary function from ``operator`` or a compatible callable.
        reverse : bool, optional
            If False, compute ``fn(self, other)`` semantics using ``_binary_op``.
            If True, compute ``fn(other, self)`` semantics using ``_binary_rop``.
            Defaults to False.

        Returns
        -------
        CSR or CSC or Data
            Result of the operation.
        """
        if reverse:
            return self._binary_rop(other, fn)
        return self._binary_op(other, fn)

    def __mul__(self, other: Data):
        """
        Element-wise multiplication: ``self * other``.

        When ``other`` is a scalar, each stored element is multiplied by that
        scalar.  When ``other`` is a dense matrix of the same shape, only the
        values at the stored positions are multiplied.  When ``other`` is a
        sparse matrix with identical structure (same ``indices`` and
        ``indptr`` objects), the data arrays are multiplied directly.

        Parameters
        ----------
        other : Data
            Scalar, dense array, or structurally identical sparse matrix.

        Returns
        -------
        CSR or CSC
            A new sparse matrix containing the element-wise product.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix with different structure, or a
            dense array whose shape is incompatible.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 3.0]), indices, indptr, shape=(3, 3))
            >>> result = csr * 2.0
        """
        return self.apply2(other, operator.mul)

    def __truediv__(self, other):
        """
        Element-wise true division: ``self / other``.

        Divides every stored element by ``other``.  Semantics for scalar,
        dense, and structurally identical sparse operands match those of
        :meth:`__mul__`.

        Parameters
        ----------
        other : Data
            Scalar, dense array, or structurally identical sparse matrix.

        Returns
        -------
        CSR or CSC
            A new sparse matrix containing the element-wise quotient.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix with different structure, or a
            dense array whose shape is incompatible.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([2.0, 4.0, 6.0]), indices, indptr, shape=(3, 3))
            >>> result = csr / 2.0
        """
        return self.apply2(other, operator.truediv)

    def __add__(self, other):
        """
        Element-wise addition: ``self + other``.

        For addition and subtraction the sparse matrix is first converted to a
        dense matrix via :meth:`todense`, so the result is always a dense
        array.

        Parameters
        ----------
        other : array_like
            Dense array with a shape broadcastable to ``self.shape``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Dense result of the addition.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0]), indices, indptr, shape=(3, 3))
            >>> dense = jnp.ones((3, 3))
            >>> result = csr + dense
        """
        return self.apply2(other, operator.add)

    def __sub__(self, other):
        """
        Element-wise subtraction: ``self - other``.

        The sparse matrix is first converted to a dense matrix via
        :meth:`todense`, so the result is always a dense array.

        Parameters
        ----------
        other : array_like
            Dense array with a shape broadcastable to ``self.shape``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Dense result of the subtraction.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0]), indices, indptr, shape=(3, 3))
            >>> dense = jnp.ones((3, 3))
            >>> result = csr - dense
        """
        return self.apply2(other, operator.sub)

    def _binary_rop(self, other, op):
        raise NotImplementedError

    def __rmul__(self, other: Data):
        """
        Reflected element-wise multiplication: ``other * self``.

        Called when the left operand does not support multiplication with this
        sparse type.  Semantics are identical to :meth:`__mul__` because
        multiplication is commutative for scalars and element-wise operations.

        Parameters
        ----------
        other : Data
            Scalar, dense array, or structurally identical sparse matrix.

        Returns
        -------
        CSR or CSC
            A new sparse matrix containing the element-wise product.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix with different structure, or a
            dense array whose shape is incompatible.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 3.0]), indices, indptr, shape=(3, 3))
            >>> result = 2.0 * csr
        """
        return self.apply2(other, operator.mul, reverse=True)

    def __rtruediv__(self, other):
        """
        Reflected element-wise true division: ``other / self``.

        Computes ``other / x`` for every stored element *x*.  Note that this
        is **not** equivalent to ``self / other``; the operand order matters.

        Parameters
        ----------
        other : Data
            Scalar, dense array, or structurally identical sparse matrix.

        Returns
        -------
        CSR or CSC
            A new sparse matrix containing the element-wise quotient.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix with different structure, or a
            dense array whose shape is incompatible.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0, 4.0]), indices, indptr, shape=(3, 3))
            >>> result = 8.0 / csr  # stored values become [8.0, 4.0, 2.0]
        """
        return self.apply2(other, operator.truediv, reverse=True)

    def __radd__(self, other):
        """
        Reflected element-wise addition: ``other + self``.

        The sparse matrix is first converted to dense, so the result is always
        a dense array.  Because addition is commutative the result equals
        ``self + other``.

        Parameters
        ----------
        other : array_like
            Dense array with a shape broadcastable to ``self.shape``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Dense result of the addition.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0]), indices, indptr, shape=(3, 3))
            >>> dense = jnp.ones((3, 3))
            >>> result = dense + csr
        """
        return self.apply2(other, operator.add, reverse=True)

    def __rsub__(self, other):
        """
        Reflected element-wise subtraction: ``other - self``.

        The sparse matrix is first converted to dense, so the result is always
        a dense array.  Note that ``other - self`` is **not** equivalent to
        ``self - other``.

        Parameters
        ----------
        other : array_like
            Dense array with a shape broadcastable to ``self.shape``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Dense result of the subtraction.

        Examples
        --------
        .. code-block:: python

            >>> csr = CSR(jnp.array([1.0, 2.0]), indices, indptr, shape=(3, 3))
            >>> dense = jnp.ones((3, 3))
            >>> result = dense - csr
        """
        return self.apply2(other, operator.sub, reverse=True)

    def dt2t(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity]
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute a sparse matrix-vector product mapping y-w space to w space.

        This method is used in event-driven neural simulations to efficiently
        compute the effect of synaptic connections.  Given a per-target array
        ``y_dim_arr`` and a per-synapse weight array ``w_dim_arr``, it
        performs a specialised sparse product that accumulates contributions
        along the connectivity defined by this matrix.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the weight / synapse dimension.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Result of the sparse transformation, preserving physical units
            when present.
        """
        raise NotImplementedError

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity]
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute the transposed sparse matrix-vector product mapping y-w space to w space.

        This is the adjoint of :meth:`dt2t`.  It is useful for
        back-propagation or adjoint computations in event-driven neural
        simulations.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the weight / synapse dimension.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Result of the transposed sparse transformation, preserving
            physical units when present.

        See Also
        --------
        dt2t : The forward (non-transposed) variant.
        """
        raise NotImplementedError

    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=jnp.int32):
        """
        Create a compressed sparse matrix from a dense matrix.

        Parameters
        ----------
        mat : array_like
            The dense matrix to convert.
        nse : int, optional
            Number of stored (non-zero) elements.  If ``None`` it is
            inferred from ``mat``.
        index_dtype : dtype, optional
            Data type for index arrays.  Defaults to ``jnp.int32``.

        Returns
        -------
        CSR or CSC
            A new sparse matrix in the appropriate compressed format.
        """
        raise NotImplementedError

    def with_data(self, data: ArrayData):
        """
        Create a new sparse matrix with the same structure but different data.

        Unlike :meth:`apply`, the new ``data`` must have the same shape,
        dtype, and unit as the original.

        Parameters
        ----------
        data : Data
            Replacement data array.

        Returns
        -------
        CSR or CSC
            New sparse matrix sharing indices and indptr with this instance.

        Raises
        ------
        AssertionError
            If the shape, dtype, or unit of ``data`` does not match the
            original.
        """
        raise NotImplementedError

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the sparse matrix to a dense array.

        Returns
        -------
        jax.Array or brainunit.Quantity
            A dense ``(num_rows, num_columns)`` matrix equivalent to this
            sparse matrix.
        """
        raise NotImplementedError

    def diag_add(self, other):
        """
        Add values to the matrix diagonal and return a new sparse matrix.

        Computes ``A + diag(other)`` exactly. Diagonal entries that are missing
        from the current sparsity pattern are **inserted**, so in general the
        returned matrix has a different ``indices``/``indptr`` (and a larger
        ``nse``) than ``self``: its pattern is the union of ``self``'s pattern
        and the full main diagonal ``{(i, i) : 0 <= i < min(shape)}``.

        Parameters
        ----------
        other : array-like
            The diagonal values to add, with one entry per diagonal element
            (i.e. length ``min(self.shape)``). It should share the dtype and be
            unit-compatible with the matrix's non-zero elements.

        Returns
        -------
        CSR or CSC
            A new sparse matrix of the same class and shape holding
            ``A + diag(other)`` with the diagonal-augmented sparsity pattern.

        Raises
        ------
        AssertionError
            If ``other`` is an instance of ``JAXSparse``, as this operation does
            not support sparse operands.
        ValueError
            If the structure is traced and no plan was prepared (see
            :meth:`prepare_diag_add`).

        Notes
        -----
        - The structural plan describing where existing entries move and where
          diagonals are inserted depends only on the sparsity structure, so it is
          computed once (on the host, from a concrete structure) and cached. The
          returned matrix already has its full diagonal present, so it carries a
          matching plan that subsequent ``diag_add`` calls reuse without
          recomputation, also under ``jax.jit``. For a matrix that enters
          ``jax.jit`` as an argument, call :meth:`prepare_diag_add` first.
        - Structure-dependent caches (e.g. the transposed-view weight indices) are
          intentionally not propagated, since the new matrix has a different
          sparsity pattern.
        - This method relies on :func:`csr_diag_position` to plan the structural
          change and :func:`csr_diag_add` to compute the augmented values.
        """
        assert not isinstance(other, u.sparse.SparseMatrix), "diag_add does not support JAXSparse objects."
        plan = self._diag_plan()
        new_data = csr_diag_add(self.data, plan, other)
        new_indptr, new_indices, _, diag_dest = plan
        # The result already contains the full diagonal, so its own plan is an
        # identity relocation of every stored element plus the same diagonal
        # destinations -- reusable by a later diag_add without touching Numba.
        # Computed eagerly when the plan is concrete so it can be cached.
        with jax.ensure_compile_time_eval() if is_concrete(new_indices) else contextlib.nullcontext():
            identity = jnp.arange(new_indices.shape[0], dtype=new_indices.dtype)
        cache = StructureCache(
            primary_hint=nnz_hint(new_indptr),
            diag_plan=(new_indptr, new_indices, identity, diag_dest),
        )
        # The augmented structure is valid by construction (numba planner), so
        # skip re-validation. ``csr_diag_position`` already raised for the
        # int64-offset case, so ``new_indptr`` fits int32 here.
        return type(self)._from_parts(
            new_data, new_indices, new_indptr, shape=self.shape, backend=self.backend, cache=cache,
        )

    def _diag_plan(self):
        """Return the cached ``diag_add`` plan, building it from a concrete structure."""
        cache = self._cache
        if cache.diag_plan is None:
            if not is_concrete(self.indices, self.indptr):
                raise ValueError(
                    f"{type(self).__name__}.diag_add needs a structural plan that can only be "
                    f"computed from a concrete structure. Call `M = M.prepare_diag_add()` "
                    f"before `jax.jit` when the matrix is a jitted-function argument."
                )
            with jax.ensure_compile_time_eval():
                plan = csr_diag_position(self.indptr, self.indices, shape=self.shape)
                cache.diag_plan = tuple(jnp.asarray(a) for a in plan)
        return cache.diag_plan

    def prepare_diag_add(self):
        """Return a matrix sharing these arrays with the :meth:`diag_add` plan built.

        :meth:`diag_add` may insert missing diagonal entries, which changes the
        structure; where they go is planned on the host from the concrete
        structure. The plan is carried as pytree leaves, so after this call
        ``diag_add`` also works when the matrix is an argument of ``jax.jit``.

        Returns
        -------
        CSR or CSC
            A matrix of the same class sharing ``data``, ``indices`` and
            ``indptr``, with the plan cached.

        Raises
        ------
        ValueError
            If the structure is traced.

        Examples
        --------
        .. code-block:: python

            >>> import jax
            >>> import jax.numpy as jnp
            >>> import brainevent
            >>> W = brainevent.CSR.fromdense(jnp.array([[1., 0.], [2., 0.]]))
            >>> W = W.prepare_diag_add()
            >>> f = jax.jit(lambda M, d: M.diag_add(d).todense())
            >>> f(W, jnp.array([10., 20.]))
            Array([[11.,  0.],
                   [ 2., 20.]], dtype=float32)
        """
        self._diag_plan()
        return self._same_structure(self.data)

    def solve(self, b: Union[jax.Array, u.Quantity]) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system ``A x = b`` where ``A`` is this sparse matrix.

        Parameters
        ----------
        b : jax.Array or brainunit.Quantity
            Right-hand side vector of the linear system.  Its first dimension
            must match ``self.shape[0]``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Solution vector *x* satisfying ``A x = b``.

        Raises
        ------
        AssertionError
            If the first dimension of ``b`` does not match the number of rows
            in the matrix.
        """
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class CSR(CompressedSparseData):
    """
    Event-driven and Unit-aware Compressed Sparse Row (CSR) matrix.

    This class represents a sparse matrix in CSR format, which is efficient for
    row-wise operations and matrix-vector multiplications. It is compatible with
    JAX's tree utilities and supports unit-aware computations.

    The class supports arithmetic with scalars and dense arrays, plus sparse-dense
    matrix multiplication via ``@``. Sparse-sparse operations are limited.

    Attributes
    ----------
    data : Data
        Array of the non-zero values in the matrix.
    indices : jax.Array
        Array of column indices for the non-zero values.
    indptr : jax.Array
        Array of row pointers indicating where each row starts in the data and indices arrays.
    shape : tuple[int, int]
        The shape of the matrix as (rows, columns).
    nse : int
        Number of stored elements (non-zero entries).
    dtype : dtype
        Data type of the matrix values.

    Notes
    -----
    In CSR format a matrix of shape ``(m, n)`` is stored as three arrays:

    * ``indptr`` of length ``m + 1`` -- the *i*-th row occupies entries
      ``indptr[i]`` to ``indptr[i+1]`` in the ``data`` and ``indices``
      arrays.
    * ``indices`` -- column indices of the stored elements.
    * ``data`` -- the corresponding non-zero values.

    The ``@`` operator dispatches to optimised kernels depending on the
    right-hand operand type:

    * :class:`~brainevent.BinaryArray` -- event-driven binary CSR MV/MM.
    * :class:`~brainevent.WeightedArray` -- event-driven value-weighted CSR MV/MM.
    * Dense ``jax.Array`` / ``brainunit.Quantity`` -- standard float CSR
      MV/MM with automatic dtype promotion.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import brainevent

        data    = jnp.array([1.0, 2.0, 3.0])
        indices = jnp.array([0, 2, 1])
        indptr  = jnp.array([0, 1, 2, 3])
        csr     = brainevent.CSR((data, indices, indptr), shape=(3, 3))

        # Sparse-dense matrix-vector product
        x = jnp.ones(3)
        y = csr @ x

    See Also
    --------
    CSC : Compressed Sparse Column format.
    """
    __module__ = 'brainevent'
    _compressed_format = 'csr'

    @classmethod
    def fromdense(
        cls,
        mat,
        *,
        nse: Optional[int] = None,
        index_dtype=jnp.int32,
        indptr_dtype="auto",
        backend: Optional[str] = None,
        precompute_weight_indices: bool = False,
    ) -> 'CSR':
        """
        Create a CSR matrix from a dense matrix.

        This method converts a dense matrix to a Compressed Sparse Row (CSR) format.

        Parameters
        ----------
        mat : array_like
            The dense matrix to be converted to CSR format.
        nse : int, optional
            The number of non-zero elements in the matrix. If None, it will be
            calculated from the input matrix.
        index_dtype : dtype, optional
            The data type for the ``indices`` (column) array. Must be ``int32``;
            ``indices`` are always int32. To control the ``indptr`` precision use
            ``indptr_dtype`` instead.
        indptr_dtype : {"auto", int32, int64}, optional
            The precision policy for the ``indptr`` offsets. ``"auto"`` (default)
            promotes to int64 only when the nnz exceeds the int32 range (and
            requires ``jax_enable_x64``).
        backend : str or None, optional
            Compute backend to attach to the matrix. Default ``None``.
        precompute_weight_indices : bool, optional
            If ``True``, eagerly build and cache the column-major (CSC-like)
            weight indices used by the *unfavorable* ``CSR @ event`` direction
            (see :meth:`build_weight_indices`). If ``False`` (default), the
            indices are built lazily on first use. Default ``False``.

        Returns
        -------
        CSR
            A new CSR matrix object created from the input dense matrix.

        See Also
        --------
        build_weight_indices : Eagerly build the cached weight indices.
        CSR._weight_indices : Lazily build/return the cached weight indices.

        Examples
        --------
        .. code-block:: python

            import jax.numpy as jnp
            import brainevent

            dense = jnp.array([[1.0, 0.0], [0.0, 2.0]])
            csr = brainevent.CSR.fromdense(dense)
        """
        if jnp.dtype(index_dtype) != jnp.dtype(jnp.int32):
            raise ValueError(
                "CSR.fromdense: `indices` are always int32; got index_dtype="
                f"{jnp.dtype(index_dtype)}. Use `indptr_dtype` to control the "
                "indptr precision instead."
            )
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csr = u.sparse.csr_fromdense(mat, nse=nse, index_dtype=jnp.int32)
        out = CSR(
            (csr.data, csr.indices, csr.indptr),
            shape=csr.shape, backend=backend, indptr_dtype=indptr_dtype,
        )
        if precompute_weight_indices:
            out = out.build_weight_indices()
        return out

    def with_data(self, data: ArrayData) -> 'CSR':
        """
        Create a new CSR matrix with updated data while keeping the same structure.

        This method creates a new CSR matrix instance with the provided data,
        maintaining the original indices, indptr, and shape.

        Parameters
        ----------
        data : Data
            The new data array to replace the existing data in the CSR matrix.
            It must have the same shape, dtype, and unit as the original data.

        Returns
        -------
        CSR
            A new CSR matrix instance with updated data and the same structure as the original.

        Raises
        ------
        AssertionError
            If the shape, dtype, or unit of the new data doesn't match the original data.

        Examples
        --------
        .. code-block:: python

            new_data = jnp.array([10.0, 20.0, 30.0])
            new_csr = csr.with_data(new_data)
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return self._same_structure(data)

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the CSR matrix to a dense matrix.

        This method transforms the compressed sparse row (CSR) representation
        into a full dense matrix.

        Returns
        -------
        jax.Array or brainunit.Quantity
            A dense matrix of shape ``self.shape`` containing all the values
            (including zeros) of the sparse matrix.

        Examples
        --------
        .. code-block:: python

            dense = csr.todense()
        """
        return _csr_todense(self.data, self.indices, self.indptr, shape=self.shape)

    def tocsr(self) -> 'CSR':
        """Return this matrix in CSR format (a no-op that returns ``self``).

        Provided for a uniform conversion interface across data
        representations; ``CSR`` is already row-compressed.

        Returns
        -------
        CSR
            ``self``, unchanged.

        See Also
        --------
        tocsc : Re-encode the same logical matrix column-major.
        tocoo : Convert to coordinate format.
        """
        return self

    def tocsc(self) -> 'CSC':
        """Re-encode the same logical matrix in CSC format.

        Unlike :meth:`transpose` (which reinterprets the arrays as ``W.T`` with
        swapped shape), ``tocsc`` returns a :class:`CSC` describing the *same*
        matrix ``W`` with the *same* ``shape`` -- the entries are resorted into
        column-major order.

        Returns
        -------
        CSC
            The same logical matrix in CSC format, ``shape`` unchanged.

        See Also
        --------
        tocsr : Identity conversion.
        transpose : Logical transpose (swaps ``shape``).
        """
        csc_indptr, csc_indices, perm = self._weight_indices()
        csc_data = self.data if self.data.size == 1 else self.data[perm]
        return CSC._from_parts(csc_data, csc_indices, csc_indptr, shape=self.shape, backend=self.backend)

    def tocoo(self) -> u.sparse.COO:
        """Convert to coordinate (COO) format.

        Returns
        -------
        brainunit.sparse.COO
            The same logical matrix in COO format, ``shape`` unchanged. A
            homogeneous (size-1) value is broadcast to one entry per stored
            element.

        See Also
        --------
        tocsr : Identity conversion.
        tocsc : Re-encode the same logical matrix column-major.
        """
        rows, cols = _csr_to_coo(self.indices, self.indptr)
        data = self.data if self.data.size == rows.size else u.math.broadcast_to(self.data, rows.shape)
        return u.sparse.COO((data, rows, cols), shape=self.shape, rows_sorted=True)

    def transpose(self, axes=None) -> 'CSC':
        """
        Transpose the CSR matrix.

        This method returns the transpose of the CSR matrix as a CSC matrix.
        Because the transpose of a CSR matrix is a CSC matrix with the same
        underlying arrays, this operation is essentially free (no data is
        copied or rearranged).

        Parameters
        ----------
        axes : None
            This parameter is not used and must be None. Included for compatibility
            with numpy's transpose function signature.

        Returns
        -------
        CSC
            The transpose of the CSR matrix as a CSC (Compressed Sparse Column) matrix.

        Raises
        ------
        AssertionError
            If axes is not None, as this implementation doesn't support custom axis ordering.

        Examples
        --------
        .. code-block:: python

            csc = csr.transpose()
            # or equivalently:
            csc = csr.T
        """
        assert axes is None, "transpose does not support axes argument."
        # ``W.T`` as CSC stores the very same arrays, so it shares the
        # structure cache: the mirror of one is the mirror of the other.
        return CSC._from_parts(
            self.data, self.indices, self.indptr,
            shape=self.shape[::-1], backend=self.backend, cache=self._cache,
        )

    def apply(self, fn) -> 'CSR':
        """
        Apply a unary function to the stored data values.

        Creates a new :class:`CSR` matrix with ``fn(self.data)`` while
        preserving the sparsity structure (indices, indptr, shape, and cached
        diagonal positions).

        Parameters
        ----------
        fn : callable
            A function that accepts a single array argument and returns an
            array of the same shape.  The dtype and unit may differ from the
            input.

        Returns
        -------
        CSR
            A new CSR matrix with transformed data.

        Raises
        ------
        ValueError
            If the transformed value buffer does not have the same shape as
            ``self.data``.

        Examples
        --------
        .. code-block:: python

            squared = csr.apply(lambda x: x ** 2)
        """
        return self._same_structure(self._apply_data(fn))

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None) -> 'CSR':
        """Apply a presynaptic-spike-triggered STDP update, returning a new CSR.

        Convenience wrapper around :func:`brainevent.update_csr_on_binary_pre`
        that keeps the sparsity structure (and therefore the cached weight
        indices) intact.  For each firing presynaptic neuron ``i`` every stored
        synapse is updated ``W[i, j] <- clip(W[i, j] + post_trace[j], w_min, w_max)``.

        Parameters
        ----------
        pre_spike : jax.Array
            Binary/boolean presynaptic spikes, shape ``(shape[0],)``.
        post_trace : jax.Array or Quantity
            Postsynaptic eligibility trace, shape ``(shape[1],)``.
        w_min, w_max : optional
            Clipping bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        CSR
            A new CSR matrix with updated data and identical structure.

        See Also
        --------
        update_on_post : Postsynaptic-spike-triggered counterpart.
        brainevent.update_csr_on_binary_pre : Underlying module function.
        """
        new_w = update_csr_on_binary_pre(
            self.data, self.indices, self.indptr, pre_spike, post_trace,
            w_min, w_max, shape=self.shape, backend=self.backend,
        )
        return self.with_data(new_w)

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None) -> 'CSR':
        """Apply a postsynaptic-spike-triggered STDP update, returning a new CSR.

        Convenience wrapper around :func:`brainevent.update_csr_on_binary_post`.
        Iterating by postsynaptic spike is the *unfavorable* direction for CSR,
        so this reuses the cached column-major weight indices
        (:meth:`_weight_indices`) to scatter the updates back into canonical
        order.  For each firing postsynaptic neuron ``j`` every stored synapse is
        updated ``W[i, j] <- clip(W[i, j] + pre_trace[i], w_min, w_max)``.

        Parameters
        ----------
        pre_trace : jax.Array or Quantity
            Presynaptic eligibility trace, shape ``(shape[0],)``.
        post_spike : jax.Array
            Binary/boolean postsynaptic spikes, shape ``(shape[1],)``.
        w_min, w_max : optional
            Clipping bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        CSR
            A new CSR matrix with updated data and identical structure.

        See Also
        --------
        update_on_pre : Presynaptic-spike-triggered counterpart.
        brainevent.update_csr_on_binary_post : Underlying module function.
        """
        mirror = self._mirror()
        if mirror is None:
            warn_mirror_fallback('CSR', 'update_on_post')
            new_w = _update_csr_on_binary_post_masked(
                self.data, self.indices, self.indptr, pre_trace, post_spike,
                w_min, w_max, shape=self.shape,
            )
        else:
            csc_indptr, csc_indices, perm = mirror
            new_w = update_csr_on_binary_post(
                self.data, csc_indices, csc_indptr, perm, pre_trace, post_spike,
                w_min, w_max, shape=self.shape, backend=self.backend,
            )
        return self.with_data(new_w)

    def _edge_state_or_data(self, edge_state):
        if edge_state is not None:
            return edge_state
        if u.math.size(self.data) == 1 and self.nse > 1:
            raise ValueError(
                "Plasticity rules require per-synapse (heterogeneous) weights, but this matrix "
                "stores a homogeneous (size-1) weight. Materialize per-synapse weights first "
                "or pass an explicit edge_state."
            )
        return self.data

    def apply_on_pre(self, rule, pre_spike, *, edge_state=None, pre_vals=None, post_vals=None, params=None):
        """Apply a per-edge plasticity rule to the synapses of firing presynaptic neurons.

        Method form of :func:`brainevent.csr_apply_on_pre` using this matrix's
        structure: for every stored synapse ``(i, j)`` with ``pre_spike[i]``
        active, ``edge_state[e] <- rule(edge_state[e], pre_vals[i], post_vals[j], params)``.

        Parameters
        ----------
        rule : callable
            ``rule(edge_e, pre_e, post_e, params) -> new_edge_e``, applied per
            edge with :func:`jax.vmap`.
        pre_spike : jax.Array
            Presynaptic spikes (bool or numeric), shape ``(shape[0],)``.
        edge_state : pytree or None, optional
            Per-edge state in CSR data order (leaves of leading dimension
            ``nse``).  ``None`` uses ``self.data``.
        pre_vals, post_vals : pytree or None, optional
            Per-neuron values of leading dimension ``shape[0]`` / ``shape[1]``.
        params : pytree or None, optional
            Rule parameters, broadcast to every edge.

        Returns
        -------
        pytree
            The new edge state (the new data array when ``edge_state`` is
            ``None``; rebuild a matrix with :meth:`with_data`).

        Raises
        ------
        ValueError
            If ``edge_state`` is ``None`` and the matrix stores a homogeneous
            weight, or if shapes do not match (see
            :func:`brainevent.csr_apply_on_pre`).

        See Also
        --------
        apply_on_post : Postsynaptic-spike-triggered counterpart.
        update_on_pre : Fast event-driven additive update.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent
            >>> csr = brainevent.CSR.fromdense(jnp.array([[1., 0.5], [0., 2.]]))
            >>> new_w = csr.apply_on_pre(
            ...     lambda w, pre, post, lr: w + lr * post, jnp.array([True, False]),
            ...     post_vals=jnp.array([1., 2.]), params=0.1,
            ... )
            >>> new_w
            Array([1.1, 0.7, 2. ], dtype=float32)
        """
        return csr_apply_on_pre(
            rule, self._edge_state_or_data(edge_state), self.indices, self.indptr, pre_spike,
            pre_vals=pre_vals, post_vals=post_vals, params=params, shape=self.shape,
        )

    def apply_on_post(self, rule, post_spike, *, edge_state=None, pre_vals=None, post_vals=None, params=None):
        """Apply a per-edge plasticity rule to the synapses of firing postsynaptic neurons.

        Method form of :func:`brainevent.csr_apply_on_post` using this matrix's
        structure: for every stored synapse ``(i, j)`` with ``post_spike[j]``
        active, ``edge_state[e] <- rule(edge_state[e], pre_vals[i], post_vals[j], params)``.

        Parameters
        ----------
        rule : callable
            ``rule(edge_e, pre_e, post_e, params) -> new_edge_e``, applied per
            edge with :func:`jax.vmap`.
        post_spike : jax.Array
            Postsynaptic spikes (bool or numeric), shape ``(shape[1],)``.
        edge_state : pytree or None, optional
            Per-edge state in CSR data order (leaves of leading dimension
            ``nse``).  ``None`` uses ``self.data``.
        pre_vals, post_vals : pytree or None, optional
            Per-neuron values of leading dimension ``shape[0]`` / ``shape[1]``.
        params : pytree or None, optional
            Rule parameters, broadcast to every edge.

        Returns
        -------
        pytree
            The new edge state (the new data array when ``edge_state`` is
            ``None``).

        Raises
        ------
        ValueError
            If ``edge_state`` is ``None`` and the matrix stores a homogeneous
            weight, or if shapes do not match.

        See Also
        --------
        apply_on_pre : Presynaptic-spike-triggered counterpart.
        update_on_post : Fast event-driven additive update.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent
            >>> csr = brainevent.CSR.fromdense(jnp.array([[1., 0.5], [0., 2.]]))
            >>> csr.apply_on_post(
            ...     lambda w, pre, post, lr: w - lr * pre, jnp.array([False, True]),
            ...     pre_vals=jnp.array([1., 2.]), params=0.1,
            ... )
            Array([1. , 0.4, 1.8], dtype=float32)
        """
        return csr_apply_on_post(
            rule, self._edge_state_or_data(edge_state), self.indices, self.indptr, post_spike,
            pre_vals=pre_vals, post_vals=post_vals, params=params, shape=self.shape,
        )

    def delay_deliver(self, ring, events, delays, t, *, weighted: Optional[bool] = None, backend: Optional[str] = None):
        """Deliver presynaptic events into an arrival ring with per-edge delays.

        Convenience wrapper around :func:`brainevent.csr_delay_deliver` using
        this matrix as the ``(n_pre, n_post)`` connectivity::

            ring_new[(t + delays[e]) % L, indices[e]] += data[e] * f(events)[row(e)]

        Parameters
        ----------
        ring : jax.Array or Quantity
            Arrival buffer of shape ``(L, shape[1])``.
        events : jax.Array, BinaryArray or WeightedArray
            Presynaptic events of shape ``(shape[0],)``. A
            :class:`WeightedArray` selects weighted delivery.
        delays : jax.Array or int
            Integer per-edge delays in steps, shape ``(nse,)`` aligned with
            ``self.indices``, or a scalar.
        t : int or jax.Array
            Current integer time step (may be traced).
        weighted : bool or None, optional
            Binary (``False``) or event-value-weighted (``True``) delivery.
            ``None`` (default) infers it from the event type.
        backend : str or None, optional
            Backend override; defaults to ``self.backend``.

        Returns
        -------
        jax.Array or Quantity
            The updated ring.

        See Also
        --------
        brainevent.csr_delay_deliver : Underlying function.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent
            >>> csr = brainevent.CSR.fromdense(jnp.array([[1., 0.], [0., 2.]]))
            >>> ring = jnp.zeros((3, 2))
            >>> csr.delay_deliver(ring, jnp.array([True, True]), jnp.array([1, 2]), 0,
            ...                   backend='jax_raw')
            Array([[0., 0.],
                   [1., 0.],
                   [0., 2.]], dtype=float32)
        """
        from .delay.delay import csr_delay_deliver  # pylint: disable=import-outside-toplevel

        return csr_delay_deliver(
            ring, events, self.data, self.indices, self.indptr, delays, t,
            shape=self.shape, weighted=weighted,
            backend=self.backend if backend is None else backend,
        )

    def delay_layout(self, delays, n_slots: int):
        """Build a :class:`brainevent.CSRDelayLayout` of this matrix.

        The layout reorders the edges of every row by delay so that
        :func:`brainevent.csr_delay_deliver_layout` can deliver spikes delay
        by delay, which is much faster on the GPU than :meth:`delay_deliver`
        when the ring does not fit in the L2 cache.

        Parameters
        ----------
        delays : array_like or int
            Concrete integer delays in steps, shape ``(nse,)`` aligned with
            ``self.indices``, or a scalar; each in ``[1, n_slots)``.
        n_slots : int
            Ring length ``L``.

        Returns
        -------
        brainevent.CSRDelayLayout
            The reordered connectivity, holding ``self.data`` in layout order.

        See Also
        --------
        brainevent.CSRDelayLayout.build : Underlying constructor.
        delay_deliver : Delivery without precomputation.

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> import brainevent
            >>> csr = brainevent.CSR.fromdense(jnp.array([[1., 0.], [0., 2.]]))
            >>> layout = csr.delay_layout(jnp.array([1, 2]), n_slots=3)
            >>> layout.deliver(jnp.zeros((3, 2)), jnp.array([True, True]), 0, backend='jax_raw')
            Array([[0., 0.],
                   [1., 0.],
                   [0., 2.]], dtype=float32)
        """
        from .delay.delay_layout import CSRDelayLayout  # pylint: disable=import-outside-toplevel

        return CSRDelayLayout.build(
            self.data, self.indices, self.indptr, delays, n_slots=n_slots, shape=self.shape,
        )

    def __getitem__(self, index):
        """Extract rows of the matrix ``W`` as a dense array.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0. Negative indices wrap; Python slices are
            supported. Concrete out-of-bounds indices raise ``IndexError``.
            A tuple is a list of rows: ``M[1, 2]`` returns rows 1 and 2, not the
            NumPy element ``(1, 2)``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            ``(n_cols,)`` for a single int, otherwise ``(len(rows), n_cols)``.
        """
        rows = normalize_row_index(index, self.shape[0])
        return csr_slice_rows(
            self.data, self.indices, self.indptr, rows,
            shape=self.shape, backend=self.backend,
        )

    def slice_rows(self, index) -> 'CSR':
        """Return ``W[rows, :]`` as a new :class:`CSR` (outside ``jax.jit``).

        The output non-zero count is data-dependent, so ``index`` must be
        concrete. Accepts the same selectors as :meth:`__getitem__`; a single
        int yields a ``1 x n_cols`` matrix.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0.

        Returns
        -------
        CSR
            Sparse sub-matrix of shape ``(len(rows), n_cols)``.
        """
        rows = jnp.atleast_1d(normalize_row_index(index, self.shape[0]))
        new_data, new_indices, new_indptr, shape = build_sub_csr(
            self.data, self.indices, self.indptr, rows, self.shape[1],
        )
        return CSR._from_parts(new_data, new_indices, new_indptr, shape=shape, backend=self.backend)

    def _binary_op(self, other, op) -> 'CSR':
        if op in [operator.add, operator.sub]:
            other = u.math.asarray(other)
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            return op(dense, other)

        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return self._same_structure(op(self.data, other.data))
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            other = other.reshape(())
            return self._same_structure(op(self.data, other))

        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return self._same_structure(op(self.data, other))

        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'CSR':
        if op in [operator.add, operator.sub]:
            other = u.math.asarray(other)
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            return op(other, dense)

        if isinstance(other, CSR):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return self._same_structure(op(other.data, self.data))
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            other = other.reshape(())
            return self._same_structure(op(other, self.data))
        elif other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return self._same_structure(op(other, self.data))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        """
        Sparse-dense matrix multiplication: ``self @ other``.

        Dispatches to an optimised kernel based on the type and
        dimensionality of ``other``:

        * 1-D array -- sparse matrix-vector product (MV).
        * 2-D array -- sparse matrix-matrix product (MM).
        * :class:`~brainevent.BinaryArray` / :class:`~brainevent.WeightedArray`
          -- event-driven binary / value-weighted kernel.
        * Dense ``jax.Array`` / ``brainunit.Quantity`` -- standard float
          kernel with automatic dtype promotion.

        Parameters
        ----------
        other : jax.Array, brainunit.Quantity, BinaryArray, or WeightedArray
            The right-hand operand.  Must be 1-D or 2-D.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The result of the matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix (sparse-sparse matmul is not
            supported) or if ``other`` has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            x = jnp.ones(n)
            y = csr @ x          # matrix-vector

            X = jnp.ones((n, k))
            Y = csr @ X          # matrix-matrix
        """
        # csr @ other
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        event = as_event_operand(other)
        if event is not None:
            other, weighted = event
            if other.ndim in (1, 2):
                return self._stored_event_product(other, weighted, transpose=False)
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=False,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                return csrmm(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=False,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        """
        Reflected sparse-dense matrix multiplication: ``other @ self``.

        Computes the product with ``self`` on the right by using the
        transposed CSR kernel.  Dispatch logic mirrors :meth:`__matmul__`.

        Parameters
        ----------
        other : jax.Array, brainunit.Quantity, BinaryArray, or WeightedArray
            The left-hand operand.  Must be 1-D or 2-D.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The result of the matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            x = jnp.ones(m)
            y = x @ csr          # vector-matrix

            X = jnp.ones((k, m))
            Y = X @ csr          # matrix-matrix
        """
        # other @ csr
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")

        event = as_event_operand(other)
        if event is not None:
            other, weighted = event
            if other.ndim == 1:
                return self._stored_event_product(other, weighted, transpose=True)
            if other.ndim == 2:
                return self._stored_event_product(other.T, weighted, transpose=True).T
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(self.data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=True,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                other = other.T
                r = csrmm(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape,
                    transpose=True,
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def solve(self, b: Union[jax.Array, u.Quantity], tol=1e-6, reorder=1) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system ``A x = b`` where ``A`` is this CSR matrix.

        Uses a sparse direct solver via the underlying ``csr_solve`` routine.

        Parameters
        ----------
        b : jax.Array or brainunit.Quantity
            Right-hand side vector.  Its first dimension must equal
            ``self.shape[0]``.
        tol : float, optional
            Tolerance for singularity detection.  Defaults to ``1e-6``.
        reorder : int, optional
            Fill-reducing reordering scheme: ``0`` for no reordering,
            ``1`` for symrcm, ``2`` for symamd, ``3`` for csrmetisnd.
            Defaults to ``1``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Solution vector *x* satisfying ``A x = b``.

        Raises
        ------
        AssertionError
            If ``b.shape[0] != self.shape[0]``.

        Examples
        --------
        .. code-block:: python

            x = csr.solve(b)
        """
        assert self.shape[0] == b.shape[0], ("The number of rows in the matrix must match "
                                             "the size of the right-hand side vector b.")
        return csr_solve(self.data, self.indices, self.indptr, b, tol=tol, reorder=reorder)

    def dt2t(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute a sparse transformation from y-w space to w space.

        Performs a specialised sparse matrix-vector product optimised for
        event-driven neural simulations, accumulating contributions from the
        target (post-synaptic) dimension ``y_dim_arr`` weighted by the
        per-synapse values ``w_dim_arr`` according to the connectivity
        defined by this CSR matrix.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Accumulated result, preserving physical units when present.

        See Also
        --------
        dt2t_transposed : The transposed (adjoint) variant.

        Notes
        -----
        Internally calls ``csrmv_dt2t`` with ``transpose=False``.
        """
        return csrmv_dt2t(y_dim_arr, w_dim_arr, self.indices, self.indptr,
                          shape=self.shape, transpose=False, backend=self.backend)

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute the transposed sparse transformation from y-w space to w space.

        This is the adjoint of :meth:`dt2t`, useful for back-propagation
        or adjoint computations in event-driven neural simulations.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Accumulated result of the transposed operation, preserving
            physical units when present.

        See Also
        --------
        dt2t : The forward (non-transposed) variant.

        Notes
        -----
        Internally calls ``csrmv_dt2t`` with ``transpose=True``.
        """
        return csrmv_dt2t(y_dim_arr, w_dim_arr, self.indices, self.indptr,
                          shape=self.shape, transpose=True, backend=self.backend)


@jax.tree_util.register_pytree_node_class
class CSC(CompressedSparseData):
    """
    Event-driven and Unit-aware Compressed Sparse Column (CSC) matrix.

    This class represents a sparse matrix in CSC format, which is efficient for
    column-wise operations. It is compatible with JAX's tree utilities and
    supports unit-aware computations.

    The class supports arithmetic with scalars and dense arrays, plus sparse-dense
    matrix multiplication via ``@``. Sparse-sparse operations are limited.

    Attributes
    ----------
    data : Data
        Array of the non-zero values in the matrix.
    indices : jax.Array
        Array of row indices for the non-zero values.
    indptr : jax.Array
        Array of column pointers indicating where each column starts in the data and indices arrays.
    shape : tuple[int, int]
        The shape of the matrix as (rows, columns).
    nse : int
        Number of stored elements (non-zero entries).
    dtype : dtype
        Data type of the matrix values.

    Notes
    -----
    In CSC format a matrix of shape ``(m, n)`` is stored as three arrays:

    * ``indptr`` of length ``n + 1`` -- the *j*-th column occupies entries
      ``indptr[j]`` to ``indptr[j+1]`` in the ``data`` and ``indices``
      arrays.
    * ``indices`` -- row indices of the stored elements.
    * ``data`` -- the corresponding non-zero values.

    Internally, CSC operations are implemented by treating the underlying
    arrays as a CSR matrix with transposed shape and applying the appropriate
    transpose flags to the CSR kernels.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import brainevent

        data    = jnp.array([1.0, 2.0, 3.0])
        indices = jnp.array([0, 2, 1])
        indptr  = jnp.array([0, 1, 2, 3])
        csc     = brainevent.CSC((data, indices, indptr), shape=(3, 3))

        # Sparse-dense matrix-vector product
        x = jnp.ones(3)
        y = csc @ x

    See Also
    --------
    CSR : Compressed Sparse Row format.
    """
    __module__ = 'brainevent'
    _compressed_format = 'csc'
    _mirror_key = 'csr'

    @classmethod
    def fromdense(
        cls,
        mat,
        *,
        nse: Optional[int] = None,
        index_dtype=jnp.int32,
        indptr_dtype="auto",
        backend: Optional[str] = None,
        precompute_weight_indices: bool = False,
    ) -> 'CSC':
        """
        Create a CSC (Compressed Sparse Column) matrix from a dense matrix.

        This method converts a dense matrix to CSC format, which is an efficient
        storage format for sparse matrices.

        Parameters
        ----------
        mat : array_like
            The dense matrix to be converted to CSC format.
        nse : int, optional
            The number of non-zero elements in the matrix. If None, it will be
            calculated from the input matrix.
        index_dtype : dtype, optional
            The data type for the ``indices`` (row) array. Must be ``int32``;
            ``indices`` are always int32. To control the ``indptr`` precision use
            ``indptr_dtype`` instead.
        indptr_dtype : {"auto", int32, int64}, optional
            The precision policy for the ``indptr`` offsets. ``"auto"`` (default)
            promotes to int64 only when the nnz exceeds the int32 range (and
            requires ``jax_enable_x64``).
        backend : str or None, optional
            Compute backend to attach to the matrix. Default ``None``.
        precompute_weight_indices : bool, optional
            If ``True``, eagerly build and cache the row-major (CSR-like) weight
            indices used by the *unfavorable* ``event @ CSC`` direction (see
            :meth:`build_weight_indices`). If ``False`` (default), the indices
            are built lazily on first use. Default ``False``.

        Returns
        -------
        CSC
            A new CSC matrix instance created from the input dense matrix.

        See Also
        --------
        build_weight_indices : Eagerly build the cached weight indices.
        CSC._weight_indices : Lazily build/return the cached weight indices.

        Examples
        --------
        .. code-block:: python

            import jax.numpy as jnp
            import brainevent

            dense = jnp.array([[1.0, 0.0], [0.0, 2.0]])
            csc = brainevent.CSC.fromdense(dense)
        """
        if jnp.dtype(index_dtype) != jnp.dtype(jnp.int32):
            raise ValueError(
                "CSC.fromdense: `indices` are always int32; got index_dtype="
                f"{jnp.dtype(index_dtype)}. Use `indptr_dtype` to control the "
                "indptr precision instead."
            )
        if nse is None:
            nse = (u.get_mantissa(mat) != 0).sum()
        csc = u.sparse.csr_fromdense(mat.T, nse=nse, index_dtype=jnp.int32).T
        out = CSC(
            (csc.data, csc.indices, csc.indptr),
            shape=csc.shape, backend=backend, indptr_dtype=indptr_dtype,
        )
        if precompute_weight_indices:
            out = out.build_weight_indices()
        return out

    def with_data(self, data: ArrayData) -> 'CSC':
        """
        Create a new CSC matrix with updated data while keeping the same structure.

        This method creates a new CSC matrix instance with the provided data,
        maintaining the original indices, indptr, and shape.

        Parameters
        ----------
        data : Data
            The new data array to replace the existing data in the CSC matrix.
            It must have the same shape, dtype, and unit as the original data.

        Returns
        -------
        CSC
            A new CSC matrix instance with updated data and the same structure as the original.

        Raises
        ------
        AssertionError
            If the shape, dtype, or unit of the new data doesn't match the original data.

        Examples
        --------
        .. code-block:: python

            new_data = jnp.array([10.0, 20.0, 30.0])
            new_csc = csc.with_data(new_data)
        """
        assert data.shape == self.data.shape
        assert data.dtype == self.data.dtype
        assert u.get_unit(data) == u.get_unit(self.data)
        return self._same_structure(data)

    def todense(self) -> Union[jax.Array, u.Quantity]:
        """
        Convert the CSC matrix to a dense matrix.

        Transposes the underlying CSR-style storage, converts to dense, and
        transposes back.

        Returns
        -------
        jax.Array or brainunit.Quantity
            A dense matrix of shape ``self.shape`` containing all the values
            (including zeros) of the sparse matrix.

        Examples
        --------
        .. code-block:: python

            dense = csc.todense()
        """
        return self.T.todense().T

    def tocsc(self) -> 'CSC':
        """Return this matrix in CSC format (a no-op that returns ``self``).

        Provided for a uniform conversion interface across data
        representations; ``CSC`` is already column-compressed.

        Returns
        -------
        CSC
            ``self``, unchanged.

        See Also
        --------
        tocsr : Re-encode the same logical matrix row-major.
        tocoo : Convert to coordinate format.
        """
        return self

    def tocsr(self) -> 'CSR':
        """Re-encode the same logical matrix in CSR format.

        Unlike :meth:`transpose` (which reinterprets the arrays as ``W.T`` with
        swapped shape), ``tocsr`` returns a :class:`CSR` describing the *same*
        matrix ``W`` with the *same* ``shape`` -- the entries are resorted into
        row-major order.

        Returns
        -------
        CSR
            The same logical matrix in CSR format, ``shape`` unchanged.

        See Also
        --------
        tocsc : Identity conversion.
        transpose : Logical transpose (swaps ``shape``).
        """
        csr_indptr, csr_indices, perm = self._weight_indices()
        csr_data = self.data if self.data.size == 1 else self.data[perm]
        return CSR._from_parts(csr_data, csr_indices, csr_indptr, shape=self.shape, backend=self.backend)

    def tocoo(self) -> u.sparse.COO:
        """Convert to coordinate (COO) format.

        Returns
        -------
        brainunit.sparse.COO
            The same logical matrix in COO format, ``shape`` unchanged. A
            homogeneous (size-1) value is broadcast to one entry per stored
            element.

        See Also
        --------
        tocsr : Re-encode the same logical matrix row-major.
        tocsc : Identity conversion.
        """
        cols, rows = _csr_to_coo(self.indices, self.indptr)
        data = self.data if self.data.size == rows.size else u.math.broadcast_to(self.data, rows.shape)
        return u.sparse.COO((data, rows, cols), shape=self.shape, cols_sorted=True)

    def transpose(self, axes=None) -> 'CSR':
        """
        Transpose the CSC matrix.

        Returns the transpose as a :class:`CSR` matrix.  Because the
        transpose of a CSC matrix is a CSR matrix with the same underlying
        arrays, this operation is essentially free.

        Parameters
        ----------
        axes : None
            Must be ``None``.  Included for API compatibility with NumPy.

        Returns
        -------
        CSR
            The transpose of the CSC matrix as a CSR (Compressed Sparse Row) matrix.

        Raises
        ------
        AssertionError
            If ``axes`` is not ``None``.

        Examples
        --------
        .. code-block:: python

            csr = csc.transpose()
            # or equivalently:
            csr = csc.T
        """
        assert axes is None
        # ``W.T`` as CSR stores the very same arrays, so it shares the
        # structure cache: the mirror of one is the mirror of the other.
        return CSR._from_parts(
            self.data, self.indices, self.indptr,
            shape=self.shape[::-1], backend=self.backend, cache=self._cache,
        )

    def apply(self, fn) -> 'CSC':
        """
        Apply a unary function to the stored data values.

        Creates a new :class:`CSC` matrix with ``fn(self.data)`` while
        preserving the sparsity structure (indices, indptr, shape, and cached
        diagonal positions).

        Parameters
        ----------
        fn : callable
            A function that accepts a single array argument and returns an
            array of the same shape.  The dtype and unit may differ from the
            input.

        Returns
        -------
        CSC
            A new CSC matrix with transformed data.

        Raises
        ------
        ValueError
            If the transformed value buffer does not have the same shape as
            ``self.data``.

        Examples
        --------
        .. code-block:: python

            squared = csc.apply(lambda x: x ** 2)
        """
        return self._same_structure(self._apply_data(fn))

    def update_on_pre(self, pre_spike, post_trace, w_min=None, w_max=None) -> 'CSC':
        """Apply a presynaptic-spike-triggered STDP update, returning a new CSC.

        Iterating by presynaptic spike is the *unfavorable* direction for CSC,
        so this reuses the cached row-major weight indices
        (:meth:`_weight_indices`) and routes through
        :func:`brainevent.update_csr_on_binary_post`, scattering updates back
        into canonical CSC order.  For each firing presynaptic neuron ``i`` every
        stored synapse is updated
        ``W[i, j] <- clip(W[i, j] + post_trace[j], w_min, w_max)``.

        Parameters
        ----------
        pre_spike : jax.Array
            Binary/boolean presynaptic spikes, shape ``(shape[0],)``.
        post_trace : jax.Array or Quantity
            Postsynaptic eligibility trace, shape ``(shape[1],)``.
        w_min, w_max : optional
            Clipping bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        CSC
            A new CSC matrix with updated data and identical structure.

        See Also
        --------
        update_on_post : Postsynaptic-spike-triggered counterpart.
        brainevent.update_csc_on_binary_pre : Equivalent module function.
        """
        mirror = self._mirror()
        if mirror is None:
            warn_mirror_fallback('CSC', 'update_on_pre')
            new_w = _update_csr_on_binary_post_masked(
                self.data, self.indices, self.indptr, post_trace, pre_spike,
                w_min, w_max, shape=self.shape[::-1],
            )
        else:
            csr_indptr, csr_indices, perm = mirror
            new_w = update_csr_on_binary_post(
                self.data, csr_indices, csr_indptr, perm, post_trace, pre_spike,
                w_min, w_max, shape=self.shape[::-1], backend=self.backend,
            )
        return self.with_data(new_w)

    def update_on_post(self, pre_trace, post_spike, w_min=None, w_max=None) -> 'CSC':
        """Apply a postsynaptic-spike-triggered STDP update, returning a new CSC.

        Iterating by postsynaptic spike is the *favorable* direction for CSC, so
        this streams directly over the stored arrays (no permutation) via
        :func:`brainevent.update_csr_on_binary_pre` on the transposed shape.  For
        each firing postsynaptic neuron ``j`` every stored synapse is updated
        ``W[i, j] <- clip(W[i, j] + pre_trace[i], w_min, w_max)``.

        Parameters
        ----------
        pre_trace : jax.Array or Quantity
            Presynaptic eligibility trace, shape ``(shape[0],)``.
        post_spike : jax.Array
            Binary/boolean postsynaptic spikes, shape ``(shape[1],)``.
        w_min, w_max : optional
            Clipping bounds; ``None`` disables the corresponding bound.

        Returns
        -------
        CSC
            A new CSC matrix with updated data and identical structure.

        See Also
        --------
        update_on_pre : Presynaptic-spike-triggered counterpart.
        brainevent.update_csc_on_binary_post : Equivalent module function.
        """
        new_w = update_csr_on_binary_pre(
            self.data, self.indices, self.indptr, post_spike, pre_trace,
            w_min, w_max, shape=self.shape[::-1], backend=self.backend,
        )
        return self.with_data(new_w)

    def __getitem__(self, index):
        """Extract rows of the matrix ``W`` as a dense array.

        Row slicing is the *unfavorable* direction for CSC, so it reuses the
        cached row-major (CSR-of-``W``) view from :meth:`_weight_indices` and the
        shared ``csr_slice_rows`` kernel.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0. Negative indices wrap; Python slices are
            supported. Concrete out-of-bounds indices raise ``IndexError``.
            A tuple is a list of rows: ``M[1, 2]`` returns rows 1 and 2, not the
            NumPy element ``(1, 2)``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            ``(n_cols,)`` for a single int, otherwise ``(len(rows), n_cols)``.
        """
        rows = normalize_row_index(index, self.shape[0])
        csr_indptr, csr_indices, perm = self._weight_indices()
        weights = self.data if self.data.size == 1 else self.data.reshape(-1)[perm]
        return csr_slice_rows(
            weights, csr_indices, csr_indptr, rows,
            shape=self.shape, backend=self.backend,
        )
    def slice_rows(self, index) -> 'CSC':
        """Return ``W[rows, :]`` as a new :class:`CSC` (outside ``jax.jit``).

        Builds the CSR arrays of ``W[rows, :]`` through the cached CSR-of-``W``
        view, then converts to CSC. The output non-zero count is data-dependent,
        so ``index`` must be concrete.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0.

        Returns
        -------
        CSC
            Sparse sub-matrix of shape ``(len(rows), n_cols)``.
        """
        rows = jnp.atleast_1d(normalize_row_index(index, self.shape[0]))
        csr_indptr, csr_indices, perm = self._weight_indices()
        data = cast(jax.Array, self.data)
        weights = data if data.size == 1 else data.reshape(-1)[perm]
        sub_data, sub_indices, sub_indptr, shape = build_sub_csr(
            weights, csr_indices, csr_indptr, rows, self.shape[1],
        )
        with jax.ensure_compile_time_eval():
            csc_indptr, csc_indices, cperm = csr_to_csc_index(
                sub_indptr, sub_indices, shape=shape,
            )
        csc_data = sub_data if sub_data.size == 1 else sub_data[cperm]
        return CSC._from_parts(csc_data, csc_indices, csc_indptr, shape=shape, backend=self.backend)

    def _binary_op(self, other, op) -> 'CSC':
        if op in [operator.add, operator.sub]:
            other = u.math.asarray(other)
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            return op(dense, other)
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return self._same_structure(op(self.data, other.data))
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            other = other.reshape(())
            return self._same_structure(op(self.data, other))
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return self._same_structure(op(self.data, other))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def _binary_rop(self, other, op) -> 'CSC':
        if op in [operator.add, operator.sub]:
            other = u.math.asarray(other)
            jnp.broadcast_shapes(self.shape, other.shape)
            dense = self.todense()
            return op(other, dense)
        if isinstance(other, CSC):
            if id(other.indices) == id(self.indices) and id(other.indptr) == id(self.indptr):
                return self._same_structure(op(other.data, self.data))
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(f"binary operation {op} between two sparse objects.")

        other = u.math.asarray(other)
        if other.size == 1:
            other = other.reshape(())
            return self._same_structure(op(other, self.data))
        elif other.ndim == 2 and other.shape == self.shape:
            cols, rows = _csr_to_coo(self.indices, self.indptr)
            other = other[rows, cols]
            return self._same_structure(op(other, self.data))
        else:
            raise NotImplementedError(f"mul with object of shape {other.shape}")

    def __matmul__(self, other):
        """
        Sparse-dense matrix multiplication: ``self @ other``.

        Dispatches to an optimised kernel based on the type and
        dimensionality of ``other``.  Internally the CSC storage is treated
        as a transposed CSR matrix, so the CSR kernels are called with
        ``shape=self.shape[::-1]`` and ``transpose=True``.

        Parameters
        ----------
        other : jax.Array, brainunit.Quantity, BinaryArray, or WeightedArray
            The right-hand operand.  Must be 1-D or 2-D.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The result of the matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            x = jnp.ones(n)
            y = csc @ x          # matrix-vector

            X = jnp.ones((n, k))
            Y = csc @ X          # matrix-matrix
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        event = as_event_operand(other)
        if event is not None:
            other, weighted = event
            if other.ndim in (1, 2):
                return self._stored_event_product(other, weighted, transpose=True)
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:

            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=True,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                return csrmm(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=True,
                    backend=self.backend,
                )
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __rmatmul__(self, other):
        """
        Reflected sparse-dense matrix multiplication: ``other @ self``.

        Computes the product with ``self`` on the right.  Internally the CSC
        storage is treated as a transposed CSR matrix with
        ``transpose=False``.

        Parameters
        ----------
        other : jax.Array, brainunit.Quantity, BinaryArray, or WeightedArray
            The left-hand operand.  Must be 1-D or 2-D.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The result of the matrix multiplication.

        Raises
        ------
        NotImplementedError
            If ``other`` is a sparse matrix or has more than 2 dimensions.

        Examples
        --------
        .. code-block:: python

            x = jnp.ones(m)
            y = x @ csc          # vector-matrix

            X = jnp.ones((k, m))
            Y = X @ csc          # matrix-matrix
        """
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects.")
        data = self.data

        event = as_event_operand(other)
        if event is not None:
            other, weighted = event
            if other.ndim == 1:
                return self._stored_event_product(other, weighted, transpose=False)
            if other.ndim == 2:
                return self._stored_event_product(other.T, weighted, transpose=False).T
            raise NotImplementedError(f"matmul with object of shape {other.shape}")

        else:
            other = u.math.asarray(other)
            data, other = u.math.promote_dtypes(data, other)
            if other.ndim == 1:
                return csrmv(
                    data,
                    self.indices,
                    self.indptr,
                    other,
                    shape=self.shape[::-1],
                    transpose=False,
                    backend=self.backend,
                )
            elif other.ndim == 2:
                other = other.T
                r = csrmm(
                    data,
                    self.indices,
                    self.indptr, other,
                    shape=self.shape[::-1],
                    transpose=False,
                    backend=self.backend,
                )
                return r.T
            else:
                raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def solve(self, b: Union[jax.Array, u.Quantity], tol=1e-6, reorder=1) -> Union[jax.Array, u.Quantity]:
        """
        Solve the linear system ``A x = b`` where ``A`` is this CSC matrix.

        Delegates to the CSR solver on :meth:`tocsr`, the same logical matrix.

        Parameters
        ----------
        b : jax.Array or brainunit.Quantity
            Right-hand side vector.  Its first dimension must equal
            ``self.shape[0]``.
        tol : float, optional
            Tolerance for singularity detection.  Defaults to ``1e-6``.
        reorder : int, optional
            Fill-reducing reordering scheme: ``0`` for no reordering,
            ``1`` for symrcm, ``2`` for symamd, ``3`` for csrmetisnd.
            Defaults to ``1``.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Solution vector *x* satisfying ``A x = b``.

        Raises
        ------
        AssertionError
            If ``b.shape[0] != self.shape[0]``.

        Examples
        --------
        .. code-block:: python

            x = csc.solve(b)
        """
        assert self.shape[0] == b.shape[0], ("The number of rows in the matrix must match "
                                             "the size of the right-hand side vector b.")
        return self.tocsr().solve(b, tol=tol, reorder=reorder)

    def dt2t(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute a sparse transformation from y-w space to w space.

        Performs a specialised sparse matrix-vector product optimised for
        event-driven neural simulations.  The CSC storage is treated as a
        transposed CSR matrix, so this method calls the CSR kernel with
        ``shape=self.shape[::-1]`` and ``transpose=True``.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Accumulated result, preserving physical units when present.

        See Also
        --------
        dt2t_transposed : The transposed (adjoint) variant.

        Notes
        -----
        Internally calls ``csrmv_dt2t`` with ``transpose=True`` and reversed
        shape to account for the column-oriented storage format.
        """
        return csrmv_dt2t(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape[::-1], transpose=True,
                          backend=self.backend)

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """
        Compute the transposed sparse transformation from y-w space to w space.

        This is the adjoint of :meth:`dt2t`, useful for back-propagation
        or adjoint computations in event-driven neural simulations.  Uses
        ``transpose=False`` with the reversed shape to compute the
        appropriate transposed operation for CSC storage.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Values in the target (post-synaptic) dimension.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-synapse weight values.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Accumulated result of the transposed operation, preserving
            physical units when present.

        See Also
        --------
        dt2t : The forward (non-transposed) variant.

        Notes
        -----
        Internally calls ``csrmv_dt2t`` with ``transpose=False`` and reversed
        shape.
        """
        return csrmv_dt2t(y_dim_arr, w_dim_arr, self.indices, self.indptr, shape=self.shape[::-1], transpose=False,
                          backend=self.backend)
