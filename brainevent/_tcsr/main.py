# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Provide TCSR storage with flag-based logical transpose views."""

from __future__ import annotations

import operator
from typing import Any, Callable, Optional, Self, Union, cast

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._csr.main import CSC as PlainCSC
from brainevent._csr.main import CSR as PlainCSR
from brainevent._csr.slice import csr_slice_rows
from brainevent._data import DataRepresentation
from brainevent._event import BinaryArray
from brainevent._misc import (
    _csr_to_coo,
    _csr_todense,
    build_sub_csr,
    normalize_row_index,
)
from brainevent._typing import Data, MatrixShape

from .binary import _validate_backward_algorithm
from .preprocess import (
    TCSBuffers,
    TCSCMirror,
    build_hybrid_workspace,
    build_tcsr_structure,
    build_tcsc_mirror,
    sort_csr,
)

__all__ = ["TCSR", "TiledCompressedSparseData"]


@jax.tree_util.register_pytree_node_class
class TiledCompressedSparseData(DataRepresentation):
    """Manage shared tile-compressed storage for TCSR logical views.

    This implementation class owns one canonical row-compressed structure and
    lazily builds a data-free column-compressed mirror. A logical transpose is
    an O(1) view over the same storage; heterogeneous values are exposed in the
    compressed order of the current view, while a homogeneous value remains a
    size-one buffer.

    Parameters
    ----------
    source : brainevent._csr.main.CSR
        Plain CSR whose target indices are already nondecreasing within each
        row. The default constructor preserves the source entry order.
    backend : str, optional
        Backend used by non-binary sparse operations.
    binary_backend : str, optional
        Backend used by binary sparse operations.
    backward_algorithm : {"bptt", "pp_prop"}, optional
        Weight-gradient interpretation for floating event operands. ``bptt``
        treats them as binary activations; ``pp_prop`` preserves Batch1
        eligibility values. Default is ``bptt``.

    Attributes
    ----------
    data : jax.Array or brainunit.Quantity
        Values in the current logical view's compressed order. Its shape is
        ``(nse,)`` for conventional heterogeneous storage; homogeneous storage
        is represented by a scalar or size-one array.
    indices : jax.Array
        Minor-axis coordinates in the current logical view's CSR order.
    indptr : jax.Array
        Int64 row pointers in the current logical view's CSR order.
    shape : tuple[int, int]
        Current logical matrix shape.
    nse : int
        Number of represented sparse entries.
    dtype : numpy.dtype
        Dtype of the value buffer.

    Notes
    -----
    ``source`` must already have nondecreasing column indices within each row.
    Direct construction does not verify this invariant. Use :meth:`fromcsr`
    when source ordering is not guaranteed.

    TCSR requires ``jax_enable_x64=True`` because canonical and mirror row
    pointers are always int64. Enable x64 before constructing a TCSR object.

    The forward TCSR structure owns the canonical value order. The TCSC mirror
    contains structure and a permutation into that value array, but no second
    value array.

    See Also
    --------
    TCSR : Public concrete tiled CSR matrix.
    fromcsr : Validate ordering by stably sorting a plain CSR source.
    """

    _compressed_format = "tcs"

    def __init__(
        self,
        source: PlainCSR,
        *,
        backend: Optional[str] = None,
        binary_backend: Optional[str] = None,
        backward_algorithm: str = "bptt",
    ) -> None:
        """Build shared tile-compressed storage from a sorted plain CSR.

        Parameters
        ----------
        source : brainevent._csr.main.CSR
            Plain CSR whose target indices are already nondecreasing within
            every row.
        backend : str, optional
            Backend used by non-binary sparse operations.
        binary_backend : str, optional
            Backend used by binary sparse operations.
        backward_algorithm : {"bptt", "pp_prop"}, optional
            Weight-gradient interpretation for floating event operands.

        Raises
        ------
        TypeError
            If ``source`` is not a plain CSR matrix.
        ValueError
            If ``backward_algorithm`` is unsupported or a source row exceeds
            the tile metadata limit.

        Notes
        -----
        This trusted constructor does not verify or sort row-local target
        indices. Use :meth:`fromcsr` whenever row-local sortedness is not
        guaranteed. TCSR construction also requires ``jax_enable_x64=True``
        because its row pointers are int64.

        See Also
        --------
        fromcsr : Stably sort an arbitrary plain CSR before construction.
        from_sorted_csr : Named trusted entry point for an already sorted CSR.
        """
        _validate_backward_algorithm(backward_algorithm)
        if not isinstance(source, PlainCSR):
            raise TypeError(
                "TCSR construction requires a sorted plain CSR source"
            )
        components, local_targets, tile_offsets = build_tcsr_structure(source)
        tcs_buffers = TCSBuffers(
            tcsr_local_targets=local_targets,
            tcsr_tile_offsets=tile_offsets,
            tcsr_workspace=build_hybrid_workspace(components.indptr),
        )
        self._canonical_data = components.data
        self._tcsr_indices = components.indices
        self._tcsr_indptr = components.indptr
        self._tcs_buffers = tcs_buffers
        self._base_shape: MatrixShape = (
            int(source.shape[0]),
            int(source.shape[1]),
        )
        self._transpose_state = False
        self.backend = backend
        self.binary_backend = binary_backend
        self.backward_algorithm = backward_algorithm
        super().__init__(
            (self._canonical_data, self._tcsr_indices, self._tcsr_indptr),
            shape=self._base_shape,
            buffers={"_tcs_buffers": tcs_buffers},
        )

    @classmethod
    def _from_tcs_parts(
        cls,
        canonical_data: Data,
        tcsr_indices: jax.Array,
        tcsr_indptr: jax.Array,
        tcs_buffers: TCSBuffers,
        *,
        base_shape: MatrixShape,
        transpose_state: bool,
        backend: Optional[str],
        binary_backend: Optional[str],
        backward_algorithm: str,
    ) -> Self:
        """Create one view over already validated shared TCS leaves."""
        obj = cls.__new__(cls)
        obj._canonical_data = canonical_data
        obj._tcsr_indices = tcsr_indices
        obj._tcsr_indptr = tcsr_indptr
        obj._tcs_buffers = tcs_buffers
        obj._base_shape = (int(base_shape[0]), int(base_shape[1]))
        obj._transpose_state = bool(transpose_state)
        obj.backend = backend
        obj.binary_backend = binary_backend
        obj.backward_algorithm = backward_algorithm
        shape = (
            obj._base_shape[::-1]
            if obj._transpose_state
            else obj._base_shape
        )
        DataRepresentation.__init__(
            obj,
            (obj._canonical_data, obj._tcsr_indices, obj._tcsr_indptr),
            shape=shape,
            buffers={"_tcs_buffers": tcs_buffers},
        )
        return obj

    def _tcs_children(self) -> tuple[Any, ...]:
        """Return dynamic shared-storage components for the PyTree protocol."""
        children = (
            self._canonical_data,
            self._tcsr_indices,
            self._tcsr_indptr,
            self._tcs_buffers.tcsr_local_targets,
            self._tcs_buffers.tcsr_tile_offsets,
            self._tcs_buffers.tcsr_workspace,
        )
        mirror = self._tcs_buffers.tcsc
        return children if mirror is None else (*children, mirror)

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Flatten shared arrays and workspaces for the JAX pytree protocol.

        Returns
        -------
        children : tuple
            Dynamic value, structure, workspace, and optional mirror arrays.
        aux_data : tuple
            Static logical shape, view state, and backend configuration.

        See Also
        --------
        tree_unflatten : Restore a TCSR view from flattened components.
        """
        aux = (
            self._base_shape,
            self._transpose_state,
            self.backend,
            self.binary_backend,
            self.backward_algorithm,
            self.has_tcsc_mirror,
        )
        return self._tcs_children(), aux

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[Any, ...],
        children: tuple[Any, ...],
    ) -> Self:
        """Restore a TCS view from JAX pytree leaves and static metadata.

        Parameters
        ----------
        aux_data : tuple
            Static metadata returned by :meth:`tree_flatten`.
        children : tuple
            Dynamic arrays returned by :meth:`tree_flatten`.

        Returns
        -------
        TiledCompressedSparseData
            Reconstructed view sharing the supplied TCS components.

        Raises
        ------
        ValueError
            If mirror metadata and mirror children disagree.

        See Also
        --------
        tree_flatten : Produce the components accepted by this method.
        """
        (
            base_shape,
            transpose_state,
            backend,
            binary_backend,
            backward_algorithm,
            has_tcsc,
        ) = aux_data
        (
            canonical_data,
            tcsr_indices,
            tcsr_indptr,
            tcsr_local_targets,
            tcsr_tile_offsets,
            tcsr_workspace,
            *optional_mirror,
        ) = children
        if bool(has_tcsc) != bool(optional_mirror):
            raise ValueError("TCSC mirror PyTree state is inconsistent")
        tcs_buffers = TCSBuffers(
            tcsr_local_targets=tcsr_local_targets,
            tcsr_tile_offsets=tcsr_tile_offsets,
            tcsr_workspace=tcsr_workspace,
            tcsc=optional_mirror[0] if optional_mirror else None,
        )
        return cls._from_tcs_parts(
            canonical_data,
            tcsr_indices,
            tcsr_indptr,
            tcs_buffers,
            base_shape=base_shape,
            transpose_state=transpose_state,
            backend=backend,
            binary_backend=binary_backend,
            backward_algorithm=backward_algorithm,
        )

    def _new_view(
        self,
        *,
        transpose_state: bool,
    ) -> Self:
        """Return a same-type view sharing every TCS leaf with this instance."""
        return type(self)._from_tcs_parts(
            self._canonical_data,
            self._tcsr_indices,
            self._tcsr_indptr,
            self._tcs_buffers,
            base_shape=self._base_shape,
            transpose_state=transpose_state,
            backend=self.backend,
            binary_backend=self.binary_backend,
            backward_algorithm=self.backward_algorithm,
        )

    def _uses_mirror_data_order(self) -> bool:
        """Return whether the current view exposes mirror-ordered values."""
        return False

    def _ensure_tcsc_mirror(self) -> TCSCMirror:
        """Build and cache the data-free TCSC mirror when first required."""
        mirror = self._tcs_buffers.tcsc
        if mirror is not None:
            return mirror
        if isinstance(self._tcsr_indices, jax.core.Tracer) or isinstance(
            self._tcsr_indptr, jax.core.Tracer
        ):
            raise RuntimeError(
                "TCSC mirror must be materialized before a mirror-free TCSR "
                "is passed as a dynamic argument to a jitted transpose=False "
                "route"
            )
        with jax.ensure_compile_time_eval():
            mirror = build_tcsc_mirror(
                self._tcsr_indices,
                self._tcsr_indptr,
                shape=self._base_shape,
            )
        self._tcs_buffers.tcsc = mirror
        return mirror

    def materialize_tcsc_mirror(self) -> TiledCompressedSparseData:
        """Materialize the shared TCSC mirror before a dynamic JIT boundary.

        Returns
        -------
        TiledCompressedSparseData
            This matrix or view with its shared TCSC mirror materialized.

        Raises
        ------
        RuntimeError
            If mirror construction is first requested after structural arrays
            have become JAX tracers.

        Notes
        -----
        The operation is idempotent. All views that share this matrix's
        structural buffers observe the same mirror.
        """
        self._ensure_tcsc_mirror()
        return self

    def _from_canonical_data(self, data: Data) -> Self:
        """Return a same-view object with replacement canonical values."""
        return type(self)._from_tcs_parts(
            data,
            self._tcsr_indices,
            self._tcsr_indptr,
            self._tcs_buffers,
            base_shape=self._base_shape,
            transpose_state=self._transpose_state,
            backend=self.backend,
            binary_backend=self.binary_backend,
            backward_algorithm=self.backward_algorithm,
        )

    def _canonicalize_view_data(self, data: Data) -> Data:
        """Convert current-view value order to canonical TCSR value order."""
        if int(getattr(data, "size", 1)) == 1 or not self._uses_mirror_data_order():
            return data
        mirror = self._ensure_tcsc_mirror()
        data_mantissa = u.get_mantissa(data)
        canonical_mantissa = jnp.empty_like(data_mantissa).at[
            mirror.permutation
        ].set(data_mantissa)
        return u.maybe_decimal(canonical_mantissa * u.get_unit(data))

    def _from_view_data(self, data: Data) -> Self:
        """Rebuild from values expressed in the current view's order."""
        return self._from_canonical_data(self._canonicalize_view_data(data))

    def apply(self, fn: Callable[[Any], Any]) -> Self:
        """Apply a function to stored values and preserve the TCSR structure.

        Unlike :meth:`with_data`, this method permits the function to change
        the value dtype or physical unit.

        Parameters
        ----------
        fn : callable
            Function applied to values in the current view's compressed order.

        Returns
        -------
        TiledCompressedSparseData
            Same concrete logical view with transformed values.

        Raises
        ------
        ValueError
            If the transformed value buffer has a different shape from
            ``self.data``.

        Notes
        -----
        The function may change dtype or unit, but it cannot switch between
        homogeneous and heterogeneous storage. It is evaluated exactly once.
        """
        return self._from_view_data(self._apply_data(fn))

    def __abs__(self) -> Self:
        """Return element-wise absolute stored values.

        Returns
        -------
        TiledCompressedSparseData
            Same concrete logical view with absolute values.
        """
        return self.apply(operator.abs)

    def __neg__(self) -> Self:
        """Return element-wise negated stored values.

        Returns
        -------
        TiledCompressedSparseData
            Same concrete logical view with negated values.
        """
        return self.apply(operator.neg)

    def __pos__(self) -> Self:
        """Apply unary positive to every stored value.

        Returns
        -------
        TiledCompressedSparseData
            Same concrete logical view with positive values.
        """
        return self.apply(operator.pos)

    def _shares_logical_structure(self, other: Any) -> bool:
        """Return whether two views can combine their values position-wise."""
        return (
            isinstance(other, TiledCompressedSparseData)
            and self._tcsr_indices is other._tcsr_indices
            and self._tcsr_indptr is other._tcsr_indptr
            and self._transpose_state == other._transpose_state
            and self.shape == other.shape
        )

    def _binary_op(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
    ) -> Any:
        if isinstance(other, TiledCompressedSparseData):
            if self._shares_logical_structure(other):
                return self._from_view_data(op(self.data, other.data))
            raise NotImplementedError(
                f"binary operation {op} requires identical TCSR structure and orientation"
            )
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(
                f"binary operation {op} between two sparse objects"
            )

        other = u.math.asarray(other)
        if op in (operator.add, operator.sub):
            jnp.broadcast_shapes(self.shape, other.shape)
            return op(self.todense(), other)
        if other.size == 1:
            return self._from_view_data(op(self.data, other))
        if other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            values = other[rows, cols]
            return self._from_view_data(op(self.data, values))
        raise NotImplementedError(
            f"binary operation {op} with object of shape {other.shape}"
        )

    def _binary_rop(
        self,
        other: Any,
        op: Callable[[Any, Any], Any],
    ) -> Any:
        if isinstance(other, TiledCompressedSparseData):
            if self._shares_logical_structure(other):
                return self._from_view_data(op(other.data, self.data))
            raise NotImplementedError(
                f"binary operation {op} requires identical TCSR structure and orientation"
            )
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError(
                f"binary operation {op} between two sparse objects"
            )

        other = u.math.asarray(other)
        if op in (operator.add, operator.sub):
            jnp.broadcast_shapes(self.shape, other.shape)
            return op(other, self.todense())
        if other.size == 1:
            return self._from_view_data(op(other, self.data))
        if other.ndim == 2 and other.shape == self.shape:
            rows, cols = _csr_to_coo(self.indices, self.indptr)
            values = other[rows, cols]
            return self._from_view_data(op(values, self.data))
        raise NotImplementedError(
            f"binary operation {op} with object of shape {other.shape}"
        )

    def apply2(
        self,
        other: Any,
        fn: Callable[[Any, Any], Any],
        *,
        reverse: bool = False,
    ) -> Any:
        """Apply a binary function with sparse value semantics.

        Parameters
        ----------
        other : Any
            Operand combined with this logical TCSR view.
        fn : callable
            Binary function to apply.
        reverse : bool, optional
            Compute ``fn(other, self)`` when true. Default is false.

        Returns
        -------
        TiledCompressedSparseData or Data
            Sparse value result or dense addition/subtraction result.

        Notes
        -----
        Unlike unary :meth:`apply`, a position-dependent dense operand may
        legitimately expand homogeneous storage into heterogeneous values.
        """
        if reverse:
            return self._binary_rop(other, fn)
        return self._binary_op(other, fn)

    def __mul__(self, other: Any) -> Any:
        """Multiply stored values element-wise.

        Parameters
        ----------
        other : Any
            Scalar, matching dense matrix, or identical-structure TCSR view.

        Returns
        -------
        TiledCompressedSparseData
            Same logical view containing the multiplied stored values.
        """
        return self.apply2(other, operator.mul)

    def __truediv__(self, other: Any) -> Any:
        """Divide stored values element-wise.

        Parameters
        ----------
        other : Any
            Scalar, matching dense matrix, or identical-structure TCSR view.

        Returns
        -------
        TiledCompressedSparseData
            Same logical view containing the divided stored values.
        """
        return self.apply2(other, operator.truediv)

    def __add__(self, other: Any) -> Any:
        """Add an operand element-wise.

        Parameters
        ----------
        other : Any
            Dense broadcastable operand or identical-structure TCSR view.

        Returns
        -------
        TiledCompressedSparseData or Data
            Sparse result for an identical TCSR structure, otherwise dense.
        """
        return self.apply2(other, operator.add)

    def __sub__(self, other: Any) -> Any:
        """Subtract an operand element-wise.

        Parameters
        ----------
        other : Any
            Dense broadcastable operand or identical-structure TCSR view.

        Returns
        -------
        TiledCompressedSparseData or Data
            Sparse result for an identical TCSR structure, otherwise dense.
        """
        return self.apply2(other, operator.sub)

    def __rmul__(self, other: Any) -> Any:
        """Multiply stored values by a reflected operand.

        Parameters
        ----------
        other : Any
            Scalar or matching dense matrix.

        Returns
        -------
        TiledCompressedSparseData
            Same logical view containing the multiplied stored values.
        """
        return self.apply2(other, operator.mul, reverse=True)

    def __rtruediv__(self, other: Any) -> Any:
        """Divide a reflected operand by stored values element-wise.

        Parameters
        ----------
        other : Any
            Scalar or matching dense matrix.

        Returns
        -------
        TiledCompressedSparseData
            Same logical view containing the divided stored values.
        """
        return self.apply2(other, operator.truediv, reverse=True)

    def __radd__(self, other: Any) -> Any:
        """Add this matrix to a reflected operand element-wise.

        Parameters
        ----------
        other : Any
            Dense broadcastable operand or identical-structure TCSR view.

        Returns
        -------
        TiledCompressedSparseData or Data
            Sparse result for an identical TCSR structure, otherwise dense.
        """
        return self.apply2(other, operator.add, reverse=True)

    def __rsub__(self, other: Any) -> Any:
        """Subtract this matrix from a reflected operand element-wise.

        Parameters
        ----------
        other : Any
            Dense broadcastable operand or identical-structure TCSR view.

        Returns
        -------
        TiledCompressedSparseData or Data
            Sparse result for an identical TCSR structure, otherwise dense.
        """
        return self.apply2(other, operator.sub, reverse=True)

    def with_data(self, data: Data) -> Self:
        """Return a new view with replacement values in current-view order.

        Parameters
        ----------
        data : Data
            Replacement values in the same compressed-entry order as
            ``self.data``. Shape, dtype, and unit must match the current
            values.

        Returns
        -------
        TiledCompressedSparseData
            New view preserving the concrete type, logical orientation,
            sparse structure, tile metadata, and workspaces.

        Raises
        ------
        AssertionError
            If shape, dtype, or physical unit differs from ``self.data``.

        Notes
        -----
        Column-major mirror-ordered values are scattered back to canonical
        TCSR order with the existing transpose permutation. No sparse preprocessing or
        workspace allocation is performed.

        Examples
        --------
        .. code-block:: python

            >>> import jax
            >>> import jax.numpy as jnp
            >>> from brainevent._tcsr import TCSR
            >>> previous = jax.config.jax_explicit_x64_dtypes
            >>> jax.config.update("jax_explicit_x64_dtypes", "allow")
            >>> matrix = TCSR.fromdense(
            ...     jnp.asarray([[1.0, 0.0], [0.0, 2.0]])
            ... )
            >>> updated = matrix.with_data(matrix.data * 3.0)
            >>> updated.todense()
            Array([[3., 0.],
                   [0., 6.]], dtype=float32)
            >>> jax.config.update("jax_explicit_x64_dtypes", previous)
        """
        current_data = self.data
        assert getattr(data, "shape", ()) == getattr(current_data, "shape", ()), (
            "replacement data shape must match current data shape"
        )
        assert getattr(data, "dtype", None) == getattr(
            current_data, "dtype", None
        ), (
            "replacement data dtype must match current data dtype"
        )
        assert u.get_unit(data) == u.get_unit(current_data), (
            "replacement data unit must match current data unit"
        )

        return self._from_canonical_data(self._canonicalize_view_data(data))

    def _logical_csr_components(self) -> tuple[Data, Any, Any]:
        """Resolve a plain CSR representation of the current logical matrix."""
        if not self._transpose_state:
            return self._canonical_data, self._tcsr_indices, self._tcsr_indptr
        mirror = self._ensure_tcsc_mirror()
        data = self._mirror_data()
        return data, mirror.indices, mirror.indptr

    def _mirror_data(self) -> Data:
        """Resolve mirror-ordered values without expanding homogeneous data."""
        canonical_data: Any = self._canonical_data
        if int(canonical_data.size) == 1:
            return canonical_data
        return canonical_data[self._ensure_tcsc_mirror().permutation]

    @property
    def data(self) -> Data:
        """Return values ordered for the current compressed view.

        Returns
        -------
        Data
            Heterogeneous values in current-view CSR order, or the unchanged
            size-one homogeneous value.
        """
        return self._logical_csr_components()[0]

    @property
    def indices(self) -> Any:
        """Return coordinates ordered for the current compressed view.

        Returns
        -------
        jax.Array
            Minor-axis coordinates in current-view CSR order.
        """
        return self._logical_csr_components()[1]

    @property
    def indptr(self) -> Any:
        """Return pointers ordered for the current compressed view.

        Returns
        -------
        jax.Array
            Int64 row pointers in current-view CSR order.
        """
        return self._logical_csr_components()[2]

    @property
    def nse(self) -> int:
        """Return the number of explicitly stored entries.

        Returns
        -------
        int
            Number of coordinates represented by the sparse structure. This
            may exceed ``data.size`` for homogeneous storage.
        """
        return int(self._tcsr_indices.size)

    def sum(self, axis: Any = None) -> Data:
        """Return the sum of all logically represented matrix entries.

        Parameters
        ----------
        axis : int, sequence of int, or None, optional
            Axis or axes to reduce. Only ``None`` is currently supported.

        Returns
        -------
        Data
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

    @property
    def dtype(self) -> Any:
        """Return the canonical value dtype.

        Returns
        -------
        numpy.dtype
            Dtype of the canonical value buffer.
        """
        canonical_data: Any = self._canonical_data
        return canonical_data.dtype

    @property
    def has_tcsc_mirror(self) -> bool:
        """Return whether the data-free TCSC mirror is available.

        Returns
        -------
        bool
            ``True`` after the shared mirror has been materialized.
        """
        return self._tcs_buffers.tcsc is not None

    def todense(self) -> Data:
        """Materialize the current logical matrix as a dense array.

        Returns
        -------
        Data
            Dense matrix with the current logical shape.

        Notes
        -----
        Homogeneous values are broadcast to all ``nse`` represented entries.
        Duplicate coordinates are accumulated.
        """
        data, indices, indptr = self._logical_csr_components()
        return _csr_todense(data, indices, indptr, shape=self.shape)

    def __getitem__(self, index: Any) -> Data:
        """Extract logical rows as a dense array with CSR index semantics.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Row selector along axis 0. Negative indices wrap and concrete
            out-of-bounds indices raise ``IndexError``.

        Returns
        -------
        Data
            One-dimensional data for an integer selector, otherwise a dense
            matrix with one row per selected index.

        Raises
        ------
        IndexError
            If a concrete selector is non-integral or out of bounds.
        """
        rows = normalize_row_index(index, self.shape[0])
        data, indices, indptr = self._logical_csr_components()
        return csr_slice_rows(
            data,
            indices,
            indptr,
            rows,
            shape=self.shape,
            backend=self.backend,
        )

    def slice_rows(self, index: Any) -> Self:
        """Build an independent TCSR from selected logical rows.

        Parameters
        ----------
        index : int, list, tuple, array, or slice
            Concrete row selector along axis 0. A single integer produces a
            matrix with one row.

        Returns
        -------
        TiledCompressedSparseData
            Canonically oriented tiled matrix containing the selected rows.

        Raises
        ------
        RuntimeError
            If called with traced row indices because the output number of
            stored entries is data-dependent.

        Notes
        -----
        Tile metadata and mirror state are rebuilt for the new structure.
        Compute backend and backward-algorithm configuration are preserved.
        """
        rows = jnp.atleast_1d(normalize_row_index(index, self.shape[0]))
        data, indices, indptr = self._logical_csr_components()
        new_data, new_indices, new_indptr, shape = build_sub_csr(
            data,
            indices,
            indptr,
            rows,
            self.shape[1],
        )
        source = PlainCSR._from_parts(
            new_data,
            new_indices,
            new_indptr,
            shape=shape,
            backend=self.backend,
        )
        return type(self).from_sorted_csr(
            source,
            backend=self.backend,
            binary_backend=self.binary_backend,
            backward_algorithm=self.backward_algorithm,
        )

    def tocoo(self) -> u.sparse.COO:
        """Materialize the current logical matrix as a COO matrix.

        Returns
        -------
        brainunit.sparse.COO
            COO matrix with the current logical shape.

        Notes
        -----
        Homogeneous values are expanded to one value per coordinate. The
        conversion preserves the current logical orientation.
        """
        data, indices, indptr = self._logical_csr_components()
        data_array: Any = data
        rows, cols = _csr_to_coo(indices, indptr)
        if int(data_array.size) == 1:
            data = u.math.ones(indices.shape, dtype=data_array.dtype) * data
        return u.sparse.COO(
            (data, rows, cols),
            shape=self.shape,
            rows_sorted=True,
        )

    def tocsr(self) -> PlainCSR:
        """Materialize the current logical matrix as a plain CSR matrix.

        Returns
        -------
        brainevent._csr.main.CSR
            Plain CSR matrix with the current logical shape.

        Notes
        -----
        The returned CSR keeps homogeneous data compact and uses the current
        logical view's compressed ordering.
        """
        data, indices, indptr = self._logical_csr_components()
        return PlainCSR._from_parts(
            data,
            indices,
            indptr,
            shape=self.shape,
            backend=self.backend,
        )

    @classmethod
    def fromdense(
        cls,
        source: Any,
        *,
        nse: Optional[int] = None,
        index_dtype: Any = jnp.int32,
        **kwargs: Any,
    ) -> Self:
        """Build tiled shared storage from a dense matrix.

        Parameters
        ----------
        source : array-like
            Dense two-dimensional matrix.
        nse : int, optional
            Number of stored entries used by dense-to-CSR conversion.
        index_dtype : dtype, optional
            Coordinate dtype requested for dense-to-CSR conversion.
        **kwargs
            Construction options forwarded to ``cls``.

        Returns
        -------
        TiledCompressedSparseData
            New TCSR view.

        Notes
        -----
        This checked factory converts through plain CSR and sorts each row.
        TCSR canonical and mirror row pointers always use int64, so callers
        must enable ``jax_enable_x64=True`` before construction.

        See Also
        --------
        fromcsr : Build from a plain CSR and stably sort its rows.
        """
        csr = PlainCSR.fromdense(
            source,
            nse=nse,
            index_dtype=index_dtype,
            indptr_dtype=jnp.int64,
        )
        return cls.fromcsr(csr, **kwargs)

    @classmethod
    def fromcsr(
        cls,
        source: PlainCSR,
        **kwargs: Any,
    ) -> Self:
        """Sort a plain CSR matrix and build tiled shared storage.

        Parameters
        ----------
        source : brainevent._csr.main.CSR
            Plain CSR source. Target indices and corresponding heterogeneous
            values are sorted stably within each row.
        **kwargs
            Construction options forwarded to ``cls``.

        Returns
        -------
        TiledCompressedSparseData
            New TCSR view.

        Notes
        -----
        Use this entry point when source ordering is unknown. Sorting preserves
        the alignment between coordinates and heterogeneous values; homogeneous
        size-one data remains compact. ``jax_enable_x64=True`` is required.

        See Also
        --------
        from_sorted_csr : Skip sorting when the caller guarantees row order.
        """
        return cls(sort_csr(source), **kwargs)

    @classmethod
    def from_sorted_csr(
        cls,
        source: PlainCSR,
        **kwargs: Any,
    ) -> Self:
        """Build tiled shared storage from an already sorted plain CSR.

        Parameters
        ----------
        source : brainevent._csr.main.CSR
            Plain CSR with nondecreasing target indices in every row.
        **kwargs
            Construction options forwarded to ``cls``.

        Returns
        -------
        TiledCompressedSparseData
            New TCSR view that preserves the source entry order.

        Notes
        -----
        This trusted entry point does not sort or verify row-local ordering.
        If ordering cannot be guaranteed, use :meth:`fromcsr` instead.
        ``jax_enable_x64=True`` is required before construction.

        See Also
        --------
        fromcsr : Stably sort an arbitrary plain CSR source.
        """
        return cls(source, **kwargs)

    @classmethod
    def fromcsc(
        cls,
        source: PlainCSC,
        **kwargs: Any,
    ) -> Self:
        """Convert a plain CSC matrix and build checked tiled storage.

        Parameters
        ----------
        source : brainevent._csr.main.CSC
            Plain CSC source.
        **kwargs
            Construction options forwarded to ``cls``.

        Returns
        -------
        TiledCompressedSparseData
            New TCSR view.

        Notes
        -----
        This checked factory converts to CSR and stably sorts each row.
        ``jax_enable_x64=True`` is required before construction.
        """
        return cls.fromcsr(source.tocsr(), **kwargs)


@jax.tree_util.register_pytree_node_class
class TCSR(TiledCompressedSparseData):
    """Represent a unit-aware sparse matrix in tiled CSR storage.

    TCSR stores a logical matrix ``W`` with shape ``(num_pre, num_post)`` in a
    canonical tiled CSR structure. ``W.T`` is an O(1) logical view sharing the
    same values, tile metadata, workspaces, and lazily materialized TCSC mirror.

    Parameters
    ----------
    source : brainevent.CSR
        Plain CSR whose column indices are nondecreasing within every row.
        Direct construction trusts this ordering and does not verify it.
    backend : str, optional
        Backend used by floating-point sparse operations.
    binary_backend : str, optional
        Backend used by event-driven binary sparse operations.
    backward_algorithm : {"bptt", "pp_prop"}, optional
        Weight-gradient interpretation for floating event operands. Default is
        ``"bptt"``.

    Attributes
    ----------
    data : jax.Array or brainunit.Quantity
        Current-view compressed values. Heterogeneous storage has one value per
        entry; homogeneous storage has one shared value.
    indices : jax.Array
        Current-view minor-axis coordinates.
    indptr : jax.Array
        Current-view int64 row pointers.
    shape : tuple[int, int]
        Current logical matrix shape.
    nse : int
        Number of represented sparse entries.

    Notes
    -----
    Enable JAX x64 before construction:
    ``jax.config.update("jax_enable_x64", True)``. TCSR requires int64 row
    pointers for both canonical and mirror storage.

    Use :meth:`fromcsr` when CSR row ordering is unknown. Use direct
    construction or :meth:`from_sorted_csr` only when every row is already
    sorted by nondecreasing column index.

    For a logical shape ``(m, n)``, public matrix multiplication follows normal
    array conventions: ``W @ X`` accepts ``X.shape == (n, batch)`` and returns
    ``(m, batch)``, while ``X @ W`` accepts ``X.shape == (batch, m)`` and
    returns ``(batch, n)``. The same rule applies to ``W.T`` using its reversed
    logical shape. Internal neuron-batch and batch-neuron service layouts are
    selected automatically.

    See Also
    --------
    fromcsr : Build from CSR while enforcing row-local ordering.
    from_sorted_csr : Build from a CSR already known to be sorted.
    brainevent.CSR : Plain compressed sparse row representation.

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import brainevent
        >>> jax.config.update("jax_enable_x64", True)
        >>> source = brainevent.CSR(
        ...     (
        ...         jnp.asarray([2.0, 1.0], dtype=jnp.float32),
        ...         jnp.asarray([1, 0], dtype=jnp.int32),
        ...         jnp.asarray([0, 2], dtype=jnp.int64),
        ...     ),
        ...     shape=(1, 2),
        ... )
        >>> matrix = brainevent.TCSR.fromcsr(source)
        >>> matrix.todense()
        Array([[1., 2.]], dtype=float32)
    """

    __module__ = "brainevent"
    _compressed_format = "tcsr"

    def _uses_mirror_data_order(self) -> bool:
        """Return whether this TCSR view exposes mirror-ordered values."""
        return self._transpose_state

    def _float_matmul(self, other: Any, *, reverse: bool) -> Data:
        """Dispatch a float product using canonical CSR storage."""
        if isinstance(other, u.sparse.SparseMatrix):
            raise NotImplementedError("matmul between two sparse objects")

        from . import float as float_ops

        other = u.math.asarray(other)
        data, other = u.math.promote_dtypes(self._canonical_data, other)
        transpose = not self._transpose_state if reverse else self._transpose_state
        if other.ndim == 1:
            return float_ops.csrmv(
                data,
                self._tcsr_indices,
                self._tcsr_indptr,
                other,
                shape=self._base_shape,
                transpose=transpose,
                backend=self.backend,
            )
        if other.ndim == 2:
            operand = other.T if reverse else other
            result = float_ops.csrmm(
                data,
                self._tcsr_indices,
                self._tcsr_indptr,
                operand,
                shape=self._base_shape,
                transpose=transpose,
                backend=self.backend,
            )
            return result.T if reverse else result
        raise NotImplementedError(f"matmul with object of shape {other.shape}")

    def __matmul__(self, other: Any) -> Data:
        """Multiply this TCSR view by a dense or binary vector or matrix.

        For a current logical shape ``(m, n)``, a vector must have shape
        ``(n,)`` and produces ``(m,)``. A rank-two operand must have shape
        ``(n, batch)`` and produces ``(m, batch)``. These rules also apply to a
        transposed view using its reversed logical shape; JAX dispatch selects
        the required internal batch layout automatically.

        Parameters
        ----------
        other : array-like, brainunit.Quantity, or brainevent.BinaryArray
            Vector of shape ``(self.shape[1],)`` or matrix of shape
            ``(self.shape[1], batch)`` on the right.

        Returns
        -------
        Data
            Vector of shape ``(self.shape[0],)`` or matrix of shape
            ``(self.shape[0], batch)``.

        Raises
        ------
        NotImplementedError
            If ``other`` is sparse or has unsupported rank.

        See Also
        --------
        __rmatmul__ : Multiply a left operand by this logical view.
        transpose : Create a shared transposed logical view.
        """
        if not isinstance(other, BinaryArray):
            return self._float_matmul(other, reverse=False)
        events = other.value
        if events.ndim not in (1, 2):
            raise NotImplementedError(
                f"binary matmul with object of shape {events.shape}"
            )
        from . import binary

        service_events = (
            events
            if events.ndim == 1 or not self._transpose_state
            else events.T
        )
        result: Any
        if events.ndim == 1:
            result = binary.binary_csrmv(
                self._canonical_data,
                self._tcsr_indices,
                self._tcsr_indptr,
                service_events,
                shape=self._base_shape,
                workspace=self._tcs_buffers.tcsr_workspace,
                local_targets=self._tcs_buffers.tcsr_local_targets,
                tile_offsets=self._tcs_buffers.tcsr_tile_offsets,
                buffers=self._tcs_buffers,
                transpose=self._transpose_state,
                backend=self.binary_backend,
                backward_algorithm=self.backward_algorithm,
            )
        else:
            result = binary.binary_csrmm(
                self._canonical_data,
                self._tcsr_indices,
                self._tcsr_indptr,
                service_events,
                shape=self._base_shape,
                workspace=self._tcs_buffers.tcsr_workspace,
                local_targets=self._tcs_buffers.tcsr_local_targets,
                tile_offsets=self._tcs_buffers.tcsr_tile_offsets,
                buffers=self._tcs_buffers,
                transpose=self._transpose_state,
                backend=self.binary_backend,
                backward_algorithm=self.backward_algorithm,
            )
        return (
            result
            if events.ndim == 1 or not self._transpose_state
            else result.T
        )

    def __rmatmul__(self, other: Any) -> Data:
        """Multiply a dense or binary vector or matrix by this TCSR view.

        For a current logical shape ``(m, n)``, a vector must have shape
        ``(m,)`` and produces ``(n,)``. A rank-two operand must have shape
        ``(batch, m)`` and produces ``(batch, n)``. The same rule applies to a
        transposed view; internal layout conversion is automatic.

        Parameters
        ----------
        other : array-like, brainunit.Quantity, or brainevent.BinaryArray
            Vector of shape ``(self.shape[0],)`` or matrix of shape
            ``(batch, self.shape[0])`` on the left.

        Returns
        -------
        Data
            Vector of shape ``(self.shape[1],)`` or matrix of shape
            ``(batch, self.shape[1])``.

        Raises
        ------
        NotImplementedError
            If ``other`` is sparse or has unsupported rank.

        See Also
        --------
        __matmul__ : Multiply this logical view by a right operand.
        transpose : Create a shared transposed logical view.
        """
        if not isinstance(other, BinaryArray):
            return self._float_matmul(other, reverse=True)
        events = other.value
        if events.ndim not in (1, 2):
            raise NotImplementedError(
                f"binary matmul with object of shape {events.shape}"
            )

        from . import binary

        service_events = (
            events
            if events.ndim == 1 or not self._transpose_state
            else events.T
        )
        compute_transpose = not self._transpose_state
        result: Any
        if events.ndim == 1:
            result = binary.binary_csrmv(
                self._canonical_data,
                self._tcsr_indices,
                self._tcsr_indptr,
                service_events,
                shape=self._base_shape,
                workspace=self._tcs_buffers.tcsr_workspace,
                local_targets=self._tcs_buffers.tcsr_local_targets,
                tile_offsets=self._tcs_buffers.tcsr_tile_offsets,
                buffers=self._tcs_buffers,
                transpose=compute_transpose,
                backend=self.binary_backend,
                backward_algorithm=self.backward_algorithm,
            )
        else:
            result = binary.binary_csrmm(
                self._canonical_data,
                self._tcsr_indices,
                self._tcsr_indptr,
                service_events,
                shape=self._base_shape,
                workspace=self._tcs_buffers.tcsr_workspace,
                local_targets=self._tcs_buffers.tcsr_local_targets,
                tile_offsets=self._tcs_buffers.tcsr_tile_offsets,
                buffers=self._tcs_buffers,
                transpose=compute_transpose,
                backend=self.binary_backend,
                backward_algorithm=self.backward_algorithm,
            )
        return (
            result
            if events.ndim == 1 or not self._transpose_state
            else result.T
        )

    def dt2t(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Expand logical row values over the current view's sparse slots.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Logical row values with shape ``(self.shape[0],)``.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-slot values in the current view's compressed order.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-slot products in the same order as ``w_dim_arr``.

        See Also
        --------
        dt2t_transposed : Expand logical column values over sparse slots.
        """
        from .dt2t import csrmv_dt2t

        if not self._transpose_state:
            result = csrmv_dt2t(
                y_dim_arr,
                w_dim_arr,
                self._tcsr_indices,
                self._tcsr_indptr,
                shape=self._base_shape,
                transpose=False,
                buffers=self._tcs_buffers,
                backend=self.backend,
            )
        else:
            mirror = self._ensure_tcsc_mirror()
            result = csrmv_dt2t(
                y_dim_arr,
                w_dim_arr,
                mirror.indices,
                mirror.indptr,
                shape=(int(self.shape[0]), int(self.shape[1])),
                transpose=False,
                backend=self.backend,
            )
        return cast(Union[jax.Array, u.Quantity], result)

    def dt2t_transposed(
        self,
        y_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
        w_dim_arr: Union[jax.Array, np.ndarray, u.Quantity],
    ) -> Union[jax.Array, u.Quantity]:
        """Expand logical column values over the current view's sparse slots.

        Parameters
        ----------
        y_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Logical column values with shape ``(self.shape[1],)``.
        w_dim_arr : jax.Array, numpy.ndarray, or brainunit.Quantity
            Per-slot values in the current view's compressed order.

        Returns
        -------
        jax.Array or brainunit.Quantity
            Per-slot products in the same order as ``w_dim_arr``.

        See Also
        --------
        dt2t : Expand logical row values over sparse slots.
        """
        from .dt2t import _csrmv_dt2t_prepared, csrmv_dt2t

        if not self._transpose_state:
            result = csrmv_dt2t(
                y_dim_arr,
                w_dim_arr,
                self._tcsr_indices,
                self._tcsr_indptr,
                shape=self._base_shape,
                transpose=True,
                buffers=self._tcs_buffers,
                backend=self.backend,
            )
        else:
            mirror = self._ensure_tcsc_mirror()
            inverse_permutation = jnp.empty_like(mirror.permutation).at[
                mirror.permutation
            ].set(jnp.arange(self.nse, dtype=mirror.permutation.dtype))
            result = _csrmv_dt2t_prepared(
                y_dim_arr,
                w_dim_arr,
                self._tcsr_indices,
                self._tcsr_indptr,
                inverse_permutation,
                shape=(int(self.shape[0]), int(self.shape[1])),
                backend=self.backend,
            )
        return cast(Union[jax.Array, u.Quantity], result)

    def update_on_pre(
        self,
        pre_spike: Data,
        post_trace: Data,
        w_min: Any = None,
        w_max: Any = None,
    ) -> Self:
        """Reject presynaptic-triggered STDP until TCSR supports it.

        Parameters
        ----------
        pre_spike : Data
            Presynaptic spike vector.
        post_trace : Data
            Postsynaptic eligibility trace.
        w_min : Any, optional
            Reserved lower clipping bound.
        w_max : Any, optional
            Reserved upper clipping bound.

        Raises
        ------
        NotImplementedError
            Always, because TCSR presynaptic STDP is not implemented.
        """
        del pre_spike, post_trace, w_min, w_max
        raise NotImplementedError("TCSR.update_on_pre is not implemented")

    def update_on_post(
        self,
        pre_trace: Data,
        post_spike: Data,
        w_min: Any = None,
        w_max: Any = None,
    ) -> Self:
        """Reject postsynaptic-triggered STDP until TCSR supports it.

        Parameters
        ----------
        pre_trace : Data
            Presynaptic eligibility trace.
        post_spike : Data
            Postsynaptic spike vector.
        w_min : Any, optional
            Reserved lower clipping bound.
        w_max : Any, optional
            Reserved upper clipping bound.

        Raises
        ------
        NotImplementedError
            Always, because TCSR postsynaptic STDP is not implemented.
        """
        del pre_trace, post_spike, w_min, w_max
        raise NotImplementedError("TCSR.update_on_post is not implemented")

    def solve(self, b: Data, *args: Any, **kwargs: Any) -> Data:
        """Reject sparse solve until TCSR has a supported solver contract.

        Parameters
        ----------
        b : array-like or brainunit.Quantity
            Right-hand side supplied by the caller.
        *args
            Reserved positional solver options.
        **kwargs
            Reserved keyword solver options.

        Raises
        ------
        NotImplementedError
            Always, because TCSR solve is not implemented.
        """
        del b, args, kwargs
        raise NotImplementedError("TCSR.solve is not implemented")

    def tocsc(self) -> PlainCSC:
        """Materialize the current logical matrix as a plain CSC matrix.

        Returns
        -------
        brainevent._csr.main.CSC
            Plain CSC matrix with the same logical values and shape.

        Notes
        -----
        A canonical TCSR view materializes the shared TCSC mirror lazily. A
        transposed view can reuse canonical CSR structure directly. Homogeneous
        data remains a size-one value buffer.
        """
        if self._transpose_state:
            data = self._canonical_data
            indices = self._tcsr_indices
            indptr = self._tcsr_indptr
        else:
            mirror = self._ensure_tcsc_mirror()
            data = self._mirror_data()
            indices = mirror.indices
            indptr = mirror.indptr
        return PlainCSC._from_parts(
            data,
            indices,
            indptr,
            shape=self.shape,
            backend=self.backend,
        )

    def transpose(self, axes: Any = None) -> TCSR:
        """Return a transposed TCSR view sharing the same storage.

        Parameters
        ----------
        axes : None or tuple, optional
            Accepted transpose axes. Only ``None`` and ``(1, 0)`` are valid.

        Returns
        -------
        TCSR
            TCSR view with reversed logical shape and orientation state.

        Raises
        ------
        ValueError
            If ``axes`` does not describe a matrix transpose.

        Notes
        -----
        The returned view shares canonical values, tile metadata, workspace,
        and mirror state. No sparse coordinates are rebuilt.
        """
        if axes not in (None, (1, 0)):
            raise ValueError("TCSR transpose axes must be None or (1, 0)")
        return self._new_view(
            transpose_state=not self._transpose_state
        )

    @property
    def T(self) -> TCSR:
        """Return the transposed TCSR view.

        Returns
        -------
        TCSR
            O(1) logical transpose sharing all underlying TCS storage.

        See Also
        --------
        transpose : Method form with optional axes validation.
        """
        return self.transpose()
