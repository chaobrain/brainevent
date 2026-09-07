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

from typing import Any, Optional, Self

import brainunit as u
import jax
import jax.numpy as jnp

from brainevent._csr.main import CSC as PlainCSC
from brainevent._csr.main import CSR as PlainCSR
from brainevent._data import DataRepresentation
from brainevent._event import BinaryArray
from brainevent._misc import _csr_to_coo, _csr_todense
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
    """Own shared tile-compressed data for TCSR logical views.

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

    Notes
    -----
    The forward TCSR structure owns the canonical value order. The TCSC mirror
    contains structure and a permutation into that value array, but no second
    value array.
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
        This constructor does not verify or sort row-local target indices.
        Use :meth:`fromcsr` for an unchecked plain CSR source.
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
        self._transposed = False
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
        transposed: bool,
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
        obj._transposed = bool(transposed)
        obj.backend = backend
        obj.binary_backend = binary_backend
        obj.backward_algorithm = backward_algorithm
        shape = obj._base_shape[::-1] if obj._transposed else obj._base_shape
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
        """Flatten shared arrays and workspaces for the JAX pytree protocol."""
        aux = (
            self._base_shape,
            self._transposed,
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
        """Restore a TCS view from JAX pytree leaves and static metadata."""
        (
            base_shape,
            transposed,
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
            transposed=transposed,
            backend=backend,
            binary_backend=binary_backend,
            backward_algorithm=backward_algorithm,
        )

    def _new_view(
        self,
        *,
        transposed: bool,
    ) -> Self:
        """Return a same-type view sharing every TCS leaf with this instance."""
        return type(self)._from_tcs_parts(
            self._canonical_data,
            self._tcsr_indices,
            self._tcsr_indptr,
            self._tcs_buffers,
            base_shape=self._base_shape,
            transposed=transposed,
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

        Notes
        -----
        The operation is idempotent. All views that share this matrix's
        structural buffers observe the same mirror.
        """
        self._ensure_tcsc_mirror()
        return self

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

        canonical_data = data
        if int(getattr(data, "size", 1)) != 1 and self._uses_mirror_data_order():
            mirror = self._ensure_tcsc_mirror()
            data_mantissa = u.get_mantissa(data)
            canonical_mantissa = jnp.empty_like(data_mantissa).at[
                mirror.permutation
            ].set(
                data_mantissa
            )
            canonical_data = u.maybe_decimal(
                canonical_mantissa * u.get_unit(data)
            )

        return type(self)._from_tcs_parts(
            canonical_data,
            self._tcsr_indices,
            self._tcsr_indptr,
            self._tcs_buffers,
            base_shape=self._base_shape,
            transposed=self._transposed,
            backend=self.backend,
            binary_backend=self.binary_backend,
            backward_algorithm=self.backward_algorithm,
        )

    def _logical_csr_components(self) -> tuple[Data, Any, Any]:
        """Resolve a plain CSR representation of the current logical matrix."""
        if not self._transposed:
            return self._canonical_data, self._tcsr_indices, self._tcsr_indptr
        mirror = self._ensure_tcsc_mirror()
        data = self._mirror_data()
        return data, mirror.indices, mirror.indptr

    def _mirror_data(self) -> Data:
        """Resolve mirror-ordered values without expanding homogeneous data."""
        if int(self._canonical_data.size) == 1:
            return self._canonical_data
        return self._canonical_data[self._ensure_tcsc_mirror().permutation]

    @property
    def data(self) -> Data:
        """Return values ordered for the current compressed view."""
        return self._logical_csr_components()[0]

    @property
    def indices(self) -> Any:
        """Return coordinates ordered for the current compressed view."""
        return self._logical_csr_components()[1]

    @property
    def indptr(self) -> Any:
        """Return pointers ordered for the current compressed view."""
        return self._logical_csr_components()[2]

    @property
    def nse(self) -> int:
        """Return the number of explicitly stored entries."""
        return int(self._tcsr_indices.size)

    @property
    def dtype(self) -> Any:
        """Return the canonical value dtype."""
        return self._canonical_data.dtype

    @property
    def has_tcsc_mirror(self) -> bool:
        """Return whether the data-free TCSC mirror is available."""
        return self._tcs_buffers.tcsc is not None

    def todense(self) -> Data:
        """Materialize the current logical matrix as a dense array.

        Returns
        -------
        Data
            Dense matrix with the current logical shape.
        """
        data, indices, indptr = self._logical_csr_components()
        return _csr_todense(data, indices, indptr, shape=self.shape)

    def tocoo(self) -> u.sparse.COO:
        """Materialize the current logical matrix as a COO matrix.

        Returns
        -------
        brainunit.sparse.COO
            COO matrix with the current logical shape.
        """
        data, indices, indptr = self._logical_csr_components()
        rows, cols = _csr_to_coo(indices, indptr)
        if int(data.size) == 1:
            data = u.math.ones(indices.shape, dtype=data.dtype) * data
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
        indptr_dtype: Any = "auto",
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
        indptr_dtype : {"auto", int32, int64}, optional
            Row-pointer dtype policy used for dense-to-CSR conversion.
        **kwargs
            Construction options forwarded to ``cls``.

        Returns
        -------
        TiledCompressedSparseData
            New TCSR view.
        """
        csr = PlainCSR.fromdense(
            source,
            nse=nse,
            index_dtype=index_dtype,
            indptr_dtype=indptr_dtype,
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
        """
        return cls.fromcsr(source.tocsr(), **kwargs)


@jax.tree_util.register_pytree_node_class
class TCSR(TiledCompressedSparseData):
    """Provide the TCSR view over shared tile-compressed storage."""

    _compressed_format = "tcsr"

    def _uses_mirror_data_order(self) -> bool:
        """Return whether this TCSR view exposes mirror-ordered values."""
        return self._transposed

    def __matmul__(self, other: Any) -> Data:
        """Multiply this TCSR view by a binary vector or matrix.

        Parameters
        ----------
        other : brainevent.BinaryArray
            Binary vector or conventional neuron-batch matrix on the right.

        Returns
        -------
        Data
            Product in conventional matrix-multiplication layout.

        Raises
        ------
        NotImplementedError
            If ``other`` is not a binary array or has unsupported rank.
        """
        if not isinstance(other, BinaryArray):
            raise NotImplementedError(
                f"matmul with object of type {type(other).__name__}"
            )
        events = other.value
        if events.ndim not in (1, 2):
            raise NotImplementedError(
                f"binary matmul with object of shape {events.shape}"
            )
        from . import binary

        service_events = (
            events
            if events.ndim == 1 or not self._transposed
            else events.T
        )
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
                transpose=self._transposed,
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
                transpose=self._transposed,
                backend=self.binary_backend,
                backward_algorithm=self.backward_algorithm,
            )
        return result if events.ndim == 1 or not self._transposed else result.T

    def __rmatmul__(self, other: Any) -> Data:
        """Multiply a binary vector or BN-layout matrix by this TCSR view.

        Parameters
        ----------
        other : brainevent.BinaryArray
            Binary vector or batch-neuron matrix on the left.

        Returns
        -------
        Data
            Product in batch-neuron layout.

        Raises
        ------
        NotImplementedError
            If ``other`` is not a binary array or has unsupported rank.
        """
        if not isinstance(other, BinaryArray):
            raise NotImplementedError(
                f"matmul with object of type {type(other).__name__}"
            )
        events = other.value
        if events.ndim not in (1, 2):
            raise NotImplementedError(
                f"binary matmul with object of shape {events.shape}"
            )

        from . import binary

        service_events = (
            events
            if events.ndim == 1 or not self._transposed
            else events.T
        )
        compute_transpose = not self._transposed
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
        return result if events.ndim == 1 or not self._transposed else result.T

    def tocsc(self) -> PlainCSC:
        """Materialize the current logical matrix as a plain CSC matrix.

        Returns
        -------
        brainevent._csr.main.CSC
            Plain CSC matrix with the same logical values and shape.
        """
        if self._transposed:
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
        """
        if axes not in (None, (1, 0)):
            raise ValueError("TCSR transpose axes must be None or (1, 0)")
        return self._new_view(transposed=not self._transposed)

    @property
    def T(self) -> TCSR:
        """Return the transposed TCSR view."""
        return self.transpose()
