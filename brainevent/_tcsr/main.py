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

"""Sorted CSR matrices with lazily prepared CUDA binary MM tiles."""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._csr.config import CSRConfig, _structure_device
from brainevent._csr.main import CSC, CSR
from brainevent._error import PerformanceWarning
from brainevent._event import as_event_operand
from brainevent._op.lowering_target import cuda_available as _cuda_available
from brainevent._op.util import CUDA_VALUE_DTYPES
from brainevent._structure import is_concrete
from brainevent._tcsr._dispatch import dispatch_binary_mm
from brainevent._tcsr.binary.combined.planner import INT32_MAX, tile_schedule
from brainevent._tcsr.preprocess import (
    TileGeometry,
    build_tile_geometries,
    sort_csr,
    validate_preparation,
)

__all__ = ['TCSR']


def _source_argument(value: Any, source: Any, name: str) -> Any:
    """Resolve a legacy source keyword without inspecting array values."""
    if source is not None:
        if value is not None:
            raise TypeError(f'Pass only one of {name!r} and legacy \'source\'.')
        return source
    return value


_REMOVED = object()
_BACKWARD_ALGORITHM_REMOVED = (
    "TCSR(backward_algorithm=...) was removed. Weight gradients now follow the "
    "event type: BinaryArray events use their activity mask, and "
    "brainevent.WeightedArray events use their values. Wrap events in "
    "WeightedArray to obtain value-weighted gradients (the former 'pp_prop')."
)


_UNPREPARED_WARNED: set = set()


def _warn_unprepared(shape: tuple[int, int], batch: int, transpose: bool) -> None:
    """Warn once per structure shape, batch and direction about a traced miss."""
    key = (shape, batch, transpose)
    if key in _UNPREPARED_WARNED:
        return
    _UNPREPARED_WARNED.add(key)
    product = 'events @ matrix' if transpose else 'matrix @ events'
    warnings.warn(
        f'TCSR {shape}: tiles for `{product}` with batch {batch} are not prepared and the '
        'matrix structure is traced (for example a jit argument), so the CSR kernel runs '
        f'instead. Call `matrix = matrix.prepare_binary_mm({batch}, '
        f'transpose={transpose})` outside jit, or run one eager product first.',
        PerformanceWarning, stacklevel=4,
    )


def _construction_backend(
    backend: str | None, binary_backend: str | None, inherited: str | None = None,
) -> str | None:
    """Normalize legacy backend spelling into the single CSR preference."""
    backend = 'jax_raw' if backend == 'jax' else backend
    binary_backend = 'jax_raw' if binary_backend == 'jax' else binary_backend
    if backend is not None and binary_backend is not None and backend != binary_backend:
        raise ValueError('backend and legacy binary_backend must select the same backend.')
    selected = backend if backend is not None else binary_backend
    if selected is None:
        selected = inherited
    return 'jax_raw' if selected == 'jax' else selected


@jax.tree_util.register_pytree_node_class
class TCSR(CSR):
    """A sorted CSR matrix supporting prepared CUDA binary matrix products.

    Parameters
    ----------
    csr : CSR
        Source matrix. Columns are stably sorted within each row and
        heterogeneous weights follow their edges. Duplicate edges remain
        separate; homogeneous weights retain their scalar/singleton shape.
    backend : str or None, optional
        Backend preference. ``None`` inherits the source preference.
    source : CSR or None, optional
        Legacy keyword alias for ``csr``. Supply only one source argument.
    binary_backend : str or None, optional
        Legacy alias for the single ``backend`` preference. The old ``'jax'``
        spelling maps to ``'jax_raw'``; conflicting preferences raise ValueError.

    Notes
    -----
    Construction requires concrete structure and does not modify ``csr``.
    Sorting moves edges; :attr:`sort_permutation` records the move so that
    per-edge arrays kept in the source order can be realigned.

    Binary event products ``events @ W`` (primary tiles) and ``W @ events``
    (mirror tiles) use a tiled CUDA kernel when the backend is ``None`` or
    ``'cuda_raw'`` and a CUDA device is present. Tile geometry depends on the
    batch size and the accumulator width (8 bytes for float64, else 4), so
    it is built lazily, for the widths that actually run: on the first eager
    product of a batch, or at trace time when the matrix is a concrete
    closure. Each width costs about ``2 * nnz`` bytes plus one offset per row
    and tile. When the matrix is a traced ``jit`` argument, missing geometry
    cannot be built; the product then uses the CSR kernel and emits a
    :class:`~brainevent.PerformanceWarning`. Call :meth:`prepare_binary_mm`
    outside ``jit`` to avoid that. Other products, and every product on
    other backends or platforms, use the inherited CSR implementation.

    Transposition retains the public CSR contract: ``W.T`` is a CSC and
    ``W.T.T`` is a CSR.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent import BinaryArray, CSR, TCSR
        >>> matrix = TCSR(CSR.fromdense(jnp.eye(2), backend='jax_raw'))
        >>> (BinaryArray(jnp.ones((3, 2), dtype=bool)) @ matrix).shape
        (3, 2)
        >>>
        >>> source = CSR(jnp.array([1., 2.]), jnp.array([1, 0]), jnp.array([0, 2]), shape=(1, 2))
        >>> sorted_matrix = TCSR(source)
        >>> sorted_matrix.sort_permutation
        array([1, 0])
        >>> delays = jnp.array([5, 7])  # one value per source edge
        >>> delays[sorted_matrix.sort_permutation]
        Array([7, 5], dtype=int32)
    """

    __module__ = 'brainevent'
    _sort_permutation: np.ndarray | None = None

    def __init__(
        self, csr: CSR | None = None, *, backend: str | None = None,
        source: CSR | None = None, binary_backend: str | None = None,
        backward_algorithm: Any = _REMOVED,
    ):
        if backward_algorithm is not _REMOVED:
            raise TypeError(_BACKWARD_ALGORITHM_REMOVED)
        csr = _source_argument(csr, source, 'csr')
        if not isinstance(csr, CSR):
            raise TypeError('TCSR requires a CSR source matrix.')
        backend = _construction_backend(backend, binary_backend, csr.backend)
        data, indices, permutation = sort_csr(csr)
        # The source validated (or explicitly trusted) its structure, and sorting
        # within rows keeps it valid, so skip a second nnz-sized host read.
        super().__init__(
            data, indices, csr.indptr, shape=csr.shape,
            backend=backend,
            config=csr.config,
            indptr_dtype=csr.indptr.dtype,
            check_structure=False,
        )
        self._sort_permutation = permutation

    @property
    def sort_permutation(self) -> np.ndarray | None:
        """Source edge slot of each stored edge, recorded when sorting.

        Returns
        -------
        numpy.ndarray or None
            Read-only host integer array ``p`` of length ``nse`` such that
            ``self.indices == source.indices[p]`` (and likewise for
            heterogeneous ``data``), where ``source`` is the CSR passed to the
            constructor. Index any per-edge array that follows the source
            order, such as delays or edge state, with ``p`` to align it with
            this matrix. ``None`` means no edge moved: the source rows were
            already sorted, or this matrix was not produced by sorting (for
            example :meth:`from_sorted_csr`, value updates or pytree
            reconstruction, which keep their input's order).

        Examples
        --------
        .. code-block:: python

            >>> import jax.numpy as jnp
            >>> from brainevent import CSR, TCSR
            >>> source = CSR(jnp.array([1., 2.]), jnp.array([1, 0]), jnp.array([0, 2]), shape=(1, 2))
            >>> TCSR(source).sort_permutation
            array([1, 0])
        """
        return self._sort_permutation

    @classmethod
    def from_sorted_csr(
        cls, csr: CSR | None = None, *, backend: str | None = None,
        source: CSR | None = None, binary_backend: str | None = None,
    ) -> TCSR:
        """Wrap sorted CSR arrays and initialize any missing primary geometry.

        Parameters
        ----------
        csr : CSR
            A CSR with nondecreasing columns within every row. The caller
            guarantees sortedness and the validity of its structure caches.
        backend : str or None, optional
            Preference to attach; ``None`` inherits the source preference.
        source : CSR or None, optional
            Legacy keyword alias for ``csr``; supply only one source.
        binary_backend : str or None, optional
            Legacy alias for ``backend`` under the constructor's conflict rules.

        Returns
        -------
        TCSR
            A matrix sharing source arrays and configuration with an independent
            cache container. Valid same-structure geometry is shared; missing
            tiles and the mirror are prepared lazily.

        Notes
        -----
        Sortedness is trusted and never checked, so no host read happens here.
        :attr:`sort_permutation` is ``None``. This entry point also preserves
        the source configuration when overriding ``backend``.
        """
        csr = _source_argument(csr, source, 'csr')
        if not isinstance(csr, CSR):
            raise TypeError('from_sorted_csr requires a CSR source matrix.')
        result = cls._from_parts(
            csr.data, csr.indices, csr.indptr, shape=csr.shape,
            backend=_construction_backend(backend, binary_backend, csr.backend),
            config=csr.config,
            cache=csr._cache.copy(),
        )
        return result

    @classmethod
    def fromcsr(
        cls, csr: CSR | None = None, *, backend: str | None = None,
        source: CSR | None = None, binary_backend: str | None = None,
    ) -> TCSR:
        """Create a TCSR by stably sorting a CSR source.

        Parameters
        ----------
        csr : CSR
            Source matrix; heterogeneous weights follow their sorted edges.
        backend : str or None, optional
            Compute preference; ``None`` inherits the source preference.
        source : CSR or None, optional
            Legacy alias for ``csr``; supply only one source argument.
        binary_backend : str or None, optional
            Legacy alias for ``backend``; conflicting preferences are rejected.

        Returns
        -------
        TCSR
            Sorted matrix sharing the source configuration.
        """
        return cls(csr, backend=backend, source=source, binary_backend=binary_backend)

    @classmethod
    def fromcsc(
        cls, csc: CSC | None = None, *, backend: str | None = None,
        source: CSC | None = None, binary_backend: str | None = None,
    ) -> TCSR:
        """Create a sorted TCSR from a CSC matrix.

        Parameters
        ----------
        csc : CSC
            Source column-compressed matrix, converted through public CSR.
        backend : str or None, optional
            Compute preference; ``None`` inherits the source preference.
        source : CSC or None, optional
            Legacy alias for ``csc``; supply only one source argument.
        binary_backend : str or None, optional
            Legacy alias for ``backend``; conflicting preferences are rejected.

        Returns
        -------
        TCSR
            Sorted row-compressed matrix, preserving source units and tuning.
        """
        csc = _source_argument(csc, source, 'csc')
        if not isinstance(csc, CSC):
            raise TypeError('fromcsc requires a CSC source matrix.')
        return cls(csc.tocsr(), backend=backend, binary_backend=binary_backend)

    @classmethod
    def fromdense(
        cls, mat: Any = None, *, nse: int | None = None, index_dtype: Any = jnp.int32,
        indptr_dtype: Any = 'auto', backend: str | None = None,
        precompute_weight_indices: bool = False,
        config: CSRConfig | None = None,
        source: Any = None, binary_backend: str | None = None,
    ) -> TCSR:
        """Create a sorted TCSR through the public CSR dense factory.

        Parameters
        ----------
        mat : array_like or brainunit.Quantity
            Dense two-dimensional matrix.
        nse : int or None, optional
            Number of stored edges, as in :meth:`CSR.fromdense`.
        index_dtype : dtype, optional
            Column dtype; only int32 is supported.
        indptr_dtype : {'auto', int32, int64}, optional
            Row-pointer precision policy.
        backend : str or None, optional
            Compute backend preference.
        precompute_weight_indices : bool, optional
            Build the shared mirror before returning.
        config : CSRConfig or None, optional
            CSR scheduler parameters.
        source : array_like or brainunit.Quantity, optional
            Legacy alias for ``mat``; supply only one dense source argument.
        binary_backend : str or None, optional
            Legacy alias for ``backend`` under the constructor's conflict rules.

        Returns
        -------
        TCSR
            Sorted matrix; tile geometry is prepared lazily.
        """
        mat = _source_argument(mat, source, 'mat')
        if mat is None:
            raise TypeError('fromdense requires a dense mat source.')
        backend = _construction_backend(backend, binary_backend)
        # CSR's factory supplies sorted columns and owns its padding policy.
        csr = CSR.fromdense(
            mat, nse=nse, index_dtype=index_dtype, indptr_dtype=indptr_dtype,
            backend=backend,
            precompute_weight_indices=precompute_weight_indices,
            config=config,
        )
        return cls.from_sorted_csr(csr)

    def materialize_tcsc_mirror(self) -> TCSR:
        """Build the public CSR mirror through the legacy method name.

        Returns
        -------
        TCSR
            The result of :meth:`build_weight_indices`, sharing canonical
            values, structure and valid primary tiles. No private TCSC exists.
        """
        return self.build_weight_indices()

    @property
    def has_tcsc_mirror(self) -> bool:
        """Report whether the public CSR mirror has already been prepared.

        Returns
        -------
        bool
            Whether the structure cache carries its public mirror. This
            query never materializes a mirror or reads device array contents.
        """
        return self._cache.mirror is not None

    def slice_rows(self, index: Any) -> TCSR:
        """Select rows and return a TCSR with a fresh structure cache.

        Parameters
        ----------
        index : int, slice, or array_like
            Concrete row selector, following :meth:`CSR.slice_rows`.

        Returns
        -------
        TCSR
            Sorted row selection with lazy tiles and a lazy mirror.
        """
        return type(self).from_sorted_csr(super().slice_rows(index))

    def prepare_binary_mm(
        self, batch: int, *, transpose: bool | None = None, dtype: Any = None,
    ) -> TCSR:
        """Prepare the tiles one batch size needs, for use under ``jit``.

        Eager products prepare their tiles on first use, so this is needed
        only when the matrix is passed to a ``jit``-compiled function as an
        argument (its structure is then traced and tiles cannot be built).

        Parameters
        ----------
        batch : int
            Event batch size, from 0 to 65535. Zero needs no tiles.
        transpose : bool or None, optional
            ``True`` prepares tiles for ``events @ W`` (primary traversal),
            ``False`` for ``W @ events`` (builds the mirror), ``None`` both.
        dtype : dtype or None, optional
            Weight dtype the product will use; defaults to ``self.dtype``.
            float64 selects narrower tiles than the other dtypes.

        Returns
        -------
        TCSR
            Same matrix with an independent cache container that holds the
            prepared tiles. The original object is unchanged.

        Raises
        ------
        ValueError
            Called while tracing, the structure is traced, or the arguments
            are unsupported.

        Examples
        --------
        .. code-block:: python

            >>> import jax
            >>> import jax.numpy as jnp
            >>> from brainevent import BinaryArray, TCSR
            >>> matrix = TCSR.fromdense(jnp.eye(3)).prepare_binary_mm(8, transpose=True)
            >>> step = jax.jit(lambda m, x: BinaryArray(x) @ m)
            >>> step(matrix, jnp.ones((8, 3), dtype=bool)).shape
            (8, 3)
        """
        value_dtype = self.dtype if dtype is None else dtype
        batch = validate_preparation(batch, value_dtype)
        if transpose is not None and not isinstance(transpose, (bool, np.bool_)):
            raise ValueError('transpose must be True, False, or None.')
        if not is_concrete(self.indices, self.indptr):
            raise ValueError('prepare_binary_mm requires concrete structure outside JIT.')
        result = type(self)._from_parts(
            self.data, self.indices, self.indptr, shape=self.shape,
            backend=self.backend,
            config=self.config, cache=self._cache.copy(),
        )
        result._sort_permutation = self._sort_permutation
        directions = (True, False) if transpose is None else (bool(transpose),)
        for direction in directions:
            if batch and result._get_tile_geometry(
                batch, transpose=direction, dtype=value_dtype, prepare=True,
            ) is None:
                raise ValueError('prepare_binary_mm requires concrete structure outside JIT.')
        return result

    def _get_tile_geometry(
        self, batch: int, *, transpose: bool, dtype: Any = None, prepare: bool = False,
    ) -> TileGeometry | None:
        """Return one schedule's geometry, optionally building just that width.

        ``prepare=False`` is a pure query. ``prepare=True`` builds a missing
        width (and, for ``transpose=False``, the mirror) when the structure is
        concrete, publishing it to the shared structure cache. Empty batches
        never need tiles and return ``None``.
        """
        value_dtype = np.dtype(self.dtype if dtype is None else dtype)
        if value_dtype not in CUDA_VALUE_DTYPES or not 0 < batch <= 65535:
            return None
        target = self.shape[1] if transpose else self.shape[0]
        schedule = tile_schedule(target, batch, 8 if value_dtype == np.dtype('float64') else 4)
        if transpose:
            pointers, indices = self.indptr, self.indices
        else:
            mirror = self._cache.mirror
            if mirror is None:
                if any(not key[0] for key in self._cache.tile_geometries):
                    raise ValueError('TCSR mirror tiles require their matching mirror structure.')
                if not prepare:
                    return None
                mirror = self._mirror()
                if mirror is None:
                    return None
            pointers, indices = mirror[0], mirror[1]
        offset_dtype = np.dtype(pointers.dtype)
        key = (transpose, schedule.tile_width, offset_dtype)
        geometry = self._cache.tile_geometries.get(key)
        if geometry is None:
            if not prepare or not is_concrete(indices, pointers):
                return None
            # Like CSR._mirror(), a concrete closure may prepare at trace time;
            # the cache then receives eager arrays and never a tracer.
            with jax.ensure_compile_time_eval():
                prepared = build_tile_geometries(
                    indices, pointers, shape=self.shape, transpose=transpose,
                    widths=(schedule.tile_width,),
                )
            geometry = prepared[key]
            # Publish a new dict so siblings never observe a partial update.
            self._cache.tile_geometries = {**self._cache.tile_geometries, key: geometry}
        if not isinstance(geometry, TileGeometry):
            raise ValueError('Invalid TCSR binary MM geometry entry.')
        if (geometry.key != key or geometry.shape != self.shape or geometry.nnz != self.nse):
            raise ValueError('TCSR binary MM geometry does not match its structure.')
        source = self.shape[0] if transpose else self.shape[1]
        if (geometry.local_targets.shape != (self.nse,)
                or geometry.local_targets.dtype != jnp.uint16
                or geometry.tile_offsets.shape != (source * (schedule.tile_count + 1),)
                or np.dtype(geometry.tile_offsets.dtype) != offset_dtype):
            raise ValueError('TCSR binary MM tile arrays do not match the prepared metadata.')
        return geometry

    def _binary_mm_batch(self, x: Any, weighted: bool, *, transpose: bool) -> int | None:
        """Return the batch of a tile-eligible operand from static metadata only.

        ``x`` keeps its public layout: ``(batch, rows)`` for ``x @ W``
        (``transpose=True``) and ``(cols, batch)`` for ``W @ x``.
        """
        if weighted or x.ndim != 2:
            return None
        source, batch = (x.shape[1], x.shape[0]) if transpose else x.shape
        if source != (self.shape[0] if transpose else self.shape[1]):
            return None
        if not 0 < batch <= 65535 or max(self.shape) > INT32_MAX:
            return None
        if not (np.dtype(self.dtype) in CUDA_VALUE_DTYPES
                and np.dtype(x.dtype) in (np.dtype('bool'), np.dtype('float32'))
                and self.indices.ndim == 1 and self.indices.dtype == jnp.int32
                and np.dtype(self.indptr.dtype) in (np.dtype('int32'), np.dtype('int64'))):
            return None
        return int(batch)

    def _tiles_reachable(self) -> bool:
        """Whether a product could run the CUDA tiles; checked before any preparation."""
        if self.backend is not None and self.backend != 'cuda_raw':
            return False
        if not _cuda_available():
            return False
        device = _structure_device(self.indices)
        return device is None or getattr(device, 'platform', None) == 'gpu'

    def _binary_mm_geometry(self, x: Any, weighted: bool, *, transpose: bool) -> TileGeometry | None:
        """Select, lazily preparing, the tiles of an eligible binary product."""
        batch = self._binary_mm_batch(x, weighted, transpose=transpose)
        if batch is None or not self._tiles_reachable():
            return None
        geometry = self._get_tile_geometry(batch, transpose=transpose, prepare=True)
        if geometry is None:
            _warn_unprepared(self.shape, batch, transpose)
        return geometry

    def __matmul__(self, other):
        """Compute ``self @ other`` with automatic binary MM selection.

        Parameters
        ----------
        other : array_like, brainunit.Quantity, or event array
            Right operand with the same shapes and semantics as CSR.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The public CSR product. Eligible CUDA Binary MM uses mirror tiles;
            all other calls retain the original operand for the parent route.
        """
        event = as_event_operand(other)
        if event is not None:
            geometry = self._binary_mm_geometry(event[0], event[1], transpose=False)
            if geometry is not None:
                return dispatch_binary_mm(self, other, transpose=False, plan=geometry)
        return super().__matmul__(other)

    def __rmatmul__(self, other):
        """Compute ``other @ self`` with automatic binary MM selection.

        Parameters
        ----------
        other : array_like, brainunit.Quantity, or event array
            Left operand with the same shapes and semantics as CSR.

        Returns
        -------
        jax.Array or brainunit.Quantity
            The public CSR product. Eligible CUDA Binary MM uses primary tiles;
            all other calls retain the original operand for the parent route.
        """
        event = as_event_operand(other)
        if event is not None:
            geometry = self._binary_mm_geometry(event[0], event[1], transpose=True)
            if geometry is not None:
                return dispatch_binary_mm(self, other, transpose=True, plan=geometry)
        return super().__rmatmul__(other)

    def tree_flatten(self):
        """Flatten CSR payload followed by the dynamic arrays of each tile.

        Returns
        -------
        children : tuple
            Parent payload followed by local targets and offsets per plan.
        aux_data : tuple
            Parent metadata and sorted immutable tile descriptions.
        """
        children, parent_aux = super().tree_flatten()
        keys = sorted(self._cache.tile_geometries, key=lambda k: (k[0], k[1], k[2].name))
        plans = tuple(self._cache.tile_geometries[key] for key in keys)
        leaves = tuple(leaf for plan in plans for leaf in (plan.local_targets, plan.tile_offsets))
        return (*children, *leaves), (parent_aux, tuple(plan.metadata for plan in plans))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Restore leaves without conversion, validation or host inspection.

        Parameters
        ----------
        aux_data : tuple
            Immutable metadata produced by :meth:`tree_flatten`.
        children : tuple
            CSR and tile leaves, including JAX placeholders when applicable.

        Returns
        -------
        TCSR
            Reconstructed matrix without running public construction.
        """
        parent_aux, descriptions = aux_data
        # Tile leaves trail the parent's; the parent validates its own count.
        parent_size = len(children) - 2 * len(descriptions)
        if parent_size < 0:
            raise ValueError(
                f'TCSR.tree_unflatten expects at least {2 * len(descriptions)} leaves, got {len(children)}.'
            )
        # Calling the classmethod implementation with cls preserves TCSR
        # while using exactly the parent's storage/placeholder protocol.
        result = CSR.tree_unflatten.__func__(cls, parent_aux, children[:parent_size])
        for i, metadata in enumerate(descriptions):
            offset = parent_size + 2 * i
            plan = TileGeometry.from_metadata(metadata, children[offset], children[offset + 1])
            result._cache.tile_geometries[plan.key] = plan
        return result
