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

"""Sorted CSR matrices with automatic primary and lazy mirror MM tiles."""

from __future__ import annotations

from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._csr.config import CSRConfig, _backend_preference, _structure_device
from brainevent._csr.main import CSC, CSR
from brainevent._event import as_event_operand
from brainevent._structure import is_concrete
from ._dispatch import dispatch_binary_mm
from .binary.combined.planner import tile_schedule
from .preprocess import TileGeometry, build_tile_geometries, geometry_widths, sort_csr

__all__ = ['TCSR']

_VALUE_DTYPES = frozenset(np.dtype(t) for t in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64))


def _source_argument(value: Any, source: Any, name: str) -> Any:
    """Resolve a legacy source keyword without inspecting array values."""
    if source is not None:
        if value is not None:
            raise TypeError(f'Pass only one of {name!r} and legacy \'source\'.')
        return source
    return value


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
    Primary geometry is prepared during construction for all supported MM
    batches and weight dtypes. Mirror geometry is prepared lazily after the
    parent provides its mirror. No explicit tile preparation is required.
    Other products use the inherited CSR implementation.
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
    """

    __module__ = 'brainevent'

    def __init__(
        self, csr: CSR | None = None, *, backend: str | None = None,
        source: CSR | None = None, binary_backend: str | None = None,
    ):
        csr = _source_argument(csr, source, 'csr')
        if not isinstance(csr, CSR):
            raise TypeError('TCSR requires a CSR source matrix.')
        backend = _construction_backend(backend, binary_backend, csr.backend)
        data, indices = sort_csr(csr)
        # Host metadata has no device preference. Use the concrete weights'
        # placement to prepare inherited CUDA MM, following CSR's policy.
        preferred = _backend_preference('gpu', backend)
        needs_cuda_info = preferred == 'cuda_raw' or (backend is None and preferred != 'jax_raw')
        device = None
        if needs_cuda_info and _structure_device(indices) is None:
            device = _structure_device(u.get_mantissa(data))
        super().__init__(
            data, indices, csr.indptr, shape=csr.shape,
            backend=backend,
            backward_algorithm=csr.backward_algorithm, config=csr.config,
            indptr_dtype=csr.indptr.dtype, device=device,
        )
        self._initialize_primary_tiles()

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
            cache container. Valid same-structure geometry is shared; a concrete
            source with no primary geometry prepares it before returning.
            The mirror remains lazy.

        Notes
        -----
        Sortedness is trusted; initial primary preparation still reads a new
        concrete source's structure. Existing complete geometry and internal
        pytree reconstruction never require host preparation. Traced new
        structures keep the applicable parent route without preparing tiles.
        This entry point also preserves the source configuration when
        overriding ``backend``. After selecting CUDA from a configuration
        prepared only for JAX, place the matrix on the target GPU and call
        :meth:`with_rebuilt_config` outside JIT before using inherited CUDA
        MM. Its ``device`` parameter selects configuration placement without
        moving the matrix arrays. :meth:`fromcsr` prepares configuration for
        its requested backend during construction.
        """
        csr = _source_argument(csr, source, 'csr')
        if not isinstance(csr, CSR):
            raise TypeError('from_sorted_csr requires a CSR source matrix.')
        result = cls._from_parts(
            csr.data, csr.indices, csr.indptr, shape=csr.shape,
            backend=_construction_backend(backend, binary_backend, csr.backend),
            backward_algorithm=csr.backward_algorithm, config=csr.config,
            cache=csr._cache.copy(),
        )
        result._initialize_primary_tiles()
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
            Sorted matrix with freshly bound structure configuration.
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
        backward_algorithm: str = 'weighted_float', precompute_weight_indices: bool = False,
        config: CSRConfig | None = None, device: Any = None,
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
        backward_algorithm : {'binary', 'weighted_float'}, optional
            Public CSR weight-gradient convention. Default ``'weighted_float'``
            uses event values; ``'binary'`` uses their activity mask.
        precompute_weight_indices : bool, optional
            Build the shared mirror before returning.
        config : CSRConfig or None, optional
            CSR scheduler parameters.
        device : jax.Device or None, optional
            Device used to prepare CSR configuration.
        source : array_like or brainunit.Quantity, optional
            Legacy alias for ``mat``; supply only one dense source argument.
        binary_backend : str or None, optional
            Legacy alias for ``backend`` under the constructor's conflict rules.

        Returns
        -------
        TCSR
            Sorted matrix with primary geometry ready for every supported batch.
        """
        mat = _source_argument(mat, source, 'mat')
        if mat is None:
            raise TypeError('fromdense requires a dense mat source.')
        backend = _construction_backend(backend, binary_backend)
        # CSR's factory supplies sorted columns and owns its padding policy.
        csr = CSR.fromdense(
            mat, nse=nse, index_dtype=index_dtype, indptr_dtype=indptr_dtype,
            backend=backend, backward_algorithm=backward_algorithm,
            precompute_weight_indices=precompute_weight_indices,
            config=config, device=device,
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
            Sorted row selection with new primary geometry and a lazy mirror.
        """
        return type(self).from_sorted_csr(super().slice_rows(index))

    def _ensure_tile_geometries(self, indices, indptr, *, transpose: bool) -> None:
        """Publish one direction's complete geometry without caching tracers."""
        target = self.shape[1] if transpose else self.shape[0]
        offset_dtype = np.dtype(indptr.dtype)
        expected = {(transpose, width, offset_dtype) for width in geometry_widths(target)}
        existing = {key for key in self._cache.tile_geometries if key[0] == transpose}
        if existing:
            if existing != expected:
                raise ValueError('Incomplete TCSR tile geometry table for its structure.')
            return
        if not is_concrete(indices, indptr):
            return
        # Like CSR._mirror(), concrete closures can prepare at trace time;
        # their cache receives eager arrays and cannot leak a tracer.
        with jax.ensure_compile_time_eval():
            prepared = build_tile_geometries(
                indices, indptr, shape=self.shape, transpose=transpose,
            )
        if set(prepared) != expected:
            raise ValueError('TCSR preparation did not produce complete tile geometry.')
        # All arrays must be built successfully before siblings see new keys.
        geometries = dict(self._cache.tile_geometries)
        geometries.update(prepared)
        self._cache.tile_geometries = geometries

    def _initialize_primary_tiles(self) -> None:
        """Prepare primary geometry for a concrete new structure."""
        self._ensure_tile_geometries(self.indices, self.indptr, transpose=True)

    def _mirror(self):
        """Use CSR's mirror, then lazily prepare its concrete tile geometry."""
        if self._cache.mirror is None and any(not key[0] for key in self._cache.tile_geometries):
            raise ValueError('TCSR mirror tiles require their matching mirror structure.')
        mirror = super()._mirror()
        if mirror is not None:
            indptr, indices, _ = mirror
            self._ensure_tile_geometries(indices, indptr, transpose=False)
        return mirror

    def diag_add(self, other):
        """Add the diagonal through CSR and initialize concrete result geometry.

        Parameters
        ----------
        other : array_like
            Diagonal values following the public CSR shape and unit contract.

        Returns
        -------
        TCSR
            New matrix with obsolete tiles discarded. Concrete results carry
            complete primary geometry; traced results use CSR until geometry
            is available, without reading traced structure on the host.
        """
        result = super().diag_add(other)
        result._initialize_primary_tiles()
        return result

    def prepare_binary_mm(
        self, batch: int, *, transpose: bool | None = None, dtype: Any = None,
    ) -> TCSR:
        """Optionally materialize mirror geometry using the legacy preparation API.

        Parameters
        ----------
        batch : int
            Legacy batch argument, validated in the range 0..65535. Geometry
            already covers every supported batch, so this does not bind it.
        transpose : bool or None, optional
            True retains primary geometry; False or None also materializes
            the public mirror and its geometry on the returned object.
        dtype : dtype or None, optional
            Legacy supported weight dtype argument. Both accumulator families
            are already covered; this does not change matrix values.

        Returns
        -------
        TCSR
            Same canonical matrix with an independent cache container. Normal
            matrix calls do not require this compatibility method.

        Raises
        ------
        ValueError
            Called while tracing, or the legacy arguments are unsupported.
        """
        from .preprocess import validate_preparation

        validate_preparation(batch, self.dtype if dtype is None else dtype)
        if transpose is not None and not isinstance(transpose, (bool, np.bool_)):
            raise ValueError('transpose must be True, False, or None.')
        result = type(self)._from_parts(
            self.data, self.indices, self.indptr, shape=self.shape,
            backend=self.backend, backward_algorithm=self.backward_algorithm,
            config=self.config, cache=self._cache.copy(),
        )
        result._initialize_primary_tiles()
        if transpose is None or not transpose:
            if result._mirror() is None:
                raise ValueError('prepare_binary_mm requires concrete mirror structure outside JIT.')
        return result

    def _get_tile_geometry(
        self, batch: int, *, transpose: bool, dtype: Any = None,
    ) -> TileGeometry | None:
        """Query one schedule's geometry without constructing or reading arrays."""
        value_dtype = np.dtype(self.dtype if dtype is None else dtype)
        if value_dtype not in _VALUE_DTYPES or not 0 <= batch <= 65535:
            return None
        target = self.shape[1] if transpose else self.shape[0]
        schedule = tile_schedule(target, batch, 8 if value_dtype == np.dtype('float64') else 4)
        mirror = None if transpose else self._cache.mirror
        if not transpose and mirror is None:
            if any(not key[0] for key in self._cache.tile_geometries):
                raise ValueError('TCSR mirror tiles require their matching mirror structure.')
            return None
        if transpose:
            pointers = self.indptr
        else:
            assert mirror is not None
            pointers = mirror[0]
        offset_dtype = np.dtype(pointers.dtype)
        key = (transpose, schedule.tile_width, offset_dtype)
        geometry = self._cache.tile_geometries.get(key)
        if geometry is None:
            if any(key[0] == transpose for key in self._cache.tile_geometries):
                raise ValueError('Incomplete TCSR tile geometry table for its structure.')
            return None
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

    def _get_binary_mm_plan(
        self, batch: int, *, transpose: bool, dtype: Any = None,
    ) -> TileGeometry | None:
        """Query geometry through the former private batch-plan entry point."""
        return self._get_tile_geometry(batch, transpose=transpose, dtype=dtype)

    def _is_binary_mm_candidate(self, x: Any, weighted: bool, *, transpose: bool) -> bool:
        """Check cheap static MM conditions before requesting a mirror."""
        if weighted or x.ndim != 2 or not 0 <= x.shape[1] <= 65535:
            return False
        source = self.shape[0] if transpose else self.shape[1]
        if x.shape[0] != source or max(self.shape) > (1 << 31) - 1:
            return False
        return (np.dtype(self.dtype) in _VALUE_DTYPES
                and np.dtype(x.dtype) in (np.dtype('bool'), np.dtype('float32'))
                and self.indices.ndim == 1 and self.indices.dtype == jnp.int32
                and np.dtype(self.indptr.dtype) in (np.dtype('int32'), np.dtype('int64')))

    def _can_use_binary_mm(self, x: Any, weighted: bool, *, transpose: bool) -> bool:
        """Check a candidate's already available geometry without preparing it."""
        return (self._is_binary_mm_candidate(x, weighted, transpose=transpose)
                and self._get_tile_geometry(x.shape[1], transpose=transpose) is not None)

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
            x, weighted = event
            if self._is_binary_mm_candidate(x, weighted, transpose=False):
                mirror = self._mirror()
                if mirror is not None:
                    geometry = self._get_tile_geometry(x.shape[1], transpose=False)
                    if geometry is not None:
                        return self._dispatch_binary_mm(other, transpose=False, plan=geometry)
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
            x, weighted = event
            if x.ndim == 2:
                if self._can_use_binary_mm(x.T, weighted, transpose=True):
                    geometry = self._get_tile_geometry(x.shape[0], transpose=True)
                    assert geometry is not None
                    return self._dispatch_binary_mm(other, transpose=True, plan=geometry)
        return super().__rmatmul__(other)

    def _dispatch_binary_mm(self, other: Any, *, transpose: bool, plan: TileGeometry):
        """Select actual platform/backend resources, preserving the original operand."""
        return dispatch_binary_mm(self, other, transpose=transpose, plan=plan)

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
        _, _, _, has_mirror, has_diag = parent_aux
        parent_size = 4 + 3 * has_mirror + 4 * has_diag
        expected = parent_size + 2 * len(descriptions)
        if len(children) != expected:
            raise ValueError(f'TCSR.tree_unflatten expects {expected} leaves, got {len(children)}.')
        # Calling the classmethod implementation with cls preserves TCSR
        # while using exactly the parent's storage/placeholder protocol.
        result = CSR.tree_unflatten.__func__(cls, parent_aux, children[:parent_size])
        for i, metadata in enumerate(descriptions):
            offset = parent_size + 2 * i
            plan = TileGeometry.from_metadata(metadata, children[offset], children[offset + 1])
            result._cache.tile_geometries[plan.key] = plan
        return result
