"""Immutable CSR scheduling snapshots and structure-bound MM preparation.

This module performs no persistence, device discovery, or kernel loading.
Prepared masks are pytree leaves; kernel specifications contain only hashable
metadata so tracing and differentiation can retain a configuration snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import operator
from typing import Any, ClassVar, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'CSRConfig', 'CSRDeviceInfo', 'CSRKernelSpec',
    'MMStructureSpec', 'MMStructurePlan', 'MMCallPlan',
]

_INT32_MAX = int(np.iinfo(np.int32).max)
_INT64_MAX = int(np.iinfo(np.int64).max)
_MM_ITERATIONS = 16
_MM_TASK_EDGES = 32 * _MM_ITERATIONS
_MM_L2_DIVISOR = 4
_MM_MAX_CHUNK = 65535
_MM_GEOMETRY_VERSION = 1


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be an integer') from exc
    if result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return result


@dataclass(frozen=True)
class CSRDeviceInfo:
    """Device properties captured outside tracing.

    Parameters
    ----------
    platform : str
        JAX platform, usually ``'gpu'`` or ``'cpu'``.
    device_kind : str
        Device model used to select persisted tuning.
    l2_bytes : int
        GPU L2 capacity in bytes; zero when unavailable.
    sms : int
        GPU multiprocessor count; zero when unavailable.
    device_id : int or None, optional
        Informational local device ordinal.
    """

    platform: str
    device_kind: str
    l2_bytes: int = 0
    sms: int = 0
    device_id: int | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        _integer(self.l2_bytes, 'l2_bytes')
        _integer(self.sms, 'sms')
        if self.device_id is not None:
            _integer(self.device_id, 'device_id')


@dataclass(frozen=True)
class MMStructureSpec:
    """Hashable statistics for one compressed traversal.

    Parameters
    ----------
    rows, cols, nnz : int
        Traversal dimensions and number of stored edges.
    max_row_nnz : int
        Longest compressed row.
    group, long_threshold, long_tasks : int
        Lane group, long-row threshold, and all-active task count per batch.
    geometry_version : int, optional
        Fixed implementation version, included in static kernel identity.
    """

    rows: int
    cols: int
    nnz: int
    max_row_nnz: int
    group: int
    long_threshold: int
    long_tasks: int
    geometry_version: int = field(default=_MM_GEOMETRY_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> MMStructureSpec:
        """Validate scalar structure metadata against the native geometry.

        Returns
        -------
        MMStructureSpec
            This specification when its dimensions and constants are valid.

        Raises
        ------
        ValueError
            Dimensions, statistics or geometry version are inconsistent.
        """
        for name in ('rows', 'cols', 'nnz', 'max_row_nnz', 'long_tasks'):
            _integer(getattr(self, name), name)
        if self.rows > _INT32_MAX or self.cols > _INT32_MAX:
            raise ValueError('MM dimensions exceed int32 range')
        if (self.geometry_version != _MM_GEOMETRY_VERSION or self.group not in (4, 8, 16, 32)
                or self.long_threshold != _MM_ITERATIONS * self.group):
            raise ValueError('MM group and long_threshold disagree with the fixed kernel geometry')
        if self.max_row_nnz > self.nnz or self.long_tasks > self.nnz:
            raise ValueError('MM row/task statistics exceed nnz')
        if self.nnz and (not self.rows or not self.cols or not self.max_row_nnz):
            raise ValueError('nonempty MM structure requires nonzero dimensions and row length')
        return self


@dataclass(frozen=True)
class MMStructurePlan(MMStructureSpec):
    """A traversal specification with its prepared long-row mask.

    Parameters
    ----------
    rows, cols, nnz, max_row_nnz, group, long_threshold, long_tasks : int
        See :class:`MMStructureSpec`.
    long_mask : array_like
        ``uint32[ceil(rows / 32)]`` bitmask carried as a pytree array leaf.
    """

    long_mask: Any = field(compare=False, hash=False, repr=False)

    def specification(self) -> MMStructureSpec:
        """Return the array-free traversal metadata.

        Returns
        -------
        MMStructureSpec
            Metadata suitable for a primitive parameter.
        """
        return MMStructureSpec(self.rows, self.cols, self.nnz, self.max_row_nnz,
                               self.group, self.long_threshold, self.long_tasks,
                               geometry_version=self.geometry_version)


@dataclass(frozen=True)
class MMCallPlan:
    """Static launch parameters and scratch sizes for one MM signature.

    Parameters
    ----------
    rows, cols, nnz : int
        Selected compressed traversal dimensions.
    group, long_threshold, long_tasks, chunk, task_capacity : int
        Native hybrid MM launch attributes.
    workspace_words, short_rows_size : int
        Scratch lengths in uint32 and int32 elements, respectively.
    batch : int
        Number of event vectors after batching transformations.
    dtype : numpy.dtype
        Weight and output dtype.
    weighted : bool
        Whether the launch uses event amplitudes rather than binary activity.
    """

    rows: int
    cols: int
    nnz: int
    group: int
    long_threshold: int
    long_tasks: int
    chunk: int
    task_capacity: int
    workspace_words: int
    short_rows_size: int
    batch: int
    dtype: np.dtype[Any]
    weighted: bool


class _Scheduler:
    block_size: int
    fixed_scatter_blocks: int
    tpr_threshold: int
    task_nnz: int

    def compile_flags(self) -> list[str]:
        """Return the MV/SDDMV scheduler compiler definitions.

        Returns
        -------
        list of str
            Four ``-DBE_HYBRID_*`` compiler flags.
        """
        return [f'-DBE_HYBRID_BLOCK_SIZE={self.block_size}',
                f'-DBE_HYBRID_FIXED_SCATTER_BLOCKS={self.fixed_scatter_blocks}',
                f'-DBE_HYBRID_TPR_THRESHOLD={self.tpr_threshold}',
                f'-DBE_HYBRID_TASK_NNZ={self.task_nnz}']

    def module_suffix(self) -> str:
        """Return the configuration-dependent MV/SDDMV module suffix.

        Returns
        -------
        str
            A suffix identifying the four compiled scheduler constants.
        """
        return (f'_b{self.block_size}_s{self.fixed_scatter_blocks}'
                f'_t{self.tpr_threshold}_n{self.task_nnz}')

    def mv_capacity(self, nse: int, n_rows: int, max_row_nnz: int | None = None) -> int:
        """Bound the MV task queue from static dimensions.

        Parameters
        ----------
        nse, n_rows : int
            Stored edge count and number of compressed rows.
        max_row_nnz : int or None, optional
            Upper bound on the longest row.

        Returns
        -------
        int
            A capacity covering every compatible row structure.
        """
        nse, n_rows = _integer(nse, 'nse'), _integer(n_rows, 'n_rows')
        if max_row_nnz is not None:
            max_row_nnz = _integer(max_row_nnz, 'max_row_nnz')
        if nse == 0 or n_rows == 0:
            return 0
        heavy = min(n_rows, nse // (self.tpr_threshold + 1))
        bound = -(-nse // self.task_nnz) + heavy
        if max_row_nnz is not None:
            if max_row_nnz <= self.tpr_threshold:
                return 0
            bound = min(bound, heavy * -(-min(max_row_nnz, nse) // self.task_nnz))
        if bound > _INT32_MAX:
            raise ValueError('binary task capacity exceeds int32 range')
        return bound


def _validate_scheduler(config: _Scheduler) -> None:
    block = _integer(config.block_size, 'block_size', 1)
    if block % 32 or block > 1024:
        raise ValueError('block_size must be a multiple of 32 and must not exceed 1024')
    _integer(config.fixed_scatter_blocks, 'fixed_scatter_blocks', 1)
    _integer(config.tpr_threshold, 'tpr_threshold')
    _integer(config.task_nnz, 'task_nnz', 1)


def _format_parameters(config: _Scheduler) -> str:
    return (f'CSRConfig(block_size={config.block_size}, fixed_scatter_blocks={config.fixed_scatter_blocks}, '
            f'tpr_threshold={config.tpr_threshold}, task_nnz={config.task_nnz})')


@dataclass(frozen=True)
class CSRKernelSpec(_Scheduler):
    """Hashable configuration selected for one primitive traversal.

    Parameters
    ----------
    block_size, fixed_scatter_blocks, tpr_threshold, task_nnz : int, optional
        MV/SDDMV scheduler constants.
    structure : MMStructureSpec or None, optional
        Selected primary or mirror metadata.
    device_info : CSRDeviceInfo or None, optional
        Previously captured target device properties.
    """

    block_size: int = 256
    fixed_scatter_blocks: int = 2048
    tpr_threshold: int = 128
    task_nnz: int = 4096
    structure: MMStructureSpec | None = None
    device_info: CSRDeviceInfo | None = None

    def __post_init__(self) -> None:
        _validate_scheduler(self)


@dataclass(frozen=True)
class _BenchmarkRecord:
    """Timing of one candidate scheduling configuration.

    Parameters
    ----------
    config : CSRConfig
        Candidate configuration.
    elapsed_ms, per_call_us : float
        Total duration and duration per kernel call.
    task_capacity : int or None, optional
        Workspace capacity used by the tuning workload.
    """

    config: CSRConfig
    elapsed_ms: float
    per_call_us: float
    task_capacity: int | None = None


def _structure_plan(lengths: np.ndarray[Any, Any], cols: int) -> MMStructurePlan:
    rows = len(lengths)
    eligible = lengths[(lengths > 0) & (lengths <= _MM_TASK_EDGES)]
    quarter = (int(eligible.sum()) // len(eligible) + 3) // 4 if eligible.size else 128
    group = 4 if quarter <= 4 else 8 if quarter <= 8 else 16 if quarter <= 16 else 32
    threshold = _MM_ITERATIONS * group
    long_rows = lengths > threshold
    long_tasks = int(((lengths[long_rows] + _MM_TASK_EDGES - 1) // _MM_TASK_EDGES).sum())
    mask = np.zeros((rows + 31) // 32, dtype=np.uint32)
    positions = np.flatnonzero(long_rows)
    np.bitwise_or.at(mask, positions // 32, np.left_shift(np.uint32(1), (positions % 32).astype(np.uint32)))
    return MMStructurePlan(rows, cols, int(lengths.sum()), int(lengths.max(initial=0)),
                           group, threshold, long_tasks, jnp.asarray(mask))


@lru_cache(maxsize=256)
def _call_plan(structure: MMStructureSpec, device: CSRDeviceInfo, batch: int,
               dtype: np.dtype[Any], weighted: bool) -> MMCallPlan:
    if batch == 0 or structure.rows == 0 or structure.cols == 0 or structure.nnz == 0:
        return MMCallPlan(structure.rows, structure.cols, structure.nnz, structure.group,
                          structure.long_threshold, structure.long_tasks, 1, 0, 0, 0, batch, dtype, weighted)
    value_bytes = dtype.itemsize
    fit = (device.l2_bytes // _MM_L2_DIVISOR) // (structure.cols * value_bytes) if structure.cols else batch
    chunk = max(1, min(fit, batch, _MM_MAX_CHUNK))
    capacity = chunk * structure.long_tasks
    groups = (batch + chunk - 1) // chunk
    queue_bytes = ((capacity * value_bytes + 3) // 4 * 4) if weighted else 0
    workspace_bytes = ((8 * batch + 7) // 8 * 8) + 8 * groups + 12 * capacity + queue_bytes
    short_rows = batch * structure.rows
    if capacity > _INT64_MAX or workspace_bytes > _INT64_MAX or short_rows > _INT64_MAX:
        raise ValueError('MM plan or scratch size exceeds int64 range')
    return MMCallPlan(structure.rows, structure.cols, structure.nnz, structure.group,
                      structure.long_threshold, structure.long_tasks, chunk, capacity,
                      workspace_bytes // 4, short_rows, batch, dtype, weighted)


@dataclass(frozen=True)
class _ConfigAux:
    primary: CSRKernelSpec
    mirror: MMStructureSpec | None
    records: tuple[_BenchmarkRecord, ...] = field(compare=False, hash=False)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class CSRConfig(_Scheduler):
    """A CSR scheduling snapshot with optional prepared traversal plans.

    Parameters
    ----------
    block_size, fixed_scatter_blocks, tpr_threshold, task_nnz : int, optional
        The four persisted MV/SDDMV scheduler constants.
    benchmark_records : tuple of _BenchmarkRecord, optional
        Diagnostic tuning results, excluded from kernel identity.
    device_info : CSRDeviceInfo or None, optional
        Eagerly captured device metadata; this class never queries a device.
    primary, mirror : MMStructurePlan or None, optional
        Prepared plans for the original and mirrored compressed layouts.

    Notes
    -----
    This object is a pytree, not a hashable static argument. Use
    :meth:`kernel_args` to separate a static specification from its array leaf.
    """

    block_size: int = 256
    fixed_scatter_blocks: int = 2048
    tpr_threshold: int = 128
    task_nnz: int = 4096
    benchmark_records: tuple[_BenchmarkRecord, ...] = field(default=(), compare=False, repr=False)
    device_info: CSRDeviceInfo | None = None
    primary: MMStructurePlan | None = field(default=None, compare=False, repr=False)
    mirror: MMStructurePlan | None = field(default=None, compare=False, repr=False)
    __hash__: ClassVar[Any] = None

    def __post_init__(self) -> None:
        _validate_scheduler(self)

    @property
    def mm_iterations(self) -> int:
        """Return the fixed native MM iteration limit.

        Returns
        -------
        int
            ``16``; this is an implementation constant, not a tuning option.
        """
        return _MM_ITERATIONS

    @property
    def mm_task_edges(self) -> int:
        """Return the fixed maximum number of edges in one MM task.

        Returns
        -------
        int
            ``512``, matching the native kernel's 32 lanes times 16 iterations.
        """
        return _MM_TASK_EDGES

    @property
    def geometry_version(self) -> int:
        """Return the fixed MM geometry version used by prepared plans.

        Returns
        -------
        int
            Current native planner geometry version.
        """
        return _MM_GEOMETRY_VERSION

    def validate(self) -> CSRConfig:
        """Validate tuning parameters and any prepared layout metadata.

        Returns
        -------
        CSRConfig
            This configuration when all available metadata is consistent.

        Raises
        ------
        ValueError
            A mask or paired layout has inconsistent dimensions or dtype.

        Notes
        -----
        This checks array shape and dtype only; it never copies mask contents
        to the host and is safe with tracer masks.
        """
        _validate_scheduler(self)
        for plan in (self.primary, self.mirror):
            if plan is None:
                continue
            plan.validate()
            if getattr(plan.long_mask, 'shape', None) != ((plan.rows + 31) // 32,):
                raise ValueError('prepared CSRConfig long-row mask has the wrong shape')
            if getattr(plan.long_mask, 'dtype', None) != np.dtype(np.uint32):
                raise ValueError('prepared CSRConfig long-row mask must have uint32 dtype')
        if self.primary is not None and self.mirror is not None:
            if ((self.primary.rows, self.primary.cols, self.primary.nnz)
                    != (self.mirror.cols, self.mirror.rows, self.mirror.nnz)):
                raise ValueError('primary and mirror plan dimensions disagree')
        return self

    def __str__(self) -> str:
        base = _format_parameters(self)
        if not self.benchmark_records:
            return base
        return '\n'.join([base, 'benchmark results:', *[
            f'  {i}. {_format_parameters(record.config)}: '
            f'{record.elapsed_ms:.3f} ms ({record.per_call_us:.3f} us/call)'
            + (f', task_capacity={record.task_capacity}' if record.task_capacity is not None else '')
            for i, record in enumerate(self.benchmark_records, 1)]])

    def parameter_snapshot(self) -> CSRConfig:
        """Return this snapshot without structure-specific preparation.

        Returns
        -------
        CSRConfig
            Tuning, diagnostics and device metadata without primary/mirror plans.
        """
        return replace(self, primary=None, mirror=None)

    def prepare_structure(self, indices: Any, indptr: Any, *, shape: Sequence[int],
                          device_info: CSRDeviceInfo | None = None) -> CSRConfig:
        """Prepare primary and mirror row statistics from concrete structure.

        Parameters
        ----------
        indices, indptr : array_like
            Concrete compressed indices and offsets. Tracers remain unprepared.
        shape : sequence of int
            Shape in compressed-row traversal order.
        device_info : CSRDeviceInfo or None, optional
            Replacement device metadata; omitted values retain the snapshot.

        Returns
        -------
        CSRConfig
            A new snapshot; the input configuration is unchanged.
        """
        base = replace(self.parameter_snapshot(), device_info=device_info or self.device_info)
        if isinstance(indices, jax.core.Tracer) or isinstance(indptr, jax.core.Tracer):
            return base
        if len(shape) != 2:
            raise ValueError('shape must contain two dimensions')
        rows, cols = (_integer(value, 'shape dimension') for value in shape)
        if max(rows, cols) > _INT32_MAX:
            raise ValueError('MM dimensions exceed int32 range')
        idx, ptr = np.asarray(indices), np.asarray(indptr)
        if idx.ndim != 1 or ptr.ndim != 1:
            raise ValueError('indices and indptr must be one-dimensional')
        if idx.dtype not in (np.dtype(np.int32), np.dtype(np.int64)) or ptr.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            raise ValueError('indices and indptr must have int32 or int64 dtype')
        if ptr.size != rows + 1 or ptr[0] != 0 or ptr[-1] != idx.size:
            raise ValueError('indptr must start at zero, match rows, and end at nnz')
        lengths = np.diff(ptr.astype(np.int64))
        if np.any(lengths < 0):
            raise ValueError('indptr must be nondecreasing')
        if idx.size and (idx.min() < 0 or idx.max() >= cols):
            raise ValueError('indices are outside the matrix shape')
        mirror_lengths = np.bincount(idx.astype(np.int64), minlength=cols).astype(np.int64)
        return replace(base, primary=_structure_plan(lengths, cols),
                       mirror=_structure_plan(mirror_lengths, rows))

    def swap_layouts(self) -> CSRConfig:
        """Return a snapshot whose compressed traversal layouts are exchanged.

        Returns
        -------
        CSRConfig
            A new object sharing the prepared mask arrays.
        """
        return replace(self, primary=self.mirror, mirror=self.primary)

    def kernel_args(self, layout: str = 'primary') -> tuple[CSRKernelSpec, Any]:
        """Separate selected static configuration from the dynamic mask operand.

        Parameters
        ----------
        layout : {'primary', 'mirror'}, optional
            Compressed traversal selected by the caller.

        Returns
        -------
        spec : CSRKernelSpec
            Hashable parameters, excluding benchmark diagnostics.
        mask : jax.Array
            Prepared mask, or an empty uint32 array when unprepared.
        """
        if layout not in ('primary', 'mirror'):
            raise ValueError("layout must be 'primary' or 'mirror'")
        plan = self.primary if layout == 'primary' else self.mirror
        spec = CSRKernelSpec(self.block_size, self.fixed_scatter_blocks, self.tpr_threshold,
                             self.task_nnz, None if plan is None else plan.specification(), self.device_info)
        return spec, jnp.empty((0,), dtype=jnp.uint32) if plan is None else plan.long_mask

    @classmethod
    def from_kernel_args(cls, spec: CSRKernelSpec, mask: Any) -> CSRConfig:
        """Reconstruct a selected snapshot without scanning or querying state.

        Parameters
        ----------
        spec : CSRKernelSpec
            Static primitive parameters.
        mask : array_like
            Corresponding mask operand, which may be a tracer.

        Returns
        -------
        CSRConfig
            A snapshot with the selected traversal in ``primary``.
        """
        structure = spec.structure
        plan = None if structure is None else MMStructurePlan(
            structure.rows, structure.cols, structure.nnz, structure.max_row_nnz,
            structure.group, structure.long_threshold, structure.long_tasks, mask,
            geometry_version=structure.geometry_version)
        return cls(spec.block_size, spec.fixed_scatter_blocks, spec.tpr_threshold, spec.task_nnz,
                   device_info=spec.device_info, primary=plan)

    def require_prepared(self, *, rows: int | None = None, cols: int | None = None,
                         nnz: int | None = None) -> MMStructurePlan:
        """Require preparation matching a hybrid MM traversal.

        Parameters
        ----------
        rows, cols, nnz : int or None, optional
            Expected traversal dimensions; omitted values are not checked.

        Returns
        -------
        MMStructurePlan
            Prepared primary traversal.

        Raises
        ------
        ValueError
            Structure or GPU metadata is missing, or dimensions disagree.
        """
        plan, device = self.primary, self.device_info
        if plan is None or device is None or device.platform != 'gpu' or device.l2_bytes <= 0 or device.sms <= 0:
            raise ValueError('hybrid MM requires a prepared CSRConfig with GPU device information; '
                             'prepare the CSR/config outside jax.jit before calling this kernel')
        for name, expected in (('rows', rows), ('cols', cols), ('nnz', nnz)):
            if expected is not None and getattr(plan, name) != expected:
                raise ValueError(f'prepared CSRConfig {name} does not match the selected structure')
        self.validate()
        return plan

    def mm_plan(self, batch: int, value_dtype: Any, weighted: bool = False) -> MMCallPlan | None:
        """Derive a launch plan from prepared metadata and static call inputs.

        Parameters
        ----------
        batch : int
            Event batch after any vmap folding.
        value_dtype : dtype
            Weight/output floating dtype.
        weighted : bool, optional
            Allocate the additional event-value queue.

        Returns
        -------
        MMCallPlan or None
            Cached scalar plan, or ``None`` when preparation is incomplete or
            the value dtype has no native hybrid MM implementation. A caller
            can still use that dtype with a different backend.
        """
        batch = _integer(batch, 'batch')
        if batch > _INT32_MAX:
            raise ValueError('MM batch exceeds int32 range')
        dtype = np.dtype(value_dtype)
        if dtype.name not in ('float16', 'bfloat16', 'float32', 'float64'):
            # Planning must not impose CUDA dtype restrictions on a JAX,
            # Numba, or gather invocation. The native adapter rejects an
            # unsupported dtype only when it actually selects hybrid MM.
            return None
        device, structure = self.device_info, self.primary
        if structure is None or device is None or device.platform != 'gpu' or device.l2_bytes <= 0 or device.sms <= 0:
            return None
        return _call_plan(structure.specification(), device, batch, dtype, bool(weighted))

    def tree_flatten(self) -> tuple[tuple[Any, ...], _ConfigAux]:
        """Split prepared masks from hashable scheduling metadata.

        Returns
        -------
        children : tuple
            Prepared primary and mirror masks, when present.
        auxiliary : object
            Static metadata with diagnostic records excluded from equality.
        """
        spec = CSRKernelSpec(self.block_size, self.fixed_scatter_blocks, self.tpr_threshold,
                             self.task_nnz, None if self.primary is None else self.primary.specification(),
                             self.device_info)
        masks = (() if self.primary is None else (self.primary.long_mask,))
        if self.mirror is not None:
            masks += (self.mirror.long_mask,)
        return masks, _ConfigAux(spec, None if self.mirror is None else self.mirror.specification(),
                                 self.benchmark_records)

    @classmethod
    def tree_unflatten(cls, auxiliary: _ConfigAux, children: Sequence[Any]) -> CSRConfig:
        """Restore a snapshot without preparing structures or reading state.

        Parameters
        ----------
        auxiliary : object
            Metadata returned by ``tree_flatten``.
        children : sequence
            Prepared mask leaves.

        Returns
        -------
        CSRConfig
            Reconstructed configuration.
        """
        has_primary = auxiliary.primary.structure is not None
        result = cls.from_kernel_args(auxiliary.primary, children[0] if has_primary else None)
        mirror = auxiliary.mirror
        if mirror is not None:
            result = replace(result, mirror=MMStructurePlan(
                mirror.rows, mirror.cols, mirror.nnz, mirror.max_row_nnz, mirror.group,
                mirror.long_threshold, mirror.long_tasks, children[int(has_primary)],
                geometry_version=mirror.geometry_version))
        return replace(result, benchmark_records=auxiliary.records)
