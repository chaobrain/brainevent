# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""CSR scheduling snapshots and structure-bound preparation.

Configuration helpers resolve persisted MV tuning and capture device properties
outside tracing. Prepared MM masks are pytree leaves; kernel specifications
contain only hashable metadata for JAX tracing and differentiation.
"""

from __future__ import annotations

import contextlib
import hashlib
import operator
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from brainevent._structure import is_concrete
from brainevent.config import _backend_preference

__all__ = [
    'CSRConfig', 'CSRDeviceInfo', 'MVSchedule', 'CSRKernelSpec',
    'MMStructureSpec', 'MMStructurePlan', 'MMCallPlan', 'get_device_info',
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


@lru_cache(maxsize=None)
def _query_cuda_device_info(device):
    """Read CUDA launch properties once per device through a small native probe."""
    from brainevent._op import load_cuda_file

    source = Path(__file__).parent / '_support' / 'device_info.cu'
    include_root = Path(__file__).parents[1] / 'include'
    dependencies = (source, *sorted(include_root.rglob('*.h')))
    digest = hashlib.sha256(b''.join(path.read_bytes() for path in dependencies)).hexdigest()[:12]
    module = f'csr_device_info_{digest}'
    # The device probe synchronizes its stream before returning host-backed data.
    load_cuda_file(source, name=module, allow_cuda_graph=False)
    dummy = jax.device_put(np.zeros((1,), np.uint8), device)
    info = jax.ffi.ffi_call(
        module + '.csr_device_info',
        jax.ShapeDtypeStruct((2,), jnp.int32),
    )(dummy)
    sms, l2_bytes = map(int, np.asarray(jax.device_get(info)))
    return CSRDeviceInfo(
        platform='gpu', device_kind=str(device.device_kind),
        l2_bytes=l2_bytes, sms=sms, device_id=device.id,
    )


def get_device_info(*arrays, device=None):
    """Read and cache the preparation device's CUDA launch properties.

    Parameters
    ----------
    *arrays : array_like
        Concrete arrays whose placement identifies the preparation device.
    device : jax.Device or None, optional
        Explicit preparation device. Otherwise use the arrays' device when
        unambiguous. CPU shards, host arrays, and tracers need no CUDA probe.

    Returns
    -------
    CSRDeviceInfo or None
        The selected CUDA device's SM count and L2 size, or ``None`` without
        an unambiguous concrete GPU placement.

    Raises
    ------
    ValueError
        The arrays include multiple GPU placements and no explicit device was supplied.
    """
    if device is None:
        devices = set()
        for array in arrays:
            if isinstance(array, jax.core.Tracer):
                continue
            if hasattr(array, 'devices'):
                devices.update(array.devices())
        if devices and all(d.platform != 'gpu' for d in devices):
            return None
        if len(devices) > 1:
            raise ValueError('CSR preparation requires a single device; pass device explicitly.')
        if not devices:
            return None
        device = next(iter(devices))
    if device.platform != 'gpu':
        return None
    platform_version = str(getattr(getattr(device, 'client', None), 'platform_version', '')).lower()
    if 'cuda' not in platform_version:
        return None
    return _query_cuda_device_info(device)


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
class MMStructurePlan:
    """MM-specific scheduling geometry and its prepared long-row mask.

    Parameters
    ----------
    group, long_threshold, long_tasks : int
        Lane group, long-row threshold, and all-active task count per batch.
    long_mask : array_like
        ``uint32[ceil(rows / 32)]`` bitmask carried as a pytree array leaf.
    """

    group: int
    long_threshold: int
    long_tasks: int
    long_mask: Any = field(compare=False, hash=False, repr=False)
    geometry_version: int = field(default=_MM_GEOMETRY_VERSION, kw_only=True)

    def specification(self, rows: int, cols: int, nnz: int, max_row_nnz: int) -> MMStructureSpec:
        """Return the array-free traversal metadata.

        Parameters
        ----------
        rows, cols, nnz, max_row_nnz : int
            Shared dimensions and row statistic for the selected traversal.

        Returns
        -------
        MMStructureSpec
            Metadata suitable for a primitive parameter.
        """
        return MMStructureSpec(rows, cols, nnz, max_row_nnz,
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


@dataclass(frozen=True)
class MVSchedule:
    """Frozen hybrid MV/SDDMV scheduler constants.

    Parameters
    ----------
    block_size, fixed_scatter_blocks, tpr_threshold, task_nnz : int, optional
        CUDA block width, scatter grid cap, heavy-row threshold, and task size.
    """

    block_size: int = 512
    fixed_scatter_blocks: int = 1024
    tpr_threshold: int = 128
    task_nnz: int = 512

    def __post_init__(self) -> None:
        _validate_scheduler(self)

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


def _validate_scheduler(config: MVSchedule) -> None:
    block = _integer(config.block_size, 'block_size', 1)
    if block % 32 or block > 1024:
        raise ValueError('block_size must be a multiple of 32 and must not exceed 1024')
    _integer(config.fixed_scatter_blocks, 'fixed_scatter_blocks', 1)
    _integer(config.tpr_threshold, 'tpr_threshold')
    _integer(config.task_nnz, 'task_nnz', 1)


def _format_parameters(config: MVSchedule | CSRConfig) -> str:
    return (f'CSRConfig(block_size={config.block_size}, fixed_scatter_blocks={config.fixed_scatter_blocks}, '
            f'tpr_threshold={config.tpr_threshold}, task_nnz={config.task_nnz})')


@dataclass(frozen=True)
class CSRKernelSpec(MVSchedule):
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

    structure: MMStructureSpec | None = None
    device_info: CSRDeviceInfo | None = None

    @property
    def mv(self) -> MVSchedule:
        """Return the MV scheduler portion of this static specification.

        Returns
        -------
        MVSchedule
            The four MV/SDDMV scheduler constants.
        """
        return MVSchedule(self.block_size, self.fixed_scatter_blocks,
                          self.tpr_threshold, self.task_nnz)


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


def _structure_plan(lengths: np.ndarray[Any, Any]) -> MMStructurePlan:
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
    return MMStructurePlan(group, threshold, long_tasks, jnp.asarray(mask))


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
    shape: tuple[int, int] | None
    nnz: int | None
    primary_max_row_nnz: int | None
    mirror_max_row_nnz: int | None
    records: tuple[_BenchmarkRecord, ...] = field(compare=False, hash=False)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class CSRConfig:
    """A CSR scheduling snapshot with MV tuning and optional MM preparation.

    Parameters
    ----------
    block_size, fixed_scatter_blocks, tpr_threshold, task_nnz : int, optional
        Legacy flat spelling of the four persisted MV/SDDMV constants.
    mv : MVSchedule or None, optional
        Explicit grouped MV scheduler; cannot be combined with flat constants.
    benchmark_records : tuple of _BenchmarkRecord, optional
        Diagnostic tuning results, excluded from kernel identity.
    device_info : CSRDeviceInfo or None, optional
        Eagerly captured device metadata; this class never queries a device.
    shape, nnz, primary_max_row_nnz, mirror_max_row_nnz : optional
        Shared structure identity and row statistics in compressed traversal
        order. These values are cleared when the structure is replaced.
    primary, mirror : MMStructurePlan or None, optional
        MM-specific plans for the original and mirrored compressed layouts.

    Notes
    -----
    This object is a pytree, not a hashable static argument. Use
    :meth:`kernel_args` to separate a static specification from its array leaf.
    """

    mv: MVSchedule = field(default_factory=MVSchedule)
    benchmark_records: tuple[_BenchmarkRecord, ...] = field(default=(), compare=False, repr=False)
    device_info: CSRDeviceInfo | None = None
    shape: tuple[int, int] | None = None
    nnz: int | None = None
    primary_max_row_nnz: int | None = None
    mirror_max_row_nnz: int | None = None
    primary: MMStructurePlan | None = field(default=None, compare=False, repr=False)
    mirror: MMStructurePlan | None = field(default=None, compare=False, repr=False)
    __hash__: ClassVar[Any] = None

    def __init__(
        self,
        block_size: int | None = None,
        fixed_scatter_blocks: int | None = None,
        tpr_threshold: int | None = None,
        task_nnz: int | None = None,
        benchmark_records: tuple[_BenchmarkRecord, ...] = (),
        device_info: CSRDeviceInfo | None = None,
        primary: MMStructurePlan | None = None,
        mirror: MMStructurePlan | None = None,
        *,
        mv: MVSchedule | None = None,
        shape: Sequence[int] | None = None,
        nnz: int | None = None,
        primary_max_row_nnz: int | None = None,
        mirror_max_row_nnz: int | None = None,
    ) -> None:
        if mv is None:
            mv = MVSchedule(
                512 if block_size is None else block_size,
                1024 if fixed_scatter_blocks is None else fixed_scatter_blocks,
                128 if tpr_threshold is None else tpr_threshold,
                512 if task_nnz is None else task_nnz,
            )
        elif any(value is not None for value in
                 (block_size, fixed_scatter_blocks, tpr_threshold, task_nnz)):
            raise ValueError('mv cannot be combined with flat MV scheduler parameters')
        if not isinstance(mv, MVSchedule):
            raise TypeError('mv must be an MVSchedule')
        object.__setattr__(self, 'mv', mv)
        object.__setattr__(self, 'benchmark_records', tuple(benchmark_records))
        object.__setattr__(self, 'device_info', device_info)
        object.__setattr__(self, 'shape', None if shape is None else tuple(shape))
        object.__setattr__(self, 'nnz', nnz)
        object.__setattr__(self, 'primary_max_row_nnz', primary_max_row_nnz)
        object.__setattr__(self, 'mirror_max_row_nnz', mirror_max_row_nnz)
        object.__setattr__(self, 'primary', primary)
        object.__setattr__(self, 'mirror', mirror)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_scheduler(self.mv)

    @property
    def block_size(self) -> int:
        """Return the MV CUDA block width.

        Returns
        -------
        int
            Threads per hybrid MV block.
        """
        return self.mv.block_size

    @property
    def fixed_scatter_blocks(self) -> int:
        """Return the MV scatter grid cap.

        Returns
        -------
        int
            Maximum fixed scatter blocks.
        """
        return self.mv.fixed_scatter_blocks

    @property
    def tpr_threshold(self) -> int:
        """Return the MV heavy-row threshold.

        Returns
        -------
        int
            Row length at which hybrid MV schedules extra tasks.
        """
        return self.mv.tpr_threshold

    @property
    def task_nnz(self) -> int:
        """Return the MV edges per task.

        Returns
        -------
        int
            Number of stored edges assigned to a heavy-row task.
        """
        return self.mv.task_nnz

    def compile_flags(self) -> list[str]:
        """Return the grouped MV scheduler's CUDA compiler definitions.

        Returns
        -------
        list of str
            Four ``-DBE_HYBRID_*`` flags.
        """
        return self.mv.compile_flags()

    def module_suffix(self) -> str:
        """Return the grouped MV scheduler's module-name suffix.

        Returns
        -------
        str
            Suffix encoding the four MV scheduler constants.
        """
        return self.mv.module_suffix()

    def mv_capacity(self, nse: int, n_rows: int, max_row_nnz: int | None = None) -> int:
        """Return the grouped MV scheduler's task-queue capacity bound.

        Parameters
        ----------
        nse, n_rows : int
            Stored edge and compressed row counts.
        max_row_nnz : int or None, optional
            Optional longest-row bound.

        Returns
        -------
        int
            Capacity covering compatible structures.
        """
        return self.mv.mv_capacity(nse, n_rows, max_row_nnz)

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
        _validate_scheduler(self.mv)
        if self.shape is not None:
            if len(self.shape) != 2:
                raise ValueError('prepared CSRConfig shape must contain two dimensions')
            for dimension in self.shape:
                if _integer(dimension, 'shape dimension') > _INT32_MAX:
                    raise ValueError('MM dimensions exceed int32 range')
        if self.nnz is not None:
            _integer(self.nnz, 'nnz')
        for name in ('primary_max_row_nnz', 'mirror_max_row_nnz'):
            maximum = getattr(self, name)
            if maximum is not None:
                _integer(maximum, name)
                if self.nnz is None or maximum > self.nnz:
                    raise ValueError(f'{name} requires nnz and must not exceed it')
        for layout, plan in (('primary', self.primary), ('mirror', self.mirror)):
            if plan is None:
                continue
            structure = self._structure_spec(layout)
            if structure is None:
                raise ValueError('prepared CSRConfig requires structure metadata')
            if getattr(plan.long_mask, 'shape', None) != ((structure.rows + 31) // 32,):
                raise ValueError('prepared CSRConfig long-row mask has the wrong shape')
            if getattr(plan.long_mask, 'dtype', None) != np.dtype(np.uint32):
                raise ValueError('prepared CSRConfig long-row mask must have uint32 dtype')
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
            Tuning, diagnostics and device metadata without structure metadata
            or primary/mirror plans.
        """
        return replace(self, shape=None, nnz=None,
                       primary_max_row_nnz=None, mirror_max_row_nnz=None,
                       primary=None, mirror=None)

    def prepare_structure(self, indices: Any, indptr: Any, *, shape: Sequence[int],
                          device_info: CSRDeviceInfo | None = None) -> CSRConfig:
        """Prepare primary and mirror row statistics from concrete structure.

        Masks for a CPU-sharded structure are replicated across its mesh.

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
        prepared = replace(base, shape=(rows, cols), nnz=int(idx.size),
                           primary_max_row_nnz=int(lengths.max(initial=0)),
                           mirror_max_row_nnz=int(mirror_lengths.max(initial=0)),
                           primary=_structure_plan(lengths),
                           mirror=_structure_plan(mirror_lengths))
        return _place_prepared_masks(prepared, _replicated_cpu_sharding(indices, indptr))

    def swap_layouts(self) -> CSRConfig:
        """Return a snapshot whose compressed traversal layouts are exchanged.

        Returns
        -------
        CSRConfig
            A new object sharing the prepared mask arrays.
        """
        return replace(self, shape=None if self.shape is None else self.shape[::-1],
                       primary_max_row_nnz=self.mirror_max_row_nnz,
                       mirror_max_row_nnz=self.primary_max_row_nnz,
                       primary=self.mirror, mirror=self.primary)

    def _structure_spec(self, layout: str) -> MMStructureSpec | None:
        plan = self.primary if layout == 'primary' else self.mirror
        if plan is None:
            return None
        maximum = self.primary_max_row_nnz if layout == 'primary' else self.mirror_max_row_nnz
        if self.shape is None or self.nnz is None or maximum is None:
            raise ValueError('prepared CSRConfig requires shape, nnz, and row statistics')
        rows, cols = self.shape if layout == 'primary' else self.shape[::-1]
        return plan.specification(rows, cols, self.nnz, maximum)

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
                             self.task_nnz, self._structure_spec(layout), self.device_info)
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
            structure.group, structure.long_threshold, structure.long_tasks, mask,
            geometry_version=structure.geometry_version)
        return cls(mv=spec.mv, device_info=spec.device_info,
                   shape=None if structure is None else (structure.rows, structure.cols),
                   nnz=None if structure is None else structure.nnz,
                   primary_max_row_nnz=None if structure is None else structure.max_row_nnz,
                   primary=plan)

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
        structure = self._structure_spec('primary')
        assert structure is not None
        for name, expected in (('rows', rows), ('cols', cols), ('nnz', nnz)):
            if expected is not None and getattr(structure, name) != expected:
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
        device = self.device_info
        if (self.primary is None or device is None or device.platform != 'gpu'
                or device.l2_bytes <= 0 or device.sms <= 0):
            return None
        structure = self._structure_spec('primary')
        assert structure is not None
        return _call_plan(structure, device, batch, dtype, bool(weighted))

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
                             self.task_nnz, self._structure_spec('primary'), self.device_info)
        masks = (() if self.primary is None else (self.primary.long_mask,))
        if self.mirror is not None:
            masks += (self.mirror.long_mask,)
        return masks, _ConfigAux(spec, self._structure_spec('mirror'),
                                 self.shape, self.nnz, self.primary_max_row_nnz,
                                 self.mirror_max_row_nnz, self.benchmark_records)

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
                mirror.group, mirror.long_threshold, mirror.long_tasks, children[int(has_primary)],
                geometry_version=mirror.geometry_version))
        return replace(result, shape=auxiliary.shape, nnz=auxiliary.nnz,
                       primary_max_row_nnz=auxiliary.primary_max_row_nnz,
                       mirror_max_row_nnz=auxiliary.mirror_max_row_nnz,
                       benchmark_records=auxiliary.records)


def _structure_device(indices: Any, device: Any = None) -> Any:
    """Identify a concrete structure's placement without querying GPU properties."""
    if device is not None:
        return device
    if hasattr(indices, 'devices') and not isinstance(indices, jax.core.Tracer):
        placed = indices.devices()
        if len(placed) == 1:
            return next(iter(placed))
    return None


def _replicated_cpu_sharding(*arrays: Any) -> Any:
    """Return a replicated placement for a concrete multi-CPU structure."""
    for array in arrays:
        if isinstance(array, jax.core.Tracer):
            continue
        sharding = getattr(array, 'sharding', None)
        if isinstance(sharding, jax.sharding.NamedSharding):
            devices = sharding.device_set
            if len(devices) > 1 and all(getattr(device, 'platform', None) == 'cpu' for device in devices):
                return jax.sharding.NamedSharding(sharding.mesh, jax.sharding.PartitionSpec())
    return None


def _place_prepared_masks(config: CSRConfig, sharding: Any) -> CSRConfig:
    """Place whole-structure masks on the CPU mesh or an explicit device."""
    if sharding is None:
        return config
    primary = (None if config.primary is None else
               replace(config.primary, long_mask=jax.device_put(config.primary.long_mask, sharding)))
    mirror = (None if config.mirror is None else
              replace(config.mirror, long_mask=jax.device_put(config.mirror.long_mask, sharding)))
    return replace(config, primary=primary, mirror=mirror)


def _host_structure_array(original: Any, converted: Any) -> np.ndarray[Any, Any] | None:
    """Reuse host input, or read one concrete device array once."""
    if not is_concrete(converted):
        return None
    if isinstance(original, (np.ndarray, list, tuple)):
        return np.asarray(original)
    return np.asarray(jax.device_get(converted))


def _prepare_csr_config(indices: Any, indptr: Any, *, shape: Sequence[int],
                        backend: str | None, config: CSRConfig | None = None,
                        device: Any = None,
                        host_structure: tuple[Any, Any] | None = None) -> CSRConfig:
    """Resolve a parameter snapshot and prepare concrete structure outside tracing."""
    from brainevent._csr.binary.initialize import load_csr_config

    cpu_sharding = _replicated_cpu_sharding(indices, indptr) if device is None else None
    target = _structure_device(indices, device)
    if target is None and cpu_sharding is not None:
        target = min(cpu_sharding.device_set, key=lambda item: item.id)
    device_info = None
    preferred = _backend_preference('gpu', backend)
    # A global choice other than jax_raw may be unsupported by an indexed
    # primitive, which then falls back to its CUDA default. Keep its plan ready.
    needs_cuda_info = preferred == 'cuda_raw' or (backend is None and preferred != 'jax_raw')
    if (needs_cuda_info or device is not None) and is_concrete(indices, indptr):
        with jax.ensure_compile_time_eval():
            device_info = get_device_info(indices, indptr, device=device)
    if config is None:
        device_kind = (device_info.device_kind if device_info is not None
                       else getattr(target, 'device_kind', None))
        config = load_csr_config(device_kind=device_kind)
    if not isinstance(config, CSRConfig):
        raise TypeError('config must be a CSRConfig or None.')
    snapshot = replace(config.parameter_snapshot(), device_info=device_info)
    if not is_concrete(indices, indptr):
        return snapshot
    with jax.ensure_compile_time_eval(), (jax.default_device(target) if target is not None else contextlib.nullcontext()):
        source_indices, source_indptr = (indices, indptr) if host_structure is None else host_structure
        prepared = snapshot.prepare_structure(source_indices, source_indptr, shape=shape, device_info=device_info)
        return _place_prepared_masks(prepared, device if device is not None else cpu_sharding)


def _config_for_new_structure(config: CSRConfig, indices: Any, indptr: Any, *,
                              shape: Sequence[int], device: Any = None) -> CSRConfig:
    """Preserve MV tuning while preparing or invalidating a replacement structure."""
    snapshot = config.parameter_snapshot()
    if not is_concrete(indices, indptr):
        return snapshot
    target = _structure_device(indices, device)
    cpu_sharding = _replicated_cpu_sharding(indices, indptr) if device is None else None
    if cpu_sharding is None and device is None and (target is None or target.platform == 'cpu'):
        masks = (None if config.primary is None else config.primary.long_mask,
                 None if config.mirror is None else config.mirror.long_mask)
        cpu_sharding = _replicated_cpu_sharding(*masks)
    if target is None and cpu_sharding is not None:
        target = min(cpu_sharding.device_set, key=lambda item: item.id)
    with jax.ensure_compile_time_eval(), (jax.default_device(target) if target is not None else contextlib.nullcontext()):
        prepared = snapshot.prepare_structure(
            indices, indptr, shape=shape, device_info=config.device_info,
        )
        return _place_prepared_masks(prepared, device if device is not None else cpu_sharding)


def _config_with_mirror(config: CSRConfig, mirror: Sequence[Any], *,
                        shape: Sequence[int], device: Any = None) -> CSRConfig:
    """Bind a replacement mirror plan while preserving the primary plan."""
    mirror_indptr, mirror_indices, _ = mirror
    prepared = _config_for_new_structure(
        config, mirror_indices, mirror_indptr, shape=shape, device=device,
    )
    return replace(config, mirror=prepared.primary,
                   mirror_max_row_nnz=prepared.primary_max_row_nnz)
