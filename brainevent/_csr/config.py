# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""CSR scheduler tuning and the shape-only hybrid MM plan.

A :class:`CSRConfig` holds persisted MV scheduler tuning only. It carries no
structure statistics, masks or device properties, so matrices with the same
shape and nnz share compiled executables however they were built. The hybrid
MM launch is planned during lowering from static shapes and the target
device's L2 size (:func:`_mm_plan`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar, Sequence

import jax
import numpy as np

from brainevent._misc import _static_integer as _integer
from brainevent._structure import is_concrete

__all__ = ['CSRConfig', 'MVSchedule']

_INT32_MAX = int(np.iinfo(np.int32).max)
_INT64_MAX = int(np.iinfo(np.int64).max)
# Native hybrid MM geometry: a lane group walks 16 edges of a short row per
# iteration, and long rows split into 512-edge tasks.
_MM_ITERATIONS = 16
_MM_TASK_EDGES = 32 * _MM_ITERATIONS
_MM_L2_DIVISOR = 4
_MM_MAX_CHUNK = 65535
_logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class _ConfigAux:
    """Hashable pytree metadata; benchmark records are excluded from equality."""

    mv: MVSchedule
    records: tuple[_BenchmarkRecord, ...] = field(compare=False, hash=False)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class CSRConfig:
    """Persisted MV scheduler tuning for CSR event kernels.

    ``CSRConfig()`` holds the default MV scheduler parameters. A config does
    not depend on a matrix's structure or device: the hybrid MM launch is
    planned from static shapes and the target device when it is lowered, so
    one config serves any matrix, including one built inside ``jax.jit``.

    Parameters
    ----------
    block_size, fixed_scatter_blocks, tpr_threshold, task_nnz : int, optional
        Flat spelling of the four persisted MV/SDDMV constants.
    benchmark_records : tuple of _BenchmarkRecord, optional
        Diagnostic tuning results, excluded from kernel identity.
    mv : MVSchedule or None, optional
        Explicit grouped MV scheduler; cannot be combined with flat constants.

    Notes
    -----
    This object is a pytree without array leaves. Its scheduler is part of
    the tree structure, so two matrices compile the same executable when
    their configs hold the same scheduler.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent import CSRConfig
        >>> CSRConfig(block_size=256).mv.block_size
        256
    """

    mv: MVSchedule = field(default_factory=MVSchedule)
    benchmark_records: tuple[_BenchmarkRecord, ...] = field(default=(), compare=False, repr=False)
    __hash__: ClassVar[Any] = None

    def __init__(
        self,
        block_size: int | None = None,
        fixed_scatter_blocks: int | None = None,
        tpr_threshold: int | None = None,
        task_nnz: int | None = None,
        benchmark_records: tuple[_BenchmarkRecord, ...] = (),
        *,
        mv: MVSchedule | None = None,
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

    def validate(self) -> CSRConfig:
        """Validate the tuning parameters.

        Returns
        -------
        CSRConfig
            This configuration when its scheduler constants are valid.

        Raises
        ------
        ValueError
            A scheduler constant is out of range.
        """
        _validate_scheduler(self.mv)
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

    def tree_flatten(self) -> tuple[tuple[Any, ...], _ConfigAux]:
        """Flatten to no leaves and the hashable scheduler.

        Returns
        -------
        children : tuple
            Always empty.
        auxiliary : object
            Scheduler metadata, with diagnostic records excluded from equality.
        """
        return (), _ConfigAux(self.mv, self.benchmark_records)

    @classmethod
    def tree_unflatten(cls, auxiliary: _ConfigAux, children: Sequence[Any]) -> CSRConfig:
        """Restore a config from its flattened scheduler.

        Parameters
        ----------
        auxiliary : object
            Metadata returned by ``tree_flatten``.
        children : sequence
            Unused; a config has no leaves.

        Returns
        -------
        CSRConfig
            Reconstructed configuration.
        """
        return cls(mv=auxiliary.mv, benchmark_records=auxiliary.records)


def _config_from_spec(spec: MVSchedule | None) -> CSRConfig:
    """The config of a primitive's ``config_spec`` parameter."""
    return CSRConfig() if spec is None else CSRConfig(mv=spec)


# ---------------------------------------------------------------------------
# Shape-only hybrid MM plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MMPlan:
    """Native hybrid MM launch attributes and scratch sizes for one call.

    Parameters
    ----------
    group, long_threshold, long_tasks, chunk, task_capacity : int
        Launch attributes of the native kernel.
    workspace_words, short_rows_size : int
        Scratch lengths in uint32 and int32 elements. The short-row lists hold
        one batch chunk (``chunk * rows``).
    """

    group: int
    long_threshold: int
    long_tasks: int
    chunk: int
    task_capacity: int
    workspace_words: int
    short_rows_size: int


def _mm_group(rows: int, nnz: int) -> int:
    """Lane group from the mean row length: a quarter of it, rounded up to 4, 8, 16 or 32."""
    mean = min(nnz // rows, _MM_TASK_EDGES) if rows else 0
    quarter = (mean + 3) // 4
    return 4 if quarter <= 4 else 8 if quarter <= 8 else 16 if quarter <= 16 else 32


@lru_cache(maxsize=256)
def _mm_plan(rows: int, cols: int, nnz: int, batch: int, dtype: Any,
             weighted: bool, l2_bytes: int) -> _MMPlan:
    """Plan a native hybrid MM call from static shapes and the target L2 size.

    Parameters
    ----------
    rows, cols, nnz : int
        Compressed rows, output columns and stored edges of the traversal.
    batch : int
        Event columns after any vmap folding.
    dtype : dtype
        Weight dtype; outputs accumulate in float32 (float64 for float64).
    weighted : bool
        Reserve the event-value queue of weighted events.
    l2_bytes : int
        L2 size of the target device.

    Returns
    -------
    _MMPlan
        The launch plan. Its task queue holds every long-row task of any
        structure with these shapes, ``ceil(nnz / 512)`` plus one per
        possible long row, so it never overflows.
    """
    # Compute with Python ints: NumPy integer shapes would overflow silently.
    rows, cols, nnz, batch = (_integer(value, name) for name, value in (
        ('rows', rows), ('cols', cols), ('nnz', nnz), ('batch', batch)))
    l2_bytes = _integer(l2_bytes, 'l2_bytes')
    if max(rows, cols, batch) > _INT32_MAX:
        raise ValueError('MM dimensions exceed int32 range')
    group = _mm_group(rows, nnz)
    threshold = _MM_ITERATIONS * group
    if not (rows and cols and nnz and batch):
        return _MMPlan(group, threshold, 0, 1, 0, 0, 0)
    long_tasks = -(-nnz // _MM_TASK_EDGES) + min(rows, nnz // (threshold + 1))
    value_bytes = np.dtype(dtype).itemsize
    # One chunk of accumulator columns should stay resident in a quarter of L2.
    fit = (l2_bytes // _MM_L2_DIVISOR) // (cols * max(4, value_bytes))
    chunk = max(1, min(fit, batch, _MM_MAX_CHUNK))
    capacity = chunk * long_tasks
    groups = -(-batch // chunk)
    queue_bytes = ((capacity * value_bytes + 3) // 4 * 4) if weighted else 0
    workspace_bytes = ((8 * batch + 7) // 8 * 8) + 8 * groups + 12 * capacity + queue_bytes
    short_rows = chunk * rows
    if max(capacity, workspace_bytes, short_rows) > _INT64_MAX:
        raise ValueError('MM plan or scratch size exceeds int64 range')
    return _MMPlan(group, threshold, long_tasks, chunk, capacity, workspace_bytes // 4, short_rows)


# ---------------------------------------------------------------------------
# Matrix construction helpers
# ---------------------------------------------------------------------------

def _structure_device(indices: Any) -> Any:
    """The single device holding a concrete array, or ``None``."""
    if hasattr(indices, 'devices') and not isinstance(indices, jax.core.Tracer):
        placed = indices.devices()
        if len(placed) == 1:
            return next(iter(placed))
    return None


def _device_kind(indices: Any) -> str | None:
    """Device kind of a concrete array, when all its devices share one kind."""
    if isinstance(indices, jax.core.Tracer) or not hasattr(indices, 'devices'):
        return None
    kinds = {str(getattr(device, 'device_kind', '')) for device in indices.devices()}
    return kinds.pop() if len(kinds) == 1 else None


def _host_structure_array(original: Any, converted: Any) -> np.ndarray[Any, Any] | None:
    """Reuse host input, or read one concrete device array once."""
    if not is_concrete(converted):
        return None
    if isinstance(original, (np.ndarray, list, tuple)):
        return np.asarray(original)
    return np.asarray(jax.device_get(converted))


def _select_config(indices: Any, *, shape: Sequence[int], config: CSRConfig | None = None) -> CSRConfig:
    """An explicit config, or the persisted tuning for this matrix's shape, nnz and device kind.

    Selection reads only static metadata, so it also works with traced arrays.
    """
    from brainevent._csr.binary.initialize import load_csr_config

    if config is not None:
        if not isinstance(config, CSRConfig):
            raise TypeError('config must be a CSRConfig or None.')
        _logger.debug('CSR config source=explicit shape=%s nnz=%s parameters=%s',
                      tuple(shape), indices.size, config.mv)
        return config
    metadata = {'shape': tuple(shape), 'nnz': indices.size}
    return load_csr_config(device_kind=_device_kind(indices), matrix_metadata=metadata)
