# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Unified CSR scheduling policies and shape-only hybrid MM planning.

A :class:`CSRConfig` holds immutable MV constants and MM chunk policies.
Concrete structures, masks and device properties remain outside the config;
resolved MM plans are computed during lowering from static call metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar, Mapping, Sequence

import jax
import numpy as np

from brainevent._misc import _static_integer as _integer
from brainevent._structure import is_concrete

__all__ = ['CSRConfig']

_INT32_MAX = int(np.iinfo(np.int32).max)
_INT64_MAX = int(np.iinfo(np.int64).max)
_logger = logging.getLogger(__name__)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class CSRConfig:
    """Immutable scheduling parameters for CSR MV and hybrid MM.

    MV uses the four compiler constants. Hybrid MM derives its launch plan
    from the call's shapes, dtype, event mode and target L2, using the five
    MM chunk policies. A config contains no structure arrays or resolved
    device state and can be shared by different matrices.

    Parameters
    ----------
    block_size, fixed_scatter_blocks, tpr_threshold, task_nnz : int or None, optional
        MV CUDA block width, scatter grid cap, heavy-row threshold and task
        size. Defaults are 512, 1024, 128 and 512, respectively.
    benchmark_records : sequence of mappings, optional
        Diagnostic timing records. They are copied and excluded from
        equality, hashing and JAX compilation identity.
    mm_l2_divisor : int, optional
        Divisor of L2 used for the compact output chunk. Default is 4.
    mm_sparse_l2_divisor : int, optional
        L2 divisor for four-lane schedules on large caches. Default is 2.
    mm_sparse_l2_min_bytes : int, optional
        L2 size at which the sparse allowance applies. Default is 32 MiB.
    mm_short_rows_floor : int, optional
        Minimum short-row list allowance in int32 entries. Default is 2**20.
    mm_max_chunk : int, optional
        Maximum batch chunk, between 1 and 65535. Default is 65535.

    Notes
    -----
    All scheduling fields participate in equality and hashing. The config
    is its own static pytree auxiliary data and has no array leaves.
    Changing any policy can retrace a JIT using the matrix, including a
    policy for an operation that the function does not execute. Benchmark
    diagnostics never change this identity.

    MM iterations and task edges are fixed class constants matching CUDA,
    rather than user overrides. Masks belong to the matrix's structure
    cache and use :meth:`mm_threshold`.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent import CSRConfig
        >>> config = CSRConfig(block_size=256, mm_max_chunk=32)
        >>> config.block_size, config.mm_max_chunk
        (256, 32)
    """

    block_size: int = 512
    fixed_scatter_blocks: int = 1024
    tpr_threshold: int = 128
    task_nnz: int = 512
    mm_l2_divisor: int = field(default=4, kw_only=True)
    mm_sparse_l2_divisor: int = field(default=2, kw_only=True)
    mm_sparse_l2_min_bytes: int = field(default=32 << 20, kw_only=True)
    mm_short_rows_floor: int = field(default=1 << 20, kw_only=True)
    mm_max_chunk: int = field(default=65535, kw_only=True)
    benchmark_records: tuple[dict[str, Any], ...] = field(
        default=(), compare=False, hash=False, repr=False,
    )

    MV_FIELDS: ClassVar[tuple[str, ...]] = (
        'block_size', 'fixed_scatter_blocks', 'tpr_threshold', 'task_nnz',
    )
    MM_FIELDS: ClassVar[tuple[str, ...]] = (
        'mm_l2_divisor', 'mm_sparse_l2_divisor', 'mm_sparse_l2_min_bytes',
        'mm_short_rows_floor', 'mm_max_chunk',
    )
    PARAMETER_FIELDS: ClassVar[tuple[str, ...]] = MV_FIELDS + MM_FIELDS
    MM_ITERATIONS: ClassVar[int] = 16
    MM_TASK_EDGES: ClassVar[int] = 32 * MM_ITERATIONS
    _MM_PLAN_FIELDS: ClassVar[tuple[str, ...]] = (
        'group', 'long_threshold', 'long_tasks', 'chunk', 'task_capacity',
        'workspace_words', 'short_rows_size',
    )

    def __init__(
        self,
        block_size: int | None = None,
        fixed_scatter_blocks: int | None = None,
        tpr_threshold: int | None = None,
        task_nnz: int | None = None,
        benchmark_records: Sequence[Mapping[str, Any]] = (),
        *,
        mm_l2_divisor: int = 4,
        mm_sparse_l2_divisor: int = 2,
        mm_sparse_l2_min_bytes: int = 32 << 20,
        mm_short_rows_floor: int = 1 << 20,
        mm_max_chunk: int = 65535,
    ) -> None:
        values = (
            512 if block_size is None else block_size,
            1024 if fixed_scatter_blocks is None else fixed_scatter_blocks,
            128 if tpr_threshold is None else tpr_threshold,
            512 if task_nnz is None else task_nnz,
            mm_l2_divisor, mm_sparse_l2_divisor, mm_sparse_l2_min_bytes,
            mm_short_rows_floor, mm_max_chunk,
        )
        for name, value in zip(self.PARAMETER_FIELDS, values):
            object.__setattr__(self, name, _integer(value, name))
        records = []
        for record in benchmark_records:
            copied = dict(record)
            if isinstance(copied.get('config'), Mapping):
                copied['config'] = dict(copied['config'])
            records.append(copied)
        object.__setattr__(self, 'benchmark_records', tuple(records))
        self.validate()

    def validate(self) -> CSRConfig:
        """Validate all scheduling parameters.

        Returns
        -------
        CSRConfig
            This configuration when all parameters are valid.

        Raises
        ------
        ValueError
            A scheduling field is invalid or exceeds the native limit.
        """
        block = _integer(self.block_size, 'block_size', 1)
        if block % 32 or block > 1024:
            raise ValueError('block_size must be a multiple of 32 and must not exceed 1024')
        _integer(self.fixed_scatter_blocks, 'fixed_scatter_blocks', 1)
        _integer(self.tpr_threshold, 'tpr_threshold')
        _integer(self.task_nnz, 'task_nnz', 1)
        _integer(self.mm_l2_divisor, 'mm_l2_divisor', 1)
        _integer(self.mm_sparse_l2_divisor, 'mm_sparse_l2_divisor', 1)
        _integer(self.mm_sparse_l2_min_bytes, 'mm_sparse_l2_min_bytes')
        _integer(self.mm_short_rows_floor, 'mm_short_rows_floor', 1)
        chunk = _integer(self.mm_max_chunk, 'mm_max_chunk', 1)
        if chunk > 65535:
            raise ValueError('mm_max_chunk must not exceed 65535')
        return self

    def compile_flags(self) -> list[str]:
        """Return the MV/SDDMV scheduler compiler definitions.

        Returns
        -------
        list of str
            Four ``-DBE_HYBRID_*`` compiler flags.
        """
        return [
            f'-DBE_HYBRID_BLOCK_SIZE={self.block_size}',
            f'-DBE_HYBRID_FIXED_SCATTER_BLOCKS={self.fixed_scatter_blocks}',
            f'-DBE_HYBRID_TPR_THRESHOLD={self.tpr_threshold}',
            f'-DBE_HYBRID_TASK_NNZ={self.task_nnz}',
        ]

    def module_suffix(self) -> str:
        """Return the MV/SDDMV configuration-dependent module suffix.

        Returns
        -------
        str
            Suffix encoding the four MV compiler constants.
        """
        return (f'_b{self.block_size}_s{self.fixed_scatter_blocks}'
                f'_t{self.tpr_threshold}_n{self.task_nnz}')

    def mv_capacity(self, nse: int, n_rows: int, max_row_nnz: int | None = None) -> int:
        """Bound the MV task queue from static dimensions.

        Parameters
        ----------
        nse, n_rows : int
            Stored edge count and compressed row count.
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

    @staticmethod
    def mm_group(rows: int, nnz: int) -> int:
        """Choose the MM lane group from the mean row length.

        Parameters
        ----------
        rows, nnz : int
            Compressed rows and stored edges.

        Returns
        -------
        int
            One of 4, 8, 16 or 32 lanes.
        """
        rows, nnz = _integer(rows, 'rows'), _integer(nnz, 'nnz')
        mean = min(nnz // rows, CSRConfig.MM_TASK_EDGES) if rows else 0
        quarter = (mean + 3) // 4
        return 4 if quarter <= 4 else 8 if quarter <= 8 else 16 if quarter <= 16 else 32

    @classmethod
    def mm_threshold(cls, rows: int, nnz: int) -> int:
        """Return the shared MM long-row threshold used by masks and plans.

        Parameters
        ----------
        rows, nnz : int
            Compressed rows and stored edges.

        Returns
        -------
        int
            Row lengths above this threshold use long-row tasks.
        """
        return cls.MM_ITERATIONS * cls.mm_group(rows, nnz)

    def mm_plan(self, rows: int, cols: int, nnz: int, batch: int, dtype: Any,
                weighted: bool, l2_bytes: int) -> dict[str, int]:
        """Plan a native hybrid MM from call metadata and this config's policies.

        Parameters
        ----------
        rows, cols, nnz : int
            Compressed rows, output columns and stored edges.
        batch : int
            Event columns after vmap folding.
        dtype : dtype
            Weight dtype; accumulation uses at least float32.
        weighted : bool
            Reserve an event-value queue for weighted events.
        l2_bytes : int
            Smallest L2 size among target devices.

        Returns
        -------
        dict of str to int
            Fresh mapping with ``group``, ``long_threshold``, ``long_tasks``,
            ``chunk``, ``task_capacity``, ``workspace_words`` and
            ``short_rows_size``. Mutating it cannot alter cached plans.

        Notes
        -----
        Only immutable integer tuples are cached. All computations happen
        during lowering, and the config never captures array values or
        target-device properties.
        """
        rows, cols, nnz, batch = (_integer(value, name) for name, value in (
            ('rows', rows), ('cols', cols), ('nnz', nnz), ('batch', batch)))
        l2_bytes = _integer(l2_bytes, 'l2_bytes')
        if max(rows, cols, batch) > _INT32_MAX:
            raise ValueError('MM dimensions exceed int32 range')
        values = self._mm_plan_values(rows, cols, nnz, batch, np.dtype(dtype), bool(weighted), l2_bytes)
        return dict(zip(self._MM_PLAN_FIELDS, values))

    @lru_cache(maxsize=256)
    def _mm_plan_values(self, rows: int, cols: int, nnz: int, batch: int,
                        dtype: np.dtype[Any], weighted: bool, l2_bytes: int) -> tuple[int, ...]:
        group = self.mm_group(rows, nnz)
        threshold = self.mm_threshold(rows, nnz)
        if not (rows and cols and nnz and batch):
            return group, threshold, 0, 1, 0, 0, 0
        long_tasks = -(-nnz // self.MM_TASK_EDGES) + min(rows, nnz // (threshold + 1))
        value_bytes = dtype.itemsize
        output_bytes = cols * max(4, value_bytes)
        compact_chunk = max(1, min(
            (l2_bytes // self.mm_l2_divisor) // output_bytes,
            max(nnz, self.mm_short_rows_floor) // rows, batch, self.mm_max_chunk,
        ))
        divisor = (self.mm_sparse_l2_divisor
                   if group == 4 and l2_bytes >= self.mm_sparse_l2_min_bytes else self.mm_l2_divisor)
        cache_budget = l2_bytes // divisor
        fit = cache_budget // output_bytes
        short_rows_fit = max(nnz, self.mm_short_rows_floor, cache_budget // 4) // rows
        chunk = max(1, min(fit, short_rows_fit, batch, self.mm_max_chunk))
        groups = -(-batch // chunk)
        if groups == -(-batch // compact_chunk):
            chunk = compact_chunk
        capacity = chunk * long_tasks
        queue_bytes = ((capacity * value_bytes + 3) // 4 * 4) if weighted else 0
        workspace_bytes = ((8 * batch + 7) // 8 * 8) + 8 * groups + 12 * capacity + queue_bytes
        short_rows = chunk * rows
        if max(capacity, workspace_bytes, short_rows) > _INT64_MAX:
            raise ValueError('MM plan or scratch size exceeds int64 range')
        return group, threshold, long_tasks, chunk, capacity, workspace_bytes // 4, short_rows

    def to_dict(self) -> dict[str, int]:
        """Serialize scheduling fields without plans, arrays or diagnostics.

        Returns
        -------
        dict of str to int
            The four MV constants and five MM policies.
        """
        return {name: int(getattr(self, name)) for name in self.PARAMETER_FIELDS}

    @classmethod
    def from_dict(cls, parameters: Mapping[str, Any]) -> CSRConfig:
        """Construct a config from flat fields, defaulting omitted policies.

        Parameters
        ----------
        parameters : mapping
            Flat scheduling parameters. Unknown metadata keys are ignored.

        Returns
        -------
        CSRConfig
            Validated immutable configuration.
        """
        return cls(**{name: parameters[name] for name in cls.PARAMETER_FIELDS if name in parameters})

    def __str__(self) -> str:
        base = repr(self)
        if not self.benchmark_records:
            return base
        lines = [base, 'benchmark results:']
        for i, record in enumerate(self.benchmark_records, 1):
            candidate = record['config']
            if isinstance(candidate, Mapping):
                candidate = self.from_dict(candidate)
            line = (f'  {i}. {candidate!r}: '
                    f'{record["elapsed_ms"]:.3f} ms ({record["per_call_us"]:.3f} us/call)')
            if record.get('task_capacity') is not None:
                line += f', task_capacity={record["task_capacity"]}'
            lines.append(line)
        return '\n'.join(lines)

    def tree_flatten(self) -> tuple[tuple[Any, ...], CSRConfig]:
        """Flatten to no array leaves and this immutable static config.

        Returns
        -------
        children : tuple
            Always empty.
        auxiliary : CSRConfig
            This config, with diagnostics excluded from equality and hashing.
        """
        return (), self

    @classmethod
    def tree_unflatten(cls, auxiliary: CSRConfig, children: Sequence[Any]) -> CSRConfig:
        """Restore the immutable configuration from its auxiliary data.

        Parameters
        ----------
        auxiliary : CSRConfig
            Metadata returned by ``tree_flatten``.
        children : sequence
            Unused; a config has no array leaves.

        Returns
        -------
        CSRConfig
            The configuration carried by the tree structure.
        """
        return auxiliary


def _config_from_spec(spec: CSRConfig | None) -> CSRConfig:
    """Resolve a primitive's static config, without dropping MM policies."""
    if spec is None:
        return CSRConfig()
    if not isinstance(spec, CSRConfig):
        raise TypeError('config_spec must be a CSRConfig or None')
    return spec


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
                      tuple(shape), indices.size, config)
        return config
    metadata = {'shape': tuple(shape), 'nnz': indices.size}
    return load_csr_config(device_kind=_device_kind(indices), matrix_metadata=metadata)
