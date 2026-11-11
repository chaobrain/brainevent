
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Explicit CSR hybrid tuning and metadata-aware per-device configuration.

Initialization benchmarks the production binary SpMV kernel at two matrix
densities. Loading never benchmarks or prepares a matrix; it resolves a fresh
parameter snapshot from an override, a matching workload, a device fallback,
or the library defaults.
"""

from __future__ import annotations

import json
import logging
import math
import operator
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from brainevent._op import get_cache_dir
from brainevent._csr.config import _BenchmarkRecord, CSRConfig, MVSchedule

CONFIG_FILENAME = 'csr_hybrid_config.json'
_ENV_OVERRIDE = 'BRAINEVENT_CSR_HYBRID_CONFIG'
_CONFIG_FIELDS = ('block_size', 'fixed_scatter_blocks', 'tpr_threshold', 'task_nnz')

_CUDA_SOURCE = Path(__file__).parent / "hybrid_mv" / "hybrid_spmv_binary.cu"
_MODULE_BASE = "csr_hybrid_spmv_binary_tune"
_TARGET_FUNCTION = "binary_csrmv_wat_hybrid_homo_f32_bool"
_INT32_MAX = int(np.iinfo(np.int32).max)
_LOG = logging.getLogger(__name__)

#: Neurons per unit ``scale`` for the tuning workload: ``n_pre = n_post = _N_BASE * scale``.
#: dev/"COBA EI benchmark.py" uses ``_N=4000``; tuning uses a lighter base to keep the
#: default sweep's memory footprint modest.
_N_BASE = 1000

#: Candidate axes, ordered as the four fields in ``MVSchedule``.
PARAMETER_VALUES = ((512, 256), (512, 1024, 2048), (128, 256), (512, 1024, 2048, 4096))
DEFAULT_CANDIDATES: tuple[CSRConfig, ...] = tuple(
    CSRConfig(mv=MVSchedule(*values)) for values in product(*PARAMETER_VALUES)
)
DEFAULT_DENSITIES = (0.03, 0.003)


def _current_device_kind() -> str:
    try:
        import jax
        return str(jax.devices()[0].device_kind)
    except Exception:
        return ''


def _mv_from_mapping(data: Mapping[str, Any]) -> MVSchedule:
    return MVSchedule(**{name: data[name] for name in _CONFIG_FIELDS})


def _mv_to_mapping(schedule: MVSchedule) -> dict[str, int]:
    return {key: int(getattr(schedule, key)) for key in _CONFIG_FIELDS}


def _matrix_key(metadata: Mapping[str, Any] | None) -> tuple[int, int, int] | None:
    """Read only static identity metadata; symbolic or missing values cannot match."""
    if not isinstance(metadata, Mapping):
        return None
    try:
        shape = metadata.get('shape', (metadata.get('n_pre'), metadata.get('n_post')))
        if not isinstance(shape, (tuple, list)):
            return None
        values = (*shape, metadata['nnz'])
        if len(values) != 3 or any(
                not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                for value in values):
            return None
        rows, cols, nnz = map(operator.index, values)
        if min(rows, cols, nnz) < 0:
            return None
        return rows, cols, nnz
    except (KeyError, TypeError, ValueError):
        return None


def _config_from_mapping(entry: Mapping[str, Any]) -> CSRConfig:
    schedule = _mv_from_mapping(entry.get('best_config', entry))
    records = []
    diagnostics = entry.get('benchmark_records', ())
    if not isinstance(diagnostics, (list, tuple)):
        diagnostics = ()
    for record in diagnostics:
        try:
            if record.get('status', 'ok') != 'ok':
                continue
            elapsed, per_call = float(record['elapsed_ms']), float(record['per_call_us'])
            if not math.isfinite(elapsed) or not math.isfinite(per_call) or min(elapsed, per_call) < 0:
                continue
            records.append(_BenchmarkRecord(
                config=CSRConfig(mv=_mv_from_mapping(record['config'])),
                elapsed_ms=elapsed, per_call_us=per_call,
                task_capacity=(None if record.get('task_capacity') is None
                               else operator.index(record['task_capacity'])),
            ))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return CSRConfig(mv=schedule, benchmark_records=tuple(records))


def load_csr_config(
    device_kind: str | None = None,
    *,
    matrix_metadata: Mapping[str, Any] | None = None,
) -> CSRConfig:
    """Resolve a fresh parameter snapshot for a device and optional matrix.

    Parameters
    ----------
    device_kind : str or None, optional
        Device model. When omitted, use the first JAX device's model.
    matrix_metadata : mapping or None, optional
        Static ``shape`` and ``nnz`` identify an exact workload. Missing or
        symbolic metadata skips matching and uses a device fallback.

    Returns
    -------
    CSRConfig
        Environment override, matching workload, device fallback, or library
        defaults, in that order. A caller's explicit configuration is resolved
        before this function is called.

    Notes
    -----
    Existing flat GPU entries remain valid. Workload matching uses integer
    shape/nnz metadata rather than floating-point density keys. Existing
    matrices retain their snapshots when the persisted configuration changes.
    """
    raw = os.environ.get(_ENV_OVERRIDE)
    if raw:
        config = _config_from_mapping(json.loads(raw))
        _LOG.debug('CSR config source=explicit environment parameters=%s', config)
        return config

    path = Path(get_cache_dir()) / CONFIG_FILENAME
    device_kind = _current_device_kind() if device_kind is None else device_kind
    reason = 'no valid configuration for target GPU'
    try:
        store = json.loads(path.read_text(encoding='utf-8'))
        entry = store.get(device_kind, {})
        workloads = entry.get('workloads', {})
        valid_workloads = []
        if isinstance(workloads, Mapping):
            for name, workload in workloads.items():
                try:
                    key = _matrix_key(workload.get('metadata'))
                    if key is not None and workload.get('status', 'ok') == 'ok':
                        valid_workloads.append((key, str(name), _config_from_mapping(workload)))
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
        # Equal shapes also order densities by nnz. The persistent label breaks
        # ties when distinct requested densities round to the same matrix.
        valid_workloads.sort(key=lambda item: (item[0], item[1]))
        key = _matrix_key(matrix_metadata)
        for saved_key, _, config in valid_workloads:
            if key is not None and saved_key == key:
                _LOG.debug('CSR config source=local_matched path=%s gpu=%s metadata=%s parameters=%s',
                           path, device_kind, key, config)
                return config
        try:
            config = _config_from_mapping(entry)
        except (ValueError, KeyError, TypeError, AttributeError):
            if not valid_workloads:
                raise ValueError(reason) from None
            config = valid_workloads[0][2]
        _LOG.debug('CSR config source=local_fallback path=%s gpu=%s requested=%s parameters=%s',
                   path, device_kind, key, config)
        return config
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        reason = str(exc)
    config = CSRConfig()
    _LOG.debug('CSR config source=library_default path=%s gpu=%s reason=%s parameters=%s',
               path, device_kind, reason, config)
    return config


def _serialize_records(records: Sequence[_BenchmarkRecord | Mapping[str, Any]]) -> list[dict]:
    saved = []
    for record in records:
        row: dict[str, Any]
        if isinstance(record, _BenchmarkRecord):
            row = dict(config=record.config, elapsed_ms=record.elapsed_ms,
                       per_call_us=record.per_call_us, task_capacity=record.task_capacity)
        else:
            row = dict(record)
        row['config'] = _mv_to_mapping(row['config'].mv)
        for name in ('elapsed_ms', 'per_call_us'):
            if name in row:
                row[name] = float(row[name])
        if row.get('task_capacity') is not None:
            row['task_capacity'] = int(row['task_capacity'])
        saved.append(row)
    return saved


def save_csr_config(
    config: CSRConfig,
    device_kind: str | None = None,
    benchmark_records: Sequence[_BenchmarkRecord | Mapping[str, Any]] | None = None,
    *,
    matrix_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Merge a device fallback or one workload winner into the local store.

    Parameters
    ----------
    config : CSRConfig
        Snapshot whose four scheduling constants are saved.
    device_kind : str or None, optional
        Target device model; defaults to the first JAX device's model.
    benchmark_records : sequence or None, optional
        Tuning records. Defaults to records carried by ``config``.
    matrix_metadata : mapping or None, optional
        Workload settings including integer ``shape`` and ``nnz``. When given,
        save a separate workload without replacing the flat GPU fallback.

    Returns
    -------
    pathlib.Path
        Updated configuration file. Other device entries are retained.

    Notes
    -----
    Device properties, primary/mirror masks, and MM plans are never persisted.
    Other GPU entries and workload records are retained. The path still uses
    the current compilation cache directory; cross-version migration is not
    performed.
    """
    if device_kind is None:
        device_kind = _current_device_kind()
    if not device_kind:
        raise RuntimeError('cannot determine device_kind; pass it explicitly')
    path = Path(get_cache_dir()) / CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    store: dict[str, Any] = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(old, dict):
                store = old
        except (OSError, ValueError):
            pass
    existing = store.get(device_kind, {})
    entry: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    records = config.benchmark_records if benchmark_records is None else benchmark_records
    parameters = _mv_to_mapping(config.mv)
    config.validate()
    if matrix_metadata is None:
        entry.update(parameters)
        entry.pop('benchmark_records', None)
        if records:
            entry['benchmark_records'] = _serialize_records(records)
    else:
        key = _matrix_key(matrix_metadata)
        if key is None:
            raise ValueError('matrix_metadata must contain concrete shape and nnz')
        rows, cols, nnz = key
        metadata = dict(matrix_metadata, shape=[rows, cols], nnz=nnz)
        metadata['actual_density'] = nnz / (rows * cols) if rows and cols else 0.0
        workload_id = f'{rows}x{cols}:nnz={nnz}'
        if metadata.get('requested_density') is not None:
            requested_density = float(metadata['requested_density'])
            if not math.isfinite(requested_density) or not 0 < requested_density <= 1:
                raise ValueError('requested_density must be finite and in (0, 1]')
            metadata['requested_density'] = requested_density
            workload_id += f':density={requested_density:.17g}'
        previous_workloads = entry.get('workloads', {})
        workloads = dict(previous_workloads) if isinstance(previous_workloads, Mapping) else {}
        workloads[workload_id] = {
            'metadata': metadata, 'best_config': parameters,
            'benchmark_records': _serialize_records(records), 'status': 'ok',
        }
        entry['workloads'] = workloads
    store[device_kind] = entry
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)
    return path


def _progress_bar(completed: int, total: int, *, width: int = 30) -> str:
    total = max(int(total), 1)
    completed = min(max(int(completed), 0), total)
    ratio = completed / total
    filled = int(round(width * ratio))
    bar = "#" * filled + "." * (width - filled)
    return f"CSR hybrid tuning [{bar}] {completed}/{total} ({ratio * 100.0:5.1f}%)"


def _write_progress(completed: int, total: int) -> None:
    sys.stderr.write("\r" + _progress_bar(completed, total))
    if completed >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _task_capacity_for_uniform_rows(*, n_rows: int, row_conn: int, config: CSRConfig) -> int:
    if row_conn <= config.mv.tpr_threshold:
        return 0
    chunks_per_row = (row_conn + config.mv.task_nnz - 1) // config.mv.task_nnz
    capacity = int(n_rows) * int(chunks_per_row)
    if capacity > _INT32_MAX:
        raise ValueError("task_capacity exceeds int32 range")
    return capacity


def _make_uniform_csr(*, n_pre: int, n_post: int, conn: int, seed: int):
    if not 1 <= conn <= n_post:
        raise ValueError("conn must be in [1, n_post]")
    nnz = int(n_pre) * int(conn)
    offset_dtype = np.int64 if nnz > _INT32_MAX else np.int32
    rng = np.random.default_rng(seed)
    indices_np = rng.integers(0, n_post, size=nnz, dtype=np.int32)
    indptr_np = (np.arange(n_pre + 1, dtype=np.int64) * conn).astype(offset_dtype, copy=False)
    return indices_np, indptr_np


def _make_spike_batch(*, batch_size: int, n_pre: int, spike_sparsity: float, seed: int) -> np.ndarray:
    if not 0.0 <= spike_sparsity <= 1.0:
        raise ValueError("spike_sparsity must be in [0, 1]")
    active_count = min(max(int(round(n_pre * spike_sparsity)), 0), n_pre)
    spikes = np.zeros((batch_size, n_pre), dtype=np.bool_)
    if active_count == 0:
        return spikes
    rng = np.random.default_rng(seed)
    for b in range(batch_size):
        spikes[b, rng.choice(n_pre, size=active_count, replace=False)] = True
    return spikes


def _benchmark_config(
    config: CSRConfig,
    *,
    weights,
    indices,
    indptr,
    spikes,
    n_pre: int,
    n_post: int,
    conn: int,
    batch_size: int,
    warmup: int,
    steps: int,
    force_rebuild: bool,
    verbose_compile: bool,
    expected_output: np.ndarray | None = None,
) -> dict:
    import jax
    import jax.numpy as jnp

    from brainevent._op import load_cuda_file

    module_name = _MODULE_BASE + config.mv.module_suffix()
    module = load_cuda_file(
        _CUDA_SOURCE,
        name=module_name,
        extra_include_paths=[str(_CUDA_SOURCE.parent)],
        extra_cuda_cflags=config.mv.compile_flags(),
        force_rebuild=force_rebuild,
        verbose=verbose_compile,
        allow_cuda_graph=True,
    )
    if _TARGET_FUNCTION not in module.function_names:
        raise RuntimeError(f"{_TARGET_FUNCTION} was not registered by {module.path}")
    batch_size = int(batch_size)
    steps = int(steps)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    task_capacity = _task_capacity_for_uniform_rows(n_rows=n_pre, row_conn=conn, config=config)
    task_dtype = jnp.dtype(indptr.dtype)
    task_begin = jnp.empty((task_capacity,), dtype=task_dtype)
    task_end = jnp.empty((task_capacity,), dtype=task_dtype)
    status = jnp.empty((2,), dtype=jnp.int32)

    target_name = f"{module_name}.{_TARGET_FUNCTION}"
    outs = (
        jax.ShapeDtypeStruct((n_post,), weights.dtype),
        jax.ShapeDtypeStruct((task_capacity,), task_dtype),
        jax.ShapeDtypeStruct((task_capacity,), task_dtype),
        jax.ShapeDtypeStruct((2,), np.dtype(np.int32)),
    )

    @jax.jit
    def call(weights, indices, indptr, vector, task_begin, task_end, status):
        return jax.ffi.ffi_call(
            target_name, outs, input_output_aliases={4: 1, 5: 2, 6: 3}
        )(weights, indices, indptr, vector, task_begin, task_end, status, task_capacity=task_capacity)

    # Even warmup=0 needs one untimed call to exclude JIT compilation.
    for step in range(max(int(warmup), 1)):
        output, task_begin, task_end, status = call(
            weights,
            indices,
            indptr,
            spikes[step % batch_size],
            task_begin,
            task_end,
            status,
        )
    jax.block_until_ready((output, task_begin, task_end, status))
    if int(np.asarray(jax.device_get(status[1]))):
        raise RuntimeError(f"CUDA hybrid task queue overflowed during warmup for config {config}")

    start = time.perf_counter()
    for step in range(steps):
        output, task_begin, task_end, status = call(
            weights,
            indices,
            indptr,
            spikes[step % batch_size],
            task_begin,
            task_end,
            status,
        )
    jax.block_until_ready((output, task_begin, task_end, status))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if int(np.asarray(jax.device_get(status[1]))):
        raise RuntimeError(f"CUDA hybrid task queue overflowed for config {config}")
    if expected_output is not None:
        actual = np.asarray(jax.device_get(output))
        if actual.shape != expected_output.shape or not np.allclose(
                actual, expected_output, rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"CUDA hybrid numerical output mismatch for config {config}")

    return {
        "config": config,
        "status": "ok",
        "elapsed_ms": elapsed_ms,
        "per_call_us": elapsed_ms * 1000.0 / float(steps),
        "task_capacity": task_capacity,
        "checked": expected_output is not None,
    }


def _workload_shape(scale: float, n_pre: int | None, n_post: int | None) -> tuple[int, int]:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be positive and finite')
    rows = int(_N_BASE * scale) if n_pre is None else operator.index(n_pre)
    cols = int(_N_BASE * scale) if n_post is None else operator.index(n_post)
    if not 0 < rows <= _INT32_MAX or not 0 < cols <= _INT32_MAX:
        raise ValueError('n_pre and n_post must be positive int32 dimensions')
    return rows, cols


def _density_conn(n_post: int, density: float) -> int:
    if not math.isfinite(density) or not 0 < density <= 1:
        raise ValueError('density must be finite and in (0, 1]')
    return max(1, int(round(n_post * density)))


def _reference_output(indices: np.ndarray, indptr: np.ndarray,
                      vector: np.ndarray, n_post: int) -> np.ndarray:
    # Count all active edges, including duplicate targets, independently of CUDA.
    expected = np.zeros(n_post, dtype=np.float64)
    for row in np.flatnonzero(vector):
        np.add.at(expected, indices[int(indptr[row]):int(indptr[row + 1])], 1.0)
    return expected


def run_benchmark(
    *,
    scale: float = 500.0,
    conn: int | None = None,
    density: float | None = None,
    n_pre: int | None = None,
    n_post: int | None = None,
    batch_size: int = 100,
    steps: int = 200,
    spike_sparsity: float = 1.0 / 250.0,
    seed: int = 123,
    warmup: int = 50,
    candidates: Sequence[CSRConfig] = DEFAULT_CANDIDATES,
    force_rebuild: bool = False,
    verbose_compile: bool = False,
    show_progress: bool = True,
) -> list[dict]:
    """Measure candidates on one equal-row CSR workload and retain failures.

    Parameters
    ----------
    scale : float, optional
        Default dimensions in units of 1000 neurons.
    conn : int or None, optional
        Connections per row. Defaults to 2000 when density is also omitted.
    density : float or None, optional
        Matrix entry density in (0, 1], converted to round(n_post * density)
        connections per row. Cannot be supplied together with conn.
    n_pre, n_post : int or None, optional
        Explicit dimensions overriding scale.
    batch_size : int, optional
        Number of generated bool event vectors reused during timing.
    steps : int, optional
        Timed FFI calls per candidate.
    spike_sparsity : float, optional
        Fraction of active rows, independent of matrix density.
    seed : int, optional
        Reproducible matrix and event seed.
    warmup : int, optional
        Untimed calls. At least one call primes compilation even when zero.
    candidates : sequence of CSRConfig, optional
        Scheduling candidates; defaults to the 48-configuration product.
    force_rebuild, verbose_compile : bool, optional
        CUDA compiler controls.
    show_progress : bool, optional
        Write candidate progress to stderr.

    Returns
    -------
    list of dict
        Successful records sorted by elapsed time, followed by failed records
        with status and error. Candidate output is checked outside timing.

    Notes
    -----
    Time covers Python submission and GPU completion, including kernel output
    and status initialization. Input generation, uploads, scratch allocation,
    compilation, and numerical validation are outside timing. Matrices with
    more than int32 nnz require jax_enable_x64 before allocation.
    """
    import jax
    import jax.numpy as jnp

    n_pre, n_post = _workload_shape(scale, n_pre, n_post)
    if conn is not None and density is not None:
        raise ValueError('conn and density cannot be supplied together')
    conn = _density_conn(n_post, density) if density is not None else (2000 if conn is None else operator.index(conn))
    if not 1 <= conn <= n_post:
        raise ValueError('conn must be in [1, n_post]')
    batch_size, steps, warmup = map(operator.index, (batch_size, steps, warmup))
    if batch_size <= 0 or steps <= 0 or warmup < 0:
        raise ValueError('batch_size and steps must be positive; warmup must be nonnegative')
    if not math.isfinite(spike_sparsity) or not 0 <= spike_sparsity <= 1:
        raise ValueError('spike_sparsity must be finite and in [0, 1]')
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError('candidates must not be empty')
    for config in candidates:
        config.validate()
    nnz = n_pre * conn
    if nnz > _INT32_MAX and not jax.config.jax_enable_x64:
        raise RuntimeError('this workload requires jax_enable_x64 for int64 CSR offsets')

    # Avoid allocating a host copy of a matrix known not to fit on the device.
    devices = getattr(jax, 'devices', None)
    if devices is not None:
        device = devices()[0]
        stats = device.memory_stats() if hasattr(device, 'memory_stats') else None
        if stats and stats.get('bytes_limit') is not None:
            offset_bytes = 8 if nnz > _INT32_MAX else 4
            estimate = (4 * nnz + offset_bytes * (n_pre + 1)
                        + batch_size * n_pre + 4 * n_post + 12)
            available = stats['bytes_limit'] - stats.get('bytes_in_use', 0)
            if estimate > available:
                raise MemoryError(f'CSR workload needs at least {estimate} device bytes; {available} available')

    indices_np, indptr_np = _make_uniform_csr(n_pre=n_pre, n_post=n_post, conn=conn, seed=seed)
    spikes_np = _make_spike_batch(batch_size=batch_size, n_pre=n_pre, spike_sparsity=spike_sparsity, seed=seed + 1)
    expected = _reference_output(indices_np, indptr_np, spikes_np[(steps - 1) % batch_size], n_post)
    weights = jnp.asarray([1.0], dtype=jnp.float32)
    indices = jnp.asarray(indices_np, dtype=jnp.int32)
    indptr = jnp.asarray(indptr_np, dtype=indptr_np.dtype)
    spikes = jnp.asarray(spikes_np, dtype=jnp.bool_)
    jax.block_until_ready((weights, indices, indptr, spikes))

    records = []
    total = len(candidates)
    if show_progress:
        _write_progress(0, total)
    for i, config in enumerate(candidates, start=1):
        try:
            record = _benchmark_config(
                config, weights=weights, indices=indices, indptr=indptr, spikes=spikes,
                n_pre=n_pre, n_post=n_post, conn=conn, batch_size=batch_size,
                warmup=warmup, steps=steps, force_rebuild=force_rebuild,
                verbose_compile=verbose_compile, expected_output=expected,
            )
            record.setdefault('status', 'ok')
        except (MemoryError, RuntimeError, ValueError) as exc:
            record = {'config': config, 'status': 'error', 'error': str(exc)}
        records.append(record)
        if show_progress:
            _write_progress(i, total)
    records.sort(key=lambda record: (record['status'] != 'ok', record.get('elapsed_ms', math.inf)))
    return records


def init_csr_config(
    *,
    save: bool = True,
    device_kind: str | None = None,
    densities: Sequence[float] = DEFAULT_DENSITIES,
    candidates: Sequence[CSRConfig] = DEFAULT_CANDIDATES,
    scale: float = 500.0,
    n_pre: int | None = None,
    n_post: int | None = None,
    batch_size: int = 100,
    steps: int = 200,
    spike_sparsity: float = 1.0 / 250.0,
    seed: int = 123,
    warmup: int = 50,
    force_rebuild: bool = False,
    verbose_compile: bool = False,
    show_progress: bool = True,
) -> list[dict]:
    """Tune equal-size matrices at separate densities and persist each winner.

    Parameters
    ----------
    save : bool, optional
        Merge successful workload results into the per-device JSON store.
    device_kind : str or None, optional
        Device model key; omitted means the current device.
    densities : sequence of float, optional
        Matrix densities. Defaults to 3% and 0.3%, in that order.
    candidates : sequence of CSRConfig, optional
        Scheduling candidates shared by all workloads.
    scale : float, optional
        Default dimensions in units of 1000 neurons.
    n_pre, n_post : int or None, optional
        Common matrix dimensions overriding scale.
    batch_size, steps, spike_sparsity, seed, warmup : optional
        Input generation and timing settings shared by all workloads.
    force_rebuild, verbose_compile : bool, optional
        CUDA compiler controls. Rebuild is requested for the first workload;
        subsequent workloads reuse compiled modules.
    show_progress : bool, optional
        Write candidate-sweep progress to stderr.

    Returns
    -------
    list of dict
        One result per requested density, containing metadata, best_config,
        records, status and optional error/config_path. A failed workload has
        no winner and does not replace an existing saved record.

    Notes
    -----
    Call explicitly outside JIT. Loading and matrix construction never invoke
    initialization. The existing flat GPU fallback is retained when saving
    workload-specific winners.
    """
    n_pre, n_post = _workload_shape(scale, n_pre, n_post)
    densities, candidates = tuple(densities), tuple(candidates)
    if not densities or not candidates:
        raise ValueError('densities and candidates must not be empty')
    if len(set(densities)) != len(densities):
        raise ValueError('densities must not contain duplicates')
    connections = tuple(_density_conn(n_post, density) for density in densities)
    results = []
    for index, (density, conn) in enumerate(zip(densities, connections)):
        nnz = n_pre * conn
        metadata = dict(
            shape=(n_pre, n_post), n_pre=n_pre, n_post=n_post, nnz=nnz, conn=conn,
            requested_density=float(density), actual_density=conn / n_post,
            weight_dtype='float32', event_dtype='bool', index_dtype='int32',
            offset_dtype='int64' if nnz > _INT32_MAX else 'int32', homogeneous=True,
            batch_size=batch_size, steps=steps, warmup=warmup, spike_sparsity=spike_sparsity,
            seed=seed, timing_method='python_perf_counter_then_block_until_ready',
        )
        result: dict[str, Any] = dict(metadata=metadata, best_config=None, records=[], status='error')
        try:
            records = run_benchmark(
                n_pre=n_pre, n_post=n_post, density=density, batch_size=batch_size,
                steps=steps, spike_sparsity=spike_sparsity, seed=seed, warmup=warmup,
                candidates=candidates, force_rebuild=force_rebuild and index == 0,
                verbose_compile=verbose_compile, show_progress=show_progress,
            )
            result['records'] = records
            successful = [record for record in records if record.get('status', 'ok') == 'ok']
            if successful:
                best_record = min(successful, key=lambda record: record['elapsed_ms'])
                result.update(best_config=best_record['config'], status='ok')
            else:
                result['error'] = 'no candidate configuration completed successfully'
        except (MemoryError, RuntimeError) as exc:
            result['error'] = str(exc)
        if save and result['status'] == 'ok':
            path = save_csr_config(result['best_config'], device_kind, result['records'], matrix_metadata=metadata)
            result['config_path'] = str(path)
        results.append(result)
    return results
