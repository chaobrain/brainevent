
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

"""One-time per-GPU initialization for the CSR *hybrid* scheduler constants.

The public entry point is :func:`init_csr_config`.  It is **not** called automatically and
must never run inside a JIT closure.  A user runs it once on their GPU (e.g. before a long
simulation); the winning :class:`~brainevent._csr.config.CSRConfig` is persisted
per GPU model via :func:`save_csr_config`, and every later process picks it up
through :func:`load_csr_config`.

The benchmark compiles the **production** ``hybrid_mv/hybrid_spmv_binary.cu`` (with per-config
``-DBE_HYBRID_*`` flags) — so the timed kernel is exactly the one used at run time.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from brainevent._op import get_cache_dir
from ..config import _BenchmarkRecord, CSRConfig, MVSchedule

CONFIG_FILENAME = 'csr_hybrid_config.json'
_ENV_OVERRIDE = 'BRAINEVENT_CSR_HYBRID_CONFIG'
_CONFIG_FIELDS = ('block_size', 'fixed_scatter_blocks', 'tpr_threshold', 'task_nnz')

_CUDA_SOURCE = Path(__file__).parent / "hybrid_mv" / "hybrid_spmv_binary.cu"
_MODULE_BASE = "csr_hybrid_spmv_binary_tune"
_TARGET_FUNCTION = "binary_csrmv_wat_hybrid_homo_f32_bool"
_INT32_MAX = int(np.iinfo(np.int32).max)

#: Neurons per unit ``scale`` for the tuning workload: ``n_pre = n_post = _N_BASE * scale``.
#: dev/"COBA EI benchmark.py" uses ``_N=4000``; tuning uses a lighter base to keep the
#: default sweep's memory footprint modest.
_N_BASE = 1000

#: Candidate sweep (block, fixed_scatter_blocks, tpr_threshold, task_nnz).
DEFAULT_CANDIDATES: tuple[CSRConfig, ...] = (
    CSRConfig(mv=MVSchedule(512, 1024, 128, 512)),
    CSRConfig(mv=MVSchedule(128, 1024, 128, 2048)),
    CSRConfig(mv=MVSchedule(128, 2048, 128, 4096)),
    CSRConfig(mv=MVSchedule(256, 1024, 128, 4096)),
    CSRConfig(mv=MVSchedule(256, 2048, 128, 4096)),
    CSRConfig(mv=MVSchedule(256, 4096, 128, 4096)),
    CSRConfig(mv=MVSchedule(512, 2048, 128, 4096)),
    CSRConfig(mv=MVSchedule(512, 4096, 128, 4096)),
    CSRConfig(mv=MVSchedule(256, 2048, 256, 4096)),
    CSRConfig(mv=MVSchedule(128, 2048, 512, 2048)),
    CSRConfig(mv=MVSchedule(256, 1024, 512, 2048)),
    CSRConfig(mv=MVSchedule(256, 2048, 512, 2048)),
    CSRConfig(mv=MVSchedule(512, 2048, 512, 2048)),
    CSRConfig(mv=MVSchedule(256, 2048, 512, 4096)),
    CSRConfig(mv=MVSchedule(256, 2048, 1024, 4096)),
    CSRConfig(mv=MVSchedule(256, 4096, 128, 1024)),
    CSRConfig(mv=MVSchedule(256, 2048, 128, 8192)),
)


def _current_device_kind() -> str:
    try:
        import jax
        return str(jax.devices()[0].device_kind)
    except Exception:
        return ''


def _mv_from_mapping(data: Mapping[str, Any]) -> MVSchedule:
    return MVSchedule(
        block_size=int(data['block_size']),
        fixed_scatter_blocks=int(data['fixed_scatter_blocks']),
        tpr_threshold=int(data['tpr_threshold']),
        task_nnz=int(data['task_nnz']),
    )


def _mv_to_mapping(schedule: MVSchedule) -> dict[str, int]:
    return {key: int(getattr(schedule, key)) for key in _CONFIG_FIELDS}


def load_csr_config(device_kind: str | None = None) -> CSRConfig:
    """Read a fresh tuning snapshot for a selected device.

    Parameters
    ----------
    device_kind : str or None, optional
        Device model. When omitted, use the first JAX device's model.

    Returns
    -------
    CSRConfig
        The environment override, saved device entry, or default parameters,
        in that order. Loading never benchmarks or prepares a matrix.

    Notes
    -----
    ``BRAINEVENT_CSR_HYBRID_CONFIG`` and ``csr_hybrid_config.json`` retain their
    original formats. Existing CSR objects retain their own snapshots when a
    later load or save changes the persisted configuration.
    """
    raw = os.environ.get(_ENV_OVERRIDE)
    if raw:
        entry = json.loads(raw)
    else:
        path = Path(get_cache_dir()) / CONFIG_FILENAME
        if not path.exists():
            return CSRConfig()
        try:
            store = json.loads(path.read_text(encoding='utf-8'))
            entry = store.get(_current_device_kind() if device_kind is None else device_kind)
            if entry is None:
                return CSRConfig()
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return CSRConfig()

    try:
        records = []
        for record in entry.get('benchmark_records', ()):
            candidate = record['config']
            records.append(_BenchmarkRecord(
                config=CSRConfig(mv=_mv_from_mapping(candidate)),
                elapsed_ms=float(record['elapsed_ms']),
                per_call_us=float(record['per_call_us']),
                task_capacity=(None if record.get('task_capacity') is None
                               else int(record['task_capacity'])),
            ))
        return CSRConfig(
            mv=_mv_from_mapping(entry),
            benchmark_records=tuple(records),
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        if raw:
            raise
        return CSRConfig()


def save_csr_config(
    config: CSRConfig,
    device_kind: str | None = None,
    benchmark_records: Sequence[_BenchmarkRecord | Mapping[str, Any]] | None = None,
) -> Path:
    """Persist tuning parameters and diagnostic records for one device.

    Parameters
    ----------
    config : CSRConfig
        Snapshot whose four scheduling constants are saved.
    device_kind : str or None, optional
        Target device model; defaults to the first JAX device's model.
    benchmark_records : sequence or None, optional
        Tuning records. Defaults to records carried by ``config``.

    Returns
    -------
    pathlib.Path
        Updated configuration file. Other device entries are retained.

    Notes
    -----
    Device properties, primary/mirror masks, and MM plans are never persisted.
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
        except ValueError:
            pass
    entry: dict[str, Any] = _mv_to_mapping(config.mv)
    records = config.benchmark_records if benchmark_records is None else benchmark_records
    if records:
        saved_records = []
        for record in records:
            if isinstance(record, _BenchmarkRecord):
                candidate, elapsed_ms, per_call_us, capacity = (
                    record.config, record.elapsed_ms, record.per_call_us, record.task_capacity)
            else:
                candidate, elapsed_ms, per_call_us, capacity = (
                    record['config'], record['elapsed_ms'], record['per_call_us'], record.get('task_capacity'))
            saved_records.append({
                'config': _mv_to_mapping(candidate.mv),
                'elapsed_ms': float(elapsed_ms),
                'per_call_us': float(per_call_us),
                'task_capacity': None if capacity is None else int(capacity),
            })
        entry['benchmark_records'] = saved_records
    store[device_kind] = entry
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True), encoding='utf-8')
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

    output = None
    for step in range(max(int(warmup), 0)):
        output, task_begin, task_end, status = call(
            weights,
            indices,
            indptr,
            spikes[step % batch_size],
            task_begin,
            task_end,
            status,
        )
    if output is not None:
        jax.block_until_ready((output, task_begin, task_end, status))

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

    return {
        "config": config,
        "elapsed_ms": elapsed_ms,
        "per_call_us": elapsed_ms * 1000.0 / float(steps),
        "task_capacity": task_capacity,
    }


def run_benchmark(
    *,
    scale: float = 500.0,
    conn: int = 2000,
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
    """Time every candidate on a synthetic COBA-like CSR; return records sorted best-first.

    Parameters
    ----------
    scale : float, optional
        Matrix dimensions in units of 1000 neurons.
    conn : int, optional
        Connections per row.
    n_pre, n_post : int or None, optional
        Explicit dimensions overriding ``scale``.
    batch_size : int, optional
        Number of generated spike vectors reused during timing.
    steps : int, optional
        Number of timed kernel calls.
    spike_sparsity : float, optional
        Fraction of active neurons in each spike vector.
    seed : int, optional
        Seed for synthetic structure and events.
    warmup : int, optional
        Untimed calls before measurement.
    candidates : sequence of CSRConfig, optional
        Scheduling candidates to benchmark.
    force_rebuild, verbose_compile : bool, optional
        CUDA compiler controls.
    show_progress : bool, optional
        Write candidate-sweep progress to stderr.

    Returns
    -------
    list of dict
        Configuration, timing and task-capacity records sorted by elapsed time.

    Notes
    -----
    The default operating point mirrors a ``scale=500, conn=2000`` COBA EI network
    (``n_pre = n_post = _N_BASE * scale = 500_000`` neurons, 2000 connections/row,
    ~4 GB of int32 indices).  Pass ``n_pre``/``n_post`` to override the neuron counts
    derived from ``scale``.  ``batch_size`` controls how many spike vectors are
    materialized; timed calls run for ``steps`` iterations and cycle through that batch.
    """
    import jax
    import jax.numpy as jnp

    num = int(_N_BASE * scale)
    n_pre = num if n_pre is None else int(n_pre)
    n_post = num if n_post is None else int(n_post)
    batch_size = int(batch_size)
    steps = int(steps)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    candidates = tuple(candidates)
    indices_np, indptr_np = _make_uniform_csr(n_pre=n_pre, n_post=n_post, conn=conn, seed=seed)
    spikes_np = _make_spike_batch(batch_size=batch_size, n_pre=n_pre, spike_sparsity=spike_sparsity, seed=seed + 1)

    weights = jnp.asarray([1.0], dtype=jnp.float32)
    indices = jnp.asarray(indices_np, dtype=jnp.int32)
    indptr = jnp.asarray(indptr_np, dtype=indptr_np.dtype)
    spikes = jnp.asarray(spikes_np, dtype=jnp.bool_)
    jax.block_until_ready((weights, indices, indptr, spikes))

    records = []
    total = len(candidates)
    if show_progress and total:
        _write_progress(0, total)
    for i, config in enumerate(candidates, start=1):
        record = _benchmark_config(
            config,
            weights=weights,
            indices=indices,
            indptr=indptr,
            spikes=spikes,
            n_pre=n_pre,
            n_post=n_post,
            conn=conn,
            batch_size=batch_size,
            warmup=warmup,
            steps=steps,
            force_rebuild=force_rebuild,
            verbose_compile=verbose_compile,
        )
        records.append(record)
        if show_progress:
            _write_progress(i, total)
    records.sort(key=lambda r: r["elapsed_ms"])
    return records


def init_csr_config(
    *,
    save: bool = True,
    device_kind: str | None = None,
    candidates: Sequence[CSRConfig] = DEFAULT_CANDIDATES,
    **benchmark_kwargs,
) -> CSRConfig:
    """Initialize the per-GPU CSR config: benchmark candidates and (by default) persist the winner.

    Parameters
    ----------
    save : bool, optional
        Persist the winning scheduling parameters and all timing records.
    device_kind : str or None, optional
        Device-model key for persistence; defaults to the current device.
    candidates : sequence of CSRConfig, optional
        Candidate scheduling configurations.
    **benchmark_kwargs
        Additional arguments forwarded to :func:`run_benchmark`.

    Returns
    -------
    CSRConfig
        Winning parameter snapshot, without matrix-specific preparation.

    Notes
    -----
    Run this **once** manually before simulating; later processes read the saved config via
    :func:`load_csr_config`.  Returns the winning
    :class:`~brainevent._csr.config.CSRConfig`.
    """
    records = run_benchmark(candidates=candidates, **benchmark_kwargs)
    if not records:
        raise RuntimeError("no candidate configs were benchmarked")
    best = records[0]["config"]
    if save:
        save_csr_config(best, device_kind or None, benchmark_records=records)
    return best
