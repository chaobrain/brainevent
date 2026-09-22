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

"""Configure and benchmark the TCSR hybrid CUDA preprocessing scheduler.

The three hybrid kernels (``binary_csrmv_hybrid.cu``,
``binary_indexed_csrmv_hybrid.cu``, and
``binary_indexed_csrmm_hybrid.cu``) expose four scheduler constants via
``-DBE_HYBRID_*`` compile-time macros. Direct TCSR matrix-matrix dispatch uses
the separate ``binary_csrmm_tile.cu`` ABI.

- ``block_size``            → ``BE_HYBRID_BLOCK_SIZE``
- ``fixed_scatter_blocks``  → ``BE_HYBRID_FIXED_SCATTER_BLOCKS``
- ``tpr_threshold``         → ``BE_HYBRID_TPR_THRESHOLD``
- ``task_nnz``              → ``BE_HYBRID_TASK_NNZ``

The macros are baked into the compiled ``.so`` as ``constexpr`` literals — there is no
runtime plumbing and no coupling with JAX arrays.  Two of the constants
(``tpr_threshold``, ``task_nnz``) *also* determine the size of the Python-side task
workspace buffers, so this module is the **only** place that defines them: both the
compile flags and the workspace sizing read from :func:`get_hybrid_config`, keeping the
``.so`` and the host allocation in lockstep.

The best values are GPU-specific. Users initialize them once with
:func:`init_csr_config` (manual, GPU-only); the winner is persisted per GPU
model in ``<cache_dir>/csr_hybrid_config.json``, in the same directory family
as the compiled ``.so``. Resolution order in :func:`get_hybrid_config`:

1. ``$BRAINEVENT_CSR_HYBRID_CONFIG`` — a JSON object (CI / one-off override).
2. The per-``device_kind`` entry in ``csr_hybrid_config.json``.
3. :data:`DEFAULT_HYBRID_CONFIG` — the values baked into the ``.cu`` defaults.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from brainevent._op import get_cache_dir

__all__ = [
    "HybridConfig",
    "HybridBenchmarkRecord",
    "DEFAULT_HYBRID_CONFIG",
    "validate_config",
    "compile_flags_for_config",
    "module_suffix_for_config",
    "get_hybrid_config",
    "save_hybrid_config",
    "hybrid_task_capacity",
    "current_device_kind",
    "DEFAULT_CANDIDATES",
    "run_benchmark",
    "init_csr_config",
]

CONFIG_FILENAME = "csr_hybrid_config.json"
_ENV_OVERRIDE = "BRAINEVENT_CSR_HYBRID_CONFIG"
_INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass(frozen=True)
class HybridConfig:
    """Represent scheduler constants shared by all hybrid CSR kernels.

    Parameters
    ----------
    block_size : int, optional
        CUDA thread-block size.
    fixed_scatter_blocks : int, optional
        Number of fixed scatter work blocks.
    tpr_threshold : int, optional
        Row-length threshold above which task-queue processing is used.
    task_nnz : int, optional
        Maximum number of nonzero entries represented by one queued task.
    benchmark_records : tuple of HybridBenchmarkRecord, optional
        Persisted timing records associated with this selected configuration.
    """

    block_size: int = 256
    fixed_scatter_blocks: int = 2048
    tpr_threshold: int = 128
    task_nnz: int = 4096
    benchmark_records: tuple["HybridBenchmarkRecord", ...] = field(
        default_factory=tuple,
        compare=False,
        repr=False,
    )

    def __str__(self) -> str:
        """Format the configuration and any associated benchmark records.

        Returns
        -------
        str
            Human-readable scheduler configuration and benchmark summary.
        """
        base = _format_config(self)
        if not self.benchmark_records:
            return base
        lines = [base, "benchmark results:"]
        for i, record in enumerate(self.benchmark_records, start=1):
            task_capacity = ""
            if record.task_capacity is not None:
                task_capacity = f", task_capacity={record.task_capacity}"
            lines.append(
                f"  {i}. {_format_config(record.config)}: "
                f"{record.elapsed_ms:.3f} ms ({record.per_call_us:.3f} us/call)"
                f"{task_capacity}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class HybridBenchmarkRecord:
    """Represent a timing result for one scheduler configuration.

    Parameters
    ----------
    config : HybridConfig
        Scheduler configuration measured by the benchmark.
    elapsed_ms : float
        Total elapsed time for all timed calls, in milliseconds.
    per_call_us : float
        Average elapsed time per call, in microseconds.
    task_capacity : int or None, optional
        Task-queue capacity allocated for the measured workload.
    """

    config: HybridConfig
    elapsed_ms: float
    per_call_us: float
    task_capacity: int | None = None


#: Matches the ``#ifndef BE_HYBRID_*`` defaults compiled into the ``.cu`` files.
DEFAULT_HYBRID_CONFIG = HybridConfig()


_CONFIG_FIELDS = ("block_size", "fixed_scatter_blocks", "tpr_threshold", "task_nnz")


def _format_config(config: HybridConfig) -> str:
    args = ", ".join(f"{field_name}={getattr(config, field_name)}" for field_name in _CONFIG_FIELDS)
    return f"HybridConfig({args})"


def validate_config(config: HybridConfig) -> HybridConfig:
    """Validate scheduler constants against the CUDA compile-time constraints.

    Parameters
    ----------
    config : HybridConfig
        Scheduler configuration to validate.

    Returns
    -------
    HybridConfig
        The unchanged validated configuration.

    Raises
    ------
    ValueError
        If a scheduler value violates a CUDA constraint.
    """
    if config.block_size <= 0:
        raise ValueError("block_size must be positive")
    if config.block_size % 32 != 0:
        raise ValueError("block_size must be a multiple of 32")
    if config.block_size > 1024:
        raise ValueError("block_size must not exceed 1024")
    if config.fixed_scatter_blocks <= 0:
        raise ValueError("fixed_scatter_blocks must be positive")
    if config.tpr_threshold < 0:
        raise ValueError("tpr_threshold must be non-negative")
    if config.task_nnz <= 0:
        raise ValueError("task_nnz must be positive")
    return config


def compile_flags_for_config(config: HybridConfig) -> list[str]:
    """Build NVCC definition flags for a scheduler configuration.

    Parameters
    ----------
    config : HybridConfig
        Scheduler configuration to encode.

    Returns
    -------
    list of str
        ``-DBE_HYBRID_*`` flags that bake the configuration into a CUDA module.

    Raises
    ------
    ValueError
        If the scheduler configuration is invalid.
    """
    config = validate_config(config)
    return [
        f"-DBE_HYBRID_BLOCK_SIZE={config.block_size}",
        f"-DBE_HYBRID_FIXED_SCATTER_BLOCKS={config.fixed_scatter_blocks}",
        f"-DBE_HYBRID_TPR_THRESHOLD={config.tpr_threshold}",
        f"-DBE_HYBRID_TASK_NNZ={config.task_nnz}",
    ]


def module_suffix_for_config(config: HybridConfig) -> str:
    """Build the configuration-dependent CUDA module-name suffix.

    Appended to the FFI module ``name=`` so two processes resolving different configs
    register distinct FFI targets instead of clobbering one another.

    Parameters
    ----------
    config : HybridConfig
        Scheduler configuration to encode.

    Returns
    -------
    str
        Stable suffix containing all scheduler fields.

    Raises
    ------
    ValueError
        If the scheduler configuration is invalid.
    """
    config = validate_config(config)
    return (
        f"_b{config.block_size}"
        f"_s{config.fixed_scatter_blocks}"
        f"_t{config.tpr_threshold}"
        f"_n{config.task_nnz}"
    )


def current_device_kind() -> str:
    """Resolve the first JAX device kind.

    Returns
    -------
    str
        First device's ``device_kind``, or an empty string when unavailable.
    """
    try:
        import jax

        return str(jax.devices()[0].device_kind)
    except Exception:
        return ""


def _config_path() -> Path:
    return Path(get_cache_dir()) / CONFIG_FILENAME


def _config_from_mapping(data: dict) -> HybridConfig:
    records = tuple(
        _benchmark_record_from_mapping(record)
        for record in data.get("benchmark_records", ())
    )
    return validate_config(HybridConfig(
        **{k: int(data[k]) for k in _CONFIG_FIELDS},
        benchmark_records=records,
    ))


def _config_to_mapping(config: HybridConfig) -> dict:
    return {field_name: int(getattr(config, field_name)) for field_name in _CONFIG_FIELDS}


def _benchmark_record_from_mapping(data: Mapping) -> HybridBenchmarkRecord:
    return HybridBenchmarkRecord(
        config=_config_from_mapping(dict(data["config"])),
        elapsed_ms=float(data["elapsed_ms"]),
        per_call_us=float(data["per_call_us"]),
        task_capacity=None if data.get("task_capacity") is None else int(data["task_capacity"]),
    )


def _benchmark_record_to_mapping(record) -> dict:
    if isinstance(record, HybridBenchmarkRecord):
        config = record.config
        elapsed_ms = record.elapsed_ms
        per_call_us = record.per_call_us
        task_capacity = record.task_capacity
    else:
        config = record["config"]
        elapsed_ms = record["elapsed_ms"]
        per_call_us = record["per_call_us"]
        task_capacity = record.get("task_capacity")
    return {
        "config": _config_to_mapping(validate_config(config)),
        "elapsed_ms": float(elapsed_ms),
        "per_call_us": float(per_call_us),
        "task_capacity": None if task_capacity is None else int(task_capacity),
    }


@functools.cache
def get_hybrid_config() -> HybridConfig:
    """Resolve and cache the hybrid configuration for this process.

    Never runs the benchmark; initialization is explicit via
    :func:`init_csr_config`. Falls back to
    :data:`DEFAULT_HYBRID_CONFIG` when nothing is configured, so it is safe to call on
    a machine without a GPU.

    Returns
    -------
    HybridConfig
        Environment override, persisted device configuration, or default
        configuration, in precedence order.
    """
    raw = os.environ.get(_ENV_OVERRIDE)
    if raw:
        return _config_from_mapping(json.loads(raw))

    path = _config_path()
    if path.exists():
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
            entry = store.get(current_device_kind())
            if entry is not None:
                return _config_from_mapping(entry)
        except (OSError, ValueError, KeyError):
            # A corrupt/partial file must never break kernel loading — use defaults.
            pass

    return DEFAULT_HYBRID_CONFIG


def save_hybrid_config(
    config: HybridConfig,
    device_kind: str | None = None,
    benchmark_records: Sequence[HybridBenchmarkRecord | Mapping] | None = None,
) -> Path:
    """Persist a device configuration and clear the process lookup cache.

    Updates (rather than replaces) the per-GPU JSON store so tuning one device does not
    drop entries for others.

    Parameters
    ----------
    config : HybridConfig
        Scheduler configuration to persist.
    device_kind : str or None, optional
        Device key. The first JAX device kind is used when omitted.
    benchmark_records : sequence of HybridBenchmarkRecord or Mapping or None, optional
        Benchmark records stored with the selected configuration. Records from
        ``config`` are used when omitted.

    Returns
    -------
    pathlib.Path
        Path of the updated JSON configuration store.

    Raises
    ------
    RuntimeError
        If a nonempty device kind cannot be resolved.
    ValueError
        If the scheduler configuration is invalid.
    """
    config = validate_config(config)
    if device_kind is None:
        device_kind = current_device_kind()
    if not device_kind:
        raise RuntimeError("cannot determine device_kind; pass it explicitly")

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    store: dict = {}
    if path.exists():
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            store = {}
    entry = _config_to_mapping(config)
    if benchmark_records is None:
        benchmark_records = config.benchmark_records
    if benchmark_records:
        entry["benchmark_records"] = [
            _benchmark_record_to_mapping(record)
            for record in benchmark_records
        ]
    store[device_kind] = entry

    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)

    get_hybrid_config.cache_clear()
    return path


def hybrid_task_capacity(indptr) -> int:
    """Calculate task-queue capacity using the resolved hybrid configuration.

    Must match the ``kTprThreshold`` / ``kTaskNnz`` compiled into the ``.so`` — both
    read from :func:`get_hybrid_config`, so they cannot drift.

    Parameters
    ----------
    indptr : array-like
        One-dimensional CSR row pointers.

    Returns
    -------
    int
        Required number of hybrid task slots.

    Raises
    ------
    ValueError
        If the row pointers are empty, multidimensional, decreasing, or require
        more than the int32 task-capacity limit.
    """
    config = get_hybrid_config()
    import jax

    indptr_np = np.asarray(jax.device_get(indptr), dtype=np.int64)
    if indptr_np.ndim != 1:
        raise ValueError(f"indptr must be one-dimensional, got shape={indptr_np.shape}.")
    if indptr_np.size == 0:
        raise ValueError("indptr must contain at least one element.")
    row_lengths = np.diff(indptr_np)
    if np.any(row_lengths < 0):
        raise ValueError("CSR row lengths must be non-negative.")

    chunks = np.where(
        row_lengths > config.tpr_threshold,
        (row_lengths + config.task_nnz - 1) // config.task_nnz,
        0,
    )
    task_capacity = int(chunks.sum())
    if task_capacity > _INT32_MAX:
        raise ValueError("binary task capacity exceeds int32 range.")
    return task_capacity


_CUDA_SOURCE = Path(__file__).parent / "binary_csrmv_hybrid.cu"
_MODULE_BASE = "csr_binary_csrmv_hybrid_tune"
_TARGET_FUNCTION = "binary_csrmv_wat_hybrid_homo_f32_bool"
_N_BASE = 1000


DEFAULT_CANDIDATES: tuple[HybridConfig, ...] = (
    HybridConfig(128, 1024, 128, 2048),
    HybridConfig(128, 2048, 128, 4096),
    HybridConfig(256, 1024, 128, 4096),
    HybridConfig(256, 2048, 128, 4096),
    HybridConfig(256, 4096, 128, 4096),
    HybridConfig(512, 2048, 128, 4096),
    HybridConfig(512, 4096, 128, 4096),
    HybridConfig(256, 2048, 256, 4096),
    HybridConfig(128, 2048, 512, 2048),
    HybridConfig(256, 1024, 512, 2048),
    HybridConfig(256, 2048, 512, 2048),
    HybridConfig(512, 2048, 512, 2048),
    HybridConfig(256, 2048, 512, 4096),
    HybridConfig(256, 2048, 1024, 4096),
    HybridConfig(256, 4096, 128, 1024),
    HybridConfig(256, 2048, 128, 8192),
)


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


def _task_capacity_for_uniform_rows(
    *,
    n_rows: int,
    row_conn: int,
    config: HybridConfig,
) -> int:
    if row_conn <= config.tpr_threshold:
        return 0
    chunks_per_row = (row_conn + config.task_nnz - 1) // config.task_nnz
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
    indptr_np = (
        np.arange(n_pre + 1, dtype=offset_dtype) * offset_dtype(conn)
    ).astype(offset_dtype, copy=False)
    return indices_np, indptr_np


def _make_spike_batch(
    *,
    batch_size: int,
    n_pre: int,
    spike_sparsity: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 <= spike_sparsity <= 1.0:
        raise ValueError("spike_sparsity must be in [0, 1]")
    active_count = min(max(int(round(n_pre * spike_sparsity)), 0), n_pre)
    spikes: np.ndarray = np.zeros((batch_size, n_pre), dtype=np.bool_)
    if active_count == 0:
        return spikes
    rng = np.random.default_rng(seed)
    for batch_index in range(batch_size):
        spikes[
            batch_index,
            rng.choice(n_pre, size=active_count, replace=False),
        ] = True
    return spikes


def _benchmark_config(
    config: HybridConfig,
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

    validate_config(config)
    module_name = _MODULE_BASE + module_suffix_for_config(config)
    module = load_cuda_file(
        _CUDA_SOURCE,
        name=module_name,
        extra_cuda_cflags=compile_flags_for_config(config),
        force_rebuild=force_rebuild,
        verbose=verbose_compile,
        allow_cuda_graph=False,
    )
    if _TARGET_FUNCTION not in module.function_names:
        raise RuntimeError(
            f"{_TARGET_FUNCTION} was not registered by {module.path}"
        )
    batch_size = int(batch_size)
    steps = int(steps)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")

    task_capacity = _task_capacity_for_uniform_rows(
        n_rows=n_pre,
        row_conn=conn,
        config=config,
    )
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
            target_name,
            outs,
            input_output_aliases={4: 1, 5: 2, 6: 3},
        )(
            weights,
            indices,
            indptr,
            vector,
            task_begin,
            task_end,
            status,
            task_capacity=task_capacity,
        )

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
        raise RuntimeError(
            f"CUDA hybrid task queue overflowed for config {config}"
        )

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
    candidates: Sequence[HybridConfig] = DEFAULT_CANDIDATES,
    force_rebuild: bool = False,
    verbose_compile: bool = False,
    show_progress: bool = True,
) -> list[dict]:
    """Benchmark hybrid scheduler candidates on a synthetic CSR workload.

    Parameters
    ----------
    scale : float, optional
        Multiplier applied to the base neuron count when explicit dimensions
        are omitted.
    conn : int, optional
        Number of generated connections per input row.
    n_pre : int or None, optional
        Explicit input-row count. Derived from ``scale`` when omitted.
    n_post : int or None, optional
        Explicit output-column count. Derived from ``scale`` when omitted.
    batch_size : int, optional
        Number of spike vectors materialized and reused by timed calls.
    steps : int, optional
        Number of timed kernel calls per candidate.
    spike_sparsity : float, optional
        Fraction of active entries in each generated spike vector.
    seed : int, optional
        NumPy random seed for structure and event generation.
    warmup : int, optional
        Number of untimed calls before measurement.
    candidates : sequence of HybridConfig, optional
        Scheduler configurations to benchmark.
    force_rebuild : bool, optional
        Whether to force rebuilding each CUDA module.
    verbose_compile : bool, optional
        Whether to print CUDA compiler output.
    show_progress : bool, optional
        Whether to write aggregate progress to standard error.

    Returns
    -------
    list of dict
        Benchmark records sorted by increasing elapsed time.

    Raises
    ------
    ValueError
        If dimensions, connection count, batch size, steps, sparsity, or a
        candidate configuration is invalid.
    RuntimeError
        If CUDA compilation, registration, execution, or synchronization
        fails.
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

    candidates = tuple(validate_config(candidate) for candidate in candidates)
    indices_np, indptr_np = _make_uniform_csr(
        n_pre=n_pre,
        n_post=n_post,
        conn=conn,
        seed=seed,
    )
    spikes_np = _make_spike_batch(
        batch_size=batch_size,
        n_pre=n_pre,
        spike_sparsity=spike_sparsity,
        seed=seed + 1,
    )

    weights = jnp.asarray([1.0], dtype=jnp.float32)
    indices = jnp.asarray(indices_np, dtype=jnp.int32)
    indptr = jnp.asarray(indptr_np, dtype=indptr_np.dtype)
    spikes = jnp.asarray(spikes_np, dtype=jnp.bool_)
    jax.block_until_ready((weights, indices, indptr, spikes))

    records = []
    total = len(candidates)
    if show_progress and total:
        _write_progress(0, total)
    for index, config in enumerate(candidates, start=1):
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
            _write_progress(index, total)
    records.sort(key=lambda record: record["elapsed_ms"])
    return records


def init_csr_config(
    *,
    save: bool = True,
    device_kind: str | None = None,
    candidates: Sequence[HybridConfig] = DEFAULT_CANDIDATES,
    **benchmark_kwargs,
) -> HybridConfig:
    """Benchmark and initialize the hybrid scheduler configuration.

    Parameters
    ----------
    save : bool, optional
        Whether to persist the winning configuration.
    device_kind : str or None, optional
        Device key used for persistence. The current JAX device kind is used
        when omitted.
    candidates : sequence of HybridConfig, optional
        Scheduler configurations passed to :func:`run_benchmark`.
    **benchmark_kwargs
        Additional keyword arguments passed to :func:`run_benchmark`.

    Returns
    -------
    HybridConfig
        Winning scheduler configuration.

    Raises
    ------
    RuntimeError
        If no candidate configurations are benchmarked or the configuration
        cannot be persisted.
    ValueError
        If benchmark arguments or candidate configurations are invalid.
    """
    records = run_benchmark(candidates=candidates, **benchmark_kwargs)
    if not records:
        raise RuntimeError("no candidate configs were benchmarked")
    best = records[0]["config"]
    if save:
        save_hybrid_config(
            best,
            device_kind or current_device_kind(),
            benchmark_records=records,
        )
    return best
