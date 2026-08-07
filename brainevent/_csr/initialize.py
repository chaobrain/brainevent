
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
simulation); the winning :class:`~brainevent._csr.hybrid_config.HybridConfig` is persisted
per GPU model via :func:`~brainevent._csr.hybrid_config.save_hybrid_config`, and every later
process picks it up through :func:`~brainevent._csr.hybrid_config.get_hybrid_config`.

The benchmark compiles the **production** ``binary_csrmv_hybrid.cu`` (with per-config
``-DBE_HYBRID_*`` flags) — so the timed kernel is exactly the one used at run time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .hybrid_config import (
    HybridConfig,
    compile_flags_for_config,
    current_device_kind,
    module_suffix_for_config,
    save_hybrid_config,
    validate_config,
)

_CUDA_SOURCE = Path(__file__).parent / "binary_csrmv_hybrid.cu"
_MODULE_BASE = "csr_binary_csrmv_hybrid_tune"
_TARGET_FUNCTION = "binary_csrmv_wat_hybrid_homo_f32_bool"
_INT32_MAX = int(np.iinfo(np.int32).max)

#: Neurons per unit ``scale`` for the tuning workload: ``n_pre = n_post = _N_BASE * scale``.
#: dev/"COBA EI benchmark.py" uses ``_N=4000``; tuning uses a lighter base to keep the
#: default sweep's memory footprint modest.
_N_BASE = 1000

#: Candidate sweep (block, fixed_scatter_blocks, tpr_threshold, task_nnz).
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


def _task_capacity_for_uniform_rows(*, n_rows: int, row_conn: int, config: HybridConfig) -> int:
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
    indptr_np = (np.arange(n_pre + 1, dtype=offset_dtype) * offset_dtype(conn)).astype(offset_dtype, copy=False)
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
    candidates: Sequence[HybridConfig] = DEFAULT_CANDIDATES,
    force_rebuild: bool = False,
    verbose_compile: bool = False,
    show_progress: bool = True,
) -> list[dict]:
    """Time every candidate on a synthetic COBA-like CSR; return records sorted best-first.

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

    candidates = tuple(validate_config(c) for c in candidates)
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
    candidates: Sequence[HybridConfig] = DEFAULT_CANDIDATES,
    **benchmark_kwargs,
) -> HybridConfig:
    """Initialize the per-GPU CSR config: benchmark candidates and (by default) persist the winner.

    Run this **once** manually before simulating; later processes read the saved config via
    :func:`~brainevent._csr.hybrid_config.get_hybrid_config`.  Returns the winning
    :class:`~brainevent._csr.hybrid_config.HybridConfig`.
    """
    records = run_benchmark(candidates=candidates, **benchmark_kwargs)
    if not records:
        raise RuntimeError("no candidate configs were benchmarked")
    best = records[0]["config"]
    if save:
        save_hybrid_config(best, device_kind or current_device_kind(), benchmark_records=records)
    return best
