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

"""Single source of truth for the CSR *hybrid* CUDA scheduler tuning constants.

The four hybrid kernels (``binary_csrmv_hybrid.cu``, ``binary_csrmm_hybrid.cu``,
``binary_indexed_csrmv_hybrid.cu``, ``binary_indexed_csrmm_hybrid.cu``) expose four
scheduler constants via ``-DBE_HYBRID_*`` compile-time macros:

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

The best values are GPU-specific.  Users initialize them once with
:func:`brainevent._csr.initialize.init_csr_config` (manual, GPU-only); the winner is persisted per GPU
model in ``<cache_dir>/csr_hybrid_config.json`` — the same directory family as the
compiled ``.so`` (see :func:`brainevent._op.get_cache_dir`).  Resolution order in
:func:`get_hybrid_config`:

1. ``$BRAINEVENT_CSR_HYBRID_CONFIG`` — a JSON object (CI / one-off override).
2. The per-``device_kind`` entry in ``csr_hybrid_config.json``.
3. :data:`DEFAULT_HYBRID_CONFIG` — the values baked into the ``.cu`` defaults.
"""

from __future__ import annotations

import functools
import json
import os
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
]

CONFIG_FILENAME = "csr_hybrid_config.json"
_ENV_OVERRIDE = "BRAINEVENT_CSR_HYBRID_CONFIG"
_INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass(frozen=True)
class HybridConfig:
    """The four tunable scheduler constants shared by all hybrid CSR kernels."""

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
    """Timing result for one tested hybrid scheduler config."""

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
    """Validate a :class:`HybridConfig`; mirrors the ``.cu`` ``static_assert``s."""
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
    """``-DBE_HYBRID_*`` nvcc flags that bake *config* into the compiled ``.so``."""
    config = validate_config(config)
    return [
        f"-DBE_HYBRID_BLOCK_SIZE={config.block_size}",
        f"-DBE_HYBRID_FIXED_SCATTER_BLOCKS={config.fixed_scatter_blocks}",
        f"-DBE_HYBRID_TPR_THRESHOLD={config.tpr_threshold}",
        f"-DBE_HYBRID_TASK_NNZ={config.task_nnz}",
    ]


def module_suffix_for_config(config: HybridConfig) -> str:
    """Config-dependent module-name suffix.

    Appended to the FFI module ``name=`` so two processes resolving different configs
    register distinct FFI targets instead of clobbering one another.
    """
    config = validate_config(config)
    return (
        f"_b{config.block_size}"
        f"_s{config.fixed_scatter_blocks}"
        f"_t{config.tpr_threshold}"
        f"_n{config.task_nnz}"
    )


def current_device_kind() -> str:
    """Return the first JAX device's ``device_kind`` (``""`` if unavailable)."""
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
    """Resolve the hybrid config for this process (memoized — read once).

    Never runs the benchmark; initialization is explicit via
    :func:`brainevent._csr.initialize.init_csr_config`.  Falls back to
    :data:`DEFAULT_HYBRID_CONFIG` when nothing is configured, so it is safe to call on
    a machine without a GPU.
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
    """Persist *config* for *device_kind* (default: current GPU) and clear the cache.

    Updates (rather than replaces) the per-GPU JSON store so tuning one device does not
    drop entries for others.  Returns the config file path.
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
    """Task-queue capacity for *indptr*, using the resolved hybrid config.

    Must match the ``kTprThreshold`` / ``kTaskNnz`` compiled into the ``.so`` — both
    read from :func:`get_hybrid_config`, so they cannot drift.
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
