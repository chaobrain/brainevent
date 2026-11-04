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

"""Load and save the persisted portion of :class:`CSRConfig`.

The existing JSON filename, environment variable and per-device schema remain
unchanged. Structure plans and device properties belong to in-memory snapshots.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from brainevent._op import get_cache_dir
from ..config import _BenchmarkRecord, CSRConfig

__all__ = ['load_csr_config', 'save_csr_config']

CONFIG_FILENAME = 'csr_hybrid_config.json'
_ENV_OVERRIDE = 'BRAINEVENT_CSR_HYBRID_CONFIG'
_CONFIG_FIELDS = ('block_size', 'fixed_scatter_blocks', 'tpr_threshold', 'task_nnz')


def _current_device_kind() -> str:
    try:
        import jax
        return str(jax.devices()[0].device_kind)
    except Exception:
        return ''


def _config_path() -> Path:
    return Path(get_cache_dir()) / CONFIG_FILENAME


def _config_from_mapping(data: Mapping[str, Any]) -> CSRConfig:
    records = tuple(_record_from_mapping(record) for record in data.get('benchmark_records', ()))
    return CSRConfig(int(data['block_size']), int(data['fixed_scatter_blocks']),
                     int(data['tpr_threshold']), int(data['task_nnz']), benchmark_records=records)


def _config_to_mapping(config: CSRConfig) -> dict[str, int]:
    return {key: int(getattr(config, key)) for key in _CONFIG_FIELDS}


def _record_from_mapping(data: Mapping[str, Any]) -> _BenchmarkRecord:
    return _BenchmarkRecord(
        config=_config_from_mapping(data['config']), elapsed_ms=float(data['elapsed_ms']),
        per_call_us=float(data['per_call_us']),
        task_capacity=None if data.get('task_capacity') is None else int(data['task_capacity']),
    )


def _record_to_mapping(record: _BenchmarkRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, _BenchmarkRecord):
        config, elapsed_ms, per_call_us, capacity = (
            record.config, record.elapsed_ms, record.per_call_us, record.task_capacity)
    else:
        config, elapsed_ms, per_call_us, capacity = (
            record['config'], record['elapsed_ms'], record['per_call_us'], record.get('task_capacity'))
    return {'config': _config_to_mapping(config), 'elapsed_ms': float(elapsed_ms),
            'per_call_us': float(per_call_us),
            'task_capacity': None if capacity is None else int(capacity)}


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
        return _config_from_mapping(json.loads(raw))
    path = _config_path()
    if path.exists():
        try:
            store = json.loads(path.read_text(encoding='utf-8'))
            entry = store.get(_current_device_kind() if device_kind is None else device_kind)
            if entry is not None:
                return _config_from_mapping(entry)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
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
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    store: dict[str, Any] = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(old, dict):
                store = old
        except ValueError:
            pass
    entry: dict[str, Any] = dict(_config_to_mapping(config))
    records = config.benchmark_records if benchmark_records is None else benchmark_records
    if records:
        entry['benchmark_records'] = [_record_to_mapping(record) for record in records]
    store[device_kind] = entry
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(store, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)
    return path
