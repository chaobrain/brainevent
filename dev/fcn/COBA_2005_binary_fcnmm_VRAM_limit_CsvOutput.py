# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

"""
VRAM boundary benchmark for COBA 2005 (binary_fcnmm operator).

This version connects:
1. BT.TestingParamsGenerator_mm.generate_params_steps(...)
2. the simulation / operator run path
3. CSV recording of both theoretical memory estimate and actual runtime outcome

Recommended use for memory-boundary probing:
    backend='cuda_fake'
"""

import gc
import sys
import time
import warnings
import logging
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import brainunit as u
import jax

import brainevent
from COBA_2005_benchmark import make_simulation_batch_run

backends = ['cuda_fake']
DEFAULT_DATA_SIZE = 4


class _WarningCollector(logging.Handler):
    """Temporary logging handler that collects warning-level records."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def has_memory_warning(self) -> bool:
        keywords = ('memory', 'allocation', 'resource', 'oom', 'exceeds')
        for r in self.records:
            msg = r.getMessage().lower()
            if any(k in msg for k in keywords):
                return True
        return False

    def clear(self):
        self.records.clear()


def _estimate_mm_bytes(scale: int, batch_size: int, conn: int, _N: int, homo: bool, data_size: int) -> int:
    """
    Match the current static memory model used by TestingParamsGenerator_mm.is_valid_mm:
        size = scale * _N
        mem_elems = conn * size * times + batch_size * size + size * batch_size
        mem_bytes = mem_elems * data_size
    """
    times = 1 if homo else 2
    size = scale * _N
    mem_elems = conn * size * times + batch_size * size + size * batch_size
    return int(mem_elems * data_size)


def _safe_clear_runtime_cache():
    """Best-effort cleanup between test points."""
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()


def benchmark_vram_limit(
    data_type: str = 'binary',
    duration=1e2 * u.ms,
    homo: bool = True,
    backend: str | None = None,
    efferent_target: str = 'pre',
    _N: int = 4000,
    vram_limit: int = 23,
    sample_points: int = 100,
    scale_max: int = 1000,
    conn_max: int = 1000,
    batch_max: int = 1000,
    data_size: int = DEFAULT_DATA_SIZE,
    stop_on_oom: bool = False,
    do_warmup: bool = True,
):
    """
    VRAM-boundary benchmark for the fcnmm operator.

    Workflow
    --------
    1. Build candidate (scale, batch_size, conn) points with
       BT.TestingParamsGenerator_mm.generate_params_steps(...).
    2. Run the operator path through make_simulation_batch_run(...)
       using the selected backend (recommended: cuda_fake).
    3. Record per-point outcome into CSV.

    Notes
    -----
    - The generated parameter points are filtered by the BT-side static memory model.
    - Actual runtime may still show JAX warnings or OOM due to allocator/runtime overhead.
    """

    import dev.fcn.BenchmarkTools as BT

    print('=== VRAM boundary benchmark (fcnmm) ===')

    backends_to_use = [backend] if backend is not None else backends

    generator = BT.TestingParamsGenerator_mm(
        limit_GB=vram_limit,
        _N=_N,
        scale_max=scale_max,
        conn_max=conn_max,
        batch_max=batch_max,
    )

    # Hook into the BT generator the user is building.
    vram_params = generator.generate_params_steps(
    target_points=100,
    step_ratio=(1.0, 1.0, 4.0),
    data_size=data_size,
    homo=homo,
    )

    print(f'Generated {len(vram_params)} candidate points from BT generator.')

    csv_recorder = BT.CSV_record(
        f'vram_limit_{efferent_target}',
        'fcnmm',
        'coba',
        duration=duration,
    )

    log_collector = _WarningCollector()
    jax_logger = logging.getLogger('jax')
    xla_logger = logging.getLogger('jaxlib')
    root_logger = logging.getLogger()
    for lgr in (jax_logger, xla_logger, root_logger):
        lgr.addHandler(log_collector)

    homo_str = 'homo' if homo else 'hetero'
    flush_file_name = f'mm-vram_limit_{data_type}_{homo_str}_{efferent_target}'
    last_path = None

    total_points = len(vram_params)
    n_ok = 0
    n_warn = 0
    n_oom = 0
    n_error = 0

    try:
        for back in backends_to_use:
            print(f'\n--- Backend: {back} ---')
            brainevent.config.set_backend('gpu', back)

            csv_recorder.print_header(
                operator='fcnmm',
                data_type=data_type,
                backend=back,
                mode=efferent_target,
                duration=duration,
                homo=homo_str,
            )
            csv_recorder.print_table_header(show_conn=True, show_batch=True)

            for idx, (scale, batch_size, conn) in enumerate(vram_params, start=1):
                _safe_clear_runtime_cache()
                log_collector.clear()

                current_vram_bytes = _estimate_mm_bytes(
                    scale=scale,
                    batch_size=batch_size,
                    conn=conn,
                    _N=_N,
                    homo=homo,
                    data_size=data_size,
                )
                current_vram_gib = current_vram_bytes / (1024 ** 3)

                csv_recorder.add_tag('current_VRAM_GiB', f'{current_vram_gib:.6f}')
                csv_recorder.add_tag('current_VRAM_bytes', current_vram_bytes)
                csv_recorder.add_tag('limit_GB', vram_limit)
                csv_recorder.add_tag('backend_test', back)
                csv_recorder.add_tag('point_index', f'{idx}/{total_points}')
                csv_recorder.add_tag('params', f'scale={scale},batch={batch_size},conn={conn}')

                try:
                    with warnings.catch_warnings(record=True) as py_warnings:
                        warnings.simplefilter('always')

                        run = make_simulation_batch_run(
                            scale=scale,
                            batch_size=batch_size,
                            data_type=data_type,
                            efferent_target=efferent_target,
                            duration=duration,
                            conn_num=conn,
                            homo=homo,
                        )

                        if do_warmup:
                            jax.block_until_ready(run())

                        t0 = time.time()
                        out = jax.block_until_ready(run())
                        t1 = time.time()
                        elapsed = t1 - t0

                    # Keep compatibility with the original benchmark's expected return.
                    if isinstance(out, tuple) and len(out) >= 2:
                        n, rate = out[0], out[1]
                        rate_value = float(rate)
                    else:
                        n = -1
                        rate_value = -1.0

                    jax_warned = log_collector.has_memory_warning()
                    py_mem_warned = any(
                        any(k in str(w.message).lower() for k in ('memory', 'allocation', 'resource', 'oom', 'exceeds'))
                        for w in py_warnings
                    )

                    if jax_warned or py_mem_warned:
                        msg = 'JAX_warning'
                        n_warn += 1
                    else:
                        msg = 'OK'
                        n_ok += 1

                    csv_recorder.add_tag('message', msg)
                    csv_recorder.print_row(
                        scale,
                        n,
                        elapsed,
                        rate_value,
                        conn_num=conn,
                        batch_size=batch_size,
                    )
                    csv_recorder.single_COBA_data_add(
                        'fcnmm',
                        data_type,
                        back,
                        efferent_target,
                        conn,
                        scale,
                        elapsed,
                        rate_value,
                        duration,
                        homo=homo_str,
                        batch_size=batch_size,
                    )
                    last_path = csv_recorder.flush_and_clear(
                        flush_file_name,
                        dir='result-vram-limit_cuda-fake',
                    )

                    print(
                        f'[{idx:04d}/{total_points:04d}] {msg:<12} '
                        f'scale={scale:<4d} batch={batch_size:<6d} conn={conn:<6d} '
                        f'est={current_vram_gib:8.4f} GiB elapsed={elapsed:8.4f}s'
                    )

                except Exception as e:
                    error_msg = str(e).lower()
                    is_oom = any(
                        kw in error_msg for kw in (
                            'resource_exhausted',
                            'resource exhausted',
                            'out of memory',
                            'oom',
                            'failed to allocate',
                        )
                    )

                    if is_oom:
                        n_oom += 1
                        csv_recorder.add_tag('message', 'Device_error')
                    else:
                        n_error += 1
                        csv_recorder.add_tag('message', 'Error')

                    csv_recorder.add_tag('exception', str(e))
                    csv_recorder.add_tag('current_VRAM_GiB', f'{current_vram_gib:.6f}')
                    csv_recorder.add_tag('current_VRAM_bytes', current_vram_bytes)

                    csv_recorder.single_COBA_data_add(
                        'fcnmm',
                        data_type,
                        back,
                        efferent_target,
                        conn,
                        scale,
                        -1,
                        -1,
                        duration,
                        homo=homo_str,
                        batch_size=batch_size,
                    )
                    last_path = csv_recorder.flush_and_clear(
                        flush_file_name,
                        dir='result-vram-limit_cuda-fake',
                    )

                    label = 'Device_error' if is_oom else 'Error'
                    print(
                        f'[{idx:04d}/{total_points:04d}] {label:<12} '
                        f'scale={scale:<4d} batch={batch_size:<6d} conn={conn:<6d} '
                        f'est={current_vram_gib:8.4f} GiB err={e}'
                    )

                    if is_oom and stop_on_oom:
                        print('Stopping on first OOM as requested.')
                        break

            print(
                f'Backend {back} summary: '
                f'OK={n_ok}, JAX_warning={n_warn}, Device_error={n_oom}, Error={n_error}'
            )

    finally:
        for lgr in (jax_logger, xla_logger, root_logger):
            lgr.removeHandler(log_collector)

    if last_path:
        print(f'\nDone. Results saved to: {last_path}')


if __name__ == '__main__':
    benchmark_vram_limit(
        data_type='binary',
        duration=1e2 * u.ms,
        efferent_target='post',
        homo=True,
        backend='cuda_fake',
        vram_limit=6,
        sample_points=4,
        data_size=4,
        stop_on_oom=False,
    )
