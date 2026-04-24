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

#
# Unified VRAM-limit progressive benchmark for COBA 2005 / COBA EI (fcnmv).
#
# Starts from a small VRAM budget and increases step by step using boundary
# candidate points from BenchmarkTools.
# - When JAX emits memory-related warnings  -> tag ``message = JAX_warning``
# - When RESOURCE_EXHAUSTED is raised       -> tag ``message = Device_error``
#

from __future__ import annotations

import gc
import importlib.util
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, cast

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import brainunit as u
import jax

import brainevent

backends = ['cuda_raw']
data_size = 4

_BENCHMARK_FILES = {
    'coba_2005': 'COBA_2005_benchmark.py',
    'coba_ei': 'COBA EI benchmark.py',
}
_VALID_TARGETS = ('pre', 'post')
_VALID_LAYOUTS = ('col_scatter', 'row_gather', 'auto')

CompiledRun = Callable[[], tuple[int, Any]]
RunFactory = Callable[..., CompiledRun]


def _load_make_simulation_run(benchmark_name: str) -> RunFactory:
    if benchmark_name not in _BENCHMARK_FILES:
        raise ValueError(
            f'benchmark_name must be one of {tuple(_BENCHMARK_FILES)}, '
            f'got {benchmark_name!r}.'
        )

    benchmark_path = Path(__file__).with_name(_BENCHMARK_FILES[benchmark_name])
    spec = importlib.util.spec_from_file_location(f'vram_limit_{benchmark_name}', benchmark_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load benchmark module from {benchmark_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    make_simulation_run = getattr(module, 'make_simulation_run', None)
    if make_simulation_run is None:
        raise AttributeError(f'{benchmark_path} does not define make_simulation_run().')
    return cast(RunFactory, make_simulation_run)


def _run_and_block(run: CompiledRun) -> tuple[int, Any]:
    result = run()
    blocked_result = jax.block_until_ready(result)
    return cast(tuple[int, Any], blocked_result)


def _announce_runtime_platform() -> str:
    platform = jax.default_backend()
    devices = ', '.join(str(device) for device in jax.devices())
    print(f'Runtime platform: {platform}; devices: {devices}')
    return platform


def _is_oom_error(exc: Exception) -> bool:
    error_msg = str(exc).lower()
    return any(token in error_msg for token in (
        'resource_exhausted',
        'resource exhausted',
        'out of memory',
        'oom',
        'failed to allocate',
        'new constant',
    ))


def _safe_clear_runtime_cache(run: CompiledRun | None = None):
    """Best-effort cleanup between benchmark points."""
    if run is not None and hasattr(run, 'clear_cache'):
        try:
            run.clear_cache()
        except Exception:
            pass
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()


def _summarize_live_arrays(max_items: int = 20) -> list[str]:
    """Summarize live device arrays to help identify leaked buffers."""
    lines: list[str] = []
    try:
        arrays = sorted(jax.live_arrays(), key=lambda arr: arr.nbytes, reverse=True)
    except Exception as exc:
        return [f'live_arrays_unavailable: {exc}']

    total_bytes = sum(arr.nbytes for arr in arrays)
    lines.append(f'live_arrays={len(arrays)} total_bytes={total_bytes}')
    for i, arr in enumerate(arrays[:max_items], start=1):
        try:
            device = getattr(arr, 'device', None)
            device_str = str(device) if device is not None else 'unknown'
            lines.append(
                f'{i:02d}. bytes={arr.nbytes} shape={tuple(arr.shape)} '
                f'dtype={arr.dtype} device={device_str}'
            )
        except Exception as exc:
            lines.append(f'{i:02d}. summary_failed: {exc}')
    return lines


def _write_memory_debug_snapshot(
    *,
    case_name: str,
    output_dir: Path,
    include_device_profile: bool,
):
    """Persist best-effort memory diagnostics for a benchmark point."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / f'{case_name}_memory.txt'
    profile_path = output_dir / f'{case_name}.prof'

    lines: list[str] = []
    try:
        device = jax.devices()[0]
        lines.append(f'device={device}')
        try:
            stats = device.memory_stats()
            for key in sorted(stats):
                lines.append(f'{key}={stats[key]}')
        except Exception as exc:
            lines.append(f'memory_stats_failed={exc}')
    except Exception as exc:
        lines.append(f'device_query_failed={exc}')

    lines.append('')
    lines.extend(_summarize_live_arrays())
    stats_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    if include_device_profile:
        try:
            jax.profiler.save_device_memory_profile(str(profile_path))
        except Exception as exc:
            profile_path.with_suffix('.prof.error.txt').write_text(str(exc) + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Logging handler to capture JAX / XLA memory warnings
# ---------------------------------------------------------------------------
class _WarningCollector(logging.Handler):
    """Temporary logging handler that collects warning-level records."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)

    def has_memory_warning(self) -> bool:
        keywords = ('memory', 'allocation', 'resource', 'oom', 'exceeds')
        for record in self.records:
            msg = record.getMessage().lower()
            if any(keyword in msg for keyword in keywords):
                return True
        return False

    def clear(self):
        self.records.clear()


def benchmark_vram_limit(
    benchmark_name: str = 'coba_2005',
    data_type: str = 'binary',
    duration=1e2 * u.ms,
    homo: bool = True,
    backend: str | None = None,
    efferent_target: str = 'post',
    _N: int = 4000,
    vram_start: int = 4,
    vram_end: int = 24,
    vram_step: int = 1,
    sample_points: int = 3,
    scale_max: int = 2000,
    conn_max: int = 4000,
    mv_layout: str = 'row_gather',
    clear_runtime_cache_between_cases: bool = True,
    save_memory_profile_on_error: bool = True,
    debug_output_dir: str = 'result-vram-limit-debug',
):
    """Progressive VRAM-limit benchmark driven by boundary candidate points."""
    import dev.fcn.BenchmarkTools as BT

    if efferent_target not in _VALID_TARGETS:
        raise ValueError(f'efferent_target must be one of {_VALID_TARGETS}, got {efferent_target!r}.')
    if mv_layout not in _VALID_LAYOUTS:
        raise ValueError(f'mv_layout must be one of {_VALID_LAYOUTS}, got {mv_layout!r}.')
    if vram_step <= 0:
        raise ValueError(f'vram_step must be > 0, got {vram_step}.')
    if vram_start > vram_end:
        raise ValueError(f'vram_start must be <= vram_end, got {vram_start} > {vram_end}.')

    print(f'=== VRAM-limit progressive benchmark ({benchmark_name}) ===')

    make_simulation_run = _load_make_simulation_run(benchmark_name)
    backends_to_use = [backend] if backend is not None else backends
    runtime_platform = _announce_runtime_platform()
    vram_steps = list(range(vram_start, vram_end + 1, vram_step))

    generator = BT.TestingParamsGenerator_mv(
        limit_GB=vram_end,
        _N=_N,
        scale_max=scale_max,
        conn_max=conn_max,
    )
    vram_params = generator.generate_coba_vram_sequence(
        vram_steps=vram_steps,
        sample_points=sample_points,
        homo=homo,
        data_size=data_size,
    )
    if not vram_params:
        raise ValueError(
            f'No boundary candidate states generated for benchmark_name={benchmark_name!r}, '
            f'vram range {vram_start}..{vram_end} GB.'
        )

    log_collector = _WarningCollector()
    jax_logger = logging.getLogger('jax')
    xla_logger = logging.getLogger('jaxlib')
    root_logger = logging.getLogger()
    for logger in (jax_logger, xla_logger, root_logger):
        logger.addHandler(log_collector)

    try:
        for back in backends_to_use:
            brainevent.config.set_backend(runtime_platform, back)
            csv_recorder = BT.CSV_record(
                f'vram_limit_{benchmark_name}_{efferent_target}',
                'fcnmv',
                'coba',
                duration=duration,
            )
            csv_recorder.add_tag('benchmark_name', benchmark_name)
            csv_recorder.add_tag('limit_GB', vram_end)
            csv_recorder.add_tag('mv_layout', mv_layout)

            csv_recorder.print_header(
                operator='fcnmv',
                data_type=data_type,
                backend=back,
                mode=efferent_target,
                duration=duration,
                homo=('homo' if homo else 'hetero'),
                benchmark_name=benchmark_name,
                mv_layout=mv_layout,
                limit_GB=vram_end,
            )
            csv_recorder.print_table_header(show_conn=True)

            device_error_hit = False

            for vram_gb, pairs in vram_params.items():
                if device_error_hit:
                    break

                print(f'\n--- Testing VRAM budget: {vram_gb} GB ({len(pairs)} parameter pairs) ---')

                for scale, conn in pairs:
                    if device_error_hit:
                        break

                    csv_recorder.add_tag('current_VRAM', vram_gb)
                    log_collector.clear()
                    run: CompiledRun | None = None
                    case_name = (
                        f'{benchmark_name}_{back}_{data_type}_{efferent_target}_{mv_layout}_'
                        f'vram{vram_gb}_scale{scale}_conn{conn}'
                    )

                    try:
                        with warnings.catch_warnings(record=True) as py_warnings:
                            warnings.simplefilter('always')

                            run = cast(CompiledRun, make_simulation_run(
                                scale=scale,
                                data_type=data_type,
                                efferent_target=efferent_target,
                                duration=duration,
                                conn_num=conn,
                                homo=homo,
                                mv_layout=mv_layout,
                            ))

                            _run_and_block(run)

                            t0 = time.time()
                            n, rate = _run_and_block(run)
                            t1 = time.time()
                            elapsed = t1 - t0

                        jax_warned = log_collector.has_memory_warning()
                        py_mem_warned = any(
                            any(keyword in str(w.message).lower() for keyword in ('memory', 'allocation', 'resource'))
                            for w in py_warnings
                        )
                        message = 'JAX_warning' if (jax_warned or py_mem_warned) else 'OK'
                        csv_recorder.add_tag('message', message)

                        if message == 'JAX_warning':
                            print(f'  [JAX_warning] scale={scale}, conn={conn}, VRAM={vram_gb}GB')

                        csv_recorder.print_row(scale, n, elapsed, float(rate), conn_num=conn)
                        csv_recorder.single_COBA_data_add(
                            'fcnmv',
                            data_type,
                            back,
                            efferent_target,
                            conn,
                            scale,
                            elapsed,
                            float(rate),
                            duration,
                            homo=('homo' if homo else 'hetero'),
                            benchmark_name=benchmark_name,
                            limit_GB=vram_end,
                            current_VRAM=vram_gb,
                            mv_layout=mv_layout,
                            message=message,
                        )
                    except Exception as exc:
                        if save_memory_profile_on_error:
                            _write_memory_debug_snapshot(
                                case_name=case_name,
                                output_dir=Path(__file__).resolve().parent / debug_output_dir,
                                include_device_profile=True,
                            )
                        if _is_oom_error(exc):
                            csv_recorder.add_tag('message', 'Device_error')
                            csv_recorder.single_COBA_data_add(
                                'fcnmv',
                                data_type,
                                back,
                                efferent_target,
                                conn,
                                scale,
                                -1,
                                -1,
                                duration,
                                homo=('homo' if homo else 'hetero'),
                                benchmark_name=benchmark_name,
                                limit_GB=vram_end,
                                current_VRAM=vram_gb,
                                mv_layout=mv_layout,
                                message='Device_error',
                            )
                            print(
                                f'  [Device_error] RESOURCE_EXHAUSTED at '
                                f'scale={scale}, conn={conn}, VRAM={vram_gb}GB'
                            )
                            print('  Stopping test.')
                            device_error_hit = True
                            break

                        print(f'  [Error] scale={scale}, conn={conn}: {exc}')
                        continue
                    finally:
                        if clear_runtime_cache_between_cases:
                            _safe_clear_runtime_cache(run)

            homo_str = 'homo' if homo else 'hetero'
            output_name = (
                f'vram_limit_{benchmark_name}_{data_type}_{homo_str}_'
                f'{back}_{efferent_target}_{mv_layout}'
            )
            output_path = csv_recorder.flush_and_clear(output_name, dir='result-vram-limit')
            if output_path is not None:
                print(f'Results saved to: {output_path}')
    finally:
        for logger in (jax_logger, xla_logger, root_logger):
            logger.removeHandler(log_collector)


if __name__ == '__main__':
    benchmark_vram_limit(
        benchmark_name='coba_2005',#coba_2005 coba_ei
        data_type='compact',
        duration=10 * u.ms,
        efferent_target='pre',
        homo=True,
        backend='cuda_raw',
        mv_layout='row_gather',#row_gather col_scatter
    )
