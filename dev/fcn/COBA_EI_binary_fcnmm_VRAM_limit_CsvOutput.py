from __future__ import annotations

import gc
import importlib.util
import logging
import math
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, cast

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import brainunit as u
import jax

import brainevent

DEFAULT_BACKENDS = ['cuda_raw']
DEFAULT_DATA_SIZE = 4
DEFAULT_FIXED_BATCH_SIZES = (64, 128)
_BENCHMARK_FILE = 'COBA EI benchmark.py'
_BENCHMARK_NAME = 'coba_ei'
_VALID_TARGETS = ('pre', 'post')
_VALID_LAYOUTS = ('col_scatter', 'row_gather', 'auto')

CompiledRun = Callable[[], tuple[int, Any]]
RunFactory = Callable[..., CompiledRun]
BoundaryPairs = list[tuple[int, int]]
VramPlan = OrderedDict[int, OrderedDict[int, BoundaryPairs]]
CaseRecord = dict[str, Any]


def _load_make_simulation_batch_run(benchmark_path: Path | None = None) -> RunFactory:
    benchmark_file = benchmark_path or Path(__file__).with_name(_BENCHMARK_FILE)
    spec = importlib.util.spec_from_file_location('vram_limit_coba_ei_fcnmm', benchmark_file)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load benchmark module from {benchmark_file}')

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(f'Unable to load benchmark module from {benchmark_file}: {exc}') from exc

    make_simulation_batch_run = getattr(module, 'make_simulation_batch_run', None)
    if make_simulation_batch_run is None:
        raise AttributeError(f'{benchmark_file} does not define make_simulation_batch_run().')
    return cast(RunFactory, make_simulation_batch_run)


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


def _safe_clear_runtime_cache(run: CompiledRun | None = None) -> None:
    if run is not None and hasattr(run, 'release'):
        try:
            run.release()
            return
        except Exception:
            pass
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
) -> None:
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

    def clear(self) -> None:
        self.records.clear()


def _resolve_output_dir(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_batch_sizes(fixed_batch_sizes: Iterable[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_value in fixed_batch_sizes:
        batch_size = int(raw_value)
        if batch_size <= 0:
            raise ValueError(f'All fixed batch sizes must be > 0, got {batch_size}.')
        if batch_size not in seen:
            seen.add(batch_size)
            normalized.append(batch_size)
    if not normalized:
        raise ValueError('fixed_batch_sizes must contain at least one positive batch size.')
    return tuple(normalized)


def _estimate_mm_bytes(
    scale: int,
    batch_size: int,
    conn: int,
    _N: int,
    homo: bool,
    data_size: int,
) -> int:
    times = 1 if homo else 2
    size = scale * _N
    mem_elems = conn * size * times + 2 * batch_size * size
    return int(mem_elems * data_size)


def _sample_scale_points(min_scale: int, max_scale: int, sample_points: int) -> list[int]:
    import numpy as np

    if sample_points < 1:
        raise ValueError(f'sample_points must be >= 1, got {sample_points}.')
    if min_scale > max_scale:
        return []
    if sample_points == 1 or min_scale == max_scale:
        return [int(max_scale)]

    geom = np.rint(np.geomspace(min_scale, max_scale, num=sample_points)).astype(int)
    geom[0] = min_scale
    geom[-1] = max_scale
    unique = sorted({int(v) for v in geom})

    if len(unique) < sample_points:
        linear = np.rint(np.linspace(min_scale, max_scale, num=sample_points)).astype(int)
        unique = sorted({*unique, *(int(v) for v in linear)})

    if len(unique) <= sample_points:
        return unique

    selected: list[int] = []
    last_index = len(unique) - 1
    step = last_index / (sample_points - 1)
    for i in range(sample_points):
        index = last_index if i == sample_points - 1 else int(i * step)
        selected.append(unique[index])
    return sorted({int(v) for v in selected})


def _generate_mm_boundary_pairs(
    limit_gb: int,
    batch_size: int,
    sample_points_per_batch: int,
    _N: int,
    scale_max: int,
    conn_max: int,
    homo: bool,
    data_size: int,
) -> BoundaryPairs:
    if limit_gb <= 0:
        raise ValueError(f'limit_gb must be > 0, got {limit_gb}.')
    if batch_size <= 0:
        raise ValueError(f'batch_size must be > 0, got {batch_size}.')
    if sample_points_per_batch < 1:
        raise ValueError(
            f'sample_points_per_batch must be >= 1, got {sample_points_per_batch}.'
        )
    if scale_max < 1:
        raise ValueError(f'scale_max must be >= 1, got {scale_max}.')
    if conn_max < 1:
        raise ValueError(f'conn_max must be >= 1, got {conn_max}.')

    min_scale = min(scale_max, 20)
    min_conn = min(conn_max, 20)
    times = 1 if homo else 2
    limit_bytes = limit_gb * (1024 ** 3)
    budget_elems = limit_bytes / data_size

    s_min = max(
        min_scale,
        math.ceil(budget_elems / (_N * (times * conn_max + 2 * batch_size))),
    )
    s_max = min(
        scale_max,
        int(budget_elems // (_N * (times * min_conn + 2 * batch_size))),
    )
    if s_min > s_max:
        return []

    scales = _sample_scale_points(s_min, s_max, sample_points_per_batch)
    valid_pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for scale in scales:
        size = scale * _N
        sparse_budget = budget_elems - 2 * batch_size * size
        if sparse_budget <= 0:
            continue
        max_conn_by_mem = int(sparse_budget // (size * times))
        conn = min(max_conn_by_mem, conn_max, size)
        if conn < min_conn:
            continue
        pair = (int(scale), int(conn))
        if pair in seen:
            continue
        if _estimate_mm_bytes(scale, batch_size, conn, _N, homo, data_size) > limit_bytes:
            continue
        seen.add(pair)
        valid_pairs.append(pair)
    return valid_pairs


def _generate_mm_vram_sequence(
    *,
    vram_steps: Iterable[int],
    fixed_batch_sizes: Iterable[int],
    sample_points_per_batch: int,
    _N: int,
    scale_max: int,
    conn_max: int,
    homo: bool,
    data_size: int,
) -> VramPlan:
    result: VramPlan = OrderedDict()
    for vram_gb in vram_steps:
        batch_plan: OrderedDict[int, BoundaryPairs] = OrderedDict()
        for batch_size in fixed_batch_sizes:
            batch_plan[int(batch_size)] = _generate_mm_boundary_pairs(
                limit_gb=int(vram_gb),
                batch_size=int(batch_size),
                sample_points_per_batch=sample_points_per_batch,
                _N=_N,
                scale_max=scale_max,
                conn_max=conn_max,
                homo=homo,
                data_size=data_size,
            )
        result[int(vram_gb)] = batch_plan
    return result


def _count_total_candidate_points(vram_plan: VramPlan) -> int:
    return sum(len(pairs) for batch_plan in vram_plan.values() for pairs in batch_plan.values())


def _summary_key_for_message(message: str) -> str:
    mapping = {
        'OK': 'ok',
        'JAX_warning': 'jax_warning',
        'Device_error': 'device_error',
        'Error': 'error',
    }
    return mapping.get(message, 'error')


def _summarize_case_records(vram_plan: VramPlan, case_records: list[CaseRecord]):
    level_summaries: OrderedDict[int, OrderedDict[int, dict[str, Any]]] = OrderedDict()
    empty_groups = 0
    for vram_gb, batch_plan in vram_plan.items():
        batch_summaries: OrderedDict[int, dict[str, Any]] = OrderedDict()
        for batch_size, pairs in batch_plan.items():
            if not pairs:
                empty_groups += 1
            batch_summaries[batch_size] = {
                'planned_points': len(pairs),
                'run_points': 0,
                'ok': 0,
                'jax_warning': 0,
                'device_error': 0,
                'error': 0,
                'no_valid_points': len(pairs) == 0,
                'first_oom_point': None,
            }
        level_summaries[vram_gb] = batch_summaries

    summary_counts = {
        'total_cases': len(case_records),
        'ok': 0,
        'jax_warning': 0,
        'device_error': 0,
        'error': 0,
        'empty_groups': empty_groups,
    }

    for record in case_records:
        vram_gb = int(record['current_VRAM'])
        batch_size = int(record['batch_size'])
        message = str(record['message'])
        key = _summary_key_for_message(message)
        summary_counts[key] += 1
        batch_summary = level_summaries[vram_gb][batch_size]
        batch_summary['run_points'] += 1
        batch_summary[key] += 1
        if message == 'Device_error' and batch_summary['first_oom_point'] is None:
            batch_summary['first_oom_point'] = (
                f"scale={record['scale']}, conn={record['conn_num']}, point={record['point_index']}"
            )

    return summary_counts, level_summaries


def _build_txt_report(
    *,
    created_at: str,
    runtime_platform: str,
    backend: str,
    data_type: str,
    homo: bool,
    efferent_target: str,
    mv_layout: str,
    fixed_batch_sizes: tuple[int, ...],
    sample_points_per_batch: int,
    _N: int,
    vram_steps: list[int],
    scale_max: int,
    conn_max: int,
    total_candidate_points: int,
    summary_counts: dict[str, int],
    level_summaries: OrderedDict[int, OrderedDict[int, dict[str, Any]]],
    case_records: list[CaseRecord],
    csv_output_path: Path | None,
) -> str:
    lines: list[str] = [
        '# COBA EI FCNMM VRAM-limit Report',
        '',
        '[Header]',
        f'created_at={created_at}',
        f'benchmark_name={_BENCHMARK_NAME}',
        f'runtime_platform={runtime_platform}',
        f'backend={backend}',
        f'data_type={data_type}',
        f'homo={"homo" if homo else "hetero"}',
        f'efferent_target={efferent_target}',
        f'mv_layout={mv_layout}',
        f'fixed_batch_sizes={",".join(str(batch) for batch in fixed_batch_sizes)}',
        f'sample_points_per_batch={sample_points_per_batch}',
        f'_N={_N}',
        f'vram_steps={",".join(str(v) for v in vram_steps)}',
        f'scale_max={scale_max}',
        f'conn_max={conn_max}',
        f'total_candidate_points={total_candidate_points}',
        f'csv_output_path={csv_output_path if csv_output_path is not None else "not_written"}',
        '',
        '[Summary]',
        f'total_cases={summary_counts["total_cases"]}',
        f'ok={summary_counts["ok"]}',
        f'jax_warning={summary_counts["jax_warning"]}',
        f'device_error={summary_counts["device_error"]}',
        f'error={summary_counts["error"]}',
        f'empty_groups={summary_counts["empty_groups"]}',
        '',
        '[VRAM-level summary]',
    ]

    for vram_gb, batch_summaries in level_summaries.items():
        lines.append(f'VRAM {vram_gb} GB')
        for batch_size, summary in batch_summaries.items():
            if summary['no_valid_points']:
                lines.append(
                    f'  batch={batch_size}: planned=0, run=0, status=no_valid_points'
                )
                continue
            line = (
                f'  batch={batch_size}: planned={summary["planned_points"]}, '
                f'run={summary["run_points"]}, OK={summary["ok"]}, '
                f'JAX_warning={summary["jax_warning"]}, '
                f'Device_error={summary["device_error"]}, Error={summary["error"]}'
            )
            if summary['first_oom_point'] is not None:
                line += f', first_oom={summary["first_oom_point"]}'
            lines.append(line)

    lines.extend(['', '[Detailed cases]'])
    if not case_records:
        lines.append('none')
    else:
        for record in case_records:
            fields = [
                f'point={record["point_index"]}',
                f'vram={record["current_VRAM"]}GB',
                f'batch={record["batch_size"]}',
                f'scale={record["scale"]}',
                f'conn={record["conn_num"]}',
                f'neurons={record["neurons"]}',
                f'message={record["message"]}',
                f'est_bytes={record["current_VRAM_bytes"]}',
                f'est_gib={record["current_VRAM_GiB"]:.6f}',
                f'elapsed_s={record["elapsed_s"]:.6f}',
                f'firing_rate={record["firing_rate"]:.6f}',
            ]
            if record.get('exception'):
                fields.append(f'exception={record["exception"]}')
            lines.append(' | '.join(fields))

    return '\n'.join(lines) + '\n'


def benchmark_vram_limit(
    data_type: str = 'binary',
    duration=0.5 * u.ms,
    homo: bool = True,
    backend: str | None = None,
    efferent_target: str = 'post',
    _N: int = 4000,
    vram_start: int = 3,
    vram_end: int = 24,
    vram_step: int = 1,
    sample_points_per_batch: int = 2,
    fixed_batch_sizes: tuple[int, ...] = DEFAULT_FIXED_BATCH_SIZES,
    scale_max: int = 1250,
    conn_max: int = 2000,
    data_size: int = DEFAULT_DATA_SIZE,
    mv_layout: str = 'row_gather',
    clear_runtime_cache_between_cases: bool = True,
    save_memory_profile_on_error: bool = True,
    debug_output_dir: str = 'result-vram-limit-debug',
    output_dir: str = 'result-vram-limit',
    report_dir: str = 'result-vram-limit',
):
    import dev.fcn.BenchmarkTools as BT

    if efferent_target not in _VALID_TARGETS:
        raise ValueError(f'efferent_target must be one of {_VALID_TARGETS}, got {efferent_target!r}.')
    if mv_layout not in _VALID_LAYOUTS:
        raise ValueError(f'mv_layout must be one of {_VALID_LAYOUTS}, got {mv_layout!r}.')
    if vram_step <= 0:
        raise ValueError(f'vram_step must be > 0, got {vram_step}.')
    if vram_start > vram_end:
        raise ValueError(f'vram_start must be <= vram_end, got {vram_start} > {vram_end}.')
    if sample_points_per_batch < 1:
        raise ValueError(
            f'sample_points_per_batch must be >= 1, got {sample_points_per_batch}.'
        )

    normalized_batches = _normalize_batch_sizes(fixed_batch_sizes)
    vram_steps = list(range(vram_start, vram_end + 1, vram_step))
    vram_plan = _generate_mm_vram_sequence(
        vram_steps=vram_steps,
        fixed_batch_sizes=normalized_batches,
        sample_points_per_batch=sample_points_per_batch,
        _N=_N,
        scale_max=scale_max,
        conn_max=conn_max,
        homo=homo,
        data_size=data_size,
    )
    total_candidate_points = _count_total_candidate_points(vram_plan)
    if total_candidate_points == 0:
        raise ValueError(
            f'No boundary candidate states generated for {_BENCHMARK_NAME!r}, '
            f'vram range {vram_start}..{vram_end} GB.'
        )

    print(f'=== VRAM-limit progressive benchmark ({_BENCHMARK_NAME}) ===')
    make_simulation_batch_run = _load_make_simulation_batch_run()
    backends_to_use = [backend] if backend is not None else DEFAULT_BACKENDS
    runtime_platform = _announce_runtime_platform()

    debug_dir_path = _resolve_output_dir(debug_output_dir)
    output_dir_path = _resolve_output_dir(output_dir)
    report_dir_path = _resolve_output_dir(report_dir)

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
                f'vram_limit_{_BENCHMARK_NAME}_{efferent_target}',
                'fcnmm',
                'coba_ei',
                duration=duration,
            )
            csv_recorder.print_header(
                operator='fcnmm',
                data_type=data_type,
                backend=back,
                mode=efferent_target,
                duration=duration,
                homo=('homo' if homo else 'hetero'),
                benchmark_name=_BENCHMARK_NAME,
                mv_layout=mv_layout,
                limit_GB=vram_end,
                fixed_batch_sizes=','.join(str(batch) for batch in normalized_batches),
                sample_points_per_batch=sample_points_per_batch,
            )
            csv_recorder.print_table_header(show_conn=True, show_batch=True)

            case_records: list[CaseRecord] = []
            device_error_hit = False
            point_counter = 0

            for vram_gb, batch_plan in vram_plan.items():
                if device_error_hit:
                    break

                print(f'\n--- Testing VRAM budget: {vram_gb} GB ---')
                for batch_size, pairs in batch_plan.items():
                    if device_error_hit:
                        break

                    if not pairs:
                        print(
                            f'  [Skip] batch_size={batch_size}: no valid boundary points '
                            f'under {vram_gb}GB'
                        )
                        continue

                    print(f'  batch_size={batch_size}: {len(pairs)} boundary points')
                    for scale, conn in pairs:
                        if device_error_hit:
                            break

                        point_counter += 1
                        point_index = f'{point_counter}/{total_candidate_points}'
                        current_vram_bytes = _estimate_mm_bytes(
                            scale=scale,
                            batch_size=batch_size,
                            conn=conn,
                            _N=_N,
                            homo=homo,
                            data_size=data_size,
                        )
                        current_vram_gib = current_vram_bytes / (1024 ** 3)
                        log_collector.clear()
                        run: CompiledRun | None = None
                        case_name = (
                            f'{_BENCHMARK_NAME}_{back}_{data_type}_{efferent_target}_{mv_layout}_'
                            f'vram{vram_gb}_batch{batch_size}_scale{scale}_conn{conn}'
                        )
                        record_base = {
                            'benchmark_name': _BENCHMARK_NAME,
                            'backend': back,
                            'data_type': data_type,
                            'synaptic_type': efferent_target,
                            'scale': int(scale),
                            'conn_num': int(conn),
                            'batch_size': int(batch_size),
                            'limit_GB': int(vram_end),
                            'current_VRAM': int(vram_gb),
                            'current_VRAM_bytes': int(current_vram_bytes),
                            'current_VRAM_GiB': float(current_vram_gib),
                            'mv_layout': mv_layout,
                            'point_index': point_index,
                            'homo': 'homo' if homo else 'hetero',
                        }

                        try:
                            with warnings.catch_warnings(record=True) as py_warnings:
                                warnings.simplefilter('always')
                                run = cast(CompiledRun, make_simulation_batch_run(
                                    scale=scale,
                                    batch_size=batch_size,
                                    data_type=data_type,
                                    efferent_target=efferent_target,
                                    duration=duration,
                                    conn_num=conn,
                                    homo=homo,
                                    mv_layout=mv_layout,
                                ))

                                _run_and_block(run)

                                t0 = time.time()
                                out = _run_and_block(run)
                                t1 = time.time()
                                elapsed = t1 - t0

                            if isinstance(out, tuple) and len(out) >= 2:
                                n, rate = out[0], out[1]
                                neuron_count = int(n)
                                rate_value = float(rate)
                            else:
                                neuron_count = -1
                                rate_value = -1.0

                            jax_warned = log_collector.has_memory_warning()
                            py_mem_warned = any(
                                any(keyword in str(w.message).lower() for keyword in (
                                    'memory', 'allocation', 'resource', 'oom', 'exceeds',
                                ))
                                for w in py_warnings
                            )
                            message = 'JAX_warning' if (jax_warned or py_mem_warned) else 'OK'
                            if message == 'JAX_warning':
                                print(
                                    f'    [JAX_warning] scale={scale}, batch={batch_size}, '
                                    f'conn={conn}, VRAM={vram_gb}GB'
                                )

                            csv_recorder.print_row(
                                scale,
                                neuron_count,
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
                                homo=('homo' if homo else 'hetero'),
                                batch_size=batch_size,
                                benchmark_name=_BENCHMARK_NAME,
                                limit_GB=vram_end,
                                current_VRAM=vram_gb,
                                mv_layout=mv_layout,
                                message=message,
                                current_VRAM_bytes=current_vram_bytes,
                                current_VRAM_GiB=current_vram_gib,
                                point_index=point_index,
                            )
                            case_records.append({
                                **record_base,
                                'neurons': neuron_count,
                                'elapsed_s': float(elapsed),
                                'firing_rate': float(rate_value),
                                'message': message,
                                'exception': '',
                            })
                        except Exception as exc:
                            if save_memory_profile_on_error:
                                _write_memory_debug_snapshot(
                                    case_name=case_name,
                                    output_dir=debug_dir_path,
                                    include_device_profile=True,
                                )
                            message = 'Device_error' if _is_oom_error(exc) else 'Error'
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
                                homo=('homo' if homo else 'hetero'),
                                batch_size=batch_size,
                                benchmark_name=_BENCHMARK_NAME,
                                limit_GB=vram_end,
                                current_VRAM=vram_gb,
                                mv_layout=mv_layout,
                                message=message,
                                current_VRAM_bytes=current_vram_bytes,
                                current_VRAM_GiB=current_vram_gib,
                                point_index=point_index,
                                exception=str(exc),
                            )
                            case_records.append({
                                **record_base,
                                'neurons': -1,
                                'elapsed_s': -1.0,
                                'firing_rate': -1.0,
                                'message': message,
                                'exception': str(exc),
                            })
                            if message == 'Device_error':
                                print(
                                    f'    [Device_error] scale={scale}, batch={batch_size}, '
                                    f'conn={conn}, VRAM={vram_gb}GB'
                                )
                                print('    Stopping test.')
                                device_error_hit = True
                                break

                            print(
                                f'    [Error] scale={scale}, batch={batch_size}, '
                                f'conn={conn}: {exc}'
                            )
                        finally:
                            if clear_runtime_cache_between_cases:
                                _safe_clear_runtime_cache(run)

            homo_str = 'homo' if homo else 'hetero'
            output_name = (
                f'vram_limit_{_BENCHMARK_NAME}_{data_type}_{homo_str}_'
                f'{back}_{efferent_target}_{mv_layout}'
            )
            if output_dir_path == Path(__file__).resolve().parent:
                flush_dir = ''
            else:
                flush_dir = str(output_dir_path.relative_to(Path(__file__).resolve().parent))
            output_path = csv_recorder.flush_and_clear(output_name, dir=flush_dir)
            summary_counts, level_summaries = _summarize_case_records(vram_plan, case_records)
            report_text = _build_txt_report(
                created_at=datetime.now().isoformat(timespec='seconds'),
                runtime_platform=runtime_platform,
                backend=back,
                data_type=data_type,
                homo=homo,
                efferent_target=efferent_target,
                mv_layout=mv_layout,
                fixed_batch_sizes=normalized_batches,
                sample_points_per_batch=sample_points_per_batch,
                _N=_N,
                vram_steps=vram_steps,
                scale_max=scale_max,
                conn_max=conn_max,
                total_candidate_points=total_candidate_points,
                summary_counts=summary_counts,
                level_summaries=level_summaries,
                case_records=case_records,
                csv_output_path=output_path,
            )
            report_path = report_dir_path / f'{output_name}.txt'
            report_path.write_text(report_text, encoding='utf-8')

            if output_path is not None:
                print(f'Results saved to: {output_path}')
            print(f'Report saved to: {report_path}')
    finally:
        for logger in (jax_logger, xla_logger, root_logger):
            logger.removeHandler(log_collector)


if __name__ == '__main__':

    '''
    benchmark_vram_limit(
        data_type='binary',
        duration=0.5 * u.ms,
        efferent_target='post',
        homo=True,
        backend='cuda_raw',
        mv_layout='row_gather',
    )

    # compact + cuda_raw
    benchmark_vram_limit(
        data_type='compact',
        duration=0.5 * u.ms,
        efferent_target='post',
        homo=True,
        backend='cuda_raw',
        mv_layout='row_gather',
    )

    # bitpack + cuda_raw
    benchmark_vram_limit(
        data_type='bitpack',
        duration=0.5 * u.ms,
        efferent_target='post',
        homo=True,
        backend='cuda_raw',
        mv_layout='row_gather',
    )

    # binary + alternate backends (enable when needed)
    benchmark_vram_limit(
        data_type='binary',
        duration=0.5 * u.ms,
        efferent_target='post',
        homo=True,
        backend='jax_raw',
        mv_layout='row_gather',
    )
    benchmark_vram_limit(
        data_type='binary',
        duration=0.5 * u.ms,
        efferent_target='post',
        homo=True,
        backend='cuda_raw',
        mv_layout='row_gather',
    )
    benchmark_vram_limit(
        data_type='binary',
        duration=0.5 * u.ms,
        efferent_target='post',
        homo=True,
        backend='binary_fcnmm_atx_streaming-x_scatter',
        mv_layout='row_gather',
    )
    benchmark_vram_limit(
        data_type='binary',
        duration=0.5 * u.ms,
        efferent_target='post',
        homo=True,
        backend='binary_fcnmm_atx_fcn-x_scatter',
        mv_layout='row_gather',
    )
    benchmark_vram_limit(
        data_type='binary',
        duration=0.5 * u.ms,
        efferent_target='post',
        homo=True,
        backend='binary_fcnmm_atx_csr-x_scatter',
        mv_layout='row_gather',
    )
    benchmark_vram_limit(
        data_type='binary',
        duration=0.5 * u.ms,
        efferent_target='post',
        homo=True,
        backend='binary_fcnmm_atx_csr_compact-x_scatter',
        mv_layout='row_gather',
    )
    '''
