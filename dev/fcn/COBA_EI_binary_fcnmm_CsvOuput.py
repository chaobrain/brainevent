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

#
# COBA EI batch benchmark wrapper for FCN-MM.
#
# This mirrors the explicit FixedPostNumConn-based COBA EI benchmark and
# records batch simulation runs that exercise FCN matrix-matrix routes.
#

import importlib.util
import gc
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import brainunit as u
import jax

import brainevent

_BENCHMARK_PATH = Path(__file__).with_name('COBA EI benchmark.py')
_SPEC = importlib.util.spec_from_file_location('coba_ei_benchmark', _BENCHMARK_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'Unable to load benchmark module from {_BENCHMARK_PATH}')
_BENCHMARK_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK_MODULE)
make_simulation_batch_run = _BENCHMARK_MODULE.make_simulation_batch_run

scales = [1, 2, 4, 6, 8, 10, 20, 40, 60]
backends = ['jax_raw', 'cuda_raw']
conn_nums = [20, 40, 80, 160, 320, 640]
probs = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
default_batch_sizes = [16, 32, 64, 128, 256]

CompiledRun = Callable[[], tuple[int, Any]]


def _run_and_block(run: CompiledRun) -> tuple[int, Any]:
    result = run()
    blocked_result = jax.block_until_ready(result)
    return cast(tuple[int, Any], blocked_result)


def _announce_runtime_platform() -> str:
    platform = jax.default_backend()
    devices = ', '.join(str(device) for device in jax.devices())
    print(f'Runtime platform: {platform}; devices: {devices}')
    return platform


def _release_run(run: CompiledRun | None) -> None:
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


def benchmark_conn(
    mode: str = 'post',
    conn_num=None,
    conn_prob=None,
    batch_size=None,
    data_type: str = 'binary',
    duration=1e4 * u.ms,
    homo: bool = True,
    backend: str | None = None,
    params_type: str = 'conn',
    probs_or_conn: str | None = None,
    _N: int = 4000,
    limit_GB: int = 16,
    target_samples: int = 30,
    dis_type: str = 'uniform',
    data_size: int = 4,
    scale_max: int = 2000,
    conn_max: int = 4000,
    batch_max: int = 256,
    mv_layout: str = 'row_gather',
    non_repeat: bool = True,
):
    import BenchmarkTools as BT

    duration = cast(u.Quantity, duration)
    csv_duration = cast(Any, duration)

    print(f'Benchmarking {mode}-synaptic batch connection updates...')
    runtime_platform = _announce_runtime_platform()

    if probs_or_conn is not None:
        params_type = probs_or_conn
    if params_type == 'probs':
        params_type = 'prob'

    batch_list = [batch_size] if batch_size is not None else default_batch_sizes
    backends_to_use = [backend] if backend is not None else backends
    homo_str = 'homo' if homo else 'hetero'

    csv_recorder = BT.CSV_record(f'binary_{mode}', 'fcnmm', 'coba_ei', duration=csv_duration, conn=conn_num)

    last_path = None
    header_conn_num = conn_num if params_type == 'conn' and conn_num is not None else None

    for back in backends_to_use:
        flush_file_name = (
            f'COBA-EI-fcnmm-4.27_{data_type}_{homo_str}_{back}_{mode}-{mv_layout}-float-input-{limit_GB}GB'
        )
        resume_csv_path = None
        if non_repeat:
            resume_csv_path = str(
                Path(__file__).resolve().parent / 'benchmarker-test' / f'{flush_file_name}.csv'
            )

        TPGenerator = BT.TestingParamsGenerator_mm(
            limit_GB=limit_GB,
            _N=_N,
            conn_max=conn_max,
            scale_max=scale_max,
            batch_max=batch_max,
            non_repeat=non_repeat,
            flush_file_name=resume_csv_path,
        )

        if params_type == 'conn':
            conn_nums_to_use = [conn_num] if conn_num is not None else conn_nums
            valid_pairs = []
            for s in scales:
                for b in batch_list:
                    for cn in conn_nums_to_use:
                        valid_pairs.append((s, None, cn, b))
        elif params_type == 'prob':
            probs_to_use = [conn_prob] if conn_prob is not None else probs
            valid_pairs = TPGenerator.make_simulation_params_probs(
                probs_to_use,
                scales,
                batch_list,
                data_size=data_size,
                homo=homo,
            )
        elif params_type == 'dist':
            valid_states = TPGenerator.generate_params(
                dis_type=dis_type,
                target_samples=target_samples,
                data_size=data_size,
                homo=homo,
            )
            valid_pairs = [(scale, None, cn, batch) for scale, batch, cn in valid_states]
        else:
            raise ValueError(
                f'Unsupported params_type: {params_type!r}. '
                "Choose from 'conn', 'prob', or 'dist'."
            )

        if not valid_pairs:
            print(
                f'No valid mm benchmark cases generated for params_type={params_type!r}. '
                f'Please adjust limit_GB={limit_GB}, scales, batch sizes, or connection settings.'
            )
            continue

        brainevent.config.set_backend(runtime_platform, back)
        csv_recorder.print_header(
            operator='fcnmm',
            data_type=data_type,
            backend=back,
            mode=mode,
            conn_num=header_conn_num,
            batch_size=batch_size,
            duration=duration,
            homo=('homo' if homo else 'hetero'),
        )
        csv_recorder.print_table_header(show_batch=True, show_conn=True)

        for scale, prob, cn, batch in valid_pairs:
            run: CompiledRun | None = None
            try:
                case_t0 = time.time()
                run = cast(CompiledRun, make_simulation_batch_run(
                    scale=scale,
                    batch_size=batch,
                    data_type=data_type,
                    efferent_target=mode,
                    duration=duration,
                    conn_num=cn,
                    homo=homo,
                    mv_layout=mv_layout,
                ))

                first_run_t0 = time.time()
                _run_and_block(run)
                first_run_t1 = time.time()
                first_run_elapsed = first_run_t1 - first_run_t0

                steady_t0 = time.time()
                n, rate = _run_and_block(run)
                steady_t1 = time.time()
                elapsed = steady_t1 - steady_t0

                csv_recorder.add_tag('limit_GB', f'{limit_GB}')
                csv_recorder.add_tag('mv_layout', f'{mv_layout}')

                if prob is not None:
                    csv_recorder.add_tag('conn_prob', prob)

                csv_recorder.print_row(scale, n, elapsed, float(rate), batch_size=batch, conn_num=cn)

                pre_flush_elapsed = time.time() - case_t0
                csv_recorder.single_COBA_data_add(
                    'fcnmm',
                    data_type,
                    back,
                    mode,
                    cn,
                    scale,
                    elapsed,
                    float(rate),
                    csv_duration,
                    homo=('homo' if homo else 'hetero'),
                    batch_size=batch,
                    first_run_s=first_run_elapsed,
                    pre_flush_s=pre_flush_elapsed,
                )

                flush_t0 = time.time()
                last_path = csv_recorder.flush_and_clear(flush_file_name, dir='benchmarker-test')
                flush_t1 = time.time()

                flush_elapsed = flush_t1 - flush_t0
                case_elapsed = flush_t1 - case_t0

                print(
                    f'    timing: first_run={first_run_elapsed:.6f}s, '
                    f'steady={elapsed:.6f}s, flush={flush_elapsed:.6f}s, '
                    f'end_to_end={case_elapsed:.6f}s'
                )
            except Exception as e:
                print(f'  [Error] scale={scale}, batch_size={batch}, conn_num={cn}: {e}')
                continue
            finally:
                _release_run(run)
                run = None

    if last_path:
        print(f'\nDone. Results saved to: {last_path}')


if __name__ == '__main__':
    # benchmark_conn(
    #     data_type='binary',
    #     mode='post',
    #     duration=1e2 * u.ms,
    #     params_type='dist',
    #     homo=True,
    #     backend='cuda_raw',
    #     limit_GB=16,
    #     mv_layout='row_gather',
    # )
    # benchmark_conn(
    #     data_type='bitpack',
    #     mode='post',
    #     duration=1e2 * u.ms,
    #     params_type='dist',
    #     homo=True,
    #     backend='cuda_raw',
    #     limit_GB=16,
    #     mv_layout='row_gather',
    # )
    benchmark_conn(
        data_type='binary',
        mode='post',
        duration=1e2 * u.ms,
        params_type='dist',
        homo=True,
        backend='cuda_raw',
        limit_GB=16,
        mv_layout='row_gather',
    )
