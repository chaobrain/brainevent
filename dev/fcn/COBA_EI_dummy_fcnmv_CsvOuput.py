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

"""Single-file COBA-EI-style dummy FCNMV benchmark.

This benchmark keeps the FCN sparse-matrix initialization shape close to the
COBA-EI benchmark, but removes neuron / synapse dynamics entirely.  Each loop
iteration only:

1. Generates a 60 Hz sparse spike vector online
2. Builds the requested event representation
3. Runs one FCNMV dummy scatter operator
4. Accumulates a checksum to keep the loop observable

The goal is to isolate preprocessing plus dummy-kernel costs.
"""

from __future__ import annotations

import gc
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, cast

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import brainevent

conn_num_base = 80

CompiledRun = Callable[[], tuple[Any, Any, Any]]


class DummyCase(NamedTuple):
    case_label: str
    data_type: str
    backend: str


_DUMMY_CASES = (
    DummyCase('binary_dummy', 'binary', 'dummy_kernel'),
    DummyCase('bitpack_dummy', 'bitpack', 'dummy_kernel'),
    DummyCase('compact_packed_dummy', 'compact', 'dummy_kernel'),
    DummyCase('compact_vector_full_dummy', 'compact', 'dummy_kernel_vector_full'),
    DummyCase('compact_vector_active_dummy', 'compact', 'dummy_kernel_vector_active'),
)
_DUMMY_CASE_MAP = {case.case_label: case for case in _DUMMY_CASES}
_RAW_REPR_CASES = (
    'binary_dummy',
    'bitpack_dummy',
    'compact_packed_dummy',
)
_COMPACT_LAUNCH_CASES = (
    'compact_vector_full_dummy',
    'compact_vector_active_dummy',
)
_ALL_CASES = tuple(case.case_label for case in _DUMMY_CASES)
_FCN_MODE = 'post'
_FCN_LAYOUT = 'row_gather'
_LOGICAL_DT = 0.1 * u.ms
_TARGET_RATE_HZ = 60.0
_DEFAULT_LOOP_COUNT = 1_000_0
_DUMMY_COMPILER_OPTIONS = {'xla_gpu_enable_command_buffer': ''}


def _run_and_block(run: CompiledRun) -> tuple[Any, Any, Any]:
    result = run()
    blocked_result = jax.block_until_ready(result)
    return cast(tuple[Any, Any, Any], blocked_result)


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


@contextmanager
def _force_backend(platform: str, backend: str):
    old_backend = brainevent.config.get_backend(platform)
    brainevent.config.set_backend(platform, backend)
    try:
        yield
    finally:
        brainevent.config.set_backend(platform, old_backend)


def _bind_runtime_run(compiled_fn, *runtime_args):
    holder = {
        'compiled_fn': compiled_fn,
        'runtime_args': runtime_args,
    }

    def run():
        active_fn = holder['compiled_fn']
        active_args = holder['runtime_args']
        if active_fn is None:
            raise RuntimeError('This benchmark run has been released.')
        return active_fn(*active_args)

    def _clear_cache():
        active_fn = holder['compiled_fn']
        if active_fn is not None and hasattr(active_fn, 'clear_cache'):
            active_fn.clear_cache()

    def _release():
        _clear_cache()
        holder['compiled_fn'] = None
        holder['runtime_args'] = ()
        try:
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()

    run.clear_cache = _clear_cache
    run.release = _release
    return run


def _resolve_cases(selected_cases: Iterable[str] | None) -> tuple[DummyCase, ...]:
    if selected_cases is None:
        return _DUMMY_CASES
    resolved = []
    for label in selected_cases:
        case = _DUMMY_CASE_MAP.get(label)
        if case is None:
            raise ValueError(
                f'Unknown case_label {label!r}. '
                f'Expected one of {tuple(_DUMMY_CASE_MAP)}.'
            )
        resolved.append(case)
    return tuple(resolved)


def _validate_case_backend(case: DummyCase, platform: str) -> None:
    if case.data_type == 'binary':
        available = tuple(brainevent.binary_fcnmv_p.available_backends(platform))
    elif case.data_type == 'bitpack':
        available = tuple(brainevent.bitpack_binary_fcnmv_p.available_backends(platform))
    elif case.data_type == 'compact':
        available = tuple(brainevent.compact_binary_fcnmv_p.available_backends(platform))
    else:
        raise ValueError(f'Unsupported dummy case data_type: {case.data_type!r}')
    if case.backend not in available:
        raise ValueError(
            f'Backend {case.backend!r} is unavailable for case {case.case_label!r} '
            f'on platform {platform!r}. Available: {available}'
        )


def _resolve_operand_builder(case: DummyCase):
    if case.data_type == 'binary':
        return lambda spikes: brainevent.BinaryArray(spikes)
    if case.data_type == 'bitpack':
        return lambda spikes: brainevent.BitPackedBinary(spikes)
    if case.backend == 'dummy_kernel':
        return lambda spikes: brainevent.CompactBinary.from_array(spikes)
    compact_only_ctor = getattr(
        brainevent.CompactBinary,
        'compacy_only_vector',
        getattr(brainevent.CompactBinary, 'compact_only_vector', None),
    )
    if compact_only_ctor is None:
        raise AttributeError('CompactBinary is missing a compact-only 1D constructor.')
    return lambda spikes: compact_only_ctor(spikes)


def _make_post_conn(
    scale: int,
    conn_num: int,
    *,
    _N: int = 4000,
    conn_weight_base: u.Quantity = 0.6 * u.mS,
) -> brainevent.FixedPostNumConn:
    if conn_num < 1:
        raise ValueError(f'conn_num must be >= 1, got {conn_num}.')

    num = int(scale * _N)
    shape = (num, num)
    with jax.ensure_compile_time_eval():
        indices_np = np.random.randint(0, num, size=(num, conn_num)).astype(np.int32, copy=False)
        conn_weight = conn_weight_base * conn_num_base / conn_num
        weight_scalar = float(cast(u.Quantity, conn_weight).to_decimal(u.mS))
        weights = jnp.asarray([weight_scalar], dtype=jnp.float32)

    return brainevent.FixedPostNumConn(
        (weights, jnp.asarray(indices_np, dtype=jnp.int32)),
        shape=shape,
    )


def make_simulation_run(
    *,
    scale: int,
    data_type: str,
    backend: str,
    loop_count: int = _DEFAULT_LOOP_COUNT,
    conn_num: int = 80,
    dt: u.Quantity = _LOGICAL_DT,
    target_rate_hz: float = _TARGET_RATE_HZ,
    _N: int = 4000,
    seed: int = 1234,
) -> CompiledRun:
    case = next(
        (c for c in _DUMMY_CASES if c.data_type == data_type and c.backend == backend),
        None,
    )
    if case is None:
        raise ValueError(
            f'Unsupported dummy case combination data_type={data_type!r}, backend={backend!r}.'
        )
    if loop_count < 1:
        raise ValueError(f'loop_count must be >= 1, got {loop_count}.')

    conn = _make_post_conn(scale, int(conn_num), _N=_N)
    operand_builder = _resolve_operand_builder(case)
    num = int(conn.shape[0])
    dt_s = float(cast(u.Quantity, dt).to_decimal(u.second))
    p_active = float(target_rate_hz) * dt_s
    probe_count = min(8, int(conn.indices.shape[0]))
    probe_indices = jnp.asarray(conn.indices[:probe_count, 0], dtype=jnp.int32)
    if not (0.0 <= p_active <= 1.0):
        raise ValueError(
            f'Invalid per-step spike probability {p_active}. '
            f'Check target_rate_hz={target_rate_hz} and dt={dt}.'
        )
    base_key = jax.random.PRNGKey(seed)

    @jax.jit(compiler_options=_DUMMY_COMPILER_OPTIONS)
    def _run(conn):
        n = conn.shape[0]
        p = jnp.asarray(p_active, dtype=jnp.float32)
        dt_s_arr = jnp.asarray(dt_s, dtype=jnp.float32)
        n_arr = jnp.asarray(float(n), dtype=jnp.float32)

        def body(i, carry):
            checksum, mean_active = carry
            step_key = jax.random.fold_in(base_key, i)
            spikes = jax.random.bernoulli(step_key, p=p, shape=(n,))
            operand = operand_builder(spikes)
            y = operand @ conn
            active_count = jnp.asarray(jnp.sum(spikes, dtype=jnp.int32), dtype=jnp.float32)
            step = jnp.asarray(i + 1, dtype=jnp.float32)
            mean_active = mean_active + (active_count - mean_active) / step
            # The checksum is not a metric we compare across cases.
            # It only keeps the loop observably dependent on operator outputs so
            # XLA cannot discard the FCNMV call as dead work.
            checksum = checksum + jnp.asarray(jnp.sum(y[probe_indices]), dtype=jnp.float32)
            return checksum, mean_active

        checksum, mean_active = jax.lax.fori_loop(
            0,
            loop_count,
            body,
            (jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32)),
        )
        realized_rate_hz = mean_active / (n_arr * dt_s_arr)
        return jnp.asarray(n, dtype=jnp.int32), realized_rate_hz, checksum

    return _bind_runtime_run(_run, conn)


def benchmark_conn(
    *,
    limit_GB: int = 16,
    _N: int = 4000,
    loop_count: int = _DEFAULT_LOOP_COUNT,
    dt: u.Quantity = _LOGICAL_DT,
    target_rate_hz: float = _TARGET_RATE_HZ,
    non_repeat: bool = True,
    selected_cases: Iterable[str] | None = None,
):
    import BenchmarkTools as BT

    runtime_platform = _announce_runtime_platform()
    if runtime_platform != 'gpu':
        raise RuntimeError(
            f'This dummy benchmark requires GPU runtime; got platform={runtime_platform!r}.'
        )

    logical_duration = loop_count * dt
    dt_ms = float(cast(u.Quantity, dt).to_decimal(u.ms))
    cases = _resolve_cases(selected_cases)
    for case in cases:
        _validate_case_backend(case, runtime_platform)

    base_dir = Path(__file__).resolve().parent
    resume_dir = base_dir / 'benchmarker-test' / 'dummy'
    resume_dir.mkdir(parents=True, exist_ok=True)

    print(
        f'Benchmarking COBA-EI dummy FCNMV cases: '
        f'{", ".join(case.case_label for case in cases)}'
    )

    for case in cases:
        flush_file_name = (
            f'COBA-EI no op -dummy_{case.case_label}_{case.data_type}_{case.backend}_'
            f'{_FCN_MODE}-{_FCN_LAYOUT}_loop-{loop_count}-float-input-{limit_GB}GB'
        )
        resume_csv_path = None
        if non_repeat:
            resume_csv_path = str((resume_dir / f'{flush_file_name}.csv').resolve())

        tp_generator = BT.TestingParamsGenerator_mv(
            limit_GB=limit_GB,
            _N=_N,
            sample_points=30,
            conn_max=4000,
            scale_max=2000,
            mode=_FCN_MODE,
            data_type=case.data_type,
            mv_layout=_FCN_LAYOUT,
            non_repeat=non_repeat,
            flush_file_name=resume_csv_path,
        )
        valid_pairs = tp_generator.generate_params(
            dis_type='uniform',
            target_samples=30,
            data_size=4,
            homo=True,
        )

        csv_recorder = BT.CSV_record(
            f'dummy_{_FCN_MODE}',
            'fcnmv',
            'coba_ei_dummy',
            duration=logical_duration,
        )
        csv_recorder.add_tag('limit_GB', f'{limit_GB}')
        csv_recorder.add_tag('mv_layout', _FCN_LAYOUT)
        csv_recorder.add_tag('loop_count', loop_count)
        csv_recorder.add_tag('dt_ms', dt_ms)
        csv_recorder.add_tag('target_rate_hz', float(target_rate_hz))
        csv_recorder.add_tag('case_label', case.case_label)

        csv_recorder.print_header(
            operator='fcnmv',
            data_type=case.data_type,
            backend=case.backend,
            mode=_FCN_MODE,
            duration=logical_duration,
            homo='homo',
            case_label=case.case_label,
            loop_count=loop_count,
            dt_ms=dt_ms,
            target_rate_hz=float(target_rate_hz),
        )
        csv_recorder.print_table_header(show_conn=True)

        last_path = None
        with _force_backend(runtime_platform, case.backend):
            for scale, _, conn_num in valid_pairs:
                run: CompiledRun | None = None
                try:
                    case_t0 = time.time()
                    run = make_simulation_run(
                        scale=int(scale),
                        data_type=case.data_type,
                        backend=case.backend,
                        loop_count=loop_count,
                        conn_num=int(conn_num),
                        dt=dt,
                        target_rate_hz=target_rate_hz,
                        _N=_N,
                    )

                    first_run_t0 = time.time()
                    _, _, _ = _run_and_block(run)
                    first_run_t1 = time.time()
                    first_run_elapsed = first_run_t1 - first_run_t0

                    steady_t0 = time.time()
                    n, rate, checksum = _run_and_block(run)
                    steady_t1 = time.time()
                    elapsed = steady_t1 - steady_t0

                    csv_recorder.print_row(scale, n, elapsed, float(rate), conn_num=int(conn_num))

                    pre_flush_elapsed = time.time() - case_t0
                    csv_recorder.single_COBA_data_add(
                        'fcnmv',
                        case.data_type,
                        case.backend,
                        _FCN_MODE,
                        int(conn_num),
                        int(scale),
                        elapsed,
                        float(rate),
                        logical_duration,
                        homo='homo',
                        first_run_s=first_run_elapsed,
                        pre_flush_s=pre_flush_elapsed,
                        checksum=float(checksum),
                    )

                    flush_t0 = time.time()
                    last_path = csv_recorder.flush_and_clear(flush_file_name, dir='benchmarker-test/dummy')
                    flush_t1 = time.time()

                    flush_elapsed = flush_t1 - flush_t0
                    case_elapsed = flush_t1 - case_t0

                    print(
                        f'    timing: first_run={first_run_elapsed:.6f}s, '
                        f'steady={elapsed:.6f}s, flush={flush_elapsed:.6f}s, '
                        f'end_to_end={case_elapsed:.6f}s, checksum={float(checksum):.6f}'
                    )
                except Exception as exc:
                    print(
                        f'  [Error] case={case.case_label}, scale={scale}, conn_num={conn_num}: {exc}'
                    )
                    continue
                finally:
                    _release_run(run)
                    run = None

        if last_path:
            print(f'Done. Results saved to: {last_path}')

if __name__ == '__main__':
    # First-stage raw comparison:
    #   bitpack_dummy - binary_dummy
    #   compact_packed_dummy - binary_dummy
    # approximates preprocessing + representation-path overhead relative to the
    # binary baseline. It is not a pure preprocessing number, because the dummy
    # kernels still differ slightly in launch/read path.
    '''
    benchmark_conn(
        limit_GB=16,
        loop_count=_DEFAULT_LOOP_COUNT,
        selected_cases=_RAW_REPR_CASES,
    )   
    '''
    benchmark_conn(
         limit_GB=16,
         loop_count=_DEFAULT_LOOP_COUNT,
         selected_cases=_COMPACT_LAUNCH_CASES
     )

    # Extra launch-organization comparison for compact-only paths:
    # benchmark_conn(
    #     limit_GB=16,
    #     loop_count=_DEFAULT_LOOP_COUNT,
    #     selected_cases=_COMPACT_LAUNCH_CASES,
    # )

    # All five dummy paths:
    # benchmark_conn(
    #     limit_GB=16,
    #     loop_count=_DEFAULT_LOOP_COUNT,
    #     selected_cases=_ALL_CASES,
    # )

    # Quick smoke:
    # benchmark_conn(
    #     limit_GB=1,
    #     loop_count=1_000,
    #     selected_cases=('binary_dummy',),
    # )
