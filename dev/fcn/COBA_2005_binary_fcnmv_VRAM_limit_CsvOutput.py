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
# VRAM-limit progressive benchmark for COBA 2005 (fcnmv operator).
#
# Starts from a small VRAM budget and increases step by step.
# - When JAX emits memory-related warnings  → tag ``message = JAX_warning``
# - When RESOURCE_EXHAUSTED is raised        → tag ``message = Device_error``, stop.
#

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
from COBA_2005_benchmark import make_simulation_run

backends = ['jax_raw']
data_size = 4

# ---------------------------------------------------------------------------
#  Logging handler to capture JAX / XLA memory warnings
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
        for r in self.records:
            msg = r.getMessage().lower()
            if any(k in msg for k in keywords):
                return True
        return False

    def clear(self):
        self.records.clear()


def benchmark_vram_limit(
    data_type: str = 'binary',
    duration=1e2 * u.ms,
    homo: bool = True,
    backend: str | None = None,
    efferent_target: str = 'pre',
    _N: int = 4000,
    vram_start: int = 1,
    vram_end: int = 24,
    vram_step: int = 1,
    sample_points: int = 5,
    scale_max: int = 2000,
    conn_max: int = 4000,
):
    """Progressive VRAM-limit benchmark.

    For each VRAM level from *vram_start* to *vram_end* (step *vram_step*),
    generate ``(scale, conn)`` pairs near that boundary using
    ``TestingParamsGenerator.generate_cs_pairs`` and run the COBA simulation.

    Tags recorded per row:
      - ``current_VRAM``  – the VRAM budget (GB) being tested.
      - ``message``       – ``OK`` | ``JAX_warning`` | ``Device_error``.
    """
    import dev.fcn.BenchmarkTools as BT

    print('=== VRAM-limit progressive benchmark ===')

    backends_to_use = [backend] if backend is not None else backends
    vram_steps = list(range(vram_start, vram_end + 1, vram_step))

    #data_size = data_size

    # Generate parameter sequence for all VRAM levels
    generator = BT.TestingParamsGenerator(
        limit_GB=vram_end, _N=_N,
        scale_max=scale_max, conn_max=conn_max,
    )
    vram_params = generator.generate_coba_vram_sequence(
        vram_steps=vram_steps,
        sample_points=sample_points,
        homo=homo,
        data_size=data_size,
    )

    csv_recorder = BT.CSV_record(
        f'vram_limit_{efferent_target}', 'fcnmv', 'coba', duration=duration,
    )

    # Attach a temporary log handler to capture JAX/XLA warnings
    log_collector = _WarningCollector()
    jax_logger = logging.getLogger('jax')
    xla_logger = logging.getLogger('jaxlib')
    root_logger = logging.getLogger()
    for lgr in (jax_logger, xla_logger, root_logger):
        lgr.addHandler(log_collector)

    device_error_hit = False

    try:
        for back in backends_to_use:
            brainevent.config.set_backend('gpu', back)

            csv_recorder.print_header(
                operator='fcnmv', data_type=data_type, backend=back,
                mode=efferent_target, duration=duration,
                homo=('homo' if homo else 'hetero'),
            )
            csv_recorder.print_table_header(show_conn=True)

            for vram_gb, pairs in vram_params.items():
                if device_error_hit:
                    break

                print(f'\n--- Testing VRAM budget: {vram_gb} GB '
                      f'({len(pairs)} parameter pairs) ---')

                for scale, conn in pairs:
                    if device_error_hit:
                        break

                    csv_recorder.add_tag('current_VRAM', vram_gb)
                    log_collector.clear()

                    try:
                        # Also catch Python-level warnings
                        with warnings.catch_warnings(record=True) as py_warnings:
                            warnings.simplefilter('always')

                            run = make_simulation_run(
                                scale=scale,
                                data_type=data_type,
                                efferent_target=efferent_target,
                                duration=duration,
                                conn_num=conn,
                                homo=homo,
                            )

                            # Warm-up run
                            jax.block_until_ready(run())

                            # Timed run
                            t0 = time.time()
                            n, rate = jax.block_until_ready(run())
                            t1 = time.time()
                            elapsed = t1 - t0

                        # Determine message tag
                        jax_warned = log_collector.has_memory_warning()
                        py_mem_warned = any(
                            any(k in str(w.message).lower()
                                for k in ('memory', 'allocation', 'resource'))
                            for w in py_warnings
                        )

                        if jax_warned or py_mem_warned:
                            csv_recorder.add_tag('message', 'JAX_warning')
                            print(f'  [JAX_warning] scale={scale}, conn={conn}, '
                                  f'VRAM={vram_gb}GB')
                        else:
                            csv_recorder.add_tag('message', 'OK')

                        csv_recorder.print_row(scale, n, elapsed, float(rate),
                                               conn_num=conn)
                        csv_recorder.single_COBA_data_add(
                            'fcnmv', data_type, back, efferent_target, conn,
                            scale, elapsed, float(rate), duration,
                            homo=('homo' if homo else 'hetero'),
                        )

                    except Exception as e:
                        error_msg = str(e).lower()
                        is_oom = any(kw in error_msg for kw in (
                            'resource_exhausted', 'resource exhausted',
                            'out of memory', 'oom',
                        ))

                        if is_oom:
                            csv_recorder.add_tag('message', 'Device_error')
                            csv_recorder.add_tag('current_VRAM', vram_gb)
                            csv_recorder.single_COBA_data_add(
                                'fcnmv', data_type, back, efferent_target,
                                conn, scale, -1, -1, duration,
                                homo=('homo' if homo else 'hetero'),
                            )
                            print(f'  [Device_error] RESOURCE_EXHAUSTED at '
                                  f'scale={scale}, conn={conn}, '
                                  f'VRAM={vram_gb}GB')
                            print('  Stopping test.')
                            device_error_hit = True
                            break
                        else:
                            print(f'  [Error] scale={scale}, conn={conn}: {e}')
                            continue

    finally:
        # Clean up logging handlers
        for lgr in (jax_logger, xla_logger, root_logger):
            lgr.removeHandler(log_collector)

    homo_str = 'homo' if homo else 'hetero'
    csv_recorder.record_finish(
        dir='result-vram-limit_jax',
        file_name=f'vram_limit_{data_type}_{homo_str}_{efferent_target}',
    )


if __name__ == '__main__':


    '''
    benchmark_vram_limit(
                data_type='binary',
                duration=1e2 * u.ms,
                efferent_target='pre',
                homo=True,
            )
    
    for homo in [True, False]:
        for data_type in ['pre','post']:
            benchmark_vram_limit(
                data_type='binary',
                duration=1e2 * u.ms,
                efferent_target=data_type,
                homo=homo,
                backend='cuda_raw'
            )
    '''        
    benchmark_vram_limit(
                data_type='binary',
                duration=1e2 * u.ms,
                efferent_target=data_type,
                homo=homo,
                backend='cuda_raw'
            )

    
