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
# Implementation of the paper:
#
# - Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al. (2007),
#   Simulation of networks of spiking neurons: a review of tools and strategies., J. Comput. Neurosci., 23, 3, 349–98
#
# which is based on the balanced network proposed by:
#
# - Vogels, T. P. and Abbott, L. F. (2005), Signal propagation and logic gating in networks of integrate-and-fire neurons., J. Neurosci., 25, 46, 10786–95
#


import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

'''
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
'''

import time

import brainunit as u
import jax

import brainevent
from COBA_2005_benchmark import make_simulation_run

#JAX_CAPTURED_CONSTANTS_REPORT_FRAMES = -1

backends = ['cuda_raw']
homo = True


def _is_oom_error(exc: Exception) -> bool:
    error_msg = str(exc).lower()
    return any(token in error_msg for token in (
        'resource_exhausted',
        'resource exhausted',
        'out of memory',
        'oom',
    ))


def benchmark_conn(
    conn_num=None,
    conn_prob=None,
    mode = 'pre',
    data_type='binary',
    duration=1e4 * u.ms,
    homo: bool = True,
    backend: str | None = None,
    probs_or_conn='conn',
    _N : int = 4000,
    limit_gb: int = 16,
    target_samples: int = 50
):
    import BenchmarkTools as BT

    if mode not in ('pre', 'post'):
        raise ValueError("mode must be either 'pre' or 'post'.")

    print(f'Benchmarking {mode}-synaptic connection updates...')

    backends_to_use = [backend] if backend is not None else backends

    generator = BT.TestingParamsGenerator(limit_GB=limit_gb, _N=_N)
    valid_states = generator.generate_boundary_params(
        homo_or_not=homo,
        _N=_N,
        sample_points=target_samples,
    )
    if not valid_states:
        raise ValueError(f'No valid boundary states generated under {limit_gb}GB.')

    csv_recorder = BT.CSV_record(f'{data_type}_{mode}_boundary', 'fcnmv', 'coba', duration=duration)
    csv_recorder.add_tag('VRAM-limit', limit_gb)
    homo_str = 'homo' if homo else 'hetero'
    last_path = None

    for back in backends_to_use:
        brainevent.config.set_backend('gpu', back)
        csv_recorder.print_header(
            operator='fcnmv', data_type=data_type, backend=back,
            mode=mode, duration=duration, homo=('homo' if homo else 'hetero')
        )
        csv_recorder.print_table_header(show_conn=True)

        for s, cn in valid_states:
            try:
                run = make_simulation_run(
                    scale=s,
                    data_type=data_type,
                    efferent_target=mode,
                    duration=duration,
                    conn_num=cn,
                    homo=homo
                )

                jax.block_until_ready(run())

                t0 = time.time()
                n, rate = jax.block_until_ready(run())
                t1 = time.time()
                elapsed = t1 - t0
                
                csv_recorder.print_row(s, n, elapsed, float(rate), conn_num=cn)
                csv_recorder.single_COBA_data_add(
                    'fcnmv', data_type, back, mode, cn, s, elapsed, float(rate), duration, 
                    homo=('homo' if homo else 'hetero')
                )

                flush_file_name = f'mv-boundary_{data_type}_{homo_str}_{back}_{mode}-float-input-16GB'

                last_path = csv_recorder.flush_and_clear(flush_file_name, dir='result-boundary-mv-4.1-final')

            except Exception as exc:
                if _is_oom_error(exc):
                    print(f'Skipping scale={s}, conn_num={cn} due to OOM: {exc}')
                    continue
                raise

    if last_path is not None:
        print(f'Results saved to: {last_path}')

    #csv_recorder.record_finish(dir='result-stage2',file_name='post-boundary-compact-homo-jaxandcuda')


if __name__ == '__main__':
    #benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='jax_raw')
    #benchmark_post_conn(data_type='binary', duration=1e2 * u.ms, homo = homo)
    #benchmark_post_conn(data_type='compact', duration=1e2 * u.ms, homo = True)
    #benchmark_conn(data_type='bitpack', mode='pre', duration=1e2 * u.ms, homo = True, backend='cuda_raw')
    benchmark_conn(data_type='binary', mode='pre', duration=1e2 * u.ms, homo = True, backend='jax_raw')
    #benchmark_pre_conn(data_type='compact', duration=1e2 * u.ms, homo= False)
    #benchmark_pre_conn(data_type='binary',duration=1e2 * u.ms)
    