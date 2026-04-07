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
# Boundary benchmark for fcnmm (matrix-matrix) operator.
# Tests across three dimensions: scale, batch_size, conn_num.
#

import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time

import brainunit as u
import jax

import brainevent
from COBA_2005_benchmark import make_simulation_batch_run

backends = ['cuda_raw']
homo = True


def benchmark_conn(
    data_type='binary',
    duration=1e2 * u.ms,
    homo: bool = True,
    mode: str = 'post',
    backend: str | None = None,
    _N: int = 4000,
    limit_gb: int = 6,
    scales: list | None = None,
    batch_sizes: list | None = None,
    conn_nums: list | None = None,
    target_samples : int = 25
):
    import dev.fcn.BenchmarkTools as BT

    print('Benchmarking...')

    backends_to_use = [backend] if backend is not None else backends

    generator = BT.TestingParamsGenerator_mm(
        limit_GB=limit_gb, _N=_N, conn_max = 2000, scale_max=500, batch_max=400
    )
    valid_states = generator.generate_params(dis_type='uniform', target_samples=target_samples, homo=homo)
    
    device_error_hit = False
    homo_str = 'homo' if homo else 'hetero'
    last_path = None
    csv_recorder = BT.CSV_record(f'binary_{mode}', 'fcnmm', 'coba', duration=duration)

    for back in backends_to_use:
        brainevent.config.set_backend('gpu', back)
        csv_recorder.print_header(
            operator='fcnmm', data_type=data_type, backend=back,
            mode=mode, duration=duration, homo=('homo' if homo else 'hetero')
        )
        csv_recorder.print_table_header(show_conn=True, show_batch=True)

        for s, bs, cn in valid_states:
            try:
                run = make_simulation_batch_run(
                    scale=s,
                    batch_size=bs,
                    data_type=data_type,
                    efferent_target=mode,
                    duration=duration,
                    conn_num=cn,
                    homo=homo,
                )

                jax.block_until_ready(run())

                t0 = time.time()
                n, rate = jax.block_until_ready(run())
                t1 = time.time()
                elapsed = t1 - t0

                csv_recorder.add_tag('VRAM-limit', limit_gb)
                csv_recorder.print_row(s, n, elapsed, float(rate), conn_num=cn, batch_size = bs)
                csv_recorder.single_COBA_data_add(
                    'fcnmm', data_type, back, mode, cn, s, elapsed, float(rate), duration,
                    homo=('homo' if homo else 'hetero'),
                    batch_size=bs,
                )

                flush_file_name = f'mm-boundary_{data_type}_{homo_str}_{back}_{mode}'

                last_path = csv_recorder.flush_and_clear(flush_file_name, dir='result-boundary-mm')
            except Exception as e:
                error_msg = str(e).lower()
                continue

    if last_path:
        print(f'\nDone. Results saved to: {last_path}')

if __name__ == '__main__':
    
    #benchmark_conn(data_type='binary', mode = 'pre', duration=1e2 * u.ms, homo=True, backend='cuda_raw')
    #benchmark_conn(data_type='compack',  mode = 'post',duration=1e2 * u.ms, homo=True, backend='cuda_raw')
    benchmark_conn(data_type='binary', mode = 'pre', duration=1e2 * u.ms, homo=True, backend='jax_raw')
    #benchmark_pre_conn(data_type='binary', duration=1e2 * u.ms, homo=True)
