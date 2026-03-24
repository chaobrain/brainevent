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

import time

import brainunit as u
import jax

import brainevent
from COBA_2005_benchmark import make_simulation_run


backends = ['cuda_raw']

def benchmark_post_conn(
    conn_num=None,
    conn_prob=None,
    data_type='binary',
    duration=1e4 * u.ms,
    homo: bool = True,
    backend: str | None = None,
    probs_or_conn='conn',
    _N : int = 4000,
    limit_gb: int = 24,
    target_samples: int = 50
):
    import dev.fcn.BenchmarkTools as BT

    print('Benchmarking post-synaptic connection updates...')

    backends_to_use = [backend] if backend is not None else backends

    valid_states = BT.generate_params(dis_type= 'uniform' ,_N=_N, limit_gb=limit_gb, target_samples=target_samples)

    csv_recorder = BT.CSV_record('binary_post', 'fcnmv', 'coba', duration=duration,)

    for back in backends_to_use:
        brainevent.config.set_backend('gpu', back)
        csv_recorder.print_header(
            operator='fcnmv', data_type=data_type, backend=back,
            mode='post', duration=duration, homo=('homo' if homo else 'hetero')
        )
        csv_recorder.print_table_header(show_conn=True)

        for s, cn in valid_states:
            try:
                run = make_simulation_run(
                    scale=s,
                    data_type=data_type,
                    efferent_target='post',
                    duration=duration,
                    conn_num=cn,
                    homo=homo
                )

                jax.block_until_ready(run())

                t0 = time.time()
                n, rate = jax.block_until_ready(run())
                t1 = time.time()
                elapsed = t1 - t0
                
                csv_recorder.add_tag('warp_or_thread', 'tpr')
                csv_recorder.print_row(s, n, elapsed, float(rate), conn_num=cn)
                csv_recorder.single_COBA_data_add(
                    'fcnmv', data_type, back, 'post', cn, s, elapsed, float(rate), duration, 
                    homo=('homo' if homo else 'hetero')
                )
            except Exception as e:
                print(f'  [Error] VRAM Boundary Exception at scale={s}, conn_num={cn}: {e}')
                continue

    csv_recorder.record_finish('boundary_boolmode-default')

def benchmark_pre_conn(
        conn_num=None, 
        conn_prob=None,
        data_type='binary', 
        duration=1e2 * u.ms, 
        homo:bool = True, 
        backend: str | None = None,
        probs_or_conn='conn',
        _N : int = 4000,
        limit_gb: int = 16,
        target_samples: int = 50
        ):
    print('Benchmarking pre-synaptic connection updates...')
    import dev.fcn.BenchmarkTools as BT

    backends_to_use = [backend] if backend is not None else backends

    
    valid_states = BT.generate_params(dis_type= 'uniform' ,_N=_N, limit_gb=limit_gb, target_samples=target_samples)

    csv_recorder = BT.CSV_record('binary_pre', 'fcnmv', 'coba', duration=duration, conn=conn_num)

    for back in backends_to_use:
        brainevent.config.set_backend('gpu', back)
        csv_recorder.print_header(
            operator='fcnmv', data_type=data_type, backend=back,
            mode='pre', duration=duration, homo=('homo' if homo else 'hetero')
        )
        csv_recorder.print_table_header(show_conn=True)

        for s, cn in valid_states:
            try:
                run = make_simulation_run(
                    scale=s,
                    data_type=data_type,
                    efferent_target='pre',
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
                    'fcnmv', data_type, back, 'pre', cn, s, elapsed, float(rate), duration, 
                    homo=('homo' if homo else 'hetero')
                )
            except Exception as e:
                print(f'  [Error] VRAM Boundary Exception at scale={s}, conn_num={cn}: {e}')
                continue

    csv_recorder.record_finish('boundary_floatmode_bitpack')


if __name__ == '__main__':
    #benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='jax_raw')
    benchmark_post_conn(data_type='binary', duration=1e2 * u.ms)
    #benchmark_post_conn(data_type='compact', duration=1e2 * u.ms)
    #benchmark_pre_conn(data_type='bitpack', duration=1e2 * u.ms)
    #benchmark_pre_conn(data_type='binary',duration=1e2 * u.ms)
    