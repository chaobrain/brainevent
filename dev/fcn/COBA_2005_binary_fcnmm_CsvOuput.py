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
from COBA_2005_benchmark import make_simulation_batch_run

# Global configuration

scales_ALL = [400,600,800,100,200,]
backends = ['jax_raw', 'cuda_raw']
conn_nums = [20,40, 80, 160]
probs = [0.001]
default_batch_sizes = [64, 128, 256]
'''
scales_ALL = [1]
conn_nums = [160]
default_batch_sizes = [256]
'''
def benchmark_conn( mode = 'post',  
                   conn_num=None,
                   conn_prob=None,
                   batch_size=None,
                   data_type='binary',
                   duration=1e3 * u.ms,
                   homo: bool = True,
                   backend: str | None = None,
                   params_type='conn',
                   probs_or_conn=None,
                   _N: int = 4000,
                   limit_GB: int = 6,
                   target_samples: int = 50,
                   dis_type: str = 'uniform',
                   data_size: int = 4,
                   scale_max: int = 800,
                   conn_max: int = 800,
                   batch_max: int = 256):

    print('Benchmarking post-synaptic connection updates...')

    import dev.fcn.BenchmarkTools as BT

    backends_to_use = [backend] if backend is not None else backends

    if probs_or_conn is not None:
        params_type = probs_or_conn
    if params_type == 'probs':
        params_type = 'prob'

    if mode == 'post':
        scales = scales_ALL
    else:
        scales = scales_ALL

    batch_list = [batch_size] if batch_size is not None else default_batch_sizes
    TPGenerator = BT.TestingParamsGenerator_mm(
        limit_GB=limit_GB,
        _N=_N,
        conn_max=conn_max,
        scale_max=scale_max,
        batch_max=batch_max,
    )

    if params_type == 'conn':
        conn_nums_to_use = [conn_num] if conn_num is not None else conn_nums
        valid_pairs = []
        for s in scales:
            for b in batch_list:
                for cn in conn_nums_to_use:
                    #if TPGenerator.is_valid_mm(s, b, cn, homo):
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
            f"Unknown params_type: {params_type!r}. "
            "Choose from 'conn', 'prob', or 'dist'."
        )

    if not valid_pairs:
        print(
            f'No valid benchmark cases generated for params_type={params_type!r}. '
            f'Please adjust limit_GB={limit_GB}, scales, batch sizes, or connection settings.'
        )
        return

    csv_recorder = BT.CSV_record(f'binary_{mode}', 'fcnmm', 'coba', duration=duration)

    homo_str = 'homo' if homo else 'hetero'
    last_path = None

    for back in backends_to_use:
        brainevent.config.set_backend('gpu', back)
        csv_recorder.print_header(operator='fcnmm', data_type=data_type, backend=back,
                mode=mode, duration=duration, homo=('homo' if homo else 'hetero'))
        csv_recorder.print_table_header(show_batch=True, show_conn=True)

        for scale, prob, cn, batch in valid_pairs:

            try:
                run = make_simulation_batch_run(
                    scale=scale,
                    batch_size=batch,
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
                csv_recorder.print_row(scale, n, elapsed, float(rate), batch_size=batch, conn_num=cn)
                
                csv_recorder.single_COBA_data_add('fcnmm', data_type, back, mode, cn, scale, elapsed, float(rate), duration, homo=('homo' if homo else 'hetero'), batch_size=batch)
                
                flush_file_name = f'mm_{data_type}_{homo_str}_{back}_{mode}_{params_type}'

                last_path = csv_recorder.flush_and_clear(flush_file_name, dir='result-mm-structural-dist')
            except Exception as e:
                print(f'  [Error] scale={scale}, conn_num={cn}: {e}')
                continue

    if last_path:
        print(f'\nDone. Results saved to: {last_path}')

if __name__ == '__main__':
    #benchmark_conn(data_type='compact',mode='post',  duration=0.5 * u.ms, homo = True, backend='cuda_raw', probs_or_conn='conn')
    #benchmark_conn(data_type='compact',mode='pre',  duration=1e2 * u.ms, homo = True, backend='cuda_raw', probs_or_conn='conn')
    #benchmark_conn(data_type='bitpack',mode='post',  duration=1e2 * u.ms, homo = True, backend='cuda_raw', probs_or_conn='conn')
    #benchmark_conn(data_type='bitpack',mode='pre',  duration=1e2 * u.ms, homo = True, backend='cuda_raw', probs_or_conn='conn')
    '''
    benchmark_conn(
        data_type='binary',
        mode='post',
        duration=100 * u.ms,
        homo=True,
        backend='binary_fcnmm_atx_streaming-x_scatter',
        params_type='conn',
    )
    
    
    benchmark_conn(
        data_type='binary',
        mode='post',
        duration=100 * u.ms,
        homo=True,
        backend='binary_fcnmm_atx_fcn-x_scatter',
        params_type='conn',
    )

    benchmark_conn(
        data_type='binary',
        mode='post',
        duration=100 * u.ms,
        homo=True,
        backend='binary_fcnmm_atx_csr-x_scatter',
        params_type='conn',
    )
    '''
    benchmark_conn(
        data_type='binary',
        mode='post',
        duration=100 * u.ms,
        homo=True,
        backend='binary_fcnmm_atx_csr_compact-x_scatter',
        params_type='conn',
    )
    '''
    benchmark_conn(
        data_type='compact',
        mode='post',
        duration=100 * u.ms,
        homo=True,
        backend='cuda_raw',
        params_type='conn',
    )
    benchmark_conn(
        data_type='bitpack',
        mode='post',
        duration=100 * u.ms,
        homo=True,
        backend='cuda_raw',
        params_type='conn',
    )
    '''
''''''
    #benchmark_conn(data_type='binary',mode='post',  duration=1e2 * u.ms, homo = True, backend='jax_raw')
    #benchmark_pre_conn(batch_size=16, conn_num=80, data_type='binary', duration=1e3 * u.ms, homo = True)
    #bench_fcnmm()
    
