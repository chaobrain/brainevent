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
from CsvOutput import CSV_record, ResultPrinting

#brainevent.config.set_backend('gpu', 'jax_raw')
brainevent.config.set_backend('gpu', 'cuda_raw')


scales = [1, 60]
backends = ['jax_raw', 'cuda_raw']


rp = ResultPrinting()


def benchmark_post_conn(
    conn_num=80, data_type='binary', duration=1e4 * u.ms
): 
    print('Benchmarking post-synaptic connection updates...')
    csv_recorder = CSV_record('binary_post', 'fcnmv', 'coba')
    dur_ms = float(duration / u.ms)
    for backend in backends:
        brainevent.config.set_backend('gpu', backend)
        rp.print_header(operator='fcnmv', data_type=data_type, backend=backend,
                        mode='post', conn_num=conn_num, duration_ms=dur_ms)
        rp.print_table_header()
        for s in scales:
            run = make_simulation_run(
                scale=s,
                data_type=data_type,
                efferent_target='post',
                duration=duration,
                conn_num=conn_num
            )

            jax.block_until_ready(run())

            t0 = time.time()
            n, rate = jax.block_until_ready(run())
            t1 = time.time()
            elapsed = t1 - t0
            rp.print_row(s, n, elapsed, float(rate))
            csv_recorder.single_COBA_data_add('fcnmv', data_type, backend, 'post', conn_num, s, elapsed, float(rate), dur_ms)
    csv_recorder.record_finish('default')

def benchmark_pre_conn(conn_num=80, data_type='binary', duration=1e2 * u.ms):
    print('Benchmarking pre-synaptic connection updates...')

    csv_recorder = CSV_record('binary_pre', 'fcnmv', 'coba')
    dur_ms = float(duration / u.ms)
    for backend in backends:
        brainevent.config.set_backend('gpu', backend)
        rp.print_header(operator='fcnmv', data_type=data_type, backend=backend,
                        mode='pre', conn_num=conn_num, duration_ms=dur_ms)
        rp.print_table_header()
        for s in scales:
            run = make_simulation_run(
                scale=s,
                data_type=data_type,
                efferent_target='pre',
                duration=duration,
                conn_num=conn_num,
            )

            jax.block_until_ready(run())

            t0 = time.time()
            n, rate = jax.block_until_ready(run())
            t1 = time.time()
            elapsed = t1 - t0
            rp.print_row(s, n, elapsed, float(rate))
            csv_recorder.single_COBA_data_add('fcnmv', data_type, backend, 'pre', conn_num, s, elapsed, float(rate), dur_ms)

    csv_recorder.record_finish('default')


if __name__ == '__main__':
    #benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms)
    #benchmark_post_conn(conn_num=80, data_type='bitpack', duration=1e4 * u.ms)
    benchmark_pre_conn()
