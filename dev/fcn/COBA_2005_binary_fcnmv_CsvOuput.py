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


scales = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]
backends = ['cuda_raw','jax_raw']

conn_nums = [20, 40, 80, 160, 320, 640, ]

probs = [0.001, 0.004,  0.016 ,0.064, 0.128, 0.256]


def memory_limit( conn_nums, scale:int ,_N:int = 4000 , limit: int = 16):
    if conn_nums * 4 * scale * _N > limit * (1024 ** 3):
        return True
    else:
        return False


def benchmark_post_conn(
    conn_num=None,
    conn_prob=None,
    data_type='binary',
    duration=1e4 * u.ms,
    homo: bool = True,
    backend: str | None = None,
    probs_or_conn='conn',
    _N : int = 4000
):
    import CsvOutput as RP

    print('Benchmarking post-synaptic connection updates...')

    backends_to_use = [backend] if backend is not None else backends

    if probs_or_conn == 'conn':
        use_conn_nums = True
        conn_nums_to_use = [conn_num] if conn_num is not None else conn_nums
    else:
        use_conn_nums = False
        probs_to_use = [conn_prob] if conn_prob is not None else probs

    csv_recorder = RP.CSV_record('binary_post', 'fcnmv', 'coba', duration=duration, conn=conn_num)

    for back in backends_to_use:
        brainevent.config.set_backend('gpu', back)

        if use_conn_nums:
            for cn in conn_nums_to_use:
                csv_recorder.print_header(operator='fcnmv', data_type=data_type, backend=back,
                        mode='post', conn_num=cn, duration=duration,
                        homo=('homo' if homo else 'hetero'))
                csv_recorder.print_table_header()

                for s in scales:
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
                        csv_recorder.print_row(s, n, elapsed, float(rate))
                        csv_recorder.single_COBA_data_add('fcnmv', data_type, back, 'post', cn, s, elapsed, float(rate), duration, homo=('homo' if homo else 'hetero'))
                    except Exception as e:
                        print(f'  [Error] scale={s}, conn_num={cn}: {e}')
                        continue
        else:
            for prob in probs_to_use:
                csv_recorder.print_header(operator='fcnmv', data_type=data_type, backend=back,
                        mode='post', duration=duration,
                        homo=('homo' if homo else 'hetero'), prob=prob)
                csv_recorder.print_table_header(show_conn=True)

                for s in scales:
                    actual_conn_num = int(s * prob * _N)
                    if actual_conn_num < 1 : actual_conn_num = 1
                    if memory_limit(actual_conn_num, scale=s): continue
                    try:
                        run = make_simulation_run(
                            scale=s,
                            data_type=data_type,
                            efferent_target='post',
                            duration=duration,
                            conn_num=actual_conn_num,
                            homo=homo
                        )

                        jax.block_until_ready(run())

                        t0 = time.time()
                        n, rate = jax.block_until_ready(run())
                        t1 = time.time()
                        elapsed = t1 - t0
                        csv_recorder.print_row(s, n, elapsed, float(rate), conn_num=actual_conn_num)
                        csv_recorder.single_COBA_data_add('fcnmv', data_type, back, 'post', actual_conn_num, s, elapsed, float(rate), duration, homo=('homo' if homo else 'hetero'))
                    except Exception as e:
                        print(f'  [Error] scale={s}, conn_num={actual_conn_num}: {e}')
                        continue

    csv_recorder.record_finish('post_cuda-jax_great_scale')

def benchmark_pre_conn(
        conn_num=None, 
        conn_prob=None,
        data_type='binary', 
        duration=1e2 * u.ms, 
        homo:bool = True, 
        backend: str | None = None,
        probs_or_conn='conn',
        _N : int = 4000
        ):
    print('Benchmarking pre-synaptic connection updates...')
    import CsvOutput as RP

    backends_to_use = [backend] if backend is not None else backends

    if probs_or_conn == 'conn':
        use_conn_nums = True
        conn_nums_to_use = [conn_num] if conn_num is not None else conn_nums
    else:
        use_conn_nums = False
        probs_to_use = [conn_prob] if conn_prob is not None else probs

    csv_recorder = RP.CSV_record('binary_pre', 'fcnmv', 'coba', duration=duration, conn=conn_num)

    for back in backends_to_use:
        brainevent.config.set_backend('gpu', back)

        if use_conn_nums:
            for cn in conn_nums_to_use:
                csv_recorder.print_header(operator='fcnmv', data_type=data_type, backend=back,
                        mode='pre', conn_num=cn, duration=duration,
                        homo=('homo' if homo else 'hetero'))
                csv_recorder.print_table_header()

                for s in scales:
                    try:
                        run = make_simulation_run(
                            scale=s,
                            data_type=data_type,
                            efferent_target='pre',
                            duration=duration,
                            conn_num=cn,
                            homo=homo,
                        )

                        jax.block_until_ready(run())

                        t0 = time.time()
                        n, rate = jax.block_until_ready(run())
                        t1 = time.time()
                        elapsed = t1 - t0
                        csv_recorder.print_row(s, n, elapsed, float(rate))
                        csv_recorder.single_COBA_data_add('fcnmv', data_type, back, 'pre', cn, s, elapsed, float(rate), duration, homo=('homo' if homo else 'hetero'))
                    except Exception as e:
                        print(f'  [Error] scale={s}, conn_num={cn}: {e}')
                        continue
        else:
            for prob in probs_to_use:
                csv_recorder.print_header(operator='fcnmv', data_type=data_type, backend=back,
                        mode='pre', duration=duration,
                        homo=('homo' if homo else 'hetero'), prob=prob)
                csv_recorder.print_table_header(show_conn=True)

                for s in scales:
                    actual_conn_num = int(s * prob * _N)  # non-linear: conn_num scales with network size
                    if actual_conn_num < 1 : actual_conn_num = 1
                    if memory_limit(actual_conn_num, scale=s): continue
                    try:
                        run = make_simulation_run(
                            scale=s,
                            data_type=data_type,
                            efferent_target='pre',
                            duration=duration,
                            conn_num=actual_conn_num,
                            homo=homo,
                        )

                        jax.block_until_ready(run())

                        t0 = time.time()
                        n, rate = jax.block_until_ready(run())
                        t1 = time.time()
                        elapsed = t1 - t0
                        csv_recorder.print_row(s, n, elapsed, float(rate), conn_num=actual_conn_num)
                        csv_recorder.single_COBA_data_add('fcnmv', data_type, back, 'pre', actual_conn_num, s, elapsed, float(rate), duration, homo=('homo' if homo else 'hetero'))
                    except Exception as e:
                        print(f'  [Error] scale={s}, conn_num={actual_conn_num}: {e}')
                        continue

    csv_recorder.record_finish('test')


if __name__ == '__main__':
    #benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='jax_raw')
    benchmark_post_conn(data_type='binary', duration=1e2 * u.ms, probs_or_conn='prob')
    #benchmark_pre_conn(conn_num=80, data_type='bitpack', duration=1e3 * u.ms)
    #benchmark_pre_conn(data_type='binary',duration=1e3 * u.ms,)