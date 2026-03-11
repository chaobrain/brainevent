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


import time

import brainunit as u
import jax

import brainevent
from COBA_2005_benchmark import make_simulation_run


# brainevent.config.set_backend('gpu', 'warp')

# C:\Users\adadu\miniconda3\envs\brainx\python.exe D:\codes\projects\brainevent\dev\csr\COBA_2005_binary_csrmv.py
# Benchmarking post-synaptic connection updates...
# scale=1, size=4000, time = 9.541418552398682 s, firing rate = 59.56615447998047 Hz
# scale=2, size=8000, time = 14.272186756134033 s, firing rate = 59.56580352783203 Hz
# scale=4, size=16000, time = 17.181252479553223 s, firing rate = 59.57089614868164 Hz
# scale=6, size=24000, time = 24.311490297317505 s, firing rate = 59.569007873535156 Hz
# scale=8, size=32000, time = 37.761569023132324 s, firing rate = 59.57065200805664 Hz
# scale=10, size=40000, time = 43.324846267700195 s, firing rate = 59.56809997558594 Hz
# scale=20, size=80000, time = 57.94103240966797 s, firing rate = 59.570098876953125 Hz
# Benchmarking pre-synaptic connection updates...
# scale=1, size=4000, time = 62.55446171760559 s, firing rate = 74.62295532226562 Hz
# scale=2, size=8000, time = 118.97933387756348 s, firing rate = 74.49755096435547 Hz
# scale=4, size=16000, time = 237.37970495224 s, firing rate = 74.56958770751953 Hz
# scale=6, size=24000, time = 365.3048918247223 s, firing rate = 74.51143646240234 Hz
# scale=8, size=32000, time = 489.07046341896057 s, firing rate = 74.61072540283203 Hz
# scale=10, size=40000, time = 622.9728593826294 s, firing rate = 74.59778594970703 Hz
# scale=20, size=80000, time = 1306.9135026931763 s, firing rate = 74.61448669433594 Hz
#
# Process finished with exit code 0


scales = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]


def benchmark_post_conn(
    conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='cuda_raw'
):
    brainevent.config.set_backend('gpu', backend)
    print(
        f'Benchmarking post-synaptic connection, '
        f'conn_num={conn_num}, '
        f'data_type={data_type}, '
        f'duration={duration}, '
        f'backend={backend}'
    )

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
        print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')


def benchmark_pre_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='cuda_raw'):
    brainevent.config.set_backend('gpu', backend)
    print(
        f'Benchmarking pre-synaptic connection, '
        f'conn_num={conn_num}, '
        f'data_type={data_type}, '
        f'duration={duration}, '
        f'backend={backend}'
    )

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
        print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')


if __name__ == '__main__':
    benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='warp')
    benchmark_pre_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='warp')

    benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='cuda_raw')
    benchmark_pre_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='cuda_raw')

    benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='jax_raw')
    benchmark_pre_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='jax_raw')

    benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='cusparse')
    benchmark_pre_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='cusparse')

    benchmark_post_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='pallas')
    benchmark_pre_conn(conn_num=80, data_type='binary', duration=1e4 * u.ms, backend='pallas')
