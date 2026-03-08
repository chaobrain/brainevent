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

brainevent.config.set_backend('gpu', 'cuda_raw')


conn_num, data_type, duration = 80, 'binary', 1e4 * u.ms
conn_num, data_type, duration = 80, 'bitpack', 1e4 * u.ms



def benchmark_post_conn():
    # --------------------------------
    # 2026/03/08, conn_num, data_type, duration = 80, 'binary', 1e4 * u.ms
    # --------------------------------
    #
    # scale=1, size=4000, time = 1.3123691082000732 s, firing rate = 59.57392883300781 Hz
    # scale=2, size=8000, time = 1.4384987354278564 s, firing rate = 59.57027816772461 Hz
    # scale=4, size=16000, time = 1.5230646133422852 s, firing rate = 59.569297790527344 Hz
    # scale=6, size=24000, time = 1.6051039695739746 s, firing rate = 59.56822967529297 Hz
    # scale=8, size=32000, time = 1.53263258934021 s, firing rate = 59.57051086425781 Hz
    # scale=10, size=40000, time = 1.6150670051574707 s, firing rate = 59.57093811035156 Hz
    # scale=20, size=80000, time = 2.0971567630767822 s, firing rate = 59.56829833984375 Hz
    # scale=40, size=160000, time = 2.7911770343780518 s, firing rate = 59.57014465332031 Hz
    # scale=60, size=240000, time = 4.094450235366821 s, firing rate = 59.57121276855469 Hz
    # scale=80, size=320000, time = 5.091517448425293 s, firing rate = 59.57069396972656 Hz
    # scale=100, size=400000, time = 6.501585245132446 s, firing rate = 59.56977462768555 Hz
    #
    # --------------------------------
    # 2026/03/08, conn_num, data_type, duration = 80, 'bitpack', 1e4 * u.ms
    # --------------------------------
    #
    # scale=1, size=4000, time = 1.3123691082000732 s, firing rate = 59.57392883300781 Hz
    # scale=2, size=8000, time = 1.4384987354278564 s, firing rate = 59.57027816772461 Hz
    # scale=4, size=16000, time = 1.5230646133422852 s, firing rate = 59.569297790527344 Hz
    # scale=6, size=24000, time = 1.6051039695739746 s, firing rate = 59.56822967529297 Hz
    # scale=8, size=32000, time = 1.53263258934021 s, firing rate = 59.57051086425781 Hz
    # scale=10, size=40000, time = 1.6150670051574707 s, firing rate = 59.57093811035156 Hz
    # scale=20, size=80000, time = 2.0971567630767822 s, firing rate = 59.56829833984375 Hz
    # scale=40, size=160000, time = 2.7911770343780518 s, firing rate = 59.57014465332031 Hz
    # scale=60, size=240000, time = 4.094450235366821 s, firing rate = 59.57121276855469 Hz
    # scale=80, size=320000, time = 5.091517448425293 s, firing rate = 59.57069396972656 Hz
    # scale=100, size=400000, time = 6.501585245132446 s, firing rate = 59.56977462768555 Hz
    #

    print('Benchmarking post-synaptic connection updates...')

    for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
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


def benchmark_pre_conn():
    print('Benchmarking pre-synaptic connection updates...')

    for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
        run = make_simulation_run(
            scale=s,
            data_type='binary',
            efferent_target='pre',
            duration=1e2 * u.ms,
            conn_num=conn_num,
        )

        jax.block_until_ready(run())

        t0 = time.time()
        n, rate = jax.block_until_ready(run())
        t1 = time.time()
        print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')


if __name__ == '__main__':
    benchmark_post_conn()
    # benchmark_pre_conn()
