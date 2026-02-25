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

import jax

import brainevent
from COBA_2005_benchmark import make_simulation_run

brainevent.config.set_backend('gpu', 'tvmffi')


def benchmark_post_conn():
    print('Benchmarking post-synaptic connection updates...')

    for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
        run = make_simulation_run(scale=s, data_type='binary', efferent_target='post')

        jax.block_until_ready(run())

        t0 = time.time()
        n, rate = jax.block_until_ready(run())
        t1 = time.time()
        print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')


def benchmark_pre_conn():
    print('Benchmarking pre-synaptic connection updates...')

    for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
        run = make_simulation_run(scale=s, data_type='binary', efferent_target='pre')

        jax.block_until_ready(run())

        t0 = time.time()
        n, rate = jax.block_until_ready(run())
        t1 = time.time()
        print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')


if __name__ == '__main__':
    benchmark_post_conn()
    benchmark_pre_conn()
