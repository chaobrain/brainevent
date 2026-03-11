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
from COBA_2005_benchmark import make_simulation_batch_run

brainevent.config.set_backend('gpu', 'cuda_raw')

batch_size, conn_num, data_type, duration = 16, 80, 'binary', 1e3 * u.ms


def benchmark_post_conn():
    print('Benchmarking post-synaptic connection updates...')

    for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
        run = make_simulation_batch_run(
            scale=s,
            batch_size=batch_size,
            data_type=data_type,
            efferent_target='post',
            duration=duration,
            conn_num=conn_num,
        )

        jax.block_until_ready(run())

        t0 = time.time()
        n, rate = jax.block_until_ready(run())
        t1 = time.time()
        print(f'scale={s}, size={n}, time = {t1 - t0} s, firing rate = {rate} Hz')


def benchmark_pre_conn():
    print('Benchmarking pre-synaptic connection updates...')

    for s in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
        run = make_simulation_batch_run(
            scale=s,
            batch_size=batch_size,
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


def run_benchmark(batch_size, conn_num, mode='post'):

    print(f"\n{'=' * 70}")
    print(f"  batch_size={batch_size}, conn_num={conn_num} "
          f"[{mode}-synaptic]")
    print(f"{'=' * 70}")

    # Scales to benchmark (network sizes: scale * 4000 neurons)
    SCALES = [1, 4, 10, 40, 100]

    for s in SCALES:
        dur = 1e3 * u.ms if mode == 'post' else 1e2 * u.ms
        run = make_simulation_batch_run(
            scale=s,
            batch_size=batch_size,
            data_type='binary',
            efferent_target=mode,
            duration=dur,
            conn_num=conn_num,
        )

        # Warmup
        jax.block_until_ready(run())

        # Timed run
        t0 = time.time()
        n, rate = jax.block_until_ready(run())
        t1 = time.time()
        elapsed = t1 - t0
        print(f"  scale={s:>3d}, neurons={n:>6d}, "
              f"time={elapsed:>8.3f}s, rate={rate:.1f} Hz")


def bench_csrmm():
    brainevent.config.set_backend('gpu', 'cuda_raw')

    print("Binary CSRMM Kernel Benchmark")
    print("=" * 70)

    # Configurations to test
    CONFIGS = [
        # (batch_size, conn_num)
        (16, 80, "default"),
        (16, 128, "large conn"),
        (32, 80, "large batch"),
        (32, 128, "large batch+conn"),
        (64, 80, "very large batch"),
    ]

    # Test gather mode (post-synaptic / transpose=True)
    for batch_size, conn_num, desc in CONFIGS:
        run_benchmark(batch_size, conn_num, mode='post')

    # Test scatter mode (pre-synaptic / transpose=False) for a subset
    print("\n\n" + "#" * 70)
    print("# Scatter mode (pre-synaptic)")
    print("#" * 70)
    scatter_configs = [
        (16, 80, "default"),
        (32, 80, "large batch"),
    ]
    for batch_size, conn_num, desc in scatter_configs:
        run_benchmark(batch_size, conn_num, mode='pre')


if __name__ == '__main__':
    benchmark_post_conn()
    # bench_csrmm()
    # benchmark_pre_conn()
