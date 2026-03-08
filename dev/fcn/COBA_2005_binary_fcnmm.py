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


batch_size, conn_num, data_type, duration = 16, 80, 'binary', 1e3 * u.ms


def benchmark_post_conn():
    # --------------------------------
    # 2026/03/08, batch_size, conn_num, data_type, duration = 16, 80, 'binary', 1e3 * u.ms
    # --------------------------------
    #
    # scale=1, size=4000, time = 0.2206423282623291 s, firing rate = 59.4550666809082 Hz
    # scale=2, size=8000, time = 0.31554341316223145 s, firing rate = 59.44181442260742 Hz
    # scale=4, size=16000, time = 0.6682348251342773 s, firing rate = 59.4439697265625 Hz
    # scale=6, size=24000, time = 0.9589388370513916 s, firing rate = 59.44413375854492 Hz
    # scale=8, size=32000, time = 1.2952101230621338 s, firing rate = 59.44554901123047 Hz
    # scale=10, size=40000, time = 1.708672285079956 s, firing rate = 59.44718551635742 Hz
    # scale=20, size=80000, time = 3.7858314514160156 s, firing rate = 59.445011138916016 Hz
    # scale=40, size=160000, time = 7.926112413406372 s, firing rate = 59.444637298583984 Hz
    # scale=60, size=240000, time = 12.231749296188354 s, firing rate = 59.444488525390625 Hz
    # scale=80, size=320000, time = 16.75143575668335 s, firing rate = 59.44343566894531 Hz
    # scale=100, size=400000, time = 21.40729546546936 s, firing rate = 59.444156646728516 Hz
    #
    #

    print('Benchmarking post-synaptic connection updates...')
    brainevent.config.set_backend('gpu', 'cuda_raw')

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
          f"({'warp' if conn_num <= 32 else 'basic'} kernel) "
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
        print(f"  scale={s:>3d}, neurons={n:>6d}, time={elapsed:>8.3f}s, rate={rate:.1f} Hz")


def bench_fcnmm():
    #

    brainevent.config.set_backend('gpu', 'cuda_raw')

    print("Binary FCNMM Kernel Benchmark")
    print("=" * 70)

    # Configurations to test
    CONFIGS = [
        # (batch_size, conn_num, description)
        # --- Warp kernel path (conn_num <= 32) ---
        (16, 16, "warp, small batch"),
        (16, 32, "warp, boundary"),
        (32, 16, "warp, large batch"),
        (32, 32, "warp, large batch boundary"),
        # --- Basic kernel path (conn_num > 32) ---
        (16, 80, "basic, default"),
        (16, 128, "basic, large conn"),
        (32, 80, "basic, large batch"),
        (32, 128, "basic, large batch+conn"),
        (64, 80, "basic, very large batch"),
    ]
    # Test gather mode (post-synaptic / transpose=True)
    for batch_size, conn_num, desc in CONFIGS:
        run_benchmark(batch_size, conn_num, mode='post')

    # Test scatter mode (pre-synaptic / transpose=False) for a subset
    print("\n\n" + "#" * 70)
    print("# Scatter mode (pre-synaptic)")
    print("#" * 70)
    scatter_configs = [
        (16, 16, "warp"),
        (16, 80, "basic"),
        (32, 80, "basic large batch"),
    ]
    for batch_size, conn_num, desc in scatter_configs:
        run_benchmark(batch_size, conn_num, mode='pre')


if __name__ == '__main__':
    # benchmark_post_conn()
    bench_fcnmm()
    # benchmark_pre_conn()
