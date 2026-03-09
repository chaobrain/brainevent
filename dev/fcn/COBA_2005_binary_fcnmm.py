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

batch_size, conn_num, data_type, duration = 16, 80, 'binary', 2e3 * u.ms


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
        print(f"  scale={s:>3d}, neurons={n:>6d}, "
              f"time={elapsed:>8.3f}s, rate={rate:.1f} Hz")


def bench_fcnmm():
    # Binary FCNMM Kernel Benchmark, 2026/03/08
    # ======================================================================
    # 
    #   batch_size=16, conn_num=16 (warp kernel) [post-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.158s, rate=53.5 Hz
    #   scale=  4, neurons= 16000, time=   0.451s, rate=53.5 Hz
    #   scale= 10, neurons= 40000, time=   1.118s, rate=53.5 Hz
    #   scale= 40, neurons=160000, time=   4.851s, rate=53.5 Hz
    #   scale=100, neurons=400000, time=  12.255s, rate=53.5 Hz
    # 
    #   batch_size=16, conn_num=32 (warp kernel) [post-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.197s, rate=54.2 Hz
    #   scale=  4, neurons= 16000, time=   0.502s, rate=54.2 Hz
    #   scale= 10, neurons= 40000, time=   1.144s, rate=54.2 Hz
    #   scale= 40, neurons=160000, time=   5.175s, rate=54.2 Hz
    #   scale=100, neurons=400000, time=  13.472s, rate=54.2 Hz
    # 
    #   batch_size=32, conn_num=16 (warp kernel) [post-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.207s, rate=53.5 Hz
    #   scale=  4, neurons= 16000, time=   0.802s, rate=53.5 Hz
    #   scale= 10, neurons= 40000, time=   2.146s, rate=53.5 Hz
    #   scale= 40, neurons=160000, time=   8.700s, rate=53.5 Hz
    #   scale=100, neurons=400000, time=  22.231s, rate=53.5 Hz
    # 
    #   batch_size=32, conn_num=32 (warp kernel) [post-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.202s, rate=54.2 Hz
    #   scale=  4, neurons= 16000, time=   0.799s, rate=54.2 Hz
    #   scale= 10, neurons= 40000, time=   2.286s, rate=54.2 Hz
    #   scale= 40, neurons=160000, time=   9.847s, rate=54.2 Hz
    #   scale=100, neurons=400000, time=  25.579s, rate=54.2 Hz
    # 
    #   batch_size=16, conn_num=80 (basic kernel) [post-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.223s, rate=59.5 Hz
    #   scale=  4, neurons= 16000, time=   0.651s, rate=59.5 Hz
    #   scale= 10, neurons= 40000, time=   1.742s, rate=59.4 Hz
    #   scale= 40, neurons=160000, time=   8.071s, rate=59.4 Hz
    #   scale=100, neurons=400000, time=  21.558s, rate=59.4 Hz
    # 
    #   batch_size=16, conn_num=128 (basic kernel) [post-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.226s, rate=70.6 Hz
    #   scale=  4, neurons= 16000, time=   0.653s, rate=70.6 Hz
    #   scale= 10, neurons= 40000, time=   1.763s, rate=70.6 Hz
    #   scale= 40, neurons=160000, time=   9.650s, rate=70.6 Hz
    #   scale=100, neurons=400000, time=  28.570s, rate=70.6 Hz
    # 
    #   batch_size=32, conn_num=80 (basic kernel) [post-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.350s, rate=59.4 Hz
    #   scale=  4, neurons= 16000, time=   1.318s, rate=59.4 Hz
    #   scale= 10, neurons= 40000, time=   3.556s, rate=59.4 Hz
    #   scale= 40, neurons=160000, time=  15.561s, rate=59.4 Hz
    #   scale=100, neurons=400000, time=  41.009s, rate=59.4 Hz
    # 
    #   batch_size=32, conn_num=128 (basic kernel) [post-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.344s, rate=70.6 Hz
    #   scale=  4, neurons= 16000, time=   1.262s, rate=70.6 Hz
    #   scale= 10, neurons= 40000, time=   3.759s, rate=70.6 Hz
    #   scale= 40, neurons=160000, time=  21.000s, rate=70.6 Hz
    #   scale=100, neurons=400000, time=  68.620s, rate=70.6 Hz
    # 
    #   batch_size=64, conn_num=80 (basic kernel) [post-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   1.220s, rate=59.4 Hz
    #   scale=  4, neurons= 16000, time=   5.679s, rate=59.4 Hz
    #   scale= 10, neurons= 40000, time=  16.934s, rate=59.4 Hz
    #   scale= 40, neurons=160000, time=  78.643s, rate=59.4 Hz
    #   scale=100, neurons=400000, time= 205.231s, rate=59.4 Hz
    # 
    # 
    # ######################################################################
    # # Scatter mode (pre-synaptic)
    # ######################################################################
    # 
    #   batch_size=16, conn_num=16 (warp kernel) [pre-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.107s, rate=53.8 Hz
    #   scale=  4, neurons= 16000, time=   0.110s, rate=53.8 Hz
    #   scale= 10, neurons= 40000, time=   0.249s, rate=53.8 Hz
    #   scale= 40, neurons=160000, time=   1.565s, rate=53.9 Hz
    #   scale=100, neurons=400000, time=   4.551s, rate=53.9 Hz
    # 
    #   batch_size=16, conn_num=80 (basic kernel) [pre-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.079s, rate=73.6 Hz
    #   scale=  4, neurons= 16000, time=   0.294s, rate=73.5 Hz
    #   scale= 10, neurons= 40000, time=   1.102s, rate=73.6 Hz
    #   scale= 40, neurons=160000, time=   6.170s, rate=73.6 Hz
    #   scale=100, neurons=400000, time=  18.775s, rate=73.6 Hz
    # 
    #   batch_size=32, conn_num=80 (basic kernel) [pre-synaptic]
    # -------
    #   scale=  1, neurons=  4000, time=   0.116s, rate=73.6 Hz
    #   scale=  4, neurons= 16000, time=   0.494s, rate=73.6 Hz
    #   scale= 10, neurons= 40000, time=   1.349s, rate=73.6 Hz
    #   scale= 40, neurons=160000, time=  12.581s, rate=73.6 Hz
    #   scale=100, neurons=400000, time=  43.576s, rate=73.6 Hz
    # 

    # Binary FCNMM Kernel Benchmark, 2026/03/09
    # ======================================================================
    #
    # ======================================================================
    #   batch_size=16, conn_num=16 (warp kernel) [post-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.165s, rate=53.5 Hz
    #   scale=  4, neurons= 16000, time=   0.381s, rate=53.5 Hz
    #   scale= 10, neurons= 40000, time=   0.948s, rate=53.5 Hz
    #   scale= 40, neurons=160000, time=   4.219s, rate=53.5 Hz
    #   scale=100, neurons=400000, time=  10.843s, rate=53.5 Hz
    #
    # ======================================================================
    #   batch_size=16, conn_num=32 (warp kernel) [post-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.196s, rate=54.2 Hz
    #   scale=  4, neurons= 16000, time=   0.482s, rate=54.2 Hz
    #   scale= 10, neurons= 40000, time=   1.014s, rate=54.2 Hz
    #   scale= 40, neurons=160000, time=   4.761s, rate=54.2 Hz
    #   scale=100, neurons=400000, time=  12.749s, rate=54.2 Hz
    #
    # ======================================================================
    #   batch_size=32, conn_num=16 (warp kernel) [post-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.197s, rate=53.5 Hz
    #   scale=  4, neurons= 16000, time=   0.788s, rate=53.5 Hz
    #   scale= 10, neurons= 40000, time=   2.038s, rate=53.5 Hz
    #   scale= 40, neurons=160000, time=   8.396s, rate=53.5 Hz
    #   scale=100, neurons=400000, time=  21.495s, rate=53.5 Hz
    #
    # ======================================================================
    #   batch_size=32, conn_num=32 (warp kernel) [post-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.218s, rate=54.2 Hz
    #   scale=  4, neurons= 16000, time=   0.794s, rate=54.2 Hz
    #   scale= 10, neurons= 40000, time=   2.192s, rate=54.2 Hz
    #   scale= 40, neurons=160000, time=   9.736s, rate=54.2 Hz
    #   scale=100, neurons=400000, time=  25.324s, rate=54.2 Hz
    #
    # ======================================================================
    #   batch_size=16, conn_num=80 (basic kernel) [post-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.215s, rate=59.4 Hz
    #   scale=  4, neurons= 16000, time=   0.579s, rate=59.4 Hz
    #   scale= 10, neurons= 40000, time=   1.446s, rate=59.4 Hz
    #   scale= 40, neurons=160000, time=   7.382s, rate=59.4 Hz
    #   scale=100, neurons=400000, time=  20.410s, rate=59.4 Hz
    #
    # ======================================================================
    #   batch_size=16, conn_num=128 (basic kernel) [post-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.207s, rate=70.6 Hz
    #   scale=  4, neurons= 16000, time=   0.588s, rate=70.6 Hz
    #   scale= 10, neurons= 40000, time=   1.502s, rate=70.6 Hz
    #   scale= 40, neurons=160000, time=   9.337s, rate=70.6 Hz
    #   scale=100, neurons=400000, time=  28.471s, rate=70.6 Hz
    #
    # ======================================================================
    #   batch_size=32, conn_num=80 (basic kernel) [post-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.236s, rate=59.4 Hz
    #   scale=  4, neurons= 16000, time=   0.923s, rate=59.4 Hz
    #   scale= 10, neurons= 40000, time=   2.891s, rate=59.4 Hz
    #   scale= 40, neurons=160000, time=  14.404s, rate=59.4 Hz
    #   scale=100, neurons=400000, time=  39.242s, rate=59.4 Hz
    #
    # ======================================================================
    #   batch_size=32, conn_num=128 (basic kernel) [post-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.243s, rate=70.6 Hz
    #   scale=  4, neurons= 16000, time=   0.964s, rate=70.6 Hz
    #   scale= 10, neurons= 40000, time=   3.355s, rate=70.6 Hz
    #   scale= 40, neurons=160000, time=  20.675s, rate=70.6 Hz
    #   scale=100, neurons=400000, time=  58.189s, rate=70.6 Hz
    #
    # ======================================================================
    #   batch_size=64, conn_num=80 (basic kernel) [post-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.501s, rate=59.5 Hz
    #   scale=  4, neurons= 16000, time=   2.040s, rate=59.4 Hz
    #   scale= 10, neurons= 40000, time=   6.091s, rate=59.4 Hz
    #   scale= 40, neurons=160000, time=  30.282s, rate=59.4 Hz
    #   scale=100, neurons=400000, time=  80.439s, rate=59.4 Hz
    #
    #
    # ######################################################################
    # # Scatter mode (pre-synaptic)
    # ######################################################################
    #
    # ======================================================================
    #   batch_size=16, conn_num=16 (warp kernel) [pre-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.089s, rate=53.9 Hz
    #   scale=  4, neurons= 16000, time=   0.104s, rate=53.9 Hz
    #   scale= 10, neurons= 40000, time=   0.212s, rate=53.8 Hz
    #   scale= 40, neurons=160000, time=   0.734s, rate=53.9 Hz
    #   scale=100, neurons=400000, time=   1.781s, rate=53.9 Hz
    #
    # ======================================================================
    #   batch_size=16, conn_num=80 (basic kernel) [pre-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.116s, rate=73.5 Hz
    #   scale=  4, neurons= 16000, time=   0.381s, rate=73.6 Hz
    #   scale= 10, neurons= 40000, time=   0.841s, rate=73.6 Hz
    #   scale= 40, neurons=160000, time=   2.992s, rate=73.6 Hz
    #   scale=100, neurons=400000, time=   8.067s, rate=73.6 Hz
    #
    # ======================================================================
    #   batch_size=32, conn_num=80 (basic kernel) [pre-synaptic]
    # ======================================================================
    #   scale=  1, neurons=  4000, time=   0.137s, rate=73.6 Hz
    #   scale=  4, neurons= 16000, time=   0.360s, rate=73.6 Hz
    #   scale= 10, neurons= 40000, time=   0.947s, rate=73.6 Hz
    #   scale= 40, neurons=160000, time=   3.346s, rate=73.6 Hz
    #   scale=100, neurons=400000, time=   8.856s, rate=73.6 Hz



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
