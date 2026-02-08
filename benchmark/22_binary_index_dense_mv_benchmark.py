# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax

import brainstate
from utils import visualize
import brainevent


def forward(n_pre, n_post, spk_prob):
    weight = brainstate.random.randn(n_pre, n_post)
    spike = (brainstate.random.rand(n_pre) < spk_prob)

    spike = spike.astype(float)

    indices = brainstate.random.randn(n_pre).astype(int)

    count = 0
    for i in range(spike.shape[0]):
        if spike[i] != 0.:
            indices = indices.at[count].set(i)
            count = count + 1

    @jax.jit
    def f1(spikes, weights):
        return (
            brainevent.BinaryArray(spikes, indexed=True) @ weights
        )

    @jax.jit
    def f2(spikes, weights):
        return (
            (spikes @ weights)
        )

    y1 = jax.block_until_ready(f1(spike, weight))
    y2 = jax.block_until_ready(f2(spike, weight))
    print('max difference:', jax.numpy.abs(y1 - y2).max())

    # warm up
    for _ in range(20):
        y1 = jax.block_until_ready(f1(spike, weight))
        y2 = jax.block_until_ready(f2(spike, weight))

    n = 100
    t0 = time.time()
    for _ in range(n):
        y1 = jax.block_until_ready(f1(spike, weight))
    r1 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post},  spike probability: {spk_prob}, Linear: {r1} s")

    t0 = time.time()
    for _ in range(n):
        y2 = jax.block_until_ready(f2(spike, weight))
    r2 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post},  spike probability: {spk_prob}, Matmul: {r2} s")

    ratio = (r2 / r1 - 1.) if r2 > r1 else -(r1 / r2 - 1.)
    print('Acceleration ratio:', ratio)
    print()
    return ratio


def benchmark_forward(prob=0.1):
    # platform = brainstate.environ.get_platform()

    results = {}
    for transpose in [True, False]:
        for n_pre, n_post in [
            (1000, 10000),
            (10000, 1000),
            (10000, 10000),
            (20000, 10000),
            (10000, 20000),
            (20000, 20000),
            (10000, 100000),
            (1000, 1000),
            (1000, 10000),
            (2000, 20000),
            (4000, 40000),
            (10000, 20000),
        ]:
            results[f'{n_pre}x{n_post}x{transpose}'] = forward(n_pre, n_post, prob)

    visualize(
        results,
        title=f'Acceleration Ratio (p={prob})',
        # filename=f'results/event-mv-transpose={transpose}-prob={prob}-{platform}.pdf'
    )


if __name__ == '__main__':
    # benchmark_forward(0.1)
    # benchmark_forward(0.01)
    benchmark_forward(0.01)
