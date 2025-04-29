# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import os
import time

import brainevent

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import sys

sys.path.append('../')

import brainstate
from utils import visualize


# brainstate.environ.set(platform='cpu')


def forward(n_pre, n_post, spk_prob, as_float: bool, transpose: bool = False):
    weight = brainstate.init.KaimingUniform()((n_pre, n_post))
    spike = (brainstate.random.rand(n_pre if transpose else n_post) < spk_prob)

    if as_float:
        spike = spike.astype(float)

    @jax.jit
    def f1(spike):
        return (
            (brainevent.EventArray(spike) @ weight)
            if transpose else
            (weight @ brainevent.EventArray(spike))
        )

    @jax.jit
    def f2(spike):
        return (
            (spike @ weight)
            if transpose else
            (weight @ spike)
        )

    y1 = jax.block_until_ready(f1(spike))
    y2 = jax.block_until_ready(f2(spike))
    print('max difference:', jax.numpy.abs(y1 - y2).max())

    n = 100
    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f1(spike))
    r1 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post}, spike probability: {spk_prob}, Linear: {r1} s")

    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f2(spike))
    r2 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post}, spike probability: {spk_prob}, Matmul: {r2} s")

    ratio = (r2 / r1 - 1.) if r2 > r1 else -(r1 / r2 - 1.)
    print('Acceleration ratio:', ratio)
    print()
    return ratio


def benchmark_forward(prob=0.1):
    platform = brainstate.environ.get_platform()

    results = {}
    for transpose in [True]:
        for n_pre, n_post in [
            # # (1000, 1000),
            # (1000, 10000),
            # (10000, 1000),
            # (10000, 10000),
            # (20000, 10000),
            # (10000, 20000),
            # (20000, 20000),
            # # (10000, 100000),
            # (1000, 1000),
            (1000, 10000),
            (2000, 20000),
            (4000, 40000),
            (10000, 20000),
            # (10000, 100000),
        ]:
            results[f'{n_pre}x{n_post}x{transpose}'] = forward(n_pre, n_post, prob, True, transpose)

    visualize(
        results,
        title=f'Acceleration Ratio (p={prob})',
        filename=f'results/event-mv-transpose={transpose}-prob={prob}-{platform}.pdf'
    )


if __name__ == '__main__':
    benchmark_forward(0.1)
    benchmark_forward(0.01)
    benchmark_forward(0.001)
