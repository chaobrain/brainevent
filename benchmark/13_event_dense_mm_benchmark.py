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

from brainevent._dense_impl_binary import (
    dense_mat_dot_binary_mat,
    binary_mat_dot_dense_mat,
)
import brainstate
from utils import visualize
# brainstate.environ.set_platform('cpu')
# brainevent.config.gpu_kernel_backend = 'pallas'


def matrix_event(m, k, n, spk_prob, as_float: bool, transpose: bool, n_run = 100):
    if transpose:
        weight = brainstate.init.KaimingUniform()((m, k))
        spike = (brainstate.random.rand(k, n) < spk_prob)
    else:
        spike = (brainstate.random.rand(m, k) < spk_prob)
        weight = brainstate.init.KaimingUniform()((k, n))
    if as_float:
        spike = spike.astype(float)

    @jax.jit
    def f1(spike, weight):
        return (
            dense_mat_dot_binary_mat(weight, spike)
            if transpose
            else binary_mat_dot_dense_mat(spike, weight)
        )

    @jax.jit
    def f2(spike, weight):
        return (
            weight @ spike
            if transpose
            else spike @ weight
        )

    y1 = jax.block_until_ready(f1(spike, weight))
    y2 = jax.block_until_ready(f2(spike, weight))
    print('max difference:', jax.numpy.abs(y1 - y2).max())

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f1(spike, weight))
    r1 = time.time() - t0
    print(f"m: {m}, k: {k}, n: {n}, spike probability: {spk_prob}, Linear: {r1} s")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f2(spike, weight))
    r2 = time.time() - t0
    print(f"m: {m}, k: {k}, n: {n}, spike probability: {spk_prob}, Matmul: {r2} s")

    ratio = (r2 / r1 - 1.) if r2 > r1 else -(r1 / r2 - 1.)
    print('Acceleration ratio:', ratio)
    print()
    return ratio


def benchmark(prob=0.1, transpose=False):
    results = {}
    for m, k, n in [
        # (100, 100, 100),
        # (500, 500, 500),
        # (1000, 1000, 1000),
        # (2000, 2000, 1000),
        # (1000, 1000, 10000),
        (10000, 10000, 10000),
        # (20000, 20000, 10000),
        (10000, 10000, 20000),
    ]:
        key = f'{m}x{k}x{n}xtranspose' if transpose else f'{m}x{k}x{n}'
        results[key] = matrix_event(m, k, n, prob, as_float=False, transpose=transpose)

    visualize(
        results,
        title=f'Acceleration Ratio (p={prob})',
        # filename=f'results/matrix-event-prob={prob}-{platform}.pdf'
    )


if __name__ == '__main__':
    # benchmark(0.1, transpose=True)
    # benchmark(0.01, transpose=True)
    # benchmark(0.01, transpose=False)
    benchmark(0.001, transpose=True)
    benchmark(0.001, transpose=False)

