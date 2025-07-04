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


# n_pre: 1000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.004549980163574219 s
# n_pre: 1000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 0.04318690299987793 s
# Acceleration ratio: 8.491668413330538
#
# n_pre: 1000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.005620718002319336 s
# n_pre: 1000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 1.3311548233032227 s
# Acceleration ratio: 235.83003181336161
#
# n_pre: 10000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.015388727188110352 s
# n_pre: 10000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 10.791011333465576 s
# Acceleration ratio: 700.2283213262065
#
# n_pre: 10000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.01043844223022461 s
# n_pre: 10000, n_post: 1000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 0.8944694995880127 s
# Acceleration ratio: 84.68994107167329
#
# n_pre: 10000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.021282196044921875 s
# n_pre: 10000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 21.388156414031982 s
# Acceleration ratio: 1003.9788268506901
#
# n_pre: 20000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.025498151779174805 s
# n_pre: 20000, n_post: 10000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 21.211663246154785 s
# Acceleration ratio: 830.8902259997943
#
# n_pre: 20000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.044051408767700195 s
# n_pre: 20000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 42.31502842903137 s
# Acceleration ratio: 959.5828647200498
#
# n_pre: 20000, n_post: 30000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.06666803359985352 s
# n_pre: 20000, n_post: 30000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 62.46805453300476 s
# Acceleration ratio: 936.0016057162067
#
# n_pre: 30000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Linear: 0.08313393592834473 s
# n_pre: 30000, n_post: 20000, conn_prob: 0.01, spk_prob: 0.01, Matmul: 63.61667847633362 s
# Acceleration ratio: 764.231163013459
#
#


import os

# os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
#
# jax.config.update('jax_cpu_enable_async_dispatch', False)

import time
import brainstate
from utils import visualize
import brainevent


# brainstate.environ.set(platform='cpu')


def event_matrix(n_pre, n_post, conn_prob, spk_prob, as_float: bool, transpose: bool):
    n_conn = int(conn_prob * n_post)
    weight = 1.
    weight = brainstate.random.randn(n_pre, n_conn)
    index = brainstate.random.randint(0, n_post, (n_pre, n_conn))
    csr = brainevent.FixedPostNumConn((weight, index), shape=(n_pre, n_post))
    spike = (brainstate.random.rand(n_pre if transpose else n_post) < spk_prob)

    if as_float:
        spike = spike.astype(float)

    @jax.jit
    def f1(spike):
        if transpose:
            return brainevent.EventArray(spike) @ csr
        else:
            return csr @ brainevent.EventArray(spike)

    weight2 = brainstate.init.Normal()([n_pre, n_post])

    @jax.jit
    def f2(spike):
        if transpose:
            return spike @ weight2
        else:
            return weight2 @ spike

    y1 = jax.block_until_ready(f1(spike))
    y2 = jax.block_until_ready(f2(spike))
    # print('max difference:', jax.numpy.abs(y1 - y2).max())

    n = 1000
    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f1(spike))
    r1 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post}, conn_prob: {conn_prob}, spk_prob: {spk_prob}, Linear: {r1} s")

    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f2(spike))
    r2 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post}, conn_prob: {conn_prob}, spk_prob: {spk_prob}, Matmul: {r2} s")

    ratio = (r2 / r1 - 1.) if r2 > r1 else -(r1 / r2 - 1.)
    print('Acceleration ratio:', ratio)
    print()
    return ratio


def benchmark_event_matrix(transpose: bool, conn_prob=0.01, spk_prob=0.01):
    platform = brainstate.environ.get_platform()
    results = {}
    for n_pre, n_post in [
        # (1000, 1000),
        (1000, 10000),
        (10000, 10000),
        (10000, 1000),
        (10000, 20000),
        (20000, 10000),
        (20000, 20000),
        (20000, 30000),
        (30000, 20000),
    ]:
        r = event_matrix(n_pre, n_post, conn_prob, spk_prob, False, transpose)
        results[f'{n_pre}x{n_post}'] = r

    visualize(
        results,
        title=f'Acceleration Ratio (conn_prob={conn_prob}, spk_prob={spk_prob})',
        filename=f'results/fixed_num_conn-transpose={transpose}-conn_prob={spk_prob}-spk_prob={spk_prob}-{platform}.pdf'
    )


if __name__ == '__main__':
    pass

    benchmark_event_matrix(False, spk_prob=0.1)
    benchmark_event_matrix(True, spk_prob=0.1)
