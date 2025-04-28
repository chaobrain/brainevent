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

import numpy as np
import brainstate
import matplotlib.pyplot as plt


# brainstate.environ.set(platform='cpu')


def event_matrix(m, k, n, spk_prob, as_float: bool):
    spike = (brainstate.random.rand(m, k) < spk_prob)
    weight = brainstate.init.KaimingUniform()((k, n))

    if as_float:
        spike = spike.astype(float)

    @jax.jit
    def f1(spike, weight):
        return brainevent.EventArray(spike) @ weight

    @jax.jit
    def f2(spike, weight):
        return spike @ weight

    y1 = jax.block_until_ready(f1(spike, weight))
    y2 = jax.block_until_ready(f2(spike, weight))
    print('max difference:', jax.numpy.abs(y1 - y2).max())

    n = 100
    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f1(spike, weight))
    r1 = time.time() - t0
    print(f"m: {m}, k: {k}, n: {n}, spike probability: {spk_prob}, Linear: {r1} s")

    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f2(spike, weight))
    r2 = time.time() - t0
    print(f"m: {m}, k: {k}, n: {n}, spike probability: {spk_prob}, Matmul: {r2} s")

    ratio = (r2 / r1 - 1.) if r2 > r1 else -(r1 / r2 - 1.)
    print('Acceleration ratio:', ratio)
    print()
    return ratio


def matrix_event(m, k, n, spk_prob, as_float: bool):
    weight = brainstate.init.KaimingUniform()((m, k))
    spike = (brainstate.random.rand( k, n) < spk_prob)

    if as_float:
        spike = spike.astype(float)

    @jax.jit
    def f1(spike):
        return weight @ brainevent.EventArray(spike)

    @jax.jit
    def f2(spike):
        return weight @ spike

    y1 = jax.block_until_ready(f1(spike))
    y2 = jax.block_until_ready(f2(spike))
    print('max difference:', jax.numpy.abs(y1 - y2).max())

    n = 100
    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f1(spike))
    r1 = time.time() - t0
    print(f"m: {m}, k: {k}, n: {n}, spike probability: {spk_prob}, Linear: {r1} s")

    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f2(spike))
    r2 = time.time() - t0
    print(f"m: {m}, k: {k}, n: {n}, spike probability: {spk_prob}, Matmul: {r2} s")

    ratio = (r2 / r1 - 1.) if r2 > r1 else -(r1 / r2 - 1.)
    print('Acceleration ratio:', ratio)
    print()
    return ratio


def visualize(results, title='Acceleration Ratio', filename=None):
    labels = list(results.keys())
    ratio = list(results.values())

    x = np.arange(len(labels))  # x轴的位置
    width = 0.35  # 条形的宽度

    fig, ax = plt.subplots()
    bars = ax.bar(x, ratio, width, label='Ratio')

    # 添加标签
    ax.set_xlabel('Configurations')
    ax.set_ylabel('Acceleration Ratio')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # 在每个条形上添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',  # 格式化数值
                    xy=(bar.get_x() + bar.get_width() / 2, height),  # 标签位置
                    xytext=(0, 3),  # 偏移量
                    textcoords="offset points",
                    ha='center',
                    va='bottom')

    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def benchmark_event_matrix(prob=0.1):
    platform = brainstate.environ.get_platform()

    results = {}
    for m, k, n in [
        (100, 100, 100),
        (500, 500, 500),
        (1000, 1000, 1000),
        (2000, 2000, 1000),
        (2000, 2000, 1000),
        (1000, 1000, 10000),
        # (10000, 10000, 10000),
        # (20000, 20000, 10000),
        # (10000, 10000, 20000),
    ]:
        results[f'{m}x{k}x{n}'] = event_matrix(m, k, n, prob, True)

    visualize(
        results,
        title=f'Acceleration Ratio (p={prob})',
        filename=f'results/event-matrix-prob={prob}-{platform}.pdf'
    )


def benchmark_matrix_event(prob=0.1):
    platform = brainstate.environ.get_platform()

    results = {}
    for m, k, n in [
        (100, 100, 100),
        (500, 500, 500),
        (1000, 1000, 1000),
        (2000, 2000, 1000),
        (2000, 2000, 1000),
        (1000, 1000, 10000),
        # (10000, 10000, 10000),
        # (20000, 20000, 10000),
        # (10000, 10000, 20000),
    ]:
        results[f'{m}x{k}x{n}'] = matrix_event(m, k, n, prob, False)

    visualize(
        results,
        title=f'Acceleration Ratio (p={prob})',
        filename=f'results/matrix-event-prob={prob}-{platform}.pdf'
    )


if __name__ == '__main__':
    # benchmark_forward(0.1)

    benchmark_event_matrix(0.1)
    benchmark_event_matrix(0.01)
    benchmark_event_matrix(0.001)

    # benchmark_matrix_event(0.01)
    # benchmark_matrix_event(0.001)

