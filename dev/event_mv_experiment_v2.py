# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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
# -*- coding: utf-8 -*-
import time

import jax.numpy
import numpy as np
import warp
from warp.jax_experimental.ffi import jax_kernel

warp.config.enable_backward = False
# warp.clear_kernel_cache()

TILE_SIZE = 128


def cdiv(a, b):
    return (a + b - 1) // b


@warp.kernel
def kernel_transpose_v1(
    weights: warp.array2d(dtype=warp.float32),
    spikes: warp.array1d(dtype=warp.bool),
    out: warp.array1d(dtype=warp.float32),
):
    i_tile = warp.tid()
    temp = warp.tile_zeros(shape=(TILE_SIZE,), dtype=warp.float32)
    for i_row in range(0, spikes.shape[0], TILE_SIZE):
        spk = warp.tile_load(spikes, shape=(TILE_SIZE,), offset=(i_row,))
        for j in range(min(TILE_SIZE, spikes.shape[0] - i_row)):
            if spk[j]:
                temp += warp.tile_load(weights[i_row + j], shape=(TILE_SIZE,), offset=(i_tile * TILE_SIZE,))
    warp.tile_store(out, temp, offset=(i_tile * TILE_SIZE,))


jax_kernel_transpose_v1 = jax_kernel(kernel_transpose_v1)


# @warp.kernel
# def kernel_v1(
#     weights: warp.array2d(dtype=warp.float32),
#     spikes: warp.array1d(dtype=warp.bool),
#     out: warp.array1d(dtype=warp.float32),
# ):
#     i_tile = warp.tid()
#     temp = warp.tile_zeros(shape=(TILE_SIZE, ), dtype=warp.float32)
#     for i_col in range(0, spikes.shape[0], TILE_SIZE):
#         spk = warp.tile_load(spikes, shape=(TILE_SIZE,), offset=(i_col,))
#         for j in range(min(TILE_SIZE, spikes.shape[0] - i_col)):
#             if spk[j]:
#                 w = warp.tile_load(weights, shape=(TILE_SIZE, 1), offset=(i_tile * TILE_SIZE, i_col + j))
#                 # w = warp.tile_load(weights[:, i_col + j], shape=(TILE_SIZE,), offset=(i_tile * TILE_SIZE,))
#                 temp += w
#     warp.tile_store(out, temp, offset=(i_tile * TILE_SIZE,))


@warp.kernel
def kernel_transpose_v2(
    weights: warp.array2d(dtype=warp.float32),
    spikes: warp.array1d(dtype=warp.bool),
    out: warp.array1d(dtype=warp.float32),
):
    i = warp.tid()
    temp = warp.float32(0.)
    for i_row in range(0, spikes.shape[0]):
        spk = spikes[i_row]
        if spk:
            temp += weights[i_row, i]
    out[i] = temp


jax_kernel_transpose_v2 = jax_kernel(kernel_transpose_v2)


@warp.kernel
def kernel_v2(
    weights: warp.array2d(dtype=warp.float32),
    spikes: warp.array1d(dtype=warp.bool),
    out: warp.array1d(dtype=warp.float32),
):
    i_row = warp.tid()
    temp = warp.float32(0.)
    for i_col in range(0, spikes.shape[0]):
        spk = spikes[i_col]
        if spk:
            temp += weights[i_row, i_col]
    out[i_row] = temp


jax_kernel_v2 = jax_kernel(kernel_v2)


def benchmark_kernel1(m=1024, n=1024, p=0.1, transpose=True):
    spikes = jax.numpy.asarray(np.random.random((m if transpose else n,)) < p)
    matrix = jax.numpy.asarray(np.random.randn(m, n).astype(np.float32))

    if transpose:
        @jax.jit
        def f():
            return jax_kernel_transpose_v1(matrix, spikes, output_dims=n)

        jax.block_until_ready(f())
        t0 = time.time()
        for _ in range(100):
            jax.block_until_ready(f())
        t1 = time.time()

        print(f'm={m}, n={n}, p={p}, transpose={transpose}, kernel  1  time: {t1 - t0} s')


def benchmark_kernel2(m=1024, n=1024, p=0.1, transpose=True):
    spikes = jax.numpy.asarray(np.random.random((m if transpose else n,)) < p)
    matrix = jax.numpy.asarray(np.random.randn(m, n).astype(np.float32))

    @jax.jit
    def f():
        if transpose:
            return jax_kernel_transpose_v2(matrix, spikes, output_dims=n)
        else:
            return jax_kernel_v2(matrix, spikes, output_dims=n)

    jax.block_until_ready(f())
    t0 = time.time()
    for _ in range(100):
        jax.block_until_ready(f())
    t1 = time.time()

    # print(out_wp)
    print(f'm={m}, n={n}, p={p}, transpose={transpose}, kernel  2  time: {t1 - t0} s')


def benchmark_jax(m=1024, n=1024, p=0.1, transpose=True):
    spikes = jax.numpy.asarray(np.random.random((m if transpose else n,)) < p)
    matrix = jax.numpy.asarray(np.random.randn(m, n).astype(np.float32))

    @jax.jit
    def f():
        if transpose:
            return spikes @ matrix
        else:
            return matrix @ spikes

    out = jax.block_until_ready(f())
    t0 = time.time()
    for _ in range(100):
        out = jax.block_until_ready(f())
    t1 = time.time()

    # print(out)
    print(f'm={m}, n={n}, p={p}, transpose={transpose}, kernel jax time: {t1 - t0} s')


if __name__ == '__main__':
    transpose = False
    transpose = True
    for m, n, p in [
        (1024, 1024, 0.1),
        (2048, 2048, 0.1),
        (4096, 4096, 0.1),
        (8192, 8192, 0.1),
        (16384, 16384, 0.1),

        (1024, 2048, 0.1),
        (2048, 4096, 0.1),
        (4096, 8192, 0.1),
        (8192, 16384, 0.1),
        (16384, 16384 * 2, 0.1),

        (2048, 1024, 0.1),
        (4096, 2048, 0.1),
        (8192, 4096, 0.1),
        (16384, 8192, 0.1),
        (16384 * 2, 16384, 0.1),

        (1024, 1024, 0.01),
        (2048, 2048, 0.01),
        (4096, 4096, 0.01),
        (8192, 8192, 0.01),
        (16384, 16384, 0.01),
    ]:
        benchmark_kernel1(m, n, p, transpose=transpose)
        # benchmark_kernel2(m, n, p, transpose=transpose)
        benchmark_jax(m, n, p, transpose=transpose)
        print()
