import os
import time

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax
import sys

sys.path.append('../')

import brainstate
from utils import visualize
import brainevent
from brainevent._array_binary_impl import (
    indices_dot_dense_mat,
    binary_vec_get_indices

)
brainevent.config.gpu_kernel_backend = 'warp'
# brainevent.config.gpu_kernel_backend = 'pallas'

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
    
    count_arr = brainstate.random.randn(count).astype(int)
    @jax.jit
    def f1(spikes, weights, count_arr):
        return (
            indices_dot_dense_mat(indices, weights, count_arr)
        )

    @jax.jit
    def f2(spikes, weights):
        return (
            (spikes @ weights)
        )
    @jax.jit
    def get_idx(spikes):
        return (
            binary_vec_get_indices(spikes)
        )
    idx,cnt = jax.block_until_ready(get_idx(spike))
    #count_arr2 = brainstate.random.randn(cnt[0]).astype(int)
    # print(idx)
    # print(indices)
 
    y1 = jax.block_until_ready(f1(indices, weight, count_arr))
    #y1 = jax.block_until_ready(f1(idx, weight, count_arr2))
    y2 = jax.block_until_ready(f2(spike, weight))
    print('max difference:', jax.numpy.abs(y1 - y2).max())

    n = 100
    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f1(indices, weight, count_arr))
    r1 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post},  spike probability: {spk_prob}, Linear: {r1} s")

    t0 = time.time()
    for _ in range(n):
        jax.block_until_ready(f2(spike, weight))
    r2 = time.time() - t0
    print(f"n_pre: {n_pre}, n_post: {n_post},  spike probability: {spk_prob}, Matmul: {r2} s")

    ratio = (r2 / r1 - 1.) if r2 > r1 else -(r1 / r2 - 1.)
    print('Acceleration ratio:', ratio)
    print()
    return ratio



def benchmark_forward(prob=0.1):
    #platform = brainstate.environ.get_platform()

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
