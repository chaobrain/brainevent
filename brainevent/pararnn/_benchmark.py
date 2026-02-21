# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""
Performance benchmarks for parallel RNN training modes.

Compares:
- Sequential (jax.lax.scan)
- Parallel (jax.lax.associative_scan)
- CUDA parallel reduction (TVM FFI)
- Fused CUDA (TVM FFI)

Usage:
    python brainevent/pararnn/_benchmark.py
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time

import jax
import jax.numpy as jnp
import jax.random as jr

from brainevent.pararnn._gru import GRUDiagMH
from brainevent.pararnn._lstm import LSTMCIFGDiagMH
from brainevent.pararnn._parallel_reduce import parallel_reduce_diag, parallel_reduce_block_diag
from brainevent.pararnn._fused_cuda import fused_cuda_available
from brainevent.pararnn._parallel_reduce_cuda import (
    parallel_reduce_diag_cuda,
    parallel_reduce_block2_cuda,
    cuda_available,
)


def _time_fn(fn, x, n_warmup=5, n_runs=20):
    """Time a function with warmup."""
    # Warmup
    for _ in range(n_warmup):
        y = fn(x)
        y.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        y = fn(x)
        y.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms
    return min(times), sum(times) / len(times)


def _time_fn_with_grad(fn, x, n_warmup=5, n_runs=20):
    """Time forward + backward with warmup."""

    def loss(x):
        return jnp.sum(fn(x) ** 2)

    grad_fn = jax.grad(loss)

    # Warmup
    for _ in range(n_warmup):
        g = grad_fn(x)
        g.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        g = grad_fn(x)
        g.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms
    return min(times), sum(times) / len(times)


def benchmark_gru(
    batch_sizes=(1, 4),
    seq_lengths=(64, 256, 1024),
    hidden_dims=(32, 128),
    input_dim=32,
    num_heads=2
):
    """Benchmark GRU across modes."""
    print("=" * 80)
    print("GRU Benchmark")
    print("=" * 80)

    modes = ['sequential', 'parallel']
    modes = ['parallel']

    # Check for fused CUDA
    if fused_cuda_available():
        modes.append('fused')

    header = f"{'B':>4} {'T':>6} {'N':>5} | "
    header += " | ".join(f"{m:>12}" for m in modes)
    header += " | "
    header += " | ".join(f"{m + '+bwd':>12}" for m in modes)
    print(header)
    print("-" * len(header))

    for B in batch_sizes:
        for T in seq_lengths:
            for N in hidden_dims:
                if input_dim % num_heads != 0 or N % num_heads != 0:
                    continue

                x = jr.normal(jr.PRNGKey(0), (B, T, input_dim)) * 0.1

                fwd_results = {}
                bwd_results = {}

                for mode in modes:
                    try:
                        gru = GRUDiagMH(
                            input_dim=input_dim, state_dim=N,
                            num_heads=num_heads, mode=mode,
                        )
                        fn = jax.jit(gru)
                        min_t, avg_t = _time_fn(fn, x)
                        fwd_results[mode] = f"{avg_t:8.2f}ms"

                        min_t, avg_t = _time_fn_with_grad(fn, x)
                        bwd_results[mode] = f"{avg_t:8.2f}ms"
                    except Exception as e:
                        fwd_results[mode] = f"{'ERR':>8}"
                        bwd_results[mode] = f"{'ERR':>8}"

                row = f"{B:>4} {T:>6} {N:>5} | "
                row += " | ".join(f"{fwd_results.get(m, 'N/A'):>12}" for m in modes)
                row += " | "
                row += " | ".join(f"{bwd_results.get(m, 'N/A'):>12}" for m in modes)
                print(row)

    print()


def benchmark_lstm(
    batch_sizes=(1, 4),
    seq_lengths=(64, 256, 1024),
    hidden_dims=(32, 128),
    input_dim=32,
    num_heads=2,
):
    """Benchmark LSTM-CIFG across modes."""
    print("=" * 80)
    print("LSTM-CIFG Benchmark")
    print("=" * 80)

    modes = ['sequential', 'parallel']
    modes = ['parallel']

    if fused_cuda_available():
        modes.append('fused')

    header = f"{'B':>4} {'T':>6} {'N':>5} | "
    header += " | ".join(f"{m:>12}" for m in modes)
    header += " | "
    header += " | ".join(f"{m + '+bwd':>12}" for m in modes)
    print(header)
    print("-" * len(header))

    for B in batch_sizes:
        for T in seq_lengths:
            for N in hidden_dims:
                if input_dim % num_heads != 0 or N % num_heads != 0:
                    continue

                x = jr.normal(jr.PRNGKey(0), (B, T, input_dim)) * 0.1

                fwd_results = {}
                bwd_results = {}

                for mode in modes:
                    try:
                        lstm = LSTMCIFGDiagMH(
                            input_dim=input_dim, state_dim=N,
                            num_heads=num_heads, mode=mode,
                        )
                        fn = jax.jit(lstm)
                        min_t, avg_t = _time_fn(fn, x)
                        fwd_results[mode] = f"{avg_t:8.2f}ms"

                        min_t, avg_t = _time_fn_with_grad(fn, x)
                        bwd_results[mode] = f"{avg_t:8.2f}ms"
                    except Exception as e:
                        fwd_results[mode] = f"{'ERR':>8}"
                        bwd_results[mode] = f"{'ERR':>8}"

                row = f"{B:>4} {T:>6} {N:>5} | "
                row += " | ".join(f"{fwd_results.get(m, 'N/A'):>12}" for m in modes)
                row += " | "
                row += " | ".join(f"{bwd_results.get(m, 'N/A'):>12}" for m in modes)
                print(row)

    print()


def benchmark_parallel_reduce():
    """Benchmark standalone parallel reduction kernels."""
    print("=" * 80)
    print("Standalone Parallel Reduction Benchmark")
    print("=" * 80)

    has_cuda = cuda_available()

    print("\nDiagonal reduction:")
    print(f"{'B':>4} {'T':>6} {'N':>5} | {'JAX':>12} | {'CUDA':>12}")
    print("-" * 50)

    for B in (1, 4):
        for T in (64, 256, 1024, 4096):
            for N in (32, 128):
                key = jr.PRNGKey(0)
                k1, k2 = jr.split(key)
                jac = jr.normal(k1, (B, T, N)) * 0.5
                rhs = jr.normal(k2, (B, T, N))

                fn_jax = jax.jit(parallel_reduce_diag)
                _, avg_jax = _time_fn(
                    lambda x: fn_jax(jac, x), rhs,
                    n_warmup=3, n_runs=10,
                )

                if has_cuda:
                    fn_cuda = jax.jit(parallel_reduce_diag_cuda)
                    _, avg_cuda = _time_fn(
                        lambda x: fn_cuda(jac, x), rhs,
                        n_warmup=3, n_runs=10,
                    )
                    cuda_str = f"{avg_cuda:8.2f}ms"
                else:
                    cuda_str = "N/A"

                print(f"{B:>4} {T:>6} {N:>5} | {avg_jax:8.2f}ms | {cuda_str:>12}")

    if has_cuda:
        print("\n2x2 Block-diagonal reduction:")
        print(f"{'B':>4} {'T':>6} {'N':>5} | {'JAX':>12} | {'CUDA':>12}")
        print("-" * 50)

        for B in (1, 4):
            for T in (64, 256, 1024):
                for N in (32, 64):
                    key = jr.PRNGKey(0)
                    k1, k2 = jr.split(key)
                    jac = jr.normal(k1, (B, T, N, 2, 2)) * 0.3
                    rhs = jr.normal(k2, (B, T, N, 2))

                    fn_jax = jax.jit(parallel_reduce_block_diag)
                    _, avg_jax = _time_fn(
                        lambda x: fn_jax(jac, x), rhs,
                        n_warmup=3, n_runs=10,
                    )

                    fn_cuda = jax.jit(parallel_reduce_block2_cuda)
                    _, avg_cuda = _time_fn(
                        lambda x: fn_cuda(jac, x), rhs,
                        n_warmup=3, n_runs=10,
                    )

                    print(
                        f"{B:>4} {T:>6} {N:>5} | "
                        f"{avg_jax:8.2f}ms | {avg_cuda:8.2f}ms"
                    )

    print()


if __name__ == '__main__':
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    # benchmark_parallel_reduce()
    benchmark_gru(
        batch_sizes=(8, 16, 32, 64),
        hidden_dims=(128, 256, 512, 1024),
    )
    benchmark_lstm(
        batch_sizes=(8, 16, 32, 64),
        hidden_dims=(128, 256, 512, 1024),
    )
