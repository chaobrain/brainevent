"""
Binary FCN Matrix-Vector Multiplication Benchmark
==================================================

Benchmarks all available GPU backends for ``binary_fcnmv``
(gather and scatter modes, homo/hetero weights, bool/float spikes)
across a range of problem sizes.

The key metric is the speedup of the CUDA ``tvmffi`` backend relative to the
``jax_raw`` baseline.  For typical SNN firing rates (5–50 % spike density),
the event-driven CUDA kernels avoid touching inactive entries and therefore
outperform a dense computation.

Backends compared
-----------------
- ``tvmffi``   : NVRTC-compiled CUDA kernels (gather: warp/basic; scatter: warp/basic)
- ``pallas``   : JAX Pallas / Triton GPU kernels
- ``jax_raw``  : Pure JAX reference (segment_sum / vmap gather)

Usage
-----
    python dev/fcn/benchmark_binary_fcnmv.py
    python dev/fcn/benchmark_binary_fcnmv.py --n_warmup 5 --n_runs 50
    python dev/fcn/benchmark_binary_fcnmv.py --spike_rate 0.05
"""

import argparse
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np

from brainevent import BenchmarkConfig
from brainevent._fcn.binary import binary_fcnmv_p

# Problem sizes: (n_pre, n_post, n_conn)
CONFIGS = [
    (500, 1000, 10),
    (1000, 1000, 32),
    (1000, 1000, 50),
    (1000, 1000, 100),
    (1000, 1000, 128),
    (5000, 5000, 100),
    (5000, 5000, 200),
    (10000, 10000, 500),
    (10000, 10000, 1000),
]


def _make_benchmark_data(*, platform, spike_rate=None):
    rng = np.random.default_rng(42)
    dtype = jnp.float32
    spike_rates = (spike_rate,) if spike_rate is not None else (0.01, 0.05, 0.1)
    for spike_rate in spike_rates:
        for n_pre, n_post, n_conn in CONFIGS:
            indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn), dtype=np.int32))
            for transpose in (False, True):
                for homo in (True, False):
                    for bool_event in (True, False):
                        if homo:
                            weights = jnp.ones(1, dtype=dtype)
                        else:
                            weights = jnp.asarray(rng.standard_normal((n_pre, n_conn)).astype(np.float32))
                        v_size = n_post if not transpose else n_pre
                        if bool_event:
                            spikes = jnp.asarray(rng.random(v_size) < spike_rate, dtype=jnp.bool_)
                        else:
                            # Float spikes: positive values are active
                            raw = rng.standard_normal(v_size).astype(np.float32)
                            # Zero out ~(1-spike_rate) fraction to mimic sparse events
                            mask = rng.random(v_size) < spike_rate
                            spikes = jnp.asarray(np.where(mask, np.abs(raw), 0.0))

                        name = (
                            f"{'T' if transpose else 'NT'},"
                            f"{'homo' if homo else 'hetero'},"
                            f"{'bool' if bool_event else 'float'},"
                            f"{n_pre}x{n_post}x{n_conn}"
                        )
                        yield BenchmarkConfig(
                            name=name,
                            args=(weights, indices, spikes),
                            kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                            data_kwargs={
                                'n_pre': n_pre, 'n_post': n_post, 'n_conn': n_conn,
                                'transpose': transpose, 'homo': homo,
                                'bool_event': bool_event, 'spike_rate': spike_rate,
                            },
                        )


def main():
    parser = argparse.ArgumentParser(description="binary_fcnmv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10,
                        help="Number of warmup iterations (default: 10)")
    parser.add_argument("--n_runs", type=int, default=50,
                        help="Number of timed iterations (default: 50)")
    parser.add_argument("--spike_rate", type=float, default=0.1,
                        help="Fraction of active spikes (default: 0.1 = 10%%)")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.  Run this script on a CUDA-enabled machine.")
        return

    print(f"binary_fcnmv benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}  spike_rate={args.spike_rate:.0%}")
    print()

    # Inject our benchmark data generator (spike_rate-aware)
    def _data_gen(*, platform):
        yield from _make_benchmark_data(platform=platform, spike_rate=args.spike_rate)

    binary_fcnmv_p.def_benchmark_data(_data_gen)

    result = binary_fcnmv_p.benchmark(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        compare_results=True,
        verbose=True,
    )
    result.print(
        order_by=['transpose', 'shape', 'backend'],
        highlight_best=True,
        speedup_vs='jax_raw',
    )


if __name__ == "__main__":
    main()
