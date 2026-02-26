"""
Binary FCN Matrix-Matrix Multiplication Benchmark
==================================================

Benchmarks all available GPU backends for ``binary_fcnmm``
(gather and scatter modes, homo/hetero weights, bool/float matrix inputs)
across a range of problem sizes.

The key metric is the speedup of the CUDA ``cuda_raw`` backend relative to the
``jax_raw`` baseline.  For typical SNN firing rates (5–50 % spike density),
the event-driven CUDA kernels avoid touching inactive entries and therefore
outperform a dense computation.

Backends compared
-----------------
- ``cuda_raw``   : NVRTC-compiled CUDA kernels (gather: warp/basic; scatter: warp/basic)
- ``pallas``   : JAX Pallas / Triton GPU kernels
- ``jax_raw``  : Pure JAX reference (segment_sum / vmap)

Usage
-----
    python dev/fcn/benchmark_binary_fcnmm.py
    python dev/fcn/benchmark_binary_fcnmm.py --n_warmup 5 --n_runs 50
    python dev/fcn/benchmark_binary_fcnmm.py --spike_rate 0.05 --n_batch 64
"""

import argparse
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json
import jax
import jax.numpy as jnp
import numpy as np

import brainstate
from brainevent import BenchmarkConfig, binary_fcnmm_p

# Problem sizes: (n_pre, n_post, n_conn)
'''
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
'''

current_name = 'binary_fcnmm'
benchmark_data_type = 'typeB'

# Problem sizes: (n_pre, n_post, n_conn)

def load_benchmark_config(json_path: str, benchmark_data_type: str, operator_name: str, config_key: str = 'config_1') -> dict:
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        
    if benchmark_data_type not in raw_data:
        raise KeyError(f"Type '{benchmark_data_type}' not found in configuration file.")
        
    if operator_name not in raw_data[benchmark_data_type]["operator"]:
        raise KeyError(f"operator '{benchmark_data_type}' not found in configuration file.")
    
    operator_data = raw_data[benchmark_data_type]

    if config_key not in operator_data:
        raise KeyError(f"Configuration block '{config_key}' not found under operator '{operator_name}'.")
        
    return operator_data[config_key]

config_file_path = 'benchmark_config.json'
parsed_config = load_benchmark_config(config_file_path, benchmark_data_type,  current_name)

dist_type = parsed_config.get('dist_type', 'uniform')
transpose_list = parsed_config.get('transpose', [False, True])
homo_list = parsed_config.get('homo_weight', [True, False])
matrix_configs = parsed_config.get('configs', [])
bool_event_list = parsed_config.get('bool_event', [True, False]) # binary_fcnmv specific

len_config = len(matrix_configs) * len(transpose_list) * len(homo_list) * len(bool_event_list)


def _make_benchmark_data(*, platform, spike_rate=None, n_batch=None):
    brainstate.environ.set(precision=32)  # change to 16 or 64 for other precisions
    rng = np.random.default_rng(42)
    dtype = brainstate.environ.dftype()
    spike_rates = (spike_rate,) if spike_rate is not None else (0.01, 0.05, 0.1)
    n_batches = (n_batch,) if n_batch is not None else (16, 64)

    for spike_rate in spike_rates:
        for n_b in n_batches:
            for cfg in matrix_configs:
                n_pre = cfg['n_rows']
                n_post = cfg['n_cols']
                prob = cfg['density']

                n_conn = max(1, int(n_post * prob))
                indices = jnp.asarray(rng.integers(0, n_post, (n_pre, n_conn), dtype=np.int32))

                for transpose in transpose_list:
                    for homo in homo_list:
                        for bool_event in bool_event_list:
                            if homo:
                                weights = jnp.ones(1, dtype=dtype)
                            else:
                                weights = jnp.asarray(rng.standard_normal((n_pre, n_conn)), dtype=dtype)
                            m_rows = n_post if not transpose else n_pre
                            if bool_event:
                                mat = jnp.asarray(rng.random((m_rows, n_b)) < spike_rate, dtype=jnp.bool_)
                            else:
                                raw = rng.standard_normal((m_rows, n_b))
                                mask = rng.random((m_rows, n_b)) < spike_rate
                                mat = jnp.asarray(np.where(mask, np.abs(raw), 0.0), dtype=dtype)

                            name = (
                                f"{'T' if transpose else 'NT'},"
                                f"{'homo' if homo else 'hetero'},"
                                f"{'bool' if bool_event else 'float'},"
                                f"batch={n_b},"
                                f"{n_pre}x{n_post}x{prob}"
                            )
                            yield BenchmarkConfig(
                                name=name,
                                args=(weights, indices, mat),
                                kernel_kwargs={
                                    'shape': (n_pre, n_post),
                                    'transpose': transpose,
                                },
                                data_kwargs={
                                    'n_pre': n_pre, 'n_post': n_post, 'prob': prob,
                                    'n_batch': n_b, 'transpose': transpose,
                                    'homo': homo, 'bool_event': bool_event,
                                    'spike_rate': spike_rate,
                                },
                            )


def main():
    parser = argparse.ArgumentParser(description="binary_fcnmm backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10,
                        help="Number of warmup iterations (default: 10)")
    parser.add_argument("--n_runs", type=int, default=50,
                        help="Number of timed iterations (default: 50)")
    parser.add_argument("--spike_rate", type=float, default=0.1,
                        help="Fraction of active matrix entries (default: 0.1 = 10%%)")
    parser.add_argument("--n_batch", type=int, default=32,
                        help="Batch (n) dimension of the input matrix (default: 32)")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.  Run this script on a CUDA-enabled machine.")
        return

    print(f"binary_fcnmm benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}  "
          f"spike_rate={args.spike_rate:.0%}  n_batch={args.n_batch}")
    print()

    def _data_gen(*, platform):
        yield from _make_benchmark_data(
            platform=platform,
            spike_rate=args.spike_rate,
            n_batch=args.n_batch,
        )

    binary_fcnmm_p.def_benchmark_data(_data_gen)

    total_len_config = len_config * (1 if args.spike_rate is not None else 3)

    result = binary_fcnmm_p.benchmark_csv_output(
        platform='gpu',
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        n_batch_per_run=1,
        compare_results=True,
        verbose=False,
        catch_errors=False,
        len_config=total_len_config
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='jax_raw')


if __name__ == "__main__":
    main()
