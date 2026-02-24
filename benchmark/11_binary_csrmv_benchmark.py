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

"""
Benchmark for binary_csrmv_p (binary CSR matrix-vector multiplication).

This benchmark tests the performance of event-driven sparse matrix-vector
multiplication across different:
- Matrix sizes
- Sparsity levels
- Transpose modes
- Weight types (homogeneous vs heterogeneous)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from brainevent import binary_csrmv_p


def generate_csr_matrix(n_rows: int, n_cols: int, density: float, seed: int = 42):
    """Generate a random CSR matrix with given density."""
    np.random.seed(seed)
    nnz = int(n_rows * n_cols * density)

    # Generate random row and column indices
    rows = np.random.randint(0, n_rows, size=nnz)
    cols = np.random.randint(0, n_cols, size=nnz)
    data = np.ones(nnz, dtype=np.float32)

    # Create COO matrix and convert to CSR to handle duplicates
    coo = sparse.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    csr = coo.tocsr()

    indices = jnp.asarray(csr.indices, dtype=jnp.int32)
    indptr = jnp.asarray(csr.indptr, dtype=jnp.int32)

    return indices, indptr, csr.nnz


def run_benchmark(
    n_rows: int,
    n_cols: int,
    density: float,
    transpose: bool,
    homo_weight: bool,
    platform: str,
    n_warmup: int = 5,
    n_runs: int = 20,
    batch_mode: bool = False,
):
    """Run benchmark for a specific configuration."""
    # Generate CSR matrix
    indices, indptr, nnz = generate_csr_matrix(n_rows, n_cols, density)

    # Generate weights
    if homo_weight:
        weights = jnp.ones(1, dtype=jnp.float32)
    else:
        weights = jnp.ones(nnz, dtype=jnp.float32)

    # Generate input vector
    if transpose:
        vector = jnp.ones(n_rows, dtype=jnp.float32)
    else:
        vector = jnp.ones(n_cols, dtype=jnp.float32)

    # Run benchmark
    report = binary_csrmv_p.benchmark(
        weights, indices, indptr, vector,
        shape=(n_rows, n_cols),
        transpose=transpose,
        platform=platform,
        n_warmup=n_warmup,
        n_runs=n_runs,
        batch_mode=batch_mode,
        compare_results=True,
        catch_error=False,
    )

    return report, nnz


def main():
    parser = argparse.ArgumentParser(description='Benchmark binary_csrmv_p')
    parser.add_argument('--platform', type=str, default='gpu', choices=['cpu', 'gpu', 'tpu'],
                        help='Platform to benchmark on')
    parser.add_argument('--n-warmup', type=int, default=50, help='Number of warmup runs')
    parser.add_argument('--n-runs', type=int, default=200, help='Number of timed runs')
    parser.add_argument('--batch-mode', action='store_true', default=True, help='Use batch timing mode')
    args = parser.parse_args()

    print(f"Binary CSR Matrix-Vector Multiplication Benchmark")
    print(f"Platform: {args.platform}")
    print(f"Available backends: {binary_csrmv_p.available_backends(args.platform)}")
    print(f"Warmup runs: {args.n_warmup}, Timed runs: {args.n_runs}")
    print(f"Batch mode: {args.batch_mode}")
    print("=" * 80)
    print()

    # Benchmark configurations
    configs = [
        # Small matrices
        {'n_rows': 1000, 'n_cols': 1000, 'density': 0.01},
        {'n_rows': 1000, 'n_cols': 1000, 'density': 0.1},

        # Medium matrices
        {'n_rows': 10000, 'n_cols': 10000, 'density': 0.001},
        {'n_rows': 10000, 'n_cols': 10000, 'density': 0.01},

        # Large matrices (sparse)
        {'n_rows': 100000, 'n_cols': 100000, 'density': 0.0001},
        {'n_rows': 100000, 'n_cols': 100000, 'density': 0.001},

        # Rectangular matrices
        {'n_rows': 10000, 'n_cols': 1000, 'density': 0.01},
        {'n_rows': 1000, 'n_cols': 10000, 'density': 0.01},
    ]

    results_table = []

    for config in configs:
        n_rows = config['n_rows']
        n_cols = config['n_cols']
        density = config['density']

        for transpose in [False, True]:
            for homo_weight in [True, False]:
                weight_type = "homo" if homo_weight else "heter"
                trans_str = "T" if transpose else "N"

                print(f"Config: {n_rows}x{n_cols}, density={density}, "
                      f"transpose={trans_str}, weight={weight_type}")

                report, nnz = run_benchmark(
                    n_rows=n_rows,
                    n_cols=n_cols,
                    density=density,
                    transpose=transpose,
                    homo_weight=homo_weight,
                    platform=args.platform,
                    n_warmup=args.n_warmup,
                    n_runs=args.n_runs,
                    batch_mode=args.batch_mode,
                )
                print(report.summary())

                fastest = report.fastest()
                if fastest:
                    print(f"  nnz={nnz:,}, fastest={fastest.backend}, "
                          f"mean={fastest.mean_time * 1000:.3f}ms, "
                          f"std={fastest.std_time * 1000:.3f}ms")

                    results_table.append(
                        {
                            'shape': f"{n_rows}x{n_cols}",
                            'density': density,
                            'nnz': nnz,
                            'transpose': trans_str,
                            'weight': weight_type,
                            'backend': fastest.backend,
                            'mean_ms': fastest.mean_time * 1000,
                            'std_ms': fastest.std_time * 1000,
                        }
                    )

                if report.mismatches:
                    print(f"  WARNING: Result mismatches: {report.mismatches}")

                print()

    # Print summary table
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Shape':<15} {'Density':<10} {'NNZ':<12} {'Trans':<6} {'Weight':<6} "
        f"{'Backend':<10} {'Mean(ms)':<10} {'Std(ms)':<10}"
    )
    print("-" * 80)
    for r in results_table:
        print(
            f"{r['shape']:<15} {r['density']:<10} {r['nnz']:<12,} {r['transpose']:<6} "
            f"{r['weight']:<6} {r['backend']:<10} {r['mean_ms']:<10.3f} {r['std_ms']:<10.3f}"
        )


if __name__ == '__main__':
    main()
