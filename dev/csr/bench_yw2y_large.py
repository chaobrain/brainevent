"""
Benchmark csrmv_yw2y on LARGE sparse matrices (target regime: 1-10% density)
"""
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
import jax
import jax.numpy as jnp
import numpy as np

from brainevent._csr.yw2y import csrmv_yw2y

# Target regime: large matrices, 1-10% density
CONFIGS = [
    # (n_pre, n_post, conn_prob, label)
    (50_000,  50_000,  0.01,  "50K×50K,1%,avg_nnz=500"),
    (100_000, 100_000, 0.01,  "100K×100K,1%,avg_nnz=1K"),
    (200_000, 200_000, 0.005, "200K×200K,0.5%,avg_nnz=1K"),
    (100_000, 100_000, 0.05,  "100K×100K,5%,avg_nnz=5K"),
]

def benchmark_config(n_pre, n_post, prob, label, transpose, backend, n_warmup=5, n_runs=20):
    """Benchmark a single configuration."""
    rng = np.random.default_rng(42)
    dtype = np.float32

    # Generate CSR matrix
    n_conn = max(1, int(n_post * prob))
    nse = n_pre * n_conn
    indptr = jnp.asarray(np.arange(n_pre + 1, dtype=np.int32) * n_conn)
    indices = jnp.asarray(rng.integers(0, n_post, nse, dtype=np.int32))
    w = jnp.asarray(rng.standard_normal(nse), dtype=dtype)

    # Generate input vector
    y_size = n_post if transpose else n_pre
    y = jnp.asarray(rng.standard_normal(y_size), dtype=dtype)

    # Compile
    fn = jax.jit(lambda: csrmv_yw2y(
        y, w, indices, indptr,
        shape=(n_pre, n_post),
        transpose=transpose,
        backend=backend
    ))

    # Warmup
    for _ in range(n_warmup):
        result = fn()
        jax.block_until_ready(result)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)

    times = np.array(times)
    mean_time = np.mean(times)
    min_time = np.min(times)
    std_time = np.std(times)

    # Calculate bandwidth
    # Memory traffic: w(read) + y(read, amortized) + output(write)
    # For NT: ~nse*4 (w) + ~n_pre*4 (y, amortized) + nse*4 (out) ≈ 2*nse*4 + n_pre*4
    # For T: ~nse*4 (w) + ~nse*4 (y, scattered) + nse*4 (indices) + nse*4 (out) ≈ 4*nse*4
    if transpose:
        bytes_transferred = 4 * nse * 4  # w + y[indices] + indices + output
    else:
        bytes_transferred = 2 * nse * 4 + n_pre * 4  # w + y + output

    bandwidth_gb_s = (bytes_transferred / 1e9) / mean_time

    return {
        'label': label,
        'transpose': transpose,
        'backend': backend,
        'n_pre': n_pre,
        'n_post': n_post,
        'prob': prob,
        'nse': nse,
        'mean_ms': mean_time * 1000,
        'min_ms': min_time * 1000,
        'std_ms': std_time * 1000,
        'bandwidth_gb_s': bandwidth_gb_s,
    }

def main():
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print(f"csrmv_yw2y Large Matrix Benchmark  —  GPU: {gpu}")
    print("="*100)

    # Get available backends
    from brainevent._csr.yw2y import csrmv_yw2y_p
    backends = csrmv_yw2y_p.available_backends('gpu')
    print(f"Available backends: {backends}\n")

    results = []

    for n_pre, n_post, prob, label in CONFIGS:
        for transpose in [False, True]:
            trans_label = 'T' if transpose else 'NT'
            for backend in backends:
                print(f"Running: {trans_label:<3} {label:<30} backend={backend:<10}", end=' ... ', flush=True)
                try:
                    result = benchmark_config(n_pre, n_post, prob, label, transpose, backend)
                    results.append(result)
                    print(f"✓ {result['mean_ms']:7.3f}ms  {result['bandwidth_gb_s']:6.1f} GB/s")
                except Exception as e:
                    print(f"✗ FAILED: {e}")

    print("\n" + "="*100)
    print("Summary Table")
    print("="*100)
    print(f"{'Config':<35} {'Trans':<6} {'Backend':<10} {'Time(ms)':<10} {'BW(GB/s)':<10} {'NSE':<12}")
    print("-"*100)

    for r in results:
        print(f"{r['label']:<35} {'T' if r['transpose'] else 'NT':<6} {r['backend']:<10} "
              f"{r['mean_ms']:7.3f}    {r['bandwidth_gb_s']:7.1f}    {r['nse']:<12,}")

    print("\n" + "="*100)
    print("Roofline Analysis (A100: 1555 GB/s peak bandwidth)")
    print("="*100)
    for r in results:
        efficiency = (r['bandwidth_gb_s'] / 1555.0) * 100
        print(f"{r['label']:<35} {'T' if r['transpose'] else 'NT':<6} {r['backend']:<10} "
              f"Efficiency: {efficiency:5.1f}%")

if __name__ == "__main__":
    main()
