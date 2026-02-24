"""
Micro-benchmark for yw2y roofline analysis
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

def benchmark_kernel(n_pre, n_post, prob, transpose, backend='tvmffi', n_warmup=10, n_runs=50):
    """Benchmark and compute roofline metrics"""
    rng = np.random.default_rng(42)
    dtype = np.float32

    # Generate CSR matrix
    n_conn = max(1, int(n_post * prob))
    nse = n_pre * n_conn
    indptr = jnp.asarray(np.arange(n_pre + 1, dtype=np.int32) * n_conn)
    indices = jnp.asarray(rng.integers(0, n_post, nse, dtype=np.int32))
    w = jnp.asarray(rng.standard_normal(nse), dtype=dtype)

    y_size = n_post if transpose else n_pre
    y = jnp.asarray(rng.standard_normal(y_size), dtype=dtype)

    # JIT compile
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
    mean_time_ms = np.mean(times) * 1000
    min_time_ms = np.min(times) * 1000

    # Memory traffic analysis
    if transpose:
        # T: read w[j] (4B), indices[j] (4B), y[indices[j]] (4B scattered), write out[j] (4B)
        bytes_per_elem = 4 + 4 + 4 + 4  # 16 bytes
    else:
        # NT: read w[j] (4B), y[row] (4B, amortized/cached), write out[j] (4B)
        # Plus binary search overhead for NT_nz_thread: ~log2(m) indptr reads per group
        # With VEC_SIZE=4, amortized to ~log2(m) / 4 per element
        bytes_per_elem = 4 + 4 + 4  # 12 bytes (optimistic, assuming y is cached)
        avg_nnz = n_conn
        if avg_nnz > 512:
            # NT_nz_thread: add binary search overhead
            # Each thread searches once for 4 elements, so ~log2(m)*4 / 4 = log2(m) bytes/elem
            binary_search_bytes = int(np.log2(n_pre)) * 4
            bytes_per_elem += binary_search_bytes / 4  # Amortized over VEC_SIZE=4

    total_bytes = nse * bytes_per_elem
    total_gb = total_bytes / 1e9

    # Bandwidth
    bandwidth_gb_s = total_gb / (mean_time_ms / 1000)

    # Arithmetic intensity
    flops_per_elem = 1  # One multiply
    total_flops = nse * flops_per_elem
    arith_intensity = flops_per_elem / bytes_per_elem

    # Roofline bounds (A100)
    peak_bw_gb_s = 1555  # HBM2 bandwidth
    peak_tflops = 19.5   # FP32 peak

    # Determine if bandwidth or compute bound
    time_bw_bound_ms = (total_gb / peak_bw_gb_s) * 1000
    time_compute_bound_ms = (total_flops / (peak_tflops * 1e12)) * 1000

    if time_bw_bound_ms > time_compute_bound_ms:
        bound_type = "bandwidth"
        theoretical_time_ms = time_bw_bound_ms
    else:
        bound_type = "compute"
        theoretical_time_ms = time_compute_bound_ms

    efficiency_pct = (theoretical_time_ms / mean_time_ms) * 100

    return {
        'config': f"{n_pre}×{n_post} p={prob} {'T' if transpose else 'NT'}",
        'avg_nnz': n_conn,
        'nse': nse,
        'mean_ms': mean_time_ms,
        'min_ms': min_time_ms,
        'bytes_per_elem': bytes_per_elem,
        'total_gb': total_gb,
        'bandwidth_gb_s': bandwidth_gb_s,
        'arith_intensity': arith_intensity,
        'bound_type': bound_type,
        'theoretical_ms': theoretical_time_ms,
        'efficiency_pct': efficiency_pct,
        'peak_bw_pct': (bandwidth_gb_s / peak_bw_gb_s) * 100,
    }

if __name__ == "__main__":
    print("="*100)
    print("ROOFLINE ANALYSIS — csrmv_yw2y TVM FFI Kernels")
    print("="*100)
    print(f"Target GPU: {jax.devices('gpu')[0]}")
    print(f"Peak BW: 1555 GB/s (A100), Peak FP32: 19.5 TFLOPS")
    print("="*100)
    print()

    # Test cases covering different kernel paths
    configs = [
        # Small avg_nnz: NT_row_thread (avg_nnz < 8)
        (10000, 10000, 0.0005, False, "NT_row_thread path (avg_nnz=5)"),

        # Medium avg_nnz: NT_row_warp (8 <= avg_nnz < 512)
        (10000, 10000, 0.01, False, "NT_row_warp path (avg_nnz=100)"),
        (10000, 10000, 0.025, False, "NT_row_warp path (avg_nnz=250)"),

        # Large avg_nnz: NT_nz_thread (avg_nnz >= 512)
        (100000, 100000, 0.01, False, "NT_nz_thread path (avg_nnz=1K)"),
        (200000, 200000, 0.005, False, "NT_nz_thread path (avg_nnz=1K)"),

        # Transpose
        (100000, 100000, 0.01, True, "T_nz_thread (avg_nnz=1K)"),
        (200000, 200000, 0.005, True, "T_nz_thread (avg_nnz=1K)"),
    ]

    results = []
    for n_pre, n_post, prob, transpose, label in configs:
        print(f"Benchmarking: {label}")
        result = benchmark_kernel(n_pre, n_post, prob, transpose)
        results.append(result)
        print(f"  Time: {result['mean_ms']:.3f} ms  |  BW: {result['bandwidth_gb_s']:.1f} GB/s "
              f"({result['peak_bw_pct']:.1f}% of peak)  |  Efficiency: {result['efficiency_pct']:.1f}%")
        print(f"  Bound: {result['bound_type']}  |  AI: {result['arith_intensity']:.4f} FLOPs/byte  "
              f"|  Theoretical: {result['theoretical_ms']:.3f} ms")
        print()

    print("="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Config':<35} {'Time(ms)':<10} {'BW(GB/s)':<10} {'Peak%':<8} {'Eff%':<8} {'Bound':<10}")
    print("-"*100)
    for r in results:
        print(f"{r['config']:<35} {r['mean_ms']:<10.3f} {r['bandwidth_gb_s']:<10.1f} "
              f"{r['peak_bw_pct']:<8.1f} {r['efficiency_pct']:<8.1f} {r['bound_type']:<10}")
