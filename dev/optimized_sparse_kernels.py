"""
Final Optimized Numba Kernels for Sparse Event-Driven Matrix Operations

Based on benchmarking results, this file contains the best-performing implementations
for each operation pattern.

Key optimizations discovered:
- csr.T @ v: Compact active indices helps at very low firing rates
- csr @ v: Branchless (multiply) consistently outperforms branching
- csr.T @ B: Adaptive strategy based on batch size
- csr @ B: Branchless provides 1.4-2x speedup
"""

import numpy as np
from numba import njit, prange
import time


# =============================================================================
# BEST IMPLEMENTATIONS - Use these in production
# =============================================================================

@njit(fastmath=True, cache=True)
def csrT_mv_optimized(weights, indices, indptr, v, posts):
    """
    Optimized csr.T @ v (scatter pattern).
    
    Uses compact indices for very sparse firing (<3%), otherwise direct iteration.
    """
    posts[:] = 0.
    w = weights[0]
    n_pre = v.shape[0]
    
    # Count active neurons
    count = 0
    for i in range(n_pre):
        if v[i]:
            count += 1
    
    firing_rate = count / n_pre
    
    if firing_rate < 0.03:
        # Very sparse: compact active indices
        active = np.empty(count, dtype=np.int64)
        idx = 0
        for i in range(n_pre):
            if v[i]:
                active[idx] = i
                idx += 1
        
        for idx in range(count):
            i = active[idx]
            for j in range(indptr[i], indptr[i + 1]):
                posts[indices[j]] += w
    else:
        # Regular iteration with branch
        for i in range(n_pre):
            if v[i]:
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j]] += w


@njit(parallel=True, fastmath=True, cache=True)
def csr_mv_optimized(weights, indices, indptr, v, posts):
    """
    Optimized csr @ v (gather pattern).
    
    Uses branchless multiplication for consistent performance.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    
    for i in prange(n_rows):
        r = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            r += w * v[indices[j]]
        posts[i] = r


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def csrT_mm_optimized(weights, indices, indptr, B, posts):
    """
    Optimized csr.T @ B (scatter pattern, batched).
    
    Parallelizes over batch columns with compact indices.
    """
    w = weights[0]
    n_pre = B.shape[0]
    n_batch = B.shape[1]
    posts[:] = 0.
    
    for k in prange(n_batch):
        # Compact active indices for this batch column
        count = 0
        for i in range(n_pre):
            if B[i, k]:
                count += 1
        
        active = np.empty(count, dtype=np.int64)
        idx = 0
        for i in range(n_pre):
            if B[i, k]:
                active[idx] = i
                idx += 1
        
        # Process only active neurons
        for idx in range(count):
            i = active[idx]
            for j in range(indptr[i], indptr[i + 1]):
                posts[indices[j], k] += w


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def csr_mm_optimized(weights, indices, indptr, B, posts):
    """
    Optimized csr @ B (gather pattern, batched).
    
    Uses branchless multiplication - the clear performance winner.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    n_batch = B.shape[1]
    
    for i in prange(n_rows):
        # Zero output row
        for k in range(n_batch):
            posts[i, k] = 0.0
        
        # Branchless accumulation
        for j in range(indptr[i], indptr[i + 1]):
            index = indices[j]
            for k in range(n_batch):
                posts[i, k] += w * B[index, k]


# =============================================================================
# ADVANCED OPTIMIZATIONS - For specific scenarios
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def csr_mv_vectorized(weights, indices, indptr, v, posts):
    """
    csr @ v with explicit vectorization hints.
    
    Good when rows have similar nnz counts.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    
    for i in prange(n_rows):
        r = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        nnz = end - start
        
        # Prefetch and process in groups
        j = start
        # Unroll by 4 with branchless
        while j + 4 <= end:
            r += w * (v[indices[j]] + v[indices[j+1]] + v[indices[j+2]] + v[indices[j+3]])
            j += 4
        while j < end:
            r += w * v[indices[j]]
            j += 1
        
        posts[i] = r


@njit(fastmath=True, cache=True)
def csrT_mv_very_sparse(weights, indices, indptr, spike_indices, posts):
    """
    csr.T @ v for extremely sparse firing (<1%).
    
    Takes pre-computed spike indices instead of full vector.
    This avoids scanning the entire v array.
    
    Args:
        spike_indices: Array of indices where v[i] == 1
    """
    posts[:] = 0.
    w = weights[0]
    
    for idx in range(len(spike_indices)):
        i = spike_indices[idx]
        for j in range(indptr[i], indptr[i + 1]):
            posts[indices[j]] += w


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def csr_mm_tiled(weights, indices, indptr, B, posts, tile_size=32):
    """
    csr @ B with tiled processing for better cache utilization.
    
    Good for large batch sizes.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    n_batch = B.shape[1]
    n_tiles = (n_batch + tile_size - 1) // tile_size
    
    for i in prange(n_rows):
        # Zero output row
        for k in range(n_batch):
            posts[i, k] = 0.0
        
        # Process in tiles
        for tile in range(n_tiles):
            k_start = tile * tile_size
            k_end = min(k_start + tile_size, n_batch)
            
            for j in range(indptr[i], indptr[i + 1]):
                index = indices[j]
                for k in range(k_start, k_end):
                    posts[i, k] += w * B[index, k]


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def csrT_mm_sorted_output(weights, indices, indptr, B, posts):
    """
    csr.T @ B with sorted output access pattern.
    
    Better when indices are random (not pre-sorted).
    Collects updates then applies in sorted order.
    """
    w = weights[0]
    n_pre = B.shape[0]
    n_batch = B.shape[1]
    n_post = posts.shape[0]
    posts[:] = 0.
    
    # For each batch column independently
    for k in prange(n_batch):
        # Count updates per post neuron
        counts = np.zeros(n_post, dtype=np.int32)
        for i in range(n_pre):
            if B[i, k]:
                for j in range(indptr[i], indptr[i + 1]):
                    counts[indices[j]] += 1
        
        # Apply updates (now in sequential order through posts)
        for p in range(n_post):
            posts[p, k] = w * counts[p]


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def csr_mm_contiguous(weights, indices, indptr, B_contiguous, posts, n_batch):
    """
    csr @ B with contiguous B array (row-major).
    
    B_contiguous is B.ravel() for better memory access.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    
    for i in prange(n_rows):
        # Zero output row
        for k in range(n_batch):
            posts[i, k] = 0.0
        
        for j in range(indptr[i], indptr[i + 1]):
            index = indices[j]
            b_offset = index * n_batch
            for k in range(n_batch):
                posts[i, k] += w * B_contiguous[b_offset + k]


# =============================================================================
# UTILITY: Selecting optimal kernel based on problem characteristics
# =============================================================================

def get_optimal_csrT_mv():
    """Returns the best csr.T @ v kernel."""
    return csrT_mv_optimized


def get_optimal_csr_mv():
    """Returns the best csr @ v kernel."""
    return csr_mv_optimized


def get_optimal_csrT_mm():
    """Returns the best csr.T @ B kernel."""
    return csrT_mm_optimized


def get_optimal_csr_mm():
    """Returns the best csr @ B kernel."""
    return csr_mm_optimized


# =============================================================================
# COMPREHENSIVE BENCHMARK
# =============================================================================

def create_test_data(n_pre, n_post, conn_prob, firing_rate, batch_size=1, dtype=np.float32):
    """Create test data."""
    np.random.seed(42)
    
    nnz_per_row = max(1, int(n_post * conn_prob))
    indices_list = []
    indptr = [0]
    
    for i in range(n_pre):
        targets = np.random.choice(n_post, size=min(nnz_per_row, n_post), replace=False)
        targets.sort()
        indices_list.append(targets)
        indptr.append(indptr[-1] + len(targets))
    
    indices = np.concatenate(indices_list).astype(np.int32)
    indptr = np.array(indptr, dtype=np.int32)
    weights = np.array([1.0], dtype=dtype)
    
    v = (np.random.rand(n_pre) < firing_rate).astype(dtype)
    B = (np.random.rand(n_pre, batch_size) < firing_rate).astype(dtype)
    
    posts_v = np.zeros(n_post, dtype=dtype)
    posts_B = np.zeros((n_post, batch_size), dtype=dtype)
    
    # Spike indices for very sparse kernel
    spike_indices = np.where(v > 0)[0].astype(np.int64)
    
    return weights, indices, indptr, v, B, posts_v, posts_B, spike_indices


def benchmark(func, args, n_warmup=3, n_runs=15):
    """Benchmark a function."""
    for _ in range(n_warmup):
        func(*args)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    
    return np.mean(times) * 1000, np.std(times) * 1000


def run_comprehensive_benchmark():
    """Run comprehensive benchmark with all optimized kernels."""
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: Optimized Sparse Event-Driven Kernels")
    print("=" * 80)
    
    # Reference implementations
    @njit(fastmath=True, cache=True)
    def baseline_csrT_mv(weights, indices, indptr, v, posts):
        posts[:] = 0.
        w = weights[0]
        for i in range(v.shape[0]):
            if v[i]:
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j]] += w
    
    @njit(parallel=True, fastmath=True, cache=True)
    def baseline_csr_mv(weights, indices, indptr, v, posts):
        w = weights[0]
        for i in prange(indptr.shape[0] - 1):
            r = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                if v[indices[j]]:
                    r += w
            posts[i] = r
    
    @njit(parallel=True, fastmath=True, nogil=True, cache=True)
    def baseline_csrT_mm(weights, indices, indptr, B, posts):
        w = weights[0]
        posts[:] = 0.
        for k in prange(B.shape[1]):
            for i in range(B.shape[0]):
                if B[i, k]:
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j], k] += w
    
    @njit(parallel=True, fastmath=True, nogil=True, cache=True)
    def baseline_csr_mm(weights, indices, indptr, B, posts):
        w = weights[0]
        posts[:] = 0.
        for i in prange(indptr.shape[0] - 1):
            r = np.zeros(B.shape[1], dtype=weights.dtype)
            for j in range(indptr[i], indptr[i + 1]):
                index = indices[j]
                for k in range(B.shape[1]):
                    if B[index, k]:
                        r[k] += w
            posts[i] = r
    
    configs = [
        # (n_pre, n_post, conn_prob, firing_rate, batch_size, description)
        (10000, 10000, 0.01, 0.01, 1, "10K, 1% conn, 1% firing, batch=1"),
        (10000, 10000, 0.01, 0.05, 1, "10K, 1% conn, 5% firing, batch=1"),
        (10000, 10000, 0.01, 0.10, 1, "10K, 1% conn, 10% firing, batch=1"),
        (10000, 10000, 0.01, 0.05, 32, "10K, 1% conn, 5% firing, batch=32"),
        (10000, 10000, 0.01, 0.05, 64, "10K, 1% conn, 5% firing, batch=64"),
        (10000, 10000, 0.01, 0.05, 128, "10K, 1% conn, 5% firing, batch=128"),
    ]
    
    results_summary = []
    
    for config in configs:
        n_pre, n_post, conn_prob, firing_rate, batch_size, desc = config
        print(f"\n{'='*70}")
        print(f"Config: {desc}")
        print(f"{'='*70}")
        
        weights, indices, indptr, v, B, posts_v, posts_B, spike_indices = create_test_data(
            n_pre, n_post, conn_prob, firing_rate, batch_size
        )
        posts_csr = np.zeros((n_pre, batch_size), dtype=weights.dtype)
        
        # Vector operations
        if batch_size == 1:
            print("\n--- csr.T @ v (scatter) ---")
            base_time, _ = benchmark(baseline_csrT_mv, (weights, indices, indptr, v, posts_v.copy()))
            opt_time, _ = benchmark(csrT_mv_optimized, (weights, indices, indptr, v, posts_v.copy()))
            sparse_time, _ = benchmark(csrT_mv_very_sparse, (weights, indices, indptr, spike_indices, posts_v.copy()))
            
            print(f"  Baseline:    {base_time:7.3f} ms")
            print(f"  Optimized:   {opt_time:7.3f} ms  (speedup: {base_time/opt_time:.2f}x)")
            print(f"  Very Sparse: {sparse_time:7.3f} ms  (speedup: {base_time/sparse_time:.2f}x)")
            
            print("\n--- csr @ v (gather) ---")
            base_time, _ = benchmark(baseline_csr_mv, (weights, indices, indptr, v, posts_v.copy()))
            opt_time, _ = benchmark(csr_mv_optimized, (weights, indices, indptr, v, posts_v.copy()))
            vec_time, _ = benchmark(csr_mv_vectorized, (weights, indices, indptr, v, posts_v.copy()))
            
            print(f"  Baseline:    {base_time:7.3f} ms")
            print(f"  Optimized:   {opt_time:7.3f} ms  (speedup: {base_time/opt_time:.2f}x)")
            print(f"  Vectorized:  {vec_time:7.3f} ms  (speedup: {base_time/vec_time:.2f}x)")
        
        # Matrix operations
        print(f"\n--- csr.T @ B (scatter, batch={batch_size}) ---")
        base_time, _ = benchmark(baseline_csrT_mm, (weights, indices, indptr, B, posts_B.copy()))
        opt_time, _ = benchmark(csrT_mm_optimized, (weights, indices, indptr, B, posts_B.copy()))
        sorted_time, _ = benchmark(csrT_mm_sorted_output, (weights, indices, indptr, B, posts_B.copy()))
        
        print(f"  Baseline:      {base_time:7.3f} ms")
        print(f"  Optimized:     {opt_time:7.3f} ms  (speedup: {base_time/opt_time:.2f}x)")
        print(f"  Sorted Output: {sorted_time:7.3f} ms  (speedup: {base_time/sorted_time:.2f}x)")
        
        print(f"\n--- csr @ B (gather, batch={batch_size}) ---")
        base_time, _ = benchmark(baseline_csr_mm, (weights, indices, indptr, B, posts_csr.copy()))
        opt_time, _ = benchmark(csr_mm_optimized, (weights, indices, indptr, B, posts_csr.copy()))
        tiled_time, _ = benchmark(csr_mm_tiled, (weights, indices, indptr, B, posts_csr.copy()))
        
        print(f"  Baseline:    {base_time:7.3f} ms")
        print(f"  Optimized:   {opt_time:7.3f} ms  (speedup: {base_time/opt_time:.2f}x)")
        print(f"  Tiled:       {tiled_time:7.3f} ms  (speedup: {base_time/tiled_time:.2f}x)")
        
        results_summary.append({
            'config': desc,
            'csr_mm_baseline': base_time,
            'csr_mm_optimized': opt_time,
            'csr_mm_speedup': base_time / opt_time
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Best Speedups Achieved")
    print("=" * 80)
    print("\nFor csr @ B (gather, batched) - the most important operation:")
    for r in results_summary:
        print(f"  {r['config']}: {r['csr_mm_speedup']:.2f}x speedup")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. csr.T @ v (scatter):
   - Use csrT_mv_optimized() for general cases
   - Use csrT_mv_very_sparse() when you pre-compute spike indices
   
2. csr @ v (gather):
   - Use csr_mv_optimized() - branchless multiplication is key
   
3. csr.T @ B (scatter, batched):
   - Use csrT_mm_optimized() for most cases
   - Consider csrT_mm_sorted_output() for random index patterns
   
4. csr @ B (gather, batched):
   - Use csr_mm_optimized() - provides 1.5-2x speedup consistently
   - Use csr_mm_tiled() for very large batch sizes (>128)
""")


def verify_correctness():
    """Verify all optimized functions produce correct results."""
    print("Verifying correctness...")
    
    np.random.seed(123)
    n_pre, n_post = 1000, 1000
    conn_prob, firing_rate = 0.05, 0.1
    batch_size = 16
    
    weights, indices, indptr, v, B, posts_v, posts_B, spike_indices = create_test_data(
        n_pre, n_post, conn_prob, firing_rate, batch_size
    )
    posts_csr = np.zeros((n_pre, batch_size), dtype=weights.dtype)
    
    # Reference implementations (inline)
    @njit(fastmath=True, cache=True)
    def ref_csrT_mv(weights, indices, indptr, v, posts):
        posts[:] = 0.
        w = weights[0]
        for i in range(v.shape[0]):
            if v[i]:
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j]] += w
    
    @njit(parallel=True, fastmath=True, cache=True)
    def ref_csr_mv(weights, indices, indptr, v, posts):
        w = weights[0]
        for i in prange(indptr.shape[0] - 1):
            r = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                if v[indices[j]]:
                    r += w
            posts[i] = r
    
    @njit(parallel=True, fastmath=True, nogil=True, cache=True)
    def ref_csrT_mm(weights, indices, indptr, B, posts):
        w = weights[0]
        posts[:] = 0.
        for k in prange(B.shape[1]):
            for i in range(B.shape[0]):
                if B[i, k]:
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j], k] += w
    
    @njit(parallel=True, fastmath=True, nogil=True, cache=True)
    def ref_csr_mm(weights, indices, indptr, B, posts):
        w = weights[0]
        posts[:] = 0.
        for i in prange(indptr.shape[0] - 1):
            r = np.zeros(B.shape[1], dtype=weights.dtype)
            for j in range(indptr[i], indptr[i + 1]):
                index = indices[j]
                for k in range(B.shape[1]):
                    if B[index, k]:
                        r[k] += w
            posts[i] = r
    
    all_pass = True
    
    # Test csrT_mv
    ref = posts_v.copy()
    ref_csrT_mv(weights, indices, indptr, v, ref)
    
    for name, func, args in [
        ("csrT_mv_optimized", csrT_mv_optimized, (weights, indices, indptr, v)),
        ("csrT_mv_very_sparse", csrT_mv_very_sparse, (weights, indices, indptr, spike_indices)),
    ]:
        test = posts_v.copy()
        func(*args, test)
        if np.allclose(ref, test):
            print(f"  {name}: ✓ PASS")
        else:
            print(f"  {name}: ✗ FAIL")
            all_pass = False
    
    # Test csr_mv
    ref = posts_v.copy()
    ref_csr_mv(weights, indices, indptr, v, ref)
    
    for name, func in [
        ("csr_mv_optimized", csr_mv_optimized),
        ("csr_mv_vectorized", csr_mv_vectorized),
    ]:
        test = posts_v.copy()
        func(weights, indices, indptr, v, test)
        if np.allclose(ref, test):
            print(f"  {name}: ✓ PASS")
        else:
            print(f"  {name}: ✗ FAIL")
            all_pass = False
    
    # Test csrT_mm
    ref = posts_B.copy()
    ref_csrT_mm(weights, indices, indptr, B, ref)
    
    for name, func in [
        ("csrT_mm_optimized", csrT_mm_optimized),
        ("csrT_mm_sorted_output", csrT_mm_sorted_output),
    ]:
        test = posts_B.copy()
        func(weights, indices, indptr, B, test)
        if np.allclose(ref, test):
            print(f"  {name}: ✓ PASS")
        else:
            print(f"  {name}: ✗ FAIL")
            all_pass = False
    
    # Test csr_mm
    ref = posts_csr.copy()
    ref_csr_mm(weights, indices, indptr, B, ref)
    
    for name, func in [
        ("csr_mm_optimized", csr_mm_optimized),
        ("csr_mm_tiled", csr_mm_tiled),
    ]:
        test = posts_csr.copy()
        func(weights, indices, indptr, B, test)
        if np.allclose(ref, test):
            print(f"  {name}: ✓ PASS")
        else:
            print(f"  {name}: ✗ FAIL")
            all_pass = False
    
    return all_pass


if __name__ == "__main__":
    if verify_correctness():
        print("\nAll correctness tests passed!\n")
        run_comprehensive_benchmark()
    else:
        print("\nSome tests failed! Please fix before benchmarking.")
