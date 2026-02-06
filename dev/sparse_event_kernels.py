"""
Optimized Numba kernels for sparse event-driven matrix-vector/matrix operations.

Operations:
1. csr.T @ v - Transpose CSR times vector (scatter pattern)
2. csr @ v - CSR times vector (gather pattern)  
3. csr.T @ B - Transpose CSR times batch matrix (scatter pattern)
4. csr @ B - CSR times batch matrix (gather pattern)

All operations work with binary (0/1) spike vectors/matrices.
"""

import numpy as np
import numba
from numba import njit, prange
import time
from typing import Tuple

# =============================================================================
# Original Implementations (Baseline)
# =============================================================================

@njit(fastmath=True, cache=True)
def mv_csrT_v_baseline(weights, indices, indptr, v, posts):
    """Baseline: csr.T @ v (scatter pattern)"""
    posts[:] = 0.
    w = weights[0]
    for i in range(v.shape[0]):
        if v[i]:
            for j in range(indptr[i], indptr[i + 1]):
                posts[indices[j]] += w


@njit(parallel=True, fastmath=True, cache=True)
def mv_csr_v_baseline(weights, indices, indptr, v, posts):
    """Baseline: csr @ v (gather pattern)"""
    w = weights[0]
    for i in prange(indptr.shape[0] - 1):
        r = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            if v[indices[j]]:
                r += w
        posts[i] = r


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csrT_B_baseline(weights, indices, indptr, B, posts):
    """Baseline: csr.T @ B (scatter pattern, batched)"""
    w = weights[0]
    posts[:] = 0.
    for k in prange(B.shape[1]):
        for i in range(B.shape[0]):
            if B[i, k]:
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j], k] += w


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csr_B_baseline(weights, indices, indptr, B, posts):
    """Baseline: csr @ B (gather pattern, batched)"""
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


# =============================================================================
# Optimized: csr.T @ v (scatter pattern)
# =============================================================================

@njit(fastmath=True, cache=True)
def mv_csrT_v_opt1_compact(weights, indices, indptr, v, posts):
    """
    Optimization 1: Compact spike indices first, then process.
    Reduces branch mispredictions for sparse firing.
    """
    posts[:] = 0.
    w = weights[0]
    
    # Count active neurons
    count = 0
    for i in range(v.shape[0]):
        if v[i]:
            count += 1
    
    # Compact active indices
    active = np.empty(count, dtype=np.int64)
    idx = 0
    for i in range(v.shape[0]):
        if v[i]:
            active[idx] = i
            idx += 1
    
    # Process only active neurons
    for idx in range(count):
        i = active[idx]
        for j in range(indptr[i], indptr[i + 1]):
            posts[indices[j]] += w


@njit(fastmath=True, cache=True)
def mv_csrT_v_opt2_blocked(weights, indices, indptr, v, posts, block_size=64):
    """
    Optimization 2: Process in blocks to improve cache locality.
    """
    posts[:] = 0.
    w = weights[0]
    n_pre = v.shape[0]
    
    for block_start in range(0, n_pre, block_size):
        block_end = min(block_start + block_size, n_pre)
        for i in range(block_start, block_end):
            if v[i]:
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j]] += w


@njit(fastmath=True, cache=True)
def mv_csrT_v_opt3_sorted_access(weights, indices, indptr, v, posts):
    """
    Optimization 3: Sort post indices for better cache behavior.
    Only beneficial if indices within each row aren't already sorted.
    """
    posts[:] = 0.
    w = weights[0]
    
    # Collect all (post_idx, weight) pairs
    total_ops = 0
    for i in range(v.shape[0]):
        if v[i]:
            total_ops += indptr[i + 1] - indptr[i]
    
    if total_ops == 0:
        return
    
    post_indices = np.empty(total_ops, dtype=np.int32)
    idx = 0
    for i in range(v.shape[0]):
        if v[i]:
            for j in range(indptr[i], indptr[i + 1]):
                post_indices[idx] = indices[j]
                idx += 1
    
    # Sort for cache-friendly access
    post_indices.sort()
    
    # Accumulate
    for idx in range(total_ops):
        posts[post_indices[idx]] += w


# =============================================================================
# Optimized: csr @ v (gather pattern)
# =============================================================================

@njit(parallel=True, fastmath=True, cache=True)
def mv_csr_v_opt1_unrolled(weights, indices, indptr, v, posts):
    """
    Optimization 1: Loop unrolling for inner loop.
    """
    w = weights[0]
    for i in prange(indptr.shape[0] - 1):
        r = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        
        # Unroll by 4
        j = start
        while j + 4 <= end:
            if v[indices[j]]:
                r += w
            if v[indices[j + 1]]:
                r += w
            if v[indices[j + 2]]:
                r += w
            if v[indices[j + 3]]:
                r += w
            j += 4
        
        # Handle remainder
        while j < end:
            if v[indices[j]]:
                r += w
            j += 1
        
        posts[i] = r


@njit(parallel=True, fastmath=True, cache=True)
def mv_csr_v_opt2_branchless(weights, indices, indptr, v, posts):
    """
    Optimization 2: Branchless accumulation using multiplication.
    v[i] is 0 or 1, so we can multiply instead of branching.
    """
    w = weights[0]
    for i in prange(indptr.shape[0] - 1):
        r = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            r += w * v[indices[j]]
        posts[i] = r


@njit(parallel=True, fastmath=True, cache=True)
def mv_csr_v_opt3_compact_set(weights, indices, indptr, v, posts):
    """
    Optimization 3: Use boolean set for faster lookup.
    """
    w = weights[0]
    
    # Create boolean mask (already is one if v is 0/1)
    # Process with direct lookup
    for i in prange(indptr.shape[0] - 1):
        r = 0.0
        for j in range(indptr[i], indptr[i + 1]):
            r += w * v[indices[j]]
        posts[i] = r


@njit(parallel=True, fastmath=True, cache=True) 
def mv_csr_v_opt4_unrolled8(weights, indices, indptr, v, posts):
    """
    Optimization 4: Unroll by 8 with branchless.
    """
    w = weights[0]
    for i in prange(indptr.shape[0] - 1):
        r = 0.0
        start = indptr[i]
        end = indptr[i + 1]
        
        # Unroll by 8
        j = start
        while j + 8 <= end:
            r += w * v[indices[j]]
            r += w * v[indices[j + 1]]
            r += w * v[indices[j + 2]]
            r += w * v[indices[j + 3]]
            r += w * v[indices[j + 4]]
            r += w * v[indices[j + 5]]
            r += w * v[indices[j + 6]]
            r += w * v[indices[j + 7]]
            j += 8
        
        while j < end:
            r += w * v[indices[j]]
            j += 1
        
        posts[i] = r


# =============================================================================
# Optimized: csr.T @ B (scatter pattern, batched)
# =============================================================================

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csrT_B_opt1_compact(weights, indices, indptr, B, posts):
    """
    Optimization 1: Compact active indices per batch column.
    """
    w = weights[0]
    n_pre = B.shape[0]
    n_batch = B.shape[1]
    posts[:] = 0.
    
    for k in prange(n_batch):
        # Compact active pre-neurons for this batch
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
        
        # Process only active
        for idx in range(count):
            i = active[idx]
            for j in range(indptr[i], indptr[i + 1]):
                posts[indices[j], k] += w


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csrT_B_opt2_transposed(weights, indices, indptr, B_T, posts):
    """
    Optimization 2: Work with transposed B for better cache access.
    B_T has shape (n_batch, n_pre).
    """
    w = weights[0]
    n_batch = B_T.shape[0]
    n_pre = B_T.shape[1]
    posts[:] = 0.
    
    for k in prange(n_batch):
        for i in range(n_pre):
            if B_T[k, i]:
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j], k] += w


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csrT_B_opt3_blocked_batch(weights, indices, indptr, B, posts):
    """
    Optimization 3: Process multiple batch columns together for vectorization.
    """
    w = weights[0]
    n_pre = B.shape[0]
    n_batch = B.shape[1]
    posts[:] = 0.
    
    batch_block = 4
    n_full_blocks = n_batch // batch_block
    
    # Process full blocks
    for kb in prange(n_full_blocks):
        k_start = kb * batch_block
        for i in range(n_pre):
            # Check if any of the batch columns are active
            has_active = False
            for kk in range(batch_block):
                if B[i, k_start + kk]:
                    has_active = True
                    break
            
            if has_active:
                for j in range(indptr[i], indptr[i + 1]):
                    post_idx = indices[j]
                    for kk in range(batch_block):
                        if B[i, k_start + kk]:
                            posts[post_idx, k_start + kk] += w
    
    # Handle remainder
    k_start = n_full_blocks * batch_block
    for k in range(k_start, n_batch):
        for i in range(n_pre):
            if B[i, k]:
                for j in range(indptr[i], indptr[i + 1]):
                    posts[indices[j], k] += w


# =============================================================================
# Optimized: csr @ B (gather pattern, batched)
# =============================================================================

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csr_B_opt1_no_alloc(weights, indices, indptr, B, posts):
    """
    Optimization 1: Avoid allocation inside loop - write directly to posts.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    n_batch = B.shape[1]
    
    for i in prange(n_rows):
        # Zero this row
        for k in range(n_batch):
            posts[i, k] = 0.0
        
        for j in range(indptr[i], indptr[i + 1]):
            index = indices[j]
            for k in range(n_batch):
                if B[index, k]:
                    posts[i, k] += w


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csr_B_opt2_branchless(weights, indices, indptr, B, posts):
    """
    Optimization 2: Branchless using multiplication.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    n_batch = B.shape[1]
    
    for i in prange(n_rows):
        for k in range(n_batch):
            posts[i, k] = 0.0
        
        for j in range(indptr[i], indptr[i + 1]):
            index = indices[j]
            for k in range(n_batch):
                posts[i, k] += w * B[index, k]


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csr_B_opt3_unrolled(weights, indices, indptr, B, posts):
    """
    Optimization 3: Unroll batch dimension.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    n_batch = B.shape[1]
    
    for i in prange(n_rows):
        for k in range(n_batch):
            posts[i, k] = 0.0
        
        for j in range(indptr[i], indptr[i + 1]):
            index = indices[j]
            # Unroll by 4
            k = 0
            while k + 4 <= n_batch:
                posts[i, k] += w * B[index, k]
                posts[i, k + 1] += w * B[index, k + 1]
                posts[i, k + 2] += w * B[index, k + 2]
                posts[i, k + 3] += w * B[index, k + 3]
                k += 4
            while k < n_batch:
                posts[i, k] += w * B[index, k]
                k += 1


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csr_B_opt4_transposed(weights, indices, indptr, B_T, posts):
    """
    Optimization 4: Work with B transposed (n_batch, n_pre).
    Better cache locality for accessing B.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    n_batch = B_T.shape[0]
    
    for i in prange(n_rows):
        for k in range(n_batch):
            r = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                index = indices[j]
                r += w * B_T[k, index]
            posts[i, k] = r


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def mm_csr_B_opt5_row_major(weights, indices, indptr, B, posts):
    """
    Optimization 5: Process in row-major order with explicit vectorization hints.
    """
    w = weights[0]
    n_rows = indptr.shape[0] - 1
    n_batch = B.shape[1]
    
    # Zero output
    for i in prange(n_rows):
        for k in range(n_batch):
            posts[i, k] = 0.0
    
    # Accumulate row by row
    for i in prange(n_rows):
        for j in range(indptr[i], indptr[i + 1]):
            index = indices[j]
            # Process entire batch row at once
            for k in range(n_batch):
                posts[i, k] += w * B[index, k]


# =============================================================================
# Benchmarking Infrastructure
# =============================================================================

def create_test_data(n_pre: int, n_post: int, conn_prob: float, 
                     firing_rate: float, batch_size: int = 1,
                     dtype=np.float32) -> Tuple:
    """Create test data for benchmarking."""
    np.random.seed(42)
    
    nnz_per_row = int(n_post * conn_prob)
    indices_list = []
    indptr = [0]
    
    for i in range(n_pre):
        targets = np.random.choice(n_post, size=nnz_per_row, replace=False)
        targets.sort()
        indices_list.append(targets)
        indptr.append(indptr[-1] + nnz_per_row)
    
    indices = np.concatenate(indices_list).astype(np.int32)
    indptr = np.array(indptr, dtype=np.int32)
    weights = np.array([1.0], dtype=dtype)
    
    v = (np.random.rand(n_pre) < firing_rate).astype(dtype)
    B = (np.random.rand(n_pre, batch_size) < firing_rate).astype(dtype)
    
    posts_v = np.zeros(n_post, dtype=dtype)
    posts_B = np.zeros((n_post, batch_size), dtype=dtype)
    
    return weights, indices, indptr, v, B, posts_v, posts_B


def benchmark_function(func, args, n_warmup=3, n_runs=10, name=""):
    """Benchmark a function with warmup and multiple runs."""
    for _ in range(n_warmup):
        func(*args)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    return {
        'name': name,
        'mean': np.mean(times) * 1000,
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
    }


def run_correctness_tests():
    """Verify all optimizations produce correct results."""
    print("=" * 70)
    print("Correctness Verification")
    print("=" * 70)
    
    n_pre, n_post, conn_prob, firing_rate, batch_size = 500, 500, 0.05, 0.1, 8
    weights, indices, indptr, v, B, posts_v, posts_B = create_test_data(
        n_pre, n_post, conn_prob, firing_rate, batch_size
    )
    
    print("\ncsr.T @ v operations:")
    # Baseline result
    posts_baseline = posts_v.copy()
    mv_csrT_v_baseline(weights, indices, indptr, v, posts_baseline)
    
    for name, func in [
        ("Opt1: Compact", mv_csrT_v_opt1_compact),
        ("Opt2: Blocked", mv_csrT_v_opt2_blocked),
        ("Opt3: Sorted", mv_csrT_v_opt3_sorted_access),
    ]:
        posts_test = posts_v.copy()
        func(weights, indices, indptr, v, posts_test)
        correct = np.allclose(posts_baseline, posts_test, rtol=1e-5)
        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print("\ncsr @ v operations:")
    posts_baseline = posts_v.copy()
    mv_csr_v_baseline(weights, indices, indptr, v, posts_baseline)
    
    for name, func in [
        ("Opt1: Unrolled", mv_csr_v_opt1_unrolled),
        ("Opt2: Branchless", mv_csr_v_opt2_branchless),
        ("Opt3: Compact Set", mv_csr_v_opt3_compact_set),
        ("Opt4: Unrolled8", mv_csr_v_opt4_unrolled8),
    ]:
        posts_test = posts_v.copy()
        func(weights, indices, indptr, v, posts_test)
        correct = np.allclose(posts_baseline, posts_test, rtol=1e-5)
        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print("\ncsr.T @ B operations:")
    posts_baseline = posts_B.copy()
    mm_csrT_B_baseline(weights, indices, indptr, B, posts_baseline)
    
    for name, func, B_arg in [
        ("Opt1: Compact", mm_csrT_B_opt1_compact, B),
        ("Opt2: B Transposed", mm_csrT_B_opt2_transposed, B.T.copy()),
        ("Opt3: Blocked Batch", mm_csrT_B_opt3_blocked_batch, B),
    ]:
        posts_test = posts_B.copy()
        func(weights, indices, indptr, B_arg, posts_test)
        correct = np.allclose(posts_baseline, posts_test, rtol=1e-5)
        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print("\ncsr @ B operations:")
    posts_csr_B = np.zeros((n_pre, batch_size), dtype=weights.dtype)
    mm_csr_B_baseline(weights, indices, indptr, B, posts_csr_B)
    
    for name, func, B_arg in [
        ("Opt1: No Alloc", mm_csr_B_opt1_no_alloc, B),
        ("Opt2: Branchless", mm_csr_B_opt2_branchless, B),
        ("Opt3: Unrolled", mm_csr_B_opt3_unrolled, B),
        ("Opt4: B Transposed", mm_csr_B_opt4_transposed, B.T.copy()),
        ("Opt5: Row Major", mm_csr_B_opt5_row_major, B),
    ]:
        posts_test = np.zeros((n_pre, batch_size), dtype=weights.dtype)
        func(weights, indices, indptr, B_arg, posts_test)
        correct = np.allclose(posts_csr_B, posts_test, rtol=1e-5)
        status = "✓ PASS" if correct else "✗ FAIL"
        print(f"  {name}: {status}")


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("\n" + "=" * 70)
    print("Performance Benchmarks")
    print("=" * 70)
    
    # Test configurations
    configs = [
        (5000, 5000, 0.01, 0.01, 1, "5K neurons, 1% conn, 1% firing"),
        (5000, 5000, 0.01, 0.05, 1, "5K neurons, 1% conn, 5% firing"),
        (5000, 5000, 0.01, 0.10, 1, "5K neurons, 1% conn, 10% firing"),
        (10000, 10000, 0.01, 0.05, 1, "10K neurons, 1% conn, 5% firing"),
    ]
    
    batch_configs = [
        (5000, 5000, 0.01, 0.05, 16, "Batch=16"),
        (5000, 5000, 0.01, 0.05, 32, "Batch=32"),
        (5000, 5000, 0.01, 0.05, 64, "Batch=64"),
    ]
    
    # =========================================================================
    # csr.T @ v
    # =========================================================================
    print("\n" + "-" * 70)
    print("Operation: csr.T @ v (scatter)")
    print("-" * 70)
    
    funcs_csrT_v = [
        ("Baseline", mv_csrT_v_baseline),
        ("Compact", mv_csrT_v_opt1_compact),
        ("Blocked", mv_csrT_v_opt2_blocked),
        ("Sorted", mv_csrT_v_opt3_sorted_access),
    ]
    
    for config in configs:
        n_pre, n_post, conn_prob, firing_rate, batch_size, desc = config
        print(f"\n{desc}:")
        
        weights, indices, indptr, v, B, posts_v, posts_B = create_test_data(
            n_pre, n_post, conn_prob, firing_rate, batch_size
        )
        
        baseline_time = None
        for name, func in funcs_csrT_v:
            posts = posts_v.copy()
            result = benchmark_function(func, (weights, indices, indptr, v, posts), name=name)
            if baseline_time is None:
                baseline_time = result['mean']
            speedup = baseline_time / result['mean']
            print(f"  {name:12s}: {result['mean']:7.3f} ms  (speedup: {speedup:.2f}x)")
    
    # =========================================================================
    # csr @ v
    # =========================================================================
    print("\n" + "-" * 70)
    print("Operation: csr @ v (gather)")
    print("-" * 70)
    
    funcs_csr_v = [
        ("Baseline", mv_csr_v_baseline),
        ("Unrolled4", mv_csr_v_opt1_unrolled),
        ("Branchless", mv_csr_v_opt2_branchless),
        ("Unrolled8", mv_csr_v_opt4_unrolled8),
    ]
    
    for config in configs:
        n_pre, n_post, conn_prob, firing_rate, batch_size, desc = config
        print(f"\n{desc}:")
        
        weights, indices, indptr, v, B, posts_v, posts_B = create_test_data(
            n_pre, n_post, conn_prob, firing_rate, batch_size
        )
        
        baseline_time = None
        for name, func in funcs_csr_v:
            posts = posts_v.copy()
            result = benchmark_function(func, (weights, indices, indptr, v, posts), name=name)
            if baseline_time is None:
                baseline_time = result['mean']
            speedup = baseline_time / result['mean']
            print(f"  {name:12s}: {result['mean']:7.3f} ms  (speedup: {speedup:.2f}x)")
    
    # =========================================================================
    # csr.T @ B
    # =========================================================================
    print("\n" + "-" * 70)
    print("Operation: csr.T @ B (scatter, batched)")
    print("-" * 70)
    
    for config in batch_configs:
        n_pre, n_post, conn_prob, firing_rate, batch_size, desc = config
        print(f"\n{desc} (5K neurons, 1% conn, 5% firing):")
        
        weights, indices, indptr, v, B, posts_v, posts_B = create_test_data(
            n_pre, n_post, conn_prob, firing_rate, batch_size
        )
        
        funcs = [
            ("Baseline", mm_csrT_B_baseline, B),
            ("Compact", mm_csrT_B_opt1_compact, B),
            ("Transposed B", mm_csrT_B_opt2_transposed, B.T.copy()),
            ("Blocked", mm_csrT_B_opt3_blocked_batch, B),
        ]
        
        baseline_time = None
        for name, func, B_arg in funcs:
            posts = posts_B.copy()
            result = benchmark_function(func, (weights, indices, indptr, B_arg, posts), name=name)
            if baseline_time is None:
                baseline_time = result['mean']
            speedup = baseline_time / result['mean']
            print(f"  {name:12s}: {result['mean']:7.3f} ms  (speedup: {speedup:.2f}x)")
    
    # =========================================================================
    # csr @ B
    # =========================================================================
    print("\n" + "-" * 70)
    print("Operation: csr @ B (gather, batched)")
    print("-" * 70)
    
    for config in batch_configs:
        n_pre, n_post, conn_prob, firing_rate, batch_size, desc = config
        print(f"\n{desc} (5K neurons, 1% conn, 5% firing):")
        
        weights, indices, indptr, v, B, posts_v, posts_B = create_test_data(
            n_pre, n_post, conn_prob, firing_rate, batch_size
        )
        posts_csr_B = np.zeros((n_pre, batch_size), dtype=weights.dtype)
        
        funcs = [
            ("Baseline", mm_csr_B_baseline, B),
            ("No Alloc", mm_csr_B_opt1_no_alloc, B),
            ("Branchless", mm_csr_B_opt2_branchless, B),
            ("Unrolled", mm_csr_B_opt3_unrolled, B),
            ("Transposed", mm_csr_B_opt4_transposed, B.T.copy()),
        ]
        
        baseline_time = None
        for name, func, B_arg in funcs:
            posts = posts_csr_B.copy()
            result = benchmark_function(func, (weights, indices, indptr, B_arg, posts), name=name)
            if baseline_time is None:
                baseline_time = result['mean']
            speedup = baseline_time / result['mean']
            print(f"  {name:12s}: {result['mean']:7.3f} ms  (speedup: {speedup:.2f}x)")


if __name__ == "__main__":
    run_correctness_tests()
    run_benchmarks()
