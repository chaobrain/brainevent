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
Benchmark: Event-Driven Binary COO Sparse Matrix-Matrix Multiplication
======================================================================

Compares backend performance for ``binary_coomm``:
  - jax       : pure-JAX scatter-add reference
  - pallas    : JAX Pallas/Triton GPU kernels (2-D atomic grid)
  - cuda_raw    : custom CUDA kernels via CUDA
      - CT  (column-tiled, 1 warp/block):  n ≤ 64
      - WPE (warp-per-entry, 8 warps/blk): n > 64

Usage
-----
    # Run the full benchmark suite (default mode):
    python dev/coo/benchmark_binary_coomm.py

    # Sweep spike densities at fixed matrix size:
    python dev/coo/benchmark_binary_coomm.py --mode density

    # Sweep output column counts (n) to show CT/WPE dispatch boundary:
    python dev/coo/benchmark_binary_coomm.py --mode ncols

    # Verify numerical correctness across dtypes:
    python dev/coo/benchmark_binary_coomm.py --mode dtype

    # Vary number of warmup / timing iterations:
    python dev/coo/benchmark_binary_coomm.py --n_warmup 20 --n_runs 100
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the development version of brainevent is imported, not the installed one.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np

from brainevent import BenchmarkConfig
from brainevent._coo.binary import binary_coomm_p, binary_coomm_p_call

# ---------------------------------------------------------------------------
# Problem-size configuration tables
# ---------------------------------------------------------------------------

# Default: varied matrix sizes, connection probabilities, and n_cols
CONFIGS_DEFAULT = [
    # (n_pre, n_post, conn_prob, n_cols)
    (500,   500,   0.10,  10),   # small,  dense,   small-n  → CT regime
    (1000,  1000,  0.10,  10),   # medium, dense,   small-n  → CT regime
    (1000,  1000,  0.01,  10),   # medium, sparse,  small-n  → CT regime
    (5000,  5000,  0.01,  10),   # large,  sparse,  small-n  → CT regime
    (5000,  5000,  0.001, 10),   # large,  v-sparse, small-n → CT regime
    (1000,  1000,  0.10,  128),  # medium, dense,   large-n  → WPE regime
    (1000,  1000,  0.01,  128),  # medium, sparse,  large-n  → WPE regime
    (5000,  5000,  0.001, 128),  # large,  v-sparse, large-n → WPE regime
    (10000, 10000, 0.01,  64),   # xlarge, sparse,  n=64 boundary
    (10000, 10000, 0.02,  128),  # xlarge, moderate, large-n
]

# Density sweep: fixed matrix, varied spike rates and n_cols
CONFIGS_DENSITY = [
    (2000, 2000, p, nc)
    for p  in [0.001, 0.005, 0.01, 0.05, 0.10]
    for nc in [10, 50, 128]
]

# n_cols sweep: fixed matrix, varied output column counts
CONFIGS_NCOLS = [
    (2000, 2000, 0.01, nc)
    for nc in [1, 4, 16, 32, 64, 128, 256, 512]
]

_DTYPE = jnp.float32


# ---------------------------------------------------------------------------
# Benchmark data generator
# ---------------------------------------------------------------------------

def _make_benchmark_data(
    *,
    platform,
    configs=None,
    transpose_opts=(False, True),
    homo_opts=(True, False),
    bool_event_opts=(True, False),
    spike_rate=0.1,
):
    """Yield BenchmarkConfig instances for ``binary_coomm`` benchmarking.

    Parameters
    ----------
    platform : str
        Target compute platform (``'gpu'``).
    configs : list of (n_pre, n_post, prob, n_cols), optional
        Problem-size table.  Defaults to ``CONFIGS_DEFAULT``.
    transpose_opts : iterable of bool
        Whether to include NT / T directions.
    homo_opts : iterable of bool
        Whether to include homo / hetero weight variants.
    bool_event_opts : iterable of bool
        Whether to include bool / float spike variants.
    spike_rate : float
        Fraction of B entries that are active (for bool spikes).
    """
    if configs is None:
        configs = CONFIGS_DEFAULT

    rng = np.random.default_rng(42)

    for (n_pre, n_post, prob, n_cols) in configs:
        nnz = max(1, int(n_pre * n_post * prob))
        row = jnp.asarray(rng.integers(0, n_pre,  nnz, dtype=np.int32))
        col = jnp.asarray(rng.integers(0, n_post, nnz, dtype=np.int32))

        for transpose in transpose_opts:
            for homo in homo_opts:
                for bool_event in bool_event_opts:
                    weights = (
                        jnp.ones(1, dtype=_DTYPE) if homo
                        else jnp.asarray(rng.standard_normal(nnz), dtype=_DTYPE)
                    )
                    b_rows = n_post if not transpose else n_pre
                    if bool_event:
                        B = jnp.asarray(
                            rng.random((b_rows, n_cols)) < spike_rate,
                            dtype=jnp.bool_,
                        )
                    else:
                        raw  = rng.standard_normal((b_rows, n_cols))
                        mask = rng.random((b_rows, n_cols)) < spike_rate
                        B = jnp.asarray(np.where(mask, np.abs(raw), 0.0), dtype=_DTYPE)

                    dir_tag = 'T' if transpose else 'NT'
                    w_tag   = 'homo' if homo else 'hetero'
                    s_tag   = 'bool' if bool_event else 'float'
                    name = (
                        f"{dir_tag},{w_tag},{s_tag},"
                        f"{n_pre}x{n_post},p={prob:.3f},n={n_cols}"
                    )
                    yield BenchmarkConfig(
                        name=name,
                        args=(weights, row, col, B),
                        kernel_kwargs={'shape': (n_pre, n_post), 'transpose': transpose},
                        data_kwargs={
                            'n_pre': n_pre, 'n_post': n_post,
                            'nnz': nnz, 'n_cols': n_cols,
                            'prob': prob, 'transpose': transpose,
                        },
                    )


# ---------------------------------------------------------------------------
# Automated benchmark using XLACustomKernel infrastructure
# ---------------------------------------------------------------------------

def run_automated(n_warmup: int = 10, n_runs: int = 50, configs=None, spike_rate=0.1):
    """Run benchmark using the built-in ``XLACustomKernel`` benchmark machinery.

    Compares all registered GPU backends side-by-side and highlights the
    fastest one.  Speedup is reported relative to the ``jax`` reference.
    """
    print("\n" + "=" * 70)
    print("binary_coomm  — automated benchmark (GPU)")
    print("=" * 70)

    def _data_gen(*, platform):
        yield from _make_benchmark_data(
            platform=platform,
            configs=configs or CONFIGS_DEFAULT,
            transpose_opts=(False,),
            homo_opts=(True,),
            bool_event_opts=(True,),
            spike_rate=spike_rate,
        )

    binary_coomm_p.def_benchmark_data(_data_gen)
    result = binary_coomm_p.benchmark(
        platform='gpu',
        n_warmup=n_warmup,
        n_runs=n_runs,
        verbose=True,
    )
    result.print(vary_by='backend', highlight_best=True, speedup_vs='cusparse')
    return result


# ---------------------------------------------------------------------------
# Manual micro-benchmark with per-backend timing
# ---------------------------------------------------------------------------

def _measure_us(fn_jit, args, n_warmup, n_runs):
    """Return (median_µs, std_µs) for the given jit-compiled function."""
    for _ in range(n_warmup):
        fn_jit(*args).block_until_ready()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn_jit(*args).block_until_ready()
        times.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(times)), float(np.std(times))


def run_manual(
    n_warmup: int = 20,
    n_runs: int = 100,
    configs=None,
    spike_rate: float = 0.1,
    backends=('jax', 'pallas', 'cuda_raw'),
):
    """Manual micro-benchmark comparing backends with stable timing statistics.

    Reports median kernel time in µs and speedup vs. the ``jax`` baseline.
    This bypasses the ``XLACustomKernel`` timing machinery for direct
    wall-clock measurements with explicit block_until_ready synchronisation.
    """
    if configs is None:
        configs = CONFIGS_DEFAULT

    print("\n" + "=" * 70)
    print(f"binary_coomm  — manual micro-benchmark (GPU)")
    print(f"warmup={n_warmup}, runs={n_runs}, spike_rate={spike_rate:.1%}")
    print("=" * 70)

    col_w = 14
    header_parts = [f"{'config':<50s}"]
    for b in backends:
        header_parts.append(f"{b:>{col_w}s}")
    for b in backends[1:]:
        header_parts.append(f"{'x_vs_jax_' + b:>{col_w}s}")
    header = "  ".join(header_parts)
    print(header)
    print("-" * len(header))

    rng = np.random.default_rng(0)

    for (n_pre, n_post, prob, n_cols) in configs:
        nnz = max(1, int(n_pre * n_post * prob))
        row = jnp.asarray(rng.integers(0, n_pre,  nnz, dtype=np.int32))
        col = jnp.asarray(rng.integers(0, n_post, nnz, dtype=np.int32))
        weights = jnp.ones(1, dtype=_DTYPE)
        B = jnp.asarray(
            rng.random((n_post, n_cols)) < spike_rate, dtype=jnp.bool_
        )
        shape = (n_pre, n_post)

        jit_fns = {
            b: jax.jit(
                lambda w, r, c, mat, bk=b: binary_coomm_p_call(
                    w, r, c, mat, shape=shape, transpose=False, backend=bk
                )[0]
            )
            for b in backends
        }

        times = {}
        for b in backends:
            try:
                med, _ = _measure_us(jit_fns[b], (weights, row, col, B), n_warmup, n_runs)
                times[b] = med
            except Exception as exc:
                print(f"  [{b}] FAILED: {exc}")
                times[b] = float('nan')

        t_jax = times.get('jax', float('nan'))
        cfg_str = f"NT,homo,bool,{n_pre}x{n_post},p={prob:.3f},n={n_cols}"
        parts = [f"{cfg_str:<50s}"]
        for b in backends:
            parts.append(f"{times[b]:>{col_w}.1f}")
        for b in backends[1:]:
            sp = t_jax / times[b] if times[b] > 0 else float('nan')
            parts.append(f"{sp:>{col_w}.2f}")
        print("  ".join(parts))

    print("-" * len(header))
    print("  (times in µs, median;  speedup = jax_time / backend_time,  >1 = faster)")


# ---------------------------------------------------------------------------
# Spike density sweep
# ---------------------------------------------------------------------------

def run_density_sweep(n_warmup=20, n_runs=100, spike_rate=0.1):
    """Density sweep: show how event-driven savings scale with spike activity."""
    print("\n" + "=" * 70)
    print("binary_coomm  — spike density sweep")
    print("Matrix: 2000×2000, varied probabilities and n_cols")
    print("=" * 70)
    run_manual(
        n_warmup=n_warmup, n_runs=n_runs,
        configs=CONFIGS_DENSITY, spike_rate=spike_rate,
    )


# ---------------------------------------------------------------------------
# n_cols sweep (CT / WPE dispatch boundary)
# ---------------------------------------------------------------------------

def run_ncols_sweep(n_warmup=20, n_runs=100, spike_rate=0.1):
    """n_cols sweep: illustrate CT→WPE dispatch boundary at n=64."""
    print("\n" + "=" * 70)
    print("binary_coomm  — output column (n) sweep")
    print("Matrix: 2000×2000, p=0.01, varied n_cols")
    print("Dispatch: CT kernel for n≤64, WPE kernel for n>64")
    print("=" * 70)
    run_manual(
        n_warmup=n_warmup, n_runs=n_runs,
        configs=CONFIGS_NCOLS, spike_rate=spike_rate,
    )


# ---------------------------------------------------------------------------
# Dtype correctness check
# ---------------------------------------------------------------------------

def run_dtype_check():
    """Verify numerical correctness of cuda_raw vs. jax across all supported dtypes."""
    print("\n" + "=" * 70)
    print("binary_coomm  — dtype correctness check")
    print("=" * 70)

    rng = np.random.default_rng(123)
    n_pre, n_post, prob, n_cols = 500, 500, 0.05, 32
    nnz = max(1, int(n_pre * n_post * prob))
    row_np = rng.integers(0, n_pre,  nnz, dtype=np.int32)
    col_np = rng.integers(0, n_post, nnz, dtype=np.int32)
    row = jnp.asarray(row_np)
    col = jnp.asarray(col_np)

    weight_dtypes = [jnp.float32, jnp.float16, jnp.bfloat16, jnp.float64]
    spike_dtypes  = [jnp.bool_, jnp.float32]

    all_pass = True
    for wt_dtype in weight_dtypes:
        for spk_dtype in spike_dtypes:
            w_np = rng.standard_normal(nnz).astype(np.float32)
            weights = jnp.asarray(w_np, dtype=wt_dtype)

            B_np = rng.random((n_post, n_cols))
            if spk_dtype == jnp.bool_:
                B = jnp.asarray(B_np > 0.9, dtype=jnp.bool_)
            else:
                B = jnp.asarray(B_np, dtype=spk_dtype)

            shape = (n_pre, n_post)
            ref = binary_coomm_p_call(
                weights, row, col, B, shape=shape, transpose=False, backend='jax',
            )[0]

            try:
                cuda_out = binary_coomm_p_call(
                    weights, row, col, B, shape=shape, transpose=False, backend='cuda_raw',
                )[0]
                ref32  = ref.astype(jnp.float32)
                cuda32 = cuda_out.astype(jnp.float32)
                # f16/bf16 accumulate in f32 but final output is reduced precision
                tol = 1e-2 if wt_dtype in (jnp.float16, jnp.bfloat16) else 1e-4
                max_err = float(jnp.max(jnp.abs(ref32 - cuda32)))
                ok = max_err <= tol
                status = f"PASS  max_err={max_err:.3e}"
            except Exception as exc:
                ok = False
                status = f"ERROR: {exc}"

            if not ok:
                all_pass = False
            wt_name  = str(wt_dtype).split('.')[-1]
            spk_name = 'bool' if spk_dtype == jnp.bool_ else 'float32'
            flag = '✓' if ok else '✗'
            print(f"  [{flag}] weight={wt_name:<8s} spike={spk_name:<8s} → {status}")

    print()
    print("All dtypes: " + ("PASS" if all_pass else "FAIL"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark binary_coomm CUDA kernels (CT and WPE variants)",
    )
    parser.add_argument(
        "--mode",
        choices=["default", "density", "ncols", "dtype", "all"],
        default="default",
        help="Benchmark mode (default: default)",
    )
    parser.add_argument("--n_warmup",   type=int,   default=20)
    parser.add_argument("--n_runs",     type=int,   default=100)
    parser.add_argument("--spike_rate", type=float, default=0.1,
                        help="Fraction of active spikes (default: 10%%)")
    args = parser.parse_args()

    print(f"JAX devices:         {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

    kw = dict(n_warmup=args.n_warmup, n_runs=args.n_runs, spike_rate=args.spike_rate)

    if args.mode in ("default", "all"):
        run_automated(**kw)
        run_manual(**kw)

    if args.mode in ("density", "all"):
        run_density_sweep(**kw)

    if args.mode in ("ncols", "all"):
        run_ncols_sweep(**kw)

    if args.mode in ("dtype", "all"):
        run_dtype_check()


if __name__ == "__main__":
    main()
