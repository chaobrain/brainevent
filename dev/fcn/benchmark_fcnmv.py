"""
FCN Matrix-Vector Multiplication Benchmark
==========================================

Compares all available backends for ``fcnmv`` (gather and scatter modes,
homo and hetero weights) across a range of problem sizes.

Usage
-----
    # Full benchmark suite (all configs)
    python dev/fcn/benchmark_fcnmv.py

    # Quick run with fewer iterations
    python dev/fcn/benchmark_fcnmv.py --n_warmup 5 --n_runs 50

    # Only benchmark specific modes
    python dev/fcn/benchmark_fcnmv.py --mode gather
    python dev/fcn/benchmark_fcnmv.py --mode scatter

Requirements
------------
    pip install jax jaxlib brainevent
    # For tvmffi backend:
    pip install jax-tvm-ffi tvm-ffi
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure the development brainevent package (project root) takes precedence
# over any installed version, so that local changes are picked up when
# running the script directly (e.g. ``python dev/fcn/benchmark_fcnmv.py``).
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp
import numpy as np

from brainevent import fcnmv, fcnmv_p

# ---------------------------------------------------------------------------
# Problem configs: (n_pre, n_post, n_conn, label)
# ---------------------------------------------------------------------------
CONFIGS = [
    (500,   1000,  10,   "small  (500×10)"),
    (1000,  1000,  50,   "medium (1K×50)"),
    (1000,  1000,  100,  "medium (1K×100)"),
    (1000,  1000,  128,  "medium (1K×128, vec4)"),
    (5000,  5000,  200,  "large  (5K×200)"),
    (5000,  5000,  500,  "large  (5K×500)"),
    (10000, 10000, 1000, "xlarge (10K×1K)"),
]

# Backends to probe (skipped gracefully if unavailable)
BACKENDS = fcnmv_p.available_backends('gpu')


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _time_fn(fn, args, n_warmup: int, n_runs: int) -> Dict:
    """Run fn(*args), return timing stats in µs."""
    # warmup
    for _ in range(n_warmup):
        out = fn(*args)
    jax.block_until_ready(out)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    arr = np.array(times) * 1e6  # → µs
    return {
        "mean":   arr.mean(),
        "median": float(np.median(arr)),
        "std":    arr.std(),
        "min":    arr.min(),
    }


def _make_jit(weights, indices, vector, shape, transpose, backend):
    @jax.jit
    def fn(w, idx, v):
        return fcnmv(w, idx, v, shape=shape, transpose=transpose, backend=backend)
    return fn


# ---------------------------------------------------------------------------
# Reference (pure JAX, no custom backend)
# ---------------------------------------------------------------------------

def _jax_reference(weights, indices, vector, *, shape, transpose):
    n_pre, n_post = shape
    n_conn = indices.shape[1]
    homo = weights.shape[0] == 1
    w = weights[0] if homo else weights

    if not transpose:
        # gather: y[i] = sum_k w[i,k] * v[idx[i,k]]
        if homo:
            return jax.vmap(lambda ind: w * jnp.sum(vector[ind]))(indices)
        else:
            return jax.vmap(lambda ww, ind: jnp.sum(ww * vector[ind]))(w, indices)
    else:
        # scatter: y[idx[i,k]] += w[i,k] * v[i]
        out = jnp.zeros(n_post, dtype=vector.dtype)
        if homo:
            masked = w * vector[:, None] * jnp.ones((n_pre, n_conn), dtype=vector.dtype)
        else:
            masked = w * vector[:, None]
        return out.at[indices.ravel()].add(masked.ravel())


# ---------------------------------------------------------------------------
# Single config benchmark
# ---------------------------------------------------------------------------

def run_config(
    n_pre: int,
    n_post: int,
    n_conn: int,
    label: str,
    transpose: bool,
    homo: bool,
    n_warmup: int,
    n_runs: int,
    backends: List[str],
):
    dtype = jnp.float32
    shape = (n_pre, n_post)

    # Build arrays on GPU
    rng = np.random.default_rng(42)
    indices_np = rng.integers(0, n_post, (n_pre, n_conn), dtype=np.int32)
    indices = jax.device_put(jnp.asarray(indices_np), jax.devices("gpu")[0])

    if homo:
        weights = jax.device_put(jnp.ones(1, dtype=dtype), jax.devices("gpu")[0])
    else:
        weights = jax.device_put(
            jnp.asarray(rng.standard_normal((n_pre, n_conn)).astype(np.float32)),
            jax.devices("gpu")[0],
        )

    v_size = n_post if not transpose else n_pre
    vector = jax.device_put(
        jnp.asarray(rng.standard_normal(v_size).astype(np.float32)),
        jax.devices("gpu")[0],
    )

    mode = "T" if transpose else "NT"
    weight_kind = "homo" if homo else "hetero"
    header = f"  [{mode},{weight_kind}] {label}  n_conn={n_conn}"

    # Reference output for correctness check
    ref_fn = jax.jit(
        lambda w, idx, v: _jax_reference(w, idx, v, shape=shape, transpose=transpose)
    )
    ref = ref_fn(weights, indices, vector)
    jax.block_until_ready(ref)

    results = {}

    # JAX reference
    try:
        stats = _time_fn(ref_fn, (weights, indices, vector), n_warmup, n_runs)
        results["jax-ref"] = {"stats": stats, "ok": True, "diff": 0.0}
    except Exception as e:
        results["jax-ref"] = {"error": str(e)}

    # Backend kernels
    for backend in backends:
        try:
            fn = _make_jit(weights, indices, vector, shape, transpose, backend)
            out = fn(weights, indices, vector)
            jax.block_until_ready(out)
            diff = float(jnp.max(jnp.abs(out - ref)))
            ok = diff < 1e-4
            stats = _time_fn(fn, (weights, indices, vector), n_warmup, n_runs)
            results[backend] = {"stats": stats, "ok": ok, "diff": diff}
        except Exception as e:
            import traceback
            results[backend] = {"error": str(e)}

    return header, results


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_results(header: str, results: Dict):
    print(header)
    col_w = 14
    all_keys = list(results.keys())
    print(f"    {'Backend':<12}  {'Mean µs':>9}  {'Median µs':>9}  {'Std µs':>8}  {'Status'}")
    print(f"    {'-'*12}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*14}")

    best_t, best_k = float("inf"), None
    for k, v in results.items():
        if "error" in v:
            print(f"    {k:<12}  {'ERROR':>9}  {'':>9}  {'':>8}  {v['error']}")
        else:
            s = v["stats"]
            status = "OK" if v["ok"] else f"FAIL(diff={v['diff']:.1e})"
            print(f"    {k:<12}  {s['mean']:>9.2f}  {s['median']:>9.2f}  {s['std']:>8.2f}  {status}")
            if v["ok"] and s["mean"] < best_t:
                best_t, best_k = s["mean"], k

    if best_k:
        print(f"    → fastest: {best_k} ({best_t:.2f} µs)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="fcnmv backend benchmark")
    parser.add_argument("--n_warmup", type=int, default=10)
    parser.add_argument("--n_runs",   type=int, default=100)
    parser.add_argument("--mode",     choices=["gather", "scatter", "both"], default="both")
    parser.add_argument("--weights",  choices=["homo", "hetero", "both"],    default="both")
    args = parser.parse_args()

    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        print("ERROR: No GPU device found.")
        return

    print("=" * 70)
    print(f"fcnmv benchmark  —  GPU: {gpu}")
    print(f"warmup={args.n_warmup}  runs={args.n_runs}")
    print("=" * 70)
    print()

    transposes = []
    if args.mode in ("gather", "both"):
        transposes.append((False, "Gather (transpose=False)"))
    if args.mode in ("scatter", "both"):
        transposes.append((True,  "Scatter (transpose=True)"))

    weight_modes = []
    if args.weights in ("homo", "both"):
        weight_modes.append(True)
    if args.weights in ("hetero", "both"):
        weight_modes.append(False)

    for transpose, mode_label in transposes:
        print(f"{'─'*70}")
        print(f"  {mode_label}")
        print(f"{'─'*70}")
        for homo in weight_modes:
            for n_pre, n_post, n_conn, label in CONFIGS:
                header, results = run_config(
                    n_pre, n_post, n_conn, label,
                    transpose=transpose,
                    homo=homo,
                    n_warmup=args.n_warmup,
                    n_runs=args.n_runs,
                    backends=BACKENDS,
                )
                _print_results(header, results)

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
