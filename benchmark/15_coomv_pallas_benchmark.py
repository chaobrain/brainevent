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

# -*- coding: utf-8 -*-


"""
Benchmark Pallas GPU COO-matvec against JAX sparse COO (cuSPARSE on GPU).

This benchmark covers:
- transpose in {False, True}
- homogeneous (weights size == 1) and heterogeneous (weights size == nnz)
- binary events: bool and float
- distributions: uniform


### How to run on GPU

> python benchmark/15_coomv_pallas_benchmark.py --platform gpu --n-warmup 20 --n-runs 80

Quick run:

> python benchmark/15_coomv_pallas_benchmark.py --platform gpu --n-warmup 5 --n-runs 20

"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[1]))

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import COO as JaxCOO

import brainevent


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    n_rows: int
    n_cols: int
    density: float

    @property
    def nnz(self) -> int:
        return max(1, int(self.n_rows * self.n_cols * self.density))


@dataclass
class BenchmarkRow:
    case: str
    shape: str
    nnz: int
    density: float
    transpose: bool
    homo_weight: bool
    bool_event: bool
    pallas_ms: float
    cusparse_ms: float
    pallas_gbs: float
    cusparse_gbs: float
    pallas_over_cusparse_time: float
    allclose: bool
    max_abs_diff: float


DEFAULT_CASES = [
    BenchmarkCase("sq_small", 4096, 4096, 0.010),
    BenchmarkCase("sq_med", 16384, 16384, 0.001),
    BenchmarkCase("rect_tall", 32768, 8192, 0.002),
    BenchmarkCase("rect_wide", 8192, 32768, 0.002),
]


def _timed_ms(fn, *args, n_warmup, n_runs):
    out = None
    for _ in range(n_warmup):
        out = fn(*args)
    jax.block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = fn(*args)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    return ((t1 - t0) * 1000.0 / n_runs), out


def _effective_bytes(*, nnz, out_size, value_itemsize, index_itemsize, homo_weight):
    weight_bytes = value_itemsize if homo_weight else nnz * value_itemsize
    index_bytes = nnz * (2 * index_itemsize)
    vector_bytes = nnz * value_itemsize
    out_bytes = out_size * value_itemsize
    return weight_bytes + index_bytes + vector_bytes + out_bytes


def _geometric_mean(vals):
    vals = [v for v in vals if v > 0.0 and np.isfinite(v)]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def run_case(
    *,
    case,
    transpose,
    homo_weight,
    bool_event,
    seed,
    n_warmup,
    n_runs,
    dtype,
    index_dtype,
    atol,
    rtol,
):
    rng = np.random.default_rng(seed)
    nnz = case.nnz
    row_np = rng.integers(0, case.n_rows, size=nnz, dtype=index_dtype)
    col_np = rng.integers(0, case.n_cols, size=nnz, dtype=index_dtype)

    if homo_weight:
        weights_np = np.asarray([1.0], dtype=dtype)
    else:
        weights_np = rng.standard_normal(nnz).astype(dtype)

    vec_size = case.n_rows if transpose else case.n_cols
    if bool_event:
        vector_np = (rng.random(vec_size) > 0.5).astype(np.bool_)
    else:
        vector_np = rng.standard_normal(vec_size).astype(dtype)

    row = jnp.asarray(row_np)
    col = jnp.asarray(col_np)
    weights = jnp.asarray(weights_np)
    vector = jnp.asarray(vector_np)

    # Build cuSPARSE reference
    weights_dense = (
        jnp.full((nnz,), weights[0], dtype=jnp.float32)
        if homo_weight else
        jnp.asarray(weights, dtype=jnp.float32)
    )
    vector_float = jnp.asarray(vector, dtype=jnp.float32) if bool_event else vector
    jax_coo = JaxCOO((weights_dense, row, col), shape=(case.n_rows, case.n_cols))
    jax_coo = jax_coo._sort_indices()

    if bool_event:
        @jax.jit
        def pallas_fn(w, r, c, v):
            return brainevent.binary_coomv(
                w, r, c, v,
                shape=(case.n_rows, case.n_cols),
                transpose=transpose,
                backend="pallas",
            )
    else:
        @jax.jit
        def pallas_fn(w, r, c, v):
            return brainevent.coomv(
                w, r, c, v,
                shape=(case.n_rows, case.n_cols),
                transpose=transpose,
                backend="pallas",
            )

    @jax.jit
    def cusparse_fn(v):
        return (jax_coo.T @ v) if transpose else (jax_coo @ v)

    pallas_ms, pallas_out = _timed_ms(pallas_fn, weights, row, col, vector, n_warmup=n_warmup, n_runs=n_runs)
    cusparse_ms, cusparse_out = _timed_ms(cusparse_fn, vector_float, n_warmup=n_warmup, n_runs=n_runs)

    allclose = bool(jnp.allclose(pallas_out, cusparse_out, atol=atol, rtol=rtol))
    max_abs_diff = float(jnp.max(jnp.abs(jnp.asarray(pallas_out, dtype=jnp.float32) - cusparse_out)))

    out_size = case.n_cols if transpose else case.n_rows
    bytes_moved = _effective_bytes(
        nnz=nnz,
        out_size=out_size,
        value_itemsize=np.dtype(dtype).itemsize,
        index_itemsize=np.dtype(index_dtype).itemsize,
        homo_weight=homo_weight,
    )
    pallas_gbs = (bytes_moved / (pallas_ms / 1000.0)) / 1e9
    cusparse_gbs = (bytes_moved / (cusparse_ms / 1000.0)) / 1e9

    return BenchmarkRow(
        case=case.name,
        shape=f"{case.n_rows}x{case.n_cols}",
        nnz=nnz,
        density=case.density,
        transpose=transpose,
        homo_weight=homo_weight,
        bool_event=bool_event,
        pallas_ms=pallas_ms,
        cusparse_ms=cusparse_ms,
        pallas_gbs=pallas_gbs,
        cusparse_gbs=cusparse_gbs,
        pallas_over_cusparse_time=pallas_ms / cusparse_ms,
        allclose=allclose,
        max_abs_diff=max_abs_diff,
    )


def _print_header(args):
    print("COO SpMV Benchmark: Pallas GPU vs JAX COO (cuSPARSE on GPU)")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"brainstate platform target: {args.platform}")
    print(f"Warmup={args.n_warmup}, Runs={args.n_runs}")
    print(f"Tolerance: allclose(atol={args.atol}, rtol={args.rtol})")
    if jax.default_backend() != "gpu":
        print("WARNING: JAX is not using GPU. The JAX COO baseline will not use cuSPARSE.")
    print("-" * 130)
    print(
        f"{'case':<10} {'shape':<14} {'nnz':>10} {'T':<2} {'W':<3} {'E':<5} "
        f"{'pallas(ms)':>10} {'cusp(ms)':>10} {'pallas GB/s':>12} {'cusp GB/s':>10} "
        f"{'pallas/cusp':>11} {'allclose':>9} {'max|diff|':>12}"
    )
    print("-" * 130)


def _print_row(r):
    print(
        f"{r.case:<10} {r.shape:<14} {r.nnz:>10,} "
        f"{('T' if r.transpose else 'N'):<2} "
        f"{('H' if r.homo_weight else 'X'):<3} "
        f"{('bool' if r.bool_event else 'flt'):<5} "
        f"{r.pallas_ms:>10.3f} {r.cusparse_ms:>10.3f} "
        f"{r.pallas_gbs:>12.3f} {r.cusparse_gbs:>10.3f} "
        f"{r.pallas_over_cusparse_time:>11.3f} "
        f"{str(r.allclose):>9} {r.max_abs_diff:>12.3e}"
    )


def _save_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "case,shape,nnz,density,transpose,homo_weight,bool_event,"
            "pallas_ms,cusparse_ms,pallas_gbs,cusparse_gbs,"
            "pallas_over_cusparse_time,allclose,max_abs_diff\n"
        )
        for r in rows:
            f.write(
                f"{r.case},{r.shape},{r.nnz},{r.density:.8f},"
                f"{int(r.transpose)},{int(r.homo_weight)},{int(r.bool_event)},"
                f"{r.pallas_ms:.8f},{r.cusparse_ms:.8f},"
                f"{r.pallas_gbs:.8f},{r.cusparse_gbs:.8f},"
                f"{r.pallas_over_cusparse_time:.8f},"
                f"{int(r.allclose)},{r.max_abs_diff:.8e}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Pallas GPU COO-matvec against JAX COO baseline.")
    parser.add_argument("--platform", type=str, default="gpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--index-dtype", type=str, default="int32", choices=["int32", "int64"])
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark/results/15_coomv_pallas_benchmark.csv",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any benchmark row fails allclose.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    brainstate.environ.set(platform=args.platform)

    value_dtype = np.float32 if args.dtype == "float32" else np.float64
    index_dtype = np.int32 if args.index_dtype == "int32" else np.int64

    rows: List[BenchmarkRow] = []
    failures = []

    _print_header(args)
    for i_case, case in enumerate(DEFAULT_CASES):
        for i_t, transpose in enumerate((False, True)):
            for i_w, homo_weight in enumerate((True, False)):
                for i_e, bool_event in enumerate((True, False)):
                    case_seed = (
                        args.seed
                        + i_case * 100000
                        + i_t * 10000
                        + i_w * 1000
                        + i_e * 100
                    )
                    try:
                        row = run_case(
                            case=case,
                            transpose=transpose,
                            homo_weight=homo_weight,
                            bool_event=bool_event,
                            seed=case_seed,
                            n_warmup=args.n_warmup,
                            n_runs=args.n_runs,
                            dtype=value_dtype,
                            index_dtype=index_dtype,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        rows.append(row)
                        _print_row(row)
                        if not row.allclose:
                            failures.append((case.name, transpose, homo_weight, bool_event, row.max_abs_diff))
                    except Exception as e:
                        tag = (
                            f"{case.name}/T={transpose}/H={homo_weight}"
                            f"/E={'bool' if bool_event else 'float'}"
                        )
                        print(f"{tag:<60} ERROR: {type(e).__name__}: {e}")
                        failures.append((case.name, transpose, homo_weight, bool_event, str(e)))

    print("-" * 130)
    if rows:
        time_ratios = [r.pallas_over_cusparse_time for r in rows]
        print(
            f"Geomean pallas/cusparse time ratio (lower is better): "
            f"{_geometric_mean(time_ratios):.4f}"
        )
        print(f"Rows benchmarked: {len(rows)}")

    out_path = Path(args.output_csv)
    _save_csv(rows, out_path)
    print(f"Saved CSV: {out_path}")

    if failures:
        print(f"Failures/mismatches: {len(failures)}")
        for fail in failures[:10]:
            print(f"  {fail}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("All benchmark rows passed allclose checks.")


if __name__ == "__main__":
    main()
