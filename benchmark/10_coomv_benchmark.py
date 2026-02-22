# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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
Benchmark COO-matvec against JAX sparse COO (cuSPARSE on GPU).

This benchmark covers:
- transpose in {False, True}
- homogeneous (weights size == 1) and heterogeneous (weights size == nnz)
- index distributions: uniform, hotspot-row, hotspot-col
- duplicate COO indices (generated with replacement)


### How to run on GPU

> python benchmark/10_coomv_benchmark.py --platform gpu --n-warmup 20 --n-runs 80

Quick run:

> python benchmark/10_coomv_benchmark.py --platform gpu --n-warmup 5 --n-runs 20 --distributions uniform hotspot_row


"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import COO as JaxCOO

import brainevent

platform = jax.default_backend()


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
    dist: str
    transpose: bool
    homo_weight: bool
    sorted_output: bool
    warp_ms: float
    cusparse_ms: float
    warp_gbs: float
    cusparse_gbs: float
    warp_over_cusparse_time: float
    warp_over_cusparse_gbs: float
    allclose: bool
    max_abs_diff: float


DEFAULT_CASES = [
    BenchmarkCase("sq_small", 4096, 4096, 0.010),
    BenchmarkCase("sq_med", 16384, 16384, 0.001),
    BenchmarkCase("rect_tall", 32768, 8192, 0.002),
    BenchmarkCase("rect_wide", 8192, 32768, 0.002),
]

DEFAULT_DISTRIBUTIONS = ("uniform", "hotspot_row", "hotspot_col")


def _build_indices(
    *,
    n_rows: int,
    n_cols: int,
    nnz: int,
    dist: str,
    rng: np.random.Generator,
    hotspot_frac: float,
    hotspot_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    if dist == "uniform":
        row = rng.integers(0, n_rows, size=nnz, dtype=np.int32)
        col = rng.integers(0, n_cols, size=nnz, dtype=np.int32)
        return row, col

    if dist == "hotspot_row":
        n_hot = max(1, int(n_rows * hotspot_frac))
        hot_rows = rng.choice(n_rows, size=n_hot, replace=False)
        n_hot_nnz = int(nnz * hotspot_mass)
        hot_sel = hot_rows[rng.integers(0, n_hot, size=n_hot_nnz)]
        cold_sel = rng.integers(0, n_rows, size=nnz - n_hot_nnz, dtype=np.int32)
        row = np.concatenate([hot_sel.astype(np.int32), cold_sel], axis=0)
        col = rng.integers(0, n_cols, size=nnz, dtype=np.int32)
        return row, col

    if dist == "hotspot_col":
        n_hot = max(1, int(n_cols * hotspot_frac))
        hot_cols = rng.choice(n_cols, size=n_hot, replace=False)
        n_hot_nnz = int(nnz * hotspot_mass)
        hot_sel = hot_cols[rng.integers(0, n_hot, size=n_hot_nnz)]
        cold_sel = rng.integers(0, n_cols, size=nnz - n_hot_nnz, dtype=np.int32)
        col = np.concatenate([hot_sel.astype(np.int32), cold_sel], axis=0)
        row = rng.integers(0, n_rows, size=nnz, dtype=np.int32)
        return row, col

    raise ValueError(f"Unsupported dist='{dist}'. Expected one of {DEFAULT_DISTRIBUTIONS}.")


def _build_weights(
    *,
    nnz: int,
    homo_weight: bool,
    rng: np.random.Generator,
    dtype: np.dtype,
) -> np.ndarray:
    if homo_weight:
        return np.asarray([1.0], dtype=dtype)
    return rng.standard_normal(nnz).astype(dtype)


def _timed_ms(
    fn,
    *args,
    n_warmup: int,
    n_runs: int,
) -> tuple[float, jax.Array]:
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


def _effective_bytes(
    *,
    nnz: int,
    out_size: int,
    value_itemsize: int,
    index_itemsize: int,
    homo_weight: bool,
) -> int:
    # Approximate COO SpMV traffic:
    # row + col + data + vector gather + output writeback
    weight_bytes = value_itemsize if homo_weight else nnz * value_itemsize
    index_bytes = nnz * (2 * index_itemsize)
    vector_bytes = nnz * value_itemsize
    out_bytes = out_size * value_itemsize
    return weight_bytes + index_bytes + vector_bytes + out_bytes


def _geometric_mean(vals: Iterable[float]) -> float:
    vals = [v for v in vals if v > 0.0 and np.isfinite(v)]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def run_case(
    *,
    case: BenchmarkCase,
    dist: str,
    transpose: bool,
    homo_weight: bool,
    sorted_output: bool,
    seed: int,
    n_warmup: int,
    n_runs: int,
    dtype: np.dtype,
    index_dtype: np.dtype,
    hotspot_frac: float,
    hotspot_mass: float,
    atol: float,
    rtol: float,
) -> BenchmarkRow:
    rng = np.random.default_rng(seed)
    nnz = case.nnz
    row_np, col_np = _build_indices(
        n_rows=case.n_rows,
        n_cols=case.n_cols,
        nnz=nnz,
        dist=dist,
        rng=rng,
        hotspot_frac=hotspot_frac,
        hotspot_mass=hotspot_mass,
    )
    weights_np = _build_weights(nnz=nnz, homo_weight=homo_weight, rng=rng, dtype=dtype)
    vec_size = case.n_rows if transpose else case.n_cols
    vector_np = rng.standard_normal(vec_size).astype(dtype)

    if sorted_output:
        if transpose:
            order = np.lexsort((row_np, col_np))
        else:
            order = np.lexsort((col_np, row_np))
        row_np = row_np[order]
        col_np = col_np[order]
        if not homo_weight:
            weights_np = weights_np[order]

    row = jnp.asarray(row_np, dtype=index_dtype)
    col = jnp.asarray(col_np, dtype=index_dtype)
    weights = jnp.asarray(weights_np, dtype=dtype)
    vector = jnp.asarray(vector_np, dtype=dtype)

    weights_dense = (
        jnp.full((nnz,), weights[0], dtype=vector.dtype)
        if homo_weight else
        weights
    )
    jax_coo = JaxCOO((weights_dense, row, col), shape=(case.n_rows, case.n_cols))
    jax_coo = jax_coo._sort_indices()

    @jax.jit
    def warp_fn(w, r, c, v):
        return brainevent.coomv(
            w,
            r,
            c,
            v,
            shape=(case.n_rows, case.n_cols),
            transpose=transpose,
        )

    @jax.jit
    def cusparse_fn(v):
        return (jax_coo.T @ v) if transpose else (jax_coo @ v)

    warp_ms, warp_out = _timed_ms(warp_fn, weights, row, col, vector, n_warmup=n_warmup, n_runs=n_runs)
    cusparse_ms, cusparse_out = _timed_ms(cusparse_fn, vector, n_warmup=n_warmup, n_runs=n_runs)

    allclose = bool(jnp.allclose(warp_out, cusparse_out, atol=atol, rtol=rtol))
    max_abs_diff = float(jnp.max(jnp.abs(warp_out - cusparse_out)))

    out_size = case.n_cols if transpose else case.n_rows
    bytes_moved = _effective_bytes(
        nnz=nnz,
        out_size=out_size,
        value_itemsize=np.dtype(dtype).itemsize,
        index_itemsize=np.dtype(index_dtype).itemsize,
        homo_weight=homo_weight,
    )
    warp_gbs = (bytes_moved / (warp_ms / 1000.0)) / 1e9
    cusparse_gbs = (bytes_moved / (cusparse_ms / 1000.0)) / 1e9

    return BenchmarkRow(
        case=case.name,
        shape=f"{case.n_rows}x{case.n_cols}",
        nnz=nnz,
        density=case.density,
        dist=dist,
        transpose=transpose,
        homo_weight=homo_weight,
        sorted_output=sorted_output,
        warp_ms=warp_ms,
        cusparse_ms=cusparse_ms,
        warp_gbs=warp_gbs,
        cusparse_gbs=cusparse_gbs,
        warp_over_cusparse_time=warp_ms / cusparse_ms,
        warp_over_cusparse_gbs=warp_gbs / cusparse_gbs,
        allclose=allclose,
        max_abs_diff=max_abs_diff,
    )


def _print_header(args):
    print("COO SpMV Benchmark: Warp vs JAX COO (cuSPARSE on GPU)")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"brainstate platform target: {args.platform}")
    print(f"Warmup={args.n_warmup}, Runs={args.n_runs}")
    print(
        f"Tolerance: allclose(atol={args.atol}, rtol={args.rtol}), "
        f"hotspot_frac={args.hotspot_frac}, hotspot_mass={args.hotspot_mass}"
    )
    if jax.default_backend() != "gpu":
        print(
            "WARNING: JAX is not using GPU. The JAX COO baseline will not run on cuSPARSE."
        )
    print("-" * 140)
    print(
        f"{'case':<10} {'shape':<14} {'nnz':>10} {'dist':<12} {'T':<2} {'W':<3} {'S':<2} "
        f"{'warp(ms)':>10} {'cusp(ms)':>10} {'warp GB/s':>10} {'cusp GB/s':>10} "
        f"{'warp/cusp t':>11} {'warp/cusp bw':>12} {'allclose':>9} {'max|diff|':>12}"
    )
    print("-" * 140)


def _print_row(r: BenchmarkRow):
    print(
        f"{r.case:<10} {r.shape:<14} {r.nnz:>10,} {r.dist:<12} "
        f"{('T' if r.transpose else 'N'):<2} "
        f"{('H' if r.homo_weight else 'X'):<3} "
        f"{('S' if r.sorted_output else 'U'):<2} "
        f"{r.warp_ms:>10.3f} {r.cusparse_ms:>10.3f} "
        f"{r.warp_gbs:>10.3f} {r.cusparse_gbs:>10.3f} "
        f"{r.warp_over_cusparse_time:>11.3f} {r.warp_over_cusparse_gbs:>12.3f} "
        f"{str(r.allclose):>9} {r.max_abs_diff:>12.3e}"
    )


def _save_csv(rows: List[BenchmarkRow], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "case,shape,nnz,density,dist,transpose,homo_weight,sorted_output,warp_ms,cusparse_ms,"
            "warp_gbs,cusparse_gbs,warp_over_cusparse_time,warp_over_cusparse_gbs,"
            "allclose,max_abs_diff\n"
        )
        for r in rows:
            f.write(
                f"{r.case},{r.shape},{r.nnz},{r.density:.8f},{r.dist},"
                f"{int(r.transpose)},{int(r.homo_weight)},{int(r.sorted_output)},"
                f"{r.warp_ms:.8f},{r.cusparse_ms:.8f},"
                f"{r.warp_gbs:.8f},{r.cusparse_gbs:.8f},"
                f"{r.warp_over_cusparse_time:.8f},{r.warp_over_cusparse_gbs:.8f},"
                f"{int(r.allclose)},{r.max_abs_diff:.8e}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Warp COO-matvec against JAX COO baseline.")
    parser.add_argument("--platform", type=str, default=platform, choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--index-dtype", type=str, default="int32", choices=["int32", "int64"])
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--hotspot-frac", type=float, default=0.05)
    parser.add_argument("--hotspot-mass", type=float, default=0.80)
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=list(DEFAULT_DISTRIBUTIONS),
        choices=list(DEFAULT_DISTRIBUTIONS),
    )
    parser.add_argument(
        "--sorted-output",
        action="store_true",
        help="Sort COO entries by output dim (row for NT, col for T) to unlock atomics-free Warp kernel.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmark/results/24_coomv_benchmark.csv",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any benchmark row fails allclose.",
    )
    parser.add_argument(
        "--allow-non-gpu",
        action="store_true",
        help="Allow running when JAX backend is not GPU (debug only; not cuSPARSE).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    brainstate.environ.set(platform=args.platform)

    if args.platform == "gpu":
        gpu_backends = brainevent.coomv_p.available_backends("gpu")
        if "warp" not in gpu_backends:
            raise SystemExit(f"Warp backend is not registered for GPU. Available backends: {gpu_backends}")
        brainevent.coomv_p.set_default("gpu", "warp", persist=False)

    value_dtype = np.float32 if args.dtype == "float32" else np.float64
    index_dtype = np.int32 if args.index_dtype == "int32" else np.int64

    rows: List[BenchmarkRow] = []
    failures = []

    _print_header(args)
    for i_case, case in enumerate(DEFAULT_CASES):
        for i_dist, dist in enumerate(args.distributions):
            for i_t, transpose in enumerate((False, True)):
                for i_w, homo_weight in enumerate((True, False)):
                    case_seed = (
                        args.seed
                        + i_case * 100000
                        + i_dist * 10000
                        + i_t * 1000
                        + i_w * 100
                    )
                    try:
                        row = run_case(
                            case=case,
                            dist=dist,
                            transpose=transpose,
                            homo_weight=homo_weight,
                            sorted_output=args.sorted_output,
                            seed=case_seed,
                            n_warmup=args.n_warmup,
                            n_runs=args.n_runs,
                            dtype=value_dtype,
                            index_dtype=index_dtype,
                            hotspot_frac=args.hotspot_frac,
                            hotspot_mass=args.hotspot_mass,
                            atol=args.atol,
                            rtol=args.rtol,
                        )
                        rows.append(row)
                        _print_row(row)
                        if not row.allclose:
                            failures.append(
                                (
                                    case.name,
                                    dist,
                                    transpose,
                                    homo_weight,
                                    row.max_abs_diff,
                                )
                            )
                    except Exception as e:
                        tag = f"{case.name}/{dist}/T={transpose}/H={homo_weight}/S={args.sorted_output}"
                        print(f"{tag:<70} ERROR: {type(e).__name__}: {e}")
                        failures.append((case.name, dist, transpose, homo_weight, str(e)))

    print("-" * 140)
    if rows:
        time_ratios = [r.warp_over_cusparse_time for r in rows]
        bw_ratios = [r.warp_over_cusparse_gbs for r in rows]
        print(
            f"Geomean warp/cusparse time ratio (lower is better): "
            f"{_geometric_mean(time_ratios):.4f}"
        )
        print(
            f"Geomean warp/cusparse bandwidth ratio (higher is better): "
            f"{_geometric_mean(bw_ratios):.4f}"
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
