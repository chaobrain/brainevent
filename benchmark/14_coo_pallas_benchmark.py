"""Benchmark COO Pallas kernels (SpMV + SpMM) against JAX sparse COO baseline.

Usage:
  python benchmark/14_coo_pallas_benchmark.py --n-warmup 10 --n-runs 30
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import COO as JaxCOO

sys.path.append(str(Path(__file__).resolve().parents[1]))

import brainevent


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    n: int
    density: float

    @property
    def nnz(self) -> int:
        return max(1, int(self.m * self.n * self.density))


@dataclass
class Row:
    case: str
    shape: str
    nnz: int
    dist: str
    transpose: bool
    homo_weight: bool
    coomv_pallas_ms: float
    coomv_jaxcoo_ms: float
    coomm_pallas_ms: float
    coomm_jaxcoo_ms: float
    coomv_allclose: bool
    coomm_allclose: bool
    coomv_max_abs: float
    coomm_max_abs: float


CASES = (
    Case('sq_2k', 2048, 2048, 0.002),
    Case('sq_4k', 4096, 4096, 0.001),
    Case('rect_8k4k', 8192, 4096, 0.0015),
)


def _timed_ms(fn, *args, n_warmup: int, n_runs: int):
    out = None
    for _ in range(n_warmup):
        out = fn(*args)
    jax.block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = fn(*args)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / n_runs, out


def _indices(m: int, n: int, nnz: int, dist: str, rng: np.random.Generator):
    if dist == 'uniform':
        row = rng.integers(0, m, size=nnz, dtype=np.int32)
        col = rng.integers(0, n, size=nnz, dtype=np.int32)
        return row, col

    if dist == 'hotspot_row':
        hot = max(1, m // 32)
        hot_rows = rng.integers(0, hot, size=nnz, dtype=np.int32)
        cold_rows = rng.integers(0, m, size=nnz, dtype=np.int32)
        choose_hot = rng.random(nnz) < 0.85
        row = np.where(choose_hot, hot_rows, cold_rows).astype(np.int32)
        col = rng.integers(0, n, size=nnz, dtype=np.int32)
        return row, col

    raise ValueError(f'Unsupported distribution: {dist}')


def run_case(case: Case, dist: str, transpose: bool, homo_weight: bool, n_rhs: int,
             n_warmup: int, n_runs: int, seed: int, atol: float, rtol: float) -> Row:
    rng = np.random.default_rng(seed)
    nnz = case.nnz
    row_np, col_np = _indices(case.m, case.n, nnz, dist, rng)

    if homo_weight:
        w_np = np.asarray([rng.standard_normal()], dtype=np.float32)
    else:
        w_np = rng.standard_normal(nnz).astype(np.float32)

    x_len = case.m if transpose else case.n
    b_rows = case.m if transpose else case.n
    x_np = rng.standard_normal(x_len).astype(np.float32)
    b_np = rng.standard_normal((b_rows, n_rhs)).astype(np.float32)

    row = jnp.asarray(row_np)
    col = jnp.asarray(col_np)
    w = jnp.asarray(w_np)
    x = jnp.asarray(x_np)
    B = jnp.asarray(b_np)

    dense_w = jnp.full((nnz,), w[0], dtype=jnp.float32) if homo_weight else w
    jax_coo = JaxCOO((dense_w, row, col), shape=(case.m, case.n))._sort_indices()

    @jax.jit
    def pallas_mv(weights, rr, cc, vec):
        return brainevent.coomv(weights, rr, cc, vec, shape=(case.m, case.n), transpose=transpose, backend='pallas')

    @jax.jit
    def pallas_mm(weights, rr, cc, mat):
        return brainevent.coomm(weights, rr, cc, mat, shape=(case.m, case.n), transpose=transpose, backend='pallas')

    @jax.jit
    def jax_mv(vec):
        return (jax_coo.T @ vec) if transpose else (jax_coo @ vec)

    @jax.jit
    def jax_mm(mat):
        return (jax_coo.T @ mat) if transpose else (jax_coo @ mat)

    coomv_pallas_ms, out_mv = _timed_ms(pallas_mv, w, row, col, x, n_warmup=n_warmup, n_runs=n_runs)
    coomv_jax_ms, ref_mv = _timed_ms(jax_mv, x, n_warmup=n_warmup, n_runs=n_runs)

    coomm_pallas_ms, out_mm = _timed_ms(pallas_mm, w, row, col, B, n_warmup=n_warmup, n_runs=max(1, n_runs // 2))
    coomm_jax_ms, ref_mm = _timed_ms(jax_mm, B, n_warmup=n_warmup, n_runs=max(1, n_runs // 2))

    return Row(
        case=case.name,
        shape=f'{case.m}x{case.n}',
        nnz=nnz,
        dist=dist,
        transpose=transpose,
        homo_weight=homo_weight,
        coomv_pallas_ms=coomv_pallas_ms,
        coomv_jaxcoo_ms=coomv_jax_ms,
        coomm_pallas_ms=coomm_pallas_ms,
        coomm_jaxcoo_ms=coomm_jax_ms,
        coomv_allclose=bool(jnp.allclose(out_mv, ref_mv, atol=atol, rtol=rtol)),
        coomm_allclose=bool(jnp.allclose(out_mm, ref_mm, atol=atol, rtol=rtol)),
        coomv_max_abs=float(jnp.max(jnp.abs(out_mv - ref_mv))),
        coomm_max_abs=float(jnp.max(jnp.abs(out_mm - ref_mm))),
    )


def parse_args():
    p = argparse.ArgumentParser(description='Benchmark Pallas COO kernels (SpMV + SpMM).')
    p.add_argument('--n-warmup', type=int, default=10)
    p.add_argument('--n-runs', type=int, default=30)
    p.add_argument('--n-rhs', type=int, default=64)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--atol', type=float, default=1e-4)
    p.add_argument('--rtol', type=float, default=1e-4)
    p.add_argument('--output-csv', type=str, default='benchmark/results/14_coo_pallas_benchmark.csv')
    return p.parse_args()


def main():
    args = parse_args()
    if jax.default_backend() != 'gpu':
        raise SystemExit('This benchmark requires a GPU backend.')

    print('COO Pallas Benchmark (SpMV + SpMM)')
    print(f'JAX backend: {jax.default_backend()}')
    print(f'Warmup={args.n_warmup}, Runs={args.n_runs}, RHS={args.n_rhs}')
    print('-' * 160)
    print(
        f"{'case':<10} {'shape':<12} {'nnz':>10} {'dist':<11} {'T':<2} {'W':<2} "
        f"{'mv p(ms)':>10} {'mv jax(ms)':>10} {'mm p(ms)':>10} {'mm jax(ms)':>10} "
        f"{'mv ok':>6} {'mm ok':>6} {'mv max|d|':>11} {'mm max|d|':>11}"
    )
    print('-' * 160)

    rows = []
    for i_case, case in enumerate(CASES):
        for i_dist, dist in enumerate(('uniform', 'hotspot_row')):
            for i_t, transpose in enumerate((False, True)):
                for i_w, homo in enumerate((False, True)):
                    seed = args.seed + i_case * 1000 + i_dist * 100 + i_t * 10 + i_w
                    row = run_case(
                        case=case,
                        dist=dist,
                        transpose=transpose,
                        homo_weight=homo,
                        n_rhs=args.n_rhs,
                        n_warmup=args.n_warmup,
                        n_runs=args.n_runs,
                        seed=seed,
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                    rows.append(row)
                    print(
                        f"{row.case:<10} {row.shape:<12} {row.nnz:>10,} {row.dist:<11} "
                        f"{('T' if row.transpose else 'N'):<2} {('H' if row.homo_weight else 'X'):<2} "
                        f"{row.coomv_pallas_ms:>10.3f} {row.coomv_jaxcoo_ms:>10.3f} "
                        f"{row.coomm_pallas_ms:>10.3f} {row.coomm_jaxcoo_ms:>10.3f} "
                        f"{str(row.coomv_allclose):>6} {str(row.coomm_allclose):>6} "
                        f"{row.coomv_max_abs:>11.3e} {row.coomm_max_abs:>11.3e}"
                    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'case', 'shape', 'nnz', 'dist', 'transpose', 'homo_weight',
            'coomv_pallas_ms', 'coomv_jaxcoo_ms', 'coomm_pallas_ms', 'coomm_jaxcoo_ms',
            'coomv_allclose', 'coomm_allclose', 'coomv_max_abs', 'coomm_max_abs',
        ])
        for row in rows:
            writer.writerow([
                row.case, row.shape, row.nnz, row.dist, int(row.transpose), int(row.homo_weight),
                row.coomv_pallas_ms, row.coomv_jaxcoo_ms, row.coomm_pallas_ms, row.coomm_jaxcoo_ms,
                int(row.coomv_allclose), int(row.coomm_allclose), row.coomv_max_abs, row.coomm_max_abs,
            ])
    print('-' * 160)
    print(f'Saved CSV: {out_path}')


if __name__ == '__main__':
    main()
