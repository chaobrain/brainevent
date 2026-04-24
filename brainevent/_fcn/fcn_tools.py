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

import gc
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Support running this file directly via `python brainevent/_fcn/fcn_tools.py`.
if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from brainevent._event.bitpack_binary import bitpack
from brainevent._event.compact_binary import CompactBinary
from brainevent._test_util import generate_fixed_conn_num_indices
from brainevent._fcn.bitpack_binary import bitpack_binary_fcnmv
from brainevent._fcn.compact_binary import compact_binary_fcnmv

__all__ = [
    'BinaryDumpRecorder',
    'binary_dump_recorder',
    'run_boundary_tests_generic',
    'run_compact_tol_tests',
    'run_bitpack_tol_tests',
    'main',
]


N_CONN = 4

DEFAULT_LIMIT_GB = 8.0
DEFAULT_SCALE_MAX = 2000
DEFAULT_BASE_NEURONS = 4000
DEFAULT_DATA_SIZE = 4
DEFAULT_MAX_CONN = 4000
DEFAULT_POINTS_PER_CURVE = 3
DEFAULT_INCLUDE_DENSE_REF = True
DEFAULT_TEST_HOMO = True
DEFAULT_TEST_HETERO = True
DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = 1e-3
DEFAULT_REL_EPS = 1e-12
DEFAULT_COMPACT_TRANSPOSE = True
DEFAULT_BITPACK_TRANSPOSE = False

DEFAULT_DUMP_DIR = Path.cwd() / 'binary_fcnmm_row_sparse_dump'


def _mk_indices(shape, n_conn=N_CONN):
    m, n = shape
    return generate_fixed_conn_num_indices(m, n, min(n_conn, n))


def _mk_homo_w(dtype=jnp.float32):
    return jnp.array([1.5], dtype=dtype)


def _mk_hetero_w(indices, dtype=jnp.float32, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(indices.shape).astype(np.float32), dtype=dtype)


def _mk_spikes(size, p=0.5, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random(size) < p, dtype=jnp.float32)


def _mk_matrix(rows, cols, p=0.5, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.random((rows, cols)) < p, dtype=jnp.float32)


def _dense_mat(weights, indices, shape):
    n_pre, _ = shape
    rows = jnp.repeat(jnp.arange(n_pre, dtype=indices.dtype), indices.shape[1])
    cols = indices.reshape(-1)
    weights = jnp.asarray(weights)
    if weights.size == 1:
        values = jnp.full(indices.size, weights.reshape(())[()], dtype=weights.dtype)
    else:
        values = weights.reshape(-1)
    return jnp.zeros(shape, dtype=weights.dtype).at[rows, cols].add(values)


def _ref_mv(weights, indices, spikes, shape, transpose):
    dense = _dense_mat(weights, indices, shape)
    spike_values = jnp.asarray(spikes > 0, dtype=dense.dtype)
    if transpose:
        return spike_values @ dense
    return dense @ spike_values


def _ref_mm(weights, indices, matrix, shape, transpose):
    dense = _dense_mat(weights, indices, shape)
    matrix_values = jnp.asarray(matrix > 0, dtype=dense.dtype)
    if transpose:
        return dense.T @ matrix_values
    return dense @ matrix_values


def compute_error_metrics(y, y_ref, rel_eps: float = DEFAULT_REL_EPS) -> Dict[str, float]:
    diff = jnp.abs(y - y_ref)
    denom = jnp.maximum(jnp.abs(y_ref), rel_eps)
    rel = diff / denom

    flat_idx = int(jnp.argmax(diff))
    max_err_index = tuple(int(x) for x in jnp.unravel_index(flat_idx, diff.shape))

    return {
        'max_abs_err': float(jnp.max(diff)),
        'mean_abs_err': float(jnp.mean(diff)),
        'max_rel_err': float(jnp.max(rel)),
        'mean_rel_err': float(jnp.mean(rel)),
        'max_err_index': max_err_index,
        'y_at_max_err': float(jnp.ravel(y)[flat_idx]),
        'y_ref_at_max_err': float(jnp.ravel(y_ref)[flat_idx]),
    }


def estimate_empirical_bytes(
    m: int,
    conn: int,
    homo_or_not: bool,
    data_size: int = DEFAULT_DATA_SIZE,
    include_dense_ref: bool = True,
) -> int:
    times = 1 if homo_or_not else 2
    sparse_bytes = m * conn * data_size * times
    dense_bytes = m * m * data_size if include_dense_ref else 0
    return sparse_bytes + dense_bytes


def budget_sequence_gb(limit_GB: float) -> List[float]:
    if limit_GB <= 0:
        raise ValueError(f'limit_GB must be > 0, got {limit_GB}')

    budgets = [float(k) for k in range(1, int(math.floor(limit_GB)) + 1)]

    if not budgets:
        budgets = [float(limit_GB)]
    elif abs(budgets[-1] - limit_GB) > 1e-12:
        budgets.append(float(limit_GB))

    return budgets


def select_evenly_spaced_points(points: List[Dict[str, Any]], num_points: int = 3) -> List[Dict[str, Any]]:
    if len(points) <= num_points:
        return points

    idxs = np.linspace(0, len(points) - 1, num_points)
    idxs = [int(round(x)) for x in idxs]

    selected = []
    seen = set()
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            selected.append(points[idx])

    return selected


def generate_boundary_curve_points(
    target_GB: float,
    homo_or_not: bool = True,
    scale_max: int = DEFAULT_SCALE_MAX,
    _N: int = DEFAULT_BASE_NEURONS,
    data_size: int = DEFAULT_DATA_SIZE,
    max_conn: int = DEFAULT_MAX_CONN,
    points_per_curve: int = DEFAULT_POINTS_PER_CURVE,
    include_dense_ref: bool = True,
) -> List[Dict[str, Any]]:
    target_bytes = target_GB * (1024 ** 3)
    times = 1 if homo_or_not else 2

    curve_points = []

    for scale in range(1, scale_max + 1):
        m = scale * _N

        dense_bytes = (m * m * data_size) if include_dense_ref else 0
        remain_bytes = target_bytes - dense_bytes
        if remain_bytes <= 0:
            break

        per_conn_row_bytes = m * data_size * times
        if per_conn_row_bytes <= 0:
            continue

        conn = int(remain_bytes // per_conn_row_bytes)
        conn = min(conn, max_conn, m)

        if conn < 1:
            continue

        used_bytes = estimate_empirical_bytes(
            m=m,
            conn=conn,
            homo_or_not=homo_or_not,
            data_size=data_size,
            include_dense_ref=include_dense_ref,
        )

        curve_points.append({
            'target_GB': float(target_GB),
            'target_bytes': int(target_bytes),
            'scale': int(scale),
            'm': int(m),
            'conn': int(conn),
            'homo_or_not': bool(homo_or_not),
            'used_bytes': int(used_bytes),
            'used_GB': used_bytes / (1024 ** 3),
            'utilization': float(used_bytes / target_bytes),
        })

    return select_evenly_spaced_points(curve_points, num_points=points_per_curve)


def generate_all_boundary_cases(
    limit_GB: float,
    homo_or_not: bool = True,
    scale_max: int = DEFAULT_SCALE_MAX,
    _N: int = DEFAULT_BASE_NEURONS,
    data_size: int = DEFAULT_DATA_SIZE,
    max_conn: int = DEFAULT_MAX_CONN,
    points_per_curve: int = DEFAULT_POINTS_PER_CURVE,
    include_dense_ref: bool = True,
) -> List[Dict[str, Any]]:
    all_cases = []
    for budget in budget_sequence_gb(limit_GB):
        cases = generate_boundary_curve_points(
            target_GB=budget,
            homo_or_not=homo_or_not,
            scale_max=scale_max,
            _N=_N,
            data_size=data_size,
            max_conn=max_conn,
            points_per_curve=points_per_curve,
            include_dense_ref=include_dense_ref,
        )
        all_cases.extend(cases)
    return all_cases


def summarize_boundary_results(results):
    if not results:
        print('No results.')
        return

    ok_rows = [row for row in results if row['status'] == 'OK']
    fail_rows = [row for row in results if row['status'] != 'OK']

    print('\n' + '#' * 140)
    print('SUMMARY')
    print('#' * 140)
    print(f'total cases : {len(results)}')
    print(f'ok          : {len(ok_rows)}')
    print(f'fail        : {len(fail_rows)}')

    if ok_rows:
        worst_abs = max(ok_rows, key=lambda row: row['max_abs_err'])
        worst_rel = max(ok_rows, key=lambda row: row['max_rel_err'])

        print('\nWorst absolute error case:')
        print(
            f"  encoding={worst_abs['encoding']}, "
            f"mode={worst_abs['mode']}, "
            f"k={worst_abs['target_GB']}GB, "
            f"shape=({worst_abs['m']}, {worst_abs['m']}), "
            f"conn={worst_abs['conn']}, "
            f"max_abs_err={worst_abs['max_abs_err']:.6e}, "
            f"max_err_index={worst_abs['max_err_index']}"
        )

        print('\nWorst relative error case:')
        print(
            f"  encoding={worst_rel['encoding']}, "
            f"mode={worst_rel['mode']}, "
            f"k={worst_rel['target_GB']}GB, "
            f"shape=({worst_rel['m']}, {worst_rel['m']}), "
            f"conn={worst_rel['conn']}, "
            f"max_rel_err={worst_rel['max_rel_err']:.6e}, "
            f"max_err_index={worst_rel['max_err_index']}"
        )

    if fail_rows:
        print('\nFailed cases:')
        for row in fail_rows:
            print(
                f"  encoding={row['encoding']}, "
                f"mode={row['mode']}, "
                f"k={row['target_GB']}GB, "
                f"shape=({row['m']}, {row['m']}), "
                f"conn={row['conn']}, "
                f"error={row.get('error', 'unknown')}"
            )


class BinaryDumpRecorder:
    def __init__(self, dump_dir: Optional[Path] = None):
        self.dump_dir = Path(dump_dir) if dump_dir is not None else DEFAULT_DUMP_DIR
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}

    def _write_dump(self, operator_name: str, suffix: str, writer: Callable[[Any, int], None]):
        dump_path = self.dump_dir / f'{operator_name}_{suffix}.txt'
        self.dump_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            dt = self._counters.get(operator_name, 0)
            self._counters[operator_name] = dt + 1
            mode = 'w' if dt == 0 else 'a'
            with dump_path.open(mode, encoding='utf-8') as stream:
                writer(stream, dt)

    def _write_row_sparse_dump(self, spike_indices, *, operator_name: str, n_cols: int):
        spike_indices = np.asarray(spike_indices)
        row_counts = np.count_nonzero(spike_indices, axis=1)
        total_nnz = int(row_counts.sum())

        if n_cols > 0:
            active_cols = spike_indices[spike_indices > 0].astype(np.int64) - 1
            if active_cols.size > 0:
                col_counts = np.bincount(active_cols, minlength=n_cols)
            else:
                col_counts = np.zeros((n_cols,), dtype=np.int64)
            avg_col_nnz = float(col_counts.mean())
        else:
            avg_col_nnz = 0.0

        avg_row_nnz = float(row_counts.mean()) if row_counts.size > 0 else 0.0

        def writer(stream, dt):
            stream.write(f'dt={dt}\n')
            stream.write(f'operator={operator_name}\n')
            stream.write(f'compressed_shape={tuple(spike_indices.shape)}\n')
            stream.write(f'source_n_cols={n_cols}\n')
            stream.write(f'total_nnz={total_nnz}\n')
            stream.write(f'avg_row_nnz={avg_row_nnz:.6f}\n')
            stream.write(f'avg_col_nnz={avg_col_nnz:.6f}\n')
            stream.write('spike_indices=\n')
            np.savetxt(stream, spike_indices, fmt='%d')
            stream.write('---\n\n')

        self._write_dump(operator_name, 'row_sparse_dump', writer)

    def dump_row_sparse_encode(self, spike_indices, *, operator_name: str, n_cols: int):
        jax.debug.callback(
            lambda value: self._write_row_sparse_dump(value, operator_name=operator_name, n_cols=n_cols),
            spike_indices,
            ordered=False,
        )
        return spike_indices

    def _write_pair_stream_dump(
        self,
        pair_stream,
        n_pairs,
        *,
        operator_name: str,
        n_rows: int,
        n_cols: int,
    ):
        pair_stream = np.asarray(pair_stream)
        n_pairs = int(np.asarray(n_pairs).reshape(-1)[0])
        valid_n_pairs = max(0, min(n_pairs, pair_stream.shape[0]))
        valid_pairs = pair_stream[:valid_n_pairs]

        if n_rows > 0 and valid_n_pairs > 0:
            row_counts = np.bincount(valid_pairs[:, 0].astype(np.int64), minlength=n_rows)
        else:
            row_counts = np.zeros((n_rows,), dtype=np.int64)

        if n_cols > 0 and valid_n_pairs > 0:
            col_counts = np.bincount(valid_pairs[:, 1].astype(np.int64), minlength=n_cols)
        else:
            col_counts = np.zeros((n_cols,), dtype=np.int64)

        total_nnz = valid_n_pairs
        avg_row_nnz = float(row_counts.mean()) if row_counts.size > 0 else 0.0
        avg_col_nnz = float(col_counts.mean()) if col_counts.size > 0 else 0.0

        def writer(stream, dt):
            stream.write(f'dt={dt}\n')
            stream.write(f'operator={operator_name}\n')
            stream.write(f'pair_stream_capacity_shape={tuple(pair_stream.shape)}\n')
            stream.write(f'valid_pair_stream_shape={tuple(valid_pairs.shape)}\n')
            stream.write(f'source_n_rows={n_rows}\n')
            stream.write(f'source_n_cols={n_cols}\n')
            stream.write(f'n_pairs={valid_n_pairs}\n')
            stream.write(f'total_nnz={total_nnz}\n')
            stream.write(f'avg_row_nnz={avg_row_nnz:.6f}\n')
            stream.write(f'avg_col_nnz={avg_col_nnz:.6f}\n')
            stream.write('pair_stream=\n')
            np.savetxt(stream, valid_pairs, fmt='%d')
            stream.write('---\n\n')

        self._write_dump(operator_name, 'pair_dump', writer)

    def dump_pair_stream_encode(
        self,
        pair_stream,
        n_pairs,
        *,
        operator_name: str,
        n_rows: int,
        n_cols: int,
    ):
        jax.debug.callback(
            lambda value, count: self._write_pair_stream_dump(
                value,
                count,
                operator_name=operator_name,
                n_rows=n_rows,
                n_cols=n_cols,
            ),
            pair_stream,
            n_pairs,
            ordered=False,
        )
        return pair_stream, n_pairs

    def _write_csr_dump(
        self,
        csr_indices,
        csr_indptr,
        *,
        operator_name: str,
        n_rows: int,
        n_cols: int,
    ):
        csr_indices = np.asarray(csr_indices)
        csr_indptr = np.asarray(csr_indptr)
        nnz_from_indptr = int(csr_indptr[-1]) if csr_indptr.size > 0 else 0
        valid_nnz = max(0, min(nnz_from_indptr, csr_indices.shape[0]))
        valid_indices = csr_indices[:valid_nnz]

        if csr_indptr.size > 1:
            row_counts = np.diff(csr_indptr.astype(np.int64))
        else:
            row_counts = np.zeros((n_rows,), dtype=np.int64)

        if n_cols > 0 and valid_nnz > 0:
            col_counts = np.bincount(valid_indices.astype(np.int64), minlength=n_cols)
        else:
            col_counts = np.zeros((n_cols,), dtype=np.int64)

        total_nnz = valid_nnz
        avg_row_nnz = float(row_counts.mean()) if row_counts.size > 0 else 0.0
        avg_col_nnz = float(col_counts.mean()) if col_counts.size > 0 else 0.0

        def writer(stream, dt):
            stream.write(f'dt={dt}\n')
            stream.write(f'operator={operator_name}\n')
            stream.write(f'csr_indices_capacity_shape={tuple(csr_indices.shape)}\n')
            stream.write(f'csr_indptr_shape={tuple(csr_indptr.shape)}\n')
            stream.write(f'valid_nnz={valid_nnz}\n')
            stream.write(f'source_n_rows={n_rows}\n')
            stream.write(f'source_n_cols={n_cols}\n')
            stream.write(f'total_nnz={total_nnz}\n')
            stream.write(f'avg_row_nnz={avg_row_nnz:.6f}\n')
            stream.write(f'avg_col_nnz={avg_col_nnz:.6f}\n')
            stream.write('csr_indptr=\n')
            np.savetxt(stream, csr_indptr.reshape((1, -1)), fmt='%d')
            stream.write('csr_indices=\n')
            np.savetxt(stream, valid_indices.reshape((1, -1)), fmt='%d')
            stream.write('---\n\n')

        self._write_dump(operator_name, 'csr_dump', writer)

    def dump_csr_encode(
        self,
        csr_indices,
        csr_indptr,
        *,
        operator_name: str,
        n_rows: int,
        n_cols: int,
    ):
        jax.debug.callback(
            lambda indices, indptr: self._write_csr_dump(
                indices,
                indptr,
                operator_name=operator_name,
                n_rows=n_rows,
                n_cols=n_cols,
            ),
            csr_indices,
            csr_indptr,
            ordered=False,
        )
        return csr_indices, csr_indptr

    def _write_dense_spike_dump(self, matrix, *, operator_name: str):
        matrix = np.asarray(matrix)
        row_counts = np.count_nonzero(matrix, axis=1)
        col_counts = np.count_nonzero(matrix, axis=0)
        total_nnz = int(np.count_nonzero(matrix))
        avg_row_nnz = float(row_counts.mean()) if row_counts.size > 0 else 0.0
        avg_col_nnz = float(col_counts.mean()) if col_counts.size > 0 else 0.0
        fmt = '%d' if (
            np.issubdtype(matrix.dtype, np.integer) or np.issubdtype(matrix.dtype, np.bool_)
        ) else '%.8g'

        def writer(stream, dt):
            stream.write(f'dt={dt}\n')
            stream.write(f'operator={operator_name}\n')
            stream.write(f'matrix_shape={tuple(matrix.shape)}\n')
            stream.write(f'total_nnz={total_nnz}\n')
            stream.write(f'avg_row_nnz={avg_row_nnz:.6f}\n')
            stream.write(f'avg_col_nnz={avg_col_nnz:.6f}\n')
            stream.write('matrix=\n')
            np.savetxt(stream, matrix, fmt=fmt)
            stream.write('---\n\n')

        self._write_dump(operator_name, 'dense_spike_dump', writer)

    def dump_dense_spike_matrix(self, matrix, *, operator_name: str):
        jax.debug.callback(
            lambda value: self._write_dense_spike_dump(value, operator_name=operator_name),
            matrix,
            ordered=False,
        )
        return matrix


binary_dump_recorder = BinaryDumpRecorder()


def _run_compact_case(weights, indices, spikes, shape, transpose):
    compact = CompactBinary.from_array(spikes)
    return compact_binary_fcnmv(
        weights,
        indices,
        compact.packed,
        compact.active_ids,
        compact.n_active,
        compact.value,
        shape=shape,
        transpose=transpose,
    )


def _run_bitpack_case(weights, indices, spikes, shape, transpose):
    packed = bitpack(spikes, axis=0)
    return bitpack_binary_fcnmv(
        weights,
        indices,
        packed,
        spikes,
        shape=shape,
        transpose=transpose,
    )


def run_boundary_tests_generic(
    run_case: Callable[[jax.Array, jax.Array, jax.Array, Tuple[int, int], bool], jax.Array],
    *,
    encoding_name: str,
    limit_GB: float = DEFAULT_LIMIT_GB,
    scale_max: int = DEFAULT_SCALE_MAX,
    _N: int = DEFAULT_BASE_NEURONS,
    data_size: int = DEFAULT_DATA_SIZE,
    max_conn: int = DEFAULT_MAX_CONN,
    points_per_curve: int = DEFAULT_POINTS_PER_CURVE,
    include_dense_ref: bool = DEFAULT_INCLUDE_DENSE_REF,
    transpose: bool,
    test_homo: bool = DEFAULT_TEST_HOMO,
    test_hetero: bool = DEFAULT_TEST_HETERO,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    rel_eps: float = DEFAULT_REL_EPS,
):
    results = []

    modes = []
    if test_homo:
        modes.append(True)
    if test_hetero:
        modes.append(False)
    if not modes:
        raise ValueError('At least one of test_homo/test_hetero must be True.')

    for homo_w in modes:
        mode_name = 'homo' if homo_w else 'hetero'
        cases = generate_all_boundary_cases(
            limit_GB=limit_GB,
            homo_or_not=homo_w,
            scale_max=scale_max,
            _N=_N,
            data_size=data_size,
            max_conn=max_conn,
            points_per_curve=points_per_curve,
            include_dense_ref=include_dense_ref,
        )

        print('=' * 140)
        print(f'encoding={encoding_name}, mode={mode_name}, total_cases={len(cases)}')
        print('=' * 140)

        for index, case in enumerate(cases, start=1):
            k_gb = case['target_GB']
            scale = case['scale']
            m = case['m']
            conn = case['conn']
            shape = (m, m)
            t0 = time.perf_counter()

            try:
                indices = generate_fixed_conn_num_indices(m, m, conn)
                weights = _mk_homo_w() if homo_w else _mk_hetero_w(indices)
                spikes = _mk_spikes(m)
                y = run_case(weights, indices, spikes, shape, transpose)
                y_ref = _ref_mv(weights, indices, spikes, shape, transpose)

                y = jax.block_until_ready(y)
                y_ref = jax.block_until_ready(y_ref)

                errs = compute_error_metrics(y, y_ref, rel_eps=rel_eps)
                allclose_ok = bool(jnp.allclose(y, y_ref, rtol=rtol, atol=atol))
                t1 = time.perf_counter()

                row = {
                    'status': 'OK',
                    'encoding': encoding_name,
                    'mode': mode_name,
                    'target_GB': k_gb,
                    'scale': scale,
                    'm': m,
                    'conn': conn,
                    'used_GB': case['used_GB'],
                    'utilization': case['utilization'],
                    'transpose': transpose,
                    'allclose': allclose_ok,
                    'rtol': rtol,
                    'atol': atol,
                    'elapsed_sec': t1 - t0,
                    **errs,
                }
                results.append(row)

                print(
                    f'[{index:02d}] '
                    f'k={k_gb:>4.1f}GB  '
                    f'scale={scale:>4d}  '
                    f'shape={shape!s:<18}  '
                    f'conn={conn:>5d}  '
                    f'used={case["used_GB"]:.3f}GB  '
                    f'util={case["utilization"]:.4f}  '
                    f'allclose={allclose_ok!s:<5}  '
                    f'max_abs={errs["max_abs_err"]:.6e}  '
                    f'mean_abs={errs["mean_abs_err"]:.6e}  '
                    f'max_rel={errs["max_rel_err"]:.6e}  '
                    f'mean_rel={errs["mean_rel_err"]:.6e}  '
                    f'time={t1 - t0:.3f}s'
                )
            except Exception as exc:
                t1 = time.perf_counter()
                row = {
                    'status': 'FAIL',
                    'encoding': encoding_name,
                    'mode': mode_name,
                    'target_GB': k_gb,
                    'scale': scale,
                    'm': m,
                    'conn': conn,
                    'used_GB': case['used_GB'],
                    'utilization': case['utilization'],
                    'transpose': transpose,
                    'allclose': False,
                    'rtol': rtol,
                    'atol': atol,
                    'elapsed_sec': t1 - t0,
                    'max_abs_err': float('nan'),
                    'mean_abs_err': float('nan'),
                    'max_rel_err': float('nan'),
                    'mean_rel_err': float('nan'),
                    'max_err_index': None,
                    'y_at_max_err': float('nan'),
                    'y_ref_at_max_err': float('nan'),
                    'error': repr(exc),
                }
                results.append(row)

                print(
                    f'[{index:02d}] '
                    f'k={k_gb:>4.1f}GB  '
                    f'scale={scale:>4d}  '
                    f'shape={shape!s:<18}  '
                    f'conn={conn:>5d}  '
                    f'used={case["used_GB"]:.3f}GB  '
                    f'util={case["utilization"]:.4f}  '
                    f'status=FAIL  '
                    f'error={repr(exc)}'
                )

            gc.collect()

    return results


def run_compact_tol_tests(
    limit_GB: float = DEFAULT_LIMIT_GB,
    scale_max: int = DEFAULT_SCALE_MAX,
    _N: int = DEFAULT_BASE_NEURONS,
    data_size: int = DEFAULT_DATA_SIZE,
    max_conn: int = DEFAULT_MAX_CONN,
    points_per_curve: int = DEFAULT_POINTS_PER_CURVE,
    include_dense_ref: bool = DEFAULT_INCLUDE_DENSE_REF,
    transpose: bool = DEFAULT_COMPACT_TRANSPOSE,
    test_homo: bool = DEFAULT_TEST_HOMO,
    test_hetero: bool = DEFAULT_TEST_HETERO,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    rel_eps: float = DEFAULT_REL_EPS,
):
    return run_boundary_tests_generic(
        _run_compact_case,
        encoding_name='compact',
        limit_GB=limit_GB,
        scale_max=scale_max,
        _N=_N,
        data_size=data_size,
        max_conn=max_conn,
        points_per_curve=points_per_curve,
        include_dense_ref=include_dense_ref,
        transpose=transpose,
        test_homo=test_homo,
        test_hetero=test_hetero,
        rtol=rtol,
        atol=atol,
        rel_eps=rel_eps,
    )


def run_bitpack_tol_tests(
    limit_GB: float = DEFAULT_LIMIT_GB,
    scale_max: int = DEFAULT_SCALE_MAX,
    _N: int = DEFAULT_BASE_NEURONS,
    data_size: int = DEFAULT_DATA_SIZE,
    max_conn: int = DEFAULT_MAX_CONN,
    points_per_curve: int = DEFAULT_POINTS_PER_CURVE,
    include_dense_ref: bool = DEFAULT_INCLUDE_DENSE_REF,
    transpose: bool = DEFAULT_BITPACK_TRANSPOSE,
    test_homo: bool = DEFAULT_TEST_HOMO,
    test_hetero: bool = DEFAULT_TEST_HETERO,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    rel_eps: float = DEFAULT_REL_EPS,
):
    return run_boundary_tests_generic(
        _run_bitpack_case,
        encoding_name='bitpack',
        limit_GB=limit_GB,
        scale_max=scale_max,
        _N=_N,
        data_size=data_size,
        max_conn=max_conn,
        points_per_curve=points_per_curve,
        include_dense_ref=include_dense_ref,
        transpose=transpose,
        test_homo=test_homo,
        test_hetero=test_hetero,
        rtol=rtol,
        atol=atol,
        rel_eps=rel_eps,
    )


def _print_menu():
    print('\nFCN Tools Menu')
    print('1. Run compact tol tests')
    print('2. Run bitpack tol tests')
    print('0. Exit')


def main():
    while True:
        _print_menu()
        try:
            choice = input('Select an option: ').strip()
        except EOFError:
            print('\nEOF received. Exiting.')
            return

        if choice == '1':
            results = run_compact_tol_tests()
            summarize_boundary_results(results)
        elif choice == '2':
            results = run_bitpack_tol_tests()
            summarize_boundary_results(results)
        elif choice == '0':
            print('Exiting.')
            return
        else:
            print(f'Invalid selection: {choice!r}. Please choose 1, 2, or 0.')


if __name__ == '__main__':
    main()
