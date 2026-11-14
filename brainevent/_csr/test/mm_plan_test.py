# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the shape-only hybrid MM plan in ``brainevent/_csr/config.py``.

See ``docs/specs/2026-11-14-csr-mm-shape-plan.md`` (D1-D5). The plan reads
only static shapes, the value dtype and the target L2 size; its task queue
must hold every long-row task the native kernel can emit for *any* structure
with those shapes.
"""

import json
import shutil
import subprocess
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.config import _MMPlan, _mm_group, _mm_plan
from brainevent._misc import _long_row_mask

TASK_EDGES = 512  # 32 lanes x 16 iterations: edges per long-row task
INT64_MAX = int(np.iinfo(np.int64).max)
L2 = 4 << 20


def _plan(rows, cols, nnz, batch=8, dtype=np.float32, weighted=False, l2_bytes=L2):
    return _mm_plan(rows, cols, nnz, batch, np.dtype(dtype), weighted, l2_bytes)


def _expected_group(mean):
    """Spec D3, independently: a quarter of min(mean, 512), rounded up to 4, 8, 16 or 32."""
    quarter = -(-min(mean, 512) // 4)
    return next((group for group in (4, 8, 16, 32) if quarter <= group), 32)


# Lane group (D3)
# ---------------

@pytest.mark.parametrize('mean, group', [
    (0, 4), (1, 4), (16, 4),
    (17, 8), (32, 8),
    (33, 16), (64, 16),
    (65, 32), (128, 32), (512, 32), (513, 32), (2048, 32), (2049, 32), (10 ** 7, 32),
])
def test_mm_group_boundaries_follow_the_mean_row_length(mean, group):
    rows = 10
    # Floor division: up to rows - 1 extra edges keep the same mean.
    for nnz in (mean * rows, mean * rows + rows - 1):
        assert _mm_group(rows, nnz) == group
        plan = _plan(rows, 7, nnz)
        assert (plan.group, plan.long_threshold) == (group, 16 * group)


def test_mm_group_matches_the_spec_rule_for_every_mean():
    for mean in range(0, 3000):
        assert _mm_group(3, 3 * mean + 2) == _expected_group(mean), mean
    assert _mm_group(0, 0) == 4


# Long-task capacity (D4)
# -----------------------

def _kernel_task_demand(lengths, plan, active):
    """Largest number of queue slots one batch group of the native kernel uses.

    K1 turns every active row whose mask bit is set into ``ceil(len / 512)``
    tasks; all batches of one chunk share one queue of ``task_capacity``.
    """
    lengths = np.asarray(lengths, np.int64)
    tasks = np.where(lengths > plan.long_threshold, -(-lengths // TASK_EDGES), 0)
    per_batch = active.astype(np.int64) @ tasks
    groups = -(-active.shape[0] // plan.chunk)
    return max(int(per_batch[g * plan.chunk:(g + 1) * plan.chunk].sum()) for g in range(groups))


def _fill(rows, nnz, lengths):
    """Pad or trim ``lengths`` to ``rows`` rows holding exactly ``nnz`` edges."""
    lengths = np.zeros(rows, np.int64) if lengths is None else np.asarray(lengths, np.int64)[:rows].copy()
    lengths = np.pad(lengths, (0, rows - lengths.size))
    while lengths.sum() > nnz:
        excess = int(lengths.sum() - nnz)
        row = int(np.argmax(lengths))
        lengths[row] -= min(excess, int(lengths[row]))
    lengths[0] += nnz - lengths.sum()
    return lengths


def _spread(rows, nnz):
    """``nnz`` edges spread as evenly as possible over ``rows`` rows."""
    return np.full(rows, nnz // rows, np.int64) + (np.arange(rows) < nnz % rows) if rows else np.zeros(0, np.int64)


def _adversarial_structures(rows, nnz, threshold, rng):
    """Row-length vectors with ``rows`` rows and ``nnz`` edges that stress the task queue."""
    just_long = threshold + 1
    # Smallest length above the threshold that wastes almost a whole task: 512k + 1.
    wasteful = (just_long + TASK_EDGES - 1) // TASK_EDGES * TASK_EDGES + 1
    count_long, count_waste = min(rows, nnz // just_long), min(rows, nnz // wasteful)
    hub = np.zeros(rows, np.int64)
    hub[-1] = nnz
    pareto = np.floor(rng.pareto(1.2, rows) * 50)
    pareto = np.floor(pareto * (nnz / max(1.0, pareto.sum()))).astype(np.int64)
    return {
        'hub': hub,
        # As many long rows as the edges allow, the remainder in one more row.
        'just_long': _fill(rows, nnz, np.full(count_long, just_long)),
        # The same long rows, the remainder spread over short rows.
        'just_long_rest_short': np.r_[np.full(count_long, just_long),
                                      _spread(rows - count_long, nnz - count_long * just_long)]
        if count_long < rows else _fill(rows, nnz, np.full(count_long, just_long)),
        'wasteful': _fill(rows, nnz, np.full(count_waste, wasteful)),
        'at_threshold': _fill(rows, nnz, np.full(min(rows, nnz // threshold), threshold)),
        'uniform': _spread(rows, nnz),
        'pareto': _fill(rows, nnz, pareto),
        'multinomial': rng.multinomial(nnz, np.full(rows, 1 / rows)),
    }


@pytest.mark.parametrize('rows', [1, 2, 7, 31, 32, 33, 100, 1000])
def test_mm_plan_long_tasks_bound_every_structure_with_these_shapes(rows):
    rng = np.random.default_rng(rows)
    for mean in (0.5, 1, 16, 17, 33, 64, 65, 300, 513, 1500, 5000):
        nnz = max(1, int(rows * mean))
        for batch in (1, 3, 64):
            plan = _plan(rows, 97, nnz, batch, l2_bytes=97 * 4 * 4 * 2)  # chunk 2 when batch >= 2
            structures = _adversarial_structures(rows, nnz, plan.long_threshold, rng)
            for name, lengths in structures.items():
                assert lengths.shape == (rows,) and lengths.sum() == nnz and lengths.min() >= 0, name
                exact = int(np.where(lengths > plan.long_threshold, -(-lengths // TASK_EDGES), 0).sum())
                assert exact <= plan.long_tasks, (rows, nnz, name)
                active = np.ones((batch, rows), bool)
                assert _kernel_task_demand(lengths, plan, active) <= plan.task_capacity
                sparse = rng.random((batch, rows)) < 0.3
                assert _kernel_task_demand(lengths, plan, sparse) <= plan.task_capacity


def test_mm_plan_long_tasks_bound_random_row_lengths():
    rng = np.random.default_rng(2026)
    for _ in range(400):
        rows = int(rng.integers(1, 300))
        scale = int(rng.choice([4, 70, 600, 5000]))
        lengths = rng.integers(0, scale, size=rows)
        if rng.random() < 0.4:
            lengths[rng.integers(rows)] = int(rng.integers(0, 40000))
        nnz = int(lengths.sum())
        if not nnz:
            continue
        batch = int(rng.integers(1, 40))
        plan = _plan(rows, int(rng.integers(1, 3000)), nnz, batch)
        active = rng.random((batch, rows)) < rng.random()
        active[0] = True
        assert _kernel_task_demand(lengths, plan, active) <= plan.task_capacity
        assert int(np.where(lengths > plan.long_threshold, -(-lengths // TASK_EDGES), 0).sum()) <= plan.long_tasks


@pytest.mark.parametrize('rows, nnz', [(1, 1), (1, 513), (64, 512), (64, 64 * 65), (1000, 300_000), (10, 10 ** 6)])
def test_mm_plan_capacity_is_tight_on_adversarial_structures(rows, nnz):
    """Each term of the bound is reached: one hub row, and many just-long rows."""
    plan = _plan(rows, 64, nnz)
    threshold = plan.long_threshold
    tasks_of = lambda lengths: int(np.where(lengths > threshold, -(-lengths // TASK_EDGES), 0).sum())
    hub = np.zeros(rows, np.int64)
    hub[0] = nnz
    hub_tasks = tasks_of(hub)
    count = min(rows, nnz // (threshold + 1))
    just_long = _fill(rows, nnz, np.full(count, threshold + 1))
    just_long_tasks = tasks_of(just_long)
    assert plan.long_tasks == -(-nnz // TASK_EDGES) + count
    assert max(hub_tasks, just_long_tasks) <= plan.long_tasks
    if nnz > threshold:
        assert hub_tasks == -(-nnz // TASK_EDGES)  # the ceil(nnz / 512) term alone
    assert just_long_tasks >= count  # every just-long row needs at least one task
    assert plan.long_tasks <= hub_tasks + just_long_tasks + 1


def test_mm_plan_mask_and_capacity_cover_the_kernel_on_device_masks():
    """Use the lowered mask helper itself to select long rows."""
    rng = np.random.default_rng(7)
    for rows, nnz in ((33, 33 * 70), (95, 4000), (64, 64 * 17)):
        plan = _plan(rows, 50, nnz, batch=5)
        for lengths in _adversarial_structures(rows, nnz, plan.long_threshold, rng).values():
            indptr = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32))
            mask = np.asarray(_long_row_mask(indptr, plan.long_threshold))
            bits = np.unpackbits(mask.view(np.uint8), bitorder='little')[:rows].astype(bool)
            np.testing.assert_array_equal(bits, lengths > plan.long_threshold)
            tasks = int((-(-lengths[bits] // TASK_EDGES)).sum())
            assert plan.chunk * tasks <= plan.task_capacity


# Batch chunk and scratch sizes (D5)
# ----------------------------------

@pytest.mark.parametrize('dtype, element_bytes', [
    (np.float16, 4), (jnp.bfloat16, 4), (np.float32, 4), (np.float64, 8),
])
def test_mm_plan_chunk_keeps_one_output_chunk_in_a_quarter_of_l2(dtype, element_bytes):
    cols = 1000
    fit = (L2 // 4) // (cols * element_bytes)  # half values accumulate in float32
    assert _plan(10, cols, 100, batch=1000, dtype=dtype).chunk == fit
    assert _plan(10, cols, 100, batch=fit - 1, dtype=dtype).chunk == fit - 1
    assert _plan(10, cols, 100, batch=1, dtype=dtype).chunk == 1
    # A smaller L2 gives a smaller chunk; a row wider than a quarter of L2 still gets one batch.
    assert _plan(10, cols, 100, batch=1000, dtype=dtype, l2_bytes=L2 // 2).chunk == fit // 2
    assert _plan(10, L2, 100, batch=1000, dtype=dtype).chunk == 1


def test_mm_plan_chunk_is_capped_at_the_native_group_limit():
    plan = _plan(4, 1, 2048, batch=70_000, l2_bytes=1 << 30)
    assert plan.chunk == 65535
    assert plan.task_capacity == 65535 * plan.long_tasks
    assert plan.short_rows_size == 65535 * 4
    # Two batch groups, each with one 8-byte task counter.
    binary = 8 * 70_000 + 8 * 2 + 12 * plan.task_capacity
    assert plan.workspace_words * 4 == binary


SHORT_ROWS_FLOOR = 1 << 20  # int32 entries (4 MiB)


@pytest.mark.parametrize('rows, cols, nnz, batch', [
    (1_000_000, 10, 10_000_000, 1024),  # tall and narrow: the L2 rule alone gives chunk 1024
    (20_000, 8, 20_000, 256),
    (10_000_000, 2, 1_000_000, 4096),  # fewer edges than rows: one batch per group
])
def test_mm_plan_short_rows_never_exceed_the_index_array(rows, cols, nnz, batch):
    plan = _plan(rows, cols, nnz, batch, l2_bytes=6 << 20)
    assert plan.short_rows_size == plan.chunk * rows
    assert plan.short_rows_size <= max(nnz, SHORT_ROWS_FLOOR, rows)
    assert plan.chunk == max(1, max(nnz, SHORT_ROWS_FLOOR) // rows)
    assert plan.task_capacity == plan.chunk * plan.long_tasks


@pytest.mark.parametrize('rows, cols, nnz', [
    (4000, 4000, 320_000), (20_000, 20_000, 4_000_000), (100_000, 100_000, 10_000_000),
    (10_000, 200_000, 10_000_000),
])
def test_short_row_bound_leaves_the_benchmarked_shapes_on_the_l2_rule(rows, cols, nnz):
    for batch in (1, 8, 64, 256):
        fit = ((6 << 20) // 4) // (cols * 4)
        assert _plan(rows, cols, nnz, batch, l2_bytes=6 << 20).chunk == max(1, min(fit, batch))


@pytest.mark.parametrize('dtype', [np.float16, jnp.bfloat16, np.float32, np.float64])
@pytest.mark.parametrize('batch', [1, 5, 300])
def test_mm_plan_workspace_matches_the_native_layout(dtype, batch):
    value_bytes = np.dtype(dtype).itemsize
    for rows, cols, nnz in ((5, 128, 1571), (1000, 3000, 80_000), (3, 1, 2), (64, 600, 512)):
        binary = _plan(rows, cols, nnz, batch, dtype, weighted=False)
        weighted = _plan(rows, cols, nnz, batch, dtype, weighted=True)
        # The value queue is the only difference between the two layouts.
        assert (binary.group, binary.long_threshold, binary.long_tasks, binary.chunk,
                binary.task_capacity, binary.short_rows_size) == \
               (weighted.group, weighted.long_threshold, weighted.long_tasks, weighted.chunk,
                weighted.task_capacity, weighted.short_rows_size)
        groups = -(-batch // binary.chunk)
        layout = 8 * batch + 8 * groups + 12 * binary.task_capacity
        assert binary.workspace_words * 4 == layout
        queue = -(-binary.task_capacity * value_bytes // 4) * 4
        assert weighted.workspace_words * 4 == layout + queue
        assert binary.short_rows_size == binary.chunk * rows


@pytest.mark.parametrize('dtype', [np.float16, jnp.bfloat16])
def test_half_weighted_queue_pads_an_odd_capacity_to_uint32(dtype):
    # One row, one edge: one task per batch, so the capacity equals the chunk.
    for chunk in (1, 3, 5):
        binary = _plan(1, 1, 1, batch=chunk, dtype=dtype)
        weighted = _plan(1, 1, 1, batch=chunk, dtype=dtype, weighted=True)
        assert binary.task_capacity == chunk
        # chunk half values take ceil(2 * chunk / 4) uint32 words.
        assert weighted.workspace_words - binary.workspace_words == (2 * chunk + 3) // 4


@pytest.mark.parametrize('rows, cols, nnz, batch', [
    (0, 0, 0, 4), (0, 5, 0, 4), (5, 0, 0, 4), (5, 7, 0, 4), (5, 7, 3, 0), (0, 0, 0, 0),
])
def test_empty_mm_plan_reserves_no_scratch(rows, cols, nnz, batch):
    for weighted in (False, True):
        for dtype in (np.float16, np.float64):
            plan = _plan(rows, cols, nnz, batch, dtype, weighted)
            group = _mm_group(rows, nnz)
            assert plan == _MMPlan(group, 16 * group, 0, 1, 0, 0, 0)


def test_mm_plan_satisfies_the_native_attribute_checks():
    """The FFI rejects plans outside these invariants (``plan_from_attrs``)."""
    rng = np.random.default_rng(11)
    for _ in range(300):
        rows, cols = int(rng.integers(1, 5000)), int(rng.integers(1, 200_000))
        nnz = int(rng.integers(1, rows * 2000))
        batch = int(rng.integers(1, 100_000))
        dtype = (np.float16, np.float32, np.float64)[int(rng.integers(3))]
        plan = _plan(rows, cols, nnz, batch, dtype, bool(rng.integers(2)),
                     int(rng.choice([1 << 20, 6 << 20, 96 << 20])))
        assert plan.group in (4, 8, 16, 32) and plan.long_threshold == 16 * plan.group
        assert 1 <= plan.chunk <= min(batch, 65535)
        assert 0 < plan.long_tasks <= INT64_MAX // plan.chunk
        assert plan.task_capacity == plan.chunk * plan.long_tasks
        assert plan.short_rows_size == plan.chunk * rows


# Identity, caching and validation
# --------------------------------

def test_mm_plan_is_cached_per_shape_and_bounded():
    _mm_plan.cache_clear()
    first = _plan(10, 20, 30, 4)
    assert _plan(10, 20, 30, 4) is first
    assert _mm_plan.cache_info().hits == 1
    assert _plan(10, 20, 30, 4, weighted=True) is not first
    for batch in range(1, 300):
        _plan(10, 20, 30, batch)
    assert _mm_plan.cache_info().currsize <= 256


@pytest.mark.parametrize('kwargs, match', [
    (dict(batch=1.5), 'integer'), (dict(batch=True), 'integer'), (dict(rows=-1), '>= 0'),
    (dict(nnz=-1), '>= 0'), (dict(rows=2 ** 31), 'int32'), (dict(cols=2 ** 31), 'int32'),
    (dict(batch=2 ** 31), 'int32'),
])
def test_mm_plan_rejects_invalid_dimensions(kwargs, match):
    arguments = dict(rows=4, cols=4, nnz=4, batch=4)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        _plan(**arguments)


def test_mm_plan_rejects_scratch_beyond_int64():
    with pytest.raises(ValueError, match='int64'):
        _plan(1, 1, INT64_MAX, batch=65535, l2_bytes=1 << 62)


# Native layout parity
# --------------------

_NATIVE_SOURCE = r'''
#include <cstdint>
#include <iostream>
#include <vector>
#define __host__
#define __device__
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorInvalidValue = 1;
constexpr int cudaDevAttrMultiProcessorCount = 0, cudaDevAttrL2CacheSize = 1;
int cudaGetDevice(int *);
int cudaDeviceGetAttribute(int *, int, int);
namespace hybrid
@DECLARATIONS@
#include @PLAN_HEADER@
int main() {
    int rows, cols, batch, l2, out_bytes, value_bytes, weighted;
    int64_t nnz, threshold, long_tasks;
    while (std::cin >> rows >> cols >> batch >> l2 >> out_bytes >> value_bytes >> weighted >> nnz
                    >> threshold >> long_tasks) {
        std::vector<int64_t> ptr(rows + 1);
        for (auto &value : ptr) std::cin >> value;
        hybrid::Plan plan;
        plan.long_threshold = threshold;
        plan.chunk = std::min(hybrid::chunk_batches(cols, batch, l2, out_bytes), hybrid::kMaxGroupBatches);
        plan.long_tasks = long_tasks;
        plan.task_capacity = plan.chunk * long_tasks;
        std::cout << hybrid::group_size(nnz, rows) << ' ' << plan.chunk << ' '
                  << hybrid::workspace_bytes(plan, batch, weighted ? value_bytes : 0) / 4;
        for (auto word : hybrid::long_row_mask(ptr.data(), rows, plan)) std::cout << ' ' << word;
        std::cout << '\n';
    }
}
'''


def test_mm_plan_matches_the_native_sizing_and_mask(tmp_path):
    """Compare with the packaged C++ helpers the FFI uses to check scratch sizes."""
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable for native planner parity')
    root = Path(__file__).parents[1] / 'binary' / 'hybrid_mm'
    declarations = (root / 'hybrid_batchedv3_spmm.cuh').read_text().split('namespace hybrid', 1)[1]
    declarations = declarations.split('// device_info,', 1)[0]
    source = (_NATIVE_SOURCE.replace('@DECLARATIONS@', declarations)
              .replace('@PLAN_HEADER@', json.dumps(str(root / 'hybrid_batchedv3_spmm_plan.cuh'))))
    cpp, binary = tmp_path / 'plan.cpp', tmp_path / 'plan'
    cpp.write_text(source)
    subprocess.run([compiler, '-std=c++17', '-O2', str(cpp), '-o', str(binary)], check=True,
                   capture_output=True, text=True)

    rng = np.random.default_rng(4)
    golden = [[1, 1, 1, 1, 65], [16, 17, 513], [64, 65, 513], [0, 513, 1024, 1025],
              [512, 513], [1] * 32 + [65], [1] * 64 + [65], [1025], [3000] * 40]
    cases = []
    for i in range(48):
        lengths = np.array(golden[i]) if i < len(golden) else rng.integers(0, 1400, size=int(rng.integers(1, 80)))
        indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        rows, nnz = lengths.size, int(indptr[-1])
        if not nnz:
            continue
        cols, batch = int(rng.integers(1, 2048)), int(rng.integers(1, 128))
        dtype = (np.float16, np.float32, np.float64)[i % 3]
        weighted = bool(i % 2)
        l2 = int(rng.choice([4096, 1 << 20, 6 << 20]))
        if i == 1:
            cols, batch, l2 = 1, 70_000, 1 << 30  # the 65535 cap
        plan = _mm_plan(rows, cols, nnz, batch, np.dtype(dtype), weighted, l2)
        itemsize = np.dtype(dtype).itemsize
        line = [rows, cols, batch, l2, max(4, itemsize), itemsize, int(weighted), nnz,
                plan.long_threshold, plan.long_tasks, *indptr]
        cases.append((' '.join(map(str, line)), plan, indptr))
    native = subprocess.run([str(binary)], input='\n'.join(case[0] for case in cases) + '\n',
                            capture_output=True, check=True, text=True)
    outputs = native.stdout.splitlines()
    assert len(outputs) == len(cases)
    for (_, plan, indptr), line in zip(cases, outputs):
        numbers = list(map(int, line.split()))
        assert numbers[:3] == [plan.group, plan.chunk, plan.workspace_words]
        mask = _long_row_mask(jnp.asarray(indptr.astype(np.int32)), plan.long_threshold)
        np.testing.assert_array_equal(numbers[3:], np.asarray(mask))


def test_mm_plan_computes_with_python_integers():
    """NumPy integer shapes plan like Python ints, without fixed-width overflow."""
    shapes = (100_000, 1, 10**6, 65535)
    _mm_plan.cache_clear()  # equal NumPy and Python keys would share one cache entry
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        planned = _mm_plan(*map(np.int32, shapes), np.dtype(np.float32), False, np.int64(1 << 30))
    _mm_plan.cache_clear()
    assert planned == _mm_plan(*shapes, np.dtype(np.float32), False, 1 << 30)
    assert all(type(value) is int for value in vars(planned).values())
