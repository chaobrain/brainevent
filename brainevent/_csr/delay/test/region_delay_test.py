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

"""Tests for ``brainevent/_csr/delay/region_delay.py``, grouped by behavior."""

import importlib
from copy import copy
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

import brainevent
from brainevent import CSR, BinaryArray, KernelCompilationError, WeightedArray
from brainevent._csr.delay import delay as operator
from brainevent._csr.delay import region_delay as module
from brainevent._csr.delay.delay import csr_delay_deliver_layout_p
from brainevent._csr.delay.region_delay import (
    TASK_NNZ,
    TPR_THRESHOLD,
    HybridWorkspace,
    WeightedHybridWorkspace,
    csr_delay_deliver_layout,
    delaycsr,
)
from brainevent._csr.delay.test._helpers import BACKENDS, GPU, cuda_step
from brainevent._csr.delay.test._reference import (
    EVENT_DTYPES,
    HYBRID_ROW_LENGTHS,
    INDEX_DTYPES,
    VALUE_DTYPES,
    active_mask,
    assert_close,
    assert_same_bits,
    csr_from_lengths,
    dtype_name,
    large_indices,
    large_structure,
    numpy_project_step,
    random_events,
    random_values,
)


# Prepared delay state and native execution
# -----------------------------------------

COLUMNS = 3000


DELAYS = (0, 1, 3, 4)


SLOTS = 6


SUM_SCALE = 20.0


def _arrays(data: delaycsr) -> dict[str, jax.Array]:
    return {name: jnp.asarray(getattr(data, name)) for name in
            ("weights", "indices", "indptr", "delay_count", "delay_off", "row_task_count")}


def _workspace(data: delaycsr, offset_dtype: Any = jnp.int32) -> HybridWorkspace:
    return data.workspace._replace(
        task_out_off=data.workspace.task_out_off.astype(offset_dtype),
        status=jnp.array([17, 1], jnp.int32),
    )


def _arrivals(weights: NDArray[Any], columns: NDArray[Any], ptr: NDArray[Any],
              delays: NDArray[Any], events: NDArray[Any], n: int, slots: int) -> tuple[NDArray[Any], NDArray[Any]]:
    ring = np.zeros((slots, n), np.float64)
    outputs = []
    for time in range(events.shape[1]):
        ring = numpy_project_step(weights, columns, ptr, delays, events[:, time], time, ring)
        outputs.append(ring[(time + 1) % slots].copy())
    return np.asarray(outputs), ring


def _run(data: delaycsr, events: NDArray[Any], arrays: dict[str, jax.Array]) -> tuple[NDArray[Any], NDArray[Any]]:
    ring = jnp.zeros((data.L, data.shape[1]), jnp.float32)
    work = _workspace(data, arrays["delay_off"].dtype)
    hybrid = jax.jit(cuda_step)
    outputs = []
    for time in range(events.shape[1]):
        slot = time % data.L
        head, event = jnp.array([slot], jnp.int32), jnp.asarray(events[:, time])
        ring, work = hybrid(**arrays, events=event, head=head, buffer=ring, workspace=work)
        expected_tasks = int(data.row_task_count[events[:, time] != 0].sum())
        np.testing.assert_array_equal(work.status, [expected_tasks, 0])
        outputs.append(np.asarray(ring)[(slot + 1) % data.L].copy())
    return np.stack(outputs), np.asarray(ring)


def _small(delays: tuple[int, ...] = (0, 2, 0, 2)) -> delaycsr:
    return delaycsr(
        (np.array([2, 3, 5, 7], np.float32), np.array([3, 0, 2, 1], np.int32), np.array([0, 4, 4], np.int32)),
        edge_delays=np.asarray(delays),
        shape=(2, 4),
        delay_steps=(0, 1, 2),
    )


def _dtype_problem(seed: int, value_dtype, index_dtype, ptr_dtype, homogeneous: bool = False, steps: int = 3,
                   event_dtype=jnp.bool_):
    rng = np.random.default_rng(seed)
    lengths = np.tile(np.asarray(HYBRID_ROW_LENGTHS), 2)
    indptr, indices = csr_from_lengths(rng, lengths, COLUMNS, ptr_dtype)
    indices = indices.astype(index_dtype)
    weights = random_values(rng, 1 if homogeneous else indices.size, value_dtype)
    delays = rng.choice(DELAYS, size=indices.size)
    events = random_events(rng, (lengths.size, steps), event_dtype, rate=0.5)
    data = delaycsr(
        (weights, indices, indptr),
        edge_delays=delays,
        shape=(len(indptr) - 1, COLUMNS),
        delay_steps=DELAYS,
        buffer_slots=SLOTS,
    )
    return data, (weights, indices, indptr, delays, events)


def _weighted_problem(dtype, index_dtype=np.int32, edge_dtype=np.int32, homo=False, wide=False):
    rng = np.random.default_rng(42)
    lengths = np.array([0, 1, 128, 129, 4096, 4097, 9500])
    ptr = np.r_[0, lengths.cumsum()].astype(edge_dtype)
    indices = rng.integers(8193, size=ptr[-1], dtype=index_dtype)
    weights = (np.array([0.125]) if homo else rng.integers(-4, 5, size=indices.size) / 32).astype(jnp.dtype(dtype))
    delays = rng.choice([0, 1, 4], size=indices.size)
    delays[ptr[-2]:] = np.repeat([0, 1, 4], [17, 5001, 4482])
    events = np.array([[1, 0, 2], [-2, 0, 0.5], [0.25, -1, 0], [-0.5, 0, 1.5],
                       [1.25, 0, 0], [-1.5, 0, -0.25], [0.5, 0, -2]], dtype=jnp.dtype(dtype))
    data = delaycsr(
        (weights, indices, ptr),
        edge_delays=delays,
        shape=(len(ptr) - 1, 8193),
        delay_steps=(0, 1, 3, 4),
        buffer_slots=6,
        weighted=True,
    )
    if wide:
        data.delay_off = data.delay_off.astype(jnp.int64)
        data.workspace = data._make_workspace(data.ring.dtype, data.weighted)
    return data, (weights, indices, ptr, delays), events


def test_numpy_step_matches_hand_calculated_ring() -> None:
    """Validate duplicate edges, binary activity and wrapped arrivals by hand."""
    columns, ptr = np.array([1, 1, 2, 0]), np.array([0, 3, 4])
    delays, events = np.array([0, 0, 1, 2]), np.array([-4, 0], np.float32)
    initial = np.ones((3, 3), np.float64)
    for weights, expected in (
        (np.array([2, 3, 5, 7], np.float32), [[1, 1, 6], [1, 1, 1], [1, 6, 1]]),
        (np.array([0.5], np.float32), [[1, 1, 1.5], [1, 1, 1], [1, 2, 1]]),
    ):
        actual = numpy_project_step(weights, columns, ptr, delays, events, 2, initial)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(initial, 1)
        silent = actual.copy()
        np.testing.assert_array_equal(
            numpy_project_step(weights, columns, ptr, delays, np.zeros(2), 0, actual), silent)


@GPU
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("pointer_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("event_dtype", [np.bool_, np.float32])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("wide_offsets", [False, True])
def test_projection_matches_original_edges(
    index_dtype: Any, pointer_dtype: Any, event_dtype: Any, homogeneous: bool, wide_offsets: bool,
) -> None:
    """Cover independent edge delays, empty segments, long rows and T != K."""
    lengths = np.array([0, 1, 128, 129, 4095, 4096, 4097, 5400])
    ptr = np.r_[0, np.cumsum(lengths)].astype(pointer_dtype)
    rng = np.random.default_rng(91)
    columns = rng.integers(4099, size=int(ptr[-1]), dtype=index_dtype)
    weights = np.array([0.25], np.float32) if homogeneous else rng.uniform(-2, 2, columns.size).astype(np.float32)
    delays = rng.choice([0, 2, 5], size=columns.size)
    delays[int(ptr[-2]):] = np.repeat([0, 2, 5], [100, 5000, 300])
    events = (rng.random((len(lengths), 13)) < .4).astype(event_dtype)
    events[:, 0], events[:, 1] = 1, 0
    if event_dtype == np.float32:
        events[:, 2] = [-2, 0, -1, 3, -4, 5, 0, -6]
    data = delaycsr(
        (weights, columns, ptr),
        edge_delays=delays,
        shape=(len(ptr) - 1, 4099),
        delay_steps=(0, 1, 2, 5),
        buffer_slots=8,
    )
    arrays = _arrays(data)
    if wide_offsets:
        arrays["delay_off"] = arrays["delay_off"].astype(jnp.int64)
    expected_outputs = _arrivals(weights, columns, ptr, delays, events, 4099, 8)
    for observed, expected in zip(_run(data, events, arrays), expected_outputs):
        np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-5)


@GPU
@pytest.mark.parametrize("duration_steps,empty_segment", [
    pytest.param(8, None, id="T-equals-K"),
    pytest.param(17, None, id="T-greater-than-K-wraps"),
    pytest.param(3, None, id="T-less-than-K"),
    pytest.param(3, 4, id="T-less-than-K-empty-segment"),
])
@pytest.mark.parametrize("event_dtype", [np.bool_, np.float32])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_each_event_column_matches_numpy_step(
    duration_steps: int, empty_segment: int | None, event_dtype: Any, homogeneous: bool,
) -> None:
    """Compare 10k-square, eight-delay projections at 30% activity with NumPy.

    Parameters
    ----------
    duration_steps : int
        Event column count T; K is fixed at eight delay segments.
    empty_segment : int or None
        Delay segment deliberately left empty, if any.
    event_dtype : numpy dtype
        Bool or float32 nonzero activity, including negative active values.
    homogeneous : bool
        Whether all edges share one weight.
    """
    n, k, event_density = 10_000, 8, 0.3
    rng = np.random.default_rng(2026)
    lengths = rng.integers(0, 6, size=n)
    lengths[:5] = [0, 1, 128, 129, 4097]
    ptr = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    columns = rng.integers(n, size=int(ptr[-1]), dtype=np.int32)
    weights = np.array([0.25], np.float32) if homogeneous else rng.uniform(-2, 2, columns.size).astype(np.float32)
    delays = rng.integers(k, size=columns.size, dtype=np.int32)
    if empty_segment is not None:
        delays[delays == empty_segment] = 0
    events = (rng.random((n, duration_steps)) < event_density).astype(event_dtype)
    if event_dtype == np.float32:
        events *= rng.choice(np.array([-2, 3], np.float32), size=events.shape)
    events = np.asfortranarray(events)

    data = delaycsr(
        (weights, columns, ptr),
        edge_delays=delays,
        shape=(len(ptr) - 1, n),
        delay_steps=tuple(range(k)),
    )
    arrays = _arrays(data)
    device_events = jnp.asarray(events)
    hybrid_ring = jnp.zeros((k, n), jnp.float32)
    work = _workspace(data)
    hybrid = jax.jit(cuda_step)
    numpy_ring = np.zeros((k, n), np.float64)
    for time in range(duration_steps):
        slot = time % k
        numpy_ring = numpy_project_step(weights, columns, ptr, delays, events[:, time], slot, numpy_ring)
        head, event = jnp.array([slot], jnp.int32), device_events[:, time]
        hybrid_ring, work = hybrid(
            **arrays, events=event, head=head, buffer=hybrid_ring, workspace=work)
        expected_output = numpy_ring[(slot + 1) % k].copy()
        message = f"step={time}, head={slot}, K={k}, T={duration_steps}"
        observed = np.asarray(hybrid_ring)
        np.testing.assert_allclose(observed[(slot + 1) % k], expected_output, rtol=2e-5, atol=2e-5,
                                   err_msg=message + ": next output")
        np.testing.assert_allclose(observed, numpy_ring, rtol=2e-5, atol=2e-5,
                                   err_msg=message + ": complete projected ring")
        expected_tasks = int(data.row_task_count[events[:, time] != 0].sum())
        np.testing.assert_array_equal(work.status, [expected_tasks, 0], err_msg=f"hybrid step={time}")


@GPU
@pytest.mark.parametrize("offset_dtype", [jnp.int32, jnp.int64])
@pytest.mark.parametrize("head", [0, 2])
def test_one_step_covers_all_segments_and_preserves_existing_ring(offset_dtype: Any, head: int) -> None:
    """Check empty segments, wrapping and existing ring contents."""
    data = _small()
    arrays = _arrays(data)
    arrays["delay_off"] = arrays["delay_off"].astype(offset_dtype)
    ring, event, position = jnp.ones((3, 4), jnp.float32), jnp.array([-2, 0], jnp.float32), jnp.array([head], jnp.int32)
    hybrid_ring, work = jax.jit(cuda_step)(
        **arrays, events=event, head=position, buffer=ring, workspace=_workspace(data, offset_dtype))
    expected = np.ones((3, 4), np.float32)
    expected[head, 2:] += [5, 2]
    expected[(head + 2) % 3, :2] += [3, 7]
    np.testing.assert_array_equal(work.status, [0, 0])
    np.testing.assert_array_equal(hybrid_ring, expected)


@GPU
@pytest.mark.parametrize("rows,columns", [(0, 0), (0, 7), (3, 0), (3, 7)])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_empty_graphs(rows: int, columns: int, homogeneous: bool) -> None:
    """Handle zero dimensions and empty segments while resetting hybrid status."""
    events = np.ones((rows, 4), bool)
    data = delaycsr(
        (np.ones(1 if homogeneous else 0, np.float32), np.array([], np.int32), np.zeros(rows + 1, np.int32)),
        edge_delays=np.array([], np.int32),
        shape=(rows, columns),
        delay_steps=(0, 3),
    )
    arrays = _arrays(data)
    current, ring = _run(data, events, arrays)
    np.testing.assert_array_equal(current, np.zeros((4, columns)))
    np.testing.assert_array_equal(ring, np.zeros((4, columns)))


@GPU
def test_single_slot_and_repeated_projection_sequences() -> None:
    """Accumulate zero-delay contributions and reuse prepared inputs for new runs."""
    weights, columns, ptr = np.array([2, -3], np.float32), np.array([1, 0], np.int32), np.array([0, 1, 2], np.int32)
    delays, events = np.zeros(2, int), np.array([[1, 0, 1], [0, 1, 1]], bool)
    data = delaycsr((weights, columns, ptr), edge_delays=delays, shape=(len(ptr) - 1, 2))
    arrays = _arrays(data)
    for _ in range(2):
        current, ring = _run(data, events, arrays)
        np.testing.assert_array_equal(current, [[0, 2], [-3, 2], [-6, 4]])
        np.testing.assert_array_equal(ring, [[-6, 4]])


@GPU
def test_hybrid_concurrent_rows_have_complete_task_coverage() -> None:
    """Check task reservations across multiple extract blocks without assuming order."""
    rows, width = 259, 135
    data = delaycsr(
        (np.ones(1, np.float32), np.tile(np.arange(width, dtype=np.int32), rows), np.arange(rows + 1, dtype=np.int32) * width),
        edge_delays=np.tile(np.repeat([0, 2, 5], 45), rows),
        shape=(rows, width),
    )
    ring, work = jax.jit(cuda_step)(
        **_arrays(data), events=jnp.ones(rows, bool), head=jnp.zeros(1, jnp.int32),
        buffer=jnp.zeros((6, width), jnp.float32), workspace=_workspace(data))
    np.testing.assert_array_equal(work.status, [rows * 3, 0])
    tasks = sorted(zip(*(np.asarray(a).tolist() for a in work[:3])))
    assert tasks == [(row * width + k * 45, row * width + (k + 1) * 45, delay * width)
                     for row in range(rows) for k, delay in enumerate((0, 2, 5))]
    expected = np.zeros((6, width), np.float32)
    expected[0, :45], expected[2, 45:90], expected[5, 90:] = rows, rows, rows
    np.testing.assert_array_equal(ring, expected)


@GPU
@pytest.mark.parametrize("field", ["row_task_count", "delay_count"])
def test_hybrid_bad_metadata_is_reported_and_next_call_resets_status(field: str) -> None:
    """Reject unwritten long-row tasks and reset the error on the next step."""
    data = delaycsr(
        (np.ones(1, np.float32), np.arange(129, dtype=np.int32), np.array([0, 129], np.int32)),
        edge_delays=np.zeros(129, int),
        shape=(1, 129),
    )
    arrays = _arrays(data)
    arrays[field] = jnp.zeros(1, jnp.int32) if field == "row_task_count" else jnp.array([[130]], jnp.int32)
    project = jax.jit(cuda_step)
    ring, work = project(**arrays, events=jnp.ones(1, bool), head=jnp.zeros(1, jnp.int32),
                         buffer=jnp.ones((1, 129), jnp.float32), workspace=_workspace(data))
    assert int(work.status[1]) == 3
    np.testing.assert_array_equal(ring, 1)
    _, work = project(**_arrays(data), events=jnp.zeros(1, bool), head=jnp.zeros(1, jnp.int32),
                      buffer=ring, workspace=work)
    np.testing.assert_array_equal(work.status, [0, 0])


@GPU
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("ptr_dtype", INDEX_DTYPES, ids=dtype_name)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES, ids=dtype_name)
@pytest.mark.parametrize("event_dtype", EVENT_DTYPES, ids=dtype_name)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES, ids=dtype_name)
def test_project_and_read_next_dtype_matrix(value_dtype, event_dtype, index_dtype, ptr_dtype, homogeneous) -> None:
    """Deliver and read next for a few steps; compare each arrival with a float64 reference."""
    data, (weights, indices, indptr, delays, events) = _dtype_problem(
        0, value_dtype, index_dtype, ptr_dtype, homogeneous, event_dtype=event_dtype)
    assert events.dtype == jnp.dtype(event_dtype)
    arrays = _arrays(data)
    ring = jnp.zeros((SLOTS, COLUMNS), value_dtype)
    work = _workspace(data)
    expected_ring = np.zeros((SLOTS, COLUMNS))
    for step in range(events.shape[1]):
        head = step % SLOTS
        column = jnp.asarray(events[:, step])  # One event vector per call.
        ring, work = cuda_step(**arrays, events=column, head=jnp.array([head], jnp.int32),
                               buffer=ring, workspace=work)
        assert ring.dtype == jnp.dtype(value_dtype)
        assert np.asarray(work.status)[1] == 0
        expected_ring = numpy_project_step(weights, indices, indptr, delays, events[:, step], head, expected_ring)
        current = ring[(head + 1) % SLOTS]
        assert current.dtype == jnp.dtype(value_dtype)
        assert_close(current, expected_ring[(head + 1) % SLOTS], value_dtype, SUM_SCALE)
        assert_close(ring, expected_ring, value_dtype, SUM_SCALE)


@GPU
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES, ids=dtype_name)
def test_inactive_rows_ignore_nonfinite_weights(value_dtype) -> None:
    """NaN/Inf weights on silent rows never reach the ring; on active rows they do."""
    data, (weights, indices, indptr, delays, events) = _dtype_problem(
        1, value_dtype, np.int32, np.int64, steps=1, event_dtype=jnp.float32)
    active = active_mask(events[:, 0])
    rows = np.repeat(np.arange(active.size), np.diff(data.indptr.astype(np.int64)))
    poisoned = np.asarray(data.weights).copy()
    poisoned[~active[rows]] = np.asarray(np.nan, poisoned.dtype)
    arrays = _arrays(data)
    arrays["weights"] = jnp.asarray(poisoned)
    ring, _ = cuda_step(**arrays, events=jnp.asarray(events[:, 0]), head=jnp.array([0], jnp.int32),
                        buffer=jnp.zeros((SLOTS, COLUMNS), value_dtype), workspace=_workspace(data))
    expected = numpy_project_step(weights, indices, indptr, delays, events[:, 0], 0, np.zeros((SLOTS, COLUMNS)))
    assert_close(ring, expected, value_dtype, SUM_SCALE)

    poisoned = np.asarray(data.weights).copy()
    first_active = int(np.flatnonzero(active[rows])[0])
    poisoned[first_active] = np.asarray(np.inf, poisoned.dtype)
    arrays["weights"] = jnp.asarray(poisoned)
    ring, _ = cuda_step(**arrays, events=jnp.asarray(events[:, 0]), head=jnp.array([0], jnp.int32),
                        buffer=jnp.zeros((SLOTS, COLUMNS), value_dtype), workspace=_workspace(data))
    assert not np.isfinite(np.asarray(ring).astype(np.float64)).all()


@GPU
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES, ids=dtype_name)
def test_silent_projection_preserves_every_slot(value_dtype) -> None:
    data, _ = _dtype_problem(2, value_dtype, np.int32, np.int32, steps=1)
    arrays = _arrays(data)
    ring = jnp.asarray(random_values(np.random.default_rng(2), (SLOTS, COLUMNS), value_dtype))
    expected = np.asarray(ring).copy()
    after, _ = cuda_step(**arrays, events=jnp.zeros(data.shape[0], jnp.bool_),
                         head=jnp.array([SLOTS - 1], jnp.int32), buffer=ring, workspace=_workspace(data))
    assert_same_bits(after, expected)


@GPU
def test_dtype_validation() -> None:
    data, _ = _dtype_problem(3, jnp.float32, np.int32, np.int32, steps=1)
    arrays = _arrays(data)
    head = jnp.array([0], jnp.int32)
    events = jnp.ones(data.shape[0], jnp.bool_)
    with pytest.raises(TypeError, match="weights and ring of one dtype"):
        cuda_step(**arrays, events=events, head=head, buffer=jnp.zeros((SLOTS, COLUMNS), jnp.float64),
                  workspace=_workspace(data))
    with pytest.raises(TypeError, match="Binary CUDA events"):
        cuda_step(**arrays, events=events.astype(jnp.complex64), head=head,
                  buffer=jnp.zeros((SLOTS, COLUMNS), jnp.float32), workspace=_workspace(data))


@GPU
@pytest.mark.parametrize("dtype", VALUE_DTYPES, ids=dtype_name)
@pytest.mark.parametrize("index_dtype,edge_dtype", [(np.int32, np.int32), (np.int32, np.int64),
                                                    (np.int64, np.int32), (np.int64, np.int64)])
@pytest.mark.parametrize("homo", [False, True])
@pytest.mark.parametrize("wide", [False, True])
def test_weighted_steps_and_task_values(dtype, index_dtype, edge_dtype, homo, wide):
    data, original, event_columns = _weighted_problem(dtype, index_dtype, edge_dtype, homo, wide)
    np.testing.assert_array_equal(event_columns[1], [-2, 0, 0.5])
    arrays = _arrays(data)
    work = data.workspace
    ring = jnp.full((data.L, data.shape[1]), 0.125, dtype)
    expected = np.asarray(ring).astype(np.float64)
    project = jax.jit(cuda_step, static_argnames=("weighted",))
    for step, head in enumerate([5, 0, 1]):
        events = event_columns[:, step]
        ring, work = project(**arrays, events=jnp.asarray(events), head=jnp.array([head], jnp.int32),
                             buffer=ring, workspace=work, weighted=True)
        count = int(data.row_task_count[events != 0].sum())
        np.testing.assert_array_equal(work.status, [count, 0])
        assert work.task_value.dtype == jnp.dtype(dtype)
        begins, ends, offsets, values = (np.asarray(a)[:count] for a in work[:-1])
        for begin, end, offset, value in zip(begins, ends, offsets, values):
            row = int(np.searchsorted(data.indptr, begin, side="right") - 1)
            bounds = data.indptr[row] + np.cumsum(data.delay_count[row].astype(np.int64))
            segment = int(np.searchsorted(bounds, begin, side="right"))
            assert begin < end <= bounds[segment]
            assert end - begin <= 4096
            assert offset == ((head + data.delay_steps[segment]) % data.L) * data.shape[1]
            assert value == events[row]
        expected = numpy_project_step(*original, events, head, expected, weighted=True)
        assert_close(ring, expected, dtype, 5.0)
        assert_close(ring[(head + 1) % data.L], expected[(head + 1) % data.L], dtype, 5.0)


@GPU
@pytest.mark.parametrize("dtype", VALUE_DTYPES, ids=dtype_name)
def test_nonfinite_and_exact_queue_precision(dtype):
    lengths = np.array([1, 129, 4097, 1, 129, 129])
    ptr = np.r_[0, lengths.cumsum()].astype(np.int64)
    indices = np.arange(ptr[-1], dtype=np.int32)
    events = np.array([0, -0.0, -1.0 - 2**-40, np.inf, np.nan, 2**-40], dtype=jnp.dtype(dtype))
    weights = np.ones(indices.size, dtype=jnp.dtype(dtype))
    weights[:ptr[2]] = np.nan  # zero events must skip nonfinite weights
    delays = (indices % 2).astype(np.int32)
    data = delaycsr(
        (weights, indices, ptr),
        edge_delays=delays,
        shape=(len(ptr) - 1, indices.size),
        weighted=True,
    )
    arrays = _arrays(data)
    ring, work = cuda_step(**arrays, events=jnp.asarray(events), head=jnp.array([1], jnp.int32),
                           buffer=jnp.zeros((2, indices.size), dtype), workspace=data.workspace, weighted=True)
    expected = numpy_project_step(weights, indices, ptr, delays, events, 1, np.zeros((2, indices.size)), weighted=True)
    np.testing.assert_array_equal(np.asarray(ring).astype(np.float64), expected)
    count = int(data.row_task_count[events != 0].sum())
    np.testing.assert_array_equal(work.status, [count, 0])
    rows = np.searchsorted(data.indptr, np.asarray(work.task_begin)[:count], side="right") - 1
    np.testing.assert_array_equal(np.asarray(work.task_value)[:count].astype(np.float64),
                                  events[rows].astype(np.float64))


@GPU
@pytest.mark.parametrize("lengths", [[], [0, 0], [1, 128]])
def test_zero_capacity(lengths):
    ptr = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    cols = np.zeros(ptr[-1], np.int32)
    events = np.full((len(lengths), 1), -2, np.float32)
    data = delaycsr(
        (np.array([0.25], np.float32), cols, ptr),
        edge_delays=cols,
        shape=(len(ptr) - 1, 1),
        weighted=True,
    )
    arrays = _arrays(data)
    work = data.workspace
    assert work.task_value.size == 0
    ring, work = cuda_step(**arrays, events=jnp.asarray(events[:, 0]), head=jnp.array([0], jnp.int32),
                           buffer=jnp.ones((1, 1), jnp.float32), workspace=work, weighted=True)
    np.testing.assert_array_equal(work.status, [0, 0])
    np.testing.assert_array_equal(ring, [[1 - ptr[-1] * 0.5]])


@GPU
@pytest.mark.parametrize("error", ["capacity", "head", "counts"])
def test_weighted_status_prevents_invalid_task_reads(error):
    ptr = np.array([0, 129], np.int32)
    data = delaycsr(
        (np.array([1], np.float32), np.zeros(129, np.int32), ptr),
        edge_delays=np.zeros(129, np.int32),
        shape=(len(ptr) - 1, 1),
        weighted=True,
    )
    arrays = _arrays(data)
    work = data.workspace
    head = jnp.array([0], jnp.int32)
    if error == "capacity":
        work = WeightedHybridWorkspace(*(a[:0] for a in work[:-1]), work.status)
    elif error == "head":
        head = jnp.array([1], jnp.int32)
    else:
        arrays["delay_count"] = jnp.array([[130]], jnp.int32)
    if error == "head":
        with pytest.raises(RuntimeError, match="head"):
            jax.block_until_ready(cuda_step(
                **arrays, events=jnp.ones(1, jnp.float32), head=head,
                buffer=jnp.ones((1, 1), jnp.float32), workspace=work, weighted=True))
        # Recover through an explicit compiled step, as in the head/scan tests.
        ring, work = jax.jit(cuda_step, static_argnames=("weighted",))(
            **arrays, events=jnp.ones(1, jnp.float32), head=jnp.zeros(1, jnp.int32),
            buffer=jnp.ones((1, 1), jnp.float32), workspace=work, weighted=True)
        np.testing.assert_array_equal(ring, [[130]])
        np.testing.assert_array_equal(work.status, [1, 0])
        jax.effects_barrier()
        return
    ring, work = cuda_step(**arrays, events=jnp.ones(1, jnp.float32), head=head,
                           buffer=jnp.ones((1, 1), jnp.float32), workspace=work, weighted=True)
    assert int(work.status[1]) == {"capacity": 1, "head": 2, "counts": 3}[error]
    np.testing.assert_array_equal(ring, [[1]])


@GPU
def test_weighted_validation_and_preparation():
    data, _, event_columns = _weighted_problem(jnp.float32)
    arrays = _arrays(data)
    events, head = jnp.asarray(event_columns[:, 0]), jnp.array([0], jnp.int32)
    ring, work = jnp.zeros((6, data.shape[1]), jnp.float32), data.workspace
    with pytest.raises(TypeError, match="events"):
        cuda_step(**arrays, events=events.astype(jnp.complex64), head=head,
                  buffer=ring, workspace=work, weighted=True)
    with pytest.raises(ValueError, match="workspace"):
        cuda_step(**arrays, events=events, head=head, buffer=ring,
                  workspace=HybridWorkspace(work.task_begin, work.task_end, work.task_out_off, work.status),
                  weighted=True)
    for task_value in (jnp.zeros(data.capacity, jnp.float64), jnp.zeros(data.capacity + 1, jnp.float32)):
        with pytest.raises((TypeError, ValueError), match="task_value"):
            cuda_step(**arrays, events=events, head=head, buffer=ring,
                      workspace=work._replace(task_value=task_value), weighted=True)


def test_weighted_preparation_rejects_complex_events():
    data, _, event_columns = _weighted_problem(jnp.float32)
    with pytest.raises(TypeError, match="events"):
        data.forward(jnp.asarray(event_columns[:, 0], jnp.complex64), 0)


# Construction, ownership and weight replacement
# ----------------------------------------------

def _delaycsr_init_case(**changes: Any) -> Any:
    values: dict[str, Any] = dict(
        weights=np.array([2, 3, 5, 7], np.float32),
        indices=np.array([3, 0, 2, 2], np.int32),
        indptr=np.array([0, 4, 4], np.int64),
        edge_delays=np.array([2, 0, 2, 2]),
        columns=4, delay_steps=(0, 1, 2),
    )
    values.update(changes)
    weights, indices, indptr = (values.pop(name) for name in ("weights", "indices", "indptr"))
    columns = values.pop("columns")
    return delaycsr((weights, indices, indptr), shape=(len(indptr) - 1, columns), **values)


def _delaycsr_init_matrix(*, homogeneous=False, weighted=False, backend=None):
    return delaycsr(
        (np.array([2] if homogeneous else [2, 3, 5, 7], np.float32),
         np.array([2, 0, 1, 0], np.int32), np.array([0, 3, 4], np.int32)),
        edge_delays=np.array([2, 1, 1, 2], np.int32), shape=(2, 3),
        buffer_slots=4, ring=jnp.arange(12, dtype=jnp.float32).reshape(4, 3), head=3,
        weighted=weighted, backend=backend,
    )


def _delaycsr_init_assert_same_structure_and_state(original, updated):
    assert isinstance(updated, delaycsr)
    assert updated is not original
    for name in ("indices", "indptr", "delay_count", "delay_off", "row_task_count",
                 "ring", "head", "workspace"):
        assert getattr(updated, name) is getattr(original, name), name
    for name in ("shape", "backend", "delay_steps", "capacity", "L", "weighted"):
        assert getattr(updated, name) == getattr(original, name), name


def _delaycsr_init_source(backend=None):
    return CSR((jnp.array([2, -1, .5, 3], jnp.float32), jnp.array([1, 0, 1, 0], jnp.int32),
                jnp.array([0, 3, 4], jnp.int64)), shape=(2, 2), backend=backend,
               indptr_dtype=jnp.int64)


def _delaycsr_init_prepared_arguments(matrix: delaycsr) -> dict[str, Any]:
    return dict(
        data=(matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape,
        delay_steps=matrix.delay_steps, delay_count=matrix.delay_count,
        delay_off=matrix.delay_off, row_task_count=matrix.row_task_count,
        capacity=matrix.capacity, buffer_slots=matrix.L, weighted=matrix.weighted,
        ring=matrix.ring, head=matrix.head, workspace=matrix.workspace,
        backend=matrix.backend,
    )


def _delaycsr_init_aligned_host_array(values, dtype):
    values = np.asarray(values, dtype)
    storage = np.empty(values.nbytes + 64, np.uint8)
    start = (-storage.ctypes.data) % 64
    result = storage[start:start + values.nbytes].view(dtype).reshape(values.shape)
    result[...] = values
    return result


def test_grouping_preserves_weight_alignment_and_duplicate_order() -> None:
    prepared = _delaycsr_init_case()
    np.testing.assert_array_equal(prepared.indices, [0, 2, 2, 3])
    np.testing.assert_array_equal(prepared.weights, [3, 5, 7, 2])
    np.testing.assert_array_equal(prepared.delay_count, [[1, 0, 3], [0, 0, 0]])
    np.testing.assert_array_equal(prepared.delay_off, [0, 4, 8])
    assert prepared.delay_count.dtype == np.int64
    assert prepared.delay_off.dtype == np.int32
    assert isinstance(prepared, CSR)
    assert prepared.indptr.dtype == jnp.int64
    np.testing.assert_array_equal(prepared.data, prepared.weights)
    assert prepared.capacity == 0 and prepared.L == 3 and int(prepared.head) == 0


def test_homogeneous_weights_and_empty_segments() -> None:
    prepared = _delaycsr_init_case(weights=np.array([4], np.float32))
    assert prepared.weights.shape == (1,)
    assert prepared.delay_steps == (0, 1, 2)
    assert prepared.capacity == 0


def test_from_prepared_bypasses_initialization_and_reuses_device_layout(monkeypatch) -> None:
    original = _delaycsr_init_matrix()
    arguments = _delaycsr_init_prepared_arguments(original)

    def unexpected_preprocessing(*args, **kwargs):
        raise AssertionError("prepared input must not run initialization or preprocessing")

    for owner, name in ((CSR, "__init__"), (delaycsr, "__init__"),
                        (delaycsr, "_sort_by_delay"), (delaycsr, "_build_delay_metadata")):
        monkeypatch.setattr(owner, name, unexpected_preprocessing)
    updated = delaycsr.from_prepared(**arguments)
    _delaycsr_init_assert_same_structure_and_state(original, updated)
    assert updated.data is original.data
    assert updated.config is not None
    assert updated._cache.mirror is None


def test_from_prepared_static_does_not_read_device_layout(monkeypatch) -> None:
    original = _delaycsr_init_matrix()
    arguments = _delaycsr_init_prepared_arguments(original)
    asarray = np.asarray

    def guarded_asarray(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            raise AssertionError("static construction must not copy device layout to the host")
        return asarray(value, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(module.np, "asarray", guarded_asarray)
        updated = delaycsr.from_prepared(**arguments)
    _delaycsr_init_assert_same_structure_and_state(original, updated)


@pytest.mark.parametrize("source_form", ["csr", "tuple", "separate", "scalar"])
def test_from_prepared_source_forms_and_default_state(source_form) -> None:
    original = _delaycsr_init_matrix(homogeneous=True, backend="jax_raw")
    arguments = _delaycsr_init_prepared_arguments(original)
    for name in ("ring", "head", "workspace", "buffer_slots", "backend"):
        arguments.pop(name)
    if source_form == "csr":
        arguments["data"] = original
        arguments.pop("shape")
    elif source_form == "separate":
        arguments["data"], arguments["indices"], arguments["indptr"] = arguments["data"]
    elif source_form == "scalar":
        arguments["data"] = (np.float32(2), original.indices, original.indptr)
    updated = delaycsr.from_prepared(**arguments, validate="full")
    assert updated.shape == original.shape
    assert updated.L == 3 and int(updated.head) == 0
    assert updated.data.shape == (1,)
    assert updated.workspace.task_begin.shape == (0,)
    assert updated.backend == ("jax_raw" if source_form == "csr" else None)
    np.testing.assert_array_equal(updated.ring, np.zeros((3, 3), np.float32))


@pytest.mark.parametrize("rows,columns", [(0, 0), (0, 3), (4, 0), (4, 3)])
def test_from_prepared_empty_layout(rows, columns) -> None:
    original = delaycsr(
        (np.array([], np.float32), np.array([], np.int64), np.zeros(rows + 1, np.int64)),
        edge_delays=np.array([], np.int64), shape=(rows, columns), backend="jax_raw",
    )
    updated = delaycsr.from_prepared(**_delaycsr_init_prepared_arguments(original), validate="full")
    actual = updated.csr_delay_deliver_layout(updated.ring, jnp.zeros(rows, jnp.bool_), 0)
    assert actual.shape == (1, columns)
    assert updated.capacity == 0
    assert updated.indices.dtype == jnp.int64
    assert updated.indptr.dtype == jnp.int64


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("backend", BACKENDS)
def test_from_prepared_long_rows_and_overallocated_workspace(weighted, backend) -> None:
    lengths = [128, 129, 8293]
    pointers = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    delays = np.r_[np.zeros(128 + 129 + 4096), np.full(4097, 2), np.full(100, 5)].astype(np.int32)
    original = delaycsr(
        (np.ones(pointers[-1], np.float32), np.zeros(pointers[-1], np.int32), pointers),
        edge_delays=delays, shape=(3, 1), weighted=weighted, backend=backend,
    )
    arguments = _delaycsr_init_prepared_arguments(original)
    arguments.update(capacity=7, workspace=None)
    updated = delaycsr.from_prepared(**arguments, validate="full")
    np.testing.assert_array_equal(updated.row_task_count, [0, 1, 4])
    assert updated.workspace.task_begin.shape == (7,)
    if weighted:
        assert updated.workspace.task_value.dtype == updated.ring.dtype
    events = jnp.array([1, 0, -2], jnp.float32)
    expected = original.csr_delay_deliver_layout(original.ring, events, 5, weighted=weighted)
    actual = updated.csr_delay_deliver_layout(updated.ring, events, 5, weighted=weighted)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name,value,exception", [
    ("shape", None, ValueError),
    ("shape", (True, 3), ValueError),
    ("backend", "numba", ValueError),
    ("weighted", 1, TypeError),
    ("validate", "none", ValueError),
    ("capacity", -1, ValueError),
    ("capacity", True, ValueError),
    ("capacity", 2**31, ValueError),
    ("buffer_slots", 2, ValueError),
    ("buffer_slots", 4.0, ValueError),
    ("delay_steps", (), ValueError),
    ("delay_steps", (1, 1), ValueError),
    ("weights", np.ones(2, np.float32), ValueError),
    ("weights", np.ones(4, np.int32), TypeError),
    ("indices", np.zeros((2, 2), np.int32), ValueError),
    ("indices", np.zeros(4, np.uint32), TypeError),
    ("indptr", np.zeros(4, np.int32), ValueError),
    ("indptr", np.array([0, 3, 4], np.uint32), TypeError),
    ("delay_count", np.zeros((2, 1), np.int32), ValueError),
    ("delay_count", np.zeros((2, 2), np.int64), TypeError),
    ("delay_off", np.zeros(1, np.int32), ValueError),
    ("delay_off", np.zeros(2, np.float32), TypeError),
    ("row_task_count", np.zeros(3, np.int32), ValueError),
    ("row_task_count", np.zeros(2, np.int64), TypeError),
    ("ring", np.zeros((3, 3), np.float32), ValueError),
    ("ring", np.zeros((4, 3), np.int32), TypeError),
    ("workspace", (), TypeError),
])
def test_from_prepared_rejects_invalid_static_contract(name, value, exception) -> None:
    arguments = _delaycsr_init_prepared_arguments(_delaycsr_init_matrix())
    if name in ("weights", "indices", "indptr"):
        position = ("weights", "indices", "indptr").index(name)
        parts = list(arguments["data"])
        parts[position] = value
        arguments["data"] = tuple(parts)
    else:
        arguments[name] = value
    with pytest.raises(exception):
        delaycsr.from_prepared(**arguments)


@pytest.mark.parametrize("name,value,match", [
    ("indices", [1, 0, 2, 0], "sorted"),
    ("indices", [-1, 1, 2, 0], "bounds"),
    ("indices", [0, 1, 3, 0], "bounds"),
    ("indptr", [1, 3, 4], "indptr"),
    ("indptr", [0, 3, 3], "indptr"),
    ("indptr", [0, 5, 4], "indptr"),
    ("indptr", [0, -1, 4], "indptr"),
    ("delay_count", [[-1, 4], [0, 1]], "delay_count"),
    ("delay_count", [[3, 1], [0, 1]], "delay_count"),
    ("delay_count", [[1, 1], [0, 1]], "delay_count"),
    ("delay_off", [0, 6], "delay_off"),
    ("row_task_count", [1, 0], "row_task_count"),
    ("row_task_count", [-1, 0], "row_task_count"),
])
def test_from_prepared_full_checks_layout_values(name, value, match) -> None:
    arguments = _delaycsr_init_prepared_arguments(_delaycsr_init_matrix())
    value = np.array(value, np.int32)
    if name in ("indices", "indptr"):
        parts = list(arguments["data"])
        parts[("weights", "indices", "indptr").index(name)] = value
        arguments["data"] = tuple(parts)
    else:
        arguments[name] = value
    with pytest.raises(ValueError, match=match):
        delaycsr.from_prepared(**arguments, validate="full")


def test_from_prepared_full_rejects_insufficient_worst_case_capacity() -> None:
    original = delaycsr(
        (np.ones(129, np.float32), np.zeros(129, np.int32), np.array([0, 129], np.int32)),
        shape=(1, 1), edge_delays=np.r_[np.zeros(64, np.int32), np.ones(65, np.int32)],
    )
    arguments = _delaycsr_init_prepared_arguments(original)
    arguments.update(capacity=1, workspace=None)
    with pytest.raises(ValueError, match="capacity"):
        delaycsr.from_prepared(**arguments, validate="full")


def test_from_prepared_static_retains_runtime_metadata_checks() -> None:
    arguments = _delaycsr_init_prepared_arguments(_delaycsr_init_matrix(backend="jax_raw"))
    arguments["delay_count"] = jnp.zeros((2, 2), jnp.int32)
    updated = delaycsr.from_prepared(**arguments)
    with pytest.raises((ValueError, RuntimeError), match="inconsistent delay/task metadata"):
        jax.block_until_ready(
            updated.csr_delay_deliver_layout(updated.ring, jnp.zeros(2, jnp.bool_), 0))


@pytest.mark.parametrize("field,value,exception", [
    ("task_begin", np.zeros(1, np.int32), ValueError),
    ("task_end", np.zeros(0, np.int64), TypeError),
    ("task_out_off", np.zeros(0, np.int64), TypeError),
    ("status", np.zeros(1, np.int32), ValueError),
    ("status", np.zeros(2, np.int64), TypeError),
])
def test_from_prepared_checks_workspace_shape_and_dtype(field, value, exception) -> None:
    arguments = _delaycsr_init_prepared_arguments(_delaycsr_init_matrix())
    arguments["workspace"] = arguments["workspace"]._replace(**{field: value})
    with pytest.raises(exception, match="workspace"):
        delaycsr.from_prepared(**arguments)


def test_from_prepared_owns_numpy_layout_ring_and_workspace() -> None:
    original = _delaycsr_init_matrix()
    arguments = _delaycsr_init_prepared_arguments(original)
    arguments["data"] = tuple(np.array(value, copy=True) for value in arguments["data"])
    for name in ("delay_count", "delay_off", "row_task_count", "ring"):
        arguments[name] = np.array(arguments[name], copy=True)
    arguments["workspace"] = type(original.workspace)(
        *(np.array(value, copy=True) for value in original.workspace))
    updated = delaycsr.from_prepared(**arguments, validate="full")
    arrays = (*arguments["data"], *(arguments[name] for name in (
        "delay_count", "delay_off", "row_task_count", "ring")), *arguments["workspace"])
    for value in arrays:
        value.fill(99)
    for name in ("data", "indices", "indptr", "delay_count", "delay_off", "row_task_count", "ring"):
        np.testing.assert_array_equal(getattr(updated, name), getattr(original, name))
    for actual, expected in zip(updated.workspace, original.workspace):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("weighted", [False, True])
def test_from_prepared_preserves_units_and_independent_ring_precision(weighted) -> None:
    original = delaycsr(
        (np.array([2, 3, 5, 7], np.float32) * u.mV,
         np.array([2, 0, 1, 0], np.int32), np.array([0, 3, 4], np.int32)),
        shape=(2, 3), edge_delays=np.array([2, 1, 1, 2], np.int32),
        buffer_slots=4, ring=np.arange(12, dtype=np.float16).reshape(4, 3) * u.mV,
        weighted=weighted, backend="jax_raw",
    )
    updated = delaycsr.from_prepared(**_delaycsr_init_prepared_arguments(original), validate="full")
    assert updated.data.dtype == jnp.float32
    assert updated.ring.dtype == jnp.float16
    assert u.get_unit(updated.data) == u.mV
    assert u.get_unit(updated.ring) == u.mV
    if weighted:
        assert updated.workspace.task_value.dtype == jnp.float16
    events = jnp.array([1, -2], jnp.float32)
    expected = original.csr_delay_deliver_layout(original.ring, events, 2, weighted=weighted)
    actual = updated.csr_delay_deliver_layout(updated.ring, events, 2, weighted=weighted)
    assert u.get_unit(actual) == u.mV
    np.testing.assert_array_equal(u.get_mantissa(actual), u.get_mantissa(expected))


def test_from_prepared_checks_weighted_task_value_precision() -> None:
    arguments = _delaycsr_init_prepared_arguments(_delaycsr_init_matrix(weighted=True))
    arguments["ring"] = arguments["ring"].astype(jnp.float16)
    with pytest.raises(TypeError, match="workspace dtypes"):
        delaycsr.from_prepared(**arguments)
    arguments["workspace"] = None
    updated = delaycsr.from_prepared(**arguments)
    assert updated.workspace.task_value.dtype == jnp.float16
    assert updated.data.dtype == jnp.float32


@pytest.mark.parametrize("name", ["weights", "indices", "indptr", "delay_count", "delay_off", "ring"])
def test_from_prepared_requires_x64_for_explicit_wide_inputs(name) -> None:
    arguments = _delaycsr_init_prepared_arguments(_delaycsr_init_matrix())
    dtype = np.float64 if name in ("weights", "ring") else np.int64
    if name in ("weights", "indices", "indptr"):
        parts = list(arguments["data"])
        position = ("weights", "indices", "indptr").index(name)
        parts[position] = np.asarray(parts[position]).astype(dtype)
        arguments["data"] = tuple(parts)
    else:
        arguments[name] = np.asarray(arguments[name]).astype(dtype)
    previous = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", False)
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            delaycsr.from_prepared(**arguments)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_from_prepared_large_ring_requires_wide_offsets(monkeypatch) -> None:
    arguments = _delaycsr_init_prepared_arguments(_delaycsr_init_matrix())
    arguments.update(buffer_slots=8, ring=None, workspace=None)
    monkeypatch.setattr(module, "_I32_MAX", 15)
    with pytest.raises(ValueError, match="int64 delay_off"):
        delaycsr.from_prepared(**arguments)
    arguments["delay_off"] = arguments["delay_off"].astype(jnp.int64)
    updated = delaycsr.from_prepared(**arguments, validate="full")
    assert updated.indices.dtype == jnp.int32
    assert updated.delay_off.dtype == jnp.int64
    assert updated.workspace.task_out_off.dtype == jnp.int64
    assert updated.ring.shape == (8, 3)


def test_from_prepared_checks_byte_size_before_allocating_ring(monkeypatch) -> None:
    arguments = _delaycsr_init_prepared_arguments(_delaycsr_init_matrix())
    arguments.update(ring=None, workspace=None)
    monkeypatch.setattr(module, "_I64_MAX", 31)

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("invalid ring byte size must be rejected before allocation")

    monkeypatch.setattr(module.jnp, "zeros", unexpected_allocation)
    with pytest.raises(ValueError, match="byte size"):
        delaycsr.from_prepared(**arguments)


@pytest.mark.parametrize("changes,exception,match", [
    ({"indices": np.array([0, 1, 2, 0], np.int32)}, ValueError, "already supplies"),
    ({"shape": (2, 4)}, ValueError, "shape disagrees"),
    ({"data": []}, TypeError, "supply CSR"),
])
def test_from_prepared_rejects_inconsistent_source(changes, exception, match) -> None:
    original = _delaycsr_init_matrix()
    arguments = _delaycsr_init_prepared_arguments(original)
    arguments["data"] = original
    arguments.update(changes)
    with pytest.raises(exception, match=match):
        delaycsr.from_prepared(**arguments)


def test_from_prepared_requires_static_delay_steps() -> None:
    arguments = _delaycsr_init_prepared_arguments(_delaycsr_init_matrix())
    arguments["delay_steps"] = jnp.array([1, 2], jnp.int32)
    with pytest.raises(TypeError, match="static host integers"):
        delaycsr.from_prepared(**arguments)


def test_from_prepared_full_requires_concrete_layout_under_jit() -> None:
    original = _delaycsr_init_matrix()

    @jax.jit
    def reconstruct(matrix):
        return delaycsr.from_prepared(**_delaycsr_init_prepared_arguments(matrix), validate="full")

    with pytest.raises(ValueError, match="requires concrete layout"):
        reconstruct(original)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_from_prepared_jit_scan_and_differentiation(weighted, homogeneous) -> None:
    original = _delaycsr_init_matrix(homogeneous=homogeneous, weighted=weighted, backend="jax_raw")
    events = jnp.array([[1, 0], [0, -2], [.5, 1]], jnp.float32)
    times = jnp.array([3, 4, 5], jnp.int32)

    def scan(matrix, xs):
        def step(carry, xt):
            spike = carry.forward(*xt)
            return carry, spike
        return jax.lax.scan(step, matrix, (xs, times))

    def run(matrix, weights, xs):
        arguments = _delaycsr_init_prepared_arguments(matrix)
        arguments["data"] = (weights, matrix.indices, matrix.indptr)
        return scan(delaycsr.from_prepared(**arguments), xs)

    def reference(matrix, weights, xs):
        return scan(matrix.with_data(weights), xs)

    actual, actual_spikes = jax.jit(run)(original, original.data, events)
    expected, expected_spikes = jax.jit(reference)(original, original.data, events)
    np.testing.assert_array_equal(actual.ring, expected.ring)
    np.testing.assert_array_equal(actual_spikes, expected_spikes)
    np.testing.assert_array_equal(actual.head, expected.head)

    def loss(fn, matrix, weights, xs):
        state, spikes = fn(matrix, weights, xs)
        return (state.ring ** 2).sum() + (spikes ** 2).sum()

    for fn in (run, reference):
        gradients = jax.jit(jax.grad(lambda matrix, weights, xs: loss(fn, matrix, weights, xs),
                                     argnums=(1, 2)))(original, original.data, events)
        if fn is run:
            actual_gradients = gradients
        else:
            for actual_gradient, expected_gradient in zip(actual_gradients, gradients):
                np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-6)
    actual_jvp = jax.jvp(lambda weights: loss(run, original, weights, events),
                         (original.data,), (jnp.ones_like(original.data),))
    expected_jvp = jax.jvp(lambda weights: loss(reference, original, weights, events),
                           (original.data,), (jnp.ones_like(original.data),))
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("scalar", [2, np.array(2, np.int32), np.array([2], np.int32),
                                     jnp.array(2, jnp.int32)])
def test_scalar_delay_matches_per_edge_delays(scalar: Any) -> None:
    scalar_layout = _delaycsr_init_case(edge_delays=scalar, delay_steps=None, backend="jax_raw")
    per_edge_layout = _delaycsr_init_case(edge_delays=np.full(4, 2, np.int32), delay_steps=None,
                            backend="jax_raw")
    assert scalar_layout.delay_steps == per_edge_layout.delay_steps == (2,)
    assert scalar_layout.L == per_edge_layout.L == 3
    for name in ("data", "indices", "delay_count", "delay_off"):
        np.testing.assert_array_equal(getattr(scalar_layout, name),
                                      getattr(per_edge_layout, name))
    events = jnp.array([True, False])
    actual = scalar_layout.csr_delay_deliver_layout(scalar_layout.ring, events, 0)
    expected = per_edge_layout.csr_delay_deliver_layout(per_edge_layout.ring, events, 0)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[2], [3, 0, 12, 2])


def test_csr_delay_layout_accepts_scalar_delay() -> None:
    source = _delaycsr_init_source("jax_raw")
    scalar = source.delay_layout(1, n_slots=3)
    per_edge = source.delay_layout(np.ones(source.nse, np.int32), n_slots=3)
    np.testing.assert_array_equal(scalar.delay_count, per_edge.delay_count)
    np.testing.assert_array_equal(scalar.data, per_edge.data)
    np.testing.assert_array_equal(scalar.indices, per_edge.indices)


def test_zero_scalar_delay_delivers_into_current_slot() -> None:
    prepared = _delaycsr_init_case(edge_delays=0, delay_steps=None, backend="jax_raw")
    assert prepared.delay_steps == (0,)
    assert prepared.L == 1
    ring = prepared.csr_delay_deliver_layout(prepared.ring, jnp.array([True, False]), 0)
    np.testing.assert_array_equal(ring, [[3, 0, 12, 2]])


def test_whole_row_classification_and_segment_chunk_boundaries() -> None:
    lengths = [128, 129, 8293]
    ptr = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    delays = np.r_[np.zeros(128 + 129 + 4096), np.full(4097, 2), np.full(100, 5)].astype(np.int32)
    prepared = delaycsr((np.ones(ptr[-1], np.float32), np.zeros(ptr[-1], np.int32), ptr),
                        edge_delays=delays, shape=(3, 1))
    assert prepared.delay_steps == (0, 2, 5)
    np.testing.assert_array_equal(prepared.row_task_count, [0, 1, 4])
    assert prepared.capacity == 5


@pytest.mark.parametrize("rows,columns", [(0, 0), (0, 3), (4, 0), (4, 3)])
def test_empty_inputs_keep_explicit_shapes(rows: int, columns: int) -> None:
    prepared = delaycsr((np.array([], np.float32), np.array([], np.int64), np.zeros(rows + 1, np.int64)),
                        edge_delays=np.array([], np.int64), shape=(rows, columns))
    assert prepared.delay_steps == (0,)
    assert prepared.delay_count.shape == (rows, 1)
    assert prepared.capacity == 0


def test_scalar_delay_on_empty_graph_keeps_empty_default_metadata() -> None:
    prepared = delaycsr(
        (np.array([], np.float32), np.array([], np.int32), np.array([0, 0], np.int32)),
        edge_delays=2, shape=(1, 1), backend="jax_raw",
    )
    assert prepared.delay_steps == (0,)
    assert prepared.L == 1
    np.testing.assert_array_equal(prepared.delay_count, [[0]])


def test_large_ring_offsets_are_independent_of_column_dtype(monkeypatch) -> None:
    monkeypatch.setattr(module, "_I32_MAX", 15)
    prepared = _delaycsr_init_case(delay_steps=(0, 1, 2), buffer_slots=8)
    assert prepared.delay_off.dtype == np.int64
    assert prepared.indices.dtype == np.int32


@pytest.mark.parametrize("changes,exception", [
    ({"weights": np.ones(4, np.int32)}, TypeError),
    ({"indices": np.zeros(4, np.uint32)}, TypeError),
    ({"indptr": np.zeros(3, np.float32)}, TypeError),
    ({"weights": np.ones((1, 4), np.float32)}, ValueError),
    ({"indptr": np.array([], np.int32)}, ValueError),
    ({"weights": np.ones(2, np.float32)}, ValueError),
    ({"indptr": np.array([1, 4, 4], np.int32)}, ValueError),
    ({"indptr": np.array([0, 5, 4], np.int32)}, ValueError),
    ({"indptr": np.array([0, -1, 4], np.int32)}, ValueError),
    ({"columns": -1}, ValueError),
    ({"columns": True}, ValueError),
    ({"columns": 2**31}, ValueError),
    ({"indices": np.array([0, 1, -1, 2], np.int32)}, ValueError),
    ({"indices": np.array([0, 1, 4, 2], np.int32)}, ValueError),
    ({"edge_delays": np.zeros(3, np.int32)}, ValueError),
    ({"edge_delays": -1}, ValueError),
    ({"edge_delays": 1.0}, ValueError),
    ({"edge_delays": True}, ValueError),
    ({"edge_delays": 2**31}, ValueError),
    ({"edge_delays": np.array([0, -1, 2, 2])}, ValueError),
    ({"edge_delays": np.zeros(4, np.float32)}, ValueError),
    ({"edge_delays": np.full(4, 2**31, np.uint64)}, ValueError),
    ({"delay_steps": (0, 1)}, ValueError),
    ({"buffer_slots": 2}, ValueError),
    ({"buffer_slots": 3.0}, ValueError),
])
def test_invalid_data_is_rejected(changes: dict[str, Any], exception: type[Exception]) -> None:
    with pytest.raises(exception):
        _delaycsr_init_case(**changes)


@pytest.mark.parametrize("values", [[], [0, 0], [2, 0], [-1, 2], [0.0, 1.0], [[0]], [2**31]])
def test_invalid_delay_value_sets(values: Any) -> None:
    with pytest.raises(ValueError):
        _delaycsr_init_case(delay_steps=values)


def test_byte_size_and_task_counter_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_I64_MAX", 31)
    zeros = jnp.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if shape == (3, 4):
            raise AssertionError("oversized ring must be rejected before allocation")
        return zeros(shape, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(module.jnp, "zeros", guarded_zeros)
        with pytest.raises(ValueError, match="byte size"):
            _delaycsr_init_case()
    monkeypatch.setattr(module, "_I64_MAX", 2**63 - 1)
    monkeypatch.setattr(module, "_I32_MAX", 4)
    monkeypatch.setattr(module, "TPR_THRESHOLD", 0)
    monkeypatch.setattr(module, "TASK_NNZ", 1)
    with pytest.raises(ValueError, match="capacity"):
        delaycsr((np.ones(5, np.float32), np.zeros(5, np.int32), np.array([0, 5], np.int64)),
                 edge_delays=np.zeros(5, np.int32), shape=(1, 1))


@pytest.mark.parametrize("changes", [
    {"edge_delays": np.array([2, 0, 2, 2])},  # decreasing delay
    {"edge_delays": np.array([0, 0, 2, 2])},  # decreasing column within delay 0
])
def test_sorted_flag_checks_actual_row_order(changes):
    with pytest.raises(ValueError, match="sorted_by_delay=True"):
        _delaycsr_init_case(sorted_by_delay=True, **changes)


def test_sorted_input_skips_sorting_and_keeps_weight_order(monkeypatch):
    def unexpected_sort(*args):
        raise AssertionError("sorted input must not be sorted again")

    monkeypatch.setattr(delaycsr, "_sort_by_delay", unexpected_sort)
    w = _delaycsr_init_case(weights=np.array([3, 5, 7, 2], np.float32), indices=np.array([0, 2, 2, 3], np.int32),
              edge_delays=np.array([0, 2, 2, 2]), sorted_by_delay=True)
    np.testing.assert_array_equal(w.data, [3, 5, 7, 2])
    np.testing.assert_array_equal(w.delay_count, [[1, 0, 3], [0, 0, 0]])


def test_unsorted_input_sorts_each_row_once(monkeypatch):
    calls = []
    original = np.lexsort

    def record(keys):
        calls.append(len(keys[0]))
        return original(keys)

    monkeypatch.setattr(module.np, "lexsort", record)
    w = _delaycsr_init_case(sorted_by_delay=False)
    assert calls == [4, 0]
    np.testing.assert_array_equal(w.data, [3, 5, 7, 2])


def test_member_sort_overwrites_csr_and_invalidates_old_structure_cache():
    w = _delaycsr_init_case()
    # Structural CSR operations must not create a partially initialized delaycsr.
    with pytest.warns(UserWarning, match="plain CSR without delay"):
        assert type(w.build_weight_indices()) is CSR
    assert w._cache.mirror is not None
    w.data = jnp.array([2, 3, 5, 7], jnp.float32)
    w.indices = jnp.array([3, 0, 2, 2], jnp.int32)
    ring, head = w.ring, w.head
    w.sort_by_delay(np.array([2, 0, 2, 2]))
    np.testing.assert_array_equal(w.indices, [0, 2, 2, 3])
    np.testing.assert_array_equal(w.data, [3, 5, 7, 2])
    np.testing.assert_array_equal(w.delay_count, [[1, 0, 3], [0, 0, 0]])
    assert w._cache.mirror is None
    assert w.ring is ring and w.head is head


def test_member_sort_accepts_scalar_delay() -> None:
    prepared = _delaycsr_init_case()
    prepared.sort_by_delay(2)
    np.testing.assert_array_equal(prepared.indices, [0, 2, 2, 3])
    np.testing.assert_array_equal(prepared.data, [3, 5, 7, 2])
    np.testing.assert_array_equal(prepared.delay_count, [[0, 0, 4], [0, 0, 0]])


def test_constructor_has_no_event_input_or_event_state():
    with pytest.raises(TypeError, match="events"):
        _delaycsr_init_case(events=np.zeros((2, 3), np.float32))
    with pytest.raises(TypeError, match="sorted_by_delay"):
        _delaycsr_init_case(sorted_by_delay=1)
    w = _delaycsr_init_case()
    assert not {"event", "events", "duration_steps"} & vars(w).keys()


@pytest.mark.parametrize("array_type", [np.asarray, jnp.asarray], ids=["numpy", "jax"])
def test_with_data_preserves_supplied_current_order_and_owned_state(array_type):
    matrix = _delaycsr_init_matrix()
    # Initialization permutes original edges as [1, 2, 0, 3]. New weights
    # already follow the resulting layout and must not be permuted again.
    np.testing.assert_array_equal(matrix.data, [3, 5, 2, 7])
    np.testing.assert_array_equal(matrix.indices, [0, 1, 2, 0])
    new_weights = array_type([10, 20, 30, 40], dtype=np.float32)
    updated = matrix.with_data(new_weights)
    np.testing.assert_array_equal(updated.data, new_weights)
    np.testing.assert_array_equal(matrix.data, [3, 5, 2, 7])
    _delaycsr_init_assert_same_structure_and_state(matrix, updated)
    updated.clear_head()
    np.testing.assert_array_equal(updated.ring[3], 0)
    np.testing.assert_array_equal(matrix.ring, np.arange(12).reshape(4, 3))
    assert int(matrix.head) == int(updated.head) == 3


def test_shared_weight_keeps_its_exact_shape():
    matrix = _delaycsr_init_matrix(homogeneous=True)
    updated = matrix.with_data(jnp.array([11], jnp.float32))
    assert updated.data.shape == (1,)
    np.testing.assert_array_equal(updated.data, [11])
    np.testing.assert_array_equal(matrix.data, [2])
    _delaycsr_init_assert_same_structure_and_state(matrix, updated)


@GPU
@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_delivery_uses_replacement_weights_on_current_edges(weighted):
    matrix = _delaycsr_init_matrix()
    updated = matrix.with_data(jnp.array([10, 20, 30, 40], jnp.float32))
    initial = jnp.ones((4, 3), jnp.float32)
    events = jnp.array([-2, .5], jnp.float32)
    result = jax.jit(lambda state, ring, event: state.csr_delay_deliver_layout(
        ring, event, 3, weighted=weighted))(updated, initial, events)
    expected = np.ones((4, 3), np.float32)
    expected[0, :2] += [-20, -40] if weighted else [10, 20]
    expected[1, [0, 2]] += [20, -60] if weighted else [40, 30]
    np.testing.assert_array_equal(result, expected)
    _delaycsr_init_assert_same_structure_and_state(matrix, updated)
    np.testing.assert_array_equal(matrix.data, [3, 5, 2, 7])


def test_jit_and_grad_use_current_weight_order_with_jax_delivery():
    matrix = _delaycsr_init_matrix()
    weights = jnp.array([10, 20, 30, 40], jnp.float32)
    cotangent = jnp.arange(1, 13, dtype=jnp.float32).reshape(4, 3)

    def loss(new_weights):
        updated = matrix.with_data(new_weights)
        work = updated.workspace
        # Different arrival locations make each weight's derivative distinct.
        ring = csr_delay_deliver_layout_p.call(
            jnp.zeros_like(updated.ring), jnp.array([True, True]), updated.data,
            updated.indices, updated.indptr, updated.delay_count, updated.delay_off,
            updated.head.reshape(1), updated.row_task_count, *work, backend="jax_raw",
        )[0]
        return jnp.sum(ring * cotangent)

    value, gradient = jax.jit(jax.value_and_grad(loss))(weights)
    # Sorted edges arrive at (slot, column): (0,0), (0,1), (1,2), (1,0).
    np.testing.assert_array_equal(gradient, [1, 2, 6, 4])
    np.testing.assert_array_equal(value, 390)
    np.testing.assert_array_equal(matrix.data, [3, 5, 2, 7])


@pytest.mark.parametrize("value", [None, 2.0, [1, 2, 3, 4], (1, 2, 3, 4)])
def test_non_array_weights_are_rejected(value):
    with pytest.raises(TypeError, match="array"):
        _delaycsr_init_matrix().with_data(value)


@pytest.mark.parametrize("sparse_type", ["csr", "delaycsr"])
def test_sparse_objects_are_not_weight_arrays(sparse_type):
    matrix = _delaycsr_init_matrix()
    value = matrix if sparse_type == "delaycsr" else CSR(
        (matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape)
    with pytest.raises(TypeError, match="array"):
        matrix.with_data(value)


@pytest.mark.parametrize("homogeneous,shape", [
    (False, ()), (False, (1,)), (False, (3,)), (False, (2, 2)), (True, (4,)),
])
def test_replacement_shape_must_match_exactly(homogeneous, shape):
    with pytest.raises(ValueError, match="shape"):
        _delaycsr_init_matrix(homogeneous=homogeneous).with_data(np.ones(shape, np.float32))


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32])
def test_replacement_dtype_must_match(dtype):
    with pytest.raises(TypeError, match="dtype"):
        _delaycsr_init_matrix().with_data(np.ones(4, dtype))


def test_replacement_units_must_match():
    with pytest.raises(TypeError, match="unit"):
        _delaycsr_init_matrix().with_data(jnp.ones(4, jnp.float32) * u.mS)


@pytest.mark.parametrize("receiver", ["csr", "uninitialized"])
def test_receiver_must_be_an_initialized_delaycsr(receiver):
    matrix = _delaycsr_init_matrix()
    value = (CSR((matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape)
             if receiver == "csr" else object.__new__(delaycsr))
    with pytest.raises(TypeError, match="initializ"):
        delaycsr.with_data(value, jnp.ones(4, jnp.float32))


@GPU
def test_inherited_plasticity_updates_keep_array_based_weight_replacement():
    matrix = _delaycsr_init_matrix()
    pre = matrix.update_on_pre(
        jnp.array([True, False]), jnp.array([1, 10, 100], jnp.float32))
    np.testing.assert_array_equal(pre.data, [4, 15, 102, 7])
    _delaycsr_init_assert_same_structure_and_state(matrix, pre)
    post = matrix.update_on_post(
        jnp.array([10, 100], jnp.float32), jnp.array([True, False, True]))
    np.testing.assert_array_equal(post.data, [13, 5, 12, 107])
    _delaycsr_init_assert_same_structure_and_state(matrix, post)
    np.testing.assert_array_equal(matrix.data, [3, 5, 2, 7])


@pytest.mark.parametrize("backend", ["cuda", "cusparse", "numba", "invalid"])
def test_unsupported_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="backend"):
        _delaycsr_init_case(backend=backend)


@pytest.mark.parametrize("backend", [None, "cuda_raw", "jax_raw"])
def test_source_backend_is_inherited(backend):
    original = _delaycsr_init_source(backend)
    before = tuple(np.asarray(value).copy() for value in (original.data, original.indices, original.indptr))
    inherited = delaycsr(original, edge_delays=np.array([2, 0, 1, 3]))
    for current, expected in zip((original.data, original.indices, original.indptr), before):
        np.testing.assert_array_equal(current, expected)
    np.testing.assert_array_equal(inherited.data, [-1, .5, 2, 3])
    assert inherited.indptr.dtype == original.indptr.dtype == jnp.int64
    assert inherited.backend == original.backend == backend


@pytest.mark.parametrize("source_backend", [None, "cuda_raw", "jax_raw"])
@pytest.mark.parametrize("override", ["cuda_raw", "jax_raw"])
def test_source_backend_can_be_overridden_without_mutating_source(source_backend, override):
    original = _delaycsr_init_source(source_backend)
    state = delaycsr(original, edge_delays=np.array([2, 0, 1, 3]), backend=override)
    assert state.backend == override
    assert original.backend == source_backend


def test_unsupported_source_backend_needs_an_explicit_supported_override():
    original = _delaycsr_init_source("numba")
    with pytest.raises(ValueError, match="backend"):
        delaycsr(original, edge_delays=np.array([2, 0, 1, 3]))
    state = delaycsr(original, edge_delays=np.array([2, 0, 1, 3]), backend="jax_raw")
    assert state.backend == "jax_raw"
    assert original.backend == "numba"


@pytest.mark.parametrize("backend", [None, "cuda_raw", "jax_raw"])
def test_pytree_and_with_data_keep_backend(backend):
    state = _delaycsr_init_case(backend=backend)
    assert isinstance(state.workspace, module.HybridWorkspace)
    leaves, tree = jax.tree.flatten(state)
    restored = jax.tree.unflatten(tree, leaves)
    updated = state.with_data(jnp.full_like(state.data, 2))
    assert isinstance(restored, delaycsr)
    np.testing.assert_array_equal(restored.delay_count, state.delay_count)
    assert restored.backend == updated.backend == state.backend == backend
    assert restored.delay_steps == updated.delay_steps == state.delay_steps
    assert updated.ring is state.ring
    assert updated.workspace is state.workspace


@pytest.mark.parametrize("source_kind", ["arrays", "csr"])
@pytest.mark.parametrize("sorted_by_delay", [False, True])
def test_constructor_owns_host_arrays(source_kind, sorted_by_delay):
    weights = _delaycsr_init_aligned_host_array([2., 3.], np.float32)
    columns = _delaycsr_init_aligned_host_array([0, 1], np.int32)
    pointers = _delaycsr_init_aligned_host_array([0, 2], np.int32)
    delays = _delaycsr_init_aligned_host_array([1, 2], np.int32)
    arrays = (weights, columns, pointers)
    source = CSR(arrays, shape=(1, 2), backend="jax_raw") if source_kind == "csr" else arrays
    prepared = delaycsr(source, shape=(1, 2), edge_delays=delays,
                        sorted_by_delay=sorted_by_delay, backend="jax_raw")
    weights[:] = 9
    columns[:] = 9
    pointers[:] = 7
    delays[:] = 0
    np.testing.assert_array_equal(prepared.data, [2, 3])
    np.testing.assert_array_equal(prepared.indices, [0, 1])
    np.testing.assert_array_equal(prepared.indptr, [0, 2])
    np.testing.assert_array_equal(prepared.delay_count, [[1, 1]])


def test_with_data_owns_numpy_replacement():
    matrix = _delaycsr_init_matrix()
    weights = _delaycsr_init_aligned_host_array([10., 20., 30., 40.], np.float32)
    updated = matrix.with_data(weights)
    weights[:] = -1
    np.testing.assert_array_equal(updated.data, [10, 20, 30, 40])
    _delaycsr_init_assert_same_structure_and_state(matrix, updated)


def test_constructor_prepares_weighted_workspace_in_explicit_ring_dtype():
    ring = np.full((3, 1), 5., np.float64)
    matrix = delaycsr(
        (np.array([2.], np.float32), np.zeros(129, np.int32), np.array([0, 129], np.int32)),
        shape=(1, 1), edge_delays=np.ones(129, np.int32), buffer_slots=3,
        ring=ring, weighted=True, backend="jax_raw",
    )
    ring[:] = -1
    assert matrix.data.dtype == jnp.float32
    assert matrix.ring.dtype == matrix.workspace.task_value.dtype == jnp.float64
    np.testing.assert_array_equal(matrix.ring, np.full((3, 1), 5.))
    assert matrix.capacity > 0


def test_quantity_sorting_weight_replacement_and_owned_ring():
    matrix = delaycsr(
        (jnp.array([2., 3.], jnp.float32) * u.mV, np.array([0, 1], np.int32),
         np.array([0, 2], np.int32)), shape=(1, 2), edge_delays=np.array([2, 1], np.int32),
        backend="jax_raw",
    )
    assert u.get_unit(matrix.data) == u.mV
    assert u.get_unit(matrix.ring) == u.mV
    np.testing.assert_array_equal(u.get_mantissa(matrix.data), [3, 2])
    updated = matrix.with_data(jnp.array([5., 7.], jnp.float32) * u.mV)
    assert u.get_unit(updated.data) == u.mV
    _delaycsr_init_assert_same_structure_and_state(matrix, updated)
    output = updated.forward(jnp.array([True]), 0)
    assert u.get_unit(output) == u.mV
    np.testing.assert_array_equal(u.get_mantissa(output), [0, 5])


def test_explicit_float64_ring_requires_x64_before_device_transfer():
    with jax.enable_x64(False):
        with pytest.raises(RuntimeError, match="jax_enable_x64.*ring"):
            delaycsr((np.array([2.], np.float32), np.array([0], np.int32),
                      np.array([0, 1], np.int32)), shape=(1, 1),
                     edge_delays=np.array([1], np.int32), ring=np.zeros((2, 1), np.float64))


def test_list_ring_uses_jax_default_precision_when_x64_is_disabled():
    with jax.enable_x64(False):
        matrix = delaycsr(
            (np.array([2.], np.float32), np.array([0], np.int32),
             np.array([0, 1], np.int32)), shape=(1, 1),
            edge_delays=np.array([1], np.int32), ring=[[1.], [2.]],
            backend="jax_raw",
        )
        assert matrix.ring.dtype == jnp.float32
        np.testing.assert_array_equal(matrix.forward(jnp.array([True]), 0), [4.])


# Owned state, time steps and backend dispatch
# --------------------------------------------

def _delaycsr_problem(dtype=jnp.float32, weighted=False, homo=False, backend=None):
    weights = np.asarray([.5] if homo else [2, -1, 3, .5], dtype=jnp.dtype(dtype))
    indices = np.array([1, 0, 0, 1], np.int32)
    indptr = np.array([0, 3, 4], np.int64)
    delays = np.array([2, 1, 0, 1], np.int32)
    matrix = delaycsr((weights, indices, indptr), shape=(2, 2), edge_delays=delays,
                      buffer_slots=4, weighted=weighted, backend=backend)
    return matrix, (weights, indices, indptr, delays)


def _delaycsr_reference(ring, original, events, t, weighted):
    result = ring.copy()
    head = t % len(ring)
    result[head] = 0
    return numpy_project_step(*original, events, head, result, weighted=weighted)


@GPU
@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_forward_and_matmul_return_next_preserving_full_ring(dtype, weighted, homo):
    w, original = _delaycsr_problem(dtype, weighted, homo)
    w.ring = jnp.arange(8, dtype=dtype).reshape(4, 2)
    expected = np.asarray(w.ring).astype(np.float64)
    metadata = (w.delay_count, w.delay_off, w.row_task_count, w.indices, w.indptr)
    for t in range(11, 20):
        x = np.array([-2, .5] if t % 3 else [0, 0], dtype=jnp.dtype(dtype if weighted else jnp.float32))
        expected = _delaycsr_reference(expected, original, x, t, weighted)
        if t % 2:
            out = w.forward(jnp.asarray(x), t)
        else:
            w.head = t % w.L
            out = jnp.asarray(x) @ w
        np.testing.assert_array_equal(out, expected[(t + 1) % w.L])
        np.testing.assert_array_equal(w.ring, expected)
        assert int(w.head) == t % w.L
        for target in (t, t + 1, t + 2):
            np.testing.assert_array_equal(w.read_spike(target), expected[target % w.L])
        np.testing.assert_array_equal(w.ring, expected)
    assert all(a is b for a, b in zip(metadata, (w.delay_count, w.delay_off, w.row_task_count,
                                                w.indices, w.indptr)))


@GPU
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('rows,columns', [(0, 0), (0, 3), (2, 0), (2, 3)])
def test_no_edges_still_clear_head(weighted, rows, columns):
    w = delaycsr((np.array([1], np.float32), np.array([], np.int32), np.zeros(rows + 1, np.int32)),
                 shape=(rows, columns), edge_delays=np.array([], np.int32), buffer_slots=3,
                 weighted=weighted, ring=jnp.ones((3, columns), jnp.float32))
    y = w.forward(jnp.zeros(rows, jnp.float32), 5)
    expected = np.ones((3, columns), np.float32)
    expected[2] = 0
    np.testing.assert_array_equal(w.ring, expected)
    np.testing.assert_array_equal(y, expected[0])


@GPU
@pytest.mark.parametrize('weighted', [False, True])
def test_explicit_state_jit_and_scan(weighted):
    w, original = _delaycsr_problem(weighted=weighted)
    x = jnp.array([[-2, .5], [0, 0], [.25, -1], [1, 2], [-1, .5]], jnp.float32)

    def step(state, event_time):
        event, t = event_time
        y = state.forward(event, t)
        return state, y

    expected = np.zeros((4, 2))
    outputs = []
    for t, event in enumerate(np.asarray(x), 7):
        expected = _delaycsr_reference(expected, original, event, t, weighted)
        outputs.append(expected[(t + 1) % 4].copy())
    after, outputs_scan = jax.jit(lambda state: jax.lax.scan(step, state, (x, jnp.arange(7, 12))))(w)
    np.testing.assert_array_equal(outputs_scan, outputs)
    np.testing.assert_array_equal(after.ring, expected)
    np.testing.assert_array_equal(w.ring, 0)
    assert not isinstance(w.head, jax.core.Tracer)
    after, y = jax.jit(step)(after, (jnp.zeros(2, jnp.float32), jnp.array(12)))
    expected = _delaycsr_reference(expected, original, np.zeros(2), 12, weighted)
    np.testing.assert_array_equal(after.ring, expected)
    np.testing.assert_array_equal(y, expected[1])


@GPU
def test_wrapped_events_and_large_time():
    binary, _ = _delaycsr_problem()
    weighted, _ = _delaycsr_problem(weighted=True)
    x = jnp.array([-2, .5], jnp.float32)
    np.testing.assert_array_equal(binary.forward(BinaryArray(x), 2**40 + 1), [-1, .5])
    np.testing.assert_array_equal(weighted.forward(WeightedArray(x), 2**40 + 1), [2, .25])
    with pytest.raises(TypeError, match='mode'):
        weighted.forward(BinaryArray(x), 0)


def test_head_and_time_validation():
    w, _ = _delaycsr_problem()
    for head in (-1, 4, 2**32):
        with pytest.raises(ValueError, match='head'):
            w.head = head
    for t in (True, .5, [1]):
        with pytest.raises(TypeError, match='integer'):
            w.forward(jnp.zeros(2, jnp.float32), t)


@pytest.mark.parametrize('dtype', [np.int64, np.uint64])
@pytest.mark.parametrize('array_kind,compiled,output', [
    ('numpy', False, 'head'), ('jax', False, 'head'),
    ('numpy', True, 'head'), ('jax', True, 'clear'),
])
def test_head_setter_checks_wide_scalar_before_narrowing(dtype, array_kind, compiled, output):
    with jax.enable_x64(True):
        state, _ = _delaycsr_problem(backend='jax_raw')
        state.ring = jnp.ones_like(state.ring)
        array = np.asarray if array_kind == 'numpy' else jnp.asarray

        def assign(matrix, value):
            matrix.head = value
            return matrix.clear_head() if output == 'clear' else matrix.head

        run = jax.jit(assign) if compiled else assign
        result = jax.block_until_ready(run(state, array(3, dtype=dtype)))
        if output == 'clear':
            np.testing.assert_array_equal(result, [[1, 1], [1, 1], [1, 1], [0, 0]])
        else:
            assert result.dtype == jnp.int32
            np.testing.assert_array_equal(result, 3)
        with pytest.raises(Exception if compiled else ValueError, match='head'):
            jax.block_until_ready(run(state, array(2**32, dtype=dtype)))


@pytest.mark.parametrize('dtype', [np.int64, np.uint64])
def test_numpy_head_setter_checks_original_width_without_x64(dtype):
    state, _ = _delaycsr_problem(backend='jax_raw')
    with jax.enable_x64(False):
        state.head = np.asarray(3, dtype=dtype)
        np.testing.assert_array_equal(state.head, 3)
        with pytest.raises(ValueError, match='head'):
            state.head = np.asarray(2**32, dtype=dtype)
        np.testing.assert_array_equal(state.head, 3)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_with_data_gradient_without_jit_preserves_source_state(backend, weighted, homo):
    state, original = _delaycsr_problem(weighted=weighted, homo=homo, backend=backend)
    ring = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    state.ring = ring
    leaves = jax.tree.leaves(state)
    events = jnp.array([-2, .5], jnp.float32)

    def loss(matrix, weights):
        local = matrix.with_data(weights)
        output = local.forward(events, 3)
        return local.ring.sum() + output.sum()

    expected_ring = _delaycsr_reference(np.asarray(ring), original, np.asarray(events), 3, weighted)
    expected_value = expected_ring.sum() + expected_ring[0].sum()
    # Prepared edge order has delays [0, 1, 2, 1]. Delay-one edges also
    # contribute to the returned next-slot output and therefore count twice.
    expected_gradient = np.array([-2, -4, -2, 1] if weighted else [1, 2, 1, 2])
    if homo:
        expected_gradient = expected_gradient.sum(keepdims=True)
    differentiate = jax.value_and_grad(loss, argnums=1)
    compiled_value, compiled_gradient = jax.jit(differentiate)(state, state.data)
    value, gradient = differentiate(state, state.data)
    for actual in (value, compiled_value):
        np.testing.assert_array_equal(actual, expected_value)
    for actual in (gradient, compiled_gradient):
        np.testing.assert_array_equal(actual, expected_gradient)
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(state)))
    np.testing.assert_array_equal(state.ring, ring)
    np.testing.assert_array_equal(state.head, 0)
    for array in state.workspace:
        np.testing.assert_array_equal(array, 0)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
def test_unbatched_state_vmap_without_jit_preserves_source_state(backend, weighted):
    state, original = _delaycsr_problem(weighted=weighted, backend=backend)
    ring = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    state.ring = ring
    leaves = jax.tree.leaves(state)
    events = jnp.array([[-2, 0], [0, .5]], jnp.float32)
    times = jnp.array([3, 4], jnp.int32)

    def step(matrix, event, time):
        output = matrix.forward(event, time)
        return matrix, output

    mapped = jax.vmap(step, in_axes=(None, 0, 0))
    compiled_state, compiled_outputs = jax.jit(mapped)(state, events, times)
    batched_state, outputs = mapped(state, events, times)
    expected_rings = np.stack([
        _delaycsr_reference(np.asarray(ring), original, event, int(time), weighted)
        for event, time in zip(np.asarray(events), np.asarray(times))
    ])
    expected_outputs = expected_rings[np.arange(2), (np.asarray(times) + 1) % state.L]
    for actual in (outputs, compiled_outputs):
        np.testing.assert_array_equal(actual, expected_outputs)
    for actual in (batched_state, compiled_state):
        np.testing.assert_array_equal(actual.ring, expected_rings)
        np.testing.assert_array_equal(actual.head, np.asarray(times) % state.L)
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(state)))
    np.testing.assert_array_equal(state.ring, ring)
    np.testing.assert_array_equal(state.head, 0)
    for array in state.workspace:
        np.testing.assert_array_equal(array, 0)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('dtype', [
    jnp.bool_, jnp.int8, jnp.uint8, jnp.int32, jnp.int64,
    jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64,
])
def test_matmul_accepts_main_event_types(weighted, dtype, backend):
    w, original = _delaycsr_problem(weighted=weighted, backend=backend)
    value = 2 if jnp.dtype(dtype).kind in 'bu' else -2
    events = jnp.array([value, 0], dtype)
    expected = _delaycsr_reference(np.zeros((4, 2)), original, np.asarray(events), 0, weighted)
    np.testing.assert_array_equal(events @ w, expected[1])
    np.testing.assert_array_equal(w.ring, expected)
    assert w.ring.dtype == jnp.float32


@pytest.mark.parametrize('weighted', [False, True])
def test_invalid_event_type_preserves_ring(weighted):
    w, _ = _delaycsr_problem(weighted=weighted, backend='jax_raw')
    w.ring = jnp.ones_like(w.ring)
    workspace = w.workspace
    with pytest.raises(TypeError, match='events'):
        w.__rmatmul__(jnp.ones(2, jnp.complex64))
    np.testing.assert_array_equal(w.ring, 1)
    assert w.workspace is workspace


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
def test_half_ring_preserves_main_underflow_and_nonfinite_semantics(backend, weighted):
    layout = delaycsr(
        (np.array([np.inf], np.float16), np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        shape=(1, 1), edge_delays=np.ones(1, np.int32), backend=backend,
    )
    ring = jnp.zeros((2, 1), jnp.float16)
    tiny = jnp.array([1e-8], jnp.float32)
    np.testing.assert_array_equal(tiny.astype(jnp.float16), 0)

    def deliver(events):
        return layout.csr_delay_deliver_layout(ring, events, 0, weighted=weighted)

    output = jax.jit(deliver)(tiny)
    if not weighted:
        assert np.isinf(np.asarray(output)[1, 0])
    elif backend == 'jax_raw':
        assert np.isnan(np.asarray(output)[1, 0])
    else:
        np.testing.assert_array_equal(output, 0)

    primal, tangent = jax.jit(lambda x: jax.jvp(
        deliver, (x,), (tiny,)))(jnp.zeros_like(tiny))
    np.testing.assert_array_equal(primal, 0)
    assert tangent.dtype == jnp.float16
    if backend == 'jax_raw':
        assert np.isnan(np.asarray(tangent)[1, 0])
    else:
        np.testing.assert_array_equal(tangent, 0)

    def deliver_weights(weights):
        return layout.with_data(weights).csr_delay_deliver_layout(
            ring, tiny, 0, weighted=weighted)

    _, pullback = jax.vjp(deliver_weights, layout.data)
    gradient, = pullback(jnp.full_like(ring, jnp.inf))
    assert gradient.dtype == layout.data.dtype
    assert (np.isnan(np.asarray(gradient)[0]) if weighted
            else np.isinf(np.asarray(gradient)[0]))


@pytest.mark.parametrize('corruption', ['ring_shape', 'ring_dtype', 'workspace'])
def test_matmul_rejects_invalid_mutable_state(corruption):
    w, _ = _delaycsr_problem()
    if corruption == 'ring_shape':
        w.ring = jnp.zeros((4, 3), jnp.float32)
        exception, message = ValueError, 'ring|buffer'
    elif corruption == 'ring_dtype':
        w.ring = w.ring.astype(jnp.int32)
        exception, message = TypeError, 'ring|buffer'
    else:
        w.workspace = None
        exception, message = RuntimeError, 'workspace'
    with pytest.raises(exception, match=message):
        w.__rmatmul__(jnp.ones(2, jnp.float32))


@pytest.mark.parametrize('backend', BACKENDS[1:])
def test_quantity_state_with_mixed_dtype_survives_scan(backend):
    state = delaycsr(
        (np.array([2], np.float32) * u.mV, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        edge_delays=np.ones(1, np.int32), shape=(1, 1),
        weighted=True, ring=jnp.ones((2, 1), jnp.float16) * u.mV,
        backend=backend,
    )

    def step(matrix, inputs):
        events, time = inputs
        result = matrix.forward(events, time)
        return matrix, result

    after, outputs = jax.jit(lambda matrix: jax.lax.scan(
        step, matrix, (jnp.array([[.5], [1.]], jnp.float32), jnp.arange(2))))(state)
    assert u.get_unit(after.ring) == u.mV
    assert after.ring.dtype == jnp.float16
    assert after.data.dtype == jnp.float32
    np.testing.assert_array_equal(outputs.to_decimal(u.mV), [[2], [2]])
    np.testing.assert_array_equal(after.read_spike().to_decimal(u.mV), [2])
    np.testing.assert_array_equal(state.ring.to_decimal(u.mV), 1)
    after.clear_head()
    np.testing.assert_array_equal(after.ring.to_decimal(u.mV), [[2], [0]])


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('control_flow,dtype,weight_unit,event_unit,ring_unit', [
    pytest.param('scan', jnp.float16, u.mV, u.UNITLESS, u.volt, id='scan-f16-voltage'),
    pytest.param('scan', jnp.float32, u.mS, u.mV, u.mA, id='scan-f32-compound'),
    pytest.param('while', jnp.bfloat16, u.mS, u.mV, u.mA, id='while-bf16-compound'),
    pytest.param('while', jnp.float64, u.mV, u.UNITLESS, u.volt, id='while-f64-voltage'),
])
def test_quantity_state_preserves_ring_scale_in_control_flow(
        backend, control_flow, dtype, weight_unit, event_unit, ring_unit):
    state = delaycsr(
        (np.array([2], np.float32) * weight_unit, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        edge_delays=np.ones(1, np.int32), shape=(1, 1), weighted=True,
        ring=jnp.full((2, 1), .001, dtype) * ring_unit, backend=backend,
    )
    leaves = jax.tree.leaves(state)
    events = jnp.array([[.5], [1.]], jnp.float32) * event_unit

    if control_flow == 'scan':
        def step(matrix, inputs):
            event, time = inputs
            return_value = matrix.forward(event, time)
            return matrix, return_value

        after, outputs = jax.jit(lambda matrix: jax.lax.scan(
            step, matrix, (events, jnp.arange(2))))(state)
        assert u.get_unit(outputs) == ring_unit
        np.testing.assert_allclose(u.get_mantissa(outputs), [[.002], [.002]], rtol=1e-3)
    else:
        def step(carry):
            matrix, time = carry
            matrix.forward(events[time], time)
            return matrix, time + 1

        after, _ = jax.jit(lambda matrix: jax.lax.while_loop(
            lambda carry: carry[1] < 2, step, (matrix, jnp.array(0, jnp.int32))))(state)

    tolerance = 1e-2 if dtype == jnp.bfloat16 else 1e-3 if dtype == jnp.float16 else 1e-6
    assert u.get_unit(after.ring) == ring_unit
    assert after.ring.dtype == dtype
    np.testing.assert_allclose(u.get_mantissa(after.ring), [[.002], [0]], rtol=tolerance)
    np.testing.assert_allclose(u.get_mantissa(after.read_spike()), [.002], rtol=tolerance)
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(state)))
    assert u.get_unit(state.ring) == ring_unit
    np.testing.assert_array_equal(u.get_mantissa(state.ring), jnp.full((2, 1), .001, dtype))
    np.testing.assert_array_equal(state.head, 0)


@GPU
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('rows', [0, 1])
@pytest.mark.parametrize('head', [-1, 3])
def test_low_level_rejects_bad_head_eager_and_jit(weighted, rows, head):
    data = delaycsr(
        (np.array([1], np.float32), np.array([], np.int32), np.zeros(rows + 1, np.int32)),
        edge_delays=np.array([], np.int32),
        shape=(rows, 2),
        buffer_slots=3,
        weighted=weighted,
    )
    args = [jnp.asarray(a) for a in (data.weights, data.indices, data.indptr)]
    meta = [jnp.asarray(a) for a in (data.delay_count, data.delay_off, data.row_task_count)]
    values = (*args, jnp.zeros(rows, jnp.float32), *meta, jnp.array([head], jnp.int32),
              jnp.ones((3, 2), jnp.float32), data.workspace)
    with pytest.raises(RuntimeError, match='head'):
        jax.block_until_ready(cuda_step(*values, weighted=weighted))
    op = jax.jit(cuda_step, static_argnames=('weighted',))
    with pytest.raises(RuntimeError, match='head'):
        jax.block_until_ready(op(*values, weighted=weighted))
    valid = (*values[:7], jnp.zeros(1, jnp.int32), *values[8:])
    after, work = op(*valid, weighted=weighted)
    np.testing.assert_array_equal(after, np.ones((3, 2)))
    np.testing.assert_array_equal(work.status, [0, 0])


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_forward_rejects_invalid_event_shape(weighted, shape):
    w, _ = _delaycsr_problem(weighted=weighted)
    with pytest.raises(ValueError, match="events must have shape"):
        w.forward(jnp.zeros(shape, jnp.float32), 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_forward_and_jit_use_the_selected_backend(backend):
    state, _ = _delaycsr_problem(backend=backend)
    state.ring = jnp.ones_like(state.ring)

    def step(matrix, event):
        result = matrix.forward(event, 0)
        return matrix, result

    updated, result = jax.jit(step)(state, jnp.array([-2, .5], jnp.float32))
    np.testing.assert_array_equal(result, [0, 1.5])
    np.testing.assert_array_equal(updated.ring,
                                  [[3, 0], [0, 1.5], [1, 3], [1, 1]])
    assert updated.backend == state.backend == backend
    np.testing.assert_array_equal(state.ring, 1)


@pytest.mark.parametrize("backend", ["jax_raw", pytest.param(None, marks=pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="default uses CUDA on GPU"))])
def test_jax_path_never_loads_cuda(monkeypatch, backend):
    def unexpected_load(*args, **kwargs):
        pytest.fail("JAX delivery attempted to load CUDA")

    monkeypatch.setattr(operator, "_load", unexpected_load)
    state, _ = _delaycsr_problem(backend=backend)
    state.ring = jnp.ones_like(state.ring)
    # The public inherited field remains the source of dispatch decisions.
    if backend == "jax_raw":
        state.backend = "cuda_raw"
        state.backend = "jax_raw"

    def step(matrix, event):
        result = matrix.forward(event, 0)
        return matrix, result

    compiled = jax.jit(step).lower(state, jnp.array([True, True]))
    assert ".region_delay_" not in compiled.as_text()
    updated, result = compiled.compile()(state, jnp.array([True, True]))
    np.testing.assert_array_equal(result, [0, 1.5])
    assert updated.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_allow_more_edges_than_dense_matrix_entries(backend):
    args = (np.array([.5], np.float32), np.zeros(9, np.int32), np.array([0, 9], np.int32))
    state = delaycsr(args, shape=(1, 1), edge_delays=np.ones(9, np.int32),
                     backend=backend, weighted=True)
    np.testing.assert_array_equal(state.forward(jnp.array([2], jnp.float32), 0), [9])


# Public delivery, units and pure state updates
# ---------------------------------------------

def _deliver_layout_make(backend=None, weighted=False, delay=1):
    return delaycsr((np.array([3.], np.float32), np.array([0], np.int32),
                     np.array([0, 1], np.int32)),
                    shape=(1, 1), edge_delays=np.array([delay], np.int32),
                    buffer_slots=2, backend=backend, weighted=weighted,
                    ring=jnp.array([[10.], [20.]], jnp.float32))


def _deliver_layout_source(backend):
    return brainevent.CSR(
        (np.array([2, 3, 5, 7], np.float32), np.array([2, 0, 1, 0], np.int32),
         np.array([0, 3, 4], np.int32)), shape=(2, 3), backend=backend,
    )


def test_public_delay_exports_share_one_prepared_implementation():
    packages = [brainevent, importlib.import_module("brainevent._csr"),
                importlib.import_module("brainevent._csr.delay")]
    for package in packages:
        assert package.delaycsr is brainevent.delaycsr
        assert package.csr_delay_deliver_layout is brainevent.csr_delay_deliver_layout
        assert package.csr_delay_deliver_layout_p is brainevent.csr_delay_deliver_layout_p
        for removed in ("CSRDelayLayout", "csr_delay_deliver", "csr_delay_deliver_p"):
            assert not hasattr(package, removed)
    assert not hasattr(brainevent.CSR, "delay_deliver")
    assert set(brainevent.csr_delay_deliver_layout_p.available_backends("gpu")) == {
        "jax_raw", "cuda_raw",
    }


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('delay', [0, 1])
def test_deliver_preserves_head_accumulates_and_keeps_owned_state(backend, weighted, delay):
    state = _deliver_layout_make(backend, delay=delay)
    leaves = jax.tree.leaves(state)
    events = jnp.array([2.], jnp.float32)
    result = csr_delay_deliver_layout(state.ring, events, state, 0, weighted=weighted)
    again = state.csr_delay_deliver_layout(result, events, 0, weighted=weighted)
    contribution = 6 if weighted else 3
    expected = np.array([[10], [20]])
    expected[delay, 0] += contribution
    np.testing.assert_array_equal(result, expected)
    expected[delay, 0] += contribution
    np.testing.assert_array_equal(again, expected)
    inactive = csr_delay_deliver_layout(state.ring, jnp.zeros_like(events), state, 0,
                                        weighted=weighted)
    np.testing.assert_array_equal(inactive, [[10], [20]])
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(state)))
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    assert int(state.head) == 0


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('t', [-1, 0, 2**40 + 1])
def test_supported_high_level_call_matches_arrival_rule(backend, t):
    state = _deliver_layout_make(backend)
    events = WeightedArray(jnp.array([-2.], jnp.float32))
    expected = np.asarray(state.ring).copy()
    expected[(t + 1) % 2, 0] += -2 * 3
    actual = jax.jit(lambda r: csr_delay_deliver_layout(r, events, state, t))(state.ring)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('backend', BACKENDS)
def test_delivery_gradients_and_stateful_step_follow_selected_backend(backend):
    state = _deliver_layout_make(backend, weighted=True)
    events = jnp.array([2.], jnp.float32)
    project = lambda r: state.csr_delay_deliver_layout(r, events, 0, weighted=True)
    result, tangent = jax.jit(lambda r: jax.jvp(project, (r,), (jnp.ones_like(r),)))(state.ring)
    np.testing.assert_array_equal(result, [[10], [26]])
    np.testing.assert_array_equal(tangent, [[1], [1]])

    def loss(weights, ring, event):
        return state.with_data(weights).csr_delay_deliver_layout(ring, event, 0, weighted=True).sum()

    value, (weight_gradient, ring_gradient, event_gradient) = jax.jit(jax.value_and_grad(
        loss, argnums=(0, 1, 2)))(state.data, state.ring, events)
    np.testing.assert_array_equal(value, 36)
    np.testing.assert_array_equal(weight_gradient, [2])
    np.testing.assert_array_equal(ring_gradient, [[1], [1]])
    np.testing.assert_array_equal(event_gradient, [3])
    assert state.backend == backend
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    np.testing.assert_array_equal(state.forward(events, 0), [26])
    np.testing.assert_array_equal(state.ring, [[0], [26]])


def test_clear_head_is_a_jax_member_with_local_gradient():
    state = _deliver_layout_make('jax_raw')

    def cleared(ring):
        local = copy(state)
        local.ring = ring
        return local.clear_head()

    for operation in (cleared, jax.jit(cleared)):
        value, tangent = jax.jvp(operation, (state.ring,), (jnp.ones_like(state.ring),))
        np.testing.assert_array_equal(value, [[0], [20]])
        np.testing.assert_array_equal(tangent, [[0], [1]])
        np.testing.assert_array_equal(jax.grad(lambda r: operation(r).sum())(state.ring), [[0], [1]])
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    state.head = 1
    assert state.clear_head() is state.ring
    np.testing.assert_array_equal(state.ring, [[10], [0]])
    assert int(state.head) == 1


def test_clear_head_unbatched_state_vmap_without_jit():
    state = _deliver_layout_make('jax_raw')
    leaves = jax.tree.leaves(state)

    def clear(matrix, head):
        matrix.head = head
        result = matrix.clear_head()
        return matrix, result

    heads = jnp.array([0, 1], jnp.int32)
    mapped = jax.vmap(clear, in_axes=(None, 0))
    compiled_state, compiled_rings = jax.jit(mapped)(state, heads)
    batched_state, rings = mapped(state, heads)
    expected = [[[0], [20]], [[10], [0]]]
    for result in (rings, compiled_rings, batched_state.ring, compiled_state.ring):
        np.testing.assert_array_equal(result, expected)
    for result in (batched_state.head, compiled_state.head):
        np.testing.assert_array_equal(result, heads)
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(state)))
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    np.testing.assert_array_equal(state.head, 0)


@pytest.mark.parametrize('backend', BACKENDS)
def test_pure_delivery_scan_reuses_layout_without_advancing_owned_head(backend):
    state = _deliver_layout_make(backend)

    def step(ring, t):
        ring = csr_delay_deliver_layout(ring, jnp.array([True]), state, t)
        return ring, ring[t % 2]

    ring, arrivals = jax.jit(lambda r: jax.lax.scan(step, r, jnp.arange(3)))(state.ring)
    np.testing.assert_array_equal(ring, [[13], [26]])
    np.testing.assert_array_equal(arrivals, [[10], [23], [13]])
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    assert int(state.head) == 0


def test_only_existing_delaycsr_layout_is_accepted():
    with pytest.raises(TypeError, match='delaycsr'):
        csr_delay_deliver_layout(jnp.zeros((2, 1)), jnp.array([True]), object(), 0)


@pytest.mark.parametrize('backend', ['cuda', 'cusparse', 'numba', 'invalid'])
def test_deliver_rejects_unsupported_backend_without_changing_state(backend):
    state = _deliver_layout_make()
    before = jax.tree.leaves(state)
    with pytest.raises(ValueError, match='backend'):
        state.csr_delay_deliver_layout(state.ring, jnp.array([True]), 0, backend=backend)
    assert all(old is new for old, new in zip(before, jax.tree.leaves(state)))


@pytest.mark.parametrize('owned_backend', [None, 'jax_raw', 'cuda_raw'])
@pytest.mark.parametrize('override', ['jax_raw', pytest.param('cuda_raw', marks=GPU)])
def test_backend_override_uses_a_pure_copy_of_the_existing_layout(owned_backend, override):
    state = _deliver_layout_make(owned_backend)
    before = jax.tree.leaves(state)
    result = jax.jit(lambda ring: csr_delay_deliver_layout(
        ring, jnp.array([True]), state, 0, backend=override))(state.ring)
    np.testing.assert_array_equal(result, [[10], [23]])
    assert state.backend == owned_backend
    assert all(old is new for old, new in zip(before, jax.tree.leaves(state)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_csr_delay_layout_preserves_source_and_inherits_backend(backend):
    source = _deliver_layout_source(backend)
    original = tuple(np.asarray(x).copy() for x in (source.data, source.indices, source.indptr))
    layout = source.delay_layout(np.array([2, 0, 1, 2], np.int32), n_slots=4)
    assert isinstance(layout, brainevent.delaycsr)
    assert isinstance(layout, brainevent.CSR)
    assert layout.backend == source.backend == backend
    np.testing.assert_array_equal(layout.data, [3, 5, 2, 7])
    np.testing.assert_array_equal(layout.indices, [0, 1, 2, 0])
    for old, current in zip(original, (source.data, source.indices, source.indptr)):
        np.testing.assert_array_equal(current, old)

    events = jnp.array([2, -1], jnp.float32)
    ring = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    expected = np.asarray(ring).copy()
    for row, col, weight, delay in zip([0, 0, 0, 1], original[1], original[0], [2, 0, 1, 2]):
        expected[(3 + delay) % 4, col] += weight * np.asarray(events)[row]

    def deliver(weights, ring, events):
        updated = layout.with_data(weights)
        return brainevent.csr_delay_deliver_layout(
            ring, brainevent.WeightedArray(events), updated, 3,
        )

    np.testing.assert_array_equal(jax.jit(deliver)(layout.data, ring, events), expected)
    gradients = jax.jit(jax.grad(lambda w, r, x: deliver(w, r, x).sum(),
                                 argnums=(0, 1, 2)))(layout.data, ring, events)
    np.testing.assert_array_equal(gradients[0], [2, 2, 2, -1])
    np.testing.assert_array_equal(gradients[1], np.ones((4, 3), np.float32))
    np.testing.assert_array_equal(gradients[2], [10, 7])
    np.testing.assert_array_equal(layout.ring, np.zeros((4, 3), np.float32))
    assert int(layout.head) == 0
    assert layout.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
def test_csr_delay_layout_state_carries_through_jitted_scan(backend):
    source = brainevent.CSR(
        (np.array([3], np.float32), np.array([0], np.int32), np.array([0, 1], np.int32)),
        shape=(1, 1), backend=backend,
    )
    layout = source.delay_layout(np.array([1], np.int32), n_slots=2)
    layout.ring = jnp.array([[10], [20]], jnp.float32)

    def step(state, inputs):
        t, event = inputs
        arrival = state.forward(event, t)
        return state, arrival

    state, arrivals = jax.jit(lambda state: jax.lax.scan(
        step, state, (jnp.arange(3, dtype=jnp.int32), jnp.array([[True], [True], [False]])),
    ))(layout)
    np.testing.assert_array_equal(arrivals, [[23], [3], [0]])
    np.testing.assert_array_equal(state.ring, [[0], [0]])
    np.testing.assert_array_equal(layout.ring, [[10], [20]])
    assert state.backend == layout.backend == backend


@pytest.mark.parametrize("backend", ["jax_raw", pytest.param("cuda_raw", marks=GPU)])
@pytest.mark.parametrize("corruption", ["head", "metadata"])
def test_sharded_delivery_keeps_runtime_validation_and_distinct_rings(backend, corruption):
    # A size-one mesh still tracks manual varying axes; on multiple devices,
    # distinct rings also check that callback handling never replicates data.
    count = min(2, jax.device_count())
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:count]), ("delay_shard",))
    partition = jax.sharding.PartitionSpec("delay_shard")

    def local(rings, heads, counts):
        queue = jnp.zeros(0, jnp.int32)
        result = brainevent.csr_delay_deliver_layout_p.call(
            rings[0], jnp.array([True]), jnp.array([3], jnp.float32),
            jnp.array([0], jnp.int32), jnp.array([0, 1], jnp.int32), counts[0],
            jnp.array([1], jnp.int32), heads[0], jnp.zeros(1, jnp.int32),
            queue, queue, queue, jnp.zeros(2, jnp.int32), backend=backend,
        )
        return result[0][None]

    deliver = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(partition,) * 3,
                                   out_specs=partition))
    rings = jnp.arange(count * 2, dtype=jnp.float32).reshape(count, 2, 1)
    heads = jnp.zeros((count, 1), jnp.int32)
    counts = jnp.ones((count, 1, 1), jnp.int32)
    expected = np.asarray(rings).copy()
    expected[:, 1, 0] += 3
    np.testing.assert_array_equal(deliver(rings, heads, counts), expected)
    bad_heads = heads.at[-1, 0].set(2) if corruption == "head" else heads
    bad_counts = counts.at[-1, 0, 0].set(0) if corruption == "metadata" else counts
    with pytest.raises(Exception, match=corruption):
        jax.block_until_ready(deliver(rings, bad_heads, bad_counts))
    np.testing.assert_array_equal(deliver(rings, heads, counts), expected)
    jax.jit(lambda: jax.debug.callback(lambda: None, ordered=False))()
    jax.effects_barrier()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.int8, jnp.int32, jnp.float16,
                                         jnp.bfloat16, jnp.float32, jnp.float64])
def test_public_delivery_accepts_main_event_dtypes(backend, weighted, event_dtype):
    layout = _deliver_layout_make(backend)
    before = jax.tree.leaves(layout)
    events = jnp.array([True] if event_dtype == jnp.bool_ else [-2], event_dtype)
    result = jax.jit(lambda x: csr_delay_deliver_layout(
        layout.ring, x, layout, 0, weighted=weighted))(events)
    contribution = -6 if weighted and event_dtype != jnp.bool_ else 3
    np.testing.assert_array_equal(result, [[10], [20 + contribution]])
    assert result.dtype == jnp.float32
    assert all(old is new for old, new in zip(before, jax.tree.leaves(layout)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weight_dtype,ring_dtype", [(jnp.float16, jnp.float32),
                                                     (jnp.float64, jnp.float32),
                                                     (jnp.float32, jnp.float64)])
def test_public_ring_dtype_controls_delivery_and_gradient(backend, weight_dtype, ring_dtype):
    source = brainevent.CSR((jnp.array([3.], weight_dtype), jnp.array([0], jnp.int32),
                            jnp.array([0, 1], jnp.int32)), shape=(1, 1), backend=backend)
    layout = source.delay_layout(np.array([1], np.int32), 3)
    before = jax.tree.leaves(layout)
    ring = jnp.array([[1], [2], [3]], ring_dtype)
    events = jnp.array([-2.], jnp.float64)

    def project(weights, event):
        return csr_delay_deliver_layout(ring, event, layout.with_data(weights), 0, weighted=True)

    result = jax.jit(project)(layout.data, events)
    np.testing.assert_array_equal(result, [[1], [-4], [3]])
    assert result.dtype == ring_dtype
    dw, dx = jax.jit(jax.grad(lambda w, x: project(w, x).sum(), argnums=(0, 1)))(layout.data, events)
    np.testing.assert_array_equal(dw, [-2])
    np.testing.assert_array_equal(dx, [3])
    assert dw.dtype == weight_dtype and dx.dtype == events.dtype
    assert all(old is new for old, new in zip(before, jax.tree.leaves(layout)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("event_unit", [u.UNITLESS, u.mV])
def test_public_quantity_delivery_normalizes_and_restores_units(backend, event_unit):
    source = brainevent.CSR((jnp.array([2.], jnp.float16) * u.mS,
                            jnp.array([0], jnp.int32), jnp.array([0, 1], jnp.int32)),
                           shape=(1, 1), backend=backend)
    layout = source.delay_layout(np.array([1], np.int32), 3)
    unit = u.mS * event_unit
    ring = jnp.ones((3, 1), jnp.float32) * unit
    events = jnp.array([-2.], jnp.float64) * event_unit
    result = jax.jit(lambda r, x: csr_delay_deliver_layout(r, x, layout, 0, weighted=True))(ring, events)
    assert u.get_unit(result) == unit
    assert result.dtype == jnp.float32
    np.testing.assert_array_equal(u.get_mantissa(result), [[1], [-3], [1]])
    assert u.get_unit(layout.data) == u.mS
    np.testing.assert_array_equal(u.get_mantissa(layout.ring), np.zeros((3, 1)))


def test_csr_delay_layout_isolated_from_reused_source_host_pointer():
    storage = np.empty(72, np.uint8)
    offset = (-storage.ctypes.data) % 64
    pointer = storage[offset:offset + 8].view(np.int32)
    pointer[:] = [0, 3]
    source = brainevent.CSR((np.array([2.], np.float32), np.array([1, 0, 2], np.int32), pointer),
                           shape=(1, 3), backend="jax_raw")
    layout = source.delay_layout(np.ones(3, np.int32), 2)
    pointer[:] = 7
    np.testing.assert_array_equal(layout.indptr, [0, 3])
    result = csr_delay_deliver_layout(jnp.zeros((2, 3), jnp.float32), jnp.array([True]), layout, 0)
    np.testing.assert_array_equal(result, [[0, 0, 0], [2, 2, 2]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_public_quantity_delivery_preserves_ring_scale_and_rejects_incompatible_units(backend):
    source = brainevent.CSR((jnp.array([2.], jnp.float32) * u.mV,
                            jnp.array([0], jnp.int32), jnp.array([0, 1], jnp.int32)),
                           shape=(1, 1), backend=backend)
    layout = source.delay_layout(np.array([1], np.int32), 3)
    events = jnp.array([2.], jnp.float32)
    result = csr_delay_deliver_layout(jnp.full((3, 1), .001, jnp.float32) * u.volt,
                                     events, layout, 0, weighted=True)
    assert u.get_unit(result) == u.volt
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(u.get_mantissa(result), [[.001], [.005], [.001]], rtol=1e-6)
    with pytest.raises(u.UnitMismatchError):
        csr_delay_deliver_layout(jnp.zeros((3, 1), jnp.float32) * u.ms,
                                 events, layout, 0, weighted=True)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('weight_unit,event_unit,ring_unit', [
    pytest.param(u.mV, u.UNITLESS, u.volt, id='voltage'),
    pytest.param(u.mS, u.mV, u.mA, id='compound'),
])
def test_quantity_delivery_ring_scale_gradients(backend, weighted, weight_unit, event_unit, ring_unit):
    layout = delaycsr(
        (np.array([2], np.float64) * weight_unit, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        shape=(1, 1), edge_delays=np.ones(1, np.int32), buffer_slots=3, backend=backend,
    )
    leaves = jax.tree.leaves(layout)
    weights = u.get_mantissa(layout.data)
    ring = jnp.full((3, 1), .1, jnp.float32)
    events = jnp.array([2], jnp.float32)

    def deliver(w, r, x):
        return layout.with_data(w * weight_unit).csr_delay_deliver_layout(
            r * ring_unit, x * event_unit, 0, weighted=weighted)

    def values(w, r, x):
        return u.get_mantissa(deliver(w, r, x))

    _, (dw, dr, dx) = jax.jit(jax.value_and_grad(
        lambda w, r, x: values(w, r, x).sum(), argnums=(0, 1, 2)))(weights, ring, events)
    np.testing.assert_allclose(dw, [.002 if weighted else .001], rtol=1e-6)
    np.testing.assert_array_equal(dr, jnp.ones_like(ring))
    np.testing.assert_allclose(dx, [.002], rtol=1e-6)
    assert dw.dtype == weights.dtype and dx.dtype == events.dtype
    result, tangent = jax.jvp(values, (weights, ring, events),
                              (jnp.ones_like(weights), jnp.ones_like(ring), jnp.ones_like(events)))
    np.testing.assert_allclose(result, [[.1], [.104 if weighted else .102], [.1]], rtol=1e-6)
    np.testing.assert_allclose(tangent, [[1], [1.004 if weighted else 1.003], [1]], rtol=1e-6)
    quantity = deliver(weights, ring, events)
    assert u.get_unit(quantity) == ring_unit
    assert quantity.dtype == ring.dtype
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(layout)))
    np.testing.assert_array_equal(ring, jnp.full((3, 1), .1, jnp.float32))


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
def test_quantity_delivery_ring_scale_keeps_inactive_half_ring_finite(backend, weighted):
    layout = delaycsr(
        (np.array([2], np.float16) * u.mV, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        shape=(1, 1), edge_delays=np.ones(1, np.int32), backend=backend,
    )
    ring = jnp.full((2, 1), 100, jnp.float16) * u.volt
    result = jax.jit(lambda r: layout.csr_delay_deliver_layout(
        r, jnp.zeros(1, jnp.float16), 0, weighted=weighted))(ring)
    # Converting the whole buffer to mV would overflow float16 before delivery.
    np.testing.assert_array_equal(u.get_mantissa(result), [[100], [100]])
    assert u.get_unit(result) == u.volt
    assert result.dtype == jnp.float16


@pytest.mark.parametrize('backend', BACKENDS[1:])
def test_quantity_delivery_ring_scale_keeps_binary_event_activity(backend):
    layout = delaycsr(
        (np.array([2], np.float16) * u.mV, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        shape=(1, 1), edge_delays=np.ones(1, np.int32), backend=backend,
    )
    tiny = jnp.array([1e-5], jnp.float16)
    assert bool(tiny[0] != 0)
    np.testing.assert_array_equal(tiny * jnp.float16(.001), 0)
    result = jax.jit(lambda events: layout.csr_delay_deliver_layout(
        jnp.zeros((2, 1), jnp.float16) * u.volt, events, 0))(tiny)
    assert u.get_unit(result) == u.volt
    np.testing.assert_allclose(u.get_mantissa(result), [[0], [.002]], rtol=1e-3)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('ring_unit', [u.mV / u.volt, u.UNITLESS], ids=['scaled', 'unitless'])
def test_scaled_dimensionless_quantity_ring_keeps_pytree_for_delivery_and_clear(backend, ring_unit):
    layout = _deliver_layout_make(backend)
    leaves = jax.tree.leaves(layout)
    ring = u.Quantity(jnp.ones((2, 1), jnp.float32), unit=ring_unit)
    result = jax.jit(lambda r: layout.csr_delay_deliver_layout(
        r, jnp.array([True]), 0))(ring)
    assert jax.tree.structure(result) == jax.tree.structure(ring)
    assert u.get_unit(result) == ring.unit
    contribution = float(u.Quantity(3, unit=u.UNITLESS).to_decimal(ring_unit))
    np.testing.assert_allclose(u.get_mantissa(result), [[1], [1 + contribution]], rtol=1e-6)

    def clear(r):
        local = copy(layout)
        local.ring = r
        return local.clear_head()

    cleared = jax.jit(clear)(ring)
    assert jax.tree.structure(cleared) == jax.tree.structure(ring)
    assert u.get_unit(cleared) == ring.unit
    np.testing.assert_array_equal(u.get_mantissa(cleared), [[0], [1]])
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(layout)))
    np.testing.assert_array_equal(u.get_mantissa(ring), 1)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weight,weight_unit,factor', [
    pytest.param(1e-5, u.mV, .001, id='tiny-mV-to-V'),
    pytest.param(.001, u.Mvolt, 1e6, id='small-MV-to-V'),
])
def test_quantity_ring_scale_converts_half_weights_at_ring_precision(backend, weight, weight_unit, factor):
    layout = delaycsr(
        (np.array([weight], np.float16) * weight_unit, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        shape=(1, 1), edge_delays=np.ones(1, np.int32), backend=backend,
    )
    ring = jnp.zeros((2, 1), jnp.float32) * u.volt
    result = jax.jit(lambda r: layout.csr_delay_deliver_layout(
        r, jnp.array([True]), 0))(ring)
    expected = float(np.float16(weight)) * factor
    np.testing.assert_allclose(u.get_mantissa(result), [[0], [expected]], rtol=1e-6)
    assert u.get_unit(result) == u.volt
    assert result.dtype == jnp.float32


# Native derivatives and transformations
# --------------------------------------

def _region_delay_transform_problem(weighted, homo=False, dtype=jnp.float32):
    columns = np.array([0, 2, 1, 1, 0, 2], np.int32)
    pointers = np.array([0, 3, 3, 6], np.int32)
    delays = np.array([0, 1, 3, 0, 1, 3], np.int32)
    data = np.array([.5] if homo else [2, -1, .5, 3, -2, .25], dtype)
    state = delaycsr((data, columns, pointers), edge_delays=delays, shape=(3, 3),
                     delay_steps=(0, 1, 2, 3), buffer_slots=5, sorted_by_delay=True,
                     weighted=weighted, backend="cuda_raw")
    row_ids = jnp.array([0, 0, 0, 2, 2, 2])
    columns, delays = jnp.asarray(columns), jnp.asarray(delays)

    def project(ring, events, weights, head):
        return cuda_step(
            weights, state.indices, state.indptr, events, state.delay_count,
            state.delay_off, state.row_task_count, head, ring, state.workspace, weighted=weighted)[0]

    def reference(ring, events, weights, head):
        if weighted:
            factors = events
        else:
            # The main library's binary straight-through convention.
            factors = ((events != 0).astype(events.dtype)
                       + events - jax.lax.stop_gradient(events))
        return ring.at[(head[0] + delays) % 5, columns].add(
            jnp.broadcast_to(weights, (6,)) * factors[row_ids])

    return state, project, reference


@GPU
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homo", [False, True])
def test_jvp_and_reverse_mode(weighted, homo):
    state, project, reference = _region_delay_transform_problem(weighted, homo)
    ring = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 8
    events = jnp.array([-2, .5, 0], jnp.float32)
    head = jnp.array([4], jnp.int32)
    ct = jnp.arange(1, 16, dtype=jnp.float32).reshape(5, 3) / 4
    args = (ring, events, state.data)
    tangents = tuple(jnp.ones_like(x) / 4 for x in args)
    actual = jax.jit(lambda *xs: jax.jvp(lambda *ys: project(*ys, head), xs, tangents))(*args)
    expected = jax.jvp(lambda *ys: reference(*ys, head), args, tangents)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    grad = jax.jit(jax.grad(lambda *xs: (project(*xs, head) * ct).sum(), argnums=(0, 1, 2)))(*args)
    want = jax.grad(lambda *xs: (reference(*xs, head) * ct).sum(), argnums=(0, 1, 2))(*args)
    for got, expected_grad in zip(grad, want):
        np.testing.assert_allclose(got, expected_grad, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(grad[0], ct)
    assert grad[1][2] != 0  # An inactive event still has the straight-through/weighted derivative.


@GPU
@pytest.mark.parametrize("operand", [0, 1, 2])
def test_registered_transpose(operand):
    state, project, reference = _region_delay_transform_problem(True)
    args = [jnp.ones((5, 3), jnp.float32), jnp.array([-2, 0, .5], jnp.float32), state.data]
    head = jnp.array([0], jnp.int32)
    def linear(fn, x):
        inputs = args.copy()
        inputs[operand] = x
        return fn(*inputs, head)

    ct = jnp.arange(15, dtype=jnp.float32).reshape(5, 3)
    got = jax.jit(lambda x: jax.linear_transpose(lambda y: linear(project, y), x)(ct))(args[operand])
    want = jax.linear_transpose(lambda x: linear(reference, x), args[operand])(ct)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-6, atol=1e-6)


@GPU
@pytest.mark.parametrize("weighted", [False, True])
def test_vmap_nonleading_axes_and_grad(weighted):
    state, project, reference = _region_delay_transform_problem(weighted)
    rings = jnp.arange(30, dtype=jnp.float32).reshape(5, 3, 2)
    events = jnp.array([[-2, 0], [.5, 1], [0, 3]], jnp.float32)
    weights = jnp.stack([state.data, state.data * 2])
    heads = jnp.array([[0], [4]], jnp.int32)
    axes = (2, 1, 0, 0)
    mapped = jax.vmap(project, in_axes=axes)
    expected = jax.vmap(reference, in_axes=axes)
    np.testing.assert_allclose(jax.jit(mapped)(rings, events, weights, heads),
                               expected(rings, events, weights, heads))
    loss = lambda fn, x: (fn(rings, x, weights, heads) ** 2).sum()
    got = jax.jit(jax.grad(lambda x: loss(mapped, x)))(events)
    want = jax.grad(lambda x: loss(expected, x))(events)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@GPU
def test_class_with_data_gradient_and_batched_state():
    state, _, reference = _region_delay_transform_problem(True)
    events = jnp.array([[-2, 0, .5], [1, 0, 3]], jnp.float32)
    times = jnp.array([4, 5], jnp.int32)

    def step(w, x, t):
        spike = w.forward(x, t)
        return w, spike

    batched, spikes = jax.jit(jax.vmap(step, in_axes=(None, 0, 0)))(state, events, times)
    for i in range(2):
        expected = reference(state.ring.at[times[i] % state.L].set(0), events[i], state.data,
                             (times[i] % state.L).reshape(1))
        np.testing.assert_array_equal(batched.ring[i], expected)
        np.testing.assert_array_equal(spikes[i], expected[(times[i] + 1) % state.L])
    np.testing.assert_array_equal(state.ring, 0)

    def loss(w, weights):
        w = w.with_data(weights)
        w, outputs = jax.lax.scan(lambda carry, xt: step(carry, *xt), w, (events, times))
        return (outputs ** 2).sum() + (w.ring ** 2).sum()

    def expected_loss(weights):
        ring, total = state.ring, jnp.float32(0)
        for i in range(2):
            ring = reference(ring.at[times[i] % state.L].set(0), events[i], weights,
                             (times[i] % state.L).reshape(1))
            total += (ring[(times[i] + 1) % state.L] ** 2).sum()
        return total + (ring ** 2).sum()

    got = jax.jit(jax.grad(loss, argnums=1))(state, state.data)
    np.testing.assert_allclose(got, jax.grad(expected_loss)(state.data), rtol=1e-6, atol=1e-6)


@GPU
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_binary_event_tangent_uses_weight_dtype(dtype):
    state, project, _ = _region_delay_transform_problem(False, dtype=dtype)
    x = jnp.array([-2, 0, 0], jnp.float32)
    fn = lambda e: project(state.ring, e, state.data, jnp.array([4], jnp.int32))
    _, dot = jax.jit(lambda e: jax.jvp(fn, (e,), (jnp.ones_like(e),)))(x)
    assert dot.dtype == dtype
    grad = jax.jit(jax.grad(lambda e: fn(e).sum()))(x)
    assert grad.dtype == jnp.float32
    np.testing.assert_array_equal(grad, [1.5, 0, 1.25])


@GPU
@pytest.mark.parametrize("weighted", [False, True])
def test_event_tangent_populates_long_row_task_value(weighted):
    nnz = 8194
    state = delaycsr((np.array([.5], np.float32), np.zeros(nnz, np.int32),
                      np.array([0, nnz], np.int32)), shape=(1, 1),
                     edge_delays=np.repeat([0, 1], nnz // 2), buffer_slots=3,
                     weighted=weighted, sorted_by_delay=True)
    assert state.capacity > 0

    def project(x):
        return cuda_step(
            state.data, state.indices, state.indptr, x, state.delay_count,
            state.delay_off, state.row_task_count, jnp.array([2], jnp.int32),
            state.ring, state.workspace, weighted=weighted)[0]

    x = jnp.array([-2], jnp.float32)
    _, dot = jax.jit(lambda e: jax.jvp(project, (e,), (jnp.ones_like(e) / 4,)))(x)
    np.testing.assert_array_equal(dot, [[512.125], [0], [512.125]])
    np.testing.assert_array_equal(jax.jit(jax.grad(lambda e: project(e).sum()))(x), [4097])


@GPU
@pytest.mark.parametrize("rows,cols", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_empty_graph_gradients(rows, cols):
    state = delaycsr((np.array([1], np.float32), np.array([], np.int32),
                      np.zeros(rows + 1, np.int32)), shape=(rows, cols),
                     edge_delays=np.array([], np.int32), buffer_slots=3,
                     weighted=True, backend="cuda_raw")

    def loss(w, weights, events, ring):
        w = w.with_data(weights)
        w.ring = ring
        w.forward(events, 2)
        return w.ring.sum()

    gw, gx, gr = jax.jit(jax.grad(loss, argnums=(1, 2, 3)))(
        state, state.data, jnp.zeros(rows, jnp.float32), state.ring)
    np.testing.assert_array_equal(gw, 0)
    np.testing.assert_array_equal(gx, 0)
    expected = np.ones((3, cols), np.float32)
    expected[2] = 0
    np.testing.assert_array_equal(gr, expected)


@GPU
def test_workspace_is_not_differentiable():
    state = delaycsr((np.array([.5], np.float32), np.zeros(129, np.int32),
                      np.array([0, 129], np.int32)), shape=(1, 1),
                     edge_delays=np.ones(129, np.int32), weighted=True)
    assert state.capacity == 1

    def loss(task_value, events):
        workspace = state.workspace._replace(task_value=task_value)
        ring, result = cuda_step(
            state.data, state.indices, state.indptr, events,
            state.delay_count, state.delay_off, state.row_task_count,
            jnp.array([0], jnp.int32), state.ring, workspace, weighted=True)
        return result.task_value.sum()

    gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(state.workspace.task_value, jnp.ones(1, jnp.float32))
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, 0)


# Opt-in structures beyond INT32_MAX
# ----------------------------------

@pytest.mark.slow
@GPU
def test_edges_beyond_int32(request) -> None:
    if not request.config.getoption("--run-large-delay-tests"):
        pytest.skip("requires --run-large-delay-tests (about 20-30 GB GPU memory)")
    cols = 1021
    indptr, tail = large_structure(cols)
    nnz = int(indptr[-1])
    indices = large_indices(tail)
    weights = jnp.ones(nnz, jnp.float32)  # ring entries are exact edge counts
    lengths = np.diff(indptr)
    # One delay segment (delay 0): rows need no sorting, so the metadata is built directly.
    delay_count = lengths.reshape(-1, 1)
    delay_off = np.zeros(1, np.int32)
    tasks = np.where(lengths > TPR_THRESHOLD, (lengths + TASK_NNZ - 1) // TASK_NNZ, 0).astype(np.int32)
    capacity = int(tasks.sum())
    work = HybridWorkspace(jnp.zeros(capacity, jnp.int64), jnp.zeros(capacity, jnp.int64),
                           jnp.zeros(capacity, jnp.int32), jnp.zeros(2, jnp.int32))
    events = np.array([0, 1, 1, -2, np.nan], np.float32)  # the leading row stays inactive
    ring, work = cuda_step(weights, indices, jnp.asarray(indptr), jnp.asarray(events),
                                      jnp.asarray(delay_count), jnp.asarray(delay_off), jnp.asarray(tasks),
                                      jnp.array([0], jnp.int32), jnp.zeros((1, cols), jnp.float32), work)
    assert np.asarray(work.status)[1] == 0
    np.testing.assert_array_equal(np.asarray(ring)[0], np.bincount(tail, minlength=cols).astype(np.float32))

    with pytest.raises(KernelCompilationError, match="int32 indptr"):
        cuda_step(weights, indices, jnp.asarray(indptr.astype(np.int32)), jnp.asarray(events),
                             jnp.asarray(delay_count.astype(np.int32)), jnp.asarray(delay_off), jnp.asarray(tasks),
                             jnp.array([0], jnp.int32), ring, work)
