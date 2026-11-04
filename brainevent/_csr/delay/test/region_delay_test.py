"""CUDA binary and weighted delivery against original edges, dtypes and task queues."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import pytest

from brainevent import KernelCompilationError
from brainevent._csr.delay.region_delay import HybridWorkspace, WeightedHybridWorkspace, delaycsr
from brainevent._csr.delay.test._helpers import GPU, cuda_step
from brainevent._csr.delay.test._reference import (
    EVENT_DTYPES, HYBRID_ROW_LENGTHS, INDEX_DTYPES, VALUE_DTYPES, active_mask,
    assert_close, assert_same_bits, csr_from_lengths, dtype_name, numpy_project_step,
    random_events, random_values,
)

COLUMNS = 3000
DELAYS = (0, 1, 3, 4)
SLOTS = 6
# Each ring element receives a few hundred atomic contributions per step.
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
@pytest.mark.parametrize("field,error", [("row_task_count", 1), ("delay_count", 3)])
def test_hybrid_bad_metadata_is_reported_and_next_call_resets_status(field: str, error: int) -> None:
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
    assert int(work.status[1]) == error
    np.testing.assert_array_equal(ring, 1)
    _, work = project(**arrays, events=jnp.zeros(1, bool), head=jnp.zeros(1, jnp.int32), buffer=ring, workspace=work)
    np.testing.assert_array_equal(work.status, [0, 0])


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
    with pytest.raises(KernelCompilationError, match="weights and ring of one dtype"):
        cuda_step(**arrays, events=events, head=head, buffer=jnp.zeros((SLOTS, COLUMNS), jnp.float64),
                  workspace=_workspace(data))
    with pytest.raises(KernelCompilationError, match="binary events must be bool or float32"):
        cuda_step(**arrays, events=events.astype(jnp.int32), head=head,
                  buffer=jnp.zeros((SLOTS, COLUMNS), jnp.float32), workspace=_workspace(data))


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
        data._allocate_workspace()
    return data, (weights, indices, ptr, delays), events


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
    with pytest.raises(KernelCompilationError, match="events"):
        cuda_step(**arrays, events=events.astype(jnp.bool_), head=head,
                  buffer=ring, workspace=work, weighted=True)
    with pytest.raises(KernelCompilationError, match="WeightedHybridWorkspace"):
        cuda_step(**arrays, events=events, head=head, buffer=ring,
                  workspace=HybridWorkspace(work.task_begin, work.task_end, work.task_out_off, work.status),
                  weighted=True)
    for task_value in (jnp.zeros(data.capacity, jnp.float64), jnp.zeros(data.capacity + 1, jnp.float32)):
        with pytest.raises(KernelCompilationError, match="task_value"):
            cuda_step(**arrays, events=events, head=head, buffer=ring,
                      workspace=work._replace(task_value=task_value), weighted=True)


def test_weighted_preparation_rejects_binary_events():
    data, _, event_columns = _weighted_problem(jnp.float32)
    with pytest.raises(TypeError, match="events"):
        data.forward(jnp.asarray(event_columns[:, 0], jnp.bool_), 0)
