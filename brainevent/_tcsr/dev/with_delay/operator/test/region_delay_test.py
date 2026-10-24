"""Check the three CUDA operators against original edges and arrival times."""

from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import pytest

from brainevent._tcsr.dev.with_delay.operator.region_delay import (
    DelayInputs, HybridWorkspace, SplitCSR, prepare_delay_inputs,
    project_region_delay, project_region_delay_cusparse, split_region_delay,
)


@pytest.fixture(autouse=True)
def _gpu_x64() -> Iterator[None]:
    if jax.default_backend() != "gpu":
        pytest.skip("requires CUDA")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _arrays(data: DelayInputs) -> dict[str, jax.Array]:
    return {name: jnp.asarray(getattr(data, name)) for name in
            ("weights", "indices", "indptr", "delay_count", "delay_off", "row_task_count")}


def _workspace(data: DelayInputs, offset_dtype: Any = jnp.int32) -> HybridWorkspace:
    return HybridWorkspace(
        jnp.zeros(data.task_capacity, data.indptr.dtype), jnp.zeros(data.task_capacity, data.indptr.dtype),
        jnp.zeros(data.task_capacity, offset_dtype), jnp.array([17, 1], jnp.int32),
    )


def _split_reference(weights: NDArray[Any], columns: NDArray[Any], ptr: NDArray[Any],
                     delays: NDArray[Any], steps: tuple[int, ...]) -> tuple[NDArray[Any], ...]:
    pointers = np.zeros((len(steps), len(ptr)), np.int64)
    packed_columns, packed_values = [], []
    values = np.broadcast_to(weights, columns.shape)
    for segment, delay in enumerate(steps):
        for row in range(len(ptr) - 1):
            edges = np.arange(ptr[row], ptr[row + 1])
            edges = edges[delays[edges] == delay]
            edges = edges[np.argsort(columns[edges], kind="stable")]
            pointers[segment, row + 1] = pointers[segment, row] + len(edges)
            packed_columns.extend(columns[edges])
            packed_values.extend(values[edges])
    return (pointers, np.r_[0, np.cumsum(pointers[:, -1])],
            np.asarray(packed_columns), np.asarray(packed_values))


def _arrivals(weights: NDArray[Any], columns: NDArray[Any], ptr: NDArray[Any],
              delays: NDArray[Any], events: NDArray[Any], n: int, slots: int) -> tuple[NDArray[Any], NDArray[Any]]:
    future = np.zeros((events.shape[1] + slots, n), np.float64)
    for time in range(events.shape[1]):
        for row in np.flatnonzero(events[:, time]):
            first, last = int(ptr[row]), int(ptr[row + 1])
            values = weights[0] if weights.size == 1 else weights[first:last]
            np.add.at(future, (time + delays[first:last], columns[first:last]), values)
    ring = np.zeros((slots, n), np.float64)
    for time in range(events.shape[1], len(future)):
        ring[time % slots] += future[time]
    return future[:events.shape[1]], ring


def _numpy_project_step(
    weights: NDArray[Any], columns: NDArray[Any], ptr: NDArray[Any],
    delays: NDArray[Any], events: NDArray[Any], head: int, ring: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project one event column from original edges without consuming the ring."""
    result = ring.copy()
    for row in np.flatnonzero(events):
        first, last = int(ptr[row]), int(ptr[row + 1])
        slots = (head + delays[first:last]) % len(ring)
        values = weights[0] if weights.size == 1 else weights[first:last]
        np.add.at(result, (slots, columns[first:last]), values)
    return result


def test_numpy_step_matches_hand_calculated_ring() -> None:
    """Validate duplicate edges, binary activity and wrapped arrivals by hand."""
    columns, ptr = np.array([1, 1, 2, 0]), np.array([0, 3, 4])
    delays, events = np.array([0, 0, 1, 2]), np.array([-4, 0], np.float32)
    initial = np.ones((3, 3), np.float64)
    for weights, expected in (
        (np.array([2, 3, 5, 7], np.float32), [[1, 1, 6], [1, 1, 1], [1, 6, 1]]),
        (np.array([0.5], np.float32), [[1, 1, 1.5], [1, 1, 1], [1, 2, 1]]),
    ):
        actual = _numpy_project_step(weights, columns, ptr, delays, events, 2, initial)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(initial, 1)
        np.testing.assert_array_equal(
            _numpy_project_step(weights, columns, ptr, delays, np.zeros(2), 0, actual), actual)


def _run(data: DelayInputs, events: NDArray[Any], arrays: dict[str, jax.Array],
         split: SplitCSR | None) -> tuple[NDArray[Any], NDArray[Any]]:
    ring = jnp.zeros((data.buffer_slots, data.shape[1]), jnp.float32)
    work, scratch = _workspace(data, arrays["delay_off"].dtype), jnp.zeros(data.shape[1], jnp.float32)
    hybrid, cusparse = jax.jit(project_region_delay), jax.jit(project_region_delay_cusparse)
    outputs = []
    for time in range(events.shape[1]):
        slot = time % data.buffer_slots
        head, event = jnp.array([slot], jnp.int32), jnp.asarray(events[:, time])
        if split is None:
            ring, work = hybrid(**arrays, events=event, head=head, buffer=ring, workspace=work)
            expected_tasks = int(data.row_task_count[events[:, time] != 0].sum())
            np.testing.assert_array_equal(work.status, [expected_tasks, 0])
        else:
            ring, scratch = cusparse(split, event, arrays["delay_off"], head, ring, scratch)
        outputs.append(np.asarray(ring)[slot].copy())
        ring = ring.at[slot].set(0)  # Test-owned consume/clear; head advances once per event column.
    return np.stack(outputs), np.asarray(ring)


def _small(delays: tuple[int, ...] = (0, 2, 0, 2)) -> DelayInputs:
    return prepare_delay_inputs(
        np.array([2, 3, 5, 7], np.float32), np.array([3, 0, 2, 1], np.int32),
        np.array([0, 4, 4], np.int32), np.asarray(delays), np.ones((2, 3), bool),
        columns=4, delay_steps=(0, 1, 2),
    )


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("pointer_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("event_dtype", [np.bool_, np.float32])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("wide_offsets", [False, True])
def test_split_and_both_projections_match_original_edges(
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
    data = prepare_delay_inputs(weights, columns, ptr, delays, events,
                               columns=4099, delay_steps=(0, 1, 2, 5), buffer_slots=8)
    arrays = _arrays(data)
    if wide_offsets:
        arrays["delay_off"] = arrays["delay_off"].astype(jnp.int64)
    split = split_region_delay(*(arrays[name] for name in ("weights", "indices", "indptr", "delay_count")))
    assert split.sub_indptr.dtype == split.sub_base.dtype == split.sub_indices.dtype == jnp.int32
    assert split.sub_values.dtype == jnp.float32
    for actual, expected in zip(split, _split_reference(weights, columns, ptr, delays, data.delay_steps)):
        np.testing.assert_array_equal(actual, expected)
    expected_outputs = _arrivals(weights, columns, ptr, delays, events, 4099, 8)
    for prepared_split in (None, split):
        for observed, expected in zip(_run(data, events, arrays, prepared_split), expected_outputs):
            np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-5)


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

    data = prepare_delay_inputs(weights, columns, ptr, delays, events, columns=n, delay_steps=tuple(range(k)))
    arrays = _arrays(data)
    # Split once before the time loop; both operators reuse one uploaded [N,T] matrix.
    split = split_region_delay(*(arrays[name] for name in ("weights", "indices", "indptr", "delay_count")))
    assert np.count_nonzero(np.diff(np.asarray(split.sub_base))) == k - (empty_segment is not None)
    device_events = jnp.asarray(events)
    hybrid_ring, sparse_ring = jnp.zeros((k, n), jnp.float32), jnp.zeros((k, n), jnp.float32)
    work, scratch = _workspace(data), jnp.zeros(n, jnp.float32)
    hybrid, cusparse = jax.jit(project_region_delay), jax.jit(project_region_delay_cusparse)
    numpy_ring = np.zeros((k, n), np.float64)
    numpy_outputs = []
    for time in range(duration_steps):
        slot = time % k
        numpy_ring = _numpy_project_step(weights, columns, ptr, delays, events[:, time], slot, numpy_ring)
        head, event = jnp.array([slot], jnp.int32), device_events[:, time]
        hybrid_ring, work = hybrid(
            **arrays, events=event, head=head, buffer=hybrid_ring, workspace=work)
        sparse_ring, scratch = cusparse(split, event, arrays["delay_off"], head, sparse_ring, scratch)
        expected_output = numpy_ring[slot].copy()
        for backend, ring in (("hybrid", hybrid_ring), ("cusparse", sparse_ring)):
            message = f"backend={backend}, step={time}, head={slot}, K={k}, T={duration_steps}"
            observed = np.asarray(ring)
            np.testing.assert_allclose(observed[slot], expected_output, rtol=2e-5, atol=2e-5,
                                       err_msg=message + ": current output")
            np.testing.assert_allclose(observed, numpy_ring, rtol=2e-5, atol=2e-5,
                                       err_msg=message + ": complete projected ring")
        expected_tasks = int(data.row_task_count[events[:, time] != 0].sum())
        np.testing.assert_array_equal(work.status, [expected_tasks, 0], err_msg=f"hybrid step={time}")
        numpy_outputs.append(expected_output)
        numpy_ring[slot] = 0
        hybrid_ring, sparse_ring = hybrid_ring.at[slot].set(0), sparse_ring.at[slot].set(0)

    # Cross-check the stepwise NumPy implementation against absolute arrival times.
    outputs, remaining = _arrivals(weights, columns, ptr, delays, events, n, k)
    np.testing.assert_allclose(numpy_outputs, outputs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(numpy_ring, remaining, rtol=1e-12, atol=1e-12)
    for backend, ring in (("hybrid", hybrid_ring), ("cusparse", sparse_ring)):
        np.testing.assert_allclose(ring, remaining, rtol=2e-5, atol=2e-5,
                                   err_msg=f"backend={backend}: final remaining ring")


@pytest.mark.parametrize("sub_dtype", [jnp.int32, jnp.int64])
@pytest.mark.parametrize("offset_dtype", [jnp.int32, jnp.int64])
@pytest.mark.parametrize("head", [-1, 0, 2, 3])
def test_one_step_covers_all_segments_and_preserves_existing_ring(sub_dtype: Any, offset_dtype: Any, head: int) -> None:
    """Check one FFI call, empty segments, wrapping and invalid-head behavior."""
    data = _small()
    arrays = _arrays(data)
    arrays["delay_off"] = arrays["delay_off"].astype(offset_dtype)
    split = split_region_delay(*(arrays[name] for name in ("weights", "indices", "indptr", "delay_count")))
    split = SplitCSR(split.sub_indptr.astype(sub_dtype), split.sub_base.astype(sub_dtype),
                     split.sub_indices.astype(sub_dtype), split.sub_values)
    ring, event, position = jnp.ones((3, 4), jnp.float32), jnp.array([-2, 0], jnp.float32), jnp.array([head], jnp.int32)
    args = (split, event, arrays["delay_off"], position, ring, jnp.zeros(4, jnp.float32))
    traced = jax.make_jaxpr(project_region_delay_cusparse)(*args)
    assert sum(eq.primitive.name == "ffi_call" for eq in traced.jaxpr.eqns) == 1
    sparse_ring, _ = jax.jit(project_region_delay_cusparse)(*args)
    hybrid_ring, work = jax.jit(project_region_delay)(
        **arrays, events=event, head=position, buffer=ring, workspace=_workspace(data, offset_dtype))
    expected = np.ones((3, 4), np.float32)
    if 0 <= head < 3:
        expected[head, 2:] += [5, 2]
        expected[(head + 2) % 3, :2] += [3, 7]
    np.testing.assert_array_equal(work.status, [0, 0 if 0 <= head < 3 else 2])
    np.testing.assert_array_equal(sparse_ring, expected)
    np.testing.assert_array_equal(hybrid_ring, expected)


@pytest.mark.parametrize("rows,columns", [(0, 0), (0, 7), (3, 0), (3, 7)])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_empty_graphs(rows: int, columns: int, homogeneous: bool) -> None:
    """Handle zero dimensions and empty segments while resetting hybrid status."""
    events = np.ones((rows, 4), bool)
    data = prepare_delay_inputs(np.ones(1 if homogeneous else 0, np.float32), np.array([], np.int32),
        np.zeros(rows + 1, np.int32), np.array([], np.int32), events, columns=columns, delay_steps=(0, 3))
    arrays = _arrays(data)
    split = split_region_delay(*(arrays[name] for name in ("weights", "indices", "indptr", "delay_count")))
    np.testing.assert_array_equal(split.sub_indptr, np.zeros((2, rows + 1)))
    np.testing.assert_array_equal(split.sub_base, [0, 0, 0])
    assert split.sub_indices.size == split.sub_values.size == 0
    for prepared_split in (None, split):
        current, ring = _run(data, events, arrays, prepared_split)
        np.testing.assert_array_equal(current, np.zeros((4, columns)))
        np.testing.assert_array_equal(ring, np.zeros((4, columns)))


def test_single_slot_and_repeated_projection_sequences() -> None:
    """Consume zero-delay contributions and reuse the same split for new runs."""
    weights, columns, ptr = np.array([2, -3], np.float32), np.array([1, 0], np.int32), np.array([0, 1, 2], np.int32)
    delays, events = np.zeros(2, int), np.array([[1, 0, 1], [0, 1, 1]], bool)
    data = prepare_delay_inputs(weights, columns, ptr, delays, events, columns=2)
    arrays = _arrays(data)
    split = split_region_delay(*(arrays[name] for name in ("weights", "indices", "indptr", "delay_count")))
    for prepared_split in (None, split, split):
        current, ring = _run(data, events, arrays, prepared_split)
        np.testing.assert_array_equal(current, [[0, 2], [-3, 0], [-3, 2]])
        np.testing.assert_array_equal(ring, 0)


def test_cusparse_compiled_step_accepts_new_split_boundaries() -> None:
    """Pass a different split to the same compiled step without rebuilding it."""
    data = _small()
    arrays = _arrays(data)
    split = split_region_delay(*(arrays[name] for name in ("weights", "indices", "indptr", "delay_count")))
    args = (jnp.ones(2, bool), jnp.asarray(data.delay_off), jnp.zeros(1, jnp.int32),
            jnp.zeros((3, 4), jnp.float32), jnp.zeros(4, jnp.float32))
    executable = jax.jit(project_region_delay_cusparse).lower(split, *args).compile()
    changed = _arrays(_small(delays=(0, 2, 0, 0)))
    different = split_region_delay(*(changed[name] for name in ("weights", "indices", "indptr", "delay_count")))
    np.testing.assert_array_equal(different.sub_base, [0, 3, 3, 4])
    ring, _ = executable(different, *args)
    np.testing.assert_array_equal(ring, [[0, 7, 5, 2], [0, 0, 0, 0], [3, 0, 0, 0]])


@pytest.mark.parametrize("bases", [[1, 2, 2, 4], [0, 2, 2, 3], [0, -1, 2, 4], [0, 5, 2, 4]])
def test_cusparse_rejects_malformed_split_boundaries(bases: list[int]) -> None:
    """Surface CUDA boundary checks through both JAX 0.8 and newer wrappers."""
    data = _small()
    arrays = _arrays(data)
    split = split_region_delay(*(arrays[name] for name in ("weights", "indices", "indptr", "delay_count")))
    split = split._replace(sub_base=jnp.asarray(bases, jnp.int32))
    with pytest.raises((ValueError, RuntimeError), match="split boundaries"):
        jax.block_until_ready(jax.jit(project_region_delay_cusparse)(
            split, jnp.ones(2, bool), arrays["delay_off"], jnp.zeros(1, jnp.int32),
            jnp.zeros((3, 4), jnp.float32), jnp.zeros(4, jnp.float32)))


def test_hybrid_concurrent_rows_have_complete_task_coverage() -> None:
    """Check task reservations across multiple extract blocks without assuming order."""
    rows, width = 259, 135
    data = prepare_delay_inputs(np.ones(1, np.float32), np.tile(np.arange(width, dtype=np.int32), rows),
        np.arange(rows + 1, dtype=np.int32) * width, np.tile(np.repeat([0, 2, 5], 45), rows),
        np.ones((rows, 1), bool), columns=width)
    ring, work = jax.jit(project_region_delay)(
        **_arrays(data), events=jnp.ones(rows, bool), head=jnp.zeros(1, jnp.int32),
        buffer=jnp.zeros((6, width), jnp.float32), workspace=_workspace(data))
    np.testing.assert_array_equal(work.status, [rows * 3, 0])
    tasks = sorted(zip(*(np.asarray(a).tolist() for a in work[:3])))
    assert tasks == [(row * width + k * 45, row * width + (k + 1) * 45, delay * width)
                     for row in range(rows) for k, delay in enumerate((0, 2, 5))]
    expected = np.zeros((6, width), np.float32)
    expected[0, :45], expected[2, 45:90], expected[5, 90:] = rows, rows, rows
    np.testing.assert_array_equal(ring, expected)


@pytest.mark.parametrize("field,error", [("row_task_count", 1), ("delay_count", 3)])
def test_hybrid_bad_metadata_is_reported_and_next_call_resets_status(field: str, error: int) -> None:
    """Reject unwritten long-row tasks and reset the error on the next step."""
    data = prepare_delay_inputs(np.ones(1, np.float32), np.arange(129, dtype=np.int32),
        np.array([0, 129], np.int32), np.zeros(129, int), np.ones((1, 2), bool), columns=129)
    arrays = _arrays(data)
    arrays[field] = jnp.zeros(1, jnp.int32) if field == "row_task_count" else jnp.array([[130]], jnp.int32)
    project = jax.jit(project_region_delay)
    ring, work = project(**arrays, events=jnp.ones(1, bool), head=jnp.zeros(1, jnp.int32),
                         buffer=jnp.ones((1, 129), jnp.float32), workspace=_workspace(data))
    assert int(work.status[1]) == error
    np.testing.assert_array_equal(ring, 1)
    _, work = project(**arrays, events=jnp.zeros(1, bool), head=jnp.zeros(1, jnp.int32), buffer=ring, workspace=work)
    np.testing.assert_array_equal(work.status, [0, 0])
