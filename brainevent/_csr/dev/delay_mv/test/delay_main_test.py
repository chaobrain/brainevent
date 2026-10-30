"""Test metadata, backend preparation and Python ownership of delay state."""

from collections.abc import Iterator
from typing import Any, cast

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR
from brainevent._event import BinaryArray

from brainevent._tcsr.delay_mv import binary_delay, delay_main
from brainevent._tcsr.delay_mv.delay_main import DelayMetadata, _DelayWorkspace, delay_tcsr, prepare_delay_metadata

_GPU = any(d.platform == "gpu" for d in jax.devices())


@pytest.fixture(autouse=True)
def _x64() -> Iterator[None]:
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _source(
    rows: list[list[int]], *, n_post: int = 3, kind: str = "heterogeneous",
    ptr_dtype: Any = np.int32,
) -> tuple[CSR, np.ndarray]:
    delays = np.asarray([d for row in rows for d in row], dtype=np.int32)
    ptr = np.asarray([0, *np.cumsum([len(row) for row in rows])], dtype=ptr_dtype)
    # Descending columns ensure construction cannot silently sort the source.
    indices = (np.arange(delays.size)[::-1] % max(n_post, 1)).astype(np.int32)
    weights = np.arange(1, delays.size + 1, dtype=np.float32)
    if kind == "shared":
        weights = np.array([2.0], dtype=np.float32)
    elif kind == "scalar":
        weights = np.array(2.0, dtype=np.float32)
    return CSR((weights, indices, ptr), shape=(len(rows), n_post), indptr_dtype=ptr_dtype), delays


@pytest.mark.parametrize("ptr_dtype", [np.int32, np.int64])
def test_metadata_for_segmented_rows(ptr_dtype: Any) -> None:
    """Count sparse delays while allowing the next row to restart at zero."""
    delays = np.array([0, 2, 2, 0, 5], dtype=np.int32)
    original = delays.copy()
    steps, counts, offsets = prepare_delay_metadata(
        delays, np.array([0, 3, 3, 5], dtype=ptr_dtype), n_post=10,
    )
    assert steps == (0, 2, 5)
    np.testing.assert_array_equal(counts, [[1, 2, 0], [0, 0, 0], [1, 0, 1]])
    np.testing.assert_array_equal(offsets, [0, 20, 50])
    np.testing.assert_array_equal(delays, original)
    assert counts.dtype == ptr_dtype
    assert offsets.dtype == np.int32


@pytest.mark.parametrize("rows,n_post", [(0, 3), (2, 0), (3, 4)])
def test_empty_graph(rows: int, n_post: int) -> None:
    """Keep one zero-delay segment and valid state for empty graphs."""
    source, delays = _source([[] for _ in range(rows)], n_post=n_post)
    op = delay_tcsr(source, delays)
    assert op.delay_steps == (0,)
    np.testing.assert_array_equal(op.delay_count, np.zeros((rows, 1), dtype=np.int32))
    np.testing.assert_array_equal(op.delay_off, [0])
    assert op.buffer.shape == (1, n_post)
    assert isinstance(op._workspace, _DelayWorkspace)
    assert op._workspace.task_capacity == 0
    assert op._workspace.task_out_off.shape == (0,)


@pytest.mark.parametrize("slots,expected_dtype", [(2**31 - 1, np.int32), (2**30, np.int64)])
def test_offsets_use_full_ring_size(slots: int, expected_dtype: Any) -> None:
    """Use int64 for a large ring even when its only delay offset is zero."""
    n_post = 1 if expected_dtype == np.int32 else 2
    _, _, offsets = prepare_delay_metadata(
        np.array([0], dtype=np.int32), np.array([0, 1], dtype=np.int32),
        n_post=n_post, buffer_slots=slots,
    )
    assert offsets.dtype == expected_dtype
    np.testing.assert_array_equal(offsets, [0])


@pytest.mark.parametrize("delays,ptr,error,match", [
    ([-1], [0, 1], ValueError, "nonnegative"),
    ([2**31 - 1], [0, 1], ValueError, "below INT32_MAX"),
    ([1.5], [0, 1], TypeError, "integer edge_delays"),
    ([True], [0, 1], TypeError, "integer edge_delays"),
    ([[0]], [0, 1], ValueError, "vectors"),
    ([0], [], ValueError, "nonempty"),
    ([0], [[0, 1]], ValueError, "vectors"),
    ([0], [1, 1], ValueError, "span"),
    ([0], [0, 0], ValueError, "span"),
    ([0], [0, -1, 1], ValueError, "span"),
    ([0], [0, 2, 1], ValueError, "nondecreasing"),
    ([2, 2, 0], [0, 3], ValueError, "within each CSR row"),
])
def test_invalid_metadata(delays: Any, ptr: Any, error: Any, match: str) -> None:
    """Reject metadata that would associate the wrong edges with a delay."""
    with pytest.raises(error, match=match):
        prepare_delay_metadata(delays, np.asarray(ptr, dtype=np.int32), n_post=2)


@pytest.mark.parametrize("ptr_dtype", [np.float32, np.uint32])
def test_invalid_pointer_dtype(ptr_dtype: Any) -> None:
    """Require signed index types supported by the CUDA ABI."""
    with pytest.raises(TypeError, match="int32/int64 indptr"):
        prepare_delay_metadata([0], np.array([0, 1], dtype=ptr_dtype), n_post=2)


@pytest.mark.parametrize("n_post", [-1, 2**31, 1.5, True])
def test_invalid_target_count(n_post: Any) -> None:
    """Reject target counts outside the kernel dimension contract."""
    with pytest.raises(ValueError, match="target counts"):
        prepare_delay_metadata([0], np.array([0, 1], dtype=np.int32), n_post=n_post)


@pytest.mark.parametrize("slots", [0, 2, -1, 2**31, 3.5, True])
def test_invalid_ring_length(slots: Any) -> None:
    """Require enough integer time slots to distinguish pending arrivals."""
    with pytest.raises(ValueError, match="buffer_slots"):
        prepare_delay_metadata([2], np.array([0, 1], dtype=np.int32),
                               n_post=2, buffer_slots=slots)


def test_ring_byte_overflow() -> None:
    """Reject an unaddressable ring before allocating metadata or state."""
    with pytest.raises(ValueError, match="byte size overflows"):
        prepare_delay_metadata([0], np.array([0, 1], dtype=np.int32),
                               n_post=2**31 - 1, buffer_slots=2**31 - 1)


@pytest.mark.parametrize("ptr_dtype,slots", [(np.int64, 1), (np.int32, 2**30)])
def test_x64_required_for_metadata(ptr_dtype: Any, slots: int) -> None:
    """Prevent silent narrowing of row pointers or ring offsets."""
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(RuntimeError, match="jax_enable_x64"):
        prepare_delay_metadata([0], np.array([0, 1], dtype=ptr_dtype),
                               n_post=2, buffer_slots=slots)


@pytest.mark.parametrize("kind", ["heterogeneous", "shared", "scalar"])
@pytest.mark.parametrize("ptr_dtype", [np.int32, np.int64])
def test_initial_state_preserves_source(kind: str, ptr_dtype: Any) -> None:
    """Allocate zero state without reordering weights, indices or pointers."""
    source, delays = _source([[0, 2, 2], [], [5]], kind=kind, ptr_dtype=ptr_dtype)
    op = delay_tcsr(source, delays, buffer_slots=8, binary_backend="jax")
    assert op.indices is source.indices and op.indptr is source.indptr
    assert isinstance(op._inputs, binary_delay._DelayCSRInputs)
    assert isinstance(op._inputs.indices, jax.Array) and isinstance(op._inputs.indptr, jax.Array)
    np.testing.assert_array_equal(op._inputs.indices, source.indices)
    np.testing.assert_array_equal(op._inputs.indptr, source.indptr)
    np.testing.assert_array_equal(op.data, source.data.reshape(-1))
    assert op.shape == source.shape and op.binary_backend == "jax"
    assert op.buffer_slots == 8 and op.buffer.dtype == np.float32
    np.testing.assert_array_equal(op.buffer, np.zeros((8, 3)))
    np.testing.assert_array_equal(op.head, [0])
    assert op.head.dtype == np.int32
    np.testing.assert_array_equal(op.row_task_count, [0, 0, 0])
    assert op.row_task_count.dtype == np.int32
    assert isinstance(op._workspace, _DelayWorkspace)
    for array in op._workspace[1:]:
        np.testing.assert_array_equal(array, np.zeros(array.shape))
    assert op._workspace.task_begin.dtype == ptr_dtype
    assert op._workspace.task_end.dtype == ptr_dtype
    assert op._workspace.task_out_off.dtype == op.delay_off.dtype
    assert op._workspace.status.shape == (2,) and op._workspace.status.dtype == np.int32


@pytest.mark.parametrize("row_delays,expected", [
    ([0] * 128, 0),
    ([0] * 129, 1),
    ([0] * 4095, 1),
    ([0] * 4096, 1),
    ([0] * 4097, 2),
    ([0] * 64 + [2] * 65, 2),
    ([0] * 100 + [1] * 5000 + [2] * 300, 4),
])
def test_segment_task_capacity(row_delays: list[int], expected: int) -> None:
    """Size tasks per segment after classifying the entire row."""
    source, delays = _source([row_delays, []])
    op = delay_tcsr(source, delays)
    np.testing.assert_array_equal(op.row_task_count, [expected, 0])
    assert isinstance(op._workspace, _DelayWorkspace)
    assert op._workspace.task_capacity == expected
    for array in op._workspace[1:4]:
        assert array.shape == (expected,)


def test_weight_units_are_retained() -> None:
    """Keep unit-bearing data intact for the backend wrapper."""
    source, delays = _source([[0, 1]])
    millivolt = getattr(u, "mV")
    source.data = source.data * millivolt
    op = delay_tcsr(source, delays)
    assert op.data is source.data
    assert u.get_unit(op.data) == millivolt


@pytest.mark.parametrize("field,value,error,match", [
    ("indices", np.array([[0]], dtype=np.int32), ValueError, "indices must be a vector"),
    ("indices", np.array([-1], dtype=np.int32), ValueError, "valid target columns"),
    ("indices", np.array([3], dtype=np.int32), ValueError, "valid target columns"),
    ("indices", np.array([0], dtype=np.float32), TypeError, "indices must be int32"),
    ("indptr", np.array([0], dtype=np.int32), ValueError, "indptr shape"),
    ("indptr", np.array([0, 0], dtype=np.int32), ValueError, "final offset"),
    ("data", np.array([1], dtype=np.float64), TypeError, "float32"),
    ("data", np.array([[1]], dtype=np.float32), ValueError, "weights must"),
    ("data", np.array([1, 2], dtype=np.float32), ValueError, "length nnz"),
    ("shape", (1, -1), ValueError, "source.shape"),
    ("shape", (1, 3, 4), ValueError, "source.shape"),
])
def test_invalid_csr(field: str, value: Any, error: Any, match: str) -> None:
    """Validate CSR objects whose fields were changed after construction."""
    source, delays = _source([[0]])
    setattr(source, field, value)
    with pytest.raises(error, match=match):
        delay_tcsr(source, delays)


def test_source_type_and_backend() -> None:
    """Reject unsupported source objects and retired backend names."""
    with pytest.raises(TypeError, match="PlainCSR"):
        delay_tcsr(None, [0])  # type: ignore[arg-type]
    source, delays = _source([[0]])
    with pytest.raises(ValueError, match="Unsupported"):
        delay_tcsr(source, delays, binary_backend="numba")


def test_x64_required_for_source_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve int64 input indices instead of implicitly narrowing them."""
    source, delays = _source([[0]])
    source.indices = jnp.asarray(source.indices, dtype=jnp.int64)
    jax.config.update("jax_enable_x64", False)
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Reject int64 before staging a narrowing JAX operation")
    monkeypatch.setattr(jax, "jit", forbidden)
    with pytest.raises(RuntimeError, match="int64 indices"):
        delay_tcsr(source, delays)


@pytest.mark.parametrize("platform,explicit,global_backend,expected", [
    ("cpu", None, None, "jax"),
    ("tpu", None, None, "jax"),
    ("gpu", None, None, "cuda_raw"),
    ("gpu", None, "cusparse", "cusparse"),
    ("gpu", "jax", "cusparse", "jax"),
])
def test_backend_priority(
    monkeypatch: pytest.MonkeyPatch, platform: str, explicit: str | None,
    global_backend: str | None, expected: str,
) -> None:
    primitive = binary_delay.binary_csrmv_delay_p
    available, default = primitive.available_backends(platform), primitive.get_default(platform)
    monkeypatch.setattr(primitive, "available_backends", lambda _: available)
    monkeypatch.setattr(primitive, "get_default", lambda _: default)
    monkeypatch.setattr(delay_main, "get_backend", lambda _: global_backend)
    # Simulate platform selection only; numerical split tests use the real GPU.
    monkeypatch.setattr(binary_delay, "split_region_delay", lambda *args: binary_delay._SplitCSR(
        jnp.array([[0, 1]], jnp.int32), jnp.array([0, 1], jnp.int32),
        jnp.array([0], jnp.int32), jnp.array([1], jnp.float32),
    ))
    source, delays = _source([[0]])
    assert delay_tcsr(source, delays, binary_backend=explicit).binary_backend == expected


def test_primitive_default_and_unavailable_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(delay_main, "get_backend", lambda platform: "numba")
    monkeypatch.setattr(binary_delay.binary_csrmv_delay_p, "_warned_backends", set())
    monkeypatch.setattr(binary_delay.binary_csrmv_delay_p, "get_default", lambda platform: "jax")
    source, delays = _source([[0]])
    with pytest.warns(UserWarning, match="using the primitive default"):
        assert delay_tcsr(source, delays).binary_backend == "jax"
    source.indptr = jax.device_put(source.indptr, jax.devices("cpu")[0])
    for backend in ("cuda_raw", "cusparse"):
        with pytest.raises(ValueError, match="unavailable on cpu"):
            delay_tcsr(source, delays, binary_backend=backend)


def test_backend_fixed_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    source, delays = _source([[0], [2]])
    monkeypatch.setattr(delay_main, "get_backend", lambda platform: "jax")
    op = delay_tcsr(source, delays)
    monkeypatch.setattr(delay_main, "get_backend", lambda platform: "cusparse")
    assert op.binary_backend == "jax"
    assert isinstance(op._inputs, binary_delay._DelayCSRInputs)
    np.testing.assert_array_equal(np.ones(2, np.float32) @ op, [0, 1, 0])


@pytest.mark.parametrize("backend", ["jax", pytest.param("cuda_raw", marks=pytest.mark.skipif(not _GPU, reason="GPU required"))])
def test_csr_backends_never_split(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    def forbidden(*args: Any, **kw: Any) -> Any:
        raise AssertionError("CSR backend must not split")
    monkeypatch.setattr(binary_delay, "split_region_delay", forbidden)
    source, delays = _source([[0], [2]])
    op = delay_tcsr(source, delays, binary_backend=backend)
    np.testing.assert_array_equal(np.ones(2, np.float32) @ op, [0, 1, 0])


@pytest.mark.skipif(not _GPU, reason="GPU required")
def test_cusparse_splits_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = binary_delay.split_region_delay
    calls = []
    def split(*args: Any) -> Any:
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(binary_delay, "split_region_delay", split)
    source, delays = _source([[0], [2]])
    op = delay_tcsr(source, delays, binary_backend="cusparse")
    prepared = op._inputs
    assert isinstance(prepared, binary_delay._SplitCSR)
    assert isinstance(op._workspace, binary_delay._CusparseWorkspace)
    np.testing.assert_array_equal(op._workspace.scratch, np.zeros(3))
    assert op._workspace.scratch.dtype == np.float32
    for events in ([1, 1], [0, 0], [0, 0], [1, 1]):
        np.asarray(events, np.float32) @ op
        assert op._inputs is prepared
    assert len(calls) == 1


def test_cusparse_nnz_limit_checked_before_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binary_delay.binary_csrmv_delay_p, "available_backends", lambda _: ["cusparse"])
    source, delays = _source([[0, 0]], n_post=1)
    with pytest.raises(ValueError, match="segment nnz exceeds M\\*N"):
        delay_tcsr(source, delays, binary_backend="cusparse")


@pytest.mark.parametrize("input_kind", ["numpy", "jax", "binary"])
@pytest.mark.parametrize("boolean", [False, True])
def test_left_dispatch_and_contract(
    monkeypatch: pytest.MonkeyPatch, input_kind: str, boolean: bool,
) -> None:
    """Forward raw activity and adopt every state returned by a stand-in."""
    source, delays = _source([[0], [1], [2], [2]])
    op = delay_tcsr(source, delays, binary_backend="jax")
    result = jnp.array([4, 5, 6], dtype=jnp.float32)
    new_buffer, new_head = op.buffer + 1, op.head + 1
    new_workspace = cast(_DelayWorkspace, op._workspace)._replace(status=jnp.array([0, 0], dtype=jnp.int32))

    def step(inputs: Any, events: Any, **kw: Any) -> Any:
        assert inputs is op._inputs
        assert events.dtype == (np.bool_ if boolean else np.float32)
        np.testing.assert_array_equal(np.asarray(events) > 0, [False, False, True, False])
        assert kw["shape"] == op.shape and kw["backend"] == "jax"
        for name in ("delay_off", "buffer", "head"):
            assert kw[name] is getattr(op, name)
        assert kw["workspace"] is op._workspace
        return result, new_buffer, new_head, new_workspace

    monkeypatch.setattr(binary_delay, "binary_csrmv_delay_step", step, raising=False)
    values: np.ndarray = np.array([-3, 0, 2, np.nan], dtype=np.float32)
    if boolean:
        values = values > 0
    events: Any = values if input_kind == "numpy" else jnp.asarray(values)
    if input_kind == "binary":
        events = BinaryArray(events)
    assert events @ op is result
    assert op.buffer is new_buffer and op.head is new_head and op._workspace is new_workspace


@pytest.mark.parametrize("events,error,match", [
    (np.array(1, dtype=np.float32), ValueError, "vector of length M"),
    (np.ones((2, 1), dtype=np.float32), ValueError, "vector of length M"),
    (np.ones(3, dtype=np.float32), ValueError, "vector of length M"),
    (np.ones(2, dtype=np.int32), TypeError, "bool or float32"),
    (np.ones(2, dtype=np.float64), TypeError, "bool or float32"),
])
def test_invalid_events(events: Any, error: Any, match: str) -> None:
    """Reject invalid events before a backend can mutate state."""
    source, delays = _source([[0], [1]])
    op = delay_tcsr(source, delays)
    with pytest.raises(error, match=match):
        events @ op


def test_multistep_state_flow_with_test_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise ring wrap and all workspace outputs using a NumPy stand-in."""
    source = CSR((np.r_[np.ones(129), 2].astype(np.float32),
                  np.r_[np.zeros(129), 1].astype(np.int32),
                  np.array([0, 129, 130], dtype=np.int32)), shape=(2, 2))
    op = delay_tcsr(source, np.r_[np.full(129, 2), 0].astype(np.int32))
    previous: list[Any] = [op.buffer, op.head, op._workspace]

    def step(inputs: Any, events: Any, **kw: Any) -> Any:
        data, indices, indptr = inputs[:3]
        assert all(a is b for a, b in zip(previous, (kw["buffer"], kw["head"], kw["workspace"])))
        ring, head = np.array(kw["buffer"], copy=True), int(kw["head"][0])
        for row in range(2):
            edge = int(indptr[row])
            for k, count in enumerate(np.asarray(inputs.delay_count)[row]):
                slot = (head + int(kw["delay_off"][k]) // 2) % 3
                for e in range(edge, edge + int(count)):
                    if bool(events[row]):
                        ring[slot, int(indices[e])] += float(data[e])
                edge += int(count)
        current = ring[head].copy()
        ring[head] = 0
        workspace = _DelayWorkspace(
            1, jnp.array([0], dtype=jnp.int32), jnp.array([129], dtype=jnp.int32),
            jnp.array([((head + 2) % 3) * 2], dtype=jnp.int32),
            jnp.array([int(events[0]), 0], dtype=jnp.int32),
        )
        previous[:] = [jnp.asarray(ring), jnp.array([(head + 1) % 3], dtype=jnp.int32), workspace]
        return jnp.asarray(current), *previous

    monkeypatch.setattr(binary_delay, "binary_csrmv_delay_step", step, raising=False)
    sequence = [[1, 1], [0, 0], [0, 1], [1, 0], [0, 0], [0, 0]]
    expected = [[0, 2], [0, 0], [129, 2], [0, 0], [0, 0], [129, 0]]
    for t, (events, arrivals) in enumerate(zip(sequence, expected)):
        np.testing.assert_array_equal(np.array(events, dtype=np.float32) @ op, arrivals)
        np.testing.assert_array_equal(op.head, [(t + 1) % 3])
        assert op._workspace is previous[2]
    np.testing.assert_array_equal(op.buffer, np.zeros((3, 2)))


def test_backend_failure_preserves_member_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave member bindings alone when the backend raises before returning."""
    source, delays = _source([[0]])
    op = delay_tcsr(source, delays)
    before = (op.buffer, op.head, op._workspace)

    def fail(*args: Any, **kw: Any) -> Any:
        raise RuntimeError("backend failed")

    monkeypatch.setattr(binary_delay, "binary_csrmv_delay_step", fail, raising=False)
    with pytest.raises(RuntimeError, match="backend failed"):
        jnp.ones(1, dtype=jnp.float32) @ op
    assert all(a is b for a, b in zip(before, (op.buffer, op.head, op._workspace)))


def test_outer_jit_rejected_without_state_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject both dynamic and closed-over events under an outer JIT."""
    source, delays = _source([[0]])
    op = delay_tcsr(source, delays)
    before = (op.buffer, op.head, op._workspace)
    events = jnp.ones(1, dtype=jnp.float32)

    def step(*args: Any, **kw: Any) -> Any:
        return jnp.zeros(3), kw["buffer"] + 1, kw["head"] + 1, kw["workspace"]

    monkeypatch.setattr(binary_delay, "binary_csrmv_delay_step", step, raising=False)
    with pytest.raises(NotImplementedError, match="Python loop"):
        jax.jit(lambda x: x @ op)(events)
    with pytest.raises(NotImplementedError, match="Python loop"):
        jax.jit(lambda: events @ op)()
    assert all(a is b for a, b in zip(before, (op.buffer, op.head, op._workspace)))


def test_internal_compiled_backend_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow a Python left multiply to call compiled numerical work."""
    source, delays = _source([[0]])
    op = delay_tcsr(source, delays, buffer_slots=2)
    compiled = jax.jit(lambda buffer, head: (buffer[head[0]] + 1, buffer + 1, (head + 1) % 2))

    def step(*args: Any, **kw: Any) -> Any:
        y, buffer, head = compiled(kw["buffer"], kw["head"])
        return y, buffer, head, kw["workspace"]

    monkeypatch.setattr(binary_delay, "binary_csrmv_delay_step", step, raising=False)
    np.testing.assert_array_equal(jnp.ones(1, dtype=jnp.bool_) @ op, np.ones(3))
    np.testing.assert_array_equal(op.head, [1])


def _metadata(source: CSR, delays: np.ndarray, slots: int = 8) -> DelayMetadata:
    steps, counts, offsets = prepare_delay_metadata(delays, source.indptr, n_post=source.shape[1], buffer_slots=slots)
    tasks = ((np.asarray(counts) + 4095) // 4096).sum(axis=1)
    tasks[np.diff(np.asarray(source.indptr)) <= 128] = 0
    return DelayMetadata(steps, np.asarray(counts), np.asarray(offsets), tasks.astype(np.int32), int(tasks.sum()), slots)


@pytest.mark.parametrize("backend", ["jax", "cuda_raw", "cusparse"])
@pytest.mark.parametrize("kind", ["heterogeneous", "shared", "scalar"])
def test_precomputed_matches_edge_input(backend: str, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep values, units, state progression and backend preparation identical."""
    if backend != "jax" and not _GPU:
        pytest.skip("Requires a real CUDA device")
    # Keep each segment below the existing cuSPARSE M*N bound while retaining a long row.
    source, delays = _source([[0] * 64 + [2] * 65, [], [7]], n_post=64, kind=kind, ptr_dtype=np.int64)
    source = CSR((source.data * getattr(u, "mV"), source.indices, source.indptr),
                 shape=source.shape, indptr_dtype=np.int64)
    meta = _metadata(source, delays)
    original = delay_tcsr(source, delays, buffer_slots=8, binary_backend=backend)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Prepared metadata must not expand or recount edges")

    monkeypatch.setattr(delay_main, "prepare_delay_metadata", forbidden)
    prepared = delay_tcsr(source, delay_metadata=meta, buffer_slots=8, binary_backend=backend)
    for step in range(17):
        events = jnp.array([step % 2, 0, step % 3 == 0], dtype=jnp.float32)
        left, right = events @ original, events @ prepared
        assert u.get_unit(left) == u.get_unit(right) == getattr(u, "mV")
        np.testing.assert_allclose(u.get_mantissa(left), u.get_mantissa(right), rtol=1e-5)
        np.testing.assert_allclose(original.buffer, prepared.buffer, rtol=1e-5)
        np.testing.assert_array_equal(original.head, prepared.head)
    assert prepared.indices is source.indices and prepared.indptr is source.indptr


@pytest.mark.parametrize("field,value", [
    ("delay_steps", ()), ("delay_steps", (2, 0)), ("delay_steps", (-1, 2)), ("delay_steps", (False, 2)),
    ("buffer_slots", 2), ("buffer_slots", True),
    ("delay_count", np.array([[1, 1]], np.int64)),
    ("delay_count", np.array([[-1, 3], [0, 0]], np.int64)),
    ("delay_count", np.zeros((2, 2), np.int64)),
    ("delay_count", np.zeros((2, 2), np.int32)),
    ("delay_off", np.array([0, 3], np.int32)), ("delay_off", np.array([0, 6], np.int64)),
    ("row_task_count", np.array([1, 0], np.int32)),
    ("row_task_count", np.array([0], np.int32)),
    ("task_capacity", 1), ("task_capacity", True),
])
def test_precomputed_rejects_inconsistent_metadata(field: str, value: Any) -> None:
    """Validate segmentation and capacity before they can reach device kernels."""
    source, delays = _source([[0, 2], []], ptr_dtype=np.int64)
    meta = _metadata(source, delays)._replace(**{field: value})
    with pytest.raises((ValueError, TypeError)):
        delay_tcsr(source, delay_metadata=meta, binary_backend="jax")


def test_precomputed_input_modes_and_empty_bins() -> None:
    """Reject ambiguous inputs and preserve empty physical delay buckets."""
    source, delays = _source([[], []], ptr_dtype=np.int64)
    meta = DelayMetadata(tuple(range(8)), np.zeros((2, 8), np.int64), np.arange(8, dtype=np.int32) * 3,
                         np.zeros(2, np.int32), 0, 8)
    with pytest.raises(ValueError, match="exactly one"):
        delay_tcsr(source)
    with pytest.raises(ValueError, match="exactly one"):
        delay_tcsr(source, delays, delay_metadata=meta)
    with pytest.raises(ValueError, match="match"):
        delay_tcsr(source, delay_metadata=meta, buffer_slots=9)
    with pytest.raises(TypeError, match="DelayMetadata"):
        delay_tcsr(source, delay_metadata=cast(Any, {}))
    matrix = delay_tcsr(source, delay_metadata=meta, binary_backend="jax")
    assert matrix.delay_steps == tuple(range(8)) and matrix.buffer.shape == (8, 3)
    np.testing.assert_array_equal(jnp.zeros(2, np.float32) @ matrix, np.zeros(3))
    np.testing.assert_array_equal(matrix.head, [1])
