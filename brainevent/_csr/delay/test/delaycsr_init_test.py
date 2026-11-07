"""Validate delay CSR construction, sorting, backend metadata, and weight replacement."""

from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import CSR
from brainevent._csr.delay import csr_delay_deliver_layout_p, region_delay as module
from brainevent._csr.delay.region_delay import delaycsr
from brainevent._csr.delay.test._helpers import BACKENDS, GPU


def _case(**changes: Any) -> Any:
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


def _matrix(*, homogeneous=False, weighted=False, backend=None):
    return delaycsr(
        (np.array([2] if homogeneous else [2, 3, 5, 7], np.float32),
         np.array([2, 0, 1, 0], np.int32), np.array([0, 3, 4], np.int32)),
        edge_delays=np.array([2, 1, 1, 2], np.int32), shape=(2, 3),
        buffer_slots=4, ring=jnp.arange(12, dtype=jnp.float32).reshape(4, 3), head=3,
        weighted=weighted, backend=backend,
    )


def _assert_same_structure_and_state(original, updated):
    assert isinstance(updated, delaycsr)
    assert updated is not original
    for name in ("indices", "indptr", "delay_count", "delay_off", "row_task_count",
                 "ring", "head", "workspace"):
        assert getattr(updated, name) is getattr(original, name), name
    for name in ("shape", "backend", "delay_steps", "capacity", "L", "weighted"):
        assert getattr(updated, name) == getattr(original, name), name


def _source(backend=None):
    return CSR((jnp.array([2, -1, .5, 3], jnp.float32), jnp.array([1, 0, 1, 0], jnp.int32),
                jnp.array([0, 3, 4], jnp.int64)), shape=(2, 2), backend=backend,
               indptr_dtype=jnp.int64)


def test_grouping_preserves_weight_alignment_and_duplicate_order() -> None:
    prepared = _case()
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
    prepared = _case(weights=np.array([4], np.float32))
    assert prepared.weights.shape == (1,)
    assert prepared.delay_steps == (0, 1, 2)
    assert prepared.capacity == 0


def _prepared_arguments(matrix: delaycsr) -> dict[str, Any]:
    return dict(
        data=(matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape,
        delay_steps=matrix.delay_steps, delay_count=matrix.delay_count,
        delay_off=matrix.delay_off, row_task_count=matrix.row_task_count,
        capacity=matrix.capacity, buffer_slots=matrix.L, weighted=matrix.weighted,
        ring=matrix.ring, head=matrix.head, workspace=matrix.workspace,
        backend=matrix.backend,
    )


def test_from_prepared_bypasses_initialization_and_reuses_device_layout(monkeypatch) -> None:
    original = _matrix()
    arguments = _prepared_arguments(original)

    def unexpected_preprocessing(*args, **kwargs):
        raise AssertionError("prepared input must not run initialization or preprocessing")

    for owner, name in ((CSR, "__init__"), (delaycsr, "__init__"),
                        (delaycsr, "_sort_by_delay"), (delaycsr, "_build_delay_metadata")):
        monkeypatch.setattr(owner, name, unexpected_preprocessing)
    updated = delaycsr.from_prepared(**arguments)
    _assert_same_structure_and_state(original, updated)
    assert updated.data is original.data
    assert updated.config is not None
    assert updated._cache.mirror is None


def test_from_prepared_static_does_not_read_device_layout(monkeypatch) -> None:
    original = _matrix()
    arguments = _prepared_arguments(original)
    asarray = np.asarray

    def guarded_asarray(value, *args, **kwargs):
        if isinstance(value, jax.Array):
            raise AssertionError("static construction must not copy device layout to the host")
        return asarray(value, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(module.np, "asarray", guarded_asarray)
        updated = delaycsr.from_prepared(**arguments)
    _assert_same_structure_and_state(original, updated)


@pytest.mark.parametrize("source_form", ["csr", "tuple", "separate", "scalar"])
def test_from_prepared_source_forms_and_default_state(source_form) -> None:
    original = _matrix(homogeneous=True, backend="jax_raw")
    arguments = _prepared_arguments(original)
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
    updated = delaycsr.from_prepared(**_prepared_arguments(original), validate="full")
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
    arguments = _prepared_arguments(original)
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
    arguments = _prepared_arguments(_matrix())
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
    arguments = _prepared_arguments(_matrix())
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
    arguments = _prepared_arguments(original)
    arguments.update(capacity=1, workspace=None)
    with pytest.raises(ValueError, match="capacity"):
        delaycsr.from_prepared(**arguments, validate="full")


def test_from_prepared_static_retains_runtime_metadata_checks() -> None:
    arguments = _prepared_arguments(_matrix(backend="jax_raw"))
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
    arguments = _prepared_arguments(_matrix())
    arguments["workspace"] = arguments["workspace"]._replace(**{field: value})
    with pytest.raises(exception, match="workspace"):
        delaycsr.from_prepared(**arguments)


def test_from_prepared_owns_numpy_layout_ring_and_workspace() -> None:
    original = _matrix()
    arguments = _prepared_arguments(original)
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
    updated = delaycsr.from_prepared(**_prepared_arguments(original), validate="full")
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
    arguments = _prepared_arguments(_matrix(weighted=True))
    arguments["ring"] = arguments["ring"].astype(jnp.float16)
    with pytest.raises(TypeError, match="workspace dtypes"):
        delaycsr.from_prepared(**arguments)
    arguments["workspace"] = None
    updated = delaycsr.from_prepared(**arguments)
    assert updated.workspace.task_value.dtype == jnp.float16
    assert updated.data.dtype == jnp.float32


@pytest.mark.parametrize("name", ["weights", "indices", "indptr", "delay_count", "delay_off", "ring"])
def test_from_prepared_requires_x64_for_explicit_wide_inputs(name) -> None:
    arguments = _prepared_arguments(_matrix())
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
    arguments = _prepared_arguments(_matrix())
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
    arguments = _prepared_arguments(_matrix())
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
    original = _matrix()
    arguments = _prepared_arguments(original)
    arguments["data"] = original
    arguments.update(changes)
    with pytest.raises(exception, match=match):
        delaycsr.from_prepared(**arguments)


def test_from_prepared_requires_static_delay_steps() -> None:
    arguments = _prepared_arguments(_matrix())
    arguments["delay_steps"] = jnp.array([1, 2], jnp.int32)
    with pytest.raises(TypeError, match="static host integers"):
        delaycsr.from_prepared(**arguments)


def test_from_prepared_full_requires_concrete_layout_under_jit() -> None:
    original = _matrix()

    @jax.jit
    def reconstruct(matrix):
        return delaycsr.from_prepared(**_prepared_arguments(matrix), validate="full")

    with pytest.raises(ValueError, match="requires concrete layout"):
        reconstruct(original)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_from_prepared_jit_scan_and_differentiation(weighted, homogeneous) -> None:
    original = _matrix(homogeneous=homogeneous, weighted=weighted, backend="jax_raw")
    events = jnp.array([[1, 0], [0, -2], [.5, 1]], jnp.float32)
    times = jnp.array([3, 4, 5], jnp.int32)

    def scan(matrix, xs):
        def step(carry, xt):
            spike = carry.forward(*xt)
            return carry, spike
        return jax.lax.scan(step, matrix, (xs, times))

    def run(matrix, weights, xs):
        arguments = _prepared_arguments(matrix)
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
    scalar_layout = _case(edge_delays=scalar, delay_steps=None, backend="jax_raw")
    per_edge_layout = _case(edge_delays=np.full(4, 2, np.int32), delay_steps=None,
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
    source = _source("jax_raw")
    scalar = source.delay_layout(1, n_slots=3)
    per_edge = source.delay_layout(np.ones(source.nse, np.int32), n_slots=3)
    np.testing.assert_array_equal(scalar.delay_count, per_edge.delay_count)
    np.testing.assert_array_equal(scalar.data, per_edge.data)
    np.testing.assert_array_equal(scalar.indices, per_edge.indices)


def test_zero_scalar_delay_delivers_into_current_slot() -> None:
    prepared = _case(edge_delays=0, delay_steps=None, backend="jax_raw")
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
    prepared = _case(delay_steps=(0, 1, 2), buffer_slots=8)
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
        _case(**changes)


@pytest.mark.parametrize("values", [[], [0, 0], [2, 0], [-1, 2], [0.0, 1.0], [[0]], [2**31]])
def test_invalid_delay_value_sets(values: Any) -> None:
    with pytest.raises(ValueError):
        _case(delay_steps=values)


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
            _case()
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
        _case(sorted_by_delay=True, **changes)


def test_sorted_input_skips_sorting_and_keeps_weight_order(monkeypatch):
    def unexpected_sort(*args):
        raise AssertionError("sorted input must not be sorted again")

    monkeypatch.setattr(delaycsr, "_sort_by_delay", unexpected_sort)
    w = _case(weights=np.array([3, 5, 7, 2], np.float32), indices=np.array([0, 2, 2, 3], np.int32),
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
    w = _case(sorted_by_delay=False)
    assert calls == [4, 0]
    np.testing.assert_array_equal(w.data, [3, 5, 7, 2])


def test_member_sort_overwrites_csr_and_invalidates_old_structure_cache():
    w = _case()
    # Structural CSR operations must not create a partially initialized delaycsr.
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
    prepared = _case()
    prepared.sort_by_delay(2)
    np.testing.assert_array_equal(prepared.indices, [0, 2, 2, 3])
    np.testing.assert_array_equal(prepared.data, [3, 5, 7, 2])
    np.testing.assert_array_equal(prepared.delay_count, [[0, 0, 4], [0, 0, 0]])


def test_constructor_has_no_event_input_or_event_state():
    with pytest.raises(TypeError, match="events"):
        _case(events=np.zeros((2, 3), np.float32))
    with pytest.raises(TypeError, match="sorted_by_delay"):
        _case(sorted_by_delay=1)
    w = _case()
    assert not {"event", "events", "duration_steps"} & vars(w).keys()


@pytest.mark.parametrize("array_type", [np.asarray, jnp.asarray], ids=["numpy", "jax"])
def test_with_data_preserves_supplied_current_order_and_owned_state(array_type):
    matrix = _matrix()
    # Initialization permutes original edges as [1, 2, 0, 3]. New weights
    # already follow the resulting layout and must not be permuted again.
    np.testing.assert_array_equal(matrix.data, [3, 5, 2, 7])
    np.testing.assert_array_equal(matrix.indices, [0, 1, 2, 0])
    new_weights = array_type([10, 20, 30, 40], dtype=np.float32)
    updated = matrix.with_data(new_weights)
    np.testing.assert_array_equal(updated.data, new_weights)
    np.testing.assert_array_equal(matrix.data, [3, 5, 2, 7])
    _assert_same_structure_and_state(matrix, updated)
    updated.clear_head()
    np.testing.assert_array_equal(updated.ring[3], 0)
    np.testing.assert_array_equal(matrix.ring, np.arange(12).reshape(4, 3))
    assert int(matrix.head) == int(updated.head) == 3


def test_shared_weight_keeps_its_exact_shape():
    matrix = _matrix(homogeneous=True)
    updated = matrix.with_data(jnp.array([11], jnp.float32))
    assert updated.data.shape == (1,)
    np.testing.assert_array_equal(updated.data, [11])
    np.testing.assert_array_equal(matrix.data, [2])
    _assert_same_structure_and_state(matrix, updated)


@GPU
@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_delivery_uses_replacement_weights_on_current_edges(weighted):
    matrix = _matrix()
    updated = matrix.with_data(jnp.array([10, 20, 30, 40], jnp.float32))
    initial = jnp.ones((4, 3), jnp.float32)
    events = jnp.array([-2, .5], jnp.float32)
    result = jax.jit(lambda state, ring, event: state.csr_delay_deliver_layout(
        ring, event, 3, weighted=weighted))(updated, initial, events)
    expected = np.ones((4, 3), np.float32)
    expected[0, :2] += [-20, -40] if weighted else [10, 20]
    expected[1, [0, 2]] += [20, -60] if weighted else [40, 30]
    np.testing.assert_array_equal(result, expected)
    _assert_same_structure_and_state(matrix, updated)
    np.testing.assert_array_equal(matrix.data, [3, 5, 2, 7])


def test_jit_and_grad_use_current_weight_order_with_jax_delivery():
    matrix = _matrix()
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
        _matrix().with_data(value)


@pytest.mark.parametrize("sparse_type", ["csr", "delaycsr"])
def test_sparse_objects_are_not_weight_arrays(sparse_type):
    matrix = _matrix()
    value = matrix if sparse_type == "delaycsr" else CSR(
        (matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape)
    with pytest.raises(TypeError, match="array"):
        matrix.with_data(value)


@pytest.mark.parametrize("homogeneous,shape", [
    (False, ()), (False, (1,)), (False, (3,)), (False, (2, 2)), (True, (4,)),
])
def test_replacement_shape_must_match_exactly(homogeneous, shape):
    with pytest.raises(ValueError, match="shape"):
        _matrix(homogeneous=homogeneous).with_data(np.ones(shape, np.float32))


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32])
def test_replacement_dtype_must_match(dtype):
    with pytest.raises(TypeError, match="dtype"):
        _matrix().with_data(np.ones(4, dtype))


def test_replacement_units_must_match():
    with pytest.raises(TypeError, match="unit"):
        _matrix().with_data(jnp.ones(4, jnp.float32) * u.mS)


@pytest.mark.parametrize("receiver", ["csr", "uninitialized"])
def test_receiver_must_be_an_initialized_delaycsr(receiver):
    matrix = _matrix()
    value = (CSR((matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape)
             if receiver == "csr" else object.__new__(delaycsr))
    with pytest.raises(TypeError, match="initializ"):
        delaycsr.with_data(value, jnp.ones(4, jnp.float32))


@GPU
def test_inherited_plasticity_updates_keep_array_based_weight_replacement():
    matrix = _matrix()
    pre = matrix.update_on_pre(
        jnp.array([True, False]), jnp.array([1, 10, 100], jnp.float32))
    np.testing.assert_array_equal(pre.data, [4, 15, 102, 7])
    _assert_same_structure_and_state(matrix, pre)
    post = matrix.update_on_post(
        jnp.array([10, 100], jnp.float32), jnp.array([True, False, True]))
    np.testing.assert_array_equal(post.data, [13, 5, 12, 107])
    _assert_same_structure_and_state(matrix, post)
    np.testing.assert_array_equal(matrix.data, [3, 5, 2, 7])


@pytest.mark.parametrize("backend", ["cuda", "cusparse", "numba", "invalid"])
def test_unsupported_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="backend"):
        _case(backend=backend)


@pytest.mark.parametrize("backend", [None, "cuda_raw", "jax_raw"])
def test_source_backend_is_inherited(backend):
    original = _source(backend)
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
    original = _source(source_backend)
    state = delaycsr(original, edge_delays=np.array([2, 0, 1, 3]), backend=override)
    assert state.backend == override
    assert original.backend == source_backend


def test_unsupported_source_backend_needs_an_explicit_supported_override():
    original = _source("numba")
    with pytest.raises(ValueError, match="backend"):
        delaycsr(original, edge_delays=np.array([2, 0, 1, 3]))
    state = delaycsr(original, edge_delays=np.array([2, 0, 1, 3]), backend="jax_raw")
    assert state.backend == "jax_raw"
    assert original.backend == "numba"


@pytest.mark.parametrize("backend", [None, "cuda_raw", "jax_raw"])
def test_pytree_and_with_data_keep_backend(backend):
    state = _case(backend=backend)
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


def _aligned_host_array(values, dtype):
    values = np.asarray(values, dtype)
    storage = np.empty(values.nbytes + 64, np.uint8)
    start = (-storage.ctypes.data) % 64
    result = storage[start:start + values.nbytes].view(dtype).reshape(values.shape)
    result[...] = values
    return result


@pytest.mark.parametrize("source_kind", ["arrays", "csr"])
@pytest.mark.parametrize("sorted_by_delay", [False, True])
def test_constructor_owns_host_arrays(source_kind, sorted_by_delay):
    weights = _aligned_host_array([2., 3.], np.float32)
    columns = _aligned_host_array([0, 1], np.int32)
    pointers = _aligned_host_array([0, 2], np.int32)
    delays = _aligned_host_array([1, 2], np.int32)
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
    matrix = _matrix()
    weights = _aligned_host_array([10., 20., 30., 40.], np.float32)
    updated = matrix.with_data(weights)
    weights[:] = -1
    np.testing.assert_array_equal(updated.data, [10, 20, 30, 40])
    _assert_same_structure_and_state(matrix, updated)


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
    _assert_same_structure_and_state(matrix, updated)
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
