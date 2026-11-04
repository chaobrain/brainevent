"""Exercise owned delay state, explicit time steps, and execution backend dispatch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, WeightedArray
from brainevent._csr.delay import delay as operator
from brainevent._csr.delay.region_delay import delaycsr
from brainevent._csr.delay.test._helpers import BACKENDS, GPU, cuda_step
from brainevent._csr.delay.test._reference import numpy_project_step


def problem(dtype=jnp.float32, weighted=False, homo=False, backend=None):
    weights = np.asarray([.5] if homo else [2, -1, 3, .5], dtype=jnp.dtype(dtype))
    indices = np.array([1, 0, 0, 1], np.int32)
    indptr = np.array([0, 3, 4], np.int64)
    delays = np.array([2, 1, 0, 1], np.int32)
    matrix = delaycsr((weights, indices, indptr), shape=(2, 2), edge_delays=delays,
                      buffer_slots=4, weighted=weighted, backend=backend)
    return matrix, (weights, indices, indptr, delays)


def reference(ring, original, events, t, weighted):
    result = ring.copy()
    head = t % len(ring)
    result[head] = 0
    return numpy_project_step(*original, events, head, result, weighted=weighted)


@GPU
@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('homo', [False, True])
def test_forward_and_matmul_return_next_preserving_full_ring(dtype, weighted, homo):
    w, original = problem(dtype, weighted, homo)
    w.ring = jnp.arange(8, dtype=dtype).reshape(4, 2)
    expected = np.asarray(w.ring).astype(np.float64)
    metadata = (w.delay_count, w.delay_off, w.row_task_count, w.indices, w.indptr)
    for t in range(11, 20):
        x = np.array([-2, .5] if t % 3 else [0, 0], dtype=jnp.dtype(dtype if weighted else jnp.float32))
        expected = reference(expected, original, x, t, weighted)
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
    w, original = problem(weighted=weighted)
    x = jnp.array([[-2, .5], [0, 0], [.25, -1], [1, 2], [-1, .5]], jnp.float32)

    def step(state, event_time):
        event, t = event_time
        y = state.forward(event, t)
        return state, y

    expected = np.zeros((4, 2))
    outputs = []
    for t, event in enumerate(np.asarray(x), 7):
        expected = reference(expected, original, event, t, weighted)
        outputs.append(expected[(t + 1) % 4].copy())
    after, outputs_scan = jax.jit(lambda state: jax.lax.scan(step, state, (x, jnp.arange(7, 12))))(w)
    np.testing.assert_array_equal(outputs_scan, outputs)
    np.testing.assert_array_equal(after.ring, expected)
    np.testing.assert_array_equal(w.ring, 0)
    assert not isinstance(w.head, jax.core.Tracer)
    after, y = jax.jit(step)(after, (jnp.zeros(2, jnp.float32), jnp.array(12)))
    expected = reference(expected, original, np.zeros(2), 12, weighted)
    np.testing.assert_array_equal(after.ring, expected)
    np.testing.assert_array_equal(y, expected[1])


@GPU
def test_wrapped_events_and_large_time():
    binary, _ = problem()
    weighted, _ = problem(weighted=True)
    x = jnp.array([-2, .5], jnp.float32)
    np.testing.assert_array_equal(binary.forward(BinaryArray(x), 2**40 + 1), [-1, .5])
    np.testing.assert_array_equal(weighted.forward(WeightedArray(x), 2**40 + 1), [2, .25])
    with pytest.raises(TypeError, match='mode'):
        weighted.forward(BinaryArray(x), 0)


def test_validation_and_no_hidden_jit_state():
    w, _ = problem()
    for head in (-1, 4, 2**32):
        with pytest.raises(ValueError, match='head'):
            w.head = head
    for t in (True, .5, [1]):
        with pytest.raises(TypeError, match='integer'):
            w.forward(jnp.zeros(2, jnp.float32), t)
    with pytest.raises(RuntimeError, match='JIT/scan'):
        jax.jit(w.forward)(jnp.zeros(2, jnp.float32), 0)
    assert not isinstance(w.head, jax.core.Tracer)
    x = jnp.zeros(2, jnp.float32)
    with pytest.raises(RuntimeError, match='JIT/scan'):
        jax.jit(lambda: w.forward(x, 0))()
    assert not isinstance(w.head, jax.core.Tracer)


@pytest.mark.parametrize('weighted,dtype', [
    (False, jnp.int32), (False, jnp.float16),
    (True, jnp.bool_), (True, jnp.float64),
])
def test_matmul_rejects_event_dtype_with_valid_shape(weighted, dtype):
    w, _ = problem(weighted=weighted)
    with pytest.raises(TypeError, match='events'):
        w.__rmatmul__(jnp.ones(2, dtype))


@pytest.mark.parametrize('corruption', ['ring_shape', 'ring_dtype', 'workspace'])
def test_matmul_rejects_invalid_mutable_state(corruption):
    w, _ = problem()
    if corruption == 'ring_shape':
        w.ring = jnp.zeros((4, 3), jnp.float32)
        exception, message = ValueError, 'ring|buffer'
    elif corruption == 'ring_dtype':
        w.ring = w.ring.astype(jnp.float64)
        exception, message = TypeError, 'ring|buffer'
    else:
        w.workspace = None
        exception, message = RuntimeError, 'workspace'
    with pytest.raises(exception, match=message):
        w.__rmatmul__(jnp.ones(2, jnp.float32))


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
    w, _ = problem(weighted=weighted)
    with pytest.raises(ValueError, match="events must have shape"):
        w.forward(jnp.zeros(shape, jnp.float32), 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_forward_and_jit_use_the_selected_backend(backend):
    state, _ = problem(backend=backend)
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
    state, _ = problem(backend=backend)
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
