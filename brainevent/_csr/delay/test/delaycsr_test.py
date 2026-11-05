"""Exercise owned delay state, explicit time steps, and execution backend dispatch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import brainunit as u

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


def test_head_and_time_validation():
    w, _ = problem()
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
        state, _ = problem(backend='jax_raw')
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
    state, _ = problem(backend='jax_raw')
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
    state, original = problem(weighted=weighted, homo=homo, backend=backend)
    ring = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    state.ring = ring
    leaves = jax.tree.leaves(state)
    events = jnp.array([-2, .5], jnp.float32)

    def loss(matrix, weights):
        local = matrix.with_data(weights)
        output = local.forward(events, 3)
        return local.ring.sum() + output.sum()

    expected_ring = reference(np.asarray(ring), original, np.asarray(events), 3, weighted)
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
    state, original = problem(weighted=weighted, backend=backend)
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
        reference(np.asarray(ring), original, event, int(time), weighted)
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
    w, original = problem(weighted=weighted, backend=backend)
    value = 2 if jnp.dtype(dtype).kind in 'bu' else -2
    events = jnp.array([value, 0], dtype)
    expected = reference(np.zeros((4, 2)), original, np.asarray(events), 0, weighted)
    np.testing.assert_array_equal(events @ w, expected[1])
    np.testing.assert_array_equal(w.ring, expected)
    assert w.ring.dtype == jnp.float32


@pytest.mark.parametrize('weighted', [False, True])
def test_invalid_event_type_preserves_ring(weighted):
    w, _ = problem(weighted=weighted, backend='jax_raw')
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
    w, _ = problem()
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
