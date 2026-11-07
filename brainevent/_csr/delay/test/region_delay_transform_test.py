"""Compare delay primitives' AD and batching with an independent edge scatter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.delay.region_delay import delaycsr
from brainevent._csr.delay.test._helpers import GPU, cuda_step


pytestmark = GPU


def problem(weighted, homo=False, dtype=jnp.float32):
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


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homo", [False, True])
def test_jvp_and_reverse_mode(weighted, homo):
    state, project, reference = problem(weighted, homo)
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


@pytest.mark.parametrize("operand", [0, 1, 2])
def test_registered_transpose(operand):
    state, project, reference = problem(True)
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


@pytest.mark.parametrize("weighted", [False, True])
def test_vmap_nonleading_axes_and_grad(weighted):
    state, project, reference = problem(weighted)
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


def test_class_with_data_gradient_and_batched_state():
    state, _, reference = problem(True)
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


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_binary_event_tangent_uses_weight_dtype(dtype):
    state, project, _ = problem(False, dtype=dtype)
    x = jnp.array([-2, 0, 0], jnp.float32)
    fn = lambda e: project(state.ring, e, state.data, jnp.array([4], jnp.int32))
    _, dot = jax.jit(lambda e: jax.jvp(fn, (e,), (jnp.ones_like(e),)))(x)
    assert dot.dtype == dtype
    grad = jax.jit(jax.grad(lambda e: fn(e).sum()))(x)
    assert grad.dtype == jnp.float32
    np.testing.assert_array_equal(grad, [1.5, 0, 1.25])


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
