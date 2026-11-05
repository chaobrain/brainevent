"""The JAX backend consumes the existing prepared delay-delivery ABI."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.delay.delay import csr_delay_deliver_layout_p, csr_delay_deliver_layout_p_call
from brainevent._csr.delay.test._helpers import GPU


ROWS = np.array([0, 0, 0, 2, 2, 2], np.int32)
COLUMNS = np.array([0, 2, 1, 1, 0, 2], np.int32)
DELAYS = np.array([0, 1, 3, 0, 1, 3], np.int32)


def _operands(weighted=False, homogeneous=False, dtype=jnp.float32):
    """Prepare two nonempty rows, one empty row and one empty delay segment."""
    queue = jnp.zeros(0, jnp.int32)
    workspace = [queue, queue, queue]
    if weighted:
        workspace.append(jnp.zeros(0, dtype))
    workspace.append(jnp.zeros(2, jnp.int32))
    return [jnp.arange(15, dtype=dtype).reshape(5, 3) / 8,
            jnp.array([-2, 0, .5], dtype if weighted else jnp.float32),
            jnp.array([.5] if homogeneous else [2, -1, .5, 3, -2, .25], dtype),
            jnp.asarray(COLUMNS), jnp.array([0, 3, 3, 6], jnp.int32),
            jnp.array([[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1]], jnp.int32),
            jnp.array([0, 3, 6, 9], jnp.int32), jnp.array([4], jnp.int32),
            jnp.zeros(3, jnp.int32), *workspace]


def _project(args, *, weighted=False, backend="jax_raw", check_status=True):
    return csr_delay_deliver_layout_p_call(
        *args, backend=backend, weighted=weighted, check_status=check_status)


def _numpy_reference(ring, events, data, head, weighted):
    result = np.asarray(ring).copy()
    weights = np.broadcast_to(np.asarray(data), (6,))
    values = np.asarray(events)
    for edge, row in enumerate(ROWS):
        if values[row] != 0:
            value = weights[edge] * values[row] if weighted else weights[edge]
            result[(int(np.asarray(head)[0]) + int(DELAYS[edge])) % 5, COLUMNS[edge]] += value
    return result


@jax.custom_jvp
def _activity(events):
    return (events != 0).astype(events.dtype)


@_activity.defjvp
def _activity_jvp(primals, tangents):
    return _activity(primals[0]), tangents[0]


def _jax_reference(ring, events, data, head, weighted):
    factors = events if weighted else _activity(events)
    return ring.at[(head[0] + jnp.asarray(DELAYS)) % 5, jnp.asarray(COLUMNS)].add(
        jnp.broadcast_to(data, (6,)) * factors[jnp.asarray(ROWS)])


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_jit_delivery_preserves_nonzero_head_and_repeated_accumulation(weighted, homogeneous, dtype):
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX x64")
    args = _operands(weighted, homogeneous, dtype)
    original = np.asarray(args[0]).copy()
    fn = jax.jit(lambda *xs: _project(xs, weighted=weighted))
    result = fn(*args)
    expected = _numpy_reference(*args[:3], args[7], weighted)
    np.testing.assert_array_equal(result[0], expected)
    second = fn(result[0], *args[1:])
    np.testing.assert_array_equal(second[0], _numpy_reference(expected, args[1], args[2], args[7], weighted))
    np.testing.assert_array_equal(args[0], original)
    assert len(result) == (6 if weighted else 5)
    for actual, source in zip(result[1:], args[9:]):
        assert actual.shape == source.shape and actual.dtype == source.dtype
        np.testing.assert_array_equal(actual, 0)


@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.float32])
def test_binary_events_and_silent_step(event_dtype):
    args = _operands()
    args[1] = args[1].astype(event_dtype)
    np.testing.assert_array_equal(_project(args)[0], _numpy_reference(*args[:3], args[7], False))
    args[1] = jnp.zeros_like(args[1])
    np.testing.assert_array_equal(jax.jit(lambda *xs: _project(xs)[0])(*args), args[0])


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_jvp_vjp_and_nonleading_vmap(weighted, homogeneous):
    args = _operands(weighted, homogeneous)
    # An inactive nonempty row must still have an event tangent/STE.
    args[1] = args[1].at[2].set(0)

    def project(ring, events, data, head):
        return _project((ring, events, data, *args[3:7], head, *args[8:]), weighted=weighted)[0]

    ref = lambda ring, events, data, head: _jax_reference(ring, events, data, head, weighted)
    inputs = tuple(args[:3])
    tangents = tuple(jnp.full_like(x, .25) for x in inputs)
    actual = jax.jit(lambda *xs: jax.jvp(lambda *ys: project(*ys, args[7]), xs, tangents))(*inputs)
    expected = jax.jvp(lambda *xs: ref(*xs, args[7]), inputs, tangents)
    for got, wanted in zip(actual, expected):
        np.testing.assert_allclose(got, wanted, rtol=1e-6, atol=1e-6)
    cotangent = jnp.arange(1, 16, dtype=jnp.float32).reshape(5, 3) / 4
    gradient = jax.jit(jax.grad(lambda *xs: (project(*xs, args[7]) * cotangent).sum(),
                                argnums=(0, 1, 2)))(*inputs)
    expected_grad = jax.grad(lambda *xs: (ref(*xs, args[7]) * cotangent).sum(), argnums=(0, 1, 2))(*inputs)
    for got, wanted in zip(gradient, expected_grad):
        np.testing.assert_allclose(got, wanted, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(gradient[0], cotangent)
    assert gradient[1][2] != 0
    batches = (jnp.stack([args[0], args[0] + 1], axis=2),
               jnp.stack([args[1], args[1] * 2], axis=1),
               jnp.stack([args[2], args[2] / 2]), jnp.array([[0], [4]], jnp.int32))
    axes = (2, 1, 0, 0)
    np.testing.assert_allclose(jax.jit(jax.vmap(project, in_axes=axes))(*batches),
                               jax.vmap(ref, in_axes=axes)(*batches), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("weighted", [False, True])
def test_inactive_nonfinite_weights_and_data_cotangents(weighted):
    args = _operands(weighted)
    args[1] = args[1].at[2].set(0)
    args[2] = args[2].at[3:].set(jnp.array([jnp.nan, jnp.inf, -jnp.inf], args[2].dtype))

    def project(data):
        return _project((args[0], args[1], data, *args[3:]), weighted=weighted)[0]

    np.testing.assert_array_equal(jax.jit(project)(args[2]), _numpy_reference(*args[:3], args[7], weighted))
    _, pullback = jax.vjp(project, args[2])
    cotangent = jnp.ones_like(args[0]).at[4, 1].set(jnp.nan).at[0, 0].set(jnp.inf).at[2, 2].set(-jnp.inf)
    gradient = pullback(cotangent)[0]
    np.testing.assert_array_equal(gradient[3:], 0)


@pytest.mark.parametrize("weighted", [False, True])
def test_mixed_event_jvp_keeps_activity_before_value_conversion(weighted):
    args = _operands(weighted)
    args[1] = jnp.zeros(3, jnp.float64)
    args[2] = args[2].at[0].set(jnp.inf)

    def project(events):
        return _project((args[0], events, *args[2:]), weighted=weighted)[0]

    primal, tangent = jax.jit(lambda x: jax.jvp(
        project, (x,), (jnp.array([1e-300, 0., 0.], jnp.float64),)))(args[1])
    np.testing.assert_array_equal(primal, args[0])
    # main's JAX delay path tests the original tangent before casting its
    # value to float32: the active Inf * 0 contribution remains NaN.
    assert tangent.dtype == jnp.float32
    assert np.isnan(np.asarray(tangent)[4, 0])
    np.testing.assert_array_equal(np.asarray(tangent)[[0, 1, 2, 3]], 0)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("rows,cols", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_empty_graph_preserves_ring_and_gradients(weighted, rows, cols):
    args = _operands(weighted)
    args[:7] = [jnp.ones((5, cols), jnp.float32), jnp.zeros(rows, jnp.float32),
                jnp.zeros(0, jnp.float32), jnp.zeros(0, jnp.int32),
                jnp.zeros(rows + 1, jnp.int32), jnp.zeros((rows, 4), jnp.int32),
                jnp.arange(4, dtype=jnp.int32) * cols]
    args[8] = jnp.zeros(rows, jnp.int32)
    result = jax.jit(lambda *xs: _project(xs, weighted=weighted))(*args)
    np.testing.assert_array_equal(result[0], args[0])
    grad = jax.jit(jax.grad(lambda ring: _project((ring, *args[1:]), weighted=weighted)[0].sum()))(args[0])
    np.testing.assert_array_equal(grad, np.ones((5, cols), np.float32))


@pytest.mark.parametrize("head", [-1, 5])
@pytest.mark.parametrize("check_status", [False, True])
def test_dynamic_invalid_head_is_always_checked(head, check_status):
    args = _operands()
    fn = jax.jit(lambda h: _project((*args[:7], h, *args[8:]), check_status=check_status))
    with pytest.raises(Exception, match="head"):
        jax.block_until_ready(fn(jnp.array([head], jnp.int32)))
    jax.block_until_ready(fn(args[7]))
    # GPU runtimes may retain the already-asserted failed callback's token.
    jax.jit(lambda: jax.debug.callback(lambda: None, ordered=False))()
    jax.effects_barrier()


def _long_row(weighted=False):
    nnz = 129
    queue = jnp.zeros(1, jnp.int32)
    workspace = [queue, queue, queue]
    if weighted:
        workspace.append(jnp.zeros(1, jnp.float32))
    workspace.append(jnp.zeros(2, jnp.int32))
    return [jnp.ones((2, 1), jnp.float32), jnp.array([-2.], jnp.float32),
            jnp.array([.5], jnp.float32), jnp.zeros(nnz, jnp.int32),
            jnp.array([0, nnz], jnp.int32), jnp.array([[nnz]], jnp.int32),
            jnp.array([1], jnp.int32), jnp.array([1], jnp.int32),
            jnp.ones(1, jnp.int32), *workspace]


@pytest.mark.parametrize("field", ["counts", "tasks", "capacity"])
def test_checked_malformed_metadata_reports_an_error(field):
    args = _long_row()
    if field == "counts":
        args[5] = jnp.array([[130]], jnp.int32)
    elif field == "tasks":
        args[8] = jnp.zeros(1, jnp.int32)
    else:
        args[9:12] = [value[:0] for value in args[9:12]]
    with pytest.raises(Exception, match="metadata|capacity|task"):
        jax.block_until_ready(jax.jit(lambda *xs: _project(xs))(*args))
    jax.block_until_ready(jax.jit(lambda *xs: _project(xs))(*_long_row()))
    jax.jit(lambda: jax.debug.callback(lambda: None, ordered=False))()
    jax.effects_barrier()


def test_workspace_values_are_unused_and_have_zero_derivatives():
    args = _long_row(weighted=True)
    args[9:12] = [jnp.full_like(value, 17) for value in args[9:12]]
    args[12] = jnp.full_like(args[12], 42)
    result = jax.jit(lambda *xs: _project(xs, weighted=True))(*args)
    np.testing.assert_array_equal(result[0], [[-128.], [1.]])
    for workspace in result[1:-1]:
        np.testing.assert_array_equal(workspace, 0)
    np.testing.assert_array_equal(result[-1][1], 0)

    def loss(task_value, events):
        values = [args[0], events, *args[2:12], task_value, args[13]]
        return _project(values, weighted=True)[4].sum()

    for gradient in jax.jit(jax.grad(loss, argnums=(0, 1)))(args[12], args[1]):
        np.testing.assert_array_equal(gradient, 0)


@GPU
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_jax_backend_matches_cuda_on_the_same_prepared_arrays(weighted, homogeneous):
    args = _operands(weighted, homogeneous)
    jax_ring = jax.jit(lambda *xs: _project(xs, weighted=weighted)[0])(*args)
    cuda_ring = jax.jit(lambda *xs: _project(xs, weighted=weighted, backend="cuda_raw")[0])(*args)
    np.testing.assert_array_equal(jax_ring, cuda_ring)


def test_backends_use_platform_defaults_without_aliases():
    assert set(csr_delay_deliver_layout_p.available_backends("cpu")) == {"jax_raw"}
    assert set(csr_delay_deliver_layout_p.available_backends("gpu")) == {"cuda_raw", "jax_raw"}
    args = _operands()
    graph = jax.make_jaxpr(lambda *xs: csr_delay_deliver_layout_p_call(*xs))(*args).jaxpr
    assert graph.eqns[-1].params["backend"] is None
