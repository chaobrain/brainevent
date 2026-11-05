"""Prepared delivery primitive: registration, call validation and backend dispatch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.delay import delay, region_delay as state_module
from brainevent._csr.delay.test._helpers import BACKENDS, GPU


def _operands(weighted):
    ring = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    events = jnp.array([-2, .5], jnp.float32)
    data = jnp.array([2, -1, 3, 4], jnp.float32)
    indices = jnp.array([0, 1, 1, 0], jnp.int32)
    indptr = jnp.array([0, 2, 4], jnp.int32)
    counts = jnp.ones((2, 2), jnp.int32)
    offsets, head = jnp.array([0, 4], jnp.int32), jnp.array([1], jnp.int32)
    ranges = tuple(jnp.zeros(0, jnp.int32) for _ in range(3))
    status = jnp.zeros(2, jnp.int32)
    workspace = (delay.WeightedHybridWorkspace(*ranges, jnp.zeros(0, jnp.float32), status)
                 if weighted else delay.HybridWorkspace(*ranges, status))
    cuda = (ring, events, data, indices, indptr, counts, offsets, head,
            jnp.zeros(2, jnp.int32), *workspace)
    expected = np.array([[2, 3], [-2, 4.5], [4, 5]] if weighted
                        else [[4, 0], [4, 6], [4, 5]], np.float32)
    return cuda, expected


def _args(weighted=False):
    operands, _ = _operands(weighted)
    return list(operands)


def _trace(args, **kwargs):
    return jax.make_jaxpr(
        lambda *xs: delay.csr_delay_deliver_layout_p.call(*xs, **kwargs)
    )(*args).jaxpr


@pytest.mark.parametrize("weighted", [False, True])
def test_matmul_binds_the_shared_primitive_directly(monkeypatch, weighted):
    backend = "cuda_raw"
    cuda, _ = _operands(weighted)
    state = state_module.delaycsr(
        (cuda[2], cuda[3], cuda[4]), shape=(2, 2),
        edge_delays=np.array([0, 2, 0, 2], np.int32), delay_steps=(0, 2),
        buffer_slots=3, sorted_by_delay=True, weighted=weighted, backend=backend,
        ring=cuda[0], head=1,
    )
    calls = []
    primitive = delay.csr_delay_deliver_layout_p

    def bind(*args, **kwargs):
        calls.append(kwargs)
        return primitive(*args, **kwargs)

    monkeypatch.setattr(state_module, "csr_delay_deliver_layout_p", bind)

    def step(matrix, events):
        spike = matrix.__rmatmul__(events)
        return matrix, spike

    equations = jax.make_jaxpr(step)(state, cuda[1]).jaxpr.eqns
    deliveries = [equation for equation in equations if equation.primitive is primitive.primitive]
    assert len(calls) == len(deliveries) == 1
    assert calls[0]["backend"] == deliveries[0].params["backend"] == backend
    assert calls[0]["weighted"] is weighted
    assert calls[0]["check_status"] is True
    assert len(calls[0]["outs"]) == (6 if weighted else 5)


@pytest.mark.parametrize("backend", [pytest.param("cuda_raw", marks=GPU), "jax_raw"])
@pytest.mark.parametrize("weighted", [False, True])
def test_raw_primitive_selects_backend_and_preserves_output_contract(backend, weighted):
    operands, expected = _operands(weighted)
    mutable = operands[9:]
    outputs = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in (operands[0], *mutable))

    def project(*args):
        return delay.csr_delay_deliver_layout_p(
            *args, outs=outputs, backend=backend, weighted=weighted, check_status=True)

    result = jax.jit(project)(*operands)
    np.testing.assert_array_equal(result[0], expected)
    assert len(result) == len(mutable) + 1
    for output, workspace in zip(result[1:], mutable):
        assert output.shape == workspace.shape
        assert output.dtype == workspace.dtype
    np.testing.assert_array_equal(result[-1], [0, 0])


def test_raw_primitive_omitted_backend_supports_platform_default_and_grad():
    operands, expected = _operands(weighted=True)
    outputs = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype)
                    for value in (operands[0], *operands[9:]))

    def loss(weights):
        return delay.csr_delay_deliver_layout_p(
            *operands[:2], weights, *operands[3:],
            outs=outputs, weighted=True, check_status=True,
        )[0].sum()

    value, gradient = jax.jit(jax.value_and_grad(loss))(operands[2])
    np.testing.assert_array_equal(value, expected.sum())
    np.testing.assert_array_equal(gradient, [-2, -2, .5, .5])


@pytest.mark.parametrize("backend", [None, "cuda_raw", "jax_raw"])
@pytest.mark.parametrize("weighted", [False, True])
def test_registered_call_preserves_operands_and_infers_outputs(backend, weighted):
    args = _args(weighted)
    graph = _trace(args, backend=backend, weighted=np.bool_(weighted), check_status=np.bool_(False))
    primitive = delay.csr_delay_deliver_layout_p
    equations = [eq for eq in graph.eqns if eq.primitive is primitive.primitive]
    assert len(equations) == 1
    equation = equations[0]
    assert equation.params["backend"] == backend
    assert equation.params["weighted"] is weighted
    assert equation.params["check_status"] is False
    assert equation.invars == graph.invars
    mutable = args[9:]
    expected = [args[0], *mutable]
    assert len(graph.outvars) == len(expected)
    for output, operand in zip(graph.outvars, expected):
        assert output.aval.shape == operand.shape
        assert output.aval.dtype == operand.dtype


def test_call_is_exported_tagged_and_preserves_the_platform_default():
    assert "csr_delay_deliver_layout_p_call" in delay.__all__
    assert {"csr", "delay"} <= delay.csr_delay_deliver_layout_p._tags
    args = _args()
    registered = _trace(args)
    direct = jax.make_jaxpr(delay.csr_delay_deliver_layout_p_call)(*args).jaxpr
    assert str(registered) == str(direct)
    assert registered.eqns[-1].params["backend"] is None
    assert registered.eqns[-1].params["check_status"] is True


def test_numpy_inputs_and_scalar_shared_weight_are_normalized():
    args = [np.asarray(value) for value in _args()]
    args[2] = np.float32(2)
    graph = _trace(args)
    equation = graph.eqns[-1]
    assert equation.invars[2].aval.shape == (1,)
    assert equation.invars[2].aval.dtype == jnp.float32
    assert graph.outvars[0].aval.shape == args[0].shape


@pytest.mark.parametrize("backend", ["cuda_raw", "jax_raw"])
def test_call_does_not_wrap_head(backend):
    args = _args()
    args[7] = jnp.array([-1], jnp.int32)
    graph = _trace(args, backend=backend)
    # Range validation remains a device check; the call must not turn -1 into L-1.
    equation = graph.eqns[-1]
    assert equation.invars[7] == graph.invars[7]
    assert len(graph.eqns) == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("head", [-1, 3])
def test_concrete_invalid_head_is_reported_on_execution(backend, head):
    args = _args()
    args[7] = jnp.array([head], jnp.int32)
    with pytest.raises(Exception, match="head"):
        jax.block_until_ready(delay.csr_delay_deliver_layout_p.call(*args, backend=backend))


@pytest.mark.parametrize("field", ["indices", "indptr", "offsets"])
def test_cuda_accepts_independent_int64_structure_types(field):
    args = _args()
    positions = {"indices": (3,), "indptr": (4, 5, 9, 10), "offsets": (6, 11)}
    for position in positions[field]:
        args[position] = args[position].astype(jnp.int64)
    graph = _trace(args)
    equation = graph.eqns[-1]
    for position in positions[field]:
        assert equation.invars[position].aval.dtype == jnp.int64


@pytest.mark.parametrize("shape", [(2**31, 2), (65536, 65536)])
def test_oversized_ring_is_checked_without_allocating_it(shape):
    args = _args()
    args[0] = jax.ShapeDtypeStruct(shape, jnp.float32)
    with pytest.raises(ValueError, match="int32|INT32_MAX"):
        _trace(args)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "cusparse", "invalid"])
def test_unknown_backend_is_rejected_before_lowering(backend):
    with pytest.raises(ValueError, match="backend"):
        _trace(_args(), backend=backend)


@pytest.mark.parametrize("parameter", ["weighted", "check_status"])
@pytest.mark.parametrize("value", [0, 1, None, "False"])
def test_static_flags_require_booleans(parameter, value):
    # A valid bool and its numeric lookalike share Python's cache key. The
    # public entry must still reject 0/1 after a valid call has been traced.
    valid_flag = bool(value) if isinstance(value, int) else False
    args = _args(weighted=parameter == 'weighted' and valid_flag)
    _trace(args, **{parameter: valid_flag})
    with pytest.raises(TypeError, match=parameter):
        _trace(args, **{parameter: value})


@pytest.mark.parametrize("backend", ["cuda_raw", "jax_raw"])
@pytest.mark.parametrize(
    "position,shape,dtype",
    [
        pytest.param(0, (6,), jnp.float32, id="ring-rank"),
        pytest.param(0, (0, 2), jnp.float32, id="ring-no-slots"),
        pytest.param(0, (3, 2), jnp.int32, id="ring-dtype"),
        pytest.param(1, (3,), jnp.float32, id="event-count"),
        pytest.param(1, (2,), jnp.complex64, id="binary-event-dtype"),
        pytest.param(2, (3,), jnp.float32, id="weight-count"),
        pytest.param(2, (2, 2), jnp.float32, id="weight-rank"),
        pytest.param(2, (4,), jnp.float64, id="weight-dtype"),
        pytest.param(3, (2, 2), jnp.int32, id="indices-rank"),
        pytest.param(3, (4,), jnp.uint32, id="indices-dtype"),
        pytest.param(6, (2, 1), jnp.int32, id="offsets-rank"),
        pytest.param(6, (1,), jnp.int32, id="offsets-count"),
        pytest.param(6, (2,), jnp.float32, id="offsets-dtype"),
        pytest.param(7, (), jnp.int32, id="head-scalar"),
        pytest.param(7, (1,), jnp.int64, id="head-dtype"),
        pytest.param(4, (1, 3), jnp.int32, id="indptr-rank"),
        pytest.param(4, (0,), jnp.int32, id="indptr-empty"),
        pytest.param(4, (3,), jnp.float32, id="indptr-dtype"),
        pytest.param(5, (2, 1), jnp.int32, id="counts-shape"),
        pytest.param(5, (2, 2), jnp.int64, id="counts-dtype"),
        pytest.param(8, (1,), jnp.int32, id="row-tasks-count"),
        pytest.param(8, (2,), jnp.int64, id="row-tasks-dtype"),
        pytest.param(9, (0, 1), jnp.int32, id="task-begin-rank"),
        pytest.param(9, (0,), jnp.int64, id="task-begin-dtype"),
        pytest.param(10, (1,), jnp.int32, id="task-end-capacity"),
        pytest.param(10, (0,), jnp.int64, id="task-end-dtype"),
        pytest.param(11, (1,), jnp.int32, id="task-offset-capacity"),
        pytest.param(11, (0,), jnp.int64, id="task-offset-dtype"),
        pytest.param(12, (1,), jnp.int32, id="status-count"),
        pytest.param(12, (2,), jnp.int64, id="status-dtype"),
    ],
)
def test_invalid_metadata_is_rejected_during_tracing(backend, position, shape, dtype):
    args = _args()
    args[position] = jnp.zeros(shape, dtype)
    with pytest.raises((TypeError, ValueError)):
        _trace(args, backend=backend)


@pytest.mark.parametrize("backend", ["cuda_raw", "jax_raw"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("extra", [False, True])
def test_workspace_arity_is_checked(backend, weighted, extra):
    args = _args(weighted)
    args = [*args, args[-1]] if extra else args[:-1]
    with pytest.raises((TypeError, ValueError)):
        _trace(args, backend=backend, weighted=weighted)


@pytest.mark.parametrize("position,shape,dtype", [
    (1, (2,), jnp.complex64),
    (12, (1,), jnp.float32),
    (12, (0,), jnp.float64),
])
@pytest.mark.parametrize("backend", ["cuda_raw", "jax_raw"])
def test_weighted_event_and_task_value_contract(position, shape, dtype, backend):
    args = _args(weighted=True)
    args[position] = jnp.zeros(shape, dtype)
    with pytest.raises((TypeError, ValueError)):
        _trace(args, weighted=True, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("event_dtype", [
    jnp.bool_, jnp.int8, jnp.uint8, jnp.int16, jnp.uint16, jnp.int32,
    jnp.uint32, jnp.int64, jnp.uint64, jnp.float16, jnp.bfloat16,
    jnp.float32, jnp.float64,
])
def test_numeric_events_keep_the_main_library_dtype_contract(backend, weighted, event_dtype):
    args = _args(weighted)
    args[1] = jnp.array([0, 3], event_dtype)
    actual = jax.jit(lambda *xs: delay.csr_delay_deliver_layout_p.call(
        *xs, backend=backend, weighted=weighted))(*args)[0]
    expected = np.asarray(args[0]).copy()
    factor = float(args[1][1]) if weighted else 1.
    expected[1, 1] += 3 * factor
    expected[0, 0] += 4 * factor
    assert actual.dtype == args[0].dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("invalid", ["event", "task_value", "out_shape", "out_dtype", "out_count"])
def test_raw_primitive_checks_the_same_abstract_contract(invalid):
    args = _args(weighted=True)
    if invalid == "event":
        args[1] = args[1].astype(jnp.complex64)
    elif invalid == "task_value":
        args[12] = args[12].astype(jnp.float64)
    outputs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in (args[0], *args[9:])]
    if invalid == "out_shape":
        outputs[0] = jax.ShapeDtypeStruct((2, 3), jnp.float32)
    elif invalid == "out_dtype":
        outputs[0] = jax.ShapeDtypeStruct(args[0].shape, jnp.float64)
    elif invalid == "out_count":
        outputs.pop()
    with pytest.raises((TypeError, ValueError)):
        jax.make_jaxpr(lambda *xs: delay.csr_delay_deliver_layout_p(
            *xs, outs=tuple(outputs), backend="jax_raw", weighted=True,
            check_status=True))(*args)


@pytest.mark.parametrize("backend", BACKENDS)
def test_binary_activity_is_determined_before_lossy_event_conversion(backend):
    args = _args()
    args[1] = jnp.array([1e-300, 0], jnp.float64)
    args[2] = args[2].at[2:].set(jnp.array([jnp.nan, jnp.inf], jnp.float32))
    actual = jax.jit(lambda *xs: delay.csr_delay_deliver_layout_p.call(
        *xs, backend=backend))(*args)[0]
    np.testing.assert_array_equal(actual, [[0, 0], [4, 3], [4, 5]])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weighted", [False, True])
def test_mixed_event_dtype_jvp_and_vjp_preserve_operand_dtypes(backend, weighted):
    args = _args(weighted)
    events = args[1].astype(jnp.float64)

    def project(x, data):
        return delay.csr_delay_deliver_layout_p.call(
            args[0], x, data, *args[3:], backend=backend, weighted=weighted)[0]

    _, tangent = jax.jit(lambda x: jax.jvp(
        lambda y: project(y, args[2]), (x,),
        (jnp.array([.25, -.5], jnp.float64),)))(events)
    np.testing.assert_array_equal(tangent, [[-2, -.25], [.5, -1.5], [0, 0]])
    dx, dw = jax.jit(jax.grad(lambda x, w: project(x, w).sum(), argnums=(0, 1)))(events, args[2])
    assert dx.dtype == jnp.float64 and dw.dtype == jnp.float32
    np.testing.assert_array_equal(dx, [1, 7])
    np.testing.assert_array_equal(dw, [-2, -2, .5, .5] if weighted else [1, 1, 1, 1])


@pytest.mark.parametrize("backend", ["cuda_raw", "jax_raw"])
@pytest.mark.parametrize("rows,cols", [(0, 0), (0, 2), (2, 0)])
def test_empty_graph_shapes_are_valid(backend, rows, cols):
    args = _args()
    args[:5] = [jnp.zeros((3, cols), jnp.float32), jnp.zeros(rows, jnp.float32),
                jnp.zeros(0, jnp.float32), jnp.zeros(0, jnp.int32),
                jnp.zeros((rows + 1,), jnp.int32)]
    args[5] = jnp.zeros((rows, 2), jnp.int32)
    args[6] = jnp.array([0, 2 * cols], jnp.int32)
    args[8] = jnp.zeros(rows, jnp.int32)
    graph = _trace(args, backend=backend)
    assert graph.outvars[0].aval.shape == (3, cols)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weighted", [False, True])
def test_registered_call_executes_prepared_backends(backend, weighted):
    args = _args(weighted)
    result = jax.jit(lambda *xs: delay.csr_delay_deliver_layout_p.call(
        *xs, backend=backend, weighted=weighted))(*args)
    np.testing.assert_array_equal(result[0], _operands(weighted)[1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_registered_call_preserves_autodiff_and_nonleading_vmap(backend):
    args = _args(weighted=True)
    rows = jnp.array([0, 0, 1, 1])
    delays = jnp.array([0, 2, 0, 2])

    def project(ring, events, weights):
        return delay.csr_delay_deliver_layout_p.call(
            ring, events, weights, *args[3:], backend=backend, weighted=True)[0]

    def reference(ring, events, weights):
        return ring.at[
            (args[7][0] + delays) % 3, args[3]
        ].add(weights * events[rows])

    inputs = tuple(args[:3])
    tangents = tuple(jnp.ones_like(value) / 4 for value in inputs)
    actual = jax.jit(lambda *xs: jax.jvp(project, xs, tangents))(*inputs)
    expected = jax.jvp(reference, inputs, tangents)
    for result, wanted in zip(actual, expected):
        np.testing.assert_allclose(result, wanted, rtol=1e-6, atol=1e-6)

    mapped = jax.vmap(project, in_axes=(2, 1, 0))
    mapped_reference = jax.vmap(reference, in_axes=(2, 1, 0))
    batched = (jnp.stack([args[0], args[0] * 2], axis=2),
               jnp.stack([args[1], args[1] * -2], axis=1),
               jnp.stack([args[2], args[2] / 2]))
    np.testing.assert_allclose(jax.jit(mapped)(*batched), mapped_reference(*batched))
    loss = lambda fn, events: jnp.square(fn(batched[0], events, batched[2])).sum()
    actual = jax.jit(jax.grad(lambda events: loss(mapped, events)))(batched[1])
    expected = jax.grad(lambda events: loss(mapped_reference, events))(batched[1])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
