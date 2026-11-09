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

"""Tests for ``brainevent/_csr/delay/delay.py``, grouped by behavior."""

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.interpreters import ad

from brainevent._csr.delay import delay
from brainevent._csr.delay import delay as operator
from brainevent._csr.delay import region_delay as state_module
from brainevent._csr.delay.delay import (
    _edge_locations,
    csr_delay_deliver_layout_p,
    csr_delay_deliver_layout_p_call,
)
from brainevent._csr.delay.test._helpers import BACKENDS, GPU


# Primitive binding and abstract contracts
# ----------------------------------------

def _delay_primitive_operands(weighted):
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


def _delay_primitive_args(weighted=False):
    operands, _ = _delay_primitive_operands(weighted)
    return list(operands)


def _delay_primitive_trace(args, **kwargs):
    return jax.make_jaxpr(
        lambda *xs: delay.csr_delay_deliver_layout_p.call(*xs, **kwargs)
    )(*args).jaxpr


@pytest.mark.parametrize("weighted", [False, True])
def test_matmul_binds_the_shared_primitive_directly(monkeypatch, weighted):
    backend = "cuda_raw"
    cuda, _ = _delay_primitive_operands(weighted)
    state = state_module.delaycsr(
        (cuda[2], cuda[3], cuda[4]), shape=(2, 2),
        edge_delays=np.array([0, 2, 0, 2], np.int32), delay_steps=(0, 2),
        n_slots=3, sorted_by_delay=True, weighted=weighted, backend=backend,
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
    operands, expected = _delay_primitive_operands(weighted)
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
    operands, expected = _delay_primitive_operands(weighted=True)
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
    args = _delay_primitive_args(weighted)
    graph = _delay_primitive_trace(args, backend=backend, weighted=np.bool_(weighted), check_status=np.bool_(False))
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
    args = _delay_primitive_args()
    registered = _delay_primitive_trace(args)
    direct = jax.make_jaxpr(delay.csr_delay_deliver_layout_p_call)(*args).jaxpr
    assert str(registered) == str(direct)
    assert registered.eqns[-1].params["backend"] is None
    assert registered.eqns[-1].params["check_status"] is True


def test_numpy_inputs_and_scalar_shared_weight_are_normalized():
    args = [np.asarray(value) for value in _delay_primitive_args()]
    args[2] = np.float32(2)
    graph = _delay_primitive_trace(args)
    equation = graph.eqns[-1]
    assert equation.invars[2].aval.shape == (1,)
    assert equation.invars[2].aval.dtype == jnp.float32
    assert graph.outvars[0].aval.shape == args[0].shape


@pytest.mark.parametrize("backend", ["cuda_raw", "jax_raw"])
def test_call_does_not_wrap_head(backend):
    args = _delay_primitive_args()
    args[7] = jnp.array([-1], jnp.int32)
    graph = _delay_primitive_trace(args, backend=backend)
    # Range validation remains a device check; the call must not turn -1 into L-1.
    equation = graph.eqns[-1]
    assert equation.invars[7] == graph.invars[7]
    assert len(graph.eqns) == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("head", [-1, 3])
def test_concrete_invalid_head_is_reported_on_execution(backend, head):
    args = _delay_primitive_args()
    args[7] = jnp.array([head], jnp.int32)
    with pytest.raises(Exception, match="head"):
        jax.block_until_ready(delay.csr_delay_deliver_layout_p.call(*args, backend=backend))


@pytest.mark.parametrize("field", ["indices", "indptr", "offsets"])
def test_cuda_accepts_independent_int64_structure_types(field):
    args = _delay_primitive_args()
    positions = {"indices": (3,), "indptr": (4, 5, 9, 10), "offsets": (6, 11)}
    for position in positions[field]:
        args[position] = args[position].astype(jnp.int64)
    graph = _delay_primitive_trace(args)
    equation = graph.eqns[-1]
    for position in positions[field]:
        assert equation.invars[position].aval.dtype == jnp.int64


@pytest.mark.parametrize("shape", [(2**31, 2), (65536, 65536)])
def test_oversized_ring_is_checked_without_allocating_it(shape):
    args = _delay_primitive_args()
    args[0] = jax.ShapeDtypeStruct(shape, jnp.float32)
    with pytest.raises(ValueError, match="int32|INT32_MAX"):
        _delay_primitive_trace(args)


@pytest.mark.parametrize("backend", ["cpu", "cuda", "cusparse", "invalid"])
def test_unknown_backend_is_rejected_before_lowering(backend):
    with pytest.raises(ValueError, match="backend"):
        _delay_primitive_trace(_delay_primitive_args(), backend=backend)


@pytest.mark.parametrize("parameter", ["weighted", "check_status"])
@pytest.mark.parametrize("value", [0, 1, None, "False"])
def test_static_flags_require_booleans(parameter, value):
    # A valid bool and its numeric lookalike share Python's cache key. The
    # public entry must still reject 0/1 after a valid call has been traced.
    valid_flag = bool(value) if isinstance(value, int) else False
    args = _delay_primitive_args(weighted=parameter == 'weighted' and valid_flag)
    _delay_primitive_trace(args, **{parameter: valid_flag})
    with pytest.raises(TypeError, match=parameter):
        _delay_primitive_trace(args, **{parameter: value})


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
    args = _delay_primitive_args()
    args[position] = jnp.zeros(shape, dtype)
    with pytest.raises((TypeError, ValueError)):
        _delay_primitive_trace(args, backend=backend)


@pytest.mark.parametrize("backend", ["cuda_raw", "jax_raw"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("extra", [False, True])
def test_workspace_arity_is_checked(backend, weighted, extra):
    args = _delay_primitive_args(weighted)
    args = [*args, args[-1]] if extra else args[:-1]
    with pytest.raises((TypeError, ValueError)):
        _delay_primitive_trace(args, backend=backend, weighted=weighted)


@pytest.mark.parametrize("position,shape,dtype", [
    (1, (2,), jnp.complex64),
    (12, (1,), jnp.float32),
    (12, (0,), jnp.float64),
])
@pytest.mark.parametrize("backend", ["cuda_raw", "jax_raw"])
def test_weighted_event_and_task_value_contract(position, shape, dtype, backend):
    args = _delay_primitive_args(weighted=True)
    args[position] = jnp.zeros(shape, dtype)
    with pytest.raises((TypeError, ValueError)):
        _delay_primitive_trace(args, weighted=True, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("event_dtype", [
    jnp.bool_, jnp.int8, jnp.uint8, jnp.int16, jnp.uint16, jnp.int32,
    jnp.uint32, jnp.int64, jnp.uint64, jnp.float16, jnp.bfloat16,
    jnp.float32, jnp.float64,
])
def test_numeric_events_keep_the_main_library_dtype_contract(backend, weighted, event_dtype):
    args = _delay_primitive_args(weighted)
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
    args = _delay_primitive_args(weighted=True)
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
    args = _delay_primitive_args()
    args[1] = jnp.array([1e-300, 0], jnp.float64)
    args[2] = args[2].at[2:].set(jnp.array([jnp.nan, jnp.inf], jnp.float32))
    actual = jax.jit(lambda *xs: delay.csr_delay_deliver_layout_p.call(
        *xs, backend=backend))(*args)[0]
    np.testing.assert_array_equal(actual, [[0, 0], [4, 3], [4, 5]])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weighted", [False, True])
def test_mixed_event_dtype_jvp_and_vjp_preserve_operand_dtypes(backend, weighted):
    args = _delay_primitive_args(weighted)
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
    args = _delay_primitive_args()
    args[:5] = [jnp.zeros((3, cols), jnp.float32), jnp.zeros(rows, jnp.float32),
                jnp.zeros(0, jnp.float32), jnp.zeros(0, jnp.int32),
                jnp.zeros((rows + 1,), jnp.int32)]
    args[5] = jnp.zeros((rows, 2), jnp.int32)
    args[6] = jnp.array([0, 2 * cols], jnp.int32)
    args[8] = jnp.zeros(rows, jnp.int32)
    graph = _delay_primitive_trace(args, backend=backend)
    assert graph.outvars[0].aval.shape == (3, cols)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weighted", [False, True])
def test_registered_call_executes_prepared_backends(backend, weighted):
    args = _delay_primitive_args(weighted)
    result = jax.jit(lambda *xs: delay.csr_delay_deliver_layout_p.call(
        *xs, backend=backend, weighted=weighted))(*args)
    np.testing.assert_array_equal(result[0], _delay_primitive_operands(weighted)[1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_registered_call_preserves_autodiff_and_nonleading_vmap(backend):
    args = _delay_primitive_args(weighted=True)
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


# JAX execution, autodiff and batching
# ------------------------------------

_delay_jax_ROWS = np.array([0, 0, 0, 2, 2, 2], np.int32)


_delay_jax_COLUMNS = np.array([0, 2, 1, 1, 0, 2], np.int32)


_delay_jax_DELAYS = np.array([0, 1, 3, 0, 1, 3], np.int32)


def _delay_jax_operands(weighted=False, homogeneous=False, dtype=jnp.float32):
    """Prepare two nonempty rows, one empty row and one empty delay segment."""
    queue = jnp.zeros(0, jnp.int32)
    workspace = [queue, queue, queue]
    if weighted:
        workspace.append(jnp.zeros(0, dtype))
    workspace.append(jnp.zeros(2, jnp.int32))
    return [jnp.arange(15, dtype=dtype).reshape(5, 3) / 8,
            jnp.array([-2, 0, .5], dtype if weighted else jnp.float32),
            jnp.array([.5] if homogeneous else [2, -1, .5, 3, -2, .25], dtype),
            jnp.asarray(_delay_jax_COLUMNS), jnp.array([0, 3, 3, 6], jnp.int32),
            jnp.array([[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1]], jnp.int32),
            jnp.array([0, 3, 6, 9], jnp.int32), jnp.array([4], jnp.int32),
            jnp.zeros(3, jnp.int32), *workspace]


def _delay_jax_project(args, *, weighted=False, backend="jax_raw", check_status=True):
    return csr_delay_deliver_layout_p_call(
        *args, backend=backend, weighted=weighted, check_status=check_status)


def _delay_jax_numpy_reference(ring, events, data, head, weighted):
    result = np.asarray(ring).copy()
    weights = np.broadcast_to(np.asarray(data), (6,))
    values = np.asarray(events)
    for edge, row in enumerate(_delay_jax_ROWS):
        if values[row] != 0:
            value = weights[edge] * values[row] if weighted else weights[edge]
            result[(int(np.asarray(head)[0]) + int(_delay_jax_DELAYS[edge])) % 5, _delay_jax_COLUMNS[edge]] += value
    return result


@jax.custom_jvp
def _delay_jax_activity(events):
    return (events != 0).astype(events.dtype)


@_delay_jax_activity.defjvp
def _delay_jax_activity_jvp(primals, tangents):
    return _delay_jax_activity(primals[0]), tangents[0]


def _delay_jax_jax_reference(ring, events, data, head, weighted):
    factors = events if weighted else _delay_jax_activity(events)
    return ring.at[(head[0] + jnp.asarray(_delay_jax_DELAYS)) % 5, jnp.asarray(_delay_jax_COLUMNS)].add(
        jnp.broadcast_to(data, (6,)) * factors[jnp.asarray(_delay_jax_ROWS)])


def _delay_jax_long_row(weighted=False):
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


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_jit_delivery_preserves_nonzero_head_and_repeated_accumulation(weighted, homogeneous, dtype):
    if dtype == jnp.float64 and not jax.config.x64_enabled:
        pytest.skip("float64 requires JAX x64")
    args = _delay_jax_operands(weighted, homogeneous, dtype)
    original = np.asarray(args[0]).copy()
    fn = jax.jit(lambda *xs: _delay_jax_project(xs, weighted=weighted))
    result = fn(*args)
    expected = _delay_jax_numpy_reference(*args[:3], args[7], weighted)
    np.testing.assert_array_equal(result[0], expected)
    second = fn(result[0], *args[1:])
    np.testing.assert_array_equal(second[0], _delay_jax_numpy_reference(expected, args[1], args[2], args[7], weighted))
    np.testing.assert_array_equal(args[0], original)
    assert len(result) == (6 if weighted else 5)
    for actual, source in zip(result[1:], args[9:]):
        assert actual.shape == source.shape and actual.dtype == source.dtype
        np.testing.assert_array_equal(actual, 0)


@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.float32])
def test_binary_events_and_silent_step(event_dtype):
    args = _delay_jax_operands()
    args[1] = args[1].astype(event_dtype)
    np.testing.assert_array_equal(_delay_jax_project(args)[0], _delay_jax_numpy_reference(*args[:3], args[7], False))
    args[1] = jnp.zeros_like(args[1])
    np.testing.assert_array_equal(jax.jit(lambda *xs: _delay_jax_project(xs)[0])(*args), args[0])


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_jvp_vjp_and_nonleading_vmap(weighted, homogeneous):
    args = _delay_jax_operands(weighted, homogeneous)
    # An inactive nonempty row must still have an event tangent/STE.
    args[1] = args[1].at[2].set(0)

    def project(ring, events, data, head):
        return _delay_jax_project((ring, events, data, *args[3:7], head, *args[8:]), weighted=weighted)[0]

    ref = lambda ring, events, data, head: _delay_jax_jax_reference(ring, events, data, head, weighted)
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
    args = _delay_jax_operands(weighted)
    args[1] = args[1].at[2].set(0)
    args[2] = args[2].at[3:].set(jnp.array([jnp.nan, jnp.inf, -jnp.inf], args[2].dtype))

    def project(data):
        return _delay_jax_project((args[0], args[1], data, *args[3:]), weighted=weighted)[0]

    np.testing.assert_array_equal(jax.jit(project)(args[2]), _delay_jax_numpy_reference(*args[:3], args[7], weighted))
    _, pullback = jax.vjp(project, args[2])
    cotangent = jnp.ones_like(args[0]).at[4, 1].set(jnp.nan).at[0, 0].set(jnp.inf).at[2, 2].set(-jnp.inf)
    gradient = pullback(cotangent)[0]
    np.testing.assert_array_equal(gradient[3:], 0)


@pytest.mark.parametrize("weighted", [False, True])
def test_mixed_event_jvp_keeps_activity_before_value_conversion(weighted):
    args = _delay_jax_operands(weighted)
    args[1] = jnp.zeros(3, jnp.float64)
    args[2] = args[2].at[0].set(jnp.inf)

    def project(events):
        return _delay_jax_project((args[0], events, *args[2:]), weighted=weighted)[0]

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
    args = _delay_jax_operands(weighted)
    args[:7] = [jnp.ones((5, cols), jnp.float32), jnp.zeros(rows, jnp.float32),
                jnp.zeros(0, jnp.float32), jnp.zeros(0, jnp.int32),
                jnp.zeros(rows + 1, jnp.int32), jnp.zeros((rows, 4), jnp.int32),
                jnp.arange(4, dtype=jnp.int32) * cols]
    args[8] = jnp.zeros(rows, jnp.int32)
    result = jax.jit(lambda *xs: _delay_jax_project(xs, weighted=weighted))(*args)
    np.testing.assert_array_equal(result[0], args[0])
    grad = jax.jit(jax.grad(lambda ring: _delay_jax_project((ring, *args[1:]), weighted=weighted)[0].sum()))(args[0])
    np.testing.assert_array_equal(grad, np.ones((5, cols), np.float32))


@pytest.mark.parametrize("head", [-1, 5])
@pytest.mark.parametrize("check_status", [False, True])
def test_dynamic_invalid_head_is_always_checked(head, check_status):
    args = _delay_jax_operands()
    fn = jax.jit(lambda h: _delay_jax_project((*args[:7], h, *args[8:]), check_status=check_status))
    with pytest.raises(Exception, match="head"):
        jax.block_until_ready(fn(jnp.array([head], jnp.int32)))
    jax.block_until_ready(fn(args[7]))
    # GPU runtimes may retain the already-asserted failed callback's token.
    jax.jit(lambda: jax.debug.callback(lambda: None, ordered=False))()
    jax.effects_barrier()


@pytest.mark.parametrize("field", ["counts", "tasks", "capacity"])
def test_checked_malformed_metadata_reports_an_error(field):
    args = _delay_jax_long_row()
    if field == "counts":
        args[5] = jnp.array([[130]], jnp.int32)
    elif field == "tasks":
        args[8] = jnp.zeros(1, jnp.int32)
    else:
        args[9:12] = [value[:0] for value in args[9:12]]
    with pytest.raises(Exception, match="metadata|capacity|task"):
        jax.block_until_ready(jax.jit(lambda *xs: _delay_jax_project(xs))(*args))
    jax.block_until_ready(jax.jit(lambda *xs: _delay_jax_project(xs))(*_delay_jax_long_row()))
    jax.jit(lambda: jax.debug.callback(lambda: None, ordered=False))()
    jax.effects_barrier()


def test_workspace_values_are_unused_and_have_zero_derivatives():
    args = _delay_jax_long_row(weighted=True)
    args[9:12] = [jnp.full_like(value, 17) for value in args[9:12]]
    args[12] = jnp.full_like(args[12], 42)
    result = jax.jit(lambda *xs: _delay_jax_project(xs, weighted=True))(*args)
    np.testing.assert_array_equal(result[0], [[-128.], [1.]])
    for workspace in result[1:-1]:
        np.testing.assert_array_equal(workspace, 0)
    np.testing.assert_array_equal(result[-1][1], 0)

    def loss(task_value, events):
        values = [args[0], events, *args[2:12], task_value, args[13]]
        return _delay_jax_project(values, weighted=True)[4].sum()

    for gradient in jax.jit(jax.grad(loss, argnums=(0, 1)))(args[12], args[1]):
        np.testing.assert_array_equal(gradient, 0)


@GPU
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_jax_backend_matches_cuda_on_the_same_prepared_arrays(weighted, homogeneous):
    args = _delay_jax_operands(weighted, homogeneous)
    jax_ring = jax.jit(lambda *xs: _delay_jax_project(xs, weighted=weighted)[0])(*args)
    cuda_ring = jax.jit(lambda *xs: _delay_jax_project(xs, weighted=weighted, backend="cuda_raw")[0])(*args)
    np.testing.assert_array_equal(jax_ring, cuda_ring)


def test_backends_use_platform_defaults_without_aliases():
    assert set(csr_delay_deliver_layout_p.available_backends("cpu")) == {"jax_raw"}
    assert set(csr_delay_deliver_layout_p.available_backends("gpu")) == {"cuda_raw", "jax_raw"}
    args = _delay_jax_operands()
    graph = jax.make_jaxpr(lambda *xs: csr_delay_deliver_layout_p_call(*xs))(*args).jaxpr
    assert graph.eqns[-1].params["backend"] is None


# Wide ring indexing
# ------------------

@pytest.mark.parametrize("slots,cols,offset_dtype,x64", [
    (65536, 32767, np.int32, False),
    (50000, 40000, np.int32, False),
    (2**31 - 1, 1, np.int32, False),
    (65536, 32769, np.int64, True),
    (2**31 - 1, 2**30 - 1, np.int64, True),
    (1, 5, np.int32, False),
])
@pytest.mark.parametrize("undefined_ring", [False, True], ids=["forward", "transpose"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_large_ring_slots_do_not_overflow(slots, cols, offset_dtype, x64,
                                         undefined_ring, compiled):
    # Python integer arithmetic is the reference, including when even the
    # slot-level head + delay exceeds INT32_MAX. Only O(nnz) arrays are created.
    delays = sorted({0, min(1, slots - 1), slots // 2, slots - 1})
    counts = np.ones((2, len(delays)), np.int32)
    counts[1, 0] = 0
    counts[1, -1] = 2
    edge_rows = [row for row in range(2) for k in range(len(delays))
                 for _ in range(int(counts[row, k]))]
    edge_delays = [delay for row in range(2) for k, delay in enumerate(delays)
                   for _ in range(int(counts[row, k]))]
    nnz = len(edge_rows)
    ring = (ad.UndefinedPrimal(jax.core.ShapedArray((slots, cols), jnp.float32))
            if undefined_ring else jax.ShapeDtypeStruct((slots, cols), jnp.float32))

    with jax.enable_x64(x64):
        args = (ring, jnp.ones(2), jnp.ones(nnz), jnp.zeros(nnz, jnp.int32),
                jnp.asarray([0, int(counts[0].sum()), nnz], jnp.int32),
                jnp.asarray(counts))
        offsets = jnp.asarray([delay * cols for delay in delays], offset_dtype)

        def locate(head, offsets):
            return _edge_locations((*args, offsets, head))[1:3]

        if compiled:
            locate = jax.jit(locate)
        for head in sorted({0, min(1, slots - 1), slots // 2, slots - 1}):
            rows, actual = locate(jnp.asarray([head], jnp.int32), offsets)
            expected = [(head + delay) % slots for delay in edge_delays]
            np.testing.assert_array_equal(rows, edge_rows)
            np.testing.assert_array_equal(actual, expected,
                                          err_msg=f"head={head}, shape={(slots, cols)}")
            assert actual.dtype == np.dtype(offset_dtype)


def test_empty_zero_column_ring_has_no_edge_locations():
    with jax.enable_x64(False):
        args = (jax.ShapeDtypeStruct((2**31 - 1, 0), jnp.float32),
                jnp.zeros(2), jnp.zeros(0), jnp.zeros(0, jnp.int32),
                jnp.zeros(3, jnp.int32), jnp.zeros((2, 1), jnp.int32),
                jnp.zeros(1, jnp.int32), jnp.asarray([2**31 - 2], jnp.int32))
        rows, edge_rows, slots, head = jax.jit(lambda: _edge_locations(args))()
        assert int(rows) == 2
        assert edge_rows.shape == slots.shape == (0,)
        np.testing.assert_array_equal(head, args[7])


# Input validation and runtime status
# -----------------------------------

def _input_validation_operands():
    # One long row needs a nonempty task queue, so consuming only a workspace
    # array cannot disappear merely because its shape is zero.
    queue = jnp.zeros(1, jnp.int32)
    return [
        jnp.ones((2, 2), jnp.float32), jnp.array([True]),
        jnp.array([.5], jnp.float32), jnp.zeros(129, jnp.int32),
        jnp.array([0, 129], jnp.int32), jnp.array([[129]], jnp.int32),
        jnp.array([2], jnp.int32), jnp.array([0], jnp.int32),
        jnp.ones(1, jnp.int32), queue, queue, queue, jnp.zeros(2, jnp.int32),
    ]


def _input_validation_project(args, *, backend, entry, check_status, weighted=False):
    if entry == 'call':
        return csr_delay_deliver_layout_p.call(
            *args, backend=backend, weighted=weighted, check_status=check_status)
    outputs = tuple(jax.core.ShapedArray(value.shape, value.dtype)
                    for value in (args[0], *args[9:]))
    return csr_delay_deliver_layout_p.primitive.bind(
        *args, outs=outputs, backend=backend, weighted=weighted, check_status=check_status)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('entry', ['call', 'bind'])
@pytest.mark.parametrize('output', ['ring', 'status', 'workspace'])
@pytest.mark.parametrize('check_status', [True, False])
def test_dynamic_head_checked_for_each_consumed_output(backend, entry, output, check_status):
    args = _input_validation_operands()
    position = {'ring': 0, 'status': -1, 'workspace': 1}[output]
    run = jax.jit(lambda *xs: _input_validation_project(
        xs, backend=backend, entry=entry, check_status=check_status)[position])
    valid = jax.block_until_ready(run(*args))
    if output == 'ring':
        np.testing.assert_array_equal(valid, [[1, 1], [65.5, 1]])
    elif output == 'status':
        np.testing.assert_array_equal(valid[1], 0)
    else:
        assert valid.shape == (1,)

    invalid = list(args)
    invalid[7] = jnp.array([2], jnp.int32)
    with pytest.raises(Exception, match='head'):
        jax.block_until_ready(run(*invalid))
    np.testing.assert_array_equal(jax.block_until_ready(run(*args)), valid)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('field,output', [
    ('indptr_start', 'status'), ('indptr_end', 'workspace'),
    ('counts', 'status'), ('tasks', 'workspace'), ('indices', 'ring'),
    ('offset_range', 'status'), ('offset_alignment', 'workspace'), ('capacity', 'ring'),
])
def test_metadata_validation_and_unchecked_status(backend, field, output):
    args = _input_validation_operands()
    if field == 'counts':
        args[5] = jnp.array([[130]], jnp.int32)
    elif field == 'tasks':
        args[8] = jnp.zeros(1, jnp.int32)
    elif field == 'capacity':
        args[9:12] = [value[:0] for value in args[9:12]]
    else:
        # Metadata validity is independent of activity. Inactive rows also
        # keep the legacy CUDA kernel from dereferencing malformed pointers
        # while this regression test runs against the unfixed implementation.
        args[1] = jnp.array([False])
        if field == 'indptr_start':
            args[4] = jnp.array([1, 129], jnp.int32)
        elif field == 'indptr_end':
            args[4] = jnp.array([0, 130], jnp.int32)
        elif field == 'indices':
            args[3] = args[3].at[0].set(2)
        else:
            args[6] = jnp.array([4 if field == 'offset_range' else 1], jnp.int32)

    unchecked = jax.jit(lambda *xs: _input_validation_project(
        xs, backend=backend, entry='bind', check_status=False)[-1])
    status = jax.block_until_ready(unchecked(*args))
    np.testing.assert_array_equal(status[1], 1 if field == 'capacity' else 3)

    position = {'ring': 0, 'status': -1, 'workspace': 1}[output]
    checked = jax.jit(lambda *xs: _input_validation_project(
        xs, backend=backend, entry='bind', check_status=True)[position])
    with pytest.raises(Exception, match='capacity|metadata|task'):
        jax.block_until_ready(checked(*args))
    jax.block_until_ready(checked(*_input_validation_operands()))


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('check_status', [True, False])
def test_weighted_dynamic_head_checked_when_only_status_is_used(backend, check_status):
    args = _input_validation_operands()
    args[1] = jnp.array([-2], jnp.float32)
    args.insert(-1, jnp.zeros(1, jnp.float32))
    run = jax.jit(lambda head: _input_validation_project(
        (*args[:7], head, *args[8:]), backend=backend, entry='bind',
        check_status=check_status, weighted=True)[-1])
    np.testing.assert_array_equal(jax.block_until_ready(run(args[7]))[1], 0)
    with pytest.raises(Exception, match='head'):
        jax.block_until_ready(run(jnp.array([-1], jnp.int32)))
    np.testing.assert_array_equal(jax.block_until_ready(run(args[7]))[1], 0)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('check_status', [True, False])
def test_empty_ring_dynamic_head_checked_when_only_status_is_used(backend, check_status):
    args = _input_validation_operands()
    args[0] = jnp.zeros((2, 0), jnp.float32)
    args[3] = jnp.zeros(0, jnp.int32)
    args[4] = jnp.zeros(2, jnp.int32)
    args[5] = jnp.zeros((1, 1), jnp.int32)
    args[6] = jnp.zeros(1, jnp.int32)
    args[8] = jnp.zeros(1, jnp.int32)
    run = jax.jit(lambda head: _input_validation_project(
        (*args[:7], head, *args[8:]), backend=backend, entry='call',
        check_status=check_status)[-1])
    np.testing.assert_array_equal(jax.block_until_ready(run(args[7]))[1], 0)
    with pytest.raises(Exception, match='head'):
        jax.block_until_ready(run(jnp.array([2], jnp.int32)))
    np.testing.assert_array_equal(jax.block_until_ready(run(args[7]))[1], 0)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
def test_unchecked_metadata_error_prevents_partial_ring_writes(backend, weighted):
    args = _input_validation_operands()
    args[1] = jnp.array([-2, .5], jnp.float32)
    args[3] = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.ones(129, jnp.int32)))
    args[4] = jnp.array([0, 1, 130], jnp.int32)
    args[5] = jnp.array([[1], [130]], jnp.int32)
    args[8] = jnp.array([0, 1], jnp.int32)
    if weighted:
        args.insert(-1, jnp.zeros(1, jnp.float32))

    def project(*xs):
        result = _input_validation_project(xs, backend=backend, entry='bind', check_status=False, weighted=weighted)
        return result[0], result[-1]

    run = jax.jit(project)
    ring, status = jax.block_until_ready(run(*args))
    np.testing.assert_array_equal(status[1], 3)
    # The first row is valid and active. It must not write before validation
    # rejects the second row's mismatched segment length.
    np.testing.assert_array_equal(ring, args[0])
    args[5] = jnp.array([[1], [129]], jnp.int32)
    ring, status = jax.block_until_ready(run(*args))
    np.testing.assert_array_equal(status[1], 0)
    np.testing.assert_array_equal(ring, [[1, 1], [0, 33.25] if weighted else [1.5, 65.5]])
    np.testing.assert_array_equal(args[0], 1)


@pytest.mark.parametrize('backend', BACKENDS[1:])
def test_jitted_vmapped_head_checks_each_batch_element(backend):
    args = _input_validation_operands()
    run = jax.jit(jax.vmap(lambda head: _input_validation_project(
        (*args[:7], head, *args[8:]), backend=backend, entry='call', check_status=True)[-1]))
    valid = jnp.array([[0], [1]], jnp.int32)
    np.testing.assert_array_equal(jax.block_until_ready(run(valid))[:, 1], 0)
    with pytest.raises(Exception, match='head'):
        jax.block_until_ready(run(valid.at[1, 0].set(2)))
    np.testing.assert_array_equal(jax.block_until_ready(run(valid))[:, 1], 0)


@pytest.mark.parametrize('field', ['counts', 'indptr', 'many_counts'])
def test_input_metadata_overflow_is_rejected_without_x64(field):
    with jax.enable_x64(False):
        args = _input_validation_operands()
        args[3] = jnp.zeros(1, jnp.int32)
        if field == 'counts':
            args[4] = jnp.array([0, 1], jnp.int32)
            args[5] = jnp.array([[2**31 - 1, 2**31 - 1, 3]], jnp.int32)
            args[6] = jnp.zeros(3, jnp.int32)
            args[8] = jnp.zeros(1, jnp.int32)
        elif field == 'indptr':
            args[1] = jnp.zeros(3, jnp.bool_)
            args[4] = jnp.array([0, 1431655765, -1431655766, 1], jnp.int32)
            args[5] = jnp.diff(args[4])[:, None]
            args[6] = jnp.zeros(1, jnp.int32)
            args[8] = args[5][:, 0] // 4096 + (args[5][:, 0] % 4096 != 0)
        else:
            args[1] = jnp.array([False])
            args[3] = jnp.zeros(100000, jnp.int32)
            args[4] = jnp.array([0, 100000], jnp.int32)
            args[5] = jnp.concatenate((jnp.full(42950, 100000, jnp.int32),
                                       jnp.array([67296], jnp.int32)))[None]
            args[6] = jnp.zeros(args[5].shape[1], jnp.int32)
            args[8] = jnp.sum(args[5] // 4096 + (args[5] % 4096 != 0), axis=1)
        # Probe validation alone: these malformed pointers must never reach
        # a delivery implementation, even while testing the unfixed code.
        code = jax.jit(operator._input_error_code)(*args)
        np.testing.assert_array_equal(code, 3)


@pytest.mark.parametrize('check_status', [True, False])
@pytest.mark.parametrize('output', ['status', 'workspace'])
def test_backend_execution_status_uses_shared_output_check(check_status, output):
    def fake_generator(**params):
        def kernel(*args):
            return (args[0], *(jnp.full_like(value, 7) for value in args[9:-1]),
                    jnp.array([0, 3], jnp.int32))
        return kernel

    checked = operator._validated_kernel(fake_generator)(check_status=check_status)
    position = -1 if output == 'status' else 1
    run = jax.jit(lambda *args: checked(*args)[position])
    if check_status:
        with pytest.raises(Exception, match='metadata'):
            jax.block_until_ready(run(*_input_validation_operands()))
    else:
        np.testing.assert_array_equal(jax.block_until_ready(run(*_input_validation_operands())),
                                      [0, 3] if output == 'status' else [7])


# Standalone native interfaces
# ----------------------------

_native_interfaces_ROOT = Path(__file__).resolve().parents[1]


def _native_interfaces_run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@GPU
def test_native_interfaces(tmp_path):
    nvcc, cxx = shutil.which("nvcc"), shutil.which("g++")
    if nvcc is None or cxx is None:
        pytest.skip("nvcc and g++ required")
    cuda = Path(nvcc).resolve().parents[1]
    objects = []
    for stem in ("hybrid_delay_spmv_weighted", "hybrid_delay_spmv_binary"):
        obj = tmp_path / (stem + ".o")
        _native_interfaces_run([nvcc, "-std=c++17", "-O2", "-arch=native", "-DBE_REGION_DELAY_STANDALONE",
              "-c", str(_native_interfaces_ROOT / "hybrid" / (stem + ".cu")), "-o", str(obj)])
        objects.append(str(obj))
    for weighted_only in (True, False):
        caller = tmp_path / f"caller_{weighted_only}.o"
        flags = ["-DREGION_DELAY_TEST_WEIGHTED_ONLY"] if weighted_only else []
        _native_interfaces_run([cxx, "-std=c++17", "-O2", "-I" + str(cuda / "include"), *flags,
              "-c", str(_native_interfaces_ROOT / "test/native_interfaces.cpp"), "-o", str(caller)])
        executable = tmp_path / f"native_{weighted_only}"
        _native_interfaces_run([nvcc, str(caller), *(objects[:1] if weighted_only else objects), "-o", str(executable)])
        _native_interfaces_run([str(executable)])
