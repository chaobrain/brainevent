"""Exercise public delay layout construction, additive delivery, autodiff, and sharding."""

from copy import copy
import importlib

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent import WeightedArray
from brainevent._csr.delay import csr_delay_deliver_layout, delaycsr
from brainevent._csr.delay.test._helpers import BACKENDS, GPU


def make(backend=None, weighted=False, delay=1):
    return delaycsr((np.array([3.], np.float32), np.array([0], np.int32),
                     np.array([0, 1], np.int32)),
                    shape=(1, 1), edge_delays=np.array([delay], np.int32),
                    buffer_slots=2, backend=backend, weighted=weighted,
                    ring=jnp.array([[10.], [20.]], jnp.float32))


def _source(backend):
    return brainevent.CSR(
        (np.array([2, 3, 5, 7], np.float32), np.array([2, 0, 1, 0], np.int32),
         np.array([0, 3, 4], np.int32)), shape=(2, 3), backend=backend,
    )


def test_public_delay_exports_share_one_prepared_implementation():
    packages = [brainevent, importlib.import_module("brainevent._csr"),
                importlib.import_module("brainevent._csr.delay")]
    for package in packages:
        assert package.delaycsr is brainevent.delaycsr
        assert package.csr_delay_deliver_layout is brainevent.csr_delay_deliver_layout
        assert package.csr_delay_deliver_layout_p is brainevent.csr_delay_deliver_layout_p
        for removed in ("CSRDelayLayout", "csr_delay_deliver", "csr_delay_deliver_p"):
            assert not hasattr(package, removed)
    assert not hasattr(brainevent.CSR, "delay_deliver")
    assert set(brainevent.csr_delay_deliver_layout_p.available_backends("gpu")) == {
        "jax_raw", "cuda_raw",
    }


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('delay', [0, 1])
def test_deliver_preserves_head_accumulates_and_keeps_owned_state(backend, weighted, delay):
    state = make(backend, delay=delay)
    leaves = jax.tree.leaves(state)
    events = jnp.array([2.], jnp.float32)
    result = csr_delay_deliver_layout(state.ring, events, state, 0, weighted=weighted)
    again = state.csr_delay_deliver_layout(result, events, 0, weighted=weighted)
    contribution = 6 if weighted else 3
    expected = np.array([[10], [20]])
    expected[delay, 0] += contribution
    np.testing.assert_array_equal(result, expected)
    expected[delay, 0] += contribution
    np.testing.assert_array_equal(again, expected)
    inactive = csr_delay_deliver_layout(state.ring, jnp.zeros_like(events), state, 0,
                                        weighted=weighted)
    np.testing.assert_array_equal(inactive, [[10], [20]])
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(state)))
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    assert int(state.head) == 0


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('t', [-1, 0, 2**40 + 1])
def test_supported_high_level_call_matches_arrival_rule(backend, t):
    state = make(backend)
    events = WeightedArray(jnp.array([-2.], jnp.float32))
    expected = np.asarray(state.ring).copy()
    expected[(t + 1) % 2, 0] += -2 * 3
    actual = jax.jit(lambda r: csr_delay_deliver_layout(r, events, state, t))(state.ring)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('backend', BACKENDS)
def test_delivery_gradients_and_stateful_step_follow_selected_backend(backend):
    state = make(backend, weighted=True)
    events = jnp.array([2.], jnp.float32)
    project = lambda r: state.csr_delay_deliver_layout(r, events, 0, weighted=True)
    result, tangent = jax.jit(lambda r: jax.jvp(project, (r,), (jnp.ones_like(r),)))(state.ring)
    np.testing.assert_array_equal(result, [[10], [26]])
    np.testing.assert_array_equal(tangent, [[1], [1]])

    def loss(weights, ring, event):
        return state.with_data(weights).csr_delay_deliver_layout(ring, event, 0, weighted=True).sum()

    value, (weight_gradient, ring_gradient, event_gradient) = jax.jit(jax.value_and_grad(
        loss, argnums=(0, 1, 2)))(state.data, state.ring, events)
    np.testing.assert_array_equal(value, 36)
    np.testing.assert_array_equal(weight_gradient, [2])
    np.testing.assert_array_equal(ring_gradient, [[1], [1]])
    np.testing.assert_array_equal(event_gradient, [3])
    assert state.backend == backend
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    np.testing.assert_array_equal(state.forward(events, 0), [26])
    np.testing.assert_array_equal(state.ring, [[0], [26]])


def test_clear_head_is_a_jax_member_with_local_gradient():
    state = make('jax_raw')

    def cleared(ring):
        local = copy(state)
        local.ring = ring
        return local.clear_head()

    for operation in (cleared, jax.jit(cleared)):
        value, tangent = jax.jvp(operation, (state.ring,), (jnp.ones_like(state.ring),))
        np.testing.assert_array_equal(value, [[0], [20]])
        np.testing.assert_array_equal(tangent, [[0], [1]])
        np.testing.assert_array_equal(jax.grad(lambda r: operation(r).sum())(state.ring), [[0], [1]])
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    state.head = 1
    assert state.clear_head() is state.ring
    np.testing.assert_array_equal(state.ring, [[10], [0]])
    assert int(state.head) == 1


def test_clear_head_unbatched_state_vmap_without_jit():
    state = make('jax_raw')
    leaves = jax.tree.leaves(state)

    def clear(matrix, head):
        matrix.head = head
        result = matrix.clear_head()
        return matrix, result

    heads = jnp.array([0, 1], jnp.int32)
    mapped = jax.vmap(clear, in_axes=(None, 0))
    compiled_state, compiled_rings = jax.jit(mapped)(state, heads)
    batched_state, rings = mapped(state, heads)
    expected = [[[0], [20]], [[10], [0]]]
    for result in (rings, compiled_rings, batched_state.ring, compiled_state.ring):
        np.testing.assert_array_equal(result, expected)
    for result in (batched_state.head, compiled_state.head):
        np.testing.assert_array_equal(result, heads)
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(state)))
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    np.testing.assert_array_equal(state.head, 0)


@pytest.mark.parametrize('backend', BACKENDS)
def test_pure_delivery_scan_reuses_layout_without_advancing_owned_head(backend):
    state = make(backend)

    def step(ring, t):
        ring = csr_delay_deliver_layout(ring, jnp.array([True]), state, t)
        return ring, ring[t % 2]

    ring, arrivals = jax.jit(lambda r: jax.lax.scan(step, r, jnp.arange(3)))(state.ring)
    np.testing.assert_array_equal(ring, [[13], [26]])
    np.testing.assert_array_equal(arrivals, [[10], [23], [13]])
    np.testing.assert_array_equal(state.ring, [[10], [20]])
    assert int(state.head) == 0


def test_only_existing_delaycsr_layout_is_accepted():
    with pytest.raises(TypeError, match='delaycsr'):
        csr_delay_deliver_layout(jnp.zeros((2, 1)), jnp.array([True]), object(), 0)


@pytest.mark.parametrize('backend', ['cuda', 'cusparse', 'numba', 'invalid'])
def test_deliver_rejects_unsupported_backend_without_changing_state(backend):
    state = make()
    before = jax.tree.leaves(state)
    with pytest.raises(ValueError, match='backend'):
        state.csr_delay_deliver_layout(state.ring, jnp.array([True]), 0, backend=backend)
    assert all(old is new for old, new in zip(before, jax.tree.leaves(state)))


@pytest.mark.parametrize('owned_backend', [None, 'jax_raw', 'cuda_raw'])
@pytest.mark.parametrize('override', ['jax_raw', pytest.param('cuda_raw', marks=GPU)])
def test_backend_override_uses_a_pure_copy_of_the_existing_layout(owned_backend, override):
    state = make(owned_backend)
    before = jax.tree.leaves(state)
    result = jax.jit(lambda ring: csr_delay_deliver_layout(
        ring, jnp.array([True]), state, 0, backend=override))(state.ring)
    np.testing.assert_array_equal(result, [[10], [23]])
    assert state.backend == owned_backend
    assert all(old is new for old, new in zip(before, jax.tree.leaves(state)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_csr_delay_layout_preserves_source_and_inherits_backend(backend):
    source = _source(backend)
    original = tuple(np.asarray(x).copy() for x in (source.data, source.indices, source.indptr))
    layout = source.delay_layout(np.array([2, 0, 1, 2], np.int32), n_slots=4)
    assert isinstance(layout, brainevent.delaycsr)
    assert isinstance(layout, brainevent.CSR)
    assert layout.backend == source.backend == backend
    np.testing.assert_array_equal(layout.data, [3, 5, 2, 7])
    np.testing.assert_array_equal(layout.indices, [0, 1, 2, 0])
    for old, current in zip(original, (source.data, source.indices, source.indptr)):
        np.testing.assert_array_equal(current, old)

    events = jnp.array([2, -1], jnp.float32)
    ring = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    expected = np.asarray(ring).copy()
    for row, col, weight, delay in zip([0, 0, 0, 1], original[1], original[0], [2, 0, 1, 2]):
        expected[(3 + delay) % 4, col] += weight * np.asarray(events)[row]

    def deliver(weights, ring, events):
        updated = layout.with_data(weights)
        return brainevent.csr_delay_deliver_layout(
            ring, brainevent.WeightedArray(events), updated, 3,
        )

    np.testing.assert_array_equal(jax.jit(deliver)(layout.data, ring, events), expected)
    gradients = jax.jit(jax.grad(lambda w, r, x: deliver(w, r, x).sum(),
                                 argnums=(0, 1, 2)))(layout.data, ring, events)
    np.testing.assert_array_equal(gradients[0], [2, 2, 2, -1])
    np.testing.assert_array_equal(gradients[1], np.ones((4, 3), np.float32))
    np.testing.assert_array_equal(gradients[2], [10, 7])
    np.testing.assert_array_equal(layout.ring, np.zeros((4, 3), np.float32))
    assert int(layout.head) == 0
    assert layout.backend == backend


@pytest.mark.parametrize("backend", BACKENDS)
def test_csr_delay_layout_state_carries_through_jitted_scan(backend):
    source = brainevent.CSR(
        (np.array([3], np.float32), np.array([0], np.int32), np.array([0, 1], np.int32)),
        shape=(1, 1), backend=backend,
    )
    layout = source.delay_layout(np.array([1], np.int32), n_slots=2)
    layout.ring = jnp.array([[10], [20]], jnp.float32)

    def step(state, inputs):
        t, event = inputs
        arrival = state.forward(event, t)
        return state, arrival

    state, arrivals = jax.jit(lambda state: jax.lax.scan(
        step, state, (jnp.arange(3, dtype=jnp.int32), jnp.array([[True], [True], [False]])),
    ))(layout)
    np.testing.assert_array_equal(arrivals, [[23], [3], [0]])
    np.testing.assert_array_equal(state.ring, [[0], [0]])
    np.testing.assert_array_equal(layout.ring, [[10], [20]])
    assert state.backend == layout.backend == backend


@pytest.mark.parametrize("backend", ["jax_raw", pytest.param("cuda_raw", marks=GPU)])
@pytest.mark.parametrize("corruption", ["head", "metadata"])
def test_sharded_delivery_keeps_runtime_validation_and_distinct_rings(backend, corruption):
    # A size-one mesh still tracks manual varying axes; on multiple devices,
    # distinct rings also check that callback handling never replicates data.
    count = min(2, jax.device_count())
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:count]), ("delay_shard",))
    partition = jax.sharding.PartitionSpec("delay_shard")

    def local(rings, heads, counts):
        queue = jnp.zeros(0, jnp.int32)
        result = brainevent.csr_delay_deliver_layout_p.call(
            rings[0], jnp.array([True]), jnp.array([3], jnp.float32),
            jnp.array([0], jnp.int32), jnp.array([0, 1], jnp.int32), counts[0],
            jnp.array([1], jnp.int32), heads[0], jnp.zeros(1, jnp.int32),
            queue, queue, queue, jnp.zeros(2, jnp.int32), backend=backend,
        )
        return result[0][None]

    deliver = jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(partition,) * 3,
                                   out_specs=partition))
    rings = jnp.arange(count * 2, dtype=jnp.float32).reshape(count, 2, 1)
    heads = jnp.zeros((count, 1), jnp.int32)
    counts = jnp.ones((count, 1, 1), jnp.int32)
    expected = np.asarray(rings).copy()
    expected[:, 1, 0] += 3
    np.testing.assert_array_equal(deliver(rings, heads, counts), expected)
    bad_heads = heads.at[-1, 0].set(2) if corruption == "head" else heads
    bad_counts = counts.at[-1, 0, 0].set(0) if corruption == "metadata" else counts
    with pytest.raises(Exception, match=corruption):
        jax.block_until_ready(deliver(rings, bad_heads, bad_counts))
    np.testing.assert_array_equal(deliver(rings, heads, counts), expected)
    jax.jit(lambda: jax.debug.callback(lambda: None, ordered=False))()
    jax.effects_barrier()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.int8, jnp.int32, jnp.float16,
                                         jnp.bfloat16, jnp.float32, jnp.float64])
def test_public_delivery_accepts_main_event_dtypes(backend, weighted, event_dtype):
    layout = make(backend)
    before = jax.tree.leaves(layout)
    events = jnp.array([True] if event_dtype == jnp.bool_ else [-2], event_dtype)
    result = jax.jit(lambda x: csr_delay_deliver_layout(
        layout.ring, x, layout, 0, weighted=weighted))(events)
    contribution = -6 if weighted and event_dtype != jnp.bool_ else 3
    np.testing.assert_array_equal(result, [[10], [20 + contribution]])
    assert result.dtype == jnp.float32
    assert all(old is new for old, new in zip(before, jax.tree.leaves(layout)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weight_dtype,ring_dtype", [(jnp.float16, jnp.float32),
                                                     (jnp.float64, jnp.float32),
                                                     (jnp.float32, jnp.float64)])
def test_public_ring_dtype_controls_delivery_and_gradient(backend, weight_dtype, ring_dtype):
    source = brainevent.CSR((jnp.array([3.], weight_dtype), jnp.array([0], jnp.int32),
                            jnp.array([0, 1], jnp.int32)), shape=(1, 1), backend=backend)
    layout = source.delay_layout(np.array([1], np.int32), 3)
    before = jax.tree.leaves(layout)
    ring = jnp.array([[1], [2], [3]], ring_dtype)
    events = jnp.array([-2.], jnp.float64)

    def project(weights, event):
        return csr_delay_deliver_layout(ring, event, layout.with_data(weights), 0, weighted=True)

    result = jax.jit(project)(layout.data, events)
    np.testing.assert_array_equal(result, [[1], [-4], [3]])
    assert result.dtype == ring_dtype
    dw, dx = jax.jit(jax.grad(lambda w, x: project(w, x).sum(), argnums=(0, 1)))(layout.data, events)
    np.testing.assert_array_equal(dw, [-2])
    np.testing.assert_array_equal(dx, [3])
    assert dw.dtype == weight_dtype and dx.dtype == events.dtype
    assert all(old is new for old, new in zip(before, jax.tree.leaves(layout)))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("event_unit", [u.UNITLESS, u.mV])
def test_public_quantity_delivery_normalizes_and_restores_units(backend, event_unit):
    source = brainevent.CSR((jnp.array([2.], jnp.float16) * u.mS,
                            jnp.array([0], jnp.int32), jnp.array([0, 1], jnp.int32)),
                           shape=(1, 1), backend=backend)
    layout = source.delay_layout(np.array([1], np.int32), 3)
    unit = u.mS * event_unit
    ring = jnp.ones((3, 1), jnp.float32) * unit
    events = jnp.array([-2.], jnp.float64) * event_unit
    result = jax.jit(lambda r, x: csr_delay_deliver_layout(r, x, layout, 0, weighted=True))(ring, events)
    assert u.get_unit(result) == unit
    assert result.dtype == jnp.float32
    np.testing.assert_array_equal(u.get_mantissa(result), [[1], [-3], [1]])
    assert u.get_unit(layout.data) == u.mS
    np.testing.assert_array_equal(u.get_mantissa(layout.ring), np.zeros((3, 1)))


def test_csr_delay_layout_isolated_from_reused_source_host_pointer():
    storage = np.empty(72, np.uint8)
    offset = (-storage.ctypes.data) % 64
    pointer = storage[offset:offset + 8].view(np.int32)
    pointer[:] = [0, 3]
    source = brainevent.CSR((np.array([2.], np.float32), np.array([1, 0, 2], np.int32), pointer),
                           shape=(1, 3), backend="jax_raw")
    layout = source.delay_layout(np.ones(3, np.int32), 2)
    pointer[:] = 7
    np.testing.assert_array_equal(layout.indptr, [0, 3])
    result = csr_delay_deliver_layout(jnp.zeros((2, 3), jnp.float32), jnp.array([True]), layout, 0)
    np.testing.assert_array_equal(result, [[0, 0, 0], [2, 2, 2]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_public_quantity_delivery_preserves_ring_scale_and_rejects_incompatible_units(backend):
    source = brainevent.CSR((jnp.array([2.], jnp.float32) * u.mV,
                            jnp.array([0], jnp.int32), jnp.array([0, 1], jnp.int32)),
                           shape=(1, 1), backend=backend)
    layout = source.delay_layout(np.array([1], np.int32), 3)
    events = jnp.array([2.], jnp.float32)
    result = csr_delay_deliver_layout(jnp.full((3, 1), .001, jnp.float32) * u.volt,
                                     events, layout, 0, weighted=True)
    assert u.get_unit(result) == u.volt
    assert result.dtype == jnp.float32
    np.testing.assert_allclose(u.get_mantissa(result), [[.001], [.005], [.001]], rtol=1e-6)
    with pytest.raises(u.UnitMismatchError):
        csr_delay_deliver_layout(jnp.zeros((3, 1), jnp.float32) * u.ms,
                                 events, layout, 0, weighted=True)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('weight_unit,event_unit,ring_unit', [
    pytest.param(u.mV, u.UNITLESS, u.volt, id='voltage'),
    pytest.param(u.mS, u.mV, u.mA, id='compound'),
])
def test_quantity_delivery_ring_scale_gradients(backend, weighted, weight_unit, event_unit, ring_unit):
    layout = delaycsr(
        (np.array([2], np.float64) * weight_unit, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        shape=(1, 1), edge_delays=np.ones(1, np.int32), buffer_slots=3, backend=backend,
    )
    leaves = jax.tree.leaves(layout)
    weights = u.get_mantissa(layout.data)
    ring = jnp.full((3, 1), .1, jnp.float32)
    events = jnp.array([2], jnp.float32)

    def deliver(w, r, x):
        return layout.with_data(w * weight_unit).csr_delay_deliver_layout(
            r * ring_unit, x * event_unit, 0, weighted=weighted)

    def values(w, r, x):
        return u.get_mantissa(deliver(w, r, x))

    _, (dw, dr, dx) = jax.jit(jax.value_and_grad(
        lambda w, r, x: values(w, r, x).sum(), argnums=(0, 1, 2)))(weights, ring, events)
    np.testing.assert_allclose(dw, [.002 if weighted else .001], rtol=1e-6)
    np.testing.assert_array_equal(dr, jnp.ones_like(ring))
    np.testing.assert_allclose(dx, [.002], rtol=1e-6)
    assert dw.dtype == weights.dtype and dx.dtype == events.dtype
    result, tangent = jax.jvp(values, (weights, ring, events),
                              (jnp.ones_like(weights), jnp.ones_like(ring), jnp.ones_like(events)))
    np.testing.assert_allclose(result, [[.1], [.104 if weighted else .102], [.1]], rtol=1e-6)
    np.testing.assert_allclose(tangent, [[1], [1.004 if weighted else 1.003], [1]], rtol=1e-6)
    quantity = deliver(weights, ring, events)
    assert u.get_unit(quantity) == ring_unit
    assert quantity.dtype == ring.dtype
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(layout)))
    np.testing.assert_array_equal(ring, jnp.full((3, 1), .1, jnp.float32))


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weighted', [False, True])
def test_quantity_delivery_ring_scale_keeps_inactive_half_ring_finite(backend, weighted):
    layout = delaycsr(
        (np.array([2], np.float16) * u.mV, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        shape=(1, 1), edge_delays=np.ones(1, np.int32), backend=backend,
    )
    ring = jnp.full((2, 1), 100, jnp.float16) * u.volt
    result = jax.jit(lambda r: layout.csr_delay_deliver_layout(
        r, jnp.zeros(1, jnp.float16), 0, weighted=weighted))(ring)
    # Converting the whole buffer to mV would overflow float16 before delivery.
    np.testing.assert_array_equal(u.get_mantissa(result), [[100], [100]])
    assert u.get_unit(result) == u.volt
    assert result.dtype == jnp.float16


@pytest.mark.parametrize('backend', BACKENDS[1:])
def test_quantity_delivery_ring_scale_keeps_binary_event_activity(backend):
    layout = delaycsr(
        (np.array([2], np.float16) * u.mV, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        shape=(1, 1), edge_delays=np.ones(1, np.int32), backend=backend,
    )
    tiny = jnp.array([1e-5], jnp.float16)
    assert bool(tiny[0] != 0)
    np.testing.assert_array_equal(tiny * jnp.float16(.001), 0)
    result = jax.jit(lambda events: layout.csr_delay_deliver_layout(
        jnp.zeros((2, 1), jnp.float16) * u.volt, events, 0))(tiny)
    assert u.get_unit(result) == u.volt
    np.testing.assert_allclose(u.get_mantissa(result), [[0], [.002]], rtol=1e-3)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('ring_unit', [u.mV / u.volt, u.UNITLESS], ids=['scaled', 'unitless'])
def test_scaled_dimensionless_quantity_ring_keeps_pytree_for_delivery_and_clear(backend, ring_unit):
    layout = make(backend)
    leaves = jax.tree.leaves(layout)
    ring = u.Quantity(jnp.ones((2, 1), jnp.float32), unit=ring_unit)
    result = jax.jit(lambda r: layout.csr_delay_deliver_layout(
        r, jnp.array([True]), 0))(ring)
    assert jax.tree.structure(result) == jax.tree.structure(ring)
    assert u.get_unit(result) == ring.unit
    contribution = float(u.Quantity(3, unit=u.UNITLESS).to_decimal(ring_unit))
    np.testing.assert_allclose(u.get_mantissa(result), [[1], [1 + contribution]], rtol=1e-6)

    def clear(r):
        local = copy(layout)
        local.ring = r
        return local.clear_head()

    cleared = jax.jit(clear)(ring)
    assert jax.tree.structure(cleared) == jax.tree.structure(ring)
    assert u.get_unit(cleared) == ring.unit
    np.testing.assert_array_equal(u.get_mantissa(cleared), [[0], [1]])
    assert all(old is new for old, new in zip(leaves, jax.tree.leaves(layout)))
    np.testing.assert_array_equal(u.get_mantissa(ring), 1)


@pytest.mark.parametrize('backend', BACKENDS[1:])
@pytest.mark.parametrize('weight,weight_unit,factor', [
    pytest.param(1e-5, u.mV, .001, id='tiny-mV-to-V'),
    pytest.param(.001, u.Mvolt, 1e6, id='small-MV-to-V'),
])
def test_quantity_ring_scale_converts_half_weights_at_ring_precision(backend, weight, weight_unit, factor):
    layout = delaycsr(
        (np.array([weight], np.float16) * weight_unit, np.array([0], np.int32),
         np.array([0, 1], np.int32)),
        shape=(1, 1), edge_delays=np.ones(1, np.int32), backend=backend,
    )
    ring = jnp.zeros((2, 1), jnp.float32) * u.volt
    result = jax.jit(lambda r: layout.csr_delay_deliver_layout(
        r, jnp.array([True]), 0))(ring)
    expected = float(np.float16(weight)) * factor
    np.testing.assert_allclose(u.get_mantissa(result), [[0], [expected]], rtol=1e-6)
    assert u.get_unit(result) == u.volt
    assert result.dtype == jnp.float32
