# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Contracts for delaycsr methods inherited from the static CSR API."""

import operator
from copy import copy

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSR, PerformanceWarning, WeightedArray, delaycsr
from brainevent._csr.config import CSRConfig
from brainevent._csr.delay import region_delay
from brainevent._csr.delay.test._helpers import BACKENDS
from brainevent._csr.delay.test._reference import numpy_project_step


def _matrix(*, shared=False, index_dtype=np.int32, ptr_dtype=np.int32,
            unit=None, ring_dtype=np.float32, ring_unit=None, backend="jax_raw"):
    weights = np.array([2.] if shared else [2., 3., 5., 7.], np.float32)
    if unit is not None:
        weights = weights * unit
    ring = jnp.arange(12, dtype=ring_dtype).reshape(4, 3)
    if ring_unit is not None:
        ring = ring * ring_unit
    elif unit is not None:
        ring = ring * unit
    return delaycsr(
        (weights, np.array([0, 2, 1, 0], index_dtype),
         np.array([0, 2, 4], ptr_dtype)),
        edge_delays=np.array([2, 0, 1, 2], np.int32), shape=(2, 3),
        n_slots=4, ring=ring, head=3, weighted=True, backend=backend,
    )


def _values(value):
    return np.asarray(u.get_mantissa(value))


def _same_state(result, source):
    assert isinstance(result, delaycsr)
    assert result is not source
    for name in ("indices", "indptr", "delay_count", "delay_off", "row_task_count",
                 "ring", "head", "workspace", "config", "_cache"):
        assert getattr(result, name) is getattr(source, name), name
    assert result.delay_steps == source.delay_steps
    assert result.L == source.L
    assert result.weighted == source.weighted


def _snapshot(source):
    snapshot = {name: np.array(_values(getattr(source, name)), copy=True)
                for name in ("data", "indices", "indptr", "delay_count", "delay_off",
                             "row_task_count", "ring", "head")}
    snapshot.update({f"workspace:{i}": np.array(array, copy=True)
                     for i, array in enumerate(source.workspace)})
    return snapshot


def _unchanged(source, snapshot):
    for name, expected in snapshot.items():
        actual = (source.workspace[int(name.split(":")[1])] if name.startswith("workspace:")
                  else getattr(source, name))
        np.testing.assert_array_equal(_values(actual), expected,
                                      err_msg=f"source {name} changed")


@pytest.mark.parametrize("method,args", [
    ("__matmul__", (jnp.ones(3, jnp.float32),)),
    ("__getitem__", (0,)),
    ("prepare_diag_add", ()),
])
def test_unsupported_entry_points_do_not_enter_parent(method, args, monkeypatch):
    source = _matrix()
    snapshot = _snapshot(source)

    def forbidden(*args, **kwargs):
        pytest.fail(f"parent {method} was called")

    monkeypatch.setattr(CSR, method, forbidden)
    with pytest.raises(NotImplementedError):
        getattr(source, method)(*args)
    _unchanged(source, snapshot)


def test_build_weight_indices_warns_and_exports_static_csr():
    source = _matrix()
    snapshot = _snapshot(source)
    with pytest.warns(UserWarning, match="(?i)delay") as warnings:
        result = source.build_weight_indices()
    assert warnings[0].filename == __file__
    assert type(result) is CSR
    for name in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(_values(getattr(result, name)),
                                      _values(getattr(source, name)))
    assert not hasattr(result, "delay_count")
    assert result._weight_indices() is not None
    _unchanged(source, snapshot)


@pytest.mark.parametrize("transform", [
    lambda data: data.astype(jnp.float64),
    lambda data: data * u.ms,
])
def test_apply_rejects_dtype_and_unit_changes_early(transform):
    source = _matrix()
    snapshot = _snapshot(source)
    with pytest.raises(TypeError):
        source.apply(transform)
    _unchanged(source, snapshot)


def test_sparse_arithmetic_rejects_different_delay_layout():
    source = _matrix()
    other = copy(source)
    other.delay_steps = (0, 1, 3)
    other.delay_off = jnp.asarray([0, 3, 9], jnp.int32)
    with pytest.raises(ValueError, match="(?i)delay|layout"):
        source * other


_INHERITED_I32_CALLS = [
    ("update_on_pre", lambda w: w.update_on_pre(jnp.array([True, False]), jnp.ones(3, jnp.float32))),
    ("update_on_post", lambda w: w.update_on_post(jnp.ones(2, jnp.float32), jnp.array([True, False, True]))),
    ("apply_on_pre", lambda w: w.apply_on_pre(lambda e, pre, post, params: e, jnp.array([True, False]))),
    ("apply_on_post", lambda w: w.apply_on_post(lambda e, pre, post, params: e, jnp.array([True, False, True]))),
    ("dt2t", lambda w: w.dt2t(jnp.ones(2, jnp.float32), jnp.ones(4, jnp.float32))),
    ("dt2t_transposed", lambda w: w.dt2t_transposed(jnp.ones(3, jnp.float32), jnp.ones(4, jnp.float32))),
]


@pytest.mark.parametrize("method,invoke", _INHERITED_I32_CALLS)
def test_int64_indices_rejected_before_parent(method, invoke, monkeypatch):
    source = _matrix(index_dtype=np.int64)
    snapshot = _snapshot(source)

    def forbidden(*args, **kwargs):
        pytest.fail(f"parent {method} was called before rejecting int64 indices")

    monkeypatch.setattr(CSR, method, forbidden)
    with pytest.raises(TypeError, match="int32.*indices|indices.*int32"):
        invoke(source)
    _unchanged(source, snapshot)


def test_fromdense_uses_template_edge_order_and_retains_state():
    source = _matrix()
    dense = jnp.array([[11., 0., 13.], [17., 19., 0.]], jnp.float32)
    with pytest.warns(UserWarning, match="(?i)delay") as warnings:
        result = delaycsr.fromdense(dense, template=source)
    assert warnings[0].filename == __file__
    np.testing.assert_array_equal(result.data, [13., 11., 19., 17.])
    _same_state(result, source)


def test_slice_rows_returns_delay_layout_in_requested_order():
    source = _matrix()
    result = source.slice_rows([1, 0, 1])
    assert isinstance(result, delaycsr)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result.indptr, [0, 2, 4, 6])
    np.testing.assert_array_equal(result.indices, [1, 0, 2, 0, 1, 0])
    np.testing.assert_array_equal(result.data, [5., 7., 3., 2., 5., 7.])
    np.testing.assert_array_equal(result.delay_count, _values(source.delay_count)[[1, 0, 1]])
    np.testing.assert_array_equal(result.ring, np.zeros((4, 3)))
    assert int(result.head) == 0


def test_diag_add_requires_explicit_delays():
    source = _matrix()
    with pytest.raises(TypeError, match="diag_delays"):
        source.diag_add(jnp.ones(2, jnp.float32))


@pytest.mark.parametrize("index", [slice(None), -1, [0, 1], jnp.array([1, 0], jnp.int32)])
def test_getitem_has_one_explicit_unsupported_contract(index):
    source = _matrix()
    with pytest.raises(NotImplementedError):
        source[index]


@pytest.mark.parametrize("operand", [
    jnp.ones(3, jnp.float32), BinaryArray(jnp.ones(3, jnp.float32)),
    WeightedArray(jnp.ones(3, jnp.float32)),
])
def test_matmul_rejects_array_and_event_wrappers(operand):
    source = _matrix()
    snapshot = _snapshot(source)
    with pytest.raises(NotImplementedError):
        source @ operand
    _unchanged(source, snapshot)


def test_apply_preserves_delay_state_and_requires_original_storage_shape():
    source = _matrix(unit=u.mV, ring_unit=u.mV * u.second)
    snapshot = _snapshot(source)
    result = source.apply(lambda data: data * jnp.float32(2))
    np.testing.assert_array_equal(_values(result.data), _values(source.data) * 2)
    assert u.get_unit(result.data) == u.mV
    _same_state(result, source)
    _unchanged(source, snapshot)
    with pytest.raises(ValueError):
        _matrix(shared=True).apply(lambda data: jnp.broadcast_to(data, (4,)))
    with pytest.raises(TypeError):
        source.apply(lambda data: data.to(u.volt))


@pytest.mark.parametrize("reverse", [False, True])
def test_apply2_sparse_same_layout_uses_receiver_state(reverse):
    source = _matrix()
    other = source.with_data(source.data * jnp.float32(2))
    other.ring = jnp.full_like(source.ring, 99)
    other.head = 0
    other.workspace = other._make_workspace(other.ring.dtype, other.weighted)
    result = source.apply2(other, operator.truediv, reverse=reverse)
    expected = 2. if reverse else .5
    np.testing.assert_array_equal(result.data, np.full(4, expected, np.float32))
    _same_state(result, source)


@pytest.mark.parametrize("op,reverse", [
    (operator.add, False), (operator.sub, False),
    (operator.add, True), (operator.sub, True),
])
def test_apply2_sparse_add_sub_return_static_dense(op, reverse):
    source = _matrix()
    other = source.with_data(source.data * jnp.float32(2))
    result = source.apply2(other, op, reverse=reverse)
    expected = (op(other.todense(), source.todense()) if reverse
                else op(source.todense(), other.todense()))
    assert not isinstance(result, delaycsr)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("internal,reverse", [(False, False), (False, True), (True, False), (True, True)])
def test_sparse_operands_must_share_delay_layout(internal, reverse):
    source = _matrix()
    independent = _matrix()
    plain = CSR._from_parts(source.data, source.indices, source.indptr, shape=source.shape)
    for other in (independent, plain):
        with pytest.raises((TypeError, ValueError)):
            if internal:
                method = source._binary_rop if reverse else source._binary_op
                method(other, operator.mul)
            else:
                source.apply2(other, operator.mul, reverse=reverse)


@pytest.mark.parametrize("field,value", [("L", 5), ("weighted", False), ("delay_steps", (0, 1, 3))])
def test_shared_indices_do_not_prove_static_delay_properties_equal(field, value):
    source = _matrix()
    other = copy(source)
    setattr(other, field, value)
    with pytest.raises(ValueError):
        source * other


def test_apply2_shared_weights_expand_and_reject_bad_result_dtype_and_unit():
    source = _matrix(shared=True)
    dense = jnp.array([[2., 0., 3.], [5., 7., 0.]], jnp.float32)
    result = source * dense
    np.testing.assert_array_equal(result.data, [6., 4., 14., 10.])
    _same_state(result, source)
    for value in (jnp.float64(2), jnp.float32(2) * u.ms):
        with pytest.raises(TypeError):
            source * value
        with pytest.raises(TypeError):
            source.apply2(value, operator.truediv, reverse=True)
    with pytest.raises(ValueError):
        source._same_structure(jnp.ones(3, jnp.float32))


def test_scalar_and_dense_apply2_remain_jittable_and_differentiable():
    source = _matrix()

    @jax.jit
    def transformed(data, scale):
        return (source.with_data(data) * scale).data

    np.testing.assert_array_equal(transformed(source.data, jnp.float32(2)), source.data * 2)
    np.testing.assert_array_equal(jax.grad(lambda data: jnp.sum(transformed(data, jnp.float32(2))))(source.data),
                                  np.full(4, 2., np.float32))
    same = source.with_data(source.data * jnp.float32(2))
    result = jax.jit(lambda data: (source.with_data(data) * same).data)(source.data)
    np.testing.assert_array_equal(result, source.data * same.data)
    independent = _matrix()
    with pytest.raises(ValueError):
        jax.jit(lambda data: (source.with_data(data) * independent).data)(source.data)


@pytest.mark.parametrize("method,invoke", _INHERITED_I32_CALLS)
def test_int32_indices_delegate_and_do_not_reject_int64_indptr(method, invoke, monkeypatch):
    source = _matrix(ptr_dtype=np.int64)
    sentinel = object()
    seen = []

    def parent(instance, *args, **kwargs):
        seen.append(instance)
        assert instance.indices.dtype == jnp.int32
        assert instance.indptr.dtype == jnp.int64
        return sentinel

    monkeypatch.setattr(CSR, method, parent)
    assert invoke(source) is sentinel
    assert seen == [source]


@pytest.mark.parametrize("alias,y_size", [("yw_to_w", 2), ("yw_to_w_transposed", 3)])
def test_deprecated_aliases_use_int64_indices_guard(alias, y_size):
    source = _matrix(index_dtype=np.int64)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(TypeError, match="int32.*indices|indices.*int32"):
            getattr(source, alias)(jnp.ones(y_size, jnp.float32), jnp.ones(4, jnp.float32))


@pytest.mark.parametrize("backend", BACKENDS)
def test_int64_indices_native_delivery_stays_supported(backend):
    source = _matrix(index_dtype=np.int64, backend=backend)
    result = source.deliver(jnp.zeros((4, 3), jnp.float32),
                                            jnp.array([1., 2.], jnp.float32), 0, weighted=True)
    expected = np.array([[0., 0., 3.], [0., 10., 0.], [16., 0., 0.], [0., 0., 0.]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_int32_plasticity_and_dt2t_keep_parent_numerics_and_return_types(backend):
    source = _matrix(ptr_dtype=np.int64, backend=backend)
    snapshot = _snapshot(source)
    initial_config = source.config
    assert initial_config.mirror is None
    assert source._cache.mirror is None
    pre = jnp.array([True, False])
    post = jnp.array([True, False, True])
    updated_pre = source.update_on_pre(pre, jnp.array([1., 2., 3.], jnp.float32))
    np.testing.assert_array_equal(updated_pre.data, [6., 3., 5., 7.])
    _same_state(updated_pre, source)
    assert source.config is initial_config
    assert source._cache.mirror is None

    # Results share the config snapshot at creation. The post update then
    # materializes the mirror and binds a new snapshot to the source.
    updated_post = source.update_on_post(
        jnp.array([10., 20.], jnp.float32), post,
    )
    np.testing.assert_array_equal(updated_post.data, [13., 12., 5., 27.])
    _same_state(updated_post, source)
    assert updated_pre.config is initial_config
    assert initial_config.mirror is None
    assert initial_config.mirror_max_row_nnz is None
    assert source.config is not initial_config
    assert source.config.primary is initial_config.primary
    mirror = source._cache.mirror
    assert mirror is not None
    plan, maximum = source._cache.mirror_preparation
    assert source.config.mirror is plan
    assert source.config.mirror_max_row_nnz == maximum
    assert updated_pre._mirror() is mirror
    assert updated_pre.config.mirror is plan
    assert updated_pre.config.mirror_max_row_nnz == maximum
    assert initial_config.mirror is None
    rule = lambda edge, pre_value, post_value, params: edge * jnp.float32(2)
    edge_pre = source.apply_on_pre(rule, pre)
    edge_post = source.apply_on_post(rule, post)
    assert not isinstance(edge_pre, CSR)
    assert not isinstance(edge_post, CSR)
    np.testing.assert_array_equal(edge_pre, [6., 4., 5., 7.])
    np.testing.assert_array_equal(edge_post, [6., 4., 5., 14.])
    np.testing.assert_array_equal(source.dt2t(jnp.array([2., 3.], jnp.float32), source.data),
                                  [6., 4., 15., 21.])
    np.testing.assert_array_equal(source.dt2t_transposed(jnp.array([2., 3., 5.], jnp.float32), source.data),
                                  [15., 4., 15., 14.])
    _unchanged(source, snapshot)


def test_fromdense_zero_weights_are_edges_and_shared_storage_expands():
    source = _matrix(shared=True)
    snapshot = _snapshot(source)
    dense = jnp.array([[0., 0., 13.], [17., 0., 0.]], jnp.float32)
    with pytest.warns(UserWarning, match="(?i)delay"):
        result = delaycsr.fromdense(dense, template=source)
    np.testing.assert_array_equal(result.data, [13., 0., 0., 17.])
    assert result.nse == 4
    assert result.data.shape == (4,)
    _same_state(result, source)
    _unchanged(source, snapshot)


@pytest.mark.parametrize("dense", [
    np.array([[0., 11., 13.], [17., 19., 0.]], np.float32),
    np.ones((3, 3), np.float32),
    np.zeros((2, 3), np.float64),
])
def test_fromdense_rejects_new_connections_shape_and_dtype(dense):
    source = _matrix()
    snapshot = _snapshot(source)
    with pytest.raises((TypeError, ValueError)):
        delaycsr.fromdense(dense, template=source)
    _unchanged(source, snapshot)


def test_fromdense_requires_initialized_template_and_rejects_duplicates():
    dense = jnp.zeros((2, 3), jnp.float32)
    with pytest.raises(TypeError):
        delaycsr.fromdense(dense)
    with pytest.raises((TypeError, ValueError)):
        delaycsr.fromdense(dense, template=CSR.fromdense(dense))
    with pytest.raises((TypeError, ValueError)):
        delaycsr.fromdense(dense, template=object.__new__(delaycsr))
    duplicate = delaycsr(
        (np.array([2., 3.], np.float32), np.array([0, 0], np.int32), np.array([0, 2], np.int32)),
        shape=(1, 2), edge_delays=[0, 2], backend="jax_raw")
    with pytest.raises(ValueError, match="(?i)duplicate"):
        delaycsr.fromdense(jnp.array([[5., 0.]], jnp.float32), template=duplicate)


def test_fromdense_preserves_weight_units_and_independent_ring_units():
    source = _matrix(unit=u.mV, ring_unit=u.mV * u.second)
    dense = jnp.array([[11., 0., 13.], [17., 19., 0.]], jnp.float32)
    with pytest.warns(UserWarning, match="(?i)delay"):
        result = delaycsr.fromdense(dense * u.mV, template=source)
    _same_state(result, source)
    assert u.get_unit(result.data) == u.mV
    np.testing.assert_array_equal(_values(result.data), [13., 11., 19., 17.])
    for wrong in (dense, dense * u.volt, dense * u.ms):
        with pytest.raises(TypeError):
            delaycsr.fromdense(wrong, template=source)


def test_fromdense_rejects_traced_dense_content():
    source = _matrix()
    with pytest.raises((RuntimeError, TypeError), match="(?i)jit|trac|concrete|eager"):
        jax.jit(lambda dense: delaycsr.fromdense(dense, template=source).data)(
            jnp.zeros(source.shape, jnp.float32))


def test_fromdense_empty_template_returns_per_edge_empty_storage():
    source = delaycsr(
        (np.array([2.], np.float32), np.array([], np.int32), np.array([0, 0, 0], np.int32)),
        shape=(2, 3), edge_delays=0, n_slots=4, backend="jax_raw")
    with pytest.warns(UserWarning, match="(?i)delay"):
        result = delaycsr.fromdense(jnp.zeros(source.shape, jnp.float32), template=source)
    assert result.data.shape == (0,)
    _same_state(result, source)


def _fresh_state(result, source):
    assert isinstance(result, delaycsr)
    assert result is not source
    assert result.ring is not source.ring
    assert result.ring.dtype == source.ring.dtype
    assert u.get_unit(result.ring) == u.get_unit(source.ring)
    np.testing.assert_array_equal(_values(result.ring), np.zeros((result.L, result.shape[1])))
    assert int(result.head) == 0
    assert result.workspace is not source.workspace
    assert type(result.workspace) is type(source.workspace)
    for array in result.workspace:
        np.testing.assert_array_equal(array, np.zeros(array.shape))
    assert result._cache is not source._cache
    assert result._cache.mirror is None
    assert result._cache.diag_plan is None
    assert result.config is not source.config
    assert result.config.shape == result.shape
    assert result.config.nnz == result.nse
    assert result.backend == source.backend
    assert result.weighted == source.weighted
    assert result.backward_algorithm == source.backward_algorithm


@pytest.mark.parametrize("index,rows", [
    (-1, [1]), (slice(None, None, -1), [1, 0]),
    ([-1, 0, -1], [1, 0, 1]), ([], []),
    (np.array([], np.int64), []),
])
def test_slice_rows_negative_repeated_and_empty_selection(index, rows):
    source = _matrix(shared=True, ring_dtype=np.float16, unit=u.mV,
                     ring_unit=u.mV * u.second)
    source.backward_algorithm = "weighted_float"
    snapshot = _snapshot(source)
    result = source.slice_rows(index)
    assert result.shape == (len(rows), 3)
    assert result.data.shape == (1,)
    assert u.get_unit(result.data) == u.mV
    np.testing.assert_array_equal(result.delay_count, _values(source.delay_count)[rows])
    np.testing.assert_array_equal(result.row_task_count, _values(source.row_task_count)[rows])
    assert result.delay_steps == source.delay_steps
    np.testing.assert_array_equal(result.delay_off, source.delay_off)
    assert result.L == source.L
    _fresh_state(result, source)
    _unchanged(source, snapshot)


@pytest.mark.parametrize("index", [2, -3, [2**32], np.array([2**32], np.int64)])
def test_slice_rows_rejects_out_of_range_before_integer_narrowing(index):
    with pytest.raises(IndexError):
        _matrix().slice_rows(index)


def test_slice_rows_clears_caches_and_prepares_config_from_blank_source():
    source = _matrix()
    source._mirror()
    source._cache.diag_plan = (jnp.zeros(4, jnp.int32),)
    source.config = CSRConfig()
    source.backward_algorithm = "weighted_float"
    result = source.slice_rows([1, 0])
    _fresh_state(result, source)
    assert result.config.primary is not None


def test_slice_rows_long_row_task_capacity_counts_repetitions():
    sizes = np.array([130, 3, 129], np.int32)
    source = delaycsr(
        (np.ones(int(sizes.sum()), np.float32), np.arange(int(sizes.sum()), dtype=np.int32) % 5,
         np.array([0, 130, 133, 262], np.int32)),
        shape=(3, 5), edge_delays=np.r_[np.arange(130) % 2 * 2, [0, 1, 2], np.full(129, 2)],
        n_slots=4, weighted=True, backend="jax_raw")
    np.testing.assert_array_equal(source.row_task_count, [2, 0, 1])
    source.workspace = type(source.workspace)(*(jnp.ones_like(a) for a in source.workspace))
    snapshot = _snapshot(source)
    result = source.slice_rows([2, 0, 2])
    assert result.capacity == 4
    np.testing.assert_array_equal(result.row_task_count, [1, 2, 1])
    assert result.workspace.task_begin.shape == (4,)
    assert result.workspace.task_value.shape == (4,)
    _fresh_state(result, source)
    _unchanged(source, snapshot)


def test_slice_rows_checks_task_capacity_before_allocating(monkeypatch):
    source = delaycsr(
        (np.ones(130, np.float32), np.arange(130, dtype=np.int32) % 3,
         np.array([0, 130], np.int32)),
        shape=(1, 3), edge_delays=np.arange(130) % 2, backend="jax_raw")
    monkeypatch.setattr(region_delay, "_I32_MAX", 3)
    with pytest.raises(ValueError, match="(?i)capacity|task"):
        source.slice_rows([0, 0])


def test_slice_rows_widens_indptr_before_cumsum_and_requires_x64(monkeypatch):
    source = delaycsr(
        (np.arange(1, 5, dtype=np.float32), np.zeros(4, np.int32), np.array([0, 2, 4], np.int32)),
        shape=(2, 1), edge_delays=[0, 1, 0, 1], n_slots=2, backend="jax_raw")
    monkeypatch.setattr(region_delay, "_I32_MAX", 4)
    result = source.slice_rows([0, 1, 0])
    assert result.indices.dtype == jnp.int32
    assert result.indptr.dtype == jnp.int64
    np.testing.assert_array_equal(result.indptr, [0, 2, 4, 6])
    with jax.enable_x64(False):
        with pytest.raises((RuntimeError, ValueError), match="(?i)x64|int64"):
            source.slice_rows([0, 1, 0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_slice_rows_arrivals_follow_the_selected_original_rows(backend):
    source = _matrix(backend=backend)
    snapshot = _snapshot(source)
    result = source.slice_rows([1, 0, 1])
    events = np.array([1., 2., 3.], np.float32)
    expected = numpy_project_step(
        np.array([5., 7., 2., 3., 5., 7.], np.float32),
        np.array([1, 0, 0, 2, 1, 0], np.int32), np.array([0, 2, 4, 6], np.int32),
        np.array([1, 2, 2, 0, 1, 2], np.int32), events, 0,
        np.zeros((4, 3)), weighted=True)
    actual = result.deliver(result.ring, events, 0, weighted=True)
    np.testing.assert_array_equal(actual, expected)
    _unchanged(source, snapshot)


def test_slice_rows_is_eager_when_row_selection_is_traced():
    source = _matrix()
    with pytest.raises((RuntimeError, TypeError), match="(?i)jit|trac|concrete|eager"):
        jax.jit(lambda rows: source.slice_rows(rows).data)(jnp.array([0], jnp.int32))


def test_diag_add_rebuilds_via_init_preserves_edges_and_resets_state(monkeypatch):
    source = _matrix(ring_dtype=np.float16)
    source.backward_algorithm = "weighted_float"
    source._mirror()
    snapshot = _snapshot(source)
    calls = []
    original_init = delaycsr.__init__

    def tracked_init(self, *args, **kwargs):
        calls.append(kwargs.copy())
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(delaycsr, "__init__", tracked_init)
    with pytest.warns(PerformanceWarning, match="(?i)rebuild|init") as warnings:
        result = source.diag_add([11., 13.], diag_delays=[4, 1])
    assert warnings[0].filename == __file__
    assert len(calls) == 1
    assert calls[0].get("sorted_by_delay", False) is False
    assert result.shape == source.shape
    assert result.nse == source.nse + 2
    assert result.L == 5
    assert result.delay_steps == (0, 1, 2, 4)
    np.testing.assert_array_equal(result.delay_count, [[1, 0, 1, 1], [0, 2, 1, 0]])
    np.testing.assert_array_equal(result.todense(), [[13., 0., 3.], [7., 18., 0.]])
    _fresh_state(result, source)
    _unchanged(source, snapshot)


@pytest.mark.parametrize("backend", BACKENDS)
def test_diag_add_increments_arrive_separately_at_their_own_delay(backend):
    source = _matrix(backend=backend)
    with pytest.warns(PerformanceWarning):
        result = source.diag_add(jnp.array([11., 13.], jnp.float32), diag_delays=[4, 1])
    actual = result.deliver(result.ring, jnp.array([1., 2.], jnp.float32),
                                            0, weighted=True)
    np.testing.assert_array_equal(actual, [[0., 0., 3.], [0., 36., 0.], [16., 0., 0.],
                                         [0., 0., 0.], [11., 0., 0.]])


def test_diag_add_retains_shared_weights_and_zero_increments_as_separate_edges():
    source = _matrix(shared=True)
    with pytest.warns(PerformanceWarning):
        result = source.diag_add(jnp.array([0., 0.], jnp.float32), diag_delays=2)
    assert result.nse == 6
    assert result.data.shape == (6,)
    np.testing.assert_array_equal(result.todense(), source.todense())
    assert np.count_nonzero(np.asarray(result.data) == 0) == 2
    np.testing.assert_array_equal(result.delay_count.sum(axis=1), [3, 3])
    _fresh_state(result, source)


@pytest.mark.parametrize("bad", [True, 1.5, -1, [1], [[1, 2]], [0, -1],
                                  np.array([2**63, 1], np.uint64),
                                  np.array([2**64 - 1, 1], np.uint64), np.iinfo(np.int32).max])
def test_diag_add_rejects_invalid_delay_values(bad):
    source = _matrix()
    snapshot = _snapshot(source)
    with pytest.raises((TypeError, ValueError)):
        source.diag_add(jnp.ones(2, jnp.float32), diag_delays=bad)
    _unchanged(source, snapshot)


@pytest.mark.parametrize("bad", [1., jnp.array(1., jnp.float32), jnp.ones(1, jnp.float32),
                                  jnp.eye(2, dtype=jnp.float32)])
def test_diag_add_requires_a_diagonal_vector(bad):
    with pytest.raises((TypeError, ValueError)):
        _matrix().diag_add(bad, diag_delays=0)


def test_diag_add_rejects_sparse_increment_and_incompatible_units():
    source = _matrix(unit=u.mV)
    with pytest.raises((TypeError, ValueError)):
        source.diag_add(CSR.fromdense(jnp.eye(2, dtype=jnp.float32)), diag_delays=0)
    with pytest.raises((TypeError, ValueError, u.UnitMismatchError)):
        source.diag_add(jnp.ones(2, jnp.float32) * u.ms, diag_delays=0)


def test_diag_add_converts_units_promotes_weights_and_preserves_ring_precision():
    source = _matrix(unit=u.mV, ring_dtype=np.float16, ring_unit=u.mV * u.second)
    with pytest.warns(PerformanceWarning):
        result = source.diag_add(jnp.array([.001, .002], jnp.float64) * u.volt,
                                 diag_delays=jnp.array([0, 3], jnp.int64))
    assert result.data.dtype == jnp.float64
    assert u.get_unit(result.data) == u.mV
    np.testing.assert_allclose(_values(result.todense()), [[3., 0., 3.], [7., 7., 0.]])
    _fresh_state(result, source)
    with pytest.warns(PerformanceWarning):
        integer_increment = _matrix().diag_add(jnp.ones(2, jnp.int32), diag_delays=0)
    assert integer_increment.data.dtype == jnp.result_type(jnp.float32, jnp.int32)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
def test_diag_add_empty_diagonal_does_not_expand_ring_for_unused_scalar_delay(shape):
    source = delaycsr(
        (np.array([], np.float32), np.array([], np.int32), np.zeros(shape[0] + 1, np.int32)),
        shape=shape, edge_delays=0, delay_steps=(0, 2), n_slots=4, backend="jax_raw")
    with pytest.warns(PerformanceWarning):
        result = source.diag_add(jnp.array([], jnp.float32), diag_delays=100)
    assert result.shape == shape
    assert result.nse == 0
    assert result.L == 4
    assert result.delay_steps == (0, 2)
    _fresh_state(result, source)


def test_diag_add_widens_indptr_before_appending_edges(monkeypatch):
    source = delaycsr(
        (np.arange(1, 5, dtype=np.float32), np.zeros(4, np.int32), np.array([0, 2, 4], np.int32)),
        shape=(2, 1), edge_delays=[0, 1, 0, 1], n_slots=2, backend="jax_raw")
    monkeypatch.setattr(region_delay, "_I32_MAX", 4)
    with pytest.warns(PerformanceWarning):
        result = source.diag_add(jnp.ones(1, jnp.float32), diag_delays=0)
    assert result.indptr.dtype == jnp.int64
    assert result.indices.dtype == jnp.int32
    np.testing.assert_array_equal(result.indptr, [0, 3, 5])
    with jax.enable_x64(False):
        with pytest.raises((RuntimeError, ValueError), match="(?i)x64|int64"):
            source.diag_add(jnp.ones(1, jnp.float32), diag_delays=0)


def test_diag_add_is_eager_when_increment_values_are_traced():
    source = _matrix()
    with pytest.raises((RuntimeError, TypeError), match="(?i)jit|trac|concrete|eager"):
        jax.jit(lambda diagonal: source.diag_add(diagonal, diag_delays=0).data)(
            jnp.ones(2, jnp.float32))
