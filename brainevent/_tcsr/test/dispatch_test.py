# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Actual-platform and public-backend contracts of tiled event dispatch."""

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, TCSR
from brainevent._csr.main import CSR
from brainevent._tcsr.binary.combined.planner import tile_schedule
from brainevent._tcsr.preprocess import build_tile_geometries

api = importlib.import_module('brainevent._tcsr._dispatch')
tiled = importlib.import_module('brainevent._tcsr.binary.binary')
config = importlib.import_module('brainevent.config')
resources = importlib.import_module('brainevent._tcsr._resources')


@pytest.fixture(autouse=True)
def _isolate_mock_backend_selection():
    # Public set_backend/set_default clear JIT caches. Tests patch their
    # internal dictionaries and XLA_FLAGS, so perform the matching teardown.
    yield
    jax.clear_caches()


def _case(transpose, *, homo=False, backend=None):
    matrix = CSR(
        jnp.array([2.] if homo else [2., -1., 3., 4.], jnp.float32),
        jnp.array([0, 2, 1, 2], jnp.int32), jnp.array([0, 2, 4], jnp.int32),
        shape=(2, 3), backend=backend,
    )
    if transpose:
        ptr, idx, cols = matrix.indptr, matrix.indices, matrix.shape[1]
    else:
        ptr, idx, _ = matrix._mirror()
        cols = matrix.shape[0]
    width = tile_schedule(cols, 3).tile_width
    (p,) = build_tile_geometries(idx, ptr, shape=matrix.shape, transpose=transpose,
                                 widths=(width,)).values()
    plan = SimpleNamespace(local_targets=p.local_targets,
                           tile_offsets=p.tile_offsets, tile_width=p.tile_width)
    source = matrix.shape[0 if transpose else 1]
    events = jnp.resize(jnp.array([0., -2., .5, 1., 0., 3.], jnp.float32), (source, 3))
    return matrix, events, plan


def _call(matrix, events, plan, transpose):
    original = BinaryArray(events.T if transpose else events)
    output = api.dispatch_binary_mm(matrix, original, transpose=transpose, plan=plan)
    return output.T if transpose else output


@pytest.mark.parametrize('transpose', [True, False])
def test_cpu_compilation_on_gpu_default_never_lowers_cuda(monkeypatch, transpose):
    matrix, events, plan = _case(transpose)
    monkeypatch.setitem(api.csr_binary_csrmm_p._defaults, 'cpu', 'jax_raw')

    def forbidden(**kwargs):
        raise AssertionError('CPU compilation reached the combined CUDA generator')

    monkeypatch.setattr(tiled.binary_csrmm_p._kernels['gpu']['cuda_raw'],
                        'kernel_generator', forbidden)
    jax.clear_caches()
    fn = jax.jit(lambda x: _call(matrix, x, plan, transpose), backend='cpu')
    result = fn(events)
    expected = (np.asarray(matrix.todense()).T if transpose else np.asarray(matrix.todense())) @ (np.asarray(events) != 0)
    np.testing.assert_allclose(result, expected)
    assert next(iter(result.devices())).platform == 'cpu'
    assert 'tcsr_binary_csrmm' not in fn.lower(events).as_text()


@pytest.mark.parametrize('selection', ['explicit', 'global', 'default', 'deterministic'])
def test_backend_policy_can_select_parent_on_cuda(monkeypatch, selection):
    matrix, events, plan = _case(True, backend='jax_raw' if selection == 'explicit' else None)
    if selection == 'global':
        monkeypatch.setitem(config._global_backends, 'gpu', 'jax_raw')
    elif selection == 'default':
        monkeypatch.setitem(api.csr_binary_csrmm_p._defaults, 'gpu', 'jax_raw')
    elif selection == 'deterministic':
        monkeypatch.setenv('XLA_FLAGS', '--xla_gpu_deterministic_ops=true')
    jax.clear_caches()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('the selected parent backend reached tiled MM')

    monkeypatch.setattr(api, 'binary_csrmm', forbidden)
    result = jax.jit(lambda x: _call(matrix, x, plan, True))(events)
    np.testing.assert_allclose(result, np.asarray(matrix.todense()).T @ (np.asarray(events) != 0))
    assert calls == []


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('homo', [True, False])
def test_cuda_dispatch_runs_canonical_forward_and_gradients(transpose, homo):
    try:
        gpu = jax.devices('gpu')[0]
    except RuntimeError:
        pytest.skip('CUDA device required')
    if 'cuda' not in gpu.client.platform_version.lower():
        pytest.skip('CUDA device required')
    matrix, events, plan = _case(transpose, homo=homo)
    fn = jax.jit(lambda w, x: _call(matrix.with_data(w), x, plan, transpose))
    result = fn(matrix.data, events)
    dense = np.asarray(matrix.todense())
    np.testing.assert_allclose(result, (dense.T if transpose else dense) @ (np.asarray(events) != 0))
    grad = jax.jit(jax.grad(lambda w: fn(w, events).sum()))(matrix.data)
    # Binary events: the weight gradient is the exact activity-mask derivative.
    values = np.asarray(events != 0).sum(axis=1)
    units = np.array([0, 0, 1, 1]) if transpose else np.array([0, 2, 1, 2])
    expected = values[units]
    if homo:
        expected = expected.sum(keepdims=True)
    np.testing.assert_allclose(grad, expected)
    ir = fn.lower(matrix.data, events).as_text()
    assert 'tcsr_' in ir


@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('selection', ['explicit', 'global'])
def test_explicit_and_global_cuda_override_default_and_determinism(monkeypatch, transpose, selection):
    try:
        gpu = jax.devices('gpu')[0]
    except RuntimeError:
        pytest.skip('CUDA device required')
    if 'cuda' not in gpu.client.platform_version.lower():
        pytest.skip('CUDA device required')
    matrix, events, plan = _case(transpose, backend='cuda_raw' if selection == 'explicit' else None)
    monkeypatch.setitem(config._global_backends, 'gpu', 'jax_raw' if selection == 'explicit' else 'cuda_raw')
    monkeypatch.setitem(api.csr_binary_csrmm_p._defaults, 'gpu', 'jax_raw')
    monkeypatch.setitem(api.binary_csrmm_indexed_p._defaults, 'gpu', 'jax_raw')
    monkeypatch.setenv('XLA_FLAGS', '--xla_gpu_deterministic_ops=true')
    jax.clear_caches()
    fn = jax.jit(lambda x: _call(matrix, x, plan, transpose))
    result = fn(events)
    dense = np.asarray(matrix.todense())
    np.testing.assert_allclose(result, (dense.T if transpose else dense) @ (np.asarray(events) != 0))
    assert 'tcsr_' in fn.lower(events).as_text()


@pytest.mark.parametrize('transpose', [True, False])
def test_eager_committed_cpu_operands_use_cpu_with_a_gpu_default(monkeypatch, transpose):
    matrix, events, plan = _case(transpose)
    cpu = jax.devices('cpu')[0]
    matrix = jax.device_put(matrix, cpu)
    events = jax.device_put(events, cpu)
    plan.local_targets = jax.device_put(plan.local_targets, cpu)
    plan.tile_offsets = jax.device_put(plan.tile_offsets, cpu)
    monkeypatch.setitem(api.csr_binary_csrmm_p._defaults, 'cpu', 'jax_raw')
    jax.clear_caches()
    result = _call(matrix, events, plan, transpose)
    dense = np.asarray(matrix.todense())
    np.testing.assert_allclose(result, (dense.T if transpose else dense) @ (np.asarray(events) != 0))
    assert next(iter(result.devices())).platform == 'cpu'


@pytest.mark.parametrize('transpose', [True, False])
def test_resource_shortage_executes_original_parent_operator(monkeypatch, transpose):
    try:
        gpu = jax.devices('gpu')[0]
    except RuntimeError:
        pytest.skip('CUDA device required')
    if 'cuda' not in gpu.client.platform_version.lower():
        pytest.skip('CUDA device required')
    matrix, events, plan = _case(transpose)
    monkeypatch.setattr(resources, '_query_launch_resources',
                        lambda device: resources.LaunchResources(0, 0, 0, (0, 0)))
    original = BinaryArray(events.T if transpose else events)
    calls = []
    name = '__rmatmul__' if transpose else '__matmul__'
    parent = getattr(CSR, name)

    def inherited(self, operand):
        calls.append((type(operand), operand.value.shape, operand.value.dtype))
        return parent(self, operand)

    def unexpected_execution(value):
        raise AssertionError('resource shortage executed the tiled branch')

    def unavailable_tiled(weights, indices, indptr, x, *, shape, transpose, **kwargs):
        # Tracing both branches is legal; this callback must never execute.
        jax.debug.callback(unexpected_execution, jnp.int32(0))
        return jnp.zeros((shape[1] if transpose else shape[0], x.shape[1]), weights.dtype)

    monkeypatch.setattr(CSR, name, inherited)
    monkeypatch.setattr(api, 'binary_csrmm', unavailable_tiled)
    jax.clear_caches()
    result = api.dispatch_binary_mm(matrix, original, transpose=transpose, plan=plan)
    dense = np.asarray(matrix.todense())
    expected = (np.asarray(original.value) != 0) @ dense if transpose else dense @ (np.asarray(original.value) != 0)
    np.testing.assert_allclose(result, expected)
    assert calls and all(c == (BinaryArray, original.value.shape, original.value.dtype) for c in calls)


def test_explicit_parent_selection_preserves_operand_identity(monkeypatch):
    matrix, events, plan = _case(True, backend='jax_raw')
    operand = BinaryArray(events.T)
    received = []

    def inherited(self, other):
        received.append(other)
        return jnp.zeros((events.shape[1], matrix.shape[1]), matrix.dtype)

    monkeypatch.setattr(CSR, '__rmatmul__', inherited)
    api.dispatch_binary_mm(matrix, operand, transpose=True, plan=plan)
    assert received == [operand]


@pytest.mark.parametrize('decision', [True, False, None])
def test_trace_time_resource_decision_traces_only_the_selected_branch(monkeypatch, decision):
    """Known resources select one CUDA branch; only unknown targets stage the runtime check."""
    matrix, events, plan = _case(True)
    monkeypatch.setattr(api, 'static_launch_supported', lambda **kwargs: decision)
    jax.clear_caches()
    text = str(jax.make_jaxpr(lambda x: _call(matrix, x, plan, True))(events))
    assert ('tcsr_launch_supported' in text) == (decision is None)
    assert ('tcsr_combined_binary_csrmm' in text) == (decision is not False)


def test_preparing_another_tile_width_does_not_retrace_dispatch(monkeypatch):
    """Tiles are separate operands, so new widths leave the dispatch program cached."""
    monkeypatch.setitem(api.csr_binary_csrmm_p._defaults, 'cpu', 'jax_raw')
    source = CSR(jnp.ones(4, jnp.float32), jnp.array([0, 5, 1, 63], jnp.int32),
                 jnp.array([0, 2, 4], jnp.int32), shape=(2, 64))
    matrix = TCSR(source).prepare_binary_mm(2, transpose=True)
    plan = matrix._get_tile_geometry(2, transpose=True)
    events = BinaryArray(jnp.ones((2, 2), bool))
    expected = api.dispatch_binary_mm(matrix, events, transpose=True, plan=plan)
    cached = api._dispatch_on_platform._cache_size()
    wider = matrix.prepare_binary_mm(8, transpose=True)
    assert len(wider._cache.tile_geometries) == 2
    result = api.dispatch_binary_mm(wider, events, transpose=True, plan=plan)
    assert api._dispatch_on_platform._cache_size() == cached
    np.testing.assert_allclose(result, expected)
