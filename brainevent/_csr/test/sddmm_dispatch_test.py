"""CSR sampled gradients share dispatch, backend selection and derivatives."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest


dispatch = importlib.import_module('brainevent._sddmm.dispatch')
float_csr = importlib.import_module('brainevent._csr.float.float')
binary_csr = importlib.import_module('brainevent._csr.binary.binary')
indexed_csr = importlib.import_module('brainevent._csr.binary.binary_indexed')

_CASES = ['float', 'binary', 'weighted', 'indexed_binary', 'indexed_weighted']


def _operands(case, transpose, batch, homo=False, backend='jax_raw'):
    indices = jnp.array([2, 0, 1, 1], jnp.int32)
    indptr = jnp.array([0, 2, 2, 4], jnp.int32)
    rows = jnp.array([0, 0, 2, 2], jnp.int32)
    # Two physical slots share one canonical weight.
    perm = jnp.array([2, 0, 1, 1], jnp.int32)
    weights = jnp.arange(1 if homo else 3 if case.startswith('indexed') else 4,
                          dtype=jnp.float32) + 1
    event_shape = (3 if transpose else 4,) + (() if batch is None else (batch,))
    events = jnp.asarray(np.resize([0., -2., 1., 3.], event_shape))
    if case == 'float':
        op = float_csr.csrmv if batch is None else float_csr.csrmm
        call = lambda w: op(w, indices, indptr, events, shape=(3, 4),
                            transpose=transpose, backend=backend)
    elif case.startswith('indexed'):
        op = indexed_csr.binary_csrmv_indexed if batch is None else indexed_csr.binary_csrmm_indexed
        call = lambda w: op(
            w, indices, indptr, perm, events, shape=(3, 4), transpose=transpose,
            weighted=case.endswith('weighted'), backend=backend)
    else:
        op = binary_csr.binary_csrmv if batch is None else binary_csr.binary_csrmm
        call = lambda w: op(
            w, indices, indptr, events, shape=(3, 4), transpose=transpose,
            weighted=case == 'weighted', backend=backend)
    return call, weights, events, rows, indices, perm


@pytest.mark.parametrize('case,batch', [(case, batch) for batch in (None, 0, 1, 3)
                                       for case in _CASES])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('nonfinite', [False, True])
def test_csr_consumers_dispatch_slot_gradients(monkeypatch, case, transpose, batch, homo, nonfinite):
    calls = []
    original = dispatch.sddmm
    def record(a, b, indices, indptr=None, **kwargs):
        backend = kwargs.get('backend')
        calls.append((a.shape[1] if a.ndim == 2 else 1, indices.ndim,
                      None if indptr is None else indptr.shape,
                      kwargs.get('mode', 'float'), getattr(backend, 'requested', backend)))
        return original(a, b, indices, indptr, **kwargs)
    monkeypatch.setattr(dispatch, 'sddmm', record)
    jax.clear_caches()
    call, w, events, rows, cols, perm = _operands(case, transpose, batch, homo)
    ct_shape = (4 if transpose else 3,) + (() if batch is None else (batch,))
    ct = jnp.asarray(np.resize([2., -.5, 3., 0.], ct_shape))
    if nonfinite and batch != 0:
        if batch is None:
            ct = ct.at[0].set(jnp.inf).at[1].set(jnp.nan)
        else:
            ct = ct.at[0, 0].set(jnp.inf).at[1, 0].set(jnp.nan)
    def pullback(w, ct):
        return jax.vjp(call, w)[1](ct)[0]
    actual = jax.jit(pullback)(w, ct)
    unit, out = (rows, cols) if transpose else (cols, rows)
    x, terms = events[unit], ct[out]
    if case == 'float' or case.endswith('weighted'):
        terms = x * terms
    if case != 'float':
        terms = jnp.where(x != 0, terms, 0.)
    per_slot = terms if batch is None else jnp.sum(terms, axis=1)
    if homo:
        expected = jnp.sum(per_slot).reshape(1)
    elif case.startswith('indexed'):
        expected = jnp.zeros_like(w).at[perm].add(per_slot)
    else:
        expected = per_slot
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    mode = 'float' if case == 'float' else 'weighted' if case.endswith('weighted') else 'binary'
    backend = None if case == 'float' else 'jax_raw'
    assert calls == [(1 if batch is None else batch, 1, (4,), mode, backend)]


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('batch', [None, 1, 3])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('indexed', [False, True])
def test_event_consumers_preserve_activity_before_narrowing(transpose, batch, weighted, indexed):
    rows = jnp.array([0, 0, 1, 2], jnp.int32)
    cols = jnp.array([2, 0, 1, 1], jnp.int32)
    indptr = jnp.array([0, 2, 3, 4], jnp.int32)
    perm = jnp.array([2, 0, 3, 1], jnp.int32)
    size = 3 if transpose else 4
    events = jnp.array([0., 1e-8, -2., 3.], jnp.float32)[:size]
    if batch is not None:
        events = jnp.broadcast_to(events[:, None], (size, batch))
    ct_shape = (4 if transpose else 3,) + (() if batch is None else (batch,))
    ct = jnp.full(ct_shape, jnp.inf, jnp.float16)
    weights = jnp.ones(4, jnp.float16)
    if indexed:
        op = indexed_csr.binary_csrmv_indexed if batch is None else indexed_csr.binary_csrmm_indexed
        call = lambda w: op(w, cols, indptr, perm, events, shape=(3, 4), transpose=transpose,
                            weighted=weighted, backend='jax_raw')
    else:
        op = binary_csr.binary_csrmv if batch is None else binary_csr.binary_csrmm
        call = lambda w: op(w, cols, indptr, events, shape=(3, 4), transpose=transpose,
                            weighted=weighted, backend='jax_raw')
    actual = jax.jit(lambda w, c: jax.vjp(call, w)[1](c)[0])(weights, ct)
    unit, out = (rows, cols) if transpose else (cols, rows)
    x = events[unit]
    terms = x.astype(ct.dtype) * ct[out] if weighted else ct[out]
    expected = jnp.where(x != 0, terms, 0)
    if batch is not None:
        expected = jnp.sum(expected, axis=1)
    if indexed:
        expected = jnp.zeros_like(weights).at[perm].add(expected)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual.dtype == ct.dtype


@pytest.mark.parametrize('case,batch', [(case, batch) for batch in (None, 1, 3)
                                       for case in _CASES])
@pytest.mark.parametrize('transpose', [False, True])
def test_migrated_consumers_preserve_second_order_gradients(case, transpose, batch):
    call, w, _, _, _, _ = _operands(case, transpose, batch)
    _check_second_order_gradients(call, w)


def _check_second_order_gradients(call, w):
    loss = lambda w: jnp.sum(call(w) ** 2)
    direction = jnp.arange(w.size, dtype=w.dtype) + .5
    got_fwd = jax.jit(lambda w: jax.jvp(jax.grad(loss), (w,), (direction,))[1])(w)
    got_rev = jax.jit(jax.grad(lambda w: jnp.vdot(jax.grad(loss)(w), direction)))(w)
    # The forward is linear in weights; its explicit basis matrix gives the
    # Hessian 2*J.T@J independently of the SDDMM transpose implementation.
    basis_outputs = jax.vmap(call)(jnp.eye(w.size, dtype=w.dtype)).reshape(w.size, -1)
    expected = 2 * basis_outputs @ (basis_outputs.T @ direction)
    np.testing.assert_allclose(got_fwd, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got_rev, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('indexed', [False, True])
def test_shared_mv_gradient_avoids_unit_weight_forward_overflow(transpose, indexed):
    indices = jnp.array([0, 0] if transpose else [0, 1], jnp.int32)
    indptr = jnp.array([0, 1, 2] if transpose else [0, 2], jnp.int32)
    shape = (2, 1) if transpose else (1, 2)
    events = jnp.full((2,), 60000, jnp.float16)
    weights = jnp.array([1e-4], jnp.float16)
    ct = jnp.array([1e-4], jnp.float16)
    if indexed:
        call = lambda w: indexed_csr.binary_csrmv_indexed(
            w, indices, indptr, jnp.zeros(2, jnp.int32), events,
            shape=shape, transpose=transpose, weighted=True, backend='jax_raw')
    else:
        call = lambda w: binary_csr.binary_csrmv(
            w, indices, indptr, events, shape=shape, transpose=transpose,
            weighted=True, backend='jax_raw')
    actual = jax.jit(lambda w, c: jax.vjp(call, w)[1](c)[0])(weights, ct)
    expected = jnp.sum(events * ct).reshape(1)
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('case', _CASES[1:])
@pytest.mark.parametrize('batch', [None, 1, 3])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('selection', ['jax_raw', 'cuda_raw', 'default_jax', 'default_cuda'])
def test_event_weight_gradient_follows_csr_backend(monkeypatch, case, batch, transpose, selection):
    if jax.default_backend() != 'gpu':
        pytest.skip('inspect the selected CSR and sampled CUDA backends on a GPU')
    indexed = case.startswith('indexed')
    module = indexed_csr if indexed else binary_csr
    name = 'binary_csrmv' if batch is None else 'binary_csrmm'
    primitive = getattr(module, name + ('_indexed_p' if indexed else '_p'))
    if selection.startswith('default_'):
        selected = 'cuda_raw' if selection == 'default_cuda' else 'jax_raw'
        monkeypatch.setitem(primitive._defaults, 'gpu', selected)
        backend = None
    else:
        selected = backend = selection
    native = selected == 'cuda_raw' and transpose
    recorded = []
    original = dispatch.sddmm

    def record(a, b, indices, indptr=None, **kwargs):
        request = kwargs.get('backend')
        recorded.append(getattr(request, 'requested', request))
        return original(a, b, indices, indptr, **kwargs)

    monkeypatch.setattr(dispatch, 'sddmm', record)
    if native:
        def fail_coo(*args, **kwargs):
            raise AssertionError('native weight gradients must retain the CSR structure')
        leaf = 'mv.sddmv' if batch in (None, 1) else 'mm.sddmm'
        monkeypatch.setattr(importlib.import_module(f'brainevent._sddmm._{leaf}'), '_csr_to_coo', fail_coo)
    jax.clear_caches()
    call, w, events, rows, cols, perm = _operands(case, transpose, batch, backend=backend)
    shape = (4 if transpose else 3,) + (() if batch is None else (batch,))
    ct = jnp.asarray(np.resize([2., -.5, 3., 0.], shape), jnp.float32)
    fn = jax.jit(lambda weights, cotangent: jax.vjp(call, weights)[1](cotangent)[0])
    result = fn(w, ct)
    unit, out = (rows, cols) if transpose else (cols, rows)
    terms = events[unit] * ct[out] if case.endswith('weighted') else ct[out]
    expected = jnp.where(events[unit] != 0, terms, 0.)
    if batch is not None:
        expected = jnp.sum(expected, axis=1)
    if indexed:
        expected = jnp.zeros_like(w).at[perm].add(expected)
    np.testing.assert_allclose(result, expected)
    assert recorded == [backend]
    ir = fn.lower(w, ct).as_text()
    target = 'hybrid_sddmv_' if batch in (None, 1) else 'batched_v3_sddmm_'
    assert (target in ir) == native
    if native:
        assert 'stablehlo.is_finite' not in ir


@pytest.mark.parametrize('case', _CASES[1:])
@pytest.mark.parametrize('batch', [None, 3])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('backend', [None, 'cuda_raw'])
def test_native_event_consumers_preserve_second_order_gradients(monkeypatch, case, batch, transpose, backend):
    if jax.default_backend() != 'gpu':
        pytest.skip('native sampled derivatives require a GPU')
    module = indexed_csr if case.startswith('indexed') else binary_csr
    name = 'binary_csrmv' if batch is None else 'binary_csrmm'
    primitive = getattr(module, name + ('_indexed_p' if case.startswith('indexed') else '_p'))
    monkeypatch.setitem(primitive._defaults, 'gpu', 'cuda_raw')
    jax.clear_caches()
    call, w, _, _, _, _ = _operands(case, transpose, batch, backend=backend)
    if case.startswith('indexed'):
        # The native indexed forward requires an nnz-sized canonical buffer.
        # Keep the repeated perm entries and leave its fourth weight unused.
        w = jnp.arange(1, 5, dtype=w.dtype)
    _check_second_order_gradients(call, w)


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('batch', [None, 3])
@pytest.mark.parametrize('transpose', [False, True])
def test_event_gradient_resolves_backend_on_compilation_platform(monkeypatch, indexed, batch, transpose):
    if jax.default_backend() != 'gpu':
        pytest.skip('the default device must differ from the CPU compilation target')
    module = indexed_csr if indexed else binary_csr
    name = 'binary_csrmv' if batch is None else 'binary_csrmm'
    primitive = getattr(module, name + ('_indexed_p' if indexed else '_p'))
    monkeypatch.setitem(primitive._defaults, 'gpu', 'cuda_raw')
    monkeypatch.setitem(primitive._defaults, 'cpu', 'jax_raw' if indexed else 'numba')
    jax.clear_caches()
    case = 'indexed_weighted' if indexed else 'weighted'
    call, w, events, rows, cols, perm = _operands(case, transpose, batch, backend=None)
    shape = (4 if transpose else 3,) + (() if batch is None else (batch,))
    ct = jnp.ones(shape, jnp.float32)
    fn = jax.jit(lambda weights, cotangent: jax.vjp(call, weights)[1](cotangent)[0], backend='cpu')
    actual = fn(w, ct)
    unit = rows if transpose else cols
    expected = events[unit]
    if batch is not None:
        expected = expected.sum(axis=1)
    if indexed:
        expected = jnp.zeros_like(w).at[perm].add(expected)
    np.testing.assert_allclose(actual, expected)
    assert next(iter(actual.devices())).platform == 'cpu'
    ir = fn.lower(w, ct).as_text()
    assert 'hybrid_sddmv_' not in ir and 'batched_v3_sddmm_' not in ir


@pytest.mark.parametrize('batch', [None, 3])
@pytest.mark.parametrize('default,global_backend,requested,native', [
    ('cuda_raw', 'jax_raw', None, False),
    ('jax_raw', 'cuda_raw', None, True),
    ('cuda_raw', 'cuda_raw', 'jax_raw', False),
    ('jax_raw', 'jax_raw', 'cuda_raw', True),
])
def test_event_gradient_respects_backend_priority(monkeypatch, batch, default, global_backend, requested, native):
    if jax.default_backend() != 'gpu':
        pytest.skip('CUDA backend precedence requires a GPU')
    from brainevent import config
    primitive = binary_csr.binary_csrmv_p if batch is None else binary_csr.binary_csrmm_p
    monkeypatch.setitem(primitive._defaults, 'gpu', default)
    monkeypatch.setitem(config._global_backends, 'gpu', global_backend)
    jax.clear_caches()
    call, w, _, _, _, _ = _operands('binary', True, batch, backend=requested)
    shape = (4,) + (() if batch is None else (batch,))
    ct = jnp.ones(shape, jnp.float32)
    fn = jax.jit(lambda weights, cotangent: jax.vjp(call, weights)[1](cotangent)[0])
    result = fn(w, ct)
    reference, _, _, _, _, _ = _operands('binary', True, batch)
    expected = jax.vjp(reference, w)[1](ct)[0]
    np.testing.assert_allclose(result, expected)
    ir = fn.lower(w, ct).as_text()
    target = 'hybrid_sddmv_' if batch is None else 'batched_v3_sddmm_'
    assert (target in ir) == native
