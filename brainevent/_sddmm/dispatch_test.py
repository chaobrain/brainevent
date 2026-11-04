"""The shared facade routes contraction size one to MV, all others to MM."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._sddmm import (
    sddmm, csr_sddmm_dweight, csr_sddmv_dweight,
)
from brainevent._registry import get_registry

dispatch = importlib.import_module('brainevent._sddmm.dispatch')
mv = importlib.import_module('brainevent._sddmm._mv.sddmv')
mm = importlib.import_module('brainevent._sddmm._mm.sddmm')


@pytest.mark.parametrize('requested', [None, 'cuda_raw'])
def test_csr_gradient_dispatch_preserves_forward_config(monkeypatch, requested):
    """Explicit CUDA must retain the snapshot when backend metadata is unwrapped."""
    from brainevent._csr.config import CSRConfig
    from brainevent._sddmm._common import _CSRBackend
    config_spec = CSRConfig(tpr_threshold=16, task_nnz=64).kernel_args()[0]
    captured = []

    def sampled(a, b, indices, indptr, **kwargs):
        captured.append(kwargs['config_spec'])
        return jnp.zeros(indices.shape, jnp.float32)

    monkeypatch.setattr(mv, 'sampled', sampled)
    source = _CSRBackend(object(), requested, config_spec)
    indices, indptr = jnp.array([0], jnp.int32), jnp.array([0, 1], jnp.int32)
    fn = jax.jit(lambda a, b: dispatch.sddmm(
        a, b, indices, indptr, mode='weighted', backend=source))
    np.testing.assert_array_equal(fn(jnp.ones(1), jnp.ones(1)), jnp.zeros(1))
    assert captured == [config_spec]


@pytest.mark.parametrize('batch', [0, 1, 3])
def test_dispatch_selects_leaf_by_contraction_size(monkeypatch, batch):
    calls = []
    for name, module in [('mv', mv), ('mm', mm)]:
        original = module.sampled
        def record(*args, _name=name, _fn=original, **kwargs):
            calls.append(_name)
            return _fn(*args, **kwargs)
        monkeypatch.setattr(module, 'sampled', record)
    a = jnp.arange(2 * batch, dtype=jnp.float32).reshape(2, batch)
    b = jnp.arange(batch * 3, dtype=jnp.float32).reshape(batch, 3)
    coordinates = jnp.array([[1, 2], [0, 0], [1, 2]], dtype=jnp.int32)
    expected = (a @ b)[coordinates[:, 0], coordinates[:, 1]]
    np.testing.assert_allclose(jax.jit(sddmm)(a, b, coordinates), expected)
    assert calls == ['mv' if batch == 1 else 'mm']


@pytest.mark.parametrize('name,batch', [
    ('csr_sddmv', None), ('csr_sddmm', None),
    ('csr_sddmm', 0), ('csr_sddmm', 1), ('csr_sddmm', 3),
])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_registry_calls_adapt_csr_operands_and_select_leaf(monkeypatch, name, batch, transpose, weighted):
    calls = []
    for leaf, module in [('mv', mv), ('mm', mm)]:
        original = module.sampled

        def record(*args, _leaf=leaf, _fn=original, **kwargs):
            calls.append(_leaf)
            return _fn(*args, **kwargs)

        monkeypatch.setattr(module, 'sampled', record)
    indices = jnp.array([2, 0, 1, 1], jnp.int32)
    indptr = jnp.array([0, 2, 2, 4], jnp.int32)
    rows = jnp.array([0, 0, 2, 2], jnp.int32)
    tail = () if batch is None else (batch,)
    events = jnp.asarray(np.resize([0., -2., 1., 3.], (3 if transpose else 4,) + tail))
    ct = jnp.asarray(np.resize([2., 1., -3., .5], (4 if transpose else 3,) + tail))
    unit, out = (rows, indices) if transpose else (indices, rows)
    terms = events[unit] * ct[out] if weighted else ct[out]
    expected = jnp.where(events[unit] != 0, terms, 0.)
    if batch is not None:
        expected = expected.sum(axis=1)
    primitive = get_registry()[name]
    fn = jax.jit(lambda e, c: primitive.call(
        e, c, indices, indptr, transpose=transpose, weighted=weighted, backend='jax_raw'))
    np.testing.assert_array_equal(fn(events, ct), expected)
    assert calls == ['mv' if batch in (None, 1) else 'mm']


@pytest.mark.parametrize('batch', [0, 1, 4])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('nonfinite', [False, True])
def test_csr_event_and_plain_semantics(batch, mode, transpose, nonfinite):
    e = jnp.asarray(np.resize([0., -2., 1.], (3 if transpose else 4, batch)), jnp.float32)
    c = jnp.asarray(np.resize([1., 0., -3.], (4 if transpose else 3, batch)), jnp.float32)
    if nonfinite and batch:
        c = c.at[0, 0].set(jnp.inf)
    a, b = (e, c.T) if transpose else (c, e.T)
    indices = jnp.array([2, 0, 1, 1], jnp.int32)
    indptr = jnp.array([0, 2, 2, 4], jnp.int32)
    rows = jnp.array([0, 0, 2, 2], jnp.int32)
    unit, out = (rows, indices) if transpose else (indices, rows)
    values = c[out] if mode == 'binary' else e[unit] * c[out]
    if mode != 'float':
        values = jnp.where(e[unit] != 0, values, 0)
    expected = jnp.sum(values, axis=1)
    result = jax.jit(lambda x,y: sddmm(x,y,indices,indptr,mode=mode,
                                      transpose=transpose,backend='jax_raw'))(a,b)
    np.testing.assert_allclose(result, expected, equal_nan=True)
    if mode != 'float':
        wrapped = csr_sddmm_dweight(e,c,indices,indptr,transpose=transpose,
                                    weighted=mode == 'weighted',backend='jax_raw')
        np.testing.assert_allclose(wrapped, expected, equal_nan=True)


def test_vectors_and_singleton_matrices_agree():
    a = jnp.array([2., 3.])
    b = jnp.array([4., 5., 6.])
    coords = jnp.array([[0, 1], [1, 2]], jnp.int32)
    np.testing.assert_array_equal(sddmm(a,b,coords), sddmm(a[:,None],b[None,:],coords))
    np.testing.assert_array_equal(sddmm(a,b,coords), [10.,18.])


def test_ordinary_mm_is_callable_without_cuda(monkeypatch):
    def fail_cuda(*args, **kwargs):
        raise AssertionError('ordinary MM must use JAX')
    monkeypatch.setattr(mv, 'load_cuda_file', fail_cuda)
    monkeypatch.setattr(mm, 'load_cuda_file', fail_cuda)
    a, b = jnp.ones((2, 3)), jnp.ones((3, 4))
    coords = jnp.array([[0, 1]], jnp.int32)
    np.testing.assert_array_equal(jax.jit(sddmm)(a,b,coords), [3.])
    with pytest.raises(NotImplementedError, match='CUDA'):
        sddmm(a,b,coords,backend='cuda_raw')


@pytest.mark.parametrize('batch', [1, 3])
def test_ordinary_jax_derivatives_and_vmap(batch):
    a = jnp.arange(2 * batch, dtype=jnp.float32).reshape(2, batch) + 1
    b = jnp.arange(batch * 3, dtype=jnp.float32).reshape(batch, 3) + 2
    coords = jnp.array([[0, 2], [1, 0]], jnp.int32)
    loss = lambda x: jnp.sum(sddmm(x,b,coords) ** 2)
    reference = lambda x: jnp.sum((x @ b)[coords[:,0],coords[:,1]] ** 2)
    np.testing.assert_allclose(jax.grad(loss)(a), jax.grad(reference)(a))
    np.testing.assert_allclose(jax.jacfwd(jax.grad(loss))(a), jax.jacfwd(jax.grad(reference))(a))
    actual = jax.jit(jax.vmap(lambda x: sddmm(x,b,coords)))(jnp.stack([a,a*2]))
    np.testing.assert_allclose(actual, jnp.stack([sddmm(a,b,coords),sddmm(a*2,b,coords)]))


def test_plain_float_keeps_zero_times_inf():
    i, p = jnp.array([0], jnp.int32), jnp.array([0,1], jnp.int32)
    e, c = jnp.array([0.]), jnp.array([jnp.inf])
    assert jnp.isnan(sddmm(e,c,i,p)[0])
    assert csr_sddmv_dweight(e,c,i,p,transpose=True,weighted=True,backend='jax_raw')[0] == 0


@pytest.mark.parametrize('transpose', [False, True])
def test_csr_singleton_matrices_and_vector_wrappers_share_native_routing(transpose):
    e, c = jnp.array([0., -2.]), jnp.array([jnp.inf, 3.])
    i, p = jnp.array([0, 1], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    for weighted in (False, True):
        expected = jnp.array([0., -6. if weighted else 3.])
        np.testing.assert_array_equal(csr_sddmv_dweight(
            e, c, i, p, transpose=transpose, weighted=weighted), expected)
        np.testing.assert_array_equal(get_registry()['csr_sddmv'].call(
            e, c, i, p, transpose=transpose, weighted=weighted), expected)
        fn = jax.jit(lambda x, y: csr_sddmm_dweight(
            x, y, i, p, transpose=transpose, weighted=weighted, backend='auto'))
        np.testing.assert_array_equal(fn(e[:, None], c[:, None]), expected)
        ir = str(fn.lower(e[:, None], c[:, None]).compiler_ir())
        assert ('hybrid_sddmv' in ir) == (transpose and jax.default_backend() == 'gpu')


@pytest.mark.parametrize('batch', [1, 2])
def test_backend_errors_and_nontransposed_event_fallback(monkeypatch, batch):
    def fail_cuda(*args, **kwargs):
        raise AssertionError('JAX execution must not load CUDA')
    monkeypatch.setattr(mv, 'load_cuda_file', fail_cuda)
    monkeypatch.setattr(mm, 'load_cuda_file', fail_cuda)
    a, b = jnp.ones((2, batch)), jnp.ones((batch, 3))
    i, p = jnp.array([0, 1], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    with pytest.raises(ValueError, match='backend'):
        sddmm(a, b, i, p, backend='invalid')
    with pytest.raises(NotImplementedError, match='CUDA'):
        sddmm(a, b, i, p, backend='cuda_raw')
    for backend in (None, 'auto', 'jax_raw'):
        np.testing.assert_array_equal(sddmm(a, b, i, p, mode='weighted', transpose=False,
                                           backend=backend), [batch, batch])
    with pytest.raises(ValueError, match='transpose=True'):
        sddmm(a, b, i, p, mode='binary', transpose=False, backend='cuda_raw')


@pytest.mark.parametrize('change,error,match', [
    ({'mode': 'invalid'}, ValueError, 'mode'),
    ({'transpose': 1}, TypeError, 'transpose'),
    ({'A': jnp.ones((2, 1, 1))}, ValueError, 'rank-2'),
    ({'B': jnp.ones((2, 3))}, ValueError, 'contraction'),
    ({'indices': jnp.zeros((2, 2), jnp.int32)}, ValueError, 'rank 1'),
    ({'indices': jnp.zeros(2)}, TypeError, 'indices'),
    ({'indptr': jnp.zeros(3)}, TypeError, 'indptr'),
    ({'indptr': jnp.zeros(2, jnp.int32)}, ValueError, 'row count'),
    ({'B': jnp.ones((1, 0))}, ValueError, 'positive'),
    ({'mode': 'binary', 'A': jnp.ones((2, 1), jnp.complex64)}, TypeError, 'events'),
    ({'mode': 'binary', 'B': jnp.ones((1, 3), jnp.int32)}, TypeError, 'ct'),
    ({'indptr': None, 'indices': jnp.zeros(2, jnp.int32)}, ValueError, 'COO'),
    ({'indptr': None, 'indices': jnp.zeros((2, 2))}, TypeError, 'COO'),
])
def test_dispatch_rejects_invalid_metadata(change, error, match):
    args = dict(A=jnp.ones((2, 1)), B=jnp.ones((1, 3)), indices=jnp.array([0, 1], jnp.int32),
                indptr=jnp.array([0, 1, 2], jnp.int32))
    args.update(change)
    with pytest.raises(error, match=match):
        sddmm(**args)


def test_csr_wrappers_reject_ambiguous_ranks_and_flags():
    e, c = jnp.ones(2), jnp.ones(3)
    i, p = jnp.array([0, 1], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    with pytest.raises(TypeError, match='bool'):
        csr_sddmm_dweight(e, c, i, p, transpose=True, weighted=1)
    with pytest.raises(ValueError, match='both'):
        csr_sddmm_dweight(e[:, None], c, i, p, transpose=True)
    with pytest.raises(ValueError, match='rank 1'):
        get_registry()['csr_sddmv'].call(e[:, None], c[:, None], i, p, transpose=True)


@pytest.mark.parametrize('batch', [0, 1, 3])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_explicit_jax_csr_keeps_derivatives(batch, transpose, weighted):
    indices = jnp.array([2, 0, 1, 1], jnp.int32)
    indptr = jnp.array([0, 2, 2, 4], jnp.int32)
    rows = jnp.array([0, 0, 2, 2], jnp.int32)
    events = jnp.asarray(np.resize([0., -2., 1., 3.], (3 if transpose else 4, batch)), jnp.float32)
    ct_shape = (4 if transpose else 3, batch)
    ct = jnp.arange(ct_shape[0] * batch, dtype=jnp.float32).reshape(ct_shape)
    unit, out = (rows, indices) if transpose else (indices, rows)

    def call(c):
        return csr_sddmm_dweight(events, c, indices, indptr, transpose=transpose,
                                 weighted=weighted, backend='jax_raw')

    def reference(c):
        x = events[unit]
        return jnp.sum(jnp.where(x != 0, x * c[out] if weighted else c[out], 0), axis=1)

    tangent = jnp.ones_like(ct)
    actual = jax.jit(lambda c: jax.jvp(call, (c,), (tangent,))[1])(ct)
    np.testing.assert_allclose(actual, reference(tangent))
    def hvp(fn):
        loss = lambda c: jnp.sum(fn(c) ** 2)
        return jax.jit(lambda c: jax.jvp(jax.grad(loss), (c,), (tangent,))[1])(ct)

    np.testing.assert_allclose(hvp(call), hvp(reference))
