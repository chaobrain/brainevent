"""The shared facade routes contraction size one to MV, all others to MM."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._sddmm import (
    sddmm, sddmm_indices, sddmm_coo_indices, sddmm_bcoo,
    csr_sddmm_dweight, csr_sddmv_dweight, csr_sddmv_dweight_binary,
    csr_sddmv_dweight_weighted, csr_sddmv_p,
)
from jax.experimental.sparse import BCOO

dispatch = importlib.import_module('brainevent._sddmm.dispatch')
mv = importlib.import_module('brainevent._sddmm._mv.sddmv')
mm = importlib.import_module('brainevent._sddmm._mm.sddmm')


@pytest.mark.parametrize('batch', [0, 1, 3])
def test_wrappers_enter_one_dispatcher_and_expected_leaf(monkeypatch, batch):
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
    for call in [lambda: sddmm(a, b, coordinates),
                 lambda: sddmm_indices(a, b, coordinates).data,
                 lambda: sddmm_coo_indices(a, b, coordinates[:, 0], coordinates[:, 1]).data,
                 lambda: sddmm_bcoo(a, b, BCOO((jnp.ones(3), coordinates), shape=(2, 3))).data]:
        # Each wrapper should trace through the dispatcher independently.
        jax.clear_caches()
        np.testing.assert_allclose(call(), expected)
    assert calls == ['mv' if batch == 1 else 'mm'] * 4


def test_legacy_wrappers_reuse_their_jit_compilation(monkeypatch):
    traces = []
    original = mm.sampled
    def record(*args, **kwargs):
        traces.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(mm, 'sampled', record)
    a, b = jnp.ones((2, 3)), jnp.ones((3, 4))
    coords = jnp.array([[0, 1]], jnp.int32)
    for call in [lambda: sddmm_indices(a, b, coords),
                 lambda: sddmm_coo_indices(a, b, coords[:, 0], coords[:, 1]),
                 lambda: sddmm_bcoo(a, b, BCOO((jnp.ones(1), coords), shape=(2, 4)))]:
        jax.clear_caches()
        traces.clear()
        np.testing.assert_array_equal(call().data, [3.])
        np.testing.assert_array_equal(call().data, [3.])
        assert len(traces) == 1


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


def test_mm_is_callable_without_cuda_and_has_clear_cuda_placeholder(monkeypatch):
    def fail_cuda(*args, **kwargs):
        raise AssertionError('MM must use JAX until its CUDA backend is implemented')
    monkeypatch.setattr(mv, 'load_cuda_file', fail_cuda)
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
    for weighted, wrapper in [(False, csr_sddmv_dweight_binary),
                              (True, csr_sddmv_dweight_weighted)]:
        expected = jnp.array([0., -6. if weighted else 3.])
        np.testing.assert_array_equal(wrapper(e, c, i, p, transpose=transpose), expected)
        np.testing.assert_array_equal(csr_sddmv_p.call(
            e, c, i, p, transpose=transpose, weighted=weighted), expected)
        fn = jax.jit(lambda x, y: csr_sddmm_dweight(
            x, y, i, p, transpose=transpose, weighted=weighted, backend='auto'))
        np.testing.assert_array_equal(fn(e[:, None], c[:, None]), expected)
        ir = str(fn.lower(e[:, None], c[:, None]).compiler_ir())
        assert ('hybrid_sddmv' in ir) == (transpose and jax.default_backend() == 'gpu')


@pytest.mark.parametrize('batch', [1, 2])
def test_backend_errors_and_mm_event_fallback(monkeypatch, batch):
    def fail_cuda(*args, **kwargs):
        raise AssertionError('JAX execution must not load CUDA')
    monkeypatch.setattr(mv, 'load_cuda_file', fail_cuda)
    a, b = jnp.ones((2, batch)), jnp.ones((batch, 3))
    i, p = jnp.array([0, 1], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    with pytest.raises(ValueError, match='backend'):
        sddmm(a, b, i, p, backend='invalid')
    with pytest.raises(NotImplementedError, match='CUDA'):
        sddmm(a, b, i, p, backend='cuda_raw')
    if batch > 1:
        for backend in (None, 'auto', 'jax_raw'):
            np.testing.assert_array_equal(sddmm(a, b, i, p, mode='weighted', backend=backend), [2., 2.])
        with pytest.raises(NotImplementedError, match='CUDA'):
            sddmm(a, b, i, p, mode='binary', backend='cuda_raw')
    else:
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
        csr_sddmv_dweight(e[:, None], c[:, None], i, p, transpose=True)
