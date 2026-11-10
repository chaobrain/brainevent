# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_sddmm/dispatch.py``, grouped by behavior."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from brainevent._compatible_import import varying_axes
from brainevent._registry import get_registry
from brainevent._sddmm import csr_sddmm_dweight, csr_sddmv_dweight, sddmm
from brainevent._test_util import jax_x64_enabled


# Contraction dispatch, validation and backend routing
# ----------------------------------------------------

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


@pytest.mark.parametrize('batch', [1, 3, 65536])
@pytest.mark.parametrize('mode', ['float', 'binary', 'weighted'])
@pytest.mark.parametrize('transpose', [False, True])
def test_csr_gradient_dispatch_preserves_explicit_cusparse(monkeypatch, batch, mode, transpose):
    from brainevent._sddmm._common import _CSRBackend

    captured = []

    def sampled(a, b, indices, indptr, **kwargs):
        captured.append(kwargs['backend'])
        return jnp.full(indices.shape, 7, jnp.float32)

    monkeypatch.setattr(mv if batch == 1 else mm, 'sampled', sampled)
    indices = jnp.array([0, 1], jnp.int32)
    indptr = jnp.array([0, 1, 2], jnp.int32)
    source = _CSRBackend(object(), 'cusparse')
    result = jax.jit(lambda a, b: sddmm(
        a, b, indices, indptr, mode=mode, transpose=transpose, backend=source,
    ))(jnp.ones((2, batch), jnp.float32), jnp.ones((batch, 3), jnp.float32))

    np.testing.assert_array_equal(result, [7, 7])
    assert captured == ['cusparse']


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


# Sampled values, coordinate order, derivatives and sharding
# ----------------------------------------------------------

def _sampled_eqns(f, *args):
    """Collect equations, including nested jaxprs, before primitive lowering."""
    eqns = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            eqns.append(eqn)
            for value in eqn.params.values():
                for sub in value if isinstance(value, (tuple, list)) else (value,):
                    sub = getattr(sub, 'jaxpr', sub)
                    if hasattr(sub, 'eqns'):
                        walk(sub)

    walk(jax.make_jaxpr(f)(*args).jaxpr)
    return eqns


def _sampled_shard_operands():
    A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 1
    B = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 2
    indices = jnp.array([[0, 0], [1, 2], [0, 3]], jnp.int32)
    return A, B, indices


def _sampled_run_sharded(a_spec, *, check_vma=True):
    A, B, indices = _sampled_shard_operands()
    seen = []

    def local(a, b):
        values = sddmm(a[0] if a_spec == P('s') else a, b[0], indices)
        seen.append(varying_axes(jax.typeof(values)))
        return values[None]

    mesh = Mesh(np.asarray(jax.devices()[:1]), ('s',))
    f = jax.shard_map(local, mesh=mesh, in_specs=(a_spec, P('s')),
                      out_specs=P('s'), check_vma=check_vma)
    args = (A[None] if a_spec == P('s') else A, B[None])
    actual = jax.jit(f)(*args)
    np.testing.assert_array_equal(actual[0], jax.jit(sddmm)(A, B, indices))
    return seen, _sampled_eqns(f, *args)


@pytest.mark.parametrize('batch', [0, 1, 3])
@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_sampled_values_preserve_order_duplicates_and_dtype(batch, dtype):
    with jax_x64_enabled():
        A = (jnp.arange(4 * batch, dtype=dtype).reshape(4, batch) - 3) / 4
        B = (jnp.arange(batch * 5, dtype=dtype).reshape(batch, 5) - 2) / 8
        indices = jnp.array([[3, 4], [0, 1], [2, 3], [3, 4], [1, 0]], jnp.int32)
        expected = (A @ B)[indices[:, 0], indices[:, 1]]
        for actual in (sddmm(A, B, indices), jax.jit(sddmm)(A, B, indices)):
            assert actual.shape == (indices.shape[0],)
            assert actual.dtype == jnp.dtype(dtype)
            np.testing.assert_array_equal(np.asarray(actual, np.float64),
                                          np.asarray(expected, np.float64))


@pytest.mark.parametrize('batch', [0, 1, 3])
def test_empty_pattern(batch):
    A, B = jnp.ones((4, batch)), jnp.ones((batch, 5))
    actual = jax.jit(sddmm)(A, B, jnp.empty((0, 2), jnp.int32))
    assert actual.shape == (0,)
    assert actual.dtype == A.dtype


def test_sampling_all_coordinates_matches_dense_product():
    A = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 4
    B = jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 8
    rows, cols = jnp.meshgrid(jnp.arange(4), jnp.arange(5), indexing='ij')
    indices = jnp.stack([rows.ravel(), cols.ravel()], axis=1)
    np.testing.assert_array_equal(sddmm(A, B, indices), (A @ B).ravel())


@pytest.mark.parametrize('batch', [1, 3])
def test_both_operand_gradients_match_dense_product(batch):
    A = jnp.arange(4 * batch, dtype=jnp.float32).reshape(4, batch) / 4
    B = jnp.arange(batch * 5, dtype=jnp.float32).reshape(batch, 5) / 8
    indices = jnp.array([[3, 4], [0, 1], [3, 4], [1, 0]], jnp.int32)
    loss = lambda a, b: jnp.sum(sddmm(a, b, indices) ** 2)
    reference = lambda a, b: jnp.sum((a @ b)[indices[:, 0], indices[:, 1]] ** 2)
    actual = jax.jit(jax.grad(loss, argnums=(0, 1)))(A, B)
    expected = jax.grad(reference, argnums=(0, 1))(A, B)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('a_spec', [P('s'), P()])
def test_shard_map_varying_operands_preserve_axes_and_precision(a_spec):
    # #245: JAX's sampled sparse primitive types its result as replicated.
    # Varying operands must instead use gathers and a highest-precision dot.
    seen, eqns = _sampled_run_sharded(a_spec)
    assert seen and all(axes == frozenset({'s'}) for axes in seen)
    assert 'bcoo_dot_general_sampled' not in {eqn.primitive.name for eqn in eqns}
    dots = [eqn for eqn in eqns if eqn.primitive.name == 'dot_general']
    assert len(dots) == 1
    highest = jax.lax.Precision.HIGHEST
    assert dots[0].params['precision'] in (highest, (highest, highest))


def test_out_of_range_coordinates_are_zero_with_and_without_sharding():
    A, B, _ = _sampled_shard_operands()
    indices = jnp.array([[0, 0], [2, 1], [1, 4]], jnp.int32)
    unsharded = sddmm(A, B, indices)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('s',))
    f = jax.shard_map(lambda a, b: sddmm(a[0], b, indices)[None], mesh=mesh,
                      in_specs=(P('s'), P()), out_specs=P('s'))
    sharded = jax.jit(f)(A[None], B)[0]
    np.testing.assert_array_equal(np.asarray(unsharded)[1:], [0., 0.])
    np.testing.assert_array_equal(sharded, unsharded)


def test_shard_map_without_vma_keeps_sampled_sparse_primitive():
    seen, eqns = _sampled_run_sharded(P('s'), check_vma=False)
    assert seen and all(axes == frozenset() for axes in seen)
    assert 'bcoo_dot_general_sampled' in {eqn.primitive.name for eqn in eqns}


def test_unsharded_values_match_sampled_sparse_primitive_bit_for_bit():
    rng = np.random.default_rng(245)
    A = jnp.asarray(rng.standard_normal((7, 5)), jnp.float32)
    B = jnp.asarray(rng.standard_normal((5, 6)), jnp.float32)
    indices = jnp.asarray(np.stack([rng.integers(0, 7, 20), rng.integers(0, 6, 20)], 1), jnp.int32)
    # Compile both: an eager dot can select a different GPU algorithm.
    reference = jax.jit(lambda *args: sparse.bcoo_dot_general_sampled(
        *args, dimension_numbers=(((1,), (0,)), ((), ())),
    ))
    np.testing.assert_array_equal(jax.jit(sddmm)(A, B, indices), reference(A, B, indices))
    primitives = {eqn.primitive.name for eqn in _sampled_eqns(sddmm, A, B, indices)}
    assert 'bcoo_dot_general_sampled' in primitives
    assert 'gather' not in primitives
