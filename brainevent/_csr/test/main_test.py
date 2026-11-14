# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_csr/main.py``, grouped by behavior."""

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import jax.numpy as _jnp
import numpy as np
import numpy as _np
import pytest

import brainevent
import brainevent as _be
import brainevent as be
import brainevent._csr.main as csr_main
from brainevent import CSC, CSR, BinaryArray, FixedNumPerPost, FixedNumPerPre, WeightedArray
from brainevent._csr.test._test_util import (
    WEIGHTED_EVENT_DTYPE_NAMES,
    csr_matrix,
    csr_vector,
    event_dtype_context,
    event_mask,
    event_values,
    get_csr,
    matrix_csr,
    small_csr,
    vector_csr,
)
from brainevent._csr.binary.binary import binary_csrmv_p
from brainevent._csr.config import CSRConfig
from brainevent._misc import csc_to_csr_index, csr_to_csc_index
from brainevent._test_util import check_event_op_grads, jax_x64_enabled


# Sparse objects, conversions, arithmetic and JAX transformations
# ---------------------------------------------------------------

platform = jax.default_backend()


BINARY_CSRMV_IMPLEMENTATIONS = tuple(brainevent.binary_csrmv_p.available_backends(platform))


FLOAT_CSRMM_IMPLEMENTATIONS = tuple(brainevent.csrmm_p.available_backends(platform))


def gen_events(shape, prob=0.5, asbool=True):
    events = brainstate.random.random(shape) < prob
    if not asbool:
        events = jnp.asarray(events, dtype=float)
    return brainevent.BinaryArray(events)


def gen_sparse_matrix(shape, prob=0.2):
    """
    Generate a sparse matrix with the given shape and sparsity probability.
    """
    matrix = np.random.rand(*shape)
    matrix = np.where(matrix < prob, matrix, 0.)
    return jnp.asarray(matrix, dtype=float)


def ones_like(x):
    return jax.tree.map(jnp.ones_like, x)


def _make_float_data(homo_w, shape):
    if homo_w:
        return jnp.asarray(1.5, dtype=jnp.float32)
    return braintools.init.Normal(0.0, 1.0)(shape)


def _rand_dense(rng, m, n, p=0.4):
    return _jnp.asarray((rng.random((m, n)) < p) * rng.random((m, n)), _jnp.float32)


def _csr_cases():
    rng = np.random.default_rng(0)
    m, k = 6, 5
    for homo in (True, False):
        dense = (rng.random((m, k)) + 0.5) * (rng.random((m, k)) > 0.5)
        dense = jnp.asarray(dense, dtype=jnp.float32)
        csr = CSR.fromdense(dense)
        if homo:
            csr = CSR((jnp.asarray([1.5], jnp.float32), csr.indices, csr.indptr), shape=csr.shape)
        yield csr


def _mask(x, dtype):
    x = jnp.asarray(x)
    return jnp.asarray(x > 0, dtype=dtype) if x.dtype != jnp.bool_ else jnp.asarray(x, dtype=dtype)


_W_M, _W_N, _W_B = 12, 9, 3


def _weighted_matrix_factory(cls, homo, backend, seed=0):
    """Return ``(make, data)`` where ``make(d)`` builds a ``cls`` matrix with values ``d``."""
    rng = np.random.default_rng(seed)
    dense = (rng.random((_W_M, _W_N)) < 0.4) * rng.standard_normal((_W_M, _W_N))
    base = cls.fromdense(jnp.asarray(dense, dtype=jnp.float32))
    data = jnp.array([1.5], dtype=jnp.float32) if homo else base.data

    def make(d):
        return cls((d, base.indices, base.indptr), shape=base.shape, backend=backend)

    return make, data


def _weighted_tol(dtype_name):
    if dtype_name in ('float16', 'bfloat16'):
        return dict(rtol=2e-2, atol=2e-2)
    return dict(rtol=1e-4, atol=1e-4)


_CLASS_BACKENDS = (None,) + tuple(binary_csrmv_p.available_backends(jax.default_backend()))


_PYTREE_BACKENDS = ('jax_raw', 'numba') if platform == 'cpu' else ('jax_raw', 'cuda_raw')


def _compressed_parts(dense, cls):
    # CSC of ``W`` is, array for array, the CSR of ``W.T``.
    mat = np.asarray(dense if cls is CSR else dense.T)
    rows, cols = np.nonzero(mat)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=mat.shape[0]))])
    return (
        jnp.asarray(mat[rows, cols], jnp.float32),
        jnp.asarray(cols, jnp.int32),
        jnp.asarray(indptr, jnp.int32),
    )


def _pytree_matrix(cls, backend=None, shape=(12, 9), seed=0, prob=0.35):
    rng = np.random.default_rng(seed)
    dense = ((rng.random(shape) < prob) * rng.uniform(0.5, 1.5, shape)).astype(np.float32)
    return cls(_compressed_parts(dense, cls), shape=shape, backend=backend), dense


def _spike_vector(n, seed=1):
    return jnp.asarray(np.random.default_rng(seed).random(n) < 0.5)


def _record_task_capacities(monkeypatch):
    """Spy on primitive calls: record ``(name, task_capacity, indptr)`` and skip execution."""
    from brainevent._op.main import XLACustomKernel
    calls = []
    real = XLACustomKernel.__call__

    def spy(self, *ins, outs, **kwargs):
        if 'task_capacity' in kwargs:
            calls.append((self.name, int(kwargs['task_capacity']), np.asarray(ins[2])))
            return jax.tree.map(lambda o: jnp.zeros(o.shape, o.dtype), outs)
        return real(self, *ins, outs=outs, **kwargs)

    monkeypatch.setattr(XLACustomKernel, '__call__', spy)
    return calls


@pytest.mark.slow
class Test_CSR_BinaryOperator:
    def test_event_homo_bool(self):
        for dat in [1., 2., 3.]:
            mask = (brainstate.random.rand(10, 20) < 0.1).astype(float) * dat
            csr = u.sparse.CSR.fromdense(mask)
            csr = brainevent.CSR((dat, csr.indices, csr.indptr), shape=mask.shape)

            v = brainevent.BinaryArray(brainstate.random.rand(20) < 0.5)
            assert u.math.allclose(
                mask.astype(float) @ v.value.astype(float),
                csr @ v,
            )

            v = brainevent.BinaryArray(brainstate.random.rand(10) < 0.5)
            assert u.math.allclose(
                v.value.astype(float) @ mask.astype(float),
                v @ csr,
            )

            jax.block_until_ready((mask,))

    def test_event_homo_heter(self):
        mat = brainstate.random.rand(10, 20)
        mask = (brainstate.random.rand(10, 20) < 0.1) * mat
        csr = u.sparse.CSR.fromdense(mask)
        csr = brainevent.CSR((csr.data, csr.indices, csr.indptr), shape=mask.shape)

        v = brainevent.BinaryArray(brainstate.random.rand(20) < 0.5)
        assert u.math.allclose(
            mask.astype(float) @ v.value.astype(float),
            csr @ v,
        )

        v = brainevent.BinaryArray(brainstate.random.rand(10) < 0.5)
        assert u.math.allclose(
            v.value.astype(float) @ mask.astype(float),
            v @ csr,
        )

        jax.block_until_ready((mat, mask))

    def test_event_heter_float_as_bool(self):
        mat = brainstate.random.rand(10, 20)
        mask = (mat < 0.1).astype(float) * mat
        csr = u.sparse.CSR.fromdense(mask)
        csr = brainevent.CSR((csr.data, csr.indices, csr.indptr), shape=mask.shape)

        v = brainevent.BinaryArray((brainstate.random.rand(20) < 0.5).astype(float))
        assert u.math.allclose(
            mask.astype(float) @ v.value.astype(float),
            csr @ v,
        )

        v = brainevent.BinaryArray((brainstate.random.rand(10) < 0.5).astype(float))
        assert u.math.allclose(
            v.value.astype(float) @ mask.astype(float),
            v @ csr,
        )

        jax.block_until_ready((mat, mask))


@pytest.mark.slow
class Test_CSR_FloatVectorOperator:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr(self, homo_w):
        m, n = 20, 40
        x = brainstate.random.rand(m)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_float_data(homo_w, indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        y = x @ csr
        y_ref = vector_csr(x, csr.data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_vector(self, homo_w):
        m, n = 20, 40
        x = brainstate.random.rand(n)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_float_data(homo_w, indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        y = csr @ x
        y_ref = csr_vector(x, csr.data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_vector_csr_vmap_vector(self, homo_w):
        n_batch, m, n = 10, 20, 40
        xs = brainstate.random.rand(n_batch, m)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_float_data(homo_w, indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        y = brainstate.transform.vmap2(lambda x: x @ csr)(xs)
        y_ref = brainstate.transform.vmap2(lambda x: vector_csr(x, csr.data, indices, indptr, (m, n)))(xs)
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((xs, indptr, indices, data, y, y_ref))

    # Covering set: every value of homo_w/replace/transpose appears at least once (4 rows
    # instead of the full 2x2x2 = 8 product).
    @pytest.mark.parametrize('homo_w,replace,transpose', [
        (True, True, True),
        (False, False, False),
        (True, False, True),
        (False, True, False),
    ])
    def test_vjp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in) if transpose else brainstate.random.rand(n_out)

        indptr, indices = get_csr(n_in, n_out, 0.2, replace=replace)
        w = _make_float_data(homo_w, indices.shape)
        csr = brainevent.CSR((w, indices, indptr), shape=shape)

        def f_brainevent(x_arg, w_arg):
            if transpose:
                return (x_arg @ csr.with_data(w_arg)).sum()
            return (csr.with_data(w_arg) @ x_arg).sum()

        r = jax.grad(f_brainevent, argnums=(0, 1))(x, w)

        def f_ref(x_arg, w_arg):
            if transpose:
                return vector_csr(x_arg, w_arg, indices, indptr, shape=shape).sum()
            return csr_vector(x_arg, w_arg, indices, indptr, shape=shape).sum()

        r_ref = jax.grad(f_ref, argnums=(0, 1))(x, w)
        assert jnp.allclose(r[0], r_ref[0], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(r[1], r_ref[1], rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, r, r_ref))

    @pytest.mark.parametrize('homo_w,replace,transpose', [
        (True, True, True),
        (False, False, False),
        (True, False, True),
        (False, True, False),
    ])
    def test_jvp(self, homo_w, replace, transpose):
        n_in = 20
        n_out = 30
        shape = (n_in, n_out)
        x = brainstate.random.rand(n_in if transpose else n_out)

        indptr, indices = get_csr(n_in, n_out, 0.1, replace=replace)
        w = _make_float_data(homo_w, indices.shape)
        csr = brainevent.CSR((w, indices, indptr), shape=shape)

        def f_brainevent(x_arg, w_arg):
            if transpose:
                return x_arg @ csr.with_data(w_arg)
            return csr.with_data(w_arg) @ x_arg

        o1, r1 = jax.jvp(f_brainevent, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))

        def f_ref(x_arg, w_arg):
            if transpose:
                return vector_csr(x_arg, w_arg, indices, indptr, shape=shape)
            return csr_vector(x_arg, w_arg, indices, indptr, shape=shape)

        o2, r2 = jax.jvp(f_ref, (x, w), (jnp.ones_like(x), jnp.ones_like(w)))
        assert jnp.allclose(r1, r2, rtol=1e-3, atol=1e-3)
        assert jnp.allclose(o1, o2, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, w, o1, r1, o2, r2))


@pytest.mark.slow
@pytest.mark.skipif(
    not FLOAT_CSRMM_IMPLEMENTATIONS,
    reason=f'No csrmm implementation on platform={platform}',
)
class Test_CSR_FloatMatrixOperator:
    @pytest.mark.parametrize('homo_w', [True, False])
    def test_matrix_csr(self, homo_w):
        k, m, n = 10, 20, 40
        x = brainstate.random.rand(k, m)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_float_data(homo_w, indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        y = x @ csr
        y_ref = matrix_csr(x, csr.data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))

    @pytest.mark.parametrize('homo_w', [True, False])
    def test_csr_matrix(self, homo_w):
        m, n, k = 20, 40, 10
        x = brainstate.random.rand(n, k)
        indptr, indices = get_csr(m, n, 0.1)
        data = _make_float_data(homo_w, indices.shape)
        csr = brainevent.CSR((data, indices, indptr), shape=(m, n))

        y = csr @ x
        y_ref = csr_matrix(x, csr.data, indices, indptr, (m, n))
        assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)

        jax.block_until_ready((x, indptr, indices, data, y, y_ref))


@pytest.mark.slow
class Test_CSC_CSR_Conversion:
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_todense(self, shape, transpose):
        matrix = gen_sparse_matrix(shape)
        if transpose:
            matrix = matrix.T
        csr = brainevent.CSR.fromdense(matrix)
        csc = csr.T

        out1 = csr.todense()
        out2 = csc.todense().T
        out3 = csr.T.todense().T
        out4 = csc.T.todense()
        assert jnp.allclose(out1, out2)
        assert jnp.allclose(out1, out3)
        assert jnp.allclose(out1, out4)

        jax.block_until_ready((matrix, out1, out2, out3, out4))

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_csr_vec(self, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csc = csr.T

        vector = jnp.asarray(np.random.rand(shape[1]))

        out1 = csr @ vector
        out2 = vector @ csc
        assert jnp.allclose(out1, out2)

        jax.block_until_ready((matrix, vector, out1, out2))

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vec_csr(self, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csc = csr.T

        vector = jnp.asarray(np.random.rand(shape[0]))

        out1 = vector @ csr
        out2 = csc @ vector
        assert jnp.allclose(out1, out2)

        jax.block_until_ready((matrix, vector, out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_csr_mat(self, k, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csc = csr.T

        matrix = jnp.asarray(np.random.rand(shape[1], k))

        out1 = jax.jit(lambda: csr @ matrix)()
        out2 = jax.jit(lambda: (matrix.T @ csc).T)()
        assert jnp.allclose(out1, out2)

        jax.block_until_ready((matrix, out1, out2))

    # TODO: GPU pallas bug
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_mat_csr(self, k, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csc = csr.T

        matrix = jnp.asarray(np.random.rand(k, shape[0]))

        out1 = jax.jit(lambda: matrix @ csr)()
        out2 = jax.jit(lambda: (csc @ matrix.T).T)()
        assert jnp.allclose(out1, out2, atol=1e-4, rtol=1e-4)

        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_csr_vec_event(self, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csc = csr.T

        vector = gen_events(shape[1])

        out1 = jax.jit(lambda: csr @ vector)()
        out2 = jax.jit(lambda: vector @ csc)()
        assert jnp.allclose(out1, out2)

        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vec_csr_event(self, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csc = csr.T

        vector = gen_events(shape[0])

        out1 = jax.jit(lambda: vector @ csr)()
        out2 = jax.jit(lambda: csc @ vector)()
        assert jnp.allclose(out1, out2)

        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_csr_mat_event(self, k, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csc = csr.T

        matrix = gen_events([shape[1], k])

        out1 = jax.jit(lambda: csr @ matrix)()
        out2 = jax.jit(lambda: (matrix.value.T @ csc).T)()
        assert jnp.allclose(out1, out2)

        jax.block_until_ready((matrix, out1, out2))

    # TODO: GPU test error: CUDA_ERROR_ILLEGAL_ADDRESS
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_mat_csr_event(self, k, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csc = csr.T

        matrix = gen_events([k, shape[0]])

        out1 = jax.jit(lambda: matrix @ csr)()
        out2 = jax.jit(lambda: (csc @ matrix.value.T).T)()
        assert jnp.allclose(out1, out2, atol=1e-4, rtol=1e-4)

        jax.block_until_ready((matrix, out1, out2))


@pytest.mark.slow
class Test_CSR:
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_todense(self, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        dense = csr.todense()
        assert jnp.allclose(matrix, dense)

        jax.block_until_ready((matrix, dense))

    # TODO: GPU pallas error: CUDA_ERROR_ILLEGAL_ADDRESS
    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_mat_vjp_heter_weight(self, shape, k, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        if transpose:
            xs = brainstate.random.randn(shape[0], k)
        else:
            xs = brainstate.random.randn(shape[1], k)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, (g00, g01) = jax.jit(lambda: jax.value_and_grad(f_brainevent, argnums=(0, 1))(xs, csr.data))()
        r2, (g10, g11) = jax.jit(lambda: jax.value_and_grad(f_dense, argnums=(0, 1))(xs, csr.data))()

        rtol = 1e-1 if brainstate.environ.get_platform() == 'gpu' else 1e-4
        assert jnp.allclose(r1, r2, rtol=rtol, atol=rtol)
        assert jnp.allclose(g00, g10, rtol=rtol, atol=rtol)
        assert jnp.allclose(g01, g11, rtol=rtol, atol=rtol)

        jax.block_until_ready((xs, r1, r2, g00, g01, g10, g11))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_mat_vjp_homo_weight(self, shape, k, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csr = brainevent.CSR((1.5, csr.indices, csr.indptr), shape=shape)

        if transpose:
            xs = brainstate.random.randn(shape[0], k)
        else:
            xs = brainstate.random.randn(shape[1], k)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, (g00, g01) = jax.jit(lambda: jax.value_and_grad(f_brainevent, argnums=(0, 1))(xs, csr.data))()
        r2, (g10, g11) = jax.jit(lambda: jax.value_and_grad(f_dense, argnums=(0, 1))(xs, csr.data))()

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g00, g10, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g01, g11, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((xs, r1, r2, g00, g01, g10, g11))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_vec_vjp_heter_weight(self, shape, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        if transpose:
            xs = brainstate.random.randn(shape[0])
        else:
            xs = brainstate.random.randn(shape[1])

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, (g00, g01) = jax.jit(lambda: jax.value_and_grad(f_brainevent, argnums=(0, 1))(xs, csr.data))()
        r2, (g10, g11) = jax.jit(lambda: jax.value_and_grad(f_dense, argnums=(0, 1))(xs, csr.data))()

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g00, g10, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g01, g11, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((xs, r1, r2, g00, g01, g10, g11))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_vec_vjp_homo_weight(self, shape, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csr = brainevent.CSR((1.5, csr.indices, csr.indptr), shape=shape)

        if transpose:
            xs = brainstate.random.randn(shape[0])
        else:
            xs = brainstate.random.randn(shape[1])

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, (g00, g01) = jax.jit(lambda: jax.value_and_grad(f_brainevent, argnums=(0, 1))(xs, csr.data))()
        r2, (g10, g11) = jax.jit(lambda: jax.value_and_grad(f_dense, argnums=(0, 1))(xs, csr.data))()

        tol = 1e-1 if brainstate.environ.get_platform() else 1e-4
        assert jnp.allclose(r1, r2, rtol=tol, atol=tol)
        assert jnp.allclose(g00, g10, rtol=tol, atol=tol)
        assert jnp.allclose(g01, g11, rtol=tol, atol=tol)

        jax.block_until_ready((xs, r1, r2, g00, g01, g10, g11))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_mat_jvp_heter_weight(self, shape, k, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        if transpose:
            xs = brainstate.random.randn(shape[0], k)
        else:
            xs = brainstate.random.randn(shape[1], k)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, g1 = jax.jit(lambda: jax.jvp(f_brainevent,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()
        r2, g2 = jax.jit(lambda: jax.jvp(f_dense,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()

        tol = 1e-1 if brainstate.environ.get_platform() else 1e-4
        assert jnp.allclose(r1, r2, rtol=tol, atol=tol)
        assert jnp.allclose(g1, g2, rtol=tol, atol=tol)

        jax.block_until_ready((xs, r1, g1, r2, g2))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_mat_jvp_homo_weight(self, shape, k, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csr = brainevent.CSR((1.5, csr.indices, csr.indptr), shape=shape)

        if transpose:
            xs = brainstate.random.randn(shape[0], k)
        else:
            xs = brainstate.random.randn(shape[1], k)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, g1 = jax.jit(lambda: jax.jvp(f_brainevent,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()
        r2, g2 = jax.jit(lambda: jax.jvp(f_dense,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()

        tol = 1e-1 if brainstate.environ.get_platform() else 1e-4
        assert jnp.allclose(r1, r2, rtol=tol, atol=tol)
        assert jnp.allclose(g1, g2, rtol=tol, atol=tol)

        jax.block_until_ready((xs, r1, g1, r2, g2))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_vec_jvp_heter_weight(self, shape, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        if transpose:
            xs = brainstate.random.randn(shape[0])
        else:
            xs = brainstate.random.randn(shape[1])

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, g1 = jax.jit(lambda: jax.jvp(f_brainevent,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()
        r2, g2 = jax.jit(lambda: jax.jvp(f_dense,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g1, g2, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((xs, r1, g1, r2, g2))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_vec_jvp_homo_weight(self, shape, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csr = brainevent.CSR((1.5, csr.indices, csr.indptr), shape=shape)

        if transpose:
            xs = brainstate.random.randn(shape[0])
        else:
            xs = brainstate.random.randn(shape[1])

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, g1 = jax.jit(lambda: jax.jvp(f_brainevent,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()
        r2, g2 = jax.jit(lambda: jax.jvp(f_dense,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()

        print(r1, r2)
        print(g1, g2)
        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g1, g2, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((xs, r1, g1, r2, g2))


@pytest.mark.slow
class Test_CSR_Event:

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_mat_vjp_heter_weight(self, shape, k, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)

        if transpose:
            xs = gen_events([shape[0], k], asbool=False)
        else:
            xs = gen_events([shape[1], k], asbool=False)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, (g00, g01) = jax.jit(lambda: jax.value_and_grad(f_brainevent, argnums=(0, 1))(xs, csr.data))()
        r2, (g10, g11) = jax.jit(lambda: jax.value_and_grad(f_dense, argnums=(0, 1))(xs, csr.data))()

        assert isinstance(g00, brainevent.BinaryArray)
        assert isinstance(g10, brainevent.BinaryArray)

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g00.value, g10.value, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g01, g11, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((r1, r2, g00, g01, g10, g11))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_mat_vjp_homo_weight(self, shape, k, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csr = brainevent.CSR((1.5, csr.indices, csr.indptr), shape=shape)

        if transpose:
            xs = gen_events([shape[0], k], asbool=False)
        else:
            xs = gen_events([shape[1], k], asbool=False)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, (g00, g01) = jax.jit(lambda: jax.value_and_grad(f_brainevent, argnums=(0, 1))(xs, csr.data))()
        r2, (g10, g11) = jax.jit(lambda: jax.value_and_grad(f_dense, argnums=(0, 1))(xs, csr.data))()

        assert isinstance(g00, brainevent.BinaryArray)
        assert isinstance(g10, brainevent.BinaryArray)

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g00.value, g10.value, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g01, g11, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((r1, r2, g00, g01, g10, g11))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_vec_vjp_heter_weight(self, shape, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)

        if transpose:
            xs = gen_events([shape[0], ], asbool=False)
        else:
            xs = gen_events([shape[1], ], asbool=False)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, (g00, g01) = jax.jit(lambda: jax.value_and_grad(f_brainevent, argnums=(0, 1))(xs, csr.data))()
        r2, (g10, g11) = jax.jit(lambda: jax.value_and_grad(f_dense, argnums=(0, 1))(xs, csr.data))()

        assert isinstance(g00, brainevent.BinaryArray)
        assert isinstance(g10, brainevent.BinaryArray)

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g00.value, g10.value, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g01, g11, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((r1, r2, g00, g01, g10, g11))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_vec_vjp_homo_weight(self, shape, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csr = brainevent.CSR((1.5, csr.indices, csr.indptr), shape=shape)

        if transpose:
            xs = gen_events([shape[0], ], asbool=False)
        else:
            xs = gen_events([shape[1], ], asbool=False)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, (g00, g01) = jax.jit(lambda: jax.value_and_grad(f_brainevent, argnums=(0, 1))(xs, csr.data))()
        r2, (g10, g11) = jax.jit(lambda: jax.value_and_grad(f_dense, argnums=(0, 1))(xs, csr.data))()

        assert isinstance(g00, brainevent.BinaryArray)
        assert isinstance(g10, brainevent.BinaryArray)

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g00.value, g10.value, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g01, g11, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((r1, r2, g00, g01, g10, g11))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_mat_jvp_heter_weight(self, shape, k, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        if transpose:
            xs = gen_events([shape[0], k], asbool=False)
        else:
            xs = gen_events([shape[1], k], asbool=False)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, g1 = jax.jit(lambda: jax.jvp(f_brainevent,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()
        r2, g2 = jax.jit(lambda: jax.jvp(f_dense,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g1, g2, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((r1, g1, r2, g2))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_mat_jvp_homo_weight(self, shape, k, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csr = brainevent.CSR((1.5, csr.indices, csr.indptr), shape=shape)

        if transpose:
            xs = gen_events([shape[0], k], asbool=False)
        else:
            xs = gen_events([shape[1], k], asbool=False)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, g1 = jax.jit(lambda: jax.jvp(f_brainevent,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()
        r2, g2 = jax.jit(lambda: jax.jvp(f_dense,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g1, g2, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((r1, g1, r2, g2))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_vec_jvp_heter_weight(self, shape, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)

        if transpose:
            xs = gen_events([shape[0], ], asbool=False)
        else:
            xs = gen_events([shape[1], ], asbool=False)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, g1 = jax.jit(lambda: jax.jvp(f_brainevent,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()
        r2, g2 = jax.jit(lambda: jax.jvp(f_dense,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g1, g2, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((r1, g1, r2, g2))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    def test_csr_vec_jvp_homo_weight(self, shape, transpose):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        csr = brainevent.CSR((1.5, csr.indices, csr.indptr), shape=shape)

        if transpose:
            xs = gen_events([shape[0], ], asbool=False)
        else:
            xs = gen_events([shape[1], ], asbool=False)

        def f_brainevent(x, w):
            if transpose:
                return (csr.with_data(w).T @ x).sum()
            else:
                return (csr.with_data(w) @ x).sum()

        def f_dense(x, w):
            if transpose:
                return (csr.with_data(w).T.todense() @ x).sum()
            else:
                return (csr.with_data(w).todense() @ x).sum()

        r1, g1 = jax.jit(lambda: jax.jvp(f_brainevent,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()
        r2, g2 = jax.jit(lambda: jax.jvp(f_dense,
                                         (xs, csr.data),
                                         (ones_like(xs), ones_like(csr.data))))()

        assert jnp.allclose(r1, r2, rtol=1e-2, atol=1e-2)
        assert jnp.allclose(g1, g2, rtol=1e-2, atol=1e-2)

        jax.block_until_ready((r1, g1, r2, g2))


@pytest.mark.slow
class Test_CSC:
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_todense(self, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix).T
        dense = csr.todense()
        assert jnp.allclose(matrix.T, dense)

        jax.block_until_ready((dense,))


class Test_compressed_elementwise_ops:
    dense = np.array([[1., 0., 2.], [0., 3., 0.]], dtype=np.float32)

    @pytest.mark.parametrize('cls', [CSR, CSC])
    def test_mul_matrix_with_shared_structure(self, cls):
        m = cls.fromdense(jnp.asarray(self.dense))
        m2 = m.with_data(m.data * 2)
        for out, ref in [(m * m2, self.dense * self.dense * 2),
                         (m2 / m, np.where(self.dense != 0, 2., 0.))]:
            assert isinstance(out, cls)
            np.testing.assert_allclose(np.asarray(out.todense()), ref)

    @pytest.mark.parametrize('cls', [CSR, CSC])
    @pytest.mark.parametrize('scalar', [1, 2.5, np.float32(1.5)])
    def test_add_sub_python_scalar(self, cls, scalar):
        m = cls.fromdense(jnp.asarray(self.dense))
        np.testing.assert_allclose(np.asarray(m + scalar), self.dense + scalar)
        np.testing.assert_allclose(np.asarray(scalar + m), scalar + self.dense)
        np.testing.assert_allclose(np.asarray(m - scalar), self.dense - scalar)
        np.testing.assert_allclose(np.asarray(scalar - m), scalar - self.dense)

    @pytest.mark.parametrize('cls', [CSR, CSC])
    def test_add_incompatible_shape_raises(self, cls):
        m = cls.fromdense(jnp.asarray(self.dense))
        with pytest.raises(ValueError):
            m + jnp.ones((3, 3))

    @pytest.mark.parametrize('cls', [CSR, CSC])
    @pytest.mark.parametrize('n_data', [0, 2, 4])
    def test_constructor_rejects_data_size_mismatch(self, cls, n_data):
        # nse == 3: data must hold 1 (homogeneous) or 3 values.
        indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        shape = (2, 3) if cls is CSR else (3, 2)
        with pytest.raises(ValueError, match='data'):
            cls((jnp.ones(n_data), indices, indptr), shape=shape)

    @pytest.mark.parametrize('cls', [CSR, CSC])
    @pytest.mark.parametrize('n_data', [1, 3])
    def test_constructor_accepts_homogeneous_and_full_data(self, cls, n_data):
        indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        shape = (2, 3) if cls is CSR else (3, 2)
        m = cls((jnp.ones(n_data), indices, indptr), shape=shape)
        assert m.nse == 3

    @pytest.mark.parametrize('cls', [CSR, CSC])
    def test_constructor_accepts_scalar_homogeneous_data(self, cls):
        indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        shape = (2, 3) if cls is CSR else (3, 2)
        scalar = cls((jnp.asarray(2.0), indices, indptr), shape=shape)
        homo = cls((jnp.full((1,), 2.0), indices, indptr), shape=shape)
        assert jnp.array_equal(scalar.todense(), homo.todense())


@pytest.mark.slow
def test_csr_build_weight_indices_and_cache():
    rng = _np.random.default_rng(0)
    dense = (rng.random((4, 5)) > 0.5) * rng.random((4, 5))
    csr = _be.CSR.fromdense(_jnp.asarray(dense, _jnp.float32))
    csr2 = csr.build_weight_indices()
    assert 'csc' in csr2.buffers and csr2.buffers['csc'] is not None
    cscp, csci, perm = csr2._weight_indices()
    assert perm.shape == csr.data.shape
    # apply preserves structure -> cache must survive (same perm values)
    applied = csr2.apply(lambda x: x * 2)
    assert _jnp.array_equal(applied._weight_indices()[2], perm)


@pytest.mark.slow
def test_csr_lazy_weight_indices():
    rng = _np.random.default_rng(9)
    dense = (rng.random((3, 6)) > 0.5) * rng.random((3, 6))
    csr = _be.CSR.fromdense(_jnp.asarray(dense, _jnp.float32))
    assert csr.buffers.get('csc') is None         # not built yet
    cscp, csci, perm = csr._weight_indices()       # lazily builds
    assert perm.shape == csr.data.shape


@pytest.mark.slow
def test_csr_eager_precompute():
    dense = _jnp.asarray((_np.random.default_rng(1).random((3, 4)) > 0.5) * 1.0, _jnp.float32)
    csr = _be.CSR.fromdense(dense, precompute_weight_indices=True)
    assert csr.buffers.get('csc') is not None


@pytest.mark.slow
@pytest.mark.usefixtures('_backend_restore_global_backends')
@pytest.mark.parametrize('cuda', [False, True], ids=['no_cuda', 'cuda'])
def test_csr_at_event_default_route_builds_mirror_only_with_cuda(monkeypatch, cuda):
    # With no backend chosen, the unfavorable product scatters over the mirror
    # only when the process has a CUDA device; otherwise it gathers directly.
    monkeypatch.setattr(csr_main, '_cuda_available', lambda: cuda)
    rng = _np.random.default_rng(3)
    m, k = 5, 7
    csr = _be.CSR.fromdense(_rand_dense(rng, m, k))
    ev = _jnp.asarray(rng.random(k) > 0.5)
    got = csr @ _be.BinaryArray(ev)
    ref = csr.todense() @ ev.astype(_jnp.float32)
    assert got.shape == (m,)
    assert _jnp.allclose(got, ref, atol=1e-5)
    assert (csr.buffers.get('csc') is not None) == cuda


@pytest.mark.slow
def test_event_at_csr_favorable_matches_dense():
    # event @ CSR is favorable -> direct event-driven scatter, unchanged.
    rng = _np.random.default_rng(4)
    m, k = 5, 7
    csr = _be.CSR.fromdense(_rand_dense(rng, m, k))
    ev = _jnp.asarray(rng.random(m) > 0.5)
    got = _be.BinaryArray(ev) @ csr
    ref = ev.astype(_jnp.float32) @ csr.todense()
    assert got.shape == (k,)
    assert _jnp.allclose(got, ref, atol=1e-5)


@pytest.mark.slow
@pytest.mark.usefixtures('_backend_restore_global_backends')
@pytest.mark.parametrize('cuda', [False, True], ids=['no_cuda', 'cuda'])
def test_event_at_csc_default_route_builds_mirror_only_with_cuda(monkeypatch, cuda):
    # With no backend chosen, the unfavorable product scatters over the mirror
    # only when the process has a CUDA device; otherwise it gathers directly.
    monkeypatch.setattr(csr_main, '_cuda_available', lambda: cuda)
    rng = _np.random.default_rng(5)
    m, n = 6, 4
    csc = _be.CSC.fromdense(_rand_dense(rng, m, n))
    ev = _jnp.asarray(rng.random(m) > 0.5)
    got = _be.BinaryArray(ev) @ csc
    ref = ev.astype(_jnp.float32) @ csc.todense()
    assert got.shape == (n,)
    assert _jnp.allclose(got, ref, atol=1e-5)
    assert (csc.buffers.get('csr') is not None) == cuda


@pytest.mark.slow
def test_csc_at_event_favorable_matches_dense():
    # CSC @ event is favorable -> direct event-driven scatter, unchanged.
    rng = _np.random.default_rng(6)
    m, n = 6, 4
    csc = _be.CSC.fromdense(_rand_dense(rng, m, n))
    ev = _jnp.asarray(rng.random(n) > 0.5)
    got = csc @ _be.BinaryArray(ev)
    ref = csc.todense() @ ev.astype(_jnp.float32)
    assert got.shape == (m,)
    assert _jnp.allclose(got, ref, atol=1e-5)


@pytest.mark.slow
def test_csr_at_event_jit_matches_dense():
    rng = _np.random.default_rng(7)
    m, k = 5, 7
    csr = _be.CSR.fromdense(_rand_dense(rng, m, k))
    ev = _jnp.asarray(rng.random(k) > 0.5)
    f = jax.jit(lambda w, e: csr.with_data(w) @ _be.BinaryArray(e))
    got = f(csr.data, ev)
    ref = csr.todense() @ ev.astype(_jnp.float32)
    assert _jnp.allclose(got, ref, atol=1e-5)


@pytest.mark.slow
def test_csr_transpose_hands_off_weight_indices():
    # The CSC-like view of W equals the CSR-like view of W.T, so the cached
    # perm must transfer across transpose with identical values.
    rng = _np.random.default_rng(8)
    m, k = 4, 6
    csr = _be.CSR.fromdense(_rand_dense(rng, m, k)).build_weight_indices()
    perm_csr = csr._weight_indices()[2]
    csc = csr.transpose()
    assert csc.buffers.get('csr') is not None          # re-keyed on transpose
    perm_csc = csc._weight_indices()[2]
    assert _jnp.array_equal(perm_csr, perm_csc)
    # and the transposed matrix multiplies correctly
    ev = _jnp.asarray(rng.random(k) > 0.5)
    got = _be.BinaryArray(ev) @ csc                     # event @ (W.T) , len m
    ref = ev.astype(_jnp.float32) @ csc.todense()
    assert _jnp.allclose(got, ref, atol=1e-5)


@pytest.mark.slow
def test_csc_transpose_hands_off_weight_indices():
    rng = _np.random.default_rng(10)
    m, n = 5, 3
    csc = _be.CSC.fromdense(_rand_dense(rng, m, n)).build_weight_indices()
    perm_csc = csc._weight_indices()[2]
    csr = csc.transpose()
    assert csr.buffers.get('csc') is not None
    perm_csr = csr._weight_indices()[2]
    assert _jnp.array_equal(perm_csc, perm_csr)


@pytest.mark.slow
def test_csc_eager_precompute():
    rng = _np.random.default_rng(12)
    csc = _be.CSC.fromdense(_rand_dense(rng, 4, 5), precompute_weight_indices=True)
    assert csc.buffers.get('csr') is not None


@pytest.mark.slow
def test_csr_matmat_golden():
    rng = np.random.default_rng(7)
    for csr in _csr_cases():
        dense = jnp.asarray(csr.todense(), dtype=jnp.float32)
        m, k = csr.shape
        for ev in (jnp.bool_, jnp.float32):
            for n in (1, 3, 8):
                right = jnp.asarray(rng.random((k, n)) > 0.5, dtype=ev)   # CSR @ M (unfavorable)
                got_r = csr @ BinaryArray(right)
                ref_r = dense @ _mask(right, jnp.float32)
                assert jnp.allclose(got_r, ref_r, atol=1e-5), ('CSR@M', str(ev), n)

                left = jnp.asarray(rng.random((n, m)) > 0.5, dtype=ev)    # M @ CSR (favorable)
                got_l = BinaryArray(left) @ csr
                ref_l = _mask(left, jnp.float32) @ dense
                assert jnp.allclose(got_l, ref_l, atol=1e-5), ('M@CSR', str(ev), n)


@pytest.mark.slow
def test_csc_matmat_golden():
    rng = np.random.default_rng(11)
    for csr in _csr_cases():
        csc = csr.T                          # CSC view; dense transposes too
        dense = jnp.asarray(csc.todense(), dtype=jnp.float32)
        p, q = csc.shape
        for ev in (jnp.bool_, jnp.float32):
            for n in (1, 3, 8):
                right = jnp.asarray(rng.random((q, n)) > 0.5, dtype=ev)   # CSC @ M (favorable)
                got_r = csc @ BinaryArray(right)
                ref_r = dense @ _mask(right, jnp.float32)
                assert jnp.allclose(got_r, ref_r, atol=1e-5), ('CSC@M', str(ev), n)

                left = jnp.asarray(rng.random((n, p)) > 0.5, dtype=ev)    # M @ CSC (unfavorable)
                got_l = BinaryArray(left) @ csc
                ref_l = _mask(left, jnp.float32) @ dense
                assert jnp.allclose(got_l, ref_l, atol=1e-5), ('M@CSC', str(ev), n)


@pytest.mark.slow
def test_csr_matmat_unfavorable_jax_raw_route_skips_weight_indices():
    # jax_raw BinaryArray routing uses the direct CSR structure.
    # Explicit jax_raw overrides any global CUDA choice and does not build 'csc'.
    rng = np.random.default_rng(2)
    csr = next(_csr_cases())
    csr = CSR((csr.data, csr.indices, csr.indptr), shape=csr.shape, backend="jax_raw")
    k = csr.shape[1]
    right = jnp.asarray(rng.random((k, 4)) > 0.5, dtype=jnp.bool_)
    assert csr.buffers.get('csc') is None
    _ = csr @ BinaryArray(right)
    assert csr.buffers.get('csc') is None


@pytest.mark.slow
def test_csc_matmat_unfavorable_jax_raw_route_skips_weight_indices():
    # jax_raw BinaryArray routing uses the direct CSC structure.
    # Explicit jax_raw overrides any global CUDA choice and does not build 'csr'.
    rng = np.random.default_rng(3)
    csc = next(_csr_cases()).T
    csc = CSC((csc.data, csc.indices, csc.indptr), shape=csc.shape, backend="jax_raw")
    p = csc.shape[0]
    left = jnp.asarray(rng.random((4, p)) > 0.5, dtype=jnp.bool_)
    assert csc.buffers.get('csr') is None
    _ = BinaryArray(left) @ csc
    assert csc.buffers.get('csr') is None


@pytest.mark.slow
def test_csr_matmat_units_and_jit():
    rng = np.random.default_rng(9)
    csr = next(_csr_cases())
    m, k = csr.shape
    csr_u = CSR((csr.data * u.mV, csr.indices, csr.indptr), shape=csr.shape)
    M = jnp.asarray(rng.random((k, 4)) > 0.5, dtype=jnp.bool_)
    got = csr_u @ BinaryArray(M)
    assert u.get_unit(got) == u.mV
    ref = (jnp.asarray(csr.todense(), jnp.float32) @ jnp.asarray(M, jnp.float32))
    assert jnp.allclose(u.get_mantissa(got), ref, atol=1e-5)

    f = jax.jit(lambda d: CSR((d, csr.indices, csr.indptr), shape=csr.shape) @ BinaryArray(M))
    got_jit = f(csr.data)
    assert jnp.allclose(got_jit, ref, atol=1e-5)


@pytest.mark.parametrize("cls,shape", [(CSR, (2, 3)), (CSC, (3, 2))])
def test_constructor_indices_int32_indptr_int32(cls, shape):
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    m = cls((data, indices, indptr), shape=shape)
    assert m.indices.dtype == jnp.int32
    assert m.indptr.dtype == jnp.int32


@pytest.mark.parametrize("cls,shape", [(CSR, (2, 3)), (CSC, (3, 2))])
def test_constructor_coerces_int64_indices_to_int32(cls, shape):
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = np.array([0, 2, 1], dtype=np.int64)
    indptr = np.array([0, 2, 3], dtype=np.int64)
    m = cls((data, indices, indptr), shape=shape)
    assert m.indices.dtype == jnp.int32
    # Small nnz -> indptr resolves to int32 regardless of input width.
    assert m.indptr.dtype == jnp.int32


def test_constructor_explicit_int64_indptr_gated_when_x64_off():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    with pytest.raises(ValueError, match="requires an int64 array"):
        CSR((data, indices, indptr), shape=(2, 3), indptr_dtype=np.int64)
    # The failed construction must not have toggled the global config.
    assert jax.config.jax_enable_x64 is False


def test_constructor_explicit_int64_indptr_ok_when_x64_on():
    with jax_x64_enabled():
        data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        m = CSR((data, indices, indptr), shape=(2, 3), indptr_dtype=np.int64)
        assert m.indices.dtype == jnp.int32
        assert m.indptr.dtype == jnp.int64


@pytest.mark.parametrize("cls,shape", [(CSR, (2, 3)), (CSC, (3, 2))])
def test_structure_rebuild_preserves_explicit_int64_indptr(cls, shape):
    with jax_x64_enabled():
        data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 3], dtype=jnp.int64)
        m = cls((data, indices, indptr), shape=shape, indptr_dtype=np.int64)
        leaves, treedef = jax.tree_util.tree_flatten(m)

        rebuilt = (
            m.with_data(m.data),
            m.apply(lambda values: values),
            m.transpose(),
            jax.tree_util.tree_unflatten(treedef, leaves),
        )
        for result in rebuilt:
            assert result.indptr.dtype == jnp.int64


def test_constructor_rejects_non_monotonic_indptr():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 3, 2], dtype=jnp.int32)  # decreasing
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        CSR((data, indices, indptr), shape=(2, 3))


def test_constructor_rejects_indptr_tail_mismatch():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 2], dtype=jnp.int32)  # tail != nse (3)
    with pytest.raises(ValueError, match="must equal the number of"):
        CSR((data, indices, indptr), shape=(2, 3))


def test_constructor_rejects_wrong_indptr_length():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 3], dtype=jnp.int32)  # length should be n_rows + 1 = 3
    with pytest.raises(ValueError, match="indptr length"):
        CSR((data, indices, indptr), shape=(2, 3))


def test_constructor_rejects_nonzero_indptr_head():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([1, 2, 3], dtype=jnp.int32)  # head != 0
    with pytest.raises(ValueError, match=r"indptr\[0\] must be 0"):
        CSR((data, indices, indptr), shape=(2, 3))


def test_with_data_preserves_structure_dtype():
    m = small_csr()
    m2 = m.with_data(jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32))
    assert m2.indices.dtype == jnp.int32
    assert m2.indptr.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(m2.indices), np.asarray(m.indices))


def test_transpose_preserves_structure_dtype():
    m = small_csr()
    mt = m.transpose()
    assert isinstance(mt, CSC)
    assert mt.indices.dtype == jnp.int32
    assert mt.indptr.dtype == jnp.int32


def test_with_data_under_jit_no_host_readback():
    # A structure-preserving reconstruction under jit must not try to read the
    # (traced) data on host; indices/indptr remain concrete int32.
    m = small_csr()

    @jax.jit
    def scale(data):
        return m.with_data(data * 2.0).data.sum()

    val = scale(m.data)
    assert float(val) == pytest.approx(2.0 * float(m.data.sum()))


def test_csr_cuda_sources_do_not_cast_indptr_to_int32():
    # Package-wide invariant over every CSR CUDA source: an int64 indptr must
    # survive into the kernel signature rather than being narrowed at the
    # boundary, which would silently corrupt offsets past int32 range.
    csr_dir = Path(__file__).parents[1]
    for path in csr_dir.rglob('*.cu'):
        if 'dev' in path.relative_to(csr_dir).parts:
            continue
        text = path.read_text()
        assert 'static_cast<const int32_t*>(indptr.data_ptr())' not in text, path.name
        assert 'const int32_t*  __restrict__ indptr' not in text, path.name
        assert 'const int32_t*   __restrict__ indptr' not in text, path.name


@pytest.mark.parametrize("cls,shape", [(CSR, (2, 3)), (CSC, (3, 2))])
def test_compressed_apply_rejects_value_shape_changes(cls, shape):
    """Keep apply from changing the physical sparse-value representation."""
    matrix = cls(
        (
            jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
            jnp.asarray([0, 2, 0, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 4], dtype=jnp.int32),
        ),
        shape=shape,
    )

    with pytest.raises(ValueError, match="apply.*shape"):
        matrix.apply(lambda values: values.reshape((2, 2)))


@pytest.mark.parametrize("cls,shape", [(CSR, (2, 3)), (CSC, (3, 2))])
@pytest.mark.parametrize(
    "storage", ["heterogeneous", "homogeneous_scalar", "homogeneous_vector"]
)
def test_compressed_sum_matches_logical_dense_sum(cls, shape, storage):
    """Sum represented values rather than only the physical data buffer."""
    if storage == "heterogeneous":
        data = jnp.asarray([1.0, -2.0, 3.0, 4.0], dtype=jnp.float32)
    elif storage == "homogeneous_scalar":
        data = jnp.asarray(2.0, dtype=jnp.float32)
    else:
        data = jnp.asarray([2.0], dtype=jnp.float32)
    matrix = cls(
        (
            data,
            jnp.asarray([0, 2, 0, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 4], dtype=jnp.int32),
        ),
        shape=shape,
    )

    assert u.math.allclose(matrix.sum(), matrix.todense().sum())
    with pytest.raises(NotImplementedError, match="sum with axis"):
        matrix.sum(axis=0)


def test_compressed_homogeneous_sum_preserves_units():
    """Retain the shared value's physical unit in logical reduction."""
    matrix = CSR(
        (
            jnp.asarray([2.0], dtype=jnp.float32) * u.mV,
            jnp.asarray([0, 2, 0, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 4], dtype=jnp.int32),
        ),
        shape=(2, 3),
    )

    assert u.get_unit(matrix.sum()) == u.mV
    assert u.math.allclose(matrix.sum(), 8.0 * u.mV)


@pytest.mark.slow
@pytest.mark.parametrize('backend', BINARY_CSRMV_IMPLEMENTATIONS)
@pytest.mark.parametrize('dtype_name', WEIGHTED_EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('ndim', [1, 2])
@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('cls', [CSR, CSC])
def test_weighted_array_matmul_matches_dense_and_grads(cls, side, ndim, homo, dtype_name, backend):
    """``WeightedArray(x) @ M`` / ``M @ WeightedArray(x)`` equal the dense product with exact gradients."""
    with event_dtype_context(dtype_name):
        make, data = _weighted_matrix_factory(cls, homo, backend, seed=ndim)
        inner = _W_M if side == 'left' else _W_N
        if ndim == 1:
            x = event_values(dtype_name, inner, seed=ndim)
        elif side == 'left':
            x = event_values(dtype_name, (_W_B, inner), seed=ndim)
        else:
            x = event_values(dtype_name, (inner, _W_B), seed=ndim)

        def op(d, v):
            m = make(d)
            return WeightedArray(v) @ m if side == 'left' else m @ WeightedArray(v)

        def dense_op(d, z):
            W = make(d).todense()
            z = z.astype(d.dtype)
            return z @ W if side == 'left' else W @ z

        check_event_op_grads(op, dense_op, data, x, mode='weighted',
                             check_vmap=dtype_name == 'float32', **_weighted_tol(dtype_name))


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('cls', [CSR, CSC])
def test_weighted_array_matmul_jax_raw_differs_from_binary(cls, side):
    """Weighted and binary dispatch use different semantics on the same values."""
    make, data = _weighted_matrix_factory(cls, False, 'jax_raw', seed=3)
    M = make(data)
    W = np.asarray(M.todense(), dtype=np.float64)
    x = event_values('float32', _W_M if side == 'left' else _W_N, seed=3)
    x64 = np.asarray(x, np.float64)
    if side == 'left':
        weighted, binary = WeightedArray(x) @ M, BinaryArray(x) @ M
        exp_w, exp_b = x64 @ W, event_mask(x) @ W
    else:
        weighted, binary = M @ WeightedArray(x), M @ BinaryArray(x)
        exp_w, exp_b = W @ x64, W @ event_mask(x)
    np.testing.assert_allclose(np.asarray(weighted, np.float64), exp_w, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(binary, np.float64), exp_b, rtol=1e-5, atol=1e-5)


def test_weighted_array_integer_events_return_weight_dtype():
    make, data = _weighted_matrix_factory(CSR, False, 'jax_raw', seed=4)
    x = event_values('int8', _W_M, seed=4)
    y = WeightedArray(x) @ make(data)
    assert y.dtype == data.dtype
    np.testing.assert_allclose(np.asarray(y, np.float64),
                               np.asarray(x, np.float64) @ np.asarray(make(data).todense(), np.float64),
                               rtol=1e-5, atol=1e-5)


def test_weighted_array_cuda_raw_unfavorable_routes_pass_weighted(monkeypatch):
    """The explicit ``cuda_raw`` indexed route forwards ``weighted=True``."""
    import brainevent._csr.main as csr_main

    dense = jnp.array([[0.0, 2.0, 5.0], [3.0, 4.0, 0.0]], dtype=jnp.float32)
    csr = CSR.fromdense(dense, backend="cuda_raw")
    csc = CSC.fromdense(dense, backend="cuda_raw")
    seen = []

    def fake(data, indices, indptr, perm, v, *, shape, transpose, backend=None, weighted=False, **kw):
        seen.append(weighted)
        out = shape[1] if transpose else shape[0]
        return jnp.zeros((out,) + v.shape[1:], dtype=data.dtype)

    monkeypatch.setattr(csr_main, "binary_csrmv_indexed", fake)
    monkeypatch.setattr(csr_main, "binary_csrmm_indexed", fake)
    csr @ WeightedArray(jnp.array([1.0, 0.0, 2.0]))
    csr @ WeightedArray(jnp.ones((3, 2)))
    WeightedArray(jnp.array([1.0, 0.0])) @ csc
    WeightedArray(jnp.ones((2, 2))) @ csc
    csr @ BinaryArray(jnp.array([1.0, 0.0, 2.0]))
    assert seen == [True, True, True, True, False]


def test_bitpacked_binary_matmul_csr_is_binary():
    """``as_event_operand`` routes BitPackedBinary through the binary kernels."""
    make, data = _weighted_matrix_factory(CSR, False, 'jax_raw', seed=5)
    x = jnp.asarray(event_values('float32', _W_M, seed=5) != 0)
    y = brainevent.BitPackedBinary(x) @ make(data)
    np.testing.assert_allclose(np.asarray(y, np.float64),
                               event_mask(x) @ np.asarray(make(data).todense(), np.float64),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _CLASS_BACKENDS)
@pytest.mark.parametrize('fmt', ['csr', 'csc'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('batch', [None, 3])
def test_event_matmul_ignores_nonfinite_weights_of_inactive_events(backend, fmt, weighted, side, batch):
    rng = np.random.default_rng(41)
    m, k = 7, 6
    dense = rng.standard_normal((m, k)) * (rng.random((m, k)) < 0.6)
    n_ev = m if side == 'left' else k
    n_cols = 1 if batch is None else batch
    vals = np.resize(np.array([2.0, 0.0, -1.0, 0.0, 0.5]), n_ev)[:, None] * np.ones((1, n_cols))
    if weighted:
        vals[:, -1] *= 2.0
    inactive = vals[:, 0] == 0
    # NaN / Inf on every stored entry read by an inactive event.
    if side == 'left':
        dense[inactive] = np.where(dense[inactive] != 0, np.nan, 0.0)
        dense[np.flatnonzero(inactive)[0], 0] = np.inf
    else:
        dense[:, inactive] = np.where(dense[:, inactive] != 0, -np.inf, 0.0)
        dense[0, np.flatnonzero(inactive)[0]] = np.nan
    cls = CSR if fmt == 'csr' else CSC
    mat = cls.fromdense(jnp.asarray(dense, jnp.float32))
    if backend is not None:
        mat = cls((mat.data, mat.indices, mat.indptr), shape=mat.shape, backend=backend)
    f = vals if weighted else (vals != 0).astype(float)
    # Left products take events as (batch, n_ev), right products as (n_ev, batch).
    raw = vals[:, 0] if batch is None else (vals.T if side == 'left' else vals)
    x = jnp.asarray(raw, jnp.float32)
    ev = brainevent.WeightedArray(x) if weighted else BinaryArray(x != 0)
    if side == 'left':
        out = ev @ mat
        ref = np.where(f.T[:, :, None] != 0, f.T[:, :, None] * np.where(np.isfinite(dense), dense, 0.0)[None], 0.0).sum(1)
        ref = ref[0] if batch is None else ref
    else:
        out = mat @ ev
        ref = (np.where(np.isfinite(dense), dense, 0.0) @ f)
        ref = ref[:, 0] if batch is None else ref
    out = np.asarray(out)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('cls', [CSR, CSC])
@pytest.mark.parametrize('scalar_shape', [(), (1,), (1, 1)])
def test_compressed_mul_by_size_one_operand_keeps_data_shape(cls, scalar_shape):
    dense = np.array([[1., 0., 2.], [0., 3., 0.]], dtype=np.float32)
    m = cls.fromdense(jnp.asarray(dense))
    for out in (m * jnp.full(scalar_shape, 2.), jnp.full(scalar_shape, 2.) * m):
        assert out.data.shape == m.data.shape
        np.testing.assert_allclose(np.asarray(out.todense()), dense * 2)


@pytest.mark.parametrize('cls', [CSR, CSC])
@pytest.mark.parametrize('backend', _PYTREE_BACKENDS)
def test_closed_over_matrix_binary_matmul_survives_two_jits(cls, backend):
    # Regression: the binary task workspace was built lazily (``jnp.empty``)
    # under the first trace and cached on the closed-over matrix, so the next
    # trace, or ``jax.clear_caches()``, raised ``UnexpectedTracerError``.
    M, dense = _pytree_matrix(cls, backend)
    x, y = _spike_vector(M.shape[1]), _spike_vector(M.shape[0], seed=2)
    right = jax.jit(lambda s: M @ BinaryArray(s))
    left = jax.jit(lambda s: BinaryArray(s) @ M)
    for _ in range(2):
        np.testing.assert_allclose(right(x), dense @ np.asarray(x, np.float32), rtol=1e-5)
        np.testing.assert_allclose(left(y), np.asarray(y, np.float32) @ dense, rtol=1e-5)
        jax.clear_caches()
    np.testing.assert_allclose(M @ BinaryArray(x), dense @ np.asarray(x, np.float32), rtol=1e-5)


@pytest.mark.parametrize('cls', [CSR, CSC])
@pytest.mark.parametrize('binary', [False, True])
def test_jit_reuses_one_entry_for_structurally_equal_matrices(cls, binary):
    # Regression: index arrays lived in pytree aux data, so passing a second,
    # structurally equal matrix (distinct array objects) raised while JAX
    # compared the metadata.
    M1, dense = _pytree_matrix(cls, 'jax_raw')
    M2 = cls(
        (M1.data * 2, jnp.array(np.asarray(M1.indices)), jnp.array(np.asarray(M1.indptr))),
        shape=M1.shape, backend='jax_raw',
    )
    x = _spike_vector(M1.shape[1])
    f = jax.jit(lambda M, s: M @ (BinaryArray(s) if binary else s.astype(jnp.float32)))
    expected = dense @ np.asarray(x, np.float32)
    np.testing.assert_allclose(f(M1, x), expected, rtol=1e-5)
    np.testing.assert_allclose(f(M2, x), 2 * expected, rtol=1e-5)
    assert f._cache_size() == 1


@pytest.mark.parametrize('cls', [CSR, CSC])
@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('backend', _PYTREE_BACKENDS)
def test_scan_carry_with_binary_matmul_and_plasticity(cls, mirror, backend):
    # Regression: a lazily registered workspace changed the pytree structure of
    # a ``lax.scan`` carry. Without a mirror, the unfavorable direction must
    # fall back to a mirror-free kernel instead of building one under the trace.
    M, dense = _pytree_matrix(cls, backend)
    if mirror:
        M = M.build_weight_indices()
    n_pre, n_post = M.shape
    rng = np.random.default_rng(3)
    pre = jnp.asarray(rng.random((4, n_pre)) < 0.5)
    post = jnp.asarray(rng.random((4, n_post)) < 0.5)
    pre_trace = jnp.asarray(rng.random(n_pre), jnp.float32)
    post_trace = jnp.asarray(rng.random(n_post), jnp.float32)

    def step(W, spikes):
        pre_s, post_s = spikes
        out = (W @ BinaryArray(post_s), BinaryArray(pre_s) @ W)
        W = W.update_on_pre(pre_s, post_trace, w_max=2.0)
        W = W.update_on_post(pre_trace, post_s, w_max=2.0)
        return W, out

    final, (outs_r, outs_l) = jax.lax.scan(step, M, (pre, post))

    W = M
    for t in range(4):
        W, (r, l) = step(W, (pre[t], post[t]))
        np.testing.assert_allclose(outs_r[t], r, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(outs_l[t], l, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(final.todense(), W.todense(), rtol=1e-5, atol=1e-6)
    assert type(final) is cls


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_task_capacity_covers_heavy_rows_in_every_orientation(cls, monkeypatch):
    # Regression: ``.T`` re-keyed only one direction of the cached workspace, so
    # ``events @ csr.T`` after ``events @ csr`` on cuda_raw ran with a task
    # queue sized for the other orientation and silently dropped work.
    calls = _record_task_capacities(monkeypatch)
    rng = np.random.default_rng(0)
    shape = (4096, 3)
    dense = (rng.random(shape) < 0.9).astype(np.float32)  # three heavy columns
    M = cls(_compressed_parts(dense, cls), shape=shape, backend='cuda_raw')
    with jax.disable_jit():  # the spy reads ``indptr`` concretely
        for V in (M, M.T, M.T.T, M.T, M):
            V @ BinaryArray(jnp.ones(V.shape[1], bool))
            BinaryArray(jnp.ones(V.shape[0], bool)) @ V
    assert len(calls) == 10
    for name, capacity, indptr in calls:
        lengths = np.diff(np.asarray(indptr, dtype=np.int64))
        heavy = lengths[lengths > M.config.tpr_threshold]
        exact = np.sum((heavy + M.config.task_nnz - 1) // M.config.task_nnz)
        assert capacity >= exact, (name, capacity)


@pytest.mark.skipif(platform != 'gpu', reason='cuda_raw parity needs a GPU')
@pytest.mark.parametrize('cls', [CSR, CSC])
@pytest.mark.parametrize('shape', [(4096, 3), (3, 8192)])
def test_cuda_raw_heavy_lines_match_dense_after_transpose(cls, shape):
    rng = np.random.default_rng(0)
    dense = (rng.random(shape) < 0.9).astype(np.float32)
    M = cls(_compressed_parts(dense, cls), shape=shape, backend='cuda_raw')
    for V, D in ((M, dense), (M.T, dense.T), (M.T.T, dense), (M.T, dense.T)):
        x, y = _spike_vector(V.shape[1]), _spike_vector(V.shape[0], seed=2)
        np.testing.assert_allclose(V @ BinaryArray(x), D @ np.asarray(x, np.float32), rtol=1e-5)
        np.testing.assert_allclose(BinaryArray(y) @ V, np.asarray(y, np.float32) @ D, rtol=1e-5)


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_structure_is_a_pytree_leaf_and_unflatten_does_not_validate(cls):
    # Regression: ``tree_unflatten`` ran the constructor, so ``jax.eval_shape``,
    # ``jit(...).lower`` and ``tree.map`` with non-array leaves crashed.
    M, dense = _pytree_matrix(cls, 'jax_raw')
    v = jnp.ones(M.shape[1], jnp.float32)
    leaves = jax.tree.leaves(M)
    assert any(leaf is M.indices for leaf in leaves)
    assert any(leaf is M.indptr for leaf in leaves)
    assert jax.eval_shape(lambda W, x: W @ x, M, v).shape == (M.shape[0],)
    assert jax.eval_shape(lambda W, x: W @ BinaryArray(x), M, v > 0).shape == (M.shape[0],)
    jax.jit(lambda W, x: W @ x).lower(M, v)
    nones = jax.tree.map(lambda _: None, M)
    assert type(nones) is cls and nones.indices is None
    shapes = jax.tree.map(lambda a: a.shape, M)
    assert shapes.indptr == M.indptr.shape


@pytest.mark.parametrize('cls', [CSR, CSC])
@pytest.mark.parametrize('nnz_per_row', [20, 200])
def test_hlo_size_does_not_grow_with_nse(cls, nnz_per_row):
    # Regression: index arrays in aux data were embedded as HLO constants
    # (160 MB HLO and 10-30x slower compiles at 10M non-zeros).
    n = 1000
    rng = np.random.default_rng(0)
    indptr = jnp.asarray(np.arange(n + 1) * nnz_per_row, jnp.int32)
    indices = jnp.asarray(rng.integers(0, n, n * nnz_per_row), jnp.int32)
    M = cls((jnp.ones(n * nnz_per_row, jnp.float32), indices, indptr), shape=(n, n), backend='jax_raw')
    text = jax.jit(lambda W, x: W @ x).lower(M, jnp.ones(n, jnp.float32)).as_text()
    assert len(text) < 30_000


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_gradients_through_data_and_allow_int(cls):
    M, dense = _pytree_matrix(cls, 'jax_raw')
    v = jnp.asarray(np.random.default_rng(4).random(M.shape[1]), jnp.float32)
    mask = jnp.asarray(dense != 0)

    def loss(W):
        return jnp.sum((W @ v) ** 2)

    g_data = jax.grad(lambda d: loss(M.with_data(d)))(M.data)
    g_dense = jax.grad(lambda D: jnp.sum((D @ v) ** 2))(jnp.asarray(dense))
    np.testing.assert_allclose(cls(_compressed_parts(np.where(mask, g_dense, 0), cls), shape=M.shape).data,
                               g_data, rtol=1e-5)
    g_M = jax.grad(loss, allow_int=True)(M)
    assert type(g_M) is cls
    np.testing.assert_allclose(g_M.data, g_data, rtol=1e-5)
    assert g_M.indices.dtype == jax.dtypes.float0


@pytest.mark.parametrize('backend', _PYTREE_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
def test_public_binary_functions_need_no_workspace(backend, transpose):
    # Callers pass math arrays and config; lowering allocates scratch.
    M, dense = _pytree_matrix(CSR, backend)
    D = dense.T if transpose else dense
    x = _spike_vector(D.shape[1])
    X = jnp.asarray(np.random.default_rng(5).random((D.shape[1], 3)) < 0.5)
    mv = jax.jit(lambda d, i, p, s: brainevent.binary_csrmv(
        d, i, p, s, shape=M.shape, transpose=transpose, backend=backend, config=M.config))
    mm = jax.jit(lambda d, i, p, s: brainevent.binary_csrmm(
        d, i, p, s, shape=M.shape, transpose=transpose, backend=backend, config=M.config))
    np.testing.assert_allclose(mv(M.data, M.indices, M.indptr, x), D @ np.asarray(x, np.float32), rtol=1e-5)
    np.testing.assert_allclose(mm(M.data, M.indices, M.indptr, X), D @ np.asarray(X, np.float32), rtol=1e-5)


# Object and global backend selection
# -----------------------------------

@pytest.fixture(autouse=False)
def _backend_restore_global_backends():
    previous = {platform: be.config.get_backend(platform) for platform in ('cpu', 'gpu', 'tpu')}
    be.config.clear_backends()
    try:
        yield
    finally:
        be.config.clear_backends()
        for platform, backend in previous.items():
            if backend is not None:
                be.config.set_backend(platform, backend)


def _backend_product(matrix, values, weighted):
    events = be.WeightedArray(values) if weighted else be.BinaryArray(values)
    return matrix @ events if isinstance(matrix, be.CSR) else events @ matrix


def _backend_spy_products(monkeypatch):
    calls = []

    def fake(indexed):
        def product(*args, shape, transpose, backend, weighted, **kwargs):
            calls.append((indexed, transpose, backend, weighted))
            rows = shape[1] if transpose else shape[0]
            return jnp.zeros((rows,) + args[-1].shape[1:], dtype=args[0].dtype)
        return product

    for name in ('binary_csrmv', 'binary_csrmm'):
        monkeypatch.setattr(csr_main, name, fake(False))
        monkeypatch.setattr(csr_main, name + '_indexed', fake(True))
    return calls


@pytest.mark.usefixtures('_backend_restore_global_backends')
@pytest.mark.parametrize('cuda', [False, True], ids=['no_cuda', 'cuda'])
@pytest.mark.parametrize('cls', [be.CSR, be.CSC])
@pytest.mark.parametrize('ndim', [1, 2], ids=['mv', 'mm'])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('backend,global_backend,prebuilt,indexed', [
    (None, 'cuda_raw', False, True),
    (None, 'cuda_raw', True, True),
    (None, None, False, None),
    (None, 'jax_raw', False, False),
    ('jax_raw', 'cuda_raw', False, False),
    ('cuda_raw', 'jax_raw', False, True),
], ids=['global_cuda_lazy', 'global_cuda_prebuilt', 'unset', 'global_jax',
        'object_jax', 'object_cuda'])
def test_event_backend_precedence(monkeypatch, cuda, cls, ndim, weighted, backend,
                                  global_backend, prebuilt, indexed):
    # An explicit or global choice decides the route; with neither, the GPU
    # default (``indexed=None``) scatters over the mirror only when CUDA exists.
    monkeypatch.setattr(csr_main, '_cuda_available', lambda: cuda)
    if indexed is None:
        indexed = cuda
    matrix = cls.fromdense(jnp.array([[1., 0., 2.], [0., 3., 4.]]), backend=backend,
                           precompute_weight_indices=prebuilt)
    be.config.set_backend('gpu', global_backend)
    calls = _backend_spy_products(monkeypatch)
    size = matrix.shape[1] if cls is be.CSR else matrix.shape[0]
    shape = (size,) if ndim == 1 else ((size, 4) if cls is be.CSR else (4, size))

    _backend_product(matrix, jnp.ones(shape), weighted)

    assert calls == [(indexed, indexed, backend, weighted)]
    assert matrix.backend == backend  # Do not turn the GPU default into an explicit CPU request.
    assert (matrix._cache.mirror is not None) == (prebuilt or indexed)


@pytest.mark.usefixtures('_backend_restore_global_backends')
@pytest.mark.parametrize('backend,indexed', [(None, False), ('cuda_raw', True)], ids=['default', 'explicit_cuda'])
def test_deterministic_gpu_ops_keep_the_default_off_the_mirror(monkeypatch, backend, indexed):
    # Under --xla_gpu_deterministic_ops the GPU default resolves to jax_raw, so
    # even a CUDA process gathers; an explicit cuda_raw request is kept.
    monkeypatch.setattr(csr_main, '_cuda_available', lambda: True)
    monkeypatch.setenv('XLA_FLAGS', '--xla_gpu_deterministic_ops=true')
    matrix = be.CSR.fromdense(jnp.array([[1., 0., 2.], [0., 3., 4.]]), backend=backend)
    calls = _backend_spy_products(monkeypatch)
    matrix @ be.BinaryArray(jnp.ones(3))
    matrix @ be.BinaryArray(jnp.ones((3, 2)))
    assert calls == [(indexed, indexed, backend, False)] * 2
    assert (matrix._cache.mirror is not None) == indexed


@pytest.mark.usefixtures('_backend_restore_global_backends')
@pytest.mark.parametrize('cls', [be.CSR, be.CSC])
@pytest.mark.parametrize('backend,global_backend,warns', [
    (None, None, False),
    (None, 'jax_raw', False),
    (None, 'cuda_raw', True),
    ('cuda_raw', None, True),
], ids=['default', 'global_jax', 'global_cuda', 'object_cuda'])
def test_traced_structure_warns_only_when_the_mirror_scatter_was_requested(
    monkeypatch, cls, backend, global_backend, warns,
):
    # A structure built inside jit carries no mirror, so the unfavorable product
    # gathers. The GPU default merely prefers the scatter, and gathering is what it
    # did before routing by CUDA availability, so it stays quiet; an explicit or
    # global cuda_raw choice asked for the scatter and is told once.
    import brainevent._structure as structure
    monkeypatch.setattr(csr_main, '_cuda_available', lambda: True)
    monkeypatch.setattr(structure, '_WARNED', set())
    be.config.set_backend('gpu', global_backend)
    source = cls.fromdense(jnp.array([[1., 0., 2.], [0., 3., 4.]]))
    calls = _backend_spy_products(monkeypatch)
    size = source.shape[1] if cls is be.CSR else source.shape[0]

    def product(data, indices, indptr, values):
        matrix = cls((data, indices, indptr), shape=source.shape, backend=backend)
        return _backend_product(matrix, values, False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for batch in (None, 4, 2):  # three distinct traces: one MV, two MMs
            shape = (size,) if batch is None else ((size, batch) if cls is be.CSR else (batch, size))
            jax.make_jaxpr(product)(source.data, source.indices, source.indptr, jnp.ones(shape))
    performance = [w for w in caught if issubclass(w.category, be.PerformanceWarning)]
    assert len(performance) == int(warns)
    assert [indexed for indexed, *_ in calls] == [False] * 3


@pytest.mark.usefixtures('_backend_restore_global_backends')
@pytest.mark.parametrize('cuda', [False, True], ids=['no_cuda', 'cuda'])
def test_global_changes_retrace_existing_jit_route(monkeypatch, cuda):
    monkeypatch.setattr(csr_main, '_cuda_available', lambda: cuda)
    matrix = be.CSR.fromdense(jnp.array([[1., 0., 2.], [0., 3., 4.]]))
    calls = _backend_spy_products(monkeypatch)
    product = jax.jit(lambda values: matrix @ be.BinaryArray(values))
    values = jnp.ones(3)
    for backend in ('jax_raw', 'cuda_raw', None):
        be.config.set_backend('gpu', backend)
        product(values).block_until_ready()
    # Clearing the global choice returns to the GPU default, which takes the
    # indexed route only on a CUDA process.
    assert [call[0] for call in calls] == [False, True, cuda]
    assert all(call[2] is None for call in calls)


@pytest.mark.usefixtures('_backend_restore_global_backends')
def test_gpu_global_backend_keeps_cpu_execution_on_cpu(monkeypatch):
    device = jax.devices('cpu')[0]
    with jax.default_device(device):
        dense = jnp.array([[1., 0., 2.], [0., 3., 4.]])
        matrix = be.CSR.fromdense(dense)
        values = jnp.array([1., 0., -2.])
        be.config.set_backend('cpu', 'jax_raw')
        be.config.set_backend('gpu', 'cuda_raw')
        calls = []
        original = csr_main.binary_csrmv_indexed

        def indexed(*args, **kwargs):
            calls.append(kwargs['backend'])
            return original(*args, **kwargs)

        monkeypatch.setattr(csr_main, 'binary_csrmv_indexed', indexed)
        np.testing.assert_allclose(matrix @ be.WeightedArray(values), dense @ values)
        assert calls == [None]


@pytest.mark.usefixtures('_backend_restore_global_backends')
@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='CUDA hybrid lowering needs a GPU')
@pytest.mark.parametrize('cls', [be.CSR, be.CSC])
@pytest.mark.parametrize('ndim', [1, 2], ids=['mv', 'mm'])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
def test_global_cuda_uses_hybrid_forward_and_weight_gradient(cls, ndim, weighted):
    dense = jnp.array([[1., 0., 2.], [0., 3., 4.]])
    matrix = cls.fromdense(dense, precompute_weight_indices=True)
    be.config.set_backend('gpu', 'cuda_raw')
    size = matrix.shape[1] if cls is be.CSR else matrix.shape[0]
    shape = (size,) if ndim == 1 else ((size, 4) if cls is be.CSR else (4, size))
    values = (jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) % 3) - 1
    factors = values if weighted else (values != 0).astype(values.dtype)
    expected = dense @ factors if cls is be.CSR else factors @ dense
    product = jax.jit(lambda mat, x: _backend_product(mat, x, weighted))
    lowered = product.lower(matrix, values)
    symbol = 'hybrid_spmv_col_major_' if ndim == 1 else 'batched_spmm_v3_col_major_'
    assert symbol in lowered.as_text()
    np.testing.assert_allclose(lowered.compile()(matrix, values), expected, rtol=1e-5, atol=1e-6)

    gradient = jax.jit(jax.grad(lambda data: _backend_product(matrix.with_data(data), values, weighted).sum()))
    lowered_gradient = gradient.lower(matrix.data)
    symbol = 'hybrid_sddmv_' if ndim == 1 else 'batched_v3_sddmm'
    assert symbol in lowered_gradient.as_text()
    coo = matrix.tocoo()
    # Binary events differentiate through their activity mask, weighted ones
    # through their values: the same factors as the forward product.
    per_input = (
        factors if ndim == 1 else factors.sum(axis=1 if cls is be.CSR else 0)
    )
    expected_gradient = per_input[coo.col if cls is be.CSR else coo.row]
    np.testing.assert_allclose(lowered_gradient.compile()(matrix.data), expected_gradient,
                               rtol=1e-5, atol=1e-6)


# Configuration lifecycle on sparse objects
# -----------------------------------------

def _lifecycle_matrix(cls=CSR, **kwargs):
    shape = (2, 3) if cls is CSR else (3, 2)
    return cls(
        (jnp.array([1., 2., 3.]), jnp.array([0, 2, 1], jnp.int32),
         jnp.array([0, 2, 3], jnp.int32)),
        shape=shape, backend='jax_raw', **kwargs,
    )


def _no_load(*args, **kwargs):
    raise AssertionError('reconstruction must not load persisted configuration')


def _count_mirror_builds(monkeypatch):
    """Record every eager CSR/CSC mirror construction."""
    builds = []
    original = csr_main.csr_to_csc_index

    def record(*args, **kwargs):
        builds.append(kwargs.get('shape'))
        return original(*args, **kwargs)

    monkeypatch.setattr(csr_main, 'csr_to_csc_index', record)
    return builds


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_lazy_mirror_is_built_once_and_shared_by_existing_siblings(monkeypatch, cls):
    matrix = _lifecycle_matrix(cls, config=CSRConfig())
    sibling = matrix.with_data(matrix.data * 2)
    transposed = matrix.T
    assert matrix._cache.mirror is None
    builds = _count_mirror_builds(monkeypatch)
    mirror = matrix._mirror()
    assert builds == [matrix._storage_shape]
    assert sibling._mirror() is mirror
    assert transposed._mirror() is mirror
    assert matrix._mirror() is mirror
    assert len(builds) == 1
    assert sibling.config is transposed.config is matrix.config


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_pytree_round_trip_carries_shared_mirror_and_config_adds_no_leaves(monkeypatch, cls):
    matrix = _lifecycle_matrix(cls, config=CSRConfig(tpr_threshold=7))
    sibling = matrix.with_data(matrix.data * 2)
    matrix.build_weight_indices()  # fills the cache the sibling shares
    key = matrix._mirror_key
    builds = _count_mirror_builds(monkeypatch)

    leaves = jax.tree.leaves(sibling)
    assert jax.tree.leaves(sibling.config) == []
    # data, indices, indptr and the three mirror arrays; nothing from config.
    assert len(leaves) == 6
    assert all(a is b for a, b in zip(leaves[3:], matrix.buffers[key]))

    restored = jax.jit(lambda value: value)(sibling)
    assert builds == []
    assert restored.config.mv == sibling.config.mv
    for got, want in zip(restored.buffers[key], matrix.buffers[key]):
        np.testing.assert_array_equal(got, want)
    mapped = jax.tree.map(lambda _: None, sibling)
    assert mapped.config.tpr_threshold == 7
    assert mapped._cache.mirror == (None, None, None)


@pytest.mark.skipif(
    jax.default_backend() != 'gpu', reason='native CUDA lowering requires a GPU',
)
@pytest.mark.parametrize('cls', [CSR, CSC])
@pytest.mark.parametrize('operation', ['build_mirror', 'convert'])
@pytest.mark.parametrize('weighted', [False, True])
def test_structure_derived_inside_jit_keeps_native_cuda_mm(cls, operation, weighted):
    # A mirror or a format conversion computed inside ``jax.jit`` carries no
    # host statistics. MM plans come from shapes, so the event-driven
    # directions still lower to the native hybrid kernel.
    matrix = _lifecycle_matrix(cls, config=CSRConfig())
    matrix.backend = 'cuda_raw'
    if operation == 'build_mirror':
        derived = jax.jit(lambda value: value.build_weight_indices())(matrix)
    else:
        convert = (
            (lambda value: value.tocsc()) if cls is CSR
            else (lambda value: value.tocsr())
        )
        derived = jax.jit(convert)(matrix)
    dense = np.asarray(matrix.todense())
    kind = WeightedArray if weighted else BinaryArray
    # ``events @ CSR`` and ``CSC @ events`` scatter over the stored layout; the
    # other direction is event-driven only through a carried mirror.
    favorable_left = isinstance(derived, CSR)
    for left in (False, True):
        size = derived.shape[0 if left else 1]
        values = jnp.arange(1, 1 + 2 * size, dtype=jnp.float32)
        values = values.reshape((2, size) if left else (size, 2))
        values = values.at[0, 0].set(0)
        call = jax.jit(
            (lambda m, x: kind(x) @ m) if left
            else (lambda m, x: m @ kind(x))
        )
        if operation == 'build_mirror' or left == favorable_left:
            text = call.lower(derived, values).compile().as_text()
            targets = re.findall(r'custom_call_target="([^"]+)"', text)
            assert any(t.startswith('csr_hybrid_mm') for t in targets), targets
        actual = call(derived, values)
        events = np.asarray(values) if weighted else np.asarray(values) != 0
        expected = events @ dense if left else dense @ events
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_config_snapshot_is_loaded_once_and_survives_reconstruction(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    calls = []
    snapshot = CSRConfig(block_size=128, tpr_threshold=7)
    monkeypatch.setattr(initialize, 'load_csr_config', lambda **kw: calls.append(kw) or snapshot)
    matrix = _lifecycle_matrix(cls)
    assert len(calls) == 1
    # Storage (CSR-traversal) shape, whichever class holds it.
    assert calls[0]['matrix_metadata'] == {'shape': (2, 3), 'nnz': 3}
    assert matrix.config is snapshot

    monkeypatch.setattr(initialize, 'load_csr_config', _no_load)
    assert matrix.with_data(matrix.data * 2).config is matrix.config
    assert matrix.T.config is matrix.config
    transformed = jax.jit(lambda value: value.with_data(value.data * 2))(matrix)
    np.testing.assert_array_equal(transformed.data, matrix.data * 2)
    assert transformed.config.mv == snapshot.mv
    leaves = jax.tree.leaves(matrix)
    assert len(leaves) == 3
    assert leaves[1] is matrix.indices and leaves[2] is matrix.indptr


def _selection_matrix(cls, indices_device=None, *, traced_indices=None):
    if cls in (CSR, CSC):
        indices = np.array([0, 1, 1], np.int32)
        indptr = jnp.array([0, 2, 3], jnp.int32)
        shape = (2, 2)
    else:
        indices = np.array([[0, 1], [1, 0]], np.int32)
        indptr = None
        shape = (2, 2)
    indices = (jnp.asarray(indices) if indices_device is None
               else jax.device_put(indices, indices_device))
    if traced_indices is not None:
        indices = traced_indices
    data = jnp.ones(indices.size)
    if cls in (CSR, CSC):
        return cls(data, indices, indptr, shape=shape, backend='jax_raw')
    return cls(data.reshape(indices.shape), indices, shape=shape, backend='jax_raw')


@pytest.mark.parametrize('cls', [CSR, CSC, FixedNumPerPre, FixedNumPerPost])
def test_constructor_selects_persisted_tuning_by_structure_device_kind(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    selected = []
    snapshot = CSRConfig(tpr_threshold=9)

    def load(device_kind=None, *, matrix_metadata=None):
        selected.append((device_kind, matrix_metadata))
        return snapshot

    monkeypatch.setattr(initialize, 'load_csr_config', load)
    metadata = {'shape': (2, 2), 'nnz': 3 if cls in (CSR, CSC) else 4}
    cpu = jax.devices('cpu')[0]

    eager = _selection_matrix(cls)
    on_cpu = _selection_matrix(cls, cpu)
    # Traced indices carry no device; the store then uses its own fallback.
    traced = jax.jit(lambda i: _selection_matrix(cls, traced_indices=i).data)
    traced(eager.indices)

    default_kind = next(iter(eager.indices.devices())).device_kind
    assert selected == [(default_kind, metadata), (cpu.device_kind, metadata), (None, metadata)]
    assert eager.config is snapshot and on_cpu.config is snapshot


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_from_parts_and_unflatten_never_load_tuning(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    matrix = _lifecycle_matrix(cls, config=CSRConfig(tpr_threshold=5))
    monkeypatch.setattr(initialize, 'load_csr_config', _no_load)
    reconstructed = cls._from_parts(
        None, None, None, shape=matrix.shape, config=matrix.config,
    )
    assert reconstructed.config is matrix.config
    assert cls._from_parts(None, None, None, shape=matrix.shape).config.mv == CSRConfig().mv
    mapped = jax.tree.map(lambda _: None, matrix)
    assert mapped.indices is None
    assert mapped.config.tpr_threshold == 5


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_format_conversion_keeps_config_without_reloading(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    matrix = _lifecycle_matrix(cls, config=CSRConfig(tpr_threshold=5))
    matrix.build_weight_indices()
    monkeypatch.setattr(initialize, 'load_csr_config', _no_load)
    convert = (lambda value: value.tocsc()) if cls is CSR else (lambda value: value.tocsr())
    converted = convert(matrix)
    assert converted.config is matrix.config
    np.testing.assert_allclose(converted.todense(), matrix.todense())
    traced = jax.jit(convert)(matrix)
    assert traced.config.mv == matrix.config.mv
    np.testing.assert_allclose(traced.todense(), matrix.todense())


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_structure_changing_operations_keep_config(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    matrix = _lifecycle_matrix(cls, config=CSRConfig(tpr_threshold=5))
    dense = jnp.array([[0., 1.], [2., 0.]])
    square = cls.fromdense(dense, backend='jax_raw', config=CSRConfig(tpr_threshold=5))
    prepared = square.prepare_diag_add()
    monkeypatch.setattr(initialize, 'load_csr_config', _no_load)

    sliced = matrix.slice_rows([0])
    assert sliced.config is matrix.config
    np.testing.assert_allclose(sliced.todense(), matrix.todense()[:1])

    eager = square.diag_add(jnp.ones(2))
    assert eager.nse == 4
    assert eager.config is square.config
    traced = jax.jit(lambda value: value.diag_add(jnp.ones(2)))(prepared)
    assert traced.config.mv == square.config.mv
    for result in (eager, traced):
        np.testing.assert_allclose(result.todense(), dense + jnp.eye(2))


@pytest.mark.parametrize('cls', [CSR, CSC, FixedNumPerPre, FixedNumPerPost])
@pytest.mark.parametrize('setter', ['register_buffer', 'set_buffer'])
@pytest.mark.parametrize('reordered', [False, True], ids=['same', 'reordered'])
def test_setting_valid_mirror_copies_cache_and_drives_indexed_route(monkeypatch, cls, setter, reordered):
    # CSR/CSC take the indexed mirror route only for CUDA; force it so the
    # products below traverse the replacement mirror on every platform.
    monkeypatch.setattr(csr_main, '_use_cuda_indexed_binary_route', lambda *args: True)
    if cls in (CSR, CSC):
        matrix = cls(
            jnp.array([1., 2., 3., 4.]), jnp.array([0, 2, 1, 2], jnp.int32),
            jnp.array([0, 2, 4], jnp.int32),
            shape=(2, 3) if cls is CSR else (3, 2),
            backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
        key = 'csc' if cls is CSR else 'csr'
    else:
        matrix = cls(
            jnp.array([[1., 2.], [3., 4.]]), jnp.array([[0, 2], [1, 2]], jnp.int32),
            shape=(2, 3) if cls is FixedNumPerPre else (3, 2),
            backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
        key = 'csc'
    original = matrix.buffers[key]
    mirror = original
    if reordered:
        # Swap the two entries of the last mirror row, with their permutation.
        order = jnp.array([0, 1, 3, 2])
        mirror = (mirror[0], mirror[1][order], mirror[2][order])
    sibling = matrix.with_data(matrix.data * 2)
    original_cache, config = matrix._cache, matrix.config
    getattr(matrix, setter)(key, mirror)

    # The replacement goes to a copy: siblings keep the previous mirror.
    assert matrix._cache is not original_cache
    assert sibling._cache is original_cache
    assert all(a is b for a, b in zip(sibling.buffers[key], original))
    assert all(a is b for a, b in zip(matrix.buffers[key], mirror))
    assert matrix.config is config
    if cls in (CSR, FixedNumPerPre):
        events = BinaryArray(jnp.array([[True, False], [False, True], [True, True]]))
        actual = matrix @ events
        expected = matrix.todense() @ events.value
    else:
        events = BinaryArray(jnp.array([[True, False, True], [False, True, True]]))
        actual = events @ matrix
        expected = events.value @ matrix.todense()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='CUDA hybrid MM requires a GPU')
@pytest.mark.parametrize('setter', ['register_buffer', 'set_buffer'])
@pytest.mark.parametrize('reordered', [False, True], ids=['same', 'reordered'])
def test_fixednum_mirror_update_keeps_cuda_mm_usable(setter, reordered):
    matrix = FixedNumPerPre(
        jnp.array([[1., 2.], [3., 4.]], jnp.float32),
        jnp.array([[0, 2], [1, 2]], jnp.int32),
        shape=(2, 3), backend='cuda_raw', config=CSRConfig(),
    ).build_weight_indices()
    events = BinaryArray(jnp.array([[True, False], [False, True], [True, True]]))
    expected = jnp.array([[3., 2.], [4., 7.]])
    np.testing.assert_allclose(matrix @ events, expected)
    mirror = matrix.buffers['csc']
    if reordered:
        order = jnp.array([0, 1, 3, 2])
        mirror = (mirror[0], mirror[1][order], mirror[2][order])
    getattr(matrix, setter)('csc', mirror)
    np.testing.assert_allclose(matrix @ events, expected)


@pytest.mark.parametrize('cls', [CSR, FixedNumPerPre])
def test_invalid_mirror_update_is_rejected_and_keeps_previous_cache(cls):
    if cls is CSR:
        matrix = cls(
            jnp.ones(2), jnp.array([0, 1], jnp.int32),
            jnp.array([0, 1, 2], jnp.int32), shape=(2, 2),
            backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
    else:
        matrix = cls(
            jnp.ones((2, 1)), jnp.array([[0], [1]], jnp.int32),
            shape=(2, 2), backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
    cache, config = matrix._cache, matrix.config
    _, indices, perm = matrix.buffers['csc']
    bad_mirror = (jnp.zeros((3,), jnp.int32), indices, perm)
    with pytest.raises(ValueError, match='indptr'):
        matrix.set_buffer('csc', bad_mirror)
    assert matrix._cache is cache
    assert matrix.config is config


def _mirror_source(cls):
    """A 2x3 matrix with three stored edges and its concrete mirror."""
    if cls in (CSR, CSC):
        matrix = cls(jnp.ones(3), jnp.array([0, 2, 1], jnp.int32), jnp.array([0, 2, 3], jnp.int32),
                     shape=(2, 3) if cls is CSR else (3, 2), backend='jax_raw', config=CSRConfig())
        args = (matrix.data, matrix.indices, matrix.indptr)
    else:
        matrix = cls(jnp.ones((3, 1)), jnp.array([[0], [1], [1]], jnp.int32),
                     shape=(3, 2) if cls is FixedNumPerPre else (2, 3), backend='jax_raw',
                     config=CSRConfig())
        args = (matrix.data, matrix.indices)
    return matrix, args, matrix.build_weight_indices().buffers['csc' if cls is not CSC else 'csr']


@pytest.mark.parametrize('cls', [CSR, CSC, FixedNumPerPre, FixedNumPerPost])
@pytest.mark.parametrize('defect', ['indptr_end', 'indptr_order', 'indptr_length',
                                    'indices_length', 'perm_length', 'arity'])
def test_constructor_rejects_a_malformed_mirror_buffer(cls, defect):
    matrix, args, (indptr, indices, perm) = _mirror_source(cls)
    bad = {
        'indptr_end': (indptr.at[-1].add(-1), indices, perm),
        'indptr_order': (indptr.at[1].set(indptr[-1] + 1), indices, perm),
        'indptr_length': (indptr[:-1], indices, perm),
        'indices_length': (indptr, indices[:-1], perm),
        'perm_length': (indptr, indices, perm[:-1]),
        'arity': (indptr, indices),
    }[defect]
    key = 'csr' if cls is CSC else 'csc'
    with pytest.raises(ValueError):
        cls(*args, shape=matrix.shape, backend='jax_raw', config=CSRConfig(), buffers={key: bad})
    rebuilt = cls(*args, shape=matrix.shape, backend='jax_raw', config=CSRConfig(),
                  buffers={key: (indptr, indices, perm)})
    assert rebuilt._cache.mirror is not None


def test_traced_mirror_buffer_is_checked_statically_only():
    matrix, args, mirror = _mirror_source(CSR)

    @jax.jit
    def build(indptr, indices, perm):
        rebuilt = CSR(*args, shape=matrix.shape, backend='jax_raw', config=CSRConfig(),
                      buffers={'csc': (indptr, indices, perm)})
        return rebuilt @ jnp.ones(matrix.shape[1])

    np.testing.assert_allclose(build(*mirror), matrix.todense() @ jnp.ones(matrix.shape[1]))


@pytest.mark.parametrize('cls', [CSR, CSC, FixedNumPerPre, FixedNumPerPost])
def test_constructor_ingests_supplied_mirror_and_loads_tuning_once(monkeypatch, cls):
    from brainevent._csr.binary import initialize

    if cls in (CSR, CSC):
        source = _lifecycle_matrix(cls, config=CSRConfig()).build_weight_indices()
        args = (source.data, source.indices, source.indptr)
    else:
        source = cls(
            jnp.ones((2, 2)), jnp.array([[0, 2], [1, 2]], jnp.int32),
            shape=(2, 3) if cls is FixedNumPerPre else (3, 2),
            backend='jax_raw', config=CSRConfig(),
        ).build_weight_indices()
        args = (source.data, source.indices)
    key = 'csr' if cls is CSC else 'csc'
    supplied = source.buffers[key]
    loads = []
    monkeypatch.setattr(initialize, 'load_csr_config', lambda **kw: loads.append(kw) or CSRConfig())
    matrix = cls(
        *args, shape=source.shape, buffers=source.buffers,
        backend='cuda_raw' if jax.default_backend() == 'gpu' else 'jax_raw',
    )
    assert len(loads) == 1
    dense = matrix.todense()
    right = jnp.ones((matrix.shape[1], 4), bool)
    left = jnp.ones((4, matrix.shape[0]), bool)
    np.testing.assert_allclose(matrix @ BinaryArray(right), dense @ right)
    np.testing.assert_allclose(BinaryArray(left) @ matrix, left @ dense)
    # The supplied mirror is used as given, never rebuilt.
    assert all(a is b for a, b in zip(matrix.buffers[key], supplied))


@pytest.mark.parametrize('check_structure', [False, True])
def test_construction_reads_device_structure_only_to_validate_it(monkeypatch, check_structure):
    # Configuration selection reads only static metadata, so without a
    # structure check nothing is copied to the host; with one, each structure
    # array is copied exactly once and that copy serves every validation.
    reads = []
    device_get = jax.device_get

    def record(value):
        reads.append(value)
        return device_get(value)

    monkeypatch.setattr(jax, 'device_get', record)
    indices = jnp.array([0, 2, 1, 2, 0], jnp.int32)
    indptr = jnp.array([0, 2, 5], jnp.int32)
    matrix = CSR((jnp.ones(5), indices, indptr), shape=(2, 3), backend='jax_raw',
                 config=CSRConfig(), check_structure=check_structure)
    if check_structure:
        assert len(reads) == 2
        assert sum(value is matrix.indices for value in reads) == 1
        assert sum(value is matrix.indptr for value in reads) == 1
    else:
        assert reads == []
    np.testing.assert_array_equal(matrix.todense(), [[1, 0, 1], [1, 1, 1]])


@pytest.mark.parametrize('check_structure', [False, True])
def test_numpy_structure_construction_needs_no_device_readback(monkeypatch, check_structure):
    def no_readback(*args, **kwargs):
        raise AssertionError('host structure is already available')

    monkeypatch.setattr(jax, 'device_get', no_readback)
    matrix = CSR(
        np.ones(3), np.array([0, 2, 1], np.int32), np.array([0, 2, 3], np.int32),
        shape=(2, 3), backend='jax_raw', config=CSRConfig(), check_structure=check_structure,
    )
    assert matrix.nse == 3


@pytest.mark.parametrize('config', [None, CSRConfig()], ids=['persisted', 'explicit'])
def test_construction_from_closed_concrete_structure_under_trace_emits_no_ops(config):
    indices = jnp.array([0, 1], jnp.int32)
    indptr = jnp.array([0, 1, 2], jnp.int32)
    make = lambda data: CSR(data, indices, indptr, shape=(2, 2), config=config)
    closed = jax.make_jaxpr(lambda data: make(data).data)(jnp.ones(2))
    assert not closed.jaxpr.eqns
    matrix = jax.jit(make)(jnp.ones(2))
    np.testing.assert_array_equal(matrix.todense(), np.eye(2))


def test_cpu_sharded_structure_builds_mirror_and_multiplies():
    # JAX must see the extra CPU devices before this process imports it.
    script = '''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from brainevent import CSR, CSC, FixedNumPerPre, FixedNumPerPost, BinaryArray, config
from brainevent._csr.binary import initialize
from brainevent._csr.config import CSRConfig

devices = jax.devices('cpu')
assert len(devices) == 2
mesh = Mesh(np.asarray(devices), ('cpu_shard',))
config.set_backend('cpu', 'jax_raw')
kinds = []
load = initialize.load_csr_config
initialize.load_csr_config = lambda device_kind=None, **kw: kinds.append(device_kind) or load(device_kind, **kw)

def placed(value, spec):
    return jax.device_put(jnp.asarray(value), NamedSharding(mesh, spec))

ones = placed(np.ones(2, np.float32), P())
for cls in (CSR, CSC, FixedNumPerPre, FixedNumPerPost):
    if cls in (CSR, CSC):
        args = (
            placed(np.ones(4, np.float32), P('cpu_shard')),
            placed(np.array([0, 1, 0, 1], np.int32), P('cpu_shard')),
            placed(np.array([0, 2, 4], np.int32), P()),
        )
    else:
        args = (
            placed(np.ones((2, 2), np.float32), P('cpu_shard')),
            placed(np.array([[0, 1], [0, 1]], np.int32), P('cpu_shard')),
        )
    matrix = cls(*args, shape=(2, 2))
    assert jax.tree.leaves(matrix.config) == []
    result = jax.jit(lambda m, x: m @ BinaryArray(x))(matrix, ones)
    np.testing.assert_array_equal(np.asarray(result), [2, 2])
    mirror = matrix.build_weight_indices()
    rebuilt = cls(*args, shape=(2, 2), buffers=mirror.buffers, config=CSRConfig())
    for m in (mirror, rebuilt):
        np.testing.assert_array_equal(np.asarray(jax.jit(lambda m, x: m @ BinaryArray(x))(m, ones)), [2, 2])
        np.testing.assert_array_equal(np.asarray(jax.jit(lambda m, x: BinaryArray(x) @ m)(m, ones)), [2, 2])
# Both shards are CPU devices, so the store is keyed by one device kind.
assert kinds == [devices[0].device_kind] * 4, kinds
'''
    env = dict(os.environ, JAX_PLATFORM_NAME='cpu',
               XLA_FLAGS='--xla_force_host_platform_device_count=2')
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=Path(__file__).parents[3],
        env=env, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


# Empty structures and sparse object operations
# ---------------------------------------------

_empty_EMPTY_SHAPES = [(0, 0), (0, 3), (3, 0)]


def _empty_empty_csr(shape, *, backend='jax_raw', homogeneous=False):
    return CSR(
        jnp.ones((1 if homogeneous else 0,), jnp.float32),
        jnp.empty((0,), jnp.int32),
        jnp.zeros((shape[0] + 1,), jnp.int32),
        shape=shape, backend=backend, config=CSRConfig(),
    )


@pytest.mark.parametrize('shape', _empty_EMPTY_SHAPES)
@pytest.mark.parametrize('method', ['coo', 'numpy', 'gpu_column_block'])
@pytest.mark.parametrize('array_module', [np, jnp])
@pytest.mark.parametrize('include_perm', [False, True])
def test_empty_mirror_helpers(shape, method, array_module, include_perm):
    indptr = array_module.zeros((shape[0] + 1,), dtype=np.int32)
    indices = array_module.empty((0,), dtype=np.int32)
    csc_indptr, csc_indices, perm = csr_to_csc_index(
        indptr, indices, shape=shape, method=method, include_perm=include_perm,
    )
    np.testing.assert_array_equal(csc_indptr, np.zeros(shape[1] + 1, np.int32))
    assert csc_indices.shape == (0,) and csc_indices.dtype == np.int32
    assert csc_indptr.dtype == np.int32
    if include_perm:
        assert perm.shape == (0,) and perm.dtype == np.int32
    else:
        assert perm is None
    restored_indptr, restored_indices, restored_perm = csc_to_csr_index(
        csc_indptr, csc_indices, shape=shape, include_perm=include_perm,
    )
    np.testing.assert_array_equal(restored_indptr, indptr)
    np.testing.assert_array_equal(restored_indices, indices)
    assert (restored_perm is None) == (not include_perm)


@pytest.mark.parametrize('shape', _empty_EMPTY_SHAPES)
def test_empty_csr_builds_mirror_eagerly_and_under_jit(shape):
    matrix = _empty_empty_csr(shape)
    # Exercise the traced helper before the eager call can populate the cache.
    traced = jax.jit(lambda value: value.build_weight_indices())(matrix)
    eager = matrix.build_weight_indices()
    for prepared in (eager, traced):
        indptr, indices, perm = prepared.buffers['csc']
        np.testing.assert_array_equal(indptr, np.zeros(shape[1] + 1, np.int32))
        assert indices.shape == perm.shape == (0,)
        np.testing.assert_array_equal(prepared.tocsc().tocsr().todense(), np.zeros(shape))


@pytest.mark.parametrize('shape', _empty_EMPTY_SHAPES)
def test_empty_csr_diagonal_addition_eagerly_and_under_jit(shape):
    matrix = _empty_empty_csr(shape)
    diagonal = jnp.empty((0,), jnp.float32)
    eager = matrix.diag_add(diagonal)
    prepared = matrix.prepare_diag_add()
    traced = jax.jit(lambda value, diag: value.diag_add(diag))(prepared, diagonal)
    for result in (eager, traced):
        assert result.shape == shape and result.nse == 0
        np.testing.assert_array_equal(result.indptr, np.zeros(shape[0] + 1, np.int32))
        np.testing.assert_array_equal(result.todense(), np.zeros(shape))


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='native CUDA dispatch requires a GPU')
@pytest.mark.parametrize('shape', _empty_EMPTY_SHAPES)
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('homogeneous', [False, True])
def test_empty_csr_cuda_event_products(shape, weighted, homogeneous):
    matrix = _empty_empty_csr(shape, backend='cuda_raw', homogeneous=homogeneous)
    event_type = WeightedArray if weighted else BinaryArray
    dtype = jnp.float32 if weighted else jnp.bool_
    for batch in (None, 2):
        right_shape = (shape[1],) if batch is None else (shape[1], batch)
        left_shape = (shape[0],) if batch is None else (batch, shape[0])
        right = event_type(jnp.ones(right_shape, dtype))
        left = event_type(jnp.ones(left_shape, dtype))
        right_result_shape = (shape[0],) if batch is None else (shape[0], batch)
        left_result_shape = (shape[1],) if batch is None else (batch, shape[1])
        # The eager unfavorable product must build the empty mirror itself.
        np.testing.assert_array_equal(matrix @ right, np.zeros(right_result_shape))
        np.testing.assert_array_equal(left @ matrix, np.zeros(left_result_shape))
        np.testing.assert_array_equal(
            jax.jit(lambda value, events: value @ events)(matrix, right),
            np.zeros(right_result_shape),
        )
        np.testing.assert_array_equal(
            jax.jit(lambda value, events: events @ value)(matrix, left),
            np.zeros(left_result_shape),
        )


# Sparse object event-value gradient routing
# --------------------------------------

_GRAD_ROWS = np.array([0, 0, 1, 1])


_GRAD_COLS = np.array([0, 2, 1, 2])


_GRAD_DENSE = np.array(
    [[1., 0., 2.], [0., 3., 4.]], dtype=np.float32)


def _grad_matrix(homogeneous):
    data = jnp.array([1.] if homogeneous else [1., 2., 3., 4.], dtype=jnp.float32)
    return CSR(
        data,
        jnp.array(_GRAD_COLS, dtype=jnp.int32),
        jnp.array([0, 2, 4], dtype=jnp.int32),
        shape=(2, 3), backend="jax_raw",
    )


@pytest.mark.parametrize("weighted", [False, True], ids=["binary", "weighted"])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("op", ["right_mv", "left_mv", "right_mm", "left_mm"])
def test_event_forward_and_weight_cotangent(weighted, homogeneous, op):
    """BinaryArray uses its activity mask, WeightedArray its values, in both passes."""
    matrix = _grad_matrix(homogeneous)
    wrap = WeightedArray if weighted else BinaryArray
    if op == "right_mv":
        x = jnp.array([2., 0., -1.], dtype=jnp.float32)
        ct = jnp.array([5., 7.], dtype=jnp.float32)
        factor = np.asarray(x)[_GRAD_COLS]
    elif op == "left_mv":
        x = jnp.array([2., -1.], dtype=jnp.float32)
        ct = jnp.array([5., 6., 7.], dtype=jnp.float32)
        factor = np.asarray(x)[_GRAD_ROWS]
    elif op == "right_mm":
        x = jnp.array([[2., 0.], [0., 3.], [-1., 2.]], dtype=jnp.float32)
        ct = jnp.array([[5., 6.], [7., 8.]], dtype=jnp.float32)
        factor = np.asarray(x)[_GRAD_COLS]
    else:
        x = jnp.array([[2., -1.], [0., 3.]], dtype=jnp.float32)
        ct = jnp.array([[5., 6., 7.], [8., 9., 10.]], dtype=jnp.float32)
        factor = np.asarray(x)[:, _GRAD_ROWS]
    left = op.startswith("left")
    product = (lambda m: wrap(x) @ m) if left else (lambda m: m @ wrap(x))
    values = np.asarray(x) if weighted else (np.asarray(x) != 0).astype(np.float32)
    dense = _GRAD_DENSE
    if homogeneous:
        dense = np.zeros_like(_GRAD_DENSE)
        dense[_GRAD_ROWS, _GRAD_COLS] = 1
    expected = values @ dense if left else dense @ values
    np.testing.assert_allclose(product(matrix), expected)
    if not weighted:
        factor = (factor != 0).astype(np.float32)
    if op == "right_mv":
        slots = factor * np.asarray(ct)[_GRAD_ROWS]
    elif op == "left_mv":
        slots = factor * np.asarray(ct)[_GRAD_COLS]
    elif op == "right_mm":
        slots = np.sum(factor * np.asarray(ct)[_GRAD_ROWS], axis=1)
    else:
        slots = np.sum(factor * np.asarray(ct)[:, _GRAD_COLS], axis=0)
    expected_grad = np.array([np.sum(slots)], dtype=np.float32) if homogeneous else slots
    actual_grad = jax.grad(lambda data: jnp.vdot(product(matrix.with_data(data)), ct))(matrix.data)
    np.testing.assert_allclose(actual_grad, expected_grad)


@pytest.mark.parametrize("weighted", [False, True], ids=["binary", "weighted"])
@pytest.mark.parametrize("op", ["mv", "mm"])
def test_indexed_mirror_weight_cotangent_follows_event_type(
    monkeypatch, op, weighted,
):
    monkeypatch.setattr(csr_main, "_use_cuda_indexed_binary_route", lambda *args: True)
    matrix = _grad_matrix(False).build_weight_indices()
    wrap = WeightedArray if weighted else BinaryArray
    x = jnp.array([2., 0., -1.], dtype=jnp.float32)
    if op == "mm":
        x = jnp.stack([x, 2 * x], axis=1)
        ct = jnp.array([[5., 6.], [7., 8.]], dtype=jnp.float32)
    else:
        ct = jnp.array([5., 7.], dtype=jnp.float32)
    values = np.asarray(x) if weighted else (np.asarray(x) != 0).astype(np.float32)
    if op == "mm":
        slots = np.sum(values[_GRAD_COLS] * np.asarray(ct)[_GRAD_ROWS], axis=1)
    else:
        slots = values[_GRAD_COLS] * np.asarray(ct)[_GRAD_ROWS]
    product = lambda m: m @ wrap(x)
    np.testing.assert_allclose(product(matrix), _GRAD_DENSE @ values)
    actual = jax.grad(lambda data: jnp.vdot(product(matrix.with_data(data)), ct))(matrix.data)
    np.testing.assert_allclose(actual, slots)


# Sparse object selection of direct and indexed primitives
# --------------------------------------------------------

def test_csr_unfavorable_binary_matvec_uses_direct_route_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSR.fromdense(dense, backend="jax_raw")
    events = jnp.array([True, False])
    buffer_keys = set(m.buffers)

    got = m @ BinaryArray(events)
    got_again = m @ BinaryArray(events)

    assert got.shape == (2,)
    assert jnp.allclose(got, dense @ events.astype(jnp.float32))
    assert jnp.allclose(got_again, got)
    assert m.buffers.get("csc") is None  # no mirror on jax_raw
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csr_unfavorable_binary_matmat_uses_direct_route_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSR.fromdense(dense, backend="jax_raw")
    events = jnp.array([[True, False], [False, True]])
    buffer_keys = set(m.buffers)

    got = m @ BinaryArray(events)
    got_again = m @ BinaryArray(events)

    assert got.shape == (2, 2)
    assert jnp.allclose(got, dense @ events.astype(jnp.float32))
    assert jnp.allclose(got_again, got)
    assert m.buffers.get("csc") is None  # no mirror on jax_raw
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csr_direct_binary_rmatmat_leaves_buffers_untouched():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSR.fromdense(dense, backend="jax_raw")
    events = jnp.array([[True, False], [False, True]])
    buffer_keys = set(m.buffers)

    got = BinaryArray(events) @ m
    got_again = BinaryArray(events) @ m

    assert got.shape == (2, 2)
    assert jnp.allclose(got, events.astype(jnp.float32) @ dense)
    assert jnp.allclose(got_again, got)
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csc_unfavorable_binary_rmatvec_uses_direct_route_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSC.fromdense(dense, backend="jax_raw")
    events = jnp.array([True, False])
    buffer_keys = set(m.buffers)

    got = BinaryArray(events) @ m
    got_again = BinaryArray(events) @ m

    assert got.shape == (2,)
    assert jnp.allclose(got, events.astype(jnp.float32) @ dense)
    assert jnp.allclose(got_again, got)
    assert m.buffers.get("csr") is None  # no mirror on jax_raw
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csc_direct_binary_matmat_leaves_buffers_untouched():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSC.fromdense(dense, backend="jax_raw")
    events = jnp.array([[True, False], [False, True]])
    buffer_keys = set(m.buffers)

    got = m @ BinaryArray(events)
    got_again = m @ BinaryArray(events)

    assert got.shape == (2, 2)
    assert jnp.allclose(got, dense @ events.astype(jnp.float32))
    assert jnp.allclose(got_again, got)
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csc_unfavorable_binary_rmatmat_uses_direct_route_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSC.fromdense(dense, backend="jax_raw")
    events = jnp.array([[True, False], [False, True]])
    buffer_keys = set(m.buffers)

    got = BinaryArray(events) @ m
    got_again = BinaryArray(events) @ m

    assert got.shape == (2, 2)
    assert jnp.allclose(got, events.astype(jnp.float32) @ dense)
    assert jnp.allclose(got_again, got)
    assert m.buffers.get("csr") is None  # no mirror on jax_raw
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_cuda_raw_unfavorable_binary_routes_use_indexed_mirror(monkeypatch):
    dense = jnp.array([[0.0, 2.0, 5.0], [3.0, 4.0, 0.0]], dtype=jnp.float32)
    csr = CSR.fromdense(dense, backend="cuda_raw")
    csc = CSC.fromdense(dense, backend="cuda_raw")
    calls = []

    def fail_direct(*args, **kwargs):
        raise AssertionError("cuda_raw unfavorable BinaryArray route must use indexed primitive")

    def fake_mv_indexed(data, indices, indptr, perm, vector, *, shape, transpose, backend=None, weighted=False,
                        config=None):
        calls.append({
            "kind": "mv",
            "shape": shape,
            "transpose": transpose,
            "backend": backend,
            "indptr": indptr,
            "perm": perm,
            "config": config,
        })
        out_size = shape[1] if transpose else shape[0]
        return jnp.zeros((out_size,), dtype=data.dtype)

    def fake_mm_indexed(data, indices, indptr, perm, matrix, *, shape, transpose, backend=None, weighted=False,
                        config=None):
        calls.append({
            "kind": "mm",
            "shape": shape,
            "transpose": transpose,
            "backend": backend,
            "indptr": indptr,
            "perm": perm,
            "config": config,
        })
        out_size = shape[1] if transpose else shape[0]
        return jnp.zeros((out_size, matrix.shape[1]), dtype=data.dtype)

    monkeypatch.setattr(csr_main, "binary_csrmv", fail_direct)
    monkeypatch.setattr(csr_main, "binary_csrmm", fail_direct)
    monkeypatch.setattr(csr_main, "binary_csrmv_indexed", fake_mv_indexed)
    monkeypatch.setattr(csr_main, "binary_csrmm_indexed", fake_mm_indexed)

    assert (csr @ BinaryArray(jnp.array([True, False, True]))).shape == (2,)
    assert (csr @ BinaryArray(jnp.array([[True, False], [False, True], [True, True]]))).shape == (2, 2)
    assert (BinaryArray(jnp.array([True, False])) @ csc).shape == (3,)
    assert (BinaryArray(jnp.array([[True, False], [False, True]])) @ csc).shape == (2, 3)

    assert [(call["kind"], call["shape"], call["transpose"], call["backend"]) for call in calls] == [
        ("mv", (3, 2), True, "cuda_raw"),
        ("mm", (3, 2), True, "cuda_raw"),
        ("mv", (2, 3), True, "cuda_raw"),
        ("mm", (2, 3), True, "cuda_raw"),
    ]
    # Each route traverses the matrix's cached mirror structure.
    csr_mirror = csr._weight_indices()
    csc_mirror = csc._weight_indices()
    for call, mirror in zip(calls, (csr_mirror, csr_mirror, csc_mirror, csc_mirror)):
        assert call["indptr"] is mirror[0]
        assert call["perm"] is mirror[2]


# Executable reuse across redrawn structures
# ------------------------------------------

def _redrawn_csr(seed, rows, cols, nnz):
    rng = np.random.default_rng(seed)
    slots = np.sort(rng.choice(rows * cols, nnz, replace=False))
    counts = np.bincount(slots // cols, minlength=rows)
    indptr = np.r_[0, np.cumsum(counts)].astype(np.int32)
    weights = jnp.asarray(rng.standard_normal(nnz), jnp.float32)
    matrix = CSR((weights, jnp.asarray(slots % cols, jnp.int32), jnp.asarray(indptr)), shape=(rows, cols))
    return matrix, int(counts.max())


@pytest.mark.parametrize('binary,matmat', [(False, False), (False, True), (True, False)],
                         ids=['float_vector', 'float_matrix', 'binary_vector'])
@pytest.mark.parametrize('rows,cols,nnz', [(400, 300, 6000), (40, 3000, 24000)], ids=['short_rows', 'long_rows'])
def test_redrawn_structures_with_equal_shape_and_nnz_share_one_executable(rows, cols, nnz, binary, matmat):
    # Kernels are planned from static shapes only, so a fresh random structure
    # of the same shape and nnz reuses the cached executable. (Binary MM over
    # redraws is covered by ``binary/test/shape_plan_test.py``.)
    if binary:
        values = jnp.asarray(np.random.default_rng(7).random(rows) < 0.5)
        product = jax.jit(lambda matrix, events: BinaryArray(events) @ matrix)
        events = values
        dense_events = np.asarray(values, np.float32)
    else:
        product = jax.jit(lambda matrix, events: events @ matrix)
        events = jnp.ones((2, rows) if matmat else (rows,), jnp.float32)
        dense_events = np.asarray(events)
    longest = set()
    for seed in range(5):
        matrix, maximum = _redrawn_csr(seed, rows, cols, nnz)
        longest.add(maximum)
        expected = dense_events @ np.asarray(matrix.todense())
        np.testing.assert_allclose(product(matrix, events), expected, rtol=1e-4, atol=1e-4)
    assert len(longest) > 1
    assert product._cache_size() == 1



def _csr_from_row_lengths(lengths, cols):
    lengths = np.asarray(lengths)
    indptr = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    indices = np.concatenate([np.arange(n) for n in lengths]).astype(np.int32)
    weights = jnp.ones(int(indptr[-1]), jnp.float32)
    return CSR((weights, jnp.asarray(indices), jnp.asarray(indptr)), shape=(len(lengths), cols))


def test_few_long_rows_share_one_executable_across_task_counts():
    # Same shape and nnz, but the long rows split into 7 vs 11 tasks of 512
    # edges. The event-driven scatter sizes its task queue from shapes alone,
    # so these structures share one pytree structure and one executable.
    cols = 4096
    a = _csr_from_row_lengths([3584] + [8] * 999, cols)                          # 7 tasks
    b = _csr_from_row_lengths([2560] + [65] * 6 + [9] * 682 + [8] * 311, cols)  # 5 + 6 tasks
    assert a.nse == b.nse and a.shape == b.shape
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    product = jax.jit(lambda matrix, events: BinaryArray(events) @ matrix)
    events = jnp.ones((a.shape[0],), bool)
    for matrix in (a, b):
        expected = np.asarray(events, np.float32) @ np.asarray(matrix.todense())
        np.testing.assert_allclose(product(matrix, events), expected, rtol=1e-5)
    assert product._cache_size() == 1
