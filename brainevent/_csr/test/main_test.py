# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

from pathlib import Path

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent import CSR, CSC, BinaryArray
from brainevent._csr._test_util import (
    get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix, small_csr,
)
from brainevent._test_util import jax_x64_enabled

# The operator tests below dispatch to the native ``numba`` backend (the default backend),
# which compiles per test and dominates wall-clock, so each carries ``@pytest.mark.slow`` and
# the default ``pytest`` run skips it; CI runs them via ``pytest -m ""``. Kernel correctness on
# the cheap ``jax_raw`` backend is still covered by ``float_test.py`` in the default run. The
# marker is per-test rather than module-wide so the pure-Python structural tests further down
# (pytree contract, dtype policy) stay in the default run.

platform = jax.default_backend()
BINARY_CSRMV_IMPLEMENTATIONS = tuple(brainevent.binary_csrmv_p.available_backends(platform))
FLOAT_CSRMV_IMPLEMENTATIONS = tuple(brainevent.csrmv_p.available_backends(platform))
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


@pytest.mark.slow
class Test_diag_add:
    # ``diag_add`` computes ``A + diag(d)`` *exactly*: diagonal entries absent
    # from the sparsity pattern must be inserted, which changes ``indices`` /
    # ``indptr``. The expected dense result is therefore ``dense + diag(d)`` with
    # NO re-masking -- re-applying the sparsity mask (as a previous version of
    # these tests did) would have hidden the dropped-insertion bug.
    @pytest.mark.parametrize('shape', [(200, 300), (100, 50), (400, 400)])
    def test_csr(self, shape):
        dense = brainstate.random.rand(*shape)
        mask = (dense < 0.1) & (dense != 0.)
        dense = jnp.where(mask, dense, 0.)
        csr = brainevent.CSR.fromdense(dense)
        diag = brainstate.random.rand(min(shape)).astype(dense.dtype)
        new_csr = csr.diag_add(diag)

        new_dense = new_csr.todense()
        expected = dense.at[jnp.diag_indices(min(shape))].add(diag)

        assert jnp.allclose(new_dense, expected)
        # Missing diagonals were inserted, so the structure grew.
        assert new_csr.nse >= csr.nse

        jax.block_until_ready((expected, diag, new_dense))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50), (400, 400)])
    def test_csc(self, shape):
        dense = brainstate.random.rand(*shape)
        mask = (dense < 0.1) & (dense != 0.)
        dense = jnp.where(mask, dense, 0.)
        csc = brainevent.CSC.fromdense(dense)
        diag = brainstate.random.rand(min(shape)).astype(dense.dtype)
        new_csc = csc.diag_add(diag)

        new_dense = new_csc.todense()
        expected = dense.at[jnp.diag_indices(min(shape))].add(diag)

        assert jnp.allclose(new_dense, expected)
        assert new_csc.nse >= csc.nse

        jax.block_until_ready((expected, diag, new_dense))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50), (400, 400)])
    def test_csr_and_csc(self, shape):
        dense = brainstate.random.rand(*shape)
        mask = (dense < 0.1) & (dense != 0.)
        dense = jnp.where(mask, dense, 0.)
        csr = brainevent.CSR.fromdense(dense)
        csc = csr.T
        diag = brainstate.random.rand(min(shape)).astype(dense.dtype)
        new_csc = csc.diag_add(diag)

        new_dense = new_csc.todense().T
        expected = dense.at[jnp.diag_indices(min(shape))].add(diag)

        assert jnp.allclose(new_dense, expected)

        jax.block_until_ready((expected, diag, new_dense))

    # ------------------------------------------------------------------
    # Exactness and structural behaviour on small, explicit matrices.
    # ------------------------------------------------------------------
    def test_inserts_missing_diagonal_exact(self):
        # Diagonals (0, 0) and (1, 1) are missing; (2, 2) is present.
        dense = jnp.array([[0., 7., 0.],
                           [4., 0., 0.],
                           [0., 0., 5.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        diag = jnp.array([1., 2., 3.], dtype=jnp.float32)
        new_csr = csr.diag_add(diag)

        expected = jnp.array([[1., 7., 0.],
                              [4., 2., 0.],
                              [0., 0., 8.]], dtype=jnp.float32)
        assert jnp.allclose(new_csr.todense(), expected)
        # Two missing diagonals were inserted.
        assert int(new_csr.nse) == int(csr.nse) + 2
        # The full diagonal is now structurally present.
        assert int(new_csr.nse) == 5

    def test_inserted_indices_stay_sorted_within_row(self):
        # Row 0 has columns [2]; inserting the diagonal at column 0 must keep
        # the row's column indices ascending -> [0, 2].
        dense = jnp.array([[0., 0., 9.],
                           [0., 0., 0.],
                           [0., 0., 0.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        new_csr = csr.diag_add(jnp.array([1., 1., 1.], dtype=jnp.float32))

        indptr = np.asarray(new_csr.indptr)
        indices = np.asarray(new_csr.indices)
        for r in range(dense.shape[0]):
            row_cols = indices[indptr[r]:indptr[r + 1]]
            assert list(row_cols) == sorted(row_cols)
        assert jnp.allclose(new_csr.todense(),
                            jnp.array([[1., 0., 9.],
                                       [0., 1., 0.],
                                       [0., 0., 1.]], dtype=jnp.float32))

    def test_empty_rows_get_diagonal(self):
        # An entirely empty matrix becomes a pure diagonal matrix.
        n = 5
        csr = brainevent.CSR.fromdense(jnp.zeros((n, n), dtype=jnp.float32))
        diag = jnp.arange(1, n + 1, dtype=jnp.float32)
        new_csr = csr.diag_add(diag)
        assert int(new_csr.nse) == n
        assert jnp.allclose(new_csr.todense(), jnp.diag(diag))

    def test_full_diagonal_present_keeps_structure(self):
        # When every diagonal already exists, no insertion happens: the indices
        # and indptr are unchanged and only the values move.
        dense = jnp.array([[2., 1., 0.],
                           [0., 3., 1.],
                           [1., 0., 4.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        diag = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
        new_csr = csr.diag_add(diag)
        assert int(new_csr.nse) == int(csr.nse)
        np.testing.assert_array_equal(np.asarray(new_csr.indices),
                                      np.asarray(csr.indices))
        np.testing.assert_array_equal(np.asarray(new_csr.indptr),
                                      np.asarray(csr.indptr))
        assert jnp.allclose(new_csr.todense(),
                            dense.at[jnp.diag_indices(3)].add(diag))

    def test_zero_diag_value_still_materialises_full_diagonal(self):
        # Adding a zero diagonal does not change the dense values but still
        # materialises the (explicit-zero) diagonal entries -> structure grows.
        dense = jnp.array([[0., 2.],
                           [3., 0.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        new_csr = csr.diag_add(jnp.zeros(2, dtype=jnp.float32))
        assert int(new_csr.nse) == 4
        assert jnp.allclose(new_csr.todense(), dense)

    @pytest.mark.parametrize('shape', [(7, 7), (10, 4), (4, 10)])
    def test_csc_matches_csr(self, shape):
        dense = brainstate.random.rand(*shape)
        dense = jnp.where(dense < 0.3, dense, 0.).astype(jnp.float32)
        diag = brainstate.random.rand(min(shape)).astype(jnp.float32)
        csr_res = brainevent.CSR.fromdense(dense).diag_add(diag).todense()
        csc_res = brainevent.CSC.fromdense(dense).diag_add(diag).todense()
        assert jnp.allclose(csr_res, csc_res)
        assert jnp.allclose(csr_res, dense.at[jnp.diag_indices(min(shape))].add(diag))

    def test_diag_add_under_jit(self):
        dense = jnp.array([[0., 1., 0.],
                           [0., 0., 2.],
                           [3., 0., 0.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        diag = jnp.array([1., 1., 1.], dtype=jnp.float32)

        @jax.jit
        def run(mat, d):
            return mat.diag_add(d)

        # A jit argument has a traced structure, so the host-side plan must be
        # prepared beforehand.
        with pytest.raises(ValueError, match='prepare_diag_add'):
            run(csr, diag)
        new_csr = run(csr.prepare_diag_add(), diag)
        assert not isinstance(new_csr.indices, jax.core.Tracer)
        assert jnp.allclose(new_csr.todense(),
                            dense.at[jnp.diag_indices(3)].add(diag))
        # The result carries its own plan, so a second jitted diag_add works.
        again = run(new_csr, diag)
        assert jnp.allclose(again.todense(),
                            dense.at[jnp.diag_indices(3)].add(2 * diag))
        # A closed-over (concrete) matrix needs no preparation.
        closed = jax.jit(lambda d: csr.diag_add(d).todense())(diag)
        assert jnp.allclose(closed, dense.at[jnp.diag_indices(3)].add(diag))

    def test_repeated_diag_add_accumulates(self):
        dense = jnp.array([[0., 1.],
                           [2., 0.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        d = jnp.array([1., 1.], dtype=jnp.float32)
        r1 = csr.diag_add(d)
        r2 = r1.diag_add(d)
        # The second add operates on the already-augmented (full-diagonal)
        # structure, so the pattern is stable and values accumulate.
        np.testing.assert_array_equal(np.asarray(r1.indices), np.asarray(r2.indices))
        np.testing.assert_array_equal(np.asarray(r1.indptr), np.asarray(r2.indptr))
        assert jnp.allclose(r2.todense(), dense.at[jnp.diag_indices(2)].add(d * 2))

    def test_diag_add_preserves_units(self):
        dense = jnp.array([[0., 2.],
                           [3., 0.]], dtype=jnp.float32) * u.mV
        csr = brainevent.CSR.fromdense(dense)
        diag = jnp.array([1., 1.], dtype=jnp.float32) * u.mV
        new_csr = csr.diag_add(diag)
        expected = dense + u.math.diag(diag)
        assert u.get_unit(new_csr.todense()) == u.mV
        assert u.math.allclose(new_csr.todense(), expected, atol=1e-6 * u.mV)


@pytest.mark.slow
class Test_solve:
    @pytest.mark.parametrize('shape', [(200, 200), (400, 400)])
    def test_csr(self, shape: brainstate.typing.Shape):
        dense = brainstate.random.rand(*shape)
        mask = dense < 0.1
        dense = jnp.where(mask, dense, 0.)
        csr = brainevent.CSR.fromdense(dense)
        b = brainstate.random.randn(shape[0])

        x = csr.solve(b)
        assert jnp.allclose(csr @ x, b, atol=1e0, rtol=1e0)

        x2 = jnp.linalg.solve(dense, b)
        assert jnp.allclose(x, x2, atol=1e0, rtol=1e0)

        jax.block_until_ready((dense, b, x, x2))

    @pytest.mark.parametrize('shape', [(200, 200), (400, 400)])
    def test_csc(self, shape: brainstate.typing.Shape):
        dense = brainstate.random.rand(*shape)
        mask = dense < 0.1
        dense = jnp.where(mask, dense, 0.)
        csc = brainevent.CSR.fromdense(dense)
        b = brainstate.random.randn(shape[0])

        x = csc.solve(b)
        assert jnp.allclose(csc @ x, b, atol=1e0, rtol=1e0)

        x2 = jnp.linalg.solve(dense, b)
        assert jnp.allclose(x, x2, atol=1e0, rtol=1e0)

        jax.block_until_ready((dense, b, x, x2))

    @pytest.mark.parametrize('cls', [CSR, CSC])
    def test_solve_nonsymmetric_matches_dense(self, cls):
        # A non-symmetric, well-conditioned matrix: a solver that silently
        # solves ``W.T x = b`` gives a large residual here.
        dense = np.array([[4., 1., 0.], [0., 3., 2.], [1., 0., 5.]], dtype=np.float32)
        b = np.array([1., 2., 3.], dtype=np.float32)
        x = cls.fromdense(jnp.asarray(dense)).solve(jnp.asarray(b))
        np.testing.assert_allclose(dense @ np.asarray(x), b, atol=1e-5)
        np.testing.assert_allclose(np.asarray(x), np.linalg.solve(dense, b), atol=1e-5)

    def test_csr_solve_forwards_tol_and_reorder(self, monkeypatch):
        seen = {}

        def fake_solve(data, indices, indptr, b, tol=1e-6, reorder=1):
            seen.update(tol=tol, reorder=reorder)
            return b

        monkeypatch.setattr(brainevent._csr.main, 'csr_solve', fake_solve)
        csr = CSR.fromdense(jnp.eye(3))
        csr.solve(jnp.ones(3), tol=1e-3, reorder=0)
        assert seen == {'tol': 1e-3, 'reorder': 0}


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


# ---- weight-indices lifecycle (Phase 4) ----
import numpy as _np
import jax.numpy as _jnp
import brainevent as _be


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


# ---- event-driven dispatch via weight-indices (Phase 5) ----

def _rand_dense(rng, m, n, p=0.4):
    return _jnp.asarray((rng.random((m, n)) < p) * rng.random((m, n)), _jnp.float32)


@pytest.mark.slow
def test_csr_at_event_matches_dense_without_mirror_cache_by_default():
    # Default/JAX-style routing stays on the direct CSR binary path.
    rng = _np.random.default_rng(3)
    m, k = 5, 7
    csr = _be.CSR.fromdense(_rand_dense(rng, m, k))
    ev = _jnp.asarray(rng.random(k) > 0.5)
    got = csr @ _be.BinaryArray(ev)
    ref = csr.todense() @ ev.astype(_jnp.float32)
    assert got.shape == (m,)
    assert _jnp.allclose(got, ref, atol=1e-5)
    assert csr.buffers.get('csc') is None


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
def test_event_at_csc_matches_dense_without_mirror_cache_by_default():
    # Default/JAX-style routing stays on the direct CSC-as-transposed-CSR path.
    rng = _np.random.default_rng(5)
    m, n = 6, 4
    csc = _be.CSC.fromdense(_rand_dense(rng, m, n))
    ev = _jnp.asarray(rng.random(m) > 0.5)
    got = _be.BinaryArray(ev) @ csc
    ref = ev.astype(_jnp.float32) @ csc.todense()
    assert got.shape == (n,)
    assert _jnp.allclose(got, ref, atol=1e-5)
    assert csc.buffers.get('csr') is None


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


# ---- transpose cache hand-off (Phase 6) ----

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


# ---- OO plasticity method parity (Phase 8) ----

@pytest.mark.slow
def test_csr_update_on_pre_method_parity():
    rng = _np.random.default_rng(20)
    csr = _be.CSR.fromdense(_rand_dense(rng, 4, 5))
    pre_spike = _jnp.asarray(rng.random(4) > 0.5)
    post_trace = _jnp.asarray(rng.random(5), _jnp.float32)
    got = csr.update_on_pre(pre_spike, post_trace, 0.0, 1.5).data
    ref = _be.update_csr_on_binary_pre(
        csr.data, csr.indices, csr.indptr, pre_spike, post_trace, 0.0, 1.5, shape=csr.shape)
    assert _jnp.allclose(got, ref, atol=1e-6)


@pytest.mark.slow
def test_csr_update_on_post_method_parity():
    rng = _np.random.default_rng(21)
    csr = _be.CSR.fromdense(_rand_dense(rng, 4, 5))
    pre_trace = _jnp.asarray(rng.random(4), _jnp.float32)
    post_spike = _jnp.asarray(rng.random(5) > 0.5)
    got = csr.update_on_post(pre_trace, post_spike, 0.0, 1.5).data
    cscp, csci, perm = _be.csr_to_csc_index(csr.indptr, csr.indices, shape=csr.shape)
    ref = _be.update_csr_on_binary_post(
        csr.data, csci, cscp, perm, pre_trace, post_spike, 0.0, 1.5, shape=csr.shape)
    assert _jnp.allclose(got, ref, atol=1e-6)


@pytest.mark.slow
def test_csc_update_on_pre_method_parity():
    rng = _np.random.default_rng(22)
    csc = _be.CSC.fromdense(_rand_dense(rng, 5, 3))
    pre_spike = _jnp.asarray(rng.random(5) > 0.5)
    post_trace = _jnp.asarray(rng.random(3), _jnp.float32)
    got = csc.update_on_pre(pre_spike, post_trace, 0.0, 1.5).data
    ref = _be.update_csc_on_binary_pre(
        csc.data, csc.indices, csc.indptr, pre_spike, post_trace, 0.0, 1.5, shape=csc.shape)
    assert _jnp.allclose(got, ref, atol=1e-6)


@pytest.mark.slow
def test_csc_update_on_post_method_parity():
    rng = _np.random.default_rng(23)
    csc = _be.CSC.fromdense(_rand_dense(rng, 5, 3))
    pre_trace = _jnp.asarray(rng.random(5), _jnp.float32)
    post_spike = _jnp.asarray(rng.random(3) > 0.5)
    got = csc.update_on_post(pre_trace, post_spike, 0.0, 1.5).data
    ref = _be.update_csc_on_binary_post(
        csc.data, csc.indices, csc.indptr, pre_trace, post_spike, 0.0, 1.5, shape=csc.shape)
    assert _jnp.allclose(got, ref, atol=1e-6)


# --- merged from brainevent/_csr/mm_golden_parity_test.py ---
# Golden parity for binary CSR/CSC matrix-matrix dispatch against ``M.todense()``.


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
    # Only explicit cuda_raw uses the mirror indexed route that builds 'csc'.
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
    # Only explicit cuda_raw uses the mirror indexed route that builds 'csr'.
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


# ---------------------------------------------------------------------------
# Constructor dtype contract: ``indices`` always int32, ``indptr`` int32 unless
# nnz forces int64. The helper-level policy tests live in ``_misc_test.py``;
# these exercise it end-to-end through the public CSR/CSC constructors.
# Pure structural checks -- no kernel compilation, so not marked ``slow``.
# ---------------------------------------------------------------------------


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


# -- Constructor structural validation --------------------------------------

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


# -- Structure-preserving paths keep the contract and do not host-readback ---

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


# ---------------------------------------------------------------------------
# WeightedArray dispatch (#201): ``CSR``/``CSC`` ``@`` a value-weighted event
# operand is the exact product, on every backend and route (including the
# explicit ``cuda_raw`` indexed route for the unfavourable direction).
# ---------------------------------------------------------------------------

from brainevent import WeightedArray  # noqa: E402
from brainevent._csr._test_util import (  # noqa: E402
    WEIGHTED_EVENT_DTYPE_NAMES, event_dtype_context, event_mask, event_values,
)
from brainevent._test_util import check_event_op_grads  # noqa: E402

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


# ---------------------------------------------------------------------------
# Inactive events contribute exactly zero, whatever the weight (NaN/Inf).
# docs/specs/2026-10-24-inactive-row-nan-isolation.md
# ---------------------------------------------------------------------------

from brainevent._csr.binary.binary import binary_csrmv_p  # noqa: E402

_CLASS_BACKENDS = (None,) + tuple(binary_csrmv_p.available_backends(jax.default_backend()))


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


# ---------------------------------------------------------------------------
# Pytree / workspace contract (0.4.0): ``indices``/``indptr`` (and cached
# mirrors) are dynamic pytree leaves, and no lazily built buffer may mutate a
# matrix under a trace. Gradients go through ``M.data`` + ``with_data``.
# ---------------------------------------------------------------------------

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


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_task_capacity_covers_heavy_rows_in_every_orientation(cls, monkeypatch):
    # Regression: ``.T`` re-keyed only one direction of the cached workspace, so
    # ``events @ csr.T`` after ``events @ csr`` on cuda_raw ran with a task
    # queue sized for the other orientation and silently dropped work.
    from brainevent._csr.binary.hybrid_config import hybrid_task_capacity
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
        assert capacity >= hybrid_task_capacity(indptr), (name, capacity)


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
    # Regression: the public ``binary_csrmv``/``binary_csrmm`` required a
    # private ``workspace=`` built from a concrete ``indptr``.
    M, dense = _pytree_matrix(CSR, backend)
    D = dense.T if transpose else dense
    x = _spike_vector(D.shape[1])
    X = jnp.asarray(np.random.default_rng(5).random((D.shape[1], 3)) < 0.5)
    mv = jax.jit(lambda d, i, p, s: brainevent.binary_csrmv(
        d, i, p, s, shape=M.shape, transpose=transpose, backend=backend))
    mm = jax.jit(lambda d, i, p, s: brainevent.binary_csrmm(
        d, i, p, s, shape=M.shape, transpose=transpose, backend=backend))
    np.testing.assert_allclose(mv(M.data, M.indices, M.indptr, x), D @ np.asarray(x, np.float32), rtol=1e-5)
    np.testing.assert_allclose(mm(M.data, M.indices, M.indptr, X), D @ np.asarray(X, np.float32), rtol=1e-5)


def test_passing_a_workspace_is_deprecated():
    M, dense = _pytree_matrix(CSR, 'jax_raw')
    x = _spike_vector(M.shape[1])
    with pytest.warns(DeprecationWarning, match='workspace'):
        out = brainevent.binary_csrmv(M.data, M.indices, M.indptr, x, shape=M.shape, workspace=object())
    np.testing.assert_allclose(out, dense @ np.asarray(x, np.float32), rtol=1e-5)


@pytest.mark.parametrize('indexed', [False, True])
def test_understated_max_row_nnz_is_rejected_for_concrete_indptr(indexed):
    # An understated hint shrinks the cuda_raw task queue and silently drops
    # work, so it is rejected whenever ``indptr`` is concrete.
    M, dense = _pytree_matrix(CSR, 'jax_raw')
    longest = int(np.diff(np.asarray(M.indptr)).max())
    x = _spike_vector(M.shape[1])
    if indexed:
        perm = jnp.arange(M.nse)
        call = lambda hint: brainevent.binary_csrmv_indexed(
            M.data, M.indices, M.indptr, perm, x, shape=M.shape, max_row_nnz=hint)
    else:
        call = lambda hint: brainevent.binary_csrmv(M.data, M.indices, M.indptr, x, shape=M.shape, max_row_nnz=hint)
    with pytest.raises(ValueError, match='max_row_nnz'):
        call(longest - 1)
    np.testing.assert_allclose(call(longest), dense @ np.asarray(x, np.float32), rtol=1e-5)


@pytest.mark.parametrize('cls', [CSR, CSC])
def test_nnz_hint_is_a_power_of_two_upper_bound(cls):
    # The hint lives in the pytree metadata; rounding keeps re-drawn matrices
    # of one family on the same compilation.
    M, _ = _pytree_matrix(cls, 'jax_raw')
    longest = int(np.diff(np.asarray(M.indptr)).max())
    hint = jax.tree_util.tree_flatten(M)[1].node_data()[1][4]
    assert hint >= longest and hint & (hint - 1) == 0
    M2 = M.build_weight_indices()
    mirror_longest = int(np.diff(np.asarray(M2.buffers[M2._mirror_key][0])).max())
    mirror_hint = jax.tree_util.tree_flatten(M2)[1].node_data()[1][5]
    assert mirror_hint >= mirror_longest and mirror_hint & (mirror_hint - 1) == 0
