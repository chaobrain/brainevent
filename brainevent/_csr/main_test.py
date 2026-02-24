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


import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._csr.test_util import get_csr, vector_csr, matrix_csr, csr_vector, csr_matrix

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

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
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

    @pytest.mark.parametrize('homo_w', [True, False])
    @pytest.mark.parametrize('replace', [True, False])
    @pytest.mark.parametrize('transpose', [True, False])
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


class Test_CSR:
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_to_coo(self, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix)
        coo = csr.tocoo()
        dense = coo.todense()

        assert jnp.allclose(matrix, dense)

        jax.block_until_ready((matrix, dense))

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


class Test_CSC:
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_to_coo(self, shape):
        matrix = gen_sparse_matrix(shape)
        csc = brainevent.CSR.fromdense(matrix).T
        coo = csc.tocoo()
        dense = coo.todense()
        assert jnp.allclose(matrix.T, dense)

        jax.block_until_ready((dense,))

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_todense(self, shape):
        matrix = gen_sparse_matrix(shape)
        csr = brainevent.CSR.fromdense(matrix).T
        dense = csr.todense()
        assert jnp.allclose(matrix.T, dense)

        jax.block_until_ready((dense,))


class Test_diag_add:
    @pytest.mark.parametrize('shape', [(200, 300), (100, 50), (400, 400)])
    def test_csr(self, shape):
        dense = brainstate.random.rand(*shape)
        mask = dense < 0.1
        dense = jnp.where(mask, dense, 0.)
        csr = brainevent.CSR.fromdense(dense)
        diag = brainstate.random.rand(min(shape))
        new_csr = csr.diag_add(diag)

        new_dense = new_csr.todense()
        dense = dense.at[jnp.diag_indices(min(shape))].add(diag)
        dense = jnp.where(mask, dense, 0.)

        print(new_dense)
        print(dense)

        assert jnp.allclose(new_dense, dense)

        jax.block_until_ready((dense, diag, new_dense))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50), (400, 400)])
    def test_csc(self, shape):
        dense = brainstate.random.rand(*shape)
        mask = dense < 0.1
        dense = jnp.where(mask, dense, 0.)
        csc = brainevent.CSC.fromdense(dense)
        diag = brainstate.random.rand(min(shape))
        new_csr = csc.diag_add(diag)

        new_dense = new_csr.todense()
        dense = dense.at[jnp.diag_indices(min(shape))].add(diag)
        dense = jnp.where(mask, dense, 0.)

        print(new_dense)
        print(dense)

        assert jnp.allclose(new_dense, dense)

        jax.block_until_ready((dense, diag, new_dense))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50), (400, 400)])
    def test_csr_and_csc(self, shape):
        dense = brainstate.random.rand(*shape)
        mask = dense < 0.1
        dense = jnp.where(mask, dense, 0.)
        csr = brainevent.CSR.fromdense(dense)
        csc = csr.T
        diag = brainstate.random.rand(min(shape))
        new_csr = csc.diag_add(diag)

        new_dense = new_csr.todense().T
        dense = dense.at[jnp.diag_indices(min(shape))].add(diag)
        dense = jnp.where(mask, dense, 0.)

        print(new_dense)
        print(dense)

        assert jnp.allclose(new_dense, dense)

        jax.block_until_ready((dense, diag, new_dense))


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
