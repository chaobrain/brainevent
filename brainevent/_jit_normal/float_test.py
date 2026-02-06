# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


from typing import Tuple

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent


class Test_JITCNormalR:
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec(self, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.01, prob, 123), shape=shape)
        vector = jnp.asarray(np.random.rand(shape[1]))
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))

    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat(self, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.01, prob, 123), shape=shape)
        vector = jnp.asarray(np.random.rand(shape[0]))
        out1 = vector @ jitc
        out2 = vector @ jitc.todense()
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat(self, prob, weight, shape: Tuple[int, int], k):
        jitc = brainevent.JITCNormalR((weight, weight * 0.01, prob, 123), shape=shape)
        matrix = jnp.asarray(np.random.rand(shape[1], k))
        out1 = jitc @ matrix
        out2 = jitc.todense() @ matrix
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit(self, k, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.01, prob, 123), shape=shape)
        matrix = jnp.asarray(np.random.rand(k, shape[0]))
        out1 = matrix @ jitc
        out2 = matrix @ jitc.todense()
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))

    def test_todense_weight_batching(self):
        def f(weight):
            jitc = brainevent.JITCNormalR((weight, weight * 0.01, 0.1, 123), shape=(100, 50))
            return jitc.todense()

        weights = brainstate.random.rand(10)

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    def test_todense_prob_batching(self):
        def f(prob):
            jitc = brainevent.JITCNormalR((1.5, 0.1, prob, 123), shape=(100, 50))
            return jitc.todense()

        probs = brainstate.random.rand(10)

        matrices = jax.vmap(f)(probs)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, probs)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    def test_todense_seed_batching(self):
        def f(seed):
            jitc = brainevent.JITCNormalR((1.5, 0.1, 0.1, seed), shape=(100, 50))
            return jitc.todense()

        seeds = brainstate.random.randint(0, 100000, 10)

        matrices = jax.vmap(f)(seeds)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, seeds)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.skipif(
        brainstate.environ.get_platform() == 'cpu',
        reason="CPU no need to test large matrix."
    )
    def test_large_matrix(self):
        m = 10000
        jitc = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=(m, m))
        vector = jnp.asarray(np.random.rand(m))
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))


class Test_JITCNormalR_Gradients:

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_jvp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1]))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_vjp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1]))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x)
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x)

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_jvp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0]))

        def f_brainevent(x):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_vjp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_jvp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1], k))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_vjp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1], k))

        def f_brainevent(x, ):
            return (jitc @ x).sum()

        def f_dense(x, ):
            return (dense @ x).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_jvp(self, weight, prob, corder, k, shape: Tuple[int, int]):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(k, shape[0]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))
        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_vjp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(k, shape[0]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))


class Test_JITCNormalR_Batching:
    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_vector(self, batch_size, shape: Tuple[int, int], corder):
        vectors = brainstate.random.rand(batch_size, shape[1])

        def f(vector):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f)(vectors)
        assert matrices.shape == (batch_size, shape[0])

        matrices_loop = brainstate.transform.for_loop(f, vectors)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_vector_axis1(self, batch_size, shape: Tuple[int, int], corder):
        vectors = brainstate.random.rand(shape[1], batch_size)

        def f(vector):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f, in_axes=1, out_axes=1)(vectors)
        assert matrices.shape == (shape[0], batch_size)

        matrices_loop = brainstate.transform.for_loop(f, vectors.T)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop.T)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_weight(self, batch_size, shape: Tuple[int, int], corder):
        weights = brainstate.random.rand(batch_size)
        vector = brainstate.random.rand(shape[1], )

        def f(w):
            jitc = brainevent.JITCNormalR((w, 0.1, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[0])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_vector(self, batch_size, shape: Tuple[int, int], corder):
        vectors = brainstate.random.rand(batch_size, shape[0])

        def f(vector):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f)(vectors)
        assert matrices.shape == (batch_size, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, vectors)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_vector_axis1(self, batch_size, shape: Tuple[int, int], corder):
        vectors = brainstate.random.rand(shape[0], batch_size)

        def f(vector):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f, in_axes=1, out_axes=1)(vectors)
        assert matrices.shape == (shape[1], batch_size)

        matrices_loop = brainstate.transform.for_loop(f, vectors.T)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop.T)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_weight(self, batch_size, shape: Tuple[int, int], corder):
        weights = brainstate.random.rand(batch_size)
        vector = brainstate.random.rand(shape[0], )

        def f(w):
            jitc = brainevent.JITCNormalR((w, 0.1, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix(self, batch_size, k, shape: Tuple[int, int], corder):
        matrices = brainstate.random.rand(batch_size, shape[1], k)

        def f(mat):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        outs_loop = brainstate.transform.for_loop(f, matrices)
        assert outs_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, outs_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix_axis1(self, batch_size, k, shape: Tuple[int, int], corder):
        matrices = brainstate.random.rand(shape[1], batch_size, k)

        def f(mat):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f, in_axes=1)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, jnp.transpose(matrices, axes=(1, 0, 2)))
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix_axis2(self, batch_size, k, shape: Tuple[int, int], corder):
        matrices = brainstate.random.rand(shape[1], k, batch_size, )

        def f(mat):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f, in_axes=2)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, jnp.transpose(matrices, axes=(2, 0, 1)))
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_weight(self, batch_size, k, shape: Tuple[int, int], corder):
        weights = brainstate.random.rand(batch_size)
        matrix = brainstate.random.rand(shape[1], k)

        def f(w):
            jitc = brainevent.JITCNormalR((w, 0.1, 0.1, 123), shape=shape, corder=corder)
            return jitc @ matrix

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix(self, batch_size, k, shape: Tuple[int, int], corder):
        matrix = brainstate.random.rand(batch_size, k, shape[0])

        def f(mat):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f)(matrix)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, matrix)
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix_axis1(self, batch_size, k, shape: Tuple[int, int], corder):
        matrix = brainstate.random.rand(k, batch_size, shape[0])

        def f(mat):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f, in_axes=1)(matrix)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, jnp.transpose(matrix, (1, 0, 2)))
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix_axis2(self, batch_size, k, shape: Tuple[int, int], corder):
        matrix = brainstate.random.rand(k, shape[0], batch_size)

        def f(mat):
            jitc = brainevent.JITCNormalR((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f, in_axes=2)(matrix)
        assert matrices.shape == (batch_size, k, shape[1],)

        matrices_loop = brainstate.transform.for_loop(f, jnp.transpose(matrix, (2, 0, 1)))
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_weight(self, batch_size, k, shape: Tuple[int, int], corder):
        weights = brainstate.random.rand(batch_size)
        mat = brainstate.random.rand(k, shape[0], )

        def f(w):
            jitc = brainevent.JITCNormalR((w, 0.1, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)


class Test_JITCNormalR_Transpose:
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec(self, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape).T
        vector = jnp.asarray(np.random.rand(shape[0]))
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat(self, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape).T
        vector = jnp.asarray(np.random.rand(shape[1]))
        out1 = vector @ jitc
        out2 = vector @ jitc.todense()
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat(self, prob, weight, shape: Tuple[int, int], k):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape).T
        matrix = jnp.asarray(np.random.rand(shape[0], k))
        out1 = jitc @ matrix
        out2 = jitc.todense() @ matrix
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit(self, k, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape).T
        matrix = jnp.asarray(np.random.rand(k, shape[1]))
        out1 = matrix @ jitc
        out2 = matrix @ jitc.todense()
        assert u.math.allclose(out1, out2)


class Test_JITCNormalR_Transpose_Gradients:

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_jvp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0]))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_vjp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0]))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x)
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x)

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_jvp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1]))

        def f_brainevent(x):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_vjp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_jvp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0], k))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_vjp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0], k))

        def f_brainevent(x, ):
            return (jitc @ x).sum()

        def f_dense(x, ):
            return (dense @ x).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_jvp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(k, shape[1]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))
        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_vjp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalR((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(k, shape[1]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))


class Test_JITCNormalC:
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec(self, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape)
        vector = jnp.asarray(np.random.rand(shape[1]))
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat(self, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape)
        vector = jnp.asarray(np.random.rand(shape[0]))
        out1 = vector @ jitc
        out2 = vector @ jitc.todense()
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat(self, prob, weight, shape: Tuple[int, int], k):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape)
        matrix = jnp.asarray(np.random.rand(shape[1], k))
        out1 = jitc @ matrix
        out2 = jitc.todense() @ matrix
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit(self, k, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape)
        matrix = jnp.asarray(np.random.rand(k, shape[0]))
        out1 = matrix @ jitc
        out2 = matrix @ jitc.todense()
        assert u.math.allclose(out1, out2)

    def test_todense_weight_batching(self):
        def f(weight):
            jitc = brainevent.JITCNormalC((weight, 0.1, 0.1, 123), shape=(100, 50))
            return jitc.todense()

        weights = brainstate.random.rand(10)

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    def test_todense_prob_batching(self):
        def f(prob):
            jitc = brainevent.JITCNormalC((1.5, 0.1, prob, 123), shape=(100, 50))
            return jitc.todense()

        probs = brainstate.random.rand(10)

        matrices = jax.vmap(f)(probs)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, probs)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    def test_todense_seed_batching(self):
        def f(seed):
            jitc = brainevent.JITCNormalC((1.5, 0.1, 0.1, seed), shape=(100, 50))
            return jitc.todense()

        seeds = brainstate.random.randint(0, 100000, 10)

        matrices = jax.vmap(f)(seeds)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, seeds)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.skipif(
        brainstate.environ.get_platform() == 'cpu',
        reason="CPU no need to test large matrix."
    )
    def test_large_matrix(self):
        m = 10000
        jitc = brainevent.JITCNormalC((1.5, 0.1, 0.1, 123), shape=(m, m))
        vector = jnp.asarray(np.random.rand(m))
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))


class Test_JITCNormalC_Gradients:

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_jvp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1]))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_vjp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1]))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x)
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x)

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_jvp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0]))

        def f_brainevent(x):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_vjp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_jvp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1], k))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_vjp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1], k))

        def f_brainevent(x, ):
            return (jitc @ x).sum()

        def f_dense(x, ):
            return (dense @ x).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_jvp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(k, shape[0]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))
        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_vjp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder)
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(k, shape[0]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))


class Test_JITCNormalC_Batching:
    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_vector(self, batch_size, shape: Tuple[int, int], corder):
        vectors = brainstate.random.rand(batch_size, shape[1])

        def f(vector):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f)(vectors)
        assert matrices.shape == (batch_size, shape[0])

        matrices_loop = brainstate.transform.for_loop(f, vectors)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_vector_axis1(self, batch_size, shape: Tuple[int, int], corder):
        vectors = brainstate.random.rand(shape[1], batch_size)

        def f(vector):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f, in_axes=1, out_axes=1)(vectors)
        assert matrices.shape == (shape[0], batch_size)

        matrices_loop = brainstate.transform.for_loop(f, vectors.T)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop.T)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_weight(self, batch_size, shape: Tuple[int, int], corder):
        weights = brainstate.random.rand(batch_size)
        vector = brainstate.random.rand(shape[1], )

        def f(w):
            jitc = brainevent.JITCNormalC((w, 0.1, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[0])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_vector(self, batch_size, shape: Tuple[int, int], corder):
        vectors = brainstate.random.rand(batch_size, shape[0])

        def f(vector):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f)(vectors)
        assert matrices.shape == (batch_size, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, vectors)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_vector_axis1(self, batch_size, shape: Tuple[int, int], corder):
        vectors = brainstate.random.rand(shape[0], batch_size)

        def f(vector):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f, in_axes=1, out_axes=1)(vectors)
        assert matrices.shape == (shape[1], batch_size)

        matrices_loop = brainstate.transform.for_loop(f, vectors.T)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop.T)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_weight(self, batch_size, shape: Tuple[int, int], corder):
        weights = brainstate.random.rand(batch_size)
        vector = brainstate.random.rand(shape[0], )

        def f(w):
            jitc = brainevent.JITCNormalC((w, 0.1, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix(self, batch_size, k, shape: Tuple[int, int], corder):
        matrices = brainstate.random.rand(batch_size, shape[1], k)

        def f(mat):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        outs_loop = brainstate.transform.for_loop(f, matrices)
        assert outs_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, outs_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix_axis1(self, batch_size, k, shape: Tuple[int, int], corder):
        matrices = brainstate.random.rand(shape[1], batch_size, k)

        def f(mat):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f, in_axes=1)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, jnp.transpose(matrices, axes=(1, 0, 2)))
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix_axis2(self, batch_size, k, shape: Tuple[int, int], corder):
        matrices = brainstate.random.rand(shape[1], k, batch_size, )

        def f(mat):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f, in_axes=2)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, jnp.transpose(matrices, axes=(2, 0, 1)))
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_weight(self, batch_size, k, shape: Tuple[int, int], corder):
        weights = brainstate.random.rand(batch_size)
        matrix = brainstate.random.rand(shape[1], k)

        def f(w):
            jitc = brainevent.JITCNormalC((w, 0.1, 0.1, 123), shape=shape, corder=corder)
            return jitc @ matrix

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix(self, batch_size, k, shape: Tuple[int, int], corder):
        matrix = brainstate.random.rand(batch_size, k, shape[0])

        def f(mat):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f)(matrix)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, matrix)
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix_axis1(self, batch_size, k, shape: Tuple[int, int], corder):
        matrix = brainstate.random.rand(k, batch_size, shape[0])

        def f(mat):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f, in_axes=1)(matrix)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, jnp.transpose(matrix, (1, 0, 2)))
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix_axis2(self, batch_size, k, shape: Tuple[int, int], corder):
        matrix = brainstate.random.rand(k, shape[0], batch_size)

        def f(mat):
            jitc = brainevent.JITCNormalC((1.05 * u.mA, 0.1 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f, in_axes=2)(matrix)
        assert matrices.shape == (batch_size, k, shape[1],)

        matrices_loop = brainstate.transform.for_loop(f, jnp.transpose(matrix, (2, 0, 1)))
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_weight(self, batch_size, k, shape: Tuple[int, int], corder):
        weights = brainstate.random.rand(batch_size)
        mat = brainstate.random.rand(k, shape[0], )

        def f(w):
            jitc = brainevent.JITCNormalC((w, 0.1, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)


class Test_JITCNormalC_Transpose:
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec(self, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape).T
        vector = jnp.asarray(np.random.rand(shape[0]))
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat(self, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape).T
        vector = jnp.asarray(np.random.rand(shape[1]))
        out1 = vector @ jitc
        out2 = vector @ jitc.todense()
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat(self, prob, weight, shape: Tuple[int, int], k):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape).T
        matrix = jnp.asarray(np.random.rand(shape[0], k))
        out1 = jitc @ matrix
        out2 = jitc.todense() @ matrix
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit(self, k, prob, weight, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape).T
        matrix = jnp.asarray(np.random.rand(k, shape[1]))
        out1 = matrix @ jitc
        out2 = matrix @ jitc.todense()
        assert u.math.allclose(out1, out2)


class Test_JITCNormalC_Transpose_Gradients:

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_jvp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0]))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))
        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_vjp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0]))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x)
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x)

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_jvp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1]))

        def f_brainevent(x):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_vjp(self, weight, prob, corder, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[1]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_jvp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0], k))

        def f_brainevent(x):
            return (jitc @ x).sum()

        def f_dense(x):
            return (dense @ x).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))

        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_vjp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(shape[0], k))

        def f_brainevent(x, ):
            return (jitc @ x).sum()

        def f_dense(x, ):
            return (dense @ x).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_jvp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(k, shape[1]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, jvp_x1 = jax.jvp(f_brainevent, (x,), (jnp.ones_like(x),))
        out2, jvp_x2 = jax.jvp(f_dense, (x,), (jnp.ones_like(x),))

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_vjp(self, weight, prob, corder, k, shape: Tuple[int, int], ):
        jitc = brainevent.JITCNormalC((weight, weight * 0.1, prob, 123), shape=shape, corder=corder).T
        dense = jitc.todense()
        x = jnp.asarray(np.random.rand(k, shape[1]))

        def f_brainevent(x, ):
            return (x @ jitc).sum()

        def f_dense(x, ):
            return (x @ dense).sum()

        out1, (vjp_x1,) = jax.value_and_grad(f_brainevent, argnums=(0,))(x, )
        out2, (vjp_x2,) = jax.value_and_grad(f_dense, argnums=(0,))(x, )

        assert (jnp.allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (jnp.allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
