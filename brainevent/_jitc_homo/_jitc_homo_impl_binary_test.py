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


import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import pytest

import brainevent
from brainevent._test_util import allclose, gen_events, ones_like


# brainevent.config.gpu_kernel_backend = 'warp'

class Test_JITC_RC_Conversion:

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matvec(self, shape, corder, asbool):
        jitcr = brainevent.JITCHomoR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[1], asbool=asbool)

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_vecmat(self, shape, corder, asbool):
        jitcr = brainevent.JITCHomoR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[0], asbool=asbool)

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_jitmat(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCHomoR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([shape[1], k], asbool=asbool)

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matjit(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCHomoR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events((k, shape[0]), asbool=asbool)

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.T).T
        print(out1 - out2)
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)


class Test_JITCHomoR:
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec(self, prob, weight, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape)
        vector = gen_events(shape[1])
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))

    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat(self, prob, weight, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape)
        vector = gen_events(shape[0])
        out1 = vector @ jitc
        out2 = vector @ jitc.todense()
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat(self, prob, weight, shape, k):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape)
        matrix = gen_events([shape[1], k])
        out1 = jitc @ matrix
        out2 = jitc.todense() @ matrix
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit(self, k, prob, weight, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape)
        matrix = gen_events((k, shape[0]))
        out1 = matrix @ jitc
        out2 = matrix @ jitc.todense()
        print(out1 - out2)
        assert u.math.allclose(out1, out2, rtol=1e-4 * u.get_unit(out1), atol=1e-4 * u.get_unit(out1))

    def test_todense_weight_batching(self):
        def f(weight):
            jitc = brainevent.JITCHomoR((weight, 0.1, 123), shape=(100, 50))
            return jitc.todense()

        weights = brainstate.random.rand(10)

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    def test_todense_prob_batching(self):
        def f(prob):
            jitc = brainevent.JITCHomoR((1.5, prob, 123), shape=(100, 50))
            return jitc.todense()

        probs = brainstate.random.rand(10)

        matrices = jax.vmap(f)(probs)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, probs)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    def test_todense_seed_batching(self):
        def f(seed):
            jitc = brainevent.JITCHomoR((1.5, 0.1, seed), shape=(100, 50))
            return jitc.todense()

        seeds = brainstate.random.randint(0, 100000, 10)

        matrices = jax.vmap(f)(seeds)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, seeds)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)


class Test_JITCHomoR_Gradients:

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_jvp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder)
        x = gen_events(shape[1], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_vjp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder)
        x = gen_events(shape[1], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_jvp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder)
        x = gen_events(shape[0], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_vjp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder)
        x = gen_events(shape[0], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_jvp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([shape[1], k], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_vjp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([shape[1], k], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_jvp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([k, shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_vjp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([k, shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))


class Test_JITCHomoR_Batching:
    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_vector(self, batch_size, shape, corder):
        vectors = gen_events([batch_size, shape[1]])

        def f(vector):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f)(vectors)
        assert matrices.shape == (batch_size, shape[0])

        matrices_loop = brainstate.transform.for_loop(f, vectors)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_vector_axis1(self, batch_size, shape, corder):
        vectors = gen_events([shape[1], batch_size])

        def f(vector):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f, in_axes=1, out_axes=1)(vectors)
        assert matrices.shape == (shape[0], batch_size)

        matrices_loop = brainstate.transform.for_loop(f, vectors.T)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop.T)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_weight(self, batch_size, shape, corder):
        weights = brainstate.random.rand(batch_size)
        vector = gen_events(shape[1])

        def f(w):
            jitc = brainevent.JITCHomoR((w, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[0])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_vector(self, batch_size, shape, corder):
        vectors = gen_events([batch_size, shape[0]])

        def f(vector):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f)(vectors)
        assert matrices.shape == (batch_size, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, vectors)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_vector_axis1(self, batch_size, shape, corder):
        vectors = gen_events([shape[0], batch_size])

        def f(vector):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f, in_axes=1, out_axes=1)(vectors)
        assert matrices.shape == (shape[1], batch_size)

        matrices_loop = brainstate.transform.for_loop(f, vectors.T)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop.T)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_weight(self, batch_size, shape, corder):
        weights = brainstate.random.rand(batch_size)
        vector = gen_events(shape[0])

        def f(w):
            jitc = brainevent.JITCHomoR((w, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix(self, batch_size, k, shape, corder):
        matrices = gen_events([batch_size, shape[1], k])

        def f(mat):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        outs_loop = brainstate.transform.for_loop(f, matrices)
        assert outs_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, outs_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix_axis1(self, batch_size, k, shape, corder):
        matrices = gen_events([shape[1], batch_size, k])

        def f(mat):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f, in_axes=1)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, matrices.transpose(1, 0, 2))
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix_axis2(self, batch_size, k, shape, corder):
        matrices = gen_events([shape[1], k, batch_size])

        def f(mat):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f, in_axes=2)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, matrices.transpose(2, 0, 1))
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_weight(self, batch_size, k, shape, corder):
        weights = brainstate.random.rand(batch_size)
        matrix = gen_events([shape[1], k])

        def f(w):
            jitc = brainevent.JITCHomoR((w, 0.1, 123), shape=shape, corder=corder)
            return jitc @ matrix

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix(self, batch_size, k, shape, corder):
        matrix = gen_events([batch_size, k, shape[0]])

        def f(mat):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f)(matrix)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, matrix)
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix_axis1(self, batch_size, k, shape, corder):
        matrix = gen_events([k, batch_size, shape[0]])

        def f(mat):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f, in_axes=1)(matrix)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, matrix.transpose(1, 0, 2))
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix_axis2(self, batch_size, k, shape, corder):
        matrix = gen_events([k, shape[0], batch_size])

        def f(mat):
            jitc = brainevent.JITCHomoR((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f, in_axes=2)(matrix)
        assert matrices.shape == (batch_size, k, shape[1],)

        matrices_loop = brainstate.transform.for_loop(f, matrix.transpose(2, 0, 1))
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_weight(self, batch_size, k, shape, corder):
        weights = brainstate.random.rand(batch_size)
        mat = gen_events([k, shape[0]])

        def f(w):
            jitc = brainevent.JITCHomoR((w, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)


class Test_JITCHomoR_Transpose:
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec(self, prob, weight, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape).T
        vector = gen_events([shape[0]])
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat(self, prob, weight, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape).T
        vector = gen_events(shape[1])
        out1 = vector @ jitc
        out2 = vector @ jitc.todense()
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat(self, prob, weight, shape, k):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape).T
        matrix = gen_events([shape[0], k], asbool=False)
        out1 = jitc @ matrix
        out2 = jitc.todense() @ matrix
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit(self, k, prob, weight, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape).T
        matrix = gen_events([k, shape[1]])
        out1 = matrix @ jitc
        out2 = matrix @ jitc.todense()
        assert u.math.allclose(out1, out2)


class Test_JITCHomoR_Transpose_Gradients:
    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_jvp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_vjp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_jvp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_vjp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_jvp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[0], k], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_vjp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[0], k], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_jvp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([k, shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_vjp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoR((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoR((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([k, shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))


class Test_JITCHomoC:
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec(self, prob, weight, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape)
        vector = gen_events([shape[1]])
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat(self, prob, weight, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape)
        vector = gen_events([shape[0]])
        out1 = vector @ jitc
        out2 = vector @ jitc.todense()
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat(self, prob, weight, shape, k):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape)
        matrix = gen_events([shape[1], k])
        out1 = jitc @ matrix
        out2 = jitc.todense() @ matrix
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit(self, k, prob, weight, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape)
        matrix = gen_events([k, shape[0]], asbool=False)
        out1 = matrix @ jitc
        out2 = matrix @ jitc.todense()
        assert u.math.allclose(out1, out2)

    def test_todense_weight_batching(self):
        def f(weight):
            jitc = brainevent.JITCHomoC((weight, 0.1, 123), shape=(100, 50))
            return jitc.todense()

        weights = brainstate.random.rand(10)

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    def test_todense_prob_batching(self):
        def f(prob):
            jitc = brainevent.JITCHomoC((1.5, prob, 123), shape=(100, 50))
            return jitc.todense()

        probs = brainstate.random.rand(10)

        matrices = jax.vmap(f)(probs)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, probs)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)

    def test_todense_seed_batching(self):
        def f(seed):
            jitc = brainevent.JITCHomoC((1.5, 0.1, seed), shape=(100, 50))
            return jitc.todense()

        seeds = brainstate.random.randint(0, 100000, 10)

        matrices = jax.vmap(f)(seeds)
        assert matrices.shape == (10, 100, 50)

        matrices_loop = brainstate.transform.for_loop(f, seeds)
        assert matrices_loop.shape == (10, 100, 50)

        assert u.math.allclose(matrices, matrices_loop)


class Test_JITCHomoC_Gradients:

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_jvp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_vjp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_jvp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_vjp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_jvp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([shape[1], k], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_vjp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([shape[1], k], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_jvp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([k, shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_vjp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder)
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder)
        x = gen_events([k, shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))


class Test_JITCHomoC_Batching:
    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_vector(self, batch_size, shape, corder):
        vectors = gen_events([batch_size, shape[1]])

        def f(vector):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f)(vectors)
        assert matrices.shape == (batch_size, shape[0])

        matrices_loop = brainstate.transform.for_loop(f, vectors)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_vector_axis1(self, batch_size, shape, corder):
        vectors = gen_events([shape[1], batch_size])

        def f(vector):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f, in_axes=1, out_axes=1)(vectors)
        assert matrices.shape == (shape[0], batch_size)

        matrices_loop = brainstate.transform.for_loop(f, vectors.T)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop.T)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec_batching_weight(self, batch_size, shape, corder):
        weights = brainstate.random.rand(batch_size)
        vector = gen_events([shape[1]])

        def f(w):
            jitc = brainevent.JITCHomoC((w, 0.1, 123), shape=shape, corder=corder)
            return jitc @ vector

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[0])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[0])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_vector(self, batch_size, shape, corder):
        vectors = gen_events([batch_size, shape[0]])

        def f(vector):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f)(vectors)
        assert matrices.shape == (batch_size, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, vectors)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_vector_axis1(self, batch_size, shape, corder):
        vectors = gen_events([shape[0], batch_size])

        def f(vector):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f, in_axes=1, out_axes=1)(vectors)
        assert matrices.shape == (shape[1], batch_size)

        matrices_loop = brainstate.transform.for_loop(f, vectors.T)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop.T)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_batching_weight(self, batch_size, shape, corder):
        weights = brainstate.random.rand(batch_size)
        vector = gen_events([shape[0]])

        def f(w):
            jitc = brainevent.JITCHomoC((w, 0.1, 123), shape=shape, corder=corder)
            return vector @ jitc

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix(self, batch_size, k, shape, corder):
        matrices = gen_events([batch_size, shape[1], k])

        def f(mat):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        outs_loop = brainstate.transform.for_loop(f, matrices)
        assert outs_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, outs_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix_axis1(self, batch_size, k, shape, corder):
        matrices = gen_events([shape[1], batch_size, k])

        def f(mat):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f, in_axes=1)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, matrices.transpose(1, 0, 2))
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_matrix_axis2(self, batch_size, k, shape, corder):
        matrices = gen_events([shape[1], k, batch_size])

        def f(mat):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return jitc @ mat

        outs = jax.vmap(f, in_axes=2)(matrices)
        assert outs.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, matrices.transpose(2, 0, 1))
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(outs, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_batching_weight(self, batch_size, k, shape, corder):
        weights = brainstate.random.rand(batch_size)
        matrix = gen_events([shape[1], k])

        def f(w):
            jitc = brainevent.JITCHomoC((w, 0.1, 123), shape=shape, corder=corder)
            return jitc @ matrix

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, shape[0], k)

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, shape[0], k)

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix(self, batch_size, k, shape, corder):
        matrix = gen_events([batch_size, k, shape[0]])

        def f(mat):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f)(matrix)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, matrix)
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix_axis1(self, batch_size, k, shape, corder):
        matrix = gen_events([k, batch_size, shape[0]])

        def f(mat):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f, in_axes=1)(matrix)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, matrix.transpose(1, 0, 2))
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_matrix_axis2(self, batch_size, k, shape, corder):
        matrix = gen_events([k, shape[0], batch_size])

        def f(mat):
            jitc = brainevent.JITCHomoC((1.05 * u.mA, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f, in_axes=2)(matrix)
        assert matrices.shape == (batch_size, k, shape[1],)

        matrices_loop = brainstate.transform.for_loop(f, matrix.transpose(2, 0, 1))
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)

    @pytest.mark.parametrize('batch_size', [10, 15])
    @pytest.mark.parametrize('k', [5])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_batching_weight(self, batch_size, k, shape, corder):
        weights = brainstate.random.rand(batch_size)
        mat = gen_events([k, shape[0]])

        def f(w):
            jitc = brainevent.JITCHomoC((w, 0.1, 123), shape=shape, corder=corder)
            return mat @ jitc

        matrices = jax.vmap(f)(weights)
        assert matrices.shape == (batch_size, k, shape[1])

        matrices_loop = brainstate.transform.for_loop(f, weights)
        assert matrices_loop.shape == (batch_size, k, shape[1])

        assert u.math.allclose(matrices, matrices_loop)


class Test_JITCHomoC_Transpose:
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec(self, prob, weight, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape).T
        vector = gen_events([shape[0]])
        out1 = jitc @ vector
        out2 = jitc.todense() @ vector
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat(self, prob, weight, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape).T
        vector = gen_events([shape[1]])
        out1 = vector @ jitc
        out2 = vector @ jitc.todense()
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat(self, prob, weight, shape, k):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape).T
        matrix = gen_events([shape[0], k], asbool=False)
        out1 = jitc @ matrix
        out2 = jitc.todense() @ matrix
        assert u.math.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('prob', [0.1, 0.2])
    @pytest.mark.parametrize('weight', [1.5, 2.1 * u.mV])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit(self, k, prob, weight, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape).T
        matrix = gen_events([k, shape[1]])
        out1 = matrix @ jitc
        out2 = matrix @ jitc.todense()
        assert u.math.allclose(out1, out2)


class Test_JITCHomoC_Transpose_Gradients:
    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_jvp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matvec_vjp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[0]], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_jvp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_vecmat_vjp(self, weight, prob, corder, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_jvp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[0], k], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_jitmat_vjp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([shape[0], k], asbool=False)

        def f_brainevent(x, w):
            return (jitc.with_data(w) @ x).sum()

        def f_dense(x, w):
            return ((dense * w) @ x).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_jvp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([k, shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, jvp_x1 = jax.jvp(
            f_brainevent,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        out2, jvp_x2 = jax.jvp(
            f_dense,
            (x, jnp.array(weight)),
            (ones_like(x), jnp.array(1.0))
        )

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(jvp_x1, jvp_x2, rtol=1e-4, atol=1e-4))

    @pytest.mark.parametrize('weight', [1.5])
    @pytest.mark.parametrize('prob', [0.1])
    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    def test_matjit_vjp(self, weight, prob, corder, k, shape):
        jitc = brainevent.JITCHomoC((weight, prob, 123), shape=shape, corder=corder).T
        dense = brainevent.JITCHomoC((1., prob, 123), shape=shape, corder=corder).T
        x = gen_events([k, shape[1]], asbool=False)

        def f_brainevent(x, w):
            return (x @ jitc.with_data(w)).sum()

        def f_dense(x, w):
            return (x @ (dense * w)).sum()

        out1, (vjp_x1, vjp_w1) = jax.value_and_grad(f_brainevent, argnums=(0, 1))(x, jnp.array(weight))
        out2, (vjp_x2, vjp_w2) = jax.value_and_grad(f_dense, argnums=(0, 1))(x, jnp.array(weight))

        assert (allclose(out1, out2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_x1, vjp_x2, rtol=1e-4, atol=1e-4))
        assert (allclose(vjp_w1, vjp_w2, rtol=1e-4, atol=1e-4))
