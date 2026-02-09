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
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._test_util import allclose, gen_events

platform = jax.default_backend()

if platform == 'cpu':
    shapes = [
        (200, 300),
        (100, 500)
    ]
else:
    shapes = [
        (2000, 3000),
        (1000, 5000)
    ]


class Test_JITC_RC_Conversion:

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec(self, shape, corder):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[1]))

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2)
        jax.block_until_ready((vector, out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat(self, shape, corder):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[0]))

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2)
        jax.block_until_ready((vector, out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat(self, k, shape, corder):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(shape[1], k))

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert allclose(out1, out2)
        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit(self, k, shape, corder):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(k, shape[0]))

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.T).T
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matvec_event(self, shape, corder, asbool):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[1], asbool=asbool)

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_vecmat_event(self, shape, corder, asbool):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[0], asbool=asbool)

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_jitmat_event(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([shape[1], k], asbool=asbool)

        out1 = jitcr @ matrix
        out2 = (matrix.value.T @ jitcc).T
        assert allclose(out1, out2)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matjit_event(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([k, shape[0]], asbool=asbool)

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.value.T).T
        print(out1 - out2)
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((out1, out2))


class Test_JITC_Operator_Behavior:
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_r_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        dense = mat.todense()

        left_vec = gen_events(shape[0], asbool=False).value
        right_vec = gen_events(shape[1], asbool=False).value
        left_mat = gen_events((5, shape[0]), asbool=False).value
        right_mat = gen_events((shape[1], 4), asbool=False).value

        r1 = left_vec @ mat
        r2 = left_vec @ dense
        r3 = mat @ right_vec
        r4 = dense @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense
        r7 = mat @ right_mat
        r8 = dense @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_c_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarC((1.5, 0.1, 123), shape=shape, corder=corder)
        dense = mat.todense()

        left_vec = gen_events(shape[0], asbool=False).value
        right_vec = gen_events(shape[1], asbool=False).value
        left_mat = gen_events((5, shape[0]), asbool=False).value
        right_mat = gen_events((shape[1], 4), asbool=False).value

        r1 = left_vec @ mat
        r2 = left_vec @ dense
        r3 = mat @ right_vec
        r4 = dense @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense
        r7 = mat @ right_mat
        r8 = dense @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_r_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder).T
        dense = mat.todense()

        left_vec = jnp.asarray(np.random.rand(shape[1]))
        right_vec = jnp.asarray(np.random.rand(shape[0]))
        left_mat = jnp.asarray(np.random.rand(5, shape[1]))
        right_mat = jnp.asarray(np.random.rand(shape[0], 4))

        r1 = left_vec @ mat
        r2 = left_vec @ dense
        r3 = mat @ right_vec
        r4 = dense @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense
        r7 = mat @ right_mat
        r8 = dense @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_c_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarC((1.5, 0.1, 123), shape=shape, corder=corder).T
        dense = mat.todense()

        left_vec = jnp.asarray(np.random.rand(shape[1]))
        right_vec = jnp.asarray(np.random.rand(shape[0]))
        left_mat = jnp.asarray(np.random.rand(5, shape[1]))
        right_mat = jnp.asarray(np.random.rand(shape[0], 4))

        r1 = left_vec @ mat
        r2 = left_vec @ dense
        r3 = mat @ right_vec
        r4 = dense @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense
        r7 = mat @ right_mat
        r8 = dense @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    def test_jitc_scalar_unit_operator_behavior(self, cls):
        import brainunit as u
        shape = (20, 30)
        weight = 2.1 * u.mV
        mat = cls((weight, 0.2, 123), shape=shape)
        dense = mat.todense()

        right_vec = jnp.asarray(np.random.rand(shape[1]))
        left_vec = jnp.asarray(np.random.rand(shape[0]))

        r1 = mat @ right_vec
        r2 = dense @ right_vec
        r3 = left_vec @ mat
        r4 = left_vec @ dense
        assert u.math.allclose(r1, r2,
                               rtol=1e-4 * u.get_unit(r2),
                               atol=1e-4 * u.get_unit(r2))
        assert u.math.allclose(r3, r4,
                               rtol=1e-4 * u.get_unit(r4),
                               atol=1e-4 * u.get_unit(r4))
        jax.block_until_ready((right_vec, left_vec))


class Test_JITC_To_Dense:

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('corder', [True, False])
    def test_todense(self, shape, transpose, corder):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        out1 = jitcr.todense()
        out2 = jitcc.todense().T
        out3 = jitcr.T.todense().T
        out4 = jitcc.T.todense()
        assert allclose(out1, out2)
        assert allclose(out1, out3)
        assert allclose(out1, out4)
        jax.block_until_ready((out1, out2, out3, out4))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('weight', [-1., 1.])
    def test_vjp(self, shape, corder, weight):
        base = brainevent.JITCScalarR((1.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_vjp(weight):
            res = base * weight
            return res

        ct = brainstate.random.random(shape)
        primals, f_vjp = jax.vjp(f_dense_vjp, weight)
        true_weight_grad, = f_vjp(ct)

        expected_weight_grad = (ct * base).sum()
        assert allclose(true_weight_grad, expected_weight_grad)

        def f_jitc_vjp(weight):
            mat = brainevent.JITCScalarR((weight, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, f_vjp2 = jax.vjp(f_jitc_vjp, weight)
        jitc_weight_grad, = f_vjp2(ct)

        assert allclose(true_weight_grad, jitc_weight_grad)
        jax.block_until_ready((base, ct, primals, true_weight_grad, expected_weight_grad, jitc_weight_grad))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('weight', [-1., 1.])
    def test_jvp(self, shape, corder, weight):
        base = brainevent.JITCScalarR((1., 0.1, 123), shape=shape, corder=corder).todense()
        tagents = (brainstate.random.random(),)

        def f_dense_jvp(weight):
            res = base * weight
            return res

        def f_jitc_jvp(weight):
            mat = brainevent.JITCScalarR((weight, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, true_grad = jax.jvp(f_dense_jvp, (weight,), tagents)
        primals, jitc_grad = jax.jvp(f_jitc_jvp, (weight,), tagents)
        assert allclose(true_grad, jitc_grad)
        jax.block_until_ready((base, tagents[0], primals, true_grad, jitc_grad))


class Test_JITC_Scalar_Validation:
    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('prob', [-0.1, 1.1, float('nan')])
    def test_invalid_prob_raises(self, cls, prob):
        with pytest.raises(ValueError, match='prob'):
            cls((1.5, prob, 123), shape=(8, 6))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_zero_prob_dense_matvec_matmat(self, cls, corder):
        shape = (8, 6)
        mat = cls((1.5, 0.0, 123), shape=shape, corder=corder)

        dense = mat.todense()
        assert allclose(dense, jnp.zeros_like(dense))

        vec = jnp.ones(shape[1])
        out_mv = mat @ vec
        assert allclose(out_mv, jnp.zeros_like(out_mv))

        B = jnp.ones((shape[1], 4))
        out_mm = mat @ B
        assert allclose(out_mm, jnp.zeros_like(out_mm))
        jax.block_until_ready((dense, vec, out_mv, B, out_mm))

    def test_with_data_accepts_scalar(self):
        mat = brainevent.JITCScalarR((1.5, 0.1, 123), shape=(8, 6))
        updated = mat.with_data(2.0)
        assert allclose(updated.weight, 2.0)
        assert updated.prob == mat.prob
        assert updated.seed == mat.seed
        assert updated.shape == mat.shape
