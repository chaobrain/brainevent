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

# Keep GPU matmul reference numerics stable (avoid TF32 drift in dense @ B checks).
if jax.default_backend() == 'gpu' and jax.config.jax_default_matmul_precision is None:
    jax.config.update('jax_default_matmul_precision', 'highest')

import brainevent
from brainevent._test_util import allclose, gen_events
from brainevent._typing import MatrixShape

platform = jax.default_backend()

if platform == 'cpu':
    shapes = [
        (200, 300),
        (100, 500),
    ]
else:
    shapes = [
        (2000, 3000),
        (1000, 5000),
    ]


class Test_JITC_RC_Conversion:
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitvec(self, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[1]))

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2)
        jax.block_until_ready((vector, out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecjit(self, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[0]))

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2)
        jax.block_until_ready((vector, out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat(self, k, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(shape[1], k))

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert allclose(out1, out2, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit(self, k, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(k, shape[0]))

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.T).T
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitvec_event(self, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[1])

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecjit_event(self, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[0])

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_jitmat_event(self, k, shape: MatrixShape, corder, asbool):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([shape[1], k], asbool=asbool)

        out1 = jitcr @ matrix
        out2 = (matrix.value.T @ jitcc).T
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matjit_event(self, k, shape: MatrixShape, corder, asbool):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([k, shape[0]], asbool=asbool)

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.value.T).T
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((out1, out2))


class Test_JITC_Operator_Behavior:
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_uniform_r_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCUniformR((-1.5, 1.5, 0.1, 123), shape=shape, corder=corder)
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
    def test_jitc_uniform_c_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCUniformC((-1.5, 1.5, 0.1, 123), shape=shape, corder=corder)
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
    def test_jitc_uniform_r_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCUniformR((-1.5, 1.5, 0.1, 123), shape=shape, corder=corder).T
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
    def test_jitc_uniform_c_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCUniformC((-1.5, 1.5, 0.1, 123), shape=shape, corder=corder).T
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

    @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    def test_jitc_uniform_unit_operator_behavior(self, cls):
        import brainunit as u

        shape = (20, 30)
        weight = 2.1 * u.mV
        mat = cls((-weight, weight, 0.2, 123), shape=shape)
        dense = mat.todense()

        right_vec = jnp.asarray(np.random.rand(shape[1]))
        left_vec = jnp.asarray(np.random.rand(shape[0]))

        r1 = mat @ right_vec
        r2 = dense @ right_vec
        r3 = left_vec @ mat
        r4 = left_vec @ dense
        assert u.math.allclose(
            r1,
            r2,
            rtol=1e-4 * u.get_unit(r2),
            atol=1e-4 * u.get_unit(r2),
        )
        assert u.math.allclose(
            r3,
            r4,
            rtol=1e-4 * u.get_unit(r4),
            atol=1e-4 * u.get_unit(r4),
        )
        jax.block_until_ready((right_vec, left_vec))


class Test_JITC_To_Dense:
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('corder', [True, False])
    def test_todense(self, shape: MatrixShape, transpose, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
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
    @pytest.mark.parametrize('wlow', [-1., 0.])
    @pytest.mark.parametrize('whigh', [1., 2.])
    def test_vjp(self, shape, corder, wlow, whigh):
        base = brainevent.JITCUniformR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_vjp(wlow, whigh):
            res = base * (whigh - wlow) + wlow
            return res

        ct = brainstate.random.random(shape)
        primals, f_vjp = jax.vjp(f_dense_vjp, wlow, whigh)
        true_wlow_grad, true_whigh_grad = f_vjp(ct)

        expected_wlow_grad = (ct * (-base + 1.)).sum()
        expected_whigh_grad = (ct * base).sum()

        assert allclose(true_wlow_grad, expected_wlow_grad)
        assert allclose(true_whigh_grad, expected_whigh_grad)

        print(true_wlow_grad, true_whigh_grad)
        print(expected_wlow_grad, expected_whigh_grad)

        def f_jitc_vjp(wlow, whigh):
            mat = brainevent.JITCUniformR((wlow, whigh, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, f_vjp2 = jax.vjp(f_jitc_vjp, wlow, whigh)
        jitc_wlow_grad, jitc_whigh_grad = f_vjp2(ct)

        assert allclose(true_wlow_grad, jitc_wlow_grad)
        assert allclose(true_whigh_grad, jitc_whigh_grad)
        jax.block_until_ready(
            (base, ct, primals, true_wlow_grad, true_whigh_grad, expected_wlow_grad, expected_whigh_grad,
             jitc_wlow_grad, jitc_whigh_grad))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('wlow', [-1., 0.])
    @pytest.mark.parametrize('whigh', [1., 2.])
    def test_jvp(self, shape, corder, wlow, whigh):
        base = brainevent.JITCUniformR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()
        tagents = (brainstate.random.random(), brainstate.random.random())

        def f_dense_jvp(wlow, whigh):
            res = base * (whigh - wlow) + wlow
            return res

        def f_jitc_jvp(wlow, whigh):
            mat = brainevent.JITCUniformR((wlow, whigh, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals1, true_grad = jax.jvp(f_dense_jvp, (wlow, whigh), tagents)
        primals2, jitc_grad = jax.jvp(f_jitc_jvp, (wlow, whigh), tagents)
        assert allclose(true_grad, jitc_grad)
        jax.block_until_ready((base, tagents[0], tagents[1], primals1, true_grad, primals2, jitc_grad))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('wlow', [-1., 0.])
    @pytest.mark.parametrize('whigh', [1., 2.])
    @pytest.mark.parametrize('dwlow', [1., 2.])
    def test_jvp_wlow(self, shape, corder, wlow, whigh, dwlow):
        base = brainevent.JITCUniformR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_jvp(wlow):
            res = base * (whigh - wlow) + wlow
            return res

        primals1, true_grad = jax.jvp(f_dense_jvp, (wlow,), (dwlow,))
        expected_grad = (-base + 1.) * dwlow
        assert allclose(true_grad, expected_grad)

        def f_jitc_jvp(wlow):
            mat = brainevent.JITCUniformR((wlow, whigh, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals2, jitc_grad = jax.jvp(f_jitc_jvp, (wlow,), (dwlow,))
        assert allclose(true_grad, jitc_grad)
        jax.block_until_ready((base, primals1, true_grad, expected_grad, primals2, jitc_grad))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('wlow', [-1., 0.])
    @pytest.mark.parametrize('whigh', [1., 2.])
    @pytest.mark.parametrize('dw_high', [1., 2.])
    def test_jvp_whigh(self, shape, corder, wlow, whigh, dw_high):
        base = brainevent.JITCUniformR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_jvp(whigh):
            res = base * (whigh - wlow) + wlow
            return res

        primals1, true_grad = jax.jvp(f_dense_jvp, (whigh,), (dw_high,))
        expected_grad = base * dw_high
        assert allclose(true_grad, expected_grad)

        def f_jitc_jvp(whigh):
            mat = brainevent.JITCUniformR((wlow, whigh, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals2, jitc_grad = jax.jvp(f_jitc_jvp, (whigh,), (dw_high,))
        assert allclose(true_grad, jitc_grad)
        jax.block_until_ready((base, primals1, true_grad, expected_grad, primals2, jitc_grad))


class Test_JITC_Uniform_Validation:
    @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    @pytest.mark.parametrize('prob', [-0.1, 1.1, float('nan')])
    def test_invalid_prob_raises(self, cls, prob):
        with pytest.raises(ValueError, match='prob'):
            cls((-1.0, 1.0, prob, 123), shape=(8, 6))

    @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    def test_invalid_bounds_raises(self, cls):
        with pytest.raises(ValueError, match='wlow'):
            cls((1.0, -1.0, 0.1, 123), shape=(8, 6))

    # @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    # @pytest.mark.parametrize('corder', [True, False])
    # def test_zero_prob_dense_matvec_matmat(self, cls, corder):
    #     shape = (8, 6)
    #     mat = cls((-1.0, 1.0, 0.0, 123), shape=shape, corder=corder)
    #
    #     dense = mat.todense()
    #     assert allclose(dense, jnp.zeros_like(dense))
    #
    #     vec = jnp.ones(shape[1])
    #     out_mv = mat @ vec
    #     assert allclose(out_mv, jnp.zeros_like(out_mv))
    #
    #     B = jnp.ones((shape[1], 4))
    #     out_mm = mat @ B
    #     assert allclose(out_mm, jnp.zeros_like(out_mm))
    #     jax.block_until_ready((dense, vec, out_mv, B, out_mm))
