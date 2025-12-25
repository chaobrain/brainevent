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
from brainevent._typing import MatrixShape

brainevent.config.gpu_kernel_backend = 'pallas'

if brainstate.environ.get_platform() == 'cpu':
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
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecjit(self, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[0]))

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat(self, k, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(shape[1], k))

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)

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

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitvec_event(self, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[1])

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecjit_event(self, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[0])

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_jitmat_event(self, k, shape: MatrixShape, corder, asbool):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([shape[1], k], asbool=asbool)

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matjit_event(self, k, shape: MatrixShape, corder, asbool):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([k, shape[0]], asbool=asbool)

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.T).T
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)


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
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)
        assert allclose(out1, out3)
        assert allclose(out1, out4)

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

        primals, true_grad = jax.jvp(f_dense_jvp, (wlow, whigh), tagents)
        primals, jitc_grad = jax.jvp(f_jitc_jvp, (wlow, whigh), tagents)
        assert allclose(true_grad, jitc_grad)

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

        primals, true_grad = jax.jvp(f_dense_jvp, (wlow,), (dwlow,))
        expected_grad = (-base + 1.) * dwlow
        assert allclose(true_grad, expected_grad)

        def f_jitc_jvp(wlow):
            mat = brainevent.JITCUniformR((wlow, whigh, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, jitc_grad = jax.jvp(f_jitc_jvp, (wlow,), (dwlow,))
        assert allclose(true_grad, jitc_grad)

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

        primals, true_grad = jax.jvp(f_dense_jvp, (whigh,), (dw_high,))
        expected_grad = base * dw_high
        assert allclose(true_grad, expected_grad)

        def f_jitc_jvp(whigh):
            mat = brainevent.JITCUniformR((wlow, whigh, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, jitc_grad = jax.jvp(f_jitc_jvp, (whigh,), (dw_high,))
        assert allclose(true_grad, jitc_grad)
