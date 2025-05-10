# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import jax.numpy as jnp
import numpy as np
import pytest
import jax

import brainevent
import brainstate


# brainevent.config.gpu_kernel_backend = 'pallas'


def gen_events(shape, prob=0.5, asbool=True):
    events = brainstate.random.random(shape) < prob
    if not asbool:
        events = jnp.asarray(events, dtype=float)
    return brainevent.EventArray(events)


class Test_JITC_RC_Conversion:

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('corder', [True, False])
    def test_todense(self, shape, transpose, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        out1 = jitcr.todense()
        out2 = jitcc.todense().T
        out3 = jitcr.T.todense().T
        out4 = jitcc.T.todense()
        assert jnp.allclose(out1, out2)
        assert jnp.allclose(out1, out3)
        assert jnp.allclose(out1, out4)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matvec(self, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[1]))

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat(self, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[0]))

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat(self, k, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(shape[1], k))

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit(self, k, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(k, shape[0]))

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.T).T
        print(out1 - out2)
        assert jnp.allclose(out1, out2, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matvec_event(self, shape, corder, asbool):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[1], asbool=asbool)

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_vecmat_event(self, shape, corder, asbool):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[0], asbool=asbool)

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_jitmat_event(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([shape[1], k], asbool=asbool)

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matjit_event(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([k, shape[0]], asbool=asbool)

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.T).T
        print(out1 - out2)
        assert jnp.allclose(out1, out2, atol=1e-4, rtol=1e-4)


class Test_JITC_Gradient:
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('wloc', [-1., 0.])
    @pytest.mark.parametrize('wscale', [1., 2.])
    def test_vjp(self, shape, corder, wloc, wscale):
        base = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_vjp(wloc, wscale):
            res = base * wscale + wloc
            return res

        ct = brainstate.random.random(shape)
        primals, f_vjp = jax.vjp(f_dense_vjp, wloc, wscale)
        true_wloc_grad, true_wscale_grad = f_vjp(ct)

        expected_wloc_grad = ct.sum()
        expected_wscale_grad = (ct * base).sum()

        assert jnp.allclose(true_wloc_grad, expected_wloc_grad)
        assert jnp.allclose(true_wscale_grad, expected_wscale_grad)

        print(true_wloc_grad, true_wscale_grad)
        print(expected_wloc_grad, expected_wscale_grad)

        def f_jitc_vjp(wloc, wscale):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, f_vjp2 = jax.vjp(f_jitc_vjp, wloc, wscale)
        jitc_wloc_grad, jitc_wscale_grad = f_vjp2(ct)

        assert jnp.allclose(true_wloc_grad, jitc_wloc_grad)
        assert jnp.allclose(true_wscale_grad, jitc_wscale_grad)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('wloc', [-1., 0.])
    @pytest.mark.parametrize('wscale', [1., 2.])
    def test_jvp(self, shape, corder, wloc, wscale):
        base = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()
        tagents = (brainstate.random.random(), brainstate.random.random())

        def f_dense_jvp(wloc, wscale):
            res = base * wscale + wloc
            return res

        def f_jitc_jvp(wloc, wscale):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, true_grad = jax.jvp(f_dense_jvp, (wloc, wscale), tagents)
        primals, jitc_grad = jax.jvp(f_jitc_jvp, (wloc, wscale), tagents)
        assert jnp.allclose(true_grad, jitc_grad, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('wloc', [-1., 0.])
    @pytest.mark.parametrize('wscale', [1., 2.])
    @pytest.mark.parametrize('dwloc', [1., 2.])
    def test_jvp_wloc(self, shape, corder, wloc, wscale, dwloc):
        base = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_jvp(wloc):
            res = base * wscale + wloc
            return res

        primals, true_grad = jax.jvp(f_dense_jvp, (wloc,), (dwloc,))
        expected_grad = jnp.ones_like(primals) * dwloc
        assert jnp.allclose(true_grad, expected_grad)

        def f_jitc_jvp(wloc):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, jitc_grad = jax.jvp(f_jitc_jvp, (wloc,), (dwloc,))
        assert jnp.allclose(true_grad, jitc_grad)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('wloc', [-1., 0.])
    @pytest.mark.parametrize('wscale', [1., 2.])
    @pytest.mark.parametrize('dw_high', [1., 2.])
    def test_jvp_wscale(self, shape, corder, wloc, wscale, dw_high):
        base = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_jvp(wscale):
            res = base * wscale + wloc
            return res

        primals, true_grad = jax.jvp(f_dense_jvp, (wscale,), (dw_high,))
        expected_grad = base * dw_high
        assert jnp.allclose(true_grad, expected_grad)

        def f_jitc_jvp(wscale):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, jitc_grad = jax.jvp(f_jitc_jvp, (wscale,), (dw_high,))
        assert jnp.allclose(true_grad, jitc_grad)
