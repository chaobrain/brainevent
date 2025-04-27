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

import brainevent
import brainstate


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
    def test_matvec_event(self, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[1])

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat_event(self, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[0])

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat_event(self, k, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([shape[1], k])

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit_event(self, k, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([k, shape[0]])

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.T).T
        print(out1 - out2)
        assert jnp.allclose(out1, out2, atol=1e-4, rtol=1e-4)
