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
import jax.numpy as jnp
import pytest

import brainevent
from brainevent._coo.plasticity import coo_on_pre, coo_on_post


class Test_coo_on_pre:
    def test_coo_on_pre_v1(self):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))
        mask = mat < 0.5
        mat = jnp.where(mask, mat, 0.)
        pre_spike = brainstate.random.random((n_pre,)) < 0.1
        post_trace = brainstate.random.random((n_post,))

        coo = brainevent.COO.fromdense(mat)
        coo = coo.with_data(coo_on_pre(coo.data, coo.row, coo.col, pre_spike, post_trace))

        mat = mat + jnp.outer(pre_spike, post_trace)
        mat = jnp.where(mask, mat, 0.)

        assert jnp.allclose(coo.todense(), mat, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
    @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
    def test_coo_on_pre_unit(self, mat_unit, trace_unit):
        def run():
            n_pre = 100
            n_post = 100
            mat = brainstate.random.random((n_pre, n_post))
            mask = mat < 0.5
            mat = jnp.where(mask, mat, 0.) * mat_unit
            pre_spike = brainstate.random.random((n_pre,)) < 0.1
            post_trace = brainstate.random.random((n_post,)) * trace_unit

            coo = brainevent.COO.fromdense(mat)
            coo = coo.with_data(coo_on_pre(coo.data, coo.row, coo.col, pre_spike, post_trace))

            mat = mat + u.math.outer(pre_spike, post_trace)
            mat = u.math.where(mask, mat, 0. * mat_unit)

            assert u.math.allclose(coo.todense(), mat)

        if mat_unit.has_same_dim(trace_unit):
            run()
        else:
            with pytest.raises(u.UnitMismatchError):
                run()

    @pytest.mark.parametrize('w_in', [None, 0.1])
    @pytest.mark.parametrize('w_max', [None, 0.5])
    def test_coo_on_pre_v2(self, w_in, w_max):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))
        mask = mat < 0.5
        mat = jnp.where(mask, mat, 0.)
        pre_spike = brainstate.random.random((n_pre,)) < 0.1
        post_trace = brainstate.random.random((n_post,))

        coo = brainevent.COO.fromdense(mat)
        coo = coo.with_data(coo_on_pre(coo.data, coo.row, coo.col, pre_spike, post_trace, w_in, w_max))

        mat = mat + jnp.outer(pre_spike, post_trace)
        mat = jnp.clip(mat, w_in, w_max)
        mat = jnp.where(mask, mat, 0.)

        assert jnp.allclose(coo.todense(), mat)


class Test_coo_on_post:

    def test_coo_on_post_v1(self):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))
        mask = mat < 0.5
        mat = jnp.where(mask, mat, 0.)
        pre_trace = brainstate.random.random((n_pre,))
        post_spike = brainstate.random.random((n_post,)) < 0.1

        coo = brainevent.COO.fromdense(mat)
        coo = coo.with_data(coo_on_post(coo.data, coo.row, coo.col, pre_trace, post_spike))

        mat = mat + jnp.outer(pre_trace, post_spike)
        mat = jnp.where(mask, mat, 0.)

        assert jnp.allclose(coo.todense(), mat)

    @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
    @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
    def test_coo_on_post_unit(self, mat_unit, trace_unit):
        def run():
            n_pre = 100
            n_post = 100
            mat = brainstate.random.random((n_pre, n_post))
            mask = mat < 0.5
            mat = jnp.where(mask, mat, 0.) * mat_unit
            pre_trace = brainstate.random.random((n_pre,)) * trace_unit
            post_spike = brainstate.random.random((n_post,)) < 0.1

            coo = brainevent.COO.fromdense(mat)
            coo = coo.with_data(coo_on_post(coo.data, coo.row, coo.col, pre_trace, post_spike))

            mat = mat + u.math.outer(pre_trace, post_spike)
            mat = u.math.where(mask, mat, 0. * mat_unit)

            assert u.math.allclose(coo.todense(), mat)

        if mat_unit.has_same_dim(trace_unit):
            run()
        else:
            with pytest.raises(u.UnitMismatchError):
                run()

    @pytest.mark.parametrize('w_in', [None, 0.1])
    @pytest.mark.parametrize('w_max', [None, 0.5])
    def test_coo_on_post_v2(self, w_in, w_max):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))
        mask = mat < 0.5
        mat = jnp.where(mask, mat, 0.)
        pre_trace = brainstate.random.random((n_pre,))
        post_spike = brainstate.random.random((n_post,)) < 0.1

        coo = brainevent.COO.fromdense(mat)
        coo = coo.with_data(coo_on_post(coo.data, coo.row, coo.col, pre_trace, post_spike, w_in, w_max))

        mat = mat + jnp.outer(pre_trace, post_spike)
        mat = jnp.clip(mat, w_in, w_max)
        mat = jnp.where(mask, mat, 0.)

        assert jnp.allclose(coo.todense(), mat)
