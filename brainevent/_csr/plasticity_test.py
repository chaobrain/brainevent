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
from brainevent._csr.plasticity import csr_on_pre


class Test_csr_on_pre:
    def test_csr_on_pre_v1(self):
        n_pre = 20
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))
        mask = mat < 0.5
        mat = jnp.where(mask, mat, 0.)

        pre_spike = brainstate.random.random((n_pre,)) < 0.5
        post_trace = brainstate.random.random((n_post,))

        csr = brainevent.CSR.fromdense(mat)
        csr2 = csr.with_data(csr_on_pre(csr.data, csr.indices, csr.indptr, pre_spike, post_trace, shape=csr.shape))
        dense2 = jnp.where(mask, mat + jnp.outer(pre_spike.astype(float), post_trace), 0.)

        assert jnp.allclose(csr2.todense(), dense2)

    @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
    @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
    def test_csr_on_pre_with_unit(self, mat_unit, trace_unit):
        def run():
            n_pre = 100
            n_post = 100
            mat = brainstate.random.random((n_pre, n_post))
            mask = mat < 0.5
            mat = jnp.where(mask, mat, 0.) * mat_unit
            pre_spike = brainstate.random.random((n_pre,)) < 0.1
            post_trace = brainstate.random.random((n_post,)) * trace_unit

            csr = brainevent.CSR.fromdense(mat)
            csr = csr.with_data(csr_on_pre(csr.data, csr.indices, csr.indptr, pre_spike, post_trace, shape=csr.shape))

            dense = mat + u.math.outer(pre_spike.astype(float), post_trace)
            dense = u.math.where(mask, dense, 0. * mat_unit)

            assert u.math.allclose(csr.todense(), dense)

        if mat_unit.has_same_dim(trace_unit):
            run()
        else:
            with pytest.raises(u.UnitMismatchError):
                run()

    @pytest.mark.parametrize('w_in', [None, 0.1])
    @pytest.mark.parametrize('w_max', [None, 0.5])
    def test_csr_on_pre_v2(self, w_in, w_max):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))
        mask = mat < 0.5
        mat = jnp.where(mask, mat, 0.)
        pre_spike = brainstate.random.random((n_pre,)) < 0.1
        post_trace = brainstate.random.random((n_post,))

        csr = brainevent.CSR.fromdense(mat)
        csr = csr.with_data(
            csr_on_pre(csr.data, csr.indices, csr.indptr, pre_spike, post_trace,
                       w_min=w_in, w_max=w_max, shape=csr.shape)
        )

        mat = mat + jnp.outer(pre_spike.astype(float), post_trace)
        mat = u.math.clip(mat, a_min=w_in, a_max=w_max)

        mat = jnp.where(mask, mat, 0.)
        assert jnp.allclose(csr.todense(), mat)

# class Test_on_post:
#     def test_csr_on_post_v1(self):
#         n_pre = 20
#         n_post = 100
#         mat = brainstate.random.random((n_pre, n_post))
#         mask = mat < 0.5
#         mat = jnp.where(mask, mat, 0.)
#
#         post_spike = brainstate.random.random((n_post,)) < 0.5
#         pre_trace = brainstate.random.random((n_pre,))
#
#         csr = brainevent.CSR.fromdense(mat)
#         w_indices = np.arange(csr.indices.shape[0])
#         csr2 = csr.with_data(
#             csr2csc_on_post(csr.data, csr.indices, csr.indptr, w_indices, pre_trace, post_spike, shape=csr.shape)
#         )
#         dense2 = jnp.where(mask, mat + jnp.outer(pre_trace, post_spike.astype(float)), 0.)
#
#         assert jnp.allclose(csr2.todense(), dense2)
#
#     @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
#     @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
#     def test_csr_on_post_with_unit(self, mat_unit, trace_unit):
#         def run():
#             n_pre = 100
#             n_post = 100
#             mat = brainstate.random.random((n_pre, n_post))
#             mask = mat < 0.5
#             mat = jnp.where(mask, mat, 0.) * mat_unit
#             post_spike = brainstate.random.random((n_post,)) < 0.1
#             pre_trace = brainstate.random.random((n_pre,)) * trace_unit
#
#             csr = brainevent.CSR.fromdense(mat)
#             w_indices = np.arange(csr.indices.shape[0])
#             csr = csr.with_data(
#                 csr2csc_on_post(csr.data, csr.indices, csr.indptr, w_indices, pre_trace, post_spike, shape=csr.shape)
#             )
#
#             dense = mat + u.math.outer(pre_trace, post_spike.astype(float))
#             dense = u.math.where(mask, dense, 0. * mat_unit)
#
#             assert u.math.allclose(csr.todense(), dense)
#
#         if mat_unit.has_same_dim(trace_unit):
#             run()
#         else:
#             with pytest.raises(u.UnitMismatchError):
#                 run()
#
#     @pytest.mark.parametrize('w_in', [None, 0.1])
#     @pytest.mark.parametrize('w_max', [None, 0.5])
#     def test_csr_on_post_v2(self, w_in, w_max):
#         n_pre = 100
#         n_post = 100
#         mat = brainstate.random.random((n_pre, n_post))
#         mask = mat < 0.5
#         mat = jnp.where(mask, mat, 0.)
#         post_spike = brainstate.random.random((n_post,)) < 0.1
#         pre_trace = brainstate.random.random((n_pre,))
#
#         csr = brainevent.CSR.fromdense(mat)
#         w_indices = np.arange(csr.indices.shape[0])
#         csr = csr.with_data(
#             csr2csc_on_post(csr.data, csr.indices, csr.indptr, w_indices, pre_trace,post_spike,
#                             w_min=w_in, w_max=w_max, shape=csr.shape)
#         )
#
#         mat = mat + jnp.outer(pre_trace, post_spike.astype(float))
#         mat = u.math.clip(mat, a_min=w_in, a_max=w_max)
#
#         mat = jnp.where(mask, mat, 0.)
#         assert jnp.allclose(csr.todense(), mat)
