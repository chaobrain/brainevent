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

from brainevent._dense.plasticity import dense_on_pre, dense_on_post


class Test_dense_on_pre:
    def test_dense_on_pre(self):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))

        pre_spike = brainstate.random.random((n_pre,)) < 0.1
        post_trace = brainstate.random.random((n_post,))

        mat2 = dense_on_pre(mat, pre_spike, post_trace)

        mat = mat + jnp.outer(pre_spike, post_trace)
        assert jnp.allclose(mat2, mat)

    @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
    @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
    def test_with_unit(self, mat_unit, trace_unit):
        def run():
            n_pre = 100
            n_post = 100
            mat = brainstate.random.random((n_pre, n_post))
            mask = mat < 0.5
            mat = jnp.where(mask, mat, 0.) * mat_unit
            pre_spike = brainstate.random.random((n_pre,)) < 0.1
            post_trace = brainstate.random.random((n_post,)) * trace_unit

            mat2 = dense_on_pre(mat, pre_spike, post_trace)

            mat = mat + u.math.outer(pre_spike, post_trace)
            assert u.math.allclose(mat2, mat)

        if mat_unit.has_same_dim(trace_unit):
            run()
        else:
            with pytest.raises(u.UnitMismatchError):
                run()

    @pytest.mark.parametrize('w_min', [None, 0.1])
    @pytest.mark.parametrize('w_max', [None, 0.5])
    def test_min_max(self, w_min, w_max):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))

        pre_spike = brainstate.random.random((n_pre,)) < 0.1
        post_trace = brainstate.random.random((n_post,))

        mat2 = dense_on_pre(mat, pre_spike, post_trace, w_min=w_min, w_max=w_max)

        mat = mat + jnp.outer(pre_spike, post_trace)
        if w_min is not None:
            mat = jnp.maximum(mat, w_min)
        if w_max is not None:
            mat = jnp.minimum(mat, w_max)

        assert jnp.allclose(mat2, mat)


class Test_dense_on_post:
    def test_dense_on_post(self):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))

        pre_trace = brainstate.random.random((n_pre,))
        post_spike = brainstate.random.random((n_post,)) < 0.1

        mat2 = dense_on_post(mat, pre_trace, post_spike)

        mat = mat + jnp.outer(pre_trace, post_spike)
        assert jnp.allclose(mat2, mat)

    @pytest.mark.parametrize('mat_unit', [u.mV, u.ms])
    @pytest.mark.parametrize('trace_unit', [u.mV, u.ms])
    def test_with_unit(self, mat_unit, trace_unit):
        def run():
            n_pre = 100
            n_post = 100
            mat = brainstate.random.random((n_pre, n_post))
            mask = mat < 0.5
            mat = jnp.where(mask, mat, 0.) * mat_unit
            pre_trace = brainstate.random.random((n_pre,)) * trace_unit
            post_spike = brainstate.random.random((n_post,)) < 0.1

            mat2 = dense_on_post(mat, pre_trace, post_spike)

            mat = mat + u.math.outer(pre_trace, post_spike)
            assert u.math.allclose(mat2, mat)

        if mat_unit.has_same_dim(trace_unit):
            run()
        else:
            with pytest.raises(u.UnitMismatchError):
                run()

    @pytest.mark.parametrize('w_min', [None, 0.1])
    @pytest.mark.parametrize('w_max', [None, 0.5])
    def test_min_max(self, w_min, w_max):
        n_pre = 100
        n_post = 100
        mat = brainstate.random.random((n_pre, n_post))
        pre_trace = brainstate.random.random((n_pre,))
        post_spike = brainstate.random.random((n_post,)) < 0.1
        mat2 = dense_on_post(mat, pre_trace, post_spike, w_min=w_min, w_max=w_max)
        mat = mat + jnp.outer(pre_trace, post_spike)
        if w_min is not None:
            mat = jnp.maximum(mat, w_min)
        if w_max is not None:
            mat = jnp.minimum(mat, w_max)
        assert jnp.allclose(mat2, mat)
