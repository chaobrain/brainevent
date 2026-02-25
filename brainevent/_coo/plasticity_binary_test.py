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
import numpy as np
import pytest

import brainevent
from brainevent._coo.plasticity_binary import (
    update_coo_on_binary_pre,
    update_coo_on_binary_post,
    update_coo_on_binary_pre_p,
    update_coo_on_binary_post_p,
)

PLATFORM = jax.default_backend()
PRE_BACKENDS = tuple(update_coo_on_binary_pre_p.available_backends(PLATFORM))
POST_BACKENDS = tuple(update_coo_on_binary_post_p.available_backends(PLATFORM))


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
        coo = coo.with_data(update_coo_on_binary_pre(coo.data, coo.row, coo.col, pre_spike, post_trace))

        mat = mat + jnp.outer(pre_spike, post_trace)
        mat = jnp.where(mask, mat, 0.)

        assert jnp.allclose(coo.todense(), mat, rtol=1e-2, atol=1e-2)
        jax.block_until_ready((mat, pre_spike, post_trace))

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
            coo = coo.with_data(update_coo_on_binary_pre(coo.data, coo.row, coo.col, pre_spike, post_trace))

            mat = mat + u.math.outer(pre_spike, post_trace)
            mat = u.math.where(mask, mat, 0. * mat_unit)

            assert u.math.allclose(coo.todense(), mat)
            jax.block_until_ready((mat, pre_spike, post_trace))

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
        coo = coo.with_data(update_coo_on_binary_pre(coo.data, coo.row, coo.col, pre_spike, post_trace, w_in, w_max))

        mat = mat + jnp.outer(pre_spike, post_trace)
        mat = jnp.clip(mat, w_in, w_max)
        mat = jnp.where(mask, mat, 0.)

        assert jnp.allclose(coo.todense(), mat)
        jax.block_until_ready((mat, pre_spike, post_trace))


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
        coo = coo.with_data(update_coo_on_binary_post(coo.data, coo.row, coo.col, pre_trace, post_spike))

        mat = mat + jnp.outer(pre_trace, post_spike)
        mat = jnp.where(mask, mat, 0.)

        assert jnp.allclose(coo.todense(), mat)
        jax.block_until_ready((mat, pre_trace, post_spike))

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
            coo = coo.with_data(update_coo_on_binary_post(coo.data, coo.row, coo.col, pre_trace, post_spike))

            mat = mat + u.math.outer(pre_trace, post_spike)
            mat = u.math.where(mask, mat, 0. * mat_unit)

            assert u.math.allclose(coo.todense(), mat)
            jax.block_until_ready((mat, pre_trace, post_spike))

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
        coo = coo.with_data(update_coo_on_binary_post(coo.data, coo.row, coo.col, pre_trace, post_spike, w_in, w_max))

        mat = mat + jnp.outer(pre_trace, post_spike)
        mat = jnp.clip(mat, w_in, w_max)
        mat = jnp.where(mask, mat, 0.)

        assert jnp.allclose(coo.todense(), mat)
        jax.block_until_ready((mat, pre_trace, post_spike))


@pytest.mark.skipif(PLATFORM != 'gpu', reason='GPU backend parity tests require CUDA GPU.')
class Test_coo_gpu_backend_parity:
    @staticmethod
    def _call_pre_backend(weight, pre_ids, post_ids, pre_spike, post_trace, backend):
        return update_coo_on_binary_pre_p(
            weight, pre_ids, post_ids, pre_spike, post_trace,
            outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
            weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
            pre_ids_info=jax.ShapeDtypeStruct(pre_ids.shape, pre_ids.dtype),
            post_ids_info=jax.ShapeDtypeStruct(post_ids.shape, post_ids.dtype),
            spike_info=jax.ShapeDtypeStruct(pre_spike.shape, pre_spike.dtype),
            trace_info=jax.ShapeDtypeStruct(post_trace.shape, post_trace.dtype),
            backend=backend,
        )[0]

    @staticmethod
    def _call_post_backend(weight, pre_ids, post_ids, pre_trace, post_spike, backend):
        return update_coo_on_binary_post_p(
            weight, pre_ids, post_ids, pre_trace, post_spike,
            outs=[jax.ShapeDtypeStruct(weight.shape, weight.dtype)],
            weight_info=jax.ShapeDtypeStruct(weight.shape, weight.dtype),
            pre_ids_info=jax.ShapeDtypeStruct(pre_ids.shape, pre_ids.dtype),
            post_ids_info=jax.ShapeDtypeStruct(post_ids.shape, post_ids.dtype),
            trace_info=jax.ShapeDtypeStruct(pre_trace.shape, pre_trace.dtype),
            spike_info=jax.ShapeDtypeStruct(post_spike.shape, post_spike.dtype),
            backend=backend,
        )[0]

    @staticmethod
    def _make_ids(n_syn, n_pre, n_post, seed):
        rng = np.random.default_rng(seed)
        if n_syn == 0:
            return jnp.empty((0,), dtype=jnp.int32), jnp.empty((0,), dtype=jnp.int32)
        # Skew toward a hotspot to stress duplicate gathers.
        pre_ids = rng.integers(0, n_pre, size=n_syn, dtype=np.int32)
        post_ids = rng.integers(0, n_post, size=n_syn, dtype=np.int32)
        hot = max(1, n_syn // 8)
        pre_ids[:hot] = pre_ids[0]
        post_ids[:hot] = post_ids[0]
        return jnp.asarray(pre_ids), jnp.asarray(post_ids)

    @pytest.mark.parametrize('backend', PRE_BACKENDS)
    @pytest.mark.parametrize('dtype', [jnp.float32, jnp.float16])
    @pytest.mark.parametrize('bool_spike', [True, False])
    @pytest.mark.parametrize('n_syn', [0, 33, 4097])
    def test_pre_backend_matches_reference(self, backend, dtype, bool_spike, n_syn):
        n_pre, n_post = 257, 193
        rng = np.random.default_rng(123)
        pre_ids, post_ids = self._make_ids(n_syn, n_pre, n_post, seed=321)
        weight = jnp.asarray(rng.standard_normal(n_syn), dtype=dtype)
        post_trace = jnp.asarray(rng.standard_normal(n_post), dtype=dtype)
        spike = rng.random(n_pre) > 0.6
        pre_spike = jnp.asarray(spike, dtype=jnp.bool_)
        if not bool_spike:
            pre_spike = jnp.asarray(spike, dtype=dtype)
        spike_mask = spike[pre_ids]

        f = jax.jit(lambda: self._call_pre_backend(weight, pre_ids, post_ids, pre_spike, post_trace, backend))
        out = f()
        ref = weight + jnp.where(spike_mask, post_trace[post_ids], 0)

        rtol, atol = (5e-3, 5e-3) if dtype == jnp.float16 else (1e-5, 1e-5)
        assert jnp.allclose(out, ref, rtol=rtol, atol=atol)
        jax.block_until_ready((out, ref))

    @pytest.mark.parametrize('backend', POST_BACKENDS)
    @pytest.mark.parametrize('dtype', [jnp.float32, jnp.float16])
    @pytest.mark.parametrize('bool_spike', [True, False])
    @pytest.mark.parametrize('n_syn', [0, 33, 4097])
    def test_post_backend_matches_reference(self, backend, dtype, bool_spike, n_syn):
        n_pre, n_post = 257, 193
        rng = np.random.default_rng(456)
        pre_ids, post_ids = self._make_ids(n_syn, n_pre, n_post, seed=654)
        weight = jnp.asarray(rng.standard_normal(n_syn), dtype=dtype)
        pre_trace = jnp.asarray(rng.standard_normal(n_pre), dtype=dtype)
        spike = rng.random(n_post) > 0.6
        post_spike = jnp.asarray(spike, dtype=jnp.bool_)
        if not bool_spike:
            post_spike = jnp.asarray(spike, dtype=dtype)
        spike_mask = spike[post_ids]

        f = jax.jit(lambda: self._call_post_backend(weight, pre_ids, post_ids, pre_trace, post_spike, backend))
        out = f()
        ref = weight + jnp.where(spike_mask, pre_trace[pre_ids], 0)

        rtol, atol = (5e-3, 5e-3) if dtype == jnp.float16 else (1e-5, 1e-5)
        assert jnp.allclose(out, ref, rtol=rtol, atol=atol)
        jax.block_until_ready((out, ref))

    def test_default_backend_routes_bf16_to_supported_kernel(self):
        n_pre, n_post, n_syn = 127, 111, 4099
        rng = np.random.default_rng(777)
        pre_ids, post_ids = self._make_ids(n_syn, n_pre, n_post, seed=778)
        weight = jnp.asarray(rng.standard_normal(n_syn), dtype=jnp.bfloat16)
        pre_trace = jnp.asarray(rng.standard_normal(n_pre), dtype=jnp.bfloat16)
        post_trace = jnp.asarray(rng.standard_normal(n_post), dtype=jnp.bfloat16)
        pre_spike = jnp.asarray(rng.random(n_pre) > 0.5, dtype=jnp.bool_)
        post_spike = jnp.asarray(rng.random(n_post) > 0.5, dtype=jnp.bool_)

        out_pre = update_coo_on_binary_pre(weight, pre_ids, post_ids, pre_spike, post_trace, backend='pallas')
        out_post = update_coo_on_binary_post(weight, pre_ids, post_ids, pre_trace, post_spike, backend='pallas')

        ref_pre = weight + jnp.where(pre_spike[pre_ids], post_trace[post_ids], 0)
        ref_post = weight + jnp.where(post_spike[post_ids], pre_trace[pre_ids], 0)

        assert jnp.allclose(out_pre, ref_pre, rtol=2e-2, atol=2e-2)
        assert jnp.allclose(out_post, ref_post, rtol=2e-2, atol=2e-2)
        jax.block_until_ready((out_pre, out_post, ref_pre, ref_post))

    @pytest.mark.skipif(not jax.config.read('jax_enable_x64'), reason='Requires jax_enable_x64=True')
    def test_pallas_int64_indices_parity(self):
        if 'pallas' not in PRE_BACKENDS or 'pallas' not in POST_BACKENDS:
            pytest.skip('Pallas backend unavailable for COO plasticity kernels.')

        n_pre, n_post, n_syn = 257, 193, 2048
        rng = np.random.default_rng(999)
        pre_ids = jnp.asarray(rng.integers(0, n_pre, size=n_syn, dtype=np.int64))
        post_ids = jnp.asarray(rng.integers(0, n_post, size=n_syn, dtype=np.int64))
        weight = jnp.asarray(rng.standard_normal(n_syn), dtype=jnp.float32)
        pre_trace = jnp.asarray(rng.standard_normal(n_pre), dtype=jnp.float32)
        post_trace = jnp.asarray(rng.standard_normal(n_post), dtype=jnp.float32)
        pre_spike = jnp.asarray(rng.standard_normal(n_pre), dtype=jnp.float32)
        post_spike = jnp.asarray(rng.standard_normal(n_post), dtype=jnp.float32)

        out_pre = jax.jit(
            lambda: self._call_pre_backend(weight, pre_ids, post_ids, pre_spike, post_trace, backend='pallas')
        )()
        out_post = jax.jit(
            lambda: self._call_post_backend(weight, pre_ids, post_ids, pre_trace, post_spike, backend='pallas')
        )()

        ref_pre = weight + jnp.where(pre_spike[pre_ids] != 0., post_trace[post_ids], 0)
        ref_post = weight + jnp.where(post_spike[post_ids] != 0., pre_trace[pre_ids], 0)

        assert jnp.allclose(out_pre, ref_pre, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(out_post, ref_post, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((out_pre, out_post, ref_pre, ref_post))
