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
from brainevent._jit_normal.binary import binary_jitnmv, binary_jitnmm
from brainevent._jit_normal.float import jitn, jitnmv, jitnmm
from brainevent._jit_normal.csr import jitn_to_csr
from brainevent._jit_normal.csr import jitn_csr_count_p
from brainevent._test_util import allclose, gen_events

# Every test in this module dispatches to the native ``numba`` backend (the only backend for
# JIT-connectivity kernels), which compiles per test and dominates wall-clock. Mark the whole
# module ``slow`` so the default ``pytest`` run skips it; CI runs it via ``pytest -m ""``.
pytestmark = pytest.mark.slow

platform = jax.default_backend()
CSR_IMPLEMENTATIONS = tuple(jitn_csr_count_p.available_backends(platform))

# JITC CSR materialization goes through the ``jitn_csr_count`` primitive; run
# these tests on any platform that has a registered CSR backend.
requires_csr_backend = pytest.mark.skipif(
    not CSR_IMPLEMENTATIONS,
    reason=f'No JITC CSR materialization backend on platform={platform}',
)

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
    def test_jitvec(self, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[1]))

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((vector, out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecjit(self, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[0]))

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((vector, out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitmat(self, k, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(shape[1], k))

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert allclose(out1, out2, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit(self, k, shape, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(k, shape[0]))

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.T).T
        print(out1 - out2)
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matvec_event(self, shape, corder, asbool):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[1], asbool=asbool)

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_vecmat_event(self, shape, corder, asbool):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[0], asbool=asbool)

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_jitmat_event(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
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
    def test_matjit_event(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([k, shape[0]], asbool=asbool)

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.value.T).T
        print(out1 - out2)
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((out1, out2))


class Test_JITC_Operator_Behavior:
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_normal_r_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCNormalR((1.5, 0.1, 0.2, 123), shape=shape, corder=corder)
        dense_mv = mat.todense()
        dense_mm = mat.todense()

        left_vec = gen_events(shape[0], asbool=False).value
        right_vec = gen_events(shape[1], asbool=False).value
        left_mat = gen_events((5, shape[0]), asbool=False).value
        right_mat = gen_events((shape[1], 4), asbool=False).value

        r1 = left_vec @ mat
        r2 = left_vec @ dense_mv
        r3 = mat @ right_vec
        r4 = dense_mv @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense_mm
        r7 = mat @ right_mat
        r8 = dense_mm @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat,
                               r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_normal_c_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCNormalC((1.5, 0.1, 0.2, 123), shape=shape, corder=corder)
        dense_mv = mat.todense()
        dense_mm = mat.todense()

        left_vec = gen_events(shape[0], asbool=False).value
        right_vec = gen_events(shape[1], asbool=False).value
        left_mat = gen_events((5, shape[0]), asbool=False).value
        right_mat = gen_events((shape[1], 4), asbool=False).value

        r1 = left_vec @ mat
        r2 = left_vec @ dense_mv
        r3 = mat @ right_vec
        r4 = dense_mv @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense_mm
        r7 = mat @ right_mat
        r8 = dense_mm @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat,
                               r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_normal_r_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCNormalR((1.5, 0.15, 0.2, 123), shape=shape, corder=corder).T
        dense_mv = mat.todense()
        dense_mm = mat.todense()

        left_vec = jnp.asarray(np.random.rand(shape[1]))
        right_vec = jnp.asarray(np.random.rand(shape[0]))
        left_mat = jnp.asarray(np.random.rand(5, shape[1]))
        right_mat = jnp.asarray(np.random.rand(shape[0], 4))

        r1 = left_vec @ mat
        r2 = left_vec @ dense_mv
        r3 = mat @ right_vec
        r4 = dense_mv @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense_mm
        r7 = mat @ right_mat
        r8 = dense_mm @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat,
                               r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_normal_c_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCNormalC((1.5, 0.15, 0.2, 123), shape=shape, corder=corder).T
        dense_mv = mat.todense()
        dense_mm = mat.todense()

        left_vec = jnp.asarray(np.random.rand(shape[1]))
        right_vec = jnp.asarray(np.random.rand(shape[0]))
        left_mat = jnp.asarray(np.random.rand(5, shape[1]))
        right_mat = jnp.asarray(np.random.rand(shape[0], 4))

        r1 = left_vec @ mat
        r2 = left_vec @ dense_mv
        r3 = mat @ right_vec
        r4 = dense_mv @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense_mm
        r7 = mat @ right_mat
        r8 = dense_mm @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat,
                               r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    def test_jitc_normal_unit_operator_behavior(self, cls):
        import brainunit as u
        shape = (20, 30)
        weight = 2.1 * u.mV
        mat = cls((weight, weight * 0.01, 0.2, 123), shape=shape)
        dense_mv = mat.todense()

        right_vec = jnp.asarray(np.random.rand(shape[1]))
        left_vec = jnp.asarray(np.random.rand(shape[0]))

        r1 = mat @ right_vec
        r2 = dense_mv @ right_vec
        r3 = left_vec @ mat
        r4 = left_vec @ dense_mv
        # Compare unit and mantissa separately: mixing a dimensionless ``rtol`` with a
        # unit-carrying ``atol`` trips saiunit's ``allclose``; the operator preserves mV.
        assert u.get_unit(r1) == u.get_unit(r2)
        assert u.get_unit(r3) == u.get_unit(r4)
        assert allclose(u.get_mantissa(r1), u.get_mantissa(r2), rtol=1e-4, atol=1e-4)
        assert allclose(u.get_mantissa(r3), u.get_mantissa(r4), rtol=1e-4, atol=1e-4)
        jax.block_until_ready((right_vec, left_vec))


class Test_JITC_To_Dense:
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('corder', [True, False])
    def test_todense(self, shape, transpose, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        out1 = jitcr.todense()
        out2 = jitcc.todense().T
        out3 = jitcr.T.todense().T
        out4 = jitcc.T.todense()
        assert allclose(out1, out2)
        assert allclose(out1, out3)
        assert allclose(out1, out4)
        jax.block_until_ready((out1, out2, out3, out4))

    # Covering set: every value of shape/corder/wloc/wscale appears at least once (4 rows
    # instead of the full 2x2x2x2 = 16 product; gradient correctness is value-independent).
    @pytest.mark.parametrize('shape,corder,wloc,wscale', [
        (shapes[0], True, -1., 1.),
        (shapes[1], False, 0., 2.),
        (shapes[0], False, -1., 2.),
        (shapes[1], True, 0., 1.),
    ])
    def test_vjp(self, shape, corder, wloc, wscale):
        z = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()
        mask = brainevent.JITCNormalR((1.0, 0.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_vjp(wloc, wscale):
            res = z * wscale + mask * wloc
            return res

        ct = brainstate.random.random(shape)
        primals, f_vjp = jax.vjp(f_dense_vjp, wloc, wscale)
        true_wloc_grad, true_wscale_grad = f_vjp(ct)

        expected_wloc_grad = (ct * mask).sum()
        expected_wscale_grad = (ct * z).sum()

        assert allclose(true_wloc_grad, expected_wloc_grad)
        assert allclose(true_wscale_grad, expected_wscale_grad)

        print(true_wloc_grad, true_wscale_grad)
        print(expected_wloc_grad, expected_wscale_grad)

        def f_jitc_vjp(wloc, wscale):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, f_vjp2 = jax.vjp(f_jitc_vjp, wloc, wscale)
        jitc_wloc_grad, jitc_wscale_grad = f_vjp2(ct)

        assert allclose(true_wloc_grad, jitc_wloc_grad, rtol=1e-2, atol=1e-2)
        assert allclose(true_wscale_grad, jitc_wscale_grad, rtol=1e-2, atol=1e-2)
        jax.block_until_ready(
            (z, mask, ct, primals, true_wloc_grad, true_wscale_grad,
             expected_wloc_grad, expected_wscale_grad, jitc_wloc_grad, jitc_wscale_grad)
        )

    @pytest.mark.parametrize('shape,corder,wloc,wscale', [
        (shapes[0], True, -1., 1.),
        (shapes[1], False, 0., 2.),
        (shapes[0], False, -1., 2.),
        (shapes[1], True, 0., 1.),
    ])
    def test_jvp(self, shape, corder, wloc, wscale):
        z = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()
        mask = brainevent.JITCNormalR((1.0, 0.0, 0.1, 123), shape=shape, corder=corder).todense()
        tagents = (brainstate.random.random(), brainstate.random.random())

        def f_dense_jvp(wloc, wscale):
            res = z * wscale + mask * wloc
            return res

        def f_jitc_jvp(wloc, wscale):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, true_grad = jax.jvp(f_dense_jvp, (wloc, wscale), tagents)
        primals, jitc_grad = jax.jvp(f_jitc_jvp, (wloc, wscale), tagents)
        assert allclose(true_grad, jitc_grad, atol=1e-3, rtol=1e-3)
        jax.block_until_ready((z, mask, tagents[0], tagents[1], primals, true_grad, jitc_grad))

    # Covering set over the 2x2x2x2x2 = 32 product (every value appears at least once).
    @pytest.mark.parametrize('shape,corder,wloc,wscale,dwloc', [
        (shapes[0], True, -1., 1., 1.),
        (shapes[1], False, 0., 2., 2.),
        (shapes[0], False, -1., 2., 2.),
        (shapes[1], True, 0., 1., 1.),
    ])
    def test_jvp_wloc(self, shape, corder, wloc, wscale, dwloc):
        z = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()
        mask = brainevent.JITCNormalR((1.0, 0.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_jvp(wloc):
            res = z * wscale + mask * wloc
            return res

        primals, true_grad = jax.jvp(f_dense_jvp, (wloc,), (dwloc,))
        expected_grad = mask * dwloc
        assert allclose(true_grad, expected_grad)

        def f_jitc_jvp(wloc):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, jitc_grad = jax.jvp(f_jitc_jvp, (wloc,), (dwloc,))
        assert allclose(true_grad, jitc_grad)
        jax.block_until_ready((z, mask, primals, true_grad, expected_grad, jitc_grad))

    @pytest.mark.parametrize('shape,corder,wloc,wscale,dw_high', [
        (shapes[0], True, -1., 1., 1.),
        (shapes[1], False, 0., 2., 2.),
        (shapes[0], False, -1., 2., 2.),
        (shapes[1], True, 0., 1., 1.),
    ])
    def test_jvp_wscale(self, shape, corder, wloc, wscale, dw_high):
        z = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).todense()
        mask = brainevent.JITCNormalR((1.0, 0.0, 0.1, 123), shape=shape, corder=corder).todense()

        def f_dense_jvp(wscale):
            res = z * wscale + mask * wloc
            return res

        primals, true_grad = jax.jvp(f_dense_jvp, (wscale,), (dw_high,))
        expected_grad = z * dw_high
        assert allclose(true_grad, expected_grad)

        def f_jitc_jvp(wscale):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, jitc_grad = jax.jvp(f_jitc_jvp, (wscale,), (dw_high,))
        assert allclose(true_grad, jitc_grad)
        jax.block_until_ready((z, mask, primals, true_grad, expected_grad, jitc_grad))

# class Test_JITC_Normal_Validation:
#     @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
#     @pytest.mark.parametrize('corder', [True, False])
#     def test_zero_prob_dense_matvec_matmat(self, cls, corder):
#         shape = (8, 6)
#         mat = cls((1.5, 0.1, 0.0, 123), shape=shape, corder=corder)
#
#         dense = mat.todense()
#         assert allclose(dense, jnp.zeros_like(dense))
#
#         vec = jnp.ones(shape[1])
#         out_mv = mat @ vec
#         assert allclose(out_mv, jnp.zeros_like(out_mv))
#
#         B = jnp.ones((shape[1], 4))
#         out_mm = mat @ B
#         assert allclose(out_mm, jnp.zeros_like(out_mm))
#         jax.block_until_ready((dense, vec, out_mv, B, out_mm))


class Test_JITC_Materialization:
    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    def test_all_materializations_agree(self, cls):
        # There is one matrix, so every materialization route works and they agree.
        mat = cls((1.5, 0.1, 0.2, 123), shape=(8, 6))
        dense = mat.todense()
        assert dense.shape == mat.shape
        assert allclose(mat.tocsr().todense(), dense)
        assert allclose(mat.tocsc().todense(), dense)
        assert allclose(mat.tocoo().todense(), dense)


@requires_csr_backend
class Test_JITC_Materialization_Matches_Binary:
    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_mv_matvec_dense_csr_binary_agree(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder)

        dense = mat.todense()
        csr = mat.tocsr()
        spike = gen_events(dense.shape[1], asbool=False).value

        out_binary = binary_jitnmv(
            mat.wloc, mat.wscale, mat.prob, spike, mat.seed,
            shape=mat.shape, transpose=False, corder=mat.corder,
        )

        assert allclose(dense @ spike, out_binary, rtol=1e-4, atol=1e-4)
        assert allclose(csr @ spike, out_binary, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('k', [4, 7])
    def test_mm_matmat_dense_csr_binary_agree(self, cls, corder, k):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder)

        dense = mat.todense()
        csr = mat.tocsr()
        spikes = gen_events((dense.shape[1], k), asbool=False).value

        out_binary = binary_jitnmm(
            mat.wloc, mat.wscale, mat.prob, spikes, mat.seed,
            shape=mat.shape, transpose=False, corder=mat.corder,
        )

        assert allclose(dense @ spikes, out_binary, rtol=1e-4, atol=1e-4)
        assert allclose(csr @ spikes, out_binary, rtol=1e-4, atol=1e-4)


@requires_csr_backend
class Test_JITC_To_CSR:
    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_roundtrip(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder)

        csr = mat.tocsr()
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        # Converting to CSR and back to dense must reproduce the dense matrix.
        assert allclose(csr.todense(), mat.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_transpose_roundtrip(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder).T

        csr = mat.tocsr()
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == mat.shape
        assert allclose(csr.todense(), mat.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_structure_valid(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder)

        csr = mat.tocsr()
        indptr = np.asarray(csr.indptr)
        indices = np.asarray(csr.indices)
        assert indptr.shape == (shape[0] + 1,)
        assert indptr[0] == 0
        assert indptr[-1] == indices.shape[0]
        assert np.all(np.diff(indptr) >= 0)
        assert np.all((indices >= 0) & (indices < shape[1]))
        # Each CSR row is column-sorted with no duplicate columns (CPU backend).
        for r in range(shape[0]):
            seg = indices[indptr[r]:indptr[r + 1]]
            assert np.all(np.diff(seg) > 0)

    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    def test_tocsr_units(self, cls):
        import brainunit as u

        shape = (20, 30)
        mat = cls((1.5 * u.mV, 0.2 * u.mV, 0.2, 123), shape=shape)

        csr = mat.tocsr()
        dense = mat.todense()
        assert isinstance(csr, brainevent.CSR)
        assert u.get_unit(csr.data) == u.get_unit(dense)
        assert u.math.allclose(csr.todense(), dense)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))


class Test_JITC_Normal_Data_API:
    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    def test_data_returns_only_weights(self, cls):
        # .data exposes only the trainable (wloc, wscale) pair, excluding prob/seed.
        mat = cls((1.5, 0.3, 0.2, 123), shape=(8, 6))
        assert isinstance(mat.data, tuple)
        assert len(mat.data) == 2
        wloc, wscale = mat.data
        assert allclose(wloc, mat.wloc)
        assert allclose(wscale, mat.wscale)

    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    def test_with_data_roundtrips_from_data(self, cls):
        # with_data accepts exactly what .data returns and preserves structure.
        mat = cls((1.5, 0.3, 0.2, 123), shape=(8, 6), corder=True)
        rebuilt = mat.with_data(mat.data)
        assert type(rebuilt) is cls
        assert allclose(rebuilt.wloc, mat.wloc)
        assert allclose(rebuilt.wscale, mat.wscale)
        assert rebuilt.prob == mat.prob
        assert rebuilt.seed == mat.seed
        assert rebuilt.shape == mat.shape
        assert rebuilt.corder == mat.corder

    def test_with_data_updates_both_params(self):
        mat = brainevent.JITCNormalR((1.5, 0.3, 0.2, 123), shape=(8, 6))
        updated = mat.with_data((2.5, 0.7))
        assert allclose(updated.wloc, 2.5)
        assert allclose(updated.wscale, 0.7)

    def test_with_data_preserves_unit(self):
        import brainunit as u
        mat = brainevent.JITCNormalR((1.5 * u.mV, 0.3 * u.mV, 0.2, 123), shape=(8, 6))
        updated = mat.with_data((2.5 * u.mV, 0.7 * u.mV))
        assert u.get_unit(updated.wloc) == u.get_unit(u.mV)
        assert allclose(u.get_mantissa(updated.wloc), 2.5)

    def test_with_data_unit_mismatch_raises(self):
        import brainunit as u
        mat = brainevent.JITCNormalR((1.5 * u.mV, 0.3 * u.mV, 0.2, 123), shape=(8, 6))
        with pytest.raises(AssertionError):
            mat.with_data((2.5, 0.7))  # dimensionless where mV is expected


# ---- The dense form must be the matvec's matrix ----
# ``chunk_size`` is derived from the *walked* dimension, so the matrix depends
# only on the generated matrix's own geometry: the column-oriented classes can
# materialize with their own shape and still agree with ``mat @ v``.

@pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
@pytest.mark.parametrize('corder', [True, False])
def test_todense_matches_matmul(cls, corder):
    shape = (12, 20)
    mat = cls((1.5, 0.15, 0.3, 42), shape=shape, corder=corder)
    v = jnp.asarray(np.random.rand(shape[1]).astype(np.float32))
    assert allclose(mat @ v, mat.todense() @ v, rtol=1e-4, atol=1e-4)


@requires_csr_backend
class Test_Notrans_Trans_Dense_CSR_Cross_Check:
    """One matrix, five independent producers of the same two products.

    The ``notrans`` and ``trans`` CUDA entry points are *separate kernels* --
    ``notrans`` gathers (``acc += w * v[j]`` into ``output[row]``), ``trans``
    scatters (``atomic_add(output[j], w * v[row])``). They are required to
    replay the same ``(seed, row, chunk_id, lane)`` stream, so for one matrix
    ``M`` of shape ``(a, b)``:

    * the ``notrans`` entry computes ``M @ v``   -- ``jit*mv(shape=(a, b), corder=True)``
    * the ``trans``   entry computes ``M.T @ u`` -- ``jit*mv(shape=(b, a), corder=False)``

    Both products are then recomputed from the two *materializations* --
    ``todense()`` and ``tocsr()`` -- and from the matrix recovered by feeding
    unit vectors through each kernel. Every route must agree.
    """

    SHAPES = [(12, 20), (17, 9), (33, 33)]

    @staticmethod
    def _matrix():
        return None

    @pytest.mark.parametrize('shape', SHAPES)
    def test_kernels_and_materializations_agree(self, shape):
        a, b = shape
        w, prob, seed = (1.5, 0.15), 0.2, 123
        v = jnp.asarray(np.random.rand(b).astype(np.float32))   # for M @ v
        u = jnp.asarray(np.random.rand(a).astype(np.float32))   # for M.T @ u

        # --- the two kernels, each through its own entry point ---------------
        y_notrans = jitnmv(*w, prob, v, seed, shape=(a, b), transpose=False, corder=True)
        y_trans = jitnmv(*w, prob, u, seed, shape=(b, a), transpose=False, corder=False)
        assert y_notrans.shape == (a,)
        assert y_trans.shape == (b,)

        # --- the same matrix, materialized two ways --------------------------
        dense = jitn(*w, prob, seed, shape=(a, b), transpose=False, corder=True)
        csr = jitn_to_csr(*w, prob, seed, shape=(a, b), corder=True)
        assert allclose(csr.todense(), dense)

        # --- and recovered from each kernel by unit vectors ------------------
        eye_b, eye_a = np.eye(b, dtype=np.float32), np.eye(a, dtype=np.float32)
        rec_notrans = np.stack(
            [np.asarray(jitnmv(*w, prob, jnp.asarray(eye_b[j]), seed, shape=(a, b),
                           transpose=False, corder=True)) for j in range(b)], axis=1)
        rec_trans = np.stack(
            [np.asarray(jitnmv(*w, prob, jnp.asarray(eye_a[i]), seed, shape=(b, a),
                           transpose=False, corder=False)) for i in range(a)], axis=1).T

        # every route describes one matrix
        assert np.array_equal(rec_notrans != 0, rec_trans != 0)
        assert allclose(rec_notrans, rec_trans, rtol=1e-5, atol=1e-5)
        assert allclose(rec_notrans, np.asarray(dense), rtol=1e-5, atol=1e-5)
        assert allclose(rec_notrans, np.asarray(csr.todense()), rtol=1e-5, atol=1e-5)

        # --- the same *operation*, cross-checked across all producers --------
        assert allclose(y_notrans, dense @ v, rtol=1e-4, atol=1e-4)
        assert allclose(y_notrans, csr @ v, rtol=1e-4, atol=1e-4)
        assert allclose(y_trans, u @ dense, rtol=1e-4, atol=1e-4)
        assert allclose(y_trans, u @ csr, rtol=1e-4, atol=1e-4)
        jax.block_until_ready((y_notrans, y_trans, dense, csr.data))

    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('k', [1, 4])
    def test_matmat_kernels_and_materializations_agree(self, shape, k):
        a, b = shape
        w, prob, seed = (1.5, 0.15), 0.2, 123
        B = jnp.asarray(np.random.rand(b, k).astype(np.float32))
        U = jnp.asarray(np.random.rand(a, k).astype(np.float32))

        y_notrans = jitnmm(*w, prob, B, seed, shape=(a, b), transpose=False, corder=True)
        y_trans = jitnmm(*w, prob, U, seed, shape=(b, a), transpose=False, corder=False)

        dense = jitn(*w, prob, seed, shape=(a, b), transpose=False, corder=True)
        csr = jitn_to_csr(*w, prob, seed, shape=(a, b), corder=True)

        assert allclose(y_notrans, dense @ B, rtol=1e-4, atol=1e-4)
        assert allclose(y_notrans, csr @ B, rtol=1e-4, atol=1e-4)
        assert allclose(y_trans, dense.T @ U, rtol=1e-4, atol=1e-4)
        assert allclose(y_trans, csr.todense().T @ U, rtol=1e-4, atol=1e-4)
        jax.block_until_ready((y_notrans, y_trans))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_event_driven_kernels_agree(self, shape):
        a, b = shape
        w, prob, seed = (1.5, 0.15), 0.2, 123
        spikes_b = jnp.asarray(np.random.rand(b) < 0.5)
        spikes_a = jnp.asarray(np.random.rand(a) < 0.5)

        y_notrans = binary_jitnmv(*w, prob, spikes_b, seed, shape=(a, b),
                              transpose=False, corder=True)
        y_trans = binary_jitnmv(*w, prob, spikes_a, seed, shape=(b, a),
                            transpose=False, corder=False)

        dense = jitn(*w, prob, seed, shape=(a, b), transpose=False, corder=True)
        csr = jitn_to_csr(*w, prob, seed, shape=(a, b), corder=True)
        fv = spikes_b.astype(dense.dtype)
        fu = spikes_a.astype(dense.dtype)

        assert allclose(y_notrans, dense @ fv, rtol=1e-4, atol=1e-4)
        assert allclose(y_notrans, csr @ fv, rtol=1e-4, atol=1e-4)
        assert allclose(y_trans, fu @ dense, rtol=1e-4, atol=1e-4)
        assert allclose(y_trans, fu @ csr, rtol=1e-4, atol=1e-4)
        jax.block_until_ready((y_notrans, y_trans))


# ---------------------------------------------------------------------------
# WeightedArray (#201): ``WeightedArray @ JITC`` routes to the float JITC
# primitives, which multiply by the event value, skip zero events and draw the
# same matrix as the binary kernels.
# ---------------------------------------------------------------------------

from contextlib import nullcontext as _w_nullcontext

from brainevent._jit_normal.float import jitnmv_p, jitnmm_p
from brainevent._test_util import check_event_op_grads as _w_check_grads
from brainevent._test_util import jax_x64_enabled as _w_x64

_W_MV_BACKENDS = tuple(jitnmv_p.available_backends(platform))
_W_MM_BACKENDS = tuple(jitnmm_p.available_backends(platform))
_W_DTYPES = [
    jnp.int8, jnp.uint8, jnp.int16, jnp.int32,
    jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64,
]
_W_SHAPE = (20, 30)
_W_PROB = 0.3
_W_SEED = 123
_W_CLASSES = ['JITCNormalR', 'JITCNormalC']


def _w_dtype_id(dtype):
    return np.dtype(dtype).name


def _w_ctx(dtype):
    return _w_x64() if np.dtype(dtype) == np.float64 else _w_nullcontext()


def _w_weight_dtype(event_dtype):
    # float64 events exercise the float64 kernel instantiation.
    return jnp.float64 if np.dtype(event_dtype) == np.float64 else jnp.float32


def _w_params(dtype=jnp.float32):
    return (jnp.asarray(1.5, dtype=dtype), jnp.asarray(0.3, dtype=dtype))


def _w_matrix(cls_name, params, corder, backend=None):
    return getattr(brainevent, cls_name)(
        (params[0], params[1], _W_PROB, _W_SEED), shape=_W_SHAPE, corder=corder, backend=backend,
    )


def _w_event_shape(op):
    m, n = _W_SHAPE
    return {'mv': (n,), 'vm': (m,), 'mm': (n, 3), 'mr': (3, m)}[op]


def _w_apply(op, mat, x):
    return mat @ x if op in ('mv', 'mm') else x @ mat


def _w_events(shape, dtype, base=None):
    """Mostly-zero events with values ``[2, -1, 0, 0.5]`` (``uint8``: ``[2, 3, 0, 1]``)."""
    dtype = np.dtype(dtype)
    if base is None:
        if dtype == np.uint8:
            base = [2, 3, 0, 1]
        elif np.issubdtype(dtype, np.integer):
            base = [2, -1, 0, 1]
        else:
            base = [2.0, -1.0, 0.0, 0.5]
    n = int(np.prod(shape))
    values = np.resize(np.asarray(base), n)
    values = values[np.random.RandomState(0).permutation(n)].reshape(shape)
    return jnp.asarray(values.astype(dtype))


def _w_backends(op):
    return _W_MV_BACKENDS if op in ('mv', 'vm') else _W_MM_BACKENDS


_W_CASES = [
    pytest.param(backend, op, id=f'{backend}-{op}')
    for op in ('mv', 'vm', 'mm', 'mr')
    for backend in _w_backends(op)
]


@pytest.mark.parametrize('backend,op', _W_CASES)
@pytest.mark.parametrize('cls_name', _W_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('dtype', _W_DTYPES, ids=_w_dtype_id)
def test_weighted_jitc_grads_match_dense(backend, op, cls_name, corder, dtype):
    """Forward, weight/event gradients, JVP/VJP identity and vmap are exact."""
    half = np.dtype(dtype).itemsize == 2 and jnp.issubdtype(dtype, jnp.floating)
    tol = 1e-2 if half else 1e-4
    with _w_ctx(dtype):
        x = _w_events(_w_event_shape(op), dtype)
        params = _w_params(_w_weight_dtype(dtype))

        def op_fn(p, v):
            return _w_apply(op, _w_matrix(cls_name, p, corder, backend=backend), brainevent.WeightedArray(v))

        def dense_fn(p, z):
            return _w_apply(op, _w_matrix(cls_name, p, corder).todense(), z)

        _w_check_grads(op_fn, dense_fn, params, x, mode='weighted', rtol=tol, atol=tol)


@pytest.mark.parametrize('backend,op', _W_CASES)
@pytest.mark.parametrize('cls_name', _W_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
def test_weighted_jitc_draws_same_matrix_as_binary(backend, op, cls_name, corder):
    """``WeightedArray(mask)`` equals ``BinaryArray(mask)``; ``WeightedArray(x)`` equals ``x @ dense``."""
    params = _w_params()
    mat = _w_matrix(cls_name, params, corder, backend=backend)
    x = _w_events(_w_event_shape(op), jnp.float32)
    mask = x != 0
    y_bin = _w_apply(op, mat, brainevent.BinaryArray(mask))
    y_wmask = _w_apply(op, mat, brainevent.WeightedArray(mask.astype(jnp.float32)))
    assert np.allclose(np.asarray(y_bin), np.asarray(y_wmask), rtol=1e-5, atol=1e-5)
    y_w = _w_apply(op, mat, brainevent.WeightedArray(x))
    with jax.default_matmul_precision('highest'):
        y_ref = _w_apply(op, _w_matrix(cls_name, params, corder).todense(), x)
    assert np.allclose(np.asarray(y_w), np.asarray(y_ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('backend,op', _W_CASES)
@pytest.mark.parametrize('cls_name', _W_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
def test_weighted_jitc_sparse_events_and_bool(backend, op, cls_name, corder):
    """Very sparse (incl. all-zero) and boolean weighted events match the dense product."""
    params = _w_params()
    mat = _w_matrix(cls_name, params, corder, backend=backend)
    dense = _w_matrix(cls_name, params, corder).todense()
    shape = _w_event_shape(op)
    sparse = _w_events(shape, jnp.float32, base=[3.0] + [0.0] * 19)
    for x in (sparse, jnp.zeros(shape, jnp.float32), sparse != 0):
        y = _w_apply(op, mat, brainevent.WeightedArray(x))
        with jax.default_matmul_precision('highest'):
            y_ref = _w_apply(op, dense, x.astype(jnp.float32))
        assert y.dtype == jnp.float32
        assert np.allclose(np.asarray(y), np.asarray(y_ref), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('backend,op', _W_CASES)
def test_weighted_jitc_vmap_over_events(backend, op):
    params = _w_params()
    mat = _w_matrix(_W_CLASSES[0], params, False, backend=backend)
    dense = _w_matrix(_W_CLASSES[0], params, False).todense()
    xs = jnp.stack([_w_events(_w_event_shape(op), jnp.int8), -_w_events(_w_event_shape(op), jnp.int8)])
    ys = jax.vmap(lambda v: _w_apply(op, mat, brainevent.WeightedArray(v)))(xs)
    with jax.default_matmul_precision('highest'):
        ys_ref = jax.vmap(lambda v: _w_apply(op, dense, v.astype(jnp.float32)))(xs)
    assert np.allclose(np.asarray(ys), np.asarray(ys_ref), rtol=1e-4, atol=1e-4)


def test_weighted_jitc_integer_event_gradient_is_float0():
    params = _w_params()
    x = _w_events(_w_event_shape('mv'), jnp.int8)

    def f(p, v):
        return (_w_matrix(_W_CLASSES[0], p, True) @ brainevent.WeightedArray(v)).sum()

    _, g_x = jax.grad(f, argnums=(0, 1), allow_int=True)(params, x)
    assert g_x.dtype == jax.dtypes.float0


@pytest.mark.parametrize('backend,op', [c for c in _W_CASES if c.values[1] in ('mv', 'vm')])
@pytest.mark.parametrize('cls_name', _W_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
def test_weighted_jitc_mv_skips_zero_events(backend, op, cls_name, corder):
    """Zero events are skipped (gather and scatter kernels): a NaN weight in an inactive row does not propagate."""
    nan_params = (jnp.asarray(jnp.nan, dtype=jnp.float32), jnp.asarray(0.6, dtype=jnp.float32))
    mat = _w_matrix(cls_name, nan_params, corder, backend=backend)
    shape = _w_event_shape(op)
    y = _w_apply(op, mat, brainevent.WeightedArray(jnp.zeros(shape, jnp.float32)))
    assert np.array_equal(np.asarray(y), np.zeros(y.shape, np.float32))


@pytest.mark.skipif('numba' not in _W_MV_BACKENDS + _W_MM_BACKENDS, reason='requires the numba backend')
@pytest.mark.parametrize('op', ['mv', 'vm', 'mm', 'mr'])
@pytest.mark.parametrize('cls_name', _W_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
def test_weighted_jitc_numba_nan_event_propagates(op, cls_name, corder):
    """A NaN weighted event is active: ``NaN * w`` reaches exactly the connected outputs (gather and scatter)."""
    params = _w_params()
    mat = _w_matrix(cls_name, params, corder, backend='numba')
    dense = np.asarray(_w_matrix(cls_name, params, corder).todense())
    x = np.array(_w_events(_w_event_shape(op), jnp.float32))
    flat = x.reshape(-1)
    flat[np.flatnonzero(flat)[1]] = np.nan
    y = np.asarray(_w_apply(op, mat, brainevent.WeightedArray(jnp.asarray(x))))
    nan_in = np.isnan(x).astype(np.float32)
    expected_nan = _w_apply(op, (dense != 0).astype(np.float32), nan_in) > 0
    assert expected_nan.any()
    assert np.array_equal(np.isnan(y), expected_nan)
    y_ref = _w_apply(op, dense, np.nan_to_num(x, nan=0.0))
    assert np.allclose(y[~expected_nan], y_ref[~expected_nan], rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Inactive events contribute exactly zero (docs/specs/2026-10-24-inactive-row-
# nan-isolation.md). JITC weights come from parameters, so the case is a NaN /
# +-Inf *parameter*: outputs that receive only inactive contributions must be
# exactly 0 on every backend, outputs reached by an active event stay
# non-finite, and the parameter cotangent ignores inactive events even when
# the output cotangent is NaN there.
# ---------------------------------------------------------------------------

from brainevent._jit_normal.binary import binary_jitnmv_p, binary_jitnmm_p

_N_BAD_PARAMS = (
    (np.nan, 0.3), (np.inf, 0.3), (-np.inf, 0.3), (1.5, np.nan), (1.5, np.inf),
)
_N_KINDS = ('bool', 'float', 'weighted')


def _n_backends(op):
    if op in ('mv', 'vm'):
        prims = (binary_jitnmv_p, jitnmv_p)
    else:
        prims = (binary_jitnmm_p, jitnmm_p)
    names = []
    for prim in prims:
        for name in prim.available_backends(platform):
            if name not in names:
                names.append(name)
    return tuple(names)


_N_CASES = [
    pytest.param(backend, op, id=f'{backend}-{op}')
    for op in ('mv', 'vm', 'mm', 'mr')
    for backend in _n_backends(op)
]


def _n_partial_events(shape):
    """Events with values ``2, -1`` on a sparse subset and ``0`` elsewhere."""
    n = int(np.prod(shape))
    values = np.zeros(n, np.float32)
    active = np.random.RandomState(1).permutation(n)[: max(1, n // 6)]
    values[active] = np.resize(np.asarray([2.0, -1.0], np.float32), active.size)
    return values.reshape(shape)


def _n_wrap(kind, x):
    if kind == 'bool':
        return brainevent.BinaryArray(jnp.asarray(x != 0))
    if kind == 'float':
        return brainevent.BinaryArray(jnp.asarray(x))
    return brainevent.WeightedArray(jnp.asarray(x))


def _n_active_count(op, cls_name, corder, x):
    """Number of active events reaching each output, from the finite pattern."""
    pattern = (np.asarray(_w_matrix(cls_name, _w_params(), corder).todense()) != 0).astype(np.float32)
    return _w_apply(op, pattern, (x != 0).astype(np.float32))


def _n_params(bad):
    if isinstance(bad, tuple):
        return tuple(jnp.asarray(b, dtype=jnp.float32) for b in bad)
    return jnp.asarray(bad, dtype=jnp.float32)


@pytest.mark.parametrize('backend,op', _N_CASES)
@pytest.mark.parametrize('cls_name', _W_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('kind', _N_KINDS)
def test_inactive_events_isolate_nonfinite_params(backend, op, cls_name, corder, kind):
    """A NaN/Inf parameter reaches only outputs fed by an active event; the rest are exactly 0."""
    shape = _w_event_shape(op)
    partial = _n_partial_events(shape)
    count = _n_active_count(op, cls_name, corder, partial)
    assert (count > 0).any() and (count == 0).any()
    for bad in _N_BAD_PARAMS:
        mat = _w_matrix(cls_name, _n_params(bad), corder, backend=backend)
        y0 = np.asarray(_w_apply(op, mat, _n_wrap(kind, np.zeros(shape, np.float32))))
        assert np.array_equal(y0, np.zeros(y0.shape, y0.dtype)), (bad, y0)
        y = np.asarray(_w_apply(op, mat, _n_wrap(kind, partial)))
        assert np.array_equal(y[count == 0], np.zeros(int((count == 0).sum()), y.dtype)), bad
        # Active events times a non-finite parameter still propagate.
        assert not np.isfinite(y[count > 0]).any(), bad


@pytest.mark.parametrize('backend,op', _N_CASES)
@pytest.mark.parametrize('cls_name', _W_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('kind', _N_KINDS)
def test_inactive_events_param_cotangent_is_exactly_zero(backend, op, cls_name, corder, kind):
    """Parameter cotangents ignore inactive events, even under a NaN output cotangent."""
    shape = _w_event_shape(op)
    params = _w_params()
    for x in (np.zeros(shape, np.float32), _n_partial_events(shape)):
        ev = _n_wrap(kind, x)
        count = _n_active_count(op, cls_name, corder, x)

        def f(p):
            return _w_apply(op, _w_matrix(cls_name, p, corder, backend=backend), ev)

        y, vjp = jax.vjp(f, params)
        ct = jnp.asarray(np.random.RandomState(2).rand(*y.shape).astype(np.float32))
        silent = jnp.asarray(count == 0)
        g_nan = vjp(jnp.where(silent, jnp.nan, ct))[0]
        g_ref = vjp(jnp.where(silent, 0.0, ct))[0]
        for a, b in zip(jax.tree.leaves(g_nan), jax.tree.leaves(g_ref)):
            a, b = np.asarray(a), np.asarray(b)
            assert np.isfinite(a).all(), (x.any(), a)
            assert np.allclose(a, b, rtol=1e-5, atol=1e-5)
            if not x.any():
                assert np.array_equal(a, np.zeros_like(a))


_N_DTYPE_CASES = [
    pytest.param(backend, op, dtype, id=f'{backend}-{op}-{np.dtype(dtype).name}')
    for op in ('mv', 'mm')
    for backend in _n_backends(op)
    for dtype in ((jnp.float64, jnp.float16, jnp.bfloat16) if backend == 'cuda_raw' else (jnp.float64,))
]


@pytest.mark.parametrize('backend,op,dtype', _N_DTYPE_CASES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('kind', _N_KINDS)
def test_inactive_events_isolate_nonfinite_params_dtypes(backend, op, dtype, corder, kind):
    """Every weight-dtype kernel instantiation selects on activity."""
    cls_name = _W_CLASSES[0]
    with _w_ctx(dtype):
        shape = _w_event_shape(op)
        partial = _n_partial_events(shape)
        count = _n_active_count(op, cls_name, corder, partial)
        bad = _N_BAD_PARAMS[0]
        params = jax.tree.map(lambda b: jnp.asarray(b, dtype=dtype), _n_params(bad))
        mat = _w_matrix(cls_name, params, corder, backend=backend)
        for x in (np.zeros(shape, np.float32), partial):
            ev = _n_wrap(kind, x.astype(np.dtype(dtype)) if kind == 'weighted' else x)
            y = np.asarray(_w_apply(op, mat, ev), dtype=np.float64)
            silent = count == 0 if x.any() else np.ones(y.shape, bool)
            assert np.array_equal(y[silent], np.zeros(int(silent.sum()))), (bad, y)
            assert not np.isfinite(y[~silent]).any(), bad
