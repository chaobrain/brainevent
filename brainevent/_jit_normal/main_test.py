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
from brainevent._test_util import allclose, gen_events

# Every test in this module dispatches to the native ``numba`` backend (the only backend for
# JIT-connectivity kernels), which compiles per test and dominates wall-clock. Mark the whole
# module ``slow`` so the default ``pytest`` run skips it; CI runs it via ``pytest -m ""``.
pytestmark = pytest.mark.slow

platform = jax.default_backend()

# JITC CSR materialization goes through the ``jitn_csr_count`` primitive, which
# registers only a CUDA backend (``def_cuda_raw_kernel``); on CPU it raises
# ``NotImplementedError: ... not found for platform cpu``. Gate the CSR
# materialization tests to GPU.
requires_gpu = pytest.mark.skipif(
    platform != 'gpu',
    reason='JITC CSR materialization (csr_count) is a CUDA-only primitive',
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
        dense_mv = mat.mv.todense()
        dense_mm = mat.mm.todense()

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
        dense_mv = mat.mv.todense()
        dense_mm = mat.mm.todense()

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
        dense_mv = mat.mv.todense()
        dense_mm = mat.mm.todense()

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
        dense_mv = mat.mv.todense()
        dense_mm = mat.mm.todense()

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
        dense_mv = mat.mv.todense()

        right_vec = jnp.asarray(np.random.rand(shape[1]))
        left_vec = jnp.asarray(np.random.rand(shape[0]))

        r1 = mat @ right_vec
        r2 = dense_mv @ right_vec
        r3 = left_vec @ mat
        r4 = left_vec @ dense_mv
        assert u.math.allclose(r1, r2,
                               rtol=1e-4,
                               atol=1e-4 * u.get_unit(r2))
        assert u.math.allclose(r3, r4,
                               rtol=1e-4,
                               atol=1e-4 * u.get_unit(r4))
        jax.block_until_ready((right_vec, left_vec))


class Test_JITC_To_Dense:
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('transpose', [True, False])
    @pytest.mark.parametrize('corder', [True, False])
    def test_todense(self, shape, transpose, corder):
        jitcr = brainevent.JITCNormalR((1.5, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        out1 = jitcr.mv.todense()
        out2 = jitcc.mv.todense().T
        out3 = jitcr.T.mv.todense().T
        out4 = jitcc.T.mv.todense()
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
        z = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).mv.todense()
        mask = brainevent.JITCNormalR((1.0, 0.0, 0.1, 123), shape=shape, corder=corder).mv.todense()

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
            return mat.mv.todense()

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
        z = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).mv.todense()
        mask = brainevent.JITCNormalR((1.0, 0.0, 0.1, 123), shape=shape, corder=corder).mv.todense()
        tagents = (brainstate.random.random(), brainstate.random.random())

        def f_dense_jvp(wloc, wscale):
            res = z * wscale + mask * wloc
            return res

        def f_jitc_jvp(wloc, wscale):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.mv.todense()

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
        z = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).mv.todense()
        mask = brainevent.JITCNormalR((1.0, 0.0, 0.1, 123), shape=shape, corder=corder).mv.todense()

        def f_dense_jvp(wloc):
            res = z * wscale + mask * wloc
            return res

        primals, true_grad = jax.jvp(f_dense_jvp, (wloc,), (dwloc,))
        expected_grad = mask * dwloc
        assert allclose(true_grad, expected_grad)

        def f_jitc_jvp(wloc):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.mv.todense()

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
        z = brainevent.JITCNormalR((0.0, 1.0, 0.1, 123), shape=shape, corder=corder).mv.todense()
        mask = brainevent.JITCNormalR((1.0, 0.0, 0.1, 123), shape=shape, corder=corder).mv.todense()

        def f_dense_jvp(wscale):
            res = z * wscale + mask * wloc
            return res

        primals, true_grad = jax.jvp(f_dense_jvp, (wscale,), (dw_high,))
        expected_grad = z * dw_high
        assert allclose(true_grad, expected_grad)

        def f_jitc_jvp(wscale):
            mat = brainevent.JITCNormalR((wloc, wscale, 0.1, 123), shape=shape, corder=corder)
            return mat.mv.todense()

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


class Test_JITC_Ambiguous_Materialization:
    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    def test_bare_materialization_raises(self, cls):
        mat = cls((1.5, 0.1, 0.2, 123), shape=(8, 6))
        with pytest.raises(NotImplementedError):
            mat.todense()
        with pytest.raises(NotImplementedError):
            mat.tocsr()
        with pytest.raises(NotImplementedError):
            mat.tocsc()
        with pytest.raises(NotImplementedError):
            mat.tocoo()
        assert mat.mv.todense().shape == mat.shape
        assert mat.mm.todense().shape == mat.shape


@requires_gpu
class Test_JITC_Materialization_Matches_Binary:
    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_mv_matvec_dense_csr_binary_agree(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder)

        dense = mat.mv.todense()
        csr = mat.mv.tocsr()
        spike = gen_events(dense.shape[1], asbool=False).value

        gen_shape, gen_transpose = mat._materialize_params()
        out_binary = binary_jitnmv(
            mat.wloc, mat.wscale, mat.prob, spike, mat.seed,
            shape=gen_shape, transpose=gen_transpose, corder=mat.corder,
        )

        assert allclose(dense @ spike, out_binary, rtol=1e-4, atol=1e-4)
        assert allclose(csr @ spike, out_binary, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('k', [4, 7])
    def test_mm_matmat_dense_csr_binary_agree(self, cls, corder, k):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder)

        dense = mat.mm.todense()
        csr = mat.mm.tocsr()
        spikes = gen_events((dense.shape[1], k), asbool=False).value

        gen_shape, gen_transpose = mat._materialize_params()
        out_binary = binary_jitnmm(
            mat.wloc, mat.wscale, mat.prob, spikes, mat.seed,
            shape=gen_shape, transpose=gen_transpose, corder=mat.corder,
        )

        assert allclose(dense @ spikes, out_binary, rtol=1e-4, atol=1e-4)
        assert allclose(csr @ spikes, out_binary, rtol=1e-4, atol=1e-4)


@requires_gpu
class Test_JITC_To_CSR:
    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_roundtrip(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder)

        csr = mat.mv.tocsr()
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        # Converting to CSR and back to dense must reproduce the dense matrix.
        assert allclose(csr.todense(), mat.mv.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_transpose_roundtrip(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder).T

        csr = mat.mv.tocsr()
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == mat.shape
        assert allclose(csr.todense(), mat.mv.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCNormalR, brainevent.JITCNormalC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_structure_valid(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 0.2, 123), shape=shape, corder=corder)

        csr = mat.mv.tocsr()
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

        csr = mat.mv.tocsr()
        dense = mat.mv.todense()
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
