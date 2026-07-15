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

# Every test in this module dispatches to the native ``numba`` backend (the only backend for
# JIT-connectivity kernels), which compiles per test and dominates wall-clock. Mark the whole
# module ``slow`` so the default ``pytest`` run skips it; CI runs it via ``pytest -m ""``.
pytestmark = pytest.mark.slow

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
        assert allclose(out1, out2, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((vector, out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecjit(self, shape: MatrixShape, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[0]))

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2, rtol=1e-5, atol=1e-5)
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
        dense_mv = mat.todense(matrix_mode="mv")
        dense_mm = mat.todense(matrix_mode="mm")

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
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_uniform_c_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCUniformC((-1.5, 1.5, 0.1, 123), shape=shape, corder=corder)
        dense_mv = mat.todense(matrix_mode="mv")
        dense_mm = mat.todense(matrix_mode="mm")

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
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_uniform_r_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCUniformR((-1.5, 1.5, 0.1, 123), shape=shape, corder=corder).T
        dense_mv = mat.todense(matrix_mode="mv")
        dense_mm = mat.todense(matrix_mode="mm")

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
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_uniform_c_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCUniformC((-1.5, 1.5, 0.1, 123), shape=shape, corder=corder).T
        dense_mv = mat.todense(matrix_mode="mv")
        dense_mm = mat.todense(matrix_mode="mm")

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
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

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
        assert u.get_unit(r1) == u.get_unit(r2)
        assert u.get_unit(r3) == u.get_unit(r4)
        assert allclose(u.get_mantissa(r1), u.get_mantissa(r2), rtol=1e-4, atol=1e-4)
        assert allclose(u.get_mantissa(r3), u.get_mantissa(r4), rtol=1e-4, atol=1e-4)
        jax.block_until_ready((right_vec, left_vec))


class Test_JITC_To_Dense:
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('matrix_mode', ['mv', 'mm'])
    @pytest.mark.parametrize('corder', [True, False])
    def test_todense(self, shape: MatrixShape, matrix_mode, corder):
        jitcr = brainevent.JITCUniformR((-0.1, 0.1, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        out1 = jitcr.todense(matrix_mode=matrix_mode)
        out2 = jitcc.todense(matrix_mode=matrix_mode).T
        out3 = jitcr.T.todense(matrix_mode=matrix_mode).T
        out4 = jitcc.T.todense(matrix_mode=matrix_mode)
        assert allclose(out1, out2)
        assert allclose(out1, out3)
        assert allclose(out1, out4)
        jax.block_until_ready((out1, out2, out3, out4))

    # Covering set: every value of shape/corder/wlow/whigh appears at least once (4 rows
    # instead of the full 2x2x2x2 = 16 product; gradient correctness is value-independent).
    @pytest.mark.parametrize('shape,corder,wlow,whigh', [
        (shapes[0], True, -1., 1.),
        (shapes[1], False, 0., 2.),
        (shapes[0], False, -1., 2.),
        (shapes[1], True, 0., 1.),
    ])
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

        assert allclose(true_wlow_grad, expected_wlow_grad, atol=1e-3, rtol=1e-3)
        assert allclose(true_whigh_grad, expected_whigh_grad, atol=1e-3, rtol=1e-3)

        print(true_wlow_grad, true_whigh_grad)
        print(expected_wlow_grad, expected_whigh_grad)

        def f_jitc_vjp(wlow, whigh):
            mat = brainevent.JITCUniformR((wlow, whigh, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

        primals, f_vjp2 = jax.vjp(f_jitc_vjp, wlow, whigh)
        jitc_wlow_grad, jitc_whigh_grad = f_vjp2(ct)

        assert allclose(true_wlow_grad, jitc_wlow_grad, rtol=1e-3, atol=1e-3)
        assert allclose(true_whigh_grad, jitc_whigh_grad, rtol=1e-3, atol=1e-3)
        jax.block_until_ready(
            (base, ct, primals, true_wlow_grad,
             true_whigh_grad, expected_wlow_grad, expected_whigh_grad,
             jitc_wlow_grad, jitc_whigh_grad)
        )

    @pytest.mark.parametrize('shape,corder,wlow,whigh', [
        (shapes[0], True, -1., 1.),
        (shapes[1], False, 0., 2.),
        (shapes[0], False, -1., 2.),
        (shapes[1], True, 0., 1.),
    ])
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

    # Covering set over the 2x2x2x2x2 = 32 product (every value appears at least once).
    @pytest.mark.parametrize('shape,corder,wlow,whigh,dwlow', [
        (shapes[0], True, -1., 1., 1.),
        (shapes[1], False, 0., 2., 2.),
        (shapes[0], False, -1., 2., 2.),
        (shapes[1], True, 0., 1., 1.),
    ])
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

    @pytest.mark.parametrize('shape,corder,wlow,whigh,dw_high', [
        (shapes[0], True, -1., 1., 1.),
        (shapes[1], False, 0., 2., 2.),
        (shapes[0], False, -1., 2., 2.),
        (shapes[1], True, 0., 1., 1.),
    ])
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


class Test_JITC_To_CSR:
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('matrix_mode', ['mv', 'mm'])
    def test_tocsr_matches_todense_for_r_and_c_views(self, corder, matrix_mode):
        shape = (20, 30)
        data = (-0.1, 0.1, 0.1, 123)

        jitcr = brainevent.JITCUniformR(data, shape=shape, corder=corder)
        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr_r = jitcr.tocsr(matrix_mode=matrix_mode)
        dense_r = jitcr.todense(matrix_mode=matrix_mode)
        assert allclose(csr_r.todense(), dense_r)

        jitcc = brainevent.JITCUniformC(data, shape=shape, corder=corder)
        base_r = brainevent.JITCUniformR(data, shape=shape[::-1], corder=not corder)
        expected_c = base_r.todense(matrix_mode=matrix_mode).T
        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr_c = jitcc.tocsr(matrix_mode=matrix_mode)
        assert allclose(jitcc.todense(matrix_mode=matrix_mode), expected_c)
        assert allclose(csr_c.todense(), expected_c)
        jax.block_until_ready((csr_r.data, csr_c.data, dense_r, expected_c))

    @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('matrix_mode', ['mv', 'mm'])
    def test_tocsr_accepts_light_matrix_mode(self, cls, corder, matrix_mode):
        shape = (20, 30)
        mat = cls((-0.1, 0.1, 0.0, 123), shape=shape, corder=corder)

        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr = mat.tocsr(matrix_mode=matrix_mode)
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        assert np.asarray(csr.indices).shape == (0,)
        assert np.asarray(csr.data).shape == (0,)
        assert np.all(np.asarray(csr.indptr) == 0)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_transpose_roundtrip(self, cls, corder):
        shape = (20, 30)
        mat = cls((-0.1, 0.1, 0.0, 123), shape=shape, corder=corder).T

        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr = mat.tocsr(matrix_mode="mv")
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == mat.shape
        assert np.asarray(csr.indices).shape == (0,)
        assert np.asarray(csr.data).shape == (0,)
        assert np.all(np.asarray(csr.indptr) == 0)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_structure_valid(self, cls, corder):
        shape = (20, 30)
        mat = cls((-0.1, 0.1, 0.0, 123), shape=shape, corder=corder)

        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr = mat.tocsr(matrix_mode="mm")
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

    @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    def test_tocsr_units(self, cls):
        import brainunit as u

        shape = (20, 30)
        weight = 2.1 * u.mV
        mat = cls((-weight, weight, 0.0, 123), shape=shape)

        with pytest.warns(UserWarning, match="corder.*ignored"):
            csr = mat.tocsr(matrix_mode="mv")
        assert isinstance(csr, brainevent.CSR)
        assert u.get_unit(csr.data) == u.get_unit(weight)
        assert np.asarray(u.get_mantissa(csr.data)).shape == (0,)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))


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


class Test_JITC_Uniform_Data_API:
    @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    def test_data_returns_only_weights(self, cls):
        # .data exposes only the trainable (wlow, whigh) pair, excluding prob/seed.
        mat = cls((0.1, 0.5, 0.2, 123), shape=(8, 6))
        assert isinstance(mat.data, tuple)
        assert len(mat.data) == 2
        wlow, whigh = mat.data
        assert allclose(wlow, mat.wlow)
        assert allclose(whigh, mat.whigh)

    @pytest.mark.parametrize('cls', [brainevent.JITCUniformR, brainevent.JITCUniformC])
    def test_with_data_roundtrips_from_data(self, cls):
        # with_data accepts exactly what .data returns and preserves structure.
        mat = cls((0.1, 0.5, 0.2, 123), shape=(8, 6), corder=True)
        rebuilt = mat.with_data(mat.data)
        assert type(rebuilt) is cls
        assert allclose(rebuilt.wlow, mat.wlow)
        assert allclose(rebuilt.whigh, mat.whigh)
        assert rebuilt.prob == mat.prob
        assert rebuilt.seed == mat.seed
        assert rebuilt.shape == mat.shape
        assert rebuilt.corder == mat.corder

    def test_with_data_updates_both_bounds(self):
        mat = brainevent.JITCUniformR((0.1, 0.5, 0.2, 123), shape=(8, 6))
        updated = mat.with_data((0.2, 0.8))
        assert allclose(updated.wlow, 0.2)
        assert allclose(updated.whigh, 0.8)

    def test_with_data_preserves_unit(self):
        import brainunit as u
        mat = brainevent.JITCUniformR((0.1 * u.mV, 0.5 * u.mV, 0.2, 123), shape=(8, 6))
        updated = mat.with_data((0.2 * u.mV, 0.8 * u.mV))
        assert u.get_unit(updated.wlow) == u.get_unit(u.mV)
        assert allclose(u.get_mantissa(updated.whigh), 0.8)

    def test_with_data_unit_mismatch_raises(self):
        import brainunit as u
        mat = brainevent.JITCUniformR((0.1 * u.mV, 0.5 * u.mV, 0.2, 123), shape=(8, 6))
        with pytest.raises(AssertionError):
            mat.with_data((0.2, 0.8))  # dimensionless where mV is expected

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
