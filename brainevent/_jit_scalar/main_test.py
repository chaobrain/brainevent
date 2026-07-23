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
from brainevent._jit_scalar.binary import binary_jitsmv, binary_jitsmm
from brainevent._test_util import allclose, gen_events

# Keep GPU matmul reference numerics stable (avoid TF32 drift in dense @ B checks).
if jax.default_backend() == 'gpu' and jax.config.jax_default_matmul_precision is None:
    jax.config.update('jax_default_matmul_precision', 'highest')

# Every test in this module dispatches to the native ``numba`` backend (the only backend for
# JIT-connectivity kernels), which compiles per test and dominates wall-clock. Mark the whole
# module ``slow`` so the default ``pytest`` run skips it; CI runs it via ``pytest -m ""``.
pytestmark = pytest.mark.slow

platform = jax.default_backend()

# JITC CSR materialization goes through the ``jits_csr_count`` primitive, which
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
    def test_matvec(self, shape, corder):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = jnp.asarray(np.random.rand(shape[1]))

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2, rtol=1e-5, atol=1e-5)
        jax.block_until_ready((vector, out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_vecmat(self, shape, corder):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
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
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(shape[1], k))

        out1 = jitcr @ matrix
        out2 = (matrix.T @ jitcc).T
        assert allclose(out1, out2, rtol=1e-3, atol=1e-3)
        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    def test_matjit(self, k, shape, corder):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = jnp.asarray(np.random.rand(k, shape[0]))

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.T).T
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((matrix, out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matvec_event(self, shape, corder, asbool):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[1], asbool=asbool)

        out1 = jitcr @ vector
        out2 = vector @ jitcc
        assert allclose(out1, out2)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_vecmat_event(self, shape, corder, asbool):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        vector = gen_events(shape[0], asbool=asbool)

        out1 = vector @ jitcr
        out2 = jitcc @ vector
        assert allclose(out1, out2)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_jitmat_event(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([shape[1], k], asbool=asbool)

        out1 = jitcr @ matrix
        out2 = (matrix.value.T @ jitcc).T
        assert allclose(out1, out2)
        jax.block_until_ready((out1, out2))

    @pytest.mark.parametrize('k', [10])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('asbool', [True, False])
    def test_matjit_event(self, k, shape, corder, asbool):
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        matrix = gen_events([k, shape[0]], asbool=asbool)

        out1 = matrix @ jitcr
        out2 = (jitcc @ matrix.value.T).T
        print(out1 - out2)
        assert allclose(out1, out2, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((out1, out2))


class Test_JITC_Operator_Behavior:
    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_r_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
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
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_c_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarC((1.5, 0.1, 123), shape=shape, corder=corder)
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
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_r_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder).T
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
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_c_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarC((1.5, 0.1, 123), shape=shape, corder=corder).T
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
        assert allclose(r5, r6, atol=1e-2, rtol=1e-2)
        assert allclose(r7, r8, atol=1e-2, rtol=1e-2)
        jax.block_until_ready((dense_mv, dense_mm, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    def test_jitc_scalar_unit_operator_behavior(self, cls):
        import brainunit as u
        shape = (20, 30)
        weight = 2.1 * u.mV
        mat = cls((weight, 0.2, 123), shape=shape)
        dense_mv = mat.mv.todense()
        dense_mm = mat.mm.todense()

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
        jitcr = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder)
        jitcc = jitcr.T

        out1 = jitcr.mv.todense()
        out2 = jitcc.mv.todense().T
        out3 = jitcr.T.mv.todense().T
        out4 = jitcc.T.mv.todense()
        assert allclose(out1, out2)
        assert allclose(out1, out3)
        assert allclose(out1, out4)
        jax.block_until_ready((out1, out2, out3, out4))

    # Covering set: every value of shape/corder/weight appears at least once (4 rows instead
    # of the full 2x2x2 = 8 product; gradient correctness is value-independent).
    @pytest.mark.parametrize('shape,corder,weight', [
        (shapes[0], True, -1.),
        (shapes[1], False, 1.),
        (shapes[0], False, 1.),
        (shapes[1], True, -1.),
    ])
    def test_vjp(self, shape, corder, weight):
        base = brainevent.JITCScalarR((1.0, 0.1, 123), shape=shape, corder=corder).mv.todense()

        def f_dense_vjp(weight):
            res = base * weight
            return res

        ct = brainstate.random.random(shape)
        primals, f_vjp = jax.vjp(f_dense_vjp, weight)
        true_weight_grad, = f_vjp(ct)

        expected_weight_grad = (ct * base).sum()
        assert allclose(true_weight_grad, expected_weight_grad)

        def f_jitc_vjp(weight):
            mat = brainevent.JITCScalarR((weight, 0.1, 123), shape=shape, corder=corder)
            return mat.mv.todense()

        primals, f_vjp2 = jax.vjp(f_jitc_vjp, weight)
        jitc_weight_grad, = f_vjp2(ct)

        assert allclose(true_weight_grad, jitc_weight_grad)
        jax.block_until_ready((base, ct, primals, true_weight_grad, expected_weight_grad, jitc_weight_grad))

    @pytest.mark.parametrize('shape,corder,weight', [
        (shapes[0], True, -1.),
        (shapes[1], False, 1.),
        (shapes[0], False, 1.),
        (shapes[1], True, -1.),
    ])
    def test_jvp(self, shape, corder, weight):
        base = brainevent.JITCScalarR((1., 0.1, 123), shape=shape, corder=corder).mv.todense()
        tagents = (brainstate.random.random(),)

        def f_dense_jvp(weight):
            res = base * weight
            return res

        def f_jitc_jvp(weight):
            mat = brainevent.JITCScalarR((weight, 0.1, 123), shape=shape, corder=corder)
            return mat.mv.todense()

        primals, true_grad = jax.jvp(f_dense_jvp, (weight,), tagents)
        primals, jitc_grad = jax.jvp(f_jitc_jvp, (weight,), tagents)
        assert allclose(true_grad, jitc_grad)
        jax.block_until_ready((base, tagents[0], primals, true_grad, jitc_grad))


class Test_JITC_Scalar_Validation:
    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('prob', [-0.1, 1.1, float('nan')])
    def test_invalid_prob_raises(self, cls, prob):
        with pytest.raises(ValueError, match='prob'):
            cls((1.5, prob, 123), shape=(8, 6))

    # @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    # @pytest.mark.parametrize('corder', [True, False])
    # def test_zero_prob_dense_matvec_matmat(self, cls, corder):
    #     shape = (8, 6)
    #     mat = cls((1.5, 0.0, 123), shape=shape, corder=corder)
    #
    #     dense = mat.mv.todense()
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

    def test_with_data_accepts_scalar(self):
        mat = brainevent.JITCScalarR((1.5, 0.1, 123), shape=(8, 6))
        updated = mat.with_data(2.0)
        assert allclose(updated.weight, 2.0)
        assert updated.prob == mat.prob
        assert updated.seed == mat.seed
        assert updated.shape == mat.shape

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    def test_data_returns_only_weight(self, cls):
        # .data exposes only the trainable weight, excluding prob/seed.
        mat = cls((1.5, 0.1, 123), shape=(8, 6))
        assert allclose(mat.data, mat.weight)
        # A scalar family has a single trainable weight, not a (weight, prob, seed) tuple.
        assert not isinstance(mat.data, tuple)

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    def test_with_data_roundtrips_from_data(self, cls):
        # with_data accepts exactly what .data returns and preserves structure.
        mat = cls((1.5, 0.1, 123), shape=(8, 6), corder=True)
        rebuilt = mat.with_data(mat.data)
        assert type(rebuilt) is cls
        assert allclose(rebuilt.weight, mat.weight)
        assert rebuilt.prob == mat.prob
        assert rebuilt.seed == mat.seed
        assert rebuilt.shape == mat.shape
        assert rebuilt.corder == mat.corder

    def test_with_data_preserves_unit(self):
        import brainunit as u
        mat = brainevent.JITCScalarR((1.5 * u.mV, 0.1, 123), shape=(8, 6))
        updated = mat.with_data(2.0 * u.mV)
        assert u.get_unit(updated.weight) == u.get_unit(u.mV)
        assert allclose(u.get_mantissa(updated.weight), 2.0)

    def test_with_data_unit_mismatch_raises(self):
        import brainunit as u
        mat = brainevent.JITCScalarR((1.5 * u.mV, 0.1, 123), shape=(8, 6))
        with pytest.raises(AssertionError):
            mat.with_data(2.0)  # dimensionless where mV is expected


class Test_JITC_Ambiguous_Materialization:
    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    def test_bare_materialization_raises(self, cls):
        # mv and mm draw different matrices, so bare todense/tocsr/tocsc/tocoo are
        # ambiguous and must raise; mat.mv / mat.mm select the mode explicitly.
        mat = cls((1.5, 0.1, 123), shape=(8, 6))
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
class Test_JITC_To_CSR:
    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_roundtrip(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 123), shape=shape, corder=corder)

        csr = mat.mv.tocsr()
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        # Converting to CSR and back to dense must reproduce the dense matrix.
        assert allclose(csr.todense(), mat.mv.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_transpose_roundtrip(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 123), shape=shape, corder=corder).T

        csr = mat.mv.tocsr()
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == mat.shape
        assert allclose(csr.todense(), mat.mv.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_structure_valid(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 123), shape=shape, corder=corder)

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
        # Scalar matrices store the constant weight at every connection.
        assert allclose(csr.data, jnp.full_like(csr.data, 1.5))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    def test_tocsr_units(self, cls):
        import brainunit as u

        shape = (20, 30)
        mat = cls((1.5 * u.mV, 0.2, 123), shape=shape)

        csr = mat.mv.tocsr()
        dense_mv = mat.mv.todense()
        assert isinstance(csr, brainevent.CSR)
        assert u.get_unit(csr.data) == u.get_unit(dense_mv)
        assert u.math.allclose(csr.todense(), dense_mv)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))


@requires_gpu
class Test_JITC_Materialization_Matches_Binary:
    """Cross-check: the materialized matrix must reproduce exactly the matrix the
    event-driven ``binary_jits*`` operators use internally.

    The same ``(weight, prob, seed, shape, corder)`` connectivity is materialized
    *two independent ways* — ``mat.<mode>.todense()`` and ``mat.<mode>.tocsr()`` —
    and a **float 0/1 spike** is pushed through ``@``.  Both results must equal the
    direct event-driven computation (``binary_jitsmv`` for mv, ``binary_jitsmm`` for
    mm), which never materializes the matrix.  Since the mv (32-lane) and mm
    (4-thread AW-T4) kernels draw *different* matrices, each mode is validated
    against its own binary operator, covering all cases.
    """

    # The mv/mm views regenerate ``jits(shape, transpose, corder, matrix_mode)``
    # from the matrix's ``_materialize_params()``; feeding those same
    # ``(shape, transpose, corder)`` to ``binary_jits*`` reproduces the identical
    # matrix, so ``materialized @ spike`` must equal the event-driven result.

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_mv_matvec_dense_csr_binary_agree(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 123), shape=shape, corder=corder)

        # The concrete mv matrix, materialized two independent ways.
        dense = mat.mv.todense()
        csr = mat.mv.tocsr()
        assert isinstance(csr, brainevent.CSR)

        # Float 0/1 spike vector sized to the materialized matrix's columns.
        spike = gen_events(dense.shape[1], asbool=False).value
        out_dense = dense @ spike
        out_csr = csr @ spike

        # binary_jitsmv regenerates the SAME mv matrix from identical parameters.
        gen_shape, gen_transpose = mat._materialize_params()
        out_binary = binary_jitsmv(
            mat.weight, mat.prob, spike, mat.seed,
            shape=gen_shape, transpose=gen_transpose, corder=mat.corder,
        )

        assert allclose(out_dense, out_binary, rtol=1e-4, atol=1e-4)
        assert allclose(out_csr, out_binary, rtol=1e-4, atol=1e-4)
        jax.block_until_ready((out_dense, out_csr, out_binary))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    @pytest.mark.parametrize('k', [4, 7])
    def test_mm_matmat_dense_csr_binary_agree(self, cls, corder, k):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 123), shape=shape, corder=corder)

        # The concrete mm matrix (different from mv), materialized two ways.
        dense = mat.mm.todense()
        csr = mat.mm.tocsr()
        assert isinstance(csr, brainevent.CSR)

        # Float 0/1 spike matrix sized to the materialized matrix's columns.
        spikes = gen_events((dense.shape[1], k), asbool=False).value
        out_dense = dense @ spikes
        out_csr = csr @ spikes

        # binary_jitsmm regenerates the SAME mm matrix from identical parameters.
        gen_shape, gen_transpose = mat._materialize_params()
        out_binary = binary_jitsmm(
            mat.weight, mat.prob, spikes, mat.seed,
            shape=gen_shape, transpose=gen_transpose, corder=mat.corder,
        )

        assert allclose(out_dense, out_binary, rtol=1e-4, atol=1e-4)
        assert allclose(out_csr, out_binary, rtol=1e-4, atol=1e-4)
        jax.block_until_ready((out_dense, out_csr, out_binary))
