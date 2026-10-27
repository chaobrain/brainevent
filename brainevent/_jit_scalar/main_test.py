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
from brainevent._jit_scalar.float import jits, jitsmv, jitsmm
from brainevent._jit_scalar.csr import jits_to_csr, jits_csr_count_p
from brainevent._test_util import allclose, gen_events

# Keep GPU matmul reference numerics stable (avoid TF32 drift in dense @ B checks).
if jax.default_backend() == 'gpu' and jax.config.jax_default_matmul_precision is None:
    jax.config.update('jax_default_matmul_precision', 'highest')

# Every test in this module dispatches to the native ``numba`` backend (the only backend for
# JIT-connectivity kernels), which compiles per test and dominates wall-clock. Mark the whole
# module ``slow`` so the default ``pytest`` run skips it; CI runs it via ``pytest -m ""``.
pytestmark = pytest.mark.slow

platform = jax.default_backend()
CSR_IMPLEMENTATIONS = tuple(jits_csr_count_p.available_backends(platform))

# JITC CSR materialization goes through the ``jits_csr_count`` primitive; run
# these tests on any platform that has a registered CSR backend.
requires_csr_backend = pytest.mark.skipif(
    not CSR_IMPLEMENTATIONS,
    reason=f'No JITC CSR materialization backend on platform={platform}',
)

platform = jax.default_backend()

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
        dense = mat.todense()

        left_vec = gen_events(shape[0], asbool=False).value
        right_vec = gen_events(shape[1], asbool=False).value
        left_mat = gen_events((5, shape[0]), asbool=False).value
        right_mat = gen_events((shape[1], 4), asbool=False).value

        r1 = left_vec @ mat
        r2 = left_vec @ dense
        r3 = mat @ right_vec
        r4 = dense @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense
        r7 = mat @ right_mat
        r8 = dense @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_c_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarC((1.5, 0.1, 123), shape=shape, corder=corder)
        dense = mat.todense()

        left_vec = gen_events(shape[0], asbool=False).value
        right_vec = gen_events(shape[1], asbool=False).value
        left_mat = gen_events((5, shape[0]), asbool=False).value
        right_mat = gen_events((shape[1], 4), asbool=False).value

        r1 = left_vec @ mat
        r2 = left_vec @ dense
        r3 = mat @ right_vec
        r4 = dense @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense
        r7 = mat @ right_mat
        r8 = dense @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_r_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarR((1.5, 0.1, 123), shape=shape, corder=corder).T
        dense = mat.todense()

        left_vec = jnp.asarray(np.random.rand(shape[1]))
        right_vec = jnp.asarray(np.random.rand(shape[0]))
        left_mat = jnp.asarray(np.random.rand(5, shape[1]))
        right_mat = jnp.asarray(np.random.rand(shape[0], 4))

        r1 = left_vec @ mat
        r2 = left_vec @ dense
        r3 = mat @ right_vec
        r4 = dense @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense
        r7 = mat @ right_mat
        r8 = dense @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-4, rtol=1e-4)
        assert allclose(r7, r8, atol=1e-4, rtol=1e-4)
        jax.block_until_ready((dense, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('corder', [True, False])
    def test_jitc_scalar_c_transpose_operator_behavior(self, corder):
        shape = (20, 30)
        mat = brainevent.JITCScalarC((1.5, 0.1, 123), shape=shape, corder=corder).T
        dense = mat.todense()

        left_vec = jnp.asarray(np.random.rand(shape[1]))
        right_vec = jnp.asarray(np.random.rand(shape[0]))
        left_mat = jnp.asarray(np.random.rand(5, shape[1]))
        right_mat = jnp.asarray(np.random.rand(shape[0], 4))

        r1 = left_vec @ mat
        r2 = left_vec @ dense
        r3 = mat @ right_vec
        r4 = dense @ right_vec
        r5 = left_mat @ mat
        r6 = left_mat @ dense
        r7 = mat @ right_mat
        r8 = dense @ right_mat
        assert allclose(r1, r2, atol=1e-4, rtol=1e-4)
        assert allclose(r3, r4, atol=1e-4, rtol=1e-4)
        assert allclose(r5, r6, atol=1e-2, rtol=1e-2)
        assert allclose(r7, r8, atol=1e-2, rtol=1e-2)
        jax.block_until_ready((dense, left_vec, right_vec, left_mat, right_mat, r1, r2, r3, r4, r5, r6, r7, r8))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    def test_jitc_scalar_unit_operator_behavior(self, cls):
        import brainunit as u
        shape = (20, 30)
        weight = 2.1 * u.mV
        mat = cls((weight, 0.2, 123), shape=shape)
        dense = mat.todense()

        right_vec = jnp.asarray(np.random.rand(shape[1]))
        left_vec = jnp.asarray(np.random.rand(shape[0]))

        r1 = mat @ right_vec
        r2 = dense @ right_vec
        r3 = left_vec @ mat
        r4 = left_vec @ dense
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

        out1 = jitcr.todense()
        out2 = jitcc.todense().T
        out3 = jitcr.T.todense().T
        out4 = jitcc.T.todense()
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
        base = brainevent.JITCScalarR((1.0, 0.1, 123), shape=shape, corder=corder).todense()

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
            return mat.todense()

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
        base = brainevent.JITCScalarR((1., 0.1, 123), shape=shape, corder=corder).todense()
        tagents = (brainstate.random.random(),)

        def f_dense_jvp(weight):
            res = base * weight
            return res

        def f_jitc_jvp(weight):
            mat = brainevent.JITCScalarR((weight, 0.1, 123), shape=shape, corder=corder)
            return mat.todense()

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


class Test_JITC_Materialization:
    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    def test_all_materializations_agree(self, cls):
        # There is one matrix, so every materialization route works and they agree.
        mat = cls((1.5, 0.1, 123), shape=(8, 6))
        dense = mat.todense()
        assert dense.shape == mat.shape
        assert allclose(mat.tocsr().todense(), dense)
        assert allclose(mat.tocsc().todense(), dense)
        assert allclose(mat.tocoo().todense(), dense)

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarC, brainevent.JITCScalarR])
    @pytest.mark.parametrize('corder', [True, False])
    def test_todense_matches_matmul(self, cls, corder):
        # ``chunk_size`` is derived from the walked dimension, so the two
        # ``(shape, transpose)`` pairs that describe this matrix draw the same
        # thing and the naive materialization agrees with ``mat @ v``.
        shape = (12, 20)
        mat = cls((1.5, 0.3, 42), shape=shape, corder=corder)
        v = brainstate.random.rand(shape[1])
        assert allclose(mat @ v, mat.todense() @ v, atol=1e-4, rtol=1e-4)


class Test_JITC_To_CSR:
    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_roundtrip(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 123), shape=shape, corder=corder)

        csr = mat.tocsr()
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == shape
        # Converting to CSR and back to dense must reproduce the dense matrix.
        assert allclose(csr.todense(), mat.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_transpose_roundtrip(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 123), shape=shape, corder=corder).T

        csr = mat.tocsr()
        assert isinstance(csr, brainevent.CSR)
        assert csr.shape == mat.shape
        assert allclose(csr.todense(), mat.todense())
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_tocsr_structure_valid(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 123), shape=shape, corder=corder)

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
        # Scalar matrices store the constant weight at every connection.
        assert allclose(csr.data, jnp.full_like(csr.data, 1.5))

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    def test_tocsr_units(self, cls):
        import brainunit as u

        shape = (20, 30)
        mat = cls((1.5 * u.mV, 0.2, 123), shape=shape)

        csr = mat.tocsr()
        dense = mat.todense()
        assert isinstance(csr, brainevent.CSR)
        assert u.get_unit(csr.data) == u.get_unit(dense)
        assert u.math.allclose(csr.todense(), dense)
        jax.block_until_ready((csr.data, csr.indices, csr.indptr))


class Test_JITC_Materialization_Matches_Binary:
    """Cross-check: the materialized matrix must reproduce exactly the matrix the
    event-driven ``binary_jits*`` operators use internally.

    The same ``(weight, prob, seed, shape, corder)`` connectivity is materialized
    *two independent ways* — ``mat.todense()`` and ``mat.tocsr()`` — and a
    **float 0/1 spike** is pushed through ``@``.  Both results must equal the
    direct event-driven computation (``binary_jitsmv`` for matvec,
    ``binary_jitsmm`` for matmat), which never materializes the matrix.  Matvec
    and matmat draw the same matrix, so both are checked against the one
    materialization.
    """

    # ``todense`` / ``tocsr`` regenerate ``jits(self.shape, transpose=False,
    # corder)``; feeding those same parameters to ``binary_jits*`` reproduces the
    # identical matrix, so ``materialized @ spike`` must equal the event-driven
    # result.

    @pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
    @pytest.mark.parametrize('corder', [True, False])
    def test_mv_matvec_dense_csr_binary_agree(self, cls, corder):
        shape = (20, 30)
        mat = cls((1.5, 0.2, 123), shape=shape, corder=corder)

        # The matrix, materialized two independent ways.
        dense = mat.todense()
        csr = mat.tocsr()
        assert isinstance(csr, brainevent.CSR)

        # Float 0/1 spike vector sized to the materialized matrix's columns.
        spike = gen_events(dense.shape[1], asbool=False).value
        out_dense = dense @ spike
        out_csr = csr @ spike

        # binary_jitsmv regenerates the SAME matrix from identical parameters.
        out_binary = binary_jitsmv(
            mat.weight, mat.prob, spike, mat.seed,
            shape=mat.shape, transpose=False, corder=mat.corder,
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

        # The same matrix, materialized two ways.
        dense = mat.todense()
        csr = mat.tocsr()
        assert isinstance(csr, brainevent.CSR)

        # Float 0/1 spike matrix sized to the materialized matrix's columns.
        spikes = gen_events((dense.shape[1], k), asbool=False).value
        out_dense = dense @ spikes
        out_csr = csr @ spikes

        # binary_jitsmm regenerates the SAME matrix from identical parameters.
        out_binary = binary_jitsmm(
            mat.weight, mat.prob, spikes, mat.seed,
            shape=mat.shape, transpose=False, corder=mat.corder,
        )

        assert allclose(out_dense, out_binary, rtol=1e-4, atol=1e-4)
        assert allclose(out_csr, out_binary, rtol=1e-4, atol=1e-4)
        jax.block_until_ready((out_dense, out_csr, out_binary))


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
        w, prob, seed = (1.5,), 0.2, 123
        v = jnp.asarray(np.random.rand(b).astype(np.float32))   # for M @ v
        u = jnp.asarray(np.random.rand(a).astype(np.float32))   # for M.T @ u

        # --- the two kernels, each through its own entry point ---------------
        y_notrans = jitsmv(*w, prob, v, seed, shape=(a, b), transpose=False, corder=True)
        y_trans = jitsmv(*w, prob, u, seed, shape=(b, a), transpose=False, corder=False)
        assert y_notrans.shape == (a,)
        assert y_trans.shape == (b,)

        # --- the same matrix, materialized two ways --------------------------
        dense = jits(*w, prob, seed, shape=(a, b), transpose=False, corder=True)
        csr = jits_to_csr(*w, prob, seed, shape=(a, b), corder=True)
        assert allclose(csr.todense(), dense)

        # --- and recovered from each kernel by unit vectors ------------------
        eye_b, eye_a = np.eye(b, dtype=np.float32), np.eye(a, dtype=np.float32)
        rec_notrans = np.stack(
            [np.asarray(jitsmv(*w, prob, jnp.asarray(eye_b[j]), seed, shape=(a, b),
                           transpose=False, corder=True)) for j in range(b)], axis=1)
        rec_trans = np.stack(
            [np.asarray(jitsmv(*w, prob, jnp.asarray(eye_a[i]), seed, shape=(b, a),
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
        w, prob, seed = (1.5,), 0.2, 123
        B = jnp.asarray(np.random.rand(b, k).astype(np.float32))
        U = jnp.asarray(np.random.rand(a, k).astype(np.float32))

        y_notrans = jitsmm(*w, prob, B, seed, shape=(a, b), transpose=False, corder=True)
        y_trans = jitsmm(*w, prob, U, seed, shape=(b, a), transpose=False, corder=False)

        dense = jits(*w, prob, seed, shape=(a, b), transpose=False, corder=True)
        csr = jits_to_csr(*w, prob, seed, shape=(a, b), corder=True)

        assert allclose(y_notrans, dense @ B, rtol=1e-4, atol=1e-4)
        assert allclose(y_notrans, csr @ B, rtol=1e-4, atol=1e-4)
        assert allclose(y_trans, dense.T @ U, rtol=1e-4, atol=1e-4)
        assert allclose(y_trans, csr.todense().T @ U, rtol=1e-4, atol=1e-4)
        jax.block_until_ready((y_notrans, y_trans))

    @pytest.mark.parametrize('shape', SHAPES)
    def test_event_driven_kernels_agree(self, shape):
        a, b = shape
        w, prob, seed = (1.5,), 0.2, 123
        spikes_b = jnp.asarray(np.random.rand(b) < 0.5)
        spikes_a = jnp.asarray(np.random.rand(a) < 0.5)

        y_notrans = binary_jitsmv(*w, prob, spikes_b, seed, shape=(a, b),
                              transpose=False, corder=True)
        y_trans = binary_jitsmv(*w, prob, spikes_a, seed, shape=(b, a),
                            transpose=False, corder=False)

        dense = jits(*w, prob, seed, shape=(a, b), transpose=False, corder=True)
        csr = jits_to_csr(*w, prob, seed, shape=(a, b), corder=True)
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

from brainevent._jit_scalar.float import jitsmv_p, jitsmm_p
from brainevent._test_util import check_event_op_grads as _w_check_grads
from brainevent._test_util import jax_x64_enabled as _w_x64

_W_MV_BACKENDS = tuple(jitsmv_p.available_backends(platform))
_W_MM_BACKENDS = tuple(jitsmm_p.available_backends(platform))
_W_DTYPES = [
    jnp.int8, jnp.uint8, jnp.int16, jnp.int32,
    jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64,
]
_W_SHAPE = (20, 30)
_W_PROB = 0.3
_W_SEED = 123
_W_CLASSES = ['JITCScalarR', 'JITCScalarC']


def _w_dtype_id(dtype):
    return np.dtype(dtype).name


def _w_ctx(dtype):
    return _w_x64() if np.dtype(dtype) == np.float64 else _w_nullcontext()


def _w_weight_dtype(event_dtype):
    # float64 events exercise the float64 kernel instantiation.
    return jnp.float64 if np.dtype(event_dtype) == np.float64 else jnp.float32


def _w_params(dtype=jnp.float32):
    return jnp.asarray(1.5, dtype=dtype)


def _w_matrix(cls_name, params, corder, backend=None):
    return getattr(brainevent, cls_name)(
        (params, _W_PROB, _W_SEED), shape=_W_SHAPE, corder=corder, backend=backend,
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

from brainevent._jit_scalar.binary import binary_jitsmv_p, binary_jitsmm_p

_N_BAD_PARAMS = (np.nan, np.inf, -np.inf)
_N_KINDS = ('bool', 'float', 'weighted')


def _n_backends(op):
    if op in ('mv', 'vm'):
        prims = (binary_jitsmv_p, jitsmv_p)
    else:
        prims = (binary_jitsmm_p, jitsmm_p)
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


# ---- reductions ----
import brainunit as _u


@pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
@pytest.mark.parametrize('axis', [None, 0, 1])
def test_sum_matches_dense(cls, axis):
    # Regression: ``sum()`` returned the bare homogeneous weight.
    M = cls((1.5 * _u.mV, 0.5, 3), shape=(6, 8))
    assert _u.math.allclose(M.sum(axis), M.todense().sum(axis))


@pytest.mark.parametrize('cls', [brainevent.JITCScalarR, brainevent.JITCScalarC])
def test_sum_rejects_bad_axis(cls):
    with pytest.raises(ValueError):
        cls((1.5, 0.5, 3), shape=(6, 8)).sum(2)


# ---- Exact density by thinning, and the ``corder=True`` default (0.4.0) ----
from brainevent._jit_scalar._test_util import dense_scalar_reference as _t_reference
from brainevent._jit_scalar.binary import binary_jitsmv as _t_binary_mv
from brainevent._jit_scalar.float import jits as _t_generate, jitsmv as _t_mv
from brainevent._test_util import (
    JITC_GOLDEN_0_3_0,
    check_jitc_grads_match_dense,
    check_jitc_ops_match_dense,
    jitc_kernel_orientations,
    jitc_mask_digest,
)

_T_PARAMS = (1.5,)
_T_THINNED = (0.3, 0.7, 0.93)
_T_CLASSES = ['JITCScalarR', 'JITCScalarC']


def _t_matrix(cls_name, prob, *, corder=True, seed=11, shape=(60, 90), params=_T_PARAMS):
    return getattr(brainevent, cls_name)((*params, prob, seed), shape=shape, corder=corder)


@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('prob', [0.6, 0.7, 0.9, 0.99, 1 / 3, 0.3, 0.15])
def test_density_is_exact_for_any_prob(prob, corder):
    """Density used to be ``2 / ceil(2 / p)`` (0.5 for p=0.6, 0.667 for p=0.9)."""
    shape = (300, 400)
    n = shape[0] * shape[1]
    dense = _t_matrix('JITCScalarR', prob, corder=corder, seed=3, shape=shape).todense()
    nnz = int((np.asarray(dense) != 0).sum())
    assert abs(nnz - prob * n) <= 4 * np.sqrt(n * prob * (1 - prob)) + 1


@pytest.mark.parametrize('cls_name', _T_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('prob', _T_THINNED)
def test_thinned_matrix_matches_reference_and_every_op(cls_name, prob, corder):
    mat = _t_matrix(cls_name, prob, corder=corder)
    expected = _t_reference(*_T_PARAMS, prob, 11, shape=(60, 90), corder=corder)
    np.testing.assert_array_equal(np.asarray(mat.todense()) != 0, expected != 0)
    check_jitc_ops_match_dense(mat)


@pytest.mark.parametrize('prob, corder', sorted(JITC_GOLDEN_0_3_0))
def test_integer_stride_draws_the_0_3_0_matrix(prob, corder):
    """When ``2 / p`` is an integer nothing is thinned: the 0.3.0 matrix, bit for bit."""
    mat = _t_matrix('JITCScalarR', prob, corder=corder, seed=123, shape=(37, 129))
    assert jitc_mask_digest(mat.todense()) == JITC_GOLDEN_0_3_0[(prob, corder)]


@pytest.mark.parametrize('cls_name', _T_CLASSES)
def test_default_corder_makes_events_at_matrix_event_driven(cls_name):
    mat = getattr(brainevent, cls_name)((*_T_PARAMS, 0.1, 0), shape=(20, 30))
    assert mat.corder is True
    assert mat.T.corder is False
    # ``events @ M`` must take the scatter walk (one stream per event) ...
    left = jitc_kernel_orientations(lambda x: brainevent.BinaryArray(x) @ mat, jnp.zeros(20, bool))
    assert [c for _, _, c in left] == [False]
    # ... and ``M @ v`` the row-parallel gather.
    right = jitc_kernel_orientations(lambda x: mat @ x, jnp.zeros(30, jnp.float32))
    assert [c for _, _, c in right] == [True]


@pytest.mark.parametrize('prob', _T_THINNED)
def test_matrix_passed_into_jit_draws_the_same_connectivity(prob):
    """Inside ``jit`` ``prob`` is a tracer; ``(cl, thr)`` must come out identical."""
    mat = _t_matrix('JITCScalarR', prob)
    v = jnp.linspace(-1., 1., 90, dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda m: m.todense())(mat)), np.asarray(mat.todense()))
    np.testing.assert_allclose(np.asarray(jax.jit(lambda m, x: m @ x)(mat, v)), np.asarray(mat @ v),
                               rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('cls_name', _T_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
def test_thinned_gradients_match_dense(cls_name, corder):
    check_jitc_grads_match_dense(
        lambda *p: _t_matrix(cls_name, 0.7, corder=corder, params=p), _T_PARAMS,
    )


@pytest.mark.parametrize('prob', [1.5, -0.1, float('nan'), float('inf')])
def test_functional_api_rejects_invalid_prob(prob):
    v = jnp.ones(5, jnp.float32)
    with pytest.raises(ValueError, match='prob'):
        _t_generate(*_T_PARAMS, prob, 0, shape=(4, 5))
    with pytest.raises(ValueError, match='prob'):
        _t_mv(*_T_PARAMS, prob, v, 0, shape=(4, 5))
    with pytest.raises(ValueError, match='prob'):
        _t_binary_mv(*_T_PARAMS, prob, v > 0, 0, shape=(4, 5))


def test_vmap_over_prob_matches_per_prob_calls():
    """A batched ``prob`` batches ``clen = (cl, thr)`` to shape ``(B, 2)``."""
    probs = jnp.asarray([0.1, 0.3, 0.7], jnp.float32)
    v = jnp.linspace(-1., 1., 30, dtype=jnp.float32)
    for f in (lambda p: _t_mv(*_T_PARAMS, p, v, 0, shape=(20, 30)),
              lambda p: _t_binary_mv(*_T_PARAMS, p, v > 0, 0, shape=(20, 30)),
              lambda p: _t_generate(*_T_PARAMS, p, 0, shape=(20, 30))):
        batched = np.asarray(jax.vmap(f)(probs))
        looped = np.stack([np.asarray(f(float(p))) for p in probs])
        np.testing.assert_allclose(batched, looped, rtol=1e-5, atol=1e-5)


from brainevent._test_util import jax_x64_enabled as _t_x64


@pytest.mark.parametrize('cls_name', _T_CLASSES)
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('prob', [0.3, 0.7])
def test_thinned_matrix_is_identical_under_x64(cls_name, prob, corder):
    """``(cl, thr)`` is computed in float32 either way, so x64 draws the same matrix."""
    digest = jitc_mask_digest(_t_matrix(cls_name, prob, corder=corder).todense())
    with _t_x64():
        mat = _t_matrix(cls_name, prob, corder=corder)
        assert jitc_mask_digest(mat.todense()) == digest
        check_jitc_ops_match_dense(mat)


def _t_within(observed, expected, sigma, what):
    assert abs(observed - expected) <= 5 * sigma, (
        f'{what}: observed {observed:.5g}, expected {expected:.5g} (5 sigma = {5 * sigma:.3g})'
    )


@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('prob', [0.25, 0.3, 0.7])
def test_thinned_walk_has_no_structure(prob, corder):
    """Thinning must not correlate streams, lanes or chunk positions.

    The three JITC families share one connectivity walk, so this scalar test
    covers all of them. A stream is a row of the matrix for ``corder=True`` and
    a column for ``corder=False``; it walks its positions in 4 chunks, lane
    ``l`` taking positions ``l, l + 32, ...`` of each chunk. ``p = 0.25`` is the
    unthinned (integer ``2 / p``) baseline. The sigmas are binomial, an upper
    bound for the walk's more regular counts.
    """
    dense = np.asarray(_t_matrix('JITCScalarR', prob, corder=corder, seed=5, shape=(400, 640)).todense()) != 0
    streams = dense if corder else dense.T
    n_streams, length = streams.shape
    chunk = -(-length // 4)
    sq = prob ** 2
    _t_within(streams.sum(1).mean(), length * prob,
              np.sqrt(length * prob * (1 - prob) / n_streams), 'non-zeros per stream')
    _t_within((streams[1:] & streams[:-1]).sum(1).mean(), length * sq,
              np.sqrt(length * sq * (1 - sq) / (n_streams - 1)), 'overlap of adjacent streams')
    _t_within((streams[:, 1:] & streams[:, :-1]).sum(0).mean(), n_streams * sq,
              np.sqrt(n_streams * sq * (1 - sq) / (length - 1)), 'overlap of adjacent positions')
    local = np.arange(length) % chunk
    for lane in range(32):
        part = streams[:, local % 32 == lane]
        _t_within(part.mean(), prob, np.sqrt(prob * (1 - prob) / part.size), f'density of lane {lane}')
    for quarter in range(4):
        part = streams[:, local * 4 // chunk == quarter]
        _t_within(part.mean(), prob, np.sqrt(prob * (1 - prob) / part.size), f'density of chunk quarter {quarter}')
