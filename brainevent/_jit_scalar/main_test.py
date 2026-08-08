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
