# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Tests for ``brainevent/_csr/diag/diag_add.py``, grouped by behavior."""

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
import brainevent._csr.diag.diag_add as diag_add_mod
from brainevent._csr._test_util import small_csr


# Module tests
# ------------

def test_diag_add_int64_case_raises(monkeypatch):
    # Materialising a >int32_max structure is impractical, so shrink the
    # threshold to force the int64 branch on a tiny matrix.
    monkeypatch.setattr(diag_add_mod, "_INT32_MAX", 1)
    m = small_csr()
    with pytest.raises(NotImplementedError, match="int64 indptr offsets"):
        m.diag_add(jnp.array([1.0, 1.0], dtype=jnp.float32))


def test_diag_add_int32_case_still_works():
    m = small_csr()
    out = m.diag_add(jnp.array([5.0, 7.0], dtype=jnp.float32))
    assert out.indices.dtype == jnp.int32
    assert out.indptr.dtype == jnp.int32
    dense = np.asarray(out.todense())
    expected = np.array([[6.0, 0.0, 2.0], [0.0, 10.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(dense, expected)


@pytest.mark.parametrize('cls_name', ['CSR', 'CSC'])
def test_diag_add_promotes_a_float64_diagonal_to_the_common_dtype(cls_name):
    """Under x64, ``jnp.ones(n)`` is float64; a float32 matrix must accept it like ``W + diag(d)``."""
    import brainevent
    from brainevent._test_util import jax_x64_enabled

    dense = np.array([[1., 0., 2.], [0., 0., 3.], [4., 0., 0.]], dtype=np.float32)
    with jax_x64_enabled():
        mat = getattr(brainevent, cls_name).fromdense(jnp.asarray(dense))
        diag = jnp.asarray([0.5, 1.5, 2.5])
        assert mat.data.dtype == jnp.float32 and diag.dtype == jnp.float64
        out = mat.diag_add(diag)
        assert out.data.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(out.todense()), dense + np.diag(np.asarray(diag)))


def test_diag_add_promotes_mixed_dtypes_and_keeps_units():
    import brainunit as u
    import brainevent
    from brainevent._test_util import jax_x64_enabled

    dense = np.array([[1., 0., 2.], [0., 0., 3.], [4., 0., 0.]], dtype=np.float32)
    with jax_x64_enabled():
        mat = brainevent.CSR.fromdense(jnp.asarray(dense) * u.mV)
        out = mat.diag_add(jnp.asarray([1., 2., 3.]) * u.mV)
        assert u.get_unit(out.data) == u.mV
        assert out.data.dtype == jnp.float64
        np.testing.assert_allclose(u.get_mantissa(out.todense()), dense + np.diag([1., 2., 3.]))
        # Promotion does not relax the unit check.
        with pytest.raises(u.DimensionMismatchError):
            mat.diag_add(jnp.asarray([1., 2., 3.]) * u.second)


def test_diag_add_promotes_integer_data_with_a_float_diagonal():
    import brainevent

    dense = np.array([[1, 0, 2], [0, 0, 3], [4, 0, 0]], dtype=np.int32)
    mat = brainevent.CSR.fromdense(jnp.asarray(dense))
    out = mat.diag_add(jnp.asarray([0.5, 0.5, 0.5], dtype=jnp.float32))
    assert out.data.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(out.todense()), dense + 0.5 * np.eye(3))


# Sparse object diagonal addition
# -------------------------------

@pytest.mark.slow
class Test_diag_add:
    # ``diag_add`` computes ``A + diag(d)`` *exactly*: diagonal entries absent
    # from the sparsity pattern must be inserted, which changes ``indices`` /
    # ``indptr``. The expected dense result is therefore ``dense + diag(d)`` with
    # NO re-masking -- re-applying the sparsity mask (as a previous version of
    # these tests did) would have hidden the dropped-insertion bug.
    @pytest.mark.parametrize('shape', [(200, 300), (100, 50), (400, 400)])
    def test_csr(self, shape):
        dense = brainstate.random.rand(*shape)
        mask = (dense < 0.1) & (dense != 0.)
        dense = jnp.where(mask, dense, 0.)
        csr = brainevent.CSR.fromdense(dense)
        diag = brainstate.random.rand(min(shape)).astype(dense.dtype)
        new_csr = csr.diag_add(diag)

        new_dense = new_csr.todense()
        expected = dense.at[jnp.diag_indices(min(shape))].add(diag)

        assert jnp.allclose(new_dense, expected)
        # Missing diagonals were inserted, so the structure grew.
        assert new_csr.nse >= csr.nse

        jax.block_until_ready((expected, diag, new_dense))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50), (400, 400)])
    def test_csc(self, shape):
        dense = brainstate.random.rand(*shape)
        mask = (dense < 0.1) & (dense != 0.)
        dense = jnp.where(mask, dense, 0.)
        csc = brainevent.CSC.fromdense(dense)
        diag = brainstate.random.rand(min(shape)).astype(dense.dtype)
        new_csc = csc.diag_add(diag)

        new_dense = new_csc.todense()
        expected = dense.at[jnp.diag_indices(min(shape))].add(diag)

        assert jnp.allclose(new_dense, expected)
        assert new_csc.nse >= csc.nse

        jax.block_until_ready((expected, diag, new_dense))

    @pytest.mark.parametrize('shape', [(200, 300), (100, 50), (400, 400)])
    def test_csr_and_csc(self, shape):
        dense = brainstate.random.rand(*shape)
        mask = (dense < 0.1) & (dense != 0.)
        dense = jnp.where(mask, dense, 0.)
        csr = brainevent.CSR.fromdense(dense)
        csc = csr.T
        diag = brainstate.random.rand(min(shape)).astype(dense.dtype)
        new_csc = csc.diag_add(diag)

        new_dense = new_csc.todense().T
        expected = dense.at[jnp.diag_indices(min(shape))].add(diag)

        assert jnp.allclose(new_dense, expected)

        jax.block_until_ready((expected, diag, new_dense))

    # ------------------------------------------------------------------
    # Exactness and structural behaviour on small, explicit matrices.
    # ------------------------------------------------------------------
    def test_inserts_missing_diagonal_exact(self):
        # Diagonals (0, 0) and (1, 1) are missing; (2, 2) is present.
        dense = jnp.array([[0., 7., 0.],
                           [4., 0., 0.],
                           [0., 0., 5.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        diag = jnp.array([1., 2., 3.], dtype=jnp.float32)
        new_csr = csr.diag_add(diag)

        expected = jnp.array([[1., 7., 0.],
                              [4., 2., 0.],
                              [0., 0., 8.]], dtype=jnp.float32)
        assert jnp.allclose(new_csr.todense(), expected)
        # Two missing diagonals were inserted.
        assert int(new_csr.nse) == int(csr.nse) + 2
        # The full diagonal is now structurally present.
        assert int(new_csr.nse) == 5

    def test_inserted_indices_stay_sorted_within_row(self):
        # Row 0 has columns [2]; inserting the diagonal at column 0 must keep
        # the row's column indices ascending -> [0, 2].
        dense = jnp.array([[0., 0., 9.],
                           [0., 0., 0.],
                           [0., 0., 0.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        new_csr = csr.diag_add(jnp.array([1., 1., 1.], dtype=jnp.float32))

        indptr = np.asarray(new_csr.indptr)
        indices = np.asarray(new_csr.indices)
        for r in range(dense.shape[0]):
            row_cols = indices[indptr[r]:indptr[r + 1]]
            assert list(row_cols) == sorted(row_cols)
        assert jnp.allclose(new_csr.todense(),
                            jnp.array([[1., 0., 9.],
                                       [0., 1., 0.],
                                       [0., 0., 1.]], dtype=jnp.float32))

    def test_empty_rows_get_diagonal(self):
        # An entirely empty matrix becomes a pure diagonal matrix.
        n = 5
        csr = brainevent.CSR.fromdense(jnp.zeros((n, n), dtype=jnp.float32))
        diag = jnp.arange(1, n + 1, dtype=jnp.float32)
        new_csr = csr.diag_add(diag)
        assert int(new_csr.nse) == n
        assert jnp.allclose(new_csr.todense(), jnp.diag(diag))

    def test_full_diagonal_present_keeps_structure(self):
        # When every diagonal already exists, no insertion happens: the indices
        # and indptr are unchanged and only the values move.
        dense = jnp.array([[2., 1., 0.],
                           [0., 3., 1.],
                           [1., 0., 4.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        diag = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
        new_csr = csr.diag_add(diag)
        assert int(new_csr.nse) == int(csr.nse)
        np.testing.assert_array_equal(np.asarray(new_csr.indices),
                                      np.asarray(csr.indices))
        np.testing.assert_array_equal(np.asarray(new_csr.indptr),
                                      np.asarray(csr.indptr))
        assert jnp.allclose(new_csr.todense(),
                            dense.at[jnp.diag_indices(3)].add(diag))

    def test_zero_diag_value_still_materialises_full_diagonal(self):
        # Adding a zero diagonal does not change the dense values but still
        # materialises the (explicit-zero) diagonal entries -> structure grows.
        dense = jnp.array([[0., 2.],
                           [3., 0.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        new_csr = csr.diag_add(jnp.zeros(2, dtype=jnp.float32))
        assert int(new_csr.nse) == 4
        assert jnp.allclose(new_csr.todense(), dense)

    @pytest.mark.parametrize('shape', [(7, 7), (10, 4), (4, 10)])
    def test_csc_matches_csr(self, shape):
        dense = brainstate.random.rand(*shape)
        dense = jnp.where(dense < 0.3, dense, 0.).astype(jnp.float32)
        diag = brainstate.random.rand(min(shape)).astype(jnp.float32)
        csr_res = brainevent.CSR.fromdense(dense).diag_add(diag).todense()
        csc_res = brainevent.CSC.fromdense(dense).diag_add(diag).todense()
        assert jnp.allclose(csr_res, csc_res)
        assert jnp.allclose(csr_res, dense.at[jnp.diag_indices(min(shape))].add(diag))

    def test_diag_add_under_jit(self):
        dense = jnp.array([[0., 1., 0.],
                           [0., 0., 2.],
                           [3., 0., 0.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        diag = jnp.array([1., 1., 1.], dtype=jnp.float32)

        @jax.jit
        def run(mat, d):
            return mat.diag_add(d)

        # A jit argument has a traced structure, so the host-side plan must be
        # prepared beforehand.
        with pytest.raises(ValueError, match='prepare_diag_add'):
            run(csr, diag)
        new_csr = run(csr.prepare_diag_add(), diag)
        assert not isinstance(new_csr.indices, jax.core.Tracer)
        assert jnp.allclose(new_csr.todense(),
                            dense.at[jnp.diag_indices(3)].add(diag))
        # The result carries its own plan, so a second jitted diag_add works.
        again = run(new_csr, diag)
        assert jnp.allclose(again.todense(),
                            dense.at[jnp.diag_indices(3)].add(2 * diag))
        # A closed-over (concrete) matrix needs no preparation.
        closed = jax.jit(lambda d: csr.diag_add(d).todense())(diag)
        assert jnp.allclose(closed, dense.at[jnp.diag_indices(3)].add(diag))

    def test_repeated_diag_add_accumulates(self):
        dense = jnp.array([[0., 1.],
                           [2., 0.]], dtype=jnp.float32)
        csr = brainevent.CSR.fromdense(dense)
        d = jnp.array([1., 1.], dtype=jnp.float32)
        r1 = csr.diag_add(d)
        r2 = r1.diag_add(d)
        # The second add operates on the already-augmented (full-diagonal)
        # structure, so the pattern is stable and values accumulate.
        np.testing.assert_array_equal(np.asarray(r1.indices), np.asarray(r2.indices))
        np.testing.assert_array_equal(np.asarray(r1.indptr), np.asarray(r2.indptr))
        assert jnp.allclose(r2.todense(), dense.at[jnp.diag_indices(2)].add(d * 2))

    def test_diag_add_preserves_units(self):
        dense = jnp.array([[0., 2.],
                           [3., 0.]], dtype=jnp.float32) * u.mV
        csr = brainevent.CSR.fromdense(dense)
        diag = jnp.array([1., 1.], dtype=jnp.float32) * u.mV
        new_csr = csr.diag_add(diag)
        expected = dense + u.math.diag(diag)
        assert u.get_unit(new_csr.todense()) == u.mV
        assert u.math.allclose(new_csr.todense(), expected, atol=1e-6 * u.mV)
