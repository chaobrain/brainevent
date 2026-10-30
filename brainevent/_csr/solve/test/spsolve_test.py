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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._csr.solve.spsolve import csr_solve
from brainevent._test_util import jax_x64_enabled

_DENSE = np.array([[4., 1., 0.], [1., 3., 0.], [0., 0., 2.]])


@pytest.mark.parametrize('data_dtype, b_dtype', [(jnp.float32, jnp.float64), (jnp.float64, jnp.float32)])
def test_csr_solve_promotes_mixed_dtypes(data_dtype, b_dtype):
    """``spsolve`` needs one dtype; mixed float32/float64 operands are promoted first."""
    with jax_x64_enabled():
        csr = brainevent.CSR.fromdense(jnp.asarray(_DENSE, dtype=data_dtype))
        b = jnp.asarray([1., 2., 3.], dtype=b_dtype)
        x = csr_solve(csr.data, csr.indices, csr.indptr, b)
        assert x.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(x), np.linalg.solve(_DENSE, np.asarray(b, np.float64)), rtol=1e-5)


@pytest.mark.parametrize('cls_name', ['CSR', 'CSC'])
def test_matrix_solve_accepts_a_float64_rhs_under_x64(cls_name):
    with jax_x64_enabled():
        mat = getattr(brainevent, cls_name).fromdense(jnp.asarray(_DENSE, dtype=jnp.float32))
        b = jnp.asarray([1., 2., 3.])
        x = mat.solve(b)
        np.testing.assert_allclose(np.asarray(x), np.linalg.solve(_DENSE, np.asarray(b)), rtol=1e-5)


def test_csr_solve_promotes_mixed_dtypes_and_divides_units():
    import brainunit as u

    with jax_x64_enabled():
        csr = brainevent.CSR.fromdense(jnp.asarray(_DENSE, dtype=jnp.float32) * u.ohm)
        b = jnp.asarray([1., 2., 3.]) * u.volt
        x = csr.solve(b)
        assert u.get_unit(x) == u.amp
        assert u.get_mantissa(x).dtype == jnp.float64
        np.testing.assert_allclose(u.get_mantissa(x), np.linalg.solve(_DENSE, [1., 2., 3.]), rtol=1e-5)


# ---------------------------------------------------------------------------
# Non-canonical CSR (#225): repeated (row, col) entries add, and columns need
# not be sorted within a row. jax's ``spsolve`` assumes a canonical CSR (cuSOLVER
# keeps one of the repeats; the SciPy callback sorts read-only buffers in place).
# ---------------------------------------------------------------------------

def _coo_csr(rows, cols, data, n):
    """CSR arrays of COO entries in the given order within each row, and the dense sum."""
    rows, cols, data = np.asarray(rows), np.asarray(cols), np.asarray(data, np.float64)
    order = np.argsort(rows, kind='stable')
    rows, cols, data = rows[order], cols[order], data[order]
    dense = np.zeros((n, n))
    np.add.at(dense, (rows, cols), data)
    indptr = np.searchsorted(rows, np.arange(n + 1))
    return (jnp.asarray(data, jnp.float32), jnp.asarray(cols, jnp.int32),
            jnp.asarray(indptr, jnp.int32), dense)


def _random_repeated(n, nse, seed, sort_cols):
    """A diagonally dominant matrix whose entries repeat; columns sorted or shuffled per row."""
    rng = np.random.default_rng(seed)
    rows = np.concatenate([np.arange(n), rng.integers(0, n, nse - n)])
    cols = np.concatenate([np.arange(n), rng.integers(0, n, nse - n)])
    data = np.concatenate([np.full(n, 2.0 * nse / n), rng.uniform(-1.0, 1.0, nse - n)])
    keys = (cols if sort_cols else rng.random(nse), rows)
    order = np.lexsort(keys)
    return _coo_csr(rows[order], cols[order], data[order], n)


_NON_CANONICAL = {
    # [[4, 1], [0, 3]]: (0, 0) stored twice, nse <= n * n.
    'repeated': ([0, 0, 0, 1], [0, 0, 1, 1], [2.0, 2.0, 1.0, 3.0], 2),
    # the first copy is 0: cuSOLVER would solve with 0 at (0, 0)
    'repeated, zero first': ([0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [0.0, 4.0, 1.0, 1.0, 3.0], 2),
    # nse = 7 > n * n = 4
    'more entries than elements': ([0, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1, 1],
                                   [3.0, 0.5, 1.0, 0.5, 1.0, 1.0, 2.0], 2),
    # no repeats, columns descending within each row
    'unsorted columns': ([0, 0, 1, 1, 2], [1, 0, 2, 1, 2], [1.0, 4.0, 1.0, 3.0, 2.0], 3),
    # repeats that are not adjacent within the row
    'repeated and unsorted': ([0, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 1], [0.5, 4.0, 0.5, 1.0, 1.0, 2.0], 2),
}


@pytest.mark.parametrize('case', list(_NON_CANONICAL))
def test_csr_solve_non_canonical_matches_dense(case):
    rows, cols, data, n = _NON_CANONICAL[case]
    data, indices, indptr, dense = _coo_csr(rows, cols, data, n)
    b = np.arange(1.0, n + 1.0)
    x = brainevent.CSR((data, indices, indptr), shape=(n, n)).solve(jnp.asarray(b, jnp.float32))
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(dense, b), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('reorder', [0, 1, 2, 3])
@pytest.mark.parametrize('sort_cols', [True, False])
def test_csr_solve_random_repeated_entries(reorder, sort_cols):
    n = 30
    data, indices, indptr, dense = _random_repeated(n, nse=8 * n, seed=reorder, sort_cols=sort_cols)
    b = np.random.default_rng(7).standard_normal(n)
    x = csr_solve(data, indices, indptr, jnp.asarray(b, jnp.float32), reorder=reorder)
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(dense, b), rtol=1e-4, atol=1e-5)


def test_csc_solve_repeated_entries():
    # CSC stores the transpose's CSR: (0, 0) twice and column 1 unsorted.
    dense = np.array([[4., 1.], [0., 3.]])
    csc = brainevent.CSC(
        (jnp.asarray([2.0, 2.0, 3.0, 1.0]), jnp.asarray([0, 0, 1, 0], jnp.int32), jnp.asarray([0, 2, 4], jnp.int32)),
        shape=(2, 2),
    )
    np.testing.assert_allclose(np.asarray(csc.todense()), dense)
    x = csc.solve(jnp.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(dense, [1.0, 2.0]), rtol=1e-5)


def test_csr_solve_homogeneous_weight():
    # A single weight shared by every stored entry (repeats add: (1, 1) holds 2 * 2).
    rows, cols = [0, 0, 1, 1], [0, 1, 1, 1]
    _, indices, indptr, _ = _coo_csr(rows, cols, np.ones(4), 2)
    dense = np.array([[2., 2.], [0., 4.]])
    x = brainevent.CSR((jnp.asarray([2.0]), indices, indptr), shape=(2, 2)).solve(jnp.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(dense, [1.0, 2.0]), rtol=1e-5)


def test_csr_solve_canonical_matches_spsolve_bitwise():
    # A canonical matrix reaches ``spsolve`` unchanged.
    from jax.experimental.sparse.linalg import spsolve
    csr = brainevent.CSR.fromdense(jnp.asarray(_DENSE, jnp.float32))
    b = jnp.asarray([1., 2., 3.], jnp.float32)
    got = csr_solve(csr.data, csr.indices, csr.indptr, b)
    want = spsolve(csr.data, csr.indices, csr.indptr, b)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_csr_solve_traced_structure_under_jit():
    rows, cols, data, n = _NON_CANONICAL['repeated and unsorted']
    data, indices, indptr, dense = _coo_csr(rows, cols, data, n)
    b = jnp.asarray([1.0, 2.0])
    x = jax.jit(lambda d, i, p, b: csr_solve(d, i, p, b))(data, indices, indptr, b)
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(dense, [1.0, 2.0]), rtol=1e-5)


@pytest.mark.parametrize('case', ['canonical', 'repeated and unsorted', 'more entries than elements'])
@pytest.mark.parametrize('mode', ['fwd', 'rev'])
def test_csr_solve_gradients_match_dense(case, mode):
    # Derivatives with respect to the stored weights and the right-hand side,
    # against the dense solve of the same matrix (repeated entries summed).
    if case == 'canonical':
        dense0 = np.array([[4., 1., 0.], [0., 3., 2.], [1., 0., 5.]])
        r, c = np.nonzero(dense0)
        rows, cols, data, n = r, c, dense0[r, c], 3
    else:
        rows, cols, data, n = _NON_CANONICAL[case]
    data, indices, indptr, _ = _coo_csr(rows, cols, data, n)
    row_ids = jnp.asarray(np.repeat(np.arange(n), np.diff(np.asarray(indptr))))
    b = jnp.asarray(np.arange(1.0, n + 1.0), jnp.float32)
    rng = np.random.default_rng(3)
    t_data = jnp.asarray(rng.standard_normal(data.shape), jnp.float32)
    t_b = jnp.asarray(rng.standard_normal(n), jnp.float32)

    def sparse(d, b):
        return csr_solve(d, indices, indptr, b)

    def dense(d, b):
        return jnp.linalg.solve(jnp.zeros((n, n), d.dtype).at[row_ids, indices].add(d), b)

    if mode == 'fwd':
        _, got = jax.jvp(sparse, (data, b), (t_data, t_b))
        _, want = jax.jvp(dense, (data, b), (t_data, t_b))
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-5)
    else:
        ct = jnp.asarray(rng.standard_normal(n), jnp.float32)
        got = jax.vjp(sparse, data, b)[1](ct)
        want = jax.vjp(dense, data, b)[1](ct)
        for g, w in zip(got, want):
            np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-4, atol=1e-5)


def test_canonical_csr_layout():
    from brainevent._csr.solve.spsolve import _canonical_csr
    # 2 x 2, 7 entries: (0, 1) x2, (0, 0) x2, (1, 1) x2, (1, 0) x1 -> 4 distinct.
    rows = jnp.asarray([0, 0, 0, 0, 1, 1, 1], jnp.int32)
    cols = jnp.asarray([1, 0, 1, 0, 1, 0, 1], jnp.int32)
    data = jnp.asarray([1., 2., 3., 4., 5., 6., 7.])
    d, i, p = _canonical_csr(data, rows, cols, 2, jnp.int32)
    # static size min(nse, n * n) = 4; here every slot is used
    np.testing.assert_array_equal(np.asarray(d), [6., 4., 6., 12.])
    np.testing.assert_array_equal(np.asarray(i), [0, 1, 0, 1])
    np.testing.assert_array_equal(np.asarray(p), [0, 2, 4])
    # 3 x 3, 4 entries with one repeat: 3 distinct, one zero slot past indptr[-1]
    d, i, p = _canonical_csr(jnp.asarray([1., 2., 3., 4.]), jnp.asarray([2, 0, 2, 1], jnp.int32),
                             jnp.asarray([1, 2, 1, 0], jnp.int32), 3, jnp.int32)
    np.testing.assert_array_equal(np.asarray(d), [2., 4., 4., 0.])
    np.testing.assert_array_equal(np.asarray(i), [2, 0, 1, 0])
    np.testing.assert_array_equal(np.asarray(p), [0, 1, 2, 3])


def test_csr_solve_no_stored_entries_raises():
    # nse = 0 is singular; cuSOLVER would fail inside csrlsvqr, SciPy returns nan.
    empty = (jnp.zeros((0,)), jnp.zeros((0,), jnp.int32), jnp.zeros((3,), jnp.int32))
    with pytest.raises(ValueError, match='stores no entries'):
        csr_solve(*empty, jnp.ones(2))
    with pytest.raises(ValueError, match='stores no entries'):
        brainevent.CSR(empty, shape=(2, 2)).solve(jnp.ones(2))
    with pytest.raises(ValueError, match='stores no entries'):  # a shared weight with no entries
        csr_solve(jnp.ones((1,)), *empty[1:], jnp.ones(2))


def test_csr_solve_empty_system():
    x = csr_solve(jnp.zeros((0,)), jnp.zeros((0,), jnp.int32), jnp.zeros((1,), jnp.int32), jnp.zeros((0,)))
    assert x.shape == (0,)


@pytest.mark.filterwarnings('ignore::scipy.sparse.linalg.MatrixRankWarning')
def test_csr_solve_all_zero_data_follows_spsolve():
    # Stored entries, all zero: singular. As jax's spsolve: nan on CPU, an error on GPU.
    data = jnp.zeros((3,))
    indices = jnp.asarray([0, 1, 1], jnp.int32)
    indptr = jnp.asarray([0, 2, 3], jnp.int32)
    if jax.default_backend() == 'cpu':
        x = csr_solve(data, indices, indptr, jnp.ones(2))
        assert np.isnan(np.asarray(x)).all()
    else:
        with pytest.raises(Exception, match='Singular matrix'):
            jax.block_until_ready(csr_solve(data, indices, indptr, jnp.ones(2)))


def test_canonical_csr_turns_negative_zero_positive():
    # The one change to a canonical input: each slot is 0 + value.
    from brainevent._csr.solve.spsolve import _canonical_csr
    d, i, p = _canonical_csr(jnp.asarray([-0.0, 2.0]), jnp.asarray([0, 1], jnp.int32),
                             jnp.asarray([0, 1], jnp.int32), 2, jnp.int32)
    assert not np.signbit(np.asarray(d)).any()
    np.testing.assert_array_equal(np.asarray(d), [0.0, 2.0])
    np.testing.assert_array_equal(np.asarray(i), [0, 1])
    np.testing.assert_array_equal(np.asarray(p), [0, 1, 2])
