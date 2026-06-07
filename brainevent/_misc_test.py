# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import unittest

import numpy as np

from brainevent._misc import generate_block_dim, coo2csr


class TestCoo2Csr(unittest.TestCase):
    def test_basic_conversion(self):
        row_ids = np.array([0, 2, 1, 0, 2])
        col_ids = np.array([0, 3, 1, 2, 0])
        indptr, indices, order = coo2csr(row_ids, col_ids, shape=(3, 4))
        np.testing.assert_array_equal(indptr, [0, 2, 3, 5])
        np.testing.assert_array_equal(indices, [0, 2, 1, 3, 0])
        np.testing.assert_array_equal(order, [0, 3, 2, 1, 4])

    def test_empty_rows(self):
        # row 1 has no stored entries
        row_ids = np.array([0, 0, 2])
        col_ids = np.array([1, 3, 0])
        indptr, indices, order = coo2csr(row_ids, col_ids, shape=(3, 4))
        np.testing.assert_array_equal(indptr, [0, 2, 2, 3])
        np.testing.assert_array_equal(indices, [1, 3, 0])

    def test_data_reorder_matches_dense(self):
        # The `order` permutation must turn COO data into CSR data such that
        # both reconstruct the same dense matrix (duplicates accumulate).
        row_ids = np.array([0, 2, 1, 0, 2])
        col_ids = np.array([0, 3, 1, 2, 0])
        data = np.array([10., 20., 30., 40., 50.])
        indptr, indices, order = coo2csr(row_ids, col_ids, shape=(3, 4))
        csr_data = data[order]

        dense_csr = np.zeros((3, 4))
        for r in range(3):
            for k in range(int(indptr[r]), int(indptr[r + 1])):
                dense_csr[r, int(indices[k])] += csr_data[k]

        dense_coo = np.zeros((3, 4))
        for r, c, v in zip(row_ids, col_ids, data):
            dense_coo[int(r), int(c)] += v

        np.testing.assert_allclose(dense_csr, dense_coo)


class TestGenerateBlockDim(unittest.TestCase):
    def test_small_connections_returns_32(self):
        self.assertEqual(generate_block_dim(10), 32)
        self.assertEqual(generate_block_dim(32), 32)

    def test_medium_connections_returns_64(self):
        self.assertEqual(generate_block_dim(33), 64)
        self.assertEqual(generate_block_dim(64), 64)

    def test_large_connections_returns_128(self):
        self.assertEqual(generate_block_dim(65), 128)
        self.assertEqual(generate_block_dim(128), 128)

    def test_very_large_connections_returns_256(self):
        self.assertEqual(generate_block_dim(129), 256)
        self.assertEqual(generate_block_dim(256), 256)

    def test_connections_above_maximum_returns_maximum(self):
        self.assertEqual(generate_block_dim(257), 256)
        self.assertEqual(generate_block_dim(1000), 256)

    def test_custom_maximum_constrains_block_size(self):
        self.assertEqual(generate_block_dim(100, maximum=64), 64)
        self.assertEqual(generate_block_dim(200, maximum=128), 128)

    def test_small_maximum_returns_maximum(self):
        self.assertEqual(generate_block_dim(50, maximum=16), 16)

    def test_boundary_conditions(self):
        self.assertEqual(generate_block_dim(0), 32)
        self.assertEqual(generate_block_dim(1), 32)

    def test_negative_connections_returns_32(self):
        self.assertEqual(generate_block_dim(-5), 32)

    def test_maximum_zero_returns_zero(self):
        self.assertEqual(generate_block_dim(100, maximum=0), 0)


def test_csc_to_csr_index_roundtrip():
    import numpy as np
    from brainevent._misc import csr_to_csc_index, csc_to_csr_index
    indptr = np.array([0, 2, 3, 5])
    indices = np.array([0, 2, 1, 0, 3])
    shape = (3, 4)
    csc_indptr, csc_indices, perm = csr_to_csc_index(indptr, indices, shape=shape)
    # CSC of W (shape 3x4); its CSR-structure must reproduce the original CSR arrays.
    back_indptr, back_indices, perm2 = csc_to_csr_index(csc_indptr, csc_indices, shape=shape)
    np.testing.assert_array_equal(np.asarray(back_indptr), indptr)
    np.testing.assert_array_equal(np.asarray(back_indices), indices)
    # perm composition returns to identity over the canonical CSR order.
    np.testing.assert_array_equal(np.asarray(perm)[np.asarray(perm2)], np.arange(len(perm)))


class TestCsrToCooIndex(unittest.TestCase):
    def test_expands_indptr_into_row_ids(self):
        from brainevent._misc import csr_to_coo_index
        indptr = np.array([0, 2, 3, 5])
        indices = np.array([0, 2, 1, 0, 3])
        row_ids, col_ids = csr_to_coo_index(indptr, indices)
        # row i repeats (indptr[i+1]-indptr[i]) times; columns pass through.
        np.testing.assert_array_equal(row_ids, [0, 0, 1, 2, 2])
        np.testing.assert_array_equal(col_ids, indices)

    def test_empty_row_produces_no_entries(self):
        from brainevent._misc import csr_to_coo_index
        # Row 1 is empty -> never appears in the expanded row ids.
        indptr = np.array([0, 2, 2, 3])
        indices = np.array([1, 3, 0])
        row_ids, col_ids = csr_to_coo_index(indptr, indices)
        np.testing.assert_array_equal(row_ids, [0, 0, 2])
        np.testing.assert_array_equal(col_ids, [1, 3, 0])

    def test_roundtrips_back_to_csr_via_coo2csr(self):
        from brainevent._misc import csr_to_coo_index, coo2csr
        indptr = np.array([0, 2, 3, 5])
        indices = np.array([0, 2, 1, 0, 3])
        row_ids, col_ids = csr_to_coo_index(indptr, indices)
        new_indptr, new_indices, _ = coo2csr(row_ids, col_ids, shape=(3, 4))
        np.testing.assert_array_equal(np.asarray(new_indptr), indptr)
        np.testing.assert_array_equal(np.asarray(new_indices), indices)


class TestCooToCscIndex(unittest.TestCase):
    def test_matches_dense_column_structure(self):
        from brainevent._misc import coo_to_csc_index
        row_ids = np.array([0, 0, 1, 2, 2])
        col_ids = np.array([0, 2, 1, 0, 3])
        data = np.array([10., 20., 30., 40., 50.])
        shape = (3, 4)
        csc_indptr, csc_rows, perm = coo_to_csc_index(row_ids, col_ids, shape=shape)

        # Column pointer has n_cols + 1 entries and brackets the nnz.
        self.assertEqual(np.asarray(csc_indptr).shape, (shape[1] + 1,))
        self.assertEqual(int(np.asarray(csc_indptr)[0]), 0)
        self.assertEqual(int(np.asarray(csc_indptr)[-1]), col_ids.size)

        # Reconstruct the dense matrix column-by-column from the CSC structure
        # plus the permuted data, and compare against the COO ground truth.
        csc_indptr = np.asarray(csc_indptr)
        csc_rows = np.asarray(csc_rows)
        csc_data = data[np.asarray(perm)]
        dense_csc = np.zeros(shape)
        for c in range(shape[1]):
            for k in range(int(csc_indptr[c]), int(csc_indptr[c + 1])):
                dense_csc[int(csc_rows[k]), c] += csc_data[k]

        dense_coo = np.zeros(shape)
        for r, c, v in zip(row_ids, col_ids, data):
            dense_coo[int(r), int(c)] += v

        np.testing.assert_allclose(dense_csc, dense_coo)

    def test_empty_column_yields_zero_width_pointer_gap(self):
        from brainevent._misc import coo_to_csc_index
        # No entry in column 2 -> indptr is flat across that column.
        row_ids = np.array([0, 1, 2])
        col_ids = np.array([0, 1, 3])
        csc_indptr, _, _ = coo_to_csc_index(row_ids, col_ids, shape=(3, 4))
        csc_indptr = np.asarray(csc_indptr)
        # column 2 spans [csc_indptr[2], csc_indptr[3]) and must be empty.
        self.assertEqual(int(csc_indptr[2]), int(csc_indptr[3]))


class TestIndexDtypeContract(unittest.TestCase):
    """The public index helpers emit ``int32`` index arrays.

    ``int32`` is brainevent's canonical index dtype (see ``CSR`` /
    ``index_dtype=jnp.int32``). These assertions lock that contract so the
    removal of the historical ``brainstate.environ.ditype()`` cast (which also
    resolved to ``int32``) stays behaviour-preserving.
    """

    def test_coo2csr_emits_int32_numpy(self):
        indptr, indices, _ = coo2csr(np.array([0, 2, 1, 0, 2]),
                                     np.array([0, 3, 1, 2, 0]), shape=(3, 4))
        self.assertEqual(np.asarray(indptr).dtype, np.int32)
        self.assertEqual(np.asarray(indices).dtype, np.int32)

    def test_coo2csr_emits_int32_even_for_int64_inputs(self):
        # NumPy's default integer dtype is int64 on Linux/macOS; the output must
        # still be the canonical int32, independent of the input index dtype.
        row_ids = np.array([0, 2, 1, 0, 2], dtype=np.int64)
        col_ids = np.array([0, 3, 1, 2, 0], dtype=np.int64)
        indptr, indices, _ = coo2csr(row_ids, col_ids, shape=(3, 4))
        self.assertEqual(np.asarray(indptr).dtype, np.int32)
        self.assertEqual(np.asarray(indices).dtype, np.int32)

    def test_coo2csr_emits_int32_jax(self):
        import jax.numpy as jnp
        indptr, indices, _ = coo2csr(jnp.array([0, 2, 1, 0, 2]),
                                     jnp.array([0, 3, 1, 2, 0]), shape=(3, 4))
        self.assertEqual(jnp.asarray(indices).dtype, jnp.int32)
        self.assertEqual(jnp.asarray(indptr).dtype, jnp.int32)

    def test_coo_to_csc_index_emits_int32(self):
        from brainevent._misc import coo_to_csc_index
        csc_indptr, csc_rows, _ = coo_to_csc_index(np.array([0, 0, 1, 2, 2]),
                                                   np.array([0, 2, 1, 0, 3]), shape=(3, 4))
        self.assertEqual(np.asarray(csc_indptr).dtype, np.int32)
        self.assertEqual(np.asarray(csc_rows).dtype, np.int32)


class TestNoBrainstateRuntimeDependency(unittest.TestCase):
    """``import brainevent`` must not require the optional ``brainstate`` package.

    ``brainstate`` is *not* a declared dependency of brainevent, so a clean
    ``pip install brainevent`` does not provide it. This regression test pins
    that the import graph reachable from ``import brainevent`` -- including the
    index helpers in :mod:`brainevent._misc` -- stays free of a hard
    ``brainstate`` import. Run in a subprocess with ``brainstate`` blocked so the
    parent process's already-imported modules cannot mask the dependency.
    """

    def test_import_brainevent_and_index_helpers_without_brainstate(self):
        import os
        import sys
        import subprocess
        import textwrap
        import brainevent._misc as _misc

        pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(_misc.__file__)))
        code = textwrap.dedent(
            """
            import sys, builtins
            sys.path.insert(0, %r)
            _real_import = builtins.__import__
            def _blocked(name, *args, **kwargs):
                if name == 'brainstate' or name.startswith('brainstate.'):
                    raise ImportError('brainstate blocked (simulating a clean install)')
                return _real_import(name, *args, **kwargs)
            builtins.__import__ = _blocked

            import numpy as np
            import brainevent  # must not pull in brainstate
            from brainevent._misc import coo2csr, coo_to_csc_index

            indptr, indices, _ = coo2csr(
                np.array([0, 2, 1, 0, 2]), np.array([0, 3, 1, 2, 0]), shape=(3, 4))
            assert np.asarray(indices).dtype == np.int32, np.asarray(indices).dtype
            assert list(np.asarray(indptr)) == [0, 2, 3, 5], list(np.asarray(indptr))

            csc_indptr, _, _ = coo_to_csc_index(
                np.array([0, 0, 1, 2, 2]), np.array([0, 2, 1, 0, 3]), shape=(3, 4))
            assert np.asarray(csc_indptr).dtype == np.int32, np.asarray(csc_indptr).dtype

            assert 'brainstate' not in sys.modules, 'brainstate was imported by brainevent'
            print('OK')
            """ % pkg_parent
        )
        proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(
            proc.returncode, 0,
            msg=f"subprocess failed.\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}",
        )
        self.assertIn('OK', proc.stdout)
