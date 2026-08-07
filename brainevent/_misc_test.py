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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._misc import (
    _INT32_MAX,
    _as_indptr,
    _as_int32_cuda_offsets,
    _as_int32_indices,
    _check_compressed_structure,
    _require_jax_x64_for_int64,
    _resolve_indptr_dtype,
    coo2csr,
    generate_block_dim,
)
from brainevent._test_util import jax_x64_enabled


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


class TestCsrToCscIndexMethods(unittest.TestCase):
    def test_default_matches_explicit_coo(self):
        from brainevent._misc import csr_to_csc_index
        indptr = np.array([0, 2, 3, 5], dtype=np.int32)
        indices = np.array([0, 2, 1, 0, 3], dtype=np.int32)
        default = csr_to_csc_index(indptr, indices, shape=(3, 4))
        explicit = csr_to_csc_index(indptr, indices, shape=(3, 4), method="coo")
        for got, expected in zip(default, explicit):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))

    def test_numpy_matches_coo_reordered_data(self):
        from brainevent._misc import csr_to_csc_index
        indptr = np.array([0, 2, 3, 5], dtype=np.int32)
        indices = np.array([0, 2, 1, 0, 3], dtype=np.int32)
        data = np.array([10., 20., 30., 40., 50.])
        coo_indptr, coo_rows, coo_perm = csr_to_csc_index(indptr, indices, shape=(3, 4), method="coo")
        np_indptr, np_rows, np_perm = csr_to_csc_index(indptr, indices, shape=(3, 4), method="numpy")
        np.testing.assert_array_equal(np.asarray(np_indptr), np.asarray(coo_indptr))
        np.testing.assert_array_equal(np.asarray(np_rows), np.asarray(coo_rows))
        np.testing.assert_array_equal(data[np.asarray(np_perm)], data[np.asarray(coo_perm)])

    def test_include_perm_false(self):
        from brainevent._misc import csr_to_csc_index, csc_to_csr_index
        indptr = np.array([0, 2, 3, 5], dtype=np.int32)
        indices = np.array([0, 2, 1, 0, 3], dtype=np.int32)
        csc_indptr, csc_indices, perm = csr_to_csc_index(
            indptr, indices, shape=(3, 4), method="numpy", include_perm=False
        )
        self.assertIsNone(perm)
        _, _, back_perm = csc_to_csr_index(csc_indptr, csc_indices, shape=(3, 4), include_perm=False)
        self.assertIsNone(back_perm)

    def test_numpy_resolves_indptr_from_nnz_not_input_dtype(self):
        # The output indptr precision follows the nnz (auto policy), not the
        # input indptr dtype: a small-nnz matrix resolves to int32 even when the
        # caller supplies an int64 indptr. ``indices`` are always int32.
        from brainevent._misc import csr_to_csc_index
        indptr = np.array([0, 2, 3, 5], dtype=np.int64)
        indices = np.array([0, 2, 1, 0, 3], dtype=np.int32)
        csc_indptr, csc_indices, perm = csr_to_csc_index(
            indptr, indices, shape=(3, 4), method="numpy"
        )
        self.assertEqual(np.asarray(csc_indptr).dtype, np.int32)
        self.assertEqual(np.asarray(csc_indices).dtype, np.int32)
        self.assertEqual(np.asarray(perm).dtype, np.int32)

    def test_unknown_method_raises(self):
        from brainevent._misc import csr_to_csc_index
        indptr = np.array([0, 1], dtype=np.int32)
        indices = np.array([0], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "Unknown csr_to_csc_index method"):
            csr_to_csc_index(indptr, indices, shape=(1, 1), method="bogus")

    def test_numpy_coerces_indices_to_int32(self):
        # ``indices`` are secondary-axis coordinates and are always emitted int32,
        # even when the caller supplies an int64 coordinate array.
        from brainevent._misc import csr_to_csc_index
        indptr = np.array([0, 2, 3, 5], dtype=np.int32)
        indices = np.array([0, 2, 1, 0, 3], dtype=np.int64)
        csc_indptr, csc_indices, perm = csr_to_csc_index(
            indptr, indices, shape=(3, 4), method="numpy"
        )
        self.assertEqual(np.asarray(csc_indptr).dtype, np.int32)
        self.assertEqual(np.asarray(csc_indices).dtype, np.int32)
        self.assertEqual(np.asarray(perm).dtype, np.int32)

    def test_numpy_jax_output_does_not_toggle_x64_for_small_nnz(self):
        # The silent ``jax_enable_x64`` auto-toggle was removed: with x64 off, a
        # small-nnz conversion resolves to int32 offsets and never mutates the
        # global config.
        from brainevent._misc import csr_to_csc_index
        old_x64 = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", False)
        try:
            indptr = np.array([0, 2, 3, 5], dtype=np.int64)
            indices = jnp.array([0, 2, 1, 0, 3], dtype=jnp.int32)
            csc_indptr, csc_indices, perm = csr_to_csc_index(
                indptr, indices, shape=(3, 4), method="numpy"
            )
            self.assertEqual(csc_indptr.dtype, jnp.int32)
            self.assertEqual(csc_indices.dtype, jnp.int32)
            self.assertEqual(perm.dtype, jnp.int32)
            self.assertFalse(jax.config.jax_enable_x64)
        finally:
            jax.config.update("jax_enable_x64", old_x64)

    def test_offset_index_dtype_promotes_large_nse(self):
        from brainevent._misc import _offset_index_dtype
        self.assertEqual(_offset_index_dtype(np.iinfo(np.int32).max), np.int32)
        self.assertEqual(_offset_index_dtype(np.iinfo(np.int32).max + 1), np.int64)
        self.assertEqual(_offset_index_dtype(3, preferred=np.int64), np.int64)


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


class TestCsrStructureDtypes(unittest.TestCase):
    def test_public_structure_dtype_contract_accepts_int32_indices(self):
        # ``indices`` must be int32; ``indptr`` may be int32 or int64.
        from brainevent._misc import _check_csr_structure_dtypes
        _check_csr_structure_dtypes(
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
        )
        _check_csr_structure_dtypes(
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int64),
        )

    def test_public_structure_dtype_contract_rejects_unsigned_indices(self):
        from brainevent._misc import _check_csr_structure_dtypes
        with self.assertRaisesRegex(AssertionError, "Indices must be int32"):
            _check_csr_structure_dtypes(
                np.array([0, 1], dtype=np.uint32),
                np.array([0, 2], dtype=np.int32),
            )

    def test_public_structure_dtype_contract_rejects_int64_indices(self):
        # ``indices`` are always int32; int64 coordinates are rejected regardless
        # of the ``indptr`` dtype.
        from brainevent._misc import _check_csr_structure_dtypes
        with self.assertRaisesRegex(AssertionError, "Indices must be int32"):
            _check_csr_structure_dtypes(
                np.array([0, 1], dtype=np.int64),
                np.array([0, 2], dtype=np.int32),
            )

    def test_cuda_structure_dtype_contract_accepts_int32_indices_and_int64_indptr(self):
        from brainevent._misc import _check_csr_cuda_structure_dtypes
        _check_csr_cuda_structure_dtypes(
            jax.ShapeDtypeStruct((2,), jnp.int32),
            jax.ShapeDtypeStruct((2,), jnp.int64),
        )

    def test_cuda_structure_dtype_contract_rejects_int64_indices(self):
        from brainevent._misc import _check_csr_cuda_structure_dtypes
        with self.assertRaisesRegex(TypeError, "indices with dtype int32"):
            _check_csr_cuda_structure_dtypes(
                jax.ShapeDtypeStruct((2,), jnp.int64),
                jax.ShapeDtypeStruct((2,), jnp.int64),
            )

    def test_cuda_structure_dtype_contract_rejects_unsigned_indptr(self):
        from brainevent._misc import _check_csr_cuda_structure_dtypes
        with self.assertRaisesRegex(TypeError, "indptr with dtype int32 or int64"):
            _check_csr_cuda_structure_dtypes(
                jax.ShapeDtypeStruct((2,), jnp.int32),
                jax.ShapeDtypeStruct((2,), jnp.uint32),
            )


def test_gpu_column_block_method_rejects_non_positive_block_size():
    from brainevent._misc import csr_to_csc_index
    indptr = np.array([0, 1], dtype=np.int32)
    indices = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="positive integer"):
        csr_to_csc_index(
            indptr, indices, shape=(1, 1), method="gpu_column_block",
            column_block_size=0,
        )


def test_load_csr_to_csc_cuda_module_is_lazy_and_cached(monkeypatch):
    import brainevent._misc as misc
    import brainevent._op as op

    calls = []
    fake_module = object()

    def fake_load_cuda_file(path, *, name):
        calls.append((path, name))
        return fake_module

    monkeypatch.setattr(misc, "_CSR_TO_CSC_CUDA_MODULE", None)
    monkeypatch.setattr(op, "load_cuda_file", fake_load_cuda_file)

    assert misc._load_csr_to_csc_cuda_module() is fake_module
    assert misc._load_csr_to_csc_cuda_module() is fake_module
    assert len(calls) == 1
    assert calls[0][0].name == "csr_to_csc.cu"
    assert calls[0][1] == "csr_to_csc"


def test_gpu_column_block_method_falls_back_to_numpy(monkeypatch):
    import brainevent._misc as misc

    def fail_load():
        raise RuntimeError("simulated CUDA loader failure")

    monkeypatch.setattr(misc, "_load_csr_to_csc_cuda_module", fail_load)
    indptr = np.array([0, 2, 3, 5], dtype=np.int32)
    indices = np.array([0, 2, 1, 0, 3], dtype=np.int32)

    got = misc.csr_to_csc_index(
        indptr, indices, shape=(3, 4), method="gpu_column_block",
        column_block_size=2,
    )
    expected = misc.csr_to_csc_index(
        indptr, indices, shape=(3, 4), method="numpy",
    )

    for got_arr, expected_arr in zip(got, expected):
        np.testing.assert_array_equal(np.asarray(got_arr), np.asarray(expected_arr))


def test_gpu_column_block_method_stitches_column_blocks_correctly(monkeypatch):
    import brainevent._misc as misc

    shape = (64, 64)
    n_rows, n_cols = shape
    per_row = 8
    indptr = np.arange(n_rows + 1, dtype=np.int32) * per_row
    row_ids = np.repeat(np.arange(n_rows, dtype=np.int32), per_row)
    offsets = np.tile(np.array([0, 1, 7, 13, 21, 34, 45, 63], dtype=np.int32), n_rows)
    indices = ((row_ids * 17 + offsets) % n_cols).astype(np.int32)
    column_block_size = 11
    fill_blocks = []

    monkeypatch.setattr(misc, "_load_csr_to_csc_cuda_module", lambda: object())
    monkeypatch.setattr(misc.jax, "devices", lambda kind=None: [object()] if kind == "gpu" else [])
    monkeypatch.setattr(misc.jax, "device_put", lambda x, device=None: x)

    def fake_ffi_call(name, out_info):
        if name == "csr_to_csc.csr_to_csc_count":
            def count(csr_indices, csr_indptr):
                return np.bincount(np.asarray(csr_indices), minlength=shape[1]).astype(out_info.dtype)
            return count

        if name == "csr_to_csc.csr_to_csc_fill_block":
            _, rows_info, perm_info = out_info

            def fill_block(csr_indices, csr_indptr, initial_pos, *, col_start, col_end):
                col_start = int(col_start)
                col_end = int(col_end)
                fill_blocks.append((col_start, col_end, np.asarray(initial_pos).copy()))

                csr_indices_np = np.asarray(csr_indices)
                csr_indptr_np = np.asarray(csr_indptr)
                positions = np.arange(csr_indices_np.size)
                row_ids = np.searchsorted(csr_indptr_np, positions, side="right") - 1
                in_block = (col_start <= csr_indices_np) & (csr_indices_np < col_end)
                block_positions = positions[in_block]
                order_chunks = []
                for col in range(col_start, col_end):
                    col_positions = block_positions[csr_indices_np[block_positions] == col]
                    order_chunks.append(col_positions[::-1])
                order = np.concatenate(order_chunks) if order_chunks else np.zeros(0, dtype=np.int64)
                scratch = np.zeros(out_info[0].shape, dtype=out_info[0].dtype)
                return (
                    scratch,
                    row_ids[order].astype(rows_info.dtype),
                    order.astype(perm_info.dtype),
                )
            return fill_block

        raise AssertionError(f"unexpected FFI call: {name}")

    monkeypatch.setattr(misc.jax.ffi, "ffi_call", fake_ffi_call)

    got = misc.csr_to_csc_index(
        indptr, indices, shape=shape, method="gpu_column_block",
        column_block_size=column_block_size,
    )
    expected = misc.csr_to_csc_index(indptr, indices, shape=shape, method="numpy")

    got_indptr, got_rows, got_perm = [np.asarray(arr) for arr in got]
    expected_indptr, expected_rows, expected_perm = [np.asarray(arr) for arr in expected]

    np.testing.assert_array_equal(got_indptr, expected_indptr)
    for col in range(n_cols):
        got_start, got_end = int(got_indptr[col]), int(got_indptr[col + 1])
        expected_start, expected_end = int(expected_indptr[col]), int(expected_indptr[col + 1])
        got_pairs = sorted(zip(got_rows[got_start:got_end], got_perm[got_start:got_end]))
        expected_pairs = sorted(zip(expected_rows[expected_start:expected_end], expected_perm[expected_start:expected_end]))
        assert got_pairs == expected_pairs, col

    expected_blocks = [
        (start, min(start + column_block_size, n_cols))
        for start in range(0, n_cols, column_block_size)
    ]
    assert [(start, end) for start, end, _ in fill_blocks] == expected_blocks

    got_no_perm = misc.csr_to_csc_index(
        indptr, indices, shape=shape, method="gpu_column_block",
        include_perm=False, column_block_size=column_block_size,
    )
    np.testing.assert_array_equal(np.asarray(got_no_perm[0]), np.asarray(expected[0]))
    got_no_perm_rows = np.asarray(got_no_perm[1])
    for col in range(n_cols):
        start, end = int(got_indptr[col]), int(got_indptr[col + 1])
        np.testing.assert_array_equal(
            np.sort(got_no_perm_rows[start:end]),
            np.sort(expected_rows[start:end]),
        )
    assert got_no_perm[2] is None


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


# ---------------------------------------------------------------------------
# The ``indptr`` int64 auto-precision + ``indices`` always-int32 policy.
#
# * ``indices`` are secondary-axis coordinates and are *always* int32; int64 (or
#   out-of-range) coordinates raise rather than widen.
# * ``indptr``/offset arrays default to int32 and auto-promote to int64 only when
#   ``nnz`` exceeds the int32 range. Creating an int64 offset array requires
#   ``jax_enable_x64``; the library raises instead of toggling the global config.
#
# Materialising a ``nnz > int32_max`` array is impractical, so the promotion
# threshold and the x64 gating are exercised here at the helper level; the
# end-to-end constructor behaviour lives in ``_csr/main_test.py``.
# ---------------------------------------------------------------------------


# -- _resolve_indptr_dtype: auto promotion threshold + explicit requests -----

def test_resolve_auto_picks_int32_within_range():
    assert _resolve_indptr_dtype(0, "auto") == np.dtype(np.int32)
    assert _resolve_indptr_dtype(1_000, "auto") == np.dtype(np.int32)
    assert _resolve_indptr_dtype(_INT32_MAX, "auto") == np.dtype(np.int32)


def test_resolve_auto_picks_int64_above_range():
    assert _resolve_indptr_dtype(_INT32_MAX + 1, "auto") == np.dtype(np.int64)
    assert _resolve_indptr_dtype(10 * _INT32_MAX, "auto") == np.dtype(np.int64)


def test_resolve_explicit_int32_overflows():
    with pytest.raises(OverflowError, match="exceeds the int32 range"):
        _resolve_indptr_dtype(_INT32_MAX + 1, np.int32)


def test_resolve_explicit_int32_within_range_ok():
    assert _resolve_indptr_dtype(_INT32_MAX, np.int32) == np.dtype(np.int32)


def test_resolve_explicit_int64_honoured():
    # dtype resolution does not gate on x64; gating is a separate step.
    assert _resolve_indptr_dtype(10, np.int64) == np.dtype(np.int64)


def test_resolve_rejects_unknown_string():
    with pytest.raises(ValueError, match="indptr_dtype must be"):
        _resolve_indptr_dtype(10, "float")


# -- _require_jax_x64_for_int64: gating without mutating global config ------

def test_require_x64_raises_for_int64_when_disabled():
    assert jax.config.jax_enable_x64 is False
    with pytest.raises(ValueError, match="requires an int64 array"):
        _require_jax_x64_for_int64(np.int64, "test context")
    # The gate must never toggle the global config.
    assert jax.config.jax_enable_x64 is False


def test_require_x64_allows_int32_when_disabled():
    # int32 never needs x64.
    _require_jax_x64_for_int64(np.int32, "test context")


def test_require_x64_allows_int64_when_enabled():
    with jax_x64_enabled():
        _require_jax_x64_for_int64(np.int64, "test context")


# -- _as_int32_indices: indices are always int32 ----------------------------

def test_indices_int32_passthrough():
    idx = jnp.array([0, 1, 2], dtype=jnp.int32)
    out = _as_int32_indices(idx, 3, "ctx")
    assert out.dtype == jnp.int32


def test_indices_int64_coerced_when_in_range():
    idx = np.array([0, 1, 2], dtype=np.int64)
    out = _as_int32_indices(idx, 3, "ctx")
    assert out.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(out), [0, 1, 2])


def test_indices_negative_raises():
    idx = np.array([0, -1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="must be non-negative"):
        _as_int32_indices(idx, 3, "ctx")


def test_indices_out_of_bounds_raises():
    idx = np.array([0, 5, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="out of bounds"):
        _as_int32_indices(idx, 3, "ctx")


def test_indices_non_integer_raises():
    idx = np.array([0.0, 1.0], dtype=np.float32)
    with pytest.raises(TypeError, match="must be an integer array"):
        _as_int32_indices(idx, 3, "ctx")


def test_indices_secondary_dim_beyond_int32_raises():
    idx = np.array([0, 1], dtype=np.int32)
    with pytest.raises(OverflowError, match="int32-representable"):
        _as_int32_indices(idx, _INT32_MAX + 2, "ctx")


def test_indices_traced_int64_rejected():
    def f(idx):
        return _as_int32_indices(idx, 3, "ctx")

    # An int64 tracer only exists when x64 is enabled (otherwise the input is
    # truncated to int32 before tracing).
    with jax_x64_enabled():
        with pytest.raises(TypeError, match="traced int64 array"):
            jax.jit(f)(jnp.array([0, 1, 2], dtype=jnp.int64))


def test_indices_traced_int32_ok_no_host_readback():
    # Under a tracer only the static dtype is checked; no host value readback.
    def f(idx):
        out = _as_int32_indices(idx, 3, "ctx")
        return out.sum()

    val = jax.jit(f)(jnp.array([0, 1, 2], dtype=jnp.int32))
    assert int(val) == 3


# -- _as_indptr: resolves dtype + gates int64 -------------------------------

def test_as_indptr_small_is_int32():
    ptr = np.array([0, 2, 3], dtype=np.int64)
    out = _as_indptr(ptr, 3, "auto", "ctx")
    assert out.dtype == jnp.int32


def test_as_indptr_explicit_int64_gated_when_x64_off():
    ptr = np.array([0, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="requires an int64 array"):
        _as_indptr(ptr, 3, np.int64, "ctx")


def test_as_indptr_explicit_int64_ok_when_x64_on():
    with jax_x64_enabled():
        ptr = np.array([0, 2, 3], dtype=np.int64)
        out = _as_indptr(ptr, 3, np.int64, "ctx")
        assert out.dtype == jnp.int64


# -- _as_int32_cuda_offsets: int32-only CUDA/JITC ABI guard -----------------

def test_cuda_offsets_int32_passthrough():
    off = jnp.array([0, 2, 3], dtype=jnp.int32)
    out = _as_int32_cuda_offsets(off, "ctx")
    assert out.dtype == jnp.int32


def test_cuda_offsets_int64_raises_not_implemented():
    with jax_x64_enabled():
        off = jnp.array([0, 2, 3], dtype=jnp.int64)
        with pytest.raises(NotImplementedError, match="int32 ABI"):
            _as_int32_cuda_offsets(off, "ctx")


# -- _check_compressed_structure: tracer path skips value checks ------------

def test_check_structure_tracer_skips_value_checks():
    # A concrete non-monotonic indptr would raise; under a tracer the value
    # checks are skipped, so no error is raised for the (static) checks.
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)

    def f(bad_ptr):
        _check_compressed_structure(indices, bad_ptr, (2, 3), format="csr")
        return bad_ptr.sum()

    # Non-monotonic values but valid shape/dtype: tracer path must not raise.
    jax.jit(f)(jnp.array([0, 3, 2], dtype=jnp.int32))
