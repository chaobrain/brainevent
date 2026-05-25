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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from importlib.util import find_spec

from brainevent._event.compact import (
    binary_1d_array_index_p,
    binary_1d_array_index_p_call,
    binary_2d_array_index_p,
    binary_2d_array_index_p_call,
    binary_2d_compact_only_p_call,
    binary_2d_compact_only_p,
    binary_2d_csc_encode_p_call,
    binary_2d_csc_encode_p,
    binary_2d_csc_from_array,
    binary_2d_csr_encode_p_call,
    binary_2d_csr_fill_p,
    binary_2d_csr_fill_p_call,
    binary_2d_csr_row_count_p,
    binary_2d_csr_row_count_p_call,
    binary_2d_pair_stream_encode_p_call,
    binary_2d_pair_stream_encode_p,
    binary_2d_row_sparse_encode_p_call,
    binary_2d_row_sparse_encode_p,
)


platform = jax.default_backend()


def _binary_mask(spikes):
    spikes = np.asarray(spikes)
    if spikes.dtype == np.bool_:
        return spikes
    return spikes != 0


def _reconstruct_pair_stream_mask(pair_stream, n_pairs, shape):
    dense = np.zeros(shape, dtype=np.bool_)
    valid_n_pairs = int(np.asarray(n_pairs).reshape(-1)[0])
    pairs = np.asarray(pair_stream[:valid_n_pairs], dtype=np.int32)
    for row, col in pairs:
        dense[row, col] = True
    return dense


def _reconstruct_row_sparse_mask(spike_indices, shape):
    n_src, n_batch = shape
    dense = np.zeros(shape, dtype=np.bool_)
    spike_indices = np.asarray(spike_indices, dtype=np.int32)
    for row in range(n_src):
        for value in spike_indices[row]:
            if value == 0:
                break
            dense[row, value - 1] = True
    return dense


def _reconstruct_csr_mask(indices, indptr, shape):
    dense = np.zeros(shape, dtype=np.bool_)
    valid_nnz = int(np.asarray(indptr)[-1])
    indices = np.asarray(indices[:valid_nnz], dtype=np.int32)
    indptr = np.asarray(indptr, dtype=np.int32)
    for row in range(shape[0]):
        for col in indices[indptr[row]:indptr[row + 1]]:
            dense[row, col] = True
    return dense


def _reconstruct_csc_mask(indices, indptr, shape):
    dense = np.zeros(shape, dtype=np.bool_)
    valid_nnz = int(np.asarray(indptr)[-1])
    indices = np.asarray(indices[:valid_nnz], dtype=np.int32)
    indptr = np.asarray(indptr, dtype=np.int32)
    for col in range(shape[1]):
        for row in indices[indptr[col]:indptr[col + 1]]:
            dense[row, col] = True
    return dense


def _available_backend_candidates(primitive):
    return primitive.available_backends(platform)


def _cpu_backends_or_skip(primitive):
    backends = primitive.available_backends('cpu')
    if not backends:
        pytest.skip(f'No CPU backends for {primitive.name}')
    return backends


def _gpu_backends_or_skip(primitive):
    if platform != 'gpu':
        pytest.skip('GPU-specific backend test.')
    backends = primitive.available_backends(platform)
    if not backends:
        pytest.skip(f'No GPU backends for {primitive.name}')
    return backends


def _run_on_cpu(fn, *args, **kwargs):
    cpu = jax.devices('cpu')[0]
    args = [jax.device_put(arg, cpu) for arg in args]
    return fn(*args, **kwargs)


def _numba_available():
    return find_spec('numba') is not None


class TestBackendAvailability:
    @pytest.mark.parametrize(
        ('primitive', 'expected'),
        [
            (binary_2d_compact_only_p, {'jax_raw'}),
            (binary_1d_array_index_p, {'jax_raw'}),
            (binary_2d_array_index_p, {'jax_raw'}),
            (binary_2d_pair_stream_encode_p, {'jax_raw'}),
            (binary_2d_row_sparse_encode_p, {'jax_raw'}),
            (binary_2d_csr_row_count_p, {'jax_raw'}),
            (binary_2d_csr_fill_p, {'jax_raw'}),
            (binary_2d_csc_encode_p, {'jax_raw'}),
        ],
    )
    def test_gpu_backends_include_jax_raw(self, primitive, expected):
        if platform != 'gpu':
            pytest.skip('GPU-specific backend availability test.')
        assert expected.issubset(set(_available_backend_candidates(primitive)))


class TestBinary1DArrayIndex:
    @pytest.mark.parametrize(
        'spikes',
        [
            jnp.asarray([False, True, False, True, True], dtype=jnp.bool_),
            jnp.asarray([0.0, 2.0, 0.0, 3.0, 1.5], dtype=jnp.float32),
        ],
    )
    def test_matches_active_ids(self, spikes):
        active_ids, n_active = binary_1d_array_index_p_call(spikes, backend='jax_raw')
        expected = np.where(_binary_mask(spikes))[0].astype(np.int32)
        valid = np.sort(np.asarray(active_ids[: int(n_active[0])], dtype=np.int32))
        np.testing.assert_array_equal(valid, expected)

    def test_backend_consistency(self):
        spikes = jnp.asarray([False, True, False, True, True], dtype=jnp.bool_)
        expected = binary_1d_array_index_p_call(spikes, backend='jax_raw')
        cpu_backends = _cpu_backends_or_skip(binary_1d_array_index_p)
        if 'numba' in cpu_backends and _numba_available():
            got = _run_on_cpu(binary_1d_array_index_p_call, spikes, backend='numba')
            np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(expected[1]))
            np.testing.assert_array_equal(
                np.asarray(got[0][: int(got[1][0])]),
                np.asarray(expected[0][: int(expected[1][0])]),
            )

        gpu_backends = _gpu_backends_or_skip(binary_1d_array_index_p)
        if 'cuda_raw' in gpu_backends:
            got = binary_1d_array_index_p_call(spikes, backend='cuda_raw')
            np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(expected[1]))
            np.testing.assert_array_equal(
                np.asarray(got[0][: int(got[1][0])]),
                np.asarray(expected[0][: int(expected[1][0])]),
            )


class TestBinary2DArrayIndex:
    def test_packed_bits_and_active_rows(self):
        spikes = jnp.asarray(
            [
                [True, False, True, False],
                [False, False, False, False],
                [False, True, True, False],
            ],
            dtype=jnp.bool_,
        )
        packed, active_ids, n_active = binary_2d_array_index_p_call(spikes, backend='jax_raw')

        np.testing.assert_array_equal(np.asarray(packed[:, 0]), np.array([5, 0, 6], dtype=np.uint32))
        np.testing.assert_array_equal(
            np.sort(np.asarray(active_ids[: int(n_active[0])], dtype=np.int32)),
            np.array([0, 2], dtype=np.int32),
        )

    def test_backend_consistency(self):
        spikes = jnp.asarray(
            [[True, False, True], [False, True, False]],
            dtype=jnp.bool_,
        )
        expected = binary_2d_array_index_p_call(spikes, backend='jax_raw')
        cpu_backends = _cpu_backends_or_skip(binary_2d_array_index_p)
        if 'numba' in cpu_backends and _numba_available():
            got = _run_on_cpu(binary_2d_array_index_p_call, spikes, backend='numba')
            np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(expected[0]))
            np.testing.assert_array_equal(np.asarray(got[2]), np.asarray(expected[2]))
            np.testing.assert_array_equal(
                np.sort(np.asarray(got[1][: int(got[2][0])], dtype=np.int32)),
                np.sort(np.asarray(expected[1][: int(expected[2][0])], dtype=np.int32)),
            )

        gpu_backends = _gpu_backends_or_skip(binary_2d_array_index_p)
        if 'cuda_raw' in gpu_backends:
            got = binary_2d_array_index_p_call(spikes, backend='cuda_raw')
            np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(expected[0]))
            np.testing.assert_array_equal(np.asarray(got[2]), np.asarray(expected[2]))
            np.testing.assert_array_equal(
                np.sort(np.asarray(got[1][: int(got[2][0])], dtype=np.int32)),
                np.sort(np.asarray(expected[1][: int(expected[2][0])], dtype=np.int32)),
            )


class TestBinary2DCompactOnly:
    @pytest.mark.parametrize(
        'spikes',
        [
            jnp.asarray(
                [
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, True, True],
                ],
                dtype=jnp.bool_,
            ),
            jnp.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [1.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=jnp.float32,
            ),
        ],
    )
    def test_matches_active_rows(self, spikes):
        active_ids, n_active = binary_2d_compact_only_p_call(spikes)

        mask = _binary_mask(spikes)
        expected_rows = np.where(np.any(mask, axis=1))[0].astype(np.int32)
        valid = np.sort(np.asarray(active_ids[: int(n_active[0])], dtype=np.int32))

        assert int(n_active[0]) == expected_rows.size
        np.testing.assert_array_equal(valid, expected_rows)

    def test_jit(self):
        spikes = jnp.asarray(
            [[False, True], [False, False], [True, True]],
            dtype=jnp.bool_,
        )
        active_ids, n_active = jax.jit(binary_2d_compact_only_p_call)(spikes)
        np.testing.assert_array_equal(
            np.sort(np.asarray(active_ids[: int(n_active[0])], dtype=np.int32)),
            np.array([0, 2], dtype=np.int32),
        )

    def test_backend_consistency(self):
        spikes = jnp.asarray(
            [[False, True], [False, False], [True, True]],
            dtype=jnp.bool_,
        )
        expected = binary_2d_compact_only_p_call(spikes, backend='jax_raw')
        cpu_backends = _cpu_backends_or_skip(binary_2d_compact_only_p)
        if 'numba' in cpu_backends and _numba_available():
            got = _run_on_cpu(binary_2d_compact_only_p_call, spikes, backend='numba')
            np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(expected[1]))
            np.testing.assert_array_equal(
                np.sort(np.asarray(got[0][: int(got[1][0])], dtype=np.int32)),
                np.sort(np.asarray(expected[0][: int(expected[1][0])], dtype=np.int32)),
            )

        gpu_backends = _gpu_backends_or_skip(binary_2d_compact_only_p)
        if 'cuda_raw' in gpu_backends:
            got = binary_2d_compact_only_p_call(spikes, backend='cuda_raw')
            np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(expected[1]))
            np.testing.assert_array_equal(
                np.sort(np.asarray(got[0][: int(got[1][0])], dtype=np.int32)),
                np.sort(np.asarray(expected[0][: int(expected[1][0])], dtype=np.int32)),
            )


class TestBinary2DPairStreamEncode:
    @pytest.mark.parametrize(
        'spikes',
        [
            jnp.asarray(
                [[True, False, True], [False, True, False]],
                dtype=jnp.bool_,
            ),
            jnp.asarray(
                [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]],
                dtype=jnp.float32,
            ),
        ],
    )
    def test_roundtrip_mask(self, spikes):
        pair_stream, n_pairs = binary_2d_pair_stream_encode_p_call(spikes)
        reconstructed = _reconstruct_pair_stream_mask(pair_stream, n_pairs, spikes.shape)
        np.testing.assert_array_equal(reconstructed, _binary_mask(spikes))

    def test_jit(self):
        spikes = jnp.asarray([[False, True], [True, False]], dtype=jnp.bool_)
        pair_stream, n_pairs = jax.jit(binary_2d_pair_stream_encode_p_call)(spikes)
        assert int(n_pairs[0]) == 2
        np.testing.assert_array_equal(
            _reconstruct_pair_stream_mask(pair_stream, n_pairs, spikes.shape),
            np.asarray(spikes),
        )

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="`spikes` must be 2D"):
            binary_2d_pair_stream_encode_p_call(jnp.zeros((2, 3, 4), dtype=jnp.bool_))

    def test_backend_consistency(self):
        spikes = jnp.asarray([[True, False, True], [False, True, False]], dtype=jnp.bool_)
        ref_mask = _binary_mask(spikes)

        pair_stream, n_pairs = binary_2d_pair_stream_encode_p_call(spikes, backend='jax_raw')
        np.testing.assert_array_equal(
            _reconstruct_pair_stream_mask(pair_stream, n_pairs, spikes.shape),
            ref_mask,
        )

        cpu_backends = _cpu_backends_or_skip(binary_2d_pair_stream_encode_p)
        if 'numba' in cpu_backends and _numba_available():
            got_stream, got_n = _run_on_cpu(binary_2d_pair_stream_encode_p_call, spikes, backend='numba')
            np.testing.assert_array_equal(
                _reconstruct_pair_stream_mask(got_stream, got_n, spikes.shape),
                ref_mask,
            )

        gpu_backends = _gpu_backends_or_skip(binary_2d_pair_stream_encode_p)
        if 'cuda_raw' in gpu_backends:
            got_stream, got_n = binary_2d_pair_stream_encode_p_call(spikes, backend='cuda_raw')
            np.testing.assert_array_equal(
                _reconstruct_pair_stream_mask(got_stream, got_n, spikes.shape),
                ref_mask,
            )


class TestBinary2DRowSparseEncode:
    def test_roundtrip_and_padding(self):
        spikes = jnp.asarray(
            [
                [True, False, True, False],
                [False, False, False, False],
                [False, True, True, False],
            ],
            dtype=jnp.bool_,
        )
        spike_indices, = binary_2d_row_sparse_encode_p_call(spikes, row_size=2)

        np.testing.assert_array_equal(
            np.asarray(spike_indices),
            np.array([[1, 3], [0, 0], [2, 3]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            _reconstruct_row_sparse_mask(spike_indices, spikes.shape),
            np.asarray(spikes),
        )

    def test_float_threshold(self):
        spikes = jnp.asarray(
            [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]],
            dtype=jnp.float32,
        )
        spike_indices, = binary_2d_row_sparse_encode_p_call(spikes, row_size=2)
        np.testing.assert_array_equal(
            _reconstruct_row_sparse_mask(spike_indices, spikes.shape),
            np.asarray(spikes != 0.0),
        )

    def test_invalid_row_size(self):
        spikes = jnp.zeros((2, 3), dtype=jnp.bool_)
        with pytest.raises(ValueError, match="must be positive"):
            binary_2d_row_sparse_encode_p_call(spikes, row_size=0)
        with pytest.raises(ValueError, match="must be <= n_batch"):
            binary_2d_row_sparse_encode_p_call(spikes, row_size=4)

    def test_capacity_overflow(self):
        spikes = jnp.asarray([[True, True, False]], dtype=jnp.bool_)
        with pytest.raises(ValueError, match="too small"):
            binary_2d_row_sparse_encode_p_call(spikes, row_size=1)

    def test_backend_consistency(self):
        spikes = jnp.asarray(
            [[True, False, True, False], [False, True, False, False]],
            dtype=jnp.bool_,
        )
        ref_mask = _binary_mask(spikes)
        expected, = binary_2d_row_sparse_encode_p_call(spikes, row_size=2, backend='jax_raw')
        np.testing.assert_array_equal(_reconstruct_row_sparse_mask(expected, spikes.shape), ref_mask)

        cpu_backends = _cpu_backends_or_skip(binary_2d_row_sparse_encode_p)
        if 'numba' in cpu_backends and _numba_available():
            got, = _run_on_cpu(binary_2d_row_sparse_encode_p_call, spikes, row_size=2, backend='numba')
            np.testing.assert_array_equal(_reconstruct_row_sparse_mask(got, spikes.shape), ref_mask)

        gpu_backends = _gpu_backends_or_skip(binary_2d_row_sparse_encode_p)
        if 'cuda_raw' in gpu_backends:
            got, = binary_2d_row_sparse_encode_p_call(spikes, row_size=2, backend='cuda_raw')
            np.testing.assert_array_equal(_reconstruct_row_sparse_mask(got, spikes.shape), ref_mask)


class TestBinary2DCSREncode:
    @pytest.mark.parametrize(
        'spikes',
        [
            jnp.asarray(
                [[True, False, True], [False, True, False], [False, False, False]],
                dtype=jnp.bool_,
            ),
            jnp.asarray(
                [[0.0, 1.0, 0.0], [2.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
                dtype=jnp.float32,
            ),
        ],
    )
    def test_roundtrip_mask(self, spikes):
        indices, indptr = binary_2d_csr_encode_p_call(spikes)
        reconstructed = _reconstruct_csr_mask(indices, indptr, spikes.shape)
        np.testing.assert_array_equal(reconstructed, _binary_mask(spikes))

    def test_row_count_plus_fill_matches_encode(self):
        spikes = jnp.asarray(
            [[True, False, True], [False, True, False], [True, True, False]],
            dtype=jnp.bool_,
        )
        indices, indptr = binary_2d_csr_encode_p_call(spikes)
        reconstructed = _reconstruct_csr_mask(indices, indptr, spikes.shape)
        np.testing.assert_array_equal(reconstructed, np.asarray(spikes))
        np.testing.assert_array_equal(
            np.asarray(indptr),
            np.array([0, 2, 3, 5], dtype=np.int32),
        )

    def test_jit(self):
        spikes = jnp.asarray([[False, True], [True, False]], dtype=jnp.bool_)
        indices, indptr = jax.jit(binary_2d_csr_encode_p_call)(spikes)
        np.testing.assert_array_equal(
            _reconstruct_csr_mask(indices, indptr, spikes.shape),
            np.asarray(spikes),
        )

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="`spikes` must be 2D"):
            binary_2d_csr_encode_p_call(jnp.zeros((2, 3, 4), dtype=jnp.bool_))

    def test_backend_consistency(self):
        spikes = jnp.asarray(
            [[True, False, True], [False, True, False], [False, False, False]],
            dtype=jnp.bool_,
        )
        ref_mask = _binary_mask(spikes)
        indices, indptr = binary_2d_csr_encode_p_call(spikes, backend='jax_raw')
        np.testing.assert_array_equal(_reconstruct_csr_mask(indices, indptr, spikes.shape), ref_mask)

        cpu_backends = _cpu_backends_or_skip(binary_2d_csr_row_count_p)
        if 'numba' in cpu_backends and _numba_available():
            got_indices, got_indptr = _run_on_cpu(binary_2d_csr_encode_p_call, spikes, backend='numba')
            np.testing.assert_array_equal(_reconstruct_csr_mask(got_indices, got_indptr, spikes.shape), ref_mask)

        gpu_backends = _gpu_backends_or_skip(binary_2d_csr_row_count_p)
        if 'cuda_raw' in gpu_backends:
            got_indices, got_indptr = binary_2d_csr_encode_p_call(spikes, backend='cuda_raw')
            np.testing.assert_array_equal(_reconstruct_csr_mask(got_indices, got_indptr, spikes.shape), ref_mask)


class TestBinary2DCSCEncode:
    def test_basic_bool(self):
        spikes = jnp.asarray(
            [
                [True, False, True],
                [False, True, False],
                [True, True, False],
            ],
            dtype=jnp.bool_,
        )

        indices, indptr = binary_2d_csc_encode_p_call(spikes, backend='jax_raw')

        assert indices.shape == (9,)
        assert indices.dtype == jnp.int32
        assert indptr.shape == (4,)
        assert indptr.dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(indptr), np.array([0, 2, 4, 5], dtype=np.int32))
        np.testing.assert_array_equal(
            np.asarray(indices[:5]),
            np.array([0, 2, 1, 2, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            _reconstruct_csc_mask(indices, indptr, spikes.shape),
            np.asarray(spikes),
        )

    def test_float_input(self):
        spikes = jnp.asarray(
            [
                [0.0, 1.5, 0.0],
                [2.0, 0.0, 3.0],
            ],
            dtype=jnp.float32,
        )

        indices, indptr = binary_2d_csc_encode_p_call(spikes, backend='jax_raw')

        np.testing.assert_array_equal(np.asarray(indptr), np.array([0, 1, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(
            np.asarray(indices[:3]),
            np.array([1, 0, 1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            _reconstruct_csc_mask(indices, indptr, spikes.shape),
            np.asarray(spikes != 0.0),
        )

    def test_jit(self):
        spikes = jnp.asarray(
            [
                [False, True, False, True],
                [True, False, False, False],
            ],
            dtype=jnp.bool_,
        )

        indices, indptr = jax.jit(binary_2d_csc_from_array)(spikes)

        np.testing.assert_array_equal(np.asarray(indptr), np.array([0, 1, 2, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(
            _reconstruct_csc_mask(indices, indptr, spikes.shape),
            np.asarray(spikes),
        )

    def test_function_wrapper_matches_primitive(self):
        spikes = jnp.asarray([[False, True], [True, False]], dtype=jnp.bool_)
        indices_a, indptr_a = binary_2d_csc_from_array(spikes, backend='jax_raw')
        indices_b, indptr_b = binary_2d_csc_encode_p_call(spikes, backend='jax_raw')
        np.testing.assert_array_equal(np.asarray(indices_a), np.asarray(indices_b))
        np.testing.assert_array_equal(np.asarray(indptr_a), np.asarray(indptr_b))

    def test_invalid_ndim(self):
        with pytest.raises(ValueError, match="`spikes` must be 2D"):
            binary_2d_csc_from_array(jnp.zeros((2, 3, 4), dtype=jnp.bool_))
