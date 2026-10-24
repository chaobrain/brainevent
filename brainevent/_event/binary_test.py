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

from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent import BinaryArray
from brainevent._test_util import jax_x64_enabled


class TestBinaryArray:
    def test_initialization(self):
        """Test basic initialization of BinaryArray."""
        # Create from numpy array
        data = np.array([0, 1, 0, 1], dtype=np.uint8)
        binary_array = BinaryArray(data)
        assert binary_array.shape == (4,)
        assert np.array_equal(binary_array.value, data)

        # Test if BinaryArray is an alias for BinaryArray
        event_array = BinaryArray(data)
        assert type(event_array) == type(binary_array)

        # Test with 2D array
        data_2d = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        binary_array_2d = BinaryArray(data_2d)
        assert binary_array_2d.shape == (2, 2)
        assert np.array_equal(binary_array_2d.value, data_2d)

    def test_matmul_1d_binary_with_2d_dense(self):
        """Test matrix multiplication between 1D binary array and 2D dense array."""
        # 1D binary array (1x3) @ 2D dense array (3x2)
        binary_vec = BinaryArray(np.array([0, 1, 1], dtype=np.uint8))
        dense_mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = binary_vec @ dense_mat
        expected = np.array([8.0, 10.0])  # [0*1 + 1*3 + 1*5, 0*2 + 1*4 + 1*6]

        assert np.allclose(result, expected)
        jax.block_until_ready((result,))

    def test_matmul_2d_binary_with_2d_dense(self):
        """Test matrix multiplication between 2D binary array and 2D dense array."""
        # 2D binary array (2x3) @ 2D dense array (3x2)
        binary_mat = BinaryArray(np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8))
        dense_mat = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = binary_mat @ dense_mat
        expected = np.array(
            [[8.0, 10.0], [6.0, 8.0]])  # [[0*1 + 1*3 + 1*5, 0*2 + 1*4 + 1*6], [1*1 + 0*3 + 1*5, 1*2 + 0*4 + 1*6]]

        assert np.allclose(result, expected)
        jax.block_until_ready((result,))

    def test_rmatmul_2d_dense_with_1d_binary(self):
        """Test reverse matrix multiplication between 2D dense array and 1D binary array."""
        # 2D dense array (2x3) @ 1D binary array (3x1)
        dense_mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        binary_vec = BinaryArray(np.array([0, 1, 1], dtype=np.uint8))

        result = dense_mat @ binary_vec
        expected = np.array([5.0, 11.0])  # [1*0 + 2*1 + 3*1, 4*0 + 5*1 + 6*1]

        assert np.allclose(result, expected)
        jax.block_until_ready((result,))

    def test_rmatmul_2d_dense_with_2d_binary(self):
        """Test reverse matrix multiplication between 2D dense array and 2D binary array."""
        # 2D dense array (2x3) @ 2D binary array (3x2)
        dense_mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        binary_mat = BinaryArray(np.array([[0, 1], [1, 0], [1, 1]], dtype=np.uint8))

        result = dense_mat @ binary_mat
        expected = np.array(
            [[5.0, 4.0], [11.0, 10.0]])  # [[1*0 + 2*1 + 3*1, 1*1 + 2*0 + 3*1], [4*0 + 5*1 + 6*1, 4*1 + 5*0 + 6*1]]

        assert np.allclose(result, expected)
        jax.block_until_ready((result,))

    def test_error_conditions(self):
        """Test error conditions for matrix multiplication."""
        # Test with incompatible dimensions
        binary_mat = BinaryArray(np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8))
        dense_mat_wrong_dim = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2 matrix

        with pytest.raises(AssertionError):
            _ = binary_mat @ dense_mat_wrong_dim

        # Test with scalar array (0D)
        scalar_array = BinaryArray(np.array(1, dtype=np.uint8))
        dense_mat = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(brainevent.MathError):
            _ = scalar_array @ dense_mat

        # Test with 3D array
        array_3d = np.zeros((2, 2, 2), dtype=np.uint8)
        binary_array_3d = BinaryArray(array_3d)

        with pytest.raises(brainevent.MathError):
            _ = binary_array_3d @ dense_mat

        # Test right operand with wrong dimension
        dense_vec = np.array([1.0, 2.0, 3.0])  # 1D array

        with pytest.raises(AssertionError):
            _ = binary_mat @ dense_vec


# ---------------------------------------------------------------------------
# Activity contract: numeric events are active iff ``x != 0``.
# ---------------------------------------------------------------------------

_ACTIVITY_DTYPES = [
    jnp.bool_, jnp.int8, jnp.uint8, jnp.int16, jnp.int32,
    jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64,
]


def _activity_events(shape, dtype):
    """Events from ``[2, -1, 0, 0.5]`` (integers: ``[2, -1, 0, 1]``) as ``dtype``."""
    n = int(np.prod(shape))
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        base = np.asarray([True, True, False, True])
    elif np.issubdtype(dtype, np.integer):
        base = np.asarray([2, -1, 0, 1])
    else:
        base = np.asarray([2.0, -1.0, 0.0, 0.5])
    values = np.resize(base, n)
    return values[np.random.RandomState(0).permutation(n)].reshape(shape).astype(dtype)


def _activity_context(dtype):
    return jax_x64_enabled() if np.dtype(dtype) == np.float64 else nullcontext()


@pytest.mark.parametrize('dtype', _ACTIVITY_DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize('shape, axis', [((37,), 0), ((5, 37), 1), ((37, 5), 0)])
def test_bitpack_packs_nonzero_events(dtype, shape, axis):
    from brainevent._event.bitpack_binary import bitpack

    host = _activity_events(shape, dtype)
    with _activity_context(dtype):
        packed = np.asarray(bitpack(jnp.asarray(host), axis=axis))
    mask = np.moveaxis(host != 0, axis, -1)
    n = mask.shape[-1]
    padded = np.zeros(mask.shape[:-1] + (64,), dtype=np.uint64)
    padded[..., :n] = mask
    words = np.stack(
        [(padded[..., 32 * w:32 * (w + 1)] << np.arange(32, dtype=np.uint64)).sum(-1)
         for w in range((n + 31) // 32)],
        axis=-1,
    ).astype(np.uint32)
    np.testing.assert_array_equal(np.moveaxis(packed, axis, -1), words)


def test_bitpack_treats_nan_as_active():
    from brainevent._event.bitpack_binary import bitpack

    packed = bitpack(jnp.asarray([np.nan, 0.0, -0.0, 1.0], dtype=jnp.float32), axis=0)
    np.testing.assert_array_equal(np.asarray(packed), np.asarray([0b1001], dtype=np.uint32))


@pytest.mark.parametrize('dtype', _ACTIVITY_DTYPES, ids=lambda d: np.dtype(d).name)
def test_binary_array_matmul_uses_nonzero_activity(dtype):
    host = _activity_events((6,), dtype)
    host_mat = _activity_events((3, 6), dtype)
    weight = np.arange(1.0, 13.0, dtype=np.float32).reshape(6, 2)
    with _activity_context(dtype):
        w = jnp.asarray(weight)
        vec = np.asarray(BinaryArray(jnp.asarray(host)) @ w)
        mat = np.asarray(BinaryArray(jnp.asarray(host_mat)) @ w)
    np.testing.assert_allclose(vec, (host != 0).astype(np.float32) @ weight, rtol=1e-6)
    np.testing.assert_allclose(mat, (host_mat != 0).astype(np.float32) @ weight, rtol=1e-6)
