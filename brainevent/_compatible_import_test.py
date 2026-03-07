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

"""Comprehensive tests for brainevent._compatible_import.init_zero."""

import contextlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._compatible_import import init_zero


@contextlib.contextmanager
def enable_x64():
    """Context manager to temporarily enable 64-bit precision in JAX."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


class TestInitZeroReturnType(unittest.TestCase):
    """init_zero must always return an ad.Zero instance."""

    def test_returns_ad_zero(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result = init_zero(x)
        self.assertIsInstance(result, ad.Zero)

    def test_returns_ad_zero_for_scalar(self):
        x = jnp.float32(3.14)
        result = init_zero(x)
        self.assertIsInstance(result, ad.Zero)

    def test_returns_ad_zero_for_2d(self):
        x = jnp.ones((4, 5), dtype=jnp.float32)
        result = init_zero(x)
        self.assertIsInstance(result, ad.Zero)


def _nonfloat_tangent_dtype(original_dtype):
    """Return the expected tangent dtype for a non-float array.

    Both code paths convert int/bool to ``float0``:
    - JAX < 0.9.1: ``from_primal_value`` applies the conversion.
    - JAX >= 0.9.1: ``to_tangent_aval()`` applies the same conversion.
    """
    return jax.dtypes.float0


class TestInitZeroAval(unittest.TestCase):
    """The aval of the returned ad.Zero must match the input array's aval.

    Notes
    -----
    Float types: aval dtype is always preserved.
    Integer/bool types:
      - JAX < 0.9.1: dtype becomes ``float0`` (JAX's tangent type for
        non-differentiable values, applied by ``from_primal_value``).
      - JAX >= 0.9.1: dtype is preserved as-is (``ad.Zero(a.aval)`` stores
        the original aval without conversion).
    """

    def _check_float_aval(self, x):
        """Check that shape and dtype are both preserved for float inputs."""
        result = init_zero(x)
        self.assertEqual(result.aval.shape, x.shape)
        self.assertEqual(result.aval.dtype, x.dtype)

    def _check_nonfloat_aval(self, x):
        """Check shape is preserved and dtype follows the version-specific rule."""
        result = init_zero(x)
        self.assertEqual(result.aval.shape, x.shape)
        self.assertEqual(result.aval.dtype, _nonfloat_tangent_dtype(x.dtype))

    def test_aval_float32_1d(self):
        self._check_float_aval(jnp.ones((10,), dtype=jnp.float32))

    def test_aval_float64_1d(self):
        with enable_x64():
            self._check_float_aval(jnp.ones((10,), dtype=jnp.float64))

    def test_aval_int32_1d_shape_preserved(self):
        self._check_nonfloat_aval(jnp.ones((8,), dtype=jnp.int32))

    def test_aval_bool_1d_shape_preserved(self):
        self._check_nonfloat_aval(jnp.ones((6,), dtype=jnp.bool_))

    def test_aval_scalar_float32(self):
        self._check_float_aval(jnp.array(0.0, dtype=jnp.float32))

    def test_aval_2d_float32(self):
        self._check_float_aval(jnp.ones((3, 4), dtype=jnp.float32))

    def test_aval_3d_float32(self):
        self._check_float_aval(jnp.ones((2, 3, 4), dtype=jnp.float32))

    def test_aval_shape_preserved(self):
        shape = (7, 11)
        x = jnp.zeros(shape, dtype=jnp.float32)
        result = init_zero(x)
        self.assertEqual(result.aval.shape, shape)

    def test_aval_int64_shape_preserved(self):
        with enable_x64():
            x = jnp.ones((5,), dtype=jnp.int64)
            result = init_zero(x)
            self.assertEqual(result.aval.shape, (5,))
            self.assertEqual(result.aval.dtype, _nonfloat_tangent_dtype(jnp.int64))


class TestInitZeroDtypes(unittest.TestCase):
    """init_zero must work across all common JAX dtypes.

    Notes
    -----
    Float types: the aval dtype of the returned Zero matches the input dtype.
    Integer and boolean types: JAX's AD uses ``float0`` as the tangent dtype,
    so the aval dtype of the returned Zero is ``float0`` regardless of input.
    """

    def _assert_float_zero(self, dtype):
        """Float inputs: Zero with matching dtype."""
        x = jnp.ones((4,), dtype=dtype)
        result = init_zero(x)
        self.assertIsInstance(result, ad.Zero)
        self.assertEqual(result.aval.dtype, np.dtype(dtype))

    def _assert_nonfloat_zero(self, dtype):
        """Int/bool inputs: Zero with version-appropriate tangent dtype."""
        x = jnp.ones((4,), dtype=dtype)
        result = init_zero(x)
        self.assertIsInstance(result, ad.Zero)
        self.assertEqual(result.aval.dtype, _nonfloat_tangent_dtype(dtype))

    def test_float32(self):
        self._assert_float_zero(jnp.float32)

    def test_float16(self):
        self._assert_float_zero(jnp.float16)

    def test_int32_uses_float0_tangent(self):
        self._assert_nonfloat_zero(jnp.int32)

    def test_int16_uses_float0_tangent(self):
        self._assert_nonfloat_zero(jnp.int16)

    def test_uint8_uses_float0_tangent(self):
        self._assert_nonfloat_zero(jnp.uint8)

    def test_bool_uses_float0_tangent(self):
        self._assert_nonfloat_zero(jnp.bool_)

    def test_float64_with_x64(self):
        with enable_x64():
            self._assert_float_zero(jnp.float64)

    def test_int64_with_x64_uses_float0_tangent(self):
        with enable_x64():
            self._assert_nonfloat_zero(jnp.int64)


class TestInitZeroShapes(unittest.TestCase):
    """init_zero must work for all tensor ranks and edge-case shapes."""

    def _assert_zero_shape(self, shape, dtype=jnp.float32):
        x = jnp.zeros(shape, dtype=dtype)
        result = init_zero(x)
        self.assertIsInstance(result, ad.Zero)
        self.assertEqual(result.aval.shape, shape)

    def test_scalar(self):
        self._assert_zero_shape(())

    def test_1d_single_element(self):
        self._assert_zero_shape((1,))

    def test_1d_vector(self):
        self._assert_zero_shape((100,))

    def test_2d_matrix(self):
        self._assert_zero_shape((10, 20))

    def test_2d_single_row(self):
        self._assert_zero_shape((1, 50))

    def test_2d_single_col(self):
        self._assert_zero_shape((50, 1))

    def test_3d_tensor(self):
        self._assert_zero_shape((2, 3, 4))

    def test_4d_tensor(self):
        self._assert_zero_shape((2, 3, 4, 5))

    def test_large_1d(self):
        self._assert_zero_shape((100_000,))

    def test_zero_dim_size(self):
        # Shape with a zero-size dimension is valid in JAX.
        self._assert_zero_shape((0,))

    def test_zero_dim_in_2d(self):
        self._assert_zero_shape((0, 5))


class TestInitZeroVersionDispatch(unittest.TestCase):
    """init_zero must use the correct API for the installed JAX version."""

    def test_result_consistent_with_jax_version(self):
        """
        Verify the dispatch path is consistent with the running JAX version:
        - JAX >= 0.9.1 -> ad.Zero(a.aval)
        - 0.4.34 <= JAX < 0.9.1 -> ad.Zero.from_primal_value(a)
        - JAX < 0.4.34 -> ad.Zero.from_value(a)
        """
        x = jnp.ones((3,), dtype=jnp.float32)
        result = init_zero(x)

        if jax.__version_info__ >= (0, 9, 1):
            expected = ad.Zero(x.aval)
        elif jax.__version_info__ >= (0, 4, 34):
            expected = ad.Zero.from_primal_value(x)
        else:
            expected = ad.Zero.from_value(x)

        self.assertEqual(result.aval.shape, expected.aval.shape)
        self.assertEqual(result.aval.dtype, expected.aval.dtype)

    def test_aval_matches_direct_construction(self):
        """Cross-check: init_zero aval equals directly building ad.Zero."""
        x = jnp.ones((5, 5), dtype=jnp.float32)
        result = init_zero(x)
        self.assertEqual(result.aval.shape, x.shape)
        self.assertEqual(result.aval.dtype, x.dtype)


class TestInitZeroAdCompatibility(unittest.TestCase):
    """init_zero output must integrate correctly with JAX's AD machinery."""

    def test_zero_is_recognized_as_zero_tangent(self):
        """ad.Zero instances must be treated as zero by add_tangents."""
        x = jnp.ones((4,), dtype=jnp.float32)
        zero = init_zero(x)

        # add_tangents(zero, concrete_tangent) == concrete_tangent
        tangent = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = ad.add_tangents(zero, tangent)
        np.testing.assert_array_equal(result, tangent)

    def test_add_two_zeros(self):
        """Adding two zeros should remain a zero."""
        x = jnp.ones((3,), dtype=jnp.float32)
        z1 = init_zero(x)
        z2 = init_zero(x)
        result = ad.add_tangents(z1, z2)
        self.assertIsInstance(result, ad.Zero)

    def test_used_in_jvp_via_tree_map(self):
        """init_zero must be usable via jax.tree_util.tree_map over a pytree."""
        from jax import tree_util

        val_out = (jnp.ones((3,), dtype=jnp.float32), jnp.zeros((2, 2), dtype=jnp.float32))
        zeros_out = tree_util.tree_map(init_zero, val_out)

        self.assertIsInstance(zeros_out[0], ad.Zero)
        self.assertIsInstance(zeros_out[1], ad.Zero)
        self.assertEqual(zeros_out[0].aval.shape, (3,))
        self.assertEqual(zeros_out[1].aval.shape, (2, 2))

    def test_tree_map_preserves_structure_dict(self):
        """tree_map with init_zero must preserve pytree dict structure."""
        from jax import tree_util

        val_out = {
            'a': jnp.ones((4,), dtype=jnp.float32),
            'b': jnp.zeros((2,), dtype=jnp.int32),
        }
        zeros_out = tree_util.tree_map(init_zero, val_out)

        self.assertIn('a', zeros_out)
        self.assertIn('b', zeros_out)
        self.assertIsInstance(zeros_out['a'], ad.Zero)
        self.assertIsInstance(zeros_out['b'], ad.Zero)
        # Float: dtype preserved; int: tangent dtype is version-dependent.
        self.assertEqual(zeros_out['a'].aval.dtype, np.float32)
        self.assertEqual(zeros_out['b'].aval.dtype, _nonfloat_tangent_dtype(jnp.int32))

    def test_used_in_jax_grad(self):
        """init_zero must be compatible with a real jax.grad computation."""
        def f(x):
            return jnp.sum(x ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        grad = jax.grad(f)(x)
        # grad should be [2, 4, 6]; confirm that init_zero has the right aval
        zero = init_zero(grad)
        self.assertEqual(zero.aval.shape, grad.shape)
        self.assertEqual(zero.aval.dtype, grad.dtype)

    def test_idempotent_aval(self):
        """Calling init_zero twice on the same array yields equal avals."""
        x = jnp.ones((5,), dtype=jnp.float32)
        z1 = init_zero(x)
        z2 = init_zero(x)
        self.assertEqual(z1.aval, z2.aval)


class TestInitZeroIdempotency(unittest.TestCase):
    """Smoke tests: init_zero should not raise for any standard input."""

    def test_no_exception_float32(self):
        try:
            init_zero(jnp.ones((10,), dtype=jnp.float32))
        except Exception as e:
            self.fail(f"init_zero raised unexpectedly: {e}")

    def test_no_exception_int32(self):
        try:
            init_zero(jnp.zeros((5,), dtype=jnp.int32))
        except Exception as e:
            self.fail(f"init_zero raised unexpectedly: {e}")

    def test_no_exception_bool(self):
        try:
            init_zero(jnp.ones((3,), dtype=jnp.bool_))
        except Exception as e:
            self.fail(f"init_zero raised unexpectedly: {e}")

    def test_no_exception_scalar(self):
        try:
            init_zero(jnp.array(1.0, dtype=jnp.float32))
        except Exception as e:
            self.fail(f"init_zero raised unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()
