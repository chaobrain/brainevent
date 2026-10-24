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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent import BinaryArray, WeightedArray
from brainevent._event import as_event_operand
from brainevent._error import MathError


class TestWeightedArray:
    def test_is_not_a_binary_array(self):
        """Existing ``isinstance(x, BinaryArray)`` checks must not catch weighted events."""
        x = WeightedArray(jnp.array([0., 2.]))
        assert not isinstance(x, BinaryArray)
        assert isinstance(x, brainevent.EventRepresentation)

    @pytest.mark.parametrize('dtype', [jnp.int8, jnp.uint8, jnp.int16, jnp.int32,
                                       jnp.float16, jnp.bfloat16, jnp.float32, jnp.bool_])
    def test_accepts_numeric_dtypes(self, dtype):
        x = WeightedArray(jnp.zeros(3, dtype=dtype))
        assert x.dtype == dtype

    def test_rejects_complex(self):
        with pytest.raises(TypeError, match='boolean, integer or floating-point'):
            WeightedArray(jnp.zeros(3, dtype=jnp.complex64))

    def test_pytree_roundtrip_under_jit(self):
        x = WeightedArray(jnp.array([0, 3, 1], dtype=jnp.int8))
        leaves, treedef = jax.tree_util.tree_flatten(x)
        y = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(y, WeightedArray)
        out = jax.jit(lambda e: e)(x)
        assert isinstance(out, WeightedArray)
        np.testing.assert_array_equal(out.value, x.value)

    def test_transpose(self):
        v = jnp.arange(6.).reshape(2, 3)
        x = WeightedArray(v)
        np.testing.assert_array_equal(x.T, v.T)
        np.testing.assert_array_equal(x.transpose(1, 0), v.T)

    def test_matmul_shape_errors(self):
        W = jnp.ones((3, 2))
        with pytest.raises(MathError):
            WeightedArray(jnp.array(1.)) @ W
        with pytest.raises(MathError):
            WeightedArray(jnp.ones((2, 2, 3))) @ W
        with pytest.raises(AssertionError):
            WeightedArray(jnp.ones(4)) @ W
        with pytest.raises(AssertionError):
            W @ WeightedArray(jnp.ones(3))


class TestAsEventOperand:
    def test_classification(self):
        v = jnp.array([0., 2.])
        value, weighted = as_event_operand(WeightedArray(v))
        assert weighted and value is v
        value, weighted = as_event_operand(BinaryArray(v))
        assert not weighted and value is v
        assert as_event_operand(v) is None
        assert as_event_operand(np.asarray(v)) is None
