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


_W = jnp.asarray([[1., 10., -2.], [100., 0., 3.], [0., 1000., 4.], [5., 6., 7.]], dtype=jnp.float32)


class TestWeightedArrayDenseMatmul:
    """``WeightedArray`` with raw dense matrices (``binary_dense*(weighted=True)``)."""

    @pytest.mark.parametrize('dtype', [jnp.int8, jnp.uint8, jnp.int16, jnp.int32,
                                       jnp.float16, jnp.bfloat16, jnp.float32])
    def test_forward_all_dtypes(self, dtype):
        raw = np.array([2., -1., 0., 0.5])
        if jnp.issubdtype(dtype, jnp.integer):
            raw = np.trunc(raw).astype(np.int64)
        v = jnp.asarray(raw).astype(dtype)
        vf = np.asarray(v.astype(jnp.float32), dtype=np.float64)
        w = np.asarray(_W, dtype=np.float64)
        np.testing.assert_allclose(WeightedArray(v) @ _W, vf @ w, rtol=1e-5)
        np.testing.assert_allclose(_W.T @ WeightedArray(v), w.T @ vf, rtol=1e-5)
        V = jnp.stack([v, jnp.roll(v, 1)])                       # (2, 4)
        Vf = np.asarray(V.astype(jnp.float32), dtype=np.float64)
        np.testing.assert_allclose(WeightedArray(V) @ _W, Vf @ w, rtol=1e-5)
        np.testing.assert_allclose(_W.T @ WeightedArray(V.T), w.T @ Vf.T, rtol=1e-5)

    def test_differs_from_binary(self):
        v = jnp.asarray([2, -1, 0, 3], dtype=jnp.int8)
        np.testing.assert_allclose(WeightedArray(v) @ _W, v.astype(jnp.float32) @ _W)
        np.testing.assert_allclose(BinaryArray(v) @ _W, (v != 0).astype(jnp.float32) @ _W)

    @pytest.mark.parametrize('left', [True, False])
    @pytest.mark.parametrize('ndim', [1, 2])
    @pytest.mark.parametrize('dtype', [jnp.float32, jnp.int8])
    def test_grads(self, left, ndim, dtype):
        from brainevent._test_util import check_event_op_grads

        x = jnp.asarray([2., -1., 0., 3.]).astype(dtype)
        if ndim == 2:
            x = jnp.stack([x, jnp.roll(x, 1)])                   # (2, 4)
        if left:
            op = lambda p, v: WeightedArray(v) @ p
            dense = lambda p, z: z.astype(p.dtype) @ p
        else:
            x = x.T
            op = lambda p, v: p.T @ WeightedArray(v)
            dense = lambda p, z: p.T @ z.astype(p.dtype)
        check_event_op_grads(op, dense, _W, x, mode='weighted')

    def test_jit(self):
        x = WeightedArray(jnp.asarray([0, 2, 1, 0], dtype=jnp.int8))
        out = jax.jit(lambda e, w: e @ w)(x, _W)
        np.testing.assert_allclose(out, jnp.asarray([0., 2., 1., 0.]) @ _W)

    def test_matches_dense_class(self):
        x = WeightedArray(jnp.asarray([0, 2, -1, 1], dtype=jnp.int8))
        np.testing.assert_allclose(x @ _W, x @ brainevent.Dense(_W))
        np.testing.assert_allclose(_W.T @ x, brainevent.Dense(_W.T) @ x)
