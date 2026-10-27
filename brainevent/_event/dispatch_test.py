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

import jax.numpy as jnp
import numpy as np
import pytest

import brainevent as be
from brainevent._event.dispatch import as_event_operand


def _spikes(shape):
    return jnp.asarray(np.random.default_rng(0).random(shape) > 0.5)


@pytest.mark.parametrize('wrap', [be.BinaryArray, be.BitPackedBinary, be.CompactBinary.from_array])
@pytest.mark.parametrize('shape', [(6,), (6, 3)])
def test_binary_event_types_are_classified_binary(wrap, shape):
    spikes = _spikes(shape)
    value, weighted = as_event_operand(wrap(spikes))
    assert weighted is False
    assert jnp.array_equal(jnp.asarray(value, bool), spikes)


def test_weighted_and_dense_operands():
    vals = jnp.array([0., 2., 0.5])
    value, weighted = as_event_operand(be.WeightedArray(vals))
    assert weighted is True and jnp.array_equal(value, vals)
    assert as_event_operand(vals) is None


def _matrices():
    rng = np.random.default_rng(1)
    dense = (rng.random((5, 6)) < 0.5) * (rng.random((5, 6)) + 0.5)
    indices = np.stack([rng.choice(6, 2, replace=False) for _ in range(5)])
    return [
        be.CSR.fromdense(jnp.asarray(dense, jnp.float32)),
        be.CSC.fromdense(jnp.asarray(dense, jnp.float32)),
        be.FixedNumPerPre((jnp.ones((5, 2), jnp.float32), jnp.asarray(indices, jnp.int32)), shape=(5, 6)),
        be.JITCScalarR((1.5, 0.5, 3), shape=(5, 6)),
    ]


@pytest.mark.parametrize('M', _matrices(), ids=lambda m: type(m).__name__)
@pytest.mark.parametrize('batch', [None, 3])
def test_compact_binary_matmul_matches_binary_array(M, batch):
    # Regression: ``CompactBinary`` was exported and documented as an event
    # type, but no matrix product recognised it.
    right = _spikes((6,) if batch is None else (6, batch))
    left = _spikes((5,) if batch is None else (batch, 5))
    expected_right = M @ be.BinaryArray(right)
    expected_left = be.BinaryArray(left) @ M
    assert jnp.allclose(M @ be.CompactBinary.from_array(right), expected_right)
    assert jnp.allclose(be.CompactBinary.from_array(left) @ M, expected_left)
