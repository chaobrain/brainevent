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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._dense.indexed_binary import (
    ibdvm_p_call,
    indexed_bdvm_p,
    indexed_bdmm_p_call,
    indexed_bdmm_p,
)

platform = jax.default_backend()
IBDVM_IMPLEMENTATIONS = tuple(indexed_bdvm_p.available_backends(platform))
IBDMM_IMPLEMENTATIONS = tuple(indexed_bdmm_p.available_backends(platform))


@pytest.mark.skipif(not IBDVM_IMPLEMENTATIONS, reason=f'No indexed_bdvm implementation on platform={platform}')
@pytest.mark.parametrize('implementation', IBDVM_IMPLEMENTATIONS)
def test_indexed_bdvm_bounds_and_count_clamp(implementation):
    n_input, n_output = 8, 6
    weights = brainstate.random.randn(n_input, n_output)
    spikes = jnp.ones((n_input,), dtype=jnp.float32)
    indices = jnp.asarray([1, 7, 99, -3], dtype=jnp.int32)
    count = jnp.asarray([16], dtype=jnp.int32)
    result = ibdvm_p_call(spikes, indices, count, weights, backend=implementation)[0]

    weights_np = np.asarray(weights)
    indices_np = np.asarray(indices)
    valid_indices = indices_np[(indices_np >= 0) & (indices_np < n_input)]
    expected = jnp.asarray(weights_np[valid_indices].sum(axis=0), dtype=weights.dtype)
    assert u.math.allclose(result, expected, atol=1e-5, rtol=1e-5)
    jax.block_until_ready((result, expected))


@pytest.mark.skipif(not IBDMM_IMPLEMENTATIONS, reason=f'No indexed_bdmm implementation on platform={platform}')
@pytest.mark.parametrize('implementation', IBDMM_IMPLEMENTATIONS)
def test_indexed_bdmm_bounds_and_count_clamp(implementation):
    batch, n_input, n_output = 2, 9, 5
    weights = brainstate.random.randn(n_input, n_output)
    spikes = jnp.ones((batch, n_input), dtype=jnp.float32)
    indices = jnp.asarray([[0, 3, 99, -1], [8, 1, 4, 7]], dtype=jnp.int32)
    count = jnp.asarray([4, 12], dtype=jnp.int32)
    result = indexed_bdmm_p_call(spikes, indices, count, weights, backend=implementation)[0]

    weights_np = np.asarray(weights)
    indices_np = np.asarray(indices)
    expected_rows = []
    for b in range(batch):
        idx = indices_np[b]
        valid_indices = idx[(idx >= 0) & (idx < n_input)]
        expected_rows.append(weights_np[valid_indices].sum(axis=0))
    expected = jnp.asarray(np.stack(expected_rows, axis=0), dtype=weights.dtype)

    assert u.math.allclose(result, expected, atol=1e-5, rtol=1e-5)
    jax.block_until_ready((result, expected))
