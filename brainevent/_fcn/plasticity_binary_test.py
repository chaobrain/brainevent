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

from brainevent._fcn.plasticity_binary import (
    fcn_plasticity_row_p,
    fcn_plasticity_col_p,
    fcn_plasticity_row_prim_call,
    fcn_plasticity_col_prim_call,
)

PLATFORM = jax.default_backend()
ROW_BACKENDS = tuple(fcn_plasticity_row_p.available_backends(PLATFORM))
COL_BACKENDS = tuple(fcn_plasticity_col_p.available_backends(PLATFORM))


def _ell_ref_row(data, indices, row_spike, col_trace):
    data = np.asarray(data, dtype=np.float64)
    idx = np.asarray(indices)
    active = (np.asarray(row_spike) != 0)
    ct = np.asarray(col_trace, dtype=np.float64)
    out = data.copy()
    for r in range(data.shape[0]):
        if active[r]:
            for k in range(data.shape[1]):
                out[r, k] += ct[idx[r, k]]
    return out


def _ell_ref_col(data, indices, col_spike, row_trace):
    data = np.asarray(data, dtype=np.float64)
    idx = np.asarray(indices)
    active = (np.asarray(col_spike) != 0)
    rt = np.asarray(row_trace, dtype=np.float64)
    out = data.copy()
    for r in range(data.shape[0]):
        for k in range(data.shape[1]):
            if active[idx[r, k]]:
                out[r, k] += rt[r]
    return out


@pytest.mark.parametrize("backend", ROW_BACKENDS)
@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
def test_row_prim(backend, spike_dtype):
    rng = np.random.default_rng(0)
    n_row, n_conn, n_col = 5, 3, 7
    data = jnp.asarray(rng.random((n_row, n_conn)), dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_col, (n_row, n_conn)), dtype=jnp.int32)
    spike = jnp.asarray(rng.random(n_row) > 0.5, dtype=spike_dtype)
    trace = jnp.asarray(rng.random(n_col), dtype=jnp.float32)
    got = fcn_plasticity_row_prim_call(data, indices, spike, trace, backend=backend)[0]
    ref = _ell_ref_row(data, indices, spike, trace)
    assert np.allclose(np.asarray(got), ref, atol=1e-5)


@pytest.mark.parametrize("backend", COL_BACKENDS)
@pytest.mark.parametrize("spike_dtype", [jnp.bool_, jnp.float32])
def test_col_prim(backend, spike_dtype):
    rng = np.random.default_rng(1)
    n_row, n_conn, n_col = 6, 4, 5
    data = jnp.asarray(rng.random((n_row, n_conn)), dtype=jnp.float32)
    indices = jnp.asarray(rng.integers(0, n_col, (n_row, n_conn)), dtype=jnp.int32)
    spike = jnp.asarray(rng.random(n_col) > 0.5, dtype=spike_dtype)
    trace = jnp.asarray(rng.random(n_row), dtype=jnp.float32)
    got = fcn_plasticity_col_prim_call(data, indices, spike, trace, backend=backend)[0]
    ref = _ell_ref_col(data, indices, spike, trace)
    assert np.allclose(np.asarray(got), ref, atol=1e-5)
