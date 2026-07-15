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
"""
  1. gather   — mat @ BinaryArray(ones)          [reference]
  2. scatter  — BinaryArray(ones) @ mat_T        [dot identity]
  3. tocsr    — mat.tocsr() → CSR @ ones
  4. todense  — mat.todense() → dense @ ones     [n ≤ 5000]
  5. tofloat  — mat @ ones_f32                   [float kernel]
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent._event.binary import BinaryArray

# ── GPU-only ───────────────────────────────────────────────────────────
pytestmark = pytest.mark.skipif(
    jax.default_backend() != 'gpu',
    reason='requires GPU backend (cuda_raw)',
)

# ── constants ──────────────────────────────────────────────────────────
BACKEND = 'cuda_raw'
DENSE_MAX_N = 5000
BASE_SIZE = 1000
SEED = 123
RTOL = 1e-4
ATOL = 1e-2

# Scalar
WEIGHT = 0.5


def _dot_dtype():
    return jnp.float64 if bool(jax.config.jax_enable_x64) else jnp.float32


def _max_abs_diff(a, b):
    dt = _dot_dtype()
    return float(np.asarray(jnp.max(jnp.abs(a.astype(dt) - b.astype(dt)))))


def _check_backend():
    """Skip if the required backend is not available on this platform."""
    import brainevent._jit_scalar.binary as _bin
    platform = jax.default_backend()
    available = tuple(_bin.binary_jitsmv_p.available_backends(platform))
    if BACKEND not in available:
        pytest.skip(f'{BACKEND!r} not available for {platform!r}. available={available!r}')


@pytest.mark.parametrize('scale', [1, 2, 5, 10])
@pytest.mark.parametrize('conn', [10, 50, 200])
def test_5way_consistency(scale, conn):
    """gather / scatter / tocsr / todense / tofloat."""
    _check_backend()

    n = scale * BASE_SIZE
    prob = conn / n
    shape = (n, n)
    ones = jnp.ones(n, dtype=jnp.bool_)
    ones_f64 = ones.astype(_dot_dtype())
    tolerance = float(ATOL) + float(RTOL) * float(n) * abs(WEIGHT)

    weight = jnp.asarray(WEIGHT, dtype=jnp.float32)
    mat = brainevent.JITCScalarR(
        (weight, prob, SEED),
        shape=shape, corder=True, backend=BACKEND,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        # ---- 1. gather (reference): mat @ BinaryArray(ones) ----
        @jax.jit
        def _gather(ev):
            return mat @ BinaryArray(ev)
        gather = jax.block_until_ready(_gather(ones))
        gather_f64 = gather.astype(_dot_dtype())

        # ---- 2. scatter (dot identity): BinaryArray(ones) @ mat.T ----
    
        @jax.jit
        def _scatter(ev):
            return BinaryArray(ev) @ mat
        scatter = jax.block_until_ready(_scatter(ones))
        lhs = jnp.dot(gather_f64, ones_f64)
        rhs = jnp.dot(ones_f64, scatter.astype(_dot_dtype()))
        assert float(jnp.abs(lhs - rhs)) <= tolerance, \
            f'scatter dot-identity mismatch: |{lhs} - {rhs}| = {float(jnp.abs(lhs - rhs))} > {tolerance}'

        # ---- 3. tocsr: mat.tocsr() → CSR @ ones ----
        csr = mat.tocsr(matrix_mode='mv')
        diff = _max_abs_diff(csr @ ones_f64, gather_f64)
        assert diff <= tolerance, f'tocsr mismatch: diff={diff} > {tolerance}'

        # ---- 4. todense: mat.todense() → dense @ ones (skip for large n) ----
        if n <= DENSE_MAX_N:
            dense = mat.todense()
            diff = _max_abs_diff(dense @ ones_f64, gather_f64)
            assert diff <= tolerance, f'todense mismatch: diff={diff} > {tolerance}'

        # ---- 5. tofloat: mat @ float vector ----
        ones_f32 = ones.astype(jnp.float32)
        fresult = mat @ ones_f32
        diff = _max_abs_diff(fresult, gather_f64)
        assert diff <= tolerance, f'tofloat mismatch: diff={diff} > {tolerance}'
