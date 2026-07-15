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
  1. gather   — binary_jitumv(transpose=False)  [reference]
  2. scatter  — binary_jitumv(transpose=True)   [dot identity]
  3. tocsr    — jitu_to_csr → CSR @ ones
  4. todense  — jitu → dense @ ones             [n ≤ 5000]
  5. tofloat  — jitumv(ones)                    [float kernel]
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._jit_uniform.binary import binary_jitumv, binary_jitumv_p
from brainevent._jit_uniform.csr import jitu_to_csr
from brainevent._jit_uniform.float import jitu, jitumv

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

W_LOW = 0.1
W_HIGH = 0.9


def _dot_dtype():
    return jnp.float64 if bool(jax.config.jax_enable_x64) else jnp.float32


def _max_abs_diff(a, b):
    dt = _dot_dtype()
    return float(np.asarray(jnp.max(jnp.abs(a.astype(dt) - b.astype(dt)))))


def _check_backend():
    platform = jax.default_backend()
    available = tuple(binary_jitumv_p.available_backends(platform))
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
    w_low = jnp.asarray(W_LOW, dtype=jnp.float32)
    w_high = jnp.asarray(W_HIGH, dtype=jnp.float32)
    tolerance = float(ATOL) + float(RTOL) * float(n) * float(W_HIGH)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)

        # ---- 1. gather (reference) ----
        @jax.jit
        def _gather(ev):
            return binary_jitumv(w_low, w_high, prob, ev, SEED,
                                 shape=shape, transpose=False, corder=True, backend=BACKEND)
        gather = jax.block_until_ready(_gather(ones))
        gather_f64 = gather.astype(_dot_dtype())

        # ---- 2. scatter (dot identity) ----
        @jax.jit
        def _scatter(ev):
            return binary_jitumv(w_low, w_high, prob, ev, SEED,
                                 shape=shape, transpose=True, corder=True, backend=BACKEND)
        scatter = jax.block_until_ready(_scatter(ones))
        lhs = jnp.dot(gather_f64, ones_f64)
        rhs = jnp.dot(ones_f64, scatter.astype(_dot_dtype()))
        assert float(jnp.abs(lhs - rhs)) <= tolerance, \
            f'scatter dot-identity mismatch: |{lhs} - {rhs}| = {float(jnp.abs(lhs - rhs))} > {tolerance}'

        # ---- 3. tocsr — CSR @ ones ----
        csr = jitu_to_csr(w_low, w_high, prob, SEED,
                          shape=shape, corder=True, backend=BACKEND, matrix_mode='mv')
        diff = _max_abs_diff(csr @ ones_f64, gather_f64)
        assert diff <= tolerance, f'tocsr mismatch: diff={diff} > {tolerance}'

        # ---- 4. todense (skip for large n) ----
        if n <= DENSE_MAX_N:
            dense = jitu(w_low, w_high, prob, SEED,
                         shape=shape, transpose=False, corder=True, backend=BACKEND)
            diff = _max_abs_diff(dense @ ones_f64, gather_f64)
            assert diff <= tolerance, f'todense mismatch: diff={diff} > {tolerance}'

        # ---- 5. tofloat — float kernel matvec ----
        ones_f32 = ones.astype(jnp.float32)
        fresult = jitumv(w_low, w_high, prob, ones_f32, SEED,
                         shape=shape, transpose=False, corder=True, backend=BACKEND)
        diff = _max_abs_diff(fresult, gather_f64)
        assert diff <= tolerance, f'tofloat mismatch: diff={diff} > {tolerance}'
