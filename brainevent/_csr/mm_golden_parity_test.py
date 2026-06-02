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

"""Golden parity for binary CSR/CSC matrix-matrix dispatch.

Captured BEFORE the event-driven scatter refactor so the change can be checked
for behavioral drift. All assertions are against the dense reference
``M.todense()``, independent of internal layout machinery.
"""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from brainevent import CSR, CSC, BinaryArray


def _csr_cases():
    rng = np.random.default_rng(0)
    m, k = 6, 5
    for homo in (True, False):
        dense = (rng.random((m, k)) + 0.5) * (rng.random((m, k)) > 0.5)
        dense = jnp.asarray(dense, dtype=jnp.float32)
        csr = CSR.fromdense(dense)
        if homo:
            csr = CSR((jnp.asarray([1.5], jnp.float32), csr.indices, csr.indptr), shape=csr.shape)
        yield csr


def _mask(x, dtype):
    x = jnp.asarray(x)
    return jnp.asarray(x > 0, dtype=dtype) if x.dtype != jnp.bool_ else jnp.asarray(x, dtype=dtype)


def test_csr_matmat_golden():
    rng = np.random.default_rng(7)
    for csr in _csr_cases():
        dense = jnp.asarray(csr.todense(), dtype=jnp.float32)
        m, k = csr.shape
        for ev in (jnp.bool_, jnp.float32):
            for n in (1, 3, 8):
                right = jnp.asarray(rng.random((k, n)) > 0.5, dtype=ev)   # CSR @ M (unfavorable)
                got_r = csr @ BinaryArray(right)
                ref_r = dense @ _mask(right, jnp.float32)
                assert jnp.allclose(got_r, ref_r, atol=1e-5), ('CSR@M', str(ev), n)

                left = jnp.asarray(rng.random((n, m)) > 0.5, dtype=ev)    # M @ CSR (favorable)
                got_l = BinaryArray(left) @ csr
                ref_l = _mask(left, jnp.float32) @ dense
                assert jnp.allclose(got_l, ref_l, atol=1e-5), ('M@CSR', str(ev), n)


def test_csc_matmat_golden():
    rng = np.random.default_rng(11)
    for csr in _csr_cases():
        csc = csr.T                          # CSC view; dense transposes too
        dense = jnp.asarray(csc.todense(), dtype=jnp.float32)
        p, q = csc.shape
        for ev in (jnp.bool_, jnp.float32):
            for n in (1, 3, 8):
                right = jnp.asarray(rng.random((q, n)) > 0.5, dtype=ev)   # CSC @ M (favorable)
                got_r = csc @ BinaryArray(right)
                ref_r = dense @ _mask(right, jnp.float32)
                assert jnp.allclose(got_r, ref_r, atol=1e-5), ('CSC@M', str(ev), n)

                left = jnp.asarray(rng.random((n, p)) > 0.5, dtype=ev)    # M @ CSC (unfavorable)
                got_l = BinaryArray(left) @ csc
                ref_l = _mask(left, jnp.float32) @ dense
                assert jnp.allclose(got_l, ref_l, atol=1e-5), ('M@CSC', str(ev), n)


def test_csr_matmat_unfavorable_builds_weight_indices():
    # Routing to the indexed scatter primitive populates the cached transposed
    # structure as a side effect (the gather path never builds it).
    # CSR._weight_indices() caches the CSC view under the 'csc' buffer key.
    rng = np.random.default_rng(2)
    csr = next(_csr_cases())
    k = csr.shape[1]
    right = jnp.asarray(rng.random((k, 4)) > 0.5, dtype=jnp.bool_)
    assert csr.buffers.get('csc') is None
    _ = csr @ BinaryArray(right)
    assert csr.buffers.get('csc') is not None


def test_csc_matmat_unfavorable_builds_weight_indices():
    # CSC._weight_indices() caches the CSR view under the 'csr' buffer key.
    rng = np.random.default_rng(3)
    csc = next(_csr_cases()).T
    p = csc.shape[0]
    left = jnp.asarray(rng.random((4, p)) > 0.5, dtype=jnp.bool_)
    assert csc.buffers.get('csr') is None
    _ = BinaryArray(left) @ csc
    assert csc.buffers.get('csr') is not None


def test_csr_matmat_units_and_jit():
    rng = np.random.default_rng(9)
    csr = next(_csr_cases())
    m, k = csr.shape
    csr_u = CSR((csr.data * u.mV, csr.indices, csr.indptr), shape=csr.shape)
    M = jnp.asarray(rng.random((k, 4)) > 0.5, dtype=jnp.bool_)
    got = csr_u @ BinaryArray(M)
    assert u.get_unit(got) == u.mV
    ref = (jnp.asarray(csr.todense(), jnp.float32) @ jnp.asarray(M, jnp.float32))
    assert jnp.allclose(u.get_mantissa(got), ref, atol=1e-5)

    f = jax.jit(lambda d: CSR((d, csr.indices, csr.indptr), shape=csr.shape) @ BinaryArray(M))
    got_jit = f(csr.data)
    assert jnp.allclose(got_jit, ref, atol=1e-5)
