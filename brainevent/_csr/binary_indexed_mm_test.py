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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.binary import binary_csrmv, binary_csrmm
from brainevent._csr.binary_indexed_mm import binary_csrmm_indexed


def _structure(rng, m, k, nse):
    indices = rng.integers(0, k, size=nse).astype(np.int32)
    base, rem = nse // m, nse - (nse // m) * m
    rows = np.full(m, base, dtype=int)
    rows[:rem] += 1
    indptr = np.concatenate([[0], np.cumsum(rows)]).astype(np.int32)
    perm = rng.permutation(nse).astype(np.int32)
    return jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(perm)


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("homo", [True, False])
@pytest.mark.parametrize("ev", [jnp.bool_, jnp.float32])
@pytest.mark.parametrize("n", [1, 3])
def test_indexed_mm_matches_materialized(transpose, homo, ev, n):
    rng = np.random.default_rng(0)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.ones(1, jnp.float32) if homo else jnp.asarray(rng.random(nse), jnp.float32)
    rows = m if transpose else k
    B = jnp.asarray(rng.random((rows, n)) > 0.5, dtype=ev)
    got = binary_csrmm_indexed(weights, indices, indptr, perm, B, shape=(m, k), transpose=transpose)
    ref_w = weights if homo else weights[perm]
    ref = binary_csrmm(ref_w, indices, indptr, B, shape=(m, k), transpose=transpose)
    assert jnp.allclose(got, ref, atol=1e-5), (transpose, homo, ev, n)


def test_indexed_mm_single_column_equals_mv():
    rng = np.random.default_rng(3)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.bool_)
    got = binary_csrmm_indexed(weights, indices, indptr, perm, v[:, None], shape=(m, k), transpose=False)
    from brainevent._csr.binary_indexed import binary_csrmv_indexed
    ref = binary_csrmv_indexed(weights, indices, indptr, perm, v, shape=(m, k), transpose=False)
    assert jnp.allclose(got[:, 0], ref, atol=1e-5)


@pytest.mark.parametrize("transpose", [True, False])
def test_indexed_mm_grad_weights(transpose):
    rng = np.random.default_rng(1)
    m, k, nse, n = 4, 5, 9, 3
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    rows = m if transpose else k
    B = jnp.asarray(rng.random((rows, n)) > 0.5, dtype=jnp.float32)
    f = lambda w: binary_csrmm_indexed(w, indices, indptr, perm, B, shape=(m, k), transpose=transpose).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmm(w[perm], indices, indptr, B, shape=(m, k), transpose=transpose).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5), transpose


def test_indexed_mm_grad_weights_homo():
    rng = np.random.default_rng(8)
    m, k, nse, n = 4, 5, 9, 3
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray([0.7], jnp.float32)
    B = jnp.asarray(rng.random((k, n)) > 0.5, dtype=jnp.float32)
    f = lambda w: binary_csrmm_indexed(w, indices, indptr, perm, B, shape=(m, k), transpose=False).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmm(w, indices, indptr, B, shape=(m, k), transpose=False).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5)


def test_indexed_mm_jit():
    rng = np.random.default_rng(5)
    m, k, nse, n = 4, 5, 9, 3
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    B = jnp.asarray(rng.random((k, n)) > 0.5, dtype=jnp.bool_)
    f = jax.jit(lambda w: binary_csrmm_indexed(w, indices, indptr, perm, B, shape=(m, k)))
    got = f(weights)
    ref = binary_csrmm(weights[perm], indices, indptr, B, shape=(m, k))
    assert jnp.allclose(got, ref, atol=1e-5)


def test_indexed_mm_check_grads_weights():
    from jax.test_util import check_grads
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(2)
        m, k, nse, n = 4, 5, 9, 3
        indices, indptr, perm = _structure(rng, m, k, nse)
        weights = jnp.asarray(rng.random(nse), jnp.float64)
        B = jnp.asarray(rng.random((k, n)) > 0.5, dtype=jnp.bool_)
        f = lambda w: binary_csrmm_indexed(w, indices, indptr, perm, B, shape=(m, k)).sum()
        check_grads(f, (weights,), order=2, modes=['rev'])
    finally:
        jax.config.update("jax_enable_x64", False)


def test_indexed_mm_cuda_kernel_selects_perm_names():
    import inspect
    import brainevent._csr.binary_indexed_mm as mod
    src = inspect.getsource(mod._binary_csrmm_indexed_cuda_kernel)
    # hetero selects the perm kernels and passes perm; homo reuses plain kernels.
    assert "binary_csrmm_t_warp_perm_hetero" in src
    assert "binary_csrmm_nt_auto_perm_hetero" in src
    assert "binary_csrmm_t_warp_homo" in src
    assert "binary_csrmm_nt_auto_homo" in src

    from pathlib import Path
    cu = Path(mod.__file__).with_name("binary_csrmm.cu").read_text()
    assert "DEFINE_CSRMM_T_WARP_PERM_HETERO" in cu
    assert "DEFINE_CSRMM_NT_WARP_PERM_HETERO" in cu
    assert "DEFINE_CSRMM_NT_BLOCK_PERM_HETERO" in cu
    assert "binary_csrmm_t_warp_perm_hetero_f32_bool" in cu
    assert "binary_csrmm_nt_auto_perm_hetero_f32_bool" in cu


def test_import_brainevent_needs_no_nvcc():
    # Importing the module must not compile CUDA (load_cuda_file is lazy).
    import importlib
    import brainevent._csr.binary_indexed_mm as mod
    importlib.reload(mod)
    assert hasattr(mod, "binary_csrmm_indexed")


def test_indexed_mm_is_exported():
    import brainevent
    from brainevent import binary_csrmm_indexed, binary_csrmm_indexed_p
    assert "binary_csrmm_indexed" in brainevent.__all__
    assert "binary_csrmm_indexed_p" in brainevent.__all__
    assert binary_csrmm_indexed is brainevent.binary_csrmm_indexed
