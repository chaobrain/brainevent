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

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.binary import binary_csrmv, binary_csrmm
from brainevent._csr.binary_indexed import (
    binary_csrmv_indexed,
    binary_csrmv_indexed_p_call,
    binary_csrmm_indexed,
)
from brainevent._csr.main import _make_binary_task_workspace


@contextmanager
def _jax_x64_enabled():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def _structure(rng, m, k, nse):
    indices = rng.integers(0, k, size=nse).astype(np.int32)
    # contiguous indptr summing to nse
    counts = np.diff(np.sort(rng.integers(0, nse + 1, size=m - 1))) if m > 1 else np.array([], dtype=int)
    # simpler deterministic indptr
    base = nse // m
    rem = nse - base * m
    rows = np.full(m, base, dtype=int)
    rows[:rem] += 1
    indptr = np.concatenate([[0], np.cumsum(rows)]).astype(np.int32)
    perm = rng.permutation(nse).astype(np.int32)
    return jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(perm)


def test_indexed_mv_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
        events = jnp.array([True, False, True])
        workspace = _make_binary_task_workspace(indptr)

        got = binary_csrmv_indexed(
            weights, indices, indptr, perm, events, shape=(2, 3), backend='jax_raw', workspace=workspace
        )

        assert jnp.allclose(got, jnp.array([40.0, 20.0], dtype=jnp.float32))


def test_binary_csrmv_indexed_jax_raw_accepts_workspace_and_preserves_result():
    weights = jnp.array([10.0, 20.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    vector = jnp.array([True, False])
    workspace = _make_binary_task_workspace(indptr)

    got = binary_csrmv_indexed(
        weights,
        indices,
        indptr,
        perm,
        vector,
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        workspace=workspace,
    )

    assert got.shape == (1,)
    assert jnp.allclose(got, jnp.array([20.0], dtype=jnp.float32))


def test_binary_csrmv_indexed_p_call_always_returns_task_outputs():
    weights = jnp.array([1.0], dtype=jnp.float32)
    indices = jnp.array([0], dtype=jnp.int32)
    indptr = jnp.array([0, 1], dtype=jnp.int32)
    perm = jnp.array([0], dtype=jnp.int32)
    vector = jnp.array([True])
    workspace = _make_binary_task_workspace(indptr)

    out, task_begin, task_end, status = binary_csrmv_indexed_p_call(
        weights,
        indices,
        indptr,
        perm,
        vector,
        shape=(1, 1),
        transpose=False,
        backend="jax_raw",
        workspace=workspace,
    )

    assert out.shape == (1,)
    assert task_begin.shape == workspace.task_begin.shape
    assert task_end.shape == workspace.task_end.shape
    assert status.shape == workspace.status.shape


def test_indexed_mv_rejects_int64_indices_with_int32_indptr():
    with _jax_x64_enabled():
        weights = jnp.ones(2, dtype=jnp.float32)
        indices = jnp.array([0, 1], dtype=jnp.int64)
        indptr = jnp.array([0, 2], dtype=jnp.int32)
        perm = jnp.array([0, 1], dtype=jnp.int32)
        events = jnp.ones(2, dtype=bool)
        workspace = _make_binary_task_workspace(indptr)

        with pytest.raises(AssertionError, match="same dtype"):
            binary_csrmv_indexed(
                weights, indices, indptr, perm, events, shape=(1, 2), backend='jax_raw', workspace=workspace
            )


def test_indexed_mm_accepts_int32_indices_with_int64_indptr_on_jax_raw():
    with _jax_x64_enabled():
        weights = jnp.array([10.0, 20.0, 30.0, 40.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int64)
        perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
        events = jnp.array([[True, False], [False, True], [True, True]])

        got = binary_csrmm_indexed(
            weights, indices, indptr, perm, events, shape=(2, 3), backend='jax_raw'
        )
        expected = jnp.array([[40.0, 10.0], [20.0, 60.0]], dtype=jnp.float32)

        assert jnp.allclose(got, expected)


def test_indexed_mm_rejects_unsigned_structure_dtype():
    weights = jnp.ones(2, dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.uint32)
    indptr = jnp.array([0, 2], dtype=jnp.uint32)
    perm = jnp.array([0, 1], dtype=jnp.int32)
    events = jnp.ones((2, 1), dtype=bool)

    with pytest.raises(AssertionError, match="signed int32 or int64"):
        binary_csrmm_indexed(
            weights, indices, indptr, perm, events, shape=(1, 2), backend='jax_raw'
        )


@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("homo", [True, False])
@pytest.mark.parametrize("ev", [jnp.bool_, jnp.float32])
def test_indexed_matches_materialized(transpose, homo, ev):
    rng = np.random.default_rng(0)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.ones(1, jnp.float32) if homo else jnp.asarray(rng.random(nse), jnp.float32)
    vlen = m if transpose else k
    v = jnp.asarray(rng.random(vlen) > 0.5, dtype=ev)
    workspace = _make_binary_task_workspace(indptr)
    got = binary_csrmv_indexed(
        weights, indices, indptr, perm, v, shape=(m, k), transpose=transpose, workspace=workspace
    )
    ref_w = weights if homo else weights[perm]
    ref = binary_csrmv(ref_w, indices, indptr, v, shape=(m, k), transpose=transpose, workspace=workspace)
    assert jnp.allclose(got, ref, atol=1e-5), (transpose, homo, ev)


def test_indexed_jit():
    rng = np.random.default_rng(5)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.bool_)
    workspace = _make_binary_task_workspace(indptr)
    f = jax.jit(lambda w: binary_csrmv_indexed(w, indices, indptr, perm, v, shape=(m, k), workspace=workspace))
    got = f(weights)
    ref = binary_csrmv(weights[perm], indices, indptr, v, shape=(m, k), workspace=workspace)
    assert jnp.allclose(got, ref, atol=1e-5)


def test_indexed_grad_weights():
    rng = np.random.default_rng(1)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.float32)
    workspace = _make_binary_task_workspace(indptr)
    f = lambda w: binary_csrmv_indexed(
        w, indices, indptr, perm, v, shape=(m, k), transpose=False, workspace=workspace
    ).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmv(
        w[perm], indices, indptr, v, shape=(m, k), transpose=False, workspace=workspace
    ).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5)


def test_indexed_grad_weights_transpose():
    rng = np.random.default_rng(11)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(m) > 0.5, dtype=jnp.float32)
    workspace = _make_binary_task_workspace(indptr)
    f = lambda w: binary_csrmv_indexed(
        w, indices, indptr, perm, v, shape=(m, k), transpose=True, workspace=workspace
    ).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmv(
        w[perm], indices, indptr, v, shape=(m, k), transpose=True, workspace=workspace
    ).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5)


def test_indexed_check_grads_weights():
    # Only ``weights`` is genuinely differentiable: the event indicator
    # e(v) = (v > 0) is a step function, so the v-gradient is a surrogate
    # (identical routing to binary_csrmv_p) and is not finite-difference checkable.
    # With v held constant the output is exactly linear in weights.
    from jax.test_util import check_grads
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(2)
        m, k, nse = 4, 5, 9
        indices, indptr, perm = _structure(rng, m, k, nse)
        weights = jnp.asarray(rng.random(nse), jnp.float64)
        v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.bool_)
        workspace = _make_binary_task_workspace(indptr)
        f = lambda w: binary_csrmv_indexed(w, indices, indptr, perm, v, shape=(m, k), workspace=workspace).sum()
        check_grads(f, (weights,), order=2, modes=['rev'])
    finally:
        jax.config.update("jax_enable_x64", False)


# =========================================================================== #
# Matrix-matrix product: binary_csrmm_indexed
# =========================================================================== #


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
    workspace = _make_binary_task_workspace(indptr)
    ref = binary_csrmv_indexed(
        weights, indices, indptr, perm, v, shape=(m, k), transpose=False, workspace=workspace
    )
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
    import brainevent._csr.binary_indexed as mod
    src = inspect.getsource(mod._binary_csrmm_indexed_cuda_kernel)
    # hetero selects the perm kernels and passes perm; homo reuses plain kernels.
    assert "binary_csrmm_t_warp_perm_hetero" in src
    assert "binary_csrmm_nt_auto_perm_hetero" in src
    assert "binary_csrmm_t_warp_homo" in src
    assert "binary_csrmm_nt_auto_homo" in src

    from pathlib import Path
    cu = Path(mod.__file__).with_name("binary_indexed_csrmm.cu").read_text()
    assert "DEFINE_CSRMM_T_WARP_PERM_HETERO" in cu
    assert "DEFINE_CSRMM_NT_WARP_PERM_HETERO" in cu
    assert "DEFINE_CSRMM_NT_BLOCK_PERM_HETERO" in cu
    assert "binary_csrmm_t_warp_perm_hetero_f32_bool" in cu
    assert "binary_csrmm_nt_auto_perm_hetero_f32_bool" in cu


def test_import_brainevent_needs_no_nvcc():
    # Importing the module must not compile CUDA (load_cuda_file is lazy).
    import importlib
    import brainevent._csr.binary_indexed as mod
    importlib.reload(mod)
    assert hasattr(mod, "binary_csrmm_indexed")


def test_indexed_mm_is_exported():
    import brainevent
    from brainevent import binary_csrmm_indexed, binary_csrmm_indexed_p
    assert "binary_csrmm_indexed" in brainevent.__all__
    assert "binary_csrmm_indexed_p" in brainevent.__all__
    assert binary_csrmm_indexed is brainevent.binary_csrmm_indexed
