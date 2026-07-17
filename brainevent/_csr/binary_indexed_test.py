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

import brainevent._csr.main as csr_main
from brainevent._csr.binary import binary_csrmv, binary_csrmm, _workspace_from_task_operands
from brainevent._csr.binary_indexed import (
    _csrmm_idx_batching,
    _idx_batching,
    binary_csrmv_indexed,
    binary_csrmv_indexed_p_call,
    binary_csrmm_indexed,
    binary_csrmm_indexed_p_call,
)
from brainevent._csr.main import _binary_workspace, _make_binary_task_workspace
from brainevent import BinaryArray, CSC, CSR, FixedNumPerPre


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


def test_binary_csrmv_indexed_vmap_keeps_workspace_outputs_unbatched():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    vectors = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
    workspace = _make_binary_task_workspace(indptr)

    out_info = jax.ShapeDtypeStruct((1,), weights.dtype)
    task_begin_info = jax.ShapeDtypeStruct(workspace.task_begin.shape, workspace.task_begin.dtype)
    task_end_info = jax.ShapeDtypeStruct(workspace.task_end.shape, workspace.task_end.dtype)
    status_info = jax.ShapeDtypeStruct(workspace.status.shape, workspace.status.dtype)

    (out, task_begin, task_end, status), out_dims = _idx_batching(
        (weights, indices, indptr, perm, vectors, workspace.task_begin, workspace.task_end, workspace.status),
        (None, None, None, None, 0, None, None, None),
        outs=(out_info, task_begin_info, task_end_info, status_info),
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vectors.shape[1:], vectors.dtype),
        task_begin_info=task_begin_info,
        task_end_info=task_end_info,
        status_info=status_info,
        task_capacity=workspace.task_capacity,
    )

    assert out.shape == (1, 2)
    assert task_begin.shape == workspace.task_begin.shape
    assert task_end.shape == workspace.task_end.shape
    assert status.shape == workspace.status.shape
    assert out_dims == (1, None, None, None)


def test_binary_csrmv_indexed_vmap_matches_materialized_weights():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    vectors = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
    workspace = _make_binary_task_workspace(indptr)

    got = jax.vmap(
        lambda vector: binary_csrmv_indexed(
            weights,
            indices,
            indptr,
            perm,
            vector,
            workspace=workspace,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
        )
    )(vectors)
    ref = jax.vmap(
        lambda vector: binary_csrmv(
            weights[perm],
            indices,
            indptr,
            vector,
            workspace=workspace,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
        )
    )(vectors)

    assert got.shape == (2, 1)
    assert jnp.allclose(got, ref)


def test_binary_csrmm_indexed_jax_raw_accepts_workspace_and_preserves_result():
    weights = jnp.array([10.0, 20.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    matrix = jnp.array([[True, False], [False, True]])
    workspace = _make_binary_task_workspace(indptr)

    got = binary_csrmm_indexed(
        weights,
        indices,
        indptr,
        perm,
        matrix,
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        workspace=workspace,
    )

    assert got.shape == (1, 2)
    assert jnp.allclose(got, jnp.array([[20.0, 10.0]], dtype=jnp.float32))


def test_binary_csrmm_indexed_p_call_always_returns_task_outputs():
    weights = jnp.array([1.0], dtype=jnp.float32)
    indices = jnp.array([0], dtype=jnp.int32)
    indptr = jnp.array([0, 1], dtype=jnp.int32)
    perm = jnp.array([0], dtype=jnp.int32)
    matrix = jnp.array([[True]])
    workspace = _make_binary_task_workspace(indptr)

    out, task_begin, task_end, status = binary_csrmm_indexed_p_call(
        weights,
        indices,
        indptr,
        perm,
        matrix,
        shape=(1, 1),
        transpose=False,
        backend="jax_raw",
        workspace=workspace,
    )

    assert out.shape == (1, 1)
    assert task_begin.shape == workspace.task_begin.shape
    assert task_end.shape == workspace.task_end.shape
    assert status.shape == workspace.status.shape


@pytest.mark.parametrize(
    ("batch_axis", "input_shape", "output_shape", "output_axis"),
    [
        (0, (2, 2, 2), (1, 2, 2), 1),
        (1, (2, 2, 2), (1, 2, 2), 1),
        (2, (2, 2, 2), (1, 2, 2), 2),
    ],
)
def test_binary_csrmm_indexed_vmap_keeps_workspace_outputs_unbatched(
    batch_axis,
    input_shape,
    output_shape,
    output_axis,
):
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    matrices = jnp.arange(np.prod(input_shape)).reshape(input_shape) % 2 == 0
    workspace = _make_binary_task_workspace(indptr)

    out_info = jax.ShapeDtypeStruct((1, 2), weights.dtype)
    task_begin_info = jax.ShapeDtypeStruct(workspace.task_begin.shape, workspace.task_begin.dtype)
    task_end_info = jax.ShapeDtypeStruct(workspace.task_end.shape, workspace.task_end.dtype)
    status_info = jax.ShapeDtypeStruct(workspace.status.shape, workspace.status.dtype)

    (out, task_begin, task_end, status), out_dims = _csrmm_idx_batching(
        (weights, indices, indptr, perm, matrices, workspace.task_begin, workspace.task_end, workspace.status),
        (None, None, None, None, batch_axis, None, None, None),
        outs=(out_info, task_begin_info, task_end_info, status_info),
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(matrices.shape[1:], matrices.dtype),
        task_begin_info=task_begin_info,
        task_end_info=task_end_info,
        status_info=status_info,
        task_capacity=workspace.task_capacity,
    )

    assert out.shape == output_shape
    assert task_begin.shape == workspace.task_begin.shape
    assert task_end.shape == workspace.task_end.shape
    assert status.shape == workspace.status.shape
    assert out_dims == (output_axis, None, None, None)


def test_binary_csrmm_indexed_vmap_matches_materialized_weights():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    matrices = jnp.array(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ],
        dtype=jnp.bool_,
    )
    workspace = _make_binary_task_workspace(indptr)

    got = jax.vmap(
        lambda matrix: binary_csrmm_indexed(
            weights,
            indices,
            indptr,
            perm,
            matrix,
            workspace=workspace,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
        )
    )(matrices)
    ref = jax.vmap(
        lambda matrix: binary_csrmm(
            weights[perm],
            indices,
            indptr,
            matrix,
            workspace=workspace,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
        )
    )(matrices)

    assert got.shape == (2, 1, 2)
    assert jnp.allclose(got, ref)


def test_binary_csrmm_indexed_weight_jvp_zeroes_task_tangents():
    weights = jnp.array([10.0, 20.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    matrix = jnp.array([[True, False], [False, True]])
    workspace = _workspace_from_task_operands(
        2,
        jnp.array([3, 5], dtype=jnp.int32),
        jnp.array([7, 11], dtype=jnp.int32),
        jnp.array([13, 17], dtype=jnp.int32),
    )

    def call(w):
        return binary_csrmm_indexed_p_call(
            w,
            indices,
            indptr,
            perm,
            matrix,
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
            workspace=workspace,
        )

    _, tangents = jax.jvp(call, (weights,), (jnp.ones_like(weights),))
    math_tangent, task_begin_tangent, task_end_tangent, status_tangent = tangents

    assert jnp.allclose(math_tangent, jnp.array([[1.0, 1.0]], dtype=jnp.float32))
    assert jnp.array_equal(task_begin_tangent, jnp.zeros_like(workspace.task_begin))
    assert jnp.array_equal(task_end_tangent, jnp.zeros_like(workspace.task_end))
    assert jnp.array_equal(status_tangent, jnp.zeros_like(workspace.status))


def test_csr_unfavorable_binary_matvec_mounts_direct_workspace_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    csr = CSR.fromdense(dense, backend="jax_raw")
    vector = jnp.array([True, False])

    got = csr @ BinaryArray(vector)

    assert got.shape == (2,)
    assert jnp.allclose(got, dense @ vector.astype(jnp.float32))
    first = _binary_workspace(csr, "csr")
    assert csr.buffers.get("csc") is None

    got_again = csr @ BinaryArray(vector)
    second = _binary_workspace(csr, "csr")

    assert jnp.allclose(got_again, got)
    assert second.task_begin is first.task_begin


def test_csr_unfavorable_binary_matmat_mounts_direct_workspace_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    csr = CSR.fromdense(dense, backend="jax_raw")
    matrix = jnp.array([[True, False], [False, True]])

    got = csr @ BinaryArray(matrix)

    assert got.shape == (2, 2)
    assert jnp.allclose(got, dense @ matrix.astype(jnp.float32))
    first = _binary_workspace(csr, "csr")
    assert csr.buffers.get("csc") is None

    got_again = csr @ BinaryArray(matrix)
    second = _binary_workspace(csr, "csr")

    assert jnp.allclose(got_again, got)
    assert second.task_begin is first.task_begin


def test_csr_direct_binary_rmatmat_mounts_workspace():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    csr = CSR.fromdense(dense, backend="jax_raw")
    matrix = jnp.array([[True, False], [False, True]])

    got = BinaryArray(matrix) @ csr

    assert got.shape == (2, 2)
    assert jnp.allclose(got, matrix.astype(jnp.float32) @ dense)
    first = _binary_workspace(csr, "csr")

    got_again = BinaryArray(matrix) @ csr
    second = _binary_workspace(csr, "csr")

    assert jnp.allclose(got_again, got)
    assert second.task_begin is first.task_begin


def test_csc_unfavorable_binary_rmatvec_mounts_direct_workspace_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    csc = CSC.fromdense(dense, backend="jax_raw")
    vector = jnp.array([True, False])

    got = BinaryArray(vector) @ csc

    assert got.shape == (2,)
    assert jnp.allclose(got, vector.astype(jnp.float32) @ dense)
    first = _binary_workspace(csc, "csc")
    assert csc.buffers.get("csr") is None

    got_again = BinaryArray(vector) @ csc
    second = _binary_workspace(csc, "csc")

    assert jnp.allclose(got_again, got)
    assert second.task_begin is first.task_begin


def test_csc_direct_binary_matmat_mounts_workspace():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    csc = CSC.fromdense(dense, backend="jax_raw")
    matrix = jnp.array([[True, False], [False, True]])

    got = csc @ BinaryArray(matrix)

    assert got.shape == (2, 2)
    assert jnp.allclose(got, dense @ matrix.astype(jnp.float32))
    first = _binary_workspace(csc, "csc")

    got_again = csc @ BinaryArray(matrix)
    second = _binary_workspace(csc, "csc")

    assert jnp.allclose(got_again, got)
    assert second.task_begin is first.task_begin


def test_csc_unfavorable_binary_rmatmat_mounts_direct_workspace_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    csc = CSC.fromdense(dense, backend="jax_raw")
    matrix = jnp.array([[True, False], [False, True]])

    got = BinaryArray(matrix) @ csc

    assert got.shape == (2, 2)
    assert jnp.allclose(got, matrix.astype(jnp.float32) @ dense)
    first = _binary_workspace(csc, "csc")
    assert csc.buffers.get("csr") is None

    got_again = BinaryArray(matrix) @ csc
    second = _binary_workspace(csc, "csc")

    assert jnp.allclose(got_again, got)
    assert second.task_begin is first.task_begin


def test_cuda_raw_unfavorable_binary_routes_use_indexed_mirror_workspaces(monkeypatch):
    dense = jnp.array([[0.0, 2.0, 5.0], [3.0, 4.0, 0.0]], dtype=jnp.float32)
    csr = CSR.fromdense(dense, backend="cuda_raw")
    csc = CSC.fromdense(dense, backend="cuda_raw")
    calls = []

    def fail_direct(*args, **kwargs):
        raise AssertionError("cuda_raw unfavorable BinaryArray route must use indexed primitive")

    def fake_mv_indexed(data, indices, indptr, perm, vector, *, shape, transpose, backend=None, workspace=None):
        calls.append({
            "kind": "mv",
            "shape": shape,
            "transpose": transpose,
            "backend": backend,
            "workspace": workspace,
        })
        out_size = shape[1] if transpose else shape[0]
        return jnp.zeros((out_size,), dtype=data.dtype)

    def fake_mm_indexed(data, indices, indptr, perm, matrix, *, shape, transpose, backend=None, workspace=None):
        calls.append({
            "kind": "mm",
            "shape": shape,
            "transpose": transpose,
            "backend": backend,
            "workspace": workspace,
        })
        out_size = shape[1] if transpose else shape[0]
        return jnp.zeros((out_size, matrix.shape[1]), dtype=data.dtype)

    monkeypatch.setattr(csr_main, "binary_csrmv", fail_direct)
    monkeypatch.setattr(csr_main, "binary_csrmm", fail_direct)
    monkeypatch.setattr(csr_main, "binary_csrmv_indexed", fake_mv_indexed)
    monkeypatch.setattr(csr_main, "binary_csrmm_indexed", fake_mm_indexed)

    assert (csr @ BinaryArray(jnp.array([True, False, True]))).shape == (2,)
    assert (csr @ BinaryArray(jnp.array([[True, False], [False, True], [True, True]]))).shape == (2, 2)
    assert (BinaryArray(jnp.array([True, False])) @ csc).shape == (3,)
    assert (BinaryArray(jnp.array([[True, False], [False, True]])) @ csc).shape == (2, 3)

    csr_workspace = _binary_workspace(csr, "csc")
    csc_workspace = _binary_workspace(csc, "csr")
    assert [(call["kind"], call["shape"], call["transpose"], call["backend"]) for call in calls] == [
        ("mv", (3, 2), True, "cuda_raw"),
        ("mm", (3, 2), True, "cuda_raw"),
        ("mv", (2, 3), True, "cuda_raw"),
        ("mm", (2, 3), True, "cuda_raw"),
    ]
    assert calls[0]["workspace"] is csr_workspace
    assert calls[1]["workspace"] is csr_workspace
    assert calls[2]["workspace"] is csc_workspace
    assert calls[3]["workspace"] is csc_workspace


def test_fcn_unfavorable_binary_matvec_uses_indexed_workspace():
    data = jnp.array([[1.0, 2.0], [3.0, 0.0]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    conn = FixedNumPerPre((data, indices), shape=(2, 2), backend="jax_raw").build_weight_indices()
    vector = jnp.array([False, True])

    got = conn @ BinaryArray(vector)

    assert got.shape == (2,)
    assert jnp.allclose(got, conn.todense() @ vector.astype(jnp.float32))
    first = conn.buffers["binary_workspace"]

    got_again = conn @ BinaryArray(vector)
    second = conn.buffers["binary_workspace"]

    assert jnp.allclose(got_again, got)
    assert second.task_begin is first.task_begin


def test_fcn_unfavorable_binary_matmat_uses_indexed_workspace():
    data = jnp.array([[1.0, 2.0], [3.0, 0.0]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    conn = FixedNumPerPre((data, indices), shape=(2, 2), backend="jax_raw").build_weight_indices()
    matrix = jnp.array([[False, True], [True, False]])

    got = conn @ BinaryArray(matrix)

    assert got.shape == (2, 2)
    assert jnp.allclose(got, conn.todense() @ matrix.astype(jnp.float32))
    first = conn.buffers["binary_workspace"]

    got_again = conn @ BinaryArray(matrix)
    second = conn.buffers["binary_workspace"]

    assert jnp.allclose(got_again, got)
    assert second.task_begin is first.task_begin


def test_indexed_mv_rejects_int64_indices_with_int32_indptr():
    with _jax_x64_enabled():
        weights = jnp.ones(2, dtype=jnp.float32)
        indices = jnp.array([0, 1], dtype=jnp.int64)
        indptr = jnp.array([0, 2], dtype=jnp.int32)
        perm = jnp.array([0, 1], dtype=jnp.int32)
        events = jnp.ones(2, dtype=bool)
        workspace = _make_binary_task_workspace(indptr)

        with pytest.raises(AssertionError, match="indices with dtype int32"):
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
        workspace = _make_binary_task_workspace(indptr)

        got = binary_csrmm_indexed(
            weights, indices, indptr, perm, events, shape=(2, 3), backend='jax_raw', workspace=workspace
        )
        expected = jnp.array([[40.0, 10.0], [20.0, 60.0]], dtype=jnp.float32)

        assert jnp.allclose(got, expected)


def test_indexed_mm_rejects_unsigned_structure_dtype():
    weights = jnp.ones(2, dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.uint32)
    indptr = jnp.array([0, 2], dtype=jnp.uint32)
    perm = jnp.array([0, 1], dtype=jnp.int32)
    events = jnp.ones((2, 1), dtype=bool)
    workspace = _make_binary_task_workspace(indptr)

    with pytest.raises(AssertionError, match="indices with dtype int32"):
        binary_csrmm_indexed(
            weights, indices, indptr, perm, events, shape=(1, 2), backend='jax_raw', workspace=workspace
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
    workspace = _make_binary_task_workspace(indptr)
    got = binary_csrmm_indexed(
        weights, indices, indptr, perm, B, shape=(m, k), transpose=transpose, workspace=workspace
    )
    ref_w = weights if homo else weights[perm]
    ref = binary_csrmm(ref_w, indices, indptr, B, shape=(m, k), transpose=transpose, workspace=workspace)
    assert jnp.allclose(got, ref, atol=1e-5), (transpose, homo, ev, n)


def test_indexed_mm_single_column_equals_mv():
    rng = np.random.default_rng(3)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.bool_)
    workspace = _make_binary_task_workspace(indptr)
    got = binary_csrmm_indexed(
        weights, indices, indptr, perm, v[:, None], shape=(m, k), transpose=False, workspace=workspace
    )
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
    workspace = _make_binary_task_workspace(indptr)
    f = lambda w: binary_csrmm_indexed(
        w, indices, indptr, perm, B, shape=(m, k), transpose=transpose, workspace=workspace
    ).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmm(
        w[perm], indices, indptr, B, shape=(m, k), transpose=transpose, workspace=workspace
    ).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5), transpose


def test_indexed_mm_grad_weights_homo():
    rng = np.random.default_rng(8)
    m, k, nse, n = 4, 5, 9, 3
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray([0.7], jnp.float32)
    B = jnp.asarray(rng.random((k, n)) > 0.5, dtype=jnp.float32)
    workspace = _make_binary_task_workspace(indptr)
    f = lambda w: binary_csrmm_indexed(
        w, indices, indptr, perm, B, shape=(m, k), transpose=False, workspace=workspace
    ).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmm(
        w, indices, indptr, B, shape=(m, k), transpose=False, workspace=workspace
    ).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5)


def test_indexed_mm_jit():
    rng = np.random.default_rng(5)
    m, k, nse, n = 4, 5, 9, 3
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    B = jnp.asarray(rng.random((k, n)) > 0.5, dtype=jnp.bool_)
    workspace = _make_binary_task_workspace(indptr)
    f = jax.jit(lambda w: binary_csrmm_indexed(w, indices, indptr, perm, B, shape=(m, k), workspace=workspace))
    got = f(weights)
    ref = binary_csrmm(weights[perm], indices, indptr, B, shape=(m, k), workspace=workspace)
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
        workspace = _make_binary_task_workspace(indptr)
        f = lambda w: binary_csrmm_indexed(w, indices, indptr, perm, B, shape=(m, k), workspace=workspace).sum()
        check_grads(f, (weights,), order=2, modes=['rev'])
    finally:
        jax.config.update("jax_enable_x64", False)


def test_indexed_mm_cuda_kernel_selects_perm_names():
    import inspect
    import brainevent._csr.binary_indexed as mod
    src = inspect.getsource(mod._binary_csrmm_indexed_cuda_kernel)
    # Transpose routes use hybrid kernels; non-transpose keeps the old nt_auto path.
    assert "binary_indexed_csrmm_sraw_hybrid_hetero" in src
    assert "binary_csrmm_sraw_hybrid_homo" in src
    assert "binary_csrmm_nt_auto_perm_hetero" in src
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
