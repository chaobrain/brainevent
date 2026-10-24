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
import brainevent._csr.binary_indexed as binary_indexed_mod
from brainevent._csr.main import _binary_workspace, _make_binary_task_workspace
from brainevent._csr._test_util import (
    cuda_kwargs, int64_structure, recording_ffi_call, requires_gpu_backend,
    shape_of, strip_hybrid_suffix,
)
from brainevent._test_util import jax_x64_enabled
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

    def fake_mv_indexed(data, indices, indptr, perm, vector, *, shape, transpose, backend=None, workspace=None,
                        weighted=False):
        calls.append({
            "kind": "mv",
            "shape": shape,
            "transpose": transpose,
            "backend": backend,
            "workspace": workspace,
        })
        out_size = shape[1] if transpose else shape[0]
        return jnp.zeros((out_size,), dtype=data.dtype)

    def fake_mm_indexed(data, indices, indptr, perm, matrix, *, shape, transpose, backend=None, workspace=None,
                        weighted=False):
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

        with pytest.raises(AssertionError, match="Indices must be int32"):
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

    with pytest.raises(AssertionError, match="Indices must be int32"):
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
    # ``nt_auto_perm`` is the only exported permuted entry point: it dispatches
    # internally to the warp and block device kernels, which therefore stay.
    assert "DEFINE_CSRMM_NT_WARP_PERM_HETERO" in cu
    assert "DEFINE_CSRMM_NT_BLOCK_PERM_HETERO" in cu
    assert "binary_csrmm_nt_auto_perm_hetero_f32_bool" in cu
    # The transpose route moved to the hybrid kernels, so the permuted
    # ``t_warp`` family has no caller and must not come back.
    assert "DEFINE_CSRMM_T_WARP_PERM_HETERO" not in cu
    assert "FFI_CSRMM_T_WARP_PERM_HETERO" not in cu
    assert "binary_csrmm_t_warp_perm_hetero" not in cu


def test_nontransposed_raw_cuda_perm_uses_indptr_dtype():
    from pathlib import Path

    csr_dir = Path(binary_indexed_mod.__file__).parent
    for filename in ("binary_indexed_csrmv.cu", "binary_indexed_csrmm.cu"):
        source = (csr_dir / filename).read_text()
        assert "const IndptrT*  __restrict__ perm" in source
        assert "static_cast<const IndptrT*>(perm.data_ptr())" in source
        assert "BE_CHECK(perm.dtype() == indptr.dtype())" in source
        assert "abort()" not in source
        assert "const int32_t*  __restrict__ perm" not in source


def test_import_brainevent_needs_no_nvcc():
    # Importing the module must not compile CUDA (load_cuda_file is lazy).
    import importlib
    import brainevent._csr.binary_indexed as mod
    importlib.reload(mod)
    assert hasattr(mod, "binary_csrmm_indexed")


def test_indexed_mm_is_exported():
    import brainevent
    from brainevent import binary_csrmm_indexed
    assert "binary_csrmm_indexed" in brainevent.__all__
    assert "binary_csrmm_indexed_p" in brainevent.__all__
    assert binary_csrmm_indexed is brainevent.binary_csrmm_indexed


# ---------------------------------------------------------------------------
# int64 ``indptr`` policy on the CUDA path.
#
# ``indices`` stay int32 (the CUDA ABI is int32-only for coordinates) while
# ``indptr`` may widen to int64. The generator tests run without a real GPU by
# stubbing ``load_cuda_file``/``ffi_call``; the ``accepts`` tests need one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'factory,args,kwargs',
    [
        (
            binary_indexed_mod._binary_csrmv_indexed_cuda_kernel,
            (shape_of(jnp.float32), shape_of(jnp.bool_), False),
            {'outs': [shape_of(jnp.float32)], 'perm_info': shape_of(jnp.int32)},
        ),
        (
            binary_indexed_mod._binary_csrmm_indexed_cuda_kernel,
            (shape_of(jnp.float32), shape_of(jnp.bool_, (2, 1)), False),
            {'outs': [shape_of(jnp.float32, (1, 1))], 'perm_info': shape_of(jnp.int32)},
        ),
    ],
)
def test_cuda_kernel_generators_reject_int64_indices_before_loading_cuda(factory, args, kwargs):
    call_kwargs = cuda_kwargs()
    call_kwargs.update(kwargs)

    with pytest.raises(TypeError, match="indices with dtype int32"):
        factory(*args, **call_kwargs)


def test_binary_indexed_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(binary_indexed_mod, "load_cuda_file", lambda path, name, **kwargs: load_calls.append((path, name, kwargs)))
    monkeypatch.setattr(binary_indexed_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    with jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)
        perm = jnp.array([1, 0], dtype=jnp.int32)
        workspace = _make_binary_task_workspace(indptr)
        task_kwargs = {
            'task_begin_info': shape_of(workspace.task_begin.dtype, workspace.task_begin.shape),
            'task_end_info': shape_of(workspace.task_end.dtype, workspace.task_end.shape),
            'status_info': shape_of(workspace.status.dtype, workspace.status.shape),
            'task_capacity': workspace.task_capacity,
        }
        mv_task_outs = (
            shape_of(jnp.float32),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )
        mm_nt_task_outs = (
            shape_of(jnp.float32, (1, 1)),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )
        mm_t_task_outs = (
            shape_of(jnp.float32, (2, 1)),
            task_kwargs['task_begin_info'],
            task_kwargs['task_end_info'],
            task_kwargs['status_info'],
        )

        mv_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (2,)),
            False,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'perm_info': shape_of(jnp.int32, (2,)),
                **task_kwargs,
            },
        )
        mv_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            perm,
            jnp.array([True, False]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mv_t_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (1,)),
            True,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'perm_info': shape_of(jnp.int64, (2,)),
                'outs': mv_task_outs,
                **task_kwargs,
            },
        )
        mv_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            perm.astype(jnp.int64),
            jnp.array([True]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mv_t_homo_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (1,)),
            True,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'perm_info': shape_of(jnp.int64, (2,)),
                'outs': mv_task_outs,
                **task_kwargs,
            },
        )
        mv_t_homo_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            perm.astype(jnp.int64),
            jnp.array([True]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_nt_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (2, 1)),
            False,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_nt_task_outs,
                'perm_info': shape_of(jnp.int32, (2,)),
                **task_kwargs,
            },
        )
        mm_nt_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            perm,
            jnp.array([[True], [False]]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_t_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (1, 1)),
            True,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_t_task_outs,
                'perm_info': shape_of(jnp.int64, (2,)),
                **task_kwargs,
            },
        )
        mm_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32),
            indices,
            indptr,
            perm.astype(jnp.int64),
            jnp.array([[True]]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

        mm_t_homo_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (1, 1)),
            True,
            **{
                **cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64),
                'outs': mm_t_task_outs,
                'perm_info': shape_of(jnp.int64, (2,)),
                **task_kwargs,
            },
        )
        mm_t_homo_kernel(
            jnp.array([2.0], dtype=jnp.float32),
            indices,
            indptr,
            perm.astype(jnp.int64),
            jnp.array([[True]]),
            workspace.task_begin,
            workspace.task_end,
            workspace.status,
        )

    assert [strip_hybrid_suffix(name) for _, name, _ in load_calls] == [
        'csr_binary_indexed_csrmv',
        'csr_binary_indexed_csrmv_hybrid',
        'csr_binary_csrmv_hybrid',
        'csr_binary_indexed_csrmm',
        'csr_binary_indexed_csrmm_hybrid',
        'csr_binary_csrmm_hybrid',
    ]
    assert [strip_hybrid_suffix(call[0]) for call in ffi_calls] == [
        'csr_binary_indexed_csrmv.binary_csrmv_nt_auto_perm_hetero_f32_bool',
        'csr_binary_indexed_csrmv_hybrid.binary_indexed_csrmv_wat_hybrid_hetero_f32_bool',
        'csr_binary_csrmv_hybrid.binary_csrmv_wat_hybrid_homo_f32_bool',
        'csr_binary_indexed_csrmm.binary_csrmm_nt_auto_perm_hetero_f32_bool',
        'csr_binary_indexed_csrmm_hybrid.binary_indexed_csrmm_sraw_hybrid_hetero_f32_bool',
        'csr_binary_csrmm_hybrid.binary_csrmm_sraw_hybrid_homo_f32_bool',
    ]


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
def test_binary_indexed_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    vector = jnp.array([True, False], dtype=jnp.bool_) if transpose else jnp.array([True, False, True])
    workspace64 = _make_binary_task_workspace(indptr64)
    workspace32 = _make_binary_task_workspace(indptr32)

    got = binary_csrmv_indexed(weights, indices, indptr64, perm, vector, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw', workspace=workspace64)
    expected = binary_csrmv_indexed(weights, indices, indptr32, perm, vector, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw', workspace=workspace32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
def test_binary_indexed_csrmm_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    matrix = (
        jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        if transpose else
        jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    )
    workspace64 = _make_binary_task_workspace(indptr64)
    workspace32 = _make_binary_task_workspace(indptr32)

    got = binary_csrmm_indexed(weights, indices, indptr64, perm, matrix, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw', workspace=workspace64)
    expected = binary_csrmm_indexed(weights, indices, indptr32, perm, matrix, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw', workspace=workspace32)

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Event activity contract (``x != 0``), CUDA event ABI and #198 weight
# gradients for the indexed (perm-gather) products.
# ---------------------------------------------------------------------------

from brainevent._csr.binary_indexed import binary_csrmv_indexed_p, binary_csrmm_indexed_p  # noqa: E402
from brainevent._csr._test_util import (  # noqa: E402
    EVENT_DTYPE_NAMES, event_dtype_context, event_mask, event_values,
    random_csr_numpy, run_python_isolated,
)
from brainevent._test_util import check_event_op_grads, requires_gpu  # noqa: E402

_PLATFORM = jax.default_backend()
_IDX_MV_BACKENDS = tuple(binary_csrmv_indexed_p.available_backends(_PLATFORM))
_IDX_MM_BACKENDS = tuple(binary_csrmm_indexed_p.available_backends(_PLATFORM))
_ACT_M, _ACT_K, _ACT_N = 11, 8, 3


def _hp_matmul(a, b):
    """Full-precision float32 matmul (GPU defaults to TF32, ~1e-3 error)."""
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _indexed_operands(homo, seed=0):
    """Return ``(weights, indices, indptr, perm, rows)`` with a random slot permutation."""
    rng = np.random.default_rng(seed)
    _, indices, indptr, _ = random_csr_numpy(rng, _ACT_M, _ACT_K)
    nse = indices.size
    perm = rng.permutation(nse).astype(np.int32)
    weights = (np.array([1.5], np.float32) if homo
               else rng.standard_normal(nse).astype(np.float32))
    rows = np.repeat(np.arange(_ACT_M), np.diff(indptr))
    return jnp.asarray(weights), jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(perm), rows


def _indexed_dense(weights, indices, perm, rows):
    """Differentiable dense matrix with ``D[rows[j], indices[j]] = weights[perm[j]]``."""
    cols = np.asarray(indices)
    vals = jnp.broadcast_to(weights, (cols.size,)) if weights.size == 1 else weights[perm]
    return jnp.zeros((_ACT_M, _ACT_K), weights.dtype).at[rows, cols].add(vals)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mv_event_is_active_iff_nonzero(backend, dtype_name, homo, transpose):
    with event_dtype_context(dtype_name):
        weights, indices, indptr, perm, rows = _indexed_operands(homo)
        dense = np.asarray(_indexed_dense(weights, indices, perm, rows), dtype=np.float64)
        x = event_values(dtype_name, _ACT_M if transpose else _ACT_K)
        got = binary_csrmv_indexed(
            weights, indices, indptr, perm, x, shape=(_ACT_M, _ACT_K), transpose=transpose,
            backend=backend, workspace=_make_binary_task_workspace(indptr),
        )
        mask = event_mask(x)
        expected = mask @ dense if transpose else dense @ mask
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('dtype_name', EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mm_event_is_active_iff_nonzero(backend, dtype_name, homo, transpose):
    with event_dtype_context(dtype_name):
        weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=1)
        dense = np.asarray(_indexed_dense(weights, indices, perm, rows), dtype=np.float64)
        B = event_values(dtype_name, (_ACT_M if transpose else _ACT_K, _ACT_N), seed=1)
        got = binary_csrmm_indexed(
            weights, indices, indptr, perm, B, shape=(_ACT_M, _ACT_K), transpose=transpose,
            backend=backend, workspace=_make_binary_task_workspace(indptr),
        )
        mask = event_mask(B)
        expected = dense.T @ mask if transpose else dense @ mask
        np.testing.assert_allclose(np.asarray(got, dtype=np.float64), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mv_gradients_use_event_mask(backend, homo, transpose):
    """#198: the perm-scattered weight cotangent uses the activity mask."""
    weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=2)
    x = event_values('float32', _ACT_M if transpose else _ACT_K, seed=2)

    def op(w, v):
        return binary_csrmv_indexed(w, indices, indptr, perm, v, shape=(_ACT_M, _ACT_K),
                                    transpose=transpose, backend=backend,
                                    workspace=_make_binary_task_workspace(indptr))

    def dense_op(w, z):
        D = _indexed_dense(w, indices, perm, rows)
        return _hp_matmul(z, D) if transpose else _hp_matmul(D, z)

    check_event_op_grads(op, dense_op, weights, x, mode='binary')


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mm_gradients_use_event_mask(backend, homo, transpose):
    """#198 for the indexed matrix product (SDDMM on the mask, scattered via perm)."""
    weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=3)
    B = event_values('float32', (_ACT_M if transpose else _ACT_K, _ACT_N), seed=3)

    def op(w, v):
        return binary_csrmm_indexed(w, indices, indptr, perm, v, shape=(_ACT_M, _ACT_K),
                                    transpose=transpose, backend=backend,
                                    workspace=_make_binary_task_workspace(indptr))

    def dense_op(w, z):
        D = _indexed_dense(w, indices, perm, rows)
        return _hp_matmul(D.T, z) if transpose else _hp_matmul(D, z)

    check_event_op_grads(op, dense_op, weights, B, mode='binary')


_BAD_PERM_CALL = r'''
import sys
from pathlib import Path
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import brainevent._csr.binary_indexed as mod
from brainevent._op import load_cuda_file
load_cuda_file(Path(mod.__file__).parent / 'binary_indexed_csrmv.cu', name='csr_binary_indexed_csrmv')
weights = jnp.ones(3, jnp.float32)
indices = jnp.array([0, 1, 2], jnp.int32)
indptr = jnp.array([0, 2, 3], jnp.int32)
perm = jnp.array([2, 0, 1], jnp.int64)  # must match indptr (int32)
vector = jnp.array([1, 0, 1], jnp.int8)
try:
    r = jax.ffi.ffi_call('csr_binary_indexed_csrmv.binary_csrmv_nt_auto_perm_hetero_f32_bool',
                         jax.ShapeDtypeStruct((2,), jnp.float32))(weights, indices, indptr, perm, vector)
    jax.block_until_ready(r)
except Exception as e:
    print('RAISED', type(e).__name__, e)
    sys.exit(0)
print('NO ERROR')
sys.exit(3)
'''


@requires_gpu
def test_indexed_cuda_perm_dtype_mismatch_raises_python_exception():
    """The perm/indptr dtype check throws (Python exception) instead of aborting."""
    proc = run_python_isolated(_BAD_PERM_CALL)
    output = proc.stdout + proc.stderr
    assert proc.returncode >= 0, f'process killed by signal {-proc.returncode}:\n{output}'
    assert proc.returncode == 0, output
    assert 'require perm dtype' in output, output


# ---------------------------------------------------------------------------
# Value-weighted events (``weighted=True``, WeightedArray, #201).
# ---------------------------------------------------------------------------

from brainevent._csr._test_util import WEIGHTED_EVENT_DTYPE_NAMES, heavy_row_csr_numpy  # noqa: E402


def _weighted_tol(dtype_name):
    # Event tangents/cotangents of float16/bfloat16 events are rounded to the event dtype.
    if dtype_name in ('float16', 'bfloat16'):
        return dict(rtol=2e-2, atol=2e-2)
    return dict(rtol=1e-4, atol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('dtype_name', WEIGHTED_EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mv_weighted_matches_dense_and_grads(backend, dtype_name, homo, transpose):
    """Weighted indexed mat-vec: exact product, weight cotangent scattered via perm."""
    with event_dtype_context(dtype_name):
        weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=4)
        x = event_values(dtype_name, _ACT_M if transpose else _ACT_K, seed=4)

        def op(w, v):
            return binary_csrmv_indexed(w, indices, indptr, perm, v, shape=(_ACT_M, _ACT_K),
                                        transpose=transpose, backend=backend,
                                        workspace=_make_binary_task_workspace(indptr), weighted=True)

        def dense_op(w, z):
            D = _indexed_dense(w, indices, perm, rows)
            z = z.astype(w.dtype)
            return _hp_matmul(z, D) if transpose else _hp_matmul(D, z)

        check_event_op_grads(op, dense_op, weights, x, mode='weighted',
                             check_vmap=dtype_name == 'float32', **_weighted_tol(dtype_name))


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('dtype_name', WEIGHTED_EVENT_DTYPE_NAMES)
@pytest.mark.parametrize('homo', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_indexed_mm_weighted_matches_dense_and_grads(backend, dtype_name, homo, transpose):
    with event_dtype_context(dtype_name):
        weights, indices, indptr, perm, rows = _indexed_operands(homo, seed=5)
        B = event_values(dtype_name, (_ACT_M if transpose else _ACT_K, _ACT_N), seed=5)

        def op(w, v):
            return binary_csrmm_indexed(w, indices, indptr, perm, v, shape=(_ACT_M, _ACT_K),
                                        transpose=transpose, backend=backend,
                                        workspace=_make_binary_task_workspace(indptr), weighted=True)

        def dense_op(w, z):
            D = _indexed_dense(w, indices, perm, rows)
            z = z.astype(w.dtype)
            return _hp_matmul(D.T, z) if transpose else _hp_matmul(D, z)

        check_event_op_grads(op, dense_op, weights, B, mode='weighted',
                             check_vmap=dtype_name == 'float32', **_weighted_tol(dtype_name))


def _heavy_indexed(seed, m, k, row_nnz, homo):
    rng = np.random.default_rng(seed)
    _, indices, indptr, _ = heavy_row_csr_numpy(rng, m, k, row_nnz)
    nse = indices.size
    perm = rng.permutation(nse).astype(np.int32)
    weights = np.array([1.5], np.float32) if homo else rng.standard_normal(nse).astype(np.float32)
    rows = np.repeat(np.arange(m), np.diff(indptr))
    dense = np.zeros((m, k))
    dense[rows, indices] = weights[0] if homo else weights[perm]
    return jnp.asarray(weights), jnp.asarray(indices), jnp.asarray(indptr), jnp.asarray(perm), dense


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
def test_indexed_mv_weighted_heavy_rows_use_row_value(backend, homo):
    """Multi-chunk hybrid rows re-read their own event value (indexed scatter)."""
    weights, indices, indptr, perm, dense = _heavy_indexed(7, 6, 9000, 8500, homo)
    x = jnp.array([2.0, 0.0, -1.0, 0.5, 3.0, 0.0], dtype=jnp.float32)
    got = binary_csrmv_indexed(weights, indices, indptr, perm, x, shape=dense.shape, transpose=True,
                               backend=backend, workspace=_make_binary_task_workspace(indptr), weighted=True)
    np.testing.assert_allclose(np.asarray(got, np.float64), np.asarray(x, np.float64) @ dense,
                               rtol=1e-4, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
def test_indexed_mm_weighted_heavy_rows_use_row_value(backend, homo):
    weights, indices, indptr, perm, dense = _heavy_indexed(8, 5, 9000, 8500, homo)
    B = jnp.array([[2.0, 0.0], [0.0, -1.0], [-1.0, 0.5], [0.5, 0.0], [0.0, 3.0]], dtype=jnp.float32)
    got = binary_csrmm_indexed(weights, indices, indptr, perm, B, shape=dense.shape, transpose=True,
                               backend=backend, workspace=_make_binary_task_workspace(indptr), weighted=True)
    np.testing.assert_allclose(np.asarray(got, np.float64), dense.T @ np.asarray(B, np.float64),
                               rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize('homo', [True, False])
def test_indexed_cuda_generators_select_weighted_abi(monkeypatch, homo):
    """Indexed CUDA kernels use the ``_wv`` symbols with events cast to the weight dtype."""
    ffi_calls = []
    monkeypatch.setattr(binary_indexed_mod, "load_cuda_file", lambda path, name, **kwargs: None)
    monkeypatch.setattr(binary_indexed_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    workspace = _make_binary_task_workspace(indptr)
    task_kwargs = {
        'task_begin_info': shape_of(workspace.task_begin.dtype, workspace.task_begin.shape),
        'task_end_info': shape_of(workspace.task_end.dtype, workspace.task_end.shape),
        'status_info': shape_of(workspace.status.dtype, workspace.status.shape),
        'task_capacity': workspace.task_capacity,
    }
    tail = (task_kwargs['task_begin_info'], task_kwargs['task_end_info'], task_kwargs['status_info'])
    n_w = 1 if homo else 2
    weights = jnp.ones(n_w, dtype=jnp.float32)
    if homo:
        names = [
            'csr_binary_csrmv.binary_csrmv_nt_auto_homo_f32_wv',
            'csr_binary_csrmv_hybrid.binary_csrmv_wat_hybrid_homo_f32_wv',
            'csr_binary_csrmm.binary_csrmm_nt_auto_homo_f32_wv',
            'csr_binary_csrmm_hybrid.binary_csrmm_sraw_hybrid_homo_f32_wv',
        ]
    else:
        names = [
            'csr_binary_indexed_csrmv.binary_csrmv_nt_auto_perm_hetero_f32_wv',
            'csr_binary_indexed_csrmv_hybrid.binary_indexed_csrmv_wat_hybrid_hetero_f32_wv',
            'csr_binary_indexed_csrmm.binary_csrmm_nt_auto_perm_hetero_f32_wv',
            'csr_binary_indexed_csrmm_hybrid.binary_indexed_csrmm_sraw_hybrid_hetero_f32_wv',
        ]
    cases = [
        (binary_indexed_mod._binary_csrmv_indexed_cuda_kernel, (2,), False, (1,)),
        (binary_indexed_mod._binary_csrmv_indexed_cuda_kernel, (1,), True, (2,)),
        (binary_indexed_mod._binary_csrmm_indexed_cuda_kernel, (2, 3), False, (1, 3)),
        (binary_indexed_mod._binary_csrmm_indexed_cuda_kernel, (1, 3), True, (2, 3)),
    ]
    for (factory, ev_shape, transpose, out_shape), expected in zip(cases, names):
        ffi_calls.clear()
        events = jnp.full(ev_shape, 4, dtype=jnp.uint8)
        kernel = factory(
            shape_of(jnp.float32, (n_w,)), shape_of(jnp.uint8, ev_shape), transpose,
            **{**cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int32),
               'outs': (shape_of(jnp.float32, out_shape), *tail), 'weighted': True, **task_kwargs},
        )
        kernel(weights, indices, indptr, perm, events, workspace.task_begin, workspace.task_end, workspace.status)
        assert len(ffi_calls) == 1
        name, _, _, args, _ = ffi_calls[0]
        assert strip_hybrid_suffix(name) == expected
        # Homogeneous weights reuse the plain kernels, which take no ``perm``.
        event_arg = args[3] if homo else args[4]
        assert event_arg.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(event_arg), np.full(ev_shape, 4.0))


# ---------------------------------------------------------------------------
# Inactive events contribute exactly zero, whatever the weight (NaN/Inf).
# docs/specs/2026-10-24-inactive-row-nan-isolation.md
# ---------------------------------------------------------------------------

from brainevent._csr._test_util import (  # noqa: E402
    EVENT_KINDS, NONFINITE_VALUES, assert_select_reference, homo_ct_case, inactive_nonfinite_case,
)


def _nonfinite_indexed_operands(case, seed=0):
    """Store the CSR-order weights of ``case`` permuted: ``data[perm[j]]`` is slot ``j``."""
    nse = case.indices.size
    if case.data.size == 1:
        return jnp.asarray(case.data), jnp.zeros(nse, jnp.int32)
    perm = np.random.default_rng(seed).permutation(nse).astype(np.int32)
    data = np.empty_like(case.data)
    data[perm] = case.data
    return jnp.asarray(data), jnp.asarray(perm)


def _indexed_call(case, shape, transpose, backend, mm, data=None):
    stored, perm = _nonfinite_indexed_operands(case)
    data = stored if data is None else data
    indptr = jnp.asarray(case.indptr)
    op = binary_csrmm_indexed if mm else binary_csrmv_indexed
    return op(
        data, jnp.asarray(case.indices), indptr, perm, case.events, shape=shape,
        workspace=_make_binary_task_workspace(indptr), transpose=transpose,
        backend=backend, weighted=case.weighted,
    )


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('bad', tuple(NONFINITE_VALUES))
def test_indexed_mv_inactive_nonfinite_weights_contribute_zero(backend, transpose, homo, kind, bad):
    shape = (7, 6)
    case = inactive_nonfinite_case(10, *shape, transpose=transpose, homo=homo, kind=kind, bad=bad)
    assert_select_reference(_indexed_call(case, shape, transpose, backend, mm=False), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('bad', tuple(NONFINITE_VALUES))
def test_indexed_mm_inactive_nonfinite_weights_contribute_zero(backend, transpose, homo, kind, bad):
    shape = (7, 6)
    case = inactive_nonfinite_case(
        11, *shape, transpose=transpose, homo=homo, kind=kind, bad=bad, n_batch=4,
    )
    assert_select_reference(_indexed_call(case, shape, transpose, backend, mm=True), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
def test_indexed_mm_nonfinite_weight_isolated_per_event_column(backend, transpose, kind):
    shape = (7, 6)
    case = inactive_nonfinite_case(
        12, *shape, transpose=transpose, homo=False, kind=kind, bad='nan', n_batch=4, partial=True,
    )
    assert_select_reference(_indexed_call(case, shape, transpose, backend, mm=True), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('homo', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('mm', [False, True])
def test_indexed_inactive_nonfinite_weights_heavy_rows(backend, homo, kind, mm):
    if mm and backend not in _IDX_MM_BACKENDS:
        pytest.skip(f'{backend} has no indexed csrmm kernel')
    shape = (4, 300)
    case = inactive_nonfinite_case(
        13, *shape, transpose=True, homo=homo, kind=kind, bad='nan',
        n_batch=3 if mm else None, heavy_row_nnz=200,
    )
    assert_select_reference(_indexed_call(case, shape, True, backend, mm=mm), case)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
def test_indexed_active_nan_weight_still_propagates(backend, transpose):
    shape = (7, 6)
    case = inactive_nonfinite_case(14, *shape, transpose=transpose, homo=False, kind='bool', bad='nan')
    e = int(np.flatnonzero(case.active[:, 0])[0])
    data = case.data.copy()
    data[e] = np.nan
    out = np.asarray(_indexed_call(case._replace(data=data), shape, transpose, backend, mm=False))
    target = case.indices[e] if transpose else int(np.searchsorted(case.indptr, e, side='right') - 1)
    assert np.isnan(out[target])
    assert np.isfinite(out).any()


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('mm', [False, True])
def test_indexed_weight_cotangent_of_inactive_entries_is_exact_zero(backend, transpose, kind, mm):
    if mm and backend not in _IDX_MM_BACKENDS:
        pytest.skip(f'{backend} has no indexed csrmm kernel')
    shape = (7, 6)
    case = inactive_nonfinite_case(
        15, *shape, transpose=transpose, homo=False, kind=kind, bad='nan',
        n_batch=3 if mm else None,
    )
    case = case._replace(data=np.where(np.isfinite(case.data), case.data, 1.0).astype(np.float32))
    stored, perm = _nonfinite_indexed_operands(case)

    def f(d):
        return _indexed_call(case, shape, transpose, backend, mm=mm, data=d)

    out, vjp = jax.vjp(f, stored)
    (ct_stored,) = vjp(jnp.full_like(out, jnp.nan))
    ct_slot = np.asarray(ct_stored)[np.asarray(perm)]
    inactive = case.inactive_edge
    assert inactive.any() and (~inactive).any()
    np.testing.assert_array_equal(ct_slot[inactive], 0.0)
    assert np.isnan(ct_slot[~inactive]).all()


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MV_BACKENDS)
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('kind', EVENT_KINDS)
@pytest.mark.parametrize('mm', [False, True])
def test_indexed_homo_weight_cotangent_ignores_nan_on_idle_outputs(backend, transpose, kind, mm):
    if mm and backend not in _IDX_MM_BACKENDS:
        pytest.skip(f'{backend} has no indexed csrmm kernel')
    shape = (7, 6)
    case = inactive_nonfinite_case(
        16, *shape, transpose=transpose, homo=True, kind=kind, bad='nan', n_batch=3 if mm else None,
    )
    ct, expected = homo_ct_case(case, transpose)

    def f(d):
        return _indexed_call(case, shape, transpose, backend, mm=mm, data=d)

    _, vjp = jax.vjp(f, jnp.ones(1, jnp.float32))
    (ct_w,) = vjp(ct)
    assert np.isfinite(expected)
    np.testing.assert_allclose(np.asarray(ct_w), [expected], rtol=1e-5)
