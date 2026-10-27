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

import warnings
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent._csr.main as csr_main
from brainevent._csr.binary import binary_csrmv, binary_csrmm
from brainevent._csr.binary_indexed import (
    _csrmm_idx_batching,
    _idx_batching,
    binary_csrmv_indexed,
    binary_csrmv_indexed_p,
    binary_csrmv_indexed_p_call,
    binary_csrmm_indexed,
    binary_csrmm_indexed_p,
    binary_csrmm_indexed_p_call,
)
import brainevent._csr.binary_indexed as binary_indexed_mod
from brainevent._csr.hybrid_config import get_hybrid_config, hybrid_task_capacity
from brainevent._csr._test_util import (
    cuda_kwargs, int64_structure, recording_ffi_call, requires_gpu_backend,
    shape_of, strip_hybrid_suffix,
)
from brainevent._test_util import jax_x64_enabled, with_jax_x64
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

        got = binary_csrmv_indexed(
            weights, indices, indptr, perm, events, shape=(2, 3), backend='jax_raw'
        )

        assert jnp.allclose(got, jnp.array([40.0, 20.0], dtype=jnp.float32))


def _assert_workspace_deprecated(call, expected):
    """Passing ``workspace`` warns at the caller on every call and changes nothing."""
    expected = np.asarray(expected)
    for _ in range(2):  # the second call hits the jit cache and must still warn
        with pytest.warns(DeprecationWarning, match='workspace') as record:
            got = call(object())
        assert any(
            w.filename == __file__ for w in record if issubclass(w.category, DeprecationWarning)
        ), 'the DeprecationWarning must point at the caller'
        assert got.dtype == expected.dtype
        np.testing.assert_allclose(np.asarray(got), expected)
    # The default (``None``) is silent.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        np.testing.assert_allclose(np.asarray(call(None)), expected)
    assert not [w for w in record if 'workspace' in str(w.message)]


def test_binary_csrmv_indexed_workspace_is_deprecated_and_ignored():
    weights = jnp.array([10.0, 20.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    vector = jnp.array([True, False])

    _assert_workspace_deprecated(
        lambda deprecated: binary_csrmv_indexed(
            weights, indices, indptr, perm, vector, shape=(1, 2), backend="jax_raw", workspace=deprecated,
        ),
        np.array([20.0], dtype=np.float32),
    )


def _heavy_row_structure():
    """2-row CSR whose first row is longer than ``tpr_threshold``, with a reversing ``perm``."""
    heavy = get_hybrid_config().tpr_threshold + 1
    indices = jnp.concatenate([jnp.arange(heavy, dtype=jnp.int32), jnp.zeros(1, jnp.int32)])
    indptr = jnp.array([0, heavy, heavy + 1], dtype=jnp.int32)
    perm = jnp.arange(heavy + 1, dtype=jnp.int32)[::-1]
    return indices, indptr, perm, heavy


def _kernel_eqn(primitive, fn, *args):
    """The only ``primitive`` equation in the jaxpr of ``fn(*args)``."""
    # Match by name: reloading the module (some tests do) rebinds the primitive object.
    eqns = [e for e in jax.make_jaxpr(fn)(*args).jaxpr.eqns if e.primitive.name == primitive.primitive.name]
    assert len(eqns) == 1
    return eqns[0]


def _assert_task_capacity_param(eqn, indptr, max_row_nnz):
    params = eqn.params
    assert not {'task_begin_info', 'task_end_info', 'status_info'} & params.keys()
    assert params['max_row_nnz'] == max_row_nnz
    assert type(params['task_capacity']) is int
    assert params['task_capacity'] >= hybrid_task_capacity(indptr) > 0


@pytest.mark.parametrize('hint', [False, True])
def test_binary_csrmv_indexed_p_call_returns_one_output_and_static_task_capacity(hint):
    indices, indptr, perm, heavy = _heavy_row_structure()
    max_row_nnz = heavy if hint else None
    weights = jnp.arange(1, heavy + 2, dtype=jnp.float32)
    vector = jnp.array([True, True])

    def call(w, v):
        return binary_csrmv_indexed_p_call(w, indices, indptr, perm, v, shape=(2, heavy), transpose=True,
                                           backend="jax_raw", max_row_nnz=max_row_nnz)

    out = call(weights, vector)
    assert len(out) == 1
    ref = binary_csrmv(weights[perm], indices, indptr, vector, shape=(2, heavy), transpose=True,
                       backend="jax_raw")
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(ref))

    eqn = _kernel_eqn(binary_csrmv_indexed_p, call, weights, vector)
    assert (len(eqn.invars), len(eqn.outvars)) == (5, 1)
    _assert_task_capacity_param(eqn, indptr, max_row_nnz)


def test_binary_csrmv_indexed_batching_folds_vectors_into_matmat():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    vectors = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)

    out, out_dims = _idx_batching(
        (weights, indices, indptr, perm, vectors),
        (None, None, None, None, 0),
        outs=[jax.ShapeDtypeStruct((1,), weights.dtype)],
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vectors.shape[1:], vectors.dtype),
        max_row_nnz=None,
        task_capacity=binary_indexed_mod._task_capacity(indices, indptr, None),
    )

    assert len(out) == 1
    assert out[0].shape == (1, 2)
    # weights[perm] == [2, 1]: batch 0 hits column 0, batch 1 hits column 1.
    np.testing.assert_allclose(np.asarray(out[0]), [[2.0, 1.0]])
    assert list(out_dims) == [1]


def test_binary_csrmv_indexed_vmap_matches_materialized_weights():
    weights = jnp.array([1.0, 2.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    vectors = jnp.array([[True, False], [False, True]], dtype=jnp.bool_)

    got = jax.vmap(
        lambda vector: binary_csrmv_indexed(
            weights,
            indices,
            indptr,
            perm,
            vector,
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
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
        )
    )(vectors)

    assert got.shape == (2, 1)
    assert jnp.allclose(got, ref)


def test_binary_csrmm_indexed_workspace_is_deprecated_and_ignored():
    weights = jnp.array([10.0, 20.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    matrix = jnp.array([[True, False], [False, True]])

    _assert_workspace_deprecated(
        lambda deprecated: binary_csrmm_indexed(
            weights, indices, indptr, perm, matrix, shape=(1, 2), backend="jax_raw", workspace=deprecated,
        ),
        np.array([[20.0, 10.0]], dtype=np.float32),
    )


@pytest.mark.parametrize('hint', [False, True])
def test_binary_csrmm_indexed_p_call_returns_one_output_and_static_task_capacity(hint):
    indices, indptr, perm, heavy = _heavy_row_structure()
    max_row_nnz = heavy if hint else None
    weights = jnp.arange(1, heavy + 2, dtype=jnp.float32)
    matrix = jnp.array([[True, False, True], [True, True, False]])

    def call(w, b):
        return binary_csrmm_indexed_p_call(w, indices, indptr, perm, b, shape=(2, heavy), transpose=True,
                                           backend="jax_raw", max_row_nnz=max_row_nnz)

    out = call(weights, matrix)
    assert len(out) == 1
    ref = binary_csrmm(weights[perm], indices, indptr, matrix, shape=(2, heavy), transpose=True,
                       backend="jax_raw")
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(ref))

    eqn = _kernel_eqn(binary_csrmm_indexed_p, call, weights, matrix)
    assert (len(eqn.invars), len(eqn.outvars)) == (5, 1)
    _assert_task_capacity_param(eqn, indptr, max_row_nnz)


@pytest.mark.parametrize(
    ("batch_axis", "input_shape", "output_shape", "output_axis"),
    [
        (0, (2, 2, 2), (1, 2, 2), 1),
        (1, (2, 2, 2), (1, 2, 2), 1),
        (2, (2, 2, 2), (1, 2, 2), 2),
    ],
)
def test_binary_csrmm_indexed_batching_folds_batch_into_columns(
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

    out, out_dims = _csrmm_idx_batching(
        (weights, indices, indptr, perm, matrices),
        (None, None, None, None, batch_axis),
        outs=[jax.ShapeDtypeStruct((1, 2), weights.dtype)],
        shape=(1, 2),
        transpose=False,
        backend="jax_raw",
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        perm_info=jax.ShapeDtypeStruct(perm.shape, perm.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(matrices.shape[1:], matrices.dtype),
        max_row_nnz=None,
        task_capacity=binary_indexed_mod._task_capacity(indices, indptr, None),
    )

    assert len(out) == 1
    assert out[0].shape == output_shape
    assert list(out_dims) == [output_axis]
    # Every batch slice equals the unbatched product.
    for b in range(input_shape[batch_axis]):
        ref = binary_csrmm(weights[perm], indices, indptr, jnp.take(matrices, b, axis=batch_axis),
                           shape=(1, 2), backend="jax_raw")
        np.testing.assert_allclose(np.asarray(jnp.take(out[0], b, axis=output_axis)), np.asarray(ref))


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

    got = jax.vmap(
        lambda matrix: binary_csrmm_indexed(
            weights,
            indices,
            indptr,
            perm,
            matrix,
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
            shape=(1, 2),
            transpose=False,
            backend="jax_raw",
        )
    )(matrices)

    assert got.shape == (2, 1, 2)
    assert jnp.allclose(got, ref)


def test_binary_csrmm_indexed_weight_jvp_has_single_tangent():
    weights = jnp.array([10.0, 20.0], dtype=jnp.float32)
    indices = jnp.array([0, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2], dtype=jnp.int32)
    perm = jnp.array([1, 0], dtype=jnp.int32)
    matrix = jnp.array([[True, False], [False, True]])

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
        )

    primals, tangents = jax.jvp(call, (weights,), (jnp.ones_like(weights),))

    assert len(primals) == len(tangents) == 1
    assert jnp.allclose(primals[0], jnp.array([[20.0, 10.0]], dtype=jnp.float32))
    assert jnp.allclose(tangents[0], jnp.array([[1.0, 1.0]], dtype=jnp.float32))


def test_csr_unfavorable_binary_matvec_uses_direct_route_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSR.fromdense(dense, backend="jax_raw")
    events = jnp.array([True, False])
    buffer_keys = set(m.buffers)

    got = m @ BinaryArray(events)
    got_again = m @ BinaryArray(events)

    assert got.shape == (2,)
    assert jnp.allclose(got, dense @ events.astype(jnp.float32))
    assert jnp.allclose(got_again, got)
    assert m.buffers.get("csc") is None  # no mirror on jax_raw
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csr_unfavorable_binary_matmat_uses_direct_route_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSR.fromdense(dense, backend="jax_raw")
    events = jnp.array([[True, False], [False, True]])
    buffer_keys = set(m.buffers)

    got = m @ BinaryArray(events)
    got_again = m @ BinaryArray(events)

    assert got.shape == (2, 2)
    assert jnp.allclose(got, dense @ events.astype(jnp.float32))
    assert jnp.allclose(got_again, got)
    assert m.buffers.get("csc") is None  # no mirror on jax_raw
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csr_direct_binary_rmatmat_leaves_buffers_untouched():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSR.fromdense(dense, backend="jax_raw")
    events = jnp.array([[True, False], [False, True]])
    buffer_keys = set(m.buffers)

    got = BinaryArray(events) @ m
    got_again = BinaryArray(events) @ m

    assert got.shape == (2, 2)
    assert jnp.allclose(got, events.astype(jnp.float32) @ dense)
    assert jnp.allclose(got_again, got)
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csc_unfavorable_binary_rmatvec_uses_direct_route_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSC.fromdense(dense, backend="jax_raw")
    events = jnp.array([True, False])
    buffer_keys = set(m.buffers)

    got = BinaryArray(events) @ m
    got_again = BinaryArray(events) @ m

    assert got.shape == (2,)
    assert jnp.allclose(got, events.astype(jnp.float32) @ dense)
    assert jnp.allclose(got_again, got)
    assert m.buffers.get("csr") is None  # no mirror on jax_raw
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csc_direct_binary_matmat_leaves_buffers_untouched():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSC.fromdense(dense, backend="jax_raw")
    events = jnp.array([[True, False], [False, True]])
    buffer_keys = set(m.buffers)

    got = m @ BinaryArray(events)
    got_again = m @ BinaryArray(events)

    assert got.shape == (2, 2)
    assert jnp.allclose(got, dense @ events.astype(jnp.float32))
    assert jnp.allclose(got_again, got)
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_csc_unfavorable_binary_rmatmat_uses_direct_route_on_jax_raw():
    dense = jnp.array([[0.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    m = CSC.fromdense(dense, backend="jax_raw")
    events = jnp.array([[True, False], [False, True]])
    buffer_keys = set(m.buffers)

    got = BinaryArray(events) @ m
    got_again = BinaryArray(events) @ m

    assert got.shape == (2, 2)
    assert jnp.allclose(got, events.astype(jnp.float32) @ dense)
    assert jnp.allclose(got_again, got)
    assert m.buffers.get("csr") is None  # no mirror on jax_raw
    assert set(m.buffers) == buffer_keys  # nothing is mounted on the matrix


def test_cuda_raw_unfavorable_binary_routes_use_indexed_mirror(monkeypatch):
    dense = jnp.array([[0.0, 2.0, 5.0], [3.0, 4.0, 0.0]], dtype=jnp.float32)
    csr = CSR.fromdense(dense, backend="cuda_raw")
    csc = CSC.fromdense(dense, backend="cuda_raw")
    calls = []

    def fail_direct(*args, **kwargs):
        raise AssertionError("cuda_raw unfavorable BinaryArray route must use indexed primitive")

    # No ``workspace`` parameter: the routes must not pass the deprecated argument.
    def fake_mv_indexed(data, indices, indptr, perm, vector, *, shape, transpose, backend=None, weighted=False,
                        max_row_nnz=None):
        calls.append({
            "kind": "mv",
            "shape": shape,
            "transpose": transpose,
            "backend": backend,
            "indptr": indptr,
            "perm": perm,
        })
        out_size = shape[1] if transpose else shape[0]
        return jnp.zeros((out_size,), dtype=data.dtype)

    def fake_mm_indexed(data, indices, indptr, perm, matrix, *, shape, transpose, backend=None, weighted=False,
                        max_row_nnz=None):
        calls.append({
            "kind": "mm",
            "shape": shape,
            "transpose": transpose,
            "backend": backend,
            "indptr": indptr,
            "perm": perm,
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

    assert [(call["kind"], call["shape"], call["transpose"], call["backend"]) for call in calls] == [
        ("mv", (3, 2), True, "cuda_raw"),
        ("mm", (3, 2), True, "cuda_raw"),
        ("mv", (2, 3), True, "cuda_raw"),
        ("mm", (2, 3), True, "cuda_raw"),
    ]
    # Each route traverses the matrix's cached mirror structure.
    csr_mirror = csr._weight_indices()
    csc_mirror = csc._weight_indices()
    for call, mirror in zip(calls, (csr_mirror, csr_mirror, csc_mirror, csc_mirror)):
        assert call["indptr"] is mirror[0]
        assert call["perm"] is mirror[2]


def _spy_on(monkeypatch, module, name):
    """Wrap ``module.name``; return the list of ``(args, kwargs)`` it gets called with."""
    calls = []
    real = getattr(module, name)

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


def _assert_fcn_indexed_calls(calls, conn, n_calls):
    # Every call reads the cached CSC mirror and passes no ``workspace``.
    csc_indptr, csc_indices, perm = conn.buffers["csc"]
    assert len(calls) == n_calls
    for args, kwargs in calls:
        assert args[1] is csc_indices and args[2] is csc_indptr and args[3] is perm
        assert "workspace" not in kwargs


def test_fcn_unfavorable_binary_matvec_uses_indexed_route(monkeypatch):
    import brainevent._fcn.main as fcn_main
    data = jnp.array([[1.0, 2.0], [3.0, 0.0]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    conn = FixedNumPerPre((data, indices), shape=(2, 2), backend="jax_raw").build_weight_indices()
    vector = jnp.array([False, True])
    expected = conn.todense() @ vector.astype(jnp.float32)
    calls = _spy_on(monkeypatch, fcn_main, "binary_csrmv_indexed")

    got = conn @ BinaryArray(vector)
    got_again = conn @ BinaryArray(vector)

    assert got.shape == (2,)
    assert jnp.allclose(got, expected)
    assert jnp.allclose(got_again, got)
    _assert_fcn_indexed_calls(calls, conn, 2)


def test_fcn_unfavorable_binary_matmat_uses_indexed_route(monkeypatch):
    import brainevent._fcn.main as fcn_main
    data = jnp.array([[1.0, 2.0], [3.0, 0.0]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    conn = FixedNumPerPre((data, indices), shape=(2, 2), backend="jax_raw").build_weight_indices()
    matrix = jnp.array([[False, True], [True, False]])
    expected = conn.todense() @ matrix.astype(jnp.float32)
    calls = _spy_on(monkeypatch, fcn_main, "binary_csrmm_indexed")

    got = conn @ BinaryArray(matrix)
    got_again = conn @ BinaryArray(matrix)

    assert got.shape == (2, 2)
    assert jnp.allclose(got, expected)
    assert jnp.allclose(got_again, got)
    _assert_fcn_indexed_calls(calls, conn, 2)


def test_indexed_mv_rejects_int64_indices_with_int32_indptr():
    with _jax_x64_enabled():
        weights = jnp.ones(2, dtype=jnp.float32)
        indices = jnp.array([0, 1], dtype=jnp.int64)
        indptr = jnp.array([0, 2], dtype=jnp.int32)
        perm = jnp.array([0, 1], dtype=jnp.int32)
        events = jnp.ones(2, dtype=bool)

        with pytest.raises(AssertionError, match="Indices must be int32"):
            binary_csrmv_indexed(
                weights, indices, indptr, perm, events, shape=(1, 2), backend='jax_raw'
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

    with pytest.raises(AssertionError, match="Indices must be int32"):
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
    got = binary_csrmv_indexed(
        weights, indices, indptr, perm, v, shape=(m, k), transpose=transpose
    )
    ref_w = weights if homo else weights[perm]
    ref = binary_csrmv(ref_w, indices, indptr, v, shape=(m, k), transpose=transpose)
    assert jnp.allclose(got, ref, atol=1e-5), (transpose, homo, ev)


def test_indexed_jit():
    rng = np.random.default_rng(5)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.bool_)
    f = jax.jit(lambda w: binary_csrmv_indexed(w, indices, indptr, perm, v, shape=(m, k)))
    got = f(weights)
    ref = binary_csrmv(weights[perm], indices, indptr, v, shape=(m, k))
    assert jnp.allclose(got, ref, atol=1e-5)


def test_indexed_grad_weights():
    rng = np.random.default_rng(1)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.float32)
    f = lambda w: binary_csrmv_indexed(
        w, indices, indptr, perm, v, shape=(m, k), transpose=False
    ).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmv(
        w[perm], indices, indptr, v, shape=(m, k), transpose=False
    ).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5)


def test_indexed_grad_weights_transpose():
    rng = np.random.default_rng(11)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(m) > 0.5, dtype=jnp.float32)
    f = lambda w: binary_csrmv_indexed(
        w, indices, indptr, perm, v, shape=(m, k), transpose=True
    ).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmv(
        w[perm], indices, indptr, v, shape=(m, k), transpose=True
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
        f = lambda w: binary_csrmv_indexed(w, indices, indptr, perm, v, shape=(m, k)).sum()
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
    got = binary_csrmm_indexed(
        weights, indices, indptr, perm, B, shape=(m, k), transpose=transpose
    )
    ref_w = weights if homo else weights[perm]
    ref = binary_csrmm(ref_w, indices, indptr, B, shape=(m, k), transpose=transpose)
    assert jnp.allclose(got, ref, atol=1e-5), (transpose, homo, ev, n)


def test_indexed_mm_single_column_equals_mv():
    rng = np.random.default_rng(3)
    m, k, nse = 4, 5, 9
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray(rng.random(nse), jnp.float32)
    v = jnp.asarray(rng.random(k) > 0.5, dtype=jnp.bool_)
    got = binary_csrmm_indexed(
        weights, indices, indptr, perm, v[:, None], shape=(m, k), transpose=False
    )
    ref = binary_csrmv_indexed(
        weights, indices, indptr, perm, v, shape=(m, k), transpose=False
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
    f = lambda w: binary_csrmm_indexed(
        w, indices, indptr, perm, B, shape=(m, k), transpose=transpose
    ).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmm(
        w[perm], indices, indptr, B, shape=(m, k), transpose=transpose
    ).sum()
    gref = jax.grad(fref)(weights)
    assert jnp.allclose(g, gref, atol=1e-5), transpose


def test_indexed_mm_grad_weights_homo():
    rng = np.random.default_rng(8)
    m, k, nse, n = 4, 5, 9, 3
    indices, indptr, perm = _structure(rng, m, k, nse)
    weights = jnp.asarray([0.7], jnp.float32)
    B = jnp.asarray(rng.random((k, n)) > 0.5, dtype=jnp.float32)
    f = lambda w: binary_csrmm_indexed(
        w, indices, indptr, perm, B, shape=(m, k), transpose=False
    ).sum()
    g = jax.grad(f)(weights)
    fref = lambda w: binary_csrmm(
        w, indices, indptr, B, shape=(m, k), transpose=False
    ).sum()
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


def _assert_hybrid_scratch_call(call, n_math, task_capacity, indptr_dtype):
    """A hybrid FFI call gets the math operands plus three scratch buffers made in the lowering."""
    _, out_info, ffi_kwargs, args, kwargs = call
    scratch = [((task_capacity,), indptr_dtype), ((task_capacity,), indptr_dtype), ((2,), jnp.int32)]
    assert len(args) == n_math + len(scratch)
    assert ffi_kwargs['input_output_aliases'] == {n_math + i: 1 + i for i in range(len(scratch))}
    assert kwargs == {'task_capacity': task_capacity}
    assert [(a.shape, a.dtype) for a in args[n_math:]] == [(s, jnp.dtype(d)) for s, d in scratch]
    assert [(o.shape, o.dtype) for o in out_info[1:]] == [(s, jnp.dtype(d)) for s, d in scratch]


def test_binary_indexed_cuda_generators_accept_int64_indptr_without_real_cuda(monkeypatch):
    ffi_calls = []
    load_calls = []

    monkeypatch.setattr(binary_indexed_mod, "load_cuda_file", lambda path, name, **kwargs: load_calls.append((path, name, kwargs)))
    monkeypatch.setattr(binary_indexed_mod.jax.ffi, "ffi_call", recording_ffi_call(ffi_calls))

    outs = []
    with jax_x64_enabled():
        indices = jnp.array([0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2], dtype=jnp.int64)
        perm = jnp.array([1, 0], dtype=jnp.int32)
        task_capacity = binary_indexed_mod._task_capacity(indices, indptr, None)
        common = {**cuda_kwargs(indices_dtype=jnp.int32, indptr_dtype=jnp.int64), 'task_capacity': task_capacity}
        mv_outs = [shape_of(jnp.float32)]
        mm_nt_outs = [shape_of(jnp.float32, (1, 1))]
        mm_t_outs = [shape_of(jnp.float32, (2, 1))]

        mv_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (2,)),
            False,
            **{**common, 'perm_info': shape_of(jnp.int32, (2,))},
        )
        outs.append(mv_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32), indices, indptr, perm, jnp.array([True, False]),
        ))

        mv_t_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (1,)),
            True,
            **{**common, 'perm_info': shape_of(jnp.int64, (2,)), 'outs': mv_outs},
        )
        outs.append(mv_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32), indices, indptr, perm.astype(jnp.int64), jnp.array([True]),
        ))

        mv_t_homo_kernel = binary_indexed_mod._binary_csrmv_indexed_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (1,)),
            True,
            **{**common, 'perm_info': shape_of(jnp.int64, (2,)), 'outs': mv_outs},
        )
        outs.append(mv_t_homo_kernel(
            jnp.array([2.0], dtype=jnp.float32), indices, indptr, perm.astype(jnp.int64), jnp.array([True]),
        ))

        mm_nt_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (2, 1)),
            False,
            **{**common, 'outs': mm_nt_outs, 'perm_info': shape_of(jnp.int32, (2,))},
        )
        outs.append(mm_nt_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32), indices, indptr, perm, jnp.array([[True], [False]]),
        ))

        mm_t_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            shape_of(jnp.float32, (2,)),
            shape_of(jnp.bool_, (1, 1)),
            True,
            **{**common, 'outs': mm_t_outs, 'perm_info': shape_of(jnp.int64, (2,))},
        )
        outs.append(mm_t_kernel(
            jnp.array([2.0, 3.0], dtype=jnp.float32), indices, indptr, perm.astype(jnp.int64), jnp.array([[True]]),
        ))

        mm_t_homo_kernel = binary_indexed_mod._binary_csrmm_indexed_cuda_kernel(
            shape_of(jnp.float32, (1,)),
            shape_of(jnp.bool_, (1, 1)),
            True,
            **{**common, 'outs': mm_t_outs, 'perm_info': shape_of(jnp.int64, (2,))},
        )
        outs.append(mm_t_homo_kernel(
            jnp.array([2.0], dtype=jnp.float32), indices, indptr, perm.astype(jnp.int64), jnp.array([[True]]),
        ))

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
    # Kernels return only the math output, in the logical layout.
    assert [len(out) for out in outs] == [1] * 6
    assert [out[0].shape for out in outs] == [(2,), (2,), (2,), (1, 1), (2, 1), (2, 1)]
    # The direct perm kernels take five math operands. Hybrid kernels append
    # their own scratch; the homogeneous ones drop ``perm`` (four operands).
    assert len(ffi_calls[0][3]) == 5 and len(ffi_calls[3][3]) == 5
    for i, n_math in ((1, 5), (2, 4), (4, 5), (5, 4)):
        _assert_hybrid_scratch_call(ffi_calls[i], n_math, task_capacity, jnp.int64)


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@with_jax_x64
def test_binary_indexed_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    vector = jnp.array([True, False], dtype=jnp.bool_) if transpose else jnp.array([True, False, True])

    got = binary_csrmv_indexed(weights, indices, indptr64, perm, vector, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw')
    expected = binary_csrmv_indexed(weights, indices, indptr32, perm, vector, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


@requires_gpu_backend
@pytest.mark.parametrize('transpose', [False, True])
@with_jax_x64
def test_binary_indexed_csrmm_cuda_accepts_int64_indptr(transpose):
    weights, indices, indptr32 = int64_structure(jnp.int32)
    indptr64 = indptr32.astype(jnp.int64)
    perm = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    matrix = (
        jnp.array([[True, False], [False, True]], dtype=jnp.bool_)
        if transpose else
        jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    )

    got = binary_csrmm_indexed(weights, indices, indptr64, perm, matrix, shape=(2, 3),
                               transpose=transpose, backend='cuda_raw')
    expected = binary_csrmm_indexed(weights, indices, indptr32, perm, matrix, shape=(2, 3),
                                    transpose=transpose, backend='jax_raw')

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Event activity contract (``x != 0``), CUDA event ABI and #198 weight
# gradients for the indexed (perm-gather) products.
# ---------------------------------------------------------------------------

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
            backend=backend,
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
            backend=backend,
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
                                    transpose=transpose, backend=backend)

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
                                    transpose=transpose, backend=backend)

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
                                        weighted=True)

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
                                        weighted=True)

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
                               backend=backend, weighted=True)
    np.testing.assert_allclose(np.asarray(got, np.float64), np.asarray(x, np.float64) @ dense,
                               rtol=1e-4, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize('backend', _IDX_MM_BACKENDS)
@pytest.mark.parametrize('homo', [True, False])
def test_indexed_mm_weighted_heavy_rows_use_row_value(backend, homo):
    weights, indices, indptr, perm, dense = _heavy_indexed(8, 5, 9000, 8500, homo)
    B = jnp.array([[2.0, 0.0], [0.0, -1.0], [-1.0, 0.5], [0.5, 0.0], [0.0, 3.0]], dtype=jnp.float32)
    got = binary_csrmm_indexed(weights, indices, indptr, perm, B, shape=dense.shape, transpose=True,
                               backend=backend, weighted=True)
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
    task_capacity = binary_indexed_mod._task_capacity(indices, indptr, None)
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
               'outs': [shape_of(jnp.float32, out_shape)], 'weighted': True, 'task_capacity': task_capacity},
        )
        kernel(weights, indices, indptr, perm, events)
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
        transpose=transpose,
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
