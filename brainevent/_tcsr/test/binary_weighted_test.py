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

"""Tests for weighted (``WeightedArray``) TCSR products (#201).

A weighted event multiplies each weight by its value: ``WeightedArray(x) @ W``
equals ``x @ dense(W)`` exactly, visiting only ``x != 0``. The weight
cotangent is value-weighted (the ``pp_prop`` rule) whatever
``backward_algorithm`` says; the event cotangent is the float op.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR as PlainCSR
from brainevent._event import BinaryArray, WeightedArray
from brainevent._tcsr import binary
from brainevent._tcsr.binary import binary_csrmm_p, binary_csrmv_p
from brainevent._test_util import check_event_op_grads

requires_gpu_backend = pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="requires a JAX GPU backend",
)

_WEIGHTED_DTYPES = [
    jnp.int8, jnp.uint8, jnp.int16, jnp.int32,
    jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64,
]
_SHAPE = (37, 41)
_BATCH = 4


@contextmanager
def _explicit_int64_allowed():
    """TCSR needs int64 row pointers; this also enables x64 (float64 events)."""
    previous_explicit = jax.config.jax_explicit_x64_dtypes
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_explicit_x64_dtypes", previous_explicit)


def _backend_params(primitive):
    backends = primitive.available_backends(jax.default_backend())
    if not backends:
        return [pytest.param(None, marks=pytest.mark.skip(reason="no backend"), id="none")]
    return [pytest.param(b, id=b) for b in backends]


def _weighted_events(shape, dtype, seed=0):
    """Events from ``[2, -1, 0, 0.5]`` (signed ints ``[2, -1, 0, 1]``, uint8 ``[2, 3, 0, 1]``)."""
    n = int(np.prod(shape))
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        base = np.asarray([2, 3, 0, 1])
    elif np.issubdtype(dtype, np.integer):
        base = np.asarray([2, -1, 0, 1])
    else:
        base = np.asarray([2.0, -1.0, 0.0, 0.5])
    values = np.resize(base, n)
    values = values[np.random.RandomState(seed).permutation(n)].reshape(shape)
    return jnp.asarray(values.astype(np.float64)).astype(dtype)


def _tcsr(binary_backend, homogeneous, *, shape=_SHAPE, heavy_rows=(), heavy_cols=(),
          backward_algorithm="bptt", dtype=np.float32):
    """TCSR with an empty row (and optional dense rows) plus its float32 dense matrix."""
    from brainevent._tcsr.main import TCSR

    rows, cols = shape
    rng = np.random.RandomState(3)
    mask = rng.rand(rows, cols) < 0.3
    mask[min(5, rows - 1)] = False
    for row in heavy_rows:
        mask[row] = True
    for col in heavy_cols:
        mask[:, col] = True
    indices = np.concatenate([np.flatnonzero(r) for r in mask]).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(mask.sum(1))]).astype(np.int64)
    if homogeneous:
        data = np.asarray([1.5], dtype=dtype)
    else:
        data = rng.uniform(-1.0, 1.0, indices.size).astype(dtype)
    source = PlainCSR(
        (jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr)),
        shape=shape,
    )
    return TCSR.from_sorted_csr(
        source,
        binary_backend=binary_backend,
        backward_algorithm=backward_algorithm,
    )


def _event_shape(op, shape=_SHAPE, batch=_BATCH):
    rows, cols = shape
    return {
        "mv": (cols,), "vm": (rows,),
        "mm": (cols, batch), "mr": (batch, rows),
    }[op]


def _apply(op, mat, events):
    return mat @ events if op in ("mv", "mm") else events @ mat


def _tolerance(dtype):
    # float16/bfloat16 event cotangents are rounded to the event dtype.
    if np.dtype(dtype) in (np.dtype(jnp.float16), np.dtype(jnp.bfloat16)):
        return 2e-2
    return 1e-4


def _check_weighted(binary_backend, homogeneous, op, dtype, **tcsr_kwargs):
    # float64 events run with float64 weights; other events are read in the
    # (float32) weight dtype, which is exact for the test values.
    weight_dtype = np.float64 if np.dtype(dtype) == np.float64 else np.float32
    with _explicit_int64_allowed():
        matrix = _tcsr(binary_backend, homogeneous, dtype=weight_dtype, **tcsr_kwargs)
        shape = matrix.shape
        x = _weighted_events(_event_shape(op, shape), dtype)

        def op_fn(data, v):
            return _apply(op, matrix.with_data(data), WeightedArray(v))

        def dense_fn(data, z):
            return _apply(op, matrix.with_data(data).todense(), z.astype(data.dtype))

        tol = _tolerance(dtype)
        check_event_op_grads(
            op_fn, dense_fn, matrix.data, x,
            mode="weighted", rtol=tol, atol=tol,
        )


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmv_p))
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["mv", "vm"])
@pytest.mark.parametrize("dtype", _WEIGHTED_DTYPES, ids=lambda d: np.dtype(d).name)
def test_tcsr_weighted_mv_matches_dense_with_exact_grads(binary_backend, homogeneous, op, dtype):
    _check_weighted(binary_backend, homogeneous, op, dtype)


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmm_p))
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["mm", "mr"])
@pytest.mark.parametrize("dtype", _WEIGHTED_DTYPES, ids=lambda d: np.dtype(d).name)
def test_tcsr_weighted_mm_matches_dense_with_exact_grads(binary_backend, homogeneous, op, dtype):
    _check_weighted(binary_backend, homogeneous, op, dtype)


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmm_p))
@pytest.mark.parametrize("backward_algorithm", ["bptt", "pp_prop"])
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
def test_tcsr_weighted_grads_ignore_backward_algorithm(binary_backend, backward_algorithm, op):
    """Weighted events always use the value-weighted weight cotangent."""
    _check_weighted(
        binary_backend, False, op, jnp.float32,
        backward_algorithm=backward_algorithm,
    )


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmm_p))
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
def test_tcsr_weighted_differs_from_binary_only_by_values(binary_backend, op):
    """The binary product of the same events stays mask-based."""
    with _explicit_int64_allowed():
        matrix = _tcsr(binary_backend, False)
        dense = np.asarray(matrix.todense())
        x = _weighted_events(_event_shape(op), jnp.float32)
        weighted = np.asarray(_apply(op, matrix, WeightedArray(x)))
        binary_out = np.asarray(_apply(op, matrix, BinaryArray(x)))
    xn = np.asarray(x)
    np.testing.assert_allclose(weighted, _apply(op, dense, xn), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        binary_out, _apply(op, dense, (xn != 0).astype(np.float32)), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmv_p))
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["mv", "vm"])
def test_tcsr_weighted_vmap_over_events_matches_dense(binary_backend, homogeneous, op):
    """``vmap`` of a weighted MV lowers to the weighted MM primitive."""
    with _explicit_int64_allowed():
        matrix = _tcsr(binary_backend, homogeneous)
        dense = np.asarray(matrix.todense())
        xs = _weighted_events((3,) + _event_shape(op), jnp.float32)
        got = jax.vmap(lambda v: _apply(op, matrix, WeightedArray(v)))(xs)
    expected = np.stack([_apply(op, dense, np.asarray(v)) for v in xs])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


def test_tcsr_weighted_rejects_non_numeric_events():
    with _explicit_int64_allowed():
        matrix = _tcsr("jax", False)
        with pytest.raises(TypeError, match="weighted events"):
            binary.binary_csrmv(
                matrix.data,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                jnp.ones((_SHAPE[0],), dtype=jnp.complex64),
                shape=_SHAPE,
                workspace=matrix._tcs_buffers.tcsr_workspace,
                local_targets=matrix._tcs_buffers.tcsr_local_targets,
                tile_offsets=matrix._tcs_buffers.tcsr_tile_offsets,
                transpose=True,
                backend="jax",
                weighted=True,
            )


def test_tcsr_weighted_inactive_nan_weight_does_not_propagate():
    """``0 * w`` terms are skipped: a NaN weight in an inactive row stays out."""
    with _explicit_int64_allowed():
        matrix = _tcsr("jax", False)
        data = np.asarray(matrix.data).copy()
        indptr = np.asarray(matrix._tcsr_indptr)
        x = np.zeros((_SHAPE[0],), np.float32)
        x[0] = 2.0
        data[indptr[1]:indptr[2]] = np.nan  # row 1 is inactive
        # cuSPARSE is a float SpMV of ``x``, so ``0 * NaN`` is not skipped there.
        backends = [
            b for b in binary_csrmv_p.available_backends(jax.default_backend())
            if b != "cusparse"
        ]
        for backend in backends:
            m = _tcsr(backend, False).with_data(jnp.asarray(data))
            out = np.asarray(WeightedArray(jnp.asarray(x)) @ m)
            assert np.all(np.isfinite(out)), backend


# ---------------------------------------------------------------------------
# CUDA routing and ABI (the FFI call is recorded, nothing is compiled).
# ---------------------------------------------------------------------------


def _recording_ffi_call(calls):
    def ffi_call(target, out_info, **ffi_kwargs):
        def invoke(*args, **attrs):
            calls.append((target, out_info, args))
            return tuple(jnp.zeros(i.shape, dtype=i.dtype) for i in out_info)

        return invoke

    return ffi_call


def _hybrid_kwargs(out_shape, out_dtype, *, mirror_enabled, shape=(2, 3)):
    task = jax.ShapeDtypeStruct((1,), jnp.int64)
    return dict(
        shape=shape,
        mirror_enabled=mirror_enabled,
        indices_info=jax.ShapeDtypeStruct((3,), jnp.int32),
        indptr_info=jax.ShapeDtypeStruct((shape[0] + 1,), jnp.int64),
        outs=(
            jax.ShapeDtypeStruct(out_shape, out_dtype),
            task,
            task,
            jax.ShapeDtypeStruct((2,), jnp.int32),
        ),
        task_capacity=1,
        weighted=True,
    )


def _invoke(kernel, weights, events, *, n_rows):
    return kernel(
        weights,
        jnp.asarray([0, 2, 1], dtype=jnp.int32),
        jnp.asarray([0, 2, 3, 3][: n_rows + 1], dtype=jnp.int64),
        events,
        jnp.zeros((1,), dtype=jnp.int64),
        jnp.zeros((1,), dtype=jnp.int64),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.asarray([0, 2, 1], dtype=jnp.uint16),
        jnp.zeros((n_rows, 2), dtype=jnp.int32),
        jnp.asarray([0, 1, 2], dtype=jnp.int64),
    )


@pytest.mark.parametrize("weight_dtype", [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.int8, jnp.float32, jnp.float64],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize(
    ("homogeneous", "mirror_enabled", "entry"),
    [
        (False, False, "binary_csrmv_wat_hybrid_hetero"),
        (True, False, "binary_csrmv_wat_hybrid_homo"),
        (True, True, "binary_csrmv_wat_hybrid_homo"),
        (False, True, "binary_indexed_csrmv_wat_hybrid_hetero"),
    ],
)
def test_weighted_cuda_mv_uses_wv_entry_with_weight_dtype_events(
    monkeypatch, weight_dtype, event_dtype, homogeneous, mirror_enabled, entry,
):
    calls = []
    monkeypatch.setattr(binary, "load_cuda_file", lambda *a, **k: None)
    monkeypatch.setattr(binary.jax.ffi, "ffi_call", _recording_ffi_call(calls))
    with _explicit_int64_allowed():
        n_weights = 1 if homogeneous else 3
        kernel = binary._binary_csrmv_cuda_kernel(
            jax.ShapeDtypeStruct((n_weights,), weight_dtype),
            jax.ShapeDtypeStruct((2,), event_dtype),
            not mirror_enabled,
            **_hybrid_kwargs((3,), weight_dtype, mirror_enabled=mirror_enabled),
        )
        events = jnp.asarray([2, 0], dtype=jnp.float64).astype(event_dtype)
        _invoke(kernel, jnp.ones((n_weights,), weight_dtype), events, n_rows=2)
    target, _, args = calls[0]
    suffix = {"float32": "_f32", "float64": "_f64", "float16": "_f16", "bfloat16": "_bf16"}[
        np.dtype(weight_dtype).name
    ]
    assert target.split(".", 1)[1] == f"{entry}{suffix}_wv"
    event_arg = args[4] if entry.startswith("binary_indexed") else args[3]
    assert event_arg.dtype == jnp.dtype(weight_dtype)
    np.testing.assert_array_equal(
        np.asarray(event_arg.astype(jnp.float32)), np.asarray(events.astype(jnp.float32))
    )


@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.int8, jnp.float16, jnp.float64],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize(
    ("homogeneous", "transpose", "entry"),
    [
        (False, True, "binary_csrmm_sraw_hybrid_hetero"),
        (True, True, "binary_csrmm_sraw_hybrid_homo"),
        (False, False, "binary_indexed_csrmm_sraw_hybrid_hetero"),
        (True, False, "binary_csrmm_sraw_hybrid_homo"),
    ],
)
def test_weighted_cuda_mm_routes_to_hybrid_not_tile(
    monkeypatch, event_dtype, homogeneous, transpose, entry,
):
    """Weighted MM never reaches the tile kernel (it only sees active rows)."""
    calls = []
    monkeypatch.setattr(binary, "load_cuda_file", lambda *a, **k: None)
    monkeypatch.setattr(binary.jax.ffi, "ffi_call", _recording_ffi_call(calls))
    with _explicit_int64_allowed():
        n_weights = 1 if homogeneous else 3
        batch = 5
        # transpose=True: BN events (batch, rows=2) -> (batch, cols=3).
        # transpose=False: NB events (rows=3, batch) on the mirror -> (2, batch).
        if transpose:
            event_shape, out_shape, n_rows = (batch, 2), (batch, 3), 2
        else:
            event_shape, out_shape, n_rows = (3, batch), (2, batch), 3
        kernel = binary._binary_csrmm_tile_cuda_kernel(
            jax.ShapeDtypeStruct((n_weights,), jnp.float32),
            jax.ShapeDtypeStruct(event_shape, event_dtype),
            transpose,
            **_hybrid_kwargs(out_shape, jnp.float32, mirror_enabled=not transpose),
        )
        events = _weighted_events(event_shape, event_dtype if event_dtype != jnp.bool_ else jnp.float32)
        events = events.astype(event_dtype)
        out = _invoke(kernel, jnp.ones((n_weights,), jnp.float32), events, n_rows=n_rows)
    target, out_info, args = calls[0]
    assert "tile" not in target
    assert target.split(".", 1)[1] == f"{entry}_f32_wv"
    events_nb = args[4]
    assert events_nb.dtype == jnp.float32
    expected_nb = np.asarray(events.astype(jnp.float32))
    np.testing.assert_array_equal(np.asarray(events_nb), expected_nb.T if transpose else expected_nb)
    assert out_info[0].shape == (batch, out_shape[1] if transpose else out_shape[0])
    assert out[0].shape == out_shape


def test_binary_cuda_paths_keep_binary_entries(monkeypatch):
    """``weighted=False`` keeps the original kernel symbols."""
    calls = []
    monkeypatch.setattr(binary, "load_cuda_file", lambda *a, **k: None)
    monkeypatch.setattr(binary.jax.ffi, "ffi_call", _recording_ffi_call(calls))
    with _explicit_int64_allowed():
        kwargs = _hybrid_kwargs((3,), jnp.float32, mirror_enabled=False)
        kwargs["weighted"] = False
        kernel = binary._binary_csrmv_cuda_kernel(
            jax.ShapeDtypeStruct((3,), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.int8),
            True,
            **kwargs,
        )
        _invoke(kernel, jnp.ones((3,), jnp.float32), jnp.asarray([2, 0], jnp.int8), n_rows=2)
    assert calls[0][0].endswith("binary_csrmv_wat_hybrid_hetero_f32_bool")
    assert calls[0][2][3].dtype == jnp.bool_


# ---------------------------------------------------------------------------
# GPU: heavy rows exercise the hybrid task (block) kernels, whose row lookup
# recovers each task's event value.
# ---------------------------------------------------------------------------


@requires_gpu_backend
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
@pytest.mark.parametrize("dtype", [jnp.int8, jnp.float16, jnp.float32], ids=lambda d: np.dtype(d).name)
def test_cuda_weighted_heavy_rows_match_dense(homogeneous, op, dtype):
    shape = (300, 9000)
    with _explicit_int64_allowed():
        # Rows 3, 7 and 200 are dense (9000 > 2 * 4096 slots, several tasks
        # on the canonical side); columns 11 and 4000 are dense (300 > 128
        # slots, tasks on the mirror side used by mv/mm); row 5 is empty.
        matrix = _tcsr(
            "cuda_raw", homogeneous, shape=shape,
            heavy_rows=(3, 7, 200), heavy_cols=(11, 4000),
        )
        x = _weighted_events(_event_shape(op, shape, batch=4), dtype, seed=1)
        got = np.asarray(_apply(op, matrix, WeightedArray(x)))
        dense = np.asarray(matrix.todense(), dtype=np.float64)
    expected = _apply(op, dense, np.asarray(x.astype(jnp.float32), dtype=np.float64))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


@requires_gpu_backend
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
def test_cuda_weighted_heavy_rows_grads(homogeneous, op):
    shape = (300, 9000)
    with _explicit_int64_allowed():
        matrix = _tcsr(
            "cuda_raw", homogeneous, shape=shape,
            heavy_rows=(3, 7, 200), heavy_cols=(11, 4000),
        )
        x = _weighted_events(_event_shape(op, shape, batch=4), jnp.float32, seed=2)

        def op_fn(data, v):
            return _apply(op, matrix.with_data(data), WeightedArray(v))

        def dense_fn(data, z):
            return _apply(op, matrix.with_data(data).todense(), z)

        check_event_op_grads(
            op_fn, dense_fn, matrix.data, x,
            mode="weighted", rtol=1e-3, atol=1e-3, check_vmap=False,
        )
