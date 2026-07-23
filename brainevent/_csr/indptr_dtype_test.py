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

"""Tests for the ``indptr`` int64 auto-precision + ``indices`` always-int32 policy.

Covers the unified dtype contract introduced for CSR/CSC structures:

* ``indices`` are secondary-axis coordinates and are *always* int32; int64 (or
  out-of-range) coordinates raise rather than widen.
* ``indptr``/offset arrays default to int32 and auto-promote to int64 only when
  ``nnz`` exceeds the int32 range. Creating an int64 offset array requires
  ``jax_enable_x64``; the library raises instead of toggling the global config.

Because materialising a ``nnz > int32_max`` array is impractical, the promotion
threshold and the x64 gating are exercised at the helper/unit level and via an
explicit ``indptr_dtype=int64`` request, while structural and coercion behaviour
is exercised end-to-end through the public constructors.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
import brainevent._csr.diag_add as diag_add_mod
from brainevent._misc import (
    _INT32_MAX,
    _resolve_indptr_dtype,
    _require_jax_x64_for_int64,
    _as_int32_indices,
    _as_indptr,
    _as_int32_cuda_offsets,
    _check_compressed_structure,
)

CSR = brainevent.CSR
CSC = brainevent.CSC


@contextmanager
def _jax_x64_enabled():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def _small_csr():
    #  [[1, 0, 2],
    #   [0, 3, 0]]
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    return CSR((data, indices, indptr), shape=(2, 3))


# ---------------------------------------------------------------------------
# _resolve_indptr_dtype -- auto promotion threshold + explicit requests
# ---------------------------------------------------------------------------

def test_resolve_auto_picks_int32_within_range():
    assert _resolve_indptr_dtype(0, "auto") == np.dtype(np.int32)
    assert _resolve_indptr_dtype(1_000, "auto") == np.dtype(np.int32)
    assert _resolve_indptr_dtype(_INT32_MAX, "auto") == np.dtype(np.int32)


def test_resolve_auto_picks_int64_above_range():
    assert _resolve_indptr_dtype(_INT32_MAX + 1, "auto") == np.dtype(np.int64)
    assert _resolve_indptr_dtype(10 * _INT32_MAX, "auto") == np.dtype(np.int64)


def test_resolve_explicit_int32_overflows():
    with pytest.raises(OverflowError, match="exceeds the int32 range"):
        _resolve_indptr_dtype(_INT32_MAX + 1, np.int32)


def test_resolve_explicit_int32_within_range_ok():
    assert _resolve_indptr_dtype(_INT32_MAX, np.int32) == np.dtype(np.int32)


def test_resolve_explicit_int64_honoured():
    # dtype resolution does not gate on x64; gating is a separate step.
    assert _resolve_indptr_dtype(10, np.int64) == np.dtype(np.int64)


def test_resolve_rejects_unknown_string():
    with pytest.raises(ValueError, match="indptr_dtype must be"):
        _resolve_indptr_dtype(10, "float")


# ---------------------------------------------------------------------------
# _require_jax_x64_for_int64 -- gating without mutating global config
# ---------------------------------------------------------------------------

def test_require_x64_raises_for_int64_when_disabled():
    assert jax.config.jax_enable_x64 is False
    with pytest.raises(ValueError, match="requires an int64 array"):
        _require_jax_x64_for_int64(np.int64, "test context")
    # The gate must never toggle the global config.
    assert jax.config.jax_enable_x64 is False


def test_require_x64_allows_int32_when_disabled():
    # int32 never needs x64.
    _require_jax_x64_for_int64(np.int32, "test context")


def test_require_x64_allows_int64_when_enabled():
    with _jax_x64_enabled():
        _require_jax_x64_for_int64(np.int64, "test context")


# ---------------------------------------------------------------------------
# _as_int32_indices -- indices are always int32
# ---------------------------------------------------------------------------

def test_indices_int32_passthrough():
    idx = jnp.array([0, 1, 2], dtype=jnp.int32)
    out = _as_int32_indices(idx, 3, "ctx")
    assert out.dtype == jnp.int32


def test_indices_int64_coerced_when_in_range():
    idx = np.array([0, 1, 2], dtype=np.int64)
    out = _as_int32_indices(idx, 3, "ctx")
    assert out.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(out), [0, 1, 2])


def test_indices_negative_raises():
    idx = np.array([0, -1, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="must be non-negative"):
        _as_int32_indices(idx, 3, "ctx")


def test_indices_out_of_bounds_raises():
    idx = np.array([0, 5, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="out of bounds"):
        _as_int32_indices(idx, 3, "ctx")


def test_indices_non_integer_raises():
    idx = np.array([0.0, 1.0], dtype=np.float32)
    with pytest.raises(TypeError, match="must be an integer array"):
        _as_int32_indices(idx, 3, "ctx")


def test_indices_secondary_dim_beyond_int32_raises():
    idx = np.array([0, 1], dtype=np.int32)
    with pytest.raises(OverflowError, match="int32-representable"):
        _as_int32_indices(idx, _INT32_MAX + 2, "ctx")


def test_indices_traced_int64_rejected():
    def f(idx):
        return _as_int32_indices(idx, 3, "ctx")

    # An int64 tracer only exists when x64 is enabled (otherwise the input is
    # truncated to int32 before tracing).
    with _jax_x64_enabled():
        with pytest.raises(TypeError, match="traced int64 array"):
            jax.jit(f)(jnp.array([0, 1, 2], dtype=jnp.int64))


def test_indices_traced_int32_ok_no_host_readback():
    # Under a tracer only the static dtype is checked; no host value readback.
    def f(idx):
        out = _as_int32_indices(idx, 3, "ctx")
        return out.sum()

    val = jax.jit(f)(jnp.array([0, 1, 2], dtype=jnp.int32))
    assert int(val) == 3


# ---------------------------------------------------------------------------
# _as_indptr -- resolves dtype + gates int64
# ---------------------------------------------------------------------------

def test_as_indptr_small_is_int32():
    ptr = np.array([0, 2, 3], dtype=np.int64)
    out = _as_indptr(ptr, 3, "auto", "ctx")
    assert out.dtype == jnp.int32


def test_as_indptr_explicit_int64_gated_when_x64_off():
    ptr = np.array([0, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="requires an int64 array"):
        _as_indptr(ptr, 3, np.int64, "ctx")


def test_as_indptr_explicit_int64_ok_when_x64_on():
    with _jax_x64_enabled():
        ptr = np.array([0, 2, 3], dtype=np.int64)
        out = _as_indptr(ptr, 3, np.int64, "ctx")
        assert out.dtype == jnp.int64


# ---------------------------------------------------------------------------
# _as_int32_cuda_offsets -- int32-only CUDA/JITC ABI guard
# ---------------------------------------------------------------------------

def test_cuda_offsets_int32_passthrough():
    off = jnp.array([0, 2, 3], dtype=jnp.int32)
    out = _as_int32_cuda_offsets(off, "ctx")
    assert out.dtype == jnp.int32


def test_cuda_offsets_int64_raises_not_implemented():
    with _jax_x64_enabled():
        off = jnp.array([0, 2, 3], dtype=jnp.int64)
        with pytest.raises(NotImplementedError, match="int32 ABI"):
            _as_int32_cuda_offsets(off, "ctx")


# ---------------------------------------------------------------------------
# Constructor end-to-end: CSR / CSC dtype contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,shape", [(CSR, (2, 3)), (CSC, (3, 2))])
def test_constructor_indices_int32_indptr_int32(cls, shape):
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    m = cls((data, indices, indptr), shape=shape)
    assert m.indices.dtype == jnp.int32
    assert m.indptr.dtype == jnp.int32


@pytest.mark.parametrize("cls,shape", [(CSR, (2, 3)), (CSC, (3, 2))])
def test_constructor_coerces_int64_indices_to_int32(cls, shape):
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = np.array([0, 2, 1], dtype=np.int64)
    indptr = np.array([0, 2, 3], dtype=np.int64)
    m = cls((data, indices, indptr), shape=shape)
    assert m.indices.dtype == jnp.int32
    # Small nnz -> indptr resolves to int32 regardless of input width.
    assert m.indptr.dtype == jnp.int32


def test_constructor_explicit_int64_indptr_gated_when_x64_off():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    with pytest.raises(ValueError, match="requires an int64 array"):
        CSR((data, indices, indptr), shape=(2, 3), indptr_dtype=np.int64)
    # The failed construction must not have toggled the global config.
    assert jax.config.jax_enable_x64 is False


def test_constructor_explicit_int64_indptr_ok_when_x64_on():
    with _jax_x64_enabled():
        data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        indices = jnp.array([0, 2, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
        m = CSR((data, indices, indptr), shape=(2, 3), indptr_dtype=np.int64)
        assert m.indices.dtype == jnp.int32
        assert m.indptr.dtype == jnp.int64


# ---------------------------------------------------------------------------
# Constructor structural validation
# ---------------------------------------------------------------------------

def test_constructor_rejects_non_monotonic_indptr():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 3, 2], dtype=jnp.int32)  # decreasing
    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        CSR((data, indices, indptr), shape=(2, 3))


def test_constructor_rejects_indptr_tail_mismatch():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 2], dtype=jnp.int32)  # tail != nse (3)
    with pytest.raises(ValueError, match="must equal the number of"):
        CSR((data, indices, indptr), shape=(2, 3))


def test_constructor_rejects_wrong_indptr_length():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 3], dtype=jnp.int32)  # length should be n_rows + 1 = 3
    with pytest.raises(ValueError, match="indptr length"):
        CSR((data, indices, indptr), shape=(2, 3))


def test_constructor_rejects_nonzero_indptr_head():
    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([1, 2, 3], dtype=jnp.int32)  # head != 0
    with pytest.raises(ValueError, match=r"indptr\[0\] must be 0"):
        CSR((data, indices, indptr), shape=(2, 3))


# ---------------------------------------------------------------------------
# Structure-preserving paths keep the contract and do not host-readback
# ---------------------------------------------------------------------------

def test_with_data_preserves_structure_dtype():
    m = _small_csr()
    m2 = m.with_data(jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32))
    assert m2.indices.dtype == jnp.int32
    assert m2.indptr.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(m2.indices), np.asarray(m.indices))


def test_transpose_preserves_structure_dtype():
    m = _small_csr()
    mt = m.transpose()
    assert isinstance(mt, CSC)
    assert mt.indices.dtype == jnp.int32
    assert mt.indptr.dtype == jnp.int32


def test_with_data_under_jit_no_host_readback():
    # A structure-preserving reconstruction under jit must not try to read the
    # (traced) data on host; indices/indptr remain concrete int32.
    m = _small_csr()

    @jax.jit
    def scale(data):
        return m.with_data(data * 2.0).data.sum()

    val = scale(m.data)
    assert float(val) == pytest.approx(2.0 * float(m.data.sum()))


# ---------------------------------------------------------------------------
# _check_compressed_structure -- tracer path skips value checks
# ---------------------------------------------------------------------------

def test_check_structure_tracer_skips_value_checks():
    # A concrete non-monotonic indptr would raise; under a tracer the value
    # checks are skipped, so no error is raised for the (static) checks.
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)

    def f(bad_ptr):
        _check_compressed_structure(indices, bad_ptr, (2, 3), format="csr")
        return bad_ptr.sum()

    # Non-monotonic values but valid shape/dtype: tracer path must not raise.
    jax.jit(f)(jnp.array([0, 3, 2], dtype=jnp.int32))


# ---------------------------------------------------------------------------
# diag_add -- int64 case raises NotImplementedError (deferred)
# ---------------------------------------------------------------------------

def test_diag_add_int64_case_raises(monkeypatch):
    # Materialising a >int32_max structure is impractical, so shrink the
    # threshold to force the int64 branch on a tiny matrix.
    monkeypatch.setattr(diag_add_mod, "_INT32_MAX", 1)
    m = _small_csr()
    with pytest.raises(NotImplementedError, match="int64 indptr offsets"):
        m.diag_add(jnp.array([1.0, 1.0], dtype=jnp.float32))


def test_diag_add_int32_case_still_works():
    m = _small_csr()
    out = m.diag_add(jnp.array([5.0, 7.0], dtype=jnp.float32))
    assert out.indices.dtype == jnp.int32
    assert out.indptr.dtype == jnp.int32
    dense = np.asarray(out.todense())
    expected = np.array([[6.0, 0.0, 2.0], [0.0, 10.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(dense, expected)
