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

# -*- coding: utf-8 -*-

"""Tests for adding a diagonal to a CSR matrix.

The int64-indptr branch is deferred rather than implemented, so it must raise a
clear ``NotImplementedError`` instead of silently producing wrong offsets.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import brainevent._csr.diag_add as diag_add_mod
from brainevent._csr._test_util import small_csr


def test_diag_add_int64_case_raises(monkeypatch):
    # Materialising a >int32_max structure is impractical, so shrink the
    # threshold to force the int64 branch on a tiny matrix.
    monkeypatch.setattr(diag_add_mod, "_INT32_MAX", 1)
    m = small_csr()
    with pytest.raises(NotImplementedError, match="int64 indptr offsets"):
        m.diag_add(jnp.array([1.0, 1.0], dtype=jnp.float32))


def test_diag_add_int32_case_still_works():
    m = small_csr()
    out = m.diag_add(jnp.array([5.0, 7.0], dtype=jnp.float32))
    assert out.indices.dtype == jnp.int32
    assert out.indptr.dtype == jnp.int32
    dense = np.asarray(out.todense())
    expected = np.array([[6.0, 0.0, 2.0], [0.0, 10.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(dense, expected)


@pytest.mark.parametrize('cls_name', ['CSR', 'CSC'])
def test_diag_add_promotes_a_float64_diagonal_to_the_common_dtype(cls_name):
    """Under x64, ``jnp.ones(n)`` is float64; a float32 matrix must accept it like ``W + diag(d)``."""
    import brainevent
    from brainevent._test_util import jax_x64_enabled

    dense = np.array([[1., 0., 2.], [0., 0., 3.], [4., 0., 0.]], dtype=np.float32)
    with jax_x64_enabled():
        mat = getattr(brainevent, cls_name).fromdense(jnp.asarray(dense))
        diag = jnp.asarray([0.5, 1.5, 2.5])
        assert mat.data.dtype == jnp.float32 and diag.dtype == jnp.float64
        out = mat.diag_add(diag)
        assert out.data.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(out.todense()), dense + np.diag(np.asarray(diag)))


def test_diag_add_promotes_mixed_dtypes_and_keeps_units():
    import brainunit as u
    import brainevent
    from brainevent._test_util import jax_x64_enabled

    dense = np.array([[1., 0., 2.], [0., 0., 3.], [4., 0., 0.]], dtype=np.float32)
    with jax_x64_enabled():
        mat = brainevent.CSR.fromdense(jnp.asarray(dense) * u.mV)
        out = mat.diag_add(jnp.asarray([1., 2., 3.]) * u.mV)
        assert u.get_unit(out.data) == u.mV
        assert out.data.dtype == jnp.float64
        np.testing.assert_allclose(u.get_mantissa(out.todense()), dense + np.diag([1., 2., 3.]))
        # Promotion does not relax the unit check.
        with pytest.raises(u.DimensionMismatchError):
            mat.diag_add(jnp.asarray([1., 2., 3.]) * u.second)


def test_diag_add_promotes_integer_data_with_a_float_diagonal():
    import brainevent

    dense = np.array([[1, 0, 2], [0, 0, 3], [4, 0, 0]], dtype=np.int32)
    mat = brainevent.CSR.fromdense(jnp.asarray(dense))
    out = mat.diag_add(jnp.asarray([0.5, 0.5, 0.5], dtype=jnp.float32))
    assert out.data.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(out.todense()), dense + 0.5 * np.eye(3))
