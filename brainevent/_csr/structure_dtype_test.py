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

from brainevent import CSR, CSC


def test_csr_constructor_auto_downcasts_small_numpy_int64_indptr():
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)
    csr = CSR((jnp.ones(2, dtype=jnp.float32), indices, indptr), shape=(2, 2))
    assert csr.indices.dtype == jnp.int32
    assert csr.indptr.dtype == jnp.int32


def test_csr_constructor_rejects_numpy_int64_indptr_that_would_truncate():
    indptr = np.array([0, np.iinfo(np.int32).max + 1], dtype=np.int64)
    indices = np.array([0], dtype=np.int32)
    with pytest.raises(OverflowError, match="int32"):
        CSR((jnp.ones(1, dtype=jnp.float32), indices, indptr), shape=(1, 1))


def test_csr_constructor_explicit_int64_requires_jax_x64():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            CSR(
                (
                    jnp.ones(2, dtype=jnp.float32),
                    np.array([0, 1], dtype=np.int32),
                    np.array([0, 1, 2], dtype=np.int32),
                ),
                shape=(2, 2),
                indptr_dtype=jnp.int64,
            )
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def test_csr_constructor_explicit_int64_preserves_through_structure_preserving_ops():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        csr = CSR(
            (
                jnp.ones(2, dtype=jnp.float32),
                np.array([0, 1], dtype=np.int32),
                np.array([0, 1, 2], dtype=np.int32),
            ),
            shape=(2, 2),
            indptr_dtype=jnp.int64,
        )
        assert csr.indptr.dtype == jnp.int64
        assert csr.with_data(jnp.ones(2, dtype=jnp.float32)).indptr.dtype == jnp.int64
        assert csr.apply(lambda x: x + 1).indptr.dtype == jnp.int64
        assert csr.T.indptr.dtype == jnp.int64
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def test_fromdense_rejects_int64_index_dtype():
    dense = jnp.eye(2, dtype=jnp.float32)
    with pytest.raises(ValueError, match="only supports int32 indices"):
        CSR.fromdense(dense, index_dtype=jnp.int64)
    with pytest.raises(ValueError, match="only supports int32 indices"):
        CSC.fromdense(dense, index_dtype=jnp.int64)


def test_fromdense_explicit_int64_indptr_requires_jax_x64():
    old_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        dense = jnp.eye(2, dtype=jnp.float32)
        with pytest.raises(ValueError, match="jax_enable_x64"):
            CSR.fromdense(dense, indptr_dtype=jnp.int64)
        with pytest.raises(ValueError, match="jax_enable_x64"):
            CSC.fromdense(dense, indptr_dtype=jnp.int64)
    finally:
        jax.config.update("jax_enable_x64", old_x64)


def test_csc_constructor_auto_downcasts_small_numpy_int64_indptr():
    indptr = np.array([0, 1, 2], dtype=np.int64)
    indices = np.array([0, 1], dtype=np.int64)
    csc = CSC((jnp.ones(2, dtype=jnp.float32), indices, indptr), shape=(2, 2))
    assert csc.indices.dtype == jnp.int32
    assert csc.indptr.dtype == jnp.int32
