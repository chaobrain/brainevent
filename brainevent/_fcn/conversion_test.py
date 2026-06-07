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

# -*- coding: utf-8 -*-

"""Tests for the :class:`FixedNumConn` format conversions.

These exercise :meth:`todense`, :meth:`tocsr`, and :meth:`tocsc`.  All
correctness checks route through the dense round-trip, which uses the pure-JAX
``jax.experimental.sparse`` primitives (``coo_todense`` / ``csr_todense``) and
therefore needs no compilation-heavy backend -- so this module is *not* marked
``slow`` and runs in the default ``pytest`` lane.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainunit as u

from brainevent import CSC, CSR, FixedNumPerPost, FixedNumPerPre
from brainevent._test_util import allclose, generate_fixed_conn_num_indices

# Duplicate columns within a row exercise the accumulation semantics shared by
# ``todense`` and the matmul kernels: row 0 has column ``2`` twice; row 1 has
# columns ``3`` and ``1`` twice.
DUP_INDICES = jnp.array([[0, 1, 2, 2], [1, 3, 3, 1], [2, 0, 3, 1]], dtype=jnp.int32)
DUP_DATA = jnp.array([[1., 9., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]], dtype=jnp.float32)


def _make(cls, homo, shape, n_conn, seed=0):
    """Build a connection of ``cls`` with random structure and values."""
    pre = shape[0] if cls is FixedNumPerPre else shape[1]
    n_post = shape[1] if cls is FixedNumPerPre else shape[0]
    # ``replace=True`` keeps the cheap ``randint`` path (no ``for_loop``); the
    # duplicate connections it may produce are themselves useful coverage.
    indices = generate_fixed_conn_num_indices(pre, n_post, n_conn, replace=True)
    if homo:
        data = jnp.asarray(1.5, dtype=jnp.float32)
    else:
        rng = np.random.default_rng(seed)
        data = jnp.asarray(rng.standard_normal(indices.shape), dtype=jnp.float32)
    return cls((data, indices), shape=shape)


CLASSES = [FixedNumPerPre, FixedNumPerPost]
SHAPES = [(6, 8), (10, 5)]


class TestToDense:
    @pytest.mark.parametrize('cls', CLASSES)
    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('homo', [True, False])
    def test_todense_shape(self, cls, shape, homo):
        conn = _make(cls, homo, shape, n_conn=3)
        assert conn.todense().shape == shape


class TestToCsr:
    @pytest.mark.parametrize('cls', CLASSES)
    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('homo', [True, False])
    def test_roundtrip_matches_dense(self, cls, shape, homo):
        conn = _make(cls, homo, shape, n_conn=4)
        csr = conn.tocsr()
        assert isinstance(csr, CSR)
        assert csr.shape == shape
        assert allclose(csr.todense(), conn.todense())

    @pytest.mark.parametrize('cls', CLASSES)
    def test_metadata(self, cls):
        shape = (6, 8)
        conn = _make(cls, homo=False, shape=shape, n_conn=3)
        csr = conn.tocsr()
        assert csr.nse == conn.indices.size
        assert csr.dtype == conn.dtype
        assert csr.indptr.shape == (shape[0] + 1,)

    @pytest.mark.parametrize('cls', CLASSES)
    def test_homogeneous_weight_is_kept_compact(self, cls):
        conn = _make(cls, homo=True, shape=(6, 8), n_conn=3)
        csr = conn.tocsr()
        # A single shared value is preserved rather than materialised per entry.
        assert csr.data.size == 1
        assert allclose(csr.todense(), conn.todense())

    @pytest.mark.parametrize('cls', CLASSES)
    def test_num_conn_one(self, cls):
        conn = _make(cls, homo=False, shape=(5, 7), n_conn=1)
        assert allclose(conn.tocsr().todense(), conn.todense())

    def test_duplicates_preserved_pre(self):
        conn = FixedNumPerPre((DUP_DATA, DUP_INDICES), shape=(3, 4))
        assert allclose(conn.tocsr().todense(), conn.todense())

    def test_duplicates_preserved_post(self):
        conn = FixedNumPerPost((DUP_DATA, DUP_INDICES), shape=(4, 3))
        assert allclose(conn.tocsr().todense(), conn.todense())

    def test_units_preserved(self):
        conn = FixedNumPerPre((DUP_DATA * u.mS, DUP_INDICES), shape=(3, 4))
        csr = conn.tocsr()
        assert u.get_unit(csr.data) == u.mS
        assert u.get_unit(csr.todense()) == u.mS
        assert allclose(u.get_mantissa(csr.todense()), u.get_mantissa(conn.todense()))

    def test_backend_propagated(self):
        conn = FixedNumPerPre((DUP_DATA, DUP_INDICES), shape=(3, 4), backend='numba')
        assert conn.tocsr().backend == 'numba'

    def test_requires_outside_jit(self):
        conn = FixedNumPerPre((DUP_DATA, DUP_INDICES), shape=(3, 4))
        with pytest.raises(RuntimeError, match='outside'):
            jax.jit(lambda m: m.tocsr())(conn)


class TestToCsc:
    @pytest.mark.parametrize('cls', CLASSES)
    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('homo', [True, False])
    def test_roundtrip_matches_dense(self, cls, shape, homo):
        conn = _make(cls, homo, shape, n_conn=4)
        csc = conn.tocsc()
        assert isinstance(csc, CSC)
        assert csc.shape == shape
        assert allclose(csc.todense(), conn.todense())

    def test_duplicates_preserved_pre(self):
        conn = FixedNumPerPre((DUP_DATA, DUP_INDICES), shape=(3, 4))
        assert allclose(conn.tocsc().todense(), conn.todense())

    def test_duplicates_preserved_post(self):
        conn = FixedNumPerPost((DUP_DATA, DUP_INDICES), shape=(4, 3))
        assert allclose(conn.tocsc().todense(), conn.todense())

    def test_units_preserved(self):
        conn = FixedNumPerPost((DUP_DATA * u.mV, DUP_INDICES), shape=(4, 3))
        csc = conn.tocsc()
        assert u.get_unit(csc.data) == u.mV
        assert u.get_unit(csc.todense()) == u.mV
        assert allclose(u.get_mantissa(csc.todense()), u.get_mantissa(conn.todense()))


class TestCrossFormatConsistency:
    @pytest.mark.parametrize('cls', CLASSES)
    @pytest.mark.parametrize('homo', [True, False])
    def test_csr_csc_dense_agree(self, cls, homo):
        conn = _make(cls, homo, (7, 9), n_conn=3)
        dense = conn.todense()
        assert allclose(conn.tocsr().todense(), dense)
        assert allclose(conn.tocsc().todense(), dense)
        # Transpose equivalence: CSR of W^T equals CSC of W, array-for-array.
        assert allclose(conn.tocsr().todense().T, conn.transpose().tocsr().todense())
        jax.block_until_ready((dense,))
