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


import jax.numpy as jnp

from brainevent._csr.diag_add import csr_diag_position
from brainevent._csr.test_util import get_csr


class TestCSRDiagPosition:
    def test_basic_position(self):
        """Test basic diagonal position finding."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        positions = csr_diag_position(indptr, indices, shape=(m, n))

        assert positions.shape == (min(m, n),)
        assert jnp.issubdtype(positions.dtype, jnp.integer)

        # Verify positions: for each diagonal element, check if the position is correct
        for i in range(min(m, n)):
            pos = positions[i]
            if pos >= 0:
                # Check that indices[pos] == i (the column index should match the diagonal)
                assert indices[pos] == i

    def test_rectangular_matrix(self):
        """Test diagonal position finding for rectangular matrices."""
        for m, n in [(10, 5), (5, 10)]:
            indptr, indices = get_csr(m, n, 0.5)
            positions = csr_diag_position(indptr, indices, shape=(m, n))

            assert positions.shape == (min(m, n),)

    def test_position_v2(self):
        """Test the v2 version of diagonal position finding."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        csr_pos, diag_pos = csr_diag_position(indptr, indices, shape=(m, n))

        assert csr_pos.ndim == 1
        if diag_pos is not None:
            assert diag_pos.ndim == 1
            assert len(csr_pos) == len(diag_pos)

    def test_no_diagonal_elements(self):
        """Test case where there are no diagonal elements."""
        # Create CSR with off-diagonal elements only
        indptr = jnp.array([0, 1, 2, 3])
        indices = jnp.array([1, 2, 0])  # No diagonal elements

        positions = csr_diag_position(indptr, indices, shape=(3, 3))

        # All positions should be -1
        assert jnp.all(positions == -1)
