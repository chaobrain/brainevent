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

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import brainstate
import jax
import jax.numpy as jnp

from brainevent._csr.diag_add import (
    csr_diag_position,
    csr_diag_position_v2,
    csr_diag_add,
    csr_diag_add_v2,
    csr_diag_add_call,
)
from brainevent._csr.test_util import get_csr


class TestCSRDiagPosition:
    def test_basic_position(self):
        """Test basic diagonal position finding."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        positions = csr_diag_position(indptr, indices, (m, n))

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
            positions = csr_diag_position(indptr, indices, (m, n))

            assert positions.shape == (min(m, n),)

    def test_position_v2(self):
        """Test the v2 version of diagonal position finding."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        csr_pos, diag_pos = csr_diag_position_v2(indptr, indices, shape=(m, n))

        assert csr_pos.ndim == 1
        if diag_pos is not None:
            assert diag_pos.ndim == 1
            assert len(csr_pos) == len(diag_pos)

    def test_no_diagonal_elements(self):
        """Test case where there are no diagonal elements."""
        # Create CSR with off-diagonal elements only
        indptr = jnp.array([0, 1, 2, 3])
        indices = jnp.array([1, 2, 0])  # No diagonal elements

        positions = csr_diag_position(indptr, indices, (3, 3))

        # All positions should be -1
        assert jnp.all(positions == -1)


class TestCSRDiagAdd:
    def test_basic_add(self):
        """Test basic diagonal addition."""
        m, n = 10, 10
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        result = csr_diag_add(csr_value, diag_position, diag_value)

        assert result.shape == csr_value.shape
        assert result.dtype == csr_value.dtype

        # Verify the diagonal values were added correctly
        for i in range(m):
            pos = diag_position[i]
            if pos >= 0:
                expected = csr_value[pos] + diag_value[i]
                assert jnp.allclose(result[pos], expected)

    def test_add_with_negative_positions(self):
        """Test diagonal addition ignores negative positions."""
        csr_value = jnp.array([1.0, 2.0, 3.0, 4.0])
        diag_position = jnp.array([0, -1, 2])  # -1 should be ignored
        diag_value = jnp.array([10.0, 20.0, 30.0])

        result = csr_diag_add(csr_value, diag_position, diag_value)

        expected = jnp.array([11.0, 2.0, 33.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_add_v2(self):
        """Test the v2 version of diagonal addition."""
        m, n = 10, 10
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        positions = csr_diag_position_v2(indptr, indices, shape=(m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        result = csr_diag_add_v2(csr_value, positions, diag_value)

        assert result.shape == csr_value.shape
        assert result.dtype == csr_value.dtype

    def test_dtype_consistency(self):
        """Test that different dtypes work correctly."""
        # Only test float32 since float64 may not be available in all JAX configurations
        dtype = jnp.float32
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(dtype)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(dtype)

        result = csr_diag_add(csr_value, diag_position, diag_value)

        assert result.dtype == dtype


class TestCSRDiagAddGradients:
    def test_jvp_csr_value(self):
        """Test JVP with respect to csr_value."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        def f(x):
            return csr_diag_add(x, diag_position, diag_value)

        primals, tangents = jax.jvp(f, (csr_value,), (jnp.ones_like(csr_value),))

        assert primals.shape == csr_value.shape
        assert tangents.shape == csr_value.shape

    def test_jvp_csr_value_is_identity(self):
        """Test that JVP w.r.t. csr_value is the identity (tangent == perturbation)."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        perturbation = brainstate.random.rand(nse).astype(jnp.float32)

        def f(x):
            return csr_diag_add(x, diag_position, diag_value)

        _, tangents = jax.jvp(f, (csr_value,), (perturbation,))
        assert jnp.allclose(tangents, perturbation), (
            "JVP w.r.t. csr_value should be identity: tangent must equal perturbation"
        )

    def test_jvp_diag_value(self):
        """Test JVP with respect to diag_value."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        def f(x):
            return csr_diag_add(csr_value, diag_position, x)

        primals, tangents = jax.jvp(f, (diag_value,), (jnp.ones_like(diag_value),))

        assert primals.shape == csr_value.shape
        assert tangents.shape == csr_value.shape

    def test_jvp_diag_value_scatter(self):
        """Test that JVP w.r.t. diag_value scatters the perturbation correctly."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        perturbation = brainstate.random.rand(m).astype(jnp.float32)

        def f(x):
            return csr_diag_add(csr_value, diag_position, x)

        _, tangents = jax.jvp(f, (diag_value,), (perturbation,))

        # Expected: zeros.at[diag_position].add(perturbation)
        expected = jnp.zeros_like(csr_value)
        for i in range(m):
            pos = int(diag_position[i])
            if pos >= 0:
                expected = expected.at[pos].add(perturbation[i])

        assert jnp.allclose(tangents, expected), (
            "JVP w.r.t. diag_value should scatter perturbation to diag positions"
        )

    def test_jvp_both(self):
        """Test JVP with respect to both csr_value and diag_value."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        def f(csr, diag):
            return csr_diag_add(csr, diag_position, diag)

        primals, tangents = jax.jvp(
            f,
            (csr_value, diag_value),
            (jnp.ones_like(csr_value), jnp.ones_like(diag_value))
        )

        assert primals.shape == csr_value.shape
        assert tangents.shape == csr_value.shape

    def test_jvp_combined_vs_reference(self):
        """Test combined JVP against manual reference computation."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        csr_dot = brainstate.random.rand(nse).astype(jnp.float32)
        diag_dot = brainstate.random.rand(m).astype(jnp.float32)

        def f(csr, diag):
            return csr_diag_add(csr, diag_position, diag)

        _, tangents = jax.jvp(f, (csr_value, diag_value), (csr_dot, diag_dot))

        # Reference: d/dt [csr + scatter(diag)] = csr_dot + scatter(diag_dot)
        expected = csr_dot
        for i in range(m):
            pos = int(diag_position[i])
            if pos >= 0:
                expected = expected.at[pos].add(diag_dot[i])

        assert jnp.allclose(tangents, expected, atol=1e-6), (
            "Combined JVP should equal csr_dot + scatter(diag_dot)"
        )


class TestCSRDiagAddJIT:
    def test_jit_basic(self):
        """Test JIT compilation of csr_diag_add."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        @jax.jit
        def f(csr, diag):
            return csr_diag_add(csr, diag_position, diag)

        result = f(csr_value, diag_value)

        # Compare with non-JIT version
        expected = csr_diag_add(csr_value, diag_position, diag_value)

        assert jnp.allclose(result, expected)

    def test_jit_call_function(self):
        """Test JIT compilation of csr_diag_add_call."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        @jax.jit
        def f(csr, pos, diag):
            return csr_diag_add_call(csr, pos, diag)

        result = f(csr_value, diag_position, diag_value)

        assert result[0].shape == csr_value.shape

    def test_jit_with_grad(self):
        """Test JIT compilation with gradients."""
        m, n = 5, 5
        indptr, indices = get_csr(m, n, 0.5)
        nse = indices.shape[0]

        csr_value = brainstate.random.rand(nse).astype(jnp.float32)
        diag_position = csr_diag_position(indptr, indices, (m, n))
        diag_value = brainstate.random.rand(m).astype(jnp.float32)

        @jax.jit
        def f(csr, diag):
            return jnp.sum(csr_diag_add(csr, diag_position, diag))

        # JVP should work under JIT
        _, tangent = jax.jvp(f, (csr_value, diag_value), (jnp.ones_like(csr_value), jnp.ones_like(diag_value)))

        assert tangent.shape == ()
