# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test the formal P0 TCSR data-structure surface."""

from contextlib import contextmanager
import operator

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR as PlainCSR
from brainevent._tcsr.main import TCSR


@contextmanager
def _explicit_int64_allowed():
    previous_explicit = jax.config.jax_explicit_x64_dtypes
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_explicit_x64_dtypes", previous_explicit)


def _dense() -> jax.Array:
    return jnp.asarray(
        [[1.0, 0.0, -2.0], [3.0, -4.0, 0.0]],
        dtype=jnp.float32,
    )


def _matrix(*, backend: str = "jax_raw") -> TCSR:
    source = PlainCSR(
        (
            jnp.asarray([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32),
            jnp.asarray([0, 2, 0, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 4], dtype=jnp.int32),
        ),
        shape=(2, 3),
        backend=backend,
    )
    return TCSR.from_sorted_csr(
        source,
        backend=backend,
        binary_backend="jax",
    )


def _homogeneous_matrix(*, backend: str = "jax_raw") -> TCSR:
    source = PlainCSR(
        (
            jnp.asarray([2.0], dtype=jnp.float32),
            jnp.asarray([0, 2, 0, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 4], dtype=jnp.int32),
        ),
        shape=(2, 3),
        backend=backend,
    )
    return TCSR.from_sorted_csr(
        source,
        backend=backend,
        binary_backend="jax",
    )


@pytest.mark.parametrize("transposed", [False, True])
def test_unary_and_apply_preserve_view_and_shared_structure(transposed):
    """Apply value-only transformations without rebuilding TCSR storage."""
    with _explicit_int64_allowed():
        matrix = _matrix()
        view = matrix.T if transposed else matrix
        results = (
            view.apply(lambda values: values * 2.0),
            abs(view),
            -view,
            +view,
        )
        expected = view.todense()
        expected_results = (expected * 2.0, abs(expected), -expected, +expected)
        for result, expected_result in zip(results, expected_results):
            assert isinstance(result, TCSR)
            assert result.shape == view.shape
            assert result._transpose_state is transposed
            assert result._tcsr_indices is view._tcsr_indices
            assert result._tcsr_indptr is view._tcsr_indptr
            assert result._tcs_buffers is view._tcs_buffers
            np.testing.assert_allclose(result.todense(), expected_result)


@pytest.mark.parametrize("transposed", [False, True])
def test_elementwise_operations_follow_sparse_value_semantics(transposed):
    """Match CSR scalar, dense, reflected, and dense-result behavior."""
    with _explicit_int64_allowed():
        matrix = _matrix()
        view = matrix.T if transposed else matrix
        dense = jnp.arange(np.prod(view.shape), dtype=jnp.float32).reshape(
            view.shape
        ) + 1.0

        sparse_results = (
            view * 2.0,
            view / 2.0,
            2.0 * view,
            8.0 / view,
            view * dense,
            dense / view,
            view.apply2(view, operator.mul),
            view + view,
            view - view,
        )

        logical = view.todense()
        rows, cols = view.tocoo().row, view.tocoo().col
        stored_dense = dense[rows, cols]
        stored = view.data
        expected_data = (
            stored * 2.0,
            stored / 2.0,
            2.0 * stored,
            8.0 / stored,
            stored * stored_dense,
            stored_dense / stored,
            stored * stored,
            stored + stored,
            stored - stored,
        )
        for result, values in zip(sparse_results, expected_data):
            assert isinstance(result, TCSR)
            assert result._tcs_buffers is view._tcs_buffers
            np.testing.assert_allclose(result.data, values)

        np.testing.assert_allclose(view + dense, logical + dense)
        np.testing.assert_allclose(view - dense, logical - dense)
        np.testing.assert_allclose(dense + view, dense + logical)
        np.testing.assert_allclose(dense - view, dense - logical)


@pytest.mark.parametrize("transposed", [False, True])
def test_dense_value_operation_expands_homogeneous_data(transposed):
    """Expand a shared value when dense factors differ by stored position."""
    with _explicit_int64_allowed():
        source = PlainCSR(
            (
                jnp.asarray([2.0], dtype=jnp.float32),
                jnp.asarray([0, 2, 0, 1], dtype=jnp.int32),
                jnp.asarray([0, 2, 4], dtype=jnp.int32),
            ),
            shape=(2, 3),
        )
        matrix = TCSR.from_sorted_csr(source)
        view = matrix.T if transposed else matrix
        factors = jnp.arange(np.prod(view.shape), dtype=jnp.float32).reshape(
            view.shape
        ) + 1.0

        result = view * factors

        assert result.data.shape == (matrix.nse,)
        np.testing.assert_allclose(result.todense(), view.todense() * factors)


def test_elementwise_operations_reject_incompatible_structures_and_shapes():
    """Reject operations that need sparse structure alignment or rebuilding."""
    with _explicit_int64_allowed():
        matrix = _matrix()
        unrelated = _matrix()
        plain = matrix.tocsr()

    with pytest.raises(NotImplementedError):
        _ = matrix * unrelated
    with pytest.raises(NotImplementedError):
        _ = matrix * matrix.T
    with pytest.raises(NotImplementedError):
        _ = matrix * jnp.ones((2,), dtype=jnp.float32)
    with pytest.raises(NotImplementedError):
        matrix._binary_rop(jnp.ones((2,), dtype=jnp.float32), operator.mul)
    with pytest.raises(NotImplementedError, match="sparse"):
        matrix._binary_op(plain, operator.mul)
    with pytest.raises(NotImplementedError, match="sparse"):
        matrix._binary_rop(plain, operator.mul)

    reflected = matrix.apply2(matrix, operator.sub, reverse=True)
    assert isinstance(reflected, TCSR)
    np.testing.assert_allclose(reflected.data, jnp.zeros_like(matrix.data))


@pytest.mark.parametrize("transposed", [False, True])
def test_dense_multiply_expands_homogeneous_values_in_canonical_order(transposed):
    """Expand one shared value when dense entries differ by connection."""
    with _explicit_int64_allowed():
        matrix = _homogeneous_matrix()
        view = matrix.T if transposed else matrix
        dense = jnp.arange(np.prod(view.shape), dtype=jnp.float32).reshape(
            view.shape
        ) + 1.0

        result = view * dense

        assert result._canonical_data.shape == matrix._tcsr_indices.shape
        np.testing.assert_allclose(result.todense(), view.todense() * dense)


def test_float_matmul_accumulates_duplicates_and_preserves_empty_rows():
    """Match dense multiplication with duplicate entries and an empty row."""
    with _explicit_int64_allowed():
        source = PlainCSR(
            (
                jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
                jnp.asarray([1, 1, 3], dtype=jnp.int32),
                jnp.asarray([0, 2, 2, 3], dtype=jnp.int32),
            ),
            shape=(3, 4),
            backend="jax_raw",
        )
        matrix = TCSR.from_sorted_csr(source, backend="jax_raw")
        vector = jnp.asarray([2.0, 4.0, 6.0, 8.0], dtype=jnp.float32)

        result = matrix @ vector

        np.testing.assert_allclose(result, jnp.asarray([12.0, 0.0, 24.0]))
        assert matrix.has_tcsc_mirror is False


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("rank", [1, 2])
def test_float_dispatch_uses_canonical_storage_and_computation_flag(
    monkeypatch,
    transposed,
    side,
    rank,
):
    """Map object orientation to float compute flags without using a mirror."""
    from brainevent._tcsr import float as float_ops

    with _explicit_int64_allowed():
        matrix = _matrix(backend="sentinel")
        view = matrix.T if transposed else matrix

    calls = []

    def fake_float(data, indices, indptr, operand, **kwargs):
        calls.append((data, indices, indptr, operand, kwargs))
        out_rows = kwargs["shape"][1 if kwargs["transpose"] else 0]
        shape = (out_rows,) if operand.ndim == 1 else (out_rows, operand.shape[1])
        return jnp.zeros(shape, dtype=jnp.float32)

    monkeypatch.setattr(float_ops, "csrmv", fake_float)
    monkeypatch.setattr(float_ops, "csrmm", fake_float)

    input_axis = view.shape[0] if side == "left" else view.shape[1]
    operand_shape = (input_axis,) if rank == 1 else (
        (4, input_axis) if side == "left" else (input_axis, 4)
    )
    operand = jnp.ones(operand_shape, dtype=jnp.float32)
    result = (
        view.__rmatmul__(operand) if side == "left" else view.__matmul__(operand)
    )

    assert len(calls) == 1
    data, indices, indptr, service_operand, kwargs = calls[0]
    np.testing.assert_array_equal(data, view._canonical_data)
    assert indices is view._tcsr_indices
    assert indptr is view._tcsr_indptr
    assert kwargs == {
        "shape": view._base_shape,
        "transpose": transposed if side == "right" else not transposed,
        "backend": "sentinel",
    }
    expected_operand = operand.T if side == "left" and rank == 2 else operand
    np.testing.assert_array_equal(service_operand, expected_operand)
    expected_shape = (
        (view.shape[1],) if side == "left" else (view.shape[0],)
    )
    if rank == 2:
        expected_shape = (
            (4, view.shape[1]) if side == "left" else (view.shape[0], 4)
        )
    assert result.shape == expected_shape
    assert view.has_tcsc_mirror is False


@pytest.mark.parametrize("transposed", [False, True])
def test_float_matmul_matches_dense_for_both_sides_and_ranks(transposed):
    """Match non-square dense references in every float multiplication form."""
    with _explicit_int64_allowed():
        matrix = _matrix()
        view = matrix.T if transposed else matrix
        dense = _dense().T if transposed else _dense()
        right_vector = jnp.arange(view.shape[1], dtype=jnp.float32) + 1.0
        left_vector = jnp.arange(view.shape[0], dtype=jnp.float32) + 1.0
        right_matrix = jnp.arange(
            view.shape[1] * 4, dtype=jnp.float32
        ).reshape(view.shape[1], 4)
        left_matrix = jnp.arange(
            4 * view.shape[0], dtype=jnp.float32
        ).reshape(4, view.shape[0])

        np.testing.assert_allclose(view @ right_vector, dense @ right_vector)
        np.testing.assert_allclose(left_vector @ view, left_vector @ dense)
        np.testing.assert_allclose(view @ right_matrix, dense @ right_matrix)
        np.testing.assert_allclose(left_matrix @ view, left_matrix @ dense)
        assert view.has_tcsc_mirror is False


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("side", ["left", "right"])
def test_float_vector_vmap_lets_jax_manage_batch_layout(transposed, side):
    """Lower mapped vector products through the registered JAX batching rule."""
    with _explicit_int64_allowed():
        matrix = _matrix()
        view = matrix.T if transposed else matrix
        dense = _dense().T if transposed else _dense()
        input_axis = view.shape[0] if side == "left" else view.shape[1]
        vectors = jnp.arange(4 * input_axis, dtype=jnp.float32).reshape(
            4, input_axis
        )

        if side == "left":
            result = jax.vmap(lambda vector: vector @ view)(vectors)
            expected = vectors @ dense
        else:
            result = jax.vmap(lambda vector: view @ vector)(vectors)
            expected = jax.vmap(lambda vector: dense @ vector)(vectors)

        np.testing.assert_allclose(result, expected)
        assert view.has_tcsc_mirror is False


@pytest.mark.parametrize("transposed", [False, True])
def test_float_matmul_supports_jit_jvp_and_vjp(transposed):
    """Preserve float transforms while deriving orientation from view state."""
    with _explicit_int64_allowed():
        matrix = _matrix()
        data = matrix._canonical_data
        vector_size = matrix.shape[0] if transposed else matrix.shape[1]
        vector = jnp.arange(vector_size, dtype=jnp.float32) + 1.0
        rows = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
        cols = matrix._tcsr_indices

        def operation(values, operand):
            view = matrix.with_data(values)
            if transposed:
                view = view.T
            return view @ operand

        def reference(values, operand):
            dense = jnp.zeros(matrix.shape, dtype=values.dtype).at[
                rows, cols
            ].set(values)
            if transposed:
                dense = dense.T
            return dense @ operand

        compiled = jax.jit(operation)(data, vector)
        expected = reference(data, vector)
        np.testing.assert_allclose(compiled, expected)

        primal, tangent = jax.jvp(
            operation,
            (data, vector),
            (jnp.ones_like(data), jnp.ones_like(vector)),
        )
        ref_primal, ref_tangent = jax.jvp(
            reference,
            (data, vector),
            (jnp.ones_like(data), jnp.ones_like(vector)),
        )
        np.testing.assert_allclose(primal, ref_primal)
        np.testing.assert_allclose(tangent, ref_tangent)

        gradients = jax.grad(lambda values: operation(values, vector).sum())(data)
        ref_gradients = jax.grad(
            lambda values: reference(values, vector).sum()
        )(data)
        np.testing.assert_allclose(gradients, ref_gradients)


def test_float_matmul_preserves_physical_units():
    """Multiply matrix and operand units through the float service."""
    with _explicit_int64_allowed():
        matrix = _matrix().apply(lambda values: values * u.siemens)
        vector = jnp.arange(matrix.shape[1], dtype=jnp.float32) * u.mV

        result = matrix @ vector
        expected = (_dense() * u.siemens) @ vector

        assert u.get_unit(result) == u.get_unit(expected)
        assert u.math.allclose(result, expected)


def test_float_matmul_rejects_sparse_and_unsupported_rank():
    """Reject sparse-sparse multiplication and operands above matrix rank."""
    with _explicit_int64_allowed():
        matrix = _matrix()

    with pytest.raises(NotImplementedError, match="sparse"):
        matrix.__matmul__(matrix)
    with pytest.raises(NotImplementedError, match="shape"):
        matrix.__matmul__(jnp.ones((3, 1, 1), dtype=jnp.float32))


def test_solve_is_an_explicit_unsupported_boundary():
    """Keep solve visible without claiming a working TCSR implementation."""
    with _explicit_int64_allowed():
        matrix = _matrix()

    with pytest.raises(NotImplementedError, match="TCSR.solve"):
        matrix.solve(jnp.ones((2,), dtype=jnp.float32))


def test_tcsr_is_exported_as_a_public_data_structure():
    """Expose the same public TCSR class from package and project roots."""
    import brainevent
    from brainevent._tcsr import TCSR as PackageTCSR

    assert brainevent.TCSR is TCSR
    assert PackageTCSR is TCSR
    assert "TCSR" in brainevent.__all__
    assert TCSR.__module__ == "brainevent"
