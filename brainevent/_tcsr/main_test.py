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

from __future__ import annotations

import operator
import subprocess
import sys
import warnings
from contextlib import contextmanager
from typing import Any

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSC as PlainCSC
from brainevent._csr.main import CSR
from brainevent._csr.main import CSR as PlainCSR
from brainevent._event import BinaryArray
from brainevent._tcsr import main
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
        shape = (out_rows,) if operand.ndim == 1 else (operand.shape[0], out_rows)
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
    assert kwargs.pop("local_targets") is view._tcs_buffers.tcsr_local_targets
    assert kwargs.pop("tile_offsets") is view._tcs_buffers.tcsr_tile_offsets
    assert kwargs == {
        "shape": view._base_shape,
        "transpose": transposed if side == "right" else not transposed,
        "backend": "sentinel",
    }
    expected_operand = operand.T if side == "right" and rank == 2 else operand
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


# ============================================================================
# BPSA isolation from stable TCSR storage views (formerly test/main_test.py).
# ============================================================================


def _sorted_dense():
    return jnp.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=jnp.float32)


def _sorted_plain_csr():
    return PlainCSR(
        (
            jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
            jnp.asarray([0, 1, 2], dtype=jnp.int32),
        ),
        shape=(2, 2),
    )


def _sorted_tcsr(**kwargs):
    from brainevent._tcsr.main import TCSR

    return TCSR.from_sorted_csr(_sorted_plain_csr(), **kwargs)


def _non_square_dense():
    return jnp.asarray(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 5.0],
        ],
        dtype=jnp.float32,
    )


def _non_square_plain_csr(data=None):
    if data is None:
        data = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    return PlainCSR(
        (
            data,
            jnp.asarray([0, 2, 1, 0, 3], dtype=jnp.int32),
            jnp.asarray([0, 2, 3, 5], dtype=jnp.int32),
        ),
        shape=(3, 4),
    )


def _non_square_tcsr(data=None, **kwargs):
    from brainevent._tcsr.main import TCSR

    return TCSR.from_sorted_csr(_non_square_plain_csr(data), **kwargs)


def test_stable_main_import_does_not_load_bpsa():
    """Keep importing stable storage independent from the BPSA package."""
    script = """
import sys
import brainevent._tcsr.main
assert 'brainevent._tcsr.bpsa' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_tcsr_rejects_removed_binary_grad_mode_keyword():
    """Reject the removed stable BPSA mode instead of retaining a dead API."""
    from brainevent._tcsr.main import TCSR

    with pytest.raises(TypeError, match="binary_grad_mode"):
        TCSR(_sorted_dense(), binary_grad_mode="bpsa")


def test_views_and_pytree_have_no_binary_grad_mode_or_eager_mirror():
    """Keep normal storage transformations free of BPSA state and side effects."""
    from brainevent._tcsr.main import TCSR

    with _explicit_int64_allowed():
        matrix = TCSR.from_sorted_csr(
            _sorted_plain_csr(),
            backward_algorithm="pp_prop",
        )
        transposed = matrix.T
        updated = matrix.with_data(matrix.data + 1.0)
        leaves, tree = jax.tree_util.tree_flatten(matrix)
        restored = jax.tree_util.tree_unflatten(tree, leaves)

    for view in (matrix, transposed, updated, restored):
        assert not hasattr(view, "binary_grad_mode")
        assert view.backward_algorithm == "pp_prop"
        assert view.has_tcsc_mirror is False


def test_tcsr_has_no_direction_specific_binary_service_selector():
    """Keep direct-versus-indexed selection below the object layer."""
    from brainevent._tcsr.main import TCSR

    assert not hasattr(TCSR, "_direct_binary_service")


def test_checked_factories_and_constructor_validation():
    """Convert checked sources and reject invalid construction contracts."""
    from brainevent._tcsr.main import TCSR

    plain_csr = _sorted_plain_csr()
    plain_csc = plain_csr.tocsc()
    assert isinstance(plain_csc, PlainCSC)
    with _explicit_int64_allowed():
        sorted_matrix = TCSR.from_sorted_csr(plain_csr)
        from_csc = TCSR.fromcsc(plain_csc)
        from_dense = TCSR.fromdense(_sorted_dense())
        np.testing.assert_allclose(sorted_matrix.todense(), _sorted_dense())
        np.testing.assert_allclose(from_csc.todense(), _sorted_dense())
        np.testing.assert_allclose(from_dense.todense(), _sorted_dense())

    with pytest.raises(TypeError, match="sorted plain CSR"):
        TCSR(object())
    with pytest.raises(TypeError, match="sorted plain CSR"):
        TCSR(jnp.ones((2,)))
    with pytest.raises(ValueError, match="backward algorithm"):
        TCSR.from_sorted_csr(
            _sorted_plain_csr(), backward_algorithm="unknown"
        )


def test_mirror_is_lazy_shared_cached_and_preserved_by_pytree():
    """Create one shared mirror on demand and retain it through serialization."""
    from brainevent._tcsr.main import TCSR

    with _explicit_int64_allowed():
        matrix = _sorted_tcsr()
        transposed = matrix.T
        assert matrix.has_tcsc_mirror is False
        assert matrix.materialize_tcsc_mirror() is matrix
        mirror = matrix._tcs_buffers.tcsc
        assert mirror is not None
        assert transposed.materialize_tcsc_mirror()._tcs_buffers.tcsc is mirror
        leaves, tree = jax.tree_util.tree_flatten(transposed)
        restored = jax.tree_util.tree_unflatten(tree, leaves)
        assert restored.has_tcsc_mirror is True
        np.testing.assert_allclose(restored.todense(), _sorted_dense().T)

        children, aux = matrix.tree_flatten()
        inconsistent_aux = (*aux[:-1], False)
        with pytest.raises(ValueError, match="PyTree state is inconsistent"):
            TCSR.tree_unflatten(inconsistent_aux, children)


def test_tcsr_transpose_returns_tcsr_views_with_inverted_state():
    """Keep both logical orientations in one shared TCSR object type."""
    from brainevent._tcsr.main import TCSR

    with _explicit_int64_allowed():
        matrix = _sorted_tcsr()
        transposed = matrix.transpose()
        property_view = matrix.T
        restored = transposed.T
        leaves, tree = jax.tree_util.tree_flatten(transposed)
        pytree_view = jax.tree_util.tree_unflatten(tree, leaves)

    for view in (transposed, property_view):
        assert isinstance(view, TCSR)
        assert view._transpose_state is True
        assert view.shape == matrix.shape[::-1]
        assert view._tcs_buffers is matrix._tcs_buffers
        assert view.has_tcsc_mirror is False

    assert isinstance(pytree_view, TCSR)
    assert pytree_view._transpose_state is True
    assert pytree_view.shape == matrix.shape[::-1]
    assert pytree_view.has_tcsc_mirror is False
    np.testing.assert_array_equal(
        pytree_view._tcsr_indices, transposed._tcsr_indices
    )
    np.testing.assert_array_equal(
        pytree_view._tcsr_indptr, transposed._tcsr_indptr
    )

    assert isinstance(restored, TCSR)
    assert restored._transpose_state is False
    assert restored.shape == matrix.shape
    assert restored._tcs_buffers is matrix._tcs_buffers


def test_tcsr_tocsc_materializes_plain_csc():
    """Materialize plain CSC instead of exposing a second tiled view class."""
    with _explicit_int64_allowed():
        matrix = _sorted_tcsr()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            converted = matrix.tocsc()
        expected = matrix.todense()
        actual = converted.todense()

    assert isinstance(converted, PlainCSC)
    np.testing.assert_allclose(actual, expected)


def test_transposed_tcsr_tocsc_uses_canonical_structure_without_mirror():
    """Materialize a transposed view as plain CSC without building its mirror."""
    with _explicit_int64_allowed():
        matrix = _sorted_tcsr()
        transposed = matrix.T
        converted = transposed.tocsc()
        actual = converted.todense()

    assert isinstance(converted, PlainCSC)
    assert matrix.has_tcsc_mirror is False
    np.testing.assert_allclose(actual, _sorted_dense().T)


def test_tcsr_main_does_not_export_tcsc_view_class():
    """Expose TCSR as the only concrete tiled sparse matrix class."""
    from brainevent._tcsr import main

    assert "TCSC" not in main.__all__
    assert not hasattr(main, "TCSC")


def test_tcsr_main_removes_obsolete_compressed_component_hook():
    """Expose logical CSR components without the removed TCSC dispatch hook."""
    from brainevent._tcsr.main import TCSR, TiledCompressedSparseData

    assert not hasattr(TiledCompressedSparseData, "_compressed_components")
    assert not hasattr(TCSR, "_compressed_components")


def test_view_conversion_data_replacement_and_sparse_materialization():
    """Preserve values and shared structure across TCSR logical views."""
    from brainevent._tcsr.main import TCSR

    with _explicit_int64_allowed():
        matrix = _sorted_tcsr(backend="jax", binary_backend="jax")
        converted = matrix.tocsc()
        transposed = matrix.T
        restored = transposed.T

        assert isinstance(converted, PlainCSC)
        assert isinstance(transposed, TCSR)
        assert isinstance(restored, TCSR)
        assert converted.shape == matrix.shape
        assert transposed.shape == matrix.shape[::-1]
        assert restored.shape == matrix.shape
        assert transposed._tcs_buffers is matrix._tcs_buffers
        assert restored._tcs_buffers is matrix._tcs_buffers
        assert matrix.nse == 2
        assert matrix.dtype == jnp.float32
        np.testing.assert_allclose(matrix.todense(), _sorted_dense())
        np.testing.assert_allclose(converted.todense(), _sorted_dense())
        np.testing.assert_allclose(transposed.todense(), _sorted_dense().T)
        np.testing.assert_allclose(matrix.tocoo().todense(), _sorted_dense())

        replacement = jnp.asarray([5.0, 7.0], dtype=jnp.float32)
        updated = transposed.with_data(replacement)
        assert isinstance(updated, TCSR)
        assert updated._transpose_state is True
        np.testing.assert_allclose(updated.data, replacement)
        np.testing.assert_allclose(
            updated.todense(), [[5.0, 0.0], [0.0, 7.0]]
        )
        assert updated._tcs_buffers is matrix._tcs_buffers

    with pytest.raises(AssertionError, match="shape"):
        matrix.with_data(jnp.ones((3,), dtype=jnp.float32))
    with pytest.raises(AssertionError, match="dtype"):
        matrix.with_data(jnp.ones((2,), dtype=jnp.float16))
    with pytest.raises(ValueError, match="TCSR transpose axes"):
        matrix.transpose((0, 1))
    with pytest.raises(ValueError, match="TCSR transpose axes"):
        transposed.transpose((0, 1))


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("event_ndim", [1, 2])
def test_binary_dispatch_uses_only_unified_binary_service(
    monkeypatch, transposed, side, event_ndim
):
    """Route every logical product through binary MV/MM with one direction."""
    from brainevent._tcsr import binary

    with _explicit_int64_allowed():
        view = _sorted_tcsr(binary_backend="jax", backward_algorithm="pp_prop")
        if transposed:
            view = view.T

    calls = []

    def fake_binary(data, indices, indptr, events, **kwargs):
        calls.append((data, indices, indptr, events, kwargs))
        if events.ndim == 1:
            output_shape = (
                (kwargs["shape"][1],)
                if kwargs["transpose"]
                else (kwargs["shape"][0],)
            )
        elif kwargs["transpose"]:
            output_shape = (events.shape[0], kwargs["shape"][1])
        else:
            output_shape = (kwargs["shape"][0], events.shape[1])
        return jnp.zeros(output_shape, dtype=jnp.float32)

    monkeypatch.setattr(binary, "binary_csrmv", fake_binary)
    monkeypatch.setattr(binary, "binary_csrmm", fake_binary)

    logical_axis = view.shape[0] if side == "left" else view.shape[1]
    if event_ndim == 1:
        event_shape = (logical_axis,)
    elif side == "left":
        event_shape = (3, logical_axis)
    else:
        event_shape = (logical_axis, 3)
    events = BinaryArray(jnp.ones(event_shape, dtype=jnp.bool_))

    result = (
        view.__rmatmul__(events) if side == "left" else view.__matmul__(events)
    )

    assert len(calls) == 1
    call = calls[0]
    kwargs = call[-1]
    assert "binary_grad_mode" not in kwargs
    assert kwargs["backward_algorithm"] == "pp_prop"
    assert kwargs["transpose"] is (
        transposed if side == "right" else not transposed
    )
    assert kwargs["buffers"] is view._tcs_buffers
    expected_events = events.value
    if event_ndim == 2 and transposed:
        expected_events = expected_events.T
    np.testing.assert_array_equal(call[3], expected_events)

    expected_shape = (
        (view.shape[1],) if side == "left" else (view.shape[0],)
    )
    if event_ndim == 2:
        expected_shape = (
            (3, view.shape[1]) if side == "left" else (view.shape[0], 3)
        )
    assert result.shape == expected_shape


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("side", ["left", "right"])
def test_binary_matmul_rejects_rank_three(transposed, side):
    """Reject unsupported binary ranks before selecting a backend."""
    with _explicit_int64_allowed():
        view = _sorted_tcsr()
        if transposed:
            view = view.T

    operation = view.__rmatmul__ if side == "left" else view.__matmul__
    with pytest.raises(NotImplementedError, match="binary matmul"):
        operation(BinaryArray(jnp.ones((1, 1, 1), dtype=jnp.bool_)))


@pytest.mark.parametrize("transposed", [False, True])
def test_dense_row_indexing_matches_logical_view_and_jit(transposed):
    """Apply NumPy-like row selectors to either logical orientation."""
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        view = matrix.T if transposed else matrix
        dense = _non_square_dense().T if transposed else _non_square_dense()

        for selector in (1, -1, slice(None, None, 2), [2, 0, 2], slice(0, 0)):
            actual = view[selector]
            expected = np.asarray(dense)[selector]
            np.testing.assert_allclose(actual, expected)

        compiled = jax.jit(lambda row: view[row])(
            jnp.asarray(1, dtype=jnp.int32)
        )
        np.testing.assert_allclose(compiled, dense[1])

        if transposed:
            view.materialize_tcsc_mirror()
        dynamic = jax.jit(lambda operand, row: operand[row])(
            view,
            jnp.asarray(1, dtype=jnp.int32),
        )
        np.testing.assert_allclose(dynamic, dense[1])

        if transposed:
            matrix.materialize_tcsc_mirror()
        dynamic = jax.jit(lambda operand, row: operand[row])(
            view,
            jnp.asarray(0, dtype=jnp.int32),
        )
        np.testing.assert_allclose(dynamic, dense[0])


@pytest.mark.parametrize("transposed", [False, True])
def test_sparse_row_slice_returns_independent_canonical_tcsr(transposed):
    """Rebuild selected logical rows as an independently owned TCSR."""
    from brainevent._tcsr.main import TCSR

    with _explicit_int64_allowed():
        matrix = _non_square_tcsr(
            backend="jax_raw",
            binary_backend="jax_raw",
            backward_algorithm="pp_prop",
        )
        view = matrix.T if transposed else matrix
        dense = _non_square_dense().T if transposed else _non_square_dense()
        rows = [view.shape[0] - 1, 0, view.shape[0] - 1]
        sliced = view.slice_rows(rows)

        assert isinstance(sliced, TCSR)
        assert sliced.shape == (3, view.shape[1])
        assert sliced._transpose_state is False
        assert sliced.backend == "jax_raw"
        assert sliced.binary_backend == "jax_raw"
        assert sliced.backward_algorithm == "pp_prop"
        assert sliced._tcs_buffers is not view._tcs_buffers
        assert sliced.has_tcsc_mirror is False
        np.testing.assert_allclose(sliced.todense(), dense[jnp.asarray(rows)])


def test_sparse_row_slice_handles_scalar_empty_homogeneous_and_units():
    """Preserve row shape, compact weights, and physical units at boundaries."""
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        one_row = matrix.slice_rows(-1)
        empty = matrix.slice_rows(slice(0, 0))

        homogeneous = _non_square_tcsr(
            jnp.asarray([2.0], dtype=jnp.float32)
        ).slice_rows([2, 0])
        unitful = _non_square_tcsr(
            jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
            * u.mV
        ).slice_rows([2, 0])

        assert one_row.shape == (1, 4)
        np.testing.assert_allclose(
            one_row.todense(), _non_square_dense()[jnp.asarray([-1])]
        )
        assert empty.shape == (0, 4)
        assert empty.nse == 0
        assert empty.todense().shape == (0, 4)
        assert homogeneous.data.shape == (1,)
        np.testing.assert_allclose(
            homogeneous.todense(),
            [[2.0, 0.0, 0.0, 2.0], [2.0, 0.0, 2.0, 0.0]],
        )
        assert u.get_unit(unitful.data) == u.mV
        assert u.math.allclose(
            unitful.todense(), _non_square_dense()[jnp.asarray([2, 0])] * u.mV
        )


def test_sparse_row_slice_rejects_traced_output_structure():
    """Reject sparse slicing when selected nonzero count is not static."""
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        operation = jax.jit(lambda rows: matrix.slice_rows(rows))
        with pytest.raises(RuntimeError, match="requires concrete"):
            operation(jnp.asarray([0, 2], dtype=jnp.int32))


def test_row_indexing_rejects_invalid_concrete_selectors():
    """Reject non-integer and out-of-bounds row selectors before dispatch."""
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()

        with pytest.raises(IndexError, match="integer"):
            _ = matrix[jnp.asarray([0.5])]
        with pytest.raises(IndexError, match="out of bounds"):
            _ = matrix.slice_rows([3])


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        (
            "update_on_pre",
            (jnp.ones((3,), dtype=jnp.bool_), jnp.ones((4,), dtype=jnp.float32)),
        ),
        (
            "update_on_post",
            (jnp.ones((3,), dtype=jnp.float32), jnp.ones((4,), dtype=jnp.bool_)),
        ),
        ("solve", (jnp.ones((3,), dtype=jnp.float32),)),
    ],
)
def test_unsupported_tcsr_operations_are_explicit(method_name, args):
    """Expose stable method names without claiming unsupported behavior."""
    from brainevent._tcsr.main import TCSR

    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()

    assert method_name in TCSR.__dict__
    with pytest.raises(NotImplementedError, match=rf"TCSR\.{method_name}"):
        getattr(matrix, method_name)(*args)


@pytest.mark.parametrize("transposed", [False, True])
def test_tcsr_apply_rejects_value_shape_changes(transposed):
    """Keep apply from changing the shared physical value-buffer shape."""
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        view = matrix.T if transposed else matrix

        with pytest.raises(ValueError, match="apply.*shape"):
            view.apply(lambda values: values.reshape((1, values.size)))


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(
    "storage", ["heterogeneous", "homogeneous_scalar", "homogeneous_vector"]
)
def test_tcsr_sum_matches_logical_dense_sum(transposed, storage):
    """Reduce all logical entries in either TCSR orientation."""
    if storage == "heterogeneous":
        data = None
    elif storage == "homogeneous_scalar":
        data = jnp.asarray(2.0, dtype=jnp.float32)
    else:
        data = jnp.asarray([2.0], dtype=jnp.float32)
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr(data)
        view = matrix.T if transposed else matrix
        assert u.math.allclose(view.sum(), view.todense().sum())
        with pytest.raises(NotImplementedError, match="sum with axis"):
            view.sum(axis=0)


def test_tcsr_homogeneous_sum_preserves_units():
    """Retain units when reducing a shared TCSR value."""
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr(jnp.asarray([2.0], dtype=jnp.float32))
        matrix = matrix.apply(lambda values: values * u.mV)
        assert u.get_unit(matrix.sum()) == u.mV
        assert u.math.allclose(matrix.sum(), 10.0 * u.mV)


# ============================================================================
# Staged TCSR construction (formerly test/main_construction_test.py).
# ============================================================================


def test_default_constructor_requires_sorted_plain_csr() -> None:
    """Keep dense conversion and sorting outside the default constructor."""
    with pytest.raises(TypeError, match="sorted plain CSR"):
        TCSR(jnp.eye(2, dtype=jnp.float32))


def test_default_constructor_rejects_removed_trusted_keyword() -> None:
    """Require the factory name, rather than a flag, to express trust."""
    source = PlainCSR(
        (
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([0], dtype=jnp.int32),
            jnp.asarray([0, 1], dtype=jnp.int32),
        ),
        shape=(1, 1),
    )

    with jax.enable_x64(), pytest.raises(TypeError, match="trusted"):
        TCSR(source, trusted=True)


def test_from_sorted_csr_enters_default_constructor_without_reordering() -> None:
    """Preserve canonical order when the caller establishes sortedness."""
    source = PlainCSR(
        (
            jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float32),
            jnp.asarray([0, 2, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 3], dtype=jnp.int32),
        ),
        shape=(2, 3),
    )

    with jax.enable_x64():
        matrix = TCSR.from_sorted_csr(source, binary_backend="jax")

    np.testing.assert_array_equal(matrix.indices, [0, 2, 1])
    np.testing.assert_array_equal(matrix.data, [10.0, 20.0, 30.0])
    assert matrix.has_tcsc_mirror is False


def test_sorted_entry_points_do_not_call_sort_csr(monkeypatch) -> None:
    """Keep sorting completely outside trusted construction paths."""
    source = PlainCSR(
        (
            jnp.asarray([10.0, 20.0], dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
            jnp.asarray([0, 2], dtype=jnp.int32),
        ),
        shape=(1, 2),
    )

    def fail_if_called(source: PlainCSR) -> PlainCSR:
        raise AssertionError(f"unexpected sort of {source!r}")

    monkeypatch.setattr(main, "sort_csr", fail_if_called)
    with jax.enable_x64():
        direct = TCSR(source)
        forwarded = TCSR.from_sorted_csr(source)

    np.testing.assert_array_equal(direct.data, source.data)
    np.testing.assert_array_equal(forwarded.data, source.data)


@pytest.mark.parametrize("factory", ["fromcsr", "fromdense", "fromcsc"])
def test_checked_factories_call_sort_csr_once(monkeypatch, factory: str) -> None:
    """Route every checked source through the single sorting stage once."""
    source = PlainCSR(
        (
            jnp.asarray([20.0, 10.0], dtype=jnp.float32),
            jnp.asarray([1, 0], dtype=jnp.int32),
            jnp.asarray([0, 2], dtype=jnp.int32),
        ),
        shape=(1, 2),
    )
    original_sort = main.sort_csr
    calls = 0

    def counted_sort(csr: PlainCSR) -> PlainCSR:
        nonlocal calls
        calls += 1
        return original_sort(csr)

    monkeypatch.setattr(main, "sort_csr", counted_sort)
    checked_source = {
        "fromcsr": source,
        "fromdense": source.todense(),
        "fromcsc": source.tocsc(),
    }[factory]

    with jax.enable_x64():
        matrix = getattr(TCSR, factory)(checked_source)

    assert calls == 1
    np.testing.assert_array_equal(matrix.indices, [0, 1])


def test_fromcsr_sorts_indices_and_corresponding_data() -> None:
    """Build checked TCSR storage with aligned canonical value order."""
    source = PlainCSR(
        (
            jnp.asarray([30.0, 10.0, 20.0], dtype=jnp.float32),
            jnp.asarray([2, 0, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 3], dtype=jnp.int32),
        ),
        shape=(2, 3),
    )

    with jax.enable_x64():
        matrix = TCSR.fromcsr(source)
        dense = matrix.todense()

    np.testing.assert_array_equal(matrix.indices, [0, 2, 1])
    np.testing.assert_array_equal(matrix.data, [10.0, 30.0, 20.0])
    np.testing.assert_array_equal(
        dense,
        [[10.0, 0.0, 30.0], [0.0, 20.0, 0.0]],
    )


def test_fromdense_uses_checked_csr_construction() -> None:
    """Convert dense input through the checked CSR factory."""
    source = jnp.asarray(
        [[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]],
        dtype=jnp.float32,
    )

    with jax.enable_x64():
        matrix = TCSR.fromdense(source)
        dense = matrix.todense()

    np.testing.assert_array_equal(dense, source)


def test_fromcsc_uses_checked_csr_construction() -> None:
    """Convert CSC input through the same checked CSR factory."""
    expected = jnp.asarray(
        [[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]],
        dtype=jnp.float32,
    )

    with jax.enable_x64():
        source = PlainCSC.fromdense(expected)
        matrix = TCSR.fromcsc(source)
        dense = matrix.todense()

    np.testing.assert_array_equal(dense, expected)


# ============================================================================
# GPU CSR vs. TCSR event-driven products across backends
# (formerly test/gpu_correctness_test.py). GPU-only.
# ============================================================================


requires_gpu_backend = pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="requires a JAX GPU backend",
)

_SIZE = 10_000
_DENSITY = 1e-2
_BATCH_SIZE = 4
_CONNECTIONS_PER_ROW = int(_SIZE * _DENSITY)
_NNZ = _SIZE * _CONNECTIONS_PER_ROW
_RTOL = 1e-5
_ATOL = 1e-5
_SMALL_SHAPE = (37, 53)
_BINARY_BACKEND_NAMES = (
    "CSR cusparse",
    "CSR cuda_raw",
    "TCSR cuda_raw",
    "TCSR jax",
    "CSR jax_raw",
)


def _make_case() -> tuple[jax.Array, ...]:
    """Create one deterministic, exactly one-percent-dense CSR workload."""
    rows = np.arange(_SIZE, dtype=np.int64)[:, None]
    offsets = np.arange(_CONNECTIONS_PER_ROW, dtype=np.int64)[None, :]
    indices = np.sort((rows * 131 + offsets * 97) % _SIZE, axis=1)
    indptr = np.arange(0, _NNZ + 1, _CONNECTIONS_PER_ROW, dtype=np.int32)

    rng = np.random.default_rng(20260908)
    data = rng.uniform(-1.0, 1.0, size=_NNZ).astype(np.float32)
    vector = rng.random(_SIZE) < 0.1
    matrix = rng.random((_BATCH_SIZE, _SIZE)) < 0.1
    return (
        jnp.asarray(data),
        jnp.asarray(indices.reshape(-1), dtype=jnp.int32),
        jnp.asarray(indptr),
        jnp.asarray(vector, dtype=jnp.bool_),
        jnp.asarray(matrix, dtype=jnp.bool_),
    )


def _assert_matches(
    actual: Any,
    expected: Any,
    *,
    backend: str,
    operation: str,
) -> None:
    """Compare every result element and identify the failing execution path."""
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"{backend} {operation} differs from CSR cuSPARSE",
    )


def _expanded_weights(data: jax.Array, nnz: int) -> np.ndarray:
    """Expand compact homogeneous weights for independent references."""
    weights = np.asarray(data)
    if weights.size == 1:
        return np.full(nnz, weights.item(), dtype=weights.dtype)
    return weights


def _csr_reference(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    operand: jax.Array,
    *,
    shape: tuple[int, int],
    reverse: bool,
    binary: bool,
) -> np.ndarray:
    """Evaluate sparse products independently with NumPy scatter-adds."""
    host_indices = np.asarray(indices)
    host_indptr = np.asarray(indptr)
    host_operand = np.asarray(operand)
    weights = _expanded_weights(data, host_indices.size)
    row_ids = np.repeat(
        np.arange(shape[0], dtype=np.int32),
        np.diff(host_indptr),
    )
    values = host_operand != 0 if binary else host_operand

    if reverse:
        if values.ndim == 1:
            result = np.zeros(shape[1], dtype=weights.dtype)
            np.add.at(result, host_indices, weights * values[row_ids])
            return result
        result = np.zeros((values.shape[0], shape[1]), dtype=weights.dtype)
        for batch_index in range(values.shape[0]):
            np.add.at(
                result[batch_index],
                host_indices,
                weights * values[batch_index, row_ids],
            )
        return result

    if values.ndim == 1:
        result = np.zeros(shape[0], dtype=weights.dtype)
        np.add.at(result, row_ids, weights * values[host_indices])
        return result
    result = np.zeros((shape[0], values.shape[1]), dtype=weights.dtype)
    for batch_index in range(values.shape[1]):
        np.add.at(
            result[:, batch_index],
            row_ids,
            weights * values[host_indices, batch_index],
        )
    return result


def _make_csr_case(
    shape: tuple[int, int],
    row_lengths: np.ndarray,
    *,
    homogeneous: bool,
    seed: int,
    value_dtype: Any = np.float32,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create deterministic sorted CSR arrays with requested row lengths."""
    if row_lengths.shape != (shape[0],):
        raise ValueError("row_lengths must contain one value per matrix row")
    rng = np.random.default_rng(seed)
    host_indices = np.concatenate(
        [
            np.sort(rng.choice(shape[1], int(length), replace=False))
            for length in row_lengths
        ]
    ).astype(np.int32)
    host_indptr = np.concatenate(
        [np.zeros(1, dtype=np.int32), np.cumsum(row_lengths, dtype=np.int32)]
    )
    weight_size = 1 if homogeneous else host_indices.size
    host_data = rng.uniform(-1.0, 1.0, size=weight_size).astype(value_dtype)
    return (
        jnp.asarray(host_data),
        jnp.asarray(host_indices),
        jnp.asarray(host_indptr),
    )


def _make_events(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    seed: int,
) -> jax.Array:
    """Create deterministic events including zero and (active) negative values."""
    rng = np.random.default_rng(seed)
    if jnp.dtype(dtype) == jnp.dtype(jnp.bool_):
        values = rng.random(shape) < 0.3
    elif jnp.dtype(dtype) == jnp.dtype(jnp.int8):
        values = rng.integers(-1, 2, size=shape, dtype=np.int8)
    else:
        values = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return jnp.asarray(values, dtype=dtype)


def _binary_matrices(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Construct all five binary sparse execution paths."""
    source = CSR((data, indices, indptr), shape=shape)
    return {
        "CSR cusparse": CSR(
            (data, indices, indptr), shape=shape, backend="cusparse"
        ),
        "CSR cuda_raw": CSR(
            (data, indices, indptr), shape=shape, backend="cuda_raw"
        ),
        "TCSR cuda_raw": TCSR.from_sorted_csr(
            source, binary_backend="cuda_raw"
        ),
        "TCSR jax": TCSR.from_sorted_csr(source, binary_backend="jax"),
        "CSR jax_raw": CSR(
            (data, indices, indptr), shape=shape, backend="jax_raw"
        ),
    }


def _binary_product(sparse: Any, operand: jax.Array, *, reverse: bool) -> Any:
    """Apply a binary operand on the selected side of a sparse matrix."""
    events = BinaryArray(operand)
    return events @ sparse if reverse else sparse @ events


def _float_matrix(
    data: Any,
    indices: jax.Array,
    indptr: jax.Array,
    shape: tuple[int, int],
    *,
    backend: str,
) -> TCSR:
    """Construct a TCSR matrix for one floating-point backend."""
    source = CSR((data, indices, indptr), shape=shape)
    return TCSR.from_sorted_csr(source, backend=backend)


def _float_product(sparse: TCSR, operand: jax.Array, *, reverse: bool) -> Any:
    """Apply a floating operand on the selected side of a TCSR matrix."""
    return operand @ sparse if reverse else sparse @ operand


def _jax_dense_from_csr(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    shape: tuple[int, int],
) -> jax.Array:
    """Build a differentiable dense reference for small CSR cases."""
    row_ids = jnp.repeat(
        jnp.arange(shape[0], dtype=indptr.dtype),
        jnp.diff(indptr),
        total_repeat_length=indices.size,
    )
    weights = (
        jnp.broadcast_to(data[0], indices.shape)
        if data.size == 1
        else data
    )
    return jnp.zeros(shape, dtype=data.dtype).at[row_ids, indices].add(weights)


@requires_gpu_backend
def test_gpu_csr_and_tcsr_mv_mm_match_cusparse() -> None:
    """Match CSR and TCSR MV/MM results against cuSPARSE on GPU."""
    data, indices, indptr, vector, matrix = _make_case()
    assert data.size == _NNZ
    assert data.size / (_SIZE * _SIZE) == _DENSITY
    reference_mv = _csr_reference(
        data,
        indices,
        indptr,
        vector,
        shape=(_SIZE, _SIZE),
        reverse=True,
        binary=True,
    )
    reference_mm = _csr_reference(
        data,
        indices,
        indptr,
        matrix,
        shape=(_SIZE, _SIZE),
        reverse=True,
        binary=True,
    )

    with jax.enable_x64():
        csr_cusparse = CSR(
            (data, indices, indptr),
            shape=(_SIZE, _SIZE),
            backend="cusparse",
        )
        csr_cuda = CSR(
            (data, indices, indptr),
            shape=(_SIZE, _SIZE),
            backend="cuda_raw",
        )
        csr_jax = CSR(
            (data, indices, indptr),
            shape=(_SIZE, _SIZE),
            backend="jax_raw",
        )
        tcsr_source = CSR(
            (data, indices, indptr),
            shape=(_SIZE, _SIZE),
        )
        tcsr_cuda = TCSR.from_sorted_csr(
            tcsr_source,
            binary_backend="cuda_raw",
        )
        tcsr_jax = TCSR.from_sorted_csr(
            tcsr_source,
            binary_backend="jax",
        )

        vector_events = BinaryArray(vector)
        matrix_events = BinaryArray(matrix)
        expected_mv = vector_events @ csr_cusparse
        expected_mm = matrix_events @ csr_cusparse
        results = {
            "CSR cuda_raw": (
                vector_events @ csr_cuda,
                matrix_events @ csr_cuda,
            ),
            "TCSR cuda_raw": (
                vector_events @ tcsr_cuda,
                matrix_events @ tcsr_cuda,
            ),
            "TCSR jax": (
                vector_events @ tcsr_jax,
                matrix_events @ tcsr_jax,
            ),
            "CSR jax_raw": (
                vector_events @ csr_jax,
                matrix_events @ csr_jax,
            ),
        }
        jax.block_until_ready((expected_mv, expected_mm, results))

    assert expected_mv.shape == (_SIZE,)
    assert expected_mm.shape == (_BATCH_SIZE, _SIZE)
    _assert_matches(
        expected_mv,
        reference_mv,
        backend="CSR cusparse",
        operation="MV",
    )
    _assert_matches(
        expected_mm,
        reference_mm,
        backend="CSR cusparse",
        operation="MM",
    )
    for backend, (actual_mv, actual_mm) in results.items():
        assert actual_mv.shape == expected_mv.shape
        assert actual_mm.shape == expected_mm.shape
        _assert_matches(
            actual_mv,
            expected_mv,
            backend=backend,
            operation="MV",
        )
        _assert_matches(
            actual_mm,
            expected_mm,
            backend=backend,
            operation="MM",
        )


@requires_gpu_backend
@pytest.mark.parametrize(
    ("reverse", "rank", "homogeneous", "event_dtype", "value_dtype"),
    [
        (True, 1, False, jnp.bool_, np.float32),
        (False, 1, True, jnp.float32, np.float32),
        (True, 2, True, jnp.int8, np.float32),
        (False, 2, False, jnp.bool_, np.float64),
    ],
    ids=(
        "left-mv-heterogeneous-bool",
        "right-mv-homogeneous-float",
        "left-mm-homogeneous-int8",
        "right-mm-heterogeneous-bool",
    ),
)
@pytest.mark.parametrize(
    "backend",
    _BINARY_BACKEND_NAMES,
    ids=("csr-cusparse", "csr-cuda", "tcsr-cuda", "tcsr-jax", "csr-jax"),
)
def test_gpu_binary_five_backend_orthogonal_cases(
    backend: str,
    reverse: bool,
    rank: int,
    homogeneous: bool,
    event_dtype: Any,
    value_dtype: Any,
) -> None:
    """Match representative directions, ranks, weights, and event dtypes."""
    row_lengths = (np.arange(_SMALL_SHAPE[0], dtype=np.int32) * 7) % 54
    with jax.enable_x64():
        data, indices, indptr = _make_csr_case(
            _SMALL_SHAPE,
            row_lengths,
            homogeneous=homogeneous,
            seed=101 + rank + int(reverse),
            value_dtype=value_dtype,
        )
    operand_size = _SMALL_SHAPE[0] if reverse else _SMALL_SHAPE[1]
    operand_shape = (
        (operand_size,)
        if rank == 1
        else ((_BATCH_SIZE, operand_size) if reverse else (operand_size, _BATCH_SIZE))
    )
    operand = _make_events(operand_shape, event_dtype, seed=211 + rank)
    expected = _csr_reference(
        data,
        indices,
        indptr,
        operand,
        shape=_SMALL_SHAPE,
        reverse=reverse,
        binary=True,
    )

    with jax.enable_x64():
        matrices = _binary_matrices(data, indices, indptr, _SMALL_SHAPE)
        actual = _binary_product(matrices[backend], operand, reverse=reverse)
        jax.block_until_ready(actual)

    _assert_matches(
        actual,
        expected,
        backend=backend,
        operation="MV" if rank == 1 else "MM",
    )


@requires_gpu_backend
def test_gpu_tcsr_transposed_view_matches_reference_in_four_quadrants() -> None:
    """Match both operand sides and ranks for transposed TCSR GPU views."""
    row_lengths = (np.arange(_SMALL_SHAPE[0], dtype=np.int32) * 11) % 54
    data, indices, indptr = _make_csr_case(
        _SMALL_SHAPE,
        row_lengths,
        homogeneous=False,
        seed=307,
    )

    with jax.enable_x64():
        source = CSR((data, indices, indptr), shape=_SMALL_SHAPE)
        matrices = {
            "TCSR cuda_raw transpose": TCSR.from_sorted_csr(
                source, binary_backend="cuda_raw"
            ).T,
            "TCSR jax transpose": TCSR.from_sorted_csr(
                source, binary_backend="jax"
            ).T,
        }
        results: list[tuple[str, str, Any, np.ndarray]] = []
        for reverse in (True, False):
            for rank in (1, 2):
                view_shape = _SMALL_SHAPE[::-1]
                operand_size = view_shape[0] if reverse else view_shape[1]
                operand_shape = (
                    (operand_size,)
                    if rank == 1
                    else (
                        (_BATCH_SIZE, operand_size)
                        if reverse
                        else (operand_size, _BATCH_SIZE)
                    )
                )
                operand = _make_events(
                    operand_shape,
                    jnp.bool_,
                    seed=401 + rank + int(reverse),
                )
                reference_operand = operand if rank == 1 else operand.T
                expected = _csr_reference(
                    data,
                    indices,
                    indptr,
                    reference_operand,
                    shape=_SMALL_SHAPE,
                    reverse=not reverse,
                    binary=True,
                )
                if rank == 2:
                    expected = expected.T
                for backend, sparse in matrices.items():
                    actual = _binary_product(sparse, operand, reverse=reverse)
                    results.append(
                        (
                            backend,
                            f"{'left' if reverse else 'right'}-{'MV' if rank == 1 else 'MM'}",
                            actual,
                            expected,
                        )
                    )
        jax.block_until_ready([actual for _, _, actual, _ in results])

    for backend, operation, actual, expected in results:
        _assert_matches(
            actual,
            expected,
            backend=backend,
            operation=operation,
        )


@requires_gpu_backend
def test_gpu_binary_ragged_rows_and_dispatch_boundaries() -> None:
    """Match empty, ragged, and CUDA dispatch-boundary rows."""
    shape = (13, 1024)
    row_lengths = np.asarray(
        [0, 1, 15, 16, 31, 32, 33, 511, 512, 513, 7, 64, 1024],
        dtype=np.int32,
    )
    data, indices, indptr = _make_csr_case(
        shape,
        row_lengths,
        homogeneous=False,
        seed=503,
    )

    cases: list[tuple[str, jax.Array, bool]] = []
    for reverse in (True, False):
        operand_size = shape[0] if reverse else shape[1]
        cases.extend(
            [
                (
                    f"{'left' if reverse else 'right'}-MV-ragged",
                    _make_events((operand_size,), jnp.bool_, seed=601),
                    reverse,
                ),
                (
                    f"{'left' if reverse else 'right'}-MM-ragged",
                    _make_events(
                        (_BATCH_SIZE, operand_size)
                        if reverse
                        else (operand_size, _BATCH_SIZE),
                        jnp.bool_,
                        seed=607,
                    ),
                    reverse,
                ),
            ]
        )

    with jax.enable_x64():
        matrices = _binary_matrices(data, indices, indptr, shape)
        results = []
        for operation, operand, reverse in cases:
            expected = _csr_reference(
                data,
                indices,
                indptr,
                operand,
                shape=shape,
                reverse=reverse,
                binary=True,
            )
            for backend, sparse in matrices.items():
                actual = _binary_product(sparse, operand, reverse=reverse)
                results.append((backend, operation, actual, expected))
        jax.block_until_ready([actual for _, _, actual, _ in results])

    for backend, operation, actual, expected in results:
        _assert_matches(
            actual,
            expected,
            backend=backend,
            operation=operation,
        )


@requires_gpu_backend
def test_gpu_binary_object_jit_and_vmap_match_direct_mm() -> None:
    """Match compiled object calls and mapped MV against direct MM."""
    row_lengths = (np.arange(_SMALL_SHAPE[0], dtype=np.int32) * 5) % 54
    data, indices, indptr = _make_csr_case(
        _SMALL_SHAPE,
        row_lengths,
        homogeneous=False,
        seed=701,
    )
    events = _make_events(
        (_BATCH_SIZE, _SMALL_SHAPE[0]),
        jnp.bool_,
        seed=709,
    )
    expected = _csr_reference(
        data,
        indices,
        indptr,
        events,
        shape=_SMALL_SHAPE,
        reverse=True,
        binary=True,
    )

    with jax.enable_x64():
        matrices = _binary_matrices(data, indices, indptr, _SMALL_SHAPE)
        results = []
        for backend, sparse in matrices.items():
            direct = BinaryArray(events) @ sparse
            compiled = jax.jit(
                lambda values, matrix=sparse: BinaryArray(values) @ matrix
            )(events)
            mapped = jax.jit(
                jax.vmap(
                    lambda values, matrix=sparse: BinaryArray(values) @ matrix
                )
            )(events)
            results.append((backend, compiled, mapped, direct))
        jax.block_until_ready(
            [
                (compiled, mapped, direct)
                for _, compiled, mapped, direct in results
            ]
        )

    for backend, compiled, mapped, direct in results:
        _assert_matches(
            compiled,
            expected,
            backend=backend,
            operation="jitted MM",
        )
        _assert_matches(
            mapped,
            direct,
            backend=backend,
            operation="vmapped MV",
        )


@requires_gpu_backend
@pytest.mark.parametrize(
    ("backend", "reverse", "rank", "homogeneous", "value_dtype"),
    [
        ("cuda_raw", True, 1, False, np.float32),
        ("cuda_raw", True, 1, True, np.float64),
        ("cuda_raw", False, 1, False, np.float64),
        ("cuda_raw", False, 1, True, np.float32),
        ("cuda_raw", True, 2, False, np.float64),
        ("cuda_raw", True, 2, True, np.float32),
        ("cuda_raw", False, 2, False, np.float32),
        ("cuda_raw", False, 2, True, np.float64),
        ("jax_raw", True, 2, True, np.float64),
        ("jax_raw", False, 1, False, np.float32),
    ],
    ids=(
        "cuda-left-mv-heterogeneous-f32",
        "cuda-left-mv-homogeneous-f64",
        "cuda-right-mv-heterogeneous-f64",
        "cuda-right-mv-homogeneous-f32",
        "cuda-left-mm-heterogeneous-f64",
        "cuda-left-mm-homogeneous-f32",
        "cuda-right-mm-heterogeneous-f32",
        "cuda-right-mm-homogeneous-f64",
        "jax-left-mm-homogeneous-f64",
        "jax-right-mv-heterogeneous-f32",
    ),
)
def test_gpu_tcsr_float_forward_orthogonal_cases(
    backend: str,
    reverse: bool,
    rank: int,
    homogeneous: bool,
    value_dtype: Any,
) -> None:
    """Match TCSR floating MV/MM branches against an independent reference."""
    row_lengths = (np.arange(_SMALL_SHAPE[0], dtype=np.int32) * 13) % 54
    with jax.enable_x64():
        data, indices, indptr = _make_csr_case(
            _SMALL_SHAPE,
            row_lengths,
            homogeneous=homogeneous,
            seed=809 + rank,
            value_dtype=value_dtype,
        )
        operand_size = _SMALL_SHAPE[0] if reverse else _SMALL_SHAPE[1]
        operand_shape = (
            (operand_size,)
            if rank == 1
            else (
                (_BATCH_SIZE, operand_size)
                if reverse
                else (operand_size, _BATCH_SIZE)
            )
        )
        rng = np.random.default_rng(821 + rank)
        operand = jnp.asarray(
            rng.uniform(-1.0, 1.0, size=operand_shape).astype(value_dtype)
        )
        expected = _csr_reference(
            data,
            indices,
            indptr,
            operand,
            shape=_SMALL_SHAPE,
            reverse=reverse,
            binary=False,
        )
        sparse = _float_matrix(
            data,
            indices,
            indptr,
            _SMALL_SHAPE,
            backend=backend,
        )
        actual = jax.jit(
            lambda values: _float_product(sparse, values, reverse=reverse)
        )(operand)
        jax.block_until_ready(actual)

    _assert_matches(
        actual,
        expected,
        backend=f"TCSR {backend}",
        operation="float MV" if rank == 1 else "float MM",
    )


@requires_gpu_backend
@pytest.mark.parametrize(
    ("backend", "reverse", "rank", "homogeneous"),
    [
        ("cuda_raw", True, 1, False),
        ("cuda_raw", False, 2, True),
        ("jax_raw", True, 2, True),
        ("jax_raw", False, 1, False),
    ],
    ids=(
        "cuda-left-mv-heterogeneous",
        "cuda-right-mm-homogeneous",
        "jax-left-mm-homogeneous",
        "jax-right-mv-heterogeneous",
    ),
)
def test_gpu_tcsr_float_jvp_and_vjp_match_dense(
    backend: str,
    reverse: bool,
    rank: int,
    homogeneous: bool,
) -> None:
    """Match representative TCSR floating JVP and VJP results."""
    shape = (11, 17)
    row_lengths = (np.arange(shape[0], dtype=np.int32) * 3) % 10
    with jax.enable_x64():
        data, indices, indptr = _make_csr_case(
            shape,
            row_lengths,
            homogeneous=homogeneous,
            seed=907 + rank,
        )
        operand_size = shape[0] if reverse else shape[1]
        operand_shape = (
            (operand_size,)
            if rank == 1
            else (
                (_BATCH_SIZE, operand_size)
                if reverse
                else (operand_size, _BATCH_SIZE)
            )
        )
        operand = jnp.asarray(
            np.random.default_rng(919 + rank)
            .uniform(-1.0, 1.0, size=operand_shape)
            .astype(np.float32)
        )
        template = _float_matrix(
            data,
            indices,
            indptr,
            shape,
            backend=backend,
        )

        def sparse_function(weights, values):
            matrix = template.with_data(weights)
            return _float_product(matrix, values, reverse=reverse)

        def dense_function(weights, values):
            matrix = _jax_dense_from_csr(weights, indices, indptr, shape)
            lhs, rhs = (values, matrix) if reverse else (matrix, values)
            return jnp.matmul(lhs, rhs, precision=jax.lax.Precision.HIGHEST)

        tangents = (jnp.full_like(data, 0.25), jnp.full_like(operand, -0.5))
        actual_primal, actual_tangent = jax.jit(
            lambda weights, values: jax.jvp(
                sparse_function,
                (weights, values),
                tangents,
            )
        )(data, operand)
        expected_primal, expected_tangent = jax.jit(
            lambda weights, values: jax.jvp(
                dense_function,
                (weights, values),
                tangents,
            )
        )(data, operand)
        cotangent = jnp.full_like(expected_primal, 0.75)

        def sparse_vjp(weights, values, cotangent_value):
            primal, pullback = jax.vjp(sparse_function, weights, values)
            return primal, *pullback(cotangent_value)

        def dense_vjp(weights, values, cotangent_value):
            primal, pullback = jax.vjp(dense_function, weights, values)
            return primal, *pullback(cotangent_value)

        actual_vjp = jax.jit(sparse_vjp)(data, operand, cotangent)
        expected_vjp = jax.jit(dense_vjp)(data, operand, cotangent)
        jax.block_until_ready(
            (
                actual_primal,
                actual_tangent,
                expected_primal,
                expected_tangent,
                actual_vjp,
                expected_vjp,
            )
        )

    _assert_matches(
        actual_primal,
        expected_primal,
        backend=f"TCSR {backend}",
        operation="float JVP primal",
    )
    _assert_matches(
        actual_tangent,
        expected_tangent,
        backend=f"TCSR {backend}",
        operation="float JVP tangent",
    )
    for name, actual, expected in zip(
        ("primal", "weight cotangent", "operand cotangent"),
        actual_vjp,
        expected_vjp,
    ):
        _assert_matches(
            actual,
            expected,
            backend=f"TCSR {backend}",
            operation=f"float VJP {name}",
        )


@requires_gpu_backend
@pytest.mark.parametrize("backend", ("cuda_raw", "jax_raw"))
def test_gpu_tcsr_float_mv_mm_preserve_weight_units(backend: str) -> None:
    """Preserve weight units through TCSR floating MV and MM products."""
    shape = (7, 9)
    row_lengths = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)
    with jax.enable_x64():
        data, indices, indptr = _make_csr_case(
            shape,
            row_lengths,
            homogeneous=False,
            seed=1009,
        )
        sparse = _float_matrix(
            data * u.mV,
            indices,
            indptr,
            shape,
            backend=backend,
        )
        vector = jnp.linspace(-1.0, 1.0, shape[0], dtype=jnp.float32)
        matrix = jnp.broadcast_to(vector, (_BATCH_SIZE, shape[0]))
        actual_mv = vector @ sparse
        actual_mm = matrix @ sparse
        jax.block_until_ready((actual_mv, actual_mm))

    expected_mv = _csr_reference(
        data,
        indices,
        indptr,
        vector,
        shape=shape,
        reverse=True,
        binary=False,
    )
    expected_mm = _csr_reference(
        data,
        indices,
        indptr,
        matrix,
        shape=shape,
        reverse=True,
        binary=False,
    )
    assert u.get_unit(actual_mv) == u.mV
    assert u.get_unit(actual_mm) == u.mV
    _assert_matches(
        u.get_mantissa(actual_mv),
        expected_mv,
        backend=f"TCSR {backend}",
        operation="unitful float MV",
    )
    _assert_matches(
        u.get_mantissa(actual_mm),
        expected_mm,
        backend=f"TCSR {backend}",
        operation="unitful float MM",
    )


@requires_gpu_backend
@pytest.mark.parametrize("neurons", [4095, 4096, 8191, 8192, 8193])
@pytest.mark.parametrize("value_dtype", [np.float32, np.float64])
def test_gpu_tcsr_float_crosses_compute_and_metadata_tile_boundaries(
    value_dtype: Any,
    neurons: int,
) -> None:
    """Cover exact compute/metadata tile dimensions and sparse edge cases."""
    shape = (4, neurons)
    probes = [value for value in (0, 4095, 4096, 8191, 8192) if value < neurons]
    probes.extend((neurons - 1, neurons - 1))
    host_indices = np.sort(np.asarray(probes, dtype=np.int32))
    row_one_end = host_indices.size - 2
    host_indptr = np.asarray(
        [0, 0, row_one_end, row_one_end + 1, host_indices.size],
        dtype=np.int64,
    )
    host_data = np.linspace(-1.0, 1.0, host_indices.size, dtype=value_dtype)
    with jax.enable_x64():
        data = jnp.asarray(host_data)
        indices = jnp.asarray(host_indices)
        indptr = jnp.asarray(host_indptr)
        sparse = _float_matrix(data, indices, indptr, shape, backend="cuda_raw")
        operands = (
            (jnp.linspace(-0.5, 0.5, shape[0], dtype=data.dtype), True),
            (jnp.linspace(-0.5, 0.5, shape[1], dtype=data.dtype), False),
            (
                jnp.arange(_BATCH_SIZE * shape[0], dtype=data.dtype).reshape(
                    _BATCH_SIZE, shape[0]
                ) / 7,
                True,
            ),
            (
                jnp.arange(shape[1] * _BATCH_SIZE, dtype=data.dtype).reshape(
                    shape[1], _BATCH_SIZE
                ) / 11,
                False,
            ),
        )
        results = []
        for operand, reverse in operands:
            actual = _float_product(sparse, operand, reverse=reverse)
            expected = _csr_reference(
                data,
                indices,
                indptr,
                operand,
                shape=shape,
                reverse=reverse,
                binary=False,
            )
            results.append((actual, expected))
        jax.block_until_ready([actual for actual, _ in results])

    for actual, expected in results:
        _assert_matches(
            actual,
            expected,
            backend="TCSR cuda_raw",
            operation=f"float tile boundaries {np.dtype(value_dtype).name}",
        )
    assert sparse.has_tcsc_mirror is False


@requires_gpu_backend
@pytest.mark.parametrize("rows", [4095, 4096, 4097])
def test_gpu_tcsr_float_crosses_4096_row_chunks(rows: int) -> None:
    """Accumulate right-side MV and MM at exact 4096-row chunk boundaries."""
    shape = (rows, 17)
    host_indices = np.arange(shape[0], dtype=np.int32) % shape[1]
    host_indptr = np.arange(shape[0] + 1, dtype=np.int64)
    with jax.enable_x64():
        data = jnp.asarray([0.75], dtype=jnp.float32)
        indices = jnp.asarray(host_indices)
        indptr = jnp.asarray(host_indptr)
        sparse = _float_matrix(data, indices, indptr, shape, backend="cuda_raw")
        vector = jnp.linspace(-1.0, 1.0, shape[1], dtype=jnp.float32)
        matrix = jnp.broadcast_to(vector[:, None], (shape[1], _BATCH_SIZE))
        actual_mv = sparse @ vector
        actual_mm = sparse @ matrix
        jax.block_until_ready((actual_mv, actual_mm))

    for actual, operand in ((actual_mv, vector), (actual_mm, matrix)):
        expected = _csr_reference(
            data,
            indices,
            indptr,
            operand,
            shape=shape,
            reverse=False,
            binary=False,
        )
        _assert_matches(
            actual,
            expected,
            backend="TCSR cuda_raw",
            operation="float 4096-row chunk",
        )
    assert sparse.has_tcsc_mirror is False


@requires_gpu_backend
def test_gpu_tcsr_float_zero_nnz_returns_zero_in_four_directions() -> None:
    """Return initialized BN outputs without launching sparse work."""
    shape = (3, 5)
    with jax.enable_x64():
        data = jnp.asarray([], dtype=jnp.float32)
        indices = jnp.asarray([], dtype=jnp.int32)
        indptr = jnp.zeros(shape[0] + 1, dtype=jnp.int64)
        sparse = _float_matrix(data, indices, indptr, shape, backend="cuda_raw")
        operands = (
            (jnp.ones(shape[0], dtype=jnp.float32), True),
            (jnp.ones(shape[1], dtype=jnp.float32), False),
            (jnp.ones((_BATCH_SIZE, shape[0]), dtype=jnp.float32), True),
            (jnp.ones((shape[1], _BATCH_SIZE), dtype=jnp.float32), False),
        )
        results = [
            _float_product(sparse, operand, reverse=reverse)
            for operand, reverse in operands
        ]
        jax.block_until_ready(results)

    for result in results:
        np.testing.assert_array_equal(result, np.zeros(result.shape, result.dtype))
    assert sparse.has_tcsc_mirror is False


@requires_gpu_backend
def test_gpu_tcsr_float_accepts_65536_entries_in_one_row() -> None:
    """Use int32 tile offsets beyond the former uint16 row limit."""
    shape = (1, 8193)
    host_indices = np.sort(
        np.arange(65536, dtype=np.int64) % shape[1]
    ).astype(np.int32)
    host_data = np.linspace(-0.25, 0.25, host_indices.size, dtype=np.float32)
    with jax.enable_x64():
        data = jnp.asarray(host_data)
        indices = jnp.asarray(host_indices)
        indptr = jnp.asarray([0, host_indices.size], dtype=jnp.int64)
        sparse = _float_matrix(data, indices, indptr, shape, backend="cuda_raw")
        vector = jnp.linspace(-1.0, 1.0, shape[1], dtype=jnp.float32)
        actual = sparse @ vector
        jax.block_until_ready(actual)

    expected = _csr_reference(
        data,
        indices,
        indptr,
        vector,
        shape=shape,
        reverse=False,
        binary=False,
    )
    _assert_matches(
        actual,
        expected,
        backend="TCSR cuda_raw",
        operation="float 65536-entry row",
    )
    assert sparse.has_tcsc_mirror is False


@requires_gpu_backend
@pytest.mark.parametrize("value_dtype", [jnp.float16, jnp.bfloat16])
def test_gpu_tcsr_float_small_dtypes_use_bn_jax_fallback(value_dtype: Any) -> None:
    """Preserve mixed TCSR index dtypes in the f16 and bf16 fallback."""
    shape = (3, 5)
    with jax.enable_x64():
        data = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=value_dtype)
        indices = jnp.asarray([0, 4, 2, 3], dtype=jnp.int32)
        indptr = jnp.asarray([0, 2, 3, 4], dtype=jnp.int64)
        sparse = _float_matrix(data, indices, indptr, shape, backend="cuda_raw")
        dense = _jax_dense_from_csr(data, indices, indptr, shape)
        left = jnp.arange(_BATCH_SIZE * shape[0], dtype=value_dtype).reshape(
            _BATCH_SIZE, shape[0]
        )
        right = jnp.arange(shape[1] * _BATCH_SIZE, dtype=value_dtype).reshape(
            shape[1], _BATCH_SIZE
        )
        results = (
            (left[0] @ sparse, left[0] @ dense),
            (sparse @ right[:, 0], dense @ right[:, 0]),
            (left @ sparse, left @ dense),
            (sparse @ right, dense @ right),
        )
        jax.block_until_ready(results)

    for actual, expected in results:
        np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)
    assert sparse.has_tcsc_mirror is False
