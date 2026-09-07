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

"""Test BPSA isolation from stable TCSR storage views."""

from contextlib import contextmanager
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSC as PlainCSC
from brainevent._csr.main import CSR as PlainCSR
from brainevent._event import BinaryArray


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


def _dense():
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
        TCSR(_dense(), binary_grad_mode="bpsa")


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
        from_dense = TCSR.fromdense(_dense())
        np.testing.assert_allclose(sorted_matrix.todense(), _dense())
        np.testing.assert_allclose(from_csc.todense(), _dense())
        np.testing.assert_allclose(from_dense.todense(), _dense())

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
        np.testing.assert_allclose(restored.todense(), _dense().T)

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
        assert view._transposed is True
        assert view.shape == matrix.shape[::-1]
        assert view._tcs_buffers is matrix._tcs_buffers
        assert view.has_tcsc_mirror is False

    assert isinstance(pytree_view, TCSR)
    assert pytree_view._transposed is True
    assert pytree_view.shape == matrix.shape[::-1]
    assert pytree_view.has_tcsc_mirror is False
    np.testing.assert_array_equal(
        pytree_view._tcsr_indices, transposed._tcsr_indices
    )
    np.testing.assert_array_equal(
        pytree_view._tcsr_indptr, transposed._tcsr_indptr
    )

    assert isinstance(restored, TCSR)
    assert restored._transposed is False
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
    np.testing.assert_allclose(actual, _dense().T)


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
        np.testing.assert_allclose(matrix.todense(), _dense())
        np.testing.assert_allclose(converted.todense(), _dense())
        np.testing.assert_allclose(transposed.todense(), _dense().T)
        np.testing.assert_allclose(matrix.tocoo().todense(), _dense())

        replacement = jnp.asarray([5.0, 7.0], dtype=jnp.float32)
        updated = transposed.with_data(replacement)
        assert isinstance(updated, TCSR)
        assert updated._transposed is True
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
def test_binary_matmul_rejects_non_binary_and_rank_three(transposed, side):
    """Reject unsupported product operands before selecting a backend."""
    with _explicit_int64_allowed():
        view = _sorted_tcsr()
        if transposed:
            view = view.T

    operation = view.__rmatmul__ if side == "left" else view.__matmul__
    with pytest.raises(NotImplementedError, match="matmul with object"):
        operation(jnp.ones((2,), dtype=jnp.float32))
    with pytest.raises(NotImplementedError, match="binary matmul"):
        operation(BinaryArray(jnp.ones((1, 1, 1), dtype=jnp.bool_)))
