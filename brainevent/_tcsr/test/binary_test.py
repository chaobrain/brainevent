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

"""Tests for unified TCSR binary direction routing."""

from contextlib import contextmanager
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _non_square_tcsr():
    from brainevent._tcsr.main import TCSR

    source = PlainCSR(
        (
            jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
            jnp.asarray([0, 2, 1, 2], dtype=jnp.int32),
            jnp.asarray([0, 2, 4], dtype=jnp.int32),
        ),
        shape=(2, 3),
    )
    return TCSR.from_sorted_csr(source, binary_backend="jax")


def _event(values):
    return jnp.asarray(values, dtype=jnp.bool_)


def _dense_weight():
    return jnp.asarray(
        [[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]], dtype=jnp.float32
    )


@pytest.mark.parametrize("rank", [1, 2])
def test_direct_binary_placeholder_preserves_int64_with_x64_disabled(
    rank: int,
) -> None:
    """Avoid recreating an int64 placeholder under disabled global x64."""
    previous_explicit = jax.config.jax_explicit_x64_dtypes
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    jax.config.update("jax_enable_x64", True)
    try:
        matrix = _non_square_tcsr()
    finally:
        jax.config.update("jax_enable_x64", False)
        jax.config.update("jax_explicit_x64_dtypes", "warn")

    try:
        events = (
            _event([1, 0])
            if rank == 1
            else _event([[1, 0], [0, 1]])
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            actual = BinaryArray(events) @ matrix
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_explicit_x64_dtypes", previous_explicit)

    expected = events @ _dense_weight()
    np.testing.assert_allclose(actual, expected)


def test_public_binary_services_support_both_directions_and_native_layouts():
    """Route standalone MV/MM calls and build an ephemeral mirror for false."""
    from brainevent._tcsr import binary

    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        data = matrix._canonical_data
        indices = matrix._tcsr_indices
        indptr = matrix._tcsr_indptr
        buffers = matrix._tcs_buffers
        common = dict(
            shape=(2, 3),
            workspace=buffers.tcsr_workspace,
            local_targets=buffers.tcsr_local_targets,
            tile_offsets=buffers.tcsr_tile_offsets,
            backend="jax",
        )

        left_vector = _event([1, 0, 1])
        right_vector = _event([1, 0])
        np.testing.assert_allclose(
            binary.binary_csrmv(
                data, indices, indptr, left_vector, transpose=False, **common
            ),
            _dense_weight() @ left_vector,
        )
        np.testing.assert_allclose(
            binary.binary_csrmv(
                data, indices, indptr, right_vector, transpose=True, **common
            ),
            right_vector @ _dense_weight(),
        )

        left_matrix = _event([[1, 0], [0, 1], [1, 1]])
        right_matrix = _event([[1, 0], [0, 1], [1, 1], [0, 0]])
        np.testing.assert_allclose(
            binary.binary_csrmm(
                data, indices, indptr, left_matrix, transpose=False, **common
            ),
            _dense_weight() @ left_matrix,
        )
        np.testing.assert_allclose(
            binary.binary_csrmm(
                data, indices, indptr, right_matrix, transpose=True, **common
            ),
            right_matrix @ _dense_weight(),
        )

        assert matrix.has_tcsc_mirror is False


def test_binary_mm_handles_homogeneous_duplicates_empty_rows_and_float_events():
    """Preserve event threshold semantics across both physical routes."""
    from brainevent._tcsr import binary
    from brainevent._tcsr.main import TCSR

    with _explicit_int64_allowed():
        source = PlainCSR(
            (
                jnp.asarray([2.0], dtype=jnp.float32),
                jnp.asarray([1, 1, 3, 0], dtype=jnp.int32),
                jnp.asarray([0, 0, 3, 4], dtype=jnp.int32),
            ),
            shape=(3, 4),
        )
        matrix = TCSR.from_sorted_csr(source, binary_backend="jax")
        buffers = matrix._tcs_buffers
        common = dict(
            shape=(3, 4),
            workspace=buffers.tcsr_workspace,
            local_targets=buffers.tcsr_local_targets,
            tile_offsets=buffers.tcsr_tile_offsets,
            buffers=buffers,
            backend="jax",
        )
        dense = jnp.asarray(
            [[0.0, 0.0, 0.0, 0.0],
             [0.0, 4.0, 0.0, 2.0],
             [2.0, 0.0, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        events_nb = jnp.asarray(
            [[1.0, -1.0], [0.0, 2.0], [-3.0, 0.0], [1.0, 1.0]],
            dtype=jnp.float32,
        )
        events_bn = jnp.asarray(
            [[1.0, 0.0, -2.0], [0.0, 3.0, 1.0]], dtype=jnp.float32
        )

        false_result = binary.binary_csrmm(
            matrix._canonical_data,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            events_nb,
            transpose=False,
            **common,
        )
        true_result = binary.binary_csrmm(
            matrix._canonical_data,
            matrix._tcsr_indices,
            matrix._tcsr_indptr,
            events_bn,
            transpose=True,
            **common,
        )

    np.testing.assert_allclose(false_result, dense @ (events_nb > 0))
    np.testing.assert_allclose(true_result, (events_bn > 0) @ dense)


def test_tcsr_binary_matmul_four_quadrants_and_lazy_mirror_cache():
    """Compute all side/view combinations with their documented MM layouts."""
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        weight = _dense_weight()

        right = _event([[1, 0], [0, 1], [1, 1]])
        np.testing.assert_allclose(
            matrix @ BinaryArray(right),
            weight @ right,
        )
        assert matrix.has_tcsc_mirror is True
        mirror = matrix._tcs_buffers.tcsc

        left = _event([[1, 0], [0, 1], [1, 1], [0, 0]])
        np.testing.assert_allclose(
            BinaryArray(left) @ matrix,
            left @ weight,
        )
        assert matrix._tcs_buffers.tcsc is mirror

        transposed = matrix.T
        transposed_right = _event([[1, 0], [0, 1]])
        np.testing.assert_allclose(
            transposed @ BinaryArray(transposed_right),
            weight.T @ transposed_right,
        )
        assert transposed._tcs_buffers.tcsc is mirror

        transposed_left = _event([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        np.testing.assert_allclose(
            BinaryArray(transposed_left) @ transposed,
            transposed_left @ weight.T,
        )
        assert transposed._tcs_buffers.tcsc is mirror


def test_tcsr_binary_matvec_four_quadrants():
    """Compute every operator-side and transpose-view combination for MV."""
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        weight = _dense_weight()
        right = _event([1, 0, 1])
        transposed_right = _event([1, 1])
        left = _event([1, 0])
        transposed_left = _event([1, 0, 1])

        np.testing.assert_allclose(
            matrix @ BinaryArray(right), weight @ right
        )
        np.testing.assert_allclose(
            matrix.T @ BinaryArray(transposed_right),
            weight.T @ transposed_right,
        )
        np.testing.assert_allclose(
            BinaryArray(left) @ matrix, left @ weight
        )
        np.testing.assert_allclose(
            BinaryArray(transposed_left) @ matrix.T,
            transposed_left @ weight.T,
        )


def test_raw_false_kernel_requires_mirror_enabled_state():
    """Reject only a low-level false route that bypasses mirror preparation."""
    from brainevent._tcsr import binary

    with pytest.raises(ValueError, match="enabled TCSC mirror"):
        binary._binary_csrmv_jax_kernel(
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((3,), jnp.bool_),
            shape=(2, 3),
            transpose=False,
            mirror_enabled=False,
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            outs=(jax.ShapeDtypeStruct((2,), jnp.float32),),
        )


def test_binary_mv_vmap_uses_direction_native_mm_layouts():
    """Batch mapped MV calls through MM without losing the false mirror route."""
    from brainevent._tcsr import binary

    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        buffers = matrix._tcs_buffers
        common = dict(
            shape=(2, 3),
            workspace=buffers.tcsr_workspace,
            local_targets=buffers.tcsr_local_targets,
            tile_offsets=buffers.tcsr_tile_offsets,
            buffers=buffers,
            backend="jax",
        )
        left_vectors = _event([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        right_vectors = _event([[1, 0], [0, 1], [1, 1]])
        false_result = jax.vmap(
            lambda vector: binary.binary_csrmv(
                matrix._canonical_data,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                vector,
                transpose=False,
                **common,
            )
        )(left_vectors)
        true_result = jax.vmap(
            lambda vector: binary.binary_csrmv(
                matrix._canonical_data,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                vector,
                transpose=True,
                **common,
            )
        )(right_vectors)

    np.testing.assert_allclose(false_result, left_vectors @ _dense_weight().T)
    np.testing.assert_allclose(true_result, right_vectors @ _dense_weight())


def test_prepared_mirror_crosses_dynamic_tcsr_jit_boundary():
    """Use cached mirror arrays when the TCSR itself becomes a traced input."""
    with _explicit_int64_allowed():
        matrix = _non_square_tcsr().materialize_tcsc_mirror()
        events = _event([[1, 0], [0, 1], [1, 1]])
        operation = jax.jit(
            lambda sparse, values: sparse @ BinaryArray(values)
        )
        result = operation(matrix, events)

    np.testing.assert_allclose(result, _dense_weight() @ events)


@pytest.mark.parametrize("transpose", [False, True])
def test_binary_mm_weight_gradient_preserves_canonical_order(transpose):
    """Scatter mirror-route weight gradients back to canonical TCSR slots."""
    from brainevent._tcsr import binary

    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        buffers = matrix._tcs_buffers
        events = (
            _event([[1, 0], [0, 1], [1, 1]])
            if not transpose
            else _event([[1, 0], [0, 1], [1, 1]])
        )

        def loss(weights):
            return jnp.sum(
                binary.binary_csrmm(
                    weights,
                    matrix._tcsr_indices,
                    matrix._tcsr_indptr,
                    events,
                    shape=(2, 3),
                    workspace=buffers.tcsr_workspace,
                    local_targets=buffers.tcsr_local_targets,
                    tile_offsets=buffers.tcsr_tile_offsets,
                    buffers=buffers,
                    transpose=transpose,
                    backend="jax",
                )
            )

        actual = jax.grad(loss)(matrix._canonical_data)

    if transpose:
        row_activity = jnp.sum(events, axis=0)
        expected = jnp.repeat(row_activity, 2)
    else:
        column_activity = jnp.sum(events, axis=1)
        expected = column_activity[matrix._tcsr_indices]
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("rank", "service_stem"),
    [("mv", "sddmv"), ("mm", "sddmm")],
)
@pytest.mark.parametrize(
    ("backward_algorithm", "service_suffix"),
    [("pp_prop", "float"), ("bptt", "binary")],
)
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_cuda_raw_weight_transpose_routes_to_sampled_gradient(
    monkeypatch: pytest.MonkeyPatch,
    rank: str,
    service_stem: str,
    backward_algorithm: str,
    service_suffix: str,
    transpose: bool,
    homogeneous: bool,
) -> None:
    """Route CUDA weight cotangents through the configured sampled service."""
    from jax.interpreters import ad

    from brainevent._tcsr import binary

    sentinel = jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float32)
    calls = []

    def sampled_service(name):
        def invoke(*args, **kwargs):
            calls.append((name, args, kwargs))
            return sentinel

        return invoke

    service_names = (
        "tcsr_sddmv_dweight_float",
        "tcsr_sddmv_dweight_binary",
        "tcsr_sddmm_dweight_float",
        "tcsr_sddmm_dweight_binary",
    )
    for name in service_names:
        monkeypatch.setattr(binary, name, sampled_service(name))

    if transpose:
        indices = jnp.asarray([0, 2, 1], dtype=jnp.int32)
        indptr = jnp.asarray([0, 2, 3], dtype=jnp.int32)
        local_targets = jnp.asarray([0, 2, 1], dtype=jnp.uint16)
        tile_offsets = jnp.asarray([[0, 2], [0, 1]], dtype=jnp.int32)
        permutation = jnp.asarray([0, 1, 2], dtype=jnp.int32)
        events = (
            jnp.asarray([2.0, -1.0], dtype=jnp.float32)
            if rank == "mv"
            else jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
        )
        ct = (
            jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
            if rank == "mv"
            else jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        )
        expected_events = events
        expected_ct = ct
    else:
        indices = jnp.asarray([0, 1, 0], dtype=jnp.int32)
        indptr = jnp.asarray([0, 1, 2, 3], dtype=jnp.int32)
        local_targets = jnp.asarray([0, 1, 0], dtype=jnp.uint16)
        tile_offsets = jnp.asarray(
            [[0, 1], [0, 1], [0, 1]], dtype=jnp.int32
        )
        permutation = jnp.asarray([0, 2, 1], dtype=jnp.int32)
        events = (
            jnp.asarray([2.0, -1.0, 4.0], dtype=jnp.float32)
            if rank == "mv"
            else jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
        )
        ct = (
            jnp.asarray([1.0, 2.0], dtype=jnp.float32)
            if rank == "mv"
            else jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
        )
        expected_events = events if rank == "mv" else events.T
        expected_ct = ct if rank == "mv" else ct.T

    data_shape = (1,) if homogeneous else (3,)
    data = ad.UndefinedPrimal(
        jax.core.ShapedArray(data_shape, jnp.dtype(jnp.float32))
    )
    task_begin = jnp.zeros((1,), dtype=indptr.dtype)
    task_end = jnp.zeros((1,), dtype=indptr.dtype)
    status = jnp.zeros((2,), dtype=jnp.int32)
    rule = (
        binary._csrmv_transpose_rule
        if rank == "mv"
        else binary._csrmm_transpose_rule
    )
    result = rule(
        (ct,),
        data,
        indices,
        indptr,
        events,
        task_begin,
        task_end,
        status,
        local_targets,
        tile_offsets,
        permutation,
        shape=(2, 3),
        transpose=transpose,
        backend="cuda_raw",
        backward_algorithm=backward_algorithm,
        task_capacity=1,
        mirror_enabled=not transpose,
    )

    expected_service = f"tcsr_{service_stem}_dweight_{service_suffix}"
    assert [name for name, _, _ in calls] == [expected_service]
    _, sampled_args, sampled_kwargs = calls[0]
    np.testing.assert_array_equal(sampled_args[0], expected_events)
    np.testing.assert_array_equal(sampled_args[1], expected_ct)
    np.testing.assert_array_equal(sampled_args[2], indices)
    np.testing.assert_array_equal(sampled_args[3], indptr)
    np.testing.assert_array_equal(sampled_args[4], local_targets)
    np.testing.assert_array_equal(sampled_args[5], tile_offsets)
    assert sampled_kwargs == {"transpose": True}

    if homogeneous:
        expected_gradient = jnp.asarray([60.0], dtype=jnp.float32)
    elif transpose:
        expected_gradient = sentinel
    else:
        expected_gradient = jnp.asarray([10.0, 30.0, 20.0], jnp.float32)
    np.testing.assert_array_equal(result[0], expected_gradient)


@pytest.mark.parametrize("rank", ["mv", "mm"])
@pytest.mark.parametrize("backward_algorithm", ["pp_prop", "bptt"])
@pytest.mark.parametrize("transpose", [True, False])
def test_cuda_raw_sampled_weight_gradient_matches_dense_reference(
    rank: str,
    backward_algorithm: str,
    transpose: bool,
) -> None:
    """Match sampled CUDA gradients for every rank, algorithm, and route."""
    from brainevent._tcsr import binary

    with _explicit_int64_allowed():
        matrix = _non_square_tcsr()
        buffers = matrix._tcs_buffers
        if rank == "mv":
            events = (
                jnp.asarray([2.0, -1.0], dtype=jnp.float32)
                if transpose
                else jnp.asarray([2.0, -1.0, 3.0], dtype=jnp.float32)
            )
            ct = (
                jnp.asarray([1.0, 4.0, -2.0], dtype=jnp.float32)
                if transpose
                else jnp.asarray([1.5, -3.0], dtype=jnp.float32)
            )

            def loss(weights):
                result = binary.binary_csrmv(
                    weights,
                    matrix._tcsr_indices,
                    matrix._tcsr_indptr,
                    events,
                    shape=(2, 3),
                    workspace=buffers.tcsr_workspace,
                    local_targets=buffers.tcsr_local_targets,
                    tile_offsets=buffers.tcsr_tile_offsets,
                    buffers=buffers,
                    transpose=transpose,
                    backend="cuda_raw",
                    backward_algorithm=backward_algorithm,
                )
                return jnp.vdot(result, ct)

        else:
            events = (
                jnp.asarray(
                    [[2.0, -1.0], [0.0, 3.0], [-2.0, 4.0], [1.0, 0.0]],
                    dtype=jnp.float32,
                )
                if transpose
                else jnp.asarray(
                    [[2.0, 0.0, -1.0, 3.0],
                     [-1.0, 4.0, 0.0, 2.0],
                     [3.0, -2.0, 1.0, 0.0]],
                    dtype=jnp.float32,
                )
            )
            ct = (
                jnp.asarray(
                    [[1.0, 2.0, -1.0], [0.0, 3.0, 2.0],
                     [-2.0, 1.0, 4.0], [3.0, -1.0, 0.0]],
                    dtype=jnp.float32,
                )
                if transpose
                else jnp.asarray(
                    [[1.0, 0.0, -2.0, 3.0],
                     [2.0, -1.0, 4.0, 0.0]],
                    dtype=jnp.float32,
                )
            )

            def loss(weights):
                result = binary.binary_csrmm(
                    weights,
                    matrix._tcsr_indices,
                    matrix._tcsr_indptr,
                    events,
                    shape=(2, 3),
                    workspace=buffers.tcsr_workspace,
                    local_targets=buffers.tcsr_local_targets,
                    tile_offsets=buffers.tcsr_tile_offsets,
                    buffers=buffers,
                    transpose=transpose,
                    backend="cuda_raw",
                    backward_algorithm=backward_algorithm,
                )
                return jnp.vdot(result, ct)

        actual = jax.grad(loss)(matrix._canonical_data)

    event_values = np.asarray(events)
    activity = (
        event_values if backward_algorithm == "pp_prop" else event_values > 0
    )
    ct_values = np.asarray(ct)
    indptr = np.asarray(matrix._tcsr_indptr)
    row_ids: np.ndarray = np.repeat(
        np.arange(matrix.shape[0]), np.diff(indptr)
    )
    indices = np.asarray(matrix._tcsr_indices)
    if rank == "mv" and transpose:
        expected = activity[row_ids] * ct_values[indices]
    elif rank == "mv":
        expected = ct_values[row_ids] * activity[indices]
    elif transpose:
        expected = np.sum(activity[:, row_ids] * ct_values[:, indices], axis=0)
    else:
        expected = np.sum(ct_values[row_ids, :] * activity[indices, :], axis=1)
    np.testing.assert_allclose(actual, expected)
