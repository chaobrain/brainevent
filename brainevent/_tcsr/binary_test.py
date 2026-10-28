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

import warnings
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR as PlainCSR
from brainevent._event import BinaryArray, WeightedArray
from brainevent._tcsr import binary
from brainevent._tcsr.binary import binary_csrmm_p, binary_csrmv_p
from brainevent._test_util import check_event_op_grads


requires_gpu_backend = pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="requires a JAX GPU backend",
)


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


def _recording_tile_ffi_call(calls):
    """Return an FFI stand-in that records TileMM calls."""

    def ffi_call(target, out_info, **ffi_kwargs):
        def invoke(*args, **attrs):
            calls.append((target, out_info, ffi_kwargs, args, attrs))
            return tuple(
                jnp.zeros(info.shape, dtype=info.dtype) for info in out_info
            )

        return invoke

    return ffi_call


def _tile_kernel_metadata(weight_size, event_dtype, weight_dtype):
    """Build abstract metadata for a direct two-row TileMM call."""
    task_info = jax.ShapeDtypeStruct((1,), jnp.int64)
    status_info = jax.ShapeDtypeStruct((2,), jnp.int32)
    return (
        jax.ShapeDtypeStruct((weight_size,), weight_dtype),
        jax.ShapeDtypeStruct((2, 2), event_dtype),
        {
            "shape": (2, 3),
            "mirror_enabled": False,
            "indices_info": jax.ShapeDtypeStruct((3,), jnp.int32),
            "indptr_info": jax.ShapeDtypeStruct((3,), jnp.int64),
            "outs": (
                jax.ShapeDtypeStruct((2, 3), weight_dtype),
                task_info,
                task_info,
                status_info,
            ),
        },
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

    np.testing.assert_allclose(false_result, dense @ (events_nb != 0))
    np.testing.assert_allclose(true_result, (events_bn != 0) @ dense)


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


@requires_gpu_backend
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
        event_values if backward_algorithm == "pp_prop" else event_values != 0
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


@pytest.mark.parametrize("weight_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.int8, jnp.float32])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_tile_kernel_routes_supported_dtypes_to_canonical_int8(
    monkeypatch: pytest.MonkeyPatch,
    weight_dtype,
    event_dtype,
    homogeneous: bool,
) -> None:
    """Select the weight ABI while passing only canonical int8 spikes."""
    from brainevent._tcsr import binary

    loads = []
    calls = []
    monkeypatch.setattr(
        binary,
        "load_cuda_file",
        lambda path, name, **kwargs: loads.append((path, name, kwargs)),
    )
    monkeypatch.setattr(
        binary.jax.ffi,
        "ffi_call",
        _recording_tile_ffi_call(calls),
    )

    with _explicit_int64_allowed():
        weight_size = 1 if homogeneous else 3
        weight_info, event_info, kwargs = _tile_kernel_metadata(
            weight_size, event_dtype, weight_dtype
        )
        kernel = binary._binary_csrmm_tile_cuda_kernel(
            weight_info,
            event_info,
            True,
            **kwargs,
        )
        weights = jnp.arange(1, weight_size + 1, dtype=weight_dtype)
        events = jnp.asarray([[1, 0], [-1, 2]], dtype=event_dtype)
        task_begin = jnp.zeros((1,), dtype=jnp.int64)
        task_end = jnp.zeros((1,), dtype=jnp.int64)
        status = jnp.zeros((2,), dtype=jnp.int32)
        kernel(
            weights,
            jnp.asarray([0, 2, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 3], dtype=jnp.int64),
            events,
            task_begin,
            task_end,
            status,
            jnp.asarray([0, 2, 1], dtype=jnp.uint16),
            jnp.asarray([[0, 2], [0, 1]], dtype=jnp.int32),
            jnp.asarray([], dtype=jnp.int64),
        )

    suffix = "f32" if weight_dtype == jnp.float32 else "f64"
    homo = "_homo" if homogeneous else ""
    assert len(loads) == 1
    assert calls[0][0] == (
        f"tcsr_binary_csrmm_tile.binary_csrmm_tile{homo}_{suffix}"
    )
    assert calls[0][1][0].dtype == jnp.dtype(weight_dtype)
    spike_bn = calls[0][3][4]
    assert spike_bn.dtype == jnp.int8
    np.testing.assert_array_equal(spike_bn, np.asarray(events) != 0)


def test_mapped_homogeneous_tile_kernel_uses_f64_and_canonical_int8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the same f64 and spike contract to mapped homogeneous MV."""
    from brainevent._tcsr import binary

    calls = []
    monkeypatch.setattr(binary, "load_cuda_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        binary.jax.ffi,
        "ffi_call",
        _recording_tile_ffi_call(calls),
    )

    with _explicit_int64_allowed():
        task_info = jax.ShapeDtypeStruct((1,), jnp.int64)
        status_info = jax.ShapeDtypeStruct((2,), jnp.int32)
        kernel = binary._binary_csrmm_indexed_cuda_kernel(
            jax.ShapeDtypeStruct((1,), jnp.float64),
            jax.ShapeDtypeStruct((2, 2), jnp.int8),
            False,
            shape=(3, 2),
            mirror_enabled=True,
            indices_info=jax.ShapeDtypeStruct((3,), jnp.int32),
            indptr_info=jax.ShapeDtypeStruct((3,), jnp.int64),
            outs=(
                jax.ShapeDtypeStruct((3, 2), jnp.float64),
                task_info,
                task_info,
                status_info,
            ),
        )
        kernel(
            jnp.asarray([2.0], dtype=jnp.float64),
            jnp.asarray([0, 2, 1], dtype=jnp.int32),
            jnp.asarray([0, 2, 3], dtype=jnp.int64),
            jnp.asarray([[1, 0], [-1, 2]], dtype=jnp.int8),
            jnp.zeros((1,), dtype=jnp.int64),
            jnp.zeros((1,), dtype=jnp.int64),
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.asarray([0, 2, 1], dtype=jnp.uint16),
            jnp.asarray([[0, 2], [0, 1]], dtype=jnp.int32),
            jnp.asarray([0, 1, 2], dtype=jnp.int64),
        )

    assert calls[0][0].endswith("binary_csrmm_tile_homo_f64")
    assert calls[0][1][0].dtype == jnp.float64
    assert calls[0][3][4].dtype == jnp.int8
    # events.T == [[1, -1], [0, 2]]; -1 is active under ``x != 0``.
    np.testing.assert_array_equal(calls[0][3][4], [[1, 1], [0, 1]])


@pytest.mark.parametrize(
    ("weight_dtype", "event_dtype", "error"),
    [
        (jnp.float16, jnp.bool_, "float32 or float64"),
        (jnp.float32, jnp.complex64, "boolean, integer, or floating-point"),
    ],
)
def test_tile_kernel_rejects_unsupported_dtypes_before_cuda_load(
    monkeypatch: pytest.MonkeyPatch,
    weight_dtype,
    event_dtype,
    error: str,
) -> None:
    """Reject unsupported TileMM dtypes before compiling CUDA."""
    from brainevent._tcsr import binary

    def unexpected_load(*args, **kwargs):
        raise AssertionError("CUDA must not load for unsupported dtypes")

    monkeypatch.setattr(binary, "load_cuda_file", unexpected_load)
    weight_info, event_info, kwargs = _tile_kernel_metadata(
        3, event_dtype, weight_dtype
    )
    with pytest.raises(TypeError, match=error):
        binary._binary_csrmm_tile_cuda_kernel(
            weight_info,
            event_info,
            True,
            **kwargs,
        )


@requires_gpu_backend
@pytest.mark.parametrize("weight_dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.int8, jnp.float32])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_cuda_tile_mm_matches_dense_across_f64_subtile_boundaries(
    weight_dtype,
    event_dtype,
    homogeneous: bool,
) -> None:
    """Compute boundary-heavy TileMM cases in both supported precisions."""
    from brainevent._tcsr import binary
    from brainevent._tcsr.main import TCSR

    indices_np = np.asarray(
        [4095, 4096, 4096, 8191, 8192, 8200, 0, 4096, 8192],
        dtype=np.int32,
    )
    indptr_np = np.asarray([0, 6, 9], dtype=np.int64)
    slot_values = np.asarray(
        [0.5, 1.0, -2.0, 3.0, 4.0, -1.5, 2.0, -0.25, 1.25]
    )
    raw_events = np.asarray([[1, 0], [0, 2], [-1, 3], [0, 0]])
    if event_dtype == jnp.bool_:
        raw_events = raw_events != 0

    with _explicit_int64_allowed():
        data = (
            jnp.asarray([1.25], dtype=weight_dtype)
            if homogeneous
            else jnp.asarray(slot_values, dtype=weight_dtype)
        )
        source = PlainCSR(
            (
                data,
                jnp.asarray(indices_np, dtype=jnp.int32),
                jnp.asarray(indptr_np, dtype=jnp.int64),
            ),
            shape=(2, 8201),
        )
        matrix = TCSR.from_sorted_csr(source, binary_backend="cuda_raw")
        buffers = matrix._tcs_buffers
        events = jnp.asarray(raw_events, dtype=event_dtype)

        def operation(values, spikes):
            return binary.binary_csrmm(
                values,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                spikes,
                shape=matrix.shape,
                workspace=buffers.tcsr_workspace,
                local_targets=buffers.tcsr_local_targets,
                tile_offsets=buffers.tcsr_tile_offsets,
                buffers=buffers,
                transpose=True,
                backend="cuda_raw",
            )

        actual = jax.jit(operation)(data, events)

    dense = np.zeros((2, 8201), dtype=np.dtype(weight_dtype))
    values_np = (
        np.full(indices_np.shape, 1.25, dtype=np.dtype(weight_dtype))
        if homogeneous
        else slot_values.astype(np.dtype(weight_dtype))
    )
    for row in range(2):
        begin, end = indptr_np[row : row + 2]
        np.add.at(dense[row], indices_np[begin:end], values_np[begin:end])
    expected = (np.asarray(raw_events) != 0).astype(dense.dtype) @ dense

    assert actual.dtype == jnp.dtype(weight_dtype)
    tolerance = 1e-12 if weight_dtype == jnp.float64 else 1e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@requires_gpu_backend
def test_cuda_tile_mm_f64_bptt_weight_gradient_matches_dense_reference() -> None:
    """Differentiate the f64 TileMM path across the 4096-column split."""
    from brainevent._tcsr import binary
    from brainevent._tcsr.main import TCSR

    with _explicit_int64_allowed():
        indices = jnp.asarray([4095, 4096, 8192, 4096], dtype=jnp.int32)
        indptr = jnp.asarray([0, 3, 4], dtype=jnp.int64)
        data = jnp.asarray([0.5, 1.0, -2.0, 3.0], dtype=jnp.float64)
        source = PlainCSR((data, indices, indptr), shape=(2, 8193))
        matrix = TCSR.from_sorted_csr(source, binary_backend="cuda_raw")
        buffers = matrix._tcs_buffers
        events = jnp.asarray(
            [[1, 0], [0, 2], [-1, 3], [1, 1]], dtype=jnp.int8
        )
        cotangent = jnp.arange(4 * 8193, dtype=jnp.float64).reshape(4, 8193)

        def loss(values):
            output = binary.binary_csrmm(
                values,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                events,
                shape=matrix.shape,
                workspace=buffers.tcsr_workspace,
                local_targets=buffers.tcsr_local_targets,
                tile_offsets=buffers.tcsr_tile_offsets,
                buffers=buffers,
                transpose=True,
                backend="cuda_raw",
            )
            return jnp.vdot(output, cotangent)

        actual = jax.jit(jax.grad(loss))(data)

    active = np.asarray(events) != 0
    expected = np.asarray(
        [
            np.sum(active[:, 0] * np.asarray(cotangent)[:, 4095]),
            np.sum(active[:, 0] * np.asarray(cotangent)[:, 4096]),
            np.sum(active[:, 0] * np.asarray(cotangent)[:, 8192]),
            np.sum(active[:, 1] * np.asarray(cotangent)[:, 4096]),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Activity contract: a numeric event is active iff ``x != 0`` (negative values
# and NaN included), a boolean event iff ``x``, on every binary backend, and the
# default (``bptt``) weight gradient uses the same mask (#198).
# ---------------------------------------------------------------------------

from brainevent._tcsr.binary import binary_csrmm_p as _tcsr_binary_csrmm_p
from brainevent._tcsr.binary import binary_csrmv_p as _tcsr_binary_csrmv_p
from brainevent._test_util import check_event_op_grads

_ACTIVITY_DTYPES = [
    jnp.bool_, jnp.int8, jnp.uint8, jnp.int16, jnp.int32,
    jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64,
]
_ACTIVITY_SHAPE = (37, 41)
_ACTIVITY_BATCH = 4


def _activity_backend_params(primitive):
    platform = jax.default_backend()
    backends = primitive.available_backends(platform)
    if not backends:
        return [pytest.param(None, marks=pytest.mark.skip(reason="no backend"), id="none")]
    return [pytest.param(b, id=b) for b in backends]


def _activity_events(shape, dtype):
    """Events from ``[2, -1, 0, 0.5]`` (integers ``[2, -1, 0, 1]``) as ``dtype``.

    Floating dtypes also carry one NaN, which is active.
    """
    n = int(np.prod(shape))
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        base = np.asarray([True, True, False, True])
    elif np.issubdtype(dtype, np.integer):
        base = np.asarray([2, -1, 0, 1])
    else:
        base = np.asarray([2.0, -1.0, 0.0, 0.5])
    values = np.resize(base, n)
    values = values[np.random.RandomState(0).permutation(n)].reshape(shape).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        flat = values.reshape(-1)
        flat[np.flatnonzero(flat)[1]] = np.nan
    return values


def _activity_tcsr(binary_backend, homogeneous):
    """Small TCSR (with an empty row) and its dense float32 reference."""
    from brainevent._tcsr.main import TCSR

    rows, cols = _ACTIVITY_SHAPE
    rng = np.random.RandomState(3)
    mask = rng.rand(rows, cols) < 0.3
    mask[5] = False
    indices = np.concatenate([np.flatnonzero(r) for r in mask]).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(mask.sum(1))]).astype(np.int64)
    if homogeneous:
        data = np.asarray([1.5], dtype=np.float32)
        dense = mask * np.float32(1.5)
    else:
        data = rng.uniform(-1.0, 1.0, indices.size).astype(np.float32)
        dense = np.zeros((rows, cols), np.float32)
        dense[np.repeat(np.arange(rows), mask.sum(1)), indices] = data
    source = PlainCSR(
        (jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr)),
        shape=_ACTIVITY_SHAPE,
    )
    return TCSR.from_sorted_csr(source, binary_backend=binary_backend), dense.astype(np.float32)


def _activity_event_shape(op):
    rows, cols = _ACTIVITY_SHAPE
    return {
        "mv": (cols,), "vm": (rows,),
        "mm": (cols, _ACTIVITY_BATCH), "mr": (_ACTIVITY_BATCH, rows),
    }[op]


def _activity_apply(op, mat, events):
    return mat @ events if op in ("mv", "mm") else events @ mat


def _check_tcsr_activity(binary_backend, homogeneous, op, dtype):
    host = _activity_events(_activity_event_shape(op), dtype)
    with _explicit_int64_allowed():
        matrix, dense = _activity_tcsr(binary_backend, homogeneous)
        got = np.asarray(_activity_apply(op, matrix, BinaryArray(jnp.asarray(host))))
    ref = _activity_apply(op, dense, (host != 0).astype(np.float32))
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("binary_backend", _activity_backend_params(_tcsr_binary_csrmv_p))
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("op", ["mv", "vm"])
@pytest.mark.parametrize("dtype", _ACTIVITY_DTYPES, ids=lambda d: np.dtype(d).name)
def test_tcsr_binary_mv_activity_all_dtypes_match_dense(binary_backend, homogeneous, op, dtype):
    _check_tcsr_activity(binary_backend, homogeneous, op, dtype)


@pytest.mark.parametrize("binary_backend", _activity_backend_params(_tcsr_binary_csrmm_p))
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("op", ["mm", "mr"])
@pytest.mark.parametrize("dtype", _ACTIVITY_DTYPES, ids=lambda d: np.dtype(d).name)
def test_tcsr_binary_mm_activity_all_dtypes_match_dense(binary_backend, homogeneous, op, dtype):
    _check_tcsr_activity(binary_backend, homogeneous, op, dtype)


def _check_tcsr_activity_grads(binary_backend, homogeneous, op):
    n = int(np.prod(_activity_event_shape(op)))
    values = np.resize(np.asarray([2.0, -1.0, 0.0], dtype=np.float32), n)
    values = values[np.random.RandomState(1).permutation(n)].reshape(_activity_event_shape(op))
    with _explicit_int64_allowed():
        matrix, _ = _activity_tcsr(binary_backend, homogeneous)

        def op_fn(data, v):
            return _activity_apply(op, matrix.with_data(data), BinaryArray(v))

        def dense_fn(data, z):
            return _activity_apply(op, matrix.with_data(data).todense(), z)

        # Keep the dense GPU reference out of TF32.
        with jax.default_matmul_precision("highest"):
            check_event_op_grads(
                op_fn, dense_fn, matrix.data, jnp.asarray(values),
                mode="binary", rtol=1e-4, atol=1e-4,
            )


@pytest.mark.parametrize("binary_backend", _activity_backend_params(_tcsr_binary_csrmv_p))
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("op", ["mv", "vm"])
def test_tcsr_binary_mv_grads_use_activity_mask(binary_backend, homogeneous, op):
    _check_tcsr_activity_grads(binary_backend, homogeneous, op)


@pytest.mark.parametrize("binary_backend", _activity_backend_params(_tcsr_binary_csrmm_p))
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("op", ["mm", "mr"])
def test_tcsr_binary_mm_grads_use_activity_mask(binary_backend, homogeneous, op):
    _check_tcsr_activity_grads(binary_backend, homogeneous, op)


@requires_gpu_backend
def test_cuda_hybrid_dtype_check_raises_python_exception_instead_of_aborting():
    """A failing hybrid ``BE_CSRMV_WAT_HYBRID_CHECK`` must raise, not ``abort()``.

    Feed float64 weights to the float32 hybrid kernel (bypassing the Python
    dispatch, which never does this) in a subprocess: before the fix the
    host-side check called ``std::abort()`` and killed the interpreter.
    """
    import os
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import jax, jax.numpy as jnp
        jax.config.update('jax_enable_x64', True)
        from brainevent._tcsr import binary
        f32, i32 = jnp.float32, jnp.int32
        cap = 4
        kernel = binary._binary_csrmv_cuda_kernel(
            jax.ShapeDtypeStruct((3,), f32),
            jax.ShapeDtypeStruct((2,), jnp.bool_),
            True,
            mirror_enabled=False,
            indices_info=jax.ShapeDtypeStruct((3,), i32),
            indptr_info=jax.ShapeDtypeStruct((3,), i32),
            outs=(
                jax.ShapeDtypeStruct((3,), f32),
                jax.ShapeDtypeStruct((cap,), i32),
                jax.ShapeDtypeStruct((cap,), i32),
                jax.ShapeDtypeStruct((2,), i32),
            ),
            task_capacity=cap,
        )
        try:
            out = kernel(
                jnp.ones((3,), jnp.float64),  # wrong: kernel expects float32
                jnp.asarray([0, 2, 1], i32),
                jnp.asarray([0, 2, 3], i32),
                jnp.asarray([True, False]),
                jnp.zeros((cap,), i32), jnp.zeros((cap,), i32), jnp.zeros((2,), i32),
                None, None, None,
            )
            jax.block_until_ready(out)
        except Exception as e:
            print('RAISED', 'expects matching weights dtype' in str(e))
        else:
            print('NO_ERROR')
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        timeout=900, env=os.environ.copy(),
    )
    assert proc.returncode >= 0, f"process killed by signal {-proc.returncode}:\n{proc.stderr[-2000:]}"
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RAISED True" in proc.stdout, (proc.stdout, proc.stderr[-2000:])


# ============================================================================
# Weighted (``WeightedArray``) events (formerly binary_weighted_test.py, #201).
#
# ``WeightedArray(x) @ W`` equals ``x @ dense(W)`` exactly, visiting only
# ``x != 0``. The weight cotangent is value-weighted (the ``pp_prop`` rule)
# whatever ``backward_algorithm`` says; the event cotangent is the float op.
# ============================================================================


_WEIGHTED_DTYPES = [
    jnp.int8, jnp.uint8, jnp.int16, jnp.int32,
    jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64,
]
_SHAPE = (37, 41)
_BATCH = 4


def _backend_params(primitive):
    backends = primitive.available_backends(jax.default_backend())
    if not backends:
        return [pytest.param(None, marks=pytest.mark.skip(reason="no backend"), id="none")]
    return [pytest.param(b, id=b) for b in backends]


def _weighted_events(shape, dtype, seed=0):
    """Events from ``[2, -1, 0, 0.5]`` (signed ints ``[2, -1, 0, 1]``, uint8 ``[2, 3, 0, 1]``)."""
    n = int(np.prod(shape))
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        base = np.asarray([2, 3, 0, 1])
    elif np.issubdtype(dtype, np.integer):
        base = np.asarray([2, -1, 0, 1])
    else:
        base = np.asarray([2.0, -1.0, 0.0, 0.5])
    values = np.resize(base, n)
    values = values[np.random.RandomState(seed).permutation(n)].reshape(shape)
    return jnp.asarray(values.astype(np.float64)).astype(dtype)


def _tcsr(binary_backend, homogeneous, *, shape=_SHAPE, heavy_rows=(), heavy_cols=(),
          backward_algorithm="bptt", dtype=np.float32):
    """TCSR with an empty row (and optional dense rows) plus its float32 dense matrix."""
    from brainevent._tcsr.main import TCSR

    rows, cols = shape
    rng = np.random.RandomState(3)
    mask = rng.rand(rows, cols) < 0.3
    mask[min(5, rows - 1)] = False
    for row in heavy_rows:
        mask[row] = True
    for col in heavy_cols:
        mask[:, col] = True
    indices = np.concatenate([np.flatnonzero(r) for r in mask]).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(mask.sum(1))]).astype(np.int64)
    if homogeneous:
        data = np.asarray([1.5], dtype=dtype)
    else:
        data = rng.uniform(-1.0, 1.0, indices.size).astype(dtype)
    source = PlainCSR(
        (jnp.asarray(data), jnp.asarray(indices), jnp.asarray(indptr)),
        shape=shape,
    )
    return TCSR.from_sorted_csr(
        source,
        binary_backend=binary_backend,
        backward_algorithm=backward_algorithm,
    )


def _event_shape(op, shape=_SHAPE, batch=_BATCH):
    rows, cols = shape
    return {
        "mv": (cols,), "vm": (rows,),
        "mm": (cols, batch), "mr": (batch, rows),
    }[op]


def _apply(op, mat, events):
    return mat @ events if op in ("mv", "mm") else events @ mat


def _tolerance(dtype):
    # float16/bfloat16 event cotangents are rounded to the event dtype.
    if np.dtype(dtype) in (np.dtype(jnp.float16), np.dtype(jnp.bfloat16)):
        return 2e-2
    return 1e-4


def _check_weighted(binary_backend, homogeneous, op, dtype, **tcsr_kwargs):
    # float64 events run with float64 weights; other events are read in the
    # (float32) weight dtype, which is exact for the test values.
    weight_dtype = np.float64 if np.dtype(dtype) == np.float64 else np.float32
    with _explicit_int64_allowed():
        matrix = _tcsr(binary_backend, homogeneous, dtype=weight_dtype, **tcsr_kwargs)
        shape = matrix.shape
        x = _weighted_events(_event_shape(op, shape), dtype)

        def op_fn(data, v):
            return _apply(op, matrix.with_data(data), WeightedArray(v))

        def dense_fn(data, z):
            return _apply(op, matrix.with_data(data).todense(), z.astype(data.dtype))

        tol = _tolerance(dtype)
        check_event_op_grads(
            op_fn, dense_fn, matrix.data, x,
            mode="weighted", rtol=tol, atol=tol,
        )


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmv_p))
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["mv", "vm"])
@pytest.mark.parametrize("dtype", _WEIGHTED_DTYPES, ids=lambda d: np.dtype(d).name)
def test_tcsr_weighted_mv_matches_dense_with_exact_grads(binary_backend, homogeneous, op, dtype):
    _check_weighted(binary_backend, homogeneous, op, dtype)


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmm_p))
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["mm", "mr"])
@pytest.mark.parametrize("dtype", _WEIGHTED_DTYPES, ids=lambda d: np.dtype(d).name)
def test_tcsr_weighted_mm_matches_dense_with_exact_grads(binary_backend, homogeneous, op, dtype):
    _check_weighted(binary_backend, homogeneous, op, dtype)


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmm_p))
@pytest.mark.parametrize("backward_algorithm", ["bptt", "pp_prop"])
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
def test_tcsr_weighted_grads_ignore_backward_algorithm(binary_backend, backward_algorithm, op):
    """Weighted events always use the value-weighted weight cotangent."""
    _check_weighted(
        binary_backend, False, op, jnp.float32,
        backward_algorithm=backward_algorithm,
    )


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmm_p))
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
def test_tcsr_weighted_differs_from_binary_only_by_values(binary_backend, op):
    """The binary product of the same events stays mask-based."""
    with _explicit_int64_allowed():
        matrix = _tcsr(binary_backend, False)
        dense = np.asarray(matrix.todense())
        x = _weighted_events(_event_shape(op), jnp.float32)
        weighted = np.asarray(_apply(op, matrix, WeightedArray(x)))
        binary_out = np.asarray(_apply(op, matrix, BinaryArray(x)))
    xn = np.asarray(x)
    np.testing.assert_allclose(weighted, _apply(op, dense, xn), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        binary_out, _apply(op, dense, (xn != 0).astype(np.float32)), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmv_p))
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["mv", "vm"])
def test_tcsr_weighted_vmap_over_events_matches_dense(binary_backend, homogeneous, op):
    """``vmap`` of a weighted MV lowers to the weighted MM primitive."""
    with _explicit_int64_allowed():
        matrix = _tcsr(binary_backend, homogeneous)
        dense = np.asarray(matrix.todense())
        xs = _weighted_events((3,) + _event_shape(op), jnp.float32)
        got = jax.vmap(lambda v: _apply(op, matrix, WeightedArray(v)))(xs)
    expected = np.stack([_apply(op, dense, np.asarray(v)) for v in xs])
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-5, atol=1e-5)


def test_tcsr_weighted_rejects_non_numeric_events():
    with _explicit_int64_allowed():
        matrix = _tcsr("jax", False)
        with pytest.raises(TypeError, match="weighted events"):
            binary.binary_csrmv(
                matrix.data,
                matrix._tcsr_indices,
                matrix._tcsr_indptr,
                jnp.ones((_SHAPE[0],), dtype=jnp.complex64),
                shape=_SHAPE,
                workspace=matrix._tcs_buffers.tcsr_workspace,
                local_targets=matrix._tcs_buffers.tcsr_local_targets,
                tile_offsets=matrix._tcs_buffers.tcsr_tile_offsets,
                transpose=True,
                backend="jax",
                weighted=True,
            )


def test_tcsr_weighted_inactive_nan_weight_does_not_propagate():
    """``0 * w`` terms are skipped: a NaN weight in an inactive row stays out."""
    with _explicit_int64_allowed():
        matrix = _tcsr("jax", False)
        data = np.asarray(matrix.data).copy()
        indptr = np.asarray(matrix._tcsr_indptr)
        x = np.zeros((_SHAPE[0],), np.float32)
        x[0] = 2.0
        data[indptr[1]:indptr[2]] = np.nan  # row 1 is inactive
        # cuSPARSE is a float SpMV of ``x``, so ``0 * NaN`` is not skipped there.
        backends = [
            b for b in binary_csrmv_p.available_backends(jax.default_backend())
            if b != "cusparse"
        ]
        for backend in backends:
            m = _tcsr(backend, False).with_data(jnp.asarray(data))
            out = np.asarray(WeightedArray(jnp.asarray(x)) @ m)
            assert np.all(np.isfinite(out)), backend


# ---------------------------------------------------------------------------
# CUDA routing and ABI (the FFI call is recorded, nothing is compiled).
# ---------------------------------------------------------------------------


def _recording_ffi_call(calls):
    def ffi_call(target, out_info, **ffi_kwargs):
        def invoke(*args, **attrs):
            calls.append((target, out_info, args))
            return tuple(jnp.zeros(i.shape, dtype=i.dtype) for i in out_info)

        return invoke

    return ffi_call


def _hybrid_kwargs(out_shape, out_dtype, *, mirror_enabled, shape=(2, 3)):
    task = jax.ShapeDtypeStruct((1,), jnp.int64)
    return dict(
        shape=shape,
        mirror_enabled=mirror_enabled,
        indices_info=jax.ShapeDtypeStruct((3,), jnp.int32),
        indptr_info=jax.ShapeDtypeStruct((shape[0] + 1,), jnp.int64),
        outs=(
            jax.ShapeDtypeStruct(out_shape, out_dtype),
            task,
            task,
            jax.ShapeDtypeStruct((2,), jnp.int32),
        ),
        task_capacity=1,
        weighted=True,
    )


def _invoke(kernel, weights, events, *, n_rows):
    return kernel(
        weights,
        jnp.asarray([0, 2, 1], dtype=jnp.int32),
        jnp.asarray([0, 2, 3, 3][: n_rows + 1], dtype=jnp.int64),
        events,
        jnp.zeros((1,), dtype=jnp.int64),
        jnp.zeros((1,), dtype=jnp.int64),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.asarray([0, 2, 1], dtype=jnp.uint16),
        jnp.zeros((n_rows, 2), dtype=jnp.int32),
        jnp.asarray([0, 1, 2], dtype=jnp.int64),
    )


@pytest.mark.parametrize("weight_dtype", [jnp.float32, jnp.float64, jnp.float16, jnp.bfloat16],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.int8, jnp.float32, jnp.float64],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize(
    ("homogeneous", "mirror_enabled", "entry"),
    [
        (False, False, "binary_csrmv_wat_hybrid_hetero"),
        (True, False, "binary_csrmv_wat_hybrid_homo"),
        (True, True, "binary_csrmv_wat_hybrid_homo"),
        (False, True, "binary_indexed_csrmv_wat_hybrid_hetero"),
    ],
)
def test_weighted_cuda_mv_uses_wv_entry_with_weight_dtype_events(
    monkeypatch, weight_dtype, event_dtype, homogeneous, mirror_enabled, entry,
):
    calls = []
    monkeypatch.setattr(binary, "load_cuda_file", lambda *a, **k: None)
    monkeypatch.setattr(binary.jax.ffi, "ffi_call", _recording_ffi_call(calls))
    with _explicit_int64_allowed():
        n_weights = 1 if homogeneous else 3
        kernel = binary._binary_csrmv_cuda_kernel(
            jax.ShapeDtypeStruct((n_weights,), weight_dtype),
            jax.ShapeDtypeStruct((2,), event_dtype),
            not mirror_enabled,
            **_hybrid_kwargs((3,), weight_dtype, mirror_enabled=mirror_enabled),
        )
        events = jnp.asarray([2, 0], dtype=jnp.float64).astype(event_dtype)
        _invoke(kernel, jnp.ones((n_weights,), weight_dtype), events, n_rows=2)
    target, _, args = calls[0]
    suffix = {"float32": "_f32", "float64": "_f64", "float16": "_f16", "bfloat16": "_bf16"}[
        np.dtype(weight_dtype).name
    ]
    assert target.split(".", 1)[1] == f"{entry}{suffix}_wv"
    event_arg = args[4] if entry.startswith("binary_indexed") else args[3]
    assert event_arg.dtype == jnp.dtype(weight_dtype)
    np.testing.assert_array_equal(
        np.asarray(event_arg.astype(jnp.float32)), np.asarray(events.astype(jnp.float32))
    )


@pytest.mark.parametrize("event_dtype", [jnp.bool_, jnp.int8, jnp.float16, jnp.float64],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize(
    ("homogeneous", "transpose", "entry"),
    [
        (False, True, "binary_csrmm_sraw_hybrid_hetero"),
        (True, True, "binary_csrmm_sraw_hybrid_homo"),
        (False, False, "binary_indexed_csrmm_sraw_hybrid_hetero"),
        (True, False, "binary_csrmm_sraw_hybrid_homo"),
    ],
)
def test_weighted_cuda_mm_routes_to_hybrid_not_tile(
    monkeypatch, event_dtype, homogeneous, transpose, entry,
):
    """Weighted MM never reaches the tile kernel (it only sees active rows)."""
    calls = []
    monkeypatch.setattr(binary, "load_cuda_file", lambda *a, **k: None)
    monkeypatch.setattr(binary.jax.ffi, "ffi_call", _recording_ffi_call(calls))
    with _explicit_int64_allowed():
        n_weights = 1 if homogeneous else 3
        batch = 5
        # transpose=True: BN events (batch, rows=2) -> (batch, cols=3).
        # transpose=False: NB events (rows=3, batch) on the mirror -> (2, batch).
        if transpose:
            event_shape, out_shape, n_rows = (batch, 2), (batch, 3), 2
        else:
            event_shape, out_shape, n_rows = (3, batch), (2, batch), 3
        kernel = binary._binary_csrmm_tile_cuda_kernel(
            jax.ShapeDtypeStruct((n_weights,), jnp.float32),
            jax.ShapeDtypeStruct(event_shape, event_dtype),
            transpose,
            **_hybrid_kwargs(out_shape, jnp.float32, mirror_enabled=not transpose),
        )
        events = _weighted_events(event_shape, event_dtype if event_dtype != jnp.bool_ else jnp.float32)
        events = events.astype(event_dtype)
        out = _invoke(kernel, jnp.ones((n_weights,), jnp.float32), events, n_rows=n_rows)
    target, out_info, args = calls[0]
    assert "tile" not in target
    assert target.split(".", 1)[1] == f"{entry}_f32_wv"
    events_nb = args[4]
    assert events_nb.dtype == jnp.float32
    expected_nb = np.asarray(events.astype(jnp.float32))
    np.testing.assert_array_equal(np.asarray(events_nb), expected_nb.T if transpose else expected_nb)
    assert out_info[0].shape == (batch, out_shape[1] if transpose else out_shape[0])
    assert out[0].shape == out_shape


def test_binary_cuda_paths_keep_binary_entries(monkeypatch):
    """``weighted=False`` keeps the original kernel symbols."""
    calls = []
    monkeypatch.setattr(binary, "load_cuda_file", lambda *a, **k: None)
    monkeypatch.setattr(binary.jax.ffi, "ffi_call", _recording_ffi_call(calls))
    with _explicit_int64_allowed():
        kwargs = _hybrid_kwargs((3,), jnp.float32, mirror_enabled=False)
        kwargs["weighted"] = False
        kernel = binary._binary_csrmv_cuda_kernel(
            jax.ShapeDtypeStruct((3,), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.int8),
            True,
            **kwargs,
        )
        _invoke(kernel, jnp.ones((3,), jnp.float32), jnp.asarray([2, 0], jnp.int8), n_rows=2)
    assert calls[0][0].endswith("binary_csrmv_wat_hybrid_hetero_f32_bool")
    assert calls[0][2][3].dtype == jnp.bool_


# ---------------------------------------------------------------------------
# GPU: heavy rows exercise the hybrid task (block) kernels, whose row lookup
# recovers each task's event value.
# ---------------------------------------------------------------------------


@requires_gpu_backend
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
@pytest.mark.parametrize("dtype", [jnp.int8, jnp.float16, jnp.float32], ids=lambda d: np.dtype(d).name)
def test_cuda_weighted_heavy_rows_match_dense(homogeneous, op, dtype):
    shape = (300, 9000)
    with _explicit_int64_allowed():
        # Rows 3, 7 and 200 are dense (9000 > 2 * 4096 slots, several tasks
        # on the canonical side); columns 11 and 4000 are dense (300 > 128
        # slots, tasks on the mirror side used by mv/mm); row 5 is empty.
        matrix = _tcsr(
            "cuda_raw", homogeneous, shape=shape,
            heavy_rows=(3, 7, 200), heavy_cols=(11, 4000),
        )
        x = _weighted_events(_event_shape(op, shape, batch=4), dtype, seed=1)
        got = np.asarray(_apply(op, matrix, WeightedArray(x)))
        dense = np.asarray(matrix.todense(), dtype=np.float64)
    expected = _apply(op, dense, np.asarray(x.astype(jnp.float32), dtype=np.float64))
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)


@requires_gpu_backend
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
def test_cuda_weighted_heavy_rows_grads(homogeneous, op):
    shape = (300, 9000)
    with _explicit_int64_allowed():
        matrix = _tcsr(
            "cuda_raw", homogeneous, shape=shape,
            heavy_rows=(3, 7, 200), heavy_cols=(11, 4000),
        )
        x = _weighted_events(_event_shape(op, shape, batch=4), jnp.float32, seed=2)

        def op_fn(data, v):
            return _apply(op, matrix.with_data(data), WeightedArray(v))

        def dense_fn(data, z):
            return _apply(op, matrix.with_data(data).todense(), z)

        check_event_op_grads(
            op_fn, dense_fn, matrix.data, x,
            mode="weighted", rtol=1e-3, atol=1e-3, check_vmap=False,
        )


# ---------------------------------------------------------------------------
# Inactive events contribute exactly zero (NaN/Inf weight isolation,
# docs/specs/2026-10-24-inactive-row-nan-isolation.md): NaN/±Inf weights
# placed only on entries whose event is inactive must not reach the output,
# and the weight cotangent of an inactive event is exactly zero even when the
# output cotangent is NaN/Inf. Active entries keep IEEE semantics.
# ---------------------------------------------------------------------------

_ISOLATION_KINDS = ["bool", "float", "int8", "weighted"]
_BAD_CYCLE = (np.nan, np.inf, -np.inf)


def _isolation_events(op, shape, kind, seed=0, batch=_BATCH):
    """Sparse events plus the poison set of event indices.

    Returns ``(x, xb, poison)`` where ``x`` has the native layout of ``op``,
    ``xb`` is the ``(batch, n_event)`` view of ``x`` and ``poison`` lists event
    indices whose weights may be poisoned: indices inactive in every batch
    and, for matrix ops, indices active only in batch 0 (so batch 0 sees the
    poison through an active event while the other batches must not).
    """
    rows, cols = shape
    n_event = rows if op in ("vm", "mr") else cols
    n_batch = batch if op in ("mm", "mr") else 1
    rng = np.random.RandomState(seed)
    active = rng.rand(n_batch, n_event) < 0.2
    inactive_all = np.arange(0, n_event, 3)
    active[:, inactive_all] = False
    poison = list(inactive_all)
    if n_batch > 1:
        batch0_only = np.arange(1, n_event, 7)
        active[:, batch0_only] = False
        active[0, batch0_only] = True
        poison += list(batch0_only)
    values = rng.choice(np.asarray([2.0, -1.0, 0.5]), size=active.shape)
    xb = np.where(active, values, 0.0)
    if kind == "bool":
        xb = active
    elif kind == "int8":
        xb = xb.astype(np.int8)  # 0.5 -> 0 would change activity
        xb = np.where(active & (xb == 0), 1, xb).astype(np.int8)
    else:
        xb = xb.astype(np.float32)
    x = xb[0] if op in ("vm", "mv") else (xb if op == "mr" else xb.T)
    return jnp.asarray(x), xb, np.asarray(sorted(set(poison)))


def _isolation_wrap(kind, x):
    return WeightedArray(x) if kind == "weighted" else BinaryArray(x)


def _isolation_factor(kind, xb, backward_algorithm="bptt"):
    """Per-event factor multiplying the weights (forward or weight cotangent)."""
    if kind == "weighted" or (backward_algorithm == "pp_prop" and kind != "int8"):
        return xb.astype(np.float64)
    return (xb != 0).astype(np.float64)


def _isolation_slots(matrix, op):
    """Canonical slot -> (event index, output index) for the logical op."""
    indptr = np.asarray(matrix._tcsr_indptr)
    indices = np.asarray(matrix._tcsr_indices)
    row_ids = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
    if op in ("vm", "mr"):
        return row_ids, indices
    return indices, row_ids


def _isolation_reference(op, matrix, data, xb, kind):
    """Select-based dense reference: inactive (batch, event) pairs add 0."""
    ev, out = _isolation_slots(matrix, op)
    rows, cols = matrix.shape
    n_out = cols if op in ("vm", "mr") else rows
    data = np.asarray(data, dtype=np.float64)
    w = np.broadcast_to(data, ev.shape) if data.size == 1 else data
    active = xb != 0
    factor = _isolation_factor(kind, xb)
    result = np.zeros((xb.shape[0], n_out), dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        for b in range(xb.shape[0]):
            sel = active[b, ev]
            np.add.at(result[b], out[sel], factor[b, ev[sel]] * w[sel])
    if op in ("vm", "mv"):
        return result[0]
    return result if op == "mr" else result.T


def _poisoned_data(matrix, op, poison, weights):
    """Canonical data with NaN/±Inf on every slot of a poisoned event."""
    if weights.startswith("homo"):
        bad = {"homo-nan": np.nan, "homo-inf": np.inf, "homo-neginf": -np.inf}[weights]
        return jnp.asarray([bad], dtype=matrix.data.dtype)
    ev, _ = _isolation_slots(matrix, op)
    data = np.asarray(matrix.data).copy()
    slots = np.flatnonzero(np.isin(ev, poison))
    assert slots.size > 0
    data[slots] = np.resize(np.asarray(_BAD_CYCLE, data.dtype), slots.size)
    return jnp.asarray(data)


def _check_isolation_forward(binary_backend, weights, op, kind, *, shape=_SHAPE, **tcsr_kwargs):
    homogeneous = weights.startswith("homo")
    with _explicit_int64_allowed():
        matrix = _tcsr(binary_backend, homogeneous, shape=shape, **tcsr_kwargs)
        x, xb, poison = _isolation_events(op, shape, kind)
        data = _poisoned_data(matrix, op, poison, weights)
        got = np.asarray(_apply(op, matrix.with_data(data), _isolation_wrap(kind, x)))
        expected = _isolation_reference(op, matrix, data, xb, kind)
    if op in ("vm", "mv") and not homogeneous:
        assert np.all(np.isfinite(got)), "inactive NaN/Inf weight reached the output"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4, equal_nan=True)


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmm_p))
@pytest.mark.parametrize("weights", ["hetero", "homo-nan", "homo-inf", "homo-neginf"])
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
@pytest.mark.parametrize("kind", _ISOLATION_KINDS)
def test_tcsr_inactive_nan_inf_weights_do_not_propagate(binary_backend, weights, op, kind):
    """Forward: inactive events add exactly 0 whatever their weights are."""
    _check_isolation_forward(binary_backend, weights, op, kind)


@requires_gpu_backend
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
@pytest.mark.parametrize("kind", _ISOLATION_KINDS)
def test_cuda_inactive_nan_inf_weights_heavy_rows(op, kind):
    """Heavy rows/columns route through the hybrid task (block) kernels."""
    _check_isolation_forward(
        "cuda_raw", "hetero", op, kind,
        shape=(300, 9000), heavy_rows=(3, 7, 200), heavy_cols=(11, 4000),
    )


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmv_p))
@pytest.mark.parametrize("kind", _ISOLATION_KINDS)
def test_tcsr_active_nan_weight_still_propagates(binary_backend, kind):
    """Regression guard: an active event times a NaN weight stays NaN."""
    with _explicit_int64_allowed():
        matrix = _tcsr(binary_backend, False)
        x, xb, _ = _isolation_events("vm", _SHAPE, kind)
        row = int(np.flatnonzero(xb[0])[0])
        indptr = np.asarray(matrix._tcsr_indptr)
        indices = np.asarray(matrix._tcsr_indices)
        assert indptr[row + 1] > indptr[row]
        data = np.asarray(matrix.data).copy()
        data[indptr[row]] = np.nan
        got = np.asarray(_apply("vm", matrix.with_data(jnp.asarray(data)), _isolation_wrap(kind, x)))
    assert np.isnan(got[indices[indptr[row]]])
    assert np.isfinite(np.delete(got, indices[indptr[row]])).all()


def _check_isolation_weight_cotangent(binary_backend, homogeneous, op, kind, backward_algorithm, bad):
    with _explicit_int64_allowed():
        matrix = _tcsr(binary_backend, homogeneous, backward_algorithm=backward_algorithm)
        x, xb, _ = _isolation_events(op, _SHAPE, kind)
        ev, out = _isolation_slots(matrix, op)
        active = xb != 0
        # Output (batch, o) reached by an active event of that batch.
        rows, cols = _SHAPE
        n_out = cols if op in ("vm", "mr") else rows
        reached = np.zeros((xb.shape[0], n_out), bool)
        for b in range(xb.shape[0]):
            reached[b, out[active[b, ev]]] = True
        rng = np.random.RandomState(7)
        ct_b = rng.normal(size=reached.shape)
        ct_b[~reached] = bad  # only inactive events ever meet these entries
        assert (~reached).any()
        ct = ct_b[0] if op in ("vm", "mv") else (ct_b if op == "mr" else ct_b.T)
        ct = jnp.asarray(ct, dtype=matrix.data.dtype)

        def f(data):
            return _apply(op, matrix.with_data(data), _isolation_wrap(kind, x))

        _, vjp = jax.vjp(f, matrix.data)
        (got,) = vjp(ct)
        got = np.asarray(got)
    factor = _isolation_factor(kind, xb, backward_algorithm)
    slot = np.zeros(ev.shape, np.float64)
    for b in range(xb.shape[0]):
        sel = active[b, ev]
        slot[sel] += factor[b, ev[sel]] * ct_b[b, out[sel]]
    expected = slot.sum(keepdims=True) if homogeneous else slot
    assert np.all(np.isfinite(got)), "NaN/Inf cotangent leaked into an inactive weight"
    if not homogeneous:
        inactive_slots = ~active[:, ev].any(axis=0)
        assert np.all(got[inactive_slots] == 0)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("binary_backend", _backend_params(binary_csrmm_p))
@pytest.mark.parametrize("homogeneous", [False, True], ids=["hetero", "homo"])
@pytest.mark.parametrize("op", ["vm", "mv", "mr", "mm"])
@pytest.mark.parametrize(
    "kind,backward_algorithm",
    [("bool", "bptt"), ("float", "bptt"), ("bool", "pp_prop"),
     ("float", "pp_prop"), ("weighted", "bptt")],
)
@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_tcsr_inactive_weight_cotangent_is_exactly_zero(
    binary_backend, homogeneous, op, kind, backward_algorithm, bad
):
    """Weight cotangent selects on activity: a NaN/Inf output cotangent that
    only meets inactive events leaves their weight cotangent exactly 0."""
    _check_isolation_weight_cotangent(
        binary_backend, homogeneous, op, kind, backward_algorithm, bad
    )


def _random_tcsr(**kwargs):
    from brainevent._tcsr.main import TCSR

    rng = np.random.default_rng(0)
    dense = (rng.random((40, 30)) < 0.3) * (rng.random((40, 30)) + 0.5)
    with _explicit_int64_allowed():
        return TCSR.fromdense(jnp.asarray(dense, jnp.float32), **kwargs), dense


@pytest.mark.parametrize("weighted", [False, True])
def test_vmap_events_on_axis_one(weighted):
    # Regression: an event batch on any axis but 0 raised NotImplementedError.
    matrix, dense = _random_tcsr()
    rng = np.random.default_rng(1)
    events = jnp.asarray(rng.random((30, 5)) > 0.5)
    left = jnp.asarray(rng.random((40, 5)) > 0.5)
    wrap = WeightedArray if weighted else BinaryArray
    if weighted:
        events, left = events.astype(jnp.float32) * 2, left.astype(jnp.float32) * 2
    got = jax.vmap(lambda e: matrix @ wrap(e), in_axes=1)(events)
    np.testing.assert_allclose(got, (dense @ np.asarray(events, np.float64)).T, atol=1e-4)
    got = jax.vmap(lambda e: wrap(e) @ matrix, in_axes=1)(left)
    np.testing.assert_allclose(got, (np.asarray(left, np.float64).T @ dense), atol=1e-4)
    batched = jnp.stack([events, events], axis=2)
    got = jax.vmap(lambda e: matrix @ wrap(e), in_axes=2)(batched)
    expected = dense @ np.asarray(events, np.float64)
    np.testing.assert_allclose(got, np.stack([expected, expected]), atol=1e-4)


@requires_gpu_backend
@pytest.mark.parametrize("batch", [3, 4, 600])
@pytest.mark.parametrize("side", ["left", "right"])
def test_cuda_raw_weight_grad_any_batch(batch, side):
    # Regression: the cuda_raw SDDMM weight-gradient kernel only takes batches
    # of 4..512, and had no fallback for the others.
    matrix, dense = _random_tcsr(binary_backend="cuda_raw")
    rng = np.random.default_rng(2)
    events = jnp.asarray(rng.random((batch, 40) if side == "left" else (30, batch)) > 0.5)
    ev = np.asarray(events, np.float64)

    def loss(data):
        m = matrix.with_data(data)
        out = BinaryArray(events) @ m if side == "left" else m @ BinaryArray(events)
        return jnp.sum(out * jnp.arange(out.size, dtype=out.dtype).reshape(out.shape) / out.size)

    grad_dense = jax.grad(
        lambda d: jnp.sum(
            ((jnp.asarray(ev) @ d) if side == "left" else (d @ jnp.asarray(ev)))
            * jnp.arange(batch * (30 if side == "left" else 40)).reshape(
                (batch, 30) if side == "left" else (40, batch)
            ) / (batch * (30 if side == "left" else 40))
        )
    )(jnp.asarray(dense))
    with _explicit_int64_allowed():
        grad = jax.grad(loss)(matrix.data)
        expected = matrix.with_data(grad).todense()
    np.testing.assert_allclose(
        np.asarray(expected), np.where(dense != 0, np.asarray(grad_dense), 0.), rtol=1e-4, atol=1e-3
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_zero_nnz_products_are_zero(weighted):
    # Regression: the CUDA tile MM (default on GPU for ``events @ M``) raised
    # for a matrix without connections.
    from brainevent._tcsr.main import TCSR

    with _explicit_int64_allowed():
        matrix = TCSR.fromdense(jnp.zeros((8, 6), jnp.float32))
    wrap = WeightedArray if weighted else BinaryArray
    dt = jnp.float32 if weighted else jnp.bool_
    cases = {
        (8,): lambda: matrix @ wrap(jnp.ones(6, dt)),
        (6,): lambda: wrap(jnp.ones(8, dt)) @ matrix,
        (8, 4): lambda: matrix @ wrap(jnp.ones((6, 4), dt)),
        (4, 6): lambda: wrap(jnp.ones((4, 8), dt)) @ matrix,
    }
    for shape, product in cases.items():
        out = product()
        assert out.shape == shape
        assert not np.any(np.asarray(out))


# ---------------------------------------------------------------------------
# Repeated (row, col) entries add, also when they outnumber the matrix
# elements (#223). cuSPARSE rejects a CSR descriptor with ``nnz > rows * cols``,
# so the ``cusparse`` backend must not hand such a matrix to it.
# ---------------------------------------------------------------------------

def _repeated_entry_tcsr(binary_backend, homogeneous, m, k, nse, seed=0):
    """TCSR of ``nse`` entries drawn with replacement, plus its dense sum."""
    from brainevent._tcsr.main import TCSR

    rng = np.random.default_rng(seed)
    rows = rng.integers(0, m, nse)
    cols = rng.integers(0, k, nse)
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    w = np.full(nse, 1.5) if homogeneous else rng.standard_normal(nse)
    dense = np.zeros((m, k))
    np.add.at(dense, (rows, cols), w)
    source = PlainCSR(
        (
            jnp.asarray([1.5] if homogeneous else w, jnp.float32),
            jnp.asarray(cols, jnp.int32),
            jnp.asarray(np.searchsorted(rows, np.arange(m + 1)), jnp.int64),
        ),
        shape=(m, k),
    )
    return TCSR.from_sorted_csr(source, binary_backend=binary_backend), dense


@pytest.mark.parametrize(
    "binary_backend",
    [b for b in ("jax", "cuda_raw", "cusparse")
     if b in binary_csrmv_p.available_backends(jax.default_backend()) or b == "jax"],
)
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("m,k,nse", [(2, 3, 8), (7, 5, 105), (6, 5, 20)])
def test_repeated_entries_add(binary_backend, homogeneous, weighted, m, k, nse):
    with _explicit_int64_allowed():
        matrix, dense = _repeated_entry_tcsr(binary_backend, homogeneous, m, k, nse)
        rng = np.random.default_rng(1)
        x = (rng.random(m) < 0.6) * rng.standard_normal(m)
        X = (rng.random((3, m)) < 0.6) * rng.standard_normal((3, m))
        wrap = WeightedArray if weighted else BinaryArray
        for events, out in (
            (x, wrap(jnp.asarray(x, jnp.float32)) @ matrix),
            (X, wrap(jnp.asarray(X, jnp.float32)) @ matrix),
        ):
            factor = events if weighted else (events != 0)
            np.testing.assert_allclose(np.asarray(out), factor @ dense, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("generator,jax_kernel,rank", [
    ("_binary_csrmv_cusparse_kernel", "_binary_csrmv_jax_kernel", 1),
    ("_binary_csrmm_cusparse_kernel", "_binary_csrmm_jax_kernel", 2),
])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("nse,uses_cusparse", [(6, True), (7, False)])
def test_cusparse_kernel_skips_cusparse_beyond_matrix_size(
    monkeypatch, generator, jax_kernel, rank, homogeneous, nse, uses_cusparse,
):
    # Static routing, checked without a GPU: a (2, 3) matrix storing at most 6
    # entries goes to cuSPARSE; storing more takes the exact pure-JAX kernel.
    sentinel = object()
    monkeypatch.setattr(binary, jax_kernel, lambda *a, **kw: sentinel)
    kernel = getattr(binary, generator)(
        jax.ShapeDtypeStruct((1 if homogeneous else nse,), jnp.float32),
        jax.ShapeDtypeStruct((2,) if rank == 1 else (2, 2), jnp.float32),
        (2, 3),
        True,
        mirror_enabled=False,
        indices_info=jax.ShapeDtypeStruct((nse,), jnp.int32),
        outs=[jax.ShapeDtypeStruct((3,) if rank == 1 else (2, 3), jnp.float32)],
    )
    assert (kernel is not sentinel) == uses_cusparse
