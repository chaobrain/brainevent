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

"""Test the TCSR float BN service contract and batching rules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._tcsr import float as float_ops


def _structure() -> tuple[jax.Array, ...]:
    """Create a small sorted TCSR structure and its tile metadata."""
    with jax.enable_x64():
        return (
            jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
            jnp.asarray([0, 2, 1, 2], dtype=jnp.int32),
            jnp.asarray([0, 2, 4], dtype=jnp.int64),
            jnp.asarray([0, 2, 1, 2], dtype=jnp.uint16),
            jnp.asarray([[0, 2], [0, 2]], dtype=jnp.int32),
        )


@pytest.mark.parametrize(
    ("operand", "expected_exception", "message"),
    [
        ("local_rank", ValueError, "local_targets"),
        ("local_length", ValueError, "local_targets"),
        ("local_dtype", TypeError, "uint16"),
        ("offset_shape", ValueError, "tile_offsets"),
        ("offset_dtype", TypeError, "int32"),
    ],
)
def test_float_primitives_validate_tile_metadata(
    operand,
    expected_exception,
    message,
):
    """Reject malformed canonical tile metadata before primitive binding."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()
    if operand == "local_rank":
        local_targets = local_targets.reshape(2, 2)
    elif operand == "local_length":
        local_targets = local_targets[:-1]
    elif operand == "local_dtype":
        local_targets = local_targets.astype(jnp.int32)
    elif operand == "offset_shape":
        tile_offsets = tile_offsets[:, :-1]
    else:
        tile_offsets = np.asarray(tile_offsets, dtype=np.int64)

    with pytest.raises(expected_exception, match=message):
        float_ops.csrmv_p_call(
            weights,
            indices,
            indptr,
            jnp.ones(3, dtype=jnp.float32),
            local_targets,
            tile_offsets,
            shape=(2, 3),
            transpose=False,
            backend="jax_raw",
        )


@pytest.mark.parametrize("operation", ["mv", "mm"])
def test_float_primitives_reject_invalid_weight_length(operation):
    """Require either one homogeneous weight or one weight per entry."""
    _, indices, indptr, local_targets, tile_offsets = _structure()
    weights = jnp.ones(2, dtype=jnp.float32)
    dense = (
        jnp.ones(3, dtype=jnp.float32)
        if operation == "mv"
        else jnp.ones((4, 3), dtype=jnp.float32)
    )
    primitive = (
        float_ops.csrmv_p_call if operation == "mv" else float_ops.csrmm_p_call
    )

    with pytest.raises(ValueError, match="one value per nonzero"):
        primitive(
            weights,
            indices,
            indptr,
            dense,
            local_targets,
            tile_offsets,
            shape=(2, 3),
            transpose=False,
            backend="jax_raw",
        )


def test_csrmv_xw_cuda_ffi_keeps_metadata_operands(monkeypatch):
    """Keep canonical metadata visible to the WPR FFI validator."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()
    vector = jnp.ones(2, dtype=jnp.float32)
    calls = []

    monkeypatch.setattr(float_ops, "load_cuda_file", lambda *args, **kwargs: None)

    def fake_ffi_call(name, out_info):
        def invoke(*args):
            calls.append((name, out_info, args))
            return (jnp.zeros(3, dtype=jnp.float32),)

        return invoke

    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi_call)
    kernel = float_ops._csrmv_cuda_kernel(
        jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        True,
        shape=(2, 3),
        outs=(jax.ShapeDtypeStruct((3,), weights.dtype),),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        local_targets_info=jax.ShapeDtypeStruct(
            local_targets.shape, local_targets.dtype
        ),
        tile_offsets_info=jax.ShapeDtypeStruct(
            tile_offsets.shape, tile_offsets.dtype
        ),
    )
    kernel(weights, indices, indptr, vector, local_targets, tile_offsets)

    expected_args = (
        weights,
        indices,
        indptr,
        local_targets,
        tile_offsets,
        vector,
    )
    assert len(calls[0][2]) == len(expected_args)
    assert all(
        actual is expected
        for actual, expected in zip(calls[0][2], expected_args)
    )


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_csrmv_numba_generator_matches_dense(
    monkeypatch,
    transpose,
    homogeneous,
):
    """Exercise each compiled MV algorithm body against dense algebra."""
    import numba

    weights, indices, indptr, local_targets, tile_offsets = _structure()
    if homogeneous:
        weights = jnp.asarray([2.0], dtype=jnp.float32)
    neurons = 2 if transpose else 3
    vector = jnp.arange(1, neurons + 1, dtype=jnp.float32)
    output_size = 3 if transpose else 2

    monkeypatch.setattr(numba, "njit", lambda *args, **kwargs: lambda fn: fn)
    monkeypatch.setattr(numba, "prange", range)

    def fake_numba_kernel(function, *, outs):
        def invoke(*args):
            output = np.empty(outs[0].shape, dtype=outs[0].dtype)
            function(*(np.asarray(arg) for arg in args), output)
            return (jnp.asarray(output),)

        return invoke

    monkeypatch.setattr(float_ops, "numba_kernel", fake_numba_kernel)
    kernel = float_ops._csrmv_numba_kernel_generator(
        jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        transpose,
        outs=(jax.ShapeDtypeStruct((output_size,), weights.dtype),),
    )
    actual = kernel(
        weights,
        indices,
        indptr,
        vector,
        local_targets,
        tile_offsets,
    )[0]

    expanded = (
        np.full(indices.size, weights[0])
        if homogeneous
        else np.asarray(weights)
    )
    dense = np.zeros((2, 3), dtype=np.float32)
    row_ids = np.repeat(np.arange(2), np.diff(np.asarray(indptr)))
    np.add.at(dense, (row_ids, np.asarray(indices)), expanded)
    expected = dense.T @ vector if transpose else dense @ vector
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("homogeneous", [False, True])
def test_csrmm_numba_generator_matches_dense(
    monkeypatch,
    transpose,
    homogeneous,
):
    """Exercise each compiled BN MM algorithm body against dense algebra."""
    import numba

    weights, indices, indptr, local_targets, tile_offsets = _structure()
    if homogeneous:
        weights = jnp.asarray([2.0], dtype=jnp.float32)
    neurons = 2 if transpose else 3
    matrix_bn = jnp.arange(1, 2 * neurons + 1, dtype=jnp.float32).reshape(
        2, neurons
    )
    output_size = 3 if transpose else 2

    monkeypatch.setattr(numba, "njit", lambda *args, **kwargs: lambda fn: fn)
    monkeypatch.setattr(numba, "prange", range)

    def fake_numba_kernel(function, *, outs):
        def invoke(*args):
            output = np.empty(outs[0].shape, dtype=outs[0].dtype)
            function(*(np.asarray(arg) for arg in args), output)
            return (jnp.asarray(output),)

        return invoke

    monkeypatch.setattr(float_ops, "numba_kernel", fake_numba_kernel)
    kernel = float_ops._csrmm_numba_kernel_generator(
        jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        transpose,
        outs=(jax.ShapeDtypeStruct((2, output_size), weights.dtype),),
    )
    actual = kernel(
        weights,
        indices,
        indptr,
        matrix_bn,
        local_targets,
        tile_offsets,
    )[0]

    expanded = (
        np.full(indices.size, weights[0])
        if homogeneous
        else np.asarray(weights)
    )
    dense = np.zeros((2, 3), dtype=np.float32)
    row_ids = np.repeat(np.arange(2), np.diff(np.asarray(indptr)))
    np.add.at(dense, (row_ids, np.asarray(indices)), expanded)
    expected = matrix_bn @ dense if transpose else matrix_bn @ dense.T
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "generator",
    [float_ops._csrmv_cuda_kernel, float_ops._csrmm_cuda_kernel],
)
@pytest.mark.parametrize("invalid_operand", ["indptr", "local", "offsets"])
def test_float_cuda_generators_reject_invalid_metadata_dtype(
    generator,
    invalid_operand,
):
    """Validate every fixed-width CUDA metadata dtype before compilation."""
    indptr_dtype = jnp.int32 if invalid_operand == "indptr" else jnp.int64
    local_dtype = jnp.int32 if invalid_operand == "local" else jnp.uint16
    offsets_dtype = jnp.int64 if invalid_operand == "offsets" else jnp.int32

    with pytest.raises(TypeError):
        generator(
            jax.ShapeDtypeStruct((4,), jnp.float32),
            False,
            shape=(2, 3),
            outs=(jax.ShapeDtypeStruct((2,), jnp.float32),),
            indices_info=jax.ShapeDtypeStruct((4,), jnp.int32),
            indptr_info=jax.ShapeDtypeStruct((3,), indptr_dtype),
            local_targets_info=jax.ShapeDtypeStruct((4,), local_dtype),
            tile_offsets_info=jax.ShapeDtypeStruct((2, 2), offsets_dtype),
        )


@pytest.mark.parametrize(
    "batching_rule",
    [float_ops._csrmv_batching, float_ops._csrmm_batching],
)
def test_float_batching_without_mapped_axis_uses_general_rule(
    monkeypatch,
    batching_rule,
):
    """Delegate unbatched transformation calls to the common rule."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()
    dense = jnp.ones(3, dtype=jnp.float32)
    if batching_rule is float_ops._csrmm_batching:
        dense = dense[None, :]
    sentinel = object()
    monkeypatch.setattr(
        float_ops,
        "general_batching_rule",
        lambda *args, **kwargs: sentinel,
    )

    actual = batching_rule(
        (weights, indices, indptr, dense, local_targets, tile_offsets),
        (None,) * 6,
        shape=(2, 3),
        transpose=False,
        backend="jax_raw",
    )

    assert actual is sentinel


def test_csrmm_batching_rejects_non_nested_dense_rank():
    """Reject a mapped MM operand unless it contains outer and BN axes."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()

    with pytest.raises(ValueError, match="rank three"):
        float_ops._csrmm_batching(
            (
                weights,
                indices,
                indptr,
                jnp.ones((4, 3), dtype=jnp.float32),
                local_targets,
                tile_offsets,
            ),
            (None, None, None, 0, None, None),
            shape=(2, 3),
            transpose=False,
            backend="jax_raw",
        )


@pytest.mark.parametrize(
    "benchmark_data",
    [float_ops._csrmv_benchmark_data, float_ops._csrmm_benchmark_data],
)
def test_float_benchmark_data_uses_six_operand_bn_abi(benchmark_data):
    """Keep generated benchmark inputs aligned with the float primitive ABI."""
    configs = benchmark_data(platform="cpu")

    assert len(configs) == 4
    for config in configs:
        assert len(config.args) == 6
        assert config.args[4].dtype == jnp.uint16
        assert config.args[5].dtype == jnp.int32
        if benchmark_data is float_ops._csrmm_benchmark_data:
            assert config.args[3].shape[0] == 10


@pytest.mark.parametrize("transpose", [False, True])
def test_csrmm_p_call_uses_bn_input_and_output(monkeypatch, transpose):
    """Keep batch on axis zero for both sparse multiplication directions."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()
    neurons = 2 if transpose else 3
    output_neurons = 3 if transpose else 2
    operand_bn = jnp.arange(4 * neurons, dtype=jnp.float32).reshape(4, neurons)
    calls = []

    def fake_primitive(*args, **kwargs):
        calls.append((args, kwargs))
        return (jnp.zeros(kwargs["outs"][0].shape, dtype=jnp.float32),)

    monkeypatch.setattr(float_ops, "csrmm_p", fake_primitive)
    result = float_ops.csrmm_p_call(
        weights,
        indices,
        indptr,
        operand_bn,
        local_targets,
        tile_offsets,
        shape=(2, 3),
        transpose=transpose,
        backend="sentinel",
    )[0]

    assert result.shape == (4, output_neurons)
    assert calls[0][0][3].shape == (4, neurons)
    assert calls[0][0][4] is local_targets
    assert calls[0][0][5] is tile_offsets


@pytest.mark.parametrize("mapped_axis", [0, 1])
@pytest.mark.parametrize("transpose", [False, True])
def test_csrmv_batching_moves_mapped_axis_to_bn(
    monkeypatch,
    mapped_axis,
    transpose,
):
    """Normalize every mapped vector axis to the leading batch dimension."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()
    neurons = 2 if transpose else 3
    output_neurons = 3 if transpose else 2
    vector_bn = jnp.arange(4 * neurons, dtype=jnp.float32).reshape(4, neurons)
    batched_vector = vector_bn if mapped_axis == 0 else vector_bn.T
    calls = []

    def fake_csrmm(*args, **kwargs):
        calls.append((args, kwargs))
        return (jnp.zeros((4, output_neurons), dtype=jnp.float32),)

    monkeypatch.setattr(float_ops, "csrmm_p_call", fake_csrmm)
    result, result_axes = float_ops._csrmv_batching(
        (weights, indices, indptr, batched_vector, local_targets, tile_offsets),
        (None, None, None, mapped_axis, None, None),
        shape=(2, 3),
        transpose=transpose,
        backend="sentinel",
    )

    np.testing.assert_array_equal(calls[0][0][3], vector_bn)
    assert calls[0][0][4] is local_targets
    assert calls[0][0][5] is tile_offsets
    assert result[0].shape == (4, output_neurons)
    assert result_axes == [0]


@pytest.mark.parametrize("mapped_axis", [0, 1, 2])
@pytest.mark.parametrize("transpose", [False, True])
def test_csrmm_batching_flattens_outer_and_inner_bn_batches(
    monkeypatch,
    mapped_axis,
    transpose,
):
    """Flatten nested batches only after moving the mapped axis to zero."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()
    neurons = 2 if transpose else 3
    output_neurons = 3 if transpose else 2
    canonical = jnp.arange(5 * 4 * neurons, dtype=jnp.float32).reshape(
        5, 4, neurons
    )
    batched_operand = jnp.moveaxis(canonical, 0, mapped_axis)
    calls = []

    def fake_csrmm(*args, **kwargs):
        calls.append((args, kwargs))
        return (jnp.zeros((20, output_neurons), dtype=jnp.float32),)

    monkeypatch.setattr(float_ops, "csrmm_p_call", fake_csrmm)
    result, result_axes = float_ops._csrmm_batching(
        (weights, indices, indptr, batched_operand, local_targets, tile_offsets),
        (None, None, None, mapped_axis, None, None),
        shape=(2, 3),
        transpose=transpose,
        backend="sentinel",
    )

    np.testing.assert_array_equal(calls[0][0][3], canonical.reshape(20, neurons))
    assert result[0].shape == (5, 4, output_neurons)
    assert result_axes == [0]


@pytest.mark.parametrize("batching_rule", [
    float_ops._csrmv_batching,
    float_ops._csrmm_batching,
])
def test_float_batching_rejects_mapped_sparse_metadata(batching_rule):
    """Reject mapped structure instead of silently changing TCSR metadata."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()
    dense = jnp.ones((4, 3), dtype=jnp.float32)

    with pytest.raises(NotImplementedError, match="dense operand"):
        batching_rule(
            (weights, indices, indptr, dense, local_targets, tile_offsets),
            (None, 0, None, 0, None, None),
            shape=(2, 3),
            transpose=False,
            backend="sentinel",
        )


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("mapped_axis", [0, 1])
def test_csrmv_vmap_normalizes_real_mapped_axes(transpose, mapped_axis):
    """Execute MV batching through the registered BN MM primitive path."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()
    dense = jnp.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])
    neurons = dense.shape[0] if transpose else dense.shape[1]
    vectors_bn = jnp.arange(4 * neurons, dtype=jnp.float32).reshape(4, neurons)
    vectors = vectors_bn if mapped_axis == 0 else vectors_bn.T

    def operation(vector):
        return float_ops.csrmv(
            weights,
            indices,
            indptr,
            vector,
            shape=dense.shape,
            local_targets=local_targets,
            tile_offsets=tile_offsets,
            transpose=transpose,
            backend="jax_raw",
        )

    with jax.enable_x64():
        actual = jax.vmap(operation, in_axes=mapped_axis)(vectors)
    expected = vectors_bn @ dense if transpose else vectors_bn @ dense.T
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("mapped_axis", [0, 1, 2])
def test_csrmm_vmap_normalizes_real_nested_batch_axes(transpose, mapped_axis):
    """Execute nested batching with outer batch restored on output axis zero."""
    weights, indices, indptr, local_targets, tile_offsets = _structure()
    dense = jnp.asarray([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])
    neurons = dense.shape[0] if transpose else dense.shape[1]
    canonical = jnp.arange(5 * 4 * neurons, dtype=jnp.float32).reshape(
        5, 4, neurons
    )
    operand = jnp.moveaxis(canonical, 0, mapped_axis)

    def operation(matrix_bn):
        return float_ops.csrmm(
            weights,
            indices,
            indptr,
            matrix_bn,
            shape=dense.shape,
            local_targets=local_targets,
            tile_offsets=tile_offsets,
            transpose=transpose,
            backend="jax_raw",
        )

    with jax.enable_x64():
        actual = jax.vmap(operation, in_axes=mapped_axis)(operand)
    expected = canonical @ (dense if transpose else dense.T)
    np.testing.assert_allclose(actual, expected)
