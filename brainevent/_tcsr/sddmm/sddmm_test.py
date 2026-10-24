"""Test the formal TCSR sampled-gradient API and CUDA registration."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from brainevent._tcsr import binary
from brainevent._tcsr.sddmm import sddmm


_FORMAL_NAMES = (
    "tcsr_sddmm_dweight_binary",
    "tcsr_sddmm_dweight_float",
    "tcsr_sddmv_dweight_binary",
    "tcsr_sddmv_dweight_float",
)

_EXPERIMENTAL_NAMES = (
    "csr_sddmm_dweight",
    "csr_sddmm_dweight_batchn_float",
    "csr_sddmm_dweight_single",
    "csr_sddmm_dweight_single_float",
)


def _recording_ffi_call(calls):
    """Return an FFI stand-in that records targets and returns shaped arrays."""

    def ffi_call(target, out_info, **ffi_kwargs):
        def invoke(*args, **attrs):
            calls.append((target, out_info, ffi_kwargs, args, attrs))
            outputs = tuple(
                jnp.zeros(info.shape, dtype=info.dtype) for info in out_info
            )
            return outputs

        return invoke

    return ffi_call


def _batched_inputs(event_dtype):
    events = jnp.array(
        [[1, 0], [0, 1], [1, 1], [0, 0]], dtype=event_dtype
    )
    ct = jnp.ones((4, 3), dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    local_targets = jnp.array([0, 2, 1], dtype=jnp.uint16)
    tile_offsets = jnp.array([[0, 2], [0, 1]], dtype=jnp.int32)
    return events, ct, indices, indptr, local_targets, tile_offsets


def _vector_inputs(event_dtype):
    event = jnp.array([1, 0], dtype=event_dtype)
    ct = jnp.ones((3,), dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    local_targets = jnp.array([0, 2, 1], dtype=jnp.uint16)
    tile_offsets = jnp.array([[0, 2], [0, 1]], dtype=jnp.int32)
    return event, ct, indices, indptr, local_targets, tile_offsets


def test_sddmm_package_owns_formal_api_used_by_binary():
    """Keep sampled-gradient ownership out of the binary implementation."""
    import brainevent._tcsr.sddmm as sddmm_package

    for name in _FORMAL_NAMES:
        implementation = getattr(sddmm, name)
        assert getattr(sddmm_package, name) is implementation
        assert getattr(binary, name) is implementation
        assert name not in binary.__all__

    for name in _EXPERIMENTAL_NAMES:
        assert not hasattr(sddmm_package, name)
        assert not hasattr(binary, name)


@pytest.mark.parametrize(
    (
        "function_name",
        "event_dtype",
        "source_name",
        "module_name",
        "target_name",
    ),
    (
        (
            "tcsr_sddmm_dweight_binary",
            jnp.bool_,
            "sddmm_binary.cu",
            "tcsr_sddmm_binary",
            "tcsr_sddmm_binary.tcsr_sddmm_dweight_binary_f32_bool_t",
        ),
        (
            "tcsr_sddmm_dweight_binary",
            jnp.float32,
            "sddmm_binary.cu",
            "tcsr_sddmm_binary",
            "tcsr_sddmm_binary.tcsr_sddmm_dweight_binary_f32_float_t",
        ),
        (
            "tcsr_sddmm_dweight_float",
            jnp.float32,
            "sddmm_float.cu",
            "tcsr_sddmm_float",
            "tcsr_sddmm_float.tcsr_sddmm_dweight_float_f32_t",
        ),
        (
            "tcsr_sddmv_dweight_binary",
            jnp.bool_,
            "sddmv_binary.cu",
            "tcsr_sddmv_binary",
            "tcsr_sddmv_binary.tcsr_sddmv_dweight_binary_f32_bool_t",
        ),
        (
            "tcsr_sddmv_dweight_binary",
            jnp.float32,
            "sddmv_binary.cu",
            "tcsr_sddmv_binary",
            "tcsr_sddmv_binary.tcsr_sddmv_dweight_binary_f32_float_t",
        ),
        (
            "tcsr_sddmv_dweight_float",
            jnp.float32,
            "sddmv_float.cu",
            "tcsr_sddmv_float",
            "tcsr_sddmv_float.tcsr_sddmv_dweight_float_f32_t",
        ),
    ),
)
def test_formal_sddmm_api_registers_owned_cuda_target(
    monkeypatch,
    function_name,
    event_dtype,
    source_name,
    module_name,
    target_name,
):
    """Load each owned CUDA source and invoke its formally named target."""
    loads = []
    calls = []

    def fake_load_cuda_file(path, *, name):
        loads.append((Path(path), name))
        return object()

    monkeypatch.setattr(sddmm, "_SDDMM_BINARY_CUDA_MODULE", None)
    monkeypatch.setattr(sddmm, "_SDDMM_FLOAT_CUDA_MODULE", None)
    monkeypatch.setattr(sddmm, "_SDDMV_BINARY_CUDA_MODULE", None)
    monkeypatch.setattr(sddmm, "_SDDMV_FLOAT_CUDA_MODULE", None)
    monkeypatch.setattr(sddmm, "load_cuda_file", fake_load_cuda_file)
    monkeypatch.setattr(jax.ffi, "ffi_call", _recording_ffi_call(calls))

    inputs = (
        _batched_inputs(event_dtype)
        if "sddmm" in function_name
        else _vector_inputs(event_dtype)
    )
    result = getattr(sddmm, function_name)(*inputs, transpose=True)

    assert loads == [(Path(sddmm.__file__).with_name(source_name), module_name)]
    assert calls[0][0] == target_name
    assert result.shape == inputs[2].shape


@pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="requires a JAX GPU backend",
)
def test_formal_sddmm_api_matches_hand_computed_gradients():
    """Match binary and value-preserving MM/MV gradients on CUDA."""
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    local_targets = jnp.array([0, 2, 1], dtype=jnp.uint16)
    tile_offsets = jnp.array([[0, 2], [0, 1]], dtype=jnp.int32)
    ct_mm = jnp.array(
        [
            [2, 3, 5],
            [7, 11, 13],
            [17, 19, 23],
            [29, 31, 37],
        ],
        dtype=jnp.float32,
    )
    events_mm = jnp.array(
        [[2, 0], [0, -1], [3, 4], [-2, 0]], dtype=jnp.float32
    )
    common_mm = (indices, indptr, local_targets, tile_offsets)

    binary_mm = jax.jit(
        sddmm.tcsr_sddmm_dweight_binary,
        static_argnames=("transpose",),
    )(events_mm, ct_mm, *common_mm, transpose=True)
    float_mm = jax.jit(
        sddmm.tcsr_sddmm_dweight_float,
        static_argnames=("transpose",),
    )(events_mm, ct_mm, *common_mm, transpose=True)
    event_mv = jnp.array([2, -1], dtype=jnp.float32)
    ct_mv = jnp.array([2, 3, 5], dtype=jnp.float32)
    common_mv = (indices, indptr, local_targets, tile_offsets)
    binary_mv = jax.jit(
        sddmm.tcsr_sddmv_dweight_binary,
        static_argnames=("transpose",),
    )(event_mv, ct_mv, *common_mv, transpose=True)
    float_mv = jax.jit(
        sddmm.tcsr_sddmv_dweight_float,
        static_argnames=("transpose",),
    )(event_mv, ct_mv, *common_mv, transpose=True)

    # Binary gradients use the ``x != 0`` mask: negative events are active.
    assert jnp.allclose(binary_mm, jnp.array([48, 65, 30], jnp.float32))
    assert jnp.allclose(float_mm, jnp.array([-3, 5, 65], jnp.float32))
    assert jnp.allclose(binary_mv, jnp.array([2, 5, 3], jnp.float32))
    assert jnp.allclose(float_mv, jnp.array([4, 10, -3], jnp.float32))


# ============================================================================
# Dtype routing and CUDA execution (formerly sddmm/test/sddmm_dtype_test.py).
# ============================================================================


@pytest.fixture
def _enable_x64():
    """Enable x64 while exercising float64 SDDMM contracts."""
    previous_explicit = jax.config.jax_explicit_x64_dtypes
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_explicit_x64_dtypes", "allow")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_explicit_x64_dtypes", previous_explicit)


def _inputs(function_name, event_dtype, value_dtype):
    """Build valid MM or MV operands for one dtype combination."""
    if "sddmm" in function_name:
        event = jnp.ones((4, 2), dtype=event_dtype)
        ct = jnp.ones((4, 3), dtype=value_dtype)
    else:
        event = jnp.ones((2,), dtype=event_dtype)
        ct = jnp.ones((3,), dtype=value_dtype)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    local_targets = jnp.array([0, 2, 1], dtype=jnp.uint16)
    tile_offsets = jnp.array([[0, 2], [0, 1]], dtype=jnp.int32)
    return event, ct, indices, indptr, local_targets, tile_offsets


_ROUTING_CASES = (
    ("tcsr_sddmm_dweight_binary", jnp.bool_, jnp.float32, "binary_f32_bool_t"),
    ("tcsr_sddmm_dweight_binary", jnp.float32, jnp.float32, "binary_f32_float_t"),
    ("tcsr_sddmm_dweight_binary", jnp.bool_, jnp.float64, "binary_f64_bool_t"),
    ("tcsr_sddmm_dweight_binary", jnp.int8, jnp.float64, "binary_f64_int8_t"),
    ("tcsr_sddmm_dweight_binary", jnp.float32, jnp.float64, "binary_f64_float_t"),
    ("tcsr_sddmm_dweight_binary", jnp.float64, jnp.float64, "binary_f64_double_t"),
    ("tcsr_sddmv_dweight_binary", jnp.bool_, jnp.float32, "binary_f32_bool_t"),
    ("tcsr_sddmv_dweight_binary", jnp.float32, jnp.float32, "binary_f32_float_t"),
    ("tcsr_sddmv_dweight_binary", jnp.bool_, jnp.float64, "binary_f64_bool_t"),
    ("tcsr_sddmv_dweight_binary", jnp.int8, jnp.float64, "binary_f64_int8_t"),
    ("tcsr_sddmv_dweight_binary", jnp.float32, jnp.float64, "binary_f64_float_t"),
    ("tcsr_sddmv_dweight_binary", jnp.float64, jnp.float64, "binary_f64_double_t"),
    # Event dtypes without a native kernel use their ``!= 0`` mask (bool ABI).
    ("tcsr_sddmm_dweight_binary", jnp.int8, jnp.float32, "binary_f32_bool_t"),
    ("tcsr_sddmm_dweight_binary", jnp.float64, jnp.float32, "binary_f32_bool_t"),
    ("tcsr_sddmm_dweight_binary", jnp.int16, jnp.float64, "binary_f64_bool_t"),
    ("tcsr_sddmv_dweight_binary", jnp.float16, jnp.float32, "binary_f32_bool_t"),
    ("tcsr_sddmv_dweight_binary", jnp.bfloat16, jnp.float64, "binary_f64_bool_t"),
    ("tcsr_sddmv_dweight_binary", jnp.uint8, jnp.float32, "binary_f32_bool_t"),
    ("tcsr_sddmm_dweight_float", jnp.float32, jnp.float32, "float_f32_t"),
    ("tcsr_sddmm_dweight_float", jnp.float64, jnp.float64, "float_f64_t"),
    ("tcsr_sddmv_dweight_float", jnp.float32, jnp.float32, "float_f32_t"),
    ("tcsr_sddmv_dweight_float", jnp.float64, jnp.float64, "float_f64_t"),
)


@pytest.mark.usefixtures("_enable_x64")
@pytest.mark.parametrize(
    ("function_name", "event_dtype", "value_dtype", "target_suffix"),
    _ROUTING_CASES,
)
def test_sddmm_dtype_pair_selects_exact_ffi_target(
    monkeypatch,
    function_name,
    event_dtype,
    value_dtype,
    target_suffix,
):
    """Route every supported dtype pair to its exact CUDA ABI."""
    loads = []
    calls = []

    def fake_load_cuda_file(path, *, name):
        loads.append((Path(path), name))
        return object()

    for cache_name in (
        "_SDDMM_BINARY_CUDA_MODULE",
        "_SDDMM_FLOAT_CUDA_MODULE",
        "_SDDMV_BINARY_CUDA_MODULE",
        "_SDDMV_FLOAT_CUDA_MODULE",
    ):
        monkeypatch.setattr(sddmm, cache_name, None)
    monkeypatch.setattr(sddmm, "load_cuda_file", fake_load_cuda_file)
    monkeypatch.setattr(jax.ffi, "ffi_call", _recording_ffi_call(calls))

    inputs = _inputs(function_name, event_dtype, value_dtype)
    result = getattr(sddmm, function_name)(*inputs, transpose=True)

    operation = "sddmm" if "sddmm" in function_name else "sddmv"
    family = "binary" if "binary" in function_name else "float"
    module = f"tcsr_{operation}_{family}"
    assert loads == [
        (Path(sddmm.__file__).with_name(f"{operation}_{family}.cu"), module)
    ]
    assert calls[0][0] == f"{module}.tcsr_{operation}_dweight_{target_suffix}"
    assert result.dtype == jnp.dtype(value_dtype)
    assert result.shape == inputs[2].shape


@pytest.mark.usefixtures("_enable_x64")
@pytest.mark.parametrize(
    ("function_name", "event_dtype", "value_dtype"),
    (
        ("tcsr_sddmm_dweight_binary", jnp.bool_, jnp.float16),
        ("tcsr_sddmv_dweight_binary", jnp.complex64, jnp.float32),
        ("tcsr_sddmm_dweight_float", jnp.float32, jnp.float64),
        ("tcsr_sddmv_dweight_float", jnp.float64, jnp.float32),
    ),
)
def test_sddmm_rejects_unsupported_dtype_pair_before_cuda_load(
    monkeypatch,
    function_name,
    event_dtype,
    value_dtype,
):
    """Reject unsupported and mixed dtype pairs before loading CUDA."""

    def unexpected_load(*args, **kwargs):
        raise AssertionError("CUDA must not load for an invalid dtype pair")

    monkeypatch.setattr(sddmm, "load_cuda_file", unexpected_load)
    inputs = _inputs(function_name, event_dtype, value_dtype)

    with pytest.raises(TypeError):
        getattr(sddmm, function_name)(*inputs, transpose=True)


def _binary_events(dtype, *, batched):
    """Return events with a fixed signed activity pattern (negatives are active)."""
    if jnp.dtype(dtype) == jnp.dtype(jnp.bool_):
        values = [[True, False], [False, False], [True, True], [False, False]]
    else:
        values = [[2, 0], [0, -1], [3, 4], [-2, 0]]
    if batched:
        return jnp.array(values, dtype=dtype)
    vector = [True, False] if jnp.dtype(dtype) == jnp.dtype(jnp.bool_) else [2, -1]
    return jnp.array(vector, dtype=dtype)


def _numeric_inputs(event, value_dtype):
    """Build hand-checkable numeric operands for MM or MV."""
    if event.ndim == 2:
        ct = jnp.array(
            [[2, 3, 5], [7, 11, 13], [17, 19, 23], [29, 31, 37]],
            dtype=value_dtype,
        )
    else:
        ct = jnp.array([2, 3, 5], dtype=value_dtype)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    local_targets = jnp.array([0, 2, 1], dtype=jnp.uint16)
    tile_offsets = jnp.array([[0, 2], [0, 1]], dtype=jnp.int32)
    return ct, indices, indptr, local_targets, tile_offsets


@pytest.mark.usefixtures("_enable_x64")
@pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="requires a JAX GPU backend",
)
@pytest.mark.parametrize("event_dtype", (jnp.bool_, jnp.int8, jnp.float32, jnp.float64))
@pytest.mark.parametrize("operation", ("sddmm", "sddmv"))
def test_binary_f64_cuda_uses_nonzero_event_bits(operation, event_dtype):
    """Compute f64 binary gradients from ``x != 0`` event bits on CUDA."""
    batched = operation == "sddmm"
    event = _binary_events(event_dtype, batched=batched)
    inputs = _numeric_inputs(event, jnp.float64)

    result = getattr(sddmm, f"tcsr_{operation}_dweight_binary")(
        event, *inputs, transpose=True
    )

    if jnp.dtype(event_dtype) == jnp.dtype(jnp.bool_):
        expected = [19, 28, 19] if batched else [2, 5, 0]
    else:
        # Negative events are active: masks [[1,0],[0,1],[1,1],[1,0]] / [1,1].
        expected = [48, 65, 30] if batched else [2, 5, 3]
    assert result.dtype == jnp.float64
    assert jnp.allclose(result, jnp.array(expected, dtype=jnp.float64))


@pytest.mark.usefixtures("_enable_x64")
@pytest.mark.skipif(
    not any(device.platform == "gpu" for device in jax.devices()),
    reason="requires a JAX GPU backend",
)
@pytest.mark.parametrize("value_dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize("operation", ("sddmm", "sddmv"))
def test_float_cuda_preserves_matching_precision(operation, value_dtype):
    """Preserve signed float values at matching precision on CUDA."""
    batched = operation == "sddmm"
    event = jnp.array(
        [[2, 0], [0, -1], [3, 4], [-2, 0]], dtype=value_dtype
    )
    if not batched:
        event = jnp.array([2, -1], dtype=value_dtype)
    inputs = _numeric_inputs(event, value_dtype)

    result = getattr(sddmm, f"tcsr_{operation}_dweight_float")(
        event, *inputs, transpose=True
    )

    expected = [-3, 5, 65] if batched else [4, 10, -3]
    assert result.dtype == jnp.dtype(value_dtype)
    assert jnp.allclose(result, jnp.array(expected, dtype=value_dtype))
