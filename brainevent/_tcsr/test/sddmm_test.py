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

    assert jnp.allclose(binary_mm, jnp.array([19, 28, 19], jnp.float32))
    assert jnp.allclose(float_mm, jnp.array([-3, 5, 65], jnp.float32))
    assert jnp.allclose(binary_mv, jnp.array([2, 5, 0], jnp.float32))
    assert jnp.allclose(float_mv, jnp.array([4, 10, -3], jnp.float32))
