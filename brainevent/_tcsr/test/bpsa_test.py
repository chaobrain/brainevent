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

"""Test the explicitly imported experimental TCSR BPSA package."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _masked_operands():
    return {
        "weights": jnp.asarray([2.0], dtype=jnp.float32),
        "indptr": jnp.asarray([0, 1], dtype=jnp.int32),
        "local_targets": jnp.asarray([0], dtype=jnp.uint16),
        "tile_offsets": jnp.asarray([[0, 1]], dtype=jnp.int32),
        "mask_bn": jnp.asarray([[1]], dtype=jnp.uint8),
        "ct_bn": jnp.asarray([[3.0]], dtype=jnp.float32),
    }


def _single_operands():
    return {
        "weights": jnp.asarray([2.0], dtype=jnp.float32),
        "local_targets": jnp.asarray([0], dtype=jnp.uint16),
        "indptr": jnp.asarray([0, 1], dtype=jnp.int32),
        "tile_offsets": jnp.asarray([[0, 1]], dtype=jnp.int32),
        "event": jnp.asarray([True]),
        "ct": jnp.asarray([3.0], dtype=jnp.float32),
    }


def test_explicit_bpsa_package_exports_standalone_operations():
    """Expose BPSA only through its explicit experimental package."""
    from brainevent._tcsr import bpsa

    assert callable(bpsa.pack_event_mask_bn)
    assert callable(bpsa.csr_bpsa_dinput_masked)
    assert callable(bpsa.csr_bpsa_dinput_single)

    packed = bpsa.pack_event_mask_bn(
        jnp.asarray([[True, False, True, False, False, False, False, True, True]])
    )
    np.testing.assert_array_equal(packed, np.asarray([[0x85, 0x01]], np.uint8))

    float_packed = bpsa.pack_event_mask_bn(
        jnp.asarray([[1.0, 0.0, -1.0, 2.0]], dtype=jnp.float32)
    )
    np.testing.assert_array_equal(float_packed, np.asarray([[0x09]], np.uint8))


@pytest.mark.parametrize(
    ("events", "error", "message"),
    [
        (jnp.ones((2,), dtype=jnp.float32), ValueError, "rank 2"),
        (jnp.empty((0, 2), dtype=jnp.float32), ValueError, "positive"),
        (jnp.ones((1, 2), dtype=jnp.int32), TypeError, "bool or float32"),
    ],
)
def test_pack_event_mask_rejects_invalid_inputs(events, error, message):
    """Reject inputs that cannot represent a non-empty BN event mask."""
    from brainevent._tcsr import bpsa

    with pytest.raises(error, match=message):
        bpsa.pack_event_mask_bn(events)


def test_masked_bpsa_loads_packaged_mm_cuda_source(monkeypatch):
    """Load the matrix BPSA kernel from its packaged source path."""
    from brainevent._tcsr.bpsa import bpsa as implementation

    loaded_paths = []

    def fake_load_cuda_file(path, *, name):
        loaded_paths.append((Path(path), name))
        return object()

    def fake_ffi_call(*args, **kwargs):
        del args, kwargs
        return lambda *operands: jnp.zeros((1, 1), dtype=jnp.float32)

    monkeypatch.setattr(implementation, "_BPSA_CUDA_MODULE", None)
    monkeypatch.setattr(implementation, "load_cuda_file", fake_load_cuda_file)
    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi_call)

    operands = _masked_operands()
    result = implementation.csr_bpsa_dinput_masked(**operands)
    cached_result = implementation.csr_bpsa_dinput_masked(**operands)

    assert loaded_paths == [
        (Path(implementation.__file__).with_name("bpsa_mm.cu"), "tcsr_bpsa")
    ]
    assert result.shape == (1, 1)
    assert cached_result.shape == (1, 1)


@pytest.mark.parametrize(
    ("name", "value", "error", "message"),
    [
        ("weights", jnp.ones((1, 1), jnp.float32), ValueError, "rank 1"),
        ("weights", jnp.ones((1,), jnp.float16), TypeError, "float32"),
        ("indptr", jnp.asarray([0, 1], jnp.uint32), TypeError, "int32 or int64"),
        ("local_targets", jnp.asarray([0], jnp.int32), TypeError, "uint16"),
        ("tile_offsets", jnp.asarray([[0, 1]], jnp.uint16), TypeError, "int32"),
        ("mask_bn", jnp.asarray([[1]], jnp.int8), TypeError, "uint8"),
        ("ct_bn", jnp.ones((1, 1), jnp.float16), TypeError, "float32"),
        ("local_targets", jnp.asarray([], jnp.uint16), ValueError, "sizes"),
        ("indptr", jnp.asarray([0], jnp.int32), ValueError, "at least one"),
        ("mask_bn", jnp.ones((2, 1), jnp.uint8), ValueError, "batch"),
        ("mask_bn", jnp.ones((1, 2), jnp.uint8), ValueError, "mask_bn shape"),
        ("ct_bn", jnp.empty((1, 0), jnp.float32), ValueError, "positive"),
        (
            "tile_offsets",
            jnp.asarray([[0, 1, 1]], jnp.int32),
            ValueError,
            "tile_offsets shape",
        ),
    ],
)
def test_masked_bpsa_rejects_invalid_operands_before_cuda(
    monkeypatch, name, value, error, message
):
    """Validate every packed-mask ABI invariant before loading CUDA."""
    from brainevent._tcsr.bpsa import bpsa as implementation

    def reject_cuda_load(*args, **kwargs):
        raise AssertionError("invalid operands must not load CUDA")

    monkeypatch.setattr(implementation, "load_cuda_file", reject_cuda_load)
    operands = _masked_operands()
    operands[name] = value

    with pytest.raises(error, match=message):
        implementation.csr_bpsa_dinput_masked(**operands)


def test_single_bpsa_loads_packaged_mv_cuda_source(monkeypatch):
    """Load the vector BPSA kernel from its packaged source path."""
    from brainevent._tcsr.bpsa import bpsa as implementation

    loaded_paths = []

    def fake_load_cuda_file(path, *, name):
        loaded_paths.append((Path(path), name))
        return object()

    def fake_ffi_call(*args, **kwargs):
        del args, kwargs
        return lambda *operands: jnp.zeros((1,), dtype=jnp.float32)

    monkeypatch.setattr(implementation, "_BPSA_SINGLE_CUDA_MODULE", None)
    monkeypatch.setattr(implementation, "load_cuda_file", fake_load_cuda_file)
    monkeypatch.setattr(jax.ffi, "ffi_call", fake_ffi_call)

    operands = _single_operands()
    result = implementation.csr_bpsa_dinput_single(**operands)
    operands["event"] = jnp.asarray([1.0], dtype=jnp.float32)
    float_result = implementation.csr_bpsa_dinput_single(**operands)

    assert loaded_paths == [
        (
            Path(implementation.__file__).with_name("bpsa_mv.cu"),
            "tcsr_bpsa_single",
        )
    ]
    assert result.shape == (1,)
    assert float_result.shape == (1,)


@pytest.mark.parametrize(
    ("name", "value", "error", "message"),
    [
        ("weights", jnp.ones((1, 1), jnp.float32), ValueError, "rank 1"),
        ("weights", jnp.ones((1,), jnp.float16), TypeError, "float32"),
        ("local_targets", jnp.asarray([0], jnp.int32), TypeError, "uint16"),
        ("indptr", jnp.asarray([0, 1], jnp.uint32), TypeError, "int32 or int64"),
        ("tile_offsets", jnp.asarray([[0, 1]], jnp.uint16), TypeError, "int32"),
        ("event", jnp.asarray([1], jnp.int32), TypeError, "bool or float32"),
        ("ct", jnp.ones((1,), jnp.float16), TypeError, "float32"),
        ("local_targets", jnp.asarray([], jnp.uint16), ValueError, "slot sizes"),
        ("indptr", jnp.asarray([0], jnp.int32), ValueError, "at least one"),
        ("event", jnp.asarray([True, False]), ValueError, "row count"),
        ("ct", jnp.asarray([], jnp.float32), ValueError, "positive"),
        (
            "tile_offsets",
            jnp.asarray([[0, 1, 1]], jnp.int32),
            ValueError,
            "tile_offsets shape",
        ),
    ],
)
def test_single_bpsa_rejects_invalid_operands_before_cuda(
    monkeypatch, name, value, error, message
):
    """Validate every vector ABI invariant before loading CUDA."""
    from brainevent._tcsr.bpsa import bpsa as implementation

    def reject_cuda_load(*args, **kwargs):
        raise AssertionError("invalid operands must not load CUDA")

    monkeypatch.setattr(implementation, "load_cuda_file", reject_cuda_load)
    operands = _single_operands()
    operands[name] = value

    with pytest.raises(error, match=message):
        implementation.csr_bpsa_dinput_single(**operands)
