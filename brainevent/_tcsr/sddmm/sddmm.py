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

"""Provide TCSR sampled weight-gradient CUDA operators."""

from pathlib import Path

import jax
import jax.numpy as jnp

from brainevent._op import load_cuda_file


__all__ = [
    "tcsr_sddmm_dweight_binary",
    "tcsr_sddmm_dweight_float",
    "tcsr_sddmv_dweight_binary",
    "tcsr_sddmv_dweight_float",
]

_SUPPORTED_BATCHES = frozenset((4, 8, 16, 32, 64, 128, 256, 512))
_TILE_SIZE = 8192
_F32 = jnp.dtype(jnp.float32)
_F64 = jnp.dtype(jnp.float64)
_BINARY_TARGET_SUFFIXES = {
    (_F32, jnp.dtype(jnp.bool_)): ("f32", "bool"),
    (_F32, _F32): ("f32", "float"),
    (_F64, jnp.dtype(jnp.bool_)): ("f64", "bool"),
    (_F64, jnp.dtype(jnp.int8)): ("f64", "int8"),
    (_F64, _F32): ("f64", "float"),
    (_F64, _F64): ("f64", "double"),
}
_SDDMM_BINARY_CUDA_MODULE = None
_SDDMM_FLOAT_CUDA_MODULE = None
_SDDMV_BINARY_CUDA_MODULE = None
_SDDMV_FLOAT_CUDA_MODULE = None


def tcsr_sddmm_dweight_binary(
    events: jax.Array,
    ct: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    local_targets: jax.Array,
    tile_offsets: jax.Array,
    *,
    transpose: bool,
) -> jax.Array:
    """Compute heterogeneous CSR-slot gradients from binary events.

    Treat numeric event values as active exactly when they are positive. The
    operation consumes CSR structure directly and supports only the direct
    transposed binary CSRMM orientation.

    Parameters
    ----------
    events : jax.Array
        Event matrix with shape ``(batch, rows)``. Float32 cotangents accept
        bool or float32 events. Float64 cotangents accept bool, int8, float32,
        or float64 events.
    ct : jax.Array
        Float32 or float64 cotangent matrix with shape ``(batch, cols)``.
    indices : jax.Array
        Int32 CSR column indices with shape ``(nnz,)``.
    indptr : jax.Array
        Int32 or int64 CSR row offsets with shape ``(rows + 1,)``.
    local_targets : jax.Array
        Uint16 target offsets inside 8192-column base tiles.
    tile_offsets : jax.Array
        Int32 row-local tile boundaries with shape
        ``(rows, ceil(cols / 8192) + 1)``.
    transpose : bool
        Require the direct transposed binary CSRMM orientation.

    Returns
    -------
    jax.Array
        Gradient values with shape ``(nnz,)`` and the same dtype as ``ct``.

    Raises
    ------
    TypeError
        If an input dtype or the orientation type is unsupported.
    ValueError
        If an input rank, shape, batch size, or orientation is unsupported.

    Notes
    -----
    Supported batch sizes are 4, 8, 16, 32, 64, 128, 256, and 512. This
    function is GPU-only and intentionally has no generic SDDMM fallback.
    """
    if not isinstance(transpose, bool):
        raise TypeError("transpose must be a bool")
    if not transpose:
        raise ValueError("masked CSR SDDMM supports only transpose=True")

    events = jnp.asarray(events)
    ct = jnp.asarray(ct)
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    local_targets = jnp.asarray(local_targets)
    tile_offsets = jnp.asarray(tile_offsets)

    for name, value, rank in (
        ("events", events, 2),
        ("ct", ct, 2),
        ("indices", indices, 1),
        ("indptr", indptr, 1),
        ("local_targets", local_targets, 1),
        ("tile_offsets", tile_offsets, 2),
    ):
        if value.ndim != rank:
            raise ValueError(f"{name} must be rank {rank}")

    event_dtype = jnp.dtype(events.dtype)
    ct_dtype = jnp.dtype(ct.dtype)
    target_suffixes = _BINARY_TARGET_SUFFIXES.get((ct_dtype, event_dtype))
    if target_suffixes is None:
        raise TypeError(
            "unsupported binary (ct, events) dtype pair; expected "
            "(float32, bool|float32) or "
            "(float64, bool|int8|float32|float64)"
        )
    if jnp.dtype(indices.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("indices dtype must be int32")
    if jnp.dtype(indptr.dtype) not in (
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.int64),
    ):
        raise TypeError("indptr dtype must be int32 or int64")
    if jnp.dtype(local_targets.dtype) != jnp.dtype(jnp.uint16):
        raise TypeError("local_targets dtype must be uint16")
    if jnp.dtype(tile_offsets.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("tile_offsets dtype must be int32")

    if events.shape[0] != ct.shape[0]:
        raise ValueError("events and ct batch dimensions must match")
    batch = events.shape[0]
    if batch not in _SUPPORTED_BATCHES:
        supported = ", ".join(
            str(value) for value in sorted(_SUPPORTED_BATCHES)
        )
        raise ValueError(f"supported batch sizes are {supported}; got {batch}")
    if indptr.size < 2:
        raise ValueError("indptr must describe at least one CSR row")

    rows = indptr.size - 1
    if events.shape[1] != rows:
        raise ValueError("events neuron dimension must equal the CSR row count")
    cols = ct.shape[1]
    if cols <= 0:
        raise ValueError("dense neuron dimensions must be positive")
    if local_targets.shape != indices.shape:
        raise ValueError("local_targets shape must equal indices shape")
    tile_count = (cols + _TILE_SIZE - 1) // _TILE_SIZE
    expected_offsets_shape = (rows, tile_count + 1)
    if tile_offsets.shape != expected_offsets_shape:
        raise ValueError(
            "tile_offsets shape must be "
            f"{expected_offsets_shape}; got {tile_offsets.shape}"
        )

    value_suffix, event_suffix = target_suffixes
    global _SDDMM_BINARY_CUDA_MODULE
    if _SDDMM_BINARY_CUDA_MODULE is None:
        _SDDMM_BINARY_CUDA_MODULE = load_cuda_file(
            Path(__file__).with_name("sddmm_binary.cu"),
            name="tcsr_sddmm_binary",
        )
    target = (
        "tcsr_sddmm_binary."
        f"tcsr_sddmm_dweight_binary_{value_suffix}_{event_suffix}_t"
    )
    phases = (batch + 127) // 128
    output_info = (
        jax.ShapeDtypeStruct((indices.size,), ct_dtype),
        jax.ShapeDtypeStruct((phases, rows, 4), jnp.uint32),
    )
    dweight, _ = jax.ffi.ffi_call(
        target,
        output_info,
        input_layouts=[(0, 1), (0, 1), (0,), (0,), (0, 1)],
        output_layouts=[(0,), (0, 1, 2)],
    )(events.T, ct, local_targets, indptr, tile_offsets)
    return dweight


def tcsr_sddmm_dweight_float(
    B: jax.Array,
    ct: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    local_targets: jax.Array,
    tile_offsets: jax.Array,
    *,
    transpose: bool,
) -> jax.Array:
    """Compute value-preserving batched TCSR-slot gradients.

    Parameters
    ----------
    B : jax.Array
        Float32 or float64 eligibility values with shape ``(batch, rows)``.
    ct : jax.Array
        Cotangents with shape ``(batch, cols)`` and the same dtype as ``B``.
    indices : jax.Array
        Int32 global targets in TileCSR slot order. CUDA does not consume this
        operand; it validates slot alignment at the Python boundary.
    indptr : jax.Array
        Int32 or int64 CSR row offsets with shape ``(rows + 1,)``.
    local_targets : jax.Array
        Uint16 target offsets within 8192-column base tiles.
    tile_offsets : jax.Array
        Int32 row-local base-tile slot boundaries.
    transpose : bool
        Require the direct transposed CSRMM orientation.

    Returns
    -------
    jax.Array
        Weight gradients with shape ``(nnz,)`` and the same dtype as ``B``.

    Raises
    ------
    TypeError
        If the orientation selector or an operand dtype is unsupported.
    ValueError
        If an operand rank, shape, batch size, or orientation is unsupported.

    Notes
    -----
    The standalone CUDA kernel supports fixed batches 4 through 512. It uses
    global ``B != 0`` masks to enumerate active positions while preserving
    each complete signed B value in the sampled product. Mixed float32 and
    float64 inputs are not supported.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.sddmm import tcsr_sddmm_dweight_float
        >>> B = jnp.ones((4, 1), dtype=jnp.float32)
        >>> ct = jnp.ones((4, 1), dtype=jnp.float32)
        >>> indices = jnp.array([0], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 1], dtype=jnp.int32)
        >>> local = jnp.array([0], dtype=jnp.uint16)
        >>> offsets = jnp.array([[0, 1]], dtype=jnp.int32)
        >>> result = tcsr_sddmm_dweight_float(  # doctest: +SKIP
        ...     B, ct, indices, indptr, local, offsets, transpose=True
        ... )
        >>> result.shape  # doctest: +SKIP
        (1,)
    """
    if not isinstance(transpose, bool):
        raise TypeError("transpose must be a bool")
    if not transpose:
        raise ValueError(
            "batched float TCSR SDDMM supports only transpose=True"
        )

    B = jnp.asarray(B)
    ct = jnp.asarray(ct)
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    local_targets = jnp.asarray(local_targets)
    tile_offsets = jnp.asarray(tile_offsets)

    for name, value, rank in (
        ("B", B, 2),
        ("ct", ct, 2),
        ("indices", indices, 1),
        ("indptr", indptr, 1),
        ("local_targets", local_targets, 1),
        ("tile_offsets", tile_offsets, 2),
    ):
        if value.ndim != rank:
            raise ValueError(f"{name} must be rank {rank}")

    value_dtype = jnp.dtype(B.dtype)
    if value_dtype not in (_F32, _F64):
        raise TypeError("B dtype must be float32 or float64")
    if jnp.dtype(ct.dtype) != value_dtype:
        raise TypeError("B and ct dtypes must match float32 or float64")
    if jnp.dtype(indices.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("indices dtype must be int32")
    if jnp.dtype(indptr.dtype) not in (
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.int64),
    ):
        raise TypeError("indptr dtype must be int32 or int64")
    if jnp.dtype(local_targets.dtype) != jnp.dtype(jnp.uint16):
        raise TypeError("local_targets dtype must be uint16")
    if jnp.dtype(tile_offsets.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("tile_offsets dtype must be int32")

    if B.shape[0] != ct.shape[0]:
        raise ValueError("B and ct batch dimensions must match")
    batch = B.shape[0]
    if batch not in _SUPPORTED_BATCHES:
        supported = ", ".join(
            str(value) for value in sorted(_SUPPORTED_BATCHES)
        )
        raise ValueError(f"supported batch sizes are {supported}; got {batch}")
    if indptr.size < 2:
        raise ValueError("indptr must describe at least one CSR row")
    rows = indptr.size - 1
    if B.shape[1] != rows:
        raise ValueError("B neuron dimension must equal the CSR row count")
    cols = ct.shape[1]
    if cols <= 0:
        raise ValueError("dense neuron dimensions must be positive")
    if local_targets.shape != indices.shape:
        raise ValueError("local_targets shape must equal indices shape")
    tile_count = (cols + _TILE_SIZE - 1) // _TILE_SIZE
    expected_offsets_shape = (rows, tile_count + 1)
    if tile_offsets.shape != expected_offsets_shape:
        raise ValueError(
            "tile_offsets shape must be "
            f"{expected_offsets_shape}; got {tile_offsets.shape}"
        )

    global _SDDMM_FLOAT_CUDA_MODULE
    if _SDDMM_FLOAT_CUDA_MODULE is None:
        _SDDMM_FLOAT_CUDA_MODULE = load_cuda_file(
            Path(__file__).with_name("sddmm_float.cu"),
            name="tcsr_sddmm_float",
        )
    phases = (batch + 127) // 128
    row_chunks = (rows + 127) // 128
    output_info = (
        jax.ShapeDtypeStruct(indices.shape, value_dtype),
        jax.ShapeDtypeStruct((phases, rows, 4), jnp.uint32),
        jax.ShapeDtypeStruct((rows,), jnp.uint8),
        jax.ShapeDtypeStruct((phases, row_chunks), jnp.uint16),
        jax.ShapeDtypeStruct((phases, row_chunks, 128), jnp.uint8),
    )
    dweight, _, _, _, _ = jax.ffi.ffi_call(
        "tcsr_sddmm_float.tcsr_sddmm_dweight_float_"
        f"{'f32' if value_dtype == _F32 else 'f64'}_t",
        output_info,
        input_layouts=[(0, 1), (0, 1), (0,), (0,), (0, 1)],
        output_layouts=[(0,), (0, 1, 2), (0,), (0, 1), (0, 1, 2)],
    )(B.T, ct, local_targets, indptr, tile_offsets)
    return dweight


def tcsr_sddmv_dweight_binary(
    event: jax.Array,
    ct: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    local_targets: jax.Array,
    tile_offsets: jax.Array,
    *,
    transpose: bool,
) -> jax.Array:
    """Compute heterogeneous CSR-slot gradients for one event vector.

    Parameters
    ----------
    event : jax.Array
        Source events with shape ``(rows,)``. Float32 cotangents accept bool or
        float32 events. Float64 cotangents accept bool, int8, float32, or
        float64 events.
    ct : jax.Array
        Float32 or float64 cotangent values with shape ``(cols,)``.
    indices : jax.Array
        Int32 CSR column indices with shape ``(nnz,)``. The CUDA ABI does not
        consume this operand; it validates slot alignment at the Python edge.
    indptr : jax.Array
        Int32 or int64 CSR row offsets with shape ``(rows + 1,)``.
    local_targets : jax.Array
        Uint16 target offsets inside 8192-column base tiles.
    tile_offsets : jax.Array
        Int32 row-local tile boundaries.
    transpose : bool
        Require the direct transposed MV orientation.

    Returns
    -------
    jax.Array
        Gradients with shape ``(nnz,)`` and the same dtype as ``ct``.

    Raises
    ------
    TypeError
        If an orientation selector or operand dtype is unsupported.
    ValueError
        If an operand rank, dimension, or orientation is invalid.

    Notes
    -----
    This is a GPU-only single-batch kernel. Active rows are compacted before
    their TileCSR slots are distributed across persistent CUDA blocks.
    """
    if not isinstance(transpose, bool):
        raise TypeError("transpose must be a bool")
    if not transpose:
        raise ValueError(
            "single-batch TileCSR SDDMM supports only transpose=True"
        )

    event = jnp.asarray(event)
    ct = jnp.asarray(ct)
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    local_targets = jnp.asarray(local_targets)
    tile_offsets = jnp.asarray(tile_offsets)

    for name, value, rank in (
        ("event", event, 1),
        ("ct", ct, 1),
        ("indices", indices, 1),
        ("indptr", indptr, 1),
        ("local_targets", local_targets, 1),
        ("tile_offsets", tile_offsets, 2),
    ):
        if value.ndim != rank:
            raise ValueError(f"{name} must be rank {rank}")

    event_dtype = jnp.dtype(event.dtype)
    ct_dtype = jnp.dtype(ct.dtype)
    target_suffixes = _BINARY_TARGET_SUFFIXES.get((ct_dtype, event_dtype))
    if target_suffixes is None:
        raise TypeError(
            "unsupported binary (ct, event) dtype pair; expected "
            "(float32, bool|float32) or "
            "(float64, bool|int8|float32|float64)"
        )
    if jnp.dtype(indices.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("indices dtype must be int32")
    if jnp.dtype(indptr.dtype) not in (
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.int64),
    ):
        raise TypeError("indptr dtype must be int32 or int64")
    if jnp.dtype(local_targets.dtype) != jnp.dtype(jnp.uint16):
        raise TypeError("local_targets dtype must be uint16")
    if jnp.dtype(tile_offsets.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("tile_offsets dtype must be int32")

    if indptr.size < 2:
        raise ValueError("indptr must describe at least one CSR row")
    rows = indptr.size - 1
    if event.size != rows:
        raise ValueError("event size must equal the CSR row count")
    if ct.size <= 0:
        raise ValueError("ct dimension must be positive")
    if local_targets.shape != indices.shape:
        raise ValueError("local_targets shape must equal indices shape")
    tile_count = (ct.size + _TILE_SIZE - 1) // _TILE_SIZE
    expected_offsets_shape = (rows, tile_count + 1)
    if tile_offsets.shape != expected_offsets_shape:
        raise ValueError(
            "tile_offsets shape must be "
            f"{expected_offsets_shape}; got {tile_offsets.shape}"
        )

    global _SDDMV_BINARY_CUDA_MODULE
    if _SDDMV_BINARY_CUDA_MODULE is None:
        _SDDMV_BINARY_CUDA_MODULE = load_cuda_file(
            Path(__file__).with_name("sddmv_binary.cu"),
            name="tcsr_sddmv_binary",
        )
    value_suffix, event_suffix = target_suffixes
    output_info = (
        jax.ShapeDtypeStruct(local_targets.shape, ct_dtype),
        jax.ShapeDtypeStruct(event.shape, jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    dweight, _, _ = jax.ffi.ffi_call(
        "tcsr_sddmv_binary."
        f"tcsr_sddmv_dweight_binary_{value_suffix}_{event_suffix}_t",
        output_info,
        input_layouts=[(0,), (0,), (0,), (0,), (0, 1)],
        output_layouts=[(0,), (0,), (0,)],
    )(event, ct, local_targets, indptr, tile_offsets)
    return dweight


def tcsr_sddmv_dweight_float(
    event: jax.Array,
    ct: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    local_targets: jax.Array,
    tile_offsets: jax.Array,
    *,
    transpose: bool,
) -> jax.Array:
    """Compute value-preserving Batch1 CSR-slot gradients.

    Parameters
    ----------
    event : jax.Array
        Float32 or float64 source eligibility values with shape ``(rows,)``.
    ct : jax.Array
        Cotangent values with shape ``(cols,)`` and the same dtype as
        ``event``.
    indices : jax.Array
        Int32 CSR targets. The CUDA ABI uses this operand only for Python-side
        slot-alignment validation.
    indptr : jax.Array
        Int32 or int64 CSR row offsets with shape ``(rows + 1,)``.
    local_targets : jax.Array
        Uint16 targets local to each 8192-column tile.
    tile_offsets : jax.Array
        Int32 row-local tile boundaries.
    transpose : bool
        Require the direct transposed MV orientation.

    Returns
    -------
    jax.Array
        Values ``event[row] * ct[target]`` in TileCSR slot order, with the same
        dtype as ``event``.

    Raises
    ------
    TypeError
        If the orientation or an operand dtype is unsupported.
    ValueError
        If an operand rank, shape, or orientation is invalid.

    Notes
    -----
    This GPU-only path preserves nonzero positive and negative eligibility
    values. It retains the production Batch1 tile-panel schedule.
    """
    if not isinstance(transpose, bool):
        raise TypeError("transpose must be a bool")
    if not transpose:
        raise ValueError(
            "single-batch float TileCSR SDDMM supports only transpose=True"
        )

    event = jnp.asarray(event)
    ct = jnp.asarray(ct)
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    local_targets = jnp.asarray(local_targets)
    tile_offsets = jnp.asarray(tile_offsets)

    for name, value, rank in (
        ("event", event, 1),
        ("ct", ct, 1),
        ("indices", indices, 1),
        ("indptr", indptr, 1),
        ("local_targets", local_targets, 1),
        ("tile_offsets", tile_offsets, 2),
    ):
        if value.ndim != rank:
            raise ValueError(f"{name} must be rank {rank}")

    value_dtype = jnp.dtype(event.dtype)
    if value_dtype not in (_F32, _F64):
        raise TypeError("event dtype must be float32 or float64")
    if jnp.dtype(ct.dtype) != value_dtype:
        raise TypeError("event and ct dtypes must match float32 or float64")
    if jnp.dtype(indices.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("indices dtype must be int32")
    if jnp.dtype(indptr.dtype) not in (
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.int64),
    ):
        raise TypeError("indptr dtype must be int32 or int64")
    if jnp.dtype(local_targets.dtype) != jnp.dtype(jnp.uint16):
        raise TypeError("local_targets dtype must be uint16")
    if jnp.dtype(tile_offsets.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("tile_offsets dtype must be int32")

    if indptr.size < 2:
        raise ValueError("indptr must describe at least one CSR row")
    rows = indptr.size - 1
    if event.size != rows:
        raise ValueError("event size must equal the CSR row count")
    if ct.size <= 0:
        raise ValueError("ct dimension must be positive")
    if local_targets.shape != indices.shape:
        raise ValueError("local_targets shape must equal indices shape")
    tile_count = (ct.size + _TILE_SIZE - 1) // _TILE_SIZE
    expected_offsets_shape = (rows, tile_count + 1)
    if tile_offsets.shape != expected_offsets_shape:
        raise ValueError(
            "tile_offsets shape must be "
            f"{expected_offsets_shape}; got {tile_offsets.shape}"
        )

    global _SDDMV_FLOAT_CUDA_MODULE
    if _SDDMV_FLOAT_CUDA_MODULE is None:
        _SDDMV_FLOAT_CUDA_MODULE = load_cuda_file(
            Path(__file__).with_name("sddmv_float.cu"),
            name="tcsr_sddmv_float",
        )
    output_info = (
        jax.ShapeDtypeStruct(local_targets.shape, value_dtype),
        jax.ShapeDtypeStruct(event.shape, jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    dweight, _, _ = jax.ffi.ffi_call(
        "tcsr_sddmv_float.tcsr_sddmv_dweight_float_"
        f"{'f32' if value_dtype == _F32 else 'f64'}_t",
        output_info,
        input_layouts=[(0,), (0,), (0,), (0,), (0, 1)],
        output_layouts=[(0,), (0,), (0,)],
    )(event, ct, local_targets, indptr, tile_offsets)
    return dweight
