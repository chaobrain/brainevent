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
_SDDMM_BINARY_CUDA_MODULE = None
_SDDMM_FLOAT_CUDA_MODULE = None
_SDDMV_CUDA_MODULE = None


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

    Treat floating event values as active exactly when they are positive. The
    operation consumes CSR structure directly and supports only the direct
    transposed binary CSRMM orientation.

    Parameters
    ----------
    events : jax.Array
        Boolean or float32 event matrix with shape ``(batch, rows)``.
    ct : jax.Array
        Float32 cotangent matrix with shape ``(batch, cols)``.
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
        Float32 gradient values with shape ``(nnz,)`` in CSR-slot order.

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
    if event_dtype not in (jnp.dtype(jnp.bool_), jnp.dtype(jnp.float32)):
        raise TypeError("events dtype must be bool or float32")
    if jnp.dtype(ct.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("ct dtype must be float32")
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

    event_suffix = "bool" if event_dtype == jnp.dtype(jnp.bool_) else "float"
    global _SDDMM_BINARY_CUDA_MODULE
    if _SDDMM_BINARY_CUDA_MODULE is None:
        _SDDMM_BINARY_CUDA_MODULE = load_cuda_file(
            Path(__file__).with_name("sddmm_binary.cu"),
            name="tcsr_sddmm_binary",
        )
    target = (
        "tcsr_sddmm_binary."
        f"tcsr_sddmm_dweight_binary_f32_{event_suffix}_t"
    )
    phases = (batch + 127) // 128
    output_info = (
        jax.ShapeDtypeStruct((indices.size,), jnp.float32),
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
        Float32 eligibility values with shape ``(batch, rows)``.
    ct : jax.Array
        Float32 cotangents with shape ``(batch, cols)``.
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
        Float32 weight gradients with shape ``(nnz,)``.

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
    each complete signed float32 B value in the sampled product.

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

    if jnp.dtype(B.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("B dtype must be float32")
    if jnp.dtype(ct.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("ct dtype must be float32")
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
        jax.ShapeDtypeStruct(indices.shape, jnp.float32),
        jax.ShapeDtypeStruct((phases, rows, 4), jnp.uint32),
        jax.ShapeDtypeStruct((rows,), jnp.uint8),
        jax.ShapeDtypeStruct((phases, row_chunks), jnp.uint16),
        jax.ShapeDtypeStruct((phases, row_chunks, 128), jnp.uint8),
    )
    dweight, _, _, _, _ = jax.ffi.ffi_call(
        "tcsr_sddmm_float.tcsr_sddmm_dweight_float_f32_t",
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
        Boolean or float32 source events with shape ``(rows,)``.
    ct : jax.Array
        Float32 cotangent values with shape ``(cols,)``.
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
        Float32 gradients with shape ``(nnz,)`` in TileCSR slot order.

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
    if event_dtype not in (jnp.dtype(jnp.bool_), jnp.dtype(jnp.float32)):
        raise TypeError("event dtype must be bool or float32")
    if jnp.dtype(ct.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("ct dtype must be float32")
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

    global _SDDMV_CUDA_MODULE
    if _SDDMV_CUDA_MODULE is None:
        _SDDMV_CUDA_MODULE = load_cuda_file(
            Path(__file__).with_name("sddmv.cu"),
            name="tcsr_sddmv",
        )
    suffix = "bool" if event_dtype == jnp.dtype(jnp.bool_) else "float"
    output_info = (
        jax.ShapeDtypeStruct(local_targets.shape, jnp.float32),
        jax.ShapeDtypeStruct(event.shape, jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    dweight, _, _ = jax.ffi.ffi_call(
        f"tcsr_sddmv.tcsr_sddmv_dweight_binary_f32_{suffix}_t",
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
        Float32 source eligibility values with shape ``(rows,)``.
    ct : jax.Array
        Float32 cotangent values with shape ``(cols,)``.
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
        Float32 values ``event[row] * ct[target]`` in TileCSR slot order.

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

    if jnp.dtype(event.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("event dtype must be float32")
    if jnp.dtype(ct.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("ct dtype must be float32")
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

    global _SDDMV_CUDA_MODULE
    if _SDDMV_CUDA_MODULE is None:
        _SDDMV_CUDA_MODULE = load_cuda_file(
            Path(__file__).with_name("sddmv.cu"),
            name="tcsr_sddmv",
        )
    output_info = (
        jax.ShapeDtypeStruct(local_targets.shape, jnp.float32),
        jax.ShapeDtypeStruct(event.shape, jnp.int32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    dweight, _, _ = jax.ffi.ffi_call(
        "tcsr_sddmv.tcsr_sddmv_dweight_float_f32_t",
        output_info,
        input_layouts=[(0,), (0,), (0,), (0,), (0, 1)],
        output_layouts=[(0,), (0,), (0,)],
    )(event, ct, local_targets, indptr, tile_offsets)
    return dweight
