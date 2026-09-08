"""Provide the packed-mask BPSA input-gradient operator."""

from pathlib import Path
import jax
import jax.numpy as jnp

from brainevent._op import load_cuda_file

__all__ = [
    "csr_bpsa_dinput_masked",
    "csr_bpsa_dinput_single",
    "pack_event_mask_bn",
]

_BASE_TILE_SIZE = 8192
_BPSA_CUDA_MODULE = None
_BPSA_SINGLE_CUDA_MODULE = None


def _load_bpsa_cuda_module():
    global _BPSA_CUDA_MODULE
    if _BPSA_CUDA_MODULE is None:
        _BPSA_CUDA_MODULE = load_cuda_file(
            Path(__file__).with_name("bpsa_mm.cu"),
            name="tcsr_bpsa",
        )
    return _BPSA_CUDA_MODULE


def pack_event_mask_bn(event_bn: jax.Array) -> jax.Array:
    """Pack positive BN events along the neuron axis.

    Parameters
    ----------
    event_bn : jax.Array
        Boolean or float32 event matrix in BN layout.

    Returns
    -------
    jax.Array
        Little-endian uint8 mask with shape
        ``(batch, ceil(neurons / 8))``.

    Raises
    ------
    ValueError
        If the event matrix is not rank two or has an empty dimension.
    TypeError
        If the event matrix is neither boolean nor float32.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.bpsa import pack_event_mask_bn
        >>> events = jnp.array([[True, False, True]])
        >>> pack_event_mask_bn(events).tolist()
        [[5]]
    """
    event_bn = jnp.asarray(event_bn)
    if event_bn.ndim != 2:
        raise ValueError("event_bn must be rank 2")
    if event_bn.shape[0] <= 0 or event_bn.shape[1] <= 0:
        raise ValueError("event_bn dimensions must be positive")

    event_dtype = jnp.dtype(event_bn.dtype)
    if event_dtype == jnp.dtype(jnp.bool_):
        active_bn = event_bn
    elif event_dtype == jnp.dtype(jnp.float32):
        active_bn = event_bn > 0
    else:
        raise TypeError("event_bn dtype must be bool or float32")
    return jnp.packbits(active_bn, axis=1, bitorder="little")


def csr_bpsa_dinput_masked(
    weights: jax.Array,
    indptr: jax.Array,
    local_targets: jax.Array,
    tile_offsets: jax.Array,
    mask_bn: jax.Array,
    ct_bn: jax.Array,
) -> jax.Array:
    """Compute a packed-mask BPSA input gradient in BN layout.

    Parameters
    ----------
    weights : jax.Array
        Float32 CSR values in row-sorted TileCSR slot order.
    indptr : jax.Array
        Int32 or int64 CSR row offsets.
    local_targets : jax.Array
        Uint16 target offsets inside 8192-column tiles, in CSR-slot order.
    tile_offsets : jax.Array
        Int32 row-local slot boundaries with shape
        ``(rows, ceil(cols / 8192) + 1)``.
    mask_bn : jax.Array
        Little-endian uint8 input-event mask in BN layout.
    ct_bn : jax.Array
        Float32 output cotangent in BN layout.

    Returns
    -------
    jax.Array
        Masked input gradient ``(ct_bn @ W.T) * mask_bn``.

    Raises
    ------
    ValueError
        If operand ranks or logical dimensions are inconsistent.
    TypeError
        If an operand dtype is unsupported.

    Notes
    -----
    This operator is GPU-only and defines no standalone autodiff rule. Full
    CSR column indices are not part of its FFI contract.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.bpsa import csr_bpsa_dinput_masked
        >>> weights = jnp.array([2.0], dtype=jnp.float32)
        >>> indptr = jnp.array([0, 1], dtype=jnp.int32)
        >>> targets = jnp.array([0], dtype=jnp.uint16)
        >>> offsets = jnp.array([[0, 1]], dtype=jnp.int32)
        >>> mask = jnp.array([[1]], dtype=jnp.uint8)
        >>> ct = jnp.array([[3.0]], dtype=jnp.float32)
        >>> db = csr_bpsa_dinput_masked(  # doctest: +SKIP
        ...     weights, indptr, targets, offsets, mask, ct
        ... )
        >>> db.shape  # doctest: +SKIP
        (1, 1)
    """
    weights = jnp.asarray(weights)
    indptr = jnp.asarray(indptr)
    local_targets = jnp.asarray(local_targets)
    tile_offsets = jnp.asarray(tile_offsets)
    mask_bn = jnp.asarray(mask_bn)
    ct_bn = jnp.asarray(ct_bn)

    for name, value, rank in (
        ("weights", weights, 1),
        ("indptr", indptr, 1),
        ("local_targets", local_targets, 1),
        ("tile_offsets", tile_offsets, 2),
        ("mask_bn", mask_bn, 2),
        ("ct_bn", ct_bn, 2),
    ):
        if value.ndim != rank:
            raise ValueError(f"{name} must be rank {rank}")

    if jnp.dtype(weights.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("weights dtype must be float32")
    if jnp.dtype(indptr.dtype) not in (
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.int64),
    ):
        raise TypeError("indptr dtype must be int32 or int64")
    if jnp.dtype(local_targets.dtype) != jnp.dtype(jnp.uint16):
        raise TypeError("local_targets dtype must be uint16")
    if jnp.dtype(tile_offsets.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("tile_offsets dtype must be int32")
    if jnp.dtype(mask_bn.dtype) != jnp.dtype(jnp.uint8):
        raise TypeError("mask_bn dtype must be uint8")
    if jnp.dtype(ct_bn.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("ct_bn dtype must be float32")

    if weights.size != local_targets.size:
        raise ValueError("weights and local_targets sizes must match")
    if indptr.size < 2:
        raise ValueError("indptr must describe at least one CSR row")
    if mask_bn.shape[0] != ct_bn.shape[0]:
        raise ValueError("mask_bn and ct_bn batch dimensions must match")
    rows = indptr.size - 1
    expected_mask_width = (rows + 7) // 8
    if mask_bn.shape != (ct_bn.shape[0], expected_mask_width):
        raise ValueError(
            "mask_bn shape must be "
            f"{(ct_bn.shape[0], expected_mask_width)}, got {mask_bn.shape}"
        )
    if ct_bn.shape[0] <= 0 or ct_bn.shape[1] <= 0:
        raise ValueError("ct_bn dimensions must be positive")
    tile_count = (ct_bn.shape[1] + _BASE_TILE_SIZE - 1) // _BASE_TILE_SIZE
    expected_offsets_shape = (rows, tile_count + 1)
    if tile_offsets.shape != expected_offsets_shape:
        raise ValueError(
            "tile_offsets shape must be "
            f"{expected_offsets_shape}, got {tile_offsets.shape}"
        )

    _load_bpsa_cuda_module()
    return jax.ffi.ffi_call(
        "tcsr_bpsa.csr_bpsa_dinput_f32",
        jax.ShapeDtypeStruct((ct_bn.shape[0], rows), jnp.float32),
        input_layouts=[(0,), (0,), (0,), (0, 1), (0, 1), (0, 1)],
        output_layouts=(0, 1),
    )(weights, indptr, local_targets, tile_offsets, mask_bn, ct_bn)


def csr_bpsa_dinput_single(
    weights: jax.Array,
    local_targets: jax.Array,
    indptr: jax.Array,
    tile_offsets: jax.Array,
    event: jax.Array,
    ct: jax.Array,
) -> jax.Array:
    """Compute a single-vector BPSA input gradient.

    Parameters
    ----------
    weights : jax.Array
        Float32 CSR values in TileCSR slot order.
    local_targets : jax.Array
        Uint16 target offsets inside 8192-column tiles.
    indptr : jax.Array
        Int32 or int64 CSR row offsets.
    tile_offsets : jax.Array
        Int32 row-local tile boundaries.
    event : jax.Array
        Boolean or float32 source events with shape ``(rows,)``.
    ct : jax.Array
        Float32 output cotangent with shape ``(cols,)``.

    Returns
    -------
    jax.Array
        Masked float32 input gradient with shape ``(rows,)``.

    Raises
    ------
    TypeError
        If an operand dtype is unsupported.
    ValueError
        If an operand rank or logical dimension is invalid.

    Notes
    -----
    This operator is GPU-only. Float events are active only when positive.
    """
    weights = jnp.asarray(weights)
    local_targets = jnp.asarray(local_targets)
    indptr = jnp.asarray(indptr)
    tile_offsets = jnp.asarray(tile_offsets)
    event = jnp.asarray(event)
    ct = jnp.asarray(ct)

    for name, value, rank in (
        ("weights", weights, 1),
        ("local_targets", local_targets, 1),
        ("indptr", indptr, 1),
        ("tile_offsets", tile_offsets, 2),
        ("event", event, 1),
        ("ct", ct, 1),
    ):
        if value.ndim != rank:
            raise ValueError(f"{name} must be rank {rank}")

    if jnp.dtype(weights.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("weights dtype must be float32")
    if jnp.dtype(local_targets.dtype) != jnp.dtype(jnp.uint16):
        raise TypeError("local_targets dtype must be uint16")
    if jnp.dtype(indptr.dtype) not in (
        jnp.dtype(jnp.int32),
        jnp.dtype(jnp.int64),
    ):
        raise TypeError("indptr dtype must be int32 or int64")
    if jnp.dtype(tile_offsets.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("tile_offsets dtype must be int32")
    if jnp.dtype(event.dtype) not in (
        jnp.dtype(jnp.bool_),
        jnp.dtype(jnp.float32),
    ):
        raise TypeError("event dtype must be bool or float32")
    if jnp.dtype(ct.dtype) != jnp.dtype(jnp.float32):
        raise TypeError("ct dtype must be float32")

    if weights.size != local_targets.size:
        raise ValueError("weights and local_targets slot sizes must match")
    if indptr.size < 2:
        raise ValueError("indptr must describe at least one CSR row")
    rows = indptr.size - 1
    if event.size != rows:
        raise ValueError("event size must equal the CSR row count")
    if ct.size <= 0:
        raise ValueError("ct dimension must be positive")
    tile_count = (ct.size + _BASE_TILE_SIZE - 1) // _BASE_TILE_SIZE
    expected_offsets_shape = (rows, tile_count + 1)
    if tile_offsets.shape != expected_offsets_shape:
        raise ValueError(
            "tile_offsets shape must be "
            f"{expected_offsets_shape}, got {tile_offsets.shape}"
        )

    global _BPSA_SINGLE_CUDA_MODULE
    if _BPSA_SINGLE_CUDA_MODULE is None:
        _BPSA_SINGLE_CUDA_MODULE = load_cuda_file(
            Path(__file__).with_name("bpsa_mv.cu"),
            name="tcsr_bpsa_single",
        )
    event_suffix = "bool" if event.dtype == jnp.bool_ else "float"
    return jax.ffi.ffi_call(
        "tcsr_bpsa_single."
        f"csr_bpsa_dinput_single_f32_{event_suffix}",
        jax.ShapeDtypeStruct(event.shape, jnp.float32),
        input_layouts=[(0,), (0,), (0,), (0, 1), (0,), (0,)],
        output_layouts=(0,),
    )(weights, local_targets, indptr, tile_offsets, event, ct)
