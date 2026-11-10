# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Delay-delivery reference helpers: dtypes, tolerances, CSR and event generators."""

from typing import Any

import jax.numpy as jnp
import numpy as np

VALUE_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)
EVENT_DTYPES = (jnp.bool_, jnp.float32)
INDEX_DTYPES = (jnp.int32, jnp.int64)

# Row lengths around the hybrid thresholds: empty, short (<= 128), heavy
# single-task (<= 4096) and heavy multi-task rows.
HYBRID_ROW_LENGTHS = (0, 1, 7, 128, 129, 300, 4096, 4097, 9000, 0, 33)

_TOLERANCE = {
    jnp.dtype(jnp.float16): 1e-2,
    jnp.dtype(jnp.bfloat16): 5e-2,
    jnp.dtype(jnp.float32): 1e-5,
    jnp.dtype(jnp.float64): 1e-12,
}


def dtype_name(dtype: Any) -> str:
    """Short name used in pytest ids, e.g. ``'bf16'`` or ``'bool'``."""
    return {"float16": "f16", "bfloat16": "bf16", "float32": "f32", "float64": "f64",
            "bool": "bool", "int32": "i32", "int64": "i64"}[jnp.dtype(dtype).name]


def tolerance(dtype: Any) -> float:
    """Relative and absolute tolerance for results stored in ``dtype``."""
    return _TOLERANCE[jnp.dtype(dtype)]


def csr_from_lengths(rng: np.random.Generator, lengths: Any, cols: int,
                     index_dtype: Any = np.int64) -> tuple[np.ndarray, np.ndarray]:
    """Random CSR structure with the given row lengths.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random source.
    lengths : sequence of int
        Stored entries per row; columns may repeat within a row.
    cols : int
        Column count (> 0 unless every length is 0).
    index_dtype : dtype, optional
        dtype of ``indptr``.

    Returns
    -------
    indptr, indices : numpy.ndarray
        ``indptr`` in ``index_dtype`` and int32 ``indices``.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    indptr = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = rng.integers(0, max(cols, 1), size=int(indptr[-1]), dtype=np.int32)
    return indptr.astype(index_dtype), indices


def random_values(rng: np.random.Generator, shape: Any, dtype: Any) -> np.ndarray:
    """Values in [-1, 1) rounded to ``dtype`` (returned as a numpy array of ``dtype``)."""
    return np.asarray(rng.uniform(-1.0, 1.0, size=shape)).astype(jnp.dtype(dtype))


def random_events(rng: np.random.Generator, shape: Any, dtype: Any, rate: float = 0.5) -> np.ndarray:
    """Binary events; float32 events mix 1, negative values and NaN among the active ones.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random source.
    shape : int or tuple of int
        Event shape.
    dtype : dtype
        ``bool`` or ``float32``.
    rate : float, optional
        Fraction of active events.

    Returns
    -------
    numpy.ndarray
        Events whose activity is ``events != 0``.
    """
    active = rng.random(shape) < rate
    if jnp.dtype(dtype) == jnp.dtype(jnp.bool_):
        return active
    kind = rng.integers(0, 3, size=shape)
    value = np.where(kind == 0, 1.0, np.where(kind == 1, -2.5, np.nan)).astype(np.float32)
    return np.where(active, value, np.float32(0.0))


def active_mask(events: np.ndarray) -> np.ndarray:
    """Event activity ``events != 0`` (NaN and negative values are active)."""
    return np.asarray(events) != 0


def numpy_project_step(weights: np.ndarray, indices: np.ndarray, indptr: np.ndarray,
                       delays: np.ndarray, events: np.ndarray, head: int,
                       ring: np.ndarray, *, weighted: bool = False) -> np.ndarray:
    """Project original CSR edges into a copy of the ring in float64.

    Parameters
    ----------
    weights, indices, indptr : numpy.ndarray
        Original CSR arrays; a single weight is shared by every edge.
    delays : numpy.ndarray
        Delay per original edge, before any sorting by delay.
    events : numpy.ndarray
        Event vector. Nonzero values, including NaN, are active.
    head : int
        Current ring slot.
    ring : numpy.ndarray
        Existing ring contents, left unchanged.
    weighted : bool, optional
        Multiply active edges by their event values instead of binary activity.

    Returns
    -------
    numpy.ndarray
        Updated float64 ring, including contributions to wrapped slots.
    """
    result = np.asarray(ring, dtype=np.float64).copy()
    values = np.broadcast_to(np.asarray(weights, dtype=np.float64), indices.shape)
    for row in np.flatnonzero(active_mask(events)):
        edges = slice(int(indptr[row]), int(indptr[row + 1]))
        contribution = values[edges] * float(events[row]) if weighted else values[edges]
        np.add.at(result, ((head + delays[edges]) % len(result), indices[edges]), contribution)
    return result


def assert_same_bits(actual: Any, expected: np.ndarray) -> None:
    """Assert bitwise equality (exact copies; NaN-safe also for bfloat16, which numpy cannot compare)."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    view = np.dtype(f"u{expected.dtype.itemsize}")
    np.testing.assert_array_equal(actual.view(view), expected.view(view))


def assert_close(actual: Any, expected: np.ndarray, dtype: Any, scale: float = 1.0) -> None:
    """Compare in float64 with ``scale`` times the tolerance of ``dtype``; NaN must match NaN."""
    tol = tolerance(dtype) * scale
    np.testing.assert_allclose(np.asarray(actual).astype(np.float64), np.asarray(expected).astype(np.float64),
                               rtol=tol, atol=tol, equal_nan=True)


# nnz > 2**31 structure for int64 indptr checks: one inactive leading row of LARGE_HEAD edges,
# then rows whose edge offsets all exceed INT32_MAX (the first short row straddles it).
LARGE_HEAD = 2**31 - 40
LARGE_TAIL = (100, 5000, 3, 600)


def large_structure(cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Row pointers (int64) and the int32 columns of the edges after the leading row.

    The leading row's columns are all 0; tail edge ``k`` uses column ``k % cols``.
    """
    lengths = np.asarray((LARGE_HEAD, *LARGE_TAIL), dtype=np.int64)
    indptr = np.zeros(lengths.size + 1, np.int64)
    np.cumsum(lengths, out=indptr[1:])
    tail = (np.arange(sum(LARGE_TAIL)) % cols).astype(np.int32)
    return indptr, tail


def large_indices(tail: np.ndarray) -> Any:
    """Device int32 column array: LARGE_HEAD zeros followed by ``tail``."""
    return jnp.concatenate([jnp.zeros(LARGE_HEAD, jnp.int32), jnp.asarray(tail)])
