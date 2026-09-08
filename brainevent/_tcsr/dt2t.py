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

"""Provide TCSR per-slot diagonal expansion primitives."""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp

from brainevent._op import XLACustomKernel, load_cuda_file
from brainevent._op.util import dtype_suffix
from brainevent._typing import Data, Indptr, Index, MatrixShape

from .preprocess import TCSBuffers, build_tcsc_mirror, validate_tcsc_mirror

__all__ = [
    "csrmv_dt2t",
    "csrmv_dt2t_p",
    "csrmv_dt2t_p_call",
    "csrmm_dt2t",
    "csrmm_dt2t_p",
    "csrmm_dt2t_p_call",
]


def _validate_tcsr_structure(indices: Index, indptr: Indptr) -> None:
    """Validate the canonical TCSR structure dtype contract."""
    if indices.ndim != 1:
        raise AssertionError("Indices must be 1D.")
    if indptr.ndim != 1:
        raise AssertionError("Indptr must be 1D.")
    if jnp.dtype(indices.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("TCSR dt2t indices must use int32")
    if jnp.dtype(indptr.dtype) != jnp.dtype(jnp.int64):
        raise TypeError("TCSR dt2t indptr must use int64")


def _prepare_direction(
    indices: jax.Array,
    indptr: jax.Array,
    *,
    shape: MatrixShape,
    transpose: bool,
    buffers: TCSBuffers | None,
    permutation: jax.Array | None,
    mirror_enabled: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, bool]:
    """Resolve canonical or mirror arrays before primitive binding."""
    if not transpose:
        if mirror_enabled:
            raise ValueError("transpose=False requires canonical TCSR inputs")
        return indices, indptr, indptr[:0], False

    if mirror_enabled:
        if permutation is None:
            raise ValueError("prepared transpose=True inputs require permutation")
        return indices, indptr, permutation, True

    mirror = None if buffers is None else buffers.tcsc
    if mirror is None:
        if isinstance(indices, jax.core.Tracer) or isinstance(
            indptr, jax.core.Tracer
        ):
            raise RuntimeError(
                "TCSC mirror must be materialized before a mirror-free TCSR "
                "crosses a dynamic JIT boundary"
            )
        with jax.ensure_compile_time_eval():
            mirror = build_tcsc_mirror(indices, indptr, shape=shape)
        if buffers is not None:
            buffers.tcsc = mirror
    if not isinstance(mirror.indices, jax.core.Tracer) and not isinstance(
        mirror.indptr, jax.core.Tracer
    ):
        mirror = validate_tcsc_mirror(
            mirror,
            shape=shape,
            nnz=int(indices.size),
        )
    return mirror.indices, mirror.indptr, mirror.permutation, True


def _validate_prepared_direction(
    *,
    transpose: bool,
    mirror_enabled: bool,
    permutation_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    indptr_info: jax.ShapeDtypeStruct,
) -> None:
    """Validate the selected physical structure at lowering time."""
    if transpose != mirror_enabled:
        expected = "mirror" if transpose else "canonical"
        raise ValueError(f"transpose={transpose} requires prepared {expected} inputs")
    if jnp.dtype(indices_info.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("TCSR dt2t CUDA indices must use int32")
    if jnp.dtype(indptr_info.dtype) != jnp.dtype(jnp.int64):
        raise TypeError("TCSR dt2t CUDA indptr must use int64")
    if jnp.dtype(permutation_info.dtype) != jnp.dtype(jnp.int64):
        raise TypeError("TCSR dt2t CUDA permutation must use int64")
    expected_size = indices_info.size if transpose else 0
    if permutation_info.size != expected_size:
        raise ValueError(
            f"prepared permutation size must be {expected_size}, got "
            f"{permutation_info.size}"
        )


def _row_ids(indptr: Indptr, *, nnz: int) -> jax.Array:
    """Expand compressed row pointers into one row id per physical slot."""
    return jnp.repeat(
        jnp.arange(indptr.size - 1, dtype=indptr.dtype),
        jnp.diff(indptr),
        total_repeat_length=nnz,
    )


def csrmv_dt2t(
    y: Data,
    w: Data,
    indices: Index,
    indptr: Indptr,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    buffers: TCSBuffers | None = None,
    backend: Optional[str] = None,
) -> Data:
    """Expand a neuron vector over TCSR per-slot values.

    Parameters
    ----------
    y : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense vector of shape ``(shape[0],)`` for the non-transposed route or
        ``(shape[1],)`` for the transposed route.
    w : jax.Array, numpy.ndarray, or brainunit.Quantity
        Per-slot values with shape ``(nnz,)`` in canonical TCSR order.
    indices : jax.Array
        Canonical int32 TCSR target indices.
    indptr : jax.Array
        Canonical int64 TCSR row pointers.
    shape : tuple of int
        Canonical sparse matrix shape ``(rows, columns)``.
    transpose : bool, optional
        Index ``y`` by sparse columns when true. Default is false.
    buffers : TCSBuffers, optional
        Shared buffers in which a lazily constructed mirror is cached.
    backend : str, optional
        Compute backend.

    Returns
    -------
    jax.Array or brainunit.Quantity
        Per-slot output with the same shape, order, dtype, and unit as ``w``.

    Notes
    -----
    The unit of ``y`` is deliberately ignored to match CSR dt2t semantics.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.dt2t import csrmv_dt2t
        >>> y = jnp.array([2.0, 3.0])
        >>> w = jnp.array([5.0, 7.0])
        >>> indices = jnp.array([0, 1], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 1, 2], dtype=jnp.int64)
        >>> csrmv_dt2t(y, w, indices, indptr, shape=(2, 2), backend="jax_raw")
        Array([10., 21.], dtype=float32)
    """
    w, w_unit = u.split_mantissa_unit(w)
    y, _ = u.split_mantissa_unit(y)
    w = jnp.asarray(w)
    y = jnp.asarray(y)
    result = csrmv_dt2t_p_call(
        y,
        w,
        indices,
        indptr,
        shape=shape,
        transpose=transpose,
        buffers=buffers,
        backend=backend,
    )[0]
    return u.maybe_decimal(result * w_unit)


def csrmv_dt2t_p_call(
    y: jax.Array,
    w: jax.Array,
    indices: Index,
    indptr: Indptr,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    buffers: TCSBuffers | None = None,
    backend: Optional[str] = None,
    permutation: jax.Array | None = None,
    mirror_enabled: bool = False,
):
    """Validate and bind the TCSR vector dt2t primitive.

    Parameters
    ----------
    y : jax.Array
        Dense neuron vector.
    w : jax.Array
        Canonical per-slot values.
    indices : jax.Array
        Canonical or prepared mirror int32 indices.
    indptr : jax.Array
        Canonical or prepared mirror int64 row pointers.
    shape : tuple of int
        Logical sparse matrix shape.
    transpose : bool, optional
        Select indexed column expansion when true. Default is false.
    buffers : TCSBuffers, optional
        Shared owner used to cache a mirror for an unprepared T call.
    backend : str, optional
        Compute backend.
    permutation : jax.Array, optional
        Prepared physical-slot to value-slot permutation.
    mirror_enabled : bool, optional
        Whether ``indices`` and ``indptr`` are already mirror inputs.

    Returns
    -------
    tuple of jax.Array
        Single per-slot output followed by no auxiliary values.

    Raises
    ------
    AssertionError
        If ranks, shapes, or value dtypes violate the CSR dt2t contract.
    TypeError
        If structure dtypes violate the TCSR contract.
    RuntimeError
        If a required mirror cannot be constructed across a traced boundary.
    """
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    shape = (int(shape[0]), int(shape[1]))
    _validate_tcsr_structure(indices, indptr)
    if y.dtype != w.dtype:
        raise AssertionError(
            f"y and w must have the same dtype, but got {y.dtype} and {w.dtype}."
        )
    if y.ndim != 1 or w.ndim != 1:
        raise AssertionError("y and w must both be 1D.")
    if not jnp.issubdtype(w.dtype, jnp.floating):
        raise AssertionError("Weights must be a floating-point type.")
    if w.shape != indices.shape:
        raise AssertionError(
            f"Weights shape mismatch, expected {indices.shape}, got {w.shape}."
        )
    expected_y = int(shape[1] if transpose else shape[0])
    if y.shape[0] != expected_y:
        raise AssertionError(
            f"y length mismatch, expected {expected_y}, got {y.shape[0]}."
        )

    indices, indptr, permutation, mirror_enabled = _prepare_direction(
        indices,
        indptr,
        shape=shape,
        transpose=transpose,
        buffers=buffers,
        permutation=permutation,
        mirror_enabled=mirror_enabled,
    )
    if indptr.shape != (expected_y + 1,):
        raise AssertionError(
            f"Selected indptr shape mismatch, expected {(expected_y + 1,)}, "
            f"got {indptr.shape}."
        )
    return csrmv_dt2t_p(
        y,
        w,
        indices,
        indptr,
        permutation,
        outs=[jax.ShapeDtypeStruct(w.shape, w.dtype)],
        shape=shape,
        transpose=transpose,
        backend=backend,
        mirror_enabled=mirror_enabled,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        permutation_info=jax.ShapeDtypeStruct(
            permutation.shape, permutation.dtype
        ),
        w_info=jax.ShapeDtypeStruct(w.shape, w.dtype),
    )


def _csrmv_dt2t_prepared(
    y: Data,
    w: Data,
    indices: Index,
    indptr: Indptr,
    permutation: jax.Array,
    *,
    shape: MatrixShape,
    backend: Optional[str],
) -> Data:
    """Call the indexed MV route with a selected transpose structure."""
    w, w_unit = u.split_mantissa_unit(w)
    y, _ = u.split_mantissa_unit(y)
    w = jnp.asarray(w)
    y = jnp.asarray(y)
    result = csrmv_dt2t_p_call(
        y,
        w,
        indices,
        indptr,
        shape=shape,
        transpose=True,
        backend=backend,
        permutation=permutation,
        mirror_enabled=True,
    )[0]
    return u.maybe_decimal(result * w_unit)


def _csrmv_dt2t_jax_kernel(*, transpose: bool, **kwargs):
    """Build the pure-JAX vector diagonal-expansion kernel."""
    nnz = kwargs["indices_info"].size

    def kernel(y, w, indices, indptr, permutation):
        del indices
        rows = _row_ids(indptr, nnz=nnz)
        values = w if not transpose else w[permutation]
        expanded = values * y[rows]
        if transpose:
            return (jnp.zeros_like(w).at[permutation].set(expanded),)
        return (expanded,)

    return kernel


def _csrmv_dt2t_cuda_kernel(
    *,
    transpose: bool,
    mirror_enabled: bool,
    w_info: jax.ShapeDtypeStruct,
    **kwargs,
):
    """Build the CUDA FFI vector diagonal-expansion call."""
    _validate_prepared_direction(
        transpose=transpose,
        mirror_enabled=mirror_enabled,
        permutation_info=kwargs["permutation_info"],
        indices_info=kwargs["indices_info"],
        indptr_info=kwargs["indptr_info"],
    )
    load_cuda_file(Path(__file__).with_suffix(".cu"), name="tcsr_dt2t")
    direction = "t_indexed" if transpose else "nt"
    kernel_name = f"tcsr_dt2t.csrmv_dt2t_{direction}{dtype_suffix(w_info.dtype)}"

    def kernel(y, w, indices, indptr, permutation):
        return jax.ffi.ffi_call(kernel_name, kwargs["outs"])(
            y, w, indices, indptr, permutation
        )

    return kernel


def _csrmv_dt2t_jvp_y(
    y_dot, y, w, indices, indptr, permutation, *, shape, transpose, **kwargs
):
    return csrmv_dt2t_p_call(
        y_dot,
        w,
        indices,
        indptr,
        shape=shape,
        transpose=transpose,
        backend=kwargs["backend"],
        permutation=permutation,
        mirror_enabled=kwargs["mirror_enabled"],
    )


def _csrmv_dt2t_jvp_w(
    w_dot, y, w, indices, indptr, permutation, *, shape, transpose, **kwargs
):
    return csrmv_dt2t_p_call(
        y,
        w_dot,
        indices,
        indptr,
        shape=shape,
        transpose=transpose,
        backend=kwargs["backend"],
        permutation=permutation,
        mirror_enabled=kwargs["mirror_enabled"],
    )


def _normalize_mapped_pair(args, axes):
    """Move mapped data axes to zero and broadcast an unmapped partner."""
    y, w = args[:2]
    y_axis, w_axis = axes[:2]
    if y_axis is None and w_axis is None:
        raise ValueError("dt2t batching requires a mapped y or w operand")
    if y_axis is not None:
        y = y if y_axis == 0 else jnp.moveaxis(y, y_axis, 0)
        mapped_size = y.shape[0]
    else:
        mapped_size = w.shape[w_axis]
    if w_axis is not None:
        w = w if w_axis == 0 else jnp.moveaxis(w, w_axis, 0)
        if w.shape[0] != mapped_size:
            raise ValueError("mapped y and w batch sizes must match")
    else:
        w = jnp.broadcast_to(w, (mapped_size, *w.shape))
    if y_axis is None:
        y = jnp.broadcast_to(y, (mapped_size, *y.shape))
    return y, w


def _csrmv_dt2t_batching(args, axes, **kwargs):
    """Route mapped MV operands through the native BN MM primitive."""
    if any(axis is not None for axis in axes[2:]):
        raise NotImplementedError("TCSR dt2t sparse structure cannot be batched")
    y, w = _normalize_mapped_pair(args, axes)
    result = csrmm_dt2t_p_call(
        y,
        w,
        args[2],
        args[3],
        shape=kwargs["shape"],
        transpose=kwargs["transpose"],
        backend=kwargs["backend"],
        permutation=args[4],
        mirror_enabled=kwargs["mirror_enabled"],
    )[0]
    return (result,), (0,)


csrmv_dt2t_p = XLACustomKernel(
    "tcsr_csrmv_dt2t",
    doc="TCSR per-slot vector diagonal expansion.",
)
csrmv_dt2t_p.def_cuda_raw_kernel(_csrmv_dt2t_cuda_kernel, asdefault=True)
for _platform in ("cpu", "gpu", "tpu"):
    csrmv_dt2t_p.def_kernel("jax_raw", _platform, _csrmv_dt2t_jax_kernel)
csrmv_dt2t_p.def_jvp_rule2(
    _csrmv_dt2t_jvp_y,
    _csrmv_dt2t_jvp_w,
    None,
    None,
    None,
)
csrmv_dt2t_p.def_batching_rule(_csrmv_dt2t_batching)
csrmv_dt2t_p.def_call(csrmv_dt2t_p_call)
csrmv_dt2t_p.def_tags("tcsr", "float")


def csrmm_dt2t(
    y: Data,
    w: Data,
    indices: Index,
    indptr: Indptr,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    buffers: TCSBuffers | None = None,
    backend: Optional[str] = None,
) -> Data:
    """Expand BN neuron values over batched TCSR per-slot values.

    Parameters
    ----------
    y : jax.Array, numpy.ndarray, or brainunit.Quantity
        BN array shaped ``(batch, shape[0])`` for the non-transposed route or
        ``(batch, shape[1])`` for the transposed route.
    w : jax.Array, numpy.ndarray, or brainunit.Quantity
        Batched per-slot values shaped ``(batch, nnz)``.
    indices : jax.Array
        Canonical int32 TCSR target indices.
    indptr : jax.Array
        Canonical int64 TCSR row pointers.
    shape : tuple of int
        Canonical sparse matrix shape ``(rows, columns)``.
    transpose : bool, optional
        Index ``y`` by sparse columns when true. Default is false.
    buffers : TCSBuffers, optional
        Shared buffers in which a lazily constructed mirror is cached.
    backend : str, optional
        Compute backend.

    Returns
    -------
    jax.Array or brainunit.Quantity
        BN per-slot output with the same shape, order, dtype, and unit as
        ``w``.

    Notes
    -----
    The unit of ``y`` is deliberately ignored to match CSR dt2t semantics.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.dt2t import csrmm_dt2t
        >>> y = jnp.array([[2.0, 3.0]])
        >>> w = jnp.array([[5.0, 7.0]])
        >>> indices = jnp.array([0, 1], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 1, 2], dtype=jnp.int64)
        >>> csrmm_dt2t(y, w, indices, indptr, shape=(2, 2), backend="jax_raw")
        Array([[10., 21.]], dtype=float32)
    """
    w, w_unit = u.split_mantissa_unit(w)
    y, _ = u.split_mantissa_unit(y)
    w = jnp.asarray(w)
    y = jnp.asarray(y)
    result = csrmm_dt2t_p_call(
        y,
        w,
        indices,
        indptr,
        shape=shape,
        transpose=transpose,
        buffers=buffers,
        backend=backend,
    )[0]
    return u.maybe_decimal(result * w_unit)


def csrmm_dt2t_p_call(
    y: jax.Array,
    w: jax.Array,
    indices: Index,
    indptr: Indptr,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    buffers: TCSBuffers | None = None,
    backend: Optional[str] = None,
    permutation: jax.Array | None = None,
    mirror_enabled: bool = False,
):
    """Validate and bind the TCSR BN matrix dt2t primitive.

    Parameters
    ----------
    y : jax.Array
        BN neuron values shaped ``(batch, neurons)``.
    w : jax.Array
        BN per-slot values shaped ``(batch, nnz)``.
    indices : jax.Array
        Canonical or prepared mirror int32 indices.
    indptr : jax.Array
        Canonical or prepared mirror int64 row pointers.
    shape : tuple of int
        Logical sparse matrix shape.
    transpose : bool, optional
        Select indexed column expansion when true. Default is false.
    buffers : TCSBuffers, optional
        Shared owner used to cache a mirror for an unprepared T call.
    backend : str, optional
        Compute backend.
    permutation : jax.Array, optional
        Prepared physical-slot to value-slot permutation.
    mirror_enabled : bool, optional
        Whether ``indices`` and ``indptr`` are already mirror inputs.

    Returns
    -------
    tuple of jax.Array
        Single BN per-slot output followed by no auxiliary values.

    Raises
    ------
    AssertionError
        If ranks, shapes, batch sizes, or dtypes violate the CSR contract.
    TypeError
        If structure dtypes violate the TCSR contract.
    RuntimeError
        If a required mirror cannot be constructed across a traced boundary.
    """
    indices = jnp.asarray(indices)
    indptr = jnp.asarray(indptr)
    shape = (int(shape[0]), int(shape[1]))
    _validate_tcsr_structure(indices, indptr)
    if y.dtype != w.dtype:
        raise AssertionError(
            f"y and w must have the same dtype, but got {y.dtype} and {w.dtype}."
        )
    if y.ndim != 2:
        raise AssertionError("y must be 2D (batch, vector).")
    if w.ndim != 2:
        raise AssertionError("w must be 2D (batch, nnz).")
    if not jnp.issubdtype(w.dtype, jnp.floating):
        raise AssertionError("Weights must be a floating-point type.")
    if w.shape[0] != y.shape[0]:
        raise AssertionError(
            f"Batch mismatch, y has batch {y.shape[0]} but w has batch "
            f"{w.shape[0]}."
        )
    if w.shape[1:] != indices.shape:
        raise AssertionError(
            f"Weights shape mismatch, expected {indices.shape}, got {w.shape[1:]}."
        )
    expected_y = int(shape[1] if transpose else shape[0])
    if y.shape[1] != expected_y:
        raise AssertionError(
            f"y neuron dimension mismatch, expected {expected_y}, got "
            f"{y.shape[1]}."
        )

    indices, indptr, permutation, mirror_enabled = _prepare_direction(
        indices,
        indptr,
        shape=shape,
        transpose=transpose,
        buffers=buffers,
        permutation=permutation,
        mirror_enabled=mirror_enabled,
    )
    if indptr.shape != (expected_y + 1,):
        raise AssertionError(
            f"Selected indptr shape mismatch, expected {(expected_y + 1,)}, "
            f"got {indptr.shape}."
        )
    return csrmm_dt2t_p(
        y,
        w,
        indices,
        indptr,
        permutation,
        outs=[jax.ShapeDtypeStruct(w.shape, w.dtype)],
        shape=shape,
        transpose=transpose,
        backend=backend,
        mirror_enabled=mirror_enabled,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        permutation_info=jax.ShapeDtypeStruct(
            permutation.shape, permutation.dtype
        ),
        w_info=jax.ShapeDtypeStruct(w.shape, w.dtype),
    )


def _csrmm_dt2t_jax_kernel(*, transpose: bool, **kwargs):
    """Build the pure-JAX BN matrix diagonal-expansion kernel."""
    nnz = kwargs["indices_info"].size

    def kernel(y, w, indices, indptr, permutation):
        del indices
        rows = _row_ids(indptr, nnz=nnz)
        values = w if not transpose else w[:, permutation]
        expanded = values * y[:, rows]
        if transpose:
            return (jnp.zeros_like(w).at[:, permutation].set(expanded),)
        return (expanded,)

    return kernel


def _csrmm_dt2t_cuda_kernel(
    *,
    transpose: bool,
    mirror_enabled: bool,
    w_info: jax.ShapeDtypeStruct,
    **kwargs,
):
    """Build the CUDA FFI BN matrix diagonal-expansion call."""
    _validate_prepared_direction(
        transpose=transpose,
        mirror_enabled=mirror_enabled,
        permutation_info=kwargs["permutation_info"],
        indices_info=kwargs["indices_info"],
        indptr_info=kwargs["indptr_info"],
    )
    load_cuda_file(Path(__file__).with_suffix(".cu"), name="tcsr_dt2t")
    direction = "t_indexed" if transpose else "nt"
    kernel_name = f"tcsr_dt2t.csrmm_dt2t_{direction}{dtype_suffix(w_info.dtype)}"

    def kernel(y, w, indices, indptr, permutation):
        return jax.ffi.ffi_call(kernel_name, kwargs["outs"])(
            y, w, indices, indptr, permutation
        )

    return kernel


def _csrmm_dt2t_jvp_y(
    y_dot, y, w, indices, indptr, permutation, *, shape, transpose, **kwargs
):
    return csrmm_dt2t_p_call(
        y_dot,
        w,
        indices,
        indptr,
        shape=shape,
        transpose=transpose,
        backend=kwargs["backend"],
        permutation=permutation,
        mirror_enabled=kwargs["mirror_enabled"],
    )


def _csrmm_dt2t_jvp_w(
    w_dot, y, w, indices, indptr, permutation, *, shape, transpose, **kwargs
):
    return csrmm_dt2t_p_call(
        y,
        w_dot,
        indices,
        indptr,
        shape=shape,
        transpose=transpose,
        backend=kwargs["backend"],
        permutation=permutation,
        mirror_enabled=kwargs["mirror_enabled"],
    )


def _csrmm_dt2t_batching(args, axes, **kwargs):
    """Normalize an outer mapped axis while preserving inner BN layout."""
    if any(axis is not None for axis in axes[2:]):
        raise NotImplementedError("TCSR dt2t sparse structure cannot be batched")
    y, w = _normalize_mapped_pair(args, axes)
    outer_size, batch_size, neuron_count = y.shape
    if w.shape[:2] != (outer_size, batch_size):
        raise ValueError("mapped y and w BN dimensions must match")
    flat_y = y.reshape(outer_size * batch_size, neuron_count)
    flat_w = w.reshape(outer_size * batch_size, w.shape[2])
    result = csrmm_dt2t_p_call(
        flat_y,
        flat_w,
        args[2],
        args[3],
        shape=kwargs["shape"],
        transpose=kwargs["transpose"],
        backend=kwargs["backend"],
        permutation=args[4],
        mirror_enabled=kwargs["mirror_enabled"],
    )[0]
    return (result.reshape(outer_size, batch_size, result.shape[1]),), (0,)


csrmm_dt2t_p = XLACustomKernel(
    "tcsr_csrmm_dt2t",
    doc="TCSR per-slot BN matrix diagonal expansion.",
)
csrmm_dt2t_p.def_cuda_raw_kernel(_csrmm_dt2t_cuda_kernel, asdefault=True)
for _platform in ("cpu", "gpu", "tpu"):
    csrmm_dt2t_p.def_kernel("jax_raw", _platform, _csrmm_dt2t_jax_kernel)
csrmm_dt2t_p.def_jvp_rule2(
    _csrmm_dt2t_jvp_y,
    _csrmm_dt2t_jvp_w,
    None,
    None,
    None,
)
csrmm_dt2t_p.def_batching_rule(_csrmm_dt2t_batching)
csrmm_dt2t_p.def_call(csrmm_dt2t_p_call)
csrmm_dt2t_p.def_tags("tcsr", "float")
