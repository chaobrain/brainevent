# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

from pathlib import Path
from typing import Optional, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import (
    _check_csr_cuda_structure_dtypes,
    _check_csr_structure_dtypes,
    _csr_to_coo,
    namescope,
)
from brainevent._op import load_cuda_file
from brainevent._op import numba_kernel, XLACustomKernel, general_batching_rule
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._typing import Data, Indptr, Index, MatrixShape
from brainevent.config import get_numba_parallel
from brainevent._op.util import dtype_suffix

__all__ = [
    'csrmv',
    'csrmv_p',
    'csrmm',
    'csrmm_p',
]


_TILE_SIZE = 8192


def _validate_tile_metadata(local_targets, tile_offsets, *, shape, nnz):
    """Validate canonical TCSR metadata passed to float primitives."""
    expected_offsets = (shape[0], (shape[1] + _TILE_SIZE - 1) // _TILE_SIZE + 1)
    if local_targets.ndim != 1 or local_targets.shape[0] != nnz:
        raise ValueError("local_targets must be rank one with one entry per nonzero")
    if jnp.dtype(local_targets.dtype) != jnp.dtype(jnp.uint16):
        raise TypeError("local_targets must use uint16")
    if tuple(tile_offsets.shape) != expected_offsets:
        raise ValueError(
            f"tile_offsets must have shape {expected_offsets}, got "
            f"{tuple(tile_offsets.shape)}"
        )
    if jnp.dtype(tile_offsets.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("tile_offsets must use int32")
    return local_targets, tile_offsets


@namescope(static_argnames=("shape", "transpose"))
def csrmv(
    data: Data,
    indices: Index,
    indptr: Indptr,
    v: Data,
    *,
    shape: MatrixShape,
    local_targets: Index,
    tile_offsets: Index,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """Multiply a TCSR sparse matrix by a dense vector.

    Computes ``y = A @ v`` (or ``y = A.T @ v`` when ``transpose=True``)
    where ``A`` is stored in Compressed Sparse Row format and ``v`` is a
    dense vector.  Unlike the binary (event-driven) variant, every element
    of ``v`` contributes to the result regardless of sign or magnitude.

    The function supports physical units via :mod:`brainunit`.  If ``data``
    or ``v`` carry units, the result is returned in the corresponding
    product unit.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Non-zero values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights or ``(1,)`` for a single homogeneous weight
        shared across all connections.
    indices : jax.Array or numpy.ndarray
        Column indices of the non-zero elements. Shape ``(nse,)`` with
        ``int32`` dtype.
    indptr : jax.Array or numpy.ndarray
        Row index pointer array. Shape ``(shape[0] + 1,)`` with ``int32``
        or ``int64`` dtype. Canonical TCSR uses ``int64``.
    v : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense vector.  Shape ``(shape[0],)`` when ``transpose=True`` or
        ``(shape[1],)`` when ``transpose=False``.
    local_targets : jax.Array
        Uint16 target offsets within each 8192-neuron metadata tile.
    tile_offsets : jax.Array
        Int32 row-local sparse boundaries for every metadata tile.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix ``A``.
    transpose : bool, optional
        If ``True``, the sparse matrix is transposed before multiplication,
        i.e. compute ``A.T @ v``.  Default is ``False``.
    backend : str or None, optional
        Compute backend to use.  One of ``'numba'``,
        ``'pallas'``, or ``None`` (auto-select).  Default is ``None``.

    Returns
    -------
    y : jax.Array or brainunit.Quantity
        Result vector.  Shape ``(shape[1],)`` when ``transpose=True`` or
        ``(shape[0],)`` when ``transpose=False``.

    See Also
    --------
    csrmm : CSR matrix--matrix multiplication.
    binary_csrmv : Event-driven (binary) CSR matrix--vector multiplication.

    Notes
    -----
    This operation is differentiable with respect to both ``data`` and
    ``v`` via custom JVP and transpose rules.

    Mathematically, the non-transposed operation computes:

    ``y[i] = sum_{j in nz(i)} A[i, j] * v[j]``

    where ``nz(i)`` denotes the set of column indices with non-zero
    entries in row ``i`` of the CSR matrix.

    When ``transpose=True``, the transposed operation computes:

    ``y[j] = sum_{i in nz_col(j)} A[i, j] * v[i]``

    where ``nz_col(j)`` denotes the set of row indices with non-zero
    entries in column ``j``.

    For homogeneous weights (``data`` of shape ``(1,)``), ``A[i, j]``
    equals the constant ``data[0]`` for all structural non-zero
    positions.

    References
    ----------
    .. [1] Y. Saad, *Iterative Methods for Sparse Linear Systems*,
       2nd ed., SIAM, 2003, ch. 3.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.float import csrmv
        >>> data = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> local_targets = jnp.array([0, 2, 1, 2], dtype=jnp.uint16)
        >>> tile_offsets = jnp.array([[0, 2], [0, 2]], dtype=jnp.int32)
        >>> v = jnp.array([1.0, 2.0, 3.0])
        >>> result = csrmv(
        ...     data, indices, indptr, v, shape=(2, 3),
        ...     local_targets=local_targets, tile_offsets=tile_offsets,
        ...     backend="jax_raw")
        >>> result.shape
        (2,)
    """
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = csrmv_p_call(
        data,
        indices,
        indptr,
        v,
        local_targets,
        tile_offsets,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


def _csrmv_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    if weight_info.size == 1:
        if transpose:
            # [m, k].T @ [m] - cannot parallelize due to race condition
            @numba.njit(fastmath=True)
            def mv(weights, indices, indptr, vector, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(vector.shape[0]):
                    wsp = w * vector[i]
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += wsp

        else:
            # [m, k] @ [k] - can parallelize by row
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mv(weights, indices, indptr, vector, posts):
                w = weights[0]
                for i_m in numba.prange(indptr.shape[0] - 1):
                    r = 0.0
                    for j in range(indptr[i_m], indptr[i_m + 1]):
                        r += vector[indices[j]]
                    posts[i_m] = w * r

    else:
        if transpose:
            # [m, k].T @ [m] - cannot parallelize due to race condition
            @numba.njit(fastmath=True)
            def mv(weights, indices, indptr, vector, posts):
                posts[:] = 0.
                for i in range(vector.shape[0]):
                    sp = vector[i]
                    for j in range(indptr[i], indptr[i + 1]):
                        posts[indices[j]] += weights[j] * sp

        else:
            # [m, k] @ [k] - can parallelize by row
            @numba.njit(parallel=get_numba_parallel(), fastmath=True)
            def mv(weights, indices, indptr, vector, posts):
                for i in numba.prange(indptr.shape[0] - 1):
                    r = 0.0
                    for j in range(indptr[i], indptr[i + 1]):
                        r += weights[j] * vector[indices[j]]
                    posts[i] = r

    def kernel(weights, indices, indptr, vector, local_targets, tile_offsets):
        del local_targets, tile_offsets
        return numba_kernel(mv, outs=kwargs['outs'])(weights, indices, indptr, vector)

    return kernel


def _csrmv_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
    supported = jnp.dtype(weight_info.dtype) in (
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float64),
    )
    if supported:
        if jnp.dtype(kwargs['indptr_info'].dtype) != jnp.dtype(jnp.int64):
            raise TypeError("TCSR float CUDA kernels require int64 indptr")
        if jnp.dtype(kwargs['local_targets_info'].dtype) != jnp.dtype(jnp.uint16):
            raise TypeError("TCSR float CUDA kernels require uint16 local_targets")
        if jnp.dtype(kwargs['tile_offsets_info'].dtype) != jnp.dtype(jnp.int32):
            raise TypeError("TCSR float CUDA kernels require int32 tile_offsets")
        load_cuda_file(
            Path(__file__).parent.joinpath('float_csrmv.cu'),
            name='tcsr_float_csrmv',
            allow_cuda_graph=False,
        )
        out_info = kwargs['outs']
        wt_sfx = dtype_suffix(weight_info.dtype)
        homo = '_homo' if weight_info.size == 1 else ''
        kernel_name = (
            f'tcsr_float_csrmv.csrmv_xw_wpr{homo}{wt_sfx}'
            if transpose
            else f'tcsr_float_csrmv.csrmv_wx_tile{homo}{wt_sfx}'
        )

        def kernel(weights, indices, indptr, vector, local_targets, tile_offsets):
            vector = vector.astype(weight_info.dtype)
            if transpose:
                return jax.ffi.ffi_call(kernel_name, out_info)(
                    weights,
                    indices,
                    indptr,
                    local_targets,
                    tile_offsets,
                    vector,
                )
            return jax.ffi.ffi_call(kernel_name, out_info)(
                weights, indptr, local_targets, tile_offsets, vector
            )

    else:
        def kernel(weights, indices, indptr, vector, local_targets, tile_offsets):
            del local_targets, tile_offsets
            m, k = kwargs['shape']
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=kwargs['indices_info'].size,
            )
            vector = vector.astype(weight_info.dtype)
            physical_weights = weights[0] if weight_info.size == 1 else weights
            if transpose:
                result = jnp.zeros(k, dtype=weight_info.dtype).at[indices].add(
                    physical_weights * vector[row_ids]
                )
            else:
                result = jnp.zeros(m, dtype=weight_info.dtype).at[row_ids].add(
                    physical_weights * vector[indices]
                )
            return (result,)

    return kernel


def _csrmv_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for CSR matrix-vector multiplication with float weights.

    Mirrors the structure of the binary (event-driven) ``jax_raw`` kernel but
    uses every element of ``vector`` directly instead of gating on events. It
    backs the ``jax_raw`` gradient path: the binary ``binary_csrmv`` transpose
    rule delegates the vector cotangent to ``csrmv`` with the same backend.
    """
    m, k = shape
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, indices, indptr, vector, local_targets, tile_offsets):
            del local_targets, tile_offsets
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            v_row = vector[row_ids].astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(k, dtype=out_dtype).at[indices].add(w * v_row),)
    else:
        def kernel(weights, indices, indptr, vector, local_targets, tile_offsets):
            del local_targets, tile_offsets
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            v_col = vector[indices].astype(out_dtype)
            w = weights[0] if is_homo else weights
            return (jnp.zeros(m, dtype=out_dtype).at[row_ids].add(w * v_col),)

    return kernel


def _csrmv_jvp_v(
    v_dot,
    data,
    indices,
    indptr,
    v,
    local_targets,
    tile_offsets,
    *,
    shape,
    transpose,
    **kwargs,
):
    return [csrmv(
        data,
        indices,
        indptr,
        v_dot,
        shape=shape,
        local_targets=local_targets,
        tile_offsets=tile_offsets,
        transpose=transpose,
        backend=kwargs['backend'],
    )]


def _csrmv_jvp_weights(
    data_dot,
    data,
    indices,
    indptr,
    v,
    local_targets,
    tile_offsets,
    *,
    shape,
    transpose,
    **kwargs,
):
    return csrmv_p_call(
        data_dot,
        indices,
        indptr,
        v,
        local_targets,
        tile_offsets,
        shape=shape,
        transpose=transpose,
        backend=kwargs['backend'],
    )


def _csrmv_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    vector,
    local_targets,
    tile_offsets,
    *,
    shape,
    transpose,
    **kwargs,
):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(vector):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(vector)
        else:
            ct_events = csrmv(
                data,
                indices,
                indptr,
                ct,
                shape=shape,
                local_targets=local_targets,
                tile_offsets=tile_offsets,
                transpose=not transpose,
                backend=kwargs['backend'],
            )
        return data, indices, indptr, ct_events, local_targets, tile_offsets
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_values = csrmv_p_call(
                    jnp.ones(1, dtype=data.aval.dtype),
                    indices,
                    indptr,
                    vector,
                    local_targets,
                    tile_offsets,
                    shape=shape,
                    transpose=transpose,
                    backend=kwargs['backend'],
                )[0]
                ct_values = jnp.inner(ct, ct_values).reshape(*data.aval.shape)
            else:  # heterogeneous values
                row, col = _csr_to_coo(indices, indptr)
                ct_values = vector[row] * ct[col] if transpose else vector[col] * ct[row]
        return ct_values, indices, indptr, vector, local_targets, tile_offsets


def _csrmv_batching(args, axes, **kwargs):
    axes = tuple(axes)
    if any(axis is not None for index, axis in enumerate(axes) if index != 3):
        raise NotImplementedError(
            "TCSR float batching only supports a mapped dense operand"
        )
    dense_axis = axes[3]
    if dense_axis is None:
        return general_batching_rule(csrmv_p, args, axes, **kwargs)
    if args[3].ndim != 2:
        raise ValueError("batched TCSR float MV requires a rank-two operand")
    vector_bn = jnp.moveaxis(args[3], dense_axis, 0)
    result = csrmm_p_call(
        args[0],
        args[1],
        args[2],
        vector_bn,
        args[4],
        args[5],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
    )
    return result, [0]


def _csrmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indptr = np.arange(n_pre + 1, dtype=np.int64) * n_conn
            row_indices = np.arange(n_conn, dtype=np.int32)
            indices = np.tile(row_indices, n_pre)
            local_targets = jnp.asarray(indices, dtype=jnp.uint16)
            tile_offsets = jnp.tile(
                jnp.asarray([[0, n_conn]], dtype=jnp.int32),
                (n_pre, 1),
            )
            weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(n_pre * n_conn, dtype=dtype)
            v_size = n_post if not transpose else n_pre
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (
                        weights,
                        indices,
                        jnp.asarray(indptr),
                        vector,
                        local_targets,
                        tile_offsets,
                    ),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def csrmv_p_call(
    weights,
    indices,
    indptr,
    vector,
    local_targets,
    tile_offsets,
    *,
    shape: Sequence[int],
    transpose: bool,
    backend: Optional[str] = None,
):
    """Call the low-level TCSR matrix-vector primitive.

    Prepares inputs, validates shapes and dtypes, and dispatches the
    ``csrmv_p`` XLA custom kernel to compute ``y = A @ v`` (or
    ``y = A.T @ v``), where ``A`` is a CSR matrix and ``v`` is a dense
    vector.

    Parameters
    ----------
    weights : jax.Array
        Non-zero values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights, ``(1,)`` for a homogeneous weight, or a
        scalar (automatically promoted to shape ``(1,)``).
    indices : jax.Array
        Column indices of non-zero elements. Shape ``(nse,)`` with ``int32``
        dtype.
    indptr : jax.Array
        Row index pointer array. Shape ``(shape[0] + 1,)`` with ``int32``
        or ``int64`` dtype. Canonical TCSR uses ``int64``.
    vector : jax.Array
        Dense vector.  Shape ``(shape[0],)`` when ``transpose=True`` or
        ``(shape[1],)`` when ``transpose=False``.
    local_targets : jax.Array
        Uint16 target offsets within each 8192-neuron metadata tile.
    tile_offsets : jax.Array
        Int32 row-local sparse boundaries for every metadata tile.
    shape : sequence of int
        Two-element sequence ``(m, k)`` giving the logical shape of the
        sparse matrix.
    transpose : bool
        If ``True``, transpose the sparse matrix before multiplication.
    backend : str or None, optional
        Compute backend to use.  Default is ``None`` (auto-select).

    Returns
    -------
    list of jax.Array
        A single-element list containing the result vector.  Shape
        ``(shape[1],)`` when ``transpose=True`` or ``(shape[0],)`` when
        ``transpose=False``.

    Raises
    ------
    AssertionError
        If ``indices`` is not ``int32`` or ``indptr`` is neither ``int32``
        nor ``int64``.
    AssertionError
        If ``indptr`` or ``indices`` is not 1-D.
    AssertionError
        If ``weights`` does not have a floating-point dtype.
    AssertionError
        If there is a shape mismatch between ``vector`` and the sparse
        matrix ``shape`` (considering the ``transpose`` flag).

    See Also
    --------
    csrmv : High-level wrapper with unit support.

    Notes
    -----
    Scalar ``weights`` (0-d arrays) are automatically promoted to
    shape ``(1,)`` to indicate a homogeneous weight across all
    connections.

    The computation performed is:

    ``y[i] = sum_{j in nz(i)} w[j] * v[j]``  (non-transposed)

    ``y[j] = sum_{i in nz_col(j)} w[i] * v[i]``  (transposed)

    where ``w[j]`` is either ``weights[j]`` (heterogeneous) or
    ``weights[0]`` (homogeneous), and ``nz(i)`` is the set of column
    indices with structural non-zeros in row ``i``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.float import csrmv_p_call
        >>> weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> vector = jnp.array([1.0, 2.0, 3.0])
        >>> local_targets = jnp.array([0, 2, 1, 2], dtype=jnp.uint16)
        >>> tile_offsets = jnp.array([[0, 2], [0, 2]], dtype=jnp.int32)
        >>> result = csrmv_p_call(
        ...     weights, indices, indptr, vector, local_targets, tile_offsets,
        ...     shape=(2, 3), transpose=False, backend="jax_raw")
        >>> result[0].shape
        (2,)
    """
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    assert vector.ndim == 1, "Vector must be 1D."
    _check_csr_structure_dtypes(indices, indptr)
    local_targets, tile_offsets = _validate_tile_metadata(
        local_targets,
        tile_offsets,
        shape=shape,
        nnz=indices.size,
    )
    if transpose:
        assert shape[0] == vector.shape[0], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == vector.shape[0], "Shape mismatch for non-transpose operation."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    if weights.size not in (1, indices.size):
        raise ValueError("weights must contain one value or one value per nonzero")

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], weights.dtype)
    )
    return csrmv_p(
        weights,
        indices,
        indptr,
        vector,
        local_targets,
        tile_offsets,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        local_targets_info=jax.ShapeDtypeStruct(
            local_targets.shape, local_targets.dtype
        ),
        tile_offsets_info=jax.ShapeDtypeStruct(
            tile_offsets.shape, tile_offsets.dtype
        ),
    )


csrmv_p = XLACustomKernel(
    'tcsr_csrmv',
    doc="""
Low-level XLA custom-kernel primitive for ``csrmv``.

This ``XLACustomKernel`` instance dispatches the CSR sparse matrix-vector multiplication with floating-point weights
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

All elements of the input vector contribute to the result, regardless of sign or magnitude,
performing standard sparse matrix-vector multiplication with explicit floating-point weights.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``csrmv_p.available_backends(platform)``,
and the default backend can be configured with ``csrmv_p.set_default(platform, backend)``.

See Also
--------
csrmv : High-level user-facing function wrapper.
"""
)
csrmv_p.def_numba_kernel(_csrmv_numba_kernel_generator)
csrmv_p.def_cuda_raw_kernel(_csrmv_cuda_kernel, asdefault=True)
csrmv_p.def_kernel('jax_raw', 'cpu', _csrmv_jax_kernel)
csrmv_p.def_kernel('jax_raw', 'gpu', _csrmv_jax_kernel)
csrmv_p.def_kernel('jax_raw', 'tpu', _csrmv_jax_kernel)
csrmv_p.def_jvp_rule2(
    _csrmv_jvp_weights,
    None,
    None,
    _csrmv_jvp_v,
    None,
    None,
)
csrmv_p.def_transpose_rule(_csrmv_transpose_rule)
csrmv_p.def_batching_rule(_csrmv_batching)
csrmv_p.def_call(csrmv_p_call)
csrmv_p.def_tags('csr', 'float')
csrmv_p.def_benchmark_data(_csrmv_benchmark_data)


@namescope(static_argnames=("shape", "transpose"))
def csrmm(
    data: Data,
    indices: Index,
    indptr: Indptr,
    B: Data,
    *,
    shape: MatrixShape,
    local_targets: Index,
    tile_offsets: Index,
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Data:
    """Multiply batches of dense vectors by a TCSR sparse matrix.

    With canonical ``A.shape == (m, k)``, ``transpose=False`` computes a
    batch of ``A @ x`` products and ``transpose=True`` computes a batch of
    ``x @ A`` products. Inputs and outputs always use batch-neuron layout.

    The function supports physical units via :mod:`brainunit`.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Non-zero values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights or ``(1,)`` for a single homogeneous weight.
    indices : jax.Array or numpy.ndarray
        Column indices of the non-zero elements.  Shape ``(nse,)`` with
        integer dtype.
    indptr : jax.Array or numpy.ndarray
        Row index pointer array. Shape ``(shape[0] + 1,)`` with ``int32``
        or ``int64`` dtype. Canonical TCSR uses ``int64``.
    B : jax.Array, numpy.ndarray, or brainunit.Quantity
        Dense BN matrix. Shape ``(batch, shape[0])`` when ``transpose=True``
        or ``(batch, shape[1])`` when ``transpose=False``.
    local_targets : jax.Array
        Uint16 target offsets within each 8192-neuron metadata tile.
    tile_offsets : jax.Array
        Int32 row-local sparse boundaries for every metadata tile.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix ``A``.
    transpose : bool, optional
        If ``True``, transpose ``A`` before multiplication.  Default is
        ``False``.
    backend : str or None, optional
        Compute backend.  One of ``'numba'``, ``'pallas'``, or
        ``None`` (auto-select).  Default is ``None``.

    Returns
    -------
    C : jax.Array or brainunit.Quantity
        BN result. Shape ``(batch, shape[1])`` when ``transpose=True`` or
        ``(batch, shape[0])`` when ``transpose=False``.

    See Also
    --------
    csrmv : CSR matrix--vector multiplication.
    binary_csrmm : Event-driven (binary) CSR matrix--matrix multiplication.

    Notes
    -----
    Custom JVP and transpose rules are provided for automatic
    differentiation with respect to ``data`` and ``B``.

    Mathematically, the non-transposed operation computes:

    ``C[b, i] = sum_{j in nz(i)} A[i, j] * B[b, j]``

    where ``nz(i)`` denotes the set of column indices with non-zero
    entries in row ``i`` of the CSR matrix.

    When ``transpose=True``, the transposed operation computes:

    ``C[b, j] = sum_{i in nz_col(j)} B[b, i] * A[i, j]``

    where ``nz_col(j)`` denotes the set of row indices with non-zero
    entries in column ``j``.

    For homogeneous weights (``data`` of shape ``(1,)``), ``A[i, j]``
    equals the constant ``data[0]`` for all structural non-zero
    positions.

    References
    ----------
    .. [1] F. G. Gustavson, "Two Fast Algorithms for Sparse Matrices:
       Multiplication and Permuted Transposition," *ACM Transactions on
       Mathematical Software*, vol. 4, no. 3, pp. 250--269, 1978.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.float import csrmm
        >>> data = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> local_targets = jnp.array([0, 2, 1, 2], dtype=jnp.uint16)
        >>> tile_offsets = jnp.array([[0, 2], [0, 2]], dtype=jnp.int32)
        >>> B = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        >>> result = csrmm(
        ...     data, indices, indptr, B, shape=(2, 3),
        ...     local_targets=local_targets, tile_offsets=tile_offsets,
        ...     backend="jax_raw")
        >>> result.shape
        (2, 2)
    """
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        local_targets,
        tile_offsets,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def _csrmm_numba_kernel_generator(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba  # pylint: disable=import-outside-toplevel

    homogeneous = weight_info.size == 1
    if transpose:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True)
        def mm(weights, indices, indptr, matrix_bn, output_bn):
            for batch in numba.prange(matrix_bn.shape[0]):
                for col in range(output_bn.shape[1]):
                    output_bn[batch, col] = 0.0
                for row in range(indptr.shape[0] - 1):
                    dense_value = matrix_bn[batch, row]
                    for entry in range(indptr[row], indptr[row + 1]):
                        weight = weights[0] if homogeneous else weights[entry]
                        output_bn[batch, indices[entry]] += weight * dense_value
    else:
        @numba.njit(parallel=get_numba_parallel(), fastmath=True)
        def mm(weights, indices, indptr, matrix_bn, output_bn):
            for batch in numba.prange(matrix_bn.shape[0]):
                for row in range(indptr.shape[0] - 1):
                    total = 0.0
                    for entry in range(indptr[row], indptr[row + 1]):
                        weight = weights[0] if homogeneous else weights[entry]
                        total += weight * matrix_bn[batch, indices[entry]]
                    output_bn[batch, row] = total

    def kernel(weights, indices, indptr, B, local_targets, tile_offsets):
        del local_targets, tile_offsets
        return numba_kernel(mm, outs=kwargs['outs'])(weights, indices, indptr, B)

    return kernel


def _csrmm_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])
    supported = jnp.dtype(weight_info.dtype) in (
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float64),
    )
    if supported:
        if jnp.dtype(kwargs['indptr_info'].dtype) != jnp.dtype(jnp.int64):
            raise TypeError("TCSR float CUDA kernels require int64 indptr")
        if jnp.dtype(kwargs['local_targets_info'].dtype) != jnp.dtype(jnp.uint16):
            raise TypeError("TCSR float CUDA kernels require uint16 local_targets")
        if jnp.dtype(kwargs['tile_offsets_info'].dtype) != jnp.dtype(jnp.int32):
            raise TypeError("TCSR float CUDA kernels require int32 tile_offsets")
        load_cuda_file(
            Path(__file__).parent.joinpath('float_csrmm.cu'),
            name='tcsr_float_csrmm',
            allow_cuda_graph=False,
        )
        out_info = kwargs['outs']
        wt_sfx = dtype_suffix(weight_info.dtype)
        homo = '_homo' if weight_info.size == 1 else ''
        kernel_name = (
            f'tcsr_float_csrmm.csrmm_xw_tile{homo}{wt_sfx}'
            if transpose
            else f'tcsr_float_csrmm.csrmm_wx_tile{homo}{wt_sfx}'
        )

        def kernel(weights, indices, indptr, B, local_targets, tile_offsets):
            del indices
            B = B.astype(weight_info.dtype)
            return jax.ffi.ffi_call(kernel_name, out_info)(
                weights, indptr, local_targets, tile_offsets, B
            )

    else:
        def kernel(weights, indices, indptr, B, local_targets, tile_offsets):
            del local_targets, tile_offsets
            m, k = kwargs['shape']
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=kwargs['indices_info'].size,
            )
            B = B.astype(weight_info.dtype)
            physical_weights = weights[0] if weight_info.size == 1 else weights
            if transpose:
                result = jnp.zeros((B.shape[0], k), dtype=weight_info.dtype)
                result = result.at[:, indices].add(
                    physical_weights * B[:, row_ids]
                )
            else:
                result = jnp.zeros((B.shape[0], m), dtype=weight_info.dtype)
                result = result.at[:, row_ids].add(
                    physical_weights * B[:, indices]
                )
            return (result,)

    return kernel


def _csrmm_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Pure-JAX kernel for CSR matrix-matrix multiplication with float weights.

    Float analogue of the binary ``jax_raw`` matmul kernel (no event gating).
    Backs the ``jax_raw`` gradient/batching path that delegates to ``csrmm``.
    """
    m, k = shape
    batch = vector_info.shape[0]
    is_homo = (weight_info.size == 1)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if transpose:
        def kernel(weights, indices, indptr, B, local_targets, tile_offsets):
            del local_targets, tile_offsets
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            B_rows = B[:, row_ids].astype(out_dtype)  # [batch, nse]
            w = weights[0] if is_homo else weights[None, :]
            return (
                jnp.zeros((batch, k), dtype=out_dtype)
                .at[:, indices]
                .add(w * B_rows),
            )
    else:
        def kernel(weights, indices, indptr, B, local_targets, tile_offsets):
            del local_targets, tile_offsets
            row_ids = jnp.repeat(
                jnp.arange(m, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=nse,
            )
            B_rows = B[:, indices].astype(out_dtype)  # [batch, nse]
            w = weights[0] if is_homo else weights[None, :]
            return (
                jnp.zeros((batch, m), dtype=out_dtype)
                .at[:, row_ids]
                .add(w * B_rows),
            )

    return kernel


def _csrmm_jvp_data(
    data_dot, data, indices, indptr, B, local_targets, tile_offsets,
    *, shape, transpose, **kwargs,
):
    return [csrmm(
        data_dot,
        indices,
        indptr,
        B,
        shape=shape,
        local_targets=local_targets,
        tile_offsets=tile_offsets,
        transpose=transpose,
        backend=kwargs['backend'],
    )]


def _csrmm_jvp_B(
    B_dot, data, indices, indptr, B, local_targets, tile_offsets,
    *, shape, transpose, **kwargs,
):
    return [csrmm(
        data,
        indices,
        indptr,
        B_dot,
        shape=shape,
        local_targets=local_targets,
        tile_offsets=tile_offsets,
        transpose=transpose,
        backend=kwargs['backend'],
    )]


def _csrmm_transpose_rule(
    ct, data, indices, indptr, B, local_targets, tile_offsets,
    *, shape, transpose, **kwargs,
):
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)
    ct = ct[0]

    if ad.is_undefined_primal(B):
        dB = csrmm(
            data,
            indices,
            indptr,
            ct,
            shape=shape,
            local_targets=local_targets,
            tile_offsets=tile_offsets,
            transpose=not transpose,
            backend=kwargs['backend'],
        )
        return data, indices, indptr, dB, local_targets, tile_offsets
    else:
        B = jnp.asarray(B)
        if data.aval.shape[0] == 1:  # scalar
            r = csrmm_p_call(
                jnp.ones(1, dtype=data.aval.dtype),
                indices,
                indptr,
                B,
                local_targets,
                tile_offsets,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend']
            )[0]
            return (
                jnp.expand_dims(jnp.sum(r * ct), axis=0),
                indices,
                indptr,
                B,
                local_targets,
                tile_offsets,
            )
        else:
            row, col = _csr_to_coo(indices, indptr)
            if transpose:
                d_data = jnp.sum(B[:, row] * ct[:, col], axis=0)
            else:
                d_data = jnp.sum(B[:, col] * ct[:, row], axis=0)
            return d_data, indices, indptr, B, local_targets, tile_offsets


def _csrmm_batching(args, axes, **kwargs):
    axes = tuple(axes)
    if any(axis is not None for index, axis in enumerate(axes) if index != 3):
        raise NotImplementedError(
            "TCSR float batching only supports a mapped dense operand"
        )
    dense_axis = axes[3]
    if dense_axis is None:
        return general_batching_rule(csrmm_p, args, axes, **kwargs)
    if args[3].ndim != 3:
        raise ValueError("nested TCSR float MM batching requires rank three")
    matrix_obn = jnp.moveaxis(args[3], dense_axis, 0)
    outer, batch, neurons = matrix_obn.shape
    result = csrmm_p_call(
        args[0],
        args[1],
        args[2],
        matrix_obn.reshape(outer * batch, neurons),
        args[4],
        args[5],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
    )[0]
    return [result.reshape(outer, batch, result.shape[1])], [0]


def _csrmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indptr = np.arange(n_pre + 1, dtype=np.int64) * n_conn
            row_indices = np.arange(n_conn, dtype=np.int32)
            indices = np.tile(row_indices, n_pre)
            local_targets = jnp.asarray(indices, dtype=jnp.uint16)
            tile_offsets = jnp.tile(
                jnp.asarray([[0, n_conn]], dtype=jnp.int32),
                (n_pre, 1),
            )
            weights = jnp.ones(1, dtype=dtype) if homo else jnp.ones(n_pre * n_conn, dtype=dtype)
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(10, b_rows), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (
                        weights,
                        indices,
                        jnp.asarray(indptr),
                        B,
                        local_targets,
                        tile_offsets,
                    ),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    local_targets,
    tile_offsets,
    *,
    shape: Sequence[int],
    transpose: bool,
    backend: Optional[str] = None,
):
    """Call the low-level BN TCSR matrix-matrix primitive.

    Prepares inputs, validates shapes and dtypes, and dispatches the
    ``csrmm_p`` XLA custom kernel. Both multiplication directions accept and
    return batch-neuron matrices.

    Parameters
    ----------
    weights : jax.Array
        Non-zero values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights, ``(1,)`` for a homogeneous weight, or a
        scalar (automatically promoted to shape ``(1,)``).
    indices : jax.Array
        Column indices of non-zero elements. Shape ``(nse,)`` with ``int32``
        dtype.
    indptr : jax.Array
        Row index pointer array. Shape ``(shape[0] + 1,)`` with ``int32``
        or ``int64`` dtype. Canonical TCSR uses ``int64``.
    B : jax.Array
        Dense BN matrix. Shape ``(batch, shape[0])`` when ``transpose=True``
        or ``(batch, shape[1])`` when ``transpose=False``.
    local_targets : jax.Array
        Uint16 target offsets within each 8192-neuron metadata tile.
    tile_offsets : jax.Array
        Int32 row-local sparse boundaries for every metadata tile.
    shape : sequence of int
        Two-element sequence ``(m, k)`` giving the logical shape of the
        sparse matrix.
    transpose : bool
        If ``True``, transpose the sparse matrix before multiplication.
    backend : str or None, optional
        Compute backend to use.  Default is ``None`` (auto-select).

    Returns
    -------
    list of jax.Array
        Single BN result. Shape ``(batch, shape[1])`` when
        ``transpose=True`` or ``(batch, shape[0])`` when ``transpose=False``.

    Raises
    ------
    AssertionError
        If ``indices`` is not ``int32`` or ``indptr`` is neither ``int32``
        nor ``int64``.
    AssertionError
        If ``indptr`` or ``indices`` is not 1-D.
    AssertionError
        If ``weights`` does not have a floating-point dtype.
    AssertionError
        If there is a shape mismatch between ``B`` and the sparse
        matrix ``shape`` (considering the ``transpose`` flag).

    See Also
    --------
    csrmm : High-level wrapper with unit support.

    Notes
    -----
    Scalar ``weights`` (0-d arrays) are automatically promoted to
    shape ``(1,)`` to indicate a homogeneous weight across all
    connections.

    The computation performed is:

    ``C[b, i] = sum_{j in nz(i)} w[j] * B[b, j]``  (non-transposed)

    ``C[b, j] = sum_{i in nz_col(j)} w[i] * B[b, i]``  (transposed)

    where ``w[j]`` is either ``weights[j]`` (heterogeneous) or
    ``weights[0]`` (homogeneous), and ``nz(i)`` is the set of column
    indices with structural non-zeros in row ``i``.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.float import csrmm_p_call
        >>> weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> local_targets = jnp.array([0, 2, 1, 2], dtype=jnp.uint16)
        >>> tile_offsets = jnp.array([[0, 2], [0, 2]], dtype=jnp.int32)
        >>> B = jnp.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        >>> result = csrmm_p_call(
        ...             weights, indices, indptr, B,
        ...             local_targets, tile_offsets,
        ...             shape=(2, 3), transpose=False, backend="jax_raw")
        >>> result[0].shape
        (2, 2)
    """
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    _check_csr_structure_dtypes(indices, indptr)
    local_targets, tile_offsets = _validate_tile_metadata(
        local_targets,
        tile_offsets,
        shape=shape,
        nnz=indices.size,
    )
    assert B.ndim == 2, "Dense input must use rank-two BN layout."
    if transpose:
        assert shape[0] == B.shape[1], "Shape mismatch for transpose operation."
    else:
        assert shape[1] == B.shape[1], "Shape mismatch for non-transpose operation."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])
    if weights.size not in (1, indices.size):
        raise ValueError("weights must contain one value or one value per nonzero")

    out_info = (
        jax.ShapeDtypeStruct([B.shape[0], shape[1]], weights.dtype)
        if transpose else
        jax.ShapeDtypeStruct([B.shape[0], shape[0]], weights.dtype)
    )
    return csrmm_p(
        weights,
        indices,
        indptr,
        B,
        local_targets,
        tile_offsets,
        outs=[out_info],
        shape=shape,
        transpose=transpose,
        backend=backend,
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        local_targets_info=jax.ShapeDtypeStruct(
            local_targets.shape, local_targets.dtype
        ),
        tile_offsets_info=jax.ShapeDtypeStruct(
            tile_offsets.shape, tile_offsets.dtype
        ),
    )


csrmm_p = XLACustomKernel(
    'tcsr_csrmm',
    doc="""
Low-level XLA custom-kernel primitive for ``csrmm``.

This ``XLACustomKernel`` instance dispatches the CSR sparse matrix-matrix multiplication with floating-point weights
operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

All elements of the input matrix contribute to the result, performing standard
sparse matrix-matrix multiplication with explicit floating-point weights.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``csrmm_p.available_backends(platform)``,
and the default backend can be configured with ``csrmm_p.set_default(platform, backend)``.

See Also
--------
csrmm : High-level user-facing function wrapper.
"""
)
csrmm_p.def_numba_kernel(_csrmm_numba_kernel_generator)
csrmm_p.def_cuda_raw_kernel(_csrmm_cuda_kernel, asdefault=True)
csrmm_p.def_kernel('jax_raw', 'cpu', _csrmm_jax_kernel)
csrmm_p.def_kernel('jax_raw', 'gpu', _csrmm_jax_kernel)
csrmm_p.def_kernel('jax_raw', 'tpu', _csrmm_jax_kernel)
csrmm_p.def_jvp_rule2(
    _csrmm_jvp_data,
    None,
    None,
    _csrmm_jvp_B,
    None,
    None,
)
csrmm_p.def_transpose_rule(_csrmm_transpose_rule)
csrmm_p.def_batching_rule(_csrmm_batching)
csrmm_p.def_call(csrmm_p_call)
csrmm_p.def_tags('csr', 'float')
csrmm_p.def_benchmark_data(_csrmm_benchmark_data)
