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
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp
from jax.interpreters import ad

from brainevent._misc import (
    _check_csr_cuda_structure_dtypes,
    _check_csr_structure_dtypes,
)
from brainevent._op import load_cuda_file
from brainevent._op import XLACustomKernel, general_batching_rule, numba_kernel
from brainevent._op.util import dtype_suffix
from .preprocess_config import (
    get_hybrid_config,
    compile_flags_for_config,
    module_suffix_for_config,
)
from brainevent._typing import Data, Indptr, Index, MatrixShape
from brainevent.config import get_numba_parallel
from .float import csrmv, csrmm, csrmv_p, csrmm_p
from .preprocess import (
    HybridWorkspace,
    TCSBuffers,
    build_tcsc_mirror,
    validate_tcsc_mirror,
    workspace_from_operands,
    workspace_operands,
)
from .sddmm import (
    tcsr_sddmm_dweight_binary,
    tcsr_sddmm_dweight_float,
    tcsr_sddmv_dweight_binary,
    tcsr_sddmv_dweight_float,
)
__all__ = [
    'BINARY_BACKENDS',
    'binary_csrmv',
    'binary_csrmv_p',
    'binary_csrmm',
    'binary_csrmm_p',
]

BINARY_BACKENDS = frozenset({'jax', 'numba', 'cusparse', 'cuda_raw'})
BACKWARD_ALGORITHMS = frozenset({'bptt', 'pp_prop'})
_TILE_SIZE = 8192

_BinaryCsrmvTaskWorkspace = HybridWorkspace
_workspace_task_operands = workspace_operands
_workspace_from_task_operands = workspace_from_operands


def _tile_weight_suffix(weight_dtype, event_dtype) -> str:
    """Validate TileMM dtypes and return the exact weight ABI suffix."""
    weight_dtype = jnp.dtype(weight_dtype)
    if weight_dtype == jnp.dtype(jnp.float32):
        suffix = "f32"
    elif weight_dtype == jnp.dtype(jnp.float64):
        suffix = "f64"
    else:
        raise TypeError(
            f"TCSR CUDA tile MM requires float32 or float64 data, got "
            f"{weight_dtype}"
        )

    event_dtype = jnp.dtype(event_dtype)
    if not (
        event_dtype == jnp.dtype(jnp.bool_)
        or event_dtype == jnp.dtype(jnp.int8)
        or jnp.issubdtype(event_dtype, jnp.floating)
    ):
        raise TypeError(
            "TCSR CUDA tile MM events must be bool, int8, or floating-point, "
            f"got {event_dtype}"
        )
    return suffix


def _tile_task_operands(
    local_targets,
    tile_offsets,
    *,
    shape: MatrixShape,
    nnz: int,
):
    """Validate and return raw tile metadata primitive operands."""
    if tuple(local_targets.shape) != (int(nnz),):
        raise ValueError("tile local_targets shape must equal nnz")
    if jnp.dtype(local_targets.dtype) != jnp.dtype(jnp.uint16):
        raise TypeError("tile local_targets must use uint16")
    tile_count = (int(shape[1]) + _TILE_SIZE - 1) // _TILE_SIZE
    expected_offsets_shape = (int(shape[0]), tile_count + 1)
    if tuple(tile_offsets.shape) != expected_offsets_shape:
        raise ValueError(
            "tile tile_offsets shape must be "
            f"{expected_offsets_shape}, got {tuple(tile_offsets.shape)}"
        )
    if jnp.dtype(tile_offsets.dtype) != jnp.dtype(jnp.int32):
        raise TypeError("tile tile_offsets must use int32")
    return local_targets, tile_offsets


def _validate_backend(backend: Optional[str]) -> None:
    if backend is not None and backend not in BINARY_BACKENDS:
        supported = ", ".join(sorted(BINARY_BACKENDS))
        raise ValueError(
            f"Unsupported TCSR binary backend {backend!r}. "
            f"Supported backends: {supported}."
        )


def _validate_backward_algorithm(backward_algorithm: str) -> None:
    """Validate the TCSR weight-gradient algorithm selector."""
    if backward_algorithm not in BACKWARD_ALGORITHMS:
        supported = ", ".join(sorted(BACKWARD_ALGORITHMS))
        raise ValueError(
            f"Unsupported TCSR backward algorithm {backward_algorithm!r}. "
            f"Supported algorithms: {supported}."
        )


def _prepare_non_transposed_components(
    indices,
    indptr,
    *,
    shape: MatrixShape,
    buffers: TCSBuffers | None,
):
    """Build or reuse the mirror required by a non-transposed route."""
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
    if isinstance(mirror.indices, jax.core.Tracer) or isinstance(
        mirror.indptr, jax.core.Tracer
    ):
        return mirror
    return validate_tcsc_mirror(
        mirror,
        shape=shape,
        nnz=int(indices.size),
    )


def _validate_prepared_direction(
    *, transpose: bool, mirror_enabled: bool
) -> None:
    """Reject primitive inputs whose physical structure is not prepared."""
    if transpose and mirror_enabled:
        raise ValueError("transpose=True requires canonical TCSR inputs")
    if not transpose and not mirror_enabled:
        raise ValueError(
            "transpose=False kernel calls require an enabled TCSC mirror; "
            "call binary_csrmv_p_call or binary_csrmm_p_call first"
        )


def binary_csrmv(
    data: Data,
    indices: Index,
    indptr: Indptr,
    v: Data,
    *,
    shape: MatrixShape,
    workspace,
    local_targets,
    tile_offsets,
    buffers: TCSBuffers | None = None,
    transpose: bool = True,
    backend: Optional[str] = None,
    backward_algorithm: str = "bptt",
) -> Data:
    """Multiply event vector by a tile-ordered sparse weight matrix.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Canonical connection weights, or one homogeneous weight.
    indices : jax.Array or numpy.ndarray
        TCSR target indices.
    indptr : jax.Array or numpy.ndarray
        TCSR row pointer.
    v : jax.Array, numpy.ndarray, or brainunit.Quantity
        Event vector. Positive floating values and true boolean values are
        active.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.
    workspace : HybridWorkspace
        Hybrid task state for the direct TCSR traversal.
    local_targets : jax.Array
        Uint16 target offsets within tiles.
    tile_offsets : jax.Array
        Int32 row-local boundaries for each tile.
    buffers : TCSBuffers or None, optional
        Shared traversal buffers used to cache a lazily built mirror.
    transpose : bool, optional
        Select ``v @ W`` when true or ``W @ v`` when false. Default is true.
    backend : str or None, optional
        One of ``jax``, ``numba``, ``cusparse``, or ``cuda_raw``.
    backward_algorithm : {"bptt", "pp_prop"}, optional
        Weight-gradient rule. ``bptt`` uses binary activity; ``pp_prop``
        preserves Batch1 float eligibility values. Default is ``bptt``.
    Returns
    -------
    y : jax.Array or brainunit.Quantity
        BN-convention result vector.

    Notes
    -----
    ``transpose=False`` prepares a data-free mirror in the routing boundary;
    the public service never requires callers to supply its permutation.
    """
    _validate_backend(backend)
    _validate_backward_algorithm(backward_algorithm)
    workspace_operands(workspace, indptr)
    _tile_task_operands(
        local_targets, tile_offsets, shape=shape, nnz=indices.size
    )
    data, unitd = u.split_mantissa_unit(data)
    v, unitv = u.split_mantissa_unit(v)
    res = binary_csrmv_p_call(
        data,
        indices,
        indptr,
        v,
        workspace,
        local_targets,
        tile_offsets,
        buffers=buffers,
        shape=shape,
        transpose=transpose,
        backend=backend,
        backward_algorithm=backward_algorithm,
    )[0]
    return u.maybe_decimal(res * (unitd * unitv))


def binary_csrmm(
    data: Data,
    indices: Index,
    indptr: Indptr,
    B: Data,
    *,
    shape: MatrixShape,
    workspace,
    local_targets,
    tile_offsets,
    buffers: TCSBuffers | None = None,
    transpose: bool = True,
    backend: Optional[str] = None,
    backward_algorithm: str = "bptt",
) -> Data:
    """Multiply a BN event matrix by a tile-ordered sparse matrix.

    Parameters
    ----------
    data : jax.Array, numpy.ndarray, or brainunit.Quantity
        Canonical connection weights, or one homogeneous weight.
    indices : jax.Array or numpy.ndarray
        TCSR target indices.
    indptr : jax.Array or numpy.ndarray
        TCSR row pointer.
    B : jax.Array, numpy.ndarray, or brainunit.Quantity
        Event matrix in ``(batch, neuron)`` layout.
    shape : tuple of int
        Logical matrix shape ``(n_pre, n_post)``.
    workspace : HybridWorkspace
        Hybrid task state carried by the shared primitive ABI.
    local_targets : jax.Array
        Uint16 target offsets within tiles.
    tile_offsets : jax.Array
        Int32 row-local boundaries for each tile.
    buffers : TCSBuffers or None, optional
        Shared traversal buffers used to cache a lazily built mirror.
    transpose : bool, optional
        Select ``B @ W`` in BN layout when true or ``W @ B`` in NB layout
        when false. Default is true.
    backend : str or None, optional
        One of ``jax``, ``numba``, ``cusparse``, or ``cuda_raw``.
    backward_algorithm : {"bptt", "pp_prop"}, optional
        Weight-gradient rule. ``pp_prop`` preserves float32 eligibility values
        for supported logical batch sizes. Default is ``bptt``.
    Returns
    -------
    C : jax.Array or brainunit.Quantity
        Direction-native result in BN layout for true or NB layout for false.

    Notes
    -----
    Direct CUDA uses the tile kernel. Portable backends preserve the same BN
    contract.
    """
    _validate_backend(backend)
    _validate_backward_algorithm(backward_algorithm)
    workspace_operands(workspace, indptr)
    _tile_task_operands(
        local_targets, tile_offsets, shape=shape, nnz=indices.size
    )
    data, unitd = u.split_mantissa_unit(data)
    B, unitb = u.split_mantissa_unit(B)
    res = binary_csrmm_p_call(
        data,
        indices,
        indptr,
        B,
        workspace,
        local_targets,
        tile_offsets,
        buffers=buffers,
        shape=shape,
        transpose=transpose,
        backend=backend,
        backward_algorithm=backward_algorithm,
    )[0]
    return u.maybe_decimal(res * (unitd * unitb))


def _csrmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    """Build the CPU Numba direct TCSR MV kernel."""
    import numba

    mirror_enabled = kwargs['mirror_enabled']
    _validate_prepared_direction(
        transpose=transpose, mirror_enabled=mirror_enabled
    )
    homogeneous = weight_info.size == 1
    boolean_events = vector_info.dtype == jnp.bool_

    @numba.njit(fastmath=True, nogil=True)
    def mv(data, indices, indptr, permutation, events, output):
        output[:] = 0
        for row in range(indptr.shape[0] - 1):
            active = events[row] if boolean_events else events[row] > 0
            if active:
                for slot in range(indptr[row], indptr[row + 1]):
                    weight_slot = permutation[slot] if mirror_enabled else slot
                    weight = data[0] if homogeneous else data[weight_slot]
                    output[indices[slot]] += weight

    def kernel(data, indices, indptr, events, task_begin, task_end, status,
               local_targets, tile_offsets, permutation):
        del local_targets, tile_offsets
        output = numba_kernel(mv, outs=kwargs['outs'][0])(
            data, indices, indptr, permutation, events
        )
        math_out = output[0] if isinstance(output, (tuple, list)) else output
        return math_out, task_begin, task_end, status

    return kernel


def _binary_csrmv_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Build the pure-JAX direct TCSR MV kernel."""
    mirror_enabled = kwargs['mirror_enabled']
    _validate_prepared_direction(
        transpose=transpose, mirror_enabled=mirror_enabled
    )
    rows, cols = shape if transpose else shape[::-1]
    nnz = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype
    homogeneous = weight_info.size == 1
    boolean_events = vector_info.dtype == jnp.bool_
    if nnz > jnp.iinfo(kwargs['indices_info'].dtype).max:
        raise NotImplementedError(
            "direct TCSR JAX MV requires indptr values to fit indices dtype"
        )

    def kernel(data, indices, indptr, events, task_begin, task_end, status,
               local_targets, tile_offsets, permutation):
        del local_targets, tile_offsets
        indptr = indptr.astype(indices.dtype) if indices.dtype != indptr.dtype else indptr
        row_ids = jnp.repeat(
            jnp.arange(rows, dtype=indptr.dtype),
            jnp.diff(indptr),
            total_repeat_length=nnz,
        )
        active = events[row_ids]
        active = (
            active.astype(out_dtype)
            if boolean_events
            else (active > 0).astype(out_dtype)
        )
        weights = data[0] if homogeneous else (
            data if transpose else data[permutation]
        )
        result = jnp.zeros((cols,), dtype=out_dtype).at[indices].add(
            active * weights
        )
        return result, task_begin, task_end, status

    return kernel


def _binary_csrmv_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed binary CSR SpMV kernel via ``jax.experimental.sparse`` (GPU only)."""
    import jax.experimental.sparse as jsparse
    mirror_enabled = kwargs['mirror_enabled']
    _validate_prepared_direction(
        transpose=transpose, mirror_enabled=mirror_enabled
    )
    m, k = shape if transpose else shape[::-1]
    is_homo = (weight_info.size == 1)
    is_bool = (vector_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if nse > jnp.iinfo(kwargs['indices_info'].dtype).max:
        raise NotImplementedError(
            "direct TCSR cuSPARSE MV requires indptr values to fit indices dtype"
        )

    def kernel(weights, indices, indptr, vector, task_begin, task_end, status,
               local_targets, tile_offsets, permutation):
        del local_targets, tile_offsets
        events = vector.astype(out_dtype) if is_bool else (vector > 0.).astype(out_dtype)
        indptr = indptr.astype(indices.dtype) if indices.dtype != indptr.dtype else indptr
        if is_homo:
            data = jnp.ones(nse, dtype=out_dtype)
            mat = jsparse.CSR((data, indices, indptr), shape=(m, k))
            math_out = jsparse.csr_matvec(mat, events, transpose=True) * weights[0].astype(out_dtype)
            return math_out, task_begin, task_end, status
        physical_weights = weights if transpose else weights[permutation]
        mat = jsparse.CSR((physical_weights.astype(out_dtype), indices, indptr), shape=(m, k))
        math_out = jsparse.csr_matvec(mat, events, transpose=True)
        return math_out, task_begin, task_end, status

    return kernel


def _binary_csrmv_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    mirror_enabled = kwargs['mirror_enabled']
    _validate_prepared_direction(
        transpose=transpose, mirror_enabled=mirror_enabled
    )
    _check_csr_cuda_structure_dtypes(kwargs['indices_info'], kwargs['indptr_info'])

    # Determine if weights are homogeneous or heterogeneous
    is_homo = (weight_info.size == 1)
    homo_suffix = '_homo' if is_homo else '_hetero'

    # Spike type suffix
    spk_suffix = '_bool' if vector_info.dtype == jnp.bool_ else '_float'

    # Weight dtype suffix
    wt_sfx = dtype_suffix(weight_info.dtype)

    config = get_hybrid_config()
    indexed = mirror_enabled and not is_homo
    module = (
        'csr_binary_indexed_csrmv_hybrid'
        if indexed
        else 'csr_binary_csrmv_hybrid'
    ) + module_suffix_for_config(config)
    load_cuda_file(
        Path(__file__).parent.joinpath(
            'binary_indexed_csrmv_hybrid.cu'
            if indexed
            else 'binary_csrmv_hybrid.cu'
        ),
        name=module,
        extra_cuda_cflags=compile_flags_for_config(config),
        allow_cuda_graph=False,
    )
    kernel_name = (
        f'{module}.binary_indexed_csrmv_wat_hybrid_hetero{wt_sfx}{spk_suffix}'
        if indexed
        else f'{module}.binary_csrmv_wat_hybrid{homo_suffix}{wt_sfx}{spk_suffix}'
    )

    def kernel(weights, indices, indptr, vector, task_begin, task_end, status,
               local_targets, tile_offsets, permutation):
        del local_targets, tile_offsets
        if indexed:
            return jax.ffi.ffi_call(
                kernel_name,
                kwargs['outs'],
                input_output_aliases={5: 1, 6: 2, 7: 3},
            )(
                weights,
                indices,
                indptr,
                permutation,
                vector,
                task_begin,
                task_end,
                status,
                task_capacity=kwargs['task_capacity'],
            )
        return jax.ffi.ffi_call(
            kernel_name,
            kwargs['outs'],
            input_output_aliases={4: 1, 5: 2, 6: 3},
        )(
            weights,
            indices,
            indptr,
            vector,
            task_begin,
            task_end,
            status,
            task_capacity=kwargs['task_capacity'],
        )

    return kernel


def _grad_backend(backend, primitive):
    """Backend to use when an autodiff rule defers to a *float* CSR primitive.

    The binary primitives expose GPU-only suffixed backends that the
    plain float :func:`csrmv` / :func:`csrmm` used to form tangents and cotangents
    do not register.  Forwarding such a name would raise
    :class:`~brainevent.KernelFallbackExhaustedError` during the backward pass, so
    map each binary backend to the float ``cuda_raw`` implementation. Unknown
    names are forwarded only when the float primitive explicitly registers
    them; otherwise automatic selection is used.

    Parameters
    ----------
    backend : str or None
        The backend requested for the forward (binary) primitive.
    primitive : XLACustomKernel
        The float primitive the autodiff rule dispatches to.

    Returns
    -------
    str or None
        Float backend used by the derivative primitive.
    """
    del primitive
    return {
        'jax': 'jax_raw',
        'numba': 'numba',
        'cuda_raw': 'cuda_raw',
        'cusparse': None,
        None: None,
    }[backend]


def _csrmv_jvp_v(
    v_dot,
    data,
    indices,
    indptr,
    v,
    task_begin,
    task_end,
    status,
    local_targets,
    tile_offsets,
    permutation,
    *,
    shape,
    transpose,
    **kwargs,
):
    physical_shape = shape if transpose else shape[::-1]
    physical_data = data if transpose or data.shape[0] == 1 else data[permutation]
    return (
        csrmv(physical_data, indices, indptr, v_dot, shape=physical_shape, transpose=True,
              backend=_grad_backend(kwargs['backend'], csrmv_p)),
        jnp.zeros_like(task_begin),
        jnp.zeros_like(task_end),
        jnp.zeros_like(status),
    )


def _csrmv_jvp_weights(
    data_dot,
    data,
    indices,
    indptr,
    v,
    task_begin,
    task_end,
    status,
    local_targets,
    tile_offsets,
    permutation,
    *,
    shape,
    transpose,
    **kwargs,
):
    backend = kwargs['backend']
    workspace = _workspace_from_task_operands(kwargs['task_capacity'], task_begin, task_end, status)
    tangent = binary_csrmv_p_call(
        data_dot,
        indices,
        indptr,
        v,
        shape=shape,
        transpose=transpose,
        backend=backend,
        workspace=workspace,
        local_targets=local_targets,
        tile_offsets=tile_offsets,
        backward_algorithm=kwargs['backward_algorithm'],
        permutation=permutation,
        mirror_enabled=kwargs['mirror_enabled'],
    )[0]
    return tangent, jnp.zeros_like(task_begin), jnp.zeros_like(task_end), jnp.zeros_like(status)


def _csrmv_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    events,
    task_begin,
    task_end,
    status,
    local_targets,
    tile_offsets,
    permutation,
    *,
    shape,
    transpose,
    **kwargs,
):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(permutation):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]
    workspace = _workspace_from_task_operands(kwargs['task_capacity'], task_begin, task_end, status)

    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(events):
        if type(ct) is ad.Zero:
            ct_events = ad.Zero(events)
        else:
            physical_shape = shape if transpose else shape[::-1]
            physical_data = (
                data
                if transpose or data.shape[0] == 1
                else data[permutation]
            )
            ct_events = csrmv(physical_data, indices, indptr, ct,
                              shape=physical_shape, transpose=False,
                              backend=_grad_backend(kwargs['backend'], csrmv_p))
        return (
            data,
            indices,
            indptr,
            ct_events,
            ad.Zero(task_begin),
            ad.Zero(task_end),
            ad.Zero(status),
            ad.Zero(local_targets),
            ad.Zero(tile_offsets),
            ad.Zero(permutation),
        )
    else:
        if type(ct) is ad.Zero:
            ct_values = ad.Zero(data)
        else:
            if kwargs['backend'] == 'cuda_raw':
                sampled_gradient = (
                    tcsr_sddmv_dweight_float
                    if kwargs['backward_algorithm'] == 'pp_prop'
                    else tcsr_sddmv_dweight_binary
                )
                slot_values = sampled_gradient(
                    events,
                    ct,
                    indices,
                    indptr,
                    local_targets,
                    tile_offsets,
                    transpose=True,
                )
            else:
                row_ids = jnp.repeat(
                    jnp.arange(indptr.shape[0] - 1, dtype=indptr.dtype),
                    jnp.diff(indptr),
                    total_repeat_length=indices.size,
                )
                activity = (
                    events
                    if kwargs['backward_algorithm'] == 'pp_prop'
                    else (events > 0).astype(ct.dtype)
                )
                slot_values = activity[row_ids] * ct[indices]
            if data.aval.shape[0] == 1:
                ct_values = jnp.sum(slot_values).reshape(*data.aval.shape)
            elif transpose:
                ct_values = slot_values
            else:
                ct_values = jnp.zeros(data.aval.shape, data.aval.dtype)
                ct_values = ct_values.at[permutation].add(slot_values)
        return (
            ct_values,
            indices,
            indptr,
            events,
            ad.Zero(task_begin),
            ad.Zero(task_end),
            ad.Zero(status),
            ad.Zero(local_targets),
            ad.Zero(tile_offsets),
            ad.Zero(permutation),
        )


def _csrmv_batching(args, axes, **kwargs):
    axes = tuple(axes)
    task_capacity = kwargs.get('task_capacity', args[4].shape[0])
    if axes == (None, None, None, 0, None, None, None, None, None, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        events = args[3] if kwargs['transpose'] else args[3].T
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            events,
            _workspace_from_task_operands(task_capacity, args[4], args[5], args[6]),
            args[7],
            args[8],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
            backward_algorithm=kwargs['backward_algorithm'],
            permutation=args[9],
            mirror_enabled=kwargs['mirror_enabled'],
        )
        math_out = r[0] if kwargs['transpose'] else r[0].T
        return (math_out, args[4], args[5], args[6]), (0, None, None, None)

    if axes[3] is not None and axes[3] != 0:
        raise NotImplementedError(
            "binary CSRMV batching requires the event batch dimension on axis 0"
        )

    def prepared_call(*call_args, **call_kwargs):
        workspace = _workspace_from_task_operands(
            call_kwargs['task_capacity'],
            call_args[4],
            call_args[5],
            call_args[6],
        )
        return binary_csrmv_p_call(
            call_args[0], call_args[1], call_args[2], call_args[3],
            workspace, call_args[7], call_args[8],
            shape=call_kwargs['shape'],
            transpose=call_kwargs['transpose'],
            backend=call_kwargs['backend'],
            backward_algorithm=call_kwargs['backward_algorithm'],
            permutation=call_args[9],
            mirror_enabled=call_kwargs['mirror_enabled'],
        )

    return general_batching_rule(prepared_call, args, axes, **kwargs)


def binary_csrmv_p_call(
    weights,
    indices,
    indptr,
    vector,
    workspace,
    local_targets,
    tile_offsets,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
    backward_algorithm: str = "bptt",
    buffers: TCSBuffers | None = None,
    permutation=None,
    mirror_enabled: bool = False,
):
    """
    Low-level primitive call for event-driven CSR matrix--vector
    multiplication.

    Prepares inputs, validates shapes and dtypes, and dispatches the
    ``binary_csrmv_p`` XLA custom kernel. A false route lazily replaces the
    canonical structure with its data-free mirror before binding.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights, ``(1,)`` for a homogeneous weight, or a
        scalar (automatically promoted to shape ``(1,)``).
    indices : jax.Array
        Column indices of non-zero elements.  Shape ``(nse,)`` with dtype
        ``int32``, ``int64``, ``uint32``, or ``uint64``.
    indptr : jax.Array
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.
    vector : jax.Array
        Dense event vector.  Shape ``(shape[0],)`` when
        ``transpose=True`` or ``(shape[1],)`` when ``transpose=False``.
        Dtype may be boolean or floating-point.
    workspace
        Explicit binary task workspace with ``task_capacity``,
        ``task_begin``, ``task_end``, and ``status`` attributes.  This
        low-level function requires the private workspace explicitly; CSR/CSC
        wrappers are intended to hide construction from normal callers.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix.
    transpose : bool
        If ``True``, transpose the sparse matrix before multiplication.
    backend : str or None, optional
        Compute backend to use.  Default is ``None`` (auto-select).
    backward_algorithm : {"bptt", "pp_prop"}, optional
        Weight-gradient interpretation retained as a static primitive
        parameter. Default is ``bptt``.
    local_targets, tile_offsets : jax.Array
        Raw tile metadata carried as primitive operands so mapped MV calls can
        lower to the direct tile MM implementation.
    buffers : TCSBuffers or None, optional
        Shared owner in which a lazily constructed mirror is cached.
    permutation : jax.Array or None, optional
        Prepared mirror-to-canonical slot mapping used by transformation rules.
    mirror_enabled : bool, optional
        Whether the positional structure is already mirror-prepared.

    Returns
    -------
    tuple of jax.Array
        The result vector followed by ``task_begin``, ``task_end``, and
        ``status`` task workspace outputs.  The result vector has shape
        ``(shape[1],)`` when ``transpose=True`` or ``(shape[0],)`` when
        ``transpose=False``.

    Raises
    ------
    AssertionError
        If ``indices`` or ``indptr`` have a dtype other than ``int32``,
        ``int64``, ``uint32``, or ``uint64``.
    AssertionError
        If ``indices`` and ``indptr`` do not share the same dtype.
    AssertionError
        If ``indptr`` or ``indices`` is not 1-D.
    AssertionError
        If ``weights`` does not have a floating-point dtype.
    AssertionError
        If there is a shape mismatch between ``vector`` and the sparse
        matrix ``shape`` (considering the ``transpose`` flag).

    See Also
    --------
    binary_csrmv : High-level wrapper with unit support.

    Notes
    -----
    Scalar ``weights`` (0-d arrays) are automatically promoted to
    shape ``(1,)`` to indicate a homogeneous weight across all
    connections.

    The computation performed is:

    ``y[i] = sum_{j in nz(i)} w[j] * e(v[j])``  (non-transposed)

    ``y[j] = sum_{i in nz_col(j)} w[i] * e(v[i])``  (transposed)

    where ``e(x)`` is ``1`` when ``x`` is ``True`` (boolean) or
    ``x > 0`` (float), and ``0`` otherwise.  ``w[j]`` is either
    ``weights[j]`` (heterogeneous) or ``weights[0]`` (homogeneous).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.binary import binary_csrmv_p_call
        >>> from brainevent._tcsr.preprocess import build_hybrid_workspace
        >>> weights = jnp.array([0.5])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> vector = jnp.array([True, False])
        >>> workspace = build_hybrid_workspace(indptr)
        >>> local_targets = jnp.array([0, 2, 1, 2], dtype=jnp.uint16)
        >>> tile_offsets = jnp.zeros((2, 2), dtype=jnp.int32)
        >>> result = binary_csrmv_p_call(
        ...     weights, indices, indptr, vector, workspace,
        ...     local_targets, tile_offsets,
        ...     shape=(2, 3), transpose=True, backend="jax")[0]
        >>> result.shape
        (3,)
    """
    _validate_backward_algorithm(backward_algorithm)
    if not transpose and not mirror_enabled:
        mirror = _prepare_non_transposed_components(
            indices,
            indptr,
            shape=shape,
            buffers=buffers,
        )
        indices = mirror.indices
        indptr = mirror.indptr
        permutation = mirror.permutation
        workspace = mirror.workspace
        local_targets = mirror.local_targets
        tile_offsets = mirror.tile_offsets
        mirror_enabled = True
    elif transpose and mirror_enabled:
        raise ValueError("transpose=True cannot use mirror-prepared inputs")
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    _check_csr_structure_dtypes(indices, indptr)
    expected_neurons = shape[0] if transpose else shape[1]
    assert expected_neurons == vector.shape[0], "Binary CSRMV neuron dimension mismatch."
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    # Check if weights is a scalar. If so, convert it to a one-dimensional array.
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    # Determine the output shape and data type based on whether the sparse matrix is transposed.
    out_size = shape[1] if transpose else shape[0]
    out_info = jax.ShapeDtypeStruct([out_size], weights.dtype)

    task_capacity, task_begin, task_end, status = _workspace_task_operands(workspace, indptr)
    task_begin_info = jax.ShapeDtypeStruct(task_begin.shape, task_begin.dtype)
    task_end_info = jax.ShapeDtypeStruct(task_end.shape, task_end.dtype)
    status_info = jax.ShapeDtypeStruct(status.shape, status.dtype)

    physical_shape = shape if transpose else shape[::-1]
    local_targets, tile_offsets = _tile_task_operands(
        local_targets,
        tile_offsets,
        shape=physical_shape,
        nnz=indices.size,
    )
    if permutation is None:
        permutation = indptr[:0]
    if not transpose:
        assert permutation.ndim == 1, "Mirror permutation must be 1D."
        assert permutation.shape == indices.shape, (
            "Mirror permutation must have the same shape as indices."
        )
        if jnp.dtype(permutation.dtype) != jnp.dtype(indptr.dtype):
            permutation = permutation.astype(indptr.dtype)
    primitive_args = (
        weights,
        indices,
        indptr,
        vector,
        task_begin,
        task_end,
        status,
        local_targets,
        tile_offsets,
        permutation,
    )

    # Call the binary_csrmv_p custom operation to perform the matrix-vector multiplication.
    return binary_csrmv_p(
        *primitive_args,
        # Initialize a zero vector with the output shape and data type.
        outs=(out_info, task_begin_info, task_end_info, status_info),
        shape=shape,
        transpose=transpose,
        backend=backend,
        backward_algorithm=backward_algorithm,
        mirror_enabled=mirror_enabled,
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for v.
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        task_begin_info=task_begin_info,
        task_end_info=task_end_info,
        status_info=status_info,
        task_capacity=task_capacity,
        tile_shape=tuple(shape),
        tile_nnz=int(indices.size),
        tile_size=_TILE_SIZE,
        permutation_info=jax.ShapeDtypeStruct(
            permutation.shape, permutation.dtype
        ),
    )


binary_csrmv_p = XLACustomKernel(
    'tcsr_binary_csrmv',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_csrmv``.

This primitive dispatches direct TCSR event-vector multiplication to the
``jax``, ``numba``, ``cusparse``, and ``cuda_raw`` backends.

Only entries of ``v`` that are ``True`` (boolean) or positive (float) are considered active events
and contribute to the output, enabling efficient event-driven sparse-dense products commonly used
in spiking neural networks.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_csrmv_p.available_backends(platform)``,
and the default backend can be configured with ``binary_csrmv_p.set_default(platform, backend)``.

See Also
--------
binary_csrmv : High-level user-facing function wrapper.
"""
)
binary_csrmv_p.def_numba_kernel(_csrmv_numba_kernel, asdefault=True)
binary_csrmv_p.def_kernel('jax', 'cpu', _binary_csrmv_jax_kernel)
binary_csrmv_p.def_kernel('jax', 'gpu', _binary_csrmv_jax_kernel)
binary_csrmv_p.def_kernel('jax', 'tpu', _binary_csrmv_jax_kernel, asdefault=True)
binary_csrmv_p.def_kernel('cusparse', 'gpu', _binary_csrmv_cusparse_kernel)
binary_csrmv_p.def_cuda_raw_kernel(_binary_csrmv_cuda_kernel, asdefault=True)
binary_csrmv_p.def_jvp_rule2(
    _csrmv_jvp_weights,
    None,
    None,
    _csrmv_jvp_v,
    None,
    None,
    None,
    None,
    None,
    None,
)
binary_csrmv_p.def_transpose_rule(_csrmv_transpose_rule)
binary_csrmv_p.def_batching_rule(_csrmv_batching)
binary_csrmv_p.def_call(binary_csrmv_p_call)
binary_csrmv_p.def_tags('csr', 'binary')


def _csrmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    """Build the CPU Numba direct TCSR BN MM kernel."""
    import numba

    mirror_enabled = kwargs['mirror_enabled']
    _validate_prepared_direction(
        transpose=transpose, mirror_enabled=mirror_enabled
    )
    homogeneous = weight_info.size == 1
    boolean_events = vector_info.dtype == jnp.bool_

    @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
    def mm(data, indices, indptr, permutation, events, output):
        output[:] = 0
        batch_size = events.shape[0] if transpose else events.shape[1]
        for batch in numba.prange(batch_size):
            for row in range(indptr.shape[0] - 1):
                value = events[batch, row] if transpose else events[row, batch]
                active = value if boolean_events else value > 0
                if active:
                    for slot in range(indptr[row], indptr[row + 1]):
                        weight_slot = permutation[slot] if mirror_enabled else slot
                        weight = data[0] if homogeneous else data[weight_slot]
                        if transpose:
                            output[batch, indices[slot]] += weight
                        else:
                            output[indices[slot], batch] += weight

    def kernel(data, indices, indptr, events_bn, task_begin, task_end, status,
               local_targets, tile_offsets, permutation):
        del local_targets, tile_offsets
        output = numba_kernel(mm, outs=kwargs['outs'][0])(
            data, indices, indptr, permutation, events_bn
        )
        math_out = output[0] if isinstance(output, (tuple, list)) else output
        return math_out, task_begin, task_end, status

    return kernel


def _binary_csrmm_jax_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """Build the pure-JAX direct TCSR BN MM kernel."""
    mirror_enabled = kwargs['mirror_enabled']
    _validate_prepared_direction(
        transpose=transpose, mirror_enabled=mirror_enabled
    )
    rows, cols = shape if transpose else shape[::-1]
    nnz = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype
    homogeneous = weight_info.size == 1
    boolean_events = vector_info.dtype == jnp.bool_
    if nnz > jnp.iinfo(kwargs['indices_info'].dtype).max:
        raise NotImplementedError(
            "direct TCSR JAX MM requires indptr values to fit indices dtype"
        )

    def kernel(data, indices, indptr, events_bn, task_begin, task_end, status,
               local_targets, tile_offsets, permutation):
        del local_targets, tile_offsets
        indptr = indptr.astype(indices.dtype) if indices.dtype != indptr.dtype else indptr
        row_ids = jnp.repeat(
            jnp.arange(rows, dtype=indptr.dtype),
            jnp.diff(indptr),
            total_repeat_length=nnz,
        )
        active = (
            events_bn[:, row_ids]
            if transpose
            else events_bn[row_ids, :]
        )
        active = (
            active.astype(out_dtype)
            if boolean_events
            else (active > 0).astype(out_dtype)
        )
        physical_weights = data if transpose else data[permutation]
        if transpose:
            weights = data[0] if homogeneous else physical_weights[None, :]
            result = jnp.zeros((events_bn.shape[0], cols), dtype=out_dtype)
            result = result.at[:, indices].add(active * weights)
        else:
            weights = data[0] if homogeneous else physical_weights[:, None]
            result = jnp.zeros((cols, events_bn.shape[1]), dtype=out_dtype)
            result = result.at[indices, :].add(active * weights)
        return result, task_begin, task_end, status

    return kernel


def _binary_csrmm_cusparse_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    shape: MatrixShape,
    transpose: bool,
    **kwargs,
):
    """cuSPARSE-backed binary CSR SpMM kernel via ``jax.experimental.sparse`` (GPU only)."""
    import jax.experimental.sparse as jsparse
    mirror_enabled = kwargs['mirror_enabled']
    _validate_prepared_direction(
        transpose=transpose, mirror_enabled=mirror_enabled
    )
    m, k = shape if transpose else shape[::-1]
    is_homo = (weight_info.size == 1)
    is_bool = (vector_info.dtype == jnp.bool_)
    nse = kwargs['indices_info'].size
    out_dtype = kwargs['outs'][0].dtype

    if nse > jnp.iinfo(kwargs['indices_info'].dtype).max:
        raise NotImplementedError(
            "direct TCSR cuSPARSE MM requires indptr values to fit indices dtype"
        )

    def kernel(weights, indices, indptr, B, task_begin, task_end, status,
               local_targets, tile_offsets, permutation):
        del local_targets, tile_offsets
        events = B.astype(out_dtype) if is_bool else (B > 0.).astype(out_dtype)
        events_nb = events.T if transpose else events
        indptr = indptr.astype(indices.dtype) if indices.dtype != indptr.dtype else indptr
        if is_homo:
            data = jnp.ones(nse, dtype=out_dtype)
            mat = jsparse.CSR((data, indices, indptr), shape=(m, k))
            math_out = (
                jsparse.csr_matmat(mat, events_nb, transpose=True)
                * weights[0].astype(out_dtype)
            )
            return (math_out.T if transpose else math_out), task_begin, task_end, status
        physical_weights = weights if transpose else weights[permutation]
        mat = jsparse.CSR((physical_weights.astype(out_dtype), indices, indptr), shape=(m, k))
        math_out = jsparse.csr_matmat(mat, events_nb, transpose=True)
        math_out = math_out.T if transpose else math_out
        return math_out, task_begin, task_end, status

    return kernel


def _binary_csrmm_indexed_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    """Build the fused indexed CUDA MM route for mirror-prepared inputs."""
    _validate_prepared_direction(
        transpose=transpose,
        mirror_enabled=kwargs['mirror_enabled'],
    )
    _check_csr_cuda_structure_dtypes(
        kwargs['indices_info'], kwargs['indptr_info']
    )
    if weight_info.size == 1:
        weight_suffix = _tile_weight_suffix(
            weight_info.dtype, vector_info.dtype
        )
        if jnp.dtype(kwargs['indptr_info'].dtype) != jnp.dtype(jnp.int64):
            raise TypeError("TCSR CUDA tile MM currently requires int64 indptr")
        rows, cols = kwargs['shape'][::-1]
        batch = vector_info.shape[1]
        chunk_count = (rows + 4095) // 4096
        tile_mm_dir = Path(__file__).parent.joinpath("tile_mm")
        load_cuda_file(
            tile_mm_dir.joinpath("binary_csrmm_tile.cu"),
            name="tcsr_binary_csrmm_tile",
            extra_include_paths=[str(tile_mm_dir)],
            allow_cuda_graph=False,
        )
        ffi_outs = (
            jax.ShapeDtypeStruct((batch, cols), weight_info.dtype),
            jax.ShapeDtypeStruct((batch, rows), jnp.int32),
            jax.ShapeDtypeStruct(
                (batch, 2 + chunk_count), jnp.int32
            ),
        )

        def homogeneous_kernel(
            weights,
            indices,
            indptr,
            events_nb,
            task_begin,
            task_end,
            status,
            local_targets,
            tile_offsets,
            permutation,
        ):
            del indices, permutation
            spike_bn = (events_nb.T > 0).astype(jnp.int8)
            output_bn, _, _ = jax.ffi.ffi_call(
                "tcsr_binary_csrmm_tile.binary_csrmm_tile_homo_"
                f"{weight_suffix}",
                ffi_outs,
            )(
                weights,
                indptr,
                local_targets,
                tile_offsets,
                spike_bn,
            )
            return output_bn.T, task_begin, task_end, status

        return homogeneous_kernel
    spk_suffix = '_bool' if vector_info.dtype == jnp.bool_ else '_float'
    wt_sfx = dtype_suffix(weight_info.dtype)
    config = get_hybrid_config()
    module = (
        'csr_binary_indexed_csrmm_hybrid'
        + module_suffix_for_config(config)
    )
    load_cuda_file(
        Path(__file__).parent.joinpath('binary_indexed_csrmm_hybrid.cu'),
        name=module,
        extra_cuda_cflags=compile_flags_for_config(config),
        allow_cuda_graph=False,
    )
    kernel_name = (
        f'{module}.binary_indexed_csrmm_sraw_hybrid_hetero'
        f'{wt_sfx}{spk_suffix}'
    )
    output_nb_info = kwargs['outs'][0]
    ffi_outs = (
        jax.ShapeDtypeStruct(
            (vector_info.shape[1], output_nb_info.shape[0]),
            output_nb_info.dtype,
        ),
        *kwargs['outs'][1:],
    )

    def kernel(
        weights,
        indices,
        indptr,
        events_nb,
        task_begin,
        task_end,
        status,
        local_targets,
        tile_offsets,
        permutation,
    ):
        del local_targets, tile_offsets
        output_bn, task_begin, task_end, status = jax.ffi.ffi_call(
            kernel_name,
            ffi_outs,
            input_output_aliases={5: 1, 6: 2, 7: 3},
        )(
            weights,
            indices,
            indptr,
            permutation,
            events_nb,
            task_begin,
            task_end,
            status,
            task_capacity=kwargs['task_capacity'],
        )
        return output_bn.T, task_begin, task_end, status

    return kernel


def _binary_csrmm_tile_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs,
):
    mirror_enabled = kwargs['mirror_enabled']
    _validate_prepared_direction(
        transpose=transpose, mirror_enabled=mirror_enabled
    )
    if not transpose:
        return _binary_csrmm_indexed_cuda_kernel(
            weight_info,
            vector_info,
            transpose,
            **kwargs,
        )
    weight_suffix = _tile_weight_suffix(weight_info.dtype, vector_info.dtype)
    if jnp.dtype(kwargs['indptr_info'].dtype) != jnp.dtype(jnp.int64):
        raise TypeError("TCSR CUDA tile MM currently requires int64 indptr")
    nnz = kwargs['indices_info'].size
    if nnz == 0:
        raise NotImplementedError("TCSR CUDA tile MM does not support zero nnz")
    is_homogeneous = weight_info.size == 1
    if not is_homogeneous and weight_info.size != nnz:
        raise ValueError(
            "TCSR CUDA tile MM requires one homogeneous value or one value "
            "per connection"
        )

    rows, cols = kwargs["shape"]
    if vector_info.shape[1] != rows:
        raise ValueError(
            "TCSR CUDA tile MM input neuron dimension must match shape[0]"
        )
    batch = vector_info.shape[0]
    chunk_count = (rows + 4095) // 4096
    tile_mm_dir = Path(__file__).parent.joinpath("tile_mm")
    load_cuda_file(
        tile_mm_dir.joinpath("binary_csrmm_tile.cu"),
        name="tcsr_binary_csrmm_tile",
        extra_include_paths=[str(tile_mm_dir)],
        allow_cuda_graph=False,
    )
    ffi_outs = (
        jax.ShapeDtypeStruct((batch, cols), weight_info.dtype),
        jax.ShapeDtypeStruct((batch, rows), jnp.int32),
        jax.ShapeDtypeStruct((batch, 2 + chunk_count), jnp.int32),
    )
    ffi_target = (
        f"tcsr_binary_csrmm_tile.binary_csrmm_tile_homo_{weight_suffix}"
        if is_homogeneous
        else f"tcsr_binary_csrmm_tile.binary_csrmm_tile_{weight_suffix}"
    )

    def kernel(
        weights,
        indices,
        indptr,
        B,
        task_begin,
        task_end,
        status,
        tile_local_targets,
        tile_offsets,
        permutation,
    ):
        del indices
        spike_bn = (B > 0).astype(jnp.int8)
        output_bn, _, _ = jax.ffi.ffi_call(
            ffi_target,
            ffi_outs,
        )(
            weights,
            indptr,
            tile_local_targets,
            tile_offsets,
            spike_bn,
        )
        return output_bn, task_begin, task_end, status

    return kernel


def _csrmm_jvp_data(
    data_dot,
    data,
    indices,
    indptr,
    B,
    task_begin,
    task_end,
    status,
    local_targets,
    tile_offsets,
    permutation,
    *,
    shape,
    transpose,
    **kwargs,
):
    workspace = _workspace_from_task_operands(kwargs['task_capacity'], task_begin, task_end, status)
    tangent = binary_csrmm_p_call(
        data_dot,
        indices,
        indptr,
        B,
        workspace,
        local_targets,
        tile_offsets,
        shape=shape,
        transpose=transpose,
        backend=kwargs['backend'],
        backward_algorithm=kwargs['backward_algorithm'],
        permutation=permutation,
        mirror_enabled=kwargs['mirror_enabled'],
    )[0]
    return tangent, jnp.zeros_like(task_begin), jnp.zeros_like(task_end), jnp.zeros_like(status)


def _csrmm_jvp_B(
    B_dot,
    data,
    indices,
    indptr,
    B,
    task_begin,
    task_end,
    status,
    local_targets,
    tile_offsets,
    permutation,
    *,
    shape,
    transpose,
    **kwargs,
):
    physical_shape = shape if transpose else shape[::-1]
    physical_data = data if transpose or data.shape[0] == 1 else data[permutation]
    physical_B = B_dot.T if transpose else B_dot
    tangent = csrmm(
        physical_data,
        indices,
        indptr,
        physical_B,
        shape=physical_shape,
        transpose=True,
        backend=_grad_backend(kwargs['backend'], csrmm_p),
    )
    return (
        tangent.T if transpose else tangent,
        jnp.zeros_like(task_begin),
        jnp.zeros_like(task_end),
        jnp.zeros_like(status),
    )


def _csrmm_transpose_rule(
    ct,
    data,
    indices,
    indptr,
    B,
    task_begin,
    task_end,
    status,
    local_targets,
    tile_offsets,
    permutation,
    *,
    shape,
    transpose,
    **kwargs,
):
    assert not ad.is_undefined_primal(indices)
    assert not ad.is_undefined_primal(indptr)
    assert not ad.is_undefined_primal(permutation)

    ct = ct[0]
    workspace = _workspace_from_task_operands(kwargs['task_capacity'], task_begin, task_end, status)
    if ad.is_undefined_primal(B):
        physical_shape = shape if transpose else shape[::-1]
        physical_data = data if transpose or data.shape[0] == 1 else data[permutation]
        physical_ct = ct.T if transpose else ct
        dB = csrmm(
            physical_data,
            indices,
            indptr,
            physical_ct,
            shape=physical_shape,
            transpose=False,
            backend=_grad_backend(kwargs['backend'], csrmm_p),
        )
        dB = dB.T if transpose else dB
        return (
            data,
            indices,
            indptr,
            dB,
            ad.Zero(task_begin),
            ad.Zero(task_end),
            ad.Zero(status),
            ad.Zero(local_targets),
            ad.Zero(tile_offsets),
            ad.Zero(permutation),
        )
    else:
        if kwargs['backend'] == 'cuda_raw':
            sampled_gradient = (
                tcsr_sddmm_dweight_float
                if kwargs['backward_algorithm'] == 'pp_prop'
                else tcsr_sddmm_dweight_binary
            )
            slot_values = sampled_gradient(
                B if transpose else B.T,
                ct if transpose else ct.T,
                indices,
                indptr,
                local_targets,
                tile_offsets,
                transpose=True,
            )
        else:
            row_ids = jnp.repeat(
                jnp.arange(indptr.shape[0] - 1, dtype=indptr.dtype),
                jnp.diff(indptr),
                total_repeat_length=indices.size,
            )
            physical_B = B.T if transpose else B
            physical_ct = ct.T if transpose else ct
            activity = (
                physical_B
                if kwargs['backward_algorithm'] == 'pp_prop'
                else (physical_B > 0).astype(physical_ct.dtype)
            )
            slot_values = jnp.sum(
                activity[row_ids, :] * physical_ct[indices, :], axis=1
            )
        if data.aval.shape[0] == 1:
            d_data = jnp.sum(slot_values).reshape(*data.aval.shape)
        elif transpose:
            d_data = slot_values
        else:
            d_data = jnp.zeros(data.aval.shape, data.aval.dtype)
            d_data = d_data.at[permutation].add(slot_values)
        return (
            d_data,
            indices,
            indptr,
            B,
            ad.Zero(task_begin),
            ad.Zero(task_end),
            ad.Zero(status),
            ad.Zero(local_targets),
            ad.Zero(tile_offsets),
            ad.Zero(permutation),
        )


def _csrmm_batching(args, axes, **kwargs):
    axes = tuple(axes)
    if axes == (None, None, None, 0, None, None, None, None, None, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        outer_size = args[3].shape[0]
        if kwargs['transpose']:
            batch_size, neurons = args[3].shape[1:]
            B = args[3].reshape(outer_size * batch_size, neurons)
        else:
            neurons, batch_size = args[3].shape[1:]
            B = jnp.moveaxis(args[3], 0, 1).reshape(
                neurons, outer_size * batch_size
            )
        workspace = _workspace_from_task_operands(kwargs['task_capacity'], args[4], args[5], args[6])
        r = binary_csrmm_p_call(
            args[0],
            args[1],
            args[2],
            B,
            workspace,
            args[7],
            args[8],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
            backward_algorithm=kwargs['backward_algorithm'],
            permutation=args[9],
            mirror_enabled=kwargs['mirror_enabled'],
        )[0]
        if kwargs['transpose']:
            r = jnp.reshape(r, [outer_size, batch_size, r.shape[1]])
        else:
            r = jnp.moveaxis(
                jnp.reshape(r, [r.shape[0], outer_size, batch_size]),
                1,
                0,
            )
        return (r, args[4], args[5], args[6]), (0, None, None, None)

    if axes[3] is not None:
        raise NotImplementedError(
            "binary CSRMM nested batching requires the outer batch dimension on axis 0"
        )

    def prepared_call(*call_args, **call_kwargs):
        workspace = _workspace_from_task_operands(
            call_kwargs['task_capacity'],
            call_args[4],
            call_args[5],
            call_args[6],
        )
        return binary_csrmm_p_call(
            call_args[0], call_args[1], call_args[2], call_args[3],
            workspace, call_args[7], call_args[8],
            shape=call_kwargs['shape'],
            transpose=call_kwargs['transpose'],
            backend=call_kwargs['backend'],
            backward_algorithm=call_kwargs['backward_algorithm'],
            permutation=call_args[9],
            mirror_enabled=call_kwargs['mirror_enabled'],
        )

    return general_batching_rule(prepared_call, args, axes, **kwargs)


def binary_csrmm_p_call(
    weights,
    indices,
    indptr,
    B,
    workspace,
    local_targets,
    tile_offsets,
    *,
    shape: MatrixShape,
    transpose: bool,
    backend: Optional[str] = None,
    backward_algorithm: str = "bptt",
    buffers: TCSBuffers | None = None,
    permutation=None,
    mirror_enabled: bool = False,
):
    """
    Low-level primitive call for event-driven CSR matrix--matrix
    multiplication. Direction-native MM layout is BN for true and NB for
    false.

    Prepares inputs, validates shapes and dtypes, and dispatches the
    ``binary_csrmm_p`` XLA custom kernel.  The default computes
    ``C = event(B) @ A.T``; ``transpose=True`` computes
    ``C = event(B) @ A``.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values of the CSR matrix.  Shape ``(nse,)`` for
        heterogeneous weights, ``(1,)`` for a homogeneous weight, or a
        scalar (automatically promoted to shape ``(1,)``).
    indices : jax.Array
        Column indices of non-zero elements.  Shape ``(nse,)`` with dtype
        ``int32``, ``int64``, ``uint32``, or ``uint64``.
    indptr : jax.Array
        Row index pointer array.  Shape ``(shape[0] + 1,)`` and same dtype
        as ``indices``.
    B : jax.Array
        Dense event matrix. Shape ``(batch, shape[0])`` in BN layout when
        ``transpose=True`` or ``(shape[1], batch)`` in NB layout when
        ``transpose=False``. Dtype may be boolean, int8, or floating-point.
    shape : tuple of int
        Two-element tuple ``(m, k)`` giving the logical shape of the
        sparse matrix.
    transpose : bool
        If ``True``, transpose the sparse matrix before multiplication.
    backend : str or None, optional
        Compute backend to use.  Default is ``None`` (auto-select).
    backward_algorithm : {"bptt", "pp_prop"}, optional
        Weight-gradient interpretation retained as a static primitive
        parameter. ``pp_prop`` supports batch one and the fixed Batch-N
        specializations. Default is ``bptt``.
    local_targets, tile_offsets : jax.Array
        Raw tile metadata consumed by the direct CUDA implementation.
    buffers : TCSBuffers or None, optional
        Shared owner in which a lazily constructed mirror is cached.
    permutation : jax.Array or None, optional
        Prepared mirror-to-canonical slot mapping used by transformation rules.
    mirror_enabled : bool, optional
        Whether the positional structure is already mirror-prepared.

    Returns
    -------
    tuple of jax.Array
        The result matrix followed by ``task_begin``, ``task_end``, and
        ``status`` task workspace outputs.  The result matrix has shape
        ``(batch, shape[1])`` when ``transpose=True`` or
        ``(shape[0], batch)`` when ``transpose=False``.

    Raises
    ------
    AssertionError
        If ``indices`` or ``indptr`` have a dtype other than ``int32``,
        ``int64``, ``uint32``, or ``uint64``.
    AssertionError
        If ``indices`` and ``indptr`` do not share the same dtype.
    AssertionError
        If ``indptr`` or ``indices`` is not 1-D.
    AssertionError
        If ``weights`` does not have a floating-point dtype.
    AssertionError
        If there is a shape mismatch between ``B`` and the sparse
        matrix ``shape`` (considering the ``transpose`` flag).

    See Also
    --------
    binary_csrmm : High-level wrapper with unit support.

    Notes
    -----
    Scalar ``weights`` (0-d arrays) are automatically promoted to
    shape ``(1,)`` to indicate a homogeneous weight across all
    connections.

    The computation performed is:

    ``C[b, i] = sum_{j in nz(i)} w[j] * e(B[b, j])``  (non-transposed)

    ``C[b, j] = sum_{i in nz_col(j)} w[i] * e(B[b, i])``  (transposed)

    where ``e(x)`` is ``1`` when ``x`` is ``True`` (boolean) or
    ``x > 0`` (float), and ``0`` otherwise.  ``w[j]`` is either
    ``weights[j]`` (heterogeneous) or ``weights[0]`` (homogeneous).

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._tcsr.binary import binary_csrmm_p_call
        >>> from brainevent._tcsr.preprocess import build_hybrid_workspace
        >>> weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
        >>> indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        >>> B = jnp.array([[True, False], [False, True]])
        >>> workspace = build_hybrid_workspace(indptr)
        >>> local_targets = jnp.array([0, 2, 1, 2], dtype=jnp.uint16)
        >>> tile_offsets = jnp.zeros((2, 2), dtype=jnp.int32)
        >>> result = binary_csrmm_p_call(
        ...     weights, indices, indptr, B, workspace,
        ...     local_targets, tile_offsets,
        ...     shape=(2, 3), transpose=True, backend="jax")[0]
        >>> result.shape
        (2, 3)
    """
    _validate_backward_algorithm(backward_algorithm)
    if not transpose and not mirror_enabled:
        mirror = _prepare_non_transposed_components(
            indices,
            indptr,
            shape=shape,
            buffers=buffers,
        )
        indices = mirror.indices
        indptr = mirror.indptr
        permutation = mirror.permutation
        workspace = mirror.workspace
        local_targets = mirror.local_targets
        tile_offsets = mirror.tile_offsets
        mirror_enabled = True
    elif transpose and mirror_enabled:
        raise ValueError("transpose=True cannot use mirror-prepared inputs")
    assert indptr.ndim == 1, "Indptr must be 1D."
    assert indices.ndim == 1, "Indices must be 1D."
    _check_csr_structure_dtypes(indices, indptr)
    assert B.ndim == 2, "B must be 2D."
    expected_neurons = shape[0] if transpose else shape[1]
    actual_neurons = B.shape[1] if transpose else B.shape[0]
    assert actual_neurons == expected_neurons, (
        "B neuron dimension mismatch: "
        f"expected {expected_neurons}, got {actual_neurons}."
    )
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'

    # Check if weights is a scalar. If so, convert it to a one-dimensional array.
    if jnp.ndim(weights) == 0:
        weights = jnp.asarray([weights])

    out_shape = (
        (B.shape[0], shape[1])
        if transpose
        else (shape[0], B.shape[1])
    )
    out_info = jax.ShapeDtypeStruct(out_shape, weights.dtype)
    task_capacity, task_begin, task_end, status = _workspace_task_operands(workspace, indptr)
    task_begin_info = jax.ShapeDtypeStruct(task_begin.shape, task_begin.dtype)
    task_end_info = jax.ShapeDtypeStruct(task_end.shape, task_end.dtype)
    status_info = jax.ShapeDtypeStruct(status.shape, status.dtype)

    # Call the binary_csrmm_p custom operation to perform the matrix-matrix multiplication.
    physical_shape = shape if transpose else shape[::-1]
    local_targets, tile_offsets = _tile_task_operands(
        local_targets,
        tile_offsets,
        shape=physical_shape,
        nnz=indices.size,
    )
    if permutation is None:
        permutation = indptr[:0]
    if not transpose:
        assert permutation.ndim == 1, "Mirror permutation must be 1D."
        assert permutation.shape == indices.shape, (
            "Mirror permutation must have the same shape as indices."
        )
        if jnp.dtype(permutation.dtype) != jnp.dtype(indptr.dtype):
            permutation = permutation.astype(indptr.dtype)
    primitive_args = (
        weights,
        indices,
        indptr,
        B,
        task_begin,
        task_end,
        status,
        local_targets,
        tile_offsets,
        permutation,
    )

    return binary_csrmm_p(
        *primitive_args,
        outs=(out_info, task_begin_info, task_end_info, status_info),
        shape=shape,
        transpose=transpose,
        backend=backend,
        backward_algorithm=backward_algorithm,
        mirror_enabled=mirror_enabled,
        # Provide shape and data type information for indices.
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        # Provide shape and data type information for indptr.
        indptr_info=jax.ShapeDtypeStruct(indptr.shape, indptr.dtype),
        # Provide shape and data type information for weights.
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        # Provide shape and data type information for B.
        vector_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        task_begin_info=task_begin_info,
        task_end_info=task_end_info,
        status_info=status_info,
        task_capacity=task_capacity,
        tile_shape=tuple(shape),
        tile_nnz=int(indices.size),
        tile_size=_TILE_SIZE,
        permutation_info=jax.ShapeDtypeStruct(
            permutation.shape, permutation.dtype
        ),
    )


binary_csrmm_p = XLACustomKernel(
    'tcsr_binary_csrmm',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_csrmm``.

This primitive dispatches direct TCSR BN event-matrix multiplication to the
``jax``, ``numba``, ``cusparse``, and ``cuda_raw`` backends.

Only entries of ``B`` that are ``True`` (boolean) or positive (float) are considered active events
and contribute to the output, enabling efficient event-driven sparse-dense products commonly used
in spiking neural networks.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_csrmm_p.available_backends(platform)``,
and the default backend can be configured with ``binary_csrmm_p.set_default(platform, backend)``.

See Also
--------
binary_csrmm : High-level user-facing function wrapper.
"""
)
binary_csrmm_p.def_numba_kernel(_csrmm_numba_kernel, asdefault=True)
binary_csrmm_p.def_kernel('jax', 'cpu', _binary_csrmm_jax_kernel)
binary_csrmm_p.def_kernel('jax', 'gpu', _binary_csrmm_jax_kernel)
binary_csrmm_p.def_kernel('jax', 'tpu', _binary_csrmm_jax_kernel, asdefault=True)
binary_csrmm_p.def_kernel('cusparse', 'gpu', _binary_csrmm_cusparse_kernel)
binary_csrmm_p.def_cuda_raw_kernel(_binary_csrmm_tile_cuda_kernel, asdefault=True)
binary_csrmm_p.def_jvp_rule2(
    _csrmm_jvp_data,
    None,
    None,
    _csrmm_jvp_B,
    None,
    None,
    None,
    None,
    None,
    None,
)
binary_csrmm_p.def_transpose_rule(_csrmm_transpose_rule)
binary_csrmm_p.def_batching_rule(_csrmm_batching)
binary_csrmm_p.def_call(binary_csrmm_p_call)
binary_csrmm_p.def_tags('csr', 'binary')
