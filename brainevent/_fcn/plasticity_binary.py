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

"""Native ELL event-driven STDP weight updates for fixed-connection matrices.

This module provides the **favorable** (row-driven) plasticity kernel: the spike
vector indexes the ELL row axis, so the update streams over the spiking rows --
``data[r, k] += trace[indices[r, k]]``.  Each ``data[r, k]`` is written by exactly
one ``(r, k)`` pair, so no atomics or scatter permutation are required.

The **unfavorable** direction (spikes index the stored column ids) is served by
the reused perm-fused CSR primitive
:func:`brainevent.update_csr_on_binary_post`, which touches only the synapses of
spiking neurons (event-driven) rather than scanning every stored synapse.  See
:meth:`brainevent.FixedNumPerPre.update_on_post` and
:meth:`brainevent.FixedNumPerPost.update_on_pre`.
"""

from pathlib import Path
from typing import Optional

import brainunit as u
import jax
import jax.numpy as jnp

from brainevent._misc import namescope
from brainevent._op import XLACustomKernel, numba_kernel, load_cuda_file
from brainevent._typing import MatrixShape
from brainevent.config import get_numba_parallel

__all__ = [
    'update_fixed_post_conn_on_binary_pre',
    'update_fixed_pre_conn_on_binary_post',
    'fcn_plasticity_row_p',
]

_HOMO_MSG = (
    "Plasticity updates require per-synapse (heterogeneous) weights, but received "
    "a homogeneous (size-1) weight. Materialize per-synapse weights first "
    "(e.g. broadcast to the connectivity shape) before applying a plasticity update."
)


def _check_heterogeneous(data, indices):
    if data.size == 1 and indices.size > 1:
        raise ValueError(_HOMO_MSG)


# --------------------------------------------------------------------------- #
# jax_raw kernels
# --------------------------------------------------------------------------- #

def _fcn_plasticity_row_jax_kernel(spike_info: jax.ShapeDtypeStruct, **kwargs):
    is_bool = (spike_info.dtype == jnp.bool_)

    def kernel(data, indices, spike, trace):
        active = spike if is_bool else (spike != 0)
        delta = jnp.where(active[:, None], trace[indices], jnp.zeros_like(data))
        return [data + delta]

    return kernel


# --------------------------------------------------------------------------- #
# numba kernels (CPU, in-place via input_output_aliases={0: 0})
# --------------------------------------------------------------------------- #

def _fcn_plasticity_row_numba_kernel(spike_info: jax.ShapeDtypeStruct, **kwargs):
    import numba
    parallel = get_numba_parallel()

    if spike_info.dtype == jnp.bool_:
        @numba.njit(parallel=parallel, fastmath=True, nogil=True)
        def kern(data, indices, spike, trace, out):
            for r in numba.prange(data.shape[0]):
                if spike[r]:
                    for k in range(data.shape[1]):
                        out[r, k] += trace[indices[r, k]]
    else:
        @numba.njit(parallel=parallel, fastmath=True, nogil=True)
        def kern(data, indices, spike, trace, out):
            for r in numba.prange(data.shape[0]):
                if spike[r] != 0.:
                    for k in range(data.shape[1]):
                        out[r, k] += trace[indices[r, k]]

    def kernel(data, indices, spike, trace):
        return numba_kernel(kern, outs=kwargs['outs'], input_output_aliases={0: 0})(
            data, indices, spike, trace
        )

    return kernel


# --------------------------------------------------------------------------- #
# cuda_raw kernel (row-driven / favorable direction only)
# --------------------------------------------------------------------------- #

def _fcn_plasticity_row_cuda_kernel(
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs,
):
    """CUDA Raw kernel for the favorable (row-driven) ELL plasticity update.

    Dispatches to ``update_fcn_row{wt}{spk}`` compiled from
    ``plasticity_row_driven.cu``; the entry point auto-selects a thread-,
    warp-, or block-per-row sub-kernel based on ``num_conn``.  Only int32
    indices are supported -- callers with int64 indices should select
    ``backend='jax_raw'``.
    """
    if indices_info.dtype == jnp.int64:
        raise TypeError(
            "fcn row-driven plasticity 'cuda_raw' backend only supports int32 "
            "indices. Use backend='jax_raw' for int64 indices."
        )
    load_cuda_file(
        Path(__file__).parent.joinpath('plasticity_row_driven.cu'),
        name='fcn_plasticity_row_driven',
    )
    out_info = kwargs['outs']
    spk_suffix = '_bool' if spike_info.dtype == jnp.bool_ else '_float'
    _dtype_sfx = {
        jnp.dtype('float16'): '_f16',
        jnp.dtype('float32'): '_f32',
        jnp.dtype('float64'): '_f64',
        jnp.dtype('bfloat16'): '_bf16',
    }
    wt_sfx = _dtype_sfx.get(jnp.dtype(weight_info.dtype), '_f32')
    kernel_name = f'fcn_plasticity_row_driven.update_fcn_row{wt_sfx}{spk_suffix}'

    def kernel(data, indices, spike, trace):
        return jax.ffi.ffi_call(kernel_name, out_info, input_output_aliases={0: 0})(
            data, indices, spike, trace
        )

    return kernel


# --------------------------------------------------------------------------- #
# Primitive calls
# --------------------------------------------------------------------------- #

def fcn_plasticity_row_prim_call(data, indices, spike, trace, *, backend: Optional[str] = None):
    assert data.ndim == 2, 'FCN plasticity requires 2D (heterogeneous) weight data.'
    assert data.shape == indices.shape, f'data shape {data.shape} must equal indices shape {indices.shape}.'
    assert spike.ndim == 1, 'spike must be 1D.'
    assert trace.ndim == 1, 'trace must be 1D.'
    assert spike.shape[0] == data.shape[0], (
        f'row_spike length {spike.shape[0]} must equal number of ELL rows {data.shape[0]}.'
    )
    return fcn_plasticity_row_p(
        data, indices, spike, trace,
        outs=[jax.ShapeDtypeStruct(data.shape, data.dtype)],
        weight_info=jax.ShapeDtypeStruct(data.shape, data.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        spike_info=jax.ShapeDtypeStruct(spike.shape, spike.dtype),
        trace_info=jax.ShapeDtypeStruct(trace.shape, trace.dtype),
        backend=backend,
    )


# --------------------------------------------------------------------------- #
# Primitive registration
# --------------------------------------------------------------------------- #

fcn_plasticity_row_p = XLACustomKernel(
    'fcn_plasticity_row',
    doc="""
Favorable (row-driven) ELL plasticity primitive.

For each row ``r`` whose spike is active, ``data[r, k] += trace[indices[r, k]]``.
Backs ``FixedNumPerPre.update_on_pre`` and ``FixedNumPerPost.update_on_post``.
""",
)
fcn_plasticity_row_p.def_numba_kernel(_fcn_plasticity_row_numba_kernel)
fcn_plasticity_row_p.def_cuda_raw_kernel(_fcn_plasticity_row_cuda_kernel, asdefault=True)
fcn_plasticity_row_p.def_kernel('jax_raw', 'cpu', _fcn_plasticity_row_jax_kernel)
fcn_plasticity_row_p.def_kernel('jax_raw', 'gpu', _fcn_plasticity_row_jax_kernel)
fcn_plasticity_row_p.def_kernel('jax_raw', 'tpu', _fcn_plasticity_row_jax_kernel)
fcn_plasticity_row_p.def_call(fcn_plasticity_row_prim_call)
fcn_plasticity_row_p.def_tags('fcn', 'plasticity')


# --------------------------------------------------------------------------- #
# Module functions (units, homogeneous guard, clip)
# --------------------------------------------------------------------------- #

def _apply_row(data, indices, spike, trace, w_min, w_max, backend):
    data, wunit = u.split_mantissa_unit(data)
    trace = u.Quantity(trace).to(wunit).mantissa
    _check_heterogeneous(data, indices)
    new = fcn_plasticity_row_prim_call(data, indices, spike, trace, backend=backend)[0]
    new = u.maybe_decimal(new * wunit)
    return u.math.clip(new, w_min, w_max)


@namescope(static_argnames=['shape'])
def update_fixed_post_conn_on_binary_pre(
    data, indices, pre_spike, post_trace,
    w_min=None, w_max=None, *, shape: MatrixShape, backend: Optional[str] = None,
):
    """Pre-spike STDP update for a ``FixedNumPerPre`` (favorable, row-driven).

    For each firing pre neuron ``i`` and every stored synapse ``(i, j)``:
    ``W[i, j] <- clip(W[i, j] + post_trace[j], w_min, w_max)``.

    Parameters
    ----------
    data : jax.Array or Quantity
        Heterogeneous ELL weights, shape ``(num_pre, num_conn)``.
    indices : jax.Array
        Post-synaptic ids, shape ``(num_pre, num_conn)``.
    pre_spike : jax.Array
        Pre-synaptic spikes (bool or float), shape ``(num_pre,)``.
    post_trace : jax.Array or Quantity
        Post-synaptic trace, shape ``(num_post,)``.
    w_min, w_max : jax.Array, Quantity, number, or None, optional
        Clip bounds (``None`` disables the corresponding bound).
    shape : tuple of int
        Logical ``(num_pre, num_post)``.
    backend : str or None, optional
        Backend override.

    Returns
    -------
    jax.Array or Quantity
        Updated weights, shape ``(num_pre, num_conn)``.

    Raises
    ------
    ValueError
        If ``data`` is homogeneous (size-1) while the connectivity stores more
        than one synapse.

    See Also
    --------
    brainevent.FixedNumPerPre.update_on_post : Post-spike (unfavorable) counterpart,
        served by the perm-fused CSR plasticity primitive.

    Examples
    --------
    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.plasticity_binary import update_fixed_post_conn_on_binary_pre
        >>> data = jnp.array([[0.5, 0.3], [0.8, 0.2]], dtype=jnp.float32)
        >>> indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
        >>> pre_spike = jnp.array([True, False])
        >>> post_trace = jnp.array([0.1, 0.2, 0.05], dtype=jnp.float32)
        >>> update_fixed_post_conn_on_binary_pre(
        ...     data, indices, pre_spike, post_trace, shape=(2, 3))
    """
    assert indices.shape[0] == shape[0], 'indices rows must equal num_pre.'
    assert pre_spike.shape[0] == shape[0], 'pre_spike length must equal num_pre.'
    assert post_trace.shape[0] == shape[1], 'post_trace length must equal num_post.'
    return _apply_row(data, indices, pre_spike, post_trace, w_min, w_max, backend)


@namescope(static_argnames=['shape'])
def update_fixed_pre_conn_on_binary_post(
    data, indices, pre_trace, post_spike,
    w_min=None, w_max=None, *, shape: MatrixShape, backend: Optional[str] = None,
):
    """Post-spike STDP update for a ``FixedNumPerPost`` (favorable, row-driven).

    For each firing post neuron ``j`` and every stored synapse ``(i, j)``:
    ``W[i, j] <- clip(W[i, j] + pre_trace[i], w_min, w_max)``.

    Parameters
    ----------
    data : jax.Array or Quantity
        Heterogeneous ELL weights, shape ``(num_post, num_conn)``.
    indices : jax.Array
        Pre-synaptic ids, shape ``(num_post, num_conn)``.
    pre_trace : jax.Array or Quantity
        Pre-synaptic trace, shape ``(num_pre,)``.
    post_spike : jax.Array
        Post-synaptic spikes (bool or float), shape ``(num_post,)``.
    w_min, w_max : jax.Array, Quantity, number, or None, optional
        Clip bounds (``None`` disables the corresponding bound).
    shape : tuple of int
        Logical ``(num_pre, num_post)``.
    backend : str or None, optional
        Backend override.

    Returns
    -------
    jax.Array or Quantity
        Updated weights, shape ``(num_post, num_conn)``.

    Raises
    ------
    ValueError
        If ``data`` is homogeneous (size-1) while the connectivity stores more
        than one synapse.

    See Also
    --------
    brainevent.FixedNumPerPost.update_on_pre : Pre-spike (unfavorable) counterpart,
        served by the perm-fused CSR plasticity primitive.
    """
    assert indices.shape[0] == shape[1], 'indices rows must equal num_post.'
    assert pre_trace.shape[0] == shape[0], 'pre_trace length must equal num_pre.'
    assert post_spike.shape[0] == shape[1], 'post_spike length must equal num_post.'
    return _apply_row(data, indices, post_spike, pre_trace, w_min, w_max, backend)
