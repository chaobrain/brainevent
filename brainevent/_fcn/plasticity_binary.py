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

Two write-conflict-free kernels back all four FCN plasticity operations:

* **row-driven** (favorable): the spike vector indexes the ELL row axis, so the
  update streams over the spiking rows -- ``data[r, k] += trace[indices[r, k]]``.
* **column-scan** (unfavorable): the spike vector indexes the stored column ids,
  so every synapse is scanned -- ``data[r, k] += trace[r]`` when
  ``spike[indices[r, k]]`` is active.

Each ``data[r, k]`` is written by exactly one ``(r, k)`` pair, so no atomics or
scatter permutation are required.
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
    'update_fixed_post_conn_on_binary_post',
    'update_fixed_pre_conn_on_binary_pre',
    'update_fixed_pre_conn_on_binary_post',
    'fcn_plasticity_row_p',
    'fcn_plasticity_col_p',
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


def _fcn_plasticity_col_jax_kernel(spike_info: jax.ShapeDtypeStruct, **kwargs):
    is_bool = (spike_info.dtype == jnp.bool_)

    def kernel(data, indices, spike, trace):
        active = spike if is_bool else (spike != 0)
        gathered = active[indices]
        delta = jnp.where(gathered, trace[:, None], jnp.zeros_like(data))
        return [data + delta]

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


def fcn_plasticity_col_prim_call(data, indices, spike, trace, *, backend: Optional[str] = None):
    assert data.ndim == 2, 'FCN plasticity requires 2D (heterogeneous) weight data.'
    assert data.shape == indices.shape, f'data shape {data.shape} must equal indices shape {indices.shape}.'
    assert spike.ndim == 1, 'spike must be 1D.'
    assert trace.ndim == 1, 'trace must be 1D.'
    assert trace.shape[0] == data.shape[0], (
        f'row_trace length {trace.shape[0]} must equal number of ELL rows {data.shape[0]}.'
    )
    return fcn_plasticity_col_p(
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
Backs ``FixedPostNumConn.update_on_pre`` and ``FixedPreNumConn.update_on_post``.
""",
)
fcn_plasticity_row_p.def_kernel('jax_raw', 'cpu', _fcn_plasticity_row_jax_kernel)
fcn_plasticity_row_p.def_kernel('jax_raw', 'gpu', _fcn_plasticity_row_jax_kernel)
fcn_plasticity_row_p.def_kernel('jax_raw', 'tpu', _fcn_plasticity_row_jax_kernel)
fcn_plasticity_row_p.def_call(fcn_plasticity_row_prim_call)
fcn_plasticity_row_p.def_tags('fcn', 'plasticity')

fcn_plasticity_col_p = XLACustomKernel(
    'fcn_plasticity_col',
    doc="""
Unfavorable (column-scan) ELL plasticity primitive.

Scans every stored synapse: ``data[r, k] += trace[r]`` when ``spike[indices[r, k]]``
is active.  Backs ``FixedPostNumConn.update_on_post`` and
``FixedPreNumConn.update_on_pre``.
""",
)
fcn_plasticity_col_p.def_kernel('jax_raw', 'cpu', _fcn_plasticity_col_jax_kernel)
fcn_plasticity_col_p.def_kernel('jax_raw', 'gpu', _fcn_plasticity_col_jax_kernel)
fcn_plasticity_col_p.def_kernel('jax_raw', 'tpu', _fcn_plasticity_col_jax_kernel)
fcn_plasticity_col_p.def_call(fcn_plasticity_col_prim_call)
fcn_plasticity_col_p.def_tags('fcn', 'plasticity')
