# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Optional, Tuple, Union

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, check_fixed_conn_num_shape, namescope
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, jaxinfo_to_warpinfo
from brainevent._op.benchmark import BenchmarkConfig
from brainevent._typing import MatrixShape
from brainevent.config import get_numba_parallel
from brainevent.kernix import load_cuda_file
from .float import fcnmv_p_call, fcnmm_p_call

__all__ = [
    'spfloat_fcnmv',
    'spfloat_fcnmv_p',
    'spfloat_fcnmm',
    'spfloat_fcnmm_p',
]


@namescope(static_argnames=['shape', 'transpose'])
def spfloat_fcnmv(
    weights,
    indices,
    spikes,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse-float event-driven matrix--vector product with fixed connection number.

    Computes ``y = W @ s`` (or ``y = W^T @ s`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``s`` is a sparse-float vector.  Non-zero entries of ``s``
    contribute their actual floating-point value (not just ``1``) to the
    accumulation, combining the sparsity benefit of event-driven processing
    with floating-point precision.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    spikes : jax.Array or u.Quantity
        Sparse-float vector.  Entries equal to zero are skipped; non-zero
        entries are multiplied by the corresponding weight.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool, optional
        If ``False`` (default), compute ``W @ s`` (fixed post-synaptic
        connections, gather mode).  If ``True``, compute ``W^T @ s``
        (fixed pre-synaptic connections, scatter mode).
    backend : str or None, optional
        Execution backend override (``'numba'``,
        ``'pallas'``, ``'cuda_raw'``, or ``None`` for automatic selection).

    Returns
    -------
    jax.Array or u.Quantity
        Result vector.  Shape is ``(num_pre,)`` when ``transpose=False``
        or ``(num_post,)`` when ``transpose=True``.

    See Also
    --------
    spfloat_fcnmm : Sparse-float event-driven matrix--matrix product.
    binary_fcnmv : Binary event-driven variant (ignores spike values).
    fcnmv : Dense float variant (no event-driven skipping).

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    Unlike the binary variant (:func:`binary_fcnmv`) which treats non-zero
    entries as ``1``, this sparse-float variant preserves the actual
    floating-point values of the spike vector.  When ``transpose=False``
    (gather mode):

        ``y[i] = sum_{k=0}^{n_conn-1} weights[i, k] * s[indices[i, k]]``

    where only terms with ``s[indices[i, k]] != 0`` are accumulated.  For
    homogeneous weights (``weights`` has shape ``(1,)``):

        ``y[i] = w * sum_{k=0}^{n_conn-1} s[indices[i, k]]``

    When ``transpose=True`` (scatter mode):

        ``y[indices[i, k]] += weights[i, k] * s[i]``    for all ``i, k`` where ``s[i] != 0``

    This formulation is mathematically equivalent to :func:`fcnmv` but skips
    zero entries of ``s``, providing a speedup when the spike vector is sparse.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.sparse_float import spfloat_fcnmv
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)  # homogeneous
        >>> indices = jnp.array([[0, 1], [1, 2]])      # (2, 2)
        >>> spikes = jnp.array([0.0, 2.0, 3.0])
        >>> y = spfloat_fcnmv(weights, indices, spikes, shape=(2, 3))
        >>> print(y)
        [2. 5.]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, spikes, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    spikes, v_unit = u.split_mantissa_unit(spikes)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = spfloat_fcnmv_p_call(
        weights,
        indices,
        spikes,
        shape=shape,
        transpose=transpose,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * v_unit * w_unit)


def _spfloat_fcnmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        if weight_info.size == 1:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, spikes, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0.:
                        wsp = w * sp
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += wsp
        else:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, spikes, posts):
                posts[:] = 0.
                for i in range(spikes.shape[0]):
                    sp = spikes[i]
                    if sp != 0.:
                        for j in range(indices.shape[1]):
                            posts[indices[i, j]] += weights[i, j] * sp

    else:
        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, spikes, posts):
                w = weights[0]
                for i in numba.prange(indices.shape[0]):
                    r = 0.
                    for j in range(indices.shape[1]):
                        index = indices[i, j]
                        sp = spikes[index]
                        if sp != 0.:
                            r += sp
                    posts[i] = r * w
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, spikes, posts):
                for i in numba.prange(indices.shape[0]):
                    r = 0.
                    for j in range(indices.shape[1]):
                        index = indices[i, j]
                        sp = spikes[index]
                        if sp != 0.:
                            r += weights[i, j] * sp
                    posts[i] = r

    def kernel(weights, indices, spikes):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, spikes)

    return kernel


def _spfloat_fcnmv_pallas_kernel(
    transpose: int,
    shape: Tuple[int, int],
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    block_dim = generate_block_dim(indices_info.shape[1])

    if transpose:
        # Sparse Matrix: [k, m]
        # vector: [k]

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            vector_ref,  # [n_pre]
            _,
            out_ref,  # [n_post]
        ):
            i_row = pl.program_id(0)
            vector = vector_ref[i_row]

            @pl.when(vector != 0. if vector_ref.dtype != jnp.bool_ else vector)
            def run():
                if homo:
                    wv = weight_ref[0] * vector
                    homo_data = jnp.ones(block_dim, dtype=weight_info.dtype) * wv

                def loop_fn(i_col_block, _):
                    i_col = i_col_block * block_dim
                    mask = i_col + jnp.arange(block_dim) < n_conn
                    ind = index_ref[i_row, pl.dslice(i_col, block_dim)]
                    ind = jnp.where(mask, ind, 0)
                    if homo:
                        data = homo_data
                    else:
                        data = weight_ref[i_row, pl.dslice(i_col, block_dim)]
                        data = jnp.where(mask, data * vector, 0.0)
                    atomic_add(out_ref, ind, data, mask=mask)

                jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, None)

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre,),
                input_output_aliases={3: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            out_info = kwargs['outs'][0]
            return fn(weights, indices, vector, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        # Sparse Matrix: [m, k]
        # vector: [k]

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            vector_ref,  # [n_post]
            out_ref,  # [n_pre]
        ):
            i_row = pl.program_id(0)

            def loop_fn(i_col_block, out):
                i_col = i_col_block * block_dim
                mask = i_col + jnp.arange(block_dim) < n_conn
                ind = index_ref[i_row, pl.dslice(i_col, block_dim)]
                ind = jnp.where(mask, ind, 0)
                vec = vector_ref[ind]
                vec = jnp.where(mask, vec, 0.0)
                if homo:
                    return out + jnp.sum(vec)
                else:
                    weight = weight_ref[i_row, pl.dslice(i_col, block_dim)]
                    weight = jnp.where(mask, weight * vec, 0.0)
                    return out + jnp.sum(weight)

            i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
            if homo:
                i_row_sum = i_row_sum * weight_ref[0]
            out_ref[i_row] = i_row_sum

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(_raw_kernel, grid=(n_pre,), out_shape=kwargs['outs'], backend='triton')
            return fn(weights, indices, vector)

    return kernel


def _spfloat_fcnmv_cuda_kernel(
    transpose: bool,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('sparse_float_fcnmv.cu'),
        name='fcn_sparse_float_mv',
    )

    out_info = kwargs['outs']
    weight_info = kwargs['weight_info']
    _dtype_sfx = {np.dtype('float16'): '_f16', np.dtype('float32'): '_f32', np.dtype('float64'): '_f64'}
    sfx = _dtype_sfx.get(np.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'

    if transpose:
        # Scatter mode: y[idx[i,k]] += w[i,k] * s[i]  (skip when s[i] == 0)
        kernel_name = f'fcn_sparse_float_mv.spfloat_fcnmv_scatter{mode_sfx}_auto{sfx}'

    else:
        # Gather mode: y[i] = sum_k w[i,k] * s[idx[i,k]]  (skip when s == 0)
        kernel_name = f'fcn_sparse_float_mv.spfloat_fcnmv_gather{mode_sfx}_auto{sfx}'

    def kernel(weights, indices, vector):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, vector)

    return kernel


def _spfloat_fcnmv_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    n_pre, n_post = shape

    def kernel(weights, indices, vector):
        if transpose:
            # Scatter: y[indices[i,k]] += weights[i,k] * vector[i]
            masked = jnp.broadcast_to(vector[:, None] * weights, indices.shape)
            return jax.ops.segment_sum(masked.ravel(), indices.ravel(), num_segments=n_post),
        else:
            # Gather: y[i] = sum_k weights[i,k] * vector[indices[i,k]]
            if weights.ndim == 0 or weights.size == 1:
                w = weights.ravel()[0]
                return jax.vmap(lambda ind: w * jnp.sum(vector[ind]))(indices),
            else:
                return jax.vmap(lambda w, ind: jnp.sum(w * vector[ind]))(weights, indices),

    return kernel


def _spfloat_fcnmv_jvp_spikes(spk_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return fcnmv_p_call(weights, indices, spk_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _spfloat_fcnmv_jvp_weights(w_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return spfloat_fcnmv_p_call(w_dot, indices, spikes, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _spfloat_fcnmv_transpose_rule(ct, weights, indices, spikes, *, shape, transpose, weight_info, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # dL/dspk = dL/dy * dy/dspk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(spikes):
        if type(ct) is ad.Zero:
            ct_spk = ad.Zero(spikes)
        else:
            ct_spk = fcnmv_p_call(
                weights, indices, ct, shape=shape, transpose=not transpose, backend=kwargs['backend']
            )[0]
        return weights, indices, ct_spk

    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_gmax = ad.Zero(weights)
        elif homo:
            # scalar
            ct_gmax = spfloat_fcnmv_p_call(
                jnp.asarray(1., dtype=weight_info.dtype),
                indices,
                spikes,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend']
            )
            ct_gmax = jnp.inner(ct, ct_gmax[0]).reshape(*weight_info.shape)
        else:
            if transpose:
                ct_gmax = jax.vmap(lambda v, ind: v * ct[ind])(spikes, indices)
            else:
                ct_gmax = jax.vmap(lambda c, ind: c * spikes[ind])(ct, indices)
        return ct_gmax, indices, spikes


def _spfloat_fcnmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = spfloat_fcnmm_p_call(
            args[0],
            args[1],
            args[2].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = spfloat_fcnmm_p_call(
            args[0],
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(spfloat_fcnmv_p, args, axes, **kwargs)


def _spfloat_fcnmv_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indices = jnp.asarray(np.random.randint(0, n_post, (n_pre, n_conn), dtype=np.int32))
            if homo:
                weights = jnp.ones(1, dtype=dtype)
            else:
                weights = jnp.ones((n_pre, n_conn), dtype=dtype)
            v_size = n_post if not transpose else n_pre
            vector_data = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            vector_index = jnp.asarray(
                np.sort(np.random.choice(v_size, min(v_size // 5, v_size), replace=False)),
                dtype=jnp.int32,
            )
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, vector_data),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def spfloat_fcnmv_p_call(
    weights,
    indices,
    spikes,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for sparse-float event-driven matrix--vector
    product with fixed connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba, Pallas, or CUDA) without performing any
    physical-unit bookkeeping.  It is typically called from
    :func:`spfloat_fcnmv` or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    spikes : jax.Array
        Sparse-float vector; zero entries are treated as inactive.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool, optional
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).  Default is ``False``.
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, ``'cuda_raw'``, or
        ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result vector.

    See Also
    --------
    spfloat_fcnmv : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, spikes, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return spfloat_fcnmv_p(
        weights,
        indices,
        spikes,
        outs=[out],
        shape=shape,
        transpose=transpose,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        spike_info=jax.ShapeDtypeStruct(spikes.shape, spikes.dtype),
        backend=backend,
    )


spfloat_fcnmv_p = XLACustomKernel(
    'spfloat_fcnmv',
    doc="""
Low-level XLA custom-kernel primitive for ``spfloat_fcnmv``.

This ``XLACustomKernel`` instance dispatches the fixed-connection matrix-vector
multiplication operation with sparse-float inputs to registered backends
(``numba``, ``pallas``, ``cuda_raw``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. This sparse-float variant skips zero entries
in the input vector while preserving their actual floating-point values (unlike
the binary variant which treats all non-zero entries as 1).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``spfloat_fcnmv_p.available_backends(platform)``,
and the default backend can be configured with ``spfloat_fcnmv_p.set_default(platform, backend)``.

See Also
--------
spfloat_fcnmv : High-level user-facing function wrapper.
"""
)
def _spfloat_fcnmv_warp_kernel(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    spike_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    shape: Tuple[int, int],
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    spike_warp_info = jaxinfo_to_warpinfo(spike_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        if weight_info.size == 1:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                spikes: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                w = weights[0]
                sp = spikes[i]
                if sp != 0.:
                    wsp = w * sp
                    for j in range(indices.shape[1]):
                        warp.atomic_add(posts, indices[i, j], wsp)
        else:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                spikes: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                sp = spikes[i]
                if sp != 0.:
                    for j in range(indices.shape[1]):
                        warp.atomic_add(posts, indices[i, j], weights[i, j] * sp)

        def kernel(weights, indices, spikes):
            out_info = kwargs['outs'][0]
            dim = spike_info.shape[0]
            fn = jax_kernel(ell_mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, spikes, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        if weight_info.size == 1:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                spikes: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                w = weights[0]
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += spikes[indices[i, j]]
                posts[i] = r * w
        else:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                spikes: spike_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += weights[i, j] * spikes[indices[i, j]]
                posts[i] = r

        def kernel(weights, indices, spikes):
            out_info = kwargs['outs'][0]
            dim = indices_info.shape[0]
            fn = jax_kernel(ell_mv, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, spikes)

    return kernel


def _spfloat_fcnmm_warp_kernel(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    B_warp_info = jaxinfo_to_warpinfo(matrix_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        if weight_info.size == 1:
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                indices: indices_warp_info,
                B: B_warp_info,
                posts: out_warp_info,
            ):
                i_k, i_n = warp.tid()
                sp = B[i_k, i_n]
                if sp != weights.dtype(0.):
                    v = sp * weights[0]
                    for j in range(indices.shape[1]):
                        i_m = indices[i_k, j]
                        warp.atomic_add(posts, i_m, i_n, v)
        else:
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                indices: indices_warp_info,
                B: B_warp_info,
                posts: out_warp_info,
            ):
                i_k, i_n = warp.tid()
                b = B[i_k, i_n]
                if b != weights.dtype(0.):
                    for j in range(indices.shape[1]):
                        i_m = indices[i_k, j]
                        w = weights[i_k, j]
                        warp.atomic_add(posts, i_m, i_n, w * b)

        def kernel(weights, indices, matrix):
            out_info = kwargs['outs'][0]
            dim = (matrix_info.shape[0], matrix_info.shape[1])
            fn = jax_kernel(mm, launch_dims=dim, num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, matrix, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        if weight_info.size == 1:
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                indices: indices_warp_info,
                B: B_warp_info,
                posts: out_warp_info,
            ):
                i_m, i_n = warp.tid()
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    i_k = indices[i_m, j]
                    r += B[i_k, i_n]
                posts[i_m, i_n] = r * weights[0]
        else:
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                indices: indices_warp_info,
                B: B_warp_info,
                posts: out_warp_info,
            ):
                i_m, i_n = warp.tid()
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    i_k = indices[i_m, j]
                    w = weights[i_m, j]
                    r += w * B[i_k, i_n]
                posts[i_m, i_n] = r

        def kernel(weights, indices, matrix):
            out_info = kwargs['outs'][0]
            dim = (indices_info.shape[0], matrix_info.shape[1])
            fn = jax_kernel(mm, launch_dims=dim, num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, matrix)

    return kernel

spfloat_fcnmv_p.def_numba_kernel(_spfloat_fcnmv_numba_kernel)
spfloat_fcnmv_p.def_warp_kernel(_spfloat_fcnmv_warp_kernel)
spfloat_fcnmv_p.def_pallas_kernel('gpu', _spfloat_fcnmv_pallas_kernel)
spfloat_fcnmv_p.def_cuda_raw_kernel(_spfloat_fcnmv_cuda_kernel)
spfloat_fcnmv_p.def_kernel('jax_raw', 'cpu', _spfloat_fcnmv_jax_kernel)
spfloat_fcnmv_p.def_kernel('jax_raw', 'gpu', _spfloat_fcnmv_jax_kernel)
spfloat_fcnmv_p.def_kernel('jax_raw', 'tpu', _spfloat_fcnmv_jax_kernel)
spfloat_fcnmv_p.def_jvp_rule2(_spfloat_fcnmv_jvp_weights, None, _spfloat_fcnmv_jvp_spikes, None)
spfloat_fcnmv_p.def_transpose_rule(_spfloat_fcnmv_transpose_rule)
spfloat_fcnmv_p.def_batching_rule(_spfloat_fcnmv_batching)
spfloat_fcnmv_p.def_call(spfloat_fcnmv_p_call)
spfloat_fcnmv_p.def_tags('fcn', 'sparse_float')
spfloat_fcnmv_p.def_benchmark_data(_spfloat_fcnmv_benchmark_data)


@namescope(static_argnames=['shape', 'transpose'])
def spfloat_fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse-float event-driven matrix--matrix product with fixed connection number.

    Computes ``Y = W @ M`` (or ``Y = W^T @ M`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``M`` is a dense matrix whose entries may be sparse-float
    values.  Non-zero entries of ``M`` contribute their actual
    floating-point value to the accumulation.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    matrix : jax.Array or u.Quantity
        Dense matrix to multiply with, of shape ``(k, n)`` where ``k``
        matches the appropriate sparse-matrix dimension.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool
        If ``False``, compute ``W @ M`` (fixed post-synaptic connections,
        gather mode).  If ``True``, compute ``W^T @ M`` (fixed
        pre-synaptic connections, scatter mode).
    backend : str or None, optional
        Execution backend override.

    Returns
    -------
    jax.Array or u.Quantity
        Result matrix of shape ``(num_pre, n)`` when ``transpose=False``
        or ``(num_post, n)`` when ``transpose=True``.

    See Also
    --------
    spfloat_fcnmv : Sparse-float event-driven matrix--vector product.
    binary_fcnmm : Binary event-driven variant.
    fcnmm : Dense float variant.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    Unlike the binary variant (:func:`binary_fcnmm`) which treats non-zero
    matrix entries as ``1``, this sparse-float variant preserves the actual
    floating-point values.  When ``transpose=False`` (gather mode):

        ``Y[i, j] = sum_{k=0}^{n_conn-1} weights[i, k] * M[indices[i, k], j]``

    where only terms with ``M[indices[i, k], j] != 0`` are accumulated.  For
    homogeneous weights (``weights`` has shape ``(1,)``):

        ``Y[i, j] = w * sum_{k=0}^{n_conn-1} M[indices[i, k], j]``

    When ``transpose=True`` (scatter mode):

        ``Y[indices[i, k], j] += weights[i, k] * M[i, j]``    for all ``i, k, j`` where ``M[i, j] != 0``

    This formulation is mathematically equivalent to :func:`fcnmm` but skips
    zero entries of ``M``, providing a speedup when the matrix is sparse.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.sparse_float import spfloat_fcnmm
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> matrix = jnp.array([[0.0, 1.0],
        ...                     [2.0, 0.0],
        ...                     [3.0, 4.0]])
        >>> y = spfloat_fcnmm(weights, indices, matrix, shape=(2, 3), transpose=False)
        >>> print(y)
        [[2. 1.]
         [5. 4.]]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = spfloat_fcnmm_p_call(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * m_unit * w_unit)


def _spfloat_fcnmm_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # fixed pre connection number
        #
        # CSR: [k, m]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, matrix, posts):
                posts[:] = 0.
                w = weights[0]
                for i_k in range(matrix.shape[0]):
                    wv = w * matrix[i_k]
                    for i_conn in range(indices.shape[1]):
                        posts[indices[i_k, i_conn]] += wv
        else:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, matrix, posts):
                posts[:] = 0.
                for i_k in range(matrix.shape[0]):
                    for i_conn in range(indices.shape[1]):
                        posts[indices[i_k, i_conn]] += weights[i_k, i_conn] * matrix[i_k]

    else:
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, matrix, posts):
                w = weights[0]
                for i_m in numba.prange(indices.shape[0]):
                    posts[i_m] = w * np.sum(matrix[indices[i_m]], axis=0)
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, matrix, posts):
                for i_m in numba.prange(indices.shape[0]):
                    posts[i_m] = weights[i_m] @ matrix[indices[i_m]]

    def kernel(weights, indices, matrix):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, matrix)

    return kernel


def _spfloat_fcnmm_pallas_kernel(
    shape: MatrixShape,
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add

    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    n_col = matrix_info.shape[1]

    if transpose:
        #
        # fixed pre connection number
        #
        # - CSR: [k, m]
        # - matrix: [k, n]
        #

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            matrix_ref,  # [k, n]
            _,
            out_ref,  # [n_post, n]
        ):
            i_k = pl.program_id(0)
            i_n = pl.program_id(1)
            b = matrix_ref[i_k, i_n]

            def loop_fn(j, _):
                i_m = index_ref[i_k, j]
                if homo:
                    val = weight_ref[0] * b
                else:
                    val = weight_ref[i_k, j] * b
                atomic_add(out_ref, (i_m, i_n), val)

            jax.lax.fori_loop(0, n_conn, loop_fn, None)

        def kernel(weights, indices, matrix):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre, n_col),
                input_output_aliases={3: 0},
                out_shape=kwargs['outs'],
                backend='triton',
            )
            out_info = kwargs['outs'][0]
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)
            return fn(weights, indices, matrix, placeholder)

    else:

        #
        # fixed post connection number
        #
        # CSR: [m, k]
        # matrix: [k, n]
        #

        def _raw_kernel(
            weight_ref,  # [1] or [n_pre, n_conn]
            index_ref,  # [n_pre, n_conn]
            matrix_ref,  # [k, n]
            out_ref,  # [n_pre, n]
        ):
            i_m = pl.program_id(0)
            i_n = pl.program_id(1)

            def loop_fn(j, acc):
                i_k = index_ref[i_m, j]
                if homo:
                    return acc + matrix_ref[i_k, i_n]
                else:
                    return acc + weight_ref[i_m, j] * matrix_ref[i_k, i_n]

            result = jax.lax.fori_loop(0, n_conn, loop_fn, jnp.zeros((), dtype=matrix_ref.dtype))
            if homo:
                result = result * weight_ref[0]
            out_ref[i_m, i_n] = result

        def kernel(weights, indices, matrix):
            fn = pl.pallas_call(
                _raw_kernel,
                grid=(n_pre, n_col),
                out_shape=kwargs['outs'],
                backend='triton',
            )
            return fn(weights, indices, matrix)

    return kernel


def _spfloat_fcnmm_cuda_kernel(
    transpose: bool,
    indices_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('sparse_float_fcnmm.cu'),
        name='fcn_sparse_float_mm',
    )

    out_info = kwargs['outs']
    weight_info = kwargs['weight_info']
    _dtype_sfx = {np.dtype('float16'): '_f16', np.dtype('float32'): '_f32', np.dtype('float64'): '_f64'}
    sfx = _dtype_sfx.get(np.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'

    if transpose:
        # Scatter mode: Y[idx[i,k],j] += w[i,k] * M[i,j]  (skip when M[i,j] == 0)
        kernel_name = f'fcn_sparse_float_mm.spfloat_fcnmm_scatter{mode_sfx}_auto{sfx}'
    else:
        # Gather mode: Y[i,j] = sum_k w[i,k] * M[idx[i,k],j]  (skip when M[...] == 0)
        kernel_name = f'fcn_sparse_float_mm.spfloat_fcnmm_gather{mode_sfx}_auto{sfx}'

    def kernel(weights, indices, matrix):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, matrix)

    return kernel


def _spfloat_fcnmm_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    n_pre, n_post = shape

    def kernel(weights, indices, matrix):
        if transpose:
            # Scatter: Y[indices[i,k], j] += weights[i,k] * matrix[i, j]
            n = matrix.shape[1]
            n_conn = indices.shape[1]
            M_exp = jnp.broadcast_to(matrix[:, None, :], (n_pre, n_conn, n))
            if weights.ndim == 0 or weights.size == 1:
                vals = weights.ravel()[0] * M_exp
            else:
                vals = weights[:, :, None] * M_exp
            return jax.ops.segment_sum(vals.reshape(-1, n), indices.ravel(), num_segments=n_post),
        else:
            # Gather: Y[i, j] = sum_k weights[i,k] * matrix[indices[i,k], j]
            if weights.ndim == 0 or weights.size == 1:
                w = weights.ravel()[0]
                return jax.vmap(lambda ind: w * jnp.sum(matrix[ind], axis=0))(indices),
            else:
                return jax.vmap(lambda w, ind: jnp.sum(w[:, None] * matrix[ind], axis=0))(weights, indices),

    return kernel


def _spfloat_fcnmm_jvp_matrix(matrix_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return fcnmm_p_call(weights, indices, matrix_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _spfloat_fcnmm_jvp_weights(weights_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return spfloat_fcnmm_p_call(
        weights_dot, indices, matrix, shape=shape, transpose=transpose, backend=kwargs['backend'],
    )


def _spfloat_fcnmm_transpose_rule(ct, weights, indices, matrix, *, shape, transpose, weight_info, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # dL/dspk = dL/dy * dy/dspk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(matrix):
        if type(ct) is ad.Zero:
            ct_vector = ad.Zero(matrix)

        else:
            ct_vector = fcnmm_p_call(
                weights,
                indices,
                ct,
                shape=shape,
                transpose=not transpose,
                backend=kwargs['backend'],
            )[0]

        return weights, indices, ct_vector
    else:
        # dL/dw = dL/dy * dy/dw
        if type(ct) is ad.Zero:
            ct_weight = ad.Zero(weights)

        elif homo:
            ct_weight = spfloat_fcnmm_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                matrix,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )[0]
            ct_weight = jnp.sum(ct * ct_weight).reshape(*weight_info.shape)

        else:
            if transpose:
                # inputs: [k, n] @ [k, n_conn]
                # ct: [m, n]
                ct_weight = jax.vmap(lambda mat, ind: ct[ind] @ mat)(matrix, indices)
            else:
                # inputs: [m, n] @ [m, n_conn]
                # ct: [k, n]
                ct_weight = jax.vmap(lambda c, ind: (matrix[ind] @ c))(ct, indices)
        return ct_weight, indices, matrix


def _batching_base_fn(args, axis=1, **kwargs):
    assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[2].shape
    B = args[2].reshape(m, maybe_batch1 * maybe_batch2)
    r = spfloat_fcnmm_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _spfloat_fcnmm_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[2] = jnp.transpose(args[2], (1, 0, 2))
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 1):
        return _batching_base_fn(args, **kwargs)

    elif tuple(axes) == (None, None, 2):
        return _batching_base_fn(args, axis=2, **kwargs)

    else:
        return general_batching_rule(spfloat_fcnmm_p, args, axes, **kwargs)


def _spfloat_fcnmm_benchmark_data(*, platform):
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for homo in (True, False):
            n_conn = max(1, int(n_post * prob))
            indices = jnp.asarray(np.random.randint(0, n_post, (n_pre, n_conn), dtype=np.int32))
            if homo:
                weights = jnp.ones(1, dtype=dtype)
            else:
                weights = jnp.ones((n_pre, n_conn), dtype=dtype)
            b_rows = n_post if not transpose else n_pre
            B = jnp.asarray(np.random.randn(b_rows, 10), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, B),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def spfloat_fcnmm_p_call(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for sparse-float event-driven matrix--matrix
    product with fixed connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba, Pallas, or CUDA) without performing any
    physical-unit bookkeeping.  It is typically called from
    :func:`spfloat_fcnmm` or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    matrix : jax.Array or u.Quantity
        Dense matrix to multiply with, of shape ``(k, n)``.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, ``'cuda_raw'``, or
        ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result matrix.

    Notes
    -----
    The ``transpose=True`` path uses scatter-based accumulation via
    ``atomic_add`` on GPU backends, while ``transpose=False`` uses a
    gather-based reduction.

    See Also
    --------
    spfloat_fcnmm : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return spfloat_fcnmm_p(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        matrix_info=jax.ShapeDtypeStruct(matrix.shape, matrix.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=[out],
        backend=backend,
    )


spfloat_fcnmm_p = XLACustomKernel(
    'spfloat_fcnmm',
    doc="""
Low-level XLA custom-kernel primitive for ``spfloat_fcnmm``.

This ``XLACustomKernel`` instance dispatches the fixed-connection matrix-matrix
multiplication operation with sparse-float inputs to registered backends
(``numba``, ``pallas``, ``cuda_raw``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. This sparse-float variant skips zero entries
in the input matrix while preserving their actual floating-point values (unlike
the binary variant which treats all non-zero entries as 1).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``spfloat_fcnmm_p.available_backends(platform)``,
and the default backend can be configured with ``spfloat_fcnmm_p.set_default(platform, backend)``.

See Also
--------
spfloat_fcnmm : High-level user-facing function wrapper.
"""
)
spfloat_fcnmm_p.def_numba_kernel(_spfloat_fcnmm_numba_kernel)
spfloat_fcnmm_p.def_warp_kernel(_spfloat_fcnmm_warp_kernel)
spfloat_fcnmm_p.def_pallas_kernel('gpu', _spfloat_fcnmm_pallas_kernel)
spfloat_fcnmm_p.def_cuda_raw_kernel(_spfloat_fcnmm_cuda_kernel)
spfloat_fcnmm_p.def_kernel('jax_raw', 'cpu', _spfloat_fcnmm_jax_kernel)
spfloat_fcnmm_p.def_kernel('jax_raw', 'gpu', _spfloat_fcnmm_jax_kernel)
spfloat_fcnmm_p.def_kernel('jax_raw', 'tpu', _spfloat_fcnmm_jax_kernel)
spfloat_fcnmm_p.def_jvp_rule2(_spfloat_fcnmm_jvp_weights, None, _spfloat_fcnmm_jvp_matrix, None)
spfloat_fcnmm_p.def_transpose_rule(_spfloat_fcnmm_transpose_rule)
spfloat_fcnmm_p.def_batching_rule(_spfloat_fcnmm_batching)
spfloat_fcnmm_p.def_call(spfloat_fcnmm_p_call)
spfloat_fcnmm_p.def_tags('fcn', 'sparse_float')
spfloat_fcnmm_p.def_benchmark_data(_spfloat_fcnmm_benchmark_data)
