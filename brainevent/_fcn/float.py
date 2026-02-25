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
from typing import Optional, Union, Tuple, Sequence

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainevent._misc import generate_block_dim, check_fixed_conn_num_shape, namescope
from brainevent._op import (
    general_batching_rule, XLACustomKernel, numba_kernel,
    BenchmarkConfig, jaxinfo_to_warpinfo
)
from brainevent.config import get_numba_parallel
from brainevent.kernix import load_cuda_file

__all__ = [
    'fcnmv',
    'fcnmv_p',
    'fcnmm',
    'fcnmm_p',
]


@namescope(static_argnames=['shape', 'transpose'])
def fcnmv(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    vector: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse matrix--vector product with fixed connection number.

    Computes ``y = W @ v`` (or ``y = W^T @ v`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``v`` is a dense floating-point vector.

    Parameters
    ----------
    weights : jax.Array or u.Quantity
        Non-zero weight values.  Shape is ``(1,)`` for homogeneous weights
        or ``(num_pre, num_conn)`` for heterogeneous weights.  Must have a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)`` specifying
        the post-synaptic (column) indices of each connection.
    vector : jax.Array or u.Quantity
        Dense vector to multiply with.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` shape of the equivalent dense
        weight matrix.
    transpose : bool
        If ``False``, compute ``W @ v`` (fixed post-synaptic connections,
        gather mode).  If ``True``, compute ``W^T @ v`` (fixed
        pre-synaptic connections, scatter mode).
    backend : str or None, optional
        Execution backend override (``'numba'``,
        ``'pallas'``, ``'tvmffi'``, or ``None`` for automatic selection).

    Returns
    -------
    jax.Array or u.Quantity
        Result vector.  Shape is ``(num_pre,)`` when ``transpose=False``
        or ``(num_post,)`` when ``transpose=True``.

    See Also
    --------
    fcnmm : Float sparse matrix--matrix product with fixed connection number.
    binary_fcnmv : Event-driven (binary) variant.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    When ``transpose=False`` (gather mode), the matrix-vector product computes:

        ``y[i] = sum_{k=0}^{n_conn-1} weights[i, k] * v[indices[i, k]]``

    For homogeneous weights (``weights`` has shape ``(1,)``):

        ``y[i] = w * sum_{k=0}^{n_conn-1} v[indices[i, k]]``

    When ``transpose=True`` (scatter mode), the computation distributes each
    row's contributions to the target columns:

        ``y[indices[i, k]] += weights[i, k] * v[i]``    for all ``i, k``

    The computational cost is ``O(num_pre * n_conn)`` regardless of the number
    of post-synaptic neurons, making this efficient for sparse connectivity.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.float import fcnmv
        >>>
        >>> weights = jnp.array([[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float32)
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> vector = jnp.array([1.0, 2.0, 3.0])
        >>> y = fcnmv(weights, indices, vector, shape=(2, 3), transpose=False)
        >>> print(y)
        [2.5 9. ]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, vector, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    vector, v_unit = u.split_mantissa_unit(vector)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = fcnmv_p_call(
        weights,
        indices,
        vector,
        transpose=transpose,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * v_unit * w_unit)


def _fcnmv_numba_kernel(
    weight_info: jax.ShapeDtypeStruct,
    transpose: bool,
    **kwargs
):
    import numba

    if transpose:
        # fixed pre connection number
        if weight_info.size == 1:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, vector, posts):
                posts[:] = 0.
                w = weights[0]
                for i in range(vector.shape[0]):
                    wv = w * vector[i]
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += wv
        else:
            @numba.njit(fastmath=True)
            def ell_mv(weights, indices, vector, posts):
                posts[:] = 0.
                for i in range(vector.shape[0]):
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += weights[i, j] * vector[i]

    else:
        # fixed post connection number
        if weight_info.size == 1:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, vector, posts):
                w = weights[0]
                for i in numba.prange(indices.shape[0]):
                    posts[i] = w * np.sum(vector[indices[i]])
        else:
            @numba.njit(parallel=get_numba_parallel(), fastmath=True, nogil=True)
            def ell_mv(weights, indices, vector, posts):
                for i in numba.prange(indices.shape[0]):
                    posts[i] = np.sum(weights[i] * vector[indices[i]])

    def kernel(weights, indices, vector):
        return numba_kernel(ell_mv, outs=kwargs['outs'])(weights, indices, vector)

    return kernel


def _fcnmv_pallas_kernel(
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    if len(shape) != 2:
        raise ValueError("shape must be a tuple of length 2")
    n_pre, n_post = shape
    n_conn = indices_info.shape[1]
    homo = weight_info.size == 1
    block_dim = generate_block_dim(indices_info.shape[1], maximum=128)

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
            if homo:
                wv = vector * weight_ref[0]
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
            placeholder = jnp.zeros(out_info.shape, out_info.dtype)
            return fn(weights, indices, vector, placeholder)

    else:
        # Sparse Matrix: [m, k]
        # vector: [k]

        def _raw_kernel(
            weight_ref,  # [1]
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
                    weight = jnp.where(mask, weight, 0.0)
                    return out + jnp.sum(weight * vec)

            i_row_sum = jax.lax.fori_loop(0, pl.cdiv(n_conn, block_dim), loop_fn, 0.)
            if homo:
                i_row_sum = i_row_sum * weight_ref[0]
            out_ref[i_row] = i_row_sum

        def kernel(weights, indices, vector):
            fn = pl.pallas_call(_raw_kernel, grid=(n_pre,), out_shape=kwargs['outs'], backend='triton')
            return fn(weights, indices, vector)

    return kernel


def _fcnmv_cuda_kernel(
    transpose: bool,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('float_fcnmv.cu'),
        name='fcn_float_mv',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    _dtype_sfx = {
        np.dtype('float16'): '_f16',
        np.dtype('float32'): '_f32',
        np.dtype('float64'): '_f64',
        np.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(np.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'

    if transpose:
        # Scatter mode: y[idx[i,k]] += w[i,k] * v[i]
        kernel_name = f'fcn_float_mv.fcnmv_scatter{mode_sfx}_auto{sfx}'

    else:
        # Gather mode: y[i] = sum_k w[i,k] * v[idx[i,k]]
        kernel_name = f'fcn_float_mv.fcnmv_gather{mode_sfx}_auto{sfx}'

    def kernel(weights, indices, vector):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, vector)

    return kernel


def _fcnmv_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    def kernel(weights, indices, vector):
        out, weights, n_pre, n_post = check_fixed_conn_num_shape(
            weights, indices, vector, shape, transpose, require_scalar_weight=True,
        )
        if transpose:
            masked_weights = jnp.broadcast_to(vector[:, None] * weights, indices.shape)
            return jax.ops.segment_sum(
                masked_weights.ravel(), indices.ravel(), num_segments=n_post
            ),

        else:
            scalar_weight = weights.ndim == 0
            if scalar_weight:
                return jax.vmap(lambda ind: weights * u.math.sum(vector[ind]))(indices),
            else:
                return jax.vmap(lambda w, ind: u.math.sum(w * vector[ind]))(weights, indices),

    return kernel


def _fcnmv_jvp_vector(spk_dot, weights, indices, spikes, *, shape, transpose, **kwargs):
    return fcnmv_p_call(weights, indices, spk_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _fcnmv_jvp_weights(w_dot, weights, indices, vector, *, shape, transpose, **kwargs):
    return fcnmv_p_call(w_dot, indices, vector, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _fcnmv_transpose_rule(ct, weights, indices, vector, *, shape, transpose, weight_info, **kwargs):
    if ad.is_undefined_primal(indices):
        raise ValueError("Cannot transpose with respect to sparse indices.")

    ct = ct[0]

    # dL/dspk = dL/dy * dy/dspk
    homo = weight_info.size == 1
    if ad.is_undefined_primal(vector):
        if type(ct) is ad.Zero:
            ct_vector = ad.Zero(vector)
        else:
            ct_vector = fcnmv_p_call(
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
            ct_weight = fcnmv_p_call(
                jnp.ones([1], dtype=weight_info.dtype),
                indices,
                vector,
                shape=shape,
                transpose=transpose,
                backend=kwargs['backend'],
            )[0]
            ct_weight = jnp.inner(ct, ct_weight).reshape(*weight_info.shape)

        else:
            if transpose:
                ct_weight = jax.vmap(lambda v, ind: v * ct[ind])(vector, indices)
            else:
                ct_weight = jax.vmap(lambda c, ind: c * vector[ind])(ct, indices)
        return ct_weight, indices, vector


def _fcnmv_batching(args, axes, **kwargs):
    if tuple(axes) == (None, None, 0):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = fcnmm_p_call(
            args[0],
            args[1],
            args[2].T,
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend']
        )
        return r, [1]
    elif tuple(axes) == (None, None, 1):
        assert args[2].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = fcnmm_p_call(
            args[0],
            args[1],
            args[2],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            backend=kwargs['backend']
        )
        return r, [1]
    else:
        return general_batching_rule(fcnmv_p, args, axes, **kwargs)


def _fcnmv_benchmark_data(*, platform):
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
            vector = jnp.asarray(np.random.randn(v_size), dtype=dtype)
            name = f"{'T' if transpose else 'NT'},{'homo' if homo else 'hetero'}"
            configs.append(
                BenchmarkConfig(
                    name,
                    (weights, indices, vector),
                    {'shape': (n_pre, n_post), 'transpose': transpose}
                )
            )
    return configs


def fcnmv_p_call(
    weights: jax.Array,
    indices: jax.Array,
    vector: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for sparse matrix--vector product with fixed
    connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba, Pallas, or TVM FFI) without performing any
    physical-unit bookkeeping.  It is typically called from :func:`fcnmv`
    or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    vector : jax.Array
        Dense vector to multiply with.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``,
        ``'tvmffi'``, or ``None``).

    Returns
    -------
    tuple[jax.Array]
        Single-element tuple containing the result vector.

    See Also
    --------
    fcnmv : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, vector, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return fcnmv_p(
        weights,
        indices,
        vector,
        transpose=transpose,
        shape=shape,
        weight_info=jax.ShapeDtypeStruct(weights.shape, weights.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        indices_info=jax.ShapeDtypeStruct(indices.shape, indices.dtype),
        outs=[out],
        backend=backend,
    )


fcnmv_p = XLACustomKernel(
    'fixed_num_mv',
    doc="""
Low-level XLA custom-kernel primitive for ``fcnmv``.

This ``XLACustomKernel`` instance dispatches the fixed-connection matrix-vector
multiplication operation with floating-point weights to registered backends
(``numba``, ``pallas``, ``tvmffi``), using runtime shape/dtype metadata
provided by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. Unlike the binary variant, this operation
uses full floating-point weights and processes all entries (not just spikes).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``fcnmv_p.available_backends(platform)``,
and the default backend can be configured with ``fcnmv_p.set_default(platform, backend)``.

See Also
--------
fcnmv : High-level user-facing function wrapper.
"""
)
def _fcnmv_warp_kernel(
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    vector_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    import warp
    from warp.jax_experimental import jax_kernel

    weight_warp_info = jaxinfo_to_warpinfo(weight_info)
    indices_warp_info = jaxinfo_to_warpinfo(indices_info)
    vector_warp_info = jaxinfo_to_warpinfo(vector_info)
    out_warp_info = jaxinfo_to_warpinfo(kwargs['outs'][0])

    if transpose:
        # Sparse Matrix: [k, m]
        # vector: [k]

        # fixed pre connection number
        if weight_info.size == 1:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                vector: vector_warp_info,
                posts: out_warp_info
            ):
                i_k = warp.tid()
                w = weights[0]
                wv = w * vector[i_k]
                for j in range(indices.shape[1]):
                    warp.atomic_add(posts, indices[i_k, j], wv)
        else:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                vector: vector_warp_info,
                posts: out_warp_info
            ):
                i = warp.tid()
                v = vector[i]
                for j in range(indices.shape[1]):
                    warp.atomic_add(posts, indices[i, j], weights[i, j] * v)

        def kernel(weights, indices, vector):
            out_info = kwargs['outs'][0]
            dim = vector_info.shape[0]
            fn = jax_kernel(ell_mv, launch_dims=[dim], num_outputs=1, in_out_argnames=['posts'])
            return fn(weights, indices, vector, jnp.zeros(out_info.shape, out_info.dtype))

    else:
        # fixed post connection number
        # Sparse Matrix: [m, k]
        # vector: [k]

        if weight_info.size == 1:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                vector: vector_warp_info,
                posts: out_warp_info
            ):
                i_m = warp.tid()
                w = weights[0]
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += vector[indices[i_m, j]]
                posts[i_m] = w * r
        else:
            @warp.kernel
            def ell_mv(
                weights: weight_warp_info,
                indices: indices_warp_info,
                vector: vector_warp_info,
                posts: out_warp_info
            ):
                i_m = warp.tid()
                r = weights.dtype(0.)
                for j in range(indices.shape[1]):
                    r += weights[i_m, j] * vector[indices[i_m, j]]
                posts[i_m] = r

        def kernel(weights, indices, vector):
            out_info = kwargs['outs'][0]
            dim = indices_info.shape[0]
            fn = jax_kernel(ell_mv, launch_dims=[dim], num_outputs=1, output_dims={'posts': out_info.shape})
            return fn(weights, indices, vector)

    return kernel


def _fcnmm_warp_kernel(
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
        # fixed pre connection number
        # Sparse Matrix: [k, m]
        # matrix: [k, n]

        if weight_info.size == 1:
            @warp.kernel
            def mm(
                weights: weight_warp_info,
                indices: indices_warp_info,
                B: B_warp_info,
                posts: out_warp_info,
            ):
                i_k, i_n = warp.tid()
                v = B[i_k, i_n] * weights[0]
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
        # fixed post connection number
        # Sparse Matrix: [m, k]
        # matrix: [k, n]

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

fcnmv_p.def_numba_kernel(_fcnmv_numba_kernel)
fcnmv_p.def_warp_kernel(_fcnmv_warp_kernel)
fcnmv_p.def_pallas_kernel('gpu', _fcnmv_pallas_kernel)
fcnmv_p.def_tvmffi_kernel('gpu', _fcnmv_cuda_kernel)
fcnmv_p.def_kernel('jax_raw', 'cpu', _fcnmv_jax_kernel)
fcnmv_p.def_kernel('jax_raw', 'gpu', _fcnmv_jax_kernel)
fcnmv_p.def_kernel('jax_raw', 'tpu', _fcnmv_jax_kernel)
fcnmv_p.def_jvp_rule2(_fcnmv_jvp_weights, None, _fcnmv_jvp_vector)
fcnmv_p.def_transpose_rule(_fcnmv_transpose_rule)
fcnmv_p.def_batching_rule(_fcnmv_batching)
fcnmv_p.def_call(fcnmv_p_call)
fcnmv_p.def_tags('fcn', 'float')
fcnmv_p.def_benchmark_data(_fcnmv_benchmark_data)


@namescope(static_argnames=['shape', 'transpose'])
def fcnmm(
    weights: Union[jax.Array, u.Quantity],
    indices: jax.Array,
    matrix: Union[jax.Array, u.Quantity],
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Union[jax.Array, u.Quantity]:
    """
    Sparse matrix--matrix product with fixed connection number.

    Computes ``Y = W @ M`` (or ``Y = W^T @ M`` when ``transpose=True``)
    where ``W`` is a sparse weight matrix stored in fixed-connection-number
    format and ``M`` is a dense floating-point matrix.

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
    fcnmv : Float sparse matrix--vector product with fixed connection number.
    binary_fcnmm : Event-driven (binary) variant.

    Notes
    -----
    The sparse weight matrix ``W`` of shape ``(num_pre, num_post)`` is stored in
    fixed-connection-number format where each row ``i`` has exactly ``n_conn``
    non-zero entries at column positions ``indices[i, :]``.

    When ``transpose=False`` (gather mode), each output element is:

        ``Y[i, j] = sum_{k=0}^{n_conn-1} weights[i, k] * M[indices[i, k], j]``

    For homogeneous weights (``weights`` has shape ``(1,)``):

        ``Y[i, j] = w * sum_{k=0}^{n_conn-1} M[indices[i, k], j]``

    When ``transpose=True`` (scatter mode), the computation distributes
    contributions to target rows:

        ``Y[indices[i, k], j] += weights[i, k] * M[i, j]``    for all ``i, k, j``

    The computational cost is ``O(num_pre * n_conn * n)`` where ``n`` is the
    number of columns in ``M``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._fcn.float import fcnmm
        >>>
        >>> weights = jnp.ones(1, dtype=jnp.float32)  # homogeneous
        >>> indices = jnp.array([[0, 1], [1, 2]])
        >>> matrix = jnp.array([[1.0, 0.5],
        ...                     [2.0, 1.0],
        ...                     [3.0, 1.5]])
        >>> y = fcnmm(weights, indices, matrix, shape=(2, 3), transpose=False)
        >>> print(y)
        [[3.  1.5]
         [5.  2.5]]
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    weights, w_unit = u.split_mantissa_unit(weights)
    matrix, m_unit = u.split_mantissa_unit(matrix)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    r = fcnmm_p_call(
        weights,
        indices,
        matrix,
        transpose=transpose,
        shape=shape,
        backend=backend,
    )[0]
    return u.maybe_decimal(r * m_unit * w_unit)


def _fcnmm_numba_kernel(
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
                for i in range(matrix.shape[0]):
                    for j in range(indices.shape[1]):
                        posts[indices[i, j]] += weights[i, j] * matrix[i]

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


def _fcnmm_pallas_kernel(
    shape: Sequence[int],
    transpose: bool,
    weight_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

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


def _fcnmm_cuda_kernel(
    transpose: bool,
    indices_info: jax.ShapeDtypeStruct,
    matrix_info: jax.ShapeDtypeStruct,
    **kwargs
):
    load_cuda_file(
        Path(__file__).parent.joinpath('float_fcnmm.cu'),
        name='fcn_float_mm',
    )

    out_info = kwargs['outs']
    n_conn = indices_info.shape[1]
    n_col = matrix_info.shape[1]
    _dtype_sfx = {
        np.dtype('float16'): '_f16',
        np.dtype('float32'): '_f32',
        np.dtype('float64'): '_f64',
        np.dtype('bfloat16'): '_bf16'
    }
    weight_info = kwargs['weight_info']
    sfx = _dtype_sfx.get(np.dtype(weight_info.dtype), '_f32')
    homo = weight_info.size == 1
    mode_sfx = '_homo' if homo else '_hetero'

    if transpose:
        # Scatter mode: Y[idx[i,k], j] += w[i,k] * M[i, j]
        kernel_name = f'fcn_float_mm.fcnmm_scatter{mode_sfx}_auto{sfx}'

    else:
        # Gather mode: Y[i, j] = sum_k w[i,k] * M[idx[i,k], j]
        # The C++ auto dispatch selects the best kernel variant
        # (vec4+shm, shared, or basic) based on n_col and n_conn.
        kernel_name = f'fcn_float_mm.fcnmm_gather{mode_sfx}_auto{sfx}'

    def kernel(weights, indices, matrix):
        return jax.ffi.ffi_call(kernel_name, out_info)(weights, indices, matrix)

    return kernel


def _fcnmm_jax_kernel(
    shape: Tuple[int, int],
    transpose: bool,
    **kwargs,
):
    def kernel(weights, indices, matrix):
        out, weights, n_pre, n_post = check_fixed_conn_num_shape(
            weights, indices, matrix, shape, transpose, require_scalar_weight=True,
        )
        if transpose:
            # Scatter mode: Y[n_post, n]
            # Y[indices[i, l], :] += weights[i, l] * matrix[i, :]
            n = matrix.shape[1]
            n_conn = indices.shape[1]
            M_exp = jnp.broadcast_to(matrix[:, None, :], (n_pre, n_conn, n))
            if weights.ndim == 0:
                vals = weights * M_exp
            else:
                vals = weights[:, :, None] * M_exp
            return jax.ops.segment_sum(vals.reshape(-1, n), indices.ravel(), num_segments=n_post),

        else:
            # Gather mode: Y[n_pre, n]
            # Y[i, :] = sum_l weights[i, l] * matrix[indices[i, l], :]
            if weights.ndim == 0:
                return jax.vmap(lambda ind: weights * jnp.sum(matrix[ind], axis=0))(indices),
            else:
                return jax.vmap(lambda w, ind: jnp.sum(w[:, None] * matrix[ind], axis=0))(weights, indices),

    return kernel


def _fcnmm_jvp_matrix(matrix_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return fcnmm_p_call(weights, indices, matrix_dot, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _fcnmm_jvp_weights(weights_dot, weights, indices, matrix, *, shape, transpose, **kwargs):
    return fcnmm_p_call(weights_dot, indices, matrix, shape=shape, transpose=transpose, backend=kwargs['backend'])


def _fcnmm_transpose_rule(ct, weights, indices, matrix, *, shape, transpose, weight_info, **kwargs):
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
            ct_weight = fcnmm_p_call(
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
    r = fcnmm_p_call(
        args[0],
        args[1],
        B,
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _fcnmm_batching(args, axes, **kwargs):
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
        return general_batching_rule(fcnmm_p, args, axes, **kwargs)


def _fcnmm_benchmark_data(*, platform):
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


def fcnmm_p_call(
    weights: jax.Array,
    indices: jax.Array,
    matrix: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool,
    backend: Optional[str] = None,
) -> Tuple[jax.Array]:
    """
    Low-level primitive call for sparse matrix--matrix product with fixed
    connection number.

    This function validates shapes and dispatches to the registered XLA
    custom kernel (Numba or Pallas) without performing any
    physical-unit bookkeeping.  It is typically called from :func:`fcnmm`
    or from autodiff rules.

    Parameters
    ----------
    weights : jax.Array
        Non-zero weight values.  Shape ``(1,)`` for homogeneous weights or
        ``(num_pre, num_conn)`` for heterogeneous weights.  Must be a
        floating-point dtype.
    indices : jax.Array
        Integer index array of shape ``(num_pre, num_conn)``.
    matrix : jax.Array
        Dense matrix to multiply with, of shape ``(k, n)``.
    shape : tuple[int, int]
        Logical ``(num_pre, num_post)`` dense-matrix shape.
    transpose : bool
        ``False`` for gather mode (fixed post-connections), ``True`` for
        scatter mode (fixed pre-connections).
    backend : str or None, optional
        Backend override (``'numba'``, ``'pallas'``, or
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
    fcnmm : High-level wrapper with unit support.
    """
    out, weights, n_pre, n_post = check_fixed_conn_num_shape(weights, indices, matrix, shape, transpose)
    assert jnp.issubdtype(weights.dtype, jnp.floating), 'Weights must be a floating-point type.'
    return fcnmm_p(
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


fcnmm_p = XLACustomKernel(
    'fixed_num_mm',
    doc="""
Low-level XLA custom-kernel primitive for ``fcnmm``.

This ``XLACustomKernel`` instance dispatches the fixed-connection matrix-matrix
multiplication operation with floating-point weights to registered backends
(``numba``, ``pallas``), using runtime shape/dtype metadata provided
by the high-level wrapper.

Fixed-connection format stores connectivity where each neuron has a fixed number
of incoming or outgoing connections. Unlike the binary variant, this operation
uses full floating-point weights and processes all entries (not just spikes).

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``fcnmm_p.available_backends(platform)``,
and the default backend can be configured with ``fcnmm_p.set_default(platform, backend)``.

See Also
--------
fcnmm : High-level user-facing function wrapper.
"""
)
fcnmm_p.def_numba_kernel(_fcnmm_numba_kernel)
fcnmm_p.def_warp_kernel(_fcnmm_warp_kernel)
fcnmm_p.def_pallas_kernel('gpu', _fcnmm_pallas_kernel)
fcnmm_p.def_tvmffi_kernel('gpu', _fcnmm_cuda_kernel)
fcnmm_p.def_kernel('jax_raw', 'cpu', _fcnmm_jax_kernel)
fcnmm_p.def_kernel('jax_raw', 'gpu', _fcnmm_jax_kernel)
fcnmm_p.def_kernel('jax_raw', 'tpu', _fcnmm_jax_kernel)
fcnmm_p.def_jvp_rule2(_fcnmm_jvp_weights, None, _fcnmm_jvp_matrix)
fcnmm_p.def_transpose_rule(_fcnmm_transpose_rule)
fcnmm_p.def_batching_rule(_fcnmm_batching)
fcnmm_p.def_call(fcnmm_p_call)
fcnmm_p.def_tags('fcn', 'float')
fcnmm_p.def_benchmark_data(_fcnmm_benchmark_data)
