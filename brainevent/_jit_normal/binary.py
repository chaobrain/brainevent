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

from typing import Optional, Sequence

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from brainevent._data import _initialize_seed, _initialize_conn_length
from brainevent._misc import generate_block_dim, namescope
from brainevent._numba_random import get_numba_lfsr_seed, get_numba_lfsr_random_integers, get_numba_lfsr_normal
from brainevent._op import XLACustomKernel, numba_kernel, general_batching_rule, BenchmarkConfig
from brainevent._pallas_random import get_pallas_lfsr_rng_class
from brainevent._typing import Data, MatrixShape
from .float import jitnmv_p_call, jitnmm_p_call

__all__ = [
    "binary_jitnmv",
    "binary_jitnmv_p",
    "binary_jitnmm",
    "binary_jitnmm_p",
]


@namescope(static_argnames=("shape", "transpose", "corder"))
def binary_jitnmv(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    vector: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    """
    Event-driven matrix-vector multiplication with a JIT normal-distributed connectivity matrix.

    Computes ``M @ v`` where ``M`` is a sparse matrix whose non-zero entries are
    drawn from a normal distribution with parameters ``w_loc`` (mean) and ``w_scale``
    (standard deviation), and ``v`` is a binary event vector. Only positions where
    ``v`` is active (True or > 0) contribute to the output, making this operation
    event-driven and efficient for sparse neural activity patterns.

    Parameters
    ----------
    w_loc : Data
        Location (mean) parameter of the normal distribution for the matrix
        weights. Scalar or 1-element array, optionally with physical units.
    w_scale : Data
        Scale (standard deviation) parameter of the normal distribution for the
        matrix weights. Must have the same physical dimension as ``w_loc``.
    prob : float
        Connection probability in the range ``[0, 1]``. Controls the sparsity of
        the generated connectivity matrix.
    vector : Data
        The binary event vector to multiply with. Elements are treated as events
        (True/False or >0/<=0). Shape must be compatible with ``shape``.
    seed : int, optional
        Random seed for reproducible matrix generation. If None, a random seed
        is generated at compile time.
    shape : MatrixShape
        Shape of the implicit connectivity matrix as ``(rows, cols)``.
    transpose : bool, optional
        If True, compute ``M.T @ v`` instead of ``M @ v``. Default is False.
    corder : bool, optional
        Memory layout order for kernel dispatch. True for C-order (row-major),
        False for Fortran-order (column-major). Default is True.
    backend : str, optional
        Compute backend to use (``'numba'``, ``'pallas'``, or None
        for automatic selection).

    Returns
    -------
    Data
        The result vector of the matrix-vector product. If the inputs carry
        physical units, the output will have units equal to the product of the
        weight units and the vector units.

    Raises
    ------
    brainunit.DimensionMismatchError
        If ``w_loc`` and ``w_scale`` do not have the same physical dimension.

    See Also
    --------
    binary_jitnmm : Event-driven matrix-matrix multiplication variant.
    jitnmv : Float (non-event) matrix-vector multiplication with normal weights.

    Notes
    -----
    The connectivity matrix ``W`` is never materialized in memory. Instead, the
    pseudo-random structure is regenerated on-the-fly using the ``seed`` and
    ``prob`` parameters, following the same PRNG sequence as ``jitn`` to ensure
    consistency with ``todense()``.

    The implicit weight matrix has entries:

    ``W[i, j] = Normal(w_loc, w_scale) * Bernoulli(prob)``

    where ``Normal(w_loc, w_scale)`` is an independent draw for each non-zero
    position, and ``Bernoulli(prob)`` is 1 with probability ``prob`` and 0
    otherwise. The connection mask is determined by a deterministic hash of
    ``(seed, i, j)``.

    The event-driven matrix-vector product computes:

    ``y[i] = sum_{j in C(i)} N_ij * spike[j]``

    where ``C(i) = {j : Bernoulli_ij = 1}`` is the set of connected
    pre-synaptic indices for post-synaptic neuron ``i``, ``N_ij ~ Normal(w_loc,
    w_scale)`` is the connection weight, and ``spike[j]`` is treated as a
    binary event (True/False or >0/<=0). Equivalently:

    ``y[i] = sum_{j in C(i) : spike[j]=1} N_ij``

    The connection length parameter ``clen = 2 / prob`` controls the average
    stride between non-zero entries.

    This operation supports automatic differentiation (JVP and transpose rules)
    for ``w_loc``, ``w_scale``, and ``vector``. Batching over the vector
    dimension is promoted to ``binary_jitnmm``.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_normal.binary import binary_jitnmv
        >>> w_loc = jnp.array([1.0])
        >>> w_scale = jnp.array([0.1])
        >>> events = jnp.array([True, False, True, True, False])
        >>> result = binary_jitnmv(w_loc, w_scale, 0.5, events, seed=42,
        ...                        shape=(3, 5))
    """
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    seed = _initialize_seed(seed)
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    vector, unitv = u.split_mantissa_unit(vector)
    clen = _initialize_conn_length(prob)
    res = binary_jitnmv_p_call(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitv)


def binary_jitnmm(
    w_loc: Data,
    w_scale: Data,
    prob: float,
    B: Data,
    seed: Optional[int] = None,
    *,
    shape: MatrixShape,
    transpose: bool = False,
    corder: bool = True,
    backend: Optional[str] = None,
) -> Data:
    """
    Event-driven matrix-matrix multiplication with a JIT normal-distributed connectivity matrix.

    Computes ``M @ B`` where ``M`` is a sparse matrix whose non-zero entries are
    drawn from a normal distribution with parameters ``w_loc`` (mean) and ``w_scale``
    (standard deviation), and ``B`` is a 2-D binary event matrix. Only positions
    where ``B`` elements are active (True or > 0) contribute to the output.

    Parameters
    ----------
    w_loc : Data
        Location (mean) parameter of the normal distribution for the matrix
        weights. Scalar or 1-element array, optionally with physical units.
    w_scale : Data
        Scale (standard deviation) parameter of the normal distribution for the
        matrix weights. Must have the same physical dimension as ``w_loc``.
    prob : float
        Connection probability in the range ``[0, 1]``. Controls the sparsity of
        the generated connectivity matrix.
    B : Data
        The binary event matrix to multiply with, shape ``(k, n)``. Elements are
        treated as events (True/False or >0/<=0).
    seed : int, optional
        Random seed for reproducible matrix generation. If None, a random seed
        is generated at compile time.
    shape : MatrixShape
        Shape of the implicit connectivity matrix as ``(rows, cols)``.
    transpose : bool, optional
        If True, compute ``M.T @ B`` instead of ``M @ B``. Default is False.
    corder : bool, optional
        Memory layout order for kernel dispatch. True for C-order (row-major),
        False for Fortran-order (column-major). Default is True.
    backend : str, optional
        Compute backend to use (``'numba'``, ``'pallas'``, or None
        for automatic selection).

    Returns
    -------
    Data
        The result matrix of the matrix-matrix product with shape
        ``(shape[0], B.shape[1])`` (or ``(shape[1], B.shape[1])`` if transposed).
        If the inputs carry physical units, the output will have units equal to
        the product of the weight units and the ``B`` units.

    Raises
    ------
    brainunit.DimensionMismatchError
        If ``w_loc`` and ``w_scale`` do not have the same physical dimension.

    See Also
    --------
    binary_jitnmv : Event-driven matrix-vector multiplication variant.
    jitnmm : Float (non-event) matrix-matrix multiplication with normal weights.

    Notes
    -----
    The connectivity matrix ``W`` is never materialized in memory. The
    pseudo-random structure is regenerated on-the-fly using the ``seed`` and
    ``prob`` parameters, matching the PRNG sequence used by ``jitn``.

    The implicit weight matrix has entries:

    ``W[i, j] = Normal(w_loc, w_scale) * Bernoulli(prob)``

    where ``Normal(w_loc, w_scale)`` is an independent draw for each non-zero
    position, and ``Bernoulli(prob)`` is 1 with probability ``prob`` and 0
    otherwise.

    The event-driven matrix-matrix product computes:

    ``Y[i, k] = sum_{j in C(i)} N_ij * spike[j, k]``

    where ``C(i) = {j : Bernoulli_ij = 1}`` is the set of connected
    pre-synaptic indices for post-synaptic neuron ``i``, ``N_ij ~ Normal(w_loc,
    w_scale)`` is the connection weight, and ``spike[j, k]`` is treated as a
    binary event. Each column ``k`` of ``B`` is processed independently.

    The connection length parameter ``clen = 2 / prob`` controls the average
    stride between non-zero entries.

    This operation supports automatic differentiation (JVP and transpose rules)
    for ``w_loc``, ``w_scale``, and ``B``. Batching over the ``B`` dimension
    is supported along axes 0, 1, and 2.

    Examples
    --------

    .. code-block:: python

        >>> import jax.numpy as jnp
        >>> from brainevent._jit_normal.binary import binary_jitnmm
        >>> w_loc = jnp.array([1.0])
        >>> w_scale = jnp.array([0.1])
        >>> B = jnp.array([[True, False], [False, True], [True, True],
        ...                [False, False], [True, False]])
        >>> result = binary_jitnmm(w_loc, w_scale, 0.5, B, seed=42,
        ...                        shape=(3, 5))
    """
    u.fail_for_dimension_mismatch(w_loc, w_scale, "w_loc and w_scale must have the same dimension.")
    seed = _initialize_seed(seed)
    w_loc, unitd = u.split_mantissa_unit(w_loc)
    w_scale = u.Quantity(w_scale).to(unitd).mantissa
    B, unitB = u.split_mantissa_unit(B)
    clen = _initialize_conn_length(prob)
    res = binary_jitnmm_p_call(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )[0]
    return u.maybe_decimal(res * unitd * unitB)


def _jitc_mv_normal_numba_kernel_generator(
    corder: bool,
    vector_info: jax.ShapeDtypeStruct,
    **kwargs
):
    r"""Generate the CPU kernel for the :func:`_jitc_matvec_normal` operation.
    """
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_normal = get_numba_lfsr_normal()

    if corder:
        # This means that the for loop is parallelized along the dimension of the output vector: ``post.shape[0]``.
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, vector, seed, posts):
                posts[:] = 0.
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_col in range(n_col):
                    state = _lfsr_seed(seed0 + i_col * n_row)
                    i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.asarray(0., dtype=posts.dtype)
                    while i_row < n_row:
                        w = _lfsr_normal(state, w_loc0, w_scale0)
                        if vector[i_row]:
                            out += w
                        i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_col] = out
        else:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, vector, seed, posts):
                posts[:] = 0.
                n_col = posts.shape[0]
                n_row = vector.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_col in range(n_col):
                    state = _lfsr_seed(seed0 + i_col * n_row)
                    i_row = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.asarray(0., dtype=posts.dtype)
                    while i_row < n_row:
                        w = _lfsr_normal(state, w_loc0, w_scale0)
                        if vector[i_row] > 0.:
                            out += w
                        i_row += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_col] = out

    else:
        if vector_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_row in range(num_row):
                    if vector[i_row]:
                        state = _lfsr_seed(seed0 + i_row * num_col)
                        i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while i_col < num_col:
                            w = _lfsr_normal(state, w_loc0, w_scale0)
                            posts[i_col] += w
                            i_col += _lfsr_random_integers(state, 1, clen0 - 1)
        else:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, vector, seed, posts):
                posts[:] = 0.
                num_col = posts.shape[0]
                num_row = vector.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                clen0 = clen[0]
                seed0 = seed[0]
                for i_row in range(num_row):
                    if vector[i_row] > 0.:
                        state = _lfsr_seed(seed0 + i_row * num_col)
                        i_col = _lfsr_random_integers(state, 0, clen0 - 1)
                        while i_col < num_col:
                            w = _lfsr_normal(state, w_loc0, w_scale0)
                            posts[i_col] += w
                            i_col += _lfsr_random_integers(state, 1, clen0 - 1)

    def run(w_loc, w_scale, clen, vector, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, vector, seed)

    return run


def _jitc_mv_normal_pallas_kernel_generator(
    vector_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Pallas GPU kernel for binary event matvec with normal-distributed JITC matrix.

    JITC matrix generation must be consistent with _jitnmv_pallas_kernel_generator
    in float.py (same RNG seeding and iteration order):
    - corder=True:  vectorize over output rows (i_cols), loop over input (i_rows)
    - corder=False: vectorize over input rows (i_rows), loop over output (i_cols)
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    dim = (out_info.shape[0] if corder else vector_info.shape[0])
    block_size = generate_block_dim(dim, maximum=128)

    if corder:
        # Matches float.py _jitnmv_pallas corder=True exactly:
        # vectorize over output, seed by i_cols, loop over i_rows.
        # Binary: accumulate w only when vector[i_row] is event (>0 or True).
        def kernel(w_loc_ref, w_scale_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_row = vector_ref.shape[0]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_col_block = pl.program_id(0)
            i_cols = i_col_block * block_size + jnp.arange(block_size)
            i_col_mask = i_cols < dim

            def body(data):
                i_rows, i_row_mask, rng, out = data
                v = vector_ref[i_rows]
                if vector_ref.dtype != jnp.bool_:
                    v = v > 0.
                w = rng.normal(w_loc, w_scale)
                out = jnp.where(i_row_mask & v, out + w, out)
                i_rows += rng.random_integers(1, clen)
                return i_rows, i_rows < num_row, rng, out

            rng = _PallasLFSRRNG(seed + i_cols * num_row)
            i_rows = rng.random_integers(0, clen)
            i_row_mask = i_rows < num_row
            out = jnp.zeros(block_size, dtype=post_ref.dtype)
            out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng, out)
            )[-1]
            post_ref[i_cols] = jnp.where(i_col_mask, out, post_ref[i_cols])

    else:
        # Matches float.py _jitnmv_pallas corder=False exactly:
        # vectorize over input, seed by i_rows, loop over i_cols.
        # Binary: only scatter w (via atomic_add) when vector element is event.
        def kernel(w_loc_ref, w_scale_ref, clen_ref, vector_ref, seed_ref, _, post_ref):
            num_col = post_ref.shape[0]
            w_loc = w_loc_ref[0]
            w_scale = w_scale_ref[0]
            clen = clen_ref[0]
            seed = seed_ref[0]
            i_row_block = pl.program_id(0)
            i_rows = i_row_block * block_size + jnp.arange(block_size)
            i_row_mask = i_rows < dim
            v = vector_ref[i_rows]
            if vector_ref.dtype != jnp.bool_:
                v = v > 0.
            # event_mask: only active lanes where the vector element is an event
            event_mask = i_row_mask & v

            def body(data):
                i_cols, i_col_mask, rng = data
                w = rng.normal(w_loc, w_scale)
                atomic_add(post_ref, (i_cols,), w, mask=event_mask & i_col_mask)
                i_cols += rng.random_integers(1, clen)
                return i_cols, i_cols < num_col, rng

            rng = _PallasLFSRRNG(seed + i_rows * num_col)
            i_cols = rng.random_integers(0, clen)
            i_col_mask = i_cols < num_col
            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng)
            )

    def run(w_loc, w_scale, clen, vector, seed):
        fn = pl.pallas_call(
            kernel,
            grid=(pl.cdiv(dim, block_size),),
            input_output_aliases={5: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_loc, w_scale, clen, vector, seed, placeholder)

    return run


def _jitc_mv_normal_jvp_v(v_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """JVP rule for the vector argument of binary_jitnmv."""
    return jitnmv_p_call(
        w_loc, w_scale, clen, v_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mv_normal_jvp_wloc(w_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """JVP rule for the w_loc argument of binary_jitnmv."""
    return binary_jitnmv_p_call(
        w_dot, w_scale, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mv_normal_jvp_wscale(w_dot, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """JVP rule for the w_scale argument of binary_jitnmv."""
    return binary_jitnmv_p_call(
        w_loc, w_dot, clen, vector, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mv_normal_transpose_rules(ct, w_loc, w_scale, clen, vector, seed, *, shape, transpose, corder, **kwargs):
    """Transpose (VJP) rule for binary_jitnmv."""
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(vector):
        r = jitnmv_p_call(
            w_loc,
            w_scale,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            backend=kwargs['backend'],
        )[0]
        return w_loc, w_scale, clen, r, seed
    elif ad.is_undefined_primal(w_loc):
        # M = (w_loc + w_scale * Z) * mask, forward: M @ event(v)
        # d(loss)/d(w_loc) = sum((mask^T @ ct) * vector)
        r = jitnmv_p_call(
            1., 0., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw_loc = jnp.expand_dims(jnp.sum(r * vector), axis=0)
        return dw_loc, w_scale, clen, vector, seed
    elif ad.is_undefined_primal(w_scale):
        # d(loss)/d(w_scale) = sum(((Z*mask)^T @ ct) * vector)
        r = jitnmv_p_call(
            0., 1., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw_scale = jnp.expand_dims(jnp.sum(r * vector), axis=0)
        return w_loc, dw_scale, clen, vector, seed
    else:
        raise NotImplementedError(
            f"Transpose rule for binary_jitnmv not implemented "
            f"when none of vector/w_loc/w_scale is an undefined primal."
        )


def _jitc_mv_normal_batching(args, axes, **kwargs):
    """Batching rule for binary_jitnmv, promoting to binary_jitnmm when batched."""
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitnmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3].T,
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs['backend'],
        )
        return r, [1]
    elif tuple(axes) == (None, None, None, 1, None):
        assert args[3].ndim == 2, 'Batching axis 0 requires 2D input.'
        r = binary_jitnmm_p_call(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            shape=kwargs['shape'],
            transpose=kwargs['transpose'],
            corder=kwargs['corder'],
            backend=kwargs['backend'],
        )
        return r, [1]
    else:
        return general_batching_rule(
            binary_jitnmv_p,
            args,
            axes,
            **kwargs,
        )


def _binary_jitnmv_benchmark_data(*, platform):
    """Generate benchmark configurations for binary_jitnmv."""
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                w_loc = jnp.ones(1, dtype=dtype)
                w_scale = jnp.ones(1, dtype=dtype) * 0.1
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                v_size = n_post if not transpose else n_pre
                if bool_event:
                    vector = jnp.asarray(np.random.rand(v_size) > 0.5, dtype=jnp.bool_)
                else:
                    vector = jnp.asarray(np.random.rand(v_size), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(BenchmarkConfig(name, (w_loc, w_scale, clen, vector, seed), {
                    'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder
                }))
    return configs


def binary_jitnmv_p_call(
    w_loc,
    w_scale,
    clen,
    vector,
    seed,
    *,
    shape: Sequence[int],
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for event-driven matrix-vector multiplication with normal weights.

    This function validates inputs, computes the output shape, and dispatches to
    the ``binary_jitnmv_p`` XLA custom kernel. It is called by ``binary_jitnmv``
    after unit handling and seed initialization, and is also used directly by
    JVP, transpose, and batching rules.

    Parameters
    ----------
    w_loc : jax.Array
        Location (mean) parameter, shape ``(1,)``.
    w_scale : jax.Array
        Scale (standard deviation) parameter, shape ``(1,)``.
    clen : jax.Array
        Connection length (approximately ``2 / prob``), shape ``(1,)``.
    vector : jax.Array
        1-D binary event vector.
    seed : jax.Array
        Random seed, shape ``(1,)``.
    shape : Sequence[int]
        Shape of the implicit connectivity matrix ``(m, n)``.
    transpose : bool
        If True, compute ``M.T @ v`` instead of ``M @ v``.
    corder : bool
        Memory layout order flag for kernel dispatch.
    backend : str, optional
        Compute backend override.

    Returns
    -------
    tuple
        A single-element tuple containing the output vector as a JAX array.

    Raises
    ------
    AssertionError
        If input shapes or dimensions are incompatible.

    See Also
    --------
    binary_jitnmv : High-level API with unit handling.
    binary_jitnmv_p : The underlying XLA custom kernel primitive.
    """
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert w_loc.shape == (1,), f"The weight shape should be (1,), but got {w_loc.shape}."
    assert w_scale.shape == (1,), f"The weight shape should be (1,), but got {w_scale.shape}."
    assert clen.shape == (1,), f"The clen shape should be (1,), but got {clen.shape}."
    assert vector.ndim == 1, f"The vector should be a 1D array, but got {vector.ndim}D."
    assert seed.shape == (1,), f"The seed shape should be (1,), but got {seed.shape}."

    if transpose:
        assert shape[0] == len(vector), f"The matrix shape and vector length do not match. {vector.shape} @ {shape}"
    else:
        assert shape[1] == len(vector), f"The matrix shape and vector length do not match. {shape} @ {vector.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1]], w_loc.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0]], w_loc.dtype)
    )

    return binary_jitnmv_p(
        w_loc,
        w_scale,
        clen,
        vector,
        seed,
        outs=[out_info],
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        vector_info=jax.ShapeDtypeStruct(vector.shape, vector.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


binary_jitnmv_p = XLACustomKernel(
    'event_jitc_mv_normal',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_jitnmv``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) JIT normal connectivity
matrix-vector multiplication operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

In this operation, the connectivity matrix has weights normally distributed with specified
mean and standard deviation, and the input vector is treated as binary events (spikes).
Only active events contribute to the output computation.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_jitnmv_p.available_backends(platform)``,
and the default backend can be configured with ``binary_jitnmv_p.set_default(platform, backend)``.

See Also
--------
binary_jitnmv : High-level user-facing function wrapper.
"""
)
binary_jitnmv_p.def_numba_kernel(_jitc_mv_normal_numba_kernel_generator)
binary_jitnmv_p.def_pallas_kernel('gpu', _jitc_mv_normal_pallas_kernel_generator)
binary_jitnmv_p.def_jvp_rule2(_jitc_mv_normal_jvp_wloc, _jitc_mv_normal_jvp_wscale, None, _jitc_mv_normal_jvp_v, None)
binary_jitnmv_p.def_transpose_rule(_jitc_mv_normal_transpose_rules)
binary_jitnmv_p.def_batching_rule(_jitc_mv_normal_batching)
binary_jitnmv_p.def_tags('jit_normal', 'binary')
binary_jitnmv_p.def_benchmark_data(_binary_jitnmv_benchmark_data)


def _jitc_mm_normal_numba_kernel_generator(
    transpose: bool,
    corder: bool,
    B_info: jax.ShapeDtypeStruct,
    **kwargs
):
    r"""
    Generate the CPU kernel for the :func:`_jitc_matmat_normal` operation.
    """
    import numba
    _lfsr_seed = get_numba_lfsr_seed()
    _lfsr_random_integers = get_numba_lfsr_random_integers()
    _lfsr_normal = get_numba_lfsr_normal()

    if corder:
        # JIT Matrix.T @ B
        # - JIT matrix: [k, m]
        # - B: [k, n]
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_m in range(m):
                    state = _lfsr_seed(seed0 + i_m * k)
                    i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n, dtype=posts.dtype)
                    while i_k < k:
                        w = _lfsr_normal(state, w_loc0, w_scale0)
                        for j in range(B.shape[1]):
                            if B[i_k, j]:
                                out[j] += w
                        i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_m] = out
        else:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                n = posts.shape[1]
                k = B.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_m in range(m):
                    state = _lfsr_seed(seed0 + i_m * k)
                    i_k = _lfsr_random_integers(state, 0, clen0 - 1)
                    out = np.zeros(n, dtype=posts.dtype)
                    while i_k < k:
                        w = _lfsr_normal(state, w_loc0, w_scale0)
                        for j in range(B.shape[1]):
                            if B[i_k, j] > 0.:
                                out[j] += w
                        i_k += _lfsr_random_integers(state, 1, clen0 - 1)
                    posts[i_m] = out

    else:
        # JIT Matrix.T @ B
        # - JIT matrix: [k, m]
        # - B: [k, n]
        if B_info.dtype == jnp.bool_:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_k in range(k):
                    state = _lfsr_seed(seed0 + i_k * m)
                    indices = np.where(B[i_k])[0]
                    i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                    while i_m < m:
                        w = _lfsr_normal(state, w_loc0, w_scale0)
                        posts[i_m, indices] += w
                        i_m += _lfsr_random_integers(state, 1, clen0 - 1)
        else:
            @numba.njit(fastmath=True)
            def kernel(w_loc, w_scale, clen, B, seed, posts):
                posts[:] = 0.
                m = posts.shape[0]
                k = B.shape[0]
                w_loc0 = w_loc[0]
                w_scale0 = w_scale[0]
                seed0 = seed[0]
                clen0 = clen[0]
                for i_k in range(k):
                    state = _lfsr_seed(seed0 + i_k * m)
                    indices = np.where(B[i_k] > 0.)[0]
                    i_m = _lfsr_random_integers(state, 0, clen0 - 1)
                    while i_m < m:
                        w = _lfsr_normal(state, w_loc0, w_scale0)
                        posts[i_m, indices] += w
                        i_m += _lfsr_random_integers(state, 1, clen0 - 1)

    def run(w_loc, w_scale, clen, B, seed):
        return numba_kernel(kernel, outs=kwargs['outs'])(w_loc, w_scale, clen, B, seed)

    return run


def _jitc_mm_normal_pallas_kernel_generator(
    B_info: jax.ShapeDtypeStruct,
    out_info: jax.ShapeDtypeStruct,
    corder: bool = True,
    **kwargs
):
    """
    Pallas GPU kernel for binary event matmat with normal-distributed JITC matrix.

    Matches _jitnmm_pallas_kernel_generator in float.py:
    - Grid: (row_or_k_blocks, B_cols) — each block processes one B column
    - corder=True:  vectorize over output rows, seed by i_rows, loop over k
    - corder=False: vectorize over k, seed by i_ks, loop over output rows
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas.triton import atomic_add  # type: ignore[assignment]

    _PallasLFSRRNG = get_pallas_lfsr_rng_class()

    B_cols = B_info.shape[1]

    if corder:
        # Match float.py _jitnmm_pallas corder=True exactly:
        # Grid: (row_blocks, B_cols). Each block processes one B column.
        out_rows = out_info.shape[0]
        row_block = generate_block_dim(out_rows, maximum=128)
        grid = (pl.cdiv(out_rows, row_block), B_cols)

        def kernel(w_loc_ref, w_scale_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            k = B_ref.shape[0]
            w_loc0 = w_loc_ref[0]
            w_scale0 = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_row_block = pl.program_id(0)
            col_j = pl.program_id(1)

            i_rows = i_row_block * row_block + jnp.arange(row_block)
            i_row_mask = i_rows < out_rows
            safe_rows = jnp.where(i_row_mask, i_rows, 0)

            rng = _PallasLFSRRNG(seed0 + i_rows * k)
            i_cols = rng.random_integers(0, clen0)
            i_col_mask = i_cols < k

            out = jnp.zeros(row_block, dtype=post_ref.dtype)

            def body(data):
                i_cols, i_col_mask, rng, out = data
                w = rng.normal(w_loc0, w_scale0)
                safe_cols = jnp.where(i_col_mask, i_cols, 0)
                b_vals = B_ref[safe_cols, col_j]
                # Binary thresholding: treat b_vals as events
                if B_ref.dtype == jnp.bool_:
                    events = jnp.asarray(b_vals, dtype=out.dtype)
                else:
                    events = jnp.where(b_vals > 0., 1., 0.)
                out += jnp.where(i_col_mask & i_row_mask, w * events, 0.)
                i_cols += rng.random_integers(1, clen0)
                return i_cols, i_cols < k, rng, out

            _, _, _, out = jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_cols, i_col_mask, rng, out)
            )
            atomic_add(post_ref, (safe_rows, col_j), out, mask=i_row_mask)

    else:
        # Match float.py _jitnmm_pallas corder=False exactly:
        # Grid: (k_blocks, B_cols). Each block processes one B column.
        k_dim = B_info.shape[0]
        k_block = generate_block_dim(k_dim, maximum=128)
        grid = (pl.cdiv(k_dim, k_block), B_cols)

        def kernel(w_loc_ref, w_scale_ref, clen_ref, B_ref, seed_ref, _, post_ref):
            m = post_ref.shape[0]
            w_loc0 = w_loc_ref[0]
            w_scale0 = w_scale_ref[0]
            clen0 = clen_ref[0]
            seed0 = seed_ref[0]
            i_k_block = pl.program_id(0)
            col_j = pl.program_id(1)

            i_ks = i_k_block * k_block + jnp.arange(k_block)
            i_k_mask = i_ks < k_dim
            safe_ks = jnp.where(i_k_mask, i_ks, 0)

            # Preload B values for this column and apply binary thresholding
            b_vals = B_ref[safe_ks, col_j]
            if B_ref.dtype == jnp.bool_:
                b_events = jnp.asarray(b_vals, dtype=post_ref.dtype)
            else:
                b_events = jnp.where(b_vals > 0., 1., 0.)
            b_events = jnp.where(i_k_mask, b_events, 0.)

            rng = _PallasLFSRRNG(seed0 + i_ks * m)
            i_rows = rng.random_integers(0, clen0)
            i_row_mask = i_rows < m

            def body(data):
                i_rows, i_row_mask, rng = data
                w = rng.normal(w_loc0, w_scale0)
                vals = jnp.where(i_k_mask & i_row_mask, w * b_events, 0.)
                safe_rows = jnp.where(i_row_mask, i_rows, 0)
                atomic_add(post_ref, (safe_rows, col_j), vals,
                           mask=i_k_mask & i_row_mask)
                i_rows += rng.random_integers(1, clen0)
                return i_rows, i_rows < m, rng

            jax.lax.while_loop(
                lambda data: jnp.sum(data[1]) > 0,
                body,
                (i_rows, i_row_mask, rng)
            )

    def run(w_loc, w_scale, clen, B, seed):
        fn = pl.pallas_call(
            kernel,
            grid=grid,
            input_output_aliases={5: 0},
            out_shape=kwargs['outs'],
            backend='triton',
        )
        placeholder = jnp.zeros(kwargs['outs'][0].shape, kwargs['outs'][0].dtype)
        return fn(w_loc, w_scale, clen, B, seed, placeholder)

    return run


def _jitc_mm_normal_jvp_wloc(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """JVP rule for the w_loc argument of binary_jitnmm."""
    return binary_jitnmm_p_call(
        w_dot, w_scale, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mm_normal_jvp_wscale(w_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """JVP rule for the w_scale argument of binary_jitnmm."""
    return binary_jitnmm_p_call(
        w_loc, w_dot, clen, B, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mm_normal_jvp_B(B_dot, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """JVP rule for the B argument of binary_jitnmm."""
    return jitnmm_p_call(
        w_loc, w_scale, clen, B_dot, seed, shape=shape, transpose=transpose, corder=corder, backend=kwargs['backend'],
    )


def _jitc_mm_normal_transpose_rules(ct, w_loc, w_scale, clen, B, seed, *, shape, transpose, corder, **kwargs):
    """Transpose (VJP) rule for binary_jitnmm."""
    assert not ad.is_undefined_primal(clen)
    assert not ad.is_undefined_primal(seed)

    ct = ct[0]
    if ad.is_undefined_primal(B):
        r = jitnmm_p_call(
            w_loc,
            w_scale,
            clen,
            ct,
            seed,
            shape=shape,
            transpose=not transpose,
            corder=not corder,
            backend=kwargs['backend'],
        )[0]
        return w_loc, w_scale, clen, r, seed
    elif ad.is_undefined_primal(w_loc):
        # M = (w_loc + w_scale * Z) * mask, forward: M @ event(B)
        # d(loss)/d(w_loc) = sum((mask^T @ ct) * B)
        r = jitnmm_p_call(
            1., 0., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw_loc = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return dw_loc, w_scale, clen, B, seed
    elif ad.is_undefined_primal(w_scale):
        # d(loss)/d(w_scale) = sum(((Z*mask)^T @ ct) * B)
        r = jitnmm_p_call(
            0., 1., clen, ct, seed,
            shape=shape, transpose=not transpose, corder=not corder,
            backend=kwargs['backend'],
        )[0]
        dw_scale = jnp.expand_dims(jnp.sum(r * B), axis=0)
        return w_loc, dw_scale, clen, B, seed
    else:
        raise NotImplementedError(
            'Transpose rules for binary_jitc_matmat_normal not implemented for '
            'non-undefined primals.'
        )


def _batching_axis1(args, axis=1, **kwargs):
    """Helper for batching binary_jitnmm along axis 1 (or transposed axis 0)."""
    assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
    m, maybe_batch1, maybe_batch2 = args[3].shape
    B = args[3].reshape(m, maybe_batch1 * maybe_batch2)
    r = binary_jitnmm_p_call(
        args[0],
        args[1],
        args[2],
        B,
        args[4],
        shape=kwargs['shape'],
        transpose=kwargs['transpose'],
        corder=kwargs['corder'],
        backend=kwargs['backend'],
    )
    r = jnp.reshape(r[0], [r[0].shape[0], maybe_batch1, maybe_batch2])
    return [r], [axis]


def _jitc_mm_normal_batching(args, axes, **kwargs):
    """Batching rule for binary_jitnmm."""
    if tuple(axes) == (None, None, None, 0, None):
        assert args[3].ndim == 3, 'Batching axis 0 requires 3D input.'
        args = list(args)
        args[3] = jnp.transpose(args[3], (1, 0, 2))
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, None, 1, None):
        return _batching_axis1(args, **kwargs)

    elif tuple(axes) == (None, None, None, 2, None):
        return _batching_axis1(args, axis=2, **kwargs)

    else:
        return general_batching_rule(binary_jitnmm_p, args, axes, **kwargs)


def _binary_jitnmm_benchmark_data(*, platform):
    """Generate benchmark configurations for binary_jitnmm."""
    n_pre, n_post, prob, dtype = 1000, 1000, 0.1, jnp.float32
    configs = []
    for transpose in (False, True):
        for corder in (True, False):
            for bool_event in (True, False):
                w_loc = jnp.ones(1, dtype=dtype)
                w_scale = jnp.ones(1, dtype=dtype) * 0.1
                clen = jnp.atleast_1d(jnp.asarray(2.0 / prob, dtype=dtype))
                b_rows = n_post if not transpose else n_pre
                if bool_event:
                    B = jnp.asarray(np.random.rand(b_rows, 10) > 0.5, dtype=jnp.bool_)
                else:
                    B = jnp.asarray(np.random.rand(b_rows, 10), dtype=dtype)
                seed = jnp.asarray(42, dtype=jnp.uint32)
                name = f"{'T' if transpose else 'NT'},{'corder' if corder else 'rorder'},{'bool' if bool_event else 'float'}"
                configs.append(
                    BenchmarkConfig(
                        name,
                        (w_loc, w_scale, clen, B, seed),
                        {'shape': (n_pre, n_post), 'transpose': transpose, 'corder': corder}
                    )
                )
    return configs


def binary_jitnmm_p_call(
    w_loc,
    w_scale,
    clen,
    B,
    seed,
    *,
    shape: MatrixShape,
    transpose: bool,
    corder: bool,
    backend: Optional[str] = None,
):
    """
    Low-level primitive call for event-driven matrix-matrix multiplication with normal weights.

    This function validates inputs, computes the output shape, and dispatches to
    the ``binary_jitnmm_p`` XLA custom kernel. It is called by ``binary_jitnmm``
    after unit handling and seed initialization, and is also used directly by
    JVP, transpose, and batching rules.

    Parameters
    ----------
    w_loc : jax.Array
        Location (mean) parameter, shape ``(1,)``.
    w_scale : jax.Array
        Scale (standard deviation) parameter, shape ``(1,)``.
    clen : jax.Array
        Connection length (approximately ``2 / prob``), shape ``(1,)``.
    B : jax.Array
        2-D binary event matrix, shape ``(k, n)``.
    seed : jax.Array
        Random seed, shape ``(1,)``.
    shape : MatrixShape
        Shape of the implicit connectivity matrix ``(m, k)`` (or ``(k, m)`` when
        ``transpose=True``).
    transpose : bool
        If True, compute ``M.T @ B`` instead of ``M @ B``.
    corder : bool
        Memory layout order flag for kernel dispatch.
    backend : str, optional
        Compute backend override.

    Returns
    -------
    tuple
        A single-element tuple containing the output matrix as a JAX array.

    Raises
    ------
    AssertionError
        If input shapes or dimensions are incompatible.

    See Also
    --------
    binary_jitnmm : High-level API with unit handling.
    binary_jitnmm_p : The underlying XLA custom kernel primitive.
    """
    w_loc = jnp.atleast_1d(w_loc)
    w_scale = jnp.atleast_1d(w_scale)
    clen = jnp.atleast_1d(clen)

    assert len(shape) == 2, "The matrix shape should be a tuple of two integers."
    assert B.ndim == 2, "The input matrix B should be a 2D array."
    assert seed.ndim == 1, "The seed should be a 1D array."
    assert w_loc.ndim == 1, "The weight should be a 1D array."
    assert w_scale.ndim == 1, "The weight should be a 1D array."
    assert clen.ndim == 1, "The clen should be a 1D array."
    assert w_loc.shape == (1,), "The weight should be a scalar."
    assert w_scale.shape == (1,), "The weight should be a scalar."
    assert clen.shape == (1,), "The clen should be a scalar."
    assert seed.shape == (1,), "The seed should be a scalar."
    if transpose:
        assert shape[0] == B.shape[0], f"The matrix shape and B shape do not match. {B.shape} @ {shape}"
    else:
        assert shape[1] == B.shape[0], f"The matrix shape and B shape do not match. {shape} @ {B.shape}"

    out_info = (
        jax.ShapeDtypeStruct([shape[1], B.shape[1]], w_loc.dtype)
        if transpose else
        jax.ShapeDtypeStruct([shape[0], B.shape[1]], w_loc.dtype)
    )

    return binary_jitnmm_p(
        w_loc,
        w_scale,
        clen,
        B,
        seed,
        outs=[out_info],
        w_loc_info=jax.ShapeDtypeStruct(w_loc.shape, w_loc.dtype),
        w_scale_info=jax.ShapeDtypeStruct(w_scale.shape, w_scale.dtype),
        clen_info=jax.ShapeDtypeStruct(clen.shape, clen.dtype),
        B_info=jax.ShapeDtypeStruct(B.shape, B.dtype),
        seed_info=jax.ShapeDtypeStruct(seed.shape, seed.dtype),
        out_info=out_info,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=backend,
    )


binary_jitnmm_p = XLACustomKernel(
    'binary_jitc_mm_normal',
    doc="""
Low-level XLA custom-kernel primitive for ``binary_jitnmm``.

This ``XLACustomKernel`` instance dispatches the binary (event-driven) JIT normal connectivity
matrix-matrix multiplication operation to registered backends (``numba``, ``pallas``),
using runtime shape/dtype metadata provided by the high-level wrapper.

In this operation, the connectivity matrix has weights normally distributed with specified
mean and standard deviation, and the input matrix is treated as binary events (spikes).
Each column of the input is processed independently as an event vector.

Beyond backend dispatch, the primitive stores JAX transformation bindings
(JVP, transpose, batching, and call registration) so the operation integrates
correctly with ``jit``, ``vmap``, and autodiff.

Available backends can be queried with ``binary_jitnmm_p.available_backends(platform)``,
and the default backend can be configured with ``binary_jitnmm_p.set_default(platform, backend)``.

See Also
--------
binary_jitnmm : High-level user-facing function wrapper.
"""
)
binary_jitnmm_p.def_numba_kernel(_jitc_mm_normal_numba_kernel_generator)
binary_jitnmm_p.def_pallas_kernel('gpu', _jitc_mm_normal_pallas_kernel_generator)
binary_jitnmm_p.def_jvp_rule2(_jitc_mm_normal_jvp_wloc, _jitc_mm_normal_jvp_wscale, None, _jitc_mm_normal_jvp_B, None)
binary_jitnmm_p.def_transpose_rule(_jitc_mm_normal_transpose_rules)
binary_jitnmm_p.def_batching_rule(_jitc_mm_normal_batching)
binary_jitnmm_p.def_tags('jit_normal', 'binary')
binary_jitnmm_p.def_benchmark_data(_binary_jitnmm_benchmark_data)
