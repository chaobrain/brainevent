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

import brainstate
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def get_csr(n_pre, n_post, prob, replace=True):
    """Generate random CSR index arrays for testing.

    Parameters
    ----------
    n_pre : int
        Number of rows (pre-synaptic neurons).
    n_post : int
        Number of columns (post-synaptic neurons).
    prob : float
        Connection probability, determining the number of non-zeros
        per row as ``int(n_post * prob)``.
    replace : bool, optional
        Whether to sample column indices with replacement.
        Default is ``True``.

    Returns
    -------
    indptr : numpy.ndarray
        Row pointer array of shape ``(n_pre + 1,)``.
    indices : jax.Array
        Column index array of shape ``(n_pre * n_conn,)``.
    """
    n_conn = int(n_post * prob)
    indptr = np.arange(n_pre + 1) * n_conn
    if replace:
        indices = brainstate.random.randint(0, n_post, (n_pre * n_conn,))
    else:
        indices = brainstate.transform.for_loop(
            lambda *args: brainstate.random.choice(n_post, n_conn, replace=False),
            length=n_pre
        ).flatten()
    return indptr, indices


def _get_n_conn(indptr):
    """Extract n_conn from indptr as a concrete integer."""
    if hasattr(indptr, '__array__'):
        arr = np.asarray(indptr)
        return int(arr[1] - arr[0])
    return int(indptr[1] - indptr[0])


def vector_csr(x, w, indices, indptr, shape, n_conn=None):
    """Reference implementation of transposed CSR matrix-vector product.

    Computes ``y = A.T @ x`` where ``A`` is a CSR sparse matrix.

    Parameters
    ----------
    x : jax.Array
        Input vector of shape ``(shape[0],)``.
    w : jax.Array
        Non-zero values.  Shape ``(nse,)`` or ``(1,)`` for homogeneous.
    indices : jax.Array
        Column indices of the non-zero entries.
    indptr : jax.Array
        Row pointer array of the CSR format.
    shape : tuple of int
        Logical matrix shape ``(m, n)``.
    n_conn : int or None, optional
        Number of connections per row.  Inferred from ``indptr`` if ``None``.

    Returns
    -------
    jax.Array
        Result vector of shape ``(shape[1],)``.
    """
    if n_conn is None:
        n_conn = _get_n_conn(indptr)
    return _vector_csr_impl(x, w, indices, indptr, shape, n_conn)


@brainstate.transform.jit(static_argnums=(4, 5), static_argnames=['shape', 'n_conn'])
def _vector_csr_impl(x, w, indices, indptr, shape, n_conn):
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((shape[1],))

    def body_fn(i_pre, post):
        start = indptr[i_pre]
        ids = jax.lax.dynamic_slice(indices, (start,), (n_conn,))
        if homo_w:
            inc = w * x[i_pre]
        else:
            ws = jax.lax.dynamic_slice(w, (start,), (n_conn,))
            inc = ws * x[i_pre]
        ids, inc = jnp.broadcast_arrays(ids, inc)
        return post.at[ids].add(inc)

    return jax.lax.fori_loop(0, x.shape[0], body_fn, post)


def matrix_csr(xs, w, indices, indptr, shape, n_conn=None):
    """Reference implementation of transposed CSR matrix-matrix product.

    Computes ``Y = A.T @ X`` where ``A`` is a CSR sparse matrix and
    ``X`` is a dense matrix.

    Parameters
    ----------
    xs : jax.Array
        Input matrix of shape ``(batch, shape[0])``.
    w : jax.Array
        Non-zero values.  Shape ``(nse,)`` or ``(1,)`` for homogeneous.
    indices : jax.Array
        Column indices of the non-zero entries.
    indptr : jax.Array
        Row pointer array of the CSR format.
    shape : tuple of int
        Logical matrix shape ``(m, n)``.
    n_conn : int or None, optional
        Number of connections per row.  Inferred from ``indptr`` if ``None``.

    Returns
    -------
    jax.Array
        Result matrix of shape ``(batch, shape[1])``.
    """
    if n_conn is None:
        n_conn = _get_n_conn(indptr)
    return _matrix_csr_impl(xs, w, indices, indptr, shape, n_conn)


@brainstate.transform.jit(static_argnums=(4, 5), static_argnames=['shape', 'n_conn'])
def _matrix_csr_impl(xs, w, indices, indptr, shape, n_conn):
    homo_w = jnp.size(w) == 1
    post = jnp.zeros((xs.shape[0], shape[1]))

    def body_fn(i_pre, post):
        start = indptr[i_pre]
        ids = jax.lax.dynamic_slice(indices, (start,), (n_conn,))
        xs_col = jax.lax.dynamic_slice(xs, (0, i_pre), (xs.shape[0], 1))
        if homo_w:
            inc = w * xs_col
        else:
            ws = jax.lax.dynamic_slice(w, (start,), (n_conn,))
            inc = ws * xs_col
        return post.at[:, ids].add(inc)

    return jax.lax.fori_loop(0, xs.shape[1], body_fn, post)


def csr_vector(x, w, indices, indptr, shape, n_conn=None):
    """Reference implementation of CSR matrix-vector product.

    Computes ``y = A @ x`` where ``A`` is a CSR sparse matrix.

    Parameters
    ----------
    x : jax.Array
        Input vector of shape ``(shape[1],)``.
    w : jax.Array
        Non-zero values.  Shape ``(nse,)`` or ``(1,)`` for homogeneous.
    indices : jax.Array
        Column indices of the non-zero entries.
    indptr : jax.Array
        Row pointer array of the CSR format.
    shape : tuple of int
        Logical matrix shape ``(m, n)``.
    n_conn : int or None, optional
        Number of connections per row.  Inferred from ``indptr`` if ``None``.

    Returns
    -------
    jax.Array
        Result vector of shape ``(shape[0],)``.
    """
    if n_conn is None:
        n_conn = _get_n_conn(indptr)
    return _csr_vector_impl(x, w, indices, indptr, shape, n_conn)


@brainstate.transform.jit(static_argnums=(4, 5), static_argnames=['shape', 'n_conn'])
def _csr_vector_impl(x, w, indices, indptr, shape, n_conn):
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0]])

    def body_fn(i, out):
        start = indptr[i]
        ids = jax.lax.dynamic_slice(indices, (start,), (n_conn,))
        ws = w if homo_w else jax.lax.dynamic_slice(w, (start,), (n_conn,))
        return out.at[i].set(jnp.sum(x[ids] * ws))

    return jax.lax.fori_loop(0, shape[0], body_fn, out)


def csr_matrix(xs, w, indices, indptr, shape, n_conn=None):
    """Reference implementation of CSR matrix-matrix product.

    Computes ``Y = A @ X`` where ``A`` is a CSR sparse matrix and
    ``X`` is a dense matrix.

    Parameters
    ----------
    xs : jax.Array
        Input matrix of shape ``(shape[1], n_cols)``.
    w : jax.Array
        Non-zero values.  Shape ``(nse,)`` or ``(1,)`` for homogeneous.
    indices : jax.Array
        Column indices of the non-zero entries.
    indptr : jax.Array
        Row pointer array of the CSR format.
    shape : tuple of int
        Logical matrix shape ``(m, n)``.
    n_conn : int or None, optional
        Number of connections per row.  Inferred from ``indptr`` if ``None``.

    Returns
    -------
    jax.Array
        Result matrix of shape ``(shape[0], n_cols)``.
    """
    if n_conn is None:
        n_conn = _get_n_conn(indptr)
    return _csr_matrix_impl(xs, w, indices, indptr, shape, n_conn)


@brainstate.transform.jit(static_argnums=(4, 5), static_argnames=['shape', 'n_conn'])
def _csr_matrix_impl(xs, w, indices, indptr, shape, n_conn):
    # CSR @ matrix
    homo_w = jnp.size(w) == 1
    out = jnp.zeros([shape[0], xs.shape[1]])

    def body_fn(i, out):
        start = indptr[i]
        ids = jax.lax.dynamic_slice(indices, (start,), (n_conn,))
        ws = w if homo_w else jnp.expand_dims(jax.lax.dynamic_slice(w, (start,), (n_conn,)), axis=1)
        return out.at[i].set(jnp.sum(xs[ids] * ws, axis=0))

    return jax.lax.fori_loop(0, shape[0], body_fn, out)


def small_csr():
    """Return a tiny 2x3 CSR with int32 structure, for dtype/structure tests.

    Returns
    -------
    brainevent.CSR
        The matrix ``[[1, 0, 2], [0, 3, 0]]`` with float32 data and int32
        ``indices``/``indptr``.

    Examples
    --------
    .. code-block:: python

        >>> from brainevent._csr._test_util import small_csr
        >>> small_csr().shape
        (2, 3)
    """
    import brainevent

    data = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 3], dtype=jnp.int32)
    return brainevent.CSR((data, indices, indptr), shape=(2, 3))


# ---------------------------------------------------------------------------
# Helpers for the CUDA int64-indptr policy tests, shared by the per-module
# test files (binary, binary_indexed, float, slice, dt2t, plasticity_binary).
# ---------------------------------------------------------------------------

#: Skip when no GPU backend is active. Distinct from
#: ``brainevent._test_util.requires_gpu``: that one probes ``jax.devices('gpu')``,
#: this one requires the *default* backend to be the GPU, because these tests
#: compare a ``cuda_raw`` result against a ``jax_raw`` reference.
requires_gpu_backend = pytest.mark.skipif(
    jax.default_backend() != 'gpu',
    reason='CUDA int64 indptr tests require a GPU backend',
)


def strip_hybrid_suffix(name):
    """Remove the config-dependent hybrid module suffix (e.g. ``_b256_s2048_t128_n4096``).

    Parameters
    ----------
    name : str
        A compiled-module or FFI-target name, possibly carrying the suffix.

    Returns
    -------
    str
        ``name`` with the active hybrid-config suffix removed.

    Notes
    -----
    The hybrid ``.so`` module names embed the active
    :func:`~brainevent._csr.hybrid_config.get_hybrid_config` so different configs
    compile to distinct modules. These tests only care about the logical module
    identity, so the suffix is normalised away to stay config-agnostic.
    """
    from brainevent._csr.hybrid_config import get_hybrid_config, module_suffix_for_config

    suffix = module_suffix_for_config(get_hybrid_config())
    return name.replace(suffix, '') if suffix else name


def int64_structure(indptr_dtype):
    """Return a tiny ``(weights, indices, indptr)`` triple with the given indptr dtype.

    Parameters
    ----------
    indptr_dtype : dtype
        Dtype for the returned ``indptr`` array.

    Returns
    -------
    tuple
        ``(weights, indices, indptr)`` describing a 2x3 CSR with 4 stored entries.
    """
    weights = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    indices = jnp.array([0, 2, 1, 2], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 4], dtype=indptr_dtype)
    return weights, indices, indptr


def shape_of(dtype, shape=(2,)):
    """Return a :class:`jax.ShapeDtypeStruct` for ``shape``/``dtype``."""
    return jax.ShapeDtypeStruct(shape, dtype)


def cuda_kwargs(indices_dtype=jnp.int64, indptr_dtype=jnp.int64, nse=2):
    """Return the common kwargs a CSR CUDA kernel generator expects."""
    return {
        'outs': [shape_of(jnp.float32)],
        'shape': (1, 2),
        'indices_info': shape_of(indices_dtype, (nse,)),
        'indptr_info': shape_of(indptr_dtype, (2,)),
    }


def recording_ffi_call(calls):
    """Return a ``jax.ffi.ffi_call`` stand-in that records invocations into ``calls``.

    Parameters
    ----------
    calls : list
        Mutated in place; each entry is
        ``(name, out_info, ffi_kwargs, args, kwargs)``.

    Returns
    -------
    callable
        A drop-in replacement for ``jax.ffi.ffi_call`` returning zeros of the
        requested output shape, so a kernel can be exercised without a GPU.
    """

    def ffi_call(name, out_info, **ffi_kwargs):
        def call(*args, **kwargs):
            calls.append((name, out_info, ffi_kwargs, args, kwargs))
            if isinstance(out_info, (tuple, list)):
                return tuple(jnp.zeros(info.shape, info.dtype) for info in out_info)
            return jnp.zeros(out_info.shape, out_info.dtype)

        return call

    return ffi_call
