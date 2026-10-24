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


# ---------------------------------------------------------------------------
# Helpers for the event-activity tests (``x != 0`` contract, see
# ``docs/specs/2026-10-24-event-activity-and-abi.md``).
# ---------------------------------------------------------------------------

#: Event dtypes swept by the activity tests. ``float64`` needs x64 mode; wrap
#: the test body in :func:`event_dtype_context`.
EVENT_DTYPE_NAMES = (
    'bool', 'int8', 'uint8', 'int16', 'int32', 'float16', 'bfloat16', 'float32', 'float64',
)

#: Event dtypes swept by the weighted (``WeightedArray``) tests: every
#: non-boolean dtype of :data:`EVENT_DTYPE_NAMES`.
WEIGHTED_EVENT_DTYPE_NAMES = tuple(name for name in EVENT_DTYPE_NAMES if name != 'bool')

_EVENT_PATTERNS = {
    'bool': [True, True, False, True, False, True, True, False],
    'uint8': [2, 255, 0, 1, 0, 7, 1, 0],
    'signed': [2, -1, 0, 3, 0, -3, 1, 0],
    'float': [2.0, -1.0, 0.0, 0.5, 0.0, -3.0, 1.0, 0.0],
}


def event_dtype_context(dtype_name):
    """Return a context manager enabling x64 when ``dtype_name`` needs it.

    Parameters
    ----------
    dtype_name : str
        One of :data:`EVENT_DTYPE_NAMES`.

    Returns
    -------
    contextlib.AbstractContextManager
        ``jax_x64_enabled()`` for ``'float64'``, otherwise a no-op context.
    """
    import contextlib
    from brainevent._test_util import jax_x64_enabled

    return jax_x64_enabled() if dtype_name == 'float64' else contextlib.nullcontext()


def event_values(dtype_name, shape, seed=0):
    """Return an event array with positive, negative, fractional and zero entries.

    Parameters
    ----------
    dtype_name : str
        One of :data:`EVENT_DTYPE_NAMES`. Must be called inside
        :func:`event_dtype_context` for ``'float64'``.
    shape : int or tuple of int
        Output shape.
    seed : int, optional
        Seed of the shuffle.

    Returns
    -------
    jax.Array
        Events of dtype ``dtype_name``. Floats contain ``2, -1, 0, 0.5``;
        signed integers ``2, -1, 0``; ``uint8`` ``2, 255, 0``. About a third of
        the entries are zero.
    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    size = int(np.prod(shape))
    if dtype_name == 'bool':
        key = 'bool'
    elif dtype_name == 'uint8':
        key = 'uint8'
    elif dtype_name.startswith('int'):
        key = 'signed'
    else:
        key = 'float'
    pattern = np.asarray(_EVENT_PATTERNS[key])
    values = np.resize(pattern, size)
    values = np.random.default_rng(seed).permutation(values).reshape(shape)
    dtype = jnp.dtype(dtype_name)
    if key == 'float':
        return jnp.asarray(values.astype(np.float32)).astype(dtype)
    return jnp.asarray(values.astype(dtype))


def random_csr_numpy(rng, m, k, density=0.4, weight_dtype=np.float32):
    """Return a random CSR matrix as numpy arrays plus its dense form.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random source.
    m, k : int
        Matrix shape.
    density : float, optional
        Fraction of stored entries.
    weight_dtype : dtype, optional
        Dtype of the stored values.

    Returns
    -------
    data : numpy.ndarray
        Stored values, shape ``(nse,)``.
    indices : numpy.ndarray
        int32 column indices, shape ``(nse,)``.
    indptr : numpy.ndarray
        int32 row pointers, shape ``(m + 1,)``. Some rows may be empty.
    dense : numpy.ndarray
        The ``(m, k)`` dense matrix (float64) for the reference product.
    """
    mask = rng.random((m, k)) < density
    rows, cols = np.nonzero(mask)
    data = rng.standard_normal(rows.size).astype(weight_dtype)
    indptr = np.concatenate([[0], np.cumsum(mask.sum(axis=1))]).astype(np.int32)
    dense = np.zeros((m, k), dtype=np.float64)
    dense[rows, cols] = data.astype(np.float64)
    return data, cols.astype(np.int32), indptr, dense


def event_mask(x):
    """Return the reference activity mask ``x != 0`` as a float64 numpy array."""
    return np.asarray(jnp.asarray(x) != 0).astype(np.float64)


def run_python_isolated(source, timeout=600):
    """Run ``source`` in a fresh Python interpreter and return the result.

    Used to check that a failing CUDA call raises a Python exception instead
    of killing the process: a signal-terminated child has a negative
    ``returncode``.

    Parameters
    ----------
    source : str
        Python source to execute with ``python -c``.
    timeout : float, optional
        Seconds before the child is killed.

    Returns
    -------
    subprocess.CompletedProcess
        With ``stdout``/``stderr`` captured as text.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run(
        [sys.executable, '-c', source],
        capture_output=True, text=True, timeout=timeout, env=env, cwd=repo_root,
    )


def heavy_row_csr_numpy(rng, m, k, row_nnz, weight_dtype=np.float32):
    """Return a CSR matrix whose rows are long enough for the hybrid task path.

    The hybrid CUDA scatter kernels split rows with more than
    ``BE_HYBRID_TPR_THRESHOLD`` (128) stored entries into chunks of
    ``BE_HYBRID_TASK_NNZ`` (4096) entries; ``row_nnz > 4096`` therefore
    exercises multi-chunk rows.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random source.
    m, k : int
        Matrix shape; ``row_nnz <= k``.
    row_nnz : int
        Stored entries per row (row ``i`` gets ``row_nnz - i``, so rows differ).
    weight_dtype : dtype, optional
        Dtype of the stored values.

    Returns
    -------
    data, indices, indptr, dense : numpy.ndarray
        As in :func:`random_csr_numpy`.
    """
    counts = [row_nnz - i for i in range(m)]
    cols = [np.sort(rng.choice(k, c, replace=False)) for c in counts]
    indices = np.concatenate(cols).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    data = rng.standard_normal(indices.size).astype(weight_dtype)
    dense = np.zeros((m, k), dtype=np.float64)
    rows = np.repeat(np.arange(m), counts)
    dense[rows, indices] = data.astype(np.float64)
    return data, indices, indptr, dense
