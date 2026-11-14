# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Shape-only hybrid MM planning (docs/specs/2026-11-14-csr-mm-shape-plan.md)."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent
from brainevent import BinaryArray, CSR
from brainevent._test_util import requires_gpu

ROWS, COLS = 64, 600
# Same shape and nnz (512); the row-length statistics differ.
UNIFORM = np.full(ROWS, 8)
SKEWED = np.r_[512 - (ROWS - 1), np.ones(ROWS - 1, dtype=int)]


def _arrays(lengths, seed=0):
    rng = np.random.default_rng(seed)
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    indices = np.concatenate([np.sort(rng.choice(COLS, n, replace=False)) for n in lengths]).astype(np.int32)
    weights = rng.normal(size=indices.size).astype(np.float32)
    return jnp.asarray(weights), jnp.asarray(indices), jnp.asarray(indptr)


def _dense(weights, indices, indptr):
    dense = np.zeros((ROWS, COLS), np.float32)
    rows = np.repeat(np.arange(ROWS), np.diff(np.asarray(indptr)))
    np.add.at(dense, (rows, np.asarray(indices)), np.asarray(weights))
    return dense


def _events(batch=5, seed=1):
    return jnp.asarray(np.random.default_rng(seed).random((ROWS, batch)) < 0.3)


def _native_targets(function, *args):
    text = jax.jit(function).lower(*args).compile().as_text()
    return [t for t in re.findall(r'custom_call_target="([^"]+)"', text) if t.startswith('csr_hybrid_mm')]


def test_matrix_config_holds_no_structure_state():
    skewed = CSR(_arrays(SKEWED), shape=(ROWS, COLS))
    uniform = CSR(_arrays(UNIFORM), shape=(ROWS, COLS))
    assert jax.tree.leaves(skewed.config) == []
    assert jax.tree.structure(skewed) == jax.tree.structure(uniform)


def test_redrawn_structures_share_one_executable():
    product = jax.jit(lambda matrix, events: BinaryArray(events.T) @ matrix)
    events = _events()
    for lengths in (UNIFORM, SKEWED, UNIFORM[::-1]):
        arrays = _arrays(lengths)
        out = product(CSR(arrays, shape=(ROWS, COLS)), events)
        np.testing.assert_allclose(out, np.asarray(events, np.float32).T @ _dense(*arrays), rtol=1e-5, atol=1e-5)
    assert product._cache_size() == 1


def _build(path, arrays):
    """``(function, args)`` computing ``(W.T @ events).T`` for one construction path."""
    shape = (ROWS, COLS)
    if path == 'eager':
        matrix = CSR(arrays, shape=shape)
        return (lambda events: BinaryArray(events.T) @ matrix), ()
    if path == 'traced':
        return (lambda w, i, p, events: BinaryArray(events.T) @ CSR((w, i, p), shape=shape)), arrays
    if path == 'cpu_built':
        cpu = jax.devices('cpu')[0]
        with jax.default_device(cpu):
            matrix = CSR(tuple(jax.device_put(a, cpu) for a in arrays), shape=shape)
        matrix = jax.device_put(matrix, jax.devices('gpu')[0])
        return (lambda m, events: BinaryArray(events.T) @ m), (matrix,)
    assert path == 'vmap_mv'
    return (lambda w, i, p, events: jax.vmap(lambda v: BinaryArray(v) @ CSR((w, i, p), shape=shape))(events.T)), arrays


@requires_gpu
@pytest.mark.parametrize('lengths', [UNIFORM, SKEWED], ids=['uniform', 'skewed'])
@pytest.mark.parametrize('path', ['eager', 'traced', 'cpu_built', 'vmap_mv'])
def test_every_construction_path_lowers_the_native_hybrid_mm(path, lengths):
    arrays = _arrays(lengths)
    events = _events()
    function, args = _build(path, arrays)
    assert _native_targets(function, *args, events)
    expected = np.asarray(events, np.float32).T @ _dense(*arrays)
    np.testing.assert_allclose(jax.jit(function)(*args, events), expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_global_cuda_raw_keeps_untransposed_sddmm_on_a_supported_kernel():
    weights, indices, indptr = _arrays(SKEWED)
    events = jnp.asarray(np.random.default_rng(2).random((COLS, 3)) < 0.5)
    ct = jnp.asarray(np.random.default_rng(3).normal(size=(ROWS, 3)).astype(np.float32))
    expected = brainevent._sddmm.csr_sddmm_dweight(events, ct, indices, indptr, transpose=False)
    previous = brainevent.config.get_backend('gpu')
    brainevent.config.set_backend('gpu', 'cuda_raw')
    try:
        out = brainevent._sddmm.csr_sddmm_dweight(events, ct, indices, indptr, transpose=False)
    finally:
        brainevent.config.set_backend('gpu', previous)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
