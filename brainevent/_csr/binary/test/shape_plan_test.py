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
from brainevent._csr.config import CSRConfig
from brainevent._test_util import requires_gpu

from brainevent import WeightedArray

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


# Edge cases of the lowering-time plan
# ------------------------------------

@requires_gpu
def test_native_mm_lowers_without_a_known_target_device(monkeypatch):
    # No device assignment, mesh or backend (an unusual portable lowering): the
    # plan assumes the default L2 instead of failing or leaving the native kernel.
    import brainevent._op.main as op_main
    from brainevent._csr.binary import binary as binary_mod
    from brainevent._op.lowering_target import DEFAULT_L2_BYTES
    seen = []
    real = binary_mod.cuda_l2_bytes
    monkeypatch.setattr(op_main, 'target_devices', lambda ctx: ())
    monkeypatch.setattr(binary_mod, 'cuda_l2_bytes', lambda devices: seen.append(tuple(devices)) or real(devices))
    arrays = _arrays(SKEWED)
    events = _events()
    function, args = _build('traced', arrays)
    assert _native_targets(function, *args, events)
    assert seen and set(seen) == {()} and real(()) == DEFAULT_L2_BYTES
    expected = np.asarray(events, np.float32).T @ _dense(*arrays)
    np.testing.assert_allclose(jax.jit(function)(*args, events), expected, rtol=1e-5, atol=1e-5)


@requires_gpu
def test_native_mm_under_shard_map_plans_for_the_mesh_devices(monkeypatch):
    from jax.sharding import Mesh, PartitionSpec as P
    from brainevent._csr.binary import binary as binary_mod
    gpus = jax.devices('gpu')
    seen = []
    real = binary_mod.cuda_l2_bytes
    monkeypatch.setattr(binary_mod, 'cuda_l2_bytes', lambda devices: seen.append(tuple(devices)) or real(devices))
    arrays = _arrays(SKEWED)
    events = _events(batch=4 * len(gpus))
    function, _ = _build('traced', arrays)
    mesh = Mesh(np.array(gpus), ('batch',))
    sharded = jax.shard_map(function, mesh=mesh, in_specs=(P(), P(), P(), P(None, 'batch')),
                            out_specs=P('batch', None))
    assert _native_targets(sharded, *arrays, events)
    assert seen and set(seen) == {tuple(gpus)}
    expected = np.asarray(events, np.float32).T @ _dense(*arrays)
    np.testing.assert_allclose(jax.jit(sharded)(*arrays, events), expected, rtol=1e-5, atol=1e-5)


WIDE = 3000  # columns, so one row can hold several 512-edge tasks
LAYOUTS = {
    'no_edges': np.zeros(ROWS, dtype=int),
    'empty_rows': np.r_[np.zeros(ROWS // 2, dtype=int), np.full(ROWS // 2, 9)],
    'one_long_row': np.r_[np.zeros(ROWS - 1, dtype=int), 2100],  # five long-row tasks
    'long_and_short': np.r_[2100, np.zeros(ROWS // 2 - 1, dtype=int), np.full(ROWS // 2, 3)],
}


def _wide_arrays(lengths, seed=0):
    rng = np.random.default_rng(seed)
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    indices = np.concatenate([np.sort(rng.choice(WIDE, n, replace=False)) for n in lengths] + [np.zeros(0, int)])
    weights = rng.normal(size=indices.size).astype(np.float32)
    dense = np.zeros((ROWS, WIDE), np.float32)
    np.add.at(dense, (np.repeat(np.arange(ROWS), lengths), indices), weights)
    return (jnp.asarray(weights), jnp.asarray(indices.astype(np.int32)), jnp.asarray(indptr)), dense


@requires_gpu
@pytest.mark.parametrize('l2', ['device', 'one_batch_per_group'])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('route', ['scatter', 'mirror'])
@pytest.mark.parametrize('layout', list(LAYOUTS))
def test_vmapped_mv_handles_extreme_row_layouts(monkeypatch, layout, route, weighted, l2):
    # vmap of the event MV becomes the hybrid MM. Rows that are empty, all
    # edges in one row, and both at once; one batch per chunk runs many groups.
    from brainevent._csr.binary import binary as binary_mod
    if l2 == 'one_batch_per_group':
        monkeypatch.setattr(binary_mod, 'cuda_l2_bytes', lambda devices: 1)
    arrays, dense = _wide_arrays(LAYOUTS[layout])
    rng = np.random.default_rng(4)
    values = np.where(rng.random((7, ROWS)) < 0.4, rng.normal(size=(7, ROWS)), 0).astype(np.float32)
    wrap = brainevent.WeightedArray if weighted else BinaryArray
    if route == 'scatter':  # events @ W scatters over the stored rows
        def one(w, i, p, v):
            return wrap(v) @ CSR((w, i, p), shape=(ROWS, WIDE))
        operands = arrays
    else:  # W.T @ events gathers unless cuda_raw scatters over its mirror, whose rows are W's rows
        matrix = CSR.fromdense(dense.T, backend='cuda_raw', precompute_weight_indices=True)
        assert matrix.nse == arrays[1].size

        def one(m, v):
            return m @ wrap(v)
        operands = (matrix,)
    function = jax.vmap(one, in_axes=(None,) * len(operands) + (0,))
    out = jax.jit(function)(*operands, jnp.asarray(values if weighted else values != 0))
    events = values if weighted else (values != 0).astype(np.float32)
    np.testing.assert_allclose(out, events @ dense, rtol=1e-4, atol=1e-4)
    if LAYOUTS[layout].sum():
        assert _native_targets(function, *operands, jnp.asarray(values if weighted else values != 0))


@requires_gpu
def test_tall_narrow_output_runs_in_short_row_bounded_chunks():
    # The L2 output rule alone would keep all batches together. Short-list
    # scratch must remain bounded by the cache budget rather than batch*rows.
    from brainevent._op.lowering_target import cuda_l2_bytes
    rows, cols = 20_000, 8
    l2 = cuda_l2_bytes(jax.devices('gpu'))
    allowance = max(rows, 1 << 20, l2 // 8)
    batch = max(1024, allowance // rows + 1)
    rng = np.random.default_rng(5)
    indptr = jnp.arange(rows + 1, dtype=jnp.int32)
    indices = jnp.asarray(rng.integers(0, cols, rows, dtype=np.int32))
    weights = jnp.asarray(rng.normal(size=rows).astype(np.float32))
    plan = CSRConfig().mm_plan(rows, cols, rows, batch, np.dtype(np.float32), False, l2)
    assert 1 < plan['chunk'] < batch
    assert plan['short_rows_size'] <= allowance
    events = jnp.asarray(rng.random((rows, batch)) < 0.1)
    out = jax.jit(lambda w, i, p, e: BinaryArray(e.T) @ CSR((w, i, p), shape=(rows, cols)))(
        weights, indices, indptr, events)
    dense = np.zeros((rows, cols), np.float32)
    np.add.at(dense, (np.arange(rows), np.asarray(indices)), np.asarray(weights))
    np.testing.assert_allclose(out, np.asarray(events, np.float32).T @ dense, rtol=1e-4, atol=1e-4)


# Reusable hybrid-MM structure masks.

def _long_mask_scan_inputs():
    rows, cols, batch, steps = 96, 128, 4, 5
    lengths = np.full(rows, 2, np.int32)
    lengths[0], lengths[1] = 65, 0
    indptr = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    indices = np.concatenate([np.arange(n, dtype=np.int32) for n in lengths])
    weights = np.full(indices.size, 0.125, np.float32)
    events = np.random.default_rng(7).random((steps, batch, rows)) < 0.25
    dense = np.zeros((rows, cols), np.float32)
    np.add.at(dense, (np.repeat(np.arange(rows), lengths), indices), weights)
    return tuple(jnp.asarray(x) for x in (weights, indices, indptr)), jnp.asarray(events), dense


def _long_mask_scan(matrix, events, *, weighted=False, vmapped=False):
    event_type = WeightedArray if weighted else BinaryArray

    def step(total, values):
        # with_data must preserve the structure cache even inside scan.
        current = matrix.with_data(matrix.data * jnp.float32(2))
        if vmapped:
            out = jax.vmap(lambda v: event_type(v) @ current)(values)
        else:
            out = event_type(values) @ current
        return total + jnp.sum(out), None

    return jax.lax.scan(step, jnp.float32(0), events)[0]


@requires_gpu
@pytest.mark.parametrize('weighted,vmapped', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('config', [CSRConfig(), CSRConfig(mm_max_chunk=1)], ids=['default', 'one_batch_chunks'])
def test_cached_matrix_scan_has_no_mask_builder(weighted, vmapped, config):
    arrays, events, dense = _long_mask_scan_inputs()
    values = events.astype(jnp.float32) * 0.5 if weighted else events
    matrix = CSR(arrays, shape=dense.shape, backend='cuda_raw', config=config)
    fn = jax.jit(lambda m, e: _long_mask_scan(m, e, weighted=weighted, vmapped=vmapped))
    compiled = fn.lower(matrix, values).compile()
    hlo = compiled.as_text()
    assert 'csr_hybrid_mm' in hlo
    # Mask construction packs uint32 bits with a shift. Output reductions are
    # float32 and remain valid; the entire cached executable needs no bit pack.
    assert 'shift-left(' not in hlo
    chunks = set(map(int, re.findall(r'chunk = (\d+) : i64', hlo)))
    assert chunks == {min(values.shape[1], config.mm_max_chunk)}
    factors = np.asarray(values, np.float32) if weighted else np.asarray(values != 0, np.float32)
    np.testing.assert_allclose(compiled(matrix, values), np.sum(factors @ (dense * 2)), rtol=1e-5)


@requires_gpu
def test_jit_created_matrix_builds_mask_before_scan():
    arrays, events, dense = _long_mask_scan_inputs()

    def simulate(weights, indices, indptr, values):
        matrix = CSR((weights, indices, indptr), shape=dense.shape, backend='cuda_raw')
        return _long_mask_scan(matrix, values)

    compiled = jax.jit(simulate).lower(*arrays, events).compile()
    hlo = compiled.as_text()
    assert 'csr_hybrid_mm' in hlo
    # Find the computation which packs the bits, and ensure its call is not
    # inside a while body. Do not rely on unstable XLA fusion names.
    computations = {}
    current = None
    for line in hlo.splitlines():
        if (line.startswith('%') or line.startswith('ENTRY ')) and line.endswith('{'):
            current = line.split(' (', 1)[0].removeprefix('ENTRY ')
            computations[current] = []
        elif current is not None:
            computations[current].append(line)
    mask_functions = {name for name, lines in computations.items() if any('shift-left(' in line for line in lines)}
    assert mask_functions
    bodies = set(re.findall(r'body=(%[\w.]+)', hlo))
    assert bodies
    for body in bodies:
        text = '\n'.join(computations[body])
        assert 'shift-left(' not in text
        assert not any(f'calls={name},' in text for name in mask_functions)
    np.testing.assert_allclose(compiled(*arrays, events), np.sum(np.asarray(events, np.float32) @ (dense * 2)), rtol=1e-5)
