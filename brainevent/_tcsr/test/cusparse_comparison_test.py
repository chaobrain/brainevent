# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""1024-square heterogeneous TCSR products against true cuSPARSE products.

The public matrix covers inherited CSR routes as well as tiled binary MM.
The separate weighted cases call the native combined direct/mirror APIs.
"""

from functools import partial
import re
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSR, TCSR, WeightedArray
from brainevent._csr.test._test_util import (
    BATCH,
    CUSPARSE_COMPARISON_DTYPE_PAIRS,
    CUSPARSE_COMPARISON_VALUE_DTYPES,
    OFFSET_DTYPES,
    SHAPE,
    assert_comparison,
    cusparse_product,
    dense_reference,
    make_case,
    requires_cuda,
)
from brainevent._csr.test._test_util import comparison_x64  # noqa: F401
from brainevent._tcsr.binary.combined import jax_ffi as direct
from brainevent._tcsr.binary.combined.col_major import jax_ffi as mirror


pytestmark = [requires_cuda, pytest.mark.usefixtures('comparison_x64')]


@partial(jax.jit, static_argnames=('mode', 'transpose'))
def _public_product(matrix, events, *, mode, transpose):
    """Normalize both public product directions to vector or NB output."""
    wrap = WeightedArray if mode == 'weighted' else BinaryArray
    if transpose:
        operand = events.T if events.ndim == 2 else events
        result = wrap(operand) @ matrix
        return result.T if events.ndim == 2 else result
    return matrix @ wrap(events)


@partial(jax.jit, static_argnames=('transpose', 'weighted'))
def _cusparse_reference(
    weights, indices, indptr, events, *, transpose, weighted,
):
    # Half operands use actual FP32 cuSPARSE accumulation after quantization.
    return cusparse_product(
        weights, indices, indptr, events,
        transpose=transpose, weighted=weighted,
    )


@partial(jax.jit, static_argnames=('plan',))
def _native_direct_product(weights, events, *, plan):
    return direct.spmm_weighted(weights, events.T, plan).T


@partial(jax.jit, static_argnames=('plan',))
def _native_mirror_product(weights, events, *, permutation, plan):
    return mirror.spmm_weighted(weights, permutation, events.T, plan).T


def _matrix(case):
    # TCSR may sort its input; the oracle receives these canonical arrays.
    tiled = TCSR(CSR(
        case.weights, case.indices, case.indptr,
        shape=SHAPE, backend='cuda_raw', indptr_dtype=case.indptr.dtype,
    )).build_weight_indices()
    assert type(tiled) is TCSR
    assert tiled.data.shape == (tiled.nse,) and tiled.nse > 1
    assert tiled.data.dtype == case.weights.dtype
    assert tiled.indices.dtype == jnp.dtype('int32')
    assert tiled.indptr.dtype == case.indptr.dtype
    return tiled


def _assert_case_dtypes(case, value_dtype, event_dtype, offset_dtype):
    assert case.weights.dtype == jnp.dtype(value_dtype)
    assert case.indices.dtype == jnp.dtype('int32')
    assert case.indptr.dtype == jnp.dtype(offset_dtype)
    assert (
        case.events_mv.dtype == case.events_mm.dtype == jnp.dtype(event_dtype)
    )


def _canonical_coordinates(matrix):
    rows = np.repeat(np.arange(SHAPE[0]), np.diff(np.asarray(matrix.indptr)))
    return rows, np.asarray(matrix.indices)


def _assert_dense_references(
    actual, expected, weights, events, rows, cols, *,
    transpose, weighted, value_dtype, label, record_property,
):
    case = SimpleNamespace(weights=weights, rows=rows, cols=cols)
    oracle = dense_reference(
        case, events, transpose=transpose, weighted=weighted,
    )
    for source, result in [('cuda', actual), ('cusparse', expected)]:
        assert_comparison(
            result, oracle, value_dtype,
            label=f'{label}_{source}_vs_dense',
            record_property=record_property,
        )


def _optimized_text(function, *args, **kwargs):
    return function.lower(*args, **kwargs).compile().as_text()


def _assert_cusparse(text):
    assert re.search(r'custom_call_target="[^"]*cusparse[^"]*"', text), text


def _assert_inherited_cuda(text, rank):
    if rank == 'mv':
        family = r'(?:csr_hybrid_spmv_|csr_binary_(?:indexed_)?csrmv\.)'
    else:
        family = r'(?:csr_hybrid_mm_|csr_binary_(?:indexed_)?csrmm\.)'
    assert re.search(
        r'custom_call_target="[^"]*' + family + r'[^"]*"', text,
    ), text


@pytest.mark.parametrize('rank', ['mv', 'mm'])
@pytest.mark.parametrize('transpose', [False, True], ids=['right', 'left'])
@pytest.mark.parametrize('mode', ['binary', 'weighted'])
@pytest.mark.parametrize(
    ('event_dtype', 'value_dtype'), CUSPARSE_COMPARISON_DTYPE_PAIRS,
)
@pytest.mark.parametrize('offset_dtype', OFFSET_DTYPES)
def test_public_tcsr_matches_cusparse(
    rank, transpose, mode, value_dtype, event_dtype, offset_dtype,
    record_property,
):
    """Compare all 832 public dtype/rank/direction combinations elementwise."""
    case = make_case(value_dtype, event_dtype, offset_dtype)
    _assert_case_dtypes(case, value_dtype, event_dtype, offset_dtype)
    tiled = _matrix(case)
    rows, cols = _canonical_coordinates(tiled)
    events = case.events_mv if rank == 'mv' else case.events_mm
    assert events.shape == ((SHAPE[0],) if rank == 'mv' else (SHAPE[0], BATCH))
    options = dict(mode=mode, transpose=transpose)

    text = _optimized_text(_public_product, tiled, events, **options)
    tiled_call = re.search(
        r'custom_call_target="[^"]*tcsr_combined_[^"]*"', text,
    )
    native_expected = (
        rank == 'mm' and mode == 'binary'
        and event_dtype in ('bool', 'float32')
    )
    assert bool(tiled_call) == native_expected, text
    if not native_expected:
        _assert_inherited_cuda(text, rank)
    _assert_cusparse(_optimized_text(
        _cusparse_reference, tiled.data, tiled.indices, tiled.indptr, events,
        transpose=transpose, weighted=mode == 'weighted',
    ))

    # Reuse scratch with changed inputs, then a fully silent call.
    steps = (
        ('initial', tiled.data, events),
        ('updated', jnp.roll(tiled.data, 1), jnp.roll(events, 1, axis=0)),
        ('silent', tiled.data, jnp.zeros_like(events)),
    )
    for label, weights, current_events in steps:
        actual = _public_product(
            tiled.with_data(weights), current_events, **options,
        )
        expected = _cusparse_reference(
            weights, tiled.indices, tiled.indptr, current_events,
            transpose=transpose, weighted=mode == 'weighted',
        )
        assert actual.dtype == expected.dtype == jnp.dtype(value_dtype)
        assert_comparison(
            actual, expected, value_dtype,
            label=f'tcsr_public_{label}', record_property=record_property,
        )
        if label != 'silent':
            _assert_dense_references(
                actual, expected, weights, current_events, rows, cols,
                transpose=transpose, weighted=mode == 'weighted',
                value_dtype=value_dtype,
                label=f'tcsr_public_{label}', record_property=record_property,
            )


@pytest.mark.parametrize('direction', ['direct', 'mirror'])
@pytest.mark.parametrize('value_dtype', CUSPARSE_COMPARISON_VALUE_DTYPES)
@pytest.mark.parametrize('offset_dtype', OFFSET_DTYPES)
def test_native_weighted_tcsr_matches_cusparse(
    direction, value_dtype, offset_dtype, record_property,
):
    """Compare 16 matching-dtype native weighted MM cases elementwise."""
    case = make_case(value_dtype, value_dtype, offset_dtype)
    _assert_case_dtypes(case, value_dtype, value_dtype, offset_dtype)
    tiled = _matrix(case)
    rows, cols = _canonical_coordinates(tiled)
    events = case.events_mm
    assert events.dtype == tiled.data.dtype
    transpose = direction == 'direct'

    if transpose:
        plan = direct.prepare(
            np.asarray(tiled.indptr), np.asarray(tiled.indices),
            SHAPE[1], BATCH, dtype=value_dtype,
        )
        run = partial(_native_direct_product, plan=plan)
        native_pattern = 'combined_weighted_'
        text = _optimized_text(
            _native_direct_product, tiled.data, events, plan=plan,
        )
    else:
        pointers, indices, permutation = tiled.buffers['csc']
        # Mirrors choose offsets from nnz independently of canonical pointers.
        # Bind both FFI offset operands to this case's requested dtype.
        pointers = np.asarray(pointers, dtype=offset_dtype)
        permutation = jnp.asarray(permutation, dtype=offset_dtype)
        assert pointers.dtype == np.dtype(offset_dtype)
        assert permutation.dtype == jnp.dtype(offset_dtype)
        plan = mirror.prepare(
            pointers, np.asarray(indices),
            SHAPE[0], BATCH, dtype=value_dtype,
        )
        run = partial(
            _native_mirror_product, permutation=permutation, plan=plan,
        )
        native_pattern = 'combined_col_major_weighted_'
        text = _optimized_text(
            _native_mirror_product, tiled.data, events,
            permutation=permutation, plan=plan,
        )

    assert plan.value_dtype == tiled.data.dtype
    assert plan.tile_offsets.dtype == np.dtype(offset_dtype)
    # Calling the native FFI directly prevents weighted public CSR fallback.
    assert re.search(
        r'custom_call_target="[^"]*' + native_pattern + r'[^"]*"', text,
    ), text
    _assert_cusparse(_optimized_text(
        _cusparse_reference, tiled.data, tiled.indices, tiled.indptr, events,
        weighted=True, transpose=transpose,
    ))
    steps = (
        ('initial', tiled.data, events),
        ('updated', jnp.roll(tiled.data, 1), jnp.roll(events, 1, axis=0)),
        ('silent', tiled.data, jnp.zeros_like(events)),
    )
    for label, weights, current_events in steps:
        actual = run(weights, current_events)
        expected = _cusparse_reference(
            weights, tiled.indices, tiled.indptr, current_events,
            weighted=True, transpose=transpose,
        )
        assert actual.dtype == expected.dtype == jnp.dtype(value_dtype)
        assert_comparison(
            actual, expected, value_dtype,
            label=f'tcsr_native_weighted_{label}',
            record_property=record_property,
        )
        if label != 'silent':
            _assert_dense_references(
                actual, expected, weights, current_events, rows, cols,
                transpose=transpose, weighted=True, value_dtype=value_dtype,
                label=f'tcsr_native_weighted_{label}',
                record_property=record_property,
            )
