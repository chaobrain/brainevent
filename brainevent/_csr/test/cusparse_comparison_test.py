# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""GPU-only 1024-square heterogeneous CSR comparisons with cuSPARSE.

The direct product covers both storage directions (832 cases). The indexed
product builds the actual CSC mirror and scatters through it (416 cases),
then compares with the original CSR gather, preserving canonical weight order.
The shared cuSPARSE reference quantizes events to the original weight dtype;
half-precision inputs use FP32 library accumulation and are rounded back.
"""

from functools import lru_cache, partial
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainevent as be
from brainevent._csr.test._test_util import (
    EVENT_DTYPES,
    OFFSET_DTYPES,
    SHAPE,
    VALUE_DTYPES,
    assert_comparison,
    cusparse_product,
    dense_reference,
    make_case,
    requires_cuda,
)
from brainevent._csr.test._test_util import comparison_x64  # noqa: F401
from brainevent._csr.config import CSRConfig


pytestmark = [requires_cuda, pytest.mark.usefixtures('comparison_x64')]

# Half values accumulate in float32 inside every native kernel and are rounded
# once, so they share the float32-accumulating cuSPARSE oracle.
_COMPARISON_VALUE_DTYPES = VALUE_DTYPES


# The hybrid MM is planned from shapes at lowering, so one structure-free
# config serves the matrix and its mirror. It is passed as a (leafless) pytree
# argument of the jitted products.
_CONFIG = CSRConfig()


@lru_cache(maxsize=2)
def _mirror(offset_dtype):
    case = make_case('float32', 'float32', offset_dtype)
    matrix = be.CSR(
        (case.weights, case.indices, case.indptr), shape=SHAPE,
        backend='cuda_raw', config=_CONFIG,
        indptr_dtype=case.indptr.dtype,
    ).build_weight_indices()
    indptr, indices, perm = matrix._mirror()
    permutation = np.asarray(perm)
    mirror_rows = np.repeat(np.arange(SHAPE[1]), np.diff(np.asarray(indptr)))
    # A synthetic permutation would miss errors in actual mirror preparation.
    np.testing.assert_array_equal(case.rows[permutation], np.asarray(indices))
    np.testing.assert_array_equal(case.cols[permutation], mirror_rows)
    assert not np.array_equal(permutation, np.arange(permutation.size))
    assert indices.dtype == jnp.int32
    # Mirror builders choose offsets from nnz; explicitly retain each tested
    # offset ABI for these normalized mirror arrays.
    indptr = indptr.astype(case.indptr.dtype)
    perm = perm.astype(case.indptr.dtype)
    assert indptr.dtype == perm.dtype == case.indptr.dtype
    return indptr, indices, perm


@partial(
    jax.jit, static_argnames=('matrix', 'transpose', 'weighted', 'backend'),
)
def _direct(
    weights, indices, indptr, config, events, *, matrix, transpose, weighted,
    backend,
):
    operation = be.binary_csrmm if matrix else be.binary_csrmv
    return operation(
        weights, indices, indptr, events, shape=SHAPE,
        transpose=transpose, weighted=weighted, backend=backend, config=config,
    )


@partial(jax.jit, static_argnames=('matrix', 'weighted'))
def _indexed(
    weights, indices, indptr, perm, config, events, *, matrix, weighted,
):
    operation = be.binary_csrmm_indexed if matrix else be.binary_csrmv_indexed
    return operation(
        weights, indices, indptr, perm, events, shape=SHAPE[::-1],
        transpose=True, weighted=weighted, backend='cuda_raw', config=config,
    )


def _targets(lowered):
    hlo = lowered.compiler_ir('hlo').as_hlo_text()
    return re.findall(r'custom_call_target="([^"]+)"', hlo)


def _assert_native_route(
    lowered, *, matrix, transpose, weighted, indexed=False,
):
    targets = _targets(lowered)
    mode = 'weighted' if weighted else 'binary'
    if indexed:
        module = (
            f'csr_hybrid_mm_col_major_{mode}_' if matrix
            else f'csr_hybrid_spmv_col_major_{mode}'
        )
        symbol = (
            f'batched_spmm_v3_col_major_{mode}_hetero' if matrix
            else f'hybrid_spmv_col_major_{mode}_hetero'
        )
    elif transpose:
        module = (
            f'csr_hybrid_mm_row_{mode}_' if matrix
            else f'csr_hybrid_spmv_{mode}'
        )
        symbol = (
            'batched_spmm_v3' if matrix else 'binary_csrmv_wat_hybrid_hetero'
        )
    else:
        operation = 'csrmm' if matrix else 'csrmv'
        module = f'csr_binary_{operation}'
        symbol = f'binary_{operation}_nt_auto_hetero'
    assert any(
        module in target and symbol in target for target in targets
    ), targets
    assert not any('cusparse' in target.lower() for target in targets), targets


def _assert_cusparse_route(lowered, *, matrix):
    targets = _targets(lowered)
    operations = ('csr_matmat', 'spmm') if matrix else ('csr_matvec', 'spmv')
    assert any(
        'cusparse' in target.lower()
        and any(op in target.lower() for op in operations)
        for target in targets
    ), targets


def _compare_repeated_calls(
    cuda, reference, cuda_args, reference_args, case, events, *, transpose,
    weighted, record_property,
):
    expected = dense_reference(
        case, events, transpose=transpose, weighted=weighted,
    )
    silent_cuda = (*cuda_args[:-1], jnp.zeros_like(events))
    silent_reference = (*reference_args[:-1], jnp.zeros_like(events))
    # Reuse the compiled executables while output and scratch allocations are
    # exercised across active, silent and active inputs.
    phases = (
        ('active', cuda_args, reference_args, expected),
        ('silent', silent_cuda, silent_reference, np.zeros_like(expected)),
        ('reactivated', cuda_args, reference_args, expected),
    )
    for phase, actual_args, library_args, dense in phases:
        actual, library = cuda(*actual_args), reference(*library_args)
        jax.block_until_ready((actual, library))
        assert actual.shape == library.shape == dense.shape
        assert actual.dtype == library.dtype == case.weights.dtype
        assert_comparison(
            actual, library, case.weights.dtype,
            label=f'{phase}_cuda_cusparse', record_property=record_property,
        )
        assert_comparison(
            actual, dense, case.weights.dtype,
            label=f'{phase}_cuda_dense', record_property=record_property,
        )
        assert_comparison(
            library, dense, case.weights.dtype,
            label=f'{phase}_cusparse_dense', record_property=record_property,
        )


@pytest.mark.parametrize('matrix', [False, True], ids=['mv', 'mm_batch4'])
@pytest.mark.parametrize('transpose', [False, True], ids=['gather', 'scatter'])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('value_dtype', _COMPARISON_VALUE_DTYPES)
@pytest.mark.parametrize('event_dtype', EVENT_DTYPES)
@pytest.mark.parametrize('offset_dtype', OFFSET_DTYPES)
def test_direct_csr_1k_matches_cusparse(
    matrix, transpose, weighted, value_dtype, event_dtype, offset_dtype,
    record_property,
):
    case = make_case(value_dtype, event_dtype, offset_dtype)
    events = case.events_mm if matrix else case.events_mv
    # Events remain the final runtime operand so the same executables can be
    # reused with a silent input without rebuilding the structure.
    args = (case.weights, case.indices, case.indptr, _CONFIG, events)
    reference_args = (case.weights, case.indices, case.indptr, events)
    cuda_lowered = _direct.lower(
        *args, matrix=matrix, transpose=transpose, weighted=weighted,
        backend='cuda_raw',
    )
    reference_lowered = cusparse_product.lower(
        *reference_args, transpose=transpose, weighted=weighted)
    _assert_native_route(
        cuda_lowered, matrix=matrix, transpose=transpose, weighted=weighted,
    )
    _assert_cusparse_route(reference_lowered, matrix=matrix)
    _compare_repeated_calls(
        cuda_lowered.compile(), reference_lowered.compile(), args,
        reference_args, case, events,
        transpose=transpose, weighted=weighted,
        record_property=record_property,
    )


@pytest.mark.parametrize('matrix', [False, True], ids=['mv', 'mm_batch4'])
@pytest.mark.parametrize('weighted', [False, True], ids=['binary', 'weighted'])
@pytest.mark.parametrize('value_dtype', _COMPARISON_VALUE_DTYPES)
@pytest.mark.parametrize('event_dtype', EVENT_DTYPES)
@pytest.mark.parametrize('offset_dtype', OFFSET_DTYPES)
def test_indexed_mirror_1k_matches_original_csr_cusparse(
    matrix, weighted, value_dtype, event_dtype, offset_dtype, record_property,
):
    case = make_case(value_dtype, event_dtype, offset_dtype)
    events = case.events_mm if matrix else case.events_mv
    mirror_indptr, mirror_indices, perm = _mirror(offset_dtype)
    cuda_args = (
        case.weights, mirror_indices, mirror_indptr, perm, _CONFIG, events,
    )
    reference_args = (case.weights, case.indices, case.indptr, events)
    cuda_lowered = _indexed.lower(*cuda_args, matrix=matrix, weighted=weighted)
    reference_lowered = cusparse_product.lower(
        *reference_args, transpose=False, weighted=weighted)
    _assert_native_route(
        cuda_lowered, matrix=matrix, transpose=True, weighted=weighted,
        indexed=True,
    )
    _assert_cusparse_route(reference_lowered, matrix=matrix)
    _compare_repeated_calls(
        cuda_lowered.compile(), reference_lowered.compile(), cuda_args,
        reference_args, case, events, transpose=False, weighted=weighted,
        record_property=record_property,
    )
