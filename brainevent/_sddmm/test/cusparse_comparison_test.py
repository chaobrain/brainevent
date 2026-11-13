# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Elementwise native CUDA/cuSPARSE comparisons for sampled edge gradients.

The 416 cases cover both sampled leaves, binary/weighted event semantics,
four cotangent dtypes, thirteen event dtypes and both CSR offset widths.
Native sampled CUDA supports CSR event products with ``transpose=True``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.test._test_util import (
    EVENT_DTYPES,
    OFFSET_DTYPES,
    VALUE_DTYPES,
    assert_comparison,
    make_case,
    requires_cuda,
)
from brainevent._csr.test._test_util import comparison_x64  # noqa: F401
from brainevent._sddmm import csr_sddmm_dweight, csr_sddmv_dweight


pytestmark = [requires_cuda, pytest.mark.usefixtures('comparison_x64')]

@partial(jax.jit, static_argnames=('weighted', 'backend'))
def _sampled(events, ct, indices, indptr, *, weighted, backend):
    function = csr_sddmv_dweight if events.ndim == 1 else csr_sddmm_dweight
    return function(events, ct, indices, indptr, transpose=True,
                    weighted=weighted, backend=backend)


def _reference(events, ct, rows, cols, *, weighted, value_dtype):
    # Weighted native paths decide activity after conversion to the ct dtype.
    # The FP64 reference then works independently of either sampled backend.
    events = np.asarray(events)
    factors = events.astype(np.dtype(value_dtype)) if weighted else events
    active = factors != 0
    ct = np.asarray(ct, dtype=np.float64)
    if events.ndim == 1:
        factors, active, ct = factors[:, None], active[:, None], ct[:, None]
    values = ct[cols]
    if weighted:
        values = factors[rows].astype(np.float64) * values
    result = np.where(active[rows], values, 0.).sum(axis=1, dtype=np.float64)
    # Outputs have ct dtype: compare against the independently computed sum
    # rounded to that storage type, including the batch-one exact checks.
    return result.astype(np.dtype(value_dtype))


@pytest.mark.parametrize('batch', [1, 4], ids=['sddmv', 'sddmm'])
@pytest.mark.parametrize('mode', ['binary', 'weighted'])
@pytest.mark.parametrize('value_dtype', VALUE_DTYPES)
@pytest.mark.parametrize('event_dtype', EVENT_DTYPES)
@pytest.mark.parametrize('offset_dtype', OFFSET_DTYPES)
def test_native_sampled_products_match_cusparse(
    batch, mode, value_dtype, event_dtype, offset_dtype, record_property,
):
    case = make_case(value_dtype, event_dtype, offset_dtype)
    events = case.events_mv if batch == 1 else case.events_mm
    ct = case.ct_mv if batch == 1 else case.ct_mm
    assert events.dtype == jnp.dtype(event_dtype)
    assert ct.dtype == jnp.dtype(value_dtype)
    assert case.indices.dtype == jnp.int32
    assert case.indptr.dtype == jnp.dtype(offset_dtype)
    # Event cuSPARSE falls back to JAX selection for nonfinite cotangents.
    # Finite inputs keep this comparison on two distinct native backends.
    assert np.all(np.isfinite(np.asarray(ct, dtype=np.float64)))
    assert np.all(np.isfinite(np.asarray(events, dtype=np.float64)))
    weighted = mode == 'weighted'
    kwargs = dict(weighted=weighted)
    operands = (events, ct, case.indices, case.indptr)
    native = _sampled(*operands, backend='cuda_raw', **kwargs)
    cusparse = _sampled(*operands, backend='cusparse', **kwargs)
    assert native.dtype == ct.dtype
    assert cusparse.dtype == ct.dtype
    assert native.shape == cusparse.shape == case.indices.shape
    expected = _reference(events, ct, case.rows, case.cols,
                          weighted=weighted, value_dtype=value_dtype)

    native_ir = _sampled.lower(
        *operands, backend='cuda_raw', **kwargs).as_text()
    cusparse_ir = _sampled.lower(
        *operands, backend='cusparse', **kwargs).as_text()
    target = 'hybrid_sddmv_' if batch == 1 else 'batched_v3_sddmm_'
    assert target in native_ir
    assert 'cusparse_sddmm_' not in native_ir
    assert 'cusparse_sddmm_' in cusparse_ir
    assert target not in cusparse_ir

    for label, actual, reference in (
        ('cuda_vs_cusparse', native, cusparse),
        ('cuda_vs_fp64', native, expected),
        ('cusparse_vs_fp64', cusparse, expected),
    ):
        assert_comparison(actual, reference, value_dtype, label=label,
                          record_property=record_property, sampled_batch=batch)

    # Reuse the compiled call with inactive events: stale scratch or dW must
    # not leak values from the preceding nonzero invocation.
    zeros = jnp.zeros_like(events)
    for backend in ('cuda_raw', 'cusparse'):
        cleared = _sampled(zeros, ct, case.indices, case.indptr,
                           backend=backend, **kwargs)
        np.testing.assert_array_equal(
            np.asarray(cleared), np.zeros(case.indices.shape))
