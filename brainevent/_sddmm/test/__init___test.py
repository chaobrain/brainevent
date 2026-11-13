# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``brainevent/_sddmm/__init__.py``, grouped by behavior."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._registry import get_registry
from brainevent._sddmm import csr_sddmm_dweight


# Public CSR adapters and registry calls
# --------------------------------------

_facade_mv = importlib.import_module('brainevent._sddmm._mv.sddmv')


_facade_mm = importlib.import_module('brainevent._sddmm._mm.sddmm')


@pytest.mark.parametrize('name,batch', [
    ('csr_sddmv', None), ('csr_sddmm', None),
    ('csr_sddmm', 0), ('csr_sddmm', 1), ('csr_sddmm', 3),
])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_registry_calls_adapt_csr_operands_and_select_leaf(monkeypatch, name, batch, transpose, weighted):
    calls = []
    for leaf, module in [('mv', _facade_mv), ('mm', _facade_mm)]:
        original = module._jax_kernel

        def record(*args, _leaf=leaf, _fn=original, **kwargs):
            calls.append(_leaf)
            return _fn(*args, **kwargs)

        monkeypatch.setattr(module, '_jax_kernel', record)
    indices = jnp.array([2, 0, 1, 1], jnp.int32)
    indptr = jnp.array([0, 2, 2, 4], jnp.int32)
    rows = jnp.array([0, 0, 2, 2], jnp.int32)
    tail = () if batch is None else (batch,)
    events = jnp.asarray(np.resize([0., -2., 1., 3.], (3 if transpose else 4,) + tail))
    ct = jnp.asarray(np.resize([2., 1., -3., .5], (4 if transpose else 3,) + tail))
    unit, out = (rows, indices) if transpose else (indices, rows)
    terms = events[unit] * ct[out] if weighted else ct[out]
    expected = jnp.where(events[unit] != 0, terms, 0.)
    if batch is not None:
        expected = expected.sum(axis=1)
    primitive = get_registry()[name]
    fn = jax.jit(lambda e, c: primitive.call(
        e, c, indices, indptr, transpose=transpose, weighted=weighted, backend='jax_raw'))
    np.testing.assert_array_equal(fn(events, ct), expected)
    assert calls == ['mv' if batch in (None, 1) else 'mm']


def test_csr_wrappers_reject_ambiguous_ranks_and_flags():
    e, c = jnp.ones(2), jnp.ones(3)
    i, p = jnp.array([0, 1], jnp.int32), jnp.array([0, 1, 2], jnp.int32)
    with pytest.raises(TypeError, match='bool'):
        csr_sddmm_dweight(e, c, i, p, transpose=True, weighted=1)
    with pytest.raises(ValueError, match='both'):
        csr_sddmm_dweight(e[:, None], c, i, p, transpose=True)
    with pytest.raises(ValueError, match='rank 1'):
        get_registry()['csr_sddmv'].call(e[:, None], c[:, None], i, p, transpose=True)
