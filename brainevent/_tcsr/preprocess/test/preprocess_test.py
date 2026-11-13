# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0

"""Sorting invariants, the recorded permutation and preparation contexts."""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.main import CSR
from brainevent._tcsr.preprocess import ensure_preparation_context, sort_csr, validate_preparation


def test_stable_sort_preserves_source_duplicates_and_units():
    source = CSR(jnp.array([10., 20., 30., 40., 50.]) * u.mV,
                 jnp.array([2, 0, 2, 1, 0]), jnp.array([0, 3, 5]), shape=(2, 3))
    data, indices, permutation = sort_csr(source)
    np.testing.assert_array_equal(indices, [0, 2, 2, 0, 1])
    np.testing.assert_array_equal(u.get_mantissa(data), [20., 10., 30., 50., 40.])
    np.testing.assert_array_equal(permutation, [1, 0, 2, 4, 3])
    assert u.get_unit(data) == u.mV
    np.testing.assert_array_equal(source.indices, [2, 0, 2, 1, 0])
    np.testing.assert_array_equal(u.get_mantissa(source.data), [10., 20., 30., 40., 50.])


@pytest.mark.parametrize('seed', range(4))
def test_vectorized_sort_matches_per_row_stable_argsort(seed):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 6, size=7)  # includes empty rows
    indptr = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    indices = rng.integers(0, 4, size=indptr[-1]).astype(np.int32)  # many duplicates
    source = CSR(jnp.arange(indices.size, dtype=jnp.float32), jnp.asarray(indices),
                 jnp.asarray(indptr), shape=(7, 4))
    data, sorted_indices, permutation = sort_csr(source)
    expected = np.concatenate([b + np.argsort(indices[b:e], kind='stable')
                               for b, e in zip(indptr[:-1], indptr[1:])] or [np.empty(0, int)])
    if permutation is None:
        np.testing.assert_array_equal(expected, np.arange(indices.size))
    else:
        np.testing.assert_array_equal(permutation, expected)
        assert not permutation.flags.writeable
    np.testing.assert_array_equal(sorted_indices, indices[expected])
    np.testing.assert_array_equal(data, np.arange(indices.size)[expected])


def test_sorted_source_returns_its_own_arrays_and_no_permutation():
    source = CSR(jnp.array([1., 2., 3.]), jnp.array([1, 2, 0], jnp.int32),
                 jnp.array([0, 2, 3], jnp.int32), shape=(2, 3))
    data, indices, permutation = sort_csr(source)
    assert permutation is None
    assert data is source.data and indices is source.indices


@pytest.mark.parametrize('data', [jnp.array(2.), jnp.array([2.])])
def test_shared_weight_is_not_expanded(data):
    source = CSR(data, jnp.array([1, 0]), jnp.array([0, 2]), shape=(1, 2))
    sorted_data, indices, permutation = sort_csr(source)
    assert sorted_data is source.data
    np.testing.assert_array_equal(indices, [0, 1])
    np.testing.assert_array_equal(permutation, [1, 0])


def test_sort_empty_structure():
    source = CSR(jnp.empty(0), jnp.empty(0, jnp.int32), jnp.zeros(3, jnp.int32), shape=(2, 0))
    data, indices, permutation = sort_csr(source)
    assert data.shape == indices.shape == (0,)
    assert permutation is None


def test_sort_rejects_jit():
    source = CSR(jnp.array(1.), jnp.array([1, 0]), jnp.array([0, 2]), shape=(1, 2))
    with pytest.raises(ValueError, match='outside JIT'):
        jax.jit(lambda x: (sort_csr(source), x)[1])(jnp.array(1.))


def test_preparation_rejects_traced_structure():
    with pytest.raises(ValueError, match='outside JIT'):
        jax.jit(ensure_preparation_context)(jnp.array([1], jnp.int32))


def test_validate_static_arguments():
    assert validate_preparation(np.int64(3), jnp.float32) == 3
    with pytest.raises(ValueError, match='f16'):
        validate_preparation(3, jnp.int32)
