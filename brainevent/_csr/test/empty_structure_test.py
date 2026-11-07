"""Zero-dimensional CSR structures remain valid through their derived layouts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSR, CSRConfig, WeightedArray
from brainevent._misc import csc_to_csr_index, csr_to_csc_index


EMPTY_SHAPES = [(0, 0), (0, 3), (3, 0)]


def _empty_csr(shape, *, backend='jax_raw', homogeneous=False):
    return CSR(
        jnp.ones((1 if homogeneous else 0,), jnp.float32),
        jnp.empty((0,), jnp.int32),
        jnp.zeros((shape[0] + 1,), jnp.int32),
        shape=shape, backend=backend, config=CSRConfig(),
    )


@pytest.mark.parametrize('shape', EMPTY_SHAPES)
@pytest.mark.parametrize('method', ['coo', 'numpy', 'gpu_column_block'])
@pytest.mark.parametrize('array_module', [np, jnp])
@pytest.mark.parametrize('include_perm', [False, True])
def test_empty_mirror_helpers(shape, method, array_module, include_perm):
    indptr = array_module.zeros((shape[0] + 1,), dtype=np.int32)
    indices = array_module.empty((0,), dtype=np.int32)
    csc_indptr, csc_indices, perm = csr_to_csc_index(
        indptr, indices, shape=shape, method=method, include_perm=include_perm,
    )
    np.testing.assert_array_equal(csc_indptr, np.zeros(shape[1] + 1, np.int32))
    assert csc_indices.shape == (0,) and csc_indices.dtype == np.int32
    assert csc_indptr.dtype == np.int32
    if include_perm:
        assert perm.shape == (0,) and perm.dtype == np.int32
    else:
        assert perm is None
    restored_indptr, restored_indices, restored_perm = csc_to_csr_index(
        csc_indptr, csc_indices, shape=shape, include_perm=include_perm,
    )
    np.testing.assert_array_equal(restored_indptr, indptr)
    np.testing.assert_array_equal(restored_indices, indices)
    assert (restored_perm is None) == (not include_perm)


@pytest.mark.parametrize('shape', EMPTY_SHAPES)
def test_empty_csr_builds_mirror_eagerly_and_under_jit(shape):
    matrix = _empty_csr(shape)
    # Exercise the traced helper before the eager call can populate the cache.
    traced = jax.jit(lambda value: value.build_weight_indices())(matrix)
    eager = matrix.build_weight_indices()
    for prepared in (eager, traced):
        indptr, indices, perm = prepared.buffers['csc']
        np.testing.assert_array_equal(indptr, np.zeros(shape[1] + 1, np.int32))
        assert indices.shape == perm.shape == (0,)
        np.testing.assert_array_equal(prepared.tocsc().tocsr().todense(), np.zeros(shape))


@pytest.mark.parametrize('shape', EMPTY_SHAPES)
def test_empty_csr_diagonal_addition_eagerly_and_under_jit(shape):
    matrix = _empty_csr(shape)
    diagonal = jnp.empty((0,), jnp.float32)
    eager = matrix.diag_add(diagonal)
    prepared = matrix.prepare_diag_add()
    traced = jax.jit(lambda value, diag: value.diag_add(diag))(prepared, diagonal)
    for result in (eager, traced):
        assert result.shape == shape and result.nse == 0
        np.testing.assert_array_equal(result.indptr, np.zeros(shape[0] + 1, np.int32))
        np.testing.assert_array_equal(result.todense(), np.zeros(shape))


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='native CUDA dispatch requires a GPU')
@pytest.mark.parametrize('shape', EMPTY_SHAPES)
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('homogeneous', [False, True])
def test_empty_csr_cuda_event_products(shape, weighted, homogeneous):
    matrix = _empty_csr(shape, backend='cuda_raw', homogeneous=homogeneous)
    event_type = WeightedArray if weighted else BinaryArray
    dtype = jnp.float32 if weighted else jnp.bool_
    for batch in (None, 2):
        right_shape = (shape[1],) if batch is None else (shape[1], batch)
        left_shape = (shape[0],) if batch is None else (batch, shape[0])
        right = event_type(jnp.ones(right_shape, dtype))
        left = event_type(jnp.ones(left_shape, dtype))
        right_result_shape = (shape[0],) if batch is None else (shape[0], batch)
        left_result_shape = (shape[1],) if batch is None else (batch, shape[1])
        # The eager unfavorable product must build the empty mirror itself.
        np.testing.assert_array_equal(matrix @ right, np.zeros(right_result_shape))
        np.testing.assert_array_equal(left @ matrix, np.zeros(left_result_shape))
        np.testing.assert_array_equal(
            jax.jit(lambda value, events: value @ events)(matrix, right),
            np.zeros(right_result_shape),
        )
        np.testing.assert_array_equal(
            jax.jit(lambda value, events: events @ value)(matrix, left),
            np.zeros(left_result_shape),
        )
