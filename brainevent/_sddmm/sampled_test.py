# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

"""Ordinary sampled products retain slot semantics and shard_map precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from jax.sharding import Mesh, PartitionSpec as P

from brainevent._compatible_import import varying_axes
from brainevent._sddmm import sddmm
from brainevent._test_util import jax_x64_enabled


@pytest.mark.parametrize('batch', [0, 1, 3])
@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64])
def test_sampled_values_preserve_order_duplicates_and_dtype(batch, dtype):
    with jax_x64_enabled():
        A = (jnp.arange(4 * batch, dtype=dtype).reshape(4, batch) - 3) / 4
        B = (jnp.arange(batch * 5, dtype=dtype).reshape(batch, 5) - 2) / 8
        indices = jnp.array([[3, 4], [0, 1], [2, 3], [3, 4], [1, 0]], jnp.int32)
        expected = (A @ B)[indices[:, 0], indices[:, 1]]
        for actual in (sddmm(A, B, indices), jax.jit(sddmm)(A, B, indices)):
            assert actual.shape == (indices.shape[0],)
            assert actual.dtype == jnp.dtype(dtype)
            np.testing.assert_array_equal(np.asarray(actual, np.float64),
                                          np.asarray(expected, np.float64))


@pytest.mark.parametrize('batch', [0, 1, 3])
def test_empty_pattern(batch):
    A, B = jnp.ones((4, batch)), jnp.ones((batch, 5))
    actual = jax.jit(sddmm)(A, B, jnp.empty((0, 2), jnp.int32))
    assert actual.shape == (0,)
    assert actual.dtype == A.dtype


def test_sampling_all_coordinates_matches_dense_product():
    A = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 4
    B = jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 8
    rows, cols = jnp.meshgrid(jnp.arange(4), jnp.arange(5), indexing='ij')
    indices = jnp.stack([rows.ravel(), cols.ravel()], axis=1)
    np.testing.assert_array_equal(sddmm(A, B, indices), (A @ B).ravel())


@pytest.mark.parametrize('batch', [1, 3])
def test_both_operand_gradients_match_dense_product(batch):
    A = jnp.arange(4 * batch, dtype=jnp.float32).reshape(4, batch) / 4
    B = jnp.arange(batch * 5, dtype=jnp.float32).reshape(batch, 5) / 8
    indices = jnp.array([[3, 4], [0, 1], [3, 4], [1, 0]], jnp.int32)
    loss = lambda a, b: jnp.sum(sddmm(a, b, indices) ** 2)
    reference = lambda a, b: jnp.sum((a @ b)[indices[:, 0], indices[:, 1]] ** 2)
    actual = jax.jit(jax.grad(loss, argnums=(0, 1)))(A, B)
    expected = jax.grad(reference, argnums=(0, 1))(A, B)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


def _eqns(f, *args):
    """Collect equations, including nested jaxprs, before primitive lowering."""
    eqns = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            eqns.append(eqn)
            for value in eqn.params.values():
                for sub in value if isinstance(value, (tuple, list)) else (value,):
                    sub = getattr(sub, 'jaxpr', sub)
                    if hasattr(sub, 'eqns'):
                        walk(sub)

    walk(jax.make_jaxpr(f)(*args).jaxpr)
    return eqns


def _shard_operands():
    A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) + 1
    B = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 2
    indices = jnp.array([[0, 0], [1, 2], [0, 3]], jnp.int32)
    return A, B, indices


def _run_sharded(a_spec, *, check_vma=True):
    A, B, indices = _shard_operands()
    seen = []

    def local(a, b):
        values = sddmm(a[0] if a_spec == P('s') else a, b[0], indices)
        seen.append(varying_axes(jax.typeof(values)))
        return values[None]

    mesh = Mesh(np.asarray(jax.devices()[:1]), ('s',))
    f = jax.shard_map(local, mesh=mesh, in_specs=(a_spec, P('s')),
                      out_specs=P('s'), check_vma=check_vma)
    args = (A[None] if a_spec == P('s') else A, B[None])
    actual = jax.jit(f)(*args)
    np.testing.assert_array_equal(actual[0], jax.jit(sddmm)(A, B, indices))
    return seen, _eqns(f, *args)


@pytest.mark.parametrize('a_spec', [P('s'), P()])
def test_shard_map_varying_operands_preserve_axes_and_precision(a_spec):
    # #245: JAX's sampled sparse primitive types its result as replicated.
    # Varying operands must instead use gathers and a highest-precision dot.
    seen, eqns = _run_sharded(a_spec)
    assert seen and all(axes == frozenset({'s'}) for axes in seen)
    assert 'bcoo_dot_general_sampled' not in {eqn.primitive.name for eqn in eqns}
    dots = [eqn for eqn in eqns if eqn.primitive.name == 'dot_general']
    assert len(dots) == 1
    highest = jax.lax.Precision.HIGHEST
    assert dots[0].params['precision'] in (highest, (highest, highest))


def test_out_of_range_coordinates_are_zero_with_and_without_sharding():
    A, B, _ = _shard_operands()
    indices = jnp.array([[0, 0], [2, 1], [1, 4]], jnp.int32)
    unsharded = sddmm(A, B, indices)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ('s',))
    f = jax.shard_map(lambda a, b: sddmm(a[0], b, indices)[None], mesh=mesh,
                      in_specs=(P('s'), P()), out_specs=P('s'))
    sharded = jax.jit(f)(A[None], B)[0]
    np.testing.assert_array_equal(np.asarray(unsharded)[1:], [0., 0.])
    np.testing.assert_array_equal(sharded, unsharded)


def test_shard_map_without_vma_keeps_sampled_sparse_primitive():
    seen, eqns = _run_sharded(P('s'), check_vma=False)
    assert seen and all(axes == frozenset() for axes in seen)
    assert 'bcoo_dot_general_sampled' in {eqn.primitive.name for eqn in eqns}


def test_unsharded_values_match_sampled_sparse_primitive_bit_for_bit():
    rng = np.random.default_rng(245)
    A = jnp.asarray(rng.standard_normal((7, 5)), jnp.float32)
    B = jnp.asarray(rng.standard_normal((5, 6)), jnp.float32)
    indices = jnp.asarray(np.stack([rng.integers(0, 7, 20), rng.integers(0, 6, 20)], 1), jnp.int32)
    # Compile both: an eager dot can select a different GPU algorithm.
    reference = jax.jit(lambda *args: sparse.bcoo_dot_general_sampled(
        *args, dimension_numbers=(((1,), (0,)), ((), ())),
    ))
    np.testing.assert_array_equal(jax.jit(sddmm)(A, B, indices), reference(A, B, indices))
    primitives = {eqn.primitive.name for eqn in _eqns(sddmm, A, B, indices)}
    assert 'bcoo_dot_general_sampled' in primitives
    assert 'gather' not in primitives
