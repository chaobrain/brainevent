"""Float CSR weight gradients retain slot semantics through sampled dispatch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._csr.float.float import csrmm, csrmv
from brainevent._test_util import jax_x64_enabled


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('batch', [None, 1, 3])
def test_shared_weight_gradient_multiplies_before_reduction(transpose, batch):
    indices = jnp.array([0, 0] if transpose else [0, 1], jnp.int32)
    indptr = jnp.array([0, 1, 2] if transpose else [0, 2], jnp.int32)
    shape = (2, 1) if transpose else (1, 2)
    event_shape = (2,) + (() if batch is None else (batch,))
    ct_shape = (1,) + (() if batch is None else (batch,))
    events = jnp.full(event_shape, 60000, jnp.float16)
    ct = jnp.full(ct_shape, 1e-4, jnp.float16)
    weights = jnp.array([1e-4], jnp.float16)
    operator = csrmv if batch is None else csrmm

    def forward(w):
        return operator(w, indices, indptr, events, shape=shape,
                        transpose=transpose, backend='jax_raw')

    actual = jax.jit(lambda w, c: jax.vjp(forward, w)[1](c)[0])(weights, ct)
    expected = jnp.sum(events * ct).reshape(1)
    assert np.isfinite(forward(weights)).all()
    assert np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('offset_dtype', [jnp.int32, jnp.int64])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('batch', [None, 1, 3])
@pytest.mark.parametrize('shared', [False, True])
def test_float_weight_gradient_preserves_structure_dtypes(offset_dtype, transpose, batch, shared):
    with jax_x64_enabled():
        indices = jnp.array([2, 0, 1, 1], jnp.int32)
        indptr = jnp.array([0, 2, 2, 4], offset_dtype)
        rows = jnp.array([0, 0, 2, 2], jnp.int32)
        event_shape = (3 if transpose else 4,) + (() if batch is None else (batch,))
        ct_shape = (4 if transpose else 3,) + (() if batch is None else (batch,))
        events = jnp.arange(np.prod(event_shape), dtype=jnp.float32).reshape(event_shape) + 1
        ct = jnp.arange(np.prod(ct_shape), dtype=jnp.float32).reshape(ct_shape) + 2
        weights = jnp.ones(1 if shared else indices.size, jnp.float32)
        operator = csrmv if batch is None else csrmm

        def forward(w):
            return operator(w, indices, indptr, events, shape=(3, 4),
                            transpose=transpose, backend='jax_raw')

        actual = jax.jit(lambda w, c: jax.vjp(forward, w)[1](c)[0])(weights, ct)
        unit, out = (rows, indices) if transpose else (indices, rows)
        expected = events[unit] * ct[out]
        if batch is not None:
            expected = jnp.sum(expected, axis=1)
        if shared:
            expected = jnp.sum(expected).reshape(1)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('batch', [None, 1, 3])
def test_shared_float_weight_gradient_remains_twice_differentiable(transpose, batch):
    indices = jnp.array([1, 0, 1], jnp.int32)
    indptr = jnp.array([0, 1, 3], jnp.int32)
    event_shape = (2,) + (() if batch is None else (batch,))
    events = jnp.arange(np.prod(event_shape), dtype=jnp.float32).reshape(event_shape) + 1
    weights = jnp.array([2.], jnp.float32)
    direction = jnp.array([.5], jnp.float32)
    operator = csrmv if batch is None else csrmm

    def forward(w):
        return operator(w, indices, indptr, events, shape=(2, 2),
                        transpose=transpose, backend='jax_raw')

    loss = lambda w: jnp.sum(forward(w) ** 2)
    actual = jax.jit(lambda w: jax.jvp(jax.grad(loss), (w,), (direction,))[1])(weights)
    expected = 2 * jnp.sum(forward(jnp.ones_like(weights)) ** 2) * direction
    np.testing.assert_allclose(actual, expected)
