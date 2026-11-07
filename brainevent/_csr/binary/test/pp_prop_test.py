"""CSR pp-prop weight cotangents keep event values while forward stays binary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent import BinaryArray, CSR, WeightedArray
import brainevent._csr.main as csr_main


_ROWS = np.array([0, 0, 1, 1])
_COLS = np.array([0, 2, 1, 2])
_DENSE = np.array([[1., 0., 2.], [0., 3., 4.]], dtype=np.float32)


def _matrix(algorithm, homogeneous):
    data = jnp.array([1.] if homogeneous else [1., 2., 3., 4.], dtype=jnp.float32)
    return CSR(
        data,
        jnp.array(_COLS, dtype=jnp.int32),
        jnp.array([0, 2, 4], dtype=jnp.int32),
        shape=(2, 3), backend="jax_raw", backward_algorithm=algorithm,
    )


@pytest.mark.parametrize("algorithm", ["bptt", "pp_prop"])
@pytest.mark.parametrize("homogeneous", [False, True])
@pytest.mark.parametrize("op", ["right_mv", "left_mv", "right_mm", "left_mm"])
def test_binary_forward_and_weight_cotangent(algorithm, homogeneous, op):
    matrix = _matrix(algorithm, homogeneous)
    if op == "right_mv":
        x = jnp.array([2., 0., -1.], dtype=jnp.float32)
        ct = jnp.array([5., 7.], dtype=jnp.float32)
        product = lambda m: m @ BinaryArray(x)
        factor = np.asarray(x)[_COLS]
        expected = _DENSE @ (np.asarray(x) != 0).astype(np.float32)
    elif op == "left_mv":
        x = jnp.array([2., -1.], dtype=jnp.float32)
        ct = jnp.array([5., 6., 7.], dtype=jnp.float32)
        product = lambda m: BinaryArray(x) @ m
        factor = np.asarray(x)[_ROWS]
        expected = (np.asarray(x) != 0).astype(np.float32) @ _DENSE
    elif op == "right_mm":
        x = jnp.array([[2., 0.], [0., 3.], [-1., 2.]], dtype=jnp.float32)
        ct = jnp.array([[5., 6.], [7., 8.]], dtype=jnp.float32)
        product = lambda m: m @ BinaryArray(x)
        factor = np.asarray(x)[_COLS]
        expected = _DENSE @ (np.asarray(x) != 0).astype(np.float32)
    else:
        x = jnp.array([[2., -1.], [0., 3.]], dtype=jnp.float32)
        ct = jnp.array([[5., 6., 7.], [8., 9., 10.]], dtype=jnp.float32)
        product = lambda m: BinaryArray(x) @ m
        factor = np.asarray(x)[:, _ROWS]
        expected = (np.asarray(x) != 0).astype(np.float32) @ _DENSE
    if homogeneous:
        dense = np.zeros_like(_DENSE)
        dense[_ROWS, _COLS] = 1
        expected = ((np.asarray(x) != 0).astype(np.float32) @ dense
                    if op.startswith("left") else dense @ (np.asarray(x) != 0).astype(np.float32))
    np.testing.assert_allclose(product(matrix), expected)
    if algorithm == "bptt":
        factor = (factor != 0).astype(np.float32)
    if op == "right_mv":
        slots = factor * np.asarray(ct)[_ROWS]
    elif op == "left_mv":
        slots = factor * np.asarray(ct)[_COLS]
    elif op == "right_mm":
        slots = np.sum(factor * np.asarray(ct)[_ROWS], axis=1)
    else:
        slots = np.sum(factor * np.asarray(ct)[:, _COLS], axis=0)
    expected_grad = np.array([np.sum(slots)], dtype=np.float32) if homogeneous else slots
    actual_grad = jax.grad(lambda data: jnp.vdot(product(matrix.with_data(data)), ct))(matrix.data)
    np.testing.assert_allclose(actual_grad, expected_grad)


@pytest.mark.parametrize("op", ["mv", "mm"])
def test_pp_prop_indexed_mirror_keeps_weighted_cotangent(monkeypatch, op):
    monkeypatch.setattr(csr_main, "_use_cuda_indexed_binary_route", lambda backend: True)
    matrix = _matrix("pp_prop", False).build_weight_indices()
    x = jnp.array([2., 0., -1.], dtype=jnp.float32)
    if op == "mm":
        x = jnp.stack([x, 2 * x], axis=1)
        ct = jnp.array([[5., 6.], [7., 8.]], dtype=jnp.float32)
        slots = np.sum(np.asarray(x)[_COLS] * np.asarray(ct)[_ROWS], axis=1)
    else:
        ct = jnp.array([5., 7.], dtype=jnp.float32)
        slots = np.asarray(x)[_COLS] * np.asarray(ct)[_ROWS]
    product = lambda m: m @ BinaryArray(x)
    np.testing.assert_allclose(product(matrix), _DENSE @ (np.asarray(x) != 0))
    actual = jax.grad(lambda data: jnp.vdot(product(matrix.with_data(data)), ct))(matrix.data)
    np.testing.assert_allclose(actual, slots)


def test_backward_algorithm_lifecycle_and_weighted_override():
    matrix = _matrix("pp_prop", False)
    assert matrix.with_data(matrix.data).backward_algorithm == "pp_prop"
    assert matrix.T.backward_algorithm == "pp_prop"
    assert matrix.tocsc().backward_algorithm == "pp_prop"
    assert jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(matrix))).backward_algorithm == "pp_prop"
    assert CSR.fromdense(jnp.eye(2, dtype=jnp.float32), backend="jax_raw", backward_algorithm="pp_prop").backward_algorithm == "pp_prop"
    with pytest.raises(ValueError, match="backward algorithm"):
        _matrix("unknown", False)

    x = jnp.array([2., 0., -1.], dtype=jnp.float32)
    ct = jnp.array([5., 7.], dtype=jnp.float32)
    product = lambda m: m @ WeightedArray(x)
    np.testing.assert_allclose(product(_matrix("bptt", False)), _DENSE @ np.asarray(x))
    actual = jax.grad(lambda data: jnp.vdot(product(_matrix("bptt", False).with_data(data)), ct))(_matrix("bptt", False).data)
    np.testing.assert_allclose(actual, np.asarray(x)[_COLS] * np.asarray(ct)[_ROWS])
