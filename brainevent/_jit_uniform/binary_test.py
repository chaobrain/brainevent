# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Keep GPU matmul reference numerics stable (avoid TF32 drift in dense @ B checks).
if jax.default_backend() == 'gpu' and jax.config.jax_default_matmul_precision is None:
    jax.config.update('jax_default_matmul_precision', 'highest')

from brainevent._jit_uniform import binary as jitu_binary
from brainevent._jit_uniform.binary import (
    binary_jitumv,
    binary_jitumv_p,
    binary_jitumm,
    binary_jitumm_p,
)
from brainevent._jit_uniform.csr import jitu_to_csr
from brainevent._test_util import allclose

platform = jax.default_backend()
JITUMV_IMPLEMENTATIONS = tuple(binary_jitumv_p.available_backends(platform))
JITUMM_IMPLEMENTATIONS = tuple(binary_jitumm_p.available_backends(platform))

if platform == 'cpu':
    SHAPES = ((20, 30), (100, 50))
else:
    SHAPES = ((20, 30), (100, 50), (400, 300))


def _implementation_params(implementations, op_name: str):
    if implementations:
        return [pytest.param(impl, id=impl) for impl in implementations]
    return [
        pytest.param(
            None,
            marks=pytest.mark.skip(reason=f'No {op_name} implementations on platform={platform}'),
            id=f'no-{op_name}',
        )
    ]


JITUMV_PARAMS = _implementation_params(JITUMV_IMPLEMENTATIONS, 'binary_jitumv')
JITUMM_PARAMS = _implementation_params(JITUMM_IMPLEMENTATIONS, 'binary_jitumm')


def _binary_events(x, dtype=jnp.float32):
    return jnp.asarray(jnp.asarray(x) > 0, dtype=dtype)


def _sample_vector(size: int, event_dtype, seed: int):
    rng = np.random.RandomState(seed)
    if event_dtype is bool:
        return jnp.asarray(rng.rand(size) > 0.5)
    raw = rng.rand(size).astype(np.float32)
    mask = rng.rand(size) > 0.5
    return jnp.asarray(raw * mask)


def _sample_matrix(rows: int, cols: int, event_dtype, seed: int):
    rng = np.random.RandomState(seed)
    if event_dtype is bool:
        return jnp.asarray(rng.rand(rows, cols) > 0.5)
    raw = rng.rand(rows, cols).astype(np.float32)
    mask = rng.rand(rows, cols) > 0.5
    return jnp.asarray(raw * mask)


def _sample_cotangent(shape, seed: int):
    rng = np.random.RandomState(seed)
    return jnp.asarray(rng.randn(*shape).astype(np.float32))


def _active_events(events):
    events = jnp.asarray(events)
    return events if events.dtype == jnp.bool_ else events > 0


def _light_csr_reference(w_low, w_high, prob, seed, events, *, shape, transpose, corder, matrix_mode, backend):
    with pytest.warns(UserWarning, match="corder.*ignored"):
        csr = jitu_to_csr(
            w_low,
            w_high,
            prob,
            seed,
            shape=shape,
            corder=corder,
            matrix_mode=matrix_mode,
            backend=backend,
        )
    events = _active_events(events)
    return csr.T @ events if transpose else csr @ events


def _light_csr_batched_mm_reference(w_low, w_high, prob, seed, matrices, *, shape, transpose, corder, backend):
    matrices = _active_events(matrices)
    batch, rows, cols = matrices.shape
    merged = jnp.transpose(matrices, (1, 0, 2)).reshape(rows, batch * cols)
    expected = _light_csr_reference(
        w_low,
        w_high,
        prob,
        seed,
        merged,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode="mm",
        backend=backend,
    )
    out_rows = shape[1] if transpose else shape[0]
    return expected.reshape(out_rows, batch, cols).transpose(1, 0, 2)


def test_binary_jitumv_warns_that_corder_is_ignored(monkeypatch):
    def fake_p_call(
        w_low,
        w_high,
        clen,
        vector,
        seed,
        *,
        shape,
        transpose,
        corder,
        chunk_size,
        target_chunks,
        backend,
    ):
        out_size = shape[1] if transpose else shape[0]
        return (jnp.zeros((out_size,), dtype=jnp.asarray(w_low).dtype),)

    monkeypatch.setattr(jitu_binary, "binary_jitumv_p_call", fake_p_call)

    with pytest.warns(UserWarning, match="corder.*ignored"):
        out = binary_jitumv(
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(0.5, dtype=jnp.float32),
            0.2,
            jnp.asarray([True, False, True]),
            42,
            shape=(2, 3),
            corder=False,
        )

    assert out.shape == (2,)

    with pytest.warns(UserWarning, match="corder.*ignored"):
        out = binary_jitumv(
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(0.5, dtype=jnp.float32),
            0.2,
            jnp.asarray([True, False, True]),
            42,
            shape=(2, 3),
            corder=False,
        )

    assert out.shape == (2,)


def test_binary_jitumm_warns_that_corder_is_ignored(monkeypatch):
    def fake_p_call(
        w_low,
        w_high,
        clen,
        B,
        seed,
        *,
        shape,
        transpose,
        corder,
        chunk_size,
        target_chunks,
        backend,
    ):
        out_rows = shape[1] if transpose else shape[0]
        return (jnp.zeros((out_rows, B.shape[1]), dtype=jnp.asarray(w_low).dtype),)

    monkeypatch.setattr(jitu_binary, "binary_jitumm_p_call", fake_p_call)

    with pytest.warns(UserWarning, match="corder.*ignored"):
        out = binary_jitumm(
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(0.5, dtype=jnp.float32),
            0.2,
            jnp.asarray([[True, False], [False, True], [True, True]]),
            42,
            shape=(2, 3),
            corder=False,
        )

    assert out.shape == (2, 2)

    with pytest.warns(UserWarning, match="corder.*ignored"):
        out = binary_jitumm(
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(0.5, dtype=jnp.float32),
            0.2,
            jnp.asarray([[True, False], [False, True], [True, True]]),
            42,
            shape=(2, 3),
            corder=False,
        )

    assert out.shape == (2, 2)


@pytest.mark.parametrize(
    ("transpose", "compute_symbol"),
    [
        (False, "binary_jitumv.gather_f32"),
        (True, "binary_jitumv.scatter_f32"),
    ],
)
def test_binary_jitumv_cuda_generator_selects_light_symbols(monkeypatch, transpose, compute_symbol):
    calls = []

    def fake_load_cuda_file(path, name):
        calls.append(("load", str(path), name))

    def fake_ffi_call(name, outs):
        calls.append(("ffi", name, outs))

        def invoke(*args, **kwargs):
            calls.append(("invoke", name, kwargs))
            if isinstance(outs, (tuple, list)):
                return tuple(jnp.zeros(out.shape, out.dtype) for out in outs)
            return jnp.zeros(outs.shape, outs.dtype)

        return invoke

    monkeypatch.setattr(jitu_binary, "load_cuda_file", fake_load_cuda_file)
    monkeypatch.setattr(jitu_binary.jax.ffi, "ffi_call", fake_ffi_call)

    out_shape = (7,) if transpose else (5,)
    vector_shape = (5,) if transpose else (7,)
    kernel = jitu_binary._binary_jitumv_cuda_kernel(
        corder=False,
        vector_info=jax.ShapeDtypeStruct(vector_shape, jnp.bool_),
        transpose=transpose,
        shape=(5, 7),
        outs=[jax.ShapeDtypeStruct(out_shape, jnp.float32)],
        w_low_info=jax.ShapeDtypeStruct((1,), jnp.float32),
    )
    kernel(
        jnp.asarray([0.1], dtype=jnp.float32),
        jnp.asarray([0.5], dtype=jnp.float32),
        jnp.asarray([10], dtype=jnp.int32),
        jnp.asarray([True] * vector_shape[0]),
        jnp.asarray([42], dtype=jnp.int32),
    )

    ffi_names = [entry[1] for entry in calls if entry[0] == "ffi"]
    assert ffi_names == ["binary_jitumv.pack_bool", compute_symbol]


@pytest.mark.parametrize(
    ("transpose", "compute_symbol"),
    [
        (False, "binary_jitumm.gather_f32"),
        (True, "binary_jitumm.scatter_f32"),
    ],
)
def test_binary_jitumm_cuda_generator_selects_light_symbols(monkeypatch, transpose, compute_symbol):
    calls = []

    def fake_load_cuda_file(path, name):
        calls.append(("load", str(path), name))

    def fake_ffi_call(name, outs):
        calls.append(("ffi", name, outs))

        def invoke(*args, **kwargs):
            calls.append(("invoke", name, kwargs))
            if isinstance(outs, (tuple, list)):
                return tuple(jnp.zeros(out.shape, out.dtype) for out in outs)
            return jnp.zeros(outs.shape, outs.dtype)

        return invoke

    monkeypatch.setattr(jitu_binary, "load_cuda_file", fake_load_cuda_file)
    monkeypatch.setattr(jitu_binary.jax.ffi, "ffi_call", fake_ffi_call)

    out_shape = (7, 3) if transpose else (5, 3)
    B_shape = (5, 3) if transpose else (7, 3)
    kernel = jitu_binary._binary_jitumm_cuda_kernel(
        corder=True,
        B_info=jax.ShapeDtypeStruct(B_shape, jnp.bool_),
        transpose=transpose,
        shape=(5, 7),
        outs=[jax.ShapeDtypeStruct(out_shape, jnp.float32)],
        w_low_info=jax.ShapeDtypeStruct((1,), jnp.float32),
    )
    kernel(
        jnp.asarray([0.1], dtype=jnp.float32),
        jnp.asarray([0.5], dtype=jnp.float32),
        jnp.asarray([10], dtype=jnp.int32),
        jnp.asarray([[True, False, True]] * B_shape[0]),
        jnp.asarray([42], dtype=jnp.int32),
    )

    ffi_names = [entry[1] for entry in calls if entry[0] == "ffi"]
    assert ffi_names == ["binary_jitumm.pack", compute_symbol]


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('prob', [0.1, 0.2])
def test_binary_jitumv_forward_matches_reference(implementation, shape, transpose, corder, event_dtype, prob):
    seed = 123
    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)
    event_size = shape[0] if transpose else shape[1]
    vector = _sample_vector(event_size, event_dtype, seed + 7)

    with pytest.warns(UserWarning, match="corder.*ignored"):
        y = binary_jitumv(
            w_low,
            w_high,
            prob,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    y_ref = _light_csr_reference(
        w_low,
        w_high,
        prob,
        seed,
        vector,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode="mv",
        backend=implementation,
    )
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_low, w_high, vector, y, y_ref))


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('prob', [0.1, 0.2])
@pytest.mark.parametrize('k', [5, 10])
def test_binary_jitumm_forward_matches_reference(implementation, shape, transpose, corder, event_dtype, prob, k):
    seed = 123
    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, event_dtype, seed + 11)

    with pytest.warns(UserWarning, match="corder.*ignored"):
        y = binary_jitumm(
            w_low,
            w_high,
            prob,
            matrix,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    y_ref = _light_csr_reference(
        w_low,
        w_high,
        prob,
        seed,
        matrix,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode="mm",
        backend=implementation,
    )
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_low, w_high, matrix, y, y_ref))


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumv_thresholds_float_events(implementation, shape, transpose, corder):
    seed = 123
    prob = 0.1
    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)
    size = shape[0] if transpose else shape[1]
    vector = _sample_vector(size, float, seed + 17)
    vector_binary = _binary_events(vector, dtype=jnp.float32)

    with pytest.warns(UserWarning, match="corder.*ignored"):
        y_float = binary_jitumv(
            w_low,
            w_high,
            prob,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    with pytest.warns(UserWarning, match="corder.*ignored"):
        y_binary = binary_jitumv(
            w_low,
            w_high,
            prob,
            vector_binary,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    assert allclose(y_float, y_binary, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_low, w_high, vector, vector_binary, y_float, y_binary))


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('shape', [(20, 30), (100, 50)])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('k', [10])
def test_binary_jitumm_thresholds_float_events(implementation, shape, transpose, corder, k):
    seed = 123
    prob = 0.1
    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, float, seed + 23)
    matrix_binary = _binary_events(matrix, dtype=jnp.float32)

    with pytest.warns(UserWarning, match="corder.*ignored"):
        y_float = binary_jitumm(
            w_low,
            w_high,
            prob,
            matrix,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    with pytest.warns(UserWarning, match="corder.*ignored"):
        y_binary = binary_jitumm(
            w_low,
            w_high,
            prob,
            matrix_binary,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    assert allclose(y_float, y_binary, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((w_low, w_high, matrix, matrix_binary, y_float, y_binary))


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.skip(reason="Light binary vector-tangent autodiff reference belongs to step 2.")
def test_binary_jitumv_jvp_and_vjp_match_reference(implementation, transpose, corder):
    pytest.skip("Light binary vector-tangent autodiff reference belongs to step 2.")


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.skip(reason="Light binary matrix-tangent autodiff reference belongs to step 2.")
def test_binary_jitumm_jvp_matches_reference(implementation, transpose, corder):
    pytest.skip("Light binary matrix-tangent autodiff reference belongs to step 2.")


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
def test_binary_jitumv_grad_w_bounds_match_reference_and_finite_difference(
    implementation,
    transpose,
    corder,
    event_dtype,
):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    eps = jnp.asarray(1e-3, dtype=jnp.float32)
    vector_size = shape[0] if transpose else shape[1]
    vector = _sample_vector(vector_size, event_dtype, seed + 47)
    cotangent = _sample_cotangent((shape[1] if transpose else shape[0],), seed + 99)

    def scalar_binary(wl, wh):
        out = binary_jitumv(
            wl,
            wh,
            prob,
            vector,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
        return jnp.sum(out * cotangent)

    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)

    grad_w_low = jax.grad(scalar_binary, argnums=0)(w_low, w_high)
    grad_w_high = jax.grad(scalar_binary, argnums=1)(w_low, w_high)

    fd_w_low = (scalar_binary(w_low + eps, w_high) - scalar_binary(w_low - eps, w_high)) / (2.0 * eps)
    fd_w_high = (scalar_binary(w_low, w_high + eps) - scalar_binary(w_low, w_high - eps)) / (2.0 * eps)

    assert allclose(grad_w_low, fd_w_low, rtol=1e-2, atol=1e-2)
    assert allclose(grad_w_high, fd_w_high, rtol=1e-2, atol=1e-2)
    jax.block_until_ready((eps, vector, cotangent, w_low, w_high, grad_w_low, grad_w_high, fd_w_low, fd_w_high))


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
def test_binary_jitumm_grad_w_bounds_match_reference_and_finite_difference(
    implementation,
    transpose,
    corder,
    event_dtype,
):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    eps = jnp.asarray(1e-3, dtype=jnp.float32)
    k = 8
    rows = shape[0] if transpose else shape[1]
    matrix = _sample_matrix(rows, k, event_dtype, seed + 53)
    out_rows = shape[1] if transpose else shape[0]
    cotangent = _sample_cotangent((out_rows, k), seed + 101)

    def scalar_binary(wl, wh):
        out = binary_jitumm(
            wl,
            wh,
            prob,
            matrix,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
        return jnp.sum(out * cotangent)

    w_low = jnp.asarray(-1.5, dtype=jnp.float32)
    w_high = jnp.asarray(1.5, dtype=jnp.float32)

    grad_w_low = jax.grad(scalar_binary, argnums=0)(w_low, w_high)
    grad_w_high = jax.grad(scalar_binary, argnums=1)(w_low, w_high)

    fd_w_low = (scalar_binary(w_low + eps, w_high) - scalar_binary(w_low - eps, w_high)) / (2.0 * eps)
    fd_w_high = (scalar_binary(w_low, w_high + eps) - scalar_binary(w_low, w_high - eps)) / (2.0 * eps)

    assert allclose(grad_w_low, fd_w_low, rtol=1e-2, atol=1e-2)
    assert allclose(grad_w_high, fd_w_high, rtol=1e-2, atol=1e-2)
    jax.block_until_ready((eps, matrix, cotangent, w_low, w_high, grad_w_low, grad_w_high, fd_w_low, fd_w_high))


@pytest.mark.parametrize('implementation', JITUMV_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumv_vmap_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    batch = 6
    event_size = shape[0] if transpose else shape[1]
    vectors = _sample_matrix(batch, event_size, float, seed + 41)

    f_binary = jax.vmap(
        lambda v: binary_jitumv(
            jnp.asarray(-1.5, dtype=jnp.float32),
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            v,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    with pytest.warns(UserWarning, match="corder.*ignored"):
        y_binary = f_binary(vectors)
    y_ref = _light_csr_reference(
        jnp.asarray(-1.5, dtype=jnp.float32),
        jnp.asarray(1.5, dtype=jnp.float32),
        prob,
        seed,
        _active_events(vectors).T,
        shape=shape,
        transpose=transpose,
        corder=corder,
        matrix_mode="mm",
        backend=implementation,
    ).T
    assert allclose(y_binary, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((vectors, y_binary, y_ref))


@pytest.mark.parametrize('implementation', JITUMM_PARAMS)
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('corder', [True, False])
def test_binary_jitumm_vmap_matches_reference(implementation, transpose, corder):
    shape = (20, 30)
    seed = 123
    prob = 0.1
    batch = 4
    k = 7
    rows = shape[0] if transpose else shape[1]
    matrices = _sample_matrix(batch * rows, k, float, seed + 43).reshape(batch, rows, k)

    f_binary = jax.vmap(
        lambda B: binary_jitumm(
            jnp.asarray(-1.5, dtype=jnp.float32),
            jnp.asarray(1.5, dtype=jnp.float32),
            prob,
            B,
            seed,
            shape=shape,
            transpose=transpose,
            corder=corder,
            backend=implementation,
        )
    )
    with pytest.warns(UserWarning, match="corder.*ignored"):
        y_binary = f_binary(matrices)
    y_ref = _light_csr_batched_mm_reference(
        jnp.asarray(-1.5, dtype=jnp.float32),
        jnp.asarray(1.5, dtype=jnp.float32),
        prob,
        seed,
        matrices,
        shape=shape,
        transpose=transpose,
        corder=corder,
        backend=implementation,
    )
    assert allclose(y_binary, y_ref, rtol=1e-4, atol=1e-4)
    jax.block_until_ready((matrices, y_binary, y_ref))
