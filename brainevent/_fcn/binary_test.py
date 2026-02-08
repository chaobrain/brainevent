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


import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._fcn.binary import binary_fcnmv, binary_fcnmv_p, binary_fcnmm, binary_fcnmm_p
from brainevent._test_util import generate_fixed_conn_num_indices

platform = jax.default_backend()
FCNMV_IMPLEMENTATIONS = tuple(binary_fcnmv_p.available_backends(platform))
FCNMM_IMPLEMENTATIONS = tuple(binary_fcnmm_p.available_backends(platform))

if platform == 'cpu':
    SHAPES = ((20, 40), (50, 30))
else:
    SHAPES = ((20, 40), (50, 30), (200, 400))


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


FCNMV_PARAMS = _implementation_params(FCNMV_IMPLEMENTATIONS, 'binary_fcnmv')
FCNMM_PARAMS = _implementation_params(FCNMM_IMPLEMENTATIONS, 'binary_fcnmm')


def _make_weights(indices, homo_w: bool):
    if homo_w:
        return jnp.asarray([1.5], dtype=jnp.float32)
    return braintools.init.Normal(0.0, 1.0)(indices.shape)


def _to_binary_events(events, dtype):
    events = jnp.asarray(events)
    if events.dtype == jnp.bool_:
        return jnp.asarray(events, dtype=dtype)
    return jnp.asarray(events > 0, dtype=dtype)


def _dense_from_fixed_conn(weights, indices, shape):
    n_pre, _ = shape
    rows = jnp.repeat(jnp.arange(n_pre, dtype=indices.dtype), indices.shape[1])
    cols = indices.reshape(-1)
    weights = jnp.asarray(weights)
    if weights.size == 1:
        value = jnp.reshape(weights, (-1,))[0]
        values = jnp.full((indices.size,), value, dtype=weights.dtype)
    else:
        values = weights.reshape(-1)
    return jnp.zeros(shape, dtype=weights.dtype).at[rows, cols].add(values)


def _mv_reference(weights, indices, events, shape, transpose):
    dense = _dense_from_fixed_conn(weights, indices, shape)
    events = _to_binary_events(events, dense.dtype)
    if transpose:
        return events @ dense
    return dense @ events


def _mm_reference(weights, indices, matrix, shape, transpose):
    dense = _dense_from_fixed_conn(weights, indices, shape)
    matrix = _to_binary_events(matrix, dense.dtype)
    if transpose:
        return dense.T @ matrix
    return dense @ matrix


@pytest.mark.parametrize('implementation', FCNMV_PARAMS)
@pytest.mark.parametrize('replace', [True, False])
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('shape', SHAPES)
def test_binary_fcnmv_forward_matches_reference(implementation, replace, homo_w, transpose, event_dtype, shape):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)), replace=replace)
    weights = _make_weights(indices, homo_w)

    event_size = m if transpose else n
    if event_dtype is bool:
        events = brainstate.random.rand(event_size) < 0.5
    else:
        raw = jnp.asarray(brainstate.random.rand(event_size), dtype=jnp.float32)
        events = jnp.where(raw > 0.4, raw, 0.0)

    y = jax.jit(
        lambda: binary_fcnmv(
            weights,
            indices,
            events,
            shape=shape,
            transpose=transpose,
            backend=implementation,
        )
    )()
    y_ref = _mv_reference(weights, indices, events, shape, transpose)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('replace', [True, False])
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
@pytest.mark.parametrize('event_dtype', [bool, float])
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('shape', SHAPES)
def test_binary_fcnmm_forward_matches_reference(implementation, replace, homo_w, transpose, event_dtype, k, shape):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)), replace=replace)
    weights = _make_weights(indices, homo_w)

    n_rows = m if transpose else n
    if event_dtype is bool:
        matrix = brainstate.random.rand(n_rows, k) < 0.5
    else:
        raw = jnp.asarray(brainstate.random.rand(n_rows, k), dtype=jnp.float32)
        matrix = jnp.where(raw > 0.4, raw, 0.0)

    y = jax.jit(
        lambda: binary_fcnmm(
            weights,
            indices,
            matrix,
            shape=shape,
            transpose=transpose,
            backend=implementation,
        )
    )()
    y_ref = _mm_reference(weights, indices, matrix, shape, transpose)
    assert jnp.allclose(y, y_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('implementation', FCNMM_PARAMS)
@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('shape', [(20, 40), (50, 30)])
@pytest.mark.parametrize('k', [10])
@pytest.mark.parametrize('transpose', [True, False])
def test_binary_fcnmm_thresholds_float_events(implementation, homo_w, shape, k, transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, max(1, int(n * 0.1)), replace=False)
    weights = _make_weights(indices, homo_w)

    rng = np.random.RandomState(42)
    n_rows = m if transpose else n
    raw = rng.rand(n_rows, k).astype(np.float32)
    mask = rng.rand(n_rows, k) > 0.5
    float_events = jnp.asarray(raw * mask)
    binary_events = jnp.asarray(float_events > 0, dtype=jnp.float32)

    y_float = jax.jit(
        lambda: binary_fcnmm(
            weights,
            indices,
            float_events,
            shape=shape,
            transpose=transpose,
            backend=implementation,
        )
    )()
    y_binary = jax.jit(
        lambda: binary_fcnmm(
            weights,
            indices,
            binary_events,
            shape=shape,
            transpose=transpose,
            backend=implementation,
        )
    )()
    assert jnp.allclose(y_float, y_binary, rtol=1e-3, atol=1e-3)
