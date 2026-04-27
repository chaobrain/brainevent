# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

import os
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import pytest

import brainevent
import brainevent._fcn.binary as binary_mod
import brainevent._fcn.bitpack_binary as bitpack_binary_mod
import brainevent._fcn.compact_binary as compact_binary_mod
from brainevent._event.bitpack_binary import bitpack
from brainevent._event.compact_binary import CompactBinary
from brainevent._fcn.binary import binary_fcnmv
from brainevent._fcn.bitpack_binary import bitpack_binary_fcnmv
from brainevent._fcn.compact_binary import compact_binary_fcnmv
from brainevent._test_util import generate_fixed_conn_num_indices


platform = jax.default_backend()
if os.environ.get('BRAINEVENT_INCLUDE_DUMMY_PYTEST') != '1':
    pytestmark = pytest.mark.skip(reason='dummy backends are excluded from default pytest coverage')


@contextmanager
def _force_backend(backend: str):
    old_backend = brainevent.config.get_backend(platform)
    brainevent.config.set_backend(platform, backend)
    try:
        yield
    finally:
        brainevent.config.set_backend(platform, old_backend)


def _mk_indices(shape, n_conn=4):
    m, n = shape
    return generate_fixed_conn_num_indices(m, n, min(n_conn, n))


def _mk_homo_w(dtype=jnp.float32):
    return jnp.asarray([1.5], dtype=dtype)


def _mk_active_bool(size):
    return jnp.asarray((jnp.arange(size) % 3) == 0, dtype=jnp.bool_)


def _mk_active_float(size):
    return jnp.where(_mk_active_bool(size), jnp.asarray(1.0, dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32))


@pytest.mark.parametrize('event_dtype', ['bool', 'float'])
def test_binary_dummy_kernel_smoke(event_dtype):
    shape = (20, 40)
    m, _ = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    spikes = _mk_active_bool(m) if event_dtype == 'bool' else _mk_active_float(m)

    y = binary_fcnmv(
        weights,
        indices,
        spikes,
        shape=shape,
        transpose=True,
        backend='dummy_kernel',
    )

    assert y.shape == (shape[1],)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))
    assert jnp.all(jnp.asarray(y) == 0)
    jax.block_until_ready(y)


def test_bitpack_dummy_kernel_smoke():
    shape = (20, 40)
    m, _ = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    spikes = _mk_active_bool(m)
    packed = bitpack(spikes, axis=0)

    with _force_backend('dummy_kernel'):
        y = bitpack_binary_fcnmv(
            weights,
            indices,
            packed,
            spikes,
            shape=shape,
            transpose=True,
        )

    assert y.shape == (shape[1],)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))
    assert jnp.all(jnp.asarray(y) == 0)
    jax.block_until_ready(y)


def test_compact_dummy_kernel_smoke():
    shape = (20, 40)
    m, _ = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    spikes = _mk_active_bool(m)
    cb = CompactBinary.from_array(spikes)

    with _force_backend('dummy_kernel'):
        y = compact_binary_fcnmv(
            weights,
            indices,
            cb.packed,
            cb.active_ids,
            cb.n_active,
            cb.value,
            shape=shape,
            transpose=True,
        )

    assert y.shape == (shape[1],)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))
    assert jnp.all(jnp.asarray(y) == 0)
    jax.block_until_ready(y)


def test_compact_dummy_vector_full_smoke():
    shape = (20, 40)
    m, _ = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    spikes = _mk_active_bool(m)
    cb = CompactBinary.compacy_only_vector(spikes)

    with _force_backend('dummy_kernel_vector_full'):
        y = compact_binary_fcnmv(
            weights,
            indices,
            cb.packed,
            cb.active_ids,
            cb.n_active,
            cb.value,
            shape=shape,
            transpose=True,
        )

    assert y.shape == (shape[1],)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))
    assert jnp.all(jnp.asarray(y) == 0)
    jax.block_until_ready(y)


def test_compact_dummy_vector_active_smoke():
    shape = (20, 40)
    m, _ = shape
    indices = _mk_indices(shape)
    weights = _mk_homo_w()
    spikes = _mk_active_bool(m)
    cb = CompactBinary.compacy_only_vector(spikes)

    with _force_backend('dummy_kernel_vector_active'):
        y = compact_binary_fcnmv(
            weights,
            indices,
            cb.packed,
            cb.active_ids,
            cb.n_active,
            cb.value,
            shape=shape,
            transpose=True,
        )

    assert y.shape == (shape[1],)
    assert jnp.all(jnp.isfinite(jnp.asarray(y)))
    assert jnp.all(jnp.asarray(y) == 0)
    jax.block_until_ready(y)


def test_binary_dummy_kernel_rejects_transpose_false():
    with pytest.raises(ValueError, match='transpose=True'):
        binary_mod._binary_fcnmv_dummy_kernel(
            transpose=False,
            spike_info=jax.ShapeDtypeStruct((20,), jnp.bool_),
            outs=[jax.ShapeDtypeStruct((40,), jnp.float32)],
            weight_info=jax.ShapeDtypeStruct((1,), jnp.float32),
        )


def test_bitpack_dummy_kernel_rejects_transpose_false():
    with pytest.raises(ValueError, match='transpose=True'):
        bitpack_binary_mod._bitpack_binary_fcnmv_dummy_kernel(
            transpose=False,
            pack_axis=0,
            outs=[jax.ShapeDtypeStruct((40,), jnp.float32)],
            weight_info=jax.ShapeDtypeStruct((1,), jnp.float32),
        )


def test_compact_dummy_kernel_rejects_transpose_false():
    with pytest.raises(ValueError, match='transpose=True'):
        compact_binary_mod._compact_binary_fcnmv_dummy_kernel(
            transpose=False,
            outs=[jax.ShapeDtypeStruct((40,), jnp.float32)],
            weight_info=jax.ShapeDtypeStruct((1,), jnp.float32),
            packed_info=jax.ShapeDtypeStruct((2,), jnp.uint32),
        )


def test_binary_dummy_kernel_rejects_hetero():
    with pytest.raises(ValueError, match='homogeneous'):
        binary_mod._binary_fcnmv_dummy_kernel(
            transpose=True,
            spike_info=jax.ShapeDtypeStruct((20,), jnp.bool_),
            outs=[jax.ShapeDtypeStruct((40,), jnp.float32)],
            weight_info=jax.ShapeDtypeStruct((20, 4), jnp.float32),
        )


def test_bitpack_dummy_kernel_rejects_hetero():
    with pytest.raises(ValueError, match='homogeneous'):
        bitpack_binary_mod._bitpack_binary_fcnmv_dummy_kernel(
            transpose=True,
            pack_axis=0,
            outs=[jax.ShapeDtypeStruct((40,), jnp.float32)],
            weight_info=jax.ShapeDtypeStruct((20, 4), jnp.float32),
        )


def test_compact_dummy_kernel_rejects_hetero():
    with pytest.raises(ValueError, match='homogeneous'):
        compact_binary_mod._compact_binary_fcnmv_dummy_kernel(
            transpose=True,
            outs=[jax.ShapeDtypeStruct((40,), jnp.float32)],
            weight_info=jax.ShapeDtypeStruct((20, 4), jnp.float32),
            packed_info=jax.ShapeDtypeStruct((2,), jnp.uint32),
        )


def test_compact_dummy_kernel_rejects_vector_only_input():
    with pytest.raises(ValueError, match='packed compact input'):
        compact_binary_mod._compact_binary_fcnmv_dummy_kernel(
            transpose=True,
            outs=[jax.ShapeDtypeStruct((40,), jnp.float32)],
            weight_info=jax.ShapeDtypeStruct((1,), jnp.float32),
            packed_info=jax.ShapeDtypeStruct((0,), jnp.uint32),
        )


def test_compact_dummy_vector_full_rejects_packed_input():
    with pytest.raises(ValueError, match='packed.size == 0'):
        compact_binary_mod._compact_binary_fcnmv_dummy_vector_full_kernel(
            transpose=True,
            outs=[jax.ShapeDtypeStruct((40,), jnp.float32)],
            weight_info=jax.ShapeDtypeStruct((1,), jnp.float32),
            packed_info=jax.ShapeDtypeStruct((2,), jnp.uint32),
        )


def test_compact_dummy_vector_active_rejects_packed_input():
    with pytest.raises(ValueError, match='packed.size == 0'):
        compact_binary_mod._compact_binary_fcnmv_dummy_vector_active_kernel(
            transpose=True,
            outs=[jax.ShapeDtypeStruct((40,), jnp.float32)],
            weight_info=jax.ShapeDtypeStruct((1,), jnp.float32),
            packed_info=jax.ShapeDtypeStruct((2,), jnp.uint32),
        )


@pytest.mark.parametrize(
    ('selector', 'packed_shape', 'expected_kernel_name'),
    [
        (
            compact_binary_mod._compact_binary_fcnmv_dummy_kernel,
            (2,),
            'fcn_dummy_mv.dummy_compact_binary_fcnmv_scatter_homo_f32',
        ),
        (
            compact_binary_mod._compact_binary_fcnmv_dummy_vector_full_kernel,
            (0,),
            'fcn_dummy_mv.dummy_compact_binary_fcnmv_scatter_homo_vector_full_f32',
        ),
        (
            compact_binary_mod._compact_binary_fcnmv_dummy_vector_active_kernel,
            (0,),
            'fcn_dummy_mv.dummy_compact_binary_fcnmv_scatter_homo_vector_active_f32',
        ),
    ],
)
def test_compact_dummy_kernel_selector(monkeypatch, selector, packed_shape, expected_kernel_name):
    called = []

    monkeypatch.setattr(compact_binary_mod, 'load_cuda_file', lambda *args, **kwargs: None)

    def _fake_ffi_call(kernel_name, out_info):
        called.append(kernel_name)
        return lambda *args, **kwargs: kernel_name

    monkeypatch.setattr(jax.ffi, 'ffi_call', _fake_ffi_call)

    kernel = selector(
        transpose=True,
        outs=[jax.ShapeDtypeStruct((5,), jnp.float32)],
        weight_info=jax.ShapeDtypeStruct((1,), jnp.float32),
        packed_info=jax.ShapeDtypeStruct(packed_shape, jnp.uint32),
    )
    result = kernel(None, None, None, None, None, None, None, None, None)

    assert called == [expected_kernel_name]
    assert result == expected_kernel_name
