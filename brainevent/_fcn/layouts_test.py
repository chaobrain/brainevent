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

import jax.numpy as jnp
import numpy as np

from brainevent._fcn.layouts import (
    EllLayout,
    CscLayout,
    ExecPlan,
    resolve_matvec,
    resolve_matmat,
)


def test_ell_layout_todense_axis0_hetero():
    data = jnp.array([[1., 2.], [3., 4.]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    ell = EllLayout(data, indices, axis=0)
    dense = ell.todense(shape=(2, 3))
    expected = jnp.array([[1., 2., 0.], [0., 3., 4.]], dtype=jnp.float32)
    assert jnp.allclose(dense, expected)


def test_ell_layout_todense_homogeneous():
    data = jnp.array([1.5], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    ell = EllLayout(data, indices, axis=0)
    dense = ell.todense(shape=(2, 3))
    expected = jnp.array([[1.5, 1.5, 0.], [0., 1.5, 1.5]], dtype=jnp.float32)
    assert jnp.allclose(dense, expected)


def test_ell_layout_a_shape_axis1_is_transposed():
    data = jnp.array([1.0], dtype=jnp.float32)
    indices = jnp.array([[0], [1], [0]], dtype=jnp.int32)
    ell = EllLayout(data, indices, axis=1)
    assert ell.a_shape(shape=(2, 3)) == (3, 2)


def test_ell_layout_is_homogeneous():
    assert EllLayout(jnp.ones(1), jnp.zeros((2, 2), jnp.int32), axis=0).is_homogeneous
    assert not EllLayout(jnp.ones((2, 2)), jnp.zeros((2, 2), jnp.int32), axis=0).is_homogeneous


def test_ell_to_csc_roundtrip_todense():
    data = jnp.array([[1., 2.], [3., 4.]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    ell = EllLayout(data, indices, axis=0)
    csc = ell.to_csc(shape=(2, 3))
    assert isinstance(csc, CscLayout)
    assert jnp.allclose(csc.todense(shape=(2, 3)), ell.todense(shape=(2, 3)))


def test_ell_to_csc_homogeneous_keeps_scalar_weight():
    data = jnp.array([1.5], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    csc = EllLayout(data, indices, axis=0).to_csc(shape=(2, 3))
    assert csc.weights.size == 1
    assert csc.indptr.shape[0] == 3 + 1


def test_csc_refresh_values_uses_perm():
    data = jnp.array([[1., 2.], [3., 4.]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    csc = EllLayout(data, indices, axis=0).to_csc(shape=(2, 3))
    new = jnp.array([[10., 20.], [30., 40.]], dtype=jnp.float32)
    refreshed = csc.with_values_from_ell(new)
    assert jnp.allclose(refreshed.weights, new.reshape(-1)[np.asarray(csc.perm)])


def test_csc_to_ell_roundtrip_when_fixed_conn():
    data = jnp.array([[1., 2.], [3., 4.]], dtype=jnp.float32)
    indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    ell0 = EllLayout(data, indices, axis=0)
    csc = ell0.to_csc(shape=(2, 3))
    ell1 = csc.to_ell(a_shape=(2, 3), num_conn=2, axis=0)
    assert jnp.allclose(ell1.todense(shape=(2, 3)), ell0.todense(shape=(2, 3)))


def test_resolve_matvec_axis0_gather_cpu_uses_ell():
    plan = resolve_matvec(axis=0, transpose_W=False, is_event=True,
                          backend_is_cuda=False, shape=(2, 3))
    assert plan.format == 'ell'
    assert plan.transpose is False
    assert plan.a_shape == (2, 3)


def test_resolve_matvec_axis0_gather_cuda_event_uses_csc():
    plan = resolve_matvec(axis=0, transpose_W=False, is_event=True,
                          backend_is_cuda=True, shape=(2, 3))
    assert plan.format == 'csc'
    assert plan.a_shape == (2, 3)


def test_resolve_matvec_axis0_scatter_cuda_uses_ell():
    plan = resolve_matvec(axis=0, transpose_W=True, is_event=True,
                          backend_is_cuda=True, shape=(2, 3))
    assert plan.format == 'ell'
    assert plan.transpose is True


def test_resolve_matvec_float_never_csc():
    plan = resolve_matvec(axis=0, transpose_W=False, is_event=False,
                          backend_is_cuda=True, shape=(2, 3))
    assert plan.format == 'ell'


def test_resolve_matvec_axis1_xor_rule():
    plan = resolve_matvec(axis=1, transpose_W=False, is_event=True,
                          backend_is_cuda=True, shape=(2, 3))
    assert plan.format == 'ell'
    assert plan.transpose is True
    assert plan.a_shape == (3, 2)


def test_resolve_matmat_always_ell():
    plan = resolve_matmat(axis=0, transpose_W=False, shape=(2, 3))
    assert plan.format == 'ell'
    assert isinstance(plan, ExecPlan)
