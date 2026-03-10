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
import pytest

from brainevent._fcn.float import fcnmv, fcnmm
from brainevent._test_util import (
    generate_fixed_conn_num_indices,
    vector_fcn,
    matrix_fcn,
    fcn_vector,
    fcn_matrix,
    allclose,
)

shape = (20, 40)
n_conn = 4


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_fcnmv(homo_w, transpose):
    m, n = shape
    indices = generate_fixed_conn_num_indices(m, n, n_conn)
    w = jnp.asarray(1.5) if homo_w else jnp.ones(indices.shape)
    if transpose:
        x = jnp.ones(m)
        y = fcnmv(w, indices, x, shape=shape, transpose=True)
        y_ref = vector_fcn(x, w, indices, shape)
    else:
        x = jnp.ones(n)
        y = fcnmv(w, indices, x, shape=shape, transpose=False)
        y_ref = fcn_vector(x, w, indices, shape)
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('homo_w', [True, False])
@pytest.mark.parametrize('transpose', [True, False])
def test_fcnmm(homo_w, transpose):
    m, n, k = *shape, 8
    indices = generate_fixed_conn_num_indices(m, n, n_conn)
    w = jnp.asarray(1.5) if homo_w else jnp.ones(indices.shape)
    if transpose:
        x = jnp.ones((k, m))
        y = fcnmm(w, indices, x.T, shape=shape, transpose=True).T
        y_ref = matrix_fcn(x, w, indices, shape)
    else:
        x = jnp.ones((n, k))
        y = fcnmm(w, indices, x, shape=shape, transpose=False)
        y_ref = fcn_matrix(x, w, indices, shape)
    assert allclose(y, y_ref, rtol=1e-4, atol=1e-4)
