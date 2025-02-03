# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


import brainunit as u
import jax


def check_shape(weights, indices, vector, shape, transpose):
    if weights.ndim == 2:
        assert weights.shape == indices.shape, f'The shape of weights and indices should be the same.'
    elif weights.ndim == 1:
        assert weights.size == 1, f'When weights is 1D, it should be a scalar, got {weights.size}.'
    elif weights.ndim == 0:
        weights = u.math.asarray([weights])
    else:
        raise ValueError(f'weight dim should be 2, 1, or 0, but got {weights.ndim}')
    assert indices.shape[0] == shape[0], f'Pre size mismatch, got {weights.shape[0]} != {shape[0]}'
    n_pre, n_post = shape
    if transpose:
        out = jax.ShapeDtypeStruct([n_post], weights.dtype)
        assert vector.shape[0] == n_pre, f'When transpose, vector shape should be {n_pre}, got {vector.shape[0]}'
    else:
        out = jax.ShapeDtypeStruct([n_pre], weights.dtype)
        assert vector.shape[0] == n_post, f'When not transpose, vector shape should be {n_post}, got {vector.shape[0]}'
    return out, weights, n_pre, n_post


def generate_block_dim(
    indices_info: jax.ShapeDtypeStruct,
    **kwargs
) -> int:
    # which is used for TPU/GPU kernel written in JAX pallas
    n_conn = indices_info.shape[1]
    if n_conn <= 32:
        block_size = 32
    elif n_conn <= 64:
        block_size = 64
    elif n_conn <= 128:
        block_size = 128
    elif n_conn <= 256:
        block_size = 256
    else:
        block_size = 128

    return block_size
