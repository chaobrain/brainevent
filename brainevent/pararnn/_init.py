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

"""
Weight initialization functions for parallel RNN cells.

Each initializer takes a JAX PRNG key, shape, and fan_in/fan_out dimensions,
and returns a JAX array with the initialized weights.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = ['INITIALIZERS', 'initialize']


def _xavier_uniform(key: jax.Array, shape: Tuple[int, ...],
                    fan_in: int, fan_out: Optional[int] = None) -> jax.Array:
    fan_out = fan_out or fan_in
    std = math.sqrt(6.0 / (fan_in + fan_out))
    return jr.uniform(key, shape, minval=-std, maxval=std)


def _kaiming_uniform(key: jax.Array, shape: Tuple[int, ...],
                     fan_in: int, fan_out: Optional[int] = None) -> jax.Array:
    std = math.sqrt(3.0 / fan_in)
    return jr.uniform(key, shape, minval=-std, maxval=std)


def _xavier_gaussian(key: jax.Array, shape: Tuple[int, ...],
                     fan_in: int, fan_out: Optional[int] = None) -> jax.Array:
    stdv = 1.0 / math.sqrt(fan_in)
    return jr.truncated_normal(key, -0.9 / stdv, 0.9 / stdv, shape) * stdv


def _xlstm(key: jax.Array, shape: Tuple[int, ...],
           fan_in: int, fan_out: Optional[int] = None) -> jax.Array:
    stdv = 2.0 / math.sqrt(5.0 * fan_in)
    return jr.normal(key, shape) * stdv


def _small_gaussian(key: jax.Array, shape: Tuple[int, ...],
                    fan_in: int, fan_out: Optional[int] = None) -> jax.Array:
    return jr.truncated_normal(key, -2.0, 2.0, shape) * 0.01


def _negative_exponential_mamba(key: jax.Array, shape: Tuple[int, ...],
                                fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    return jnp.exp(-jr.uniform(key, shape, minval=1.0, maxval=16.0))


def _negative_exponential(key: jax.Array, shape: Tuple[int, ...],
                          fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    return jnp.exp(-jr.uniform(key, shape, minval=0.0, maxval=8.0))


def _zero(key: jax.Array, shape: Tuple[int, ...],
          fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    return jnp.zeros(shape)


def _uniform(key: jax.Array, shape: Tuple[int, ...],
             fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    return jr.uniform(key, shape, minval=-0.9, maxval=0.9)


def _gazillion(key: jax.Array, shape: Tuple[int, ...],
               fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    return jr.uniform(key, shape, minval=-90000.0, maxval=900000.0)


def _constant_zero(key: jax.Array, shape: Tuple[int, ...],
                   fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    return jnp.zeros(shape)


def _bias_uniform(key: jax.Array, shape: Tuple[int, ...],
                  fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    result = jnp.zeros(shape)
    return result.at[0].set(jr.uniform(key, (shape[-1],), minval=-0.9, maxval=0.9))


def _bias_linspace(key: jax.Array, shape: Tuple[int, ...],
                   fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    result = jnp.zeros(shape)
    return result.at[0, :].set(jnp.linspace(3.0, 6.0, shape[-1]))


def _bias_minus_linspace(key: jax.Array, shape: Tuple[int, ...],
                         fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    result = jnp.zeros(shape)
    return result.at[0, :].set(-1.0 * jnp.linspace(0.0, 1.0, shape[-1]))


def _bias_minus_linspace_small(key: jax.Array, shape: Tuple[int, ...],
                               fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    result = jnp.zeros(shape)
    return result.at[0, :].set(-1.0 * jnp.linspace(0.0, 1.0, shape[-1]))


def _bias_constant_1(key: jax.Array, shape: Tuple[int, ...],
                     fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    result = jnp.zeros(shape)
    return result.at[0, :].set(1.0)


def _bias_constant_minus_1(key: jax.Array, shape: Tuple[int, ...],
                           fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    result = jnp.zeros(shape)
    return result.at[0, :].set(-1.0)


def _bias_constant_2(key: jax.Array, shape: Tuple[int, ...],
                     fan_in: int = 0, fan_out: Optional[int] = None) -> jax.Array:
    result = jnp.zeros(shape)
    return result.at[0, :].set(2.0)


# Registry: name -> init_fn(key, shape, fan_in, fan_out) -> array
INITIALIZERS: Dict[str, Callable] = {
    'xavier_uniform': _xavier_uniform,
    'kaiming_uniform': _kaiming_uniform,
    'xavier_gaussian': _xavier_gaussian,
    'xlstm': _xlstm,
    'small_gaussian': _small_gaussian,
    'negative_exponential_mamba': _negative_exponential_mamba,
    'negative_exponential': _negative_exponential,
    'zero': _zero,
    'uniform': _uniform,
    'gazillion': _gazillion,
    'constant_zero': _constant_zero,
    'bias_uniform': _bias_uniform,
    'bias_linspace': _bias_linspace,
    'bias_minus_linspace': _bias_minus_linspace,
    'bias_minus_linspace_small': _bias_minus_linspace_small,
    'bias_constant_1': _bias_constant_1,
    'bias_constant_minus_1': _bias_constant_minus_1,
    'bias_constant_2': _bias_constant_2,
}


def initialize(name: str, key: jax.Array, shape: Tuple[int, ...],
               fan_in: int = 1, fan_out: Optional[int] = None) -> jax.Array:
    """Initialize a weight tensor using the named strategy.

    Args:
        name: Name of the initialization strategy.
        key: JAX PRNG key.
        shape: Shape of the tensor to initialize.
        fan_in: Fan-in dimension (number of input features).
        fan_out: Fan-out dimension (number of output features).

    Returns:
        Initialized JAX array.

    Raises:
        KeyError: If the name is not in the registry.
    """
    if name not in INITIALIZERS:
        raise KeyError(
            f"Unknown initializer '{name}'. "
            f"Available: {list(INITIALIZERS.keys())}"
        )
    return INITIALIZERS[name](key, shape, fan_in, fan_out)
