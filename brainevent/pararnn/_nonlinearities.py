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
Activation functions and their analytical derivatives for parallel RNN training.

Each entry in NONLINEARITIES is a (function, derivative) pair. The derivative
is computed analytically (not via AD) because it is needed for Jacobian assembly
in the Newton solver, where we want explicit control over the computation.
"""

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

__all__ = ['NONLINEARITIES', 'get_nonlinearity']


def _identity(x):
    return x


def _identity_deriv(x):
    return jnp.ones_like(x)


def _relu(x):
    return jax.nn.relu(x)


def _relu_deriv(x):
    return jnp.where(x > 0, 1.0, 0.0)


def _leaky_relu(x):
    return jax.nn.leaky_relu(x, negative_slope=0.01)


def _leaky_relu_deriv(x):
    return jnp.where(x > 0, 1.0, 0.01)


def _silu(x):
    return jax.nn.silu(x)


def _silu_deriv(x):
    s = jax.nn.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _gelu(x):
    return jax.nn.gelu(x, approximate=True)


def _gelu_deriv(x):
    # Derivative of tanh-approximated GELU
    k = 0.044715
    t = jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + k * x ** 3))
    dt = 1.0 - t ** 2
    inner_deriv = jnp.sqrt(2.0 / jnp.pi) * (1.0 + 3.0 * k * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * dt * inner_deriv


def _selu(x):
    return jax.nn.selu(x)


def _selu_deriv(x):
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return jnp.where(x > 0, scale, scale * alpha * jnp.exp(x))


def _tanh(x):
    return jnp.tanh(x)


def _tanh_deriv(x):
    t = jnp.tanh(x)
    return 1.0 - t * t


def _sigmoid(x):
    return jax.nn.sigmoid(x)


def _sigmoid_deriv(x):
    s = jax.nn.sigmoid(x)
    return s * (1.0 - s)


def _half_sigmoid(x):
    return 0.5 * jax.nn.sigmoid(x)


def _half_sigmoid_deriv(x):
    s = jax.nn.sigmoid(x)
    return 0.5 * s * (1.0 - s)


def _exp(x):
    return jnp.exp(x)


def _exp_deriv(x):
    return jnp.exp(x)


def _softplus(x):
    return jax.nn.softplus(x)


def _softplus_deriv(x):
    return jax.nn.sigmoid(x)


def _elu(x):
    return jax.nn.elu(x, alpha=1.0)


def _elu_deriv(x):
    return jnp.where(x > 0, 1.0, jnp.exp(x))


# Registry: name -> (activation, derivative)
NONLINEARITIES: Dict[str, Tuple[Callable, Callable]] = {
    'identity': (_identity, _identity_deriv),
    'relu': (_relu, _relu_deriv),
    'leaky_relu': (_leaky_relu, _leaky_relu_deriv),
    'silu': (_silu, _silu_deriv),
    'gelu': (_gelu, _gelu_deriv),
    'selu': (_selu, _selu_deriv),
    'tanh': (_tanh, _tanh_deriv),
    'sigmoid': (_sigmoid, _sigmoid_deriv),
    'half_sigmoid': (_half_sigmoid, _half_sigmoid_deriv),
    'exp': (_exp, _exp_deriv),
    'softplus': (_softplus, _softplus_deriv),
    'elu': (_elu, _elu_deriv),
}


def get_nonlinearity(name: str) -> Tuple[Callable, Callable]:
    """Get (activation, derivative) pair by name.

    Args:
        name: Name of the nonlinearity.

    Returns:
        Tuple of (activation_fn, derivative_fn).

    Raises:
        KeyError: If the name is not in the registry.
    """
    if name not in NONLINEARITIES:
        raise KeyError(
            f"Unknown nonlinearity '{name}'. "
            f"Available: {list(NONLINEARITIES.keys())}"
        )
    return NONLINEARITIES[name]
