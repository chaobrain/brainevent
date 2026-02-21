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
Diagonal GRU with multi-head input projection (GRUDiagMH).

Implementation of the Fully Gated Recurrent Unit with diagonal recurrence
matrices and multi-head input projections, following the ml-pararnn paper
(arXiv:2510.21450).

GRU equations (diagonal form)::

    z     = sigma_z(diag(A_z) h + B_z x + b_z)
    r     = sigma_r(diag(A_r) h + B_r x + b_r)
    h_new = sigma_h(diag(A_h) (h * r) + B_h x + b_h)
    h     = (1 - z) * h + z * h_new

Parameters A, B, b are stored in collated form:
- A: shape ``(3, state_dim)`` — diagonal recurrence weights for [z, r, h]
- B: shape ``(num_heads, head_input_dim, 3, head_state_dim)`` — multi-head input weights
- b: shape ``(3, state_dim)`` — biases for [z, r, h]
"""

from typing import Optional

import brainstate
import jax
import jax.numpy as jnp
import jax.random as jr

from ._cell import BaseRNNCell, apply_rnn, _roll_state
from ._init import INITIALIZERS
from ._newton import NewtonConfig
from ._nonlinearities import get_nonlinearity
from ._parallel_reduce import parallel_reduce_diag

__all__ = ['GRUDiagMH']


class GRUDiagMHImpl(BaseRNNCell):
    """Diagonal GRU implementation with multi-head input projection.

    This class provides the mathematical operations (recurrence step,
    Jacobian computation, gradient backpropagation) as static methods.
    The actual module wrapping is done by ``GRUDiagMH``.

    Params order: A, B, b (3 array params), then 6 callable params.
    """

    # A, B, b are JAX arrays; nonlin/deriv functions are static
    num_array_params = 3

    @staticmethod
    def recurrence_step(x, h, A, B, b,
                        nonlin_update, nonlin_reset, nonlin_state,
                        deriv_update, deriv_reset, deriv_state):
        """GRU recurrence: h_t = f(h_{t-1}, x_t; A, B, b).

        Args:
            x: Input, shape ``(..., input_dim)``.
            h: Previous hidden state, shape ``(..., state_dim)``.
            A: Diagonal recurrence weights ``(3, state_dim)``.
            B: Multi-head input weights ``(num_heads, head_input_dim, 3, head_state_dim)``.
            b: Biases ``(3, state_dim)``.
        """
        num_heads = B.shape[0]
        # Multi-head input projection: x -> Bx
        x_heads = x.reshape((*x.shape[:-1], num_heads, -1))
        Bxpb = jnp.einsum('...hi,hivj->...vhj', x_heads, B)
        Bxpb = Bxpb.reshape((*Bxpb.shape[:-2], -1)) + b  # (..., 3, state_dim)

        # Gate pre-activations
        Ah = jnp.einsum('vj,...j->...vj', A[:2, :], h)  # (..., 2, state_dim)
        pre_z = Ah[..., 0, :] + Bxpb[..., 0, :]
        pre_r = Ah[..., 1, :] + Bxpb[..., 1, :]

        z = nonlin_update(pre_z)
        r = nonlin_reset(pre_r)

        pre_h = A[2, :] * h * r + Bxpb[..., 2, :]
        h_new = nonlin_state(pre_h)

        return z * h_new + (1.0 - z) * h

    @staticmethod
    def compute_jacobians(h, x, A, B, b,
                          nonlin_update, nonlin_reset, nonlin_state,
                          deriv_update, deriv_reset, deriv_state):
        """Compute diagonal Jacobians -df/dh_{t-1} for forward solve."""
        h_prev = _roll_state(h)
        num_heads = B.shape[0]

        x_heads = x.reshape((*x.shape[:-1], num_heads, -1))
        Bxpb = jnp.einsum('...hi,hivj->...vhj', x_heads, B)
        Bxpb = Bxpb.reshape((*Bxpb.shape[:-2], -1)) + b

        pre_z = jnp.einsum('j,...j->...j', A[0, :], h_prev) + Bxpb[..., 0, :]
        pre_r = jnp.einsum('j,...j->...j', A[1, :], h_prev) + Bxpb[..., 1, :]

        z = nonlin_update(pre_z)
        r = nonlin_reset(pre_r)
        pre_h = A[2, :] * h_prev * r + Bxpb[..., 2, :]
        h_candidate = nonlin_state(pre_h)

        grad_z = deriv_update(pre_z)
        grad_r = deriv_reset(pre_r)
        grad_h = deriv_state(pre_h)

        J_z = A[0, :] * grad_z
        J_r = A[1, :] * grad_r
        J_h = A[2, :] * grad_h

        J_h = J_h * (r + h_prev * J_r)

        jac = (1.0 - z) + (h_candidate - h_prev) * J_z + z * J_h
        return -jac

    @staticmethod
    def compute_jacobians_bwd(h, x, A, B, b,
                              nonlin_update, nonlin_reset, nonlin_state,
                              deriv_update, deriv_reset, deriv_state):
        """Compute Jacobians for backward (transposed) system."""
        jac = GRUDiagMHImpl.compute_jacobians(
            h, x, A, B, b,
            nonlin_update, nonlin_reset, nonlin_state,
            deriv_update, deriv_reset, deriv_state,
        )
        # For diagonal Jacobians, transpose is identity
        # Flip time, roll, zero first
        jac = jnp.flip(jac, axis=-2)
        jac = jnp.roll(jac, shift=1, axis=-2)
        jac = jac.at[..., 0, :].set(0.0)
        return jac

    @staticmethod
    def linear_solve(jac, rhs):
        return parallel_reduce_diag(jac, rhs)

    @staticmethod
    def post_process(h, x, *params):
        return h

    @staticmethod
    def backprop_post_process(grad_y, x, h, A, B, b, *rest):
        return (
            grad_y,  # grad_h
            jnp.zeros_like(x),  # grad_x
            jnp.zeros_like(A),  # grad_A
            jnp.zeros_like(B),  # grad_B
            jnp.zeros_like(b),  # grad_b
            None, None, None,  # nonlin fns (not differentiable)
            None, None, None,  # deriv fns
        )

    @staticmethod
    def backprop_to_params(dl_dh, x, h, A, B, b,
                           nonlin_update, nonlin_reset, nonlin_state,
                           deriv_update, deriv_reset, deriv_state):
        """Compute gradients w.r.t. x and parameters."""
        h_prev = _roll_state(h)
        num_heads = B.shape[0]

        x_heads = x.reshape((*x.shape[:-1], num_heads, -1))
        Bxpb = jnp.einsum('...hi,hivj->...vhj', x_heads, B)
        Bxpb = Bxpb.reshape((*Bxpb.shape[:-2], -1)) + b

        pre_z = jnp.einsum('j,...j->...j', A[0, :], h_prev) + Bxpb[..., 0, :]
        pre_r = jnp.einsum('j,...j->...j', A[1, :], h_prev) + Bxpb[..., 1, :]

        z = nonlin_update(pre_z)
        r = nonlin_reset(pre_r)
        pre_h = A[2, :] * h_prev * r + Bxpb[..., 2, :]
        h_new = nonlin_state(pre_h)

        # Gradients through GRU equations
        grad_h_gate = dl_dh * z * deriv_state(pre_h)
        grad_z_gate = dl_dh * (h_new - h_prev) * deriv_update(pre_z)
        grad_r_gate = grad_h_gate * A[2, :] * h_prev * deriv_reset(pre_r)

        # Stack gate gradients: (B, T, 3, state_dim)
        grad_zrh = jnp.stack([grad_z_gate, grad_r_gate, grad_h_gate], axis=-2)

        # grad_b: sum over batch and time dims
        sum_axes = tuple(range(grad_zrh.ndim - 2))
        grad_b = jnp.sum(grad_zrh, axis=sum_axes)

        # grad_B: input weight gradients
        grad_zrh_heads = grad_zrh.reshape((*grad_zrh.shape[:-1], num_heads, -1))
        grad_B = jnp.einsum('...vhi,...hj->hjvi', grad_zrh_heads, x_heads)

        # grad_x: backprop through input projection
        grad_x = jnp.einsum('...vhi,hjvi->...hj', grad_zrh_heads, B)
        grad_x = grad_x.reshape((*grad_x.shape[:-2], -1))

        # grad_A: diagonal recurrence weight gradients
        grad_zrh_for_A = grad_zrh.at[..., 2, :].set(grad_zrh[..., 2, :] * r)
        grad_A = jnp.einsum('...vi,...i->vi', grad_zrh_for_A, h_prev)

        return (
            grad_x,
            grad_A, grad_B, grad_b,
            None, None, None,  # nonlin fns
            None, None, None,  # deriv fns
        )

    @staticmethod
    def assemble_initial_guess(x, state_dim, *params):
        h0 = jnp.zeros((*x.shape[:-1], state_dim), dtype=x.dtype)
        return GRUDiagMHImpl.recurrence_step(x, h0, *params)


class GRUDiagMH(brainstate.nn.Module):
    """Diagonal GRU with multi-head input projection.

    This module wraps the ``GRUDiagMHImpl`` cell with ``brainstate.nn.Module``
    for state management and provides a clean API for training.

    Args:
        input_dim: Input feature dimension.
        state_dim: Hidden state dimension.
        num_heads: Number of heads for input projection. Must divide both
            ``input_dim`` and ``state_dim``.
        mode: Application mode (``'sequential'`` or ``'parallel'``).
        nonlin_update: Activation for update gate z (default: ``'sigmoid'``).
        nonlin_reset: Activation for reset gate r (default: ``'sigmoid'``).
        nonlin_state: Activation for candidate state (default: ``'tanh'``).
        a_init: Initialization for A weights (default: ``'xlstm'``).
        b_init: Initialization for bias b (default: ``'bias_minus_linspace'``).
        w_init: Initialization for B weights (default: ``'xavier_uniform'``).
        newton_config: Configuration for Newton solver.
        seed: Random seed for initialization.

    Example::

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> gru = GRUDiagMH(input_dim=32, state_dim=64, num_heads=2)
        >>> x = jnp.ones((4, 100, 32))  # (batch, time, features)
        >>> y = gru(x)  # (4, 100, 64)
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_heads: int = 2,
        mode: str = 'parallel',
        nonlin_update: str = 'sigmoid',
        nonlin_reset: str = 'sigmoid',
        nonlin_state: str = 'tanh',
        a_init: str = 'xlstm',
        b_init: str = 'bias_minus_linspace',
        w_init: str = 'xavier_uniform',
        newton_config: Optional[NewtonConfig] = None,
        seed: int = 0,
    ):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must divide input_dim ({input_dim})"
            )
        if state_dim % num_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must divide state_dim ({state_dim})"
            )

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.mode = mode
        self.newton_config = newton_config or NewtonConfig()

        head_input_dim = input_dim // num_heads
        head_state_dim = state_dim // num_heads

        # Get nonlinearity functions and their derivatives
        self.nonlin_update, self.deriv_update = get_nonlinearity(nonlin_update)
        self.nonlin_reset, self.deriv_reset = get_nonlinearity(nonlin_reset)
        self.nonlin_state, self.deriv_state = get_nonlinearity(nonlin_state)

        # Initialize parameters
        key = jr.PRNGKey(seed)
        k1, k2, k3 = jr.split(key, 3)

        A_data = INITIALIZERS[a_init](k1, (3, state_dim), fan_in=state_dim)
        B_data = INITIALIZERS[w_init](
            k2, (num_heads, head_input_dim, 3, head_state_dim),
            fan_in=head_input_dim, fan_out=state_dim,
        )
        b_data = INITIALIZERS[b_init](k3, (3, state_dim), fan_in=0, fan_out=3 * state_dim)

        self.A = brainstate.ParamState(A_data)
        self.B = brainstate.ParamState(B_data)
        self.b = brainstate.ParamState(b_data)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the GRU on an input sequence.

        Args:
            x: Input tensor, shape ``(B, T, input_dim)``.

        Returns:
            Hidden states, shape ``(B, T, state_dim)``.
        """
        return apply_rnn(
            GRUDiagMHImpl, x, self.state_dim, self.mode, self.newton_config,
            self.A.value, self.B.value, self.b.value,
            self.nonlin_update, self.nonlin_reset, self.nonlin_state,
            self.deriv_update, self.deriv_reset, self.deriv_state,
        )
