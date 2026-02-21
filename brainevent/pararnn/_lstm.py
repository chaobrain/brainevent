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
LSTM with Coupled Input-Forget Gate, diagonal recurrence, and multi-head
input projection (LSTMCIFGDiagMH).

This implements the efficient LSTM variant from https://arxiv.org/pdf/1503.04069
with diagonal state matrices and multi-head input splitting, adapted for
parallel training via Newton's method.

LSTM-CIFG equations (diagonal form)::

    f     = sigma_f(diag(A_f) h + B_f x + C_f c + b_f)
    c_new = sigma_c(diag(A_c) h + B_c x + b_c)
    c     = f * c + (1 - f) * c_new
    o     = sigma_o(diag(A_o) h + B_o x + C_o c + b_o)
    h     = o * sigma_h(c)

The hidden state is ``[c, h]`` concatenated, giving block-diagonal 2x2
Jacobians. Parameters:
- A: shape ``(3, state_dim)`` — diagonal weights for [f, o, c]
- B: shape ``(num_heads, head_input_dim, 3, head_state_dim)`` — multi-head input
- C: shape ``(2, state_dim)`` — peephole connections for [f, o]
- b: shape ``(3, state_dim)`` — biases for [f, o, c]
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
from ._parallel_reduce import parallel_reduce_block_diag

__all__ = ['LSTMCIFGDiagMH']


def _split_hidden(h):
    """Split concatenated [c, h] hidden state."""
    half = h.shape[-1] // 2
    return h[..., :half], h[..., half:]


def _combine_hidden(c, h):
    """Concatenate [c, h] into hidden state."""
    return jnp.concatenate([c, h], axis=-1)


class LSTMCIFGDiagMHImpl(BaseRNNCell):
    """LSTM-CIFG implementation with diagonal Jacobians and multi-head input.

    Params order: A, B, C, b (4 array params), then 8 callable params.
    """

    # A, B, C, b are JAX arrays; nonlin/deriv functions are static
    num_array_params = 4

    @staticmethod
    def recurrence_step(x, h, A, B, C, b,
                        nonlin_f, nonlin_o, nonlin_c, nonlin_state,
                        deriv_f, deriv_o, deriv_c, deriv_state):
        cc, hh = _split_hidden(h)
        num_heads = B.shape[0]

        # Multi-head input projection
        x_heads = x.reshape((*x.shape[:-1], num_heads, -1))
        linear = jnp.einsum('hivj,...hi->...vhj', B, x_heads)
        linear = linear.reshape((*linear.shape[:-2], -1)) + b  # (..., 3, state_dim)

        # Add diagonal recurrence
        Ah = jnp.einsum('vj,...j->...vj', A, hh)  # (..., 3, state_dim)
        pre_f = Ah[..., 0, :] + linear[..., 0, :]
        pre_o = Ah[..., 1, :] + linear[..., 1, :]
        pre_c = Ah[..., 2, :] + linear[..., 2, :]

        # Forget gate with peephole
        f = nonlin_f(pre_f + C[0, :] * cc)
        c_new = nonlin_c(pre_c)

        # Update cell state
        cc_new = f * cc + (1.0 - f) * c_new

        # Output gate with peephole to new cell state
        o = nonlin_o(pre_o + C[1, :] * cc_new)
        hh_new = o * nonlin_state(cc_new)

        return _combine_hidden(cc_new, hh_new)

    @staticmethod
    def compute_jacobians(h, x, A, B, C, b,
                          nonlin_f, nonlin_o, nonlin_c, nonlin_state,
                          deriv_f, deriv_o, deriv_c, deriv_state):
        """Compute block-diagonal 2x2 Jacobians for forward solve.

        Returns: jac of shape ``(B, T, state_dim//2, 2, 2)``
        """
        cc, _ = _split_hidden(h)
        h_prev = _roll_state(h)
        c_prev, h_prev_h = _split_hidden(h_prev)
        num_heads = B.shape[0]

        x_heads = x.reshape((*x.shape[:-1], num_heads, -1))
        linear = jnp.einsum('hivj,...hi->...vhj', B, x_heads)
        linear = linear.reshape((*linear.shape[:-2], -1)) + b

        Ah = jnp.einsum('vj,...j->...vj', A, h_prev_h)
        pre_f = Ah[..., 0, :] + linear[..., 0, :] + C[0, :] * c_prev
        pre_o = Ah[..., 1, :] + linear[..., 1, :] + C[1, :] * cc
        pre_c = Ah[..., 2, :] + linear[..., 2, :]

        f = nonlin_f(pre_f)
        o = nonlin_o(pre_o)
        c = nonlin_c(pre_c)

        grad_f = deriv_f(pre_f)
        grad_o = deriv_o(pre_o)
        grad_c_val = deriv_c(pre_c)

        # Partial derivatives w.r.t. h (through A)
        Jh_f = A[0, :] * grad_f
        Jh_o = A[1, :] * grad_o
        Jh_c = A[2, :] * grad_c_val

        # Partial derivatives w.r.t. c (through peephole C)
        Jc_f = C[0, :] * grad_f
        Jc_o = C[1, :] * grad_o

        o_sdercc = o * deriv_state(cc)
        scc = nonlin_state(cc)

        # Block Jacobian entries [dc/dc, dc/dh; dh/dc, dh/dh]
        Jcc = -(Jc_f * (c_prev - c) + f)
        Jch = -(Jh_f * (c_prev - c) + (1.0 - f) * Jh_c)
        Jhc = Jcc * (Jc_o * scc + o_sdercc)
        Jhh = -(scc * (Jh_o - Jc_o * Jch) - o_sdercc * Jch)

        # Pack into block-diagonal 2x2: shape (..., N, 2, 2)
        # where N = state_dim // 2
        jac = jnp.stack([
            jnp.stack([Jcc, Jch], axis=-1),
            jnp.stack([Jhc, Jhh], axis=-1),
        ], axis=-2)

        return jac

    @staticmethod
    def compute_jacobians_bwd(h, x, A, B, C, b,
                              nonlin_f, nonlin_o, nonlin_c, nonlin_state,
                              deriv_f, deriv_o, deriv_c, deriv_state):
        """Compute Jacobians for backward (transposed) system."""
        cc, _ = _split_hidden(h)
        h_prev = _roll_state(h)
        c_prev, h_prev_h = _split_hidden(h_prev)
        num_heads = B.shape[0]

        x_heads = x.reshape((*x.shape[:-1], num_heads, -1))
        linear = jnp.einsum('hivj,...hi->...vhj', B, x_heads)
        linear = linear.reshape((*linear.shape[:-2], -1)) + b

        Ah = jnp.einsum('vj,...j->...vj', A, h_prev_h)
        pre_f = Ah[..., 0, :] + linear[..., 0, :] + C[0, :] * c_prev
        pre_o = Ah[..., 1, :] + linear[..., 1, :] + C[1, :] * cc
        pre_c = Ah[..., 2, :] + linear[..., 2, :]

        f = nonlin_f(pre_f)
        o = nonlin_o(pre_o)
        c = nonlin_c(pre_c)

        grad_f = deriv_f(pre_f)
        grad_o = deriv_o(pre_o)
        grad_c_val = deriv_c(pre_c)

        Jh_f = A[0, :] * grad_f
        Jh_o = A[1, :] * grad_o
        Jh_c = A[2, :] * grad_c_val

        Jc_f = C[0, :] * grad_f
        Jc_o = C[1, :] * grad_o

        o_sdercc = o * deriv_state(cc)
        scc = nonlin_state(cc)

        Jcc = -(Jc_f * (c_prev - c) + f)
        Jch = -(Jh_f * (c_prev - c) + (1.0 - f) * Jh_c)
        Jhc = Jcc * (Jc_o * scc + o_sdercc)
        Jhh = -(scc * (Jh_o - Jc_o * Jch) - o_sdercc * Jch)

        # Transposed block: swap off-diagonal
        jacobians = jnp.stack([
            jnp.stack([Jcc, Jhc], axis=-1),  # transposed: Jch->Jhc
            jnp.stack([Jch, Jhh], axis=-1),  # transposed: Jhc->Jch
        ], axis=-2)

        # Flip time, roll, zero first
        jacobians = jnp.flip(jacobians, axis=-4)
        jacobians = jnp.roll(jacobians, shift=1, axis=-4)
        jacobians = jacobians.at[..., 0, :, :, :].set(0.0)

        return jacobians

    @staticmethod
    def linear_solve(jac, rhs):
        """Solve using block-diagonal 2x2 parallel reduction.

        Reshapes from flat (B, T, 2*N) to blocked (B, T, N, 2) form,
        runs parallel reduction, and reshapes back.
        """
        # rhs: (B, T, 2*N) -> (B, T, N, 2)
        half_n = rhs.shape[-1] // 2
        rhs_blocked = jnp.stack([rhs[..., :half_n], rhs[..., half_n:]], axis=-1)
        # jac already in (B, T, N, 2, 2) form
        h_blocked = parallel_reduce_block_diag(jac, rhs_blocked)
        # (B, T, N, 2) -> (B, T, 2*N)
        return jnp.concatenate([h_blocked[..., 0], h_blocked[..., 1]], axis=-1)

    @staticmethod
    def post_process(h, x, *params):
        """Extract h from [c, h] concatenation."""
        _, hh = _split_hidden(h)
        return hh

    @staticmethod
    def backprop_post_process(grad_y, x, h, A, B, C, b, *rest):
        """Backprop through post-processing (h extraction from [c, h])."""
        grad_h = _combine_hidden(jnp.zeros_like(grad_y), grad_y)
        return (
            grad_h,
            jnp.zeros_like(x),
            jnp.zeros_like(A),
            jnp.zeros_like(B),
            jnp.zeros_like(C),
            jnp.zeros_like(b),
            None, None, None, None,  # nonlin fns
            None, None, None, None,  # deriv fns
        )

    @staticmethod
    def backprop_to_params(dl_dh, x, h, A, B, C, b,
                           nonlin_f, nonlin_o, nonlin_c, nonlin_state,
                           deriv_f, deriv_o, deriv_c, deriv_state):
        """Compute gradients w.r.t. x and parameters."""
        cc, hh = _split_hidden(h)
        h_prev = _roll_state(h)
        c_prev, h_prev_h = _split_hidden(h_prev)
        grad_cc, grad_hh = _split_hidden(dl_dh)
        num_heads = B.shape[0]

        x_heads = x.reshape((*x.shape[:-1], num_heads, -1))
        linear = jnp.einsum('hivj,...hi->...vhj', B, x_heads)
        linear = linear.reshape((*linear.shape[:-2], -1)) + b

        Ah = jnp.einsum('vj,...j->...vj', A, h_prev_h)
        pre_f = Ah[..., 0, :] + linear[..., 0, :] + C[0, :] * c_prev
        pre_o = Ah[..., 1, :] + linear[..., 1, :] + C[1, :] * cc
        pre_c = Ah[..., 2, :] + linear[..., 2, :]

        f = nonlin_f(pre_f)
        o = nonlin_o(pre_o)
        c = nonlin_c(pre_c)

        # Gradient computation
        grad_o_gate = grad_hh * nonlin_state(cc) * deriv_o(pre_o)
        dl_dc = (grad_cc
                 + grad_hh * o * deriv_state(cc)
                 + grad_o_gate * C[1, :])

        grad_f_gate = dl_dc * (c_prev - c) * deriv_f(pre_f)
        grad_c_gate = dl_dc * (1.0 - f) * deriv_c(pre_c)

        # Stack gate grads: (..., 3, state_dim)
        grad_foc = jnp.stack([grad_f_gate, grad_o_gate, grad_c_gate], axis=-2)

        sum_axes = tuple(range(grad_foc.ndim - 2))

        grad_A = jnp.einsum('...vj,...j->vj', grad_foc, h_prev_h)
        grad_b = jnp.sum(grad_foc, axis=sum_axes)

        grad_foc_heads = grad_foc.reshape((*grad_foc.shape[:-1], num_heads, -1))
        grad_B = jnp.einsum('...vhj,...hi->hivj', grad_foc_heads, x_heads)
        grad_x = jnp.einsum('...vhj,hivj->...hi', grad_foc_heads, B)
        grad_x = grad_x.reshape((*grad_x.shape[:-2], -1))

        grad_C = jnp.stack([
            jnp.sum(grad_foc[..., 0, :] * c_prev, axis=sum_axes),
            jnp.sum(grad_foc[..., 1, :] * cc, axis=sum_axes),
        ], axis=-2)

        return (
            grad_x,
            grad_A, grad_B, grad_C, grad_b,
            None, None, None, None,  # nonlin fns
            None, None, None, None,  # deriv fns
        )

    @staticmethod
    def assemble_initial_guess(x, state_dim, *params):
        h0 = jnp.zeros((*x.shape[:-1], state_dim), dtype=x.dtype)
        return LSTMCIFGDiagMHImpl.recurrence_step(x, h0, *params)


class LSTMCIFGDiagMH(brainstate.nn.Module):
    """LSTM with Coupled Input-Forget Gate, diagonal recurrence, and multi-head
    input projection.

    The hidden state dimension passed to the constructor is the *output*
    dimension (``h``). Internally, the state is ``[c, h]`` with dimension
    ``2 * state_dim``.

    Args:
        input_dim: Input feature dimension.
        state_dim: Hidden state dimension (output dimension ``h``).
        num_heads: Number of heads for input projection.
        mode: Application mode (``'sequential'`` or ``'parallel'``).
        nonlin_f: Activation for forget gate (default: ``'sigmoid'``).
        nonlin_o: Activation for output gate (default: ``'sigmoid'``).
        nonlin_c: Activation for cell candidate (default: ``'tanh'``).
        nonlin_state: Activation for cell state (default: ``'tanh'``).
        a_init: Initialization for A weights (default: ``'xlstm'``).
        w_init: Initialization for B weights (default: ``'xavier_uniform'``).
        b_init: Initialization for bias b (default: ``'bias_minus_linspace'``).
        newton_config: Configuration for Newton solver.
        seed: Random seed for initialization.

    Example::

        >>> import jax.numpy as jnp
        >>> import brainstate
        >>> lstm = LSTMCIFGDiagMH(input_dim=32, state_dim=64, num_heads=2)
        >>> x = jnp.ones((4, 100, 32))  # (batch, time, features)
        >>> y = lstm(x)  # (4, 100, 64)  — only h is returned
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_heads: int = 2,
        mode: str = 'parallel',
        nonlin_f: str = 'sigmoid',
        nonlin_o: str = 'sigmoid',
        nonlin_c: str = 'tanh',
        nonlin_state: str = 'tanh',
        a_init: str = 'xlstm',
        w_init: str = 'xavier_uniform',
        b_init: str = 'bias_minus_linspace',
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
        self.internal_state_dim = 2 * state_dim  # [c, h]
        self.num_heads = num_heads
        self.mode = mode
        self.newton_config = newton_config or NewtonConfig()

        head_input_dim = input_dim // num_heads
        head_state_dim = state_dim // num_heads

        # Get nonlinearity functions
        self.nonlin_f, self.deriv_f = get_nonlinearity(nonlin_f)
        self.nonlin_o, self.deriv_o = get_nonlinearity(nonlin_o)
        self.nonlin_c, self.deriv_c = get_nonlinearity(nonlin_c)
        self.nonlin_state, self.deriv_state = get_nonlinearity(nonlin_state)

        # Initialize parameters
        key = jr.PRNGKey(seed)
        k1, k2, k3, k4 = jr.split(key, 4)

        A_data = INITIALIZERS[a_init](k1, (3, state_dim), fan_in=state_dim)
        B_data = INITIALIZERS[w_init](
            k2, (num_heads, head_input_dim, 3, head_state_dim),
            fan_in=head_input_dim, fan_out=head_state_dim,
        )
        C_data = INITIALIZERS[a_init](k3, (2, state_dim), fan_in=state_dim)
        b_data = INITIALIZERS[b_init](k4, (3, state_dim), fan_in=0, fan_out=3 * state_dim)

        self.A = brainstate.ParamState(A_data)
        self.B = brainstate.ParamState(B_data)
        self.C = brainstate.ParamState(C_data)
        self.b = brainstate.ParamState(b_data)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the LSTM on an input sequence.

        Args:
            x: Input tensor, shape ``(B, T, input_dim)``.

        Returns:
            Hidden states ``h``, shape ``(B, T, state_dim)``.
            (Cell states ``c`` are internal and not returned.)
        """
        return apply_rnn(
            LSTMCIFGDiagMHImpl, x, self.internal_state_dim,
            self.mode, self.newton_config,
            self.A.value, self.B.value, self.C.value, self.b.value,
            self.nonlin_f, self.nonlin_o, self.nonlin_c, self.nonlin_state,
            self.deriv_f, self.deriv_o, self.deriv_c, self.deriv_state,
        )
