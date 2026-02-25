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
Base RNN cell classes and Jacobian structure implementations.

Provides:
- ``BaseRNNCell``: Abstract base for parallel RNN cells.
- ``apply_rnn``: Main entry point for evaluating RNN cells in sequential
  or parallel mode with custom backward pass.

Concrete RNN cells (GRU, LSTM) subclass ``BaseRNNCell`` and implement:
- ``recurrence_step(x, h, *all_params) -> h_new``
- ``compute_jacobians(h, x, *all_params) -> jac``
- ``compute_jacobians_bwd(h, x, *all_params) -> jac``
- ``backprop_to_params(dl_dh, x, h, *all_params) -> (grad_x, *grad_all_params)``
- ``num_array_params`` class variable: number of leading params that are JAX arrays

All params are split into array params (differentiable) and static params
(callables like nonlinearities) to work correctly with ``jax.custom_vjp``.
"""

import abc
from enum import Enum

import jax
import jax.numpy as jnp

from ._newton import NewtonConfig, newton_solve

__all__ = ['BaseRNNCell', 'ApplicationMode', 'apply_rnn']


class ApplicationMode(str, Enum):
    """Application modes for RNN cell evaluation.

    - ``SEQUENTIAL``: O(T) sequential scan via ``jax.lax.scan``.
    - ``PARALLEL``: O(log T) parallel via Newton + ``jax.lax.associative_scan``.
    - ``FUSED``: Single CUDA kernel combining Newton + parallel reduction.
      Requires CUDA and a GPU. Falls back to ``PARALLEL`` if unavailable.
    """
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    FUSED = "fused"


def _roll_state(h: jax.Array) -> jax.Array:
    """Shift hidden states: h_prev[t] = h[t-1], h_prev[0] = 0."""
    h_prev = jnp.roll(h, shift=1, axis=-2)
    h_prev = h_prev.at[..., 0, :].set(0.0)
    return h_prev


class BaseRNNCell(abc.ABC):
    """Abstract base class for parallel RNN cells.

    Subclasses must set ``num_array_params`` to indicate how many of the
    leading ``*params`` arguments are JAX arrays (differentiable). The
    remaining params are treated as static (e.g., activation functions).
    """

    # Number of leading params that are JAX arrays
    num_array_params: int = 0

    @staticmethod
    @abc.abstractmethod
    def recurrence_step(x, h, *params):
        """Core recurrence: h_t = f(h_{t-1}, x_t; params)."""
        ...

    @staticmethod
    @abc.abstractmethod
    def compute_jacobians(h, x, *params):
        """Compute Jacobians -df/dh_{t-1} for forward solve."""
        ...

    @staticmethod
    @abc.abstractmethod
    def compute_jacobians_bwd(h, x, *params):
        """Compute Jacobians for backward (transposed) solve."""
        ...

    @staticmethod
    @abc.abstractmethod
    def backprop_to_params(dl_dh, x, h, *params):
        """Compute gradients w.r.t. x and all params.

        Returns: (grad_x, *grad_all_params) where grad for static params is None.
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def linear_solve(jac, rhs):
        """Parallel reduction for the specific Jacobian structure."""
        ...

    @staticmethod
    def post_process(h, x, *params):
        """Optional post-processing of hidden states (default: identity)."""
        return h

    @staticmethod
    def backprop_post_process(grad_y, x, h, *params):
        """Backprop through post-processing (default: pass through)."""
        return (grad_y, jnp.zeros_like(x)) + tuple(
            jnp.zeros_like(p) if isinstance(p, jax.Array) else None
            for p in params
        )

    @staticmethod
    def assemble_initial_guess(x, state_dim, *params):
        """Assemble initial guess for Newton solver (default: zeros)."""
        return jnp.zeros((*x.shape[:-1], state_dim), dtype=x.dtype)


def apply_rnn(cell_cls, x, state_dim, mode='parallel',
              newton_config=NewtonConfig(), *params):
    """Apply an RNN cell to an input sequence.

    Dispatches to sequential or parallel mode. In parallel mode, uses
    ``jax.custom_vjp`` with implicit differentiation for the backward pass.

    Args:
        cell_cls: The RNN cell class (subclass of BaseRNNCell).
        x: Input sequence, shape ``(B, T, input_dim)``.
        state_dim: Hidden state dimension.
        mode: ``'sequential'`` or ``'parallel'``.
        newton_config: Newton solver configuration (for parallel mode).
        *params: Cell parameters. The first ``cell_cls.num_array_params``
            are JAX arrays; the rest are static (e.g., activation functions).

    Returns:
        Output sequence, shape ``(B, T, output_dim)``.
    """
    n_arr = cell_cls.num_array_params
    array_params = params[:n_arr]
    static_params = params[n_arr:]

    if mode == 'sequential':
        return _apply_sequential(cell_cls, x, state_dim, array_params, static_params)
    elif mode == 'parallel':
        return _apply_parallel(cell_cls, x, state_dim, newton_config, array_params, static_params)
    elif mode == 'fused':
        # Fused mode is handled by the cell's Module class (GRUDiagMH, etc.)
        # If called directly here, fall back to parallel mode.
        return _apply_parallel(cell_cls, x, state_dim, newton_config, array_params, static_params)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Available: 'sequential', 'parallel', 'fused'."
        )


def _apply_sequential(cell_cls, x, state_dim, array_params, static_params):
    """Sequential RNN evaluation via jax.lax.scan."""
    all_params = array_params + static_params

    def scan_fn(h_prev, x_t):
        h_new = cell_cls.recurrence_step(x_t, h_prev, *all_params)
        return h_new, h_new

    h0 = jnp.zeros((*x.shape[:-2], state_dim), dtype=x.dtype)
    x_scan = jnp.moveaxis(x, -2, 0)
    _, h_scan = jax.lax.scan(scan_fn, h0, x_scan)
    h = jnp.moveaxis(h_scan, 0, -2)
    return cell_cls.post_process(h, x, *all_params)


def _apply_parallel(cell_cls, x, state_dim, newton_config, array_params, static_params):
    """Parallel RNN via Newton + associative_scan with custom_vjp."""

    # static_params are closed over, not passed through custom_vjp

    @jax.custom_vjp
    def _forward(x, *array_params):
        all_params = array_params + static_params
        y, _ = _parallel_fwd(cell_cls, x, state_dim, newton_config, all_params)
        return y

    def _forward_fwd(x, *array_params):
        all_params = array_params + static_params
        y, (_, h) = _parallel_fwd(cell_cls, x, state_dim, newton_config, all_params)
        return y, (x, h, array_params)

    def _forward_bwd(res, grad_y):
        x_res, h, arr_params_res = res
        all_params = arr_params_res + static_params
        n_arr = len(arr_params_res)

        # 1. Backprop through post-processing
        all_grads_pp = cell_cls.backprop_post_process(grad_y, x_res, h, *all_params)
        grad_h = all_grads_pp[0]
        grad_x_pp = all_grads_pp[1]
        grad_all_params_pp = all_grads_pp[2:]

        # 2. Solve transposed system for dl/dh
        rhs = jnp.flip(grad_h, axis=-2)
        jac_bwd = cell_cls.compute_jacobians_bwd(h, x_res, *all_params)
        dl_dh = jnp.flip(cell_cls.linear_solve(jac_bwd, rhs), axis=-2)

        # 3. Backprop to parameters
        all_grads_rec = cell_cls.backprop_to_params(dl_dh, x_res, h, *all_params)
        grad_x_rec = all_grads_rec[0]
        grad_all_params_rec = all_grads_rec[1:]

        # 4. Combine gradients
        grad_x = grad_x_pp + grad_x_rec

        # Only return gradients for array params (first n_arr)
        grad_arr_params = []
        for i in range(n_arr):
            g_pp = grad_all_params_pp[i] if i < len(grad_all_params_pp) else None
            g_rec = grad_all_params_rec[i] if i < len(grad_all_params_rec) else None
            if g_pp is None and g_rec is None:
                grad_arr_params.append(jnp.zeros_like(arr_params_res[i]))
            elif g_pp is None:
                grad_arr_params.append(g_rec)
            elif g_rec is None:
                grad_arr_params.append(g_pp)
            else:
                grad_arr_params.append(g_pp + g_rec)

        return (grad_x, *grad_arr_params)

    _forward.defvjp(_forward_fwd, _forward_bwd)
    return _forward(x, *array_params)


def _parallel_fwd(cell_cls, x, state_dim, newton_config, all_params):
    """Forward pass of parallel RNN via Newton + associative_scan."""
    h0 = cell_cls.assemble_initial_guess(x, state_dim, *all_params)

    def compute_neg_residuals(h):
        h_prev = _roll_state(h)
        h_new = cell_cls.recurrence_step(x, h_prev, *all_params)
        return -(h - h_new)

    def compute_jacs(h):
        return cell_cls.compute_jacobians(h, x, *all_params)

    h = newton_solve(
        h0, compute_neg_residuals, compute_jacs,
        cell_cls.linear_solve, newton_config,
    )

    y = cell_cls.post_process(h, x, *all_params)
    return y, (x, h)
