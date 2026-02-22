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
General-purpose parallel RNN API.

Provides ``parallel_rnn()`` — a single function that enables parallel training
of **any** RNN cell, with automatic Jacobian computation, backward pass, and
parameter gradients.

Usage::

    import jax
    import jax.numpy as jnp
    from brainevent.pararnn import parallel_rnn

    def elman_rnn(h_prev, x_t, W_h, W_x, b):
        return jax.nn.tanh(W_h @ h_prev + W_x @ x_t + b)

    y = parallel_rnn(elman_rnn, x, W_h, W_x, b)

The cell function ``cell_fn(h_prev, x_t, *params) -> h_new`` operates on
single vectors (no batch/time dimensions). Vectorization over batch and time
is handled automatically via ``jax.vmap``.

Three Jacobian structures are supported:
- ``'diagonal'``: element-wise gating, O(N log T)
- ``'block_diagonal'``: grouped hidden states, O(NK^2 log T)
- ``'dense'``: general cells, O(N^3 log T)
- ``'auto'``: auto-detect from the cell function (default)
"""

import jax
import jax.numpy as jnp

from ._cell import _roll_state, apply_rnn, BaseRNNCell
from ._newton import NewtonConfig
from ._parallel_reduce import (
    parallel_reduce_diag,
    parallel_reduce_block_diag,
    parallel_reduce_dense,
)

__all__ = ['parallel_rnn', 'AutoRNNCell']


# =============================================================================
# Jacobian structure detection
# =============================================================================

def _detect_jacobian_structure(cell_fn, state_dim, input_dim, params):
    """Detect Jacobian structure by evaluating one Jacobian.

    Creates fresh concrete arrays (independent of any JIT trace) for the
    detection probe, so this is safe to call inside or outside JIT.

    Returns:
        (structure, block_size) where structure is one of
        'diagonal', 'block_diagonal', 'dense' and block_size is the
        block size K (only meaningful for 'block_diagonal').
    """
    import numpy as np

    # Create concrete arrays with small random values (not zeros, because
    # zero params give zero Jacobians which are trivially diagonal).
    # Use jax.ensure_compile_time_eval() so this runs eagerly even inside JIT.
    with jax.ensure_compile_time_eval():
        key = jax.random.PRNGKey(12345)
        h_test = jax.random.normal(key, (state_dim,)) * 0.01
        x_test = jax.random.normal(jax.random.PRNGKey(12346), (input_dim,)) * 0.01
        params_test = tuple(
            jax.random.normal(
                jax.random.PRNGKey(12347 + i), p.shape
            ) * 0.01
            if hasattr(p, 'shape') else p
            for i, p in enumerate(params)
        )
        J = jax.jacobian(cell_fn, argnums=0)(h_test, x_test, *params_test)
    J_np = np.asarray(J)

    # Check diagonal
    off_diag = J_np - np.diag(np.diag(J_np))
    if np.allclose(off_diag, 0.0, atol=1e-12):
        return 'diagonal', 1

    # Check block-diagonal (need at least 2 blocks to be meaningful)
    N = state_dim
    for K in (2, 3, 4):
        if N % K != 0:
            continue
        n_blocks = N // K
        if n_blocks < 2:
            continue  # single block = dense, not block-diagonal
        is_block_diag = True
        for i in range(n_blocks):
            for j in range(n_blocks):
                if i == j:
                    continue
                block = J_np[i * K:(i + 1) * K, j * K:(j + 1) * K]
                if not np.allclose(block, 0.0, atol=1e-12):
                    is_block_diag = False
                    break
            if not is_block_diag:
                break
        if is_block_diag:
            return 'block_diagonal', K

    return 'dense', state_dim


def _to_structured_jac(J_full, structure, block_size):
    """Convert full N×N Jacobian to structure-specific format.

    Args:
        J_full: Full Jacobian, shape ``(B, T, N, N)``.
        structure: One of 'diagonal', 'block_diagonal', 'dense'.
        block_size: Block size K for block_diagonal.

    Returns:
        Structured Jacobian in the appropriate format.
    """
    if structure == 'diagonal':
        # (B, T, N, N) -> extract diagonal -> (B, T, N)
        return jnp.diagonal(J_full, axis1=-2, axis2=-1)
    elif structure == 'block_diagonal':
        # (B, T, N, N) -> (B, T, N//K, K, K)
        B, T, N, _ = J_full.shape
        K = block_size
        n_blocks = N // K
        # Reshape to (B, T, n_blocks, K, n_blocks, K) then extract diagonal blocks.
        # Note: fancy indexing with non-contiguous advanced indices would place the
        # advanced axis first, so we use jnp.stack with explicit per-block slicing.
        J_reshaped = J_full.reshape(B, T, n_blocks, K, n_blocks, K)
        blocks = jnp.stack(
            [J_reshaped[:, :, i, :, i, :] for i in range(n_blocks)],
            axis=2,
        )  # (B, T, n_blocks, K, K)
        return blocks
    else:
        # Dense: keep as-is (B, T, N, N)
        return J_full


def _transpose_jac(jac, structure):
    """Transpose Jacobian for backward solve."""
    if structure == 'diagonal':
        return jac  # scalar: transpose is identity
    elif structure == 'block_diagonal':
        return jnp.swapaxes(jac, -1, -2)
    else:
        return jnp.swapaxes(jac, -1, -2)


# =============================================================================
# AutoRNNCell
# =============================================================================

class AutoRNNCell(BaseRNNCell):
    """RNN cell with auto-derived Jacobians and gradients.

    Wraps an arbitrary cell function ``cell_fn(h_prev, x_t, *params) -> h_new``
    and automatically derives all methods required for parallel training:

    - Jacobians via ``jax.jacobian``
    - Backward Jacobians via transpose + time reversal
    - Parameter gradients via ``jax.vjp``
    - Linear solver dispatch based on Jacobian structure

    The cell function operates on single vectors (no batch/time dims).

    Args:
        cell_fn: Recurrence function ``(h_prev, x_t, *params) -> h_new``.
        jacobian_structure: ``'auto'``, ``'diagonal'``, ``'block_diagonal'``,
            or ``'dense'``.
        block_size: Block size K for ``'block_diagonal'`` structure.
            Required when ``jacobian_structure='block_diagonal'``.
            Ignored for other structures.
    """

    def __init__(self, cell_fn, jacobian_structure='auto', block_size=None):
        self.cell_fn = cell_fn
        self._jac_structure = jacobian_structure
        self._block_size = block_size
        self._resolved = False
        self._structure = None
        self._K = None

    def _resolve_structure(self, state_dim, input_dim, params):
        """Lazily resolve Jacobian structure on first call."""
        if self._resolved:
            return
        if self._jac_structure == 'auto':
            self._structure, self._K = _detect_jacobian_structure(
                self.cell_fn, state_dim, input_dim, params
            )
        elif self._jac_structure == 'diagonal':
            self._structure = 'diagonal'
            self._K = 1
        elif self._jac_structure == 'block_diagonal':
            self._structure = 'block_diagonal'
            if self._block_size is None:
                raise ValueError(
                    "block_size is required when jacobian_structure='block_diagonal'"
                )
            self._K = self._block_size
        elif self._jac_structure == 'dense':
            self._structure = 'dense'
            self._K = state_dim
        else:
            raise ValueError(f"Unknown jacobian_structure: {self._jac_structure}")
        self._resolved = True

    # We set num_array_params dynamically based on actual params count.
    # This is done via a property-like approach: the apply_rnn function
    # reads cell_cls.num_array_params to split array vs static params.
    # For AutoRNNCell, ALL params are arrays.
    num_array_params = 0  # Will be set dynamically

    def recurrence_step(self, x, h, *params):
        """Vectorized recurrence step."""
        cell_fn = self.cell_fn
        h_prev = _roll_state(h)

        # vmap over batch and time: cell_fn operates on single vectors
        def step_single(h_prev_bt, x_bt):
            return cell_fn(h_prev_bt, x_bt, *params)

        return jax.vmap(jax.vmap(step_single))(h_prev, x)

    def compute_jacobians(self, h, x, *params):
        """Auto-Jacobian via jax.jacobian, converted to structure format."""
        cell_fn = self.cell_fn
        h_prev = _roll_state(h)
        state_dim = h.shape[-1]
        input_dim = x.shape[-1]
        self._resolve_structure(state_dim, input_dim, params)

        def jac_single(h_prev_bt, x_bt):
            return jax.jacobian(cell_fn, argnums=0)(h_prev_bt, x_bt, *params)

        # Full Jacobian: (B, T, N, N)
        J_full = jax.vmap(jax.vmap(jac_single))(h_prev, x)
        # Negate and convert to structured format
        return _to_structured_jac(-J_full, self._structure, self._K)

    def compute_jacobians_bwd(self, h, x, *params):
        """Backward Jacobians: negate + transpose + flip + shift.

        The forward Jacobians are ``J = -dF/dh_prev``. The backward adjoint
        system requires ``dF/dh_prev^T = -J^T``, so we negate before
        transposing.
        """
        jac = self.compute_jacobians(h, x, *params)
        # Forward jac = -dF/dh_prev. Backward needs +dF/dh_prev^T = -J^T.
        jac_T = _transpose_jac(-jac, self._structure)

        # Determine T axis
        if self._structure == 'diagonal':
            T_axis = -2  # (B, T, N)
            jac_T = jnp.flip(jac_T, axis=T_axis)
            jac_T = jnp.roll(jac_T, shift=1, axis=T_axis)
            jac_T = jac_T.at[..., 0, :].set(0.0)
        elif self._structure == 'block_diagonal':
            T_axis = -4  # (B, T, n_blocks, K, K)
            jac_T = jnp.flip(jac_T, axis=T_axis)
            jac_T = jnp.roll(jac_T, shift=1, axis=T_axis)
            jac_T = jac_T.at[..., 0, :, :, :].set(0.0)
        else:  # dense
            T_axis = 1  # (B, T, N, N)
            jac_T = jnp.flip(jac_T, axis=T_axis)
            jac_T = jnp.roll(jac_T, shift=1, axis=T_axis)
            jac_T = jac_T.at[:, 0].set(0.0)

        return jac_T

    def backprop_to_params(self, dl_dh, x, h, *params):
        """Parameter gradients via jax.vjp."""
        cell_fn = self.cell_fn
        h_prev = _roll_state(h)

        # vjp of cell_fn w.r.t. (x, *params), treating h_prev as given
        def cell_for_vjp(x_val, *p):
            def step_single(h_prev_bt, x_bt):
                return cell_fn(h_prev_bt, x_bt, *p)
            return jax.vmap(jax.vmap(step_single))(h_prev, x_val)

        _, vjp_fn = jax.vjp(cell_for_vjp, x, *params)
        all_grads = vjp_fn(dl_dh)
        grad_x = all_grads[0]
        grad_params = all_grads[1:]

        return (grad_x, *grad_params)

    def linear_solve(self, jac, rhs):
        """Dispatch to appropriate parallel reduction."""
        if self._structure == 'diagonal':
            return parallel_reduce_diag(jac, rhs)
        elif self._structure == 'block_diagonal':
            # rhs: (B, T, N) -> (B, T, N//K, K)
            K = self._K
            half_n = rhs.shape[-1] // K
            rhs_blocked = rhs.reshape(*rhs.shape[:-1], half_n, K)
            h_blocked = parallel_reduce_block_diag(jac, rhs_blocked)
            return h_blocked.reshape(*h_blocked.shape[:-2], -1)
        else:
            return parallel_reduce_dense(jac, rhs)

    def post_process(self, h, x, *params):
        """Identity (no post-processing by default)."""
        return h

    def backprop_post_process(self, grad_y, x, h, *params):
        """Pass-through (no post-processing)."""
        return (grad_y, jnp.zeros_like(x)) + tuple(
            jnp.zeros_like(p) if isinstance(p, jax.Array) else None
            for p in params
        )

    def assemble_initial_guess(self, x, state_dim, *params):
        """Initial guess: one step from zeros."""
        h0 = jnp.zeros((*x.shape[:-1], state_dim), dtype=x.dtype)
        return self.recurrence_step(x, h0, *params)


# =============================================================================
# Helper: infer state dimension
# =============================================================================

def _infer_state_dim(cell_fn, x, params):
    """Infer hidden state dimension by calling cell_fn with abstract inputs.

    Uses ``jax.eval_shape`` to avoid materializing concrete arrays, making
    this compatible with JIT tracing.
    """
    input_dim = x.shape[-1]

    # Try candidate state dims from parameter shapes
    candidates = set()
    for p in params:
        if hasattr(p, 'shape'):
            for d in p.shape:
                if d > 0:
                    candidates.add(int(d))

    for candidate in sorted(candidates):
        try:
            out_shape = jax.eval_shape(
                cell_fn,
                jax.ShapeDtypeStruct((candidate,), jnp.float32),
                jax.ShapeDtypeStruct((input_dim,), jnp.float32),
                *[jax.ShapeDtypeStruct(p.shape, p.dtype) if hasattr(p, 'shape')
                  else p for p in params],
            )
            if out_shape.shape == (candidate,):
                return candidate
        except Exception:
            pass

    raise ValueError(
        "Could not infer state_dim from cell function and parameters. "
        "Please pass state_dim explicitly."
    )


# =============================================================================
# Main API
# =============================================================================

def _apply_parallel_auto(cell, x, state_dim, newton_config, params):
    """Parallel RNN via Newton + associative_scan with custom_vjp.

    This is a specialized version of ``_apply_parallel`` from ``_cell.py``
    that works with ``AutoRNNCell`` instances (non-static methods).
    """
    @jax.custom_vjp
    def _forward(x, *array_params):
        h0 = cell.assemble_initial_guess(x, state_dim, *array_params)

        def compute_neg_residuals(h):
            h_new = cell.recurrence_step(x, h, *array_params)
            return -(h - h_new)

        def compute_jacs(h):
            return cell.compute_jacobians(h, x, *array_params)

        from ._newton import newton_solve
        h = newton_solve(
            h0, compute_neg_residuals, compute_jacs,
            cell.linear_solve, newton_config,
        )
        return cell.post_process(h, x, *array_params)

    def _forward_fwd(x, *array_params):
        h0 = cell.assemble_initial_guess(x, state_dim, *array_params)

        def compute_neg_residuals(h):
            h_new = cell.recurrence_step(x, h, *array_params)
            return -(h - h_new)

        def compute_jacs(h):
            return cell.compute_jacobians(h, x, *array_params)

        from ._newton import newton_solve
        h = newton_solve(
            h0, compute_neg_residuals, compute_jacs,
            cell.linear_solve, newton_config,
        )
        y = cell.post_process(h, x, *array_params)
        return y, (x, h, array_params)

    def _forward_bwd(res, grad_y):
        x_res, h, arr_params = res
        n_arr = len(arr_params)

        # 1. Backprop through post-processing
        all_grads_pp = cell.backprop_post_process(
            grad_y, x_res, h, *arr_params
        )
        grad_h = all_grads_pp[0]
        grad_x_pp = all_grads_pp[1]
        grad_all_params_pp = all_grads_pp[2:]

        # 2. Solve transposed system for dl/dh
        rhs = jnp.flip(grad_h, axis=-2)
        jac_bwd = cell.compute_jacobians_bwd(h, x_res, *arr_params)
        dl_dh = jnp.flip(
            cell.linear_solve(jac_bwd, rhs),
            axis=-2,
        )

        # 3. Backprop to parameters
        all_grads_rec = cell.backprop_to_params(
            dl_dh, x_res, h, *arr_params,
        )
        grad_x_rec = all_grads_rec[0]
        grad_all_params_rec = all_grads_rec[1:]

        # 4. Combine gradients
        grad_x = grad_x_pp + grad_x_rec

        grad_arr_params = []
        for i in range(n_arr):
            g_pp = grad_all_params_pp[i] if i < len(grad_all_params_pp) else None
            g_rec = grad_all_params_rec[i] if i < len(grad_all_params_rec) else None
            if g_pp is None and g_rec is None:
                grad_arr_params.append(jnp.zeros_like(arr_params[i]))
            elif g_pp is None:
                grad_arr_params.append(g_rec)
            elif g_rec is None:
                grad_arr_params.append(g_pp)
            else:
                grad_arr_params.append(g_pp + g_rec)

        return (grad_x, *grad_arr_params)

    _forward.defvjp(_forward_fwd, _forward_bwd)
    return _forward(x, *params)


def _apply_sequential_auto(cell, x, state_dim, params):
    """Sequential RNN evaluation via jax.lax.scan.

    The cell_fn operates on single vectors, so we vmap over the batch
    dimension inside the scan.
    """
    cell_fn = cell.cell_fn

    # Vectorize cell_fn over the batch dimension
    batched_cell_fn = jax.vmap(
        lambda h, xt: cell_fn(h, xt, *params)
    )

    def scan_fn(h_prev, x_t):
        # h_prev: (B, N), x_t: (B, D)
        h_new = batched_cell_fn(h_prev, x_t)
        return h_new, h_new

    h0 = jnp.zeros((*x.shape[:-2], state_dim), dtype=x.dtype)
    x_scan = jnp.moveaxis(x, -2, 0)  # (T, B, D)
    _, h_scan = jax.lax.scan(scan_fn, h0, x_scan)
    h = jnp.moveaxis(h_scan, 0, -2)  # (B, T, N)
    return h


def parallel_rnn(
    cell_fn,
    x,
    *params,
    jacobian_structure='auto',
    block_size=None,
    state_dim=None,
    newton_config=None,
    mode='parallel',
):
    """Train any RNN cell in parallel via Newton + parallel scan.

    Given a cell function ``cell_fn(h_prev, x_t, *params) -> h_new``,
    automatically derives Jacobians, backward pass, and parameter gradients
    for O(log T) parallel training.

    The cell function operates on **single vectors** (no batch/time dims):
    - ``h_prev``: shape ``(state_dim,)``
    - ``x_t``: shape ``(input_dim,)``
    - returns: ``h_new`` with shape ``(state_dim,)``

    Args:
        cell_fn: Recurrence function ``(h_prev, x_t, *params) -> h_new``.
        x: Input sequence, shape ``(B, T, input_dim)``.
        *params: Cell parameters (JAX arrays).
        jacobian_structure: ``'auto'`` (default), ``'diagonal'``,
            ``'block_diagonal'``, or ``'dense'``.
        block_size: Block size K for ``'block_diagonal'`` structure.
        state_dim: Hidden state dimension. If ``None``, inferred from
            ``cell_fn`` and ``params``.
        newton_config: Newton solver configuration.
        mode: ``'sequential'`` or ``'parallel'`` (default).

    Returns:
        Hidden states, shape ``(B, T, state_dim)``.
    """
    if newton_config is None:
        newton_config = NewtonConfig()

    cell = AutoRNNCell(cell_fn, jacobian_structure, block_size)

    # Infer state_dim if not provided
    if state_dim is None:
        state_dim = _infer_state_dim(cell_fn, x, params)

    # Resolve Jacobian structure eagerly for parallel mode
    if mode == 'parallel':
        cell._resolve_structure(state_dim, x.shape[-1], params)
        cell.num_array_params = len(params)
        return _apply_parallel_auto(cell, x, state_dim, newton_config, params)
    elif mode == 'sequential':
        return _apply_sequential_auto(cell, x, state_dim, params)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'sequential' or 'parallel'.")
