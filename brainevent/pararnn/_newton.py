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
Newton solver for parallel RNN training.

Reformulates the recurrence h[t] = f(h[t-1], x[t]; theta) as a root-finding
problem F(h) = 0 and solves via Newton iteration. Each Newton step requires
solving a lower bidiagonal linear system via parallel reduction.

The backward pass uses the implicit function theorem: the transposed bidiagonal
system is solved (in reversed time) to propagate gradients, avoiding
differentiation through Newton iterations.
"""

from dataclasses import dataclass
from typing import Callable

import jax

__all__ = ['NewtonConfig', 'newton_solve']


@dataclass
class NewtonConfig:
    """Configuration for the Newton solver.

    Attributes:
        max_its: Maximum number of Newton iterations.
        omega_sor: Successive-over-relaxation parameter.
            < 1: stabilizing, = 1: vanilla Newton, > 1: accelerating.
    """
    max_its: int = 3
    omega_sor: float = 1.0


def newton_solve(
    h0: jax.Array,
    compute_negative_residuals: Callable,
    compute_jacobians: Callable,
    linear_solve: Callable,
    newton_config: NewtonConfig = NewtonConfig(),
) -> jax.Array:
    """Solve F(h) = 0 via Newton iteration with parallel reduction.

    At each iteration:
        1. Compute residuals: rhs = -(h - f(h_prev, x))
        2. Compute Jacobians: J = -df/dh_prev
        3. Solve bidiagonal system: dh = linear_solve(J, rhs)
        4. Update: h = h + omega * dh

    Uses ``jax.lax.fori_loop`` for JIT compatibility.

    Args:
        h0: Initial guess for hidden states, shape ``(B, T, N)``.
        compute_negative_residuals: ``h -> rhs``, computes negative residuals.
        compute_jacobians: ``h -> jac``, computes Jacobians.
        linear_solve: ``(jac, rhs) -> dh``, solves the bidiagonal system.
        newton_config: Solver configuration.

    Returns:
        Solution ``h`` with shape ``(B, T, N)``.
    """
    omega = newton_config.omega_sor

    def body(_, h):
        rhs = compute_negative_residuals(h)
        jac = compute_jacobians(h)
        dh = linear_solve(jac, rhs)
        return h + omega * dh

    return jax.lax.fori_loop(0, newton_config.max_its, body, h0)
