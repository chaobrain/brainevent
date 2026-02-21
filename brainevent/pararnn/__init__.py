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
Parallel RNN training via Newton's method and parallel prefix reduction.

Based on ml-pararnn (arXiv:2510.21450). Provides GRU and LSTM cells that
can be trained in O(log T) parallel depth instead of O(T) sequential steps.

Modules:
    GRUDiagMH: Diagonal GRU with multi-head input projection.
    LSTMCIFGDiagMH: LSTM with coupled input-forget gate, diagonal recurrence,
        and multi-head input projection.

Functions:
    parallel_reduce_diag: Parallel prefix solve for diagonal Jacobians.
    parallel_reduce_block_diag: Parallel prefix solve for block-diagonal Jacobians.
    newton_solve: Newton iteration with configurable linear solver.

Registries:
    NONLINEARITIES: Activation functions and their analytical derivatives.
    INITIALIZERS: Weight initialization strategies.
"""

from ._cell import BaseRNNCell, ApplicationMode, apply_rnn
from ._gru import GRUDiagMH
from ._init import INITIALIZERS, initialize
from ._lstm import LSTMCIFGDiagMH
from ._newton import NewtonConfig, newton_solve
from ._nonlinearities import NONLINEARITIES, get_nonlinearity
from ._parallel_reduce import (
    parallel_reduce_diag,
    parallel_reduce_diag_bwd,
    parallel_reduce_block_diag,
    parallel_reduce_block_diag_bwd,
)
from ._parallel_reduce_cuda import (
    parallel_reduce_diag_cuda,
    parallel_reduce_block2_cuda,
    cuda_available,
)
from ._fused_cuda import (
    fused_gru_diag_forward,
    fused_gru_diag_backward,
    fused_lstm_cifg_diag_forward,
    fused_lstm_cifg_diag_backward,
    fused_cuda_available,
)

__all__ = [
    # Modules
    'GRUDiagMH',
    'LSTMCIFGDiagMH',

    # Base classes
    'BaseRNNCell',
    'ApplicationMode',

    # Application
    'apply_rnn',

    # Newton solver
    'NewtonConfig',
    'newton_solve',

    # Parallel reduction (JAX native)
    'parallel_reduce_diag',
    'parallel_reduce_diag_bwd',
    'parallel_reduce_block_diag',
    'parallel_reduce_block_diag_bwd',

    # Parallel reduction (CUDA)
    'parallel_reduce_diag_cuda',
    'parallel_reduce_block2_cuda',
    'cuda_available',

    # Fused CUDA kernels
    'fused_gru_diag_forward',
    'fused_gru_diag_backward',
    'fused_lstm_cifg_diag_forward',
    'fused_lstm_cifg_diag_backward',
    'fused_cuda_available',

    # Registries
    'NONLINEARITIES',
    'INITIALIZERS',
    'get_nonlinearity',
    'initialize',
]
