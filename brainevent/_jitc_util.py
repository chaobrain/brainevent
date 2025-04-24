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

from typing import Optional, Sequence

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.interpreters import ad

from ._typing import Kernel, Data, MatrixShape
from ._xla_custom_op import XLACustomKernel
from ._xla_custom_op_numba import NumbaKernelGenerator, numba_environ
from ._xla_custom_op_warp import dtype_to_warp_type, WarpKernelGenerator


def _initialize_seed(seed=None):
    """Initialize a random seed for JAX operations.

    This function ensures a consistent format for random seeds used in JAX operations.
    If no seed is provided, it generates a random integer between 0 and 10^8 at compile time,
    ensuring reproducibility within compiled functions.

    Parameters
    ----------
    seed : int or array-like, optional
        The random seed to use. If None, a random seed is generated.

    Returns
    -------
    jax.Array
        A JAX array containing the seed value(s) with int32 dtype, ensuring it's
        in a format compatible with JAX random operations.

    Notes
    -----
    The function uses `jax.ensure_compile_time_eval()` to guarantee that random
    seed generation happens during compilation rather than during execution when
    no seed is provided, which helps maintain consistency across multiple calls
    to a JIT-compiled function.
    """
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), (1,))
    return jnp.asarray(jnp.atleast_1d(seed), dtype=jnp.int32)


def _initialize_conn_length(conn_prob: float):
    """
    Convert connection probability to connection length parameter for sparse matrix generation.

    This function transforms a connection probability (proportion of non-zero entries)
    into a connection length parameter used by the sparse sampling algorithms.
    The connection length is approximately the inverse of the connection probability,
    scaled by a factor of 2 to ensure adequate sparsity in the generated matrices.

    The function ensures the calculation happens at compile time when used in JIT-compiled
    functions by using JAX's compile_time_eval context.

    Parameters
    ----------
    conn_prob : float
        The connection probability (between 0 and 1) representing the fraction
        of non-zero entries in the randomly generated matrix.

    Returns
    -------
    jax.Array
        A JAX array containing the connection length value as an int32,
        which is approximately 2/conn_prob.

    Notes
    -----
    The connection length parameter is used in the kernels to determine the
    average distance between sampled connections when generating sparse matrices.
    Larger values result in sparser matrices (fewer connections).
    """
    with jax.ensure_compile_time_eval():
        clen = jnp.ceil(2 / conn_prob)
        clen = jnp.asarray(clen, dtype=jnp.int32)
    return clen


