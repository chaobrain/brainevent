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


import sys
import time

sys.path.append('..')

import brainstate
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.sparse import COO
from scipy.io import mmread
from scipy.sparse import csr_matrix, coo_matrix

import brainevent

# brainstate.environ.set(platform='cpu')
brainstate.environ.set(platform='gpu')

files = [
    'matrices/suitesparse/Andrianov/mip1/mip1.mtx',
    'matrices/suitesparse/Bova/rma10/rma10.mtx',
    'matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx',
    'matrices/suitesparse/IBM_EDA/dc2/dc2.mtx',
    # 'matrices/suitesparse/QCD/conf5_4-8x8-05/conf5_4-8x8-05.mtx',
    'matrices/suitesparse/Williams/cant/cant.mtx',
    'matrices/suitesparse/Williams/consph/consph.mtx',
    'matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx',
    'matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx',
]

coo_matrices = dict()
for filename in files:
    print(f'Loading {filename} ...')
    mat = mmread(filename)
    if isinstance(mat, coo_matrix):
        coo_matrices[filename] = mat
    if isinstance(mat, csr_matrix):
        coo_matrices[filename] = mat.tocoo()
print()


def visualization(results, title: str):
    filenames = list(results.keys())
    ratios = np.asarray(list(results.values())) - 1.0

    plt.figure(figsize=(10, 6))
    plt.barh(filenames, ratios, color='skyblue')
    plt.xlabel('BrainEvent / JAX Accleration Ratio')
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)  # 显示x轴刻度并旋转45度
    plt.tight_layout()
    plt.show()


def _compare_spmv_performance(filename, n_run: int = 10):
    scipy_coo = coo_matrices[filename]
    data = jnp.asarray(scipy_coo.data)
    row = jnp.asarray(scipy_coo.row)
    col = jnp.asarray(scipy_coo.col)

    jax_coo = COO((data, row, col), shape=scipy_coo.shape)
    brainevent_coo = brainevent.COO((data, row, col), shape=scipy_coo.shape)

    @jax.jit
    def f_jax(v):
        return jax_coo @ v

    @jax.jit
    def f_brainevent(v):
        return brainevent_coo @ v

    vector = jax.block_until_ready(jnp.ones(scipy_coo.shape[1]))

    r1 = jax.block_until_ready(f_jax(vector))
    r2 = jax.block_until_ready(f_brainevent(vector))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(vector))
    t1 = time.time()
    t_jax_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, JAX  COO @ Vector:       {t_jax_csr_vector:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(vector))
    t1 = time.time()
    t_be_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, BrainEvent COO @ Vector: {t_be_csr_vector:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_csr_vector / t_be_csr_vector}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    @jax.jit
    def f_jax(v):
        return jax_coo.T @ v

    @jax.jit
    def f_brainevent(v):
        return v @ brainevent_coo

    vector = jax.block_until_ready(jnp.ones(scipy_coo.shape[0]))

    r1 = jax.block_until_ready(f_jax(vector))
    r2 = jax.block_until_ready(f_brainevent(vector))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(vector))
    t1 = time.time()
    t_jax_vector_csr = (t1 - t0) / n_run
    print(f"{filename},    JAX  Vector @ COO :   {t_jax_vector_csr:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(vector))
    t1 = time.time()
    t_be_vector_csr = (t1 - t0) / n_run
    print(f"{filename}, BrainEvent Vector @ COO: {t_be_vector_csr:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_vector_csr / t_be_vector_csr}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    return t_jax_vector_csr / t_be_vector_csr


def evaluate_spmv_performance():
    results = dict()
    for filename in files:
        results[filename] = _compare_spmv_performance(
            filename, n_run=3 if brainstate.environ.get_platform() == 'cpu' else 30
        )
    title = 'Intel-i9-12900H-SpMV-COO' if brainstate.environ.get_platform() == 'cpu' else 'RTX-3080Ti-SpMV-COO'
    visualization(results, title=title)


def compare_spmm_performance(
    scipy_coo: coo_matrix,
    n_run: int = 10,
    batch_size: int = 100
):
    data = jnp.asarray(scipy_coo.data)
    row = jnp.asarray(scipy_coo.row)
    col = jnp.asarray(scipy_coo.col)

    jax_coo = COO((data, row, col), shape=scipy_coo.shape)
    brainevent_coo = brainevent.COO((data, row, col), shape=scipy_coo.shape)

    @jax.jit
    def f_jax(v):
        return jax_coo @ v

    @jax.jit
    def f_brainevent(v):
        return brainevent_coo @ v

    matrix = jax.block_until_ready(brainstate.random.randn(scipy_coo.shape[1], batch_size))

    r1 = jax.block_until_ready(f_jax(matrix))
    r2 = jax.block_until_ready(f_brainevent(matrix))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(matrix))
    t1 = time.time()
    t_jax_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, JAX  COO @ Vector:       {t_jax_csr_vector:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(matrix))
    t1 = time.time()
    t_be_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, BrainEvent COO @ Vector: {t_be_csr_vector:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_csr_vector / t_be_csr_vector}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    @jax.jit
    def f_jax(v):
        return (jax_coo.T @ v.T).T

    @jax.jit
    def f_brainevent(v):
        return v @ brainevent_coo

    matrix = jax.block_until_ready(brainstate.random.randn(batch_size, scipy_coo.shape[0]))

    r1 = jax.block_until_ready(f_jax(matrix))
    r2 = jax.block_until_ready(f_brainevent(matrix))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(matrix))
    t1 = time.time()
    t_jax_vector_csr = (t1 - t0) / n_run
    print(f"{filename},    JAX  Vector @ COO :   {t_jax_vector_csr:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(matrix))
    t1 = time.time()
    t_be_vector_csr = (t1 - t0) / n_run
    print(f"{filename}, BrainEvent Vector @ COO: {t_be_vector_csr:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_vector_csr / t_be_vector_csr}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    return t_jax_vector_csr / t_be_vector_csr


def evaluate_spmm_performance():
    results = dict()
    for filename in files:
        results[filename] = compare_spmm_performance(
            coo_matrices[filename], n_run=3 if brainstate.environ.get_platform() == 'cpu' else 30
        )
    title = 'Intel-i9-12900H-SpMM-COO' if brainstate.environ.get_platform() == 'cpu' else 'RTX-3080Ti-SpMM-COO'
    visualization(results, title=title)


evaluate_spmv_performance()
evaluate_spmm_performance()
