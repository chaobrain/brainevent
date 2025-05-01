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

import os

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
import time
import sys

sys.path.append('..')

import brainstate
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.sparse import CSR
from scipy.io import mmread
from scipy.sparse import csr_matrix, coo_matrix

import brainevent

# brainstate.environ.set(platform='cpu')
# brainstate.environ.set(platform='gpu')

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

csr_matrices = dict()
for filename in files:
    print(f'Loading {filename} ...')
    mat = mmread(filename)
    if isinstance(mat, coo_matrix):
        csr_matrices[filename] = mat.tocsr()
    if isinstance(mat, csr_matrix):
        csr_matrices[filename] = mat
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


def compare_spmv_performance(scipy_csr, n_run: int = 10, transpose=False):
    data = jnp.asarray(scipy_csr.data)
    indices = jnp.asarray(scipy_csr.indices)
    indptr = jnp.asarray(scipy_csr.indptr)

    print(scipy_csr.shape)

    jax_csr = CSR((data, indices, indptr), shape=scipy_csr.shape)
    brainevent_csr = brainevent.CSR((data, indices, indptr), shape=scipy_csr.shape)

    vector = jax.block_until_ready(jnp.ones(scipy_csr.shape[0] if transpose else scipy_csr.shape[1]))

    @jax.jit
    def f_jax(v):
        if transpose:
            return jax_csr.T @ v
        return jax_csr @ v

    @jax.jit
    def f_brainevent(v):
        if transpose:
            return v @ brainevent_csr
        return brainevent_csr @ v

    r1 = jax.block_until_ready(f_jax(vector))
    r2 = jax.block_until_ready(f_brainevent(vector))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(vector))
    t1 = time.time()
    t_jax_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, transpose={transpose}, JAX  CSR @ Vector:       {t_jax_csr_vector:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(vector))
    t1 = time.time()
    t_be_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, transpose={transpose}, BrainEvent CSR @ Vector: {t_be_csr_vector:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_csr_vector / t_be_csr_vector}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    ratio = (
        (t_jax_csr_vector / t_be_csr_vector - 1.)
        if t_jax_csr_vector > t_be_csr_vector else
        -(t_be_csr_vector / t_jax_csr_vector - 1.)
    )
    print('Acceleration ratio:', ratio)
    print()
    return ratio


def evaluate_spmv_performance(transpose):
    results = dict()
    for filename in files:
        results[filename] = compare_spmv_performance(
            csr_matrices[filename], n_run=3 if brainstate.environ.get_platform() == 'cpu' else 30,
            transpose=transpose
        )
    title = 'Intel-i9-12900H-SpMV-CSR' if brainstate.environ.get_platform() == 'cpu' else 'RTX-3080Ti-SpMV-CSR'
    visualization(results, title=title)


def compare_spmm_performance(
    scipy_csr: csr_matrix,
    n_run: int = 10,
    batch_size: int = 100,
    transpose: bool = False
):
    print(scipy_csr.shape)

    data = jnp.asarray(scipy_csr.data)
    indices = jnp.asarray(scipy_csr.indices)
    indptr = jnp.asarray(scipy_csr.indptr)

    jax_csr = CSR((data, indices, indptr), shape=scipy_csr.shape)
    brainevent_csr = brainevent.CSR((data, indices, indptr), shape=scipy_csr.shape)

    if transpose:
        matrix = jax.block_until_ready(brainstate.random.randn(scipy_csr.shape[0], batch_size))
    else:
        matrix = jax.block_until_ready(brainstate.random.randn(scipy_csr.shape[1], batch_size))

    @jax.jit
    def f_jax(v):
        if transpose:
            return jax_csr.T @ v
        return jax_csr @ v

    @jax.jit
    def f_brainevent(v):
        if transpose:
            return brainevent_csr.T @ v
        return brainevent_csr @ v

    r1 = jax.block_until_ready(f_jax(matrix))
    r2 = jax.block_until_ready(f_brainevent(matrix))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(matrix))
    t1 = time.time()
    t_jax_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, transpose={transpose}, JAX  CSR @ Vector:       {t_jax_csr_vector:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(matrix))
    t1 = time.time()
    t_be_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, transpose={transpose}, BrainEvent CSR @ Vector: {t_be_csr_vector:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_csr_vector / t_be_csr_vector}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    ratio = (
        (t_jax_csr_vector / t_be_csr_vector - 1.)
        if t_jax_csr_vector > t_be_csr_vector else
        -(t_be_csr_vector / t_jax_csr_vector - 1.)
    )
    print('Acceleration ratio:', ratio)
    print()
    return ratio


def evaluate_spmm_performance(transpose):
    results = dict()
    for filename in files:
        results[filename] = compare_spmm_performance(
            csr_matrices[filename],
            n_run=3 if brainstate.environ.get_platform() == 'cpu' else 50,
            batch_size=512,
            transpose=transpose,
        )
    title = 'Intel-i9-12900H-SpMM-CSR' if brainstate.environ.get_platform() == 'cpu' else 'RTX-3080Ti-SpMM-CSR'
    visualization(results, title=title)


if __name__ == '__main__':
    evaluate_spmv_performance(True)
    evaluate_spmv_performance(False)
    evaluate_spmm_performance(transpose=True)
    evaluate_spmm_performance(transpose=False)
