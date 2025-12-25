# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import time

import jax
import jax.numpy as jnp
from scipy.io import mmread
from scipy.sparse import csr_matrix, coo_matrix

import brainevent
import brainstate

brainstate.environ.set(platform='cpu')
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


def compare_spmv_performance(
    scipy_csr,
    n_run: int = 10,
    spike_prob: float = 0.01
):
    data = jnp.asarray(scipy_csr.data)
    indices = jnp.asarray(scipy_csr.indices)
    indptr = jnp.asarray(scipy_csr.indptr)

    csr = brainevent.CSR((data, indices, indptr), shape=scipy_csr.shape)

    @jax.jit
    def f_csr(v):
        return csr @ v

    @jax.jit
    def f_event_csr(v):
        return csr @ brainevent.EventArray(v)

    vector = jax.block_until_ready((brainstate.random.rand(scipy_csr.shape[1]) < spike_prob).astype(float))

    r1 = jax.block_until_ready(f_csr(vector))
    r2 = jax.block_until_ready(f_event_csr(vector))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_csr(vector))
    t1 = time.time()
    t_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, prob = {spike_prob}, CSR @ Vector:     {t_csr_vector:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_event_csr(vector))
    t1 = time.time()
    t_event_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, prob = {spike_prob}, EventCSR @ Vector: {t_event_csr_vector:.6f} seconds")
    print(f'CSR / EventCSR: {t_csr_vector / t_event_csr_vector}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    @jax.jit
    def f_csr_transpose(v):
        return v @ csr

    @jax.jit
    def f_event_transpose(v):
        return brainevent.EventArray(v) @ csr

    vector = jax.block_until_ready((brainstate.random.rand(scipy_csr.shape[1]) < spike_prob).astype(float))

    r1 = jax.block_until_ready(f_csr_transpose(vector))
    r2 = jax.block_until_ready(f_event_transpose(vector))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_csr_transpose(vector))
    t1 = time.time()
    t_vector_csr = (t1 - t0) / n_run
    print(f"{filename}, prob = {spike_prob}, Vector @ CSR :     {t_vector_csr:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_event_transpose(vector))
    t1 = time.time()
    t_vector_event_csr = (t1 - t0) / n_run
    print(f"{filename}, prob = {spike_prob}, Vector @ EventCSR: {t_vector_event_csr:.6f} seconds")
    print(f'CSR / EventCSR: {t_vector_csr / t_vector_event_csr}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()


def compare_spmm_performance(
    scipy_csr: csr_matrix,
    n_run: int = 10,
    spike_prob: float = 0.1,
    batch_size: int = 100
):
    data = jnp.asarray(scipy_csr.data)
    indices = jnp.asarray(scipy_csr.indices)
    indptr = jnp.asarray(scipy_csr.indptr)

    csr = brainevent.CSR((data, indices, indptr), shape=scipy_csr.shape)

    @jax.jit
    def f_csr(v):
        return csr @ v

    @jax.jit
    def f_event_csr(v):
        return csr @ brainevent.EventArray(v)

    vector = jax.block_until_ready((brainstate.random.rand(scipy_csr.shape[1], batch_size) < spike_prob).astype(float))

    r1 = jax.block_until_ready(f_csr(vector))
    r2 = jax.block_until_ready(f_event_csr(vector))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_csr(vector))
    t1 = time.time()
    t_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, prob = {spike_prob}, CSR @ Vector:     {t_csr_vector:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_event_csr(vector))
    t1 = time.time()
    t_event_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, prob = {spike_prob}, EventCSR @ Vector: {t_event_csr_vector:.6f} seconds")
    print(f'CSR / EventCSR: {t_csr_vector / t_event_csr_vector}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    @jax.jit
    def f_csr_transpose(v):
        return v @ csr

    @jax.jit
    def f_event_transpose(v):
        return brainevent.EventArray(v) @ csr

    vector = jax.block_until_ready((brainstate.random.rand(batch_size, scipy_csr.shape[1]) < spike_prob).astype(float))

    r1 = jax.block_until_ready(f_csr_transpose(vector))
    r2 = jax.block_until_ready(f_event_transpose(vector))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_csr_transpose(vector))
    t1 = time.time()
    t_vector_csr = (t1 - t0) / n_run
    print(f"{filename}, prob = {spike_prob}, Vector @ CSR :     {t_vector_csr:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_event_transpose(vector))
    t1 = time.time()
    t_vector_event_csr = (t1 - t0) / n_run
    print(f"{filename}, prob = {spike_prob}, Vector @ EventCSR: {t_vector_event_csr:.6f} seconds")
    print(f'CSR / EventCSR: {t_vector_csr / t_vector_event_csr}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()


for filename in files:
    compare_spmv_performance(
        csr_matrices[filename],
        n_run=3 if brainstate.environ.get_platform() == 'cpu' else 30,
        spike_prob=0.01
    )

# for filename in files:
#     compare_spmm_performance(
#         csr_matrices[filename],
#         n_run=3 if brainstate.environ.get_platform() == 'cpu' else 30,
#         batch_size=100
#     )
