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

import time

import jax
import jax.numpy as jnp
import scipy.io
from jax.experimental.sparse import CSR
from scipy.sparse import csr_matrix, coo_matrix

import brainevent
import brainstate as bst

#bst.environ.set(platform='cpu')
bst.environ.set(platform='gpu')


def load_sparse_matrix(filename):
    matrix = scipy.io.mmread(filename)
    return matrix


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
    mat = load_sparse_matrix(filename)
    if isinstance(mat, coo_matrix):
        csr_matrices[filename] = mat.tocsr()
    if isinstance(mat, csr_matrix):
        csr_matrices[filename] = mat
print()


def compare_spmv_performance(scipy_csr, n_run: int = 10):
    # Intel i9-12900H, WSL 2, ubuntu
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, JAX  CSR @ Vector:       2.012777 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, BrainEvent CSR @ Vector: 0.009752 seconds
    # JAX / BrainEvent: 206.39556185772844
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx,    JAX  Vector @ CSR :   2.007382 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, BrainEvent Vector @ CSR: 0.026426 seconds
    # JAX / BrainEvent: 75.96193890255564
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, JAX  CSR @ Vector:       0.458853 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, BrainEvent CSR @ Vector: 0.002388 seconds
    # JAX / BrainEvent: 192.1558891070656
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx,    JAX  Vector @ CSR :   0.526321 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, BrainEvent Vector @ CSR: 0.006309 seconds
    # JAX / BrainEvent: 83.42020934890225
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, JAX  CSR @ Vector:       1.671081 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, BrainEvent CSR @ Vector: 0.006553 seconds
    # JAX / BrainEvent: 255.02187939649735
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx,    JAX  Vector @ CSR :   1.586405 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, BrainEvent Vector @ CSR: 0.018944 seconds
    # JAX / BrainEvent: 83.74310202335054
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, JAX  CSR @ Vector:       0.190031 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, BrainEvent CSR @ Vector: 0.000896 seconds
    # JAX / BrainEvent: 212.03724394785849
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx,    JAX  Vector @ CSR :   0.179509 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, BrainEvent Vector @ CSR: 0.002131 seconds
    # JAX / BrainEvent: 84.23150357995227
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, JAX  CSR @ Vector:       0.875300 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, BrainEvent CSR @ Vector: 0.003423 seconds
    # JAX / BrainEvent: 255.71899233805436
    #
    # matrices/suitesparse/Williams/cant/cant.mtx,    JAX  Vector @ CSR :   0.833090 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, BrainEvent Vector @ CSR: 0.010198 seconds
    # JAX / BrainEvent: 81.6918017456359
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, JAX  CSR @ Vector:       1.758335 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, BrainEvent CSR @ Vector: 0.004156 seconds
    # JAX / BrainEvent: 423.07205139972467
    #
    # matrices/suitesparse/Williams/consph/consph.mtx,    JAX  Vector @ CSR :   1.271770 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, BrainEvent Vector @ CSR: 0.014446 seconds
    # JAX / BrainEvent: 88.03311713674297
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, JAX  CSR @ Vector:       0.562398 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, BrainEvent CSR @ Vector: 0.002839 seconds
    # JAX / BrainEvent: 198.0689655172414
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx,    JAX  Vector @ CSR :   0.570374 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, BrainEvent Vector @ CSR: 0.006549 seconds
    # JAX / BrainEvent: 87.09071934763615
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, JAX  CSR @ Vector:       0.964837 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, BrainEvent CSR @ Vector: 0.003670 seconds
    # JAX / BrainEvent: 262.8887421233841
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx,    JAX  Vector @ CSR :   0.903681 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, BrainEvent Vector @ CSR: 0.010366 seconds
    # JAX / BrainEvent: 87.18108702819158

    #
    # NVIDIA GeForce RTX 3080 Ti Laptop GPU, WSL 2, ubuntu
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, JAX  CSR @ Vector:       0.001310 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, BrainEvent CSR @ Vector: 0.007625 seconds
    # JAX / BrainEvent: 0.17180381181694523
    #
    # Module brainevent._csr 6c2f2e1 load on device 'cuda:0' took 0.85 ms  (cached)
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx,    JAX  Vector @ CSR :   0.001319 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, BrainEvent Vector @ CSR: 0.007955 seconds
    # JAX / BrainEvent: 0.1658451794651509
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, JAX  CSR @ Vector:       0.001085 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, BrainEvent CSR @ Vector: 0.001257 seconds
    # JAX / BrainEvent: 0.8630545859260383
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx,    JAX  Vector @ CSR :   0.001128 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, BrainEvent Vector @ CSR: 0.001358 seconds
    # JAX / BrainEvent: 0.8312374863887034
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, JAX  CSR @ Vector:       0.001224 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, BrainEvent CSR @ Vector: 0.001384 seconds
    # JAX / BrainEvent: 0.8845153250213924
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx,    JAX  Vector @ CSR :   0.001260 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, BrainEvent Vector @ CSR: 0.002355 seconds
    # JAX / BrainEvent: 0.5349107914057438
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, JAX  CSR @ Vector:       0.000216 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, BrainEvent CSR @ Vector: 0.012997 seconds
    # JAX / BrainEvent: 0.01660174761674728
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx,    JAX  Vector @ CSR :   0.001293 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, BrainEvent Vector @ CSR: 0.011234 seconds
    # JAX / BrainEvent: 0.11509207054379277
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, JAX  CSR @ Vector:       0.001241 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, BrainEvent CSR @ Vector: 0.001418 seconds
    # JAX / BrainEvent: 0.8747695807349883
    #
    # matrices/suitesparse/Williams/cant/cant.mtx,    JAX  Vector @ CSR :   0.001222 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, BrainEvent Vector @ CSR: 0.001286 seconds
    # JAX / BrainEvent: 0.9501436914804858
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, JAX  CSR @ Vector:       0.001363 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, BrainEvent CSR @ Vector: 0.001299 seconds
    # JAX / BrainEvent: 1.0494776845545173
    #
    # matrices/suitesparse/Williams/consph/consph.mtx,    JAX  Vector @ CSR :   0.002331 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, BrainEvent Vector @ CSR: 0.006609 seconds
    # JAX / BrainEvent: 0.35272544943191997
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, JAX  CSR @ Vector:       0.001293 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, BrainEvent CSR @ Vector: 0.001320 seconds
    # JAX / BrainEvent: 0.9797502348322454
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx,    JAX  Vector @ CSR :   0.001189 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, BrainEvent Vector @ CSR: 0.001339 seconds
    # JAX / BrainEvent: 0.8880907147945855
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, JAX  CSR @ Vector:       0.001057 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, BrainEvent CSR @ Vector: 0.001328 seconds
    # JAX / BrainEvent: 0.7961871480801106
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx,    JAX  Vector @ CSR :   0.001300 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, BrainEvent Vector @ CSR: 0.001272 seconds
    # JAX / BrainEvent: 1.0216478614536868

    data = jnp.asarray(scipy_csr.data)
    indices = jnp.asarray(scipy_csr.indices)
    indptr = jnp.asarray(scipy_csr.indptr)

    jax_csr = CSR([data, indices, indptr], shape=scipy_csr.shape)
    brainevent_csr = brainevent.CSR([data, indices, indptr], shape=scipy_csr.shape)

    @jax.jit
    def f_jax(v):
        return jax_csr @ v

    @jax.jit
    def f_brainevent(v):
        return brainevent_csr @ v

    vector = jax.block_until_ready(jnp.ones(scipy_csr.shape[1]))

    r1 = jax.block_until_ready(f_jax(vector))
    r2 = jax.block_until_ready(f_brainevent(vector))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(vector))
    t1 = time.time()
    t_jax_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, JAX  CSR @ Vector:       {t_jax_csr_vector:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(vector))
    t1 = time.time()
    t_be_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, BrainEvent CSR @ Vector: {t_be_csr_vector:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_csr_vector / t_be_csr_vector}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    @jax.jit
    def f_jax(v):
        return jax_csr.T @ v

    @jax.jit
    def f_brainevent(v):
        return v @ brainevent_csr

    vector = jax.block_until_ready(jnp.ones(scipy_csr.shape[0]))

    r1 = jax.block_until_ready(f_jax(vector))
    r2 = jax.block_until_ready(f_brainevent(vector))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(vector))
    t1 = time.time()
    t_jax_vector_csr = (t1 - t0) / n_run
    print(f"{filename},    JAX  Vector @ CSR :   {t_jax_vector_csr:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(vector))
    t1 = time.time()
    t_be_vector_csr = (t1 - t0) / n_run
    print(f"{filename}, BrainEvent Vector @ CSR: {t_be_vector_csr:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_vector_csr / t_be_vector_csr}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()


def compare_spmm_performance(
    scipy_csr: csr_matrix,
    n_run: int = 10,
    batch_size: int = 100
):
    # Intel i9-12900H, WSL 2, ubuntu
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, JAX  CSR @ Vector:       2.012777 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, BrainEvent CSR @ Vector: 0.009752 seconds
    # JAX / BrainEvent: 206.39556185772844
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx,    JAX  Vector @ CSR :   2.007382 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, BrainEvent Vector @ CSR: 0.026426 seconds
    # JAX / BrainEvent: 75.96193890255564
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, JAX  CSR @ Vector:       0.458853 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, BrainEvent CSR @ Vector: 0.002388 seconds
    # JAX / BrainEvent: 192.1558891070656
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx,    JAX  Vector @ CSR :   0.526321 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, BrainEvent Vector @ CSR: 0.006309 seconds
    # JAX / BrainEvent: 83.42020934890225
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, JAX  CSR @ Vector:       1.671081 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, BrainEvent CSR @ Vector: 0.006553 seconds
    # JAX / BrainEvent: 255.02187939649735
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx,    JAX  Vector @ CSR :   1.586405 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, BrainEvent Vector @ CSR: 0.018944 seconds
    # JAX / BrainEvent: 83.74310202335054
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, JAX  CSR @ Vector:       0.190031 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, BrainEvent CSR @ Vector: 0.000896 seconds
    # JAX / BrainEvent: 212.03724394785849
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx,    JAX  Vector @ CSR :   0.179509 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, BrainEvent Vector @ CSR: 0.002131 seconds
    # JAX / BrainEvent: 84.23150357995227
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, JAX  CSR @ Vector:       0.875300 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, BrainEvent CSR @ Vector: 0.003423 seconds
    # JAX / BrainEvent: 255.71899233805436
    #
    # matrices/suitesparse/Williams/cant/cant.mtx,    JAX  Vector @ CSR :   0.833090 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, BrainEvent Vector @ CSR: 0.010198 seconds
    # JAX / BrainEvent: 81.6918017456359
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, JAX  CSR @ Vector:       1.758335 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, BrainEvent CSR @ Vector: 0.004156 seconds
    # JAX / BrainEvent: 423.07205139972467
    #
    # matrices/suitesparse/Williams/consph/consph.mtx,    JAX  Vector @ CSR :   1.271770 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, BrainEvent Vector @ CSR: 0.014446 seconds
    # JAX / BrainEvent: 88.03311713674297
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, JAX  CSR @ Vector:       0.562398 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, BrainEvent CSR @ Vector: 0.002839 seconds
    # JAX / BrainEvent: 198.0689655172414
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx,    JAX  Vector @ CSR :   0.570374 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, BrainEvent Vector @ CSR: 0.006549 seconds
    # JAX / BrainEvent: 87.09071934763615
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, JAX  CSR @ Vector:       0.964837 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, BrainEvent CSR @ Vector: 0.003670 seconds
    # JAX / BrainEvent: 262.8887421233841
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx,    JAX  Vector @ CSR :   0.903681 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, BrainEvent Vector @ CSR: 0.010366 seconds
    # JAX / BrainEvent: 87.18108702819158

    #
    # NVIDIA GeForce RTX 3080 Ti Laptop GPU, WSL 2, ubuntu
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, JAX  CSR @ Vector:       0.001310 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, BrainEvent CSR @ Vector: 0.007625 seconds
    # JAX / BrainEvent: 0.17180381181694523
    #
    # Module brainevent._csr 6c2f2e1 load on device 'cuda:0' took 0.85 ms  (cached)
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx,    JAX  Vector @ CSR :   0.001319 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, BrainEvent Vector @ CSR: 0.007955 seconds
    # JAX / BrainEvent: 0.1658451794651509
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, JAX  CSR @ Vector:       0.001085 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, BrainEvent CSR @ Vector: 0.001257 seconds
    # JAX / BrainEvent: 0.8630545859260383
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx,    JAX  Vector @ CSR :   0.001128 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, BrainEvent Vector @ CSR: 0.001358 seconds
    # JAX / BrainEvent: 0.8312374863887034
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, JAX  CSR @ Vector:       0.001224 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, BrainEvent CSR @ Vector: 0.001384 seconds
    # JAX / BrainEvent: 0.8845153250213924
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx,    JAX  Vector @ CSR :   0.001260 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, BrainEvent Vector @ CSR: 0.002355 seconds
    # JAX / BrainEvent: 0.5349107914057438
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, JAX  CSR @ Vector:       0.000216 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, BrainEvent CSR @ Vector: 0.012997 seconds
    # JAX / BrainEvent: 0.01660174761674728
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx,    JAX  Vector @ CSR :   0.001293 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, BrainEvent Vector @ CSR: 0.011234 seconds
    # JAX / BrainEvent: 0.11509207054379277
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, JAX  CSR @ Vector:       0.001241 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, BrainEvent CSR @ Vector: 0.001418 seconds
    # JAX / BrainEvent: 0.8747695807349883
    #
    # matrices/suitesparse/Williams/cant/cant.mtx,    JAX  Vector @ CSR :   0.001222 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, BrainEvent Vector @ CSR: 0.001286 seconds
    # JAX / BrainEvent: 0.9501436914804858
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, JAX  CSR @ Vector:       0.001363 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, BrainEvent CSR @ Vector: 0.001299 seconds
    # JAX / BrainEvent: 1.0494776845545173
    #
    # matrices/suitesparse/Williams/consph/consph.mtx,    JAX  Vector @ CSR :   0.002331 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, BrainEvent Vector @ CSR: 0.006609 seconds
    # JAX / BrainEvent: 0.35272544943191997
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, JAX  CSR @ Vector:       0.001293 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, BrainEvent CSR @ Vector: 0.001320 seconds
    # JAX / BrainEvent: 0.9797502348322454
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx,    JAX  Vector @ CSR :   0.001189 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, BrainEvent Vector @ CSR: 0.001339 seconds
    # JAX / BrainEvent: 0.8880907147945855
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, JAX  CSR @ Vector:       0.001057 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, BrainEvent CSR @ Vector: 0.001328 seconds
    # JAX / BrainEvent: 0.7961871480801106
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx,    JAX  Vector @ CSR :   0.001300 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, BrainEvent Vector @ CSR: 0.001272 seconds
    # JAX / BrainEvent: 1.0216478614536868

    data = jnp.asarray(scipy_csr.data)
    indices = jnp.asarray(scipy_csr.indices)
    indptr = jnp.asarray(scipy_csr.indptr)

    jax_csr = CSR([data, indices, indptr], shape=scipy_csr.shape)
    brainevent_csr = brainevent.CSR([data, indices, indptr], shape=scipy_csr.shape)

    @jax.jit
    def f_jax(v):
        return jax_csr @ v

    @jax.jit
    def f_brainevent(v):
        return brainevent_csr @ v

    matrix = jax.block_until_ready(bst.random.randn(scipy_csr.shape[1], batch_size))

    r1 = jax.block_until_ready(f_jax(matrix))
    r2 = jax.block_until_ready(f_brainevent(matrix))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(matrix))
    t1 = time.time()
    t_jax_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, JAX  CSR @ Vector:       {t_jax_csr_vector:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(matrix))
    t1 = time.time()
    t_be_csr_vector = (t1 - t0) / n_run
    print(f"{filename}, BrainEvent CSR @ Vector: {t_be_csr_vector:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_csr_vector / t_be_csr_vector}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()

    @jax.jit
    def f_jax(v):
        return (jax_csr.T @ v.T).T

    @jax.jit
    def f_brainevent(v):
        return v @ brainevent_csr

    matrix = jax.block_until_ready(bst.random.randn(batch_size, scipy_csr.shape[0]))

    r1 = jax.block_until_ready(f_jax(matrix))
    r2 = jax.block_until_ready(f_brainevent(matrix))

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_jax(matrix))
    t1 = time.time()
    t_jax_vector_csr = (t1 - t0) / n_run
    print(f"{filename},    JAX  Vector @ CSR :   {t_jax_vector_csr:.6f} seconds")

    t0 = time.time()
    for _ in range(n_run):
        jax.block_until_ready(f_brainevent(matrix))
    t1 = time.time()
    t_be_vector_csr = (t1 - t0) / n_run
    print(f"{filename}, BrainEvent Vector @ CSR: {t_be_vector_csr:.6f} seconds")
    print(f'JAX / BrainEvent: {t_jax_vector_csr / t_be_vector_csr}, max value diff: {jnp.max(jnp.abs(r1 - r2))}')
    print()


for filename in files:
    compare_spmv_performance(
        csr_matrices[filename],
        n_run=3 if bst.environ.get_platform() == 'cpu' else 30
    )

# for filename in files:
#     compare_spmm_performance(
#         csr_matrices[filename],
#         n_run=3 if bst.environ.get_platform() == 'cpu' else 30,
#         batch_size=100
#     )
