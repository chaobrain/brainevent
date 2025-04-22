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
    # Intel i9-12900H, WSL 2, ubuntu, spike probability = 0.1
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.1, CSR @ Vector:     0.010333 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.1, EventCSR @ Vector: 0.012288 seconds
    # CSR / EventCSR: 0.8409349614528897, max value diff: 0.0
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.1, Vector @ CSR :     0.026951 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.1, Vector @ EventCSR: 0.004794 seconds
    # CSR / EventCSR: 5.621868938879034, max value diff: 0.0
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.1, CSR @ Vector:     0.002131 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.1, EventCSR @ Vector: 0.002699 seconds
    # CSR / EventCSR: 0.7894767835585784, max value diff: 0.0
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.1, Vector @ CSR :     0.005908 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.1, Vector @ EventCSR: 0.001034 seconds
    # CSR / EventCSR: 5.7134732149719465, max value diff: 0.0
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.1, CSR @ Vector:     0.006437 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.1, EventCSR @ Vector: 0.008128 seconds
    # CSR / EventCSR: 0.7918459131795073, max value diff: 0.0
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.1, Vector @ CSR :     0.019081 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.1, Vector @ EventCSR: 0.002812 seconds
    # CSR / EventCSR: 6.785766774065909, max value diff: 0.0
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.1, CSR @ Vector:     0.000635 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.1, EventCSR @ Vector: 0.001201 seconds
    # CSR / EventCSR: 0.5284967233732707, max value diff: 0.0
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.1, Vector @ CSR :     0.002117 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.1, Vector @ EventCSR: 0.000432 seconds
    # CSR / EventCSR: 4.89489158397648, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.1, CSR @ Vector:     0.002906 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.1, EventCSR @ Vector: 0.003767 seconds
    # CSR / EventCSR: 0.7716425783310475, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.1, Vector @ CSR :     0.009650 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.1, Vector @ EventCSR: 0.001452 seconds
    # CSR / EventCSR: 6.6443228454172365, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.1, CSR @ Vector:     0.005099 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.1, EventCSR @ Vector: 0.005494 seconds
    # CSR / EventCSR: 0.9281509019108648, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.1, Vector @ CSR :     0.014317 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.1, Vector @ EventCSR: 0.002003 seconds
    # CSR / EventCSR: 7.149212270328188, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.1, CSR @ Vector:     0.002662 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.1, EventCSR @ Vector: 0.003532 seconds
    # CSR / EventCSR: 0.7535380807739903, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.1, Vector @ CSR :     0.006922 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.1, Vector @ EventCSR: 0.001131 seconds
    # CSR / EventCSR: 6.119651514086982, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.1, CSR @ Vector:     0.003653 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.1, EventCSR @ Vector: 0.004962 seconds
    # CSR / EventCSR: 0.7362385688431909, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.1, Vector @ CSR :     0.009996 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.1, Vector @ EventCSR: 0.001243 seconds
    # CSR / EventCSR: 8.041557445176139, max value diff: 0.0

    # Intel i9-12900H, WSL 2, ubuntu, spike probability = 0.01
    #
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.01, CSR @ Vector:     0.009427 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.01, EventCSR @ Vector: 0.012399 seconds
    # CSR / EventCSR: 0.7602520222025663, max value diff: 0.0
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.01, Vector @ CSR :     0.027117 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.01, Vector @ EventCSR: 0.002042 seconds
    # CSR / EventCSR: 13.27839825660583, max value diff: 0.0
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.01, CSR @ Vector:     0.001927 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.01, EventCSR @ Vector: 0.002236 seconds
    # CSR / EventCSR: 0.8618336886993604, max value diff: 0.0
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.01, Vector @ CSR :     0.006250 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.01, Vector @ EventCSR: 0.000182 seconds
    # CSR / EventCSR: 34.2969908416921, max value diff: 0.0
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.01, CSR @ Vector:     0.006654 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.01, EventCSR @ Vector: 0.007408 seconds
    # CSR / EventCSR: 0.8981923510164672, max value diff: 0.0
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.01, Vector @ CSR :     0.020149 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.01, Vector @ EventCSR: 0.000546 seconds
    # CSR / EventCSR: 36.915987186953984, max value diff: 0.0
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.01, CSR @ Vector:     0.000816 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.01, EventCSR @ Vector: 0.000870 seconds
    # CSR / EventCSR: 0.9381339669194919, max value diff: 0.0
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.01, Vector @ CSR :     0.002086 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.01, Vector @ EventCSR: 0.000134 seconds
    # CSR / EventCSR: 15.598336304218655, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.01, CSR @ Vector:     0.002764 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.01, EventCSR @ Vector: 0.003762 seconds
    # CSR / EventCSR: 0.7346417103261788, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.01, Vector @ CSR :     0.009304 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.01, Vector @ EventCSR: 0.000186 seconds
    # CSR / EventCSR: 49.9867634500427, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.01, CSR @ Vector:     0.004843 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.01, EventCSR @ Vector: 0.006296 seconds
    # CSR / EventCSR: 0.7692637850939205, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.01, Vector @ CSR :     0.014597 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.01, Vector @ EventCSR: 0.000255 seconds
    # CSR / EventCSR: 57.21744548286604, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.01, CSR @ Vector:     0.002598 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.01, EventCSR @ Vector: 0.002995 seconds
    # CSR / EventCSR: 0.8674698795180723, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.01, Vector @ CSR :     0.007113 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.01, Vector @ EventCSR: 0.000266 seconds
    # CSR / EventCSR: 26.717611940298507, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.01, CSR @ Vector:     0.003415 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.01, EventCSR @ Vector: 0.004228 seconds
    # CSR / EventCSR: 0.8077147207549298, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.01, Vector @ CSR :     0.010670 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.01, Vector @ EventCSR: 0.000169 seconds
    # CSR / EventCSR: 63.23881300047103, max value diff: 0.0

    #
    # NVIDIA GeForce RTX 3080 Ti Laptop GPU, WSL 2, ubuntu, spike probability = 0.01
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.01, CSR @ Vector:     0.006815 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.01, EventCSR @ Vector: 0.006875 seconds
    # CSR / EventCSR: 0.9912979951172597, max value diff: 0.0
    #
    # Module brainevent._csr 6c2f2e1 load on device 'cuda:0' took 0.56 ms  (cached)
    # Module brainevent._event_csr 41f7c8b load on device 'cuda:0' took 0.47 ms  (cached)
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.01, Vector @ CSR :     0.007834 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.01, Vector @ EventCSR: 0.001346 seconds
    # CSR / EventCSR: 5.822225897139077, max value diff: 0.0
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.01, CSR @ Vector:     0.001298 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.01, EventCSR @ Vector: 0.000741 seconds
    # CSR / EventCSR: 1.7530901287553649, max value diff: 0.0
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.01, Vector @ CSR :     0.001350 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.01, Vector @ EventCSR: 0.000379 seconds
    # CSR / EventCSR: 3.5615171719713175, max value diff: 0.015625
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.01, CSR @ Vector:     0.001315 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.01, EventCSR @ Vector: 0.001320 seconds
    # CSR / EventCSR: 0.996063584545657, max value diff: 0.0
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.01, Vector @ CSR :     0.002393 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.01, Vector @ EventCSR: 0.000166 seconds
    # CSR / EventCSR: 14.39975135083441, max value diff: 512.0
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.01, CSR @ Vector:     0.012660 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.01, EventCSR @ Vector: 0.011177 seconds
    # CSR / EventCSR: 1.1325988574028847, max value diff: 0.0
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.01, Vector @ CSR :     0.010931 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.01, Vector @ EventCSR: 0.000294 seconds
    # CSR / EventCSR: 37.14868332207967, max value diff: 0.00103759765625
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.01, CSR @ Vector:     0.001340 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.01, EventCSR @ Vector: 0.001279 seconds
    # CSR / EventCSR: 1.0475441725760242, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.01, Vector @ CSR :     0.001324 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.01, Vector @ EventCSR: 0.000147 seconds
    # CSR / EventCSR: 9.016068819996754, max value diff: 0.000244140625
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.01, CSR @ Vector:     0.005755 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.01, EventCSR @ Vector: 0.002493 seconds
    # CSR / EventCSR: 2.3083045655507, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.01, Vector @ CSR :     0.006597 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.01, Vector @ EventCSR: 0.001090 seconds
    # CSR / EventCSR: 6.052783935133874, max value diff: 0.00048828125
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.01, CSR @ Vector:     0.001290 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.01, EventCSR @ Vector: 0.001294 seconds
    # CSR / EventCSR: 0.9968555583533444, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.01, Vector @ CSR :     0.001323 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.01, Vector @ EventCSR: 0.000278 seconds
    # CSR / EventCSR: 4.762846189059282, max value diff: 7.62939453125e-06
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.01, CSR @ Vector:     0.002850 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.01, EventCSR @ Vector: 0.001392 seconds
    # CSR / EventCSR: 2.0470110107138986, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.01, Vector @ CSR :     0.004647 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.01, Vector @ EventCSR: 0.001100 seconds
    # CSR / EventCSR: 4.223388197569026, max value diff: 1.52587890625e-05

    #
    # NVIDIA GeForce RTX 3080 Ti Laptop GPU, WSL 2, ubuntu, spike probability = 0.1
    #
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.1, CSR @ Vector:     0.006852 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.1, EventCSR @ Vector: 0.007899 seconds
    # CSR / EventCSR: 0.8674290520757663, max value diff: 0.0
    #
    # Module brainevent._csr 6c2f2e1 load on device 'cuda:0' took 0.58 ms  (cached)
    # Module brainevent._event_csr 41f7c8b load on device 'cuda:0' took 0.71 ms  (cached)
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.1, Vector @ CSR :     0.007889 seconds
    # matrices/suitesparse/Andrianov/mip1/mip1.mtx, prob = 0.1, Vector @ EventCSR: 0.001333 seconds
    # CSR / EventCSR: 5.917695031655757, max value diff: 0.0
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.1, CSR @ Vector:     0.001293 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.1, EventCSR @ Vector: 0.000935 seconds
    # CSR / EventCSR: 1.3834505275867053, max value diff: 0.0
    #
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.1, Vector @ CSR :     0.001334 seconds
    # matrices/suitesparse/Bova/rma10/rma10.mtx, prob = 0.1, Vector @ EventCSR: 0.000230 seconds
    # CSR / EventCSR: 5.811128038224499, max value diff: 2.0
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.1, CSR @ Vector:     0.001367 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.1, EventCSR @ Vector: 0.001318 seconds
    # CSR / EventCSR: 1.0374377996923911, max value diff: 0.0
    #
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.1, Vector @ CSR :     0.002422 seconds
    # matrices/suitesparse/DNVS/shipsec1/shipsec1.mtx, prob = 0.1, Vector @ EventCSR: 0.000426 seconds
    # CSR / EventCSR: 5.683702018129592, max value diff: 262144.0
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.1, CSR @ Vector:     0.012005 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.1, EventCSR @ Vector: 0.013980 seconds
    # CSR / EventCSR: 0.8587216282464654, max value diff: 0.0
    #
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.1, Vector @ CSR :     0.011042 seconds
    # matrices/suitesparse/IBM_EDA/dc2/dc2.mtx, prob = 0.1, Vector @ EventCSR: 0.000399 seconds
    # CSR / EventCSR: 27.69253154212593, max value diff: 0.03662109375
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.1, CSR @ Vector:     0.001367 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.1, EventCSR @ Vector: 0.001329 seconds
    # CSR / EventCSR: 1.0284914597387684, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.1, Vector @ CSR :     0.001360 seconds
    # matrices/suitesparse/Williams/cant/cant.mtx, prob = 0.1, Vector @ EventCSR: 0.000206 seconds
    # CSR / EventCSR: 6.591665382837776, max value diff: 0.00146484375
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.1, CSR @ Vector:     0.005993 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.1, EventCSR @ Vector: 0.001945 seconds
    # CSR / EventCSR: 3.0819648681123764, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.1, Vector @ CSR :     0.002418 seconds
    # matrices/suitesparse/Williams/consph/consph.mtx, prob = 0.1, Vector @ EventCSR: 0.000581 seconds
    # CSR / EventCSR: 4.159034365858326, max value diff: 0.00390625
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.1, CSR @ Vector:     0.001267 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.1, EventCSR @ Vector: 0.001265 seconds
    # CSR / EventCSR: 1.001608808225032, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.1, Vector @ CSR :     0.001359 seconds
    # matrices/suitesparse/Williams/cop20k_A/cop20k_A.mtx, prob = 0.1, Vector @ EventCSR: 0.000267 seconds
    # CSR / EventCSR: 5.084433345225354, max value diff: 1.52587890625e-05
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.1, CSR @ Vector:     0.001386 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.1, EventCSR @ Vector: 0.001361 seconds
    # CSR / EventCSR: 1.0181996531731157, max value diff: 0.0
    #
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.1, Vector @ CSR :     0.002428 seconds
    # matrices/suitesparse/Williams/pdb1HYS/pdb1HYS.mtx, prob = 0.1, Vector @ EventCSR: 0.001356 seconds
    # CSR / EventCSR: 1.790075351856843, max value diff: 3.814697265625e-05

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
