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

# -*- coding: utf-8 -*-

"""Pin the packaged hybrid FFI exports and verify task-capacity bounds."""

import re
from pathlib import Path


def test_binary_hybrid_cuda_exports_use_csr_abi_names():
    csr_dir = Path(__file__).parents[1]
    value_types = ('f32', 'f64', 'f16', 'bf16')
    event_types = ('bool', 'float')
    expected_symbols = {
        'hybrid_mv/hybrid_spmv_binary.cu': {
            f'binary_csrmv_wat_hybrid_{mode}_{value}_{event}'
            for mode in ('hetero', 'homo') for value in value_types for event in event_types
        },
        'hybrid_mv/hybrid_spmv_weighted.cu': {
            f'binary_csrmv_wat_hybrid_{mode}_{value}_wv'
            for mode in ('hetero', 'homo') for value in value_types
        },
        'hybrid_mv/hybrid_spmv_col_major_binary.cu': {
            f'hybrid_spmv_col_major_binary_hetero_{value}_{event}'
            for value in value_types for event in event_types
        },
        'hybrid_mv/hybrid_spmv_col_major_weighted.cu': {
            f'hybrid_spmv_col_major_weighted_hetero_{value}' for value in value_types
        },
        'hybrid_mm/hybrid_batchedv3_spmm_binary.cu': {
            f'batched_spmm_v3{mode}_{value}_{event}'
            for mode in ('', '_homo') for value in value_types for event in event_types
        } | {'batched_spmm_v3_device_info'},
        'hybrid_mm/hybrid_batchedv3_spmm_weighted.cu': {
            f'batched_spmm_v3_weighted{mode}_{value}'
            for mode in ('', '_homo') for value in value_types
        },
        'hybrid_mm/hybrid_batchedv3_spmm_col_major_binary.cu': {
            f'batched_spmm_v3_col_major_binary_hetero_{value}_{event}'
            for value in value_types for event in event_types
        } | {'batched_spmm_v3_col_major_device_info'},
        'hybrid_mm/hybrid_batchedv3_spmm_col_major_weighted.cu': {
            f'batched_spmm_v3_col_major_weighted_hetero_{value}' for value in value_types
        },
    }
    for filename, symbols in expected_symbols.items():
        text = (csr_dir / filename).read_text()
        exported = set(re.findall(r'^// @BE ([A-Za-z0-9_]+)$', text, re.MULTILINE))
        assert exported == symbols, filename


import numpy as np
import pytest

from brainevent._csr.config import CSRConfig


def _capacity(indptr, config):
    lengths = np.diff(indptr)
    heavy = lengths > config.tpr_threshold
    return int(((lengths[heavy] + config.task_nnz - 1) // config.task_nnz).sum())


@pytest.mark.parametrize('config', [CSRConfig(), CSRConfig(tpr_threshold=0, task_nnz=1),
                                    CSRConfig(tpr_threshold=3, task_nnz=5)])
def test_capacity_bound_is_never_below_the_exact_capacity(config):
    # The bound sizes the hybrid task queue from static shapes; it must cover
    # every structure with those shapes or the CUDA kernel drops work.
    rng = np.random.default_rng(0)
    for _ in range(300):
        n_rows = int(rng.integers(1, 40))
        lengths = rng.integers(0, int(rng.choice([4, 50, 9000])), n_rows)
        if rng.random() < 0.3:
            lengths[rng.integers(n_rows)] = int(rng.integers(0, 20000))
        indptr = np.concatenate([[0], np.cumsum(lengths)])
        nse = int(indptr[-1])
        exact = _capacity(indptr, config)
        assert config.mv_capacity(nse, n_rows) >= exact
        assert config.mv_capacity(nse, n_rows, int(np.diff(indptr).max(initial=0))) >= exact


def test_capacity_bound_uses_the_row_length_hint():
    config = CSRConfig()
    # No heavy row: the hint proves the queue is never used.
    assert config.mv_capacity(10_000, 100, max_row_nnz=config.tpr_threshold) == 0
    assert config.mv_capacity(10_000, 100) > 0
    assert config.mv_capacity(0, 100) == 0
    assert config.mv_capacity(10, 0) == 0
    # A single 9000-long row needs ceil(9000 / task_nnz) chunks.
    assert config.mv_capacity(9000, 1, max_row_nnz=9000) == 3


def test_prepared_row_hint_matches_concrete_structure():
    indptr = np.array([0, 129, 129 + 4096, 129 + 4096 + 4097], dtype=np.int64)
    indices = np.zeros(int(indptr[-1]), dtype=np.int32)
    config = CSRConfig().prepare_structure(indices, indptr, shape=(3, 1))
    assert config.primary.max_row_nnz == 4097
    assert config.mv_capacity(len(indices), 3, config.primary.max_row_nnz) >= _capacity(indptr, config)
