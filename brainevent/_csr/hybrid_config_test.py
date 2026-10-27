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

"""Tests for the hybrid-CSR ABI naming contract.

``hybrid_config.py`` derives the CUDA module/symbol suffix for a hybrid CSR
configuration. The CUDA sources have to agree with that derivation exactly --
a mismatch surfaces only at FFI lookup time, as a missing-symbol error deep in
the loader. The check below pins the exported ``// @BE`` names in the four
``*_hybrid.cu`` sources against the suffix scheme, and asserts that no
``fcnmm``/``const_block`` symbol from the neighbouring backends leaks into the
binary export list.
"""

import re
from pathlib import Path


def test_binary_hybrid_cuda_exports_use_csr_abi_names():
    csr_dir = Path(__file__).parent
    weight_suffixes = ('f32', 'f64', 'f16', 'bf16')
    event_suffixes = ('bool', 'float')
    expected_symbols = {
        'binary_csrmv_hybrid.cu': [
            f'binary_csrmv_wat_hybrid_{mode}_{weight}_{event}'
            for mode in ('hetero', 'homo')
            for weight in weight_suffixes
            for event in event_suffixes
        ],
        'binary_csrmm_hybrid.cu': [
            f'binary_csrmm_sraw_hybrid_{mode}_{weight}_{event}'
            for mode in ('hetero', 'homo')
            for weight in weight_suffixes
            for event in event_suffixes
        ],
        'binary_indexed_csrmv_hybrid.cu': [
            f'binary_indexed_csrmv_wat_hybrid_hetero_{weight}_{event}'
            for weight in weight_suffixes
            for event in event_suffixes
        ],
        'binary_indexed_csrmm_hybrid.cu': [
            f'binary_indexed_csrmm_sraw_hybrid_hetero_{weight}_{event}'
            for weight in weight_suffixes
            for event in event_suffixes
        ],
    }

    for filename, symbols in expected_symbols.items():
        text = (csr_dir / filename).read_text()
        exported = set(re.findall(r'^// @BE ([A-Za-z0-9_]+)$', text, re.MULTILINE))

        for symbol in symbols:
            assert symbol in exported
            suffix = '_' + '_'.join(symbol.split('_')[-2:])
            assert re.search(
                rf'^// @BE {symbol}\nDEFINE_[A-Z0-9_]+\(\\?{suffix},',
                text,
                re.MULTILINE,
            )

        binary_names = {name for name in exported if name.startswith('binary')}
        assert not {name for name in binary_names if 'fcnmm' in name or 'const_block' in name}


import numpy as np
import pytest

from brainevent._csr.hybrid_config import (
    HybridConfig, hybrid_capacity_bound, hybrid_task_capacity, max_row_nnz,
)


def _capacity(indptr, config):
    lengths = np.diff(indptr)
    heavy = lengths > config.tpr_threshold
    return int(((lengths[heavy] + config.task_nnz - 1) // config.task_nnz).sum())


@pytest.mark.parametrize('config', [HybridConfig(), HybridConfig(tpr_threshold=0, task_nnz=1),
                                    HybridConfig(tpr_threshold=3, task_nnz=5)])
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
        assert hybrid_capacity_bound(nse, n_rows, config=config) >= exact
        assert hybrid_capacity_bound(nse, n_rows, max_row_nnz(indptr), config=config) >= exact


def test_capacity_bound_uses_the_row_length_hint():
    config = HybridConfig()
    # No heavy row: the hint proves the queue is never used.
    assert hybrid_capacity_bound(10_000, 100, max_row_nnz=config.tpr_threshold, config=config) == 0
    assert hybrid_capacity_bound(10_000, 100, config=config) > 0
    assert hybrid_capacity_bound(0, 100, config=config) == 0
    assert hybrid_capacity_bound(10, 0, config=config) == 0
    # A single 9000-long row needs ceil(9000 / task_nnz) chunks.
    assert hybrid_capacity_bound(9000, 1, max_row_nnz=9000, config=config) == 3


def test_max_row_nnz_and_exact_capacity_agree_on_concrete_indptr():
    indptr = np.array([0, 129, 129 + 4096, 129 + 4096 + 4097])
    assert max_row_nnz(indptr) == 4097
    assert max_row_nnz(np.array([0])) == 0
    assert hybrid_task_capacity(indptr) == _capacity(indptr, HybridConfig())
