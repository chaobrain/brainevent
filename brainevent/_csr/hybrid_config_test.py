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
