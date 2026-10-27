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

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainevent._error import PerformanceWarning
from brainevent._structure import StructureCache, is_concrete, nnz_hint, warn_mirror_fallback


def test_is_concrete_flags_tracers_only():
    assert is_concrete(jnp.arange(3), np.arange(3), 1.0)
    seen = []
    jax.jit(lambda x: seen.append(is_concrete(jnp.arange(3), x)) or x)(jnp.ones(2))
    assert seen == [False]


@pytest.mark.parametrize('indptr, hint', [
    ([0], 0),                 # no rows
    ([0, 0, 0], 0),           # only empty rows
    ([0, 1], 1),
    ([0, 3, 3, 5], 4),        # longest row 3 -> 4
    ([0, 4], 4),              # powers of two stay put
    ([0, 5, 10], 8),
    ([0, 129], 256),
])
def test_nnz_hint_is_the_power_of_two_ceiling_of_the_longest_line(indptr, hint):
    assert nnz_hint(np.asarray(indptr)) == hint


def test_nnz_hint_is_unknown_for_a_traced_indptr():
    seen = []
    jax.jit(lambda p: seen.append(nnz_hint(p)) or p)(jnp.asarray([0, 2, 3]))
    assert seen == [None]


def test_structure_cache_set_mirror_records_its_hint():
    cache = StructureCache(primary_hint=2)
    mirror = cache.set_mirror([np.asarray([0, 3, 4]), np.arange(4), np.arange(4)])
    assert isinstance(mirror, tuple) and cache.mirror is mirror
    assert cache.mirror_hint == 4
    assert cache.primary_hint == 2 and cache.diag_plan is None


def test_warn_mirror_fallback_warns_once_per_owner_and_operation():
    owner = 'StructureTestOwner'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warn_mirror_fallback(owner, 'op-a')
        warn_mirror_fallback(owner, 'op-a')
        warn_mirror_fallback(owner, 'op-b')
    perf = [w for w in caught if issubclass(w.category, PerformanceWarning)]
    assert len(perf) == 2
    assert 'build_weight_indices' in str(perf[0].message)
