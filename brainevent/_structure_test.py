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

from brainevent._error import PerformanceWarning
from brainevent._structure import StructureCache, is_concrete, warn_mirror_fallback


def test_is_concrete_flags_tracers_only():
    assert is_concrete(jnp.arange(3), np.arange(3), 1.0)
    seen = []
    jax.jit(lambda x: seen.append(is_concrete(jnp.arange(3), x)) or x)(jnp.ones(2))
    assert seen == [False]


def test_structure_cache_set_mirror_does_not_read_array_values():
    class NoHostRead:
        def __array__(self, *args, **kwargs):
            raise AssertionError('storing a mirror must not scan its values')

    cache = StructureCache()
    mirror = cache.set_mirror([NoHostRead(), NoHostRead(), NoHostRead()])
    assert isinstance(mirror, tuple) and cache.mirror is mirror
    assert cache.diag_plan is None


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
