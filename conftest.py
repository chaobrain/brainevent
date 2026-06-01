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

"""Pytest configuration shared by the whole test suite.

The bulk of the suite parametrizes on a compute backend (the ``implementation`` or
``backend`` test parameter).  The native backends — ``numba`` and friends — compile their
kernels on first use, costing seconds per test and *not* caching across tests, so they
dominate the wall-clock of a full run.

To keep the default ``pytest`` invocation fast, this hook automatically tags every test
variant that exercises a compilation-heavy backend with the ``slow`` marker.  Combined with
``addopts = "-m 'not slow'"`` in ``pyproject.toml`` the default run skips those variants and
exercises only the cheap backends (e.g. ``jax_raw``) plus backend-agnostic tests.

Run the full suite (CI does this) with ``pytest -m ""``; run only the heavy variants with
``pytest -m slow``.
"""

import pytest

#: Backends whose kernels are JIT/AOT compiled on first use. The compile cost is paid per
#: test (it is not cached across the session), so these variants are the slow ones.
COMPILATION_HEAVY_BACKENDS = frozenset({'numba', 'numba_cuda', 'pallas', 'warp', 'taichi'})

#: Test-parameter names that carry a backend identifier across the suite.
_BACKEND_PARAM_NAMES = ('implementation', 'backend')


def pytest_collection_modifyitems(config, items):
    """Mark parametrized variants that run a compilation-heavy backend as ``slow``.

    Parameters
    ----------
    config : pytest.Config
        The active pytest configuration (unused, required by the hook signature).
    items : list of pytest.Item
        Collected test items, mutated in place to attach the ``slow`` marker.
    """
    slow = pytest.mark.slow
    for item in items:
        callspec = getattr(item, 'callspec', None)
        if callspec is None:
            continue
        params = callspec.params
        if any(params.get(name) in COMPILATION_HEAVY_BACKENDS for name in _BACKEND_PARAM_NAMES):
            item.add_marker(slow)
