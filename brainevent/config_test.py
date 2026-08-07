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

"""Tests for the pure-Python runtime configuration in :mod:`brainevent.config`.

The compute-capability and nvcc-discovery delegation paths are exercised by
``brainevent/_op/kernix_arch_resolution_test.py`` and
``brainevent/_op/kernix_toolchain_test.py``; this module focuses on the
remaining, otherwise-untested controls: Numba parallel/thread state, LFSR
algorithm selection, and the global per-platform backend map.
"""

from unittest import mock

import pytest

import brainevent
from brainevent import config


@pytest.fixture(autouse=True)
def _restore_config():
    """Snapshot every mutable config global and restore it after each test.

    The config module keeps process-global state; leaking a change (e.g. a
    pinned backend or numba thread count) would make sibling tests in the full
    suite order-dependent. Capturing and restoring the private globals keeps
    each test hermetic without depending on a particular default value.
    """
    saved = (
        config.get_numba_parallel(),
        config.get_numba_num_threads(),
        config.get_lfsr_algorithm(),
        dict(config._global_backends),
    )
    try:
        yield
    finally:
        parallel, num_threads, lfsr, backends = saved
        config._numba_parallel = parallel
        config._numba_num_threads = num_threads
        config._lfsr_algorithm = lfsr
        config._global_backends.clear()
        config._global_backends.update(backends)


# ---------------------------------------------------------------------------
# Numba parallel / thread configuration
# ---------------------------------------------------------------------------


def test_set_numba_parallel_toggles_flag_without_touching_threads():
    config.set_numba_parallel(True)
    assert config.get_numba_parallel() is True
    # num_threads omitted -> thread count left untouched (no numba import side effect).
    assert config.get_numba_num_threads() is None

    config.set_numba_parallel(False)
    assert config.get_numba_parallel() is False


def test_set_numba_parallel_records_thread_count_and_calls_numba():
    config.set_numba_parallel(True, num_threads=2)
    assert config.get_numba_parallel() is True
    assert config.get_numba_num_threads() == 2
    # The configured count must agree with numba's live thread-pool size.
    import numba
    assert numba.get_num_threads() == 2


def test_numba_helpers_are_exposed_on_the_package_namespace():
    # The package re-exports the getters/setters used in the docstring examples.
    assert brainevent.config.get_numba_parallel is config.get_numba_parallel
    assert brainevent.config.set_numba_parallel is config.set_numba_parallel


# ---------------------------------------------------------------------------
# LFSR algorithm selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["lfsr88", "lfsr113", "lfsr128"])
def test_set_lfsr_algorithm_accepts_each_valid_choice(algorithm):
    config.set_lfsr_algorithm(algorithm)
    assert config.get_lfsr_algorithm() == algorithm


def test_set_lfsr_algorithm_rejects_unknown_choice():
    with pytest.raises(ValueError, match="Invalid LFSR algorithm"):
        config.set_lfsr_algorithm("mersenne")
    # A rejected assignment must not mutate the current value.
    assert config.get_lfsr_algorithm() in ("lfsr88", "lfsr113", "lfsr128")


# ---------------------------------------------------------------------------
# Global per-platform backend map
# ---------------------------------------------------------------------------


def test_set_and_get_backend_roundtrip():
    config.set_backend("gpu", "warp")
    assert config.get_backend("gpu") == "warp"
    # An unset platform reports None rather than raising.
    assert config.get_backend("tpu") is None


def test_set_backend_none_clears_only_that_platform():
    config.set_backend("gpu", "warp")
    config.set_backend("cpu", "numba")
    config.set_backend("gpu", None)
    assert config.get_backend("gpu") is None
    assert config.get_backend("cpu") == "numba"


def test_set_backend_none_on_unset_platform_is_a_noop():
    # Clearing a platform that was never set must not raise (pop with default).
    config.set_backend("gpu", None)
    assert config.get_backend("gpu") is None


def test_set_backend_rejects_empty_string():
    with pytest.raises(ValueError, match="empty string"):
        config.set_backend("gpu", "")
    assert config.get_backend("gpu") is None


def test_clear_backends_removes_every_platform():
    config.set_backend("gpu", "warp")
    config.set_backend("cpu", "numba")
    config.clear_backends()
    assert config.get_backend("gpu") is None
    assert config.get_backend("cpu") is None


def test_clear_backends_on_empty_map_is_a_noop():
    config.clear_backends()
    config.clear_backends()
    assert config.get_backend("gpu") is None


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------


def test_all_names_are_resolvable_attributes():
    # Every promised public name must actually exist on the module.
    for name in config.__all__:
        assert hasattr(config, name), name


# ---------------------------------------------------------------------------
# F1 -- ``set_backend`` / ``clear_backends`` only clear JAX caches when the
# effective value actually changes (D1).
# ---------------------------------------------------------------------------


def test_f1_set_backend_and_clear_backends_are_idempotent_about_cache_clearing():
    """A no-op ``set_backend``/``clear_backends`` call must not touch JAX caches.

    Pre-fix, ``set_backend``/``clear_backends`` called ``jax.clear_caches()``
    unconditionally on every call (or not at all -- finding #1), so either a
    genuine backend switch went unnoticed by already-compiled call sites, or
    every call -- including a harmless repeat of the same value -- paid the
    cost of a global cache wipe. Post-fix, the cache is cleared exactly once
    per *effective* change and not at all for a redundant call with the same
    value.
    """
    with mock.patch('jax.clear_caches') as clear_caches:
        # First call actually changes 'cpu': None -> 'numba' -> must clear.
        config.set_backend('cpu', 'numba')
        assert clear_caches.call_count == 1

        # Repeating the same value is a no-op -> must NOT clear again.
        config.set_backend('cpu', 'numba')
        assert clear_caches.call_count == 1

        # A genuine change to a different value -> clears again.
        config.set_backend('cpu', 'warp')
        assert clear_caches.call_count == 2

        # Setting backend=None on a platform that has no override is a no-op.
        config.set_backend('gpu', None)
        assert clear_caches.call_count == 2

        # clear_backends() while backends are set actually clears -> clears.
        config.clear_backends()
        assert clear_caches.call_count == 3

        # clear_backends() again on an already-empty map is a no-op.
        config.clear_backends()
        assert clear_caches.call_count == 3
