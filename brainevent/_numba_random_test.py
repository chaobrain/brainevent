# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

import numpy as np
import pytest

from brainevent._numba_random import (
    lfsr88_seed, lfsr88_rand, lfsr88_randint, lfsr88_randn,
    lfsr88_uniform, lfsr88_normal, lfsr88_random_integers,
    lfsr113_seed, lfsr113_rand, lfsr113_randn,
    lfsr113_random_integers,
    lfsr128_seed, lfsr128_rand, lfsr128_randn,
    lfsr128_random_integers,
)


class TestLFSR88Seed:
    """Test LFSR88 seed initialization."""

    def test_state_shape(self):
        state = lfsr88_seed(42)
        assert state.shape == (4,)
        assert state.dtype == np.uint32

    def test_state_values(self):
        state = lfsr88_seed(0)
        assert state[0] == 2  # 0 + 2
        assert state[1] == 8  # 0 + 8
        assert state[2] == 16  # 0 + 16
        assert state[3] == 0

    def test_different_seeds_different_states(self):
        s1 = lfsr88_seed(42)
        s2 = lfsr88_seed(100)
        assert not np.array_equal(s1, s2)


class TestLFSR88Rand:
    """Test LFSR88 uniform random float generation."""

    def test_in_range(self):
        state = lfsr88_seed(42)
        for _ in range(200):
            v = lfsr88_rand(state)
            assert 0.0 <= v < 1.0, f"Value {v} not in [0, 1)"

    def test_deterministic(self):
        s1 = lfsr88_seed(42)
        s2 = lfsr88_seed(42)
        for _ in range(20):
            assert lfsr88_rand(s1) == lfsr88_rand(s2)

    def test_updates_state(self):
        state = lfsr88_seed(42)
        before = state.copy()
        lfsr88_rand(state)
        assert not np.array_equal(state, before)

    def test_statistical_mean_std(self):
        state = lfsr88_seed(42)
        samples = np.array([lfsr88_rand(state) for _ in range(10000)])
        assert 0.45 <= samples.mean() <= 0.55
        assert 0.25 <= samples.std() <= 0.33


class TestLFSR88Randint:
    """Test LFSR88 uint32 random integer generation."""

    def test_returns_uint32(self):
        state = lfsr88_seed(42)
        v = lfsr88_randint(state)
        assert 0 <= int(v) <= 2 ** 32 - 1

    def test_deterministic(self):
        s1 = lfsr88_seed(42)
        s2 = lfsr88_seed(42)
        for _ in range(20):
            assert lfsr88_randint(s1) == lfsr88_randint(s2)


class TestLFSR88Randn:
    """Test LFSR88 standard-normal generation."""

    def test_statistical_properties(self):
        state = lfsr88_seed(42)
        samples = np.array([lfsr88_randn(state) for _ in range(10000)])
        assert -0.1 <= samples.mean() <= 0.1
        assert 0.9 <= samples.std() <= 1.1

    def test_deterministic(self):
        s1 = lfsr88_seed(42)
        s2 = lfsr88_seed(42)
        for _ in range(10):
            assert lfsr88_randn(s1) == lfsr88_randn(s2)


class TestLFSR88Uniform:
    """Test LFSR88 uniform range generation."""

    def test_basic_range(self):
        state = lfsr88_seed(42)
        for _ in range(100):
            v = lfsr88_uniform(state, 5.0, 10.0)
            assert 5.0 <= v < 10.0

    def test_negative_range(self):
        state = lfsr88_seed(42)
        for _ in range(100):
            v = lfsr88_uniform(state, -10.0, -5.0)
            assert -10.0 <= v < -5.0

    def test_zero_width(self):
        state = lfsr88_seed(42)
        v = lfsr88_uniform(state, 5.0, 5.0)
        assert v == 5.0


class TestLFSR88Normal:
    """Test LFSR88 normal distribution generation."""

    def test_standard_normal(self):
        state = lfsr88_seed(42)
        samples = np.array([lfsr88_normal(state, 0.0, 1.0) for _ in range(10000)])
        assert -0.1 <= samples.mean() <= 0.1
        assert 0.9 <= samples.std() <= 1.1

    def test_custom_mean_std(self):
        state = lfsr88_seed(42)
        samples = np.array([lfsr88_normal(state, 100.0, 15.0) for _ in range(10000)])
        assert 98.0 <= samples.mean() <= 102.0
        assert 13.0 <= samples.std() <= 17.0

    def test_zero_std(self):
        state = lfsr88_seed(42)
        v = lfsr88_normal(state, 5.0, 0.0)
        assert v == 5.0


class TestLFSR88RandomIntegers:
    """Test LFSR88 bounded integer generation."""

    def test_basic_range(self):
        state = lfsr88_seed(42)
        for _ in range(200):
            v = lfsr88_random_integers(state, 1, 6)
            assert 1 <= v <= 6

    def test_coin_flip(self):
        state = lfsr88_seed(42)
        for _ in range(100):
            v = lfsr88_random_integers(state, 0, 1)
            assert v in (0, 1)

    def test_same_bounds(self):
        state = lfsr88_seed(42)
        assert lfsr88_random_integers(state, 5, 5) == 5

    def test_large_range(self):
        state = lfsr88_seed(42)
        for _ in range(100):
            v = lfsr88_random_integers(state, 0, 1000)
            assert 0 <= v <= 1000


# ──────────────────────────────────────────────────────────────────────
#  LFSR113 tests
# ──────────────────────────────────────────────────────────────────────

class TestLFSR113Seed:
    def test_state_shape(self):
        state = lfsr113_seed(42)
        assert state.shape == (4,)
        assert state.dtype == np.uint32

    def test_state_values(self):
        state = lfsr113_seed(0)
        assert state[0] == 2
        assert state[1] == 8
        assert state[2] == 16
        assert state[3] == 128

    def test_different_from_lfsr88(self):
        s88 = lfsr88_seed(42)
        s113 = lfsr113_seed(42)
        assert not np.array_equal(s88, s113)


class TestLFSR113Rand:
    def test_in_range(self):
        state = lfsr113_seed(42)
        for _ in range(200):
            v = lfsr113_rand(state)
            assert 0.0 <= v < 1.0

    def test_deterministic(self):
        s1 = lfsr113_seed(42)
        s2 = lfsr113_seed(42)
        for _ in range(20):
            assert lfsr113_rand(s1) == lfsr113_rand(s2)

    def test_statistical_mean_std(self):
        state = lfsr113_seed(42)
        samples = np.array([lfsr113_rand(state) for _ in range(10000)])
        assert 0.45 <= samples.mean() <= 0.55
        assert 0.25 <= samples.std() <= 0.33


class TestLFSR113Randn:
    def test_statistical_properties(self):
        state = lfsr113_seed(42)
        samples = np.array([lfsr113_randn(state) for _ in range(10000)])
        assert -0.1 <= samples.mean() <= 0.1
        assert 0.9 <= samples.std() <= 1.1


class TestLFSR113RandomIntegers:
    def test_basic_range(self):
        state = lfsr113_seed(42)
        for _ in range(200):
            v = lfsr113_random_integers(state, 1, 6)
            assert 1 <= v <= 6


# ──────────────────────────────────────────────────────────────────────
#  LFSR128 tests
# ──────────────────────────────────────────────────────────────────────

class TestLFSR128Seed:
    def test_state_shape(self):
        state = lfsr128_seed(42)
        assert state.shape == (4,)
        assert state.dtype == np.uint32

    def test_different_from_others(self):
        s88 = lfsr88_seed(42)
        s113 = lfsr113_seed(42)
        s128 = lfsr128_seed(42)
        assert not np.array_equal(s88, s128)
        assert not np.array_equal(s113, s128)


class TestLFSR128Rand:
    def test_in_range(self):
        state = lfsr128_seed(42)
        for _ in range(200):
            v = lfsr128_rand(state)
            assert 0.0 <= v < 1.0

    def test_deterministic(self):
        s1 = lfsr128_seed(42)
        s2 = lfsr128_seed(42)
        for _ in range(20):
            assert lfsr128_rand(s1) == lfsr128_rand(s2)

    def test_statistical_mean_std(self):
        state = lfsr128_seed(42)
        samples = np.array([lfsr128_rand(state) for _ in range(10000)])
        assert 0.45 <= samples.mean() <= 0.55
        assert 0.25 <= samples.std() <= 0.33


class TestLFSR128Randn:
    def test_statistical_properties(self):
        state = lfsr128_seed(42)
        samples = np.array([lfsr128_randn(state) for _ in range(10000)])
        assert -0.1 <= samples.mean() <= 0.1
        assert 0.9 <= samples.std() <= 1.1


class TestLFSR128RandomIntegers:
    def test_basic_range(self):
        state = lfsr128_seed(42)
        for _ in range(200):
            v = lfsr128_random_integers(state, 1, 6)
            assert 1 <= v <= 6


# ──────────────────────────────────────────────────────────────────────
#  Cross-implementation comparisons
# ──────────────────────────────────────────────────────────────────────

class TestCrossImplementation:

    def test_all_produce_valid_uniform(self):
        for seed_fn, rand_fn in [(lfsr88_seed, lfsr88_rand),
                                 (lfsr113_seed, lfsr113_rand),
                                 (lfsr128_seed, lfsr128_rand)]:
            state = seed_fn(42)
            for _ in range(100):
                v = rand_fn(state)
                assert 0.0 <= v < 1.0

    def test_all_deterministic(self):
        for seed_fn, rand_fn in [(lfsr88_seed, lfsr88_rand),
                                 (lfsr113_seed, lfsr113_rand),
                                 (lfsr128_seed, lfsr128_rand)]:
            s1 = seed_fn(42)
            s2 = seed_fn(42)
            for _ in range(10):
                assert rand_fn(s1) == rand_fn(s2)

    def test_different_implementations_differ(self):
        s88 = lfsr88_seed(42)
        s113 = lfsr113_seed(42)
        s128 = lfsr128_seed(42)
        v88 = lfsr88_rand(s88)
        v113 = lfsr113_rand(s113)
        v128 = lfsr128_rand(s128)
        assert v88 != v113
        assert v88 != v128
        assert v113 != v128


class TestSequenceProperties:

    def test_no_consecutive_duplicates(self):
        for seed_fn, rand_fn in [(lfsr88_seed, lfsr88_rand),
                                 (lfsr113_seed, lfsr113_rand),
                                 (lfsr128_seed, lfsr128_rand)]:
            state = seed_fn(42)
            samples = [rand_fn(state) for _ in range(1000)]
            dupes = sum(1 for i in range(len(samples) - 1) if samples[i] == samples[i + 1])
            assert dupes < 5

    def test_sequence_diversity(self):
        for seed_fn, rand_fn in [(lfsr88_seed, lfsr88_rand),
                                 (lfsr113_seed, lfsr113_rand),
                                 (lfsr128_seed, lfsr128_rand)]:
            state = seed_fn(42)
            samples = [rand_fn(state) for _ in range(1000)]
            assert len(set(samples)) > 900


class TestEdgeCases:

    def test_large_seed(self):
        state = lfsr88_seed(2 ** 31 - 1)
        v = lfsr88_rand(state)
        assert 0.0 <= v < 1.0

    def test_zero_seed(self):
        state = lfsr88_seed(0)
        v = lfsr88_rand(state)
        assert 0.0 <= v < 1.0

    def test_random_integers_zero_to_zero(self):
        state = lfsr88_seed(42)
        assert lfsr88_random_integers(state, 0, 0) == 0

    def test_uniform_negative_low(self):
        state = lfsr88_seed(42)
        v = lfsr88_uniform(state, -5.0, 5.0)
        assert -5.0 <= v < 5.0

    def test_normal_negative_std(self):
        state = lfsr88_seed(42)
        v = lfsr88_normal(state, 0.0, -1.0)
        assert isinstance(v, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
