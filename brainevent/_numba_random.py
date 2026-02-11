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

"""
Numba-compatible LFSR random number generators.

This module provides standalone ``@numba.njit``-compatible LFSR functions
that mirror the Pallas LFSR implementations in ``_pallas_random.py``.
State is represented as ``np.array([s1, s2, s3, s4], dtype=np.uint32)``
and mutated in-place for efficiency.

Three algorithm families are provided:

* **LFSR88** — Combined LFSR with period ~2^88 (3-component).
* **LFSR113** — Combined LFSR with period ~2^113 (4-component).
* **LFSR128** — Combined LFSR with period ~2^128 (4-component).
"""

import importlib.util
import math

import numpy as np

from .config import get_lfsr_algorithm

if importlib.util.find_spec('numba') is not None:
    import numba
else:
    class _NumbaStub:
        """Minimal stub so decorated functions remain plain Python callables."""

        @staticmethod
        def njit(func=None, **kwargs):
            if func is not None:
                return func
            return lambda f: f


    numba = _NumbaStub()

__all__ = [
    # LFSR88
    'lfsr88_seed',
    'lfsr88_next_key',
    'lfsr88_rand',
    'lfsr88_randint',
    'lfsr88_randn',
    'lfsr88_uniform',
    'lfsr88_normal',
    'lfsr88_random_integers',
    # LFSR113
    'lfsr113_seed',
    'lfsr113_next_key',
    'lfsr113_rand',
    'lfsr113_randint',
    'lfsr113_randn',
    'lfsr113_uniform',
    'lfsr113_normal',
    'lfsr113_random_integers',
    # LFSR128
    'lfsr128_seed',
    'lfsr128_next_key',
    'lfsr128_rand',
    'lfsr128_randint',
    'lfsr128_randn',
    'lfsr128_uniform',
    'lfsr128_normal',
    'lfsr128_random_integers',
    # Dispatch helpers (for kernel generators)
    'get_numba_lfsr_seed',
    'get_numba_lfsr_random_integers',
    'get_numba_lfsr_uniform',
    'get_numba_lfsr_normal',
    'get_numba_lfsr_funcs',
    # User-level dispatch functions
    'lfsr_seed',
    'lfsr_rand',
    'lfsr_randint',
    'lfsr_randn',
    'lfsr_uniform',
    'lfsr_normal',
    'lfsr_random_integers',
]


# ──────────────────────────────────────────────────────────────────────
#  LFSR88
# ──────────────────────────────────────────────────────────────────────

@numba.njit(inline='always')
def lfsr88_seed(seed):
    """Create an LFSR88 state array from an integer seed.

    Parameters
    ----------
    seed : int
        Integer seed value.

    Returns
    -------
    state : np.ndarray
        A ``(4,)`` ``uint32`` array ``[s1, s2, s3, 0]``.
    """
    state = np.empty(4, dtype=np.uint32)
    state[0] = np.uint32(seed + 2)
    state[1] = np.uint32(seed + 8)
    state[2] = np.uint32(seed + 16)
    state[3] = np.uint32(0)
    return state


@numba.njit(inline='always')
def lfsr88_next_key(state):
    """Advance the LFSR88 state in-place by one step."""
    s1 = state[0]
    s2 = state[1]
    s3 = state[2]

    b = ((s1 << np.uint32(13)) ^ s1) >> np.uint32(19)
    s1 = ((s1 & np.uint32(0xFFFFFFFE)) << np.uint32(12)) ^ b

    b = ((s2 << np.uint32(2)) ^ s2) >> np.uint32(25)
    s2 = ((s2 & np.uint32(0xFFFFFFF8)) << np.uint32(4)) ^ b

    b = ((s3 << np.uint32(3)) ^ s3) >> np.uint32(11)
    s3 = ((s3 & np.uint32(0xFFFFFFF0)) << np.uint32(17)) ^ b

    state[0] = s1
    state[1] = s2
    state[2] = s3
    state[3] = b


@numba.njit(inline='always')
def lfsr88_randint(state):
    """Generate a random ``uint32`` value and advance the LFSR88 state.

    Returns
    -------
    val : np.uint32
    """
    lfsr88_next_key(state)
    return state[0] ^ state[1] ^ state[2]


@numba.njit(inline='always')
def lfsr88_rand(state):
    """Generate a uniform random float in [0, 1) and advance the LFSR88 state.

    Returns
    -------
    val : float64
    """
    lfsr88_next_key(state)
    return np.float64(state[0] ^ state[1] ^ state[2]) * 2.3283064365386963e-10


@numba.njit(inline='always')
def lfsr88_randn(state, epsilon=1e-10):
    """Generate a standard-normal random value (Box-Muller) and advance the LFSR88 state.

    Returns
    -------
    val : float64
    """
    u1 = lfsr88_rand(state)
    u2 = lfsr88_rand(state)
    if u1 < epsilon:
        u1 = epsilon
    mag = math.sqrt(-2.0 * math.log(u1))
    z = mag * math.sin(2.0 * math.pi * u2)
    return z


@numba.njit(inline='always')
def lfsr88_uniform(state, low, high):
    """Generate a uniform random float in [low, high) and advance the LFSR88 state."""
    return lfsr88_rand(state) * (high - low) + low


@numba.njit(inline='always')
def lfsr88_normal(state, mu, sigma, epsilon=1e-10):
    """Generate a normal random value N(mu, sigma) and advance the LFSR88 state."""
    return mu + sigma * lfsr88_randn(state, epsilon)


@numba.njit(inline='always')
def lfsr88_random_integers(state, low, high):
    """Generate a random integer in [low, high] (inclusive) and advance the LFSR88 state."""
    val = lfsr88_randint(state)
    return np.int64(val % np.uint32(high + 1 - low)) + low


# ──────────────────────────────────────────────────────────────────────
#  LFSR113
# ──────────────────────────────────────────────────────────────────────

@numba.njit(inline='always')
def lfsr113_seed(seed):
    """Create an LFSR113 state array from an integer seed.

    Parameters
    ----------
    seed : int
        Integer seed value.

    Returns
    -------
    state : np.ndarray
        A ``(4,)`` ``uint32`` array ``[s1, s2, s3, s4]``.
    """
    state = np.empty(4, dtype=np.uint32)
    state[0] = np.uint32(seed + 2)
    state[1] = np.uint32(seed + 8)
    state[2] = np.uint32(seed + 16)
    state[3] = np.uint32(seed + 128)
    return state


@numba.njit(inline='always')
def lfsr113_next_key(state):
    """Advance the LFSR113 state in-place by one step."""
    z1 = state[0]
    z2 = state[1]
    z3 = state[2]
    z4 = state[3]

    b1 = ((z1 << np.uint32(6)) ^ z1) >> np.uint32(13)
    z1 = ((z1 & np.uint32(0xFFFFFFFE)) << np.uint32(18)) ^ b1

    b2 = ((z2 << np.uint32(2)) ^ z2) >> np.uint32(27)
    z2 = ((z2 & np.uint32(0xFFFFFFF8)) << np.uint32(2)) ^ b2

    b3 = ((z3 << np.uint32(13)) ^ z3) >> np.uint32(21)
    z3 = ((z3 & np.uint32(0xFFFFFFF0)) << np.uint32(7)) ^ b3

    b4 = ((z4 << np.uint32(3)) ^ z4) >> np.uint32(12)
    z4 = ((z4 & np.uint32(0xFFFFFF80)) << np.uint32(13)) ^ b4

    state[0] = z1
    state[1] = z2
    state[2] = z3
    state[3] = z4


@numba.njit(inline='always')
def lfsr113_randint(state):
    """Generate a random ``uint32`` value and advance the LFSR113 state."""
    lfsr113_next_key(state)
    return state[0] ^ state[1] ^ state[2] ^ state[3]


@numba.njit(inline='always')
def lfsr113_rand(state):
    """Generate a uniform random float in [0, 1) and advance the LFSR113 state."""
    lfsr113_next_key(state)
    return np.float64(state[0] ^ state[1] ^ state[2] ^ state[3]) * 2.3283064365386963e-10


@numba.njit(inline='always')
def lfsr113_randn(state, epsilon=1e-10):
    """Generate a standard-normal random value (Box-Muller) and advance the LFSR113 state."""
    u1 = lfsr113_rand(state)
    u2 = lfsr113_rand(state)
    if u1 < epsilon:
        u1 = epsilon
    mag = math.sqrt(-2.0 * math.log(u1))
    z = mag * math.sin(2.0 * math.pi * u2)
    return z


@numba.njit(inline='always')
def lfsr113_uniform(state, low, high):
    """Generate a uniform random float in [low, high) and advance the LFSR113 state."""
    return lfsr113_rand(state) * (high - low) + low


@numba.njit(inline='always')
def lfsr113_normal(state, mu, sigma, epsilon=1e-10):
    """Generate a normal random value N(mu, sigma) and advance the LFSR113 state."""
    return mu + sigma * lfsr113_randn(state, epsilon)


@numba.njit(inline='always')
def lfsr113_random_integers(state, low, high):
    """Generate a random integer in [low, high] (inclusive) and advance the LFSR113 state."""
    val = lfsr113_randint(state)
    return np.int64(val % np.uint32(high + 1 - low)) + low


# ──────────────────────────────────────────────────────────────────────
#  LFSR128
# ──────────────────────────────────────────────────────────────────────

@numba.njit(inline='always')
def lfsr128_seed(seed):
    """Create an LFSR128 state array from an integer seed.

    Parameters
    ----------
    seed : int
        Integer seed value.

    Returns
    -------
    state : np.ndarray
        A ``(4,)`` ``uint32`` array ``[s1, s2, s3, s4]``.
    """
    s = np.uint32(seed)
    state = np.empty(4, dtype=np.uint32)
    state[0] = s + np.uint32(123)
    state[1] = s ^ np.uint32(0xFEDC7890)
    state[2] = (s << np.uint32(3)) + np.uint32(0x1A2B3C4D)
    state[3] = ~(s + np.uint32(0x5F6E7D8C))
    return state


@numba.njit(inline='always')
def lfsr128_next_key(state):
    """Advance the LFSR128 state in-place by one step."""
    z1 = state[0]
    z2 = state[1]
    z3 = state[2]
    z4 = state[3]

    b1 = ((z1 << np.uint32(7)) ^ z1) >> np.uint32(9)
    z1 = ((z1 & np.uint32(0xFFFFFFFE)) << np.uint32(15)) ^ b1

    b2 = ((z2 << np.uint32(5)) ^ z2) >> np.uint32(23)
    z2 = ((z2 & np.uint32(0xFFFFFFF0)) << np.uint32(6)) ^ b2

    b3 = ((z3 << np.uint32(11)) ^ z3) >> np.uint32(17)
    z3 = ((z3 & np.uint32(0xFFFFFF80)) << np.uint32(8)) ^ b3

    b4 = ((z4 << np.uint32(13)) ^ z4) >> np.uint32(7)
    z4 = ((z4 & np.uint32(0xFFFFFFE0)) << np.uint32(10)) ^ b4

    state[0] = z1
    state[1] = z2
    state[2] = z3
    state[3] = z4


@numba.njit(inline='always')
def lfsr128_randint(state):
    """Generate a random ``uint32`` value and advance the LFSR128 state."""
    lfsr128_next_key(state)
    return state[0] ^ state[1] ^ state[2] ^ state[3]


@numba.njit(inline='always')
def lfsr128_rand(state):
    """Generate a uniform random float in [0, 1) and advance the LFSR128 state."""
    lfsr128_next_key(state)
    return np.float64(state[0] ^ state[1] ^ state[2] ^ state[3]) * 2.3283064365386963e-10


@numba.njit(inline='always')
def lfsr128_randn(state, epsilon=1e-10):
    """Generate a standard-normal random value (Box-Muller) and advance the LFSR128 state."""
    u1 = lfsr128_rand(state)
    u2 = lfsr128_rand(state)
    if u1 < epsilon:
        u1 = epsilon
    mag = math.sqrt(-2.0 * math.log(u1))
    z = mag * math.sin(2.0 * math.pi * u2)
    return z


@numba.njit(inline='always')
def lfsr128_uniform(state, low, high):
    """Generate a uniform random float in [low, high) and advance the LFSR128 state."""
    return lfsr128_rand(state) * (high - low) + low


@numba.njit(inline='always')
def lfsr128_normal(state, mu, sigma, epsilon=1e-10):
    """Generate a normal random value N(mu, sigma) and advance the LFSR128 state."""
    return mu + sigma * lfsr128_randn(state, epsilon)


@numba.njit(inline='always')
def lfsr128_random_integers(state, low, high):
    """Generate a random integer in [low, high] (inclusive) and advance the LFSR128 state."""
    val = lfsr128_randint(state)
    return np.int64(val % np.uint32(high + 1 - low)) + low


# ──────────────────────────────────────────────────────────────────────
#  Dispatch tables and helpers
# ──────────────────────────────────────────────────────────────────────

_NUMBA_LFSR_SEED = {
    'lfsr88': lfsr88_seed,
    'lfsr113': lfsr113_seed,
    'lfsr128': lfsr128_seed,
}

_NUMBA_LFSR_RANDOM_INTEGERS = {
    'lfsr88': lfsr88_random_integers,
    'lfsr113': lfsr113_random_integers,
    'lfsr128': lfsr128_random_integers,
}

_NUMBA_LFSR_RAND = {
    'lfsr88': lfsr88_rand,
    'lfsr113': lfsr113_rand,
    'lfsr128': lfsr128_rand,
}

_NUMBA_LFSR_RANDINT = {
    'lfsr88': lfsr88_randint,
    'lfsr113': lfsr113_randint,
    'lfsr128': lfsr128_randint,
}

_NUMBA_LFSR_RANDN = {
    'lfsr88': lfsr88_randn,
    'lfsr113': lfsr113_randn,
    'lfsr128': lfsr128_randn,
}

_NUMBA_LFSR_UNIFORM = {
    'lfsr88': lfsr88_uniform,
    'lfsr113': lfsr113_uniform,
    'lfsr128': lfsr128_uniform,
}

_NUMBA_LFSR_NORMAL = {
    'lfsr88': lfsr88_normal,
    'lfsr113': lfsr113_normal,
    'lfsr128': lfsr128_normal,
}


def get_numba_lfsr_seed():
    """Return the Numba LFSR seed function for the current global algorithm."""
    return _NUMBA_LFSR_SEED[get_lfsr_algorithm()]


def get_numba_lfsr_random_integers():
    """Return the Numba LFSR random_integers function for the current global algorithm."""
    return _NUMBA_LFSR_RANDOM_INTEGERS[get_lfsr_algorithm()]


def get_numba_lfsr_uniform():
    """Return the Numba LFSR uniform function for the current global algorithm."""
    return _NUMBA_LFSR_UNIFORM[get_lfsr_algorithm()]


def get_numba_lfsr_normal():
    """Return the Numba LFSR normal function for the current global algorithm."""
    return _NUMBA_LFSR_NORMAL[get_lfsr_algorithm()]


def get_numba_lfsr_funcs():
    """Return a dict of all Numba LFSR functions for the current global algorithm.

    Returns
    -------
    dict
        Keys: ``'seed'``, ``'rand'``, ``'randint'``, ``'randn'``,
        ``'uniform'``, ``'normal'``, ``'random_integers'``.
    """

    alg = get_lfsr_algorithm()
    return {
        'seed': _NUMBA_LFSR_SEED[alg],
        'rand': _NUMBA_LFSR_RAND[alg],
        'randint': _NUMBA_LFSR_RANDINT[alg],
        'randn': _NUMBA_LFSR_RANDN[alg],
        'uniform': _NUMBA_LFSR_UNIFORM[alg],
        'normal': _NUMBA_LFSR_NORMAL[alg],
        'random_integers': _NUMBA_LFSR_RANDOM_INTEGERS[alg],
    }


# ──────────────────────────────────────────────────────────────────────
#  User-level dispatch functions
# ──────────────────────────────────────────────────────────────────────

def lfsr_seed(seed):
    """Create an LFSR state array using the globally configured algorithm."""
    return get_numba_lfsr_seed()(seed)


def lfsr_rand(state):
    """Generate a uniform random float in [0, 1) using the globally configured algorithm."""
    return _NUMBA_LFSR_RAND[get_lfsr_algorithm()](state)


def lfsr_randint(state):
    """Generate a random uint32 using the globally configured algorithm."""
    return _NUMBA_LFSR_RANDINT[get_lfsr_algorithm()](state)


def lfsr_randn(state):
    """Generate a standard-normal random value using the globally configured algorithm."""
    return _NUMBA_LFSR_RANDN[get_lfsr_algorithm()](state)


def lfsr_uniform(state, low, high):
    """Generate a uniform random float in [low, high) using the globally configured algorithm."""
    return get_numba_lfsr_uniform()(state, low, high)


def lfsr_normal(state, mu, sigma):
    """Generate a normal random value N(mu, sigma) using the globally configured algorithm."""
    return get_numba_lfsr_normal()(state, mu, sigma)


def lfsr_random_integers(state, low, high):
    """Generate a random integer in [low, high] using the globally configured algorithm."""
    return get_numba_lfsr_random_integers()(state, low, high)
