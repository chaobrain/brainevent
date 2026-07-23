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

Numba is **not** imported when this module is imported. The primitives are
defined as plain-Python functions and are JIT-compiled lazily: the first call
to any ``get_numba_lfsr_*`` helper imports Numba and swaps each primitive for
its ``@numba.njit(inline='always')`` dispatcher (see ``_ensure_numba_compiled``).
This keeps ``import brainevent`` free of Numba's import cost.
"""

import math
from typing import Callable, Dict

import numpy as np

from .config import get_lfsr_algorithm

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
    # light-RNG (mirrors the CUDA ``light_rng`` connectivity sampler)
    'light_rng_mix32',
    'light_rng_bounded',
    'light_rng_next',
    'light_rng_init',
    'light_rng_uniform01',
    'light_rng_normal01',
    'light_rng_initial_q',
    # Dispatch helpers (for kernel generators)
    'get_numba_lfsr_seed',
    'get_numba_lfsr_random_integers',
    'get_numba_lfsr_uniform',
    'get_numba_lfsr_normal',
    'get_numba_lfsr_funcs',
    'get_numba_light_rng_funcs',
]


# ──────────────────────────────────────────────────────────────────────
#  LFSR88
# ──────────────────────────────────────────────────────────────────────

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


def lfsr88_randint(state):
    """Generate a random ``uint32`` value and advance the LFSR88 state.

    Returns
    -------
    val : np.uint32
    """
    lfsr88_next_key(state)
    return state[0] ^ state[1] ^ state[2]


def lfsr88_rand(state):
    """Generate a uniform random float in [0, 1) and advance the LFSR88 state.

    Returns
    -------
    val : float64
    """
    lfsr88_next_key(state)
    return np.float64(state[0] ^ state[1] ^ state[2]) * 2.3283064365386963e-10


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


def lfsr88_uniform(state, low, high):
    """Generate a uniform random float in [low, high) and advance the LFSR88 state."""
    return lfsr88_rand(state) * (high - low) + low


def lfsr88_normal(state, mu, sigma, epsilon=1e-10):
    """Generate a normal random value N(mu, sigma) and advance the LFSR88 state."""
    return mu + sigma * lfsr88_randn(state, epsilon)


def lfsr88_random_integers(state, low, high):
    """Generate a random integer in [low, high] (inclusive) and advance the LFSR88 state."""
    val = lfsr88_randint(state)
    return np.int64(val % np.uint32(high + 1 - low)) + low


# ──────────────────────────────────────────────────────────────────────
#  LFSR113
# ──────────────────────────────────────────────────────────────────────

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


def lfsr113_randint(state):
    """Generate a random ``uint32`` value and advance the LFSR113 state."""
    lfsr113_next_key(state)
    return state[0] ^ state[1] ^ state[2] ^ state[3]


def lfsr113_rand(state):
    """Generate a uniform random float in [0, 1) and advance the LFSR113 state."""
    lfsr113_next_key(state)
    return np.float64(state[0] ^ state[1] ^ state[2] ^ state[3]) * 2.3283064365386963e-10


def lfsr113_randn(state, epsilon=1e-10):
    """Generate a standard-normal random value (Box-Muller) and advance the LFSR113 state."""
    u1 = lfsr113_rand(state)
    u2 = lfsr113_rand(state)
    if u1 < epsilon:
        u1 = epsilon
    mag = math.sqrt(-2.0 * math.log(u1))
    z = mag * math.sin(2.0 * math.pi * u2)
    return z


def lfsr113_uniform(state, low, high):
    """Generate a uniform random float in [low, high) and advance the LFSR113 state."""
    return lfsr113_rand(state) * (high - low) + low


def lfsr113_normal(state, mu, sigma, epsilon=1e-10):
    """Generate a normal random value N(mu, sigma) and advance the LFSR113 state."""
    return mu + sigma * lfsr113_randn(state, epsilon)


def lfsr113_random_integers(state, low, high):
    """Generate a random integer in [low, high] (inclusive) and advance the LFSR113 state."""
    val = lfsr113_randint(state)
    return np.int64(val % np.uint32(high + 1 - low)) + low


# ──────────────────────────────────────────────────────────────────────
#  LFSR128
# ──────────────────────────────────────────────────────────────────────

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


def lfsr128_randint(state):
    """Generate a random ``uint32`` value and advance the LFSR128 state."""
    lfsr128_next_key(state)
    return state[0] ^ state[1] ^ state[2] ^ state[3]


def lfsr128_rand(state):
    """Generate a uniform random float in [0, 1) and advance the LFSR128 state."""
    lfsr128_next_key(state)
    return np.float64(state[0] ^ state[1] ^ state[2] ^ state[3]) * 2.3283064365386963e-10


def lfsr128_randn(state, epsilon=1e-10):
    """Generate a standard-normal random value (Box-Muller) and advance the LFSR128 state."""
    u1 = lfsr128_rand(state)
    u2 = lfsr128_rand(state)
    if u1 < epsilon:
        u1 = epsilon
    mag = math.sqrt(-2.0 * math.log(u1))
    z = mag * math.sin(2.0 * math.pi * u2)
    return z


def lfsr128_uniform(state, low, high):
    """Generate a uniform random float in [low, high) and advance the LFSR128 state."""
    return lfsr128_rand(state) * (high - low) + low


def lfsr128_normal(state, mu, sigma, epsilon=1e-10):
    """Generate a normal random value N(mu, sigma) and advance the LFSR128 state."""
    return mu + sigma * lfsr128_randn(state, epsilon)


def lfsr128_random_integers(state, low, high):
    """Generate a random integer in [low, high] (inclusive) and advance the LFSR128 state."""
    val = lfsr128_randint(state)
    return np.int64(val % np.uint32(high + 1 - low)) + low


# ──────────────────────────────────────────────────────────────────────
#  light-RNG (bit-exact port of the CUDA connectivity sampler)
# ──────────────────────────────────────────────────────────────────────
#
# These primitives mirror, one-to-one, the ``light_rng`` helpers used by the
# CUDA JIT-connectivity kernels (``binary_jitsmv.cu`` / ``binary_jitsmm.cu``):
# ``mix32``, ``light_rng_init_wpr``, ``light_rng_next`` (xorshift32),
# ``fast_bounded_u32`` (``__umulhi``) and ``stationary_initial_q``. The state is
# a single ``uint32`` value threaded by return value (rather than an array),
# which matches the pointer-mutation semantics of the CUDA code: every
# ``light_rng_next`` advances the stream and returns the new state.
#
# All arithmetic is done in ``uint32`` with C-style wraparound; the 32-bit
# multiplies are widened to ``uint64`` and masked so the result is independent
# of NumPy/Numba integer-promotion rules.

def light_rng_mix32(x):
    """Finalising bit-mixer (the CUDA ``mix32``)."""
    x = np.uint32(x)
    x = np.uint32(x ^ (x >> np.uint32(16)))
    x = np.uint32((np.uint64(x) * np.uint64(0x7feb352d)) & np.uint64(0xFFFFFFFF))
    x = np.uint32(x ^ (x >> np.uint32(15)))
    x = np.uint32((np.uint64(x) * np.uint64(0x846ca68b)) & np.uint64(0xFFFFFFFF))
    x = np.uint32(x ^ (x >> np.uint32(16)))
    return x


def light_rng_bounded(r, bound):
    """Unbiased ``r`` reduced to ``[0, bound)`` (the CUDA ``__umulhi``-based ``fast_bounded_u32``)."""
    return np.uint32((np.uint64(r) * np.uint64(bound)) >> np.uint64(32))


def light_rng_next(state):
    """Advance the xorshift32 stream and return the new ``uint32`` state."""
    x = np.uint32(state)
    x = np.uint32(x ^ np.uint32(x << np.uint32(13)))
    x = np.uint32(x ^ (x >> np.uint32(17)))
    x = np.uint32(x ^ np.uint32(x << np.uint32(5)))
    if x == np.uint32(0):
        x = np.uint32(0x6d2b79f5)
    return x


def light_rng_init(seed, row, chunk_id, lane):
    """Seed a per-``(row, chunk_id, lane)`` stream (the CUDA ``light_rng_init_wpr``)."""
    x = np.uint32(np.uint32(seed) ^ np.uint32(0xd1b54a35))
    x = np.uint32(x ^ np.uint32((np.uint64(np.uint32(row)) * np.uint64(0x85ebca6b)) & np.uint64(0xFFFFFFFF)))
    x = np.uint32(x ^ np.uint32((np.uint64(np.uint32(chunk_id)) * np.uint64(0xc2b2ae35)) & np.uint64(0xFFFFFFFF)))
    x = np.uint32(x ^ np.uint32((np.uint64(np.uint32(lane)) * np.uint64(0x27d4eb2d)) & np.uint64(0xFFFFFFFF)))
    x = light_rng_mix32(x)
    if x == np.uint32(0):
        x = np.uint32(0x6d2b79f5)
    return x


def light_rng_uniform01(seed, row, col):
    """CUDA-compatible 24-bit uniform variate for a generated edge."""
    h = np.uint32(np.uint32(seed) ^ np.uint32(0xa0761d65))
    h = np.uint32(h ^ np.uint32((np.uint64(np.uint32(row)) * np.uint64(0xe7037ed1)) & np.uint64(0xFFFFFFFF)))
    h = np.uint32(h ^ np.uint32((np.uint64(np.uint32(col)) * np.uint64(0x8ebc6af1)) & np.uint64(0xFFFFFFFF)))
    h = light_rng_mix32(h)
    return np.float32((h & np.uint32(0x00FFFFFF)) * np.float32(1.0 / 16777216.0))


def light_rng_normal01(seed, row, col):
    """CUDA-compatible standard-normal variate for a generated edge."""
    u = np.float32(light_rng_uniform01(seed, row, col))
    lo = np.float32(1e-10)
    hi = np.float32(1.0 - 1e-10)
    if u < lo:
        u = lo
    elif u > hi:
        u = hi

    a1 = np.float32(-39.696830)
    a2 = np.float32(220.94609)
    a3 = np.float32(-275.92851)
    a4 = np.float32(138.35775)
    a5 = np.float32(-30.664799)
    a6 = np.float32(2.5066283)
    b1 = np.float32(-54.476099)
    b2 = np.float32(161.58584)
    b3 = np.float32(-155.69898)
    b4 = np.float32(66.801312)
    b5 = np.float32(-13.280681)

    c1 = np.float32(-0.007784894)
    c2 = np.float32(-0.32239646)
    c3 = np.float32(-2.4007583)
    c4 = np.float32(-2.5497325)
    c5 = np.float32(4.3746641)
    c6 = np.float32(2.9381640)
    d1 = np.float32(0.007784696)
    d2 = np.float32(0.32246713)
    d3 = np.float32(2.4451342)
    d4 = np.float32(3.7544087)

    if u < np.float32(0.02425):
        v = np.float32(np.sqrt(np.float32(-2.0) * np.log(u)))
        z = np.float32(
            (((((c1 * v + c2) * v + c3) * v + c4) * v + c5) * v + c6) /
            ((((d1 * v + d2) * v + d3) * v + d4) * v + np.float32(1.0))
        )
        z = np.float32(-z)
    elif u > np.float32(0.97575):
        v = np.float32(np.sqrt(np.float32(-2.0) * np.log(np.float32(1.0) - u)))
        z = np.float32(
            (((((c1 * v + c2) * v + c3) * v + c4) * v + c5) * v + c6) /
            ((((d1 * v + d2) * v + d3) * v + d4) * v + np.float32(1.0))
        )
    else:
        v = np.float32(u - np.float32(0.5))
        r = np.float32(v * v)
        z = np.float32(
            (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * v /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + np.float32(1.0))
        )
    return np.float32(z)


def light_rng_initial_q(state, cl):
    """Draw the stationary residual ``q`` and return ``(q, new_state)``.

    Mirrors the CUDA ``stationary_initial_q``: two ``light_rng_next`` draws per
    rejection round, so the stream advances identically to the device code.
    """
    n = np.uint32(np.uint32(cl) - np.uint32(1))
    while True:
        state = light_rng_next(state)
        q = light_rng_bounded(state, n)
        state = light_rng_next(state)
        gate = light_rng_bounded(state, n)
        if gate < np.uint32(n - q):
            return q, state


# ──────────────────────────────────────────────────────────────────────
#  Dispatch tables and helpers
# ──────────────────────────────────────────────────────────────────────

# Names of every LFSR primitive defined above, in definition order. Used by
# ``_ensure_numba_compiled`` to swap each plain-Python function for its
# ``numba.njit`` dispatcher exactly once, on first use.
_LFSR_FUNC_NAMES = (
    'lfsr88_seed', 'lfsr88_next_key', 'lfsr88_randint', 'lfsr88_rand',
    'lfsr88_randn', 'lfsr88_uniform', 'lfsr88_normal', 'lfsr88_random_integers',
    'lfsr113_seed', 'lfsr113_next_key', 'lfsr113_randint', 'lfsr113_rand',
    'lfsr113_randn', 'lfsr113_uniform', 'lfsr113_normal', 'lfsr113_random_integers',
    'lfsr128_seed', 'lfsr128_next_key', 'lfsr128_randint', 'lfsr128_rand',
    'lfsr128_randn', 'lfsr128_uniform', 'lfsr128_normal', 'lfsr128_random_integers',
)

# The light-RNG primitives form a single (non-selectable) family; they are
# njit-compiled by ``_ensure_numba_compiled`` alongside the LFSR functions and
# rebound as module globals so their mutual references resolve at compile time.
_LIGHT_RNG_FUNC_NAMES = (
    'light_rng_mix32', 'light_rng_bounded', 'light_rng_next',
    'light_rng_init', 'light_rng_uniform01', 'light_rng_normal01', 'light_rng_initial_q',
)

# Dispatch tables mapping an algorithm name to its primitive. They are populated
# lazily by ``_ensure_numba_compiled`` so that importing this module never
# triggers ``import numba``; they stay empty until the first ``get_numba_lfsr_*``
# call.
_NUMBA_LFSR_SEED: Dict[str, Callable] = {}
_NUMBA_LFSR_RANDOM_INTEGERS: Dict[str, Callable] = {}
_NUMBA_LFSR_RAND: Dict[str, Callable] = {}
_NUMBA_LFSR_RANDINT: Dict[str, Callable] = {}
_NUMBA_LFSR_RANDN: Dict[str, Callable] = {}
_NUMBA_LFSR_UNIFORM: Dict[str, Callable] = {}
_NUMBA_LFSR_NORMAL: Dict[str, Callable] = {}

_NUMBA_COMPILED = False


def _ensure_numba_compiled():
    """Numba-compile the LFSR primitives on first use and build the dispatch tables.

    Importing this module leaves every LFSR primitive as a plain-Python callable
    so that ``import brainevent`` does not pull in (and pay the import cost of)
    Numba. The first time a kernel generator requests a primitive through one of
    the ``get_numba_lfsr_*`` helpers, this function imports Numba, replaces each
    module-level primitive with its ``@numba.njit(inline='always')`` dispatcher,
    and fills in the dispatch tables. It is idempotent.

    The module globals are rebound in place -- rather than only storing compiled
    copies in the dispatch tables -- because the primitives call one another by
    global name (for example ``lfsr88_rand`` calls ``lfsr88_next_key``). Numba
    resolves those references against the live module namespace when the
    enclosing kernel is compiled, so every primitive must already be a dispatcher
    by then.

    Notes
    -----
    If Numba is not installed the primitives are left as plain-Python functions
    and the dispatch tables are built from those, mirroring the behaviour of the
    previous import-time Numba stub.
    """
    global _NUMBA_COMPILED
    if _NUMBA_COMPILED:
        return

    g = globals()
    try:
        import numba
    except ImportError:
        numba = None

    if numba is not None:
        for name in _LFSR_FUNC_NAMES:
            g[name] = numba.njit(inline='always')(g[name])
        for name in _LIGHT_RNG_FUNC_NAMES:
            g[name] = numba.njit(inline='always')(g[name])

    _NUMBA_LFSR_SEED.update(
        lfsr88=g['lfsr88_seed'], lfsr113=g['lfsr113_seed'], lfsr128=g['lfsr128_seed'],
    )
    _NUMBA_LFSR_RANDOM_INTEGERS.update(
        lfsr88=g['lfsr88_random_integers'],
        lfsr113=g['lfsr113_random_integers'],
        lfsr128=g['lfsr128_random_integers'],
    )
    _NUMBA_LFSR_RAND.update(
        lfsr88=g['lfsr88_rand'], lfsr113=g['lfsr113_rand'], lfsr128=g['lfsr128_rand'],
    )
    _NUMBA_LFSR_RANDINT.update(
        lfsr88=g['lfsr88_randint'], lfsr113=g['lfsr113_randint'], lfsr128=g['lfsr128_randint'],
    )
    _NUMBA_LFSR_RANDN.update(
        lfsr88=g['lfsr88_randn'], lfsr113=g['lfsr113_randn'], lfsr128=g['lfsr128_randn'],
    )
    _NUMBA_LFSR_UNIFORM.update(
        lfsr88=g['lfsr88_uniform'], lfsr113=g['lfsr113_uniform'], lfsr128=g['lfsr128_uniform'],
    )
    _NUMBA_LFSR_NORMAL.update(
        lfsr88=g['lfsr88_normal'], lfsr113=g['lfsr113_normal'], lfsr128=g['lfsr128_normal'],
    )

    _NUMBA_COMPILED = True


def get_numba_lfsr_seed():
    """Return the Numba LFSR seed function for the current global algorithm."""
    _ensure_numba_compiled()
    return _NUMBA_LFSR_SEED[get_lfsr_algorithm()]


def get_numba_lfsr_random_integers():
    """Return the Numba LFSR random_integers function for the current global algorithm."""
    _ensure_numba_compiled()
    return _NUMBA_LFSR_RANDOM_INTEGERS[get_lfsr_algorithm()]


def get_numba_lfsr_uniform():
    """Return the Numba LFSR uniform function for the current global algorithm."""
    _ensure_numba_compiled()
    return _NUMBA_LFSR_UNIFORM[get_lfsr_algorithm()]


def get_numba_lfsr_normal():
    """Return the Numba LFSR normal function for the current global algorithm."""
    _ensure_numba_compiled()
    return _NUMBA_LFSR_NORMAL[get_lfsr_algorithm()]


def get_numba_light_rng_funcs():
    """Return the njit light-RNG primitives (bit-exact port of the CUDA sampler).

    Returns
    -------
    dict
        Keys: ``'init'``, ``'next'``, ``'bounded'``, ``'uniform01'``,
        ``'normal01'``, ``'initial_q'``, ``'mix32'`` -- the
        ``@numba.njit(inline='always')`` dispatchers used to
        regenerate the same JIT connectivity matrix that the CUDA kernels draw.
    """
    _ensure_numba_compiled()
    g = globals()
    return {
        'init': g['light_rng_init'],
        'next': g['light_rng_next'],
        'bounded': g['light_rng_bounded'],
        'uniform01': g['light_rng_uniform01'],
        'normal01': g['light_rng_normal01'],
        'initial_q': g['light_rng_initial_q'],
        'mix32': g['light_rng_mix32'],
    }


def get_numba_lfsr_funcs():
    """Return a dict of all Numba LFSR functions for the current global algorithm.

    Returns
    -------
    dict
        Keys: ``'seed'``, ``'rand'``, ``'randint'``, ``'randn'``,
        ``'uniform'``, ``'normal'``, ``'random_integers'``.
    """
    _ensure_numba_compiled()
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
