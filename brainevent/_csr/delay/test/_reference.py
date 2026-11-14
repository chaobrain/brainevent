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

"""Shared inputs and references for the delay-delivery tests."""

import contextlib
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from brainevent import CSRDelayLayout, csr_delay_deliver_layout_p

PLATFORM = jax.default_backend()
BACKENDS = tuple(csr_delay_deliver_layout_p.available_backends(PLATFORM))
# float16/bfloat16 events have no numba type; the numba kernel reads them
# through their mask (binary) or as float32 (weighted).
EVENT_DTYPES = ('bool', 'int8', 'int32', 'float32', 'float16', 'bfloat16')

N_PRE, N_POST, MAX_CONN, L = 40, 30, 9, 7


@contextlib.contextmanager
def x64_disabled() -> Iterator[None]:
    """Run with ``jax_enable_x64`` off, whatever the session default, then restore it."""
    old = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', False)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', old)


def make_csr(n_pre, n_post, max_conn, n_slots, seed=0, indptr_dtype=np.int32, sorted_columns=False):
    """Random CSR with repeated targets inside rows, empty rows and delays in ``[0, L)``."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, max_conn + 1, n_pre)
    if n_pre > 1:
        counts[0] = 0  # an empty row
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(indptr_dtype)
    nse = int(indptr[-1])
    indices = rng.integers(0, n_post, nse).astype(np.int32)
    if sorted_columns:
        for i in range(n_pre):
            indices[indptr[i]:indptr[i + 1]].sort()
    delays = rng.integers(0, n_slots, nse).astype(np.int32)
    weights = rng.normal(size=nse).astype(np.float32)
    return weights, indices, indptr, delays


def make_events(n_pre, dtype, seed=1):
    """Events whose activity is ``x != 0``, with negative values among the active ones."""
    rng = np.random.default_rng(seed)
    if dtype == 'bool':
        return rng.random(n_pre) < 0.5
    vals = rng.choice(np.array([2.0, -1.0, 0.0, 0.5, 0.0, 3.0]), n_pre)
    if jnp.dtype(dtype).kind == 'i':
        vals = np.round(vals * 2)
    return vals.astype(jnp.dtype(dtype))


def random_ring(n_slots=L, n_post=N_POST, dtype=jnp.float32, seed=3):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(n_slots, n_post)), dtype)


def build(seed=0, homo=False, indptr_dtype=np.int32, n_pre=N_PRE, n_post=N_POST, max_conn=MAX_CONN, n_slots=L):
    """Inputs in the original edge order and their layout."""
    w, idx, ipt, dl = make_csr(n_pre, n_post, max_conn, n_slots, seed=seed, indptr_dtype=indptr_dtype)
    data = jnp.asarray(w[:1] if homo else w)
    return data, idx, ipt, dl, CSRDelayLayout.build(data, idx, ipt, dl, n_slots=n_slots, shape=(n_pre, n_post))


def reference(ring, events, data, indices, indptr, delays, t, weighted=False):
    """Dense NumPy loop: ``ring[(t + d_e) % L, indices[e]] += w_e * f(x)[row(e)]`` (float64)."""
    out = np.array(ring, dtype=np.float64)
    n_slots = out.shape[0]
    data = np.asarray(data, dtype=np.float64).reshape(-1)
    delays = np.broadcast_to(np.asarray(delays), np.asarray(indices).shape)
    events = np.asarray(events)
    for i in range(len(indptr) - 1):
        x = events[i]
        if x != 0:
            f = float(x) if weighted else 1.0
            for e in range(int(indptr[i]), int(indptr[i + 1])):
                w = data[0] if data.size == 1 else data[e]
                out[(int(t) + int(delays[e])) % n_slots, indices[e]] += w * f
    return out


def jax_reference(ring, events, data, indices, indptr, delays, t, *, weighted: bool) -> Any:
    """Differentiable pure-JAX delivery on the original edge order.

    Encodes the derivative contract: binary events use the activity mask in the
    forward pass and are straight-through for the event cotangent; weighted
    events are exact.
    """
    indptr = np.asarray(indptr)
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    x = events[rows]
    if weighted:
        f = x.astype(ring.dtype)
    else:
        f = (x != 0).astype(ring.dtype)
        if jnp.issubdtype(x.dtype, jnp.floating):
            f = f + (x - jax.lax.stop_gradient(x)).astype(ring.dtype)
    w = jnp.broadcast_to(data.reshape(-1), (rows.size,)) if data.size == 1 else data
    delays = np.broadcast_to(np.asarray(delays), (rows.size,))
    slots = jnp.remainder(jnp.asarray(t) + jnp.asarray(delays), ring.shape[0])  # t may be traced
    return ring.at[slots, np.asarray(indices)].add(w * f)
