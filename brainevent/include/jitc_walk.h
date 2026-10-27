// Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

/*
 * jitc_walk.h -- thinning test shared by the JIT connectivity kernels
 * ===================================================================
 *
 * A JITC walk visits a fraction 2 / cl of the positions of each row (skips
 * are Uniform{1, ..., cl - 1}).  ``clen = (cl, thr)`` is chosen on the host
 * (``brainevent._data._initialize_conn_length``) so that keeping a visited
 * position ``(row, j)`` with probability ``thr / 2^24`` gives density ``p``
 * exactly.  ``thr == 2^24`` (every integer ``2 / p``) keeps every candidate
 * without hashing.
 *
 * The keep hash is keyed by the same ``(row, j)`` pair as the edge-weight
 * hash, but with its own salt and multipliers.  It must stay bit-identical to
 * ``light_rng_keep`` in ``brainevent/_numba_random.py``.
 *
 * Usage:
 *   #include "jitc_walk.h"
 *   unsigned int thr = (unsigned int)__ldg(&clen[1]);
 *   ...
 *   if (jitc_keep(thr, seed0, row, j)) { ... }
 */

#ifndef BRAINEVENT_JITC_WALK_H_
#define BRAINEVENT_JITC_WALK_H_

#define JITC_KEEP_ONE 16777216U

__device__ __forceinline__ unsigned int jitc_keep_mix32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ bool jitc_keep(
    unsigned int thr,
    unsigned int seed,
    int row,
    int col
) {
    if (thr >= JITC_KEEP_ONE) return true;
    unsigned int h = seed ^ 0x2545f491U;
    h ^= (unsigned int)row * 0x9e3779b1U;
    h ^= (unsigned int)col * 0x632be5abU;
    return (jitc_keep_mix32(h) >> 8) < thr;
}

#endif  // BRAINEVENT_JITC_WALK_H_
