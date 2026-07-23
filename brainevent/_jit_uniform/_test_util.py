import math

import numpy as np

MV_STRIDE = 32
MM_STRIDE = 4


def _u32(x):
    return np.uint32(x)


def mix32(x):
    x = _u32(x)
    x = _u32(x ^ (x >> _u32(16)))
    x = _u32((np.uint64(x) * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF))
    x = _u32(x ^ (x >> _u32(15)))
    x = _u32((np.uint64(x) * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF))
    x = _u32(x ^ (x >> _u32(16)))
    return x


def fast_bounded_u32(r, bound):
    return _u32((np.uint64(_u32(r)) * np.uint64(_u32(bound))) >> np.uint64(32))


def light_rng_next(state):
    x = _u32(state)
    x = _u32(x ^ _u32(x << _u32(13)))
    x = _u32(x ^ (x >> _u32(17)))
    x = _u32(x ^ _u32(x << _u32(5)))
    return _u32(0x6D2B79F5) if x == _u32(0) else x


def light_rng_init(seed, row, chunk_id, lane):
    x = _u32(_u32(seed) ^ _u32(0xD1B54A35))
    x = _u32(x ^ _u32((np.uint64(_u32(row)) * np.uint64(0x85EBCA6B)) & np.uint64(0xFFFFFFFF)))
    x = _u32(x ^ _u32((np.uint64(_u32(chunk_id)) * np.uint64(0xC2B2AE35)) & np.uint64(0xFFFFFFFF)))
    x = _u32(x ^ _u32((np.uint64(_u32(lane)) * np.uint64(0x27D4EB2D)) & np.uint64(0xFFFFFFFF)))
    x = mix32(x)
    return _u32(0x6D2B79F5) if x == _u32(0) else x


def stationary_initial_q(state, cl):
    n = _u32(_u32(cl) - _u32(1))
    while True:
        state = light_rng_next(state)
        q = fast_bounded_u32(state, n)
        state = light_rng_next(state)
        gate = fast_bounded_u32(state, n)
        if gate < _u32(n - q):
            return q, state


def hash_uniform01(seed, row, col):
    h = _u32(_u32(seed) ^ _u32(0xA0761D65))
    h = _u32(h ^ _u32((np.uint64(_u32(row)) * np.uint64(0xE7037ED1)) & np.uint64(0xFFFFFFFF)))
    h = _u32(h ^ _u32((np.uint64(_u32(col)) * np.uint64(0x8EBC6AF1)) & np.uint64(0xFFFFFFFF)))
    h = mix32(h)
    return np.float32((h & _u32(0x00FFFFFF)) * np.float32(1.0 / 16777216.0))


def stride_for_mode(matrix_mode):
    if matrix_mode == 'mv':
        return MV_STRIDE
    if matrix_mode == 'mm':
        return MM_STRIDE
    raise ValueError(f"matrix_mode must be 'mv' or 'mm', got {matrix_mode!r}.")


def conn_length(prob):
    prob = float(prob)
    if prob == 0.0:
        return 0
    return max(2, int(math.ceil(2.0 / prob)))


def default_chunk_size(n_cols, target_chunks=4):
    return max(1, (int(n_cols) + int(target_chunks) - 1) // int(target_chunks))


def iter_edges(seed, clen, n_rows, n_cols, *, corder, stride, chunk_size=None):
    seed0 = _u32(seed)
    cl = _u32(max(2, int(clen)))
    cs = default_chunk_size(n_cols) if chunk_size is None else int(chunk_size)
    n_chunks = 0 if n_cols <= 0 else (int(n_cols) + cs - 1) // cs
    for row in range(int(n_rows)):
        for chunk_id in range(n_chunks):
            chunk_start = chunk_id * cs
            chunk_end = min(chunk_start + cs, int(n_cols))
            chunk_width = chunk_end - chunk_start
            for lane in range(int(stride)):
                state = light_rng_init(seed0, row, chunk_id, lane)
                q, state = stationary_initial_q(state, cl)
                local_j = lane + int(stride) * int(q)
                while local_j < chunk_width:
                    col = chunk_start + local_j
                    if corder:
                        yield row, col, row, col
                    else:
                        yield col, row, row, col
                    state = light_rng_next(state)
                    q = q + _u32(1) + fast_bounded_u32(state, cl - _u32(1))
                    local_j = lane + int(stride) * int(q)


def dense_uniform_reference(w_low, w_high, prob, seed, *, shape, transpose=False, corder=True, matrix_mode='mv'):
    out_shape = tuple(reversed(shape)) if transpose else tuple(shape)
    n_rows, n_cols = out_shape if corder else tuple(reversed(out_shape))
    dtype = np.result_type(np.asarray(w_low), np.asarray(w_high), np.float32)
    out = np.zeros(out_shape, dtype=dtype)
    if float(prob) == 0.0:
        return out
    wlo = np.asarray(w_low, dtype=dtype).item()
    whi = np.asarray(w_high, dtype=dtype).item()
    span = whi - wlo
    stride = stride_for_mode(matrix_mode)
    chunk_size = default_chunk_size(shape[1])
    for out_row, out_col, rng_row, rng_col in iter_edges(
        seed, conn_length(prob), n_rows, n_cols, corder=corder, stride=stride, chunk_size=chunk_size
    ):
        out[out_row, out_col] = wlo + np.asarray(hash_uniform01(seed, rng_row, rng_col), dtype=dtype) * span
    return out
