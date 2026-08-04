import numpy as np

from brainevent._jit_uniform._test_util import (
    conn_length,
    default_chunk_size,
    hash_uniform01,
    iter_edges,
    stride_for_mode,
)


def hash_normal01(seed, row, col):
    u = np.float32(hash_uniform01(seed, row, col))
    u = np.maximum(np.minimum(u, np.float32(1.0 - 1e-10)), np.float32(1e-10))

    a1, a2, a3 = np.float32(-39.696830), np.float32(220.94609), np.float32(-275.92851)
    a4, a5, a6 = np.float32(138.35775), np.float32(-30.664799), np.float32(2.5066283)
    b1, b2, b3 = np.float32(-54.476099), np.float32(161.58584), np.float32(-155.69898)
    b4, b5 = np.float32(66.801312), np.float32(-13.280681)

    c1, c2, c3 = np.float32(-0.007784894), np.float32(-0.32239646), np.float32(-2.4007583)
    c4, c5, c6 = np.float32(-2.5497325), np.float32(4.3746641), np.float32(2.9381640)
    d1, d2, d3, d4 = np.float32(0.007784696), np.float32(0.32246713), np.float32(2.4451342), np.float32(3.7544087)

    if u < np.float32(0.02425):
        v = np.float32(np.sqrt(np.float32(-2.0) * np.log(u)))
        z = np.float32(
            (((((c1 * v + c2) * v + c3) * v + c4) * v + c5) * v + c6) /
            ((((d1 * v + d2) * v + d3) * v + d4) * v + np.float32(1.0))
        )
        return np.float32(-z)
    if u > np.float32(0.97575):
        v = np.float32(np.sqrt(np.float32(-2.0) * np.log(np.float32(1.0) - u)))
        return np.float32(
            (((((c1 * v + c2) * v + c3) * v + c4) * v + c5) * v + c6) /
            ((((d1 * v + d2) * v + d3) * v + d4) * v + np.float32(1.0))
        )

    v = np.float32(u - np.float32(0.5))
    r = np.float32(v * v)
    return np.float32(
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * v /
        (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + np.float32(1.0))
    )


def dense_normal_reference(w_loc, w_scale, prob, seed, *, shape, transpose=False, corder=True, matrix_mode='mv'):
    out_shape = tuple(reversed(shape)) if transpose else tuple(shape)
    n_rows, n_cols = out_shape if corder else tuple(reversed(out_shape))
    dtype = np.result_type(np.asarray(w_loc), np.asarray(w_scale), np.float32)
    out = np.zeros(out_shape, dtype=dtype)
    if float(prob) == 0.0:
        return out
    loc = np.asarray(w_loc, dtype=dtype).item()
    scale = np.asarray(w_scale, dtype=dtype).item()
    stride = stride_for_mode(matrix_mode)
    chunk_size = default_chunk_size(shape[1])
    for out_row, out_col, rng_row, rng_col in iter_edges(
        seed, conn_length(prob), n_rows, n_cols, corder=corder, stride=stride, chunk_size=chunk_size
    ):
        out[out_row, out_col] = loc + np.asarray(hash_normal01(seed, rng_row, rng_col), dtype=dtype) * scale
    return out
