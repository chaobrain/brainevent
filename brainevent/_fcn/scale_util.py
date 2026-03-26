# -*- coding: utf-8 -*-

"""
Utility for generating (conn, m) test pairs that approach the GPU memory
boundary while staying just below it (truncation / floor arithmetic).
"""

import math
import numpy as np


def generate_cs_pairs(
    memory_limit: float = 17,
    homo_or_not: bool = True,
    scale_max: int = 2000,
    conn_max: int = 4000,
    _N: int = 4000,
    data_size: int = 4,
    num_points: int = 5,
    include_dense_ref: bool = False,
):
    """Generate ``(conn, m)`` pairs near the GPU memory boundary.

    Returns a list of ``(conn, m)`` tuples where ``m = scale * _N``, such
    that the estimated memory usage stays just below *memory_limit* GiB.
    Integer truncation (``int()``) is used so that values always lean
    toward the **smaller** side to avoid overflow.

    Memory model
    -------------
    * **Sparse arrays only** (``include_dense_ref=False``):
      ``m * conn * data_size * times``
      where *times* = 1 (homo, indices only) or 2 (hetero, indices + weights).

    * **With dense reference** (``include_dense_ref=True``):
      ``m * conn * data_size * times  +  m² * data_size``
      Use this when the test constructs a full ``(m, m)`` dense matrix for
      correctness comparison.

    Parameters
    ----------
    memory_limit : float
        GPU memory budget in GiB.
    homo_or_not : bool
        True  → only indices are stored (1× sparse budget).
        False → indices + weights are stored (2× sparse budget).
    scale_max : int
        Upper bound for the scale factor *s* (``m = s * _N``).
    conn_max : int
        Maximum number of connections per row.
    _N : int
        Base neuron count per scale unit.
    data_size : int
        Bytes per element (4 for float32 / int32).
    num_points : int
        Number of sample points to generate.
    include_dense_ref : bool
        If True, the budget also accounts for a dense ``(m, m)`` reference
        matrix of size ``m² * data_size``.
    """
    import math
    import numpy as np
    limit_bytes = memory_limit * (1024 ** 3)
    times = 1 if homo_or_not else 2

    # K = budget expressed in units of (_N * data_size) bytes
    K = limit_bytes / (_N * data_size)

    # --- valid scale range ---------------------------------------------------
    if include_dense_ref:
        # For conn > 0 we need  K - s² * _N > 0  →  s < sqrt(K / _N)
        s_max = min(scale_max, int(math.sqrt(K / _N)))
    else:
        # conn = K / (s * times) >= 1  →  s <= K / times
        s_max = min(scale_max, int(K / times))

    # s_min: when *not* including the dense ref, large s → small conn,
    # which is fine; but very small s → conn > conn_max, wasting budget.
    if not include_dense_ref:
        s_min = max(1, math.ceil(K / (times * conn_max)))
    else:
        s_min = 1

    if s_min > s_max:
        raise ValueError(
            f"No valid (conn, m) pairs: s_min={s_min} > s_max={s_max}. "
            f"Try increasing memory_limit or decreasing scale_max/conn_max."
        )

    s_samples = np.geomspace(s_min, s_max, num_points)

    valid_pairs = []
    seen = set()

    for s_val in s_samples:
        s_int = max(1, int(round(s_val)))
        s_int = min(s_int, s_max)  # clamp

        # Compute max conn (floor to stay below the boundary)
        if include_dense_ref:
            budget_for_sparse = K - s_int ** 2 * _N
            if budget_for_sparse <= 0:
                continue
            c_int = int(budget_for_sparse / (s_int * times))
        else:
            c_int = int(K / (s_int * times))

        c_int = min(c_int, conn_max)

        m = s_int * _N

        if c_int > 0 and c_int <= m and s_int <= scale_max:
            pair = (c_int, m)
            if pair not in seen:
                seen.add(pair)
                valid_pairs.append(pair)

    return valid_pairs
